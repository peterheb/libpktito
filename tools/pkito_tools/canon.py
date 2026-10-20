"""A Parakeet TDT checkpoint's tensors under pkito's names, in float64, and
the architecture numbers the engine needs, read off its config.

Four things are done to the numbers here, each exact or closer to exact
than what NeMo computes, and each a property of the weights rather than of
the arithmetic, so that the engine does not have to know about them:

  - the 0.5 that scales each feed-forward module's output (fc_factor) is
    folded into its second matrix: a power of two, so no bit changes;
  - each convolution module's BatchNorm (eval mode: running statistics) is
    folded into the depthwise convolution before it, w * g/sqrt(var + eps)
    and b - mean * g/sqrt(var + eps), in float64;
  - each LSTM layer's two biases are summed, in float64;
  - the relative positional encoding's frequencies are stored, so that the
    engine's angles do not depend on a platform's exp: NeMo's float32
    product, then exp in float64 rounded once (pos.freq). torch's own
    float32 exp is 1 ulp from that in 48 of the 512 (core/encoder.c
    computes the sines).

The names, the shapes and every rule the engine checks a file against are
core/pkf.h's.
"""
import collections
import math

import numpy as np

BN_EPS = 1e-5        # torch.nn.BatchNorm1d's default, which NeMo leaves alone


def arch(cfg):
    """The numbers core/pkf.h's params carry, from a NeMo config; anything
    pkito does not implement is refused here rather than run wrongly."""
    pre, enc, dec, joint = cfg["preprocessor"], cfg["encoder"], cfg["decoder"], cfg["joint"]
    dur = cfg["model_defaults"]["tdt_durations"]
    want = [
        (pre.get("_target_", "").endswith("AudioToMelSpectrogramPreprocessor"), "the preprocessor"),
        (pre.get("normalize") == "per_feature", "per_feature normalization"),
        (pre.get("window") == "hann", "a Hann window"),
        (pre.get("log", True) is True, "log features"),
        (pre.get("frame_splicing", 1) == 1, "no frame splicing"),
        (pre.get("log_zero_guard_type", "add") == "add", "an additive log guard"),
        (pre.get("mag_power", 2.0) == 2.0, "a power spectrum"),
        (pre.get("exact_pad", False) is False, "a centred STFT"),
        (enc.get("_target_", "").endswith("ConformerEncoder"), "a ConformerEncoder"),
        (enc.get("subsampling") == "dw_striding", "dw_striding subsampling"),
        (enc.get("subsampling_factor") == 8, "subsampling by 8"),
        (enc.get("self_attention_model") == "rel_pos", "rel_pos attention"),
        (list(enc.get("att_context_size", [-1, -1])) == [-1, -1], "full attention"),
        (enc.get("xscaling") is False, "no xscaling"),
        (enc.get("untie_biases") is True, "per-layer position biases"),
        (enc.get("use_bias") is False, "bias-free encoder matrices"),
        (enc.get("conv_norm_type") == "batch_norm", "batch norm in the convolution module"),
        (enc.get("causal_downsampling", False) is False, "non-causal subsampling"),
        (enc.get("reduction") is None, "no reduction layer"),
        (dec.get("_target_", "").endswith("RNNTDecoder"), "an RNNTDecoder"),
        (dec.get("blank_as_pad") is True, "blank_as_pad"),
        (dec.get("normalization_mode") is None, "no decoder normalization"),
        (joint["jointnet"].get("activation") == "relu", "a ReLU joint"),
        (cfg.get("decoding", {}).get("model_type") == "tdt", "a TDT model"),
    ]
    for ok, what in want:
        if not ok:
            raise ValueError("pkito implements Parakeet TDT with %s; this checkpoint differs" % what)
    sr = int(pre["sample_rate"])
    win, hop = round(pre["window_size"] * sr), round(pre["window_stride"] * sr)
    return collections.OrderedDict([
        ("sample_rate", sr),
        ("n_fft", int(pre.get("n_fft") or 2 ** math.ceil(math.log2(win)))),
        ("win_length", win),
        ("hop_length", hop),
        ("n_mels", int(pre["features"])),
        ("preemph", float(pre.get("preemph", 0.97))),
        ("log_guard", float(pre.get("log_zero_guard_value", 2 ** -24))),
        ("norm_eps", 1e-5),                 # features.py CONSTANT, added to each std
        ("n_layers", int(enc["n_layers"])),
        ("d_model", int(enc["d_model"])),
        ("n_heads", int(enc["n_heads"])),
        ("ff_dim", int(enc["d_model"] * enc["ff_expansion_factor"])),
        ("conv_kernel", int(enc["conv_kernel_size"])),
        ("sub_channels", int(enc["subsampling_conv_channels"])),
        ("ln_eps", 1e-5),                   # torch.nn.LayerNorm's default
        ("vocab", int(dec["vocab_size"])),
        ("pred_hidden", int(dec["prednet"]["pred_hidden"])),
        ("pred_layers", int(dec["prednet"]["pred_rnn_layers"])),
        ("joint_hidden", int(joint["jointnet"]["joint_hidden"])),
        ("durations", [int(d) for d in dur]),
        ("max_symbols", int(cfg["decoding"]["greedy"]["max_symbols"])),
    ])


def pos_freq(d_model):
    """NeMo's div_term (RelPositionalEncoding.create_pe): exp of a float32
    arange times -(ln 10000 / d), the product in float32 as torch forms it,
    the exp in float64 and rounded to float32."""
    a = np.arange(0, d_model, 2, dtype=np.float32)
    arg = a * np.float32(-(math.log(10000.0) / d_model))
    return np.exp(arg.astype(np.float64)).astype(np.float32)


def tensors(state, a):
    """[(name, float64 array)] in file order, from a NeMo state dict."""
    s = {k: np.asarray(v, np.float64) for k, v in state.items()
         if not k.endswith("num_batches_tracked")}
    used = set()

    def g(name):
        used.add(name)
        return s[name]

    D, H, C = a["d_model"], a["n_heads"], a["sub_channels"]
    out = [("mel.fb", g("preprocessor.featurizer.fb")[0]),
           ("mel.window", g("preprocessor.featurizer.window")),
           ("pos.freq", pos_freq(D).astype(np.float64))]
    p = "encoder.pre_encode."
    out += [("sub.conv0.w", g(p + "conv.0.weight").reshape(C, 9)), ("sub.conv0.b", g(p + "conv.0.bias")),
            ("sub.dw1.w", g(p + "conv.2.weight").reshape(C, 9)), ("sub.dw1.b", g(p + "conv.2.bias")),
            ("sub.pw1.w", g(p + "conv.3.weight").reshape(C, C)), ("sub.pw1.b", g(p + "conv.3.bias")),
            ("sub.dw2.w", g(p + "conv.5.weight").reshape(C, 9)), ("sub.dw2.b", g(p + "conv.5.bias")),
            ("sub.pw2.w", g(p + "conv.6.weight").reshape(C, C)), ("sub.pw2.b", g(p + "conv.6.bias")),
            ("sub.out.w", g(p + "out.weight")), ("sub.out.b", g(p + "out.bias"))]
    for i in range(a["n_layers"]):
        p, q = "encoder.layers.%d." % i, "l%d." % i
        bn = p + "conv.batch_norm."
        scale = g(bn + "weight") / np.sqrt(g(bn + "running_var") + BN_EPS)
        dw = g(p + "conv.depthwise_conv.weight")[:, 0, :]
        out += [
            (q + "ff1.norm.w", g(p + "norm_feed_forward1.weight")),
            (q + "ff1.norm.b", g(p + "norm_feed_forward1.bias")),
            (q + "ff1.up", g(p + "feed_forward1.linear1.weight")),
            (q + "ff1.down", 0.5 * g(p + "feed_forward1.linear2.weight")),
            (q + "att.norm.w", g(p + "norm_self_att.weight")),
            (q + "att.norm.b", g(p + "norm_self_att.bias")),
            (q + "att.q", g(p + "self_attn.linear_q.weight")),
            (q + "att.k", g(p + "self_attn.linear_k.weight")),
            (q + "att.v", g(p + "self_attn.linear_v.weight")),
            (q + "att.pos", g(p + "self_attn.linear_pos.weight")),
            (q + "att.o", g(p + "self_attn.linear_out.weight")),
            (q + "att.bias_u", g(p + "self_attn.pos_bias_u").reshape(D)),
            (q + "att.bias_v", g(p + "self_attn.pos_bias_v").reshape(D)),
            (q + "conv.norm.w", g(p + "norm_conv.weight")),
            (q + "conv.norm.b", g(p + "norm_conv.bias")),
            (q + "conv.pw1", g(p + "conv.pointwise_conv1.weight")[:, :, 0]),
            (q + "conv.dw", dw * scale[:, None]),
            (q + "conv.dw_b", g(bn + "bias") - g(bn + "running_mean") * scale),
            (q + "conv.pw2", g(p + "conv.pointwise_conv2.weight")[:, :, 0]),
            (q + "ff2.norm.w", g(p + "norm_feed_forward2.weight")),
            (q + "ff2.norm.b", g(p + "norm_feed_forward2.bias")),
            (q + "ff2.up", g(p + "feed_forward2.linear1.weight")),
            (q + "ff2.down", 0.5 * g(p + "feed_forward2.linear2.weight")),
            (q + "out.norm.w", g(p + "norm_out.weight")),
            (q + "out.norm.b", g(p + "norm_out.bias")),
        ]
        assert H * (D // H) == D
    p = "decoder.prediction."
    out.append(("dec.embed", g(p + "embed.weight")))
    for i in range(a["pred_layers"]):
        r = p + "dec_rnn.lstm."
        out += [("dec.l%d.wx" % i, g(r + "weight_ih_l%d" % i)),
                ("dec.l%d.wh" % i, g(r + "weight_hh_l%d" % i)),
                ("dec.l%d.b" % i, g(r + "bias_ih_l%d" % i) + g(r + "bias_hh_l%d" % i))]
    out += [("joint.enc.w", g("joint.enc.weight")), ("joint.enc.b", g("joint.enc.bias")),
            ("joint.pred.w", g("joint.pred.weight")), ("joint.pred.b", g("joint.pred.bias")),
            ("joint.out.w", g("joint.joint_net.2.weight")), ("joint.out.b", g("joint.joint_net.2.bias"))]
    left = sorted(set(s) - used)
    if left:
        raise ValueError("tensors pkito does not know what to do with: %s" % left[:8])
    return out
