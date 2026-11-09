"""NVIDIA's Nemotron 3 Diarization in float64 numpy: who speaks when, as
each of eight speakers' probability every 10 ms. It is a Streaming
Sortformer, so its speaker cache is sortformer.py's; what is new is the
network, and that is this file (docs/DIARIZATION.md).

This is NeMo's arithmetic for the released checkpoint, as NeMo's repository
has it at NVIDIA-NeMo/Speech commit cd61689d41b1962d3e1957ae8faf28bf8bee0e3b
(2026-11-06; NeMo 3.0.0, the latest release, cannot load the checkpoint):
SortformerEncLabelModel.forward_streaming with its TransformerEncoder, and
SortformerModules' streaming_update and _compress_spkcache, at batch size
one:

  features    Parakeet's log-mel (reference.features), not normalized, with
              the checkpoint's own filterbank and window: Parakeet's,
              rounded to bfloat16
  pre-encode  8 feature frames side by side, the last group padded with
              zeros, and a matrix from 1024 to 512: an 80 ms frame
  a step      [speaker cache; FIFO; the chunk with its right context]
              through a LayerNorm, 31 pre-LN Transformer layers (8 heads of
              64 with rotary positions counted from the step's first frame,
              a GELU feed-forward of 2048) and a LayerNorm; a projection to
              192; a convolution over 3 frames to 8 x 192, eight 10 ms
              frames for each 80 ms one; and the head
  update      sortformer.py's: a cache of 264 frames for eight speakers,
              one silence frame each, the checkpoint's learned one, and
              each 80 ms frame scored on the mean of its eight 10 ms
              probabilities
"""

import math

import numpy as np

from . import sortformer
from .reference import _ln, _sigmoid, lin, mm

# The model card's configuration for a recording heard after the fact
# ("very high latency", 30.4 s): chunks of 340 frames with 40 of right
# context, a FIFO of 40 and a cache update every 300 frames. The cache keeps
# the checkpoint's 264 frames, as the card's table does.
OFFLINE = {
    "chunk_len": 340,
    "chunk_right_context": 40,
    "fifo_len": 40,
    "spkcache_update_period": 300,
}


def arch(cfg, streaming=None):
    """The numbers the reference needs, from the checkpoint's config, with
    `streaming` (OFFLINE, or None for the checkpoint's own) over the
    module's settings. Anything this does not implement is refused."""
    pre, enc, sm = cfg["preprocessor"], cfg["encoder"], cfg["sortformer_modules"]
    want = [
        (
            str(cfg.get("target", "")).endswith("SortformerEncLabelModel"),
            "a SortformerEncLabelModel",
        ),
        (cfg.get("streaming_mode") is True, "a streaming model"),
        (
            cfg.get("high_resolution") is True
            and cfg.get("output_subsampling_factor") == 1,
            "10 ms output",
        ),
        (not cfg.get("transformer_encoder"), "no second Transformer"),
        (
            pre.get("_target_", "").endswith("AudioToMelSpectrogramPreprocessor"),
            "the preprocessor",
        ),
        (pre.get("normalize") == "NA", "unnormalized features"),
        (pre.get("window") == "hann", "a Hann window"),
        (pre.get("frame_splicing", 1) == 1, "no frame splicing"),
        (
            enc.get("_target_", "").endswith("modules.TransformerEncoder"),
            "NeMo's TransformerEncoder",
        ),
        (
            enc.get("subsampling") == "feature_stacking"
            and enc.get("subsampling_factor") == 8,
            "8 feature frames stacked",
        ),
        (
            enc.get("self_attention_model") == "rope"
            and enc.get("rotary_fraction", 1.0) == 1.0,
            "rotary positions on whole heads",
        ),
        (enc.get("attn_mode", "full") == "full", "full attention"),
        (
            enc.get("pre_block_norm") is True and enc.get("final_norm", True) is True,
            "a LayerNorm before the first layer and after the last",
        ),
        (
            enc.get("qkv_bias") is False and enc.get("qk_norm") is False,
            "unbiased, unnormalized q, k, v",
        ),
        (enc.get("xscaling") is False, "no xscaling"),
        (enc.get("feat_out", -1) == -1, "no output projection in the encoder"),
        (sm.get("use_learnable_sil_emb") is True, "a learned silence frame"),
    ]
    for ok, what in want:
        if not ok:
            raise ValueError(
                "this reference implements Nemotron 3 Diarization with %s; "
                "this checkpoint differs" % what
            )
    sr = int(pre["sample_rate"])
    win, hop = round(pre["window_size"] * sr), round(pre["window_stride"] * sr)
    f = int(enc["subsampling_factor"])
    a = {
        "sample_rate": sr,
        "n_fft": int(pre["n_fft"]),
        "win_length": win,
        "hop_length": hop,
        "n_mels": int(pre["features"]),
        "preemph": float(pre.get("preemph", 0.97)),
        "log_guard": float(pre.get("log_zero_guard_value", 2**-24)),
        "norm_eps": 1e-5,
        "normalize": False,
        "n_layers": int(enc["n_layers"]),
        "d_model": int(enc["d_model"]),
        "n_heads": int(enc["n_heads"]),
        "ff_dim": int(enc["d_model"] * enc["ff_expansion"]),
        "rope_base": float(enc.get("rope_base", 10000.0)),
        "ln_eps": 1e-5,
        "tf_d_model": int(cfg["model_defaults"]["tf_d_model"]),
        "n_spk": int(sm["num_spks"]),
        "subsampling": f,
        "upsample": f,
        "frame": hop / sr,
        "learned_silence": True,
    }
    mod = dict(sortformer.MODULE_DEFAULTS)
    mod.update(
        {
            k: sm[k]
            for k in (
                "spkcache_len",
                "fifo_len",
                "chunk_len",
                "spkcache_update_period",
                "chunk_left_context",
                "chunk_right_context",
                "spkcache_sil_frames_per_spk",
            )
            if k in sm
        }
    )
    mod.update({k: sm[k] for k in sortformer.MODULE_DEFAULTS if k in sm})
    if streaming:
        mod.update(streaming)
    a.update(mod)
    return a


def tensors(state, a):
    """{name: float64 array} from a NeMo state dict: the mel tables, the
    stacking matrix, each layer's under Parakeet's names ("l<i>.att.q",
    "l<i>.ff.up", each bias "<name>.b"; NeMo's fused query, key and value
    matrix split in three), the projection, the upsampler as a matrix over
    three frames side by side, the head, and the silence frame."""
    s = {k: np.asarray(v, np.float64) for k, v in state.items()}
    used = set()

    def g(name):
        used.add(name)
        return s[name]

    D, C = a["d_model"], a["tf_d_model"]
    w = {
        "mel.fb": g("preprocessor.featurizer.fb")[0],
        "mel.window": g("preprocessor.featurizer.window"),
        "stack": g("encoder.pre_encode.proj.weight"),
        "sil": g("sortformer_modules.learnable_sil_emb"),
    }
    for ours, theirs in (("embed_norm", "embed_norm"), ("final_norm", "final_norm")):
        w[ours + ".w"], w[ours + ".b"] = (
            g("encoder.%s.weight" % theirs),
            g("encoder.%s.bias" % theirs),
        )
    for i in range(a["n_layers"]):
        p, q = "encoder.layers.%d." % i, "l%d." % i
        qkv = g(p + "attn.w_qkv.weight")  # rows: q, k, v; in each, head by head
        w[q + "att.q"], w[q + "att.k"], w[q + "att.v"] = (
            qkv[:D],
            qkv[D : 2 * D],
            qkv[2 * D :],
        )
        w[q + "att.o"], w[q + "att.o.b"] = (
            g(p + "attn.out_proj.weight"),
            g(p + "attn.out_proj.bias"),
        )
        w[q + "ff.up"], w[q + "ff.up.b"] = (
            g(p + "ffn.net.0.weight"),
            g(p + "ffn.net.0.bias"),
        )
        w[q + "ff.down"], w[q + "ff.down.b"] = (
            g(p + "ffn.net.3.weight"),
            g(p + "ffn.net.3.bias"),
        )
        for ours, theirs in (("att.norm", "norm1"), ("ff.norm", "norm2")):
            w[q + ours + ".w"], w[q + ours + ".b"] = (
                g(p + theirs + ".weight"),
                g(p + theirs + ".bias"),
            )
    p = "sortformer_modules."
    for ours, theirs in (
        ("proj", "encoder_proj"),
        ("head.hidden", "first_hidden_to_hidden"),
        ("head.out", "single_hidden_to_spks"),
    ):
        w[ours], w[ours + ".b"] = g(p + theirs + ".weight"), g(p + theirs + ".bias")
    up = g(p + "subpixel_upsample.weight")  # [8 C][C][3 taps]
    w["up"] = up.transpose(0, 2, 1).reshape(up.shape[0], 3 * C)  # [8 C][tap, C]
    w["up.b"] = g(p + "subpixel_upsample.bias")
    for name in sorted(s):  # training's heads only
        if name.startswith((p + "hidden_to_spks.", p + "activity_head.")):
            g(name)
    left = sorted(set(s) - used)
    if left:
        raise ValueError(
            "tensors this reference does not know what to do with: %s" % left[:8]
        )
    return w


# -------------------------------------------------------------- the model ---

# erf's Maclaurin series below 1.5 and erfc's continued fraction from there
# (Abramowitz and Stegun 7.1.5 and 7.1.14), numpy having no erf: within
# 1.4e-14 of math.erfc, relatively, at 600,000 points between -10 and 10
# (measured for docs/DIARIZATION.md).
_ERF_SERIES = [(-1) ** n / (math.factorial(n) * (2 * n + 1)) for n in range(28)]
_ERFC_DEPTH = 80


def _erfc(x):
    z = np.abs(x)
    out = np.empty_like(z)
    near = z < 1.5
    zn = z[near]
    s = np.full_like(zn, _ERF_SERIES[-1])
    for c in reversed(_ERF_SERIES[:-1]):
        s = s * (zn * zn) + c
    out[near] = 1.0 - 2.0 / math.sqrt(math.pi) * zn * s
    zf = z[~near]
    f = zf.copy()
    for k in range(_ERFC_DEPTH, 0, -1):
        f = zf + (k / 2.0) / f
    out[~near] = np.exp(-zf * zf) / (math.sqrt(math.pi) * f)
    return np.where(x < 0, 2.0 - out, out)


def _gelu(x):
    """torch's GELU, x times the normal distribution's CDF at x (not the
    tanh approximation): x/2 erfc(-x/sqrt 2), which stays accurate where
    the CDF is tiny."""
    return 0.5 * x * _erfc(-x / math.sqrt(2.0))


def pre_encode(feat, w, a):
    """FeatureStacking: each 8 feature frames [8][128] side by side as one
    of 1024, the last group padded with zeros, through a 512 x 1024 matrix
    with no bias: [ceil(T / 8)][512]."""
    f = a["subsampling"]
    T, M = feat.shape
    x = np.concatenate([feat, np.zeros((-T % f, M), feat.dtype)]).reshape(-1, f * M)
    return mm(x, w, "stack")


def rope(T, a, dtype=np.float64):
    """cos and sin [T][64] of the rotary positions 0 .. T-1: a head's
    dimensions i and i + 32 turn as a pair, by the position times
    10000^(-i/32). Computed in float64 and rounded to dtype, the
    precision the model runs in (float32 for tools/diarquant.py)."""
    dk = a["d_model"] // a["n_heads"]
    inv = 1.0 / a["rope_base"] ** (np.arange(0, dk, 2) / dk)
    ang = np.arange(T)[:, None] * inv[None, :]
    ang = np.concatenate([ang, ang], axis=1)
    return np.cos(ang).astype(dtype), np.sin(ang).astype(dtype)


def _turn(x, cos, sin):
    """Rotate every head's [T][64] by its frame's angles (NeMo's and
    torch's "rotate half": the pair is (i, i + 32), not (2i, 2i + 1))."""
    h = x.shape[-1] // 2
    return x * cos + np.concatenate([-x[..., h:], x[..., :h]], axis=-1) * sin


def layer(x, cos, sin, w, i, a):
    """A pre-LN Transformer block: x += attention(LN(x)); x += ff(LN(x)).
    The attention's queries and keys are turned by their positions, its
    scores divided by the square root of 64; the feed-forward is
    512 -> 2048, GELU, -> 512. Only the output and feed-forward matrices
    have biases."""
    q = "l%d." % i
    T, D = x.shape
    H = a["n_heads"]
    dk = D // H

    def heads(m):
        return m.reshape(T, H, dk).transpose(1, 0, 2)  # [H][T][dk]

    h = _ln(x, w[q + "att.norm.w"], w[q + "att.norm.b"], a["ln_eps"])
    Q = _turn(heads(mm(h, w, q + "att.q")), cos, sin)
    K = _turn(heads(mm(h, w, q + "att.k")), cos, sin)
    V = heads(mm(h, w, q + "att.v"))
    s = Q @ K.transpose(0, 2, 1) / math.sqrt(dk)  # [H][T][T]
    s = np.exp(s - s.max(-1, keepdims=True))
    s /= s.sum(-1, keepdims=True)
    o = (s @ V).transpose(1, 0, 2).reshape(T, D)
    x = x + lin(o, w, q + "att.o")
    h = _ln(x, w[q + "ff.norm.w"], w[q + "ff.norm.b"], a["ln_eps"])
    return x + lin(_gelu(lin(h, w, q + "ff.up")), w, q + "ff.down")


def upsample(x, w, a):
    """SortformerModules.upsample_hidden: a convolution over 3 frames, 192
    channels to 8 x 192 - each frame with the one before and the one after
    it (zeros past the step's ends) through a 1536 x 576 matrix - each
    output frame then eight 10 ms frames: [T][192] -> [8 T][192]."""
    T, C = x.shape
    z = np.zeros((1, C), x.dtype)
    taps = np.concatenate(
        [np.concatenate([z, x[:-1]]), x, np.concatenate([x[1:], z])], axis=1
    )
    return lin(taps, w, "up").reshape(T * a["upsample"], C)


def forward(x, w, a):
    """Speaker probabilities [8 T][n_spk], every 10 ms, for the pre-encoded
    frames [T][512] of one step: the encoder, the projection, the
    upsampler and the head (ReLU, a 192 x 192 matrix, ReLU, one row a
    speaker, sigmoid). The positions start at 0 at every step."""
    cos, sin = rope(len(x), a, x.dtype)
    h = _ln(x, w["embed_norm.w"], w["embed_norm.b"], a["ln_eps"])
    for i in range(a["n_layers"]):
        h = layer(h, cos, sin, w, i, a)
    h = lin(_ln(h, w["final_norm.w"], w["final_norm.b"], a["ln_eps"]), w, "proj")
    u = upsample(h, w, a)
    g = np.maximum(lin(np.maximum(u, 0), w, "head.hidden"), 0)
    return _sigmoid(lin(g, w, "head.out"))


def diarize(audio, w, a, log=None):
    """Speaker probabilities [T][n_spk], one row every 10 ms (T = len(audio)
    // 160, as NeMo returns), for 16 kHz audio, and the Stream."""
    return sortformer.run(audio, w, a, pre_encode, forward, log)
