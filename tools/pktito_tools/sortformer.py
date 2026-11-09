"""NVIDIA's Streaming Sortformer speaker diarizer in float64 numpy: who
speaks when, as each of four speakers' probability every 80 ms. It is to
a diarizer in C what reference.py is to the recognizer: the definition the
engine would be held to when NeMo is not at hand (docs/DIARIZATION.md).

This is NeMo 3.0's arithmetic for diar_streaming_sortformer_4spk-v2
(SortformerEncLabelModel.forward_streaming, and SortformerModules'
streaming_update and _compress_spkcache) at batch size one:

  features    Parakeet's log-mel (reference.features), not normalized
  pre-encode  Parakeet's subsampling (reference.subsample), a chunk at a
              time: 340 frames (27.2 s) and up to 1 frame of left and 40 of
              right context, each chunk's convolutions seeing only it
  a step      [speaker cache; FIFO; the chunk with its context] through a
              17-layer FastConformer (reference.conformer, with biases and
              xscaling), a projection to 192, an 18-layer Transformer
              (post-LN, ReLU) and a head: a sigmoid for each speaker
  update      the chunk's frames join the FIFO; what leaves it joins the
              speaker cache, which is then cut back to 188 frames: for each
              speaker, the frames most clearly that speaker alone, and three
              of the recording's mean silence (the Arrival-Order Speaker
              Cache of arxiv.org/abs/2507.18446)

The speakers come out in the order they first spoke, column 0 whoever spoke
first, and the cache is what keeps that order over an hour: a later chunk
attends to earlier frames of each speaker it has heard.

The cache, the chunking and the loop (Stream, chunks, run) are also
Nemotron 3 Diarization's, whose network is nemotron3.py's: v2's is the
pre_encode and forward here.

Where NeMo leaves an answer to the implementation, this fixes one: when
scores tie, the speaker cache keeps the lower index (torch.topk on a CPU
returns tied values in an order of its own; DIARIZATION.md, "Ties").
"""
import math

import numpy as np

from . import canon, reference
from .reference import _ln, _sigmoid, lin

# The model card's configuration for a recording heard after the fact
# ("very high latency", 30.4 s): chunks of 340 frames with 40 of right
# context, a FIFO of 40 and a cache update every 300 frames.
OFFLINE = {"chunk_len": 340, "chunk_right_context": 40, "fifo_len": 40,
           "spkcache_update_period": 300, "spkcache_len": 188}

# SortformerModules' defaults in NeMo 3.0, which the checkpoint's config
# does not override
MODULE_DEFAULTS = {"chunk_left_context": 1, "chunk_right_context": 1, "spkcache_sil_frames_per_spk": 3,
                   "pred_score_threshold": 0.25, "scores_boost_latest": 0.05, "sil_threshold": 0.2,
                   "strong_boost_rate": 0.75, "weak_boost_rate": 1.5, "min_pos_scores_rate": 0.5,
                   "max_index": 99999}


def arch(cfg, streaming=None):
    """The numbers the reference needs, from the checkpoint's config, with
    `streaming` (OFFLINE, or None for the checkpoint's own) over the
    module's settings. Anything this does not implement is refused."""
    pre, enc = cfg["preprocessor"], cfg["encoder"]
    tf, sm = cfg["transformer_encoder"], cfg["sortformer_modules"]
    want = [
        (str(cfg.get("target", "")).endswith("SortformerEncLabelModel"), "a SortformerEncLabelModel"),
        (cfg.get("streaming_mode") is True, "a streaming model"),
        (pre.get("_target_", "").endswith("AudioToMelSpectrogramPreprocessor"), "the preprocessor"),
        (pre.get("normalize") == "NA", "unnormalized features"),
        (pre.get("window") == "hann", "a Hann window"),
        (pre.get("frame_splicing", 1) == 1, "no frame splicing"),
        (enc.get("_target_", "").endswith("ConformerEncoder"), "a ConformerEncoder"),
        (enc.get("subsampling") == "dw_striding" and enc.get("subsampling_factor") == 8, "dw_striding by 8"),
        (enc.get("self_attention_model") == "rel_pos", "rel_pos attention"),
        (list(enc.get("att_context_size", [-1, -1])) == [-1, -1], "full attention"),
        (enc.get("untie_biases") is True, "per-layer position biases"),
        (enc.get("use_bias", True) is True, "biased encoder matrices"),
        (enc.get("conv_norm_type") == "batch_norm", "batch norm in the convolution module"),
        (tf.get("_target_", "").endswith("transformer_encoders.TransformerEncoder"),
         "NeMo's TransformerEncoder"),
        (tf.get("pre_ln") is False and tf.get("hidden_act") == "relu", "a post-LN ReLU Transformer"),
        (cfg["model_defaults"]["tf_d_model"] != enc["d_model"], "a projection to the Transformer's width"),
    ]
    for ok, what in want:
        if not ok:
            raise ValueError("this reference implements Streaming Sortformer with %s; "
                             "this checkpoint differs" % what)
    sr = int(pre["sample_rate"])
    win, hop = round(pre["window_size"] * sr), round(pre["window_stride"] * sr)
    a = {"sample_rate": sr, "n_fft": int(pre["n_fft"]), "win_length": win, "hop_length": hop,
         "n_mels": int(pre["features"]), "preemph": float(pre.get("preemph", 0.97)),
         "log_guard": float(pre.get("log_zero_guard_value", 2 ** -24)), "norm_eps": 1e-5, "normalize": False,
         "n_layers": int(enc["n_layers"]), "d_model": int(enc["d_model"]), "n_heads": int(enc["n_heads"]),
         "ff_dim": int(enc["d_model"] * enc["ff_expansion_factor"]),
         "conv_kernel": int(enc["conv_kernel_size"]),
         "sub_channels": int(enc["subsampling_conv_channels"]), "ln_eps": 1e-5,
         "xscale": math.sqrt(enc["d_model"]) if enc.get("xscaling", True) else None,
         "tf_layers": int(tf["num_layers"]), "tf_d_model": int(tf["hidden_size"]),
         "tf_heads": int(tf["num_attention_heads"]), "tf_inner": int(tf["inner_size"]),
         "n_spk": int(sm["num_spks"]), "subsampling": 8, "upsample": 1, "frame": 8 * hop / sr,
         "learned_silence": False}
    mod = dict(MODULE_DEFAULTS)
    mod.update({k: sm[k] for k in ("spkcache_len", "fifo_len", "chunk_len", "spkcache_update_period",
                                   "chunk_left_context", "chunk_right_context", "spkcache_sil_frames_per_spk")
                if k in sm})
    mod.update({k: sm[k] for k in MODULE_DEFAULTS if k in sm})
    if streaming:
        mod.update(streaming)
    a.update(mod)
    return a


def tensors(state, a):
    """{name: float64 array} under reference.py's names, from a NeMo state
    dict: the FastConformer's as canon.py names Parakeet's, each matrix's
    bias as "<name>.b", BatchNorm folded into the depthwise convolution
    (with its bias), the feed-forward's 0.5 into its second matrix and
    bias; then the projection, the Transformer ("t<i>.") and the head."""
    s = {k: np.asarray(v, np.float64) for k, v in state.items() if not k.endswith("num_batches_tracked")}
    used = set()

    def g(name):
        used.add(name)
        return s[name]

    D, C = a["d_model"], a["sub_channels"]
    w = {"mel.fb": g("preprocessor.featurizer.fb")[0], "mel.window": g("preprocessor.featurizer.window"),
         "pos.freq": canon.pos_freq(D).astype(np.float64)}
    p = "encoder.pre_encode."
    for name, k in (("conv0", 0), ("dw1", 2), ("pw1", 3), ("dw2", 5), ("pw2", 6)):
        wt = g(p + "conv.%d.weight" % k)
        w["sub.%s.w" % name] = wt.reshape(C, C) if name.startswith("pw") else wt.reshape(C, 9)
        w["sub.%s.b" % name] = g(p + "conv.%d.bias" % k)
    w["sub.out.w"], w["sub.out.b"] = g(p + "out.weight"), g(p + "out.bias")
    for i in range(a["n_layers"]):
        p, q = "encoder.layers.%d." % i, "l%d." % i
        bn = p + "conv.batch_norm."
        scale = g(bn + "weight") / np.sqrt(g(bn + "running_var") + canon.BN_EPS)
        for ours, theirs in (("ff1", "feed_forward1"), ("ff2", "feed_forward2")):
            w[q + ours + ".up"] = g(p + theirs + ".linear1.weight")
            w[q + ours + ".up.b"] = g(p + theirs + ".linear1.bias")
            w[q + ours + ".down"] = 0.5 * g(p + theirs + ".linear2.weight")
            w[q + ours + ".down.b"] = 0.5 * g(p + theirs + ".linear2.bias")
        for ours, theirs in (("ff1.norm", "norm_feed_forward1"), ("att.norm", "norm_self_att"),
                             ("conv.norm", "norm_conv"), ("ff2.norm", "norm_feed_forward2"),
                             ("out.norm", "norm_out")):
            w[q + ours + ".w"], w[q + ours + ".b"] = g(p + theirs + ".weight"), g(p + theirs + ".bias")
        for ours, theirs in (("q", "linear_q"), ("k", "linear_k"), ("v", "linear_v"), ("o", "linear_out")):
            w[q + "att." + ours] = g(p + "self_attn.%s.weight" % theirs)
            w[q + "att." + ours + ".b"] = g(p + "self_attn.%s.bias" % theirs)
        w[q + "att.pos"] = g(p + "self_attn.linear_pos.weight")
        w[q + "att.bias_u"] = g(p + "self_attn.pos_bias_u").reshape(D)
        w[q + "att.bias_v"] = g(p + "self_attn.pos_bias_v").reshape(D)
        w[q + "conv.pw1"] = g(p + "conv.pointwise_conv1.weight")[:, :, 0]
        w[q + "conv.pw1.b"] = g(p + "conv.pointwise_conv1.bias")
        w[q + "conv.dw"] = g(p + "conv.depthwise_conv.weight")[:, 0, :] * scale[:, None]
        dw_b = g(p + "conv.depthwise_conv.bias")
        w[q + "conv.dw_b"] = g(bn + "bias") + (dw_b - g(bn + "running_mean")) * scale
        w[q + "conv.pw2"] = g(p + "conv.pointwise_conv2.weight")[:, :, 0]
        w[q + "conv.pw2.b"] = g(p + "conv.pointwise_conv2.bias")
    p = "sortformer_modules."
    for ours, theirs in (("proj", "encoder_proj"), ("head.hidden", "first_hidden_to_hidden"),
                         ("head.out", "single_hidden_to_spks")):
        w[ours], w[ours + ".b"] = g(p + theirs + ".weight"), g(p + theirs + ".bias")
    g(p + "hidden_to_spks.weight"), g(p + "hidden_to_spks.bias")      # training's head only
    for i in range(a["tf_layers"]):
        p, q = "transformer_encoder.layers.%d." % i, "t%d." % i
        for ours, theirs in (("att.q", "first_sub_layer.query_net"),
                             ("att.k", "first_sub_layer.key_net"),
                             ("att.v", "first_sub_layer.value_net"),
                             ("att.o", "first_sub_layer.out_projection"),
                             ("ff.up", "second_sub_layer.dense_in"),
                             ("ff.down", "second_sub_layer.dense_out")):
            w[q + ours], w[q + ours + ".b"] = g(p + theirs + ".weight"), g(p + theirs + ".bias")
        for n in ("1", "2"):
            w[q + "norm%s.w" % n] = g(p + "layer_norm_%s.weight" % n)
            w[q + "norm%s.b" % n] = g(p + "layer_norm_%s.bias" % n)
    left = sorted(set(s) - used)
    if left:
        raise ValueError("tensors this reference does not know what to do with: %s" % left[:8])
    return w


# -------------------------------------------------------------- the model ---

def tf_layer(x, w, i, a):
    """NeMo's TransformerEncoderBlock, post-LN: x = LN(x + attention(x));
    x = LN(x + ff(x)). Its attention divides the queries and the keys each
    by the fourth root of the head's width, rather than the scores by its
    square root, and has no positions: the FastConformer below gave them."""
    q = "t%d." % i
    T, D = x.shape
    H = a["tf_heads"]
    dk = D // H
    root4 = math.sqrt(math.sqrt(dk))
    Q = (lin(x, w, q + "att.q") / root4).reshape(T, H, dk).transpose(1, 0, 2)
    K = (lin(x, w, q + "att.k") / root4).reshape(T, H, dk).transpose(1, 0, 2)
    V = lin(x, w, q + "att.v").reshape(T, H, dk).transpose(1, 0, 2)
    s = Q @ K.transpose(0, 2, 1)                                   # [H][T][T]
    s = np.exp(s - s.max(-1, keepdims=True))
    s /= s.sum(-1, keepdims=True)
    o = (s @ V).transpose(1, 0, 2).reshape(T, D)
    x = _ln(x + lin(o, w, q + "att.o"), w[q + "norm1.w"], w[q + "norm1.b"], a["ln_eps"])
    h = lin(np.maximum(lin(x, w, q + "ff.up"), 0), w, q + "ff.down")
    return _ln(x + h, w[q + "norm2.w"], w[q + "norm2.b"], a["ln_eps"])


def pre_encode(feat, w, a):
    """A chunk's features [T][128] to 80 ms frames [ceil(T / 8)][512]:
    Parakeet's subsampling, the chunk's edges padded as its own."""
    return reference.subsample(feat, w, a)


def forward(embs, w, a):
    """Speaker probabilities [T][n_spk] for pre-encoded frames [T][d_model]:
    the FastConformer's layers, the projection, the Transformer, and the
    head (ReLU, a 192 x 192 matrix, ReLU, one row a speaker, sigmoid)."""
    x = lin(reference.conformer(embs, w, a), w, "proj")
    for i in range(a["tf_layers"]):
        x = tf_layer(x, w, i, a)
    h = np.maximum(lin(np.maximum(x, 0), w, "head.hidden"), 0)
    return _sigmoid(lin(h, w, "head.out"))


# --------------------------------------------------------- the speaker cache ---

def _top(col, k):
    """The indices of the k largest of col, ties to the lower index."""
    return np.argsort(-col, kind="stable")[:k]


class Stream:
    """One recording's streaming state: the speaker cache and its
    probabilities, the FIFO and its, and the silence frame - the
    checkpoint's learned one (Nemotron 3), or the running mean of the
    frames the model called silence (v2). pre_encode and forward are the
    model's network. `ties` counts the cache cuts whose choice a tie
    decided."""

    def __init__(self, w, a, pre_encode, forward):
        self.w, self.a = w, a
        self.pre_encode, self.forward = pre_encode, forward
        D, S = a["d_model"], a["n_spk"]
        dt = w["mel.fb"].dtype                  # the weights' precision: float64, or float32 (diarquant.py)
        self.cache, self.cache_p = np.zeros((0, D), dt), None
        self.fifo = np.zeros((0, D), dt)
        self.sil, self.n_sil = (w["sil"] if a["learned_silence"] else np.zeros(D, dt)), 0
        self.ties = 0
        self.steps = 0
        self.S = S

    def step(self, feat, lc, rc):
        """The chunk's features (with lc and rc frames of context, in encoder
        frames) through the model: the probabilities of its n frames,
        [n * upsample][n_spk] (v2's upsample is 1, Nemotron 3's 8)."""
        a, up = self.a, self.a["upsample"]
        emb = self.pre_encode(feat, self.w, a)
        nc, nf = len(self.cache), len(self.fifo)
        out = self.forward(np.concatenate([self.cache, self.fifo, emb]), self.w, a)
        # the cache is scored a frame at a time, on the mean of the frame's
        # finer probabilities (SortformerModules.downsample_preds)
        p = out if up == 1 else out.reshape(-1, up, self.S).mean(1)
        n = len(emb) - lc - rc
        chunk, chunk_p = emb[lc:lc + n], p[nc + nf + lc:nc + nf + lc + n]
        fifo, fifo_p = np.concatenate([self.fifo, chunk]), np.concatenate([p[nc:nc + nf], chunk_p])
        if nf + n > a["fifo_len"]:
            k = min(max(a["spkcache_update_period"], n - a["fifo_len"] + nf), nf + n)
            pop, pop_p = fifo[:k], fifo_p[:k]
            if not a["learned_silence"]:
                self._silence(pop, pop_p)
            fifo = fifo[k:]
            self.cache = np.concatenate([self.cache, pop])
            if self.cache_p is not None:
                self.cache_p = np.concatenate([self.cache_p, pop_p])
            if len(self.cache) > a["spkcache_len"]:
                if self.cache_p is None:                    # the first cut: this step's view of the cache
                    self.cache_p = np.concatenate([p[:nc], pop_p])
                self.cache, self.cache_p = self._compress(self.cache, self.cache_p)
        self.fifo = fifo
        self.steps += 1
        return out[(nc + nf + lc) * up:(nc + nf + lc + n) * up]

    def _silence(self, emb, p):
        """SortformerModules._get_silence_profile: the running mean of the
        frames whose probabilities sum to less than sil_threshold."""
        sil = p.sum(1) < self.a["sil_threshold"]
        k = int(sil.sum())
        if k:
            total = self.sil * self.n_sil + emb[sil].sum(0)
            self.n_sil += k
            self.sil = total / max(self.n_sil, 1)

    def _boost(self, sc, k, scale):
        """Raise each speaker's k highest scores by -scale log 0.5."""
        for s in range(self.S):
            sc[_top(sc[:, s], k), s] -= scale * math.log(0.5)
        return sc

    def _compress(self, emb, p):
        """SortformerModules._compress_spkcache: keep spkcache_len of the
        frames, the ones _choose picks, in order, with silence frames
        (self.sil) where it says so."""
        idx, off = self._choose(p)
        emb2 = np.where(off[:, None], self.sil[None, :], emb[idx])
        p2 = np.where(off[:, None], 0.0, p[idx])
        return emb2, p2

    def _choose(self, p):
        """Which frames the cut keeps, by speaker: spkcache_len indices into
        p, in order, and where a silence frame goes instead (off). A frame's
        score for a speaker is how much more likely it is that speaker alone
        than not; frames that are not that speaker's speech, and (once a
        speaker has enough frames of its own) frames shared with another,
        never qualify; the newest frames get a little extra, each speaker's
        best few a lot (so that every speaker keeps frames) and its next best
        some (so that none takes over); and each speaker's run ends in
        silence frames."""
        a, S = self.a, self.S
        L, nsil = a["spkcache_len"], a["spkcache_sil_frames_per_spk"]
        per = L // S - nsil
        thr = a["pred_score_threshold"]
        lp = np.log(np.maximum(p, thr))
        l1p = np.log(np.maximum(1.0 - p, thr))
        sc = lp - l1p + l1p.sum(1, keepdims=True) - math.log(0.5)
        speech = p > 0.5
        sc = np.where(speech, sc, -np.inf)
        pos = sc > 0
        enough = pos.sum(0, keepdims=True) >= math.floor(per * a["min_pos_scores_rate"])
        sc = np.where(~pos & speech & enough, -np.inf, sc)
        if a["scores_boost_latest"] > 0:
            sc[L:] += a["scores_boost_latest"]
        sc = self._boost(sc, math.floor(per * a["strong_boost_rate"]), 2)
        sc = self._boost(sc, math.floor(per * a["weak_boost_rate"]), 1)
        n = len(sc)
        if nsil:
            sc = np.concatenate([sc, np.full((nsil, S), np.inf)])
        flat = sc.T.reshape(-1)                                     # speaker-major
        pick = _top(flat, L)
        cut = flat[pick[-1]]
        if np.isfinite(cut) and (flat == cut).sum() > (flat[pick] == cut).sum():
            self.ties += 1
        idx = np.where(flat[pick] != -np.inf, pick, a["max_index"])
        idx = np.sort(idx)
        off = idx == a["max_index"]
        idx = idx % (n + nsil)
        off |= idx >= n
        idx[off] = 0
        return idx, off


def chunks(T0, a):
    """SortformerModules.streaming_feat_loader: (first, last, left, right)
    feature frames of each chunk, the context in feature frames."""
    f = a["subsampling"]
    out, stt, end = [], 0, 0
    while end < T0:
        left = min(a["chunk_left_context"] * f, stt)
        end = min(stt + a["chunk_len"] * f, T0)
        right = min(a["chunk_right_context"] * f, T0 - end)
        out.append((stt, end, left, right))
        stt = end
    return out


def run(audio, w, a, pre_encode, forward, log=None):
    """A model's speaker probabilities for 16 kHz audio, a row every
    a["frame"] seconds, and the Stream: the features, then the chunks
    through the network and the cache. The rows are as many as NeMo
    returns: one for every 8 feature frames or part of 8 (v2), one for
    every feature frame (Nemotron 3). The features are computed in float64
    and the rest in the weights' precision, as quantcheck.py runs
    Parakeet."""
    feat = reference.features(audio, w, a).astype(w["mel.fb"].dtype, copy=False)
    st = Stream(w, a, pre_encode, forward)
    f, up = a["subsampling"], a["upsample"]
    out = []
    for i, (stt, end, left, right) in enumerate(chunks(len(feat), a)):
        p = st.step(feat[stt - left:end + right], round(left / f), math.ceil(right / f))
        out.append(p)
        if log:
            log("  chunk %d: frames %d to %d, cache %d, fifo %d" % (i, stt // f, stt // f + len(p) // up,
                                                                   len(st.cache), len(st.fifo)))
    n = -(-len(feat) * up // f)
    return np.concatenate(out)[:n] if out else np.zeros((0, a["n_spk"])), st


def diarize(audio, w, a, log=None):
    """v2's speaker probabilities [ceil(T / 8)][n_spk] for 16 kHz audio,
    T = len(audio) // 160, and the Stream."""
    return run(audio, w, a, pre_encode, forward, log)


def segments(p, frame, thr=0.5):
    """[(start, end, speaker)] in seconds: each speaker's runs of frames
    over thr, NeMo's default post-processing (onset = offset = thr, no
    padding, no minimum duration), sorted by start."""
    out = []
    for s in range(p.shape[1]):
        on = np.concatenate([[False], p[:, s] > thr, [False]])
        edges = np.flatnonzero(on[1:] != on[:-1])
        for b, e in zip(edges[::2], edges[1::2]):
            out.append((b * frame, e * frame, s))
    return sorted(out)
