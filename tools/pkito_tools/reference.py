"""Parakeet TDT in float64 numpy: the definition the engine is checked
against when NeMo is not at hand, and the model the quantization
experiments perturb.

It is NeMo's arithmetic (nemo 3.0: FilterbankFeatures, ConvSubsampling,
ConformerLayer, RelPositionMultiHeadAttention, RNNTDecoder, RNNTJoint and
the label-looping TDT greedy search) in the order NeMo does it, on the
canonical tensors of canon.py, at batch size one and the input's own
length: NeMo's masks, at batch size one, make each of its frames exactly
what this computes (core/encoder.c says why).
"""
import math

import numpy as np


def mm(x, w, name):
    """x @ W.T for the named weight matrix. Every product with a stored
    matrix goes through here, so that tools/quantcheck.py can put a
    quantized one in its place."""
    return x @ w[name].T


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _ln(x, w, b, eps):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


# ------------------------------------------------------------- features ---

def features(audio, w, a):
    """Log-mel features, [T][n_mels], T = len(audio) // hop: preemphasis, a
    centred STFT with zero padding, the power spectrum through the mel
    filterbank, log(x + guard), and each mel band normalized over the
    recording (mean, and the n - 1 standard deviation plus norm_eps)."""
    n_fft, win, hop = a["n_fft"], a["win_length"], a["hop_length"]
    x = np.asarray(audio, np.float64)
    T = len(x) // hop
    y = x.copy()
    y[1:] = x[1:] - a["preemph"] * x[:-1]
    y = np.concatenate([np.zeros(n_fft // 2), y, np.zeros(n_fft // 2)])
    window = np.zeros(n_fft)
    left = (n_fft - win) // 2
    window[left:left + win] = w["mel.window"]
    idx = np.arange(T)[:, None] * hop + np.arange(n_fft)[None, :]
    spec = np.fft.rfft(y[idx] * window, axis=1)
    power = spec.real ** 2 + spec.imag ** 2
    mel = np.log(power @ w["mel.fb"].T + a["log_guard"])          # [T][n_mels]
    mean = mel.mean(0)
    if T > 1:
        std = np.sqrt(((mel - mean) ** 2).sum(0) / (T - 1))
    else:
        std = np.zeros_like(mean)
    return (mel - mean) / (std + a["norm_eps"])


# -------------------------------------------------------------- encoder ---

def _conv3x3_s2(x, w, b, depthwise):
    """3x3 convolution, stride 2, padding 1, over x [C][T][F]; w [O][9] is
    one tap set per output channel, applied to channel o (depthwise) or to
    the single input channel."""
    C, T, F = x.shape
    To, Fo = (T - 1) // 2 + 1, (F - 1) // 2 + 1
    xp = np.zeros((C, T + 2, F + 2))
    xp[:, 1:-1, 1:-1] = x
    out = np.zeros((w.shape[0], To, Fo))
    for di in range(3):
        for dj in range(3):
            tap = xp[:, di:di + 2 * To:2, dj:dj + 2 * Fo:2]
            out += w[:, 3 * di + dj, None, None] * tap
    return out + b[:, None, None]


def subsample(feat, w, a):
    """[T][n_mels] -> [T/8][d_model]: three stride-2 convolutions over time
    and frequency (a full one from one channel, then two depthwise ones each
    followed by a pointwise one), ReLU after each stage, then channels and
    frequencies flattened channel-major and projected."""
    C = a["sub_channels"]
    x = feat[None]                                                # [1][T][F]
    x = np.maximum(_conv3x3_s2(x, w["sub.conv0.w"], w["sub.conv0.b"], False), 0)
    for s in ("1", "2"):
        x = _conv3x3_s2(x, w["sub.dw%s.w" % s], w["sub.dw%s.b" % s], True)
        C_, T_, F_ = x.shape
        x = mm(x.reshape(C_, T_ * F_).T, w, "sub.pw%s.w" % s).T.reshape(-1, T_, F_)
        x = x + w["sub.pw%s.b" % s][:, None, None]
        x = np.maximum(x, 0)
    T3 = x.shape[1]
    x = x.transpose(1, 0, 2).reshape(T3, C * x.shape[2])
    return mm(x, w, "sub.out.w") + w["sub.out.b"]


def pos_emb(T, w, a):
    """The relative positions T-1 down to -(T-1), [2T-1][d_model]: sin in
    the even columns and cos in the odd, of position times pos.freq, the
    product rounded to float32 as NeMo forms it."""
    pos = np.arange(T - 1, -T, -1, dtype=np.float32)[:, None]
    ang = (pos * w["pos.freq"].astype(np.float32)[None, :]).astype(np.float64)
    pe = np.zeros((2 * T - 1, a["d_model"]))
    pe[:, 0::2] = np.sin(ang)
    pe[:, 1::2] = np.cos(ang)
    return pe


def attention(x, pe, w, q, a):
    T, D = x.shape
    H = a["n_heads"]
    dk = D // H
    Q = mm(x, w, q + "att.q").reshape(T, H, dk)
    K = mm(x, w, q + "att.k").reshape(T, H, dk)
    V = mm(x, w, q + "att.v").reshape(T, H, dk)
    P = mm(pe, w, q + "att.pos").reshape(2 * T - 1, H, dk)
    qu = (Q + w[q + "att.bias_u"].reshape(H, dk)).transpose(1, 0, 2)
    qv = (Q + w[q + "att.bias_v"].reshape(H, dk)).transpose(1, 0, 2)
    ac = qu @ K.transpose(1, 2, 0)                                # [H][T][T]
    bd = qv @ P.transpose(1, 2, 0)                                # [H][T][2T-1]
    # rel_shift: score (i, j) takes the position embedding of i - j, which
    # is row T-1-i+j of pe
    i, j = np.arange(T)[:, None], np.arange(T)[None, :]
    bd = bd[:, i, T - 1 - i + j]
    s = (ac + bd) / math.sqrt(dk)
    s = np.exp(s - s.max(-1, keepdims=True))
    s /= s.sum(-1, keepdims=True)
    o = (s @ V.transpose(1, 0, 2)).transpose(1, 0, 2).reshape(T, D)
    return mm(o, w, q + "att.o")


def conv_module(x, w, q, a):
    T, D = x.shape
    h = mm(x, w, q + "conv.pw1")                                  # [T][2D]
    h = h[:, :D] * _sigmoid(h[:, D:])                             # GLU
    K = a["conv_kernel"]
    hp = np.zeros((T + K - 1, D))
    hp[K // 2:K // 2 + T] = h
    g = np.zeros((T, D))
    for k in range(K):
        g += w[q + "conv.dw"][:, k] * hp[k:k + T]
    g = _silu(g + w[q + "conv.dw_b"])                             # BatchNorm folded in
    return mm(g, w, q + "conv.pw2")


def ff(x, w, q):
    return mm(_silu(mm(x, w, q + "up")), w, q + "down")         # 0.5 folded into down


def layer(x, pe, w, i, a):
    q, eps = "l%d." % i, a["ln_eps"]
    x = x + ff(_ln(x, w[q + "ff1.norm.w"], w[q + "ff1.norm.b"], eps), w, q + "ff1.")
    x = x + attention(_ln(x, w[q + "att.norm.w"], w[q + "att.norm.b"], eps), pe, w, q, a)
    x = x + conv_module(_ln(x, w[q + "conv.norm.w"], w[q + "conv.norm.b"], eps), w, q, a)
    x = x + ff(_ln(x, w[q + "ff2.norm.w"], w[q + "ff2.norm.b"], eps), w, q + "ff2.")
    return _ln(x, w[q + "out.norm.w"], w[q + "out.norm.b"], eps)


def encode(feat, w, a, layers=None):
    x = subsample(feat, w, a)
    pe = pos_emb(x.shape[0], w, a)
    for i in range(a["n_layers"] if layers is None else layers):
        x = layer(x, pe, w, i, a)
    return x


# -------------------------------------------------------------- decoder ---

class Decoder:
    """The prediction network (an embedding and a stack of LSTMs, torch's
    gate order i f g o) and the joint, one step at a time."""

    def __init__(self, w, a):
        self.w, self.a = w, a
        self.L, self.Hd = a["pred_layers"], a["pred_hidden"]

    def start(self):
        return [np.zeros(self.Hd) for _ in range(self.L)], [np.zeros(self.Hd) for _ in range(self.L)]

    def step(self, label, state):
        h, c = state
        x = self.w["dec.embed"][label]
        h2, c2 = [], []
        for i in range(self.L):
            p = "dec.l%d." % i
            g = mm(x, self.w, p + "wx") + mm(h[i], self.w, p + "wh") + self.w[p + "b"]
            n = self.Hd
            ig, fg, gg, og = _sigmoid(g[:n]), _sigmoid(g[n:2 * n]), np.tanh(g[2 * n:3 * n]), _sigmoid(g[3 * n:])
            cc = fg * c[i] + ig * gg
            hh = og * np.tanh(cc)
            h2.append(hh)
            c2.append(cc)
            x = hh
        return x, (h2, c2)

    def joint(self, fproj, g):
        gp = mm(g, self.w, "joint.pred.w") + self.w["joint.pred.b"]
        return mm(np.maximum(fproj + gp, 0), self.w, "joint.out.w") + self.w["joint.out.b"]


def greedy(enc, w, a, trace=None):
    """NeMo's label-looping TDT greedy search (tdt_label_looping.py) at batch
    size one: [(token, frame, duration)]. The token is the argmax over the
    vocabulary and blank, the duration the argmax over the duration logits;
    a blank of duration 0 is taken as 1, and a run of max_symbols tokens at
    one frame ends in a forced step of one frame. `trace`, a list, gets
    (frame, token, duration index) for every joint evaluation."""
    dec = Decoder(w, a)
    V, durs = a["vocab"], a["durations"]
    blank = V
    T = enc.shape[0]
    fproj = mm(enc, w, "joint.enc.w") + w["joint.enc.b"]
    state = dec.start()
    g, state = dec.step(blank, state)
    out = []
    t, last_t, lasts = 0, -1, 0
    while t < T:
        while True:
            t_label = t
            z = dec.joint(fproj[t], g)
            k = int(np.argmax(z[:V + 1]))
            di = int(np.argmax(z[V + 1:]))
            d = durs[di]
            if trace is not None:
                trace.append((t, k, di))
            if k == blank and d == 0:
                d = 1
            t += d
            if k != blank or t >= T:
                break
        if k == blank:
            break
        out.append((k, t_label, d))
        lasts = lasts + 1 if last_t == t_label else 1
        last_t = t_label
        g, state = dec.step(k, state)
        if lasts >= a["max_symbols"] and last_t == t:
            t += 1
    return out


def transcribe(audio, w, a):
    feat = features(audio, w, a)
    enc = encode(feat, w, a)
    return greedy(enc, w, a), feat, enc
