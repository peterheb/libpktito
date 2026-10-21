"""A tiny .pkf: every tensor, param and piece type a real one has, at sizes
that make it a megabyte, with seeded random weights.

    pktito-tools tiny tests/data/tiny.pkf

It is not a speech recognizer; it is what the C side's checks run on when
they want a model whose every byte they can afford to touch. tests/pkfcheck
builds its malformed corpus from it and runs the whole engine on it under
the sanitizers, and its odd widths (a head of 16, two layers, a 16-band
filterbank with an empty band, 32 subsampling channels) reach shapes the
real models never call the kernels with.

The bytes are a function of the seed alone - no date, no git stamp - so
the committed file is reproducible. Float values sit on a grid of 1/64,
which keeps git's copy of the file small.
"""
import numpy as np

from . import canon, convert, pkf, spm

SEED = 20261020

PIECES = [
    ("<unk>", spm.UNKNOWN), ("<s>", spm.CONTROL), ("<|en|>", spm.USER_DEFINED),
    ("<unused>", spm.UNUSED),
    (spm.SPACE, spm.NORMAL), (spm.SPACE + "the", spm.NORMAL), (spm.SPACE + "a", spm.NORMAL),
    ("s", spm.NORMAL), ("ing", spm.NORMAL), (",", spm.NORMAL), (".", spm.NORMAL),
    ("?", spm.NORMAL), ("!", spm.NORMAL), ("'", spm.NORMAL), ("-", spm.NORMAL),
    ("¿", spm.NORMAL), ("¡", spm.NORMAL), (spm.SPACE + "été", spm.NORMAL),
    ("€", spm.NORMAL), ("日本", spm.NORMAL), (spm.SPACE + "Ü", spm.NORMAL),
    (spm.SPACE + "\U0001f600", spm.NORMAL),
] + [(c, spm.NORMAL) for c in "xyzkqwvb"] + [(spm.SPACE + c, spm.NORMAL) for c in "dfghjlmnpr"]

ARCH = [("sample_rate", 16000), ("n_fft", 512), ("win_length", 400), ("hop_length", 160),
        ("n_mels", 16), ("preemph", 0.97), ("log_guard", 2.0 ** -24), ("norm_eps", 1e-5),
        ("n_layers", 2), ("d_model", 64), ("n_heads", 4), ("ff_dim", 128), ("conv_kernel", 9),
        ("sub_channels", 32), ("ln_eps", 1e-5), ("vocab", len(PIECES)), ("pred_hidden", 32),
        ("pred_layers", 2), ("joint_hidden", 32), ("max_symbols", 10)]
DURATIONS = [0, 1, 2, 3, 4]


def _grid(a):
    return np.round(np.asarray(a, np.float64) * 64) / 64


def tensors(rng):
    """[(name, type, dims, array)] in file order."""
    a = dict(ARCH)
    D, F, C, M = a["d_model"], a["ff_dim"], a["sub_channels"], a["n_mels"]
    K, V, P, J, N = a["conv_kernel"], a["vocab"], a["pred_hidden"], a["joint_hidden"], a["n_fft"]
    out = []

    def f32(name, arr):
        arr = np.asarray(arr, np.float32)
        out.append((name, pkf.F32, arr.shape, arr))

    def q8(name, rows, cols, rows_used=None):
        m = rng.standard_normal((rows, cols)) / np.sqrt(cols)
        if rows_used is not None:
            m[rows_used:] = 0
        q, s = pkf.quantize_q8(m)
        out.append((name, pkf.Q8, q.shape, q))
        out.append((name + ".scale", pkf.F32, s.shape, s))

    def norm(name):
        f32(name + ".w", _grid(1 + 0.1 * rng.standard_normal(D)))
        f32(name + ".b", _grid(0.1 * rng.standard_normal(D)))

    # a triangular filterbank over the bins, band 0 left empty: the engine
    # must read a band with no nonzero bin
    bins = N // 2 + 1
    fb = np.zeros((M, bins))
    edges = np.linspace(1, bins - 1, M + 2)
    k = np.arange(bins)
    for b in range(1, M):
        lo, mid, hi = edges[b], edges[b + 1], edges[b + 2]
        fb[b] = np.maximum(0, np.minimum((k - lo) / (mid - lo), (hi - k) / (hi - mid)))
    f32("mel.fb", _grid(fb))
    f32("mel.window", np.hanning(a["win_length"]))
    j = np.arange(N // 2)
    f32("mel.fft.cos", np.cos(2 * np.pi * j / N))
    f32("mel.fft.sin", np.sin(2 * np.pi * j / N))
    f32("pos.freq", canon.pos_freq(D))

    for s in ("conv0", "dw1", "pw1", "dw2", "pw2"):
        if s.startswith("pw"):
            q8("sub.%s.w" % s, C, C)
        else:
            f32("sub.%s.w" % s, _grid(rng.standard_normal((C, 9)) / 3))
        f32("sub.%s.b" % s, _grid(0.1 * rng.standard_normal(C)))
    q8("sub.out.w", D, C * M // 8)
    f32("sub.out.b", _grid(0.1 * rng.standard_normal(D)))

    for i in range(a["n_layers"]):
        p = "l%d." % i
        norm(p + "ff1.norm")
        q8(p + "ff1.up", F, D)
        q8(p + "ff1.down", D, F)
        norm(p + "att.norm")
        for m in ("q", "k", "v", "pos", "o"):
            q8(p + "att." + m, D, D)
        f32(p + "att.bias_u", _grid(0.1 * rng.standard_normal(D)))
        f32(p + "att.bias_v", _grid(0.1 * rng.standard_normal(D)))
        norm(p + "conv.norm")
        q8(p + "conv.pw1", 2 * D, D)
        f32(p + "conv.dw", _grid(rng.standard_normal((D, K)) / 3))
        f32(p + "conv.dw_b", _grid(0.1 * rng.standard_normal(D)))
        q8(p + "conv.pw2", D, D)
        norm(p + "ff2.norm")
        q8(p + "ff2.up", F, D)
        q8(p + "ff2.down", D, F)
        norm(p + "out.norm")

    emb = _grid(rng.standard_normal((V + 1, P)))
    emb[V] = 0                                       # the blank is zeros
    f32("dec.embed", emb)
    for i in range(a["pred_layers"]):
        q8("dec.l%d.wx" % i, 4 * P, P)
        q8("dec.l%d.wh" % i, 4 * P, P)
        f32("dec.l%d.b" % i, _grid(0.1 * rng.standard_normal(4 * P)))
    q8("joint.enc.w", J, D)
    f32("joint.enc.b", _grid(0.1 * rng.standard_normal(J)))
    q8("joint.pred.w", J, P)
    f32("joint.pred.b", _grid(0.1 * rng.standard_normal(J)))
    rows = V + 1 + len(DURATIONS)
    pad = (-rows) % 16
    q8("joint.out.w", rows + pad, J, rows_used=rows)
    f32("joint.out.b", np.concatenate([_grid(0.1 * rng.standard_normal(rows)), np.zeros(pad)]))

    # Silero VAD's 16 kHz shapes, random weights, its exact window and twiddles
    n = 256
    f32("vad.stft.window", 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n))
    j = np.arange(n // 2)
    f32("vad.fft.cos", np.cos(2 * np.pi * j / n))
    f32("vad.fft.sin", np.sin(2 * np.pi * j / n))
    for i, (cin, cout) in enumerate([(129, 128), (128, 64), (64, 64), (64, 128)]):
        f32("vad.conv%d.w" % (i + 1), _grid(rng.standard_normal((3 * cin, cout)) / np.sqrt(3 * cin)))
        f32("vad.conv%d.b" % (i + 1), _grid(0.1 * rng.standard_normal(cout)))
    f32("vad.lstm.wx", _grid(rng.standard_normal((128, 512)) / np.sqrt(128)))
    f32("vad.lstm.wh", _grid(rng.standard_normal((128, 512)) / np.sqrt(128)))
    f32("vad.lstm.b", _grid(0.1 * rng.standard_normal(512)))
    f32("vad.head.w", _grid(rng.standard_normal(128) / np.sqrt(128)))
    # a head that leans to speech, so that a long recording has segments for
    # the CLI to cut into chunks
    f32("vad.head.b", np.array([2.0]))
    return out


def write(path):
    rng = np.random.default_rng(SEED)
    params = list(ARCH) + [("n_durations", len(DURATIONS))]
    params += [("dur%d" % i, d) for i, d in enumerate(DURATIONS)]
    pieces = [p for p, _ in PIECES]
    vocab = pkf.vocab_blob(pieces, [t for _, t in PIECES], convert.punctuation(pieces))
    ts = tensors(rng)
    meta = {"family": "tiny", "params": "%.0fK" % (sum(int(np.prod(d)) for _, _, d, _ in ts) / 1e3),
            "source": "pktito-tools tiny, seed %d" % SEED, "license": "MIT",
            "notice": "Synthetic weights for pktito's checks: seeded random, not a speech recognizer."}
    return pkf.write(path, "parakeet-tdt", "pktito-tiny", params, ts, vocab, meta)
