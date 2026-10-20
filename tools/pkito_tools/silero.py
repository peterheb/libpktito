"""Silero VAD's 16 kHz weights out of silero_vad.onnx, in vadito's layout.

This is vadito's converter (tools/vadito_tools/convert.py there) for the
16 kHz branch alone, under the names core/vad.c binds: "vad." and vadito's
own (stft.window, fft.cos, conv1.w, ..., head.b). What a Silero VAD file
holds, per sample rate:

    stft.forward_basis_buffer          [N+2, 1, N]   N = 256 at 16 kHz
    encoder.{0..3}.reparam_conv.weight [O, C, 3]     129 -> 128 -> 64 -> 64 -> 128
    encoder.{0..3}.reparam_conv.bias   [O]
    decoder.rnn.weight_ih, weight_hh   [512, 128]    LSTMCell, gates i f g o
    decoder.rnn.bias_ih, bias_hh       [512]
    decoder.decoder.2.weight, bias     [1, 128, 1], [1]

The basis is checked to be a periodic Hann window times an N-point DFT, to
fp32 rounding, since the engine computes it as an FFT; the twiddles are
computed here in float64. Every matrix is k-major, [inputs][outputs], in
panels of 16 outputs, for pk_k_gemm_f32; a convolution's input index is
tap * C + channel. The LSTM's two biases are summed in float64.
"""
import hashlib

import numpy as np

from . import onnxpb

HIDDEN = 128
PANEL = 16

SUFFIXES = {
    "stft.forward_basis_buffer": "basis",
    "decoder.rnn.weight_ih": "wih",
    "decoder.rnn.weight_hh": "whh",
    "decoder.rnn.bias_ih": "bih",
    "decoder.rnn.bias_hh": "bhh",
    "decoder.decoder.2.weight": "head.w",
    "decoder.decoder.2.bias": "head.b",
}
for _i in range(4):
    SUFFIXES["encoder.%d.reparam_conv.weight" % _i] = "conv%d.w" % (_i + 1)
    SUFFIXES["encoder.%d.reparam_conv.bias" % _i] = "conv%d.b" % (_i + 1)

CONV = [(None, 128), (128, 64), (64, 64), (64, 128)]


def branch16k(tensors):
    """{role: float64 array} for the 16 kHz branch of an ONNX file's tensors."""
    groups = {}
    for name, a in tensors:
        for suffix, role in SUFFIXES.items():
            if name == suffix or name.endswith("." + suffix) or name.endswith("__" + suffix):
                g = groups.setdefault(name[:len(name) - len(suffix)], {})
                if role in g:
                    raise ValueError("%s appears twice" % suffix)
                g[role] = a.astype(np.float64)
    for g in groups.values():
        if set(SUFFIXES.values()) <= set(g) and g["basis"].shape == (258, 1, 256):
            return g
    raise ValueError("no 16 kHz Silero VAD branch in the file")


def _panels(m):
    n, rows = m.shape
    if rows % PANEL:
        raise ValueError("%d outputs is not a multiple of %d" % (rows, PANEL))
    return m.reshape(n, rows // PANEL, PANEL).transpose(1, 0, 2).reshape(n, rows)


def tensors(path):
    """[(name, float32 array)] for the .pkf, and the file's sha256."""
    g = branch16k(onnxpb.tensors(path))
    n, bins = 256, 129
    want = {"wih": (512, HIDDEN), "whh": (512, HIDDEN), "bih": (512,), "bhh": (512,),
            "head.w": (1, HIDDEN, 1), "head.b": (1,)}
    for i, (cin, cout) in enumerate(CONV):
        want["conv%d.w" % (i + 1)] = (cout, cin or bins, 3)
        want["conv%d.b" % (i + 1)] = (cout,)
    for role, shape in want.items():
        if g[role].shape != shape:
            raise ValueError("VAD %s: shape %s, expected %s" % (role, g[role].shape, shape))
    b = g["basis"][:, 0, :]
    window = b[0]
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
    if np.abs(window - hann).max() > 1e-7:
        raise ValueError("the VAD's STFT window is not a periodic Hann window")
    k = np.arange(n // 2 + 1)[:, None]
    ang = 2 * np.pi * k * np.arange(n)[None, :] / n
    if np.abs(b - np.vstack([window * np.cos(ang), -window * np.sin(ang)])).max() > 2e-7:
        raise ValueError("the VAD's STFT basis is not the windowed DFT pkito computes as an FFT")
    j = np.arange(n // 2)
    out = [("vad.stft.window", window), ("vad.fft.cos", np.cos(2 * np.pi * j / n)),
           ("vad.fft.sin", np.sin(2 * np.pi * j / n))]
    for i in range(1, 5):
        w = g["conv%d.w" % i]
        o, c, taps = w.shape
        out.append(("vad.conv%d.w" % i, _panels(w.transpose(2, 1, 0).reshape(taps * c, o))))
        out.append(("vad.conv%d.b" % i, g["conv%d.b" % i]))
    out += [("vad.lstm.wx", _panels(g["wih"].T)), ("vad.lstm.wh", _panels(g["whh"].T)),
            ("vad.lstm.b", g["bih"] + g["bhh"]), ("vad.head.w", g["head.w"].reshape(HIDDEN)),
            ("vad.head.b", g["head.b"].reshape(1))]
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return [(name, a.astype(np.float32)) for name, a in out], digest
