"""Parakeet's encoder as a Core ML program, for Apple's Neural Engine.

  encoder.py MODEL.nemo OUT.mlmodelc [--frames 3072] [--parts 2] [--cut 16] [--fp16] [--f32] [--scale 8]

The arithmetic is reference.py's encode, on canon.py's tensors (BatchNorm
and the feed-forwards' 0.5 already folded in, in float64), written as a
Core ML program with the Neural Engine's layout: activations [1][C][1][T],
channels then time, so that every matrix is a 1x1 convolution and each
normalization is over axis 1. The weights are rounded float64 to fp16, and
the matrices then to int8, a scale an output channel, as pktito's files
have them (--fp16 keeps them fp16; --f32 keeps every weight float32, for
checking the program itself on the CPU).

The Neural Engine runs fixed shapes. The program takes one length by
default, "f3072": 3072 feature frames (30.72 s), the length pktito's 30 s
chunks fit, a multiple of 512 so that the encoder's frames are of 64 (on
the M4 Pro 384 encoder frames take 71.1 ms where 375 take 75.4). --frames
can give it several, as Core ML's enumerated shapes: the one function then
takes any of them, its time dimension symbolic, and the position tables
for the input's length as three more inputs (build's docstring). Core
ML compiles a function for each of its lengths when it loads it, and the
system keeps what it compiled, for a build of the app: the int8 encoder of
one length took 9.7 s and 650 MB of the system's cache on the M4 Pro, 16.4
s on the M1 iPad, of six (512 to 3072 frames) 38.8 s and 1.83 GB, and 76.8
s, where a function of its own for each length keeps a copy of every
weight. A change of length costs about 40 ms on the M4 Pro and 100 on the
M1, whose 3072 frames are 10% slower in the six-length program: one length
is the default, short clips padded to it (CLIMB.md, 2026-10-31). A shorter
input is padded with zeros to the next length and its true length given by
a mask of ones; the program then computes exactly what the unpadded input
would give (up to fp16), because, as NeMo does with a batch's padding:
  - the attention's keys past the end are masked out of the softmax;
  - the convolution module zeroes the frames past the end before its
    depthwise convolution;
and, which NeMo leaves to the batch's longest, the subsampling zeroes the
frames past each stage's end before the next stage's convolution sees
them, as the unpadded input's zero padding would be.

fp16 holds 65504 at most. Inside a layer the residual stream reaches about
300 (the float64 probe on LibriSpeech's 1089-134686-0000), whose square,
90000, a layer norm's variance can't hold; the stream is carried divided by
--scale instead (a power of two: exact), each block's last matrix and each
layer's output norm scaled to match, and each norm's epsilon by its square,
which leaves every norm's output what it was. The last layer's output norm
is unscaled: the program gives the encoder's own output.

The attention's position term is not NeMo's rel_shift. NeMo scores query i
against key j with (q_i + v).P(i - j), P(r) = W_pos pe(r), and finds
P(i - j) by scoring every query against all 2T-1 relative positions and
shifting each row into place (a pad, two reshapes, two slices), which the
Neural Engine does slowly: 31 of the 91 ms of a 30 s function on the M4 Pro.
pe's sinusoids make the shift unnecessary: in each (sin, cos) pair of
frequency w, pe(i - j) is pe(j) turned through i w, so

  (q_i + v).W_pos pe(i - j) = qr_i.pe(j),  qr_i = R(i)^T W_pos^T (q_i + v)

where, for each pair (a, b) of u = W_pos^T (q_i + v), R(i)^T gives
(-a cos iw + b sin iw, a sin iw + b cos iw). W_pos^T a head at a time is a
grouped 1x1 convolution, once with each pair's two halves swapped; the
turn, products with tables of cos and sin; the scores, a product with the
fixed pe(j). The same numbers, in float32 to 1e-6 of reference.py's
(enccheck.py), and 75.4 ms where rel_shift took 91.

The turn is memory-bound, its products [H][D][T] a layer: about a third
of a layer's time on the A18, 30% on the M5 Ultra (CLIMB.md, 2026-11-02).
Most of its pairs hardly turn over the window: with 384 frames, the 335
of w (T - 1) < 16, fewer than 2.6 turns, whose sin and cos columns over
the window's frames lie, to 1e-4 of their largest singular value, in the
span of 11 functions phi_k. Each such pair's sin((i - j)w) and
cos((i - j)w) is s(i)c(j) - c(i)s(j) and c(i)c(j) + s(i)s(j), and with
s and c projected on phi

  (q_i + v).W_pos pe_low(i - j) = sum_kl phi_k(i) phi_l(j) G_kl.(q_i + v)

for vectors G_kl made from W_pos's low columns and the projections: the
turned form again, phi_k(i) where the turn has cos and sin, in pairs of k
as the turn has its pairs, and phi_l(j) where it has pe(j). So the
program folds them (fold_tables): the same ops over 414 rows a head where
there were 1024, 18-19% less a call on the A18, M4, M4 Pro and M5 Ultra,
at the same accuracy (CLIMB.md, 2026-11-02). The rows depend on the
window, so only a program of one length folds; one of several lengths
keeps the turn, its tables the caller's.
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol, types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pktito_tools import canon, nemo  # noqa: E402
from pktito_tools.convert import sha256_file  # noqa: E402


def sub_lengths(T):
    """Frames after each of the subsampling's three stride-2 stages."""
    out = []
    for _ in range(3):
        T = (T - 1) // 2 + 1
        out.append(T)
    return out


def turn_tables(T, w, a):
    """For positions 0..T-1, [D][T] each: pe(j) (sin in the even rows, cos
    in the odd) less its mean over j, and the turn's two tables, (-cos, cos)
    and (sin, sin) of i w for each pair; with the permutation that swaps
    each pair's halves.

    The mean taken out of pe(j) takes qr_i.mean(pe) out of every score of
    query i, the same for every key, which the softmax does not see; but
    it is most of the position term's size (layer 12 of LibriSpeech's
    8455-210777-0028: its largest |score| 42.5, then 10.4), and fp16 holds
    a smaller number nearer: the softmax's fp16 error falls 2 to 4 times in
    every layer, measured in float64 on that recording."""
    D = a["d_model"]
    ang = (
        np.repeat(w["pos.freq"].astype(np.float64), 2)[:, None] * np.arange(T)[None, :]
    )
    pe = np.where(np.arange(D)[:, None] % 2 == 0, np.sin(ang), np.cos(ang))
    pe = pe - pe.mean(axis=1, keepdims=True)
    cos = np.cos(ang) * np.where(np.arange(D)[:, None] % 2 == 0, -1.0, 1.0)
    swap = np.arange(D).reshape(-1, 2)[:, ::-1].reshape(-1)
    return pe, cos, np.sin(ang), swap


def fold_tables(T, w, a, cut, eps=1e-4):
    """turn_tables' for one length with the pairs of w (T - 1) < cut folded
    (the docstring's fold): pe, cos and sin [D'][T], turn_tables' rows for
    the pairs kept and then the folded rows, and rows(Wh), which gives one
    head's weights for u0 and u1 [D'][dk] from its rows Wh [dk][D] of W_pos.

    phi [T][L] is the constant, then the leading left singular vectors of
    the low pairs' sin and cos columns less their means, those of singular
    value over eps of the columns' largest. With the constant one of them,
    its rows, the same for every key, are left out, as the softmax doesn't
    see them: 414 rows a head where the columns' own singular vectors took
    420, at about their error in emulated fp16 (CLIMB.md, 2026-11-02). The
    tables carry sqrt(T) and G 1/T, so that each is about 1."""
    freq = w["pos.freq"].astype(np.float64)
    pe, cos, sin, swap = turn_tables(T, w, a)
    low = np.flatnonzero(freq * (T - 1) < cut)  # the last pairs: freq falls
    if not len(low):  # a cut under them all: the turn
        return pe, cos, sin, lambda Wh: (Wh.T, Wh.T[swap])
    kept = slice(0, 2 * low[0])
    n = np.arange(T, dtype=np.float64)[:, None]
    S, C = np.sin(n * freq[low]), np.cos(n * freq[low])  # s and c, [T][the low pairs]
    M = np.concatenate([S, C], 1)
    U, sv, _ = np.linalg.svd(M - M.mean(0, keepdims=True), full_matrices=False)
    L = 1 + int((sv > eps * np.linalg.svd(M, compute_uv=False)[0]).sum())
    phi = np.concatenate([np.full((T, 1), 1 / math.sqrt(T)), U[:, : L - 1]], 1)
    sg, kp = phi.T @ S, phi.T @ C
    # sin((i - j)w) = sum_kl Sp_kl phi_k(i) phi_l(j), cos((i - j)w) the same with Cp
    Sp = np.einsum("kp,lp->pkl", sg, kp) - np.einsum("kp,lp->pkl", kp, sg)
    Cp = np.einsum("kp,lp->pkl", kp, kp) + np.einsum("kp,lp->pkl", sg, sg)
    ph = phi.T * math.sqrt(T)
    # a row for each key function l but the constant and each pair (m, m + 1)
    # of query functions, the pair's second missing past the last
    km = [(l, m) for l in range(1, L) for m in range(0, L, 2)]
    pe = np.concatenate([pe[kept], [ph[l] for l, m in km]])
    cos = np.concatenate([cos[kept], [ph[m] for l, m in km]])
    sin = np.concatenate(
        [sin[kept], [ph[m + 1] if m + 1 < L else np.zeros(T) for l, m in km]]
    )

    def rows(Wh):
        G = (
            np.einsum("cp,pkl->klc", Wh[:, 2 * low], Sp)
            + np.einsum("cp,pkl->klc", Wh[:, 2 * low + 1], Cp)
        ) / T
        return (
            np.concatenate([Wh.T[kept], [G[m, l] for l, m in km]]),
            np.concatenate(
                [
                    Wh.T[swap][kept],
                    [G[m + 1, l] if m + 1 < L else np.zeros(len(Wh)) for l, m in km],
                ]
            ),
        )

    return pe, cos, sin, rows


def build(w, a, lengths, dt, scale, cut, lo=0, hi=None):
    """The program for each of lengths, in feature frames, layers lo to hi
    of the encoder: the first part takes the features [1][1][n_mels][T] and
    their mask [1][1][1][T], a later one the residual stream "x"
    [1][D][1][T/8] and the mask; the last part gives "enc", an earlier one
    "x". With one length its shapes are fixed; with several, T is a symbol,
    and each part takes turn_tables' three for its length too, "pe",
    "turn_cos" and "turn_sin" [1][1][D][T/8], which the caller makes (the
    M1's Neural Engine compiler fails a program that cuts them, made for the
    longest, to the input's length: iPadOS 26.5, CLIMB.md 2026-10-31). With
    one length and cut, the position term is folded (fold_tables). dt is
    np.float16 or np.float32."""
    hi = a["n_layers"] if hi is None else hi
    D, H, C, K, F = (
        a["d_model"],
        a["n_heads"],
        a["sub_channels"],
        a["conv_kernel"],
        a["n_mels"],
    )
    dk = D // H
    fixed = len(lengths) == 1
    T = max(lengths)
    T1, T2, T3 = sub_lengths(T)
    t3 = T3 if fixed else -1  # the time dimension of a reshape
    s = 1.0 / scale
    eps = a["ln_eps"] * s * s
    mtype = types.fp16 if dt == np.float16 else types.fp32

    def c(x):
        return np.asarray(x, np.float64).astype(dt)

    def lin(x, W, name, bias=None):
        """W [out][in] applied to x [1][in][1][T]: a 1x1 convolution"""
        return mb.conv(
            x=x,
            weight=c(W[:, :, None, None]),
            bias=None if bias is None else c(bias),
            name=name,
        )

    def norm(x, q, last=False):
        g, b = w[q + ".w"], w[q + ".b"]
        if q.endswith("out.norm") and not last:
            g, b = g * s, b * s
        return mb.layer_norm(
            x=x,
            axes=[1],
            gamma=c(g),
            beta=c(b),
            epsilon=dt(eps) if dt == np.float32 else np.float16(eps),
        )

    def kernel3(wk):
        """reference's [O][9] taps, 3 * (time offset) + (frequency offset),
        as [O][1][3 frequency][3 time] for the [F][T] layout"""
        return c(wk.reshape(-1, 3, 3).transpose(0, 2, 1)[:, None])

    def sigmoid_div(x, y):
        """y sigmoid(x) as y / (1 + exp(-x)): the Neural Engine's own
        sigmoid and silu are approximations, 2.05% from float64 after layer
        10's depthwise convolution on LibriSpeech's 8455-210777-0028 where
        this is 0.25% (the module's output 2.29%, then 0.26%)"""
        return mb.real_div(x=y, y=mb.add(x=mb.exp(x=mb.mul(x=x, y=dt(-1))), y=dt(1)))

    def every_other(m, n):
        if fixed:
            return mb.slice_by_index(
                x=m, begin=[0, 0, 0, 0], end=[1, 1, 1, n], stride=[1, 1, 1, 2]
            )
        return mb.slice_by_index(
            x=m,
            begin=[0, 0, 0, 0],
            end=[1, 1, 1, 1],
            stride=[1, 1, 1, 2],
            end_mask=[True, True, True, True],
        )

    if fixed and cut:
        pe, cos, sin, rows = fold_tables(T3, w, a, cut)
    else:
        pe, cos, sin, swap = turn_tables(T3, w, a)

        def rows(Wh):
            return Wh.T, Wh.T[swap]

    Dp = len(pe)  # the term's rows a head, D or D'
    pe, cos, sin = (
        c(pe[None, None]),
        c(cos[None, None]),
        c(sin[None, None]),
    )  # [1][1][D'][T]

    def masks(mask):
        """each stage's mask: frame t of a stride-2 stage is frame 2t of the
        one before, so its mask is the one before's every other entry; and
        the attention's key bias, 0, or -10000 past the end"""
        m1 = every_other(mask, T)
        m2 = every_other(m1, T1)
        m3 = every_other(m2, T2)
        return m1, m2, m3, mb.mul(x=mb.sub(x=m3, y=dt(1)), y=dt(10000))

    def subsample(mel, mask, m1, m2):
        # subsampling, [1][C][F][T] at each stage
        x = mb.mul(x=mel, y=mask)
        x = mb.conv(
            x=x,
            weight=kernel3(w["sub.conv0.w"]),
            bias=c(w["sub.conv0.b"]),
            strides=[2, 2],
            pad_type="custom",
            pad=[1, 1, 1, 1],
        )
        x = mb.mul(x=mb.relu(x=x), y=m1)
        x = mb.conv(
            x=x,
            weight=kernel3(w["sub.dw1.w"]),
            bias=c(w["sub.dw1.b"]),
            strides=[2, 2],
            pad_type="custom",
            pad=[1, 1, 1, 1],
            groups=C,
        )
        x = mb.conv(
            x=x, weight=c(w["sub.pw1.w"][:, :, None, None]), bias=c(w["sub.pw1.b"])
        )
        x = mb.mul(x=mb.relu(x=x), y=m2)
        x = mb.conv(
            x=x,
            weight=kernel3(w["sub.dw2.w"]),
            bias=c(w["sub.dw2.b"]),
            strides=[2, 2],
            pad_type="custom",
            pad=[1, 1, 1, 1],
            groups=C,
        )
        x = mb.conv(
            x=x, weight=c(w["sub.pw2.w"][:, :, None, None]), bias=c(w["sub.pw2.b"])
        )
        x = mb.relu(x=x)
        F3 = x.shape[2]
        x = mb.reshape(x=x, shape=[1, C * F3, 1, t3])  # channel-major, as reference's
        x = lin(x, w["sub.out.w"] * s, "sub_out", w["sub.out.b"] * s)
        return x

    def layer(x, i, m3, kbias, pe, cos, sin):
        q = "l%d." % i
        # feed-forward 1
        h = lin(norm(x, q + "ff1.norm"), w[q + "ff1.up"], q + "ff1_up")
        h = lin(mb.silu(x=h), w[q + "ff1.down"] * s, q + "ff1_down")
        x = mb.add(x=x, y=h)
        # attention with relative positions
        xn = norm(x, q + "att.norm")
        Q = lin(xn, w[q + "att.q"], q + "q")
        Kt = mb.reshape(x=lin(xn, w[q + "att.k"], q + "k"), shape=[1, H, dk, t3])
        V = mb.reshape(x=lin(xn, w[q + "att.v"], q + "v"), shape=[1, H, dk, t3])
        qu = mb.add(
            x=mb.reshape(x=Q, shape=[1, H, dk, t3]),
            y=c(w[q + "att.bias_u"].reshape(1, H, dk, 1)),
        )
        qv = mb.add(x=Q, y=c(w[q + "att.bias_v"][None, :, None, None]))
        ac = mb.matmul(x=qu, y=Kt, transpose_x=True)  # [1][H][T][T]
        # the position term turned, not shifted (the docstring's qr), or folded
        W0, W1 = zip(
            *(rows(w[q + "att.pos"][h * dk : (h + 1) * dk]) for h in range(H))
        )  # [D'][dk] a head
        u0 = mb.conv(x=qv, weight=c(np.concatenate(W0)[:, :, None, None]), groups=H)
        u1 = mb.conv(x=qv, weight=c(np.concatenate(W1)[:, :, None, None]), groups=H)
        qr = mb.add(
            x=mb.mul(x=mb.reshape(x=u0, shape=[1, H, Dp, t3]), y=cos),
            y=mb.mul(x=mb.reshape(x=u1, shape=[1, H, Dp, t3]), y=sin),
        )  # [1][H][D'][T]
        bd = mb.matmul(x=qr, y=pe, transpose_x=True)  # [1][H][T][T]
        sc = mb.add(x=mb.mul(x=mb.add(x=ac, y=bd), y=dt(1 / math.sqrt(dk))), y=kbias)
        p = mb.softmax(x=sc, axis=-1)
        o = mb.matmul(x=V, y=p, transpose_y=True)  # [1][H][dk][T]
        o = lin(mb.reshape(x=o, shape=[1, D, 1, t3]), w[q + "att.o"] * s, q + "o")
        x = mb.add(x=x, y=o)
        # convolution module
        h = lin(norm(x, q + "conv.norm"), w[q + "conv.pw1"], q + "pw1")
        ha, hb = mb.split(x=h, num_splits=2, axis=1)
        h = mb.mul(x=sigmoid_div(hb, ha), y=m3)
        h = mb.conv(
            x=h,
            weight=c(w[q + "conv.dw"][:, None, None, :]),
            bias=c(w[q + "conv.dw_b"]),
            pad_type="custom",
            pad=[0, 0, K // 2, K // 2],
            groups=D,
        )
        h = lin(sigmoid_div(h, h), w[q + "conv.pw2"] * s, q + "pw2")
        x = mb.add(x=x, y=h)
        # feed-forward 2
        h = lin(norm(x, q + "ff2.norm"), w[q + "ff2.up"], q + "ff2_up")
        h = lin(mb.silu(x=h), w[q + "ff2.down"] * s, q + "ff2_down")
        x = mb.add(x=x, y=h)
        x = norm(x, q + "out.norm", last=i == a["n_layers"] - 1)
        return x

    def body(x, mask, tables):
        """the part from its first input, the features or the stream"""
        if lo == 0:
            m1, m2, _, _ = masks(mask)
            x = subsample(x, mask, m1, m2)
        _, _, m3, kbias = masks(mask)
        for i in range(max(lo, 0), hi):
            x = layer(x, i, m3, kbias, *tables)
        return mb.identity(x=x, name="enc" if hi == a["n_layers"] else "x")

    Tin, T3in = (T, T3) if fixed else (get_new_symbol(), get_new_symbol())
    specs = [
        mb.TensorSpec(
            shape=(1, 1, F, Tin) if lo == 0 else (1, D, 1, T3in), dtype=mtype
        ),
        mb.TensorSpec(shape=(1, 1, 1, Tin), dtype=mtype),
    ]
    if fixed:
        if lo == 0:

            @mb.program(input_specs=specs, opset_version=ct.target.iOS18)
            def prog(mel, mask):
                return body(mel, mask, (pe, cos, sin))
        else:

            @mb.program(input_specs=specs, opset_version=ct.target.iOS18)
            def prog(x, mask):
                return body(x, mask, (pe, cos, sin))
    else:
        specs += [mb.TensorSpec(shape=(1, 1, D, T3in), dtype=mtype) for _ in range(3)]
        if lo == 0:

            @mb.program(input_specs=specs, opset_version=ct.target.iOS18)
            def prog(mel, mask, pe, turn_cos, turn_sin):
                return body(mel, mask, (pe, turn_cos, turn_sin))
        else:

            @mb.program(input_specs=specs, opset_version=ct.target.iOS18)
            def prog(x, mask, pe, turn_cos, turn_sin):
                return body(x, mask, (pe, turn_cos, turn_sin))

    return prog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nemo")
    ap.add_argument("out")
    ap.add_argument(
        "--frames",
        default="3072",
        help="the lengths each function takes, in feature frames, comma-separated",
    )
    # two by default: the M1's Neural Engine compiles each 12-layer half in
    # 10 s where the whole fp16 encoder failed, and the M4 Pro runs the two
    # in the time of one (CLIMB.md, 2026-10-31)
    ap.add_argument(
        "--parts",
        type=int,
        default=2,
        help="the encoder as this many functions of consecutive layers, run one after another",
    )
    # 16: 18-19% less a call on the A18, M4, M4 Pro and M5 Ultra, against
    # float32 as the turn: v2 over LibriSpeech's 200 1.51% and 2 changed,
    # dlogp -0.0239, both; v3 over FLEURS 4.47% for 4.48%, 27 changed both,
    # -0.0183 for -0.0170. 8 folds fewer pairs, 1% slower (CLIMB.md, 2026-11-02)
    ap.add_argument(
        "--cut",
        type=float,
        default=16,
        help="with one length, fold the position pairs of w (T - 1) below this; 0 keeps the turn "
        "alone, the program enccheck.py holds to 1e-6 of reference.py with --f32",
    )
    # int8 matrices by default: half the size (610 MB to 1.2 GB), faster (M1
    # iPad 253x to 232x, M4 Pro 335x to 302x), and within 2 errors of
    # pktito's own engine on each set they were compared over, 3 of fp16's,
    # though fp16 is ten times nearer float32 (dlogp -0.0024 to -0.024 on
    # LibriSpeech's 200; CLIMB.md, 2026-10-31)
    ap.add_argument(
        "--fp16",
        action="store_true",
        help="the matrices fp16 too, rather than int8 a scale an output channel",
    )
    ap.add_argument(
        "--f32", action="store_true", help="float32 weights and arithmetic, for the CPU"
    )
    ap.add_argument(
        "--scale",
        type=float,
        default=8,
        help="the residual stream carried divided by this",
    )
    args = ap.parse_args()

    nm = nemo.Nemo(args.nemo)
    a = canon.arch(nm.config)
    w = dict(canon.tensors(nm.state, a))
    dt = np.float32 if args.f32 else np.float16
    if not args.out.endswith(".mlmodelc"):
        sys.exit("the output is a compiled model, NAME.mlmodelc")
    lengths = sorted(int(x) for x in args.frames.split(","))
    T, F, D = lengths[-1], a["n_mels"], a["d_model"]
    # the checkpoint, named as the converter names it in the .pkf's metadata:
    # pktito takes this encoder only for a model file made from the same one
    source = {
        "source_sha256": sha256_file(args.nemo),
        "source_file": os.path.basename(args.nemo),
    }
    # each function a package, the packages one, compiled: only the last kept,
    # since each package holds every weight (1.2 GB in fp16)
    with tempfile.TemporaryDirectory() as tmp:
        d = ct.utils.MultiFunctionDescriptor()
        L, n = a["n_layers"], args.parts
        for k in range(n):
            prog = build(
                w,
                a,
                lengths,
                dt,
                args.scale,
                args.cut,
                lo=k * L // n,
                hi=(k + 1) * L // n,
            )
            inputs = None
            if len(lengths) > 1:
                # the inputs' shapes, i-th with i-th; the longest the default
                def enum(shape):
                    return ct.EnumeratedShapes(
                        shapes=[shape(t) for t in lengths], default=shape(T)
                    )

                first = (
                    ct.TensorType(name="mel", shape=enum(lambda t: (1, 1, F, t)))
                    if k == 0
                    else ct.TensorType(
                        name="x", shape=enum(lambda t: (1, D, 1, sub_lengths(t)[2]))
                    )
                )
                inputs = [
                    first,
                    ct.TensorType(name="mask", shape=enum(lambda t: (1, 1, 1, t))),
                ] + [
                    ct.TensorType(
                        name=n, shape=enum(lambda t: (1, 1, D, sub_lengths(t)[2]))
                    )
                    for n in ("pe", "turn_cos", "turn_sin")
                ]
            m = ct.convert(
                prog,
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.iOS18,
                compute_precision=ct.precision.FLOAT32
                if args.f32
                else ct.precision.FLOAT16,
                inputs=inputs,
            )
            if not (args.fp16 or args.f32):
                # the matrices, as pktito's files have them: the 1x1
                # convolutions, of 65536 weights and more; the subsampling's
                # 3x3 taps and the depthwise convolutions' (9216, BatchNorm
                # folded in) stay fp16, as pktito keeps them float. The
                # position and turn tables, multiplied in, are not weights.
                # (A scale a block of 32 inputs is nearer fp16, dlogp -0.019,
                # but half the speed on the Neural Engine: CLIMB.md.)
                q8 = ct.optimize.coreml.OpLinearQuantizerConfig(
                    mode="linear_symmetric",
                    dtype="int8",
                    granularity="per_channel",
                    weight_threshold=16384,
                )
                m = ct.optimize.coreml.linear_quantize_weights(
                    m,
                    config=ct.optimize.coreml.OptimizationConfig(
                        op_type_configs={"conv": q8}
                    ),
                )
            name = "f%d" % T if n == 1 else "f%dp%d" % (T, k)
            p = os.path.join(tmp, name + ".mlpackage")
            m.save(p)
            d.add_function(p, src_function_name="main", target_function_name=name)
            print(
                "%s: %s s, layers %d to %d"
                % (
                    name,
                    ", ".join("%.2f" % (t / 100) for t in lengths),
                    k * L // n,
                    (k + 1) * L // n - 1,
                ),
                flush=True,
            )
        d.default_function_name = name
        pkg = os.path.join(
            tmp, os.path.basename(args.out)[: -len(".mlmodelc")] + ".mlpackage"
        )
        ct.utils.save_multifunction(d, pkg)
        # the model's metadata, which save_multifunction keeps from none of
        # its functions: written into the package's description
        spec_path = os.path.join(pkg, "Data", "com.apple.CoreML", "model.mlmodel")
        spec = ct.proto.Model_pb2.Model()
        with open(spec_path, "rb") as f:
            spec.ParseFromString(f.read())
        spec.description.metadata.userDefined.update(source)
        with open(spec_path, "wb") as f:
            f.write(spec.SerializeToString())
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        subprocess.run(
            [
                "xcrun",
                "coremlcompiler",
                "compile",
                pkg,
                os.path.dirname(os.path.abspath(args.out)),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    print("%d functions in %s" % (n, args.out))


if __name__ == "__main__":
    main()
