"""What each way of rounding Parakeet's weights and activations costs: the
experiment that chose pktito's numerics.

  quantcheck.py MODEL.nemo CORPUS_DIR... [--configs a,b,...] [--json OUT]

Each configuration runs the model (tools/pktito_tools/reference.py, here in
float32, which is what NeMo computes in) over every recording of the corpus
directories (tools/corpus.py's: WAVs and a refs.tsv) with some of its
matrices, and the activations entering them, rounded one way. Against the
unrounded float32 run it reports:

  WER         against the references (wer.py's normalization), errors / words
  changed     recordings whose transcript is not float32's
  dlogp       how much less likely the rounded model finds float32's own
              decisions: the log-probability of float32's every token and
              duration choice, replayed through the rounded model, less
              float32's, per decision. A paired number, like pinfer's paired
              nll, and much finer than WER: it moves before a word does.
  t           dlogp's paired t over recordings
  enc err     |enc - enc_f32| / |enc_f32|, the encoder output's relative error

The roundings (q is the integer kept, d its scale):

  w8row   weights int8, one scale an output row: d = max|row| / 127
  w8b32   weights int8, one fp16 scale a block of 32 inputs (ggml's q8_0)
  w4b32   weights 4-bit, one fp16 scale a block of 32 (ggml's q4_0: q in
          -8..7, d = the block's largest-magnitude value / -8)
  w4k     weights 4-bit in super-blocks of 256 (ggml's q4_K: 6-bit scales
          and mins for each 32)
  f16     weights fp16
  a8row   activations int8, one scale a row (a frame): d = max|x| / 127
  a8b32   activations int8, one scale a block of 32 (and a8b64, a8b128)

and a configuration is rules separated by ";", each a weight rounding,
optionally "+" an activation rounding, optionally "@" the classes it
applies to, "/"-separated (default enc). A later rule overrides an earlier
one for the classes it names:

  enc     every encoder matrix the conformer layers and the subsampling's
          projection multiply by; it is the union of
            qkv     att.q, att.k, att.v (their input: the attention's norm)
            pos     att.pos (its input: the sinusoidal position table)
            o       att.o (its input: the attention's output)
            ffup    ff1.up, ff2.up (their input: the feed-forwards' norms)
            ffdown  ff1.down, ff2.down (their input: SiLU's output)
            pw1     conv.pw1 (its input: the convolution module's norm)
            pw2     conv.pw2 (its input: the depthwise convolution's SiLU)
            subout  sub.out (its input: the subsampling's last ReLU)
  sub     the subsampling's two 256x256 pointwise convolutions
  dec     the prediction network's LSTMs
  joint   the joint's three matrices

A rule "att8" makes the attention's own three products int8 as well
(attention_int8 says how).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pktito_tools import canon, nemo, reference, spm, wavio, wer  # noqa: E402


def read_wav(path):
    a, rate = wavio.read(path)
    assert rate == 16000, path
    return a.astype(np.float32)


FINE = {"att.q": "qkv", "att.k": "qkv", "att.v": "qkv", "att.pos": "pos", "att.o": "o",
        "ff1.up": "ffup", "ff2.up": "ffup", "ff1.down": "ffdown", "ff2.down": "ffdown",
        "conv.pw1": "pw1", "conv.pw2": "pw2"}
ENC = set(FINE.values()) | {"subout"}


def classes(name):
    """The class a matrix belongs to, finest first: a set of the names a
    rule can target."""
    if name in ("sub.pw1.w", "sub.pw2.w"):
        return {"sub"}
    if name == "sub.out.w":
        return {"subout", "enc"}
    if name.startswith("l") and name.split(".", 1)[1] in FINE:
        return {FINE[name.split(".", 1)[1]], "enc"}
    if name.startswith("dec.l") and name.endswith((".wx", ".wh")):
        return {"dec"}
    if name in ("joint.enc.w", "joint.pred.w", "joint.out.w"):
        return {"joint"}
    return set()


def _f16(x):
    return x.astype(np.float16).astype(np.float32)


def q_w8row(W):
    d = np.abs(W).max(1, keepdims=True) / 127
    d[d == 0] = 1
    return np.clip(np.rint(W / d), -127, 127) * d


def _blocks(W, n):
    r, c = W.shape
    assert c % n == 0
    return W.reshape(r, c // n, n)


def q_w8b32(W):
    B = _blocks(W, 32)
    d = _f16(np.abs(B).max(2, keepdims=True) / 127)
    q = np.clip(np.rint(B / np.where(d == 0, 1, d)), -127, 127)
    return (q * d).reshape(W.shape)


def q_w4b32(W):
    B = _blocks(W, 32)
    i = np.abs(B).argmax(2)[..., None]
    mx = np.take_along_axis(B, i, 2)
    d = _f16(mx / -8)
    q = np.clip(np.rint(B / np.where(d == 0, 1, d)), -8, 7)
    return (q * d).reshape(W.shape)


def q_w4k(W):
    """ggml's q4_K, by its reference quantizer's simple form: each 32 gets
    a scale and a min fitted by least squares over its range, quantized to
    6 bits against fp16 super-scales."""
    B = _blocks(W, 32)                                   # [r][nb][32]
    lo = np.minimum(B.min(2), 0)
    hi = B.max(2)
    sc = (hi - lo) / 15
    sc[sc == 0] = 1
    mn = -lo
    S = sc.reshape(W.shape[0], -1, 8)                    # super-blocks of 8 x 32
    M = mn.reshape(W.shape[0], -1, 8)
    ds = _f16(S.max(2, keepdims=True) / 63)
    dm = _f16(M.max(2, keepdims=True) / 63)
    s6 = np.clip(np.rint(S / np.where(ds == 0, 1, ds)), 0, 63) * ds
    m6 = np.clip(np.rint(M / np.where(dm == 0, 1, dm)), 0, 63) * dm
    s6 = s6.reshape(sc.shape)[..., None]
    m6 = m6.reshape(mn.shape)[..., None]
    q = np.clip(np.rint((B + m6) / np.where(s6 == 0, 1, s6)), 0, 15)
    return (q * s6 - m6).reshape(W.shape)


WQ = {"w8row": q_w8row, "w8b32": q_w8b32, "w4b32": q_w4b32, "w4k": q_w4k, "f16": _f16}


def a8row(x):
    d = np.abs(x).max(-1, keepdims=True) / 127
    d[d == 0] = 1
    return np.rint(x / d) * d


def a8b(n):
    def q(x):
        s = x.shape
        B = x.reshape(s[:-1] + (s[-1] // n, n))
        d = np.abs(B).max(-1, keepdims=True) / 127
        d[d == 0] = 1
        return (np.rint(B / d) * d).reshape(s)
    return q


AQ = {"a8row": a8row, "a8b32": a8b(32), "a8b64": a8b(64), "a8b128": a8b(128)}


def _rows8(x):
    """int8 a row at a time (the last axis), as a q8 'weight' would be"""
    d = np.abs(x).max(-1, keepdims=True) / 127
    d[d == 0] = 1
    return np.rint(x / d) * d


def attention_int8(x, pe, w, q, a):
    """The attention's three products in int8, as pk_k_gemm_q8 would take
    them: keys, positions and the values (dims as rows, over the keys)
    rounded a row at a time like weights; the queries and the attention
    probabilities in blocks of 32 like activations."""
    T, D = x.shape
    H = a["n_heads"]
    dk = D // H
    mm = reference.mm
    Q = mm(x, w, q + "att.q").reshape(T, H, dk)
    K = mm(x, w, q + "att.k").reshape(T, H, dk)
    V = mm(x, w, q + "att.v").reshape(T, H, dk)
    P = mm(pe, w, q + "att.pos").reshape(2 * T - 1, H, dk)
    qu = a8b(32)((Q + w[q + "att.bias_u"].reshape(H, dk)).transpose(1, 0, 2))
    qv = a8b(32)((Q + w[q + "att.bias_v"].reshape(H, dk)).transpose(1, 0, 2))
    ac = qu @ _rows8(K.transpose(1, 0, 2)).transpose(0, 2, 1)
    bd = qv @ _rows8(P.transpose(1, 0, 2)).transpose(0, 2, 1)
    i, j = np.arange(T)[:, None], np.arange(T)[None, :]
    bd = bd[:, i, T - 1 - i + j]
    sc = (ac + bd) / np.sqrt(dk)
    sc = np.exp(sc - sc.max(-1, keepdims=True))
    sc /= sc.sum(-1, keepdims=True)
    Tp = (T + 31) // 32 * 32
    scp = np.zeros(sc.shape[:-1] + (Tp,), np.float32)
    scp[..., :T] = sc
    scq = a8b(32)(scp)[..., :T]
    Vt = _rows8(V.transpose(1, 2, 0))                 # [H][dk][T]: a dim's values over the keys
    o = (scq @ Vt.transpose(0, 2, 1)).transpose(1, 0, 2).reshape(T, D)
    return mm(o.astype(np.float32), w, q + "att.o")


def parse(cfg):
    """[(weight rounding, activation rounding, classes)], in order."""
    if cfg == "f32":
        return []
    rules = []
    for rule in cfg.split(";"):
        if rule == "att8":
            continue
        on = {"enc"}
        if "@" in rule:
            rule, cl = rule.split("@")
            on = set(cl.split("/"))
        wq, _, aq = rule.partition("+")
        if wq not in WQ or (aq and aq not in AQ) or not on <= ENC | {"enc", "sub", "dec", "joint"}:
            raise SystemExit("unknown configuration %r" % cfg)
        rules.append((WQ[wq], AQ.get(aq), on))
    return rules


def setup(w32, cfg):
    """The weights under cfg, and the matmul hook that goes with them."""
    rules = parse(cfg)
    wd = dict(w32)
    act = {}
    for name, W in w32.items():
        hit = None
        for r in rules:
            if classes(name) & r[2]:
                hit = r
        if hit:
            wd[name] = hit[0](W.astype(np.float64)).astype(np.float32)
            act[name] = hit[1]

    def mm(x, w, name):
        x = np.asarray(x, np.float32)
        if act.get(name):
            x = act[name](x).astype(np.float32)
        return x @ w[name].T
    return wd, mm


def path_logp(enc, w, a, trace):
    """The log-probability, under the model in w, of every decision in a
    trace another run made: a token and a duration per joint evaluation."""
    dec = reference.Decoder(w, a)
    V = a["vocab"]
    fproj = reference.mm(enc, w, "joint.enc.w") + w["joint.enc.b"]
    state = dec.start()
    g, state = dec.step(V, state)
    total = 0.0
    for t, k, di in trace:
        z = dec.joint(fproj[t], g).astype(np.float64)
        zt, zd = z[:V + 1], z[V + 1:]
        total += zt[k] - zt.max() - math.log(np.exp(zt - zt.max()).sum())
        total += zd[di] - zd.max() - math.log(np.exp(zd - zd.max()).sum())
        if k != V:
            g, state = dec.step(k, state)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nemo")
    ap.add_argument("corpus", nargs="+")
    ap.add_argument("--configs", default="w8row+a8row,w8b32+a8b32,w4b32+a8b32")
    ap.add_argument("--limit", type=int, default=0, help="recordings per corpus directory")
    ap.add_argument("--json", help="write every transcript here")
    args = ap.parse_args()

    nm = nemo.Nemo(args.nemo)
    a = canon.arch(nm.config)
    w32 = {k: v.astype(np.float32) for k, v in canon.tensors(nm.state, a)}
    sp = spm.Model(nm.tokenizer_model())
    files = []
    for d in args.corpus:
        with open(os.path.join(d, "refs.tsv"), encoding="utf-8") as f:
            rows = [line.rstrip("\n").split("\t", 1) for line in f if line.strip()]
        if args.limit:
            rows = rows[:args.limit]
        files += [(os.path.join(d, n), t) for n, t in rows]
    print("%s: %d recordings, %.1f minutes" % (os.path.basename(args.nemo), len(files),
          sum(len(read_wav(p)) for p, _ in files) / 16000 / 60), flush=True)

    float_attention = reference.attention

    def run(cfg, base=None):
        wd, mm = setup(w32, cfg)
        reference.mm = mm
        reference.attention = attention_int8 if "att8" in cfg.split(";") else float_attention
        out = []
        t0 = time.time()
        for i, (path, ref) in enumerate(files):
            audio = read_wav(path)
            feat = reference.features(audio, wd, a).astype(np.float32)
            enc = reference.encode(feat, wd, a)
            trace = []
            toks = reference.greedy(enc, wd, a, trace)
            r = {"file": path, "ref": ref, "text": sp.decode([k for k, _, _ in toks])}
            if base is None:
                r["trace"], r["enc"] = trace, enc
                r["logp"] = path_logp(enc, wd, a, trace)
            else:
                b = base[i]
                r["logp"] = path_logp(enc, wd, a, b["trace"])
                r["dlogp"] = (r["logp"] - b["logp"]) / len(b["trace"])
                r["encerr"] = float(np.linalg.norm(enc - b["enc"]) / np.linalg.norm(b["enc"]))
            out.append(r)
        return out, time.time() - t0

    base, dt = run("f32")
    e, n = wer.wer([(r["ref"], r["text"]) for r in base])
    print("%-22s WER %5.2f%% (%d/%d)  %6.0fs" % ("f32", 100 * e / n, e, n, dt), flush=True)
    results = {"f32": [{"file": r["file"], "text": r["text"]} for r in base]}
    for cfg in [c for c in args.configs.split(",") if c]:
        res, dt = run(cfg, base)
        e, n = wer.wer([(r["ref"], r["text"]) for r in res])
        changed = sum(r["text"] != b["text"] for r, b in zip(res, base))
        d = np.array([r["dlogp"] for r in res])
        t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 and d.std() > 0 else 0
        print("%-22s WER %5.2f%% (%d/%d)  changed %3d  dlogp %+.5f  t %6.1f  enc err %.4f  %6.0fs" % (
            cfg, 100 * e / n, e, n, changed, d.mean(), t, np.mean([r["encerr"] for r in res]), dt),
            flush=True)
        results[cfg] = [{"file": r["file"], "text": r["text"], "dlogp": r["dlogp"]} for r in res]
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=0)


if __name__ == "__main__":
    main()
