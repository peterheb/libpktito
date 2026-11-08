"""The Core ML encoder against float32, as tools/quantcheck.py measures a
rounding: reference.py in float32 is the base, and the Core ML encoder's
output goes through the same float32 decoder. Over every recording of the
corpus directories:

  WER, changed  as quantcheck's
  dlogp, t      float32's decisions replayed through the Core ML encoder's
                output, per decision, and its paired t
  enc err       |enc - enc_f32| / |enc_f32|, the mean over recordings

  anecheck.py MODEL.nemo ENC.mlmodelc CORPUS_DIR... [--units ne] [--base F32.pkl] [--json OUT]

--base keeps float32's run (its transcripts, decisions and encoder
outputs) in a file, made if it isn't there, for the next variant.
"""

import argparse
import json
import math
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from pktito_tools import canon, nemo, reference, spm, wer  # noqa: E402
from quantcheck import path_logp, read_wav  # noqa: E402
from ane.enccheck import UNITS, load, run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nemo")
    ap.add_argument("enc")
    ap.add_argument("corpus", nargs="+")
    ap.add_argument("--units", default="ne", choices=UNITS)
    ap.add_argument(
        "--base", help="float32's run, read if there, else made and written"
    )
    ap.add_argument("--json", help="every transcript and its dlogp")
    args = ap.parse_args()

    nm = nemo.Nemo(args.nemo)
    a = canon.arch(nm.config)
    w32 = {k: v.astype(np.float32) for k, v in canon.tensors(nm.state, a)}
    sp = spm.Model(nm.tokenizer_model())
    files = []
    for d in args.corpus:
        with open(os.path.join(d, "refs.tsv"), encoding="utf-8") as f:
            files += [
                (os.path.join(d, n), t)
                for n, t in (
                    line.rstrip("\n").split("\t", 1) for line in f if line.strip()
                )
            ]

    base = {}
    if args.base and os.path.exists(args.base):
        with open(args.base, "rb") as f:
            base = pickle.load(f)
    todo = [p for p, _ in files if p not in base]
    if todo:
        t0 = time.time()
        for p in todo:
            feat = reference.features(read_wav(p), w32, a).astype(np.float32)
            enc = reference.encode(feat, w32, a)
            trace = []
            toks = reference.greedy(enc, w32, a, trace)
            base[p] = {
                "text": sp.decode([k for k, _, _ in toks]),
                "trace": trace,
                "enc": enc,
                "logp": path_logp(enc, w32, a, trace),
            }
        print(
            "float32 for %d recordings in %.0f s" % (len(todo), time.time() - t0),
            flush=True,
        )
        if args.base:
            with open(args.base, "wb") as f:
                pickle.dump(base, f)

    t0 = time.time()
    models, dtype = load(args.enc, args.units)
    print(
        "%s (%s): loaded in %.1f s, lengths %s"
        % (
            os.path.basename(args.enc),
            args.units,
            time.time() - t0,
            ", ".join("%g s" % (T / 100) for T in sorted(models)),
        ),
        flush=True,
    )
    longest = max(models)
    longer = [p for p, _ in files if len(read_wav(p)) // a["hop_length"] > longest]
    if longer:
        print(
            "%d recordings longer than the longest function (%g s) left out: %s"
            % (
                len(longer),
                longest / 100,
                ", ".join(os.path.basename(p) for p in longer),
            ),
            flush=True,
        )
        files = [(p, r) for p, r in files if p not in longer]
    res, busy = [], 0.0
    for p, ref in files:
        b = base[p]
        feat = reference.features(read_wav(p), w32, a).astype(np.float32)
        enc, _, dt = run(models, feat, dtype)
        busy += dt
        toks = reference.greedy(enc.astype(np.float32), w32, a)
        logp = path_logp(enc.astype(np.float32), w32, a, b["trace"])
        res.append(
            {
                "file": p,
                "ref": ref,
                "text": sp.decode([k for k, _, _ in toks]),
                "dlogp": (logp - b["logp"]) / len(b["trace"]),
                "encerr": float(
                    np.linalg.norm(enc - b["enc"]) / np.linalg.norm(b["enc"])
                ),
            }
        )
    e0, n0 = wer.wer([(ref, base[p]["text"]) for p, ref in files])
    e, n = wer.wer([(r["ref"], r["text"]) for r in res])
    changed = sum(r["text"] != base[r["file"]]["text"] for r in res)
    d = np.array([r["dlogp"] for r in res])
    t = (
        d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
        if len(d) > 1 and d.std() > 0
        else 0
    )
    audio = sum(len(read_wav(p)) for p, _ in files) / 16000
    print("float32                WER %5.2f%% (%d/%d)" % (100 * e0 / n0, e0, n0))
    print(
        "%-22s WER %5.2f%% (%d/%d)  changed %3d  dlogp %+.5f  t %6.1f  enc err %.4f  encoder %.1f s for %.0f s, %.0fx"
        % (
            os.path.basename(args.enc),
            100 * e / n,
            e,
            n,
            changed,
            d.mean(),
            t,
            np.mean([r["encerr"] for r in res]),
            busy,
            audio,
            audio / busy,
        ),
        flush=True,
    )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(
                [{k: r[k] for k in ("file", "text", "dlogp", "encerr")} for r in res],
                f,
                ensure_ascii=False,
                indent=0,
            )


if __name__ == "__main__":
    main()
