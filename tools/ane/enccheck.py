"""The Core ML encoder (encoder.py's) against reference.py's in float64, on
recordings: the encoder output's relative error over the true frames, and
whether the transcript (reference.py's decoder on each) is the same.

  enccheck.py MODEL.nemo ENC.mlmodelc WAV... [--units cpu|ne|gpu|all]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import coremltools as ct

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pktito_tools import canon, nemo, reference as R, spm, wavio  # noqa: E402
from ane.encoder import sub_lengths, turn_tables  # noqa: E402

UNITS = {
    "cpu": ct.ComputeUnit.CPU_ONLY,
    "ne": ct.ComputeUnit.CPU_AND_NE,
    "gpu": ct.ComputeUnit.CPU_AND_GPU,
    "all": ct.ComputeUnit.ALL,
}


# the functions that take the position tables (several lengths), and their
# d_model: encoder.py's build
TABLES = {}


def functions(path):
    """{frames: [function names, its parts in order]} of a compiled model,
    from its metadata ("f3072", or "f3072p0", "f3072p1", ...; a function
    takes each length its mask's enumerated shapes list, or its one), and
    its inputs' type"""
    with open(os.path.join(path, "metadata.json")) as f:
        meta = json.load(f)[0]
    out, dtype = {}, np.float32
    TABLES.clear()
    for fn in meta["functions"]:
        mask = [i for i in fn["inputSchema"] if i["name"] == "mask"][0]
        pe = [i for i in fn["inputSchema"] if i["name"] == "pe"]
        if pe:
            TABLES[fn["name"]] = json.loads(pe[0]["shape"])[2]  # d_model
        for shape in json.loads(mask.get("enumeratedShapes") or "[%s]" % mask["shape"]):
            out.setdefault(shape[-1], []).append(fn["name"])
        dtype = np.float16 if mask["dataType"] == "Float16" else np.float32
    return {
        T: sorted(n, key=lambda x: int(x.split("p")[1]) if "p" in x else 0)
        for T, n in out.items()
    }, dtype


def load(path, units):
    """{frames: [models, a part each]} for each length of a compiled model,
    each function loaded once for all its lengths, and its inputs' type"""
    fns, dtype = functions(path)
    loaded = {}
    for f in {f for names in fns.values() for f in names}:
        loaded[f] = ct.models.CompiledMLModel(
            path, compute_units=UNITS[units], function_name=f
        )
        loaded[f].tables = TABLES.get(f)
    return {T: [loaded[f] for f in names] for T, names in fns.items()}, dtype


def run(models, feat, dtype):
    """feat [T][n_mels] through the shortest function that holds it"""
    T = feat.shape[0]
    Tb = min(t for t in models if t >= T)
    mel = np.zeros((1, 1, feat.shape[1], Tb), dtype)
    mel[0, 0, :, :T] = feat.T
    mask = np.zeros((1, 1, 1, Tb), dtype)
    mask[..., :T] = 1
    parts = models[Tb]
    tabs = {}
    if parts[0].tables:
        # as core/ane.m makes them: turn_tables from the model's frequencies
        D = parts[0].tables
        pe, cos, sin, _ = turn_tables(
            sub_lengths(Tb)[-1],
            {"pos.freq": canon.pos_freq(D).astype(np.float64)},
            {"d_model": D},
        )
        tabs = {
            "pe": pe[None, None].astype(dtype),
            "turn_cos": cos[None, None].astype(dtype),
            "turn_sin": sin[None, None].astype(dtype),
        }
    t0 = time.time()
    y = parts[0].predict({"mel": mel, "mask": mask, **tabs})
    for m in parts[1:]:
        y = m.predict({"x": y["x"], "mask": mask, **tabs})
    out = y["enc"]
    dt = time.time() - t0
    return np.asarray(out, np.float64)[0, :, 0, : sub_lengths(T)[-1]].T, Tb, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nemo")
    ap.add_argument("enc")
    ap.add_argument("wav", nargs="+")
    ap.add_argument("--units", default="cpu", choices=UNITS)
    args = ap.parse_args()

    nm = nemo.Nemo(args.nemo)
    a = canon.arch(nm.config)
    w = dict(canon.tensors(nm.state, a))
    sp = spm.Model(nm.tokenizer_model())
    t0 = time.time()
    models, dtype = load(args.enc, args.units)
    print(
        "loaded %s (%s) in %.1f s: frames %s"
        % (os.path.basename(args.enc), args.units, time.time() - t0, sorted(models)),
        flush=True,
    )
    for path in args.wav:
        audio = wavio.read(path)
        audio = audio[0] if isinstance(audio, tuple) else audio
        feat = R.features(audio, w, a)
        ref = R.encode(feat, w, a)
        enc, Tb, dt = run(models, feat, dtype)
        err = np.linalg.norm(enc - ref) / np.linalg.norm(ref)
        bad = int(np.sum(~np.isfinite(enc)))
        t_ref = sp.decode([k for k, _, _ in R.greedy(ref, w, a)])
        t_ane = sp.decode([k for k, _, _ in R.greedy(enc, w, a)])
        print(
            "%s: %d frames in %d, %.3f s; rel err %.5f, max |d| %.4f, non-finite %d; transcript %s"
            % (
                os.path.basename(path),
                feat.shape[0],
                Tb,
                dt,
                err,
                np.abs(enc - ref).max(),
                bad,
                "same" if t_ref == t_ane else "DIFFERS",
            ),
            flush=True,
        )
        if t_ref != t_ane:
            print("   ref: %s\n   ane: %s" % (t_ref, t_ane))


if __name__ == "__main__":
    main()
