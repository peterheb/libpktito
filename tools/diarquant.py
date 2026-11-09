"""What rounding Nemotron 3 Diarization's matrices, and the activations
entering them, costs: tools/quantcheck.py's experiment for the diarizer,
the one that chose its numerics (docs/DIARIZATION.md, "Why int8").

  diarquant.py MODEL.nemo WAV... --cache DIR [--configs a,b,...] [--jobs N] [--json OUT]

Each configuration runs the model (tools/pktito_tools/nemotron3.py, here in
float32, which is what NeMo computes in) over every WAV in the card's
30.4 s configuration, with some of its matrices, and the activations
entering them, rounded one of quantcheck.py's ways. Against the unrounded
float32 run it reports, over all the WAVs:

  labels   speaker labels (a speaker's probability over one half in a
           10 ms frame) not float32's, of all
  dBCE     how much less likely the rounded model finds float32's labels:
           the binary cross-entropy of its probabilities against them, less
           float32's own, per label, in nats. A paired number, like
           quantcheck.py's dlogp, that moves before a label does
  t        dBCE's paired t over the WAVs
  DER/f32  the DER of its labels against float32's
  DER      against the true speakers, where NAME.rttm is beside NAME.wav:
           no collar, overlapped speech scored (float32's own on its line),
           and the mean of each WAV's difference from float32's, with its t
  words    where NAME.json beside the WAV holds pktito-c --json's words:
           words whose speaker (NeMo's rule on 80 ms means, speakers.py) is
           not float32's, and words not the true speaker's (float32's on
           its line)
  turns    changes of speaker between words that float32's do not have,
           and float32's that these lack
  heard    speakers heard, a second or more, over all the WAVs
  split    float32's speakers part of whose frames alone (5% of them and
           2 s or more) the rounded run gives a second column, and merged:
           columns given two of float32's speakers that way

The probabilities of every run are kept in --cache (DIR/CONFIG/NAME.npy),
and a run found there is not computed again: --jobs N runs the work in N
processes, and a later call that adds configurations computes only theirs.

Each process holds the model, one recording's samples and one chunk's
work: at most 1.95 GB for an int8 configuration and 2.61 GB for f64 over
an hour-long recording (measured, /usr/bin/time -v, OPENBLAS_NUM_THREADS=2).
Size --jobs from those and run it under a memory cap, such as
systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0, so that a
mistake ends the run and not the machine.

A configuration is quantcheck.py's rules separated by ";", each a weight
rounding, optionally "+" an activation rounding, optionally "@" the classes
it applies to, "/"-separated (default enc); a later rule overrides an
earlier one for the classes it names. This model's classes:

  stack   the 1024 -> 512 matrix over 8 stacked feature frames (its input:
          the log-mels)
  qkv     att.q, att.k, att.v (their input: the attention's norm)
  o       att.o (its input: the attention's output)
  ffup    ff.up (its input: the feed-forward's norm)
  ffdown  ff.down (its input: GELU's output)
  proj    the 512 -> 192 projection (its input: the final norm)
  up      the upsampler, 576 -> 1536 (its input: three projected frames
          side by side, so a block of 32 never spans two)
  head    head.hidden and head.out (their inputs: ReLU's outputs)
  enc     all of them

"f32@CLASSES" leaves those classes unrounded, and the rule "f32cuts" makes
every cache cut keep the frames float32's kept, so that what is left is the
network's own rounding. "f64" alone is the float64 reference: what rounding
differently does with no int8 at all.
"""
import argparse
import itertools
import json
import math
import multiprocessing
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantcheck  # noqa: E402  (its roundings)
from pktito_tools import nemo, nemotron3, reference, sortformer, speakers, wavio  # noqa: E402

FINE = {"att.q": "qkv", "att.k": "qkv", "att.v": "qkv", "att.o": "o", "ff.up": "ffup", "ff.down": "ffdown"}
CLASSES = set(FINE.values()) | {"stack", "proj", "up", "head", "enc"}
GRID = 0.01          # seconds a probability row, and a label, covers
HEARD = 100          # rows (a second) a speaker needs to count as heard
SPLIT = (0.05, 200)  # the share of a speaker's frames, and the rows, that make a split
EPS = 1e-7           # dBCE's probabilities are kept this far from 0 and 1 (float32's sigmoid reaches both)


def classes(name):
    """The classes a weight matrix belongs to; none for anything else."""
    if name in ("stack", "proj", "up"):
        return {name, "enc"}
    if name in ("head.hidden", "head.out"):
        return {"head", "enc"}
    if name.startswith("l") and name.split(".", 1)[1] in FINE:
        return {FINE[name.split(".", 1)[1]], "enc"}
    return set()


def parse(cfg):
    """[(weight rounding or None, activation rounding or None, classes)] in
    order, and whether float32's cuts are imposed."""
    rules, fixed = [], False
    for rule in cfg.split(";"):
        if rule == "f32cuts":
            fixed = True
            continue
        on = {"enc"}
        if "@" in rule:
            rule, cl = rule.split("@")
            on = set(cl.split("/"))
        wq, _, aq = rule.partition("+")
        if not on <= CLASSES or (wq != "f32" and wq not in quantcheck.WQ) or (aq and aq not in quantcheck.AQ):
            raise SystemExit("unknown configuration %r" % cfg)
        rules.append((quantcheck.WQ.get(wq), quantcheck.AQ.get(aq), on))
    return rules, fixed


def setup(w32, rules):
    """The weights under the rules, and the product that goes with them."""
    wd, act = dict(w32), {}
    for name, W in w32.items():
        hit = None
        for r in rules:
            if classes(name) & r[2]:
                hit = r
        if hit and hit[0]:
            wd[name] = hit[0](W.astype(np.float64)).astype(np.float32)
            act[name] = hit[1]

    def mm(x, w, name):
        if act.get(name):
            x = act[name](x).astype(np.float32)
        return x @ w[name].T
    return wd, mm


# ------------------------------------------------------------- the runs ---

PLAIN_MM = reference.mm
CHOOSE = sortformer.Stream._choose
_W = {}              # a worker's model, and its last configuration's weights
_CUTS = {}           # the cut choices a run records ("record") or takes ("impose")


def _choose(self, p):
    if _CUTS.get("impose") is not None:
        return _CUTS["impose"].pop(0)
    idx, off = CHOOSE(self, p)
    if _CUTS.get("record") is not None:
        _CUTS["record"].append((idx, off))
    return idx, off


def _load(path, f64):
    nm = nemo.Nemo(path)
    a = nemotron3.arch(nm.config, nemotron3.OFFLINE)
    w = nemotron3.tensors(nm.state, a)
    del nm
    _W.update(a=a, w=w if f64 else {k: w.pop(k).astype(np.float32) for k in list(w)}, cfg=None)
    sortformer.Stream._choose = _choose


def _save(path, save, *args, **kw):
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        save(f, *args, **kw)
    os.replace(tmp, path)


def _job(job):
    """One configuration over one WAV: its probabilities saved, and
    float32's cut choices with them."""
    cfg, wav, out, cuts = job
    t0 = time.time()
    fixed = False
    if cfg == "f64":
        wd, mm = _W["w"], PLAIN_MM
    else:
        if _W["cfg"] != cfg:
            rules, _ = parse(cfg)
            _W.pop("wd", None)                  # never two rounded copies at once
            _W["wd"], _W["mm"], _W["cfg"] = setup(_W["w"], rules) + (cfg,)
        wd, mm = _W["wd"], _W["mm"]
        fixed = parse(cfg)[1]
    nemotron3.mm = reference.mm = mm
    _CUTS["record"] = [] if cfg == "f32" else None
    _CUTS["impose"] = None
    if fixed:
        z = np.load(cuts)
        _CUTS["impose"] = list(zip(z["idx"], z["off"]))
    audio, rate = wavio.read(wav)
    if rate != 16000:
        raise SystemExit("%s: %d Hz; the model takes 16000" % (wav, rate))
    p, st = nemotron3.diarize(audio, wd, _W["a"])
    _save(out, np.save, p.astype(np.float32))
    if _CUTS["record"] is not None:
        rec, L = _CUTS["record"], _W["a"]["spkcache_len"]
        _save(cuts, np.savez, idx=np.array([i for i, _ in rec], np.int64).reshape(len(rec), L),
              off=np.array([o for _, o in rec], bool).reshape(len(rec), L))
    return cfg, os.path.basename(wav), st.steps, time.time() - t0


def compute(todo, model, f64, jobs):
    if not todo:
        return
    t0 = time.time()
    if jobs <= 1:
        _load(model, f64)
        done = map(_job, todo)
    else:
        pool = multiprocessing.get_context("spawn").Pool(jobs, initializer=_load, initargs=(model, f64))
        done = pool.imap_unordered(_job, todo)
    for cfg, name, steps, dt in done:
        print("  %-24s %-12s %4d steps %6.0f s" % (cfg, name, steps, dt), flush=True)
    if jobs > 1:
        pool.close()
        pool.join()
    print("  %d runs in %.0f s" % (len(todo), time.time() - t0), flush=True)


# ------------------------------------------------------- the comparison ---

_PERMS = {}


def best_map(M):
    """The one-to-one map of M's rows to its columns (a few speakers each)
    that matches the most: {row: column}."""
    nr, nc = M.shape
    k = max(nr, nc)
    if k not in _PERMS:
        _PERMS[k] = np.array(list(itertools.permutations(range(k))))
    P = _PERMS[k]
    Mp = np.zeros((k, k))
    Mp[:nr, :nc] = M
    best = P[np.argmax(Mp[np.arange(k), P].sum(1))]
    return {r: int(best[r]) for r in range(nr) if best[r] < nc}


def der(R, H, keep=None):
    """The errors of labels H [T][columns] against R [T][speakers], each
    frame and speaker scored, under the map of R's speakers to H's columns
    that matches the most (md-eval's, on a grid): missed, false and
    confused frames, the frames of speech, and the map."""
    if keep is not None:
        R, H = R[keep], H[keep]
    M = R.T.astype(np.int64) @ H.astype(np.int64)
    m = best_map(M)
    nr, nh = R.sum(1), H.sum(1)
    miss = int(np.maximum(nr - nh, 0).sum())
    fa = int(np.maximum(nh - nr, 0).sum())
    conf = int(np.minimum(nr, nh).sum()) - int(sum(M[r, c] for r, c in m.items()))
    return {"miss": miss, "fa": fa, "conf": conf, "speech": int(nr.sum()), "map": m}


def pct(rows, key):
    """A DER over many WAVs, in percent: their errors over their speech."""
    e = sum(r[key]["miss"] + r[key]["fa"] + r[key]["conf"] for r in rows)
    return 100.0 * e / max(sum(r[key]["speech"] for r in rows), 1)


def read_rttm(path):
    segs = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if p and p[0] == "SPEAKER":
                segs.append((float(p[3]), float(p[3]) + float(p[4]), p[7]))
    return segs


def truth(segs, n):
    """The true speakers as labels [n][speakers] on the grid, and their names."""
    names = sorted({s for _, _, s in segs})
    R = np.zeros((max(n, int(round(max(b for _, b, _ in segs) / GRID)) + 1), len(names)), bool)
    for a, b, s in segs:
        R[int(round(a / GRID)):int(round(b / GRID)), names.index(s)] = True
    return R, names


def collar(segs, n, c=0.25):
    keep = np.ones(n, bool)
    k = int(round(c / GRID))
    for a, b, _ in segs:
        for t in (int(round(a / GRID)), int(round(b / GRID))):
            keep[max(t - k, 0):t + k] = False
    return keep


def means80(p):
    """NeMo's downsample_preds: each 8 rows' mean, a last partial window
    over its own rows - what NeMo's merge sees."""
    T = len(p)
    starts = np.arange(0, T, 8)
    return np.add.reduceat(p.astype(np.float64), starts, axis=0) / np.diff(np.append(starts, T))[:, None]


def true_speakers(segs, a, b):
    """A word's true speakers: those whose segments overlap it, most first,
    or the nearest segment's if none does."""
    ov = {}
    for s0, s1, k in segs:
        o = min(b, s1) - max(a, s0)
        if o > 0:
            ov[k] = ov.get(k, 0) + o
    if ov:
        return [k for k, _ in sorted(ov.items(), key=lambda kv: -kv[1])]
    mid = (a + b) / 2
    return [min(segs, key=lambda r: min(abs(r[0] - mid), abs(r[1] - mid)))[2]]


def changes(spk):
    return {i for i in range(1, len(spk)) if spk[i] != spk[i - 1]}


def splits(L0, L1):
    """(split, merged): float32 speakers whose lone frames the run gives in
    part to a second column, and columns given two of float32's speakers."""
    one0, one1 = L0.sum(1) == 1, L1.sum(1) == 1
    both = one0 & one1
    C = L0[both].T.astype(np.int64) @ L1[both].astype(np.int64)        # [f32 speaker][column]
    share, rows = SPLIT

    def big(v):
        return (v >= max(rows, share * v.sum())).sum() if v.sum() else 0
    return sum(big(C[i]) >= 2 for i in range(C.shape[0])), sum(big(C[:, j]) >= 2 for j in range(C.shape[1]))


def bce(p, y):
    p = np.clip(p.astype(np.float64), EPS, 1 - EPS)
    return -np.where(y, np.log(p), np.log1p(-p)).mean()


def compare(p0, p1, segs, words):
    """One WAV's numbers for a run p1 against float32's p0 (both [T][spk])."""
    L0, L1 = p0 > 0.5, p1 > 0.5
    T = len(p0)
    d = der(L0, L1)
    back = {c: s for s, c in d.pop("map").items()}                  # the run's column -> float32's
    r = {"labels": int((L0 != L1).sum()), "of": int(L0.size), "dbce": bce(p1, L0) - bce(p0, L0),
         "f32": d, "heard": int((L1.sum(0) >= HEARD).sum())}
    r["split"], r["merged"] = (int(x) for x in splits(L0, L1))
    if segs:
        R, names = truth(segs, T)
        H = np.zeros((len(R), L1.shape[1]), bool)
        H[:T] = L1
        r["true"] = der(R, H)
        r["collar"] = der(R, H, collar(segs, len(R)))
        r["collar"].pop("map")
    if words:
        spans = [(a, b) for a, b, _ in words]
        s0 = [s for s, _ in speakers.assign(spans, means80(p0), 0.08, "nemo")]
        mine = [s for s, _ in speakers.assign(spans, means80(p1), 0.08, "nemo")]
        s1 = [back.get(s, -1) for s in mine]
        moved = [i for i in range(len(words)) if s0[i] != s1[i]]
        c0, c1 = changes(s0), changes(s1)
        r.update(words=len(words), moved=len(moved), spurious=len(c1 - c0), lost=len(c0 - c1),
                 moved_words=[[i, words[i][0], words[i][1], words[i][2], s0[i], s1[i]] for i in moved])
        if segs:
            inv = {c: k for k, c in r["true"]["map"].items()}         # a column -> its true speaker
            tv = [true_speakers(segs, a, b) for a, b in spans]
            r["wrong"] = sum(1 for s, t in zip(mine, tv) if s not in inv or names[inv[s]] not in t)
    if segs:
        r["true"].pop("map")
    return r


def load_words(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        j = json.load(f)
    return [(w["start"], w["end"], w["word"]) for c in j["chunks"] for w in c["words"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nemo")
    ap.add_argument("wav", nargs="+")
    ap.add_argument("--cache", required=True, help="where each run's probabilities are kept")
    ap.add_argument("--configs", default="f64,w8row,w8row+a8b32,w8row+a8b32;f32cuts")
    ap.add_argument("--jobs", type=int, default=1, help="processes to run them in")
    ap.add_argument("--json", help="write every WAV's numbers and moved words here")
    args = ap.parse_args()
    configs = [c for c in args.configs.split(",") if c and c != "f32"]
    for c in configs:
        if c != "f64":
            parse(c)
    names = [os.path.splitext(os.path.basename(p))[0] for p in args.wav]
    if len(set(names)) != len(names):
        raise SystemExit("two WAVs with one name")

    def out(cfg, name):
        return os.path.join(args.cache, cfg.replace("/", "_"), name + ".npy")

    def cuts(name):
        return os.path.join(args.cache, "f32", name + ".cuts.npz")

    for cfgs, f64 in ((["f32"], False), ([c for c in configs if c != "f64"], False),
                      (["f64"] if "f64" in configs else [], True)):
        for c in cfgs:
            os.makedirs(os.path.dirname(out(c, "x")), exist_ok=True)
        compute([(c, p, out(c, n), cuts(n)) for c in cfgs for p, n in zip(args.wav, names)
                 if not os.path.exists(out(c, n))], args.nemo, f64, args.jobs)

    seconds = sum(len(np.load(out("f32", n), mmap_mode="r")) for n in names) * GRID
    print("%s: %d recordings, %.1f minutes" % (os.path.basename(args.nemo), len(names), seconds / 60))
    base = {n: np.load(out("f32", n)) for n in names}
    rttm = {n: read_rttm(os.path.splitext(p)[0] + ".rttm") if os.path.exists(os.path.splitext(p)[0] + ".rttm")
            else None for p, n in zip(args.wav, names)}
    words = {n: load_words(os.path.splitext(p)[0] + ".json") for p, n in zip(args.wav, names)}
    results = {}
    for cfg in ["f32"] + configs:
        rows = {n: compare(base[n], np.load(out(cfg, n)), rttm[n], words[n]) for n in names}
        results[cfg] = rows
        v = list(rows.values())
        lab, of = sum(r["labels"] for r in v), sum(r["of"] for r in v)
        db = np.array([r["dbce"] for r in v])
        t = db.mean() / (db.std(ddof=1) / math.sqrt(len(db))) if len(db) > 1 and db.std() > 0 else 0.0
        line = "%-24s labels %7d of %9d (%.4f%%)  dBCE %+.6f  t %6.1f  DER/f32 %.3f%%" % (
            cfg, lab, of, 100.0 * lab / of, db.mean(), t, pct(v, "f32"))
        scored = [n for n in names if rttm[n]]
        if scored:
            dd = np.array([pct([rows[n]], "true") - pct([results["f32"][n]], "true") for n in scored])
            tt = dd.mean() / (dd.std(ddof=1) / math.sqrt(len(dd))) if len(dd) > 1 and dd.std() > 0 else 0.0
            line += "  DER %.2f%% (collar %.2f%%), against float32's %+.3f (t %.1f)" % (
                pct([rows[n] for n in scored], "true"), pct([rows[n] for n in scored], "collar"), dd.mean(), tt)
        heard = [r for r, n in zip(v, names) if words[n]]
        if heard:
            line += "  words %d of %d moved, wrong %d  turns +%d -%d" % (
                sum(r["moved"] for r in heard), sum(r["words"] for r in heard),
                sum(r.get("wrong", 0) for r in heard), sum(r["spurious"] for r in heard),
                sum(r["lost"] for r in heard))
        line += "  heard %d  split %d merged %d" % (sum(r["heard"] for r in v), sum(r["split"] for r in v),
                                                   sum(r["merged"] for r in v))
        print(line, flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=0)


if __name__ == "__main__":
    main()
