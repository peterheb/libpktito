"""An end-of-recording check for one voice on two numbers: a diarizer that has
heard nothing hears three 4 s clips of every long speaker, one speaker after
another; two numbers whose clips it puts in one column are tested again as a
pair on other clips, and joined if it says one speaker twice. Scored against
ReDimNet2-B3 (cosine >= 0.40 = one voice).
usage: PKTITO_DIARIZER=... EMBEDDER=redimnet2-b3-vox2-lm.onnx AUDIO=DIR merge.py VARIANT
(reads VARIANT/SHOW.EPISODE.npz from rewind.py and AUDIO/SHOW/audio/EPISODE.mp3;
writes merge.VARIANT.json; needs onnxruntime and ffmpeg)"""

import sys, os, glob, json, subprocess, wave, tempfile, numpy as np, onnxruntime as ort

M = ort.InferenceSession(os.environ["EMBEDDER"], providers=["CPUExecutionProvider"])
PK = os.environ.get("PKTITO_C", "./pktito-c")
DIAR = os.environ["PKTITO_DIARIZER"]  # nemotron-3-diarization.pkf
tmp = tempfile.mkdtemp()
CLIP, N, T = 400, 8, 0.40


def pcm(name):
    show, ep = name.split(".", 1)
    r = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            os.path.join(os.environ["AUDIO"], show, "audio", ep + ".mp3"),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "s16le",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(r.stdout, np.int16).astype(np.float32) / 32768


def clips(mask, at):
    ok = mask.copy()
    ok[1:] &= np.diff(at) == 1
    run = np.zeros(len(ok), int)
    for i in range(1, len(ok)):
        run[i] = run[i - 1] + 1 if ok[i] and mask[i - 1] else 0
    ends = np.flatnonzero(run >= CLIP)
    out = []
    last = -(10**9)
    for e in ends:
        if e - last >= CLIP + 200:
            out.append(int(at[e]) - CLIP)
            last = e
    if len(out) > N:
        out = [out[i] for i in np.linspace(0, len(out) - 1, N).astype(int)]
    return out


def embed(a, starts):
    x = np.stack([a[s * 160 : (s + CLIP) * 160] for s in starts])
    e = M.run(None, {"waveform": x})[0]
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def fresh(a, groups):
    """the column a fresh diarizer gives each group of clips (None if it hears under 1 s), and the seconds it heard"""
    gap = np.zeros(8000, np.float32)
    parts = []
    where = []
    off = 0
    for st in groups:
        lo = off
        for s in st:
            parts += [a[s * 160 : (s + CLIP) * 160], gap]
            off += CLIP + 50
        where.append((lo, off))
    p = os.path.join(tmp, "m.wav")
    w = wave.open(p, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
    w.writeframes((np.concatenate(parts) * 32768).astype(np.int16).tobytes())
    w.close()
    o = subprocess.run(
        [PK, "probs", "-m", DIAR, p], capture_output=True, text=True, check=True
    ).stdout
    P = np.array(o.split(), dtype=np.float32).reshape(-1, 9)[:, 1:] > 0.5
    c = [P[lo:hi].sum(0) for lo, hi in where]
    return [int(x.argmax()) if x.max() >= 100 else None for x in c], off / 100


v = sys.argv[1]
rows = []
for f in sorted(glob.glob(v + "/*.npz")):
    name = os.path.basename(f)[:-4]
    z = np.load(f)
    at = z["at"]
    a = None
    for run, X in (("plain", z["B"]), ("rewind", z["F"])):
        one = X.sum(1) == 1
        lab = X.argmax(1)
        ST = {
            j: clips(one & (lab == j), at)
            for j in range(8)
            if (one & (lab == j)).sum() >= 6000
        }
        ST = {j: s for j, s in ST.items() if len(s) >= 6}
        if len(ST) < 2:
            continue
        if a is None:
            a = pcm(name)
        E = {j: embed(a, s) for j, s in ST.items()}
        K = sorted(ST)
        col, heard = fresh(a, [ST[j][0::3][:3] for j in K])
        for x in range(len(K)):
            for y in range(x + 1, len(K)):
                i, j = K[x], K[y]
                cos = float((E[i] @ E[j].T).mean())
                batch = col[x] is not None and col[x] == col[y]
                pair = None
                if batch:
                    c2, h2 = fresh(a, [ST[i][1::3][:3], ST[j][1::3][:3]])
                    heard += h2
                    pair = c2[0] is not None and c2[0] == c2[1]
                rows.append((name, run, i, j, cos, batch, pair))
        rows.append((name, run, -1, -1, heard, len(K), float(at[-1]) / 100))
    print(name, flush=True)
json.dump(rows, open("merge.%s.json" % v, "w"))
for run in ("plain", "rewind"):
    R = [r for r in rows if r[1] == run and r[2] >= 0]
    H = [r for r in rows if r[1] == run and r[2] < 0]
    same = [r for r in R if r[4] >= T]
    diff = [r for r in R if r[4] < T]
    print(
        "%s: %d pairs of long speakers; one voice by the embeddings: %d"
        % (run, len(R), len(same))
    )
    print(
        "   batch alone: joins %d of %d one-voice pairs, wrongly joins %d of %d"
        % (sum(r[5] for r in same), len(same), sum(r[5] for r in diff), len(diff))
    )
    print(
        "   batch then pair: joins %d of %d, wrongly joins %d of %d"
        % (
            sum(bool(r[6]) for r in same),
            len(same),
            sum(bool(r[6]) for r in diff),
            len(diff),
        )
    )
    c = np.array([100 * h[4] / h[6] for h in H])
    print(
        "   cost: median %.1f%%, mean %.1f%%, max %.1f%% of the recording"
        % (np.median(c), c.mean(), c.max())
    )
