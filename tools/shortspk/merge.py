"""The join check (rewind.py's, and core/hold.c's) scored against a
speaker-embedding model: for every pair of speakers it was asked about, the
mean cosine between their clips' ReDimNet2-B3 embeddings, 0.40 or more
taken as one voice, beside what the check said.
usage: EMBEDDER=redimnet2-b3-vox2-lm.onnx AUDIO=DIR merge.py RUNS
(reads RUNS/SHOW.EPISODE.json from rewind.py and AUDIO/SHOW/audio/EPISODE.mp3;
writes RUNS/merge.json; needs onnxruntime and ffmpeg)"""

import sys, os, glob, json, subprocess, numpy as np, onnxruntime as ort

M = ort.InferenceSession(os.environ["EMBEDDER"], providers=["CPUExecutionProvider"])
CLIP, T = 400, 0.40


def pcm(name):
    show, ep = name.split(".", 1)
    src = os.path.join(os.environ["AUDIO"], show, "audio", ep + ".mp3")
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", src, "-ac", "1", "-ar", "16000"]
        + ["-f", "s16le", "-"],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(r.stdout, np.int16).astype(np.float32) / 32768


def embed(a, starts):
    x = np.stack([a[s * 160 : (s + CLIP) * 160] for s in starts])
    e = M.run(None, {"waveform": x})[0]
    return e / np.linalg.norm(e, axis=1, keepdims=True)


v = sys.argv[1]
rows, cost = [], []
for f in sorted(glob.glob(v + "/*.*.json")):
    name = os.path.basename(f)[:-5]
    j = json.load(open(f))
    join = j["join"]
    cost.append(join["heard"] / (j["minutes"] * 60))
    if not join["pairs"]:
        continue
    a = pcm(name)
    E = {int(k): embed(a, s) for k, s in join["clips"].items()}
    for i, k, said, joined in join["pairs"]:
        cos = float((E[i] @ E[k].T).mean())
        rows.append((name, i, k, cos, said is not None, joined))
    print(name, flush=True)
json.dump(rows, open(v + "/merge.json", "w"))
same = [r for r in rows if r[3] >= T]
diff = [r for r in rows if r[3] < T]
print(
    "%d pairs of long speakers; one voice by the embeddings: %d"
    % (len(rows), len(same))
)
for what, k in (("one column in the first hearing", 4), ("and in every trial", 5)):
    print(
        "   %s: %d of %d one-voice pairs, %d of %d others"
        % (what, sum(r[k] for r in same), len(same), sum(r[k] for r in diff), len(diff))
    )
for r in diff:
    if r[5]:
        print(
            "   joined at cosine %.2f: %s %d and %d" % (r[3], r[0], r[1] + 1, r[2] + 1)
        )
c = np.array(cost)
print(
    "   heard: median %.1f%%, mean %.1f%%, max %.1f%% of the recording"
    % (np.median(c), c.mean(), c.max())
)
