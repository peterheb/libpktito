"""Short speakers held back, as libpktito would do it while a recording plays:
the diarizer's state is kept at every step; a speaker with under MIN seconds
who has been quiet for QUIET is cut from the audio, and the diarizer goes back
to the step before their first word and hears the rest again without them.
Before anyone is held, the state is forked and hears CLIP seconds of each
speaker held so far: a clip that lands in the column about to be cut says who
it is, their time adds up, and at MIN together they keep the number and the
held spans become theirs (no rewind). At the end the speakers still held are
played after the recording, longest first, and take what numbers are left.

Here the diarizer is pktito-c probs, which starts from the beginning each
time: going back to a step's state and hearing the cut audio from there is
the same computation as hearing the cut audio from the start, so each rewind
is one whole run here, and the cost reported is what the real one would pay
(the steps heard again).

usage: PKTITO_DIARIZER=nemotron-3-diarization.pkf [NEED=7] rewind.py FILE.wav OUT.json [MIN QUIET GROUP]
(writes OUT.json and OUT.npz: each kept frame's speakers in the plain run and in this one)"""
import sys, os, json, subprocess, wave, tempfile
import numpy as np

PK = os.environ.get("PKTITO_C", "./pktito-c")
DIAR = os.environ["PKTITO_DIARIZER"]  # nemotron-3-diarization.pkf
SR, HOP, STEP = 16000, 160, 2720          # 10 ms frames; a step is 27.2 s
MIN = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0     # a speaker with this much speech keeps a number
QUIET = float(sys.argv[4]) if len(sys.argv) > 4 else 120.0  # unheard this long, a shorter one is held
GROUP = float(sys.argv[5]) if len(sys.argv) > 5 else 60.0   # and with them, the others unheard this long
NEED = int(os.environ.get("NEED", 1))       # only once this many numbers are in use
BUDGET = lambda now: 0.25 * now + 300     # seconds heard twice, at most, by the time `now` seconds are in
CLIP = 400                                # of a held speaker, to know them again: 4 s
PAD, GAP = 30, 200                        # a held span: 0.3 s each side, pauses under 2 s filled
tmp = tempfile.mkdtemp()
runs = 0

def read(p):
    w = wave.open(p); assert w.getframerate() == SR and w.getnchannels() == 1
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16); w.close(); return a

def probs(a):
    global runs; runs += 1
    p = os.path.join(tmp, "x.wav")
    w = wave.open(p, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(a.tobytes()); w.close()
    out = subprocess.run([PK, "probs", "-m", DIAR, p], capture_output=True, text=True, check=True).stdout
    return np.array(out.split(), dtype=np.float32).reshape(-1, 9)[:, 1:] > 0.5

def spans(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))

def widen(mask):
    out = np.zeros_like(mask); last = None
    for s, e in spans(mask):
        s, e = max(0, s - PAD), min(len(mask), e + PAD)
        if last is not None and s - last < GAP: s = last
        out[s:e] = True; last = e
    return out

def clip_of(group):
    """CLIP frames from the middle of a held speaker's longest unbroken span"""
    fr = np.sort(np.concatenate([h["frames"] for h in group]))
    cuts = np.flatnonzero(np.diff(fr) > 1) + 1
    run = max(np.split(fr, cuts), key=len)
    m = max(0, (len(run) - CLIP) // 2)
    return run[m : m + CLIP]

def audio_of(a, frames):
    idx = (frames[:, None] * HOP + np.arange(HOP)[None, :]).ravel()
    return a[idx[idx < len(a)]]

def main():
    a = read(sys.argv[1]); T = len(a) // HOP
    keep = np.ones(T, bool)
    base = None; held = []; events = []; replayed = 0.0; settled = set(); tests = 0.0; ids = 0; promoted = {}
    pos = STEP                                  # the step boundary being looked at, in kept frames
    while True:
        at = np.flatnonzero(keep)               # kept frame -> the recording's frame
        A = probs(audio_of(a, at)); n = min(len(A), len(at)); A = A[:n]; at = at[:n]
        if base is None: base = A.copy()
        cut = None
        while pos <= n + STEP - 1:
            b = min(pos, n); now = at[b - 1] / 100
            secs = A[:b].sum(0) / 100
            last = np.array([at[np.flatnonzero(A[:b, k])[-1]] / 100 if secs[k] else 0 for k in range(8)])
            first = np.array([at[np.flatnonzero(A[:b, k])[0]] if secs[k] else -1 for k in range(8)])
            short = [k for k in range(8) if 0 < secs[k] < MIN and int(first[k]) // 100 not in settled]
            if any(now - last[k] >= QUIET for k in short) and b < n and (secs > 0).sum() >= NEED:
                who = [k for k in short if now - last[k] >= GROUP]
                # is one of them a speaker already held?  the fork hears a clip of each
                known = {}
                open_ids = sorted({h["id"] for h in held if h["id"] not in promoted})
                if open_ids:
                    sil = np.zeros(SR, np.int16); parts = [audio_of(a, at[:b])]; off = b; where = []
                    for i in open_ids:
                        c = clip_of([h for h in held if h["id"] == i])
                        parts += [sil, audio_of(a, c)]; where.append((off + 100, off + 100 + len(c))); off += 100 + len(c)
                        tests += 1 + len(c) / 100
                    Q = probs(np.concatenate(parts))
                    for i, (s0, e0) in zip(open_ids, where):
                        q = Q[s0:e0].sum(0) / 100; k = int(q.argmax())
                        if k in who and k not in known and q[k] >= 1.0 and q[k] >= 0.5 * q.sum(): known[k] = i
                for k in list(who):
                    if k in known and secs[k] + sum(h["seconds"] for h in held if h["id"] == known[k]) >= MIN:
                        promoted[known[k]] = int(first[k]); settled.add(int(first[k]) // 100); who.remove(k)
                        events.append({"at": now, "promoted": int(k) + 1, "cost": 0})
                if not who: pos += STEP; continue
                masks = [widen(A[:b, k]) for k in who]
                gone = np.any(masks, 0)
                back = (np.flatnonzero(gone)[0] // STEP) * STEP          # the step before the first held word
                cost = (b - back - gone[back:].sum()) / 100
                if replayed + cost > BUDGET(now):
                    for k in who: settled.add(int(first[k]) // 100)       # too dear: they keep their numbers
                    events.append({"at": now, "skipped": [int(k) + 1 for k in who], "cost": cost})
                else:
                    for k, m in zip(who, masks):
                        if k in known: i = known[k]
                        else: ids += 1; i = ids
                        held.append({"at": now, "column": int(k) + 1, "seconds": float(secs[k]), "frames": at[:b][m], "id": i})
                    replayed += cost
                    events.append({"at": now, "held": [int(k) + 1 for k in who], "cost": cost})
                    cut = (now, gone, b); break
            pos += STEP
        if cut is None: break
        now, gone, b = cut
        keep[at[:b][gone]] = False
        pos = (int(np.searchsorted(np.flatnonzero(keep), int(now * 100))) // STEP + 1) * STEP
    final, fat = A, at
    used = int((final.sum(0) > 0).sum())
    # the held speakers after the recording, longest first
    firsts = [int(fat[np.flatnonzero(final[:, k])[0]]) if final[:, k].any() else -1 for k in range(8)]
    for h in held:
        if h["id"] in promoted:
            d = [abs(f - promoted[h["id"]]) if f >= 0 else 1 << 30 for f in firsts]
            h["speaker"] = int(np.argmin(d)) + 1; h["new"] = False; h["lands"] = "promoted"
    groups = {}
    for h in held:
        if h["id"] not in promoted: groups.setdefault(h["id"], []).append(h)
    order = sorted(groups.values(), key=lambda g: -sum(h["seconds"] for h in g))
    tail = 0.0
    if order:
        sil = np.zeros(SR, np.int16); parts = [audio_of(a, fat)]; off = len(fat)
        for g in order:
            fr = np.sort(np.concatenate([h["frames"] for h in g]))
            parts += [sil, audio_of(a, fr)]
            for h in g: h["span"] = (off + 100, off + 100 + len(fr))
            off += 100 + len(fr); tail += 1 + len(fr) / 100
        E = probs(np.concatenate(parts))
        moved = int((E[: len(final) - 4000] != final[: len(final) - 4000]).sum()) if len(final) > 4000 else 0
        for h in [h for g in order for h in g]:
            Q = E[h["span"][0] : h["span"][1]]; s = Q.sum(0) / 100
            k = int(s.argmax()); h["lands"] = [round(float(x), 1) for x in s]
            h["speaker"] = k + 1 if s[k] >= 0.2 * max(1.0, Q.any(1).sum() / 100) and s[k] >= 1.0 else 0
            h["new"] = h["speaker"] > used
    else: moved = 0
    # the plain run against this one, on the frames kept, where each hears one speaker
    B = base[fat[fat < len(base)]]; F = final[: len(B)]
    one = (B.sum(1) == 1) & (F.sum(1) == 1)
    C = np.zeros((8, 8), int); np.add.at(C, (B[one].argmax(1), F[one].argmax(1)), 1)
    res = {"minutes": T / 6000, "runs": runs,
           "plain": [round(float(x), 1) for x in base.sum(0) / 100],
           "plain_first": [round(at0 / 6000, 2) if at0 >= 0 else None for at0 in [int(np.flatnonzero(base[:, k])[0]) if base[:, k].any() else -1 for k in range(8)]],
           "kept": [round(float(x), 1) for x in final.sum(0) / 100],
           "events": events, "replayed": replayed, "tail": tail, "tests": tests, "moved_by_tail": moved,
           "confusion": C.tolist(),
           "held": [{k: (v if k != "frames" else [(int(s), int(e)) for s, e in spans(np.isin(np.arange(T), v))]) for k, v in h.items()} for h in held]}
    json.dump(res, open(sys.argv[2], "w"))
    np.savez_compressed(sys.argv[2][:-5] + ".npz", B=B, F=F, at=fat[: len(B)])
    print(os.path.basename(sys.argv[1]), "%.0f min" % res["minutes"])
    print("  plain:", " ".join("%.0f" % x for x in res["plain"]))
    print("  now:  ", " ".join("%.0f" % x for x in res["kept"]), "  held:", len(held), "(%.0f s)" % sum(h["seconds"] for h in held))
    for e in events: print("   at %5.1f min:" % (e["at"] / 60), "held" if "held" in e else "known again, keeps the number:" if "promoted" in e else "too dear, kept", e.get("held", e.get("skipped", e.get("promoted"))), "heard again %.0f s" % e["cost"])
    for h in held: print("   held %d as %s (%.0f s, cut at %.1f min) ->" % (h["column"], chr(64 + h["id"]), h["seconds"], h["at"] / 60), "other" if not h["speaker"] else "speaker %d%s" % (h["speaker"], " (new)" if h["new"] else ""), h["lands"])
    tot = T / 100
    print("  heard again: %.1f%% in rewinds + %.1f%% in clips + %.1f%% at the end; one-speaker frames agreeing with the plain run: %.2f%% (rows) %.2f%% (columns) of %.0f s" %
          (100 * replayed / tot, 100 * tests / tot, 100 * tail / tot, 100 * C.max(1).sum() / max(1, C.sum()), 100 * C.max(0).sum() / max(1, C.sum()), C.sum() / 100))

main()
