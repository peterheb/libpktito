"""Which speaker said each word: a transcript's word times against a
diarizer's speaker probabilities, in numpy, the merge docs/DIARIZATION.md
("From frames to words") specifies for the C side.

rule "nemo" is NeMo's (SpeakerTaggedASR's offline merge,
get_word_dict_content_offline in NeMo 3.0's multispk_transcribe_utils): a
word's speaker is the one whose probability, averaged over the word's
frames and the frame before them (a transducer emits a token a little after
it is heard), is highest, each average raised to at least 0.01 first and
the first of equals winning. So a word the diarizer heard no one say, every
average under 0.01, goes to speaker 0: whoever spoke first.

rule "near" is the alternative DIARIZATION.md measures against it: a word
whose averages are all under one half takes the speaker heard nearest it
within NEAR seconds, and failing that the previous word's.
"""
import numpy as np

NEAR = 0.5           # seconds either side to look for a speaker for an unheard word ("near")
LEFT = 1             # frames before a word's first that its average includes (NeMo's left_frame_shift)
FLOOR = 0.01         # what NeMo raises every average to (its min_sigmoid_val)


def word_frames(start, end, frame):
    """A word's frames [f0, f1) on the diarizer's grid: the frames its
    interval touches, at least one."""
    f0 = int(np.floor(start / frame + 1e-9))
    f1 = int(np.ceil(end / frame - 1e-9))
    return f0, max(f1, f0 + 1)


def assign(words, probs, frame=0.08, rule="nemo"):
    """[(speaker, mean)] for words [(start, end)] in seconds; mean is the
    word's averaged probabilities [n_spk]. Under "near", speaker is -1 only
    when the recording has no speaker at all."""
    T, S = probs.shape
    out = []
    active = probs > 0.5
    for start, end in words:
        f0, f1 = word_frames(start, end, frame)
        lo, hi = max(f0 - LEFT, 0), min(f1, T)
        m = probs[lo:hi].mean(0) if hi > lo else np.zeros(S)
        if rule == "nemo":
            out.append((int(np.argmax(np.clip(m, FLOOR, 1.0))), m))
            continue
        if m.max() >= 0.5:
            out.append((int(np.argmax(m)), m))
            continue
        # nobody heard over the word: the nearest frame within NEAR where someone was
        k = int(round(NEAR / frame))
        best, spk = None, -1
        for t in range(max(lo - k, 0), min(hi + k, T)):
            if active[t].any():
                d = lo - t if t < lo else (t - hi + 1 if t >= hi else 0)
                if best is None or d < best:
                    best, spk = d, int(np.argmax(probs[t]))
        if spk < 0 and out:
            spk = out[-1][0]
        out.append((spk, m))
    if out and out[0][0] < 0:                    # an unheard start takes the first speaker after it
        nxt = next((s for s, _ in out if s >= 0), -1)
        out = [(nxt if s < 0 else s, m) for s, m in out]
    return out


def turns(words, speakers):
    """Runs of consecutive words of one speaker: [(speaker, first, count)],
    NeMo's "sentences" (SpeakerTaggedASR.get_sentences_values)."""
    out = []
    for i, s in enumerate(speakers):
        if out and out[-1][0] == s:
            out[-1][2] += 1
        else:
            out.append([s, i, 1])
    return [tuple(t) for t in out]
