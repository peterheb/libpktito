#!/usr/bin/env python3
"""core/segment.c against Silero's own Python, event for event.

get_speech_timestamps_from_probs and VADIterator are imported from Silero's
source (SILERO=path to a checkout, or the installed silero_vad package) and
run on the same probabilities as tests/segcheck, which drives pk_segments
and pk_iter_step. The probability sequences are made to land where the
state machines branch: runs that start and end speech, silences just
shorter and just longer than min_silence, speech that outruns a small
max_speech so that every splitting path runs, both values of
use_max_poss_sil_at_max_speech, and probabilities on the float32 values
either side of each threshold - 0.5 - 0.15 is 0.35000000000000003 in
Python, and a port that compared in float would decide 0.35f the other
way. Segments must be equal to the sample, events to the frame and sample.

    python3 tests/segcheck.py [tests/segcheck] [N]
"""
import os
import random
import struct
import subprocess
import sys


def silero():
    src = os.environ.get('SILERO')
    if src:
        sys.path.insert(0, os.path.join(src, 'src'))
    import torch  # noqa: F401  utils_vad imports it at the top
    from silero_vad.utils_vad import VADIterator, get_speech_timestamps_from_probs
    return VADIterator, get_speech_timestamps_from_probs


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def sequence(rng, n):
    """probabilities as float32 values, in runs that cross the thresholds"""
    edges = [f32(0.5), 0.5, f32(0.35), 0.35000000000000003, f32(0.15), f32(0.01)]
    out = []
    while len(out) < n:
        kind = rng.random()
        run = rng.choice([1, 2, 3, 4, 5, 8, 13, 30, 100])
        if kind < 0.3:
            out += [f32(rng.uniform(0.5, 1)) for _ in range(run)]
        elif kind < 0.6:
            out += [f32(rng.uniform(0, 0.35)) for _ in range(run)]
        elif kind < 0.8:
            out += [f32(rng.uniform(0.3, 0.55)) for _ in range(run)]
        else:
            e = rng.choice(edges)
            out += [f32(rng.choice([e, struct.unpack('<f', struct.pack('<I', struct.unpack('<I', struct.pack('<f', e))[0] + rng.choice([-1, 1])))[0]])) for _ in range(run)]
    return [min(max(x, 0.0), 1.0) for x in out[:n]]


class Fake:
    """A model that answers VADIterator with the next probability."""
    def __init__(self, probs):
        self.probs, self.i = probs, 0

    def reset_states(self):
        self.i = 0

    def __call__(self, x, sr):
        import torch
        p = self.probs[self.i]
        self.i += 1
        return torch.tensor([[p]], dtype=torch.float32)


def main():
    exe = sys.argv[1] if len(sys.argv) > 1 else 'tests/segcheck'
    cases = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    try:
        VADIterator, from_probs = silero()
    except ImportError as e:
        print('segcheck: no silero_vad or torch to compare with (%s) - skipped' % e)
        return 0
    import torch
    rng = random.Random(20261020)
    bad = 0
    for case in range(cases):
        rate = rng.choice([16000, 8000])
        frame = 512 if rate == 16000 else 256
        n = rng.choice([0, 1, 2, 5, 40, 200, 1000, 3000])
        probs = sequence(rng, n)
        thr = rng.choice([0.5, 0.5, 0.3, 0.15, 0.1, 0.8, 0.35])
        neg = rng.choice([None, None, None, 0.2, 0.35, f32(0.35)])
        min_speech = rng.choice([250, 250, 0, 32, 100, 1000])
        max_speech = rng.choice([float('inf'), float('inf'), 0.5, 1, 2.5, 10])
        min_sil = rng.choice([100, 100, 0, 32, 64, 300])
        pad = rng.choice([30, 30, 0, 10, 33.3, 200])
        min_sil_max = rng.choice([98, 98, 0, 64, 500])
        use_max = rng.choice([True, False])
        n_samples = n * frame - rng.choice([0, 0, 1, frame // 2, frame - 1]) if n else 0
        kw = dict(sampling_rate=rate, threshold=thr, min_speech_duration_ms=min_speech,
                  max_speech_duration_s=max_speech, min_silence_duration_ms=min_sil,
                  speech_pad_ms=pad, neg_threshold=neg, min_silence_at_max_speech=min_sil_max,
                  use_max_poss_sil_at_max_speech=use_max,
                  audio_length_samples=n_samples if n_samples else None)
        py = [('seg', d['start'], d['end']) for d in from_probs(probs, **kw)]
        it = VADIterator(Fake(probs), threshold=thr, sampling_rate=rate,
                         min_silence_duration_ms=min_sil, speech_pad_ms=pad, neg_threshold=neg)
        for i in range(n):
            ev = it(torch.zeros(frame))
            if ev:
                k = 'start' if 'start' in ev else 'end'
                py.append((k, i, ev[k]))
        args = [exe, str(rate), str(n_samples), repr(thr), repr(-1.0 if neg is None else neg),
                repr(min_speech), 'inf' if max_speech == float('inf') else repr(max_speech),
                repr(min_sil), repr(pad), repr(min_sil_max), '1' if use_max else '0']
        out = subprocess.run(args, input='\n'.join('%.9g' % p for p in probs), capture_output=True,
                             text=True, check=True).stdout.split('\n')
        c = [(w[0],) + tuple(int(v) for v in w[1:]) for w in (line.split() for line in out) if w]
        if c != py:
            bad += 1
            if bad <= 5:
                diff = next(i for i in range(max(len(c), len(py))) if i >= len(c) or i >= len(py) or c[i] != py[i])
                print('FAIL case %d (%s): first difference at item %d: C %s, Python %s' % (
                    case, ' '.join(args[1:]), diff, c[diff] if diff < len(c) else None,
                    py[diff] if diff < len(py) else None))
    if bad:
        print('segcheck: %d of %d cases differ' % (bad, cases))
        return 1
    print('segcheck: %d cases, segments and events identical to Silero\'s Python' % cases)
    return 0


if __name__ == '__main__':
    sys.exit(main())
