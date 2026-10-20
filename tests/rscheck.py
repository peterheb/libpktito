#!/usr/bin/env python3
"""core/resample.c against torchaudio.transforms.Resample, which is what
Silero's read_audio brings a recording to 16 kHz with.

Each rate pair resamples the same signal both ways - noise, a sweep, full
scale at both ends - with tests/rscheck pushing it through in pieces of
random sizes, and the two outputs must be the same length and agree to
fp32 rounding. The kernel is torchaudio's own float32 taps; what is left is
the order the two add them in.

    python3 tests/rscheck.py [tests/rscheck]
"""
import subprocess
import sys

PAIRS = [(44100, 16000), (48000, 16000), (22050, 16000), (11025, 16000), (32000, 16000),
         (96000, 16000), (24000, 16000), (8000, 16000), (16000, 8000), (44100, 8000),
         (12345, 16000), (16001, 16000), (7999, 8000)]


def main():
    exe = sys.argv[1] if len(sys.argv) > 1 else 'tests/rscheck'
    try:
        import numpy as np
        import torch
        import torchaudio
    except ImportError as e:
        print('rscheck: no torchaudio to compare with (%s) - skipped' % e)
        return 0
    rng = np.random.default_rng(7)
    bad = 0
    worst = 0.0
    for k, (src, dst) in enumerate(PAIRS):
        n = int(src * 2.3) + k
        t = np.arange(n) / src
        x = (0.3 * rng.standard_normal(n) + 0.5 * np.sin(2 * np.pi * (100 + 3000 * t) * t)).astype(np.float32)
        x[:7] = 1.0
        x[-5:] = -1.0
        want = torchaudio.transforms.Resample(src, dst)(torch.from_numpy(x)).numpy()
        out = subprocess.run([exe, str(src), str(dst), str(k + 1)], input=x.tobytes(),
                             capture_output=True, check=True).stdout
        got = np.frombuffer(out, np.float32)
        if len(got) != len(want):
            bad += 1
            print('FAIL %d -> %d: %d samples, torchaudio %d' % (src, dst, len(got), len(want)))
            continue
        err = float(np.abs(got - want).max())
        worst = max(worst, err)
        if err > 2e-6:
            bad += 1
            print('FAIL %d -> %d: max difference %.3g' % (src, dst, err))
    if bad:
        print('rscheck: %d of %d rate pairs wrong' % (bad, len(PAIRS)))
        return 1
    print('rscheck: %d rate pairs, lengths equal and at most %.2g from torchaudio' % (len(PAIRS), worst))
    return 0


if __name__ == '__main__':
    sys.exit(main())
