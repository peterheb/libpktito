"""Are core/hold.c's rows rewind.py's, bit for bit?
usage: same.py REF.npz FILE.wav   (REF.npz from rewind.py on FILE.wav; runs
$PKTITO_C probs -m $PKTITO_DIARIZER FILE.wav, whose rows are pk_diar_run_held's)"""

import os, subprocess, sys, numpy as np

ref = np.load(sys.argv[1])["rows"]
out = subprocess.run(
    [os.environ.get("PKTITO_C", "./pktito-c"), "probs", "-m"]
    + [os.environ["PKTITO_DIARIZER"], sys.argv[2]],
    capture_output=True,
    text=True,
    check=True,
).stdout
c = np.array(out.split(), dtype=np.float32).reshape(-1, ref.shape[1] + 1)[:, 1:]
if c.shape == ref.shape and np.array_equal(c, ref):
    print("%d rows, the same bits" % len(ref))
else:
    n = min(len(c), len(ref))
    bad = np.flatnonzero((c[:n] != ref[:n]).any(1))
    print(
        "DIFFERENT: %d and %d rows, %d differ, the first %s"
        % (len(ref), len(c), len(bad), bad[:5])
    )
    sys.exit(1)
