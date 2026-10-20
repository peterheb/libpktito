"""./pkito over corpus directories: its WER against the references, and how
its transcripts compare with another run's, float32's from
tools/quantcheck.py or another build's.

  wercheck.py PKITO MODEL.pkf CORPUS_DIR... [--against RUN.json] [--save OUT.json]

A corpus directory is tools/corpus.py's: WAVs and a refs.tsv. Each
recording is transcribed whole (--max-chunk past any of them), which is
what NeMo does with a file. --against takes quantcheck's JSON (its "f32"
run) or a --save of this script, and reports the recordings whose
transcript differs and the WER between the two runs.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from pkito_tools import wer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkito")
    ap.add_argument("model")
    ap.add_argument("corpus", nargs="+")
    ap.add_argument("--against")
    ap.add_argument("--save")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--max-chunk", default="3600", help="pkito's --max-chunk (default: whole recordings)")
    a = ap.parse_args()

    files, refs = [], {}
    for d in a.corpus:
        with open(os.path.join(d, "refs.tsv"), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    name, text = line.rstrip("\n").split("\t", 1)
                    path = os.path.join(d, name)
                    files.append(path)
                    refs[path] = text
    cmd = [a.pkito, "-m", a.model, "--max-chunk", a.max_chunk]
    if a.threads:
        cmd += ["-t", str(a.threads)]
    out = {}
    for i in range(0, len(files), 200):
        r = subprocess.run(cmd + files[i:i + 200], stdout=subprocess.PIPE, check=True)
        for line in r.stdout.decode("utf-8").splitlines():
            path, _, text = line.partition("\t")
            out[path] = text
    missing = [p for p in files if p not in out]
    if missing:
        sys.exit("no transcript for %s" % missing[:3])
    e, n = wer.wer([(refs[p], out[p]) for p in files])
    print("%s, --max-chunk %s: %d recordings, WER %.2f%% (%d/%d)" % (
        os.path.basename(a.model), a.max_chunk, len(files), 100 * e / n, e, n))
    if a.against:
        with open(a.against, encoding="utf-8") as f:
            other = json.load(f)
        other = other.get("f32", other.get("run", other))
        theirs = {r["file"]: r["text"] for r in other}
        common = [p for p in files if p in theirs or os.path.abspath(p) in theirs]
        get = lambda p: theirs.get(p, theirs.get(os.path.abspath(p)))  # noqa: E731
        diff = [p for p in common if out[p] != get(p)]
        e2, n2 = wer.wer([(get(p), out[p]) for p in common])
        e3, n3 = wer.wer([(refs[p], get(p)) for p in common])
        print("  against %s: %d of %d recordings differ, WER between the two %.2f%%; "
              "theirs against the references %.2f%%" % (os.path.basename(a.against), len(diff),
                                                         len(common), 100 * e2 / n2, 100 * e3 / n3))
    if a.save:
        with open(a.save, "w", encoding="utf-8") as f:
            json.dump({"run": [{"file": p, "text": out[p]} for p in files]}, f, ensure_ascii=False, indent=0)


if __name__ == "__main__":
    main()
