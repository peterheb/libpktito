"""The recordings the accuracy measurements run over, as 16 kHz 16-bit mono
WAVs with their reference transcripts beside them.

  corpus.py librispeech TEST_CLEAN_DIR OUT [--per-speaker N]
  corpus.py fleurs LANG.dev.tsv LANG.dev.tar.gz OUT [--count N]
  corpus.py longform TEST_CLEAN_DIR OUT [--seconds S] [--count N]

LibriSpeech test-clean (openslr.org/12) is 2620 read English utterances by
40 speakers; --per-speaker takes that many from each, spread over their
chapters, so a subset still hears every voice. FLEURS (google/fleurs on
HuggingFace) is read Wikipedia sentences in 102 languages, a few hundred an
hour for each; its dev split is the smaller download. Each OUT gets the WAVs
and refs.tsv: one line a recording, its file name and its transcript.

longform makes recordings of about S seconds (300) out of LibriSpeech
chapters: each chapter's utterances in order, 0.3 s of silence between
them (the corpus cut each chapter at its pauses and kept none of them), up
to N recordings from as many speakers - for measuring what cutting a long
recording at its silences costs against hearing it whole.

LibriSpeech is FLAC, which ffmpeg decodes (or afconvert on macOS); FLEURS
is WAV already and needs nothing.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tarfile


def _flac_to_wav(src, dst):
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-ac", "1", "-ar", "16000",
               "-sample_fmt", "s16", dst]
    elif shutil.which("afconvert"):
        cmd = ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", src, dst]
    else:
        sys.exit("decoding FLAC needs ffmpeg or afconvert")
    subprocess.run(cmd, check=True)


def librispeech(root, out, per_speaker):
    os.makedirs(out, exist_ok=True)
    refs = []
    for spk in sorted(os.listdir(root), key=lambda s: int(s) if s.isdigit() else 0):
        d = os.path.join(root, spk)
        if not os.path.isdir(d):
            continue
        utts = []
        for ch in sorted(os.listdir(d)):
            trans = os.path.join(d, ch, "%s-%s.trans.txt" % (spk, ch))
            with open(trans, encoding="utf-8") as f:
                for line in f:
                    name, _, text = line.strip().partition(" ")
                    utts.append((os.path.join(d, ch, name + ".flac"), name, text))
        if per_speaker:
            step = max(1, len(utts) // per_speaker)
            utts = utts[::step][:per_speaker]
        for src, name, text in utts:
            _flac_to_wav(src, os.path.join(out, name + ".wav"))
            refs.append((name + ".wav", text))
    _write_refs(out, refs)


def _read_wav16(path):
    import wave
    with wave.open(path) as w:
        return w.readframes(w.getnframes())


def longform(root, out, seconds, count):
    import tempfile
    import wave
    os.makedirs(out, exist_ok=True)
    refs = []
    gap = b"\0\0" * 4800                                      # 0.3 s at 16 kHz
    tmp = tempfile.mkdtemp()
    for spk in sorted(os.listdir(root), key=lambda s: int(s) if s.isdigit() else 0):
        if len(refs) >= count:
            break
        d = os.path.join(root, spk)
        if not os.path.isdir(d):
            continue
        ch = sorted(os.listdir(d))[0]
        trans = os.path.join(d, ch, "%s-%s.trans.txt" % (spk, ch))
        with open(trans, encoding="utf-8") as f:
            utts = [line.strip().partition(" ") for line in f]
        pcm, words = bytearray(), []
        for name, _, text in utts:
            wav = os.path.join(tmp, name + ".wav")
            _flac_to_wav(os.path.join(d, ch, name + ".flac"), wav)
            pcm += _read_wav16(wav) + gap
            words.append(text)
            if len(pcm) >= seconds * 32000:
                break
        name = "%s-%s.wav" % (spk, ch)
        with wave.open(os.path.join(out, name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(bytes(pcm))
        refs.append((name, " ".join(words)))
    shutil.rmtree(tmp)
    _write_refs(out, refs)


def fleurs(tsv, tar, out, count):
    os.makedirs(out, exist_ok=True)
    rows = {}
    with open(tsv, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) >= 3:
                rows[cols[1]] = cols[2]          # the file name and its raw transcription
    names = sorted(rows)
    if count:
        step = max(1, len(names) // count)
        names = names[::step][:count]
    want = set(names)
    refs = []
    with tarfile.open(tar, "r:gz") as t:
        for m in t:
            base = os.path.basename(m.name)
            if m.isfile() and base in want:
                with open(os.path.join(out, base), "wb") as f:
                    f.write(t.extractfile(m).read())
                refs.append((base, rows[base]))
    _write_refs(out, sorted(refs))


def _write_refs(out, refs):
    with open(os.path.join(out, "refs.tsv"), "w", encoding="utf-8") as f:
        for name, text in refs:
            f.write("%s\t%s\n" % (name, text))
    print("%s: %d recordings" % (out, len(refs)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("librispeech")
    a.add_argument("root")
    a.add_argument("out")
    a.add_argument("--per-speaker", type=int, default=0)
    c = sub.add_parser("longform")
    c.add_argument("root")
    c.add_argument("out")
    c.add_argument("--seconds", type=float, default=300)
    c.add_argument("--count", type=int, default=12)
    b = sub.add_parser("fleurs")
    b.add_argument("tsv")
    b.add_argument("tar")
    b.add_argument("out")
    b.add_argument("--count", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "librispeech":
        librispeech(args.root, args.out, args.per_speaker)
    elif args.cmd == "longform":
        longform(args.root, args.out, args.seconds, args.count)
    else:
        fleurs(args.tsv, args.tar, args.out, args.count)


if __name__ == "__main__":
    main()
