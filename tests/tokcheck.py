"""The C detokenizer (core/tokens.c) against sentencepiece and NeMo.

  tokcheck.py TOKCHECK MODEL.nemo MODEL.pkf [N]

NeMo's transcript text is sentencepiece's decode_pieces of the token
pieces with NeMo's one change after it: a whitespace character before one
of the vocabulary's punctuation marks taken away
(rnnt_decoding.py, decode_tokens_to_str_with_strip_punctuation). This
builds N random token sequences - every piece type, runs of punctuation,
the space piece alone, special pieces first - plus each piece alone, and
holds ./tests/tokcheck's text to that, character for character. It needs
the sentencepiece package (NeMo brings it); tools/pkito_tools/spm.py is
checked against it on the way.
"""
import os
import random
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from pkito_tools import convert, nemo, spm  # noqa: E402


def main():
    tok, nemo_path, pkf_path = sys.argv[1:4]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
    try:
        import sentencepiece
    except ImportError:
        print("tokcheck: no sentencepiece here - skipped")
        return 0
    nm = nemo.Nemo(nemo_path)
    blob = nm.tokenizer_model()
    sp = sentencepiece.SentencePieceProcessor(model_proto=blob)
    ours = spm.Model(blob)
    punct = convert.punctuation(ours.pieces)
    pat = re.compile(r"(\s)(" + "|".join(re.escape(p) for p in punct) + ")")
    V = sp.get_piece_size()
    rng = random.Random(1234)
    special = [i for i in range(V) if ours.types[i] != spm.NORMAL]
    space = [i for i in range(V) if ours.pieces[i] == spm.SPACE]
    pun = [i for i in range(V) if ours.pieces[i] and all(c in punct for c in ours.pieces[i])]
    seqs = [[i] for i in range(V)]
    for _ in range(n):
        L = rng.randint(1, 40)
        s = []
        for _ in range(L):
            r = rng.random()
            if r < 0.1 and special:
                s.append(rng.choice(special))
            elif r < 0.2 and space:
                s.append(rng.choice(space))
            elif r < 0.35 and pun:
                s.append(rng.choice(pun))
            else:
                s.append(rng.randrange(V))
        seqs.append(s)
    want = []
    for s in seqs:
        text = sp.decode_pieces([sp.id_to_piece(i) for i in s])
        if ours.decode(s) != text:
            print("tokcheck: spm.py differs from sentencepiece on %s: %r vs %r" % (s, ours.decode(s), text))
            return 1
        want.append(pat.sub(r"\2", text))
    inp = "".join(" ".join(map(str, s)) + "\n" for s in seqs)
    r = subprocess.run([tok, pkf_path], input=inp.encode(), stdout=subprocess.PIPE, check=True)
    got = r.stdout.decode("utf-8").split("\n")[:-1]
    bad = [(s, g, w) for s, g, w in zip(seqs, got, want) if g != w]
    if len(got) != len(want) or bad:
        for s, g, w in bad[:5]:
            print("tokcheck: FAIL %s\n   C:    %r\n   NeMo: %r" % (s, g, w))
        print("tokcheck: %d of %d differ" % (len(bad) + abs(len(got) - len(want)), len(want)))
        return 1
    print("tokcheck: %s: %d sequences (every piece alone and %d random), NeMo's text every time"
          % (os.path.basename(pkf_path), len(seqs), n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
