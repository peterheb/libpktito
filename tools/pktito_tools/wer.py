"""Word error rate, and the normalization both sides get first.

Parakeet writes cased, punctuated text ("Stuff it into you, his belly
counseled him.") and LibriSpeech's references are upper case without
punctuation, FLEURS's cased and punctuated in their own way. Both sides are
put through the same normalization, which is deliberately plain: NFKC,
lower case, punctuation and symbols to spaces (an apostrophe inside a word
stays, so "don't" is one word), whitespace collapsed. Numbers are left as
written, so "21st" against "twenty first" counts as errors; that is the same
for every build of a model and a paired comparison does not see it.
"""
import unicodedata


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    out = []
    for i, ch in enumerate(text):
        cat = unicodedata.category(ch)
        if ch == "'" and 0 < i < len(text) - 1 and text[i - 1].isalnum() and text[i + 1].isalnum():
            out.append(ch)
        elif cat[0] in "PSZC":
            out.append(" ")
        else:
            out.append(ch)
    return " ".join("".join(out).split())


def edits(ref, hyp):
    """Levenshtein distance between two word lists."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer(pairs):
    """(errors, reference words) over [(reference, hypothesis)] texts."""
    e = n = 0
    for ref, hyp in pairs:
        r, h = normalize(ref).split(), normalize(hyp).split()
        e += edits(r, h)
        n += len(r)
    return e, n
