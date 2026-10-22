"""Each language's commonest words, for core/language.c: the 25 languages
Parakeet TDT v3 writes, counted in FLEURS's training transcripts.

  langwords.py FLEURS_TEXT_DIR [--top N] > core/langwords.h
  langwords.py FLEURS_TEXT_DIR --lines dev test > held.tsv

The second writes the splits named, which the table never saw, as lines of
a language's code, a tab and a transcript, for tests/langcheck --eval.

FLEURS_TEXT_DIR holds FLEURS's train.tsv for each language as
<code>.train.tsv (en_us.train.tsv, ...), from huggingface.co/datasets/google/fleurs
(data/<code>/train.tsv). A transcript is the third column, as it was
written. Its words are cut at spaces and normalized as core/language.c
normalizes a transcript's: the punctuation at either end stripped, the
capitals of the Latin, Greek and Cyrillic letters the 25 write made small
(fold(), below, is language.c's fold). Each language's N commonest words
then go into one table, sorted by their bytes, each with the languages it
is among the commonest of, as bits, and its weight: 100 log2(25 / k) for a
word among the commonest of k languages, an inverse document frequency
with the languages for documents. "the", English's alone, weighs 464; "a",
common to 12, 106; a word common to all 25 would weigh nothing.

FLEURS is CC BY 4.0, its text from FLoRes-101, CC BY-SA 4.0. What this
keeps of it is no sentence of either, only which words a language uses most,
a fact of the language: the table uses data derived from FLEURS, and says so.
"""
import argparse
import collections
import math
import os
import sys

# the 25, as the model file names them, and FLEURS's code for each
LANGS = [("bg", "bg_bg"), ("cs", "cs_cz"), ("da", "da_dk"), ("de", "de_de"), ("el", "el_gr"),
         ("en", "en_us"), ("es", "es_419"), ("et", "et_ee"), ("fi", "fi_fi"), ("fr", "fr_fr"),
         ("hr", "hr_hr"), ("hu", "hu_hu"), ("it", "it_it"), ("lt", "lt_lt"), ("lv", "lv_lv"),
         ("mt", "mt_mt"), ("nl", "nl_nl"), ("pl", "pl_pl"), ("pt", "pt_br"), ("ro", "ro_ro"),
         ("ru", "ru_ru"), ("sk", "sk_sk"), ("sl", "sl_si"), ("sv", "sv_se"), ("uk", "uk_ua")]

# the punctuation stripped from a word's ends: ASCII's, and the quotes,
# dashes and marks the 25 write
PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~") | {
    0xA1, 0xAB, 0xB7, 0xBB, 0xBF, 0x2026, 0x2039, 0x203A} | set(range(0x2010, 0x2016)) | set(range(0x2018, 0x2020))
PUNCT = {c if isinstance(c, int) else ord(c) for c in PUNCT}


def fold(c):
    """A capital letter's small one, for the letters the 25 write; any other
    code point as it is. language.c's fold, rule for rule."""
    if 0x41 <= c <= 0x5A or (0xC0 <= c <= 0xDE and c != 0xD7):
        return c + 0x20                                   # ASCII, Latin-1
    if (0x100 <= c <= 0x12F or 0x132 <= c <= 0x137 or 0x14A <= c <= 0x177 or 0x218 <= c <= 0x21B) and c % 2 == 0:
        return c + 1                                      # Latin Extended-A pairs, and Romanian's Ș Ț
    if (0x139 <= c <= 0x148 or 0x179 <= c <= 0x17E) and c % 2 == 1:
        return c + 1
    if c == 0x178:
        return 0xFF                                       # Ÿ
    if 0x391 <= c <= 0x3A9 and c != 0x3A2:
        return c + 0x20                                   # Greek
    if c == 0x386:
        return 0x3AC
    if 0x388 <= c <= 0x38A:
        return c + 0x25
    if c == 0x38C:
        return 0x3CC
    if 0x38E <= c <= 0x38F:
        return c + 0x3F
    if 0x410 <= c <= 0x42F:
        return c + 0x20                                   # Cyrillic
    if 0x400 <= c <= 0x40F:
        return c + 0x50
    if c == 0x490:
        return 0x491                                      # Ukrainian's Ґ
    return c


def normal(word):
    """a word as language.c compares it: its ends' punctuation off, folded"""
    cs = [ord(ch) for ch in word]
    while cs and cs[0] in PUNCT:
        cs.pop(0)
    while cs and cs[-1] in PUNCT:
        cs.pop()
    return "".join(chr(fold(c)) for c in cs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--lines", nargs="+", metavar="SPLIT")
    args = ap.parse_args()
    if args.lines:
        for code, fleurs in LANGS:
            for split in args.lines:
                with open(os.path.join(args.dir, f"{fleurs}.{split}.tsv"), encoding="utf-8") as f:
                    for line in f:
                        cols = line.rstrip("\n").split("\t")
                        if len(cols) >= 3 and cols[2].strip():
                            sys.stdout.write(f"{code}\t{cols[2].strip()}\n")
        return
    words = collections.defaultdict(int)                 # word -> bits of the languages
    totals = []
    for bit, (code, fleurs) in enumerate(LANGS):
        count = collections.Counter()
        with open(os.path.join(args.dir, f"{fleurs}.train.tsv"), encoding="utf-8") as f:
            for line in f:
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 3:
                    continue
                for w in cols[2].split():
                    n = normal(w)
                    if n and not n.isdigit():
                        count[n] += 1
        # the N commonest; ties broken by the word, so that a table is the same
        # from the same files
        for w, _ in sorted(count.items(), key=lambda kv: (-kv[1], kv[0]))[:args.top]:
            words[w] |= 1 << bit
        totals.append((code, sum(count.values())))
    entries = sorted(words.items(), key=lambda kv: kv[0].encode("utf-8"))
    out = sys.stdout
    out.write("/* core/language.c's table, written by tools/langwords.py: each of the 25\n")
    out.write(f" * languages' {args.top} commonest words in FLEURS's training transcripts\n")
    out.write(" * (huggingface.co/datasets/google/fleurs, CC BY 4.0; its text from\n")
    out.write(" * FLoRes-101, CC BY-SA 4.0), normalized as language.c's fold and normal normalize a\n")
    out.write(" * transcript's, sorted by their bytes, each with the bits of the languages\n")
    out.write(" * it is among the commonest of and its weight, 100 log2(25 / their count).\n")
    out.write(" * Words counted: " +
              ", ".join(f"{c} {n}" for c, n in totals[:6]) + ",\n")
    rest = totals[6:]
    for i in range(0, len(rest), 6):
        chunk = ", ".join(f"{c} {n}" for c, n in rest[i:i + 6])
        out.write(f" * {chunk}{'.' if i + 6 >= len(rest) else ','}\n")
    out.write(" */\n")
    out.write("static const char *const PK_LANG_CODES[] = {\n    ")
    out.write(", ".join(f'"{c}"' for c, _ in LANGS))
    out.write("\n};\n\n")
    out.write("static const struct { const char *w; uint32_t langs; int32_t weight; } PK_LANG_WORDS[] = {\n")
    for w, bits in entries:
        k = bin(bits).count("1")
        # UTF-8 past ASCII in octal escapes, which end after three digits
        # where a hex one would swallow the letters after it
        esc = "".join(ch if 0x20 <= ord(ch) < 0x7F and ch not in '"\\' else
                      "".join(f"\\{b:03o}" for b in ch.encode("utf-8")) for ch in w)
        out.write(f'    {{ "{esc}", 0x{bits:07X}, {round(100 * math.log2(len(LANGS) / k))} }},\n')
    out.write("};\n")


if __name__ == "__main__":
    main()
