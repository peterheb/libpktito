/* The language of a transcript, from its commonest words.
 *
 * Parakeet TDT v3 writes 25 European languages and does not say which one
 * it heard, and the captioner's English rules (core/captions.c) are for
 * English alone. Each of the 25 has its commonest words in core/langwords.h,
 * which tools/langwords.py counted in FLEURS's transcripts. A transcript's
 * words that are among them count for those languages, each by its weight:
 * the fewer of the 25 share a word, the more it says (100 log2(25 / k) for a
 * word common to k: "the" 464, "a" 118). The language with the most, by a
 * margin, is the transcript's; too few words, or no clear winner, is no
 * answer, and the English rules stay off as they were.
 *
 * Integers throughout, so a transcript has the same language on every
 * machine. tests/langcheck.c has the measurements that set LANG_MIN_WORDS
 * and the margin.
 */
#include "pktito.h"

#include <stdint.h>
#include <string.h>

#include "langwords.h"

#define LANG_N   (sizeof PK_LANG_CODES / sizeof *PK_LANG_CODES)
#define WORD_MAX 64              /* bytes of a normalized word: a longer one is no common word */

#ifndef LANG_MIN_WORDS
#define LANG_MIN_WORDS 10        /* fewer words than this: no answer */
#endif
#ifndef LANG_MARGIN
#define LANG_MARGIN 150          /* the winner's score, as a percentage of the next one's at least */
#endif

/* A capital letter's small one, for the letters the 25 write; any other code
 * point as it is. tools/langwords.py's fold, rule for rule. */
static uint32_t fold(uint32_t c) {
    if ((c >= 0x41 && c <= 0x5A) || (c >= 0xC0 && c <= 0xDE && c != 0xD7))
        return c + 0x20;                                          /* ASCII, Latin-1 */
    if (((c >= 0x100 && c <= 0x12F) || (c >= 0x132 && c <= 0x137) || (c >= 0x14A && c <= 0x177) ||
         (c >= 0x218 && c <= 0x21B)) && c % 2 == 0)
        return c + 1;                                             /* Latin Extended-A pairs, Ș Ț */
    if (((c >= 0x139 && c <= 0x148) || (c >= 0x179 && c <= 0x17E)) && c % 2 == 1)
        return c + 1;
    if (c == 0x178)
        return 0xFF;                                              /* Ÿ */
    if (c >= 0x391 && c <= 0x3A9 && c != 0x3A2)
        return c + 0x20;                                          /* Greek */
    if (c == 0x386)
        return 0x3AC;
    if (c >= 0x388 && c <= 0x38A)
        return c + 0x25;
    if (c == 0x38C)
        return 0x3CC;
    if (c >= 0x38E && c <= 0x38F)
        return c + 0x3F;
    if (c >= 0x410 && c <= 0x42F)
        return c + 0x20;                                          /* Cyrillic */
    if (c >= 0x400 && c <= 0x40F)
        return c + 0x50;
    if (c == 0x490)
        return 0x491;                                             /* Ukrainian's Ґ */
    return c;
}

/* the punctuation a word's ends lose: ASCII's, and the quotes, dashes and
 * marks the 25 write */
static int punct(uint32_t c) {
    return (c < 0x80 && c > 0x20 && strchr("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~", (int)c)) || c == 0xA1 ||
           c == 0xAB || c == 0xB7 || c == 0xBB || c == 0xBF || c == 0x2026 || c == 0x2039 || c == 0x203A ||
           (c >= 0x2010 && c <= 0x2015) || (c >= 0x2018 && c <= 0x201F);
}

/* the code point at s, of n bytes, and its length; a byte that starts no
 * well-formed UTF-8 sequence is itself, one long */
static uint32_t utf8(const unsigned char *s, size_t n, size_t *len) {
    uint32_t c = s[0];
    size_t k = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 0;
    if (k <= 1 || k > n) {
        *len = 1;
        return c;
    }
    c &= 0x7F >> k;
    for (size_t i = 1; i < k; i++) {
        if ((s[i] & 0xC0) != 0x80) {
            *len = 1;
            return s[0];
        }
        c = c << 6 | (s[i] & 0x3F);
    }
    *len = k;
    return c;
}

static size_t put8(char *out, uint32_t c) {
    if (c < 0x80) { out[0] = (char)c; return 1; }
    if (c < 0x800) { out[0] = (char)(0xC0 | c >> 6); out[1] = (char)(0x80 | (c & 0x3F)); return 2; }
    if (c < 0x10000) {
        out[0] = (char)(0xE0 | c >> 12); out[1] = (char)(0x80 | (c >> 6 & 0x3F)); out[2] = (char)(0x80 | (c & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | c >> 18); out[1] = (char)(0x80 | (c >> 12 & 0x3F));
    out[2] = (char)(0x80 | (c >> 6 & 0x3F)); out[3] = (char)(0x80 | (c & 0x3F));
    return 4;
}

/* word w as the table holds words, in out[WORD_MAX]: its ends' punctuation
 * off and its capitals small; its length, 0 if nothing is left or it will
 * not fit */
static size_t normal(const char *w, char *out) {
    const unsigned char *s = (const unsigned char *)w;
    size_t n = strlen(w), a = 0, b = n, len;
    while (a < b && punct(utf8(s + a, b - a, &len)))
        a += len;
    /* the last code point: back over its continuation bytes */
    while (a < b) {
        size_t start = b - 1;
        while (start > a && (s[start] & 0xC0) == 0x80 && b - start < 4)
            start--;
        if (!punct(utf8(s + start, b - start, &len)) || start + len != b)
            break;
        b = start;
    }
    size_t o = 0;
    for (size_t i = a; i < b; i += len) {
        uint32_t c = fold(utf8(s + i, b - i, &len));
        if (o + 4 >= WORD_MAX)
            return 0;
        o += put8(out + o, c);
    }
    out[o] = 0;
    return o;
}

/* the table's entry for a normalized word, or -1 */
static long entry(const char *w) {
    size_t lo = 0, hi = sizeof PK_LANG_WORDS / sizeof *PK_LANG_WORDS;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        int c = strcmp(PK_LANG_WORDS[mid].w, w);
        if (!c)
            return (long)mid;
        if (c < 0) lo = mid + 1; else hi = mid;
    }
    return -1;
}

/* Each language's score for words [0, n), into score[LANG_N], weighted or
 * every common word 100; the number of words, punctuation alone not one. */
static size_t scores(const pk_capword *w, size_t n, int weighted, int64_t *score) {
    memset(score, 0, LANG_N * sizeof *score);
    size_t words = 0;
    char buf[WORD_MAX];
    for (size_t i = 0; i < n; i++) {
        if (!w[i].text || !normal(w[i].text, buf))
            continue;
        words++;
        long e = entry(buf);
        if (e < 0)
            continue;
        for (size_t l = 0; l < LANG_N; l++)
            if (PK_LANG_WORDS[e].langs >> l & 1)
                score[l] += weighted ? PK_LANG_WORDS[e].weight : 100;
    }
    return words;
}

/* the winner of score[], if it wins: at least LANG_MIN_WORDS words and
 * LANG_MARGIN percent of the runner-up's score */
static int winner(const int64_t *score, size_t words, int64_t margin) {
    size_t best = 0, next = 1;
    if (score[next] > score[best]) best = 1, next = 0;
    for (size_t l = 2; l < LANG_N; l++)
        if (score[l] > score[best]) next = best, best = l;
        else if (score[l] > score[next]) next = l;
    if (words < LANG_MIN_WORDS || score[best] <= 0 || score[best] * 100 < score[next] * margin)
        return -1;
    return (int)best;
}

const char *pk_language(const pk_capword *w, size_t n) {
    int64_t score[LANG_N];
    int l = winner(score, scores(w, n, 1, score), LANG_MARGIN);
    return l < 0 ? NULL : PK_LANG_CODES[l];
}
