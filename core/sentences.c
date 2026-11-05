/* Sentences from a transcript's tokens: where its lines end.
 *
 * The models write cased, punctuated text, so a sentence ends where they put
 * . ? ! or an ellipsis - at a word ending in one, before any closing quote
 * or bracket - except where that mark ends nothing:
 *
 *   - the next word starts with a small letter: the mark closes a question
 *     or a cry the sentence goes on past ("What is a caucus race?" said
 *     Alice), or is an abbreviation's ("e.g. the");
 *   - the word is shaped like an abbreviation - one capital ("Roger W.
 *     Barnett", "M. Fogg"), a period inside it ("U.S."), letters and no
 *     vowel ("Dr.", "Mrs.", "św."), and in German a number, there an
 *     ordinal ("am 6. Oktober") - and nothing in the times says a sentence
 *     ended: TDT gives the period under PERIOD frames, and the next word
 *     starts within PAUSE of the end of the word's letters.
 *
 * Both read the words' letters and times, not a language's list of
 * abbreviations, so they hold in each alphabet the models write (Latin,
 * Greek, Cyrillic), all of whose letters have a case and whose words have
 * vowels. German's ordinals, and the small m of "mister" and "misses" that
 * the models write at a sentence's start in English, are the only rules of
 * one language's.
 *
 * The times are TDT's. A sentence's period is given the pause after it, up
 * to its 4 frames, so it is the period's duration that says a sentence
 * ended, and the gap after the word's end says little; an abbreviation's
 * period lasts 1 or 2 frames, with the word it stands for said after it
 * ("M." 0.08 to 0.16 s, then "onsieur" to "Fogg" 0.16 to 0.32 s). Of the
 * 114 such words before no small letter in the README's sets, read against
 * their texts, the 76 inside a sentence had a period of 1 or 2 frames but 4,
 * and a pause of 0.48 s at most; PERIOD and PAUSE get 5 of the 114 wrong, the
 * pause alone 10 at its best (README, "Sentences"; tests/sentcheck --eval).
 *
 * And a sentence that runs past LONG is broken where it reads best:
 * pktito-app's rule, measured there on 10.3 h of LibriVox in seven
 * languages (pktito-app 2899b83 and 223d209, 2026-11-03 and 04), which
 * these lines are the same as but where the rules above differ.
 */
#include "pktito.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PERIOD   3      /* encoder frames, 0.24 s */
#define PAUSE    7      /* 0.56 s */
#define LONG     375    /* 30 s: a sentence longer than this is broken */
#define PART_MIN 125    /* into a first part of 10 */
#define PART_MAX 250    /* to 20 s */

/* ----------------------------------------------------------------- letters --- */

static uint32_t utf8(const char **s, const char *end) {
    const unsigned c = (unsigned char)*(*s)++;
    const int more = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : c >= 0xC0 ? 1 : 0;
    uint32_t cp = more ? c & (0x3Fu >> more) : c;
    for (int k = 0; k < more && *s < end && ((unsigned char)**s & 0xC0) == 0x80; k++)
        cp = cp << 6 | ((unsigned char)*(*s)++ & 0x3F);
    return cp;
}

/* c's case in the alphabets the models write - Latin's blocks as far as
 * their 25 languages use them, Greek's and Cyrillic's: 1 a capital, -1 a
 * small letter, 0 neither (a digit, a mark, another script) */
static int letter_case(uint32_t c) {
    if (c >= 'A' && c <= 'Z')
        return 1;
    if (c >= 'a' && c <= 'z')
        return -1;
    if (c >= 0xC0 && c <= 0xFF && c != 0xD7 && c != 0xF7)
        return c <= 0xDE ? 1 : -1;
    if (c >= 0x100 && c <= 0x17F) {
        if (c == 0x130 || c == 0x178)                       /* İ Ÿ */
            return 1;
        if (c == 0x131 || c == 0x138 || c == 0x149 || c == 0x17F)
            return -1;                                      /* ı ĸ ŉ ſ */
        const uint32_t odd = (c >= 0x139 && c <= 0x148) || c >= 0x179;   /* capitals odd */
        return (c & 1) == odd ? 1 : -1;
    }
    if (c >= 0x218 && c <= 0x21B)                           /* Ș ș Ț ț */
        return c & 1 ? -1 : 1;
    if (c == 0x386 || (c >= 0x388 && c <= 0x38F && c != 0x38B && c != 0x38D)
        || (c >= 0x391 && c <= 0x3AB && c != 0x3A2))
        return 1;
    if (c == 0x390 || (c >= 0x3AC && c <= 0x3CE))
        return -1;
    if (c >= 0x400 && c <= 0x45F)
        return c <= 0x42F ? 1 : -1;
    if (c >= 0x490 && c <= 0x4BF)                           /* Ґ ґ and their block's pairs */
        return c & 1 ? -1 : 1;
    return 0;
}

/* c in lower case, for the letters letter_case knows */
static uint32_t fold(uint32_t c) {
    if (letter_case(c) != 1)
        return c;
    if (c <= 'Z' || c <= 0xDE || (c >= 0x391 && c <= 0x3AB) || (c >= 0x410 && c <= 0x42F))
        return c + 32;
    if (c >= 0x400 && c <= 0x40F)
        return c + 80;
    if (c == 0x386)
        return 0x3AC;
    if (c >= 0x388 && c <= 0x38A)
        return c + 37;
    if (c == 0x38C)
        return 0x3CC;
    if (c == 0x38E || c == 0x38F)
        return c + 63;
    if (c == 0x178)
        return 0xFF;
    return c + 1;               /* İ to ı, the rest pairs */
}

/* The vowels of the models' alphabets, in lower case: Bulgarian's ъ among
 * them, the vowel of "път" and "сън" */
static const uint32_t VOWELS[] = {
    'a', 'e', 'i', 'o', 'u', 'y',
    0xE0, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE8, 0xE9, 0xEA, 0xEB, 0xEC, 0xED, 0xEE, 0xEF,
    0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD, 0xFF,
    0x101, 0x103, 0x105, 0x113, 0x115, 0x117, 0x119, 0x11B, 0x129, 0x12B, 0x12D, 0x12F, 0x131,
    0x14D, 0x14F, 0x151, 0x153, 0x169, 0x16B, 0x16D, 0x16F, 0x171, 0x173, 0x177,
    0x390, 0x3AC, 0x3AD, 0x3AE, 0x3AF, 0x3B0, 0x3B1, 0x3B5, 0x3B7, 0x3B9, 0x3BF, 0x3C5, 0x3C9,
    0x3CA, 0x3CB, 0x3CC, 0x3CD, 0x3CE,
    0x430, 0x435, 0x438, 0x43E, 0x443, 0x44A, 0x44B, 0x44D, 0x44E, 0x44F, 0x451, 0x454, 0x456, 0x457,
};

static int vowel(uint32_t c) {
    c = fold(c);
    for (size_t k = 0; k < sizeof VOWELS / sizeof *VOWELS; k++)
        if (VOWELS[k] == c)
            return 1;
    return 0;
}

/* a closing quote or bracket, which may follow a word's mark */
static int closer(uint32_t c) {
    return c == '"' || c == '\'' || c == ')' || c == ']' || c == '}' || c == 0x2019 || c == 0x201D
           || c == 0x201C || c == 0xBB || c == 0x203A;   /* ’ ” “ (German's) » › */
}

/* an opening one, or Spanish's ¿ ¡, which may come before a word's letters */
static int opener(uint32_t c) {
    return c == '"' || c == '\'' || c == '(' || c == '[' || c == '{' || c == 0x2018 || c == 0x201C
           || c == 0x201E || c == 0xAB || c == 0x2039 || c == 0xBF || c == 0xA1;
}

/* punctuation of the kinds the models write, which a piece may be alone */
static int punct(uint32_t c) {
    return (c < 0x80 && c > ' ' && !(c >= '0' && c <= '9') && !(c >= 'A' && c <= 'Z') && !(c >= 'a' && c <= 'z'))
           || closer(c) || opener(c) || c == 0x2026 || c == 0x2013 || c == 0x2014 || c == 0xB7
           || c == 0x37E;
}

/* ------------------------------------------------------------------- words --- */

typedef struct {
    const pk_model *m;
    const pk_tok   *t;
    size_t          n;
    int             english, ordinals;
    char           *text;
    uint32_t       *cp;
    size_t          cap;
} sent;

enum { PLAIN, INITIAL, DOTTED, NO_VOWEL, NUMBER };

typedef struct {
    size_t  a, b;                   /* its tokens, [a, b) */
    int32_t start, end, said;       /* frames: its first token's, its last's end, its letters' end */
    int32_t period;                 /* the duration of the token holding its last period */
    int     mark;                   /* what it ends in: 0, '.', '?' for ? ! and an ellipsis */
    int     shape;                  /* a period's word: PLAIN or one of an abbreviation's shapes */
    int     part;                   /* it ends in , ; : or a dash */
    int     first;                  /* its first letter's case, or 0 */
    int     title;                  /* English's "mister" or "misses" */
} word;

static int starts(const pk_model *m, int32_t id) {
    size_t len;
    const char *s = pk_piece(m, id, &len);
    return len >= 3 && !memcmp(s, "\xE2\x96\x81", 3);
}

/* a piece with no letter or digit in it, past its space */
static int punct_piece(const pk_model *m, int32_t id) {
    size_t len;
    const char *s = pk_piece(m, id, &len), *e = s + len;
    if (len >= 3 && !memcmp(s, "\xE2\x96\x81", 3))
        s += 3;
    if (s == e)
        return 0;
    while (s < e)
        if (!punct(utf8(&s, e)))
            return 0;
    return 1;
}

/* What word w's text, its code points c[0, nc), ends in, and the case its
 * letters start with */
static void marks(const uint32_t *c, size_t nc, int english, word *w) {
    size_t o = 0, e = nc;           /* past the openers, before the closers */
    while (o < nc && opener(c[o]))
        o++;
    while (e > o && closer(c[e - 1]))
        e--;
    for (size_t k = o; k < nc; k++)
        if (letter_case(c[k]) || (c[k] >= '0' && c[k] <= '9')) {
            w->first = letter_case(c[k]);
            break;
        }
    if (english && e - o == 6) {
        char low[7];
        for (size_t k = 0; k < 6; k++)
            low[k] = (char)(fold(c[o + k]) < 0x80 ? fold(c[o + k]) : '?');
        low[6] = 0;
        w->title = !strcmp(low, "mister") || !strcmp(low, "misses");
    }
    if (e == o)
        return;
    const uint32_t last = c[e - 1];
    w->part = last == ',' || last == ';' || last == ':' || last == 0x2014 || last == 0x2013;
    if (last == '?' || last == '!' || last == 0x2026 || (last == '.' && e - o >= 2 && c[e - 2] == '.')) {
        w->mark = '?';
        return;
    }
    if (last != '.')
        return;
    w->mark = '.';
    /* the word before its period */
    size_t letters = 0, vowels = 0, digits = 0, dots = 0;
    for (size_t k = o; k < e - 1; k++) {
        letters += letter_case(c[k]) != 0;
        vowels += vowel(c[k]);
        digits += c[k] >= '0' && c[k] <= '9';
        dots += c[k] == '.';
    }
    const size_t core = e - 1 - o;
    if (core && digits == core)
        w->shape = NUMBER;
    else if (dots)
        w->shape = DOTTED;
    else if (core == 1 && letter_case(c[o]) == 1)
        w->shape = INITIAL;
    else if (letters && !vowels)
        w->shape = NO_VOWEL;
}

/* The word whose first token is a: its tokens, times and text's marks.
 * 0, or -1 if memory runs out. */
static int describe(sent *s, size_t a, word *w) {
    const pk_tok *t = s->t;
    size_t b = a + 1;
    while (b < s->n && !starts(s->m, t[b].id))
        b++;
    memset(w, 0, sizeof *w);
    w->a = a, w->b = b;
    w->start = t[a].frame;
    w->end = w->said = t[b - 1].frame + t[b - 1].dur;
    for (size_t k = b; k-- > a;)
        if (!punct_piece(s->m, t[k].id)) {
            w->said = t[k].frame + t[k].dur;
            break;
        }
    for (size_t k = b; k-- > a;) {
        size_t len;
        const char *p = pk_piece(s->m, t[k].id, &len);
        if (memchr(p, '.', len)) {
            w->period = t[k].dur;
            break;
        }
    }
    size_t len;
    while ((len = pk_text(s->m, t + a, b - a, s->text, s->cap)) >= s->cap) {
        size_t cap = 2 * s->cap;
        char *text = realloc(s->text, cap);
        if (!text)
            return -1;
        s->text = text;
        uint32_t *cp = realloc(s->cp, cap * sizeof *cp);
        if (!cp)
            return -1;
        s->cp = cp, s->cap = cap;
    }
    size_t nc = 0;
    for (const char *p = s->text, *e = s->text + len; p < e;)
        s->cp[nc++] = utf8(&p, e);
    marks(s->cp, nc, s->english, w);
    return 0;
}

/* Whether a sentence ends at word w, with x after it */
static int ends(const sent *s, const word *w, const word *x) {
    if (!w->mark)
        return 0;
    if (x->first < 0 && !x->title)
        return 0;
    if (w->mark == '.' && w->shape != PLAIN && (w->shape != NUMBER || s->ordinals)
        && w->period < PERIOD && x->start - w->said < PAUSE)
        return 0;
    return 1;
}

/* A sentence from token a0 on, which ran past LONG at the word ending at
 * token b, from frame start: the token its second part starts at, in *at -
 * after the last word ending in , ; : or a dash whose part runs PART_MIN to
 * PART_MAX, else after the word before the longest pause in that span, the
 * later of two the same (pktito-app's Open::part_end, whose first word's
 * pause is a candidate wherever it lies). 0, or -1 if memory runs out. */
static int part_end(sent *s, size_t a0, size_t b, int32_t start, size_t *at) {
    word w, x;
    if (describe(s, a0, &w))
        return -1;
    size_t marked = 0, best = w.b, best_all = w.b;
    int32_t pause_best = INT32_MIN, pause_all = INT32_MIN;
    int spans = 0;
    for (size_t i = 0; w.b < b; i++, w = x) {
        if (describe(s, w.b, &x))
            return -1;
        const int span = w.end - start >= PART_MIN && w.end - start <= PART_MAX;
        const int32_t pause = x.start - w.end;
        if (span && w.part)
            marked = w.b;
        if ((span || i == 0) && pause >= pause_best)
            best = w.b, pause_best = pause;
        if (pause >= pause_all)
            best_all = w.b, pause_all = pause;
        spans += span;
    }
    *at = marked ? marked : spans ? best : best_all;
    return 0;
}

long pk_sentences(const pk_model *m, const pk_tok *t, size_t n, const char *lang, int done,
                  pk_sentence *out) {
    sent s = { .m = m, .t = t, .n = n, .english = lang && !strcmp(lang, "en"),
               .ordinals = lang && !strcmp(lang, "de"), .cap = 64 };
    long k = 0;
    word w, x;
    if (!n)
        return 0;
    if (!(s.text = malloc(s.cap)) || !(s.cp = malloc(s.cap * sizeof *s.cp)) || describe(&s, 0, &w))
        goto oom;
    size_t s0 = 0;              /* the open sentence's first token */
    int32_t start = w.start;
    for (;;) {
        if (w.end - start > LONG && s0 < w.a) {
            size_t at;
            if (part_end(&s, s0, w.b, start, &at))
                goto oom;
            out[k++] = (pk_sentence){ (int32_t)s0, (int32_t)(at - s0), PK_SENTENCE_LONG };
            s0 = at;
            start = t[at].frame;
        }
        if (w.b == n) {
            if (done)
                out[k++] = (pk_sentence){ (int32_t)s0, (int32_t)(n - s0), 0 };
            break;
        }
        if (describe(&s, w.b, &x))
            goto oom;
        if (ends(&s, &w, &x)) {
            out[k++] = (pk_sentence){ (int32_t)s0, (int32_t)(w.b - s0), 0 };
            s0 = w.b;
            start = x.start;
        }
        w = x;
    }
    free(s.text);
    free(s.cp);
    return k;
oom:
    free(s.text);
    free(s.cp);
    return -1;
}
