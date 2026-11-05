/* sentcheck: pk_sentences, core/sentences.c.
 *
 *   tests/sentcheck MODEL WAV                the checks: exit 0 if every one holds
 *   tests/sentcheck --eval MODEL LANG WAV... the measurement (README, "Sentences")
 *
 * The checks: its rules on words built from the model's own pieces, and on
 * WAV's transcript that the sentences tile the tokens and are the same
 * whether the tokens come at once or split anywhere, passed as a caller
 * hearing a recording chunk by chunk passes them. The letters' cases and
 * vowels in each alphabet the models write, and what a word's text ends in,
 * are checked on their own.
 *
 * The measurement: each recording's sentences against a sentence at every
 * . ? ! and ellipsis, with the same 30 s breaking - pktito-app's lines
 * before this - every break where the two differ on a line, and every word
 * shaped like an abbreviation before one that starts with no small letter,
 * ended or not, with its period's frames and the pause after its letters,
 * for reading against the text. LANG is pk_sentences' lang, or - for NULL.
 *
 * Includes core/sentences.c, for its static parts. */
#include "../core/sentences.c"
#include "wav.h"

#include <math.h>
#include <stdio.h>

static int fails;
#define CHECK(c, ...)                                                                                        \
    do {                                                                                                     \
        if (!(c)) {                                                                                          \
            fails++;                                                                                         \
            printf("sentcheck: FAIL line %d: ", __LINE__);                                                   \
            printf(__VA_ARGS__);                                                                             \
            putchar('\n');                                                                                   \
        }                                                                                                    \
    } while (0)

/* ------------------------------------------------------------ the letters --- */

static void alphabets(void) {
    for (uint32_t c = 0xC0; c <= 0xDE; c++)
        if (c != 0xD7)
            CHECK(letter_case(c) == 1 && fold(c) == c + 32 && letter_case(c + 32) == -1, "Latin-1 U+%04X", c);
    for (uint32_t c = 0x100; c <= 0x17F; c++)
        if (letter_case(c) == 1 && c != 0x130 && c != 0x178)
            CHECK(fold(c) == c + 1 && letter_case(c + 1) == -1, "Latin Extended-A U+%04X", c);
    for (uint32_t c = 0x391; c <= 0x3AB; c++)
        if (c != 0x3A2)
            CHECK(letter_case(c) == 1 && fold(c) == c + 32 && letter_case(c + 32) == -1, "Greek U+%04X", c);
    for (uint32_t c = 0x400; c <= 0x42F; c++)
        CHECK(letter_case(c) == 1 && letter_case(fold(c)) == -1, "Cyrillic U+%04X", c);
    static const uint32_t pairs[][2] = { { 0x386, 0x3AC }, { 0x388, 0x3AD }, { 0x389, 0x3AE }, { 0x38A, 0x3AF },
                                         { 0x38C, 0x3CC }, { 0x38E, 0x3CD }, { 0x38F, 0x3CE }, { 0x218, 0x219 },
                                         { 0x21A, 0x21B }, { 0x490, 0x491 }, { 0x178, 0xFF }, { 0x141, 0x142 } };
    for (size_t k = 0; k < sizeof pairs / sizeof *pairs; k++)
        CHECK(letter_case(pairs[k][0]) == 1 && fold(pairs[k][0]) == pairs[k][1] && letter_case(pairs[k][1]) == -1,
              "U+%04X to U+%04X", pairs[k][0], pairs[k][1]);
    static const char *vowels = "aeiouyAÉÖőŰąĘÿΑΩάώΈаЯёЪЇї", *consonants = "bcdfrśŁżçñßΣςжьщ";
    for (const char *s = vowels, *e = s + strlen(s); s < e;) {
        uint32_t c = utf8(&s, e);
        CHECK(vowel(c), "U+%04X is a vowel", c);
    }
    for (const char *s = consonants, *e = s + strlen(s); s < e;) {
        uint32_t c = utf8(&s, e);
        CHECK(!vowel(c), "U+%04X is no vowel", c);
    }
    static const uint32_t caseless[] = { '0', '9', '.', '?', '"', 0x2026, 0x4E00, 0x5D0 };
    for (size_t k = 0; k < sizeof caseless / sizeof *caseless; k++)
        CHECK(letter_case(caseless[k]) == 0, "U+%04X has no case", caseless[k]);
}

/* ------------------------------------------------------------- the marks --- */

static void texts(void) {
    static const struct {
        const char *text;
        int mark, shape, first, part, title;
    } T[] = {
        { "Dr.", '.', NO_VOWEL, 1, 0, 0 },          { "Mrs.", '.', NO_VOWEL, 1, 0, 0 },
        { "św.", '.', NO_VOWEL, -1, 0, 0 },         { "tzw.", '.', NO_VOWEL, -1, 0, 0 },
        { "r.", '.', NO_VOWEL, -1, 0, 0 },          /* Polish: rok */
        { "smrt.", '.', NO_VOWEL, -1, 0, 0 },       /* Czech: a word, which the times decide */
        { "F.", '.', INITIAL, 1, 0, 0 },            { "M.", '.', INITIAL, 1, 0, 0 },
        { "É.", '.', INITIAL, 1, 0, 0 },            { "u.", '.', PLAIN, -1, 0, 0 },     /* Dutch: you */
        { "U.S.", '.', DOTTED, 1, 0, 0 },           { "LibriVox.org.", '.', DOTTED, 1, 0, 0 },
        { "6.", '.', NUMBER, 0, 0, 0 },             { "1990.", '.', NUMBER, 0, 0, 0 },
        { "сън.", '.', PLAIN, -1, 0, 0 },           /* Bulgarian: ъ the vowel */
        { "race?\"", '?', PLAIN, -1, 0, 0 },        { "\xE2\x80\x9EJa!\xE2\x80\x9C", '?', PLAIN, 1, 0, 0 },
        { "so...", '?', PLAIN, -1, 0, 0 },          { "so\xE2\x80\xA6", '?', PLAIN, -1, 0, 0 },
        { "\"What", 0, PLAIN, 1, 0, 0 },            { "\xC2\xBFQu\xC3\xA9", 0, PLAIN, 1, 0, 0 },
        { "Hi,", 0, PLAIN, 1, 1, 0 },               { "said:", 0, PLAIN, -1, 1, 0 },
        { "mister", 0, PLAIN, -1, 0, 1 },           { "Misses", 0, PLAIN, 1, 0, 1 },
        { "ośle", 0, PLAIN, -1, 0, 0 },             { "Łańcuch", 0, PLAIN, 1, 0, 0 },
        { "Έτσι", 0, PLAIN, 1, 0, 0 },              { "ещё", 0, PLAIN, -1, 0, 0 },
        { "Mr.\"", '.', NO_VOWEL, 1, 0, 0 },        { "(e.g.", '.', DOTTED, -1, 0, 0 },
    };
    for (size_t k = 0; k < sizeof T / sizeof *T; k++) {
        uint32_t c[64];
        size_t n = 0;
        for (const char *s = T[k].text, *e = s + strlen(s); s < e;)
            c[n++] = utf8(&s, e);
        word w;
        memset(&w, 0, sizeof w);
        marks(c, n, 1, &w);
        CHECK(w.mark == T[k].mark && (w.mark != '.' || w.shape == T[k].shape) && w.first == T[k].first
                  && w.part == T[k].part && w.title == T[k].title,
              "%s: mark %d shape %d first %d part %d title %d", T[k].text, w.mark, w.shape, w.first, w.part,
              w.title);
    }
}

/* ----------------------------------------------------- words into tokens --- */

static const pk_model *M;
static uint32_t vocab;

/* text's pieces as the model's vocabulary has them, longest first, with
 * the space that starts a word: their count, or 0 if a character has none */
static int pieces(const char *text, int32_t *out, int cap) {
    char buf[128];
    snprintf(buf, sizeof buf, "\xE2\x96\x81%s", text);
    size_t at = 0, len = strlen(buf);
    int k = 0;
    while (at < len) {
        size_t best = 0;
        int32_t id = -1;
        for (uint32_t i = 0; i < vocab; i++) {
            size_t pl;
            const char *p = pk_piece(M, (int32_t)i, &pl);
            if (pl > best && pl <= len - at && !memcmp(buf + at, p, pl))
                best = pl, id = (int32_t)i;
        }
        if (id < 0 || k == cap)
            return 0;
        out[k++] = id;
        at += best;
    }
    return k;
}

/* A word to build: its text, the frame it starts at and its letters end
 * at, and its trailing punctuation's duration */
typedef struct {
    const char *text;
    int32_t     start, said, period;
} spec;

/* the tokens of words w[0, n) into t: how many, or 0 if a word has no pieces */
static size_t build(const spec *w, size_t n, pk_tok *t, size_t cap) {
    size_t k = 0;
    for (size_t i = 0; i < n; i++) {
        int32_t id[64];
        int np = pieces(w[i].text, id, 64), nl = 0;
        if (!np || k + (size_t)np > cap)
            return 0;
        for (int q = 0; q < np; q++)
            nl += !punct_piece(M, id[q]);
        int32_t f = w[i].start;
        int l = 0, trailing = 0;
        for (int q = 0; q < np; q++) {
            t[k].id = id[q];
            if (!punct_piece(M, id[q])) {
                int32_t to = w[i].start + (w[i].said - w[i].start) * ++l / nl;
                t[k].frame = f, t[k].dur = to - f;
                f = to;
            } else if (l < nl) {
                t[k].frame = f, t[k].dur = 0;           /* an opener, or an apostrophe */
            } else {
                t[k].frame = f, t[k].dur = trailing++ ? 0 : w[i].period;
                f += t[k].dur;
            }
            k++;
        }
    }
    return k;
}

/* The words after which pk_sentences ends a sentence, as a string of word
 * numbers ("2 4"), and the flags of each ("L" for PK_SENTENCE_LONG) */
static void ends_of(const spec *w, size_t nw, const char *lang, int done, char *out, size_t cap) {
    pk_tok t[512];
    pk_sentence s[512];
    size_t n = build(w, nw, t, 512);
    out[0] = 0;
    if (!n) {
        snprintf(out, cap, "no pieces");
        return;
    }
    long k = pk_sentences(M, t, n, lang, done, s);
    size_t at = 0;
    for (long i = 0; i < k; i++) {
        size_t end = (size_t)(s[i].first + s[i].n), word = 0;
        for (size_t j = 0; j < end; j++)
            word += j > 0 && starts(M, t[j].id);
        at += (size_t)snprintf(out + at, cap - at, "%s%zu%s", i ? " " : "", word, s[i].flags & PK_SENTENCE_LONG ? "L" : "");
    }
}

static void rules(void) {
    /* Hi, I'm Dr. Jennifer Turner. */
    spec dr[] = { { "Hi,", 0, 5, 1 }, { "I'm", 6, 9, 0 }, { "Dr.", 9, 12, 1 }, { "Jennifer", 13, 19, 0 },
                  { "Turner.", 19, 25, 4 } };
    char got[256];
    ends_of(dr, 5, "en", 1, got, sizeof got);
    CHECK(!strcmp(got, "4"), "\"Dr. Jennifer\", no pause: %s", got);
    dr[2].period = 3;
    ends_of(dr, 5, "en", 1, got, sizeof got);
    CHECK(!strcmp(got, "2 4"), "\"Dr.\" with a sentence's period: %s", got);
    dr[2].period = 1, dr[3].start = dr[2].said + PAUSE, dr[3].said = dr[3].start + 6, dr[4].start = dr[3].said,
    dr[4].said = dr[4].start + 6;
    ends_of(dr, 5, "en", 1, got, sizeof got);
    CHECK(!strcmp(got, "2 4"), "\"Dr.\" and a pause of PAUSE: %s", got);

    /* What is a caucus race? said Alice. */
    spec race[] = { { "What", 0, 4, 0 }, { "is", 4, 6, 0 }, { "a", 6, 7, 0 }, { "caucus", 7, 12, 0 },
                    { "race?", 12, 16, 4 }, { "said", 22, 25, 0 }, { "Alice.", 25, 30, 4 } };
    ends_of(race, 7, "en", 1, got, sizeof got);
    CHECK(!strcmp(got, "6"), "a question and a small letter after it: %s", got);

    /* a number's period ends a sentence but where ordinals are numbers */
    spec year[] = { { "It", 0, 2, 0 }, { "was", 2, 5, 0 }, { "1990.", 5, 12, 1 }, { "Then", 13, 16, 0 },
                    { "we", 16, 18, 0 }, { "left.", 18, 22, 4 } };
    ends_of(year, 6, "en", 1, got, sizeof got);
    CHECK(!strcmp(got, "2 5"), "\"1990. Then\" in English: %s", got);
    spec date[] = { { "Er", 0, 2, 0 }, { "kam", 2, 5, 0 }, { "am", 5, 7, 0 }, { "6.", 7, 10, 2 },
                    { "Oktober", 12, 18, 0 }, { "an.", 18, 21, 4 } };
    ends_of(date, 6, "de", 1, got, sizeof got);
    CHECK(!strcmp(got, "5"), "\"am 6. Oktober\" in German: %s", got);
    ends_of(date, 6, "en", 1, got, sizeof got);
    CHECK(!strcmp(got, "3 5"), "\"6. Oktober\" outside German: %s", got);

    /* the small m English's sentences start with in the models' writing */
    spec mister[] = { { "He", 0, 2, 0 }, { "was", 2, 5, 0 }, { "there.", 5, 10, 4 }, { "mister", 15, 19, 0 },
                      { "Solmes", 19, 24, 0 }, { "left.", 24, 28, 4 } };
    ends_of(mister, 6, "en", 1, got, sizeof got);
    CHECK(!strcmp(got, "2 5"), "\"there. mister Solmes\" in English: %s", got);
    ends_of(mister, 6, NULL, 1, got, sizeof got);
    CHECK(!strcmp(got, "5"), "a small m with no language: %s", got);

    /* a period inside a word: an abbreviation, or a sentence's end with its pause */
    spec us[] = { { "The", 0, 3, 0 }, { "U.S.", 3, 9, 1 }, { "Army", 10, 14, 0 }, { "came.", 14, 18, 4 } };
    ends_of(us, 4, "en", 1, got, sizeof got);
    CHECK(!strcmp(got, "3"), "\"U.S. Army\": %s", got);
    spec org[] = { { "visit", 0, 4, 0 }, { "LibriVox.org.", 4, 14, 4 }, { "Recording", 14, 20, 0 },
                   { "by", 20, 22, 0 }, { "me.", 22, 25, 4 } };
    ends_of(org, 5, "en", 1, got, sizeof got);
    CHECK(!strcmp(got, "1 4"), "\"LibriVox.org. Recording\": %s", got);

    /* the cost: a sentence that ends in an initial, said on without a pause */
    spec plan[] = { { "plan", 0, 4, 0 }, { "B.", 4, 7, 1 }, { "Then", 8, 12, 0 }, { "we", 12, 14, 0 },
                    { "went.", 14, 18, 4 } };
    ends_of(plan, 5, "en", 1, got, sizeof got);
    CHECK(!strcmp(got, "4"), "\"plan B. Then\" with no pause: %s", got);
    plan[1].period = 4;
    ends_of(plan, 5, "en", 1, got, sizeof got);
    CHECK(!strcmp(got, "1 4"), "\"plan B. Then\" with a sentence's period: %s", got);

    /* a word past the 64 bytes describe's buffers start with */
    spec big[] = { { "Internationalizationinternationalizationinternationalizationinternationalization.", 0, 40, 4 },
                   { "Then", 44, 48, 0 }, { "we", 48, 50, 0 }, { "went.", 50, 54, 4 } };
    ends_of(big, 4, "en", 1, got, sizeof got);
    CHECK(!strcmp(got, "0 3"), "a word of 81 bytes: %s", got);

    /* more to come: the last words wait, the word after them unknown */
    spec open[] = { { "Hi.", 0, 3, 4 }, { "There", 7, 11, 0 }, { "you", 11, 13, 0 }, { "are.", 13, 16, 4 } };
    ends_of(open, 4, "en", 0, got, sizeof got);
    CHECK(!strcmp(got, "0"), "done 0: %s", got);
    ends_of(open, 1, "en", 0, got, sizeof got);
    CHECK(!strcmp(got, ""), "done 0, one word: %s", got);
    ends_of(open, 1, "en", 1, got, sizeof got);
    CHECK(!strcmp(got, "0"), "done 1, one word: %s", got);

    /* past 30 s: a part of 10 to 20 s after its last comma, else at its
     * longest pause, the first word's pause a candidate as in pktito-app */
    spec run[40];
    for (int i = 0; i < 40; i++)
        run[i] = (spec){ "go", i * 12, i * 12 + 12, 0 };
    run[14].text = "go,", run[14].period = 1;
    run[39].text = "go.", run[39].period = 4;
    ends_of(run, 40, "en", 1, got, sizeof got);
    CHECK(!strcmp(got, "14L 39"), "past 30 s, at a comma: %s", got);
    run[14] = (spec){ "go", 14 * 12, 14 * 12 + 12, 0 };
    for (int i = 13; i < 40; i++)
        run[i].start += 5, run[i].said += 5;
    ends_of(run, 40, "en", 1, got, sizeof got);
    CHECK(!strcmp(got, "12L 39"), "past 30 s, at the longest pause: %s", got);
    for (int i = 1; i < 40; i++)
        run[i].start += 7, run[i].said += 7;
    ends_of(run, 40, "en", 1, got, sizeof got);
    CHECK(!strcmp(got, "0L 12L 39"), "past 30 s, the first word's pause the longest: %s", got);
}

/* ------------------------------------------------ split anywhere, the same --- */

/* pk_sentences on t[0, n) at once, against the same tokens split at k: the
 * first part's sentences, then the rest from where they end */
static void streaming(const pk_tok *t, size_t n, const char *lang, const char *what) {
    pk_sentence *all = malloc(n * sizeof *all), *s = malloc(n * sizeof *s);
    long na = pk_sentences(M, t, n, lang, 1, all);
    CHECK(na > 0, "%s: no sentences", what);
    size_t at = 0;
    for (long i = 0; i < na; i++) {
        CHECK(all[i].first == (int32_t)at && all[i].n > 0, "%s: sentence %ld does not follow the last", what, i);
        at += (size_t)all[i].n;
    }
    CHECK(at == n, "%s: the sentences end at token %zu of %zu", what, at, n);
    int bad = 0;
    for (size_t k = 1; k < n && !bad; k++) {
        long got = 0, n1 = pk_sentences(M, t, k, lang, 0, s);
        size_t from = n1 > 0 ? (size_t)(s[n1 - 1].first + s[n1 - 1].n) : 0;
        for (long i = 0; i < n1 && !bad; i++, got++)
            bad = got >= na || memcmp(&s[i], &all[got], sizeof *s);
        long n2 = pk_sentences(M, t + from, n - from, lang, 1, s);
        for (long i = 0; i < n2 && !bad; i++, got++) {
            s[i].first += (int32_t)from;
            bad = got >= na || memcmp(&s[i], &all[got], sizeof *s);
        }
        bad |= got != na;
        CHECK(!bad, "%s: split at token %zu, not the sentences at once", what, k);
    }
    free(all);
    free(s);
}

typedef struct {
    pk_tok *t;
    size_t  n, cap;
} heard;

static int on_chunk(void *arg, const pk_chunk_out *c) {
    heard *h = arg;
    for (long i = 0; i < c->count; i++) {
        if (h->n == h->cap) {
            h->cap = h->cap ? 2 * h->cap : 1024;
            h->t = realloc(h->t, h->cap * sizeof *h->t);
        }
        h->t[h->n] = c->toks[i];
        h->t[h->n++].frame += c->at_frame;
    }
    return 0;
}

/* The transcript of the WAV at path into h, its frames counted from the
 * recording's start; its length in seconds, or 0 if it has none */
static double hear(pk_asr *a, const char *path, heard *h) {
    char err[256];
    pk_wav_info wi;
    pk_wav *w = pk_wav_open(path, &wi, err, sizeof err);
    float *x = w ? malloc((size_t)wi.frames * sizeof *x) : NULL;
    size_t n = x ? pk_wav_read(w, x, (size_t)wi.frames) : 0;
    pk_wav_close(w);
    pk_transcribe_opts o;
    pk_transcribe_defaults(&o);
    h->n = 0;
    int ok = n && pk_asr_transcribe(a, x, n, &o, on_chunk, h) == 0 && h->n;
    free(x);
    return ok ? (double)n / pk_model_rate(M) : 0;
}

/* -------------------------------------------------------------- --eval --- */

/* pk_sentences' loop with an end at every mark: in f, at each sentence's
 * end, 1, or 2 where one is broken at 30 s */
static void plain(sent *s, const word *W, size_t nw, unsigned char *f) {
    size_t s0 = 0;
    int32_t start = W[0].start;
    for (size_t i = 0; i < nw; i++) {
        if (W[i].end - start > LONG && s0 < W[i].a) {
            size_t at;
            if (part_end(s, s0, W[i].b, start, &at))
                exit(2);
            f[at] = 2;
            s0 = at;
            start = s->t[at].frame;
        }
        if (i + 1 == nw || W[i].mark) {
            f[W[i].b] = 1;
            s0 = W[i].b;
            start = i + 1 < nw ? W[i + 1].start : 0;
        }
    }
}

/* the text of tokens [a, b) */
static const char *text_of(const pk_tok *t, size_t a, size_t b) {
    static char buf[2][1024];
    static int k;
    k ^= 1;
    pk_text(M, t + a, b - a, buf[k], sizeof buf[k]);
    return buf[k];
}

static void eval(pk_model *m, const char *lang, char **wavs, int nwav) {
    pk_asr *a = pk_asr_new(m, 0);
    heard h = { 0 };
    long n_plain = 0, n_pk = 0, small_q = 0, small_p = 0, abbr = 0, gone = 0, added = 0, other = 0;
    long shaped = 0, shaped_ended = 0;
    double seconds = 0;
    for (int r = 0; r < nwav; r++) {
        const double len = a ? hear(a, wavs[r], &h) : 0;
        if (!len) {
            fprintf(stderr, "sentcheck: %s: no transcript\n", wavs[r]);
            exit(2);
        }
        seconds += len;
        const char *name = strrchr(wavs[r], '/') ? strrchr(wavs[r], '/') + 1 : wavs[r];
        sent s = { .m = m, .t = h.t, .n = h.n, .english = lang && !strcmp(lang, "en"),
                   .ordinals = lang && !strcmp(lang, "de"), .cap = 64 };
        word *W = malloc(h.n * sizeof *W);
        pk_sentence *out = malloc(h.n * sizeof *out);
        unsigned char *pf = calloc(h.n + 1, 1), *qf = calloc(h.n + 1, 1);
        if (!(s.text = malloc(s.cap)) || !(s.cp = malloc(s.cap * sizeof *s.cp)) || !W || !out || !pf || !qf)
            exit(2);
        size_t nw = 0;
        for (size_t at = 0; at < h.n; at = W[nw++].b)
            if (describe(&s, at, &W[nw]))
                exit(2);
        plain(&s, W, nw, pf);
        const long k = pk_sentences(m, h.t, h.n, lang, 1, out);
        if (k < 0)
            exit(2);
        for (long i = 0; i < k; i++)
            qf[out[i].first + out[i].n] = out[i].flags & PK_SENTENCE_LONG ? 2 : 1;
        for (size_t i = 0; i < nw; i++) {
            const word *w = &W[i], *x = i + 1 < nw ? &W[i + 1] : NULL;
            const int p = pf[w->b], q = qf[w->b];
            n_plain += p != 0, n_pk += q != 0;
            const int small = x && x->first < 0 && !x->title;
            const int abbreviation = x && !small && w->mark == '.' && w->shape != PLAIN
                                     && (w->shape != NUMBER || s.ordinals);
            const char *kind = NULL;
            if (p == 1 && !q)
                kind = small ? (w->mark == '?' ? (small_q++, "joined, ? ! before a small letter")
                                               : (small_p++, "joined, . before a small letter"))
                             : abbreviation ? (abbr++, "joined, an abbreviation's shape") : (other++, "joined");
            else if (p != q)
                kind = p == 2 ? (gone++, "30 s break gone") : q == 2 ? (added++, "30 s break new") : (other++, "other");
            if (abbreviation) {
                shaped++, shaped_ended += q != 0;
                if (!kind)
                    kind = "ended, an abbreviation's shape";
            }
            if (kind)
                printf("%s\tperiod %d\tpause %d\t%s\t%s | %s\n", kind, w->period, x ? x->start - w->said : 0, name,
                       text_of(h.t, W[i > 3 ? i - 3 : 0].a, w->b),
                       x ? text_of(h.t, x->a, W[i + 3 < nw ? i + 3 : nw - 1].b) : "");
        }
        free(s.text), free(s.cp), free(W), free(out), free(pf), free(qf);
    }
    printf("sentcheck --eval: %d recordings, %.2f h: %ld sentences at every mark, %ld by pk_sentences\n", nwav,
           seconds / 3600, n_plain, n_pk);
    printf("  joined: %ld at ? ! before a small letter, %ld at . before one, %ld at an abbreviation's period\n",
           small_q, small_p, abbr);
    if (other)
        printf("  other: %ld\n", other);
    printf("  30 s breaks: %ld new, %ld gone\n", added, gone);
    printf("  shaped like an abbreviation, before no small letter: %ld, %ld of them ended\n", shaped, shaped_ended);
    pk_asr_free(a);
    free(h.t);
}

int main(int argc, char **argv) {
    if (argc >= 5 && !strcmp(argv[1], "--eval")) {
        char err[256];
        pk_model *m = pk_model_load(argv[2], err, sizeof err);
        if (!m) {
            fprintf(stderr, "sentcheck: %s\n", err);
            return 2;
        }
        M = m;
        eval(m, strcmp(argv[3], "-") ? argv[3] : NULL, argv + 4, argc - 4);
        pk_model_free(m);
        return 0;
    }
    if (argc != 3) {
        fprintf(stderr, "usage: sentcheck MODEL WAV, or sentcheck --eval MODEL LANG WAV...\n");
        return 2;
    }
    alphabets();
    texts();
    char err[256];
    pk_model *m = pk_model_load(argv[1], err, sizeof err);
    if (!m) {
        printf("sentcheck: FAIL - %s\n", err);
        return 1;
    }
    M = m;
    pk_model_info info;
    pk_model_describe(m, &info);
    vocab = info.vocab;
    rules();

    spec run[60];
    pk_tok t[512];
    for (int i = 0; i < 60; i++)
        run[i] = (spec){ i % 9 == 8 ? "go." : i % 5 == 4 ? "go," : i % 9 == 0 ? "Go" : "go", i * 12, i * 12 + 10,
                         i % 9 == 8 ? 4 : 1 };
    size_t nt = build(run, 60, t, 512);
    CHECK(nt > 0, "no pieces for \"go\"");
    if (nt)
        streaming(t, nt, "en", "60 words with commas and periods");
    for (int i = 0; i < 60; i++)
        run[i] = (spec){ i == 59 ? "go." : "go", i * 12 + (i % 7) * 2, i * 12 + (i % 7) * 2 + 9, 4 };
    nt = build(run, 60, t, 512);
    if (nt)
        streaming(t, nt, "en", "a sentence of 60 words, broken twice");

    pk_asr *a = pk_asr_new(m, 0);
    heard h = { 0 };
    CHECK(a && hear(a, argv[2], &h), "%s: no transcript", argv[2]);
    if (h.n) {
        streaming(h.t, h.n, "en", argv[2]);
        pk_sentence *s = malloc(h.n * sizeof *s);
        long k = pk_sentences(m, h.t, h.n, "en", 1, s);
        char text[4096];
        for (long i = 0; i < k; i++) {
            pk_text(m, h.t + s[i].first, (size_t)s[i].n, text, sizeof text);
            printf("sentcheck: %s, sentence %ld: %s\n", argv[2], i + 1, text);
        }
        free(s);
    }
    pk_asr_free(a);
    free(h.t);
    pk_model_free(m);
    if (fails)
        return 1;
    printf("sentcheck: every case holds\n");
    return 0;
}
