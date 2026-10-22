/* capcheck: core/captions.c held to docs/caption-spec.md's own cases (its
 * section 11): a candidate cue's layout and times on the spec's fixtures,
 * the boundary rules, graphemes, the greedy cut it must not make, and the
 * dynamic program against every partition of 3000 random transcripts of up
 * to 10 words, the spec's exhaustive oracle, with the properties every
 * track keeps.
 *
 *   tests/capcheck     exit 0 if every case holds
 *
 * It includes core/captions.c, for its parts that are static. */
#include "captions.c"

#include <stdio.h>
#include <time.h>

static int fails;
#define CHECK(c, ...)                                                                                        \
    do {                                                                                                     \
        if (!(c)) {                                                                                          \
            fails++;                                                                                         \
            printf("capcheck: FAIL line %d: ", __LINE__);                                                    \
            printf(__VA_ARGS__);                                                                             \
            putchar('\n');                                                                                   \
        }                                                                                                    \
    } while (0)

/* a graph of at most MAXW words, prepared as pk_captions prepares one */
#define MAXW 16
typedef struct {
    graph   g;
    int64_t s[MAXW + 1], e[MAXW + 1], gsum[MAXW + 1], csum[MAXW + 1];
    int32_t b[MAXW + 1];
    uint8_t safe[MAXW + 1], cut[MAXW + 1];
} tgraph;

static void make(tgraph *t, const pk_capword *w, int n, int english, int64_t media_end) {
    memset(t, 0, sizeof *t);
    t->g = (graph){ .n = n, .w = w, .s = t->s, .e = t->e, .gsum = t->gsum, .b = t->b, .csum = t->csum,
                    .safe = t->safe, .cut = t->cut };
    prepare(&t->g, english, media_end);
}

/* the candidate cue words [i, j), as the dynamic program weighs it */
static edge cand(const tgraph *t, int i, int j) {
    int64_t a = 0;
    int whole = 1;
    for (int k = i; k < j; k++)
        a = max64(a, t->e[k]);
    for (int k = i + 1; k < j; k++)
        if (t->safe[k])
            whole = 0;
    return cue(&t->g, i, j, a, whole);
}

/* ------------------------------------------------------------- 11.1, 11.2 --- */

static void fixtures(void) {
    tgraph t;
    pk_cue c[MAXW];

    /* basic phrase: one cue 1000-2000, "Hello world.", 12 characters (the
     * spec writes its words text,start,end) */
    pk_capword basic[] = { { "Hello", 1000, 1300 }, { "world.", 1400, 1800 } };
    long n = pk_captions(basic, 2, 1, 4000, c);
    CHECK(n == 1 && c[0].start == 1000 && c[0].end == 2000 && c[0].chars == 12 && c[0].line2 == 0 && !c[0].flags,
          "basic phrase: %ld cues, %lld-%lld, %d characters", n, (long long)c[0].start, (long long)c[0].end, c[0].chars);

    /* a required pause: two cues 1000-2000 and 3000-4000 */
    pk_capword pause[] = { { "Hello.", 1000, 1600 }, { "Again.", 3000, 3600 } };
    n = pk_captions(pause, 2, 1, 5000, c);
    CHECK(n == 2 && c[0].start == 1000 && c[0].end == 2000 && c[1].start == 3000 && c[1].end == 4000,
          "required pause: %ld cues", n);

    /* a short reply on its own: the minimum-duration tail, 800 ms after speech */
    pk_capword yes[] = { { "Yes.", 1000, 1200 } };
    n = pk_captions(yes, 1, 1, 4000, c);
    CHECK(n == 1 && c[0].start == 1000 && c[0].end == 2000 && !c[0].flags, "short reply: %lld-%lld",
          (long long)c[0].start, (long long)c[0].end);
    /* at the recording's end: 1000-1300, too short, no overrun */
    n = pk_captions(yes, 1, 1, 1300, c);
    CHECK(n == 1 && c[0].end == 1300 && c[0].flags == PK_CUE_SHORT, "short final reply: %lld, flags %d",
          (long long)c[0].end, c[0].flags);

    /* rapid speech: no readable track, every word kept; the three as one cue
     * are 32 characters in 450 ms, 71.1 a second */
    pk_capword rapid[] = { { "abcdefghij", 0, 150 }, { "abcdefghij", 150, 300 }, { "abcdefghij", 300, 450 } };
    n = pk_captions(rapid, 3, 1, 450, c);
    int kept = 0, slow = 1;
    for (long k = 0; k < n; k++) {
        kept += c[k].n;
        slow &= !(c[k].flags & (PK_CUE_FAST | PK_CUE_SHORT));
    }
    CHECK(n > 0 && kept == 3 && !slow, "rapid speech: %d words kept in %ld cues, all readable: %d", kept, n, slow);
    make(&t, rapid, 3, 1, 450);
    edge ed = cand(&t, 0, 3);
    CHECK(ed.ok && ed.chars == 32 && ed.end == 450 && (ed.flags & PK_CUE_FAST), "rapid speech, merged: %d characters, end %lld",
          ed.chars, (long long)ed.end);

    /* two full lines: 42 / 42, 85 characters, 0-5200; not refused by a sum of 84 */
    static char w42[43];
    memset(w42, 'a', 42);
    pk_capword full[] = { { w42, 0, 2500 }, { w42, 2500, 5000 } };
    make(&t, full, 2, 1, 6000);
    ed = cand(&t, 0, 2);
    CHECK(ed.ok && ed.line2 == 1 && ed.chars == 85 && ed.end == 5200, "two full lines: line2 %d, %d characters, end %lld",
          ed.line2, ed.chars, (long long)ed.end);

    /* pause classes: none, weak, weak, strong, strong, required */
    const int64_t gaps[] = { 249, 250, 499, 500, 999, 1000 };
    const int32_t costs[] = { 60, 40, 40, 10, 10, 10 }, crosses[] = { 0, 30, 30, 80, 80, 80 };
    for (int k = 0; k < 6; k++) {
        pk_capword two[] = { { "x", 0, 100 }, { "y", 100 + gaps[k], 200 + gaps[k] } };
        make(&t, two, 2, 0, 0);
        CHECK(t.b[1] == costs[k] && t.csum[1] == crosses[k] && t.cut[1] == (gaps[k] >= 1000),
              "pause of %lld ms: cost %d, cross %lld, cut %d", (long long)gaps[k], t.b[1], (long long)t.csum[1], t.cut[1]);
    }

    /* no detector: the word gap is the pause, 1600 ms, required */
    pk_capword gonow[] = { { "Go", 0, 200 }, { "now.", 1800, 2200 } };
    make(&t, gonow, 2, 1, 0);
    CHECK(t.cut[1], "Go now.: the 1600 ms gap is a required cut");

    /* an overlap chain: no cut after A or B, a cut after C; A-C's audio ends at 2000 */
    pk_capword chain[] = { { "A", 0, 2000 }, { "B", 500, 700 }, { "C", 1000, 1300 }, { "D", 2100, 2400 } };
    make(&t, chain, 4, 1, 0);
    ed = cand(&t, 0, 3);
    CHECK(!t.safe[1] && !t.safe[2] && t.safe[3] && ed.ok && ed.end == 2100, "overlap chain: safe %d %d %d, A-C ends %lld",
          t.safe[1], t.safe[2], t.safe[3], (long long)ed.end);

    /* touching ends: a boundary; a millisecond's overlap: none */
    pk_capword touch[] = { { "A", 0, 300 }, { "B", 300, 600 } };
    make(&t, touch, 2, 1, 0);
    CHECK(t.safe[1], "touching ends: a boundary");
    touch[0].end = 301;
    make(&t, touch, 2, 1, 0);
    CHECK(!t.safe[1], "overlapping ends: no boundary");

    /* a 43-grapheme word: no layout holds it; a cue of its own, forced, whole */
    static char w43[44];
    memset(w43, 'a', 43);
    pk_capword longw[] = { { w43, 0, 1000 } };
    n = pk_captions(longw, 1, 1, 0, c);
    CHECK(n == 1 && c[0].n == 1 && c[0].chars == 43 && (c[0].flags & PK_CUE_FORCED), "a 43-grapheme word: forced");

    /* 30, 30, 20: 82 in all, yet 30 / 51 and 61 / 20 fit no layout */
    static char w30[31], w20[21];
    memset(w30, 'a', 30);
    memset(w20, 'a', 20);
    pk_capword lengths[] = { { w30, 0, 1000 }, { w30, 1000, 2000 }, { w20, 2000, 3000 } };
    make(&t, lengths, 3, 0, 0);
    CHECK(!cand(&t, 0, 3).ok, "30, 30, 20: no cue holds all three");
    n = pk_captions(lengths, 3, 0, 0, c);
    CHECK(n >= 2 && c[0].first == 0 && c[n - 1].first + c[n - 1].n == 3 && !(c[0].flags & PK_CUE_FORCED),
          "30, 30, 20: %ld cues", n);

    /* graphemes: a combining accent, a ZWJ family, flags */
    CHECK(graphemes("cafe\xCC\x81") == 4, "cafe + U+0301: %d", graphemes("cafe\xCC\x81"));
    CHECK(graphemes("\xF0\x9F\x91\xA8\xE2\x80\x8D\xF0\x9F\x91\xA9\xE2\x80\x8D\xF0\x9F\x91\xA7\xE2\x80\x8D\xF0\x9F\x91\xA6") == 1,
          "a ZWJ family is one");
    CHECK(graphemes("\xF0\x9F\x87\xAB\xF0\x9F\x87\xB7\xF0\x9F\x87\xA9\xF0\x9F\x87\xAA") == 2, "two flags are two");
    static char acc42[42 * 3 + 1], acc43[43 * 3 + 1];
    for (int k = 0; k < 43; k++)
        memcpy(acc43 + 3 * k, "e\xCC\x81", 3);
    memcpy(acc42, acc43, 42 * 3);
    pk_capword a42[] = { { acc42, 0, 3000 } }, a43[] = { { acc43, 0, 3000 } };
    n = pk_captions(a42, 1, 0, 0, c);
    CHECK(n == 1 && c[0].chars == 42 && !(c[0].flags & PK_CUE_FORCED), "42 accented e's fit a line");
    n = pk_captions(a43, 1, 0, 0, c);
    CHECK(n == 1 && c[0].chars == 43 && (c[0].flags & PK_CUE_FORCED), "43 do not");

    /* punctuation: an abbreviation, an initial and a decimal end no sentence; a question does */
    CHECK(mark("Dr.", 1) == P_NONE && mark("J.", 1) == P_NONE && mark("3.14", 1) == P_NONE, "Dr. J. 3.14");
    CHECK(mark("3.14.", 1) == P_TERM && mark("Really?", 1) == P_TERM && mark("end.\xE2\x80\x9D", 1) == P_TERM,
          "3.14. Really? end.\" end sentences");
    CHECK(mark("Dr.", 0) == P_TERM, "in another language, no list of abbreviations");
    CHECK(mark("wait...", 1) == P_MAJOR && mark("so\xE2\x80\xA6", 1) == P_MAJOR && mark("this\xE2\x80\x94", 1) == P_MAJOR,
          "an ellipsis or a dash: a major break");

    /* each boundary's costs (the spec's 5.3): a cut at it, B, and crossing it */
    static const struct { const char *before, *after; int english; int64_t gap; int32_t b, cross; } costs53[] = {
        { "end.", "Next", 1, 0, 0, 60 },      /* a sentence's end */
        { "wait;", "next", 1, 0, 10, 15 },    /* a major break */
        { "so,", "next", 1, 0, 20, 0 },       /* a comma */
        { "budgie", "and", 1, 0, 25, 0 },     /* before a phrase's start */
        { "went", "to", 1, 0, 25, 0 },        /* "to" starting one */
        { "going", "to", 1, 0, 60, 0 },       /* and ending a complex verb */
        { "budgie", "and", 0, 0, 60, 0 },     /* which only English knows */
        { "budgie", "audio", 1, 0, 60, 0 },   /* nothing */
        { "the", "audio", 1, 0, 360, 0 },     /* after a function word */
        { "end.", "Next", 1, 600, 0, 140 },   /* a sentence's end at a strong pause */
        { "so,", "next", 1, 300, 0, 30 },     /* a comma at a weak one */
    };
    for (size_t k = 0; k < sizeof costs53 / sizeof *costs53; k++) {
        pk_capword two[] = { { costs53[k].before, 0, 300 }, { costs53[k].after, 300 + costs53[k].gap, 600 + costs53[k].gap } };
        make(&t, two, 2, costs53[k].english, 0);
        CHECK(t.b[1] == costs53[k].b && t.csum[1] == costs53[k].cross, "%s | %s: B %d, cross %lld, not %d and %d",
              costs53[k].before, costs53[k].after, t.b[1], (long long)t.csum[1], costs53[k].b, costs53[k].cross);
    }

    /* a candidate's scores by hand: "Yes. No." in 600 ms, the recording's
     * end at its last word: V 4 x ceil(400^2 / 1000) = 640 for its 400 ms
     * short of a second; Q 40 + 60 to cross the sentence's end + 4 x
     * ceil(600 / 100) = 124, short of the easy 1200 ms */
    pk_capword yesno[] = { { "Yes.", 0, 300 }, { "No.", 300, 600 } };
    make(&t, yesno, 2, 1, 0);
    ed = cand(&t, 0, 2);
    CHECK(ed.ok && ed.end == 600 && ed.v == 640 && ed.q == 124, "Yes. No.: end %lld, V %lld, Q %lld",
          (long long)ed.end, (long long)ed.v, (long long)ed.q);

    /* test.wav's cue at 51 s: "to" is the rest of "going to", and the line
     * breaks before the object, 24 / 23, by 60 + 1 + 2 = 63 against 60 + 7
     * for 11 / 36 before "to" */
    pk_capword tax[] = { { "Who's", 51030, 51350 }, { "going", 51350, 51510 }, { "to", 51510, 51590 },
                         { "implement", 51590, 52070 }, { "a", 52070, 52230 }, { "4%", 52230, 52630 },
                         { "global", 52790, 53350 }, { "wealth", 53350, 53910 }, { "tax?", 53910, 54710 } };
    make(&t, tax, 9, 1, 0);
    ed = cand(&t, 0, 9);
    CHECK(ed.ok && ed.line2 == 4 && ed.chars == 48, "Who's going to implement / a 4%% global wealth tax?: line 2 from word %d",
          ed.line2);

    /* the pause marker: cut at a pause in mid-sentence, a cue is flagged
     * for its ellipsis, which it counts; not after punctuation, at a pause
     * too short to cut at, or at the recording's end */
    pk_capword soi[] = { { "So", 54710, 55030 }, { "I", 55030, 55350 }, { "when", 56550, 56800 }, { "I", 56800, 56900 },
                         { "think.", 56900, 57400 } };
    n = pk_captions(soi, 5, 1, 60000, c);
    CHECK(n == 2 && c[0].n == 2 && (c[0].flags & PK_CUE_PAUSE) && c[0].chars == 5 && !(c[1].flags & PK_CUE_PAUSE),
          "So I | when I think.: %ld cues, flags %d and %d, %d characters", n, c[0].flags, n > 1 ? c[1].flags : 0, c[0].chars);
    soi[1].text = "I,";
    n = pk_captions(soi, 5, 1, 60000, c);
    CHECK(n == 2 && !(c[0].flags & PK_CUE_PAUSE), "So I, | when: no ellipsis after a comma");
    soi[1].text = "I";
    soi[2].start = 56300;
    n = pk_captions(soi, 5, 1, 60000, c);
    CHECK(!(c[0].flags & PK_CUE_PAUSE), "a pause under a second is not cut at: no ellipsis");
    pk_capword tail[] = { { "and", 0, 300 }, { "then", 300, 700 } };
    n = pk_captions(tail, 2, 1, 5000, c);
    CHECK(n == 1 && !(c[0].flags & PK_CUE_PAUSE), "the recording's end: no ellipsis");
    /* 42 graphemes before the pause fit a line, but not with the ellipsis */
    pk_capword w42p[] = { { w42, 0, 2500 }, { "next.", 4000, 4500 } };
    make(&t, w42p, 2, 1, 6000);
    ed = cand(&t, 0, 1);
    CHECK(ed.ok && ed.chars == 43 && (ed.flags & PK_CUE_FORCED) && (ed.flags & PK_CUE_PAUSE),
          "a 42-grapheme word cut at a pause: 43 with its ellipsis, forced, flags %d", ed.flags);
    w42p[0].text = "a";
    make(&t, w42p, 2, 1, 6000);
    ed = cand(&t, 0, 1);
    CHECK(ed.ok && ed.chars == 2 && (ed.flags & PK_CUE_PAUSE), "a word cut at a pause: 2 characters, flagged");

    /* English: no cue ends on "the" */
    pk_capword store[] = { { "I", 0, 200 }, { "went", 200, 500 }, { "to", 500, 600 }, { "the", 600, 700 }, { "store.", 700, 1100 } };
    make(&t, store, 5, 1, 0);
    CHECK(t.b[4] >= 300, "English: a cut after \"the\" costs %d", t.b[4]);
    make(&t, store, 5, 0, 0);
    CHECK(t.b[4] == 60, "another language: a cut after \"the\" costs %d", t.b[4]);

    /* gap closing: a gap of 249 ms to the next cue is closed, 250 is not */
    for (int64_t gap = 249; gap <= 250; gap++) {
        pk_capword hi[] = { { "Hi.", 0, 800 }, { "There.", 1000 + gap, 1500 + gap } };
        make(&t, hi, 2, 1, 0);
        ed = cand(&t, 0, 1);
        CHECK(ed.ok && ed.end == (gap < 250 ? 1000 + gap : 1000), "a gap of %lld: ends %lld", (long long)gap, (long long)ed.end);
    }

    /* unsound times are made sound, not refused: a track still */
    pk_capword bad[] = { { "b", 500, 400 }, { "a", 300, 300 }, { "c", 900, 1200 } };
    n = pk_captions(bad, 3, 1, 0, c);
    CHECK(n >= 1 && c[n - 1].first + c[n - 1].n == 3, "unsound times: %ld cues", n);
    CHECK(pk_captions(bad, 0, 1, 0, c) == 0, "no words: no cues");
}

/* ------------------------------------------------------------------ 11.3 --- */

static void greedy(void) {
    pk_capword w[9];
    for (int k = 0; k < 9; k++)
        w[k] = (pk_capword){ "abcdefgh", 500 * k, 500 * k + 500 };
    pk_cue c[9];
    long n = pk_captions(w, 9, 1, 4500, c);
    tgraph t;
    make(&t, w, 9, 1, 4500);
    int64_t v = 0, q = 0;
    for (long k = 0; k < n; k++) {
        edge ed = cand(&t, c[k].first, c[k].first + c[k].n);
        v += ed.v;
        q += ed.q;
    }
    /* the spec's 11.6: 4 + 5 by (0, 211, 2), which ties 5 + 4 and wins it by
     * its earlier first boundary; greedy packing's 8 + 1 flashes a word */
    CHECK(n == 2 && c[0].n == 4 && c[1].n == 5 && v == 0 && q == 211,
          "nine words: %ld cues, %d + %d, V %lld, Q %lld", n, c[0].n, n > 1 ? c[1].n : 0, (long long)v, (long long)q);
}

/* ------------------------------------------------------------- 11.4, 11.5 --- */

static uint64_t rng = 0x9E3779B97F4A7C15u;
static uint32_t rnd(uint32_t m) {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (uint32_t)(rng % m);
}

/* the best path's score, by every partition: (V, Q, count, then ends) */
typedef struct { int64_t v, q; int count; int ends[MAXW]; } path;

static int less(const path *a, const path *b) {
    if (a->v != b->v) return a->v < b->v;
    if (a->q != b->q) return a->q < b->q;
    if (a->count != b->count) return a->count < b->count;
    for (int k = 0; k < a->count; k++)
        if (a->ends[k] != b->ends[k])
            return a->ends[k] < b->ends[k];
    return 0;
}

static int oracle(const tgraph *t, int n, path *best) {
    int found = 0;
    for (uint32_t cuts = 0; cuts < (1u << (n - 1)); cuts++) {
        path p = { 0 };
        int i = 0, ok = 1;
        for (int j = 1; j <= n && ok; j++) {
            if (j < n && !(cuts >> (j - 1) & 1))
                continue;
            if (j < n && !t->safe[j]) {
                ok = 0;
                break;
            }
            for (int k = i + 1; k < j; k++)
                if (t->cut[k])
                    ok = 0;
            if (!ok)
                break;
            edge ed = cand(t, i, j);
            if (!ed.ok) {
                ok = 0;
                break;
            }
            p.v += ed.v;
            p.q += ed.q;
            p.ends[p.count++] = j;
            i = j;
        }
        if (ok && (!found || less(&p, best))) {
            *best = p;
            found = 1;
        }
    }
    return found;
}

static const char *const VOCAB[] = { "the", "to", "and", "of", "Dr.", "J.", "wait...", "yes.", "no,", "really?",
                                     "budgie", "transcribes", "audio", "while", "which", "abcdefghijklmnop",
                                     "x", "caf\xC3\xA9", ("\xC3\xBC" "ber"), "q;" };

static void random_tracks(void) {
    char text[MAXW][24];
    pk_capword w[MAXW];
    pk_cue c[MAXW];
    tgraph t;
    int cases = 0;
    for (int run = 0; run < 3000; run++) {
        const int n = 1 + (int)rnd(10), english = (int)rnd(2);
        int64_t at = rnd(500);
        for (int k = 0; k < n; k++) {
            snprintf(text[k], sizeof text[k], "%s", VOCAB[rnd(sizeof VOCAB / sizeof *VOCAB)]);
            /* gaps of 0 to 1.3 s, a few overlapping the word before */
            int64_t gap = rnd(6) == 0 ? -(int64_t)rnd(400) : (int64_t)rnd(1300);
            if (k)
                at += gap;
            const int64_t start = at < 0 ? 0 : at, len = 30 + rnd(rnd(4) ? 700 : 3000);
            w[k] = (pk_capword){ text[k], start, start + len };
            at = start + len;
        }
        const int64_t media_end = rnd(3) ? 0 : at + rnd(2000);
        long nc = pk_captions(w, (size_t)n, english, media_end, c);
        make(&t, w, n, english, media_end);
        path best;
        if (!oracle(&t, n, &best)) {
            CHECK(0, "run %d: the oracle found no path", run);
            continue;
        }
        path dp = { 0 };
        dp.count = (int)nc;
        for (long k = 0; k < nc; k++) {
            edge ed = cand(&t, c[k].first, c[k].first + c[k].n);
            dp.v += ed.v;
            dp.q += ed.q;
            dp.ends[k] = c[k].first + c[k].n;
        }
        CHECK(!less(&best, &dp) && !less(&dp, &best), "run %d (%d words): the program's (%lld, %lld, %d) is not the best (%lld, %lld, %d)",
              run, n, (long long)dp.v, (long long)dp.q, dp.count, (long long)best.v, (long long)best.q, best.count);

        /* the properties of a track (11.5) */
        int next = 0;
        for (long k = 0; k < nc; k++) {
            int64_t audio = 0;
            for (int x = c[k].first; x < c[k].first + c[k].n; x++)
                audio = max64(audio, t.e[x]);
            CHECK(c[k].first == next && c[k].n > 0, "run %d: cue %ld does not follow on", run, k);
            CHECK(c[k].start == t.s[c[k].first] && c[k].end >= audio && c[k].end > c[k].start,
                  "run %d: cue %ld, %lld-%lld, does not cover its words", run, k, (long long)c[k].start, (long long)c[k].end);
            CHECK(k == 0 || c[k - 1].end <= c[k].start, "run %d: cues %ld and %ld overlap", run, k - 1, k);
            const int last = c[k].first + c[k].n;
            for (int x = c[k].first + 1; x < last; x++)
                CHECK(!t.cut[x], "run %d: cue %ld spans a required cut", run, k);
            if (!(c[k].flags & PK_CUE_FORCED)) {
                int64_t l1 = c[k].line2 ? width(&t.g, c[k].first, c[k].line2) : width(&t.g, c[k].first, last);
                int64_t l2 = c[k].line2 ? width(&t.g, c[k].line2, last) : 0;
                CHECK(l1 <= LINE && l2 <= LINE, "run %d: cue %ld's lines are %lld and %lld", run, k, (long long)l1, (long long)l2);
            }
            next = last;
        }
        CHECK(next == n, "run %d: %d of %d words in cues", run, next, n);
        cases++;
    }
    printf("capcheck: %d random transcripts, the program's track the best of every partition\n", cases);
}

/* a long transcript, timed */
static void long_track(void) {
    enum { N = 20000 };
    static pk_capword w[N];
    static pk_cue c[N];
    int64_t at = 0;
    for (int k = 0; k < N; k++) {
        const int64_t len = 120 + rnd(400);
        w[k] = (pk_capword){ VOCAB[rnd(sizeof VOCAB / sizeof *VOCAB)], at, at + len };
        at += len + (rnd(8) ? rnd(150) : rnd(1500));
    }
    clock_t t0 = clock();
    long n = pk_captions(w, N, 1, 0, c);
    double s = (double)(clock() - t0) / CLOCKS_PER_SEC;
    CHECK(n > 0 && c[n - 1].first + c[n - 1].n == N, "20000 words: %ld cues", n);
    printf("capcheck: 20000 words to %ld cues in %.0f ms\n", n, s * 1000);
}

int main(void) {
    fixtures();
    greedy();
    random_tracks();
    long_track();
    if (fails) {
        printf("capcheck: %d FAILED\n", fails);
        return 1;
    }
    printf("capcheck: ok\n");
    return 0;
}
