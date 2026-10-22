/* chunkcheck: pk_chunks (core/asr.c) on recordings built for it.
 *
 * On 1000 random layouts of speech segments, each over audio that is
 * silent, audible throughout, and silent and audible by turns, and on the
 * cases at each rule's edge, it holds:
 *
 *   speech   the chunks that hold speech segments are the same whatever the
 *            audio, every segment in one of them, and over silent audio they
 *            are all there is: the speech is cut as it was before the gaps
 *            between it were transcribed
 *   gaps     every other chunk is one of the fewest equal pieces, of at most
 *            max_len, of the stretch between two speech chunks (or before
 *            the first, after the last), and a piece is a chunk exactly when
 *            it is 3 s or longer and audible: a 32 ms frame of it, counted
 *            from its start, with a mean square over 1e-5
 *   cover    over audible audio, the stretches no chunk covers are each
 *            shorter than 3 s
 *   short    a recording that fits in max_len is one chunk, the whole of it,
 *            if it has speech, and a gap if it has none
 *
 * The numbers - 3 s, 1e-5 over 32 ms, a sample that is not finite read as
 * 0 - are the README's ("Long recordings") and pktito.h's.
 */
#include "pktito.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define RATE    16000
#define GAP_MIN (3 * RATE)
#define LONGEST (400 * RATE)       /* a layout's recording at most */

static int fails;
static long cases;

#define FAIL(...)                                                                \
    do {                                                                         \
        if (++fails <= 20) {                                                     \
            printf("chunkcheck: FAIL ");                                         \
            printf(__VA_ARGS__);                                                 \
            putchar('\n');                                                       \
        }                                                                        \
    } while (0)

static uint64_t state = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return (uint32_t)(state >> 33);
}
static int64_t between(int64_t lo, int64_t hi) { return lo + (int64_t)(rnd() % (uint32_t)(hi - lo + 1)); }

/* some 32 ms frame of x[a, b), counted from a, has a mean square over 1e-5 */
static int audible(const float *x, int64_t a, int64_t b) {
    for (int64_t i = a; i < b; i += PK_VAD_SAMPLES) {
        int64_t e = i + PK_VAD_SAMPLES < b ? i + PK_VAD_SAMPLES : b;
        double ss = 0;
        for (int64_t j = i; j < e; j++) {
            double v = isfinite(x[j]) ? x[j] : 0;
            ss += v * v;
        }
        if (ss > 1e-5 * (double)(e - i))
            return 1;
    }
    return 0;
}

static int holds_speech(pk_seg c, const pk_seg *s, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (s[i].start >= c.start && s[i].end <= c.end)
            return 1;
    return 0;
}

/* pk_chunks over x; the chunks, their count in *nc, checked for what any
 * answer must be: in order, apart, inside the recording, none empty or
 * longer than max_len, no more than the room pktito.h promises */
static pk_seg *chunks(const float *x, const pk_seg *s, size_t n, int64_t total, int64_t max_len, size_t *nc,
                      const char *what) {
    size_t room = 2 * n + 2 + (size_t)(total / max_len);
    pk_seg *c = malloc(room * sizeof *c);
    *nc = pk_chunks(x, s, n, total, max_len, c);
    cases++;
    if (*nc > room)
        FAIL("%s: %zu chunks, room for %zu", what, *nc, room);
    for (size_t i = 0; i < *nc && i < room; i++)
        if (c[i].start < (i ? c[i - 1].end : 0) || c[i].end > total || c[i].end <= c[i].start
            || c[i].end - c[i].start > max_len)
            FAIL("%s: chunk %zu of %zu is %lld..%lld", what, i, *nc, (long long)c[i].start, (long long)c[i].end);
    return c;
}

/* The speech chunks and the gaps between them over x: c's chunks that hold
 * no segment must be exactly each gap's audible pieces of 3 s and more. */
static void gaps_are_pieces(const float *x, const pk_seg *s, size_t n, int64_t total, int64_t max_len,
                            const pk_seg *c, size_t nc, const char *what) {
    size_t k = 0;
    int64_t from = 0;
    for (size_t i = 0; i <= nc; i++) {
        if (i < nc && !holds_speech(c[i], s, n))
            continue;
        int64_t to = i < nc ? c[i].start : total;
        if (to > from) {
            int64_t m = (to - from + max_len - 1) / max_len;
            for (int64_t p = 0; p < m; p++) {
                int64_t a = from + (to - from) * p / m, b = from + (to - from) * (p + 1) / m;
                if (b - a < GAP_MIN || !audible(x, a, b))
                    continue;
                while (k < nc && holds_speech(c[k], s, n))
                    k++;
                cases++;
                if (k >= nc || c[k].start != a || c[k].end != b) {
                    FAIL("%s: the gap %lld..%lld's piece %lld..%lld is not a chunk", what, (long long)from,
                         (long long)to, (long long)a, (long long)b);
                    return;
                }
                k++;
            }
        }
        if (i < nc)
            from = c[i].end;
    }
    while (k < nc && holds_speech(c[k], s, n))
        k++;
    cases++;
    if (k != nc)
        FAIL("%s: chunk %zu, %lld..%lld, is neither speech nor a gap's piece", what, k, (long long)c[k].start,
             (long long)c[k].end);
}

/* the three recordings a layout is heard over, LONGEST samples each:
 * silence, a tone, and the tone by turns with silence (layout_case's) */
static float *quiet, *loud, *mixed;

static void layout_case(int it) {
    const int64_t max_len = it % 3 ? 30 * RATE : between(6 * RATE, 60 * RATE);
    pk_seg s[8];
    size_t n = 0;
    int64_t at = between(0, 20 * RATE);
    size_t want = rnd() % 4 == 0 ? 0 : 1 + rnd() % 6;
    while (n < want) {
        /* now and then a segment as long as a chunk allows, up to 30 s */
        int64_t most = max_len - RATE / 4 < 30 * RATE ? max_len - RATE / 4 : 30 * RATE;
        int64_t len = rnd() % 8 ? between(RATE / 4, most < 10 * RATE ? most : 10 * RATE) : between(most / 2, most);
        s[n].start = at;
        s[n].end = at + len;
        n++;
        uint32_t kind = rnd() % 3;
        at += len + (kind == 0 ? between(0, 2 * RATE) : kind == 1 ? between(2 * RATE, 10 * RATE)
                                                                  : between(10 * RATE, 30 * RATE));
    }
    const int64_t total = n ? s[n - 1].end + between(0, 20 * RATE) : between(1, 120 * RATE);
    for (int64_t i = 0; i < total;) {
        int64_t len = between(RATE / 10, 40 * RATE), on = rnd() % 2;
        for (int64_t j = i; j < i + len && j < total; j++)
            mixed[j] = on ? loud[j] : 0;
        i += len;
    }
    char what[64];
    snprintf(what, sizeof what, "layout %d", it);

    size_t nq, nl, nm;
    pk_seg *cq = chunks(quiet, s, n, total, max_len, &nq, what);
    pk_seg *cl = chunks(loud, s, n, total, max_len, &nl, what);
    pk_seg *cm = chunks(mixed, s, n, total, max_len, &nm, what);

    /* speech: over silence, only speech chunks; the same ones over any audio */
    for (size_t i = 0; i < nq; i++) {
        cases++;
        if (!holds_speech(cq[i], s, n))
            FAIL("%s: over silence, chunk %zu holds no speech", what, i);
    }
    const pk_seg *by[2] = { cl, cm };
    const size_t nby[2] = { nl, nm };
    for (int v = 0; v < 2; v++) {
        size_t j = 0;
        for (size_t i = 0; i < nby[v]; i++) {
            if (!holds_speech(by[v][i], s, n))
                continue;
            cases++;
            if (j >= nq || by[v][i].start != cq[j].start || by[v][i].end != cq[j].end) {
                FAIL("%s: speech chunk %zu is not the one over silence", what, i);
                break;
            }
            j++;
        }
        cases++;
        if (j != nq)
            FAIL("%s: %zu speech chunks, %zu over silence", what, j, nq);
    }
    for (size_t i = 0; i < n; i++) {
        size_t in = 0;
        for (size_t j = 0; j < nq; j++)
            in += s[i].start >= cq[j].start && s[i].end <= cq[j].end;
        cases++;
        if (in != 1)
            FAIL("%s: segment %zu is in %zu chunks", what, i, in);
    }

    /* gaps */
    gaps_are_pieces(loud, s, n, total, max_len, cl, nl, what);
    gaps_are_pieces(mixed, s, n, total, max_len, cm, nm, what);

    /* cover: over audible audio, nothing of 3 s or more left out */
    int64_t end = 0;
    for (size_t i = 0; i <= nl; i++) {
        int64_t next = i < nl ? cl[i].start : total;
        cases++;
        if (next - end >= GAP_MIN)
            FAIL("%s: %lld..%lld is audible and in no chunk", what, (long long)end, (long long)next);
        if (i < nl)
            end = cl[i].end;
    }
    free(cq); free(cl); free(cm);
}

/* x of total samples, every sample v, with no speech, cut at max_len: the
 * chunks expected, as start, end pairs */
static void flat_case(const char *what, int64_t total, float v, int64_t max_len, size_t want_n,
                      const int64_t *want) {
    float *x = malloc((size_t)total * sizeof *x);
    for (int64_t i = 0; i < total; i++)
        x[i] = v;
    size_t nc;
    pk_seg *c = chunks(x, NULL, 0, total, max_len, &nc, what);
    cases++;
    int ok = nc == want_n;
    for (size_t i = 0; ok && i < nc; i++)
        ok = c[i].start == want[2 * i] && c[i].end == want[2 * i + 1];
    if (!ok)
        FAIL("%s: %zu chunks, not %zu as expected", what, nc, want_n);
    free(c);
    free(x);
}

static void edge_cases(void) {
    const int64_t M = 30 * RATE;
    /* a gap of a constant: a mean square of 0.0031^2 is silent, 0.0032^2 is not */
    flat_case("20 s at 0.0031", 20 * RATE, 0.0031f, M, 0, NULL);
    flat_case("20 s at 0.0032", 20 * RATE, 0.0032f, M, 1, (const int64_t[]){ 0, 20 * RATE });
    /* the shortest piece: 3 s is one, a sample less is not */
    flat_case("3 s, audible", GAP_MIN, 0.1f, M, 1, (const int64_t[]){ 0, GAP_MIN });
    flat_case("3 s less a sample", GAP_MIN - 1, 0.1f, M, 0, NULL);
    flat_case("1 s, audible", RATE, 0.1f, M, 0, NULL);
    flat_case("20 s of digital silence", 20 * RATE, 0, M, 0, NULL);
    /* not finite is 0: silent */
    flat_case("20 s of NaN", 20 * RATE, NAN, M, 0, NULL);
    flat_case("20 s of infinity", 20 * RATE, INFINITY, M, 0, NULL);
    /* the fewest pieces, all of a length: 100 s in four of 25 */
    flat_case("100 s, audible", 100 * RATE, 0.1f, M, 4,
              (const int64_t[]){ 0, 25 * RATE, 25 * RATE, 50 * RATE, 50 * RATE, 75 * RATE, 75 * RATE, 100 * RATE });
    /* 61 s in three of 20.33 s, the cuts where integer division puts them */
    flat_case("61 s, audible", 61 * RATE, 0.1f, M, 3,
              (const int64_t[]){ 0, 61 * RATE / 3, 61 * RATE / 3, 2 * 61 * RATE / 3, 2 * 61 * RATE / 3, 61 * RATE });

    /* short: a recording that fits and has speech is one chunk, the whole
     * of it, even silent */
    float *x = calloc(20 * RATE, sizeof *x);
    pk_seg seg = { 5 * RATE, 6 * RATE }, c[4];
    size_t nc = pk_chunks(x, &seg, 1, 20 * RATE, M, c);
    cases++;
    if (nc != 1 || c[0].start != 0 || c[0].end != 20 * RATE)
        FAIL("a silent 20 s with speech: %zu chunks, not the whole of it", nc);
    free(x);
}

int main(void) {
    edge_cases();
    quiet = calloc(LONGEST, sizeof *quiet);
    loud = malloc(LONGEST * sizeof *loud);
    mixed = malloc(LONGEST * sizeof *mixed);
    for (int64_t i = 0; i < LONGEST; i++)
        loud[i] = (float)(0.1 * sin(0.05 * (double)i));
    for (int it = 0; it < 1000; it++)
        layout_case(it);
    free(quiet);
    free(loud);
    free(mixed);
    printf("chunkcheck: %ld cases, %s\n", cases, fails ? "FAIL" : "ok");
    return fails != 0;
}
