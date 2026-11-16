/* The diarizer and what is made of its rows.  The tiny diarizer has every
 * kind of layer the real one has and hears a second of audio cheaply enough
 * for coverage-guided fuzzing.  An input's first byte picks the case:
 *
 *   engine    arbitrary float bit patterns as a recording, heard whole
 *             (pk_diar_run) and again in pieces of fuzzed sizes
 *             (pk_diar_feed, pk_diar_finish): the same rows, bit for bit,
 *             every probability in [0, 1]; then stopped by its function at
 *             a fuzzed row, and whole once more, the same rows again
 *   speakers  rows of probabilities and words of any place and length
 *             through pk_speakers and pk_speaker_segments, the second in
 *             exactly the room pktito.h asks for
 */
#include "pktito.h"

#include "fuzz.h"

#define DIAR_MAX_SAMPLES 24000
#define SPK_MAX_ROWS     512
#define SPK_MAX_WORDS    64
#define SPK_MAX          16          /* core/model.h's PK_MAX_SPK */

static pk_model *model;
static pk_diar *diar;
static size_t S;

int LLVMFuzzerInitialize(int *argc, char ***argv) {
    (void)argc;
    (void)argv;
    fz_force_ref();
    const char *path = getenv("PK_FUZZ_DIAR");
    if (!path || !*path)
        path = "tests/data/tiny-diar.pkf";
    char err[256];
    model = pk_model_load(path, err, sizeof err);
    if (!model) {
        fprintf(stderr, "diar: %s\n", err);
        abort();
    }
    S = (size_t)pk_model_speakers(model);
    FZ_ASSERT(S > 0 && S <= SPK_MAX);
    diar = pk_diar_new(model, 1);
    FZ_ASSERT(diar);
    return 0;
}

typedef struct {
    float *p;
    size_t rows, cap;
    int64_t next;                    /* the row the next call must start at */
    int64_t stop;                    /* stop once this many rows are in, or -1 */
} rows;

static int keep(void *arg, int64_t row, size_t n, const float *p) {
    rows *r = arg;
    FZ_ASSERT(row == r->next);
    r->next += (int64_t)n;
    FZ_ASSERT(r->rows + n <= r->cap);
    for (size_t i = 0; i < n * S; i++)
        FZ_ASSERT(p[i] >= 0 && p[i] <= 1);
    memcpy(r->p + r->rows * S, p, n * S * sizeof *p);
    r->rows += n;
    return r->stop >= 0 && (int64_t)r->rows >= r->stop;
}

static void engine_case(fz_input *f) {
    const uint32_t stop_bits = fz_u32(f);
    uint8_t cut[16];
    for (size_t i = 0; i < sizeof cut; i++)
        cut[i] = fz_u8(f);
    size_t n = (f->n - f->at) / 4;
    if (n > DIAR_MAX_SAMPLES)
        n = DIAR_MAX_SAMPLES;
    float *x = malloc((n ? n : 1) * sizeof *x);
    FZ_ASSERT(x);
    for (size_t i = 0; i < n; i++)
        x[i] = fz_float(f);

    /* more rows than any recording of n samples has */
    const size_t cap = n / 16 + 64;
    rows whole = { malloc(cap * S * sizeof(float)), 0, cap, 0, -1 };
    rows parts = { malloc(cap * S * sizeof(float)), 0, cap, 0, -1 };
    FZ_ASSERT(whole.p && parts.p);
    FZ_ASSERT(pk_diar_run(diar, x, n, keep, &whole) == 0);

    /* the same samples in pieces: sizes from the input, none to a few thousand */
    size_t at = 0;
    for (size_t i = 0; at < n; i++) {
        size_t piece = (size_t)cut[i % sizeof cut] * (i % 3 ? 37 : 1);
        if (piece > n - at || i > 4096)
            piece = n - at;
        FZ_ASSERT(pk_diar_feed(diar, x + at, piece, keep, &parts) == 0);
        at += piece;
    }
    FZ_ASSERT(pk_diar_finish(diar, keep, &parts) == 0);
    FZ_ASSERT(parts.rows == whole.rows);
    FZ_ASSERT(!memcmp(parts.p, whole.p, whole.rows * S * sizeof(float)));

    /* stopped by its function part way, the diarizer hears the next
     * recording as if it had never heard this one */
    if (whole.rows) {
        rows cutoff = { parts.p, 0, cap, 0, 1 + (int64_t)(stop_bits % whole.rows) };
        const int rc = pk_diar_run(diar, x, n, keep, &cutoff);
        FZ_ASSERT(rc == 1 && cutoff.rows <= whole.rows);
        FZ_ASSERT(!memcmp(cutoff.p, whole.p, cutoff.rows * S * sizeof(float)));
        rows again = { parts.p, 0, cap, 0, -1 };
        FZ_ASSERT(pk_diar_run(diar, x, n, keep, &again) == 0);
        FZ_ASSERT(again.rows == whole.rows);
        FZ_ASSERT(!memcmp(again.p, whole.p, whole.rows * S * sizeof(float)));
    }
    free(whole.p);
    free(parts.p);
    free(x);
}

static void speakers_case(fz_input *f) {
    const size_t nrows = fz_u32(f) % (SPK_MAX_ROWS + 1);
    const size_t nw = fz_u8(f) % (SPK_MAX_WORDS + 1);
    /* the recognizer's frame in samples: Parakeet's 1280, or any other */
    const uint32_t frame = fz_u8(f) & 1 ? 1280 : 1 + fz_u32(f) % 100000;
    pk_seg *w = malloc((nw ? nw : 1) * sizeof *w);
    int32_t *who = malloc((nw ? nw : 1) * sizeof *who);
    float *p = malloc((nrows ? nrows * S : 1) * sizeof *p);
    FZ_ASSERT(w && who && p);
    for (size_t i = 0; i < nw; i++) {
        /* anywhere from the start to well past the rows, of any length, none too */
        const int64_t start = (int64_t)(fz_u32(f) % (uint32_t)(2 * SPK_MAX_ROWS * 160 + 1));
        const int64_t len = fz_u8(f) & 1 ? fz_u32(f) % 100000 : fz_u8(f) % 4;
        w[i] = (pk_seg){ start, start + len };
        who[i] = -1;
    }
    for (size_t i = 0; i < nrows * S; i++)
        p[i] = (float)fz_u8(f) / 255.0f;

    pk_speakers(model, p, nrows, frame, w, nw, who);
    for (size_t i = 0; i < nw; i++)
        FZ_ASSERT(who[i] >= 0 && (size_t)who[i] < S);

    /* exactly the room pktito.h asks for, so that one more is an overflow */
    const size_t room = S * ((nrows + 1) / 2);
    pk_spk_seg *seg = malloc((room ? room : 1) * sizeof *seg);
    FZ_ASSERT(seg);
    const size_t ns = pk_speaker_segments(model, p, nrows, seg);
    FZ_ASSERT(ns <= room);
    for (size_t i = 0; i < ns; i++) {
        FZ_ASSERT(seg[i].speaker >= 0 && (size_t)seg[i].speaker < S);
        FZ_ASSERT(seg[i].start >= 0 && seg[i].end > seg[i].start);
        if (i)
            FZ_ASSERT(seg[i].start > seg[i - 1].start
                      || (seg[i].start == seg[i - 1].start && seg[i].speaker > seg[i - 1].speaker));
    }
    free(seg);
    free(p);
    free(who);
    free(w);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    fz_input f = { data, size, 0 };
    if (fz_u8(&f) & 1)
        engine_case(&f);
    else
        speakers_case(&f);
    return 0;
}
