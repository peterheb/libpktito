/* Arbitrary float bit patterns through the model and the audio state around
 * it.  The synthetic tiny model has every kind of layer but makes a bounded
 * two-second iteration cheap enough for coverage-guided fuzzing. */
#include "pktito.h"

#include "fuzz.h"

#define AUDIO_MAX_SAMPLES 32000

static pk_model *model;
static pk_asr *asr;
static pk_vad *vad;
static pk_model_info info;
static size_t hop;

int LLVMFuzzerInitialize(int *argc, char ***argv) {
    (void)argc;
    (void)argv;
    fz_force_ref();
    const char *path = getenv("PK_FUZZ_MODEL");
    if (!path || !*path)
        path = "tests/data/tiny.pkf";
    char err[256];
    model = pk_model_load(path, err, sizeof err);
    if (!model) {
        fprintf(stderr, "audio_pipeline: %s\n", err);
        abort();
    }
    pk_model_describe(model, &info);
    hop = (size_t)(pk_model_frame_seconds(model) * pk_model_rate(model) / 8.0 + 0.5);
    FZ_ASSERT(hop > 0);
    asr = pk_asr_new(model, 1);
    vad = pk_vad_new(model);
    FZ_ASSERT(asr && vad);
    return 0;
}

static int duration_ok(int32_t d) {
    for (uint32_t i = 0; i < info.n_durations; i++)
        if (d == (int32_t)info.durations[i])
            return 1;
    return 0;
}

static void check_text(const pk_tok *tok, size_t n) {
    size_t len = pk_text(model, tok, n, NULL, 0);
    FZ_ASSERT(len <= 16 * 1024 * 1024);
    char *full = malloc(len + 2), *cut = malloc(len + 2);
    FZ_ASSERT(full && cut);
    FZ_ASSERT(pk_text(model, tok, n, full, len + 1) == len);
    FZ_ASSERT(strlen(full) == len && fz_utf8(full, len));
    static const size_t fixed[] = { 0, 1, 2, 7, 31 };
    for (size_t i = 0; i < sizeof fixed / sizeof fixed[0] + 1; i++) {
        size_t cap = i < sizeof fixed / sizeof fixed[0] ? fixed[i] : len / 2 + 1;
        if (cap > len + 1)
            cap = len + 1;
        memset(cut, 0x5a, len + 2);
        FZ_ASSERT(pk_text(model, tok, n, cut, cap) == len);
        if (cap) {
            size_t got = strlen(cut);
            FZ_ASSERT(got < cap && got <= len && !memcmp(cut, full, got) && fz_utf8(cut, got));
            FZ_ASSERT((unsigned char)cut[cap] == 0x5a);
        }
    }
    free(cut);
    free(full);
}

static void check_asr(const float *x, size_t n, int repeat) {
    const pk_tok *tok;
    long got = pk_asr_run(asr, x, n, &tok);
    FZ_ASSERT(got >= 0);
    size_t nt = (size_t)got, t0 = n / hop;
    uint32_t te = (uint32_t)t0;
    for (int i = 0; i < 3; i++)
        te = (te + 1) / 2;
    for (size_t i = 0; i < nt; i++) {
        FZ_ASSERT(tok[i].id >= 0 && (uint32_t)tok[i].id < info.vocab);
        FZ_ASSERT(tok[i].frame >= 0 && (uint32_t)tok[i].frame < te);
        FZ_ASSERT(tok[i].dur >= 0 && duration_ok(tok[i].dur));
    }
    check_text(tok, nt);

    pk_word *word = malloc((nt ? nt : 1) * sizeof *word);
    FZ_ASSERT(word);
    size_t nw = pk_words(model, tok, nt, word), next = 0;
    FZ_ASSERT(nw <= nt);
    for (size_t i = 0; i < nw; i++) {
        FZ_ASSERT(word[i].tok == (int32_t)next && word[i].ntok > 0);
        FZ_ASSERT(word[i].start >= 0 && word[i].end >= word[i].start);
        next += (size_t)word[i].ntok;
    }
    FZ_ASSERT(next == nt);
    free(word);

    if (repeat) {
        pk_tok *copy = malloc((nt ? nt : 1) * sizeof *copy);
        FZ_ASSERT(copy);
        if (nt)
            memcpy(copy, tok, nt * sizeof *copy);
        const pk_tok *again;
        long n_again = pk_asr_run(asr, x, n, &again);
        FZ_ASSERT(n_again == got);
        if (nt)
            FZ_ASSERT(!memcmp(copy, again, nt * sizeof *copy));
        free(copy);
    }
}

static void check_vad(const float *x, size_t n, int transcribe) {
    size_t nf = n / PK_VAD_SAMPLES;
    float *batch = malloc((nf ? nf : 1) * sizeof *batch);
    FZ_ASSERT(batch);
    pk_vad_reset(vad);
    pk_vad_probs(vad, x, nf, batch);
    for (size_t i = 0; i < nf; i++)
        FZ_ASSERT(isfinite(batch[i]) && batch[i] >= 0 && batch[i] <= 1);
    pk_vad_reset(vad);
    for (size_t i = 0; i < nf; i++) {
        float one = pk_vad_prob(vad, x + i * PK_VAD_SAMPLES);
        FZ_ASSERT(!memcmp(&one, &batch[i], sizeof one));
    }

    pk_seg_params p;
    pk_seg_defaults(&p);
    int64_t max_len = n > 16000 ? 16000 : (int64_t)(n ? n : 1);
    p.max_speech_s = (double)max_len / 16000.0;
    size_t seg_cap = 2 * nf + 1;
    pk_seg *seg = malloc(seg_cap * sizeof *seg);
    FZ_ASSERT(seg);
    size_t ns = pk_segments(batch, nf, 16000, (int64_t)n, &p, seg, seg_cap);
    FZ_ASSERT(ns <= seg_cap);
    size_t chunk_cap = 2 * ns + 2 + n / (size_t)max_len;
    pk_seg *chunk = malloc(chunk_cap * sizeof *chunk);
    FZ_ASSERT(chunk);
    size_t nc = pk_chunks(x, seg, ns, (int64_t)n, max_len, chunk);
    FZ_ASSERT(nc <= chunk_cap);
    for (size_t i = 0; i < nc; i++) {
        FZ_ASSERT(chunk[i].start >= 0 && chunk[i].end > chunk[i].start && chunk[i].end <= (int64_t)n);
        FZ_ASSERT(chunk[i].end - chunk[i].start <= max_len);
        if (i)
            FZ_ASSERT(chunk[i - 1].end <= chunk[i].start);
        if (transcribe)
            check_asr(x + chunk[i].start, (size_t)(chunk[i].end - chunk[i].start), 0);
    }
    free(chunk);
    free(seg);
    free(batch);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (!size || size > 1 + AUDIO_MAX_SAMPLES * sizeof(float))
        return 0;
    unsigned mode = data[0];
    size_t n = (size - 1) / sizeof(float);
    float *x = malloc((n ? n : 1) * sizeof *x);
    FZ_ASSERT(x);
    for (size_t i = 0; i < n; i++) {
        uint32_t u = (uint32_t)data[1 + 4 * i] | (uint32_t)data[2 + 4 * i] << 8
                   | (uint32_t)data[3 + 4 * i] << 16 | (uint32_t)data[4 + 4 * i] << 24;
        memcpy(&x[i], &u, sizeof x[i]);
    }
    if (mode % 3 != 1)
        check_asr(x, n, mode & 4);
    if (mode % 3 != 0)
        check_vad(x, n, mode & 8);
    free(x);
    return 0;
}
