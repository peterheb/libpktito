/* transcheck MODEL WAV: pk_asr_transcribe, the high-level path, against the
 * low-level one it is made of. Without retry every chunk is
 * pk_asr_run_chunks' (which is pk_asr_run's), token for token; fn
 * returning nonzero stops the run there; and the retry's own parts as
 * pktito-app's transcribe.rs tested them, where they came from. Includes
 * core/transcribe.c, for its static parts. */
#include "../core/transcribe.c"
#include "wav.h"

#include <stdio.h>

static int fails;
#define CHECK(c, what) do { if (!(c)) { printf("transcheck: FAIL - %s\n", what); fails++; } } while (0)

/* every chunk's tokens, as one list a chunk */
typedef struct {
    size_t n;
    pk_seg at[64];
    pk_tok *t[64];
    long nt[64];
    size_t stop_after;          /* stop after this many chunks; 0: never */
} seen;

static void keep(seen *s, size_t i, pk_seg at, const pk_tok *t, long nt) {
    if (i >= 64)
        return;
    s->at[i] = at;
    s->nt[i] = nt;
    s->t[i] = malloc(((size_t)nt + 1) * sizeof *t);
    memcpy(s->t[i], t, (size_t)nt * sizeof *t);
    s->n = i + 1;
}

static int on_out(void *arg, const pk_chunk_out *c) {
    seen *s = arg;
    keep(s, c->i, c->at, c->toks, c->count);
    return s->stop_after && s->n == s->stop_after;
}

typedef struct {
    seen *s;
    const pk_seg *c;
} plain_arg;

static void on_plain(void *arg, size_t i, const pk_tok *t, long nt) {
    plain_arg *p = arg;
    keep(p->s, i, p->c[i], t, nt);
}

static int same(const seen *a, const seen *b) {
    if (a->n != b->n)
        return 0;
    for (size_t i = 0; i < a->n; i++)
        if (a->at[i].start != b->at[i].start || a->at[i].end != b->at[i].end || a->nt[i] != b->nt[i]
            || memcmp(a->t[i], b->t[i], (size_t)a->nt[i] * sizeof *a->t[i]))
            return 0;
    return 1;
}

static void forget(seen *s) {
    for (size_t i = 0; i < s->n; i++)
        free(s->t[i]);
    s->n = 0;
}

int main(int argc, char **argv) {
    /* quietest: in the pause, else in the middle third, else the middle */
    {
        static float x[9000];
        for (int i = 0; i < 9000; i++)
            x[i] = i >= 4000 && i < 4400 ? 0 : 0.5f;
        size_t q = quietest(x, 9000, 1000);
        CHECK(q >= 4000 && q < 4400, "quietest: a pause's middle");
        for (int i = 0; i < 9000; i++)
            x[i] = 0.5f;
        q = quietest(x, 9000, 1000);
        CHECK(q >= 3000 && q < 6000, "quietest: no pause, the middle third");
        CHECK(quietest(x, 400, 1000) == 200, "quietest: too short, the middle");
    }
    /* power: the speech in a stretch, and how much of it there is */
    {
        static float x[1000];
        for (int i = 0; i < 1000; i++)
            x[i] = 1;
        const pk_seg sp[] = { { 100, 300 }, { 500, 900 } };
        int64_t h;
        power(x, sp, 2, 100, 1000, &h);
        CHECK(h == 600, "power: all the speech");
        CHECK(power(x, sp, 2, 250, 600, &h) == 150 && h == 150, "power: part of two segments");
        power(x, sp, 2, 300, 500, &h);
        CHECK(h == 0, "power: between them, none");
    }
    if (argc != 3) {
        fprintf(stderr, "usage: transcheck MODEL WAV\n");
        return 2;
    }
    char err[256];
    pk_model *m = pk_model_load(argv[1], err, sizeof err);
    pk_wav_info info;
    pk_wav *w = m ? pk_wav_open(argv[2], &info, err, sizeof err) : NULL;
    if (!w) {
        printf("transcheck: %s\n", err);
        return 1;
    }
    float *x = malloc((size_t)info.frames * sizeof *x);
    size_t n = x ? pk_wav_read(w, x, (size_t)info.frames) : 0;
    pk_wav_close(w);
    pk_asr *a = pk_asr_new(m, 0);
    seen hi = { 0 }, lo = { 0 };

    /* without retry, the low-level path's chunks */
    pk_transcribe_opts o;
    pk_transcribe_defaults(&o);
    o.retry = 0;
    CHECK(pk_asr_transcribe(a, x, n, &o, on_out, &hi) == 0, "no retry: runs to the end");
    {
        /* the chunks as ./pktito-c cut them before the call: the same steps */
        const int rate = pk_model_rate(m);
        size_t frames = (n + PK_VAD_SAMPLES - 1) / PK_VAD_SAMPLES;
        float *pad = calloc(frames * PK_VAD_SAMPLES, sizeof *pad), *probs = malloc(frames * sizeof *probs);
        pk_seg *seg = malloc((2 * frames + 1) * sizeof *seg);
        pk_vad *v = pk_vad_new(m);
        memcpy(pad, x, n * sizeof *x);
        pk_vad_probs(v, pad, frames, probs);
        pk_seg_params sp;
        pk_seg_defaults(&sp);
        sp.max_speech_s = 30 - 2 * sp.speech_pad_ms / 1000 - 0.1;
        size_t ns = pk_segments(probs, frames, rate, (int64_t)n, &sp, seg, 2 * frames + 1);
        pk_seg *c = malloc((2 * ns + 2 + n / (30 * (size_t)rate)) * sizeof *c);
        size_t nc = pk_chunks(x, seg, ns, (int64_t)n, 30 * (int64_t)rate, c);
        plain_arg p = { &lo, c };
        CHECK(pk_asr_run_chunks(a, x, c, nc, on_plain, &p) == 0, "pk_asr_run_chunks runs");
        CHECK(nc > 1, "the recording is more than one chunk");
        pk_vad_free(v);
        free(pad), free(probs), free(seg), free(c);
    }
    CHECK(same(&hi, &lo), "no retry: every chunk pk_asr_run_chunks', token for token");

    /* with retry (and opts NULL, the defaults), where the model drops
     * nothing: the same */
    seen r = { 0 };
    CHECK(pk_asr_transcribe(a, x, n, NULL, on_out, &r) == 0, "defaults: runs to the end");
    CHECK(same(&r, &lo), "retry: nothing dropped, nothing changed");
    forget(&r);

    /* fn stops it after the first chunk */
    r.stop_after = 1;
    CHECK(pk_asr_transcribe(a, x, n, NULL, on_out, &r) == 1 && r.n == 1, "stopped after one chunk");
    CHECK(r.nt[0] == lo.nt[0] && !memcmp(r.t[0], lo.t[0], (size_t)lo.nt[0] * sizeof *lo.t[0]),
          "stopped: its one chunk the same");
    forget(&r);
    r.stop_after = 0;

    /* and the transcriber is whole after a stop */
    CHECK(pk_asr_transcribe(a, x, n, &o, on_out, &r) == 0 && same(&r, &lo), "after a stop: the same");
    forget(&r);

    o.max_chunk_s = 0.5;
    CHECK(pk_asr_transcribe(a, x, n, &o, on_out, &r) == -1 && r.n == 0, "max_chunk_s under a second: -1");
    CHECK(pk_asr_transcribe(a, x, 0, NULL, on_out, &r) == 0 && r.n == 0, "no samples: no chunks");

    if (!fails)
        printf("transcheck: %zu chunks the low-level path's, a stop, the retry's parts: ok\n", lo.n);
    forget(&hi), forget(&lo);
    pk_asr_free(a);
    pk_model_free(m);
    free(x);
    return fails != 0;
}
