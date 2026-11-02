/* transcheck MODEL WAV: pk_asr_transcribe, the high-level path, against the
 * low-level one it is made of. Without the context and the retry every
 * chunk is pk_asr_run_chunks' (which is pk_asr_run's), token for token;
 * with them every word starts in its own chunk; fn returning nonzero stops
 * the run there; the retry's own parts as pktito-app's transcribe.rs tested
 * them, where they came from; and the seam's, on tokens built for it.
 * Includes core/transcribe.c, for its static parts. */
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

/* the token whose piece is text, -1 if the model has none */
static int32_t piece(const pk_model *m, const char *text) {
    for (uint32_t id = 0; id < m->n_pieces; id++) {
        size_t len;
        const char *p = pk_piece(m, (int32_t)id, &len);
        if (len == strlen(text) && !memcmp(p, text, len))
            return (int32_t)id;
    }
    return -1;
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
    /* lower and utf8: the seam's letters, of the alphabets the models write */
    {
        const uint32_t up[] = { 'A', 'Z', 0xC9, 0xDE, 0x100, 0x141, 0x17D, 0x178, 0x391, 0x3A9, 0x401, 0x410, 0x42F };
        const uint32_t lo[] = { 'a', 'z', 0xE9, 0xFE, 0x101, 0x142, 0x17E, 0xFF, 0x3B1, 0x3C9, 0x451, 0x430, 0x44F };
        int ok = 1;
        for (size_t i = 0; i < sizeof up / sizeof *up; i++)
            ok &= lower(up[i]) == lo[i] && lower(lo[i]) == lo[i];
        CHECK(ok, "lower: Latin, Greek and Cyrillic capitals, their small letters as they are");
        CHECK(lower(0xD7) == 0xD7 && lower(0xDF) == 0xDF && lower(0x131) == 0x131 && lower('1') == '1',
              "lower: the multiplication sign, sharp s, dotless i and a digit as they are");
        const char *t = "a\xc3\xa9\xc5\x82\xd0\x94\xe2\x82\xac\xf0\x9f\x98\x80", *end = t + strlen(t);
        const uint32_t cp[] = { 'a', 0xE9, 0x142, 0x414, 0x20AC, 0x1F600 };
        ok = 1;
        for (size_t i = 0; i < 6; i++)
            ok &= t < end && utf8(&t, end) == cp[i];
        CHECK(ok && t == end, "utf8: one to four bytes");
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

    /* without the context and the retry, the low-level path's chunks */
    pk_transcribe_opts o;
    pk_transcribe_defaults(&o);
    o.retry = 0;
    o.context = 0;
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

    /* with retry, where the model drops nothing: the same */
    seen r = { 0 };
    o.retry = 1;
    CHECK(pk_asr_transcribe(a, x, n, &o, on_out, &r) == 0, "retry: runs to the end");
    CHECK(same(&r, &lo), "retry: nothing dropped, nothing changed");
    forget(&r);

    /* fn stops it after the first chunk */
    r.stop_after = 1;
    CHECK(pk_asr_transcribe(a, x, n, &o, on_out, &r) == 1 && r.n == 1, "stopped after one chunk");
    CHECK(r.nt[0] == lo.nt[0] && !memcmp(r.t[0], lo.t[0], (size_t)lo.nt[0] * sizeof *lo.t[0]),
          "stopped: its one chunk the same");
    forget(&r);
    r.stop_after = 0;

    /* with the context too (opts NULL, the defaults): every word starts in
     * its own chunk, and fn still stops it */
    {
        const int64_t spf = llround(pk_model_frame_seconds(m) * pk_model_rate(m));
        CHECK(pk_asr_transcribe(a, x, n, NULL, on_out, &r) == 0 && r.n > 1, "defaults: runs to the end");
        int in = 1;
        pk_word *wd = malloc(4096 * sizeof *wd);
        for (size_t i = 0; i < r.n && i < 64; i++) {
            const size_t nw = r.nt[i] < 4096 ? pk_words(m, r.t[i], (size_t)r.nt[i], wd) : 0;
            for (size_t j = 0; j < nw; j++)
                in &= r.at[i].start + (int64_t)wd[j].start * spf < r.at[i].end;
        }
        free(wd);
        CHECK(in, "context: every word starts in its own chunk");
        forget(&r);
        r.stop_after = 1;
        CHECK(pk_asr_transcribe(a, x, n, NULL, on_out, &r) == 1 && r.n == 1, "context: stopped after one chunk");
        forget(&r);
        r.stop_after = 0;
    }

    /* the seam, on tokens built for it: a word the cut fell beside written
     * twice, a capital, and words that start apart */
    {
        const char *sp = "\xe2\x96\x81";
        char b[4][16];
        const char *words[4] = { "The", "the", "was", "no" };
        int32_t id[4];
        for (int k = 0; k < 4; k++) {
            snprintf(b[k], sizeof b[k], "%s%s", sp, words[k]);
            id[k] = piece(m, b[k]);
        }
        CHECK(id[0] >= 0 && id[1] >= 0 && id[2] >= 0 && id[3] >= 0, "seam: the model has the test's pieces");
        const int32_t The = id[0], the = id[1], was = id[2], no = id[3];
        const int64_t spf = llround(pk_model_frame_seconds(m) * pk_model_rate(m));
        const pk_seg ch = { 100 * spf, 400 * spf };
        run sr = { 0 };
        sr.a = a;
        sr.t = malloc((8 + SEAM_TOKS) * sizeof *sr.t);
        sr.w = malloc(8 * sizeof *sr.w);
        if (The >= 0 && the >= 0 && was >= 0 && no >= 0 && sr.t && sr.w) {
            /* the last chunk heard "was no" past its cut; this one "The was no" */
            sr.nseam = 2;
            sr.seam[0] = (pk_tok){ was, 0, 1 };
            sr.seam[1] = (pk_tok){ no, 0, 1 };
            sr.seam_at[0] = ch.start + 2 * spf;
            sr.seam_at[1] = ch.start + 5 * spf;
            sr.seam_word[0] = 0, sr.seam_word[1] = 1, sr.seam_word[2] = 2;
            sr.t[0] = (pk_tok){ The, 0, 1 };
            sr.t[1] = (pk_tok){ was, 3, 1 };
            sr.t[2] = (pk_tok){ no, 5, 1 };
            size_t k = seam(&sr, &ch, 3);
            CHECK(k == 2 && sr.t[0].id == was && sr.t[1].id == no, "seam: the word written twice goes");
            /* the last chunk heard "the no", this one "The no" */
            sr.seam[0] = (pk_tok){ the, 0, 1 };
            sr.seam_at[0] = ch.start;
            sr.seam_at[1] = ch.start + 3 * spf;
            sr.t[0] = (pk_tok){ The, 1, 1 };
            sr.t[1] = (pk_tok){ no, 3, 1 };
            k = seam(&sr, &ch, 2);
            CHECK(k == 2 && sr.t[0].id == the && sr.t[0].frame == 0 && sr.t[1].id == no,
                  "seam: the first word's case as the last chunk wrote it");
            /* this one's words start apart from the last chunk's */
            sr.t[0] = (pk_tok){ The, 20, 1 };
            sr.t[1] = (pk_tok){ was, 22, 1 };
            k = seam(&sr, &ch, 2);
            CHECK(k == 2 && sr.t[0].id == The && sr.t[1].id == was, "seam: words that start apart as they are");
        }
        free(sr.t);
        free(sr.w);
    }

    /* and the transcriber is whole after a stop */
    CHECK(pk_asr_transcribe(a, x, n, &o, on_out, &r) == 0 && same(&r, &lo), "after a stop: the same");
    forget(&r);

    o.max_chunk_s = 0.5;
    CHECK(pk_asr_transcribe(a, x, n, &o, on_out, &r) == -1 && r.n == 0, "max_chunk_s under a second: -1");
    CHECK(pk_asr_transcribe(a, x, 0, NULL, on_out, &r) == 0 && r.n == 0, "no samples: no chunks");

    if (!fails)
        printf("transcheck: %zu chunks the low-level path's, a stop, the retry's and the seam's parts: ok\n", lo.n);
    forget(&hi), forget(&lo);
    pk_asr_free(a);
    pk_model_free(m);
    free(x);
    return fails != 0;
}
