/* transcheck MODEL WAV: pk_asr_transcribe, the high-level path, against the
 * low-level one it is made of. Without the context and the retry every
 * chunk is pk_asr_run_chunks' (which is pk_asr_run's), token for token,
 * each chunk's place and the run's progress the same; with them every word
 * starts in its own chunk; fn returning nonzero stops either run there; two
 * lanes give one lane's chunks, in the fast mode too; the same in steps
 * (pk_steps), its jobs heard out of order on other transcribers, gives
 * pk_asr_transcribe's chunks, and a chunk put back empty its retries one at
 * a time; the VAD over a pool gives one thread's probabilities, and
 * pk_asr_vad and pk_asr_speech pk_vad_probs' and pk_segments'; the retry's
 * own parts as pktito-app's transcribe.rs tested them, where they came
 * from; and the seam's and the retry level's, on tokens and samples built
 * for them. Includes core/transcribe.c, for its static parts. */
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
    int retried[64];            /* pk_chunk_out's */
    int32_t at_frame[64];
    int64_t done[64], total;
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
    if (c->i < 64) {
        s->retried[c->i] = c->retried;
        s->at_frame[c->i] = c->at_frame;
        s->done[c->i] = c->done;
        s->total = c->total;
    }
    return s->stop_after && s->n == s->stop_after;
}

/* A run of pk_steps, as a caller with workers of its own runs it: every job
 * there is handed out at once, and the last handed out heard first, on h[0]
 * and h[1] by turns, each chunk taken into s as soon as it is done. With
 * drop, chunk 0 is put back with no tokens, as where the model drops a
 * chunk, and its retries' tokens are kept in want[0, *nwant), their frames
 * from the chunk's start (spf samples a frame), and their stretches in
 * got[0, *nst). *ok is whether the chunks were handed out once each in
 * order, the retries one at a time for the chunk the seam is at, and the
 * chunks taken in order. Returns 0, or -1 where a call failed or the run
 * stalled, nothing out and nothing to hand out. */
static int drive(pk_steps *st, pk_asr *h[2], const float *x, int64_t spf, seen *s, int drop,
                 pk_tok *want, long *nwant, pk_seg *got, int *nst, int *ok) {
    pk_job out[256];
    size_t nout = 0, next = 0, taken = 0, turn = 0;
    const size_t nc = pk_steps_chunks(st);
    int retries_out = 0, rc = 0;
    int64_t start0 = 0;         /* chunk 0's first sample, where its job starts */
    *ok = 1;
    *nwant = 0;
    *nst = 0;
    while (taken < nc) {
        pk_job j;
        while (nout < 256 && (rc = pk_steps_job(st, &j)) == 1) {
            if (j.retry)
                *ok &= ++retries_out == 1 && j.chunk == taken;
            else
                *ok &= j.chunk == next++;
            out[nout++] = j;
        }
        if (rc < 0 || nout == 0)
            return -1;
        j = out[--nout];
        const pk_tok *t;
        long nt = pk_asr_run(h[turn++ % 2], x + j.at.start, (size_t)(j.at.end - j.at.start), &t);
        if (nt < 0)
            return -1;
        if (!j.retry && j.chunk == 0) {
            start0 = j.at.start;
            if (drop)
                nt = 0;
        }
        if (j.retry) {
            retries_out--;
            if (j.chunk == 0 && *nst < 8) {
                got[(*nst)++] = j.at;
                for (long q = 0; q < nt && *nwant < 4096; q++) {
                    want[*nwant] = t[q];
                    want[(*nwant)++].frame += (int32_t)((j.at.start - start0) / spf);
                }
            }
        }
        if (pk_steps_put(st, &j, t, nt))
            return -1;
        pk_chunk_out c;
        while ((rc = pk_steps_take(st, &c)) == 1) {
            *ok &= c.i == taken++ && c.n == nc;
            on_out(s, &c);
        }
        if (rc < 0)
            return -1;
    }
    pk_job j;
    return pk_steps_job(st, &j) == 0 && nout == 0 ? 0 : -1;
}

static int same(const seen *a, const seen *b) {
    if (a->n != b->n)
        return 0;
    for (size_t i = 0; i < a->n; i++)
        if (a->at[i].start != b->at[i].start || a->at[i].end != b->at[i].end || a->nt[i] != b->nt[i]
            || a->at_frame[i] != b->at_frame[i] || a->done[i] != b->done[i]
            || memcmp(a->t[i], b->t[i], (size_t)a->nt[i] * sizeof *a->t[i]))
            return 0;
    return a->total == b->total;
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
        /* the chunks by the call's own steps: segments ended at a pause of
         * 64 ms (core/transcribe.c's CUT_SILENCE_MS), chunks of at most
         * max_chunk_s cut between them */
        const int rate = pk_model_rate(m);
        size_t frames = (n + PK_VAD_SAMPLES - 1) / PK_VAD_SAMPLES;
        float *pad = calloc(frames * PK_VAD_SAMPLES, sizeof *pad), *probs = malloc(frames * sizeof *probs);
        pk_seg *seg = malloc((2 * frames + 1) * sizeof *seg);
        pk_vad *v = pk_vad_new(m);
        memcpy(pad, x, n * sizeof *x);
        pk_vad_probs(v, pad, frames, probs);
        pk_seg_params sp;
        pk_seg_defaults(&sp);
        sp.min_silence_ms = 32;
        sp.max_speech_s = o.max_chunk_s - 2 * sp.speech_pad_ms / 1000 - 0.1;
        const int64_t max_len = (int64_t)(o.max_chunk_s * rate);
        size_t ns = pk_segments(probs, frames, rate, (int64_t)n, &sp, seg, 2 * frames + 1);
        pk_seg *c = malloc((2 * ns + 2 + (size_t)((int64_t)n / max_len)) * sizeof *c);
        size_t nc = pk_chunks(x, seg, ns, (int64_t)n, max_len, c);
        CHECK(pk_asr_run_chunks(a, x, c, nc, on_out, &lo) == 0, "pk_asr_run_chunks runs");
        CHECK(nc > 1 && lo.n == nc, "the recording is more than one chunk");
        const int64_t spf = llround(pk_model_frame_seconds(m) * rate);
        int64_t done = 0;
        int ok = 1;
        for (size_t i = 0; i < lo.n; i++) {
            done += c[i].end - c[i].start;
            ok &= lo.at[i].start == c[i].start && lo.at[i].end == c[i].end && !lo.retried[i]
                  && lo.done[i] == done && lo.at_frame[i] == (lo.at[i].start + spf / 2) / spf;
        }
        CHECK(ok && lo.total == done, "pk_asr_run_chunks: each chunk's place, its frame and the progress");
        seen one = { .stop_after = 1 };
        CHECK(pk_asr_run_chunks(a, x, c, nc, on_out, &one) == 1 && one.n == 1 && one.nt[0] == lo.nt[0],
              "pk_asr_run_chunks: stopped after one chunk");
        forget(&one);
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

    /* several chunks at once (core/asr.c, run_lanes): the recording four
     * times over in chunks of at most 10 s, through a transcriber of one
     * lane (4 threads) and one of two (8 threads; a lane for each four, and
     * four chunks a lane at least), with the context and the retry: every
     * chunk the same tokens, fn's in order, and a stop the same */
    {
        float *x4 = malloc(4 * n * sizeof *x4);
        pk_asr *one = pk_asr_new(m, 4), *two = pk_asr_new(m, 8);
        seen s1 = { 0 }, s2 = { 0 };
        pk_transcribe_opts o4;
        pk_transcribe_defaults(&o4);
        o4.max_chunk_s = 10;
        CHECK(x4 && one && two, "lanes: two transcribers");
        if (x4 && one && two) {
            for (int k = 0; k < 4; k++)
                memcpy(x4 + (size_t)k * n, x, n * sizeof *x);
            CHECK(one->lanes <= 1 && two->lanes == 2, "lanes: one for each four threads");
            CHECK(pk_asr_transcribe(one, x4, 4 * n, &o4, on_out, &s1) == 0 && s1.n >= 8,
                  "lanes: one lane runs to the end, in 8 chunks or more");
            CHECK(pk_asr_transcribe(two, x4, 4 * n, &o4, on_out, &s2) == 0, "lanes: two run to the end");
            CHECK(same(&s1, &s2), "lanes: every chunk one lane's, token for token, in order");
            forget(&s2);
            s2.stop_after = 3;
            CHECK(pk_asr_transcribe(two, x4, 4 * n, &o4, on_out, &s2) == 1 && s2.n == 3,
                  "lanes: stopped after three chunks");
            int first = s2.n == 3;
            for (size_t i = 0; first && i < 3; i++)
                first = s2.nt[i] == s1.nt[i] && !memcmp(s2.t[i], s1.t[i], (size_t)s1.nt[i] * sizeof *s1.t[i]);
            CHECK(first, "lanes: stopped, its three chunks the same");
            /* the fast mode, asked for after two's second lane was made */
            forget(&s1), forget(&s2);
            s2.stop_after = 0;
            pk_asr_set_fast_mode(one, 1), pk_asr_set_fast_mode(two, 1);
            CHECK(pk_asr_transcribe(one, x4, 4 * n, &o4, on_out, &s1) == 0
                      && pk_asr_transcribe(two, x4, 4 * n, &o4, on_out, &s2) == 0 && same(&s1, &s2),
                  "lanes: the fast mode in every lane, every chunk one lane's");
        }
        forget(&s1), forget(&s2);
        pk_asr_free(one), pk_asr_free(two);
        free(x4);
    }

    /* the same in steps (pk_steps): the recording four times over in chunks
     * of at most 10 s, with the context and the retry, the jobs handed out
     * as they come and heard last first on transcribers of 3 and 5 threads,
     * against pk_asr_transcribe on one of 4 - every chunk its tokens, as
     * many retried, in order. Then test.wav whole with chunk 0 put back
     * empty, as where the model drops a chunk: its two halves heard again
     * (core/transcribe.c, stretches), one at a time, from the chunk's start
     * to its end, and the chunk their tokens. */
    {
        const int64_t spf = llround(pk_model_frame_seconds(m) * pk_model_rate(m));
        float *x4 = malloc(4 * n * sizeof *x4);
        pk_asr *one = pk_asr_new(m, 4), *h[2] = { pk_asr_new(m, 3), pk_asr_new(m, 5) };
        pk_tok *want = malloc(4096 * sizeof *want);
        seen s1 = { 0 }, s2 = { 0 }, s3 = { 0 };
        pk_transcribe_opts o4;
        pk_transcribe_defaults(&o4);
        o4.max_chunk_s = 10;
        pk_seg got[8];
        long nwant = 0;
        int nst = 0, ok = 0;
        CHECK(x4 && one && h[0] && h[1] && want, "steps: three transcribers");
        if (x4 && one && h[0] && h[1] && want) {
            for (int k = 0; k < 4; k++)
                memcpy(x4 + (size_t)k * n, x, n * sizeof *x);
            pk_steps *st = pk_steps_new(one, x4, 4 * n, &o4);
            CHECK(pk_asr_transcribe(one, x4, 4 * n, &o4, on_out, &s1) == 0 && s1.n >= 8,
                  "steps: pk_asr_transcribe runs to the end, in 8 chunks or more");
            CHECK(st && drive(st, h, x4, spf, &s2, 0, want, &nwant, got, &nst, &ok) == 0 && ok,
                  "steps: every chunk handed out once, in order, and taken in order");
            CHECK(same(&s1, &s2) && !memcmp(s1.retried, s2.retried, sizeof s1.retried),
                  "steps: every chunk pk_asr_transcribe's, token for token, as many retried");
            pk_steps_free(st);

            st = pk_steps_new(one, x, n, NULL);
            CHECK(st && drive(st, h, x, spf, &s3, 1, want, &nwant, got, &nst, &ok) == 0 && ok,
                  "steps, a chunk dropped: the jobs as before, its retries one at a time");
            CHECK(s3.n > 1 && nst == 2 && s3.retried[0] == 2 && got[0].start == s3.at[0].start
                      && got[0].end == got[1].start && got[1].end == s3.at[0].end,
                  "steps, a chunk dropped: its two halves heard again, from its start to its end");
            CHECK(nwant > 0 && s3.nt[0] == nwant && !memcmp(s3.t[0], want, (size_t)nwant * sizeof *want),
                  "steps, a chunk dropped: the chunk its halves' tokens");
            pk_steps_free(st);
        }
        forget(&s1), forget(&s2), forget(&s3);
        pk_asr_free(one), pk_asr_free(h[0]), pk_asr_free(h[1]);
        free(want);
        free(x4);
    }

    /* pk_steps' edges: max_chunk_s under a second, no samples, and jobs
     * that are not out - a chunk not handed out, one put twice, a retry,
     * a chunk's samples not its own - each refused, and every call after
     * a refusal refused too */
    {
        pk_transcribe_opts oh;
        pk_transcribe_defaults(&oh);
        oh.max_chunk_s = 0.5;
        CHECK(pk_steps_new(a, x, n, &oh) == NULL, "steps: max_chunk_s under a second, none");
        pk_steps *st = pk_steps_new(a, x, 0, NULL);
        pk_job j;
        pk_chunk_out c;
        CHECK(st && pk_steps_chunks(st) == 0 && pk_steps_job(st, &j) == 0 && pk_steps_take(st, &c) == 0,
              "steps: no samples, no jobs, no chunks");
        pk_steps_free(st);
        st = pk_steps_new(a, x, n, NULL);
        CHECK(st && pk_steps_job(st, &j) == 1 && j.chunk == 0 && !j.retry, "steps: chunk 0 first");
        pk_steps_free(st);
        const pk_tok *heard;
        const long nt = pk_asr_run(a, x + j.at.start, (size_t)(j.at.end - j.at.start), &heard);
        pk_tok *t = malloc(((size_t)(nt > 0 ? nt : 1)) * sizeof *t);     /* past the next call on a */
        if (!t)
            return 1;
        if (nt > 0)
            memcpy(t, heard, (size_t)nt * sizeof *t);
        const pk_job later = { 1, 0, j.at }, again = { 0, 1, j.at },
                     moved = { 0, 0, { j.at.start + 1, j.at.end } };
        const struct { const pk_job *bad; int first; const char *what; } refused[] = {
            { &later, 0, "steps: a chunk not handed out, refused, and every call after it" },
            { &j, 1, "steps: a chunk put twice, refused the second time, and every call after it" },
            { &again, 0, "steps: a retry not handed out, refused, and every call after it" },
            { &moved, 0, "steps: a chunk's samples not its own, refused, and every call after it" },
        };
        for (size_t k = 0; k < sizeof refused / sizeof *refused; k++) {
            pk_job h;
            st = pk_steps_new(a, x, n, NULL);
            int ok = st && pk_steps_job(st, &h) == 1 && h.chunk == 0;
            if (ok && refused[k].first)
                ok = pk_steps_put(st, &h, t, nt) == 0;
            ok = ok && pk_steps_put(st, refused[k].bad, t, nt) == -1 && pk_steps_put(st, &h, t, nt) == -1
                 && pk_steps_job(st, &h) == -1 && pk_steps_take(st, &c) == -1;
            CHECK(ok, refused[k].what);
            pk_steps_free(st);
        }
        free(t);
    }

    /* pk_asr_vad: Silero over the transcriber's threads, its last frame of
     * 300 samples padded with silence - pk_vad_probs' bits over a copy
     * padded with zeros, twice over, its stream made new each time; and
     * pk_asr_speech pk_segments' speech from them, Silero's or not */
    {
        const int rate = pk_model_rate(m);
        const size_t nn = (n / PK_VAD_SAMPLES - 1) * PK_VAD_SAMPLES + 300;
        const size_t frames = (nn + PK_VAD_SAMPLES - 1) / PK_VAD_SAMPLES, cap = 2 * frames + 1;
        float *pad = calloc(frames * PK_VAD_SAMPLES, sizeof *pad);
        float *p1 = malloc(frames * sizeof *p1), *p2 = malloc(frames * sizeof *p2);
        pk_seg *g1 = malloc(cap * sizeof *g1), *g2 = malloc(cap * sizeof *g2);
        pk_vad *v = pk_vad_new(m);
        pk_asr *b = pk_asr_new(m, 6);
        const int made = pad && p1 && p2 && g1 && g2 && v && b;
        CHECK(made, "pk_asr_vad: made");
        if (made) {
            memcpy(pad, x, nn * sizeof *x);
            pk_vad_probs(v, pad, frames, p1);
            int ok = 1;
            for (int k = 0; k < 2; k++)
                ok &= pk_asr_vad(b, x, nn, p2) == (long)frames && !memcmp(p1, p2, frames * sizeof *p1);
            CHECK(ok, "pk_asr_vad: every probability pk_vad_probs', twice, the last frame padded with silence");
            pk_seg_params sp;
            pk_seg_defaults(&sp);
            for (int k = 0; k < 2; k++) {
                const size_t ns = pk_segments(p1, frames, rate, (int64_t)nn, &sp, g1, cap);
                const long ns2 = pk_asr_speech(b, x, nn, k ? &sp : NULL, g2, cap);
                CHECK(ns > 1 && ns2 == (long)ns && !memcmp(g1, g2, ns * sizeof *g1),
                      k ? "pk_asr_speech: pk_segments' speech, at 32 ms pauses"
                        : "pk_asr_speech: pk_segments' speech, Silero's defaults");
                sp.min_silence_ms = 32;
            }
            CHECK(pk_asr_vad(b, x, 0, p2) == 0 && pk_asr_speech(b, x, 0, NULL, g2, cap) == 0,
                  "pk_asr_vad: no samples, no frames");
        }
        pk_asr_free(b);
        pk_vad_free(v);
        free(pad), free(p1), free(p2), free(g1), free(g2);
    }

    /* Silero's encoder over a pool (core/vad.c, pk_vad_probs_on): the
     * recording four times over, cut to a block and a frame past a run, so
     * that runs follow runs and the last block is one frame, then a second
     * call that carries the stream on. Every probability pk_vad_probs's. */
    {
        const size_t f1 = 4 * (n / PK_VAD_SAMPLES) / 32 * 32 + 1, f2 = 77;
        float *xs = calloc((f1 + f2) * PK_VAD_SAMPLES, sizeof *xs);
        float *p1 = malloc((f1 + f2) * sizeof *p1), *p2 = malloc((f1 + f2) * sizeof *p2);
        pk_vad *s = pk_vad_new(m), *t = pk_vad_new(m);
        pk_pool *pool = pk_pool_new(8);
        CHECK(xs && p1 && p2 && s && t && pool, "VAD over a pool: made");
        if (xs && p1 && p2 && s && t && pool) {
            for (size_t i = 0; i < (f1 + f2) * PK_VAD_SAMPLES; i++)
                xs[i] = x[i % n];
            pk_vad_probs(s, xs, f1, p1);
            pk_vad_probs(s, xs + f1 * PK_VAD_SAMPLES, f2, p1 + f1);
            pk_vad_probs_on(t, pool, xs, f1, p2);
            pk_vad_probs_on(t, pool, xs + f1 * PK_VAD_SAMPLES, f2, p2 + f1);
            CHECK(!memcmp(p1, p2, (f1 + f2) * sizeof *p1),
                  "VAD over a pool: every probability one thread's, bit for bit");
        }
        pk_pool_free(pool);
        pk_vad_free(s), pk_vad_free(t);
        free(xs), free(p1), free(p2);
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
        sr.m = m;
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
            int64_t head = 0;
            size_t k = seam(&sr, &ch, 3, &head);
            CHECK(k == 2 && sr.t[0].id == was && sr.t[1].id == no, "seam: the word written twice goes");
            CHECK(head == 3, "seam: the retry kept off the head the last chunk's words cover");
            /* the last chunk heard "the no", this one "The no" */
            sr.seam[0] = (pk_tok){ the, 0, 1 };
            sr.seam_at[0] = ch.start;
            sr.seam_at[1] = ch.start + 3 * spf;
            sr.t[0] = (pk_tok){ The, 1, 1 };
            sr.t[1] = (pk_tok){ no, 3, 1 };
            k = seam(&sr, &ch, 2, &head);
            CHECK(k == 2 && sr.t[0].id == the && sr.t[0].frame == 0 && sr.t[1].id == no,
                  "seam: the first word's case as the last chunk wrote it");
            /* this one's words start apart from the last chunk's */
            sr.t[0] = (pk_tok){ The, 20, 1 };
            sr.t[1] = (pk_tok){ was, 22, 1 };
            k = seam(&sr, &ch, 2, &head);
            CHECK(k == 2 && sr.t[0].id == The && sr.t[1].id == was, "seam: words that start apart as they are");
            /* the cut fell in the last chunk's last word, "the", which ends
             * 4 frames into this chunk; this one opens "The no" */
            sr.ntail = 1;
            sr.tail[0] = (pk_tok){ the, 0, 1 };
            sr.tail_end = ch.start + 4 * spf;
            sr.t[0] = (pk_tok){ The, 1, 1 };
            sr.t[1] = (pk_tok){ no, 3, 1 };
            k = repeat(&sr, &ch, 2);
            CHECK(k == 1 && sr.t[0].id == no, "seam: the word the cut fell in, written again, goes");
            sr.t[0] = (pk_tok){ The, 5, 1 };
            sr.t[1] = (pk_tok){ no, 7, 1 };
            k = repeat(&sr, &ch, 2);
            CHECK(k == 2 && sr.t[0].id == The, "seam: that word starting after the last chunk's copy stays");
            sr.t[0] = (pk_tok){ was, 1, 1 };
            k = repeat(&sr, &ch, 2);
            CHECK(k == 2 && sr.t[0].id == was, "seam: another word there stays");
            /* the retry's level: speech between two words in a chunk 6 dB
             * under the recording's, 0.6 dB under the chunk's and then 4.5 */
            pk_seg speech = { ch.start, ch.start + 250 * spf };
            float *xs = malloc((size_t)ch.end * sizeof *xs);
            sr.st = malloc(8 * sizeof *sr.st);
            sr.text = malloc(sr.textcap = 256);
            const float level[2] = { 0.09f, 0.05f };
            for (int q = 0; q < 2 && xs && sr.st && sr.text; q++) {
                for (int64_t j = 0; j < ch.end; j++) {
                    const int64_t f = (j - ch.start) / spf;
                    xs[j] = j < ch.start || f >= 250 ? 0 : f >= 100 && f < 200 ? level[q] : 0.1f;
                }
                sr.x = xs, sr.sp = &speech, sr.ns = 1;
                sr.floor = 0.04 * RETRY_LEVEL;
                sr.t[0] = (pk_tok){ the, 0, 100 };
                sr.t[1] = (pk_tok){ no, 200, 50 };
                const long ns = stretches(&sr, &ch, sr.t, 2, 0);
                if (q == 0)
                    CHECK(ns == 1 && sr.st[0][0] == 100 && sr.st[0][1] == 200,
                          "retry: speech near its chunk's, under the recording's, heard again");
                else
                    CHECK(ns == 0, "retry: speech under its chunk's too, not");
            }
            free(xs);
            free(sr.st);
            free(sr.text);
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
        printf("transcheck: %zu chunks the low-level path's, a stop, two lanes one's, the steps "
               "pk_asr_transcribe's, a pool's VAD one thread's, the retry's and the seam's parts: ok\n",
               lo.n);
    forget(&hi), forget(&lo);
    pk_asr_free(a);
    pk_model_free(m);
    free(x);
    return fails != 0;
}
