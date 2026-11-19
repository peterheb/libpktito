/* growcheck [--all] MODEL WAV...: pk_asr_transcribe of a recording still
 * being read (pk_audio's n < 0 and wait) against the same recording read
 * whole. Each chunk's final tokens are the whole recording's, bit for bit,
 * its place and at_frame too, however much of the recording is there at
 * each wait: no more than was asked for, all of it, or any amount between;
 * with and without the retry and the context, at two chunk lengths, on
 * every core (with lanes, where there are eight) and on one thread. A
 * chunk comes in order, once, or twice where its first coming says so
 * (provisional), the second after every chunk has come once; and n and
 * total are 0 until then. A run stopped by fn stops, a wait that fails
 * fails the call, and a recording of no samples is no chunks. --all is
 * every one of four thread counts, three chunk lengths and four modes on
 * the first recording, and the defaults on the rest: for a corpus. */
#include "pktito.h"
#include "wav.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, what) do { if (!(c)) { printf("growcheck: FAIL - %s\n", what); fails++; } } while (0)

#define MOST 4096

typedef struct {
    size_t n;                   /* chunks seen, by their highest index */
    pk_seg at[MOST];
    pk_tok *t[MOST];
    long nt[MOST];
    int32_t at_frame[MOST];
    int times[MOST], provisional[MOST], retried[MOST];
    int counted;                /* a first coming had its n: the recording ended before its first chunk was cut */
    int order_ok, late_ok, counts_ok;
    size_t firsts;              /* first comings so far */
    int in_late;
    size_t stop_after;
    size_t calls;
} seen;

static int on_out(void *arg, const pk_chunk_out *c) {
    seen *s = arg;
    s->calls++;
    if (c->i >= MOST)
        return 1;
    const size_t i = c->i;
    if (s->times[i] == 0) {
        s->order_ok &= i == s->firsts && !s->in_late;
        s->firsts++;
        s->provisional[i] = c->provisional;
        s->counted |= c->n != 0;
        s->counts_ok &= (c->n != 0) == s->counted && (c->n != 0) == (c->total != 0);
    } else {
        s->in_late = 1;
        s->late_ok &= s->times[i] == 1 && s->provisional[i] == 1 && !c->provisional;
    }
    if (c->provisional)
        s->counts_ok &= c->n == 0 && c->total == 0;
    s->times[i]++;
    free(s->t[i]);
    s->t[i] = malloc(((size_t)c->count + 1) * sizeof *c->toks);
    memcpy(s->t[i], c->toks, (size_t)c->count * sizeof *c->toks);
    s->nt[i] = c->count;
    s->at[i] = c->at;
    s->at_frame[i] = c->at_frame;
    s->retried[i] = c->retried;
    s->n = i + 1 > s->n ? i + 1 : s->n;
    return s->stop_after && s->calls == s->stop_after;
}

static void clear(seen *s) {
    for (size_t i = 0; i < MOST; i++)
        free(s->t[i]);
    memset(s, 0, sizeof *s);
    s->order_ok = s->late_ok = s->counts_ok = 1;
}

static int same(const seen *a, const seen *b) {
    if (a->n != b->n)
        return 0;
    for (size_t i = 0; i < a->n; i++)
        if (a->nt[i] != b->nt[i] || a->at[i].start != b->at[i].start || a->at[i].end != b->at[i].end
            || a->at_frame[i] != b->at_frame[i] || a->retried[i] != b->retried[i]
            || memcmp(a->t[i], b->t[i], (size_t)a->nt[i] * sizeof *a->t[i]))
            return 0;
    return 1;
}

/* A recording that is there a part at a time */
typedef struct {
    const float *x;
    int64_t n;
    int how;                    /* 0: what was asked for; 1: all; 2: any amount between */
    _Atomic uint64_t rs;
    _Atomic int64_t most;       /* the most wait has said is there */
    int fail_at;                /* the wait that fails, from 1; 0: none */
    _Atomic int waits, bad_read;
} grow;

static int grow_read(void *arg, int64_t at, float *out, size_t count) {
    grow *g = arg;
    /* only what a wait has said is there is asked for */
    if (at < 0 || at + (int64_t)count > atomic_load(&g->most))
        atomic_store(&g->bad_read, 1);
    if (at < 0 || at + (int64_t)count > g->n)
        return 1;
    memcpy(out, g->x + at, count * sizeof *out);
    return 0;
}

/* called by several lanes at once */
static int64_t grow_wait(void *arg, int64_t want) {
    grow *g = arg;
    const int k = atomic_fetch_add(&g->waits, 1) + 1;
    if (g->fail_at && k >= g->fail_at)
        return -1;
    int64_t got = want < g->n ? want : g->n;
    if (g->how == 1) {
        got = g->n;
    } else if (g->how == 2 && got < g->n) {
        uint64_t r = atomic_load(&g->rs);
        r ^= r << 13, r ^= r >> 7, r ^= r << 17;
        atomic_store(&g->rs, r);
        got += (int64_t)(r % (uint64_t)(g->n - got + 1));
    }
    int64_t was = atomic_load(&g->most);
    while (got > was && !atomic_compare_exchange_weak(&g->most, &was, got))
        ;
    return got;
}

int main(int argc, char **argv) {
    const int all = argc > 1 && !strcmp(argv[1], "--all");
    argv += all, argc -= all;
    if (argc < 3) {
        fprintf(stderr, "usage: growcheck [--all] MODEL WAV...\n");
        return 2;
    }
    char err[256];
    pk_model *m = pk_model_load(argv[1], err, sizeof err);
    if (!m) {
        fprintf(stderr, "growcheck: %s\n", err);
        return 2;
    }
    static seen whole, part;
    size_t runs = 0, late = 0, chunks = 0;
    for (int f = 2; f < argc; f++) {
        pk_wav_info info;
        pk_wav *w = pk_wav_open(argv[f], &info, err, sizeof err);
        float *x = w ? malloc(((size_t)info.frames + 1) * sizeof *x) : NULL;
        if (!x || info.rate != (uint32_t)pk_model_rate(m)) {
            fprintf(stderr, "growcheck: %s: %s\n", argv[f], w ? "not at the model's rate" : err);
            return 2;
        }
        const size_t n = pk_wav_read(w, x, (size_t)info.frames);
        pk_wav_close(w);
        const int full = all && f == 2;
        static const int threads[] = { 0, 1, 4, 8 };
        static const double lens[] = { 30.72, 12, 5 };
        for (int ti = 0; ti < (full ? 4 : all ? 1 : 2); ti++) {
            pk_asr *a = pk_asr_new(m, threads[ti]);
            if (!a)
                return 2;
            for (int li = 0; li < (full ? 3 : all ? 1 : 2); li++)
                for (int mode = 0; mode < (full ? 4 : 1); mode++) {
                    if (!all && ti != li)       /* every core at 30.72 s, one thread at 12 */
                        continue;
                    pk_transcribe_opts o;
                    pk_transcribe_defaults(&o);
                    o.max_chunk_s = lens[li];
                    o.retry = !(mode & 1);
                    o.context = !(mode & 2);
                    pk_asr_set_fast_mode(a, mode == 3);
                    clear(&whole);
                    const pk_audio known = pk_audio_floats(x, n);
                    CHECK(pk_asr_transcribe(a, &known, &o, on_out, &whole) == 0, "the recording whole");
                    for (int how = 0; how < 3; how++) {
                        if (!all && ti && how != 2)
                            continue;
                        grow g = { x, (int64_t)n, how, 88172645463325252ull + (uint64_t)runs, 0, 0, 0, 0 };
                        const pk_audio growing = { -1, grow_read, &g, grow_wait };
                        clear(&part);
                        CHECK(pk_asr_transcribe(a, &growing, &o, on_out, &part) == 0, "the recording as it comes");
                        CHECK(same(&whole, &part), "its chunks' tokens are the whole recording's");
                        CHECK(part.order_ok, "each chunk comes first in its turn");
                        CHECK(part.late_ok, "a chunk comes again once, and only where it said it would");
                        CHECK(part.counts_ok, "n and total are 0 while a chunk is provisional");
                        CHECK(!g.bad_read, "no sample is read before a wait has said it is there");
                        for (size_t i = 0; i < part.n; i++) {
                            CHECK(part.times[i] == 1 + part.provisional[i], "a provisional chunk comes twice, another once");
                            /* a chunk that did not say so has no retry in the whole
                             * recording's either; a recording that ended before
                             * its first chunk was cut is read whole from there,
                             * its chunks counted, and none of them says so */
                            CHECK(part.counted ? !part.provisional[i] : part.provisional[i] || !whole.retried[i],
                                  "a retried chunk was provisional, in a recording cut before its end");
                            late += part.provisional[i];
                        }
                        chunks += part.n;
                        runs++;
                    }
                }
            pk_asr_set_fast_mode(a, 0);
            if (f == 2 && ti == 0) {
                /* stopped by fn at its first chunk, and at one more than the chunks */
                for (size_t stop = 1; stop <= whole.n + 1; stop += whole.n) {
                    grow g = { x, (int64_t)n, 0, 1, 0, 0, 0, 0 };
                    const pk_audio growing = { -1, grow_read, &g, grow_wait };
                    clear(&part);
                    part.stop_after = stop;
                    const int rc = pk_asr_transcribe(a, &growing, NULL, on_out, &part);
                    CHECK(part.calls <= stop && (rc == 1) == (part.calls == stop), "fn stops the run");
                }
                /* a wait that fails: the first, and the one that would find the end */
                for (int k = 1; k <= 2; k++) {
                    grow g = { x, (int64_t)n, 0, 1, 0, k, 0, 0 };
                    const pk_audio growing = { -1, grow_read, &g, grow_wait };
                    clear(&part);
                    CHECK(pk_asr_transcribe(a, &growing, NULL, on_out, &part) == -1, "a wait that fails fails the call");
                }
                /* no samples, and no wait */
                grow g = { x, 0, 0, 1, 0, 0, 0, 0 };
                const pk_audio none = { -1, grow_read, &g, grow_wait }, bad = { -1, grow_read, &g, NULL };
                clear(&part);
                CHECK(pk_asr_transcribe(a, &none, NULL, on_out, &part) == 0 && part.calls == 0, "no samples: no chunks");
                CHECK(pk_asr_transcribe(a, &bad, NULL, on_out, &part) == -1, "no length and no wait: refused");
                CHECK(pk_steps_new(a, &none, NULL) == NULL, "pk_steps takes a recording whose length is known");
            }
            pk_asr_free(a);
        }
        free(x);
    }
    clear(&whole);
    clear(&part);
    pk_model_free(m);
    if (fails)
        return 1;
    printf("growcheck: ok (%zu runs, %zu chunks, %zu of them provisional)\n", runs, chunks, late);
    return 0;
}
