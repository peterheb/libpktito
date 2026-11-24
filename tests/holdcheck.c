/* holdcheck DIARIZER WAV...: the held run as the recording is heard
 * (pk_hold_new, pk_hold_to, pk_hold_end) against pk_diar_run_held on the
 * same recording read whole. Its result is the same bits however much of
 * the recording each wait gives (no more than was asked, all of it, or any
 * amount between) and however many rows each pk_hold_to asks for; a row
 * before *from is the one the call before gave; no sample is read before
 * a wait has said it is there; and the rows given, when nothing was held
 * and no two speakers joined, are the result's. A wait that fails fails
 * the call. tests/check-c.sh runs it on conv2.wav, which holds no one; on
 * a recording that does (docs/SHORT_SPEAKERS.md's podcasts) it is the
 * check that a going back gives the same rows in pk_hold_to's run. */
#include "pktito.h"
#include "wav.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, what) do { if (!(c)) { printf("holdcheck: FAIL - %s\n", what); fails++; } } while (0)

typedef struct {
    const float *x;
    int64_t n;
    int how;                    /* 0: what was asked for; 1: all; 2: any amount between */
    uint64_t rs;
    int64_t most;               /* the most wait has said is there */
    int fail_at;                /* the wait that fails, from 1; 0: none */
    int waits, bad_read;
} grow;

static uint64_t next(uint64_t *r) {
    *r ^= *r << 13, *r ^= *r >> 7, *r ^= *r << 17;
    return *r;
}

static int grow_read(void *arg, int64_t at, float *out, size_t count) {
    grow *g = arg;
    if (at < 0 || at + (int64_t)count > g->most)
        g->bad_read = 1;
    if (at < 0 || at + (int64_t)count > g->n)
        return 1;
    memcpy(out, g->x + at, count * sizeof *out);
    return 0;
}

static int64_t grow_wait(void *arg, int64_t want) {
    grow *g = arg;
    if (g->fail_at && ++g->waits >= g->fail_at)
        return -1;
    int64_t got = want < g->n ? want : g->n;
    if (g->how == 1)
        got = g->n;
    else if (g->how == 2 && got < g->n)
        got += (int64_t)(next(&g->rs) % (uint64_t)(g->n - got + 1));
    if (got > g->most)
        g->most = got;
    return got;
}

static int same(const pk_held *a, const pk_held *b, size_t S) {
    if (a->rows != b->rows || a->n_held != b->n_held || a->again != b->again)
        return 0;
    if (memcmp(a->p, b->p, a->rows * S * sizeof *a->p) || memcmp(a->joined, b->joined, S * sizeof *a->joined))
        return 0;
    for (size_t i = 0; i < a->n_held; i++)
        if (a->held[i].start != b->held[i].start || a->held[i].end != b->held[i].end
            || a->held[i].speaker != b->held[i].speaker)
            return 0;
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: holdcheck DIARIZER WAV...\n");
        return 2;
    }
    char err[256];
    pk_model *m = pk_model_load(argv[1], err, sizeof err);
    pk_diar *d = m ? pk_diar_new(m, 0) : NULL;
    if (!d) {
        fprintf(stderr, "holdcheck: %s\n", m ? "not a diarizer" : err);
        return 2;
    }
    const size_t S = (size_t)pk_model_speakers(m);
    size_t runs = 0;
    for (int f = 2; f < argc; f++) {
        pk_wav_info info;
        pk_wav *w = pk_wav_open(argv[f], &info, err, sizeof err);
        float *x = w ? malloc(((size_t)info.frames + 1) * sizeof *x) : NULL;
        if (!x || info.rate != (uint32_t)pk_model_rate(m)) {
            fprintf(stderr, "holdcheck: %s: %s\n", argv[f], w ? "not at the model's rate" : err);
            return 2;
        }
        const size_t n = pk_wav_read(w, x, (size_t)info.frames);
        pk_wav_close(w);
        const pk_audio whole_x = pk_audio_floats(x, n);
        pk_held whole;
        if (pk_diar_run_held(d, m, &whole_x, &whole)) {
            fprintf(stderr, "holdcheck: %s: pk_diar_run_held failed\n", argv[f]);
            return 2;
        }
        const int plain = !whole.n_held;
        float *shown = calloc(whole.rows * S + 1, sizeof *shown);
        uint64_t rs = 0x9e3779b97f4a7c15ull + (uint64_t)f;
        for (int how = 0; how < 3; how++)
            for (int asks = 0; asks < 3; asks++) {
                grow g = { x, (int64_t)n, how, rs ^ (uint64_t)(how * 3 + asks + 1), 0, 0, 0, 0 };
                const pk_audio growing = { -1, grow_read, &g, grow_wait };
                const pk_audio *xs = how == 1 && asks == 0 ? &whole_x : &growing;
                pk_hold *h = pk_hold_new(d, m, xs);
                CHECK(h, "pk_hold_new");
                if (!h)
                    continue;
                /* rows asked for: one at a time, a step's worth or so, or all at once */
                int64_t given = 0, want = 0;
                int ok = 1;
                while (ok && (asks == 2 ? want < INT64_MAX : given < (int64_t)whole.rows)) {
                    want = asks == 2 ? INT64_MAX
                         : asks == 0 ? given + 1 : given + 1 + (int64_t)(next(&rs) % 700);
                    const float *p;
                    int64_t from;
                    const int64_t got = pk_hold_to(h, want, &p, &from);
                    if (got < 0) {
                        CHECK(0, "pk_hold_to failed");
                        ok = 0;
                        break;
                    }
                    CHECK(from >= 0 && from <= got && got <= (int64_t)whole.rows, "from <= rows given <= the recording's");
                    CHECK(got >= (want < (int64_t)whole.rows ? want : (int64_t)whole.rows) || got == (int64_t)whole.rows,
                          "the rows asked for are given");
                    CHECK(!memcmp(p, shown, (size_t)from * S * sizeof *p), "rows before from are as they were");
                    memcpy(shown + (size_t)from * S, p + (size_t)from * S, (size_t)(got - from) * S * sizeof *p);
                    given = got;
                    if (given == (int64_t)whole.rows)
                        break;
                }
                if (plain) {
                    int joined = 0;
                    for (size_t s = 0; s < S; s++)
                        joined |= whole.joined[s] != (int32_t)s;
                    if (!joined)
                        CHECK(!memcmp(shown, whole.p, whole.rows * S * sizeof *shown),
                              "with no one held nor joined, the rows given are the result's");
                }
                pk_held part;
                CHECK(!pk_hold_end(h, &part), "pk_hold_end");
                CHECK(same(&part, &whole, S), "pk_hold_to's run gives pk_diar_run_held's result, bit for bit");
                CHECK(!g.bad_read, "no sample is read before a wait has said it is there");
                pk_held_free(&part);
                pk_hold_free(h);
                runs++;
            }
        /* a wait that fails: the first, and one later on */
        for (int at = 1; at <= 3; at += 2) {
            grow g = { x, (int64_t)n, 0, rs, 0, at, 0, 0 };
            const pk_audio growing = { -1, grow_read, &g, grow_wait };
            pk_hold *h = pk_hold_new(d, m, &growing);
            pk_held part;
            CHECK(h && pk_hold_end(h, &part) == -1, "a wait that fails fails the call");
            pk_hold_free(h);
        }
        printf("holdcheck: %s: %zu rows, %zu held, %d ways asked: the same\n", argv[f], whole.rows, whole.n_held, 9);
        free(shown);
        pk_held_free(&whole);
        free(x);
    }
    pk_diar_free(d);
    pk_model_free(m);
    if (fails)
        return 1;
    printf("holdcheck: %zu runs, all pk_diar_run_held's\n", runs);
    return 0;
}
