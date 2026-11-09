/* A transcriber: its memory, its pool, and audio in, tokens out.
 *
 * The working memory is one block, sized for the longest recording seen so
 * far and grown when a longer one comes: a minute of audio needs about 60
 * MB of it at Parakeet 0.6B's widths, most of it the encoder's activations.
 */
#include "asr.h"
#include "kernels.h"
#include "mel.h"
#include "os.h"

#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#ifdef PK_ANE
#include "ane.h"
#include <dispatch/dispatch.h>
#endif
#ifdef PK_QNN
#include "qnn.h"
#endif

static size_t up16z(size_t x) { return (x + 15) & ~(size_t)15; }

/* The block's arrays, each 64-byte aligned: a first pass counts from 0, a
 * second assigns them from the block's aligned start, the same offsets. */
#define CARVE(dst, n, sz) do {                                         \
        uintptr_t at_ = (p + 63) & ~(uintptr_t)63;                     \
        p = at_ + (size_t)(n) * (sz);                                  \
        if (pass) (dst) = (void *)at_;                                 \
    } while (0)

static int reserve(pk_asr *a, size_t T0) {
    if (T0 <= a->cap)
        return 0;
    const pk_model *m = a->m;
    const size_t Te = pk_enc_frames((uint32_t)T0), D = m->d_model, F = m->ff_dim;
    const size_t C = m->sub_channels, M = m->n_mels, J = m->joint_hidden;
    const size_t H = m->n_heads, dk = D / H, F1 = (M + 1) / 2, F2 = (F1 + 1) / 2, F3 = M / 8;
    const size_t U = F > 2 * D ? F : 2 * D, R = 2 * Te - 1;
    const size_t Tp = up16z(Te), Pp = up16z(R) + 16;
    const size_t widest = F3 * C > D ? F3 * C : D;     /* of a->act's inputs */
    /* the subsampling's windows (core/encoder.c, sub_item) */
    const size_t s1 = 3 * F1 * C, s2 = F2 * C, s3 = 3 * F2 * C, s4 = PK_SUB_BLOCK * F3 * C;
    const size_t sq = s2 > s4 ? s2 : s4;

    void *mem = NULL;
    for (int pass = 0; pass < 2; pass++) {
        uintptr_t p = pass ? ((uintptr_t)mem + 63) & ~(uintptr_t)63 : 0;
        CARVE(a->mel_scratch, pk_mel_scratch(m, T0), sizeof(float));
        CARVE(a->feat, T0 * M, sizeof(float));
        CARVE(a->win, T0, sizeof(float *));
        CARVE(a->s5, Te * F3 * C, sizeof(float));
        CARVE(a->x, Te * D, sizeof(float));
        CARVE(a->h, Te * D, sizeof(float));
        CARVE(a->u, Te * U, sizeof(float));
        CARVE(a->q, Te * D, sizeof(float));
        CARVE(a->o, Te * D, sizeof(float));
        CARVE(a->pe, R * D, sizeof(float));
        CARVE(a->p, R * D, sizeof(float));
        CARVE(a->kp, H * Tp * dk, sizeof(float));
        CARVE(a->vp, Te * D, sizeof(float));
        CARVE(a->pp, H * Pp * dk, sizeof(float));
        CARVE(a->act.q, Te * widest, 1);
        CARVE(a->act.d, Te * widest / 32, sizeof(float));
        CARVE(a->ape.q, R * D, 1);
        CARVE(a->ape.d, R * D / 32, sizeof(float));
        CARVE(a->hid.q, Te * F, 1);
        CARVE(a->hid.d, Te * F / 32, sizeof(float));
        CARVE(a->fproj[0], Te * J, sizeof(float));
        CARVE(a->fproj[1], Te * J, sizeof(float));
        for (int t = 0; t < a->nth; t++) {
            pk_thread_ws *w = &a->tws[t];
            CARVE(w->s1, s1, sizeof(float));
            CARVE(w->s2, s2, sizeof(float));
            CARVE(w->s3, s3, sizeof(float));
            CARVE(w->s4, s4, sizeof(float));
            CARVE(w->aq, sq, 1);
            CARVE(w->ad, sq / 32, sizeof(float));
            CARVE(w->qu, PK_ATT_BLOCK * dk, sizeof(float));
            CARVE(w->qv, PK_ATT_BLOCK * dk, sizeof(float));
            CARVE(w->sc, PK_ATT_BLOCK * Tp, sizeof(float));
            CARVE(w->bd, PK_ATT_BLOCK * (Tp + 32), sizeof(float));
            CARVE(w->mel, pk_mel_thread_scratch(m), sizeof(float));
            CARVE(w->row, D > F ? D : F, sizeof(float));
        }
        if (!pass) {
            mem = malloc((size_t)p + 64);
            if (!mem)
                return -1;
            free(a->mem);
            a->mem = mem;
        }
    }
    a->cap = T0;
    return 0;
}

/* A transcriber of one lane: pk_asr_new's, and each of its other lanes */
static pk_asr *transcriber(const pk_model *m, int threads) {
    if (!m)
        return NULL;
    pk_asr *a = calloc(1, sizeof *a);
    if (!a)
        return NULL;
    a->m = m;
    a->nth = threads > 0 ? threads : pk_os_cores();
    a->pool = pk_pool_new(a->nth);
    a->tws = calloc((size_t)a->nth, sizeof *a->tws);
    const size_t P = m->pred_hidden, L = m->pred_layers, J = m->joint_hidden;
    const size_t wide = P > J ? P : J;
    a->dh = calloc(L * P, sizeof(float));
    a->dc = calloc(L * P, sizeof(float));
    a->gates = calloc(4 * P, sizeof(float));
    a->gp = calloc(J, sizeof(float));
    a->z = calloc(J, sizeof(float));
    a->logit = calloc(m->n_out_pad, sizeof(float));
    a->dact.q = calloc(wide, 1);
    a->dact.d = calloc(wide, sizeof(float));
    if (!a->pool || !a->tws || !a->dh || !a->dc || !a->gates || !a->gp || !a->z || !a->logit
        || !a->dact.q || !a->dact.d) {
        pk_asr_free(a);
        return NULL;
    }
    a->nth = pk_pool_threads(a->pool);
    a->wide = a->nth;
    a->fast = threads > 0 ? a->nth : pk_os_fast_cores();
    a->tune = a->fast > 0 && a->fast < a->nth ? PK_TUNE_AT : 0;
    return a;
}

pk_asr *pk_asr_new(const pk_model *m, int threads) {
    pk_asr *a = transcriber(m, threads);
#ifndef PK_PROFILE                       /* whose stage times are one thread's */
    if (a && a->nth / PK_LANE_THREADS > 1) {
        a->lanes = a->nth / PK_LANE_THREADS < PK_LANES_MAX ? a->nth / PK_LANE_THREADS : PK_LANES_MAX;
        if (!(a->lane = calloc((size_t)a->lanes, sizeof *a->lane))) {
            pk_asr_free(a);
            return NULL;
        }
    }
#endif
    return a;
}

int pk_asr_threads(const pk_asr *a) { return a->wide; }

void pk_asr_set_fast_mode(pk_asr *a, int on) { a->qb64 = on != 0; }

int pk_asr_use_ane(pk_asr *a, const char *path, char *err, size_t errlen) {
#ifdef PK_ANE
    pk_ane *e = NULL;
    if (path && !(e = pk_ane_open(path, a->m, err, errlen)))
        return -1;
    pk_ane_free(a->ane);
    a->ane = e;
    return 0;
#else
    (void)a;
    (void)path;
    if (err && errlen)
        snprintf(err, errlen, "this pktito was built without ANE=1");
    return -1;
#endif
}

int pk_asr_ane_warm(pk_asr *a, char *err, size_t errlen) {
#ifdef PK_ANE
    if (!a->ane) {
        if (err && errlen)
            snprintf(err, errlen, "no Core ML model given");
        return -1;
    }
    return pk_ane_warm(a->ane, err, errlen);
#else
    (void)a;
    if (err && errlen)
        snprintf(err, errlen, "this pktito was built without ANE=1");
    return -1;
#endif
}

int pk_asr_use_qnn(pk_asr *a, int on, const char *cache, char *err, size_t errlen) {
#ifdef PK_QNN
    pk_qnn *e = NULL;
    if (on && !(e = pk_qnn_open(a->m, 0, cache, err, errlen)))
        return -1;
    pk_qnn_free(a->qnn);
    a->qnn = e;
    return 0;
#else
    (void)a;
    (void)on;
    (void)cache;
    if (err && errlen)
        snprintf(err, errlen, "this pktito was built without QNN=1");
    return -1;
#endif
}

static int ends_with(const char *s, size_t n, const char *suffix) {
    const size_t k = strlen(suffix);
    return n >= k && !memcmp(s + n - k, suffix, k);
}

static int refuse(char *err, size_t errlen, const char *why) {
    if (err && errlen)
        snprintf(err, errlen, "%s", why);
    return -1;
}

int pk_asr_use_encoder(pk_asr *a, const char *path, char *err, size_t errlen) {
    char p[4096];
    size_t n = path ? strlen(path) : 0;
    while (n > 1 && (path[n - 1] == '/' || path[n - 1] == '\\'))    /* a folder given with its slash */
        n--;
    if (!n || n >= sizeof p)
        return refuse(err, errlen, n ? "path too long" : "no path");
    memcpy(p, path, n);
    p[n] = 0;
    if (ends_with(p, n, ".mlmodelc"))
        return pk_asr_use_ane(a, p, err, errlen);
    if (ends_with(p, n, ".qnn")) {
        /* a graph that isn't there would have pk_asr_use_qnn compile one
         * from the whole model and write it there, 714 MB in a minute or
         * two: what --qnn asks for, and not what naming a file does */
        FILE *f = pk_os_fopen(p, "rb");
        if (!f)
            return refuse(err, errlen, "no such file");
        fclose(f);
        return pk_asr_use_qnn(a, 1, p, err, errlen);
    }
    return refuse(err, errlen, ends_with(p, n, ".aar") ? "a Core ML model still in its archive: aa extract it first"
                                                       : "not an encoder: a Core ML model (.mlmodelc) or a QNN graph (.qnn)");
}

int pk_qnn_chip(char *chip, size_t cap, char *err, size_t errlen) {
#ifdef PK_QNN
    return pk_qnn_target(chip, cap, err, errlen);
#else
    (void)chip;
    (void)cap;
    if (err && errlen)
        snprintf(err, errlen, "this pktito was built without QNN=1");
    return -1;
#endif
}

void pk_asr_free(pk_asr *a) {
    if (!a)
        return;
    pk_pool_free(a->drive);
    for (int j = 1; a->lane && j < a->lanes; j++)
        pk_asr_free(a->lane[j]);
    free(a->lane);
    pk_vad_free(a->vad);
    pk_pool_free(a->pool);
#ifdef PK_ANE
    pk_ane_free(a->ane);
    for (int k = 0; k < 4; k++) {
        free(a->ane_feat[k]);
        free(a->ane_x[k]);
    }
#endif
#ifdef PK_QNN
    pk_qnn_free(a->qnn);
#endif
    free(a->tws);
    free(a->mem);
    free(a->dh);
    free(a->dc);
    free(a->gates);
    free(a->gp);
    free(a->z);
    free(a->logit);
    free(a->dact.q);
    free(a->dact.d);
    free(a->toks);
    free(a->pcm);
    free(a);
}

/* A chunk up to its search: its features, the encoder, and the joint's
 * encoder half of its frames, into f. Returns its encoder frames, or -1
 * where the encoder can't run: a slim model's with no accelerator's
 * encoder given, the accelerator failing it, or an input longer than it
 * takes. */
static long encode(pk_asr *a, const float *x, size_t n, float *f) {
    const size_t T0 = pk_mel_frames(a->m, n);
    if (T0 == 0)
        return 0;
    PROF(PK_P_MEL, pk_mel(a, x, n));
    uint32_t Te = pk_encode(a, (uint32_t)T0);
    if (!Te)
        return -1;
    PROF(PK_P_DEC, pk_joint_frames(a, Te, f));
    return Te;
}

/* Room in a->pcm for n samples: 0, or -1 if memory runs out. Nothing in it
 * outlives a chunk, so it is made again, not copied. */
static int reserve_pcm(pk_asr *a, size_t n) {
    if (n <= a->pcm_cap)
        return 0;
    free(a->pcm);
    a->pcm_cap = 0;
    if (n > SIZE_MAX / sizeof *a->pcm || !(a->pcm = malloc(n * sizeof *a->pcm)))
        return -1;
    a->pcm_cap = n;
    return 0;
}

/* encode for chunk c of x, its samples read into a->pcm first: -1 also
 * where the read fails */
static long encode_at(pk_asr *a, const pk_audio *x, const pk_seg *c, float *f) {
    const size_t n = (size_t)(c->end - c->start);
    if (n > a->pcm_cap || pk_audio_read(x, c->start, a->pcm, n))
        return -1;
    return encode(a, a->pcm, n, f);
}

long pk_asr_run(pk_asr *a, const float *x, size_t n, const pk_tok **toks) {
    const size_t T0 = pk_mel_frames(a->m, n);
    *toks = a->toks;
    a->ntoks = 0;
    if (T0 == 0)
        return 0;
    if (T0 > UINT32_MAX / 4 || reserve(a, T0))
        return -1;
    pk_os_throttle was = pk_os_unthrottle();  /* this thread runs a share of every job */
    const long Te = encode(a, x, n, a->fproj[0]);
    long nt = -1;
    if (Te >= 0)
        PROF(PK_P_DEC, nt = pk_decode(a, (uint32_t)Te, a->fproj[0], 1));
    pk_os_rethrottle(was);
    *toks = a->toks;
    return nt;
}

/* ----------------------------------------------------------------- lanes --- */

/* Lane j's threads of a transcriber's nth, in a run of k lanes */
static int share(int nth, int j, int k) { return nth / k + (j < nth % k); }

/* A lane's time for a chunk when k run at once, against one transcriber's
 * on every thread: k over the gain in rate that k lanes of 16 / k threads
 * gave a 9950X over 300 s of AliceBench, as processes, at k = 2, 3, 4, 8
 * and 16 (backend/avx2/CLIMB.md, 2026-11-02), between them by a straight
 * line. The least gain of the boards of 8 threads or more measured, where
 * an M5 Ultra's lanes gave 1.81 to 2.69 times its one transcriber's rate,
 * so that a run takes lanes only where they pay on the board that gains
 * least. */
static double lane_time(int k) {
    static const int at[] = { 1, 2, 3, 4, 8, 16 };
    static const double gain[] = { 1.00, 1.11, 1.22, 1.32, 1.35, 1.46 };
    int i = 1;
    while (i < 5 && at[i] < k)
        i++;
    const double g = gain[i - 1] + (gain[i] - gain[i - 1]) * (k - at[i - 1]) / (at[i] - at[i - 1]);
    return k / g;
}

/* The lanes a run of n chunks takes, of one to a->lanes: the count whose
 * run ends first, the chunks dealt in order to the lane free first, as
 * lane_job takes them, each taking its length times lane_time. One lane
 * where no count beats it. A recording's last chunk is often short, so
 * it counts lengths, not chunks: test.wav's three chunks, 28.6, 28.0 and
 * 3.2 s, take two lanes, where three would wait on the first chunk at a
 * third of the threads. */
static int lanes_for(const pk_asr *a, const pk_seg *c, size_t n) {
    int best = 1;
    double soonest = 0;
    for (int k = 1; k <= a->lanes && (size_t)k <= n; k++) {
        double ends[PK_LANES_MAX] = { 0 }, end = 0;     /* when each lane is free */
        for (size_t i = 0; i < n; i++) {
            int j = 0;
            for (int q = 1; q < k; q++)
                j = ends[q] < ends[j] ? q : j;
            ends[j] += (double)(c[i].end - c[i].start) * lane_time(k);
            end = ends[j] > end ? ends[j] : end;
        }
        if (k == 1 || end < soonest)
            best = k, soonest = end;
    }
    return best;
}

typedef struct {
    pk_asr *a;                       /* lane 0, the caller's */
    const pk_audio *x;               /* read in turns: pk_run_chunks_with */
    const pk_seg *c;
    size_t n;
    _Atomic size_t next;             /* the next chunk a lane takes */
    _Atomic int stop;                /* each stopped the run, or a lane ran out of memory */
    _Atomic int failed;              /* the latter */
    _Atomic int *done;               /* chunk i's tokens are toks[i], count[i] of them */
    pk_tok **toks;
    long *count;
    size_t given;                    /* chunks each has had: the calling thread's */
    int rc;                          /* each's last word: the calling thread's */
    pk_each_fn each;
    void *arg;
} lanes_run;

/* each for the chunks done since the last, in order, on the calling
 * thread between lane 0's chunks, where lane 0's second joint buffer is
 * free for a retry */
static void give(lanes_run *r) {
    while (r->rc == 0 && r->given < r->n
           && atomic_load_explicit(&r->done[r->given], memory_order_acquire)) {
        const size_t i = r->given++;
        r->rc = r->each(r->arg, i, r->toks[i], r->count[i], r->a->fproj[1]);
        free(r->toks[i]);
        r->toks[i] = NULL;
    }
    if (r->rc)
        atomic_store_explicit(&r->stop, 1, memory_order_relaxed);
}

/* Lane `thread`'s part of a run: the next chunk while there is one, its
 * encoder and then its search on the lane's threads, and its tokens kept
 * for each. Thread 0 is the caller's, and its lane is the transcriber. */
static void lane_job(void *arg, int item, int thread) {
    (void)item;
    lanes_run *r = arg;
    pk_asr *l = thread ? r->a->lane[thread] : r->a;
    while (!atomic_load_explicit(&r->stop, memory_order_relaxed)) {
        const size_t i = atomic_fetch_add_explicit(&r->next, 1, memory_order_relaxed);
        if (i >= r->n)
            break;
        const long Te = encode_at(l, r->x, &r->c[i], l->fproj[0]);
        const long nt = Te < 0 ? -1 : pk_decode(l, (uint32_t)Te, l->fproj[0], 1);
        pk_tok *t = nt > 0 ? malloc((size_t)nt * sizeof *t) : NULL;
        if (nt < 0 || (nt > 0 && !t)) {
            free(t);
            atomic_store_explicit(&r->failed, 1, memory_order_relaxed);
            atomic_store_explicit(&r->stop, 1, memory_order_relaxed);
            break;
        }
        if (nt)
            memcpy(t, l->toks, (size_t)nt * sizeof *t);
        r->toks[i] = t;
        r->count[i] = nt;
        atomic_store_explicit(&r->done[i], 1, memory_order_release);
        if (thread == 0)
            give(r);
    }
}

/* How many of k lanes a run can have, each with the memory for chunks of
 * `most` feature frames and `longest` samples: lane j > 0 is made the
 * first time a run needs it, with the most threads a run gives it, and
 * kept. Fewer where memory or a thread cannot be had, down to one: the
 * loop below. */
static int lanes_ready(pk_asr *a, int k, size_t most, size_t longest) {
    if (!a->drive && !(a->drive = pk_pool_new(a->lanes)))
        return 1;
    for (int j = 1; j < k; j++) {
        if (!a->lane[j]) {
            int w = 1;
            for (int q = j + 1; q <= a->lanes; q++)
                w = share(a->nth, j, q) > w ? share(a->nth, j, q) : w;
            a->lane[j] = transcriber(a->m, w);
        }
        if (!a->lane[j] || reserve(a->lane[j], most) || reserve_pcm(a->lane[j], longest))
            return j;
    }
    return k;
}

/* pk_run_chunks_with's chunks on k lanes at once, lane j on its share of
 * the transcriber's threads. The encoder's race between every core and
 * the fast kind waits for a run of one lane, whose timing the other lanes
 * would muddle. */
static int run_lanes(pk_asr *a, const pk_audio *x, const pk_seg *c, size_t n, int k,
                     pk_each_fn each, void *arg) {
    lanes_run r = { .a = a, .x = x, .c = c, .n = n, .each = each, .arg = arg };
    r.done = malloc(n * sizeof *r.done);
    r.toks = calloc(n, sizeof *r.toks);
    r.count = calloc(n, sizeof *r.count);
    if (!r.done || !r.toks || !r.count) {
        free(r.done), free(r.toks), free(r.count);
        return -1;
    }
    for (size_t i = 0; i < n; i++)
        atomic_init(&r.done[i], 0);
    atomic_init(&r.next, 0);
    atomic_init(&r.stop, 0);
    atomic_init(&r.failed, 0);

    const int wide = a->wide, tune = a->tune;
    for (int j = 0; j < k; j++) {
        pk_asr *l = j ? a->lane[j] : a;
        l->wide = share(a->nth, j, k);
        l->qb64 = a->qb64;                     /* the caller's mode, in every lane */
        pk_pool_set_active(l->pool, l->wide);
    }
    a->tune = 0;
    pk_os_throttle was = pk_os_unthrottle();  /* this thread runs lane 0 */
    pk_pool_set_active(a->drive, k);
    pk_pool_run(a->drive, lane_job, &r, k);
    a->wide = wide;
    a->tune = tune;
    pk_pool_set_active(a->pool, a->wide);
    give(&r);                                  /* what the other lanes did last */
    pk_os_rethrottle(was);
    const int rc = r.rc ? r.rc : atomic_load_explicit(&r.failed, memory_order_relaxed) ? -1 : 0;
    for (size_t i = 0; i < n; i++)
        free(r.toks[i]);
    free(r.done), free(r.toks), free(r.count);
    return rc;
}

/* ---------------------------------------------------------- one at a time --- */

/* A chunk's search, for the pool's last thread */
typedef struct {
    pk_asr *a;
    uint32_t Te;
    const float *f;
    long n;
} search_job;

static void search_aside(void *arg) {
    search_job *s = arg;
    s->n = pk_decode(s->a, s->Te, s->f, 0);
}

/* Chunk i's search, on the pool's last thread, while chunk i + 1's
 * features and encoder take the others, each chunk's joint half in a
 * buffer of its own; the last chunk's search alone, as pk_asr_run's.
 * Over test.wav's three chunks a run waited 2.8 ms on the decoder where
 * it had waited 11.8, on a 9950X at 16 threads, and ran 1.9 to 5.6%
 * faster, a GB10's X925s 2.0 to 2.8%, whose race leaves the last thread
 * idle. Where the encoder gives up a thread of its own for the search,
 * and shares its memory with the search's reads, it gives back some of
 * that: all but 0.2 to 0.4% of it on a Pi 5's four cores, 0.6 to 0.7% on
 * a Pi 4's and 0.8% on a Pi 3's (backend/avx2/CLIMB.md, 2026-10-28). */
#ifdef PK_ANE
static int ane_chunks(pk_asr *a, const pk_audio *x, const pk_seg *c, size_t n, size_t most,
                      int depth, pk_each_fn each, void *arg);

/* The Neural Engine's calls a run keeps in flight: two. AliceBench v2,
 * one process, best of three (2 of 2 on the M4 Pro), x real time:
 *   M5 Ultra (two Neural Engines): 1 380, 2 656, 3 642, 4 631
 *   M4 Pro (one):                  1 341, 2 359, 3 359
 * the transcript the same at each (tools/ane/CLIMB.md, 2026-11-02). Two
 * keep both of the Ultra's busy, a chunk every 31 ms; one Neural Engine
 * gains 5% from the CPU's side of a call hidden behind the other's.
 * PKTITO_ANE_INFLIGHT (1 to 4, 1 the one-at-a-time path) is for measuring
 * a chip again. */
static int ane_depth(void) {
    const char *v = getenv("PKTITO_ANE_INFLIGHT");
    const int d = v ? atoi(v) : 2;
    return d < 1 ? 1 : d > 4 ? 4 : d;
}
#endif

int pk_run_chunks_with(pk_asr *a, const pk_audio *x, const pk_seg *c, size_t n,
                       pk_each_fn each, void *arg) {
    size_t most = 0, longest = 0;   /* feature frames, and samples */
    for (size_t i = 0; i < n; i++) {
        const size_t len = (size_t)(c[i].end - c[i].start), T0 = pk_mel_frames(a->m, len);
        most = T0 > most ? T0 : most;
        longest = len > longest ? len : longest;
    }
    if (most > UINT32_MAX / 4 || reserve(a, most) || reserve_pcm(a, longest))
        return -1;
    if (n == 0)
        return 0;
#ifdef PK_ANE
    if (a->ane && n > 1 && ane_depth() > 1)
        return ane_chunks(a, x, c, n, most, ane_depth(), each, arg);
#endif
    /* one lane with the Neural Engine or the NPU: a lane's transcriber has
     * the CPU's encoder, so a chunk's bits would be the lane's it fell to */
    int k = a->ane || a->qnn ? 1 : lanes_for(a, c, n);
    if (k > 1 && (k = lanes_ready(a, k, most, longest)) > 1)
        return run_lanes(a, x, c, n, k, each, arg);
    pk_os_throttle was = pk_os_unthrottle();  /* this thread runs a share of every job */
    long Te = encode_at(a, x, &c[0], a->fproj[0]);
    int rc = Te < 0 ? -1 : 0;
    for (size_t i = 0; i < n && rc == 0; i++) {
        const float *f = a->fproj[i % 2];
        long nt;
        if (i + 1 < n) {
            search_job s = { a, (uint32_t)Te, f, 0 };
            pk_pool_aside(a->pool, search_aside, &s);
            Te = encode_at(a, x, &c[i + 1], a->fproj[(i + 1) % 2]);
            PROF(PK_P_DEC, pk_pool_join(a->pool));
            nt = s.n;
        } else {
            PROF(PK_P_DEC, nt = pk_decode(a, (uint32_t)Te, f, 1));
        }
        if (nt < 0)
            rc = -1;
        else
            rc = each(arg, i, a->toks, nt, a->fproj[i % 2]);
        if (rc == 0 && Te < 0)          /* the next chunk's encoder could not run */
            rc = -1;
    }
    pk_os_rethrottle(was);
    return rc;
}

#ifdef PK_ANE
/* A chunk's call to the Neural Engine, in a slot of its own */
typedef struct {
    pk_asr *a;
    int slot, cpu;                   /* cpu: longer than the Core ML model takes */
    uint32_t T0;
    long Te;
    dispatch_semaphore_t done;
} ane_call;

static void ane_call_run(void *arg) {
    ane_call *k = arg;
    const uint32_t Te = pk_ane_encode_in(k->a->ane, k->slot, k->a->ane_feat[k->slot], k->T0,
                                         k->a->ane_x[k->slot]);
    k->Te = Te ? (long)Te : -1;
    dispatch_semaphore_signal(k->done);
}

/* chunk j's features on this thread, and its call sent off; Te -1 where
 * its samples cannot be read */
static void ane_send(pk_asr *a, const pk_audio *x, const pk_seg *c, ane_call *k) {
    const size_t n = (size_t)(c->end - c->start);
    k->T0 = (uint32_t)pk_mel_frames(a->m, n);
    k->cpu = k->T0 > pk_ane_frames(a->ane);
    const int unread = !k->cpu && k->T0 && (n > a->pcm_cap || pk_audio_read(x, c->start, a->pcm, n));
    if (k->T0 == 0 || k->cpu || unread) {
        k->Te = unread ? -1 : 0;
        dispatch_semaphore_signal(k->done);
        return;
    }
    PROF(PK_P_MEL, pk_mel(a, a->pcm, n));
    memcpy(a->ane_feat[k->slot], a->feat, (size_t)k->T0 * a->m->n_mels * sizeof(float));
    dispatch_async_f(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), k, ane_call_run);
}

/* chunk j's call waited for, and its encoder frames through the joint's
 * encoder half into f, on this thread, in order; its frames, -1 if not */
static long ane_take(pk_asr *a, const pk_audio *x, const pk_seg *c, ane_call *k, float *f) {
    dispatch_semaphore_wait(k->done, DISPATCH_TIME_FOREVER);
    if (k->cpu)
        return encode_at(a, x, c, f);
    if (k->Te > 0) {
        memcpy(a->x, a->ane_x[k->slot], (size_t)k->Te * a->m->d_model * sizeof(float));
        PROF(PK_P_DEC, pk_joint_frames(a, (uint32_t)k->Te, f));
    }
    return k->Te;
}

/* pk_run_chunks_with with depth of the Neural Engine's calls in flight: a
 * chip of two Neural Engines (the M5 Ultra's H11ANE and H11ANE1) runs two
 * chunks' at once, where one call at a time left the second idle; AliceBench
 * v2 on it as separate processes, 1 380x, 2 756x, 3 842x, 4 868x
 * (libpktito, 2026-11-02). Each chunk's features are made here and its
 * call sent off; the joint and the search take the chunks in order, as the
 * one-at-a-time path does, the search beside the next chunk's joint, so the
 * tokens are that path's. */
static int ane_chunks(pk_asr *a, const pk_audio *x, const pk_seg *c, size_t n, size_t most,
                      int depth, pk_each_fn each, void *arg) {
    const pk_model *m = a->m;
    /* each slot's buffers, for the longest chunk yet: kept on pk_asr */
    if (a->ane_cap < most) {
        for (int k = 0; k < 4; k++) {
            free(a->ane_feat[k]);
            free(a->ane_x[k]);
            a->ane_feat[k] = a->ane_x[k] = NULL;
        }
        a->ane_cap = most;
    }
    size_t Tc = a->ane_cap;
    for (int i = 0; i < 3; i++)
        Tc = (Tc + 1) / 2;
    for (int k = 0; k < depth; k++) {
        if (!a->ane_feat[k])
            a->ane_feat[k] = malloc(a->ane_cap * m->n_mels * sizeof(float));
        if (!a->ane_x[k])
            a->ane_x[k] = malloc(Tc * m->d_model * sizeof(float));
        if (!a->ane_feat[k] || !a->ane_x[k])
            return -1;
    }
    ane_call call[4];
    for (int k = 0; k < depth; k++)
        call[k] = (ane_call){ a, k, 0, 0, 0, dispatch_semaphore_create(0) };
    pk_os_throttle was = pk_os_unthrottle();
    size_t sent = 0;
    for (; sent < n && sent < (size_t)depth; sent++)
        ane_send(a, x, &c[sent], &call[sent % depth]);
    long Te = ane_take(a, x, &c[0], &call[0], a->fproj[0]);
    int rc = Te < 0 ? -1 : 0;
    size_t i = 0;
    for (; i < n && rc == 0; i++) {
        const float *f = a->fproj[i % 2];
        long nt;
        if (i + 1 < n) {
            search_job s = { a, (uint32_t)Te, f, 0 };
            pk_pool_aside(a->pool, search_aside, &s);
            /* chunk i's slot is free: chunk i + depth's call goes out */
            if (sent < n) {
                ane_send(a, x, &c[sent], &call[sent % depth]);
                sent++;
            }
            Te = ane_take(a, x, &c[i + 1], &call[(i + 1) % depth], a->fproj[(i + 1) % 2]);
            PROF(PK_P_DEC, pk_pool_join(a->pool));
            nt = s.n;
        } else {
            PROF(PK_P_DEC, nt = pk_decode(a, (uint32_t)Te, f, 1));
        }
        if (nt < 0)
            rc = -1;
        else
            rc = each(arg, i, a->toks, nt, a->fproj[i % 2]);
        if (rc == 0 && Te < 0)
            rc = -1;
    }
    /* the calls still out, which hold their slots' buffers, waited for */
    for (size_t j = i + 1; j < sent; j++)
        dispatch_semaphore_wait(call[j % depth].done, DISPATCH_TIME_FOREVER);
    for (int k = 0; k < depth; k++)
        dispatch_release(call[k].done);
    pk_os_rethrottle(was);
    return rc;
}
#endif

/* pk_asr_run_chunks' fn as an each: each chunk with its place and the
 * run's progress */
typedef struct {
    pk_chunk_fn fn;
    void *arg;
    const pk_model *m;
    const pk_seg *c;
    size_t n;
    int64_t done, total;
} plain_each;

static int plain(void *arg, size_t i, const pk_tok *t, long nt, float *spare) {
    (void)spare;
    plain_each *p = arg;
    p->done += p->c[i].end - p->c[i].start;
    const pk_chunk_out out = { i, p->n, p->c[i], t, nt, 0, p->done, p->total,
                               pk_frame_of(p->m, p->c[i].start) };
    return p->fn(p->arg, &out) ? 1 : 0;
}

int pk_asr_run_chunks(pk_asr *a, const pk_audio *x, const pk_seg *c, size_t n, pk_chunk_fn fn,
                      void *arg) {
    plain_each p = { fn, arg, a->m, c, n, 0, 0 };
    for (size_t i = 0; i < n; i++)
        p.total += c[i].end - c[i].start;
    pk_turns t;
    pk_audio y;
    if (pk_audio_turns(&t, x, &y))
        return -1;
    const int rc = pk_run_chunks_with(a, &y, c, n, plain, &p);
    pk_audio_turns_free(&t);
    return rc;
}

int32_t pk_frame_of(const pk_model *m, int64_t sample) {
    const int64_t spf = llround(pk_model_frame_seconds(m) * pk_model_rate(m));
    return (int32_t)((sample + spf / 2) / spf);
}

long pk_retry_run(pk_asr *a, const float *x, size_t n, float *spare) {
    const long Te = encode(a, x, n, spare);
    return Te < 0 ? -1 : pk_decode(a, (uint32_t)Te, spare, 1);
}

const char *pk_backend(void) { return pk_k_name(); }

const char *pk_version(void) { return PKTITO_VERSION; }

/* The compiler's own words for itself; Apple's clang names a version of its
 * own, unlike the LLVM one it comes from. */
const char *pk_compiler(void) {
#if defined(__apple_build_version__)
    return "Apple clang " __clang_version__;
#elif defined(__clang__)
    return "clang " __clang_version__;
#elif defined(__GNUC__)
    return "gcc " __VERSION__;
#else
    return "an unknown compiler";
#endif
}

#ifdef PK_PROFILE
double pk_prof[PK_NPROF];
double pk_prof_now(void) { return pk_os_now(); }
#endif

/* ------------------------------------------------------------- recording --- */

static int floats_read(void *arg, int64_t at, float *out, size_t count) {
    memcpy(out, (const float *)arg + at, count * sizeof *out);
    return 0;
}

pk_audio pk_audio_floats(const float *x, size_t n) {
    return (pk_audio){ (int64_t)n, floats_read, (void *)x };
}

int pk_audio_read(const pk_audio *x, int64_t at, float *out, size_t n) {
    if (at < 0 || at > x->n || (uint64_t)(x->n - at) < n)
        return -1;
    return n == 0 || x->read(x->arg, at, out, n) == 0 ? 0 : -1;
}

static int turns_read(void *arg, int64_t at, float *out, size_t count) {
    pk_turns *t = arg;
    pk_lock_take(t->lock);
    const int rc = t->x.read(t->x.arg, at, out, count);
    pk_lock_give(t->lock);
    return rc;
}

int pk_audio_turns(pk_turns *t, const pk_audio *x, pk_audio *out) {
    t->x = *x;
    if (!(t->lock = pk_lock_new()))
        return -1;
    *out = (pk_audio){ x->n, turns_read, t };
    return 0;
}

void pk_audio_turns_free(pk_turns *t) { pk_lock_free(t->lock); }

/* ---------------------------------------------------------------- chunks --- */

/* Silence kept on the far side of a speech chunk's first and last segment,
 * at most: what lies further from the speech is a gap's. */
#define PK_CHUNK_PAD 16000

/* A gap between speech chunks is transcribed too, since Silero hears voices
 * rather than words: singing, and speech over music, are not speech to it.
 * But not in a piece shorter than PK_GAP_MIN, where the recognizer makes
 * words of anything: of 28 one-second pieces of instrumental music, v2
 * heard "Yeah." or the like in 27, and from 3 s up in none (README, "Long
 * recordings"). Nor in a silent piece, which holds nothing to hear. */
#define PK_GAP_MIN    48000      /* 3 s */
/* a 32 ms frame with a mean square under this is silent: 50 dB below full
 * scale, 18 dB under the quietest of 42 pieces of that music at its loudest */
#define PK_GAP_SILENT 1e-5

/* Samples of a gap read at a time, whole frames: 128 KB */
#define PK_GAP_READ (64 * PK_VAD_SAMPLES)

/* 1 if no 32 ms frame of x[at, at + n) is louder than PK_GAP_SILENT, else
 * 0, or -1 if a read fails; read PK_GAP_READ at a time into buf. A sample
 * that is not finite is read as 0, as pk_asr_run reads it. */
static int silent(const pk_audio *x, int64_t at, int64_t n, float *buf) {
    for (int64_t i = 0; i < n; i += PK_GAP_READ) {
        const int64_t got = n - i < PK_GAP_READ ? n - i : PK_GAP_READ;
        if (pk_audio_read(x, at + i, buf, (size_t)got))
            return -1;
        for (int64_t f = 0; f < got; f += PK_VAD_SAMPLES) {
            int64_t len = got - f < PK_VAD_SAMPLES ? got - f : PK_VAD_SAMPLES;
            double ss = 0;
            for (int64_t j = 0; j < len; j++) {
                double v = isfinite(buf[f + j]) ? buf[f + j] : 0;
                ss += v * v;
            }
            if (ss > PK_GAP_SILENT * (double)len)
                return 0;
        }
    }
    return 1;
}

/* The gap [a, b) as chunks from out[*k] on: the fewest pieces of at most
 * max_len, all of a length, but those shorter than PK_GAP_MIN or silent.
 * Silero cannot say where a gap's words are, so its cuts fall where they
 * fall. 0, or -1 if a read fails. */
static int put_gap(const pk_audio *x, int64_t a, int64_t b, int64_t max_len, pk_seg *out,
                   pk_seg *win, size_t *k, float *buf) {
    if (b <= a)
        return 0;
    int64_t m = (b - a + max_len - 1) / max_len;
    for (int64_t i = 0; i < m; i++) {
        int64_t pa = a + (b - a) * i / m, pb = a + (b - a) * (i + 1) / m;
        const int quiet = pb - pa >= PK_GAP_MIN ? silent(x, pa, pb - pa, buf) : 1;
        if (quiet < 0)
            return -1;
        if (!quiet) {
            out[*k].start = pa;
            out[*k].end = pb;
            if (win)
                win[*k] = out[*k];
            (*k)++;
        }
    }
    return 0;
}

long pk_chunks(const pk_audio *x, const pk_seg *s, size_t n, int64_t max_len, pk_seg *out) {
    return pk_chunks_past(x, s, n, max_len, 0, 0, out, NULL);
}

long pk_chunks_past(const pk_audio *x, const pk_seg *s, size_t n, int64_t max_len, int64_t ctx,
                    int64_t pause, pk_seg *out, pk_seg *win) {
    const int64_t total = x->n;
    if (total <= max_len && n) {
        out[0].start = 0;
        out[0].end = total;
        if (win)
            win[0] = out[0];
        return 1;
    }
    float *buf = malloc(PK_GAP_READ * sizeof *buf);
    if (!buf)
        return -1;
    size_t k = 0;
    int64_t done = 0;                           /* the end of the last speech chunk */
    for (size_t i = 0; i < n;) {
        /* the chunk opens half way into the silence before segment i */
        int64_t gap0 = i ? s[i].start - s[i - 1].end : s[i].start;
        int64_t lead = s[i].start - (gap0 / 2 < PK_CHUNK_PAD ? gap0 / 2 : PK_CHUNK_PAD);
        /* and ends with the last segment that fits, with what it hears
         * past that one's cut: ctx where the silence after it is short */
        size_t last = i;
        int64_t trail = 0, past = 0;
        for (size_t j = i; j < n; j++) {
            int64_t gap = j + 1 < n ? s[j + 1].start - s[j].end : total - s[j].end;
            int64_t t = s[j].end + (gap / 2 < PK_CHUNK_PAD ? gap / 2 : PK_CHUNK_PAD);
            if (j > i && t - lead > max_len)
                break;
            int64_t pj = j + 1 < n && gap < pause ? (total - t < ctx ? total - t : ctx) : 0;
            if (j == i || t - lead + pj <= max_len) {
                last = j;
                trail = t;
                past = pj;
            }
        }
        if (trail - lead > max_len) {           /* one segment, too long with its silence */
            lead = s[i].end - max_len > lead ? s[i].end - max_len : lead;
            trail = lead + max_len < trail ? lead + max_len : trail;
        }
        if (trail - lead + past > max_len)      /* no room for it */
            past = 0;
        if (put_gap(x, done, lead, max_len, out, win, &k, buf)) {
            free(buf);
            return -1;
        }
        out[k].start = lead;
        out[k].end = trail;
        if (win) {
            win[k] = out[k];
            win[k].end += past;
        }
        k++;
        done = trail;
        i = last + 1;
    }
    const int rc = put_gap(x, done, total, max_len, out, win, &k, buf);
    free(buf);
    return rc ? -1 : (long)k;
}
