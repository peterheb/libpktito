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

void pk_asr_free(pk_asr *a) {
    if (!a)
        return;
    pk_pool_free(a->drive);
    for (int j = 1; a->lane && j < a->lanes; j++)
        pk_asr_free(a->lane[j]);
    free(a->lane);
    pk_vad_free(a->vad);
    pk_pool_free(a->pool);
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
    free(a);
}

/* A chunk up to its search: its features, the encoder, and the joint's
 * encoder half of its frames, into f. Returns its encoder frames. */
static uint32_t encode(pk_asr *a, const float *x, size_t n, float *f) {
    const size_t T0 = pk_mel_frames(a->m, n);
    if (T0 == 0)
        return 0;
    PROF(PK_P_MEL, pk_mel(a, x, n));
    uint32_t Te = pk_encode(a, (uint32_t)T0);
    PROF(PK_P_DEC, pk_joint_frames(a, Te, f));
    return Te;
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
    uint32_t Te = encode(a, x, n, a->fproj[0]);
    long nt;
    PROF(PK_P_DEC, nt = pk_decode(a, Te, a->fproj[0], 1));
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
    const float *x;
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
        const uint32_t Te = encode(l, r->x + r->c[i].start, (size_t)(r->c[i].end - r->c[i].start),
                                   l->fproj[0]);
        const long nt = pk_decode(l, Te, l->fproj[0], 1);
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
 * `most` feature frames: lane j > 0 is made the first time a run needs
 * it, with the most threads a run gives it, and kept. Fewer where memory
 * or a thread cannot be had, down to one: the loop below. */
static int lanes_ready(pk_asr *a, int k, size_t most) {
    if (!a->drive && !(a->drive = pk_pool_new(a->lanes)))
        return 1;
    for (int j = 1; j < k; j++) {
        if (!a->lane[j]) {
            int w = 1;
            for (int q = j + 1; q <= a->lanes; q++)
                w = share(a->nth, j, q) > w ? share(a->nth, j, q) : w;
            a->lane[j] = transcriber(a->m, w);
        }
        if (!a->lane[j] || reserve(a->lane[j], most))
            return j;
    }
    return k;
}

/* pk_run_chunks_with's chunks on k lanes at once, lane j on its share of
 * the transcriber's threads. The encoder's race between every core and
 * the fast kind waits for a run of one lane, whose timing the other lanes
 * would muddle. */
static int run_lanes(pk_asr *a, const float *x, const pk_seg *c, size_t n, int k, pk_each_fn each,
                     void *arg) {
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
int pk_run_chunks_with(pk_asr *a, const float *x, const pk_seg *c, size_t n,
                       pk_each_fn each, void *arg) {
    size_t most = 0;
    for (size_t i = 0; i < n; i++) {
        const size_t T0 = pk_mel_frames(a->m, (size_t)(c[i].end - c[i].start));
        most = T0 > most ? T0 : most;
    }
    if (most > UINT32_MAX / 4 || reserve(a, most))
        return -1;
    if (n == 0)
        return 0;
    int k = lanes_for(a, c, n);
    if (k > 1 && (k = lanes_ready(a, k, most)) > 1)
        return run_lanes(a, x, c, n, k, each, arg);
    pk_os_throttle was = pk_os_unthrottle();  /* this thread runs a share of every job */
    uint32_t Te = encode(a, x + c[0].start, (size_t)(c[0].end - c[0].start), a->fproj[0]);
    int rc = 0;
    for (size_t i = 0; i < n && rc == 0; i++) {
        const float *f = a->fproj[i % 2];
        long nt;
        if (i + 1 < n) {
            search_job s = { a, Te, f, 0 };
            pk_pool_aside(a->pool, search_aside, &s);
            Te = encode(a, x + c[i + 1].start, (size_t)(c[i + 1].end - c[i + 1].start),
                        a->fproj[(i + 1) % 2]);
            PROF(PK_P_DEC, pk_pool_join(a->pool));
            nt = s.n;
        } else {
            PROF(PK_P_DEC, nt = pk_decode(a, Te, f, 1));
        }
        if (nt < 0)
            rc = -1;
        else
            rc = each(arg, i, a->toks, nt, a->fproj[i % 2]);
    }
    pk_os_rethrottle(was);
    return rc;
}

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

int pk_asr_run_chunks(pk_asr *a, const float *x, const pk_seg *c, size_t n, pk_chunk_fn fn,
                      void *arg) {
    plain_each p = { fn, arg, a->m, c, n, 0, 0 };
    for (size_t i = 0; i < n; i++)
        p.total += c[i].end - c[i].start;
    return pk_run_chunks_with(a, x, c, n, plain, &p);
}

int32_t pk_frame_of(const pk_model *m, int64_t sample) {
    const int64_t spf = llround(pk_model_frame_seconds(m) * pk_model_rate(m));
    return (int32_t)((sample + spf / 2) / spf);
}

long pk_retry_run(pk_asr *a, const float *x, size_t n, float *spare) {
    const uint32_t Te = encode(a, x, n, spare);
    return pk_decode(a, Te, spare, 1);
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

/* No 32 ms frame of x[0, n) louder than PK_GAP_SILENT. A sample that is not
 * finite is read as 0, as pk_asr_run reads it. */
static int silent(const float *x, int64_t n) {
    for (int64_t i = 0; i < n; i += PK_VAD_SAMPLES) {
        int64_t len = n - i < PK_VAD_SAMPLES ? n - i : PK_VAD_SAMPLES;
        double ss = 0;
        for (int64_t j = 0; j < len; j++) {
            double v = isfinite(x[i + j]) ? x[i + j] : 0;
            ss += v * v;
        }
        if (ss > PK_GAP_SILENT * (double)len)
            return 0;
    }
    return 1;
}

/* The gap [a, b) as chunks from out[*k] on: the fewest pieces of at most
 * max_len, all of a length, but those shorter than PK_GAP_MIN or silent.
 * Silero cannot say where a gap's words are, so its cuts fall where they
 * fall. */
static void put_gap(const float *x, int64_t a, int64_t b, int64_t max_len, pk_seg *out, pk_seg *win,
                    size_t *k) {
    if (b <= a)
        return;
    int64_t m = (b - a + max_len - 1) / max_len;
    for (int64_t i = 0; i < m; i++) {
        int64_t pa = a + (b - a) * i / m, pb = a + (b - a) * (i + 1) / m;
        if (pb - pa >= PK_GAP_MIN && !silent(x + pa, pb - pa)) {
            out[*k].start = pa;
            out[*k].end = pb;
            if (win)
                win[*k] = out[*k];
            (*k)++;
        }
    }
}

size_t pk_chunks(const float *x, const pk_seg *s, size_t n, int64_t total, int64_t max_len, pk_seg *out) {
    return pk_chunks_past(x, s, n, total, max_len, 0, 0, out, NULL);
}

size_t pk_chunks_past(const float *x, const pk_seg *s, size_t n, int64_t total, int64_t max_len,
                      int64_t ctx, int64_t pause, pk_seg *out, pk_seg *win) {
    if (total <= max_len && n) {
        out[0].start = 0;
        out[0].end = total;
        if (win)
            win[0] = out[0];
        return 1;
    }
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
        put_gap(x, done, lead, max_len, out, win, &k);
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
    put_gap(x, done, total, max_len, out, win, &k);
    return k;
}
