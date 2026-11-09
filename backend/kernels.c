/* backend/kernels.h's functions, each a call through the table of the
 * backend in use, and the choice of that backend.
 *
 * The Makefile compiles every backend of the machine's family that its
 * compiler can make, each with its own instruction set, and tells this file
 * which it compiled (PK_HAVE_<id>). Which runs is a question for the CPU,
 * asked once: the first whose needs it meets, fastest first, so a binary
 * built on an Alder Lake runs on a Zen 4 (AVX512-VNNI's encoding, not
 * AVX-VNNI's) and one built on a Pi 5 runs on a Pi 3 (no dot product).
 * Where that one competes (kernels.h), it and the others after it are
 * timed in the int8 product on the fastest cores, and the fastest runs.
 * They all compute ref's bits, so the choice moves a transcript's speed
 * and nothing else.
 */
#include "kernels.h"
#include "os.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const pk_kernels *const pk_kernels_built[] = {
/* neon-sve first: on the GB10 15 to 16% faster than neon-dp at its ten
 * X925s, 24% on one (backend/neon-dp/CLIMB.md, 2026-11-02) */
#ifdef PK_HAVE_neon_sve
    &pk_kernels_neon_sve,
#endif
/* neon-i8mm after it, the same loop for the cores with I8MM and no SVE,
 * where it measures the faster (backend/neon-dp/CLIMB.md, 2026-11-07) */
#ifdef PK_HAVE_neon_i8mm
    &pk_kernels_neon_i8mm,
#endif
/* neon-dp-a76 after it: neon-dp with the A76's tile, where it measures
 * the faster (backend/neon-dp/CLIMB.md, 2026-11-08) */
#ifdef PK_HAVE_neon_dp_a76
    &pk_kernels_neon_dp_a76,
#endif
#ifdef PK_HAVE_neon_dp
    &pk_kernels_neon_dp,
#endif
#ifdef PK_HAVE_neon
    &pk_kernels_neon,
#endif
/* amx-int8 first, 24-25% over avx512-vnni on Sapphire and Granite Rapids;
 * avx512-vnni before avx2-vnni: every CPU that has both (Zen 5, Sapphire
 * Rapids, Granite Rapids) runs its zmm product faster (backend/avx2/CLIMB.md,
 * 2026-10-24) */
#ifdef PK_HAVE_amx_int8
    &pk_kernels_amx_int8,
#endif
#ifdef PK_HAVE_avx512_vnni
    &pk_kernels_avx512_vnni,
#endif
#ifdef PK_HAVE_avx2_vnni
    &pk_kernels_avx2_vnni,
#endif
#ifdef PK_HAVE_avx2
    &pk_kernels_avx2,
#endif
    &pk_kernels_ref,
    NULL
};

/* The choice, made once and kept: by the first pk_model_load, on its
 * thread, before a transcriber's threads run a kernel (core/model.c), or by
 * the first kernel called in a program that loads no model. Its probes run
 * once a process, where a pktito-c run on the M4 mini had run neon-i8mm's
 * eleven times, ten at once on its first job's threads
 * (backend/neon-dp/CLIMB.md, 2026-11-07); a thread that asks while another
 * makes it waits for that one's. A PKTITO_KERNELS that cannot be had is
 * reported by pk_model_load; a kernel called without a model loaded first
 * takes the CPU's choice instead. */
static _Atomic(const pk_kernels *) g_in_use;
static atomic_int g_making;                 /* 1 while a thread makes it */

static const pk_kernels *keep(const pk_kernels *t) {
    const pk_kernels *kept = NULL;
    if (atomic_compare_exchange_strong_explicit(&g_in_use, &kept, t, memory_order_acq_rel,
                                                memory_order_acquire))
        return t;
    return kept;
}

/* What the probe timed, for pk_k_picks: the pick, the base it was timed
 * against, each competitor's time over the base's, on how many cores at
 * once, and how long the timing took with its threads' start and end. */
static struct {
    const pk_kernels *k, *base;
    int n, cores, probed;
    const pk_kernels *timed[8];
    double ratio[8], seconds;
} g_pick;

/* ---------------------------------------------------------- the probe --- */

/* A backend that competes (kernels.h) runs only where it measures faster:
 * SMMLA's neon-i8mm is the faster on the S24's X4 and A720 and not on its
 * A520, the A76's tile on a Pi 5's A76 and not on the S24's X4. Each is
 * timed in the product it would run: its gemm_q8 on a slice as
 * core/encoder.c's pk_gemm makes them, 64 rows of 1024 inputs in blocks of
 * 32, 64 KB of weights, past an A55's or an A520's L1 as the model's are,
 * and 128 columns, which the tiles stream as they do a chunk's 250 or so.
 * It is timed on every one of the fastest cores at once, as a
 * transcription runs them, sharing the memory system: on armlet's two
 * A76s the A76's tile reads 0.910 to 0.914 of SDOT's time so, against
 * 0.975 on one alone. Each kind of core timed for its own pick bought
 * nothing end to end, on armlet or the S24 (backend/neon-dp/CLIMB.md,
 * 2026-11-08), so the fastest cores' pick runs on every core. */
enum { PROBE_ROWS = 64, PROBE_N = 1024, PROBE_T = 128, PROBE_QB = 32 };

/* The probe: its threads (pk_os_on_fast_cores), one on each core, the
 * activations they share as pk_gemm's slices share a chunk's, a barrier
 * that starts each sample on every one of them at once, and each thread's
 * ratios. */
typedef struct {
    const pk_kernels *const *cand;          /* the competitors, then the base */
    int n;
    int8_t *q;
    float *d;
    atomic_int arrived, phase, failed;
    int threads, reps;                      /* thread 0's, read after the join */
    double r[PK_OS_ON_FAST][8];
} probe;

typedef struct {
    int8_t *w;
    float *ws, *out;
    pk_act a;
    probe *pr;
    int n, phase;                           /* the probe's threads, the barrier's last */
} probe_data;

/* every one of the probe's threads here before any goes on: a spin, on
 * cores they hold one each, for a sample's length at most */
static void probe_wait(probe_data *p) {
    probe *pr = p->pr;
    const int phase = p->phase++;
    if (atomic_fetch_add(&pr->arrived, 1) == p->n - 1) {
        atomic_store(&pr->arrived, 0);
        atomic_store(&pr->phase, phase + 1);
    } else
        while (atomic_load(&pr->phase) == phase)
            ;
}

static double sample(const pk_kernels *t, probe_data *p, int reps) {
    probe_wait(p);
    const double t0 = pk_os_now();
    for (int i = 0; i < reps; i++)
        t->gemm_q8(p->out, PROBE_ROWS, p->w, p->ws, NULL, PROBE_ROWS, &p->a, 0);
    return pk_os_now() - t0;
}

/* c's time over b's: b's, c's, c's, b's a run, so that a clock rising or
 * falling through the four slows both alike, and the median of 5 runs,
 * past one that a migration or an interrupt cut into. Timed b then c, the
 * best of 5 of each, neon-i8mm's tile read 8% low on a cold M5 core whose
 * clock moved: 1.08 for its 1.17 (CLIMB.md, 2026-11-07). */
static double ratio(const pk_kernels *c, const pk_kernels *b, probe_data *p, int reps) {
    double r[5];
    for (int run = 0; run < 5; run++) {
        const double b1 = sample(b, p, reps);
        const double c1 = sample(c, p, reps);
        const double c2 = sample(c, p, reps);
        const double b2 = sample(b, p, reps);
        r[run] = (c1 + c2) / (b1 + b2);
    }
    for (int i = 1; i < 5; i++) /* sorted, for the median */
        for (int j = i; j > 0 && r[j] < r[j - 1]; j--) {
            const double t = r[j];
            r[j] = r[j - 1];
            r[j - 1] = t;
        }
    return r[2];
}

/* Thread i of n: its own weights and out, written, not left to the
 * system's zero pages, which Linux maps as one page and a probe reads from
 * L1 (CLIMB.md, 2026-11-07); each competitor timed against the base in
 * samples of 50 us or more, as many calls on every thread as thread 0's
 * take. */
static int probe_core(void *arg, int i, int n) {
    probe *pr = arg;
    const pk_kernels *base = pr->cand[pr->n - 1];
    probe_data p = { .pr = pr, .n = n };
    p.w = malloc((size_t)PROBE_ROWS * PROBE_N);
    p.ws = malloc(PROBE_ROWS * sizeof(float));
    p.out = malloc((size_t)PROBE_T * PROBE_ROWS * sizeof(float));
    p.a = (pk_act){ pr->q, pr->d, PROBE_N, PROBE_QB, PROBE_T };
    if (p.w && p.ws && p.out) {
        uint32_t s = 0x9e3779b9u + (uint32_t)i;
        for (size_t x = 0; x < (size_t)PROBE_ROWS * PROBE_N; x++)
            s = s * 1664525u + 1013904223u, p.w[x] = (int8_t)((s >> 24) % 255 - 127);
        for (size_t x = 0; x < PROBE_ROWS; x++)
            p.ws[x] = 1.0f / 64;
    } else
        atomic_store(&pr->failed, 1);
    probe_wait(&p);
    const int ok = !atomic_load(&pr->failed);
    if (ok) {
        for (int c = 0; c < pr->n; c++)     /* warm */
            sample(pr->cand[c], &p, 1);
        const double one = sample(base, &p, 1);
        if (i == 0)
            pr->reps = one > 0 && one < 50e-6 ? (int)(50e-6 / one) + 1 : 1, pr->threads = n;
        probe_wait(&p);
        const int reps = pr->reps;
        for (int c = 0; c < pr->n - 1 && c < 8; c++)
            pr->r[i][c] = ratio(pr->cand[c], base, &p, reps);
    }
    free(p.w), free(p.ws), free(p.out);
    return ok;
}

/* A backend is taken where it is at least 3% the faster: the A76's tile
 * reads 0.985 to 0.995 of SDOT's on the M4, level with neon-dp end to end,
 * and by its median alone the M4 took neon-dp-a76 in two runs of three and
 * neon-dp in the third, the same speed under a name that moved
 * (CLIMB.md, 2026-11-08). */
#define PK_PROBE_FASTER 0.97

/* The pick without PKTITO_KERNELS: the first backend whose needs the CPU
 * meets, and where that one competes, the fastest of it and the
 * competitors after it, by the median of the threads' ratios, where it is
 * 3% the faster than the first that does not; that one otherwise. */
static const pk_kernels *choose(void) {
    const pk_kernels *cand[8];
    int n = 0;
    for (const pk_kernels *const *k = pk_kernels_built; *k && n < 8; k++) {
        if (pk_os_cpu_lacks((*k)->needs))
            continue;
        cand[n++] = *k;
        if (!(*k)->competes)
            break;
    }
    const pk_kernels *base = cand[n - 1], *fastest = base;
    g_pick.k = g_pick.base = base;
    if (n == 1)
        return base;
    const size_t nb = PROBE_N / PROBE_QB;
    probe pr = { .cand = cand, .n = n };
    pr.q = malloc((size_t)PROBE_T * PROBE_N);
    pr.d = malloc(PROBE_T * nb * sizeof(float));
    const double t0 = pk_os_now();
    if (pr.q && pr.d) {
        uint32_t s = 0x7f4a7c15u;
        for (size_t i = 0; i < (size_t)PROBE_T * PROBE_N; i++)
            s = s * 1664525u + 1013904223u, pr.q[i] = (int8_t)((s >> 24) % 255 - 127);
        for (size_t i = 0; i < PROBE_T * nb; i++)
            pr.d[i] = 1.0f / 127;
        pk_os_on_fast_cores(probe_core, &pr);
    }
    g_pick.seconds = pk_os_now() - t0;
    free(pr.q), free(pr.d);
    const int t = pr.threads;
    double best = 1.0;
    g_pick.cores = t, g_pick.probed = 1;
    for (int c = 0; c < n - 1 && c < 8 && t > 0; c++) {
        double v[PK_OS_ON_FAST];
        for (int i = 0; i < t; i++) {       /* sorted, for the median */
            int j = i;
            for (; j > 0 && v[j - 1] > pr.r[i][c]; j--)
                v[j] = v[j - 1];
            v[j] = pr.r[i][c];
        }
        const double r = (v[(t - 1) / 2] + v[t / 2]) / 2;
        g_pick.timed[g_pick.n] = cand[c], g_pick.ratio[g_pick.n++] = r;
        if (r < best)
            best = r, fastest = cand[c];
    }
    if (best < PK_PROBE_FASTER)
        g_pick.k = fastest;
    return g_pick.k;
}

const pk_kernels *pk_kernels_pick(char *why, size_t whylen) {
    const char *want = getenv("PKTITO_KERNELS");
    const pk_kernels *kept = atomic_load_explicit(&g_in_use, memory_order_acquire);
    if (kept && !(want && *want))
        return kept;
    if (!(want && *want)) {
        int idle = 0;
        if (!atomic_compare_exchange_strong(&g_making, &idle, 1)) {
            while (!atomic_load_explicit(&g_in_use, memory_order_acquire))
                ;                                   /* another thread's probe, milliseconds */
            return atomic_load_explicit(&g_in_use, memory_order_acquire);
        }
        return keep(choose());
    }
    for (const pk_kernels *const *k = pk_kernels_built; *k; k++) {
        if (strcmp(want, (*k)->id))
            continue;
        const char *lack = pk_os_cpu_lacks((*k)->needs);
        if (!lack)
            return keep(*k);
        if (why && whylen)
            snprintf(why, whylen, "PKTITO_KERNELS=%s uses %s, which this CPU does not have",
                     want, lack);
        return NULL;
    }
    if (why && whylen)
        snprintf(why, whylen, "PKTITO_KERNELS=%s is not among this build's backends", want);
    return NULL;
}

static const pk_kernels *k(void) {
    const pk_kernels *t = atomic_load_explicit(&g_in_use, memory_order_acquire);
    if (t)
        return t;
    t = pk_kernels_pick(NULL, 0);
    /* a PKTITO_KERNELS that cannot be had: the CPU's choice; ref, last,
     * needs nothing, so the loop ends */
    for (const pk_kernels *const *b = pk_kernels_built; !t; b++)
        if (!pk_os_cpu_lacks((*b)->needs))
            t = *b;
    return keep(t);
}

int pk_k_picks(char *out, size_t len) {
    k();
    size_t at = 0;
    out[0] = 0;
    if (!g_pick.probed)
        return 0;
    for (int i = 0; i < g_pick.n && at < len; i++)
        at += (size_t)snprintf(out + at, len - at, "%s%s %.3f", i ? ", " : "",
                               g_pick.timed[i]->id, g_pick.ratio[i]);
    if (at < len && g_pick.n)
        at += (size_t)snprintf(out + at, len - at, " of %s's time", g_pick.base->id);
    if (at < len && g_pick.cores > 1)
        at += (size_t)snprintf(out + at, len - at, " on %d cores at once", g_pick.cores);
    if (at < len)
        snprintf(out + at, len - at, "%s, timed in %.1f ms", g_pick.n ? "" : "not timed",
                 g_pick.seconds * 1e3);
    return 1;
}

const char *pk_k_name(void) { return k()->name; }

void pk_k_quantize(pk_act *a, const float *x, size_t xs) { k()->quantize(a, x, xs); }

void pk_k_gemm_q8(float *out, size_t os, const int8_t *w, const float *ws, const float *bias,
                  uint32_t rows, const pk_act *a, int add) {
    k()->gemm_q8(out, os, w, ws, bias, rows, a, add);
}

void pk_k_gemm_f32(float *out, size_t os, const float *w, uint32_t ld, uint32_t rows, uint32_t n,
                   const float *x, size_t xs, uint32_t T, const float *bias, size_t bs, int relu) {
    k()->gemm_f32(out, os, w, ld, rows, n, x, xs, T, bias, bs, relu);
}

void pk_k_layernorm(float *out, size_t os, const float *x, size_t xs, uint32_t T, uint32_t n,
                    const float *w, const float *b, float eps) {
    k()->layernorm(out, os, x, xs, T, n, w, b, eps);
}

void pk_k_softmax(float *x, uint32_t n, float scale) { k()->softmax(x, n, scale); }

void pk_k_silu(float *x, size_t n) { k()->silu(x, n); }

void pk_k_gelu(float *x, size_t n) { k()->gelu(x, n); }

void pk_k_glu(float *out, size_t os, const float *x, size_t xs, uint32_t T, uint32_t n) {
    k()->glu(out, os, x, xs, T, n);
}

void pk_k_dwconv(float *out, size_t os, const float *x, size_t xs, uint32_t T, uint32_t C,
                 const float *w, const float *b, uint32_t K) {
    k()->dwconv(out, os, x, xs, T, C, w, b, K);
}

void pk_k_conv3x3s2(float *out, const float *in, uint32_t r0, uint32_t Ti, uint32_t Fi,
                    uint32_t C, int depthwise, const float *w, const float *b, int relu,
                    uint32_t t0, uint32_t t1) {
    k()->conv3x3s2(out, in, r0, Ti, Fi, C, depthwise, w, b, relu, t0, t1);
}

void pk_k_lstm(float *h, float *c, const float *g, uint32_t H) { k()->lstm(h, c, g, H); }

void pk_k_stft(float *out, size_t os, const float *const *x, uint32_t count, const pk_fft *f,
               float *scratch) {
    k()->stft(out, os, x, count, f, scratch);
}

void pk_k_resample(float *y, const float *x, uint32_t nb, uint32_t orig, uint32_t nw, const float *k_,
                   const size_t *at, const uint32_t *first, const uint32_t *len) {
    k()->resample(y, x, nb, orig, nw, k_, at, first, len);
}
