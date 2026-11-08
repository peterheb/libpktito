/* backend/kernels.h's functions, each a call through the table of the
 * backend in use, and the choice of that backend.
 *
 * The Makefile compiles every backend of the machine's family that its
 * compiler can make, each with its own instruction set, and tells this file
 * which it compiled (PK_HAVE_<id>). Which runs is a question for the CPU,
 * asked once: the first whose needs it meets, fastest first, so a binary
 * built on an Alder Lake runs on a Zen 4 (AVX512-VNNI's encoding, not
 * AVX-VNNI's) and one built on a Pi 5 runs on a Pi 3 (no dot product). They
 * all compute ref's bits, so the choice moves a transcript's speed and
 * nothing else.
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
 * where its probe finds its tile the faster (backend/neon-dp/CLIMB.md,
 * 2026-11-07) */
#ifdef PK_HAVE_neon_i8mm
    &pk_kernels_neon_i8mm,
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
 * the first kernel called in a program that loads no model. A probe then
 * runs once a process, where a pktito-c run on the M4 mini had run
 * neon-i8mm's eleven times, ten at once on its first job's threads
 * (backend/neon-dp/CLIMB.md, 2026-11-07). Two threads that make it at once
 * keep the first made. A PKTITO_KERNELS that cannot be had is reported by
 * pk_model_load; a kernel called without a model loaded first takes the
 * CPU's choice instead. */
static _Atomic(const pk_kernels *) g_in_use;

static const pk_kernels *keep(const pk_kernels *t) {
    const pk_kernels *kept = NULL;
    if (atomic_compare_exchange_strong_explicit(&g_in_use, &kept, t, memory_order_acq_rel,
                                                memory_order_acquire))
        return t;
    return kept;
}

const pk_kernels *pk_kernels_pick(char *why, size_t whylen) {
    const char *want = getenv("PKTITO_KERNELS");
    const pk_kernels *kept = atomic_load_explicit(&g_in_use, memory_order_acquire);
    if (kept && !(want && *want))
        return kept;
    for (const pk_kernels *const *k = pk_kernels_built; *k; k++) {
        if (want && *want && strcmp(want, (*k)->id))
            continue;
        const char *lack = pk_os_cpu_lacks((*k)->needs);
        /* a probe passes a backend by where it would be slower than the
         * next, unless PKTITO_KERNELS asks for it. It is asked on the
         * fastest kind of core the process may use, which does most of a
         * run's work: neon-i8mm's tile reads 1.16 of SDOT's time on the
         * S24's A520s and 0.80 to 0.84 on its A720s and X4, and its 8
         * threads run 12.5% faster on neon-i8mm (backend/neon-dp/CLIMB.md,
         * 2026-11-07). */
        if (!lack && ((want && *want) || !(*k)->usable || pk_os_on_fast_cores((*k)->usable)))
            return keep(*k);
        if (!lack)
            continue;
        if (want && *want) {
            if (why && whylen)
                snprintf(why, whylen, "PKTITO_KERNELS=%s uses %s, which this CPU does not have",
                         want, lack);
            return NULL;
        }
    }
    /* only a PKTITO_KERNELS that names nothing here gets this far: ref, last,
     * needs nothing */
    if (why && whylen)
        snprintf(why, whylen, "PKTITO_KERNELS=%s is not among this build's backends",
                 want ? want : "");
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
