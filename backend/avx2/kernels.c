/* The x86 backends: x86-64 with AVX2 and FMA, and with AVX-VNNI.
 *
 * This file is two backends. Built as backend/avx2 it asks for AVX2 and FMA
 * (Haswell, 2013, and every desktop and laptop core since, Atom-class
 * included from Gracemont); built as backend/avx2-vnni (that directory's
 * kernels.c includes this one with PK_AVX_VNNI) it multiplies int8 with
 * VPDPBUSD, which Alder Lake's P and E cores, later Intel cores and AMD's
 * Zen 5 have in the 256-bit AVX-VNNI encoding. Only the int8 product
 * differs between them.
 *
 * x86 has no instruction that multiplies signed bytes by signed bytes.
 * VPMADDUBSW multiplies unsigned by signed and adds pairs into 16 bits,
 * saturating; with the weight's sign moved onto the activation (VPSIGNB:
 * |w| times x * sign(w)) both factors are at most 127, a pair at most
 * 2 * 127^2, and nothing saturates - which is why pk_k_quantize never makes
 * -128. (A weight of -128, which no converted file has, would still be
 * right: VPABSB leaves it 0x80, which VPMADDUBSW reads as 128, and a pair
 * of 128 * 127 fits.) VPDPBUSD multiplies unsigned by signed into 32 bits with no
 * saturation, so one factor can be offset into unsigned instead - the
 * weight, w + 128 - and 128 times the block's sum of the other taken back
 * off: one instruction a group where the sign trick takes four.
 *
 * Every kernel computes backend/ref's bits (tests/kcheck): the int8 sums
 * are integers, and every float operation is ref's in each lane, VMAXPS and
 * VMINPS with their operands in the order that makes a NaN the bound, as
 * fmaxf and fminf do, VROUNDPS to nearest for rintf, VFMADD for fmaf.
 */
#include "kernels.h"
#include "pkf.h"
#include "vmath.h"

#include <immintrin.h>

#ifdef PK_AVX_VNNI
/* VPDPBUSD in whichever encoding the build has: AVX-VNNI's VEX form
 * (Alder Lake and later, Zen 5), or AVX512-VNNI's EVEX form at 256 bits
 * (Ice Lake, Tiger Lake, Sapphire Rapids, Zen 4), the same operation. The
 * Makefile asks for the one the CPU has. Only the VEX form has run here
 * (backend/avx2/CLIMB.md); the EVEX one is built, not yet measured. */
#if defined(__AVXVNNI__)
#define PK_DPBUSD(acc, u8, s8) _mm256_dpbusd_avx_epi32(acc, u8, s8)
#elif defined(__AVX512VNNI__) && defined(__AVX512VL__)
#define PK_DPBUSD(acc, u8, s8) _mm256_dpbusd_epi32(acc, u8, s8)
#else
#error "backend/avx2-vnni wants AVX-VNNI or AVX512-VNNI with AVX512-VL"
#endif
#endif
#include <math.h>
#include <string.h>

/* ---------------------------------------------------------- vector math --- */

static inline __m256 v_exp(__m256 x) {
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(PK_EXP_LO)), _mm256_set1_ps(PK_EXP_HI));
    __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(PK_LOG2E)),
                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fmadd_ps(n, _mm256_set1_ps(-PK_LN2_HI), x);
    r = _mm256_fmadd_ps(n, _mm256_set1_ps(-PK_LN2_LO), r);
    __m256 p = _mm256_set1_ps(PK_EXP_C6);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(PK_EXP_C5));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(PK_EXP_C4));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(PK_EXP_C3));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(PK_EXP_C2));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

static inline __m256 v_sigmoid(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    __m256 nx = _mm256_xor_ps(x, _mm256_set1_ps(-0.0f));
    return _mm256_div_ps(one, _mm256_add_ps(one, v_exp(nx)));
}

static inline __m256 v_tanh(__m256 x) {
    const __m256 sign = _mm256_set1_ps(-0.0f), one = _mm256_set1_ps(1.0f);
    __m256 a = _mm256_andnot_ps(sign, x);
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 p = _mm256_set1_ps(PK_TANH_T4);
    p = _mm256_fmadd_ps(p, x2, _mm256_set1_ps(PK_TANH_T3));
    p = _mm256_fmadd_ps(p, x2, _mm256_set1_ps(PK_TANH_T2));
    p = _mm256_fmadd_ps(p, x2, _mm256_set1_ps(PK_TANH_T1));
    p = _mm256_fmadd_ps(p, x2, _mm256_set1_ps(PK_TANH_T0));
    __m256 small = _mm256_fmadd_ps(_mm256_mul_ps(x, x2), p, x);
    __m256 e = v_exp(_mm256_mul_ps(a, _mm256_set1_ps(2.0f)));
    __m256 big = _mm256_sub_ps(one, _mm256_div_ps(_mm256_set1_ps(2.0f), _mm256_add_ps(e, one)));
    big = _mm256_or_ps(big, _mm256_and_ps(x, sign));                    /* copysignf: big >= 0 */
    return _mm256_blendv_ps(big, small, _mm256_cmp_ps(a, _mm256_set1_ps(PK_TANH_X0), _CMP_LT_OQ));
}

static inline __m256 v_silu(__m256 x) { return _mm256_mul_ps(x, v_sigmoid(x)); }

/* PK_LANES' reduction: sixteen lanes in two vectors, combined as ref's
 * acc[i] += acc[i + h] for h = 8, 4, 2, 1 */
static inline float lanes_sum(__m256 a0, __m256 a1) {
    __m256 s = _mm256_add_ps(a0, a1);                                   /* h = 8 */
    __m128 q = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));  /* h = 4 */
    q = _mm_add_ps(q, _mm_movehl_ps(q, q));                             /* h = 2 */
    q = _mm_add_ss(q, _mm_shuffle_ps(q, q, 1));                         /* h = 1 */
    return _mm_cvtss_f32(q);
}

static inline float hmax(__m256 v) {
    __m128 q = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    q = _mm_max_ps(q, _mm_movehl_ps(q, q));
    q = _mm_max_ss(q, _mm_shuffle_ps(q, q, 1));
    return _mm_cvtss_f32(q);
}

static inline int32_t load32(const int8_t *p) {
    int32_t v;
    memcpy(&v, p, 4);
    return v;
}

/* ------------------------------------------------------- int8 products --- */

void pk_k_quantize(pk_act *a, const float *x, size_t xs) {
    const uint32_t nb = a->n / a->qb;
    const __m256 sign = _mm256_set1_ps(-0.0f);
    for (uint32_t t = 0; t < a->T; t++)
        for (uint32_t b = 0; b < nb; b++) {
            const float *v = x + t * xs + (size_t)b * a->qb;
            int8_t *q = a->q + (size_t)t * a->n + (size_t)b * a->qb;
            __m256 m = _mm256_setzero_ps();
            for (uint32_t i = 0; i < a->qb; i += 8)
                m = _mm256_max_ps(_mm256_andnot_ps(sign, _mm256_loadu_ps(v + i)), m);
            float amax = hmax(m);
            a->d[(size_t)t * nb + b] = amax / 127.0f;
            const __m256 id = _mm256_set1_ps(amax > 0 ? 127.0f / amax : 0.0f);
            for (uint32_t i = 0; i < a->qb; i += 32) {
                __m256i q0 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(_mm256_loadu_ps(v + i), id),
                                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                __m256i q1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(_mm256_loadu_ps(v + i + 8), id),
                                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                __m256i q2 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(_mm256_loadu_ps(v + i + 16), id),
                                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                __m256i q3 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(_mm256_loadu_ps(v + i + 24), id),
                                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                /* the packs work within 128-bit lanes; the permute puts the
                 * eight groups of four back in order */
                __m256i h = _mm256_packs_epi16(_mm256_packs_epi32(q0, q1), _mm256_packs_epi32(q2, q3));
                h = _mm256_permutevar8x32_epi32(h, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
                _mm256_storeu_si256((__m256i *)(void *)(q + i), h);
            }
        }
}

/* Activation bytes a column block may hold: the columns a call takes at a
 * time, across all of its panels, so that the block stays in L2 while every
 * panel's weights stream past it. */
#ifndef PK_Q8_BLOCK_BYTES
#define PK_Q8_BLOCK_BYTES (192 * 1024)
#endif

#ifdef PK_AVX_VNNI

/* VPDPBUSD takes one factor unsigned. Offsetting it by 128 (an XOR of its
 * sign bit) makes the product
 *
 *   sum (u + 128) v = sum u v + 128 sum v
 *
 * and the second term is taken back off. Which factor carries the offset is
 * a question of what the correction costs. At T >= PK_Q8_XOFF columns the
 * activations do: they are XORed once a column block into a thread-local
 * copy, so the inner loop does no XOR at all, and the correction is 128
 * times each block's sum of a panel's weights, taken a panel at a time: 1/cb
 * of the product. A narrower product (the decoder, one column) offsets the
 * weights instead - an XOR a weight vector, for every column of the tile -
 * and the correction is each column's block sum of activations, 1/rows of
 * the product, where the weights' sums would have cost as much as the
 * product itself (CLIMB.md, H1). */
#define PK_Q8_XOFF 16

static void colsums(int32_t *cs, const pk_act *a) {
    const uint32_t n = a->n, qb = a->qb, nb = n / qb;
    const __m256i one = _mm256_set1_epi8(1);
    for (uint32_t t = 0; t < a->T; t++) {
        const int8_t *x = a->q + (size_t)t * n;
        for (uint32_t b = 0; b < nb; b++) {
            __m256i s = _mm256_setzero_si256();
            for (uint32_t k = b * qb; k < (b + 1) * qb; k += 32)
                s = PK_DPBUSD(s, one, _mm256_loadu_si256((const __m256i *)(const void *)(x + k)));
            __m128i q = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
            q = _mm_add_epi32(q, _mm_shuffle_epi32(q, 0x4E));
            q = _mm_add_epi32(q, _mm_shuffle_epi32(q, 0xB1));
            cs[(size_t)t * nb + b] = _mm_cvtsi128_si32(q) * 128;
        }
    }
}

/* One tile: a whole panel, 16 rows in two vectors, by NC columns (up to
 * three: 6 integer and 6 float accumulators, the two weight vectors and a
 * broadcast in 15 of the 16 ymm registers). Each block's exact integer sums
 * go into the float accumulators by one fused multiply-add, as ref's chain.
 *
 * XOFF: x is the column block's offset copy (row t0 - c0 of it at
 * x + (t0 - c0) n) and wsum the panel's weight sums, two vectors a block.
 * Otherwise x is the activations and cs the tile's columns' sums. */
/* one group of four inputs into the tile: g the group's 64 weight bytes,
 * k the group's first input */
#define PK_Q8_GROUP(g, k)                                                           \
    do {                                                                            \
        __m256i w0 = _mm256_loadu_si256((const __m256i *)(const void *)(g));        \
        __m256i w1 = _mm256_loadu_si256((const __m256i *)(const void *)((g) + 32)); \
        if (!XOFF) {                                                                \
            w0 = _mm256_xor_si256(w0, off);                                         \
            w1 = _mm256_xor_si256(w1, off);                                         \
        }                                                                           \
        for (int c = 0; c < NC; c++) {                                              \
            int32_t v;                                                              \
            memcpy(&v, x[c] + (k), 4);                                              \
            const __m256i xb = _mm256_set1_epi32(v);                                \
            if (XOFF) {                                                             \
                s0[c] = PK_DPBUSD(s0[c], xb, w0);                     \
                s1[c] = PK_DPBUSD(s1[c], xb, w1);                     \
            } else {                                                                \
                s0[c] = PK_DPBUSD(s0[c], w0, xb);                     \
                s1[c] = PK_DPBUSD(s1[c], w1, xb);                     \
            }                                                                       \
        }                                                                           \
    } while (0)

static inline __attribute__((always_inline))
void q8_tile(float *out, size_t os, const int8_t *w, uint32_t n, const float *ws,
             const float *bias, const pk_act *a, uint32_t t0, int add,
             const uint8_t *xo, const __m256i *wsum, const int32_t *cs, const int XOFF,
             const int NC) {
    const uint32_t qb = a->qb, nb = n / qb;
    const __m256i off = _mm256_set1_epi8((char)0x80);
    __m256 f0[3], f1[3];
    const uint8_t *x[3];
    for (int c = 0; c < NC; c++) {
        f0[c] = f1[c] = _mm256_setzero_ps();
        x[c] = XOFF ? xo + (size_t)c * n : (const uint8_t *)(a->q + (size_t)(t0 + c) * n);
    }
    for (uint32_t b = 0; b < nb; b++) {
        __m256i s0[3], s1[3];
        for (int c = 0; c < NC; c++) s0[c] = s1[c] = _mm256_setzero_si256();
        if (qb == 32) {
            /* the encoder's blocks: eight groups at constant offsets, which
             * gcc keeps the accumulators in place across (H2) */
            const int8_t *g = w + (size_t)b * 32 * 16;
            const uint32_t k = b * 32;
            PK_Q8_GROUP(g, k);
            PK_Q8_GROUP(g + 64, k + 4);
            PK_Q8_GROUP(g + 128, k + 8);
            PK_Q8_GROUP(g + 192, k + 12);
            PK_Q8_GROUP(g + 256, k + 16);
            PK_Q8_GROUP(g + 320, k + 20);
            PK_Q8_GROUP(g + 384, k + 24);
            PK_Q8_GROUP(g + 448, k + 28);
        } else {
            for (uint32_t k = b * qb; k < (b + 1) * qb; k += 4)
                PK_Q8_GROUP(w + (size_t)k * 16, k);
        }
        for (int c = 0; c < NC; c++) {
            __m256i c0v, c1v;
            if (XOFF) {
                c0v = wsum[2 * b];
                c1v = wsum[2 * b + 1];
            } else {
                c0v = c1v = _mm256_set1_epi32(cs[(size_t)c * nb + b]);
            }
            const __m256 d = _mm256_set1_ps(a->d[(size_t)(t0 + c) * nb + b]);
            f0[c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(s0[c], c0v)), d, f0[c]);
            f1[c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(s1[c], c1v)), d, f1[c]);
        }
    }
    const __m256 sc0 = _mm256_loadu_ps(ws), sc1 = _mm256_loadu_ps(ws + 8);
    for (int c = 0; c < NC; c++) {
        float *o = out + (size_t)(t0 + c) * os;
        __m256 v0 = _mm256_mul_ps(f0[c], sc0), v1 = _mm256_mul_ps(f1[c], sc1);
        if (bias) {
            v0 = _mm256_add_ps(v0, _mm256_loadu_ps(bias));
            v1 = _mm256_add_ps(v1, _mm256_loadu_ps(bias + 8));
        }
        if (add) {
            v0 = _mm256_add_ps(_mm256_loadu_ps(o), v0);
            v1 = _mm256_add_ps(_mm256_loadu_ps(o + 8), v1);
        }
        _mm256_storeu_ps(o, v0);
        _mm256_storeu_ps(o + 8, v1);
    }
}

/* the tiles of one panel over columns [c0, c1), three at a time */
#define PK_Q8_TILES(XOFF, XO, WSUM, CS)                                             \
    do {                                                                            \
        uint32_t t = c0;                                                            \
        for (; t + 3 <= c1; t += 3)                                                 \
            q8_tile(o, os, wp, n, ws + r, b, a, t, add, XO, WSUM, CS, XOFF, 3);     \
        if (c1 - t == 2)                                                            \
            q8_tile(o, os, wp, n, ws + r, b, a, t, add, XO, WSUM, CS, XOFF, 2);     \
        else if (c1 - t == 1)                                                       \
            q8_tile(o, os, wp, n, ws + r, b, a, t, add, XO, WSUM, CS, XOFF, 1);     \
    } while (0)

void pk_k_gemm_q8(float *out, size_t os, const int8_t *w, const float *ws, const float *bias,
                  uint32_t rows, const pk_act *a, int add) {
    const uint32_t n = a->n, nb = n / a->qb;
    if (a->T < PK_Q8_XOFF) {
        int32_t cs[PK_Q8_XOFF * PK_MAX_BLOCKS];
        colsums(cs, a);
        const uint32_t c0 = 0, c1 = a->T;
        for (uint32_t r = 0; r < rows; r += 16) {
            const int8_t *wp = w + (size_t)r * n;
            float *o = out + r;
            const float *b = bias ? bias + r : NULL;
            PK_Q8_TILES(0, NULL, NULL, cs + (size_t)(t - c0) * nb);
        }
        return;
    }
    static _Thread_local _Alignas(64) uint8_t xoff[PK_Q8_BLOCK_BYTES + 3 * 32 * PK_MAX_BLOCKS];
    __m256i wsum[2 * PK_MAX_BLOCKS];
    const __m256i one = _mm256_set1_epi8(1), off = _mm256_set1_epi8((char)0x80);
    uint32_t cb = PK_Q8_BLOCK_BYTES / n;
    cb = cb < 3 ? 3 : cb - cb % 3;
    for (uint32_t c0 = 0; c0 < a->T; c0 += cb) {
        const uint32_t c1 = a->T - c0 < cb ? a->T : c0 + cb;
        const int8_t *src = a->q + (size_t)c0 * n;
        for (size_t i = 0; i < (size_t)(c1 - c0) * n; i += 32)
            _mm256_store_si256((__m256i *)(void *)(xoff + i),
                               _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(const void *)(src + i)), off));
        for (uint32_t r = 0; r < rows; r += 16) {
            const int8_t *wp = w + (size_t)r * n;
            float *o = out + r;
            const float *b = bias ? bias + r : NULL;
            for (uint32_t blk = 0; blk < nb; blk++) {
                __m256i s0 = _mm256_setzero_si256(), s1 = s0;
                for (uint32_t k = blk * a->qb; k < (blk + 1) * a->qb; k += 4) {
                    const int8_t *g = wp + (size_t)k * 16;
                    s0 = PK_DPBUSD(s0, one, _mm256_loadu_si256((const __m256i *)(const void *)g));
                    s1 = PK_DPBUSD(s1, one, _mm256_loadu_si256((const __m256i *)(const void *)(g + 32)));
                }
                wsum[2 * blk] = _mm256_slli_epi32(s0, 7);
                wsum[2 * blk + 1] = _mm256_slli_epi32(s1, 7);
            }
            PK_Q8_TILES(1, xoff + (size_t)(t - c0) * n, wsum, NULL);
        }
    }
}

#else /* VPMADDUBSW */

/* one group of four inputs into the tile: g the half panel's 32 weight
 * bytes, k the group's first input. |w| once for every column, then each
 * column's x sign(w), so that VPMADDUBSW multiplies |w| by it: w x, with
 * both factors at most 127 and a pair at most 2 * 127^2 (CLIMB.md, H3). */
#define PK_Q8_GROUP(g, k)                                                          \
    do {                                                                           \
        const __m256i wv = _mm256_loadu_si256((const __m256i *)(const void *)(g)); \
        const __m256i wa = _mm256_abs_epi8(wv);                                    \
        for (int c = 0; c < NC; c++) {                                             \
            const __m256i xb = _mm256_set1_epi32(load32(x[c] + (k)));              \
            const __m256i p = _mm256_maddubs_epi16(wa, _mm256_sign_epi8(xb, wv));  \
            s[c] = _mm256_add_epi32(s[c], _mm256_madd_epi16(p, ones));             \
        }                                                                          \
    } while (0)

/* One tile: 8 rows (half a panel, bytes 32h .. 32h+31 of each group) by NC
 * columns. Each block's integer sums go into the float accumulators by one
 * fused multiply-add, as ref's chain. */
static inline __attribute__((always_inline))
void q8_tile(float *out, size_t os, const int8_t *w, uint32_t n, uint32_t h, const float *ws,
             const float *bias, const pk_act *a, uint32_t t0, int add, const int NC) {
    const uint32_t qb = a->qb, nb = n / qb;
    __m256 f[4];
    for (int c = 0; c < NC; c++) f[c] = _mm256_setzero_ps();
    const int8_t *x[4];
    for (int c = 0; c < NC; c++) x[c] = a->q + (size_t)(t0 + c) * n;
    const __m256i ones = _mm256_set1_epi16(1);
    const int8_t *wh = w + 32 * h;
    for (uint32_t b = 0; b < nb; b++) {
        __m256i s[4];
        for (int c = 0; c < NC; c++) s[c] = _mm256_setzero_si256();
        if (qb == 32) {
            /* the encoder's blocks: eight groups at constant offsets, which
             * gcc keeps the accumulators in place across (H2, H3) */
            const int8_t *g = wh + (size_t)b * 32 * 16;
            const uint32_t k = b * 32;
            PK_Q8_GROUP(g, k);
            PK_Q8_GROUP(g + 64, k + 4);
            PK_Q8_GROUP(g + 128, k + 8);
            PK_Q8_GROUP(g + 192, k + 12);
            PK_Q8_GROUP(g + 256, k + 16);
            PK_Q8_GROUP(g + 320, k + 20);
            PK_Q8_GROUP(g + 384, k + 24);
            PK_Q8_GROUP(g + 448, k + 28);
        } else {
            for (uint32_t k = b * qb; k < (b + 1) * qb; k += 4)
                PK_Q8_GROUP(wh + (size_t)k * 16, k);
        }
        for (int c = 0; c < NC; c++) {
            __m256 d = _mm256_set1_ps(a->d[(size_t)(t0 + c) * nb + b]);
            f[c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s[c]), d, f[c]);
        }
    }
    const __m256 sc = _mm256_loadu_ps(ws + 8 * h);
    for (int c = 0; c < NC; c++) {
        float *o = out + (size_t)(t0 + c) * os + 8 * h;
        __m256 v = _mm256_mul_ps(f[c], sc);
        if (bias) v = _mm256_add_ps(v, _mm256_loadu_ps(bias + 8 * h));
        if (add) v = _mm256_add_ps(_mm256_loadu_ps(o), v);
        _mm256_storeu_ps(o, v);
    }
}

/* Column blocks; in each, panel by panel, each in its two halves of 8 rows,
 * four columns at a time across the half. */
void pk_k_gemm_q8(float *out, size_t os, const int8_t *w, const float *ws, const float *bias,
                  uint32_t rows, const pk_act *a, int add) {
    const uint32_t n = a->n;
    uint32_t cb = PK_Q8_BLOCK_BYTES / n;
    cb = cb < 4 ? 4 : cb & ~3u;
    for (uint32_t c0 = 0; c0 < a->T; c0 += cb) {
        const uint32_t c1 = a->T - c0 < cb ? a->T : c0 + cb;
        for (uint32_t r = 0; r < rows; r += 16) {
            const int8_t *wp = w + (size_t)r * n;
            float *o = out + r;
            const float *b = bias ? bias + r : NULL;
            for (uint32_t h = 0; h < 2; h++) {
                uint32_t t = c0;
                for (; t + 4 <= c1; t += 4) q8_tile(o, os, wp, n, h, ws + r, b, a, t, add, 4);
                switch (c1 - t) {
                case 3: q8_tile(o, os, wp, n, h, ws + r, b, a, t, add, 3); break;
                case 2: q8_tile(o, os, wp, n, h, ws + r, b, a, t, add, 2); break;
                case 1: q8_tile(o, os, wp, n, h, ws + r, b, a, t, add, 1); break;
                default: break;
                }
            }
        }
    }
}

#endif

/* ------------------------------------------------------ float products --- */

/* MR vectors of 8 rows by NC columns, accumulated in registers. Vector m is
 * lanes 8(m%2) .. 8(m%2)+7 of panel m/2, so a tile of 16 rows is one panel,
 * read as one stream. vadito's kernel, with its bias optional. */
static inline __attribute__((always_inline))
void f32_tile(float *out, size_t os, const float *w, uint32_t ld, uint32_t n,
              const float *x, size_t xs, const float *bias, size_t bs, int relu,
              const int MR, const int NC) {
    __m256 acc[8][4];
    for (int c = 0; c < NC; c++)
        for (int m = 0; m < MR; m++) acc[m][c] = _mm256_setzero_ps();
    for (uint32_t k = 0; k < n; k++) {
        __m256 wv[8];
        for (int m = 0; m < MR; m++)
            wv[m] = _mm256_loadu_ps(w + (size_t)(m / 2) * ld * 16 + 16 * (size_t)k + 8 * (m % 2));
        for (int c = 0; c < NC; c++) {
            __m256 xv = _mm256_broadcast_ss(x + c * xs + k);
            for (int m = 0; m < MR; m++) acc[m][c] = _mm256_fmadd_ps(wv[m], xv, acc[m][c]);
        }
    }
    /* the bias, then ref's v > 0 ? v : 0: VMAXPS gives its second operand,
     * 0, for a NaN and for -0 */
    const __m256 zero = _mm256_setzero_ps();
    for (int c = 0; c < NC; c++)
        for (int m = 0; m < MR; m++) {
            __m256 v = _mm256_add_ps(acc[m][c], bias ? _mm256_loadu_ps(bias + c * bs + 8 * m) : zero);
            _mm256_storeu_ps(out + c * os + 8 * m, relu ? _mm256_max_ps(v, zero) : v);
        }
}

void pk_k_gemm_f32(float *out, size_t os, const float *w, uint32_t ld, uint32_t rows, uint32_t n,
                   const float *x, size_t xs, uint32_t T,
                   const float *bias, size_t bs, int relu) {
    if (T == 1 && rows % 64 == 0) {
        for (uint32_t r = 0; r < rows; r += 64)
            f32_tile(out + r, os, w + (size_t)r * ld, ld, n, x, xs, bias ? bias + r : NULL, bs,
                     relu, 8, 1);
        return;
    }
    for (uint32_t r = 0; r < rows; r += 16) {
        const float *wp = w + (size_t)r * ld;
        for (uint32_t t = 0; t < T; t += 4) {
            float *o = out + t * os + r;
            const float *xt = x + t * xs, *bt = bias ? bias + t * bs + r : NULL;
            switch (T - t < 4 ? T - t : 4) {
            case 4: f32_tile(o, os, wp, ld, n, xt, xs, bt, bs, relu, 2, 4); break;
            case 3: f32_tile(o, os, wp, ld, n, xt, xs, bt, bs, relu, 2, 3); break;
            case 2: f32_tile(o, os, wp, ld, n, xt, xs, bt, bs, relu, 2, 2); break;
            default: f32_tile(o, os, wp, ld, n, xt, xs, bt, bs, relu, 2, 1); break;
            }
        }
    }
}

/* ------------------------------------------------- reductions and rows --- */

void pk_k_layernorm(float *out, size_t os, const float *x, size_t xs, uint32_t T, uint32_t n,
                    const float *w, const float *b, float eps) {
    for (uint32_t t = 0; t < T; t++) {
        const float *v = x + t * xs;
        float *o = out + t * os;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0;
        for (uint32_t i = 0; i < n; i += 16) {
            a0 = _mm256_add_ps(a0, _mm256_loadu_ps(v + i));
            a1 = _mm256_add_ps(a1, _mm256_loadu_ps(v + i + 8));
        }
        float mean = lanes_sum(a0, a1) / (float)n;
        __m256 mv = _mm256_set1_ps(mean);
        a0 = a1 = _mm256_setzero_ps();
        for (uint32_t i = 0; i < n; i += 16) {
            __m256 c0 = _mm256_sub_ps(_mm256_loadu_ps(v + i), mv);
            __m256 c1 = _mm256_sub_ps(_mm256_loadu_ps(v + i + 8), mv);
            a0 = _mm256_add_ps(a0, _mm256_mul_ps(c0, c0));
            a1 = _mm256_add_ps(a1, _mm256_mul_ps(c1, c1));
        }
        float var = lanes_sum(a0, a1) / (float)n;
        __m256 rv = _mm256_set1_ps(1.0f / sqrtf(var + eps));
        for (uint32_t i = 0; i < n; i += 8) {
            __m256 y = _mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(v + i), mv), rv),
                                     _mm256_loadu_ps(w + i));
            _mm256_storeu_ps(o + i, _mm256_add_ps(y, _mm256_loadu_ps(b + i)));
        }
    }
}

void pk_k_softmax(float *x, uint32_t n, float scale) {
    const __m256 sv = _mm256_set1_ps(scale);
    __m256 mv = _mm256_set1_ps(-INFINITY);
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 y = _mm256_mul_ps(_mm256_loadu_ps(x + i), sv);
        _mm256_storeu_ps(x + i, y);
        mv = _mm256_max_ps(y, mv);
    }
    float m = hmax(mv);
    for (; i < n; i++) {
        x[i] = x[i] * scale;
        m = fmaxf(m, x[i]);
    }
    /* lane i % 16 takes element i: vector j of two holds lanes 8j .. 8j+7 */
    const __m256 mm = _mm256_set1_ps(m);
    __m256 a[2] = { _mm256_setzero_ps(), _mm256_setzero_ps() };
    for (i = 0; i + 8 <= n; i += 8) {
        __m256 e = v_exp(_mm256_sub_ps(_mm256_loadu_ps(x + i), mm));
        _mm256_storeu_ps(x + i, e);
        a[(i / 8) % 2] = _mm256_add_ps(a[(i / 8) % 2], e);
    }
    float tail[16];
    _mm256_storeu_ps(tail, a[0]);
    _mm256_storeu_ps(tail + 8, a[1]);
    for (; i < n; i++) {
        x[i] = pk_expf(x[i] - m);
        tail[i % 16] += x[i];
    }
    float inv = 1.0f / lanes_sum(_mm256_loadu_ps(tail), _mm256_loadu_ps(tail + 8));
    const __m256 iv = _mm256_set1_ps(inv);
    for (i = 0; i + 8 <= n; i += 8) _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), iv));
    for (; i < n; i++) x[i] = x[i] * inv;
}

void pk_k_silu(float *x, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(x + i, v_silu(_mm256_loadu_ps(x + i)));
    for (; i < n; i++) x[i] = x[i] * pk_sigmoidf(x[i]);
}

void pk_k_glu(float *out, size_t os, const float *x, size_t xs, uint32_t T, uint32_t n) {
    for (uint32_t t = 0; t < T; t++) {
        const float *a = x + t * xs;
        float *o = out + t * os;
        for (uint32_t i = 0; i < n; i += 8)
            _mm256_storeu_ps(o + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), v_sigmoid(_mm256_loadu_ps(a + n + i))));
    }
}

/* the taps of 16 channels at a time, turned tap-major on the stack so that
 * a vector of them is one load */
#define PK_DW_KMAX 63

void pk_k_dwconv(float *out, size_t os, const float *x, size_t xs, uint32_t T, uint32_t C,
                 const float *w, const float *b, uint32_t K) {
    float wt[PK_DW_KMAX * 16];
    for (uint32_t c0 = 0; c0 < C; c0 += 16) {
        for (uint32_t j = 0; j < K; j++)
            for (uint32_t i = 0; i < 16; i++) wt[j * 16 + i] = w[(size_t)(c0 + i) * K + j];
        const __m256 b0 = _mm256_loadu_ps(b + c0), b1 = _mm256_loadu_ps(b + c0 + 8);
        for (uint32_t t = 0; t < T; t++) {
            __m256 a0 = _mm256_setzero_ps(), a1 = a0;
            for (uint32_t j = 0; j < K; j++) {
                int64_t s = (int64_t)t + j - K / 2;
                if (s < 0 || s >= T)
                    continue;
                const float *p = x + (size_t)s * xs + c0;
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(wt + j * 16), _mm256_loadu_ps(p), a0);
                a1 = _mm256_fmadd_ps(_mm256_loadu_ps(wt + j * 16 + 8), _mm256_loadu_ps(p + 8), a1);
            }
            _mm256_storeu_ps(out + (size_t)t * os + c0, v_silu(_mm256_add_ps(a0, b0)));
            _mm256_storeu_ps(out + (size_t)t * os + c0 + 8, v_silu(_mm256_add_ps(a1, b1)));
        }
    }
}

void pk_k_conv3x3s2(float *out, const float *in, uint32_t r0, uint32_t Ti, uint32_t Fi,
                    uint32_t C, int depthwise, const float *w, const float *b, int relu,
                    uint32_t t0, uint32_t t1) {
    const uint32_t Fo = (Fi + 1) / 2, cin = depthwise ? C : 1;
    const __m256 zero = _mm256_setzero_ps();
    for (uint32_t c = 0; c < C; c += 8) {
        __m256 wt[9];
        for (int k = 0; k < 9; k++) {
            float v[8];
            for (int i = 0; i < 8; i++) v[i] = w[(size_t)(c + i) * 9 + k];
            wt[k] = _mm256_loadu_ps(v);
        }
        const __m256 bv = _mm256_loadu_ps(b + c);
        for (uint32_t t = t0; t < t1; t++)
            for (uint32_t f = 0; f < Fo; f++) {
                __m256 acc = zero;
                for (int di = 0; di < 3; di++) {
                    int64_t ti = 2 * (int64_t)t + di - 1;
                    if (ti < 0 || ti >= Ti)
                        continue;
                    for (int dj = 0; dj < 3; dj++) {
                        int64_t fi = 2 * (int64_t)f + dj - 1;
                        if (fi < 0 || fi >= Fi)
                            continue;
                        const float *p = in + ((size_t)(ti - r0) * Fi + (size_t)fi) * cin;
                        __m256 v = depthwise ? _mm256_loadu_ps(p + c) : _mm256_broadcast_ss(p);
                        acc = _mm256_fmadd_ps(wt[3 * di + dj], v, acc);
                    }
                }
                __m256 v = _mm256_add_ps(acc, bv);
                if (relu) v = _mm256_max_ps(v, zero);      /* 0 for a NaN and for -0, as ref */
                _mm256_storeu_ps(out + ((size_t)(t - t0) * Fo + f) * C + c, v);
            }
    }
}

void pk_k_lstm(float *h, float *c, const float *g, uint32_t H) {
    for (uint32_t j = 0; j < H; j += 8) {
        __m256 i = v_sigmoid(_mm256_loadu_ps(g + j));
        __m256 f = v_sigmoid(_mm256_loadu_ps(g + H + j));
        __m256 u = v_tanh(_mm256_loadu_ps(g + 2 * H + j));
        __m256 o = v_sigmoid(_mm256_loadu_ps(g + 3 * H + j));
        __m256 cj = _mm256_add_ps(_mm256_mul_ps(f, _mm256_loadu_ps(c + j)), _mm256_mul_ps(i, u));
        _mm256_storeu_ps(c + j, cj);
        _mm256_storeu_ps(h + j, _mm256_mul_ps(o, v_tanh(cj)));
    }
}

/* --------------------------------------------------------------- STFT --- */

/* the 8x8 transpose: r[i] lane j becomes r[j] lane i */
static inline void transpose8(__m256 r[8]) {
    __m256 t[8], s[8];
    for (int i = 0; i < 4; i++) {
        t[2 * i] = _mm256_unpacklo_ps(r[2 * i], r[2 * i + 1]);
        t[2 * i + 1] = _mm256_unpackhi_ps(r[2 * i], r[2 * i + 1]);
    }
    for (int i = 0; i < 2; i++) {
        s[4 * i] = _mm256_shuffle_ps(t[4 * i], t[4 * i + 2], _MM_SHUFFLE(1, 0, 1, 0));
        s[4 * i + 1] = _mm256_shuffle_ps(t[4 * i], t[4 * i + 2], _MM_SHUFFLE(3, 2, 3, 2));
        s[4 * i + 2] = _mm256_shuffle_ps(t[4 * i + 1], t[4 * i + 3], _MM_SHUFFLE(1, 0, 1, 0));
        s[4 * i + 3] = _mm256_shuffle_ps(t[4 * i + 1], t[4 * i + 3], _MM_SHUFFLE(3, 2, 3, 2));
    }
    for (int i = 0; i < 4; i++) {
        r[i] = _mm256_permute2f128_ps(s[i], s[i + 4], 0x20);
        r[i + 4] = _mm256_permute2f128_ps(s[i], s[i + 4], 0x31);
    }
}

/* backend/ref's real FFT with eight windows in eight lanes, every line
 * ref's line on vectors (vadito's, its magnitudes left squared). */
void pk_k_stft(float *out, size_t os, const float *const *x, uint32_t count,
               const pk_fft *f, float *scratch) {
    const uint32_t n = f->n, m = n / 2;
    const float *w = f->window, *fc = f->fcos, *fs = f->fsin;
    __m256 *zr = (__m256 *)(void *)scratch, *zi = zr + m;
    for (uint32_t w0 = 0; w0 < count; w0 += 8) {
        const float *xs[8];
        for (int i = 0; i < 8; i++) xs[i] = x[w0 + i < count ? w0 + i : count - 1];
        for (uint32_t s = 0; s < n; s += 8) {
            __m256 r[8];
            for (int i = 0; i < 8; i++) r[i] = _mm256_loadu_ps(xs[i] + s);
            transpose8(r);                        /* r[i] is sample s + i of each window */
            for (int i = 0; i < 4; i++) {
                zr[f->rev[s / 2 + i]] = _mm256_mul_ps(r[2 * i], _mm256_set1_ps(w[s + 2 * i]));
                zi[f->rev[s / 2 + i]] = _mm256_mul_ps(r[2 * i + 1], _mm256_set1_ps(w[s + 2 * i + 1]));
            }
        }
        for (uint32_t len = 2; len <= m; len *= 2) {
            uint32_t half = len / 2, step = n / len;
            for (uint32_t s = 0; s < m; s += len)
                for (uint32_t j = 0; j < half; j++) {
                    __m256 cs = _mm256_set1_ps(fc[j * step]), sn = _mm256_set1_ps(fs[j * step]);
                    uint32_t p = s + j, q = p + half;
                    __m256 tr = _mm256_add_ps(_mm256_mul_ps(cs, zr[q]), _mm256_mul_ps(sn, zi[q]));
                    __m256 ti = _mm256_sub_ps(_mm256_mul_ps(cs, zi[q]), _mm256_mul_ps(sn, zr[q]));
                    zr[q] = _mm256_sub_ps(zr[p], tr);
                    zi[q] = _mm256_sub_ps(zi[p], ti);
                    zr[p] = _mm256_add_ps(zr[p], tr);
                    zi[p] = _mm256_add_ps(zi[p], ti);
                }
        }
        const __m256 half = _mm256_set1_ps(0.5f), nhalf = _mm256_set1_ps(-0.5f);
        __m256 o[8], em = _mm256_sub_ps(zr[0], zi[0]), top = _mm256_mul_ps(em, em);
        for (uint32_t k0 = 0; k0 < m; k0 += 8) {
            for (uint32_t i = 0; i < 8; i++) {
                uint32_t k = k0 + i;
                if (k == 0) {
                    __m256 e0 = _mm256_add_ps(zr[0], zi[0]);
                    o[i] = _mm256_mul_ps(e0, e0);
                    continue;
                }
                __m256 ar = zr[k], ai = zi[k], br = zr[m - k], bi = zi[m - k];
                __m256 er = _mm256_mul_ps(half, _mm256_add_ps(ar, br));
                __m256 ei = _mm256_mul_ps(half, _mm256_sub_ps(ai, bi));
                __m256 odr = _mm256_mul_ps(half, _mm256_add_ps(ai, bi));
                __m256 odi = _mm256_mul_ps(nhalf, _mm256_sub_ps(ar, br));
                __m256 cs = _mm256_set1_ps(fc[k]), sn = _mm256_set1_ps(fs[k]);
                __m256 re = _mm256_add_ps(er, _mm256_add_ps(_mm256_mul_ps(cs, odr), _mm256_mul_ps(sn, odi)));
                __m256 im = _mm256_add_ps(ei, _mm256_sub_ps(_mm256_mul_ps(cs, odi), _mm256_mul_ps(sn, odr)));
                o[i] = _mm256_add_ps(_mm256_mul_ps(re, re), _mm256_mul_ps(im, im));
            }
            transpose8(o);                        /* o[i] is window i's bins k0 .. k0+7 */
            for (uint32_t i = 0; i < 8 && w0 + i < count; i++) _mm256_storeu_ps(out + (w0 + i) * os + k0, o[i]);
        }
        float t[8];
        _mm256_storeu_ps(t, top);
        for (uint32_t i = 0; i < 8 && w0 + i < count; i++) out[(w0 + i) * os + m] = t[i];
    }
}

#ifdef PK_AVX_VNNI
const char *pk_k_name(void) { return "avx2-vnni (x86-64 AVX2 + FMA: VPDPBUSD int8, fp32)"; }
#else
const char *pk_k_name(void) { return "avx2 (x86-64 AVX2 + FMA: VPMADDUBSW int8, fp32)"; }
#endif
