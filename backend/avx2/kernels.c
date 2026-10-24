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
 * (Alder Lake and later, Zen 5), or AVX512-VNNI's EVEX form (Ice Lake,
 * Tiger Lake, Sapphire Rapids, Zen 4 and 5), the same operation. At 256
 * bits the two ran level on a 9950X (CLIMB.md, 2026-10-20); the EVEX build
 * multiplies in zmm (below), where both CPUs that have the two run it
 * faster (CLIMB.md, 2026-10-24). The Makefile builds both, and the CPU runs
 * the first it can (backend/kernels.c). */
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

static void k_quantize(pk_act *a, const float *x, size_t xs) {
    const uint32_t nb = a->n / a->qb;
    const __m256 sign = _mm256_set1_ps(-0.0f);
    for (uint32_t t = 0; t < a->T; t++)
        for (uint32_t b = 0; b < nb; b++) {
            const float *v = x + t * xs + (size_t)b * a->qb;
            int8_t *q = a->q + (size_t)t * a->n + (size_t)b * a->qb;
            __m256 m = _mm256_setzero_ps();
            __m256 bad = _mm256_setzero_ps();
            for (uint32_t i = 0; i < a->qb; i += 8) {
                __m256 z = _mm256_loadu_ps(v + i);
                bad = _mm256_or_ps(bad, _mm256_cmp_ps(z, z, _CMP_UNORD_Q));
                m = _mm256_max_ps(_mm256_andnot_ps(sign, z), m);
            }
            float amax = hmax(m);
            if (_mm256_movemask_ps(bad) || !isfinite(amax)) {
                a->d[(size_t)t * nb + b] = 0;
                memset(q, 0, a->qb);
                continue;
            }
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

#ifdef __AVX512F__
/* AVX-512 (the avx512-vnni build): a panel of 16 rows is one zmm of int32
 * sums, and a group of four inputs for them the 64 weight bytes one zmm
 * load takes, so a group is one load and one VPDPBUSD a column where the
 * ymm tile below takes two of each. The 32 registers hold PK_Q8_NC columns,
 * two zmm each (the block's integer sums and the float chain), beside the
 * weight vector and a broadcast (CLIMB.md, 2026-10-24). Every lane does
 * what the ymm tile's does, so the bits are the same. */
#ifndef PK_Q8_NC
#define PK_Q8_NC 8
#endif
#if PK_Q8_NC < 1 || PK_Q8_NC > 12
#error "PK_Q8_NC is 1 to 12: the tiles below are written for at most 12 columns"
#endif

#define PK_Q8_GROUP(g, k)                                                \
    do {                                                                 \
        __m512i wv = _mm512_loadu_si512((const void *)(g));              \
        if (!XOFF)                                                       \
            wv = _mm512_xor_si512(wv, off);                              \
        for (int c = 0; c < NC; c++) {                                   \
            int32_t v;                                                   \
            memcpy(&v, x[c] + (k), 4);                                   \
            const __m512i xb = _mm512_set1_epi32(v);                     \
            s[c] = XOFF ? _mm512_dpbusd_epi32(s[c], xb, wv)              \
                        : _mm512_dpbusd_epi32(s[c], wv, xb);             \
        }                                                                \
    } while (0)

static inline __attribute__((always_inline))
void q8_tile(float *out, size_t os, const int8_t *w, uint32_t n, const float *ws,
             const float *bias, const pk_act *a, uint32_t t0, int add,
             const uint8_t *xo, const __m512i *wsum, const int32_t *cs, const int XOFF,
             const int NC) {
    const uint32_t qb = a->qb, nb = n / qb;
    const __m512i off = _mm512_set1_epi8((char)0x80);
    __m512 f[12];
    const uint8_t *x[12];
    for (int c = 0; c < NC; c++) {
        f[c] = _mm512_setzero_ps();
        x[c] = XOFF ? xo + (size_t)c * n : (const uint8_t *)(a->q + (size_t)(t0 + c) * n);
    }
    for (uint32_t b = 0; b < nb; b++) {
        __m512i s[12];
        for (int c = 0; c < NC; c++) s[c] = _mm512_setzero_si512();
        if (qb == 32) {
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
            const __m512i cv = XOFF ? wsum[b] : _mm512_set1_epi32(cs[(size_t)c * nb + b]);
            const __m512 d = _mm512_set1_ps(a->d[(size_t)(t0 + c) * nb + b]);
            f[c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(s[c], cv)), d, f[c]);
        }
    }
    const __m512 sc = _mm512_loadu_ps(ws);
    for (int c = 0; c < NC; c++) {
        float *o = out + (size_t)(t0 + c) * os;
        __m512 v = _mm512_mul_ps(f[c], sc);
        if (bias) v = _mm512_add_ps(v, _mm512_loadu_ps(bias));
        if (add) v = _mm512_add_ps(_mm512_loadu_ps(o), v);
        _mm512_storeu_ps(o, v);
    }
}

/* the tiles of one panel over columns [c0, c1), PK_Q8_NC at a time */
#define PK_Q8_TILE_N(N, XOFF, XO, WSUM, CS)                                              \
    q8_tile(o, os, wp, n, ws + r, b, a, t, add, XO, WSUM, CS, XOFF, N)
#define PK_Q8_TILES(XOFF, XO, WSUM, CS)                                                  \
    do {                                                                                 \
        uint32_t t = c0;                                                                 \
        for (; t + PK_Q8_NC <= c1; t += PK_Q8_NC)                                        \
            PK_Q8_TILE_N(PK_Q8_NC, XOFF, XO, WSUM, CS);                                                      \
        switch (c1 - t) {                                                                \
        case 11: PK_Q8_TILE_N(11, XOFF, XO, WSUM, CS); break;                                                \
        case 10: PK_Q8_TILE_N(10, XOFF, XO, WSUM, CS); break;                                                \
        case 9: PK_Q8_TILE_N(9, XOFF, XO, WSUM, CS); break;                                                  \
        case 8: PK_Q8_TILE_N(8, XOFF, XO, WSUM, CS); break;                                                  \
        case 7: PK_Q8_TILE_N(7, XOFF, XO, WSUM, CS); break;                                                  \
        case 6: PK_Q8_TILE_N(6, XOFF, XO, WSUM, CS); break;                                                  \
        case 5: PK_Q8_TILE_N(5, XOFF, XO, WSUM, CS); break;                                                  \
        case 4: PK_Q8_TILE_N(4, XOFF, XO, WSUM, CS); break;                                                  \
        case 3: PK_Q8_TILE_N(3, XOFF, XO, WSUM, CS); break;                                                  \
        case 2: PK_Q8_TILE_N(2, XOFF, XO, WSUM, CS); break;                                                  \
        case 1: PK_Q8_TILE_N(1, XOFF, XO, WSUM, CS); break;                                                  \
        default: break;                                                                  \
        }                                                                                \
    } while (0)

static void q8_zmm(float *out, size_t os, const int8_t *w, const float *ws, const float *bias,
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
    static _Thread_local _Alignas(64) uint8_t xoff[PK_Q8_BLOCK_BYTES + PK_Q8_NC * 32 * PK_MAX_BLOCKS];
    __m512i wsum[PK_MAX_BLOCKS];
    const __m512i one = _mm512_set1_epi8(1), off = _mm512_set1_epi8((char)0x80);
    uint32_t cb = PK_Q8_BLOCK_BYTES / n;
    cb = cb < PK_Q8_NC ? PK_Q8_NC : cb - cb % PK_Q8_NC;
    for (uint32_t c0 = 0; c0 < a->T; c0 += cb) {
        const uint32_t c1 = a->T - c0 < cb ? a->T : c0 + cb;
        const int8_t *src = a->q + (size_t)c0 * n;
        for (size_t i = 0; i < (size_t)(c1 - c0) * n; i += 32)
            _mm256_store_si256((__m256i *)(void *)(xoff + i),
                               _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(const void *)(src + i)),
                                                _mm512_castsi512_si256(off)));
        for (uint32_t r = 0; r < rows; r += 16) {
            const int8_t *wp = w + (size_t)r * n;
            float *o = out + r;
            const float *b = bias ? bias + r : NULL;
            for (uint32_t blk = 0; blk < nb; blk++) {
                __m512i s = _mm512_setzero_si512();
                for (uint32_t k = blk * a->qb; k < (blk + 1) * a->qb; k += 4)
                    s = _mm512_dpbusd_epi32(s, one, _mm512_loadu_si512((const void *)(wp + (size_t)k * 16)));
                wsum[blk] = _mm512_slli_epi32(s, 7);
            }
            PK_Q8_TILES(1, xoff + (size_t)(t - c0) * n, wsum, NULL);
        }
    }
}


#ifdef __AMX_INT8__
/* AMX (the amx-int8 build). TDPBSSD multiplies tile A, signed bytes, by
 * tile B, signed bytes, into tile C's exact int32 sums: C[m][r] += sum over
 * i of A[m][i] B[i/4][4r + i%4]. A block of one panel is a B tile as the
 * file holds it (8 rows of a group's 64 bytes: four inputs for each of the
 * 16 rows), A is 16 columns' 32 bytes of the block, n apart, and C the 16
 * columns by the panel's 16 rows. Signed by signed, so no offset. Each
 * block's C goes to L1 and comes back a column at a time into that
 * column's float chain, as the zmm tile's sums do, while the next block's
 * TDPBSSD runs in the other C tile. Four C tiles in turn, to wait out
 * more of TDPBSSD's latency, ran 1.5% slower on a c7i, and panels outer
 * no faster (CLIMB.md, 2026-10-24). Columns past the last whole 16, and
 * products of fewer, take the zmm tile. */
static void amx_config(void) {
    static _Thread_local int ready;
    if (ready)
        return;
    _Alignas(64) struct {
        uint8_t palette, start, reserved[14];
        uint16_t colsb[16];
        uint8_t rows[16];
    } cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.palette = 1;
    cfg.colsb[0] = 32, cfg.rows[0] = 16;                         /* A */
    cfg.colsb[1] = 64, cfg.rows[1] = 8;                          /* B */
    cfg.colsb[2] = 64, cfg.rows[2] = 16;                         /* C, even blocks */
    cfg.colsb[3] = 64, cfg.rows[3] = 16;                         /* C, odd blocks */
    _tile_loadconfig(&cfg);
    ready = 1;
}

/* one block's 16 columns' sums, in c, into their float chains */
static inline __attribute__((always_inline))
void amx_chains(__m512 f[16], const int32_t *c, const float *d, uint32_t nb) {
    for (int m = 0; m < 16; m++)
        f[m] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512((const void *)(c + 16 * m))),
                               _mm512_set1_ps(d[(size_t)m * nb]), f[m]);
}

static void k_gemm_q8(float *out, size_t os, const int8_t *w, const float *ws, const float *bias,
                      uint32_t rows, const pk_act *a, int add) {
    const uint32_t n = a->n, qb = a->qb, nb = n / qb, T16 = a->T & ~15u;
    if (qb != 32 || T16 == 0) {
        q8_zmm(out, os, w, ws, bias, rows, a, add);
        return;
    }
    amx_config();
    _Alignas(64) int32_t cbuf[2][256];
    for (uint32_t t = 0; t < T16; t += 16) {
        const int8_t *x = a->q + (size_t)t * n;
        const float *d = a->d + (size_t)t * nb;
        for (uint32_t r = 0; r < rows; r += 16) {
            const int8_t *wp = w + (size_t)r * n;
            __m512 f[16];
            for (int m = 0; m < 16; m++) f[m] = _mm512_setzero_ps();
            /* block b's product runs in tile 2 or 3 while block b-1's sums
             * go into the chains */
            for (uint32_t b = 0; b < nb; b++) {
                _tile_loadd(0, x + (size_t)b * 32, n);
                _tile_loadd(1, wp + (size_t)b * 512, 64);
                if (b & 1) {
                    _tile_zero(3);
                    _tile_dpbssd(3, 0, 1);
                    amx_chains(f, cbuf[0], d + b - 1, nb);
                    _tile_stored(3, cbuf[1], 64);
                } else {
                    _tile_zero(2);
                    _tile_dpbssd(2, 0, 1);
                    if (b)
                        amx_chains(f, cbuf[1], d + b - 1, nb);
                    _tile_stored(2, cbuf[0], 64);
                }
            }
            amx_chains(f, cbuf[(nb - 1) & 1], d + nb - 1, nb);
            const __m512 sc = _mm512_loadu_ps(ws + r);
            for (int m = 0; m < 16; m++) {
                float *o = out + (size_t)(t + m) * os + r;
                __m512 v = _mm512_mul_ps(f[m], sc);
                if (bias) v = _mm512_add_ps(v, _mm512_loadu_ps(bias + r));
                if (add) v = _mm512_add_ps(_mm512_loadu_ps(o), v);
                _mm512_storeu_ps(o, v);
            }
        }
    }
    if (T16 < a->T) {
        const pk_act rest = { a->q + (size_t)T16 * n, a->d + (size_t)T16 * nb, n, qb, a->T - T16 };
        q8_zmm(out + (size_t)T16 * os, os, w, ws, bias, rows, &rest, add);
    }
}
#else
static void k_gemm_q8(float *out, size_t os, const int8_t *w, const float *ws, const float *bias,
                      uint32_t rows, const pk_act *a, int add) {
    q8_zmm(out, os, w, ws, bias, rows, a, add);
}
#endif /* __AMX_INT8__ */

#else
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

static void k_gemm_q8(float *out, size_t os, const int8_t *w, const float *ws, const float *bias,
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

#endif /* __AVX512F__ */

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
static void k_gemm_q8(float *out, size_t os, const int8_t *w, const float *ws, const float *bias,
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

static void k_gemm_f32(float *out, size_t os, const float *w, uint32_t ld, uint32_t rows, uint32_t n,
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

static void k_layernorm(float *out, size_t os, const float *x, size_t xs, uint32_t T, uint32_t n,
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

static void k_softmax(float *x, uint32_t n, float scale) {
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

static void k_silu(float *x, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(x + i, v_silu(_mm256_loadu_ps(x + i)));
    for (; i < n; i++) x[i] = x[i] * pk_sigmoidf(x[i]);
}

static void k_glu(float *out, size_t os, const float *x, size_t xs, uint32_t T, uint32_t n) {
    for (uint32_t t = 0; t < T; t++) {
        const float *a = x + t * xs;
        float *o = out + t * os;
        for (uint32_t i = 0; i < n; i += 8)
            _mm256_storeu_ps(o + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), v_sigmoid(_mm256_loadu_ps(a + n + i))));
    }
}

/* The convolution module's depthwise convolution 64 channels and 64 frames
 * at a time and, in that, eight channels by eight frames at once: eight
 * chains of FMAs in flight, where a frame at a time was two. The taps are
 * turned tap-major on the stack first. The frames whose taps would read
 * past either end are taken one at a time, those taps skipped, as ref. */
#define PK_DW_KMAX 63          /* the model file's bound (core/model.c) */
#define DW_TB 64

static __m256 dw_one(const float *x, size_t xs, uint32_t T, const float *wg, uint32_t K,
                     uint32_t t) {
    __m256 acc = _mm256_setzero_ps();
    for (uint32_t j = 0; j < K; j++) {
        int64_t s = (int64_t)t + j - K / 2;
        if (s >= 0 && s < T)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(wg + 8 * j), _mm256_loadu_ps(x + (size_t)s * xs), acc);
    }
    return acc;
}

static void k_dwconv(float *out, size_t os, const float *x, size_t xs, uint32_t T, uint32_t C,
                     const float *w, const float *b, uint32_t K) {
    const uint32_t h = K / 2, end = T > h ? T - h : 0;     /* [h, end): every tap inside */
    float wt[64 * PK_DW_KMAX];
    for (uint32_t c0 = 0; c0 < C; c0 += 64) {
        const uint32_t cn = C - c0 < 64 ? C - c0 : 64;
        for (uint32_t c = 0; c < cn; c++)
            for (uint32_t j = 0; j < K; j++)
                wt[(c / 8) * K * 8 + j * 8 + c % 8] = w[(size_t)(c0 + c) * K + j];
        for (uint32_t t0 = 0; t0 < T; t0 += DW_TB) {
            const uint32_t t1 = T - t0 < DW_TB ? T : t0 + DW_TB;
            const uint32_t lo = t0 > h ? t0 : h, hi = t1 < end ? t1 : end;
            for (uint32_t c = 0; c < cn; c += 8) {
                const float *wg = wt + (size_t)(c / 8) * K * 8, *xc = x + c0 + c;
                const __m256 bv = _mm256_loadu_ps(b + c0 + c);
                float *oc = out + c0 + c;
                uint32_t t = t0;
                if (hi >= lo + 8) {
                    for (; t < lo; t++)
                        _mm256_storeu_ps(oc + (size_t)t * os,
                                         v_silu(_mm256_add_ps(dw_one(xc, xs, T, wg, K, t), bv)));
                    /* the last eight overlap the eight before when hi - lo is not a multiple */
                    for (; t < hi; t += 8) {
                        const uint32_t g = t + 8 <= hi ? t : hi - 8;
                        __m256 acc[8];
                        for (int q = 0; q < 8; q++)
                            acc[q] = _mm256_setzero_ps();
                        for (uint32_t j = 0; j < K; j++) {
                            const __m256 wj = _mm256_loadu_ps(wg + 8 * j);
                            const float *p = xc + (size_t)(g + j - h) * xs;
                            for (int q = 0; q < 8; q++)
                                acc[q] = _mm256_fmadd_ps(wj, _mm256_loadu_ps(p + (size_t)q * xs), acc[q]);
                        }
                        for (int q = 0; q < 8; q++)
                            _mm256_storeu_ps(oc + (size_t)(g + q) * os, v_silu(_mm256_add_ps(acc[q], bv)));
                    }
                    t = hi;
                }
                for (; t < t1; t++)
                    _mm256_storeu_ps(oc + (size_t)t * os,
                                     v_silu(_mm256_add_ps(dw_one(xc, xs, T, wg, K, t), bv)));
            }
        }
    }
}

/* The subsampling's 3x3 convolutions a row of output at a time and, in it,
 * eight channels by eight columns at once: eight chains of FMAs in flight,
 * where a column at a time was one, and the whole block a channel group at
 * a time read every row of a 2 MB input once a group. A row's three input
 * rows stay in L2. Each column's taps are ref's, in ref's order: the rows
 * outside the image are left out of every column, and the columns a tap
 * would read past an edge (the first, and the last of an odd width) are
 * taken one at a time. The weights are put tap-major first, CONV_CW
 * channels at a time, a group's nine taps one run. */
#define CONV_CW 256

typedef struct {
    const float *row[3];     /* the input rows inside the image, column 0 */
    int di[3], n;            /* each one's tap row */
    uint32_t cin;
} conv_rows;

/* column f alone, the taps past either edge skipped */
static __m256 conv_col(const conv_rows *r, const float *wg, uint32_t f, uint32_t Fi, uint32_t c,
                       int depthwise) {
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < r->n; i++)
        for (int dj = 0; dj < 3; dj++) {
            int64_t fi = 2 * (int64_t)f + dj - 1;
            if (fi < 0 || fi >= Fi)
                continue;
            const float *p = r->row[i] + (size_t)fi * r->cin;
            __m256 v = depthwise ? _mm256_loadu_ps(p + c) : _mm256_broadcast_ss(p);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(wg + (3 * r->di[i] + dj) * 8), v, acc);
        }
    return acc;
}

/* columns f .. f+7, every tap inside the image */
static inline __attribute__((always_inline))
void conv_8(__m256 acc[8], const conv_rows *r, const float *wg, uint32_t f, uint32_t c,
            int depthwise) {
    const size_t s = r->cin;
    for (int q = 0; q < 8; q++)
        acc[q] = _mm256_setzero_ps();
    for (int i = 0; i < r->n; i++) {
        const float *p = r->row[i] + (size_t)(2 * f - 1) * s + (depthwise ? c : 0);
        const float *wk = wg + 3 * r->di[i] * 8;
        const __m256 w0 = _mm256_loadu_ps(wk), w1 = _mm256_loadu_ps(wk + 8);
        const __m256 w2 = _mm256_loadu_ps(wk + 16);
        for (int q = 0; q < 8; q++) {
            const float *x = p + (size_t)(2 * q) * s;
            __m256 x0 = depthwise ? _mm256_loadu_ps(x) : _mm256_broadcast_ss(x);
            __m256 x1 = depthwise ? _mm256_loadu_ps(x + s) : _mm256_broadcast_ss(x + s);
            __m256 x2 = depthwise ? _mm256_loadu_ps(x + 2 * s) : _mm256_broadcast_ss(x + 2 * s);
            acc[q] = _mm256_fmadd_ps(w2, x2, _mm256_fmadd_ps(w1, x1, _mm256_fmadd_ps(w0, x0, acc[q])));
        }
    }
}

static inline __m256 conv_out(__m256 acc, __m256 bv, int relu) {
    __m256 v = _mm256_add_ps(acc, bv);
    return relu ? _mm256_max_ps(v, _mm256_setzero_ps()) : v;   /* 0 for a NaN and for -0, as ref */
}

static void conv_row(float *o, const conv_rows *r, const float *wt, const float *b, uint32_t cn,
                     uint32_t C, uint32_t Fi, int depthwise, int relu) {
    const uint32_t Fo = (Fi + 1) / 2, fe = Fi % 2 ? Fo - 1 : Fo;   /* [1, fe): no edge */
    for (uint32_t c = 0; c < cn; c += 8) {
        const float *wg = wt + (size_t)(c / 8) * 72;
        const __m256 bv = _mm256_loadu_ps(b + c);
        uint32_t f = 0;
        if (fe > 8) {
            _mm256_storeu_ps(o + c, conv_out(conv_col(r, wg, 0, Fi, c, depthwise), bv, relu));
            /* the last eight overlap the eight before when fe - 1 is not a multiple */
            for (f = 1; f < fe; f += 8) {
                const uint32_t g = f + 8 <= fe ? f : fe - 8;
                __m256 acc[8];
                if (depthwise)
                    conv_8(acc, r, wg, g, c, 1);
                else
                    conv_8(acc, r, wg, g, c, 0);
                for (int q = 0; q < 8; q++)
                    _mm256_storeu_ps(o + (size_t)(g + q) * C + c, conv_out(acc[q], bv, relu));
            }
            f = fe;
        }
        for (; f < Fo; f++)
            _mm256_storeu_ps(o + (size_t)f * C + c,
                             conv_out(conv_col(r, wg, f, Fi, c, depthwise), bv, relu));
    }
}

static void k_conv3x3s2(float *out, const float *in, uint32_t r0, uint32_t Ti, uint32_t Fi,
                        uint32_t C, int depthwise, const float *w, const float *b, int relu,
                        uint32_t t0, uint32_t t1) {
    const uint32_t Fo = (Fi + 1) / 2, cin = depthwise ? C : 1;
    float wt[9 * CONV_CW];
    for (uint32_t c0 = 0; c0 < C; c0 += CONV_CW) {
        const uint32_t cn = C - c0 < CONV_CW ? C - c0 : CONV_CW;
        for (uint32_t c = 0; c < cn; c++)
            for (int k = 0; k < 9; k++)
                wt[(c / 8) * 72 + k * 8 + c % 8] = w[(size_t)(c0 + c) * 9 + k];
        for (uint32_t t = t0; t < t1; t++) {
            conv_rows r = { .n = 0, .cin = cin };
            for (int di = 0; di < 3; di++) {
                int64_t ti = 2 * (int64_t)t + di - 1;
                if (ti < 0 || ti >= Ti)
                    continue;
                r.row[r.n] = in + (size_t)(ti - r0) * Fi * cin + (depthwise ? c0 : 0);
                r.di[r.n++] = di;
            }
            conv_row(out + (size_t)(t - t0) * Fo * C + c0, &r, wt, b + c0, cn, C, Fi, depthwise, relu);
        }
    }
}

static void k_lstm(float *h, float *c, const float *g, uint32_t H) {
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
static void k_stft(float *out, size_t os, const float *const *x, uint32_t count,
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

/* The resampler's chains, eight blocks at a time at each phase: a lane a
 * block, the tap broadcast and the eight inputs, orig floats apart, one
 * VGATHERDPS, each lane ref's chain. Two phases at once keep two chains in
 * flight against the FMA's latency. The blocks that make no eight go one
 * at a time, on the FMA instruction (fmaf, with -mfma). */
static void k_resample(float *y, const float *x, uint32_t nb, uint32_t orig, uint32_t nw, const float *k,
                       const size_t *at, const uint32_t *first, const uint32_t *len) {
    const __m256i lane = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_set1_epi32((int)orig));
    uint32_t b = 0;
    for (; b + 8 <= nb; b += 8) {
        const float *xb = x + (size_t)b * orig;
        float *yb = y + (size_t)b * nw;
        uint32_t j = 0;
        for (; j + 2 <= nw; j += 2) {
            const float *k0 = k + at[j], *k1 = k + at[j + 1], *x0 = xb + first[j], *x1 = xb + first[j + 1];
            const uint32_t n0 = len[j], n1 = len[j + 1], n = n0 < n1 ? n0 : n1;
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            uint32_t m = 0;
            for (; m < n; m++) {
                a0 = _mm256_fmadd_ps(_mm256_set1_ps(k0[m]), _mm256_i32gather_ps(x0 + m, lane, 4), a0);
                a1 = _mm256_fmadd_ps(_mm256_set1_ps(k1[m]), _mm256_i32gather_ps(x1 + m, lane, 4), a1);
            }
            for (uint32_t r = m; r < n0; r++)
                a0 = _mm256_fmadd_ps(_mm256_set1_ps(k0[r]), _mm256_i32gather_ps(x0 + r, lane, 4), a0);
            for (uint32_t r = m; r < n1; r++)
                a1 = _mm256_fmadd_ps(_mm256_set1_ps(k1[r]), _mm256_i32gather_ps(x1 + r, lane, 4), a1);
            float t0[8], t1[8];
            _mm256_storeu_ps(t0, a0);
            _mm256_storeu_ps(t1, a1);
            for (int l = 0; l < 8; l++) {
                yb[(size_t)l * nw + j] = t0[l];
                yb[(size_t)l * nw + j + 1] = t1[l];
            }
        }
        for (; j < nw; j++) {
            const float *kj = k + at[j], *xj = xb + first[j];
            __m256 a = _mm256_setzero_ps();
            for (uint32_t m = 0; m < len[j]; m++)
                a = _mm256_fmadd_ps(_mm256_set1_ps(kj[m]), _mm256_i32gather_ps(xj + m, lane, 4), a);
            float t[8];
            _mm256_storeu_ps(t, a);
            for (int l = 0; l < 8; l++)
                yb[(size_t)l * nw + j] = t[l];
        }
    }
    for (; b < nb; b++)
        for (uint32_t j = 0; j < nw; j++) {
            const float *kj = k + at[j], *xj = x + (size_t)b * orig + first[j];
            float acc = 0.0f;
            for (uint32_t m = 0; m < len[j]; m++)
                acc = fmaf(kj[m], xj[m], acc);
            y[(size_t)b * nw + j] = acc;
        }
}

#ifdef PK_AVX_VNNI
#if defined(__AVXVNNI__)
static const char *const needs[] = { "avx2", "fma", "avxvnni", NULL };
const pk_kernels pk_kernels_avx2_vnni = {
    .id = "avx2-vnni",
    .name = "avx2-vnni (x86-64 AVX2 + FMA: AVX-VNNI's VPDPBUSD int8, fp32)",
#elif defined(__AMX_INT8__)
static const char *const needs[] = { "avx2", "fma", "avx512f", "avx512vnni", "avx512vl", "amx_tile",
                                     "amx_int8", NULL };
const pk_kernels pk_kernels_amx_int8 = {
    .id = "amx-int8",
    .name = "amx-int8 (x86-64 AVX-512 + FMA: AMX's TDPBSSD int8, fp32)",
#else
static const char *const needs[] = { "avx2", "fma", "avx512f", "avx512vnni", "avx512vl", NULL };
const pk_kernels pk_kernels_avx512_vnni = {
    .id = "avx512-vnni",
    .name = "avx512-vnni (x86-64 AVX2 + FMA: AVX512-VNNI's VPDPBUSD at 512 bits int8, fp32)",
#endif
#else
static const char *const needs[] = { "avx2", "fma", NULL };
const pk_kernels pk_kernels_avx2 = {
    .id = "avx2",
    .name = "avx2 (x86-64 AVX2 + FMA: VPMADDUBSW int8, fp32)",
#endif
    .needs = needs,
    .quantize = k_quantize,
    .gemm_q8 = k_gemm_q8,
    .gemm_f32 = k_gemm_f32,
    .layernorm = k_layernorm,
    .softmax = k_softmax,
    .silu = k_silu,
    .glu = k_glu,
    .dwconv = k_dwconv,
    .conv3x3s2 = k_conv3x3s2,
    .lstm = k_lstm,
    .stft = k_stft,
    .resample = k_resample,
};
