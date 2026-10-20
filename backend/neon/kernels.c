/* The NEON backends: AArch64's SIMD, for every ARMv8 core from the A53 up.
 *
 * This file is two backends. Built as backend/neon it asks for ARMv8.0 and
 * nothing more, which is what the A53 (Pi 3, Zero 2 W) and the A72 (Pi 4)
 * have; built as backend/neon-dp (that directory's kernels.c includes this
 * one with PK_NEON_DOTPROD) it multiplies int8 with SDOT, ARMv8.2's dot
 * product, which the A76 (Pi 5), the A78, the X cores and Apple's M cores
 * have. The int8 product is the only kernel where the two differ: the rest
 * is fp32, the same instructions on every core.
 *
 * Every kernel computes backend/ref's bits (tests/kcheck). The int8
 * products' inner sums are integers, exact in any order; their few float
 * operations a block, and every float kernel, put outputs in a vector's
 * lanes and run ref's scalar chain in each, down to the exp polynomial.
 */
#include "kernels.h"
#include "pkf.h"
#include "vmath.h"

#include <arm_neon.h>
#include <math.h>
#include <string.h>

/* ---------------------------------------------------------- vector math --- */

/* core/vmath.h's functions, four lanes at a time, each operation the scalar
 * one: FMINNM and FMAXNM are fminf and fmaxf (a NaN becomes the bound),
 * FRINTN is rintf in the default rounding mode, FMLA is fmaf, FDIV is the
 * division. (vadito's backend/neon, unchanged.) */
static inline float32x4_t v_exp(float32x4_t x) {
    x = vminnmq_f32(vmaxnmq_f32(x, vdupq_n_f32(PK_EXP_LO)), vdupq_n_f32(PK_EXP_HI));
    float32x4_t n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(PK_LOG2E)));
    float32x4_t r = vfmaq_f32(x, n, vdupq_n_f32(-PK_LN2_HI));
    r = vfmaq_f32(r, n, vdupq_n_f32(-PK_LN2_LO));
    float32x4_t p = vdupq_n_f32(PK_EXP_C6);
    p = vfmaq_f32(vdupq_n_f32(PK_EXP_C5), p, r);
    p = vfmaq_f32(vdupq_n_f32(PK_EXP_C4), p, r);
    p = vfmaq_f32(vdupq_n_f32(PK_EXP_C3), p, r);
    p = vfmaq_f32(vdupq_n_f32(PK_EXP_C2), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(e));
}

static inline float32x4_t v_sigmoid(float32x4_t x) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    return vdivq_f32(one, vaddq_f32(one, v_exp(vnegq_f32(x))));
}

/* both of pk_tanhf's branches, and the lanes take the one the scalar would */
static inline float32x4_t v_tanh(float32x4_t x) {
    float32x4_t a = vabsq_f32(x);
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t p = vdupq_n_f32(PK_TANH_T4);
    p = vfmaq_f32(vdupq_n_f32(PK_TANH_T3), p, x2);
    p = vfmaq_f32(vdupq_n_f32(PK_TANH_T2), p, x2);
    p = vfmaq_f32(vdupq_n_f32(PK_TANH_T1), p, x2);
    p = vfmaq_f32(vdupq_n_f32(PK_TANH_T0), p, x2);
    float32x4_t small = vfmaq_f32(x, vmulq_f32(x, x2), p);
    float32x4_t e = v_exp(vmulq_f32(a, vdupq_n_f32(2.0f)));
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t big = vsubq_f32(one, vdivq_f32(vdupq_n_f32(2.0f), vaddq_f32(e, one)));
    big = vbslq_f32(vdupq_n_u32(0x80000000u), x, big);               /* copysignf */
    return vbslq_f32(vcltq_f32(a, vdupq_n_f32(PK_TANH_X0)), small, big);
}

/* x * sigmoid(x), as ref's pk_k_silu */
static inline float32x4_t v_silu(float32x4_t x) { return vmulq_f32(x, v_sigmoid(x)); }

/* PK_LANES' reduction: sixteen lanes in four vectors, combined as ref's
 * acc[i] += acc[i + h] for h = 8, 4, 2, 1 */
static inline float lanes_sum(float32x4_t a0, float32x4_t a1, float32x4_t a2, float32x4_t a3) {
    a0 = vaddq_f32(a0, a2);                          /* h = 8 */
    a1 = vaddq_f32(a1, a3);
    a0 = vaddq_f32(a0, a1);                          /* h = 4 */
    float32x2_t s = vadd_f32(vget_low_f32(a0), vget_high_f32(a0));   /* h = 2 */
    return vget_lane_f32(s, 0) + vget_lane_f32(s, 1);                /* h = 1 */
}

/* ------------------------------------------------------- int8 products --- */

static void k_quantize(pk_act *a, const float *x, size_t xs) {
    const uint32_t nb = a->n / a->qb;
    for (uint32_t t = 0; t < a->T; t++)
        for (uint32_t b = 0; b < nb; b++) {
            const float *v = x + t * xs + (size_t)b * a->qb;
            int8_t *q = a->q + (size_t)t * a->n + (size_t)b * a->qb;
            float32x4_t m = vdupq_n_f32(0.0f);
            for (uint32_t i = 0; i < a->qb; i += 4)
                m = vmaxnmq_f32(m, vabsq_f32(vld1q_f32(v + i)));
            float amax = vmaxnmvq_f32(m);
            a->d[(size_t)t * nb + b] = amax / 127.0f;
            float32x4_t id = vdupq_n_f32(amax > 0 ? 127.0f / amax : 0.0f);
            for (uint32_t i = 0; i < a->qb; i += 16) {
                int32x4_t q0 = vcvtq_s32_f32(vrndnq_f32(vmulq_f32(vld1q_f32(v + i), id)));
                int32x4_t q1 = vcvtq_s32_f32(vrndnq_f32(vmulq_f32(vld1q_f32(v + i + 4), id)));
                int32x4_t q2 = vcvtq_s32_f32(vrndnq_f32(vmulq_f32(vld1q_f32(v + i + 8), id)));
                int32x4_t q3 = vcvtq_s32_f32(vrndnq_f32(vmulq_f32(vld1q_f32(v + i + 12), id)));
                int16x8_t h0 = vcombine_s16(vmovn_s32(q0), vmovn_s32(q1));
                int16x8_t h1 = vcombine_s16(vmovn_s32(q2), vmovn_s32(q3));
                vst1q_s8(q + i, vcombine_s8(vmovn_s16(h0), vmovn_s16(h1)));
            }
        }
}

/* One tile of the int8 product: 8 rows (half a panel: vectors h and h+1 of
 * each group's four) by NC columns, every block of qb inputs taken in
 * chunks of 16 (four groups), its integer sums turned into the float
 * accumulators by one fused multiply-add per block, as ref's chain.
 *
 * With SDOT a chunk is, per column, one 16-byte load of the activations and
 * eight SDOTs by lane against the eight weight vectors. Without it, a pair
 * of groups is SMULL/SMLAL into 16 bits (two products, at most 2 * 127^2,
 * which fits) and SADALP into 32, leaving each row's sum in two lanes that
 * a pairwise add joins at the block's end. The activation's four bytes are
 * replicated across all sixteen (rep4), so the high half of a weight vector
 * goes to SMULL2/SMLAL2 as it stands rather than being moved down first. */

/* Columns a tile takes. Four with SDOT, which needs 8 int and 8 float
 * accumulators for them; three on ARMv8.0, whose four vectors of pairwise
 * sums a column take 12 int accumulators at three columns, where four
 * columns' 16 spill (CLIMB.md, 2026-10-20, H1). */
#ifdef PK_NEON_DOTPROD
#define PK_Q8_NC 4
#else
#ifndef PK_Q8_NC_V80
#define PK_Q8_NC_V80 3
#endif
#define PK_Q8_NC PK_Q8_NC_V80
#endif

/* four activation bytes, replicated across a vector: gcc 14 makes this
 * `ldr s` + `dup`, which the A53 runs 11% faster than one LD1R (CLIMB.md,
 * 2026-10-20, H2) */
static inline int8x16_t rep4(const int8_t *p) {
    int32_t v;
    memcpy(&v, p, 4);
    return vreinterpretq_s8_s32(vdupq_n_s32(v));
}

#ifndef PK_NEON_DOTPROD
/* One step of the ARMv8.0 tile: a pair of groups, 8 inputs, for NC columns
 * at g (group k/4 of the half panel) and x[c] + k. A block's first step
 * writes the accumulators with SADDLP; the others add to them with SADALP.
 * Zeroing twelve accumulators and adding was twelve moves a block, and
 * this way the same sums cost none (CLIMB.md, 2026-10-20, H8). */
static inline __attribute__((always_inline))
void q8_step_v80(const int8_t *g, const int8_t *const *x, uint32_t k, int32x4_t s[4][4],
                 const int NC, const int first) {
    int8x16_t wa = vld1q_s8(g), wb = vld1q_s8(g + 16);          /* group k/4 */
    int8x16_t wc = vld1q_s8(g + 64), wd = vld1q_s8(g + 80);     /* group k/4 + 1 */
    for (int c = 0; c < NC; c++) {
        int8x16_t x0 = rep4(x[c] + k), x1 = rep4(x[c] + k + 4);
        int16x8_t p0 = vmull_s8(vget_low_s8(wa), vget_low_s8(x0));
        int16x8_t p1 = vmull_high_s8(wa, x0);
        int16x8_t p2 = vmull_s8(vget_low_s8(wb), vget_low_s8(x0));
        int16x8_t p3 = vmull_high_s8(wb, x0);
        p0 = vmlal_s8(p0, vget_low_s8(wc), vget_low_s8(x1));
        p1 = vmlal_high_s8(p1, wc, x1);
        p2 = vmlal_s8(p2, vget_low_s8(wd), vget_low_s8(x1));
        p3 = vmlal_high_s8(p3, wd, x1);
        if (first) {
            s[c][0] = vpaddlq_s16(p0);
            s[c][1] = vpaddlq_s16(p1);
            s[c][2] = vpaddlq_s16(p2);
            s[c][3] = vpaddlq_s16(p3);
        } else {
            s[c][0] = vpadalq_s16(s[c][0], p0);
            s[c][1] = vpadalq_s16(s[c][1], p1);
            s[c][2] = vpadalq_s16(s[c][2], p2);
            s[c][3] = vpadalq_s16(s[c][3], p3);
        }
    }
}
#endif

/* Blocks [b0, b1) of the product, for NC columns from t0. A tile that
 * takes all the blocks starts its float accumulators at zero and finishes
 * the outputs; one that takes a chunk of them (pk_k_gemm_q8 below) starts
 * from part, where the chunk before left them, unless it is the first, and
 * leaves them there unless it is the last - the same chain of fused
 * multiply-adds, in the same order, stored and reloaded exactly. part is
 * NC columns of 8 floats. */
static inline __attribute__((always_inline))
void q8_tile(float *out, size_t os, const int8_t *w, uint32_t n, uint32_t h,
             const float *ws, const float *bias, const pk_act *a, uint32_t t0, int add,
             uint32_t b0, uint32_t b1, float *part, int first, int last, const int NC) {
    const uint32_t qb = a->qb, nb = n / qb;
    float32x4_t f[4][2];
    for (int c = 0; c < NC; c++) {
        if (first) {
            f[c][0] = f[c][1] = vdupq_n_f32(0.0f);
        } else {
            f[c][0] = vld1q_f32(part + 8 * c);
            f[c][1] = vld1q_f32(part + 8 * c + 4);
        }
    }
    const int8_t *x[4];
    for (int c = 0; c < NC; c++) x[c] = a->q + (size_t)(t0 + c) * n;
    for (uint32_t b = b0; b < b1; b++) {
#ifdef PK_NEON_DOTPROD
        int32x4_t s[4][2];
        for (int c = 0; c < NC; c++) s[c][0] = s[c][1] = vdupq_n_s32(0);
        for (uint32_t k = b * qb; k < (b + 1) * qb; k += 16) {
            const int8_t *g = w + (size_t)(k / 4) * 64 + 16 * h;
            int8x16_t w0 = vld1q_s8(g), w1 = vld1q_s8(g + 16);
            int8x16_t w2 = vld1q_s8(g + 64), w3 = vld1q_s8(g + 80);
            int8x16_t w4 = vld1q_s8(g + 128), w5 = vld1q_s8(g + 144);
            int8x16_t w6 = vld1q_s8(g + 192), w7 = vld1q_s8(g + 208);
            for (int c = 0; c < NC; c++) {
                int8x16_t xv = vld1q_s8(x[c] + k);
                s[c][0] = vdotq_laneq_s32(s[c][0], w0, xv, 0);
                s[c][1] = vdotq_laneq_s32(s[c][1], w1, xv, 0);
                s[c][0] = vdotq_laneq_s32(s[c][0], w2, xv, 1);
                s[c][1] = vdotq_laneq_s32(s[c][1], w3, xv, 1);
                s[c][0] = vdotq_laneq_s32(s[c][0], w4, xv, 2);
                s[c][1] = vdotq_laneq_s32(s[c][1], w5, xv, 2);
                s[c][0] = vdotq_laneq_s32(s[c][0], w6, xv, 3);
                s[c][1] = vdotq_laneq_s32(s[c][1], w7, xv, 3);
            }
        }
        for (int c = 0; c < NC; c++) {
            float32x4_t d = vdupq_n_f32(a->d[(size_t)(t0 + c) * nb + b]);
            f[c][0] = vfmaq_f32(f[c][0], vcvtq_f32_s32(s[c][0]), d);
            f[c][1] = vfmaq_f32(f[c][1], vcvtq_f32_s32(s[c][1]), d);
        }
#else
        /* rows 4h..4h+1 and 4h+2..4h+3 of vector h, and the same of h+1:
         * each row's sum in two lanes until the block ends */
        int32x4_t s[4][4];
        uint32_t k = b * qb;
        q8_step_v80(w + (size_t)(k / 4) * 64 + 16 * h, x, k, s, NC, 1);
        for (k += 8; k < (b + 1) * qb; k += 8)
            q8_step_v80(w + (size_t)(k / 4) * 64 + 16 * h, x, k, s, NC, 0);
        for (int c = 0; c < NC; c++) {
            float32x4_t d = vdupq_n_f32(a->d[(size_t)(t0 + c) * nb + b]);
            int32x4_t r0 = vpaddq_s32(s[c][0], s[c][1]), r1 = vpaddq_s32(s[c][2], s[c][3]);
            f[c][0] = vfmaq_f32(f[c][0], vcvtq_f32_s32(r0), d);
            f[c][1] = vfmaq_f32(f[c][1], vcvtq_f32_s32(r1), d);
        }
#endif
    }
    if (!last) {
        for (int c = 0; c < NC; c++) {
            vst1q_f32(part + 8 * c, f[c][0]);
            vst1q_f32(part + 8 * c + 4, f[c][1]);
        }
        return;
    }
    const float32x4_t s0 = vld1q_f32(ws + 4 * h), s1 = vld1q_f32(ws + 4 * h + 4);
    for (int c = 0; c < NC; c++) {
        float *o = out + (size_t)(t0 + c) * os + 4 * h;
        float32x4_t v0 = vmulq_f32(f[c][0], s0), v1 = vmulq_f32(f[c][1], s1);
        if (bias) {
            v0 = vaddq_f32(v0, vld1q_f32(bias + 4 * h));
            v1 = vaddq_f32(v1, vld1q_f32(bias + 4 * h + 4));
        }
        if (add) {
            v0 = vaddq_f32(vld1q_f32(o), v0);
            v1 = vaddq_f32(vld1q_f32(o + 4), v1);
        }
        vst1q_f32(o, v0);
        vst1q_f32(o + 4, v1);
    }
}

/* Activation bytes a column block may hold: the columns a call takes at a
 * time, across all of its panels, so that the block stays in cache while
 * every half panel's weights stream past it. Without the blocking, a
 * 750-column product re-read all its activations (3 MB at 4096 inputs) from
 * L3 for every half panel. On ARMv8.0 the block is a quarter of that: four
 * A53s share 512 KB of L2, and at 48 KB (12 columns of 4096) the ff down
 * product ran 8% faster at four threads than at 192 KB, and no slower alone
 * (CLIMB.md, 2026-10-20, H3). */
#ifndef PK_Q8_BLOCK_BYTES
#ifdef PK_NEON_DOTPROD
#define PK_Q8_BLOCK_BYTES (192 * 1024)
#else
#define PK_Q8_BLOCK_BYTES (48 * 1024)
#endif
#endif

/* One column block's tiles over every panel, blocks [b0, b1) of each */
static void q8_cols(float *out, size_t os, const int8_t *w, const float *ws, const float *bias,
                    uint32_t rows, const pk_act *a, int add, uint32_t c0, uint32_t c1,
                    uint32_t b0, uint32_t b1, float *part, uint32_t cb) {
    const uint32_t n = a->n, nb = n / a->qb;
    const int first = b0 == 0, last = b1 == nb;
    /* a whole product's tiles neither read nor write their partial sums;
     * they get somewhere real to not touch, which keeps gcc's
     * -Warray-bounds from reasoning about a null one */
    float none[4 * 8];
    for (uint32_t r = 0; r < rows; r += 16) {
        const int8_t *wp = w + (size_t)r * n;
        float *o = out + r;
        const float *b = bias ? bias + r : NULL;
        for (uint32_t h = 0; h < 4; h += 2) {
            /* this half panel's partial sums: 8 floats a column of the block */
            float *ph = part ? part + ((size_t)(r / 8 + h / 2) * cb) * 8 : NULL;
            uint32_t t = c0;
#define Q8_TILE(NC) q8_tile(o, os, wp, n, h, ws + r, b, a, t, add, b0, b1, \
                            ph ? ph + (size_t)(t - c0) * 8 : none, first, last, NC)
            for (; t + PK_Q8_NC <= c1; t += PK_Q8_NC) Q8_TILE(PK_Q8_NC);
            switch (c1 - t) {
            case 3: Q8_TILE(3); break;
            case 2: Q8_TILE(2); break;
            case 1: Q8_TILE(1); break;
            default: break;
            }
#undef Q8_TILE
        }
    }
}

/* Inputs a chunk takes, on ARMv8.0: a product wider than this is taken a
 * chunk at a time within each column block, so that a panel's weights for
 * the chunk - 16 KB of cache lines at 1024 - stay in the A53's 32 KB L1D
 * for both its halves and every tile of the block. Whole, a 4096-input half
 * panel's weights span 64 KB of lines, and every tile streamed them again
 * from the L2 the four cores share (CLIMB.md, 2026-10-20, H6). The partial
 * sums between chunks live in part: at most 64 rows (pk_gemm's slices) by
 * one block's columns. */
#ifndef PK_NEON_DOTPROD
#define PK_Q8_KCHUNK 1024
#define PK_Q8_KCHUNK_ROWS 64
#define PK_Q8_KCHUNK_COLS (PK_Q8_BLOCK_BYTES / PK_Q8_KCHUNK)
#endif

/* Column blocks; in each, panel by panel, each in its two halves of 8 rows,
 * PK_Q8_NC columns at a time across the half. */
static void k_gemm_q8(float *out, size_t os, const int8_t *w, const float *ws, const float *bias,
                      uint32_t rows, const pk_act *a, int add) {
    const uint32_t n = a->n, nb = n / a->qb;
#ifdef PK_Q8_KCHUNK
    if (n > PK_Q8_KCHUNK && PK_Q8_KCHUNK % a->qb == 0 && rows <= PK_Q8_KCHUNK_ROWS && a->T > 1) {
        static _Thread_local float part[PK_Q8_KCHUNK_ROWS / 8 * PK_Q8_KCHUNK_COLS * 8];
        const uint32_t kb = PK_Q8_KCHUNK / a->qb, cb = PK_Q8_KCHUNK_COLS;
        for (uint32_t c0 = 0; c0 < a->T; c0 += cb) {
            const uint32_t c1 = a->T - c0 < cb ? a->T : c0 + cb;
            for (uint32_t b0 = 0; b0 < nb; b0 += kb)
                q8_cols(out, os, w, ws, bias, rows, a, add, c0, c1, b0,
                        nb - b0 < kb ? nb : b0 + kb, part, cb);
        }
        return;
    }
#endif
    uint32_t cb = PK_Q8_BLOCK_BYTES / n;
    cb = cb < 4 ? 4 : cb & ~3u;
    for (uint32_t c0 = 0; c0 < a->T; c0 += cb)
        q8_cols(out, os, w, ws, bias, rows, a, add, c0, a->T - c0 < cb ? a->T : c0 + cb, 0, nb,
                NULL, cb);
}

/* ------------------------------------------------------ float products --- */

/* gcc's scheduling before register allocation hoists a tile's sixteen
 * weight loads ahead of its sixteen accumulators and spills them: on a Pi
 * 5 (gcc 14.2) the attention's products ran at 71% of FMLA's peak with it
 * and 86% without, 870 ms of the minute against 727, while the int8 tile
 * above needs it (1977 ms without it against 1976 with). Taking the inputs
 * two at a time to make room spilled nothing and ran slower still, 920 ms
 * (backend/neon-dp/CLIMB.md). The A53, in order, is the other way round:
 * the scheduled code with its spills is the faster there, 2035 ms of
 * attention a minute against 2312 (backend/neon/CLIMB.md), so the pass is
 * turned off for neon-dp only. Clang does not have the pass. */
#if defined(__GNUC__) && !defined(__clang__) && defined(PK_NEON_DOTPROD)
#pragma GCC push_options
#pragma GCC optimize("no-schedule-insns")
#endif

/* MR vectors of 4 rows by NC columns of out, accumulated in registers.
 * Vector m is lanes 4(m%4) .. 4(m%4)+3 of panel m/4, so a tile of 16 rows
 * is one panel and reads its weights as one stream. The inputs are taken
 * four k at a time, one vector per column, each broadcast from its lane by
 * the multiply (FMLA by element). vadito's kernel, with its bias optional. */
static inline __attribute__((always_inline))
void f32_tile(float *out, size_t os, const float *w, uint32_t ld, uint32_t n,
              const float *x, size_t xs, const float *bias, size_t bs, int relu,
              const int MR, const int NC) {
    float32x4_t acc[16][4];
    for (int c = 0; c < NC; c++)
        for (int m = 0; m < MR; m++)
            acc[m][c] = vdupq_n_f32(0.0f);
    uint32_t k = 0;
    for (; k + 4 <= n; k += 4) {
        float32x4_t xv[4];
        for (int c = 0; c < NC; c++) xv[c] = vld1q_f32(x + c * xs + k);
        for (int m = 0; m < MR; m++) {
            const float *wm = w + (size_t)(m / 4) * ld * 16 + 16 * (size_t)k + 4 * (m % 4);
            float32x4_t a = vld1q_f32(wm), b = vld1q_f32(wm + 16);
            float32x4_t d = vld1q_f32(wm + 32), e = vld1q_f32(wm + 48);
            for (int c = 0; c < NC; c++) {
                acc[m][c] = vfmaq_laneq_f32(acc[m][c], a, xv[c], 0);
                acc[m][c] = vfmaq_laneq_f32(acc[m][c], b, xv[c], 1);
                acc[m][c] = vfmaq_laneq_f32(acc[m][c], d, xv[c], 2);
                acc[m][c] = vfmaq_laneq_f32(acc[m][c], e, xv[c], 3);
            }
        }
    }
    for (; k < n; k++)
        for (int m = 0; m < MR; m++) {
            float32x4_t a = vld1q_f32(w + (size_t)(m / 4) * ld * 16 + 16 * (size_t)k + 4 * (m % 4));
            for (int c = 0; c < NC; c++) acc[m][c] = vfmaq_n_f32(acc[m][c], a, x[c * xs + k]);
        }
    /* the bias, then ref's v > 0 ? v : 0: FMAXNM is 0 for a NaN too */
    const float32x4_t zero = vdupq_n_f32(0.0f);
    for (int c = 0; c < NC; c++)
        for (int m = 0; m < MR; m++) {
            float32x4_t v = vaddq_f32(acc[m][c], bias ? vld1q_f32(bias + c * bs + 4 * m) : zero);
            if (relu) v = vmaxnmq_f32(v, zero);
            vst1q_f32(out + c * os + 4 * m, v);
        }
}

static void k_gemm_f32(float *out, size_t os, const float *w, uint32_t ld, uint32_t rows, uint32_t n,
                       const float *x, size_t xs, uint32_t T,
                       const float *bias, size_t bs, int relu) {
    if (T == 1 && rows % 64 == 0) {
        for (uint32_t r = 0; r < rows; r += 64)
            f32_tile(out + r, os, w + (size_t)r * ld, ld, n, x, xs, bias ? bias + r : NULL, bs,
                     relu, 16, 1);
        return;
    }
    for (uint32_t r = 0; r < rows; r += 16) {
        const float *wp = w + (size_t)r * ld;
        for (uint32_t t = 0; t < T; t += 4) {
            float *o = out + t * os + r;
            const float *xt = x + t * xs, *bt = bias ? bias + t * bs + r : NULL;
            switch (T - t < 4 ? T - t : 4) {
            case 4: f32_tile(o, os, wp, ld, n, xt, xs, bt, bs, relu, 4, 4); break;
            case 3: f32_tile(o, os, wp, ld, n, xt, xs, bt, bs, relu, 4, 3); break;
            case 2: f32_tile(o, os, wp, ld, n, xt, xs, bt, bs, relu, 4, 2); break;
            default: f32_tile(o, os, wp, ld, n, xt, xs, bt, bs, relu, 4, 1); break;
            }
        }
    }
}

#if defined(__GNUC__) && !defined(__clang__) && defined(PK_NEON_DOTPROD)
#pragma GCC pop_options
#endif

/* ------------------------------------------------- reductions and rows --- */

static void k_layernorm(float *out, size_t os, const float *x, size_t xs, uint32_t T, uint32_t n,
                        const float *w, const float *b, float eps) {
    for (uint32_t t = 0; t < T; t++) {
        const float *v = x + t * xs;
        float *o = out + t * os;
        float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
        for (uint32_t i = 0; i < n; i += 16) {
            a0 = vaddq_f32(a0, vld1q_f32(v + i));
            a1 = vaddq_f32(a1, vld1q_f32(v + i + 4));
            a2 = vaddq_f32(a2, vld1q_f32(v + i + 8));
            a3 = vaddq_f32(a3, vld1q_f32(v + i + 12));
        }
        float mean = lanes_sum(a0, a1, a2, a3) / (float)n;
        float32x4_t mv = vdupq_n_f32(mean);
        a0 = a1 = a2 = a3 = vdupq_n_f32(0);
        for (uint32_t i = 0; i < n; i += 16) {
            float32x4_t c0 = vsubq_f32(vld1q_f32(v + i), mv), c1 = vsubq_f32(vld1q_f32(v + i + 4), mv);
            float32x4_t c2 = vsubq_f32(vld1q_f32(v + i + 8), mv), c3 = vsubq_f32(vld1q_f32(v + i + 12), mv);
            a0 = vaddq_f32(a0, vmulq_f32(c0, c0));
            a1 = vaddq_f32(a1, vmulq_f32(c1, c1));
            a2 = vaddq_f32(a2, vmulq_f32(c2, c2));
            a3 = vaddq_f32(a3, vmulq_f32(c3, c3));
        }
        float var = lanes_sum(a0, a1, a2, a3) / (float)n;
        float32x4_t rv = vdupq_n_f32(1.0f / sqrtf(var + eps));
        for (uint32_t i = 0; i < n; i += 4) {
            float32x4_t y = vmulq_f32(vmulq_f32(vsubq_f32(vld1q_f32(v + i), mv), rv), vld1q_f32(w + i));
            vst1q_f32(o + i, vaddq_f32(y, vld1q_f32(b + i)));
        }
    }
}

static void k_softmax(float *x, uint32_t n, float scale) {
    const float32x4_t sv = vdupq_n_f32(scale);
    float32x4_t mv = vdupq_n_f32(-INFINITY);
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t y = vmulq_f32(vld1q_f32(x + i), sv);
        vst1q_f32(x + i, y);
        mv = vmaxnmq_f32(mv, y);
    }
    float m = vmaxnmvq_f32(mv);
    for (; i < n; i++) {
        x[i] = x[i] * scale;
        m = fmaxf(m, x[i]);
    }
    /* lane i % 16 takes element i: vector j of four holds lanes 4j .. 4j+3 */
    const float32x4_t mm = vdupq_n_f32(m);
    float32x4_t a[4] = { vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0) };
    for (i = 0; i + 4 <= n; i += 4) {
        float32x4_t e = v_exp(vsubq_f32(vld1q_f32(x + i), mm));
        vst1q_f32(x + i, e);
        a[(i / 4) % 4] = vaddq_f32(a[(i / 4) % 4], e);
    }
    float tail[16] = { 0 };
    vst1q_f32(tail, a[0]);
    vst1q_f32(tail + 4, a[1]);
    vst1q_f32(tail + 8, a[2]);
    vst1q_f32(tail + 12, a[3]);
    for (; i < n; i++) {
        x[i] = pk_expf(x[i] - m);
        tail[i % 16] += x[i];
    }
    float inv = 1.0f / lanes_sum(vld1q_f32(tail), vld1q_f32(tail + 4), vld1q_f32(tail + 8),
                                 vld1q_f32(tail + 12));
    const float32x4_t iv = vdupq_n_f32(inv);
    for (i = 0; i + 4 <= n; i += 4) vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), iv));
    for (; i < n; i++) x[i] = x[i] * inv;
}

static void k_silu(float *x, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(x + i, v_silu(vld1q_f32(x + i)));
    for (; i < n; i++) x[i] = x[i] * pk_sigmoidf(x[i]);
}

static void k_glu(float *out, size_t os, const float *x, size_t xs, uint32_t T, uint32_t n) {
    for (uint32_t t = 0; t < T; t++) {
        const float *a = x + t * xs;
        float *o = out + t * os;
        for (uint32_t i = 0; i < n; i += 4)
            vst1q_f32(o + i, vmulq_f32(vld1q_f32(a + i), v_sigmoid(vld1q_f32(a + n + i))));
    }
}

/* The convolution module's depthwise convolution 64 channels and 64 frames
 * at a time and, in that, four channels by eight frames at once: eight
 * chains of FMAs in flight, where a frame at a time was one chain. The taps
 * are turned tap-major on the stack first. The frames whose taps would read
 * past either end are taken one at a time, those taps skipped, as ref. */
#define PK_DW_KMAX 63          /* the model file's bound (core/model.c) */
#define DW_TB 64

static float32x4_t dw_one(const float *x, size_t xs, uint32_t T, const float *wg, uint32_t K,
                          uint32_t t) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (uint32_t j = 0; j < K; j++) {
        int64_t s = (int64_t)t + j - K / 2;
        if (s >= 0 && s < T)
            acc = vfmaq_f32(acc, vld1q_f32(wg + 4 * j), vld1q_f32(x + (size_t)s * xs));
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
                wt[(c / 4) * K * 4 + j * 4 + c % 4] = w[(size_t)(c0 + c) * K + j];
        for (uint32_t t0 = 0; t0 < T; t0 += DW_TB) {
            const uint32_t t1 = T - t0 < DW_TB ? T : t0 + DW_TB;
            const uint32_t lo = t0 > h ? t0 : h, hi = t1 < end ? t1 : end;
            for (uint32_t c = 0; c < cn; c += 4) {
                const float *wg = wt + (size_t)(c / 4) * K * 4, *xc = x + c0 + c;
                const float32x4_t bv = vld1q_f32(b + c0 + c);
                float *oc = out + c0 + c;
                uint32_t t = t0;
                if (hi >= lo + 8) {
                    for (; t < lo; t++)
                        vst1q_f32(oc + (size_t)t * os, v_silu(vaddq_f32(dw_one(xc, xs, T, wg, K, t), bv)));
                    /* the last eight overlap the eight before when hi - lo is not a multiple */
                    for (; t < hi; t += 8) {
                        const uint32_t g = t + 8 <= hi ? t : hi - 8;
                        float32x4_t acc[8];
                        for (int q = 0; q < 8; q++)
                            acc[q] = vdupq_n_f32(0.0f);
                        for (uint32_t j = 0; j < K; j++) {
                            const float32x4_t wj = vld1q_f32(wg + 4 * j);
                            const float *p = xc + (size_t)(g + j - h) * xs;
                            for (int q = 0; q < 8; q++)
                                acc[q] = vfmaq_f32(acc[q], wj, vld1q_f32(p + (size_t)q * xs));
                        }
                        for (int q = 0; q < 8; q++)
                            vst1q_f32(oc + (size_t)(g + q) * os, v_silu(vaddq_f32(acc[q], bv)));
                    }
                    t = hi;
                }
                for (; t < t1; t++)
                    vst1q_f32(oc + (size_t)t * os, v_silu(vaddq_f32(dw_one(xc, xs, T, wg, K, t), bv)));
            }
        }
    }
}

/* The subsampling's 3x3 convolutions a row of output at a time and, in it,
 * four channels by eight columns at once: eight chains of FMAs in flight,
 * where a column at a time was one chain, four cycles a link, and the whole
 * block a channel group at a time read every row of a 2 MB input once a
 * group. A row's three input rows stay in L2. Each column's taps are ref's,
 * in ref's order: the rows outside the image are left out of every column,
 * and the columns a tap would read past an edge (the first, and the last of
 * an odd width) are taken one at a time. The weights are put tap-major
 * first, CONV_CW channels at a time, a group's nine taps one run. */
#define CONV_CW 256

typedef struct {
    const float *row[3];     /* the input rows inside the image, column 0 */
    int di[3], n;            /* each one's tap row */
    uint32_t cin;
} conv_rows;

/* column f alone, the taps past either edge skipped */
static float32x4_t conv_col(const conv_rows *r, const float *wg, uint32_t f, uint32_t Fi,
                            uint32_t c, int depthwise) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int i = 0; i < r->n; i++)
        for (int dj = 0; dj < 3; dj++) {
            int64_t fi = 2 * (int64_t)f + dj - 1;
            if (fi < 0 || fi >= Fi)
                continue;
            const float *p = r->row[i] + (size_t)fi * r->cin;
            float32x4_t v = depthwise ? vld1q_f32(p + c) : vld1q_dup_f32(p);
            acc = vfmaq_f32(acc, vld1q_f32(wg + (3 * r->di[i] + dj) * 4), v);
        }
    return acc;
}

/* columns f .. f+7, every tap inside the image */
static inline __attribute__((always_inline))
void conv_8(float32x4_t acc[8], const conv_rows *r, const float *wg, uint32_t f, uint32_t c,
            int depthwise) {
    const size_t s = r->cin;
    for (int q = 0; q < 8; q++)
        acc[q] = vdupq_n_f32(0.0f);
    for (int i = 0; i < r->n; i++) {
        const float *p = r->row[i] + (size_t)(2 * f - 1) * s + (depthwise ? c : 0);
        const float *wk = wg + 3 * r->di[i] * 4;
        const float32x4_t w0 = vld1q_f32(wk), w1 = vld1q_f32(wk + 4), w2 = vld1q_f32(wk + 8);
        for (int q = 0; q < 8; q++) {
            const float *x = p + (size_t)(2 * q) * s;
            float32x4_t x0 = depthwise ? vld1q_f32(x) : vld1q_dup_f32(x);
            float32x4_t x1 = depthwise ? vld1q_f32(x + s) : vld1q_dup_f32(x + s);
            float32x4_t x2 = depthwise ? vld1q_f32(x + 2 * s) : vld1q_dup_f32(x + 2 * s);
            acc[q] = vfmaq_f32(vfmaq_f32(vfmaq_f32(acc[q], w0, x0), w1, x1), w2, x2);
        }
    }
}

static inline float32x4_t conv_out(float32x4_t acc, float32x4_t bv, int relu) {
    float32x4_t v = vaddq_f32(acc, bv);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    return relu ? vbslq_f32(vcgtq_f32(v, zero), v, zero) : v;
}

static void conv_row(float *o, const conv_rows *r, const float *wt, const float *b, uint32_t cn,
                     uint32_t C, uint32_t Fi, int depthwise, int relu) {
    const uint32_t Fo = (Fi + 1) / 2, fe = Fi % 2 ? Fo - 1 : Fo;   /* [1, fe): no edge */
    for (uint32_t c = 0; c < cn; c += 4) {
        const float *wg = wt + (size_t)(c / 4) * 36;
        const float32x4_t bv = vld1q_f32(b + c);
        uint32_t f = 0;
        if (fe > 8) {
            vst1q_f32(o + c, conv_out(conv_col(r, wg, 0, Fi, c, depthwise), bv, relu));
            /* the last eight overlap the eight before when fe - 1 is not a multiple */
            for (f = 1; f < fe; f += 8) {
                const uint32_t g = f + 8 <= fe ? f : fe - 8;
                float32x4_t acc[8];
                if (depthwise)
                    conv_8(acc, r, wg, g, c, 1);
                else
                    conv_8(acc, r, wg, g, c, 0);
                for (int q = 0; q < 8; q++)
                    vst1q_f32(o + (size_t)(g + q) * C + c, conv_out(acc[q], bv, relu));
            }
            f = fe;
        }
        for (; f < Fo; f++)
            vst1q_f32(o + (size_t)f * C + c, conv_out(conv_col(r, wg, f, Fi, c, depthwise), bv, relu));
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
                wt[(c / 4) * 36 + k * 4 + c % 4] = w[(size_t)(c0 + c) * 9 + k];
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
    for (uint32_t j = 0; j < H; j += 4) {
        float32x4_t i = v_sigmoid(vld1q_f32(g + j));
        float32x4_t f = v_sigmoid(vld1q_f32(g + H + j));
        float32x4_t u = v_tanh(vld1q_f32(g + 2 * H + j));
        float32x4_t o = v_sigmoid(vld1q_f32(g + 3 * H + j));
        float32x4_t cj = vaddq_f32(vmulq_f32(f, vld1q_f32(c + j)), vmulq_f32(i, u));
        vst1q_f32(c + j, cj);
        vst1q_f32(h + j, vmulq_f32(o, v_tanh(cj)));
    }
}

/* --------------------------------------------------------------- STFT --- */

/* the 4x4 transpose: r[i] lane j becomes r[j] lane i */
static inline void transpose4(float32x4_t r[4]) {
    float32x4_t t0 = vtrn1q_f32(r[0], r[1]), t1 = vtrn2q_f32(r[0], r[1]);
    float32x4_t t2 = vtrn1q_f32(r[2], r[3]), t3 = vtrn2q_f32(r[2], r[3]);
    r[0] = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2)));
    r[1] = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3)));
    r[2] = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2)));
    r[3] = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3)));
}

/* backend/ref's real FFT with four windows in four lanes. Every line is
 * ref's line on vectors, so a lane's power spectrum is ref's bits. The
 * windows come in four samples at a time, transposed so that a vector is
 * one sample of each window, and the spectra go out the same way. */
static void k_stft(float *out, size_t os, const float *const *x, uint32_t count,
                   const pk_fft *f, float *scratch) {
    const uint32_t n = f->n, m = n / 2;
    const float *w = f->window, *fc = f->fcos, *fs = f->fsin;
    float32x4_t *zr = (float32x4_t *)(void *)scratch, *zi = zr + m;
    for (uint32_t w0 = 0; w0 < count; w0 += 4) {
        const float *xs[4];
        for (int i = 0; i < 4; i++) xs[i] = x[w0 + i < count ? w0 + i : count - 1];
        for (uint32_t s = 0; s < n; s += 4) {
            float32x4_t r[4] = { vld1q_f32(xs[0] + s), vld1q_f32(xs[1] + s),
                                 vld1q_f32(xs[2] + s), vld1q_f32(xs[3] + s) };
            transpose4(r);                        /* r[i] is sample s + i of each window */
            zr[f->rev[s / 2]] = vmulq_n_f32(r[0], w[s]);
            zi[f->rev[s / 2]] = vmulq_n_f32(r[1], w[s + 1]);
            zr[f->rev[s / 2 + 1]] = vmulq_n_f32(r[2], w[s + 2]);
            zi[f->rev[s / 2 + 1]] = vmulq_n_f32(r[3], w[s + 3]);
        }
        for (uint32_t len = 2; len <= m; len *= 2) {
            uint32_t half = len / 2, step = n / len;
            for (uint32_t s = 0; s < m; s += len)
                for (uint32_t j = 0; j < half; j++) {
                    float cs = fc[j * step], sn = fs[j * step];
                    uint32_t p = s + j, q = p + half;
                    float32x4_t tr = vaddq_f32(vmulq_n_f32(zr[q], cs), vmulq_n_f32(zi[q], sn));
                    float32x4_t ti = vsubq_f32(vmulq_n_f32(zi[q], cs), vmulq_n_f32(zr[q], sn));
                    zr[q] = vsubq_f32(zr[p], tr);
                    zi[q] = vsubq_f32(zi[p], ti);
                    zr[p] = vaddq_f32(zr[p], tr);
                    zi[p] = vaddq_f32(zi[p], ti);
                }
        }
        /* bins 0 .. m, four at a time; bin m has a block of its own */
        float32x4_t o[4], em = vsubq_f32(zr[0], zi[0]), top = vmulq_f32(em, em);
        for (uint32_t k0 = 0; k0 < m; k0 += 4) {
            for (uint32_t i = 0; i < 4; i++) {
                uint32_t k = k0 + i;
                if (k == 0) {
                    float32x4_t e0 = vaddq_f32(zr[0], zi[0]);
                    o[i] = vmulq_f32(e0, e0);
                    continue;
                }
                float32x4_t ar = zr[k], ai = zi[k], br = zr[m - k], bi = zi[m - k];
                float32x4_t er = vmulq_n_f32(vaddq_f32(ar, br), 0.5f);
                float32x4_t ei = vmulq_n_f32(vsubq_f32(ai, bi), 0.5f);
                float32x4_t odr = vmulq_n_f32(vaddq_f32(ai, bi), 0.5f);
                float32x4_t odi = vmulq_n_f32(vsubq_f32(ar, br), -0.5f);
                float32x4_t re = vaddq_f32(er, vaddq_f32(vmulq_n_f32(odr, fc[k]), vmulq_n_f32(odi, fs[k])));
                float32x4_t im = vaddq_f32(ei, vsubq_f32(vmulq_n_f32(odi, fc[k]), vmulq_n_f32(odr, fs[k])));
                o[i] = vaddq_f32(vmulq_f32(re, re), vmulq_f32(im, im));
            }
            transpose4(o);                        /* o[i] is window i's bins k0 .. k0+3 */
            for (uint32_t i = 0; i < 4 && w0 + i < count; i++) vst1q_f32(out + (w0 + i) * os + k0, o[i]);
        }
        float t[4];
        vst1q_f32(t, top);
        for (uint32_t i = 0; i < 4 && w0 + i < count; i++) out[(w0 + i) * os + m] = t[i];
    }
}

#ifdef PK_NEON_DOTPROD
static const char *const needs[] = { "asimddp", NULL };
const pk_kernels pk_kernels_neon_dp = {
    .id = "neon-dp",
    .name = "neon-dp (ARMv8.2 NEON: SDOT int8, fp32)",
#else
static const char *const needs[] = { NULL };      /* NEON is part of AArch64 */
const pk_kernels pk_kernels_neon = {
    .id = "neon",
    .name = "neon (ARMv8.0 NEON: SMULL/SADALP int8, fp32)",
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
};
