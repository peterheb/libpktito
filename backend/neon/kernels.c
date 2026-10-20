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

void pk_k_quantize(pk_act *a, const float *x, size_t xs) {
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
 * a pairwise add joins at the block's end. */
static inline __attribute__((always_inline))
void q8_tile(float *out, size_t os, const int8_t *w, uint32_t n, uint32_t h,
             const float *ws, const float *bias, const pk_act *a, uint32_t t0, int add,
             const int NC) {
    const uint32_t qb = a->qb, nb = n / qb;
    float32x4_t f[4][2];
    for (int c = 0; c < NC; c++) f[c][0] = f[c][1] = vdupq_n_f32(0.0f);
    const int8_t *x[4];
    for (int c = 0; c < NC; c++) x[c] = a->q + (size_t)(t0 + c) * n;
    for (uint32_t b = 0; b < nb; b++) {
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
        for (int c = 0; c < NC; c++) s[c][0] = s[c][1] = s[c][2] = s[c][3] = vdupq_n_s32(0);
        for (uint32_t k = b * qb; k < (b + 1) * qb; k += 8) {
            const int8_t *g = w + (size_t)(k / 4) * 64 + 16 * h;
            int8x16_t wa = vld1q_s8(g), wb = vld1q_s8(g + 16);          /* group k/4 */
            int8x16_t wc = vld1q_s8(g + 64), wd = vld1q_s8(g + 80);     /* group k/4 + 1 */
            for (int c = 0; c < NC; c++) {
                int32_t u0, u1;
                memcpy(&u0, x[c] + k, 4);
                memcpy(&u1, x[c] + k + 4, 4);
                int8x8_t x0 = vreinterpret_s8_s32(vdup_n_s32(u0));
                int8x8_t x1 = vreinterpret_s8_s32(vdup_n_s32(u1));
                int16x8_t p0 = vmull_s8(vget_low_s8(wa), x0);
                int16x8_t p1 = vmull_s8(vget_high_s8(wa), x0);
                int16x8_t p2 = vmull_s8(vget_low_s8(wb), x0);
                int16x8_t p3 = vmull_s8(vget_high_s8(wb), x0);
                p0 = vmlal_s8(p0, vget_low_s8(wc), x1);
                p1 = vmlal_s8(p1, vget_high_s8(wc), x1);
                p2 = vmlal_s8(p2, vget_low_s8(wd), x1);
                p3 = vmlal_s8(p3, vget_high_s8(wd), x1);
                s[c][0] = vpadalq_s16(s[c][0], p0);
                s[c][1] = vpadalq_s16(s[c][1], p1);
                s[c][2] = vpadalq_s16(s[c][2], p2);
                s[c][3] = vpadalq_s16(s[c][3], p3);
            }
        }
        for (int c = 0; c < NC; c++) {
            float32x4_t d = vdupq_n_f32(a->d[(size_t)(t0 + c) * nb + b]);
            int32x4_t r0 = vpaddq_s32(s[c][0], s[c][1]), r1 = vpaddq_s32(s[c][2], s[c][3]);
            f[c][0] = vfmaq_f32(f[c][0], vcvtq_f32_s32(r0), d);
            f[c][1] = vfmaq_f32(f[c][1], vcvtq_f32_s32(r1), d);
        }
#endif
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

/* Panel by panel, each in its two halves of 8 rows, four columns at a time
 * across the half: a half panel's weights are read from memory once and
 * from cache for every other column. */
void pk_k_gemm_q8(float *out, size_t os, const int8_t *w, const float *ws, const float *bias,
                  uint32_t rows, const pk_act *a, int add) {
    const uint32_t n = a->n;
    for (uint32_t r = 0; r < rows; r += 16) {
        const int8_t *wp = w + (size_t)r * n;
        float *o = out + r;
        const float *b = bias ? bias + r : NULL;
        for (uint32_t h = 0; h < 4; h += 2) {
            uint32_t t = 0;
            for (; t + 4 <= a->T; t += 4) q8_tile(o, os, wp, n, h, ws + r, b, a, t, add, 4);
            switch (a->T - t) {
            case 3: q8_tile(o, os, wp, n, h, ws + r, b, a, t, add, 3); break;
            case 2: q8_tile(o, os, wp, n, h, ws + r, b, a, t, add, 2); break;
            case 1: q8_tile(o, os, wp, n, h, ws + r, b, a, t, add, 1); break;
            default: break;
            }
        }
    }
}

/* ------------------------------------------------------ float products --- */

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

void pk_k_gemm_f32(float *out, size_t os, const float *w, uint32_t ld, uint32_t rows, uint32_t n,
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

/* ------------------------------------------------- reductions and rows --- */

void pk_k_layernorm(float *out, size_t os, const float *x, size_t xs, uint32_t T, uint32_t n,
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

void pk_k_softmax(float *x, uint32_t n, float scale) {
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

void pk_k_silu(float *x, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(x + i, v_silu(vld1q_f32(x + i)));
    for (; i < n; i++) x[i] = x[i] * pk_sigmoidf(x[i]);
}

void pk_k_glu(float *out, size_t os, const float *x, size_t xs, uint32_t T, uint32_t n) {
    for (uint32_t t = 0; t < T; t++) {
        const float *a = x + t * xs;
        float *o = out + t * os;
        for (uint32_t i = 0; i < n; i += 4)
            vst1q_f32(o + i, vmulq_f32(vld1q_f32(a + i), v_sigmoid(vld1q_f32(a + n + i))));
    }
}

void pk_k_dwconv(float *out, size_t os, const float *x, size_t xs, uint32_t T, uint32_t C,
                 const float *w, const float *b, uint32_t K) {
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t c = 0; c < C; c += 4) {
            /* lane j is channel c + j; its taps are K apart in w */
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (uint32_t j = 0; j < K; j++) {
                int64_t s = (int64_t)t + j - K / 2;
                if (s < 0 || s >= T)
                    continue;
                float wt[4] = { w[(size_t)c * K + j], w[(size_t)(c + 1) * K + j],
                                w[(size_t)(c + 2) * K + j], w[(size_t)(c + 3) * K + j] };
                acc = vfmaq_f32(acc, vld1q_f32(wt), vld1q_f32(x + (size_t)s * xs + c));
            }
            vst1q_f32(out + (size_t)t * os + c, v_silu(vaddq_f32(acc, vld1q_f32(b + c))));
        }
}

void pk_k_conv3x3s2(float *out, const float *in, uint32_t r0, uint32_t Ti, uint32_t Fi,
                    uint32_t C, int depthwise, const float *w, const float *b, int relu,
                    uint32_t t0, uint32_t t1) {
    const uint32_t Fo = (Fi + 1) / 2, cin = depthwise ? C : 1;
    const float32x4_t zero = vdupq_n_f32(0.0f);
    for (uint32_t c = 0; c < C; c += 4) {
        /* the four channels' nine taps, tap-major */
        float32x4_t wt[9];
        for (int k = 0; k < 9; k++) {
            float v[4] = { w[(size_t)c * 9 + k], w[(size_t)(c + 1) * 9 + k],
                           w[(size_t)(c + 2) * 9 + k], w[(size_t)(c + 3) * 9 + k] };
            wt[k] = vld1q_f32(v);
        }
        const float32x4_t bv = vld1q_f32(b + c);
        for (uint32_t t = t0; t < t1; t++)
            for (uint32_t f = 0; f < Fo; f++) {
                float32x4_t acc = zero;
                for (int di = 0; di < 3; di++) {
                    int64_t ti = 2 * (int64_t)t + di - 1;
                    if (ti < 0 || ti >= Ti)
                        continue;
                    for (int dj = 0; dj < 3; dj++) {
                        int64_t fi = 2 * (int64_t)f + dj - 1;
                        if (fi < 0 || fi >= Fi)
                            continue;
                        const float *p = in + ((size_t)(ti - r0) * Fi + (size_t)fi) * cin;
                        float32x4_t v = depthwise ? vld1q_f32(p + c) : vdupq_n_f32(p[0]);
                        acc = vfmaq_f32(acc, wt[3 * di + dj], v);
                    }
                }
                float32x4_t v = vaddq_f32(acc, bv);
                if (relu) v = vbslq_f32(vcgtq_f32(v, zero), v, zero);
                vst1q_f32(out + ((size_t)(t - t0) * Fo + f) * C + c, v);
            }
    }
}

void pk_k_lstm(float *h, float *c, const float *g, uint32_t H) {
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
void pk_k_stft(float *out, size_t os, const float *const *x, uint32_t count,
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
const char *pk_k_name(void) { return "neon-dp (ARMv8.2 NEON: SDOT int8, fp32)"; }
#else
const char *pk_k_name(void) { return "neon (ARMv8.0 NEON: SMULL/SADALP int8, fp32)"; }
#endif
