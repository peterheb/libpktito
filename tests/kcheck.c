/* Every kernel of every backend in the build that this CPU can run, against
 * backend/ref, bit for bit.
 *
 * The backends promise ref's bits, not ref's answer to within a tolerance
 * (backend/kernels.h), so this is an equality test: every case runs
 * through the backend's table and ref's, and the outputs are compared with
 * memcmp. The cases cover the shapes the model
 * calls each kernel with and the ones a split over threads makes - a run of
 * panels, a block of channels, a window of rows, every column count a tile
 * can leave over - on values the arithmetic finds awkward: zeros of both
 * signs, subnormals, saturation, the exp clamp, a block of zeros (whose
 * scale is 0), values on a rounding tie, and non-finite quantizer inputs from
 * an earlier operation's overflow.
 *
 * ref is in turn held to the same products in float64 where that is a
 * statement about the definition rather than about float: the int8 product
 * exactly (its sums are integers), the float product to fp32 rounding.
 *
 *     ./tests/kcheck [MODEL.pkf]
 *
 * With a model it also runs the real matrices against real activations.
 */
#include "kernels.h"
#include "model.h"
#include "os.h"
#include "pkf.h"
#include "vmath.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* the kernels under test, and ref's */
static const pk_kernels *kt, *kr;

static const double PI = 3.14159265358979323846;
static int g_bad, g_cases;
static uint64_t g_rng = 0x9E3779B97F4A7C15ull;

static uint32_t rnd(void) {
    g_rng ^= g_rng >> 12;
    g_rng ^= g_rng << 25;
    g_rng ^= g_rng >> 27;
    return (uint32_t)((g_rng * 0x2545F4914F6CDD1Dull) >> 32);
}
static float frand(void) { return (float)(rnd() >> 8) / 8388608.0f - 1.0f; }   /* [-1, 1) */

/* a value the arithmetic finds awkward, now and then */
static float awkward(float scale) {
    switch (rnd() % 24) {
    case 0: return 0.0f;
    case 1: return -0.0f;
    case 2: return 1e-40f;                    /* subnormal */
    case 3: return -scale * 30.0f;
    case 4: return scale * 0.5f;              /* ties, once scaled to 127 */
    default: return scale * frand();
    }
}

/* zeroed and 64-byte aligned, as the engine's buffers are: aligned by hand,
 * since Windows' C runtime has no aligned_alloc that free() releases */
static void *xmalloc(size_t n) {
    unsigned char *raw = calloc(1, n + 64), *p;
    if (!raw) { fprintf(stderr, "kcheck: out of memory\n"); exit(2); }
    p = raw + 64 - ((uintptr_t)raw & 63);
    p[-1] = (unsigned char)(p - raw);
    return p;
}

static void xfree(void *p) {
    if (p)
        free((unsigned char *)p - ((unsigned char *)p)[-1]);
}

static int same(const char *what, const void *a, const void *b, size_t n, size_t esz) {
    g_cases++;
    if (!memcmp(a, b, n * esz))
        return 1;
    for (size_t i = 0; i < n; i++)
        if (memcmp((const char *)a + i * esz, (const char *)b + i * esz, esz)) {
            if (g_bad++ < 30) {
                if (esz == 4)
                    printf("FAIL %s: element %zu: %a, ref %a\n", what, i, ((const float *)a)[i],
                           ((const float *)b)[i]);
                else
                    printf("FAIL %s: element %zu: %d, ref %d\n", what, i, ((const int8_t *)a)[i],
                           ((const int8_t *)b)[i]);
            }
            return 0;
        }
    return 1;
}

/* ------------------------------------------------------- int8 products --- */

static void quant_case(uint32_t T, uint32_t n, uint32_t qb, size_t xs, float scale) {
    float *x = xmalloc(((size_t)T * xs + n) * sizeof *x);
    for (size_t i = 0; i < (size_t)T * xs + n; i++) x[i] = awkward(scale);
    if (rnd() % 3 == 0)                       /* a block of zeros: scale 0 */
        memset(x, 0, qb * sizeof *x);
    pk_act a1 = { xmalloc((size_t)T * n), xmalloc((size_t)T * n / qb * 4), n, qb, T };
    pk_act a2 = { xmalloc((size_t)T * n), xmalloc((size_t)T * n / qb * 4), n, qb, T };
    kt->quantize(&a1, x, xs);
    kr->quantize(&a2, x, xs);
    char what[96];
    snprintf(what, sizeof what, "quantize T %u n %u qb %u: q", T, n, qb);
    same(what, a1.q, a2.q, (size_t)T * n, 1);
    snprintf(what, sizeof what, "quantize T %u n %u qb %u: d", T, n, qb);
    same(what, a1.d, a2.d, (size_t)T * n / qb, 4);
    for (size_t i = 0; i < (size_t)T * n; i++)
        if (a2.q[i] < -127) {
            if (g_bad++ < 30) printf("FAIL ref quantize: -128 at %zu\n", i);
            break;
        }
    xfree(x); xfree(a1.q); xfree(a1.d); xfree(a2.q); xfree(a2.d);
}

static void quant_nonfinite_case(void) {
    const uint32_t T = 2, n = 64, qb = 32;
    float *x = xmalloc((size_t)T * n * sizeof *x);
    for (size_t i = 0; i < (size_t)T * n; i++) x[i] = frand();
    x[3] = NAN;
    x[40] = INFINITY;
    x[n + 5] = -INFINITY;
    x[n + 45] = NAN;
    pk_act a1 = { xmalloc((size_t)T * n), xmalloc((size_t)T * n / qb * sizeof(float)), n, qb, T };
    pk_act a2 = { xmalloc((size_t)T * n), xmalloc((size_t)T * n / qb * sizeof(float)), n, qb, T };
    kt->quantize(&a1, x, n);
    kr->quantize(&a2, x, n);
    same("quantize non-finite: q", a1.q, a2.q, (size_t)T * n, 1);
    same("quantize non-finite: d", a1.d, a2.d, (size_t)T * n / qb, sizeof(float));
    for (size_t i = 0; i < (size_t)T * n; i++)
        if (a1.q[i] != 0) {
            if (g_bad++ < 30) printf("FAIL non-finite quantize: q[%zu] is %d\n", i, a1.q[i]);
            break;
        }
    for (size_t i = 0; i < (size_t)T * n / qb; i++)
        if (a1.d[i] != 0) {
            if (g_bad++ < 30) printf("FAIL non-finite quantize: d[%zu] is %a\n", i, a1.d[i]);
            break;
        }
    xfree(x); xfree(a1.q); xfree(a1.d); xfree(a2.q); xfree(a2.d);
}

/* q8 matrix element (r, k) in the panel layout */
static int8_t *wq_at(int8_t *w, uint32_t n, uint32_t r, uint32_t k) {
    return &w[((size_t)(r / PK_QR) * (n / PK_QG) + k / PK_QG) * (PK_QR * PK_QG)
              + (r % PK_QR) * PK_QG + k % PK_QG];
}

static void gemm_q8_case(const char *what, uint32_t rows, uint32_t n, uint32_t T, uint32_t qb,
                         int add, int bias, const int8_t *wreal, const float *sreal, const float *xreal) {
    int8_t *w = xmalloc((size_t)rows * n);
    float *ws = xmalloc(rows * sizeof *ws), *b = xmalloc(rows * sizeof *b);
    float *x = xmalloc((size_t)T * n * sizeof *x);
    for (uint32_t r = 0; r < rows; r++) {
        if (sreal)
            ws[r] = sreal[r];
        else if (rnd() % 8)
            ws[r] = 0.01f * (frand() + 1.5f);
        else
            ws[r] = 0.0f;
        b[r] = awkward(1.0f);
        for (uint32_t k = 0; k < n; k++)
            *wq_at(w, n, r, k) = wreal ? *wq_at((int8_t *)wreal, n, r, k)
                                       : (int8_t)(rnd() % 3 ? (int)(rnd() % 255) - 127 : 127);
    }
    for (size_t i = 0; i < (size_t)T * n; i++) x[i] = xreal ? xreal[i] : awkward(2.0f);
    pk_act a = { xmalloc((size_t)T * n), xmalloc((size_t)T * n / qb * 4), n, qb, T };
    kr->quantize(&a, x, n);
    size_t os = rows + 16 * (rnd() % 2);
    float *o1 = xmalloc((size_t)T * os * sizeof *o1), *o2 = xmalloc((size_t)T * os * sizeof *o2);
    for (size_t i = 0; i < (size_t)T * os; i++) o1[i] = o2[i] = awkward(1.0f);
    /* a run of panels from the middle, as the pool splits rows */
    uint32_t r0 = 0;
    if (rows > 16 && rnd() % 2)
        r0 = 16 * (rnd() % (rows / 16));
    uint32_t nr = rows - r0;
    kt->gemm_q8(o1 + r0, os, w + (size_t)r0 * n, ws + r0, bias ? b + r0 : NULL, nr, &a, add);
    kr->gemm_q8(o2 + r0, os, w + (size_t)r0 * n, ws + r0, bias ? b + r0 : NULL, nr, &a, add);
    char msg[160];
    snprintf(msg, sizeof msg, "gemm_q8 %s rows %u+%u n %u T %u qb %u add %d bias %d", what, r0, nr,
             n, T, qb, add, bias);
    same(msg, o1, o2, (size_t)T * os, 4);
    /* ref against the definition: the integer sums exact (in int64), each
     * block's product and add one fmaf */
    if (!add && T <= 4)
        for (uint32_t t = 0; t < T; t++)
            for (uint32_t r = r0; r < rows; r++) {
                float acc = 0;
                for (uint32_t blk = 0; blk < n / qb; blk++) {
                    int64_t s = 0;
                    for (uint32_t k = blk * qb; k < (blk + 1) * qb; k++)
                        s += (int64_t)*wq_at(w, n, r, k) * a.q[(size_t)t * n + k];
                    acc = fmaf((float)s, a.d[(size_t)t * (n / qb) + blk], acc);
                }
                float v = acc * ws[r];
                if (bias) v = v + b[r];
                if (memcmp(&v, &o2[t * os + r], 4)) {
                    if (g_bad++ < 30)
                        printf("FAIL ref gemm_q8 %s: column %u row %u: %a, definition %a\n", what,
                               t, r, o2[t * os + r], v);
                    t = T;
                    break;
                }
            }
    xfree(w); xfree(ws); xfree(b); xfree(x); xfree(a.q); xfree(a.d); xfree(o1); xfree(o2);
}

/* ------------------------------------------------------ float products --- */

static void gemm_f32_case(uint32_t rows, uint32_t ld, uint32_t k0, uint32_t n, uint32_t T,
                          size_t xs, int bias_kind, int relu) {
    size_t os = rows + 16 * (rnd() % 2);
    size_t bs = bias_kind == 2 ? os : 0;
    float *wfull = xmalloc((size_t)ld * rows * sizeof *wfull), *w = wfull + 16 * (size_t)k0;
    float *x = xmalloc(((size_t)T * xs + n) * sizeof *x);
    float *b = xmalloc(((size_t)T * os + rows) * sizeof *b);
    float *o1 = xmalloc(((size_t)T * os + rows) * sizeof *o1);
    float *o2 = xmalloc(((size_t)T * os + rows) * sizeof *o2);
    for (size_t i = 0; i < (size_t)ld * rows; i++) wfull[i] = awkward(1.0f);
    for (size_t i = 0; i < (size_t)T * xs + n; i++) x[i] = awkward(1.0f);
    for (size_t i = 0; i < (size_t)T * os + rows; i++) b[i] = o1[i] = o2[i] = awkward(2.0f);
    const float *bb = bias_kind ? b : NULL;
    kt->gemm_f32(o1, os, w, ld, rows, n, x, xs, T, bb, bs, relu);
    kr->gemm_f32(o2, os, w, ld, rows, n, x, xs, T, bb, bs, relu);
    char msg[128];
    snprintf(msg, sizeof msg, "gemm_f32 rows %u ld %u k0 %u n %u T %u bias %d relu %d", rows, ld, k0,
             n, T, bias_kind, relu);
    same(msg, o1, o2, (size_t)T * os, 4);
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t r = 0; r < rows; r++) {
            double acc = bias_kind ? b[t * bs + r] : 0, mag = fabs(acc);
            for (uint32_t k = 0; k < n; k++) {
                double p = (double)w[((size_t)(r / 16) * ld + k) * 16 + r % 16] * x[t * xs + k];
                acc += p;
                mag += fabs(p);
            }
            if (relu && acc < 0) acc = 0;
            double tol = mag * (n + 2) * 0x1p-24 + (n + 2) * 0x1p-149;
            if (fabs(o2[t * os + r] - acc) > tol) {
                if (g_bad++ < 30)
                    printf("FAIL ref gemm_f32: column %u row %u: %.9g, exact %.9g\n", t, r,
                           o2[t * os + r], acc);
                t = T;
                break;
            }
        }
    xfree(wfull); xfree(x); xfree(b); xfree(o1); xfree(o2);
}

/* -------------------------------------------------- rows and reductions --- */

static void rows_cases(void) {
    for (int rep = 0; rep < 40; rep++) {
        uint32_t T = 1 + rnd() % 21;
        uint32_t n = 16 * (1 + rnd() % 80);
        size_t xs = n + 16 * (rnd() % 3);
        float *x = xmalloc((size_t)T * xs * sizeof *x), *w = xmalloc(n * sizeof *w);
        float *b = xmalloc(n * sizeof *b);
        float *o1 = xmalloc((size_t)T * xs * sizeof *o1), *o2 = xmalloc((size_t)T * xs * sizeof *o2);
        float off = rep % 4 == 0 ? 1000.0f : 0.0f;          /* a large mean: the cancellation */
        for (size_t i = 0; i < (size_t)T * xs; i++) x[i] = off + awkward(3.0f);
        for (uint32_t i = 0; i < n; i++) { w[i] = awkward(2.0f); b[i] = awkward(1.0f); }
        kt->layernorm(o1, xs, x, xs, T, n, w, b, 1e-5f);
        kr->layernorm(o2, xs, x, xs, T, n, w, b, 1e-5f);
        for (uint32_t t = 0; t < T; t++)
            same("layernorm", o1 + t * xs, o2 + t * xs, n, 4);
        memcpy(o1, x, (size_t)T * xs * sizeof *x);           /* in place, as the out norm runs */
        memcpy(o2, x, (size_t)T * xs * sizeof *x);
        kt->layernorm(o1, xs, o1, xs, T, n, w, b, 1e-5f);
        kr->layernorm(o2, xs, o2, xs, T, n, w, b, 1e-5f);
        for (uint32_t t = 0; t < T; t++)
            same("layernorm in place", o1 + t * xs, o2 + t * xs, n, 4);

        uint32_t m = 1 + rnd() % 700;                        /* any length, tails and all */
        float *s1 = xmalloc(m * sizeof *s1), *s2 = xmalloc(m * sizeof *s2);
        for (uint32_t i = 0; i < m; i++) s1[i] = s2[i] = awkward(rep % 3 ? 20.0f : 400.0f);
        kt->softmax(s1, m, 0.0883883476f);
        kr->softmax(s2, m, 0.0883883476f);
        same("softmax", s1, s2, m, 4);

        size_t ns = 1 + rnd() % 3000;
        float *u1 = xmalloc(ns * sizeof *u1), *u2 = xmalloc(ns * sizeof *u2);
        for (size_t i = 0; i < ns; i++) u1[i] = u2[i] = awkward(rep % 2 ? 10.0f : 120.0f);
        kt->silu(u1, ns);
        kr->silu(u2, ns);
        same("silu", u1, u2, ns, 4);

        uint32_t gn = 16 * (1 + rnd() % 40);
        float *g = xmalloc((size_t)T * 2 * gn * sizeof *g);
        float *g1 = xmalloc((size_t)T * gn * sizeof *g1), *g2 = xmalloc((size_t)T * gn * sizeof *g2);
        for (size_t i = 0; i < (size_t)T * 2 * gn; i++) g[i] = awkward(rep % 2 ? 8.0f : 100.0f);
        kt->glu(g1, gn, g, 2 * gn, T, gn);
        kr->glu(g2, gn, g, 2 * gn, T, gn);
        same("glu", g1, g2, (size_t)T * gn, 4);

        uint32_t H = 16 * (1 + rnd() % 40);
        float *gt = xmalloc(4 * H * sizeof *gt), *h1 = xmalloc(H * 4), *h2 = xmalloc(H * 4);
        float *c1 = xmalloc(H * 4), *c2 = xmalloc(H * 4);
        for (uint32_t i = 0; i < 4 * H; i++) gt[i] = awkward(rep % 2 ? 4.0f : 60.0f);
        for (uint32_t i = 0; i < H; i++) c1[i] = c2[i] = awkward(3.0f);
        kt->lstm(h1, c1, gt, H);
        kr->lstm(h2, c2, gt, H);
        same("lstm h", h1, h2, H, 4);
        same("lstm c", c1, c2, H, 4);
        xfree(x); xfree(w); xfree(b); xfree(o1); xfree(o2); xfree(s1); xfree(s2); xfree(u1); xfree(u2);
        xfree(g); xfree(g1); xfree(g2); xfree(gt); xfree(h1); xfree(h2); xfree(c1); xfree(c2);
    }
}

static void conv_cases(void) {
    for (int rep = 0; rep < 30; rep++) {
        /* the convolution module's depthwise one, a block of channels; a
         * third of the time past the 64 frames a backend may take at once,
         * a fifth of the time a kernel up to the file's bound */
        uint32_t T = 1 + rnd() % (rep % 3 == 2 ? 200 : 40);
        uint32_t C = 16 * (1 + rnd() % 20);
        uint32_t K = 1 + 2 * (rnd() % (rep % 5 == 4 ? 32 : 6));
        uint32_t c0 = 16 * (rnd() % (C / 16));
        uint32_t nc = 16 * (1 + rnd() % ((C - c0) / 16));
        float *x = xmalloc((size_t)T * C * 4), *w = xmalloc((size_t)C * K * 4), *b = xmalloc(C * 4);
        float *o1 = xmalloc((size_t)T * C * 4), *o2 = xmalloc((size_t)T * C * 4);
        for (size_t i = 0; i < (size_t)T * C; i++) x[i] = awkward(2.0f);
        for (size_t i = 0; i < (size_t)C * K; i++) w[i] = awkward(1.0f);
        for (uint32_t i = 0; i < C; i++) b[i] = awkward(1.0f);
        kt->dwconv(o1 + c0, C, x + c0, C, T, nc, w + (size_t)c0 * K, b + c0, K);
        kr->dwconv(o2 + c0, C, x + c0, C, T, nc, w + (size_t)c0 * K, b + c0, K);
        same("dwconv", o1, o2, (size_t)T * C, 4);
        xfree(x); xfree(w); xfree(b); xfree(o1); xfree(o2);

        /* the subsampling's, from one channel and depthwise, a window of rows */
        uint32_t Ti = 1 + rnd() % 30;
        uint32_t Fi = 1 + rnd() % 70;
        /* one in six past 256 channels, which a backend may take in parts */
        uint32_t Cc = 16 * (1 + rnd() % 4) + (rep % 6 == 5 ? 256 : 0);
        uint32_t To = (Ti + 1) / 2, Fo = (Fi + 1) / 2;
        uint32_t t0 = rnd() % To;
        uint32_t t1 = t0 + 1 + rnd() % (To - t0);
        uint32_t r0 = t0 ? 2 * t0 - 1 : 0;
        for (int dw = 0; dw < 2; dw++) {
            uint32_t cin = dw ? Cc : 1;
            float *in = xmalloc((size_t)Ti * Fi * cin * 4), *wc = xmalloc((size_t)Cc * 9 * 4);
            float *bc = xmalloc(Cc * 4);
            float *p1 = xmalloc((size_t)To * Fo * Cc * 4), *p2 = xmalloc((size_t)To * Fo * Cc * 4);
            for (size_t i = 0; i < (size_t)Ti * Fi * cin; i++) in[i] = awkward(2.0f);
            for (uint32_t i = 0; i < Cc * 9; i++) wc[i] = awkward(1.0f);
            for (uint32_t i = 0; i < Cc; i++) bc[i] = awkward(1.0f);
            int relu = rnd() % 2;
            const float *inw = in + (size_t)r0 * Fi * cin;
            kt->conv3x3s2(p1, inw, r0, Ti, Fi, Cc, dw, wc, bc, relu, t0, t1);
            kr->conv3x3s2(p2, inw, r0, Ti, Fi, Cc, dw, wc, bc, relu, t0, t1);
            same(dw ? "conv3x3s2 depthwise" : "conv3x3s2 one channel", p1, p2,
                 (size_t)(t1 - t0) * Fo * Cc, 4);
            xfree(in); xfree(wc); xfree(bc); xfree(p1); xfree(p2);
        }
    }
}

static void stft_cases(void) {
    for (uint32_t n = 256; n <= 512; n *= 2) {
        uint32_t m = n / 2, bits = 0;
        while ((1u << bits) < m) bits++;
        uint16_t *rev = xmalloc(m * 2);
        float *win = xmalloc(n * 4), *fc = xmalloc(m * 4), *fs = xmalloc(m * 4);
        for (uint32_t j = 0; j < m; j++) {
            uint32_t r = 0;
            for (uint32_t k = 0; k < bits; k++) r |= ((j >> k) & 1u) << (bits - 1 - k);
            rev[j] = (uint16_t)r;
            fc[j] = (float)cos(2 * PI * j / n);
            fs[j] = (float)sin(2 * PI * j / n);
        }
        for (uint32_t i = 0; i < n; i++) win[i] = (float)(0.5 - 0.5 * cos(2 * PI * i / n));
        pk_fft f = { n, win, fc, fs, rev };
        uint32_t count = 1 + rnd() % 37;
        float *sig = xmalloc(((size_t)count * 160 + n) * 4);
        for (size_t i = 0; i < (size_t)count * 160 + n; i++) sig[i] = awkward(1.0f);
        const float **xs = xmalloc(count * sizeof *xs);
        for (uint32_t w = 0; w < count; w++) xs[w] = sig + (size_t)w * 160;
        float *o1 = xmalloc((size_t)count * (m + 1) * 4), *o2 = xmalloc((size_t)count * (m + 1) * 4);
        float *s1 = xmalloc(PK_K_STFT_SCRATCH(n) * 4), *s2 = xmalloc(PK_K_STFT_SCRATCH(n) * 4);
        kt->stft(o1, m + 1, xs, count, &f, s1);
        kr->stft(o2, m + 1, xs, count, &f, s2);
        same("stft", o1, o2, (size_t)count * (m + 1), 4);
        xfree(rev); xfree(win); xfree(fc); xfree(fs); xfree(sig); xfree(xs); xfree(o1); xfree(o2);
        xfree(s1); xfree(s2);
    }
}

/* ----------------------------------------------------------- the model --- */

static void model_cases(const char *path) {
    char err[256];
    pk_model *m = pk_model_load(path, err, sizeof err);
    if (!m) {
        printf("kcheck: %s\n", err);
        g_bad++;
        return;
    }
    const pk_layer *L = &m->layers[m->n_layers / 2];
    const pk_q8 *mats[] = { &L->ff1_up, &L->ff1_down, &L->q, &L->o, &L->pw1, &m->jout, &m->lstm[0].wx };
    for (size_t i = 0; i < sizeof mats / sizeof mats[0]; i++) {
        const pk_q8 *q = mats[i];
        uint32_t T = i < 5 ? 1 + rnd() % 13 : 1;
        float *x = xmalloc((size_t)T * q->cols * 4);
        for (size_t k = 0; k < (size_t)T * q->cols; k++) x[k] = awkward(3.0f);
        gemm_q8_case("real", q->rows, q->cols, T, i < 5 ? 32 : q->cols, i % 2, i % 3 == 0, q->q, q->s, x);
        xfree(x);
    }
    pk_model_free(m);
}

static void run(const char *model) {
    /* each rnd() in a statement of its own: the order a compiler evaluates
     * a call's arguments in is its own, and the cases should be the same
     * on every compiler */
    quant_nonfinite_case();
    for (int rep = 0; rep < 60; rep++) {
        uint32_t n = 32 * (1 + rnd() % 40);
        uint32_t T = 1 + rnd() % 20;
        uint32_t qb = rnd() % 2 ? 32 : n;
        size_t xs = n + 32 * (rnd() % 2);
        quant_case(T, n, qb, xs, rep % 3 ? 1.0f : 1e4f);
    }
    static const uint32_t ns[] = { 32, 64, 96, 256, 640, 1024, 4096 };
    for (int rep = 0; rep < 120; rep++) {
        uint32_t n = ns[rnd() % 7];
        uint32_t rows = 16 * (1 + rnd() % 24);
        uint32_t T = rep % 4 == 0 ? 1 : 1 + rnd() % 37;
        if (n == 4096 && rows > 64) rows = 64;
        uint32_t qb = rnd() % 3 ? 32 : n;
        int add = rnd() % 2;
        int bias = rnd() % 2;
        gemm_q8_case("random", rows, n, T, qb, add, bias, NULL, NULL, NULL);
    }
    for (int rep = 0; rep < 80; rep++) {
        uint32_t rows = 16 * (1 + rnd() % 12);
        uint32_t ld = 1 + rnd() % 200;
        uint32_t k0 = rnd() % ld;
        uint32_t n = 1 + rnd() % (ld - k0);
        uint32_t T = 1 + rnd() % 23;
        size_t xs = n + rnd() % 9;
        int bias = rnd() % 3;
        int relu = rnd() % 2;
        gemm_f32_case(rows, ld, k0, n, T, xs, bias, relu);
    }
    rows_cases();
    conv_cases();
    stft_cases();
    if (model)
        model_cases(model);
}

/* Each backend sees the same cases: the generator starts again for each.
 * A build of ref alone holds ref to its float64 definitions, which every
 * run does on the way. */
int main(int argc, char **argv) {
    kr = &pk_kernels_ref;
    int bad = 0, ran = 0;
    for (const pk_kernels *const *k = pk_kernels_built; *k; k++) {
        if (*k == kr && ran)
            break;
        const char *lack = pk_os_cpu_lacks((*k)->needs);
        if (lack) {
            printf("kcheck: %s: this CPU has no %s - skipped\n", (*k)->name, lack);
            continue;
        }
        kt = *k;
        g_rng = 0x9E3779B97F4A7C15ull;
        g_bad = g_cases = 0;
        run(argc > 1 ? argv[1] : NULL);
        printf("kcheck: %s, %d cases: %s\n", kt->name, g_cases, g_bad ? "FAIL" : "every one ref's bits");
        bad += g_bad;
        ran++;
    }
    return bad != 0;
}
