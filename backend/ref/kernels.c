/* The reference backend: every kernel in plain scalar C.
 *
 * This is the definition of correct. It builds with any C11 compiler on any
 * host and computes the same bits as a tuned backend, only slower; every
 * other backend is held to it bit for bit (tests/kcheck). The loops are
 * backend/kernels.h's contracts written out, in the order a SIMD backend
 * runs them: outputs innermost wherever an output is a chain, so that the
 * inner loop is what one vector of lanes does in one instruction.
 *
 * Built for x86 it has no -mfma (it is the one that runs anywhere), so each
 * fmaf is a call into libm. That is the price of being the definition.
 */
#include "kernels.h"
#include "pkf.h"
#include "vmath.h"

#include <math.h>
#include <string.h>

/* ------------------------------------------------------- int8 products -- */

void pk_k_quantize(pk_act *a, const float *x, size_t xs) {
    const uint32_t nb = a->n / a->qb;
    for (uint32_t t = 0; t < a->T; t++)
        for (uint32_t b = 0; b < nb; b++) {
            const float *v = x + t * xs + (size_t)b * a->qb;
            int8_t *q = a->q + (size_t)t * a->n + (size_t)b * a->qb;
            float amax = 0;
            for (uint32_t i = 0; i < a->qb; i++)
                amax = fmaxf(amax, fabsf(v[i]));
            a->d[(size_t)t * nb + b] = amax / 127.0f;
            float id = amax > 0 ? 127.0f / amax : 0.0f;
            for (uint32_t i = 0; i < a->qb; i++)
                q[i] = (int8_t)rintf(v[i] * id);
        }
}

/* q(r, k) of a PK_Q8 matrix (core/pkf.h) with n inputs */
static inline int32_t wq(const int8_t *w, uint32_t n, uint32_t r, uint32_t k) {
    return w[((size_t)(r / PK_QR) * (n / PK_QG) + k / PK_QG) * (PK_QR * PK_QG)
             + (r % PK_QR) * PK_QG + k % PK_QG];
}

void pk_k_gemm_q8(float *out, size_t os, const int8_t *w, const float *ws, const float *bias,
                  uint32_t rows, const pk_act *a, int add) {
    const uint32_t n = a->n, qb = a->qb, nb = n / qb;
    for (uint32_t t = 0; t < a->T; t++) {
        const int8_t *q = a->q + (size_t)t * n;
        const float *d = a->d + (size_t)t * nb;
        for (uint32_t r = 0; r < rows; r++) {
            float acc = 0;
            for (uint32_t b = 0; b < nb; b++) {
                int32_t s = 0;
                for (uint32_t k = b * qb; k < (b + 1) * qb; k++)
                    s += wq(w, n, r, k) * q[k];
                acc = fmaf((float)s, d[b], acc);
            }
            float v = acc * ws[r];
            if (bias)
                v = v + bias[r];
            out[t * os + r] = add ? out[t * os + r] + v : v;
        }
    }
}

/* ------------------------------------------------------ float products -- */

void pk_k_gemm_f32(float *out, size_t os, const float *w, uint32_t ld, uint32_t rows, uint32_t n,
                   const float *x, size_t xs, uint32_t T,
                   const float *bias, size_t bs, int relu) {
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t p = 0; p < rows / 16; p++) {
            /* a panel's 16 outputs, each one chain over k */
            const float *wp = w + (size_t)p * ld * 16, *xt = x + t * xs;
            float acc[16] = { 0 };
            for (uint32_t k = 0; k < n; k++)
                for (int i = 0; i < 16; i++) acc[i] = fmaf(wp[16 * k + i], xt[k], acc[i]);
            for (int i = 0; i < 16; i++) {
                float v = acc[i] + (bias ? bias[t * bs + 16 * p + i] : 0.0f);
                out[t * os + 16 * p + i] = relu ? (v > 0 ? v : 0) : v;
            }
        }
}

/* ------------------------------------------------- reductions and rows -- */

static float lanes_sum(float *acc) {
    for (int h = PK_LANES / 2; h > 0; h /= 2)
        for (int i = 0; i < h; i++) acc[i] += acc[i + h];
    return acc[0];
}

void pk_k_layernorm(float *out, size_t os, const float *x, size_t xs, uint32_t T, uint32_t n,
                    const float *w, const float *b, float eps) {
    for (uint32_t t = 0; t < T; t++) {
        const float *v = x + t * xs;
        float *o = out + t * os;
        float acc[PK_LANES] = { 0 };
        for (uint32_t i = 0; i < n; i++) acc[i % PK_LANES] += v[i];
        float mean = lanes_sum(acc) / (float)n;
        memset(acc, 0, sizeof acc);
        for (uint32_t i = 0; i < n; i++) {
            float c = v[i] - mean;
            acc[i % PK_LANES] += c * c;
        }
        float var = lanes_sum(acc) / (float)n;
        float r = 1.0f / sqrtf(var + eps);
        for (uint32_t i = 0; i < n; i++) o[i] = (v[i] - mean) * r * w[i] + b[i];
    }
}

void pk_k_softmax(float *x, uint32_t n, float scale) {
    float m = -INFINITY;
    for (uint32_t i = 0; i < n; i++) {
        x[i] = x[i] * scale;
        m = fmaxf(m, x[i]);
    }
    float acc[PK_LANES] = { 0 };
    for (uint32_t i = 0; i < n; i++) {
        x[i] = pk_expf(x[i] - m);
        acc[i % PK_LANES] += x[i];
    }
    float inv = 1.0f / lanes_sum(acc);
    for (uint32_t i = 0; i < n; i++) x[i] = x[i] * inv;
}

void pk_k_silu(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) x[i] = x[i] * pk_sigmoidf(x[i]);
}

void pk_k_glu(float *out, size_t os, const float *x, size_t xs, uint32_t T, uint32_t n) {
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t i = 0; i < n; i++)
            out[t * os + i] = x[t * xs + i] * pk_sigmoidf(x[t * xs + n + i]);
}

void pk_k_dwconv(float *out, size_t os, const float *x, size_t xs, uint32_t T, uint32_t C,
                 const float *w, const float *b, uint32_t K) {
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t c = 0; c < C; c++) {
            float acc = 0;
            for (uint32_t j = 0; j < K; j++) {
                int64_t s = (int64_t)t + j - K / 2;
                if (s >= 0 && s < T)
                    acc = fmaf(w[(size_t)c * K + j], x[(size_t)s * xs + c], acc);
            }
            float v = acc + b[c];
            out[(size_t)t * os + c] = v * pk_sigmoidf(v);
        }
}

void pk_k_conv3x3s2(float *out, const float *in, uint32_t r0, uint32_t Ti, uint32_t Fi,
                    uint32_t C, int depthwise, const float *w, const float *b, int relu,
                    uint32_t t0, uint32_t t1) {
    const uint32_t Fo = (Fi + 1) / 2, cin = depthwise ? C : 1;
    for (uint32_t t = t0; t < t1; t++)
        for (uint32_t f = 0; f < Fo; f++)
            for (uint32_t c = 0; c < C; c++) {
                float acc = 0;
                for (int di = 0; di < 3; di++) {
                    int64_t ti = 2 * (int64_t)t + di - 1;
                    if (ti < 0 || ti >= Ti)
                        continue;
                    for (int dj = 0; dj < 3; dj++) {
                        int64_t fi = 2 * (int64_t)f + dj - 1;
                        if (fi < 0 || fi >= Fi)
                            continue;
                        float v = in[((size_t)(ti - r0) * Fi + (size_t)fi) * cin + (depthwise ? c : 0)];
                        acc = fmaf(w[(size_t)c * 9 + 3 * di + dj], v, acc);
                    }
                }
                float v = acc + b[c];
                out[((size_t)(t - t0) * Fo + f) * C + c] = relu ? (v > 0 ? v : 0) : v;
            }
}

void pk_k_lstm(float *h, float *c, const float *g, uint32_t H) {
    for (uint32_t j = 0; j < H; j++) {
        float i = pk_sigmoidf(g[j]);
        float f = pk_sigmoidf(g[H + j]);
        float u = pk_tanhf(g[2 * H + j]);
        float o = pk_sigmoidf(g[3 * H + j]);
        c[j] = f * c[j] + i * u;
        h[j] = o * pk_tanhf(c[j]);
    }
}

/* --------------------------------------------------------------- STFT -- */

/* |X[k]|^2 for k = 0 .. n/2 of the windowed n points at x, by the textbook
 * real-FFT trick: the n reals are packed as n/2 complex points z[j] =
 * x[2j] + i x[2j+1], their n/2-point FFT Z is taken, and the spectra of the
 * even and odd samples are pulled back apart:
 *
 *   E[k] = (Z[k] + conj Z[n/2-k]) / 2,   O[k] = -i (Z[k] - conj Z[n/2-k]) / 2
 *   X[k] = E[k] + e^(-2 pi i k / n) O[k]
 *
 * The FFT is radix-2, decimation in time: the input goes in at its index's
 * bit reversal, then log2(n/2) passes of butterflies. Every twiddle is a
 * power of e^(-2 pi i / n), which the file holds as fcos and fsin. This is
 * vadito's FFT, its magnitudes squared. */
static void stft_one(float *out, const float *x, const pk_fft *f, float *zr, float *zi) {
    const uint32_t n = f->n, m = n / 2;
    const float *w = f->window, *fc = f->fcos, *fs = f->fsin;
    for (uint32_t j = 0; j < m; j++) {
        zr[f->rev[j]] = w[2 * j] * x[2 * j];
        zi[f->rev[j]] = w[2 * j + 1] * x[2 * j + 1];
    }
    for (uint32_t len = 2; len <= m; len *= 2) {
        uint32_t half = len / 2, step = n / len;      /* W_len^j is W_n^(j step) */
        for (uint32_t s = 0; s < m; s += len)
            for (uint32_t j = 0; j < half; j++) {
                float cs = fc[j * step], sn = fs[j * step];
                uint32_t p = s + j, q = p + half;
                float tr = cs * zr[q] + sn * zi[q];   /* (cs - i sn) z[q] */
                float ti = cs * zi[q] - sn * zr[q];
                zr[q] = zr[p] - tr;
                zi[q] = zi[p] - ti;
                zr[p] = zr[p] + tr;
                zi[p] = zi[p] + ti;
            }
    }
    /* k = 0 and k = n/2 are real: the sums of the even and odd samples,
     * added and subtracted */
    float e0 = zr[0] + zi[0], em = zr[0] - zi[0];
    out[0] = e0 * e0;
    out[m] = em * em;
    for (uint32_t k = 1; k < m; k++) {
        float ar = zr[k], ai = zi[k], br = zr[m - k], bi = zi[m - k];
        float er = 0.5f * (ar + br), ei = 0.5f * (ai - bi);     /* E[k] */
        float odr = 0.5f * (ai + bi), odi = -0.5f * (ar - br);  /* O[k] */
        float cs = fc[k], sn = fs[k];
        float re = er + (cs * odr + sn * odi);
        float im = ei + (cs * odi - sn * odr);
        out[k] = re * re + im * im;
    }
}

void pk_k_stft(float *out, size_t os, const float *const *x, uint32_t count,
               const pk_fft *f, float *scratch) {
    for (uint32_t w = 0; w < count; w++)
        stft_one(out + w * os, x[w], f, scratch, scratch + f->n / 2);
}

const char *pk_k_name(void) { return "ref (scalar C)"; }
