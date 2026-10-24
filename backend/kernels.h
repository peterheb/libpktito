/* The backend seam: every operation on the weights, and the elementwise
 * ones whose cost is an exponential. A build holds every backend its
 * compiler can make for the machine's family - neon-dp and neon on
 * AArch64; avx2-vnni, avx512-vnni and avx2 on x86-64 - and ref, and the
 * first of them whose instructions the CPU has is the one that runs
 * (backend/kernels.c): one binary for a family, whatever its CPU.
 *
 * backend/ref/ implements these in plain C and is the definition of
 * correct. Every other backend computes the same bits, not merely close
 * ones, which tests/kcheck holds them to on random data and the real
 * weights. That is what lets a transcript be the same on a Pi 3 and on a
 * desktop, at one thread and at twenty, and what lets `make check` compare
 * a transcript's bits rather than its words.
 *
 * Two shapes make the promise cheap. The int8 product's inner sums are
 * integers, which add up the same in any order, so a backend may take
 * them with whatever its CPU multiplies int8 with (SDOT, SMULL/SADALP,
 * VPMADDUBSW, VPDPBUSD) in whatever order; only the few float operations
 * per block are fixed. And every float sum is either one chain of fused
 * multiply-adds per output, with SIMD lanes over outputs (the lanes then run
 * the scalar chain exactly, whatever the vector width or tiling), or a
 * reduction in sixteen lanes combined in a fixed tree (PK_LANES below),
 * which a backend reproduces with four NEON vectors or two AVX ones.
 */
#ifndef PK_KERNELS_H
#define PK_KERNELS_H

#include <stddef.h>
#include <stdint.h>

/* ------------------------------------------------------- int8 products --
 *
 * The encoder's matrices are int8 (core/pkf.h, PK_Q8: per-row scales, in
 * panels of 16 rows and groups of 4 inputs) and so are the activations
 * they multiply, quantized per column in blocks of qb inputs, each block
 * with its own scale. qb is a whole row (one scale for the column) where the
 * activations allow it and 32 where they have outliers that a row's scale
 * would crush: README, "Why int8", has the measurement for each matrix. */

typedef struct {
    int8_t   *q;            /* [T][n], column t at q + t * n */
    float    *d;            /* [T][n / qb], each block's scale */
    uint32_t  n, qb, T;
} pk_act;

/* Quantize T columns of n inputs (column t at x + t * xs) in blocks of qb,
 * qb dividing n:
 *
 *   amax = max |x| over the block
 *   d    = amax / 127                       (a float division)
 *   q    = rint(x * (127 / amax)), or 0 when amax is 0
 *
 * so q is in [-127, 127] (never -128: AVX2's sign trick, backend/avx2,
 * needs that) and x is about q * d. If arithmetic before this overflowed,
 * a block containing a NaN or infinity becomes q = 0, d = 0; no non-finite
 * value is narrowed to an integer. */
void pk_k_quantize(pk_act *a, const float *x, size_t xs);

/* out[t * os + r] for r < rows and every column t of a:
 *
 *   acc = 0
 *   for each block b of a, in order:
 *       s   = sum of w(r, k) * q(t, k) over the block's k    (an exact int32)
 *       acc = fmaf((float)s, d(t, b), acc)
 *   v = acc * ws[r]
 *   if bias: v = v + bias[r]
 *   out = add ? out + v : v
 *
 * w is the PK_Q8 matrix (rows a multiple of 16, a->n inputs, a multiple of
 * 32) and ws its scales; a call may take a run of whole panels, w, ws and
 * bias pointing at its first row. With add, out holds a residual stream
 * that the product is added to (a feed-forward's second matrix, the
 * attention's output, the convolution module's last). An integer sum is at
 * most 127^2 n, inside an int32 for n below 133,000; Parakeet's widest is
 * 4096. */
void pk_k_gemm_q8(float *out, size_t os, const int8_t *w, const float *ws, const float *bias,
                  uint32_t rows, const pk_act *a, int add);

/* The most blocks of activations a product may have (a->n / a->qb): a
 * backend may keep a value a block on its stack. 512 blocks of 32 is an
 * input of 16384, which core/model.c holds every matrix to. */
#define PK_MAX_BLOCKS 512

/* ------------------------------------------------------ float products --
 *
 * For t < T and r < rows:
 *
 *   acc = 0
 *   for k = 0 .. n-1:  acc = fmaf(W(k, r), x[t*xs + k], acc)
 *   v = acc + (bias ? bias[t*bs + r] : 0)
 *   out[t*os + r] = relu ? (v > 0 ? v : 0) : v
 *
 * with W(k, r) = w[((r / 16) * ld + k) * 16 + r % 16]: the matrix is k-major
 * in panels of 16 rows, ld k to a panel, and w points at the first k of the
 * run this call takes. rows is a multiple of 16. bs is 0 for one bias for
 * every column. out may be bias itself (os == bs); no other argument may
 * overlap out. This is vadito's matrix kernel, and Silero's VAD runs on it
 * as it does there; here it also takes the attention's products (core/
 * encoder.c lays its operands out as panels) and the decoder's. */
void pk_k_gemm_f32(float *out, size_t os, const float *w, uint32_t ld, uint32_t rows, uint32_t n,
                   const float *x, size_t xs, uint32_t T,
                   const float *bias, size_t bs, int relu);

/* ------------------------------------------------- reductions and rows --
 *
 * A float sum over a row is taken in PK_LANES lanes, element i into lane
 * i % 16 in order, and the lanes combined as acc[i] += acc[i + h] for h =
 * 8, 4, 2, 1, leaving the sum in acc[0]. */
#define PK_LANES 16

/* T rows of n (a multiple of 16), row t at x + t * xs, normalized into
 * out + t * os (which may be x):
 *
 *   mean = sum(x) / n
 *   var  = sum((x - mean)^2) / n
 *   r    = 1 / sqrtf(var + eps)
 *   out  = (x - mean) * r * w + b         (in that order, left to right)
 *
 * with both sums in the lanes above: torch's LayerNorm, to fp32 rounding. */
void pk_k_layernorm(float *out, size_t os, const float *x, size_t xs, uint32_t T, uint32_t n,
                    const float *w, const float *b, float eps);

/* A row of attention scores, scaled and softmaxed in place:
 *
 *   y   = x * scale
 *   m   = max(y)
 *   e   = pk_expf(y - m)
 *   x   = e * (1 / sum(e))                 the sum in the lanes above */
void pk_k_softmax(float *x, uint32_t n, float scale);

/* x[i] = x[i] * pk_sigmoidf(x[i]): SiLU, in place over n. */
void pk_k_silu(float *x, size_t n);

/* GLU over T rows: out[t*os + i] = a[i] * pk_sigmoidf(a[i + n]) for i < n,
 * with a = x + t * xs: the first half of each row gated by its second. */
void pk_k_glu(float *out, size_t os, const float *x, size_t xs, uint32_t T, uint32_t n);

/* The convolution module's depthwise convolution over T rows of C
 * channels (row t of x at x + t*xs, of out at out + t*os; C a multiple of
 * 16), K taps centred, then its bias and SiLU:
 *
 *   acc = 0
 *   for j = 0 .. K-1 with 0 <= t + j - K/2 < T:
 *       acc = fmaf(w[c*K + j], x[(t + j - K/2)*xs + c], acc)
 *   v = acc + b[c]
 *   out[t*os + c] = v * pk_sigmoidf(v)
 *
 * the taps that would read padding skipped, the BatchNorm folded into w
 * and b by the converter. A call may take a block of the channels, x, out,
 * w and b each pointing at its first. out does not overlap x. */
void pk_k_dwconv(float *out, size_t os, const float *x, size_t xs, uint32_t T, uint32_t C,
                 const float *w, const float *b, uint32_t K);

/* The subsampling's 3x3 convolutions, stride 2, padding 1, over a
 * channel-last image of Ti rows by Fi columns: C channels (depthwise), or
 * one (depthwise 0, the first convolution, every output channel reading the
 * same input). Output rows [t0, t1) of the To = (Ti+1)/2 by Fo = (Fi+1)/2
 * result:
 *
 *   acc = 0
 *   for di, dj in 0..2 (di outer), taps inside the image only:
 *       acc = fmaf(w[c*9 + 3*di + dj], in(2t+di-1, 2f+dj-1, c or 0), acc)
 *   out = relu ? max(acc + b[c], 0) : acc + b[c]
 *
 * `in` holds the image from row r0 on (the subsampling runs a block of
 * rows at a time, core/encoder.c), every row a tap inside the image needs;
 * `out` is output row t0. C is a multiple of 16. */
void pk_k_conv3x3s2(float *out, const float *in, uint32_t r0, uint32_t Ti, uint32_t Fi,
                    uint32_t C, int depthwise, const float *w, const float *b, int relu,
                    uint32_t t0, uint32_t t1);

/* One LSTM step for H units (H a multiple of 16), gates in PyTorch's order:
 * g[0, H) input, [H, 2H) forget, [2H, 3H) candidate, [3H, 4H) output.
 *
 *   c[j] = sigmoid(g[H+j]) * c[j] + sigmoid(g[j]) * tanh(g[2H+j])
 *   h[j] = sigmoid(g[3H+j]) * tanh(c[j])
 *
 * with core/vmath.h's sigmoid and tanh, each product rounded on its own and
 * then summed (vadito's cell, which Silero's VAD and Parakeet's
 * prediction network both run). */
void pk_k_lstm(float *h, float *c, const float *g, uint32_t H);

/* --------------------------------------------------------------- STFT --
 *
 * What the STFT kernel needs of a model's FFT: n points, the analysis
 * window (n of them, zero where a shorter window was padded), the twiddles
 * cos and sin of 2 pi k / n for k < n/2, and the bit reversal of the n/2
 * complex points the real FFT packs them as. */
typedef struct {
    uint32_t n;
    const float *window, *fcos, *fsin;
    const uint16_t *rev;
} pk_fft;

/* Floats of scratch pk_k_stft may use at n points: two arrays of n/2
 * complex points, eight lanes wide, 64-byte aligned (a backend may address
 * it as vectors). */
#define PK_K_STFT_SCRATCH(n) (8 * (size_t)(n))

/* The power spectrum: window w is the n samples at x[w], times the window,
 * and its n/2 + 1 values re^2 + im^2 go to out + w * os. backend/ref has
 * the FFT, every operation in its order; a backend runs a window in each
 * lane. (Silero's VAD takes the square root, as vadito's kernel did inside:
 * the same bits, since sqrt(x * x) is |x| in IEEE arithmetic.) */
void pk_k_stft(float *out, size_t os, const float *const *x, uint32_t count,
               const pk_fft *f, float *scratch);

/* --------------------------------------------------------- resampling --
 *
 * torchaudio's sinc resampler (core/resample.c), with the rates reduced by
 * their gcd to orig in and nw out: the outputs come in blocks of nw, one
 * for each phase j, and block b's input starts b * orig samples into x.
 * Phase j is len[j] taps at k + at[j], over the input from its first[j]th
 * sample:
 *
 *   acc = 0
 *   for m = 0 .. len[j]-1:  acc = fmaf(k[at[j] + m], x[b*orig + first[j] + m], acc)
 *   y[b*nw + j] = acc
 *
 * for blocks b < nb. A backend runs a block in each lane, at the same phase
 * across the lanes, so that each lane runs the chain exactly. */
void pk_k_resample(float *y, const float *x, uint32_t nb, uint32_t orig, uint32_t nw, const float *k,
                   const size_t *at, const uint32_t *first, const uint32_t *len);

/* ------------------------------------------------------------ backends --
 *
 * Every function above is a call through the table of the backend in use.
 * A backend's file defines its kernels as its own and exports the table:
 * pk_kernels_ref, pk_kernels_neon and _neon_dp (backend/neon, without and
 * with PK_NEON_DOTPROD), pk_kernels_avx2, _avx2_vnni, _avx512_vnni and
 * _amx_int8 (backend/avx2, without PK_AVX_VNNI, and with it in AVX-VNNI's
 * encoding, AVX512-VNNI's, or AVX512-VNNI's with AMX), each compiled with
 * its own instruction set, the rest of the tree with the family's
 * baseline. */
typedef struct {
    const char *id;                 /* "avx2-vnni", as PKTITO_KERNELS names it */
    const char *name;               /* and what it is built for, for `pktito-c bench` */
    const char *const *needs;       /* the CPU features its code uses, NULL-ended:
                                       "avx2", "fma", "avxvnni", "avx512vnni",
                                       "avx512vl", "asimddp" (ARMv8.2's dot product) */
    void (*quantize)(pk_act *a, const float *x, size_t xs);
    void (*gemm_q8)(float *out, size_t os, const int8_t *w, const float *ws, const float *bias,
                    uint32_t rows, const pk_act *a, int add);
    void (*gemm_f32)(float *out, size_t os, const float *w, uint32_t ld, uint32_t rows,
                     uint32_t n, const float *x, size_t xs, uint32_t T,
                     const float *bias, size_t bs, int relu);
    void (*layernorm)(float *out, size_t os, const float *x, size_t xs, uint32_t T, uint32_t n,
                      const float *w, const float *b, float eps);
    void (*softmax)(float *x, uint32_t n, float scale);
    void (*silu)(float *x, size_t n);
    void (*glu)(float *out, size_t os, const float *x, size_t xs, uint32_t T, uint32_t n);
    void (*dwconv)(float *out, size_t os, const float *x, size_t xs, uint32_t T, uint32_t C,
                   const float *w, const float *b, uint32_t K);
    void (*conv3x3s2)(float *out, const float *in, uint32_t r0, uint32_t Ti, uint32_t Fi,
                      uint32_t C, int depthwise, const float *w, const float *b, int relu,
                      uint32_t t0, uint32_t t1);
    void (*lstm)(float *h, float *c, const float *g, uint32_t H);
    void (*stft)(float *out, size_t os, const float *const *x, uint32_t count,
                 const pk_fft *f, float *scratch);
    void (*resample)(float *y, const float *x, uint32_t nb, uint32_t orig, uint32_t nw, const float *k,
                     const size_t *at, const uint32_t *first, const uint32_t *len);
} pk_kernels;

extern const pk_kernels pk_kernels_ref, pk_kernels_neon, pk_kernels_neon_dp, pk_kernels_avx2,
                        pk_kernels_avx2_vnni, pk_kernels_avx512_vnni,
                        pk_kernels_amx_int8;

/* This build's backends, the fastest first and ref last, NULL-ended. */
extern const pk_kernels *const pk_kernels_built[];

/* The backend that runs: the first of pk_kernels_built whose needs this CPU
 * meets, or the one the environment's PKTITO_KERNELS names (to compare them;
 * not a setting anyone should need). NULL, with the reason in why, when
 * PKTITO_KERNELS names one the build lacks or the CPU cannot run. */
const pk_kernels *pk_kernels_pick(char *why, size_t whylen);

/* The name of the backend in use, for `pktito-c bench`. */
const char *pk_k_name(void);

#endif /* PK_KERNELS_H */
