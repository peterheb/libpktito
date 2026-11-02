/* Silero VAD: one 32 ms frame of 16 kHz audio in, the probability that it
 * is speech out. pktito runs it over a long recording to cut it at its
 * silences into pieces the encoder can take whole (pk_chunks, core/asr.c).
 *
 * This is vadito's model (core/vad.c there), with the 8 kHz branch left
 * out because Parakeet hears 16 kHz only, and with the weights bound from a
 * .pkf instead of vadito's own file. The arithmetic is vadito's, kernel for
 * kernel, so its probabilities are vadito's bits, and vadito holds those to
 * onnxruntime's within 1.4e-5 over 47 minutes (its README, "How close"). For
 * each frame:
 *
 *   input   the frame and the last 64 samples before it,
 *           reflected 64 samples past its end              640 samples
 *   STFT    4 Hann windows of 256 at hop 128, magnitudes   4 x 129
 *   conv1   3 taps, stride 1, ReLU                         4 x 128
 *   conv2   3 taps, stride 2, ReLU                         2 x 64
 *   conv3   3 taps, stride 2, ReLU                         1 x 64
 *   conv4   3 taps, stride 1, ReLU                         1 x 128
 *   LSTM    one step of a 128-unit cell, state carried     128
 *   head    sigmoid(b + w . relu(h))                       1
 *
 * Three things here are not what Silero's ONNX graph does, none moving an
 * answer by more than fp32 rounding: the STFT is an FFT (Silero ships it as
 * a 258 x 256 filter bank, a windowed DFT the converter checks it is), the
 * convolutions skip the taps that land on padding, and every sum adds its
 * bias after its products (vadito's backend/kernels.h has the numbers).
 */
#include "asr.h"
#include "model.h"
#include "kernels.h"
#include "os.h"
#include "vmath.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Frames encoded together by pk_vad_probs: about 7 KB of activations each */
#define PK_VAD_BLOCK 32

/* Blocks pk_vad_probs_on gives each of the pool's threads a run. A run
 * ends in the recurrence on the calling thread while the pool waits, so
 * fewer, longer runs park and wake it less: over AliceBench's chapter 1 on
 * a 12900HK's 14 threads, 2 a thread took 0.082 s, 4 0.075 and 8 0.066; on
 * a Pi 5's 4, whose L3 a run of 8 fills, 0.314 and 0.317 at 2 and 8
 * (backend/avx2/CLIMB.md, 2026-11-02). A run's gates are 64 KB a block. */
#define PK_VAD_RUN 8

/* Samples beyond this are clamped. The STFT squares its bins, and a bin
 * can be 128 times a sample's size, so a sample past about 1e17 overflows a
 * float there and the inf - inf that follows makes NaN of the LSTM state for
 * good. Real audio is within [-1, 1]; 1e6 leaves eleven orders of magnitude. */
#define PK_SAMPLE_MAX 1e6f

struct pk_vad {
    const pk_vadw *b;
    float h[PK_VAD_HIDDEN], c[PK_VAD_HIDDEN];
    float ctx[PK_VAD_CONTEXT];    /* the last frame's last 64 samples */
    /* one block's activations, frame-major */
    float *mag;                   /* [BLOCK][4][129] */
    float *y1;                    /* [BLOCK][4][128] */
    float *y2;                    /* [BLOCK][2][64] */
    float *y3;                    /* [BLOCK][64] */
    float *y4;                    /* [BLOCK][128] */
    float *gx;                    /* [BLOCK][512] */
    float *mag1, *y1p;            /* one frame's mag and y1, a zero frame each side */
    float *xin;                   /* [BLOCK][640]: each frame's model input, reflected */
    const float *win[PK_VAD_BLOCK * PK_VAD_WIN];
    float *fft;                   /* pk_k_stft's scratch */
    void  *mem;
    /* pk_vad_probs_on's: a stream for each of the pool's threads, which
     * lends it its activations, and a run's gates' input halves */
    const pk_model *m;
    pk_vad **worker;
    int nworker;
    float *rgx;                   /* [PK_VAD_RUN * nworker * BLOCK][512] */
};

/* A 3-tap convolution with one zero of padding at each end, over T frames'
 * worth of input at once: in is [T][tin][cin], out is [T][tout][cout]. For
 * each output position t the taps j whose input position t*stride - 1 + j
 * lands in [0, tin) are one contiguous run of k in the weights' panels and
 * one contiguous run of input, so a position is one pk_k_gemm_f32 call over
 * all T frames. */
static void conv(float *out, const float *in, uint32_t T, uint32_t tin, uint32_t cin,
                 uint32_t cout, uint32_t stride, const float *w, const float *bias) {
    uint32_t tout = (tin - 1) / stride + 1;
    for (uint32_t t = 0; t < tout; t++) {
        int p0 = (int)(t * stride) - 1;               /* tap 0's input position */
        uint32_t j0 = p0 < 0 ? (uint32_t)-p0 : 0;
        uint32_t j1 = p0 + 3 > (int)tin ? (uint32_t)((int)tin - p0) : 3;
        pk_k_gemm_f32(out + (size_t)t * cout, (size_t)tout * cout,
                      w + (size_t)16 * j0 * cin, 3 * cin, cout, (j1 - j0) * cin,
                      in + (size_t)(p0 + (int)j0) * cin, (size_t)tin * cin, T, bias, 0, 1);
    }
}

static float sane(float v) {
    if (!isfinite(v))
        return 0.0f;
    return v > PK_SAMPLE_MAX ? PK_SAMPLE_MAX : v < -PK_SAMPLE_MAX ? -PK_SAMPLE_MAX : v;
}

/* The STFT magnitudes of T consecutive frames x[T * 512] into mag, the
 * first frame's context ctx0 (the 64 samples before it, made sane).
 * Each frame's model input is built whole - its context, its samples, and
 * PyTorch's reflect pad past the end, x[len + i] = x[len - 2 - i] - and its
 * four windows go to pk_k_stft with the rest of the block's. The kernel's
 * power is made a magnitude here: sqrt(x * x) is |x| in IEEE arithmetic, so
 * these are the bits vadito's kernel made inside. */
static void stft_block(pk_vad *v, const float *x, uint32_t T, const float *ctx0, float *mag) {
    const pk_vadw *b = v->b;
    const uint32_t ctx = PK_VAD_CONTEXT, frame = PK_VAD_FRAME, len = ctx + frame;
    const uint32_t in = len + PK_VAD_NFFT / 4;
    for (uint32_t f = 0; f < T; f++) {
        const float *fx = x + (size_t)f * frame;
        float *xin = v->xin + (size_t)f * in;
        for (uint32_t i = 0; i < ctx; i++)
            xin[i] = f ? sane(fx[(int)i - (int)ctx]) : ctx0[i];
        for (uint32_t i = 0; i < frame; i++)
            xin[ctx + i] = sane(fx[i]);
        for (uint32_t i = 0; i < PK_VAD_NFFT / 4; i++)
            xin[len + i] = xin[len - 2 - i];
        for (uint32_t w = 0; w < PK_VAD_WIN; w++)
            v->win[f * PK_VAD_WIN + w] = xin + w * PK_VAD_HOP;
    }
    pk_k_stft(mag, PK_VAD_BINS, v->win, T * PK_VAD_WIN, &b->fft, v->fft);
    for (size_t i = 0; i < (size_t)T * PK_VAD_WIN * PK_VAD_BINS; i++)
        mag[i] = sqrtf(mag[i]);
}

/* The context of the frame after the one that ends at end: its last 64
 * samples, made sane */
static void context(float *ctx, const float *end) {
    for (int i = 0; i < PK_VAD_CONTEXT; i++)
        ctx[i] = sane(end[i - PK_VAD_CONTEXT]);
}

pk_vad *pk_vad_new(const pk_model *m) {
    if (!m || !m->vad.wx)
        return NULL;
    pk_vad *v = calloc(1, sizeof *v);
    if (!v)
        return NULL;
    const size_t B = PK_VAD_BLOCK;
    size_t sizes[] = {
        B * PK_VAD_WIN * PK_VAD_BINS, B * PK_VAD_WIN * PK_VAD_C1, B * 2 * PK_VAD_C2,
        B * PK_VAD_C3, B * PK_VAD_C4, B * PK_VAD_GATES,
        B * (PK_VAD_CONTEXT + PK_VAD_FRAME + PK_VAD_NFFT / 4), PK_K_STFT_SCRATCH(PK_VAD_NFFT),
        (PK_VAD_WIN + 2) * (size_t)PK_VAD_BINS, (PK_VAD_WIN + 2) * (size_t)PK_VAD_C1,
    };
    float **dst[] = { &v->mag, &v->y1, &v->y2, &v->y3, &v->y4, &v->gx, &v->xin, &v->fft,
                      &v->mag1, &v->y1p };
    size_t total = 0;
    for (size_t i = 0; i < sizeof sizes / sizeof sizes[0]; i++)
        total += (sizes[i] + 15) & ~(size_t)15;       /* each array on its own 64 bytes */
    v->mem = calloc(total * sizeof(float) + 64, 1);  /* mag1's and y1p's padding stays 0 */
    if (!v->mem) {
        free(v);
        return NULL;
    }
    float *p = (float *)(((uintptr_t)v->mem + 63) & ~(uintptr_t)63);
    for (size_t i = 0; i < sizeof sizes / sizeof sizes[0]; i++) {
        *dst[i] = p;
        p += (sizes[i] + 15) & ~(size_t)15;
    }
    v->m = m;
    v->b = &m->vad;
    pk_vad_reset(v);
    return v;
}

void pk_vad_reset(pk_vad *v) {
    memset(v->h, 0, sizeof v->h);
    memset(v->c, 0, sizeof v->c);
    memset(v->ctx, 0, sizeof v->ctx);
}

static void free_workers(pk_vad *v) {
    for (int t = 0; t < v->nworker; t++)
        pk_vad_free(v->worker[t]);
    free(v->worker);
    free(v->rgx);
    v->worker = NULL, v->rgx = NULL, v->nworker = 0;
}

void pk_vad_free(pk_vad *v) {
    if (!v)
        return;
    free_workers(v);
    free(v->mem);
    free(v);
}

/* The encoder over T <= PK_VAD_BLOCK frames, the first one's context ctx,
 * as far as the input half of every frame's gates, bias included, into
 * gx[T][PK_VAD_GATES]: the part of a frame that needs no frame before it.
 * One frame alone takes conv1 and conv2 as one call each over their input
 * with a zero frame either side, which reads each weight panel once rather
 * than once per position; a zero product leaves a sum as it was, so these
 * are the batched path's bits (vadito's core/vad.c has the measurement). */
static void encode(pk_vad *v, const float *x, uint32_t T, const float *ctx, float *gx) {
    const pk_vadw *b = v->b;
    if (T == 1) {
        stft_block(v, x, 1, ctx, v->mag1 + PK_VAD_BINS);
        pk_k_gemm_f32(v->y1p + PK_VAD_C1, PK_VAD_C1, b->w1, 3 * PK_VAD_BINS, PK_VAD_C1,
                      3 * PK_VAD_BINS, v->mag1, PK_VAD_BINS, PK_VAD_WIN, b->b1, 0, 1);
        pk_k_gemm_f32(v->y2, PK_VAD_C2, b->w2, 3 * PK_VAD_C1, PK_VAD_C2, 3 * PK_VAD_C1,
                      v->y1p, 2 * PK_VAD_C1, 2, b->b2, 0, 1);
        conv(v->y3, v->y2, 1, 2, PK_VAD_C2, PK_VAD_C3, 2, b->w3, b->b3);
        conv(v->y4, v->y3, 1, 1, PK_VAD_C3, PK_VAD_C4, 1, b->w4, b->b4);
    } else {
        stft_block(v, x, T, ctx, v->mag);
        conv(v->y1, v->mag, T, PK_VAD_WIN, PK_VAD_BINS, PK_VAD_C1, 1, b->w1, b->b1);
        conv(v->y2, v->y1, T, PK_VAD_WIN, PK_VAD_C1, PK_VAD_C2, 2, b->w2, b->b2);
        conv(v->y3, v->y2, T, 2, PK_VAD_C2, PK_VAD_C3, 2, b->w3, b->b3);
        conv(v->y4, v->y3, T, 1, PK_VAD_C3, PK_VAD_C4, 1, b->w4, b->b4);
    }
    pk_k_gemm_f32(gx, PK_VAD_GATES, b->wx, PK_VAD_C4, PK_VAD_GATES, PK_VAD_C4, v->y4,
                  PK_VAD_C4, T, b->bg, 0, 0);
}

/* The LSTM and the head over T frames whose gates' input halves are gx,
 * one frame at a time, which is the only part that has to be */
static void step(pk_vad *v, float *gx, size_t T, float *probs) {
    const pk_vadw *b = v->b;
    for (size_t f = 0; f < T; f++) {
        float *g = gx + f * PK_VAD_GATES;
        pk_k_gemm_f32(g, 0, b->wh, PK_VAD_HIDDEN, PK_VAD_GATES, PK_VAD_HIDDEN, v->h, 0, 1, g, 0, 0);
        pk_k_lstm(v->h, v->c, g, PK_VAD_HIDDEN);
        float acc = 0;                            /* the bias last, as pk_k_gemm_f32's */
        for (uint32_t j = 0; j < PK_VAD_HIDDEN; j++)
            acc = pk_fmaf(b->hw[j], v->h[j] > 0 ? v->h[j] : 0, acc);
        probs[f] = pk_sigmoidf(acc + b->hb[0]);
    }
}

static void block(pk_vad *v, const float *x, uint32_t T, float *probs) {
    encode(v, x, T, v->ctx, v->gx);
    context(v->ctx, x + (size_t)T * PK_VAD_FRAME);
    step(v, v->gx, T, probs);
}

void pk_vad_probs(pk_vad *v, const float *x, size_t n, float *probs) {
    pk_os_throttle was = pk_os_unthrottle();  /* a recording's VAD, on this thread */
    for (size_t f = 0; f < n; f += PK_VAD_BLOCK) {
        uint32_t T = n - f < PK_VAD_BLOCK ? (uint32_t)(n - f) : PK_VAD_BLOCK;
        block(v, x + f * PK_VAD_FRAME, T, probs + f);
    }
    pk_os_rethrottle(was);
}

/* ----------------------------------------------------- over a pool --- */

/* A run of pk_vad_probs_on: n frames from x, their gates gx. The call's
 * first run takes its first block's context from the stream; every other
 * block's is the samples before it. */
typedef struct {
    pk_vad *v;
    const float *x;
    size_t n;
    int first;
    float *gx;
} vad_run;

static void encode_job(void *arg, int item, int thread) {
    vad_run *r = arg;
    const size_t f = (size_t)item * PK_VAD_BLOCK;
    const uint32_t T = r->n - f < PK_VAD_BLOCK ? (uint32_t)(r->n - f) : PK_VAD_BLOCK;
    const float *x = r->x + f * PK_VAD_FRAME;
    float ctx[PK_VAD_CONTEXT];
    if (r->first && !item)
        memcpy(ctx, r->v->ctx, sizeof ctx);
    else
        context(ctx, x);
    encode(r->v->worker[thread], x, T, ctx, r->gx + f * PK_VAD_GATES);
}

/* A stream for each of k threads, whose activations a block borrows, and
 * a run's gates: made the first time a call needs them, and kept */
static int workers(pk_vad *v, int k) {
    if (v->nworker >= k)
        return 1;
    free_workers(v);
    pk_vad **w = calloc((size_t)k, sizeof *w);
    float *g = malloc((size_t)PK_VAD_RUN * k * PK_VAD_BLOCK * PK_VAD_GATES * sizeof *g);
    int t = 0;
    while (w && g && t < k && (w[t] = pk_vad_new(v->m)))
        t++;
    if (t < k) {
        while (t--)
            pk_vad_free(w[t]);
        free(w), free(g);
        return 0;
    }
    v->worker = w, v->rgx = g, v->nworker = k;
    return 1;
}

void pk_vad_probs_on(pk_vad *v, pk_pool *pool, const float *x, size_t n, float *probs) {
    const int k = pool ? pk_pool_active(pool) : 1;
    if (k < 2 || n <= PK_VAD_BLOCK || !workers(v, pk_pool_threads(pool))) {
        pk_vad_probs(v, x, n, probs);
        return;
    }
    pk_os_throttle was = pk_os_unthrottle();
    const size_t run = (size_t)PK_VAD_RUN * k * PK_VAD_BLOCK;
    for (size_t f = 0; f < n; f += run) {
        vad_run r = { v, x + f * PK_VAD_FRAME, n - f < run ? n - f : run, f == 0, v->rgx };
        pk_pool_run(pool, encode_job, &r, (int)((r.n + PK_VAD_BLOCK - 1) / PK_VAD_BLOCK));
        step(v, v->rgx, r.n, probs + f);
    }
    context(v->ctx, x + n * PK_VAD_FRAME);
    pk_os_rethrottle(was);
}

float pk_vad_prob(pk_vad *v, const float *frame) {
    float p;
    block(v, frame, 1, &p);
    return p;
}
