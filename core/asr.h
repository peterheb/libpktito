/* A transcriber's working memory and the passes that use it (asr.c,
 * encoder.c, decoder.c). Internal. */
#ifndef PK_ASR_H
#define PK_ASR_H

#include "model.h"
#include "pool.h"

/* Encoder frames the subsampling computes at a time on one thread: its
 * stages' buffers then stay about 3.5 MB a thread at 256 channels, where
 * a whole 30 s recording's first stage alone would be 98 MB. */
#define PK_SUB_BLOCK 8

/* Queries the attention takes at a time on one thread */
#define PK_ATT_BLOCK 16

/* make PROFILE=1: wall time in each stage of a transcription, read on the
 * calling thread around whole passes, never inside a kernel, and printed by
 * `pkito bench`. */
enum { PK_P_MEL, PK_P_SUB, PK_P_POS, PK_P_NORM, PK_P_QUANT, PK_P_FF, PK_P_QKV, PK_P_REPACK,
       PK_P_ATT, PK_P_ATTO, PK_P_CONV, PK_P_DW, PK_P_DEC, PK_NPROF };
#ifdef PK_PROFILE
extern double pk_prof[PK_NPROF];
double pk_prof_now(void);
#define PROF(i, stmt) do { double t0_ = pk_prof_now(); stmt; pk_prof[i] += pk_prof_now() - t0_; } while (0)
#else
#define PROF(i, stmt) do { stmt; } while (0)
#endif

typedef struct {
    float  *s1, *s2, *s3, *s4;       /* the subsampling's stages, one block's rows */
    int8_t *aq;                      /* its pointwise convolutions' int8 input */
    float  *ad;
    float  *qu, *qv;                 /* a block of queries plus each position bias */
    float  *sc, *bd;                 /* its scores, and their position half */
    float  *mel;                     /* a block of power spectra, and the FFT's scratch */
} pk_thread_ws;

struct pk_asr {
    const pk_model *m;
    pk_pool *pool;
    int nth;

    size_t cap;                      /* feature frames the buffers below hold */
    void  *mem;

    /* the features */
    float *mel_scratch, *feat;       /* feat: [T0][n_mels] */
    const float **win;

    /* the encoder, Te frames */
    float *s5;                       /* [Te][F3 C]: the subsampling's output, flattened */
    float *x, *h, *u;                /* [Te][D], [Te][D], [Te][max(F, 2D)] */
    float *q, *o;                    /* [Te][D] */
    float *pe, *p;                   /* [2Te-1][D]: positions, and their projection */
    float *kp, *vp, *pp;             /* K, V and P per head, as pk_k_gemm_f32 panels */
    pk_act act, ape;                 /* int8 activations; the positions' */
    pk_thread_ws *tws;

    /* the decoder */
    float *fproj;                    /* [Te][J]: the joint's encoder half */
    float *dx, *gates, *dh, *dc, *gp, *z, *logit;
    pk_act dact;
    pk_tok *toks;
    size_t ntoks, toks_cap;
};

/* The encoder over T0 feature frames in a->feat: a->x holds the output,
 * Te frames, which it returns. */
uint32_t pk_encode(pk_asr *a, uint32_t T0);

/* TDT's greedy search over Te encoder frames in a->x: a->toks. Returns the
 * token count, or -1 if memory runs out. */
long pk_decode(pk_asr *a, uint32_t Te);

/* Encoder frames T0 feature frames make: three halvings, each rounding up. */
static inline uint32_t pk_enc_frames(uint32_t T0) {
    uint32_t t = T0;
    for (int i = 0; i < 3; i++) t = (t + 1) / 2;
    return t;
}

/* Run fn over [0, items) on the transcriber's pool */
static inline void pk_par(pk_asr *a, pk_job_fn fn, void *arg, int items) {
    pk_pool_run(a->pool, fn, arg, items);
}

/* Matrix products and their activations, split over the pool: a.T columns
 * of x (column t at x + t * xs) quantized into act in blocks of qb, then
 * out = w act (+ bias) (+ out, with add). */
void pk_quantize(pk_asr *a, pk_act *act, const float *x, size_t xs, uint32_t T, uint32_t n,
                 uint32_t qb);
void pk_gemm(pk_asr *a, float *out, size_t os, const pk_q8 *w, const float *bias,
             const pk_act *act, int add);

#endif /* PK_ASR_H */
