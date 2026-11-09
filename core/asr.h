/* A transcriber's working memory and the passes that use it (asr.c,
 * encoder.c, decoder.c). Internal. */
#ifndef PK_ASR_H
#define PK_ASR_H

#include "model.h"
#include "passes.h"
#include "pool.h"

/* Encoder frames the subsampling deals to a thread at a time. Its stages
 * run a row at a time inside one (core/encoder.c, sub_item), so this sets
 * only how often the rows at a block's edges are computed twice. */
#define PK_SUB_BLOCK 8

/* Queries the attention takes at a time on one thread */
#define PK_ATT_BLOCK 16

/* Which recording of PK_TUNE_FRAMES or more races every core against the
 * fast kind (core/encoder.c, layers): the second */
#define PK_TUNE_AT 2

/* A chunk keeps few of many threads busy: each pass is a job over the
 * pool, and each job waits for its slowest item. Several transcriptions
 * at once, each on its share of the threads, keep more of them busy; but
 * a Pi 5's four cores gave its one transcriber's rate as two of two
 * threads and 0.91 of it as four of one (backend/avx2/CLIMB.md,
 * 2026-11-02). So a lane for each PK_LANE_THREADS threads, PK_LANES_MAX
 * at most (core/asr.c, run_lanes). */
#define PK_LANE_THREADS 4
#define PK_LANES_MAX    16

typedef struct {
    float  *s1, *s2, *s3, *s4;       /* the subsampling's stages: windows of rows, and a block's output */
    int8_t *aq;                      /* its pointwise convolutions' int8 input */
    float  *ad;
    float  *qu, *qv;                 /* a block of queries plus each position bias */
    float  *sc, *bd;                 /* its scores, and their position half */
    float  *mel;                     /* a block of power spectra, and the FFT's scratch */
    float  *row;                     /* a frame of a layer norm's output, or of the hidden's */
} pk_thread_ws;

struct pk_asr {
    const pk_model *m;
    pk_pool *pool;
    int nth;
    /* Every core, or the fast kind alone (pk_os_fast_cores, `fast` of them)
     * where the cores are of two kinds and the threads are pktito's choice:
     * `tune` the long recordings until the one that times the encoder's
     * layers on each (encoder.c, layers), `wide` the threads the passes
     * run on. */
    int fast, wide, tune;
    /* The caller's fast mode (pk_asr_set_fast_mode): the encoder's
     * activations in blocks of 64 where an input allows, not 32 */
    int qb64;
    /* The encoder on the Neural Engine, where the caller gave a Core ML
     * model (pk_asr_use_ane; make ANE=1): npu/ane.h */
    struct pk_ane *ane;
    /* the Neural Engine's calls in flight (asr.c, ane_chunks): each slot's
     * features in and encoder frames out, ane_cap feature frames each */
    float *ane_feat[4], *ane_x[4];
    size_t ane_cap;
    /* The encoder on Qualcomm's NPU, where the caller asked for it
     * (pk_asr_use_qnn; make QNN=1): npu/qnn.h */
    struct pk_qnn *qnn;
    /* A recording's chunks several at once (core/asr.c, run_lanes): at
     * most `lanes`, lane j > 0 a transcriber of its own on this one's
     * model, made the first time a run needs it, lane 0 this one; `drive`
     * the threads that run them, one a lane. */
    int lanes;
    pk_asr **lane;
    pk_pool *drive;
    /* pk_asr_transcribe's VAD, made the first time it runs and kept, with
     * a block's activations for each of the pool's threads */
    pk_vad *vad;

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
    pk_act act, ape, hid;            /* int8 activations; the positions'; the feed-forward's hidden */
    pk_thread_ws *tws;

    /* the decoder */
    float *fproj[2];                 /* [Te][J]: the joint's encoder half, a chunk's and the next's */
    int dsplit;                      /* the search's widest products split over the pool */
    float *dx, *gates, *dh, *dc, *gp, *z, *logit;
    pk_act dact;
    pk_tok *toks;
    size_t ntoks, toks_cap;
};

/* The encoder over T0 feature frames in a->feat: a->x holds the output,
 * Te frames, which it returns. */
uint32_t pk_encode(pk_asr *a, uint32_t T0);

/* The joint's encoder half of Te encoder frames in a->x, every frame at
 * once: f[Te][J] */
void pk_joint_frames(pk_asr *a, uint32_t Te, float *f);

/* TDT's greedy search over Te frames whose joint half is f: a->toks.
 * Returns the token count, or -1 if memory runs out. With split, its
 * widest products go over the pool's first threads, from the thread that
 * runs its jobs; without, the search runs on a thread of its own beside
 * them (pk_pool_aside), each product whole. */
long pk_decode(pk_asr *a, uint32_t Te, const float *f, int split);

/* pk_asr_run_chunks with a hook that can stop it: each(arg, i, toks,
 * count, spare) has chunk i's tokens, on the calling thread, and returns
 * nonzero to stop there (which the call returns). spare is a joint buffer
 * of a's free until each returns, the one chunk i used where one chunk
 * runs at a time, while the other holds chunk i + 1's encoder output:
 * pk_retry_run may use it then, and only it (core/transcribe.c). */
typedef int (*pk_each_fn)(void *arg, size_t i, const pk_tok *toks, long count, float *spare);
int pk_run_chunks_with(pk_asr *a, const float *x, const pk_seg *c, size_t n,
                       pk_each_fn each, void *arg);

/* pk_asr_run on n samples no longer than the longest chunk of the run in
 * progress, its joint half in spare: a->toks, the count, or -1. */
long pk_retry_run(pk_asr *a, const float *x, size_t n, float *spare);

/* A sample of a recording in encoder frames, rounded: pk_chunk_out's
 * at_frame. */
int32_t pk_frame_of(const pk_model *m, int64_t sample);

/* pk_vad_probs with Silero's encoder over the pool's active threads, a run
 * of blocks at a time, and the LSTM on the calling thread after each run:
 * the same bits, since a block's encoder needs only its own samples and the
 * 64 before them (core/vad.c). pk_vad_probs itself where the pool is one
 * thread, or the threads' memory cannot be had. */
void pk_vad_probs_on(pk_vad *v, pk_pool *pool, const float *x, size_t n, float *probs);

/* pk_chunks, with what each chunk is heard in (core/transcribe.c): win[k]
 * is out[k] and, where out[k]'s cut falls in a silence shorter than pause,
 * ctx of the recording past it, room left for that in its max_len. With
 * ctx 0, out is pk_chunks', and win NULL is allowed. */
size_t pk_chunks_past(const float *x, const pk_seg *s, size_t n, int64_t total, int64_t max_len,
                      int64_t ctx, int64_t pause, pk_seg *out, pk_seg *win);

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
