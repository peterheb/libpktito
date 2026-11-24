/* Matrix products and their activations, split over a pool of threads
 * (passes.c): what the encoder, the joint and the diarizer share.
 * Internal. */
#ifndef PK_PASSES_H
#define PK_PASSES_H

#include "model.h"
#include "pool.h"

#include <stdatomic.h>

/* make PROFILE=1: wall time in each stage of a transcription or a
 * diarization, read on the calling thread around whole passes, never inside
 * a kernel, and printed by `pktito-c bench`. The diarizer's stages are the
 * recognizer's where the work is the same; the head and the cache are its
 * own. */
enum { PK_P_MEL, PK_P_SUB, PK_P_POS, PK_P_NORM, PK_P_QUANT, PK_P_FF, PK_P_QKV, PK_P_ATT,
       PK_P_ATTO, PK_P_CONV, PK_P_DW, PK_P_DEC, PK_P_HEAD, PK_P_CACHE, PK_NPROF };
#ifdef PK_PROFILE
extern double pk_prof[PK_NPROF];
double pk_prof_now(void);
#define PROF(i, stmt) do { double t0_ = pk_prof_now(); stmt; pk_prof[i] += pk_prof_now() - t0_; } while (0)
#else
#define PROF(i, stmt) do { stmt; } while (0)
#endif

/* T columns of x (column t at x + t * xs) quantized into act, n inputs
 * each in blocks of qb, 16 columns an item */
void pk_quantize_on(pk_pool *p, pk_act *act, const float *x, size_t xs, uint32_t T, uint32_t n,
                    uint32_t qb);

/* What an item of a product does next with the tile it has just made,
 * rows [r0, r0 + nr) of the output's columns [c0, c0 + nc), while the tile
 * is in its core's cache (the feed-forward's SiLU or GELU, the attention's
 * panels), on thread th */
typedef void (*pk_tile_fn)(void *arg, uint32_t r0, uint32_t nr, uint32_t c0, uint32_t nc, int th);

/* A product's activations again in the backend's own order
 * (pk_k_act_pack), where the backend has one (pk_k_packs): its columns in
 * sixteens, each packed by the first item to need it and read by every item
 * of the product after it. One serves a pool's products, which run one at a
 * time, and holds the largest of their activations: cap bytes at x, cap /
 * 32 floats at d, cap / 512 bytes at st, cap what PK_PACKED_CAP gives for
 * the most columns and inputs (their blocks are 32 inputs or more, and so
 * are they). */
typedef struct {
    size_t cap;
    int8_t *x;                       /* [T up to a 16][n] */
    float *d;                        /* [T up to a 16][n / qb] */
    _Atomic unsigned char *st;       /* a sixteen's: 0, 1 while a thread packs it, 2 packed */
} pk_packed;
#define PK_PACKED_CAP(T, n) ((((size_t)(T) + 15) & ~(size_t)15) * (size_t)(n))

/* out = w act (+ bias) (+ out, with add), rows and columns split over the
 * pool, then `then` (if not NULL) on each item's tile. pk, or NULL: where
 * the product's columns are packed for its items, if the backend packs. */
void pk_gemm_on(pk_pool *p, pk_packed *pk, float *out, size_t os, const pk_q8 *w,
                const float *bias, const pk_act *act, int add, pk_tile_fn then, void *then_arg);

/* A job's items in parts, for one pk_pool_run over several passes'
 * (pk_parts_on): part k's items after those of the parts before it */
typedef struct { pk_job_fn fn; void *arg; int items; } pk_part;
void pk_parts_on(pk_pool *p, const pk_part *parts, int n);

/* pk_gemm_on's items as a part, its job in j. unit, if not 0: every item's
 * rows a multiple of unit; whole: every item every column. For a then that
 * needs them, core/diar.c's heads. Parts in one job that share a pk share
 * their activations. */
typedef struct {
    float *out;
    size_t os;
    const pk_q8 *w;
    const float *bias;
    const pk_act *act;
    int add;
    uint32_t step, rb, cstep;
    pk_tile_fn then;
    void *then_arg;
    const pk_packed *pk;             /* the columns packed, or NULL */
} pk_gemm_job;
pk_part pk_gemm_part(pk_gemm_job *j, pk_pool *p, pk_packed *pk, float *out, size_t os, const pk_q8 *w,
                     const float *bias, const pk_act *act, int add, uint32_t unit, int whole, pk_tile_fn then,
                     void *then_arg);

/* Inputs [k0, k0 + n) of column t of act from the n floats at x, whole
 * blocks quantized alone: the bytes and scales the whole column's
 * quantizing gives them */
void pk_quant_part(pk_act *act, uint32_t t, uint32_t k0, uint32_t n, const float *x);

/* w act straight into pk_k_gemm_f32's panels, a panel ps floats apart, as
 * the attention's V is laid out (passes.c, panel_item) */
void pk_gemm_panels_on(pk_pool *p, pk_packed *pk, float *out, size_t ps, const pk_q8 *w,
                       const pk_act *act);
typedef struct {
    float *out;
    size_t ps;
    const pk_q8 *w;
    const pk_act *act;
    uint32_t rb, cstep;
    const pk_packed *pk;             /* the columns packed, or NULL */
} pk_panel_job;
pk_part pk_panels_part(pk_panel_job *j, pk_pool *p, pk_packed *pk, float *out, size_t ps, const pk_q8 *w,
                       const pk_act *act);

/* A product's tile transposed into pk_k_gemm_f32's panels (passes.c,
 * pk_panels_tile): src's n columns of D, head-major in heads of dk, into
 * dst, each head's np columns (n padded with zeros) */
typedef struct { float *dst; const float *src; uint32_t n, np, D, dk; } pk_panels;
void pk_panels_tile(void *arg, uint32_t r0, uint32_t nr, uint32_t c0, uint32_t nc, int th);

#endif /* PK_PASSES_H */
