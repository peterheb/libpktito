/* Matrix products and their activations, split over a pool of threads
 * (passes.c): what the encoder, the joint and the diarizer share.
 * Internal. */
#ifndef PK_PASSES_H
#define PK_PASSES_H

#include "model.h"
#include "pool.h"

/* T columns of x (column t at x + t * xs) quantized into act, n inputs
 * each in blocks of qb, 16 columns an item */
void pk_quantize_on(pk_pool *p, pk_act *act, const float *x, size_t xs, uint32_t T, uint32_t n,
                    uint32_t qb);

/* What an item of a product does next with the tile it has just made,
 * rows [r0, r0 + nr) of the output's columns [c0, c0 + nc), while the tile
 * is in its core's cache (the feed-forward's SiLU or GELU, the attention's
 * panels), on thread th */
typedef void (*pk_tile_fn)(void *arg, uint32_t r0, uint32_t nr, uint32_t c0, uint32_t nc, int th);

/* out = w act (+ bias) (+ out, with add), rows and columns split over the
 * pool, then `then` (if not NULL) on each item's tile */
void pk_gemm_on(pk_pool *p, float *out, size_t os, const pk_q8 *w, const float *bias,
                const pk_act *act, int add, pk_tile_fn then, void *then_arg);

/* Inputs [k0, k0 + n) of column t of act from the n floats at x, whole
 * blocks quantized alone: the bytes and scales the whole column's
 * quantizing gives them */
void pk_quant_part(pk_act *act, uint32_t t, uint32_t k0, uint32_t n, const float *x);

/* w act straight into pk_k_gemm_f32's panels, a panel ps floats apart, as
 * the attention's V is laid out (passes.c, panel_item) */
void pk_gemm_panels_on(pk_pool *p, float *out, size_t ps, const pk_q8 *w, const pk_act *act);

/* A product's tile transposed into pk_k_gemm_f32's panels (passes.c,
 * pk_panels_tile): src's n columns of D, head-major in heads of dk, into
 * dst, each head's np columns (n padded with zeros) */
typedef struct { float *dst; const float *src; uint32_t n, np, D, dk; } pk_panels;
void pk_panels_tile(void *arg, uint32_t r0, uint32_t nr, uint32_t c0, uint32_t nc, int th);

#endif /* PK_PASSES_H */
