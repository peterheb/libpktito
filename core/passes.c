/* Matrix products and the activations they take, split over a pool of
 * threads: the passes the encoder (core/encoder.c), the joint
 * (core/decoder.c) and the diarizer (core/diar.c) share. Each splits its
 * output by rows, columns or frames, and every output element is one
 * item's whole, so the bits do not depend on the split or the threads.
 */
#include "passes.h"
#include "kernels.h"

#include <stdint.h>
#include <string.h>

static uint32_t min32(uint32_t a, uint32_t b) { return a < b ? a : b; }

typedef struct {
    pk_act *act;
    const float *x;
    size_t xs;
    uint32_t step;
} quant_job;

static void quant_item(void *arg, int i, int th) {
    (void)th;
    quant_job *j = arg;
    uint32_t t0 = (uint32_t)i * j->step, nt = min32(j->step, j->act->T - t0);
    pk_act sub = *j->act;
    sub.q += (size_t)t0 * sub.n;
    sub.d += (size_t)t0 * (sub.n / sub.qb);
    sub.T = nt;
    pk_k_quantize(&sub, j->x + t0 * j->xs, j->xs);
}

void pk_quantize_on(pk_pool *p, pk_act *act, const float *x, size_t xs, uint32_t T, uint32_t n,
                    uint32_t qb) {
    act->n = n;
    act->qb = qb;
    act->T = T;
    quant_job j = { act, x, xs, 16 };
    pk_pool_run(p, quant_item, &j, (int)((T + j.step - 1) / j.step));
}

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
} gemm_job;

static void gemm_item(void *arg, int i, int th) {
    gemm_job *j = arg;
    uint32_t r0 = ((uint32_t)i % j->rb) * j->step, nr = min32(j->step, j->w->rows - r0);
    uint32_t c0 = ((uint32_t)i / j->rb) * j->cstep, nc = min32(j->cstep, j->act->T - c0);
    pk_act sub = *j->act;
    sub.q += (size_t)c0 * sub.n;
    sub.d += (size_t)c0 * (sub.n / sub.qb);
    sub.T = nc;
    pk_k_gemm_q8(j->out + (size_t)c0 * j->os + r0, j->os, j->w->q + (size_t)r0 * j->w->cols,
                 j->w->s + r0, j->bias ? j->bias + r0 : NULL, nr, &sub, j->add);
    if (j->then)
        j->then(j->then_arg, r0, nr, c0, nc, th);
}

/* Rows in slices of 64: few enough that each streams its weights through
 * once for its columns, and (pool item sizes, backend/avx2/CLIMB.md) the
 * size that measured best against 32 and 128. Or of 256 where those still
 * make two rounds of the pool: each call's own work, the backend's copy of
 * its columns and its panels' sums, a quarter as often, +3 to 7% at one
 * and two threads on the 9950X, a c7i and a c8i (CLIMB.md there, pk_gemm's
 * step), where with fewer the last round's idle share costs more (one
 * round of 16 at 16 threads: -1.5%). And the columns in chunks
 * when that fills the pool's last round: a 1024-row matrix is 16 slices,
 * which six threads take in three rounds with the last a third idle, and in
 * three chunks of columns 48 items, eight whole rounds (H5 there). A chunk
 * is whole tiles of 3 and 4 columns, and at least 48 columns, so that a
 * call's per-call work - the backend's copy of its columns, its panels'
 * sums - stays a few percent of it. Every output is still one call's, so
 * the bits do not depend on the split. Each item then hands its tile to
 * then, if there is one. */
void pk_gemm_on(pk_pool *p, float *out, size_t os, const pk_q8 *w, const float *bias,
                const pk_act *act, int add, pk_tile_fn then, void *then_arg) {
    const uint32_t nth = (uint32_t)pk_pool_active(p);
    const uint32_t step = w->rows / 256 >= 2 * nth ? 256 : 64, rb = (w->rows + step - 1) / step;
    uint32_t best = 1, best_idle = UINT32_MAX, best_all = 1;
    for (uint32_t cc = 1; cc <= 8 && (cc == 1 || act->T / cc >= 48); cc++) {
        uint32_t items = rb * cc, all = (items + nth - 1) / nth * nth;
        /* the idle share of the last round, compared as fractions */
        if ((uint64_t)(all - items) * best_all < (uint64_t)best_idle * all) {
            best = cc;
            best_idle = all - items;
            best_all = all;
        }
    }
    uint32_t cstep = (act->T + best - 1) / best;
    cstep = best > 1 ? (cstep + 11) / 12 * 12 : act->T;
    uint32_t cc = cstep ? (act->T + cstep - 1) / cstep : 1;
    gemm_job j = { out, os, w, bias, act, add, step, rb, cstep ? cstep : 1, then, then_arg };
    pk_pool_run(p, gemm_item, &j, (int)(rb * cc));
}

/* Inputs [k0, k0 + n) of column t of act, from the n floats at x: whole
 * blocks, each quantized alone, so the bytes and scales that quantizing
 * the whole column gives them. */
void pk_quant_part(pk_act *act, uint32_t t, uint32_t k0, uint32_t n, const float *x) {
    const size_t at = (size_t)t * act->n + k0;
    pk_act part = { act->q + at, act->d + at / act->qb, n, act->qb, 1 };
    pk_k_quantize(&part, x, n);
}

/* V straight into pk_k_gemm_f32's panels by the int8 product, with nothing
 * to repack after: (key j, dimension d) of head h is at
 *
 *   vp + ((h dk + d) / 16) T 16 + j 16 + d % 16
 *
 * which is the product's 16-row panel r / 16 laid out with its columns 16
 * floats apart, so one call a panel, os 16, writes it in place. A 64-row
 * slice an item, as pk_gemm deals them. */
typedef struct { float *out; size_t ps; const pk_q8 *w; const pk_act *act; } panel_job;

static void panel_item(void *arg, int i, int th) {
    (void)th;
    panel_job *j = arg;
    for (uint32_t r = (uint32_t)i * 64; r < min32((uint32_t)i * 64 + 64, j->w->rows); r += 16)
        pk_k_gemm_q8(j->out + (size_t)(r / 16) * j->ps, 16, j->w->q + (size_t)r * j->w->cols,
                     j->w->s + r, NULL, 16, j->act, 0);
}

void pk_gemm_panels_on(pk_pool *p, float *out, size_t ps, const pk_q8 *w, const pk_act *act) {
    panel_job j = { out, ps, w, act };
    pk_pool_run(p, panel_item, &j, (int)((w->rows + 63) / 64));
}

/* K and the position table as pk_k_gemm_f32 panels, made by the items of
 * their products, each transposing its own tile into them while the tile
 * is in its core's cache:
 *
 *   kp: (d, key j)      at ((j/16) dk + d) 16 + j%16     keys padded to Tp with zeros
 *   pp: (d, position m) at ((m/16) dk + d) 16 + m%16     positions padded to Pp
 *
 * head h's from kp + h Tp dk and pp + h Pp dk, the padding written by the
 * items with the last columns. A panel's sixteen columns are read while a
 * dimension's sixteen values are written as one run: the columns stay in
 * L1 and every store lands beside the last. A row at a time put each store
 * a cache line from the one before, and that copy, with V's and the
 * queries' beside it, was 5.8% of a Pi 5's minute; V now needs none
 * (panel_item), the queries are made a block at a time where they are used
 * (att_item), and what was left was 2.2%. Taking K and P as their products
 * leave them and transposing each block's scores instead, which grows with
 * T squared, was slower: 5.08 s a minute where this was 4.75 (backend/
 * neon-dp/CLIMB.md). As a pass of its own after the products, a head and
 * 64 frames an item, it read lines every thread had written, and on the
 * 9950X its jobs took 54 us at 8 threads and 123 at 16; made here, the
 * products take on 39 to 55% of its time at 16 threads there. At one
 * thread they took on a third more than it until each row's head and
 * dimension were stepped along the rows rather than divided out, and a
 * whole panel was copied in a loop of sixteen with no padding test: the
 * one-thread run there 0.2 to 1.0% faster for it, back to its time with
 * the pass (backend/avx2/CLIMB.md, 2026-10-28). */
void pk_panels_tile(void *arg, uint32_t r0, uint32_t nr, uint32_t c0, uint32_t nc, int th) {
    (void)th;
    pk_panels *j = arg;
    const uint32_t dk = j->dk, c1 = c0 + nc == j->n ? j->np : c0 + nc;
    for (uint32_t p0 = c0 & ~15u; p0 < c1; p0 += 16) {
        /* the panel's columns [lo, hi) are the tile's, padding as zeros;
         * whole, all sixteen are the tile's and none is padding */
        const uint32_t lo = c0 > p0 ? c0 - p0 : 0, hi = min32(16, c1 - p0);
        const int whole = lo == 0 && hi == 16 && p0 + 16 <= j->n;
        const float *col[16];
        for (uint32_t i = lo; i < hi; i++)
            col[i] = p0 + i < j->n ? j->src + (size_t)(p0 + i) * j->D : NULL;
        /* row r is dimension d of head h, the two stepped along with it */
        for (uint32_t r = r0, h = r0 / dk, d = r0 % dk; r < r0 + nr; r++) {
            float *out = j->dst + ((size_t)h * j->np + p0) * dk + (size_t)d * 16;
            if (whole)
                for (uint32_t i = 0; i < 16; i++)
                    out[i] = col[i][r];
            else
                for (uint32_t i = lo; i < hi; i++)
                    out[i] = col[i] ? col[i][r] : 0.0f;
            if (++d == dk) {
                d = 0;
                h++;
            }
        }
    }
}
