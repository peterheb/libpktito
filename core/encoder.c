/* The encoder: log-mel features in, one 1024-wide vector each 80 ms out.
 *
 * NeMo's ConformerEncoder (FastConformer, nemo/collections/asr/modules/
 * conformer_encoder.py) in eval mode:
 *
 *   subsampling   3x3 stride-2 convolutions over time and frequency: one
 *                 from the single feature channel to 256, ReLU; then twice a
 *                 depthwise one and a pointwise one, ReLU; then channels and
 *                 frequencies flattened and projected to d_model. T/8 frames.
 *   positions     the relative positions T-1 .. -(T-1) as sinusoids
 *   24 layers     x += ff1(LN x) / 2
 *                 x += attention(LN x)       Transformer-XL's, relative
 *                 x += convolution(LN x)     GLU, depthwise 9, BatchNorm, SiLU
 *                 x += ff2(LN x) / 2
 *                 x  = LN x
 *
 * NeMo runs a batch padded to its longest member and masks the padding at
 * every step: the subsampling's convolutions see zeros past each length
 * (MaskedConvSequential), the attention gives padded keys no weight, the
 * convolution module zeroes padded frames before its depthwise
 * convolution. At batch size one that makes every real frame exactly what
 * the same computation over the real frames alone makes, which is what
 * this does: there is no padding to mask. (NeMo computes one more feature
 * frame than it counts and masks it; core/mel.c does not make it.)
 *
 * Every matrix product is pk_k_gemm_q8, int8 weights against activations
 * quantized in blocks of 32; the attention's products are float, on
 * pk_k_gemm_f32 with their operands laid out as its panels. Each pass is
 * split over the pool by rows, heads or blocks of frames, and no output
 * element depends on how: a transcript is the same bits on any number of
 * threads.
 */
#include "asr.h"
#include "kernels.h"
#include "os.h"
#include "vmath.h"

#include <math.h>
#include <string.h>

#define QB 32              /* the encoder's activation blocks (README, "Why int8") */

static uint32_t min32(uint32_t a, uint32_t b) { return a < b ? a : b; }
static uint32_t up16(uint32_t x) { return (x + 15) & ~15u; }

/* ------------------------------------------------------- split passes --- */

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

void pk_quantize(pk_asr *a, pk_act *act, const float *x, size_t xs, uint32_t T, uint32_t n,
                 uint32_t qb) {
    act->n = n;
    act->qb = qb;
    act->T = T;
    quant_job j = { act, x, xs, 16 };
    pk_par(a, quant_item, &j, (int)((T + j.step - 1) / j.step));
}

/* What an item of a product does next with the tile it has just made,
 * rows [r0, r0 + nr) of the output's columns [c0, c0 + nc), while the tile
 * is in its core's cache (the feed-forward's SiLU, the attention's panels),
 * on thread th */
typedef void (*tile_fn)(void *arg, uint32_t r0, uint32_t nr, uint32_t c0, uint32_t nc, int th);

typedef struct {
    float *out;
    size_t os;
    const pk_q8 *w;
    const float *bias;
    const pk_act *act;
    int add;
    uint32_t step, rb, cstep;
    tile_fn then;
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
static void gemm_then(pk_asr *a, float *out, size_t os, const pk_q8 *w, const float *bias,
                      const pk_act *act, int add, tile_fn then, void *then_arg) {
    const uint32_t nth = (uint32_t)pk_pool_active(a->pool);
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
    pk_par(a, gemm_item, &j, (int)(rb * cc));
}

void pk_gemm(pk_asr *a, float *out, size_t os, const pk_q8 *w, const float *bias,
             const pk_act *act, int add) {
    gemm_then(a, out, os, w, bias, act, add, NULL, NULL);
}

typedef struct {
    float *out;
    const float *x;
    const float *w, *b;
    uint32_t T, n;
    float eps;
} norm_job;

static void norm_item(void *arg, int i, int th) {
    (void)th;
    norm_job *j = arg;
    uint32_t t0 = (uint32_t)i * 16, nt = min32(16, j->T - t0);
    pk_k_layernorm(j->out + (size_t)t0 * j->n, j->n, j->x + (size_t)t0 * j->n, j->n, nt, j->n,
                   j->w, j->b, j->eps);
}

static void norm(pk_asr *a, float *out, const float *x, uint32_t T, const float *w, const float *b) {
    norm_job j = { out, x, w, b, T, a->m->d_model, a->m->ln_eps };
    pk_par(a, norm_item, &j, (int)((T + 15) / 16));
}

/* T columns of n inputs in the encoder's blocks, as act's shape */
static pk_act *shape(pk_act *act, uint32_t T, uint32_t n) {
    act->n = n;
    act->qb = QB;
    act->T = T;
    return act;
}

/* Inputs [k0, k0 + n) of column t of act, from the n floats at x: whole
 * blocks, each quantized alone, so the bytes and scales that quantizing
 * the whole column gives them. */
static void quant_part(pk_act *act, uint32_t t, uint32_t k0, uint32_t n, const float *x) {
    const size_t at = (size_t)t * act->n + k0;
    pk_act part = { act->q + at, act->d + at / act->qb, n, act->qb, 1 };
    pk_k_quantize(&part, x, n);
}

/* A layer norm straight into the int8 activations, a frame at a time
 * through a row of thread scratch: its output was read only by the
 * quantize after it, 3 MB at a minute written and read again, four times
 * a layer. */
typedef struct { pk_asr *a; const float *x, *w, *b; uint32_t T; } normq_job;

static void normq_item(void *arg, int i, int th) {
    normq_job *j = arg;
    pk_act *act = &j->a->act;
    float *row = j->a->tws[th].row;
    for (uint32_t t = (uint32_t)i * 16; t < min32((uint32_t)i * 16 + 16, j->T); t++) {
        pk_k_layernorm(row, act->n, j->x + (size_t)t * act->n, act->n, 1, act->n, j->w, j->b,
                       j->a->m->ln_eps);
        quant_part(act, t, 0, act->n, row);
    }
}

/* a->act for the T frames of n floats at x, normalized by w and b */
static void normq(pk_asr *a, const float *x, uint32_t T, uint32_t n, const float *w, const float *b) {
    normq_job j = { a, x, w, b, T };
    shape(&a->act, T, n);
    pk_par(a, normq_item, &j, (int)((T + 15) / 16));
}

/* The feed-forward's hidden activations SiLU'd and quantized by the items
 * of its up product, each over its own tile a frame at a time, while the
 * tile is in its core's cache: a slice of 64 or 256 of the hidden's rows
 * is whole blocks of 32. As a pass of their own after the product they
 * read every frame across all the slices, lines that every thread had
 * written, and on the 9950X at 16 threads, half of those in the other
 * CCD's caches, the pass's jobs took 76 us where sixteen threads' share
 * of one thread's was 16. Made by the items, with the attention's panels
 * and the convolution module's quantize (dw_item), a 16-thread run there
 * is 3.9 to 6.1% faster and the GB10's, on ten X925s in two clusters,
 * 7.6%; a Pi 5's 1%, and the other boards' and the 9950X's at one to
 * eight threads level (backend/avx2/CLIMB.md, 2026-10-28). A frame
 * goes through a row of thread scratch so that the tile is only read:
 * SiLU in place dirtied again the lines a Pi 5's 512 KB of L2 had already
 * written back, and its run took 1.3% longer than with the pass, an 8 GB
 * Pi 5's 4.6% (the same entry). */
typedef struct { pk_asr *a; pk_act *hid; } siluq_job;

static void siluq_tile(void *arg, uint32_t r0, uint32_t nr, uint32_t c0, uint32_t nc, int th) {
    siluq_job *j = arg;
    float *row = j->a->tws[th].row;
    for (uint32_t t = c0; t < c0 + nc; t++) {
        memcpy(row, j->a->u + (size_t)t * j->hid->n + r0, nr * sizeof *row);
        pk_k_silu(row, nr);
        quant_part(j->hid, t, r0, nr, row);
    }
}

typedef struct { float *out; const float *x; uint32_t T, D; } glu_job;

static void glu_item(void *arg, int i, int th) {
    (void)th;
    glu_job *j = arg;
    uint32_t t0 = (uint32_t)i * 16, nt = min32(16, j->T - t0);
    pk_k_glu(j->out + (size_t)t0 * j->D, j->D, j->x + (size_t)t0 * 2 * j->D, 2 * j->D, nt, j->D);
}

typedef struct { float *out; const float *x; const pk_layer *L; pk_act *act; uint32_t T, D, K; } dw_job;

/* 64 channels over every frame, and their int8 for pw2 while they are in
 * this core's cache: a frame's 64 are two whole blocks (backend/avx2/
 * CLIMB.md, 2026-10-28, the passes between the products). */
static void dw_item(void *arg, int i, int th) {
    (void)th;
    dw_job *j = arg;
    const uint32_t c0 = (uint32_t)i * 64, nc = min32(64, j->D - c0);
    pk_k_dwconv(j->out + c0, j->D, j->x + c0, j->D, j->T, nc, j->L->dw + (size_t)c0 * j->K,
                j->L->dw_b + c0, j->K);
    for (uint32_t t = 0; t < j->T; t++)
        quant_part(j->act, t, c0, nc, j->out + (size_t)t * j->D + c0);
}

/* -------------------------------------------------------- subsampling --- */

typedef struct { pk_asr *a; uint32_t T0, T1, T2, T3; } sub_job;

static void relu(float *x, size_t n) {
    for (size_t i = 0; i < n; i++)
        x[i] = x[i] > 0 ? x[i] : 0;
}

/* A window of rows of one stage: the rows [lo, hi) of an image n floats a
 * row, from buf on. Sliding it to start at row lo2 keeps the rows it still
 * holds from there, moved down to the front. */
typedef struct { float *buf; size_t n; uint32_t lo, hi; } rows_win;

static void slide(rows_win *r, uint32_t lo2) {
    if (r->hi > lo2)
        memmove(r->buf, r->buf + (size_t)(lo2 - r->lo) * r->n, (size_t)(r->hi - lo2) * r->n * sizeof(float));
    else
        r->hi = lo2;
    r->lo = lo2;
}

/* PK_SUB_BLOCK encoder frames [b0, b1) through the whole subsampling, on
 * one thread, an output row at a time: a stride-2 tap set reads rows 2t-1
 * .. 2t+1, so each stage keeps a window of the three rows the next reads,
 * one of them the window before's, and computes the rest. The stages'
 * windows are 192 KB and 96 KB at 256 channels, which stay in L2; a block
 * of each whole (2.3 MB, then 557 KB twice) did not, and four threads were
 * then held up on memory, every stage taking two or three times what it
 * took on one (backend/neon-dp/CLIMB.md). The rows a block reads overlap
 * the neighbouring blocks' by one or two, computed twice; a row is the same
 * bits whichever block computes it, and every block writes only its own
 * rows of s5. */
static void sub_item(void *arg, int item, int th) {
    sub_job *j = arg;
    pk_asr *a = j->a;
    const pk_model *m = a->m;
    pk_thread_ws *w = &a->tws[th];
    const uint32_t C = m->sub_channels, F0 = m->n_mels, F1 = F0 / 2, F2 = F0 / 4, F3 = F0 / 8;
    const uint32_t b0 = (uint32_t)item * PK_SUB_BLOCK, b1 = min32(b0 + PK_SUB_BLOCK, j->T3);
    rows_win s1 = { w->s1, (size_t)F1 * C, 0, 0 }, s3 = { w->s3, (size_t)F2 * C, 0, 0 };
    pk_act act = { w->aq, w->ad, C, QB, F2 };

    for (uint32_t t = b0; t < b1; t++) {
        slide(&s3, t ? 2 * t - 1 : 0);
        for (const uint32_t hi3 = min32(2 * t + 2, j->T2); s3.hi < hi3; s3.hi++) {
            const uint32_t u = s3.hi, hi1 = min32(2 * u + 2, j->T1);
            slide(&s1, u ? 2 * u - 1 : 0);
            pk_k_conv3x3s2(s1.buf + (size_t)(s1.hi - s1.lo) * s1.n, a->feat, 0, j->T0, F0, C, 0,
                           m->conv0_w, m->conv0_b, 1, s1.hi, hi1);
            s1.hi = hi1;
            pk_k_conv3x3s2(w->s2, s1.buf, s1.lo, j->T1, F1, C, 1, m->dw1_w, m->dw1_b, 0, u, u + 1);
            pk_k_quantize(&act, w->s2, C);
            float *o3 = s3.buf + (size_t)(u - s3.lo) * s3.n;
            pk_k_gemm_q8(o3, C, m->sub_pw1.q, m->sub_pw1.s, m->pw1_b, C, &act, 0);
            relu(o3, s3.n);
        }
        pk_k_conv3x3s2(w->s4 + (size_t)(t - b0) * F3 * C, s3.buf, s3.lo, j->T2, F2, C, 1,
                       m->dw2_w, m->dw2_b, 0, t, t + 1);
    }
    act.T = (b1 - b0) * F3;
    pk_k_quantize(&act, w->s4, C);
    /* row t of s5 is frame t's [F3][C], which is sub.out's input order */
    float *out = a->s5 + (size_t)b0 * F3 * C;
    pk_k_gemm_q8(out, C, m->sub_pw2.q, m->sub_pw2.s, m->pw2_b, C, &act, 0);
    relu(out, (size_t)(b1 - b0) * F3 * C);
}

/* ----------------------------------------------------------- positions --- */

typedef struct { pk_asr *a; uint32_t T; } pos_job;

/* Row i of pe is the relative position T-1-i: sin of position times each
 * frequency in the even columns, cos in the odd, the product a float32
 * multiply as NeMo forms it and the sine core/vmath.h's. */
static void pos_item(void *arg, int i, int th) {
    (void)th;
    pos_job *j = arg;
    const pk_model *m = j->a->m;
    const uint32_t D = m->d_model, R = 2 * j->T - 1;
    for (uint32_t r = (uint32_t)i * 16; r < min32((uint32_t)i * 16 + 16, R); r++) {
        float pos = (float)((int64_t)j->T - 1 - r), *row = j->a->pe + (size_t)r * D;
        for (uint32_t k = 0; k < D / 2; k++)
            pk_sincosf(pos * m->pos_freq[k], &row[2 * k], &row[2 * k + 1]);
    }
}

/* ----------------------------------------------------------- attention --- */

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

typedef struct {
    pk_asr *a;
    const pk_layer *L;
    uint32_t T, Tp, Pp;
} att_job;

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
typedef struct { float *dst; const float *src; uint32_t n, np, D, dk; } panels_job;

static void panels_tile(void *arg, uint32_t r0, uint32_t nr, uint32_t c0, uint32_t nc, int th) {
    (void)th;
    panels_job *j = arg;
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

/* One head over PK_ATT_BLOCK queries [i0, i1):
 *
 *   score(i, j) = ((q_i + u) . k_j + (q_i + v) . p_{T-1-i+j}) / sqrt(dk)
 *
 * Transformer-XL's attention, as NeMo's rel_shift lays it out: query i
 * against key j takes the position embedding of i - j, row T-1-i+j of the
 * table. The block's queries need the rows from T-i1 to 2T-2-i0, T+15 of
 * them, so the position products are taken over that window only (from a
 * panel boundary) rather than over all 2T-1 rows for every query, which is
 * half the attention's float work at T rows. Then the softmax over the T
 * keys, and the weighted sum of the values. */
static void att_item(void *arg, int item, int th) {
    att_job *j = arg;
    pk_asr *a = j->a;
    const uint32_t D = a->m->d_model, H = a->m->n_heads, dk = D / H, T = j->T;
    const uint32_t nb = (T + PK_ATT_BLOCK - 1) / PK_ATT_BLOCK;
    const uint32_t h = (uint32_t)item / nb, i0 = ((uint32_t)item % nb) * PK_ATT_BLOCK;
    const uint32_t i1 = min32(i0 + PK_ATT_BLOCK, T), nq = i1 - i0;
    pk_thread_ws *w = &a->tws[th];
    const float *kp = a->kp + (size_t)h * j->Tp * dk, *pp = a->pp + (size_t)h * j->Pp * dk;
    const float *bu = j->L->bias_u + h * dk, *bv = j->L->bias_v + h * dk;

    /* the block's queries with each position bias, a query a row */
    for (uint32_t i = 0; i < nq; i++) {
        const float *q = a->q + (size_t)(i0 + i) * D + h * dk;
        for (uint32_t d = 0; d < dk; d++) {
            w->qu[i * dk + d] = q[d] + bu[d];
            w->qv[i * dk + d] = q[d] + bv[d];
        }
    }
    pk_k_gemm_f32(w->sc, j->Tp, kp, dk, j->Tp, dk, w->qu, dk, nq, NULL, 0, 0);
    const uint32_t m_lo = T - i1, m_hi = 2 * T - 2 - i0, m_a = m_lo & ~15u;
    const uint32_t rows = up16(m_hi + 1 - m_a);
    pk_k_gemm_f32(w->bd, rows, pp + (size_t)(m_a / 16) * dk * 16, dk, rows, dk, w->qv, dk, nq,
                  NULL, 0, 0);
    const float scale = 1.0f / sqrtf((float)dk);
    for (uint32_t i = i0; i < i1; i++) {
        float *s = w->sc + (size_t)(i - i0) * j->Tp;
        const float *b = w->bd + (size_t)(i - i0) * rows + (T - 1 - i) - m_a;
        for (uint32_t k = 0; k < T; k++)
            s[k] = s[k] + b[k];
        pk_k_softmax(s, T, scale);
    }
    pk_k_gemm_f32(a->o + (size_t)i0 * D + h * dk, D, a->vp + (size_t)h * (dk / 16) * T * 16, T, dk,
                  T, w->sc, j->Tp, nq, NULL, 0, 0);
}

/* --------------------------------------------------------------- layers --- */

static void ff(pk_asr *a, uint32_t T, const float *nw, const float *nb, const pk_q8 *up,
               const pk_q8 *down) {
    const uint32_t D = a->m->d_model, F = a->m->ff_dim;
    siluq_job sj = { a, shape(&a->hid, T, F) };
    PROF(PK_P_QUANT, normq(a, a->x, T, D, nw, nb));
    PROF(PK_P_FF, gemm_then(a, a->u, F, up, NULL, &a->act, 0, siluq_tile, &sj);
                  pk_gemm(a, a->x, D, down, NULL, &a->hid, 1));     /* the 0.5 is in down */
}

static void layer(pk_asr *a, const pk_layer *L, uint32_t T) {
    const pk_model *m = a->m;
    const uint32_t D = m->d_model, H = m->n_heads, dk = D / H, R = 2 * T - 1;
    const uint32_t Tp = up16(T), Pp = up16(R) + 16;

    ff(a, T, L->ff1_nw, L->ff1_nb, &L->ff1_up, &L->ff1_down);

    PROF(PK_P_QUANT, normq(a, a->x, T, D, L->att_nw, L->att_nb));
    /* K's floats in a->o, free until the attention writes it, and P's in
     * a->p, each tile there on its way into the panels */
    panel_job vj = { a->vp, (size_t)T * 16, &L->v, &a->act };
    panels_job kj = { a->kp, a->o, T, Tp, D, dk }, pj = { a->pp, a->p, R, Pp, D, dk };
    PROF(PK_P_QKV, pk_gemm(a, a->q, D, &L->q, NULL, &a->act, 0);
                   gemm_then(a, a->o, D, &L->k, NULL, &a->act, 0, panels_tile, &kj);
                   pk_par(a, panel_item, &vj, (int)((L->v.rows + 63) / 64));
                   gemm_then(a, a->p, D, &L->pos, NULL, &a->ape, 0, panels_tile, &pj));
    att_job aj = { a, L, T, Tp, Pp };
    PROF(PK_P_ATT, pk_par(a, att_item, &aj, (int)(H * ((T + PK_ATT_BLOCK - 1) / PK_ATT_BLOCK))));
    PROF(PK_P_QUANT, pk_quantize(a, &a->act, a->o, D, T, D, QB));
    PROF(PK_P_ATTO, pk_gemm(a, a->x, D, &L->o, NULL, &a->act, 1));

    PROF(PK_P_QUANT, normq(a, a->x, T, D, L->conv_nw, L->conv_nb));
    PROF(PK_P_CONV, pk_gemm(a, a->u, 2 * D, &L->pw1, NULL, &a->act, 0));
    glu_job gj = { a->h, a->u, T, D };
    dw_job dj = { a->o, a->h, L, shape(&a->act, T, D), T, D, m->conv_kernel };
    PROF(PK_P_DW, pk_par(a, glu_item, &gj, (int)((T + 15) / 16));
                  pk_par(a, dw_item, &dj, (int)((D + 63) / 64)));
    PROF(PK_P_CONV, pk_gemm(a, a->x, D, &L->pw2, NULL, &a->act, 1));

    ff(a, T, L->ff2_nw, L->ff2_nb, &L->ff2_up, &L->ff2_down);
    PROF(PK_P_NORM, norm(a, a->x, a->x, T, L->out_nw, L->out_nb));
}

/* Every core, or the fast kind alone (asr.h)? The second recording long
 * enough to time, PK_TUNE_FRAMES, runs its layers on each by turns, two at
 * a time, and the transcriber keeps whichever's layers took less, each side
 * by the median of its own. The first such recording is the first pass
 * over the int8 weights, whose pages are mapped in as each layer runs: an
 * M4 Pro's layers took 3 to 4 times their warm time there, the model read
 * from disk. Taking turns puts the board's slow changes on both sides
 * alike, and the median leaves out what lands on a few layers: a
 * recording's setup, and stalls like a Dimensity 8300's, which took one
 * layer in seven three to ten times its neighbours' time. The layers are
 * the same shapes, so their times compare, and the bits are the same on
 * any threads. Neither kind wins on every chip: a GB10's ten X925s ran the
 * encoder faster than its twenty cores, a Core 3 304's one P-core four
 * times slower than its five. The race kept the X925s in 35 runs of
 * 36, an M4 Pro level with the faster of 8 and 12 threads in every round,
 * and the Dimensity's four A715s, 2 to 16% ahead of its eight cores, in 11
 * of 15 (backend/neon-dp/CLIMB.md, 2026-10-27 and before;
 * backend/avx2/CLIMB.md). */
#define PK_TUNE_FRAMES 100       /* 8 s */
#define PK_TUNE_LAYERS 8         /* the fewest raced: two turns a side */
#define PK_TUNE_SIDE   32        /* the most layers a side runs */

/* The median of t[0..n), n > 0, which it sorts */
static double median(double *t, uint32_t n) {
    for (uint32_t i = 1; i < n; i++)
        for (uint32_t j = i; j > 0 && t[j - 1] > t[j]; j--) {
            double x = t[j];
            t[j] = t[j - 1];
            t[j - 1] = x;
        }
    return (t[(n - 1) / 2] + t[n / 2]) / 2;
}

static void layers(pk_asr *a, uint32_t T) {
    const pk_model *m = a->m;
    /* in turns of two, as many on each side */
    const uint32_t raced = m->n_layers / 4 * 4 < 2 * PK_TUNE_SIDE ? m->n_layers / 4 * 4 : 2 * PK_TUNE_SIDE;
    uint32_t i = 0;
    if (a->tune && T >= PK_TUNE_FRAMES && raced >= PK_TUNE_LAYERS && --a->tune == 0) {
        double took[2][PK_TUNE_SIDE];    /* every core's layers, the fast kind's */
        uint32_t n[2] = { 0, 0 };
        /* the chunk before's search, beside this encoder (asr.c), done
         * first, so that each side has the threads it names */
        pk_pool_join(a->pool);
        for (; i < raced; i++) {
            const int side = (int)(i / 2) % 2;
            if (i % 2 == 0)
                pk_pool_set_active(a->pool, side ? a->fast : a->nth);
            double t0 = pk_os_now();
            layer(a, &m->layers[i], T);
            took[side][n[side]++] = pk_os_now() - t0;
        }
        a->wide = median(took[1], n[1]) < median(took[0], n[0]) ? a->fast : a->nth;
        pk_pool_set_active(a->pool, a->wide);
    }
    for (; i < m->n_layers; i++)
        layer(a, &m->layers[i], T);
}

uint32_t pk_encode(pk_asr *a, uint32_t T0) {
    const pk_model *m = a->m;
    const uint32_t T1 = (T0 + 1) / 2, T2 = (T1 + 1) / 2, T3 = (T2 + 1) / 2;
    const uint32_t C = m->sub_channels, F3 = m->n_mels / 8, D = m->d_model;
    if (!T0)
        return 0;
    sub_job sj = { a, T0, T1, T2, T3 };
    PROF(PK_P_SUB, pk_par(a, sub_item, &sj, (int)((T3 + PK_SUB_BLOCK - 1) / PK_SUB_BLOCK));
                   pk_quantize(a, &a->act, a->s5, (size_t)F3 * C, T3, F3 * C, QB);
                   pk_gemm(a, a->x, D, &m->sub_out, m->out_b, &a->act, 0));

    pos_job pj = { a, T3 };
    PROF(PK_P_POS, pk_par(a, pos_item, &pj, (int)((2 * T3 - 1 + 15) / 16));
                   pk_quantize(a, &a->ape, a->pe, D, 2 * T3 - 1, D, QB));

    layers(a, T3);
    return T3;
}
