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

typedef struct {
    float *out;
    size_t os;
    const pk_q8 *w;
    const float *bias;
    const pk_act *act;
    int add;
    uint32_t step, rb, cstep;
} gemm_job;

static void gemm_item(void *arg, int i, int th) {
    (void)th;
    gemm_job *j = arg;
    uint32_t r0 = ((uint32_t)i % j->rb) * j->step, nr = min32(j->step, j->w->rows - r0);
    uint32_t c0 = ((uint32_t)i / j->rb) * j->cstep, nc = min32(j->cstep, j->act->T - c0);
    pk_act sub = *j->act;
    sub.q += (size_t)c0 * sub.n;
    sub.d += (size_t)c0 * (sub.n / sub.qb);
    sub.T = nc;
    pk_k_gemm_q8(j->out + (size_t)c0 * j->os + r0, j->os, j->w->q + (size_t)r0 * j->w->cols,
                 j->w->s + r0, j->bias ? j->bias + r0 : NULL, nr, &sub, j->add);
}

/* Rows in slices of 64: few enough that each streams its weights through
 * once for its columns, and (pool item sizes, backend/avx2/CLIMB.md) the
 * size that measured best against 32 and 128. And the columns in chunks
 * when that fills the pool's last round: a 1024-row matrix is 16 slices,
 * which six threads take in three rounds with the last a third idle, and in
 * three chunks of columns 48 items, eight whole rounds (H5 there). A chunk
 * is whole tiles of 3 and 4 columns, and at least 48 columns, so that a
 * call's per-call work - the backend's copy of its columns, its panels'
 * sums - stays a few percent of it. Every output is still one call's, so
 * the bits do not depend on the split. */
void pk_gemm(pk_asr *a, float *out, size_t os, const pk_q8 *w, const float *bias,
             const pk_act *act, int add) {
    const uint32_t step = 64, rb = (w->rows + step - 1) / step, nth = (uint32_t)pk_pool_active(a->pool);
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
    gemm_job j = { out, os, w, bias, act, add, step, rb, cstep ? cstep : 1 };
    pk_par(a, gemm_item, &j, (int)(rb * cc));
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

/* The feed-forward's hidden activations SiLU'd and quantized in one pass,
 * a frame at a time through a row of thread scratch: the [T][F] array
 * (12 MB at a minute) is read once and never written back, where a SiLU
 * pass in place and then pk_quantize read it twice and wrote it once, and
 * on a Pi 5 that traffic, not the arithmetic, set the pace. The values
 * quantized are the same floats. */
typedef struct { pk_asr *a; const float *x, *w, *b; uint32_t T; } rowq_job;

/* row t of the int8 activations from the n floats at x */
static void quant_row(pk_act *act, uint32_t t, const float *x) {
    const uint32_t nb = act->n / act->qb;
    pk_act one = { act->q + (size_t)t * act->n, act->d + (size_t)t * nb, act->n, act->qb, 1 };
    pk_k_quantize(&one, x, act->n);
}

static void siluq_item(void *arg, int i, int th) {
    rowq_job *j = arg;
    pk_act *act = &j->a->act;
    float *row = j->a->tws[th].row;
    for (uint32_t t = (uint32_t)i * 16; t < min32((uint32_t)i * 16 + 16, j->T); t++) {
        memcpy(row, j->x + (size_t)t * act->n, act->n * sizeof *row);
        pk_k_silu(row, act->n);
        quant_row(act, t, row);
    }
}

/* A layer norm straight into the int8 activations, the same way: its
 * output was read only by the quantize after it, 3 MB at a minute written
 * and read again, four times a layer. */
static void normq_item(void *arg, int i, int th) {
    rowq_job *j = arg;
    pk_act *act = &j->a->act;
    float *row = j->a->tws[th].row;
    for (uint32_t t = (uint32_t)i * 16; t < min32((uint32_t)i * 16 + 16, j->T); t++) {
        pk_k_layernorm(row, act->n, j->x + (size_t)t * act->n, act->n, 1, act->n, j->w, j->b,
                       j->a->m->ln_eps);
        quant_row(act, t, row);
    }
}

/* a->act for T frames of n floats at x, through item, which takes 16 */
static void rowq(pk_asr *a, void (*item)(void *, int, int), const float *x, uint32_t T,
                 uint32_t n, const float *w, const float *b) {
    a->act.n = n;
    a->act.qb = QB;
    a->act.T = T;
    rowq_job j = { a, x, w, b, T };
    pk_par(a, item, &j, (int)((T + 15) / 16));
}

typedef struct { float *out; const float *x; uint32_t T, D; } glu_job;

static void glu_item(void *arg, int i, int th) {
    (void)th;
    glu_job *j = arg;
    uint32_t t0 = (uint32_t)i * 16, nt = min32(16, j->T - t0);
    pk_k_glu(j->out + (size_t)t0 * j->D, j->D, j->x + (size_t)t0 * 2 * j->D, 2 * j->D, nt, j->D);
}

typedef struct { float *out; const float *x; const pk_layer *L; uint32_t T, D, K; } dw_job;

static void dw_item(void *arg, int i, int th) {
    (void)th;
    dw_job *j = arg;
    uint32_t c0 = (uint32_t)i * 64;
    pk_k_dwconv(j->out + c0, j->D, j->x + c0, j->D, j->T, min32(64, j->D - c0),
                j->L->dw + (size_t)c0 * j->K, j->L->dw_b + c0, j->K);
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
    const float *k;          /* K, [T][D], as its product wrote it */
    uint32_t T, Tp, Pp;
} att_job;

/* K and the position table as pk_k_gemm_f32 panels, a head and 64 frames
 * (and 128 rows of the table) an item:
 *
 *   kp: (d, key j)      at ((j/16) dk + d) 16 + j%16     keys padded to Tp with zeros
 *   pp: (d, position m) at ((m/16) dk + d) 16 + m%16     positions padded to Pp
 *
 * Each is a transpose, so a panel's sixteen rows are read while a
 * dimension's sixteen values are written as one run: the rows stay in L1
 * and every store lands beside the last. A row at a time put each store a
 * cache line from the one before, and that copy, with V's and the queries'
 * beside it, was 5.8% of a Pi 5's minute; V now needs none (panel_item),
 * the queries are made a block at a time where they are used (att_item),
 * and what is left is 2.2%. Taking K and P as their products leave them
 * and transposing each block's scores instead, which grows with T squared,
 * was slower: 5.08 s a minute where this is 4.75 (backend/neon-dp/CLIMB.md). */
#define PK_REPACK 64

static void panels_t(float *dst, const float *src, size_t ld, uint32_t r0, uint32_t r1,
                     uint32_t n, uint32_t dk) {
    for (uint32_t p0 = r0; p0 < r1; p0 += 16) {
        const float *rows[16];
        for (uint32_t i = 0; i < 16; i++)
            rows[i] = p0 + i < n ? src + (size_t)(p0 + i) * ld : NULL;
        float *d0 = dst + (size_t)(p0 / 16) * dk * 16;
        for (uint32_t d = 0; d < dk; d++)
            for (uint32_t i = 0; i < 16; i++)
                d0[d * 16 + i] = rows[i] ? rows[i][d] : 0.0f;
    }
}

static void repack_item(void *arg, int item, int th) {
    (void)th;
    att_job *j = arg;
    pk_asr *a = j->a;
    const uint32_t D = a->m->d_model, dk = D / a->m->n_heads, T = j->T, R = 2 * T - 1;
    const uint32_t nb = (T + PK_REPACK - 1) / PK_REPACK, h = (uint32_t)item / nb, b = (uint32_t)item % nb;
    const uint32_t t0 = b * PK_REPACK, t1 = b + 1 == nb ? j->Tp : t0 + PK_REPACK;
    const uint32_t r0 = 2 * t0, r1 = b + 1 == nb ? j->Pp : r0 + 2 * PK_REPACK;
    panels_t(a->kp + (size_t)h * j->Tp * dk, j->k + h * dk, D, t0, t1, T, dk);
    panels_t(a->pp + (size_t)h * j->Pp * dk, a->p + h * dk, D, r0, r1, R, dk);
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
    PROF(PK_P_QUANT, rowq(a, normq_item, a->x, T, D, nw, nb));
    PROF(PK_P_FF, pk_gemm(a, a->u, F, up, NULL, &a->act, 0));
    PROF(PK_P_QUANT, rowq(a, siluq_item, a->u, T, F, NULL, NULL));
    PROF(PK_P_FF, pk_gemm(a, a->x, D, down, NULL, &a->act, 1));     /* the 0.5 is in down */
}

static void layer(pk_asr *a, const pk_layer *L, uint32_t T) {
    const pk_model *m = a->m;
    const uint32_t D = m->d_model, H = m->n_heads;
    float *k = a->o;                     /* free until the attention writes it */

    ff(a, T, L->ff1_nw, L->ff1_nb, &L->ff1_up, &L->ff1_down);

    PROF(PK_P_QUANT, rowq(a, normq_item, a->x, T, D, L->att_nw, L->att_nb));
    panel_job vj = { a->vp, (size_t)T * 16, &L->v, &a->act };
    PROF(PK_P_QKV, pk_gemm(a, a->q, D, &L->q, NULL, &a->act, 0);
                   pk_gemm(a, k, D, &L->k, NULL, &a->act, 0);
                   pk_par(a, panel_item, &vj, (int)((L->v.rows + 63) / 64));
                   pk_gemm(a, a->p, D, &L->pos, NULL, &a->ape, 0));
    att_job aj = { a, L, k, T, up16(T), up16(2 * T - 1) + 16 };
    PROF(PK_P_REPACK, pk_par(a, repack_item, &aj, (int)(H * ((T + PK_REPACK - 1) / PK_REPACK))));
    PROF(PK_P_ATT, pk_par(a, att_item, &aj, (int)(H * ((T + PK_ATT_BLOCK - 1) / PK_ATT_BLOCK))));
    PROF(PK_P_QUANT, pk_quantize(a, &a->act, a->o, D, T, D, QB));
    PROF(PK_P_ATTO, pk_gemm(a, a->x, D, &L->o, NULL, &a->act, 1));

    PROF(PK_P_QUANT, rowq(a, normq_item, a->x, T, D, L->conv_nw, L->conv_nb));
    PROF(PK_P_CONV, pk_gemm(a, a->u, 2 * D, &L->pw1, NULL, &a->act, 0));
    glu_job gj = { a->h, a->u, T, D };
    dw_job dj = { a->o, a->h, L, T, D, m->conv_kernel };
    PROF(PK_P_DW, pk_par(a, glu_item, &gj, (int)((T + 15) / 16));
                  pk_par(a, dw_item, &dj, (int)((D + 63) / 64)));
    PROF(PK_P_QUANT, pk_quantize(a, &a->act, a->o, D, T, D, QB));
    PROF(PK_P_CONV, pk_gemm(a, a->x, D, &L->pw2, NULL, &a->act, 1));

    ff(a, T, L->ff2_nw, L->ff2_nb, &L->ff2_up, &L->ff2_down);
    PROF(PK_P_NORM, norm(a, a->x, a->x, T, L->out_nw, L->out_nb));
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

    for (uint32_t i = 0; i < m->n_layers; i++)
        layer(a, &m->layers[i], T3);
    return T3;
}
