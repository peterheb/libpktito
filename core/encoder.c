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
    uint32_t step;
} gemm_job;

static void gemm_item(void *arg, int i, int th) {
    (void)th;
    gemm_job *j = arg;
    uint32_t r0 = (uint32_t)i * j->step, nr = min32(j->step, j->w->rows - r0);
    pk_k_gemm_q8(j->out + r0, j->os, j->w->q + (size_t)r0 * j->w->cols, j->w->s + r0,
                 j->bias ? j->bias + r0 : NULL, nr, j->act, j->add);
}

/* Rows in blocks of 64: enough items to balance a pool of a dozen, few
 * enough that each streams its weights through once for all the columns. */
void pk_gemm(pk_asr *a, float *out, size_t os, const pk_q8 *w, const float *bias,
             const pk_act *act, int add) {
    gemm_job j = { out, os, w, bias, act, add, 64 };
    pk_par(a, gemm_item, &j, (int)((w->rows + j.step - 1) / j.step));
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

typedef struct { float *x; size_t n, step; } silu_job;

static void silu_item(void *arg, int i, int th) {
    (void)th;
    silu_job *j = arg;
    size_t i0 = (size_t)i * j->step;
    pk_k_silu(j->x + i0, j->n - i0 < j->step ? j->n - i0 : j->step);
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

/* PK_SUB_BLOCK encoder frames [b0, b1) through the whole subsampling, on
 * one thread: each stage computes the rows the next stage's taps reach,
 * which overlap the neighbouring blocks' by a row or two, the cost of
 * keeping a block's buffers small. A row is the same bits whichever block
 * computes it, and every block writes only its own rows of s5. */
static void sub_item(void *arg, int item, int th) {
    sub_job *j = arg;
    pk_asr *a = j->a;
    const pk_model *m = a->m;
    pk_thread_ws *w = &a->tws[th];
    const uint32_t C = m->sub_channels, F0 = m->n_mels;
    const uint32_t F1 = (F0 + 1) / 2, F2 = (F1 + 1) / 2, F3 = (F2 + 1) / 2;
    const uint32_t b0 = (uint32_t)item * PK_SUB_BLOCK, b1 = min32(b0 + PK_SUB_BLOCK, j->T3);
    /* a stride-2 tap set reads rows 2t-1 .. 2t+1 */
    const uint32_t r2a = b0 ? 2 * b0 - 1 : 0, r2b = min32(2 * b1, j->T2);
    const uint32_t r1a = r2a ? 2 * r2a - 1 : 0, r1b = min32(2 * r2b, j->T1);

    pk_k_conv3x3s2(w->s1, a->feat, 0, j->T0, F0, C, 0, m->conv0_w, m->conv0_b, 1, r1a, r1b);
    pk_k_conv3x3s2(w->s2, w->s1, r1a, j->T1, F1, C, 1, m->dw1_w, m->dw1_b, 0, r2a, r2b);
    pk_act act = { w->aq, w->ad, C, QB, (r2b - r2a) * F2 };
    pk_k_quantize(&act, w->s2, C);
    pk_k_gemm_q8(w->s3, C, m->sub_pw1.q, m->sub_pw1.s, m->pw1_b, C, &act, 0);
    relu(w->s3, (size_t)(r2b - r2a) * F2 * C);
    pk_k_conv3x3s2(w->s4, w->s3, r2a, j->T2, F2, C, 1, m->dw2_w, m->dw2_b, 0, b0, b1);
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

typedef struct {
    pk_asr *a;
    const pk_layer *L;
    const float *k, *v;
    uint32_t T, Tp, Pp;
} att_job;

/* Head i's K and V and P as pk_k_gemm_f32 panels, and its queries with the
 * two position biases added:
 *
 *   kp: (d, key j)      at ((j/16) dk + d) 16 + j%16     keys padded to Tp with zeros
 *   vp: (key j, d)      at ((d/16) Tp + j) 16 + d%16
 *   pp: (d, position m) at ((m/16) dk + d) 16 + m%16     positions padded to Pp */
static void repack_item(void *arg, int h, int th) {
    (void)th;
    att_job *j = arg;
    pk_asr *a = j->a;
    const uint32_t D = a->m->d_model, dk = D / a->m->n_heads, T = j->T, R = 2 * T - 1;
    float *kp = a->kp + (size_t)h * j->Tp * dk, *vp = a->vp + (size_t)h * j->Tp * dk;
    float *pp = a->pp + (size_t)h * j->Pp * dk;
    memset(kp, 0, (size_t)j->Tp * dk * sizeof *kp);
    memset(vp, 0, (size_t)j->Tp * dk * sizeof *vp);
    memset(pp, 0, (size_t)j->Pp * dk * sizeof *pp);
    for (uint32_t t = 0; t < T; t++) {
        const float *kr = j->k + (size_t)t * D + h * dk, *vr = j->v + (size_t)t * D + h * dk;
        for (uint32_t d = 0; d < dk; d++) {
            kp[((size_t)(t / 16) * dk + d) * 16 + t % 16] = kr[d];
            vp[((size_t)(d / 16) * j->Tp + t) * 16 + d % 16] = vr[d];
        }
        const float *qr = a->q + (size_t)t * D + h * dk;
        float *qu = a->qu + (size_t)t * D + h * dk, *qv = a->qv + (size_t)t * D + h * dk;
        for (uint32_t d = 0; d < dk; d++) {
            qu[d] = qr[d] + j->L->bias_u[h * dk + d];
            qv[d] = qr[d] + j->L->bias_v[h * dk + d];
        }
    }
    for (uint32_t r = 0; r < R; r++) {
        const float *pr = a->p + (size_t)r * D + h * dk;
        for (uint32_t d = 0; d < dk; d++)
            pp[((size_t)(r / 16) * dk + d) * 16 + r % 16] = pr[d];
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
    float *sc = a->tws[th].sc, *bd = a->tws[th].bd;
    const float *kp = a->kp + (size_t)h * j->Tp * dk, *vp = a->vp + (size_t)h * j->Tp * dk;
    const float *pp = a->pp + (size_t)h * j->Pp * dk;

    pk_k_gemm_f32(sc, j->Tp, kp, dk, j->Tp, dk, a->qu + (size_t)i0 * D + h * dk, D, nq, NULL, 0, 0);
    const uint32_t m_lo = T - i1, m_hi = 2 * T - 2 - i0, m_a = m_lo & ~15u;
    const uint32_t rows = up16(m_hi + 1 - m_a);
    pk_k_gemm_f32(bd, rows, pp + (size_t)(m_a / 16) * dk * 16, dk, rows, dk,
                  a->qv + (size_t)i0 * D + h * dk, D, nq, NULL, 0, 0);
    const float scale = 1.0f / sqrtf((float)dk);
    for (uint32_t i = i0; i < i1; i++) {
        float *s = sc + (size_t)(i - i0) * j->Tp;
        const float *b = bd + (size_t)(i - i0) * rows + (T - 1 - i) - m_a;
        for (uint32_t k = 0; k < T; k++)
            s[k] = s[k] + b[k];
        pk_k_softmax(s, T, scale);
    }
    pk_k_gemm_f32(a->o + (size_t)i0 * D + h * dk, D, vp, j->Tp, dk, T, sc, j->Tp, nq, NULL, 0, 0);
}

/* --------------------------------------------------------------- layers --- */

static void ff(pk_asr *a, uint32_t T, const float *nw, const float *nb, const pk_q8 *up,
               const pk_q8 *down) {
    const uint32_t D = a->m->d_model, F = a->m->ff_dim;
    norm(a, a->h, a->x, T, nw, nb);
    pk_quantize(a, &a->act, a->h, D, T, D, QB);
    pk_gemm(a, a->u, F, up, NULL, &a->act, 0);
    silu_job sj = { a->u, (size_t)T * F, 16384 };
    pk_par(a, silu_item, &sj, (int)((sj.n + sj.step - 1) / sj.step));
    pk_quantize(a, &a->act, a->u, F, T, F, QB);
    pk_gemm(a, a->x, D, down, NULL, &a->act, 1);       /* the 0.5 is in down */
}

static void layer(pk_asr *a, const pk_layer *L, uint32_t T) {
    const pk_model *m = a->m;
    const uint32_t D = m->d_model, H = m->n_heads, dk = D / H;
    float *k = a->o, *v = a->h;          /* free between their uses below */

    ff(a, T, L->ff1_nw, L->ff1_nb, &L->ff1_up, &L->ff1_down);

    norm(a, a->u, a->x, T, L->att_nw, L->att_nb);
    pk_quantize(a, &a->act, a->u, D, T, D, QB);
    pk_gemm(a, a->q, D, &L->q, NULL, &a->act, 0);
    pk_gemm(a, k, D, &L->k, NULL, &a->act, 0);
    pk_gemm(a, v, D, &L->v, NULL, &a->act, 0);
    pk_gemm(a, a->p, D, &L->pos, NULL, &a->ape, 0);
    att_job aj = { a, L, k, v, T, up16(T), up16(2 * T - 1) + 16 };
    pk_par(a, repack_item, &aj, (int)H);
    pk_par(a, att_item, &aj, (int)(H * ((T + PK_ATT_BLOCK - 1) / PK_ATT_BLOCK)));
    pk_quantize(a, &a->act, a->o, D, T, D, QB);
    pk_gemm(a, a->x, D, &L->o, NULL, &a->act, 1);
    (void)dk;

    norm(a, a->h, a->x, T, L->conv_nw, L->conv_nb);
    pk_quantize(a, &a->act, a->h, D, T, D, QB);
    pk_gemm(a, a->u, 2 * D, &L->pw1, NULL, &a->act, 0);
    glu_job gj = { a->h, a->u, T, D };
    pk_par(a, glu_item, &gj, (int)((T + 15) / 16));
    dw_job dj = { a->o, a->h, L, T, D, m->conv_kernel };
    pk_par(a, dw_item, &dj, (int)((D + 63) / 64));
    pk_quantize(a, &a->act, a->o, D, T, D, QB);
    pk_gemm(a, a->x, D, &L->pw2, NULL, &a->act, 1);

    ff(a, T, L->ff2_nw, L->ff2_nb, &L->ff2_up, &L->ff2_down);
    norm(a, a->x, a->x, T, L->out_nw, L->out_nb);
}

uint32_t pk_encode(pk_asr *a, uint32_t T0) {
    const pk_model *m = a->m;
    const uint32_t T1 = (T0 + 1) / 2, T2 = (T1 + 1) / 2, T3 = (T2 + 1) / 2;
    const uint32_t C = m->sub_channels, F3 = m->n_mels / 8, D = m->d_model;
    if (!T0)
        return 0;
    sub_job sj = { a, T0, T1, T2, T3 };
    pk_par(a, sub_item, &sj, (int)((T3 + PK_SUB_BLOCK - 1) / PK_SUB_BLOCK));
    pk_quantize(a, &a->act, a->s5, (size_t)F3 * C, T3, F3 * C, QB);
    pk_gemm(a, a->x, D, &m->sub_out, m->out_b, &a->act, 0);

    pos_job pj = { a, T3 };
    pk_par(a, pos_item, &pj, (int)((2 * T3 - 1 + 15) / 16));
    pk_quantize(a, &a->ape, a->pe, D, 2 * T3 - 1, D, QB);

    for (uint32_t i = 0; i < m->n_layers; i++)
        layer(a, &m->layers[i], T3);
    return T3;
}
