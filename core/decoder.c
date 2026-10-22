/* The decoder: encoder frames in, tokens out, by TDT's greedy search.
 *
 * A Token-and-Duration Transducer (Xu et al. 2023, arXiv:2304.06795) reads
 * the encoder's frames with three networks:
 *
 *   prediction   an embedding of the last token and two LSTM layers: g, a
 *                summary of the text so far (blank stands for "none yet")
 *   joint        relu(enc_proj(f_t) + pred_proj(g)) through one matrix to
 *                vocab + 1 token logits and n_durations duration logits
 *   search       at frame t: the best token k and the best duration d; emit
 *                k unless it is blank, then move to frame t + d
 *
 * so it may emit several tokens at one frame (duration 0) and skip frames
 * that hold nothing (durations up to 4), where a plain transducer steps one
 * frame at a time. This is NeMo's label-looping greedy search
 * (nemo/collections/asr/parts/submodules/transducer_decoding/
 * tdt_label_looping.py), which model.transcribe() runs, at batch size one:
 *
 *   t = 0; g = prediction(blank, zero state)
 *   while t < T:
 *       repeat: k, d = joint(f_t, g); a blank of duration 0 counts as 1;
 *               t += d, until k is a token or t has run off the end
 *       if k is blank: stop
 *       emit k at the frame it was found; g = prediction(k)
 *       after max_symbols tokens found at one frame with t still there,
 *       step one frame
 *
 * The prediction network only moves when a token is emitted, so g and its
 * half of the joint are computed once a token, not once a step. Every
 * matrix is int8 with its activation quantized a row at a time (README,
 * "Why int8": that costs nothing measurable here). An argmax takes the
 * first of equal values, as torch's does.
 */
#include "asr.h"
#include "kernels.h"

#include <stdlib.h>
#include <string.h>

/* rows past which a decoder product is split over the pool: the joint's
 * output for an 8192-token vocabulary is 5.2M multiply-adds a step */
#define PK_DEC_SPLIT 2048

/* The threads the search's products are split over. A product is a few
 * microseconds' work a thread, and on every thread of a GB10's twenty it
 * waited on all of them, ten Cortex-A725s among them: the decoder took 183
 * ms of a run at 20 threads against 17 at 10, and 29 on one thread
 * (backend/neon-dp/CLIMB.md, 2026-10-22). */
#define PK_DEC_THREADS 4

static void gemv(pk_asr *a, float *out, const pk_q8 *w, const float *bias, const float *x,
                 int add) {
    pk_act *act = &a->dact;
    act->n = act->qb = w->cols;
    act->T = 1;
    pk_k_quantize(act, x, w->cols);
    if (w->rows >= PK_DEC_SPLIT)
        pk_gemm(a, out, 0, w, bias, act, add);
    else
        pk_k_gemm_q8(out, 0, w->q, w->s, bias, w->rows, act, add);
}

/* g = prediction(label): the embedding up through the LSTMs, each layer's
 * state stepped in place; then the joint's half of it, a->gp */
static void predict(pk_asr *a, uint32_t label) {
    const pk_model *m = a->m;
    const uint32_t P = m->pred_hidden;
    const float *x = m->embed + (size_t)label * P;
    for (uint32_t l = 0; l < m->pred_layers; l++) {
        float *h = a->dh + (size_t)l * P, *c = a->dc + (size_t)l * P;
        gemv(a, a->gates, &m->lstm[l].wx, m->lstm[l].b, x, 0);
        gemv(a, a->gates, &m->lstm[l].wh, NULL, h, 1);
        pk_k_lstm(h, c, a->gates, P);
        x = h;
    }
    gemv(a, a->gp, &m->jpred, m->jpred_b, x, 0);
}

/* The joint at frame t: the best token and the best duration's index */
static void joint(pk_asr *a, uint32_t t, uint32_t *k, uint32_t *di) {
    const pk_model *m = a->m;
    const uint32_t J = m->joint_hidden, V = m->vocab;
    const float *f = a->fproj + (size_t)t * J;
    for (uint32_t j = 0; j < J; j++) {
        float v = f[j] + a->gp[j];
        a->z[j] = v > 0 ? v : 0;
    }
    gemv(a, a->logit, &m->jout, m->jout_b, a->z, 0);
    uint32_t best = 0;
    for (uint32_t i = 1; i <= V; i++)
        if (a->logit[i] > a->logit[best])
            best = i;
    uint32_t bd = 0;
    for (uint32_t i = 1; i < m->n_dur; i++)
        if (a->logit[V + 1 + i] > a->logit[V + 1 + bd])
            bd = i;
    *k = best;
    *di = bd;
}

static int emit(pk_asr *a, uint32_t k, uint32_t t, uint32_t d) {
    if (a->ntoks == a->toks_cap) {
        size_t cap = a->toks_cap ? 2 * a->toks_cap : 256;
        pk_tok *p = realloc(a->toks, cap * sizeof *p);
        if (!p)
            return -1;
        a->toks = p;
        a->toks_cap = cap;
    }
    a->toks[a->ntoks].id = (int32_t)k;
    a->toks[a->ntoks].frame = (int32_t)t;
    a->toks[a->ntoks].dur = (int32_t)d;
    a->ntoks++;
    return 0;
}

/* TDT's greedy search, from frame 0 and the blank's prediction: the loop the
 * file's header has */
static long search(pk_asr *a, uint32_t T) {
    const pk_model *m = a->m;
    const uint32_t blank = m->blank;
    predict(a, blank);
    uint32_t t = 0, last = UINT32_MAX, lasts = 0;
    while (t < T) {
        uint32_t k, di, d, at;
        do {
            at = t;
            joint(a, t, &k, &di);
            d = m->dur[di];
            if (k == blank && d == 0)
                d = 1;
            t += d;
        } while (k == blank && t < T);
        if (k == blank)
            break;
        if (emit(a, k, at, m->dur[di]))
            return -1;
        lasts = last == at ? lasts + 1 : 1;
        last = at;
        predict(a, k);
        if (lasts >= m->max_symbols && last == t)
            t++;
    }
    return (long)a->ntoks;
}

long pk_decode(pk_asr *a, uint32_t T) {
    const pk_model *m = a->m;
    const uint32_t D = m->d_model, P = m->pred_hidden;
    a->ntoks = 0;
    if (!T)
        return 0;
    /* the joint's encoder half, for every frame at once */
    pk_quantize(a, &a->act, a->x, D, T, D, D);
    pk_gemm(a, a->fproj, m->joint_hidden, &m->jenc, m->jenc_b, &a->act, 0);

    memset(a->dh, 0, (size_t)m->pred_layers * P * sizeof *a->dh);
    memset(a->dc, 0, (size_t)m->pred_layers * P * sizeof *a->dc);
    pk_pool_set_active(a->pool, PK_DEC_THREADS);
    long n = search(a, T);
    pk_pool_set_active(a->pool, a->nth);
    return n;
}
