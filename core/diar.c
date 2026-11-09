/* The diarizer: who speaks when, each of up to eight speakers' probability
 * every 10 ms.
 *
 * NVIDIA's Nemotron 3 Diarization (huggingface.co/nvidia/Nemotron-3-
 * Diarization), a Streaming Sortformer, as NeMo runs it at commit cd61689
 * (SortformerEncLabelModel.forward_streaming, and SortformerModules'
 * streaming_update and _compress_spkcache) at batch size one, in the 30.4 s
 * configuration the file carries. tools/pktito_tools/nemotron3.py (the
 * network) and sortformer.py (the cache) are the float64 reference this is
 * read beside:
 *
 *   features    core/mel.c's log-mel, not normalized, with the model's own
 *               tables (Parakeet's rounded to bfloat16), each frame from the
 *               recording's samples: a chunk's frames are the whole
 *               recording's, zeros past its ends
 *   stacking    8 feature frames side by side, 1024 wide, through a matrix
 *               to 512: an 80 ms frame. The features are [T][128], so eight
 *               of them are already one row of 1024, and nothing is copied
 *   a step      [the speaker cache; the FIFO; a chunk and its right context]
 *               through a layer norm, then 31 layers of
 *                 x += attention(LN x)   8 heads of 64, the queries and keys
 *                                        turned by rotary positions counted
 *                                        from the step's first frame
 *                 x += ff(LN x)          512 to 2048, GELU, to 512
 *               then a layer norm, a projection to 192, the upsampler (a
 *               convolution over 3 frames to 8 x 192: eight 10 ms frames for
 *               each 80 ms one) and the head (ReLU, 192 x 192, ReLU, a row a
 *               speaker, sigmoid)
 *   the cache   the chunk's 80 ms frames join the FIFO; those that overflow
 *               it join the speaker cache, which is then cut back to 264:
 *               for each speaker the frames most clearly that speaker
 *               alone, and a learned silence frame after them (cut, below)
 *
 * The speakers come out in the order they first spoke, column 0 whoever
 * spoke first, and the cache keeps that order over a recording of any
 * length: each step hears frames of every speaker heard so far. The cache
 * and the FIFO hold stacked frames, the network's input, not its output.
 *
 * Every matrix product is pk_k_gemm_q8, int8 weights against activations
 * quantized in blocks of 32, the recognizer's rule (README, "Why int8";
 * docs/DIARIZATION.md, "Why int8", measures it for this model). The
 * attention's products are float, on pk_k_gemm_f32. Each pass is split over
 * the pool, and no output element depends on how, so a run is the same bits
 * on any number of threads and every backend gives ref's. The rotation, the
 * means and the cache's cut are plain loops here, the same bits on every
 * machine under -ffp-contract=off.
 */
#include "kernels.h"
#include "mel.h"
#include "model.h"
#include "os.h"
#include "passes.h"
#include "pool.h"
#include "vmath.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define QB 32                    /* the activations' blocks (README, "Why int8") */
#define ATT_BLOCK 16             /* queries an attention item takes */
#define LN_HALF -0.69314718055994530942     /* log(0.5), as NeMo's Python float */

static uint32_t min32(uint32_t a, uint32_t b) { return a < b ? a : b; }
static uint32_t up16(uint32_t x) { return (x + 15) & ~15u; }

typedef struct {
    float *sc;                   /* a block of attention scores: ATT_BLOCK x a step's keys */
    float *row;                  /* a frame of a layer norm's output, or a tile's hidden */
    float *mel;                  /* pk_mel_block's scratch */
    const float **win;           /* a block of feature frames' windows */
} diar_ws;

struct pk_diar {
    const pk_model *m;
    const pk_diarw *w;
    pk_pool *pool;
    int nth;
    uint32_t S, S16, D, F, H, dk, P, U, f;
    uint32_t tmax;               /* a step's most frames: cache, FIFO, chunk and context */
    uint32_t emax;               /* a chunk's most 80 ms frames, its context included */
    uint32_t cmax;               /* the cache's most frames, before a cut */

    /* the recording */
    int64_t got;                 /* samples fed */
    int64_t stt;                 /* the next chunk's first feature frame */
    float  *buf;                 /* samples [buf_at, buf_at + buf_n) of the recording */
    int64_t buf_at;
    size_t  buf_n, buf_cap;

    /* a chunk's features */
    float *y;                    /* its frames' samples, preemphasized */
    float *feat;                 /* [f (emax + 1)][n_mels] */

    /* a step */
    float *in;                   /* [tmax][D]: the cache, the FIFO and the chunk, stacked */
    float *x, *q, *k, *o;        /* [tmax][D]: the stream; the queries, keys, attention */
    float *u;                    /* [tmax][F]: the feed-forward's hidden */
    float *kp, *vp;              /* K and V as pk_k_gemm_f32 panels */
    float *cosr, *sinr;          /* [tmax][dk/2]: the rotary positions' */
    float *pj;                   /* [tmax][P]: the projection */
    float *ups;                  /* [U tmax][P]: the upsampled frames */
    float *hh;                   /* [U tmax][P]: the head's hidden */
    float *logit;                /* [U tmax][S16] */
    float *prob;                 /* [U tmax][S]: each 10 ms frame's probabilities */
    float *p80;                  /* [tmax][S]: each 80 ms frame's, their mean */
    pk_act act, hid, a1;         /* int8: a product's input; the hidden; the projection's */
    diar_ws *tws;
    void *mem;                   /* the block every array here is carved from */

    /* the cache */
    float   *cache, *cache_p;    /* [cmax][D], [cmax][S] */
    uint32_t nc;
    int      has_p;              /* cache_p kept: from the first cut on */
    float   *fifo;               /* [fifo_len + chunk_len][D] */
    uint32_t nf;
    float   *sc;                 /* the cut's scores, [cmax][S] */
    float   *flat;               /* [S][cmax + sil_frames]: them speaker by speaker */
    uint32_t *ord, *tmp;         /* indices, and the merge sort's scratch */
    float   *cut_e, *cut_p;      /* the cache after a cut, before it replaces the old */
};

/* ---------------------------------------------------------------- passes --- */

typedef struct { pk_diar *d; float *out; const float *x, *w, *b; uint32_t T; } norm_job;

static void norm_item(void *arg, int i, int th) {
    (void)th;
    norm_job *j = arg;
    const uint32_t D = j->d->D, t0 = (uint32_t)i * 16, nt = min32(16, j->T - t0);
    pk_k_layernorm(j->out + (size_t)t0 * D, D, j->x + (size_t)t0 * D, D, nt, D, j->w, j->b, j->d->m->ln_eps);
}

/* out = LN(x) over T frames of D */
static void norm(pk_diar *d, float *out, const float *x, uint32_t T, const float *w, const float *b) {
    norm_job j = { d, out, x, w, b, T };
    pk_pool_run(d->pool, norm_item, &j, (int)((T + 15) / 16));
}

/* A layer norm straight into the int8 activations, a frame at a time
 * through a row of thread scratch, as core/encoder.c's normq */
static void normq_item(void *arg, int i, int th) {
    norm_job *j = arg;
    pk_diar *d = j->d;
    float *row = d->tws[th].row;
    for (uint32_t t = (uint32_t)i * 16; t < min32((uint32_t)i * 16 + 16, j->T); t++) {
        pk_k_layernorm(row, d->D, j->x + (size_t)t * d->D, d->D, 1, d->D, j->w, j->b, d->m->ln_eps);
        pk_quant_part(&d->act, t, 0, d->D, row);
    }
}

/* d->act for T frames of LN(x) */
static void normq(pk_diar *d, const float *x, uint32_t T, const float *w, const float *b) {
    norm_job j = { d, NULL, x, w, b, T };
    d->act.n = d->D;
    d->act.qb = QB;
    d->act.T = T;
    pk_pool_run(d->pool, normq_item, &j, (int)((T + 15) / 16));
}

/* The feed-forward's hidden, GELU'd and quantized by the items of its up
 * product, each over its own tile while it is in its core's cache, as the
 * recognizer's SiLU is (core/encoder.c, siluq_tile) */
static void geluq_tile(void *arg, uint32_t r0, uint32_t nr, uint32_t c0, uint32_t nc, int th) {
    pk_diar *d = arg;
    float *row = d->tws[th].row;
    for (uint32_t t = c0; t < c0 + nc; t++) {
        memcpy(row, d->u + (size_t)t * d->F + r0, nr * sizeof *row);
        pk_k_gelu(row, nr);
        pk_quant_part(&d->hid, t, r0, nr, row);
    }
}

typedef struct { pk_diar *d; uint32_t T; } frames_job;

/* Each frame's queries and keys turned by its rotary angles: in each head,
 * dimensions i and i + dk/2 as a pair (torch's "rotate half"),
 *
 *   (a, b) -> (a cos - b sin, b cos + a sin)
 *
 * which is NeMo's x cos + rotate_half(x) sin, operation for operation. */
static void rope_item(void *arg, int i, int th) {
    (void)th;
    frames_job *j = arg;
    pk_diar *d = j->d;
    const uint32_t h2 = d->dk / 2;
    for (uint32_t t = (uint32_t)i * 16; t < min32((uint32_t)i * 16 + 16, j->T); t++) {
        const float *c = d->cosr + (size_t)t * h2, *s = d->sinr + (size_t)t * h2;
        float *v[2] = { d->q + (size_t)t * d->D, d->k + (size_t)t * d->D };
        for (int which = 0; which < 2; which++)
            for (uint32_t h = 0; h < d->H; h++) {
                float *z = v[which] + (size_t)h * d->dk;
                for (uint32_t k = 0; k < h2; k++) {
                    const float a = z[k], b = z[k + h2];
                    z[k] = a * c[k] - b * s[k];
                    z[k + h2] = b * c[k] + a * s[k];
                }
            }
    }
}

typedef struct { pk_panels pn; uint32_t D; } kpan_job;

/* K, turned, into pk_k_gemm_f32's panels: 64 frames an item */
static void kpan_item(void *arg, int i, int th) {
    kpan_job *j = arg;
    const uint32_t c0 = (uint32_t)i * 64;
    pk_panels_tile(&j->pn, 0, j->D, c0, min32(64, j->pn.n - c0), th);
}

typedef struct { pk_diar *d; uint32_t T, Tp; } att_job;

/* One head over ATT_BLOCK queries: their scores against every key,
 * divided by sqrt(dk) and softmaxed, and the weighted sum of the values.
 * No positions here: the rotation put them in the queries and keys. */
static void att_item(void *arg, int item, int th) {
    att_job *j = arg;
    pk_diar *d = j->d;
    const uint32_t D = d->D, dk = d->dk, T = j->T, nb = (T + ATT_BLOCK - 1) / ATT_BLOCK;
    const uint32_t h = (uint32_t)item / nb, i0 = ((uint32_t)item % nb) * ATT_BLOCK;
    const uint32_t nq = min32(ATT_BLOCK, T - i0);
    float *sc = d->tws[th].sc;
    pk_k_gemm_f32(sc, j->Tp, d->kp + (size_t)h * j->Tp * dk, dk, j->Tp, dk, d->q + (size_t)i0 * D + h * dk,
                  D, nq, NULL, 0, 0);
    const float scale = 1.0f / sqrtf((float)dk);
    for (uint32_t i = 0; i < nq; i++)
        pk_k_softmax(sc + (size_t)i * j->Tp, T, scale);
    pk_k_gemm_f32(d->o + (size_t)i0 * D + h * dk, D, d->vp + (size_t)h * (dk / 16) * T * 16, T, dk, T, sc,
                  j->Tp, nq, NULL, 0, 0);
}

typedef struct { float *x; size_t n; } relu_job;

static void relu_item(void *arg, int i, int th) {
    (void)th;
    relu_job *j = arg;
    const size_t a = (size_t)i * 4096, b = a + 4096 < j->n ? a + 4096 : j->n;
    for (size_t k = a; k < b; k++)
        j->x[k] = j->x[k] > 0 ? j->x[k] : 0;
}

static void relu(pk_diar *d, float *x, size_t n) {
    relu_job j = { x, n };
    pk_pool_run(d->pool, relu_item, &j, (int)((n + 4095) / 4096));
}

/* The upsampler's input: each frame with the one before and the one after
 * it, side by side, zeros past the step's ends. Its projection is quantized
 * a frame at a time (a1), and a frame of 192 is six whole blocks, so the
 * 576-wide columns are the three frames' int8 rows laid side by side: the
 * bytes and scales that quantizing the 576 numbers would give. */
static void taps_item(void *arg, int i, int th) {
    (void)th;
    frames_job *j = arg;
    pk_diar *d = j->d;
    const uint32_t P = d->P, nb = P / QB;
    for (uint32_t t = (uint32_t)i * 16; t < min32((uint32_t)i * 16 + 16, j->T); t++)
        for (uint32_t k = 0; k < 3; k++) {
            int8_t *q = d->act.q + (size_t)t * 3 * P + (size_t)k * P;
            float *s = d->act.d + (size_t)t * 3 * nb + (size_t)k * nb;
            const int64_t src = (int64_t)t + k - 1;
            if (src < 0 || src >= (int64_t)j->T) {
                memset(q, 0, P);
                memset(s, 0, nb * sizeof *s);
            } else {
                memcpy(q, d->a1.q + (size_t)src * P, P);
                memcpy(s, d->a1.d + (size_t)src * nb, nb * sizeof *s);
            }
        }
}

/* --------------------------------------------------------------- network --- */

static void layer(pk_diar *d, const pk_tlayer *L, uint32_t T) {
    const uint32_t D = d->D, F = d->F, Tp = up16(T);
    PROF(PK_P_QUANT, normq(d, d->x, T, L->att_nw, L->att_nb));
    PROF(PK_P_QKV, pk_gemm_on(d->pool, d->q, D, &L->q, NULL, &d->act, 0, NULL, NULL));
    PROF(PK_P_QKV, pk_gemm_on(d->pool, d->k, D, &L->k, NULL, &d->act, 0, NULL, NULL));
    PROF(PK_P_QKV, pk_gemm_panels_on(d->pool, d->vp, (size_t)T * 16, &L->v, &d->act));
    frames_job fj = { d, T };
    PROF(PK_P_POS, pk_pool_run(d->pool, rope_item, &fj, (int)((T + 15) / 16)));
    kpan_job kj = { { d->kp, d->k, T, Tp, D, d->dk }, D };
    PROF(PK_P_ATT, pk_pool_run(d->pool, kpan_item, &kj, (int)((T + 63) / 64)));
    att_job aj = { d, T, Tp };
    PROF(PK_P_ATT, pk_pool_run(d->pool, att_item, &aj, (int)(d->H * ((T + ATT_BLOCK - 1) / ATT_BLOCK))));
    PROF(PK_P_QUANT, pk_quantize_on(d->pool, &d->act, d->o, D, T, D, QB));
    PROF(PK_P_ATTO, pk_gemm_on(d->pool, d->x, D, &L->o, L->o_b, &d->act, 1, NULL, NULL));

    PROF(PK_P_QUANT, normq(d, d->x, T, L->ff_nw, L->ff_nb));
    d->hid.n = F;
    d->hid.qb = QB;
    d->hid.T = T;
    PROF(PK_P_FF, pk_gemm_on(d->pool, d->u, F, &L->up, L->up_b, &d->act, 0, geluq_tile, d));
    PROF(PK_P_FF, pk_gemm_on(d->pool, d->x, D, &L->down, L->down_b, &d->hid, 1, NULL, NULL));
}

/* The final layer norm's T frames, quantized in d->act, to each 10 ms
 * frame's speaker probabilities, d->prob[U T][S]: the projection, the
 * upsampler and the head */
static void head(pk_diar *d, uint32_t T) {
    const pk_diarw *w = d->w;
    const uint32_t P = d->P, U = d->U;
    pk_gemm_on(d->pool, d->pj, P, &w->proj, w->proj_b, &d->act, 0, NULL, NULL);

    pk_quantize_on(d->pool, &d->a1, d->pj, P, T, P, QB);
    d->act.n = 3 * P;
    d->act.qb = QB;
    d->act.T = T;
    frames_job fj = { d, T };
    pk_pool_run(d->pool, taps_item, &fj, (int)((T + 15) / 16));
    pk_gemm_on(d->pool, d->ups, (size_t)U * P, &w->up, w->up_b, &d->act, 0, NULL, NULL);

    /* the head, on U T frames of P: row t U + j of ups is 10 ms frame j of
     * 80 ms frame t */
    relu(d, d->ups, (size_t)U * T * P);
    pk_quantize_on(d->pool, &d->act, d->ups, P, U * T, P, QB);
    pk_gemm_on(d->pool, d->hh, P, &w->head_hidden, w->head_hidden_b, &d->act, 0, NULL, NULL);
    relu(d, d->hh, (size_t)U * T * P);
    pk_quantize_on(d->pool, &d->act, d->hh, P, U * T, P, QB);
    pk_gemm_on(d->pool, d->logit, d->S16, &w->head_out, w->head_out_b, &d->act, 0, NULL, NULL);
    for (size_t r = 0; r < (size_t)U * T; r++)
        for (uint32_t s = 0; s < d->S; s++)
            d->prob[r * d->S + s] = pk_sigmoidf(d->logit[r * d->S16 + s]);
}

/* The step's T stacked frames in d->in to each 10 ms frame's speaker
 * probabilities, d->prob[U T][S] */
static void forward(pk_diar *d, uint32_t T) {
    const pk_diarw *w = d->w;
    PROF(PK_P_NORM, norm(d, d->x, d->in, T, w->embed_nw, w->embed_nb));
    for (uint32_t i = 0; i < d->m->n_layers; i++)
        layer(d, &w->layers[i], T);
    PROF(PK_P_QUANT, normq(d, d->x, T, w->final_nw, w->final_nb));
    PROF(PK_P_HEAD, head(d, T));
}

/* -------------------------------------------------------------- the cut --- */

/* idx[0..n) sorted by v[idx] from the largest down, equals in the order
 * they came (a merge sort, stable): numpy's argsort(-v, kind="stable"),
 * the reference's, which puts a tie's lower index first */
static void sort_desc(uint32_t *idx, uint32_t *tmp, uint32_t n, const float *v) {
    for (uint32_t w = 1; w < n; w *= 2) {
        for (uint32_t lo = 0; lo < n; lo += 2 * w) {
            const uint32_t mid = min32(lo + w, n), hi = min32(lo + 2 * w, n);
            uint32_t a = lo, b = mid, k = lo;
            while (a < mid && b < hi)
                tmp[k++] = v[idx[b]] > v[idx[a]] ? idx[b++] : idx[a++];
            while (a < mid) tmp[k++] = idx[a++];
            while (b < hi) tmp[k++] = idx[b++];
        }
        memcpy(idx, tmp, n * sizeof *idx);
    }
}

/* Raise each speaker's k highest scores by -scale log 0.5: sc is [n][S] */
static void boost(pk_diar *d, float *sc, uint32_t n, uint32_t k, double scale) {
    const uint32_t S = d->S;
    const float by = (float)(scale * LN_HALF);
    float *col = d->flat;
    for (uint32_t s = 0; s < S; s++) {
        for (uint32_t t = 0; t < n; t++) {
            col[t] = sc[(size_t)t * S + s];
            d->ord[t] = t;
        }
        sort_desc(d->ord, d->tmp, n, col);
        for (uint32_t i = 0; i < k && i < n; i++)
            sc[(size_t)d->ord[i] * S + s] -= by;
    }
}

/* The cache cut back to cache_len frames (SortformerModules.
 * _compress_spkcache). A frame's score for a speaker is how much likelier
 * it is that speaker alone than not,
 *
 *   log max(p, thr) - log max(1 - p, thr) + sum over speakers of log max(1 - p, thr) - log 0.5
 *
 * thr the file's score_threshold; a frame that is not the speaker's speech
 * (p <= 0.5) never qualifies, nor, once a speaker has enough frames of its
 * own, one it shares (score <= 0). The newest frames, those past cache_len,
 * get boost_latest; each speaker's best few a lot more (strong_boost, so
 * that every speaker keeps frames), its next best some (weak_boost, so that
 * none takes over). Then the cache_len best of every speaker's scores
 * together, each speaker's run followed by its silence frames, which always
 * qualify: kept in order of speaker, then frame. A pick of a frame that did
 * not qualify takes a silence frame, at the end. */
static void cut(pk_diar *d) {
    const pk_diarw *w = d->w;
    const uint32_t S = d->S, n = d->nc, L = w->cache_len, ns = w->sil_frames, nn = n + ns;
    const uint32_t per = L / S - ns;
    const float thr = w->score_threshold;
    float *sc = d->sc;
    for (uint32_t t = 0; t < n; t++) {
        const float *p = d->cache_p + (size_t)t * S;
        float lp[PK_MAX_SPK], l1p[PK_MAX_SPK], sum = 0;
        for (uint32_t s = 0; s < S; s++) {
            lp[s] = pk_logf(fmaxf(p[s], thr));
            l1p[s] = pk_logf(fmaxf(1.0f - p[s], thr));
            sum += l1p[s];
        }
        for (uint32_t s = 0; s < S; s++)
            sc[(size_t)t * S + s] = p[s] > 0.5f ? lp[s] - l1p[s] + sum - (float)LN_HALF : -INFINITY;
    }
    const uint32_t enough = (uint32_t)floor((double)per * w->min_pos_scores);
    for (uint32_t s = 0; s < S; s++) {
        uint32_t pos = 0;
        for (uint32_t t = 0; t < n; t++)
            pos += sc[(size_t)t * S + s] > 0;
        if (pos >= enough)
            for (uint32_t t = 0; t < n; t++) {
                float *v = &sc[(size_t)t * S + s];
                if (!(*v > 0) && d->cache_p[(size_t)t * S + s] > 0.5f)
                    *v = -INFINITY;
            }
    }
    if (w->boost_latest > 0)
        for (uint32_t t = L; t < n; t++)
            for (uint32_t s = 0; s < S; s++)
                sc[(size_t)t * S + s] += w->boost_latest;
    boost(d, sc, n, (uint32_t)floor((double)per * w->strong_boost), 2);
    boost(d, sc, n, (uint32_t)floor((double)per * w->weak_boost), 1);

    /* every speaker's scores, speaker by speaker, its silence frames after */
    float *flat = d->flat;
    for (uint32_t s = 0; s < S; s++)
        for (uint32_t t = 0; t < nn; t++)
            flat[(size_t)s * nn + t] = t < n ? sc[(size_t)t * S + s] : INFINITY;
    const uint32_t all = S * nn;
    for (uint32_t i = 0; i < all; i++)
        d->ord[i] = i;
    sort_desc(d->ord, d->tmp, all, flat);
    /* the picks that qualify, in order of speaker then frame */
    uint32_t kept = 0;
    for (uint32_t i = 0; i < L; i++)
        if (flat[d->ord[i]] != -INFINITY)
            d->tmp[kept++] = d->ord[i];
    for (uint32_t i = 1; i < kept; i++)
        for (uint32_t j = i; j > 0 && d->tmp[j - 1] > d->tmp[j]; j--) {
            const uint32_t x = d->tmp[j];
            d->tmp[j] = d->tmp[j - 1];
            d->tmp[j - 1] = x;
        }
    for (uint32_t r = 0; r < L; r++) {
        const uint32_t t = r < kept ? d->tmp[r] % nn : n;
        float *e = d->cut_e + (size_t)r * d->D, *p = d->cut_p + (size_t)r * S;
        if (t < n) {
            memcpy(e, d->cache + (size_t)t * d->D, d->D * sizeof *e);
            memcpy(p, d->cache_p + (size_t)t * S, S * sizeof *p);
        } else {
            memcpy(e, w->sil, d->D * sizeof *e);
            memset(p, 0, S * sizeof *p);
        }
    }
    memcpy(d->cache, d->cut_e, (size_t)L * d->D * sizeof *d->cache);
    memcpy(d->cache_p, d->cut_p, (size_t)L * S * sizeof *d->cache_p);
    d->nc = L;
}

/* After a step of nc cache frames, nf FIFO frames and the chunk's ne, its
 * n own frames from lc on (SortformerModules.streaming_update): they join
 * the FIFO, and when it overflows, its oldest update_period or more join
 * the cache, which is cut when it outgrows cache_len. The FIFO's and the
 * popped frames' probabilities are this step's. */
static void update(pk_diar *d, uint32_t nc, uint32_t nf, uint32_t lc, uint32_t n) {
    const pk_diarw *w = d->w;
    const uint32_t D = d->D, S = d->S, total = nf + n;
    const float *chunk = d->in + (size_t)(nc + nf + lc) * D;
    const float *chunk_p = d->p80 + (size_t)(nc + nf + lc) * S, *fifo_p = d->p80 + (size_t)nc * S;
    if (total <= w->fifo_len) {
        memcpy(d->fifo + (size_t)nf * D, chunk, (size_t)n * D * sizeof *d->fifo);
        d->nf = total;
        return;
    }
    uint32_t k = w->update_period > n + nf - w->fifo_len ? w->update_period : n + nf - w->fifo_len;
    k = min32(k, total);
    /* the popped frames: the FIFO's first, then the chunk's */
    const int keep_p = d->has_p || d->nc + k > w->cache_len;
    if (keep_p && !d->has_p) {
        /* the first cut scores the cache as this step heard it */
        memcpy(d->cache_p, d->p80, (size_t)nc * S * sizeof *d->cache_p);
        d->has_p = 1;
    }
    for (uint32_t i = 0; i < k; i++) {
        const float *e = i < nf ? d->fifo + (size_t)i * D : chunk + (size_t)(i - nf) * D;
        const float *p = i < nf ? fifo_p + (size_t)i * S : chunk_p + (size_t)(i - nf) * S;
        memcpy(d->cache + (size_t)(d->nc + i) * D, e, D * sizeof *d->cache);
        if (keep_p)
            memcpy(d->cache_p + (size_t)(d->nc + i) * S, p, S * sizeof *d->cache_p);
    }
    d->nc += k;
    /* what is left of the FIFO and the chunk, to the FIFO's front */
    if (k < nf) {
        memmove(d->fifo, d->fifo + (size_t)k * D, (size_t)(nf - k) * D * sizeof *d->fifo);
        memcpy(d->fifo + (size_t)(nf - k) * D, chunk, (size_t)n * D * sizeof *d->fifo);
    } else
        memcpy(d->fifo, chunk + (size_t)(k - nf) * D, (size_t)(total - k) * D * sizeof *d->fifo);
    d->nf = total - k;
    if (d->nc > w->cache_len)
        cut(d);
}

/* ------------------------------------------------------------ the chunks --- */

/* Sample s of the recording, 0 past its ends or not finite */
static float sample(const pk_diar *d, int64_t s) {
    if (s < 0 || s >= d->got || s < d->buf_at)
        return 0.0f;
    const float v = d->buf[s - d->buf_at];
    return isfinite(v) ? v : 0.0f;
}

typedef struct { pk_diar *d; uint32_t T; } mel_job;

static void mel_item(void *arg, int i, int th) {
    mel_job *j = arg;
    pk_diar *d = j->d;
    const pk_model *m = d->m;
    const size_t t0 = (size_t)i * PK_MEL_BLOCK, nt = j->T - t0 < PK_MEL_BLOCK ? j->T - t0 : PK_MEL_BLOCK;
    const float **win = d->tws[th].win;
    for (size_t t = 0; t < nt; t++)
        win[t] = d->y + (t0 + t) * m->hop;
    pk_mel_block(m, win, nt, d->feat + t0 * m->n_mels, d->tws[th].mel);
}

/* Feature frames [f0, f0 + T) of the recording into d->feat, and zeros to
 * a whole stack of frames after them: frame t is the n_fft samples centred
 * on sample t hop, preemphasized (y[s] = x[s] - 0.97 x[s - 1], y[0] = x[0]),
 * zeros outside the recording, as core/mel.c makes a recording's */
static void features(pk_diar *d, int64_t f0, uint32_t T) {
    const pk_model *m = d->m;
    const int64_t s0 = f0 * m->hop - m->n_fft / 2;
    const size_t len = (size_t)(T - 1) * m->hop + m->n_fft;
    for (size_t i = 0; i < len; i++) {
        const int64_t s = s0 + (int64_t)i;
        if (s < 0 || s >= d->got)
            d->y[i] = 0.0f;
        else
            d->y[i] = s ? sample(d, s) - m->preemph * sample(d, s - 1) : sample(d, 0);
    }
    mel_job j = { d, T };
    pk_pool_run(d->pool, mel_item, &j, (int)((T + PK_MEL_BLOCK - 1) / PK_MEL_BLOCK));
    const uint32_t T8 = (T + d->f - 1) / d->f * d->f;
    memset(d->feat + (size_t)T * m->n_mels, 0, (size_t)(T8 - T) * m->n_mels * sizeof *d->feat);
}

/* The chunk of feature frames [stt, end), heard with `left` frames before
 * it and `right` after (multiples of the stacking but the recording's
 * last): its rows to fn, rows_total the recording's rows when its end is
 * known (-1 before). 1 when fn stopped the run. */
static int chunk(pk_diar *d, int64_t stt, int64_t end, int64_t left, int64_t right, int64_t rows_total,
                 pk_diar_fn fn, void *arg) {
    const pk_model *m = d->m;
    const uint32_t f = d->f, D = d->D, S = d->S, U = d->U, M = m->n_mels;
    const uint32_t nfeat = (uint32_t)(left + (end - stt) + right), ne = (nfeat + f - 1) / f;
    const uint32_t lc = (uint32_t)(left / f), rc = (uint32_t)((right + f - 1) / f), n = ne - lc - rc;
    const uint32_t nc = d->nc, nf = d->nf, T = nc + nf + ne;
    PROF(PK_P_MEL, features(d, stt - left, nfeat));
    memcpy(d->in, d->cache, (size_t)nc * D * sizeof *d->in);
    memcpy(d->in + (size_t)nc * D, d->fifo, (size_t)nf * D * sizeof *d->in);
    float *in = d->in + (size_t)(nc + nf) * D;
    PROF(PK_P_SUB, pk_quantize_on(d->pool, &d->act, d->feat, (size_t)f * M, ne, f * M, QB));
    PROF(PK_P_SUB, pk_gemm_on(d->pool, in, D, &d->w->stack_w, NULL, &d->act, 0, NULL, NULL));
    forward(d, T);
    /* each 80 ms frame's probabilities, the mean of its U, which the cache
     * scores (SortformerModules.downsample_preds) */
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t s = 0; s < S; s++) {
            float sum = 0;
            for (uint32_t k = 0; k < U; k++)
                sum += d->prob[((size_t)t * U + k) * S + s];
            d->p80[(size_t)t * S + s] = sum / (float)U;
        }
    const int64_t r0 = stt * U / f;
    int64_t rows = (int64_t)n * U;
    if (rows_total >= 0 && rows > rows_total - r0)
        rows = rows_total - r0;
    PROF(PK_P_CACHE, update(d, nc, nf, lc, n));
    if (rows > 0 && fn(arg, r0, (size_t)rows, d->prob + (size_t)(nc + nf + lc) * U * S))
        return 1;
    return 0;
}

/* Every chunk the samples so far allow, or with final every chunk left:
 * 0, or 1 when fn stopped the run. A chunk runs once the samples its right
 * context needs are in, so that where the recording ends cannot change it:
 * its frames to the end of that context, each frame's window centred on
 * it. */
static int drain(pk_diar *d, int final, pk_diar_fn fn, void *arg) {
    const pk_model *m = d->m;
    const pk_diarw *w = d->w;
    const int64_t f = d->f, hop = m->hop, half = m->n_fft / 2;
    const int64_t CL = (int64_t)w->chunk_len * f, LC = (int64_t)w->chunk_left * f, RC = (int64_t)w->chunk_right * f;
    for (;;) {
        int64_t T0, rows_total = -1;
        if (final) {
            T0 = d->got / hop;
            rows_total = (T0 * d->U + f - 1) / f;
        } else {
            const int64_t last = d->stt + CL + RC;          /* frames the chunk reads, to here */
            const int64_t need = (last - 1) * hop + half > last * hop ? (last - 1) * hop + half : last * hop;
            if (d->got < need)
                return 0;
            T0 = last;
        }
        if (d->stt >= T0)
            return 0;
        const int64_t stt = d->stt, end = stt + CL < T0 ? stt + CL : T0;
        const int64_t left = LC < stt ? LC : stt, right = RC < T0 - end ? RC : T0 - end;
        const int rc = chunk(d, stt, end, left, right, rows_total, fn, arg);
        d->stt = end;
        /* the samples the next chunk reads start one before its first frame's window */
        int64_t keep = (end - LC) * hop - half - 1;
        if (keep > d->buf_at) {
            const int64_t drop = keep - d->buf_at < (int64_t)d->buf_n ? keep - d->buf_at : (int64_t)d->buf_n;
            memmove(d->buf, d->buf + drop, (d->buf_n - (size_t)drop) * sizeof *d->buf);
            d->buf_n -= (size_t)drop;
            d->buf_at += drop;
        }
        if (rc)
            return rc;
    }
}

/* ---------------------------------------------------------------- public --- */

int pk_model_speakers(const pk_model *m) { return m && m->diar ? (int)m->dw.n_spk : 0; }

void pk_diar_free(pk_diar *d) {
    if (!d)
        return;
    pk_pool_free(d->pool);
    free(d->tws);
    free(d->mem);
    free(d);
}

/* The arrays, each 64-byte aligned (the STFT's scratch must be: backend/
 * kernels.h, PK_K_STFT_SCRATCH), carved from one block as core/asr.c
 * carves a transcriber's: a first pass counts from 0, a second assigns
 * them from the block's aligned start, the same offsets. The sizes are
 * counted in 64 bits, so that a model too big for a 32-bit system's memory
 * is refused rather than wrapping. */
#define CARVE(dst, n, sz) do {                                         \
        uint64_t at_ = (p + 63) & ~(uint64_t)63;                       \
        p = at_ + (uint64_t)(n) * (sz);                                \
        if (pass) (dst) = (void *)(uintptr_t)at_;                      \
    } while (0)

pk_diar *pk_diar_new(const pk_model *m, int threads) {
    if (!m || !m->diar)
        return NULL;
    pk_diar *d = calloc(1, sizeof *d);
    if (!d)
        return NULL;
    const pk_diarw *w = &m->dw;
    d->m = m;
    d->w = w;
    d->S = w->n_spk;
    d->S16 = (w->n_spk + 15) & ~15u;
    d->D = m->d_model;
    d->F = m->ff_dim;
    d->H = m->n_heads;
    d->dk = m->d_model / m->n_heads;
    d->P = w->proj_dim;
    d->U = w->upsample;
    d->f = w->stack;
    d->emax = w->chunk_left + w->chunk_len + w->chunk_right;
    d->tmax = w->cache_len + w->fifo_len + d->emax;
    d->cmax = w->cache_len + w->fifo_len + w->chunk_len;
    d->pool = pk_pool_new(threads > 0 ? threads : pk_os_cores());
    if (!d->pool) {
        free(d);
        return NULL;
    }
    d->nth = pk_pool_threads(d->pool);
    d->tws = calloc((size_t)d->nth, sizeof *d->tws);
    if (!d->tws) {
        pk_diar_free(d);
        return NULL;
    }
    const uint64_t D = d->D, F = d->F, T = d->tmax, Tp = up16(d->tmax), P = d->P, U = d->U, S = d->S;
    const uint64_t M = m->n_mels, E8 = (uint64_t)d->emax * d->f, C = d->cmax, NN = C + w->sil_frames;
    uint64_t wide = E8 * M;                           /* a product's int8 input, the widest */
    if (T * D > wide) wide = T * D;
    if (T * 3 * P > wide) wide = T * 3 * P;
    if (U * T * P > wide) wide = U * T * P;
    const uint64_t cap = (E8 + 2) * m->hop + m->n_fft + 1 + 65536;
    for (int pass = 0; pass < 2; pass++) {
        uint64_t p = pass ? ((uintptr_t)d->mem + 63) & ~(uintptr_t)63 : 0;
        CARVE(d->buf, cap, sizeof(float));
        CARVE(d->y, (E8 + 1) * m->hop + m->n_fft, sizeof(float));
        CARVE(d->feat, (E8 + d->f) * M, sizeof(float));
        CARVE(d->in, T * D, sizeof(float));
        CARVE(d->x, T * D, sizeof(float));
        CARVE(d->q, T * D, sizeof(float));
        CARVE(d->k, T * D, sizeof(float));
        CARVE(d->o, T * D, sizeof(float));
        CARVE(d->u, T * F, sizeof(float));
        CARVE(d->kp, Tp * D, sizeof(float));
        CARVE(d->vp, T * D, sizeof(float));
        CARVE(d->cosr, T * (d->dk / 2), sizeof(float));
        CARVE(d->sinr, T * (d->dk / 2), sizeof(float));
        CARVE(d->pj, T * P, sizeof(float));
        CARVE(d->ups, U * T * P, sizeof(float));
        CARVE(d->hh, U * T * P, sizeof(float));
        CARVE(d->logit, U * T * d->S16, sizeof(float));
        CARVE(d->prob, U * T * S, sizeof(float));
        CARVE(d->p80, T * S, sizeof(float));
        CARVE(d->act.q, wide, 1);
        CARVE(d->act.d, wide / QB, sizeof(float));
        CARVE(d->hid.q, T * F, 1);
        CARVE(d->hid.d, T * F / QB, sizeof(float));
        CARVE(d->a1.q, T * P, 1);
        CARVE(d->a1.d, T * P / QB, sizeof(float));
        CARVE(d->cache, C * D, sizeof(float));
        CARVE(d->cache_p, C * S, sizeof(float));
        CARVE(d->fifo, ((size_t)w->fifo_len + w->chunk_len) * D, sizeof(float));
        CARVE(d->sc, C * S, sizeof(float));
        CARVE(d->flat, NN * S, sizeof(float));
        CARVE(d->ord, NN * S, sizeof(uint32_t));
        CARVE(d->tmp, NN * S, sizeof(uint32_t));
        CARVE(d->cut_e, (size_t)w->cache_len * D, sizeof(float));
        CARVE(d->cut_p, (size_t)w->cache_len * S, sizeof(float));
        for (int i = 0; i < d->nth; i++) {
            diar_ws *ws = &d->tws[i];
            CARVE(ws->sc, (size_t)ATT_BLOCK * Tp, sizeof(float));
            CARVE(ws->row, D > 256 ? D : 256, sizeof(float));   /* a frame, or a product's slice */
            CARVE(ws->mel, pk_mel_thread_scratch(m), sizeof(float));
            CARVE(ws->win, PK_MEL_BLOCK, sizeof(float *));
        }
        if (!pass && (p > SIZE_MAX - 64 || !(d->mem = malloc((size_t)p + 64)))) {
            pk_diar_free(d);
            return NULL;
        }
    }
    d->buf_cap = (size_t)cap;
    /* the rotary positions' sines and cosines, position t of a step times
     * each frequency, the product a float32 multiply as NeMo forms it */
    const uint32_t h2 = d->dk / 2;
    for (uint32_t t = 0; t < d->tmax; t++)
        for (uint32_t i = 0; i < h2; i++)
            pk_sincosf((float)t * w->rope_freq[i], &d->sinr[(size_t)t * h2 + i], &d->cosr[(size_t)t * h2 + i]);
    return d;
}

int pk_diar_threads(const pk_diar *d) { return d->nth; }

void pk_diar_reset(pk_diar *d) {
    d->got = 0;
    d->stt = 0;
    d->buf_at = 0;
    d->buf_n = 0;
    d->nc = 0;
    d->nf = 0;
    d->has_p = 0;
}

int pk_diar_feed(pk_diar *d, const float *x, size_t n, pk_diar_fn fn, void *arg) {
    pk_os_throttle was = pk_os_unthrottle();
    int rc = 0;
    while (n && !rc) {
        const size_t room = d->buf_cap - d->buf_n, take = n < room ? n : room;
        if (!take) {                       /* a buffer that holds a chunk's span always drains */
            rc = -1;
            break;
        }
        memcpy(d->buf + d->buf_n, x, take * sizeof *x);
        d->buf_n += take;
        d->got += (int64_t)take;
        x += take;
        n -= take;
        rc = drain(d, 0, fn, arg);
    }
    pk_os_rethrottle(was);
    return rc;
}

int pk_diar_finish(pk_diar *d, pk_diar_fn fn, void *arg) {
    pk_os_throttle was = pk_os_unthrottle();
    const int rc = drain(d, 1, fn, arg);
    pk_diar_reset(d);
    pk_os_rethrottle(was);
    return rc;
}

int pk_diar_run(pk_diar *d, const float *x, size_t n, pk_diar_fn fn, void *arg) {
    pk_diar_reset(d);
    const int rc = pk_diar_feed(d, x, n, fn, arg);
    if (rc) {
        pk_diar_reset(d);
        return rc;
    }
    return pk_diar_finish(d, fn, arg);
}
