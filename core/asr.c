/* A transcriber: its memory, its pool, and audio in, tokens out.
 *
 * The working memory is one block, sized for the longest recording seen so
 * far and grown when a longer one comes: a minute of audio needs about 60
 * MB of it at Parakeet 0.6B's widths, most of it the encoder's activations.
 */
#include "asr.h"
#include "kernels.h"
#include "mel.h"
#include "os.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static size_t up16z(size_t x) { return (x + 15) & ~(size_t)15; }

/* The block's arrays, each 64-byte aligned: a first pass counts from 0, a
 * second assigns them from the block's aligned start, the same offsets. */
#define CARVE(dst, n, sz) do {                                         \
        uintptr_t at_ = (p + 63) & ~(uintptr_t)63;                     \
        p = at_ + (size_t)(n) * (sz);                                  \
        if (pass) (dst) = (void *)at_;                                 \
    } while (0)

static int reserve(pk_asr *a, size_t T0) {
    if (T0 <= a->cap)
        return 0;
    const pk_model *m = a->m;
    const size_t Te = pk_enc_frames((uint32_t)T0), D = m->d_model, F = m->ff_dim;
    const size_t C = m->sub_channels, M = m->n_mels, J = m->joint_hidden;
    const size_t H = m->n_heads, dk = D / H, F1 = (M + 1) / 2, F2 = (F1 + 1) / 2, F3 = M / 8;
    const size_t U = F > 2 * D ? F : 2 * D, R = 2 * Te - 1;
    const size_t Tp = up16z(Te), Pp = up16z(R) + 16;
    size_t widest = U > F3 * C ? U : F3 * C;
    widest = widest > D ? widest : D;
    /* the subsampling's windows (core/encoder.c, sub_item) */
    const size_t s1 = 3 * F1 * C, s2 = F2 * C, s3 = 3 * F2 * C, s4 = PK_SUB_BLOCK * F3 * C;
    const size_t sq = s2 > s4 ? s2 : s4;

    void *mem = NULL;
    for (int pass = 0; pass < 2; pass++) {
        uintptr_t p = pass ? ((uintptr_t)mem + 63) & ~(uintptr_t)63 : 0;
        CARVE(a->mel_scratch, pk_mel_scratch(m, T0), sizeof(float));
        CARVE(a->feat, T0 * M, sizeof(float));
        CARVE(a->win, T0, sizeof(float *));
        CARVE(a->s5, Te * F3 * C, sizeof(float));
        CARVE(a->x, Te * D, sizeof(float));
        CARVE(a->h, Te * D, sizeof(float));
        CARVE(a->u, Te * U, sizeof(float));
        CARVE(a->q, Te * D, sizeof(float));
        CARVE(a->o, Te * D, sizeof(float));
        CARVE(a->pe, R * D, sizeof(float));
        CARVE(a->p, R * D, sizeof(float));
        CARVE(a->kp, H * Tp * dk, sizeof(float));
        CARVE(a->vp, Te * D, sizeof(float));
        CARVE(a->pp, H * Pp * dk, sizeof(float));
        CARVE(a->act.q, Te * widest, 1);
        CARVE(a->act.d, Te * widest / 32, sizeof(float));
        CARVE(a->ape.q, R * D, 1);
        CARVE(a->ape.d, R * D / 32, sizeof(float));
        CARVE(a->fproj, Te * J, sizeof(float));
        for (int t = 0; t < a->nth; t++) {
            pk_thread_ws *w = &a->tws[t];
            CARVE(w->s1, s1, sizeof(float));
            CARVE(w->s2, s2, sizeof(float));
            CARVE(w->s3, s3, sizeof(float));
            CARVE(w->s4, s4, sizeof(float));
            CARVE(w->aq, sq, 1);
            CARVE(w->ad, sq / 32, sizeof(float));
            CARVE(w->qu, PK_ATT_BLOCK * dk, sizeof(float));
            CARVE(w->qv, PK_ATT_BLOCK * dk, sizeof(float));
            CARVE(w->sc, PK_ATT_BLOCK * Tp, sizeof(float));
            CARVE(w->bd, PK_ATT_BLOCK * (Tp + 32), sizeof(float));
            CARVE(w->mel, pk_mel_thread_scratch(m), sizeof(float));
            CARVE(w->row, F, sizeof(float));
        }
        if (!pass) {
            mem = malloc((size_t)p + 64);
            if (!mem)
                return -1;
            free(a->mem);
            a->mem = mem;
        }
    }
    a->cap = T0;
    return 0;
}

pk_asr *pk_asr_new(const pk_model *m, int threads) {
    if (!m)
        return NULL;
    pk_asr *a = calloc(1, sizeof *a);
    if (!a)
        return NULL;
    a->m = m;
    a->nth = threads > 0 ? threads : pk_os_cores();
    a->pool = pk_pool_new(a->nth);
    a->tws = calloc((size_t)a->nth, sizeof *a->tws);
    const size_t P = m->pred_hidden, L = m->pred_layers, J = m->joint_hidden;
    const size_t wide = P > J ? P : J;
    a->dh = calloc(L * P, sizeof(float));
    a->dc = calloc(L * P, sizeof(float));
    a->gates = calloc(4 * P, sizeof(float));
    a->gp = calloc(J, sizeof(float));
    a->z = calloc(J, sizeof(float));
    a->logit = calloc(m->n_out_pad, sizeof(float));
    a->dact.q = calloc(wide, 1);
    a->dact.d = calloc(wide, sizeof(float));
    if (!a->pool || !a->tws || !a->dh || !a->dc || !a->gates || !a->gp || !a->z || !a->logit
        || !a->dact.q || !a->dact.d) {
        pk_asr_free(a);
        return NULL;
    }
    a->nth = pk_pool_threads(a->pool);
    a->wide = a->nth;
    a->fast = threads > 0 ? a->nth : pk_os_fast_cores();
    a->tune = a->fast > 0 && a->fast < a->nth ? PK_TUNE_AT : 0;
    return a;
}

int pk_asr_threads(const pk_asr *a) { return a->wide; }

void pk_asr_free(pk_asr *a) {
    if (!a)
        return;
    pk_pool_free(a->pool);
    free(a->tws);
    free(a->mem);
    free(a->dh);
    free(a->dc);
    free(a->gates);
    free(a->gp);
    free(a->z);
    free(a->logit);
    free(a->dact.q);
    free(a->dact.d);
    free(a->toks);
    free(a);
}

long pk_asr_run(pk_asr *a, const float *x, size_t n, const pk_tok **toks) {
    const size_t T0 = pk_mel_frames(a->m, n);
    *toks = a->toks;
    a->ntoks = 0;
    if (T0 == 0)
        return 0;
    if (T0 > UINT32_MAX / 4 || reserve(a, T0))
        return -1;
    pk_os_throttle was = pk_os_unthrottle();  /* this thread runs a share of every job */
    PROF(PK_P_MEL, pk_mel(a, x, n));
    uint32_t Te = pk_encode(a, (uint32_t)T0);
    long nt;
    PROF(PK_P_DEC, nt = pk_decode(a, Te));
    pk_os_rethrottle(was);
    *toks = a->toks;
    return nt;
}

const char *pk_backend(void) { return pk_k_name(); }

/* The compiler's own words for itself; Apple's clang names a version of its
 * own, unlike the LLVM one it comes from. */
const char *pk_compiler(void) {
#if defined(__apple_build_version__)
    return "Apple clang " __clang_version__;
#elif defined(__clang__)
    return "clang " __clang_version__;
#elif defined(__GNUC__)
    return "gcc " __VERSION__;
#else
    return "an unknown compiler";
#endif
}

#ifdef PK_PROFILE
double pk_prof[PK_NPROF];
double pk_prof_now(void) { return pk_os_now(); }
#endif

/* ---------------------------------------------------------------- chunks --- */

/* Silence kept on the far side of a speech chunk's first and last segment,
 * at most: what lies further from the speech is a gap's. */
#define PK_CHUNK_PAD 16000

/* A gap between speech chunks is transcribed too, since Silero hears voices
 * rather than words: singing, and speech over music, are not speech to it.
 * But not in a piece shorter than PK_GAP_MIN, where the recognizer makes
 * words of anything: of 28 one-second pieces of instrumental music, v2
 * heard "Yeah." or the like in 27, and from 3 s up in none (README, "Long
 * recordings"). Nor in a silent piece, which holds nothing to hear. */
#define PK_GAP_MIN    48000      /* 3 s */
/* a 32 ms frame with a mean square under this is silent: 50 dB below full
 * scale, 18 dB under the quietest of 42 pieces of that music at its loudest */
#define PK_GAP_SILENT 1e-5

/* No 32 ms frame of x[0, n) louder than PK_GAP_SILENT. A sample that is not
 * finite is read as 0, as pk_asr_run reads it. */
static int silent(const float *x, int64_t n) {
    for (int64_t i = 0; i < n; i += PK_VAD_SAMPLES) {
        int64_t len = n - i < PK_VAD_SAMPLES ? n - i : PK_VAD_SAMPLES;
        double ss = 0;
        for (int64_t j = 0; j < len; j++) {
            double v = isfinite(x[i + j]) ? x[i + j] : 0;
            ss += v * v;
        }
        if (ss > PK_GAP_SILENT * (double)len)
            return 0;
    }
    return 1;
}

/* The gap [a, b) as chunks from out[*k] on: the fewest pieces of at most
 * max_len, all of a length, but those shorter than PK_GAP_MIN or silent.
 * Silero cannot say where a gap's words are, so its cuts fall where they
 * fall. */
static void put_gap(const float *x, int64_t a, int64_t b, int64_t max_len, pk_seg *out, size_t *k) {
    if (b <= a)
        return;
    int64_t m = (b - a + max_len - 1) / max_len;
    for (int64_t i = 0; i < m; i++) {
        int64_t pa = a + (b - a) * i / m, pb = a + (b - a) * (i + 1) / m;
        if (pb - pa >= PK_GAP_MIN && !silent(x + pa, pb - pa)) {
            out[*k].start = pa;
            out[*k].end = pb;
            (*k)++;
        }
    }
}

size_t pk_chunks(const float *x, const pk_seg *s, size_t n, int64_t total, int64_t max_len, pk_seg *out) {
    if (total <= max_len && n) {
        out[0].start = 0;
        out[0].end = total;
        return 1;
    }
    size_t k = 0;
    int64_t done = 0;                           /* the end of the last speech chunk */
    for (size_t i = 0; i < n;) {
        /* the chunk opens half way into the silence before segment i */
        int64_t gap0 = i ? s[i].start - s[i - 1].end : s[i].start;
        int64_t lead = s[i].start - (gap0 / 2 < PK_CHUNK_PAD ? gap0 / 2 : PK_CHUNK_PAD);
        size_t j = i;
        int64_t trail = 0;
        for (; j < n; j++) {
            int64_t gap = j + 1 < n ? s[j + 1].start - s[j].end : total - s[j].end;
            int64_t t = s[j].end + (gap / 2 < PK_CHUNK_PAD ? gap / 2 : PK_CHUNK_PAD);
            if (j > i && t - lead > max_len)
                break;
            trail = t;
        }
        if (trail - lead > max_len) {           /* one segment, too long with its silence */
            lead = s[i].end - max_len > lead ? s[i].end - max_len : lead;
            trail = lead + max_len < trail ? lead + max_len : trail;
        }
        put_gap(x, done, lead, max_len, out, &k);
        out[k].start = lead;
        out[k].end = trail;
        k++;
        done = trail;
        i = j > i ? j : i + 1;
    }
    put_gap(x, done, total, max_len, out, &k);
    return k;
}
