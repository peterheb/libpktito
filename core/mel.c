/* Audio to the encoder's input: NeMo's log-mel features.
 *
 * AudioToMelSpectrogramPreprocessor (nemo/collections/asr/parts/
 * preprocessing/features.py, FilterbankFeatures) in eval mode, for one
 * recording of n samples:
 *
 *   T frames            n / hop: torch.stft makes one more, which NeMo
 *                       counts out of seq_len and zeroes, so it is not made
 *   preemphasis         y[i] = x[i] - 0.97 x[i-1], y[0] = x[0]
 *   STFT                centred: n_fft/2 zeros each side, frame t the n_fft
 *                       samples from t hop, times the 400-sample Hann window
 *                       centred in 512 (torch.stft's padding of it)
 *   power               re^2 + im^2 (NeMo takes the root, then squares it)
 *   mel                 the filterbank's 128 bands over the 257 bins
 *   log                 log(mel + 2^-24)
 *   normalize           each band over the T frames: (x - mean) / (std + 1e-5),
 *                       std with the n - 1 denominator
 *
 * The sums over frames are taken in double, where NeMo's are float32 in
 * whatever order torch picks: the same numbers to float rounding, and
 * deterministic. The log is core/vmath.h's, not libm's, for the bits.
 * Dither is training's alone (NeMo adds it only when self.training).
 */
#include "mel.h"
#include "asr.h"
#include "kernels.h"
#include "vmath.h"

#include <math.h>
#include <string.h>

size_t pk_mel_frames(const pk_model *m, size_t n) { return n / m->hop; }

size_t pk_mel_scratch(const pk_model *m, size_t T) {
    return T * m->hop + m->n_fft + 16;             /* the padded signal */
}

size_t pk_mel_thread_scratch(const pk_model *m) {
    size_t bins = m->n_fft / 2 + 1;
    return ((PK_MEL_BLOCK * bins + 15) & ~(size_t)15) + PK_K_STFT_SCRATCH(m->n_fft);
}

typedef struct { pk_asr *a; const float *y; size_t T; } mel_job;

/* frames [i * PK_MEL_BLOCK, ...): their power spectra, through the
 * filterbank and the log */
static void frames_item(void *arg, int i, int th) {
    mel_job *j = arg;
    const pk_model *m = j->a->m;
    const size_t bins = m->n_fft / 2 + 1, M = m->n_mels;
    const size_t t0 = (size_t)i * PK_MEL_BLOCK, nt = j->T - t0 < PK_MEL_BLOCK ? j->T - t0 : PK_MEL_BLOCK;
    float *spec = j->a->tws[th].mel, *fft = spec + ((PK_MEL_BLOCK * bins + 15) & ~(size_t)15);
    const float **win = j->a->win + t0;
    for (size_t t = 0; t < nt; t++)
        win[t] = j->y + (t0 + t) * m->hop;
    pk_k_stft(spec, bins, win, (uint32_t)nt, &m->fft, fft);
    for (size_t t = 0; t < nt; t++) {
        const float *p = spec + t * bins;
        float *o = j->a->feat + (t0 + t) * M;
        for (size_t b = 0; b < M; b++) {
            const float *fb = m->mel_fb + b * bins + m->mel_lo[b];
            const float *pb = p + m->mel_lo[b];
            float acc = 0;
            for (uint32_t k = 0; k < m->mel_n[b]; k++)
                acc = pk_fmaf(fb[k], pb[k], acc);
            o[b] = pk_logf(acc + m->log_guard);
        }
    }
}

/* bands [i * 8, ...): each normalized over the recording */
static void bands_item(void *arg, int i, int th) {
    (void)th;
    mel_job *j = arg;
    const size_t M = j->a->m->n_mels, T = j->T;
    float *feat = j->a->feat;
    for (size_t b = (size_t)i * 8; b < M && b < (size_t)i * 8 + 8; b++) {
        double sum = 0;
        for (size_t t = 0; t < T; t++)
            sum += feat[t * M + b];
        double mean = sum / (double)T, ss = 0;
        for (size_t t = 0; t < T; t++) {
            double d = feat[t * M + b] - mean;
            ss += d * d;
        }
        float sd = (T > 1 ? (float)sqrt(ss / (double)(T - 1)) : 0.0f) + j->a->m->norm_eps;
        float mf = (float)mean;
        for (size_t t = 0; t < T; t++)
            feat[t * M + b] = (feat[t * M + b] - mf) / sd;
    }
}

void pk_mel(pk_asr *a, const float *x, size_t n) {
    const pk_model *m = a->m;
    const size_t T = pk_mel_frames(m, n), N = m->n_fft;
    if (!T)
        return;
    /* the centred, preemphasized signal: T hop + N samples cover every frame */
    const size_t len = T * m->hop + N;
    float *y = a->mel_scratch;
    memset(y, 0, len * sizeof *y);
    size_t used = n < len - N / 2 ? n : len - N / 2;
    for (size_t i = 0; i < used; i++) {
        float v = isfinite(x[i]) ? x[i] : 0.0f;
        float p = i ? (isfinite(x[i - 1]) ? x[i - 1] : 0.0f) : 0.0f;
        y[N / 2 + i] = i ? v - m->preemph * p : v;
    }
    mel_job j = { a, y, T };
    pk_par(a, frames_item, &j, (int)((T + PK_MEL_BLOCK - 1) / PK_MEL_BLOCK));
    pk_par(a, bands_item, &j, (int)((m->n_mels + 7) / 8));
}
