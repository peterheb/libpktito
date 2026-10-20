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
#include "kernels.h"
#include "vmath.h"

#include <math.h>
#include <string.h>

size_t pk_mel_frames(const pk_model *m, size_t n) { return n / m->hop; }

size_t pk_mel_scratch(const pk_model *m, size_t T) {
    size_t bins = m->n_fft / 2 + 1;
    return T * m->hop + m->n_fft + 16          /* the padded signal */
         + ((T * bins + 15) & ~(size_t)15)     /* the power spectra */
         + PK_K_STFT_SCRATCH(m->n_fft);
}

void pk_mel(const pk_model *m, const float *x, size_t n, float *feat, float *scratch,
            const float **win) {
    const size_t T = pk_mel_frames(m, n), N = m->n_fft, bins = N / 2 + 1, M = m->n_mels;
    if (!T)
        return;
    /* the centred, preemphasized signal: T hop + N samples cover every frame */
    const size_t len = T * m->hop + N;
    float *y = scratch, *spec = y + len + 16, *fft = spec + ((T * bins + 15) & ~(size_t)15);
    memset(y, 0, len * sizeof *y);
    size_t used = n < len - N / 2 ? n : len - N / 2;
    for (size_t i = 0; i < used; i++) {
        float v = isfinite(x[i]) ? x[i] : 0.0f;
        float p = i ? (isfinite(x[i - 1]) ? x[i - 1] : 0.0f) : 0.0f;
        y[N / 2 + i] = i ? v - m->preemph * p : v;
    }
    for (size_t t = 0; t < T; t++)
        win[t] = y + t * m->hop;
    pk_k_stft(spec, bins, win, (uint32_t)T, &m->fft, fft);

    for (size_t t = 0; t < T; t++) {
        const float *p = spec + t * bins;
        float *o = feat + t * M;
        for (size_t b = 0; b < M; b++) {
            const float *fb = m->mel_fb + b * bins + m->mel_lo[b];
            const float *pb = p + m->mel_lo[b];
            float acc = 0;
            for (uint32_t k = 0; k < m->mel_n[b]; k++)
                acc = fmaf(fb[k], pb[k], acc);
            o[b] = pk_logf(acc + m->log_guard);
        }
    }
    for (size_t b = 0; b < M; b++) {
        double sum = 0;
        for (size_t t = 0; t < T; t++)
            sum += feat[t * M + b];
        double mean = sum / (double)T, ss = 0;
        for (size_t t = 0; t < T; t++) {
            double d = feat[t * M + b] - mean;
            ss += d * d;
        }
        float sd = (T > 1 ? (float)sqrt(ss / (double)(T - 1)) : 0.0f) + m->norm_eps;
        float mf = (float)mean;
        for (size_t t = 0; t < T; t++)
            feat[t * M + b] = (feat[t * M + b] - mf) / sd;
    }
}
