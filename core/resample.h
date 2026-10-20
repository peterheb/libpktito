/* Sample-rate conversion as torchaudio.transforms.Resample does it.
 *
 * Parakeet and Silero's VAD both hear 16 kHz. A recording at another rate is
 * brought to it with torchaudio's Resample at its defaults: a Hann-windowed
 * sinc, six zero crossings each side, cut off at 0.99 of the lower Nyquist,
 * with the kernel computed the way the transform computes it. This is
 * vadito's resampler (core/resample.c there), which tests/rscheck.py holds
 * to torchaudio's output to 1.8e-7 on thirteen rate pairs. NeMo's own loader
 * resamples with librosa's soxr instead, so for a file that is not at 16 kHz
 * pkito hears a slightly different input from NeMo's; at 16 kHz, the same.
 *
 * It streams, so a recording of any length converts in a fixed amount of
 * memory, and the output is torchaudio's whole-signal output, the same
 * ceil(n * to / from) samples, however the input is cut into calls.
 */
#ifndef PK_RESAMPLE_H
#define PK_RESAMPLE_H

#include <stddef.h>
#include <stdint.h>

#define PK_RESAMPLE_MAX_RATE 1000000   /* past this the kernel no longer fits a board */

typedef struct pk_resampler pk_resampler;

/* NULL if a rate is 0 or above PK_RESAMPLE_MAX_RATE, or memory runs out. */
pk_resampler *pk_resampler_new(uint32_t from, uint32_t to);
void          pk_resampler_free(pk_resampler *r);

/* At most this many outputs come out of a call given n inputs, whatever was
 * pushed before; pk_resample_room(r, 0) bounds the flush. */
size_t pk_resample_room(const pk_resampler *r, size_t n);

/* Push n inputs; the outputs they complete go to out, and their count is
 * returned. out has pk_resample_room(r, n) floats. */
size_t pk_resample(pk_resampler *r, const float *in, size_t n, float *out);

/* After the last input: the outputs the end's zero padding completes, up to
 * torchaudio's total. out has pk_resample_room(r, 0) floats. */
size_t pk_resample_flush(pk_resampler *r, float *out);

#endif /* PK_RESAMPLE_H */
