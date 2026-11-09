/* The log-mel front end (mel.c). Internal. */
#ifndef PK_MEL_H
#define PK_MEL_H

#include "model.h"

/* Feature frames n samples make: n / hop. */
size_t pk_mel_frames(const pk_model *m, size_t n);

/* Frames a thread takes at a time, and the floats of scratch pk_mel needs:
 * for T frames on the calling thread's side, and for a block on each
 * thread's (pk_thread_ws.mel). */
#define PK_MEL_BLOCK 64
size_t pk_mel_scratch(const pk_model *m, size_t T);
size_t pk_mel_thread_scratch(const pk_model *m);

/* nt frames' log-mels, unnormalized, into out[nt][n_mels]: frame t is the
 * n_fft samples from win[t] (the preemphasized signal, centred), through
 * the window, the FFT's power, the filterbank and the log. nt is at most
 * PK_MEL_BLOCK, and scratch pk_mel_thread_scratch(m) floats, 64-byte
 * aligned (the STFT's, backend/kernels.h). The
 * recognizer's features and the diarizer's (core/diar.c) are both this. */
void pk_mel_block(const pk_model *m, const float *const *win, size_t nt, float *out, float *scratch);

/* a->feat[T][n_mels] for n samples x, the frames and then the bands split
 * over the transcriber's pool */
void pk_mel(pk_asr *a, const float *x, size_t n);

#endif /* PK_MEL_H */
