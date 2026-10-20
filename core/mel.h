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

/* a->feat[T][n_mels] for n samples x, the frames and then the bands split
 * over the transcriber's pool */
void pk_mel(pk_asr *a, const float *x, size_t n);

#endif /* PK_MEL_H */
