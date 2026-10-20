/* The log-mel front end (mel.c). Internal. */
#ifndef PK_MEL_H
#define PK_MEL_H

#include "model.h"

/* Feature frames n samples make: n / hop. */
size_t pk_mel_frames(const pk_model *m, size_t n);

/* Floats of scratch pk_mel needs for T frames. */
size_t pk_mel_scratch(const pk_model *m, size_t T);

/* feat[T][n_mels] for n samples x; win is T pointers of scratch. */
void pk_mel(const pk_model *m, const float *x, size_t n, float *feat, float *scratch,
            const float **win);

#endif /* PK_MEL_H */
