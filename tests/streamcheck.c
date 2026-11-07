/* streamcheck: the two files tests/check-c.sh holds `pktito-c --stream` to
 * on a recording that is not at the model's rate.
 *
 * --stream resamples as it reads, a block at a time (core/cli.c, stream).
 * This writes the first 20 s of IN at RATE, and the 16 kHz samples the
 * resampler makes from those. The resampler's output is the whole signal's
 * however its input is cut into calls (core/resample.h), and both files are
 * float32, which keeps every sample, so --stream must print the same lines
 * for the two: the first it resamples itself, the second is at its rate.
 *
 *     ./tests/streamcheck tests/data/test.wav 48000 at-rate.wav back.wav
 */
#include "resample.h"
#include "wav.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define SECONDS    20      /* test.wav's first line and a stretch the end cuts */
#define MODEL_RATE 16000   /* what Parakeet hears */

static int fail(const char *what, const char *err) {
    printf("streamcheck: FAIL - %s%s%s\n", what, *err ? ": " : "", err);
    return 1;
}

/* x[0, n) from `from` to `to`, the end flushed: a malloc'd array, its length
 * in *ny; NULL if there is no memory */
static float *resample(const float *x, size_t n, uint32_t from, uint32_t to, size_t *ny) {
    pk_resampler *r = pk_resampler_new(from, to);
    if (!r)
        return NULL;
    size_t room = pk_resample_room(r, n), tail = pk_resample_room(r, 0);
    float *y = room < SIZE_MAX / sizeof *y - tail ? malloc((room + tail) * sizeof *y) : NULL;
    if (y) {
        size_t k = pk_resample(r, x, n, y);
        *ny = k + pk_resample_flush(r, y + k);
    }
    pk_resampler_free(r);
    return y;
}

static int write_wav(const char *path, uint32_t rate, const float *x, size_t n, char *err, size_t errlen) {
    pk_wav_out *o = pk_wav_create(path, rate, 1, err, errlen);
    return !o || pk_wav_write(o, x, n) || pk_wav_finish(o, err, errlen);
}

int main(int argc, char **argv) {
    char err[256] = "";
    unsigned long rate = argc == 5 ? strtoul(argv[2], NULL, 10) : 0;
    if (!rate || rate > PK_RESAMPLE_MAX_RATE) {
        fprintf(stderr, "usage: streamcheck IN.wav RATE AT_RATE.wav BACK.wav\n");
        return 2;
    }
    pk_wav_info info;
    pk_wav *w = pk_wav_open(argv[1], &info, err, sizeof err);
    if (!w)
        return fail(argv[1], err);
    size_t want = (size_t)SECONDS * info.rate, n = 0, k;
    float *x = malloc(want * sizeof *x);
    if (!x)
        return fail("out of memory", "");
    while (n < want && (k = pk_wav_read(w, x + n, want - n)) > 0)
        n += k;
    pk_wav_close(w);

    size_t na = 0, nb = 0;
    float *a = resample(x, n, info.rate, (uint32_t)rate, &na);
    float *b = a ? resample(a, na, (uint32_t)rate, MODEL_RATE, &nb) : NULL;
    if (!b)
        return fail("out of memory", "");
    if (write_wav(argv[3], (uint32_t)rate, a, na, err, sizeof err))
        return fail(argv[3], err);
    if (write_wav(argv[4], MODEL_RATE, b, nb, err, sizeof err))
        return fail(argv[4], err);
    free(x);
    free(a);
    free(b);
    return 0;
}
