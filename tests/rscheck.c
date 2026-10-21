/* tests/rscheck.py's other half: raw float32 on stdin, resampled FROM -> TO
 * with core/resample.c, raw float32 on stdout. The input is pushed in pieces
 * of sizes drawn from SEED, so the check also says that how a stream is cut
 * does not change it.
 *
 *   rscheck FROM TO SEED */
#include "resample.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: rscheck FROM TO SEED\n");
        return 2;
    }
    pk_resampler *r = pk_resampler_new((uint32_t)atol(argv[1]), (uint32_t)atol(argv[2]));
    if (!r) {
        fprintf(stderr, "rscheck: no resampler for %s -> %s\n", argv[1], argv[2]);
        return 1;
    }
    if (pk_resample_room(r, SIZE_MAX) != SIZE_MAX) {
        fprintf(stderr, "rscheck: the maximum input count did not saturate its output bound\n");
        pk_resampler_free(r);
        return 1;
    }
    unsigned s = (unsigned)atol(argv[3]);
    float in[70000];
    float *out = malloc(pk_resample_room(r, 70000) * sizeof *out + 1024 * 1024);
    size_t k;
    for (;;) {
        s = s * 1103515245u + 12345u;
        size_t piece = s % 4 == 0 ? 1 + (s >> 16) % 7 : 1 + (s >> 8) % 65536;
        if ((k = fread(in, sizeof *in, piece, stdin)) == 0)
            break;
        size_t m = pk_resample(r, in, k, out);
        fwrite(out, sizeof *out, m, stdout);
    }
    size_t m = pk_resample_flush(r, out);
    fwrite(out, sizeof *out, m, stdout);
    pk_resampler_free(r);
    free(out);
    return 0;
}
