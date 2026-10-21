/* tests/segcheck.py's other half: probabilities on stdin, one a line, and
 * Silero's parameters as arguments; out come pk_segments' segments and then
 * pk_iter_step's events, a line each.
 *
 *   segcheck RATE N_SAMPLES THRESHOLD NEG MIN_SPEECH_MS MAX_SPEECH_S
 *            MIN_SILENCE_MS PAD_MS MIN_SIL_AT_MAX_MS USE_MAX
 *
 * NEG < 0 and MAX_SPEECH_S "inf" are the defaults' spellings. */
#include "pktito.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 11) {
        fprintf(stderr, "usage: segcheck RATE N THR NEG MINSP MAXSP MINSIL PAD MINSILMAX USEMAX\n");
        return 2;
    }
    int rate = atoi(argv[1]);
    long long n_samples = atoll(argv[2]);
    pk_seg_params p;
    pk_seg_defaults(&p);
    p.threshold = strtod(argv[3], NULL);
    p.neg_threshold = strtod(argv[4], NULL);
    p.min_speech_ms = strtod(argv[5], NULL);
    p.max_speech_s = strtod(argv[6], NULL);
    p.min_silence_ms = strtod(argv[7], NULL);
    p.speech_pad_ms = strtod(argv[8], NULL);
    p.min_silence_at_max_speech_ms = strtod(argv[9], NULL);
    p.use_max_poss_sil_at_max_speech = atoi(argv[10]);
    size_t cap = 1024, n = 0;
    float *probs = malloc(cap * sizeof *probs), v;
    while (scanf("%f", &v) == 1) {
        if (n == cap) probs = realloc(probs, (cap *= 2) * sizeof *probs);
        probs[n++] = v;
    }
    size_t room = 2 * n + 1;
    pk_seg *seg = malloc(room * sizeof *seg);
    size_t ns = pk_segments(probs, n, rate, n_samples, &p, seg, room);
    for (size_t i = 0; i < ns; i++) printf("seg %lld %lld\n", (long long)seg[i].start, (long long)seg[i].end);
    pk_iter it;
    pk_iter_init(&it, rate, &p);
    for (size_t i = 0; i < n; i++) {
        int64_t at;
        int e = pk_iter_step(&it, probs[i], &at);
        if (e != PK_NONE) printf("%s %zu %lld\n", e == PK_START ? "start" : "end", i, (long long)at);
    }
    return 0;
}
