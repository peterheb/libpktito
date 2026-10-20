/* core/vmath.h against the platform's double-precision functions.
 *
 * Every float x in a function's domain (or one in 61, by default) through
 * pk_expf, pk_sigmoidf and pk_tanhf, each held to the double result rounded
 * once to float, which differs from the correctly rounded float only where
 * the double lands within an ulp of a float's rounding boundary: never, for
 * any purpose here. The distance is counted in floats, so 1 is the next
 * float up or down.
 *
 *   exp      [-87, 88]       inside the clamp; past it the result is the
 *                            clamp's own e^-87 or e^88
 *   sigmoid  [-87, inf)      below it 1/(1 + e^88) at most, which is 6e-39
 *   tanh     every float
 *   log      every positive finite float
 *   sin cos  the positional encoding's angles, and floats in [-8192, 8192]
 *
 * and sqrt(x * x) is held to |x| wherever x * x is normal, which the VAD's
 * magnitudes rest on.
 *
 *     ./tests/mathcheck          one float in 61: half a second
 *     ./tests/mathcheck --all    every float: 28 s on an M4 Pro, 20 min on a Pi 3
 */
#include "vmath.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const char *name; double max_ulp; float worst_x; uint64_t n, over_half; } stat;

static int64_t ordered(float f) {
    int32_t i;
    memcpy(&i, &f, sizeof i);
    return i < 0 ? (int64_t)INT32_MIN - i : i;
}

static void check(stat *s, float x, float got, double want) {
    float w = (float)want;
    double d = (double)llabs(ordered(got) - ordered(w));
    if (d > s->max_ulp) { s->max_ulp = d; s->worst_x = x; }
    s->over_half += d > 0;
    s->n++;
}

int main(int argc, char **argv) {
    const uint32_t step = argc > 1 && !strcmp(argv[1], "--all") ? 1 : 61;
    stat se = { "exp", 0, 0, 0, 0 }, ss = { "sigmoid", 0, 0, 0, 0 }, st = { "tanh", 0, 0, 0, 0 };
    stat sl = { "log", 0, 0, 0, 0 }, sn = { "sin", 0, 0, 0, 0 }, sc = { "cos", 0, 0, 0, 0 };
    uint64_t sqrt_bad = 0, sqrt_n = 0;
    for (uint64_t u = 0; u < (1ull << 32); u += step) {
        uint32_t b = (uint32_t)u;
        float x;
        memcpy(&x, &b, sizeof x);
        if (isnan(x))
            continue;
        if (x >= -87.0f && x <= 88.0f)
            check(&se, x, pk_expf(x), exp((double)x));
        if (x >= -87.0f)
            check(&ss, x, pk_sigmoidf(x), 1.0 / (1.0 + exp(-(double)x)));
        check(&st, x, pk_tanhf(x), tanh((double)x));
        if (x > 0 && isfinite(x))
            check(&sl, x, pk_logf(x), log((double)x));
        /* the VAD takes sqrt of the STFT kernel's power for its magnitude
         * (core/vad.c), which is |x| only if sqrt(x * x) is: wherever x * x
         * is a normal float */
        float a = fabsf(x);
        if (a >= 0x1p-63f && a <= 0x1p63f) {
            sqrt_n++;
            sqrt_bad += sqrtf(x * x) != a;
        }
    }
    /* the positional encoding's angles: every position a 30-minute
     * recording reaches (22500 frames) times each frequency, which is what
     * pk_sincosf is asked; and a sweep of floats in [-2^13, 2^13] */
    for (int k = 0; k < 512; k += step == 1 ? 1 : 7) {
        float fr = (float)exp((double)(k * 2.0f * -(float)(log(10000.0) / 1024)));
        for (int p = -22500; p <= 22500; p += step == 1 ? 1 : 13) {
            float ang = (float)p * fr, s1, c1;
            pk_sincosf(ang, &s1, &c1);
            check(&sn, ang, s1, sin((double)ang));
            check(&sc, ang, c1, cos((double)ang));
        }
    }
    for (uint32_t i = 0; i < 2000000 / (step == 1 ? 1 : 10); i++) {
        float ang = ((float)i / (2000000 / (step == 1 ? 1 : 10)) * 2 - 1) * 8192.0f, s1, c1;
        pk_sincosf(ang, &s1, &c1);
        check(&sn, ang, s1, sin((double)ang));
        check(&sc, ang, c1, cos((double)ang));
    }
    /* the bounds core/vmath.h states: `--all` on an M4 Pro, a 12900HK and a
     * Pi 3B+, which agree to the last digit printed */
    const double bound[6] = { 1, 1, 2, 1, 1, 1 };
    stat *all[6] = { &se, &st, &ss, &sl, &sn, &sc };
    int bad = sqrt_bad != 0;
    printf("mathcheck sqrt     %llu floats: sqrt(x * x) is |x| %s\n", (unsigned long long)sqrt_n,
           sqrt_bad ? "NOT everywhere  FAIL" : "for every one");
    for (int i = 0; i < 6; i++) {
        stat *s = all[i];
        int over = s->max_ulp > bound[i];
        printf("mathcheck %-8s %llu floats: at most %.0f ulp (at %a), %.4f%% not the rounded double%s\n",
               s->name, (unsigned long long)s->n, s->max_ulp, s->worst_x,
               100.0 * s->over_half / s->n, over ? "  FAIL" : "");
        bad |= over;
    }
    return bad;
}
