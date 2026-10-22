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
 * magnitudes rest on. pk_fmaf_soft is held to the FMA instruction bit for
 * bit where the build has one to hold it to (FP_FAST_FMAF: every AArch64),
 * on the cases that rounding twice gets wrong, and on special and random
 * values; elsewhere it is held to one of those cases.
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

#ifdef FP_FAST_FMAF
/* pk_fmaf_soft(a, b, c) against fmaf, which with FP_FAST_FMAF is the
 * instruction: the same bits, or both a NaN */
static int fma_differs(float a, float b, float c) {
    float got = pk_fmaf_soft(a, b, c), want = fmaf(a, b, c);
    return isnan(got) ? !isnan(want) : memcmp(&got, &want, sizeof got) != 0;
}

static float from_bits(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

static uint64_t rng = 0x9e3779b97f4a7c15u;
static uint32_t next32(void) {                 /* xorshift64*, its high half */
    rng ^= rng >> 12, rng ^= rng << 25, rng ^= rng >> 27;
    return (uint32_t)((rng * 0x2545f4914f6cdd1du) >> 32);
}
#endif

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
    /* fma: a * b exactly halfway between two floats, 1 + 2^-11 and the next,
     * and c too small for a double to keep saying which way to go. Rounded
     * twice, through double or x87's long double, it goes to the even one. */
    const float fa = 0x1.001p+0f, fc = 0x1p-80f, fwant = 0x1.002002p+0f;
    const int twice_bad = pk_fmaf_soft(fa, fa, fc) != fwant || pk_fmaf_soft(-fa, fa, -fc) != -fwant || pk_fmaf(fa, fa, fc) != fwant;
    uint64_t fma_diff = 0;
#ifdef FP_FAST_FMAF
    uint64_t fma_n = 0;
    /* Random floats meet a midpoint once in about 2^29 triples, so the
     * midpoints are made: (1 + 2^-23)(1 - 2^-23) is 1 - 2^-46, so a * b with
     * a = (1 + 2^-23) 2^ea and b = (1 - 2^-23) 2^eb is 2^m less 2^(m-46),
     * m = ea + eb. With c = K 2^(m+1) on the float grid and K odd, c + a * b
     * is a hair below the midpoint c + 2^m (with b negative, a hair above
     * c - 2^m), and a double holds the midpoint but not the hair: rounded
     * twice it goes to the even neighbour, where c is the answer. Half are
     * results among the floats' normals, half among their subnormals, where
     * the midpoint is a different bit of the double. Then every triple of
     * special and boundary values, and random floats. */
    const uint64_t each = step == 1 ? 1ull << 26 : 1ull << 20;
    for (uint64_t i = 0; i < each; i++, fma_n++) {
        const int sub = i & 1, e = sub ? 0 : (int)(next32() % 200) - 100, m = sub ? -150 : e - 24;
        const uint32_t K = sub ? (next32() % (1u << 22)) | (1u << 8) | 1 : (next32() & 0x7fffffu) | 0x800000u | 1;
        float a = ldexpf(1.0f + 0x1p-23f, m / 2), b = ldexpf(1.0f - 0x1p-23f, m - m / 2), c = ldexpf((float)K, m + 1);
        if (next32() & 1)
            b = -b;
        if (next32() & 1)
            a = -a, c = -c;
        fma_diff += fma_differs(a, b, c);
    }
    static const uint32_t edge[] = { 0, 0x80000000u, 1, 0x807fffffu, 0x00800000u, 0x3f800000u, 0xbf800000u,
                                     0x3f800001u, 0x7f7fffffu, 0x7f800000u, 0xff800000u, 0x7fc00000u };
    const size_t ne = sizeof edge / sizeof *edge;
    for (size_t i = 0; i < ne; i++)
        for (size_t j = 0; j < ne; j++)
            for (size_t k = 0; k < ne; k++, fma_n++)
                fma_diff += fma_differs(from_bits(edge[i]), from_bits(edge[j]), from_bits(edge[k]));
    for (uint64_t i = 0; i < each; i++, fma_n++)
        fma_diff += fma_differs(from_bits(next32()), from_bits(next32()), from_bits(next32()));
#endif
    const int fma_bad = twice_bad || fma_diff != 0;
    printf("mathcheck fma      the twice-rounded case %s", twice_bad ? "WRONG  FAIL" : "right");
#ifdef FP_FAST_FMAF
    printf("; %llu triples, %llu not the instruction's bits%s\n", (unsigned long long)fma_n,
           (unsigned long long)fma_diff, fma_diff ? "  FAIL" : "");
#else
    printf(" (no FMA instruction in this build to sweep against)\n");
#endif

    /* the bounds core/vmath.h states: `--all` on an M4 Pro, a 12900HK and a
     * Pi 3B+, which agree to the last digit printed */
    const double bound[6] = { 1, 1, 2, 1, 1, 1 };
    stat *all[6] = { &se, &st, &ss, &sl, &sn, &sc };
    int bad = sqrt_bad != 0 || fma_bad;
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
