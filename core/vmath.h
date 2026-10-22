/* fma, exp, sigmoid and tanh: the definition every backend computes.
 *
 * These are vadito's (core/vmath.h there), and with pk_sincos below they
 * are the whole of pktito's transcendental mathematics: SiLU and GLU's
 * sigmoids, the softmax's exponentials, the LSTM cells' gates, and the
 * positional encoding's sines.
 *
 * They are pktito's own rather than libm's for two reasons. The platform
 * libms disagree in the last bit (glibc's expf, macOS's and MSVC's each round
 * somewhere the others do not), and an engine that wants the same bits on a
 * Pi and a desktop cannot call any of them. And every backend computes these
 * in its own vector registers, so the function has to be one whose every
 * operation is written down: a backend's lane does exactly what the code
 * below does, operation for operation, and tests/kcheck holds it to that bit
 * for bit. That needs the build's -ffp-contract=off, so that no compiler
 * fuses an operation here that the vector code keeps apart.
 *
 * Accuracy, from tests/mathcheck --all over every float, each against the
 * platform's double-precision function rounded once to float:
 *
 *   pk_expf     at most 1 ulp on [-87, 88]; 0.67% of floats not the rounded value
 *   pk_tanhf    at most 1 ulp everywhere; 0.17% not
 *   pk_sigmoidf at most 2 ulp on [-87, inf); 5.2% not
 *   pk_logf     at most 1 ulp on every positive float; a handful of the 2.1e9 not
 *   pk_sincosf  0 ulp on the 25 million angles the positional encoding takes
 *               at 30 minutes and on a sweep of [-8192, 8192]
 *
 * The polynomials are minimax fits of relative error, made by
 * tools/fit_vmath.py; it prints these constants.
 */
#ifndef PK_VMATH_H
#define PK_VMATH_H

#include <math.h>
#include <stdint.h>
#include <string.h>

/* a * b + c rounded once, as IEEE 754's fusedMultiplyAdd and every FMA
 * instruction compute it.
 *
 * The product of two floats is exact in a double (48 significant bits of
 * 53), so only the sum rounds, s to nearest in double and then to float.
 * That goes wrong only where s lands on a midpoint between two floats that
 * the exact sum was not on (tests/mathcheck has a case): the midpoints are
 * doubles themselves and rounding keeps order, so an s that is no midpoint
 * is on the exact sum's side of every one. Among the floats' normals a
 * midpoint is a double whose low 29 bits are 1 and then zeros, one test;
 * among their subnormals it is another bit, and those, rare, go the long
 * way too. There the sum is rounded to odd, to whichever of the two doubles
 * around the exact sum has its last bit set when the sum is not a double,
 * and the rounding to float that follows is the exact sum's, a double having
 * more than twice a float's precision (Boldo and Melquiond, "Emulation of
 * FMA and correctly rounded sums: proved algorithms using rounding to odd",
 * 2008); TwoSum gives the sum's error exactly, and a NaN or an infinity
 * skips it, as the exact sum cannot overflow a double. The short way costs
 * 2.55 ns in a chain on a 12900HK, glibc's own fmaf 2.38 without the FMA
 * instruction and 0.79 with it; always the long way, 4.98. */
static inline float pk_fmaf_soft(float a, float b, float c) {
    double p = (double)a * b, s = p + c;
    uint64_t u;
    memcpy(&u, &s, sizeof u);
    const uint64_t mag = u & 0x7fffffffffffffffu;
    if ((u & 0x1fffffffu) != 0x10000000u && (mag >= 0x3810000000000000u || mag == 0))   /* |s| >= 2^-126 or 0 */
        return (float)s;
    double v = s - p, e = (p - (s - v)) + (c - v);      /* s + e is p + c, exactly */
    if (e != 0 && !(u & 1) && (u >> 52 & 0x7ff) != 0x7ff) {
        u += (e > 0) == (s > 0) ? 1 : UINT64_MAX;       /* toward the exact sum */
        memcpy(&s, &u, sizeof s);
    }
    return (float)s;
}

/* Where the build has the instruction (every AArch64, and x86 built with
 * -mfma, as the tuned backends are), FP_FAST_FMAF says so and fmaf is that
 * instruction. Elsewhere fmaf is a call into libm, and not every libm rounds
 * it once: mingw-w64 13's with msvcrt rounds twice, which took pk_expf to 2
 * ulp and 3.96% of its results off the rounded value (0.67% with one
 * rounding) and moved four word times in each model's transcript by a frame
 * (tests/mathcheck and the --json transcripts of a build with it,
 * 2026-10-22). So without the instruction it is pk_fmaf_soft, whose bits
 * are the instruction's on any C11 compiler. */
static inline float pk_fmaf(float a, float b, float c) {
#ifdef FP_FAST_FMAF
    return fmaf(a, b, c);
#else
    return pk_fmaf_soft(a, b, c);
#endif
}

/* x = n ln2 + r. ln2 is split in two so that n * ln2 is exact to well past
 * float: HI is ln2 rounded to float, LO the rest. LOG2E is 1/ln2 rounded. */
#define PK_LOG2E   0x1.715476p+0f
#define PK_LN2_HI  0x1.62e430p-1f
#define PK_LN2_LO  -0x1.05c610p-29f

/* exp(r) on [-ln2/2, ln2/2] as c0 + c1 r + ... + c6 r^6. The fit had c0 and
 * c1 free, and both came out within 2^-25 of 1. Its own error is 0.03 ulp;
 * what is left of the 1 ulp is the float arithmetic. */
#define PK_EXP_C2  0x1.fffffap-2f
#define PK_EXP_C3  0x1.55540ap-3f
#define PK_EXP_C4  0x1.55589ap-5f
#define PK_EXP_C5  0x1.126d0cp-7f
#define PK_EXP_C6  0x1.6ab980p-10f

/* Below |x| = 0.625 tanh is x + x^3 (t0 + t1 x^2 + ... + t4 x^8), where its
 * exp form 1 - 2/(e^2x + 1) would cancel. Crossovers of 0.75, 0.875 and 1.0
 * with longer polynomials measured the same 1 ulp; this is the shortest. */
#define PK_TANH_X0 0.625f
#define PK_TANH_T0 -0x1.555532p-2f
#define PK_TANH_T1 0x1.110726p-3f
#define PK_TANH_T2 -0x1.b83c5ap-5f
#define PK_TANH_T3 0x1.52269cp-6f
#define PK_TANH_T4 -0x1.75e1cep-8f

/* The exp argument's clamp. n = rint(x log2e) then stays in [-126, 127],
 * so 2^n is a normal float and the result is finite and never a subnormal
 * built from the wrong exponent bits. The ends are e^-87 = 1.6e-38 and
 * e^88 = 1.7e38: a sigmoid or a tanh of anything beyond them is already 0,
 * 1 or -1 in float. */
#define PK_EXP_LO  -87.0f
#define PK_EXP_HI  88.0f

/* 2^n for n in [-126, 127], as the float whose exponent field is n + 127 */
static inline float pk_pow2i(int32_t n) {
    uint32_t u = (uint32_t)(n + 127) << 23;
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

static inline float pk_expf(float x) {
    x = fminf(fmaxf(x, PK_EXP_LO), PK_EXP_HI);      /* and a NaN becomes PK_EXP_LO */
    float n = rintf(x * PK_LOG2E);
    float r = pk_fmaf(n, -PK_LN2_HI, x);
    r = pk_fmaf(n, -PK_LN2_LO, r);
    float p = PK_EXP_C6;
    p = pk_fmaf(p, r, PK_EXP_C5);
    p = pk_fmaf(p, r, PK_EXP_C4);
    p = pk_fmaf(p, r, PK_EXP_C3);
    p = pk_fmaf(p, r, PK_EXP_C2);
    p = pk_fmaf(p, r, 1.0f);
    p = pk_fmaf(p, r, 1.0f);
    return p * pk_pow2i((int32_t)n);
}

static inline float pk_sigmoidf(float x) {
    return 1.0f / (1.0f + pk_expf(-x));
}

static inline float pk_tanhf(float x) {
    float a = fabsf(x);
    if (a < PK_TANH_X0) {
        float x2 = x * x;
        float p = PK_TANH_T4;
        p = pk_fmaf(p, x2, PK_TANH_T3);
        p = pk_fmaf(p, x2, PK_TANH_T2);
        p = pk_fmaf(p, x2, PK_TANH_T1);
        p = pk_fmaf(p, x2, PK_TANH_T0);
        return pk_fmaf(x * x2, p, x);
    }
    float e = pk_expf(2.0f * a);
    return copysignf(1.0f - 2.0f / (e + 1.0f), x);
}

/* sin and cos of a float angle: computed in double and rounded once, so
 * within half an ulp of float but for a double rounding that lands on a
 * float's midpoint (about one value in 2^29), and the same on every machine
 * because every step is IEEE arithmetic in a written order - no libm sin,
 * whose last bit differs between platforms.
 *
 * The angle is reduced by pi/2 (Cody-Waite: pi/2 as three doubles, the first
 * two of 33 bits so that k times each is exact for |k| < 2^20, which is
 * |a| < 1.6e6), then sin and cos on [-pi/4, pi/4] are their Taylor series
 * to r^17 and r^18, whose next terms are below 1e-19 there: the 1/n! are
 * the coefficients, no fit needed. The quadrant k mod 4 picks the signs.
 *
 * The positional encoding (core/encoder.c) is the only caller, once per
 * position and frequency per recording: 2T-1 by d_model/2 calls. */
#define PK_2_OVER_PI 0x1.45f306dc9c883p-1
#define PK_PIO2_1    0x1.921fb54400000p+0
#define PK_PIO2_2    0x1.0b4611a600000p-34
#define PK_PIO2_3    0x1.3198a2e037073p-69

static inline void pk_sincosf(float a, float *s, float *c) {
    double x = a;
    double k = rint(x * PK_2_OVER_PI);
    double r = x - k * PK_PIO2_1;
    r = r - k * PK_PIO2_2;
    r = r - k * PK_PIO2_3;
    double r2 = r * r;
    double ps = 0x1.952c77030ad4ap-49;                 /*  1/17! */
    ps = ps * r2 + -0x1.ae7f3e733b81fp-41;             /* -1/15! */
    ps = ps * r2 + 0x1.6124613a86d09p-33;              /*  1/13! */
    ps = ps * r2 + -0x1.ae64567f544e4p-26;             /* -1/11! */
    ps = ps * r2 + 0x1.71de3a556c734p-19;              /*  1/9!  */
    ps = ps * r2 + -0x1.a01a01a01a01ap-13;             /* -1/7!  */
    ps = ps * r2 + 0x1.1111111111111p-7;               /*  1/5!  */
    ps = ps * r2 + -0x1.5555555555555p-3;              /* -1/3!  */
    double sn = r + r * r2 * ps;
    double pc = -0x1.6827863b97d97p-53;                /* -1/18! */
    pc = pc * r2 + 0x1.ae7f3e733b81fp-45;              /*  1/16! */
    pc = pc * r2 + -0x1.93974a8c07c9dp-37;             /* -1/14! */
    pc = pc * r2 + 0x1.1eed8eff8d898p-29;              /*  1/12! */
    pc = pc * r2 + -0x1.27e4fb7789f5cp-22;             /* -1/10! */
    pc = pc * r2 + 0x1.a01a01a01a01ap-16;              /*  1/8!  */
    pc = pc * r2 + -0x1.6c16c16c16c17p-10;             /* -1/6!  */
    pc = pc * r2 + 0x1.5555555555555p-5;               /*  1/4!  */
    pc = pc * r2 + -0.5;
    double cs = 1.0 + r2 * pc;
    switch ((int)((int64_t)k & 3)) {
    case 0:  *s = (float)sn;  *c = (float)cs;  break;
    case 1:  *s = (float)cs;  *c = (float)-sn; break;
    case 2:  *s = (float)-sn; *c = (float)-cs; break;
    default: *s = (float)-cs; *c = (float)sn;  break;
    }
}

/* The natural log of a positive finite float, computed in double and
 * rounded once, for the same reason and with the same property as
 * pk_sincosf. x = 2^e m with m in [sqrt(1/2), sqrt(2)) (frexp is exact),
 * log m = 2 atanh(s) with s = (m - 1)/(m + 1), |s| <= 0.1716, by its series
 * s + s^3/3 + ... + s^23/23, whose next term is below 1e-18 of the sum, and
 * log x = log m + e ln 2. The log-mel features take one per band per frame
 * (core/mel.c): 12,800 a second of audio. */
#define PK_LN2_D 0x1.62e42fefa39efp-1

static inline float pk_logf(float x) {
    int e;
    double m = frexp((double)x, &e);                   /* x = m 2^e, m in [0.5, 1) */
    if (m < 0x1.6a09e667f3bcdp-1) {                    /* sqrt(1/2) */
        m *= 2;
        e -= 1;
    }
    double s = (m - 1) / (m + 1), s2 = s * s;
    double p = 1.0 / 23;
    for (int k = 21; k >= 3; k -= 2) p = p * s2 + 1.0 / k;
    p = p * s2 + 1.0;
    return (float)(2 * s * p + e * PK_LN2_D);
}

#endif /* PK_VMATH_H */
