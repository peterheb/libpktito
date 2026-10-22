/* torchaudio's sinc resampler (resample.h), streaming.
 *
 * With the rates reduced by their gcd to `orig` in and `new` out, output
 * sample i * new + j is a dot of phase j's filter with the input from
 * i * orig - width onward, where width is the filter's reach on each side:
 *
 *   y[i new + j] = sum_m k[j][m] x[i orig + m - width],   m < 2 width + orig
 *
 * and the input is zero outside the recording. torchaudio forms k densely
 * and convolves with it (_get_sinc_resample_kernel, then conv1d with stride
 * orig). Most of a dense row is exact zeros, though: past six zero
 * crossings the sinc's argument is clamped, where the Hann window is
 * cos^2(pi/2) and the tap rounds to 0.0f. So each phase here keeps only its
 * run of non-zero taps, which are torchaudio's own float32 values, and a
 * rate pair whose dense kernel would not fit in memory (16001 to 16000 has
 * 16000 rows of 16015) costs a dozen taps a row.
 *
 * The kernel is computed as the transform computes it: in float64, except
 * that -j/new is a float32 division, because torch.arange of ints divided by
 * an int is a float32 tensor before it meets the float64 one.
 */
#include "resample.h"
#include "vmath.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PK_LOWPASS 6          /* lowpass_filter_width: zero crossings each side */
#define PK_ROLLOFF 0.99

struct pk_resampler {
    uint32_t orig, nw;        /* the rates over their gcd; equal for a copy */
    uint32_t width, taps;     /* reach each side, and a dense row: 2 width + orig */
    uint32_t *first, *len;    /* phase j's first non-zero tap and how many */
    size_t   *at;             /* where phase j's taps start in k */
    float    *k;
    float    *hist;           /* the padded input from the next block's start */
    size_t    have, cap;
    uint64_t  n_in, n_out;
};

static uint32_t gcd(uint32_t a, uint32_t b) {
    while (b) {
        uint32_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

void pk_resampler_free(pk_resampler *r) {
    if (!r)
        return;
    free(r->first);
    free(r->len);
    free(r->at);
    free(r->k);
    free(r->hist);
    free(r);
}

/* one dense tap of torchaudio's kernel, as float32 */
static float tap(double tj, int64_t m, uint32_t width, uint32_t orig, double base) {
    const double pi = 3.14159265358979323846;
    double t = (tj + (double)(m - (int64_t)width) / orig) * base;
    t = t < -PK_LOWPASS ? -PK_LOWPASS : t > PK_LOWPASS ? PK_LOWPASS : t;
    double win = cos(t * pi / PK_LOWPASS / 2);
    win *= win;
    t *= pi;
    double s = t == 0 ? 1.0 : sin(t) / t;
    return (float)(s * (win * (base / orig)));
}

pk_resampler *pk_resampler_new(uint32_t from, uint32_t to) {
    if (!from || !to || from > PK_RESAMPLE_MAX_RATE || to > PK_RESAMPLE_MAX_RATE)
        return NULL;
    pk_resampler *r = calloc(1, sizeof *r);
    if (!r)
        return NULL;
    uint32_t g = gcd(from, to);
    r->orig = from / g;
    r->nw = to / g;
    if (r->orig == r->nw)
        return r;                                 /* torchaudio returns the input */
    double base = (r->orig < r->nw ? r->orig : r->nw) * PK_ROLLOFF;
    r->width = (uint32_t)ceil(PK_LOWPASS * (double)r->orig / base);
    r->taps = 2 * r->width + r->orig;
    r->first = malloc(r->nw * sizeof *r->first);
    r->len = malloc(r->nw * sizeof *r->len);
    r->at = malloc(r->nw * sizeof *r->at);
    /* |t| < 6 spans 12 orig / base input samples; two more each side */
    size_t per = (size_t)ceil(2.0 * PK_LOWPASS * r->orig / base) + 4;
    r->k = malloc((size_t)r->nw * per * sizeof *r->k);
    r->cap = 2 * (size_t)r->taps + 4096;
    r->hist = malloc(r->cap * sizeof *r->hist);
    if (!r->first || !r->len || !r->at || !r->k || !r->hist) {
        pk_resampler_free(r);
        return NULL;
    }
    size_t used = 0;
    for (uint32_t j = 0; j < r->nw; j++) {
        double tj = (double)(-(float)j / (float)r->nw);
        /* the taps where the sinc's argument is inside (-6, 6) */
        double lo = r->width + r->orig * (-PK_LOWPASS / base - tj);
        double hi = r->width + r->orig * (PK_LOWPASS / base - tj);
        int64_t m0 = (int64_t)floor(lo) - 1, m1 = (int64_t)ceil(hi) + 1;
        if (m0 < 0) m0 = 0;
        if (m1 > (int64_t)r->taps - 1) m1 = (int64_t)r->taps - 1;
        while (m0 <= m1 && tap(tj, m0, r->width, r->orig, base) == 0.0f) m0++;
        while (m1 >= m0 && tap(tj, m1, r->width, r->orig, base) == 0.0f) m1--;
        r->first[j] = (uint32_t)m0;
        r->len[j] = m1 >= m0 ? (uint32_t)(m1 - m0 + 1) : 0;
        r->at[j] = used;
        for (int64_t m = m0; m <= m1; m++) r->k[used++] = tap(tj, m, r->width, r->orig, base);
    }
    r->have = r->width;                           /* the zeros before the input */
    memset(r->hist, 0, r->have * sizeof *r->hist);
    return r;
}

/* A push completes at most (have + n - taps) / orig + 1 blocks of nw, and
 * have is always under taps; a flush at most (2 width) / orig + 3. Both are
 * inside this, whatever has been pushed before. */
size_t pk_resample_room(const pk_resampler *r, size_t n) {
    if (r->orig == r->nw)
        return n;
    if (n > SIZE_MAX - r->taps)
        return SIZE_MAX;
    size_t blocks = (n + r->taps) / r->orig;
    if (blocks > SIZE_MAX - 2)
        return SIZE_MAX;
    blocks += 2;
    return blocks > SIZE_MAX / r->nw ? SIZE_MAX : blocks * r->nw;
}

/* every block whose window is all in hist, at most `limit` outputs */
static size_t blocks(pk_resampler *r, float *out, uint64_t limit) {
    size_t made = 0, start = 0;
    while (r->have - start >= r->taps && r->n_out < limit) {
        const float *x = r->hist + start;
        for (uint32_t j = 0; j < r->nw && r->n_out < limit; j++) {
            const float *k = r->k + r->at[j], *xj = x + r->first[j];
            float acc = 0.0f;
            for (uint32_t m = 0; m < r->len[j]; m++) acc = pk_fmaf(k[m], xj[m], acc);
            out[made++] = acc;
            r->n_out++;
        }
        start += r->orig;
    }
    memmove(r->hist, r->hist + start, (r->have - start) * sizeof *r->hist);
    r->have -= start;
    return made;
}

size_t pk_resample(pk_resampler *r, const float *in, size_t n, float *out) {
    if (r->orig == r->nw) {
        memcpy(out, in, n * sizeof *in);
        r->n_in += n;
        return n;
    }
    size_t made = 0;
    while (n) {
        size_t k = r->cap - r->have < n ? r->cap - r->have : n;
        memcpy(r->hist + r->have, in, k * sizeof *in);
        r->have += k;
        r->n_in += k;
        in += k;
        n -= k;
        made += blocks(r, out + made, UINT64_MAX);
    }
    return made;
}

size_t pk_resample_flush(pk_resampler *r, float *out) {
    if (r->orig == r->nw)
        return 0;
    /* torchaudio: ceil(new_freq * length / orig_freq), in a Python float */
    uint64_t total = (uint64_t)ceil((double)r->nw * (double)r->n_in / r->orig);
    size_t made = 0;
    while (r->n_out < total) {
        size_t z = r->cap - r->have;
        memset(r->hist + r->have, 0, z * sizeof *r->hist);
        r->have += z;
        made += blocks(r, out + made, total);
    }
    return made;
}
