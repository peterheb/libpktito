/* Streaming boundaries are not part of a recording.  The same samples sent
 * whole or in fuzz-selected pieces must produce the same bits and exactly
 * torchaudio's ceil(to * n / from) output count. */
#include "resample.h"

#include "fuzz.h"

static uint32_t gcd32(uint32_t a, uint32_t b) {
    while (b) {
        uint32_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

static uint32_t rate(uint32_t v) {
    static const uint32_t common[] = {
        8000, 11025, 12000, 16000, 16001, 22050, 24000, 32000,
        44100, 48000, 88200, 96000,
    };
    if (!(v & 0x80000000u))
        return common[v % (sizeof common / sizeof common[0])];
    return 1 + v % 8192;
}

static size_t run_whole(uint32_t from, uint32_t to, const float *x, size_t n, float **out) {
    pk_resampler *r = pk_resampler_new(from, to);
    FZ_ASSERT(r);
    FZ_ASSERT(pk_resample_room(r, SIZE_MAX) == SIZE_MAX);
    size_t cap = pk_resample_room(r, n);
    FZ_ASSERT(cap != SIZE_MAX && cap <= 65536);
    size_t tail = pk_resample_room(r, 0);
    FZ_ASSERT(tail != SIZE_MAX && cap <= SIZE_MAX - tail);
    cap += tail;
    *out = malloc((cap ? cap : 1) * sizeof **out);
    FZ_ASSERT(*out);
    size_t made = pk_resample(r, x, n, *out);
    FZ_ASSERT(made <= cap);
    made += pk_resample_flush(r, *out + made);
    FZ_ASSERT(made <= cap && pk_resample_flush(r, *out + made) == 0);
    pk_resampler_free(r);
    return made;
}

static size_t run_pieces(uint32_t from, uint32_t to, const float *x, size_t n,
                         uint64_t schedule, float **out) {
    pk_resampler *r = pk_resampler_new(from, to);
    FZ_ASSERT(r);
    size_t cap = pk_resample_room(r, n), tail = pk_resample_room(r, 0);
    FZ_ASSERT(cap != SIZE_MAX && tail != SIZE_MAX && cap <= SIZE_MAX - tail);
    cap += tail;
    *out = malloc((cap ? cap : 1) * sizeof **out);
    FZ_ASSERT(*out);
    size_t at = 0, made = 0;
    while (at < n) {
        schedule = schedule * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
        size_t k = 1 + (size_t)(schedule >> 57);
        if (k > n - at)
            k = n - at;
        size_t room = pk_resample_room(r, k);
        size_t got = pk_resample(r, x + at, k, *out + made);
        FZ_ASSERT(got <= room && got <= cap - made);
        made += got;
        at += k;
    }
    size_t room = pk_resample_room(r, 0);
    size_t got = pk_resample_flush(r, *out + made);
    FZ_ASSERT(got <= room && got <= cap - made);
    made += got;
    FZ_ASSERT(pk_resample_flush(r, *out + made) == 0);
    pk_resampler_free(r);
    return made;
}

int LLVMFuzzerInitialize(int *argc, char ***argv) {
    (void)argc;
    (void)argv;
    fz_force_ref();
    return 0;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 8 || size > 8192)
        return 0;
    fz_input f = { data, size, 0 };
    uint32_t from = rate(fz_u32(&f)), to = rate(fz_u32(&f));
    uint32_t g = gcd32(from, to), orig = from / g, nw = to / g;
    if (orig > 4096 || nw > 4096 || nw > 16 * orig || orig > 16 * nw)
        return 0;                       /* keep one iteration bounded */

    size_t n = (size - f.at) / sizeof(float);
    if (n > 512)
        n = 512;
    float x[512];
    for (size_t i = 0; i < n; i++)
        x[i] = fz_float(&f);

    float *a, *b;
    size_t na = run_whole(from, to, x, n, &a);
    size_t nb = run_pieces(from, to, x, n, fz_hash(data, size), &b);
    size_t want = (size_t)ceil((double)nw * (double)n / orig);
    FZ_ASSERT(na == want && nb == want);
    FZ_ASSERT(!memcmp(a, b, want * sizeof *a));
    free(a);
    free(b);
    return 0;
}
