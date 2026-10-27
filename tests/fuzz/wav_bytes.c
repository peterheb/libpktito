/* A WAV opened repeatedly from the same bytes must not depend on how its
 * caller partitions reads.  The interleaved and mono APIs must see the same
 * frames, including NaNs and infinities stored by a float WAV. */
#include "wav.c"                      /* for parse(), reached without a path */

#include "fuzz.h"

/* pk_wav_open over the fuzzer's bytes: the same parser and decoder, fed by
 * fmemopen, so that core/wav.c keeps its one entry point and no iteration
 * writes a file. */
static pk_wav *open_bytes(const uint8_t *data, size_t size, pk_wav_info *info,
                          char *err, size_t errlen) {
    pk_wav *w = calloc(1, sizeof *w);
    unsigned char *buf = malloc(PK_WAV_BUF);
    FZ_ASSERT(w && buf);
    w->buf = buf;
    w->f = fmemopen((void *)data, size, "rb");
    w->own = 1;
    FZ_ASSERT(w->f);
    if (parse(w, err, errlen)) {
        pk_wav_close(w);
        return NULL;
    }
    *info = w->info;
    return w;
}

typedef struct {
    pk_wav_info info;
    float *x;
    size_t frames;
    int status;
} decoded;

static int same_info(const pk_wav_info *a, const pk_wav_info *b) {
    return a->rate == b->rate && a->channels == b->channels && a->format == b->format
        && a->bytes == b->bytes && a->frames == b->frames;
}

static int mono(const uint8_t *data, size_t size, size_t piece, decoded *d) {
    char err[256] = "";
    pk_wav *w = open_bytes(data, size, &d->info, err, sizeof err);
    if (!w) {
        FZ_ASSERT(err[0]);
        return -1;
    }
    size_t cap = size + 1;
    d->x = malloc(cap * sizeof *d->x);
    FZ_ASSERT(d->x);
    d->frames = 0;
    while (d->frames < cap) {
        size_t want = piece < cap - d->frames ? piece : cap - d->frames;
        size_t n = pk_wav_read(w, d->x + d->frames, want);
        FZ_ASSERT(n <= want);
        d->frames += n;
        if (n < want)
            break;
    }
    FZ_ASSERT(d->frames < cap);        /* decoded frames cannot outnumber input bytes */
    d->status = pk_wav_status(w, err, sizeof err);
    if (d->status)
        FZ_ASSERT(err[0]);
    pk_wav_close(w);
    return 0;
}

static int frames(const uint8_t *data, size_t size, size_t piece, decoded *d) {
    char err[256] = "";
    pk_wav *w = open_bytes(data, size, &d->info, err, sizeof err);
    if (!w) {
        FZ_ASSERT(err[0]);
        return -1;
    }
    size_t cap = size + 256;           /* floats; data bytes bound frames * channels */
    d->x = malloc(cap * sizeof *d->x);
    FZ_ASSERT(d->x);
    d->frames = 0;
    for (;;) {
        size_t room = cap / d->info.channels - d->frames;
        FZ_ASSERT(room);
        size_t want = piece < room ? piece : room;
        size_t n = pk_wav_read_frames(w, d->x + d->frames * d->info.channels, want);
        FZ_ASSERT(n <= want);
        d->frames += n;
        if (n < want)
            break;
    }
    d->status = pk_wav_status(w, err, sizeof err);
    if (d->status)
        FZ_ASSERT(err[0]);
    pk_wav_close(w);
    return 0;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (!size || size > 1024 * 1024)  /* fmemopen may refuse an empty buffer */
        return 0;
    decoded a = { 0 }, b = { 0 }, c = { 0 };
    int ra = mono(data, size, 1, &a);
    int rb = mono(data, size, 4096, &b);
    FZ_ASSERT(ra == rb);
    if (ra)
        return 0;

    FZ_ASSERT(same_info(&a.info, &b.info));
    FZ_ASSERT(a.frames == b.frames && a.status == b.status);
    FZ_ASSERT(!memcmp(a.x, b.x, a.frames * sizeof *a.x));

    int rc = frames(data, size, 7, &c);
    FZ_ASSERT(rc == 0 && same_info(&a.info, &c.info));
    FZ_ASSERT(a.frames == c.frames && a.status == c.status);
    for (size_t i = 0; i < a.frames; i++) {
        float v;
        if (c.info.channels == 1) {
            v = c.x[i];
        } else {
            float sum = 0;
            for (uint32_t ch = 0; ch < c.info.channels; ch++)
                sum += c.x[i * c.info.channels + ch];
            v = sum / (float)c.info.channels;
        }
        FZ_ASSERT(!memcmp(&v, &a.x[i], sizeof v));
    }
    free(a.x);
    free(b.x);
    free(c.x);
    return 0;
}
