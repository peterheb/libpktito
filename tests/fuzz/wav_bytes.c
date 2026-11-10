/* A WAV opened repeatedly from the same bytes must not depend on how its
 * caller partitions reads.  The interleaved and mono APIs must see the same
 * frames, including NaNs and infinities stored by a float WAV, and a read
 * from a frame (pk_wav_seek) what a front to back read has there, in the
 * bytes as a regular file and with the samples copied out of a stream that
 * can't seek (pk_wav_spool). */
#define _GNU_SOURCE                   /* memfd_create */
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

#ifdef __linux__
#include <sys/mman.h>

/* the bytes as a regular file, which pk_wav_seekable needs and fmemopen's
 * stream is not: a memfd, in memory too */
static FILE *memfile(const uint8_t *data, size_t size) {
    const int fd = memfd_create("wav_bytes", 0);
    FZ_ASSERT(fd >= 0);
    FILE *f = fdopen(fd, "w+b");
    FZ_ASSERT(f && fwrite(data, 1, size, f) == size && fseek(f, 0, SEEK_SET) == 0);
    return f;
}

/* reads from frames here and there in w, which nf frames can be read from,
 * against a, a front to back read */
static void reads(pk_wav *w, uint64_t nf, size_t size, const decoded *a) {
    char err[256];
    FZ_ASSERT(nf == a->frames);
    FZ_ASSERT((pk_wav_status(w, err, sizeof err) != 0) == (a->status != 0));
    float y[64];
    const uint64_t at[] = { 0, nf / 2, nf ? nf - 1 : 0, nf, (uint64_t)size * 2654435761u % (nf + 1) };
    for (size_t i = 0; i < sizeof at / sizeof *at; i++) {
        const size_t n = nf - at[i] < 64 ? (size_t)(nf - at[i]) : 64;
        FZ_ASSERT(pk_wav_seek(w, at[i]) == 0 && pk_wav_read(w, y, 64) == n);
        FZ_ASSERT(!memcmp(y, a->x + at[i], n * sizeof *y));
    }
    FZ_ASSERT(pk_wav_seek(w, nf + 1) == -1);
}

/* the bytes as a regular file (pk_wav_seekable), and as fmemopen's stream,
 * which can't seek, with its samples copied into an empty memfd
 * (pk_wav_spool) */
static void seeks(const uint8_t *data, size_t size, const decoded *a) {
    char err[256];
    pk_wav *w = open_file(memfile(data, size), 1, "wav_bytes", NULL, err, sizeof err);
    uint64_t nf = 0;
    FZ_ASSERT(w && pk_wav_seekable(w, &nf) == 0);
    reads(w, nf, size, a);
    pk_wav_close(w);
    pk_wav_info info;
    FILE *to = memfile(data, 0);
    w = open_bytes(data, size, &info, err, sizeof err);
    FZ_ASSERT(w && pk_wav_seek(w, 0) == -1 && pk_wav_spool(w, to, &nf) == 0);
    reads(w, nf, size, a);
    pk_wav_close(w);
    fclose(to);
}
#endif

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
#ifdef __linux__
    seeks(data, size, &a);
#endif
    free(a.x);
    free(b.x);
    free(c.x);
    return 0;
}
