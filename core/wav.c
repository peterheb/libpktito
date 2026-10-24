/* The WAV reader and writer (wav.h).
 *
 * A WAV is a RIFF file: "RIFF", a size, "WAVE", then chunks of a four-byte
 * id, a 32-bit size and that many bytes, each padded to an even length. Two
 * chunks matter, "fmt " (what a sample is) and "data" (the samples); every
 * other one - LIST, fact, bext, JUNK, cue - is read past. RF64 is the same
 * with "RF64" in front and a "ds64" chunk first that holds the 64-bit sizes
 * the 32-bit fields cannot.
 *
 * Everything is read byte by byte into integers, little-endian by
 * construction, so the code is the same on a big-endian host, and nothing is
 * read with fseek, so a pipe reads like a file.
 */
#include "wav.h"
#include "os.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

/* WAVE_FORMAT_EXTENSIBLE's channel mask names 18 speakers; past this a
 * channel count is not a recording. */
#define PK_WAV_MAX_CHANNELS 256
#define PK_WAV_BUF 65536              /* bytes read at a time: a frame always fits */

struct pk_wav {
    FILE    *f;
    int      own;                     /* ours to fclose, not stdin */
    pk_wav_info info;
    uint32_t block;                   /* bytes a frame: channels * bytes */
    uint64_t left;                    /* data bytes not yet read */
    int      streamed;                /* no size: read to the end of the file */
    int      cut_short, io_error;
    unsigned char *buf;
    size_t   have, pos;               /* bytes in buf, and the next one to decode */
    float    law[256];                /* A-law or mu-law, decoded */
};

static int fail(char *err, size_t errlen, const char *fmt, ...) {
    if (err && errlen) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, errlen, fmt, ap);
        va_end(ap);
    }
    return -1;
}

static uint32_t le16(const unsigned char *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8; }
static uint32_t le32(const unsigned char *p) { return le16(p) | le16(p + 2) << 16; }
static uint64_t le64(const unsigned char *p) { return le32(p) | (uint64_t)le32(p + 4) << 32; }

static int rd(FILE *f, void *dst, size_t n) { return fread(dst, 1, n, f) == n ? 0 : -1; }

/* Read past n bytes. fread, not fseek, so that stdin can be read past. */
static int skip(FILE *f, uint64_t n) {
    unsigned char tmp[4096];
    while (n) {
        size_t k = n < sizeof tmp ? (size_t)n : sizeof tmp;
        if (rd(f, tmp, k))
            return -1;
        n -= k;
    }
    return 0;
}

/* ---------------------------------------------------------------- G.711 --- */

/* ITU-T G.711 to 16-bit linear, then /32768 as the PCM samples are. A
 * mu-law byte is stored inverted: a sign, a 3-bit segment s and a 4-bit
 * step q, and |x| = ((2q + 33) << s) - 33 in units of 4. An A-law byte has
 * its even bits flipped, and |x| = (2q + 1) << 3 in segment 0, else
 * (2q + 33) << (s + 2). */
static float mulaw(unsigned u) {
    u = ~u & 0xFF;
    int mag = ((((int)(u & 0x0F) << 3) + 0x84) << ((u >> 4) & 7)) - 0x84;
    return (float)((u & 0x80) ? -mag : mag) / 32768.0f;
}

static float alaw(unsigned a) {
    a ^= 0x55;
    int q = (int)(a & 0x0F), s = (int)((a >> 4) & 7);
    int mag = s == 0 ? (q << 4) + 8 : ((q << 4) + 0x108) << (s - 1);
    return (float)((a & 0x80) ? mag : -mag) / 32768.0f;
}

/* ----------------------------------------------------------------- read --- */

/* the GUID every KSDATAFORMAT_SUBTYPE_* shares after its leading format tag */
static const unsigned char ks_guid_tail[14] = {
    0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
};

static int parse_fmt(pk_wav *w, const unsigned char *p, uint32_t size, char *err, size_t errlen) {
    if (size < 16)
        return fail(err, errlen, "a %u-byte fmt chunk (16 at least)", (unsigned)size);
    uint32_t tag = le16(p), ch = le16(p + 2), rate = le32(p + 4), align = le16(p + 12);
    uint32_t bits = le16(p + 14), valid = bits;
    int ext = tag == 0xFFFE;
    if (ext) {                                   /* WAVE_FORMAT_EXTENSIBLE */
        if (size < 40 || le16(p + 16) < 22)
            return fail(err, errlen, "a WAVE_FORMAT_EXTENSIBLE fmt chunk too short to hold its format");
        if (memcmp(p + 26, ks_guid_tail, sizeof ks_guid_tail))
            return fail(err, errlen, "an extensible format that is not one of the standard ones");
        tag = le16(p + 24);
        valid = le16(p + 18);
    }
    if (tag != PK_WAV_PCM && tag != PK_WAV_FLOAT && tag != PK_WAV_ALAW && tag != PK_WAV_MULAW)
        return fail(err, errlen, "format 0x%04x; this reads PCM, float, A-law and mu-law", (unsigned)tag);
    if (ch == 0 || ch > PK_WAV_MAX_CHANNELS)
        return fail(err, errlen, "%u channels", (unsigned)ch);
    if (rate == 0)
        return fail(err, errlen, "a sample rate of 0");
    if (align == 0 || align % ch)
        return fail(err, errlen, "a block of %u bytes for %u channels", (unsigned)align, (unsigned)ch);
    /* Plain PCM's container is its bits rounded up to bytes (20 bits in 3).
     * EXTENSIBLE says the container in bits and the valid bits apart; they
     * are the high ones, so a container reads at its own full scale. */
    uint32_t bytes = align / ch;
    int ok = tag == PK_WAV_PCM ? bytes >= 1 && bytes <= 4 && valid >= 1
                                 && (ext ? bits == 8 * bytes && valid <= bits : (bits + 7) / 8 == bytes)
           : tag == PK_WAV_FLOAT ? (bytes == 4 || bytes == 8) && bits == 8 * bytes
           : bytes == 1 && bits == 8;
    if (!ok)
        return fail(err, errlen, "%u-bit samples in %u bytes for format %u",
                    (unsigned)bits, (unsigned)bytes, (unsigned)tag);
    w->info.format = tag;
    w->info.channels = ch;
    w->info.rate = rate;
    w->info.bytes = bytes;
    w->block = align;
    if (tag == PK_WAV_ALAW || tag == PK_WAV_MULAW)
        for (unsigned i = 0; i < 256; i++) w->law[i] = tag == PK_WAV_ALAW ? alaw(i) : mulaw(i);
    return 0;
}

static int parse(pk_wav *w, char *err, size_t errlen) {
    unsigned char h[12];
    if (rd(w->f, h, 12))
        return fail(err, errlen, "shorter than a RIFF header");
    int rf64 = !memcmp(h, "RF64", 4);
    if ((memcmp(h, "RIFF", 4) && !rf64) || memcmp(h + 8, "WAVE", 4))
        return fail(err, errlen, "not a WAV file");
    uint64_t data64 = 0;
    int have_fmt = 0, first = 1;
    for (;;) {
        unsigned char c[8];
        if (rd(w->f, c, 8))
            return fail(err, errlen, have_fmt ? "no data chunk" : "no fmt chunk");
        uint32_t size = le32(c + 4);
        if (rf64 && first && !memcmp(c, "ds64", 4)) {
            unsigned char d[24];
            if (size < 24 || rd(w->f, d, 24) || skip(w->f, (uint64_t)size - 24 + (size & 1)))
                return fail(err, errlen, "a short ds64 chunk");
            data64 = le64(d + 8);
        } else if (rf64 && first) {
            return fail(err, errlen, "an RF64 file whose first chunk is not ds64");
        } else if (!memcmp(c, "fmt ", 4)) {
            unsigned char f[40];
            uint32_t keep = size < sizeof f ? size : (uint32_t)sizeof f;
            if (have_fmt)
                return fail(err, errlen, "two fmt chunks");
            if (rd(w->f, f, keep) || skip(w->f, (uint64_t)(size - keep) + (size & 1)))
                return fail(err, errlen, "a fmt chunk cut short");
            if (parse_fmt(w, f, size, err, errlen))
                return -1;
            have_fmt = 1;
        } else if (!memcmp(c, "data", 4)) {
            if (!have_fmt)
                return fail(err, errlen, "a data chunk before the fmt chunk");
            uint64_t n = rf64 && size == 0xFFFFFFFFu ? data64 : size;
            w->streamed = !rf64 && size == 0xFFFFFFFFu;
            w->left = w->streamed ? UINT64_MAX : n - n % w->block;
            w->info.frames = w->streamed ? PK_WAV_UNKNOWN : n / w->block;
            return 0;
        } else if (skip(w->f, (uint64_t)size + (size & 1))) {
            return fail(err, errlen, "a %.4s chunk cut short", (const char *)c);
        }
        first = 0;
    }
}

pk_wav *pk_wav_open(const char *path, pk_wav_info *info, char *err, size_t errlen) {
    if (!path) {
        fail(err, errlen, "no WAV path");
        return NULL;
    }
    pk_wav *w = calloc(1, sizeof *w);
    unsigned char *buf = malloc(PK_WAV_BUF);
    if (!w || !buf) {
        free(w);
        free(buf);
        fail(err, errlen, "%s: out of memory", path);
        return NULL;
    }
    w->buf = buf;
    if (!strcmp(path, "-")) {
        w->f = stdin;
#ifdef _WIN32
        _setmode(_fileno(stdin), _O_BINARY);
#endif
    } else {
        w->f = pk_os_fopen(path, "rb");
        w->own = 1;
        if (!w->f) {
            fail(err, errlen, "%s: %s", path, errno == EILSEQ ? "not a UTF-8 path" : strerror(errno));
            free(buf);
            free(w);
            return NULL;
        }
    }
    char why[160];
    if (parse(w, why, sizeof why)) {
        fail(err, errlen, "%s: %s", path, why);
        pk_wav_close(w);
        return NULL;
    }
    if (info)
        *info = w->info;
    return w;
}

/* one stored sample at p, as float */
static float sample(const pk_wav *w, const unsigned char *p) {
    switch (w->info.format) {
    case PK_WAV_FLOAT:
        if (w->info.bytes == 4) {
            uint32_t u = le32(p);
            float f;
            memcpy(&f, &u, sizeof f);
            return f;
        } else {
            uint64_t u = le64(p);
            double d;
            memcpy(&d, &u, sizeof d);
            return (float)d;
        }
    case PK_WAV_PCM: {
        /* two's complement by arithmetic: C11 leaves converting an
         * out-of-range unsigned value to a signed type to the compiler */
        int64_t v;
        switch (w->info.bytes) {
        case 1: return (float)((int)p[0] - 128) / 128.0f;
        case 2: v = le16(p); return (float)(v - (v >> 15 << 16)) / 32768.0f;
        case 3: v = le16(p) | (uint32_t)p[2] << 16; return (float)(v - (v >> 23 << 24)) / 8388608.0f;
        default: v = le32(p); return (float)(v - (v >> 31 << 32)) / 2147483648.0f;
        }
    }
    default:
        return w->law[p[0]];
    }
}

/* Up to n frames into out: each mixed to mono (n floats), or each channel
 * as it is (n * channels floats, interleaved). */
static size_t read_frames(pk_wav *w, float *out, size_t n, int mono) {
    size_t done = 0;
    const uint32_t ch = w->info.channels, bytes = w->info.bytes;
    while (done < n) {
        if (w->have - w->pos < w->block) {
            /* keep the partial frame, then fill the rest of the buffer */
            size_t part = w->have - w->pos;
            memmove(w->buf, w->buf + w->pos, part);
            w->have = part;
            w->pos = 0;
            /* no more than the frames asked for: fread on a pipe waits for
             * all it is asked for, and a live stream wants its 32 ms now */
            size_t want = PK_WAV_BUF - part;
            if ((n - done) < want / w->block)
                want = (n - done) * w->block - part;
            if (want > w->left)
                want = (size_t)w->left;
            if (want == 0)
                break;
            size_t got = fread(w->buf + part, 1, want, w->f);
            w->have += got;
            w->left -= got;
            if (got < want) {
                if (ferror(w->f))
                    w->io_error = 1;
                else if (!w->streamed)
                    w->cut_short = 1;
                w->left = 0;
            }
            if (w->have < w->block)
                break;
        }
        const unsigned char *p = w->buf + w->pos;
        if (ch == 1) {
            out[done] = sample(w, p);
        } else if (!mono) {
            for (uint32_t c = 0; c < ch; c++) out[done * ch + c] = sample(w, p + c * bytes);
        } else {
            float acc = 0.0f;
            for (uint32_t c = 0; c < ch; c++) acc += sample(w, p + c * bytes);
            out[done] = acc / (float)ch;
        }
        w->pos += w->block;
        done++;
    }
    return done;
}

size_t pk_wav_read(pk_wav *w, float *out, size_t n) {
    return read_frames(w, out, n, 1);
}

size_t pk_wav_read_frames(pk_wav *w, float *out, size_t n) {
    return read_frames(w, out, n, 0);
}

int pk_wav_status(const pk_wav *w, char *err, size_t errlen) {
    if (w->io_error)
        return fail(err, errlen, "a read error before the end of the samples");
    if (w->cut_short)
        return fail(err, errlen, "the file ends before the %llu frames its header promised",
                    (unsigned long long)w->info.frames);
    return 0;
}

void pk_wav_close(pk_wav *w) {
    if (!w)
        return;
    if (w->own && w->f)
        fclose(w->f);
    free(w->buf);
    free(w);
}

/* ---------------------------------------------------------------- write --- */

struct pk_wav_out {
    FILE    *f;
    int      own, float32, error;
    uint32_t rate;
    uint64_t bytes;                   /* sample bytes written */
};

static void put16(unsigned char *p, uint32_t v) { p[0] = (unsigned char)v; p[1] = (unsigned char)(v >> 8); }
static void put32(unsigned char *p, uint32_t v) { put16(p, v & 0xFFFF); put16(p + 2, v >> 16); }

/* The header for `bytes` of samples. A float file carries cbSize and a fact
 * chunk, as WAVE_FORMAT_IEEE_FLOAT asks; PCM needs neither. A size past 32
 * bits is written as 0xFFFFFFFF, a stream's. */
static size_t header(unsigned char *h, const pk_wav_out *o, uint64_t bytes) {
    const uint32_t fmt = o->float32 ? 18 : 16, width = o->float32 ? 4 : 2;
    const size_t len = 12 + 8 + fmt + (o->float32 ? 12 : 0) + 8;
    uint64_t riff = len - 8 + bytes;
    memcpy(h, "RIFF", 4);
    put32(h + 4, riff > 0xFFFFFFFFu ? 0xFFFFFFFFu : (uint32_t)riff);
    memcpy(h + 8, "WAVEfmt ", 8);
    put32(h + 16, fmt);
    put16(h + 20, o->float32 ? PK_WAV_FLOAT : PK_WAV_PCM);
    put16(h + 22, 1);
    put32(h + 24, o->rate);
    put32(h + 28, o->rate * width);
    put16(h + 32, width);
    put16(h + 34, 8 * width);
    unsigned char *p = h + 36;
    if (o->float32) {
        put16(p, 0);
        memcpy(p + 2, "fact", 4);
        put32(p + 6, 4);
        put32(p + 10, bytes / width > 0xFFFFFFFFu ? 0xFFFFFFFFu : (uint32_t)(bytes / width));
        p += 14;
    }
    memcpy(p, "data", 4);
    put32(p + 4, bytes > 0xFFFFFFFFu ? 0xFFFFFFFFu : (uint32_t)bytes);
    return len;
}

pk_wav_out *pk_wav_create(const char *path, uint32_t rate, int float32, char *err, size_t errlen) {
    pk_wav_out *o = calloc(1, sizeof *o);
    if (!o) {
        fail(err, errlen, "%s: out of memory", path);
        return NULL;
    }
    o->rate = rate;
    o->float32 = float32;
    if (!strcmp(path, "-")) {
        o->f = stdout;
#ifdef _WIN32
        _setmode(_fileno(stdout), _O_BINARY);
#endif
    } else {
        o->f = pk_os_fopen(path, "wb");
        o->own = 1;
        if (!o->f) {
            fail(err, errlen, "%s: %s", path, errno == EILSEQ ? "not a UTF-8 path" : strerror(errno));
            free(o);
            return NULL;
        }
    }
    unsigned char h[64];
    size_t len = header(h, o, 0xFFFFFFFFu);     /* a stream's, until finish knows */
    if (fwrite(h, 1, len, o->f) != len)
        o->error = 1;
    return o;
}

int pk_wav_write(pk_wav_out *o, const float *x, size_t n) {
    unsigned char buf[4096];
    const size_t width = o->float32 ? 4 : 2, per = sizeof buf / width;
    for (size_t i = 0; i < n; i += per) {
        size_t k = n - i < per ? n - i : per;
        for (size_t j = 0; j < k; j++) {
            float v = x[i + j];
            if (o->float32) {
                uint32_t u;
                memcpy(&u, &v, sizeof u);
                put32(buf + 4 * j, u);
            } else {
                /* round to nearest; out of range clamps, NaN is silence */
                float s = v * 32768.0f;
                long q = s != s ? 0 : s >= 32767.0f ? 32767 : s <= -32768.0f ? -32768
                       : (long)(s + (s >= 0 ? 0.5f : -0.5f));
                put16(buf + 2 * j, (uint32_t)(q & 0xFFFF));
            }
        }
        if (fwrite(buf, width, k, o->f) != k)
            o->error = 1;
        o->bytes += k * width;
    }
    return o->error ? -1 : 0;
}

int pk_wav_finish(pk_wav_out *o, char *err, size_t errlen) {
    int bad = o->error;
    /* a file gets its real sizes; a pipe cannot seek and keeps a stream's */
    if (!bad && o->own && fseek(o->f, 0, SEEK_SET) == 0) {
        unsigned char h[64];
        size_t len = header(h, o, o->bytes);
        if (fwrite(h, 1, len, o->f) != len)
            bad = 1;
    }
    if (fflush(o->f))
        bad = 1;
    if (o->own && fclose(o->f))
        bad = 1;
    free(o);
    return bad ? fail(err, errlen, "a write failed") : 0;
}
