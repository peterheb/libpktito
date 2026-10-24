/* The WAV reader and writer (core/wav.c) on files this test writes itself.
 * vadito's tests/wavcheck.c, renamed with the library it checks.
 *
 *   formats    PCM 8, 16, 24 and 32 bits, float 32 and 64, A-law and mu-law,
 *              EXTENSIBLE around PCM and float, one to six channels: every
 *              sample is the value torchaudio's conventions give, exactly,
 *              and more than one channel is their mean
 *   G.711      all 256 codes of each law: the anchors of the standard's
 *              tables, and order within each sign
 *   layout     chunks before fmt and between fmt and data, odd sizes and
 *              their pad bytes, RF64, a stream's sizes, a data chunk that
 *              stops mid-frame, a file that stops before its data does
 *   pieces     any file read a frame at a time, in odd-sized reads and all at
 *              once comes out the same
 *   writer     16-bit and float, read back
 *   malformed  every prefix of a valid file, and each field of fmt wrong in
 *              each way it can be: refused with a message, never a crash
 *
 * `make check-sanitize` runs it under ASan and UBSan.
 *
 *     ./tests/wavcheck [scratch-file]
 */
#include "wav.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_bad, g_cases;
static const char *g_tmp = "tests/wavcheck.tmp";

#define FAIL(...) do { if (g_bad++ < 40) { printf("FAIL "); printf(__VA_ARGS__); printf("\n"); } } while (0)

/* ------------------------------------------------------------ a writer --- */

typedef struct { unsigned char *p; size_t n, cap; } buf;

static void put(buf *b, const void *d, size_t n) {
    if (b->n + n > b->cap) {
        b->cap = (b->n + n) * 2;
        b->p = realloc(b->p, b->cap);
    }
    memcpy(b->p + b->n, d, n);
    b->n += n;
}
static void u8(buf *b, unsigned v) { unsigned char c = (unsigned char)v; put(b, &c, 1); }
static void u16(buf *b, unsigned v) { u8(b, v & 0xFF); u8(b, (v >> 8) & 0xFF); }
static void u32(buf *b, uint32_t v) { u16(b, v & 0xFFFF); u16(b, v >> 16); }
static void u64(buf *b, uint64_t v) { u32(b, (uint32_t)v); u32(b, (uint32_t)(v >> 32)); }
static void tag(buf *b, const char *t) { put(b, t, 4); }

typedef struct {
    unsigned format, channels, rate, bytes, bits;
    int extensible, rf64, streamed, junk_before, junk_between, cut_data;
} spec;

/* A file of `frames` frames whose sample (f, c) is `code(f, c)`, stored raw */
static buf make(const spec *s, const unsigned char *data, size_t data_len) {
    buf b = { 0 };
    tag(&b, s->rf64 ? "RF64" : "RIFF");
    u32(&b, s->rf64 || s->streamed ? 0xFFFFFFFFu : 0);   /* fixed below */
    tag(&b, "WAVE");
    if (s->rf64) {
        tag(&b, "ds64");
        u32(&b, 28);
        u64(&b, 0);
        u64(&b, data_len);
        u64(&b, data_len / (s->channels * s->bytes));
        u32(&b, 0);
    }
    if (s->junk_before) {
        tag(&b, "LIST");
        u32(&b, 5);
        put(&b, "INFOx", 5);
        u8(&b, 0);                                       /* the pad byte */
    }
    tag(&b, "fmt ");
    u32(&b, s->extensible ? 40 : 16);
    u16(&b, s->extensible ? 0xFFFE : s->format);
    u16(&b, s->channels);
    u32(&b, s->rate);
    u32(&b, s->rate * s->channels * s->bytes);
    u16(&b, s->channels * s->bytes);
    u16(&b, s->extensible ? 8 * s->bytes : s->bits);    /* EXTENSIBLE: the container */
    if (s->extensible) {
        static const unsigned char tail[14] = { 0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xAA, 0, 0x38, 0x9B, 0x71 };
        u16(&b, 22);
        u16(&b, s->bits);                                /* valid bits */
        u32(&b, 0);
        u16(&b, s->format);
        put(&b, tail, sizeof tail);
    }
    if (s->junk_between) {
        tag(&b, "fact");
        u32(&b, 3);
        put(&b, "abc", 3);
        u8(&b, 0);
    }
    tag(&b, "data");
    u32(&b, s->rf64 || s->streamed ? 0xFFFFFFFFu : (uint32_t)data_len + (uint32_t)s->cut_data);
    put(&b, data, data_len);
    if (!s->rf64 && !s->streamed) {
        uint32_t riff = (uint32_t)b.n - 8;
        memcpy(b.p + 4, (unsigned char[4]){ riff & 0xFF, (riff >> 8) & 0xFF, (riff >> 16) & 0xFF, riff >> 24 }, 4);
    }
    return b;
}

static void save(const buf *b) {
    FILE *f = fopen(g_tmp, "wb");
    if (!f || fwrite(b->p, 1, b->n, f) != b->n) {
        printf("wavcheck: cannot write %s\n", g_tmp);
        exit(2);
    }
    fclose(f);
}

/* ------------------------------------------------- the expected values --- */

static float mulaw_ref(unsigned u);
static float alaw_ref(unsigned a);

/* sample i of a test signal, as stored and as the reader should give it */
static void sample(const spec *s, uint32_t i, unsigned char *out, float *want) {
    uint32_t r = i * 2654435761u;                  /* a scramble of i, all bit patterns */
    switch (s->format == 3 ? 100 + s->bytes : s->format == 1 ? s->bytes : s->format) {
    case 1: out[0] = (unsigned char)r; *want = ((int)out[0] - 128) / 128.0f; break;
    case 2: {
        int v = (int)(r >> 16) - 32768;
        if (i == 0) v = -32768;
        if (i == 1) v = 32767;
        out[0] = (unsigned char)v; out[1] = (unsigned char)(v >> 8);
        *want = v / 32768.0f;
        break;
    }
    case 3: {
        int32_t v = (int32_t)(r >> 8) - 8388608;
        out[0] = (unsigned char)v; out[1] = (unsigned char)(v >> 8); out[2] = (unsigned char)(v >> 16);
        *want = (float)v / 8388608.0f;
        break;
    }
    case 4: {
        int64_t v = (int64_t)r - 2147483648ll;
        for (int k = 0; k < 4; k++) out[k] = (unsigned char)((uint64_t)v >> (8 * k));
        *want = (float)v / 2147483648.0f;
        break;
    }
    case 104: {
        float v = (float)((double)r - 2147483648.0) / 1e9f;
        uint32_t u;
        memcpy(&u, &v, 4);
        for (int k = 0; k < 4; k++) out[k] = (unsigned char)(u >> (8 * k));
        *want = v;
        break;
    }
    case 108: {
        double v = ((double)r - 2147483648.0) / 3e9;
        uint64_t u;
        memcpy(&u, &v, 8);
        for (int k = 0; k < 8; k++) out[k] = (unsigned char)(u >> (8 * k));
        *want = (float)v;
        break;
    }
    case 6: out[0] = (unsigned char)r; *want = alaw_ref(out[0]); break;
    case 7: out[0] = (unsigned char)r; *want = mulaw_ref(out[0]); break;
    }
}

/* G.711 from its definition, written apart from core/wav.c's: mu-law is
 * sign | 3-bit segment | 4-bit step, inverted, x = (33 + 2 step) 2^seg - 33
 * in units of 4; A-law, its even bits flipped, (1 + 2 step) 8 in segment 0
 * and (33 + 2 step) 2^(seg + 2) above it */
static float mulaw_ref(unsigned u) {
    unsigned v = 255 - u;
    int seg = (v >> 4) & 7, step = v & 15;
    int x = 4 * (((33 + 2 * step) << seg) - 33);
    return (float)((v & 128) ? -x : x) / 32768.0f;
}
static float alaw_ref(unsigned a) {
    unsigned v = a ^ 0x55;
    int seg = (v >> 4) & 7, step = v & 15;
    int x = seg == 0 ? 8 * (1 + 2 * step) : (33 + 2 * step) << (seg + 2);
    return (float)((v & 128) ? x : -x) / 32768.0f;
}

/* ---------------------------------------------------------------- read --- */

/* read `path` with reads of `piece` frames (0: all at once) */
static float *read_all(const char *path, size_t piece, size_t *frames, pk_wav_info *info, int *status) {
    char err[256];
    pk_wav *w = pk_wav_open(path, info, err, sizeof err);
    if (!w) {
        *frames = 0;
        return NULL;
    }
    size_t cap = 1 << 16, n = 0;
    float *x = malloc(cap * sizeof *x);
    for (;;) {
        size_t want = piece ? piece : 65536, k;
        if (n + want > cap)
            x = realloc(x, (cap = 2 * (n + want)) * sizeof *x);
        n += k = pk_wav_read(w, x + n, want);
        if (k < want)
            break;
    }
    *status = pk_wav_status(w, err, sizeof err);
    pk_wav_close(w);
    *frames = n;
    return x;
}

static void format_case(spec s, uint32_t frames) {
    const uint32_t ch = s.channels, bytes = s.bytes;
    size_t len = (size_t)frames * ch * bytes;
    unsigned char *data = malloc(len + 16);
    float *want = malloc((size_t)frames * sizeof *want);
    float *want_ch = malloc((size_t)frames * ch * sizeof *want_ch);
    for (uint32_t f = 0; f < frames; f++) {
        float acc = 0.0f, v = 0.0f;
        for (uint32_t c = 0; c < ch; c++) {
            sample(&s, f * ch + c, data + ((size_t)f * ch + c) * bytes, &v);
            want_ch[(size_t)f * ch + c] = v;
            acc += v;
        }
        want[f] = ch == 1 ? v : acc / (float)ch;          /* one channel is the sample itself */
    }
    size_t extra = 0;
    if (s.cut_data < 0) {                          /* a partial frame after the last whole one */
        extra = bytes;
        memset(data + len, 0x5A, extra);
    }
    buf b = make(&s, data, len + extra);
    if (s.cut_data > 0)
        s.cut_data = 0;
    save(&b);
    char what[160];
    snprintf(what, sizeof what, "format %u, %u bytes, %u channels%s%s%s%s%s", s.format, bytes, ch,
             s.extensible ? ", extensible" : "", s.rf64 ? ", RF64" : "", s.streamed ? ", streamed" : "",
             s.junk_before || s.junk_between ? ", other chunks" : "", extra ? ", a partial frame" : "");
    static const size_t pieces[] = { 0, 1, 7, 4096 };
    for (size_t pi = 0; pi < 4; pi++) {
        pk_wav_info info;
        size_t n;
        int st;
        float *x = read_all(g_tmp, pieces[pi], &n, &info, &st);
        g_cases++;
        if (!x) {
            FAIL("%s: did not open", what);
            break;
        }
        if (n != frames || st)
            FAIL("%s, reads of %zu: %zu frames of %u, status %d", what, pieces[pi], n, frames, st);
        else if (memcmp(x, want, (size_t)frames * sizeof *x)) {
            for (uint32_t f = 0; f < frames; f++)
                if (memcmp(&x[f], &want[f], sizeof *x)) {
                    FAIL("%s: frame %u is %a, want %a", what, f, x[f], want[f]);
                    break;
                }
        }
        if (info.rate != s.rate || info.channels != ch || info.bytes != bytes || info.format != s.format
            || info.frames != (s.streamed ? PK_WAV_UNKNOWN : frames))
            FAIL("%s: info %u Hz %u ch %u bytes format %u frames %llu", what, (unsigned)info.rate,
                 (unsigned)info.channels, (unsigned)info.bytes, (unsigned)info.format,
                 (unsigned long long)info.frames);
        free(x);
    }
    /* pk_wav_read_frames: every channel as it is, interleaved, the same
     * samples the mono read averages */
    static const size_t frame_pieces[] = { 0, 7 };
    for (size_t pi = 0; pi < 2; pi++) {
        pk_wav_info info;
        char err[256];
        pk_wav *w = pk_wav_open(g_tmp, &info, err, sizeof err);
        g_cases++;
        if (!w) {
            FAIL("%s: did not open for frames", what);
            break;
        }
        float *x = malloc(((size_t)frames + 4096) * ch * sizeof *x);
        size_t n = 0;
        for (;;) {
            size_t want_n = frame_pieces[pi] ? frame_pieces[pi] : 4096, k;
            if (n + want_n > (size_t)frames + 4096)
                want_n = (size_t)frames + 4096 - n;
            n += k = pk_wav_read_frames(w, x + n * ch, want_n);
            if (k < want_n || want_n == 0)
                break;
        }
        int st = pk_wav_status(w, err, sizeof err);
        pk_wav_close(w);
        if (n != frames || st)
            FAIL("%s, frames in reads of %zu: %zu of %u, status %d", what, frame_pieces[pi], n, frames, st);
        else if (memcmp(x, want_ch, (size_t)frames * ch * sizeof *x))
            FAIL("%s, frames in reads of %zu: a sample differs", what, frame_pieces[pi]);
        free(x);
    }
    free(b.p);
    free(data);
    free(want);
    free(want_ch);
}

static void refuse(const char *what, const buf *b) {
    char err[256] = "";
    save(b);
    pk_wav_info info;
    pk_wav *w = pk_wav_open(g_tmp, &info, err, sizeof err);
    g_cases++;
    if (w) {
        FAIL("%s: opened", what);
        pk_wav_close(w);
    } else if (!err[0]) {
        FAIL("%s: refused without a reason", what);
    }
}

int main(int argc, char **argv) {
    if (argc > 1)
        g_tmp = argv[1];

    {
        char err[256] = "";
        pk_wav_info info;
        pk_wav *w = pk_wav_open(NULL, &info, err, sizeof err);
        g_cases++;
        if (w || !err[0])
            FAIL("a null WAV path: %s", w ? "opened" : "no reason");
        pk_wav_close(w);
    }

    /* ---- formats */
    static const struct { unsigned format, bytes, bits; } fmts[] = {
        { 1, 1, 8 }, { 1, 2, 16 }, { 1, 3, 24 }, { 1, 3, 20 }, { 1, 4, 32 },
        { 3, 4, 32 }, { 3, 8, 64 }, { 6, 1, 8 }, { 7, 1, 8 },
    };
    for (size_t i = 0; i < sizeof fmts / sizeof fmts[0]; i++)
        for (unsigned ch = 1; ch <= 6; ch += (ch == 2 ? 3 : 1)) {
            spec s = { fmts[i].format, ch, 16000, fmts[i].bytes, fmts[i].bits, 0, 0, 0, 0, 0, 0 };
            format_case(s, 3001);
            s.extensible = fmts[i].format < 4;
            if (s.extensible) format_case(s, 3001);
        }

    /* ---- layout */
    spec s16 = { 1, 1, 16000, 2, 16, 0, 0, 0, 0, 0, 0 };
    spec s = s16;
    s.junk_before = s.junk_between = 1;
    format_case(s, 777);
    s = s16; s.rf64 = 1; format_case(s, 777);
    s = s16; s.streamed = 1; format_case(s, 777);
    s = s16; s.format = 3; s.bytes = 8; s.bits = 64; s.channels = 256;
    format_case(s, 37);                                  /* the widest legal frame */
    s = s16; s.cut_data = -1; format_case(s, 777);            /* data ends mid-frame */
    s = s16; s.channels = 2; s.cut_data = -1; format_case(s, 777);
    {
        /* the header promises 100 bytes more than there are */
        unsigned char d[200];
        for (int i = 0; i < 200; i++) d[i] = (unsigned char)i;
        spec c = s16;
        c.cut_data = 100;
        buf b = make(&c, d, 200);
        save(&b);
        pk_wav_info info;
        size_t n;
        int st;
        float *x = read_all(g_tmp, 0, &n, &info, &st);
        g_cases++;
        if (!x || n != 100 || st == 0)
            FAIL("cut short: %zu frames, status %d (want 100 and an error)", n, st);
        free(x);
        free(b.p);
    }
    {
        /* A caller may ask for every possible size_t frame; the file, not
         * that count, bounds the fixed-buffer read and its arithmetic. */
        unsigned char d[2] = { 0 };
        spec one = s16;
        one.streamed = 1;
        buf b = make(&one, d, sizeof d);
        save(&b);
        char err[256];
        pk_wav_info info;
        pk_wav *w = pk_wav_open(g_tmp, &info, err, sizeof err);
        float x;
        g_cases++;
        if (!w || pk_wav_read(w, &x, SIZE_MAX) != 1)
            FAIL("a SIZE_MAX read of a one-frame stream");
        pk_wav_close(w);
        free(b.p);
    }
    {
        /* RF64 supplies the data length as a hostile 64-bit integer. */
        unsigned char d[2] = { 0 };
        spec huge = s16;
        huge.rf64 = 1;
        buf b = make(&huge, d, sizeof d);
        memset(b.p + 28, 0xFF, 8);                       /* ds64.dataSize */
        save(&b);
        char err[256];
        pk_wav_info info;
        pk_wav *w = pk_wav_open(g_tmp, &info, err, sizeof err);
        float x[2];
        size_t n = w ? pk_wav_read(w, x, 2) : 0;
        int st = w ? pk_wav_status(w, err, sizeof err) : 0;
        g_cases++;
        if (!w || info.frames != UINT64_MAX / 2 || n != 1 || st == 0)
            FAIL("RF64 UINT64_MAX data size: %zu frames read, status %d", n, st);
        pk_wav_close(w);
        free(b.p);
    }

    /* ---- G.711 anchors and order */
    {
        g_cases++;
        if (mulaw_ref(0xFF) != 0 || mulaw_ref(0x7F) != 0 || mulaw_ref(0x00) != -32124 / 32768.0f
            || mulaw_ref(0x80) != 32124 / 32768.0f || alaw_ref(0xD5) != 8 / 32768.0f
            || alaw_ref(0x55) != -8 / 32768.0f || alaw_ref(0xAA) != 32256 / 32768.0f
            || alaw_ref(0x2A) != -32256 / 32768.0f)
            FAIL("G.711 reference anchors");
        for (unsigned u = 0x80; u < 0xFF; u++)
            if (!(mulaw_ref(u) > mulaw_ref(u + 1)))
                FAIL("mu-law order at %02x", u);
    }

    /* ---- writer */
    for (int fl = 0; fl < 2; fl++) {
        float x[1000], *y;
        for (int i = 0; i < 1000; i++) x[i] = sinf(i * 0.37f) * 1.2f;   /* past full scale too */
        x[3] = NAN;
        char err[256];
        pk_wav_out *o = pk_wav_create(g_tmp, 8000, fl, err, sizeof err);
        g_cases++;
        if (!o || pk_wav_write(o, x, 600) || pk_wav_write(o, x + 600, 400) || pk_wav_finish(o, err, sizeof err)) {
            FAIL("write %s: %s", fl ? "float" : "16-bit", err);
            continue;
        }
        pk_wav_info info;
        size_t n;
        int st;
        y = read_all(g_tmp, 0, &n, &info, &st);
        if (!y || n != 1000 || st || info.rate != 8000 || info.frames != 1000) {
            FAIL("read back %s: %zu frames", fl ? "float" : "16-bit", n);
        } else {
            for (int i = 0; i < 1000; i++) {
                float v = x[i] != x[i] ? 0.0f : x[i];
                float want = fl ? x[i]
                           : (float)(v >= 32767.0f / 32768.0f ? 32767 : v <= -1.0f ? -32768
                                     : (long)(v * 32768.0f + (v >= 0 ? 0.5f : -0.5f))) / 32768.0f;
                if (fl ? memcmp(&y[i], &want, sizeof want) : y[i] != want) {
                    FAIL("read back %s: sample %d is %a, want %a", fl ? "float" : "16-bit", i, y[i], want);
                    break;
                }
            }
        }
        free(y);
    }

    /* ---- malformed: every prefix of a valid file, then each field wrong */
    {
        unsigned char d[64] = { 0 };
        buf b = make(&s16, d, sizeof d);
        size_t data_start = b.n - sizeof d;
        for (size_t len = 0; len < data_start; len++) {
            buf c = { b.p, len, len };
            char what[64];
            snprintf(what, sizeof what, "the first %zu bytes", len);
            refuse(what, &c);
        }
        free(b.p);
    }
    {
        unsigned char d[64] = { 0 };
#define BAD(what, field, value) do { spec c = s16; c.field = value; buf b = make(&c, d, sizeof d); refuse(what, &b); free(b.p); } while (0)
        BAD("format 2 (ADPCM)", format, 2);
        BAD("0 channels", channels, 0);
        BAD("300 channels", channels, 300);
        BAD("rate 0", rate, 0);
        BAD("24 bits in 2 bytes", bits, 24);
        BAD("0 bits", bits, 0);
        BAD("float in 2 bytes", format, 3);
        BAD("A-law in 2 bytes", format, 6);
        spec c = s16;
        c.bytes = 0;
        buf b = make(&c, d, sizeof d);
        refuse("a block of 0 bytes", &b);
        free(b.p);
        b = make(&s16, d, sizeof d);
        unsigned char *p = b.p;
        size_t fmt = 12;                                  /* "fmt " follows RIFF....WAVE */
        memcpy(p + 4 + 4, "WAVX", 4);
        refuse("not WAVE", &b);
        memcpy(p + 8, "WAVE", 4);
        memcpy(p, "RIFX", 4);
        refuse("big-endian RIFX", &b);
        memcpy(p, "RIFF", 4);
        p[fmt + 4] = 12;                                  /* fmt chunk of 12 bytes */
        refuse("a 12-byte fmt chunk", &b);
        p[fmt + 4] = 16;
        p[fmt + 8 + 12] = 3;                              /* block align 3 for one 2-byte channel */
        refuse("block align 3", &b);
        p[fmt + 8 + 12] = 2;
        memcpy(p + fmt, "data", 4);                       /* data where fmt was */
        refuse("data before fmt", &b);
        memcpy(p + fmt, "fmt ", 4);
        free(b.p);
        spec e = s16;
        e.extensible = 1;
        b = make(&e, d, sizeof d);
        b.p[12 + 8 + 26] ^= 1;                            /* the subformat GUID */
        refuse("an unknown extensible GUID", &b);
        free(b.p);
        b = make(&e, d, sizeof d);
        b.p[12 + 4] = 24;                                 /* fmt chunk too short for its extension */
        refuse("a short extensible fmt", &b);
        free(b.p);
        spec r = s16;
        r.rf64 = 1;
        b = make(&r, d, sizeof d);
        memcpy(b.p + 12, "JUNK", 4);
        refuse("RF64 without ds64", &b);
        free(b.p);
        b = make(&s16, d, sizeof d);
        buf two = { 0 };
        put(&two, b.p, 36);                               /* RIFF header and fmt */
        put(&two, b.p + 12, 24);                          /* fmt again */
        put(&two, b.p + 36, b.n - 36);
        refuse("two fmt chunks", &two);
        free(two.p);
        /* a chunk whose size runs past the end of the file */
        buf big = { 0 };
        put(&big, b.p, 36);
        tag(&big, "LIST");
        u32(&big, 1000000);
        put(&big, "abc", 3);
        refuse("a chunk past the end", &big);
        free(big.p);
        big = (buf){ 0 };
        put(&big, b.p, 36);
        tag(&big, "LIST");
        u32(&big, UINT32_MAX);                           /* plus its pad must not wrap */
        put(&big, "abc", 3);
        refuse("a UINT32_MAX chunk past the end", &big);
        free(big.p);
        free(b.p);
    }

    remove(g_tmp);
    if (g_bad) {
        printf("wavcheck: %d of %d cases wrong\n", g_bad, g_cases);
        return 1;
    }
    printf("wavcheck: %d cases, every one as it should be\n", g_cases);
    return 0;
}
