/* The WAV and AIFF reader and the WAV writer (core/wav.c) on files this
 * test writes itself.
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
 *   open FILE  pk_wav_open_file on a FILE read part way: pk_wav_open's
 *              samples, and the FILE left open, a refused one too
 *   malformed  every prefix of a valid file, and each field of fmt wrong in
 *              each way it can be: refused with a message, never a crash
 *   AIFF       AIFF PCM of 5 to 32 bits (12 and 20 in wider containers), and
 *              AIFF-C's NONE, twos, in24, in32, sowt, fl32, FL32, fl64, FL64,
 *              ulaw, ULAW, alaw and ALAW, one, two and five channels: every
 *              read gives the WAV of the same samples' floats, bit for bit;
 *              rates from 1 to 2^32 - 1; other chunks, odd sizes, an SSND
 *              offset; COMM's frames and SSND's size apart, a stream's SSND
 *   AIFF bad   each 80-bit rate that is not a rate, each field of COMM and
 *              SSND wrong, compressed types, chunks past the end, COMM
 *              missing, twice or after SSND, each refused for its own
 *              reason; every prefix of an AIFF and an AIFF-C; every bit of
 *              two files' headers flipped
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

/* read `path` with reads of `piece` frames (0: all at once), each frame
 * mixed to mono, or (each) with its channels as they are, interleaved */
static float *read_any(const char *path, size_t piece, int each, size_t *frames, pk_wav_info *info, int *status) {
    char err[256];
    pk_wav *w = pk_wav_open(path, info, err, sizeof err);
    if (!w) {
        *frames = 0;
        return NULL;
    }
    const size_t ch = each ? info->channels : 1;
    size_t cap = 1 << 16, n = 0;
    float *x = malloc(cap * ch * sizeof *x);
    for (;;) {
        size_t want = piece ? piece : 65536, k;
        if (n + want > cap)
            x = realloc(x, (cap = 2 * (n + want)) * ch * sizeof *x);
        n += k = each ? pk_wav_read_frames(w, x + n * ch, want) : pk_wav_read(w, x + n, want);
        if (k < want)
            break;
    }
    *status = pk_wav_status(w, err, sizeof err);
    pk_wav_close(w);
    *frames = n;
    return x;
}

static float *read_all(const char *path, size_t piece, size_t *frames, pk_wav_info *info, int *status) {
    return read_any(path, piece, 0, frames, info, status);
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

/* ---------------------------------------------------------------- AIFF --- */

static void b16(buf *b, unsigned v) { u8(b, (v >> 8) & 0xFF); u8(b, v & 0xFF); }
static void b32(buf *b, uint32_t v) { b16(b, v >> 16); b16(b, v & 0xFFFF); }

/* an 80-bit extended float from its fields: sign, biased exponent, and the
 * 64-bit significand with its explicit integer bit */
static void ext80(unsigned char *p, int neg, unsigned exp, uint64_t sig) {
    p[0] = (unsigned char)((neg ? 0x80 : 0) | (exp >> 8 & 0x7F));
    p[1] = (unsigned char)exp;
    for (int k = 0; k < 8; k++) p[2 + k] = (unsigned char)(sig >> (56 - 8 * k));
}

/* a whole number of hertz as one: the significand shifted until its top
 * bit is the integer bit, the exponent counting the shifts */
static void ext80_rate(unsigned char *p, uint32_t rate) {
    unsigned top = 31;
    while (!(rate >> top)) top--;
    ext80(p, 0, 16383 + top, (uint64_t)rate << (63 - top));
}

typedef struct {
    const char *type;              /* AIFF-C's compression type, or NULL: AIFF */
    unsigned channels, bits;       /* COMM's numChannels and sampleSize */
    uint32_t frames;               /* COMM's numSampleFrames */
    unsigned char rate[10];        /* COMM's sample rate */
    uint32_t offset, block_size;   /* SSND's */
    uint32_t ssnd_more;            /* added to SSND's size: more than there is */
    int other_chunks;              /* odd-sized chunks before COMM, between, after */
    int ssnd_first, no_comm, no_ssnd, streamed;
} aspec;

/* the AIFF file a describes around data[len] */
static buf make_aiff(const aspec *a, const unsigned char *data, size_t len) {
    buf b = { 0 }, comm = { 0 }, ssnd = { 0 };
    tag(&comm, "COMM");
    b32(&comm, a->type ? 18 + 4 + 3 : 18);             /* AIFF-C: a 3-byte name, then a pad */
    b16(&comm, a->channels);
    b32(&comm, a->frames);
    b16(&comm, a->bits);
    put(&comm, a->rate, 10);
    if (a->type) {
        put(&comm, a->type, 4);
        put(&comm, "\x02" "ab", 3);
        u8(&comm, 0);
    }
    tag(&ssnd, "SSND");
    b32(&ssnd, a->streamed ? 0 : 8 + a->offset + (uint32_t)len + a->ssnd_more);
    b32(&ssnd, a->offset);
    b32(&ssnd, a->block_size);
    for (uint32_t i = 0; i < a->offset; i++) u8(&ssnd, 0xEE);
    put(&ssnd, data, len);

    tag(&b, "FORM");
    b32(&b, 0);                                        /* fixed below */
    tag(&b, a->type ? "AIFC" : "AIFF");
    if (a->other_chunks) {
        if (a->type) {
            tag(&b, "FVER");
            b32(&b, 4);
            b32(&b, 0xA2805140u);
        }
        tag(&b, "NAME");
        b32(&b, 5);
        put(&b, "hello", 5);
        u8(&b, 0);                                     /* the pad byte */
    }
    if (a->ssnd_first && !a->no_ssnd)
        put(&b, ssnd.p, ssnd.n);
    if (!a->no_comm)
        put(&b, comm.p, comm.n);
    if (a->other_chunks) {
        tag(&b, "FLLR");
        b32(&b, 3);
        put(&b, "abc", 3);
        u8(&b, 0);
    }
    if (!a->ssnd_first && !a->no_ssnd)
        put(&b, ssnd.p, ssnd.n);
    if (a->other_chunks && !a->streamed) {
        if (len & 1) u8(&b, 0);
        tag(&b, "ANNO");
        b32(&b, 1);
        put(&b, "z", 1);
        u8(&b, 0);
    }
    uint32_t form = a->streamed ? 0 : (uint32_t)b.n - 8;
    memcpy(b.p + 4, (unsigned char[4]){ form >> 24, (form >> 16) & 0xFF, (form >> 8) & 0xFF, form & 0xFF }, 4);
    free(comm.p);
    free(ssnd.p);
    return b;
}

/* The WAV spec s's samples as an AIFF stores them: 8-bit PCM signed, where
 * a WAV's is offset by 128, and every sample big-endian (sowt: little, as
 * a WAV's) */
static void as_aiff(const spec *s, int little, unsigned char *d, size_t len) {
    const size_t w = s->bytes;
    for (size_t i = 0; i + w <= len; i += w) {
        if (s->format == 1 && w == 1)
            d[i] ^= 0x80;
        for (size_t k = 0; !little && k < w / 2; k++) {
            unsigned char t = d[i + k];
            d[i + k] = d[i + w - 1 - k];
            d[i + w - 1 - k] = t;
        }
    }
}

/* The WAV s and the AIFF a of the same audio: every read of the AIFF, mono
 * and frame by frame, in pieces, gives the WAV's floats bit for bit, and
 * the same info. a's channels, frames and rate are s's. */
static void aiff_case(spec s, aspec a, uint32_t frames) {
    const uint32_t ch = s.channels, bytes = s.bytes;
    size_t len = (size_t)frames * ch * bytes;
    unsigned char *data = malloc(len + 1);
    float v;
    for (size_t i = 0; i < (size_t)frames * ch; i++) {
        sample(&s, (uint32_t)i, data + i * bytes, &v);
        /* fewer bits than the container: the low ones zero, in both files */
        if (s.format == 1 && s.bits < 8 * bytes)
            data[i * bytes] &= (unsigned char)(0xFF << (8 * bytes - s.bits));
    }
    a.channels = ch;
    a.frames = frames;
    ext80_rate(a.rate, s.rate);
    char what[160];
    snprintf(what, sizeof what, "%s %s, %u bits, %u channels, %u Hz%s%s", a.type ? "AIFF-C" : "AIFF",
             a.type ? a.type : "PCM", a.bits, ch, s.rate, a.other_chunks ? ", other chunks" : "",
             a.offset ? ", an SSND offset" : "");

    buf wb = make(&s, data, len);
    save(&wb);
    free(wb.p);
    pk_wav_info wi, ai;
    size_t nm, nf, n;
    int st;
    float *wm = read_any(g_tmp, 0, 0, &nm, &wi, &st);
    float *wf = read_any(g_tmp, 0, 1, &nf, &wi, &st);
    if (!wm || !wf || nm != frames || nf != frames || st) {
        FAIL("%s: the WAV it is compared with did not read", what);
        goto done;
    }
    as_aiff(&s, a.type && !strcmp(a.type, "sowt"), data, len);
    buf ab = make_aiff(&a, data, len);
    save(&ab);
    free(ab.p);
    static const size_t pieces[] = { 0, 1, 7, 4096 };
    for (int each = 0; each < 2; each++)
        for (size_t pi = 0; pi < 4; pi++) {
            float *x = read_any(g_tmp, pieces[pi], each, &n, &ai, &st);
            g_cases++;
            if (!x) {
                FAIL("%s: did not open", what);
                goto done;
            }
            if (n != frames || st)
                FAIL("%s, %s reads of %zu: %zu frames of %u, status %d", what, each ? "frame" : "mono",
                     pieces[pi], n, frames, st);
            else if (memcmp(x, each ? wf : wm, (size_t)frames * (each ? ch : 1) * sizeof *x))
                FAIL("%s, %s reads of %zu: a sample is not the WAV's", what, each ? "frame" : "mono", pieces[pi]);
            if (ai.rate != wi.rate || ai.channels != wi.channels || ai.bytes != wi.bytes
                || ai.format != wi.format || ai.frames != (a.streamed ? PK_WAV_UNKNOWN : frames))
                FAIL("%s: info %u Hz %u ch %u bytes format %u frames %llu", what, (unsigned)ai.rate,
                     (unsigned)ai.channels, (unsigned)ai.bytes, (unsigned)ai.format,
                     (unsigned long long)ai.frames);
            free(x);
        }
done:
    free(wm);
    free(wf);
    free(data);
}

/* a 16-bit mono 16 kHz AIFF, for a case to change one thing in */
static aspec aiff16(void) {
    aspec a = { 0 };
    a.channels = 1;
    a.bits = 16;
    ext80_rate(a.rate, 16000);
    return a;
}

/* a's file around n 16-bit frames: how many frames read, and the status */
static size_t aiff_read(const aspec *a, size_t n, int *st, pk_wav_info *info) {
    unsigned char *d = calloc(n * 2 + 1, 1);
    for (size_t i = 0; i < n * 2; i++) d[i] = (unsigned char)i;
    buf b = make_aiff(a, d, n * 2);
    save(&b);
    free(b.p);
    free(d);
    size_t got;
    float *x = read_all(g_tmp, 0, &got, info, st);
    if (!x)
        return SIZE_MAX;
    free(x);
    return got;
}

/* refused, and for the reason `why` names: each check is reached */
static void refuse_why(const char *what, const buf *b, const char *why) {
    char err[256] = "";
    save(b);
    pk_wav_info info;
    pk_wav *w = pk_wav_open(g_tmp, &info, err, sizeof err);
    g_cases++;
    if (w) {
        FAIL("%s: opened", what);
        pk_wav_close(w);
    } else if (!strstr(err, why)) {
        FAIL("%s: refused as \"%s\", not for \"%s\"", what, err, why);
    }
}

static void aiff_refuse(const char *what, const aspec *a, const char *why) {
    unsigned char d[64] = { 0 };
    aspec c = *a;
    c.frames = 32;
    buf b = make_aiff(&c, d, sizeof d);
    refuse_why(what, &b, why);
    free(b.p);
}

static void aiff_checks(void) {
    /* ---- the 80-bit writer, on a rate as ffmpeg and afconvert write it */
    {
        static const unsigned char want[10] = { 0x40, 0x0C, 0xFA, 0, 0, 0, 0, 0, 0, 0 };
        unsigned char r[10];
        ext80_rate(r, 16000);
        g_cases++;
        if (memcmp(r, want, 10))
            FAIL("16000 as an 80-bit float");
    }

    /* ---- every supported type, against the WAV of the same samples */
    static const struct { const char *type; unsigned format, bytes, bits, size_field; } types[] = {
        { NULL, 1, 1, 8, 8 },        { NULL, 1, 2, 16, 16 },      { NULL, 1, 3, 24, 24 },
        { NULL, 1, 4, 32, 32 },      { NULL, 1, 2, 12, 12 },      { NULL, 1, 3, 20, 20 },
        { NULL, 1, 1, 5, 5 },
        { "NONE", 1, 1, 8, 8 },      { "NONE", 1, 2, 16, 16 },    { "NONE", 1, 3, 24, 24 },
        { "NONE", 1, 4, 32, 32 },    { "NONE", 1, 2, 12, 12 },
        { "twos", 1, 1, 8, 8 },      { "twos", 1, 2, 16, 16 },
        { "in24", 1, 3, 24, 24 },    { "in32", 1, 4, 32, 32 },    { "in24", 1, 3, 18, 18 },
        { "sowt", 1, 1, 8, 8 },      { "sowt", 1, 2, 16, 16 },    { "sowt", 1, 3, 24, 24 },
        { "sowt", 1, 4, 32, 32 },
        { "fl32", 3, 4, 32, 32 },    { "FL32", 3, 4, 32, 32 },
        { "fl64", 3, 8, 64, 64 },    { "FL64", 3, 8, 64, 64 },
        { "ulaw", 7, 1, 8, 8 },      { "ULAW", 7, 1, 8, 8 },      { "ulaw", 7, 1, 8, 16 },
        { "alaw", 6, 1, 8, 8 },      { "ALAW", 6, 1, 8, 8 },      { "alaw", 6, 1, 8, 16 },
    };
    for (size_t i = 0; i < sizeof types / sizeof types[0]; i++)
        for (unsigned ch = 1; ch <= 5; ch += (ch == 2 ? 3 : 1)) {
            /* a WAV can say fewer bits than its container only as EXTENSIBLE */
            int ext = types[i].format == 1 && types[i].bits < 8 * types[i].bytes;
            spec s = { types[i].format, ch, 16000, types[i].bytes, types[i].bits, ext, 0, 0, 0, 0, 0 };
            aspec a = { 0 };
            a.type = types[i].type;
            a.bits = types[i].size_field;
            aiff_case(s, a, 3001);
        }

    /* ---- rates: every whole number of hertz a WAV can say reads as itself */
    static const uint32_t rates[] = { 1, 2, 3, 8000, 11025, 16000, 22050, 44100, 48000, 96000,
                                      192000, 2147483648u, 4294967295u };
    for (size_t i = 0; i < sizeof rates / sizeof rates[0]; i++) {
        spec s = { 1, 2, rates[i], 2, 16, 0, 0, 0, 0, 0, 0 };
        aspec a = { 0 };
        a.bits = 16;
        aiff_case(s, a, 50);
    }

    /* ---- layout: other chunks with odd sizes around COMM and SSND, an odd
     * AIFF-C name, an SSND offset and blockSize, an odd SSND */
    {
        spec s = { 1, 2, 44100, 2, 16, 0, 0, 0, 0, 0, 0 };
        aspec a = { 0 };
        a.bits = 16;
        a.other_chunks = 1;
        aiff_case(s, a, 777);
        a.type = "NONE";
        aiff_case(s, a, 777);
        a.offset = 5;
        a.block_size = 4096;
        aiff_case(s, a, 777);
        spec m = { 1, 1, 8000, 1, 8, 0, 0, 0, 0, 0, 0 };   /* 777 bytes: SSND's pad */
        a.type = NULL;
        a.bits = 8;
        aiff_case(m, a, 777);
        spec wide = { 3, 256, 16000, 8, 64, 0, 0, 0, 0, 0, 0 };   /* the widest legal frame */
        a.type = "fl64";
        a.bits = 64;
        aiff_case(wide, a, 37);
    }

    /* ---- the two counts, COMM's frames and SSND's size, apart */
    {
        aspec a = aiff16();
        pk_wav_info info;
        int st;
        size_t n;
        a.frames = 100;                                  /* SSND holds 200 */
        n = aiff_read(&a, 200, &st, &info);
        g_cases++;
        if (n != 100 || st || info.frames != 100)
            FAIL("COMM's 100 frames in SSND's 200: %zu read, status %d", n, st);
        a.frames = 300;                                  /* more than SSND holds */
        n = aiff_read(&a, 200, &st, &info);
        g_cases++;
        if (n != 200 || !st || info.frames != 200)
            FAIL("COMM's 300 frames in SSND's 200: %zu read, status %d (want 200 and an error)", n, st);
        a.frames = 300;
        a.ssnd_more = 200;                               /* SSND says 300, the file stops at 200 */
        n = aiff_read(&a, 200, &st, &info);
        g_cases++;
        if (n != 200 || !st || info.frames != 300)
            FAIL("an SSND cut short: %zu read, status %d (want 200 and an error)", n, st);
        a.ssnd_more = UINT32_MAX - 8 - 400;              /* SSND's size UINT32_MAX */
        n = aiff_read(&a, 200, &st, &info);
        g_cases++;
        if (n != 200 || !st)
            FAIL("a UINT32_MAX SSND: %zu read, status %d (want 200 and an error)", n, st);
        a.ssnd_more = 1;                                 /* a partial frame after the last */
        a.frames = 200;
        n = aiff_read(&a, 200, &st, &info);
        g_cases++;
        if (n != 200 || st)
            FAIL("an SSND with a partial frame: %zu read, status %d", n, st);
        a.ssnd_more = 0;
        a.frames = UINT32_MAX;                           /* COMM's largest count */
        n = aiff_read(&a, 200, &st, &info);
        g_cases++;
        if (n != 200 || !st)
            FAIL("COMM's UINT32_MAX frames: %zu read, status %d", n, st);
        a.frames = 0;
        n = aiff_read(&a, 200, &st, &info);
        g_cases++;
        if (n != 0 || st || info.frames != 0)
            FAIL("COMM's 0 frames: %zu read, status %d", n, st);
        a.streamed = 1;                                  /* ffmpeg's to a pipe: sizes 0 */
        a.offset = 3;
        n = aiff_read(&a, 200, &st, &info);
        g_cases++;
        if (n != 200 || st || info.frames != PK_WAV_UNKNOWN)
            FAIL("a streamed AIFF: %zu read, status %d", n, st);
        /* and a SIZE_MAX read of it, as of a streamed WAV */
        char err[256];
        pk_wav *w = pk_wav_open(g_tmp, &info, err, sizeof err);
        float x[300];
        g_cases++;
        if (!w || pk_wav_read(w, x, SIZE_MAX) != 200)
            FAIL("a SIZE_MAX read of a streamed AIFF");
        pk_wav_close(w);
    }

    /* ---- malformed: the rates */
    {
        static const struct { const char *what; int neg; unsigned exp; uint64_t sig; const char *why; } bad[] = {
            { "rate 0", 0, 0, 0, "of 0" },
            { "rate -0", 1, 0, 0, "negative" },
            { "rate -16000", 1, 16383 + 13, (uint64_t)16000 << 50, "negative" },
            { "rate -1", 1, 16383, UINT64_C(1) << 63, "negative" },
            { "rate +infinity", 0, 0x7FFF, UINT64_C(1) << 63, "infinite" },
            { "rate -infinity", 1, 0x7FFF, UINT64_C(1) << 63, "negative" },
            { "rate NaN", 0, 0x7FFF, UINT64_C(3) << 62, "not a number" },
            { "rate signalling NaN", 0, 0x7FFF, (UINT64_C(1) << 63) | 1, "not a number" },
            { "rate pseudo-infinity", 0, 0x7FFF, 0, "infinite" },
            { "rate denormal", 0, 0, 1, "normalized" },
            { "rate largest denormal", 0, 0, UINT64_MAX >> 1, "normalized" },
            { "rate pseudo-denormal", 0, 0, UINT64_C(1) << 63, "below 1 Hz" },
            { "rate unnormal 16000", 0, 16383 + 14, (uint64_t)16000 << 49, "normalized" },
            { "rate 0.5", 0, 16382, UINT64_C(1) << 63, "below 1 Hz" },
            { "rate smallest normal", 0, 1, UINT64_C(1) << 63, "below 1 Hz" },
            { "rate 16000.5", 0, 16383 + 13, (uint64_t)32001 << 49, "whole number" },
            { "rate 1.5", 0, 16383, UINT64_C(3) << 62, "whole number" },
            { "rate 4294967295.5", 0, 16383 + 31, UINT64_MAX << 31, "whole number" },
            { "rate 2^32", 0, 16383 + 32, UINT64_C(1) << 63, "2^32" },
            { "rate 1e10", 0, 16383 + 33, (uint64_t)10000000000ull << 30, "2^32" },
            { "rate largest finite", 0, 0x7FFE, UINT64_MAX, "2^32" },
        };
        for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
            aspec a = aiff16();
            ext80(a.rate, bad[i].neg, bad[i].exp, bad[i].sig);
            aiff_refuse(bad[i].what, &a, bad[i].why);
        }
    }

    /* ---- malformed: COMM, SSND and the container */
    {
        aspec a;
#define ABAD(what, why, ...) do { a = aiff16(); __VA_ARGS__; aiff_refuse(what, &a, why); } while (0)
        ABAD("AIFF 0 channels", "channels", a.channels = 0);
        ABAD("AIFF 257 channels", "channels", a.channels = 257);
        ABAD("AIFF -1 channels", "channels", a.channels = 0xFFFF);
        ABAD("AIFF 0 bits", "bit", a.bits = 0);
        ABAD("AIFF 33 bits", "bit", a.bits = 33);
        ABAD("AIFF -16 bits", "bit", a.bits = 0x10000 - 16);
        ABAD("AIFF-C 33 bits", "bit", a.type = "NONE"; a.bits = 33);
        ABAD("AIFF-C in24 of 32 bits", "bit", a.type = "in24"; a.bits = 32);
        ABAD("AIFF-C twos of 0 bits", "bit", a.type = "twos"; a.bits = 0);
        ABAD("AIFF-C ima4", "'ima4'", a.type = "ima4");
        ABAD("AIFF-C MAC3", "'MAC3'", a.type = "MAC3");
        ABAD("AIFF-C MAC6", "'MAC6'", a.type = "MAC6");
        ABAD("AIFF-C GSM", "'GSM '", a.type = "GSM ");
        ABAD("AIFF-C Qclp", "'Qclp'", a.type = "Qclp");
        ABAD("AIFF-C raw (offset 8-bit)", "'raw '", a.type = "raw ");
        ABAD("AIFF-C lowercase none", "'none'", a.type = "none");
        ABAD("AIFF-C unprintable type", "compression '", a.type = "\x01\xff\x7f\x80");
        ABAD("no COMM", "before any COMM", a.no_comm = 1);
        ABAD("no COMM, no SSND", "no COMM", a.no_comm = 1; a.no_ssnd = 1);
        ABAD("no SSND", "no SSND", a.no_ssnd = 1);
        ABAD("no SSND, other chunks", "no SSND", a.no_ssnd = 1; a.other_chunks = 1);
        ABAD("COMM after SSND", "before any COMM", a.ssnd_first = 1);
        unsigned char d[64] = { 0 };
        a = aiff16();
        a.frames = 32;
        buf b = make_aiff(&a, d, sizeof d);
        size_t ssnd = b.n - sizeof d - 16;              /* "SSND" */
        buf more = { 0 };                                /* and a chunk after it */
        put(&more, b.p, b.n);
        put(&more, "ANNO\0\0\0\x08" "abcdefgh", 16);
        more.p[ssnd + 8 + 3] = 64 + 1;                   /* offset 65 in 8 + 64 bytes */
        refuse_why("an SSND offset of 65 in a 72-byte chunk", &more, "offset");
        more.p[ssnd + 8] = 0xFF;                         /* offset 0xFF000041 */
        refuse_why("an SSND offset of 0xFF000041", &more, "offset");
        free(more.p);
        memset(b.p + ssnd + 4, 0xFF, 4);                 /* size UINT32_MAX, the offset past the file */
        b.p[ssnd + 8 + 3] = 65;
        refuse_why("an SSND offset past the end of the file", &b, "offset");
        memset(b.p + ssnd + 4, 0, 4);
        b.p[ssnd + 7] = 4;                               /* size 4: no room for offset, blockSize */
        refuse_why("a 4-byte SSND chunk", &b, "4-byte SSND");
        b.p[ssnd + 7] = 8 + 64;
        b.p[ssnd + 11] = 0;
        memcpy(b.p + 8, "AIFX", 4);
        refuse("FORM but not AIFF", &b);
        memcpy(b.p + 8, "8SVX", 4);
        refuse("an IFF 8SVX", &b);
        memcpy(b.p + 8, "AIFF", 4);
        b.p[12 + 7] = 17;                                /* COMM of 17 bytes */
        refuse_why("a 17-byte AIFF COMM", &b, "17-byte COMM");
        b.p[12 + 7] = 18;
        free(b.p);
        a.type = "NONE";
        b = make_aiff(&a, d, sizeof d);
        b.p[12 + 7] = 21;                                /* AIFF-C's COMM needs its type */
        refuse_why("a 21-byte AIFF-C COMM", &b, "21-byte COMM");
        free(b.p);

        /* two COMMs */
        a = aiff16();
        b = make_aiff(&a, d, sizeof d);
        buf two = { 0 };
        put(&two, b.p, 12 + 26);
        put(&two, b.p + 12, 26);
        put(&two, b.p + 12 + 26, b.n - 12 - 26);
        refuse_why("two COMM chunks", &two, "two COMM");
        free(two.p);
        /* chunks whose sizes run past the end of the file */
        static const uint32_t sizes[] = { 1000000, UINT32_MAX, UINT32_MAX - 1 };
        static const char *ids[] = { "NAME", "COMM" };
        for (size_t k = 0; k < 2; k++)
            for (size_t i = 0; i < 3; i++) {
                buf big = { 0 };
                put(&big, b.p, 12);
                if (k == 1) {
                    tag(&big, "COMM");
                    b32(&big, sizes[i]);
                    put(&big, b.p + 20, 18);
                } else {
                    put(&big, b.p + 12, 26);
                    tag(&big, ids[k]);
                    b32(&big, sizes[i]);
                    put(&big, "abc", 3);
                }
                char what[96];
                snprintf(what, sizeof what, "a %u-byte %s past the end", (unsigned)sizes[i], ids[k]);
                refuse(what, &big);
                free(big.p);
            }
        free(b.p);
    }

    /* ---- every prefix of a valid AIFF and AIFF-C to their first sample */
    for (int c = 0; c < 2; c++) {
        unsigned char d[64] = { 0 };
        aspec a = aiff16();
        a.frames = 32;
        a.type = c ? "fl32" : NULL;
        a.other_chunks = 1;
        a.offset = 3;
        buf b = make_aiff(&a, d, sizeof d);
        size_t data_start = b.n - sizeof d - 10;         /* the trailing ANNO */
        for (size_t len = 0; len < data_start; len++) {
            buf p = { b.p, len, len };
            char what[64];
            snprintf(what, sizeof what, "the first %zu bytes of an %s", len, c ? "AIFF-C" : "AIFF");
            refuse(what, &p);
        }
        free(b.p);
    }

    /* ---- every bit of the headers flipped: refused or read, never past the
     * file, and what opens says a sample the reader can decode */
    for (int c = 0; c < 2; c++) {
        unsigned char d[64];
        for (int i = 0; i < 64; i++) d[i] = (unsigned char)(i * 37);
        aspec a = aiff16();
        a.channels = 2;
        a.frames = 16;
        a.type = c ? "sowt" : NULL;
        a.other_chunks = 1;
        a.offset = 2;
        buf b = make_aiff(&a, d, sizeof d);
        size_t data_start = b.n - sizeof d - 10;
        for (size_t i = 0; i < data_start; i++)
            for (int bit = 0; bit < 8; bit++) {
                b.p[i] ^= (unsigned char)(1 << bit);
                save(&b);
                pk_wav_info info;
                size_t n;
                int st;
                float *x = read_any(g_tmp, 7, 1, &n, &info, &st);
                g_cases++;
                if (x && (info.channels < 1 || info.channels > 256 || info.bytes < 1 || info.bytes > 8
                          || info.rate == 0 || (size_t)n * info.channels * info.bytes > b.n))
                    FAIL("byte %zu bit %d flipped: %u channels of %u bytes at %u Hz, %zu frames",
                         i, bit, (unsigned)info.channels, (unsigned)info.bytes, (unsigned)info.rate, n);
                free(x);
                b.p[i] ^= (unsigned char)(1 << bit);
            }
        free(b.p);
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

    /* ---- open FILE: the WAV after 4 bytes the caller has read, then a refusal */
    {
        unsigned char d[600];
        for (int i = 0; i < 600; i++) d[i] = (unsigned char)(i * 7);
        spec st = s16;
        st.channels = 2;
        buf b = make(&st, d, sizeof d);
        save(&b);
        pk_wav_info want;
        size_t n;
        int status;
        float *x = read_any(g_tmp, 0, 1, &n, &want, &status);
        buf pre = { 0 };
        put(&pre, "head", 4);
        put(&pre, b.p, b.n);
        save(&pre);
        char head[4], err[256] = "";
        pk_wav_info info;
        float y[400];
        FILE *f = fopen(g_tmp, "rb");
        pk_wav *w = f && fread(head, 1, 4, f) == 4 ? pk_wav_open_file(f, "head+wav", &info, err, sizeof err) : NULL;
        size_t k = w ? pk_wav_read_frames(w, y, 200) : 0;
        g_cases++;
        if (!w || !x || k != n || memcmp(y, x, n * 2 * sizeof *y) || info.rate != want.rate
            || info.channels != want.channels || info.frames != want.frames || pk_wav_status(w, err, sizeof err))
            FAIL("pk_wav_open_file: %s", w ? "not pk_wav_open's samples" : err);
        pk_wav_close(w);
        /* still the caller's: open, and read again from the start */
        g_cases++;
        if (!f || fseek(f, 0, SEEK_SET) || fread(head, 1, 4, f) != 4 || memcmp(head, "head", 4))
            FAIL("pk_wav_open_file: the FILE not left open");
        /* from the start, "head" is no header: refused, and the FILE left open */
        w = f && !fseek(f, 0, SEEK_SET) ? pk_wav_open_file(f, "head+wav", &info, err, sizeof err) : NULL;
        g_cases++;
        if (w || strncmp(err, "head+wav: ", 10) || !f || fseek(f, 0, SEEK_SET) || fgetc(f) != 'h')
            FAIL("pk_wav_open_file on a file that is no WAV: %s", w ? "opened" : err);
        pk_wav_close(w);
        if (f)
            fclose(f);
        w = pk_wav_open_file(NULL, "nothing", &info, err, sizeof err);
        g_cases++;
        if (w || !err[0])
            FAIL("pk_wav_open_file on no FILE: %s", w ? "opened" : "no reason");
        free(x);
        free(pre.p);
        free(b.p);
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

    aiff_checks();

    remove(g_tmp);
    if (g_bad) {
        printf("wavcheck: %d of %d cases wrong\n", g_bad, g_cases);
        return 1;
    }
    printf("wavcheck: %d cases, every one as it should be\n", g_cases);
    return 0;
}
