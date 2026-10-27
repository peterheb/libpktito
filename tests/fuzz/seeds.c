/* Small valid starting points for the byte-level fuzzers.  They are made at
 * build time so the repository need not carry opaque binary corpus files. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void put16le(unsigned char *p, uint32_t v) { p[0] = (unsigned char)v; p[1] = (unsigned char)(v >> 8); }
static void put32le(unsigned char *p, uint32_t v) { put16le(p, v); put16le(p + 2, v >> 16); }
static void put16be(unsigned char *p, uint32_t v) { p[0] = (unsigned char)(v >> 8); p[1] = (unsigned char)v; }
static void put32be(unsigned char *p, uint32_t v) { put16be(p, v >> 16); put16be(p + 2, v); }

static void save(const char *root, const char *name, const void *data, size_t size) {
    char path[1024];
    if (snprintf(path, sizeof path, "%s/%s", root, name) >= (int)sizeof path)
        abort();
    FILE *fp = fopen(path, "wb");
    if (!fp || fwrite(data, 1, size, fp) != size || fclose(fp)) {
        fprintf(stderr, "cannot write fuzz seed %s\n", path);
        exit(1);
    }
}

static void wav_seed(const char *root) {
    unsigned char b[52] = { 0 };
    memcpy(b, "RIFF", 4); put32le(b + 4, sizeof b - 8); memcpy(b + 8, "WAVEfmt ", 8);
    put32le(b + 16, 16); put16le(b + 20, 1); put16le(b + 22, 1); put32le(b + 24, 16000);
    put32le(b + 28, 32000); put16le(b + 32, 2); put16le(b + 34, 16);
    memcpy(b + 36, "data", 4); put32le(b + 40, 8);
    put16le(b + 44, 0x8000); put16le(b + 46, 0); put16le(b + 48, 0x7fff); put16le(b + 50, 1);
    save(root, "wav_bytes/pcm16.wav", b, sizeof b);
}

static void aiff_seed(const char *root) {
    unsigned char b[62] = { 0 };
    memcpy(b, "FORM", 4); put32be(b + 4, sizeof b - 8); memcpy(b + 8, "AIFFCOMM", 8);
    put32be(b + 16, 18); put16be(b + 20, 1); put32be(b + 22, 4); put16be(b + 26, 16);
    /* 16000 = 1.953125 * 2^13 as an IEEE 754 80-bit extended value. */
    b[28] = 0x40; b[29] = 0x0c; b[30] = 0xfa;
    memcpy(b + 38, "SSND", 4); put32be(b + 42, 16);
    put16be(b + 54, 0x8000); put16be(b + 56, 0); put16be(b + 58, 0x7fff); put16be(b + 60, 1);
    save(root, "wav_bytes/pcm16.aiff", b, sizeof b);
}

int main(int argc, char **argv) {
    if (argc != 2)
        return 2;
    wav_seed(argv[1]);
    aiff_seed(argv[1]);

    unsigned char rs[8 + 8 * 4] = { 0 };
    put32le(rs, 8);                    /* common[8] = 44100 */
    put32le(rs + 4, 3);               /* common[3] = 16000 */
    save(argv[1], "resample/44100-to-16000", rs, sizeof rs);
    static const unsigned char segment[] = { 0, 1, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    static const unsigned char chunks[] = { 1, 0x00, 0x40, 0x00, 0x00, 0x80, 0x3e, 0x00, 0x00, 4 };
    static const unsigned char captions[] = { 2, 4, 5, 'h', 'e', 'l', 'l', 'o' };
    save(argv[1], "state/segments", segment, sizeof segment);
    save(argv[1], "state/chunks", chunks, sizeof chunks);
    save(argv[1], "state/captions", captions, sizeof captions);

    static const uint32_t edge_bits[] = {
        0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x00000001,
        0x007fffff, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
        0x7fc00000, 0x7f800001,
    };
    unsigned char edge[1 + sizeof edge_bits] = { 0 };
    for (size_t i = 0; i < sizeof edge_bits / sizeof edge_bits[0]; i++)
        put32le(edge + 1 + 4 * i, edge_bits[i]);
    save(argv[1], "audio_pipeline/edge-floats", edge, sizeof edge);
    unsigned char *silence = calloc(1 + 512 * sizeof(float), 1);
    if (!silence)
        abort();
    silence[0] = 1;
    save(argv[1], "audio_pipeline/silence-frame", silence, 1 + 512 * sizeof(float));
    free(silence);
    static const unsigned char empty_repeat[] = {
        0xd7, 0x3a, 0xd7, 0xd7, 0xd7, 0xd7, 0xd7, 0xd7,
        0xd7, 0xd9, 0xd7, 0xd7, 0xde, 0xd7, 0x7f,
    };
    save(argv[1], "audio_pipeline/empty-repeat", empty_repeat, sizeof empty_repeat);
    unsigned char *semantic = calloc(1 + 1600 * sizeof(float), 1);
    if (!semantic)
        abort();
    save(argv[1], "model_semantic/base-and-silence", semantic, 1 + 1600 * sizeof(float));
    free(semantic);
    return 0;
}
