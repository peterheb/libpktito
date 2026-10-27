#ifndef PK_FUZZ_H
#define PK_FUZZ_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const uint8_t *p;
    size_t n, at;
} fz_input;

#define FZ_ASSERT(x) do { if (!(x)) abort(); } while (0)

static inline uint8_t fz_u8(fz_input *f) {
    return f->at < f->n ? f->p[f->at++] : 0;
}

static inline uint32_t fz_u32(fz_input *f) {
    uint32_t v = 0;
    for (unsigned i = 0; i < 4; i++)
        v |= (uint32_t)fz_u8(f) << (8 * i);
    return v;
}

static inline uint64_t fz_u64(fz_input *f) {
    uint64_t lo = fz_u32(f), hi = fz_u32(f);
    return lo | hi << 32;
}

static inline float fz_float(fz_input *f) {
    uint32_t u = fz_u32(f);
    float v;
    memcpy(&v, &u, sizeof v);
    return v;
}

static inline uint64_t fz_hash(const uint8_t *p, size_t n) {
    uint64_t h = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < n; i++)
        h = (h ^ p[i]) * UINT64_C(1099511628211);
    return h;
}

/* Strict UTF-8, used only as an output oracle. */
static inline int fz_utf8(const char *text, size_t n) {
    const uint8_t *s = (const uint8_t *)text;
    for (size_t i = 0; i < n;) {
        unsigned c = s[i], k = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        if (!k || k > n - i)
            return 0;
        uint32_t cp = k == 1 ? c : c & (0x7f >> k);
        for (unsigned j = 1; j < k; j++) {
            if ((s[i + j] & 0xc0) != 0x80)
                return 0;
            cp = cp << 6 | (s[i + j] & 0x3f);
        }
        if ((k == 2 && cp < 0x80) || (k == 3 && cp < 0x800) || (k == 4 && cp < 0x10000)
            || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff))
            return 0;
        i += k;
    }
    return 1;
}

static inline unsigned char *fz_read_file(const char *path, size_t *size) {
    FILE *fp = fopen(path, "rb");
    if (!fp || fseek(fp, 0, SEEK_END) || ftell(fp) < 0)
        abort();
    long end = ftell(fp);
    if (fseek(fp, 0, SEEK_SET))
        abort();
    unsigned char *p = malloc((size_t)end ? (size_t)end : 1);
    if (!p || fread(p, 1, (size_t)end, fp) != (size_t)end || fclose(fp))
        abort();
    *size = (size_t)end;
    return p;
}

static inline void fz_force_ref(void) {
#ifdef _WIN32
    _putenv_s("PKTITO_KERNELS", "ref");
#else
    setenv("PKTITO_KERNELS", "ref", 1);
#endif
}

#endif
