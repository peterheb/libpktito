/* Loading a .pkf: map it, prove it well formed, then bind every tensor the
 * model needs by name and by the shape its params give it.
 *
 * The file is input like any other - a truncated download, some other
 * program's file, a flipped bit - and docs/MEMORY_SAFETY.md has the rules
 * this applies. The header and each descriptor are copied out before
 * anything in them is believed, every extent is checked without forming its
 * end, tensors may not overlap, and a tensor's shape is held to the one the
 * params imply rather than taken from its descriptor. Only then is a
 * pointer into the file made, and before the model is handed back every
 * float it will multiply by is checked to be finite. The int8 weights need
 * no such check: every byte is a valid one.
 */
#include "model.h"
#include "os.h"
#include "pkf.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* A model file past this is not one; refusing it bounds every size below
 * well inside a 64-bit size_t. A 1.1B Parakeet in int8 is about 1.2 GB. */
#define PK_FILE_MAX ((uint64_t)16 << 30)
#define PK_MAX_PARAMS 256
#define PK_MAX_TENSORS 4096
#define PK_MAX_PIECES (1u << 20)

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "a .pkf is little-endian and pkito reads it in place"
#endif

static int fail(char *err, size_t errlen, const char *fmt, ...) {
    if (err && errlen) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, errlen, fmt, ap);
        va_end(ap);
    }
    return -1;
}

static uint64_t align64(uint64_t x) { return (x + PK_ALIGN - 1) & ~(uint64_t)(PK_ALIGN - 1); }

/* 1 if s[0..n) is well-formed UTF-8: no overlong form, no surrogate, nothing
 * past U+10FFFF */
static int utf8_ok(const unsigned char *s, size_t n) {
    for (size_t i = 0; i < n;) {
        unsigned c = s[i];
        size_t len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        if (!len || len > n - i)
            return 0;
        uint32_t cp = len == 1 ? c : c & (0x7F >> len);
        for (size_t j = 1; j < len; j++) {
            if ((s[i + j] & 0xC0) != 0x80)
                return 0;
            cp = cp << 6 | (s[i + j] & 0x3F);
        }
        if ((len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) || (len == 4 && cp < 0x10000)
            || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))
            return 0;
        i += len;
    }
    return 1;
}

/* A fixed-size char field holds a NUL-terminated string, NUL-padded after */
static int name_ok(const char *s, size_t n) {
    const char *z = memchr(s, 0, n);
    if (!z || z == s)
        return 0;
    for (; z < s + n; z++)
        if (*z)
            return 0;
    return 1;
}

static uint32_t rd32(const unsigned char *p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

/* ------------------------------------------------------------ validate --- */

static int validate_vocab(const unsigned char *v, uint64_t size, char *err, size_t errlen) {
    if (size < 4)
        return fail(err, errlen, "the vocabulary is shorter than its count");
    uint32_t n = rd32(v);
    if (n == 0 || n > PK_MAX_PIECES)
        return fail(err, errlen, "a vocabulary of %u pieces", (unsigned)n);
    /* count, offsets, types: 4 + 4(n+1) + n, all inside size (n is bounded) */
    uint64_t head = 4 + 4 * ((uint64_t)n + 1) + n;
    if (head > size)
        return fail(err, errlen, "the vocabulary's tables run past its end");
    if (rd32(v + 4) != 0)
        return fail(err, errlen, "the vocabulary's first offset is not 0");
    for (uint32_t i = 0; i < n; i++)
        if (rd32(v + 4 + 4 * (uint64_t)i) > rd32(v + 4 + 4 * ((uint64_t)i + 1)))
            return fail(err, errlen, "the vocabulary's offsets go backward at piece %u", (unsigned)i);
    uint64_t text = rd32(v + 4 + 4 * (uint64_t)n);
    if (text > size - head || size - head - text < 4)
        return fail(err, errlen, "the vocabulary's text runs past its end");
    const unsigned char *types = v + 4 + 4 * ((uint64_t)n + 1), *t0 = types + n;
    for (uint32_t i = 0; i < n; i++) {
        if (types[i] < PK_PIECE_NORMAL || types[i] > PK_PIECE_UNUSED)
            return fail(err, errlen, "piece %u has type %u", (unsigned)i, types[i]);
        uint32_t a = rd32(v + 4 + 4 * (uint64_t)i), b = rd32(v + 4 + 4 * ((uint64_t)i + 1));
        if (!utf8_ok(t0 + a, b - a))
            return fail(err, errlen, "piece %u is not UTF-8", (unsigned)i);
    }
    uint64_t np = rd32(t0 + text);
    if (np != size - head - text - 4)
        return fail(err, errlen, "the vocabulary's punctuation is %llu bytes where %llu are left",
                    (unsigned long long)np, (unsigned long long)(size - head - text - 4));
    if (!utf8_ok(t0 + text + 4, np))
        return fail(err, errlen, "the punctuation is not UTF-8");
    return 0;
}

int pk_pkf_validate(const unsigned char *p, size_t n, char *err, size_t errlen) {
    pk_header h;
    if (n < sizeof h)
        return fail(err, errlen, "%zu bytes is shorter than a header", n);
    if ((uint64_t)n > PK_FILE_MAX)
        return fail(err, errlen, "%zu bytes is larger than any model file", n);
    memcpy(&h, p, sizeof h);
    if (h.magic != PK_MAGIC)
        return fail(err, errlen, "not a pkito model file (magic %08x)", (unsigned)h.magic);
    if (h.version != PK_VERSION)
        return fail(err, errlen, "model file version %u; this build reads %u",
                    (unsigned)h.version, PK_VERSION);
    if (h.file_bytes != n)
        return fail(err, errlen, "the header says %llu bytes and there are %zu",
                    (unsigned long long)h.file_bytes, n);
    if (!name_ok(h.arch, sizeof h.arch) || strcmp(h.arch, "parakeet-tdt"))
        return fail(err, errlen, "the architecture is not parakeet-tdt");
    if (!name_ok(h.name, sizeof h.name))
        return fail(err, errlen, "the model's name is empty or not terminated");
    for (size_t i = 0; i < sizeof h.reserved; i++)
        if (h.reserved[i])
            return fail(err, errlen, "reserved header byte %zu is not zero", i);
    if (h.n_params == 0 || h.n_params > PK_MAX_PARAMS)
        return fail(err, errlen, "%u params", (unsigned)h.n_params);
    if (h.n_tensors == 0 || h.n_tensors > PK_MAX_TENSORS)
        return fail(err, errlen, "%u tensors", (unsigned)h.n_tensors);

    /* The tables sit end to end in a fixed order, which leaves one way to
     * lay a file out; every bound below is a sum of terms bounded by
     * PK_FILE_MAX and the table counts, so none can wrap. */
    uint64_t at = sizeof h;
    if (h.params_off != at)
        return fail(err, errlen, "the params do not follow the header");
    at += (uint64_t)h.n_params * sizeof(pk_param);
    if (h.tensors_off != at)
        return fail(err, errlen, "the tensor table does not follow the params");
    at += (uint64_t)h.n_tensors * sizeof(pk_tensor);
    if (h.vocab_off != at || h.vocab_size > n || at > n - h.vocab_size)
        return fail(err, errlen, "the vocabulary is misplaced or runs past the end of the file");
    at += h.vocab_size;
    if (h.meta_off != at || h.meta_size > n || at > n - h.meta_size)
        return fail(err, errlen, "the metadata is misplaced or runs past the end of the file");
    at += h.meta_size;
    if (h.data_off != align64(at) || h.data_off > n)
        return fail(err, errlen, "the tensor data does not start after the tables");
    for (uint64_t i = at; i < h.data_off; i++)
        if (p[i])
            return fail(err, errlen, "padding byte %llu is not zero", (unsigned long long)i);

    for (uint32_t i = 0; i < h.n_params; i++) {
        pk_param q;
        memcpy(&q, p + h.params_off + (size_t)i * sizeof q, sizeof q);
        if (!name_ok(q.key, sizeof q.key))
            return fail(err, errlen, "param %u: the key is empty or not terminated", (unsigned)i);
        if (q.kind > PK_PARAM_F32)
            return fail(err, errlen, "param %s: kind %u", q.key, (unsigned)q.kind);
        for (uint32_t j = 0; j < i; j++) {
            pk_param r;
            memcpy(&r, p + h.params_off + (size_t)j * sizeof r, sizeof r);
            if (!strcmp(r.key, q.key))
                return fail(err, errlen, "param %s appears twice", q.key);
        }
    }

    if (validate_vocab(p + h.vocab_off, h.vocab_size, err, errlen))
        return -1;

    /* key\0value\0 ...: whole strings, in pairs, no empty key */
    size_t strings = 0;
    for (uint64_t i = h.meta_off; i < h.meta_off + h.meta_size; strings++) {
        const unsigned char *s = p + i;
        const unsigned char *z = memchr(s, 0, (size_t)(h.meta_off + h.meta_size - i));
        if (!z)
            return fail(err, errlen, "the metadata's last string is not terminated");
        if (z == s && strings % 2 == 0)
            return fail(err, errlen, "an empty metadata key");
        if (!utf8_ok(s, (size_t)(z - s)))
            return fail(err, errlen, "a metadata string is not UTF-8");
        i += (uint64_t)(z - s) + 1;
    }
    if (strings % 2)
        return fail(err, errlen, "a metadata key with no value");

    pk_tensor *d = malloc((size_t)h.n_tensors * sizeof *d);
    uint32_t *order = malloc((size_t)h.n_tensors * sizeof *order);
    if (!d || !order) {
        free(d);
        free(order);
        return fail(err, errlen, "out of memory");
    }
    int rc = 0;
    for (uint32_t i = 0; i < h.n_tensors && !rc; i++) {
        memcpy(&d[i], p + h.tensors_off + (size_t)i * sizeof(pk_tensor), sizeof(pk_tensor));
        const pk_tensor *t = &d[i];
        if (!name_ok(t->name, PK_NAME))
            rc = fail(err, errlen, "tensor %u: the name is empty or not terminated", (unsigned)i);
        else if (t->type >= PK_NTYPES)
            rc = fail(err, errlen, "tensor %s: type %u", t->name, (unsigned)t->type);
        else if (t->ndim < 1 || t->ndim > 2 || t->dims[0] == 0 || t->dims[1] == 0
                 || (t->ndim == 1 && t->dims[1] != 1))
            rc = fail(err, errlen, "tensor %s: shape %ux%u in %u dimensions", t->name,
                      (unsigned)t->dims[0], (unsigned)t->dims[1], (unsigned)t->ndim);
        else if (t->offset % PK_ALIGN || t->offset < h.data_off)
            rc = fail(err, errlen, "tensor %s: offset %llu is misaligned or inside the tables",
                      t->name, (unsigned long long)t->offset);
        else if (t->offset > n || t->bytes > n - t->offset)
            rc = fail(err, errlen, "tensor %s runs past the end of the file", t->name);
        /* both dims are u32, so their product times 4 fits a u64 */
        else if ((uint64_t)t->dims[0] * t->dims[1] * (t->type == PK_F32 ? 4 : 1) != t->bytes)
            rc = fail(err, errlen, "tensor %s: %llu bytes for %ux%u", t->name,
                      (unsigned long long)t->bytes, (unsigned)t->dims[0], (unsigned)t->dims[1]);
        else if (t->reserved[0] || t->reserved[1])
            rc = fail(err, errlen, "tensor %s: reserved bytes are not zero", t->name);
    }
    /* names unique, and no two tensors share a byte: sorted by offset, each
     * ends at or before the next starts */
    for (uint32_t i = 0; i < h.n_tensors && !rc; i++) {
        uint32_t j = i;
        for (; j > 0 && d[order[j - 1]].offset > d[i].offset; j--) order[j] = order[j - 1];
        order[j] = i;
    }
    for (uint32_t i = 1; i < h.n_tensors && !rc; i++) {
        const pk_tensor *a = &d[order[i - 1]], *b = &d[order[i]];
        if (a->bytes > b->offset - a->offset)
            rc = fail(err, errlen, "tensors %s and %s overlap", a->name, b->name);
    }
    for (uint32_t i = 0; i < h.n_tensors && !rc; i++)
        for (uint32_t j = i + 1; j < h.n_tensors && !rc; j++)
            if (!strcmp(d[i].name, d[j].name))
                rc = fail(err, errlen, "tensor %s appears twice", d[i].name);
    free(d);
    free(order);
    return rc;
}

/* ---------------------------------------------------------------- bind --- */

typedef struct {
    const unsigned char *buf;
    pk_header h;
    uint8_t *used;
    char *err;
    size_t errlen;
} pk_file;

static const pk_param *param(pk_file *f, const char *key, pk_param *out) {
    for (uint32_t i = 0; i < f->h.n_params; i++) {
        memcpy(out, f->buf + f->h.params_off + (size_t)i * sizeof *out, sizeof *out);
        if (!strcmp(out->key, key))
            return out;
    }
    return NULL;
}

static int u32p(pk_file *f, const char *key, uint32_t lo, uint32_t hi, uint32_t *v) {
    pk_param q;
    if (!param(f, key, &q))
        return fail(f->err, f->errlen, "no param %s", key);
    if (q.kind != PK_PARAM_U32 || q.v.u < lo || q.v.u > hi)
        return fail(f->err, f->errlen, "param %s is not a whole number in [%u, %u]", key,
                    (unsigned)lo, (unsigned)hi);
    *v = q.v.u;
    return 0;
}

static int f32p(pk_file *f, const char *key, float lo, float hi, float *v) {
    pk_param q;
    if (!param(f, key, &q))
        return fail(f->err, f->errlen, "no param %s", key);
    if (q.kind != PK_PARAM_F32 || !(q.v.f >= lo && q.v.f <= hi))
        return fail(f->err, f->errlen, "param %s is not a number in [%g, %g]", key, lo, hi);
    *v = q.v.f;
    return 0;
}

static int find(pk_file *f, const char *name, pk_tensor *t) {
    for (uint32_t i = 0; i < f->h.n_tensors; i++) {
        memcpy(t, f->buf + f->h.tensors_off + (size_t)i * sizeof *t, sizeof *t);
        if (!strcmp(t->name, name)) {
            f->used[i] = 1;
            return 0;
        }
    }
    return fail(f->err, f->errlen, "no tensor %s", name);
}

/* The f32 tensor `name`, which must be d0 x d1 (d1 == 1: a vector) and
 * hold only finite values. */
static int f32t(pk_file *f, const char *name, uint32_t d0, uint32_t d1, const float **out) {
    pk_tensor t;
    if (find(f, name, &t))
        return -1;
    if (t.type != PK_F32 || t.dims[0] != d0 || t.dims[1] != d1 || t.ndim != (d1 == 1 ? 1u : 2u))
        return fail(f->err, f->errlen, "tensor %s is %s %ux%u; the model's is f32 %ux%u", name,
                    t.type == PK_F32 ? "f32" : "q8", (unsigned)t.dims[0], (unsigned)t.dims[1],
                    (unsigned)d0, (unsigned)d1);
    const float *x = (const float *)(f->buf + t.offset);
    for (uint64_t k = 0; k < (uint64_t)d0 * d1; k++)
        if (!isfinite(x[k]))
            return fail(f->err, f->errlen, "tensor %s: element %llu is not finite",
                        name, (unsigned long long)k);
    *out = x;
    return 0;
}

/* The q8 matrix `name`, rows x cols, and its scales */
static int q8t(pk_file *f, const char *name, uint32_t rows, uint32_t cols, pk_q8 *out) {
    pk_tensor t;
    char sname[PK_NAME + 8];
    if (find(f, name, &t))
        return -1;
    if (t.type != PK_Q8 || t.ndim != 2 || t.dims[0] != rows || t.dims[1] != cols)
        return fail(f->err, f->errlen, "tensor %s is %s %ux%u; the model's is q8 %ux%u", name,
                    t.type == PK_F32 ? "f32" : "q8", (unsigned)t.dims[0], (unsigned)t.dims[1],
                    (unsigned)rows, (unsigned)cols);
    if (rows % PK_QR || cols % 32)
        return fail(f->err, f->errlen, "tensor %s: a q8 matrix is 16 rows by 32 columns at a time", name);
    snprintf(sname, sizeof sname, "%s.scale", name);
    if (f32t(f, sname, rows, 1, &out->s))
        return -1;
    out->q = (const int8_t *)(f->buf + t.offset);
    out->rows = rows;
    out->cols = cols;
    return 0;
}

/* A table of twiddles is something anyone can compute, so a file whose are
 * wrong is corrupt or was made for another FFT. The tolerance is far above
 * fp32 rounding and far below any real mistake. */
static int check_twiddles(pk_file *f, const char *what, const float *c, const float *s, uint32_t n) {
    const double pi = 3.14159265358979323846;
    for (uint32_t k = 0; k < n / 2; k++)
        if (fabs(c[k] - cos(2 * pi * k / n)) > 1e-6 || fabs(s[k] - sin(2 * pi * k / n)) > 1e-6)
            return fail(f->err, f->errlen, "%s: FFT twiddle %u is wrong", what, (unsigned)k);
    return 0;
}

static void bitrev(uint16_t *rev, uint32_t m) {
    uint32_t bits = 0;
    while ((1u << bits) < m) bits++;
    for (uint32_t j = 0; j < m; j++) {
        uint32_t r = 0;
        for (uint32_t k = 0; k < bits; k++) r |= ((j >> k) & 1u) << (bits - 1 - k);
        rev[j] = (uint16_t)r;
    }
}

static int bind_vad(pk_file *f, pk_vadw *v) {
    const uint32_t B = PK_VAD_BINS;
    if (f32t(f, "vad.stft.window", PK_VAD_NFFT, 1, &v->window)
        || f32t(f, "vad.fft.cos", PK_VAD_NFFT / 2, 1, &v->fcos)
        || f32t(f, "vad.fft.sin", PK_VAD_NFFT / 2, 1, &v->fsin)
        || f32t(f, "vad.conv1.w", 3 * B, PK_VAD_C1, &v->w1)
        || f32t(f, "vad.conv1.b", PK_VAD_C1, 1, &v->b1)
        || f32t(f, "vad.conv2.w", 3 * PK_VAD_C1, PK_VAD_C2, &v->w2)
        || f32t(f, "vad.conv2.b", PK_VAD_C2, 1, &v->b2)
        || f32t(f, "vad.conv3.w", 3 * PK_VAD_C2, PK_VAD_C3, &v->w3)
        || f32t(f, "vad.conv3.b", PK_VAD_C3, 1, &v->b3)
        || f32t(f, "vad.conv4.w", 3 * PK_VAD_C3, PK_VAD_C4, &v->w4)
        || f32t(f, "vad.conv4.b", PK_VAD_C4, 1, &v->b4)
        || f32t(f, "vad.lstm.wx", PK_VAD_C4, PK_VAD_GATES, &v->wx)
        || f32t(f, "vad.lstm.wh", PK_VAD_HIDDEN, PK_VAD_GATES, &v->wh)
        || f32t(f, "vad.lstm.b", PK_VAD_GATES, 1, &v->bg)
        || f32t(f, "vad.head.w", PK_VAD_HIDDEN, 1, &v->hw)
        || f32t(f, "vad.head.b", 1, 1, &v->hb)
        || check_twiddles(f, "the VAD", v->fcos, v->fsin, PK_VAD_NFFT))
        return -1;
    const double pi = 3.14159265358979323846;
    for (uint32_t i = 0; i < PK_VAD_NFFT; i++)
        if (fabs(v->window[i] - (0.5 - 0.5 * cos(2 * pi * i / PK_VAD_NFFT))) > 1e-6)
            return fail(f->err, f->errlen, "the VAD's window is not Hann at %u", (unsigned)i);
    bitrev(v->rev, PK_VAD_NFFT / 2);
    v->fft.n = PK_VAD_NFFT;
    v->fft.window = v->window;
    v->fft.fcos = v->fcos;
    v->fft.fsin = v->fsin;
    v->fft.rev = v->rev;
    return 0;
}

static int bind_params(pk_file *f, pk_model *m) {
    char key[16];
    if (u32p(f, "sample_rate", 16000, 16000, &m->sample_rate)
        || u32p(f, "n_fft", 64, 4096, &m->n_fft)
        || u32p(f, "win_length", 1, m->n_fft, &m->win_length)
        || u32p(f, "hop_length", 1, 4096, &m->hop)
        || u32p(f, "n_mels", 8, 1024, &m->n_mels)
        || f32p(f, "preemph", 0, 1, &m->preemph)
        || f32p(f, "log_guard", 0, 1, &m->log_guard)
        || f32p(f, "norm_eps", 0, 1, &m->norm_eps)
        || u32p(f, "n_layers", 1, 256, &m->n_layers)
        || u32p(f, "d_model", 32, 16384, &m->d_model)
        || u32p(f, "n_heads", 1, 256, &m->n_heads)
        || u32p(f, "ff_dim", 32, 65536, &m->ff_dim)
        || u32p(f, "conv_kernel", 1, 63, &m->conv_kernel)
        || u32p(f, "sub_channels", 32, 4096, &m->sub_channels)
        || f32p(f, "ln_eps", 0, 1, &m->ln_eps)
        || u32p(f, "vocab", 1, PK_MAX_PIECES, &m->vocab)
        || u32p(f, "pred_hidden", 32, 16384, &m->pred_hidden)
        || u32p(f, "pred_layers", 1, 16, &m->pred_layers)
        || u32p(f, "joint_hidden", 32, 16384, &m->joint_hidden)
        || u32p(f, "n_durations", 1, PK_MAX_DUR, &m->n_dur)
        || u32p(f, "max_symbols", 1, 1000, &m->max_symbols))
        return -1;
    for (uint32_t i = 0; i < m->n_dur; i++) {
        snprintf(key, sizeof key, "dur%u", (unsigned)i);
        if (u32p(f, key, 0, 1000, &m->dur[i]))
            return -1;
    }
    /* the shapes the kernels take: 16-row panels, 32-input blocks, and
     * heads that are whole panels for the attention's float products */
    uint32_t dk = m->n_heads ? m->d_model / m->n_heads : 0;
    if ((m->n_fft & (m->n_fft - 1)) || m->n_mels % 8 || m->d_model % 32 || m->ff_dim % 32
        || m->sub_channels % 32 || m->pred_hidden % 32 || m->joint_hidden % 32
        || m->d_model % m->n_heads || dk % 16 || m->conv_kernel % 2 == 0)
        return fail(f->err, f->errlen, "the model's shapes are not ones pkito's kernels take");
    m->blank = m->vocab;
    m->n_out = m->vocab + 1 + m->n_dur;
    m->n_out_pad = (m->n_out + 15) & ~15u;
    return 0;
}

static int bind_layer(pk_file *f, pk_model *m, uint32_t i, pk_layer *L) {
    const uint32_t D = m->d_model, F = m->ff_dim;
    char n[PK_NAME];
#define NM(s) (snprintf(n, sizeof n, "l%u." s, (unsigned)i), n)
    return f32t(f, NM("ff1.norm.w"), D, 1, &L->ff1_nw) || f32t(f, NM("ff1.norm.b"), D, 1, &L->ff1_nb)
        || q8t(f, NM("ff1.up"), F, D, &L->ff1_up) || q8t(f, NM("ff1.down"), D, F, &L->ff1_down)
        || f32t(f, NM("att.norm.w"), D, 1, &L->att_nw) || f32t(f, NM("att.norm.b"), D, 1, &L->att_nb)
        || q8t(f, NM("att.q"), D, D, &L->q) || q8t(f, NM("att.k"), D, D, &L->k)
        || q8t(f, NM("att.v"), D, D, &L->v) || q8t(f, NM("att.pos"), D, D, &L->pos)
        || q8t(f, NM("att.o"), D, D, &L->o)
        || f32t(f, NM("att.bias_u"), D, 1, &L->bias_u) || f32t(f, NM("att.bias_v"), D, 1, &L->bias_v)
        || f32t(f, NM("conv.norm.w"), D, 1, &L->conv_nw) || f32t(f, NM("conv.norm.b"), D, 1, &L->conv_nb)
        || q8t(f, NM("conv.pw1"), 2 * D, D, &L->pw1)
        || f32t(f, NM("conv.dw"), D, m->conv_kernel, &L->dw) || f32t(f, NM("conv.dw_b"), D, 1, &L->dw_b)
        || q8t(f, NM("conv.pw2"), D, D, &L->pw2)
        || f32t(f, NM("ff2.norm.w"), D, 1, &L->ff2_nw) || f32t(f, NM("ff2.norm.b"), D, 1, &L->ff2_nb)
        || q8t(f, NM("ff2.up"), F, D, &L->ff2_up) || q8t(f, NM("ff2.down"), D, F, &L->ff2_down)
        || f32t(f, NM("out.norm.w"), D, 1, &L->out_nw) || f32t(f, NM("out.norm.b"), D, 1, &L->out_nb);
#undef NM
}

static int bind(pk_file *f, pk_model *m) {
    if (bind_params(f, m))
        return -1;
    const uint32_t C = m->sub_channels, D = m->d_model, P = m->pred_hidden, J = m->joint_hidden;
    const uint32_t N = m->n_fft, bins = N / 2 + 1;
    const float *wfile, *fc, *fs;
    if (f32t(f, "mel.fb", m->n_mels, bins, &m->mel_fb)
        || f32t(f, "mel.window", m->win_length, 1, &wfile)
        || f32t(f, "mel.fft.cos", N / 2, 1, &fc) || f32t(f, "mel.fft.sin", N / 2, 1, &fs)
        || check_twiddles(f, "the features", fc, fs, N)
        || f32t(f, "pos.freq", D / 2, 1, &m->pos_freq)
        || f32t(f, "sub.conv0.w", C, 9, &m->conv0_w) || f32t(f, "sub.conv0.b", C, 1, &m->conv0_b)
        || f32t(f, "sub.dw1.w", C, 9, &m->dw1_w) || f32t(f, "sub.dw1.b", C, 1, &m->dw1_b)
        || q8t(f, "sub.pw1.w", C, C, &m->sub_pw1) || f32t(f, "sub.pw1.b", C, 1, &m->pw1_b)
        || f32t(f, "sub.dw2.w", C, 9, &m->dw2_w) || f32t(f, "sub.dw2.b", C, 1, &m->dw2_b)
        || q8t(f, "sub.pw2.w", C, C, &m->sub_pw2) || f32t(f, "sub.pw2.b", C, 1, &m->pw2_b)
        || q8t(f, "sub.out.w", D, C * (m->n_mels / 8), &m->sub_out)
        || f32t(f, "sub.out.b", D, 1, &m->out_b))
        return -1;

    /* the window centred in n_fft, as torch.stft pads a shorter one; the
     * filterbank's nonzero run in each band, which is all the features read */
    m->window = calloc(N, sizeof *m->window);
    m->rev = calloc(N / 2, sizeof *m->rev);
    m->mel_lo = calloc(m->n_mels, sizeof *m->mel_lo);
    m->mel_n = calloc(m->n_mels, sizeof *m->mel_n);
    m->layers = calloc(m->n_layers, sizeof *m->layers);
    m->lstm = calloc(m->pred_layers, sizeof *m->lstm);
    if (!m->window || !m->rev || !m->mel_lo || !m->mel_n || !m->layers || !m->lstm)
        return fail(f->err, f->errlen, "out of memory");
    memcpy(m->window + (N - m->win_length) / 2, wfile, m->win_length * sizeof(float));
    bitrev(m->rev, N / 2);
    m->fft.n = N;
    m->fft.window = m->window;
    m->fft.fcos = fc;
    m->fft.fsin = fs;
    m->fft.rev = m->rev;
    for (uint32_t b = 0; b < m->n_mels; b++) {
        const float *row = m->mel_fb + (size_t)b * bins;
        uint32_t lo = 0, hi = bins;
        while (lo < bins && row[lo] == 0) lo++;
        while (hi > lo && row[hi - 1] == 0) hi--;
        m->mel_lo[b] = lo;
        m->mel_n[b] = hi - lo;
    }

    for (uint32_t i = 0; i < m->n_layers; i++)
        if (bind_layer(f, m, i, &m->layers[i]))
            return -1;

    char n[PK_NAME];
    if (f32t(f, "dec.embed", m->vocab + 1, P, &m->embed))
        return -1;
    for (uint32_t i = 0; i < m->pred_layers; i++) {
        snprintf(n, sizeof n, "dec.l%u.wx", (unsigned)i);
        if (q8t(f, n, 4 * P, P, &m->lstm[i].wx))
            return -1;
        snprintf(n, sizeof n, "dec.l%u.wh", (unsigned)i);
        if (q8t(f, n, 4 * P, P, &m->lstm[i].wh))
            return -1;
        snprintf(n, sizeof n, "dec.l%u.b", (unsigned)i);
        if (f32t(f, n, 4 * P, 1, &m->lstm[i].b))
            return -1;
    }
    if (q8t(f, "joint.enc.w", J, D, &m->jenc) || f32t(f, "joint.enc.b", J, 1, &m->jenc_b)
        || q8t(f, "joint.pred.w", J, P, &m->jpred) || f32t(f, "joint.pred.b", J, 1, &m->jpred_b)
        || q8t(f, "joint.out.w", m->n_out_pad, J, &m->jout)
        || f32t(f, "joint.out.b", m->n_out_pad, 1, &m->jout_b))
        return -1;
    if (bind_vad(f, &m->vad))
        return -1;

    /* the vocabulary, validated already */
    const unsigned char *v = f->buf + f->h.vocab_off;
    m->n_pieces = rd32(v);
    if (m->n_pieces != m->vocab)
        return fail(f->err, f->errlen, "%u pieces for a vocabulary of %u",
                    (unsigned)m->n_pieces, (unsigned)m->vocab);
    m->piece_off = (const uint32_t *)(v + 4);
    m->piece_type = v + 4 + 4 * ((size_t)m->n_pieces + 1);
    m->piece_text = (const char *)m->piece_type + m->n_pieces;
    m->punct_len = rd32((const unsigned char *)m->piece_text + m->piece_off[m->n_pieces]);
    m->punct = m->piece_text + m->piece_off[m->n_pieces] + 4;
    m->meta = (const char *)f->buf + f->h.meta_off;
    m->meta_size = (size_t)f->h.meta_size;
    memcpy(m->name, f->h.name, sizeof m->name);

    for (uint32_t i = 0; i < f->h.n_tensors; i++)
        if (!f->used[i]) {
            pk_tensor t;
            memcpy(&t, f->buf + f->h.tensors_off + (size_t)i * sizeof t, sizeof t);
            return fail(f->err, f->errlen, "tensor %s is not part of the model", t.name);
        }
    return 0;
}

/* ---------------------------------------------------------------- load --- */

pk_model *pk_model_load(const char *path, char *err, size_t errlen) {
    char why[256];
    size_t size = 0;
    const unsigned char *map = pk_map_file(path, &size, why, sizeof why);
    if (!map) {
        fail(err, errlen, "%s: %s", path, why);
        return NULL;
    }
    pk_model *m = calloc(1, sizeof *m);
    if (!m) {
        pk_unmap(map, size);
        fail(err, errlen, "%s: out of memory", path);
        return NULL;
    }
    m->map = map;
    m->size = size;
    pk_file f = { map, { 0 }, NULL, why, sizeof why };
    int rc = pk_pkf_validate(map, size, why, sizeof why);
    if (!rc) {
        memcpy(&f.h, map, sizeof f.h);
        f.used = calloc(f.h.n_tensors, 1);
        rc = f.used ? bind(&f, m) : fail(why, sizeof why, "out of memory");
        free(f.used);
    }
    if (rc) {
        fail(err, errlen, "%s: %s", path, why);
        pk_model_free(m);
        return NULL;
    }
    return m;
}

void pk_model_free(pk_model *m) {
    if (!m)
        return;
    free(m->window);
    free(m->rev);
    free(m->mel_lo);
    free(m->mel_n);
    free(m->layers);
    free(m->lstm);
    pk_unmap(m->map, m->size);
    free(m);
}

const char *pk_model_name(const pk_model *m) { return m->name; }
int pk_model_rate(const pk_model *m) { return (int)m->sample_rate; }
double pk_model_frame_seconds(const pk_model *m) { return 8.0 * m->hop / m->sample_rate; }

int pk_model_meta_at(const pk_model *m, int i, const char **key, const char **value) {
    size_t at = 0;
    for (int k = 0; at < m->meta_size; k++) {
        const char *kk = m->meta + at;
        at += strlen(kk) + 1;
        const char *vv = m->meta + at;
        at += strlen(vv) + 1;
        if (k == i) {
            *key = kk;
            *value = vv;
            return 0;
        }
    }
    return -1;
}

const char *pk_model_meta(const pk_model *m, const char *key) {
    const char *k, *v;
    for (int i = 0; pk_model_meta_at(m, i, &k, &v) == 0; i++)
        if (!strcmp(k, key))
            return v;
    return NULL;
}
