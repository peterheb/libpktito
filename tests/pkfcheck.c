/* The model-file loader on a malformed corpus (docs/MEMORY_SAFETY.md), and
 * the engine end to end on the tiny model.
 *
 * Every file here is built in memory by a writer of this test's own,
 * written from core/pkf.h rather than from the converter, out of the tables
 * and tensors of tests/data/tiny.pkf (`pktito-tools tiny` writes it). The
 * first check is that the writer remakes the converter's bytes exactly, so
 * the format is held to its definition from both sides. Then:
 *
 *   valid      the tiny model, loaded from its path and from bytes
 *   truncated  every length of a small file, with the header's size left
 *              alone and with it patched to agree; the tiny model cut inside
 *              each tensor and inside each table
 *   corrupt    each field of the header, a param, a descriptor, the
 *              vocabulary and the metadata set to each way it can be wrong,
 *              one at a time
 *   meaning    byte-valid files that are not the model: a tensor missing,
 *              extra, of the wrong shape or type or with a value that is not
 *              finite; a param missing, unknown, of the wrong kind or out of
 *              range; a twiddle, position frequency or window wrong; a finite
 *              extreme weight whose arithmetic overflows; the vocabulary's count off
 *   flipped    random bits of everything before the tensor data changed,
 *              20,000 times over, and the engine run on what still loads
 *   engine     the tiny model through pk_asr_run, pk_text, the VAD and
 *              pk_chunks: every length round a frame's edges, samples that
 *              are not finite, the text cut at every size
 *
 * and every file must load or be refused with a message - never crash,
 * never hand back a model that should have been refused. `make
 * check-sanitize` runs it under ASan and UBSan.
 *
 *     ./tests/pkfcheck tests/data/tiny.pkf [WAV_OUT]
 *
 * WAV_OUT, if given, gets the 35 s recording the engine checks used, for
 * `make check-c` to put through ./pktito as well.
 */
#include "model.h"
#include "pkf.h"
#include "vmath.h"
#include "pktito.h"
#include "wav.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_bad, g_cases;

#define FAIL(...) do { if (g_bad++ < 40) { printf("FAIL "); printf(__VA_ARGS__); printf("\n"); } } while (0)

/* ------------------------------------------------------------ a writer --- */

#define MAXP 64
#define MAXT 256

typedef struct {
    char name[PK_NAME];
    uint32_t type, ndim, d0, d1;
    const unsigned char *data;
    size_t bytes;
} tensor;

typedef struct {
    char arch[32], name[64];
    uint8_t digest[32];
    int np, nt;
    pk_param p[MAXP];
    tensor t[MAXT];
    const unsigned char *vocab, *meta;
    size_t vocab_n, meta_n;
} model;

static size_t al(size_t x) { return (x + PK_ALIGN - 1) & ~(size_t)(PK_ALIGN - 1); }

/* A file by core/pkf.h: header, params, descriptors, vocabulary, metadata,
 * zeros to 64, then each tensor at the next 64-byte boundary, the file
 * ending where the last one does. */
static unsigned char *build(const model *m, size_t *size) {
    size_t at = sizeof(pk_header) + (size_t)m->np * sizeof(pk_param)
              + (size_t)m->nt * sizeof(pk_tensor) + m->vocab_n + m->meta_n;
    size_t data = al(at), end = data;
    for (int i = 0; i < m->nt; i++)
        end = al(end) + m->t[i].bytes;
    unsigned char *p = calloc(1, end + PK_ALIGN);
    if (!p) { printf("pkfcheck: out of memory\n"); exit(2); }
    pk_header h;
    memset(&h, 0, sizeof h);
    h.magic = PK_MAGIC;
    h.version = PK_VERSION;
    memcpy(h.arch, m->arch, sizeof h.arch);
    memcpy(h.name, m->name, sizeof h.name);
    h.n_params = (uint32_t)m->np;
    h.n_tensors = (uint32_t)m->nt;
    h.params_off = sizeof h;
    h.tensors_off = h.params_off + (uint64_t)m->np * sizeof(pk_param);
    h.vocab_off = h.tensors_off + (uint64_t)m->nt * sizeof(pk_tensor);
    h.vocab_size = m->vocab_n;
    h.meta_off = h.vocab_off + m->vocab_n;
    h.meta_size = m->meta_n;
    h.data_off = data;
    h.file_bytes = end;
    memcpy(h.digest, m->digest, sizeof h.digest);
    memcpy(p, &h, sizeof h);
    memcpy(p + h.params_off, m->p, (size_t)m->np * sizeof(pk_param));
    memcpy(p + h.vocab_off, m->vocab, m->vocab_n);
    memcpy(p + h.meta_off, m->meta, m->meta_n);
    size_t off = data;
    for (int i = 0; i < m->nt; i++) {
        pk_tensor d;
        memset(&d, 0, sizeof d);
        memcpy(d.name, m->t[i].name, sizeof d.name);
        d.type = m->t[i].type;
        d.ndim = m->t[i].ndim;
        d.dims[0] = m->t[i].d0;
        d.dims[1] = m->t[i].d1;
        off = al(off);
        d.offset = off;
        d.bytes = m->t[i].bytes;
        memcpy(p + h.tensors_off + (size_t)i * sizeof d, &d, sizeof d);
        memcpy(p + off, m->t[i].data, m->t[i].bytes);
        off += m->t[i].bytes;
    }
    *size = end;
    return p;
}

/* A file's tables back into a model, its tensors pointing into the file */
static void parse(const unsigned char *f, model *m) {
    pk_header h;
    memcpy(&h, f, sizeof h);
    memset(m, 0, sizeof *m);
    memcpy(m->arch, h.arch, sizeof m->arch);
    memcpy(m->name, h.name, sizeof m->name);
    memcpy(m->digest, h.digest, sizeof m->digest);
    m->np = (int)h.n_params;
    m->nt = (int)h.n_tensors;
    memcpy(m->p, f + h.params_off, (size_t)m->np * sizeof(pk_param));
    for (int i = 0; i < m->nt; i++) {
        pk_tensor d;
        memcpy(&d, f + h.tensors_off + (size_t)i * sizeof d, sizeof d);
        memcpy(m->t[i].name, d.name, PK_NAME);
        m->t[i].type = d.type;
        m->t[i].ndim = d.ndim;
        m->t[i].d0 = d.dims[0];
        m->t[i].d1 = d.dims[1];
        m->t[i].data = f + d.offset;
        m->t[i].bytes = (size_t)d.bytes;
    }
    m->vocab = f + h.vocab_off;
    m->vocab_n = (size_t)h.vocab_size;
    m->meta = f + h.meta_off;
    m->meta_n = (size_t)h.meta_size;
}

static pk_header *hdr(unsigned char *p) { return (pk_header *)(void *)p; }
static pk_param *prm(unsigned char *p, int i) {
    return (pk_param *)(void *)(p + hdr(p)->params_off + (size_t)i * sizeof(pk_param));
}
static pk_tensor *dsc(unsigned char *p, int i) {
    return (pk_tensor *)(void *)(p + hdr(p)->tensors_off + (size_t)i * sizeof(pk_tensor));
}
static unsigned char *voc(unsigned char *p) { return p + hdr(p)->vocab_off; }
static uint32_t rd32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static void wr32(unsigned char *p, uint32_t v) { memcpy(p, &v, 4); }

static int find_tensor(const model *m, const char *name) {
    for (int i = 0; i < m->nt; i++)
        if (!strcmp(m->t[i].name, name)) return i;
    printf("pkfcheck: the tiny model has no %s\n", name);
    exit(2);
}

static int find_param(const model *m, const char *key) {
    for (int i = 0; i < m->np; i++)
        if (!strcmp(m->p[i].key, key)) return i;
    printf("pkfcheck: the tiny model has no param %s\n", key);
    exit(2);
}

/* ----------------------------------------------------------- the checks --- */

static unsigned char *g_file;     /* tests/data/tiny.pkf, as read */
static size_t g_n;
static model g_m;                 /* its tables */

/* a built file must load (want 1) or be refused with a message (want 0) */
static void expect(const char *what, const unsigned char *p, size_t n, int want) {
    char err[256] = "";
    pk_model *m = pk_model_from_bytes(p, n, err, sizeof err);
    g_cases++;
    if (!!m != want)
        FAIL("%s: %s", what, m ? "loaded" : err);
    else if (!m && !err[0])
        FAIL("%s: refused without a reason", what);
    pk_model_free(m);
}

static void expect_model(const char *what, const model *m, int want) {
    size_t n;
    unsigned char *p = build(m, &n);
    expect(what, p, n, want);
    free(p);
}

/* a rule of the format's own, which the binder might catch for some other
 * reason: pk_pkf_validate itself must refuse the file */
static void expect_invalid(const char *what, const model *m) {
    size_t n;
    unsigned char *p = build(m, &n);
    char err[256] = "";
    g_cases++;
    if (!pk_pkf_validate(p, n, err, sizeof err))
        FAIL("%s: the validator let it through", what);
    else if (!err[0])
        FAIL("%s: the validator refused without a reason", what);
    free(p);
}

static unsigned char *fresh(size_t *n) { return build(&g_m, n); }

#define CORRUPT(what, stmt) do { size_t n_; unsigned char *c = fresh(&n_); stmt; expect(what, c, n_, 0); free(c); } while (0)

static void valid_cases(const char *path) {
    char err[256];
    err[0] = 0;
    g_cases++;
    if (pk_pkf_validate(NULL, sizeof(pk_header), err, sizeof err) == 0 || !err[0])
        FAIL("a null validation buffer: %s", err[0] ? "accepted" : "no reason");
    err[0] = 0;
    g_cases++;
    pk_model *none = pk_model_from_bytes(NULL, sizeof(pk_header), err, sizeof err);
    if (none || !err[0])
        FAIL("a null model buffer: %s", none ? "loaded" : "no reason");
    pk_model_free(none);
    err[0] = 0;
    g_cases++;
    none = pk_model_load(NULL, err, sizeof err);
    if (none || !err[0])
        FAIL("a null model path: %s", none ? "loaded" : "no reason");
    pk_model_free(none);

    pk_model *m = pk_model_load(path, err, sizeof err);
    g_cases++;
    if (!m) {
        FAIL("the tiny model from its path: %s", err);
        return;
    }
    const char *k, *v;
    int pairs = 0;
    while (pk_model_meta_at(m, pairs, &k, &v) == 0) pairs++;
    g_cases++;
    if (strcmp(pk_model_name(m), "pktito-tiny") || !pk_model_meta(m, "family")
        || strcmp(pk_model_meta(m, "family"), "tiny") || pk_model_meta(m, "no such key") || pairs < 4)
        FAIL("the tiny model's name or metadata");
    pk_model_free(m);
    size_t n;
    unsigned char *f = fresh(&n);
    g_cases++;
    if (n != g_n || memcmp(f, g_file, n))
        FAIL("this test's writer does not remake the converter's bytes (%zu against %zu)", n, g_n);
    expect("the tiny model from bytes", f, n, 1);
    free(f);
    model e = g_m;
    e.meta_n = 0;
    expect_model("no metadata at all", &e, 1);
}

static void truncated_cases(void) {
    /* a small file, valid to the validator though no model: every length */
    static const float three[3] = { 1, 2, 3 };
    static unsigned char q[16 * 32];
    static const unsigned char vocab[] = {
        2, 0, 0, 0,  0, 0, 0, 0,  4, 0, 0, 0,  5, 0, 0, 0,  1, 1,
        0xE2, 0x96, 0x81, 'a', 'b',  1, 0, 0, 0, '.' };
    static const char meta[] = "k\0v\0";
    model s;
    memset(&s, 0, sizeof s);
    strcpy(s.arch, "parakeet-tdt");
    strcpy(s.name, "small");
    s.np = 2;
    strcpy(s.p[0].key, "a");
    s.p[0].kind = PK_PARAM_U32;
    s.p[0].v.u = 1;
    strcpy(s.p[1].key, "b");
    s.p[1].kind = PK_PARAM_F32;
    s.p[1].v.f = 0.5f;
    s.nt = 2;
    s.t[0] = (tensor){ "t0", PK_F32, 1, 3, 1, (const unsigned char *)three, sizeof three };
    s.t[1] = (tensor){ "t1", PK_Q8, 2, 16, 32, q, sizeof q };
    s.vocab = vocab;
    s.vocab_n = sizeof vocab;
    s.meta = (const unsigned char *)meta;
    s.meta_n = sizeof meta - 1;
    size_t m;
    unsigned char *g = build(&s, &m);
    char e[256] = "";
    g_cases++;
    if (pk_pkf_validate(g, m, e, sizeof e))
        FAIL("the small file is not valid: %s", e);
    for (size_t len = 0; len < m; len++) {
        unsigned char *c = malloc(len ? len : 1);
        memcpy(c, g, len);
        e[0] = 0;
        g_cases++;
        int r = pk_pkf_validate(c, len, e, sizeof e);
        if (r == 0 || !e[0])
            FAIL("truncated to %zu: %s", len, r == 0 ? "accepted" : "no reason");
        if (len >= sizeof(pk_header)) {
            hdr(c)->file_bytes = len;
            e[0] = 0;
            g_cases++;
            r = pk_pkf_validate(c, len, e, sizeof e);
            if (r == 0 || !e[0])         /* the last tensor ends the file: every cut loses some */
                FAIL("truncated to %zu, size patched: %s", len, r == 0 ? "accepted" : "no reason");
        }
        free(c);
    }
    free(g);

    /* The same file with every byte after the vocabulary valid text, its
     * tensors spelled in ASCII, and the vocabulary's offsets going backward.
     * Without the order check the UTF-8 check would take a piece of 2^32 - 1
     * bytes and read on past the end of the file, which it only reaches when
     * nothing on the way is invalid; check-sanitize sees the read. The
     * buffer is the file's size exactly, as the truncations' are. */
    {
        static unsigned char text3[12], text_q[16 * 32];
        memset(text3, 'a', sizeof text3);
        memset(text_q, 'b', sizeof text_q);
        model t = s;
        static const char ascii_meta[] = "key\0value\0";
        t.t[0].data = text3;
        t.t[1].data = text_q;
        t.meta = (const unsigned char *)ascii_meta;
        t.meta_n = sizeof ascii_meta - 1;
        unsigned char v2[sizeof vocab];
        memcpy(v2, vocab, sizeof vocab);
        v2[8] = 5;                                          /* off[1] = 5 */
        v2[12] = 4;                                         /* off[2] = 4: backward */
        t.vocab = v2;
        size_t tn;
        unsigned char *tf = build(&t, &tn);
        unsigned char *exact = malloc(tn);
        memcpy(exact, tf, tn);
        e[0] = 0;
        g_cases++;
        if (!pk_pkf_validate(exact, tn, e, sizeof e) || !e[0])
            FAIL("vocabulary offsets backward, all text after: %s", e[0] ? "accepted" : "no reason");
        free(exact);
        free(tf);
    }

    /* the tiny model cut inside each tensor, and inside each table */
    size_t n;
    unsigned char *f = fresh(&n);
    pk_header h;
    memcpy(&h, f, sizeof h);
    for (int i = 0; i < g_m.nt; i++) {
        size_t cut = (size_t)(dsc(f, i)->offset + dsc(f, i)->bytes / 2);
        unsigned char *c = malloc(cut);
        memcpy(c, f, cut);
        hdr(c)->file_bytes = cut;
        expect("cut inside a tensor", c, cut, 0);
        free(c);
    }
    const uint64_t marks[] = { h.params_off + 20, h.tensors_off + 50, h.vocab_off + 3,
                               h.vocab_off + h.vocab_size / 2, h.meta_off + h.meta_size / 2,
                               h.data_off - 1, h.data_off };
    for (size_t i = 0; i < sizeof marks / sizeof marks[0]; i++) {
        unsigned char *c = malloc((size_t)marks[i]);
        memcpy(c, f, (size_t)marks[i]);
        hdr(c)->file_bytes = marks[i];
        expect("cut inside the tables", c, (size_t)marks[i], 0);
        free(c);
    }
    free(f);
}

static void header_cases(void) {
    CORRUPT("magic", hdr(c)->magic ^= 1);
    CORRUPT("version 0", hdr(c)->version = 0);
    CORRUPT("a later version", hdr(c)->version = PK_VERSION + 1);
    CORRUPT("file_bytes one more", hdr(c)->file_bytes++);
    CORRUPT("file_bytes one fewer", hdr(c)->file_bytes--);
    CORRUPT("another architecture", memcpy(hdr(c)->arch, "parakeet-ctc", 13));
    CORRUPT("architecture unterminated", memset(hdr(c)->arch, 'x', sizeof hdr(c)->arch));
    CORRUPT("bytes after the architecture", hdr(c)->arch[31] = 'x');
    CORRUPT("name empty", hdr(c)->name[0] = 0);
    CORRUPT("name unterminated", memset(hdr(c)->name, 'x', sizeof hdr(c)->name));
    CORRUPT("bytes after the name", hdr(c)->name[63] = 'x');
    for (size_t i = 0; i < sizeof hdr((unsigned char *)0)->reserved; i += 7)
        CORRUPT("a reserved byte", hdr(c)->reserved[i] = 1);
    CORRUPT("n_params 0", hdr(c)->n_params = 0);
    CORRUPT("n_params 257", hdr(c)->n_params = 257);
    CORRUPT("n_params huge", hdr(c)->n_params = 0xFFFFFFFFu);
    CORRUPT("n_params one fewer", hdr(c)->n_params--);
    CORRUPT("n_params one more", hdr(c)->n_params++);
    CORRUPT("n_tensors 0", hdr(c)->n_tensors = 0);
    CORRUPT("n_tensors 4097", hdr(c)->n_tensors = 4097);
    CORRUPT("n_tensors huge", hdr(c)->n_tensors = 0xFFFFFFFFu);
    CORRUPT("n_tensors one fewer", hdr(c)->n_tensors--);
    CORRUPT("params_off", hdr(c)->params_off += 40);
    CORRUPT("tensors_off", hdr(c)->tensors_off += 96);
    CORRUPT("tensors_off one fewer", hdr(c)->tensors_off--);
    CORRUPT("vocab_off", hdr(c)->vocab_off++);
    CORRUPT("vocab_size huge", hdr(c)->vocab_size = UINT64_MAX);
    CORRUPT("vocab_size one more", hdr(c)->vocab_size++);
    CORRUPT("vocab_size one fewer", hdr(c)->vocab_size--);
    CORRUPT("vocab_size 0", hdr(c)->vocab_size = 0);
    CORRUPT("meta_off", hdr(c)->meta_off--);
    CORRUPT("meta_size huge", hdr(c)->meta_size = UINT64_MAX - 7);
    CORRUPT("meta_size into the data", hdr(c)->meta_size += 64);
    CORRUPT("data_off a block later", hdr(c)->data_off += 64);
    CORRUPT("data_off a block earlier", hdr(c)->data_off -= 64);
    CORRUPT("data_off misaligned", hdr(c)->data_off += 1);
    CORRUPT("data_off past the end", hdr(c)->data_off = (uint64_t)1 << 40);
    {
        /* padding between the tables and the data that is not zeros */
        size_t n;
        unsigned char *c = fresh(&n);
        uint64_t end = hdr(c)->meta_off + hdr(c)->meta_size;
        if (end < hdr(c)->data_off) {
            c[end] = 1;
            expect("padding that is not zero", c, n, 0);
        } else {
            FAIL("the tiny model has no padding to corrupt");
        }
        free(c);
    }
}

static void param_cases(void) {
    int ns = find_param(&g_m, "n_fft");
    CORRUPT("param key unterminated", memset(prm(c, 3)->key, 'x', PK_KEY));
    CORRUPT("param key empty", prm(c, 3)->key[0] = 0);
    CORRUPT("bytes after a param key", prm(c, 3)->key[PK_KEY - 1] = 'x');
    CORRUPT("a param twice", memcpy(prm(c, 4)->key, prm(c, 3)->key, PK_KEY));
    CORRUPT("param kind 3", prm(c, 3)->kind = 3);
    CORRUPT("param kind huge", prm(c, 3)->kind = 0xFFFFFFFFu);
    CORRUPT("an int where a float is", prm(c, find_param(&g_m, "preemph"))->kind = PK_PARAM_U32);
    CORRUPT("a signed int where an unsigned is", prm(c, ns)->kind = PK_PARAM_I32);
    CORRUPT("sample rate 8000", prm(c, find_param(&g_m, "sample_rate"))->v.u = 8000);
    CORRUPT("n_fft not a power of two", prm(c, ns)->v.u = 500);
    CORRUPT("n_fft 32", prm(c, ns)->v.u = 32);
    CORRUPT("win_length past n_fft", prm(c, find_param(&g_m, "win_length"))->v.u = 513);
    CORRUPT("hop_length 0", prm(c, find_param(&g_m, "hop_length"))->v.u = 0);
    CORRUPT("n_mels not a multiple of 8", prm(c, find_param(&g_m, "n_mels"))->v.u = 12);
    CORRUPT("d_model not a multiple of 32", prm(c, find_param(&g_m, "d_model"))->v.u = 48);
    CORRUPT("heads that do not divide d_model", prm(c, find_param(&g_m, "n_heads"))->v.u = 3);
    CORRUPT("heads narrower than a panel", prm(c, find_param(&g_m, "n_heads"))->v.u = 8);
    CORRUPT("an even convolution", prm(c, find_param(&g_m, "conv_kernel"))->v.u = 8);
    CORRUPT("a convolution past 63", prm(c, find_param(&g_m, "conv_kernel"))->v.u = 65);
    CORRUPT("preemph NaN", prm(c, find_param(&g_m, "preemph"))->v.f = NAN);
    CORRUPT("preemph 2", prm(c, find_param(&g_m, "preemph"))->v.f = 2.0f);
    CORRUPT("ln_eps negative", prm(c, find_param(&g_m, "ln_eps"))->v.f = -1e-5f);
    CORRUPT("ln_eps 0", prm(c, find_param(&g_m, "ln_eps"))->v.f = 0.0f);
    CORRUPT("norm_eps 0", prm(c, find_param(&g_m, "norm_eps"))->v.f = 0.0f);
    CORRUPT("log_guard 0", prm(c, find_param(&g_m, "log_guard"))->v.f = 0.0f);
    CORRUPT("norm_eps subnormal", prm(c, find_param(&g_m, "norm_eps"))->v.f = 1e-40f);
    CORRUPT("ln_eps infinite", prm(c, find_param(&g_m, "ln_eps"))->v.f = INFINITY);
    CORRUPT("vocab 0", prm(c, find_param(&g_m, "vocab"))->v.u = 0);
    CORRUPT("vocab one fewer than the pieces", prm(c, find_param(&g_m, "vocab"))->v.u--);
    CORRUPT("n_layers 0", prm(c, find_param(&g_m, "n_layers"))->v.u = 0);
    CORRUPT("n_layers one more than the tensors", prm(c, find_param(&g_m, "n_layers"))->v.u++);
    CORRUPT("n_durations 0", prm(c, find_param(&g_m, "n_durations"))->v.u = 0);
    CORRUPT("n_durations 17", prm(c, find_param(&g_m, "n_durations"))->v.u = 17);
    CORRUPT("n_durations one fewer", prm(c, find_param(&g_m, "n_durations"))->v.u--);
    CORRUPT("a duration of 2000", prm(c, find_param(&g_m, "dur0"))->v.u = 2000);
    CORRUPT("max_symbols 0", prm(c, find_param(&g_m, "max_symbols"))->v.u = 0);
    CORRUPT("pred_layers one more", prm(c, find_param(&g_m, "pred_layers"))->v.u++);
    CORRUPT("a param missing", prm(c, find_param(&g_m, "hop_length"))->key[9] = 'x');
    {
        model e = g_m;
        e.p[e.np] = e.p[5];
        e.np++;
        expect_invalid("a param twice, the copy added", &e);
        e = g_m;
        e.p[e.np] = e.p[0];
        strcpy(e.p[e.np].key, "flavour");
        e.np++;
        expect_model("a param this build does not know", &e, 0);
        e = g_m;
        e.p[e.np] = e.p[find_param(&g_m, "dur4")];
        strcpy(e.p[e.np].key, "dur5");
        e.np++;
        expect_model("a duration past n_durations", &e, 0);
    }
}

static void descriptor_cases(void) {
    const int last = g_m.nt - 1;
    CORRUPT("tensor name unterminated", memset(dsc(c, 7)->name, 'x', PK_NAME));
    CORRUPT("tensor name empty", dsc(c, 7)->name[0] = 0);
    CORRUPT("bytes after a tensor name", dsc(c, 7)->name[PK_NAME - 1] = 'x');
    CORRUPT("a tensor twice", memcpy(dsc(c, 8)->name, dsc(c, 7)->name, PK_NAME));
    CORRUPT("type 2", dsc(c, 7)->type = 2);
    CORRUPT("type huge", dsc(c, 7)->type = 0xFFFFFFFFu);
    CORRUPT("ndim 0", dsc(c, 7)->ndim = 0);
    CORRUPT("ndim 3", dsc(c, 7)->ndim = 3);
    CORRUPT("a vector of two dims", dsc(c, find_tensor(&g_m, "pos.freq"))->dims[1] = 2);
    CORRUPT("dim 0", dsc(c, 7)->dims[0] = 0);
    CORRUPT("second dim 0", dsc(c, find_tensor(&g_m, "mel.fb"))->dims[1] = 0);
    CORRUPT("offset misaligned", dsc(c, 7)->offset += 4);
    CORRUPT("offset in the tables", dsc(c, 0)->offset = 64);
    CORRUPT("offset before the data", dsc(c, 0)->offset = hdr(c)->data_off - 64);
    CORRUPT("offset past the end", dsc(c, 7)->offset = (uint64_t)1 << 40);
    CORRUPT("offset at the end", dsc(c, last)->offset = (hdr(c)->file_bytes + 63) & ~(uint64_t)63);
    CORRUPT("bytes one element short", dsc(c, 7)->bytes -= 4);
    CORRUPT("bytes past the end", dsc(c, last)->bytes += 64);
    CORRUPT("dims and bytes that wrap", (dsc(c, 7)->dims[0] = 0x80000000u,
                                         dsc(c, 7)->dims[1] = 0x80000000u, dsc(c, 7)->bytes = 0));
    CORRUPT("dims whose product wraps 32 bits",
            (dsc(c, 7)->dims[0] = 0x10000u, dsc(c, 7)->dims[1] = 0x10000u,
             dsc(c, 7)->bytes = (uint64_t)4 << 32));
    CORRUPT("bytes huge", dsc(c, 7)->bytes = UINT64_MAX - 3);
    CORRUPT("reserved word 0", dsc(c, 7)->reserved[0] = 1);
    CORRUPT("reserved word 1", dsc(c, 7)->reserved[1] = 1);
    CORRUPT("tensors overlap", dsc(c, 8)->offset = dsc(c, 7)->offset);
    CORRUPT("tensors overlap by a block", dsc(c, 9)->offset = dsc(c, 8)->offset + 64);
    CORRUPT("the last two overlap", dsc(c, last)->offset = dsc(c, last - 1)->offset);
}

/* the vocabulary's fields in a built file: n, off[n+1], type[n], text, np, punct */
static unsigned char *v_off(unsigned char *c, uint32_t i) { return voc(c) + 4 + 4 * (size_t)i; }
static unsigned char *v_type(unsigned char *c, uint32_t i) {
    return voc(c) + 4 + 4 * ((size_t)rd32(voc(c)) + 1) + i;
}
static unsigned char *v_text(unsigned char *c, uint32_t i) {
    uint32_t n = rd32(voc(c));
    return voc(c) + 4 + 4 * ((size_t)n + 1) + n + rd32(v_off(c, i));
}
static unsigned char *v_np(unsigned char *c) { return v_text(c, rd32(voc(c))); }

/* the first piece whose text starts with these bytes */
static uint32_t piece_with(unsigned char *c, const char *lead) {
    uint32_t n = rd32(voc(c));
    for (uint32_t i = 0; i < n; i++)
        if (rd32(v_off(c, i + 1)) - rd32(v_off(c, i)) >= strlen(lead)
            && !memcmp(v_text(c, i), lead, strlen(lead)))
            return i;
    printf("pkfcheck: no piece starts with %s\n", lead);
    exit(2);
}

static void vocab_cases(void) {
    size_t n;
    unsigned char *c0 = fresh(&n);
    const uint32_t N = rd32(voc(c0));
    const uint32_t euro = piece_with(c0, "\xE2\x82\xAC"), acute = piece_with(c0, "\xE2\x96\x81\xC3\xA9");
    const uint32_t emoji = piece_with(c0, "\xE2\x96\x81\xF0\x9F");
    free(c0);
    CORRUPT("vocabulary count 0", wr32(voc(c), 0));
    CORRUPT("vocabulary count past the limit", wr32(voc(c), (1u << 20) + 1));
    CORRUPT("vocabulary count huge", wr32(voc(c), 0xFFFFFFFFu));
    CORRUPT("vocabulary count one more", wr32(voc(c), N + 1));
    CORRUPT("vocabulary count one fewer", wr32(voc(c), N - 1));
    CORRUPT("vocabulary count past its size", wr32(voc(c), (uint32_t)hdr(c)->vocab_size));
    CORRUPT("first offset 1", wr32(v_off(c, 0), 1));
    CORRUPT("offsets backward", wr32(v_off(c, 5), rd32(v_off(c, 6)) + 1));
    CORRUPT("last offset past the end", wr32(v_off(c, N), 0xFFFFFFF0u));
    CORRUPT("last offset one more", wr32(v_off(c, N), rd32(v_off(c, N)) + 1));
    CORRUPT("piece type 0", *v_type(c, 9) = 0);
    CORRUPT("piece type 6 (bytes)", *v_type(c, 9) = 6);
    CORRUPT("piece type 255", *v_type(c, 9) = 255);
    CORRUPT("a byte that is never UTF-8", v_text(c, euro)[1] = 0xFF);
    CORRUPT("an overlong form", memcpy(v_text(c, euro), "\xE0\x80\x80", 3));
    CORRUPT("a surrogate", memcpy(v_text(c, euro), "\xED\xA0\x80", 3));
    CORRUPT("a character cut short", v_text(c, acute)[4] = 'A');
    CORRUPT("past U+10FFFF", memcpy(v_text(c, emoji) + 3, "\xF4\x90\x80\x80", 4));
    CORRUPT("a NUL in a piece", v_text(c, 9)[0] = 0);
    CORRUPT("punctuation one byte more", wr32(v_np(c), rd32(v_np(c)) + 1));
    CORRUPT("punctuation one byte fewer", wr32(v_np(c), rd32(v_np(c)) - 1));
    CORRUPT("punctuation huge", wr32(v_np(c), 0xFFFFFFFFu));
    CORRUPT("punctuation not UTF-8", v_np(c)[4 + rd32(v_np(c)) - 1] = 0xFF);
    CORRUPT("a NUL in the punctuation", v_np(c)[4] = 0);
    {
        /* one piece fewer, the vocab param left at the old count */
        model e = g_m;
        size_t vn = g_m.vocab_n;
        unsigned char *v = malloc(vn);
        memcpy(v, g_m.vocab, vn);
        /* drop the last piece: count, its offset and its type and text */
        uint32_t cnt = rd32(v), text = rd32(v + 4 + 4 * cnt), last = rd32(v + 4 + 4 * (cnt - 1));
        size_t head = 4 + 4 * ((size_t)cnt + 1) + cnt;
        unsigned char *w = calloc(1, vn);
        size_t k = 0;
        wr32(w, cnt - 1);
        k = 4;
        memcpy(w + k, v + 4, 4 * (size_t)cnt);                    /* off[0..cnt-1] */
        k += 4 * (size_t)cnt;
        memcpy(w + k, v + 4 + 4 * ((size_t)cnt + 1), cnt - 1);    /* type[0..cnt-2] */
        k += cnt - 1;
        memcpy(w + k, v + head, last);                            /* text of pieces 0..cnt-2 */
        k += last;
        memcpy(w + k, v + head + text, vn - head - text);         /* np and punct */
        k += vn - head - text;
        e.vocab = w;
        e.vocab_n = k;
        expect_model("one piece fewer than the vocab param", &e, 0);
        free(v);
        free(w);
    }
}

static void meta_cases(void) {
    CORRUPT("metadata unterminated", c[hdr(c)->meta_off + hdr(c)->meta_size - 1] = 'x');
    CORRUPT("metadata empty key", c[hdr(c)->meta_off] = 0);
    CORRUPT("metadata not UTF-8", c[hdr(c)->meta_off + 2] = 0xC0);
    {
        static const char empty_key[] = "\0tiny\0";
        model e0 = g_m;
        e0.meta = (const unsigned char *)empty_key;
        e0.meta_n = sizeof empty_key - 1;
        expect_invalid("metadata whose one key is empty", &e0);
        static const char odd[] = "family\0tiny\0source\0";
        model e = g_m;
        e.meta = (const unsigned char *)odd;
        e.meta_n = sizeof odd - 1;
        expect_model("metadata key with no value", &e, 0);
        static const char empty_value[] = "family\0\0";
        e.meta = (const unsigned char *)empty_value;
        e.meta_n = sizeof empty_value - 1;
        expect_model("metadata with an empty value", &e, 1);
    }
}

/* a float tensor's element i set to v */
static void poke(unsigned char *c, int t, size_t i, float v) {
    memcpy(c + dsc(c, t)->offset + 4 * i, &v, 4);
}

static void engine_cases(const pk_model *m, int deep);

static void meaning_cases(void) {
    model e;
    static const char *drop[] = { "l1.att.k", "l0.ff1.up.scale", "mel.window", "vad.head.b",
                                  "dec.embed", "joint.out.b", "sub.out.w", "pos.freq" };
    for (size_t i = 0; i < sizeof drop / sizeof drop[0]; i++) {
        e = g_m;
        int k = find_tensor(&e, drop[i]);
        memmove(&e.t[k], &e.t[k + 1], (size_t)(e.nt - k - 1) * sizeof e.t[0]);
        e.nt--;
        char what[96];
        snprintf(what, sizeof what, "no %s", drop[i]);
        expect_model(what, &e, 0);
    }
    e = g_m;
    e.t[e.nt] = e.t[find_tensor(&g_m, "pos.freq")];
    strcpy(e.t[e.nt].name, "extra.w");
    e.nt++;
    expect_model("a tensor that is not the model's", &e, 0);
    e = g_m;
    e.t[e.nt] = e.t[find_tensor(&g_m, "pos.freq")];
    e.nt++;
    expect_invalid("a tensor twice, the copy added", &e);

    /* shapes and types, byte-valid */
    int conv0 = find_tensor(&g_m, "sub.conv0.w");
    CORRUPT("a shape transposed", (dsc(c, conv0)->dims[0] = g_m.t[conv0].d1,
                                   dsc(c, conv0)->dims[1] = g_m.t[conv0].d0));
    {
        int nw = find_tensor(&g_m, "l0.ff1.norm.w");
        e = g_m;
        e.t[nw].type = PK_Q8;            /* the same element count, as int8 */
        e.t[nw].bytes /= 4;
        expect_model("an int8 vector where a float one is", &e, 0);
        int qm = find_tensor(&g_m, "l0.att.q");
        e = g_m;
        unsigned char *wide = calloc(g_m.t[qm].bytes, 4);
        e.t[qm].type = PK_F32;
        e.t[qm].data = wide;
        e.t[qm].bytes *= 4;
        expect_model("a float matrix where an int8 one is", &e, 0);
        free(wide);
        int sc = find_tensor(&g_m, "l0.att.q.scale");
        e = g_m;
        e.t[sc].d0 /= 2;
        e.t[sc].bytes /= 2;
        expect_model("a scale for half the rows", &e, 0);
    }
    /* values that are not finite, and tables that are not what they say */
    static const struct { const char *name; size_t at; float v; } bad[] = {
        { "l0.ff1.norm.w", 3, NAN }, { "l1.out.norm.b", 63, INFINITY }, { "l0.att.q.scale", 5, NAN },
        { "mel.fb", 300, -INFINITY }, { "vad.lstm.wx", 1000, NAN }, { "dec.embed", 17, NAN },
        { "joint.out.b", 0, NAN }, { "pos.freq", 2, INFINITY }, { "sub.conv0.b", 1, NAN },
        { "l1.conv.dw", 8, NAN },
    };
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
        size_t n;
        unsigned char *c = fresh(&n);
        poke(c, find_tensor(&g_m, bad[i].name), bad[i].at, bad[i].v);
        char what[96];
        snprintf(what, sizeof what, "%g in %s", bad[i].v, bad[i].name);
        expect(what, c, n, 0);
        free(c);
    }
    static const char *tables[] = { "mel.fft.cos", "mel.fft.sin", "vad.fft.cos", "vad.fft.sin",
                                    "vad.stft.window" };
    for (size_t i = 0; i < sizeof tables / sizeof tables[0]; i++) {
        size_t n;
        unsigned char *c = fresh(&n);
        int t = find_tensor(&g_m, tables[i]);
        float v;
        memcpy(&v, c + dsc(c, t)->offset + 12, 4);
        poke(c, t, 3, v + 1e-3f);
        char what[96];
        snprintf(what, sizeof what, "a wrong value in %s", tables[i]);
        expect(what, c, n, 0);
        free(c);
    }
    CORRUPT("a finite position frequency above one",
            poke(c, find_tensor(&g_m, "pos.freq"), 0, 0x1.fffffep+127f));
    {
        /* Every stored float is finite, but hostile weights may overflow
         * intermediate arithmetic. They must not make a narrowing conversion
         * undefined or let its result escape an activation buffer. */
        size_t n;
        unsigned char *c = fresh(&n);
        poke(c, find_tensor(&g_m, "sub.conv0.w"), 0, 0x1.fffffep+127f);
        char err[256] = "";
        pk_model *m = pk_model_from_bytes(c, n, err, sizeof err);
        g_cases++;
        if (!m) {
            FAIL("a finite extreme weight: %s", err);
        } else {
            engine_cases(m, 0);
            pk_model_free(m);
        }
        free(c);
    }
}

/* ------------------------------------------------------------- the engine --- */

/* 1 if s[0..n) is well-formed UTF-8 */
static int utf8_ok(const unsigned char *s, size_t n) {
    for (size_t i = 0; i < n;) {
        unsigned c = s[i];
        size_t len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        if (!len || len > n - i)
            return 0;
        for (size_t j = 1; j < len; j++)
            if ((s[i + j] & 0xC0) != 0x80)
                return 0;
        i += len;
    }
    return 1;
}

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) {
    g_rng ^= g_rng >> 12;
    g_rng ^= g_rng << 25;
    g_rng ^= g_rng >> 27;
    return (uint32_t)((g_rng * 0x2545F4914F6CDD1Dull) >> 32);
}

/* a recording: a tone that comes and goes, and noise. The sine is
 * core/vmath.h's, not libm's, whose last bits differ between platforms: a
 * random model's transcript of this follows them, and with it the number
 * of cases the text checks make, which should be the same everywhere. */
static float *audio(size_t n) {
    float *x = malloc((n ? n : 1) * sizeof *x);
    for (size_t i = 0; i < n; i++) {
        float on = (i / 8000) % 3 ? 1.0f : 0.01f, sn, cs;
        pk_sincosf((float)i * 0.05f, &sn, &cs);
        x[i] = 0.3f * on * sn + 0.05f * ((float)(rnd() >> 8) / 16777216.0f - 0.5f);
    }
    return x;
}

/* the text of a run, whole and cut at every size up to it */
static void text_checks(const pk_model *m, const pk_tok *t, long nt, const char *what) {
    size_t len = pk_text(m, t, (size_t)nt, NULL, 0);
    char *full = malloc(len + 1);
    g_cases++;
    if (pk_text(m, t, (size_t)nt, full, len + 1) != len || strlen(full) != len
        || !utf8_ok((const unsigned char *)full, len)) {
        FAIL("%s: the text is not %zu bytes of UTF-8", what, len);
        free(full);
        return;
    }
    char *cut = malloc(len + 2);
    for (size_t cap = 1; cap <= len + 1; cap += 1 + cap / 8) {
        memset(cut, 0x7F, len + 2);
        pk_text(m, t, (size_t)nt, cut, cap);
        size_t k = strlen(cut);
        g_cases++;
        if (k >= cap || memcmp(cut, full, k) || !utf8_ok((const unsigned char *)cut, k)
            || (cap > len && k != len) || cut[cap] != 0x7F)
            FAIL("%s: the text cut to %zu bytes", what, cap);
    }
    free(cut);
    free(full);
}

static void run(pk_asr *a, const pk_model *m, const float *x, size_t n, const char *what) {
    const pk_tok *t;
    long nt = pk_asr_run(a, x, n, &t);
    g_cases++;
    if (nt < 0) {
        FAIL("%s: pk_asr_run failed", what);
        return;
    }
    for (long i = 0; i < nt; i++)
        if (t[i].id < 0 || t[i].id >= (int32_t)m->vocab || t[i].frame < 0 || t[i].dur < 0) {
            FAIL("%s: token %ld is %d at frame %d for %d", what, i, t[i].id, t[i].frame, t[i].dur);
            return;
        }
    text_checks(m, t, nt, what);
}

static void engine_cases(const pk_model *m, int deep) {
    static const size_t lengths[] = { 0, 1, 159, 160, 161, 319, 320, 1279, 1280, 1281, 16000,
                                      3 * 16000 + 17 };
    for (int threads = 1; threads <= (deep ? 3 : 1); threads += 2) {
        pk_asr *a = pk_asr_new(m, threads);
        g_cases++;
        if (!a) {
            FAIL("pk_asr_new with %d threads", threads);
            return;
        }
        for (size_t i = 0; i < sizeof lengths / sizeof lengths[0]; i++) {
            if (!deep && lengths[i] > 16000)
                continue;
            float *x = audio(lengths[i]);
            char what[64];
            snprintf(what, sizeof what, "%zu samples, %d threads", lengths[i], threads);
            run(a, m, x, lengths[i], what);
            free(x);
        }
        if (deep) {
            /* samples that are not finite: read as 0 */
            float *x = audio(8000);
            for (size_t i = 0; i < 8000; i += 97) x[i] = i % 2 ? NAN : -INFINITY;
            run(a, m, x, 8000, "NaN and infinite samples");
            free(x);
            /* finite and huge, as a float WAV may hold: the power spectrum
             * overflows to infinity and the log-mels to NaN, which the
             * subsampling's first ReLU makes 0 on every backend, so nothing
             * that is not finite reaches an int8 quantizer (UBSan, under
             * check-sanitize, would see the conversion) */
            x = audio(16000);
            for (size_t i = 0; i < 16000; i += 7) x[i] = i % 2 ? 1e30f : -3e38f;
            run(a, m, x, 16000, "samples of 1e30");
            free(x);
            /* longer, then shorter again, through the same transcriber */
            x = audio(5 * 16000);
            run(a, m, x, 5 * 16000, "five seconds");
            run(a, m, x, 1000, "a short one after a long one");
            free(x);
        }
        pk_asr_free(a);
    }
    if (!deep)
        return;
    size_t len;
    const char *p;
    p = pk_piece(m, -1, &len);
    g_cases++;
    if (len || !p) FAIL("pk_piece(-1)");
    p = pk_piece(m, (int32_t)m->vocab, &len);
    g_cases++;
    if (len || !p) FAIL("pk_piece(vocab)");
    p = pk_piece(m, 5, &len);
    g_cases++;
    if (!len || !utf8_ok((const unsigned char *)p, len)) FAIL("pk_piece(5)");

    /* every piece, alone and in runs, through the text */
    pk_tok t[64];
    for (uint32_t i = 0; i < m->vocab; i++) {
        t[0].id = (int32_t)i;
        t[0].frame = t[0].dur = 0;
        text_checks(m, t, 1, "a piece alone");
    }
    for (int rep = 0; rep < 200; rep++) {
        int k = 1 + (int)(rnd() % 64);
        for (int i = 0; i < k; i++) {
            t[i].id = (int32_t)(rnd() % (m->vocab + 2)) - 1;   /* and ids past both ends */
            t[i].frame = t[i].dur = 0;
        }
        text_checks(m, t, k, "random pieces");
    }

    /* the VAD over a recording with samples that are not finite or huge */
    size_t frames = 300;
    float *x = audio(frames * PK_VAD_SAMPLES), *probs = malloc(frames * sizeof *probs);
    for (size_t i = 0; i < frames * PK_VAD_SAMPLES; i += 1009)
        x[i] = i % 3 == 0 ? NAN : i % 3 == 1 ? 1e30f : -INFINITY;
    pk_vad *v = pk_vad_new(m);
    g_cases++;
    if (!v) {
        FAIL("pk_vad_new");
    } else {
        pk_vad_probs(v, x, frames, probs);
        for (size_t i = 0; i < frames; i++)
            if (!(probs[i] >= 0 && probs[i] <= 1)) {
                FAIL("VAD frame %zu: %g", i, probs[i]);
                break;
            }
        pk_vad_reset(v);
        float p1 = pk_vad_prob(v, x);
        g_cases++;
        if (memcmp(&p1, &probs[0], sizeof p1))
            FAIL("a frame alone is not the batch's first frame");
        pk_seg_params sp;
        pk_seg_defaults(&sp);
        sp.max_speech_s = 2.0;
        size_t cap = 2 * frames + 1;
        pk_seg *seg = malloc(cap * sizeof *seg), *ch = malloc((cap + 1) * sizeof *ch);
        size_t ns = pk_segments(probs, frames, 16000, (int64_t)(frames * PK_VAD_SAMPLES), &sp, seg, cap);
        size_t nc = pk_chunks(seg, ns, (int64_t)(frames * PK_VAD_SAMPLES), 3 * 16000, ch);
        for (size_t i = 0; i < nc; i++) {
            g_cases++;
            if (ch[i].start < 0 || ch[i].end > (int64_t)(frames * PK_VAD_SAMPLES) || ch[i].end <= ch[i].start
                || ch[i].end - ch[i].start > 3 * 16000 || (i && ch[i].start < ch[i - 1].end))
                FAIL("chunk %zu of %zu: %lld .. %lld", i, nc, (long long)ch[i].start, (long long)ch[i].end);
        }
        free(seg);
        free(ch);
    }
    pk_vad_free(v);
    free(x);
    free(probs);
}

/* -------------------------------------------------------------- flipped --- */

static void flip_cases(void) {
    size_t n;
    unsigned char *f = fresh(&n);
    const size_t span = (size_t)hdr(f)->data_off;
    uint64_t s = 0x243F6A8885A308D3ull;
    int loaded = 0, ran = 0;
    for (int it = 0; it < 20000; it++) {
        unsigned char *c = malloc(n);
        memcpy(c, f, n);
        for (int k = 0; k < 1 + it % 3; k++) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            c[(s >> 33) % span] ^= (unsigned char)(1u << ((s >> 29) & 7));
        }
        char e[256] = "";
        pk_model *m = pk_model_from_bytes(c, n, e, sizeof e);
        g_cases++;
        if (m) {
            /* a flip that loads hit something the binder lets vary - a
             * param within its range, text, a name's spelling - and the
             * engine must still run on it */
            loaded++;
            if (ran < 400 && it % 4 == 0) {
                ran++;
                engine_cases(m, 0);
            }
        } else if (!e[0]) {
            FAIL("flip %d: refused without a reason", it);
        }
        pk_model_free(m);
        free(c);
    }
    free(f);
    printf("pkfcheck: %d of 20000 flips loaded, the engine run on %d of them\n", loaded, ran);
}

static void write_wav(const char *path) {
    char err[256];
    pk_wav_out *o = pk_wav_create(path, 16000, 0, err, sizeof err);
    float *x = audio(35 * 16000);
    if (!o || pk_wav_write(o, x, 35 * 16000) || pk_wav_finish(o, err, sizeof err))
        FAIL("writing %s: %s", path, err);
    free(x);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: pkfcheck tests/data/tiny.pkf [WAV_OUT]\n");
        return 2;
    }
    FILE *fp = fopen(argv[1], "rb");
    if (!fp) {
        printf("pkfcheck: cannot open %s\n", argv[1]);
        return 2;
    }
    fseek(fp, 0, SEEK_END);
    g_n = (size_t)ftell(fp);
    fseek(fp, 0, SEEK_SET);
    g_file = malloc(g_n);
    if (fread(g_file, 1, g_n, fp) != g_n) {
        printf("pkfcheck: cannot read %s\n", argv[1]);
        return 2;
    }
    fclose(fp);
    char err[256];
    if (pk_pkf_validate(g_file, g_n, err, sizeof err)) {
        printf("pkfcheck: %s is not valid: %s\n", argv[1], err);
        return 1;
    }
    parse(g_file, &g_m);
    if (g_m.nt + 2 > MAXT || g_m.np + 2 > MAXP) {
        printf("pkfcheck: %s is bigger than this test's tables\n", argv[1]);
        return 2;
    }

    int at = 0, valid, trunc, corrupt, meaning, engine;
    valid_cases(argv[1]);
    valid = g_cases - at, at = g_cases;
    truncated_cases();
    trunc = g_cases - at, at = g_cases;
    header_cases();
    param_cases();
    descriptor_cases();
    vocab_cases();
    meta_cases();
    corrupt = g_cases - at, at = g_cases;
    meaning_cases();
    meaning = g_cases - at, at = g_cases;
    pk_model *m = pk_model_from_bytes(g_file, g_n, err, sizeof err);
    if (m) {
        engine_cases(m, 1);
        pk_model_free(m);
    }
    engine = g_cases - at, at = g_cases;
    flip_cases();
    if (argc > 2)
        write_wav(argv[2]);
    printf("pkfcheck: valid %d, truncated %d, corrupt %d, meaning %d, engine %d, flipped %d\n",
           valid, trunc, corrupt, meaning, engine, g_cases - at);

    free(g_file);
    if (g_bad) {
        printf("pkfcheck: %d of %d cases wrong\n", g_bad, g_cases);
        return 1;
    }
    printf("pkfcheck: %d cases, every one loaded, refused or run as it should be\n", g_cases);
    return 0;
}
