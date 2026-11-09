/* The model-file loader on a malformed corpus (docs/MEMORY_SAFETY.md), and
 * the engine end to end on the tiny model.
 *
 * Every file here is built in memory by a writer of this test's own,
 * written from core/pkf.h rather than from the converter, out of the tables
 * and tensors of tests/data/tiny.pkf (`pktito-tools tiny` writes it), or of
 * tests/data/tiny-diar.pkf, the diarizer's (`pktito-tools tiny --diarizer`),
 * whose cases below are its own where a recognizer's would not apply. The
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
 *              extreme weight whose arithmetic overflows; the vocabulary's count off;
 *              the slim file, loaded by a build with ANE=1 or QNN=1 and
 *              refused by one with neither, and every way it can be wrong;
 *              the chip its graph for the NPU is fetched for
 *   graph      in a build with QNN=1, the header of the NPU's compiled
 *              graph (pk_qnn_check): taken whole, cut at every length,
 *              and each field wrong in a way only its own check refuses
 *   flipped    random bits of everything before the tensor data changed,
 *              20,000 times over, and the engine run on what still loads
 *   engine     the tiny model through pk_asr_run, pk_text, the VAD and
 *              pk_chunks: every length round a frame's edges, samples that
 *              are not finite, the text cut at every size; the diarizer
 *              through pk_diar_run and pk_diar_feed: every length round a
 *              row's and a chunk's edges, in pieces of every size and on
 *              one thread and three, the same bits each way
 *
 * and every file must load or be refused with a message - never crash,
 * never hand back a model that should have been refused. `make
 * check-sanitize` runs it under ASan and UBSan.
 *
 *     ./tests/pkfcheck tests/data/tiny.pkf [WAV_OUT]
 *
 * WAV_OUT, if given, gets the 35 s recording the engine checks used, for
 * `make check-c` to put through ./pktito-c as well.
 */
#include "model.h"
#include "pkf.h"
#include "vmath.h"
#include "pktito.h"
#include "wav.h"
#ifdef PK_QNN
#include "qnn.h"
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_bad, g_cases;
static int g_diar;                /* the tiny model is the diarizer's */

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
    if (strcmp(pk_model_name(m), g_diar ? "pktito-tiny-diar" : "pktito-tiny") || !pk_model_meta(m, "family")
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
    CORRUPT("the other model's architecture", (memset(hdr(c)->arch, 0, sizeof hdr(c)->arch),
                                               strcpy(hdr(c)->arch, g_diar ? "parakeet-tdt" : "sortformer")));
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
    if (!g_diar)
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

/* a recognizer's params alone */
static void tdt_param_cases(void) {
    CORRUPT("n_mels not a multiple of 8", prm(c, find_param(&g_m, "n_mels"))->v.u = 12);
    CORRUPT("an even convolution", prm(c, find_param(&g_m, "conv_kernel"))->v.u = 8);
    CORRUPT("a convolution past 63", prm(c, find_param(&g_m, "conv_kernel"))->v.u = 65);
    CORRUPT("norm_eps 0", prm(c, find_param(&g_m, "norm_eps"))->v.f = 0.0f);
    CORRUPT("norm_eps subnormal", prm(c, find_param(&g_m, "norm_eps"))->v.f = 1e-40f);
    CORRUPT("vocab 0", prm(c, find_param(&g_m, "vocab"))->v.u = 0);
    CORRUPT("vocab one fewer than the pieces", prm(c, find_param(&g_m, "vocab"))->v.u--);
    CORRUPT("n_durations 0", prm(c, find_param(&g_m, "n_durations"))->v.u = 0);
    CORRUPT("n_durations 17", prm(c, find_param(&g_m, "n_durations"))->v.u = 17);
    CORRUPT("n_durations one fewer", prm(c, find_param(&g_m, "n_durations"))->v.u--);
    CORRUPT("a duration of 2000", prm(c, find_param(&g_m, "dur0"))->v.u = 2000);
    CORRUPT("max_symbols 0", prm(c, find_param(&g_m, "max_symbols"))->v.u = 0);
    CORRUPT("pred_layers one more", prm(c, find_param(&g_m, "pred_layers"))->v.u++);
    model e = g_m;
    e.p[e.np] = e.p[find_param(&g_m, "dur4")];
    strcpy(e.p[e.np].key, "dur5");
    e.np++;
    expect_model("a duration past n_durations", &e, 0);
}

/* a diarizer's params alone: the stacking, the upsampling and the speaker
 * cache, each out of its range or out of step with the tensors */
static void diar_param_cases(void) {
#define DP(what, key, stmt) CORRUPT(what, (void)(prm(c, find_param(&g_m, key))->stmt))
    DP("stack 0", "stack", v.u = 0);
    DP("a stack whose input is not whole blocks", "stack", v.u = 3);
    DP("a stack of 65", "stack", v.u = 65);
    DP("upsample 0", "upsample", v.u = 0);
    DP("an upsampling that does not divide a row's samples", "upsample", v.u = 3);
    DP("an upsampling the tensors do not have", "upsample", v.u = 4);
    DP("proj_dim not a multiple of 32", "proj_dim", v.u = 48);
    DP("no speakers", "n_spk", v.u = 0);
    DP("17 speakers", "n_spk", v.u = 17);
    DP("chunk_len 0", "chunk_len", v.u = 0);
    DP("chunk_len 2049", "chunk_len", v.u = 2049);
    DP("chunk_left 513", "chunk_left", v.u = 513);
    DP("chunk_right 513", "chunk_right", v.u = 513);
    DP("fifo_len 2049", "fifo_len", v.u = 2049);
    DP("cache_len 0", "cache_len", v.u = 0);
    DP("a cache too small for each speaker's silence frames", "cache_len", v.u = 9);
    DP("update_period 0", "update_period", v.u = 0);
    DP("sil_frames 65", "sil_frames", v.u = 65);
    DP("rope_theta 1", "rope_theta", v.f = 1.0f);
    DP("rope_theta NaN", "rope_theta", v.f = NAN);
    DP("a rope_theta the frequencies are not", "rope_theta", v.f = 500.0f);
    DP("score_threshold 0", "score_threshold", v.f = 0.0f);
    DP("score_threshold 2", "score_threshold", v.f = 2.0f);
    DP("boost_latest negative", "boost_latest", v.f = -0.05f);
    DP("strong_boost infinite", "strong_boost", v.f = INFINITY);
    DP("weak_boost NaN", "weak_boost", v.f = NAN);
    DP("min_pos_scores 1.5", "min_pos_scores", v.f = 1.5f);
    DP("an int where a float is", "score_threshold", kind = PK_PARAM_U32);
    DP("a float where an int is", "cache_len", kind = PK_PARAM_F32);
#undef DP
    model e = g_m;
    e.p[e.np] = e.p[find_param(&g_m, "n_layers")];
    strcpy(e.p[e.np].key, "vocab");
    e.np++;
    expect_model("a recognizer's param", &e, 0);
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
    CORRUPT("d_model not a multiple of 32", prm(c, find_param(&g_m, "d_model"))->v.u = 48);
    CORRUPT("heads that do not divide d_model", prm(c, find_param(&g_m, "n_heads"))->v.u = 3);
    CORRUPT("heads narrower than a panel", prm(c, find_param(&g_m, "n_heads"))->v.u = 8);
    CORRUPT("preemph NaN", prm(c, find_param(&g_m, "preemph"))->v.f = NAN);
    CORRUPT("preemph 2", prm(c, find_param(&g_m, "preemph"))->v.f = 2.0f);
    CORRUPT("ln_eps negative", prm(c, find_param(&g_m, "ln_eps"))->v.f = -1e-5f);
    CORRUPT("ln_eps 0", prm(c, find_param(&g_m, "ln_eps"))->v.f = 0.0f);
    CORRUPT("log_guard 0", prm(c, find_param(&g_m, "log_guard"))->v.f = 0.0f);
    CORRUPT("ln_eps infinite", prm(c, find_param(&g_m, "ln_eps"))->v.f = INFINITY);
    CORRUPT("n_layers 0", prm(c, find_param(&g_m, "n_layers"))->v.u = 0);
    CORRUPT("n_layers one more than the tensors", prm(c, find_param(&g_m, "n_layers"))->v.u++);
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
    }
    if (g_diar)
        diar_param_cases();
    else
        tdt_param_cases();
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
    CORRUPT("a vector of two dims", dsc(c, find_tensor(&g_m, g_diar ? "rope.freq" : "pos.freq"))->dims[1] = 2);
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
static void diar_engine_cases(const pk_model *m, int deep);

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

/* The diarizer's file: a tensor missing, extra, of another shape or type,
 * a value that is not finite or a table that is not what it says; a
 * vocabulary, which a diarizer's file has none of; and weights whose
 * arithmetic overflows, which must still run */
static void diar_meaning_cases(void) {
    model e;
    static const char *drop[] = { "l1.att.k", "l0.ff.up.scale", "mel.window", "rope.freq", "stack",
                                  "embed_norm.w", "final_norm.b", "up.b", "head.out", "sil" };
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
    e.t[e.nt] = e.t[find_tensor(&g_m, "sil")];
    strcpy(e.t[e.nt].name, "extra.w");
    e.nt++;
    expect_model("a tensor that is not the model's", &e, 0);
    e = g_m;
    e.t[e.nt] = e.t[find_tensor(&g_m, "sil")];
    e.nt++;
    expect_invalid("a tensor twice, the copy added", &e);
    {
        /* a recognizer's vocabulary, which a diarizer has no use for */
        static const unsigned char vocab[] = {
            2, 0, 0, 0,  0, 0, 0, 0,  4, 0, 0, 0,  5, 0, 0, 0,  1, 1,
            0xE2, 0x96, 0x81, 'a', 'b',  1, 0, 0, 0, '.' };
        e = g_m;
        e.vocab = vocab;
        e.vocab_n = sizeof vocab;
        expect_model("a diarizer's file with a vocabulary", &e, 0);
    }

    int up = find_tensor(&g_m, "up");
    CORRUPT("a shape transposed", (dsc(c, up)->dims[0] = g_m.t[up].d1, dsc(c, up)->dims[1] = g_m.t[up].d0));
    {
        int nw = find_tensor(&g_m, "l0.att.norm.w");
        e = g_m;
        e.t[nw].type = PK_Q8;
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
    static const struct { const char *name; size_t at; float v; } bad[] = {
        { "l0.att.norm.w", 3, NAN }, { "l1.ff.norm.b", 63, INFINITY }, { "l0.att.q.scale", 5, NAN },
        { "mel.fb", 300, -INFINITY }, { "rope.freq", 2, INFINITY }, { "stack.scale", 1, NAN },
        { "up.b", 9, NAN }, { "head.out.b", 0, NAN }, { "sil", 7, NAN }, { "l1.ff.down.b", 2, -INFINITY },
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
    static const char *tables[] = { "mel.fft.cos", "mel.fft.sin" };
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
    {
        /* the rotary frequencies: each within 2 ulp of the float that
         * theta^(-i/n) rounds to (core/model.c, check_rope), in (0, 1] */
        int rf = find_tensor(&g_m, "rope.freq");
        const float theta = g_m.p[find_param(&g_m, "rope_theta")].v.f;
        const float want = (float)pow((double)theta, -3.0 / g_m.t[rf].d0);
        const float up2 = nextafterf(nextafterf(want, 1), 1), up3 = nextafterf(up2, 1);
        CORRUPT("a rotary frequency 3 ulp off", poke(c, rf, 3, up3));
        CORRUPT("a rotary frequency above one", poke(c, rf, 0, 1.5f));
        CORRUPT("a rotary frequency of zero", poke(c, rf, 5, 0.0f));
        CORRUPT("a negative rotary frequency", poke(c, rf, 1, -want));
        size_t n;
        unsigned char *c = fresh(&n);
        poke(c, rf, 3, up2);
        expect("a rotary frequency 2 ulp off", c, n, 1);
        free(c);
    }
    /* finite and extreme: the arithmetic overflows to infinity and NaN,
     * which quantizing makes zeros and the cut's scores take as silence;
     * the engine must run, the probabilities in [0, 1] or NaN */
    static const struct { const char *name; size_t at; } extreme[] = {
        { "stack.scale", 0 }, { "l0.ff.norm.w", 1 }, { "sil", 0 }, { "up.b", 3 }, { "head.out.b", 1 },
    };
    for (size_t i = 0; i < sizeof extreme / sizeof extreme[0]; i++) {
        size_t n;
        unsigned char *c = fresh(&n);
        poke(c, find_tensor(&g_m, extreme[i].name), extreme[i].at, 0x1.fffffep+127f);
        char err[256] = "";
        pk_model *m = pk_model_from_bytes(c, n, err, sizeof err);
        g_cases++;
        if (!m) {
            FAIL("a finite extreme value in %s: %s", extreme[i].name, err);
        } else {
            diar_engine_cases(m, 0);
            pk_model_free(m);
        }
        free(c);
    }
}

/* 1 if name is an encoder's tensor: the subsampling's, or a layer's */
static int encoder_tensor(const char *name) {
    const size_t head = strcspn(name, ".");
    return (head == 3 && !strncmp(name, "sub", 3))
        || (head > 1 && name[0] == 'l' && strspn(name + 1, "0123456789") == head - 1);
}

/* The slim file (pktito-tools slim): the tiny model without its encoder's
 * tensors, and the param encoder_coreml 1. A build with ANE=1 or QNN=1
 * loads it, and with no accelerator's encoder given runs nothing:
 * pk_asr_run returns -1; with QNN=1 the NPU asked for with no compiled
 * graph's file is refused before QNN's libraries are looked for. A build
 * with neither refuses the file, saying it is a slim one. Every build
 * refuses the param with another value or kind, and a slim file that
 * holds an encoder's tensor or lacks another of the model's. */
static void slim_cases(void) {
    model s = g_m;
    s.nt = 0;
    for (int i = 0; i < g_m.nt; i++)
        if (!encoder_tensor(g_m.t[i].name))
            s.t[s.nt++] = g_m.t[i];
    memset(&s.p[s.np], 0, sizeof s.p[0]);
    strcpy(s.p[s.np].key, "encoder_coreml");
    s.p[s.np].kind = PK_PARAM_U32;
    s.p[s.np].v.u = 1;
    s.np++;
#if defined(PK_ANE) || defined(PK_QNN)
    {
        size_t n;
        unsigned char *p = build(&s, &n);
        char err[256] = "";
        pk_model *m = pk_model_from_bytes(p, n, err, sizeof err);
        g_cases++;
        if (!m) {
            FAIL("a slim file: %s", err);
        } else {
            static float x[16000];
            for (size_t i = 0; i < sizeof x / sizeof x[0]; i++)
                x[i] = 0.1f * sinf(0.05f * (float)i);
            pk_asr *a = pk_asr_new(m, 2);
            const pk_tok *t;
            g_cases++;
            if (pk_model_has_encoder(m))
                FAIL("a slim file: an encoder of its own");
            else if (!a || pk_asr_run(a, x, sizeof x / sizeof x[0], &t) != -1)
                FAIL("a slim file with no accelerator's encoder: pk_asr_run did not refuse");
#ifdef PK_QNN
            g_cases++;
            if (a && pk_asr_use_qnn(a, 1, NULL, err, sizeof err) != -1)
                FAIL("a slim file, the NPU with no compiled graph's file: not refused");
#endif
            pk_asr_free(a);
            pk_model_free(m);
        }
        free(p);
        pk_model *full = pk_model_from_bytes(g_file, g_n, err, sizeof err);
        g_cases++;
        if (!full || !pk_model_has_encoder(full))
            FAIL("the tiny model: no encoder of its own");
        pk_model_free(full);
    }
#else
    expect_model("a slim file, in a build without ANE=1 or QNN=1", &s, 0);
    {
        size_t n;
        unsigned char *p = build(&s, &n);
        char err[256] = "";
        pk_model *m = pk_model_from_bytes(p, n, err, sizeof err);
        g_cases++;
        if (m || !strstr(err, "slim model file"))
            FAIL("a slim file, in a build without ANE=1 or QNN=1: refused as %s, not as a slim file",
                 m ? "nothing" : err);
        pk_model_free(m);
        free(p);
    }
#endif
    /* the chip a slim file's graph is fetched for (pk_qnn_chip): none in a
     * build without QNN=1, nor where QNN finds no NPU, as on a PC, each
     * with why; on a phone PKTITO_QNN_TARGET's five numbers */
    {
        char chip[64] = "", why[256] = "";
        unsigned v[5];
        g_cases++;
        if (pk_qnn_chip(chip, sizeof chip, why, sizeof why) == 0
                ? sscanf(chip, "%u,%u,%u,%u,%u", &v[0], &v[1], &v[2], &v[3], &v[4]) != 5
                : !why[0])
            FAIL("pk_qnn_chip: neither a chip nor why (\"%s\")", chip);
    }
    model e = s;
    e.p[e.np - 1].v.u = 2;
    expect_model("encoder_coreml 2", &e, 0);
    e = s;
    e.p[e.np - 1].kind = PK_PARAM_F32;
    e.p[e.np - 1].v.f = 1;
    expect_model("encoder_coreml a float", &e, 0);
    static const char *keep[] = { "l1.att.k", "l1.att.k.scale", "sub.conv0.b" };
    for (size_t i = 0; i < sizeof keep / sizeof keep[0]; i++) {
        e = s;
        e.t[e.nt++] = g_m.t[find_tensor(&g_m, keep[i])];
        char what[96];
        snprintf(what, sizeof what, "a slim file with %s", keep[i]);
        expect_model(what, &e, 0);
    }
    static const char *drop[] = { "dec.embed", "joint.out.w", "pos.freq", "mel.fb", "vad.lstm.b" };
    for (size_t i = 0; i < sizeof drop / sizeof drop[0]; i++) {
        e = s;
        int k = find_tensor(&e, drop[i]);
        memmove(&e.t[k], &e.t[k + 1], (size_t)(e.nt - k - 1) * sizeof e.t[0]);
        e.nt--;
        char what[96];
        snprintf(what, sizeof what, "a slim file without %s", drop[i]);
        expect_model(what, &e, 0);
    }
}

/* -------------------------------------------------------- the NPU's graph --- */

#ifdef PK_QNN
/* A compiled graph's file (docs/MEMORY_SAFETY.md, "The NPU's graph
 * file"): its header as that describes it, 616 bytes, little-endian, then
 * QNN's binary, which pk_qnn_check never reads, so a few bytes stand for
 * it */
typedef struct {
    char magic[32], key[176], chip[32], made[64], built[64], names[6][16];
    uint32_t id[6], rank[6], dims[6][4];
    uint64_t bytes;
} graph_head;
_Static_assert(sizeof(graph_head) == 616, "the graph file's header");

#define GRAPH_BIN 40

/* the header of m's graph for T0 frames, compiled for chip: its key the
 * model's checkpoint and converter from its metadata, else digest, its
 * data's SHA-256 */
static void graph_write(graph_head *h, const pk_model *m, const uint8_t digest[32], uint32_t T0,
                        const char *chip) {
    const uint32_t T1 = (T0 + 1) / 2, T2 = (T1 + 1) / 2, T3 = (T2 + 1) / 2;
    const uint32_t shape[6][4] = { { 1, T0, m->n_mels, 1 }, { 1, T1, 1, 1 }, { 1, T2, 1, 1 },
                                   { T3, 1 },               { 1, 1, T3 },    { T3, m->d_model } };
    static const uint32_t rank[6] = { 4, 4, 4, 2, 3, 2 };
    memset(h, 0, sizeof *h);
    strcpy(h->magic, "pktito-qnn-graph 2");
    const char *src = pk_model_meta(m, "source_sha256"), *conv = pk_model_meta(m, "converter");
    if (src && conv) {
        snprintf(h->key, sizeof h->key, "%s %s %u %u", src, conv, T0, m->n_layers);
    } else {
        char hex[65];
        for (int i = 0; i < 32; i++)
            snprintf(hex + 2 * i, 3, "%02x", digest[i]);
        snprintf(h->key, sizeof h->key, "%s %u %u", hex, T0, m->n_layers);
    }
    snprintf(h->chip, sizeof h->chip, "%s", chip);
    strcpy(h->made, "pktito 0.1.0 pkfcheck");
    strcpy(h->built, "v2.51.0.260929181845");
    for (int k = 0; k < 6; k++) {
        snprintf(h->names[k], sizeof h->names[k], "t%d", k < 5 ? k : 5522);
        h->id[k] = k < 5 ? (uint32_t)k + 1 : 3398;
        h->rank[k] = rank[k];
        memcpy(h->dims[k], shape[k], sizeof shape[k]);
    }
    h->bytes = GRAPH_BIN;
}

/* pk_qnn_check on the first n bytes of h, in a file of size bytes: taken
 * (want 1) or refused with a reason (want 0). The n bytes are a buffer of
 * their own, so that a check that read past them is ASan's to see */
static void expect_graph(const char *what, const pk_model *m, uint32_t frames, const char *chip,
                         const graph_head *h, size_t n, uint64_t size, int want) {
    unsigned char *p = malloc(n ? n : 1);
    memcpy(p, h, n);
    char err[256] = "";
    g_cases++;
    const int taken = pk_qnn_check(m, frames, chip, p, n, size, err, sizeof err) == 0;
    if (taken != want)
        FAIL("%s: %s", what, taken ? "taken" : err);
    else if (!taken && !err[0])
        FAIL("%s: refused without a reason", what);
    free(p);
}

/* The checks before QNN sees a graph's binary, on the tiny model's graph:
 * the header taken whole; cut at every length; and each field wrong, one
 * at a time, in a way only its own check refuses - the magic and the key
 * a bit off anywhere, the strings unterminated or with a byte after their
 * end, another Hexagon's chip, a name empty or another's, an id another's,
 * each rank and dim off, and the binary's size against what follows the
 * header, and past 4 GiB with the file to match */
static void graph_cases(void) {
    char err[256], what[96];
    pk_model *m = pk_model_from_bytes(g_file, g_n, err, sizeof err);
    g_cases++;
    if (!m) {
        FAIL("the tiny model, for its graph: %s", err);
        return;
    }
    static const char chip[] = "57,75,8,1,0";
    const uint64_t size = sizeof(graph_head) + GRAPH_BIN;
    graph_head h0, h;
    graph_write(&h0, m, g_m.digest, 3072, chip);
#define GRAPH(why, on, stmt) do { h = h0; stmt; expect_graph(why, m, 0, on, &h, sizeof h, size, 0); } while (0)

    expect_graph("a graph", m, 0, chip, &h0, sizeof h0, size, 1);
    expect_graph("a graph, where QNN names no chip", m, 0, NULL, &h0, sizeof h0, size, 1);
    expect_graph("a graph, on another SoC of its Hexagon", m, 0, "99,75,8,1,0", &h0, sizeof h0, size, 1);
    h = h0;
    memset(h.chip, 0, sizeof h.chip);
    expect_graph("a graph that names no chip", m, 0, chip, &h, sizeof h, size, 1);
    GRAPH("a graph of another Hexagon's", "69,79,8,1,0", (void)0);

    for (size_t n = 0; n < sizeof h0; n++)
        expect_graph("a header cut short", m, 0, chip, &h0, n, n, 0);
    expect_graph("a header cut short of the file", m, 0, chip, &h0, sizeof h0 - 1, size, 0);
    expect_graph("a header longer than its file", m, 0, chip, &h0, sizeof h0, sizeof h0 - 1, 0);
    expect_graph("a binary a byte short", m, 0, chip, &h0, sizeof h0, size - 1, 0);
    expect_graph("a binary a byte long", m, 0, chip, &h0, sizeof h0, size + 1, 0);

    for (size_t i = 0; i < sizeof h0.magic; i++)
        GRAPH("the magic a bit off", chip, h.magic[i] ^= 1);
    GRAPH("graph 1's magic", chip, h.magic[17] = '1');
    for (size_t i = 0; i < sizeof h0.key; i++)
        GRAPH("the key a bit off", chip, h.key[i] ^= 1);
    graph_write(&h, m, g_m.digest, 3000, chip);
    expect_graph("a graph for 3000 frames, read for 3072", m, 0, chip, &h, sizeof h, size, 0);
    expect_graph("a graph for 3000 frames, read for 3000", m, 3000, chip, &h, sizeof h, size, 1);

    GRAPH("the chip unterminated", NULL, memset(h.chip, '7', sizeof h.chip));
    GRAPH("the chip with a byte after it", NULL, h.chip[sizeof h.chip - 1] = '7');
    GRAPH("made unterminated", chip, memset(h.made, 'x', sizeof h.made));
    GRAPH("made with a byte after it", chip, h.made[sizeof h.made - 1] = 'x');
    GRAPH("built unterminated", chip, memset(h.built, 'x', sizeof h.built));
    GRAPH("built with a byte after it", chip, h.built[sizeof h.built - 1] = 'x');

    for (int k = 0; k < 6; k++) {
        snprintf(what, sizeof what, "input or output %d's name empty", k);
        GRAPH(what, chip, memset(h.names[k], 0, sizeof h.names[k]));
        snprintf(what, sizeof what, "input or output %d's name unterminated", k);
        GRAPH(what, chip, memset(h.names[k], 'n', sizeof h.names[k]));
        snprintf(what, sizeof what, "input or output %d's name with a byte after it", k);
        GRAPH(what, chip, h.names[k][sizeof h.names[k] - 1] = 'n');
        for (int j = 0; j < k; j++) {
            snprintf(what, sizeof what, "input or output %d with %d's name", k, j);
            GRAPH(what, chip, memcpy(h.names[k], h.names[j], sizeof h.names[k]));
            snprintf(what, sizeof what, "input or output %d with %d's id", k, j);
            GRAPH(what, chip, h.id[k] = h.id[j]);
        }
        const uint32_t r = h0.rank[k], ranks[] = { 0, r - 1, r + 1, 5, UINT32_MAX };
        for (size_t i = 0; i < sizeof ranks / sizeof ranks[0]; i++) {
            snprintf(what, sizeof what, "input or output %d of rank %u", k, ranks[i]);
            GRAPH(what, chip, h.rank[k] = ranks[i]);
        }
        for (int j = 0; j < 4; j++) {
            const uint32_t v = h0.dims[k][j], vs[] = { v + 1, v ? v - 1 : 2, UINT32_MAX };
            for (size_t i = 0; i < sizeof vs / sizeof vs[0]; i++) {
                snprintf(what, sizeof what, "input or output %d's dim %d %u, not %u", k, j, vs[i], v);
                GRAPH(what, chip, h.dims[k][j] = vs[i]);
            }
        }
    }

    GRAPH("a binary of no bytes", chip, h.bytes = 0);
    GRAPH("a binary a byte shorter than the file's", chip, h.bytes = GRAPH_BIN - 1);
    GRAPH("a binary a byte longer than the file's", chip, h.bytes = GRAPH_BIN + 1);
    GRAPH("a binary of 2^64 - 1 bytes", chip, h.bytes = UINT64_MAX);
    h = h0;
    h.bytes = 0;
    expect_graph("a header alone that says so", m, 0, chip, &h, sizeof h, sizeof h, 0);
    h.bytes = (uint64_t)1 << 32;
    expect_graph("a binary of 4 GiB, the file to match", m, 0, chip, &h, sizeof h, sizeof h + h.bytes, 0);
    h.bytes = ((uint64_t)1 << 32) - 1;
    expect_graph("a binary of 4 GiB less a byte, the file to match", m, 0, chip, &h, sizeof h,
                 sizeof h + h.bytes, 1);

    /* a model keyed by its checkpoint and converter, as the converter
     * writes it: its graph taken, and one keyed by its data refused */
    static const char named[] = "source_sha256\0"
                                "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff\0"
                                "converter\0pktito-tools 0.1.0 pkfcheck\0";
    unsigned char *meta = malloc(g_m.meta_n + sizeof named - 1);
    memcpy(meta, g_m.meta, g_m.meta_n);
    memcpy(meta + g_m.meta_n, named, sizeof named - 1);
    model e = g_m;
    e.meta = meta;
    e.meta_n = g_m.meta_n + sizeof named - 1;
    size_t n;
    unsigned char *p = build(&e, &n);
    pk_model *mk = pk_model_from_bytes(p, n, err, sizeof err);
    g_cases++;
    if (!mk || !pk_model_meta(mk, "converter")) {
        FAIL("the tiny model with its checkpoint and converter: %s", mk ? "no converter" : err);
    } else {
        graph_write(&h, mk, g_m.digest, 3072, chip);
        expect_graph("a graph keyed by its model's checkpoint", mk, 0, chip, &h, sizeof h, size, 1);
        expect_graph("a graph keyed by the data, of a model with a checkpoint", mk, 0, chip, &h0, sizeof h0,
                     size, 0);
    }
    pk_model_free(mk);
    free(p);
    free(meta);
    pk_model_free(m);
#undef GRAPH
}
#endif

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
        pk_seg *seg = malloc(cap * sizeof *seg);
        size_t ns = pk_segments(probs, frames, 16000, (int64_t)(frames * PK_VAD_SAMPLES), &sp, seg, cap);
        pk_seg *ch = malloc((2 * ns + 2 + frames * PK_VAD_SAMPLES / (3 * 16000)) * sizeof *ch);
        size_t nc = pk_chunks(x, seg, ns, (int64_t)(frames * PK_VAD_SAMPLES), 3 * 16000, ch);
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

/* ------------------------------------------------------ the diarizer --- */

typedef struct {
    int64_t next;                 /* the row that should come next */
    size_t S, cap;
    float *keep;                  /* the rows, [cap][S], or NULL */
    int strict;                   /* every probability in [0, 1]; else NaN may be too */
    int bad;                      /* 1 out of order, 2 a probability outside, 4 past cap */
    int64_t stop;                 /* stop once this many rows have come, or -1 */
} diar_sink;

static int diar_rows(void *arg, int64_t row, size_t n, const float *p) {
    diar_sink *k = arg;
    if (row != k->next || !n)
        k->bad |= 1;
    for (size_t i = 0; i < n * k->S; i++)
        if (!(p[i] >= 0 && p[i] <= 1) && (k->strict || p[i] == p[i]))
            k->bad |= 2;
    if (k->keep && row >= 0 && (size_t)row + n <= k->cap)
        memcpy(k->keep + (size_t)row * k->S, p, n * k->S * sizeof *p);
    else if (k->keep)
        k->bad |= 4;
    k->next = row + (int64_t)n;
    return k->stop >= 0 && k->next >= k->stop;
}

/* the rows n samples make: a row every stack / upsample feature frames,
 * a part of one counting */
static int64_t diar_nrows(const pk_model *m, size_t n) {
    const int64_t T0 = (int64_t)(n / m->hop);
    return (T0 * m->dw.upsample + m->dw.stack - 1) / m->dw.stack;
}

/* x[0..n) through d, whole (piece 0) or fed in pieces, its rows into keep
 * (if not NULL): 0, or 1 having said what went wrong */
static int diar_run(pk_diar *d, const pk_model *m, const float *x, size_t n, size_t piece, int strict,
                    float *keep, const char *what) {
    diar_sink k = { 0, (size_t)pk_model_speakers(m), (size_t)diar_nrows(m, n), keep, strict, 0, -1 };
    int rc = 0;
    if (!piece)
        rc = pk_diar_run(d, x, n, diar_rows, &k);
    else {
        for (size_t i = 0; i < n && !rc; i += piece)
            rc = pk_diar_feed(d, x + i, n - i < piece ? n - i : piece, diar_rows, &k);
        if (!rc)
            rc = pk_diar_finish(d, diar_rows, &k);
    }
    g_cases++;
    if (rc || k.bad || k.next != diar_nrows(m, n)) {
        FAIL("%s: returned %d, %lld rows of %lld%s", what, rc, (long long)k.next, (long long)diar_nrows(m, n),
             k.bad & 1 ? ", out of order" : k.bad & 2 ? ", a probability outside [0, 1]"
             : k.bad & 4 ? ", past the end" : "");
        return 1;
    }
    return 0;
}

static void same_rows(const float *a, const float *b, size_t n, const char *what) {
    g_cases++;
    if (memcmp(a, b, n * sizeof *a))
        FAIL("%s: not the same bits", what);
}

static void diar_engine_cases(const pk_model *m, int deep) {
    const size_t S = (size_t)pk_model_speakers(m);
    if (!deep) {
        /* a model a flip or an extreme value made: it must run */
        static const size_t lengths[] = { 0, 161, 2 * 16000 };
        pk_diar *d = pk_diar_new(m, 1);
        g_cases++;
        if (!d) {
            FAIL("pk_diar_new");
            return;
        }
        for (size_t i = 0; i < sizeof lengths / sizeof lengths[0]; i++) {
            float *x = audio(lengths[i]);
            char what[64];
            snprintf(what, sizeof what, "%zu samples", lengths[i]);
            diar_run(d, m, x, lengths[i], 0, 0, NULL, what);
            free(x);
        }
        pk_diar_free(d);
        return;
    }
    g_cases += 4;
    if (pk_diar_new(NULL, 1) || pk_model_speakers(NULL))
        FAIL("a diarizer of no model");
    pk_asr *a = pk_asr_new(m, 1);
    if (a)
        FAIL("a transcriber of the diarizer's model");
    pk_asr_free(a);
    if (pk_model_speakers(m) != (int)m->dw.n_spk)
        FAIL("pk_model_speakers: %d", pk_model_speakers(m));
    if (pk_model_frame_seconds(m) != (double)m->dw.stack * m->hop / ((double)m->dw.upsample * m->sample_rate))
        FAIL("pk_model_frame_seconds: %g", pk_model_frame_seconds(m));

    pk_diar *d1 = pk_diar_new(m, 1), *d3 = pk_diar_new(m, 3);
    g_cases++;
    if (!d1 || !d3) {
        FAIL("pk_diar_new");
        pk_diar_free(d1);
        pk_diar_free(d3);
        return;
    }
    /* every length round a row's, a chunk's and a chunk's context's edges:
     * the same bits on one thread and three, and fed in pieces of 1, 7,
     * 1000 and 4096 samples */
    const size_t chunk = (size_t)m->dw.chunk_len * m->dw.stack * m->hop;
    const size_t ctx = (size_t)m->dw.chunk_right * m->dw.stack * m->hop;
    const size_t lengths[] = { 0, 1, 159, 160, 161, 319, 320, 321, 639, 640, 641, chunk - 1, chunk,
                               chunk + 1, chunk + ctx - 1, chunk + ctx, chunk + ctx + 1, 16000, 2 * 16000,
                               6 * 16000 + 17 };
    static const size_t pieces[] = { 1, 7, 1000, 4096 };
    for (size_t i = 0; i < sizeof lengths / sizeof lengths[0]; i++) {
        const size_t n = lengths[i], rows = (size_t)diar_nrows(m, n) * S;
        float *x = audio(n), *r1 = malloc((rows ? rows : 1) * sizeof *r1), *r = malloc((rows ? rows : 1) * sizeof *r);
        char what[96];
        snprintf(what, sizeof what, "%zu samples", n);
        if (!diar_run(d1, m, x, n, 0, 1, r1, what)) {
            snprintf(what, sizeof what, "%zu samples, three threads", n);
            if (!diar_run(d3, m, x, n, 0, 1, r, what))
                same_rows(r1, r, rows, what);
            for (size_t j = 0; j < sizeof pieces / sizeof pieces[0]; j++) {
                if (pieces[j] == 1 && n > 2 * 16000)
                    continue;
                snprintf(what, sizeof what, "%zu samples in pieces of %zu", n, pieces[j]);
                if (!diar_run(j % 2 ? d3 : d1, m, x, n, pieces[j], 1, r, what))
                    same_rows(r1, r, rows, what);
            }
        }
        free(x);
        free(r1);
        free(r);
    }

    /* samples that are not finite are heard as 0 */
    const size_t n = 2 * 16000, rows = (size_t)diar_nrows(m, n) * S;
    float *x = audio(n), *z = malloc(n * sizeof *z), *r1 = malloc(rows * sizeof *r1), *r = malloc(rows * sizeof *r);
    memcpy(z, x, n * sizeof *z);
    for (size_t i = 0; i < n; i += 97) {
        x[i] = i % 3 == 0 ? NAN : i % 3 == 1 ? INFINITY : -INFINITY;
        z[i] = 0;
    }
    if (!diar_run(d1, m, z, n, 0, 1, r1, "zeros where the next has NaN")
        && !diar_run(d1, m, x, n, 0, 1, r, "NaN and infinite samples"))
        same_rows(r1, r, rows, "NaN and infinite samples, against zeros there");
    /* finite and huge: the power spectrum overflows, and quantizing makes
     * those blocks zeros */
    for (size_t i = 0; i < n; i += 7)
        x[i] = i % 2 ? 1e30f : -3e38f;
    diar_run(d3, m, x, n, 0, 1, NULL, "samples of 1e30");

    /* a run fn stops: 1, and the next recording as if it had not been */
    free(x);
    x = audio(n);
    diar_run(d1, m, x, n, 0, 1, r1, "a recording");
    diar_sink k = { 0, S, 0, NULL, 1, 0, 1 };
    int rc = pk_diar_run(d1, x, n, diar_rows, &k);
    g_cases++;
    if (rc != 1 || k.next >= diar_nrows(m, n))
        FAIL("a run fn stops: returned %d after %lld rows", rc, (long long)k.next);
    if (!diar_run(d1, m, x, n, 0, 1, r, "the recording after a stopped one"))
        same_rows(r1, r, rows, "the recording after a stopped one");
    k = (diar_sink){ 0, S, 0, NULL, 1, 0, 1 };
    rc = 0;
    for (size_t i = 0; i < n && !rc; i += 5000)
        rc = pk_diar_feed(d1, x + i, n - i < 5000 ? n - i : 5000, diar_rows, &k);
    g_cases++;
    if (rc != 1)
        FAIL("a feed fn stops: returned %d", rc);
    pk_diar_reset(d1);
    if (!diar_run(d1, m, x, n, 0, 1, r, "the recording after a reset"))
        same_rows(r1, r, rows, "the recording after a reset");
    k = (diar_sink){ 0, S, 0, NULL, 1, 0, -1 };
    g_cases++;
    if (pk_diar_feed(d1, x, 0, diar_rows, &k) || pk_diar_finish(d1, diar_rows, &k) || k.next)
        FAIL("an empty recording: %lld rows", (long long)k.next);
    free(x);
    free(z);
    free(r1);
    free(r);
    pk_diar_free(d1);
    pk_diar_free(d3);
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
                if (g_diar)
                    diar_engine_cases(m, 0);
                else
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
        fprintf(stderr, "usage: pkfcheck tests/data/tiny.pkf [WAV_OUT], or tests/data/tiny-diar.pkf\n");
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
    g_diar = !strcmp(g_m.arch, "sortformer");
    if (g_m.nt + 2 > MAXT || g_m.np + 2 > MAXP) {
        printf("pkfcheck: %s is bigger than this test's tables\n", argv[1]);
        return 2;
    }

    int at = 0, valid, trunc, corrupt, meaning, graph, engine;
    valid_cases(argv[1]);
    valid = g_cases - at, at = g_cases;
    truncated_cases();
    trunc = g_cases - at, at = g_cases;
    header_cases();
    param_cases();
    descriptor_cases();
    if (!g_diar)
        vocab_cases();
    meta_cases();
    corrupt = g_cases - at, at = g_cases;
    if (g_diar) {
        diar_meaning_cases();
    } else {
        meaning_cases();
        slim_cases();
    }
    meaning = g_cases - at, at = g_cases;
#ifdef PK_QNN
    if (!g_diar)
        graph_cases();
#endif
    graph = g_cases - at, at = g_cases;
    pk_model *m = pk_model_from_bytes(g_file, g_n, err, sizeof err);
    if (m) {
        if (g_diar)
            diar_engine_cases(m, 1);
        else
            engine_cases(m, 1);
        pk_model_free(m);
    }
    engine = g_cases - at, at = g_cases;
    flip_cases();
    if (argc > 2)
        write_wav(argv[2]);
    printf("pkfcheck: valid %d, truncated %d, corrupt %d, meaning %d, graph %d, engine %d, flipped %d\n",
           valid, trunc, corrupt, meaning, graph, engine, g_cases - at);

    free(g_file);
    if (g_bad) {
        printf("pkfcheck: %d of %d cases wrong\n", g_bad, g_cases);
        return 1;
    }
    printf("pkfcheck: %d cases, every one loaded, refused or run as it should be\n", g_cases);
    return 0;
}
