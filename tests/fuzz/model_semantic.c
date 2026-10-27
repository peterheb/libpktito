/* Hostile but structurally valid model values.  A compact patch language is
 * much more likely than blind mutation of a 1.2 MB file to reach inference:
 * it preserves descriptors and shapes while changing typed tensor values,
 * bounded search params, and valid text. */
#include "model.h"
#include "pkf.h"
#include "pktito.h"

#include "fuzz.h"

#define SEM_MAX_OPS 16
#define SEM_MAX_SAMPLES 2048

static unsigned char *base;
static size_t base_n;
static pk_header head;
static pk_tensor *tensor;
static uint32_t *q8_index, *f32_index, *param_index;
static size_t nq8, nf32, nparam;
static size_t *vocab_ascii, *meta_ascii;
static size_t nvocab_ascii, nmeta_ascii;

static uint32_t rd32(const unsigned char *p) {
    uint32_t v;
    memcpy(&v, p, sizeof v);
    return v;
}

static int fixed_float_table(const char *name) {
    return !strcmp(name, "mel.fft.cos") || !strcmp(name, "mel.fft.sin") || !strcmp(name, "pos.freq")
        || !strcmp(name, "vad.fft.cos") || !strcmp(name, "vad.fft.sin") || !strcmp(name, "vad.stft.window");
}

int LLVMFuzzerInitialize(int *argc, char ***argv) {
    (void)argc;
    (void)argv;
    fz_force_ref();
    const char *path = getenv("PK_FUZZ_MODEL");
    if (!path || !*path)
        path = "tests/data/tiny.pkf";
    base = fz_read_file(path, &base_n);
    char err[256];
    FZ_ASSERT(pk_pkf_validate(base, base_n, err, sizeof err) == 0);
    memcpy(&head, base, sizeof head);
    tensor = malloc((size_t)head.n_tensors * sizeof *tensor);
    q8_index = malloc((size_t)head.n_tensors * sizeof *q8_index);
    f32_index = malloc((size_t)head.n_tensors * sizeof *f32_index);
    param_index = malloc((size_t)head.n_params * sizeof *param_index);
    vocab_ascii = malloc((size_t)head.vocab_size * sizeof *vocab_ascii);
    meta_ascii = malloc(((size_t)head.meta_size ? (size_t)head.meta_size : 1) * sizeof *meta_ascii);
    FZ_ASSERT(tensor && q8_index && f32_index && param_index && vocab_ascii && meta_ascii);

    for (uint32_t i = 0; i < head.n_tensors; i++) {
        memcpy(&tensor[i], base + head.tensors_off + (size_t)i * sizeof *tensor, sizeof *tensor);
        if (tensor[i].type == PK_Q8)
            q8_index[nq8++] = i;
        else if (!fixed_float_table(tensor[i].name))
            f32_index[nf32++] = i;
    }
    for (uint32_t i = 0; i < head.n_params; i++) {
        pk_param p;
        memcpy(&p, base + head.params_off + (size_t)i * sizeof p, sizeof p);
        if (!strcmp(p.key, "max_symbols") || !strncmp(p.key, "dur", 3))
            param_index[nparam++] = i;
    }

    const unsigned char *v = base + head.vocab_off;
    uint32_t pieces = rd32(v), text_n = rd32(v + 4 + 4 * (size_t)pieces);
    size_t text = (size_t)head.vocab_off + 4 + 4 * ((size_t)pieces + 1) + pieces;
    for (size_t i = 0; i < text_n; i++)
        if (base[text + i] > 0 && base[text + i] < 0x80)
            vocab_ascii[nvocab_ascii++] = text + i;
    for (size_t i = 0; i < head.meta_size; i++)
        if (base[head.meta_off + i] > 0 && base[head.meta_off + i] < 0x80)
            meta_ascii[nmeta_ascii++] = (size_t)head.meta_off + i;
    FZ_ASSERT(nq8 && nf32 && nparam && nvocab_ascii);
    return 0;
}

static void patch_q8(unsigned char *file, uint32_t selector, uint32_t value) {
    const pk_tensor *t = &tensor[q8_index[selector % nq8]];
    file[t->offset + value % t->bytes] = (unsigned char)(value >> 24);
}

static void patch_f32(unsigned char *file, uint32_t selector, uint32_t value) {
    const pk_tensor *t = &tensor[f32_index[selector % nf32]];
    if ((value & 0x7f800000u) == 0x7f800000u)
        value = (value & 0x807fffffu) | 0x7f000000u; /* maximum exponent, still finite */
    uint64_t elements = t->bytes / sizeof(float);
    size_t at = (size_t)t->offset + (size_t)(selector % elements) * sizeof(float);
    memcpy(file + at, &value, sizeof value);
}

static void patch_param(unsigned char *file, uint32_t selector, uint32_t value) {
    uint32_t i = param_index[selector % nparam];
    pk_param p;
    size_t at = (size_t)head.params_off + (size_t)i * sizeof p;
    memcpy(&p, file + at, sizeof p);
    p.v.u = !strcmp(p.key, "max_symbols") ? 1 + value % 64 : value % 9;
    memcpy(file + at, &p, sizeof p);
}

static void patch_text(unsigned char *file, const size_t *where, size_t n,
                       uint32_t selector, uint32_t value) {
    static const char alphabet[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'-";
    if (n)
        file[where[selector % n]] = (unsigned char)alphabet[value % (sizeof alphabet - 1)];
}

static void check_run(pk_model *m, const float *x, size_t n) {
    pk_model_info mi;
    pk_model_describe(m, &mi);
    pk_asr *a = pk_asr_new(m, 1);
    FZ_ASSERT(a);
    const pk_tok *tok;
    long nt = pk_asr_run(a, x, n, &tok);
    FZ_ASSERT(nt >= 0);
    for (long i = 0; i < nt; i++)
        FZ_ASSERT(tok[i].id >= 0 && (uint32_t)tok[i].id < mi.vocab && tok[i].frame >= 0 && tok[i].dur >= 0);
    size_t len = pk_text(m, tok, (size_t)nt, NULL, 0);
    FZ_ASSERT(len < 16 * 1024 * 1024);
    char *text = malloc(len + 1);
    FZ_ASSERT(text);
    FZ_ASSERT(pk_text(m, tok, (size_t)nt, text, len + 1) == len && strlen(text) == len && fz_utf8(text, len));
    free(text);

    size_t frames = n / PK_VAD_SAMPLES;
    if (frames) {
        pk_vad *v = pk_vad_new(m);
        float *prob = malloc(frames * sizeof *prob);
        FZ_ASSERT(v && prob);
        pk_vad_probs(v, x, frames, prob);
        for (size_t i = 0; i < frames; i++)
            FZ_ASSERT(isfinite(prob[i]) && prob[i] >= 0 && prob[i] <= 1);
        free(prob);
        pk_vad_free(v);
    }
    pk_asr_free(a);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (!size || size > 8192)
        return 0;
    unsigned char *file = malloc(base_n);
    FZ_ASSERT(file);
    memcpy(file, base, base_n);
    fz_input f = { data, size, 0 };
    unsigned ops = fz_u8(&f) % (SEM_MAX_OPS + 1);
    for (unsigned i = 0; i < ops; i++) {
        unsigned kind = fz_u8(&f) % 5;
        uint32_t selector = fz_u32(&f), value = fz_u32(&f);
        if (kind == 0) patch_q8(file, selector, value);
        else if (kind == 1) patch_f32(file, selector, value);
        else if (kind == 2) patch_param(file, selector, value);
        else if (kind == 3) patch_text(file, vocab_ascii, nvocab_ascii, selector, value);
        else patch_text(file, meta_ascii, nmeta_ascii, selector, value);
    }

    char err[256] = "";
    pk_model *m = pk_model_from_bytes(file, base_n, err, sizeof err);
    free(file);
    if (!m) {
        FZ_ASSERT(err[0]);
        return 0;
    }
    size_t n = (f.n - f.at) / sizeof(float);
    if (n > SEM_MAX_SAMPLES)
        n = SEM_MAX_SAMPLES;
    float x[SEM_MAX_SAMPLES];
    for (size_t i = 0; i < n; i++)
        x[i] = fz_float(&f);
    check_run(m, x, n);
    pk_model_free(m);
    return 0;
}
