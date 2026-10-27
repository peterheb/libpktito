/* Arbitrary bytes through both layers of the .pkf boundary.  A byte-valid
 * file need not be a model this build implements; only a model the semantic
 * binder accepts is inspected through the public API. */
#include "model.h"
#include "pktito.h"

#include "fuzz.h"

int LLVMFuzzerInitialize(int *argc, char ***argv) {
    (void)argc;
    (void)argv;
    fz_force_ref();
    return 0;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size > 4u * 1024 * 1024)
        return 0;

    char verr[256] = "", err[256] = "";
    int valid = pk_pkf_validate(data, size, verr, sizeof verr);
    if (valid) {
        FZ_ASSERT(verr[0]);
        return 0;
    }

    pk_model *m = pk_model_from_bytes(data, size, err, sizeof err);
    if (!m) {
        FZ_ASSERT(err[0]);                 /* byte-valid, but not this model */
        return 0;
    }

    pk_model_info info;
    pk_model_describe(m, &info);
    FZ_ASSERT(info.file_bytes == size);
    FZ_ASSERT(pk_model_name(m) && pk_model_name(m)[0]);
    FZ_ASSERT(pk_model_rate(m) == 16000);
    FZ_ASSERT(pk_model_frame_seconds(m) > 0);
    FZ_ASSERT(info.vocab > 0 && info.n_durations > 0 && info.n_durations <= 16);

    uint64_t h = fz_hash(data, size);
    size_t len = 0;
    const char *piece = pk_piece(m, (int32_t)(h % info.vocab), &len);
    FZ_ASSERT(piece && fz_utf8(piece, len));
    piece = pk_piece(m, -1, &len);
    FZ_ASSERT(piece && len == 0);

    for (int i = 0; i < 8; i++) {
        const char *key, *value;
        if (pk_model_meta_at(m, i, &key, &value))
            break;
        FZ_ASSERT(key && *key && value);
    }
    (void)pk_model_meta(m, "family");
    pk_model_free(m);
    return 0;
}
