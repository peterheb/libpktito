/* tests/editions-rules.json as C, for pktito-c (core/editions.h): written by
 * tools/catalog.py, which make check-python holds this file to. Edit
 * tests/editions-rules.json and run python3 tools/catalog.py, not this. */
#include "editions.h"

#include <stddef.h>

const char ed_rules_base[] = "https://example.invalid/";

static const char *const on_android[] = { "android", NULL };

static const ed_edition model0[] = {
    {
        .id = "cpu",
        .about = "any CPU",
        .kind = ED_CPU,
        .status = ED_STABLE,
        .model = { "rules-preview-npu.pkf", 1000u, "0000000000000000000000000000000000000000000000000000000000000001" },
    },
    {
        .id = "qnn-sm8750",
        .about = "the S25's NPU",
        .kind = ED_QNN,
        .status = ED_STABLE,
        .chip = "69,79,8,1,0",
        .systems = on_android,
        .model = { "rules-preview-npu-slim.pkf", 2000u, "0000000000000000000000000000000000000000000000000000000000000002" },
        .encoder = { "rules-preview-npu-sm8750.qnn", 3000u, "0000000000000000000000000000000000000000000000000000000000000003" },
    },
    {
        .id = "qnn-sm8650",
        .about = "the S24's NPU, a preview",
        .kind = ED_QNN,
        .status = ED_PREVIEW,
        .chip = "57,75,8,1,0",
        .systems = on_android,
        .model = { "rules-preview-npu-slim.pkf", 2000u, "0000000000000000000000000000000000000000000000000000000000000002" },
        .encoder = { "rules-preview-npu-sm8650.qnn", 5000u, "0000000000000000000000000000000000000000000000000000000000000005" },
    },
};

static const ed_edition model1[] = {
    {
        .id = "cpu",
        .about = "any CPU",
        .kind = ED_CPU,
        .status = ED_STABLE,
        .model = { "rules-too-new-npu.pkf", 6000u, "0000000000000000000000000000000000000000000000000000000000000006" },
    },
    {
        .id = "qnn-sm8750",
        .about = "the S25's NPU",
        .kind = ED_QNN,
        .status = ED_STABLE,
        .chip = "69,79,8,1,0",
        .systems = on_android,
        .model = { "rules-too-new-npu-slim.pkf", 7000u, "0000000000000000000000000000000000000000000000000000000000000007" },
        .encoder = { "rules-too-new-npu-sm8750.qnn", 8000u, "0000000000000000000000000000000000000000000000000000000000000008" },
    },
    {
        .id = "qnn-sm8650",
        .about = "the S24's NPU, for a later pktito",
        .kind = ED_QNN,
        .status = ED_STABLE,
        .needs = "99.0.0",
        .chip = "57,75,8,1,0",
        .systems = on_android,
        .model = { "rules-too-new-npu-slim.pkf", 7000u, "0000000000000000000000000000000000000000000000000000000000000007" },
        .encoder = { "rules-too-new-npu-sm8650.qnn", 10000u, "000000000000000000000000000000000000000000000000000000000000000a" },
    },
};

static const ed_edition model2[] = {
    {
        .id = "cpu",
        .about = "any CPU",
        .kind = ED_CPU,
        .status = ED_STABLE,
        .model = { "rules-preview-model.pkf", 11000u, "000000000000000000000000000000000000000000000000000000000000000b" },
    },
};

const ed_model ed_rules_catalog[] = {
    {
        .name = "rules-preview-npu",
        .about = "a stable model whose S24 edition is preview, not yet run there",
        .task = ED_TRANSCRIBE,
        .status = ED_STABLE,
        .needs = NULL,
        .editions = model0,
        .n_editions = 3,
    },
    {
        .name = "rules-too-new-npu",
        .about = "a stable model whose S24 edition needs a pktito no one has",
        .task = ED_TRANSCRIBE,
        .status = ED_STABLE,
        .needs = NULL,
        .editions = model1,
        .n_editions = 3,
    },
    {
        .name = "rules-preview-model",
        .about = "a preview model: its CPU edition is never chosen",
        .task = ED_TRANSCRIBE,
        .status = ED_PREVIEW,
        .needs = NULL,
        .editions = model2,
        .n_editions = 1,
    },
};
const int ed_rules_n_models = 3;
