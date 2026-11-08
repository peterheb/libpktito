/* models.json as C, for pktito-c (core/editions.h): written by
 * tools/catalog.py, which make check-python holds this file to. Edit
 * models.json and run python3 tools/catalog.py, not this. */
#include "editions.h"

#include <stddef.h>

const char ed_base[] = "https://models.pktito.net/";

static const char *const on_android[] = { "android", NULL };
static const char *const on_android_windows[] = { "android", "windows", NULL };
static const char *const on_windows[] = { "windows", NULL };

static const ed_edition model0[] = {
    {
        .id = "cpu",
        .about = "any CPU",
        .kind = ED_CPU,
        .model = { "parakeet-tdt-0.6b-v2.pkf", 624480196u, "8bfa6750b83cd7c447bad82057c7a12473563b72b3d35487fcca6da0786d83c7" },
    },
    {
        .id = "coreml",
        .about = "Apple's Neural Engine",
        .kind = ED_COREML,
        .macos = { 15, 0 },
        .ios = { 18, 0 },
        .model = { "parakeet-tdt-0.6b-v2-slim.pkf", 12115524u, "e582eb76db19e9fba4d9e15d45ce041dd9a230bac3c2970643aa6808bf966acb" },
        .encoder = { "parakeet-tdt-0.6b-v2-encoder.mlmodelc.aar", 546910291u, "70272da7508aff518225caeaea3f0dc7dedb5bf0cc829da72876c9ecae5159a1" },
    },
    {
        .id = "qnn-sm8650",
        .about = "the Snapdragon 8 Gen 3 (Galaxy S24)",
        .kind = ED_QNN,
        .chip = "57,75,8,1,0",
        .systems = on_android,
        .model = { "parakeet-tdt-0.6b-v2-slim.pkf", 12115524u, "e582eb76db19e9fba4d9e15d45ce041dd9a230bac3c2970643aa6808bf966acb" },
        .encoder = { "parakeet-tdt-0.6b-v2-sm8650.qnn", 714506856u, "3f690d2cdd0ad6dc3b805f627f5c8e0b7701af20135bc275ebcc8f55877263a1" },
    },
    {
        .id = "qnn-sm8750",
        .about = "the Snapdragon 8 Elite (Galaxy S25)",
        .kind = ED_QNN,
        .chip = "69,79,8,1,0",
        .systems = on_android,
        .model = { "parakeet-tdt-0.6b-v2-slim.pkf", 12115524u, "e582eb76db19e9fba4d9e15d45ce041dd9a230bac3c2970643aa6808bf966acb" },
        .encoder = { "parakeet-tdt-0.6b-v2-sm8750.qnn", 713568872u, "40a5b75f37954b528f04944c57096ba0215783cea23b50ef659a981b5995456c" },
    },
    {
        .id = "qnn-sm8850",
        .about = "the Snapdragon 8 Elite Gen 5 (Galaxy S26)",
        .kind = ED_QNN,
        .chip = "87,81,8,1,0",
        .systems = on_android,
        .model = { "parakeet-tdt-0.6b-v2-slim.pkf", 12115524u, "e582eb76db19e9fba4d9e15d45ce041dd9a230bac3c2970643aa6808bf966acb" },
        .encoder = { "parakeet-tdt-0.6b-v2-sm8850.qnn", 714412648u, "08388c7f4b9cbb562330111ed6e0b026d00512168cee589b04578f50673b1aff" },
    },
    {
        .id = "qnn-x1e80100",
        .about = "the Snapdragon X Elite (X1E80100)",
        .kind = ED_QNN,
        .chip = "60,73,8,1,1",
        .systems = on_android_windows,
        .model = { "parakeet-tdt-0.6b-v2-slim.pkf", 12115524u, "e582eb76db19e9fba4d9e15d45ce041dd9a230bac3c2970643aa6808bf966acb" },
        .encoder = { "parakeet-tdt-0.6b-v2-x1e80100.qnn", 714523240u, "dab2d4d69028caba976c0b82316f83ad49bd3d4bac01b39992cd66904d216513" },
    },
    {
        .id = "qnn-sc8480xp",
        .about = "the Snapdragon X2 Elite",
        .kind = ED_QNN,
        .chip = "88,81,8,1,1",
        .systems = on_windows,
        .model = { "parakeet-tdt-0.6b-v2-slim.pkf", 12115524u, "e582eb76db19e9fba4d9e15d45ce041dd9a230bac3c2970643aa6808bf966acb" },
        .encoder = { "parakeet-tdt-0.6b-v2-sc8480xp.qnn", 714445416u, "726db8ecde7934510ab0193f52de93601c64506edf1804a240612a823e90d811" },
        .note = "compiled for the X2 Elite's V81, not yet run on one",
    },
};

static const ed_edition model1[] = {
    {
        .id = "cpu",
        .about = "any CPU",
        .kind = ED_CPU,
        .model = { "parakeet-tdt-0.6b-v3.pkf", 647552644u, "35a59c9a433d4da2289640db93e73c42478b576e49a293090ef61b4ae595152a" },
    },
    {
        .id = "coreml",
        .about = "Apple's Neural Engine",
        .kind = ED_COREML,
        .macos = { 15, 0 },
        .ios = { 18, 0 },
        .model = { "parakeet-tdt-0.6b-v3-slim.pkf", 35187972u, "169ead8b6b98c67676d13c40bd592614f9d1a3b3783da027fe03967cd559b100" },
        .encoder = { "parakeet-tdt-0.6b-v3-encoder.mlmodelc.aar", 545275075u, "a7eef40e28b009fee25548fe869124d2b9ba7be8c00c2dcb1b2e8e8973524571" },
    },
    {
        .id = "qnn-sm8650",
        .about = "the Snapdragon 8 Gen 3 (Galaxy S24)",
        .kind = ED_QNN,
        .chip = "57,75,8,1,0",
        .systems = on_android,
        .model = { "parakeet-tdt-0.6b-v3-slim.pkf", 35187972u, "169ead8b6b98c67676d13c40bd592614f9d1a3b3783da027fe03967cd559b100" },
        .encoder = { "parakeet-tdt-0.6b-v3-sm8650.qnn", 714506856u, "84aae2af7273132997bb71af73550f1a9a037de35b3d45d9e95a9390a796324e" },
    },
    {
        .id = "qnn-sm8750",
        .about = "the Snapdragon 8 Elite (Galaxy S25)",
        .kind = ED_QNN,
        .chip = "69,79,8,1,0",
        .systems = on_android,
        .model = { "parakeet-tdt-0.6b-v3-slim.pkf", 35187972u, "169ead8b6b98c67676d13c40bd592614f9d1a3b3783da027fe03967cd559b100" },
        .encoder = { "parakeet-tdt-0.6b-v3-sm8750.qnn", 713568872u, "d5cf8b7659ea176f66937a94e72c80bb11caac60d18811b0f500b6b1eaa73113" },
    },
    {
        .id = "qnn-sm8850",
        .about = "the Snapdragon 8 Elite Gen 5 (Galaxy S26)",
        .kind = ED_QNN,
        .chip = "87,81,8,1,0",
        .systems = on_android,
        .model = { "parakeet-tdt-0.6b-v3-slim.pkf", 35187972u, "169ead8b6b98c67676d13c40bd592614f9d1a3b3783da027fe03967cd559b100" },
        .encoder = { "parakeet-tdt-0.6b-v3-sm8850.qnn", 714412648u, "68eb09b667c7f9f6f36bb7eb7812945078150d8ff9984c61ce169d1bdfe99eab" },
    },
    {
        .id = "qnn-x1e80100",
        .about = "the Snapdragon X Elite (X1E80100)",
        .kind = ED_QNN,
        .chip = "60,73,8,1,1",
        .systems = on_android_windows,
        .model = { "parakeet-tdt-0.6b-v3-slim.pkf", 35187972u, "169ead8b6b98c67676d13c40bd592614f9d1a3b3783da027fe03967cd559b100" },
        .encoder = { "parakeet-tdt-0.6b-v3-x1e80100.qnn", 714523240u, "1208ddc1b2c63e68568d692780297d142ad5f57ee3bb05dd86218347edd7809f" },
    },
    {
        .id = "qnn-sc8480xp",
        .about = "the Snapdragon X2 Elite",
        .kind = ED_QNN,
        .chip = "88,81,8,1,1",
        .systems = on_windows,
        .model = { "parakeet-tdt-0.6b-v3-slim.pkf", 35187972u, "169ead8b6b98c67676d13c40bd592614f9d1a3b3783da027fe03967cd559b100" },
        .encoder = { "parakeet-tdt-0.6b-v3-sc8480xp.qnn", 714445416u, "237a5a6c51abfcb969310d494f70b2663c568a43174b70b9fd3f6d624c6caa7a" },
        .note = "compiled for the X2 Elite's V81, not yet run on one",
    },
};

const ed_model ed_catalog[] = {
    { "parakeet-tdt-0.6b-v2", "English", model0, 7 },
    { "parakeet-tdt-0.6b-v3", "25 European languages", model1, 7 },
};
const int ed_n_models = 2;
