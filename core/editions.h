/* The editions of each model, as models.json catalogs them: the files one
 * kind of hardware runs a model from, and the rule that picks one for a
 * device (fit, choose), the same as pktito-app's editions.rs. pktito-c's
 * `models` and its slim model's encoder come from here. tests/editions.tsv
 * holds both implementations to the same answers. Not the library's:
 * core/pktito.h has none of it. */
#ifndef PK_EDITIONS_H
#define PK_EDITIONS_H

#include <stdint.h>

/* a file models.pktito.net serves, at its base, the first 12 digits of the
 * SHA-256, a slash and the name */
typedef struct {
    const char *file;
    uint64_t bytes;
    const char *sha256;
} ed_part;

enum { ED_CPU, ED_COREML, ED_QNN };

typedef struct {
    const char *id;                   /* "cpu", "coreml", "qnn-sm8650" */
    const char *about;                /* the hardware it runs on, in words */
    int kind;                         /* ED_* */
    int macos[2], ios[2];             /* Core ML's: the systems it needs */
    const char *chip;                 /* QNN's name for its chip, as pk_qnn_chip */
    const char *const *systems;       /* QNN's: the systems it runs on, NULL-ended */
    ed_part model;                    /* the whole .pkf for the CPU's, else the slim one */
    ed_part encoder;                  /* the NPU's; file NULL for the CPU's */
    const char *note;                 /* what models.json says of it, or NULL */
} ed_edition;

typedef struct {
    const char *name, *about;
    const ed_edition *editions;
    int n_editions;
} ed_model;

/* core/catalog.c, written from models.json by tools/catalog.py */
extern const char ed_base[];
extern const ed_model ed_catalog[];
extern const int ed_n_models;

/* a device as the rule sees it: its system (as Rust's std::env::consts::OS
 * names them), its version, and QNN's name for its NPU, "" for none */
typedef struct {
    char system[16];
    int os[2];
    char qnn[64];
} ed_device;

/* How well e fits d: 3 Core ML on a system at its floor or later, or QNN
 * for d's chip exactly; 2 QNN for another chip of the same Hexagon with no
 * more VTCM, on one of e's systems; 1 the CPU's, always; 0 none */
int ed_fit(const ed_edition *e, const ed_device *d);

/* m's edition for d: the one that fits best, of those can_run takes (NULL
 * for all). Two at the top is the catalog's mistake (tests/editioncheck
 * looks for one), and the first is taken. The CPU's always fits, so there
 * is always one. */
const ed_edition *ed_choose(const ed_model *m, const ed_device *d, int (*can_run)(const ed_edition *));

/* the model of that name, or NULL */
const ed_model *ed_find(const char *name);

/* this machine, as far as it can tell: its system and version, and QNN's
 * chip where qnn_chip is given and finds one */
void ed_this_device(ed_device *d, int (*qnn_chip)(char *chip, unsigned cap));

#endif
