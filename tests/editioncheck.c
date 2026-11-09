/* tests/editions.tsv's devices through core/editions.c: each gets the
 * edition of each model the file says it does ("-": none), and none has
 * two editions of one model at the top, which would leave the choice to
 * the catalog's order. tests/editions-rules.tsv does the same for the
 * made-up models of tests/editions-rules.json (tests/rules-catalog.c),
 * whose preview and too-new editions hold choose() to its rules; and the
 * order of versions is checked on literal pairs, which no catalog's
 * entries go stale against. pktito-app's cargo test reads the same files.
 *
 *     tests/editioncheck tests/editions.tsv */
#include "editions.h"
#include "pktito.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_COLS 16

extern const ed_model ed_rules_catalog[];
extern const int ed_rules_n_models;

/* the model of that name in models.json, else in the rules' catalog */
static const ed_model *model(const char *name) {
    const ed_model *m = ed_find(name);
    for (int i = 0; !m && i < ed_rules_n_models; i++)
        if (!strcmp(ed_rules_catalog[i].name, name))
            m = &ed_rules_catalog[i];
    return m;
}

/* ed_version_cmp on pairs whose order is known: 1, or 0 having said
 * which was wrong */
static int versions(void) {
    static const struct { const char *a, *b; int order; } pairs[] = {
        { "0.9.10", "0.9.8", 1 },  { "0.10.0", "0.9.6", 1 }, { "0.9.6", "0.9.6", 0 },
        { "0.9.8", "0.9.10", -1 }, { "1.0.0", "0.99.99", 1 }, { "0.9", "0.9.0", 0 },
        { "99.0.0", PKTITO_VERSION, 1 },
    };
    int ok = 1;
    for (size_t i = 0; i < sizeof pairs / sizeof *pairs; i++) {
        const int c = ed_version_cmp(pairs[i].a, pairs[i].b), got = (c > 0) - (c < 0);
        if (got != pairs[i].order) {
            printf("editioncheck: FAIL - %s against %s orders %d, not %d\n", pairs[i].a, pairs[i].b, got,
                   pairs[i].order);
            ok = 0;
        }
    }
    return ok;
}

/* line's tab-separated fields, in place: how many */
static int fields(char *line, char *f[MAX_COLS]) {
    int n = 0;
    line[strcspn(line, "\r\n")] = 0;
    for (char *s = line; n < MAX_COLS; s++) {
        f[n++] = s;
        if (!(s = strchr(s, '\t')))
            break;
        *s = 0;
    }
    return n;
}

int main(int argc, char **argv) {
    FILE *in = argc == 2 ? fopen(argv[1], "r") : NULL;
    if (!in) {
        fprintf(stderr, "usage: editioncheck tests/editions.tsv\n");
        return 2;
    }
    char line[1024], head[1024], *h[MAX_COLS] = { 0 }, *f[MAX_COLS];
    int nh = 0, devices = 0, bad = 0;
    while (fgets(line, sizeof line, in)) {
        if (line[0] == '#' || line[0] == '\n')
            continue;
        if (!nh) {
            memcpy(head, line, sizeof head);
            nh = fields(head, h);
            if (nh < 5 || strcmp(h[0], "device")) {
                fprintf(stderr, "editioncheck: no header line\n");
                return 2;
            }
            for (int c = 4; c < nh; c++)
                if (!model(h[c])) {
                    fprintf(stderr, "editioncheck: no model %s in models.json or tests/editions-rules.json\n", h[c]);
                    return 2;
                }
            continue;
        }
        if (fields(line, f) != nh) {
            fprintf(stderr, "editioncheck: a line of the wrong width: %s\n", f[0]);
            return 2;
        }
        ed_device d;
        memset(&d, 0, sizeof d);
        snprintf(d.system, sizeof d.system, "%s", f[1]);
        if (sscanf(f[2], "%d.%d", &d.os[0], &d.os[1]) != 2) {
            fprintf(stderr, "editioncheck: %s: os %s is not major.minor\n", f[0], f[2]);
            return 2;
        }
        snprintf(d.qnn, sizeof d.qnn, "%s", strcmp(f[3], "-") ? f[3] : "");
        devices++;
        for (int c = 4; c < nh; c++) {
            const ed_model *m = model(h[c]);
            int top = 0, at_top = 0;
            for (int i = 0; i < m->n_editions; i++) {
                if (!ed_may(m, &m->editions[i], 0))
                    continue;
                const int fit = ed_fit(&m->editions[i], &d);
                if (fit > top)
                    top = fit, at_top = 0;
                at_top += fit && fit == top;
            }
            const ed_edition *e = ed_choose(m, &d, NULL, 0);
            const char *got = e ? e->id : "-";
            if (at_top > 1 || (e && at_top != 1)) {
                printf("editioncheck: FAIL - %s: %d editions of %s fit at %d\n", f[0], at_top, m->name, top);
                bad++;
            } else if (strcmp(got, f[c])) {
                printf("editioncheck: FAIL - %s: %s's %s, not %s\n", f[0], m->name, got, f[c]);
                bad++;
            }
        }
    }
    fclose(in);
    if (!devices) {
        fprintf(stderr, "editioncheck: no devices\n");
        return 2;
    }
    if (!versions())
        bad++;
    if (!bad)
        printf("editioncheck: %d devices, each model's edition as %s says, no ties; versions in order\n", devices,
               argv[1]);
    return bad ? 1 : 0;
}
