/* tests/editions.tsv's devices through core/editions.c: each gets the
 * edition of each model the file says it does, and none has two editions
 * of one model at the top, which would leave the choice to the catalog's
 * order. pktito-app's cargo test reads the same file.
 *
 *     tests/editioncheck tests/editions.tsv */
#include "editions.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_COLS 16

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
                if (!ed_find(h[c])) {
                    fprintf(stderr, "editioncheck: no model %s in models.json\n", h[c]);
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
            const ed_model *m = ed_find(h[c]);
            int top = 0, at_top = 0;
            for (int i = 0; i < m->n_editions; i++) {
                const int fit = ed_fit(&m->editions[i], &d);
                if (fit > top)
                    top = fit, at_top = 0;
                at_top += fit == top;
            }
            const char *got = ed_choose(m, &d, NULL)->id;
            if (at_top != 1) {
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
    if (!bad)
        printf("editioncheck: %d devices, each model's edition as tests/editions.tsv says, no ties\n", devices);
    return bad ? 1 : 0;
}
