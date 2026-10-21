/* core/tokens.c from the outside: each line of stdin is token ids, and
 * each line of stdout is pk_text of them. tests/tokcheck.py drives it and
 * holds its text to sentencepiece's and NeMo's.
 *
 *     ./tests/tokcheck MODEL.pkf < ids.txt
 */
#include "pktito.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    char err[256], line[65536], text[262144];
    if (argc != 2) {
        fprintf(stderr, "usage: tokcheck MODEL.pkf < ids\n");
        return 2;
    }
    pk_model *m = pk_model_load(argv[1], err, sizeof err);
    if (!m) {
        fprintf(stderr, "tokcheck: %s\n", err);
        return 2;
    }
    pk_tok t[4096];
    while (fgets(line, sizeof line, stdin)) {
        size_t n = 0;
        for (char *p = strtok(line, " \n"); p && n < 4096; p = strtok(NULL, " \n"))
            t[n++].id = (int32_t)strtol(p, NULL, 10);
        /* and once more into a buffer too small, which must cut at a
         * character and still terminate */
        size_t len = pk_text(m, t, n, text, sizeof text);
        char small[7];
        size_t len2 = pk_text(m, t, n, small, sizeof small);
        if (len2 != len || strlen(small) > 6 || strncmp(small, text, strlen(small))) {
            fprintf(stderr, "tokcheck: the cut text is not a prefix of the whole\n");
            return 1;
        }
        printf("%s\n", text);
    }
    pk_model_free(m);
    return 0;
}
