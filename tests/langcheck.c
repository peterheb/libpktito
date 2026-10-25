/* langcheck: core/language.c on texts whose language is known.
 *
 *   tests/langcheck                   the fixtures: exit 0 if every one holds
 *   tests/langcheck --eval HELD.tsv   the measurement: each language's
 *                                     held-out transcripts as texts of 5 to
 *                                     200 words, every word weighted and not
 *
 * HELD.tsv is tools/langwords.py --lines: a language's code, a tab and a
 * transcript a line, from transcripts the table was not counted in. A
 * language's lines are run together in order and cut into texts of K words;
 * each text is right, wrong, or no answer. English's precision is the share
 * of texts called English that are, its recall the share of English texts
 * called English: what turning on the English rules rests on.
 *
 * It includes core/language.c, for its parts that are static. */
#include "language.c"

#include <stdio.h>
#include <stdlib.h>

static int fails;

/* The next word of *s, between any of the characters of sep, ended in place,
 * or NULL at the end: strtok's split without its hidden state, which
 * Android's C library marks deprecated (a warning in every make adb). */
static char *word(char **s, const char *sep) {
    char *w = *s + strspn(*s, sep);
    if (!*w)
        return NULL;
    char *e = w + strcspn(w, sep);
    *s = *e ? e + 1 : e;
    *e = '\0';
    return w;
}

static const char *guess(const char *text) {
    static char buf[4096];
    pk_capword w[512];
    size_t n = 0;
    snprintf(buf, sizeof buf, "%s", text);
    char *p = buf;
    for (char *t; n < 512 && (t = word(&p, " "));)
        w[n++] = (pk_capword){ t, 0, 0 };
    return pk_language(w, n);
}

static void expect(const char *text, const char *want) {
    const char *got = guess(text);
    if (!(got == want || (got && want && !strcmp(got, want)))) {
        fails++;
        printf("langcheck: FAIL \"%.40s...\": %s, not %s\n", text, got ? got : "none", want ? want : "none");
    }
}

static void fixtures(void) {
    expect("And says, how do I get to Dublin? I think about normative questions all the time, "
           "and what we owe to each other.", "en");
    expect("Der Hund lief über die Straße, und die Kinder sahen ihm nach, als er im Park verschwand.", "de");
    expect("El perro cruzó la calle y los niños lo miraron hasta que desapareció en el parque de la ciudad.", "es");
    expect("Le chien a traversé la rue et les enfants l'ont regardé jusqu'à ce qu'il disparaisse dans le parc.", "fr");
    /* Russian's commonest words are Ukrainian's and Bulgarian's too: one
     * sentence of them is no answer, two are Russian */
    expect("Собака перебежала улицу, и дети смотрели ей вслед, пока она не исчезла в парке.", NULL);
    expect("Собака перебежала улицу, и дети смотрели ей вслед, пока она не исчезла в парке. Потом они "
           "вернулись домой, потому что уже было поздно, и мама ждала их к ужину.", "ru");
    expect("THE DOG RAN ACROSS THE STREET AND THE CHILDREN WATCHED IT UNTIL IT WAS GONE INTO THE PARK.", "en");
    expect("So I think.", NULL);                                /* too few words */
    expect("", NULL);
    /* folding and punctuation: "Über" is "über", "«die»" is "die" */
    char buf[WORD_MAX];
    if (!normal("\xC2\xAB" "Die" "\xC2\xBB,", buf) || strcmp(buf, "die") || !normal("\xC3\x9C" "ber", buf) ||
        strcmp(buf, "\xC3\xBC" "ber") || !normal("\xD0\x9E\xD0\xBD\xD0\xB8", buf) || strcmp(buf, "\xD0\xBE\xD0\xBD\xD0\xB8") ||
        normal("...", buf)) {
        fails++;
        printf("langcheck: FAIL normalizing a word\n");
    }
    /* the table is sorted, as the search needs, and every word normalized */
    for (size_t i = 1; i < sizeof PK_LANG_WORDS / sizeof *PK_LANG_WORDS; i++)
        if (strcmp(PK_LANG_WORDS[i - 1].w, PK_LANG_WORDS[i].w) >= 0) {
            fails++;
            printf("langcheck: FAIL the table is not in order at %s\n", PK_LANG_WORDS[i].w);
            break;
        }
}

/* ------------------------------------------------------------- --eval --- */

typedef struct { char code[3]; char **words; size_t n, cap; } lang_text;

/* a copy of s (strdup is POSIX's, not C11's) */
static char *copy(const char *s) {
    size_t n = strlen(s) + 1;
    char *c = malloc(n);
    if (!c) exit(2);
    return memcpy(c, s, n);
}

static void eval(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); exit(2); }
    lang_text langs[LANG_N] = { 0 };
    char line[16384];
    while (fgets(line, sizeof line, f)) {
        char *tab = strchr(line, '\t');
        if (!tab || tab - line != 2)
            continue;
        size_t l = 0;
        while (l < LANG_N && strncmp(PK_LANG_CODES[l], line, 2)) l++;
        if (l == LANG_N)
            continue;
        lang_text *t = &langs[l];
        memcpy(t->code, line, 2);
        char *p = tab + 1;
        for (char *w; (w = word(&p, " \n\r"));) {
            if (t->n == t->cap && !(t->words = realloc(t->words, (t->cap = t->cap ? 2 * t->cap : 4096) * sizeof *t->words)))
                exit(2);
            t->words[t->n++] = copy(w);
        }
    }
    fclose(f);
    static const size_t K[] = { 5, 10, 20, 50, 100, 200 };
    printf("words  weights   texts  right  wrong  none   English: precision recall\n");
    for (size_t k = 0; k < sizeof K / sizeof *K; k++)
        for (int weighted = 1; weighted >= 0; weighted--) {
            long right = 0, wrong = 0, none = 0, total = 0, en_said = 0, en_right = 0, en_texts = 0;
            long confused[LANG_N][LANG_N] = { { 0 } };
            pk_capword *w = malloc(K[k] * sizeof *w);
            for (size_t l = 0; l < LANG_N; l++)
                for (size_t at = 0; at + K[k] <= langs[l].n; at += K[k]) {
                    for (size_t i = 0; i < K[k]; i++) w[i] = (pk_capword){ langs[l].words[at + i], 0, 0 };
                    int64_t score[LANG_N];
                    int g = winner(score, scores(w, K[k], weighted, score), LANG_MARGIN);
                    total++;
                    en_texts += l == 5;
                    if (g < 0) none++;
                    else if ((size_t)g == l) right++;
                    else { wrong++; confused[l][g]++; }
                    if (g == 5) { en_said++; en_right += l == 5; }
                }
            free(w);
            printf("%5zu  %-8s %6ld  %5.1f%% %5.2f%% %5.1f%%  %8.2f%% %6.1f%%", K[k], weighted ? "idf" : "flat", total,
                   100.0 * right / total, 100.0 * wrong / total, 100.0 * none / total,
                   en_said ? 100.0 * en_right / en_said : 100.0, en_texts ? 100.0 * en_right / en_texts : 0.0);
            /* the commonest confusions */
            for (int top = 0; top < 3; top++) {
                long most = 0; size_t a = 0, b = 0;
                for (size_t i = 0; i < LANG_N; i++)
                    for (size_t j = 0; j < LANG_N; j++)
                        if (confused[i][j] > most) most = confused[i][j], a = i, b = j;
                if (!most) break;
                printf("  %s->%s %ld", PK_LANG_CODES[a], PK_LANG_CODES[b], most);
                confused[a][b] = 0;
            }
            putchar('\n');
        }
}

int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "--eval")) {
        eval(argv[2]);
        return 0;
    }
    fixtures();
    printf("langcheck: %s\n", fails ? "FAIL" : "ok");
    return fails != 0;
}
