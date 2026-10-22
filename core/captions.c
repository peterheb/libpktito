/* Captions from words and their times: the cues of a subtitle track.
 *
 * docs/caption-spec.md (version 1.1, 2026-10-22) sets out the method, and
 * this is its kernel, sections 4 to 8. Each possible cue is a run of words,
 * an edge between two boundaries of a graph whose vertices are the
 * boundaries between words; it is laid out (one line, or two at its best
 * break) and timed before it is scored; and the track is the cheapest path
 * from the first boundary to the last, found by dynamic programming. Filling
 * each cue until it is full, as the CLI's --srt did before, leaves a word or
 * two to flash by at the end; this does not (tests/capcheck.c, the spec's
 * 11.3, where it finds the spec's own answer to the score).
 *
 * pktito's profile is the spec's en-web-v1 less what it has no use for:
 *   - no speakers, shot changes, frame snapping or gap between cues, and no
 *     detector's pauses: a pause is the gap between words' times, which the
 *     spec allows (4.2), TDT's durations making them;
 *   - the English rules (5.2: function words, phrase starters,
 *     abbreviations) only when the caller says the words are English; in
 *     the other languages, punctuation and pauses alone;
 *   - always a track: a word, or a run of words whose times overlap, that
 *     no cue can hold (wider than two lines, or said over more than 7 s) is
 *     a cue of its own anyway, flagged PK_CUE_FORCED, its words whole, where
 *     the spec stops for an editor (6, 9);
 *   - a cue cut at a pause in mid-sentence ends in an ellipsis, flagged
 *     PK_CUE_PAUSE, which its last line makes room for (cue(), below);
 *   - graphemes as UAX #29 counts them for the scripts the models write
 *     (Latin, Greek, Cyrillic) and for emoji (graphemes(), below).
 *
 * Every quantity is an integer - milliseconds, graphemes and the spec's
 * integer costs - so a track is the same on every machine.
 */
#include "pktito.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* en-web-v1 (the spec's 2.1) */
#define LINE          42    /* graphemes a line holds */
#define LINE_EASY     38    /* and past which a line costs */
#define CPS_TARGET    17    /* the reading rate a cue's time aims for */
#define CPS_MAX       20    /* the most a reader is asked to keep up with */
#define MIN_MS        1000
#define EASY_MIN_MS   1200
#define EASY_MAX_MS   6000
#define MAX_MS        7000
#define TAIL_MS       200   /* time after the last word, preferred */
#define TAIL_MAX_MS   500   /* and at most, */
#define TAIL_SHORT_MS 1000  /* or this much for a cue too short to read */
#define CLOSE_MS      250   /* a smaller gap before the next cue is closed */
#define PAUSE_WEAK    250
#define PAUSE_STRONG  500
#define PAUSE_CUT     1000  /* no cue may span a pause this long */

/* A forced cue's violation: more than any track's real ones add up to,
 * so that a track forces a cue only where nothing else will do. */
#define FORCED ((int64_t)1 << 44)
#define NEVER  INT64_MAX

static int64_t min64(int64_t a, int64_t b) { return a < b ? a : b; }
static int64_t max64(int64_t a, int64_t b) { return a > b ? a : b; }
static int64_t cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }   /* a >= 0, b > 0 */
static int64_t sq(int64_t a) { return a * a; }

/* --------------------------------------------------------------- graphemes --- */

/* The code point at *s, moving past it; a byte that is not part of
 * well-formed UTF-8 stands for itself. */
static uint32_t next_cp(const unsigned char **s) {
    const unsigned char *p = *s;
    uint32_t c = p[0];
    int n = c < 0x80 ? 0 : (c & 0xE0) == 0xC0 ? 1 : (c & 0xF0) == 0xE0 ? 2 : (c & 0xF8) == 0xF0 ? 3 : -1;
    if (n > 0) {
        uint32_t v = c & (0x3F >> n);
        for (int k = 1; k <= n; k++) {
            if ((p[k] & 0xC0) != 0x80) {
                n = 0;
                break;
            }
            v = v << 6 | (p[k] & 0x3F);
        }
        if (n > 0) {
            *s = p + n + 1;
            return v;
        }
    }
    *s = p + 1;
    return c;
}

/* What UAX #29 joins to the code point before it (GB9, GB9a): combining
 * marks, ZWNJ, variation selectors, emoji modifiers and tags, as far as the
 * models' scripts and emoji need them. */
static int extends(uint32_t c) {
    return (c >= 0x0300 && c <= 0x036F) || (c >= 0x0483 && c <= 0x0489) || (c >= 0x1AB0 && c <= 0x1AFF) ||
           (c >= 0x1DC0 && c <= 0x1DFF) || c == 0x200C || (c >= 0x20D0 && c <= 0x20FF) ||
           (c >= 0xFE00 && c <= 0xFE0F) || (c >= 0xFE20 && c <= 0xFE2F) || (c >= 0x1F3FB && c <= 0x1F3FF) ||
           (c >= 0xE0020 && c <= 0xE007F) || (c >= 0xE0100 && c <= 0xE01EF);
}

/* emoji that a ZWJ joins into one (GB11's Extended_Pictographic, in the
 * blocks that hold them) */
static int pictographic(uint32_t c) {
    return (c >= 0x1F000 && c <= 0x1FAFF) || (c >= 0x2600 && c <= 0x27BF) || (c >= 0x2300 && c <= 0x23FF);
}

static int regional(uint32_t c) { return c >= 0x1F1E6 && c <= 0x1F1FF; }

/* Graphemes in a word: what a reader counts as characters. "cafe" with a
 * combining accent is four, a family of four joined by ZWJs is one, a flag
 * of two regional indicators is one. */
static int32_t graphemes(const char *text) {
    const unsigned char *s = (const unsigned char *)text;
    int32_t n = 0;
    uint32_t prev = 0;
    int ri = 0;   /* regional indicators just before, in a row */
    for (int first = 1; *s; first = 0) {
        uint32_t c = next_cp(&s);
        int joins = !first && (extends(c) || c == 0x200D || (prev == 0x200D && pictographic(c)) ||
                               (regional(c) && ri % 2 == 1));
        n += !joins;
        ri = regional(c) ? ri + 1 : 0;
        prev = c;
    }
    return n;
}

/* ------------------------------------------------------- what a word ends in --- */

enum { P_NONE, P_COMMA, P_MAJOR, P_TERM };

/* The spec's 5.2, with its pinned lists. A word list is written with a
 * space before and after each word, so that strstr finds whole words. */
static const char FUNCTION_WORDS[] =
    " a an the this that these those my your his her its our their"                         /* determiners */
    " about above across after against along among around at before behind below beneath"  /* prepositions */
    " beside between beyond by despite down during except for from in inside into near of off"
    " on onto out outside over past since through throughout to toward towards under underneath"
    " until up upon with within without"
    " and but or nor so yet because although though unless while if when than as once whether" /* conjunctions */
    " i you he she it we they who which"                                                     /* pronouns */
    " am is are was were be been being have has had do does did can could may might must"   /* auxiliaries */
    " shall should will would not never ";
static const char PHRASE_STARTERS[] =
    " and but or nor yet so because although though unless while if when after before until since as"
    " where which who to in on at for from with by of into over under between through during without ";
/* The verbs whose "to" is the rest of a complex verb, going to, have to,
 * used to, ought to: a line broken before it splits the verb, as "Who's
 * going / to implement a 4% global wealth tax?" did in test.wav, where
 * "Who's going to implement / a 4% ..." is the break (the BBC's subtitle
 * guidelines: not between the parts of a complex verb). */
static const char SEMI_MODALS[] = " going have has had having got used ought want wants wanted need needs needed able supposed ";
static const char ABBREVIATIONS[] = " mr. mrs. ms. dr. prof. sr. jr. st. vs. etc. e.g. i.e. a.m. p.m. u.s. u.k. ";

/* the bytes of t without the closing quotes and brackets after it */
static size_t unclosed(const char *t) {
    size_t n = strlen(t);
    for (;;) {
        if (n && strchr("\"')]}", t[n - 1]))
            n--;
        else if (n >= 3 && (!memcmp(t + n - 3, "\xE2\x80\x9D", 3) || !memcmp(t + n - 3, "\xE2\x80\x99", 3)))
            n -= 3;   /* ” ’ */
        else
            return n;
    }
}

/* t's lexical form, lowercased (in ASCII: the lists are) and without the
 * punctuation around it, in out[cap]; "" if it does not fit, as no list's
 * word is that long. */
static void lexical(const char *t, char *out, size_t cap) {
    static const char *strip[] = { ".", ",", ";", ":", "!", "?", "(", ")", "[", "]", "{", "}", "\"", "'",
                                   "\xE2\x80\xA6", "\xE2\x80\x9C", "\xE2\x80\x9D", "\xE2\x80\x98", "\xE2\x80\x99" };
    size_t a = 0, b = strlen(t);
    for (int again = 1; again;) {
        again = 0;
        for (size_t k = 0; k < sizeof strip / sizeof *strip; k++) {
            size_t l = strlen(strip[k]);
            if (b - a >= l && !memcmp(t + a, strip[k], l)) {
                a += l;
                again = 1;
            }
            if (b - a >= l && !memcmp(t + b - l, strip[k], l)) {
                b -= l;
                again = 1;
            }
        }
    }
    if (b - a + 1 > cap) {
        out[0] = 0;
        return;
    }
    for (size_t k = a; k < b; k++)
        out[k - a] = (char)(t[k] >= 'A' && t[k] <= 'Z' ? t[k] - 'A' + 'a' : t[k]);
    out[b - a] = 0;
}

static int listed(const char *list, const char *w) {
    char key[40];
    size_t n = strlen(w);
    if (!n || n + 3 > sizeof key)
        return 0;
    key[0] = ' ';
    memcpy(key + 1, w, n);
    key[n + 1] = ' ';
    key[n + 2] = 0;
    return strstr(list, key) != NULL;
}

/* What the punctuation at the end of word t marks: a sentence's end, a
 * major break (; : em dash, ellipsis), a comma, or nothing. In English an
 * abbreviation's or an initial's period ends no sentence. */
static int mark(const char *t, int english) {
    size_t n = unclosed(t);
    if ((n >= 3 && !memcmp(t + n - 3, "\xE2\x80\xA6", 3)) || (n >= 3 && !memcmp(t + n - 3, "...", 3)))
        return P_MAJOR;
    if (n >= 3 && !memcmp(t + n - 3, "\xE2\x80\x94", 3))   /* — */
        return P_MAJOR;
    if (!n)
        return P_NONE;
    char c = t[n - 1];
    if (c == '?' || c == '!')
        return P_TERM;
    if (c == ';' || c == ':')
        return P_MAJOR;
    if (c == ',')
        return P_COMMA;
    if (c != '.')
        return P_NONE;
    if (english) {
        char low[16];
        if (n == 2 && t[0] >= 'A' && t[0] <= 'Z')
            return P_NONE;                                    /* an initial: "J." */
        if (n < sizeof low) {
            for (size_t k = 0; k < n; k++)
                low[k] = (char)(t[k] >= 'A' && t[k] <= 'Z' ? t[k] - 'A' + 'a' : t[k]);
            low[n] = 0;
            if (listed(ABBREVIATIONS, low))
                return P_NONE;
        }
    }
    return P_TERM;
}

/* any of . , ? ! ; : — … at the end, past closing quotes and brackets */
static int punctuated(const char *t) {
    size_t n = unclosed(t);
    if (n >= 3 && (!memcmp(t + n - 3, "\xE2\x80\x94", 3) || !memcmp(t + n - 3, "\xE2\x80\xA6", 3)))
        return 1;
    return n && strchr(".,?!;:", t[n - 1]) != NULL;
}

/* ------------------------------------------------------------------ the graph --- */

typedef struct {
    int32_t n;
    const pk_capword *w;
    int64_t *s, *e;        /* each word's times, made sound: starts in order, ends after starts */
    int64_t *gsum;         /* graphemes of words [0, k) */
    int32_t *b;            /* a cut at boundary k costs this (the spec's B(k)) */
    int64_t *csum;         /* crossing boundaries [1, k] costs this (cross(k), summed) */
    uint8_t *safe, *cut;   /* boundary k may end a cue; must */
    int64_t media_end;
} graph;

/* One possible cue, words [i, j), laid out and timed: the spec's 6 and 7,
 * and its scores, 8.2 and 8.3. ok 0: no cue can be these words. */
typedef struct {
    int     ok;
    int32_t line2, chars, flags;
    int64_t end, v, q;
} edge;

/* the width of words [i, j) on one line */
static int64_t width(const graph *g, int32_t i, int32_t j) { return g->gsum[j] - g->gsum[i] + (j - i - 1); }

/* a: when the last of the words [i, j) ends; whole: the words are one run
 * whose times overlap, with no boundary between them that a cue may end
 * at, so that if no cue can hold them, they are a forced one */
static edge cue(const graph *g, int32_t i, int32_t j, int64_t a, int whole) {
    edge r = { 0 };
    /* Cut at a pause in mid-sentence (a second or more: PAUSE_CUT), its last
     * word without the punctuation that says the words stop there, a cue
     * ends in an ellipsis that does, "So I…" before "when I think about
     * normative questions," in test.wav: the pause marker of Netflix's
     * style guides and DCMP's, and right whether the speaker trailed off,
     * hesitated or broke off, which a dash is not. A grapheme, U+2026, in
     * the last line. */
    const int ellipsis = j < g->n && g->cut[j] && !punctuated(g->w[j - 1].text);
    const int64_t c = width(g, i, j) + ellipsis;
    int64_t layout = NEVER;
    int32_t line2 = 0;
    if (c <= LINE)
        layout = 2 * sq(max64(0, c - LINE_EASY));
    else {
        /* the best break: the least by (cost, |L1 - L2|, L1 > L2, k), k
         * rising so that a full tie keeps the first */
        int64_t key[3] = { 0 };
        for (int32_t k = i + 1; k < j; k++) {
            const int64_t l1 = width(g, i, k), l2 = c - l1 - 1;
            if (l1 > LINE || l2 > LINE)
                continue;
            const int64_t diff = l1 > l2 ? l1 - l2 : l2 - l1;
            const int widows = (k - i == 1 && l1 < 12) + (j - k == 1 && l2 < 12);
            const int64_t cost = g->b[k] + 2 * (sq(max64(0, l1 - LINE_EASY)) + sq(max64(0, l2 - LINE_EASY))) +
                                 8 * widows + cdiv(diff, 4) + 2 * (l1 > l2);
            if (layout == NEVER || cost < key[0] || (cost == key[0] && (diff < key[1] || (diff == key[1] && (l1 > l2) < key[2])))) {
                key[0] = cost;
                key[1] = diff;
                key[2] = l1 > l2;
                layout = cost;
                line2 = k;
            }
        }
    }
    int forced = 0;
    if (layout == NEVER) {
        if (!whole)
            return r;
        /* too wide for any layout: the break that leaves the longer line
         * shortest, or one word on its line */
        forced = 1;
        layout = 0;
        int64_t best = NEVER;
        for (int32_t k = i + 1; k < j; k++) {
            int64_t l1 = width(g, i, k), l2 = c - l1 - 1, m = l1 > l2 ? l1 : l2;
            if (m < best) {
                best = m;
                line2 = k;
            }
        }
    }

    const int64_t s = g->s[i], span = a - s;
    const int64_t next = j < g->n ? g->s[j] : NEVER;   /* the gap to the next cue is 0 */
    const int64_t tail = max64(TAIL_MAX_MS, min64(TAIL_SHORT_MS, max64(0, MIN_MS - span)));
    int64_t cap = min64(min64(a + tail, next), min64(g->media_end, s + MAX_MS));
    if (cap < a) {
        if (!whole)
            return r;
        forced = 1;   /* said over more than MAX_MS: shown as long as it is said */
        cap = min64(min64(a + tail, next), g->media_end);
    }
    const int64_t target = cdiv(1000 * c, CPS_TARGET);
    int64_t end = min64(max64(max64(a + TAIL_MS, s + MIN_MS), s + target), cap);
    if (j < g->n && next - end > 0 && next - end < CLOSE_MS && next <= cap)
        end = next;
    const int64_t d = end - s;

    const int64_t x = max64(0, 1000 * c - CPS_MAX * d), f = max64(0, MIN_MS - d);
    r.v = cdiv(x * x, (int64_t)CPS_MAX * CPS_MAX * d) + 4 * cdiv(f * f, MIN_MS) + (forced ? FORCED : 0);
    r.q = 40 + (j == g->n || g->cut[j] ? 0 : g->b[j]) + (g->csum[j - 1] - g->csum[i]) + layout +
          4 * cdiv(max64(0, target - d), 100) + 4 * cdiv(max64(0, EASY_MIN_MS - d), 100) +
          2 * cdiv(max64(0, d - EASY_MAX_MS), 100);
    r.ok = 1;
    r.line2 = line2;
    r.chars = (int32_t)c;
    r.end = end;
    r.flags = (1000 * c > CPS_MAX * d ? PK_CUE_FAST : 0) | (d < MIN_MS ? PK_CUE_SHORT : 0) | (forced ? PK_CUE_FORCED : 0) |
              (ellipsis ? PK_CUE_PAUSE : 0);
    return r;
}

/* The words' boundaries: which may end a cue, which must, and what cutting
 * at or crossing each costs (the spec's 4.1 to 4.3 and 5.3). */
static void boundaries(graph *g, int english) {
    const int32_t n = g->n;
    int64_t reach = g->e[0];   /* the latest end of the words before boundary k */
    g->csum[0] = 0;
    g->b[0] = 0;
    for (int32_t k = 1; k < n; k++) {
        const char *before = g->w[k - 1].text, *after = g->w[k].text;
        g->safe[k] = reach <= g->s[k];
        const int64_t pause = g->safe[k] ? g->s[k] - reach : 0;
        g->cut[k] = g->safe[k] && pause >= PAUSE_CUT;
        const int m = mark(before, english);
        char lex[40];
        int base = m == P_TERM ? 0 : m == P_MAJOR ? 10 : m == P_COMMA ? 20 : 60;
        if (base == 60 && english) {
            lexical(after, lex, sizeof lex);
            if (listed(PHRASE_STARTERS, lex)) {
                base = 25;
                if (!strcmp(lex, "to")) {
                    lexical(before, lex, sizeof lex);
                    if (listed(SEMI_MODALS, lex))
                        base = 60;
                }
            }
        }
        int bond = 0;
        if (english && !punctuated(before)) {
            lexical(before, lex, sizeof lex);
            bond = listed(FUNCTION_WORDS, lex);
        }
        const int bonus = pause >= PAUSE_STRONG ? 50 : pause >= PAUSE_WEAK ? 20 : 0;
        g->b[k] = (base > bonus ? base - bonus : 0) + 300 * bond;
        const int cross = (m == P_TERM ? 60 : m == P_MAJOR ? 15 : 0) +
                          (pause >= PAUSE_STRONG ? 80 : pause >= PAUSE_WEAK ? 30 : 0);
        g->csum[k] = g->csum[k - 1] + cross;
        reach = max64(reach, g->e[k]);
    }
}

/* g's words made ready, its arrays allocated: their times made sound, as
 * the spec would have its input be (3.3) - starts in order, each word at
 * least a millisecond long - their widths and their boundaries. */
static void prepare(graph *g, int english, int64_t media_end) {
    int64_t last = 0;
    g->gsum[0] = 0;
    for (int32_t i = 0; i < g->n; i++) {
        const pk_capword *w = &g->w[i];
        g->s[i] = i ? max64(w->start, g->s[i - 1]) : max64(w->start, 0);
        g->e[i] = max64(w->end, g->s[i] + 1);
        last = max64(last, g->e[i]);
        g->gsum[i + 1] = g->gsum[i] + graphemes(w->text ? w->text : "");
    }
    g->media_end = max64(media_end, last);   /* unknown or short: the last word's end bounds the last cue */
    boundaries(g, english);
}

/* With equal scores and counts, the path whose cues end earlier wins: the
 * ends on the best paths to p and to q, compared from the first (the
 * spec's 8.4). The paths are as long as each other. */
static int earlier(const int32_t *parent, int32_t p, int32_t q, int32_t *tp, int32_t *tq) {
    int32_t np = 0, nq = 0;
    for (int32_t v = p; v > 0; v = parent[v])
        tp[np++] = v;
    for (int32_t v = q; v > 0; v = parent[v])
        tq[nq++] = v;
    for (int32_t k = 1; k <= np && k <= nq; k++)
        if (tp[np - k] != tq[nq - k])
            return tp[np - k] < tq[nq - k];
    return np < nq;
}

long pk_captions(const pk_capword *w, size_t count, int english, int64_t media_end, pk_cue *out) {
    if (count == 0)
        return 0;
    if (count > INT32_MAX / 2)
        return -1;
    const int32_t n = (int32_t)count;
    const size_t m = (size_t)n + 1;
    graph g = { .n = n, .w = w };
    int64_t *v = NULL, *q = NULL, *end = NULL;
    int32_t *cues = NULL, *parent = NULL, *line2 = NULL, *chars = NULL, *flags = NULL, *tp = NULL, *tq = NULL;
    g.s = malloc(m * sizeof *g.s);
    g.e = malloc(m * sizeof *g.e);
    g.gsum = malloc(m * sizeof *g.gsum);
    g.b = malloc(m * sizeof *g.b);
    g.csum = malloc(m * sizeof *g.csum);
    g.safe = calloc(m, 1);
    g.cut = calloc(m, 1);
    v = malloc(m * sizeof *v);
    q = malloc(m * sizeof *q);
    end = malloc(m * sizeof *end);
    cues = malloc(m * sizeof *cues);
    parent = malloc(m * sizeof *parent);
    line2 = malloc(m * sizeof *line2);
    chars = malloc(m * sizeof *chars);
    flags = malloc(m * sizeof *flags);
    tp = malloc(m * sizeof *tp);
    tq = malloc(m * sizeof *tq);
    long result = -1;
    if (!g.s || !g.e || !g.gsum || !g.b || !g.csum || !g.safe || !g.cut || !v || !q || !end || !cues || !parent ||
        !line2 || !chars || !flags || !tp || !tq)
        goto done;

    prepare(&g, english, media_end);

    /* the shortest path, from boundary 0 to n (the spec's 8.4) */
    for (int32_t j = 0; j <= n; j++)
        parent[j] = -1;
    v[0] = q[0] = 0;
    cues[0] = 0;
    parent[0] = 0;
    for (int32_t i = 0; i < n; i++) {
        if (parent[i] < 0)
            continue;
        int64_t a = 0;   /* when the words [i, j) have all been said */
        int whole = 1;
        for (int32_t j = i + 1; j <= n; j++) {
            if (j > i + 1 && g.safe[j - 1])
                whole = 0;   /* boundary j - 1, inside, could end a cue */
            a = max64(a, g.e[j - 1]);
            /* past these, every longer run fails too, and is no run of one group */
            if (!whole && (width(&g, i, j) > 2 * LINE + 1 || a - g.s[i] > MAX_MS))
                break;
            if (j == n || g.safe[j]) {
                edge ed = cue(&g, i, j, a, whole);
                if (ed.ok) {
                    const int64_t nv = v[i] + ed.v, nq = q[i] + ed.q;
                    const int32_t nc = cues[i] + 1;
                    int better = parent[j] < 0 || nv < v[j] || (nv == v[j] && nq < q[j]) ||
                                 (nv == v[j] && nq == q[j] && nc < cues[j]);
                    if (!better && nv == v[j] && nq == q[j] && nc == cues[j])
                        better = earlier(parent, i, parent[j], tp, tq);
                    if (better) {
                        v[j] = nv;
                        q[j] = nq;
                        cues[j] = nc;
                        parent[j] = i;
                        line2[j] = ed.line2;
                        chars[j] = ed.chars;
                        flags[j] = ed.flags;
                        end[j] = ed.end;
                    }
                }
            }
            if (j < n && g.cut[j])
                break;   /* no cue spans a long pause */
        }
    }

    /* every word alone, or its run of one group alone, is a cue, if forced:
     * there is always a path */
    const int32_t nc = cues[n];
    for (int32_t j = n, k = nc - 1; j > 0; j = parent[j], k--) {
        const int32_t i = parent[j];
        out[k].first = i;
        out[k].n = j - i;
        out[k].line2 = line2[j];
        out[k].chars = chars[j];
        out[k].flags = flags[j];
        out[k].start = g.s[i];
        out[k].end = end[j];
    }
    result = nc;
done:
    free(g.s);
    free(g.e);
    free(g.gsum);
    free(g.b);
    free(g.csum);
    free(g.safe);
    free(g.cut);
    free(v);
    free(q);
    free(end);
    free(cues);
    free(parent);
    free(line2);
    free(chars);
    free(flags);
    free(tp);
    free(tq);
    return result;
}
