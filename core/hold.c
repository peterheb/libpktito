/* Short speakers held out of a diarizer's numbers (docs/SHORT_SPEAKERS.md).
 *
 * Nemotron 3 Diarization has eight numbers and gives one to every voice it
 * hears, so a recording that opens on a montage, or carries ads read by
 * voices heard for ten seconds each, has none left for a guest who comes
 * late. Here a voice with under a minute of speech that has gone quiet is
 * cut out of what the diarizer hears: the diarizer goes back to its state
 * before the voice's first word (pk_diar_load) and hears the recording
 * again from there without it. At the end the voices still held are played
 * after the recording and take what numbers are left.
 *
 * With numbers to spare the model sometimes gives one voice two: a host
 * keeps a number for forty minutes and takes another for a later stretch.
 * So a last check: the diarizer, having forgotten the recording, hears
 * clips of every speaker with a minute alone, and two numbers it then
 * calls one speaker, asked five times on different clips, are joined.
 *
 * This is processing on top of the model, as pk_asr_transcribe's retry is:
 * pk_diar_run's rows are NeMo's, these leave them on purpose, and
 * tools/shortspk/rewind.py is the rule's reference. That script has no
 * state to go back to and hears each cut recording from its start; going
 * back is the same computation, so the rows here are its rows bit for bit.
 * Every count is of rows, in integers.
 *
 * A row is `kept` while it is in what the diarizer hears, and the kept rows
 * in order are the stream: at[i] is the recording's row that row i of the
 * stream is. The diarizer's chunks (steps: 27.2 s) fall on the stream.
 */
#include "model.h"
#include "pktito.h"

#include <stdlib.h>
#include <string.h>

/* The rule's constants, in seconds: tools/shortspk/rewind.py's, chosen by
 * hand on four podcasts before docs/SHORT_SPEAKERS.md's 100 were run. */
#define MIN_S    60     /* a speaker with this much speech keeps a number */
#define QUIET_S  120    /* unheard this long, a shorter one is held */
#define GROUP_S  60     /* and with them, the others unheard this long */
#define NEED     7      /* only once this many numbers are in use */
#define BUDGET_S 300    /* heard again, at most: a quarter of the recording so far and this */
#define CLIP_S   4      /* of a held speaker, to know them again */
#define PAD_CS   30     /* a held stretch: 0.3 s each side */
#define GAP_S    2      /* and pauses under this filled */

/* The join check's (docs/SHORT_SPEAKERS.md, "The join check"): one trial
 * joined a pair a person heard as two speakers, and of eight trials of
 * that pair one or three in each order said one speaker; all of four did
 * not once */
#define APART_S  2      /* a speaker's clips: this far apart */
#define CLIPS    8      /* this many, spread over the recording */
#define LEAST    6      /* and a speaker with fewer is not asked about */
#define TRIALS   4      /* of a pair, on three clips of each: all must say one speaker */

#define PIECE 2560      /* samples fed at a time: under a step, so a feed completes one chunk at most */

/* A speaker held at one step: the rows cut for them, and their probability
 * in each as the diarizer gave it before the cut */
typedef struct {
    int32_t  id;        /* the voice: stretches of one id are one speaker, known again */
    int32_t  speech;    /* rows of speech when held */
    int32_t  n;
    int32_t *row;       /* the recording's rows, ascending */
    float   *prob;
    int32_t  speaker;   /* at the end: their column, or -1 */
} stretch;

struct pk_hold {
    pk_diar *d;
    pk_audio x;
    int64_t  n;                 /* the recording's samples, once it has ended; INT64_MAX till then */
    int      ended;
    int32_t  hop, up, stack;    /* the model's: for its rows from its samples */
    int32_t  cap;               /* the recording's rows there is room for */
    size_t   n_state;           /* and the states */
    size_t  S;
    int64_t R;                  /* samples a row */
    int32_t sec;                /* rows a second */
    int32_t step, ctx;          /* rows a step, and after it that its chunk hears */
    int32_t T;                  /* the recording's rows: those heard of so far till it ends */

    int32_t *at, nk;            /* the stream */
    float   *A;                 /* its rows, [T][S] */
    int32_t  have;              /* how many of them the diarizer has given */
    int64_t  fed;               /* the stream's samples fed */
    int      finished;
    int      away;              /* the diarizer has heard something that is not the stream */

    void   **state;             /* state[j]: after the chunk that ends at row j step */
    void    *scratch;
    float   *piece;

    stretch *held;
    size_t   n_held, cap_held;
    int32_t  ids;
    int32_t *first_of;          /* [id]: the first row of the speaker an id was known as, who then
                                 * had a minute and kept the number; -1 while the id is held */
    int32_t *settled;           /* first seconds of speakers who are past holding */
    size_t   n_settled;
    int64_t  again;             /* rows heard again in rewinds */
    int64_t  extra;             /* and in the clips' and the end's runs */
    int64_t  L;                 /* the samples of the recording's last row, R or under */

    /* a step's look at the speakers */
    int32_t *speech, *first, *last, *known;
    uint8_t *who, *mask, *gone;
    int32_t *list;              /* a run's stream: at's rows, and -1 for a row of silence */
    float   *q;                 /* and its rows */
    size_t   cap_list;

    /* pk_hold_to's */
    int32_t  b;                 /* the next step's end to look at */
    int      heard;             /* the recording heard through */
    int32_t  given;             /* the recording's rows given that a step has looked at */
    int64_t  dirty;             /* the first of the recording's rows changed since the last call */
    int32_t  shown;             /* the rows the last call gave */
    float   *P;                 /* the recording's rows, [cap][S] */
};
typedef struct pk_hold hold;

/* ------------------------------------------------------------ hearing --- */

/* Samples [s, s + n) of the stream whose row i is the recording's row
 * list[i - base], or silence where that is -1, and past the recording's
 * end (its last row may be short of R samples) */
static int samples(const hold *h, const int32_t *list, int64_t base, int64_t s, float *out, size_t n) {
    while (n) {
        const int64_t i = s / h->R - base, off = s % h->R;
        int64_t run = 1;                        /* rows that follow one another in the recording */
        if (list[i] >= 0)
            while ((size_t)(run * h->R - off) < n && list[i + run] == list[i] + run)
                run++;
        const size_t k = (size_t)(run * h->R - off) < n ? (size_t)(run * h->R - off) : n;
        const int64_t at = (int64_t)list[i] * h->R + off;
        const size_t in = list[i] < 0 || at >= h->n ? 0 : (int64_t)k < h->n - at ? k : (size_t)(h->n - at);
        if (in && h->x.read(h->x.arg, at, out, in))
            return -1;
        memset(out + in, 0, (k - in) * sizeof *out);
        out += k;
        s += (int64_t)k;
        n -= k;
    }
    return 0;
}

/* The samples in a stream of n rows whose last is the recording's row
 * `last`: R a row, but the recording's last row ends where the recording
 * does */
static int64_t length(const hold *h, int64_t n, int32_t last) {
    return n * h->R - (n && last == h->T - 1 ? h->R - h->L : 0);
}

typedef struct {
    float  *p;
    int64_t first, cap;         /* p holds rows [first, first + cap) */
    size_t  S;
    int64_t have;
} sink;

static int rows_in(void *arg, int64_t row, size_t n, const float *p) {
    sink *k = arg;
    for (size_t i = 0; i < n; i++)
        if (row + (int64_t)i >= k->first && row + (int64_t)i < k->first + k->cap)
            memcpy(k->p + (size_t)(row + (int64_t)i - k->first) * k->S, p + i * k->S, k->S * sizeof *p);
    k->have = row + (int64_t)n;
    return 0;
}

/* The diarizer as it was after the chunk that ends at row e, a step's
 * multiple: the sample to feed from */
static int64_t back_to(hold *h, int32_t e) {
    if (!e) {
        pk_diar_reset(h->d);
        return 0;
    }
    return pk_diar_load(h->d, h->state[e / h->step]);
}

/* The last step at or before e whose state is kept (keep): 0, the start,
 * if none is */
static int32_t kept_to(const hold *h, int32_t e) {
    while (e && !h->state[e / h->step])
        e -= h->step;
    return e;
}

/* Room in h->list and h->q for a run of n rows */
static int room(hold *h, size_t n) {
    if (n <= h->cap_list)
        return 0;
    int32_t *list = realloc(h->list, n * sizeof *list);
    if (list)
        h->list = list;
    float *q = realloc(h->q, n * h->S * sizeof *q);
    if (q)
        h->q = q;
    if (!list || !q)
        return -1;
    h->cap_list = n;
    return 0;
}

/* A stream that is the kept rows to row e + n0 and then goes on its own
 * way, heard whole from the state at e: rows [e, e + n) into h->q. list
 * holds its rows from `base` on, n0 + n of them past e in all. */
static int hear_whole(hold *h, int32_t e, const int32_t *list, int64_t base, int64_t n) {
    sink k = { h->q, e, n, h->S, e };
    const int64_t end = length(h, e + n, n ? list[e + n - 1 - base] : -1);
    int rc = 0;
    h->away = 1;
    const int64_t from = back_to(h, e);
    if (from < 0)
        return -1;
    for (int64_t s = from; s < end && !rc;) {
        const size_t c = end - s < PIECE ? (size_t)(end - s) : PIECE;
        if (samples(h, list, base, s, h->piece, c))
            return -1;
        rc = pk_diar_feed(h->d, h->piece, c, rows_in, &k);
        s += (int64_t)c;
    }
    if (!rc)
        rc = pk_diar_finish(h->d, rows_in, &k);
    h->extra += n;
    return rc || k.have != e + n ? -1 : 0;
}

/* Room for the recording's first `rows` rows, and the stream on through
 * them: rows the recording has past the stream's last are kept. */
static int grow(hold *h, int32_t rows) {
    if (rows > h->cap) {
        int64_t want = h->cap ? 2 * (int64_t)h->cap : 4096;
        const int32_t cap = (int32_t)(want < rows ? rows : want > INT32_MAX / 8 ? INT32_MAX / 8 : want);
        const size_t S = h->S, n_state = (size_t)cap / (size_t)h->step + 2;
        int32_t *at = realloc(h->at, ((size_t)cap + 1) * sizeof *at);
        if (at)
            h->at = at;
        float *A = realloc(h->A, ((size_t)cap + 1) * S * sizeof *A);
        if (A)
            h->A = A;
        float *P = realloc(h->P, ((size_t)cap + 1) * S * sizeof *P);
        if (P)
            h->P = P;
        uint8_t *mask = realloc(h->mask, (size_t)cap + 1);
        if (mask)
            h->mask = mask;
        uint8_t *gone = realloc(h->gone, (size_t)cap + 1);
        if (gone)
            h->gone = gone;
        int32_t *settled = realloc(h->settled, ((size_t)cap / (size_t)h->sec + 2) * sizeof *settled);
        if (settled)
            h->settled = settled;
        void **state = realloc(h->state, n_state * sizeof *state);
        if (state) {
            for (size_t j = h->n_state; j < n_state; j++)
                state[j] = NULL;
            h->state = state;
            h->n_state = n_state;
        }
        if (!at || !A || !P || !mask || !gone || !settled || !state)
            return -1;
        h->cap = cap;
    }
    while (h->T < rows)
        h->at[h->nk++] = h->T++;
    return 0;
}

/* The recording has ended, n samples long: pk_diar_run's rows, the last of
 * which may be short of R samples, or have a part of a hop over, which the
 * mel does not frame */
static int the_end(hold *h, int64_t n) {
    if (n / h->R > INT32_MAX / 8 - 1)           /* rows counted in int32 with room to spare */
        return -1;
    const int32_t T = (int32_t)((n / h->hop * h->up + h->stack - 1) / h->stack);
    if (grow(h, T))
        return -1;
    h->n = n;
    h->ended = 1;
    h->L = T && n - (int64_t)(T - 1) * h->R < h->R ? n - (int64_t)(T - 1) * h->R : h->R;
    return 0;
}

/* The recording read to its row `rows` at least, or to its end: a
 * recording still being read is waited for (pk_audio's wait) */
static int more(hold *h, int64_t rows) {
    if (h->ended)
        return 0;
    const int64_t want = rows * h->R, n = h->x.wait(h->x.arg, want);
    if (n < 0)
        return -1;
    if (n < want)
        return the_end(h, n);
    if (n / h->R > INT32_MAX / 8 - 1)
        return -1;
    return grow(h, (int32_t)(n / h->R));
}

/* The stream heard on until its rows reach b, or its end: each step's
 * state kept as it is passed. A recording still being read is read on as
 * the stream needs; the rows are the same whatever the pieces it is fed
 * in (pk_diar_feed). */
static int hear_to(hold *h, int32_t b) {
    sink k = { h->A, 0, h->cap, h->S, h->have };
    for (;;) {
        const int64_t end = length(h, h->nk, h->nk ? h->at[h->nk - 1] : -1);
        while (h->have < b && h->fed < end) {
            const size_t c = end - h->fed < PIECE ? (size_t)(end - h->fed) : PIECE;
            if (samples(h, h->at, 0, h->fed, h->piece, c) || pk_diar_feed(h->d, h->piece, c, rows_in, &k))
                return -1;
            h->fed += (int64_t)c;
            if (k.have > h->have) {             /* a chunk: the state after it */
                h->have = (int32_t)k.have;
                const size_t j = (size_t)(h->have / h->step), z = pk_diar_save(h->d, h->scratch);
                free(h->state[j]);
                if (!(h->state[j] = malloc(z)))
                    return -1;
                memcpy(h->state[j], h->scratch, z);
            }
        }
        if (h->have >= b || h->ended)
            break;
        if (more(h, (int64_t)h->T + h->step))   /* fed all there is: a step more of the recording */
            return -1;
        k.p = h->A;
        k.cap = h->cap;
    }
    if (h->have < b) {                          /* the end, its last chunks with what context there is */
        if (pk_diar_finish(h->d, rows_in, &k) || k.have != h->nk)
            return -1;
        h->have = h->nk;
        h->finished = 1;
    }
    return 0;
}

/* ------------------------------------------------------------- a step --- */

static int said(const hold *h, int32_t r, size_t s) { return h->A[(size_t)r * h->S + s] > 0.5f; }

/* h->mask: speaker s's rows of the stream's first b, each stretch with
 * PAD_CS before and after it, and a pause under GAP_S filled */
static void widen(hold *h, int32_t b, size_t s) {
    const int32_t pad = PAD_CS * h->sec / 100, gap = GAP_S * h->sec;
    int32_t last = -1;
    memset(h->mask, 0, (size_t)b);
    for (int32_t r = 0; r < b;) {
        if (!said(h, r, s)) {
            r++;
            continue;
        }
        int32_t e = r;
        while (e < b && said(h, e, s))
            e++;
        int32_t from = r - pad > 0 ? r - pad : 0;
        const int32_t to = e + pad < b ? e + pad : b;
        if (last >= 0 && from - last < gap)
            from = last;
        memset(h->mask + from, 1, (size_t)(to - from));
        last = to;
        r = e;
    }
}

static int by_row(const void *a, const void *b) {
    const int32_t x = *(const int32_t *)a, y = *(const int32_t *)b;
    return (x > y) - (x < y);
}

/* Every row held for the voice `id`, ascending, to out: how many */
static int32_t rows_of(const hold *h, int32_t id, int32_t *out) {
    int32_t n = 0;
    for (size_t i = 0; i < h->n_held; i++)
        if (h->held[i].id == id) {
            memcpy(out + n, h->held[i].row, (size_t)h->held[i].n * sizeof *out);
            n += h->held[i].n;
        }
    qsort(out, (size_t)n, sizeof *out, by_row);
    return n;
}

/* CLIP_S from the middle of the longest unbroken run of them, the first of
 * equals, moved to out's front: how many */
static int32_t clip_of(const hold *h, int32_t *out, int32_t n) {
    int32_t best = 0, len = 0;
    for (int32_t i = 0; i < n;) {
        int32_t e = i + 1;
        while (e < n && out[e] == out[e - 1] + 1)
            e++;
        if (e - i > len)
            best = i, len = e - i;
        i = e;
    }
    const int32_t clip = CLIP_S * h->sec, m = len > clip ? (len - clip) / 2 : 0, c = len < clip ? len : clip;
    memmove(out, out + best + m, (size_t)c * sizeof *out);
    return c;
}

static int is_settled(const hold *h, int32_t first) {
    for (size_t i = 0; i < h->n_settled; i++)
        if (h->settled[i] == first / h->sec)
            return 1;
    return 0;
}

static void settle(hold *h, int32_t first) {
    if (!is_settled(h, first))                  /* so there are no more of them than seconds */
        h->settled[h->n_settled++] = first / h->sec;
}

/* The column that rows [r0, r1) of h->q (which starts at row e) are mostly
 * in, the first of equals: its rows to *n, every column's to *all, and the
 * rows where anyone speaks to *any */
static size_t lands(const hold *h, int32_t e, int32_t r0, int32_t r1, int32_t *n, int32_t *all, int32_t *any) {
    size_t best = 0;
    *n = *all = *any = 0;
    for (size_t s = 0; s < h->S; s++) {
        int32_t c = 0;
        for (int32_t r = r0; r < r1; r++)
            c += h->q[(size_t)(r - e) * h->S + s] > 0.5f;
        *all += c;
        if (c > *n)
            best = s, *n = c;
    }
    for (int32_t r = r0; r < r1; r++) {
        int a = 0;
        for (size_t s = 0; s < h->S; s++)
            a |= h->q[(size_t)(r - e) * h->S + s] > 0.5f;
        *any += a;
    }
    return best;
}

/* Is one of the speakers about to be held a voice held before?  The
 * diarizer as it was a step ago hears the step again and then, where the
 * recording goes on, a second of silence and a clip of each such voice: a
 * clip that lands in a column about to be held (a second or more of it,
 * and half of what was heard) says they are one speaker. A step ago,
 * because the chunk that ends at b has heard what follows b. */
static int know_again(hold *h, int32_t b) {
    const int32_t e = b - h->step, base = e > 2 ? e - 2 : 0;
    int32_t n = b - base, opened = 0;
    size_t need = (size_t)n;
    for (size_t i = 0; i < h->n_held; i++)      /* each one's rows, before the clip is cut from them */
        need += (size_t)(h->sec + h->held[i].n);
    for (size_t s = 0; s < h->S; s++)
        h->known[s] = 0;
    if (room(h, need))
        return -1;
    memcpy(h->list, h->at + base, (size_t)n * sizeof *h->list);
    for (int32_t id = 1; id <= h->ids; id++) {
        if (h->first_of[id] >= 0)
            continue;
        for (int32_t i = 0; i < h->sec; i++)
            h->list[n++] = -1;
        n += clip_of(h, h->list + n, rows_of(h, id, h->list + n));
        opened++;
    }
    if (!opened)
        return 0;
    if (hear_whole(h, e, h->list, base, base + n - e))
        return -1;
    int32_t r = b;
    for (int32_t id = 1; id <= h->ids; id++) {
        if (h->first_of[id] >= 0)
            continue;
        int32_t len = 0, c, all, any;
        r += h->sec;
        while (r + len < base + n && h->list[r + len - base] >= 0)
            len++;
        const size_t s = lands(h, e, r, r + len, &c, &all, &any);
        if (h->who[s] && !h->known[s] && c >= h->sec && 2 * c >= all)
            h->known[s] = id;
        r += len;
    }
    return 0;
}

/* The speakers at row b of the stream, a step's end: 1 if some were held
 * and the stream cut (*back the row to go back to, *next the next step's
 * end to look at), 0 if none, -1 for want of memory or a failed read. */
static int look(hold *h, int32_t b, int32_t *back, int32_t *next) {
    const size_t S = h->S;
    const int32_t now = h->at[b - 1];
    int32_t used = 0;
    for (size_t s = 0; s < S; s++) {
        h->speech[s] = 0;
        h->first[s] = h->last[s] = -1;
    }
    for (int32_t r = 0; r < b; r++)
        for (size_t s = 0; s < S; s++)
            if (said(h, r, s)) {
                h->speech[s]++;
                h->last[s] = h->at[r];
                if (h->first[s] < 0)
                    h->first[s] = h->at[r];
            }
    int quiet = 0, some = 0;
    for (size_t s = 0; s < S; s++) {
        const int brief = h->speech[s] && h->speech[s] < MIN_S * h->sec && !is_settled(h, h->first[s]);
        used += h->speech[s] > 0;
        quiet |= brief && now - h->last[s] >= QUIET_S * h->sec;
        h->who[s] = brief && now - h->last[s] >= GROUP_S * h->sec;
    }
    if (!quiet || used < NEED)
        return 0;
    if (know_again(h, b))
        return -1;
    /* one known again who has a minute with what was held keeps the number */
    for (size_t s = 0; s < S; s++) {
        if (!h->who[s] || !h->known[s])
            continue;
        int32_t all = h->speech[s];
        for (size_t i = 0; i < h->n_held; i++)
            if (h->held[i].id == h->known[s])
                all += h->held[i].speech;
        if (all >= MIN_S * h->sec) {
            h->first_of[h->known[s]] = h->first[s];
            settle(h, h->first[s]);
            h->who[s] = 0;
        }
    }
    int32_t n_gone = 0, first_gone = -1;
    memset(h->gone, 0, (size_t)b);
    for (size_t s = 0; s < S; s++) {
        if (!h->who[s])
            continue;
        some = 1;
        widen(h, b, s);
        for (int32_t r = 0; r < b; r++)
            h->gone[r] |= h->mask[r];
    }
    if (!some)
        return 0;
    for (int32_t r = 0; r < b; r++)
        if (h->gone[r]) {
            n_gone++;
            if (first_gone < 0)
                first_gone = r;
        }
    /* back to the last step whose chunk heard nothing that goes: it heard
     * ctx rows past its end, and its last frame's window reaches into the
     * row after those */
    int32_t e = first_gone - h->ctx - 1;
    e = e > 0 ? e / h->step * h->step : 0;
    int32_t cost = b - e;
    for (int32_t r = e; r < b; r++)
        cost -= h->gone[r];
    if (4 * (h->again + cost) > (int64_t)now + 4 * BUDGET_S * h->sec) {
        for (size_t s = 0; s < S; s++)          /* too dear: they keep their numbers */
            if (h->who[s])
                settle(h, h->first[s]);
        return 0;
    }
    for (size_t s = 0; s < S; s++) {
        if (!h->who[s])
            continue;
        if (h->n_held == h->cap_held) {
            const size_t cap = h->cap_held ? 2 * h->cap_held : 16;
            stretch *held = realloc(h->held, cap * sizeof *held);
            int32_t *first_of = realloc(h->first_of, (cap + 1) * sizeof *first_of);
            if (held)
                h->held = held;
            if (first_of)
                h->first_of = first_of;
            if (!held || !first_of)
                return -1;
            h->cap_held = cap;
        }
        stretch *t = &h->held[h->n_held];
        widen(h, b, s);
        int32_t n = 0;
        for (int32_t r = 0; r < b; r++)
            n += h->mask[r];
        *t = (stretch){ .id = h->known[s], .speech = h->speech[s], .n = n, .speaker = -1 };
        t->row = malloc((size_t)n * sizeof *t->row);
        t->prob = malloc((size_t)n * sizeof *t->prob);
        if (!t->row || !t->prob) {
            free(t->row);
            free(t->prob);
            return -1;
        }
        h->n_held++;
        if (!t->id) {
            t->id = ++h->ids;
            h->first_of[t->id] = -1;
        }
        n = 0;
        for (int32_t r = 0; r < b; r++)
            if (h->mask[r]) {
                t->row[n] = h->at[r];
                t->prob[n++] = h->A[(size_t)r * S + s];
            }
    }
    h->again += cost;
    /* the stream without them; the next end to look at is the first past
     * where the recording had got to */
    const int32_t idx = b - 1 - n_gone + h->gone[b - 1];
    int32_t n = first_gone;
    for (int32_t r = first_gone; r < h->nk; r++)
        if (r >= b || !h->gone[r])
            h->at[n++] = h->at[r];
    h->nk = n;
    *back = e;
    *next = (idx / h->step + 1) * h->step;
    return 1;
}

/* The states no rewind can reach, dropped (each 0.6 MB; all of them kept
 * were 83 MB an hour, docs/SHORT_SPEAKERS.md). After look at row b found
 * none to hold, a rewind goes back to before the first row of a speaker
 * held then (look: their first row, less PAD_CS and the chunk's context),
 * and that speaker is one heard now and short of a minute, or one not yet
 * heard, whose rows are past b. And it costs, within the budget, what it
 * hears again, rows (b' - e) less the rows cut: the cut rows are, of those
 * heard now, the widened rows of the speakers short of a minute, and at
 * most all of those past b; so e is no further back than b less the budget
 * at its most (the whole recording's quarter) less those widened rows. A
 * state before both bounds is dropped. A speaker's rows past a rewind are
 * heard again and may come out otherwise; should a rewind then reach a
 * dropped state, it goes back to the last kept before it and hears the
 * stream from there, which gives the same rows (pk_diar_load), only later.
 * h->mask and h->gone are free after look. */
static void keep(hold *h, int32_t b) {
    const int32_t pad = PAD_CS * h->sec / 100, margin = (GAP_S + 1) * h->sec;
    int32_t lo = b, gone = 0;
    memset(h->gone, 0, (size_t)b);
    for (size_t s = 0; s < h->S; s++) {
        if (!h->speech[s] || h->speech[s] >= MIN_S * h->sec || is_settled(h, h->first[s]))
            continue;
        int32_t r = 0;                          /* the first in the stream: at is ascending */
        while (h->at[r] < h->first[s])
            r++;
        if (r < lo)
            lo = r;
        widen(h, b, s);
        for (int32_t k = 0; k < b; k++)
            h->gone[k] |= h->mask[k];
        gone += margin;                         /* a pause bridged to a stretch past b */
    }
    for (int32_t k = 0; k < b; k++)
        gone += h->gone[k];
    const int64_t budget = (int64_t)h->T / 4 + BUDGET_S * h->sec - h->again;
    const int64_t far = (int64_t)b - budget - gone;
    if (far > lo - pad)
        lo = (int32_t)far;
    else
        lo -= pad;
    lo -= h->ctx + 1;
    for (int32_t j = 1; lo > 0 && j < lo / h->step; j++) {
        free(h->state[j]);
        h->state[j] = NULL;
    }
}

/* ------------------------------------------------------------ the end --- */

static int by_speech(const void *a, const void *b) {
    const int32_t *x = a, *y = b;               /* { speech, order, id }: longest first, then as held */
    return x[0] != y[0] ? (x[0] < y[0]) - (x[0] > y[0]) : (x[1] > y[1]) - (x[1] < y[1]);
}

/* The speakers still held, heard after the recording, longest first, a
 * second of silence before each: one whose rows land in a column (a second
 * or more of them, and a fifth of the rows where anyone speaks) is that
 * speaker, and the others are no one's. The recording's last steps are
 * heard again before them, from the last state whose chunk the end did not
 * reach. */
static int place(hold *h) {
    const size_t S = h->S;
    for (size_t s = 0; s < S; s++) {
        h->first[s] = -1;
        for (int32_t r = 0; r < h->nk && h->first[s] < 0; r++)
            if (said(h, r, s))
                h->first[s] = h->at[r];
    }
    /* one who kept a number: the speaker whose first word is nearest theirs */
    for (size_t i = 0; i < h->n_held; i++) {
        const int32_t f = h->first_of[h->held[i].id];
        int64_t best = INT64_MAX;
        for (size_t s = 0; s < S && f >= 0; s++) {
            const int64_t d = h->first[s] < 0 ? INT64_MAX - 1 : llabs((int64_t)h->first[s] - f);
            if (d < best)
                best = d, h->held[i].speaker = (int32_t)s;
        }
    }
    int32_t (*order)[3] = malloc(((size_t)h->ids + 1) * sizeof *order);
    if (!order)
        return -1;
    int32_t n_open = 0;
    for (int32_t id = 1; id <= h->ids; id++) {
        if (h->first_of[id] >= 0)
            continue;
        order[n_open][0] = 0;
        order[n_open][1] = n_open;
        order[n_open][2] = id;
        for (size_t i = 0; i < h->n_held; i++)
            if (h->held[i].id == id)
                order[n_open][0] += h->held[i].speech;
        n_open++;
    }
    if (!n_open) {
        free(order);
        return 0;
    }
    qsort(order, (size_t)n_open, sizeof *order, by_speech);
    int32_t e = h->nk - h->ctx - 1;
    e = e > 0 ? e / h->step * h->step : 0;
    e = kept_to(h, e);
    const int32_t base = e > 2 ? e - 2 : 0;
    int32_t n = h->nk - base;
    size_t need = (size_t)n;
    for (size_t i = 0; i < h->n_held; i++)
        need += (size_t)(h->sec + h->held[i].n);
    if (room(h, need)) {
        free(order);
        return -1;
    }
    memcpy(h->list, h->at + base, (size_t)n * sizeof *h->list);
    for (int32_t i = 0; i < n_open; i++) {
        for (int32_t k = 0; k < h->sec; k++)
            h->list[n++] = -1;
        n += rows_of(h, order[i][2], h->list + n);
    }
    int rc = hear_whole(h, e, h->list, base, base + n - e);
    for (int32_t i = 0, r = h->nk; i < n_open && !rc; i++) {
        int32_t len = 0, c, all, any;
        r += h->sec;
        while (r + len < base + n && h->list[r + len - base] >= 0)
            len++;
        const size_t s = lands(h, e, r, r + len, &c, &all, &any);
        if (c >= h->sec && 5 * c >= (any > h->sec ? any : h->sec))
            for (size_t k = 0; k < h->n_held; k++)
                if (h->held[k].id == order[i][2])
                    h->held[k].speaker = (int32_t)s;
        r += len;
    }
    free(order);
    return rc;
}

/* ----------------------------------------------------------- the join --- */

/* Where speaker s's clips start, the recording's rows, to out (room for
 * nk / (CLIP_S + APART_S) sec + 1): each CLIP_S of rows that follow one
 * another in the recording and are s's alone, APART_S or more after the
 * last; if over CLIPS of them, CLIPS spread over the recording. 0 for a
 * speaker with under MIN_S alone. */
static int32_t clips_of(const hold *h, size_t s, int32_t *out) {
    const int32_t clip = CLIP_S * h->sec;
    int32_t n = 0, run = 0, last = -1, alone = 0;
    int was = 0;
    for (int32_t i = 0; i < h->nk; i++) {
        int is = said(h, i, s);
        for (size_t k = 0; k < h->S && is; k++)
            is = k == s || !said(h, i, k);
        alone += is;
        run = i && is && was && h->at[i] == h->at[i - 1] + 1 ? run + 1 : 0;
        if (run >= clip && (last < 0 || i - last >= clip + APART_S * h->sec)) {
            out[n++] = h->at[i] - clip;
            last = i;
        }
        was = is;
    }
    if (alone < MIN_S * h->sec)
        return 0;
    if (n > CLIPS) {
        for (int32_t k = 0; k < CLIPS; k++)
            out[k] = out[(int64_t)k * (n - 1) / (CLIPS - 1)];
        n = CLIPS;
    }
    return n;
}

/* A diarizer that has heard nothing hears two or more groups of clips,
 * half a second of silence after each clip: the column each group is
 * mostly in, to col, or -1 for one it hears under a second of. start[g]
 * holds group g's n[g] clips. */
static int fresh(hold *h, int32_t *const *start, const int32_t *n, size_t groups, int32_t *col) {
    const int32_t clip = CLIP_S * h->sec, pause = h->sec / 2;
    size_t rows = 0;
    for (size_t g = 0; g < groups; g++)
        rows += (size_t)n[g] * (size_t)(clip + pause);
    if (room(h, rows))
        return -1;
    int32_t at = 0;
    for (size_t g = 0; g < groups; g++)
        for (int32_t i = 0; i < n[g]; i++) {
            for (int32_t r = 0; r < clip; r++)
                h->list[at++] = start[g][i] + r;
            for (int32_t r = 0; r < pause; r++)
                h->list[at++] = -1;
        }
    if (hear_whole(h, 0, h->list, 0, at))
        return -1;
    at = 0;
    for (size_t g = 0; g < groups; g++) {
        const int32_t len = n[g] * (clip + pause);
        int32_t c, all, any;
        const size_t s = lands(h, 0, at, at + len, &c, &all, &any);
        col[g] = c >= h->sec ? (int32_t)s : -1;
        at += len;
    }
    return 0;
}

/* to[s]: the speaker that s is, itself or an earlier number. Every speaker
 * with a minute alone and LEAST clips is heard by a diarizer that has
 * heard nothing, three clips of each, one speaker after another; two it
 * puts in one column are heard as a pair TRIALS times, on three other
 * clips of each, in one order and the other by turns, and are one speaker
 * if it says so every time. */
static int join(hold *h, int32_t *to) {
    const size_t S = h->S;
    int32_t *clip[PK_MAX_SPK] = { 0 }, n[PK_MAX_SPK], col[PK_MAX_SPK], of[PK_MAX_SPK];
    int32_t *g[PK_MAX_SPK], gn[PK_MAX_SPK], three[2][3];
    size_t K = 0;
    int rc = -1;
    for (size_t s = 0; s < S; s++)
        to[s] = (int32_t)s;
    for (size_t s = 0; s < S; s++) {
        if (!(clip[s] = malloc(((size_t)h->nk / (size_t)((CLIP_S + APART_S) * h->sec) + 2) * sizeof **clip)))
            goto done;
        n[s] = clips_of(h, s, clip[s]);
        if (n[s] >= LEAST)
            of[K++] = (int32_t)s;
    }
    rc = 0;
    if (K < 2)
        goto done;
    for (size_t k = 0; k < K; k++) {            /* the first, fourth and seventh clip */
        const size_t s = (size_t)of[k];
        g[k] = three[0];                        /* any room: the clips are copied below */
        gn[k] = (n[s] + 2) / 3 < 3 ? (n[s] + 2) / 3 : 3;
    }
    int32_t *firsts = malloc(K * 3 * sizeof *firsts);
    if (!firsts) {
        rc = -1;
        goto done;
    }
    for (size_t k = 0; k < K; k++) {
        g[k] = firsts + 3 * k;
        for (int32_t i = 0; i < gn[k]; i++)
            g[k][i] = clip[of[k]][3 * i];
    }
    rc = fresh(h, g, gn, K, col);
    free(firsts);
    for (size_t x = 0; x < K && !rc; x++)
        for (size_t y = x + 1; y < K && !rc; y++) {
            if (col[x] < 0 || col[x] != col[y])
                continue;
            int one = 1;
            for (int32_t t = 0; t < TRIALS && one && !rc; t++) {
                int32_t c[2];
                for (int w = 0; w < 2; w++) {   /* by turns x first, and y */
                    const size_t s = (size_t)of[(w ^ (t & 1)) ? y : x];
                    for (int32_t i = 0; i < 3; i++)
                        three[w][i] = clip[s][(t + 1 + i * (n[s] / 3)) % n[s]];
                    g[w] = three[w];
                    gn[w] = 3;
                }
                rc = fresh(h, g, gn, 2, c);
                one = !rc && c[0] >= 0 && c[0] == c[1];
            }
            if (one && !rc) {                   /* the later number is the earlier */
                const int32_t a = to[of[x]] < to[of[y]] ? to[of[x]] : to[of[y]];
                const int32_t b = to[of[x]] < to[of[y]] ? to[of[y]] : to[of[x]];
                for (size_t s = 0; s < S; s++)
                    if (to[s] == b)
                        to[s] = a;
            }
        }
done:
    for (size_t s = 0; s < S; s++)
        free(clip[s]);
    return rc;
}

/* ------------------------------------------------------------- public --- */

void pk_held_free(pk_held *o) {
    if (!o)
        return;
    free(o->p);
    free(o->held);
    free(o->joined);
    *o = (pk_held){ 0 };
}

void pk_hold_free(pk_hold *h) {
    if (!h)
        return;
    pk_diar_reset(h->d);
    for (size_t i = 0; i < h->n_held; i++) {
        free(h->held[i].row);
        free(h->held[i].prob);
    }
    for (size_t j = 0; h->state && j < h->n_state; j++)
        free(h->state[j]);
    free(h->state), free(h->scratch), free(h->piece), free(h->held), free(h->first_of), free(h->settled);
    free(h->at), free(h->A), free(h->speech), free(h->first), free(h->last), free(h->known), free(h->who);
    free(h->mask), free(h->gone), free(h->list), free(h->q), free(h->P);
    free(h);
}

pk_hold *pk_hold_new(pk_diar *d, const pk_model *m, const pk_audio *x) {
    if ((x->n < 0 && !x->wait) || m->dw.stack % m->dw.upsample)  /* a row is a whole number of the mel's hops */
        return NULL;
    hold *h = calloc(1, sizeof *h);
    if (!h)
        return NULL;
    *h = (hold){ .d = d, .x = *x, .n = INT64_MAX, .S = m->dw.n_spk };
    h->hop = (int32_t)m->hop, h->up = (int32_t)m->dw.upsample, h->stack = (int32_t)m->dw.stack;
    h->R = (int64_t)m->hop * m->dw.stack / m->dw.upsample;
    h->sec = (int32_t)(m->sample_rate / h->R);
    h->step = (int32_t)(m->dw.chunk_len * m->dw.upsample);
    h->ctx = (int32_t)((m->dw.chunk_left > m->dw.chunk_right ? m->dw.chunk_left : m->dw.chunk_right) * m->dw.upsample);
    h->L = h->R;
    h->b = h->step;
    h->dirty = INT64_MAX;
    const size_t S = h->S;
    h->scratch = malloc(pk_diar_state_size(d));
    h->piece = malloc(PIECE * sizeof *h->piece);
    h->speech = malloc(S * sizeof *h->speech);
    h->first = malloc(S * sizeof *h->first);
    h->last = malloc(S * sizeof *h->last);
    h->known = malloc(S * sizeof *h->known);
    h->who = malloc(S);
    /* a step longer than the context its chunk hears past it, as look's
     * going back needs; a step of at least PIECE samples, so that a feed
     * hands over one chunk at most and hear_to saves the state after each
     * (Nemotron 3 Diarization's step is 435,200 samples, PIECE 2,560) */
    if (h->step <= h->ctx + 3 || h->R * h->step < PIECE || h->sec < 1 || !h->scratch || !h->piece || !h->speech
        || !h->first || !h->last || !h->known || !h->who || (x->n >= 0 ? the_end(h, x->n) : grow(h, 0))) {
        pk_hold_free(h);
        return NULL;
    }
    pk_diar_reset(d);
    return h;
}

/* A step of the run: the stream heard to h->b and looked at there */
static int step(hold *h) {
    const int32_t b = h->b;
    /* what the step's chunk hears, the context past it with a frame over,
     * read before it is known whether the recording ends first */
    const int64_t need = (int64_t)b + h->ctx + 3 - h->nk;
    if (need > 0 && more(h, (int64_t)h->T + need))
        return -1;
    if (hear_to(h, b < h->nk ? b : h->nk))
        return -1;
    if (b >= h->nk) {                           /* the recording's end */
        h->heard = 1;
        h->given = h->T;
        return 0;
    }
    int32_t e, next;
    const int cut = look(h, b, &e, &next);
    if (cut < 0)
        return -1;
    if (!cut) {
        if (h->away && !h->finished)            /* the diarizer heard clips: back to where it was */
            h->fed = back_to(h, h->have);
        if (h->fed < 0)
            return -1;
        h->away = 0;
        keep(h, b);
        h->given = h->at[b - 1] + 1;
        h->b = b + h->step;
        return 0;
    }
    for (size_t j = (size_t)(e / h->step) + 1; j < h->n_state; j++) {
        free(h->state[j]);
        h->state[j] = NULL;
    }
    /* the recording's rows from the stream's row e on are heard again */
    if (h->at[e] < h->dirty)
        h->dirty = h->at[e];
    if (h->given > h->at[e])
        h->given = h->at[e];
    h->have = kept_to(h, e);                    /* e itself, but for a state keep dropped */
    h->fed = back_to(h, h->have);
    if (h->fed < 0)
        return -1;
    h->finished = h->away = 0;
    h->b = next;
    return 0;
}

int64_t pk_hold_to(pk_hold *h, int64_t want, const float **p, int64_t *from) {
    while (!h->heard && h->given < want)
        if (step(h))
            return -1;
    /* the rows changed since the last call, or given since: the stream's
     * where it was kept, 0 where it was not */
    const size_t S = h->S;
    int64_t f = h->dirty < h->shown ? h->dirty : h->shown;
    if (f > h->given)
        f = h->given;
    memset(h->P + (size_t)f * S, 0, (size_t)(h->given - f) * S * sizeof *h->P);
    int32_t lo = 0, hi = h->have;               /* the stream's first row at or past f */
    while (lo < hi) {
        const int32_t mid = lo + (hi - lo) / 2;
        if (h->at[mid] < f)
            lo = mid + 1;
        else
            hi = mid;
    }
    for (int32_t r = lo; r < h->have && h->at[r] < h->given; r++)
        memcpy(h->P + (size_t)h->at[r] * S, h->A + (size_t)r * S, S * sizeof *h->P);
    h->dirty = INT64_MAX;
    h->shown = h->given;
    *p = h->P;
    *from = f;
    return h->given;
}

int pk_hold_end(pk_hold *h, pk_held *out) {
    *out = (pk_held){ 0 };
    while (!h->heard)
        if (step(h))
            return -1;
    const size_t S = h->S, T = (size_t)h->T;
    int rc = -1;
    if (place(h))
        goto done;
    /* the recording's rows: the stream's where it was kept, and where it
     * was not, each held speaker's probability in the column they took */
    if (!(out->p = calloc((T ? T : 1) * S, sizeof *out->p)))
        goto done;
    for (int32_t r = 0; r < h->nk; r++)
        memcpy(out->p + (size_t)h->at[r] * S, h->A + (size_t)r * S, S * sizeof *out->p);
    size_t n_seg = 0;
    for (size_t i = 0; i < h->n_held; i++) {
        const stretch *t = &h->held[i];
        for (int32_t k = 0; k < t->n; k++) {
            n_seg += !k || t->row[k] != t->row[k - 1] + 1;
            if (t->speaker >= 0 && t->prob[k] > out->p[(size_t)t->row[k] * S + (size_t)t->speaker])
                out->p[(size_t)t->row[k] * S + (size_t)t->speaker] = t->prob[k];
        }
    }
    if (!(out->held = malloc((n_seg ? n_seg : 1) * sizeof *out->held)))
        goto done;
    for (size_t i = 0; i < h->n_held; i++) {
        const stretch *t = &h->held[i];
        for (int32_t k = 0; k < t->n; k++) {
            if (!k || t->row[k] != t->row[k - 1] + 1)
                out->held[out->n_held++] = (pk_spk_seg){ (int64_t)t->row[k] * h->R, 0, t->speaker };
            out->held[out->n_held - 1].end = length(h, (int64_t)t->row[k] + 1, t->row[k]);
        }
    }
    /* one voice on two numbers */
    int32_t to[PK_MAX_SPK];
    if (!(out->joined = malloc(S * sizeof *out->joined)) || join(h, to))
        goto done;
    for (size_t s = 0; s < S; s++) {
        out->joined[s] = to[s];
        if (to[s] == (int32_t)s)
            continue;
        for (size_t r = 0; r < T; r++) {
            float *p = out->p + r * S;
            if (p[s] > p[to[s]])
                p[to[s]] = p[s];
            p[s] = 0;
        }
    }
    for (size_t i = 0; i < out->n_held; i++)
        if (out->held[i].speaker >= 0)
            out->held[i].speaker = to[out->held[i].speaker];
    out->rows = T;
    out->again = (double)(h->again + h->extra) / h->sec;
    rc = 0;
done:
    pk_diar_reset(h->d);
    if (rc)
        pk_held_free(out);
    return rc;
}

int pk_diar_run_held(pk_diar *d, const pk_model *m, const pk_audio *x, pk_held *out) {
    *out = (pk_held){ 0 };
    if (x->n < 0)
        return -1;
    pk_hold *h = pk_hold_new(d, m, x);
    if (!h)
        return -1;
    const int rc = pk_hold_end(h, out);
    pk_hold_free(h);
    return rc;
}
