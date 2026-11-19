/* pk_asr_transcribe: a recording of any length to its transcript, with the
 * whole of the processing pktito does for one - the high-level path, beside
 * pk_asr_run's transcript = f(x). In order:
 *
 *   1. Silero's VAD over the recording (pk_vad_probs_on, its encoder over
 *      the transcriber's threads), and its speech segments (pk_segments),
 *      none longer than a chunk: Silero's own for the retry, and the same
 *      cut at shorter pauses for the chunks;
 *   2. the chunks (pk_chunks): the whole recording if it fits, else its
 *      speech cut at the last pause that fits, and the pieces of what lies
 *      between;
 *      a chunk whose cut falls in a short silence is heard with the speech
 *      past it too (the context below);
 *   3. each chunk through the encoder and the search, the search beside the
 *      next chunk's encoder (pk_asr_run_chunks): of what a chunk was heard
 *      in, the words that start before its cut, its first words as the
 *      chunk before heard them, up to where the two agree (the seam below);
 *   4. the stretches of a chunk that Silero heard speech in and the model
 *      wrote no word for, transcribed again on their own (the retry below);
 *   5. each chunk's tokens to the caller, who may stop the run there.
 *
 * Without the context and the retry (pk_transcribe_opts), each chunk is
 * NeMo's transcript of it, bit for bit; the two are where this path leaves
 * it, on purpose. README, "Transcribing a recording", has why each step is
 * as it is and what it was measured on.
 *
 * Steps 1 and 2 run a stretch of the recording at a time (the cuts, below):
 * the VAD a run of frames, Silero's state machine a frame (core/segment.h),
 * and a chunk cut once the segments past it say where it ends
 * (pk_chunk_at). A recording of known length is cut whole before step 3. One
 * still being read (pk_audio's wait) is cut as the lanes of step 3 ask for
 * chunks, and its step 4 waits for its end (late): the floor and the budget
 * are the whole recording's, and nothing else is, so its tokens are those
 * of the recording read whole (tests/growcheck holds it to that).
 *
 * pk_steps is the same steps one at a time, for a caller who hears steps 3
 * and 4 itself: pk_steps_new is 1 and 2, and the rest is jobs, a chunk or a
 * retry's stretch to hear, whose tokens come back in any order and go
 * through the seam and the retry in order. pk_asr_transcribe runs those
 * steps on its own, its chunks heard by pk_run_chunks_with and its
 * retries on the calling thread.
 */
#include "asr.h"
#include "os.h"
#include "segment.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* The retry, from pktito-app's transcribe.rs, where it was measured
 * (2026-10-26). The model now and then writes nothing for speech it plainly
 * hears: v2 gave nothing for the first chunk of AliceBench's chapters 8
 * and 10 (0.897-28.112 s, 0.929-28.32 s), and kept only "This is a
 * LibriVox recording" of the first chunk of chapters 2, 7, 9 and 12. At
 * every frame its search visited, blank beat the next word (by 0.02 logits
 * at the closest), and NeMo 3.0's own float32 transcription of chapter 8's
 * chunk is empty too: the model's failure, not pktito's. v3 dropped chapter
 * 5's closing chunk, chapter 1's closing credits, a title in chapter 5 and
 * a line of chapter 10. Each stretch, heard again on its own, gives its
 * words; LibriSpeech's long40 and long300 had no such stretch with either
 * model. The chunks cut since 88293f2 drop others: at b937d71 the retry
 * gets words back in 8 of v2's chunks and 9 of v3's (README, "When the
 * model drops speech").
 *
 * A stretch no word covers is heard again once, if Silero heard at least
 * RETRY_SPEECH_S of speech in it inside a sentence. Less is too little: at
 * 1 s a FLEURS Polish recording's "km", said "kilometrów kwadratowych",
 * left a second after it that gave "metrów kwadratowych." a second time. */
#define RETRY_SPEECH_S 2.0
/* Between sentences - after a word that ends one, at a chunk's start or
 * end, or the whole of a chunk that gave no words, cut in two at
 * quietest() since whole again it would give nothing again - RETRY_BREAK_S
 * is enough. At b937d71 it gets back words that 2 s misses in six of
 * AliceBench's chunks: v2's second "Prizes" in chapter 3 and its "chapter
 * eight" of "End of chapter eight", and v3's "End of chapter five.",
 * "Advice from a caterpillar.", "Alice was silent." and "End of chapter
 * eight" (2026-11-05). It was set on 5b85618's chunks, where chapter 1's
 * "End of chapter one", 1.6 s of speech after "...finished off the cake."
 * with no word, needed it, and the corpus (ls200, long40 and long300 with
 * both models, six FLEURS sets) was otherwise as at 2 s. Since 88293f2's
 * cuts v3 writes that ending whole, and v2 drops only "End of", under a
 * second of speech, too little for either. */
#define RETRY_BREAK_S 1.0
/* At most this share of the recording heard again for nothing, or one
 * chunk if that is more: music that Silero takes for speech gives no words
 * however it is cut, and then costs at most a fifth more time. A stretch
 * that gives words costs none of it, being speech the model dropped: a
 * recording of 134 s (long40's 1320-122612, v3, cut at Silero's own pauses
 * into 30.72 s chunks) dropped 15 s and then 17 s, the first left 15.8 s of
 * its 30.72, and the second, 49 words, was not heard again. Each stretch
 * is heard again once at most, so the retries hear at most the recording
 * over again, whatever they give. */
#define RETRY_SHARE 0.2
/* Only speech no more than 4 dB quieter than the recording's speech is
 * heard again: what the model dropped is the voice the rest of the
 * recording is in, where a voice in the background, which Silero hears as
 * speech too, is quieter. AliceBench's dropped stretches, as 5b85618 cut
 * the chapters, lie -3.6 to +2.7 dB from their chapters' speech.
 * LibriSpeech recordings with three other speakers mixed in at 10 dB SNR
 * had stretches retried down to -24 dB, the background's words put into
 * them: WER at 10 dB SNR 6.21% with no retry, 9.20% ungated, 6.02% at 4 dB;
 * at 5 dB 19.88%, 21.91%, 19.79% (24 recordings). Pink noise and a phone
 * band draw no retry, nor do two songs. 10^(-4/10), so that the test is on
 * mean powers and no logarithm, libm's or ours, decides a transcript. */
#define RETRY_LEVEL 0.39810717055349725
/* Or no more than 3 dB quieter than its chunk's speech, for a passage read
 * softly: AliceBench's chapter 9 (v2) dropped "And how many hours a day did
 * you do lessons?", 4.9 dB under the chapter's speech and 1.8 dB under its
 * chunk's. long40's first 24 recordings with the next three speakers'
 * looped in at 10 and 5 dB SNR (2026-11-02), WER without the chunk's test,
 * with it at 3 dB and at 4 dB: v2 3.86%, 3.86%, 3.91% and 13.60%, 13.62%,
 * 13.91%; v3 8.11%, 7.99%, 8.04% and 22.62%, 22.62%, 22.75%. A stretch of
 * a chunk that gave no words has no chunk's speech of its own to go by.
 * 10^(-3/10). */
#define RETRY_NEAR 0.5011872336272722

/* The pause a chunk may be cut in, as Silero's min_silence_ms for the
 * segments the chunks are cut between: two of its 32 ms frames, 64 ms.
 * Its own 100 ms takes five, 160 ms, and stays for the speech the retry
 * hears. A chunk ends at the last such pause that fits, so a segment
 * longer than the room left ends it early, and conversation goes 10 to
 * 21 s without 160 ms: 52 of the 26 talks' 239 chunks (README, "Where
 * chunks meet") ended under 20 s. At 64 ms, with 30.72 s chunks, the talks
 * take 217 chunks and long40 354, where they took 239 and 368; their false
 * sentence ends at a cut, v3 and v2, 7 and 2 to 5 and 2, and 10 and 11 to
 * 11 and 14; long40's WER 2.19% and 1.95% to 2.12% and 1.99%, which moves
 * as it does when any cut moves. Cut at the longest pause instead, planned
 * over the whole recording for as few chunks, more cuts fell where no
 * context is heard: the talks' false ends 7 and 2 to 9 and 5. */
#define CUT_SILENCE_MS 32

/* A chunk is heard whole, and the model ends it as it would an utterance:
 * a cut in mid-sentence gets a period, and the chunk after it a capital.
 * So a chunk whose cut falls in a silence shorter than CONTEXT_PAUSE_S is
 * heard with CONTEXT_S of the recording past the cut, and keeps the words
 * that start before it: the model has heard the sentence go on (README,
 * "Transcribing a recording", has what that changed). Past the cut only: a
 * chunk heard from inside the speech before it, not from the silence it
 * opens in, drops speech. Heard 2 s before and after each such cut, 22 of
 * long40's 386 chunks lost 771 words with v3 (WER 2.27% to 5.65%), with
 * those edges at the quietest 100 ms within half a second of 2 s, 18 lost
 * 571, and with the search carried over the cut in the last chunk's state
 * instead, 102 to 210 chunks lost speech. Chunks under CONTEXT_ROOM
 * contexts long (--max-chunk under 8 s) hear none. */
#define CONTEXT_S       2.0
#define CONTEXT_PAUSE_S 1.0
#define CONTEXT_ROOM    4

/* Where a chunk heard past its cut meets the next. At the head of its
 * window the model writes a word the cut fell beside a second time
 * ("again, there | There was no") or in part ("reve" for "grieve"), and a
 * capital; the chunk before heard those words with what came before them.
 * So up to the first of its first SEAM_WORDS words that agrees with one of
 * the last chunk's SEAM_KEEP past its cut, the chunk opens with the last
 * chunk's words, and that one has its letters as the last chunk wrote them
 * where the two differ in case alone. Two words agree where their letters
 * do, case and punctuation aside, and they start within SEAM_FRAMES
 * encoder frames of each other. And where the cut fell in the last chunk's
 * last word, which it kept since the word starts before the cut, the chunk
 * opening with that word again, before the last chunk's copy ends, loses
 * its own ("talking familiarly | Familiarly with them" in AliceBench's
 * chapter 3, "Lewis Carroll. | Carroll." in chapter 10, v2). What the last
 * chunk's words cover of the chunk's head, that word or the words the seam
 * took, is not heard again by the retry: heard, it put "Familiarly with
 * the" back. */
#define SEAM_WORDS  4
#define SEAM_KEEP   6
#define SEAM_FRAMES 4
#define SEAM_TOKS   64          /* tokens kept of those words, at most */

void pk_transcribe_defaults(pk_transcribe_opts *o) {
    /* 3072 mel frames, 384 encoder frames: the length the Neural Engine's
     * encoder is built for, which pads a shorter chunk to it - at 30 s,
     * every window was at least 2.3% padding. 30.72 s alone took long40
     * from 368 chunks to 359 and the talks from 239 to 235. */
    o->max_chunk_s = 30.72;
    o->retry = 1;
    o->context = 1;
}

/* The energy over the speech in [from, to) of the samples in buf, which
 * start at the recording's sample `base` and hold [from, to), and how many
 * samples of speech that is. A sample that is not finite is read as 0, as
 * pk_asr_run reads it. From `open` on, the segments are not all known yet
 * (a recording still being read), and every sample counts as speech, its
 * energy left out: the most that *heard can come to. */
static double power(const float *buf, int64_t base, const pk_seg *sp, size_t ns, int64_t from,
                    int64_t to, int64_t open, int64_t *heard) {
    double e = 0;
    int64_t h = 0;
    for (size_t k = 0; k < ns; k++) {
        const int64_t s0 = sp[k].start > from ? sp[k].start : from;
        const int64_t s1 = sp[k].end < to ? sp[k].end : to;
        for (int64_t j = s0; j < s1; j++) {
            const double v = isfinite(buf[j - base]) ? buf[j - base] : 0;
            e += v * v;
        }
        h += s1 > s0 ? s1 - s0 : 0;
    }
    if (open < to)
        h += to - (open > from ? open : from);
    *heard = h;
    return e;
}

/* Where a stretch x[0, n) is cut for its retry: the middle of its quietest
 * 300 ms whose start lies in its middle third, so that neither half is
 * short and the cut falls in a pause where there is one; the middle where
 * the stretch is too short for that. */
static size_t quietest(const float *x, size_t n, int rate) {
    const size_t win = (size_t)(0.3 * rate), step = (size_t)(0.05 * rate);
    double best = INFINITY;
    size_t at_best = n / 2;
    for (size_t at = n / 3; at + win <= n && at < 2 * n / 3; at += step ? step : 1) {
        double e = 0;
        for (size_t j = at; j < at + win; j++) {
            const double v = isfinite(x[j]) ? x[j] : 0;
            e += v * v;
        }
        if (e < best) {
            best = e;
            at_best = at + win / 2;
        }
    }
    return at_best;
}

/* One run's state, for each chunk as it comes: pk_steps */
typedef struct pk_steps {
    const pk_model *m;
    pk_asr *a;                  /* whose VAD hears the recording */
    pk_audio x;                 /* the recording, the caller's copied */
    float *buf;                 /* a chunk's samples, or a stretch's: bufcap of them */
    size_t bufcap;
    int retry;

    /* The cuts: the recording heard a run of the VAD's frames at a time,
     * its speech segments as each is whole and its chunks as each is sure
     * (cut_to). Of a recording still being read, the lanes' threads ask for
     * chunks while the calling thread takes their tokens, so all of this is
     * read and written under lock. */
    pk_lock *lock;
    int64_t (*wait)(void *arg, int64_t want);   /* x's, where its length was not known */
    void *wait_arg;
    int64_t max_len, ctx;
    int64_t heard;              /* samples the VAD has had */
    int64_t length;             /* the recording's, once ended */
    int ended, cut_over, cut_failed;
    pk_segs seg_sp, seg_cut;    /* Silero's segments, and those the chunks are cut between */
    pk_pads pad_sp, pad_cut;
    pk_seg *sp, *cs;            /* each one's segments so far, final */
    size_t ns, spcap, ncs, cscap;
    size_t ci;                  /* the cut segment the next speech chunk opens with */
    int64_t cut_done;           /* the end of the last speech chunk */
    pk_seg *c, *win;            /* the chunks, and what each is heard in */
    size_t nc, ccap;
    float *vx, *vp;             /* a run of frames' samples, and their probabilities */
    size_t vxcap, vpcap;

    int64_t budget, done, total;
    double floor;               /* mean power a stretch's speech must reach */
    /* the jobs: the chunks handed out, and chunk i's tokens as put, ct[i]
     * and cn[i] of them (-1 until then), kept until the seam takes them */
    size_t given;
    pk_tok **ct;
    long *cn;
    /* Of a recording still being read, whose retries wait for its end
     * (late): the tokens of each chunk that may have one, as the seam left
     * them, kn[i] of them (-1: none kept), and the frames of its head */
    pk_tok **kt;
    long *kn;
    int64_t *khead;
    size_t jcap;                /* chunks ct, cn, kt, kn and khead have room for */
    int late;                   /* the retries wait */
    int maybe;                  /* the chunk the seam is at may have one */
    /* the chunk the seam is at, those before it taken: waiting for its
     * tokens, its stretches heard again (st[0, nst), the k-th next, out or
     * not, `retried` of them heard), or done and waiting to be taken */
    size_t at;
    pk_seg cc, cw;              /* it, and what it was heard in */
    enum { AWAIT, RETRYING, DONE } state;
    int failed;                     /* a call returned -1: the run is over */
    long nst, k;
    int out, retried;
    pk_tok *t;                  /* the chunk's tokens, n of them, its retries' among them */
    size_t n, tcap;
    pk_word *w;
    size_t wcap;
    int32_t (*st)[2];           /* the stretches to hear again, in frames */
    size_t stcap;
    char *text;
    size_t textcap;
    /* the last chunk's first SEAM_KEEP words past its cut, for the seam:
     * their tokens, each token's sample, and where each word starts */
    pk_tok seam[SEAM_TOKS];
    int64_t seam_at[SEAM_TOKS];
    int seam_word[SEAM_KEEP + 1], nseam;
    /* the last chunk's last word before its cut where the word runs past
     * it: its tokens, and the sample it ends at */
    pk_tok tail[SEAM_TOKS];
    int ntail;
    int64_t tail_end;
} run;

static int grow(void **p, size_t *cap, size_t need, size_t size) {
    if (need <= *cap)
        return 0;
    size_t c = *cap ? *cap : 64;
    while (c < need)
        c *= 2;
    void *q = realloc(*p, c * size);
    if (!q)
        return -1;
    *p = q;
    *cap = c;
    return 0;
}

/* Room in r->buf for n samples: 0, or -1 if memory runs out. Nothing in it
 * outlives a read, so it is made again, not copied. */
static int room(run *r, size_t n) {
    if (n <= r->bufcap)
        return 0;
    free(r->buf);
    r->bufcap = 0;
    if (!(r->buf = malloc(n * sizeof *r->buf)))
        return -1;
    r->bufcap = n;
    return 0;
}

/* Room for chunk i's tokens as they are put and kept: 0, or -1 */
static int jobs(run *r, size_t i) {
    if (i < r->jcap)
        return 0;
    size_t cap = r->jcap ? r->jcap : 16;
    while (cap <= i)
        cap *= 2;
    pk_tok **ct = realloc(r->ct, cap * sizeof *ct);
    if (ct)
        r->ct = ct;
    pk_tok **kt = realloc(r->kt, cap * sizeof *kt);
    if (kt)
        r->kt = kt;
    long *cn = realloc(r->cn, cap * sizeof *cn);
    if (cn)
        r->cn = cn;
    long *kn = realloc(r->kn, cap * sizeof *kn);
    if (kn)
        r->kn = kn;
    int64_t *kh = realloc(r->khead, cap * sizeof *kh);
    if (kh)
        r->khead = kh;
    if (!ct || !kt || !cn || !kn || !kh)
        return -1;
    for (size_t q = r->jcap; q < cap; q++) {
        r->ct[q] = r->kt[q] = NULL;
        r->cn[q] = r->kn[q] = -1;
        r->khead[q] = 0;
    }
    r->jcap = cap;
    return 0;
}

/* The chunks cut so far, and chunk i with what it is heard in: under the
 * lock, since a lane may be cutting the next */
static size_t chunks(run *r) {
    pk_lock_take(r->lock);
    const size_t n = r->nc;
    pk_lock_give(r->lock);
    return n;
}

static void chunk(run *r, size_t i, pk_seg *c, pk_seg *win) {
    pk_lock_take(r->lock);
    *c = r->c[i];
    *win = r->win[i];
    pk_lock_give(r->lock);
}

/* Whether a word's text ends a sentence */
static int ends_sentence(run *r, const pk_tok *t, const pk_word *w) {
    const pk_model *m = r->m;
    size_t len;
    while ((len = pk_text(m, t + w->tok, (size_t)w->ntok, r->text, r->textcap)) >= r->textcap) {
        size_t cap = r->textcap;
        if (grow((void **)&r->text, &cap, 2 * r->textcap, 1))
            return -1;
        r->textcap = cap;
    }
    return len && strchr(".?!", r->text[len - 1]) != NULL;
}

/* Chunk i's stretches to hear again, into r->st, none in its first head
 * frames, which the last chunk's words cover: their count, or -1 if memory
 * runs out. Each is charged to the budget as it is heard, if it gives
 * nothing. With `most`, of a recording still being read, the stretches
 * there may be once it has ended: its speech is not louder than nothing
 * yet, the floor not being known, and the segments from `open` on taken
 * for all speech (power). Every stretch the recording's end leaves is one
 * of those, since the floor and the segments' ends only take away. */
static long stretches(run *r, const pk_seg *ch, const pk_tok *t, long nt, int64_t head, int most,
                      int64_t open) {
    const double all = most ? 0 : r->floor;
    const pk_model *m = r->m;
    const int rate = pk_model_rate(m);
    const int64_t spf = llround(pk_model_frame_seconds(m) * rate);
    const int64_t frames = (ch->end - ch->start + spf - 1) / spf;
    const int64_t inside = (int64_t)(RETRY_SPEECH_S * rate), at_break = (int64_t)(RETRY_BREAK_S * rate);
    const int64_t len = ch->end - ch->start;
    if (room(r, (size_t)len) || pk_audio_read(&r->x, ch->start, r->buf, (size_t)len))
        return -1;
    if (nt == 0) {
        int64_t heard;
        const double e = power(r->buf, ch->start, r->sp, r->ns, ch->start, ch->end, open, &heard);
        if (heard < at_break || e < all * (double)heard)
            return 0;
        const int64_t mid = (int64_t)quietest(r->buf, (size_t)len, rate) / spf;
        r->st[0][0] = (int32_t)(head < mid ? head : mid), r->st[0][1] = (int32_t)mid;
        r->st[1][0] = (int32_t)mid, r->st[1][1] = (int32_t)frames;
        return 2;
    }
    size_t nw = pk_words(m, t, (size_t)nt, r->w);
    int64_t near;           /* the chunk's speech, for a passage read softly */
    const double ce = power(r->buf, ch->start, r->sp, r->ns, ch->start, ch->end, open, &near);
    const double floor = near && ce / (double)near * RETRY_NEAR < all ? ce / (double)near * RETRY_NEAR : all;
    long k = 0;
    int64_t at = head;
    int after_break = 1;        /* a chunk's start follows a break */
    for (size_t j = 0; j <= nw; j++) {
        const int64_t f0 = at, f1 = j < nw ? r->w[j].start : frames;
        if (f1 > f0) {
            const int64_t s0 = ch->start + f0 * spf;
            const int64_t s1 = ch->start + f1 * spf < ch->end ? ch->start + f1 * spf : ch->end;
            const int64_t enough = after_break || f1 == frames ? at_break : inside;
            int64_t heard;
            const double e = power(r->buf, ch->start, r->sp, r->ns, s0, s1, open, &heard);
            if (heard >= enough && e >= floor * (double)heard) {
                r->st[k][0] = (int32_t)f0, r->st[k][1] = (int32_t)f1;
                k++;
            }
        }
        if (j < nw) {
            at = r->w[j].end > at ? r->w[j].end : at;
            const int e = ends_sentence(r, t, &r->w[j]);
            if (e < 0)
                return -1;
            after_break = e;
        }
    }
    return k;
}

/* c in lower case, in the alphabets the models write: Latin's first two
 * blocks, Greek's and Cyrillic's capitals; any other as it is */
static uint32_t lower(uint32_t c) {
    if ((c >= 'A' && c <= 'Z') || (c >= 0xC0 && c <= 0xDE && c != 0xD7)
        || (c >= 0x391 && c <= 0x3AB && c != 0x3A2) || (c >= 0x410 && c <= 0x42F))
        return c + 32;
    if (c >= 0x400 && c <= 0x40F)
        return c + 80;
    if (c == 0x178)
        return 0xFF;
    if (c >= 0x100 && c <= 0x17F && c != 0x130 && c != 0x131 && c != 0x138 && c != 0x149
        && c != 0x17F) {
        const uint32_t odd = (c >= 0x139 && c <= 0x148) || c >= 0x179;   /* capitals odd */
        return (c & 1) == odd ? c + 1 : c;
    }
    return c;
}

/* The code point UTF-8 has at *s, stepping past it */
static uint32_t utf8(const char **s, const char *end) {
    const unsigned c = (unsigned char)*(*s)++;
    const int more = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : c >= 0xC0 ? 1 : 0;
    uint32_t cp = more ? c & (0x3Fu >> more) : c;
    for (int k = 0; k < more && *s < end && ((unsigned char)**s & 0xC0) == 0x80; k++)
        cp = cp << 6 | ((unsigned char)*(*s)++ & 0x3F);
    return cp;
}

static int punct_char(uint32_t c) { return c && c < 0x80 && strchr(".,?!;:\"", (int)c); }

/* A token of punctuation alone */
static int punct(const pk_model *m, int32_t id) {
    size_t len;
    const char *s = pk_piece(m, id, &len), *end = s + len;
    if (len >= 3 && !memcmp(s, "\xe2\x96\x81", 3))
        s += 3;
    if (s == end)
        return 0;
    while (s < end)
        if (!punct_char(utf8(&s, end)))
            return 0;
    return 1;
}

/* Whether na tokens a and nb tokens b have the same letters, the space
 * before a word and punctuation left out, and in case too where exact */
static int same_letters(const pk_model *m, const pk_tok *a, int na, const pk_tok *b, int nb,
                        int exact) {
    uint32_t l[2][48];
    size_t k[2] = { 0, 0 };
    for (int side = 0; side < 2; side++)
        for (int q = 0; q < (side ? nb : na); q++) {
            size_t len;
            const char *s = pk_piece(m, (side ? b : a)[q].id, &len), *end = s + len;
            if (len >= 3 && !memcmp(s, "\xe2\x96\x81", 3))
                s += 3;
            while (s < end) {
                const uint32_t c = utf8(&s, end);
                if (!punct_char(c) && k[side] < 48)
                    l[side][k[side]++] = exact ? c : lower(c);
            }
        }
    return k[0] && k[0] == k[1] && !memcmp(l[0], l[1], k[0] * sizeof l[0][0]);
}

/* qn of the last chunk's tokens past its cut, from r->seam + q0, to r->t
 * + k, their frames from this chunk's start */
static void put(run *r, const pk_seg *ch, size_t k, int q0, int qn, int64_t spf) {
    for (int q = 0; q < qn; q++) {
        const int64_t f = (r->seam_at[q0 + q] - ch->start) / spf;
        r->t[k + (size_t)q] = r->seam[q0 + q];
        r->t[k + (size_t)q].frame = f > 0 ? (int32_t)f : 0;
    }
}

/* The seam (above) on the chunk's n tokens in r->t, which has room for
 * SEAM_TOKS more: its new count */
static size_t seam(run *r, const pk_seg *ch, size_t n, int64_t *head) {
    const pk_model *m = r->m;
    const int64_t spf = llround(pk_model_frame_seconds(m) * pk_model_rate(m));
    const size_t nw = pk_words(m, r->t, n, r->w);
    for (size_t j = 0; j < nw && j < SEAM_WORDS; j++) {
        const int64_t at = ch->start + (int64_t)r->w[j].start * spf;
        const size_t k0 = (size_t)r->w[j].tok;
        const int kn = r->w[j].ntok;
        for (int q = 0; q < r->nseam; q++) {
            const int q0 = r->seam_word[q], qn = r->seam_word[q + 1] - q0;
            if (llabs(at - r->seam_at[q0]) > SEAM_FRAMES * spf
                || !same_letters(m, r->t + k0, kn, r->seam + q0, qn, 0))
                continue;
            /* the word's letters, the tokens before its punctuation, as the
             * last chunk wrote them, where they differ in case */
            int ka = 0, qa = 0;
            while (ka < kn && !punct(m, r->t[k0 + (size_t)ka].id))
                ka++;
            while (qa < qn && !punct(m, r->seam[q0 + qa].id))
                qa++;
            if (ka && qa && !same_letters(m, r->t + k0, ka, r->seam + q0, qa, 1)) {
                memmove(r->t + k0 + (size_t)qa, r->t + k0 + (size_t)ka, (n - k0 - (size_t)ka) * sizeof *r->t);
                n = n - (size_t)ka + (size_t)qa;
                put(r, ch, k0, q0, qa, spf);
            }
            /* and the last chunk's words before it, for this one's */
            if (*head < r->w[j].start)
                *head = r->w[j].start;
            memmove(r->t + q0, r->t + k0, (n - k0) * sizeof *r->t);
            n = n - k0 + (size_t)q0;
            put(r, ch, 0, 0, q0, spf);
            return n;
        }
    }
    return n;
}

/* This chunk without its first word where that is the last chunk's last
 * word again, the one the cut fell in, starting before the last chunk's
 * copy ends; the tokens left */
static size_t repeat(run *r, const pk_seg *ch, size_t n) {
    const pk_model *m = r->m;
    const int64_t spf = llround(pk_model_frame_seconds(m) * pk_model_rate(m));
    if (!pk_words(m, r->t, n, r->w))
        return n;
    const pk_word *w = &r->w[0];
    if (ch->start + (int64_t)w->start * spf >= r->tail_end
        || !same_letters(m, r->t + w->tok, w->ntok, r->tail, r->ntail, 0))
        return n;
    const size_t drop = (size_t)(w->tok + w->ntok);
    memmove(r->t, r->t + drop, (n - drop) * sizeof *r->t);
    return n - drop;
}

/* Chunk at's tokens toks[0, nt): into r->t the words that start before its
 * cut, through the seam with the chunk before, and its stretches to hear
 * again into r->st. 0, or -1 if memory runs out. */
static int take_chunk(run *r, const pk_tok *toks, long nt) {
    chunk(r, r->at, &r->cc, &r->cw);
    const pk_seg *ch = &r->cc;
    const pk_model *m = r->m;
    const int64_t spf = llround(pk_model_frame_seconds(m) * pk_model_rate(m));
    size_t stcap = r->stcap;
    if (grow((void **)&r->t, &r->tcap, (size_t)nt + 1 + SEAM_TOKS, sizeof *r->t)
        || grow((void **)&r->w, &r->wcap, (size_t)nt + 1, sizeof *r->w)
        || grow((void **)&r->st, &stcap, (size_t)nt + 2, sizeof *r->st))
        return -1;
    r->stcap = stcap;
    /* of what the chunk was heard in, the words that start before its cut,
     * and the first SEAM_KEEP after it for the next chunk's seam */
    const pk_seg *wn = &r->cw;
    pk_tok next[SEAM_TOKS];
    int64_t next_at[SEAM_TOKS];
    int next_word[SEAM_KEEP + 1] = { 0 }, nnext = 0;
    const pk_word *last = NULL;
    size_t n = 0;
    if (wn->end == ch->end) {
        if (nt)
            memcpy(r->t, toks, (size_t)nt * sizeof *toks);
        n = (size_t)nt;
    } else if (nt) {
        const size_t nw = pk_words(m, toks, (size_t)nt, r->w);
        for (size_t j = 0; j < nw; j++) {
            const pk_word *w = &r->w[j];
            if (ch->start + (int64_t)w->start * spf < ch->end) {
                memcpy(r->t + n, toks + w->tok, (size_t)w->ntok * sizeof *toks);
                n += (size_t)w->ntok;
                last = w;
            } else if (nnext < SEAM_KEEP && next_word[nnext] + w->ntok <= SEAM_TOKS) {
                for (int32_t q = 0; q < w->ntok; q++) {
                    next[next_word[nnext] + q] = toks[w->tok + q];
                    next_at[next_word[nnext] + q] = ch->start + (int64_t)toks[w->tok + q].frame * spf;
                }
                next_word[nnext + 1] = next_word[nnext] + w->ntok;
                nnext++;
            }
        }
    }
    /* the last word kept, for the next chunk, where it runs past the cut */
    pk_tok tail[SEAM_TOKS];
    int ntail = 0;
    int64_t tail_end = 0;
    if (last && ch->start + (int64_t)last->end * spf > ch->end && last->ntok <= SEAM_TOKS) {
        memcpy(tail, toks + last->tok, (size_t)last->ntok * sizeof *toks);
        ntail = last->ntok;
        tail_end = ch->start + (int64_t)last->end * spf;
    }
    int64_t head = 0;           /* frames the last chunk's words cover */
    if (r->ntail) {
        if (n)
            n = repeat(r, ch, n);
        head = (r->tail_end - ch->start + spf - 1) / spf;
    }
    if (r->nseam && n)
        n = seam(r, ch, n, &head);
    r->nseam = nnext;
    memcpy(r->seam, next, (size_t)next_word[nnext] * sizeof *next);
    memcpy(r->seam_at, next_at, (size_t)next_word[nnext] * sizeof *next_at);
    memcpy(r->seam_word, next_word, sizeof next_word);
    r->ntail = ntail;
    memcpy(r->tail, tail, (size_t)ntail * sizeof *tail);
    r->tail_end = tail_end;
    long nst = 0;
    r->maybe = 0;
    if (r->retry && !r->late) {
        nst = stretches(r, ch, r->t, (long)n, head, 0, INT64_MAX);
    } else if (r->retry) {
        /* its retries wait for the recording's end: whether it may have
         * one, and if so its tokens as they are here, for late */
        pk_lock_take(r->lock);
        const double pad = r->pad_sp.speech_pad;
        const int64_t from = r->pad_sp.has ? r->pad_sp.held.start : pk_segs_floor(&r->seg_sp) - (int64_t)pad;
        const long may = stretches(r, ch, r->t, (long)n, head, 1, r->ended ? INT64_MAX : from);
        pk_lock_give(r->lock);
        if (may < 0 || (may && n && !(r->kt[r->at] = malloc(n * sizeof *r->t))))
            return -1;
        if (may) {
            if (n)
                memcpy(r->kt[r->at], r->t, n * sizeof *r->t);
            r->kn[r->at] = (long)n;
            r->khead[r->at] = head;
            r->maybe = 1;
        }
    }
    if (nst < 0)
        return -1;
    r->n = n;
    r->nst = nst;
    r->k = 0;
    r->retried = 0;
    return 0;
}

/* The k-th of chunk at's stretches to hear again, or the first after it
 * whose length the budget still has: 1 with its samples in *at, or 0 when
 * none is left */
static int stretch(run *r, pk_seg *at) {
    const pk_seg *ch = &r->cc;
    const int64_t spf = llround(pk_model_frame_seconds(r->m) * pk_model_rate(r->m));
    for (; r->k < r->nst; r->k++) {
        const int64_t s0 = ch->start + r->st[r->k][0] * spf;
        const int64_t s1 = ch->start + r->st[r->k][1] * spf < ch->end ? ch->start + r->st[r->k][1] * spf : ch->end;
        if (s1 > s0 && s1 - s0 <= r->budget) {
            at->start = s0, at->end = s1;
            return 1;
        }
    }
    return 0;
}

/* Chunk at done: its tokens in frame order, its own before a retry's at
 * the same frame */
static void finish(run *r) {
    for (size_t j = 1; j < r->n; j++) {
        const pk_tok v = r->t[j];
        size_t q = j;
        for (; q > 0 && r->t[q - 1].frame > v.frame; q--)
            r->t[q] = r->t[q - 1];
        r->t[q] = v;
    }
    r->state = DONE;
}

/* A stretch heard again, at: its count tokens into the chunk's, their
 * frames from the chunk's start, and the stretch charged to the budget if
 * it gave none. 0, or -1 if memory runs out. */
static int heard_again(run *r, const pk_seg *at, const pk_tok *toks, long count) {
    if (grow((void **)&r->t, &r->tcap, r->n + (size_t)count, sizeof *r->t))
        return -1;
    if (count == 0)
        r->budget -= at->end - at->start;
    for (long q = 0; q < count; q++) {
        r->t[r->n] = toks[q];
        r->t[r->n++].frame += r->st[r->k][0];
    }
    r->retried++;
    r->k++;
    return 0;
}

/* -1, and the run over: a failed take_chunk may have freed a chunk's
 * tokens or moved the seam, so no later call may go on from it */
static int fail(run *r) {
    r->failed = 1;
    return -1;
}

/* The seam as far as it can go: chunk at taken once its tokens are in, its
 * stretches heard again one at a time, and done, and no further until it
 * is taken, its tokens being r->t. 0, or -1 if memory runs out. */
static int advance(run *r) {
    pk_seg at;
    while (r->state != DONE && !r->out && r->at < chunks(r)) {
        if (r->state == AWAIT) {
            if (r->at >= r->jcap || r->cn[r->at] < 0)
                return 0;
            const int rc = take_chunk(r, r->ct[r->at], r->cn[r->at]);
            free(r->ct[r->at]);
            r->ct[r->at] = NULL;
            if (rc)
                return fail(r);
            r->state = RETRYING;
        }
        if (stretch(r, &at))
            return 0;           /* the k-th, to hand out */
        finish(r);
        r->done += r->cc.end - r->cc.start;
    }
    return 0;
}

/* The VAD reads x a run of its frames at a time (pk_vad_run_frames), each
 * run through pk_vad_probs_on as one call over all of x would take it: the
 * same bits, since its stream keeps the last frame's context between
 * calls. */
long pk_asr_vad(pk_asr *a, const pk_audio *x, float *probs) {
    if (x->n < 0)
        return -1;
    const size_t n = (size_t)x->n;
    const size_t frames = (n + PK_VAD_SAMPLES - 1) / PK_VAD_SAMPLES, whole = n / PK_VAD_SAMPLES;
    if (n == 0)
        return 0;
    if (!a->vad && !(a->vad = pk_vad_new(a->m)))
        return -1;
    pk_vad *v = a->vad;
    pk_vad_reset(v);
    const size_t run = pk_vad_run_frames(a->pool) < whole ? pk_vad_run_frames(a->pool) : whole;
    float *buf = malloc((run ? run : 1) * PK_VAD_SAMPLES * sizeof *buf);
    if (!buf)
        return -1;
    long rc = (long)frames;
    for (size_t f = 0; f < whole && rc >= 0; f += run) {
        const size_t k = whole - f < run ? whole - f : run;
        if (pk_audio_read(x, (int64_t)(f * PK_VAD_SAMPLES), buf, k * PK_VAD_SAMPLES))
            rc = -1;
        else
            pk_vad_probs_on(v, a->pool, buf, k, probs + f);
    }
    if (rc >= 0 && whole < frames) {
        float last[PK_VAD_SAMPLES] = { 0 };
        if (pk_audio_read(x, (int64_t)(whole * PK_VAD_SAMPLES), last, n - whole * PK_VAD_SAMPLES))
            rc = -1;
        else
            probs[whole] = pk_vad_prob(v, last);
    }
    free(buf);
    return rc;
}

long pk_asr_speech(pk_asr *a, const pk_audio *x, const pk_seg_params *p, pk_seg *out, size_t cap) {
    if (x->n < 0)
        return -1;
    const size_t frames = ((size_t)x->n + PK_VAD_SAMPLES - 1) / PK_VAD_SAMPLES;
    pk_seg_params def;
    if (!p) {
        pk_seg_defaults(&def);
        p = &def;
    }
    float *probs = malloc((frames + 1) * sizeof *probs);
    const long rc = probs && pk_asr_vad(a, x, probs) >= 0
                        ? (long)pk_segments(probs, frames, pk_model_rate(a->m), x->n, p, out, cap)
                        : -1;
    free(probs);
    return rc;
}

/* ------------------------------------------------------------------ the cuts --- */

static int seg_push(pk_seg **s, size_t *n, size_t *cap, pk_seg v) {
    if (grow((void **)s, cap, *n + 1, sizeof **s))
        return -1;
    (*s)[(*n)++] = v;
    return 0;
}

/* k frames' probabilities through both segmenters, and with `end` the
 * recording's end after them: each segment onto its list once the one
 * after it has come, which its padding waits for (core/segment.h). These
 * are pk_segments' segments of the whole recording's probabilities, cut
 * into calls anyhow. 0, or -1 if memory runs out. */
static int cut_frames(run *r, const float *probs, size_t k, int end) {
    pk_seg got[2], fin;
    for (size_t f = 0; f < k + (end ? 1 : 0); f++) {
        int n = f < k ? pk_segs_step(&r->seg_sp, probs[f], got) : pk_segs_end(&r->seg_sp, r->length, got);
        for (int q = 0; q < n; q++)
            if (pk_pads_put(&r->pad_sp, got[q], &fin) && seg_push(&r->sp, &r->ns, &r->spcap, fin))
                return -1;
        n = f < k ? pk_segs_step(&r->seg_cut, probs[f], got) : pk_segs_end(&r->seg_cut, r->length, got);
        for (int q = 0; q < n; q++)
            if (pk_pads_put(&r->pad_cut, got[q], &fin) && seg_push(&r->cs, &r->ncs, &r->cscap, fin))
                return -1;
    }
    if (end && pk_pads_end(&r->pad_sp, r->length, &fin) && seg_push(&r->sp, &r->ns, &r->spcap, fin))
        return -1;
    if (end && pk_pads_end(&r->pad_cut, r->length, &fin) && seg_push(&r->cs, &r->ncs, &r->cscap, fin))
        return -1;
    return 0;
}

/* Room for `more` chunks after those cut: 0, or -1 */
static int chunk_room(run *r, size_t more) {
    if (r->nc + more <= r->ccap)
        return 0;
    size_t cap = r->ccap ? r->ccap : 16;
    while (cap < r->nc + more)
        cap *= 2;
    pk_seg *c = realloc(r->c, cap * sizeof *c);
    if (c)
        r->c = c;
    pk_seg *w = realloc(r->win, cap * sizeof *w);
    if (w)
        r->win = w;
    if (!c || !w)
        return -1;
    r->ccap = cap;
    return 0;
}

/* The chunks that are sure from the segments so far, onto the list: these
 * are pk_chunks_past's of the whole recording, a chunk at a time
 * (pk_chunk_at says when one is sure, core/asr.c). A recording that ended
 * within max_len, no chunk cut, is one chunk; none was cut before its end,
 * since a chunk is sure only once speech past max_len from its start has
 * been heard. 0, or -1 if a read fails or memory runs out. */
static int cut_chunks(run *r) {
    if (r->ended && r->nc == 0 && r->ci == 0 && r->length <= r->max_len && r->ncs) {
        if (chunk_room(r, 1))
            return -1;
        r->c[0] = r->win[0] = (pk_seg){ 0, r->length };
        r->nc = 1;
        r->cut_over = 1;
        return 0;
    }
    const int64_t pause = (int64_t)(CONTEXT_PAUSE_S * pk_model_rate(r->m));
    while (r->ci < r->ncs) {
        pk_seg c, w;
        size_t last;
        if (!pk_chunk_at(r->cs, r->ncs, r->pad_cut.has ? r->pad_cut.held.start : -1, r->ended,
                         r->ended ? r->length : r->heard, r->ci, r->max_len, r->ctx, pause, &c, &w, &last))
            return 0;
        if (chunk_room(r, (size_t)pk_gap_pieces(r->cut_done, c.start, r->max_len) + 1)
            || pk_gap_chunks(&r->x, r->cut_done, c.start, r->max_len, r->c, r->win, &r->nc))
            return -1;
        r->c[r->nc] = c;
        r->win[r->nc++] = w;
        r->cut_done = c.end;
        r->ci = last + 1;
    }
    if (r->ended) {
        if (chunk_room(r, (size_t)pk_gap_pieces(r->cut_done, r->length, r->max_len))
            || pk_gap_chunks(&r->x, r->cut_done, r->length, r->max_len, r->c, r->win, &r->nc))
            return -1;
        r->cut_over = 1;
    }
    return 0;
}

/* The next run of the VAD's frames heard, on pool, and what it makes sure
 * cut: pk_asr_vad's runs, the same bits however the frames fall into
 * calls. Where the length was not known (wait), a run is no longer than a
 * chunk, so that a chunk is cut soon after it is there, and the lock is
 * let go while the recording is waited for: another lane may hear this
 * run meanwhile. Called with the lock taken; 0, or -1 if a read or the
 * wait fails or memory runs out. */
static int hear(run *r, pk_pool *pool) {
    size_t frames = pk_vad_run_frames(pool);
    if (r->wait && frames > (size_t)(r->max_len / PK_VAD_SAMPLES))
        frames = (size_t)(r->max_len / PK_VAD_SAMPLES);
    const int64_t from = r->heard, want = from + (int64_t)frames * PK_VAD_SAMPLES;
    int64_t got = r->x.n;
    if (r->wait) {
        pk_lock_give(r->lock);
        got = r->wait(r->wait_arg, want);
        pk_lock_take(r->lock);
        if (r->heard != from || r->ended)
            return 0;
        if (got < from)
            return -1;
    }
    const int end = got < want;
    const int64_t upto = end ? got : want;
    const size_t whole = (size_t)((upto - from) / PK_VAD_SAMPLES);
    const size_t rest = (size_t)(upto - from) - whole * PK_VAD_SAMPLES;
    if (grow((void **)&r->vx, &r->vxcap, (whole + 1) * PK_VAD_SAMPLES, sizeof *r->vx)
        || grow((void **)&r->vp, &r->vpcap, whole + 1, sizeof *r->vp))
        return -1;
    size_t k = whole;
    if (whole) {
        if (pk_audio_read(&r->x, from, r->vx, whole * PK_VAD_SAMPLES))
            return -1;
        pk_vad_probs_on(r->a->vad, pool, r->vx, whole, r->vp);
    }
    if (rest) {                 /* the last frame, padded with silence */
        float last[PK_VAD_SAMPLES] = { 0 };
        if (pk_audio_read(&r->x, from + (int64_t)(whole * PK_VAD_SAMPLES), last, rest))
            return -1;
        r->vp[k++] = pk_vad_prob(r->a->vad, last);
    }
    r->heard = upto;
    if (end) {
        r->ended = 1;
        r->length = got;
    }
    if (cut_frames(r, r->vp, k, end) || cut_chunks(r))
        return -1;
    if (r->cut_over) {          /* the VAD's run of samples, 8 MB on 16 threads, not kept for the chunks */
        free(r->vx);
        free(r->vp);
        r->vx = r->vp = NULL;
        r->vxcap = r->vpcap = 0;
    }
    return 0;
}

/* Chunk i cut, the recording heard until it is: 1, 0 where the recording
 * has fewer chunks, or -1 where hearing it failed, for this call and every
 * one after it. On the calling thread or a lane's, whose pool it is given. */
static int cut_to(run *r, size_t i, pk_pool *pool) {
    int rc = -1;
    pk_lock_take(r->lock);
    while (!r->cut_failed) {
        if (i < r->nc) {
            rc = 1;
            break;
        }
        if (r->cut_over) {
            rc = 0;
            break;
        }
        if (hear(r, pool))
            r->cut_failed = 1;
    }
    pk_lock_give(r->lock);
    return rc;
}

/* The floor and the budget, of all the recording's chunks: once they are
 * cut. 0, or -1 if a read fails or memory runs out. */
static int levels(run *r) {
    double e = 0;
    int64_t heard = 0;
    r->total = 0;
    for (size_t i = 0; i < r->nc; i++) {
        int64_t h;
        const pk_seg *ch = &r->c[i];
        const size_t len = (size_t)(ch->end - ch->start);
        if (room(r, len) || pk_audio_read(&r->x, ch->start, r->buf, len))
            return -1;
        e += power(r->buf, ch->start, r->sp, r->ns, ch->start, ch->end, INT64_MAX, &h);
        heard += h;
        r->total += ch->end - ch->start;
    }
    r->floor = heard ? e / (double)heard * RETRY_LEVEL : 0;
    r->budget = (int64_t)((double)r->total * RETRY_SHARE);
    if (r->budget < r->max_len)
        r->budget = r->max_len;
    return 0;
}

/* A run over x, nothing of it heard yet */
static run *run_new(pk_asr *a, const pk_audio *x, const pk_transcribe_opts *o) {
    pk_transcribe_opts def;
    if (!o) {
        pk_transcribe_defaults(&def);
        o = &def;
    }
    const pk_model *m = a->m;
    const int rate = pk_model_rate(m);
    const int64_t max_len = (int64_t)(o->max_chunk_s * rate);
    if (max_len < rate)
        return NULL;
    if (!a->vad && !(a->vad = pk_vad_new(m)))
        return NULL;
    pk_vad_reset(a->vad);
    run *r = calloc(1, sizeof *r);
    if (!r || !(r->lock = pk_lock_new()) || !(r->text = malloc(r->textcap = 256))) {
        pk_steps_free(r);
        return NULL;
    }
    r->m = m;
    r->a = a;
    r->x = *x;
    r->retry = o->retry;
    r->max_len = max_len;

    /* the speech: segments no longer than a chunk less the padding Silero
     * puts round them and the context past its cut; Silero's own for the
     * retry, and the same at shorter pauses for the cuts */
    pk_seg_params p;
    pk_seg_defaults(&p);
    r->ctx = o->context ? (int64_t)(CONTEXT_S * rate) : 0;
    if (max_len < CONTEXT_ROOM * r->ctx)
        r->ctx = 0;
    p.max_speech_s = o->max_chunk_s - (double)r->ctx / rate - 2 * p.speech_pad_ms / 1000 - 0.1;
    pk_segs_init(&r->seg_sp, rate, &p);
    pk_pads_init(&r->pad_sp, rate, &p);
    p.min_silence_ms = CUT_SILENCE_MS;
    pk_segs_init(&r->seg_cut, rate, &p);
    pk_pads_init(&r->pad_cut, rate, &p);
    return r;
}

pk_steps *pk_steps_new(pk_asr *a, const pk_audio *x, const pk_transcribe_opts *o) {
    if (x->n < 0)
        return NULL;
    run *r = run_new(a, x, o);
    if (r && (cut_to(r, SIZE_MAX, a->pool) < 0 || levels(r))) {
        pk_steps_free(r);
        return NULL;
    }
    return r;
}

size_t pk_steps_chunks(const pk_steps *r) { return r->nc; }

/* Every chunk handed out at once, without its job: pk_asr_transcribe hears
 * them all itself, and puts each as it would one pk_steps_job had handed
 * out */
static void hand_out_all(run *r) { r->given = SIZE_MAX; }

int pk_steps_job(pk_steps *r, pk_job *j) {
    if (r->failed || advance(r))
        return -1;
    const size_t nc = chunks(r);
    if (r->at < nc && r->state == RETRYING && !r->out && stretch(r, &j->at)) {
        j->chunk = r->at;
        j->retry = 1;
        r->out = 1;
        return 1;
    }
    if (r->given < nc) {
        j->chunk = r->given;
        j->retry = 0;
        j->at = r->win[r->given++];
        return 1;
    }
    return 0;
}

int pk_steps_put(pk_steps *r, const pk_job *j, const pk_tok *toks, long count) {
    if (r->failed || count < 0)
        return fail(r);
    if (j->retry) {
        pk_seg at;
        if (!r->out || j->chunk != r->at || !stretch(r, &at) || j->at.start != at.start
            || j->at.end != at.end || heard_again(r, &at, toks, count))
            return fail(r);
        r->out = 0;
    } else {
        const size_t i = j->chunk;
        pk_seg c, win;
        if (i >= r->given || i < r->at || i >= chunks(r) || jobs(r, i) || r->cn[i] >= 0)
            return fail(r);         /* not out, or put before */
        chunk(r, i, &c, &win);
        if (j->at.start != win.start || j->at.end != win.end)
            return fail(r);
        r->cn[i] = count;
        if (i == r->at) {           /* the chunk due: taken from toks, not kept */
            if (take_chunk(r, toks, count))
                return fail(r);
            r->state = RETRYING;
        } else if (count) {
            if (!(r->ct[i] = malloc((size_t)count * sizeof *toks)))
                return fail(r);
            memcpy(r->ct[i], toks, (size_t)count * sizeof *toks);
        }
    }
    return advance(r);
}

/* The chunk the seam is at, done, as fn has it. Of a recording still being
 * read (late), how many chunks there are and how long they are is not
 * known yet. */
static pk_chunk_out chunk_out(run *r) {
    return (pk_chunk_out){ r->at, r->late ? 0 : r->nc, r->cc, r->t, (long)r->n, r->retried, r->done,
                           r->late ? 0 : r->total, pk_frame_of(r->m, r->cc.start), r->maybe };
}

int pk_steps_take(pk_steps *r, pk_chunk_out *out) {
    if (r->failed || advance(r))
        return -1;
    if (r->state != DONE)
        return 0;
    *out = chunk_out(r);
    r->at++;
    r->state = AWAIT;
    return 1;
}

void pk_steps_free(pk_steps *r) {
    if (!r)
        return;
    for (size_t i = 0; i < r->jcap; i++) {
        free(r->ct[i]);
        free(r->kt[i]);
    }
    free(r->ct);
    free(r->cn);
    free(r->kt);
    free(r->kn);
    free(r->khead);
    free(r->c);
    free(r->win);
    free(r->sp);
    free(r->cs);
    free(r->vx);
    free(r->vp);
    free(r->t);
    free(r->w);
    free(r->st);
    free(r->text);
    free(r->buf);
    pk_lock_free(r->lock);
    free(r);
}

/* pk_asr_transcribe's own run of the steps: its chunks heard as
 * pk_asr_run_chunks hears them (pk_run_chunks_with: each one's search
 * beside the next one's encoder, several at once where there are threads
 * for it), each chunk's retries on the calling thread in the joint buffer
 * the chunk is done with, and fn for each chunk in order */
typedef struct {
    run *r;
    pk_asr *a;
    pk_chunk_fn fn;
    void *arg;
} own;

static int each(void *arg, size_t i, const pk_tok *toks, long nt, float *spare) {
    own *w = arg;
    pk_job chunk_job = { i, 0, { 0, 0 } }, j;
    pk_seg c;
    pk_chunk_out out;
    int rc;
    chunk(w->r, i, &c, &chunk_job.at);
    if (pk_steps_put(w->r, &chunk_job, toks, nt))
        return -1;
    while ((rc = pk_steps_job(w->r, &j)) == 1) {    /* chunk i's retries: every chunk is out */
        const size_t len = (size_t)(j.at.end - j.at.start);
        if (room(w->r, len) || pk_audio_read(&w->r->x, j.at.start, w->r->buf, len))
            return -1;
        const long more = pk_retry_run(w->a, w->r->buf, len, spare);
        if (more < 0 || pk_steps_put(w->r, &j, w->a->toks, more))
            return -1;
    }
    if (rc < 0 || pk_steps_take(w->r, &out) != 1)
        return -1;
    return w->fn(w->arg, &out) ? 1 : 0;
}

/* A recording still being read: its chunks as the lanes ask for them */
static int next_chunk(void *arg, size_t i, pk_pool *pool, pk_seg *c) {
    run *r = arg;
    const int rc = cut_to(r, i, pool);
    if (rc == 1) {
        pk_seg whole;
        chunk(r, i, &whole, c);
    }
    return rc;
}

/* The retries of a recording that was still being read while its chunks
 * were heard, now that it has ended and its floor and budget are known:
 * each chunk that might have had one (take_chunk) in order, its tokens as
 * the seam left them, its stretches heard again as each() hears them, and
 * the chunk to fn once more, whether or not they gave words. The budget is
 * spent in the chunks' order, as it is when the retries are not put off,
 * and a chunk left out here has no stretch to hear: these are the tokens
 * of the recording read whole. 0, 1 when fn stops it, or -1. */
static int late(own *w) {
    run *r = w->r;
    if (levels(r))
        return -1;
    r->late = 0;
    int rc = 0;
    pk_os_throttle was = pk_os_unthrottle();
    for (size_t i = 0; i < r->nc && i < r->jcap && rc == 0; i++) {
        if (r->kn[i] < 0)
            continue;
        const size_t n = (size_t)r->kn[i];
        size_t stcap = r->stcap;
        if (grow((void **)&r->t, &r->tcap, n + 1, sizeof *r->t) || grow((void **)&r->w, &r->wcap, n + 1, sizeof *r->w)
            || grow((void **)&r->st, &stcap, n + 2, sizeof *r->st)) {
            rc = -1;
            break;
        }
        r->stcap = stcap;
        r->at = i;
        r->cc = r->c[i];
        r->cw = r->win[i];
        if (n)
            memcpy(r->t, r->kt[i], n * sizeof *r->t);
        r->n = n;
        r->k = 0;
        r->retried = 0;
        r->maybe = 0;
        r->nst = stretches(r, &r->cc, r->t, (long)n, r->khead[i], 0, INT64_MAX);
        if (r->nst < 0)
            rc = -1;
        pk_seg at;
        while (rc == 0 && stretch(r, &at)) {
            const size_t len = (size_t)(at.end - at.start);
            long more = -1;
            if (!room(r, len) && !pk_audio_read(&r->x, at.start, r->buf, len))
                more = pk_retry_run(w->a, r->buf, len, w->a->fproj[0]);
            if (more < 0 || heard_again(r, &at, w->a->toks, more))
                rc = -1;
        }
        if (rc == 0) {
            finish(r);
            r->done = r->total;
            const pk_chunk_out out = chunk_out(r);
            rc = w->fn(w->arg, &out) ? 1 : 0;
        }
    }
    pk_os_rethrottle(was);
    return rc;
}

/* x read in turns (pk_audio_turns), since a run of lanes reads its
 * chunks on their threads while the retry reads here. A recording whose
 * length is not known is heard until its first chunk is cut; one that has
 * ended by then, a short one, goes on as a recording of known length does, and one that has not has its chunks cut
 * as the lanes ask for them and its retries at the end (late). */
int pk_asr_transcribe(pk_asr *a, const pk_audio *x, const pk_transcribe_opts *o, pk_chunk_fn fn,
                      void *arg) {
    if (x->n == 0 || (x->n < 0 && !x->wait))
        return x->n ? -1 : 0;
    pk_turns t;
    pk_audio y;
    if (pk_audio_turns(&t, x, &y))
        return -1;
    run *r = run_new(a, &y, o);
    int rc = -1;
    if (r && x->n < 0) {
        /* any sample the cuts ask for has been waited for */
        y.n = r->x.n = INT64_MAX;
        r->wait = x->wait;
        r->wait_arg = x->arg;
    }
    const int first = r ? cut_to(r, x->n < 0 ? 0 : SIZE_MAX, a->pool) : -1;
    own w = { r, a, fn, arg };
    if (first >= 0 && r->cut_over) {
        if (!levels(r)) {
            hand_out_all(r);
            rc = pk_run_chunks_with(a, &y, r->win, r->nc, each, &w);
        }
    } else if (first >= 0) {
        const pk_chunk_src src = { next_chunk, r };
        hand_out_all(r);
        r->late = 1;
        rc = pk_run_source_with(a, &y, &src, (size_t)r->max_len, each, &w);
        if (rc == 0)
            rc = late(&w);
    }
    pk_steps_free(r);
    pk_audio_turns_free(&t);
    return rc;
}
