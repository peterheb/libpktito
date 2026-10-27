/* Fast structured fuzzing for the public state machines that sit around the
 * neural network.  Inputs are arbitrary, but generated segments and buffer
 * capacities obey the contracts in pktito.h. */
#include "pktito.h"

#include "fuzz.h"

#define STATE_MAX_PROBS 512
#define STATE_MAX_SEGS  32
#define STATE_MAX_WORDS 64
#define STATE_MAX_AUDIO 65536

static void segments_case(fz_input *f) {
    pk_seg_params p;
    pk_seg_defaults(&p);
    int rate = fz_u8(f) & 1 ? 8000 : 16000;
    p.threshold = (double)fz_u8(f) / 255.0;
    p.neg_threshold = fz_u8(f) % 3 ? (double)fz_u8(f) / 255.0 : -1;
    p.min_speech_ms = fz_u32(f) % 2001;
    p.max_speech_s = 0.1 + (double)(fz_u32(f) % 20000) / 1000.0;
    p.min_silence_ms = fz_u32(f) % 2001;
    p.speech_pad_ms = fz_u32(f) % 501;
    p.min_silence_at_max_speech_ms = fz_u32(f) % 2001;
    p.use_max_poss_sil_at_max_speech = fz_u8(f) & 1;
    /* Read before the probabilities, which take every word left. */
    uint32_t trim_bits = fz_u32(f), cap_bits = fz_u32(f);

    size_t n = (f->n - f->at) / 4;
    if (n > STATE_MAX_PROBS)
        n = STATE_MAX_PROBS;
    float probs[STATE_MAX_PROBS];
    for (size_t i = 0; i < n; i++)
        probs[i] = fz_float(f);
    int64_t window = rate == 16000 ? 512 : 256;
    int64_t trim = n ? trim_bits % (uint32_t)window : 0;
    int64_t samples = (int64_t)n * window - trim;
    size_t room = 2 * n + 1;
    pk_seg *all = malloc(room * sizeof *all);
    FZ_ASSERT(all);
    size_t count = pk_segments(probs, n, rate, samples, &p, all, room);
    FZ_ASSERT(count <= room);
    FZ_ASSERT(pk_segments(probs, n, rate, samples, &p, NULL, 0) == count);

    /* Half the time a cap at or next to count, where the unfinished
     * output starts; otherwise any cap the buffer holds. */
    size_t near = count + 2 < room ? count + 2 : room;
    size_t small_cap = (cap_bits >> 1) % (cap_bits & 1 ? near : room);
    pk_seg *small = malloc((small_cap ? small_cap : 1) * sizeof *small);
    FZ_ASSERT(small);
    FZ_ASSERT(pk_segments(probs, n, rate, samples, &p, small, small_cap) == count);
    /* Past cap, out is unfinished: the padding pass reads each segment's
     * neighbour and is skipped (core/segment.c), so only a cap that holds
     * every segment has to give the same ones. */
    if (small_cap >= count)
        FZ_ASSERT(!memcmp(all, small, count * sizeof *all));
    /* A segment can be empty: Silero's max-speech cut can end one where it
     * started (pad 0 and max_speech_s 0.1 give 1536-1536 between 0-1024 and
     * 2048-3072, from utils_vad.py v6.2.3 as from pk_segments). */
    for (size_t i = 0; i < count; i++) {
        FZ_ASSERT(all[i].start >= 0 && all[i].end >= all[i].start && all[i].end <= samples);
        if (i)
            FZ_ASSERT(all[i - 1].end <= all[i].start);
    }

    /* The segments into chunks as core/cli.c makes them, max_speech_s being
     * a chunk less the pads and 0.1 s: every chunk is something to hear,
     * fits, and covers the segment it was made from. */
    int64_t max_len = (int64_t)((p.max_speech_s + 2 * p.speech_pad_ms / 1000 + 0.1) * rate);
    float *audio = calloc(samples ? (size_t)samples : 1, sizeof *audio);
    pk_seg *chunk = malloc((2 * count + 2 + (size_t)(samples / max_len)) * sizeof *chunk);
    FZ_ASSERT(audio && chunk);
    size_t nc = pk_chunks(audio, all, count, samples, max_len, chunk);
    FZ_ASSERT(nc <= 2 * count + 2 + (size_t)(samples / max_len));
    for (size_t i = 0; i < nc; i++) {
        FZ_ASSERT(chunk[i].start >= 0 && chunk[i].end > chunk[i].start && chunk[i].end <= samples);
        FZ_ASSERT(chunk[i].end - chunk[i].start <= max_len);
        if (i)
            FZ_ASSERT(chunk[i - 1].end <= chunk[i].start);
    }
    for (size_t i = 0; i < count; i++) {
        int covered = 0;
        for (size_t j = 0; j < nc; j++)
            covered |= chunk[j].start <= all[i].start && chunk[j].end >= all[i].end;
        FZ_ASSERT(covered);
    }
    free(chunk);
    free(audio);

    /* VADIterator pads each end by speech_pad and each start back by it and
     * does not merge what overlaps (pk_segments does), so a start can come
     * before the end reported just ahead of it: speech, one quiet frame and
     * speech again gives START 0, END 2000, START 0 with a 1488-sample pad.
     * What holds is that the events alternate, starting with a start, and
     * that each end is at or after its own start. */
    pk_iter it;
    pk_iter_init(&it, rate, &p);
    int64_t start = -1;
    for (size_t i = 0; i < n; i++) {
        int64_t at;
        int event = pk_iter_step(&it, probs[i], &at);
        FZ_ASSERT(event == PK_NONE || event == PK_START || event == PK_END);
        if (event == PK_START) {
            FZ_ASSERT(start < 0 && at >= 0);
            start = at;
        } else if (event == PK_END) {
            FZ_ASSERT(start >= 0 && at >= start);
            start = -1;
        }
    }
    free(small);
    free(all);
}

static void chunks_case(fz_input *f) {
    int64_t total = 1 + fz_u32(f) % STATE_MAX_AUDIO;
    int64_t max_len = 1 + fz_u32(f) % (uint32_t)total;
    size_t want = fz_u8(f) % (STATE_MAX_SEGS + 1), n = 0;
    pk_seg seg[STATE_MAX_SEGS];
    int64_t at = 0;
    while (n < want && at < total) {
        int64_t gap_room = total - at;
        int64_t gap = gap_room ? fz_u8(f) % (uint32_t)(gap_room < 1024 ? gap_room : 1024) : 0;
        at += gap;
        if (at >= total)
            break;
        int64_t room = total - at;
        int64_t lim = room < max_len ? room : max_len;
        int64_t len = 1 + fz_u32(f) % (uint32_t)lim;
        seg[n++] = (pk_seg){ at, at + len };
        at += len;
    }

    float *audio = calloc((size_t)total, sizeof *audio);
    FZ_ASSERT(audio);
    for (int64_t i = 0; i < total && f->at + 4 <= f->n; i++)
        audio[i] = fz_float(f);
    size_t cap = 2 * n + 2 + (size_t)(total / max_len);
    pk_seg *out = malloc(cap * sizeof *out);
    FZ_ASSERT(out);
    size_t nc = pk_chunks(audio, seg, n, total, max_len, out);
    FZ_ASSERT(nc <= cap);
    for (size_t i = 0; i < nc; i++) {
        FZ_ASSERT(out[i].start >= 0 && out[i].end > out[i].start && out[i].end <= total);
        FZ_ASSERT(out[i].end - out[i].start <= max_len);
        if (i)
            FZ_ASSERT(out[i - 1].end <= out[i].start);
    }
    for (size_t i = 0; i < n; i++) {
        int covered = 0;
        for (size_t j = 0; j < nc; j++)
            covered |= out[j].start <= seg[i].start && out[j].end >= seg[i].end;
        FZ_ASSERT(covered);
    }
    free(out);
    free(audio);
}

static void captions_case(fz_input *f) {
    size_t n = fz_u8(f) % (STATE_MAX_WORDS + 1);
    pk_capword words[STATE_MAX_WORDS];
    pk_cue cues[STATE_MAX_WORDS];
    char text[STATE_MAX_WORDS][32];
    static const char alphabet[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'-";
    int64_t at = 0;
    for (size_t i = 0; i < n; i++) {
        size_t len = 1 + fz_u8(f) % 31;
        for (size_t j = 0; j < len; j++)
            text[i][j] = alphabet[fz_u8(f) % (sizeof alphabet - 1)];
        text[i][len] = 0;
        int64_t overlap = i ? fz_u32(f) % 401 : 0;
        int64_t start = at > overlap ? at - overlap : 0;
        int64_t dur = 1 + fz_u32(f) % 5001;
        words[i] = (pk_capword){ text[i], start, start + dur };
        at = start + dur + fz_u32(f) % 1501;
    }
    int english = (int)(fz_u8(f) % 3) - 1;
    int64_t media_end = fz_u8(f) & 1 ? at + fz_u32(f) % 5001 : 0;
    long nc = pk_captions(words, n, english, media_end, cues);
    FZ_ASSERT(nc >= 0 && (size_t)nc <= n);
    int32_t next = 0;
    for (long i = 0; i < nc; i++) {
        FZ_ASSERT(cues[i].first == next && cues[i].n > 0);
        FZ_ASSERT(cues[i].start >= 0 && cues[i].end > cues[i].start);
        FZ_ASSERT(cues[i].line2 == 0 || (cues[i].line2 > cues[i].first
                  && cues[i].line2 < cues[i].first + cues[i].n));
        if (i)
            FZ_ASSERT(cues[i - 1].end <= cues[i].start);
        next += cues[i].n;
    }
    FZ_ASSERT(next == (int32_t)n);
    (void)pk_language(words, n);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (!size || size > 64 * 1024)
        return 0;
    fz_input f = { data, size, 0 };
    switch (fz_u8(&f) % 3) {
    case 0: segments_case(&f); break;
    case 1: chunks_case(&f); break;
    default: captions_case(&f); break;
    }
    return 0;
}
