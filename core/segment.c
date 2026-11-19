/* Probabilities to speech segments: Silero's two state machines, ported.
 *
 * pk_segments is get_speech_timestamps_from_probs and pk_vad_iter_step is
 * VADIterator.__call__, both from silero_vad/utils_vad.py at v6.2.3, line
 * for line: vadito's port (core/segment.c there), renamed. The port is
 * literal on purpose. These are what a Silero user's timestamps come out
 * of, so an answer that differs by one frame is a different answer, and
 * tests/segcheck.py holds both to the Python on 3000 random and real
 * probability sequences.
 *
 * Literal includes the arithmetic. Python mixes ints and floats freely
 * here: a sample position is an int, a duration in samples is a float
 * (16000 * 30 / 1000 is 480.0), they are compared exactly, and a padded
 * boundary goes back to an int through int(), which truncates, or through
 * //, which floors. So positions are int64_t, durations double, int() is a
 * cast and // is py_floordiv.
 *
 * Two pieces of Python state are smaller here than there. current_speech is
 * a dict that is either empty or holds 'start' alone (its 'end' is written
 * just before it is appended and emptied), so it is a flag and a start.
 * possible_ends is a list that is appended to, emptied, tested for
 * emptiness and asked for its first entry with the largest duration, so it
 * is that entry and a count.
 */
#include "segment.h"

#include <math.h>
#include <stddef.h>

void pk_seg_defaults(pk_seg_params *p) {
    p->threshold = 0.5;
    p->neg_threshold = -1;
    p->min_speech_ms = 250;
    p->max_speech_s = INFINITY;
    p->min_silence_ms = 100;
    p->speech_pad_ms = 30;
    p->min_silence_at_max_speech_ms = 98;
    p->use_max_poss_sil_at_max_speech = 1;
}

static double neg_threshold(const pk_seg_params *p) {
    if (p->neg_threshold >= 0)
        return p->neg_threshold;
    double t = p->threshold - 0.15;
    return t > 0.01 ? t : 0.01;                   /* max(threshold - 0.15, 0.01) */
}

/* Python's a // b for b > 0: rounds toward minus infinity */
static int64_t py_floordiv(int64_t a, int64_t b) {
    int64_t q = a / b;
    return (a % b != 0 && a < 0) ? q - 1 : q;
}

void pk_segs_init(pk_segs *s, int rate, const pk_seg_params *p) {
    pk_seg_params def;
    if (!p) {
        pk_seg_defaults(&def);
        p = &def;
    }
    *s = (pk_segs){ 0 };
    s->window = rate == 16000 ? 512 : 256;
    s->min_speech = rate * p->min_speech_ms / 1000;
    s->speech_pad = rate * p->speech_pad_ms / 1000;
    s->max_speech = rate * p->max_speech_s - s->window - 2 * s->speech_pad;
    s->min_silence = rate * p->min_silence_ms / 1000;
    s->min_silence_at_max = rate * p->min_silence_at_max_speech_ms / 1000;
    s->threshold = p->threshold;
    s->neg = neg_threshold(p);
    s->use_max = p->use_max_poss_sil_at_max_speech;
}

static void put(pk_seg *out, int *n, int64_t start, int64_t end) {
    out[*n].start = start;
    out[*n].end = end;
    (*n)++;
}

/* The body of utils_vad.py's loop over the probabilities, its `continue`s
 * returns here */
int pk_segs_step(pk_segs *s, float p, pk_seg out[2]) {
    const double prob = p;
    const int64_t cur_sample = s->window * s->frames++;
    int count = 0;

    if (prob >= s->threshold && s->temp_end) {
        int64_t sil_dur = cur_sample - s->temp_end;
        if (sil_dur > s->min_silence_at_max) {
            if (s->n_possible == 0 || sil_dur > s->best_dur) {
                s->best_end = s->temp_end;
                s->best_dur = sil_dur;
            }
            s->n_possible++;
        }
        s->temp_end = 0;
        if (s->next_start < s->prev_end)
            s->next_start = cur_sample;
    }

    if (prob >= s->threshold && !s->triggered) {
        s->triggered = 1;
        s->cur = 1;
        s->cur_start = cur_sample;
        return count;
    }

    if (s->triggered && cur_sample - s->cur_start > s->max_speech) {
        if (s->use_max && s->n_possible) {
            s->prev_end = s->best_end;
            put(out, &count, s->cur_start, s->prev_end);
            s->cur = 0;
            s->next_start = s->prev_end + s->best_dur;
            /* sic: prev_end + cur_sample, as utils_vad.py has it */
            if (s->next_start < s->prev_end + cur_sample) {
                s->cur = 1;
                s->cur_start = s->next_start;
            } else {
                s->triggered = 0;
            }
            s->prev_end = s->next_start = s->temp_end = 0;
            s->n_possible = 0;
        } else if (s->prev_end) {
            put(out, &count, s->cur_start, s->prev_end);
            s->cur = 0;
            if (s->next_start < s->prev_end) {
                s->triggered = 0;
            } else {
                s->cur = 1;
                s->cur_start = s->next_start;
            }
            s->prev_end = s->next_start = s->temp_end = 0;
            s->n_possible = 0;
        } else {
            put(out, &count, s->cur_start, cur_sample);
            s->cur = 0;
            s->prev_end = s->next_start = s->temp_end = 0;
            s->triggered = 0;
            s->n_possible = 0;
            return count;
        }
    }

    if (prob < s->neg && s->triggered) {
        if (!s->temp_end)
            s->temp_end = cur_sample;
        int64_t sil_dur_now = cur_sample - s->temp_end;
        if (!s->use_max && sil_dur_now > s->min_silence_at_max)
            s->prev_end = s->temp_end;
        if (sil_dur_now < s->min_silence)
            return count;
        if (s->temp_end - s->cur_start > s->min_speech)
            put(out, &count, s->cur_start, s->temp_end);
        s->cur = 0;
        s->prev_end = s->next_start = s->temp_end = 0;
        s->triggered = 0;
        s->n_possible = 0;
    }
    return count;
}

int pk_segs_end(pk_segs *s, int64_t audio_len, pk_seg *out) {
    if (!(s->cur && audio_len - s->cur_start > s->min_speech))
        return 0;
    out->start = s->cur_start;
    out->end = audio_len;
    return 1;
}

/* A segment to come starts where the open one did, or at a frame not yet
 * taken: every start the loop writes is the frame's own sample or, after a
 * cut at max_speech, a sample past the segment it has just ended. */
int64_t pk_segs_floor(const pk_segs *s) {
    return s->cur ? s->cur_start : s->window * s->frames;
}

size_t pk_segments(const float *probs, size_t n, int rate, int64_t n_samples,
                   const pk_seg_params *p, pk_seg *out, size_t cap) {
    pk_seg_params def;
    if (!p) {
        pk_seg_defaults(&def);
        p = &def;
    }
    pk_segs st;
    pk_segs_init(&st, rate, p);
    const int64_t audio_len = n_samples > 0 ? n_samples : (int64_t)n * st.window;
    const double speech_pad = st.speech_pad;
    size_t count = 0;
    pk_seg got[2];

    for (size_t i = 0; i <= n; i++) {
        const int k = i < n ? pk_segs_step(&st, probs[i], got) : pk_segs_end(&st, audio_len, got);
        for (int q = 0; q < k; q++, count++)
            if (count < cap)
                out[count] = got[q];
    }

    /* The padding pass reads each segment's neighbour, so it needs them all */
    if (count > cap)
        return count;
    for (size_t i = 0; i < count; i++) {
        pk_seg *s = &out[i];
        if (i == 0) {
            double v = s->start - speech_pad;
            s->start = v > 0 ? (int64_t)v : 0;
        }
        if (i != count - 1) {
            int64_t silence = out[i + 1].start - s->end;
            if (silence < 2 * speech_pad) {
                s->end += py_floordiv(silence, 2);
                int64_t v = out[i + 1].start - py_floordiv(silence, 2);
                out[i + 1].start = v > 0 ? v : 0;
            } else {
                double e = s->end + speech_pad;
                s->end = audio_len < e ? audio_len : (int64_t)e;
                double v = out[i + 1].start - speech_pad;
                out[i + 1].start = v > 0 ? (int64_t)v : 0;
            }
        } else {
            double e = s->end + speech_pad;
            s->end = audio_len < e ? audio_len : (int64_t)e;
        }
    }
    return count;
}

/* ---------------------------------------------------- padding, as they come --- */

void pk_pads_init(pk_pads *d, int rate, const pk_seg_params *p) {
    pk_seg_params def;
    if (!p) {
        pk_seg_defaults(&def);
        p = &def;
    }
    *d = (pk_pads){ .speech_pad = rate * p->speech_pad_ms / 1000 };
}

/* The pass above for segment i and i + 1, i the held one. Its end is not
 * held to the recording's length here, which is not known yet: where the
 * silence is two pads or more, the end with its pad stays a pad short of
 * the next segment's start, a sample of the recording. */
int pk_pads_put(pk_pads *d, pk_seg raw, pk_seg *out) {
    if (!d->any) {
        const double v = raw.start - d->speech_pad;
        raw.start = v > 0 ? (int64_t)v : 0;
        d->any = d->has = 1;
        d->held = raw;
        return 0;
    }
    pk_seg *s = &d->held;
    const int64_t silence = raw.start - s->end;
    if (silence < 2 * d->speech_pad) {
        s->end += py_floordiv(silence, 2);
        const int64_t v = raw.start - py_floordiv(silence, 2);
        raw.start = v > 0 ? v : 0;
    } else {
        s->end = (int64_t)(s->end + d->speech_pad);
        const double v = raw.start - d->speech_pad;
        raw.start = v > 0 ? (int64_t)v : 0;
    }
    *out = *s;
    d->held = raw;
    return 1;
}

int pk_pads_end(pk_pads *d, int64_t audio_len, pk_seg *out) {
    if (!d->has)
        return 0;
    const double e = d->held.end + d->speech_pad;
    d->held.end = audio_len < e ? audio_len : (int64_t)e;
    *out = d->held;
    d->has = 0;
    return 1;
}

/* ------------------------------------------------------------ iterator --- */

void pk_vad_iter_init(pk_vad_iter *it, int rate, const pk_seg_params *p) {
    pk_seg_params def;
    if (!p) {
        pk_seg_defaults(&def);
        p = &def;
    }
    it->frame = rate == 16000 ? 512 : 256;
    it->threshold = p->threshold;
    it->neg_threshold = neg_threshold(p);
    it->min_silence_samples = rate * p->min_silence_ms / 1000;
    it->speech_pad_samples = rate * p->speech_pad_ms / 1000;
    it->triggered = 0;
    it->temp_end = 0;
    it->current_sample = 0;
}

int pk_vad_iter_step(pk_vad_iter *it, float prob, int64_t *at) {
    const double pr = prob;
    it->current_sample += it->frame;

    if (pr >= it->threshold && it->temp_end)
        it->temp_end = 0;

    if (pr >= it->threshold && !it->triggered) {
        it->triggered = 1;
        double start = it->current_sample - it->speech_pad_samples - it->frame;
        *at = start > 0 ? (int64_t)start : 0;
        return PK_VAD_START;
    }

    if (pr < it->neg_threshold && it->triggered) {
        if (!it->temp_end)
            it->temp_end = it->current_sample;
        if (it->current_sample - it->temp_end < it->min_silence_samples)
            return PK_VAD_NONE;
        *at = (int64_t)(it->temp_end + it->speech_pad_samples - it->frame);
        it->temp_end = 0;
        it->triggered = 0;
        return PK_VAD_END;
    }
    return PK_VAD_NONE;
}
