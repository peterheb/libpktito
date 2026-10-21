/* Probabilities to speech segments: Silero's two state machines, ported.
 *
 * pk_segments is get_speech_timestamps_from_probs and pk_iter_step is
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
#include "pktito.h"

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

static void put(pk_seg *out, size_t cap, size_t *n, int64_t start, int64_t end) {
    if (*n < cap) {
        out[*n].start = start;
        out[*n].end = end;
    }
    (*n)++;
}

size_t pk_segments(const float *probs, size_t n, int rate, int64_t n_samples,
                   const pk_seg_params *p, pk_seg *out, size_t cap) {
    const int64_t window = rate == 16000 ? 512 : 256;
    const int64_t audio_len = n_samples > 0 ? n_samples : (int64_t)n * window;
    const double min_speech = rate * p->min_speech_ms / 1000;
    const double speech_pad = rate * p->speech_pad_ms / 1000;
    const double max_speech = rate * p->max_speech_s - window - 2 * speech_pad;
    const double min_silence = rate * p->min_silence_ms / 1000;
    const double min_silence_at_max = rate * p->min_silence_at_max_speech_ms / 1000;
    const double threshold = p->threshold, neg = neg_threshold(p);
    const int use_max = p->use_max_poss_sil_at_max_speech;

    int triggered = 0;
    int cur = 0;                                  /* current_speech is non-empty */
    int64_t cur_start = 0;
    int64_t temp_end = 0, prev_end = 0, next_start = 0;
    size_t n_possible = 0;                        /* len(possible_ends) */
    int64_t best_end = 0, best_dur = 0;           /* max(possible_ends, key=dur) */
    size_t count = 0;

    for (size_t i = 0; i < n; i++) {
        const double prob = probs[i];
        const int64_t cur_sample = window * (int64_t)i;

        if (prob >= threshold && temp_end) {
            int64_t sil_dur = cur_sample - temp_end;
            if (sil_dur > min_silence_at_max) {
                if (n_possible == 0 || sil_dur > best_dur) {
                    best_end = temp_end;
                    best_dur = sil_dur;
                }
                n_possible++;
            }
            temp_end = 0;
            if (next_start < prev_end)
                next_start = cur_sample;
        }

        if (prob >= threshold && !triggered) {
            triggered = 1;
            cur = 1;
            cur_start = cur_sample;
            continue;
        }

        if (triggered && cur_sample - cur_start > max_speech) {
            if (use_max && n_possible) {
                prev_end = best_end;
                put(out, cap, &count, cur_start, prev_end);
                cur = 0;
                next_start = prev_end + best_dur;
                /* sic: prev_end + cur_sample, as utils_vad.py has it */
                if (next_start < prev_end + cur_sample) {
                    cur = 1;
                    cur_start = next_start;
                } else {
                    triggered = 0;
                }
                prev_end = next_start = temp_end = 0;
                n_possible = 0;
            } else if (prev_end) {
                put(out, cap, &count, cur_start, prev_end);
                cur = 0;
                if (next_start < prev_end) {
                    triggered = 0;
                } else {
                    cur = 1;
                    cur_start = next_start;
                }
                prev_end = next_start = temp_end = 0;
                n_possible = 0;
            } else {
                put(out, cap, &count, cur_start, cur_sample);
                cur = 0;
                prev_end = next_start = temp_end = 0;
                triggered = 0;
                n_possible = 0;
                continue;
            }
        }

        if (prob < neg && triggered) {
            if (!temp_end)
                temp_end = cur_sample;
            int64_t sil_dur_now = cur_sample - temp_end;
            if (!use_max && sil_dur_now > min_silence_at_max)
                prev_end = temp_end;
            if (sil_dur_now < min_silence)
                continue;
            if (temp_end - cur_start > min_speech)
                put(out, cap, &count, cur_start, temp_end);
            cur = 0;
            prev_end = next_start = temp_end = 0;
            triggered = 0;
            n_possible = 0;
            continue;
        }
    }

    if (cur && audio_len - cur_start > min_speech)
        put(out, cap, &count, cur_start, audio_len);

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

/* ------------------------------------------------------------ iterator --- */

void pk_iter_init(pk_iter *it, int rate, const pk_seg_params *p) {
    it->frame = rate == 16000 ? 512 : 256;
    it->threshold = p->threshold;
    it->neg_threshold = neg_threshold(p);
    it->min_silence_samples = rate * p->min_silence_ms / 1000;
    it->speech_pad_samples = rate * p->speech_pad_ms / 1000;
    it->triggered = 0;
    it->temp_end = 0;
    it->current_sample = 0;
}

int pk_iter_step(pk_iter *it, float prob, int64_t *at) {
    const double pr = prob;
    it->current_sample += it->frame;

    if (pr >= it->threshold && it->temp_end)
        it->temp_end = 0;

    if (pr >= it->threshold && !it->triggered) {
        it->triggered = 1;
        double start = it->current_sample - it->speech_pad_samples - it->frame;
        *at = start > 0 ? (int64_t)start : 0;
        return PK_START;
    }

    if (pr < it->neg_threshold && it->triggered) {
        if (!it->temp_end)
            it->temp_end = it->current_sample;
        if (it->current_sample - it->temp_end < it->min_silence_samples)
            return PK_NONE;
        *at = (int64_t)(it->temp_end + it->speech_pad_samples - it->frame);
        it->temp_end = 0;
        it->triggered = 0;
        return PK_END;
    }
    return PK_NONE;
}
