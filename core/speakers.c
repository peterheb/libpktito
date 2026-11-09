/* Speakers for a transcript, from a diarizer's rows (core/diar.c): which
 * speaker said each word, and each speaker's stretches of speech.
 *
 * The word rule is NeMo's (SpeakerTaggedASR's offline merge,
 * get_word_dict_content_offline in multispk_transcribe_utils, the same at
 * 3.0.0 and at cd61689; docs/DIARIZATION.md, "From frames to words"). NeMo
 * hands it the diarizer's probabilities averaged to the recognizer's
 * encoder frame (80 ms, eight 10 ms rows), and a word's speaker is the one
 * whose probability, averaged over the word's frames and the frame before
 * them (a transducer emits a token a little after it is heard), is largest,
 * each average raised to at least 0.01 first, the first of equals winning.
 * So a word no one was heard saying goes to speaker 0, whoever spoke first.
 *
 * Here a word's frames come from its samples, counted from the recording's
 * start in integers, and every sum is taken in one order, so a word's
 * speaker is the same on every machine. NeMo takes a word of no frames at
 * the recording's last frame from frame n_spk - 1 on (diar_pred_out.shape[1]
 * - 1, the speakers' axis where the frames' is meant); this takes the last
 * frame.
 */
#include "model.h"
#include "pktito.h"

#include <math.h>
#include <stdlib.h>

/* samples in a row of the diarizer's probabilities (core/model.c holds the
 * upsampling to dividing them) */
static int64_t row_samples(const pk_model *d) {
    return (int64_t)d->hop * d->dw.stack / d->dw.upsample;
}

/* avg[s] += the mean of rows [r0, r1) of p, for each of S speakers */
static void add_mean(const float *p, size_t S, int64_t r0, int64_t r1, float *avg) {
    for (size_t s = 0; s < S; s++) {
        float sum = 0;
        for (int64_t r = r0; r < r1; r++)
            sum += p[(size_t)r * S + s];
        avg[s] += sum / (float)(r1 - r0);
    }
}

void pk_speakers(const pk_model *d, const float *p, size_t rows, uint32_t frame, const pk_seg *w, size_t n,
                 int32_t *out) {
    const size_t S = d->dw.n_spk;
    const int64_t row = row_samples(d), F = frame;
    /* frame j holds the rows that start in it: rows [ceil(j F / row),
     * ceil((j + 1) F / row)); the frames are those that hold a row */
    const int64_t nf = rows ? ((int64_t)rows - 1) * row / F + 1 : 0;
    for (size_t i = 0; i < n; i++) {
        float avg[PK_MAX_SPK] = { 0 };
        const int64_t f0 = w[i].start / F, e = (w[i].end + F - 1) / F, f1 = e > f0 ? e : f0 + 1;
        int64_t lo = f0 > 0 ? f0 - 1 : 0, hi = f1 < nf ? f1 : nf;
        if (lo >= hi && !nf) {          /* no rows: every average 0, so speaker 0 */
            out[i] = 0;
            continue;
        }
        if (lo >= hi)
            lo = nf - 1, hi = nf;
        for (int64_t j = lo; j < hi; j++) {
            const int64_t r0 = (j * F + row - 1) / row, r1 = ((j + 1) * F + row - 1) / row;
            add_mean(p, S, r0, r1 < (int64_t)rows ? r1 : (int64_t)rows, avg);
        }
        int32_t best = 0;
        float top = 0;
        for (size_t s = 0; s < S; s++) {
            const float v = fmaxf(avg[s] / (float)(hi - lo), 0.01f);
            if (v > top)
                top = v, best = (int32_t)s;
        }
        out[i] = best;
    }
}

static int by_start(const void *a, const void *b) {
    const pk_spk_seg *x = a, *y = b;
    return x->start != y->start ? (x->start > y->start) - (x->start < y->start)
                                : (x->speaker > y->speaker) - (x->speaker < y->speaker);
}

size_t pk_speaker_segments(const pk_model *d, const float *p, size_t rows, pk_spk_seg *out) {
    const size_t S = d->dw.n_spk;
    const int64_t row = row_samples(d);
    size_t k = 0;
    for (size_t s = 0; s < S; s++)
        for (size_t r = 0; r < rows;) {
            if (!(p[r * S + s] > 0.5f)) {
                r++;
                continue;
            }
            const size_t r0 = r;
            while (r < rows && p[r * S + s] > 0.5f)
                r++;
            out[k++] = (pk_spk_seg){ (int64_t)r0 * row, (int64_t)r * row, (int32_t)s };
        }
    qsort(out, k, sizeof *out, by_start);
    return k;
}
