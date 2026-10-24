/* The library from the outside: a WAV file to its transcript, cut at its
 * silences when it is long, in a page. Build with `make examples/transcribe`;
 * it links libpktito.a and includes pktito.h and wav.h, nothing else.
 *
 *     ./examples/transcribe models/parakeet-tdt-0.6b-v2.pkf talk.wav
 *
 * prints each chunk as "start end text", the times in seconds. A 16 kHz
 * file only: resampling is core/resample.h's, which ./pktito uses.
 *
 * The library takes paths in UTF-8 on every system. argv is UTF-8 on
 * Linux and macOS; on Windows it is the ANSI code page's, so a name past
 * it does not arrive here whole. ./pktito asks Windows for the command
 * line in UTF-16 instead (core/cli.c's utf8_args); a program can do the
 * same, or set its code page to UTF-8 in its manifest. */
#include "pktito.h"
#include "wav.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    char err[256];
    if (argc != 3) {
        fprintf(stderr, "usage: transcribe MODEL.pkf FILE.wav\n");
        return 2;
    }
    pk_model *m = pk_model_load(argv[1], err, sizeof err);
    if (!m) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }
    const int rate = pk_model_rate(m);

    /* the whole recording, as float in [-1, 1] */
    pk_wav_info info;
    pk_wav *w = pk_wav_open(argv[2], &info, err, sizeof err);
    if (!w || info.rate != (uint32_t)rate) {
        fprintf(stderr, "%s\n", w ? "want a 16 kHz file" : err);
        return 1;
    }
    size_t cap = 1 << 20, n = 0, k;
    float *x = malloc(cap * sizeof *x);
    while (x && (k = pk_wav_read(w, x + n, cap - n)) > 0)
        if ((n += k) == cap) {
            if (cap > SIZE_MAX / 2 || 2 * cap > SIZE_MAX / sizeof *x) {
                free(x);
                x = NULL;
                break;
            }
            size_t next = 2 * cap;
            float *grown = realloc(x, next * sizeof *x);
            if (!grown) {
                free(x);
                x = NULL;
                break;
            }
            x = grown;
            cap = next;
        }
    pk_wav_close(w);
    if (!x)
        return 1;

    /* Silero's speech segments, then chunks of at most 30 s cut between
     * them, the silence left out; a recording that fits is one chunk */
    size_t frames = (n + PK_VAD_SAMPLES - 1) / PK_VAD_SAMPLES;
    float *padded = calloc(frames * PK_VAD_SAMPLES, sizeof *padded);
    float *probs = malloc(frames * sizeof *probs);
    pk_seg *seg = malloc((2 * frames + 1) * sizeof *seg);
    pk_vad *v = pk_vad_new(m);
    memcpy(padded, x, n * sizeof *x);
    pk_vad_probs(v, padded, frames, probs);
    pk_seg_params sp;
    pk_seg_defaults(&sp);
    sp.max_speech_s = 30 - 0.06 - 0.1;  /* a segment and Silero's 30 ms pads fit in 30 s, as ./pktito has it */
    size_t ns = pk_segments(probs, frames, rate, (int64_t)n, &sp, seg, 2 * frames + 1);
    pk_seg *chunk = malloc((2 * ns + 2 + n / (30 * (size_t)rate)) * sizeof *chunk);
    size_t nc = pk_chunks(x, seg, ns, (int64_t)n, 30 * (int64_t)rate, chunk);

    /* each chunk through the transcriber, as NeMo would hear it as a file */
    pk_asr *a = pk_asr_new(m, 0);
    char text[65536];
    for (size_t i = 0; i < nc; i++) {
        const pk_tok *t;
        long nt = pk_asr_run(a, x + chunk[i].start, (size_t)(chunk[i].end - chunk[i].start), &t);
        if (nt < 0)
            return 1;
        pk_text(m, t, (size_t)nt, text, sizeof text);
        printf("%.2f %.2f %s\n", (double)chunk[i].start / rate, (double)chunk[i].end / rate, text);
    }
    pk_asr_free(a);
    pk_vad_free(v);
    pk_model_free(m);
    free(x); free(padded); free(probs); free(seg); free(chunk);
    return 0;
}
