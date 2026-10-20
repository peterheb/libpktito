/* pkito: speech in a WAV file to text, from the command line.
 *
 *   pkito talk.wav                   the transcript
 *   pkito --json talk.wav            chunks with their times, text and tokens
 *   pkito --srt talk.wav             subtitles: one cue a chunk
 *   pkito bench talk.wav             how fast this machine transcribes it
 *   pkito info                       the model file: where it came from
 *
 * What it does with a file is what NeMo's model.transcribe() does with one:
 * read it (mean of the channels; another rate is brought to 16 kHz) and
 * transcribe the whole of it at once. A recording longer than --max-chunk
 * is first cut at its silences, which Silero's VAD finds, into chunks no
 * longer than that, each transcribed as a recording of its own
 * (pk_chunks in pkito.h).
 */
#include "pkito.h"
#include "asr.h"
#include "os.h"
#include "resample.h"
#include "wav.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define READ_FRAMES 16384

static const char *usage =
    "usage: pkito [options] FILE.wav...     the transcript (of several: a line each,\n"
    "                                       the file's name, a tab and its text)\n"
    "       pkito bench [options] FILE.wav  how fast this machine transcribes it\n"
    "       pkito segments [options] FILE   Silero VAD's speech segments: start end\n"
    "       pkito probs [options] FILE      every 32 ms frame: time and P(speech)\n"
    "       pkito info [options]            what the model file says about itself\n"
    "FILE may be - for stdin.\n"
    "  -m, --model FILE      a .pkf (default $PKITO_MODEL, then the one .pkf in models/\n"
    "                        here or beside the program)\n"
    "  -t, --threads N       threads (default: one a core)\n"
    "      --max-chunk S     longest stretch transcribed at once, in seconds (30);\n"
    "                        a longer recording is cut at its silences\n"
    "      --json            chunks as JSON: start, end, text and every token's time\n"
    "      --srt             subtitles, a cue a chunk\n"
    "  -v, --verbose         say what was read, how, and how long it took\n";

typedef struct {
    const char *cmd, *in, *model;
    const char **ins;                /* every input file; in is the first */
    int nin, threads, json, srt, verbose;
    double max_chunk;
} opts;

static void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "pkito: ");
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    exit(1);
}

static double num(const char *flag, const char *s, double lo, double hi) {
    char *end;
    double v = s ? strtod(s, &end) : 0;
    if (!s || end == s || *end || !(v >= lo && v <= hi))
        die("%s wants a number in [%g, %g], not '%s'", flag, lo, hi, s ? s : "");
    return v;
}

static void parse(int argc, char **argv, opts *o) {
    memset(o, 0, sizeof *o);
    o->cmd = "text";
    o->max_chunk = 30;
    o->ins = calloc((size_t)argc, sizeof *o->ins);
    if (!o->ins)
        die("out of memory");
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) { fputs(usage, stdout); exit(0); }
        else if (!strcmp(a, "--version")) { puts("pkito " PKITO_VERSION); exit(0); }
        else if (!strcmp(a, "-m") || !strcmp(a, "--model")) { if (!v) die("%s wants a file", a); o->model = v; i++; }
        else if (!strcmp(a, "-t") || !strcmp(a, "--threads")) { o->threads = (int)num(a, v, 1, 256); i++; }
        else if (!strcmp(a, "--max-chunk")) { o->max_chunk = num(a, v, 1, 3600); i++; }
        else if (!strcmp(a, "--json")) o->json = 1;
        else if (!strcmp(a, "--srt")) o->srt = 1;
        else if (!strcmp(a, "-v") || !strcmp(a, "--verbose")) o->verbose = 1;
        else if (a[0] == '-' && a[1]) die("unknown option %s", a);
        else if (!o->nin && !strcmp(o->cmd, "text") && (!strcmp(a, "bench") || !strcmp(a, "info")
                 || !strcmp(a, "segments") || !strcmp(a, "probs")))
            o->cmd = a;
        else o->ins[o->nin++] = a;
    }
    o->in = o->nin ? o->ins[0] : NULL;
    if (o->json && o->srt)
        die("--json or --srt, not both");
    if (o->nin > 1 && (o->srt || strcmp(o->cmd, "text")))
        die("one input file for %s", o->srt ? "--srt" : o->cmd);
    if (!o->in && strcmp(o->cmd, "info")) {
        fputs(usage, stderr);
        exit(1);
    }
}

/* -m, $PKITO_MODEL, or the one .pkf in ./models/ or models/ beside argv[0] */
static pk_model *load_model(const char *flag, const char *argv0) {
    static const char *names[] = { "parakeet-tdt-0.6b-v3.pkf", "parakeet-tdt-0.6b-v2.pkf" };
    char err[512], path[4096], found[2][4096];
    int nfound = 0;
    const char *p = flag ? flag : getenv("PKITO_MODEL");
    if (!p) {
        const char *slash = strrchr(argv0, '/');
        int dir = slash ? (int)(slash - argv0) : 0;
        for (int d = 0; d < 2 && !nfound; d++)
            for (int i = 0; i < 2; i++) {
                if (d == 0)
                    snprintf(path, sizeof path, "models/%s", names[i]);
                else
                    snprintf(path, sizeof path, "%.*s%smodels/%s", dir, argv0, slash ? "/" : "", names[i]);
                FILE *f = fopen(path, "rb");
                if (f) {
                    fclose(f);
                    snprintf(found[nfound++], sizeof found[0], "%s", path);
                }
            }
        if (nfound == 0)
            die("no model: -m FILE.pkf, or put one in models/ (README, \"Try it\", makes one)");
        if (nfound > 1)
            die("two models, %s and %s: say which with -m", found[0], found[1]);
        p = found[0];
    }
    pk_model *m = pk_model_load(p, err, sizeof err);
    if (!m)
        die("%s", err);
    return m;
}

/* The whole file as 16 kHz mono float */
static float *read_audio(const char *path, int rate, size_t *n, const opts *o) {
    char err[256];
    pk_wav_info info;
    pk_wav *w = pk_wav_open(path, &info, err, sizeof err);
    if (!w)
        die("%s", err);
    pk_resampler *rs = NULL;
    if (info.rate != (uint32_t)rate && !(rs = pk_resampler_new(info.rate, (uint32_t)rate)))
        die("%s: cannot resample from %u Hz", path, (unsigned)info.rate);
    float *in = malloc(READ_FRAMES * sizeof *in);
    size_t cap = 1 << 20, got = 0;
    float *x = malloc(cap * sizeof *x);
    if (!in || !x)
        die("out of memory");
    for (;;) {
        size_t r = pk_wav_read(w, in, READ_FRAMES);
        size_t room = rs ? pk_resample_room(rs, r) : r;
        while (got + room > cap)
            if (!(x = realloc(x, (cap *= 2) * sizeof *x)))
                die("out of memory");
        if (rs)
            got += r ? pk_resample(rs, in, r, x + got) : pk_resample_flush(rs, x + got);
        else {
            memcpy(x + got, in, r * sizeof *x);
            got += r;
        }
        if (!r)
            break;
    }
    if (pk_wav_status(w, err, sizeof err))
        fprintf(stderr, "pkito: %s: %s\n", path, err);
    if (o->verbose)
        fprintf(stderr, "pkito: %s: %u Hz, %u channel%s, %.1f s%s\n", path, (unsigned)info.rate,
                (unsigned)info.channels, info.channels == 1 ? "" : "s", (double)got / rate,
                rs ? ", resampled to 16 kHz" : "");
    pk_wav_close(w);
    pk_resampler_free(rs);
    free(in);
    *n = got;
    return x;
}

/* The recording's chunks: the whole of it when it fits, else its speech cut
 * at its silences (pk_chunks). */
static pk_seg *chunks_of(const pk_model *m, const float *x, size_t n, double max_chunk,
                         size_t *nchunks, double *vad_s) {
    const int rate = pk_model_rate(m);
    const int64_t max_len = (int64_t)(max_chunk * rate);
    *vad_s = 0;
    if ((int64_t)n <= max_len) {
        pk_seg *c = malloc(sizeof *c);
        if (!c)
            die("out of memory");
        c->start = 0;
        c->end = (int64_t)n;
        *nchunks = 1;
        return c;
    }
    double t0 = pk_os_now();
    size_t frames = (n + PK_VAD_SAMPLES - 1) / PK_VAD_SAMPLES;
    float *pad = calloc(frames * PK_VAD_SAMPLES, sizeof *pad), *probs = malloc(frames * sizeof *probs);
    pk_vad *v = pk_vad_new(m);
    if (!pad || !probs || !v)
        die("out of memory");
    memcpy(pad, x, n * sizeof *x);
    pk_vad_probs(v, pad, frames, probs);
    pk_vad_free(v);
    free(pad);
    /* segments no longer than a chunk less the padding Silero puts round them */
    pk_seg_params sp;
    pk_seg_defaults(&sp);
    sp.max_speech_s = max_chunk - 2 * sp.speech_pad_ms / 1000 - 0.1;
    size_t cap = 2 * frames + 1;
    pk_seg *seg = malloc(cap * sizeof *seg), *c = malloc((cap + 1) * sizeof *c);
    if (!seg || !c)
        die("out of memory");
    size_t ns = pk_segments(probs, frames, rate, (int64_t)n, &sp, seg, cap);
    *nchunks = pk_chunks(seg, ns, (int64_t)n, max_len, c);
    free(seg);
    free(probs);
    *vad_s = pk_os_now() - t0;
    return c;
}

static void json_str(const char *s) {
    putchar('"');
    for (; *s; s++) {
        unsigned char ch = (unsigned char)*s;
        if (ch == '"' || ch == '\\') printf("\\%c", ch);
        else if (ch < 0x20) printf("\\u%04x", ch);
        else putchar(ch);
    }
    putchar('"');
}

static void srt_time(double s) {
    long ms = lround(s * 1000);
    printf("%02ld:%02ld:%02ld,%03ld", ms / 3600000, ms / 60000 % 60, ms / 1000 % 60, ms % 1000);
}

/* One file through the transcriber: its chunks, printed as -o asks. With
 * several files a line each, the file's name, a tab and its chunks' text. */
static void transcribe_file(const opts *o, const char *path, pk_model *m, pk_asr *a, int many,
                            double *audio_s, double *vad_total, double *asr_total) {
    size_t n;
    const int rate = pk_model_rate(m);
    float *x = read_audio(path, rate, &n, o);
    size_t nc;
    double vad_s;
    pk_seg *c = chunks_of(m, x, n, o->max_chunk, &nc, &vad_s);
    size_t cap = 1 << 16;
    char *text = malloc(cap);
    if (!text)
        die("out of memory");
    double fs = pk_model_frame_seconds(m);
    if (many && !o->json)
        printf("%s\t", path);
    if (o->json && many) {
        printf("{\"file\": ");
        json_str(path);
        printf(", \"chunks\": [");
    } else if (o->json)
        printf("[");
    for (size_t i = 0; i < nc; i++) {
        const pk_tok *t;
        double t0 = pk_os_now();
        long nt = pk_asr_run(a, x + c[i].start, (size_t)(c[i].end - c[i].start), &t);
        *asr_total += pk_os_now() - t0;
        if (nt < 0)
            die("out of memory");
        size_t len;
        while ((len = pk_text(m, t, (size_t)nt, text, cap)) >= cap)
            if (!(text = realloc(text, cap *= 2)))
                die("out of memory");
        double start = (double)c[i].start / rate, end = (double)c[i].end / rate;
        if (o->json) {
            printf("%s\n {\"start\": %.3f, \"end\": %.3f, \"text\": ", i ? "," : "", start, end);
            json_str(text);
            printf(", \"tokens\": [");
            for (long k = 0; k < nt; k++) {
                size_t pl;
                const char *piece = pk_piece(m, t[k].id, &pl);
                char buf[256];
                snprintf(buf, sizeof buf, "%.*s", (int)(pl < 255 ? pl : 255), piece);
                printf("%s{\"id\": %d, \"piece\": ", k ? ", " : "", t[k].id);
                json_str(buf);
                printf(", \"start\": %.2f, \"end\": %.2f}", start + t[k].frame * fs,
                       start + (t[k].frame + t[k].dur) * fs);
            }
            printf("]}");
        } else if (o->srt) {
            printf("%zu\n", i + 1);
            srt_time(start);
            printf(" --> ");
            srt_time(end);
            printf("\n%s\n\n", text);
        } else if (many) {
            printf("%s%s", i && len ? " " : "", text);
        } else if (len) {
            puts(text);
        }
        fflush(stdout);
    }
    if (o->json)
        printf(many ? "\n]}" : "\n]\n");
    else if (many)
        putchar('\n');
    *audio_s += (double)n / rate;
    *vad_total += vad_s;
    free(text);
    free(c);
    free(x);
}

static int transcribe(const opts *o, pk_model *m) {
    pk_asr *a = pk_asr_new(m, o->threads);
    if (!a)
        die("out of memory");
    double audio_s = 0, vad_s = 0, asr_s = 0;
    int many = o->nin > 1;
    if (o->json && many)
        printf("[");
    for (int i = 0; i < o->nin; i++) {
        if (o->json && many)
            printf(i ? ",\n" : "\n");
        transcribe_file(o, o->ins[i], m, a, many, &audio_s, &vad_s, &asr_s);
    }
    if (o->json && many)
        printf("\n]\n");
    if (o->verbose)
        fprintf(stderr, "pkito: %.1f s of audio: VAD %.2f s, transcription %.2f s (%.1fx real "
                "time), %s, %d threads\n", audio_s, vad_s, asr_s, audio_s / (asr_s + vad_s),
                pk_backend(), o->threads ? o->threads : pk_os_cores());
    pk_asr_free(a);
    return 0;
}

/* Transcribe the file five times; the best time is what a quiet machine
 * does. */
static int bench(const opts *o, pk_model *m) {
    size_t n;
    const int rate = pk_model_rate(m);
    float *x = read_audio(o->in, rate, &n, o);
    pk_asr *a = pk_asr_new(m, o->threads);
    if (!a)
        die("out of memory");
    size_t nc;
    double vad_s;
    pk_seg *c = chunks_of(m, x, n, o->max_chunk, &nc, &vad_s);
    double best = 1e30;
    long ntok = 0;
    for (int rep = 0; rep < 5; rep++) {
        double t0 = pk_os_now();
        ntok = 0;
        for (size_t i = 0; i < nc; i++) {
            const pk_tok *t;
            ntok += pk_asr_run(a, x + c[i].start, (size_t)(c[i].end - c[i].start), &t);
        }
        double dt = pk_os_now() - t0;
        best = dt < best ? dt : best;
    }
    double audio = (double)n / rate;
#ifdef PK_PROFILE
    static const char *stage[PK_NPROF] = { "mel", "subsampling", "positions", "layer norms",
        "quantize", "feed-forward", "q k v pos", "attention", "attention out", "conv pw1 pw2",
        "glu + depthwise", "decoder" };
    double total = 0;
    for (int i = 0; i < PK_NPROF; i++) total += pk_prof[i];
    for (int i = 0; i < PK_NPROF; i++)
        printf("  %-16s %7.1f ms a run  %5.1f%%\n", stage[i], pk_prof[i] / 5 * 1e3, 100 * pk_prof[i] / total);
#endif
    printf("model     %s\n", pk_model_name(m));
    printf("backend   %s, %d threads\n", pk_backend(), o->threads ? o->threads : pk_os_cores());
    printf("audio     %.1f s in %zu chunk%s, %ld tokens\n", audio, nc, nc == 1 ? "" : "s", ntok);
    printf("best of 5 %.3f s  %.1fx real time  (VAD %.3f s more)\n", best, audio / best, vad_s);
    free(c);
    free(x);
    pk_asr_free(a);
    return 0;
}

/* Silero's per-frame probabilities over the whole file, as vadito prints
 * them, or get_speech_timestamps' segments from them with Silero's
 * defaults: the VAD inside the model file, on its own. */
static int vad(const opts *o, pk_model *m) {
    size_t n;
    const int rate = pk_model_rate(m);
    float *x = read_audio(o->in, rate, &n, o);
    size_t frames = (n + PK_VAD_SAMPLES - 1) / PK_VAD_SAMPLES;
    float *pad = calloc(frames * PK_VAD_SAMPLES + 1, sizeof *pad);
    float *probs = malloc((frames + 1) * sizeof *probs);
    pk_vad *v = pk_vad_new(m);
    if (!pad || !probs || !v)
        die("out of memory");
    memcpy(pad, x, n * sizeof *x);              /* the last frame padded with zeros, as Silero does */
    pk_vad_probs(v, pad, frames, probs);
    if (!strcmp(o->cmd, "probs")) {
        for (size_t f = 0; f < frames; f++)
            printf("%.3f %.9g\n", (double)f * PK_VAD_SAMPLES / rate, probs[f]);
    } else {
        pk_seg_params sp;
        pk_seg_defaults(&sp);
        size_t cap = 2 * frames + 1;
        pk_seg *seg = malloc(cap * sizeof *seg);
        if (!seg)
            die("out of memory");
        size_t ns = pk_segments(probs, frames, rate, (int64_t)n, &sp, seg, cap);
        for (size_t i = 0; i < ns; i++)
            printf("%.3f %.3f\n", (double)seg[i].start / rate, (double)seg[i].end / rate);
        free(seg);
    }
    pk_vad_free(v);
    free(probs);
    free(pad);
    free(x);
    return 0;
}

static int info(pk_model *m) {
    printf("model     %s\n", pk_model_name(m));
    printf("backend   %s\n", pk_backend());
    const char *k, *v;
    for (int i = 0; pk_model_meta_at(m, i, &k, &v) == 0; i++)
        printf("  %-16s %s\n", k, v);
    return 0;
}

int main(int argc, char **argv) {
    opts o;
    parse(argc, argv, &o);
    pk_model *m = load_model(o.model, argv[0]);
    int rc = !strcmp(o.cmd, "bench") ? bench(&o, m) : !strcmp(o.cmd, "info") ? info(m)
           : !strcmp(o.cmd, "segments") || !strcmp(o.cmd, "probs") ? vad(&o, m)
           : transcribe(&o, m);
    pk_model_free(m);
    free(o.ins);
    return rc;
}
