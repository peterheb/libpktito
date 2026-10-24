/* pktito-c: speech in a WAV file to text, from the command line.
 *
 *   pktito-c talk.wav                   the transcript
 *   pktito-c --json talk.wav            chunks with their times, text and tokens
 *   pktito-c --srt talk.wav             subtitles, or --vtt as WebVTT: pk_captions' cues
 *   pktito-c bench talk.wav             how fast this machine transcribes it
 *   pktito-c info                       the model file: where it came from
 *
 * What it does with a file is what NeMo's model.transcribe() does with one:
 * read it (mean of the channels; another rate is brought to 16 kHz) and
 * transcribe the whole of it at once. A recording longer than --max-chunk
 * is first cut at its silences, which Silero's VAD finds, into chunks no
 * longer than that, each transcribed as a recording of its own, and its
 * silence is left out (pk_chunks in pktito.h).
 */
#include "pktito.h"
#include "asr.h"
#include "os.h"
#include "resample.h"
#include "wav.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>

/* The library takes paths in UTF-8, and Windows' C runtime hands main the
 * command line in the ANSI code page, where a name outside it is already
 * lost; Windows keeps the line itself in UTF-16. One allocation, the
 * pointers and then the strings, sized from the line and kept for the life
 * of the process. Here and not in the library, whose users would all link
 * shell32 for it. NULL if it cannot be had. */
static char **utf8_args(int *argc, char **argv) {
    (void)argv;
    int n;
    wchar_t **w = CommandLineToArgvW(GetCommandLineW(), &n);
    if (!w)
        return NULL;
    size_t bytes = 0;
    for (int i = 0; i < n; i++) {
        int k = WideCharToMultiByte(CP_UTF8, 0, w[i], -1, NULL, 0, NULL, NULL);
        if (k <= 0) {
            LocalFree((HLOCAL)w);
            return NULL;
        }
        bytes += (size_t)k;
    }
    char **out = malloc((size_t)(n + 1) * sizeof *out + bytes);
    if (!out) {
        LocalFree((HLOCAL)w);
        return NULL;
    }
    char *s = (char *)(out + n + 1), *end = s + bytes;
    for (int i = 0; i < n; i++) {
        out[i] = s;
        s += WideCharToMultiByte(CP_UTF8, 0, w[i], -1, s, (int)(end - s), NULL, NULL);
    }
    out[n] = NULL;
    LocalFree((HLOCAL)w);
    *argc = n;
    return out;
}
#else
static char **utf8_args(int *argc, char **argv) {     /* argv is the UTF-8 already */
    (void)argc;
    return argv;
}
#endif

#define READ_FRAMES 16384

static const char *usage =
    "usage: pktito-c [options] FILE.wav...     the transcript (of several: a line each,\n"
    "                                         the file's name, a tab and its text)\n"
    "       pktito-c bench [options] FILE.wav  how fast this machine transcribes it\n"
    "       pktito-c segments [options] FILE   Silero VAD's speech segments: start end\n"
    "       pktito-c probs [options] FILE      every 32 ms frame: time and P(speech)\n"
    "       pktito-c info [options]            what the model file says about itself\n"
    "FILE may be - for stdin.\n"
    "  -m, --model FILE      a .pkf (default $PKTITO_MODEL, then the one .pkf in models/\n"
    "                        here or beside the program)\n"
    "  -t, --threads N       threads (default: one a core)\n"
    "      --max-chunk S     longest stretch transcribed at once, in seconds (30);\n"
    "                        a longer recording is cut at its silences\n"
    "      --json            chunks as JSON: start, end, text, every word's and every\n"
    "                        token's time; and meta, the model, the audio's length and\n"
    "                        its language\n"
    "                        (and with -v, the run's times, backend and threads)\n"
    "      --stream          read FILE as it arrives (a pipe, a microphone) and print\n"
    "                        each stretch of speech as its pause ends it\n"
    "      --pause MS        with --stream, the silence that ends a stretch (1000):\n"
    "                        shorter is sooner and less accurate\n"
    "      --srt, --vtt      subtitles, SRT or WebVTT: cues of one or two lines of 42\n"
    "                        characters, cut where the words' pauses and punctuation\n"
    "                        read best\n"
    "  -v, --verbose         say what was read, how, and how long it took\n";

enum { SUBS_SRT = 1, SUBS_VTT = 2 };

typedef struct {
    const char *cmd, *in, *model;
    const char **ins;                /* every input file; in is the first */
    int nin, threads, json, subs, verbose, stream;   /* subs: SUBS_SRT or SUBS_VTT */
    double max_chunk, pause_ms;
} opts;

static void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "pktito-c: ");
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
    o->pause_ms = 1000;
    o->ins = calloc((size_t)argc, sizeof *o->ins);
    if (!o->ins)
        die("out of memory");
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) { fputs(usage, stdout); exit(0); }
        else if (!strcmp(a, "--version")) { puts("pktito-c " PKTITO_VERSION); exit(0); }
        else if (!strcmp(a, "-m") || !strcmp(a, "--model")) { if (!v) die("%s wants a file", a); o->model = v; i++; }
        else if (!strcmp(a, "-t") || !strcmp(a, "--threads")) { o->threads = (int)num(a, v, 1, 256); i++; }
        else if (!strcmp(a, "--max-chunk")) { o->max_chunk = num(a, v, 1, 3600); i++; }
        else if (!strcmp(a, "--pause")) { o->pause_ms = num(a, v, 32, 10000); i++; }
        else if (!strcmp(a, "--json")) o->json = 1;
        else if (!strcmp(a, "--srt")) o->subs = SUBS_SRT;
        else if (!strcmp(a, "--vtt")) o->subs = SUBS_VTT;
        else if (!strcmp(a, "--stream")) o->stream = 1;
        else if (!strcmp(a, "-v") || !strcmp(a, "--verbose")) o->verbose = 1;
        else if (a[0] == '-' && a[1]) die("unknown option %s", a);
        else if (!o->nin && !strcmp(o->cmd, "text") && (!strcmp(a, "bench") || !strcmp(a, "info")
                 || !strcmp(a, "segments") || !strcmp(a, "probs")))
            o->cmd = a;
        else o->ins[o->nin++] = a;
    }
    o->in = o->nin ? o->ins[0] : NULL;
    const char *subs = o->subs == SUBS_VTT ? "--vtt" : "--srt";
    if (o->json && o->subs)
        die("--json or %s, not both", subs);
    if (o->nin > 1 && (o->subs || o->stream || strcmp(o->cmd, "text")))
        die("one input file for %s", o->subs ? subs : o->stream ? "--stream" : o->cmd);
    if (o->stream && o->subs)
        die("--stream or %s, not both", subs);
    if (!o->in && strcmp(o->cmd, "info")) {
        fputs(usage, stderr);
        exit(1);
    }
}

/* -m, $PKTITO_MODEL, or the one .pkf in ./models/ or models/ beside argv[0] */
static pk_model *load_model(const char *flag, const char *argv0) {
    static const char *names[] = { "parakeet-tdt-0.6b-v3.pkf", "parakeet-tdt-0.6b-v2.pkf" };
    char err[512], path[4096], found[2][4096];
    int nfound = 0;
    const char *p = flag ? flag : getenv("PKTITO_MODEL");
    if (!p) {
        const char *slash = strrchr(argv0, '/');
        int dir = slash ? (int)(slash - argv0) : 0;
        for (int d = 0; d < 2 && !nfound; d++)
            for (int i = 0; i < 2; i++) {
                if (d == 0)
                    snprintf(path, sizeof path, "models/%s", names[i]);
                else
                    snprintf(path, sizeof path, "%.*s%smodels/%s", dir, argv0, slash ? "/" : "", names[i]);
                FILE *f = pk_os_fopen(path, "rb");
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
        if (room > SIZE_MAX - got)
            die("%s: too many samples", path);
        size_t need = got + room;
        while (need > cap) {
            size_t next = cap <= SIZE_MAX / 2 ? cap * 2 : SIZE_MAX;
            if (next < need)
                next = need;
            if (next > SIZE_MAX / sizeof *x)
                die("%s: too many samples", path);
            float *grown = realloc(x, next * sizeof *x);
            if (!grown)
                die("out of memory");
            x = grown;
            cap = next;
        }
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
        fprintf(stderr, "pktito-c: %s: %s\n", path, err);
    if (o->verbose)
        fprintf(stderr, "pktito-c: %s: %u Hz, %u channel%s, %.1f s%s\n", path, (unsigned)info.rate,
                (unsigned)info.channels, info.channels == 1 ? "" : "s", (double)got / rate,
                rs ? ", resampled to 16 kHz" : "");
    pk_wav_close(w);
    pk_resampler_free(rs);
    free(in);
    *n = got;
    return x;
}

/* The recording's chunks (pk_chunks): the whole of it when it fits and has
 * speech, else its speech cut at its silences, and what lies between that
 * is neither silent nor too short to hear. */
static pk_seg *chunks_of(const pk_model *m, const float *x, size_t n, double max_chunk,
                         size_t *nchunks, double *vad_s) {
    const int rate = pk_model_rate(m);
    const int64_t max_len = (int64_t)(max_chunk * rate);
    *vad_s = 0;
    *nchunks = 0;
    if (!n) {
        pk_seg *c = malloc(sizeof *c);
        if (!c)
            die("out of memory");
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
    pk_seg *seg = malloc(cap * sizeof *seg);
    if (!seg)
        die("out of memory");
    size_t ns = pk_segments(probs, frames, rate, (int64_t)n, &sp, seg, cap);
    pk_seg *c = malloc((2 * ns + 2 + n / (size_t)max_len) * sizeof *c);
    if (!c)
        die("out of memory");
    *nchunks = pk_chunks(x, seg, ns, (int64_t)n, max_len, c);
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

/* --json's "meta": the model, the audio's length and its language, the same
 * on every machine and every run, so that a transcript compares byte for
 * byte (as check-c compares test.wav's); with -v, also what the -v line on
 * stderr says of the run. top: the whole run's, with the model and the
 * machine, or else one file's of several; a language for a file's, or for
 * the run's of one file (lang NULL: none found). */
static void json_meta(const opts *o, const pk_model *m, const pk_asr *a, int top, double audio_s,
                      double vad_s, double asr_s, int with_lang, const char *lang) {
    printf("{");
    if (top) {
        printf("\"model\": ");
        json_str(pk_model_name(m));
        printf(", ");
    }
    printf("\"audio\": %.3f", audio_s);
    if (with_lang) {
        printf(", \"language\": ");
        if (lang)
            json_str(lang);
        else
            printf("null");
    }
    if (o->verbose) {
        printf(", \"vad\": %.3f, \"transcription\": %.3f", vad_s, asr_s);
        if (top) {
            printf(", \"speed\": %.1f, \"version\": ", asr_s + vad_s > 0 ? audio_s / (asr_s + vad_s) : 0.0);
            json_str(PKTITO_VERSION);
            printf(", \"backend\": ");
            json_str(pk_backend());
            printf(", \"threads\": %d", pk_asr_threads(a));
        }
    }
    printf("}");
}

/* Subtitles: every word of the file with its time, gathered chunk by
 * chunk, then pk_captions' cues of them all at once (core/captions.c), as
 * SRT or WebVTT. The English rules apply when the model's languages are
 * English alone. */
typedef struct {
    pk_capword *w;
    char *text;                      /* the words' texts, each with its NUL */
    size_t n, cap, len, tcap;
} capwords;

static void cap_add(capwords *c, const char *word, int64_t start, int64_t end) {
    size_t l = strlen(word) + 1;
    if (c->n == c->cap && !(c->w = realloc(c->w, (c->cap = c->cap ? 2 * c->cap : 1024) * sizeof *c->w)))
        die("out of memory");
    while (c->len + l > c->tcap)
        if (!(c->text = realloc(c->text, c->tcap = c->tcap ? 2 * c->tcap : 16384)))
            die("out of memory");
    memcpy(c->text + c->len, word, l);
    /* where the text is, until the buffer stops moving */
    c->w[c->n++] = (pk_capword){ (const char *)(uintptr_t)c->len, start, end };
    c->len += l;
}

static void sub_time(int64_t ms, char sep) {
    printf("%02lld:%02lld:%02lld%c%03lld", (long long)(ms / 3600000), (long long)(ms / 60000 % 60),
           (long long)(ms / 1000 % 60), sep, (long long)(ms % 1000));
}

/* WebVTT reads & < > as markup, so a word's are written as text */
static void vtt_text(const char *t) {
    for (; *t; t++)
        fputs(*t == '&' ? "&amp;" : *t == '<' ? "&lt;" : *t == '>' ? "&gt;" : (char[]){ *t, 0 }, stdout);
}

/* the words' texts as pointers, the buffer done moving */
static void cap_texts(capwords *c) {
    for (size_t i = 0; i < c->n; i++)
        c->w[i].text = c->text + (uintptr_t)c->w[i].text;
}

/* A file's language, for --json's meta: the English model's, or what its
 * words say (pk_language), NULL when they say too little. The words go. */
static const char *file_language(const pk_model *m, capwords *c) {
    const char *langs = pk_model_meta(m, "languages"), *lang = "en";
    if (!langs || strcmp(langs, "en")) {
        cap_texts(c);
        lang = pk_language(c->w, c->n);
    }
    free(c->w);
    free(c->text);
    *c = (capwords){ 0 };
    return lang;
}

static void cap_print(capwords *c, int vtt, int english, int64_t media_end) {
    cap_texts(c);
    pk_cue *cue = malloc((c->n ? c->n : 1) * sizeof *cue);
    if (!cue)
        die("out of memory");
    long nc = pk_captions(c->w, c->n, english, media_end, cue);
    if (nc < 0)
        die("out of memory");
    if (vtt)
        printf("WEBVTT\n\n");
    for (long k = 0; k < nc; k++) {
        printf("%ld\n", k + 1);
        sub_time(cue[k].start, vtt ? '.' : ',');
        printf(" --> ");
        sub_time(cue[k].end, vtt ? '.' : ',');
        putchar('\n');
        for (int32_t i = cue[k].first; i < cue[k].first + cue[k].n; i++) {
            if (i > cue[k].first)
                putchar(i == cue[k].line2 ? '\n' : ' ');
            if (vtt)
                vtt_text(c->w[i].text);
            else
                fputs(c->w[i].text, stdout);
        }
        printf("%s\n\n", cue[k].flags & PK_CUE_PAUSE ? "\xE2\x80\xA6" : "");
    }
    free(cue);
    free(c->w);
    free(c->text);
}

/* One file through the transcriber: its chunks, printed as -o asks. With
 * several files a line each, the file's name, a tab and its chunks' text. */
static void transcribe_file(const opts *o, const char *path, pk_model *m, pk_asr *a, int many,
                            double *audio_s, double *vad_total, double *asr_total, const char **lang) {
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
    capwords words = { 0 };
    if (many && !o->json)
        printf("%s\t", path);
    if (o->json && many) {
        printf("{\"file\": ");
        json_str(path);
        printf(", \"chunks\": [");
    }
    const double asr_before = *asr_total;
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
            pk_word *wd = malloc(((size_t)nt ? (size_t)nt : 1) * sizeof *wd);
            if (!wd)
                die("out of memory");
            size_t nw = pk_words(m, t, (size_t)nt, wd);
            printf(", \"words\": [");
            for (size_t k = 0; k < nw; k++) {
                char word[1024];
                pk_text(m, t + wd[k].tok, (size_t)wd[k].ntok, word, sizeof word);
                printf("%s{\"word\": ", k ? ", " : "");
                json_str(word);
                cap_add(&words, word, 0, 0);
                printf(", \"start\": %.2f, \"end\": %.2f}", start + wd[k].start * fs,
                       start + wd[k].end * fs);
            }
            free(wd);
            printf("], \"tokens\": [");
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
        } else if (o->subs) {
            pk_word *wd = malloc(((size_t)nt ? (size_t)nt : 1) * sizeof *wd);
            if (!wd)
                die("out of memory");
            size_t nw = pk_words(m, t, (size_t)nt, wd);
            for (size_t k = 0; k < nw; k++) {
                char word[1024];
                pk_text(m, t + wd[k].tok, (size_t)wd[k].ntok, word, sizeof word);
                cap_add(&words, word, llround((start + wd[k].start * fs) * 1000), llround((start + wd[k].end * fs) * 1000));
            }
            free(wd);
        } else if (many) {
            printf("%s%s", i && len ? " " : "", text);
        } else if (len) {
            puts(text);
        }
        fflush(stdout);
    }
    if (o->json)
        *lang = file_language(m, &words);
    if (o->json && many) {
        printf("\n], \"meta\": ");
        json_meta(o, m, a, 0, (double)n / rate, vad_s, *asr_total - asr_before, 1, *lang);
        putchar('}');
    } else if (many)
        putchar('\n');
    if (o->subs) {
        /* the English model's are English; the multilingual model's say */
        const char *langs = pk_model_meta(m, "languages");
        cap_print(&words, o->subs == SUBS_VTT, langs && !strcmp(langs, "en") ? 1 : -1, (int64_t)n * 1000 / rate);
    }
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
    if (o->json)
        printf(many ? "{\"files\": [" : "{\"chunks\": [");
    const char *lang = NULL;
    for (int i = 0; i < o->nin; i++) {
        if (o->json && many)
            printf(i ? ",\n" : "\n");
        transcribe_file(o, o->ins[i], m, a, many, &audio_s, &vad_s, &asr_s, &lang);
    }
    if (o->json) {
        printf("\n],\n \"meta\": ");
        json_meta(o, m, a, 1, audio_s, vad_s, asr_s, !many, lang);
        printf("}\n");
    }
    if (o->verbose)
        fprintf(stderr, "pktito-c: %.1f s of audio: VAD %.2f s, transcription %.2f s (%.1fx real "
                "time), %s, %d threads\n", audio_s, vad_s, asr_s, audio_s / (asr_s + vad_s),
                pk_backend(), pk_asr_threads(a));
    pk_asr_free(a);
    return 0;
}

/* --stream: the recording as it arrives. Silero's VADIterator watches it
 * a frame at a time, and each stretch of speech is transcribed when a pause
 * of --pause ends it, with 30 ms of padding either side. The pause is a
 * trade: a line appears that long after the speaker stops, and a shorter
 * stretch is heard with less context. Over tools/corpus.py's long-form set
 * (12 recordings, 45 minutes) WER fell from 3.10% at Silero's own 100 ms to
 * 1.66% at 1000 ms for v2 (4.02% to 1.89% for v3), and no further at 1500
 * (README, "Streaming"). Speech that runs past --max-chunk without a pause
 * is cut at the quietest frame of its last third. The audio is kept only
 * from the start of the current stretch, so a stream may run for days. */
static int stream(const opts *o, pk_model *m) {
    char err[256];
    const int rate = pk_model_rate(m);
    pk_wav_info info;
    pk_wav *w = pk_wav_open(o->in, &info, err, sizeof err);
    if (!w)
        die("%s", err);
    pk_resampler *rs = NULL;
    if (info.rate != (uint32_t)rate && !(rs = pk_resampler_new(info.rate, (uint32_t)rate)))
        die("%s: cannot resample from %u Hz", o->in, (unsigned)info.rate);
    pk_vad *v = pk_vad_new(m);
    pk_asr *a = pk_asr_new(m, o->threads);
    pk_seg_params sp;
    pk_seg_defaults(&sp);
    sp.min_silence_ms = o->pause_ms;
    pk_iter it;
    pk_iter_init(&it, rate, &sp);
    const int64_t max_len = (int64_t)(o->max_chunk * rate);
    /* buf holds the samples from `base` on; probs the frames' probabilities
     * since then, for the cut */
    size_t cap = (size_t)max_len + 4 * rate, have = 0, pcap = cap / PK_VAD_SAMPLES + 2;
    float *buf = malloc(cap * sizeof *buf), *probs = malloc(pcap * sizeof *probs);
    float *in = malloc(PK_VAD_SAMPLES * sizeof *in), *rsout = NULL;
    size_t rsn = 0, rspos = 0;
    if (rs) {
        size_t room = pk_resample_room(rs, PK_VAD_SAMPLES);
        if (room > SIZE_MAX / sizeof *rsout)
            die("%s: resampling needs too many samples", o->in);
        rsout = malloc(room * sizeof *rsout);
    }
    char *text = malloc(1 << 16);
    if (!v || !a || !buf || !probs || !in || (rs && !rsout) || !text)
        die("out of memory");
    int64_t base = 0, start = -1, frame_at = 0;
    double fs = pk_model_frame_seconds(m);
    (void)fs;

    /* transcribe [s, e) of the stream and drop what is before e */
    #define EMIT(s_, e_) do {                                                        \
        int64_t s0 = (s_) < base ? base : (s_), e0 = (e_);                            \
        const pk_tok *t;                                                              \
        /* an end the iterator found before a forced cut is no stretch */             \
        if (e0 <= s0) break;                                                          \
        long nt = pk_asr_run(a, buf + (s0 - base), (size_t)(e0 - s0), &t);            \
        if (nt < 0) die("out of memory");                                             \
        size_t len = pk_text(m, t, (size_t)nt, text, 1 << 16);                        \
        if (len) {                                                                    \
            if (o->json) {                                                            \
                printf("{\"start\": %.3f, \"end\": %.3f, \"text\": ",                  \
                       (double)s0 / rate, (double)e0 / rate);                         \
                json_str(text);                                                       \
                printf("}\n");                                                        \
            } else                                                                    \
                puts(text);                                                           \
            fflush(stdout);                                                           \
        }                                                                             \
    } while (0)

    for (int done = 0; !done;) {
        /* the next frame at the model's rate */
        size_t got = 0;
        while (got < PK_VAD_SAMPLES) {
            if (rs) {
                if (rspos == rsn) {
                    size_t r = pk_wav_read(w, in, PK_VAD_SAMPLES);
                    rsn = r ? pk_resample(rs, in, r, rsout) : pk_resample_flush(rs, rsout);
                    rspos = 0;
                    if (!r && !rsn) { done = 1; break; }
                }
                size_t k = rsn - rspos < PK_VAD_SAMPLES - got ? rsn - rspos : PK_VAD_SAMPLES - got;
                memcpy(in + got, rsout + rspos, k * sizeof *in);
                rspos += k;
                got += k;
            } else {
                size_t r = pk_wav_read(w, in + got, PK_VAD_SAMPLES - got);
                if (!r) { done = 1; break; }
                got += r;
            }
        }
        if (!got)
            break;
        if (got < PK_VAD_SAMPLES)
            memset(in + got, 0, (PK_VAD_SAMPLES - got) * sizeof *in);
        if (have + PK_VAD_SAMPLES > cap) {                     /* never in speech: drop the old */
            size_t keep = (size_t)rate;
            memmove(buf, buf + have - keep, keep * sizeof *buf);
            base += (int64_t)(have - keep);
            have = keep;
        }
        memcpy(buf + have, in, got * sizeof *in);
        have += got;
        float p = pk_vad_prob(v, in);
        size_t fi = (size_t)((frame_at - base) / PK_VAD_SAMPLES);
        if (fi < pcap) probs[fi] = p;
        frame_at += PK_VAD_SAMPLES;
        int64_t at;
        int e = pk_iter_step(&it, p, &at);
        if (e == PK_START)
            start = at;
        else if (e == PK_END && start >= 0) {
            EMIT(start, at);
            start = -1;
        } else if (start >= 0 && frame_at - start >= max_len) {
            /* no pause in max_len: cut at the quietest frame of the last third */
            int64_t from = start + 2 * max_len / 3, best = frame_at;
            float lo = 2;
            for (int64_t f = from; f + PK_VAD_SAMPLES <= frame_at; f += PK_VAD_SAMPLES) {
                size_t k = (size_t)((f - base) / PK_VAD_SAMPLES);
                if (k < pcap && probs[k] < lo) { lo = probs[k]; best = f + PK_VAD_SAMPLES / 2; }
            }
            EMIT(start, best);
            start = best;
        }
        if (start < 0 && have > (size_t)rate) {                /* outside speech keep a second */
            size_t drop = have - (size_t)rate;
            memmove(buf, buf + drop, (size_t)rate * sizeof *buf);
            base += (int64_t)drop;
            have = (size_t)rate;
            size_t df = drop / PK_VAD_SAMPLES;
            if (df < pcap) memmove(probs, probs + df, (pcap - df) * sizeof *probs);
        } else if (start >= 0 && start - base > (int64_t)rate) {
            size_t drop = (size_t)(start - base) - (size_t)rate;
            drop -= drop % PK_VAD_SAMPLES;
            memmove(buf, buf + drop, (have - drop) * sizeof *buf);
            base += (int64_t)drop;
            have -= drop;
            size_t df = drop / PK_VAD_SAMPLES;
            if (df < pcap) memmove(probs, probs + df, (pcap - df) * sizeof *probs);
        }
    }
    if (start >= 0 && base + (int64_t)have > start)
        EMIT(start, base + (int64_t)have);
    #undef EMIT
    pk_wav_close(w);
    pk_resampler_free(rs);
    pk_vad_free(v);
    pk_asr_free(a);
    free(buf);
    free(probs);
    free(in);
    free(rsout);
    free(text);
    return 0;
}

/* Transcribe the file five times; the best time is what a quiet machine
 * does. */
/* A recording not at the model's rate, resampled to it as read_audio does,
 * READ_FRAMES at a time: the best of five, from its samples in memory. 0 for
 * one at the model's rate. */
static double bench_resample(const char *path, int rate, uint32_t *from) {
    char err[256];
    pk_wav_info info;
    pk_wav *w = pk_wav_open(path, &info, err, sizeof err);
    if (!w)
        die("%s", err);
    *from = info.rate;
    if (info.rate == (uint32_t)rate) {
        pk_wav_close(w);
        return 0;
    }
    size_t n = 0, cap = 1 << 20;
    float *in = malloc(cap * sizeof *in), *out = NULL;
    if (!in)
        die("out of memory");
    for (size_t r; (r = pk_wav_read(w, in + n, cap - n)); )
        if ((n += r) == cap && !(in = realloc(in, (cap *= 2) * sizeof *in)))
            die("out of memory");
    pk_wav_close(w);
    double best = 1e30;
    for (int rep = 0; rep < 5; rep++) {
        pk_resampler *rs = pk_resampler_new(info.rate, (uint32_t)rate);
        if (!rs || !(out = realloc(out, pk_resample_room(rs, READ_FRAMES) * sizeof *out)))
            die("out of memory");
        double t0 = pk_os_now();
        for (size_t at = 0; at < n; at += READ_FRAMES)
            pk_resample(rs, in + at, n - at < READ_FRAMES ? n - at : READ_FRAMES, out);
        pk_resample_flush(rs, out);
        double dt = pk_os_now() - t0;
        best = dt < best ? dt : best;
        pk_resampler_free(rs);
    }
    free(in);
    free(out);
    return best;
}

static int bench(const opts *o, pk_model *m) {
    size_t n;
    const int rate = pk_model_rate(m);
    uint32_t from;
    const double rs_s = bench_resample(o->in, rate, &from);
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
        "quantize", "feed-forward", "q k v pos", "attention repack", "attention", "attention out",
        "conv pw1 pw2", "glu + depthwise", "decoder" };
    double total = 0;
    for (int i = 0; i < PK_NPROF; i++) total += pk_prof[i];
    for (int i = 0; i < PK_NPROF; i++)
        printf("  %-16s %7.1f ms a run  %5.1f%%\n", stage[i], pk_prof[i] / 5 * 1e3, 100 * pk_prof[i] / total);
#endif
    printf("model     %s\n", pk_model_name(m));
    printf("backend   %s, %d threads\n", pk_backend(), pk_asr_threads(a));
    printf("audio     %.1f s in %zu chunk%s, %ld tokens\n", audio, nc, nc == 1 ? "" : "s", ntok);
    printf("best of 5 %.3f s  %.1fx real time  (VAD %.3f s more)\n", best, audio / best, vad_s);
    if (rs_s > 0)
        printf("resample  %u Hz to %d, best of 5 %.1f ms\n", (unsigned)from, rate, rs_s * 1e3);
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
    pk_model_info mi;
    pk_model_describe(m, &mi);
    printf("model     %s, %.1f MB\n", pk_model_name(m), mi.file_bytes / 1e6);
    printf("encoder   %u layers of %u, %u heads, feed-forward %u; %u mel bands at %u Hz\n",
           mi.n_layers, mi.d_model, mi.n_heads, mi.ff_dim, mi.n_mels, mi.sample_rate);
    printf("decoder   %u-unit LSTM x %u, joint %u; %u tokens and blank; durations",
           mi.pred_hidden, mi.pred_layers, mi.joint_hidden, mi.vocab);
    for (uint32_t i = 0; i < mi.n_durations; i++)
        printf(" %u", mi.durations[i]);
    printf("\nbackend   %s, %d threads by default\n", pk_backend(), pk_os_cores());
    const char *k, *v;
    for (int i = 0; pk_model_meta_at(m, i, &k, &v) == 0; i++)
        printf("  %-16s %s\n", k, v);
    return 0;
}

int main(int argc, char **argv) {
    opts o;
    pk_os_stdio();
    /* every path the library takes is UTF-8, and on Windows argv is not */
    if (!(argv = utf8_args(&argc, argv)))
        die("cannot read the command line");
    parse(argc, argv, &o);
    pk_model *m = load_model(o.model, argv[0]);
    int rc = !strcmp(o.cmd, "bench") ? bench(&o, m) : !strcmp(o.cmd, "info") ? info(m)
           : !strcmp(o.cmd, "segments") || !strcmp(o.cmd, "probs") ? vad(&o, m)
           : o.stream ? stream(&o, m) : transcribe(&o, m);
    pk_model_free(m);
    free(o.ins);
    return rc;
}
