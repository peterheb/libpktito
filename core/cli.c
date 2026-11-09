/* pktito-c: speech in a WAV or AIFF file to text, from the command line.
 *
 *   pktito-c talk.wav                   the transcript, a sentence a line
 *   pktito-c --json talk.wav            chunks with their times, text and tokens
 *   pktito-c --srt talk.wav             subtitles, or --vtt as WebVTT: pk_captions' cues
 *   pktito-c bench talk.wav             how fast this machine transcribes it
 *   pktito-c info                       the model file: where it came from
 *   pktito-c models                     the files for this machine, and where to get them
 *   pktito-c -m DIAR.pkf talk.wav       who said what: a paragraph a speaker's turn
 *   pktito-c speakers -m DIAR.pkf talk.wav   who speaks when, from the diarizer alone
 *
 * A file is read (mean of the channels; another rate is brought to 16 kHz)
 * and transcribed by pk_asr_transcribe, the library's high-level path: a
 * recording that fits in --max-chunk is heard whole, as NeMo's
 * model.transcribe() hears a file, a longer one is cut at its silences,
 * which Silero's VAD finds, its silence left out, a cut in mid-sentence
 * heard past, and speech the model wrote no word for heard again (README,
 * "Transcribing a recording"). --no-context and --no-retry leave those
 * out: each chunk NeMo's transcript of it.
 *
 * With a diarizer's model named by -m (--speakers says so too) the
 * diarizer (core/diar.c) hears the whole file first, and each word goes to
 * a speaker by NeMo's rule (pk_speakers), the speakers numbered from 1 in
 * the order they first say a word. The diarizer is a preview model, which
 * pktito-c runs only when it is named (models.json's design).
 */
#include "pktito.h"
#include "asr.h"
#include "editions.h"
#include "kernels.h"
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

static void die(const char *fmt, ...);

/* An environment variable in UTF-8, from Windows' own UTF-16 copy: the C
 * runtime's getenv gives it in the ANSI code page, as main's argv, and an
 * older runtime's _wgetenv is made from that. Kept for the life of the
 * process; NULL where it is unset */
static const char *utf8_getenv(const char *name) {
    wchar_t wname[64];
    if (!MultiByteToWideChar(CP_UTF8, 0, name, -1, wname, 64))
        return NULL;
    const DWORD n = GetEnvironmentVariableW(wname, NULL, 0);    /* its length and the NUL; 0 where unset */
    if (!n)
        return NULL;
    wchar_t *w = malloc((size_t)n * sizeof *w);
    char *s = NULL;
    if (w && GetEnvironmentVariableW(wname, w, n) == n - 1) {
        const int k = WideCharToMultiByte(CP_UTF8, 0, w, -1, NULL, 0, NULL, NULL);
        if (k > 0 && (s = malloc((size_t)k)) != NULL)
            WideCharToMultiByte(CP_UTF8, 0, w, -1, s, k, NULL, NULL);
    }
    free(w);
    if (!s)
        die("cannot read $%s", name);
    return s;
}
#else
static char **utf8_args(int *argc, char **argv) {     /* argv is the UTF-8 already */
    (void)argc;
    return argv;
}

static const char *utf8_getenv(const char *name) {    /* the environment is the UTF-8 already */
    return getenv(name);
}
#endif

#define READ_FRAMES 16384

static const char *usage =
    "usage: pktito-c [options] FILE...        transcribe, a sentence a line\n"
    "       pktito-c bench [options] FILE     time a transcription\n"
    "       pktito-c segments [options] FILE  speech start and end times (s)\n"
    "       pktito-c probs [options] FILE     speech probability of each 32 ms frame; with a\n"
    "                                         diarizer's -m, each speaker's each 10 ms\n"
    "       pktito-c speakers [options] FILE  who speaks when: start, end (s), speaker\n"
    "       pktito-c info [options]           describe the model\n"
    "       pktito-c models [--all]           the model files for this machine, and\n"
    "                                         where to get them; --all every one\n"
    "FILE is WAV or AIFF, or - for stdin; several print a line each: name, tab, text.\n"
    "\n"
    "  -m, --model FILE    model (default: $PKTITO_MODEL, else the only model in\n"
    "                      ./models or in models/ beside pktito-c, in its edition\n"
    "                      for this machine where that is all there: pktito-c models);\n"
    "                      twice, a recognizer's and a diarizer's\n"
    "  -t, --threads N     threads (default: one per core)\n"
    "      --fast          up to 7% faster; transcripts may differ slightly\n"
    "      --npu FILE      the encoder on an NPU: a Core ML model (.mlmodelc) on the\n"
    "                      Neural Engine (a build with ANE=1), or a QNN graph (.qnn)\n"
    "                      on a Snapdragon (QNN=1); a slim model's is found beside it\n"
    "      --qnn           with QNN=1, the NPU's graph built here from the whole\n"
    "                      model, in a minute or two, and kept in $PKTITO_QNN_CACHE\n"
    "      --max-chunk S   longest chunk, in seconds (30.72); longer audio is cut\n"
    "                      at pauses\n"
    "      --no-retry      don't retranscribe speech that gave no words\n"
    "      --no-context    don't extend a cut made mid-sentence\n"
    "                      (with --no-retry: each chunk exactly as NeMo has it)\n"
    "      --speakers      who said what, a paragraph a turn, with a diarizer's model\n"
    "                      named by -m (nemotron-3-diarization.pkf), as -m alone asks\n"
    "      --json          chunks with word and token times, and metadata\n"
    "      --srt, --vtt    subtitles\n"
    "      --stream        transcribe FILE as it arrives, a line per pause\n"
    "      --pause MS      with --stream, the pause that ends a line (1000)\n"
    "  -v, --verbose       report the input and timings on stderr (and in --json)\n";

enum { SUBS_SRT = 1, SUBS_VTT = 2 };

typedef struct {
    const char *cmd, *in, *npu;
    const char *models[2];           /* -m's, a recognizer's and a diarizer's, in any order */
    const char **ins;                /* every input file; in is the first */
    int nmodels, nin, threads, json, subs, verbose, stream, fast, no_retry, no_context, qnn, all;   /* subs: SUBS_SRT or SUBS_VTT */
    int speakers;
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

/* pk_wav_open's reason; on Windows, for a name with a wildcard in it, the
 * real one: cmd and PowerShell leave *.wav to the program, which takes
 * names (README, "Try it"), where _wfopen says only "Invalid argument" */
static void cannot_open(const char *path, const char *err) {
#ifdef _WIN32
    if (strpbrk(path, "*?"))
        die("%s: Windows doesn't expand wildcards; see README", path);
#else
    (void)path;
#endif
    die("%s", err);
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
    pk_transcribe_opts d;
    pk_transcribe_defaults(&d);
    o->max_chunk = d.max_chunk_s;
    o->pause_ms = 1000;
    o->ins = calloc((size_t)argc, sizeof *o->ins);
    if (!o->ins)
        die("out of memory");
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) { fputs(usage, stdout); exit(0); }
        else if (!strcmp(a, "--version")) { puts("pktito-c " PKTITO_VERSION); exit(0); }
        else if (!strcmp(a, "-m") || !strcmp(a, "--model")) {
            if (!v) die("%s wants a file", a);
            if (o->nmodels == 2) die("-m at most twice: a recognizer's model and a diarizer's");
            o->models[o->nmodels++] = v;
            i++;
        }
        else if (!strcmp(a, "--speakers")) o->speakers = 1;
        else if (!strcmp(a, "-t") || !strcmp(a, "--threads")) { o->threads = (int)num(a, v, 1, 256); i++; }
        else if (!strcmp(a, "--max-chunk")) { o->max_chunk = num(a, v, 1, 3600); i++; }
        else if (!strcmp(a, "--pause")) { o->pause_ms = num(a, v, 32, 10000); i++; }
        else if (!strcmp(a, "--json")) o->json = 1;
        else if (!strcmp(a, "--srt")) o->subs = SUBS_SRT;
        else if (!strcmp(a, "--vtt")) o->subs = SUBS_VTT;
        else if (!strcmp(a, "--stream")) o->stream = 1;
        else if (!strcmp(a, "--fast")) o->fast = 1;
        else if (!strcmp(a, "--npu")) { if (!v) die("%s wants an encoder: a .mlmodelc or a .qnn", a); o->npu = v; i++; }
        else if (!strcmp(a, "--all")) o->all = 1;
        else if (!strcmp(a, "--qnn")) o->qnn = 1;
        else if (!strcmp(a, "--no-retry")) o->no_retry = 1;
        else if (!strcmp(a, "--no-context")) o->no_context = 1;
        else if (!strcmp(a, "-v") || !strcmp(a, "--verbose")) o->verbose = 1;
        else if (a[0] == '-' && a[1]) die("unknown option %s", a);
        else if (!o->nin && !strcmp(o->cmd, "text") && (!strcmp(a, "bench") || !strcmp(a, "info")
                 || !strcmp(a, "segments") || !strcmp(a, "probs") || !strcmp(a, "models")
                 || !strcmp(a, "speakers")))
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
    if (o->all && strcmp(o->cmd, "models"))
        die("--all is for pktito-c models");
    if (o->speakers && (o->subs || o->stream))
        die("--speakers with %s: not yet", o->subs ? subs : "--stream");
    if (o->speakers && strcmp(o->cmd, "text") && strcmp(o->cmd, "speakers"))
        die("--speakers is for a transcript, not %s", o->cmd);
    if (!o->in && strcmp(o->cmd, "info") && strcmp(o->cmd, "models")) {
        fputs(usage, stderr);
        exit(1);
    }
}

/* -m, $PKTITO_MODEL, or the one .pkf in ./models/ or models/ beside argv[0] */
/* What this build runs: the CPU's editions always, an NPU's where it was
 * built with that NPU's code (make ANE=1, QNN=1) */
static int can_run(const ed_edition *e) {
#ifdef PK_ANE
    if (e->kind == ED_COREML)
        return 1;
#endif
#ifdef PK_QNN
    if (e->kind == ED_QNN)
        return 1;
#endif
    return e->kind == ED_CPU;
}

static char qnn_why[256];    /* why QNN named no chip, for pktito-c models */

static int qnn_chip(char *chip, unsigned cap) {
    return pk_qnn_chip(chip, cap, qnn_why, sizeof qnn_why);
}

/* this machine as models.json's rule sees it, asked once: with QNN=1 QNN
 * loads its library to name the NPU's chip */
static const ed_device *device(void) {
    static ed_device d;
    static int asked;
    if (!asked)
        ed_this_device(&d, qnn_chip), asked = 1;
    return &d;
}

/* an edition's encoder as it sits in a models folder: a Core ML model
 * unpacked from its archive into the folder it names without .aar, as
 * pktito-app unpacks it; a graph as it downloads */
static void encoder_name(const ed_edition *e, char *out, size_t cap) {
    const char *f = e->encoder.file;
    size_t n = strlen(f);
    if (e->kind == ED_COREML && n > 4 && !strcmp(f + n - 4, ".aar"))
        n -= 4;
    snprintf(out, cap, "%.*s", (int)n, f);
}

/* path's size, or -1 where it can't be opened */
static long long file_size(const char *path) {
    FILE *f = pk_os_fopen(path, "rb");
    if (!f)
        return -1;
    long long n = fseek(f, 0, SEEK_END) ? 0 : (long long)ftell(f);
    fclose(f);
    return n;
}

/* e's encoder at path: a Core ML model is a folder, there once its
 * model.mil is */
static int encoder_here(const ed_edition *e, const char *path) {
    char mil[4608 + 16];
    if (e->kind != ED_COREML)
        return file_size(path) >= 0;
    snprintf(mil, sizeof mil, "%s/model.mil", path);
    return file_size(mil) >= 0;
}

/* where path's last slash is, Windows' backslash too, or NULL */
static const char *last_slash(const char *path) {
    const char *slash = strrchr(path, '/');
#ifdef _WIN32
    const char *back = strrchr(path, '\\');
    if (back && (!slash || back > slash))
        slash = back;
#endif
    return slash;
}

/* models/ in the working directory, then models/ beside pktito-c */
static void models_dir(int i, const char *argv0, char *out, size_t cap) {
    const char *slash = last_slash(argv0);
    if (i == 0 || !slash)
        snprintf(out, cap, "models/");
    else
        snprintf(out, cap, "%.*s/models/", (int)(slash - argv0), argv0);
}

/* which of m's files dir holds for edition e: 2 its own all, 1 some, 0 none */
static int edition_here(const ed_edition *e, const char *dir) {
    char p[4608], name[512];
    snprintf(p, sizeof p, "%s%s", dir, e->model.file);
    int n = file_size(p) >= 0;
    if (!e->encoder.file)
        return 2 * n;
    encoder_name(e, name, sizeof name);
    snprintf(p, sizeof p, "%s%s", dir, name);
    n += encoder_here(e, p);
    return n == 2 ? 2 : n > 0;
}

static const char *model_path;    /* the model file, as loaded */

/* The model where neither -m nor $PKTITO_MODEL names one: in ./models,
 * else models/ beside pktito-c, each of models.json's models in its edition
 * for this machine where that edition is all there (a slim file and its
 * encoder), else its whole .pkf, the CPU's; one model, or -m says which */
static const char *find_model(const char *argv0, char *found, size_t cap) {
    char dir[4096], p[4608], slim_alone[4608] = "";
    for (int d = 0; d < 2; d++) {
        models_dir(d, argv0, dir, sizeof dir);
        int n = 0;
        for (int i = 0; i < ed_n_models; i++) {
            const ed_model *m = &ed_catalog[i];
            char pick[4608] = "";
            snprintf(p, sizeof p, "%s%s-slim.pkf", dir, m->name);
            if (file_size(p) >= 0) {
                const ed_edition *e = ed_choose(m, device(), can_run);
                if (e->kind != ED_CPU && edition_here(e, dir) == 2)
                    snprintf(pick, sizeof pick, "%s", p);
                else
                    snprintf(slim_alone, sizeof slim_alone, "%s", p);
            }
            snprintf(p, sizeof p, "%s%s.pkf", dir, m->name);
            if (!pick[0] && file_size(p) >= 0)
                snprintf(pick, sizeof pick, "%s", p);
            if (pick[0] && n++)
                die("two models, %s and %s: say which with -m", found, pick);
            if (pick[0])
                snprintf(found, cap, "%s", pick);
        }
        if (n)
            return found;
    }
    if (slim_alone[0])
        die("%s runs only with its NPU's encoder beside it, in a build with that NPU: pktito-c models says "
            "what runs here, and where to get it", slim_alone);
    die("no model: -m FILE.pkf, or one in models/ (pktito-c models says where to get one; README, \"Try it\", "
        "makes one)");
    return NULL;
}

static pk_model *open_model(const char *path) {
    char err[512];
    pk_model *m = pk_model_load(path, err, sizeof err);
    if (!m)
        die("%s%s", err, strstr(err, "slim model file") ? " (pktito-c models says what runs here)" : "");
    return m;
}

/* the recognizer: -m's, $PKTITO_MODEL, or models/'s */
static pk_model *load_model(const char *flag, const char *argv0) {
    static char found[4608];
    const char *p = flag ? flag : utf8_getenv("PKTITO_MODEL");
    if (!p)
        p = find_model(argv0, found, sizeof found);
    pk_model *m = open_model(p);
    if (pk_model_speakers(m))
        die("%s is a diarizer's model, which says who speaks when, not what is said: --speakers takes it "
            "beside a recognizer's", p);
    model_path = p;
    return m;
}

/* -m names no diarizer: the diarizer is a preview model, which runs only
 * when it is named (models.json's design: none is chosen by itself until
 * one is stable) */
static void no_diarizer(const char *what) {
    die("%s needs a diarizer's model named by -m, such as nemotron-3-diarization.pkf (README, \"Who said "
        "what\", says how to make it)", what);
}

/* A slim model's encoder, beside it: its edition's for this machine, as
 * pktito-c models names it */
static const char *slim_encoder(const pk_model *m, char *out, size_t cap) {
    const ed_model *cm = ed_find(pk_model_name(m));
    const ed_edition *e = cm ? ed_choose(cm, device(), can_run) : NULL;
    if (!e || e->kind == ED_CPU)
        die("%s is a slim model file, which runs only on an NPU, and this pktito-c runs none here: the whole "
            "model runs on the CPU (pktito-c models)", model_path);
    char name[512];
    encoder_name(e, name, sizeof name);
    const char *slash = last_slash(model_path);
    snprintf(out, cap, "%.*s%s", slash ? (int)(slash - model_path + 1) : 0, model_path, name);
    if (!encoder_here(e, out))
        die("%s runs with its encoder for %s, %s, beside it, or one given with --npu (pktito-c models says "
            "where to get it)", model_path, e->about, name);
    return out;
}

/* Where the encoder runs, for -v and bench: NULL for the CPU, else the NPU,
 * and in npu_file the encoder it was given */
static const char *npu_name;
static char npu_file[4608];

/* A transcriber for m, its encoder on the NPU where --npu names one, or the
 * model is slim and its edition's sits beside it; the VAD alone (encodes
 * 0) takes none */
static pk_asr *asr_new(const opts *o, const pk_model *m, int encodes) {
    pk_asr *a = pk_asr_new(m, o->threads);
    if (!a)
        die("out of memory");
    pk_asr_set_fast_mode(a, o->fast);
    char err[512], here[4608];
    const char *npu = o->npu;
    if (encodes && !npu && !o->qnn && !pk_model_has_encoder(m))
        npu = slim_encoder(m, here, sizeof here);
    if (encodes && npu) {
        if (pk_asr_use_encoder(a, npu, err, sizeof err))
            die("%s: %s", npu, err);
        const size_t n = strlen(npu);
        npu_name = n > 4 && !strcmp(npu + n - 4, ".qnn") ? "the Hexagon NPU" : "Apple's Neural Engine";
        snprintf(npu_file, sizeof npu_file, "%s", npu);
    }
    if (encodes && o->qnn) {
        const char *cache = utf8_getenv("PKTITO_QNN_CACHE");
        if (pk_asr_use_qnn(a, 1, cache, err, sizeof err))
            die("--qnn: %s", err);
        npu_name = "the Hexagon NPU";
        snprintf(npu_file, sizeof npu_file, "%s", cache ? cache : "--qnn's graph");
    }
    return a;
}

/* A transcription's -1: memory run out, or, in a slim model, which has no
 * encoder of its own to fall back on, the accelerator's failing or taking
 * no chunk that long */
static void run_failed(const pk_model *m) {
    if (!pk_model_has_encoder(m))
        die("the encoder could not run: %s leaves its own out, and the accelerator's failed or takes "
            "no chunk this long (--max-chunk)", pk_model_name(m));
    die("out of memory");
}

/* The whole file as 16 kHz mono float */
static float *read_audio(const char *path, int rate, size_t *n, const opts *o) {
    char err[256];
    pk_wav_info info;
    pk_wav *w = pk_wav_open(path, &info, err, sizeof err);
    if (!w)
        cannot_open(path, err);
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
 * is neither silent nor too short to hear. The VAD runs on a's threads, as
 * pk_asr_transcribe's does. */
static pk_seg *chunks_of(pk_asr *a, const pk_model *m, const float *x, size_t n, double max_chunk,
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
    /* segments no longer than a chunk less the padding Silero puts round them */
    pk_seg_params sp;
    pk_seg_defaults(&sp);
    sp.max_speech_s = max_chunk - 2 * sp.speech_pad_ms / 1000 - 0.1;
    const size_t cap = 2 * ((n + PK_VAD_SAMPLES - 1) / PK_VAD_SAMPLES) + 1;
    pk_seg *seg = malloc(cap * sizeof *seg);
    const long ns = seg ? pk_asr_speech(a, x, n, &sp, seg, cap) : -1;
    pk_seg *c = ns < 0 ? NULL : malloc((2 * (size_t)ns + 2 + n / (size_t)max_len) * sizeof *c);
    if (!c)
        die("out of memory");
    *nchunks = pk_chunks(x, seg, (size_t)ns, (int64_t)n, max_len, c);
    free(seg);
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

/* --speakers' part of a meta: the diarizer, the speakers a file's words
 * went to (-1: none to say), the seconds it took */
typedef struct {
    const pk_model *dm;
    int heard;
    double secs;
} spk_meta;

/* --json's "meta": the model (and "fast", with --fast), the audio's length
 * and its language, the same on every machine and every run, so that a
 * transcript compares byte for byte (as check-c compares test.wav's); with
 * -v, also what the -v line on stderr says of the run. top: the whole
 * run's, with the model and the machine, or else one file's of several; a
 * language for a file's, or for the run's of one file (lang NULL: none
 * found); sm, with --speakers, the diarizer and its speakers. */
static void json_meta(const opts *o, const pk_model *m, const pk_asr *a, int top, double audio_s,
                      double asr_s, int with_lang, const char *lang, const spk_meta *sm) {
    printf("{");
    if (top) {
        printf("\"model\": ");
        json_str(pk_model_name(m));
        printf(o->fast ? ", \"fast\": true, " : ", ");
    }
    printf("\"audio\": %.3f", audio_s);
    if (with_lang) {
        printf(", \"language\": ");
        if (lang)
            json_str(lang);
        else
            printf("null");
    }
    if (sm && sm->dm) {
        if (top) {
            printf(", \"diarizer\": ");
            json_str(pk_model_name(sm->dm));
        }
        if (sm->heard >= 0)
            printf(", \"speakers\": %d", sm->heard);
    }
    if (o->verbose) {
        printf(", \"transcription\": %.3f", asr_s);   /* the VAD's time with it */
        if (sm && sm->dm)
            printf(", \"diarization\": %.3f", sm->secs);
        if (top) {
            printf(", \"speed\": %.1f, \"version\": ", asr_s > 0 ? audio_s / asr_s : 0.0);
            json_str(PKTITO_VERSION);
            printf(", \"backend\": ");
            json_str(pk_backend());
            printf(", \"compiler\": ");
            json_str(pk_compiler());
            printf(", \"threads\": %d", pk_asr_threads(a));
            if (npu_name) {
                printf(", \"npu\": ");
                json_str(npu_file);
            }
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

static void cap_print(capwords *c, int vtt, const char *lang, int64_t media_end) {
    cap_texts(c);
    pk_cue *cue = malloc((c->n ? c->n : 1) * sizeof *cue);
    if (!cue)
        die("out of memory");
    long nc = pk_captions(c->w, c->n, lang ? lang : pk_language(c->w, c->n), media_end, cue);
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

/* --speakers: a turn, the words of one speaker in a row (NeMo's
 * "sentence") */
typedef struct {
    int32_t speaker;                 /* numbered from 1 */
    double start, end;               /* seconds */
    size_t first, n;                 /* its words, counted in the file */
} turn;

/* A file's chunks as each comes out of the transcriber (pk_asr_run_chunks'
 * fn): what printing one needs, and the seconds the printing took, which
 * the transcription's time leaves out. */
typedef struct {
    const opts *o;
    const pk_model *m;
    int many, rate;
    double fs, printing;
    char *text;
    size_t cap;
    capwords words;
    /* a sentence a line: the tokens in none yet, their frames from the
     * recording's start, and the words' language once they say it */
    pk_tok *tail;
    size_t ntail, tailcap;
    const char *lang;
    /* --speakers: the diarizer's rows of the file, and its turns so far */
    const pk_model *dm;
    const float *dp;
    size_t drows;
    uint32_t frame;                  /* the recognizer's encoder frame, in samples */
    int32_t num[PK_MAX_SPK];         /* each diarizer column's speaker, 0 until it says a word */
    int heard;                       /* speakers numbered so far */
    size_t nwords;                   /* the file's words so far */
    turn *turns;
    size_t nturns, turncap;
    pk_tok *para;                    /* the text's turn so far, its tokens */
    size_t npara, paracap;
    int32_t cur;                     /* its speaker */
    int paras;                       /* turns printed */
} chunk_out;

/* The language of the words so far (pk_language), their texts where the
 * buffer is now, for a file still being heard */
static const char *words_language(const capwords *c) {
    pk_capword *w = malloc((c->n ? c->n : 1) * sizeof *w);
    if (!w)
        die("out of memory");
    for (size_t i = 0; i < c->n; i++)
        w[i] = (pk_capword){ c->text + (uintptr_t)c->w[i].text, 0, 0 };
    const char *lang = pk_language(w, c->n);
    free(w);
    return lang;
}

/* A file's sentences, a line each, as its chunks come: the tokens in none
 * yet, then the chunk's, their frames counted from the recording's start
 * as pk_sentences takes them. A sentence that runs on into the next chunk
 * waits for it, and the last chunk ends the last one. */
static void print_sentences(chunk_out *p, const pk_chunk_out *ch) {
    const pk_model *m = p->m;
    const size_t nt = (size_t)ch->count;
    if (p->ntail + nt > p->tailcap) {
        while (p->ntail + nt > p->tailcap)
            p->tailcap = p->tailcap ? 2 * p->tailcap : 1024;
        if (!(p->tail = realloc(p->tail, p->tailcap * sizeof *p->tail)))
            die("out of memory");
    }
    for (size_t k = 0; k < nt; k++) {
        p->tail[p->ntail] = ch->toks[k];
        p->tail[p->ntail++].frame += ch->at_frame;
    }
    if (!p->lang && nt) {           /* the multilingual model's, until its words say */
        pk_word *wd = malloc(nt * sizeof *wd);
        if (!wd)
            die("out of memory");
        size_t nw = pk_words(m, ch->toks, nt, wd);
        for (size_t k = 0; k < nw; k++) {
            char word[1024];
            pk_text(m, ch->toks + wd[k].tok, (size_t)wd[k].ntok, word, sizeof word);
            cap_add(&p->words, word, 0, 0);
        }
        free(wd);
        p->lang = words_language(&p->words);
    }
    pk_sentence *s = malloc((p->ntail ? p->ntail : 1) * sizeof *s);
    if (!s)
        die("out of memory");
    long ns = pk_sentences(m, p->tail, p->ntail, p->lang, ch->i + 1 == ch->n, s);
    if (ns < 0)
        die("out of memory");
    for (long k = 0; k < ns; k++) {
        size_t len;
        while ((len = pk_text(m, p->tail + s[k].first, (size_t)s[k].n, p->text, p->cap)) >= p->cap)
            if (!(p->text = realloc(p->text, p->cap *= 2)))
                die("out of memory");
        if (len)
            puts(p->text);
    }
    if (ns > 0) {
        const size_t used = (size_t)(s[ns - 1].first + s[ns - 1].n);
        memmove(p->tail, p->tail + used, (p->ntail - used) * sizeof *p->tail);
        p->ntail -= used;
    }
    free(s);
}

/* A chunk's words, and with --speakers each one's speaker (pk_speakers):
 * the speakers numbered from 1 in the order they first say a word, and
 * the file's turns grown by them */
static size_t chunk_words(chunk_out *p, const pk_chunk_out *ch, pk_word *wd, int32_t *spk) {
    const size_t nw = pk_words(p->m, ch->toks, (size_t)ch->count, wd);
    if (!p->dm || !nw)
        return nw;
    pk_seg *at = malloc(nw * sizeof *at);
    if (!at)
        die("out of memory");
    for (size_t k = 0; k < nw; k++)
        at[k] = (pk_seg){ ch->at.start + (int64_t)wd[k].start * p->frame,
                          ch->at.start + (int64_t)wd[k].end * p->frame };
    pk_speakers(p->dm, p->dp, p->drows, p->frame, at, nw, spk);
    free(at);
    const double start = (double)ch->at.start / p->rate;
    for (size_t k = 0; k < nw; k++) {
        if (!p->num[spk[k]])
            p->num[spk[k]] = ++p->heard;
        spk[k] = p->num[spk[k]];
        const double ws = start + wd[k].start * p->fs, we = start + wd[k].end * p->fs;
        turn *t = p->nturns ? &p->turns[p->nturns - 1] : NULL;
        if (t && t->speaker == spk[k]) {
            t->end = we;
            t->n++;
        } else {
            if (p->nturns == p->turncap
                && !(p->turns = realloc(p->turns, (p->turncap = p->turncap ? 2 * p->turncap : 256) * sizeof *p->turns)))
                die("out of memory");
            p->turns[p->nturns++] = (turn){ spk[k], ws, we, p->nwords, 1 };
        }
        p->nwords++;
    }
    return nw;
}

/* --speakers' text: a turn's paragraph, "Speaker 1: " and its words; with
 * several files, the turns on the file's line */
static void flush_turn(chunk_out *p) {
    size_t len;
    while ((len = pk_text(p->m, p->para, p->npara, p->text, p->cap)) >= p->cap)
        if (!(p->text = realloc(p->text, p->cap *= 2)))
            die("out of memory");
    if (p->many)
        printf("%sSpeaker %d: %s", p->paras ? " " : "", (int)p->cur, p->text);
    else
        printf("%sSpeaker %d: %s\n", p->paras ? "\n" : "", (int)p->cur, p->text);
    p->paras++;
    p->npara = 0;
}

/* each turn printed once the next speaker's first word, or the file's
 * end, says it is over */
static void print_turns(chunk_out *p, const pk_chunk_out *ch, const pk_word *wd, size_t nw, const int32_t *spk) {
    for (size_t k = 0; k < nw; k++) {
        if (p->npara && spk[k] != p->cur)
            flush_turn(p);
        p->cur = spk[k];
        const size_t nt = (size_t)wd[k].ntok;
        if (p->npara + nt > p->paracap) {
            while (p->npara + nt > p->paracap)
                p->paracap = p->paracap ? 2 * p->paracap : 1024;
            if (!(p->para = realloc(p->para, p->paracap * sizeof *p->para)))
                die("out of memory");
        }
        memcpy(p->para + p->npara, ch->toks + wd[k].tok, nt * sizeof *p->para);
        p->npara += nt;
    }
    if (ch->i + 1 == ch->n && p->npara)
        flush_turn(p);
}

/* One chunk's tokens, printed as -o asks (pk_asr_transcribe's fn) */
static int print_chunk(void *arg, const pk_chunk_out *ch) {
    chunk_out *p = arg;
    const size_t i = ch->i;
    const pk_tok *t = ch->toks;
    const long nt = ch->count;
    const opts *o = p->o;
    const pk_model *m = p->m;
    const double fs = p->fs, t0 = pk_os_now();
    size_t len;
    while ((len = pk_text(m, t, (size_t)nt, p->text, p->cap)) >= p->cap)
        if (!(p->text = realloc(p->text, p->cap *= 2)))
            die("out of memory");
    double start = (double)ch->at.start / p->rate, end = (double)ch->at.end / p->rate;
    pk_word *wd = NULL;
    int32_t *spk = NULL;
    size_t nw = 0;
    if (o->json || o->subs || p->dm) {
        wd = malloc(((size_t)nt ? (size_t)nt : 1) * sizeof *wd);
        spk = malloc(((size_t)nt ? (size_t)nt : 1) * sizeof *spk);
        if (!wd || !spk)
            die("out of memory");
        nw = chunk_words(p, ch, wd, spk);
    }
    if (o->json) {
        printf("%s\n {\"start\": %.3f, \"end\": %.3f, ", i ? "," : "", start, end);
        if (ch->retried)    /* stretches heard again (pk_asr_transcribe) */
            printf("\"retried\": %d, ", ch->retried);
        printf("\"text\": ");
        json_str(p->text);
        printf(", \"words\": [");
        for (size_t k = 0; k < nw; k++) {
            char word[1024];
            pk_text(m, t + wd[k].tok, (size_t)wd[k].ntok, word, sizeof word);
            printf("%s{\"word\": ", k ? ", " : "");
            json_str(word);
            cap_add(&p->words, word, 0, 0);
            printf(", \"start\": %.2f, \"end\": %.2f", start + wd[k].start * fs, start + wd[k].end * fs);
            if (p->dm)
                printf(", \"speaker\": %d", (int)spk[k]);
            putchar('}');
        }
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
        for (size_t k = 0; k < nw; k++) {
            char word[1024];
            pk_text(m, t + wd[k].tok, (size_t)wd[k].ntok, word, sizeof word);
            cap_add(&p->words, word, llround((start + wd[k].start * fs) * 1000), llround((start + wd[k].end * fs) * 1000));
        }
    } else if (p->dm) {
        print_turns(p, ch, wd, nw, spk);
    } else if (p->many) {
        printf("%s%s", i && len ? " " : "", p->text);
    } else {
        print_sentences(p, ch);
    }
    free(wd);
    free(spk);
    fflush(stdout);
    p->printing += pk_os_now() - t0;
    return 0;
}

/* A diarizer's rows, kept as pk_diar_run hands them over */
typedef struct {
    float *p;
    size_t n, cap, S;
} rows_buf;

static int keep_rows(void *arg, int64_t row, size_t n, const float *p) {
    rows_buf *b = arg;
    (void)row;                       /* they come in order, each once */
    if (b->n + n > b->cap) {
        size_t cap = b->cap ? b->cap : 4096;
        while (cap < b->n + n)
            cap *= 2;
        float *q = realloc(b->p, cap * b->S * sizeof *q);
        if (!q)
            return 1;
        b->p = q;
        b->cap = cap;
    }
    memcpy(b->p + b->n * b->S, p, n * b->S * sizeof *p);
    b->n += n;
    return 0;
}

/* x[0..n) through the diarizer, its rows kept: the seconds it took */
static double diarize(pk_diar *dz, const pk_model *dm, const float *x, size_t n, rows_buf *rb) {
    *rb = (rows_buf){ NULL, 0, 0, (size_t)pk_model_speakers(dm) };
    const double t0 = pk_os_now();
    if (pk_diar_run(dz, x, n, keep_rows, rb))
        die("out of memory");
    return pk_os_now() - t0;
}

/* One file through the transcriber: its chunks, printed as -o asks, by
 * default a sentence a line. With several files a line each, the file's
 * name, a tab and its chunks' text. With --speakers (dz), the diarizer
 * hears the file first, and each word goes to a speaker. */
static void transcribe_file(const opts *o, const char *path, pk_model *m, pk_asr *a, pk_diar *dz,
                            const pk_model *dm, int many, double *audio_s, double *asr_total,
                            double *diar_total, const char **lang, int *heard) {
    size_t n;
    const int rate = pk_model_rate(m);
    float *x = read_audio(path, rate, &n, o);
    const char *langs = pk_model_meta(m, "languages");
    chunk_out out = { .o = o, .m = m, .many = many, .rate = rate, .fs = pk_model_frame_seconds(m),
                      .text = malloc(1 << 16), .cap = 1 << 16, .lang = langs && !strcmp(langs, "en") ? "en" : NULL };
    if (!out.text)
        die("out of memory");
    rows_buf rb = { 0 };
    double diar_s = 0;
    if (dz) {
        diar_s = diarize(dz, dm, x, n, &rb);
        *diar_total += diar_s;
        out.dm = dm;
        out.dp = rb.p;
        out.drows = rb.n;
        out.frame = (uint32_t)llround(out.fs * rate);
    }
    if (many && !o->json)
        printf("%s\t", path);
    if (o->json && many) {
        printf("{\"file\": ");
        json_str(path);
        printf(", \"chunks\": [");
    }
    pk_transcribe_opts to;
    pk_transcribe_defaults(&to);
    to.max_chunk_s = o->max_chunk;
    to.retry = !o->no_retry;
    to.context = !o->no_context;
    const double t0 = pk_os_now();
    if (pk_asr_transcribe(a, x, n, &to, print_chunk, &out) < 0)
        run_failed(m);
    const double asr_s = pk_os_now() - t0 - out.printing;
    *asr_total += asr_s;
    *heard = out.heard;
    if (o->json)
        *lang = file_language(m, &out.words);
    if (o->json && dz) {             /* the turns, then the chunks' and their closing ] */
        printf("\n],\n \"turns\": [");
        for (size_t k = 0; k < out.nturns; k++) {
            const turn *tn = &out.turns[k];
            printf("%s\n {\"speaker\": %d, \"start\": %.2f, \"end\": %.2f, \"word\": %zu, \"words\": %zu}",
                   k ? "," : "", (int)tn->speaker, tn->start, tn->end, tn->first, tn->n);
        }
    }
    if (o->json && many) {
        const spk_meta sm = { dm, out.heard, diar_s };
        printf("\n], \"meta\": ");
        json_meta(o, m, a, 0, (double)n / rate, asr_s, 1, *lang, &sm);
        putchar('}');
    } else if (many)
        putchar('\n');
    if (o->subs) {
        /* the English model's are English; the multilingual model's say */
        cap_print(&out.words, o->subs == SUBS_VTT, out.lang, (int64_t)n * 1000 / rate);
    }
    *audio_s += (double)n / rate;
    if (!o->json && !o->subs) {     /* the words a sentence's language was read from */
        free(out.words.w);
        free(out.words.text);
    }
    free(out.tail);
    free(out.text);
    free(out.turns);
    free(out.para);
    free(rb.p);
    free(x);
}

/* dm, a diarizer's model, for --speakers, or NULL */
static int transcribe(const opts *o, pk_model *m, const pk_model *dm) {
    pk_asr *a = asr_new(o, m, 1);
    pk_diar *dz = dm ? pk_diar_new(dm, o->threads) : NULL;
    if (dm && !dz)
        die("out of memory");
    double audio_s = 0, asr_s = 0, diar_s = 0;
    int many = o->nin > 1, heard = 0;
    if (o->json)
        printf(many ? "{\"files\": [" : "{\"chunks\": [");
    const char *lang = NULL;
    for (int i = 0; i < o->nin; i++) {
        if (o->json && many)
            printf(i ? ",\n" : "\n");
        transcribe_file(o, o->ins[i], m, a, dz, dm, many, &audio_s, &asr_s, &diar_s, &lang, &heard);
    }
    if (o->json) {
        const spk_meta sm = { dm, many ? -1 : heard, diar_s };
        printf("\n],\n \"meta\": ");
        json_meta(o, m, a, 1, audio_s, asr_s, !many, lang, &sm);
        printf("}\n");
    }
    char on[sizeof npu_file + 64] = "", who[96] = "";
    if (npu_name)
        snprintf(on, sizeof on, ", the encoder on %s (%s)", npu_name, npu_file);
    if (dz)
        snprintf(who, sizeof who, "; who spoke, %.2f s more (%.1fx)", diar_s, diar_s > 0 ? audio_s / diar_s : 0.0);
    if (o->verbose)
        fprintf(stderr, "pktito-c: %.1f s of audio: transcribed in %.2f s with its VAD (%.1fx "
                "real time)%s, %s, %d threads%s%s\n", audio_s, asr_s, asr_s > 0 ? audio_s / asr_s : 0.0, who,
                pk_backend(), pk_asr_threads(a), o->fast ? ", fast mode" : "", on);
    pk_diar_free(dz);
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
        cannot_open(o->in, err);
    pk_resampler *rs = NULL;
    if (info.rate != (uint32_t)rate && !(rs = pk_resampler_new(info.rate, (uint32_t)rate)))
        die("%s: cannot resample from %u Hz", o->in, (unsigned)info.rate);
    pk_vad *v = pk_vad_new(m);
    pk_asr *a = asr_new(o, m, 1);
    pk_seg_params sp;
    pk_seg_defaults(&sp);
    sp.min_silence_ms = o->pause_ms;
    pk_vad_iter it;
    pk_vad_iter_init(&it, rate, &sp);
    const int64_t max_len = (int64_t)(o->max_chunk * rate);
    /* buf holds the samples from `base` on; probs the frames' probabilities
     * since then, for the cut */
    size_t cap = (size_t)max_len + 4 * rate, have = 0, pcap = cap / PK_VAD_SAMPLES + 2;
    float *buf = malloc(cap * sizeof *buf), *probs = malloc(pcap * sizeof *probs);
    /* in is the next frame at the model's rate; raw, a read at the file's
     * rate on its way to the resampler */
    float *in = malloc(PK_VAD_SAMPLES * sizeof *in), *raw = NULL, *rsout = NULL;
    size_t rsn = 0, rspos = 0;
    if (rs) {
        size_t room = pk_resample_room(rs, PK_VAD_SAMPLES);
        if (room > SIZE_MAX / sizeof *rsout)
            die("%s: resampling needs too many samples", o->in);
        raw = malloc(PK_VAD_SAMPLES * sizeof *raw);
        rsout = malloc(room * sizeof *rsout);
    }
    char *text = malloc(1 << 16);
    if (!v || !buf || !probs || !in || (rs && (!raw || !rsout)) || !text)
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
        if (nt < 0) run_failed(m);                                                    \
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
                    /* into raw: a refill can come mid-frame, with in part full */
                    size_t r = pk_wav_read(w, raw, PK_VAD_SAMPLES);
                    rsn = r ? pk_resample(rs, raw, r, rsout) : pk_resample_flush(rs, rsout);
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
        int e = pk_vad_iter_step(&it, p, &at);
        if (e == PK_VAD_START)
            start = at;
        else if (e == PK_VAD_END && start >= 0) {
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
    free(raw);
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
        cannot_open(path, err);
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

#ifdef PK_PROFILE
/* make PROFILE=1: each named stage's mean time over bench's five runs, and
 * its share of theirs */
typedef struct { int i; const char *name; } prof_row;

static void print_profile(const prof_row *r, int n) {
    double total = 0;
    for (int k = 0; k < n; k++) total += pk_prof[r[k].i];
    for (int k = 0; k < n; k++)
        printf("  %-16s %7.1f ms a run  %5.1f%%\n", r[k].name, pk_prof[r[k].i] / 5 * 1e3,
               100 * pk_prof[r[k].i] / total);
}
#endif

static int count_tokens(void *arg, const pk_chunk_out *c) {
    *(long *)arg += c->count;
    return 0;
}

static int bench(const opts *o, pk_model *m) {
    size_t n;
    const int rate = pk_model_rate(m);
    uint32_t from;
    const double rs_s = bench_resample(o->in, rate, &from);
    float *x = read_audio(o->in, rate, &n, o);
    pk_asr *a = asr_new(o, m, 1);
    size_t nc;
    double vad_s;
    pk_seg *c = chunks_of(a, m, x, n, o->max_chunk, &nc, &vad_s);
    double best = 1e30;
    long ntok = 0;
    for (int rep = 0; rep < 5; rep++) {
        double t0 = pk_os_now();
        ntok = 0;
        if (pk_asr_run_chunks(a, x, c, nc, count_tokens, &ntok))
            run_failed(m);
        double dt = pk_os_now() - t0;
        best = dt < best ? dt : best;
    }
    double audio = (double)n / rate;
#ifdef PK_PROFILE
    static const prof_row stage[] = { { PK_P_MEL, "mel" }, { PK_P_SUB, "subsampling" },
        { PK_P_POS, "positions" }, { PK_P_NORM, "layer norms" }, { PK_P_QUANT, "quantize" },
        { PK_P_FF, "feed-forward" }, { PK_P_QKV, "q k v pos" }, { PK_P_ATT, "attention" },
        { PK_P_ATTO, "attention out" }, { PK_P_CONV, "conv pw1 pw2" }, { PK_P_DW, "glu + depthwise" },
        { PK_P_DEC, "decoder" } };
    print_profile(stage, (int)(sizeof stage / sizeof *stage));
#endif
    printf("model     %s\n", pk_model_name(m));
    printf("backend   %s, %d threads\n", pk_backend(), pk_asr_threads(a));
    if (npu_name)
        printf("encoder   %s: %s\n", npu_name, npu_file);
    if (o->fast)
        printf("mode      fast: the encoder's activations in blocks of 64\n");
    printf("compiler  %s\n", pk_compiler());
    printf("audio     %.1f s in %zu chunk%s, %ld tokens\n", audio, nc, nc == 1 ? "" : "s", ntok);
    printf("best of 5 %.3f s  %.1fx real time  (VAD %.3f s more)\n", best, audio / best, vad_s);
    if (rs_s > 0)
        printf("resample  %u Hz to %d, best of 5 %.1f ms\n", (unsigned)from, rate, rs_s * 1e3);
    free(c);
    free(x);
    pk_asr_free(a);
    return 0;
}

/* bench on a diarizer's model: the diarizer alone over the file, best of 5 */
static int bench_diar(const opts *o, const pk_model *dm) {
    size_t n;
    const int rate = pk_model_rate(dm);
    float *x = read_audio(o->in, rate, &n, o);
    pk_diar *dz = pk_diar_new(dm, o->threads);
    if (!dz)
        die("out of memory");
    double best = 1e30;
    rows_buf rb = { 0 };
    for (int rep = 0; rep < 5; rep++) {
        free(rb.p);
        const double dt = diarize(dz, dm, x, n, &rb);
        best = dt < best ? dt : best;
    }
    const double audio = (double)n / rate;
#ifdef PK_PROFILE
    static const prof_row stage[] = { { PK_P_MEL, "features" }, { PK_P_SUB, "stacking" },
        { PK_P_NORM, "layer norm" }, { PK_P_QUANT, "norm + quantize" }, { PK_P_QKV, "q k v" },
        { PK_P_POS, "rotation" }, { PK_P_ATT, "attention" }, { PK_P_ATTO, "attention out" },
        { PK_P_FF, "ff + GELU" }, { PK_P_HEAD, "head" }, { PK_P_CACHE, "cache" } };
    print_profile(stage, (int)(sizeof stage / sizeof *stage));
#endif
    printf("model     %s\n", pk_model_name(dm));
    printf("backend   %s, %d threads\n", pk_backend(), pk_diar_threads(dz));
    printf("compiler  %s\n", pk_compiler());
    printf("audio     %.1f s, %zu rows of %d speakers\n", audio, rb.n, pk_model_speakers(dm));
    printf("best of 5 %.3f s  %.1fx real time\n", best, audio / best);
    free(rb.p);
    free(x);
    pk_diar_free(dz);
    return 0;
}

/* pktito-c probs on a diarizer's model: a line a row, its start in seconds
 * and each speaker's probability, as the VAD's are printed (vad, below) */
typedef struct {
    size_t S;
    int64_t row;                     /* samples a row */
    int rate;
} row_printer;

static int print_rows(void *arg, int64_t row, size_t n, const float *p) {
    const row_printer *rp = arg;
    for (size_t i = 0; i < n; i++) {
        printf("%.3f", (double)((row + (int64_t)i) * rp->row) / rp->rate);
        for (size_t s = 0; s < rp->S; s++)
            printf(" %.9g", p[i * rp->S + s]);
        putchar('\n');
    }
    return 0;
}

static int diar_probs(const opts *o, const pk_model *dm) {
    size_t n;
    const int rate = pk_model_rate(dm);
    float *x = read_audio(o->in, rate, &n, o);
    pk_diar *dz = pk_diar_new(dm, o->threads);
    if (!dz)
        die("out of memory");
    row_printer rp = { (size_t)pk_model_speakers(dm), llround(pk_model_frame_seconds(dm) * rate), rate };
    pk_diar_run(dz, x, n, print_rows, &rp);
    free(x);
    pk_diar_free(dz);
    return 0;
}

/* pktito-c speakers: who speaks when, from the diarizer alone. Each
 * speaker's stretches of speech (pk_speaker_segments), a line each, start
 * and end in seconds and the speaker, numbered from 1 in the order they
 * first speak; --json, the same as "segments" and a "meta". */
static int speakers(const opts *o, const pk_model *dm) {
    size_t n;
    const int rate = pk_model_rate(dm);
    float *x = read_audio(o->in, rate, &n, o);
    pk_diar *dz = pk_diar_new(dm, o->threads);
    if (!dz)
        die("out of memory");
    rows_buf rb;
    const double secs = diarize(dz, dm, x, n, &rb);
    pk_spk_seg *seg = malloc((rb.S * (rb.n + 1) / 2 + 1) * sizeof *seg);
    if (!seg)
        die("out of memory");
    const size_t ns = pk_speaker_segments(dm, rb.p, rb.n, seg);
    int32_t num[PK_MAX_SPK] = { 0 };
    int heard = 0;
    if (o->json)
        printf("{\"segments\": [");
    for (size_t i = 0; i < ns; i++) {
        if (!num[seg[i].speaker])
            num[seg[i].speaker] = ++heard;
        const double a = (double)seg[i].start / rate, b = (double)seg[i].end / rate;
        if (o->json)
            printf("%s\n {\"start\": %.3f, \"end\": %.3f, \"speaker\": %d}", i ? "," : "", a, b,
                   (int)num[seg[i].speaker]);
        else
            printf("%.3f %.3f %d\n", a, b, (int)num[seg[i].speaker]);
    }
    if (o->json) {
        printf("\n],\n \"meta\": {\"model\": ");
        json_str(pk_model_name(dm));
        printf(", \"audio\": %.3f, \"speakers\": %d", (double)n / rate, heard);
        if (o->verbose)
            printf(", \"diarization\": %.3f, \"backend\": \"%s\", \"threads\": %d", secs, pk_backend(),
                   pk_diar_threads(dz));
        printf("}}\n");
    }
    if (o->verbose)
        fprintf(stderr, "pktito-c: %.1f s of audio: %d speakers heard in %.2f s (%.1fx real time), %s, %d "
                "threads\n", (double)n / rate, heard, secs, secs > 0 ? (double)n / rate / secs : 0.0,
                pk_backend(), pk_diar_threads(dz));
    free(seg);
    free(rb.p);
    free(x);
    pk_diar_free(dz);
    return 0;
}

/* Silero's per-frame probabilities over the whole file, as vadito prints
 * them, or get_speech_timestamps' segments from them with Silero's
 * defaults: the VAD inside the model file, on its own, over -t's threads
 * (pk_asr_vad, pk_asr_speech). */
static int vad(const opts *o, pk_model *m) {
    size_t n;
    const int rate = pk_model_rate(m);
    float *x = read_audio(o->in, rate, &n, o);
    pk_asr *a = asr_new(o, m, 0);
    const size_t frames = (n + PK_VAD_SAMPLES - 1) / PK_VAD_SAMPLES, cap = 2 * frames + 1;
    if (!strcmp(o->cmd, "probs")) {
        float *probs = malloc((frames + 1) * sizeof *probs);
        if (!probs || pk_asr_vad(a, x, n, probs) < 0)
            die("out of memory");
        for (size_t f = 0; f < frames; f++)
            printf("%.3f %.9g\n", (double)f * PK_VAD_SAMPLES / rate, probs[f]);
        free(probs);
    } else {
        pk_seg *seg = malloc(cap * sizeof *seg);
        const long ns = seg ? pk_asr_speech(a, x, n, NULL, seg, cap) : -1;
        if (ns < 0)
            die("out of memory");
        for (long i = 0; i < ns; i++)
            printf("%.3f %.3f\n", (double)seg[i].start / rate, (double)seg[i].end / rate);
        free(seg);
    }
    pk_asr_free(a);
    free(x);
    return 0;
}

/* the command that fetches f into the working directory */
static void fetch(const ed_part *f) {
#ifdef _WIN32
    printf("    curl.exe -fLO %s%.12s/%s\n", ed_base, f->sha256, f->file);    /* PowerShell's curl is another */
#else
    printf("    curl -fLO %s%.12s/%s\n", ed_base, f->sha256, f->file);
#endif
}

static void runs_on(const ed_edition *e) {
    if (e->kind == ED_COREML)
        printf("macOS %d or iOS %d, or later; a build with ANE=1", e->macos[0], e->ios[0]);
    else if (e->kind == ED_QNN) {
        printf("QNN's chip %s, on ", e->chip);
        for (const char *const *s = e->systems; *s; s++)
            printf("%s%s", s == e->systems ? "" : " or ", *s);
        printf("; a build with QNN=1");
    } else
        printf("every build");
}

/* f in dir, where it is there and not the size models.json gives it: an
 * older file, or another */
static void check_size(const char *dir, const ed_part *f) {
    char p[4608];
    snprintf(p, sizeof p, "%s%s", dir, f->file);
    const long long n = file_size(p);
    if (n >= 0 && (unsigned long long)n != f->bytes)
        printf("  %s is %lld bytes, not models.json's %llu: an older file, or another\n", p, n,
               (unsigned long long)f->bytes);
}

/* pktito-c models: for each model, the edition this machine runs with this
 * build (models.json's fit and choose), the commands that fetch it into
 * models/, and how much of it is there; with --all every edition, what it
 * runs on, and its files' SHA-256s */
static int models(const opts *o, const char *argv0) {
    const ed_device *d = device();
    int ane = 0, qnn = 0;
#ifdef PK_ANE
    ane = 1;
#endif
#ifdef PK_QNN
    qnn = 1;
#endif
    printf("This machine: %s", d->system);
    if (d->os[0])
        printf(" %d.%d", d->os[0], d->os[1]);
    unsigned arch;
    if (d->qnn[0] && sscanf(d->qnn, "%*u,%u", &arch) == 1)
        printf(", its NPU a Hexagon V%u (QNN's %s)", arch, d->qnn);
    else if (d->qnn[0])
        printf(", its NPU %s to QNN", d->qnn);
    printf(". This pktito-c runs the CPU's editions%s%s.\n", ane ? " and Apple's Neural Engine's (ANE=1)" : "",
           qnn ? " and Snapdragon NPUs' (QNN=1)" : "");
    if (qnn && !d->qnn[0])
        printf("QNN finds no NPU here: %s\n", qnn_why);
    /* what another build would run here, as far as this one can tell */
    const ed_edition *coreml = NULL;
    for (int i = 0; i < ed_catalog[0].n_editions; i++)
        if (ed_catalog[0].editions[i].kind == ED_COREML)
            coreml = &ed_catalog[0].editions[i];
    if (coreml && !strcmp(d->system, "macos") && !ed_fit(coreml, d))
        printf("Apple's Neural Engine's editions need macOS %d or later.\n", coreml->macos[0]);
    else if (coreml && !ane && ed_fit(coreml, d))
        printf("A build with ANE=1 (make ANE=1) runs Apple's Neural Engine's editions here instead.\n");
#if defined(__aarch64__) || defined(_M_ARM64)
    if (!qnn && (!strcmp(d->system, "windows") || !strcmp(d->system, "android")))
        printf("On a Snapdragon, a build with QNN=1 asks QNN for the NPU's chip, and runs its edition\n"
               "(README, \"The Neural Engine and NPUs\").\n");
#endif
    for (int i = 0; i < ed_n_models; i++) {
        const ed_model *m = &ed_catalog[i];
        if (o->all) {
            printf("\n%s, %s\n", m->name, m->about);
            for (int k = 0; k < m->n_editions; k++) {
                const ed_edition *e = &m->editions[k];
                printf("  %-13s %s: ", e->id, e->about);
                runs_on(e);
                printf("%s%s\n", e->note ? "; " : "", e->note ? e->note : "");
                for (int f = 0; f < 1 + !!e->encoder.file; f++) {
                    const ed_part *p = f ? &e->encoder : &e->model;
                    printf("    %s%.12s/%s\n      %.1f MB, SHA-256 %s\n", ed_base, p->sha256, p->file, p->bytes / 1e6,
                           p->sha256);
                }
            }
            continue;
        }
        const ed_edition *e = ed_choose(m, d, can_run);
        const uint64_t bytes = e->model.bytes + (e->encoder.file ? e->encoder.bytes : 0);
        printf("\n%s (%s): the %s edition, for %s, %.0f MB\n", m->name, m->about, e->id, e->about, bytes / 1e6);
        fetch(&e->model);
        if (e->encoder.file)
            fetch(&e->encoder);
        if (e->kind == ED_COREML) {
            char name[512];
            encoder_name(e, name, sizeof name);
            printf("    mkdir %s && aa extract -i %s -d %s && rm %s\n", name, e->encoder.file, name, e->encoder.file);
        }
        if (e->kind == ED_CPU)
            printf("  or make models, which converts NVIDIA's checkpoint to the same file (README, \"Try it\")\n");
        char dir[4096];
        int state = 0, at = 0;
        for (int k = 0; k < 2 && state < 2; k++) {
            models_dir(k, argv0, dir, sizeof dir);
            const int s = edition_here(e, dir);
            if (s > state)
                state = s, at = k;
        }
        models_dir(at, argv0, dir, sizeof dir);
        if (state == 2)
            printf("  all here, in %s\n", dir);
        else
            printf(state ? "  some of it here, in %s\n" : "  not here yet\n", dir);
        check_size(dir, &e->model);
        if (e->kind == ED_QNN)
            check_size(dir, &e->encoder);
    }
    if (!o->all)
        printf("\nRun each model's lines in models/, where you run pktito-c or beside it, and\n"
               "pktito-c finds it there. Every edition, and each file's SHA-256: pktito-c models --all\n");
    return 0;
}

static int info(pk_model *m) {
    pk_model_info mi;
    pk_model_describe(m, &mi);
    printf("model     %s, %.1f MB\n", pk_model_name(m), mi.file_bytes / 1e6);
    printf("encoder   %u layers of %u, %u heads, feed-forward %u; %u mel bands at %u Hz\n",
           mi.n_layers, mi.d_model, mi.n_heads, mi.ff_dim, mi.n_mels, mi.sample_rate);
    if (pk_model_speakers(m)) {
        printf("speakers  %d, a row of probabilities every %g ms\n", pk_model_speakers(m),
               pk_model_frame_seconds(m) * 1e3);
    } else {
        printf("decoder   %u-unit LSTM x %u, joint %u; %u tokens and blank; durations",
               mi.pred_hidden, mi.pred_layers, mi.joint_hidden, mi.vocab);
        for (uint32_t i = 0; i < mi.n_durations; i++)
            printf(" %u", mi.durations[i]);
        printf("\n");
    }
    printf("backend   %s, %d threads by default\n", pk_backend(), pk_os_cores());
    char picks[256];                        /* what the pick timed, where a backend competed */
    if (pk_k_picks(picks, sizeof picks))
        printf("probe     %s\n", picks);
    printf("compiler  %s\n", pk_compiler());
    const char *k, *v;
    for (int i = 0; pk_model_meta_at(m, i, &k, &v) == 0; i++) {
        /* a value of many lines, a licence's text, by its first */
        const char *nl = strchr(v, '\n');
        int lines = 1;
        for (const char *c = v; (c = strchr(c, '\n')); c++)
            lines++;
        if (nl)
            printf("  %-16s %.*s (and %d lines more)\n", k, (int)(nl - v), v, lines - 1);
        else
            printf("  %-16s %s\n", k, v);
    }
    return 0;
}

int main(int argc, char **argv) {
    opts o;
    pk_os_stdio();
    /* A program run for its answer, window or none: its thread out of
     * Windows' power throttling for the whole run (core/os.h), where the
     * library takes it out only for the calls that work on it. What the
     * program does on it between them - the audio read and resampled, a
     * stream's VAD a frame at a time, the output - is the run's too. */
    pk_os_unthrottle();
    /* every path the library takes is UTF-8, and on Windows argv is not */
    if (!(argv = utf8_args(&argc, argv)))
        die("cannot read the command line");
    parse(argc, argv, &o);
    if (!strcmp(o.cmd, "models")) {
        const int rc = models(&o, argv[0]);
        free(o.ins);
        return rc;
    }
    /* -m's models, a recognizer's and a diarizer's, told apart by what
     * they are */
    pk_model *m = NULL, *dm = NULL;
    for (int i = 0; i < o.nmodels; i++) {
        pk_model *x = open_model(o.models[i]);
        if (pk_model_speakers(x)) {
            if (dm)
                die("two diarizers' models: -m one of them");
            dm = x;
        } else {
            if (m)
                die("two models, %s and %s: -m one of them", model_path, o.models[i]);
            m = x;
            model_path = o.models[i];
        }
    }
    int rc;
    if (!strcmp(o.cmd, "speakers")) {
        if (!dm)
            no_diarizer("pktito-c speakers");
        rc = speakers(&o, dm);
    } else if (!strcmp(o.cmd, "info")) {
        if (!m && !dm)
            m = load_model(NULL, argv[0]);
        rc = m ? info(m) : 0;
        if (dm) {
            if (m)
                putchar('\n');
            rc |= info(dm);
        }
    } else if (!strcmp(o.cmd, "probs") && dm && !m) {
        rc = diar_probs(&o, dm);
    } else if (!strcmp(o.cmd, "bench") && dm && !m) {
        rc = bench_diar(&o, dm);
    } else {
        /* a diarizer's -m asks for --speakers, as the flag does */
        if (dm && (o.subs || o.stream))
            die("a diarizer's model with %s: not yet", o.subs ? "subtitles" : "--stream");
        if (dm && strcmp(o.cmd, "text"))
            die("a diarizer's model is for a transcript with --speakers, pktito-c speakers or probs; not %s",
                o.cmd);
        if (o.speakers && !dm)
            no_diarizer("--speakers");
        if (!m)
            m = load_model(NULL, argv[0]);
        rc = !strcmp(o.cmd, "bench") ? bench(&o, m)
           : !strcmp(o.cmd, "segments") || !strcmp(o.cmd, "probs") ? vad(&o, m)
           : o.stream ? stream(&o, m) : transcribe(&o, m, dm);
    }
    pk_model_free(m);
    pk_model_free(dm);
    free(o.ins);
    return rc;
}
