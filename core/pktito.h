/* pktito: NVIDIA's Parakeet TDT speech recognizers in C11, with nothing
 * beyond the C library (and pthreads, where there are threads to use).
 *
 * Parakeet TDT 0.6B (huggingface.co/nvidia/parakeet-tdt-0.6b-v2, English,
 * and -v3, 25 European languages) turns 16 kHz audio into punctuated,
 * cased text: a FastConformer encoder hears the audio at 12.5 frames a
 * second, and a Token-and-Duration Transducer reads the text off it, a
 * token and how many frames it lasts at a time. This is that model without
 * NeMo or PyTorch, and Silero's VAD beside it to cut long recordings at
 * their silences:
 *
 *   a model      pk_model_load    a .pkf file: the model, its vocabulary and
 *                                 the VAD, with where every number came from
 *   a recording  pk_asr_transcribe  the high-level path: any length in,
 *                                 each chunk's tokens out, VAD, chunking
 *                                 and the retries done; pk_steps_new, the
 *                                 same in steps for a caller who runs the
 *                                 transcribing itself
 *   transcribe   pk_asr_run       the low-level path: 16 kHz samples in,
 *                                 tokens with times out, NeMo's to the bit
 *   text         pk_text,         tokens to UTF-8, as NeMo writes them, and
 *                pk_words         to words with times
 *   segments     pk_asr_speech    Silero's speech segments of a recording;
 *                pk_vad_probs,    its per-frame speech probabilities, and
 *                pk_segments      get_speech_timestamps' segments from them
 *   chunks       pk_chunks        a long recording cut at its silences into
 *                                 pieces the encoder takes whole, its
 *                                 silence left out; pk_asr_run_chunks
 *                                 transcribes them, each chunk's decoder
 *                                 beside the next one's encoder
 *   captions     pk_captions      words with times to a subtitle track's cues
 *   language     pk_language      which of the 25 languages words are in
 *   sentences    pk_sentences     tokens to the sentences a transcript reads in
 *
 * A model is read-only once loaded and may be shared by any number of
 * transcribers on any number of threads; a transcriber belongs to one
 * thread at a time, and runs its own pool of worker threads. Nothing here
 * keeps global state or prints.
 *
 * Throughout: a count is a size_t where nothing can fail and a long where
 * something can, -1 then; a status is an int, -1 a failure; a constructor
 * fails with NULL; each function says what its failures are, most often
 * memory run out. Every *_free takes NULL, and an options pointer may be
 * NULL for the defaults. A time is in its own unit: a token's or a word's
 * in encoder frames (pk_model_frame_seconds), a stretch of audio's in
 * samples, a caption's in milliseconds. A language is a two-letter code, as
 * pk_language names it. core/wav.h reads WAV and AIFF files, and
 * core/resample.h brings another rate to the model's.
 */
#ifndef PKTITO_H
#define PKTITO_H

#include <stddef.h>
#include <stdint.h>

#define PKTITO_VERSION "0.9.6"

/* ---------------------------------------------------------------- model --- */

typedef struct pk_model pk_model;

/* A .pkf file, mapped and checked before anything in it is used. NULL on
 * failure, with the reason in err (which may be NULL): a file that cannot
 * be read or is not a sound model, or a PKTITO_KERNELS in the environment
 * that names a backend the build lacks or the CPU cannot run. The path is
 * UTF-8 on every system: on Windows it is opened as UTF-16, so a model in
 * a user folder whose name is past ASCII loads whatever the process's
 * code page, and bytes that are not UTF-8 are refused, never read in the
 * code page instead. The checks run on the calling thread, readied for the
 * call as for a pk_asr_run. */
pk_model *pk_model_load(const char *path, char *err, size_t errlen);

/* The same from a copy the caller holds - a file embedded in a binary, or
 * read some other way. The bytes are copied, so they need not outlive the
 * call, and the copy is a second one in memory where pk_model_load's
 * mapping is the page cache's only one: for a file, load it. */
pk_model *pk_model_from_bytes(const void *bytes, size_t n, char *err, size_t errlen);
void      pk_model_free(pk_model *m);

/* The model's name ("parakeet-tdt-0.6b-v2"), and a value from the file's
 * metadata table (license, source, recipe, ...) or NULL. */
const char *pk_model_name(const pk_model *m);
const char *pk_model_meta(const pk_model *m, const char *key);
/* The i-th metadata pair, for listing: 0, or -1 past the last. */
int pk_model_meta_at(const pk_model *m, int i, const char **key, const char **value);

/* 0 for a slim model file (`pktito-tools slim`; a build with ANE=1 or
 * QNN=1 loads one), which leaves its encoder out for an accelerator's: it
 * transcribes only with the encoder pk_asr_use_ane or pk_asr_use_qnn gives
 * it, and without one pk_asr_run returns -1. 1 for every other. */
int pk_model_has_encoder(const pk_model *m);

/* The architecture's numbers, for printing (`pktito-c info`). */
typedef struct {
    size_t   file_bytes;
    uint32_t sample_rate, n_mels, n_layers, d_model, n_heads, ff_dim;
    uint32_t vocab, pred_hidden, pred_layers, joint_hidden, n_durations;
    uint32_t durations[16];
} pk_model_info;

void pk_model_describe(const pk_model *m, pk_model_info *out);

/* Samples a second the model hears (16000), and seconds an encoder frame
 * lasts (0.08: a token's frame times this is when it was heard). */
int    pk_model_rate(const pk_model *m);
double pk_model_frame_seconds(const pk_model *m);

/* -------------------------------------------------------- transcription --- */

typedef struct pk_asr pk_asr;

/* A transcriber on `threads` threads (0: one a core). NULL if memory or a
 * thread cannot be had. With 0, on a chip whose cores are of two kinds, the
 * second recording of 8 s or more times the encoder on every core and on
 * the fast kind alone, and the transcriber keeps the faster;
 * pk_asr_threads says how many it runs on. On 8 threads or more it may run
 * a recording's chunks several at once (pk_asr_run_chunks), and makes the
 * threads and the memory for that the first time a recording takes them. */
pk_asr *pk_asr_new(const pk_model *m, int threads);
void    pk_asr_free(pk_asr *a);
int     pk_asr_threads(const pk_asr *a);

/* The fast mode, off until asked for: the encoder's activations in blocks
 * of 64 inputs, not 32, wherever an input is whole blocks of 64 (every one
 * of Parakeet's). For a caller who wants the speed more than the
 * transcript nearest float32's: WER is level, a few more transcripts differ
 * from float32's (49 of FLEURS' 360 against 45), and on the boards
 * measured a run is 3.9 to 7.3% faster where the int8 product is a dot
 * product (SDOT, VPDPBUSD), 5.5% on a Pi 4 and 3.2 to 3.4% on a Pi 3 with
 * the ARMv8.0 loop, and 1 to 3% with x86's VPMADDUBSW (README, "Why int8";
 * backend/avx2/CLIMB.md and backend/neon/CLIMB.md, 2026-10-29). AMX's
 * pipeline takes blocks of 32 alone, and in the mode the amx-int8 build
 * runs its zmm tile, unmeasured. From the next call on; every backend
 * computes the same bits in either mode. */
void    pk_asr_set_fast_mode(pk_asr *a, int on);

/* The encoder in path on the accelerator it was made for, by its suffix: a
 * compiled Core ML model (a .mlmodelc folder, as models.json's
 * .mlmodelc.aar unpacks) on Apple's Neural Engine, in a build with ANE=1,
 * or a QNN graph (.qnn) on a Snapdragon's NPU, in a build with QNN=1
 * (pk_asr_use_ane and pk_asr_use_qnn below). A slim model, which has no
 * encoder of its own (pk_model_has_encoder 0), runs only with one. Its
 * arithmetic is fp16, not the int8 every backend computes the same bits
 * of. 0, or -1 with why in err: path is neither, or is not there, or this
 * build has no such accelerator, or the accelerator refuses it. */
int     pk_asr_use_encoder(pk_asr *a, const char *path, char *err, size_t errlen);

/* The encoder on Apple's Neural Engine (a build with ANE=1; macOS 15,
 * iOS 18): the compiled Core ML model tools/ane/encoder.py makes, at path,
 * or with NULL the CPU's again. It must have been made from the
 * model's own checkpoint (both name its SHA-256). An input longer than its
 * longest function still takes the CPU's encoder, where the model has one
 * (pk_model_has_encoder). Its arithmetic is fp16, not the int8 every
 * backend computes the same bits of: a transcript can differ from
 * pktito's, and from one chip's to another's. 0, or -1 with why in err. */
int     pk_asr_use_ane(pk_asr *a, const char *path, char *err, size_t errlen);

/* The Neural Engine's encoder loaded and run once, on silence, now rather
 * than at the first input that needs it: Core ML compiles it for this
 * device the first time (16 s on an M1 iPad), and keeps that for the build
 * of the app. 0, or -1 if there is none or this device can't run it, with
 * Core ML's why in err: where a slim model's can't, the caller's way on is
 * the full model file, whose encoder is pktito's own. */
int     pk_asr_ane_warm(pk_asr *a, char *err, size_t errlen);

/* The encoder on Qualcomm's Hexagon NPU (a build with QNN=1; Android or
 * Windows on a Snapdragon), on: a graph built from the model's own weights
 * (npu/qnn.h), which QNN compiles now, in a minute or two (94 to 114 s
 * on a Galaxy S24 Ultra, tools/qnn/CLIMB.md); with cache a path, the
 * compiled graph is kept there and read back the next time, in a second
 * or two.
 * Off (on 0) the CPU's again. An input longer than its 30.72 s still
 * takes the CPU's encoder. A slim model's (pk_model_has_encoder 0) is
 * the graph in cache alone, compiled ahead of time for this chip (on a
 * PC, PKTITO_QNN_TARGET: tools/qnn/CLIMB.md), and with no CPU encoder an
 * input longer than the graph's is not transcribed. Where QNN refuses the
 * graph there - made for another chip, or by a QNN this one does not
 * read - err says so and the QNN that made it, for the caller to fetch
 * the graph again. Its arithmetic is fp16 against the model's int8
 * weights, not the int8 every backend computes the same bits of: a
 * transcript can differ from pktito's, and from one chip's to another's.
 * 0, or -1 with why in err. */
int     pk_asr_use_qnn(pk_asr *a, int on, const char *cache, char *err, size_t errlen);

/* The NPU here as QNN describes it to a PC that compiles a graph for it
 * (PKTITO_QNN_TARGET): "57,75,8,1,0", the SoC's number in QNN's table,
 * its Hexagon's version, its VTCM in MB, and whether it has DLBC and a
 * signed PD, into chip, cap bytes; for the caller to fetch the graph
 * compiled for this chip, which pk_asr_use_qnn reads. 0, or -1 where QNN
 * finds none (a PC's QNN has no NPU to ask) or the build has no QNN, with
 * why in err. */
int     pk_qnn_chip(char *chip, size_t cap, char *err, size_t errlen);

/* One token of a transcript: a piece of the vocabulary, the encoder frame
 * it was emitted at, and the number of frames TDT says it lasts. */
typedef struct {
    int32_t id, frame, dur;
} pk_tok;

/* Transcribe n samples of 16 kHz mono audio, nominally in [-1, 1], as one
 * utterance: the whole of it through the encoder at once, as NeMo's
 * transcribe() does with a file. *toks points at the tokens, which belong to
 * the transcriber and last until its next call. Returns their count, or -1
 * if memory runs out or the encoder can't run (a slim model's with no
 * accelerator's encoder given, the accelerator failing it, or an input
 * longer than it takes). A sample that is not finite is read as 0.
 *
 * The encoder's attention costs grow with the square of the length, and
 * its working memory with the length: a minute is about 60 MB. Long
 * recordings are pk_chunks' business.
 *
 * The calling thread does a share of the work. For the call it is out of
 * the power throttling Windows gives a process without a window, and runs
 * SVE at 128 bits where Linux gives it more, as the transcriber's own
 * threads are for good; it leaves as it came (core/os.h). */
long pk_asr_run(pk_asr *a, const float *x, size_t n, const pk_tok **toks);

/* A stretch of a recording, in samples; end is exclusive. */
typedef struct { int64_t start, end; } pk_seg;

/* A chunk of a recording, transcribed (pk_asr_run_chunks,
 * pk_asr_transcribe, pk_steps_take). */
typedef struct {
    size_t        i, n;         /* this chunk, of how many */
    pk_seg        at;           /* its samples in the recording */
    const pk_tok *toks;         /* its tokens, frames from at.start, retries' included */
    long          count;
    int           retried;      /* stretches of it transcribed again */
    int64_t       done, total;  /* samples of chunks done, of all of them: progress */
    int32_t       at_frame;     /* at.start in encoder frames, rounded: a token's frame
                                   plus this counts from the recording's start */
} pk_chunk_out;

/* Has each chunk of a run in order, on the calling thread, its tokens
 * lasting until it returns; nonzero stops the run there. */
typedef int (*pk_chunk_fn)(void *arg, const pk_chunk_out *c);

/* pk_asr_run over a recording's chunks c[0..n) of x (pk_chunks): fn has
 * chunk i's tokens as pk_asr_run would return them, the same bits, and
 * retried 0. Each chunk's decoder runs beside the next chunk's encoder, on a
 * thread of its own, so that only the last one's is waited for, and fn has
 * a chunk once the next one's encoder is done: 1.2 to 5.6% less time than
 * through pk_asr_run a chunk at a time where there are more than four
 * cores, and about the same on four (core/asr.c).
 *
 * On 8 threads or more a run may take several chunks at once, a lane for
 * each four threads at most, each lane on its share of them with working
 * memory of its own, because one chunk keeps few of many threads busy. It
 * takes as many as its chunks' lengths say end it soonest, and one where
 * none would; fn has the chunks in order as before. AliceBench's chapters
 * ran 1.22 times as fast this way on a 9950X's 16 threads and 2.03 times on
 * an M5 Ultra's 36, and a Pi's four threads keep one lane (README, "How
 * fast"). A transcriber with the Neural Engine (pk_asr_use_ane) keeps one
 * lane, its chunks on the Neural Engine two at a time (core/asr.c,
 * ane_chunks). The working memory is the longest chunk's, and a second
 * chunk's joint half, 1 MB at 30 s, for each lane: about 45 MB a lane at
 * 30.72 s. Returns 0, 1 when fn stopped it, or -1 if memory runs out or the
 * encoder can't run, as pk_asr_run's (fn has then seen the chunks before). */
int pk_asr_run_chunks(pk_asr *a, const float *x, const pk_seg *c, size_t n, pk_chunk_fn fn,
                      void *arg);

/* A recording of any length, x[0, n), to its transcript, with all the
 * processing pktito does for one: the high-level path, where pk_asr_run is
 * the low-level one, transcript = f(x), NeMo's to the bit. In order:
 * Silero's VAD, its encoder over the transcriber's threads, and its speech
 * segments (pk_vad_probs, pk_segments); the chunks of at most max_chunk_s
 * (pk_chunks: the whole recording if it fits, else its speech cut at the
 * last pause that fits, and what lies between), a cut in mid-sentence heard
 * with the speech past it (context); each chunk transcribed, its search
 * beside the next chunk's encoder, and several chunks at once where there
 * are threads for it (pk_asr_run_chunks); and, with retry, each stretch of
 * a chunk that Silero heard speech in but the model wrote no word for,
 * transcribed again on its own - where the model drops speech, as NeMo's
 * own float32 transcription does, a whole chunk of it at times
 * (core/transcribe.c; README, "Transcribing a recording", has why and the
 * measurements). Without the context and the retry each chunk is
 * pk_asr_run's transcript of it.
 *
 * fn has each chunk as soon as it is done, and o NULL is
 * pk_transcribe_defaults'. Returns 0 when every chunk is done, 1 when fn
 * stopped it, and -1 if memory runs out, the encoder can't run (pk_asr_run)
 * or max_chunk_s is under a second. Besides the transcriber's working
 * memory, which is the longest chunk's for each chunk run at once, and its
 * VAD's, which it keeps (0.85 MB a thread), the call holds about 2% of the
 * recording's own size. */
typedef struct {
    double max_chunk_s;         /* 30.72: the most the encoder hears at once (README, "Long recordings") */
    int    retry;               /* 1: speech the model wrote nothing for heard again */
    int    context;             /* 1: a cut in mid-sentence heard with the speech past it */
} pk_transcribe_opts;

void pk_transcribe_defaults(pk_transcribe_opts *o);

int pk_asr_transcribe(pk_asr *a, const float *x, size_t n, const pk_transcribe_opts *o,
                      pk_chunk_fn fn, void *arg);

/* pk_asr_transcribe in steps, for a caller who runs the transcribing
 * itself: on threads of its own, over several transcribers, with several
 * recordings' chunks dealt among them, as a server's queue would.
 * pk_steps_new runs the VAD and cuts the chunks, on a's threads, and the
 * rest is jobs, each a stretch of x to hear: a chunk whole, or a stretch
 * of one that the retry hears again. The caller hears a job with
 * pk_asr_run on any transcriber of the same model and mode, on any thread,
 * and hands its tokens back with pk_steps_put, the jobs in any order;
 * pk_steps_take has each chunk in order once its job and its retries are
 * in. They are the bits pk_asr_transcribe's fn has: pk_asr_run's are the
 * same on every transcriber, and pk_asr_transcribe is these steps, its
 * chunks heard as pk_asr_run_chunks hears them.
 *
 * pk_steps_job hands out the next job, a stretch the retry hears again
 * before the next chunk, since a chunk waits for its retries. It returns
 * 1, 0 when there is none to hand out now, or -1 if memory runs out. With
 * every chunk handed out, the retry's stretches come as the chunks are
 * taken: a chunk's once its tokens are in and the chunk before is taken,
 * and one at a time, since each is heard only while the recording's
 * budget for them lasts, which the last one's tokens decide
 * (core/transcribe.c, RETRY_SHARE). pk_steps_put returns 0, or -1 if
 * memory runs out, count is -1 (pk_asr_run failed), or j is not a job of
 * s's that is out, as pk_steps_job handed it out. pk_steps_take returns 1
 * with the next chunk in *out, its tokens lasting until the next call on
 * s; 0 while that chunk is not done; -1 if memory runs out. The run is
 * over after pk_steps_chunks(s) of them, or at any call's -1, after which
 * every call on s returns -1 and s is for pk_steps_free alone. x must last
 * as long as s, which belongs to one thread at a time, and holds what a
 * pk_asr_transcribe call holds (above) and the tokens of chunks put and not
 * yet taken. pk_steps_new takes o NULL as pk_asr_transcribe does, and
 * returns NULL if memory runs out or max_chunk_s is under a second. */
typedef struct pk_steps pk_steps;

typedef struct {
    size_t chunk;               /* the chunk it is for */
    int    retry;               /* 0: the chunk whole; 1: a stretch of it heard again */
    pk_seg at;                  /* the samples to hear, x[at.start, at.end) */
} pk_job;

pk_steps *pk_steps_new(pk_asr *a, const float *x, size_t n, const pk_transcribe_opts *o);
size_t    pk_steps_chunks(const pk_steps *s);
int       pk_steps_job(pk_steps *s, pk_job *j);
int       pk_steps_put(pk_steps *s, const pk_job *j, const pk_tok *toks, long count);
int       pk_steps_take(pk_steps *s, pk_chunk_out *out);
void      pk_steps_free(pk_steps *s);

/* The text of n tokens, UTF-8 and NUL-terminated in out[cap]: SentencePiece's
 * decoding with NeMo's one change to it, the space before punctuation taken
 * away. Returns the text's length; if that is cap or more, out holds as much
 * as fits, cut at a character's boundary. */
size_t pk_text(const pk_model *m, const pk_tok *t, size_t n, char *out, size_t cap);

/* One piece of the vocabulary as UTF-8 and its length, with SentencePiece's
 * U+2581 still standing for a space: for printing a token on its own. */
const char *pk_piece(const pk_model *m, int32_t id, size_t *len);

/* The words of n tokens, as NeMo groups them for its word timestamps: a
 * word starts at a token whose piece starts with SentencePiece's space
 * (and at the first token), and the tokens after it that do not belong to
 * it, punctuation included. A word spans encoder frames [start, end): its
 * first token's frame to its last token's frame plus that token's duration.
 * Returns how many words there are; out has room for n. */
typedef struct {
    int32_t tok, ntok;          /* its tokens: t[tok] .. t[tok + ntok - 1] */
    int32_t start, end;         /* in encoder frames */
} pk_word;

size_t pk_words(const pk_model *m, const pk_tok *t, size_t n, pk_word *out);

/* The backend this library was built with, e.g. "neon-dp (ARMv8.2: SDOT)". */
const char *pk_backend(void);

/* The compiler that built it, e.g. "gcc 15.2.0" or "Apple clang 21.0.0":
 * a transcript is the same from any, its speed not (README, "How fast"). */
const char *pk_compiler(void);

/* The library's version, PKTITO_VERSION as it was built: for a caller that
 * loads it as a shared library, whose header may be another's. */
const char *pk_version(void);

/* ------------------------------------------------------------------ VAD --- */

/* Silero VAD, as vadito runs it: one stream's state. A new or reset
 * stream is Silero's after reset_states(). */
typedef struct pk_vad pk_vad;

pk_vad *pk_vad_new(const pk_model *m);
void    pk_vad_reset(pk_vad *v);
void    pk_vad_free(pk_vad *v);

#define PK_VAD_SAMPLES 512      /* a frame: 32 ms at 16 kHz */

/* P(speech) for the next frame, or for n consecutive frames x[n * 512] ->
 * probs[n] (the same bits, several times quicker; the calling thread
 * readied for the call as pk_asr_run's is). */
float pk_vad_prob(pk_vad *v, const float *frame);
void  pk_vad_probs(pk_vad *v, const float *x, size_t n, float *probs);

/* get_speech_timestamps' parameters (vadito's core/vadito.h has the story of each);
 * pk_seg_defaults sets Silero's defaults. */
typedef struct {
    double threshold;                      /* 0.5 */
    double neg_threshold;                  /* < 0: max(threshold - 0.15, 0.01) */
    double min_speech_ms;                  /* 250 */
    double max_speech_s;                   /* INFINITY */
    double min_silence_ms;                 /* 100 */
    double speech_pad_ms;                  /* 30 */
    double min_silence_at_max_speech_ms;   /* 98 */
    int    use_max_poss_sil_at_max_speech; /* 1 */
} pk_seg_params;

void pk_seg_defaults(pk_seg_params *p);

/* get_speech_timestamps_from_probs (silero_vad/utils_vad.py, v6.2.3): the
 * speech in n frames' probabilities, with p, or Silero's defaults where p is
 * NULL. Returns how many segments there are; if that is more than cap, out
 * is unfinished. 2n + 1 is always enough. */
size_t pk_segments(const float *probs, size_t n, int rate, int64_t n_samples,
                   const pk_seg_params *p, pk_seg *out, size_t cap);

/* Silero over a recording of any length, x[0, n), as pk_asr_transcribe
 * runs it: the transcriber's own stream made new, its encoder over the
 * transcriber's threads and its LSTM on the calling thread, and the last
 * frame padded with silence. P(speech) for each 32 ms frame into probs,
 * (n + 511) / 512 of them, the bits pk_vad_probs gives from a new stream,
 * 1.9 times as fast as one thread on a 9950X's 16 and a Pi 5's 4, 2.9 on
 * an M5 Ultra's 36 (backend/avx2/CLIMB.md, 2026-11-02). Returns their
 * count, or -1 if memory runs out. */
long pk_asr_vad(pk_asr *a, const float *x, size_t n, float *probs);

/* The speech in x[0, n): pk_asr_vad's probabilities through pk_segments
 * with p, Silero's defaults where p is NULL. Returns how many segments
 * there are, out unfinished if that is more than cap (twice the frames
 * plus one is always enough), or -1 if memory runs out. */
long pk_asr_speech(pk_asr *a, const float *x, size_t n, const pk_seg_params *p, pk_seg *out,
                   size_t cap);

/* Silero's VADIterator: speech starts and ends as they happen, from each
 * frame's probability in turn. pk_vad_iter_init starts one with p, or
 * Silero's defaults where p is NULL. pk_vad_iter_step takes the next
 * frame's probability and returns PK_VAD_START or PK_VAD_END, with the
 * sample the speech started or ended at in *at, or PK_VAD_NONE. */
typedef struct {
    int     frame;
    double  threshold, neg_threshold;
    double  min_silence_samples, speech_pad_samples;
    int     triggered;
    int64_t temp_end, current_sample;
} pk_vad_iter;

enum { PK_VAD_NONE = 0, PK_VAD_START = 1, PK_VAD_END = 2 };

void pk_vad_iter_init(pk_vad_iter *it, int rate, const pk_seg_params *p);
int  pk_vad_iter_step(pk_vad_iter *it, float prob, int64_t *at);

/* --------------------------------------------------------------- chunks --- */

/* A recording, x of `total` samples, cut into the chunks to transcribe, of
 * at most max_len samples each, from its speech segments (pk_segments'
 * output, sorted, none longer than max_len: pk_seg_params.max_speech_s sees
 * to that). A cut between two segments falls in the middle of the silence
 * between them, at most a second into it, so a speech chunk is whole speech
 * with up to a second of silence on either side. A recording that fits in
 * max_len and has any speech is one chunk, the whole of it, which is how
 * NeMo hears a file.
 *
 * What lies between the speech chunks is chunks too, since Silero hears
 * voices, not words: singing and speech over music are not speech to it. A
 * gap is cut into the fewest pieces of at most max_len, all of a length,
 * and a piece is left out if it is silent (no 32 ms of it within 50 dB of
 * full scale) or shorter than 3 s, where the recognizer makes words of
 * anything. Returns the count; out has room for 2 n_speech + 2 + total /
 * max_len. */
size_t pk_chunks(const float *x, const pk_seg *speech, size_t n_speech, int64_t total,
                 int64_t max_len, pk_seg *out);

/* ------------------------------------------------------------- captions --- */

/* A word to caption: its text, UTF-8 with the punctuation it carries (as
 * pk_text writes a pk_word's tokens), and when it was said, in
 * milliseconds. */
typedef struct {
    const char *text;
    int64_t     start, end;
} pk_capword;

/* A cue of a subtitle track: words [first, first + n), shown from start to
 * end (milliseconds), on one line, or on two with the second from word
 * line2 on (0: one line). chars is what a reader reads, graphemes and the
 * spaces between words; flags say where the track could not be all it
 * should. */
typedef struct {
    int32_t first, n, line2, chars, flags;
    int64_t start, end;
} pk_cue;

enum {
    PK_CUE_FAST   = 1,  /* over 20 characters a second to read: the words come faster than that */
    PK_CUE_SHORT  = 2,  /* on for less than a second: the next words or the recording's end come first */
    PK_CUE_FORCED = 4,  /* a word, or words said over each other, too wide or too long for any cue */
    PK_CUE_PAUSE  = 8,  /* cut at a pause in mid-sentence: shown with an ellipsis, U+2026, after its
                           last word, which chars and its last line count */
};

/* The captions of n words in order: lines of at most 42 characters, one or
 * two to a cue, each cue on from its first word until a little after its
 * last, and the words cut into cues where their pauses, punctuation and
 * timing read best - the cheapest of every possible cut by
 * core/captions.c's costs, not the first that fits. lang is the words'
 * language as pk_language names it: "en" adds the English rules (no cue or
 * line ends on "the" or "to"; an abbreviation's period ends no sentence),
 * and another, or NULL, leaves punctuation and pauses. media_end is when
 * the recording ends (ms), or 0 if unknown. out has room for n. Returns the
 * count of cues, or -1 if memory runs out (90 bytes a word, given back
 * before it returns). */
long pk_captions(const pk_capword *w, size_t n, const char *lang, int64_t media_end, pk_cue *out);

/* The language of n words, one of the 25 the multilingual model writes (bg
 * cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk),
 * as its two-letter code: the language their commonest words are commonest
 * in (core/language.c). NULL for words too few, or no language ahead by
 * enough. Only the words' text is read. */
const char *pk_language(const pk_capword *w, size_t n);

/* ------------------------------------------------------------ sentences --- */

/* A sentence: tokens [first, first + n) of those pk_sentences was given. */
typedef struct {
    int32_t first, n, flags;
} pk_sentence;

enum {
    PK_SENTENCE_LONG = 1,   /* broken because it ran past 30 s, at a comma or a pause, not at its end */
};

/* The sentences of n tokens in order, their frames counted from one start:
 * a recording's chunks laid end to end, each chunk's frames plus its
 * at_frame (pk_chunk_out). A sentence ends at a word
 * ending in . ? ! or an ellipsis, unless the next word starts with a small
 * letter, or the word is shaped like an abbreviation ("Dr.", "F.", "U.S.")
 * and its times say no sentence ended there; and one that runs past 30 s is
 * broken after a comma, semicolon, colon or dash that leaves a first part of
 * 10 to 20 s, else at the longest pause there (core/sentences.c; README,
 * "Sentences", has the measurements). lang is the words' language as
 * pk_language names it, for the rules of one language's ("en", "de"), or
 * NULL.
 *
 * done is 1 when the tokens run to the recording's end, so that the words
 * after the last sentence's end make one too, and 0 when more may come:
 * then those words are no sentence yet, and a caller hearing a recording
 * chunk by chunk passes them again with the next chunk's tokens. A
 * sentence's end depends on its own words and the word after it alone, so
 * the sentences are the same however the tokens come.
 *
 * Returns the count; out has room for n. -1 if memory runs out (five times
 * the first power of two past the longest word's text, 320 bytes at least,
 * given back before it returns). */
long pk_sentences(const pk_model *m, const pk_tok *t, size_t n, const char *lang, int done,
                  pk_sentence *out);

#endif /* PKTITO_H */
