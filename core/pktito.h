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
 *   transcribe   pk_asr_run       16 kHz samples in, tokens with times out
 *   text         pk_text          tokens to UTF-8, as NeMo writes them
 *   segments     pk_vad_probs,    Silero's per-frame speech probabilities, and
 *                pk_segments      get_speech_timestamps' segments from them
 *   chunks       pk_chunks        a long recording cut at its silences into
 *                                 pieces the encoder takes whole, its
 *                                 silence left out
 *   captions     pk_captions      words with times to a subtitle track's cues
 *
 * A model is read-only once loaded and may be shared by any number of
 * transcribers on any number of threads; a transcriber belongs to one
 * thread at a time, and runs its own pool of worker threads. Nothing here
 * keeps global state or prints.
 */
#ifndef PKTITO_H
#define PKTITO_H

#include <stddef.h>
#include <stdint.h>

#define PKTITO_VERSION "0.1.0"

/* ---------------------------------------------------------------- model --- */

typedef struct pk_model pk_model;

/* A .pkf file, mapped and checked before anything in it is used. NULL on
 * failure, with the reason in err (which may be NULL): a file that cannot
 * be read or is not a sound model, or a PKTITO_KERNELS in the environment
 * that names a backend the build lacks or the CPU cannot run. */
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

/* The architecture's numbers, for printing (`pktito info`). */
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

/* A transcriber with `threads` threads in all (0: one a core). NULL if
 * memory or a thread cannot be had. With 0, on a chip whose cores are of
 * two kinds, the first recording of 8 s or more times the encoder on every
 * core and on the fast kind alone, and the transcriber keeps the faster;
 * pk_asr_threads says how many it runs on. */
pk_asr *pk_asr_new(const pk_model *m, int threads);
void    pk_asr_free(pk_asr *a);
int     pk_asr_threads(const pk_asr *a);

/* One token of a transcript: a piece of the vocabulary, the encoder frame
 * it was emitted at, and the number of frames TDT says it lasts. */
typedef struct {
    int32_t id, frame, dur;
} pk_tok;

/* Transcribe n samples of 16 kHz mono audio, nominally in [-1, 1], as one
 * utterance: the whole of it through the encoder at once, as NeMo's
 * transcribe() does with a file. *toks points at the tokens, which belong to
 * the transcriber and last until its next call. Returns their count, or -1
 * if memory runs out. A sample that is not finite is read as 0.
 *
 * The encoder's attention costs grow with the square of the length, and
 * its working memory with the length: a minute is about 60 MB. Long
 * recordings are pk_chunks' business. */
long pk_asr_run(pk_asr *a, const float *x, size_t n, const pk_tok **toks);

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

/* ------------------------------------------------------------------ VAD --- */

/* Silero VAD, as vadito runs it: one stream's state. A new or reset
 * stream is Silero's after reset_states(). */
typedef struct pk_vad pk_vad;

pk_vad *pk_vad_new(const pk_model *m);
void    pk_vad_reset(pk_vad *v);
void    pk_vad_free(pk_vad *v);

#define PK_VAD_SAMPLES 512      /* a frame: 32 ms at 16 kHz */

/* P(speech) for the next frame, or for n consecutive frames x[n * 512] ->
 * probs[n] (the same bits, several times quicker). */
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

/* A stretch of a recording, in samples; end is exclusive. */
typedef struct { int64_t start, end; } pk_seg;

/* get_speech_timestamps_from_probs (silero_vad/utils_vad.py, v6.2.3): the
 * speech in n frames' probabilities. Returns how many segments there are;
 * if that is more than cap, out is unfinished. 2n + 1 is always enough. */
size_t pk_segments(const float *probs, size_t n, int rate, int64_t n_samples,
                   const pk_seg_params *p, pk_seg *out, size_t cap);

/* VADIterator: speech starts and ends as they happen. */
typedef struct {
    int     frame;
    double  threshold, neg_threshold;
    double  min_silence_samples, speech_pad_samples;
    int     triggered;
    int64_t temp_end, current_sample;
} pk_iter;

enum { PK_NONE = 0, PK_START = 1, PK_END = 2 };

void pk_iter_init(pk_iter *it, int rate, const pk_seg_params *p);
int  pk_iter_step(pk_iter *it, float prob, int64_t *at);

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
 * core/captions.c's costs, not the first that fits. english 1 adds the
 * English rules (no cue or line ends on "the" or "to"; an abbreviation's
 * period ends no sentence); 0 leaves punctuation and pauses; -1 adds them
 * if pk_language finds the words English, as for the multilingual model. media_end is
 * when the recording ends (ms), or 0 if unknown. out has room for n.
 * Returns the count of cues, or -1 if memory runs out (90 bytes a word,
 * given back before it returns). */
long pk_captions(const pk_capword *w, size_t n, int english, int64_t media_end, pk_cue *out);

/* The language of n words, one of the 25 the multilingual model writes (bg
 * cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk),
 * as its two-letter code: the language their commonest words are commonest
 * in (core/language.c). NULL for words too few, or no language ahead by
 * enough. Only the words' text is read. */
const char *pk_language(const pk_capword *w, size_t n);

#endif /* PKTITO_H */
