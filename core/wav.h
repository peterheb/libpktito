/* A small WAV reader and writer, in C11 and stdio, with core/os.c to open
 * a UTF-8 path on Windows.
 *
 * It reads what recordings come in: PCM at 8 (unsigned), 16, 24 and 32
 * bits, IEEE float at 32 and 64, G.711 A-law and mu-law (the 8 kHz
 * telephony formats), any of those wrapped in
 * WAVE_FORMAT_EXTENSIBLE, RF64 for files past 4 GB, and the "streamed" WAV
 * ffmpeg writes to a pipe, whose sizes it could not go back and fill in. It
 * reads front to back and never seeks, so stdin works as well as a file,
 * and it keeps one block of bytes, so a file of any length reads in the
 * same memory.
 *
 * Samples come out as float in [-1, 1) by the conventions torchaudio.load
 * and soundfile use (int16 / 32768, uint8 (x - 128) / 128, ...), and with
 * more than one channel as their mean, which is what NeMo's audio loader
 * does with them. It is vadito's reader (core/wav.c there), renamed.
 *
 * The file is untrusted input and the parser keeps docs/MEMORY_SAFETY.md:
 * every chunk size is checked before it is used, and the reader allocates
 * nothing whose size comes from the file.
 */
#ifndef PK_WAV_H
#define PK_WAV_H

#include <stddef.h>
#include <stdint.h>

enum { PK_WAV_PCM = 1, PK_WAV_FLOAT = 3, PK_WAV_ALAW = 6, PK_WAV_MULAW = 7 };

#define PK_WAV_UNKNOWN UINT64_MAX     /* frames, for a streamed file */

typedef struct {
    uint32_t rate;
    uint32_t channels;
    uint32_t format;          /* PK_WAV_*, EXTENSIBLE resolved to what it wraps */
    uint32_t bytes;           /* per sample, as stored: 1, 2, 3, 4 or 8 */
    uint64_t frames;          /* in the data chunk, or PK_WAV_UNKNOWN */
} pk_wav_info;

typedef struct pk_wav pk_wav;

/* Open path ("-" is stdin) and read up to the start of its samples. NULL on
 * failure, with the reason in err. A path here and in pk_wav_create is
 * UTF-8 on every system, Windows too, whatever its code page (core/os.h's
 * pk_os_fopen opens it). */
pk_wav *pk_wav_open(const char *path, pk_wav_info *info, char *err, size_t errlen);

/* Up to n more frames, each mixed to mono, into out[n]. Returns how many;
 * fewer than n only at the end of the data, or on a read error. */
size_t pk_wav_read(pk_wav *w, float *out, size_t n);

/* The same, with each frame's channels as they are, interleaved, into
 * out[n * channels]: for a caller that mixes them itself. */
size_t pk_wav_read_frames(pk_wav *w, float *out, size_t n);

/* 0 if the data ended where the header said it would (or a streamed file
 * reached its end), else -1 with what happened instead in err: cut short,
 * or a read error. Asked after the last pk_wav_read. */
int pk_wav_status(const pk_wav *w, char *err, size_t errlen);

void pk_wav_close(pk_wav *w);

/* A mono WAV to write, 16-bit PCM or (float32) 32-bit float; "-" is stdout,
 * where the sizes cannot be filled in afterwards and are left as a stream's. */
typedef struct pk_wav_out pk_wav_out;

pk_wav_out *pk_wav_create(const char *path, uint32_t rate, int float32, char *err, size_t errlen);
int         pk_wav_write(pk_wav_out *o, const float *x, size_t n);    /* 0, or -1 */
int         pk_wav_finish(pk_wav_out *o, char *err, size_t errlen);   /* sizes, close: 0 or -1 */

#endif /* PK_WAV_H */
