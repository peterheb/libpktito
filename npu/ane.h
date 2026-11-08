/* The encoder on Apple's Neural Engine through Core ML (make ANE=1; macOS
 * 15, iOS 18 or later): the compiled model tools/ane/encoder.py makes, its
 * functions the encoder in parts ("f3072p0", "f3072p1"), each taking one
 * length (3072 feature frames by default) or several. It stands in for
 * core/encoder.c on an input that fits its longest length. Its arithmetic
 * is fp16, not ref's, and not the same bits on every chip, which is why it
 * is this fork's and not pktito's. Internal. */
#ifndef PK_ANE_H
#define PK_ANE_H

#include <stddef.h>
#include <stdint.h>

#include "pktito.h"

typedef struct pk_ane pk_ane;

/* path's functions, listed, for model m: its features, its width, its
 * position encoding's frequencies, and the checkpoint both were made from
 * (their metadata's source_sha256, which must be the same); NULL if they
 * can't be, and why in err. A function
 * is loaded for the Neural Engine when an input first needs it: 0.1 to 0.5
 * s once Core ML has compiled it for this device, which it keeps; seconds
 * the first time, for each of its lengths (on the M4 Pro 4.9 s a part of
 * one length, 20 s of six). */
pk_ane *pk_ane_open(const char *path, const pk_model *m, char *err, size_t errlen);
void pk_ane_free(pk_ane *e);

/* Every length loaded and run once, on silence, now rather than when first
 * needed: the compiling done at a moment the caller chooses, and a device
 * whose Neural Engine can't run the encoder known before any recording. 0,
 * or -1 if one can't be, and Core ML's why in err. */
int pk_ane_warm(pk_ane *e, char *err, size_t errlen);

/* Calls that can run at once: PK_ANE_SLOTS - 1 for the chunks of a run
 * (asr.c's pipeline, slots 0 on), and the last for every other call
 * (pk_ane_encode: the retry's, the warming's). */
#define PK_ANE_SLOTS 5

/* The feature frames its longest function takes */
uint32_t pk_ane_frames(const pk_ane *e);

/* feat [T0][n_mels] through the shortest length that takes T0 frames,
 * padded to it, or the length run last where that one is less than 1536
 * frames longer (a change of length costs the Neural Engine more than its
 * frames would): x [Te][d_model], Te encoder frames, which it returns; 0 if
 * no length takes T0 or Core ML failed. */
uint32_t pk_ane_encode(pk_ane *e, const float *feat, uint32_t T0, float *x);

/* pk_ane_encode in slot (0 to PK_ANE_SLOTS - 1) of its own: calls in other
 * slots may run at the same time, from other threads */
uint32_t pk_ane_encode_in(pk_ane *e, int slot, const float *feat, uint32_t T0, float *x);

#endif
