/* pk_segments' state machine a frame at a time, for a recording whose end
 * has not come yet (core/transcribe.c): the segments as each ends, and
 * their padding as each one's neighbour is known. Internal. */
#ifndef PK_SEGMENT_H
#define PK_SEGMENT_H

#include "pktito.h"

/* get_speech_timestamps_from_probs' loop, its locals kept between frames */
typedef struct {
    int64_t window;
    double  min_speech, speech_pad, max_speech, min_silence, min_silence_at_max;
    double  threshold, neg;
    int     use_max;

    int64_t frames;                 /* probabilities taken */
    int     triggered, cur;         /* cur: current_speech is non-empty */
    int64_t cur_start, temp_end, prev_end, next_start;
    size_t  n_possible;             /* len(possible_ends) */
    int64_t best_end, best_dur;     /* max(possible_ends, key=dur) */
} pk_segs;

void pk_segs_init(pk_segs *s, int rate, const pk_seg_params *p);

/* The next frame's probability: the segments it ends into out, unpadded,
 * and their count, 0 to 2 (2 only where min_silence_ms is 0) */
int pk_segs_step(pk_segs *s, float prob, pk_seg out[2]);

/* The recording over at audio_len samples: the segment still open, if it
 * is long enough, into *out; 1 or 0 */
int pk_segs_end(pk_segs *s, int64_t audio_len, pk_seg *out);

/* No segment still to come starts before this sample, unpadded */
int64_t pk_segs_floor(const pk_segs *s);

/* pk_segments' padding pass on segments as they come. A segment's end
 * waits for the next one's start, so each is final one segment late: `held`
 * is the last one given, its start final and its end not yet. */
typedef struct {
    double speech_pad;
    int    any, has;                /* a segment has come; one is held */
    pk_seg held;
} pk_pads;

void pk_pads_init(pk_pads *d, int rate, const pk_seg_params *p);

/* The next unpadded segment: the one before it into *out, final; 1, or 0
 * for the first */
int pk_pads_put(pk_pads *d, pk_seg raw, pk_seg *out);

/* The recording over: the held segment into *out, final; 1, or 0 if none */
int pk_pads_end(pk_pads *d, int64_t audio_len, pk_seg *out);

#endif /* PK_SEGMENT_H */
