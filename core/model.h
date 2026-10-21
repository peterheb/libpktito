/* What core/model.c binds out of a .pkf and the rest of core/ runs.
 * Internal: the public interface is pktito.h. */
#ifndef PK_MODEL_H
#define PK_MODEL_H

#include "pktito.h"
#include "kernels.h"

#include <stddef.h>
#include <stdint.h>

/* ----------------------------------------------------------- the VAD --- */

/* Silero VAD at 16 kHz (core/vad.c): the shapes are its architecture's */
#define PK_VAD_NFFT    256
#define PK_VAD_BINS    129        /* n_fft / 2 + 1 */
#define PK_VAD_HOP     128
#define PK_VAD_FRAME   512        /* samples a frame: 32 ms */
#define PK_VAD_CONTEXT 64         /* samples of the last frame read again */
#define PK_VAD_WIN     4          /* STFT windows a frame */
#define PK_VAD_C1      128
#define PK_VAD_C2      64
#define PK_VAD_C3      64
#define PK_VAD_C4      128
#define PK_VAD_HIDDEN  128
#define PK_VAD_GATES   512

typedef struct {
    const float *window, *fcos, *fsin;
    const float *w1, *b1, *w2, *b2, *w3, *b3, *w4, *b4;
    const float *wx, *wh, *bg;     /* LSTM: input and hidden weights, folded bias */
    const float *hw, *hb;          /* the head */
    uint16_t rev[PK_VAD_NFFT / 2];
    pk_fft   fft;
} pk_vadw;

/* ------------------------------------------------------- the encoder --- */

typedef struct {               /* a PK_Q8 matrix (core/pkf.h) */
    const int8_t *q;
    const float  *s;           /* a scale a row */
    uint32_t rows, cols;
} pk_q8;

typedef struct {
    const float *ff1_nw, *ff1_nb, *att_nw, *att_nb, *conv_nw, *conv_nb;
    const float *ff2_nw, *ff2_nb, *out_nw, *out_nb;
    pk_q8 ff1_up, ff1_down, ff2_up, ff2_down;
    pk_q8 q, k, v, pos, o;
    const float *bias_u, *bias_v;
    pk_q8 pw1, pw2;
    const float *dw, *dw_b;
} pk_layer;

/* ------------------------------------------------------- the decoder --- */

typedef struct {               /* an LSTM layer: gates i f g o */
    pk_q8 wx, wh;              /* [4P][P] */
    const float *b;            /* [4P], the two biases summed */
} pk_lstmw;

/* ------------------------------------------------------------ a model --- */

#define PK_MAX_DUR 16

struct pk_model {
    const unsigned char *map;  /* the file */
    size_t size;
    void  *raw;                /* the copy pk_model_from_bytes made, or NULL: mapped */
    char   name[64];

    /* the features */
    uint32_t sample_rate, n_fft, win_length, hop, n_mels;
    float    preemph, log_guard, norm_eps;
    const float *mel_fb;       /* [n_mels][n_fft/2+1] */
    uint32_t *mel_lo, *mel_n;  /* each band's first nonzero bin and their count */
    float   *window;           /* [n_fft], the file's window centred in zeros */
    uint16_t *rev;             /* [n_fft/2] */
    pk_fft   fft;

    /* the encoder */
    uint32_t n_layers, d_model, n_heads, ff_dim, conv_kernel, sub_channels;
    float    ln_eps;
    const float *pos_freq;     /* [d_model/2] */
    const float *conv0_w, *conv0_b, *dw1_w, *dw1_b, *pw1_b, *dw2_w, *dw2_b, *pw2_b, *out_b;
    pk_q8    sub_pw1, sub_pw2, sub_out;
    pk_layer *layers;

    /* the decoder and the joint */
    uint32_t vocab, blank, pred_hidden, pred_layers, joint_hidden;
    uint32_t n_dur, dur[PK_MAX_DUR], max_symbols;
    const float *embed;        /* [vocab+1][pred_hidden], row-major */
    pk_lstmw *lstm;
    pk_q8    jenc, jpred, jout;     /* the joint: [J][D], [J][P], [n_out_pad][J] */
    const float *jenc_b, *jpred_b, *jout_b;
    uint32_t n_out, n_out_pad;      /* vocab + 1 + n_dur, and that rounded up to 16 */

    /* the vocabulary (core/tokens.c) */
    uint32_t n_pieces;
    const uint32_t *piece_off;
    const uint8_t  *piece_type;
    const char     *piece_text;
    const char     *punct;
    uint32_t        punct_len;

    /* the metadata table, key\0value\0 ... */
    const char *meta;
    size_t      meta_size;

    pk_vadw vad;
};

/* Every byte-level rule core/pkf.h states, on a whole file in memory: 0, or
 * -1 with the reason in err. pk_model_load calls it before forming any
 * pointer into the file; tests/pkfcheck calls it on its malformed corpus. */
int pk_pkf_validate(const unsigned char *buf, size_t n, char *err, size_t errlen);

#endif /* PK_MODEL_H */
