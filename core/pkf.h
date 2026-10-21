/* pktito's model file (.pkf): a Parakeet TDT model, its vocabulary, and the
 * Silero VAD that cuts long recordings into its input, in one file, with a
 * table saying where every number in it came from. Plain C11;
 * tools/pktito_tools/pkf.py mirrors these structs and writes the file.
 *
 *   [pk_header]                      256 bytes
 *   [pk_param * n_params]            40 bytes each: the architecture's numbers
 *   [pk_tensor * n_tensors]          96 bytes each
 *   [vocabulary]                     the pieces, see below
 *   [metadata]                       key\0value\0 pairs, UTF-8; may be empty
 *   ... zero pad to 64 ...
 *   [tensor data]                    each tensor 64-byte aligned
 *
 * The file is mapped read-only (core/os.h) and every tensor is used where
 * it lies. Every offset is from the start of the file. Integers and floats
 * are little-endian; the engine refuses to build for a big-endian target.
 *
 * The header names the architecture ("parakeet-tdt") and every shape the
 * engine needs is a param, so the engine binds tensors to shapes it derived
 * itself and never takes one from a descriptor on trust (core/model.c).
 *
 * ---------------------------------------------------------------- types --
 *
 * PK_F32   float32, row-major.
 *
 * PK_Q8    int8 weights for the encoder's matrix products (backend/kernels.h,
 *          pk_k_gemm_q8), q in [-127, 127], with one float32 scale for each
 *          output row in a tensor of its own, "<name>.scale" [rows]: element
 *          (r, k) of the matrix is q(r, k) * scale[r]. The bytes are in panels
 *          of PK_QR rows, and within a panel in groups of PK_QG inputs:
 *
 *            q(r, k) is at  ((r / 16) * (cols / 4) + k / 4) * 64 + (r % 16) * 4 + k % 4
 *
 *          so a panel is 16 rows by all of cols, one group is 64 bytes, the
 *          4 inputs of each of 16 rows, and a SIMD register of 16 bytes holds
 *          4 rows of 4 inputs - the shape ARM's SDOT and x86's VPDPBUSD
 *          multiply in one instruction, and the shape the A53's SMULL/SADALP
 *          and AVX2's VPMADDUBSW reach in two. rows is a multiple of 16 and
 *          cols of 32.
 *
 * ----------------------------------------------------------- vocabulary --
 *
 *   u32  n                      pieces, the vocabulary without the blank
 *   u32  off[n + 1]             where each piece's bytes start, from text[0];
 *                               ascending, off[0] = 0, off[n] = the text's size
 *   u8   type[n]                PK_PIECE_*
 *   ...  text                   the pieces, UTF-8, each as SentencePiece spells
 *                               it (U+2581 for a space), no terminators
 *   u32  np                     bytes of punctuation
 *   ...  punct                  UTF-8: the characters NeMo takes the space
 *                               before away from (core/tokens.c)
 *
 * --------------------------------------------------------------- params --
 *
 *   sample_rate n_fft win_length hop_length n_mels         the features
 *   preemph log_guard norm_eps                              (f32)
 *   n_layers d_model n_heads ff_dim conv_kernel sub_channels   the encoder
 *   ln_eps                                                  (f32)
 *   vocab pred_hidden pred_layers joint_hidden              the decoder
 *   n_durations dur0 .. dur<n-1> max_symbols                TDT's search
 *
 * -------------------------------------------------------------- tensors --
 *
 * D = d_model, F = ff_dim, H = n_heads, C = sub_channels, M = n_mels,
 * K = conv_kernel, P = pred_hidden, J = joint_hidden, V = vocab, N = n_fft:
 *
 *   mel.fb [M][N/2+1] f32           the mel filterbank (librosa's, slaney)
 *   mel.window [win_length] f32     the analysis window (symmetric Hann)
 *   mel.fft.cos, mel.fft.sin [N/2]  cos and sin of 2 pi k / N, the FFT's twiddles
 *   pos.freq [D/2] f32              the positional encoding's frequencies
 *   sub.conv0.w [C][9], .b [C]      3x3 stride 2, from one channel
 *   sub.dw1.w [C][9], .b [C]        3x3 stride 2, depthwise
 *   sub.pw1.w [C][C] q8, .b [C]     pointwise
 *   sub.dw2.w, .b, sub.pw2.w, .b    the same again
 *   sub.out.w [D][C*M/8] q8, .b [D] the projection; its inputs in the order
 *                                   frequency-major, channel-minor, which is
 *                                   NeMo's columns permuted (pktito keeps the
 *                                   subsampling's activations channel-last)
 *   l<i>.ff1.norm.w, .b [D]         and so on for ff2, att, conv, out
 *   l<i>.ff1.up [F][D] q8, l<i>.ff1.down [D][F] q8 (holding 0.5 x NeMo's)
 *   l<i>.att.q, .k, .v, .pos, .o [D][D] q8
 *   l<i>.att.bias_u, .bias_v [D]    the position biases, head-major
 *   l<i>.conv.pw1 [2D][D] q8
 *   l<i>.conv.dw [D][K], .dw_b [D]  the depthwise convolution, BatchNorm folded in
 *   l<i>.conv.pw2 [D][D] q8
 *   dec.embed [V+1][P] f32          row V, the blank, is zeros
 *   dec.l<i>.wx, .wh [4P][P] q8, .b [4P]   the LSTMs, gates i f g o, biases summed
 *   joint.enc.w [J][D] q8, .b [J]
 *   joint.pred.w [J][P] q8, .b [J]
 *   joint.out.w [O][J] q8, .b [O]   O = V+1+n_durations rounded up to 16, the
 *                                   rows past V+1+n_durations zero
 *   vad.*                           Silero VAD at 16 kHz, vadito's layout
 *                                   (core/vad.c lists them)
 *
 * and each q8 tensor's .scale beside it.
 */
#ifndef PK_PKF_H
#define PK_PKF_H

#include <stdint.h>

#define PK_MAGIC    0x01464b50u   /* "PKF\1" read as a little-endian u32 */
#define PK_VERSION  1
#define PK_ALIGN    64            /* every tensor, from the start of the file */
#define PK_NAME     48            /* bytes in a tensor name, its NUL included */
#define PK_KEY      32            /* bytes in a param key, its NUL included */
#define PK_QR       16            /* rows in a q8 panel */
#define PK_QG       4             /* inputs in a q8 group */

enum { PK_F32 = 0, PK_Q8 = 1, PK_NTYPES };
enum { PK_PARAM_U32 = 0, PK_PARAM_I32 = 1, PK_PARAM_F32 = 2 };
enum { PK_PIECE_NORMAL = 1, PK_PIECE_UNKNOWN = 2, PK_PIECE_CONTROL = 3,
       PK_PIECE_USER = 4, PK_PIECE_UNUSED = 5 };

typedef struct {
    uint32_t magic, version;
    char     arch[32];            /* NUL-terminated: "parakeet-tdt" */
    char     name[64];            /* NUL-terminated: "parakeet-tdt-0.6b-v2" */
    uint32_t n_params, n_tensors;
    uint64_t params_off, tensors_off;
    uint64_t vocab_off, vocab_size;
    uint64_t meta_off, meta_size;
    uint64_t data_off;            /* where tensors may begin: past every table */
    uint64_t file_bytes;          /* the whole file, so a truncation is caught at once */
    uint8_t  digest[32];          /* SHA-256 of the bytes from data_off on, for tools */
    uint8_t  reserved[256 - 8 - 32 - 64 - 8 - 8 * 8 - 32];
} pk_header;

typedef struct {
    char     key[PK_KEY];         /* NUL-terminated, NUL-padded */
    uint32_t kind;                /* PK_PARAM_* */
    union { uint32_t u; int32_t i; float f; } v;
} pk_param;

typedef struct {
    char     name[PK_NAME];       /* NUL-terminated, NUL-padded */
    uint32_t type;                /* PK_F32 or PK_Q8 */
    uint32_t ndim;                /* 1 or 2 */
    uint32_t dims[2];             /* rows, cols; cols is 1 when ndim is 1 */
    uint64_t offset;              /* from the start of the file, PK_ALIGN-aligned */
    uint64_t bytes;               /* 4 rows cols for f32, rows cols for q8 */
    uint64_t reserved[2];         /* zero */
} pk_tensor;

_Static_assert(sizeof(pk_header) == 256, "pk_header is 256 bytes");
_Static_assert(sizeof(pk_param) == 40, "pk_param is 40 bytes");
_Static_assert(sizeof(pk_tensor) == 96, "pk_tensor is 96 bytes");

#endif /* PK_PKF_H */
