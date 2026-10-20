/* backend/ref under other names, built with the backend's own flags, so
 * that tests/kcheck can hold the backend to it bit for bit in one binary. */
#define pk_k_quantize  ref_quantize
#define pk_k_gemm_q8   ref_gemm_q8
#define pk_k_gemm_f32  ref_gemm_f32
#define pk_k_layernorm ref_layernorm
#define pk_k_softmax   ref_softmax
#define pk_k_silu      ref_silu
#define pk_k_glu       ref_glu
#define pk_k_dwconv    ref_dwconv
#define pk_k_conv3x3s2 ref_conv3x3s2
#define pk_k_lstm      ref_lstm
#define pk_k_stft      ref_stft
#define pk_k_name      ref_name
#include "../backend/ref/kernels.c"
