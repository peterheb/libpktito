/* backend/avx512-vnni: backend/avx2-vnni in AVX512-VNNI's encoding of
 * VPDPBUSD (EVEX, at 256 bits with AVX512-VL) instead of AVX-VNNI's (VEX):
 * the same instruction, on CPUs that have the one and not the other - Ice
 * Lake, Tiger Lake and Zen 4 have only this one. The Makefile builds it
 * with -mavx512vnni -mavx512vl and without -mavxvnni, and backend/avx2
 * picks the encoding from what the compiler was told. */
#define PK_AVX_VNNI 1
#include "../avx2/kernels.c"
