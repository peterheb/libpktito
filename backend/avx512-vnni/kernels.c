/* backend/avx512-vnni: backend/avx2-vnni with its int8 products in zmm, on
 * AVX512-VNNI's VPDPBUSD: a panel's 16 rows in one register, up to eight
 * columns a tile in AVX-512's 32 registers. Ice Lake, Tiger Lake and Zen 4
 * have only this VPDPBUSD; Zen 5, Sapphire Rapids and Granite Rapids have
 * both, and run this one faster (backend/avx2/CLIMB.md, 2026-10-24). The
 * Makefile builds it with -mavx512f -mavx512vnni -mavx512vl and without
 * -mavxvnni, and backend/avx2 takes the zmm product from what the compiler
 * was told. */
#define PK_AVX_VNNI 1
#include "../avx2/kernels.c"
