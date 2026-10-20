/* backend/avx2-vnni: backend/avx2 with its int8 products on VPDPBUSD, the
 * AVX-VNNI dot product (Alder Lake and later Intel cores, AMD Zen 5).
 * Everything else is backend/avx2's code; that file says how the two
 * differ. */
#define PK_AVX_VNNI 1
#include "../avx2/kernels.c"
