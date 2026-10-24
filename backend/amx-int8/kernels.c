/* backend/amx-int8: backend/avx512-vnni with its int8 products on AMX's
 * tiles (Sapphire Rapids, Granite Rapids): TDPBSSD, signed bytes by signed
 * bytes into exact int32 sums, a block of 16 columns by a panel's 16 rows
 * at a time. Everything else is backend/avx2's code; that file says how
 * the variants differ. */
#define PK_AVX_VNNI 1
#include "../avx2/kernels.c"
