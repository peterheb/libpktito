/* backend/neon-dp: backend/neon with its int8 products on SDOT, ARMv8.2's
 * dot product (the A76 of a Pi 5, the A78 and X cores, Apple's M cores).
 * Everything else is backend/neon's code; that file says how the two
 * differ. */
#define PK_NEON_DOTPROD 1
#include "../neon/kernels.c"
