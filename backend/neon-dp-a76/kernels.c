/* backend/neon-dp-a76: neon-dp with its int8 products in a tile written for
 * the Cortex-A76, a whole panel by 2 columns with no SCVTF, where its
 * product measures the faster than neon-dp's on the fastest cores
 * (backend/kernels.c; backend/neon-dp/CLIMB.md, 2026-11-08). Everything
 * else is neon-dp's code, compiled with its flags; backend/neon/kernels.c
 * says how they differ. */
#define PK_NEON_DOTPROD 1
#define PK_NEON_A76 1
#include "../neon/kernels.c"
