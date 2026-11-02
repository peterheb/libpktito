/* backend/neon-sve: backend/neon-dp with its 4-column int8 tiles on I8MM's
 * SMMLA, each half panel's weights put in SMMLA's order once a call, and
 * SVE's MOVPRFX starting each block's sums: for the cores with SVE and
 * I8MM, measured on the GB10's X925 and A725. Everything else is neon-dp's
 * code, compiled with its flags; backend/neon/kernels.c says how they
 * differ. */
#define PK_NEON_DOTPROD 1
#define PK_NEON_SVE 1
#include "../neon/kernels.c"
