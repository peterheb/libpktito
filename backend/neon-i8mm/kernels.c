/* backend/neon-i8mm: backend/neon-sve's int8 loop without its one SVE
 * instruction, for the cores with I8MM and no SVE where SMMLA issues as SDOT
 * does: the Galaxy S24's and S25's (backend/neon-dp/CLIMB.md, 2026-11-06).
 * A probe of SMMLA's rate keeps it off the cores where it issues at half.
 * Everything else is neon-dp's code, compiled with its flags;
 * backend/neon/kernels.c says how they differ. */
#define PK_NEON_DOTPROD 1
#define PK_NEON_I8MM 1
#include "../neon/kernels.c"
