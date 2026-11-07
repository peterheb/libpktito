/* backend/neon-i8mm: backend/neon-sve's int8 loop without its one SVE
 * instruction, for the cores with I8MM and no SVE where its tile is the
 * faster: the Galaxy S24's big cores, the S25's and the first Snapdragon
 * X's (backend/neon-dp/CLIMB.md, 2026-11-06 and 2026-11-07). A probe of
 * the two tiles, on the fastest cores a process may use, keeps it off where
 * SDOT's is: the M4's, and the S24's A520s for a process held to them.
 * Everything else is neon-dp's code, compiled with its flags;
 * backend/neon/kernels.c says how they differ. */
#define PK_NEON_DOTPROD 1
#define PK_NEON_I8MM 1
#include "../neon/kernels.c"
