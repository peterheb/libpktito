/* backend/neon-sme: neon-dp with its int8 products on SME2's SMOPA, a
 * panel by 16 columns an instruction, for the CPUs with SME2 at a
 * streaming vector of 512 bits where its product measures the faster than
 * SDOT's with every one of the fastest cores on it at once
 * (backend/kernels.c): the M4 Pro, whose two performance clusters have an
 * SME unit each (backend/neon-dp/CLIMB.md, 2026-11-23). Everything else is
 * neon-dp's code; backend/neon/kernels.c says how they differ. */
#define PK_NEON_DOTPROD 1
#define PK_NEON_SME 1
#include "../neon/kernels.c"
