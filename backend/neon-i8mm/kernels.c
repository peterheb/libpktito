/* backend/neon-i8mm: backend/neon-dp with its int8 products on SMMLA,
 * ARMv8.6's I8MM matrix multiply (Neoverse V1 and later, the A710 and X2
 * and later, Apple's M2 and later): two columns by two rows by eight
 * inputs an instruction. A product's odd last column, and the decoder's
 * one, take the SDOT tile. Everything else is backend/neon's code; that
 * file says how the variants differ. */
#define PK_NEON_DOTPROD 1
#define PK_NEON_I8MM 1
#include "../neon/kernels.c"
