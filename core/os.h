/* What pktito asks of the operating system beyond the C library.
 *
 * The model file mapped into memory, because at 0.6 GB it should be the
 * page cache's one copy and not a second one in the heap (a Pi 3 has 0.9 GB
 * in all); the number of cores, for the thread pool; a monotonic clock; and
 * on Windows, standard output that writes the bytes it is given. core/os.c
 * answers with POSIX or Windows where it has them; built with -DPK_NO_MMAP,
 * or on a system with neither, the file is read into memory instead and
 * the pool asks for one thread.
 */
#ifndef PK_OS_H
#define PK_OS_H

#include <stddef.h>

/* A whole file, read-only, or NULL with a reason in why. *size is its
 * length. The bytes stay valid until pk_unmap. The first byte is 64-byte
 * aligned either way (a mapping is page-aligned, the fallback aligns). */
const unsigned char *pk_map_file(const char *path, size_t *size, char *why, size_t whylen);
void pk_unmap(const unsigned char *p, size_t size);

/* Cores this process may use: what -t defaults from. 1 when unknown. */
int pk_os_cores(void);

/* Seconds on a monotonic clock, for timing. */
double pk_os_now(void);

/* Standard input and output as bytes, and the console told to show UTF-8:
 * Windows' C runtime translates line ends by default. Nothing elsewhere. */
void pk_os_stdio(void);

/* The first of the NULL-ended features (backend/kernels.h's pk_k_needs)
 * this CPU, or its operating system, cannot run; NULL if none. A feature
 * it has no way to ask about counts as there. */
const char *pk_os_cpu_lacks(const char *const *needs);

#endif /* PK_OS_H */
