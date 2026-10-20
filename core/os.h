/* What pkito asks of the operating system beyond the C library.
 *
 * Two things: the model file mapped into memory, because at 0.6 GB it
 * should be the page cache's one copy and not a second one in the heap (a
 * Pi 3 has 0.9 GB in all), and the number of cores, for the thread pool.
 * core/os.c answers both with POSIX where it has it; built with
 * -DPK_NO_MMAP, or on a system without mmap, the file is read into memory
 * instead and the pool asks for one thread.
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

#endif /* PK_OS_H */
