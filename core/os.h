/* What pktito asks of the operating system beyond the C library.
 *
 * The model file mapped into memory, because at 0.6 GB it should be the
 * page cache's one copy and not a second one in the heap (a Pi 3 has 0.9 GB
 * in all); the number of cores, for the thread pool; a monotonic clock;
 * paths in UTF-8 everywhere; and on Windows, standard output that writes
 * the bytes it is given. core/os.c answers with POSIX or Windows where it
 * has them; built with -DPK_NO_MMAP, or on a system with neither, the file
 * is read into memory instead and the pool asks for one thread.
 *
 * Every path here is UTF-8. On POSIX that is the bytes the system takes;
 * Windows' ANSI functions (fopen, CreateFileA) would read them in the
 * process's code page and miss C:\Users\José, so there they become UTF-16
 * for the W functions, and bytes that are not UTF-8 fail with EILSEQ.
 */
#ifndef PK_OS_H
#define PK_OS_H

#include <stddef.h>
#include <stdio.h>

/* A whole file, read-only, or NULL with a reason in why. *size is its
 * length. The bytes stay valid until pk_unmap. The first byte is 64-byte
 * aligned either way (a mapping is page-aligned, the fallback aligns). */
const unsigned char *pk_map_file(const char *path, size_t *size, char *why, size_t whylen);
void pk_unmap(const unsigned char *p, size_t size);

/* fopen, with path in UTF-8 on every system: NULL with errno set, EILSEQ
 * for a path that is not UTF-8. */
FILE *pk_os_fopen(const char *path, const char *mode);

/* Cores this process may use: what -t defaults from. 1 when unknown. */
int pk_os_cores(void);

/* Of those, the fastest kind's, where the cores are of two kinds (P and E,
 * big and LITTLE, performance and efficiency) and the OS says which: on
 * Linux the highest cpu_capacity, else cpufreq maximum, and those within
 * 10% of it; macOS's perflevel0; Windows' highest EfficiencyClass. All of
 * them where the cores are of one kind or the OS does not say. */
int pk_os_fast_cores(void);

/* The calling thread out of the operating system's power throttling where
 * it has one: on Windows the EcoQoS its heuristics give the threads of a
 * process without a window, which it confines to the cores it calls
 * efficient.
 * Nothing elsewhere. pk_os_rethrottle, given what it returns, puts the
 * thread back as it was. A pool worker stays out; the caller's thread is
 * out for a pk_asr_run, whose every job it runs a share of, and for a
 * pk_vad_probs, which runs on it alone. */
typedef struct { unsigned long control, state; } pk_os_throttle;
pk_os_throttle pk_os_unthrottle(void);
void pk_os_rethrottle(pk_os_throttle was);

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
