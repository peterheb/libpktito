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
#include <stdint.h>
#include <stdio.h>

/* A whole file, read-only, or NULL with a reason in why. *size is its
 * length. The bytes stay valid until pk_unmap. The first byte is 64-byte
 * aligned either way (a mapping is page-aligned, the fallback aligns). */
const unsigned char *pk_map_file(const char *path, size_t *size, char *why, size_t whylen);
void pk_unmap(const unsigned char *p, size_t size);

/* fopen, with path in UTF-8 on every system: NULL with errno set, EILSEQ
 * for a path that is not UTF-8. */
FILE *pk_os_fopen(const char *path, const char *mode);

/* For a regular file - not a pipe, a terminal, or stdin from either - its
 * size in bytes, else -1. pk_os_seek goes to an offset from the start and
 * pk_os_tell says where f stands, past 2 GB on every system: 0 or the
 * offset, -1 on failure. */
int64_t pk_os_file_size(FILE *f);
int     pk_os_seek(FILE *f, int64_t off);
int64_t pk_os_tell(FILE *f);

/* A scratch file, open for writing and reading, that goes when it is closed
 * or the process ends; NULL if none can be made. It is on disk where the
 * system has one, so that its pages can leave memory: in $TMPDIR, else
 * /var/tmp, else /tmp (a Pi's Debian 13 keeps /tmp in memory, tmpfs), and
 * on Windows in its temporary folder. */
FILE *pk_os_temp(void);

/* rename and remove, the same: 0, or -1 with errno set. pk_os_rename
 * replaces a file already at to, which Windows' rename refuses to. */
int pk_os_rename(const char *from, const char *to);
int pk_os_remove(const char *path);

/* Cores this process may use: what -t defaults from. 1 when unknown. */
int pk_os_cores(void);

/* Of those, the fastest kind's, where the cores are of two kinds (P and E,
 * big and LITTLE, performance and efficiency) and the OS says which: on
 * Linux the highest cpu_capacity, else cpufreq maximum, and those within
 * 10% of it; macOS's perflevel0; Windows' highest EfficiencyClass. All of
 * them where the cores are of one kind or the OS does not say. */
int pk_os_fast_cores(void);

/* Of pk_os_cores's, every kind's but the slowest's: the cores a diarizer
 * sets against every core (core/diar.c, layers). With two kinds they are
 * pk_os_fast_cores's. With three, a phone's prime, big and little, they
 * are the prime and the big: a Galaxy S24's X4 and five A720s without its
 * two A520s, the six its diarizer was quickest on (7.55 s for the ten
 * minutes, 8.76 on all eight: backend/neon-dp/CLIMB.md, 2026-11-24). On
 * Linux a core is of the slowest kind within 10% of the lowest
 * cpu_capacity, else cpufreq maximum; on macOS in the last perflevel; on
 * Windows of the lowest EfficiencyClass. All of them where the cores are
 * of one kind or the OS does not say. */
int pk_os_cores_but_slowest(void);

/* The cluster of cores the calling thread is on as it asks, from 0: the
 * cores that share an L2, and on Apple's chips with SME the one unit
 * (backend/neon/kernels.c counts a unit's threads by it). macOS's, from
 * sysctl's perflevels: a level's logicalcpu in runs of its cpusperl2, the
 * last level's first, which is the device tree's logical-cluster-id on an
 * M4 Pro (E 0 to 3, P 4 to 7 and 8 to 11), a base M4 (E 0 to 5, P 6 to 9)
 * and an M5 Ultra (six of six; its two levels alternate by die, which runs
 * of six get right and runs of two sizes would not). 0 on every other
 * system. The thread may have moved by the time its answer is used. */
int pk_os_cluster(void);

/* fn(arg, i, n) on n of the fastest kind's cores at once (pk_os_fast_cores's),
 * all of them up to PK_OS_ON_FAST, or where the system gives no speeds,
 * of every core: on Linux and Windows from a thread of its own held to
 * each, the i-th to the i-th core, the caller's thread left as it was; on
 * macOS, which holds no thread to a core, and where no thread can be had,
 * fn(arg, 0, 1) on the calling thread. Thread 0's answer. For a probe of
 * the cores as a transcription runs them, every one busy
 * (backend/kernels.c). */
#define PK_OS_ON_FAST 16
int pk_os_on_fast_cores(int (*fn)(void *, int, int), void *arg);

/* The calling thread out of what the operating system would hold the
 * library's work back with. On Windows that is the EcoQoS its heuristics
 * give the threads of a process without a window, which it confines to the
 * cores it calls efficient. On Linux it is an SVE vector length past 128
 * bits, where neon-sve's loop runs slower (core/os.c). Nothing elsewhere.
 * pk_os_rethrottle, given what it returns, puts the thread back as it was.
 * A pool worker stays out; the caller's thread is out for a pk_asr_run,
 * whose every job it runs a share of, for a pk_vad_probs, which runs on it
 * alone, and for a pk_model_load or a pk_model_from_bytes, whose checks do.
 * A function of the caller's that one of them calls runs as the thread is. */
typedef struct {
    unsigned long control, state; /* Windows' throttling masks */
    int sve;                      /* the SVE length that Linux had, if changed */
} pk_os_throttle;
pk_os_throttle pk_os_unthrottle(void);
void pk_os_rethrottle(pk_os_throttle was);

/* Seconds on a monotonic clock, for timing. */
double pk_os_now(void);

/* Standard input and output as bytes, and the console told to show UTF-8:
 * Windows' C runtime translates line ends by default. Nothing elsewhere. */
void pk_os_stdio(void);

/* The first of the NULL-ended features (backend/kernels.h's pk_k_needs)
 * this CPU, or its operating system, cannot run; NULL if none. A feature
 * it has no way to ask about counts as one it lacks. */
const char *pk_os_cpu_lacks(const char *const *needs);

#endif /* PK_OS_H */
