/* core/os.h on POSIX (Linux, macOS, the BSDs), on Windows 10 and later,
 * and the plain-C fallback.
 *
 * The cores: the physical cores of every kind, from Linux's sysfs topology,
 * macOS's sysctl or Windows' processor records (pk_os_cores says why).
 */
#include "os.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if !defined(PK_NO_MMAP) && defined(_WIN32)
#define PK_WINDOWS 1
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#elif !defined(PK_NO_MMAP) && (defined(__unix__) || defined(__APPLE__))
#define PK_POSIX 1
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
#endif

#ifdef PK_POSIX

const unsigned char *pk_map_file(const char *path, size_t *size, char *why, size_t whylen) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        snprintf(why, whylen, "%s", strerror(errno));
        return NULL;
    }
    struct stat st;
    if (fstat(fd, &st) < 0 || st.st_size <= 0 || (uintmax_t)st.st_size > SIZE_MAX) {
        close(fd);
        snprintf(why, whylen, "%s", st.st_size <= 0 ? "empty file" : "cannot tell its size");
        return NULL;
    }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) {
        snprintf(why, whylen, "mmap: %s", strerror(errno));
        return NULL;
    }
    *size = (size_t)st.st_size;
    return p;
}

void pk_unmap(const unsigned char *p, size_t size) {
    if (p)
        munmap((void *)p, size);
}

#ifdef __linux__
static long sysfs_long(long cpu, const char *what) {
    char path[128];
    long v = -1;
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%ld/%s", cpu, what);
    FILE *f = fopen(path, "r");
    if (f) {
        if (fscanf(f, "%ld", &v) != 1)
            v = -1;
        fclose(f);
    }
    return v;
}
#endif

/* One thread a physical core, of every kind. A chip with two kinds of core
 * (Intel's P and E, ARM's big and LITTLE, Apple's performance and
 * efficiency) runs the encoder fastest on all of them: the pool deals each
 * pass's items from a counter, so a slower core takes fewer, and on a chip
 * whose fast kind is a minority the slow cores are most of its speed. The
 * minute whole (`pkito bench --max-chunk 60`), 2026-10-20: an Intel Core 3
 * 304 (1 P + 4 E) 5.02 s on its P-core, 1.39 s on all five; an i9-12900HK
 * (6 P + 8 E) 0.960 s on its P-cores, 0.906 s on all 14, though 8, 16 and
 * 18 threads take 1.10 to 1.15 s there; an M4 Pro (8 + 4) 0.498 s on its
 * performance cores, 0.488 s on all 12 (backend/avx2/CLIMB.md,
 * backend/neon-dp/CLIMB.md). The fast kind alone measured better on the
 * tree of two days before (CLIMB.md, "Threads"). A second thread on a core
 * (SMT) measured nothing (the 12900HK on 20: 0.91 to 0.93 s), so siblings
 * count once. */
int pk_os_cores(void) {
#ifdef __APPLE__
    int phys = 0;
    size_t len = sizeof phys;
    if (sysctlbyname("hw.physicalcpu", &phys, &len, NULL, 0) == 0 && phys > 0)
        return phys;
#endif
#ifdef __linux__
    long ncpu = sysconf(_SC_NPROCESSORS_CONF);
    if (ncpu > 4096)
        ncpu = 4096;
    /* one per distinct (package, core) pair */
    int seen[1024], nseen = 0;
    for (long c = 0; c < ncpu; c++) {
        long core = sysfs_long(c, "topology/core_id"), pkg = sysfs_long(c, "topology/physical_package_id");
        if (core < 0)
            continue;
        int key = (int)((pkg < 0 ? 0 : pkg) * 65536 + core), dup = 0;
        for (int i = 0; i < nseen; i++)
            dup |= seen[i] == key;
        if (!dup && nseen < 1024)
            seen[nseen++] = key;
    }
    if (nseen > 0)
        return nseen;
#endif
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n < 1 ? 1 : (int)n;
}

double pk_os_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

void pk_os_stdio(void) {}

#elif defined(PK_WINDOWS)

/* A view keeps its file open by itself, so both handles close here. */
const unsigned char *pk_map_file(const char *path, size_t *size, char *why, size_t whylen) {
    HANDLE f = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, NULL);
    if (f == INVALID_HANDLE_VALUE) {
        DWORD e = GetLastError();
        snprintf(why, whylen, "%s", e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND
                                        ? "No such file or directory" : "cannot open it");
        return NULL;
    }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(f, &sz) || sz.QuadPart <= 0 || (uint64_t)sz.QuadPart > SIZE_MAX) {
        CloseHandle(f);
        snprintf(why, whylen, "%s", "empty file or cannot tell its size");
        return NULL;
    }
    HANDLE m = CreateFileMappingA(f, NULL, PAGE_READONLY, 0, 0, NULL);
    void *p = m ? MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0) : NULL;
    DWORD e = GetLastError();
    if (m)
        CloseHandle(m);
    CloseHandle(f);
    if (!p) {
        snprintf(why, whylen, "cannot map it (error %lu)", (unsigned long)e);
        return NULL;
    }
    *size = (size_t)sz.QuadPart;
    return p;
}

void pk_unmap(const unsigned char *p, size_t size) {
    (void)size;
    if (p)
        UnmapViewOfFile(p);
}

/* As on POSIX, the physical cores (pk_os_cores there says why): one
 * processor record a core, counted where the process may run (`start
 * /affinity`), in processor group 0, which holds every machine with up to
 * 64 logical processors. */
int pk_os_cores(void) {
    DWORD_PTR pmask = ~(DWORD_PTR)0, smask;
    GetProcessAffinityMask(GetCurrentProcess(), &pmask, &smask);
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    char *buf = len ? malloc(len) : NULL;
    int n = 0;
    if (buf && GetLogicalProcessorInformationEx(RelationProcessorCore,
                   (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)(void *)buf, &len))
        for (DWORD at = 0; at < len;) {
            const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *r = (const void *)(buf + at);
            at += r->Size;
            n += (r->Processor.GroupMask[0].Mask & (KAFFINITY)pmask) != 0;
        }
    free(buf);
    return n > 0 ? n : 1;
}

double pk_os_now(void) {
    static LARGE_INTEGER hz;
    LARGE_INTEGER t;
    if (!hz.QuadPart)
        QueryPerformanceFrequency(&hz);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)hz.QuadPart;
}

/* The bytes as written: no \r added before each \n on a pipe or a file,
 * where the C runtime's text mode would add one and a transcript compared
 * with tests/data's would differ; and UTF-8 to the console, which shows
 * v3's accents and Cyrillic in place of the code page's guesses. */
void pk_os_stdio(void) {
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stdin), _O_BINARY);
    SetConsoleOutputCP(CP_UTF8);
}

#else /* the plain-C fallback: read the file, one core, the C11 clock */

const unsigned char *pk_map_file(const char *path, size_t *size, char *why, size_t whylen) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        snprintf(why, whylen, "%s", strerror(errno));
        return NULL;
    }
    long n = -1;
    if (fseek(f, 0, SEEK_END) == 0)
        n = ftell(f);
    if (n <= 0 || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        snprintf(why, whylen, "cannot tell its size");
        return NULL;
    }
    /* 64 bytes in front: the aligned start, and room to find the block */
    unsigned char *raw = malloc((size_t)n + 64);
    if (!raw) {
        fclose(f);
        snprintf(why, whylen, "out of memory");
        return NULL;
    }
    unsigned char *p = raw + 64 - ((uintptr_t)raw & 63);
    p[-1] = (unsigned char)(p - raw);
    size_t got = fread(p, 1, (size_t)n, f);
    fclose(f);
    if (got != (size_t)n) {
        free(raw);
        snprintf(why, whylen, "read %zu of %ld bytes", got, n);
        return NULL;
    }
    *size = (size_t)n;
    return p;
}

void pk_unmap(const unsigned char *p, size_t size) {
    (void)size;
    if (p)
        free((void *)(p - p[-1]));
}

int pk_os_cores(void) { return 1; }

double pk_os_now(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

void pk_os_stdio(void) {}

#endif

/* -------------------------------------------------------- CPU features --- */

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>

/* CPUID's bits for each feature, and the register state the operating
 * system must have switched on (XCR0) for its instructions to run: YMM's
 * for AVX2's, and the mask and ZMM registers too for any EVEX instruction,
 * the 256-bit ones included. */
static int cpu_has(const char *f) {
    unsigned a, b, c, d, xcr0 = 0;
    if (!__get_cpuid(1, &a, &b, &c, &d))
        return 0;
    const int fma = (c >> 12) & 1, osxsave = (c >> 27) & 1;
    if (osxsave) {
        unsigned lo, hi;
        __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        xcr0 = lo;
    }
    const int ymm = (xcr0 & 0x6) == 0x6, zmm = (xcr0 & 0xe6) == 0xe6;
    unsigned b7 = 0, c7 = 0, a71 = 0;
    if (__get_cpuid_count(7, 0, &a, &b, &c, &d)) {
        b7 = b;
        c7 = c;
        if (a >= 1 && __get_cpuid_count(7, 1, &a, &b, &c, &d))
            a71 = a;
    }
    if (!strcmp(f, "fma"))
        return fma && ymm;
    if (!strcmp(f, "avx2"))
        return ((b7 >> 5) & 1) && ymm;
    if (!strcmp(f, "avxvnni"))
        return ((a71 >> 4) & 1) && ymm;
    if (!strcmp(f, "avx512vnni"))
        return ((c7 >> 11) & 1) && zmm;
    if (!strcmp(f, "avx512vl"))
        return ((b7 >> 31) & 1) && zmm;
    return 1;
}

#elif defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif

static int cpu_has(const char *f) {
    if (!strcmp(f, "asimddp"))
        return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
    return 1;
}

#elif (defined(__aarch64__) || defined(_M_ARM64)) && defined(PK_WINDOWS)
#ifndef PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE
#define PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE 43
#endif

static int cpu_has(const char *f) {
    if (!strcmp(f, "asimddp"))
        return IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE) != 0;
    return 1;
}

#else /* macOS on Apple silicon, whose every core has the dot product, and the rest */

static int cpu_has(const char *f) {
    (void)f;
    return 1;
}

#endif

const char *pk_os_cpu_lacks(const char *const *needs) {
    for (; *needs; needs++)
        if (!cpu_has(*needs))
            return *needs;
    return NULL;
}
