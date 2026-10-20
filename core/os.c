/* core/os.h on POSIX (Linux, macOS, the BSDs), on Windows 10 and later,
 * and the plain-C fallback.
 *
 * The cores: the physical cores of the fastest kind, from Linux's sysfs
 * topology and cpufreq, macOS's sysctl or Windows' processor records
 * (pk_os_cores says why).
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

/* One thread a physical core of the fastest kind. A chip with two kinds of
 * core (Intel's P and E, ARM's big and LITTLE, Apple's performance and
 * efficiency) runs the encoder fastest on the fast kind alone: a slow core
 * holding a job's last item keeps the fast ones waiting at the job's end,
 * six hundred jobs a recording, and on a power-capped chip it takes their
 * watts too. An i9-12900HK: 6 threads on its P-cores 1.062 s, 14 on every
 * core 1.191 s (backend/avx2/CLIMB.md, "Threads"); an M4 Pro: 8 on its
 * performance cores 0.512 s, 12 on all 0.523 s. A second thread on a core
 * (SMT) measured nothing either, so siblings count once. On Linux the fast
 * kind is the cores whose cpufreq maximum is within 10% of the fastest's
 * (an Intel part's favoured cores turbo a bin or two higher than the rest
 * of their kind); where cpufreq says nothing, every core counts. */
int pk_os_cores(void) {
#ifdef __APPLE__
    int phys = 0;
    size_t len = sizeof phys;
    if (sysctlbyname("hw.perflevel0.physicalcpu", &phys, &len, NULL, 0) == 0 && phys > 0)
        return phys;
    len = sizeof phys;
    if (sysctlbyname("hw.physicalcpu", &phys, &len, NULL, 0) == 0 && phys > 0)
        return phys;
#endif
#ifdef __linux__
    long ncpu = sysconf(_SC_NPROCESSORS_CONF), fmax = 0;
    if (ncpu > 4096)
        ncpu = 4096;
    for (long c = 0; c < ncpu; c++) {
        long f = sysfs_long(c, "cpufreq/cpuinfo_max_freq");
        fmax = f > fmax ? f : fmax;
    }
    /* one per distinct (package, core) pair among the fast cpus */
    int seen[1024], nseen = 0;
    for (long c = 0; c < ncpu; c++) {
        long core = sysfs_long(c, "topology/core_id"), pkg = sysfs_long(c, "topology/physical_package_id");
        long f = sysfs_long(c, "cpufreq/cpuinfo_max_freq");
        if (core < 0 || (fmax > 0 && f > 0 && 10 * f < 9 * fmax))
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

/* As on Linux, the physical cores of the fastest kind: one processor record
 * a core, its EfficiencyClass higher the faster the kind (0 on a chip with
 * one kind), counted where the process may run (`start /affinity`), in
 * processor group 0, which holds every machine with up to 64 logical
 * processors. */
int pk_os_cores(void) {
    DWORD_PTR pmask = ~(DWORD_PTR)0, smask;
    GetProcessAffinityMask(GetCurrentProcess(), &pmask, &smask);
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    char *buf = len ? malloc(len) : NULL;
    int n = 0, best = -1;
    if (buf && GetLogicalProcessorInformationEx(RelationProcessorCore,
                   (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)(void *)buf, &len))
        for (DWORD at = 0; at < len;) {
            const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *r = (const void *)(buf + at);
            at += r->Size;
            if (!(r->Processor.GroupMask[0].Mask & (KAFFINITY)pmask))
                continue;
            int kind = r->Processor.EfficiencyClass;
            n = kind > best ? 1 : n + (kind == best);
            best = kind > best ? kind : best;
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
