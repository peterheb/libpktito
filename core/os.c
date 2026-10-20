/* core/os.h on POSIX (Linux, macOS, the BSDs), and the plain-C fallback.
 *
 * The cores: the physical cores of the fastest kind, from Linux's sysfs
 * topology and cpufreq or macOS's sysctl (pk_os_cores says why).
 */
#include "os.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if !defined(PK_NO_MMAP) && (defined(__unix__) || defined(__APPLE__))
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

#endif
