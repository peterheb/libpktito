/* core/os.h on POSIX (Linux, macOS, the BSDs), and the plain-C fallback.
 *
 * The cores: Linux's topology gives the physical cores, SMT siblings
 * counted once, because the encoder's matrix products keep a core's integer
 * units busy and a second thread on it measured nothing (backend CLIMB.md
 * files have the numbers per board). On macOS, sysctl's physical count.
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

int pk_os_cores(void) {
#ifdef __APPLE__
    int phys = 0;
    size_t len = sizeof phys;
    if (sysctlbyname("hw.physicalcpu", &phys, &len, NULL, 0) == 0 && phys > 0)
        return phys;
#endif
#ifdef __linux__
    /* one per distinct (package, core) pair among the online cpus */
    long online = sysconf(_SC_NPROCESSORS_ONLN);
    int seen[1024], nseen = 0;
    for (long c = 0; c < online && c < 4096; c++) {
        char path[128];
        int pkg = 0, core = -1;
        FILE *f;
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%ld/topology/core_id", c);
        if (!(f = fopen(path, "r")))
            continue;
        if (fscanf(f, "%d", &core) != 1)
            core = -1;
        fclose(f);
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%ld/topology/physical_package_id", c);
        if ((f = fopen(path, "r"))) {
            if (fscanf(f, "%d", &pkg) != 1)
                pkg = 0;
            fclose(f);
        }
        if (core < 0)
            continue;
        int key = pkg * 65536 + core, dup = 0;
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
