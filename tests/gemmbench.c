/* The int8 product alone, on the encoder's shapes, timed: for the kernel
 * climbs, where a whole transcription takes minutes on a Pi 3 and this takes
 * seconds.
 *
 *     ./tests/gemmbench [T] [threads] [MHz]
 *
 * Each shape is called as core/encoder.c's pk_gemm calls it: in slices of 64
 * rows, every column of a T-column activation (T 250 is a 20 s chunk, what
 * --max-chunk 30 makes of Silero's 60 s test.wav), activations in blocks of
 * 32. With more than one thread the slices are dealt from a counter, as the
 * pool deals them. MAC/cycle is per core, at MHz (default: cpu0's cpufreq,
 * read before and after; a run whose clock moved says so).
 */
#include "kernels.h"
#include "pkf.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double cur_mhz(void) {
    FILE *f = fopen("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "r");
    long khz = 0;
    if (f) {
        if (fscanf(f, "%ld", &khz) != 1) khz = 0;
        fclose(f);
    }
    return (double)khz / 1000.0;
}

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) {
    g_rng ^= g_rng >> 12;
    g_rng ^= g_rng << 25;
    g_rng ^= g_rng >> 27;
    return (uint32_t)((g_rng * 0x2545F4914F6CDD1Dull) >> 32);
}

static void *amalloc(size_t n) {
    n = (n + 63) & ~(size_t)63;
    void *p = aligned_alloc(64, n);
    if (!p) { fprintf(stderr, "gemmbench: out of memory\n"); exit(2); }
    memset(p, 0, n);
    return p;
}

typedef struct {
    float *out;
    const int8_t *w;
    const float *ws;
    uint32_t rows;
    const pk_act *a;
    _Atomic uint32_t next;
} job;

static void *worker(void *v) {
    job *j = v;
    for (;;) {
        uint32_t r0 = atomic_fetch_add(&j->next, 64);
        if (r0 >= j->rows)
            return NULL;
        uint32_t nr = j->rows - r0 < 64 ? j->rows - r0 : 64;
        pk_k_gemm_q8(j->out + r0, j->rows, j->w + (size_t)r0 * j->a->n, j->ws + r0, NULL, nr, j->a, 0);
    }
}

int main(int argc, char **argv) {
    uint32_t T = argc > 1 ? (uint32_t)atoi(argv[1]) : 250;
    int nth = argc > 2 ? atoi(argv[2]) : 1;
    double mhz = argc > 3 ? atof(argv[3]) : 0;
    static const struct { const char *name; uint32_t rows, n; } shapes[] = {
        { "ff up      4096x1024", 4096, 1024 },
        { "ff down    1024x4096", 1024, 4096 },
        { "q k v o    1024x1024", 1024, 1024 },
        { "conv pw1   2048x1024", 2048, 1024 },
    };
    printf("%s, T %u, %d thread%s\n", pk_k_name(), T, nth, nth == 1 ? "" : "s");
    double tot_mac = 0, tot_s = 0;
    for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; i++) {
        uint32_t rows = shapes[i].rows, n = shapes[i].n;
        int8_t *w = amalloc((size_t)rows * n), *q = amalloc((size_t)T * n);
        float *ws = amalloc(rows * 4), *d = amalloc((size_t)T * n / 32 * 4);
        float *out = amalloc((size_t)T * rows * 4);
        for (size_t k = 0; k < (size_t)rows * n; k++) w[k] = (int8_t)((int)(rnd() % 255) - 127);
        for (size_t k = 0; k < (size_t)T * n; k++) q[k] = (int8_t)((int)(rnd() % 255) - 127);
        for (uint32_t r = 0; r < rows; r++) ws[r] = 0.01f;
        for (size_t k = 0; k < (size_t)T * n / 32; k++) d[k] = 0.02f;
        pk_act a = { q, d, n, 32, T };
        double best = 1e30, m0 = cur_mhz(), m1 = 0;
        for (int rep = 0; rep < 5; rep++) {
            job j = { out, w, ws, rows, &a, 0 };
            pthread_t th[64];
            double t0 = now_s();
            for (int k = 1; k < nth; k++) pthread_create(&th[k], NULL, worker, &j);
            worker(&j);
            for (int k = 1; k < nth; k++) pthread_join(th[k], NULL);
            double dt = now_s() - t0;
            best = dt < best ? dt : best;
        }
        m1 = cur_mhz();
        double macs = (double)rows * n * T, f = mhz > 0 ? mhz : m0;
        printf("  %s  %8.2f ms  %6.2f GMAC/s  %5.2f MAC/cycle/core%s\n", shapes[i].name, best * 1e3,
               macs / best * 1e-9, f > 0 ? macs / best / (f * 1e6) / nth : 0.0,
               m0 != m1 ? "  (clock moved)" : "");
        tot_mac += macs;
        tot_s += best;
        free(w); free(q); free(ws); free(d); free(out);
    }
    double f = mhz > 0 ? mhz : cur_mhz();
    printf("  all four            %8.2f ms  %6.2f GMAC/s  %5.2f MAC/cycle/core\n", tot_s * 1e3,
           tot_mac / tot_s * 1e-9, f > 0 ? tot_mac / tot_s / (f * 1e6) / nth : 0.0);
    return 0;
}
