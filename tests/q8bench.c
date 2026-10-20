/* The int8 product alone, on one thread: what the backend's q8 tile does
 * per cycle, without the rest of the encoder around it.
 *
 *     ./tests/q8bench [T] [REPS]
 *
 * Runs pk_k_gemm_q8 on the encoder's three shapes - a feed-forward's two
 * (4096 x 1024 and 1024 x 4096) and the attention's (1024 x 1024) - at T
 * columns (750: a minute of audio) with the encoder's blocks of 32, and
 * the decoder's (2560 x 640, one column, one block) on random weights and
 * activations, and prints each shape's best time of REPS and its GMAC/s.
 * Divide by the core's clock under load (perf stat, turbostat) for MAC a
 * cycle. backend/avx2/CLIMB.md's kernel rows come from here.
 */
#include "kernels.h"
#include "os.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;

static uint32_t rnd(void) {
    g_rng ^= g_rng >> 12;
    g_rng ^= g_rng << 25;
    g_rng ^= g_rng >> 27;
    return (uint32_t)((g_rng * 0x2545F4914F6CDD1Dull) >> 32);
}

static void *xmalloc(size_t n) {
    n = (n + 63) & ~(size_t)63;
    void *p = aligned_alloc(64, n ? n : 64);
    if (!p) { fprintf(stderr, "q8bench: out of memory\n"); exit(2); }
    memset(p, 0, n ? n : 64);
    return p;
}

static void run(const char *what, uint32_t rows, uint32_t n, uint32_t T, uint32_t qb, int reps) {
    int8_t *w = xmalloc((size_t)rows * n);
    float *ws = xmalloc(rows * sizeof *ws), *x = xmalloc((size_t)T * n * sizeof *x);
    float *out = xmalloc((size_t)T * rows * sizeof *out);
    for (size_t i = 0; i < (size_t)rows * n; i++) w[i] = (int8_t)((int)(rnd() % 255) - 127);
    for (uint32_t r = 0; r < rows; r++) ws[r] = 0.01f;
    for (size_t i = 0; i < (size_t)T * n; i++) x[i] = (float)(rnd() >> 8) / 8388608.0f - 1.0f;
    pk_act a = { xmalloc((size_t)T * n), xmalloc((size_t)T * n / qb * sizeof(float)), n, qb, T };
    pk_k_quantize(&a, x, n);
    double best = 1e30;
    for (int i = 0; i < reps; i++) {
        double t0 = pk_os_now();
        pk_k_gemm_q8(out, rows, w, ws, NULL, rows, &a, 0);
        double dt = pk_os_now() - t0;
        best = dt < best ? dt : best;
    }
    double macs = (double)rows * n * T;
    printf("%-22s %5u x %-5u T %4u qb %4u  %8.3f ms  %6.1f GMAC/s\n", what, rows, n, T, qb,
           best * 1e3, macs / best * 1e-9);
    free(w); free(ws); free(x); free(out); free(a.q); free(a.d);
}

int main(int argc, char **argv) {
    uint32_t T = argc > 1 ? (uint32_t)atoi(argv[1]) : 750;
    int reps = argc > 2 ? atoi(argv[2]) : 5;
    printf("q8bench: %s, one thread\n", pk_k_name());
    run("feed-forward up", 4096, 1024, T, 32, reps);
    run("feed-forward down", 1024, 4096, T, 32, reps);
    run("attention q", 1024, 1024, T, 32, reps);
    run("decoder lstm (T 1)", 2560, 640, 1, 640, reps * 200);
    return 0;
}
