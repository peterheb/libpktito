/* The int8 product at its edges, for every backend of the build this CPU
 * can run, held to ref bit for bit. Each operand ends where a page nothing
 * may touch begins, so a tile that loads past its last weight, activation,
 * scale or output faults: ASan sees no load an assembly loop makes. The
 * cases are the ones such a loop gets wrong first: sums at both extremes
 * (every product 128 * 127 or -128 * 127) and ones that cancel exactly,
 * zero scales of both signs' rows, an output stride that is no multiple of
 * 4 with the floats between columns left as they were, odd columns, both
 * sides of a column block's edge, the widest input, no columns at all, and
 * the three roundings besides the default. kcheck holds the same products
 * to ref on the model's shapes and on random data.
 *
 *   tests/q8check
 */
#include "kernels.h"
#include "os.h"
#include "pkf.h"

#include <fenv.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

static const pk_kernels *kt;
static int g_cases;

/* n bytes that end where a page no access may touch begins */
typedef struct {
    void *map, *p;
    size_t size;
} guarded;

static guarded guard(size_t n) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    const size_t page = si.dwPageSize;
#else
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
#endif
    const size_t body = (n + page - 1) / page * page;
    guarded g = { NULL, NULL, body + 2 * page };
#ifdef _WIN32
    DWORD was;
    g.map = VirtualAlloc(NULL, g.size, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS);
    if (!g.map || !VirtualProtect((char *)g.map + page, body, PAGE_READWRITE, &was)) {
#else
    g.map = mmap(NULL, g.size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (g.map == MAP_FAILED || mprotect((char *)g.map + page, body, PROT_READ | PROT_WRITE)) {
#endif
        printf("q8check: no guarded memory\n");
        exit(2);
    }
    g.p = (char *)g.map + page + body - n;
    return g;
}

static void unguard(guarded g) {
#ifdef _WIN32
    VirtualFree(g.map, 0, MEM_RELEASE);
#else
    munmap(g.map, g.size);
#endif
}

static int same(const char *what, const float *got, const float *want, size_t n) {
    g_cases++;
    for (size_t i = 0; i < n; i++)
        if (memcmp(got + i, want + i, sizeof *got)) {
            printf("FAIL %s gemm_q8 %s: float %zu is %a, ref's %a\n", kt->id, what, i, got[i],
                   want[i]);
            return 1;
        }
    return 0;
}

/* T columns of n inputs in blocks of qb through 32 rows, outputs 35 floats
 * apart; bias with flags & 1, added to the outputs with flags & 2 */
static int q8_case(uint32_t n, uint32_t qb, uint32_t T, int flags) {
    const uint32_t rows = 32, os = 35, nb = n / qb;
    guarded gw = guard((size_t)rows * n), gq = guard((size_t)T * n);
    guarded gd = guard((size_t)T * nb * 4), gs = guard(rows * 4), gb = guard(rows * 4);
    guarded go = guard((size_t)T * os * 4);
    int8_t *w = gw.p, *q = gq.p;
    float *d = gd.p, *ws = gs.p, *b = gb.p, *o = go.p, *want = malloc((size_t)T * os * 4);
    if (!want)
        exit(2);
    for (uint32_t r = 0; r < rows; r++) {
        ws[r] = (r % 2 ? -1 : 1) * 0.03125f;
        b[r] = (float)(r % 7) - 3.25f;
        /* rows of -128, of 127, and of 127 and -127 or -128 by turns: the
         * extreme sums, both signs, and ones that cancel exactly (the
         * activations below are never -128, the weights may be) */
        for (uint32_t k = 0; k < n; k++)
            w[((size_t)(r / PK_QR) * (n / PK_QG) + k / PK_QG) * (PK_QR * PK_QG) +
              (r % PK_QR) * PK_QG + k % PK_QG] =
                (int8_t)(r % 4 == 0   ? -128
                         : r % 4 == 1 ? 127
                         : r % 4 == 2 ? (k % 2 ? -127 : 127)
                                      : (k % 2 ? -128 : 127));
    }
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t k = 0; k < n; k++)
            q[(size_t)t * n + k] = (int8_t)(t % 3 == 0 ? 127 : t % 3 == 1 ? -127 : (k % 2 ? -127 : 127));
        for (uint32_t k = 0; k < nb; k++)
            d[(size_t)t * nb + k] = (k + t) % 5 == 0 ? 0.0f : (float)((k + t) % 13 + 1) / 127.0f;
    }
    for (size_t i = 0; i < (size_t)T * os; i++) o[i] = want[i] = i % 2 ? -0.0f : 0.125f;
    pk_act a = { q, d, n, qb, T };
    kt->gemm_q8(o, os, w, ws, flags & 1 ? b : NULL, rows, &a, flags & 2);
    pk_kernels_ref.gemm_q8(want, os, w, ws, flags & 1 ? b : NULL, rows, &a, flags & 2);
    char what[96];
    snprintf(what, sizeof what, "n %u qb %u T %u bias %d add %d", n, qb, T, flags & 1, flags >> 1);
    const int bad = same(what, o, want, (size_t)T * os); /* the floats between columns too */
    free(want);
    unguard(gw);
    unguard(gq);
    unguard(gd);
    unguard(gs);
    unguard(gb);
    unguard(go);
    return bad;
}

static int run(void) {
    static const uint32_t ns[] = { 32, 64, 96, 128, 1024, 4096 };
    int bad = 0;
    for (size_t j = 0; j < sizeof ns / sizeof *ns; j++)
        for (uint32_t qb = 32; qb <= 64; qb *= 2)
            if (ns[j] % qb == 0)
                for (uint32_t T = 1; T <= 9; T++)
                    for (int f = 0; f < 4; f++) bad += q8_case(ns[j], qb, T, f);
    /* both sides of a column block's edge (192 KB of activations on
     * neon-dp), and the widest input */
    for (uint32_t qb = 32; qb <= 64; qb *= 2) {
        bad += q8_case(4096, qb, 49, 3) + q8_case(4096, qb, 50, 3);
        bad += q8_case(1024, qb, 193, 3) + q8_case(1024, qb, 194, 3);
        bad += q8_case(16384, qb, 13, 3) + q8_case(16384, qb, 14, 3);
    }
    /* a sum of zero is +0 rounding to nearest and may not be another way */
    static const int modes[] = { FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO };
    for (size_t i = 0; i < sizeof modes / sizeof *modes; i++) {
        if (fesetround(modes[i]))
            return bad + 1;
        bad += q8_case(128, 32, 5, 0) + q8_case(128, 64, 5, 3);
        bad += q8_case(128, 32, 37, 0) + q8_case(128, 64, 37, 3); /* the tiles of many columns */
    }
    if (fesetround(FE_TONEAREST))
        return bad + 1;
    /* no columns: nothing read, nothing written */
    pk_act none = { NULL, NULL, 32, 32, 0 };
    float out = -0.0f, want = out;
    kt->gemm_q8(&out, 16, NULL, NULL, NULL, 16, &none, 0);
    return bad + same("of no columns", &out, &want, 1);
}

int main(void) {
    int bad = 0;
    for (const pk_kernels *const *k = pk_kernels_built; *k && *k != &pk_kernels_ref; k++) {
        const char *lack = pk_os_cpu_lacks((*k)->needs);
        if (lack) {
            printf("q8check: %s: this CPU has no %s - skipped\n", (*k)->name, lack);
            continue;
        }
        kt = *k;
        g_cases = 0;
        const int b = run();
        printf("q8check: %s, %d cases at guard pages: %s\n", kt->name, g_cases,
               b ? "FAIL" : "every one ref's bits");
        bad += b;
    }
    return bad != 0;
}
