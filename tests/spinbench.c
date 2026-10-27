/* How long a pool worker spins before it parks: PK_SPIN polls of the
 * generation counter and the active count, each after the core's relax
 * hint, as core/pool.c's worker makes them (it is included, for its relax
 * and PK_SPIN), timed alone on one thread. A poll's time is the core's: the
 * best and the median of 200, for the comment on PK_SPIN. */
#include "pool.c"
#include "os.h"

#include <stdio.h>

static int cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* the two words a worker polls, where another thread could write them, as
 * the pool's are: words no other code can reach, a compiler may read once */
static struct { _Atomic uint64_t gen; _Atomic int active; } *volatile shared;

int main(void) {
    shared = calloc(1, sizeof *shared);
    if (!shared)
        return 1;
    atomic_store(&shared->active, 2);
    double us[200];
    for (int rep = 0; rep < 200; rep++) {
        const int id = 1;
        int spins = 0;
        double t0 = pk_os_now();
        while (atomic_load_explicit(&shared->gen, memory_order_acquire) == 0 && spins < PK_SPIN
               && id < atomic_load_explicit(&shared->active, memory_order_relaxed)) {
            relax();
            spins++;
        }
        us[rep] = (pk_os_now() - t0) * 1e6;
    }
    qsort(us, 200, sizeof *us, cmp);
    printf("PK_SPIN, %d polls: best %.1f us, median %.1f us (%.2f ns a poll)\n", PK_SPIN, us[0], us[100],
           us[0] * 1e3 / PK_SPIN);
    return 0;
}
