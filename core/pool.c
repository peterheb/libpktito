/* The thread pool (pool.h).
 *
 * Workers wait on a generation counter: spinning for a while, because a
 * layer of the encoder is a dozen jobs back to back and a wakeup through
 * the condition variable costs tens of microseconds each time, then parked
 * on it, so that an idle pool between recordings costs nothing. Built
 * without pthreads (-DPK_NO_THREADS) the pool is the calling thread alone.
 */
#include "pool.h"

#include <stdlib.h>

#ifdef PK_NO_THREADS

struct pk_pool { int n; };

pk_pool *pk_pool_new(int threads) {
    (void)threads;
    pk_pool *p = malloc(sizeof *p);
    if (p)
        p->n = 1;
    return p;
}
void pk_pool_free(pk_pool *p) { free(p); }
int pk_pool_threads(const pk_pool *p) { return p->n; }
void pk_pool_run(pk_pool *p, pk_job_fn fn, void *arg, int items) {
    (void)p;
    for (int i = 0; i < items; i++)
        fn(arg, i, 0);
}
void pk_pool_set_active(pk_pool *p, int k) { (void)p; (void)k; }
int pk_pool_active(const pk_pool *p) { return p->n; }

#else

#include "os.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>

/* Polls of the generation counter before a worker parks, meant to outlast
 * the gaps between a layer's jobs. A poll is the core's relax hint and two
 * loads, so their time is the core's, sixty-fold apart: 20000 last 6 us on
 * an M4 Pro, 40 on a Pi 5's A76, 169 on a Pi 4's A72, 214 on a Pi 3's A53
 * and 372 on a 9950X, whose PAUSE waits about 100 cycles (tests/spinbench,
 * best of 200, 2026-10-27). */
#define PK_SPIN 20000

static inline void relax(void) {
#if defined(__aarch64__)
    __asm__ __volatile__("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

/* A job's number and its width, the threads it runs on, travel in one word,
 * (number << 16) | width, so that one load tells a worker both that there
 * is a new job and whether it is in it. Read from two words, a worker slow
 * to look could pair one job's number with the next one's width, and count
 * itself out of a job it was in or into one it was not. */
#define PK_WIDTH_BITS 16

struct pk_pool {
    int n;
    pthread_t *th;
    pthread_mutex_t mu;
    pthread_cond_t cv;             /* a job, for the active workers asleep */
    pthread_cond_t idle;           /* the inactive sleep here (pk_pool_set_active) */
    _Atomic uint64_t gen;          /* the job: (number << PK_WIDTH_BITS) | width */
    _Atomic int next;              /* the next item to take */
    _Atomic int busy;              /* workers still in the job */
    _Atomic int active;            /* threads a job runs on, the caller one of them */
    /* read outside the lock by a worker that is in no job, so that nothing
     * else orders its read before pk_pool_free's write */
    _Atomic int quit;
    int items;
    pk_job_fn fn;
    void *arg;
};

typedef struct { pk_pool *p; int id; } pk_worker;

static void take(pk_pool *p, int id) {
    for (;;) {
        int i = atomic_fetch_add_explicit(&p->next, 1, memory_order_relaxed);
        if (i >= p->items)
            return;
        p->fn(p->arg, i, id);
    }
}

static void *worker(void *v) {
    pk_worker *w = v;
    pk_pool *p = w->p;
    int id = w->id;
    free(w);
    pk_os_unthrottle();
    uint64_t seen = 0;
    for (;;) {
        uint64_t g;
        int spins = 0;
        /* an inactive worker sleeps at once; an active one spins first */
        while ((g = atomic_load_explicit(&p->gen, memory_order_acquire)) == seen && spins < PK_SPIN
               && id < atomic_load_explicit(&p->active, memory_order_relaxed)) {
            relax();
            spins++;
        }
        if (g == seen) {
            pthread_mutex_lock(&p->mu);
            while (!atomic_load_explicit(&p->quit, memory_order_relaxed)) {
                if (id >= atomic_load_explicit(&p->active, memory_order_relaxed))
                    pthread_cond_wait(&p->idle, &p->mu);
                else if ((g = atomic_load_explicit(&p->gen, memory_order_acquire)) == seen)
                    pthread_cond_wait(&p->cv, &p->mu);
                else
                    break;
            }
            pthread_mutex_unlock(&p->mu);
        }
        if (atomic_load_explicit(&p->quit, memory_order_acquire))
            return NULL;
        seen = g;
        if (id < (int)(g & ((1u << PK_WIDTH_BITS) - 1))) {
            take(p, id);
            atomic_fetch_sub_explicit(&p->busy, 1, memory_order_release);
        }
    }
}

pk_pool *pk_pool_new(int threads) {
    pk_pool *p = calloc(1, sizeof *p);
    if (!p)
        return NULL;
    p->n = threads < 1 ? 1 : threads >= 1 << PK_WIDTH_BITS ? (1 << PK_WIDTH_BITS) - 1 : threads;
    atomic_init(&p->active, p->n);
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv, NULL);
    pthread_cond_init(&p->idle, NULL);
    p->th = calloc((size_t)p->n, sizeof *p->th);
    if (!p->th) {
        free(p);
        return NULL;
    }
    for (int i = 1; i < p->n; i++) {
        pk_worker *w = malloc(sizeof *w);
        if (!w || (w->p = p, w->id = i, pthread_create(&p->th[i], NULL, worker, w))) {
            free(w);
            p->n = i;                  /* the ones that started */
            pk_pool_free(p);
            return NULL;
        }
    }
    return p;
}

void pk_pool_free(pk_pool *p) {
    if (!p)
        return;
    pthread_mutex_lock(&p->mu);
    atomic_store_explicit(&p->quit, 1, memory_order_release);
    atomic_fetch_add_explicit(&p->gen, (uint64_t)1 << PK_WIDTH_BITS, memory_order_release);
    pthread_cond_broadcast(&p->cv);
    pthread_cond_broadcast(&p->idle);
    pthread_mutex_unlock(&p->mu);
    for (int i = 1; i < p->n; i++)
        pthread_join(p->th[i], NULL);
    pthread_cond_destroy(&p->idle);
    pthread_cond_destroy(&p->cv);
    pthread_mutex_destroy(&p->mu);
    free(p->th);
    free(p);
}

int pk_pool_threads(const pk_pool *p) { return p->n; }

void pk_pool_set_active(pk_pool *p, int k) {
    k = k < 1 ? 1 : k > p->n ? p->n : k;
    pthread_mutex_lock(&p->mu);
    int was = atomic_load_explicit(&p->active, memory_order_relaxed);
    atomic_store_explicit(&p->active, k, memory_order_relaxed);
    if (k > was)
        pthread_cond_broadcast(&p->idle);
    pthread_mutex_unlock(&p->mu);
}

int pk_pool_active(const pk_pool *p) { return atomic_load_explicit(&p->active, memory_order_relaxed); }

void pk_pool_run(pk_pool *p, pk_job_fn fn, void *arg, int items) {
    const int k = atomic_load_explicit(&p->active, memory_order_relaxed);
    if (k == 1 || items <= 1) {
        for (int i = 0; i < items; i++)
            fn(arg, i, 0);
        return;
    }
    p->fn = fn;
    p->arg = arg;
    p->items = items;
    atomic_store_explicit(&p->next, 0, memory_order_relaxed);
    atomic_store_explicit(&p->busy, k - 1, memory_order_relaxed);
    pthread_mutex_lock(&p->mu);
    uint64_t number = (atomic_load_explicit(&p->gen, memory_order_relaxed) >> PK_WIDTH_BITS) + 1;
    atomic_store_explicit(&p->gen, number << PK_WIDTH_BITS | (uint64_t)k, memory_order_release);
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
    take(p, 0);
    while (atomic_load_explicit(&p->busy, memory_order_acquire) > 0)
        relax();
}

#endif
