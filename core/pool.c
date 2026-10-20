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

#else

#include <pthread.h>
#include <stdatomic.h>

/* Polls of the generation counter before a worker parks: about 50-200 us,
 * depending on the core, which outlasts the gaps between a layer's jobs. */
#define PK_SPIN 20000

static inline void relax(void) {
#if defined(__aarch64__)
    __asm__ __volatile__("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

struct pk_pool {
    int n;
    pthread_t *th;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    _Atomic unsigned gen;          /* a new job when this changes */
    _Atomic int next;              /* the next item to take */
    _Atomic int busy;              /* workers still in the job */
    int items, quit;
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
    unsigned seen = 0;
    for (;;) {
        unsigned g;
        int spins = 0;
        while ((g = atomic_load_explicit(&p->gen, memory_order_acquire)) == seen && spins < PK_SPIN) {
            relax();
            spins++;
        }
        if (g == seen) {
            pthread_mutex_lock(&p->mu);
            while ((g = atomic_load_explicit(&p->gen, memory_order_acquire)) == seen && !p->quit)
                pthread_cond_wait(&p->cv, &p->mu);
            pthread_mutex_unlock(&p->mu);
        }
        if (p->quit)
            return NULL;
        seen = g;
        take(p, id);
        atomic_fetch_sub_explicit(&p->busy, 1, memory_order_release);
    }
}

pk_pool *pk_pool_new(int threads) {
    pk_pool *p = calloc(1, sizeof *p);
    if (!p)
        return NULL;
    p->n = threads < 1 ? 1 : threads;
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv, NULL);
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
    p->quit = 1;
    atomic_fetch_add_explicit(&p->gen, 1, memory_order_release);
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
    for (int i = 1; i < p->n; i++)
        pthread_join(p->th[i], NULL);
    pthread_cond_destroy(&p->cv);
    pthread_mutex_destroy(&p->mu);
    free(p->th);
    free(p);
}

int pk_pool_threads(const pk_pool *p) { return p->n; }

void pk_pool_run(pk_pool *p, pk_job_fn fn, void *arg, int items) {
    if (p->n == 1 || items <= 1) {
        for (int i = 0; i < items; i++)
            fn(arg, i, 0);
        return;
    }
    p->fn = fn;
    p->arg = arg;
    p->items = items;
    atomic_store_explicit(&p->next, 0, memory_order_relaxed);
    atomic_store_explicit(&p->busy, p->n - 1, memory_order_relaxed);
    pthread_mutex_lock(&p->mu);
    atomic_fetch_add_explicit(&p->gen, 1, memory_order_release);
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
    take(p, 0);
    while (atomic_load_explicit(&p->busy, memory_order_acquire) > 0)
        relax();
}

#endif
