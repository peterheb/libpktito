/* A pool of threads that run one job at a time over a range of items.
 *
 * A job is fn(arg, item, thread) for every item in [0, items), and the
 * items are dealt one at a time from a shared counter, so a fast core takes
 * more of them than a slow one (an i9-12900HK's P-cores and E-cores, or a
 * Pi whose other cores are busy) and none waits for a share it was given
 * in advance. The thread that calls pk_pool_run is one of the workers, and
 * it returns when every item is done. `thread` is in [0, threads), for
 * indexing per-thread scratch.
 *
 * An item's result must not depend on which thread ran it, and nothing in
 * pktito's does: every output element is computed whole by one item, so a
 * transcript is the same bits at one thread and at twenty.
 */
#ifndef PK_POOL_H
#define PK_POOL_H

typedef void (*pk_job_fn)(void *arg, int item, int thread);
typedef struct pk_pool pk_pool;

/* threads in all, the caller included; NULL if a thread cannot start */
pk_pool *pk_pool_new(int threads);
void     pk_pool_free(pk_pool *p);
int      pk_pool_threads(const pk_pool *p);
void     pk_pool_run(pk_pool *p, pk_job_fn fn, void *arg, int items);

/* Jobs run on the first k threads, the caller one of them, until this is
 * called again (k is held to [1, threads]); the others sleep, neither woken
 * by a job nor waited for. For a run of jobs too small for every thread:
 * pk_decode's products. Called between jobs, by the thread that runs them.
 * pk_pool_active says how many a job runs on. */
void     pk_pool_set_active(pk_pool *p, int k);
int      pk_pool_active(const pk_pool *p);

#endif /* PK_POOL_H */
