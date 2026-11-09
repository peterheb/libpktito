/* core/editions.h: which edition of a model a device runs, as pktito-app's
 * editions.rs has it, and this machine as that rule sees it */
#include "editions.h"
#include "pktito.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __APPLE__
#include <TargetConditionals.h>
#include <sys/sysctl.h>
#endif
#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif

/* a chip's five numbers in QNN's words (soc,arch,vtcm,dlbc,signed): 1, or
 * 0 for anything else */
static int numbers(const char *chip, unsigned long long n[5]) {
    for (int i = 0; i < 5; i++) {
        char *end;
        if (*chip < '0' || *chip > '9')
            return 0;
        n[i] = strtoull(chip, &end, 10);
        if (*end != (i < 4 ? ',' : '\0'))
            return 0;
        chip = end + 1;
    }
    return 1;
}

static int on(const char *const *systems, const char *system) {
    for (; systems && *systems; systems++)
        if (!strcmp(*systems, system))
            return 1;
    return 0;
}

static int at_least(const int os[2], const int floor[2]) {
    return os[0] > floor[0] || (os[0] == floor[0] && os[1] >= floor[1]);
}

int ed_fit(const ed_edition *e, const ed_device *d) {
    switch (e->kind) {
    case ED_CPU:
        return 1;
    case ED_COREML:
        if (!strcmp(d->system, "macos"))
            return at_least(d->os, e->macos) ? 3 : 0;
        if (!strcmp(d->system, "ios"))
            return at_least(d->os, e->ios) ? 3 : 0;
        return 0;
    case ED_QNN: {
        unsigned long long mine[5], its[5];
        if (!d->qnn[0] || !on(e->systems, d->system))
            return 0;
        if (!strcmp(d->qnn, e->chip))
            return 3;
        /* another SoC of the same Hexagon, whose graph needs no more VTCM
         * than this one has: snappy's QNN on Windows names no SoC
         * (2147483647,73,8,1,1) and runs the X1E80100's graph */
        return numbers(d->qnn, mine) && numbers(e->chip, its) && mine[1] == its[1] && its[2] <= mine[2] ? 2 : 0;
    }
    }
    return 0;
}

int ed_version_cmp(const char *a, const char *b) {
    while (*a || *b) {
        char *ea, *eb;
        const unsigned long x = strtoul(a, &ea, 10), y = strtoul(b, &eb, 10);
        if (x != y)
            return x < y ? -1 : 1;
        a = *ea == '.' ? ea + 1 : ea + strlen(ea);   /* "0.9" is "0.9.0" */
        b = *eb == '.' ? eb + 1 : eb + strlen(eb);
    }
    return 0;
}

int ed_may(const ed_model *m, const ed_edition *e, int named) {
    if (e->needs && ed_version_cmp(e->needs, PKTITO_VERSION) > 0)
        return 0;
    return e->status == ED_STABLE && (m->status == ED_STABLE || named);
}

const ed_edition *ed_choose(const ed_model *m, const ed_device *d, int (*can_run)(const ed_edition *), int named) {
    const ed_edition *top = NULL, *cpu = NULL;
    int best = 0;
    for (int i = 0; i < m->n_editions; i++) {
        const ed_edition *e = &m->editions[i];
        if (!ed_may(m, e, named))
            continue;
        if (e->kind == ED_CPU)
            cpu = e;
        if (can_run && !can_run(e))
            continue;
        const int f = ed_fit(e, d);
        if (f > best)
            best = f, top = e;
    }
    return top ? top : cpu;
}

const ed_edition *ed_cpu(const ed_model *m) {
    for (int i = 0; i < m->n_editions; i++)
        if (m->editions[i].kind == ED_CPU)
            return &m->editions[i];
    return NULL;
}

const ed_model *ed_find(const char *name) {
    for (int i = 0; i < ed_n_models; i++)
        if (!strcmp(ed_catalog[i].name, name))
            return &ed_catalog[i];
    return NULL;
}

void ed_this_device(ed_device *d, int (*qnn_chip)(char *chip, unsigned cap)) {
    memset(d, 0, sizeof *d);
#if defined(__APPLE__) && TARGET_OS_IPHONE
    const char *system = "ios";
#elif defined(__APPLE__)
    const char *system = "macos";
#elif defined(__ANDROID__)
    const char *system = "android";
#elif defined(_WIN32)
    const char *system = "windows";
#elif defined(__linux__)
    const char *system = "linux";
#else
    const char *system = "other";
#endif
    snprintf(d->system, sizeof d->system, "%s", system);
#ifdef __APPLE__
    /* "26.4.1": the major and minor are what Core ML's floors compare */
    char v[32];
    size_t n = sizeof v;
    if (!sysctlbyname("kern.osproductversion", v, &n, NULL, 0))
        sscanf(v, "%d.%d", &d->os[0], &d->os[1]);
#elif defined(__ANDROID__)
    char v[PROP_VALUE_MAX];
    if (__system_property_get("ro.build.version.sdk", v) > 0)
        d->os[0] = atoi(v);
#endif
    if (qnn_chip && qnn_chip(d->qnn, sizeof d->qnn))
        d->qnn[0] = 0;
}
