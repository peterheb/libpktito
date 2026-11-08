/* npu/ane.h through Core ML. The model's functions are the encoder whole
 * ("f3072") or in parts run one after another ("f3072p0", "f3072p1", ...:
 * tools/ane/encoder.py --parts), each taking one length or several (Core
 * ML's enumerated shapes, the lengths its mask input lists), and a function
 * is loaded once for all of its lengths. A length, once first needed, has
 * its inputs allocated once, as fp16 [1][1][n_mels][T] features and a
 * [1][1][1][T] mask, the program's own layout; its first part takes them,
 * each later one the part before's residual stream "x" [1][d_model][1][T/8]
 * and the mask, and the last gives "enc" [1][d_model][1][T/8]. A function
 * of several lengths takes the attention's position tables for the length
 * as well, "pe", "turn_cos" and "turn_sin" [1][1][d_model][T/8], made here
 * once a length as encoder.py's turn_tables makes them. */
#import <CoreML/CoreML.h>

#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ane.h"
#include "model.h"

#define PK_ANE_MAX_LENS 16
#define PK_ANE_MAX_PARTS 8
#define PK_ANE_MAX_FNS 16

typedef struct {
    void *name;                      /* NSString; CFBridgingRetain'd, as are the rest */
    void *model;                     /* MLModel, NULL until a length of it is first needed */
    int tables;                      /* takes the position tables */
} ane_fn;

typedef struct {
    uint32_t frames, nparts;
    uint8_t fn[PK_ANE_MAX_PARTS];    /* each part's function, 1 + its index in pk_ane's; 0 none */
    int loaded;
    void *mel[PK_ANE_SLOTS], *mask[PK_ANE_SLOTS];  /* the first part's inputs, a slot's each */
    void *pe, *cos, *sin;            /* the position tables, where its functions take them */
} ane_len;

struct pk_ane {
    uint32_t n_mels, d, n, nfn;
    uint32_t last;                   /* the length run last, 1 + its index in len; 0 none */
    float *freq;                     /* [d/2], the position encoding's frequencies */
    void *url;                       /* NSURL */
    ane_fn fn[PK_ANE_MAX_FNS];
    ane_len len[PK_ANE_MAX_LENS];    /* shortest first */
    pthread_mutex_t load;            /* a length's first load, once, whichever slot asks */
};

/* PK_ANE_TRACE=1: each step of opening and loading on stderr, with the
 * seconds since the first, for a device where one of them hangs */
static void trace(const char *fmt, ...) {
    static double t0;
    if (!getenv("PK_ANE_TRACE"))
        return;
    const double now = CFAbsoluteTimeGetCurrent();
    if (t0 == 0)
        t0 = now;
    char line[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    fprintf(stderr, "ane %7.2f s: %s\n", now - t0, line);
}

/* what, and Core ML's reason: its error's description and, under it, each
 * error it says caused that one, since Core ML can describe a failure to
 * compile for the Neural Engine as something else (on the M1 iPad, "the
 * functionName property must be nil unless the model type is ML Program") */
static void say(char *err, size_t len, NSString *what, NSError *e) {
    NSMutableString *s = [what mutableCopy];
    for (int depth = 0; e && depth < 4; depth++, e = e.userInfo[NSUnderlyingErrorKey])
        [s appendFormat:@"%@%@ (%@ %ld)", depth ? @"; under it, " : @": ", e.localizedDescription, e.domain,
                        (long)e.code];
    trace("error: %s", s.UTF8String);
    if (err && len)
        snprintf(err, len, "%s", s.UTF8String);
}

static MLMultiArray *array16(NSArray<NSNumber *> *shape, NSError **e) {
    return [[MLMultiArray alloc] initWithShape:shape dataType:MLMultiArrayDataTypeFloat16 error:e];
}

/* encoder.py's turn_tables for T feature frames, T/8 encoder frames, in
 * fp16 [1][1][d][T/8], into l: for each frequency w of the position
 * encoding, a pair of rows, pe(j) (sin j w, then cos j w) less its mean
 * over j, and the turn's (-cos i w, cos i w) and (sin i w, sin i w). In
 * double, rounded once. */
static int len_tables(pk_ane *a, ane_len *l, NSError **e) {
    const uint32_t D = a->d;
    uint32_t T3 = l->frames;
    for (int i = 0; i < 3; i++)
        T3 = (T3 + 1) / 2;
    NSArray<NSNumber *> *shape = @[ @1, @1, @(D), @(T3) ];
    MLMultiArray *t[3] = { array16(shape, e), array16(shape, e), array16(shape, e) };
    if (!t[0] || !t[1] || !t[2])
        return -1;
    for (int k = 0; k < 3; k++)
        [t[k] getMutableBytesWithHandler:^(void *bytes, NSInteger size, NSArray<NSNumber *> *strides) {
            (void)size;
            const size_t sd = strides[2].unsignedLongValue, st = strides[3].unsignedLongValue;
            for (uint32_t d = 0; d < D; d++) {
                const double w = a->freq[d / 2];
                double mean = 0;
                for (uint32_t j = 0; k == 0 && j < T3; j++)
                    mean += (d % 2 ? cos(j * w) : sin(j * w)) / T3;
                for (uint32_t j = 0; j < T3; j++)
                    ((_Float16 *)bytes)[d * sd + j * st] =
                        (_Float16)(k == 0 ? (d % 2 ? cos(j * w) : sin(j * w)) - mean
                                   : k == 1 ? (d % 2 ? cos(j * w) : -cos(j * w))
                                            : sin(j * w));
            }
        }];
    l->pe = (void *)CFBridgingRetain(t[0]);
    l->cos = (void *)CFBridgingRetain(t[1]);
    l->sin = (void *)CFBridgingRetain(t[2]);
    return 0;
}

/* l's parts loaded for the Neural Engine, those not loaded yet for another
 * length, and its inputs allocated; why not in err, if given */
static int len_load(pk_ane *a, ane_len *l, char *err, size_t errlen) {
    @autoreleasepool {
        NSError *e = nil;
        int tables = 0;
        for (uint32_t k = 0; k < l->nparts; k++) {
            ane_fn *f = &a->fn[l->fn[k] - 1];
            tables |= f->tables;
            if (f->model)
                continue;
            MLModelConfiguration *c = [MLModelConfiguration new];
            c.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
            NSString *name = (__bridge NSString *)f->name;
            if (name.length)
                c.functionName = name;
            trace("loading %s", name.UTF8String);
            MLModel *m = [MLModel modelWithContentsOfURL:(__bridge NSURL *)a->url configuration:c error:&e];
            trace("loaded %s: %s", name.UTF8String, m ? "ok" : e.localizedDescription.UTF8String);
            if (!m) {
                say(err, errlen, [NSString stringWithFormat:@"function %@ of the Core ML model can't be loaded", name], e);
                return -1;
            }
            f->model = (void *)CFBridgingRetain(m);
        }
        for (int k = 0; k < PK_ANE_SLOTS; k++) {
            MLMultiArray *mel = array16(@[ @1, @1, @(a->n_mels), @(l->frames) ], &e);
            MLMultiArray *mask = array16(@[ @1, @1, @1, @(l->frames) ], &e);
            if (!mel || !mask) {
                say(err, errlen, @"out of memory", e);
                return -1;
            }
            l->mel[k] = (void *)CFBridgingRetain(mel);
            l->mask[k] = (void *)CFBridgingRetain(mask);
        }
        if (tables && len_tables(a, l, &e)) {
            say(err, errlen, @"out of memory", e);
            return -1;
        }
        l->loaded = 1;
        return 0;
    }
}

/* name's length and part: "f3072" is part 0 of a length of one, "f3072p1"
 * part 1 */
static void name_part(NSString *name, uint32_t *part) {
    unsigned T = 0, k = 0;
    *part = sscanf(name.UTF8String, "f%up%u", &T, &k) == 2 ? k : 0;
}

pk_ane *pk_ane_open(const char *path, const pk_model *m, char *err, size_t errlen) {
    const uint32_t n_mels = m->n_mels, d_model = m->d_model;
    const char *source = pk_model_meta(m, "source_sha256");
    @autoreleasepool {
        NSURL *url = [NSURL fileURLWithPath:@(path)];
        NSError *e = nil;
        trace("opening %s", path);
        MLModelAsset *asset = [MLModelAsset modelAssetWithURL:url error:&e];
        trace("asset %s", asset ? "made" : e.localizedDescription.UTF8String);
        if (!asset) {
            say(err, errlen, @"the Core ML model can't be read", e);
            return NULL;
        }
        __block NSArray<NSString *> *names = nil;
        __block NSError *ne = nil;
        dispatch_semaphore_t got = dispatch_semaphore_create(0);
        [asset functionNamesWithCompletionHandler:^(NSArray<NSString *> *n, NSError *x) {
            names = n;
            ne = x;
            dispatch_semaphore_signal(got);
        }];
        dispatch_semaphore_wait(got, DISPATCH_TIME_FOREVER);
        trace("functions: %lu", (unsigned long)names.count);
        if (!names) {
            say(err, errlen, @"the Core ML model's functions can't be listed", ne);
            return NULL;
        }
        if (!names.count)
            names = @[ @"" ];        /* a model of one function, unnamed */
        pk_ane *a = calloc(1, sizeof *a);
        if (!a) {
            say(err, errlen, @"out of memory", nil);
            return NULL;
        }
        pthread_mutex_init(&a->load, NULL);
        a->n_mels = n_mels;
        a->d = d_model;
        a->url = (void *)CFBridgingRetain(url);
        a->freq = malloc(d_model / 2 * sizeof *a->freq);
        if (!a->freq) {
            say(err, errlen, @"out of memory", nil);
            pk_ane_free(a);
            return NULL;
        }
        memcpy(a->freq, m->pos_freq, d_model / 2 * sizeof *a->freq);
        for (NSString *name in names) {
            __block MLModelDescription *desc = nil;
            void (^done)(MLModelDescription *, NSError *) = ^(MLModelDescription *d, NSError *x) {
                desc = d;
                ne = x;
                dispatch_semaphore_signal(got);
            };
            if (name.length)
                [asset modelDescriptionOfFunctionNamed:name completionHandler:done];
            else
                [asset modelDescriptionWithCompletionHandler:done];
            dispatch_semaphore_wait(got, DISPATCH_TIME_FOREVER);
            trace("described %s", name.UTF8String);
            NSDictionary<NSString *, MLFeatureDescription *> *in = desc.inputDescriptionsByName;
            NSArray<NSNumber *> *mask = in[@"mask"].multiArrayConstraint.shape;
            uint32_t part;
            name_part(name, &part);
            NSArray<NSNumber *> *first = in[@"mel"].multiArrayConstraint.shape;
            NSArray<NSNumber *> *later = in[@"x"].multiArrayConstraint.shape;
            const BOOL fits = mask.count == 4 && part < PK_ANE_MAX_PARTS &&
                (part == 0 ? first.count == 4 && first[2].unsignedIntValue == n_mels
                           : later.count == 4 && later[1].unsignedIntValue == d_model);
            if (!fits) {
                say(err, errlen, [NSString stringWithFormat:@"function %@ of the Core ML model isn't this model's encoder", name], ne);
                pk_ane_free(a);
                return NULL;
            }
            /* made from the model file's own checkpoint, whose SHA-256
             * encoder.py writes as the converter does: a v2 file must not
             * take a v3 encoder, whose numbers would mean nothing to it */
            NSString *made = desc.metadata[MLModelCreatorDefinedKey][@"source_sha256"];
            if (!source || ![made isEqualToString:@(source)]) {
                say(err, errlen, [NSString stringWithFormat:@"the Core ML model was made from another checkpoint "
                                                            @"than the model file's (%@; the file's %s)",
                                                            made ? made : @"none named", source ? source : "none named"], nil);
                pk_ane_free(a);
                return NULL;
            }
            /* its lengths: the mask's enumerated shapes, or its one shape */
            MLMultiArrayShapeConstraint *sc = in[@"mask"].multiArrayConstraint.shapeConstraint;
            NSArray<NSArray<NSNumber *> *> *shapes =
                sc.type == MLMultiArrayShapeConstraintTypeEnumerated ? sc.enumeratedShapes : @[ mask ];
            if (a->nfn == PK_ANE_MAX_FNS) {
                say(err, errlen, @"the Core ML model has more functions than pktito takes", nil);
                pk_ane_free(a);
                return NULL;
            }
            a->fn[a->nfn].tables = in[@"pe"] != nil;
            a->fn[a->nfn++].name = (void *)CFBridgingRetain(name);
            for (NSArray<NSNumber *> *shape in shapes) {
                if (shape.count != 4) {
                    say(err, errlen, [NSString stringWithFormat:@"function %@ of the Core ML model isn't this model's encoder", name], nil);
                    pk_ane_free(a);
                    return NULL;
                }
                const uint32_t T = shape[3].unsignedIntValue;
                uint32_t i = 0;
                while (i < a->n && a->len[i].frames < T)
                    i++;
                if (i == a->n || a->len[i].frames != T) {
                    if (a->n == PK_ANE_MAX_LENS) {
                        say(err, errlen, @"the Core ML model has more lengths than pktito takes", nil);
                        pk_ane_free(a);
                        return NULL;
                    }
                    for (uint32_t j = a->n++; j > i; j--)
                        a->len[j] = a->len[j - 1];
                    a->len[i] = (ane_len){ .frames = T };
                }
                ane_len *l = &a->len[i];
                if (l->fn[part]) {
                    say(err, errlen, [NSString stringWithFormat:@"function %@ of the Core ML model repeats a part", name], nil);
                    pk_ane_free(a);
                    return NULL;
                }
                l->fn[part] = (uint8_t)a->nfn;
                l->nparts = part + 1 > l->nparts ? part + 1 : l->nparts;
            }
        }
        for (uint32_t i = 0; i < a->n; i++)
            for (uint32_t k = 0; k < a->len[i].nparts; k++)
                if (!a->len[i].fn[k]) {
                    say(err, errlen, [NSString stringWithFormat:@"the Core ML model's %u frames lack part %u",
                                                                a->len[i].frames, k], nil);
                    pk_ane_free(a);
                    return NULL;
                }
        return a;
    }
}

/* feat [T0][n_mels] through length l, padded to it, in slot k's inputs:
 * x [Te][d], Te encoder frames, which it returns; 0 if Core ML failed, and
 * why in err if given. Calls in different slots run at once: Core ML's
 * predictions on one MLModel may, each with inputs and outputs of its own. */
static uint32_t len_run(pk_ane *a, ane_len *l, int k_slot, const float *feat, uint32_t T0, float *x,
                        char *err, size_t errlen) {
    pthread_mutex_lock(&a->load);
    const int bad = !l->loaded && len_load(a, l, err, errlen);
    pthread_mutex_unlock(&a->load);
    if (bad)
        return 0;
    /* calls at once each set it, and a model of one length (encoder.py's
     * default, 3072 frames) chooses that one whatever it is */
    a->last = (uint32_t)(l - a->len) + 1;
    const uint32_t T = l->frames, M = a->n_mels, D = a->d;
    uint32_t Te = T0;
    for (int i = 0; i < 3; i++)
        Te = (Te + 1) / 2;
    @autoreleasepool {
        MLMultiArray *mel = (__bridge MLMultiArray *)l->mel[k_slot], *mask = (__bridge MLMultiArray *)l->mask[k_slot];
        /* the features transposed to [n_mels][T], zero past T0 */
        [mel getMutableBytesWithHandler:^(void *bytes, NSInteger size, NSArray<NSNumber *> *strides) {
            (void)size;
            _Float16 *p = bytes;
            const size_t sm = strides[2].unsignedLongValue, st = strides[3].unsignedLongValue;
            for (uint32_t m = 0; m < M; m++)
                for (uint32_t t = 0; t < T; t++)
                    p[m * sm + t * st] = t < T0 ? (_Float16)feat[(size_t)t * M + m] : (_Float16)0;
        }];
        [mask getMutableBytesWithHandler:^(void *bytes, NSInteger size, NSArray<NSNumber *> *strides) {
            (void)size;
            _Float16 *p = bytes;
            const size_t st = strides[3].unsignedLongValue;
            for (uint32_t t = 0; t < T; t++)
                p[t * st] = t < T0 ? (_Float16)1 : (_Float16)0;
        }];
        NSError *e = nil;
        MLFeatureValue *maskv = [MLFeatureValue featureValueWithMultiArray:mask];
        NSDictionary *tables = l->pe ? @{
            @"pe": [MLFeatureValue featureValueWithMultiArray:(__bridge MLMultiArray *)l->pe],
            @"turn_cos": [MLFeatureValue featureValueWithMultiArray:(__bridge MLMultiArray *)l->cos],
            @"turn_sin": [MLFeatureValue featureValueWithMultiArray:(__bridge MLMultiArray *)l->sin],
        } : @{};
        NSDictionary *in = @{ @"mel": [MLFeatureValue featureValueWithMultiArray:mel], @"mask": maskv };
        MLMultiArray *enc = nil;
        for (uint32_t k = 0; k < l->nparts; k++) {
            if (a->fn[l->fn[k] - 1].tables) {
                NSMutableDictionary *all = [in mutableCopy];
                [all addEntriesFromDictionary:tables];
                in = all;
            }
            MLDictionaryFeatureProvider *p = [[MLDictionaryFeatureProvider alloc] initWithDictionary:in error:&e];
            trace("predicting part %u of %u frames, %u of them the recording's", k, T, T0);
            MLModel *m = (__bridge MLModel *)a->fn[l->fn[k] - 1].model;
            id<MLFeatureProvider> out = p ? [m predictionFromFeatures:p error:&e] : nil;
            trace("predicted: %s", out ? "ok" : e.localizedDescription.UTF8String);
            if (k + 1 < l->nparts) {
                MLMultiArray *xs = [out featureValueForName:@"x"].multiArrayValue;
                if (!xs) {
                    say(err, errlen, @"the Core ML model's prediction failed", e);
                    return 0;
                }
                in = @{ @"x": [MLFeatureValue featureValueWithMultiArray:xs], @"mask": maskv };
            } else {
                enc = [out featureValueForName:@"enc"].multiArrayValue;
            }
        }
        if (!enc || enc.shape.count != 4 || enc.shape[1].unsignedIntValue != D || enc.shape[3].unsignedIntValue < Te) {
            say(err, errlen, enc ? @"the Core ML model's output isn't this model's encoder's"
                                 : @"the Core ML model's prediction failed", enc ? nil : e);
            return 0;
        }
        const size_t sd = enc.strides[1].unsignedLongValue, st = enc.strides[3].unsignedLongValue;
        const BOOL half = enc.dataType == MLMultiArrayDataTypeFloat16;
        [enc getBytesWithHandler:^(const void *bytes, NSInteger size) {
            (void)size;
            for (uint32_t t = 0; t < Te; t++)
                for (uint32_t d = 0; d < D; d++)
                    x[(size_t)t * D + d] = half ? (float)((const _Float16 *)bytes)[d * sd + t * st]
                                                : ((const float *)bytes)[d * sd + t * st];
        }];
    }
    return Te;
}

int pk_ane_warm(pk_ane *a, char *err, size_t errlen) {
    /* each length loaded and run once, on silence: Core ML compiles a
     * function for this device's Neural Engine when it loads it, and a
     * prediction is the rest of what can fail, so that a device that can't
     * run the encoder is known here rather than at the first recording */
    const uint32_t T = pk_ane_frames(a);
    float *feat = calloc((size_t)T * a->n_mels, sizeof *feat);
    float *x = malloc((size_t)(T + 7) / 8 * a->d * sizeof *x);
    int rc = feat && x ? 0 : -1;
    if (rc)
        say(err, errlen, @"out of memory", nil);
    for (uint32_t i = 0; i < a->n && rc == 0; i++)
        if (!len_run(a, &a->len[i], PK_ANE_SLOTS - 1, feat, a->len[i].frames, x, err, errlen))
            rc = -1;
    free(feat);
    free(x);
    return rc;
}

void pk_ane_free(pk_ane *a) {
    if (!a)
        return;
    for (uint32_t i = 0; i < a->nfn; i++) {
        if (a->fn[i].name)
            CFBridgingRelease(a->fn[i].name);
        if (a->fn[i].model)
            CFBridgingRelease(a->fn[i].model);
    }
    for (uint32_t i = 0; i < a->n; i++) {
        ane_len *l = &a->len[i];
        for (int k = 0; k < PK_ANE_SLOTS; k++) {
            if (l->mel[k])
                CFBridgingRelease(l->mel[k]);
            if (l->mask[k])
                CFBridgingRelease(l->mask[k]);
        }
        if (l->pe) {
            CFBridgingRelease(l->pe);
            CFBridgingRelease(l->cos);
            CFBridgingRelease(l->sin);
        }
    }
    if (a->url)
        CFBridgingRelease(a->url);
    pthread_mutex_destroy(&a->load);
    free(a->freq);
    free(a);
}

uint32_t pk_ane_frames(const pk_ane *a) { return a->n ? a->len[a->n - 1].frames : 0; }

uint32_t pk_ane_encode(pk_ane *a, const float *feat, uint32_t T0, float *x) {
    return pk_ane_encode_in(a, PK_ANE_SLOTS - 1, feat, T0, x);
}

uint32_t pk_ane_encode_in(pk_ane *a, int slot, const float *feat, uint32_t T0, float *x) {
    /* the shortest length that holds T0; but the one run last while it
     * holds T0 too and is less than 1536 frames longer: running a length
     * after another costs the Neural Engine about 40 ms more, where 1536
     * frames take 32 (the int8 encoder in two parts on the M4 Pro; on the
     * M1 iPad 100 ms, CLIMB.md 2026-10-31) */
    ane_len *l = NULL;
    for (uint32_t i = 0; i < a->n && !l; i++)
        if (a->len[i].frames >= T0)
            l = &a->len[i];
    if (l && a->last) {
        ane_len *p = &a->len[a->last - 1];
        if (p->frames >= T0 && p->frames - l->frames < 1536)
            l = p;
    }
    if (!l || !T0)
        return 0;
    return len_run(a, l, slot, feat, T0, x, NULL, 0);
}
