/* npu/qnn.h through QNN's C API. The graph is core/encoder.c's arithmetic
 * written as QNN ops, with the HTP's layout: the subsampling's images
 * [1][time][frequency][channels], as encoder.c's rows already are, and
 * then the stream [T][d_model]. Each matrix is a FullyConnected op on the
 * model's own int8 weights, a scale an output row as the .pkf has them,
 * the rows unpacked from the file's panels (the NPU takes its activations
 * in fp16 against them); everything else a float tensor, which the HTP
 * holds and computes in fp16.
 *
 * The HTP runs fixed shapes, so the graph takes one length, padded with
 * zeros, and four masks for the input's true length (made in
 * pk_qnn_encode): as tools/ane/encoder.py does for the Neural Engine, the
 * subsampling zeroes the frames past each stage's end before the next
 * stage's convolution sees them, the attention gives the keys past the end
 * no weight, and the convolution module zeroes them before its depthwise
 * convolution, so every real frame is what the unpadded input gives (up
 * to fp16). And as there:
 *
 *  - the residual stream is carried divided by PK_QNN_SCALE, which fp16
 *    holds where the stream's own size (about 300 inside a layer, squared
 *    in a norm's variance) it may not: each block's last matrix and each
 *    layer's output norm scaled to match, and each norm's epsilon by the
 *    scale's square, which leaves every norm's output what it was;
 *  - the attention's position term is the query turned rather than the
 *    scores shifted: in each (sin, cos) pair of frequency w, the encoding
 *    of position i - j is that of j turned through i w, so
 *        (q_i + v) . W_pos pe(i - j) = qr_i . pe(j)
 *    with qr_i = R(i)^T W_pos^T (q_i + v) (tools/ane/encoder.py's
 *    docstring has the algebra), and pe(j) less its mean over j, which the
 *    softmax doesn't see and fp16 holds nearer. Here a head at a time: one
 *    product of the head's q_i + v with W_pos^T and with it again with each
 *    (sin, cos) pair of columns swapped, [T][2 d_model]; the turn's tables
 *    multiplied in; one product with pe twice over, which adds the halves
 *    as it goes. For the eight heads together, as the Neural Engine has
 *    it, the same arithmetic took half the NPU's time, and so a head at a
 *    time saves a third of it (tools/qnn/CLIMB.md, 2026-11-02);
 *  - the attention's 1/sqrt(dk) is folded into K's scales and into pe.
 */
#include <assert.h>              /* the HTP headers' static_assert */
#ifdef _WIN32
#include <wchar.h>
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpGraph.h"
#include "HTP/QnnHtpPerfInfrastructure.h"
#include "QnnInterface.h"
#include "QnnOpDef.h"

#include "model.h"
#include "os.h"
#include "pkf.h"
#include "qnn.h"

#define PK_QNN_SCALE 8.0f        /* tools/ane/encoder.py's --scale: a power of two, exact */
#define PK_QNN_FRAMES 3072       /* 30.72 s */

typedef Qnn_ErrorHandle_t (*get_providers_fn)(const QnnInterface_t ***, uint32_t *);

struct pk_qnn {
    void *lib;
    QNN_INTERFACE_VER_TYPE q;
    Qnn_LogHandle_t log;
    Qnn_BackendHandle_t backend;
    Qnn_DeviceHandle_t device;
    Qnn_ContextHandle_t context;
    Qnn_GraphHandle_t graph;
    uint32_t n_mels, d, T0, T1, T2, T3;
    uint32_t layers;             /* the encoder's layers it runs: all, but for checking */
    int other;                   /* compiled for another chip (PKTITO_QNN_TARGET): kept, not run */
    char chip[32];               /* the NPU its graph is for, in PKTITO_QNN_TARGET's words; "" where QNN names none */
    Qnn_Tensor_t in[5], out;     /* mel, the three stages' masks, the keys' bias; x */
    float *buf[5], *x;           /* their data */
    QnnHtpDevice_PerfInfrastructure_t *perf; /* the HTP's clocks, voted for (vote) */
    uint32_t power;              /* the vote's power config, 0 for none */
    char names[6][16];           /* in's and out's names and shapes, their own */
    uint32_t dims[6][4];
};

/* in[0..4] then out: the graph's inputs and output */
static Qnn_Tensor_t *io(pk_qnn *e, int k) { return k < 5 ? &e->in[k] : &e->out; }

/* io(e, k)'s rank, and its shape into d, the dims past the rank zero: the
 * features [1][T0][n_mels][1], padded with zeros; the subsampling's two
 * masks, [1][T1][1][1] and [1][T2][1][1]; the convolution module's
 * [T3][1]; the keys' bias [1][1][T3]; and out, the stream [T3][d_model].
 * make_graph makes the inputs so, pk_qnn_open's buffers are these sizes,
 * and a compiled graph's file is held to them (check_head) */
static uint32_t io_shape(const pk_qnn *e, int k, uint32_t d[4]) {
    const uint32_t s[6][4] = {
        { 1, e->T0, e->n_mels, 1 }, { 1, e->T1, 1, 1 }, { 1, e->T2, 1, 1 },
        { e->T3, 1, 0, 0 },         { 1, 1, e->T3, 0 }, { e->T3, e->d, 0, 0 },
    };
    static const uint32_t rank[6] = { 4, 4, 4, 2, 3, 2 };
    memcpy(d, s[k], sizeof s[k]);
    return rank[k];
}

/* e's lengths for model m, taking frames feature frames (0, PK_QNN_FRAMES),
 * and its layers, every one */
static void lengths(pk_qnn *e, const pk_model *m, uint32_t frames) {
    e->n_mels = m->n_mels;
    e->d = m->d_model;
    e->T0 = frames ? frames : PK_QNN_FRAMES;
    e->T1 = (e->T0 + 1) / 2, e->T2 = (e->T1 + 1) / 2, e->T3 = (e->T2 + 1) / 2;
    e->layers = m->n_layers;
}

/* the inputs' and output's names and shapes copied into e, from the
 * builder's memory, which goes when the graph is compiled */
static void own_io(pk_qnn *e) {
    for (int k = 0; k < 6; k++) {
        Qnn_Tensor_t *t = io(e, k);
        snprintf(e->names[k], sizeof e->names[k], "%s", t->v1.name ? t->v1.name : "");
        memcpy(e->dims[k], t->v1.dimensions, t->v1.rank * sizeof(uint32_t));
        t->v1.name = e->names[k];
        t->v1.dimensions = e->dims[k];
    }
}

/* ------------------------------------------------------------ logging --- */

/* QNN's messages: its errors since the call err reports on, joined, kept
 * for err - the first is often the cause and the last QNN's summary
 * ("Request feature arch with value 79 unsupported", then "Failed to
 * create context from binary", for a V79's graph on a V73) - and every
 * message at PKTITO_QNN_LOG's level or below on stderr (1 errors, 2
 * warnings, 3 info, 4 verbose, 5 debug). A thread's own: two
 * transcribers' QNN calls, or the chip asked beside a transcription, each
 * keep theirs */
static _Thread_local char last_error[512];

static void log_cb(const char *fmt, QnnLog_Level_t level, uint64_t ts, va_list ap) {
    (void)ts;
    char line[512];
    vsnprintf(line, sizeof line, fmt, ap);
    if (level == QNN_LOG_LEVEL_ERROR) {
        size_t n = strlen(last_error), k = strcspn(line, "\n");
        if (n && n + 2 < sizeof last_error) {
            memcpy(last_error + n, "; ", 2);
            n += 2;
        }
        if (k > sizeof last_error - 1 - n)
            k = sizeof last_error - 1 - n;
        memcpy(last_error + n, line, k);
        last_error[n + k] = 0;
    }
    const char *v = getenv("PKTITO_QNN_LOG");
    if (v && (int)level <= atoi(v))
        fprintf(stderr, "qnn: %s%s", line, line[0] && line[strlen(line) - 1] == '\n' ? "" : "\n");
}

static int fail(char *err, size_t errlen, const char *what, Qnn_ErrorHandle_t e) {
    if (err && errlen)
        snprintf(err, errlen, "%s failed (QNN error %lu)%s%s", what, (unsigned long)(e & 0xffff),
                 last_error[0] ? ": " : "", last_error);
    return -1;
}

/* ---------------------------------------------------------- profiling --- */

/* PKTITO_QNN_PROFILE=1: QNN's detailed profile of compiling the graph and
 * of each run, an event a line on stderr - "qnn-prof", its depth among the
 * events, its type, unit (1 us, 3 cycles), value and name - the ops named
 * for their block and type (op_name). (The HTP's own linting level crashed
 * the NPU's process on this graph: tools/qnn/CLIMB.md, 2026-11-02.)
 * PKTITO_QNN_PROFILE=optrace: the same, and the compile writes the graph's
 * schematic, encoder_schematic.bin in the working directory, from which
 * Qualcomm's qnn-profile-viewer lays out a chunk's ops in time, the chunk
 * run with qnn-net-run's optrace (tools/qnn/CLIMB.md, 2026-11-03) */
static Qnn_ProfileHandle_t profile(const pk_qnn *e, int compile) {
    const char *v = getenv("PKTITO_QNN_PROFILE");
    Qnn_ProfileHandle_t p = NULL;
    if (v && e->q.profileCreate(e->backend, QNN_PROFILE_LEVEL_DETAILED, &p))
        p = NULL;
    if (p && compile && !strcmp(v, "optrace")) {
        QnnProfile_Config_t c = QNN_PROFILE_CONFIG_INIT;
        c.option = QNN_PROFILE_CONFIG_OPTION_ENABLE_OPTRACE;
        c.enableOptrace = 1;
        const QnnProfile_Config_t *cl[] = { &c, NULL };
        if (!e->q.profileSetConfig || e->q.profileSetConfig(p, cl))
            fprintf(stderr, "qnn: this QNN writes no schematic\n");
    }
    return p;
}

static void print_events(const pk_qnn *e, const QnnProfile_EventId_t *ev, uint32_t n, int depth) {
    for (uint32_t i = 0; i < n; i++) {
        QnnProfile_EventData_t d = QNN_PROFILE_EVENT_DATA_INIT;
        const QnnProfile_EventId_t *sub = NULL;
        uint32_t nsub = 0;
        if (e->q.profileGetEventData(ev[i], &d))
            continue;
        fprintf(stderr, "qnn-prof %d %u %u %llu %s\n", depth, (unsigned)d.type, (unsigned)d.unit,
                (unsigned long long)d.value, d.identifier ? d.identifier : "");
        if (!e->q.profileGetSubEvents(ev[i], &sub, &nsub))
            print_events(e, sub, nsub, depth + 1);
    }
}

static void profile_done(const pk_qnn *e, Qnn_ProfileHandle_t p) {
    const QnnProfile_EventId_t *ev = NULL;
    uint32_t n = 0;
    if (!p)
        return;
    if (!e->q.profileGetEvents(p, &ev, &n))
        print_events(e, ev, n, 0);
    e->q.profileFree(p);
}

/* -------------------------------------------------------- the builder --- */

/* A tensor of the graph being made, and its shape */
typedef struct {
    Qnn_Tensor_t t;
    uint32_t rank, d[4];
} val;

/* What making the graph holds until QNN has compiled it: every name,
 * shape and array made here, freed together after */
typedef struct {
    pk_qnn *e;
    void **mem;
    size_t nmem, cap;
    uint32_t count;              /* tensors and ops made: their names */
    const char *block;           /* the block whose ops are being made, in their names */
    Qnn_ErrorHandle_t status;    /* the first failure, and its call */
    const char *failed;
} build;

static void *keep(build *b, size_t bytes) {
    if (b->nmem == b->cap) {
        size_t cap = b->cap ? 2 * b->cap : 256;
        void **mem = realloc(b->mem, cap * sizeof *mem);
        if (!mem) {
            if (!b->status)
                b->status = QNN_COMMON_ERROR_MEM_ALLOC, b->failed = "memory";
            return NULL;
        }
        b->mem = mem;
        b->cap = cap;
    }
    void *p = calloc(1, bytes ? bytes : 1);
    if (p)
        b->mem[b->nmem++] = p;
    else if (!b->status)
        b->status = QNN_COMMON_ERROR_MEM_ALLOC, b->failed = "memory";
    return p;
}

static void build_free(build *b) {
    for (size_t i = 0; i < b->nmem; i++)
        free(b->mem[i]);
    free(b->mem);
}

static char *name(build *b) {
    char *s = keep(b, 16);
    if (s)
        snprintf(s, 16, "t%u", b->count++);
    return s;
}

/* an op's name: its block, its type and a number, which QNN's profile
 * reports it by (PKTITO_QNN_PROFILE) */
static char *op_name(build *b, const char *type) {
    char *s = keep(b, 48);
    if (s)
        snprintf(s, 48, "%s_%s_%u", b->block, type, b->count++);
    return s;
}

/* A tensor of type and data type dt, of rank and shape d; data its
 * contents for a static one, which must last until the graph is compiled */
static val tensor(build *b, Qnn_TensorType_t type, Qnn_DataType_t dt, uint32_t rank, const uint32_t *d,
                  const void *data, size_t bytes) {
    val v = { QNN_TENSOR_INIT, rank, { 0 } };
    memcpy(v.d, d, rank * sizeof *d);
    uint32_t *dims = keep(b, rank * sizeof *dims);
    char *nm = name(b);
    if (b->status)
        return v;
    memcpy(dims, d, rank * sizeof *dims);
    v.t.v1.name = nm;
    v.t.v1.type = type;
    v.t.v1.dataType = dt;
    v.t.v1.rank = rank;
    v.t.v1.dimensions = dims;
    v.t.v1.memType = QNN_TENSORMEMTYPE_RAW;
    if (data) {
        v.t.v1.clientBuf.data = (void *)data;     /* QNN only reads it */
        v.t.v1.clientBuf.dataSize = (uint32_t)bytes;
    }
    Qnn_ErrorHandle_t e = b->e->q.tensorCreateGraphTensor(b->e->graph, &v.t);
    if (e && !b->status)
        b->status = e, b->failed = "tensorCreateGraphTensor";
    return v;
}

static size_t count(uint32_t rank, const uint32_t *d) {
    size_t n = 1;
    for (uint32_t i = 0; i < rank; i++)
        n *= d[i];
    return n;
}

/* an intermediate float tensor */
static val native(build *b, uint32_t rank, const uint32_t *d) {
    return tensor(b, QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32, rank, d, NULL, 0);
}

/* a float constant: data lasts until the graph is compiled (the model's
 * own, or kept) */
static val constant(build *b, uint32_t rank, const uint32_t *d, const float *data) {
    return tensor(b, QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_32, rank, d, data,
                  count(rank, d) * sizeof(float));
}

/* n floats of the model's, times k, as a constant of rank 1 */
static val scaled(build *b, const float *x, uint32_t n, float k) {
    float *c = keep(b, n * sizeof *c);
    if (c)
        for (uint32_t i = 0; i < n; i++)
            c[i] = x[i] * k;
    return constant(b, 1, &n, c ? c : x);
}

/* Element (r, k) of a model matrix: the .pkf keeps them in panels of 16
 * rows, 4 inputs of each row together (core/pkf.h, PK_Q8), the shape
 * SDOT multiplies; QNN takes them a row at a time */
static int8_t q8_at(const pk_q8 *w, uint32_t r, uint32_t k) {
    return w->q[((size_t)(r / PK_QR) * (w->cols / PK_QG) + k / PK_QG) * (PK_QR * PK_QG) +
                (r % PK_QR) * PK_QG + k % PK_QG];
}

/* rows [r0, r0 + n) of a model matrix as int8 weights [n][cols], row-major,
 * row r's scale its own times k */
static val weights(build *b, const pk_q8 *w, uint32_t r0, uint32_t n, float k) {
    val v = { QNN_TENSOR_INIT, 2, { n, w->cols } };
    uint32_t *dims = keep(b, 2 * sizeof *dims);
    Qnn_ScaleOffset_t *so = keep(b, n * sizeof *so);
    int8_t *q = keep(b, (size_t)n * w->cols);
    char *nm = name(b);
    if (b->status)
        return v;
    dims[0] = n;
    dims[1] = w->cols;
    for (uint32_t i = 0; i < n; i++) {
        so[i] = (Qnn_ScaleOffset_t){ w->s[r0 + i] * k, 0 };
        for (uint32_t c = 0; c < w->cols; c++)
            q[(size_t)i * w->cols + c] = q8_at(w, r0 + i, c);
    }
    v.t.v1.name = nm;
    v.t.v1.type = QNN_TENSOR_TYPE_STATIC;
    v.t.v1.dataType = QNN_DATATYPE_SFIXED_POINT_8;
    v.t.v1.rank = 2;
    v.t.v1.dimensions = dims;
    v.t.v1.memType = QNN_TENSORMEMTYPE_RAW;
    v.t.v1.clientBuf.data = q;
    v.t.v1.clientBuf.dataSize = n * w->cols;
    v.t.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
    v.t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
    v.t.v1.quantizeParams.axisScaleOffsetEncoding.axis = 0;
    v.t.v1.quantizeParams.axisScaleOffsetEncoding.numScaleOffsets = n;
    v.t.v1.quantizeParams.axisScaleOffsetEncoding.scaleOffset = so;
    Qnn_ErrorHandle_t e = b->e->q.tensorCreateGraphTensor(b->e->graph, &v.t);
    if (e && !b->status)
        b->status = e, b->failed = "tensorCreateGraphTensor (int8 weights)";
    return v;
}

static Qnn_Param_t scalar_bool(const char *nm, int v) {
    Qnn_Param_t p = { .paramType = QNN_PARAMTYPE_SCALAR, .name = nm };
    p.scalarParam.dataType = QNN_DATATYPE_BOOL_8;
    p.scalarParam.bool8Value = (uint8_t)(v != 0);
    return p;
}

static Qnn_Param_t scalar_f32(const char *nm, float v) {
    Qnn_Param_t p = { .paramType = QNN_PARAMTYPE_SCALAR, .name = nm };
    p.scalarParam.dataType = QNN_DATATYPE_FLOAT_32;
    p.scalarParam.floatValue = v;
    return p;
}

/* a parameter of 32-bit integers of type dt in the shape d of the given
 * rank */
static Qnn_Param_t int_param(build *b, const char *nm, Qnn_DataType_t dt, uint32_t rank, const uint32_t *d,
                             const void *v) {
    const size_t n = count(rank, d);
    uint32_t *c = keep(b, n * sizeof *c);
    if (c)
        memcpy(c, v, n * sizeof *c);
    val t = tensor(b, QNN_TENSOR_TYPE_STATIC, dt, rank, d, c, n * sizeof *c);
    Qnn_Param_t p = { .paramType = QNN_PARAMTYPE_TENSOR, .name = nm };
    p.tensorParam = t.t;
    return p;
}

static Qnn_Param_t ints(build *b, const char *nm, uint32_t rank, const uint32_t *d, const uint32_t *v) {
    return int_param(b, nm, QNN_DATATYPE_UINT_32, rank, d, v);
}

static Qnn_Param_t scalar_u32(const char *nm, uint32_t v) {
    Qnn_Param_t p = { .paramType = QNN_PARAMTYPE_SCALAR, .name = nm };
    p.scalarParam.dataType = QNN_DATATYPE_UINT_32;
    p.scalarParam.uint32Value = v;
    return p;
}

/* an op of QNN's own package on the inputs, into out */
static void node(build *b, const char *type, const val *in, uint32_t nin, const val *out,
                 Qnn_Param_t *par, uint32_t npar) {
    Qnn_Tensor_t ins[16], outs[1] = { out->t };
    for (uint32_t i = 0; i < nin; i++)
        ins[i] = in[i].t;
    char *nm = op_name(b, type);
    if (b->status)
        return;
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.v1.name = nm;
    op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
    op.v1.typeName = type;
    op.v1.numOfParams = npar;
    op.v1.params = par;
    op.v1.numOfInputs = nin;
    op.v1.inputTensors = ins;
    op.v1.numOfOutputs = 1;
    op.v1.outputTensors = outs;
    Qnn_ErrorHandle_t e = b->e->q.graphAddNode(b->e->graph, op);
    if (e && !b->status)
        b->status = e, b->failed = type;
}

/* ------------------------------------------------------------- the ops --- */

/* Each op adds its node as it is called, and it and its tensors are named
 * by how many came before (build's count): the graph is the order of the
 * calls, and QNN compiles another order to another binary. C leaves the
 * order of a call's arguments, and of an initializer's elements, to the
 * compiler - gcc 15 evaluated make_graph's arguments last first, clang 21
 * first first - so no call here takes more than one argument that adds to
 * the graph: the others are made before it, in the order the code reads,
 * and a gcc build and a clang build compile the same binary
 * (tools/qnn/CLIMB.md, 2026-11-03). */

/* x [..][in] W^T, plus bias where there is one: [..][rows] */
static val fc(build *b, val x, const pk_q8 *w, uint32_t r0, uint32_t rows, float k, const val *bias) {
    val in[3] = { x, weights(b, w, r0, rows, k) };
    if (bias)
        in[2] = *bias;
    uint32_t d[2] = { (uint32_t)(count(x.rank, x.d) / w->cols), rows };
    val y = native(b, 2, d);
    node(b, QNN_OP_FULLY_CONNECTED, in, bias ? 3 : 2, &y, NULL, 0);
    return y;
}

/* x [..][in] W^T for a float constant W [out][in]: [..][out] */
static val fc_const(build *b, val x, val w) {
    const uint32_t d[2] = { (uint32_t)(count(x.rank, x.d) / w.d[1]), w.d[0] };
    val y = native(b, 2, d);
    val in[2] = { x, w };
    node(b, QNN_OP_FULLY_CONNECTED, in, 2, &y, NULL, 0);
    return y;
}

/* an element-wise op of two, broadcast to the first's shape */
static val binary(build *b, const char *type, val x, val y) {
    val z = native(b, x.rank, x.d);
    val in[2] = { x, y };
    node(b, type, in, 2, &z, NULL, 0);
    return z;
}

static val add(build *b, val x, val y) { return binary(b, QNN_OP_ELEMENT_WISE_ADD, x, y); }
static val mul(build *b, val x, val y) { return binary(b, QNN_OP_ELEMENT_WISE_MULTIPLY, x, y); }

static val sigmoid(build *b, val x) {
    val y = native(b, x.rank, x.d);
    node(b, QNN_OP_SIGMOID, &x, 1, &y, NULL, 0);
    return y;
}

static val silu(build *b, val x) { return mul(b, x, sigmoid(b, x)); }

static val relu(build *b, val x) {
    val y = native(b, x.rank, x.d);
    node(b, QNN_OP_RELU, &x, 1, &y, NULL, 0);
    return y;
}

static val reshape(build *b, val x, uint32_t rank, const uint32_t *d) {
    val y = native(b, rank, d);
    node(b, QNN_OP_RESHAPE, &x, 1, &y, NULL, 0);
    return y;
}

/* rank 3, axes 0 and 1 swapped */
static val swap01(build *b, val x) {
    const uint32_t d[3] = { x.d[1], x.d[0], x.d[2] }, perm[3] = { 1, 0, 2 }, n = 3;
    val y = native(b, 3, d);
    Qnn_Param_t p = ints(b, QNN_OP_TRANSPOSE_PARAM_PERM, 1, &n, perm);
    node(b, QNN_OP_TRANSPOSE, &x, 1, &y, &p, 1);
    return y;
}

/* x [T][D] as [H][T][dk] */
static val heads(build *b, val x, uint32_t H) {
    const uint32_t d[3] = { x.d[0], H, x.d[1] / H };
    return swap01(b, reshape(b, x, 3, d));
}

/* and back */
static val unheads(build *b, val x) {
    val t = swap01(b, x);
    const uint32_t d[2] = { t.d[0], t.d[1] * t.d[2] };
    return reshape(b, t, 2, d);
}

/* x [lo[0], hi[0]) [lo[1], hi[1]) .. on each of its axes */
static val slice(build *b, val x, const uint32_t *lo, const uint32_t *hi) {
    uint32_t d[4], dr[2] = { x.rank, 3 };
    int32_t r[12];
    for (uint32_t i = 0; i < x.rank; i++) {
        d[i] = hi[i] - lo[i];
        r[3 * i] = (int32_t)lo[i], r[3 * i + 1] = (int32_t)hi[i], r[3 * i + 2] = 1;
    }
    val y = native(b, x.rank, d);
    Qnn_Param_t p = int_param(b, QNN_OP_STRIDED_SLICE_PARAM_RANGES, QNN_DATATYPE_INT_32, 2, dr, r);
    node(b, QNN_OP_STRIDED_SLICE, &x, 1, &y, &p, 1);
    return y;
}

/* n tensors of one shape but on axis, end to end along it */
static val concat(build *b, const val *x, uint32_t n, uint32_t axis) {
    uint32_t d[4];
    memcpy(d, x[0].d, sizeof d);
    d[axis] = 0;
    for (uint32_t i = 0; i < n; i++)
        d[axis] += x[i].d[axis];
    val y = native(b, x[0].rank, d);
    Qnn_Param_t p = scalar_u32(QNN_OP_CONCAT_PARAM_AXIS, axis);
    node(b, QNN_OP_CONCAT, x, n, &y, &p, 1);
    return y;
}

/* x [..][m][k] times y [..][k][n], or y [..][n][k] with ty: [..][m][n] */
static val matmul(build *b, val x, val y, int ty) {
    val z = { QNN_TENSOR_INIT, x.rank, { 0 } };
    memcpy(z.d, x.d, sizeof z.d);
    z.d[x.rank - 1] = ty ? y.d[y.rank - 2] : y.d[y.rank - 1];
    z = native(b, z.rank, z.d);
    Qnn_Param_t p[2] = { scalar_bool(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, 0),
                         scalar_bool(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, ty) };
    val in[2] = { x, y };
    node(b, QNN_OP_MAT_MUL, in, 2, &z, p, 2);
    return z;
}

static val softmax(build *b, val x) {
    val y = native(b, x.rank, x.d);
    node(b, QNN_OP_SOFTMAX, &x, 1, &y, NULL, 0);
    return y;
}

/* over the last axis of x [T][D]: gamma and beta the model's times k, into
 * a tensor of type (the graph's output, for the last) */
static val norm(build *b, val x, const float *w, const float *bb, float k, float eps, Qnn_TensorType_t type) {
    const uint32_t D = x.d[x.rank - 1], axis = x.rank - 1, one = 1;
    val gamma = scaled(b, w, D, k), beta = scaled(b, bb, D, k);
    val in[3] = { x, gamma, beta };
    val y = tensor(b, type, QNN_DATATYPE_FLOAT_32, x.rank, x.d, NULL, 0);
    Qnn_Param_t p[2] = { scalar_f32(QNN_OP_LAYER_NORM_PARAM_EPSILON, eps),
                         ints(b, QNN_OP_LAYER_NORM_PARAM_AXES, 1, &one, &axis) };
    node(b, QNN_OP_LAYER_NORM, in, 3, &y, p, 2);
    return y;
}

/* A convolution over [1][H][W][C]: filter [kh][kw][1 or C][Co] and bias,
 * stride s both ways, padding ph before and after along H and pw along W,
 * depthwise or not */
static val conv(build *b, val x, val filt, val bias, uint32_t s, uint32_t ph, uint32_t pw, int depthwise) {
    const uint32_t kh = filt.d[0], kw = filt.d[1];
    const uint32_t d[4] = { 1, (x.d[1] + 2 * ph - kh) / s + 1, (x.d[2] + 2 * pw - kw) / s + 1, filt.d[3] };
    val y = native(b, 4, d);
    const uint32_t two = 2, two2[2] = { 2, 2 }, stride[2] = { s, s }, pad[4] = { ph, ph, pw, pw };
    Qnn_Param_t p[2];
    p[0] = ints(b, depthwise ? QNN_OP_DEPTH_WISE_CONV_2D_PARAM_STRIDE : QNN_OP_CONV_2D_PARAM_STRIDE, 1, &two, stride);
    p[1] = ints(b, depthwise ? QNN_OP_DEPTH_WISE_CONV_2D_PARAM_PAD_AMOUNT : QNN_OP_CONV_2D_PARAM_PAD_AMOUNT, 2, two2,
                pad);
    val in[3] = { x, filt, bias };
    node(b, depthwise ? QNN_OP_DEPTH_WISE_CONV_2D : QNN_OP_CONV_2D, in, 3, &y, p, 2);
    return y;
}

/* reference's 3x3 taps, [C][9] at 3 * (time offset) + (frequency
 * offset), as a filter [3 time][3 frequency][1][C] */
static val taps3(build *b, const float *w, uint32_t C) {
    float *f = keep(b, 9 * (size_t)C * sizeof *f);
    if (f)
        for (uint32_t c = 0; c < C; c++)
            for (uint32_t k = 0; k < 9; k++)
                f[(size_t)k * C + c] = w[(size_t)c * 9 + k];
    const uint32_t d[4] = { 3, 3, 1, C };
    return constant(b, 4, d, f ? f : w);
}

/* the subsampling's 3x3 convolution of C channels, stride 2, padded 1:
 * reference's taps w and bias wb */
static val conv3(build *b, val x, const float *w, const float *wb, uint32_t C, int depthwise) {
    val filt = taps3(b, w, C), bias = constant(b, 1, &C, wb);
    return conv(b, x, filt, bias, 2, 1, 1, depthwise);
}

/* ------------------------------------------------------------- the graph --- */

typedef struct {
    val cs, pe;                  /* the turn's tables, [T3][2 D] each */
    val m3, kbias;               /* [T3][1], [1][1][T3] */
    float s, eps;                /* the stream's scale, and the norms' epsilon for it */
} common;

/* x + ff(LN x): up, SiLU, down (its 0.5 in down's weights) */
static val ff(build *b, const common *c, val x, const float *nw, const float *nb, const pk_q8 *up,
              const pk_q8 *down) {
    val h = fc(b, norm(b, x, nw, nb, 1, c->eps, QNN_TENSOR_TYPE_NATIVE), up, 0, up->rows, 1, NULL);
    return add(b, x, fc(b, silu(b, h), down, 0, down->rows, c->s, NULL));
}

/* The turn's first product for head h, as the weights of a matrix [2 D][dk]:
 * row e of W_pos^T's head h, the model's matrix as floats, and below them
 * the same rows with each (sin, cos) pair of columns swapped */
static val pos_turn(build *b, const pk_q8 *w, uint32_t h, uint32_t dk) {
    const uint32_t D = w->cols;
    float *f = keep(b, (size_t)2 * D * dk * sizeof *f);
    if (f)
        for (uint32_t e = 0; e < 2 * D; e++)
            for (uint32_t k = 0; k < dk; k++) {
                const uint32_t r = h * dk + k;
                f[(size_t)e * dk + k] = q8_at(w, r, e < D ? e : (e - D) ^ 1) * w->s[r];
            }
    const uint32_t d[2] = { 2 * D, dk };
    return constant(b, 2, d, f);
}

static val layer(build *b, const pk_model *m, const common *c, const pk_layer *L, val x, int last) {
    const uint32_t D = m->d_model, H = m->n_heads, dk = D / H, T = x.d[0], K = m->conv_kernel;
    const float sk = 1.0f / sqrtf((float)dk);

    b->block = "ff1";
    x = ff(b, c, x, L->ff1_nw, L->ff1_nb, &L->ff1_up, &L->ff1_down);

    /* attention with relative positions */
    b->block = "att";
    val n = norm(b, x, L->att_nw, L->att_nb, 1, c->eps, QNN_TENSOR_TYPE_NATIVE);
    val Q = fc(b, n, &L->q, 0, D, 1, NULL);
    val Kh = heads(b, fc(b, n, &L->k, 0, D, sk, NULL), H);       /* 1/sqrt(dk) in K */
    val Vh = heads(b, fc(b, n, &L->v, 0, D, 1, NULL), H);
    val qu = heads(b, add(b, Q, constant(b, 1, &D, L->bias_u)), H);
    val qv = add(b, Q, constant(b, 1, &D, L->bias_v));
    val ac = matmul(b, qu, Kh, 1);                                 /* [H][T][T] */
    b->block = "pos";
    val bdh[16];
    for (uint32_t h = 0; h < H; h++) {
        /* head h's columns of q + v, turned (two products with W_pos as
         * one, then the tables) and multiplied by pe */
        const uint32_t lo[2] = { 0, h * dk }, hi[2] = { T, (h + 1) * dk }, d3[3] = { 1, T, T };
        val qh = slice(b, qv, lo, hi), turn = pos_turn(b, &L->pos, h, dk);
        val u = fc_const(b, qh, turn);                                              /* [T][2D] */
        bdh[h] = reshape(b, fc_const(b, mul(b, u, c->cs), c->pe), 3, d3);           /* [1][T][T] */
    }
    val bd = concat(b, bdh, H, 0);                                 /* 1/sqrt(dk) in pe */
    b->block = "att";
    val p = softmax(b, add(b, add(b, ac, bd), c->kbias));
    val o = unheads(b, matmul(b, p, Vh, 0));                       /* [T][D] */
    x = add(b, x, fc(b, o, &L->o, 0, D, c->s, NULL));

    /* convolution module: GLU, the padding zeroed, depthwise K, SiLU. The
     * depthwise runs along the width, time there, [1][1][T][D]: along the
     * height, [1][T][1][D], a width of 1, the SM8650 took 1.96 M cycles a
     * layer for the SiLU after it to 0.41 this way, 0.77 for the depthwise
     * to 0.60 and 0.25 for the reshape into it to none, and a chunk 5.5 ms
     * longer, the SM8750's 7.0 and the X1E80100's 6.4 (tools/qnn/CLIMB.md,
     * 2026-11-03, at compile()'s P point) */
    b->block = "conv";
    n = norm(b, x, L->conv_nw, L->conv_nb, 1, c->eps, QNN_TENSOR_TYPE_NATIVE);
    val lin = fc(b, n, &L->pw1, 0, D, 1, NULL), gate = sigmoid(b, fc(b, n, &L->pw1, D, D, 1, NULL));
    val g = mul(b, mul(b, lin, gate), c->m3);
    const uint32_t d4[4] = { 1, 1, T, D }, d2[2] = { T, D }, fd[4] = { 1, K, 1, D };
    float *f = keep(b, (size_t)K * D * sizeof *f);
    if (f)
        for (uint32_t ch = 0; ch < D; ch++)
            for (uint32_t j = 0; j < K; j++)
                f[(size_t)j * D + ch] = L->dw[(size_t)ch * K + j];
    val g4 = reshape(b, g, 4, d4), filt = constant(b, 4, fd, f), bias = constant(b, 1, &D, L->dw_b);
    g = conv(b, g4, filt, bias, 1, 0, K / 2, 1);
    g = reshape(b, silu(b, g), 2, d2);
    x = add(b, x, fc(b, g, &L->pw2, 0, D, c->s, NULL));

    b->block = "ff2";
    x = ff(b, c, x, L->ff2_nw, L->ff2_nb, &L->ff2_up, &L->ff2_down);
    /* the last layer's norm unscaled: the encoder's own output */
    b->block = "out";
    return norm(b, x, L->out_nw, L->out_nb, last ? 1 : c->s, c->eps,
                last ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE);
}

/* The turn's tables for positions 0..T-1, [T][2 D] each, in double,
 * rounded once: (-cos, cos) of j w for each pair of frequency w, then
 * (sin, sin); pe(j) less its mean over j, times 1/sqrt(dk), twice */
static void tables(const pk_model *m, uint32_t T, float *cs, float *pe) {
    const uint32_t D = m->d_model;
    const double sk = 1.0 / sqrt((double)(D / m->n_heads));
    for (uint32_t e = 0; e < D; e++) {
        const double w = m->pos_freq[e / 2];
        double mean = 0;
        for (uint32_t j = 0; j < T; j++)
            mean += (e % 2 ? cos(j * w) : sin(j * w)) / T;
        for (uint32_t j = 0; j < T; j++) {
            const size_t r = (size_t)j * 2 * D;
            pe[r + e] = pe[r + D + e] = (float)(((e % 2 ? cos(j * w) : sin(j * w)) - mean) * sk);
            cs[r + e] = (float)(e % 2 ? cos(j * w) : -cos(j * w));
            cs[r + D + e] = (float)sin(j * w);
        }
    }
}

static int make_graph(pk_qnn *e, const pk_model *m, char *err, size_t errlen) {
    build b = { e, NULL, 0, 0, 0, "sub", 0, NULL };
    const uint32_t F0 = m->n_mels, C = m->sub_channels, D = m->d_model;
    const uint32_t F1 = (F0 + 1) / 2, F2 = (F1 + 1) / 2, F3 = (F2 + 1) / 2;
    const uint32_t T2 = e->T2, T3 = e->T3;
    common c;
    c.s = 1.0f / PK_QNN_SCALE;
    c.eps = m->ln_eps * c.s * c.s;

    /* the inputs: the features, padded with zeros; the masks (io_shape) */
    val inputs[5];
    for (int k = 0; k < 5; k++) {
        uint32_t d[4];
        const uint32_t r = io_shape(e, k, d);
        inputs[k] = tensor(&b, QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32, r, d, NULL, 0);
        e->in[k] = inputs[k].t;
    }
    val mel = inputs[0], m1 = inputs[1], m2 = inputs[2];
    c.m3 = inputs[3], c.kbias = inputs[4];

    /* subsampling: [1][time][frequency][C] at each stage */
    val x = relu(&b, conv3(&b, mel, m->conv0_w, m->conv0_b, C, 0));
    x = mul(&b, x, m1);
    x = conv3(&b, x, m->dw1_w, m->dw1_b, C, 1);
    val pw1_b = constant(&b, 1, &C, m->pw1_b), pw2_b = constant(&b, 1, &C, m->pw2_b);
    const uint32_t d2[4] = { 1, T2, F2, C }, d3[4] = { 1, T3, F3, C }, flat[2] = { T3, F3 * C };
    x = mul(&b, reshape(&b, relu(&b, fc(&b, x, &m->sub_pw1, 0, C, 1, &pw1_b)), 4, d2), m2);
    x = conv3(&b, x, m->dw2_w, m->dw2_b, C, 1);
    x = reshape(&b, relu(&b, fc(&b, x, &m->sub_pw2, 0, C, 1, &pw2_b)), 4, d3);
    /* row t is frame t's [F3][C], sub.out's input order (encoder.c's s5).
     * The frames past the end are left as they come, not zeroed: a frame
     * of zeros has no variance, and its layer norm, with the scaled
     * epsilon (1.6e-7) below what fp16 holds, is 0 / 0, which every
     * query's attention then takes in. */
    val out_b = scaled(&b, m->out_b, D, c.s);
    x = fc(&b, reshape(&b, x, 2, flat), &m->sub_out, 0, D, c.s, &out_b);

    const uint32_t dt[2] = { T3, 2 * D };
    float *cs = keep(&b, (size_t)T3 * 2 * D * sizeof *cs), *pe = keep(&b, (size_t)T3 * 2 * D * sizeof *pe);
    if (cs && pe)
        tables(m, T3, cs, pe);
    c.cs = constant(&b, 2, dt, cs);
    c.pe = constant(&b, 2, dt, pe);

    for (uint32_t i = 0; i < e->layers && !b.status; i++)
        x = layer(&b, m, &c, &m->layers[i], x, i + 1 == m->n_layers);
    if (e->layers < m->n_layers) {
        /* the stream after fewer layers, for checking (PKTITO_QNN_LAYERS),
         * unscaled into the output */
        float *k = keep(&b, sizeof *k);
        if (k)
            *k = PK_QNN_SCALE;
        const uint32_t one = 1;
        b.block = "out";
        val y = tensor(&b, QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32, x.rank, x.d, NULL, 0);
        val in[2] = { x, constant(&b, 1, &one, k) };
        node(&b, QNN_OP_ELEMENT_WISE_MULTIPLY, in, 2, &y, NULL, 0);
        x = y;
    }
    e->out = x.t;

    int rc = 0;
    if (b.status)
        rc = fail(err, errlen, b.failed, b.status);
    else {
        Qnn_ProfileHandle_t prof = profile(e, 1);
        Qnn_ErrorHandle_t s = e->q.graphFinalize(e->graph, prof, NULL);
        profile_done(e, prof);
        if (s)
            rc = fail(err, errlen, "graphFinalize", s);
        else
            own_io(e);
    }
    build_free(&b);
    return rc;
}

/* ---------------------------------------------------------- the cache --- */

/* The compiled graph kept (pk_qnn_open's cache), a file the app downloads
 * for its phone's NPU: this header, then QNN's binary of the context. The
 * key is what the binary was made of: the model's checkpoint and
 * conversion as its metadata names them (source_sha256, converter), which
 * a slim file (pktito-tools slim) has byte for byte as the full file whose
 * encoder the graph holds - or, in a file without them, its own data's
 * SHA-256, the .pkf header's - and the length and layers built. The NPU
 * it was compiled for is kept too, as PKTITO_QNN_TARGET names it, so that
 * a file of another Hexagon's is refused saying both, before QNN, whose
 * refusal names neither ("Request feature arch with value 79
 * unsupported"); and the pktito and the QNN that made it. The QNN is kept
 * beside the key, not in it: QNN reads an older QNN's graph and not a
 * newer one's (QAIRT 2.51's HTP backend guide, its table of SDK 2.42's
 * and 2.43's), and contextCreateFromBinary says which. PK_QNN_GRAPH's
 * number changes with this header and with the graph make_graph and
 * compile() make, so that an older one's file is made or downloaded
 * again, not run: 2 has the depthwise along the width and P point 13
 * (tools/qnn/CLIMB.md, 2026-11-03), 1 was the first the app downloaded,
 * the development's having been "pktito-qnn 1" to "pktito-qnn 6".
 * Little-endian, 616 bytes, as tools read it. */
#define PK_QNN_GRAPH "pktito-qnn-graph 2"

/* the commit the Makefile built qnn.o from, for the header's made */
#ifndef PK_COMMIT
#define PK_COMMIT ""
#endif

typedef struct {
    char magic[32];              /* PK_QNN_GRAPH */
    char key[176];               /* cache_key's */
    char chip[32];               /* the NPU it was compiled for: soc,arch,vtcm,dlbc,signed */
    char made[64];               /* the pktito that made it: "pktito 0.1.0 e9603fd" */
    char built[64];              /* the QNN that compiled it: "v2.51.0.260929181845" */
    char names[6][16];           /* the inputs' and output's, and their ids and shapes */
    uint32_t id[6], rank[6], dims[6][4];
    uint64_t bytes;              /* the binary's, which follows */
} cache_head;
static_assert(sizeof(cache_head) == 616, "the graph file's header, as tools read it");

static void cache_key(const pk_qnn *e, const pk_model *m, char *key, size_t n) {
    const char *src = pk_model_meta(m, "source_sha256"), *conv = pk_model_meta(m, "converter");
    if (src && conv) {
        snprintf(key, n, "%s %s %u %u", src, conv, e->T0, e->layers);
        return;
    }
    const pk_header *h = (const pk_header *)m->map;
    char hex[65];
    for (int i = 0; i < 32; i++)
        snprintf(hex + 2 * i, 3, "%02x", h->digest[i]);
    snprintf(key, n, "%s %u %u", hex, e->T0, e->layers);
}

/* The Hexagon's version in a chip's words (target_words: soc,arch,...):
 * their second number, read from at most n bytes; 0 where there is none */
static unsigned hexagon(const char *w, size_t n) {
    const char *c = memchr(w, ',', n);
    unsigned v = 0;
    for (c = c ? c + 1 : w + n; c < w + n && *c >= '0' && *c <= '9'; c++)
        v = v * 10 + (unsigned)(*c - '0');
    return v;
}

/* QNN's build, as the cache keeps it */
static const char *build_id(const pk_qnn *e) {
    const char *b = NULL;
    return e->q.backendGetBuildId(&b) || !b ? "?" : b;
}

/* 1 if the n bytes at s are a string and zeros after it, as write_cache
 * leaves each of the header's (snprintf into a zeroed header) */
static int padded(const char *s, size_t n) {
    const char *z = memchr(s, 0, n);
    if (!z)
        return 0;
    while (++z < s + n)
        if (*z)
            return 0;
    return 1;
}

/* A compiled graph's file held to what e reads for m, before QNN sees a
 * byte of its binary (docs/MEMORY_SAFETY.md): head the first n bytes of a
 * file of size bytes, path its name for err. 0, or -1 and why in err */
static int check_head(const pk_qnn *e, const pk_model *m, const void *head, size_t n, uint64_t size,
                      const char *path, char *err, size_t errlen) {
    cache_head h, want;
    memset(&want, 0, sizeof want);
    snprintf(want.magic, sizeof want.magic, "%s", PK_QNN_GRAPH);
    cache_key(e, m, want.key, sizeof want.key);
    if (n < sizeof h) {
        snprintf(err, errlen, "%s is not a graph this pktito reads (" PK_QNN_GRAPH ")", path);
        return -1;
    }
    memcpy(&h, head, sizeof h);
    if (memcmp(h.magic, want.magic, sizeof h.magic)) {
        snprintf(err, errlen, "%s is not a graph this pktito reads (" PK_QNN_GRAPH ")", path);
        return -1;
    }
    if (memcmp(h.key, want.key, sizeof h.key)) {
        snprintf(err, errlen, "%s is the graph of another model, length or layers", path);
        return -1;
    }
    if (!padded(h.chip, sizeof h.chip) || !padded(h.made, sizeof h.made) || !padded(h.built, sizeof h.built)) {
        snprintf(err, errlen, "%s: its header's chip, made or built is not a string padded with zeros", path);
        return -1;
    }
    /* another Hexagon's graph, which no QNN runs; one compiled for another
     * SoC of this one's is QNN's to run or refuse - untried, each SoC here
     * having a graph of its own (v73 the X1E80100, v75 the SM8650, v79 the
     * SM8750, v81 the SM8850 and the SC8480XP, 2026-11-05) */
    if (h.chip[0] && e->chip[0] && hexagon(h.chip, sizeof h.chip) != hexagon(e->chip, sizeof e->chip)) {
        snprintf(err, errlen, "%s is the graph of the NPU %s, and this one is %s", path, h.chip, e->chip);
        return -1;
    }
    /* the inputs and output, each of a name and id of its own and of the
     * shape of pktito's buffer for it. The ids are the binary's word, given
     * by QNN when it compiled the graph and kept here for the read, so
     * nothing but their being six is checked */
    for (int k = 0; k < 6; k++) {
        uint32_t d[4];
        const uint32_t r = io_shape(e, k, d);
        int twice = 0;
        for (int j = 0; j < k; j++)
            twice |= h.id[j] == h.id[k] || !strncmp(h.names[j], h.names[k], sizeof h.names[k]);
        if (!h.names[k][0] || !padded(h.names[k], sizeof h.names[k]) || twice) {
            snprintf(err, errlen, "%s: its input or output %d has no name and id of its own", path, k);
            return -1;
        }
        if (h.rank[k] != r || memcmp(h.dims[k], d, sizeof d)) {
            snprintf(err, errlen, "%s: its input or output %d is not the shape of pktito's buffer for it", path, k);
            return -1;
        }
    }
    /* the binary the rest of the file, known before anything is allocated:
     * under 4 GiB, which keeps the sum from wrapping */
    if (h.bytes == 0 || h.bytes >= (uint64_t)1 << 32 || h.bytes + sizeof h != size) {
        snprintf(err, errlen, "%s: its header says %llu bytes of graph, in a file of %llu", path,
                 (unsigned long long)h.bytes, (unsigned long long)size);
        return -1;
    }
    return 0;
}

/* For a graph the NPU refused, on Windows: 1, and into what which of this
 * QNN's skel and .cat isn't beside QnnHtp.dll, where ADSP_LIBRARY_PATH
 * names no folder for them. Without both the NPU runs its driver's own
 * older QNN, which says only "Can't read future blob" (snappy, QNN 2.51:
 * tools/qnn/CLIMB.md, 2026-11-08); named first, ahead of QNN's words,
 * which an error buffer of 256 may not hold */
static int no_skel(const pk_qnn *e, const char *path, char *what, size_t n) {
#ifdef _WIN32
    unsigned arch;
    char name[2][32];
    wchar_t p[MAX_PATH + 32];
    const DWORD k = GetModuleFileNameW((HMODULE)e->lib, p, MAX_PATH);
    wchar_t *slash = k && k < MAX_PATH ? wcsrchr(p, L'\\') : NULL;
    if (!slash || getenv("ADSP_LIBRARY_PATH") || sscanf(e->chip, "%*u,%u", &arch) != 1)
        return 0;
    snprintf(name[0], sizeof name[0], "libQnnHtpV%uSkel.so", arch);
    snprintf(name[1], sizeof name[1], "libqnnhtpv%u.cat", arch);
    int miss = 0;
    for (int i = 0; i < 2; i++)
        if (MultiByteToWideChar(CP_UTF8, 0, name[i], -1, slash + 1, 32) && GetFileAttributesW(p) == INVALID_FILE_ATTRIBUTES)
            miss |= 1 << i;
    if (!miss)
        return 0;
    snprintf(what, n, "%s: no %s%s%s beside QnnHtp.dll: contextCreateFromBinary", path, miss & 1 ? name[0] : "",
             miss == 3 ? " or " : "", miss & 2 ? name[1] : "");
    return 1;
#else
    (void)e, (void)path, (void)what, (void)n;
    return 0;
#endif
}

/* 0 if the cache held this graph and the NPU took it; -1, and why in err */
static int read_cache(pk_qnn *e, const pk_model *m, const char *path, char *err, size_t errlen) {
    last_error[0] = 0;
    FILE *f = pk_os_fopen(path, "rb");
    if (!f) {
        snprintf(err, errlen, "no compiled graph at %s", path);
        return -1;
    }
    /* its length as pk_map_file finds a model's */
    long size = -1;
    if (fseek(f, 0, SEEK_END) == 0)
        size = ftell(f);
    cache_head h;
    const size_t n = size >= 0 && fseek(f, 0, SEEK_SET) == 0 ? fread(&h, 1, sizeof h, f) : 0;
    if (check_head(e, m, &h, n, size > 0 ? (uint64_t)size : 0, path, err, errlen)) {
        fclose(f);
        return -1;
    }
    void *bin = NULL;
    int rc = -1;
    Qnn_ErrorHandle_t s;
    char what[512];
    if (!(bin = malloc(h.bytes)) || fread(bin, 1, h.bytes, f) != h.bytes)
        snprintf(err, errlen, "%s: its graph could not be read", path);
    else if ((s = e->q.contextCreateFromBinary(e->backend, e->device, NULL, bin, h.bytes, &e->context, NULL))) {
        if (!no_skel(e, path, what, sizeof what))
            snprintf(what, sizeof what, "%s, compiled by QNN %s, in QNN %s: contextCreateFromBinary", path,
                     h.built, build_id(e));
        fail(err, errlen, what, s);
    } else if ((s = e->q.graphRetrieve(e->context, "encoder", &e->graph)))
        fail(err, errlen, "graphRetrieve", s);
    else {
        /* the header's names and ids, checked; the shapes pktito's own,
         * which the header's are */
        for (int k = 0; k < 6; k++) {
            Qnn_Tensor_t *t = io(e, k);
            *t = (Qnn_Tensor_t)QNN_TENSOR_INIT;
            memcpy(e->names[k], h.names[k], sizeof e->names[k]);
            t->v1.id = h.id[k];
            t->v1.name = e->names[k];
            t->v1.type = k < 5 ? QNN_TENSOR_TYPE_APP_WRITE : QNN_TENSOR_TYPE_APP_READ;
            t->v1.dataType = QNN_DATATYPE_FLOAT_32;
            t->v1.rank = io_shape(e, k, e->dims[k]);
            t->v1.dimensions = e->dims[k];
            t->v1.memType = QNN_TENSORMEMTYPE_RAW;
        }
        rc = 0;
    }
    if (rc && e->context) {
        e->q.contextFree(e->context, NULL);
        e->context = NULL;
    }
    free(bin);
    fclose(f);
    return rc;
}

/* 1 if a graph may be written over what is at path: nothing, or a graph
 * file of pktito's that read_cache had no use for - one of another of its
 * formats, or of this one for this model, length, layers and chip - and
 * never another file, such as a model's, nor another model's or chip's
 * graph, which another run may want (a second PKTITO_QNN_TARGET's) */
static int replaceable(const pk_qnn *e, const pk_model *m, const char *path) {
    FILE *f = pk_os_fopen(path, "rb");
    if (!f)
        return 1;
    cache_head h;
    const size_t n = fread(&h, 1, sizeof h, f);
    fclose(f);
    if (n < sizeof h.magic || strncmp(h.magic, "pktito-qnn", 10))
        return 0;
    if (strncmp(h.magic, PK_QNN_GRAPH, sizeof h.magic))
        return 1;
    char key[sizeof h.key];
    cache_key(e, m, key, sizeof key);
    return n == sizeof h && !strncmp(h.key, key, sizeof h.key) && !strncmp(h.chip, e->chip, sizeof h.chip);
}

/* the compiled graph written to path, for the next time: into path.tmp,
 * renamed over path once whole, where replaceable allows. A failure only
 * costs the next time its compile */
static void write_cache(pk_qnn *e, const pk_model *m, const char *path) {
    Qnn_ContextBinarySize_t n = 0, got = 0;
    if (!replaceable(e, m, path)) {
        if (getenv("PKTITO_QNN_LOG"))
            fprintf(stderr, "qnn: %s kept: not a graph of this model and chip to write over\n", path);
        return;
    }
    if (e->q.contextGetBinarySize(e->context, &n) || n == 0)
        return;
    const size_t len = strlen(path);
    char *tmp = malloc(len + sizeof ".tmp");
    void *bin = malloc(n);
    cache_head h;
    memset(&h, 0, sizeof h);
    snprintf(h.magic, sizeof h.magic, "%s", PK_QNN_GRAPH);
    cache_key(e, m, h.key, sizeof h.key);
    snprintf(h.chip, sizeof h.chip, "%s", e->chip);
    snprintf(h.made, sizeof h.made, "pktito %s%s%s", PKTITO_VERSION, *PK_COMMIT ? " " : "", PK_COMMIT);
    snprintf(h.built, sizeof h.built, "%s", build_id(e));
    for (int k = 0; k < 6; k++) {
        const Qnn_Tensor_t *t = io(e, k);
        memcpy(h.names[k], e->names[k], sizeof h.names[k]);
        h.id[k] = t->v1.id;
        h.rank[k] = t->v1.rank;
        memcpy(h.dims[k], e->dims[k], sizeof h.dims[k]);
    }
    FILE *f = NULL;
    if (tmp && bin) {
        memcpy(tmp, path, len);
        memcpy(tmp + len, ".tmp", sizeof ".tmp");
        f = pk_os_fopen(tmp, "wb");
    }
    if (f) {
        int ok = !e->q.contextGetBinary(e->context, bin, n, &got) && (h.bytes = got) > 0 &&
                 fwrite(&h, sizeof h, 1, f) == 1 && fwrite(bin, 1, got, f) == got;
        ok = !fclose(f) && ok;
        if (!ok || pk_os_rename(tmp, path))
            pk_os_remove(tmp);
    }
    free(tmp);
    free(bin);
}

int pk_qnn_check(const pk_model *m, uint32_t frames, const char *chip, const void *head, size_t n,
                 uint64_t size, char *err, size_t errlen) {
    pk_qnn *e = calloc(1, sizeof *e);
    if (!e) {
        snprintf(err, errlen, "out of memory");
        return -1;
    }
    lengths(e, m, frames);
    snprintf(e->chip, sizeof e->chip, "%s", chip ? chip : "");
    const int rc = check_head(e, m, head, n, size, "the file", err, errlen);
    free(e);
    return rc;
}

/* ------------------------------------------------------------ the NPU --- */

/* The transcriber's own power config, for vote(), where the HTP has the
 * infrastructure (a PC's emulation has none); destroyed with the device */
static void power(pk_qnn *e) {
    QnnDevice_Infrastructure_t infra = NULL;
    if (e->q.deviceGetInfrastructure(&infra) || !infra)
        return;
    QnnHtpDevice_PerfInfrastructure_t *perf = &((QnnHtpDevice_Infrastructure_t *)infra)->perfInfra;
    uint32_t id = 0;
    if (perf->createPowerConfigId && perf->setPowerConfig && perf->destroyPowerConfigId &&
        !perf->createPowerConfigId(0, 0, &id) && id)
        e->perf = perf, e->power = id;
}

/* The HTP's clocks voted up for a run, and down after it: chapter 1 of
 * Alice's bench took the Googlebook's X1E80100 12.35 s without the vote
 * and 4.16 s with it, the S24's SM8650 2.9 to 3.4 s either way
 * (tools/qnn/CLIMB.md, "The clock vote, measured"). Held from the open to
 * the free, the vote took the Googlebook 4.05 s, but an app keeps its
 * transcriber for the process's life, and the DSP would be kept from
 * sleeping through all of the app's idle time. Up is QAIRT 2.51's HTP
 * guide's burst: DCVS off, sleep disabled at a sleep latency of 40 us, the
 * bus and core at their highest corners. Down is the guide's released
 * state, which its lower profiles fall to between runs: DCVS on, a sleep
 * latency of 65535 us and the lowest corners, the clocks DCVS's and the
 * DSP's other clients' again. The guide's burst also asks for an RPC
 * control latency of 100 us and polls for each run's end: neither made a
 * run faster, and the polling cost up to 1.6 s of system time a bench */
static void vote(pk_qnn *e, int up) {
    if (!e->power)
        return;
    /* zeros, then each its option: the SDK's
     * QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT means the same, but
     * leaves out the braces of its union's struct, which gcc's -Wall
     * warns of */
    QnnHtpPerfInfrastructure_PowerConfig_t dcvs = { 0 };
    dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    QnnHtpPerfInfrastructure_DcvsV3_t *v = &dcvs.dcvsV3Config;
    const QnnHtpPerfInfrastructure_VoltageCorner_t corner =
        up ? DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER : DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER;
    v->contextId = e->power;
    v->setDcvsEnable = 1, v->dcvsEnable = !up;
    v->powerMode = up ? QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE
                      : QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE;
    v->setSleepLatency = 1, v->sleepLatency = up ? 40 : 65535;
    v->setSleepDisable = 1, v->sleepDisable = up;
    v->setBusParams = 1;
    v->busVoltageCornerMin = v->busVoltageCornerTarget = v->busVoltageCornerMax = corner;
    v->setCoreParams = 1;
    v->coreVoltageCornerMin = v->coreVoltageCornerTarget = v->coreVoltageCornerMax = corner;
    const QnnHtpPerfInfrastructure_PowerConfig_t *cfg[] = { &dcvs, NULL };
    e->perf->setPowerConfig(e->power, cfg);
}

/* ------------------------------------------------------- the platform --- */

/* The chip QNN compiles for, where it is not the one QNN runs on: a PC's
 * QNN has no NPU to ask (deviceGetPlatformInfo is the phone's alone), so it
 * is told one - the SoC and its Hexagon, as Qualcomm's own
 * qnn-context-binary-generator tells it, and the chip's description as
 * the phone's QNN gives it - and the graph it compiles is kept in the
 * cache for that phone to read back (pk_qnn_open). */
typedef struct {
    QnnHtpDevice_DeviceInfoExtension_t ext;
    QnnDevice_CoreInfo_t core;
    QnnDevice_HardwareDeviceInfo_t hw;
    QnnDevice_PlatformInfo_t info;
    QnnHtpDevice_CustomConfig_t soc, arch;
    QnnDevice_Config_t cfg[3];
    const QnnDevice_Config_t *list[4];
} target;

/* PKTITO_QNN_TARGET=soc,arch,vtcm,dlbc,signed - QnnTypes.h's
 * QNN_SOC_MODEL_ number, the Hexagon version, its VTCM in MB, and whether
 * it has DLBC and a signed PD, as the phone's QNN gives them
 * (platform_log): 1 with t made from it, 0 without one, -1 for one that
 * doesn't parse */
static int target_from_env(target *t) {
    const char *v = getenv("PKTITO_QNN_TARGET");
    unsigned soc, arch, vtcm, dlbc, sig;
    if (!v)
        return 0;
    if (sscanf(v, "%u,%u,%u,%u,%u", &soc, &arch, &vtcm, &dlbc, &sig) != 5)
        return -1;
    memset(t, 0, sizeof *t);
    t->ext.devType = QNN_HTP_DEVICE_TYPE_ON_CHIP;
    t->ext.onChipDevice.vtcmSize = vtcm;
    t->ext.onChipDevice.socModel = soc;
    t->ext.onChipDevice.signedPdSupport = sig != 0;
    t->ext.onChipDevice.dlbcSupport = dlbc != 0;
    t->ext.onChipDevice.arch = (QnnHtpDevice_Arch_t)arch;
    t->core = (QnnDevice_CoreInfo_t)QNN_DEVICE_CORE_INFO_INIT;
    t->core.v1.coreId = 0;
    t->core.v1.coreType = QNN_HTP_CORE_TYPE_NSP;
    t->hw = (QnnDevice_HardwareDeviceInfo_t)QNN_DEVICE_HARDWARE_DEVICE_INFO_INIT;
    t->hw.v1.deviceId = 0;
    t->hw.v1.deviceType = QNN_HTP_DEVICE_TYPE_ON_CHIP;
    t->hw.v1.numCores = 1;
    t->hw.v1.cores = &t->core;
    t->hw.v1.deviceInfoExtension = (QnnDevice_DeviceInfoExtension_t)&t->ext;
    t->info = (QnnDevice_PlatformInfo_t)QNN_DEVICE_PLATFORM_INFO_INIT;
    t->info.version = QNN_DEVICE_PLATFORM_INFO_VERSION_1;
    t->info.v1.numHwDevices = 1;
    t->info.v1.hwDevices = &t->hw;
    t->soc.option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
    t->soc.socModel = soc;
    t->arch.option = QNN_HTP_DEVICE_CONFIG_OPTION_ARCH;
    t->arch.arch.deviceId = 0;
    t->arch.arch.arch = (QnnHtpDevice_Arch_t)arch;
    for (int k = 0; k < 3; k++)
        t->cfg[k] = (QnnDevice_Config_t)QNN_DEVICE_CONFIG_INIT;
    t->cfg[0].option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
    t->cfg[0].customConfig = &t->soc;
    t->cfg[1].option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
    t->cfg[1].customConfig = &t->arch;
    t->cfg[2].option = QNN_DEVICE_CONFIG_OPTION_PLATFORM_INFO;
    t->cfg[2].hardwareInfo = &t->info;
    for (int k = 0; k < 3; k++)
        t->list[k] = &t->cfg[k];
    t->list[3] = NULL;
    return 1;
}

/* An HTP on the chip in PKTITO_QNN_TARGET's words: soc,arch,vtcm,dlbc,signed */
static void target_words(const QnnHtpDevice_DeviceInfoExtension_t *x, char *out, size_t n) {
    snprintf(out, n, "%u,%u,%zu,%d,%d", x->onChipDevice.socModel, (unsigned)x->onChipDevice.arch,
             x->onChipDevice.vtcmSize, x->onChipDevice.dlbcSupport, x->onChipDevice.signedPdSupport);
}

/* PKTITO_QNN_LOG: the chip as its QNN says, a device a line on stderr,
 * with the PKTITO_QNN_TARGET that names it to a PC */
static void platform_log(const pk_qnn *e) {
    const QnnDevice_PlatformInfo_t *p = NULL;
    if (!getenv("PKTITO_QNN_LOG") || !e->q.deviceGetPlatformInfo || e->q.deviceGetPlatformInfo(e->log, &p) || !p)
        return;
    for (uint32_t i = 0; i < p->v1.numHwDevices; i++) {
        const QnnDevice_HardwareDeviceInfoV1_t *h = &p->v1.hwDevices[i].v1;
        const QnnHtpDevice_DeviceInfoExtension_t *x = (const QnnHtpDevice_DeviceInfoExtension_t *)h->deviceInfoExtension;
        fprintf(stderr, "qnn: device %u, type %u, %u cores", h->deviceId, h->deviceType, h->numCores);
        for (uint32_t c = 0; c < h->numCores; c++)
            fprintf(stderr, "%s%u (type %u)", c ? ", " : ": ", h->cores[c].v1.coreId, h->cores[c].v1.coreType);
        if (x && x->devType == QNN_HTP_DEVICE_TYPE_ON_CHIP) {
            char w[64];
            target_words(x, w, sizeof w);
            fprintf(stderr, "; PKTITO_QNN_TARGET=%s", w);
        }
        fputc('\n', stderr);
    }
    if (e->q.deviceFreePlatformInfo)
        e->q.deviceFreePlatformInfo(e->log, p);
}

/* The NPU QNN runs on, in target_words' words, into out: 0; -1 where
 * QNN describes no chip (a PC's: target, above), -2 where the chip it
 * describes has no NPU */
static int own_chip(pk_qnn *e, char *out, size_t n) {
    const QnnDevice_PlatformInfo_t *p = NULL;
    if (!e->q.deviceGetPlatformInfo || e->q.deviceGetPlatformInfo(e->log, &p) || !p)
        return -1;
    int rc = -2;
    for (uint32_t i = 0; rc && i < p->v1.numHwDevices; i++) {
        const QnnHtpDevice_DeviceInfoExtension_t *x =
            (const QnnHtpDevice_DeviceInfoExtension_t *)p->v1.hwDevices[i].v1.deviceInfoExtension;
        if (x && x->devType == QNN_HTP_DEVICE_TYPE_ON_CHIP) {
            target_words(x, out, n);
            rc = 0;
        }
    }
    if (e->q.deviceFreePlatformInfo)
        e->q.deviceFreePlatformInfo(e->log, p);
    return rc;
}

#ifdef _WIN32
/* why load_htp loaded no QnnHtp.dll, for dl_why */
static _Thread_local char htp_why[3 * MAX_PATH + 64];

/* QnnHtp.dll from dir, whose first n characters are a folder and its
 * backslash; one that is there but won't load says why in htp_why */
static HMODULE htp_in(wchar_t *dir, size_t n) {
    if (n + sizeof "QnnHtp.dll" > MAX_PATH)
        return NULL;
    wcscpy(dir + n, L"QnnHtp.dll");
    HMODULE h = LoadLibraryExW(dir, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    const DWORD why = GetLastError();
    char u[3 * MAX_PATH];
    if (!h && !htp_why[0] && GetFileAttributesW(dir) != INVALID_FILE_ATTRIBUTES &&
        WideCharToMultiByte(CP_UTF8, 0, dir, -1, u, sizeof u, NULL, NULL))
        snprintf(htp_why, sizeof htp_why, "%s did not load (error %lu)", u, (unsigned long)why);
    return h;
}

/* QnnHtp.dll where Windows finds a program's own DLLs, but not in the
 * current folder: beside the module this code is in (libpktito.dll, for
 * Python's ctypes or Node's koffi, or the program that links libpktito.a);
 * the program's folder, System32 or one the program added; PATH's folders
 * named in full, as for a program go run or cargo run builds elsewhere
 * (examples/README.md). Its stub, skel and .cat go beside it */
static HMODULE load_htp(void) {
    static const char here = 0;
    HMODULE self = NULL, h = NULL;
    wchar_t dir[MAX_PATH + 1], *slash;
    char u[3 * MAX_PATH] = "this program";
    htp_why[0] = 0;
    DWORD n = GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                 (LPCWSTR)&here, &self)
                  ? GetModuleFileNameW(self, dir, MAX_PATH)
                  : 0;
    if (n && n < MAX_PATH && (slash = wcsrchr(dir, L'\\'))) {
        WideCharToMultiByte(CP_UTF8, 0, dir, -1, u, sizeof u, NULL, NULL);
        h = htp_in(dir, (size_t)(slash + 1 - dir));
    }
    if (!h && !(h = LoadLibraryExW(L"QnnHtp.dll", NULL, LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)) &&
        GetLastError() != ERROR_MOD_NOT_FOUND && !htp_why[0])
        snprintf(htp_why, sizeof htp_why, "QnnHtp.dll did not load (error %lu)", (unsigned long)GetLastError());
    n = h ? 0 : GetEnvironmentVariableW(L"PATH", NULL, 0);
    wchar_t *path = n ? malloc(n * sizeof *path) : NULL;
    if (path && GetEnvironmentVariableW(L"PATH", path, n) < n)
        for (wchar_t *d = path, *end; !h && *d; d = *end ? end + 1 : end) {
            end = d + wcscspn(d, L";");
            size_t k = (size_t)(end - d);
            /* a relative folder is the current folder's */
            if (k < 3 || k >= MAX_PATH || !((d[1] == L':' && (d[2] == L'\\' || d[2] == L'/')) || (d[0] == L'\\' && d[1] == L'\\')))
                continue;
            for (size_t i = 0; i < k; i++)
                dir[i] = d[i] == L'/' ? L'\\' : d[i];
            if (dir[k - 1] != L'\\')
                dir[k++] = L'\\';
            h = htp_in(dir, k);
        }
    free(path);
    if (!h && !htp_why[0])
        snprintf(htp_why, sizeof htp_why, "QnnHtp.dll is not beside %s%s or on PATH", u,
                 self != GetModuleHandleW(NULL) ? " or the program," : "");
    return h;
}
#endif

/* why the HTP library didn't load */
static const char *dl_why(void) {
#ifdef _WIN32
    return htp_why;
#else
    return dlerror();
#endif
}

/* QNN's HTP library loaded, and its log, backend and device made: for the
 * chip QNN runs on, or with t for the one t describes. 0, or -1 with why
 * in err */
static int open_device(pk_qnn *e, target *t, char *err, size_t errlen) {
    last_error[0] = 0;
    /* Windows on Snapdragon: QnnHtp.dll (load_htp) and beside it, for each
     * Hexagon the program serves, its stub and the DSP's skel and .cat (a
     * V73's QnnHtpV73Stub.dll, libQnnHtpV73Skel.so and libqnnhtpv73.cat) */
#ifdef _WIN32
    e->lib = (void *)load_htp();
    get_providers_fn gp = e->lib ? (get_providers_fn)(void (*)(void))GetProcAddress((HMODULE)e->lib, "QnnInterface_getProviders") : NULL;
#else
    e->lib = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
    get_providers_fn gp = e->lib ? (get_providers_fn)dlsym(e->lib, "QnnInterface_getProviders") : NULL;
#endif
    const QnnInterface_t **prov = NULL;
    uint32_t np = 0;
    if (!gp || gp(&prov, &np) || np == 0 || prov[0]->apiVersion.coreApiVersion.major != QNN_API_VERSION_MAJOR) {
        snprintf(err, errlen, "no QNN HTP library: %s", e->lib ? "no provider of QNN 2" : dl_why());
        return -1;
    }
    e->q = prov[0]->QNN_INTERFACE_VER_NAME;
    Qnn_ErrorHandle_t s;
    /* QNN makes no message past the level it is given: its warnings, or
     * as far as PKTITO_QNN_LOG asks, up to its debug messages */
    const char *v = getenv("PKTITO_QNN_LOG");
    int level = v ? atoi(v) : 0;
    level = level < QNN_LOG_LEVEL_WARN ? QNN_LOG_LEVEL_WARN : level > QNN_LOG_LEVEL_DEBUG ? QNN_LOG_LEVEL_DEBUG : level;
    if (e->q.logCreate(log_cb, (QnnLog_Level_t)level, &e->log))
        e->log = NULL;
    if ((s = e->q.backendCreate(e->log, NULL, &e->backend)))
        return fail(err, errlen, "backendCreate", s);
    /* a chip named to compile for is a PC's to be told, which has no NPU
     * of its own to ask (PKTITO_QNN_TARGET): a phone runs its own. An x86
     * build is a PC's, though its QNN describes a chip: the one it emulates
     * (30,68,4,0,0 in QNN 2.51), which no graph is compiled for */
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    const int pc = 1;
#else
    const int pc = 0;
#endif
    char own[32];
    if (t && !pc && own_chip(e, own, sizeof own) == 0) {
        snprintf(err, errlen, "PKTITO_QNN_TARGET names a chip for a PC to compile for, and this one has an NPU, %s", own);
        return -1;
    }
    if ((s = e->q.deviceCreate(e->log, t ? t->list : NULL, &e->device)))
        return fail(err, errlen, "deviceCreate", s);
    return 0;
}

int pk_qnn_target(char *out, size_t n, char *err, size_t errlen) {
    pk_qnn *e = calloc(1, sizeof *e);
    if (!e) {
        snprintf(err, errlen, "out of memory");
        return -1;
    }
    int rc = open_device(e, NULL, err, errlen);
    if (!rc && (rc = own_chip(e, out, n)))
        snprintf(err, errlen, rc == -1 ? "QNN describes no chip here" : "QNN finds no NPU on the chip");
    pk_qnn_free(e);
    return rc ? -1 : 0;
}

/* The graph made and compiled, into a context of its own, and kept in
 * cache where there is one: 0, or -1 with why in err */
static int compile(pk_qnn *e, const pk_model *m, const char *cache, char *err, size_t errlen) {
    Qnn_ErrorHandle_t s;
    last_error[0] = 0;           /* not a cache's refusal (read_cache) */
    if ((s = e->q.contextCreate(e->backend, e->device, NULL, &e->context)))
        return fail(err, errlen, "contextCreate", s);
    /* the compiler's highest optimization; the HTP's fp16 sigmoid of high
     * precision (SiLU's, GLU's); and at O=3 the compiler's P point 13, one
     * of the "pre-defined point[s] in the compiler's internal configuration
     * space" QNN's HTP guide leaves to be measured. With it the SM8750's
     * chunk is 94.7 ms to 134.0 without, the SM8650's 110.3 to 111.8 and the
     * X1E80100's 172.7 to 172.2, and test.wav's numbers are the same to
     * every digit (tools/qnn/CLIMB.md, 2026-11-03). What a P point does can
     * change from one QNN to the next: a new QAIRT measures them again. */
    QnnHtpGraph_CustomConfig_t opt = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT, sig = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT,
                               pp = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
    opt.option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
    opt.optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
    opt.optimizationOption.floatValue = 3;
    sig.option = QNN_HTP_GRAPH_CONFIG_OPTION_HIGH_PRECISION_SIGMOID;
    sig.highPrecisionSigmoid = true;
    pp.option = QNN_HTP_GRAPH_CONFIG_OPTION_FINALIZE_CONFIG;
    pp.finalizeConfig.key = "P";
    pp.finalizeConfig.value.dataType = QNN_DATATYPE_UINT_32;
    pp.finalizeConfig.value.uint32Value = 13;
    QnnGraph_Config_t g_opt = { .option = QNN_GRAPH_CONFIG_OPTION_CUSTOM, .customConfig = &opt };
    QnnGraph_Config_t g_sig = { .option = QNN_GRAPH_CONFIG_OPTION_CUSTOM, .customConfig = &sig };
    QnnGraph_Config_t g_pp = { .option = QNN_GRAPH_CONFIG_OPTION_CUSTOM, .customConfig = &pp };
    const QnnGraph_Config_t *gcfg[] = { &g_opt, &g_sig, &g_pp, NULL };
    if ((s = e->q.graphCreate(e->context, "encoder", gcfg, &e->graph)))
        return fail(err, errlen, "graphCreate", s);
    if (make_graph(e, m, err, errlen))
        return -1;
    if (cache)
        write_cache(e, m, cache);
    return 0;
}

pk_qnn *pk_qnn_open(const pk_model *m, uint32_t frames, const char *cache, char *err, size_t errlen) {
    /* a slim model has no weights to make a graph of: its graph is the
     * full file's, compiled ahead of time and read from cache */
    if (m->encoder_coreml && !cache) {
        snprintf(err, errlen, "the model leaves its encoder out: give its compiled graph's file");
        return NULL;
    }
    if (m->n_heads > 16) {
        snprintf(err, errlen, "the graph is made for 16 heads at most, not %u", m->n_heads);
        return NULL;
    }
    pk_qnn *e = calloc(1, sizeof *e);
    if (!e) {
        snprintf(err, errlen, "out of memory");
        return NULL;
    }
    lengths(e, m, frames);
    /* PKTITO_QNN_LAYERS=k: the graph through layer k only (0, the
     * subsampling), its stream the output, to find where the NPU's
     * numbers part from the CPU's (encoder.c's PKTITO_QNN_CHECK) */
    const char *nl = getenv("PKTITO_QNN_LAYERS");
    if (nl) {
        char *end = (char *)nl;
        unsigned long k = 0;
        if (nl[0] >= '0' && nl[0] <= '9')
            k = strtoul(nl, &end, 10);
        if (end == nl || *end || k > m->n_layers) {
            snprintf(err, errlen, "PKTITO_QNN_LAYERS is a number of the encoder's layers, 0 to its %u", m->n_layers);
            pk_qnn_free(e);
            return NULL;
        }
        e->layers = (uint32_t)k;
    }
    if (m->encoder_coreml && e->layers < m->n_layers) {
        snprintf(err, errlen, "PKTITO_QNN_LAYERS is checked against the CPU's encoder, which the model leaves out");
        pk_qnn_free(e);
        return NULL;
    }
    target t;
    const int other = target_from_env(&t);
    if (other < 0) {
        snprintf(err, errlen, "PKTITO_QNN_TARGET is soc,arch,vtcm,dlbc,signed, as a phone's PKTITO_QNN_LOG gives it");
        pk_qnn_free(e);
        return NULL;
    }
    if (open_device(e, other ? &t : NULL, err, errlen)) {
        pk_qnn_free(e);
        return NULL;
    }
    e->other = other;
    /* the NPU the graph is for: the one named, else this one */
    if (other)
        target_words(&t.ext, e->chip, sizeof e->chip);
    else if (own_chip(e, e->chip, sizeof e->chip))
        e->chip[0] = 0;
    if (!other) {
        platform_log(e);
        power(e);
    }
    if (!cache || read_cache(e, m, cache, err, errlen))
        if (m->encoder_coreml || compile(e, m, cache, err, errlen)) {
            pk_qnn_free(e);
            return NULL;
        }
    /* the inputs' and output's buffers, of their shapes' sizes */
    for (int k = 0; k < 6; k++) {
        uint32_t d[4];
        const size_t n = count(io_shape(e, k, d), d);
        float **p = k < 5 ? &e->buf[k] : &e->x;
        *p = calloc(n, sizeof(float));
        io(e, k)->v1.clientBuf.data = *p;
        io(e, k)->v1.clientBuf.dataSize = (uint32_t)(n * sizeof(float));
    }
    for (int k = 0; k < 5; k++)
        if (!e->buf[k] || !e->x) {
            snprintf(err, errlen, "out of memory");
            pk_qnn_free(e);
            return NULL;
        }
    return e;
}

void pk_qnn_free(pk_qnn *e) {
    if (!e)
        return;
    if (e->context)
        e->q.contextFree(e->context, NULL);
    if (e->power)
        e->perf->destroyPowerConfigId(e->power);
    if (e->device)
        e->q.deviceFree(e->device);
    if (e->backend)
        e->q.backendFree(e->backend);
    if (e->log)
        e->q.logFree(e->log);
    for (int k = 0; k < 5; k++)
        free(e->buf[k]);
    free(e->x);
    /* libQnnHtp.so stays loaded: QNN's documentation has it loaded once a
     * process */
    free(e);
}

uint32_t pk_qnn_frames(const pk_qnn *e) { return e->T0; }

uint32_t pk_qnn_layers(const pk_qnn *e) { return e->layers; }

uint32_t pk_qnn_encode(pk_qnn *e, const float *feat, uint32_t T0, float *x) {
    /* a graph for another chip is the cache's alone: here the CPU's
     * encoder runs (an x86 QNN would emulate the NPU's, at a fifth of real
     * time on a Ryzen 9 9950X) */
    if (T0 == 0 || T0 > e->T0 || e->other)
        return 0;
    const uint32_t T1 = (T0 + 1) / 2, T2 = (T1 + 1) / 2, T3 = (T2 + 1) / 2;
    memcpy(e->buf[0], feat, (size_t)T0 * e->n_mels * sizeof(float));
    memset(e->buf[0] + (size_t)T0 * e->n_mels, 0, (size_t)(e->T0 - T0) * e->n_mels * sizeof(float));
    for (uint32_t t = 0; t < e->T1; t++)
        e->buf[1][t] = t < T1;
    for (uint32_t t = 0; t < e->T2; t++)
        e->buf[2][t] = t < T2;
    for (uint32_t t = 0; t < e->T3; t++) {
        e->buf[3][t] = t < T3;
        e->buf[4][t] = t < T3 ? 0.0f : -10000.0f;
    }
    Qnn_ProfileHandle_t prof = profile(e, 0);
    vote(e, 1);
    Qnn_ErrorHandle_t s = e->q.graphExecute(e->graph, e->in, 5, &e->out, 1, prof, NULL);
    vote(e, 0);
    profile_done(e, prof);
    if (s)
        return 0;
    memcpy(x, e->x, (size_t)T3 * e->d * sizeof(float));
    return T3;
}
