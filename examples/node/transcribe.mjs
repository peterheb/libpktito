// The library from Node, through koffi: examples/transcribe.c line for line,
// a WAV or AIFF file to its transcript, cut at its silences when it is long.
//
//     make shared
//     cd examples/node && npm install
//     node transcribe.mjs ../../models/parakeet-tdt-0.6b-v2.pkf talk.wav
//
// prints each chunk as "start end text", the times in seconds. A 16 kHz file
// only, as there. Node has no foreign-function interface of its own; koffi
// is one, and reads C prototypes as written in core/pktito.h and core/wav.h.
// The library is the one `make shared` leaves at the top of the tree, or the
// file PKTITO_LIB names.
//
// An example of the wiring, not a package. `make check-ffi` holds its
// transcript to ./pktito-c's.
import koffi from 'koffi';
import { fileURLToPath } from 'node:url';

const name = { darwin: 'libpktito.dylib', win32: 'libpktito.dll' }[process.platform] ?? 'libpktito.so';
const lib = koffi.load(process.env.PKTITO_LIB ?? fileURLToPath(new URL(`../../${name}`, import.meta.url)));

// The handles are opaque, and the structs field for field as the headers
// declare them: koffi lays a struct out as the C compiler does. _Out_ marks
// a pointer the call writes through: koffi hands back what it wrote, into
// the object or one-element array passed for it.
koffi.opaque('pk_model');
koffi.opaque('pk_asr');
koffi.opaque('pk_vad');
koffi.opaque('pk_wav');
koffi.struct('pk_wav_info', {
    rate: 'uint32_t', channels: 'uint32_t', format: 'uint32_t', bytes: 'uint32_t', frames: 'uint64_t',
});
koffi.struct('pk_seg_params', {
    threshold: 'double', neg_threshold: 'double', min_speech_ms: 'double', max_speech_s: 'double',
    min_silence_ms: 'double', speech_pad_ms: 'double', min_silence_at_max_speech_ms: 'double',
    use_max_poss_sil_at_max_speech: 'int',
});
koffi.struct('pk_tok', { id: 'int32_t', frame: 'int32_t', dur: 'int32_t' });

// A pk_seg is two int64_t, start and end, so an array of them is passed as
// int64_t * over a BigInt64Array: the same bytes, read in place. Samples and
// probabilities are Float32Arrays, passed as float *; a subarray passes its
// part of the memory, not a copy.
const pk = {
    model_load: lib.func('pk_model *pk_model_load(const char *path, char *err, size_t errlen)'),
    model_rate: lib.func('int pk_model_rate(const pk_model *m)'),
    model_free: lib.func('void pk_model_free(pk_model *m)'),
    wav_open: lib.func('pk_wav *pk_wav_open(const char *path, _Out_ pk_wav_info *info, char *err, size_t errlen)'),
    wav_read: lib.func('size_t pk_wav_read(pk_wav *w, float *out, size_t n)'),
    wav_close: lib.func('void pk_wav_close(pk_wav *w)'),
    vad_new: lib.func('pk_vad *pk_vad_new(const pk_model *m)'),
    vad_probs: lib.func('void pk_vad_probs(pk_vad *v, const float *x, size_t n, float *probs)'),
    vad_free: lib.func('void pk_vad_free(pk_vad *v)'),
    seg_defaults: lib.func('void pk_seg_defaults(_Out_ pk_seg_params *p)'),
    segments: lib.func('size_t pk_segments(const float *probs, size_t n, int rate, int64_t n_samples, ' +
                       'const pk_seg_params *p, int64_t *out, size_t cap)'),
    chunks: lib.func('size_t pk_chunks(const float *x, const int64_t *speech, size_t n_speech, ' +
                     'int64_t total, int64_t max_len, int64_t *out)'),
    asr_new: lib.func('pk_asr *pk_asr_new(const pk_model *m, int threads)'),
    asr_run: lib.func('long pk_asr_run(pk_asr *a, const float *x, size_t n, _Out_ const pk_tok **toks)'),
    asr_free: lib.func('void pk_asr_free(pk_asr *a)'),
    text: lib.func('size_t pk_text(const pk_model *m, const pk_tok *t, size_t n, char *out, size_t cap)'),
};
const PK_VAD_SAMPLES = 512;

// a NUL-terminated string in a buffer, as UTF-8
const cstr = (buf) => buf.toString('utf8', 0, buf.indexOf(0));

function main(argv) {
    if (argv.length !== 2) {
        console.error('usage: transcribe.mjs MODEL.pkf FILE.wav');
        return 2;
    }
    // The library's paths are UTF-8 on every system, as koffi passes a string.
    const err = Buffer.alloc(256);
    const m = pk.model_load(argv[0], err, err.length);
    if (!m) {
        console.error(cstr(err));
        return 1;
    }
    const rate = pk.model_rate(m);

    // the whole recording, as float in [-1, 1]
    const info = {};
    const w = pk.wav_open(argv[1], info, err, err.length);
    if (!w || info.rate !== rate) {
        console.error(w ? 'want a 16 kHz file' : cstr(err));
        return 1;
    }
    const blocks = [];
    let n = 0;
    for (;;) {
        const block = new Float32Array(1 << 20);
        const k = pk.wav_read(w, block, block.length);
        if (k === 0) break;
        blocks.push(block.subarray(0, k));
        n += k;
    }
    pk.wav_close(w);
    const x = new Float32Array(n);
    blocks.reduce((at, b) => (x.set(b, at), at + b.length), 0);

    // Silero's speech segments, then chunks of at most 30 s cut between
    // them, the silence left out; a recording that fits is one chunk
    const frames = Math.ceil(n / PK_VAD_SAMPLES);
    const padded = new Float32Array(frames * PK_VAD_SAMPLES);
    padded.set(x);
    const probs = new Float32Array(frames);
    const seg = new BigInt64Array(2 * (2 * frames + 1));
    const v = pk.vad_new(m);
    pk.vad_probs(v, padded, frames, probs);
    const sp = {};
    pk.seg_defaults(sp);
    sp.max_speech_s = 30 - 0.06 - 0.1;  // a segment and Silero's 30 ms pads fit in 30 s, as ./pktito-c has it
    const ns = pk.segments(probs, frames, rate, n, sp, seg, seg.length / 2);
    const chunk = new BigInt64Array(2 * (2 * ns + 2 + Math.floor(n / (30 * rate))));
    const nc = pk.chunks(x, seg, ns, n, 30 * rate, chunk);

    // each chunk through the transcriber, as NeMo would hear it as a file.
    // A call holds Node's thread until it returns, here as long as the
    // chunk takes; a server would make it with koffi's async form instead,
    // pk.asr_run.async(...), which runs it on a worker thread.
    const a = pk.asr_new(m, 0);
    const t = [null];
    const text = Buffer.alloc(65536);
    for (let i = 0; i < nc; i++) {
        const start = Number(chunk[2 * i]), end = Number(chunk[2 * i + 1]);
        const nt = pk.asr_run(a, x.subarray(start, end), end - start, t);
        if (nt < 0) return 1;
        pk.text(m, t[0], nt, text, text.length);
        process.stdout.write(`${(start / rate).toFixed(2)} ${(end / rate).toFixed(2)} ${cstr(text)}\n`);
    }
    pk.asr_free(a);
    pk.vad_free(v);
    pk.model_free(m);
    return 0;
}

process.exitCode = main(process.argv.slice(2));
