"""The library from Python, through ctypes and nothing else: examples/transcribe.c
line for line, a WAV or AIFF file to its transcript, cut at its silences when
it is long.

    make shared
    python3 examples/python/transcribe.py models/parakeet-tdt-0.6b-v2.pkf talk.wav

prints each chunk as "start end text", the times in seconds. A 16 kHz file
only, as there. The library is the one `make shared` leaves at the top of the
tree, or the file PKTITO_LIB names; Python 3.8 or later, and its standard
library.

An example of the wiring, not a package: it gives the functions it calls
their C types, from core/pktito.h and core/wav.h, and a program of your own
does the same for the ones it calls. `make check-ffi` holds its transcript to
./pktito-c's.
"""
import ctypes
import os
import sys
from array import array
from ctypes import (POINTER, Structure, byref, c_char_p, c_double, c_float, c_int, c_int32,
                    c_int64, c_long, c_size_t, c_uint32, c_uint64, c_void_p)
from pathlib import Path

PK_VAD_SAMPLES = 512


# The structs the calls pass, field for field as the headers declare them:
# ctypes lays a Structure out as the C compiler does, padding and all.
class pk_wav_info(Structure):
    _fields_ = [("rate", c_uint32), ("channels", c_uint32), ("format", c_uint32),
                ("bytes", c_uint32), ("frames", c_uint64)]


class pk_seg_params(Structure):
    _fields_ = [("threshold", c_double), ("neg_threshold", c_double),
                ("min_speech_ms", c_double), ("max_speech_s", c_double),
                ("min_silence_ms", c_double), ("speech_pad_ms", c_double),
                ("min_silence_at_max_speech_ms", c_double),
                ("use_max_poss_sil_at_max_speech", c_int)]


class pk_seg(Structure):
    _fields_ = [("start", c_int64), ("end", c_int64)]


class pk_tok(Structure):
    _fields_ = [("id", c_int32), ("frame", c_int32), ("dur", c_int32)]


def load():
    """The library, each function this program calls given its C types.
    Untold, ctypes takes every result for an int, which keeps 32 bits of a
    pointer, and passes a Python int as an int, which a size_t or int64_t
    argument reads garbage beside. The opaque handles (pk_model *, pk_asr *,
    ...) are c_void_p: None when C returns NULL."""
    name = {"darwin": "libpktito.dylib", "win32": "libpktito.dll"}.get(sys.platform, "libpktito.so")
    lib = ctypes.CDLL(os.environ.get("PKTITO_LIB") or str(Path(__file__).resolve().parents[2] / name))
    floats, segs = POINTER(c_float), POINTER(pk_seg)
    for fn, restype, argtypes in [
        ("pk_model_load",   c_void_p, [c_char_p, c_char_p, c_size_t]),
        ("pk_model_rate",   c_int,    [c_void_p]),
        ("pk_model_free",   None,     [c_void_p]),
        ("pk_wav_open",     c_void_p, [c_char_p, POINTER(pk_wav_info), c_char_p, c_size_t]),
        ("pk_wav_read",     c_size_t, [c_void_p, floats, c_size_t]),
        ("pk_wav_close",    None,     [c_void_p]),
        ("pk_vad_new",      c_void_p, [c_void_p]),
        ("pk_vad_probs",    None,     [c_void_p, floats, c_size_t, floats]),
        ("pk_vad_free",     None,     [c_void_p]),
        ("pk_seg_defaults", None,     [POINTER(pk_seg_params)]),
        ("pk_segments",     c_size_t, [floats, c_size_t, c_int, c_int64, POINTER(pk_seg_params), segs, c_size_t]),
        ("pk_chunks",       c_size_t, [floats, segs, c_size_t, c_int64, c_int64, segs]),
        ("pk_asr_new",      c_void_p, [c_void_p, c_int]),
        ("pk_asr_run",      c_long,   [c_void_p, floats, c_size_t, POINTER(POINTER(pk_tok))]),
        ("pk_asr_free",     None,     [c_void_p]),
        ("pk_text",         c_size_t, [c_void_p, POINTER(pk_tok), c_size_t, c_char_p, c_size_t]),
    ]:
        f = getattr(lib, fn)
        f.restype, f.argtypes = restype, argtypes
    return lib


def view(a, start=0, end=None):
    """Floats [start, end) of an array('f') as C's float *: its memory, not a copy."""
    end = len(a) if end is None else end
    return (c_float * (end - start)).from_buffer(a, start * a.itemsize)


def main(argv):
    if len(argv) != 3:
        print("usage: transcribe.py MODEL.pkf FILE.wav", file=sys.stderr)
        return 2
    pk = load()
    # The library's paths are UTF-8 on every system, which os.fsencode
    # gives on Windows, and argv reaches Python whole there, unlike C's.
    err = ctypes.create_string_buffer(256)
    m = pk.pk_model_load(os.fsencode(argv[1]), err, len(err))
    if not m:
        print(err.value.decode(), file=sys.stderr)
        return 1
    rate = pk.pk_model_rate(m)

    # the whole recording, as float in [-1, 1]
    info = pk_wav_info()
    w = pk.pk_wav_open(os.fsencode(argv[2]), byref(info), err, len(err))
    if not w or info.rate != rate:
        print("want a 16 kHz file" if w else err.value.decode(), file=sys.stderr)
        return 1
    x, block = array("f"), (c_float * (1 << 20))()
    while (k := pk.pk_wav_read(w, block, len(block))) > 0:
        x.frombytes(memoryview(block).cast("B")[: k * x.itemsize])
    pk.pk_wav_close(w)
    n = len(x)

    # Silero's speech segments, then chunks of at most 30 s cut between
    # them, the silence left out; a recording that fits is one chunk
    frames = (n + PK_VAD_SAMPLES - 1) // PK_VAD_SAMPLES
    padded = x + array("f", [0.0] * (frames * PK_VAD_SAMPLES - n))
    probs = (c_float * frames)()
    seg = (pk_seg * (2 * frames + 1))()
    v = pk.pk_vad_new(m)
    pk.pk_vad_probs(v, view(padded), frames, probs)
    sp = pk_seg_params()
    pk.pk_seg_defaults(byref(sp))
    sp.max_speech_s = 30 - 0.06 - 0.1  # a segment and Silero's 30 ms pads fit in 30 s, as ./pktito-c has it
    ns = pk.pk_segments(probs, frames, rate, n, byref(sp), seg, len(seg))
    chunk = (pk_seg * (2 * ns + 2 + n // (30 * rate)))()
    nc = pk.pk_chunks(view(x), seg, ns, n, 30 * rate, chunk)

    # each chunk through the transcriber, as NeMo would hear it as a file;
    # ctypes lets go of the GIL for each call, and the transcriber runs its
    # own threads, so other Python threads go on meanwhile
    a = pk.pk_asr_new(m, 0)
    t = POINTER(pk_tok)()
    text = ctypes.create_string_buffer(65536)
    sys.stdout.reconfigure(encoding="utf-8")  # the text's bytes, as C prints them, whatever the code page
    for c in chunk[:nc]:
        nt = pk.pk_asr_run(a, view(x, c.start, c.end), c.end - c.start, byref(t))
        if nt < 0:
            return 1
        pk.pk_text(m, t, nt, text, len(text))
        print(f"{c.start / rate:.2f} {c.end / rate:.2f} {text.value.decode()}")
    pk.pk_asr_free(a)
    pk.pk_vad_free(v)
    pk.pk_model_free(m)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
