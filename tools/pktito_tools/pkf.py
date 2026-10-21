"""pktito's model file (.pkf): core/pkf.h's structs, written and read back.

core/pkf.h is the definition; this mirrors it. A change to one is a change
to the other in the same commit, and the struct formats below are checked
against the sizes the C asserts.
"""
import hashlib
import struct

import numpy as np

MAGIC = 0x01464B50                  # "PKF\1"
VERSION = 1
ALIGN = 64
NAME, KEY = 48, 32
QR, QG = 16, 4                      # a q8 panel's rows, a group's inputs
F32, Q8 = 0, 1
TYPE_NAMES = {F32: "f32", Q8: "q8"}
PARAM_U32, PARAM_I32, PARAM_F32 = 0, 1, 2

HEADER = struct.Struct("<II32s64sII8Q32s48s")
PARAM = struct.Struct("<32sI4s")
TENSOR = struct.Struct("<48sII2IQQ2Q")
assert HEADER.size == 256 and PARAM.size == 40 and TENSOR.size == 96

PIECE_TYPES = {1: "normal", 2: "unknown", 3: "control", 4: "user", 5: "unused"}


def _align(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


def quantize_q8(m):
    """A float matrix [rows][cols] as PK_Q8: int8 in core/pkf.h's panel
    layout, and a float32 scale a row. Each row's scale is its largest
    magnitude over 127, in float64 and rounded once; q = rint(w / scale)."""
    m = np.asarray(m, np.float64)
    rows, cols = m.shape
    if rows % QR or cols % 32:
        raise ValueError("a q8 matrix is a multiple of 16 rows and 32 columns, not %dx%d" % (rows, cols))
    s = np.abs(m).max(1) / 127
    q = np.rint(m / np.where(s == 0, 1, s)[:, None])
    q = np.clip(q, -127, 127).astype(np.int8)
    panels = q.reshape(rows // QR, QR, cols // QG, QG).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(panels).reshape(rows, cols), s.astype(np.float32)


def dequantize_q8(data, scale):
    """The float32 matrix a PK_Q8 tensor stands for."""
    rows, cols = data.shape
    q = data.reshape(rows // QR, cols // QG, QR, QG).transpose(0, 2, 1, 3).reshape(rows, cols)
    return q.astype(np.float32) * scale[:, None]


def vocab_blob(pieces, types, punct):
    """core/pkf.h's vocabulary: the count, the offsets, the types, the text,
    and the punctuation NeMo strips the space before."""
    text = [p.encode("utf-8") for p in pieces]
    off = np.zeros(len(text) + 1, np.uint32)
    off[1:] = np.cumsum([len(t) for t in text])
    pb = punct.encode("utf-8")
    return (struct.pack("<I", len(text)) + off.tobytes() + bytes(types) + b"".join(text)
            + struct.pack("<I", len(pb)) + pb)


def read_vocab(blob):
    n = struct.unpack_from("<I", blob, 0)[0]
    off = np.frombuffer(blob, np.uint32, n + 1, 4)
    types = list(blob[4 + 4 * (n + 1):4 + 4 * (n + 1) + n])
    t0 = 4 + 4 * (n + 1) + n
    pieces = [blob[t0 + off[i]:t0 + off[i + 1]].decode("utf-8") for i in range(n)]
    p0 = t0 + int(off[n])
    np_ = struct.unpack_from("<I", blob, p0)[0]
    return pieces, types, blob[p0 + 4:p0 + 4 + np_].decode("utf-8")


def meta_blob(meta):
    out = bytearray()
    for k, v in meta.items():
        k, v = str(k).encode("utf-8"), str(v).encode("utf-8")
        if not k or b"\0" in k or b"\0" in v:
            raise ValueError("metadata %r: keys are non-empty and neither has a NUL" % k)
        out += k + b"\0" + v + b"\0"
    return bytes(out)


def write(path, arch, name, params, tensors, vocab, meta):
    """params: [(key, value)] with ints as u32 (or i32 when negative) and
    floats as f32; tensors: [(name, type, dims, numpy array)] in file order."""
    pblob = bytearray()
    for k, v in params:
        if isinstance(v, float):
            pblob += PARAM.pack(k.encode(), PARAM_F32, struct.pack("<f", v))
        elif v < 0:
            pblob += PARAM.pack(k.encode(), PARAM_I32, struct.pack("<i", v))
        else:
            pblob += PARAM.pack(k.encode(), PARAM_U32, struct.pack("<I", v))
    mblob = meta_blob(meta)
    params_off = HEADER.size
    tensors_off = params_off + len(pblob)
    vocab_off = tensors_off + TENSOR.size * len(tensors)
    meta_off = vocab_off + len(vocab)
    data_off = _align(meta_off + len(mblob))
    descs, payload, at = bytearray(), [], data_off
    for tname, typ, dims, arr in tensors:
        raw = np.ascontiguousarray(arr).tobytes()
        if len(tname.encode()) >= NAME:
            raise ValueError("tensor name %r is too long" % tname)
        ndim = len(dims)
        d = list(dims) + [1] * (2 - ndim)
        want = d[0] * d[1] * (4 if typ == F32 else 1)
        if len(raw) != want:
            raise ValueError("%s: %d bytes for %s %s" % (tname, len(raw), TYPE_NAMES[typ], dims))
        descs += TENSOR.pack(tname.encode(), typ, ndim, d[0], d[1], at, len(raw), 0, 0)
        payload.append((at, raw))
        at = _align(at + len(raw))
    size = payload[-1][0] + len(payload[-1][1]) if payload else data_off
    sha = hashlib.sha256()
    pos = data_off
    for off, raw in payload:
        sha.update(b"\0" * (off - pos))
        sha.update(raw)
        pos = off + len(raw)
    head = HEADER.pack(MAGIC, VERSION, arch.encode(), name.encode(), len(params), len(tensors),
                       params_off, tensors_off, vocab_off, len(vocab), meta_off, len(mblob),
                       data_off, size, sha.digest(), b"")
    with open(path, "wb") as f:
        f.write(head + pblob + descs + vocab + mblob)
        f.write(b"\0" * (data_off - f.tell()))
        for off, raw in payload:
            f.write(b"\0" * (off - f.tell()))
            f.write(raw)
    return size


class File:
    """A .pkf read back: header fields, params, tensors (as numpy views),
    vocabulary and metadata."""

    def __init__(self, path):
        self.buf = np.memmap(path, np.uint8, "r")
        b = self.buf
        (magic, version, arch, name, n_params, n_tensors, params_off, tensors_off, vocab_off,
         vocab_size, meta_off, meta_size, data_off, file_bytes, digest, _) = HEADER.unpack_from(b, 0)
        if magic != MAGIC:
            raise ValueError("%s is not a .pkf file" % path)
        self.version, self.digest = version, digest.hex()
        self.arch, self.name = arch.rstrip(b"\0").decode(), name.rstrip(b"\0").decode()
        self.params = {}
        for i in range(n_params):
            k, kind, v = PARAM.unpack_from(b, params_off + i * PARAM.size)
            fmt = {PARAM_U32: "<I", PARAM_I32: "<i", PARAM_F32: "<f"}[kind]
            self.params[k.rstrip(b"\0").decode()] = struct.unpack(fmt, v)[0]
        self.descs = {}
        for i in range(n_tensors):
            nm, typ, ndim, d0, d1, off, nbytes, _, _ = TENSOR.unpack_from(b, tensors_off + i * TENSOR.size)
            self.descs[nm.rstrip(b"\0").decode()] = (typ, ndim, (d0, d1), off, nbytes)
        self.pieces, self.types, self.punct = read_vocab(bytes(b[vocab_off:vocab_off + vocab_size]))
        self.meta = {}
        parts = bytes(b[meta_off:meta_off + meta_size]).split(b"\0")[:-1]
        for k, v in zip(parts[0::2], parts[1::2]):
            self.meta[k.decode()] = v.decode()

    def raw(self, name):
        typ, ndim, (d0, d1), off, nbytes = self.descs[name]
        dt = np.float32 if typ == F32 else np.int8
        a = np.frombuffer(self.buf, dt, d0 * d1, off).reshape(d0, d1)
        return a if ndim == 2 else a.reshape(d0)

    def tensor(self, name):
        """A tensor as float32, a q8 matrix dequantized."""
        typ = self.descs[name][0]
        if typ == Q8:
            return dequantize_q8(self.raw(name), self.raw(name + ".scale"))
        return np.array(self.raw(name))
