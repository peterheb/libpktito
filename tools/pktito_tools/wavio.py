"""A WAV read into float64 with the standard library and numpy: PCM at 8,
16, 24 and 32 bits and IEEE float at 32 and 64, which the stdlib's wave
module (PCM only) does not cover; FLEURS ships float32. Samples are scaled
as soundfile and torchaudio scale them, and channels averaged as NeMo's
loader does."""
import struct

import numpy as np


def read(path):
    """(mono float64 samples, rate)"""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("%s is not a RIFF WAVE file" % path)
    i, fmt, pcm = 12, None, None
    while i + 8 <= len(data):
        cid, size = data[i:i + 4], struct.unpack_from("<I", data, i + 4)[0]
        body = data[i + 8:i + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack_from("<HHIIHH", body, 0)
            if fmt[0] == 0xFFFE and len(body) >= 26:          # EXTENSIBLE: the subformat's tag
                fmt = (struct.unpack_from("<H", body, 24)[0],) + fmt[1:]
        elif cid == b"data":
            pcm = body
        i += 8 + size + (size & 1)
    if fmt is None or pcm is None:
        raise ValueError("%s has no fmt or no data chunk" % path)
    tag, ch, rate, _, _, bits = fmt
    if tag == 3 and bits in (32, 64):
        a = np.frombuffer(pcm[:len(pcm) // (bits // 8) * (bits // 8)], "<f%d" % (bits // 8)).astype(np.float64)
    elif tag == 1 and bits == 16:
        a = np.frombuffer(pcm[:len(pcm) // 2 * 2], "<i2") / 32768.0
    elif tag == 1 and bits == 32:
        a = np.frombuffer(pcm[:len(pcm) // 4 * 4], "<i4") / 2147483648.0
    elif tag == 1 and bits == 8:
        a = (np.frombuffer(pcm, np.uint8).astype(np.float64) - 128) / 128
    elif tag == 1 and bits == 24:
        b = np.frombuffer(pcm[:len(pcm) // 3 * 3], np.uint8).reshape(-1, 3).astype(np.int32)
        a = ((b[:, 0] | b[:, 1] << 8 | b[:, 2] << 16) << 8 >> 8) / 8388608.0
    else:
        raise ValueError("%s: format %d at %d bits" % (path, tag, bits))
    return a[:len(a) // ch * ch].reshape(-1, ch).mean(1), rate
