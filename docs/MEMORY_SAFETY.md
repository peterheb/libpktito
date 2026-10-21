# Memory-safety rules

Three things pktito reads come from outside it: a model file, a WAV file, and
the samples a caller hands the library. A model file can be truncated,
corrupt, written by some other program or written to do harm; a WAV can come
down a pipe from anywhere; a sample can be anything a float can be. The rules
below are the review contract for the code that turns those bytes into sizes,
pointers and arithmetic: core/model.c, core/wav.c and the front ends of
core/mel.c and core/vad.c. vadito's rules are the model, and these are the
same rules at pktito's size.

## Copy, then check, then point

A header, a param or a descriptor is copied into a local struct with `memcpy`
before any field of it is believed, and only after every field is checked does
a pointer into the input get formed. A range is checked without forming its
end and without an addition that could wrap:

```c
off <= size && count <= (size - off) / sizeof(element)
```

never `off + count * sizeof(element) <= size`. Every nested extent - the
tables inside the file, the vocabulary's own tables inside its blob, each
tensor - is checked the same way, against what is left rather than against
the end.

Public file and buffer entry points reject a null path or byte pointer before
using it. The in-memory model loader also checks that adding its alignment
slack fits in `size_t` before it allocates and copies.
Counts derived while resampling a WAV saturate instead of wrapping, and the
front ends check sample-count additions and byte multiplications before they
grow a whole-recording buffer.

## The model file

`pk_pkf_validate` is the boundary: `pk_model_load`, which maps the file, and
`pk_model_from_bytes`, which copies one, both call it before they form any
pointer into the file. It checks:

- **the header**: at least 256 bytes and at most 16 GB, which bounds every sum
  below well inside 64 bits; the magic and the version; the declared size
  equal to the real one; the architecture "parakeet-tdt" and the model's name
  each NUL-terminated within its field and NUL-padded after; the reserved
  bytes zero; 1 to 256 params and 1 to 4096 tensors.
- **the layout**: the tables sit end to end in one fixed order - the params at
  byte 256, the tensor table after them, the vocabulary, the metadata, then
  zeros to the next 64-byte boundary, where the data starts - so a file has one
  way to be laid out and each offset is checked to be exactly where it must be,
  each size against what is left of the file.
- **the params**: each key NUL-terminated and padded, no key twice, a kind of
  u32, i32 or f32.
- **the vocabulary**: a count of 1 to 2^20 pieces; its offset table and type
  table inside the blob; the offsets starting at 0 and never going backward,
  the last one inside the blob with four bytes left for the punctuation's
  length; each type 1 to 5 (normal, unknown, control, user-defined, unused:
  no byte pieces); each piece and the punctuation well-formed UTF-8 - no
  overlong form, no surrogate, nothing past U+10FFFF - and free of NUL, since
  a transcript is a C string; the punctuation's length exactly what is left.
- **the metadata**: whole strings, each terminated inside the table, in key and
  value pairs, no empty key, every string UTF-8.
- **each tensor**: its name NUL-terminated within 48 bytes, NUL-padded, no name
  twice; type f32 or q8; one or two dimensions, neither zero, a vector's second
  one; the offset 64-byte aligned, at or past the data's start and inside the
  file; the byte count equal to the element count times the element size, the
  product formed in 64 bits from two 32-bit dimensions; both reserved words
  zero; and no two tensors sharing a byte (sorted by offset, each ends at or
  before the next begins).

Byte-valid is not model-valid. The binder then:

- reads every param the model needs by its key, with its kind and its range -
  the sample rate 16000, n_fft a power of two from 64 to 4096, win_length at
  most n_fft, the widths multiples of 32, the heads dividing d_model into
  widths that are multiples of 16, an odd convolution of at most 63 taps, 1 to
  16 durations, the guards against zero (log_guard, norm_eps, ln_eps) at least
  1e-30 - and refuses a param it did not read, as it refuses a tensor it did
  not bind: either is a file made for something this build does not
  implement;
- derives every tensor's shape from those params and binds each tensor by name
  to that shape and type, never taking one from a descriptor; a q8 matrix must
  have its f32 `.scale`, one a row, rows a multiple of 16, columns a multiple
  of 32 and at most `PK_MAX_BLOCKS` blocks of them, so that a backend may keep a
  value a block on its stack;
- checks every float it will multiply by to be finite (the int8 weights need no
  check: every byte is a valid one), the position frequencies to be in (0, 1],
  the two FFTs' twiddles to be the cosines and sines they must be, and the
  VAD's window to be Hann;
- holds the vocabulary's count to the vocab param.

A refused file is refused at load, with a message, never on the first
recording. A format change extends `pk_pkf_validate` and tests/pkfcheck.c in
the same change.

## The WAV file

core/wav.c reads front to back and never seeks, so its input may be a pipe,
and it never allocates from a size the file states: it keeps one fixed buffer
and decodes whole frames out of it. Every chunk's size is checked before it is
read past (by reading, so that a pipe can be read past), the fmt chunk is
copied into a fixed local before it is parsed, and a frame's size is bounded
by refusing more than 256 channels and samples wider than 8 bytes. A data chunk
that ends early is read to where it ends and reported by `pk_wav_status`; one
whose size is a stream's is read to the end of the file.
The corpus reaches the largest frame the format can describe and both 32-
and 64-bit maximum declared lengths, so those bounds stay true when the
reader changes.

## Samples

A sample that is not finite is read as 0, by the features (core/mel.c) and by
the VAD (core/vad.c). A finite sample of any size is safe too, though it is
not heard as anything useful. The VAD clamps samples to +-1e6, which keeps its
STFT finite. In the encoder's features a sample past about 1e17 overflows the
power spectrum to infinity and its log-mel to NaN, and the subsampling's first
ReLU (`v > 0 ? v : 0`, and its vector forms, which agree for a NaN) makes that
0. tests/pkfcheck runs 1e30 and -3e38 samples through the engine.

A model stores only finite floats, but hostile finite weights can still make
an intermediate overflow. An int8 quantizer turns a block containing a NaN or
infinity into zeros before any float is narrowed to an integer. tests/pkfcheck
runs such a model through the engine, and `make check-sanitize` verifies the
conversion under UBSan.

## Arithmetic and conversions

A value narrowed to a smaller type is first proven in range: the int8
quantizer's values are in [-127, 127] by construction. A two's complement
sample is decoded by arithmetic, not by converting an unsigned value to a
signed type, which C11 leaves to the implementation. Counts that come from a
caller (`n` samples, `cap` bytes of text, `cap` segments) are the caller's
contract, stated in the header that declares the function.

## Verification

`make check-c` runs both parsers on their corpora. tests/pkfcheck.c builds
every file with a writer of its own, written from core/pkf.h (it remakes
tests/data/tiny.pkf byte for byte), and runs: every truncation of a small file
with its size field as found and patched, the tiny model cut inside each
tensor and each table, each field of the header, a param, a descriptor, the
vocabulary and the metadata wrong in each way it can be, the binding failures,
20,000 random bit flips of everything before the tensor data, and the whole
engine on the tiny model and on the flipped models that still load.
tests/wavcheck.c runs every prefix of a valid WAV, each field of fmt wrong,
the widest legal frame and maximum 32- and 64-bit declared lengths.
`make check-sanitize` runs both under AddressSanitizer and
UndefinedBehaviorSanitizer.

A parser bug gets a case in its corpus in the change that fixes it, and the
case reaches the failing boundary rather than merely checking that something
was refused: disabling any one check in core/model.c makes a case fail, the
vocabulary's offset order under the sanitizers, which see the out-of-bounds
read that check prevents.
