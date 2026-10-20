# pkito

NVIDIA's [Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
speech recognizers in C11, with nothing beyond the C library and pthreads:
no NeMo, no PyTorch, no ONNX Runtime, no Python at run time. One binary, one
model file (the recognizer, its vocabulary, and Silero's VAD to cut long
recordings at their pauses), and NeMo's transcripts.

```
$ ./pkito talk.wav
And says, how do I get to Dublin? And the answer that comes back is, well, I
wouldn't start from here, Sonny. That is to say, much of political philosophy...
```

A minute of English speech takes 0.51 s on an M4 Pro (118 times real time,
eight cores), 1.66 s on a Core i9-12900HK, and 58 s on a Raspberry Pi 3B+'s
four Cortex-A53s. Every backend computes the same bits on every machine.

## What this is

| | |
|---|---|
| models | Parakeet TDT 0.6B v2 (English) and v3 (25 European languages), converted from NVIDIA's `.nemo` checkpoints |
| backends | `neon-dp` (ARMv8.2 + SDOT: Pi 5's A76, Apple M), `neon` (ARMv8.0: Pi 3's A53, Pi 4's A72), `avx2-vnni` (Alder Lake and later, Zen 5), `avx2` (Haswell and later), `ref` (scalar C, anything) |
| numerics | int8 weights and activations with exact integer sums; float32 attention, norms and decoder state (README, "Why int8") |
| what it does | a recording to punctuated, cased text, every token with its time; long recordings cut at their silences; live input transcribed at its pauses; subtitles |
| audio | WAV: PCM 8/16/24/32, float 32/64, A-law, mu-law, EXTENSIBLE, RF64, streamed from a pipe; any channel count; any rate, brought to 16 kHz |
| dependencies | the C library, libm and pthreads |

## Try it

```sh
make                          # backend detected: neon-dp, neon, avx2-vnni, avx2 or ref
make BACKEND=ref              # scalar C, for anything
```

Then a model, converted once from NVIDIA's checkpoint and Silero's VAD. The
converter is Python with numpy and nothing else: it reads the `.nemo`
archive, the PyTorch checkpoint inside it and the SentencePiece model
itself.

```sh
uv venv .venv && uv pip install --python .venv/bin/python numpy huggingface_hub
.venv/bin/hf download nvidia/parakeet-tdt-0.6b-v2 parakeet-tdt-0.6b-v2.nemo README.md \
    --local-dir models/src/parakeet-tdt-0.6b-v2          # and -v3 for the multilingual one
git clone --depth 1 --branch v6.2.3 https://github.com/snakers4/silero-vad models/src/silero-vad
make models                   # models/parakeet-tdt-0.6b-v2.pkf (624 MB), -v3.pkf (648 MB)
```

Copy a `.pkf` to a board beside the binary; the board needs no Python. Then:

```sh
./pkito talk.wav                        # the transcript
./pkito --json talk.wav                 # chunks: start, end, text, every token and its time
./pkito --srt talk.wav > talk.srt       # subtitles, a cue a chunk
./pkito a.wav b.wav c.wav               # several: a line each, the name, a tab, the text
arecord -f S16_LE -r 16000 -c 1 -t wav | ./pkito --stream -     # live, a line a pause
./pkito segments talk.wav               # Silero VAD's speech segments, as vadito prints them
./pkito bench talk.wav                  # this machine's speed
./pkito info                            # the model file: its source, licence and recipe
```

`-m` picks the model when `models/` holds both, `-t` the thread count (one
a core by default). A recording longer than `--max-chunk` (30 s) is cut at
its silences into chunks that long at most, each transcribed whole;
`--stream`'s lines end at a pause of `--pause` (1000 ms). Both defaults are
measured below.

## How close to NeMo

There are three layers, and each is held to the one above it.

**The model.** `tools/pkito_tools/reference.py` is NeMo 3.0's arithmetic in
float64 numpy, batch size one, written from NeMo's code: the features, the
FastConformer encoder, the prediction network and joint, and the
label-looping TDT search `model.transcribe()` runs. On four test recordings
(English, German and French speech, one of them a minute long) it gives
NeMo's token ids and timestamps exactly, for both models, with the encoder's
output within 2e-6 of NeMo's.

**The numerics.** pkito multiplies in int8. Against float32 - what NeMo
computes - over real speech (`tools/corpus.py`: 200 LibriSpeech test-clean
recordings, every speaker; 360 FLEURS recordings in German, French, Spanish,
Italian, Polish and Russian), whole recordings, `tests/wercheck.py`:

| | recordings | WER, float32 | WER, pkito | transcripts not float32's | WER between them |
|---|---|---|---|---|---|
| v2, LibriSpeech | 200 (26.5 min) | 1.51% | 1.51% | 4 | 0.07% |
| v3, FLEURS | 360 (70.2 min) | 4.41% | 4.42% | 45 | 0.35% |

**The text.** `tests/tokcheck.py` holds the C detokenizer to sentencepiece's
own decoding and NeMo's one change after it (the space before punctuation
taken away) on every piece alone and 3000 random sequences: 4,024 sequences
for v2 and 11,192 for v3, NeMo's text every time. And the VAD is vadito's:
its probabilities on Silero's test.wav are vadito's bits on all 1,875
frames, and vadito holds those to onnxruntime's within 1.4e-5.

And the bits are the same everywhere. `tests/data/test.v2.json` and
`test.v3.json` are the transcripts `ref` writes for test.wav, every token and
its time, and `make check-c` holds every backend on every machine to them
byte for byte: so far `neon-dp` on an M4 Pro (Apple clang 21), `neon` on a
Pi 3B+ (gcc 14.2), `avx2` and `avx2-vnni` on a 12900HK (gcc 15.2), and `ref`.

## How fast

`./pkito bench`: Silero's test.wav (60 s of English), best of five,
2026-10-20. Seconds, and how many times real time:

| | backend | threads | whole (750 frames) | in 30 s chunks |
|---|---|---|---|---|
| Apple M4 Pro | `neon-dp` | 8 | 0.509 (118x) | |
| | `neon` | 12 | 1.223 (49x) | |
| | `ref` | 12 | 22.1 (2.7x) | |
| Intel Core i9-12900HK | `avx2-vnni` | 6 | 1.656 (36x) | |
| | `avx2-vnni` | 1 | 5.46 (11x) | |
| | `avx2` | 6 | 2.26 (27x) | |
| Raspberry Pi 3B+, Cortex-A53 1.4 GHz | `neon` | 4 | | 58.4 (1.0x) |

Against [parakeet.cpp](https://github.com/mudler/parakeet.cpp) (ggml, its
q8_0 files) on the same 100 LibriSpeech test-clean recordings, the M4 Pro's
CPU at eight threads, both transcripts through one normalizer:

| | WER | real time |
|---|---|---|
| pkito, v2 | 1.71% | 119x |
| parakeet.cpp, v2 q8_0 | 1.75% | 30x |
| pkito, v3 | 1.88% | 119x |
| parakeet.cpp, v3 q8_0 | 1.96% | 30x (its Metal GPU path: 90x) |

Where the time goes (`make PROFILE=1`, the M4 Pro at eight threads, the minute
whole): the feed-forward modules' int8 products 44%, the attention's float
products 22%, q k v and the positions 14%, the convolution module 8%, the
decoder 3%, the features and subsampling 4%. The int8 product runs at about
56% of the M4's SDOT peak there; the 12900HK's at about 22 MACs a cycle of
VNNI's 64, and every stage scales 3.3x from one thread to six and no further,
which is the package's clock under all-core load. backend/*/CLIMB.md has
every measurement the kernels rest on.

## Why int8

A 0.6B model in float32 is 2.5 GB, and a Pi 3 has 0.9 GB. `tools/quantcheck.py`
runs the model in float32 with its matrices, and the activations entering
them, rounded each way, over the 200 LibriSpeech recordings, and reports
`dlogp`: how much less likely the rounded model finds float32's own token and
duration decisions, per decision, a paired measure much finer than WER.

| rounded (v2) | WER | transcripts changed | dlogp | encoder error |
|---|---|---|---|---|
| float32 | 1.51% | | | |
| fp16 weights | 1.51% | 0 | -0.00001 | 0.05% |
| int8 weights, a scale a row | 1.49% | 6 | -0.012 | 2.8% |
| ... and int8 activations, a scale a row | 1.49% | 20 | -0.080 | 10.4% |
| ... and int8 activations, a scale a block of 32 | 1.47% | 7 | -0.012 | 3.4% |
| ... blocks of 64 / 128 | 1.49% / 1.42% | 5 / 6 | -0.016 / -0.028 | 4.0% / 4.7% |
| int8 weights and activations, blocks of 32 both (ggml's q8_0) | 1.49% | 1 | -0.017 | 2.9% |
| 4-bit weights (ggml's q4_0, q4_K) | 1.51% | 36 / 31 | -0.25 / -0.20 | 20% / 18% |

Activations with a scale a row lose what blocks of 32 keep, and the loss is
the feed-forward modules': rounding only their inputs a row at a time costs
-0.060 (the second matrix's) and -0.030 (the first's), where q, k, v and the
positions cost nothing measurable. So the encoder's activations are int8 in
blocks of 32, and the decoder's and the joint's a row at a time, which costs
nothing there. On v3 over FLEURS the same choice is 4.42% against float32's
4.41%; 4-bit weights are 4.76% there, with 127 of 360 transcripts changed,
which is why the files are int8.

The attention's own three products stay float. Making them int8 as well -
keys, positions and values rounded a row at a time like weights, queries and
attention probabilities in blocks of 32 like activations (`quantcheck.py
... ;att8`) - moved nothing measurable on v2, but on v3 over FLEURS it took
dlogp from -0.024 to -0.041 (t -5.4) and the encoder's error from 5.2% to
5.9%, for a stage that is about an eighth of the time at 30 s chunks.

The integer sums are what make every backend agree: a block's 32 products
add up to the same int32 in any order, so SDOT, SMULL, VPMADDUBSW and
VPDPBUSD each take them their own way, and only the one fused multiply-add a
block that scales the sum is fixed (backend/kernels.h).

## Long recordings

The encoder hears a recording whole - its attention is over every frame -
so its work grows with the square of the length and its memory with the
length. pkito cuts anything longer than `--max-chunk` at its silences: Silero's
VAD finds the speech, segments longer than a chunk are split at their
longest pause, and each cut falls half way into the silence between two
segments (at most a second into it, so a long silence is skipped rather than
transcribed). Over 12 LibriSpeech chapters stitched into recordings of 1.5 to
5 minutes (`tools/corpus.py longform`, 45 minutes, 7,098 words):

| `--max-chunk` | 10 s | 20 s | 30 s | 60 s | 120 s | whole |
|---|---|---|---|---|---|---|
| v2 | 1.80% | 1.70% | **1.58%** | 4.27% | 1.68% | 1.58% |
| v3 | 2.27% | 2.16% | **2.06%** | 2.45% | 2.10% | 2.24% |

30 s is as good as the whole recording for v2 and better for v3, at a
fraction of the attention's cost. The 60 s outlier is one 58 s chunk that
came out as its last sentence: NeMo 3.0 in float32 and the float64 reference
lose the same on the same samples, and a start 2 ms earlier avoids it in all
three. That is the model's own failure, and a reason not to hand it long
stretches.

## Streaming

`--stream` reads its input as it arrives and transcribes each stretch of
speech when a pause of `--pause` ends it (VADIterator, Silero's streaming
state machine). The pause is a trade between how soon a line appears and how
much the model hears at once. On the same long-form set:

| `--pause` | 100 | 300 | 500 | 700 | 1000 | 1500 | 2000 ms |
|---|---|---|---|---|---|---|---|
| v2 | 3.10% | 2.21% | 2.04% | 1.86% | **1.66%** | 1.66% | 1.68% |
| v3 | 4.02% | 3.01% | 2.72% | 2.21% | **1.89%** | 1.78% | 1.73% |

## How it works

Parakeet TDT is, for a recording of n samples at 16 kHz:

| stage | | shape |
|---|---|---|
| features | preemphasis; 512-point STFT, 400-sample Hann window, hop 160; power through 128 mel bands; log; each band normalized over the recording | n/160 x 128 |
| subsampling | 3x3 stride-2 convolutions over time and frequency: one to 256 channels, then twice depthwise and pointwise, ReLUs; flattened and projected | n/1280 x 1024 |
| 24 conformer layers | half a feed-forward (1024 -> 4096 -> 1024, SiLU); Transformer-XL attention, 8 heads, relative positions; a convolution module (GLU, depthwise 9, BatchNorm, SiLU); half a feed-forward; LayerNorm | n/1280 x 1024 |
| prediction network | the last token's embedding through two 640-unit LSTMs | 640 |
| joint | relu(encoder half + prediction half) through one matrix | 1024 + 1 + 5 logits |
| TDT search | the best token and the best of five durations (0-4 frames); emit the token unless it is blank, move on by the duration | tokens |

Four things here are not what NeMo computes, and none moves a transcript
off NeMo's by more than the numerics above: the matrices are int8
(converter); BatchNorm is folded into the depthwise convolution before it,
the feed-forward's 0.5 into its second matrix and the LSTM's two biases into
one (converter, `canon.py`, each exact or closer to exact); the position
products are taken over the window of positions each block of queries uses
rather than over all of them (core/encoder.c), and the prediction network
runs once a token rather than once a step, since its input only changes with
a token (core/decoder.c). The transcendental functions are pkito's own
(core/vmath.h), not libm's, whose last bits differ between platforms.

## The library

```c
#include "pkito.h"

pk_model *m = pk_model_load("models/parakeet-tdt-0.6b-v2.pkf", err, sizeof err);
pk_asr *a = pk_asr_new(m, 0);                     /* 0: a thread a core */

const pk_tok *t;
long n = pk_asr_run(a, samples, n_samples, &t);   /* 16 kHz mono float */
pk_text(m, t, (size_t)n, text, sizeof text);      /* NeMo's transcript */
/* t[i].frame * pk_model_frame_seconds(m) is when token i was heard */

pk_vad *v = pk_vad_new(m);                        /* Silero VAD, as vadito has it */
pk_vad_probs(v, frames, n_frames, probs);
size_t ns = pk_segments(probs, n_frames, 16000, n_samples, &sp, seg, cap);
size_t nc = pk_chunks(seg, ns, n_samples, 30 * 16000, chunk);
```

A model is read-only once loaded and may be shared by transcribers on any
number of threads; a transcriber belongs to one thread at a time and runs
its own pool. `make libpkito.a` builds the library alone; core/pkito.h is
all of its interface, core/wav.h and core/resample.h the WAV library's.
examples/transcribe.c is a whole program against it in a page, and `make
check-c` holds its transcript to the CLI's.

## The model file

A `.pkf` (core/pkf.h) is the recognizer, its vocabulary and Silero VAD in
one file, mapped read-only and used in place, with a metadata table that
says what it was made from: the HuggingFace repo, file, commit and sha256
of the checkpoint, its licence and the attribution CC-BY-4.0 asks for, the
VAD's source and licence, the converter's version and the recipe. `pkito
info` prints it. The loader holds every extent, shape, type and value of the
file to what the architecture and docs/MEMORY_SAFETY.md say before it forms
a pointer, and tests/pkfcheck runs 29,059 malformed files at it.

`tools/pkito-tools convert` writes one; `info` lists one; `ref` runs the
float64 reference on a WAV from either a `.nemo` or a `.pkf`; `tiny` writes
a 1.2 MB model with random weights that has every tensor, param and piece
type, for the checks.

## Checking

```sh
make check-c                 # a compiler and a model: any board
make check                   # and the Python half: sentencepiece, the corpora
make check-sanitize          # the model file and WAV parsers under ASan and UBSan
make check-math              # every float through exp, sigmoid, tanh and log
make remote HOST=user@board  # rsync, build, check-c and bench there
```

`check-c` is the math library's bounds, every kernel against `ref` bit for
bit (tests/kcheck: 1,525 cases), the model file's validator on its
malformed corpus and the whole engine on the tiny model, the WAV reader on
its own, and test.wav through `./pkito` and through examples/transcribe:
vadito's VAD bits, and both models' transcripts byte for byte.

## The tree

```
core/        the model (encoder.c, decoder.c, mel.c), the file (model.c, pkf.h),
             the text (tokens.c), the transcriber (asr.c), the VAD (vad.c,
             segment.c), the WAV library (wav.c, resample.c), the math
             (vmath.h), the pool (pool.c), the CLI (cli.c)
backend/     the seam (kernels.h) and ref/, neon/ (and neon-dp/), avx2/ (and
             avx2-vnni/); a CLIMB.md beside each tuned backend
tools/       the converter, the float64 reference, the corpora, the
             quantization experiment
tests/       the C checks, the Python ones, and test.wav with its golden files
examples/    the library from the outside
docs/        MEMORY_SAFETY.md: the rules the two parsers keep
```

## License

MIT for the code. The weights a `.pkf` holds are NVIDIA's, under CC-BY-4.0;
whoever shares a `.pkf` owes NVIDIA that licence's attribution, which every
file carries in its metadata. Silero VAD's weights, and the parts of the code
that port its model and state machines, are MIT, Copyright (c) 2020-present
Silero Team. LICENSE has all three.
