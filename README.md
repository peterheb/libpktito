# libpktito

NVIDIA's [Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
speech recognizers in C11, with nothing beyond the C library and pthreads:
no NeMo, no PyTorch, no ONNX Runtime, no Python at run time. One binary, one
model file (the recognizer, its vocabulary, and Silero's VAD to cut long
recordings at their pauses), and NeMo's transcripts.

```
$ ./pktito-c talk.wav
And says, how do I get to Dublin? And the answer that comes back is, well, I
wouldn't start from here, Sonny. That is to say, much of political philosophy...
```

A minute of English speech takes 0.22 s on a Ryzen 9 9950X (277 times
real time), 0.49 s on an M4 Pro, 0.83 s on a Core i9-12900HK, 1.3 s on a
Core 3 304, 3.7 s on a Raspberry Pi 5 and 45 s on a Raspberry Pi 3B+, cut
at its pauses as `./pktito-c` cuts it by default. Every backend computes the
same bits on every machine.

## What this is

| | |
|---|---|
| models | Parakeet TDT 0.6B v2 (English) and v3 (25 European languages), converted from NVIDIA's `.nemo` checkpoints |
| backends | `neon-dp` (ARMv8.2 + SDOT: Pi 5's A76, Apple M), `neon` (ARMv8.0: Pi 3's A53, Pi 4's A72), `amx-int8` (AMX's tiles: Sapphire and Granite Rapids), `avx512-vnni` (AVX512-VNNI in zmm: Ice Lake and Tiger Lake, Zen 4 and 5), `avx2-vnni` (AVX-VNNI: Alder Lake and later Intel cores without AVX-512), `avx2` (Haswell and later), `ref` (scalar C, anything) |
| numerics | int8 weights and activations with exact integer sums; float32 attention, norms and decoder state (README, "Why int8") |
| what it does | a recording to punctuated, cased text, every token with its time; long recordings cut at their silences; live input transcribed at its pauses; subtitles |
| audio | WAV: PCM 8/16/24/32, float 32/64, A-law, mu-law, EXTENSIBLE, RF64, streamed from a pipe; AIFF and AIFF-C: PCM 8 to 32 bits either byte order, float 32/64, A-law, mu-law; any channel count; any rate, brought to 16 kHz |
| dependencies | the C library, libm and pthreads |
| systems | Linux, macOS, and Windows 10 and later (built with MSYS2's MinGW-w64 gcc) |

## Try it

```sh
make                          # every backend of this machine's family, and ref
make BACKEND=ref              # scalar C alone, for anything
```

A build holds every backend of its family that the compiler can make
(neon-dp and neon on AArch64; amx-int8, avx512-vnni, avx2-vnni and avx2
on x86-64) and the CPU picks the first it can run, so a binary copied to
another machine of the family runs there at that machine's best: one
built on an Alder Lake runs on a Zen 4, one built on a Pi 5 on a Pi 3.
`PKTITO_KERNELS=neon ./pktito-c ...` runs a given one of them, to compare.

On Windows the same, in an [MSYS2](https://www.msys2.org) UCRT64 shell with
`pacman -S make diffutils mingw-w64-ucrt-x86_64-gcc`; pktito-c.exe is linked
statically and runs from cmd or PowerShell as well. It reads its
arguments from Windows in UTF-16 and opens files by those names, so a
model or recording named past the code page (`C:\Users\José\ñ.wav`)
opens, and a transcript's accents and Cyrillic reach the console as
UTF-8. Neither cmd nor PowerShell expands `*.wav` for a program, and
pktito-c.exe doesn't either: in PowerShell, `.\pktito-c (Get-ChildItem
*.wav).FullName`; MSYS2's bash expands it as any Unix shell does.
Windows PowerShell 5.1's `>` decodes a program's output in the console's
code page before it writes the file, so to keep a transcript's accents
there, first `[Console]::OutputEncoding = [Text.Encoding]::UTF8` (cmd's
`>` writes pktito-c's UTF-8 as it is).

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

Copy a `.pkf` to a board beside the binary; the board needs no Python, and
memory for the file (it is mapped, and the page cache is its one copy) and
some 25 to 90 MB besides at the default chunking, more threads taking more:
652 MB at its peak on a Pi 3B+ with v2, 651 MB on a Pi 5, 663 MB on a
12900HK at 14 threads, 711 MB on an M4 Pro at 12. A 512 MB board (a Pi
Zero 2 W) cannot hold an int8 0.6B model; README, "Why int8", has what
4-bit would cost. Then:

```sh
./pktito-c talk.wav                        # the transcript
./pktito-c --json talk.wav                 # chunks: start, end, text, every token and its time; and meta
./pktito-c --srt talk.wav > talk.srt       # subtitles (--vtt for WebVTT), cut where they read best
./pktito-c a.wav b.wav c.wav               # several: a line each, the name, a tab, the text
arecord -f S16_LE -r 16000 -c 1 -t wav | ./pktito-c --stream -     # live, a line a pause
./pktito-c segments talk.wav               # Silero VAD's speech segments, as vadito prints them
./pktito-c bench talk.wav                  # this machine's speed
./pktito-c info                            # the model file: its source, licence and recipe
```

`-m` picks the model when `models/` holds both, `-t` the thread count (one
a core by default). A recording longer than `--max-chunk` (30 s) is cut at
its silences into chunks that long at most, each transcribed whole;
`--stream`'s lines end at a pause of `--pause` (1000 ms). Both defaults are
measured below.

## How close to NeMo

There are three layers, and each is held to the one above it.

**The model.** `tools/pktito_tools/reference.py` is NeMo 3.0's arithmetic in
float64 numpy, batch size one, written from NeMo's code: the features, the
FastConformer encoder, the prediction network and joint, and the
label-looping TDT search `model.transcribe()` runs. On four test recordings
(English, German and French speech, one of them a minute long) it gives
NeMo's token ids and timestamps exactly, for both models, with the encoder's
output within 2e-6 of NeMo's.

**The numerics.** pktito multiplies in int8. Against float32 - what NeMo
computes - over real speech (`tools/corpus.py`: 200 LibriSpeech test-clean
recordings, every speaker; 360 FLEURS recordings in German, French, Spanish,
Italian, Polish and Russian), whole recordings, `tests/wercheck.py`:

| | recordings | WER, float32 | WER, pktito | transcripts not float32's | WER between them |
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
byte for byte: so far `neon-dp` on an M4 Pro (Apple clang 21) and a Pi 5
(gcc 14.2), `neon` on a Pi 3B+ (gcc 14.2) and the M4 Pro, `avx2` and
`avx2-vnni` on a 12900HK (gcc 15.2) and on a Core 3 304 under Windows 11
(MSYS2's gcc 16.2), `avx2-vnni`, `avx512-vnni` and `avx2` on a Ryzen 9
9950X (gcc 15.2), `amx-int8` on a Sapphire Rapids Xeon (gcc 13.3), and
`ref`; and the 304's Windows build, copied as it was, runs `avx512-vnni` on
a Zen 4 and all three on the 9950X, the same bytes.

## How fast

`./pktito-c bench`: Silero's test.wav (60 s of English), best of five,
2026-10-20 (the 9950X's 2026-10-24). A machine's first row is what it picks itself, its default
thread count and the fastest backend its CPU can run; the rows under it
force another backend (`PKTITO_KERNELS`) or one thread. Seconds, and how
many times real time, for the minute whole (`--max-chunk 60`, 750 encoder
frames) and in the default 30 s chunks:

| | compiler | backend | threads | whole | in 30 s chunks |
|---|---|---|---|---|---|
| Apple M4 Pro | Apple clang 21.0 | `neon-dp` | 12 | 0.487 (123x) | 0.490 (122x) |
| | | `neon` | 12 | 1.047 (57x) | 1.040 (58x) |
| | | `ref` | 12 | 20.6 (2.9x) | 21.6 (2.8x) |
| Intel Core i9-12900HK | gcc 15.2 | `avx2-vnni` | 14 | 0.909 (66x) | 0.828 (72x) |
| | | `avx2-vnni` | 1 | 3.28 (18.3x) | 2.84 (21x) |
| | | `avx2` | 14 | 1.682 (36x) | 1.597 (38x) |
| | | `avx2` | 1 | 6.80 (8.8x) | 6.12 (9.8x) |
| Intel Core 3 304, Windows 11 | MSYS2 gcc 16.2 | `avx2-vnni` | 5 | 1.392 (43x) | 1.290 (47x) |
| AMD Ryzen 9 9950X, Windows 11 | MSYS2 gcc 16.2 | `avx512-vnni` | 16 | 0.228 (263x) | 0.217 (277x) |
| | | `avx2-vnni` | 16 | 0.292 (205x) | 0.274 (219x) |
| | | `avx2` | 16 | 0.534 (112x) | 0.529 (113x) |
| AMD Ryzen 7 PRO 8840HS, Windows 11 | the 304's build | `avx512-vnni` | 8 | 0.607 (99x) | 0.537 (112x) |
| Raspberry Pi 5, Cortex-A76 2.4 GHz | gcc 14.2 | `neon-dp` | 4 | 4.150 (14.5x) | 3.710 (16.2x) |
| Raspberry Pi 3B+, Cortex-A53 1.4 GHz | gcc 14.2 | `neon` | 4 | 48.1 (1.2x) | 44.6 (1.3x) |

The compiler moves the speed and never the transcript: every backend
computes `ref`'s bits whatever builds it. It can be a tenth: on a
Sapphire Rapids Xeon, `amx-int8`'s first loop ran 22.8x built by gcc 15.2
against 24.8 to 26.0x by gcc 13.3, where gcc 15 spilled it to memory, and
its rewrite runs 28.3 to 28.6x against 28.5 to 28.9x (backend/avx2/CLIMB.md,
the pipelined loop) - a tuned kernel is held to more than one compiler for
that reason. Compare a machine's rows compiler to compiler. `bench` and `info` print the compiler of the build;
where gcc and clang are both installed, `make clean && make CC=clang` and
bench each.

Against [parakeet.cpp](https://github.com/mudler/parakeet.cpp) (ggml, its
q8_0 files) on the same 100 LibriSpeech test-clean recordings, the M4 Pro's
CPU at eight threads, both transcripts through one normalizer:

| | WER | real time |
|---|---|---|
| pktito, v2 | 1.71% | 119x |
| parakeet.cpp, v2 q8_0 | 1.75% | 30x |
| pktito, v3 | 1.88% | 119x |
| parakeet.cpp, v3 q8_0 | 1.96% | 30x (its Metal GPU path: 90x) |

Where the time goes (`make PROFILE=1`, the minute whole). On the M4 Pro
at eight threads: the feed-forward modules' int8 products 44%, the
attention's float products 22%, q k v and the positions 14%, the
convolution module 9%, the decoder 3%, the attention's output 3%,
quantizing 3%, the features and subsampling 2%. On a Pi 5 at four, in
that order: 43%, 20%, 13%, 10%, 3%, 3%, 5% and 3%; quantizing, which waits
on memory there, is the share that grows. The int8 product runs at about
56% of the M4's SDOT peak, at 19 multiply-adds a cycle of SDOT's 32 on an
A76 (about 81% of what its blocks' float work leaves), and at 47 of VNNI's
64 on a 12900HK P-core (about 92%).

The default thread count is one a physical core of every kind (a second
thread on an SMT core adds nothing). A chip's slower cores take fewer of
each pass's items and still add their share: a Core 3 304, one P-core and
four E-cores, runs the minute whole in 5.02 s on its P-core and 1.39 s on
all five; a 12900HK in 0.96 s on its six P-cores and 0.91 s on all 14
(though 8, 16 and 18 threads take 1.10 to 1.15 s there, the same in every
pass); an M4 Pro in 0.50 s on its eight performance cores and 0.49 s on
all 12. backend/*/CLIMB.md has every measurement the kernels and the
defaults rest on.

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
length. pktito cuts anything longer than `--max-chunk` at its silences: Silero's
VAD finds the speech, segments longer than a chunk are split at their
longest pause, and each cut falls half way into the silence between two
segments, at most a second into it. Over 12 LibriSpeech chapters stitched
into recordings of 1.5 to 5 minutes (`tools/corpus.py longform`, 45 minutes,
7,098 words):

| `--max-chunk` | 10 s | 20 s | 30 s | 60 s | 120 s | whole |
|---|---|---|---|---|---|---|
| v2 | 1.80% | 1.70% | **1.58%** | 4.27% | 1.68% | 1.58% |
| v3 | 2.27% | 2.16% | **2.06%** | 2.45% | 2.10% | 2.24% |

30 s is as good as the whole recording for v2 and better for v3, at a
fraction of the attention's cost. The 60 s outlier is one 58 s chunk that
came out as its last sentence: NeMo 3.0 in float32 and the float64 reference
lose the same on the same samples, and a start 2 ms earlier avoids it in all
three. That is the model's own failure, and a reason not to hand it long
stretches. At 30 s it did not happen once over every speaker of test-clean
(40 recordings, 2.6 hours, 350 chunks a model): the slowest chunk came out
at 3.1 tokens a second against a median of 5.3, and WER was 1.89% (v2) and
2.27% (v3) over its 23,874 words.

### Between the speech

Silero hears voices, not words, and singing or speech over music is not
speech to it: of a 4.6-minute song sung over a synthesizer it found 30 s. So
what lies between the speech chunks is transcribed too, in the fewest pieces
of at most `--max-chunk`, all of a length. Of that song v2 gives 458 words
where it gave 118 with the gaps skipped, and NeMo 3.0's transcript of the
whole of it has 453; v3 gains 308.

Two kinds of piece are left out. A silent one - no 32 ms of it within 50 dB
of full scale - holds nothing to hear; the quietest of the 42 half-minutes
of music below is -32 dB in its loudest 32 ms. And a piece shorter than 3 s,
of which the recognizer makes words out of anything, digital silence
included. Random pieces of 7 instrumental tracks by Kevin MacLeod
(incompetech.com, CC BY 4.0: Carefree, Investigations, Local Forecast -
Elevator, Monkeys Spinning Monkeys, Sneaky Snitch, Volatile Reaction,
Wallpaper; 18.9 minutes), 28 of each length, each transcribed whole by
8c735c1, which heard any recording that fit in a chunk whole; how many came
out with words:

| piece | 1 s | 1.5 s | 2 s | 3 s | 4 s | 6 s | 9 s | 14 s | 20 s | 30 s |
|---|---|---|---|---|---|---|---|---|---|---|
| v2 | 27 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 of 42 |
| v3 | 12 | 5 | 3 | 1 | 1 | 2 | 0 | 0 | 2 | 0 of 42 |

The 1 s pieces said "Yeah.", "Mm." or "Uh". Of 12 kinds of synthetic noise
and hum, v2 heard words in 10 one-second pieces and in none from 1.5 s up;
v3 in none but one, a sentence of Spanish out of 9 s of white noise at -70
dB, which is silent now. v3 makes a phrase of music now and then ("And then
we're not going to be able to do that.", over 23 s of Investigations), as
NeMo's own v3 transcript of the whole of Sneaky Snitch has a sentence; a
gap's words are kept as they come.

A recording that fits in a chunk is one chunk if it has speech and a gap if
it has none, so the VAD runs on it too: 8 ms of the 0.25 s a 30 s recording
takes on the M4 Pro. Speech is cut as it was, and recordings of it barely
change: no recording of the LibriSpeech sets above has a gap to transcribe,
and 4.67 hours of conversational recordings gained 13 pieces, 84 s and one word
between their speech chunks ("Seven.", v2), not one of which changed. The
song costs what it holds: its transcription takes 2.14 s on the M4 Pro where
it took 0.43, hearing 276 s rather than 52.

## Streaming

`--stream` reads its input as it arrives and transcribes each stretch of
speech when a pause of `--pause` ends it (VADIterator, Silero's streaming
state machine). Only what Silero calls speech: a live input's singing is not
transcribed. The pause is a trade between how soon a line appears and how
much the model hears at once. On the same long-form set:

| `--pause` | 100 | 300 | 500 | 700 | 1000 | 1500 | 2000 ms |
|---|---|---|---|---|---|---|---|
| v2 | 3.10% | 2.21% | 2.04% | 1.86% | **1.66%** | 1.66% | 1.68% |
| v3 | 4.02% | 3.01% | 2.72% | 2.21% | **1.89%** | 1.78% | 1.73% |

## Captions

`--srt` and `--vtt` write a subtitle track from the words' times: lines of
42 characters at most, one or two to a cue, each cue shown from its first
word until a little after its last. The cut is not the greedy one, filling
each cue until the next word will not fit, which leaves a word or two to
flash by at the end. Every possible cue is laid out and timed, and the track
is the cheapest path through all of them (core/captions.c, after
docs/caption-spec.md): a long pause is always a cut, a sentence's end or a
comma is a good one, a cue shorter than a second or faster than 20
characters a second costs, and in English no cue or line ends on "the" or
"to", or splits "going to" and the other semi-modals. A cue that a pause
of a second cuts in mid-sentence ends in an ellipsis, "So I…", as the
Netflix and DCMP style guides mark a pause: right whether the speaker
trailed off, hesitated or broke off, which the times cannot tell apart.

v3 writes 25 languages and does not say which it heard, so for its
transcripts the English rules follow the words: `pk_language`
(core/language.c) scores each language by how many of its commonest
words the transcript uses, a word weighing more the fewer languages share
it, and answers when one leads by half again. Its table, each language's
100 commonest words, uses data derived from FLEURS (google/fleurs, CC BY
4.0; tools/langwords.py counts it). On FLEURS's dev and test transcripts,
which the table never saw, English is called English by mistake in none
of the texts of 20 words or more, and English texts are found 99.6% of the
time at 20 words, all of them from 50; on v3's own transcripts of FLEURS
and LibriSpeech, all of them from 20 (`tests/langcheck --eval`). `--json`'s
`meta` says the language too.
The same function is the library's `pk_captions`, which the desktop app's
subtitles come from too.

`make check-c` holds it to the spec's own cases (tests/capcheck.c),
including the nine words that greedy packing cuts 8 + 1 and this cuts 4 + 5,
and to an exhaustive search over every partition of 3000 random
transcripts. 20000 words take 17 ms on an M4 Pro.

`--json`'s `meta` is the model and the audio's length, which are the same
on every machine and run, so that transcripts compare byte for byte; with
`-v` it adds what the `-v` line says: the VAD's and transcription's
seconds, the speed, pktito's version, the backend and the threads.

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
a token (core/decoder.c). The transcendental functions are pktito's own
(core/vmath.h), not libm's, whose last bits differ between platforms.

## The library

```c
#include "pktito.h"

pk_model *m = pk_model_load("models/parakeet-tdt-0.6b-v2.pkf", err, sizeof err);
pk_asr *a = pk_asr_new(m, 0);                     /* 0: a thread a core */

const pk_tok *t;
long n = pk_asr_run(a, samples, n_samples, &t);   /* 16 kHz mono float */
pk_text(m, t, (size_t)n, text, sizeof text);      /* NeMo's transcript */
/* t[i].frame * pk_model_frame_seconds(m) is when token i was heard */

pk_vad *v = pk_vad_new(m);                        /* Silero VAD, as vadito has it */
pk_vad_probs(v, frames, n_frames, probs);
size_t ns = pk_segments(probs, n_frames, 16000, n_samples, &sp, seg, cap);
size_t nc = pk_chunks(samples, seg, ns, n_samples, 30 * 16000, chunk);
pk_asr_run_chunks(a, samples, chunk, nc, show, arg);
/* show(arg, i, t, n) has chunk i's tokens, its decoder run beside chunk i + 1's encoder */

pk_capword w[] = { { "Hello", 1000, 1300 }, { "world.", 1400, 1800 } };   /* ms */
long nq = pk_captions(w, 2, 1, 0, cue);           /* 1: English; 0: the end unknown */
/* cue[0]: words 0-1, one line, 1000-2000 ms */
```

A model is read-only once loaded and may be shared by transcribers on any
number of threads; a transcriber belongs to one thread at a time and runs
its own pool. Every path it takes is UTF-8, on Windows too, whatever the
process's code page. `make libpktito.a` builds the library alone;
core/pktito.h is all of its interface, core/wav.h and core/resample.h the
WAV and AIFF library's.
examples/transcribe.c is a whole program against it in a page, and `make
check-c` holds its transcript to the CLI's. examples/ has the same program
in Python, Go, Rust and Node, Go and Rust linking the archive and Python
and Node the shared library `make shared` builds, libpktito.so (.dylib on
macOS, .dll on Windows): examples of the wiring, not packages
(examples/README.md).

## The model file

A `.pkf` (core/pkf.h) is the recognizer, its vocabulary and Silero VAD in
one file, mapped read-only and used in place, with a metadata table that
says what it was made from: the HuggingFace repo, file, commit and sha256
of the checkpoint, its licence and the attribution CC-BY-4.0 asks for, the
VAD's source and licence, the converter's version and the recipe. `pktito
info` prints it. The loader holds every extent, shape, type and value of the
file to what the architecture and docs/MEMORY_SAFETY.md say before it forms
a pointer, and tests/pkfcheck's 29,186 cases run malformed files at it.

`tools/pktito-tools convert` writes one; `info` lists one; `ref` runs the
float64 reference on a WAV from either a `.nemo` or a `.pkf`; `tiny` writes
a 1.2 MB model with random weights that has every tensor, param and piece
type, for the checks.

## Checking

```sh
make check-c                 # a compiler and a model: any board
make check                   # and the Python half: sentencepiece, Silero, torchaudio, the corpora
make check-sanitize          # the model file and WAV/AIFF parsers under ASan and UBSan
make check-tsan              # the pool under ThreadSanitizer
make check-math              # every float through exp, sigmoid, tanh and log
make check-ffi               # the examples in other languages, where they are installed
make remote HOST=user@board  # rsync, build, check-c and bench there
```

`check-c` is the math library's bounds, every kernel of every backend the
CPU can run against `ref` bit for bit (tests/kcheck: 1,525 cases each), the
model file's validator on its malformed corpus and the whole engine on the
tiny model, the WAV and AIFF reader on its own, a WAV and a model under a name
past ASCII (on Windows, the regression test for UTF-8 paths), and
test.wav through `./pktito-c` and
through examples/transcribe:
vadito's VAD bits, and both models' transcripts byte for byte. `check` adds
the detokenizer against sentencepiece, the state machines against Silero's
Python, the resampler against torchaudio and the corpora's WER.
tests/README.md has each check.

## The tree

```
core/        the model (encoder.c, decoder.c, mel.c), the file (model.c, pkf.h),
             the text (tokens.c), the transcriber (asr.c), the VAD (vad.c,
             segment.c), the WAV and AIFF library (wav.c, resample.c), the captions
             (captions.c), the math (vmath.h), the pool (pool.c), the CLI
             (cli.c)
backend/     the seam (kernels.h, and kernels.c, which picks the backend that
             runs) and ref/, neon/ (and neon-dp/), avx2/ (and avx2-vnni/,
             avx512-vnni/, amx-int8/); a CLIMB.md beside each tuned backend
tools/       the converter, the float64 reference, the corpora, the
             quantization experiment
tests/       the C checks, the Python ones, and test.wav with its golden files
examples/    the library from the outside: from C, Python, Go, Rust and Node
docs/        MEMORY_SAFETY.md: the rules the two parsers keep; caption-spec.md:
             the method core/captions.c follows
```

## License

MIT for the code. The weights a `.pkf` holds are NVIDIA's, under CC-BY-4.0;
whoever shares a `.pkf` owes NVIDIA that licence's attribution, which every
file carries in its metadata. Silero VAD's weights, and the parts of the code
that port its model and state machines, are MIT, Copyright (c) 2020-present
Silero Team. LICENSE has all three.
