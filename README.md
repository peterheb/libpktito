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
| backends | `neon-sve` (SVE and I8MM's SMMLA: the GB10's X925 and A725), `neon-dp` (ARMv8.2 + SDOT: Pi 5's A76, Apple M), `neon` (ARMv8.0: Pi 3's A53, Pi 4's A72), `amx-int8` (AMX's tiles: Sapphire and Granite Rapids), `avx512-vnni` (AVX512-VNNI in zmm: Ice Lake and Tiger Lake, Zen 4 and 5), `avx2-vnni` (AVX-VNNI: Alder Lake and later Intel cores without AVX-512), `avx2` (Haswell and later), `ref` (scalar C, anything) |
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
(neon-sve, neon-dp and neon on AArch64; amx-int8, avx512-vnni, avx2-vnni
and avx2 on x86-64) and the CPU picks the first it can run, so a binary
copied to another machine of the family runs there at that machine's
best: one built on an Alder Lake runs on a Zen 4, one built on a Pi 5 on
a Pi 3.
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
12900HK at 14 threads, 711 MB on an M4 Pro at 12, a chunk at a time. A
recording run several chunks at once (below, "How fast") takes about 45
MB more for each chunk beyond the first: AliceBench's longest chapter
peaked at 762 MB a chunk at a time on a 9950X's 16 threads, 896 MB four
at once. The VAD's encoder keeps 0.85 MB a thread besides, 3.4 MB on a
Pi's four (below, "Transcribing a recording"). A 512 MB board (a Pi
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
a core by default). A recording longer than `--max-chunk` (30.72 s) is cut at
its pauses into chunks that long at most, each transcribed whole, a cut in
mid-sentence heard past, and speech the model wrote no word for heard again;
`--no-context --no-retry` leaves those out, for NeMo's transcript of each
chunk ("Transcribing a recording").
`--stream`'s lines end at a pause of `--pause` (1000 ms). Both defaults are
measured below. `--fast` trades a little of the transcript's nearness to
float32's for up to 7% of the time (README, "Why int8").

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
its time (`test.v2.fast.json` and `test.v3.fast.json` in the fast mode), and
`make check-c` holds every backend on every machine to them byte for byte: so
far `neon-sve` on a DGX Spark's GB10 (gcc 13.3 and clang 18.1), `neon-dp` on
an M4 Pro (Apple clang 21) and a Pi 5 (gcc 14.2), `neon` on a
Pi 3B+ (gcc 14.2) and the M4 Pro, `avx2` and `avx2-vnni` on a 12900HK (gcc
15.2) and on a Core 3 304 under Windows 11 (MSYS2's gcc 16.2), `avx2-vnni`,
`avx512-vnni` and `avx2` on a Ryzen 9 9950X (gcc 15.2), `amx-int8` on a
Sapphire Rapids Xeon (gcc 13.3), and `ref`; and the 304's Windows build,
copied as it was, runs `avx512-vnni` on a Zen 4 and all three on the 9950X,
the same bytes.

## How fast

`./pktito-c bench`: Silero's test.wav (60 s of English), best of five,
2026-10-20 (the 9950X's 2026-10-24). A machine's first row is what it picks itself, its default
thread count and the fastest backend its CPU can run; the rows under it
force another backend (`PKTITO_KERNELS`) or one thread. Seconds, and how
many times real time, for the minute whole (`--max-chunk 60`, 750 encoder
frames) and in 30 s chunks, the default then:

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
thread on an SMT core adds nothing), of those the process may run on: under
`taskset`, systemd's `CPUAffinity=`, a container's CPU set or Windows'
`start /affinity`, the ones they leave it. A chip's slower cores take fewer of
each pass's items and still add their share: a Core 3 304, one P-core and
four E-cores, runs the minute whole in 5.02 s on its P-core and 1.39 s on
all five; a 12900HK in 0.96 s on its six P-cores and 0.91 s on all 14
(though 8, 16 and 18 threads take 1.10 to 1.15 s there, the same in every
pass); an M4 Pro in 0.50 s on its eight performance cores and 0.49 s on
all 12. backend/*/CLIMB.md has every measurement the kernels and the
defaults rest on.

A recording of several chunks keeps more of many threads busy taken
several chunks at a time: one transcription's passes each wait for their
slowest item, and on a chip of many cores they wait more than they work.
On 8 threads or more `pk_asr_run_chunks` runs up to a chunk for each four
threads at once, each on its share of them, as many as the chunks'
lengths say end the recording soonest, and gives them to its caller in
order; the transcript is the same bits. Against a chunk at a time
(d2c70f0), through the v2 model, best of two interleaved rounds (the M4
Pro's and the 12900HK's three; the bench's best of five, in two to
four), 2026-11-02
(backend/avx2/CLIMB.md):

| | AliceBench | 26 talks | long40 | test.wav, `bench` in chunks |
|---|---|---|---|---|
| M5 Ultra, 36 threads, Apple clang 21 | 201x -> 407x | 189x -> 376x | 195x -> 386x | 188x -> 334x |
| M4 Pro, 12 (8 P + 4 E), Apple clang 21 | 120x -> 131x | 125x -> 130x | | 142x -> 146x |
| i9-12900HK, 14 (6 P + 8 E), gcc 15.2 | 70x -> 75x | | 70x -> 74x | 77x -> 83x |
| Ryzen 9 9950X, 16, gcc 15.2 | 259x -> 316x | 254x -> 304x | 256x -> 298x | 289x -> 308x |
| Ryzen 9 9950X, 16, clang 21.1 | 266x -> 328x | 260x -> 317x | 263x -> 312x | 296x -> 320x |
| GB10, 20 (10 X925 + 10 A725), gcc 13.3 | 141x -> 182x | 136x -> 175x | | |
| Galaxy S24 Ultra, 8 (Snapdragon 8 Gen 3), NDK clang 21 | 29x -> 31x | | | 37x -> 42x |
| Galaxy S25 Ultra, 8 (Snapdragon 8 Elite), NDK clang 21 | 37x -> 41x | | | 49x -> 54x |

The 9950X's, the 12900HK's and the GB10's AliceBench is six of the
twelve chapters, the 9950X's and the 12900HK's long40 half of the 40, the
GB10's talks 13 of the 26. The phones' AliceBench is chapter 1 alone, on
AC, the best of four rounds 180 s apart, lanes faster in every one. The GB10
ran a chunk at a time on its ten X925s, which beat all twenty there; as
lanes of their own, its A725s add their share. A Pi's four threads run a
chunk at a time, as before: on a Pi 5 two lanes of two threads
transcribed at one lane's rate and four of one 9% slower, on a Pi 4 3%
and 4% slower. The bench table at the top of this section is from
before: its chunked column now runs two lanes on 8 threads or more, and
one on the Pis.

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

Blocks of 64 halve what the blocks cost - the convert and fused
multiply-add each block's integer sum ends in, and the quantizer's maximum
and divisions - for a little of that nearness, and are the fast mode's
(`--fast`, `pk_asr_set_fast_mode`). Through `tests/wercheck.py`, v2 over
LibriSpeech is 1.47% in it, 7 transcripts not float32's (4 at 32), and v3
over FLEURS 4.43%, 49 (45); the WER between them and float32's 0.07% and
0.50% (0.07% and 0.35%). On the boards measured a run in it is 3.9 to 7.3%
faster where the int8 product is a dot product (SDOT, VPDPBUSD), 5.5% on a
Pi 4 and 3.2 to 3.4% on a Pi 3 with the ARMv8.0 loop, and 1 to 3% with
x86's VPMADDUBSW, avx2 forced on two of them (backend/avx2/CLIMB.md and
backend/neon/CLIMB.md, 2026-10-29). The default stays the transcript
nearest float32's.

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

## Transcribing a recording

The library has two ways in, a low level and a high one.

**The low level** is `pk_asr_run`: samples in, tokens with their times out,
all of the samples through the encoder at once. Its transcript is NeMo's
transcript of exactly those samples, bit for bit, and it does nothing else -
transcript = f(x). Beside it are the blocks pktito builds everything else
from, each as plain: `pk_vad_probs` and `pk_segments` (Silero's VAD and its
speech segments), `pk_chunks` (a recording cut into chunks), and
`pk_asr_run_chunks` (chunks transcribed, each chunk's decoder beside the
next one's encoder, several chunks at once where there are threads for
it). A caller who wants NeMo's behaviour, or processing of
their own, builds from these; examples/transcribe.c is a page of it.

**The high level** is `pk_asr_transcribe`: a recording of any length in, each
chunk's tokens out as soon as the chunk is done, with everything pktito
does for a long recording done for you. `pktito-c` transcribes with it. It
does, in order:

1. **Finds the speech.** Silero's VAD gives a probability of speech every
   32 ms, and `pk_segments` turns them into segments as Silero's
   `get_speech_timestamps` does, a segment longer than a chunk split at
   its longest pause. Silero ends a segment at a pause of 160 ms (its
   `min_silence_ms`, 100, is five frames); the chunks are cut between the
   same segments ended at 64 ms, two frames, and the retry goes by Silero's.
   Silero's encoder, which a frame computes from its own samples, runs
   over the transcriber's threads, and only its LSTM on one: the same
   probabilities, 1.9 times as fast on a 9950X's 16 threads and a Pi 5's
   4, 2.9 on an M5 Ultra's 36 (backend/avx2/CLIMB.md, 2026-11-02).
2. **Cuts the recording into chunks** of at most 30.72 s (`max_chunk_s`,
   `--max-chunk`). A recording that fits is one chunk, heard whole, as
   NeMo hears a file. A longer one is cut at the last pause that fits, half
   way into it and at most a second into a long one, and the rest of the
   silence is left out. 30 s is where the error rate is as low as whole
   recordings' and the encoder's cost a fraction of theirs ("Long
   recordings"), and 30.72 s is 3072 mel frames, the length the Neural
   Engine's encoder is built for. A chunk whose cut falls in a silence under
   a second is heard with 2 s of the recording past it, room left in the
   chunk for that ("Where chunks meet").
3. **Keeps what lies between the speech.** Silero hears voices, not words,
   so singing and speech over music are not speech to it. What lies between
   the speech chunks is transcribed too, in pieces of at most a chunk, but
   not a silent piece, nor one under 3 s, of which the model makes words
   out of anything ("Between the speech").
4. **Transcribes each chunk** as `pk_asr_run` would, the same bits, the
   search over one chunk running beside the next chunk's encoder: 1.2 to
   5.6% less time where there are more than four cores. Of a chunk heard
   past its cut it keeps the words that start before the cut, and the next
   chunk opens with that chunk's copies of its first words, up to where the
   two agree.
5. **Hears dropped speech again.** A stretch of a chunk that Silero heard
   speech in but the model wrote no word for is transcribed again on its
   own (below). This and the context are where the high level leaves
   NeMo's transcript, on purpose; `context = 0` and `retry = 0`
   (`--no-context`, `--no-retry`) leave them out, and each chunk is then
   `pk_asr_run`'s.
6. **Hands each chunk to the caller** as it is done: its place in the
   recording, its tokens, how many of its stretches were heard again, and
   how far the run has got. The caller's function returns nonzero to stop
   the run there.

**In steps**, `pk_steps` is the high level with the transcribing handed to
the caller, for one who runs it on threads of their own: a server dealing
several recordings' chunks among its workers. `pk_steps_new` finds the
speech and cuts the chunks (1 to 3), and the rest is jobs, each a stretch
to hear - a chunk whole, or a stretch the retry hears again - which the
caller hears with `pk_asr_run` on any transcriber, in any order, and hands
back with `pk_steps_put`. `pk_steps_take` has the chunks in order (4 to 6),
the bits `pk_asr_transcribe` gives, which is the same steps with its own
hearing of the jobs. The retry's stretches come one at a time, each heard
only while the recording's budget for them lasts. The VAD on its own is
`pk_asr_vad`, Silero's probabilities over a recording on the transcriber's
threads, and `pk_asr_speech`, its speech segments.

### Where chunks meet

The model writes a chunk as one utterance, so a cut in mid-sentence gets a
period, and the chunk after it a capital. At 30 s, of the cuts in long40
(LibriSpeech test-clean, 40 recordings, 2.6 hours) that line up with the
whole recording's transcript (`--max-chunk 3600`), v3 ended a sentence at
7% where that transcript goes on, v2 at 10%; in 26 English recordings of
court arguments, interviews and press conferences (89 minutes), v3 at 19%
and v2 at 37%.

So a chunk whose cut falls in a silence under a second is heard with 2 s of
the recording past the cut, and keeps the words that start before it: the
model has heard the sentence go on. The next chunk opens where the cut is,
and its first words, heard at the head of its window, have a copy in the
last chunk's, heard with what came before them. Up to the first of its
first four words that agrees with one of those - the same letters, case
and punctuation aside, starting within 4 encoder frames - the chunk opens
with the last chunk's words, and that word has the last chunk's letters
where the two differ in case alone. That takes out the capital, and a word
the cut fell beside that the next chunk wrote again ("again, there | There
was no") or in part ("reve" for "grieve").

| | long40, v3 | long40, v2 | 26 talks, v3 | 26 talks, v2 |
|---|---|---|---|---|
| false sentence ends at a cut | 22 → 10 (3%) | 30 → 11 (3%) | 36 → 7 (4%) | 70 → 2 (1%) |
| false capitals after one | 22 → 4 | 23 → 10 | 37 → 5 | 80 → 2 |
| sentence ends lost at one | 5 → 2 | 7 → 8 | 5 → 4 | 1 → 5 |
| WER | 2.27% → 2.19% | 1.89% → 1.95% | | |
| chunks | 350 → 368 | 350 → 368 | 226 → 239 | 226 → 239 |

The cost is the audio heard twice, 5.1% more on long40 and 7.3% on the
talks, and the chunks, which leave room for it. The false ends left are at
pauses: their median on the talks is 2 to 3 s, where none is heard past.
WER moves as it does when any cut moves - per recording, by up to six
errors either way, words the model spells either way ("colour", "color")
as its windows move. Without the retry 2 of
long40's 368 chunks dropped speech with v3 (68 words, WER 2.54%) and 1 with
v2 (53); the retry hears it again. On AliceBench v2 drops three phrases
from windows heard past their cut, phrases that chunks without context
keep (the chunks of 2026-10-31, and `--no-context`'s): "said the Cat"
(chapter 6), "it was Bill, the lizard" (11) and "And how many hours a day
did you do lessons?" (9). v3 drops none. Each drop depends on where its
window ends, not on whether the window goes past the cut. With the window
ending 0, 0.25, 0.5 and so on up to 3 s past the cut, chapters 6 and 9
lose theirs at 2 s alone, and chapter 11 at 6 of the 12 ends. The retry
hears chapter 9's again. The other two are under the speech it needs: 1 s
after a sentence ends, 2 s inside one.

A cut can fall inside the last chunk's last word, which that chunk keeps,
since the word starts before the cut. The next chunk may open with the
same word again: if it has the same letters and starts before the last
chunk's copy ends, it goes ("talking familiarly | Familiarly with them" in
AliceBench's chapter 3, "Lewis Carroll. | Carroll." in chapter 10, v2).
The retry does not hear again what the last chunk's words cover of the next
chunk's head; in chapter 3 it had put back "Familiarly with the" after
the seam took it out.

A chunk ends at the last pause that fits, which makes as few chunks as
those pauses allow. At Silero's 160 ms, conversation goes 10 to 21 s
without one, and a segment longer than the room left ended its chunk
early: 52 of the talks' 239 chunks ended under 20 s. So the chunks are cut
at pauses of 64 ms, and are 30.72 s at most, the Neural Engine encoder's
length, which pads a shorter chunk to it: at 30 s every window was at least
2.3% padding. A window there costs the same however much it holds, so the
chunks are most of its time. A CPU's work is the audio heard, and that is
about the same: on long40 4.9% of it heard twice, where it was 5.1%.

| from 160 ms and 30 s | long40, v3 | long40, v2 | 26 talks, v3 | 26 talks, v2 |
|---|---|---|---|---|
| chunks | 368 → 354 | 368 → 354 | 239 → 217 | 239 → 217 |
| false sentence ends at a cut | 10 → 11 | 11 → 14 | 7 → 5 | 2 → 2 |
| false capitals after one | 4 → 6 | 10 → 13 | 5 → 3 | 2 → 1 |
| WER | 2.19% → 2.12% | 1.95% → 1.99% | | |

Each part on its own, long40's chunks and the talks': 64 ms 364 and 223,
96 ms 366 and 231, 192 ms 371 and 243, 288 ms 373 and 250; 30.72 s 359 and
235. The fewest that could hold the words, cut anywhere between two and
with no silence or context in a chunk, are 333 and 193.

What did not work, on the same recordings (v3 unless named, no retry):

- **2 s heard before the cut too**, so that the next chunk opens in the
  sentence: 1 false capital, but a window that opens inside speech, not in
  the silence a chunk opens in, makes the model drop it - 22 of 386 chunks
  lost 771 words (WER 5.65%). With those edges at the quietest 100 ms
  within half a second of 2 s, 18 chunks lost 571.
- **The search carried over the cut** in the last chunk's state, so that
  it knows the sentence is under way: the model then writes nothing for
  the first seconds of a window, in 102 of 350 chunks with no context
  heard and in 210 of 386 with 2 s each side.
- **Whole speech segments either side of the cut**, so that every window
  opens and ends in a silence, with the cuts moved as in the next item: no
  speech dropped, but those within 6 s took v2's false ends on the talks
  only from 37% to 19%, for 17% more audio, and those within 10 s to 3%,
  for 92% more.
- **Cutting at the longest silence** in the 8 s before a chunk is full:
  fewer cuts in mid-sentence on read speech (46% to 17%), none fewer on the
  talks, and no fewer false ends.
- **1 s past the cut**: v2 kept 15 false ends on long40 and 7 on the talks.
  3 s did no better than 2, for 7.3% and 11.4% more audio.
- **Cutting at the longest pause**, the cuts planned over the whole
  recording for as few chunks as the last pause makes (both models, the
  retry on): the talks' false ends 7 to 9 (v3) and 2 to 5 (v2), for more
  cuts in a pause over a second, where none is heard past.
- **The last pause of a segment longer than a chunk**, not Silero's
  longest: 2 of the talks' 239 chunks fewer, none of long40's.

### When the model drops speech

Parakeet now and then writes nothing for speech it plainly hears. Over the
12 chapters of LibriVox's *Alice's Adventures in Wonderland* (AliceBench,
3 hours), v2 gives nothing for the first 28 s of chapters 8 and 10, of the
first chunk of chapters 2, 7, 9 and 12 keeps only LibriVox's announcement
and drops the reader's name and the chapter's title after it, and loses
chapter 1's "End of chapter one"; v3 drops chapter 5's closing chunk, chapter 1's closing credits
and lines of chapters 4, 5 and 10. It is the model's failure, not
pktito's: at every frame of chapter 10's first chunk the search visits,
blank beats the next word (by 0.02 logits at the closest), and NeMo 3.0's
own float32 transcription of chapter 8's first chunk is empty too. Heard
again on its own, each of those stretches gives its words.

So the high level hears a stretch again, once, when:

- no word of the chunk covers it - before the first word, between two, or
  after the last - and Silero heard at least 2 s of speech in it, or 1 s
  where it follows the end of a sentence or ends the chunk. At 1 s inside a
  sentence a FLEURS Polish recording's "km", said "kilometrów
  kwadratowych", left a second after it that gave "metrów kwadratowych." a
  second time; at 2 s between sentences AliceBench's chapter 1 lost "End
  of chapter one", 1.6 s after "...finished off the cake.";
- its speech is no more than 4 dB quieter than the recording's speech, or
  3 dB quieter than its chunk's (below). What the model drops is the voice
  the rest of the recording is in; a voice in the background, which Silero
  hears as speech too, is quieter, and hearing it again puts the
  background's words in. AliceBench's dropped stretches lie -3.6 to +2.7
  dB from their chapters' speech. The exception is chapter 9's "And how
  many hours a day did you do lessons?" (v2), in a passage read softly:
  4.9 dB under the chapter's speech, 1.8 dB under its chunk's. Pink noise
  and a telephone band draw no retry, nor do two songs;
- the stretches heard again for nothing so far are under a fifth of the
  recording, or a chunk if that is more: music that Silero takes for
  speech gives no words however it is cut, and costs at most a fifth more
  time. A stretch that gives words costs none of it, since it is speech
  the model dropped. When every stretch was charged, long40's 1320-122612
  (134 s, v3, chunks cut at Silero's own pauses into 30.72 s) left its
  second dropped stretch, 49 words, unheard after its first. A stretch is
  heard again at most once, so the retries hear at most the recording
  over again, whatever they give.

The level's test was set in pktito-app on LibriSpeech recordings with
three other speakers mixed in. At 10 dB SNR stretches were retried down
to -24 dB, and WER went from 6.21% to 9.20% with every one heard again,
and to 6.02% at 4 dB; at 5 dB SNR, 19.88%, 21.91% and 19.79%. Mixed again
from long40, each of its first 24 recordings with the next three
speakers looped to its length, WER is:

| | v2, 10 dB | v2, 5 dB | v3, 10 dB | v3, 5 dB |
|---|---|---|---|---|
| no retry | 3.82% | 12.05% | 14.35% | 31.48% |
| every stretch heard again | 5.48% | 15.60% | 10.57% | 24.29% |
| 4 dB under the recording's | 3.86% | 13.60% | 8.11% | 22.62% |
| or 4 dB under its chunk's | 3.91% | 13.91% | 8.04% | 22.75% |
| or 3 dB under its chunk's | 3.86% | 13.62% | 7.99% | 22.62% |

Under the babble, v3 drops much of the reader's speech, and the retry
hears it again. v2 drops less, and at 5 dB the retry costs it 1.55 points
in the background's words. A chunk's own speech at 3 dB takes back
chapter 9's line and leaves the mixes as they were. At 4 dB it lets in
more of the background.

A chunk that gave no words at all is heard again in two halves, cut at its
quietest 300 ms within its middle third, since whole it would give nothing
again. A retry's tokens go in among the chunk's at their own times. In
pktito's AliceBench run (2026-11-02) the retry gives v2 177 more words in
5 of the 12 chapters and v3 181 in 6, WER against the book's text from
2.55% to 2.18% and from 3.00% to 2.46%, and every chunk with nothing to
retry is the same bits as `--no-retry`'s. On the sets the rest of this
README measures - ls200 (v2), the six FLEURS sets (v3), and the two
LibriSpeech long-form sets with both models, 1,224 recordings in all - it
heard nothing again in the chunks of 2026-10-29, and every transcript and
WER was as without it (`tests/wercheck.py --retry`). When chunks are heard
past their cuts, a few of long40's and long300's drop speech, which the
retry hears again ("Where chunks meet").

The rules and their numbers were found in pktito-app, which shipped them
first (2026-10-26), and are core/transcribe.c's now, with the same
measurements in its comments.

### How it got here

- **2026-10-20**: recordings longer than a chunk cut at Silero's silences,
  each chunk transcribed whole; 30 s chosen over 10 s to whole recordings
  on 45 minutes of stitched LibriSpeech, then checked over every speaker of
  test-clean, no chunk lost ("Long recordings").
- **2026-10-22**: what lies between the speech transcribed too, after a
  sung recording lost three quarters of its words; silent pieces and
  pieces under 3 s left out, after the model made words of 27 of 28
  one-second pieces of music ("Between the speech").
- **2026-10-26**: the retry, in pktito-app, after AliceBench's chapters
  came back without their openings; gated at 4 dB after recordings with
  background speakers.
- **2026-10-28**: each chunk's decoder beside the next chunk's encoder
  (`pk_asr_run_chunks`), 1.2 to 5.6% faster on more than four cores.
- **2026-10-29**: all of it one call, `pk_asr_transcribe`, with a way to
  stop, so that a program gets what pktito-app does without writing it
  again, and pktito-app the pipelined run it could not use while the retry
  and its Stop button lived outside the library.
- **2026-11-01**: a cut in a short silence heard 2 s past, and the seam,
  after pktito-app's transcripts broke a sentence at every cut: false
  sentence ends on 26 talks from 19% of cuts to 4% (v3) and 37% to 1% (v2)
  ("Where chunks meet").
- **2026-11-02**: chunks cut at pauses of 64 ms and 30.72 s long, for the
  Neural Engine's windows: 4% fewer chunks on read speech, 9% on the talks,
  WER and false ends at the cuts about level ("Where chunks meet").
- **2026-11-02**: after pktito-app's AliceBench run, three fixes:
  - the word a cut fell in is no longer written twice;
  - the retry no longer hears again what the last chunk's words cover;
  - a stretch in a passage read softly is heard again, against its chunk's
    speech.
  The retry's budget is now charged only for stretches that give nothing.
- **2026-11-02**: a recording's chunks several at once on 8 threads or more
  (`pk_asr_run_chunks`), after pktito-mac's profile had an M5 Ultra's CPU
  path keeping 21 of its 36 cores busy: twice as fast there, a fifth on a
  9950X, the same bits ("How fast").
- **2026-11-02**: Silero's encoder over the transcriber's threads, its LSTM
  alone on one: the VAD 1.9 to 2.9 times as fast, a recording 2% faster on
  a 9950X and 6% on an M5 Ultra, the same bits ("Transcribing a
  recording").

## Long recordings

The encoder hears a recording whole - its attention is over every frame -
so its work grows with the square of the length and its memory with the
length. pktito cuts anything longer than `--max-chunk` at its pauses: Silero's
VAD finds the speech, segments longer than a chunk are split at their
longest pause, and each cut falls half way into the last pause of 64 ms that
fits, at most a second into it. Over 12 LibriSpeech chapters stitched
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
took on the M4 Pro with the VAD on one thread. Speech is cut as it was, and recordings of it barely
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
pk_asr_set_fast_mode(a, 1);                       /* optional: faster, "Why int8" */

/* the high level: a recording of any length, VAD, chunks and retries done */
int show(void *arg, const pk_chunk_out *c) {      /* each chunk as it is done */
    pk_text(m, c->toks, (size_t)c->count, text, sizeof text);
    return stop_wanted;                           /* nonzero stops the run here */
}
pk_asr_transcribe(a, samples, n_samples, NULL, show, arg);   /* NULL: the defaults, 30.72 s chunks */
/* c->at.start / 16000. + c->toks[i].frame * pk_model_frame_seconds(m): when token i was heard */

/* the low level: NeMo's transcript of exactly these samples */
const pk_tok *t;
long n = pk_asr_run(a, samples, n_samples, &t);   /* 16 kHz mono float */
pk_text(m, t, (size_t)n, text, sizeof text);      /* NeMo's transcript */
/* t[i].frame * pk_model_frame_seconds(m) is when token i was heard */

/* the high level in steps, for a caller who hears the chunks itself */
pk_steps *s = pk_steps_new(a, samples, n_samples, NULL);     /* the VAD and the chunks */
pk_job j;
pk_chunk_out c;
while (pk_steps_job(s, &j) == 1) {                /* any transcriber, any thread, any order */
    n = pk_asr_run(b, samples + j.at.start, (size_t)(j.at.end - j.at.start), &t);
    if (pk_steps_put(s, &j, t, n) < 0)            /* -1: memory ran out, and the run is over */
        break;
    while (pk_steps_take(s, &c) == 1)             /* the chunks in order, as show has them */
        ;
}
pk_steps_free(s);

long nsp = pk_asr_speech(a, samples, n_samples, NULL, seg, cap);  /* the speech; NULL: Silero's defaults */
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
tests/README.md has each check. A build copied to Windows without MSYS2,
which has no make or sh, runs check-c's commands with `powershell -File
tests\check-c.ps1`, given `-V2` and `-V3` for models kept outside models\.

## The tree

```
core/        the model (encoder.c, decoder.c, mel.c), the file (model.c, pkf.h),
             the text (tokens.c), the transcriber (asr.c), the VAD (vad.c,
             segment.c), the WAV and AIFF library (wav.c, resample.c), the captions
             (captions.c), the math (vmath.h), the pool (pool.c), the CLI
             (cli.c)
backend/     the seam (kernels.h, and kernels.c, which picks the backend that
             runs) and ref/, neon/ (and neon-dp/, neon-sve/), avx2/ (and
             avx2-vnni/, avx512-vnni/, amx-int8/); a CLIMB.md beside each
             tuned backend
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
