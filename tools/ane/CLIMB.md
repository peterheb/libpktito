# tools/ane: the climb

Dated records of what was measured, on which machine, for the encoder on
the Neural Engine (encoder.py's program, core/ane.m's loading): append-only.
A later section may supersede an earlier one, but the earlier one stays,
rejected hypotheses included.

The entries dated before 2026-11-08 were measured on pktito's ane fork. Its
history was squashed into the commit that brought this file to main, so the
commits they name are the fork's and are not in this repository.

# 2026-10-31 - the first program, and the position term

## The machine and the tools

|            |                                                                                                                         |
| ---------- | ----------------------------------------------------------------------------------------------------------------------- |
| machine    | Mac mini, M4 Pro (Neural Engine of 16 cores), macOS 27.0.1, in use (Chrome, Claude) while measuring                     |
| toolchain  | coremltools 9.0, Python 3.12, Xcode 27.2; programs at iOS 18's opset, fp16                                              |
| model      | parakeet-tdt-0.6b-v2.nemo through canon.py                                                                              |
| per call   | a function's prediction alone, best of 10 after one untimed, through coremltools on random features (a scratch harness) |
| whole runs | `./pktito-c bench --ane` on AliceBench ch1 (640.5 s, 23 chunks), best of 5                                              |
| accuracy   | anecheck.py over tools/corpus.py's 200 LibriSpeech recordings, against float32                                          |

## What the program costs, and where

The first program wrote the attention as NeMo does: every query scored
against the 2T-1 relative positions, then rel_shift (a pad, two reshapes,
two slices) brings score (i, j) its position i - j. Every op ran on the
Neural Engine (placement.py: 1147 of 1147), and a function's time against
its length was far from linear:

| function | ms a call | ms a second of audio | x real time |
| -------- | --------- | -------------------- | ----------- |
| 5 s      | 10.5      | 2.09                 | 477         |
| 10 s     | 13.8      | 1.38                 | 726         |
| 15 s     | 25.6      | 1.71                 | 586         |
| 20 s     | 40.6      | 2.03                 | 492         |
| 30 s     | 91.0      | 3.03                 | 330         |
| 40 s     | 139.7     | 3.49                 | 286         |

The 30 s function, with one part taken out at a time (wrong numbers, right
shapes, for the time alone):

| variant                                    | ms   |                                           |
| ------------------------------------------ | ---- | ----------------------------------------- |
| as written                                 | 91.1 |                                           |
| no rel_shift (the scores' first T columns) | 60.5 | the shift: 31 ms                          |
| no position term at all                    | 55.7 | the term with its shift: 35 ms            |
| positions' projection a constant           | 88.7 | computing it: 2.4 ms                      |
| no attention                               | 40.4 | the feed-forwards, convolutions and norms |

## The shift's other forms (rejected)

- `gather_along_axis` with constant indices: 2.58 ms for 24 in a program
  of its own, but in the encoder Core ML runs the 24 gathers on the CPU:
  223.5 ms a call.
- A flattened slice (scores to [H][T(2T-1)], a slice from T-1, back to
  [T][2T-2]): its reshapes and slices go to the CPU too: 122.6 ms.

## The position term turned, not shifted (kept)

pe(i - j) is pe(j) turned through i w in each sin/cos pair, so the term is
the query turned (W_pos^T a head at a time, a grouped 1x1 convolution, and
again with each pair's halves swapped; products with cos and sin tables)
then a product with the fixed pe(j). The same numbers to float32's 1e-6
(enccheck.py). 30 s: 75.4 ms. Its parts: 64.1 with the turn's products
left out, 65.0 with one of the two; the second product and the sum cost
10 ms. Keys in rows, queries in columns, the term a grouped convolution
whose weights are pe: 74.9. One convolution for both halves, one product
over [2D], a product with [pe; pe] in place of the sum: 77.2. Neither kept.

## Lengths of 64 encoder frames (kept)

3072 feature frames (384 encoder frames) take 71.1 ms where 3000 (375)
took 75.4: a function's lengths are multiples of 512 feature frames.

## Seven lengths or one (one kept)

Every function Core ML compiles for the Neural Engine keeps its own copy
of the weights in /Library/Caches/com.apple.aned, for each program that
loads it: 7.2 s and 1217 MB of free space for one, then 0.33 s and
nothing (a renamed pktito-c, the 5.12 s function). Seven lengths cost 8.4
GB; one, f3072, takes every chunk of pktito's 30 s. AliceBench ch1: 1.721 s
(372.1x) with seven, 1.764 s (363.1x) with one; the same transcripts.

## Weights (per-channel int8 an option)

| weights                                       | WER   | changed | dlogp  | enc err | size   | AliceBench ch1  |
| --------------------------------------------- | ----- | ------- | ------ | ------- | ------ | --------------- |
| fp16                                          | 1.47% | 10      | -0.016 | 3.7%    | 1.2 GB | 1.764 s, 363.1x |
| int8, a scale an output channel (--w8)        | 1.49% | 8       | -0.025 | 4.4%    | 610 MB | 1.551 s, 413.1x |
| int8, a scale a block of 32 inputs (rejected) | 1.47% | 10      | -0.019 | 4.2%    | 646 MB | 3.153 s, 203.2x |

(float32 1.51%; pktito's int8 1.47%, 7 changed, -0.012, 3.4%.) The blocks
are nearer fp16 and half the speed: their scales are not a cost the Neural
Engine hides.

## The scores' 1/sqrt(dk) folded into the queries (no change)

The queries and both position biases carrying 1/sqrt(dk), so that the
scores need no product over [H][T][T] a layer: the same numbers in float32
to 1e-6, and AliceBench ch1 1.763 s twice, alternated with the program as
it is at 1.762 and 1.757: the Neural Engine's compiler fuses that product
already. Not kept.

## 8-bit look-up tables for the weights (rejected)

The Neural Engine's own compression (palettized weights), here 256 values a
table by k-means (1-D Lloyd's from quantiles, 12 rounds), measured offline
as each matrix's relative error over layers 0, 11 and 23's eleven matrices:
a table a tensor 0.044, a table for each 16 output channels 0.030, where
int8 with a scale an output channel (--w8) is 0.0105 and fp16 0.0002. Rows
of very different scales share a table badly. Not built.

## Where a run's time goes (make ANE=1 PROFILE=1)

AliceBench ch1, 23 chunks, 1.765 s a run: the features 33.1 ms and the
joint's encoder half with the wait for the search 19.6 ms, the rest the
encoder's calls, 23 of about 71 ms each and some 3 ms of Core ML's work
around each (fp16 in and out, the copies). Overlapping the features and the
joint with the next chunk's call would take back at most 3%.

# 2026-10-31 - the convolution module's sigmoids

## Where the error grows

A debug copy of the program giving every block's output (the residual
stream after each of layers 0 and 1's four blocks, and after layers 2 and
3), on LibriSpeech's 8455-210777-0028 in a 10.24 s function, against
reference.py in float64: 0.05% after the subsampling, 0.12% after layer 0,
0.23% into layer 1's convolution module and 0.75% out of it, then 0.55 to
0.84% a layer on. The convolution module alone on the Neural Engine, from
float64's input to it, a stage at a time (layer 1, then layer 10): its norm
0.06%, the GLU 0.20%, the depthwise convolution 0.66% (0.26%), the SiLU
after it 2.13% (2.05%), the module's output 2.35% (2.29%).

Neither the depthwise convolution with BatchNorm folded in (0.06% alone,
though layers 1 and 22 fold a scale of 1055 and 1159 into two channels
each, whose variance is below 1e-9) nor freezing those channels (0.92%
into layer 2 where it was 0.75%) is the cause: it is the Neural Engine's
sigmoid. mb.silu and x \* mb.sigmoid(x) (which its compiler makes silu)
are 2.05% from float64 after layer 10's depthwise convolution;
x / (1 + exp(-x)) is 0.25% (the module's output 0.26%). The GLU's
sigmoid likewise, less so (the depthwise convolution after it 0.70%, then
0.14% in a program of its own).

## The module's GLU and SiLU as y / (1 + exp(-x)) (kept)

| v2, LibriSpeech's 200, against float32      | WER   | changed | dlogp   | enc err | AliceBench ch1  |
| ------------------------------------------- | ----- | ------- | ------- | ------- | --------------- |
| mb.sigmoid, mb.silu                         | 1.47% | 10      | -0.016  | 3.7%    | 1.763 s, 363.4x |
| y / (1 + exp(-x)) in the convolution module | 1.51% | 1       | -0.0024 | 0.92%   | 1.774 s, 360.9x |

float32's WER, one transcript of 200 changed, and dlogp five times nearer
float32 than pktito's own int8 (-0.012, 7 changed, 3.4%), for 0.6% of the
time. The position table centred over keys (above) came with it: alone it
was -0.0152 for -0.0155, 9 changed for 10.

## The feed-forwards' SiLU by exp as well (measured, not kept)

| v2, LibriSpeech's 200          | changed | dlogp   | enc err | AliceBench ch1  |
| ------------------------------ | ------- | ------- | ------- | --------------- |
| the convolution module's alone | 1       | -0.0024 | 0.92%   | 1.769 s, 362.2x |
| and the feed-forwards'         | 1       | -0.0017 | 0.52%   | 1.926 s, 332.5x |

The feed-forwards' hidden layer is four times the module's width, and the
exp, sum and division there cost 9%, for a dlogp already five times nearer
float32 than pktito's int8.

## v3 over FLEURS (the six languages, 358 recordings; the two longer than 30.72 s left out)

| against float32 (4.43%)                     | WER   | changed | dlogp   | enc err |
| ------------------------------------------- | ----- | ------- | ------- | ------- |
| mb.sigmoid, mb.silu                         | 4.39% | 27      | -0.039  | 3.4%    |
| y / (1 + exp(-x)) in the convolution module | 4.43% | 10      | -0.0006 | 0.74%   |

(pktito's int8 over FLEURS, README: 45 changed, dlogp -0.024, enc err 5.2%.)

# 2026-10-31 - the M1 iPad

## The device and the tools

|           |                                                                                                                            |
| --------- | -------------------------------------------------------------------------------------------------------------------------- |
| device    | iPad Pro 11 (3rd generation), M1 (a Neural Engine of 16 cores), 8 GB, iPadOS 26.5                                          |
| app       | the iOS demo's measuring build (pktito-app's ios-demo branch), Debug at -O, libpktito this branch built with ANE=1         |
| recording | AliceBench ch1 (640.5 s, 23 chunks), each round pktito on the CPU, pktito with this encoder, and Apple's SpeechTranscriber |

## In two parts (kept, encoder.py's default)

The whole 30.72 s fp16 function failed to load on the M1 after about two
minutes; in two parts of 12 layers (--parts 2, 600 MB each) each compiles
in 10 s (20.4 s, once), and a
chunk takes about 100 ms (52 ms the first part, 48 the second, the
M1's Neural Engine 1.4 times the M4 Pro's 71 ms for the whole). (An hour
of apparent hangs before that was the app sitting at its model screen:
devicectl's --remove-existing-content had emptied its Application Support.)

| round        | pktito, CPU               | this encoder                             | Apple's |
| ------------ | ------------------------- | ---------------------------------------- | ------- |
| first launch | 13.21 s, 48.5x, 5.5 cores | 2.79 s, 229.9x, 0.6 cores (load 20.43 s) | 14.03 s |
| 1            | 13.11 s, 48.8x            | 2.76 s, 231.7x (load 0.27 s)             | 13.89 s |
| 2            | 14.82 s, 43.2x            | 2.77 s, 231.0x                           | 13.94 s |
| 3            | 19.38 s, 33.0x            | 2.75 s, 232.6x                           | 14.01 s |

(cores: the app's CPU seconds over the run's; thermal nominal throughout.)
The iPad's transcript with this encoder is the M4 Pro's but for one comma,
and pktito's own (CPU, int8) but for that comma, of 2,199 words.

# 2026-10-31 - int8 weights, again

## Only the matrices (kept for --w8)

coremltools' int8 had made every convolution's weights int8, the
subsampling's 3x3 taps and the depthwise convolutions' (BatchNorm folded
in, up to 907) with them, which pktito keeps float. With the matrices alone
(weight_threshold 16384: the 1x1 convolutions, 65536 weights and more),
v2 over LibriSpeech's 200 against float32: WER 1.51%, 2 changed, dlogp
-0.0239, enc err 3.1%, where every convolution int8 was 1.47%, 4, -0.0259,
3.3%.

## Why int8 costs more here than in pktito

pktito's int8 weights, a scale a row, cost dlogp -0.012 in float32
arithmetic (README); here, with the exp-form module's fp16 at -0.0024 alone,
-0.024. One matrix on the M4 Pro's Neural Engine, against float64 with the
very same int8 weights (layer 10's ff1.up and ff1.down on LibriSpeech's
8455-210777-0028): 0.19% and 0.32% from float64 with int8 weights, 0.026%
and 0.034% with fp16. The M4's Neural Engine computes int8 weights with its
activations rounded too (its int8 path, which is also why int8 is 12%
faster there).

## v3 over FLEURS, both (against float32, 4.43%)

|                          | WER   | changed | dlogp   | enc err |
| ------------------------ | ----- | ------- | ------- | ------- |
| fp16, two parts          | 4.43% | 10      | -0.0006 | 0.74%   |
| int8 matrices, two parts | 4.48% | 27      | -0.017  | 4.3%    |

(pktito's int8: 45 changed, -0.024, 5.2%.)

## On the M1 iPad (AliceBench ch1, two rounds each after a first launch)

|         | compile, once | a run                 | cores |
| ------- | ------------- | --------------------- | ----- |
| v2 fp16 | 20.4 s        | 2.75-2.77 s, 231-233x | 0.6   |
| v2 int8 | 15.7 s        | 2.52-2.53 s, 253-254x | 0.6   |
| v3 fp16 | 20.6 s        | 2.67-2.72 s, 236-240x | 0.8   |
| v3 int8 | 16.5 s        | 2.55 s, 251x          | 0.9   |

Against pktito's own transcript of the chapter (CPU, int8; 2,199 words):
v2 fp16 a comma, v2 int8 "to night," for "tonight," (the Mac's int8 the
same, word for word), v3 fp16 "Librivox" and "Chapter" for "End of
chapter" at the end (where pktito's retry hears the closing credits again),
v3 int8 none.

# 2026-10-31 - against Apple's transcriber

## Word error rates (M4 Pro)

The sets of the comparison with Apple's transcriber (pktito-app's
ios/compare, its pa runs): tools/corpus.py's 200 LibriSpeech recordings
(26.5 min); 40 of test-clean's chapters, each one recording (2.6 h); 60 of
FLEURS's a language. Every engine's transcripts scored alike: English with
Whisper's normalizer (the Open ASR Leaderboard's), FLEURS with pktito's own
(tools/pktito_tools/wer.py); the errors in brackets. Apple's is macOS 27's
SpeechTranscriber (none for Russian or Polish). Here, `pktito-c --ane` with
encoder.py's two parts; the fp16 rows' errors are the one-part program's on
every set.

|                           | 200 recordings | 40 chapters | FLEURS de/es/fr/it | FLEURS ru/pl |
| ------------------------- | -------------- | ----------- | ------------------ | ------------ |
| v2 int8                   | 1.34% (57)     | 1.73% (415) |                    |              |
| v2 fp16                   | 1.41% (60)     | 1.71% (412) |                    |              |
| pktito v2 (CPU)           | 1.34% (57)     | 1.72% (413) |                    |              |
| v3 int8                   | 1.72% (73)     | 2.08% (501) | 4.00% (218)        | 5.45% (120)  |
| v3 fp16                   | 1.65% (70)     | 2.08% (500) | 3.97% (216)        | 5.45% (120)  |
| pktito v3 (CPU)           | 1.74% (74)     | 2.08% (500) | 4.02% (219)        | 5.36% (118)  |
| Apple's SpeechTranscriber | 1.72% (73)     | 1.89% (455) | 5.05% (275)        |              |

Each against int8, paired over the recordings: the other's errors less
int8's, and the 95% interval of 5000 resamples.

|                     | 200 recordings | 40 chapters     | FLEURS de/es/fr/it | FLEURS ru/pl |
| ------------------- | -------------- | --------------- | ------------------ | ------------ |
| pktito v2 (CPU)     | 0 [0, 0]       | -2 [-8, +4]     |                    |              |
| v2 fp16             | +3 [0, +8]     | -3 [-10, +3]    |                    |              |
| Apple's, against v2 | +16 [-3, +36]  | +40 [+1, +78]   |                    |              |
| pktito v3 (CPU)     | +1 [-3, +6]    | -1 [-6, +4]     | +1 [-4, +7]        | -2 [-6, +2]  |
| v3 fp16             | -3 [-7, 0]     | -1 [-11, +9]    | -2 [-10, +6]       | 0 [-5, +6]   |
| Apple's, against v3 | 0 [-20, +21]   | -46 [-107, +12] | +57 [+19, +97]     |              |

int8 is pktito's own engine on every set, within 2 errors. Apple's makes
10% more than v2 on the chapters and 26% more than v3 on FLEURS's four, and
9% fewer than v3 on the English chapters (not quite outside the noise),
English being v2's.

Speed, the same runs (wall time over the set, the Mac in use; the 40
chapters): v2 int8 335x, fp16 302x, pktito's CPU 118x, Apple's 71x; v3
int8 336x, fp16 302x.

## int8 matrices encoder.py's default (kept)

Half the size (610 MB to 1.2 GB, and the system's compiled copy with it),
faster on both machines (the M1 iPad's AliceBench ch1 253x to 232x for v2,
251x to 238x for v3; the M4 Pro's 40 chapters 335x to 302x), and the WER of
pktito's own engine on every set above. fp16 stays as --fp16: ten times
nearer float32 (dlogp -0.0024 to -0.024, v2 over the 200), though no set
above tells the two apart by more than 3 errors. The default's build is the
--w8 build measured above, its model.mil and weights byte for byte.

# 2026-10-31 - lengths for short clips

## One function of several lengths (Core ML's enumerated shapes)

A function takes one length; a shorter input is padded to it. Its own
function for each length keeps a copy of every weight in the system's
cache (1.2 GB a length in fp16, above). Core ML's enumerated shapes give
one function several: its time dimension symbolic, the inputs' shapes
listed, and Core ML compiles it for each when it loads it. encoder.py
--frames 512,1024,1536,2048,2560,3072, int8, two parts, M4 Pro:

|                         | one length (3072) | six lengths   |
| ----------------------- | ----------------- | ------------- |
| first load, compiled    | 4.9 + 4.8 s       | 19.8 + 20.0 s |
| the system's cache      | 650 MB            | 1.86 GB       |
| process footprint after | 105 MB            | 135 MB        |
| 3072 frames, both parts | 66.0 ms           | 67.1 ms       |

The six's 512 to 2560: 9.0, 18.4, 32.4, 42.1, 55.5 ms. So a length past
the first costs 6 s of compiling and 240 MB of cache here, not a copy of
the weights; and Core ML did evict compiled models when the disk was low
(10 GB freed during a load), compiling them again at the next load.

Running a length after another costs about 40 ms (two parts, the median
of ten): 3072 frames after 512 take 108 ms where after 3072 they take 67,
512 after 3072 45 where after 512 9, 2560 after 3072 100 where 55.

## Where short clips are

pktito's chunks, traced (core/ane.m), over the comparison's sets: the 40
chapters 350, a median of 28.0 s, 278 longer than 25.6 s; LibriSpeech's
200 one each, a median of 6.2 s; FLEURS's 360 recordings 361, a median of
10.9 s. The encoder's time with the six lengths against one, from the times
above with 40 ms a change of length:

|                                                                                     | the chapters | LibriSpeech's 200 | FLEURS |
| ----------------------------------------------------------------------------------- | ------------ | ----------------- | ------ |
| the shortest length that holds the chunk                                            | +14%         | -31%              | -18%   |
| that, but stay on the last length unless the shortest is 1536 frames shorter (kept) | +2%          | -43%              | -30%   |

## pktito-c --ane over the sets (M4 Pro, compare.py's wall time)

| encoder                                              | LibriSpeech's 200 | the 40 chapters |
| ---------------------------------------------------- | ----------------- | --------------- |
| one length                                           | 108.7x            | 335.2x          |
| six, the shortest length                             | 152.0x            | 291.5x          |
| six, the last length kept unless 1536 frames shorter | 177.1x            | 324.8x          |
| six, that, with the tables as inputs (below)         | 165.6x            | 318.8x          |

Transcripts against one length's: the same, all 200 and all 40; with the
tables as inputs, two of the 200 differ, by a capital and by a comma (57
errors either way, Whisper-normalized; the chapters' 415).

## The position tables as inputs (kept, for several lengths)

The first program cut the position tables, made for 3072 frames, to the
input's length inside the program (mb.shape, then slice_by_size). The M4
Pro compiles that; the M1 iPad's Neural Engine compiler (iPadOS 26.5) does
not: "ANECCompile() FAILED", for the whole encoder and for its first two
layers alone, after which Core ML ran them elsewhere. With the three tables
inputs instead, made for the length by the caller (core/ane.m, from the
model's pos.freq, as encoder.py's turn_tables), the two layers compiled
there, and the whole. They cost a little on the M4 Pro: two layers 2.32 ms
for 512 frames where the fixed program takes 2.01, 8.58 for 3072 where 7.87.

## On the M1 iPad (iPadOS 26.5; -aneClips 5, the whole transcription)

The six lengths compiled in 76.8 s; one length in 16.4 s (compiled again
after the app was installed anew: the system's copy is a build's). Eight
of LibriSpeech's 200, each five times running, ms, the first and then the
median of the other four:

| clip    | one length | six lengths                |
| ------- | ---------- | -------------------------- |
| 3.00 s  | 102, 93    | 215 (the first of all), 46 |
| 5.08 s  | 97, 96     | 48, 47                     |
| 7.61 s  | 99, 98     | 134, 40                    |
| 10.09 s | 103, 102   | 41, 41                     |
| 13.01 s | 105, 105   | 166, 64                    |
| 16.98 s | 107, 107   | 172, 78                    |
| 21.73 s | 115, 114   | 201, 110                   |
| 27.18 s | 116, 116   | 236, 127                   |

A change of length costs the M1 about 100 ms, and its 3072 frames are 10%
slower in the six-length program. A short clip takes half the time of a
padded one, 40-50 ms where 95, both far under what a user would notice;
the long recording, the app's case, would be slower, and the first load
after each install 60 s longer.

## One length stays the default (kept)

Several lengths stay an option (--frames, the tables as inputs, the loader
keeping its last length unless a shorter one saves 1536 frames), for a use
where short clips come one after another, live captions say: there one
short length alone, compiled as its own model, would do it without the
changes of length.

## Two lengths, 15 and 30 s (measured, not kept)

Fewer lengths, fewer changes of length. encoder.py --frames 1536,3072 on
the M4 Pro: 3072 frames 68.3 ms, 1536 33.2; a change of length 22 ms
(alternating, 3072 took 90.5 and 1536 54.7), where six lengths' cost 40;
loaded and run in 21 s, with 612 MB less free space (one length's cache
was 650 MB, six lengths' 1.83 GB; free space is a noisy measure).

In a long recording only the last chunk is short. The 40 chapters' 350
chunks, traced a recording at a time: 16 of the 40 last chunks are under
15 s and none of the 310 others (273 of those over 25.6 s); the last
chunks are 48% padding, the others 10%, 14% in all. Simulated with each
machine's times and changes of length (the shortest length that holds
each chunk), long recordings one after another take the encoder longer
with every set: 15/30 +5% on the M4 Pro and +19% on the M1 iPad, 10/20/30
+6% and +23%, six +16% and +45%; LibriSpeech's 200 in a row 33%, 41% and
30% less on the M4 Pro, 18% and 19% less and 19% more on the M1. On the M1,
AliceBench ch1 takes 2.52 s in 23 chunks, 0.11 s a chunk; its last chunk's
padding is under 2% of that, less than the change of length down and back.

# 2026-10-31 - the slim model file

## A .pkf without its encoder (kept)

The encoder's tensors, the subsampling's and the 24 layers', are 612.3 MB
of v2's 624.5 MB file and of v3's 647.6. With the encoder a Core ML model,
`pktito-tools slim` writes the rest, byte for byte the full file's, and the
param encoder_coreml 1: v2 12.1 MB (12,115,524 bytes), v3 35.2 MB (v3's
8192 pieces make its decoder 27.6 MB and its joint 6.4). A build with ANE=1
binds it without an encoder; with no Core ML encoder given it runs nothing
(pk_asr_run -1; pktito-c says to give one with --ane), and a build without
refuses the file at load ("no tensor sub.conv0.w"). tests/pkfcheck.c holds
both: the slim file loaded or refused, the param otherwise set, a slim file
with an encoder's tensor or without another of the model's.

The encoder is taken only from the model's own checkpoint: encoder.py writes
the .nemo's SHA-256 into the Core ML model's metadata, as the converter does
into the .pkf's, and core/ane.m refuses a pair that differs (the v3 file
with the v2 encoder: refused, both named). The program and weights encoder.py
writes are unchanged, byte for byte.

pktito-c --ane over the comparison's sets, the slim files against the full
ones with the same int8 encoders: 0 of the 600 transcripts differ (v2 over
LibriSpeech's 200 and the 40 chapters, v3 over FLEURS's 360), at the same
speed (the 40 chapters 335.8x where 334.4x).

# 2026-11-02 - calls in flight

## Several chunks' encoder calls at once (hypothesis)

pk_run_chunks_with calls the encoder one chunk at a time; only the search
of the chunk before runs beside it. On the M4 Pro a 30.72 s chunk's call is
about 71 ms (above, "Where a run's time goes"), which holds a serial run
near 430x real time whatever else is done; this fork measures 324 to 363x.
The M5 Ultra has two Neural Engines (ioreg: H11ANE and H11ANE1 behind
H1xANELoadBalancer), and one call at a time leaves the second idle:
AliceBench v2 through 788ed7d as separate processes, chapters split by
length, 1 process 380x, 2 756x, 3 842x, 4 868x (libpktito, 2026-11-02).
Separate processes' calls overlap, so a call isn't the limit; the calls in
flight are.

The change: N calls in flight in one process (asr.c ane_chunks; core/ane.m
pk_ane_encode_in). Each chunk's features made on the calling thread and
copied to a slot's buffer, its call sent off (dispatch_async_f), and the
joint and the search taking the chunks in order as before, the search
beside the next chunk's joint. A slot has its own fp16 inputs; the models
and the position tables are shared, read only. The retry's and the
warming's calls take a slot of their own, so a retry from inside a chunk's
callback never meets the pipeline's. The calls' order doesn't touch the
tokens: each chunk's encoder frames are the same bits whenever it runs, and
the searches go in order, so the transcript should be 788ed7d's byte for
byte at every N.

Expected: N=2 near 2x on the M5 Ultra (the second Neural Engine), and on
a chip of one Neural Engine a little from overlapping a call's CPU side
(the fp16 copies in and out, Core ML's dispatch, about 3 ms of 71) and
the two parts of the next chunk's call with this one's. N=3 or 4 pays in
one process only if the features and the joint stay off the critical
path, as libpktito's separate processes kept them. PKTITO_ANE_INFLIGHT
(1 to 4, 1 the old path) is for this measurement; N is fixed from it.

## Two calls in flight (kept)

AliceBench v2 (12 chapters, 10,731 s), pktito-c with every chapter in one
process, the encoder v2-f3072-w8, wall time from launch to exit, best of
three on the M5 Ultra and of two on the M4 Pro (pktito-cleanup's L9 paused
for them), x real time:

| calls in flight    | M5 Ultra (2 Neural Engines) | M4 Pro (1)    |
| ------------------ | --------------------------- | ------------- |
| 1 (788ed7d's path) | 28.21 s, 380x               | 31.44 s, 341x |
| 2                  | 16.35 s, 656x               | 29.93 s, 359x |
| 3                  | 16.72 s, 642x               | 29.95 s, 359x |
| 4                  | 17.00 s, 631x               |               |

The transcript is the same at every N on each, and N=1's is 788ed7d's,
byte for byte. On the Ultra, a build timing each step at N=2 (not kept),
ms a chunk over its 391: the wait for the call 30.6, the joint 0.5, the
features 0.8, what was left of the search after the call 0.2, the
callback (transcribe.c's seam and retry) 1.7; the search itself 16.2, all
of it beside the calls. Both Neural Engines are full, a chunk every 31 ms
of the two (a call about 60 ms), so a third call only waits, and costs a
little. The rest of the run is outside the chunks: pktito-c -v's 16.04 s
with its VAD, of which Silero over the 12 chapters alone is 3.0 s (pktito-c
segments), the Neural Engines idle meanwhile; separate processes, which
overlap one's VAD with another's calls, made 756x of two (above). On the
M4 Pro's one Neural Engine two calls gain 5%: one call's CPU side (the
fp16 copies, Core ML's dispatch) behind the other's.

## Where a run's time goes now, by machine (measured, nothing changed)

A build that times each step of pk_asr_transcribe (not kept): wall time,
and the process's CPU time in the same window, so CPU / wall is the cores
busy in it. v2, pktito-c with a set's recordings in one process, two calls
in flight; AliceBench (12 chapters, 10,731 s) and the 26 English talks of
pktito-cleanup's test set (5,362 s). Share of the time in
pk_asr_transcribe, and the cores busy:

|                                                  | M4 Pro ANE    | M5 Ultra ANE  | M4 Pro CPU    | M5 Ultra CPU  | Pi 5 CPU           |
| ------------------------------------------------ | ------------- | ------------- | ------------- | ------------- | ------------------ |
| x real time (chapters / talks)                   | 362 / 318     | 667 / 538     | 121 / 118     | 203 / 187     | 14 (ch1 + 6 talks) |
| cores busy on average                            | 0.6 of 12     | 1.3 of 36     | 9.5 of 12     | 20.8 of 36    | 3.9 of 4           |
| Silero's VAD, before any chunk, 1 core           | 10.4% / 9.1%  | 15.9% / 13.0% | 3.4% / 3.3%   | 4.8% / 4.6%   | 1.3%               |
| waiting for the calls (ANE) or the encoder (CPU) | 81.8% / 81.2% | 75.2% / 75.4% | 95.4% / 95.6% | 93.1% / 92.3% | 98.2%              |
| the retry (9 and 8 stretches)                    | 3.3% / 5.1%   | 3.6% / 4.9%   | 0.9% / 0.5%   | 1.6% / 2.1%   | 0.1%               |
| the last chunk's search alone                    | 0.3% / 1.3%   | 0.4% / 1.8%   | 0.1% / 0.4%   | 0.1% / 0.6%   | 0.4%               |

The search is beside the encoder everywhere (its own time 8.4 s of the
M4 Pro's chapters, the wait for it 0.2 s). On the Neural Engine the CPU is
idle but for the VAD; on the Ultra's CPU one transcriber keeps 22 of 36
cores busy in the encoder (libpktito: one at 36 threads 192x, 36 at one
thread 524x, so main's CPU path is getting chunks in flight). The VAD is
the stall common to every machine, largest where the encoder is fastest.

## A call sent at once after each take, for the retry (rejected)

Hypothesis: with the next chunk's call sent only at the top of the loop,
one call was in flight through a chunk's callback, where the retry runs,
and its own call left the second Neural Engine idle (AliceBench v2 on the
M5 Ultra, two in flight: 9 retries cost 0.77 s of pk_asr_transcribe's
time, 86 ms each). Sending chunk T + 2's call as soon as chunk T is taken
keeps two out through the callback. M5 Ultra, AliceBench v2, wall time,
best of 3: 16.32 s against 16.35 s, inside the spread; the transcript the
same. With --no-retry it ran 15.68 s, so the 9 retries still cost about
0.64 s, about 70 ms each: a retry's stretch, padded to the one 3072-frame
length, is a whole call's work, and costs it however it is overlapped. Not
kept. A shorter length for the retry's stretches would cost the change of
length each way (about 40 ms, "Seven lengths or one" above) and another
compiled function.

## Silero over the transcriber's threads, from main (kept)

main 7012b87 merged: lanes (d148a53) and Silero's encoder over the pool
(4341c15). The Neural Engine's path stays first in pk_run_chunks_with, and
a transcriber with the Neural Engine keeps one lane: a lane's own
transcriber has the CPU's encoder, so a chunk's bits would be the lane's it
fell to. What the path gains is pk_asr_transcribe's VAD, which runs before
the first chunk while the Neural Engine waits, now on all of a->wide's
threads. Wall time from launch to exit, rounds interleaved, best of three
(of two for the single file), 416757a against this merge's code (built as
2026-11-02's scratch merge, the same code), x real time; the M4 Pro in use
(load 2 to 3):

|                                     | M5 Ultra, 36 threads                        | M4 Pro, 12 threads                         |
| ----------------------------------- | ------------------------------------------- | ------------------------------------------ |
| AliceBench v2, 12 files, 10,731.4 s | 16.44 s, 652.7x -> 14.72 s, 729.0x (+11.7%) | 29.93 s, 358.6x -> 28.29 s, 379.3x (+5.8%) |
| the talks, 26 files, 5,361.8 s      | 10.04 s, 534.0x -> 9.16 s, 585.3x (+9.6%)   | 17.08 s, 313.9x -> 16.25 s, 330.0x (+5.1%) |
| AliceBench as one file, 10,731.4 s  | 15.76 s, 680.8x -> 14.07 s, 762.7x (+12.0%) |                                            |

The transcripts are 416757a's byte for byte in every round on both, and
the CPU path's too (the Ultra's AliceBench 55.23 s, 194x, -> 24.89 s, 431x,
with lanes). make check-c ANE=1 passes on the Studio. Left before the first
chunk on the Ultra: about 1.0 to 1.3 s of its 14.7 s (Silero over the 12
chapters on one thread was 3.0 s, above, and the run saves 1.72 s), the
Neural Engines idle meanwhile. Overlapping the VAD with the first chunks'
calls would take most of that (not done).

# 2026-11-02 - the A18, and the position term folded

## The devices and the tools

|            |                                                                                                                                                                                                                                                                                                                                                                    |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| devices    | iPhone 16e (A18, a Neural Engine of 16 cores), iOS 26.7.2; Mac Studio, M5 Ultra (two Neural Engines), macOS 27.2, otherwise idle; Mac mini, M4 (a Neural Engine of 16 cores), macOS 26.4.1, idle; the M4 Pro above, macOS 27.0.1, in use (load 2 to 3)                                                                                                             |
| builds     | on the M5 Ultra, coremltools 9.0, Python 3.12: a copy of encoder.py with switches for the variants below (scratch), whose turn in two parts is encoder.py's v2-f3072-w8 byte for byte (model.mil and the weights, md5)                                                                                                                                             |
| per call   | a scratch harness (Swift on Core ML: an app on the phone, a command on the Macs) running a program's functions in order on random features, each part's output the next one's input: bursts of 5 or 7 calls after one untimed, the programs' order turned each round, 1 to 6 s between bursts; the median of the bursts' medians. Thermal state nominal throughout |
| profile    | MLComputePlan on the phone: each op's device and estimated cost                                                                                                                                                                                                                                                                                                    |
| accuracy   | anecheck.py on the M5 Ultra's Neural Engine, against float32: v2 over LibriSpeech's 200, v3 over FLEURS's six                                                                                                                                                                                                                                                      |
| whole runs | M5 Ultra: pktito-c --ane (1e2bdcb), AliceBench v2's 12 chapters in one process, wall time from launch to exit, three rounds alternating. The 16e: the iOS demo's Debug build (pktito-app 848bfc6, libpktito ane 1e2bdcb) with -bench, a chapter a launch, timed from the file's opening                                                                            |

## Where a layer's time goes (A18, M5 Ultra)

On the phone every op of both functions is the Neural Engine's
(MLComputePlan: f3072p1's 2096 operations, none on the CPU or the GPU;
f3072p0 the same). A call is 76.41 ms on the A18 (39.51 + 36.89, the two
parts) and 58.89 on the M5 Ultra (30.30 + 28.58). The subsampling and four
layers as one function, a part taken out at a time (wrong numbers, right
shapes), ms a call:

|                              | A18   | M5 Ultra |
| ---------------------------- | ----- | -------- |
| the subsampling and 4 layers | 15.13 | 11.63    |
| the subsampling alone        | 3.88  | 2.48     |
| no position term             | 11.19 | 8.82     |
| no attention                 | 8.72  | 6.77     |
| no feed-forwards             | 10.66 | 8.27     |
| no convolution module        | 14.10 | 10.81    |

A layer is 2.81 ms on the A18 and 2.29 on the M5 Ultra: the position term
0.99 of it (35%) and 0.70 (31%), the rest of the attention 0.62 and 0.51,
the two feed-forwards 1.12 and 0.84, the convolution module 0.26 and 0.21
(each taken out alone, so the parts overlap). The term costs more than the
rest of the attention and nearly what the feed-forwards do with their 16.8M
weights a layer, for little arithmetic: its turn makes five [8][1024][384]
tensors a layer (u0, u1, the two products and their sum), 6.3 MB each in
fp16, and its cost goes with them (folded, below, to 42% of itself on the
A18 with 40% of the rows). The plan's estimates point the same way: adds
18% of f3072p1 and reshapes 9.6%, against 28% for every convolution.

## The turn's layout (rejected)

Two other forms of the turn: tiled, in the convolutions' own layout
[1][H*D][1][T] with the tables tiled over the heads, one reshape before the
score product; and keyconv, tiled and then the scores by a grouped 1x1
convolution whose weights are pe (keys in rows, queries in columns), the
content term, the softmax and the product with V turned to match, no
[H][D][T] reshape at all; ms a call:

|          | A18, 4 layers | M5 Ultra, 4 layers | M5 Ultra, 24 |
| -------- | ------------- | ------------------ | ------------ |
| the turn | 15.13         | 11.63              | 59.09        |
| tiled    | 15.57, +2.9%  | 11.14, -4.2%       | 55.93, -5.3% |
| keyconv  | 15.96, +5.5%  | 11.16, -4.0%       | 56.02, -5.2% |

Faster on the M5 Ultra, slower on the A18; one program serves both, and the
fold below gains more on each.

## One part, and the subsampling's masks (no change)

The whole encoder as one function: 76.30 ms on the A18 where two parts take
76.37 (-0.1%), 58.80 on the M5 Ultra where 59.09 (-0.5%). Two parts stay:
the M1 compiles only them. Without the subsampling's masks, 3.71 ms against
3.88 on the A18 (its bursts 3.5 to 4.1) and 2.55 against 2.48 on the M5
Ultra: within the spread.

## The position term's low frequencies folded (kept, encoder.py's default)

Most of the turn's pairs hardly turn over the window. With 384 frames the
pairs of w (T - 1) < 16, 177 to 511 (335 of 512), make fewer than 2.6
turns, and their sin and cos columns over the window's frames lie, to 1e-4
of their largest singular value, in the span of 11 functions. Each low
pair's sin((i - j)w) and cos((i - j)w), with s and c projected on that
basis phi, makes the term's low part sum_kl phi_k(i) phi_l(j)
G_kl.(q_i + v): the turned form again, phi_k(i) where the turn has its cos
and sin and phi_l(j) where it has pe(j) (encoder.py's docstring,
fold_tables). The same ops, fewer rows: the low pairs' 670 rows a head
become (L - 1) \* ceil(L / 2) = 60, 414 where the turn has 1024.

The basis was first the columns' own leading singular vectors (L 11, 420
rows); then the constant, and the leading vectors of the columns less their
means (L 11, 414 rows), so that the constant's rows, the same for every
key, drop out (the softmax doesn't see them). Their error in emulated fp16
(scratch: the term in float64 through the program's roundings), on
LibriSpeech's 1089-134686 (its first 30.72 s), layers 0, 4, 8, 12, 16, 20
and 23, heads 0, 2, 5 and 7, the largest over them; the score error is the
term's error less its mean over keys, over sqrt(dk), the probability error
the attention's:

|                                  | rows a head | score err | prob err |
| -------------------------------- | ----------- | --------- | -------- |
| the turn                         | 1024        | 4.2e-3    | 3.6e-4   |
| cut 8 (pairs 216.., L 8)         | 460         | 1.25e-2   | 9.2e-4   |
| cut 16 (pairs 177.., L 11)       | 414         | 1.78e-2   | 7.4e-4   |
| cut 16, the columns' own vectors | 420         | 1.56e-2   | 7.3e-4   |
| cut 24 (pairs 154.., L 14)       | 399         | 1.34e-2   | 7.8e-4   |

(eps 1e-5 adds a function or two, 420 rows at cut 16, for no less error.)

The subsampling and four layers: the A18 15.54 ms to 13.11, 2.43 of the
4.19 the term costs (58%); the M5 Ultra 11.58 to 9.73, 1.85 of 2.77 (67%).
A head's rows padded to a multiple of 32 (416) were slower, 13.19 and
10.03. The whole encoder, ms a call:

|                  | the turn              | folded, cut 16                | cut 8         |
| ---------------- | --------------------- | ----------------------------- | ------------- |
| A18 (iPhone 16e) | 76.41 (39.51 + 36.89) | 62.68 (32.66 + 30.02), -18.0% | 63.46, -16.9% |
| M4 (Mac mini)    | 68.02 (35.09 + 32.92) | 55.69 (28.93 + 26.75), -18.1% |               |
| M4 Pro           | 65.46 (33.75 + 31.71) | 53.58 (27.83 + 25.73), -18.1% |               |
| M5 Ultra         | 58.89 (30.30 + 28.58) | 47.66 (24.69 + 22.96), -19.1% | 48.24, -18.1% |

Against float32, on the M5 Ultra's Neural Engine:

| v2, LibriSpeech's 200 (float32 1.51%) | WER   | changed | dlogp    | enc err |
| ------------------------------------- | ----- | ------- | -------- | ------- |
| the turn                              | 1.51% | 2       | -0.02389 | 3.10%   |
| folded, cut 16                        | 1.51% | 2       | -0.02385 | 3.12%   |
| folded, cut 8                         | 1.51% | 3       | -0.02425 | 3.12%   |

| v3, FLEURS's six, 358 recordings (float32 4.43%) | WER   | changed | dlogp    | enc err |
| ------------------------------------------------ | ----- | ------- | -------- | ------- |
| the turn                                         | 4.48% | 27      | -0.01696 | 4.31%   |
| folded, cut 16                                   | 4.47% | 27      | -0.01826 | 4.32%   |

(The turn's v3 row is 2026-10-31's on the M4 Pro, 4.48%, 27, -0.017, 4.3%:
the same build, byte for byte.)

Whole runs. The M5 Ultra, AliceBench v2 (12 chapters, 10,731.4 s): the turn
14.87, 14.72, 14.86 s, folded 12.20, 12.23, 12.31 s; best of three 14.72 s
(729.0x) to 12.20 s (879.4x), -17.1%. The 16e's app: 41.82 s for the book
(256.6x, chapters 237.5 to 270.9x) to 36.10 s (297.3x, 277.0 to 314.8x),
-13.7%, Core ML's first compile 13.1 s and 12.5 s; the rest of the app's
time is the CPU's, the decoding included.

Folded, AliceBench's transcript is the turn's but for 6 of 27,118 words in
five places, the same five on the M5 Ultra and the A18: ch02 "There were"
for "There was" (the book's "were"), ch03 "tail" for "tale" (the book's
"tale"), and three of punctuation (ch03 "bank," and "feathers,", ch10
"griffon. Of" for "griffon of"). Against pktito's own (CPU, int8; 27,114
words as written, case and punctuation counted) the turn is 54 words apart,
folded 58; by the app's comparison, 16 words and 78 character edits from
the CPU's, and 16 and 81.

encoder.py makes it now (fold_tables, --cut 16 the default): its v2 build
is the climb's byte for byte (model.mil and the weights, md5), and --cut 0
gives the turn, encoder.py's build before, byte for byte. A program of
several lengths keeps the turn: the folded rows depend on the window, and
core/ane.m makes the turn's tables for each length.

## Anchor frequencies for the low pairs (rejected)

Before the basis, a fit: each row's low part over the offsets r = -(T - 1)
.. T - 1 (the sum over the low pairs of W sin(r w) + W' cos(r w)) refit by
least squares with a few anchor frequencies over (0, cut / (T - 1)], the
program then the turn over fewer pairs. Emulated fp16, 1089-134686, three
heads (layer 12's head 5, layer 23's 0 and 5): at cut 16, 3 and 4 anchors
fit the kernel to 0.74 to 1.3 (score errors 11 to 18), 6 to 0.035 to 0.065
(0.53 to 0.72), 8 to 1.9e-4 to 3.6e-4 (5.6e-3 to 1.4e-2, prob err up to
9.7e-4, 370 rows a head); at cut 8, 6 anchors to 8e-6 to 6e-5 (4.2e-3 to
7.1e-3). Every fit takes weights of 12 to 60 where the low pairs' are 0.2
to 0.5: sinusoids of near frequencies over a short window are near
collinear, and the fit is large terms cancelling, which the program's int8
weights would round in steps of their own size (60/127). The fold's basis
is orthonormal, its G about 1. Not built.

## W8A8 (rejected)

coremltools 9.0's linear_quantize_activations (experimental): every
convolution's input int8 too (linear, symmetric, a scale a tensor),
calibrated on 16 windows of 30.72 s, two from each of 8 of tools/corpus.py's
40 LibriSpeech chapters; the weights int8 as now; one part, from the
features. The subsampling and four layers on the A18 (calibrated on
random features, for the time alone): 15.51 ms to 10.66 (-31%), where
folded they take 13.18. v2 over the 200 against float32
(anecheck's encoder time over the 200, one call at a time on the M5 Ultra,
not a timing):

|                                          | WER   | changed | dlogp   | enc err | encoder |
| ---------------------------------------- | ----- | ------- | ------- | ------- | ------- |
| the turn                                 | 1.51% | 2       | -0.0239 | 3.1%    | 12.2 s  |
| folded                                   | 1.51% | 2       | -0.0239 | 3.1%    | 9.8 s   |
| W8A8                                     | 1.49% | 38      | -0.266  | 33.3%   | 9.9 s   |
| folded, W8A8                             | 1.59% | 41      | -0.276  | 33.6%   | 8.2 s   |
| folded, W8A8 after the layer norms alone | 1.49% | 18      | -0.063  | 9.8%    |         |

Every input int8: dlogp eleven times the turn's and 38 and 41 of the 200
transcripts changed. Rejected. Its inputs are rounded a scale a tensor, and
what a layer norm makes, normalized, might take that better than what a
SiLU, the GLU or the softmax does: W8A8 on the inputs after a layer norm
alone (the feed-forwards' up, q, k, v and the module's first, over half the
matrices' weights), folded, is the last row, still two and a half times the
turn's dlogp and nine times its changed transcripts. The M4 (Mac mini), ms
a call, interleaved as above, folded each: 55.70 as it is, 52.57 with W8A8
after the layer norms (-5.6%), 40.07 with every input (-28.1%). Rejected
too: the speed, as most of the error, is in the other inputs (the
feed-forwards' down after the SiLU, the module's after the GLU, o after the
attention), and a scale finer than a tensor's for them is not what
linear_quantize_activations offers.

# 2026-11-03 - the fold's whole runs on the M4

|        |                                                                                                                                                                                                                                                                                                                                                                                       |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| device | the Mac mini above (M4, macOS 26.4.1), idle between runs but for its aerial wallpaper and WindowServer (4 to 11% of a core each, loginwindow 13% once)                                                                                                                                                                                                                                |
| run    | pktito-c -v --ane, 1e2bdcb built there (Apple clang 21.0.0, ANE=1), with the v2 model; the turn and the fold of 2026-11-02 (encoder.py's v2-f3072-w8 at --cut 0 and 16), the files the M5 Ultra ran (md5); AliceBench v1's 12 chapters (10,731.4 s; its WAVs' samples, md5) in one process; five rounds alternating, after an untimed run each (ch12, with Core ML's compile, 11.6 s) |
| timers | transcription: pktito-c -v's "transcribed in", pk_asr_transcribe's time summed over the files (its VAD in it), the span AliceBench's timer covers. Wall: from launch to exit, with the model's and the encoder's loads and the WAVs' reading                                                                                                                                          |

AliceBench v1 is the set; the 2026-11-02 section's "AliceBench v2" is
this set with the v2 model.

| best of five  | the turn        | folded          |        |
| ------------- | --------------- | --------------- | ------ |
| transcription | 30.32 s, 353.9x | 25.35 s, 423.3x | -16.4% |
| wall          | 30.89 s, 347.4x | 25.89 s, 414.5x | -16.2% |

Each program's rounds are within 0.1% of each other, and five more
without -v gave the same wall times within 0.3% (30.80 and 25.83 s). A
call is -18.1% on this chip (above); the rest of the run is the CPU's.
The transcripts are the M5 Ultra's, the turn's and the folded (md5, the
files' paths aside): the fold's six words of 27,118 again.
ALICEBENCH.md's row for this mini, 0.9.5's pktito-cli bench with the
turn's program (its model.mil and weights, md5), is 356.6x, and 357.6x
again on 2026-11-03: 0.8% and 1.0% over the turn's here, with the same
timer and encoder, another command and build.

# 2026-11-03 - the fold on the M1 iPad

|          |                                                                                                                                                                                                                                                                                             |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| device   | the M1 iPad of 2026-10-31 (iPad Pro 11, 3rd generation, a Neural Engine of 16 cores), iPadOS 26.5                                                                                                                                                                                           |
| app      | pktito-app's ios e952fd3, 0.9.5's Debug build: -bench english over AliceBench v1's 12 chapters (its FLACs), one launch a chapter                                                                                                                                                            |
| programs | the turn, 0.9.5's v2 encoder as it shipped (model.mil 78e737d8, weights a33f5d21), and the fold of 2026-11-02 (out/fold16, 97b8727d and e7e0370c, what encoder.py builds at its defaults), each put in the app's models folder and read back (md5) before its round; six rounds alternating |
| timer    | the app's transcription, samples to segments, summed over the chapters                                                                                                                                                                                                                      |

| best of three           | the turn                                   | folded          |        |
| ----------------------- | ------------------------------------------ | --------------- | ------ |
| AliceBench v1           | 44.40 s, 241.7x                            | 39.57 s, 271.2x | -10.9% |
| Core ML's first compile | 15.5 and 15.7 s (16.7 s after a reinstall) | 14.5 to 14.8 s  |        |

Each program's rounds are within 0.15% of each other (the turn 44.46,
44.42 and 44.40 s, folded 39.62, 39.57 and 39.64), and ALICEBENCH.md's
0.9.5 row for this iPad is 240.9x. The folded program compiles on the M1,
which once refused the whole fp16 encoder as one function (2026-10-31),
in about a second less than the turn. A call alone wasn't timed here; on
the A18 the app's whole run took 13.7% less where a call took 18.0% less.

The transcript changes in more places here than on the newer chips: 16
places, 22 words of 27,143, where the A18, the M4 and the M5 Ultra change
the same five. Four of the five are among them (ch02 "There were", ch03
"bank," and "feathers,", ch10 "griffon. Of"); ch03's "tail" isn't. The
rest are the M1's own:

- case: ch07's Hatter and Dormouse in lower case four times, ch08 "and"
  for "And", ch09 "mock turtle" for "Mock Turtle", ch12 "HERE" for "Here";
- punctuation: ch01 "asking" for "asking.", ch09 "Well then," for "Well,
  then,", ch11 "strings into" for "strings. Into";
- two passages the book has and the turn leaves out, as the CPU's
  references do: ch03 "It continued," (the book's "...my dear?' it
  continued, turning to Alice") and ch06 "said the cat." ("pig, or fig?'
  said the Cat.").

Against the CPU's references, counted as a token edit distance (not
AliceBench's scorer), the turn is 57 words apart and folded 62, or 43 and
48 with case and punctuation dropped. The book's two passages count
against the fold there, because the references lack them.

The v3 turn row of 2026-11-02 (4.48%, 27, -0.01696) is the M5 Ultra's,
measured beside the folded row. Its note means that 2026-10-31's row on
the M4 Pro matches it to its digits. Both v3 rows are anecheck over
encoder.py's own builds, and ane 9c68158's encoder.py builds the folded
one again byte for byte (model.mil and the weights, md5), as it does v2's.
