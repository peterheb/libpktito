# tools/qnn: the climb

Dated records of what was measured, on which machine, for the encoder on
Qualcomm's Hexagon NPU (npu/qnn.c's graph, built with QNN=1): append-only.
A later section may supersede an earlier one, but the earlier one stays,
rejected hypotheses included.

The entries dated before 2026-11-08 were measured on pktito's ane fork. Its
history was squashed into the commit that brought this file to main, so the
commits they name are the fork's and are not in this repository. The
fork kept the graph's builder in core/qnn.c, as they name it; it is
npu/qnn.c here.

# 2026-11-02 - the first graph

## The machine and the tools

|            |                                                                                                                                                                                                                                         |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| machine    | Galaxy S24 Ultra (SM-S928U1): Snapdragon 8 Gen 3 for Galaxy (SM8650), Hexagon V75; Android 16, One UI build S928U1UES6DZI2; plugged in, 29.6 to 43.8 °C over the runs                                                                   |
| QNN        | Qualcomm AI Runtime 2.51.0.260929 (QNN API 2.40.0): libQnnHtp.so, libQnnHtpV75Stub.so and libQnnHtpPrepare.so from adb's /data/local/tmp, libQnnHtpV75Skel.so (the unsigned PD's) by ADSP_LIBRARY_PATH; the graph compiled on the phone |
| toolchain  | NDK r30's clang 21.0.0, API 29                                                                                                                                                                                                          |
| model      | parakeet-tdt-0.6b-v2.pkf                                                                                                                                                                                                                |
| whole runs | `pktito-c --qnn -v` a chapter a process, the compiled graph from PKTITO_QNN_CACHE; AliceBench v1's WAVs                                                                                                                                 |

## Can it run there at all

`qnn-platform-validator --backend dsp --testBackend` from adb's shell:
the DSP backend supported, Hexagon Architecture V75, its unit test passed.
libcdsprpc.so, which the stub reaches the NPU through, is in the vendor's
/vendor/etc/public.libraries.txt: an app's code may load it. That an app,
rather than adb's shell, gets the unsigned PD is not yet tried.

## What the matrices cost

Before any of the encoder was written: a scratch program (not kept) built
24 layers' worth of FullyConnected ops, 25 M random weights a layer
(Parakeet's 24 M, less GLU, plus 1 M), on 384 frames, a 30.72 s chunk's;
best and mean of 10 after one untimed, the HTP's clocks voted up (DCVS
off, the bus and core at their highest corners):

| activations, weights              | best ms | mean ms | compiled in |
| --------------------------------- | ------- | ------- | ----------- |
| float32 given, fp16 math; fp16    | 41.2    | 46.2    | 11.6 s      |
| fp16; int8, a scale an output row | 45.8    | 50.5    | 32.4 s      |
| uint16 (static scales); int8      | 21.5    | 25.3    | 4.7 s       |

One layer in fp16: 1.85 ms, 5.4 T multiply-adds a second. The int8
weights with fp16 activations cost 11% more than fp16 weights, at half
their size; the fully quantized form is twice as fast but needs every
activation's range fixed beforehand, calibrated, and is not tried here.
Kept: int8 weights against fp16 activations, which are the .pkf's own
matrices and tools/ane/encoder.py's numerics on the Neural Engine.

## The graph's first numbers, and two faults

The check (PKTITO_QNN_CHECK=1, core/encoder.c) runs the CPU's encoder
beside the NPU's on each chunk of tests/data/test.wav (60 s, three chunks
of 383, 375 and 40 encoder frames) and compares each frame:

| graph                                                | cosine to the CPU's, mean | worst   | transcript                              |
| ---------------------------------------------------- | ------------------------- | ------- | --------------------------------------- |
| the frames past the end zeroed after the subsampling | -                         | -       | empty                                   |
| not zeroed; the matrices in the file's panel order   | 0.018 to 0.028            | -0.071  | none of it                              |
| the matrices unpacked a row at a time                | 0.99970 to 0.99988        | 0.99084 | the CPU's but for a comma and a capital |

- A frame of zeros has no variance: its layer norm, its epsilon 1e-5 /
  64 = 1.6e-7 for the stream carried at 1/8 (below fp16's smallest normal),
  is 0 / 0, and every query's attention took the NaN in. The padding is
  now left as the subsampling makes it, as tools/ane/encoder.py leaves it.
- A .pkf's int8 matrix is in panels of 16 rows, four inputs of each row
  together (core/pkf.h, PK_Q8), the shape SDOT multiplies; QNN's weights
  are a row at a time. Unpacked when the graph is made.

## Whole runs

AliceBench chapter 1 (640.5 s, 23 chunks), `pktito-c bench`, best of 5:

| encoder                 | s      | x real time | tokens |
| ----------------------- | ------ | ----------- | ------ |
| CPU, neon-dp, 8 threads | 21.695 | 29.5        | 3945   |
| NPU                     | 4.961  | 129.1       | 3903   |

Compiling the graph on the phone took 135 s in a run of its own and 267 s
in the one that also wrote the cache, 736 MB for the English model; a run
from the cache read it back and transcribed test.wav in 2 s.

All twelve chapters (2:58:51), a process each, `-v`'s times (with the
VAD), the NPU's first, then the CPU's (the phone 41.4 °C at its start and
43.8 at its end, so the CPU's ran warmer):

| encoder | s     | x real time |
| ------- | ----- | ----------- |
| CPU     | 529.0 | 20.3        |
| NPU     | 91.6  | 117.2       |

Against the CPU's transcripts through the same pktito-c (AliceBench's
references are pktito-cli's, which spaces a quote after a full stop where
pktito-c does not, so pktito-c's own CPU transcripts are 46 words from
them by this measure): 17 words of 27114 changed, added or removed, 0.063%,
case and punctuation aside; chapter 11 identical. The words: "were" and
"was", "tale" and "tail", "Dormouse" and "door mouse", "griffin" and
"griffon", "Put, put" and "Put Pot", and the LibriVox notices of chapters 6
and 12, where the NPU's "Librivox dot org" is nearer the recording than
the CPU's "librac". The rest is commas and capitals.

# 2026-11-02 - where a chunk's time goes

## The profile

PKTITO_QNN_PROFILE=1 (core/qnn.c) prints QNN's detailed profile, each
op's cycles, the ops named for their block and type. tests/data/test.wav's
three chunks, the same graph and so the same work, took 198.9 to 199.7 ms
on the accelerator, 1241 to 1251 M of the profile's cycles (6.3 a ns, its
own unit), which the 1336 ops QNN compiled the graph to sum to. The second
chunk's, by what the ops do:

| ops                                                           | M cycles | share | ms   |
| ------------------------------------------------------------- | -------- | ----- | ---- |
| the position term: u0 and u1, the turn's products and sum, bd | 601.1    | 48.0% | 95.7 |
| every matrix (FullyConnected)                                 | 224.7    | 18.0% | 35.8 |
| the feed-forwards' SiLU, the sigmoid folded into its multiply | 118.3    | 9.5%  | 18.8 |
| the convolution module's GLU, mask, depthwise and SiLU        | 87.1     | 7.0%  | 13.9 |
| the attention's adds: bias_u, bias_v, ac + bd, the keys' bias | 68.6     | 5.5%  | 10.9 |
| the layer norms, five a layer                                 | 48.4     | 3.9%  | 7.7  |
| the subsampling but its matrices                              | 36.5     | 2.9%  | 5.8  |
| softmax                                                       | 25.6     | 2.0%  | 4.1  |
| transposes and reshapes                                       | 21.2     | 1.7%  | 3.4  |
| ac and pv                                                     | 17.0     | 1.4%  | 2.7  |

A layer is 49.0 M cycles (7.8 ms), its position term 24.5 M of it: u0 and
u1 5.3 and 4.7 M, their products with the turn's tables 6.7 and 7.0 M,
qr's sum 0.4 M and bd 0.4 M. Those four take in or give out [8][384][1024]
tensors, 6.3 MB each in fp16; bd, a product of 1.2 G multiply-adds, costs
less than either turn's product of 0.4 G.

The HTP's own linting level (QNN_HTP_PROFILE_LEVEL_LINTING), which adds
each op's waits, crashed the NPU's process on this graph in both runs
tried ("SSR detected"). QNN recovered, and pktito transcribed those chunks
on the CPU, silently. The option is not kept.

## Hypothesis: NeMo's form of the position term

The turn (2026-11-02's first graph, from tools/ane/encoder.py) is there
because the Neural Engine has no cheap way to shift each row of a matrix
by a different amount. The HTP has slices and reshapes, so NeMo's own form
fits: P = W_pos pe(r) for the 2T - 1 relative positions r, a constant of
each layer made with the graph (in double, rounded once, 1/sqrt(dk) folded
in and its mean over r taken out, which the softmax doesn't see); bd_full =
(q + v) P^T, a head at a time, [8][384][767]; then rel_shift: a column of
zeros in front, viewed as [8][768][384], the first row dropped, viewed as
[8][384][767], the first 384 columns kept, which is bd. That is 0.3 G
multiply-adds a layer where the turn has 2.0 G, and no [8][384][1024]
tensor. If the shift's three moves of 2.4 M elements cost what the heads'
transposes do (0.49 M cycles for 0.4 M elements), the position term goes
from 24.5 M cycles a layer to about 10 M, the chunk from 199 ms to about
140; if they cost what the bias adds do (1 cycle an element), to about 8
M and 135 ms. The numbers should stay as near the CPU's (PKTITO_QNN_CHECK):
P is rounded to fp16 once, where the turn rounds u0, u1, their products and
qr, and bd is a sum of 128 products where it was 1024.

## The shift, two ways: rejected

| position term                    | accelerator ms a chunk | M cycles | compiled in | cache MB | mean cosine to the CPU's, a chunk |
| -------------------------------- | ---------------------- | -------- | ----------- | -------- | --------------------------------- |
| the turn (the first graph)       | 199.1                  | 1250     | 248 s       | 736      | 0.99970 to 0.99988                |
| rel_shift by reshapes and slices | 264.3                  | 1003     | 85 s        | 649      | 0.99968 to 0.99988                |
| rel_shift by GatherElements      | 232.8                  | 905      | 105 s       | 654      | 0.61 to 0.73                      |

(The compile is QNN's finalize alone, from the profile.) The reshapes' two
views cross the innermost axis, which the HTP does as a move of its own,
5.9 M cycles each for 2.4 M elements, 28% of the chunk's cycles. The time
rose though the cycles fell: the profile's cycles are summed over the
HTP's threads (6.3 a ns in the first graph, 3.8 here), and those moves, by
the time they took, ran on few of them. The transcript of test.wav was the
turn's. GatherElements, with one index tensor for the 24 layers, was
quicker than the reshapes, but its output was not what the indices asked
for: the cosine says the scores were wrong, and the transcript changed in
places. Without the four [8][384][1024] tensors a layer the graph compiles
in a third of the time, which a way of keeping the turn may also find.

## Hypothesis: the turn a head at a time

In the first graph the turn's sum over three [8][384][1024] tensors cost
0.4 M cycles and bd, reading one, 0.4 M, while the two multiplies, which
broadcast a [384][1024] table over the eight heads, cost 6.7 and 7.0 M, and
u0 and u1, batched products against [8][128][1024] constants, 5.3 and 4.7
M. So it is the broadcast and the batched constant, not the tensors' size,
that cost. A head at a time: its 128 columns of q + v sliced out of the
[384][1024] tensor (no transpose), one FullyConnected against [W0; W1],
[2048][128] (u0 and u1 together), one multiply by [cos | sin], [384][2048],
nothing broadcast, one FullyConnected against [pe | pe], [384][2048], which
is the sum and bd together; the eight heads' [384][384] then concatenated.
Each head's tensors are 1.5 MB, which the HTP's 8 MB of VTCM can hold. If
the products run as fast as the graph's other FullyConnected ops (the
matrices' 35.8 ms for 230 G multiply-adds a chunk: 0.16 ms a G) and the
multiplies as fast as the sum did, the position term goes from about 95 ms
a chunk to some 15 to 20 (3.2 G multiply-adds a layer, as the sum is now
twice as long: 12 ms; the multiplies about 3) and the chunk to about 120
ms; and the numbers stay the turn's, as the arithmetic is.

## The turn a head at a time: kept

| graph                                | accelerator ms a chunk | test.wav | chapter 1, bench, best of 5 | AliceBench, a process a chapter | compiled in | cache MB |
| ------------------------------------ | ---------------------- | -------- | --------------------------- | ------------------------------- | ----------- | -------- |
| the heads together (the first graph) | 199.1                  | 90.8x    | 4.961 s, 129.1x             | 91.6 s, 117.2x                  | 248 s       | 736      |
| a head at a time                     | 125.6                  | 135.6x   | 3.113 s, 205.8x             | 64.6 s, 166.2x                  | 86 s        | 718      |

(The compile is QNN's finalize, from the profile; a whole run of test.wav
that compiled the graph and wrote the cache took 94 s with the phone at
34.8 °C and 114 s at 38.9 °C, and a run from the cache 1.2 s. Two compiles
wrote the same bytes.) The position term is 5.3 M cycles a layer, from
24.5: QNN folded its slices, multiplies, reshapes and concatenation into
its 384 FullyConnected ops, which have every cycle of it. A layer is 29.9
M cycles, from 49.0, and a chunk 778 M, at 6.2 a ns again. The check's
cosines are the turn's (0.99970 to 0.99988 a chunk, the worst frame
0.99116), and test.wav's transcript is. Of AliceBench's 27118 words, 2
changed against the first graph ("I daresay" for "I dare say", chapter
2), seven chapters identical and the other four in capitals and commas;
against the CPU's transcripts, 19 words of 27114, 0.070% (the first
graph: 17, 0.063%). The phone went from 35.9 to 39.1 °C over the run.
Chapter 1's NPU encoder is 7.0 times the CPU's (21.695 s, 29.5x).

A chunk now, by the profile's cycles: the matrices 29%, the position
term's products 17%, the feed-forwards' SiLU 15%, the convolution module
11%, the attention's adds 8%, the layer norms 6%, the subsampling 5% and
softmax 3.5%.

## Hypothesis: two adds folded, and the convolution's SiLU on [T][D]

A layer of the graph above is 29.9 M cycles. Two of its adds are the
attention's biases, q + u and q + v on [384][1024], 0.45 and 0.44 M; the
Q product can take u as its own bias, and each head's turn product then
W (v - u) as its, made with the graph, which leaves neither add. The
convolution module's SiLU takes the depthwise convolution's output as it
comes, [1][384][1][1024], and costs 2.0 M, where the feed-forwards' takes
2.4 M for four times the elements on [384][4096]: a width of 1, which the
HTP's tiles pad. The SiLU after the view back to [384][1024], which the
view before the convolution says costs 0.24 M, should be about 0.6 M. If
so a layer is about 27.9 M cycles, and a chunk about 118 ms from 125.6.
Neither changes the arithmetic but for where v - u is rounded.

## The adds folded, the SiLU moved: rejected

| graph, from the turn a head at a time                         | accelerator ms a chunk | M cycles a layer |
| ------------------------------------------------------------- | ---------------------- | ---------------- |
| as kept                                                       | 125.6 to 126.7         | 29.9             |
| u into Q's product, v - u into the turn's; the SiLU on [T][D] | 138.7 to 140.2         | 32.3             |
| u into Q's product, v - u into the turn's                     | 125.1 to 127.9         | 29.1             |

The SiLU after the view back to [384][1024] cost 4.9 M cycles, a sigmoid
QNN no longer folded into the multiply, against 2.0 M on the
convolution's [1][384][1][1024]: the HTP's sigmoid costs what the tensor
it is given makes it, which the shapes alone don't say. The folded biases
took both adds away, 0.9 M, but q + u's transpose then stood alone, 0.39
M where it had been folded into the add, and the 0.8 M of cycles a layer
left did not show in the time.

## What a SiLU costs, alone and after a product

A scratch program (not kept) timed chains of SiLUs (x times sigmoid x,
as the graph writes it) on one tensor of 384 x 4096, the accelerator's
time from QNN's basic profile, best of 10:

| shape             | us a SiLU, the high-precision sigmoid | without |
| ----------------- | ------------------------------------- | ------- |
| [384][4096]       | 259.5                                 | 258.6   |
| [1][1][384][4096] | 259.2                                 | 259.6   |
| [1][384][1][4096] | 303.3                                 | 303.4   |
| [1][16][24][4096] | 244.6                                 | 244.7   |
| [1][8][48][4096]  | 253.5                                 | 253.6   |

and four FullyConnected 4096 to 4096 on int8 weights, each with or without
a SiLU after it: 1447 us a product alone, 1455 with its SiLU (1464 without
the high-precision sigmoid). The shape is worth 6% at most but for a width
of 1 (17% worse); the high-precision option changed neither the time nor
the output's bits; and a SiLU after a product costs next to nothing, as
the HTP's vector units do it while its matrix unit does the next product.
The profile's cycles count that SiLU all the same, so its 15% of a chunk's
cycles are not 15% of the time. A 1x1 Conv2d on the same int8 weights,
which QNN's documentation says can fold an activation into itself, does
not compile with fp16 activations ("no properties registered for
q::QNN_Conv2d_w_scale").

# 2026-11-02 - the same graph on a Galaxy S25 Ultra

|         |                                                                                                                  |
| ------- | ---------------------------------------------------------------------------------------------------------------- |
| machine | Galaxy S25 Ultra: Snapdragon 8 Elite for Galaxy (SM8750), Hexagon V79; plugged in, 30.5 to 36.2 °C over the runs |
| QNN     | as the S24's, with libQnnHtpV79Stub.so and libQnnHtpV79Skel.so                                                   |
| code    | d2b78b3's pktito-c, the binary of the S24's numbers in "The turn a head at a time: kept"                         |

|                                              | S24 Ultra, V75               | S25 Ultra, V79               |
| -------------------------------------------- | ---------------------------- | ---------------------------- |
| HVX threads the graph used, by the profile   | 4                            | 6                            |
| the compile, QNN's finalize                  | 85.8 s                       | 59.2 s                       |
| accelerator ms a chunk                       | 125.6 to 126.7               | 131.1 to 131.6               |
| M cycles in a chunk's ops                    | 778                          | 959                          |
| of them FullyConnected                       | 356                          | 450                          |
| ElementWiseMultiply                          | 197                          | 288                          |
| ElementWiseAdd                               | 66                           | 62                           |
| LayerNorm                                    | 48                           | 43                           |
| test.wav's chunks, mean cosine to the CPU's  | 0.999863, 0.999702, 0.999886 | 0.999864, 0.999699, 0.999877 |
| chapter 1, `pktito-c bench --qnn`, best of 5 | 3.113 s, 205.8x              | 3.113 s, 205.7x              |
| chapter 1, `pktito-c bench`, the CPU's       | 21.695 s, 29.5x              | 16.573 s, 38.6x              |
| AliceBench, a process a chapter              | 64.6 s, 166.2x               | 60.8 s, 176.6x               |
| words not the CPU's, of 27114                | 19, 0.070%                   | 17, 0.063%                   |

The two NPUs' transcripts differ in 2 words of 27119 (0.007%), four
chapters identical. V79 ran the graph 4% slower a chunk with two HVX
threads more, at 7.3 of the profile's cycles a ns to V75's 6.2: as the
cycles are summed over the HTP's threads, more of them in about the same
time is threads running that do not finish the work sooner. The products'
cycles rose 27% and the element-wise multiplies' 46%, the adds', the
norms' and the softmaxes' not at all; that the one waits on the matrix
unit and the other on memory is a reading of the counters, not anything
Qualcomm documents. The S25's whole runs are the quicker for its CPU,
which does the rest of a transcription.

# 2026-11-02 - the graph compiled on a PC

|         |                                                                                                                                                                                                     |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| machine | Ryzen 9 9950X (WSL2, Ubuntu 26.04): QAIRT 2.51.0.260929's x86_64-linux-clang libQnnHtp.so and libHtpPrepare.so, with LLVM's libc++ 22 and libunwind 18; the graph run on the Galaxy S24 Ultra above |
| code    | qnn's working tree on 96b16a3: PKTITO_QNN_TARGET, PKTITO_QNN_LOG's description of the chip (core/qnn.c)                                                                                             |

The S24's QNN describes its chip (PKTITO_QNN_LOG) as one device of one
core, SoC model 57 (SM8650), V75, 8 MB of VTCM, DLBC and no signed PD:
PKTITO_QNN_TARGET=57,75,8,1,0, which the PC's QNN is given as a platform
description and as the HTP device's SOC and ARCH options, the two that
Qualcomm's qnn-context-binary-generator sets. Given the description
alone, the PC's QNN compiled the graph for itself ("RouterX86 graph
prepare"), its activations fp32, which a FullyConnected on int8 weights
does not take there ("no properties registered for
q::QNN_FullyConnected_w_scale").

| the graph               | compiled on | in             | bytes     | accelerator ms a chunk, the S24 | AliceBench, the S24 | words not the S24's own graph's |
| ----------------------- | ----------- | -------------- | --------- | ------------------------------- | ------------------- | ------------------------------- |
| the S24's own (d2b78b3) | the S24     | 85.8 s         | 718086584 | 125.6 to 126.7                  | 64.6 s, 166.2x      |                                 |
| the PC's                | the 9950X   | 49.2 s, 4.5 GB | 717939128 | 125.3 to 126.1                  | 60.7 s, 176.9x      | 0 of 27119                      |

(The compile is QNN's finalize, from the profile.) Both caches have the
same key, the same QNN build (v2.51.0.260929181845), and the S24 read the
PC's as its own, compiling nothing; two compiles on the PC wrote the same
bytes. The two graphs are not the same bytes, 147456 fewer in the PC's,
and test.wav's check differs in the fourth digit (mean cosines 0.999864,
0.999699, 0.999877 against the S24's own 0.999863, 0.999702, 0.999886):
the S25's own graph's, to the last digit. Against the CPU's transcripts,
19 words of 27114 (0.070%), as the S24's own graph. The whole run's
difference in time, with the chunk's the same, was not looked into. Run
on the PC itself, its QNN emulating the NPU, the graph took 366 s for
test.wav's 60 s; pk_qnn_encode now leaves a graph made for another chip
to the CPU.
That emulated run's transcript of test.wav was, byte for byte, the S24's
NPU's from the same graph: the PC can say what a graph for a chip
nobody here has will write, at a sixth of real time, though not how fast
the chip writes it.

# 2026-11-03 - the PC's graph on the S25

|         |                                                                                                                                                                             |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| machine | the 9950X above for the compile; the Galaxy S25 Ultra above (SM8750, V79) for the run, on its charger, with libQnnHtp.so, the V79 stub and skel, and no libQnnHtpPrepare.so |
| code    | qnn 4e48aee (core/qnn.c as ff1baff); the phone's pktito-c built with the NDK r30's clang                                                                                    |

The S25's QNN describes its chip as PKTITO_QNN_TARGET=69,79,8,1,0: SoC
model 69 (SM8750), V79, 8 MB of VTCM, DLBC and no signed PD. Without the
prepare library its QNN builds no graph ("HTP Prepare backend loading
failed", graphCreate's error 1009), so the graph it ran was the PC's or
none.

| the graph               | compiled on | in             | bytes     | accelerator ms a chunk, the S25 | chapter 1, bench, best of 5 | AliceBench, the S25 | words not the S25's own graph's |
| ----------------------- | ----------- | -------------- | --------- | ------------------------------- | --------------------------- | ------------------- | ------------------------------- |
| the S25's own (d2b78b3) | the S25     | 59.2 s         | 715526584 | 131.1 to 131.6                  | 3.113 s, 205.7x             | 60.8 s, 176.6x      |                                 |
| the PC's                | the 9950X   | 46.2 s, 4.5 GB | 715403704 | 130.7 to 132.9                  | 3.173 s, 201.8x             | 62.1 s, 172.7x      | 0 of 27119                      |

(The PC's 46.2 s is the whole process's wall time; the V75 graph's
above was 52.2 s of it, 49.2 s QNN's finalize.) The S25 read the PC's
cache as its own, the file's time and size the same after the run, and
test.wav's check against the CPU's encoder gave the S25's own graph's
numbers to every digit it prints: mean cosines 0.999863, 0.999702,
0.999886, worst 0.989012, 0.992689, 0.999466, max |diff| 0.04297,
0.06538, 0.009942. The twelve chapters' transcripts are the S25's own
graph's byte for byte; against the S24's NPU they differ in 2 words of
27119, as the S25's own do, and against the CPU's in 17 of 27114
(0.063%).

The whole run took 1.4 s longer with the chunk's time the same: chapters
1 to 3 and 5 to 7 within 0.04 s of the S25's own graph's run, the others
0.09 to 0.46 s slower. Run again, the PC's graph took 6.22 and 6.49 s
for chapter 4 (7.00 s in the run, 6.54 s with the S25's own graph) and
7.32 and 7.61 s for chapter 10 (7.49 and 7.26 s): the phone's spread
from one run to the next is as wide as the difference.

A correction to the two sections above. The S25 section's row of
test.wav's mean cosines has the phones swapped: by the runs' logs the
S24's own graph gives 0.999864, 0.999699, 0.999877 and the S25's
0.999863, 0.999702, 0.999886. The PC's graph for the S24 gave the S24's
own graph's numbers to every printed digit (worst 0.992759, 0.991163,
0.999035, max |diff| 0.03516, 0.05894, 0.01287), not the S25's, as the
PC section says. On both phones the PC's graph computes what the
phone's own does.

# 2026-11-03 - the PC's graph on a Snapdragon X Elite

|         |                                                                                                                                                                                                                          |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| machine | a Dell XPS Googlebook: X1E80100 (Snapdragon X Elite, Hexagon V73, 12 Oryon cores), Android 17, on its charger; QNN 2.51's libQnnHtp.so, V73 stub and skel, and its prepare library for the Googlebook's own compile only |
| code    | as the S25's above                                                                                                                                                                                                       |

Its QNN describes the chip as PKTITO_QNN_TARGET=60,73,8,1,1: SoC model
60 (SC8380XP, the X1E80100's), V73, 8 MB of VTCM, DLBC, and the signed
PD the phones lack. The vendor partition carries QNN 2.44
(/vendor/lib64/libQnnHtp.so, libQnnHtpPrepare.so, the V73 stub and
skel), and only libcdsprpc.so of them is in its public.libraries.txt.
With no 2.51 prepare library in LD_LIBRARY_PATH, 2.51's backend finds
the vendor's and refuses it ("Prepare lib id mismatch: expected
(v2.51.0.260929181845), detected (v2.44.0.260225143659)", graphCreate's
error 1008), so the PC's graph ran with nothing here able to compile
another.

| the graph                     | compiled on    | in             | bytes     | accelerator ms a chunk | chapter 1, bench, best of 5 | AliceBench     | words not the Googlebook's own graph's |
| ----------------------------- | -------------- | -------------- | --------- | ---------------------- | --------------------------- | -------------- | -------------------------------------- |
| the Googlebook's own          | the Googlebook | 61.6 s         | 716468664 | 182.0 to 183.5         | 4.389 s, 145.9x             | 79.2 s, 135.4x |                                        |
| the PC's                      | the 9950X      | 48.2 s, 4.5 GB | 716317112 | 181.4 to 182.5         | 4.314 s, 148.5x             | 77.7 s, 138.2x | 0 of 27118                             |
| the CPU's encoder, 12 threads |                |                |           |                        | 4.985 s, 128.5x             |                |                                        |

(The Googlebook's compile is QNN's finalize, from the profile; the PC's
is the process's wall time. The accelerator's ms are from four runs of
test.wav that alternated the two graphs from their caches; the run that
compiled the Googlebook's own took 188.1 to 188.4 in that process.) The
PC's cache was the same file, time and size, after its run, and
test.wav's check gave the Googlebook's own graph's numbers to every
printed digit: mean cosines 0.999865, 0.999685, 0.999885, worst
0.993358, 0.990308, 0.999265, max |diff| 0.03369, 0.05955, 0.01116.
Both graphs use 4 HVX threads. The twelve chapters' transcripts are the
Googlebook's own graph's byte for byte; against the S24's NPU 2 words of
27119 differ, against the S25's 4, and against the CPU's 17 of 27114
(0.063%).

Chapter 1 on this NPU is 1.16 times this CPU's 12 threads, where it is
7.0 times on the S24 and 5.3 times on the S25 (16.573 s, 38.6x, above):
a V73's chunk is 182 ms to the V75's 126, and the X Elite's CPU is the
fastest of the three by 3.3 times.

# 2026-11-03 - the slim model file with the PC's graph

|         |                                                                                                                                                                                                                                                                                                                                  |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| machine | the Googlebook above, with QNN 2.51's libQnnHtp.so, V73 stub and skel and no prepare library; the 9950X for the compiles                                                                                                                                                                                                         |
| code    | qnn's working tree on 118b50d: a build with QNN=1 takes the slim file (pktito-tools slim), its encoder the graph in the cache alone; the cache's key the model's checkpoint and conversion (source_sha256, converter) and not its file's SHA-256, the QNN that compiled a graph kept beside the key and not in it (pktito-qnn 6) |

The slim v2 file is 12,115,524 bytes to the full file's 624,480,196, its
source_sha256 and converter the full file's, so the graph the PC compiles
from the full file is the slim file's too: 716,317,176 bytes for the
Googlebook, the 64 more than the section above's the QNN's build.

| model file     | the graph                      | test.wav, mean cosines to the CPU's   | accelerator ms a chunk | chapter 1, bench, best of 5 | AliceBench     | words not the full file's with the same graph |
| -------------- | ------------------------------ | ------------------------------------- | ---------------------- | --------------------------- | -------------- | --------------------------------------------- |
| full, 624.5 MB | the PC's, pktito-qnn 5 (above) | 0.999865, 0.999685, 0.999885          | 181.4 to 182.5         | 4.314 s, 148.5x             | 77.7 s, 138.2x |                                               |
| full, 624.5 MB | the PC's, pktito-qnn 6         | 0.999865, 0.999685, 0.999885          | 181.7 to 182.3         |                             |                |                                               |
| slim, 12.1 MB  | the PC's, pktito-qnn 6         | none: no CPU encoder to check against | 181.8 to 182.1         | 4.337 s, 147.7x             | 77.6 s, 138.3x | 0 of 27118                                    |

The twelve chapters' transcripts from the slim file are the full file's
byte for byte, and test.wav's too; the graph's file had the same time and
size after the runs. Where the slim file can't run, pktito-c says so and
writes nothing:

- with no graph's file: "the model leaves its encoder out: give its
  compiled graph's file";
- with the S25's V79 graph, QNN refuses it ("Request feature arch with
  value 79 unsupported; Failed to register context to device and backend;
  Failed to create context from binary"), as QNN_CONTEXT_ERROR_CREATE_FROM_BINARY
  (5005), not the BINARY_CONFIGURATION QnnContext.h names for a binary not
  configured for the device; pktito-c names the file and both QNNs' builds;
- with --max-chunk 60, a chunk past the graph's 3072 frames has no encoder
  (pk_asr_run -1): "the encoder could not run: parakeet-tdt-0.6b-v2 leaves
  its own out, and the accelerator's failed or takes no chunk this long".

On the PC, the x86 QNN took the V79 graph for a device described as the
Googlebook's V73 without complaint: only the chip refuses another's graph.

# 2026-11-03 - pktito-qnn-graph 1, the graphs the app downloads

|         |                                                                                                                                                                                                                                                                                          |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| machine | the 9950X for the compiles (QNN 2.51.0.260929's x86_64 prepare library); the Googlebook (X1E80100, V73), the Galaxy S25 Ultra (SM8750, V79) and the S24 Ultra (SM8650, V75), each through pktito dev 0.9.5 targeting Android 17 (API 37), from a server on the 9950X through adb reverse |
| code    | qnn on c0a17e8: the graph's file "pktito-qnn-graph 1", its 616-byte header the chip it was compiled for and the pktito (made) and QNN (built) that made it; then 4249d28, a graph of another Hexagon's refused before QNN and one of another SoC of the same Hexagon given to QNN        |

The six, each compiled from the full file in a process of its own
(make clean first, so that made names c0a17e8):

| model | chip (PKTITO_QNN_TARGET) | bytes       | SHA-256                                                          | compile, wall | peak RSS (GNU time) |
| ----- | ------------------------ | ----------- | ---------------------------------------------------------------- | ------------- | ------------------- |
| v2    | SM8650, 57,75,8,1,0      | 717,939,304 | f12c6b55f288c515f11f9e63fa06c230424f23b7b39b39f474a38b2ab91b16f3 | 53.38 s       | 4.34 GiB            |
| v2    | SM8750, 69,79,8,1,0      | 715,403,880 | b2293ca9d91cade115af956dadf3fcc495e24db5b17c6aab60a002d841eadad6 | 43.17 s       | 4.24 GiB            |
| v2    | X1E80100, 60,73,8,1,1    | 716,317,288 | 48ccb9089cce638ed0555dfbf1dbcff805e17a0a657fb270d05d795ed8de5bd4 | 47.54 s       | 4.27 GiB            |
| v3    | SM8650, 57,75,8,1,0      | 717,939,304 | 589170fab8e81f64334d4b3fc7ea3dfcaf8fbd1758b07e43d628b9468d298429 | 54.14 s       | 4.35 GiB            |
| v3    | SM8750, 69,79,8,1,0      | 715,403,880 | 725c8414d6bbeeca26e2fbb5ac161cc076df3dc707f12878236d6b5eabdc9556 | 44.47 s       | 4.28 GiB            |
| v3    | X1E80100, 60,73,8,1,1    | 716,317,288 | 02a6cd893573211484344d9f4ca2adc9e7728012ce92cbd47d8fb0e59ce3ae61 | 48.71 s       | 4.30 GiB            |

QNN's binary in each, against pktito-qnn 6's of the same model and chip
from the same QNN: the SM8650's and the X1E80100's are byte for byte the
same; the SM8750's are not, v2's in 7,786,869 bytes, all in its first
11 MB or so, v3's in 1,449, in its first 1.3 MB. The V79's compile is not
the same from one run to the next where the V75's and V73's are.

Through the app, pktito dev 0.9.5 with its own download of the NPU's
edition (the slim file and the graph), chapter 1 of Alice in Wonderland:

| graph        | phone            | graph loaded | chapter 1           |
| ------------ | ---------------- | ------------ | ------------------- |
| v2, SM8650   | Galaxy S24 Ultra | 0.84 s       | 112 segments, 3.5 s |
| v2, SM8750   | Galaxy S25 Ultra | 0.91 s       | 112 segments, 3.6 s |
| v2, X1E80100 | Googlebook       | 0.92 s       | 112 segments, 4.6 s |
| v3, SM8650   | Galaxy S24 Ultra | 0.65 s       | 117 segments, 3.5 s |
| v3, SM8750   | Galaxy S25 Ultra | 0.94 s       | 117 segments, 3.5 s |
| v3, X1E80100 | Googlebook       | 0.91 s       | 117 segments, 4.6 s |

v2's 2194 words are the same on the three NPUs and one from the CPU's
(152.62 s: "Why" for "Why,"); the S24's (from its pktito-qnn 6 graph,
the same binary) and the S25's are the same to the hundredth of a
second, and the Googlebook's has one end time 0.08 s from them (436.66 s
for 436.74). v3's 2199 words are the same on the S24 and the
Googlebook, two times 0.08 s apart. The SM8750's compile, not the same
twice, gave the S24's V75 transcript.

On the PC, 4249d28's check: QNN 2.51's x86 HTP backend opened the
SM8650's graph under PKTITO_QNN_TARGET 12,75,8,1,0 as under its own
57,75,8,1,0, the slim v2 file with it (pktito-c quits "--qnn: ..." where
the graph doesn't open, and didn't); no chunk then ran, a graph for
another chip being the cache's alone on a PC ("the encoder could not
run: parakeet-tdt-0.6b-v2 leaves its own out"). Under 57,75,8,1,0 the
SM8750's graph is refused before QNN: "parakeet-tdt-0.6b-v2-sm8750.qnn
is the graph of the NPU 69,79,8,1,0, and this one is 57,75,8,1,0". A
graph of another SoC of the same Hexagon is untried on a phone, each
Hexagon here being one SoC's.

# 2026-11-03 - the VTCM and the DLBC a graph asks for

## Hypothesis: the whole VTCM, and DLBC

compile() sets two of the HTP's graph options, the compiler's
optimization (3) and the sigmoid's high precision, and leaves the rest
at QNN's defaults. qnn-context-binary-utility on pktito-qnn-graph 1's
three v2 graphs (QNN's binary, after the 616-byte header) says what that
came to: each graph was given 4 MB of VTCM (vtcmSize 4, the HTP
backend's default for a graph that names none) where each of the three
chips has 8 (PKTITO_QNN_TARGET's third number); DLBC off (htpDlbc 0),
which each of the three has (the fourth); and a spill-fill buffer of
100.5 MB for the SM8650, 102.3 for the SM8750 and 99.8 for the X1E80100,
the DDR the graph moves tensors out to and back from where its VTCM
can't hold them.

About half of the S24's 125.6 ms a chunk is the matrices: the layers'
231 G multiply-adds and the position term's 77 G would take 61 ms at
the rate of the FullyConnected chain above (2026-11-02, 45.8 ms for
230 G). The other 65 ms are the vector ops and the moves, which the
profile's cycles, summed over the HTP's threads, can't place in time. A
feed-forward's hidden tensor, [384][4096], is 3.1 MB in fp16, and the
stream 0.8 MB: in 4 MB the hidden tensor and the tiles of the weights
that make it or take it can't be held together, and what the VTCM can't
hold goes out to DDR and comes back.

So, with QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE at
QNN_HTP_GRAPH_CONFIG_OPTION_MAX (the SoC's whole VTCM, the SoC named by
QNN_HTP_DEVICE_CONFIG_OPTION_SOC, which the PC's compile does already),
the spill-fill buffer and the bytes the graph spills and fills should
fall, and the chunk with them: if a third of those 65 ms is the moves
and twice the VTCM halves them, by some 11 ms, to about 115. The compile
reports the bytes itself, on the PC, before a phone runs the graph: QNN's
profile of the finalize gives each HTP's input fill, intermediate fill
and intermediate spill, and a simulated time.

DLBC is two options, both for a graph compiled ahead of time, as ours
are: QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_DLBC_WEIGHTS, the weights
compressed in DDR and read so, and ENABLE_DLBC, which the HTP backend's
guide calls the inputs' and QnnHtpGraph.h calls reserved. The matrices
read 600 M int8 weights a chunk, 4.8 GB/s at 126 ms, where the chain
above read them at 28 GB/s (21.5 ms with 16-bit activations): they are
not waiting on their weights, and DLBC for the weights should buy little
time. If the binary keeps them compressed, the graph's file shrinks, and
it is the download. DLBC on the spills and fills should buy what the
moves cost less what the VTCM already saved.

The ops are the same in each graph, so the numbers should be the graph's
before, but for where a new tiling changes the order of a sum
(test.wav's cosines to the CPU's, and AliceBench's words against the
graph before, say). Each graph is measured on the S24: the accelerator's
ms a chunk, chapter 1's bench (best of 5), and the size of its file.

## The whole VTCM, and DLBC: rejected

The S24's v2 graph compiled on the 9950X four ways (core/qnn.c's
compile() with the options given; the knobs that set them not kept). What
QNN's description of each says, and each one's QNN binary against the
shipped graph's:

| graph                                             | vtcmSize, htpDlbc | spill-fill buffer | constants | QNN's binary not the shipped graph's | finalize |
| ------------------------------------------------- | ----------------- | ----------------- | --------- | ------------------------------------ | -------- |
| as shipped (pktito-qnn-graph 1)                   | 4, 0              | 100.5 MB          | 672.1 MB  |                                      | 49.9 s   |
| VTCM_SIZE at QNN_HTP_GRAPH_CONFIG_OPTION_MAX      | 8, 0              | 100.5 MB          | 672.1 MB  | 75,632 bytes, all in its first MB    | 50.0 s   |
| the same, and ENABLE_DLBC                         | 8, 1              | 100.5 MB          | 672.1 MB  | 6,677 bytes more, in its first MB    | 50.4 s   |
| the same, and ENABLE_DLBC_WEIGHTS for ENABLE_DLBC | 8, 0              | 100.5 MB          | 674.8 MB  | the file 2,945,024 bytes longer      | 68.9 s   |

The whole VTCM gave the same ops, the same spill-fill buffer and the same
constants; DLBC on the weights made the constants 2.7 MB longer, so the
int8 weights don't compress and there are no fewer bytes to read: that
graph wasn't run. The finalize's profile has no performance estimate for
this SoC (only its phases), so the phone decides.

On the S24 (the binaries spliced on the phone from 2026-11-02's graph,
whose QNN binary is the shipped one byte for byte, and checked against
the PC's SHA-256s), test.wav checked against the CPU's encoder and
profiled, then chapter 1's bench, the three in turn, twice, on AC, from
26.6 to 32.4 °C:

| graph                   | accelerator ms a chunk, test.wav's three, two rounds | chapter 1, bench, best of 5, two rounds |
| ----------------------- | ---------------------------------------------------- | --------------------------------------- |
| as shipped              | 125.5, 126.5, 125.8; 126.8, 124.9, 125.8             | 3.060 s, 3.166 s                        |
| the whole VTCM          | 124.4, 125.1, 124.6; 126.1, 125.7, 124.8             | 3.159 s, 3.219 s                        |
| the whole VTCM and DLBC | 126.3, 124.7, 126.2; 126.3, 126.0, 125.8             | 3.238 s, 3.251 s                        |

All three are within a graph's own spread, and test.wav's cosines are
the same to every digit in all six runs (0.999864, 0.999699, 0.999877).
The bench's times rose in the order the runs were made, with the phone's
temperature, and the accelerator's didn't: the bench has the CPU's
decoder in it. The profile below says why the VTCM changed nothing: the
graph as shipped already has 7.9 MB of VTCM in use at its peak.

# 2026-11-03 - QNN's optrace of a chunk

## How

The finalize with a detailed profile and QNN_PROFILE_CONFIG_OPTION_ENABLE_OPTRACE
writes the schematic, encoder_schematic.bin (383 MB), to the working
directory; the context binary is the shipped graph's byte for byte. On the
S24, Qualcomm's qnn-net-run ran that binary (the graph's file less its
616-byte header) three times on zeros, all frames valid, with
--perf_profile burst --profiling_level detailed --profiling_option optrace:
127.0 and 127.9 ms of the accelerator a run, test.wav's chunks' 125 to 127. No subsystem restart, unlike the HTP's linting level. On the PC,
qnn-profile-viewer with libQnnHtpOptraceProfilingReader.so and the
schematic made the timeline (chrometrace.json, 298 MB, for Perfetto) and
QNN's analysis summary, QHAS (its JSON and HTML), in 45 s.

## What a chunk is

127.5 ms, 246.2 M cycles at 1.93 GHz. The HMX is busy 70.1% of it; the
four HVX threads 66.3, 58.0, 57.8 and 66.2%. DRAM: 1.33 GB read and 0.55
GB written a chunk; the VTCM 7.9 MB at its peak.

QHAS's dominant path gives each moment of the timeline to one op, the one
on the highest of the HMX, an HVX thread, a DMA read, a DMA write, a
checkpoint and a sync that is busy then. By what QNN calls the ops:

| HTP's op                    | ms of the 127.5 | what it is                                                           |
| --------------------------- | --------------- | -------------------------------------------------------------------- |
| q::ConvLayer.fp16.s1.tcm    | 54.5            | the matrices on the HMX, every product of the graph                  |
| q::packed_quant_to_fp16     | 12.6            | the int8 weights made fp16 for the HMX, on the HVX                   |
| q::Mul.fp16                 | 10.9            | the multiplies                                                       |
| q::DilatedConvLayer_s1.fp16 | 9.4             | the convolution module's depthwise, on the HVX                       |
| q::Concat                   | 7.4             | the compiler's own: 3.0 in the subsampling, 2.2 in each feed-forward |
| q::Swish_fp16               | 5.0             | the SiLUs                                                            |
| @Fill, @Spill               | 4.0, 4.0        | DRAM and back: 568 MB in, 545 MB out                                 |
| the layer norms' three      | 5.9             |                                                                      |
| q::Add.fp16                 | 3.4             |                                                                      |
| q::Reshape                  | 2.5             |                                                                      |
| q::Softmax                  | 2.0             |                                                                      |
| everything else             | 6.0             |                                                                      |

And by the graph's own blocks (core/qnn.c's op names), ms of the 127.5:

| block                          | ms   | of which                                                                                                                             |
| ------------------------------ | ---- | ------------------------------------------------------------------------------------------------------------------------------------ |
| ff1                            | 24.7 | its two matrices 18.7, the SiLU's multiply 4.5, the norm 1.3                                                                         |
| ff2                            | 24.6 | its two matrices 18.8, the SiLU's multiply 4.6, the norm 0.9                                                                         |
| pos, the turn a head at a time | 22.5 | in its 384 FullyConnected ops, which QNN folds the rest into: the products on the HMX 13.9, the multiply by the tables 8.4           |
| att, the rest of the attention | 21.7 | the five adds 6.9, Q, K, V and O 6.5, ac and pv 2.6, the heads' transpose 2.4, softmax 2.0, the norm 1.2                             |
| conv                           | 19.0 | the depthwise 9.5, pw1 and pw2 5.2, the GLU's sigmoid 2.2, the norm 1.2                                                              |
| sub                            | 13.3 | its matrices 7.6 (131 MB from DRAM, 127 MB to it), the masks' multiplies 2.4, the ReLUs 1.6 (52 MB in, 50 MB out), the depthwise 1.2 |
| out, the layers' last norms    | 1.4  |                                                                                                                                      |

So the HMX's products are 54.5 ms and the rest of the chunk, 73 ms, is
the HVX's work and the moves that the HMX waits on. The feed-forwards'
products run at 0.24 ms a G multiply-adds (77.3 G each), the turn's at
0.29 (77.3 G, fp16 weights), against 0.20 for 2026-11-02's chain of
FullyConnected ops with int8 weights. Where the HMX waits: the weights'
conversion to fp16, which 16-bit activations would not need (the chain
with them: 21.5 ms to 45.8); the depthwise convolution, 18.8 M cycles
busy and 18.3 M of the dominant path, so on one thread with nothing
beside it, on [1][384][1][1024], a width of 1; the turn's multiply by its
tables; the compiler's own concatenations; and the subsampling's
tensors, 50 MB after its first convolution, which go out to DRAM and
back.

## Hypothesis: a P point

The HTP backend's guide has P points: settings of the compiler at O=3
(QNN_HTP_GRAPH_CONFIG_OPTION_FINALIZE_CONFIG, key "P", one of 1 to 6, 8,
13, 15 to 17 and 19 to 23), "for example, tradeoffs between latency and
DRAM bandwidth", each "bit-accurate with a graph compiled without P
points", and none better than another in general. What the profile says
the HMX waits on is the compiler's to arrange - what runs beside what,
what goes out to DRAM and comes back - so some P point may take part of
the 73 ms that aren't the HMX's products; the numbers should be the
shipped graph's to every digit. Each P point is compiled on the PC, and
each that compiles is run on the S24 (the accelerator's ms a chunk on
test.wav, and its cosines); the quickest then on chapter 1's bench and
AliceBench.

## Hypothesis: the attention a head at a time

The heads' reshapes and transposes of K, V and q + u into [8][384][128],
and of pv back, are 2.4 ms of the chunk's dominant path, and the eight
heads' bd are concatenated into [8][384][384] for the adds and the
softmax (the profile puts no q::Concat in pos, so QNN has folded that
one). A head at a time, from the
stream's columns as the turn already takes q + v's: q + u's, K's and V's
128 columns sliced out; ac, the two adds, the softmax and pv on [384][384]
and [384][128]; and the eight heads' pv concatenated along the columns
into [384][1024], which O's product takes. That leaves no [8][384][384]
concatenation and no transpose, and a concatenation of [384][128]s in
whole 32-column blocks of the HTP's layout. If the slices and that
concatenation cost nothing, the chunk is some 2 ms quicker, the
transposes' time. The arithmetic is the same op for op (products' sums of 128
and 384, the softmax of a row), so the numbers should be the graph's
before but for where a new tiling changes the order of a sum.

## Hypothesis: the depthwise convolution on the HMX, or wide

The convolution module's depthwise, 9 taps on [1][384][1][1024], is 9.5
ms of the dominant path and 18.8 M cycles busy: one HVX thread with
nothing beside it (q::DilatedConvLayer_s1.fp16), on a width of 1, which
the HTP's tiles pad. Two other forms of the same sums. A Conv2d of G
groups (the HTP supplement allows groups with fp16 weights), each D/G
input channels to D/G output channels, its filter [9][1][D/G][D] zero
but for each output channel's own input channel: the HMX's, if the HTP
runs a grouped convolution there, 384 x 9 x (D/G) x 1024 multiply-adds a
layer, 2.7 G a chunk at G = 32, some 0.6 ms at the feed-forwards' rate;
the zeros add nothing to the sums. Or the DepthWiseConv2d as it is but
with time along the width, [1][1][384][1024] and the filter [1][9][1][1024],
for whatever the HTP's tiles make of a height of 1. Either should take
most of the 9.5 ms, the chunk some 6 to 8 ms quicker; the numbers the
graph's before but for the order of the sums.

## A P point, a head at a time, the depthwise: measured

|         |                                                                                                                                                                                                                                                                      |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| machine | the S24 Ultra (SM8650, V75), the S25 Ultra (SM8750, V79) and the Googlebook (X1E80100, V73), each on its charger, with QNN 2.51's libQnnHtp.so, its Hexagon's stub and skel and its prepare library; the 9950X for the S24's first graphs                            |
| code    | qnn on 681bea0, core/qnn.c with knobs for the experiment (not kept): PKTITO_QNN_P, the P point, a uint32 (as an int32 the graph was the same); PKTITO_QNN_ATT, the attention a head at a time; PKTITO_QNN_DW, the depthwise along the width or as a Conv2d of groups |

Each graph compiled from the full v2 file in a process of its own - the
S24's P points 1 to 6, its P2 a head at a time and grouped, and its P6
wide on the 9950X, the rest on the phone that ran them - then test.wav
checked against the CPU's encoder and profiled, twice, each run a process
reading the graph from its file. Below, the accelerator's ms a chunk, the
lowest and highest of the two runs' three chunks. The batteries were at
26.6 to 37.5 °C (the S24), 33.7 to 37.7 (the S25) and 29.0 to 35.0 (the
Googlebook). The Googlebook's graph as shipped, compiled there for these
runs, took 188.3 to 188.9 where the sections above have 181.4 to 183.5
for the same graph: each table here holds one evening's runs.

The P points, the graph otherwise as shipped:

| P                | the S24                                           | the S25        | the Googlebook |
| ---------------- | ------------------------------------------------- | -------------- | -------------- |
| none, as shipped | 124.9 to 126.8                                    | 132.8 to 137.4 | 188.3 to 188.9 |
| 1, 3             | the shipped graph but for 2 and 10 bytes: not run |                |                |
| 2                | 115.0 to 116.1                                    | 127.4 to 132.8 | 183.8 to 185.1 |
| 4                | 175.4 to 183.3                                    |                |                |
| 5                | 124.2 to 125.6                                    |                |                |
| 6                | 114.8 to 116.4                                    | 100.0 to 101.5 | 175.9 to 181.2 |
| 8                | the compile stopped after 31 minutes              |                |                |
| 13               | 115.3 to 116.4                                    | 101.0 to 102.2 | 178.7 to 179.5 |
| 15               | 124.3 to 125.5                                    | 136.3 to 140.6 | 183.1 to 185.3 |
| 16               |                                                   | 129.3 to 133.6 | 183.3 to 184.4 |
| 17               |                                                   |                | 182.0 to 183.0 |
| 19               |                                                   |                | 184.7 to 186.0 |
| 20               |                                                   |                | 186.3 to 187.2 |

Every P point's cosines are the graph's without one to every digit
pktito prints, on each chip, as QNN's guide has it ("bit-accurate"). P15's
and P16's graphs are the size of the phone's own compile of the graph
without one, 715,526,760 bytes on the S25 and 716,468,840 on the
Googlebook, so likely the same graph, and their times the graph's own at
the phone's temperature then. P6 and P13 are the quickest on all three,
P2 as quick on the S24: 8% off the S24's chunk, 25% off the S25's and 5%
off the Googlebook's.

The attention a head at a time:

| graph                                    | the S24        | the S25        | the Googlebook |
| ---------------------------------------- | -------------- | -------------- | -------------- |
| P2                                       | 115.0 to 116.1 | 127.4 to 132.8 | 183.8 to 185.1 |
| P2, a head at a time                     | 112.4 to 113.5 | 133.5 to 138.2 | 178.2 to 180.0 |
| P6                                       | 114.8 to 116.4 | 100.0 to 101.5 | 175.9 to 181.2 |
| P6, a head at a time                     | 114.3 to 115.2 | 100.6 to 101.6 | 177.5 to 181.7 |
| P6, the depthwise wide                   | 109.6 to 111.2 | 97.1 to 99.5   | 171.9 to 173.4 |
| P6, the depthwise wide, a head at a time | 107.9 to 109.7 |                |                |

A head at a time is quicker at P2 on the S24 and the Googlebook and
slower on the S25; at P6, within each chip's spread. Its cosines are the
graph's without it to every digit. Rejected.

The depthwise, wide (time along the width, [1][1][384][1024], the filter
[1][9][1][1024]) or as a Conv2d of 32 groups, and the P points with it
wide:

| graph                     | the S24        | the S25        | the Googlebook |
| ------------------------- | -------------- | -------------- | -------------- |
| as shipped                | 124.9 to 126.8 | 132.8 to 137.4 | 188.3 to 188.9 |
| P2, a Conv2d of 32 groups | 149.8 to 150.9 |                |                |
| wide                      | 110.6 to 112.5 | 131.8 to 135.7 | 169.8 to 173.7 |
| wide, P2                  | 132.5 to 133.9 |                |                |
| wide, P6                  | 109.6 to 111.2 | 97.1 to 99.5   | 171.9 to 173.4 |
| wide, P13                 | 109.7 to 111.2 | 93.8 to 95.6   | 172.1 to 173.4 |
| wide, P17                 | 133.0 to 133.8 | 119.3 to 125.0 | 202.7 to 207.9 |
| wide, P19                 | 111.3 to 112.3 | 109.2 to 111.2 | 174.0 to 174.4 |
| wide, P20                 | 121.2 to 122.9 | 109.9 to 112.5 | 182.0 to 184.7 |
| wide, P21                 | 113.8 to 116.5 | 99.5 to 101.7  | 180.3 to 180.9 |
| wide, P22                 | 120.1 to 121.2 | 129.5 to 133.8 | 171.1 to 171.7 |
| wide, P23                 | 121.4 to 121.9 | 139.4 to 139.6 | 181.3 to 182.8 |

The grouped Conv2d took 25 min 52 s to compile on the 9950X, its ops
65 MB in QNN's description of the graph to the shipped graph's 29.1, and
the chunk 35 ms longer than P2's: rejected. Wide, the depthwise is the
quickest on the S24 and the Googlebook with or without a P point, 11%
and 9% off their chunks alone (its graph's ops 19.9 MB at P6); on the
S25 it takes nothing off alone and 7 ms at P13, and P13 with it is the
S25's quickest by 3.6 ms, 30% off the chunk as shipped. The P points
interact with it: P2, as quick as P6 on the S24 without it, is 23 ms
slower than P6 with it.

Wide, the depthwise's numbers are not the same: its sums in another
order of the HTP's. test.wav's mean cosines, the S24's 0.999856,
0.999685, 0.999882 to the shipped graph's 0.999864, 0.999699, 0.999877;
the S25's 0.999871, 0.999708, 0.999889 to 0.999863, 0.999702, 0.999886;
the Googlebook's 0.999866, 0.999699, 0.999876 to 0.999865, 0.999685,
0.999885. Every P point with it gives its numbers to every digit.

The S24's profile of a chunk (PKTITO_QNN_PROFILE=1, an op's cycles
summed over the HTP's units), as shipped and wide, says where the time
went: a layer's SiLU after the depthwise, the same 393,216 values either
way, took 1.83 M cycles on [1][384][1][1024] and 0.42 on [1][1][384][1024];
the depthwise 0.79 and 0.60; the reshape into it 0.23 and none. The
convolution module's ops took 117.9 M cycles of a chunk's 776.3 as
shipped and 74.9 of 699.3 wide; at P13, 120.0 and 72.9 (the SiLU 1.96
and 0.41 a layer). On a width of 1 the SiLU's multiply took four times
the cycles, as if the HTP's layout padded the width out to a tile's.

## The depthwise wide and P13: kept

core/qnn.c's layer() has the depthwise along the width, compile() P13,
the knobs are gone, and the graph's file is "pktito-qnn-graph 2", so that
a graph 1 is downloaded or compiled again, not run. Against the graph as
shipped, a chunk is 12% quicker on the S24 (109.7 to 111.2 ms), 30% on
the S25 (93.8 to 95.6) and 8% on the Googlebook (172.1 to 173.4).

# 2026-11-03 - pktito-qnn-graph 2's six graphs

|         |                                                                                                                                                                                                                                                                                           |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| machine | the 9950X for the compiles (QNN 2.51.0.260929's x86_64 prepare library); the S24 Ultra, the S25 Ultra and the Googlebook for the runs, each on its charger, with QNN 2.51's libQnnHtp.so and its Hexagon's stub and skel, and no prepare library on the library path, as the app has none |
| code    | qnn e3416a8: the depthwise along the width, P point 13, the graph's file "pktito-qnn-graph 2"                                                                                                                                                                                             |

The six, each compiled from the full file in a process of its own, two at
a time (make clean first, so that made names e3416a8):

| model | chip (PKTITO_QNN_TARGET) | bytes       | SHA-256                                                          | compile, wall | peak RSS (GNU time) |
| ----- | ------------------------ | ----------- | ---------------------------------------------------------------- | ------------- | ------------------- |
| v2    | SM8650, 57,75,8,1,0      | 714,506,856 | f687a98383c956863c049381945097c8f059dfd37e86bb421963f9896065308d | 51.88 s       | 4.37 GiB            |
| v2    | SM8750, 69,79,8,1,0      | 713,568,872 | b2181d0606b6d865fcddf020bb840e26db21dcceafab20427ec6aa111ec7d2c2 | 50.87 s       | 4.21 GiB            |
| v2    | X1E80100, 60,73,8,1,1    | 714,523,240 | 37679e4c0ebc11820300529acb192bfb297ee6f26dd90a32d812a935ec97b40a | 55.18 s       | 4.34 GiB            |
| v3    | SM8650, 57,75,8,1,0      | 714,506,856 | e5b4e142297c8f73e0f441878eff53c1581bcbf077de2b3c3165e22ea542474f | 51.97 s       | 4.39 GiB            |
| v3    | SM8750, 69,79,8,1,0      | 713,568,872 | 770baa635f7d6083f42866ce78f0ffede7220ceee95a325e1efff7c45bcc2b36 | 51.12 s       | 4.21 GiB            |
| v3    | X1E80100, 60,73,8,1,1    | 714,523,240 | 765f16df8830c9c419ed88cb325f93d1d76873a5e65bb751a86879d854d52443 | 55.45 s       | 4.35 GiB            |

Each is 3.4, 1.8 and 1.8 MB smaller than graph 1's for its chip.

On each phone, its chip's graph 2 and graph 1 (the files the app
downloads now, their SHA-256s checked on the phone), each read from its
file, by e3416a8's pktito-c for graph 2 and 681bea0's (its experiment's
knobs unset) for graph 1, both built by the NDK r30's clang: test.wav
checked against the CPU's encoder and profiled; chapter 1's bench, best
of 5, twice, the graphs alternating; then AliceBench's twelve chapters a
process each; v2, then v3.

| model | phone      | graph | accelerator ms a chunk | test.wav's mean cosines      | chapter 1, bench, best of 5, two rounds | AliceBench                |
| ----- | ---------- | ----- | ---------------------- | ---------------------------- | --------------------------------------- | ------------------------- |
| v2    | S24        | 1     | 125.6 to 126.9         | 0.999864, 0.999699, 0.999877 | 3.193 s, 3.353 s                        | hung in chapter 5 (below) |
| v2    | S24        | 2     | 113.9 to 114.8         | 0.999856, 0.999685, 0.999882 | 2.938 s, 3.100 s                        | 60.5 s, 177.5x            |
| v2    | S25        | 1     | 130.3 to 131.4         | 0.999863, 0.999702, 0.999886 | 2.956 s, 3.432 s                        | 66.8 s, 160.7x            |
| v2    | S25        | 2     | 91.8 to 92.9           | 0.999871, 0.999708, 0.999889 | 2.412 s, 2.552 s                        | 50.1 s, 214.4x            |
| v2    | Googlebook | 1     | 183.6 to 184.0         | 0.999865, 0.999685, 0.999885 | 4.382 s, 4.362 s                        | 78.7 s, 136.4x            |
| v2    | Googlebook | 2     | 172.0 to 173.1         | 0.999866, 0.999699, 0.999876 | 4.094 s, 4.077 s                        | 73.9 s, 145.2x            |
| v3    | S25        | 1     | 135.0 to 139.0         | 0.999825, 0.997701, 0.999708 | 3.377 s, 4.128 s                        | 80.9 s, 132.6x            |
| v3    | S25        | 2     | 107.8 to 109.4         | 0.999839, 0.997674, 0.999713 | 3.165 s, 3.351 s                        | 64.3 s, 166.9x            |
| v3    | Googlebook | 1     | 183.5 to 183.9         | 0.999838, 0.998727, 0.999707 | 4.367 s, 4.314 s                        | 78.7 s, 136.4x            |
| v3    | Googlebook | 2     | 169.2 to 173.2         | 0.999843, 0.998755, 0.999712 | 4.057 s, 4.055 s                        | 74.1 s, 144.8x            |

Graph 2 takes 9% off the S24's chunk, 29% off the S25's and 6% off the
Googlebook's; 25% off the S25's AliceBench and 6% off the Googlebook's.
On the S24, chapters 1 to 4 took 20.20 s with graph 2 to graph 1's 21.61
in the same run, both slower than graph 1's 19.79 s on 2026-11-02: the
S24 was slower that evening. The S25 ran from 35.2 to 42.2 °C, v3 after
v2 and hotter, both of its graphs slower for it; the S24 from 32.7 to
39.1, the Googlebook from 29.0 to 33.0. The Googlebook's graph 1 took
183.6 to 184.0 here, near the 181.4 to 183.5 of the sections before the
sweep above, whose 188.3 to 188.9 was the Googlebook's own compile of it
that evening: against graph 1, graph 2 takes 6% off the Googlebook's
chunk, not the sweep's 8%.

AliceBench's words (score.py's count: lowercased, the punctuation off,
against the first named) and its tokens as written:

| model | phone      | graph 2's words not graph 1's             | as written | graph 1's not the CPU's | graph 2's not the CPU's |
| ----- | ---------- | ----------------------------------------- | ---------- | ----------------------- | ----------------------- |
| v2    | S24        | 2 of 27,119 (graph 1's run of 2026-11-02) | 7          | 19 of 27,114            | 17                      |
| v2    | S25        | 3 of 27,119                               | 12         | 17                      | 20                      |
| v2    | Googlebook | 0 of 27,118                               | 9          | 17                      | 17                      |
| v3    | S25        | 10 of 27,155                              | 25         | 65 of 27,529            | 61                      |
| v3    | Googlebook | 6 of 27,162                               | 18         | 48                      | 48                      |

Graph 1's runs of v2 were their 2026-11-02 runs' to the word (the S25's
and the Googlebook's whole, the S24's chapters 1 to 4). v3's CPU
transcripts are the Googlebook's CPU's (148.5 s, 72.2x), the same bits as
any CPU's. With either graph v3 strays from the CPU about three times as
far as v2, and test.wav's second chunk has a frame of v3's at 0.62 cosine
on the S25 and 0.81 on the Googlebook: v3's, not this change's.

The S24's run of graph 1 hung in chapter 5: after 9,364 bytes of the
chapter's transcript (graph 2's whole is 11,775) its process waited in
the kernel's fastrpc_wait_for_completion from 21:18 on, and SIGTERM left
it there, uninterruptible, the phone cooling and drawing the power it
does at rest; nothing in logcat. The rest of the S24's run, graph 1's
chapters 5 to 12 and v3, waits behind it. The only hang in the evening's
runs on the three phones.

## Hypothesis: the graph's ops in the order the code reads

The S24's graph 2 from the PC took 113.9 to 114.8 ms a chunk here, where
the phone's own compile of the same configuration in the sweep above
(wide, P13) took 109.7 to 111.2; the S25's 91.8 to 92.9 here to its own
compile's 93.8 to 95.6; the Googlebook's 172.0 to 173.1 to 172.1 to
173.4; the cosines the same in each pair. Two things differ between the
compiles. The prepare library, the PC's x86_64 build or the phone's
aarch64 one; and the graph: make_graph adds the ops as layer()'s calls
are evaluated, and C leaves the order of a call's arguments unspecified.
gcc (the PC's) evaluated them last first and clang (the phones') first
first, so that the GLU's mul(b, fc(.. first half ..), sigmoid(b, fc(..
second half ..))) is conv_FullyConnected_230, conv_Sigmoid_232,
conv_FullyConnected_235 in the PC's graph and conv_FullyConnected_230,
conv_FullyConnected_233, conv_Sigmoid_235 in the phone's, and the
reshape into the depthwise conv_Reshape_243 and conv_Reshape_241. Both
change what QNN makes. From e3416a8 for the X1E80100, v2: the PC's gcc
build's QNN binary (the graph file less its 616-byte header) is SHA-256
b5cd0c5b04474576... and 714,523,240 bytes in its file; the PC's clang
21 build's, compiled the same way, 9dfab7f178035ae7..., the same length;
the Googlebook's own compile from the NDK's clang build, d4bcbff3157276f5...
and 122,880 bytes longer, the difference of the S24's own compiles from
the PC's. A graph should be its source's whatever compiler builds it:
with each op's value named before the call that takes it, in the order
the code reads, a gcc build and a clang build write the same graph. Then
the S24 says which of the two moved its 4 ms: the PC's compile of that
graph runs as the phone's own does if it was the order, and doesn't if it
was the prepare library's host.

## The ops in the order the code reads: the 9950X's builds

|         |                                                                                                                                                            |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| machine | the 9950X: gcc 15.2.0 and clang 21.1.8 building pktito-c, QNN 2.51.0.260929's x86_64 prepare library compiling                                             |
| code    | qnn on 0f8385b: no call in core/qnn.c's builder takes more than one argument that adds to the graph, the others made before it in the order the code reads |

The X1E80100's v2 graph (60,73,8,1,1), compiled from the full file by a
gcc build and by a clang build, each in a process of its own, the two at
once: both QNN binaries (the file less its 616-byte header) are
9dfab7f178035ae7a03b18268cd8516d81eafc08945cc7f50d96c2f013bbf1da, the
file 714,523,240 bytes, and test.wav's transcript is the same. That is
e3416a8's clang build's binary, where e3416a8's gcc build's was
b5cd0c5b04474576...: the order the code reads is clang's, and one graph
compiled in three processes by three builds of two compilers was one
binary. Each compile took 100.4 and 101.2 s, the 9950X busy with another
job besides. make check-c QNN=1 passes.

# 2026-11-03 - pktito-qnn-graph 2's six graphs at 88f5a6d

|         |                                                                                                                                                                                                                                                                     |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| machine | the 9950X for the compiles (QNN 2.51.0.260929's x86_64 prepare library); the S24 Ultra for the runs, on its charger, with QNN 2.51's libQnnHtp.so and V75's stub and skel, and no prepare library on the library path, as the app has none, but for its own compile |
| code    | qnn 88f5a6d: the ops in the order the code reads                                                                                                                                                                                                                    |

The six, each compiled from the full file in a process of its own, two at
a time (make clean first, so that made names 88f5a6d):

| model | chip (PKTITO_QNN_TARGET) | bytes       | SHA-256                                                          | compile, wall | peak RSS (GNU time) |
| ----- | ------------------------ | ----------- | ---------------------------------------------------------------- | ------------- | ------------------- |
| v2    | SM8650, 57,75,8,1,0      | 714,506,856 | 3f690d2cdd0ad6dc3b805f627f5c8e0b7701af20135bc275ebcc8f55877263a1 | 64.00 s       | 4.38 GiB            |
| v2    | SM8750, 69,79,8,1,0      | 713,568,872 | 40a5b75f37954b528f04944c57096ba0215783cea23b50ef659a981b5995456c | 61.36 s       | 4.22 GiB            |
| v2    | X1E80100, 60,73,8,1,1    | 714,523,240 | dab2d4d69028caba976c0b82316f83ad49bd3d4bac01b39992cd66904d216513 | 67.81 s       | 4.31 GiB            |
| v3    | SM8650, 57,75,8,1,0      | 714,506,856 | 84aae2af7273132997bb71af73550f1a9a037de35b3d45d9e95a9390a796324e | 63.97 s       | 4.35 GiB            |
| v3    | SM8750, 69,79,8,1,0      | 713,568,872 | d5cf8b7659ea176f66937a94e72c80bb11caac60d18811b0f500b6b1eaa73113 | 61.84 s       | 4.24 GiB            |
| v3    | X1E80100, 60,73,8,1,1    | 714,523,240 | 1208ddc1b2c63e68568d692780297d142ad5f57ee3bb05dd86218347edd7809f | 67.40 s       | 4.35 GiB            |

Each is e3416a8's size for its chip, and the X1E80100's v2 QNN binary is
the one of the section above, 9dfab7f1..., a fourth time.

## The ops in the order the code reads: on the S24

The S24, rebooted at 22:30 for its CDSP (the section before), ran v2's
graph 2 compiled from 88f5a6d on the 9950X (the table's) and on the S24
itself (QNN's aarch64 prepare library on the library path for it: 75 s,
714,629,736 bytes, 122,880 more than the 9950X's), e3416a8's from the
9950X (its gcc build's order, f687a983...), and graph 1, by 88f5a6d's
pktito-c (the NDK r30's clang) and 681bea0's for graph 1: test.wav
checked against the CPU's encoder and profiled, three rounds alternating
the four; chapter 1's bench, best of 5, two rounds alternating; then v3's
graph 2 (88f5a6d, the 9950X's) and graph 1 the same way; then
AliceBench, graph 2's v2 and v3 first and graph 1's after. The battery
went from 32.9 to 39.1 °C over v2's checks and bench, to 41.3 over v3's,
to 43.7 over graph 2's AliceBench and 45.8 over graph 1's.

| model | graph | compiled                       | accelerator ms a chunk, 9 chunks | test.wav's mean cosines      | chapter 1, bench, best of 5, two rounds |
| ----- | ----- | ------------------------------ | -------------------------------- | ---------------------------- | --------------------------------------- |
| v2    | 2     | the 9950X, 88f5a6d             | 110.4 to 111.0                   | 0.999856, 0.999685, 0.999882 | 2.896 s, 3.147 s                        |
| v2    | 2     | the S24, 88f5a6d               | 110.6 to 114.4                   | 0.999856, 0.999685, 0.999882 | 3.038 s, 3.194 s                        |
| v2    | 2     | the 9950X, e3416a8's gcc build | 109.8 to 111.5                   | 0.999856, 0.999685, 0.999882 | 3.095 s, 3.199 s                        |
| v2    | 1     | the 9950X                      | 121.9 to 124.8                   | 0.999864, 0.999699, 0.999877 | 3.363 s, 3.461 s                        |
| v3    | 2     | the 9950X, 88f5a6d             | 110.5 to 113.2                   | 0.999841, 0.998942, 0.999703 | 2.961 s, 3.274 s                        |
| v3    | 1     | the 9950X                      | 122.9 to 125.2                   | 0.999839, 0.998883, 0.999737 | 3.440 s, 3.543 s                        |

The three of v2's graph 2 take the same time to their spreads, and their
checks say the same to every digit pktito prints, the worst frame's
cosine and the largest difference too: neither the order of the ops nor
the host of the prepare library moves the S24's chunk or its numbers.
The 4 ms of the hypothesis was the evening's: graph 1 took 125.6 to
126.9 then (the section before) and 121.9 to 124.8 here. The order the
code reads is kept for what it was made for, a graph that is its
source's whichever compiler builds the pktito that makes it.

AliceBench, a process a chapter (score.py's words against the first
named, and the tokens as written):

| model | graph 2        | graph 1        | graph 2's words not graph 1's | as written | graph 2's not the CPU's | graph 1's not the CPU's |
| ----- | -------------- | -------------- | ----------------------------- | ---------- | ----------------------- | ----------------------- |
| v2    | 64.2 s, 167.1x | 69.6 s, 154.1x | 2 of 27,119                   | 7          | 17 of 27,114            | 19                      |
| v3    | 65.5 s, 163.9x | 72.2 s, 148.6x | 6 of 27,161                   | 20         | 27 of 27,157            | 25                      |

Graph 1's AliceBench ran 2 °C hotter than graph 2's (each ended 2.1 °C
above it); the chunks and the benches above alternated the graphs. Graph 2 takes 10% off the S24's
chunk here, v2's and v3's. Graph 1's run did not hang this time.

## The ops in the order the code reads: on the S25

The S25 the same way, but with no e3416a8 graph (its file gone): v2's
graph 2 compiled from 88f5a6d on the 9950X (40a5b75f...) and on the S25
itself (74 s, on its battery, 713,675,368 bytes, 106,496 more than the
9950X's), and graph 1; v3's graph 2 and graph 1. It ran on its charger,
the battery from 31.9 to 36.0 °C over v2's checks and bench, to 38.2 over
v3's, to 40.5 over graph 2's AliceBench and 41.9 over graph 1's.

| model | graph | compiled           | accelerator ms a chunk, 9 chunks | test.wav's mean cosines      | chapter 1, bench, best of 5, two rounds |
| ----- | ----- | ------------------ | -------------------------------- | ---------------------------- | --------------------------------------- |
| v2    | 2     | the 9950X, 88f5a6d | 94.0 to 97.7                     | 0.999871, 0.999708, 0.999889 | 2.357 s, 2.498 s                        |
| v2    | 2     | the S25, 88f5a6d   | 94.4 to 98.2                     | 0.999871, 0.999708, 0.999889 | 2.452 s, 2.518 s                        |
| v2    | 1     | the 9950X          | 123.5 to 135.5                   | 0.999863, 0.999702, 0.999886 | 3.267 s, 3.378 s                        |
| v3    | 2     | the 9950X, 88f5a6d | 96.1 to 100.2                    | 0.999839, 0.997674, 0.999713 | 2.390 s, 3.443 s                        |
| v3    | 1     | the 9950X          | 125.6 to 139.4                   | 0.999825, 0.997701, 0.999708 | 3.310 s, 3.632 s                        |

As on the S24, the PC's graph and the phone's own take the same time to
their spreads and their checks say the same to every digit, each of
them as e3416a8's graph said on the S25 in the section before. Graph 2
takes 26% off the S25's chunk here, v2's and v3's.

| model | graph 2        | graph 1        | graph 2's words not graph 1's | as written | graph 2's not the CPU's | graph 1's not the CPU's |
| ----- | -------------- | -------------- | ----------------------------- | ---------- | ----------------------- | ----------------------- |
| v2    | 52.0 s, 206.3x | 68.5 s, 156.6x | 3 of 27,119                   | 12         | 20 of 27,114            | 17                      |
| v3    | 60.0 s, 178.9x | 78.3 s, 137.0x | 10 of 27,155                  | 25         | 25 of 27,157            | 29                      |

Graph 2's words against graph 1's are e3416a8's on the S25 above, to
the count. Graph 1's AliceBench ran 1.4 to 2 °C hotter than graph 2's.

## A correction: v3's words against the CPU

The section before's v3 counts against the CPU (on the S25 graph 1's 65
and graph 2's 61 of 27,529, on the Googlebook 48 and 48) counted the
CPU's logs with its transcripts: its folder held each chapter's .err
beside its .txt, score.py scores every file two folders share, and the
logs' times differed as words. From the transcripts alone, the S25's
graph 1 has 29 of 27,157 and graph 2 25, the Googlebook's 24 and 24.
v2's folder held transcripts alone, and its counts stand. So with either
graph v3 strays from the CPU about as far as v2 (17 to 20 of 27,114), not
three times as far.

## The ops in the order the code reads: on the Googlebook

The Googlebook the same way, on its charger: v2's graph 2 compiled from
88f5a6d on the 9950X (dab2d4d6...) and on the Googlebook itself (72 s,
714,646,120 bytes, 122,880 more than the 9950X's), e3416a8's from the
9950X (its gcc build's order, 37679e4c...), and graph 1; then v3's graph
2 from 88f5a6d and from e3416a8 (765f16df...), and graph 1. The battery
went from 29.0 to 31.0 °C over v2's checks and bench, to 32.0 over v3's,
and stayed there.

| model | graph | compiled                       | accelerator ms a chunk, 9 chunks | test.wav's mean cosines      | chapter 1, bench, best of 5, two rounds |
| ----- | ----- | ------------------------------ | -------------------------------- | ---------------------------- | --------------------------------------- |
| v2    | 2     | the 9950X, 88f5a6d             | 170.0 to 171.8                   | 0.999866, 0.999699, 0.999876 | 4.073 s, 4.079 s                        |
| v2    | 2     | the Googlebook, 88f5a6d        | 169.9 to 172.1                   | 0.999866, 0.999699, 0.999876 | 4.060 s, 4.082 s                        |
| v2    | 2     | the 9950X, e3416a8's gcc build | 170.8 to 172.1                   | 0.999866, 0.999699, 0.999876 | 4.066 s, 4.059 s                        |
| v2    | 1     | the 9950X                      | 181.9 to 183.6                   | 0.999865, 0.999685, 0.999885 | 4.317 s, 4.327 s                        |
| v3    | 2     | the 9950X, 88f5a6d             | 169.2 to 172.1                   | 0.999843, 0.998755, 0.999712 | 4.065 s, 4.046 s                        |
| v3    | 2     | the 9950X, e3416a8's gcc build | 169.6 to 171.8                   | 0.999843, 0.998755, 0.999712 | 4.069 s, 4.059 s                        |
| v3    | 1     | the 9950X                      | 178.4 to 182.8                   | 0.999838, 0.998727, 0.999707 | 4.329 s, 4.333 s                        |

As on the phones, v2's three graphs 2 take the same time to their
spreads and check the same to every digit, and so do v3's two. Graph 2
takes 6% off the Googlebook's chunk, v2's and v3's.

| model | graph 2        | graph 1        | graph 2's words not graph 1's | as written | graph 2's not the CPU's | graph 1's not the CPU's |
| ----- | -------------- | -------------- | ----------------------------- | ---------- | ----------------------- | ----------------------- |
| v2    | 73.5 s, 146.1x | 78.1 s, 137.4x | 0 of 27,118                   | 9          | 17 of 27,114            | 17                      |
| v3    | 74.1 s, 144.9x | 78.8 s, 136.3x | 6 of 27,162                   | 18         | 24 of 27,157            | 24                      |

88f5a6d's graph 2 wrote e3416a8's AliceBench transcripts of the section
before to the token, v2's and v3's.

## Two of the hashes above

88f5a6d and 0f8385b, cited above, are 2f99f29 and 3f80cd1 before their
commit messages were corrected (2026-11-04); the trees are the same. The
six graph-2 files on models.pktito.net were compiled at 88f5a6d, and
their headers' made says so.

## Hypothesis: the clock vote for each run, not the transcriber's life

burst() has voted the HTP's clocks up once, in pk_qnn_open before the
graph is read, and never destroyed the vote. From the first open to the
process's end the CDSP is held with DCVS off, sleep disabled and the bus
and core at their highest corners, through an app's idle time between
transcriptions and after a failed open (libpktito's review, 2026-11-04).
The values are QAIRT 2.51's HTP backend guide's burst profile: its sample
app's vote (DCVS off, sleep disabled at a sleep latency of 40 us, the
highest corners) and qnn-net-run's defaults for --perf_profile burst (an
RPC control latency of 100 us, polling for 9999 us). None was measured
here. The guide's HIGH_PERFORMANCE profile votes up for each run and
falls to a relaxed state between runs.

To be measured on the S24 with graph 2's v2 SM8650 file: chapter 1's
bench (best of 5) and the process's time, two rounds alternating:

- open: the vote cast before the graph is read and held to the free, as
  burst() has it (now destroyed at the free);
- run: cast before the read and before each graphExecute, and lowered
  after each to the guide's released state (DCVS on, a 65535 us sleep
  latency, the lowest corners, no polling);
- destroy: cast the same way, the power config destroyed after each;
- off: no vote.

Expected: run and destroy within open's spread, so the vote need not
outlast a run, and off slower by DCVS's ramp on each chunk.

## The clock vote, measured

2026-11-04, with graph 2's v2 file for each chip (models.pktito.net's,
compiled at 88f5a6d) and the CLI with the review's fixes, the modes above
in a trial build of their own; chapter 1 of Alice (640.5 s, 23 chunks),
each phone on its charger.

On the S24, the two rounds in the order above ran from 29.6 to 35.4 °C
and came out in that order, each mode slower than the one before it
(open 2.833 and 3.042 s, best of 5; off 3.016 and 3.119). Four more
rounds followed, from 35.0 to 43.8 °C, the order reversed every other
round, with nopoll (run's vote, polling 0) beside the rest:

| vote    | best of 5, s, rounds 3 to 6 | mean  | system s     |
| ------- | --------------------------- | ----- | ------------ |
| open    | 3.197, 3.168, 3.302, 3.348  | 3.254 | 1.89 to 2.15 |
| run     | 3.156, 3.278, 3.366, 3.391  | 3.298 | 2.18 to 2.43 |
| destroy | 3.096, 3.248, 3.285, 3.370  | 3.250 | 1.97 to 2.35 |
| nopoll  | 3.046, 3.330, 3.356, 3.417  | 3.287 | 0.87 to 1.65 |
| off     | 2.869, 3.225, 3.267, 3.340  | 3.175 | 0.63 to 0.92 |

Profiled (PKTITO_QNN_PROFILE=1) on test.wav's three chunks, three
rounds alternating off, run, open and off again, at 41.8 °C: the
accelerator's time (QNN's event 3004) was 110.7 to 113.5 ms a run in
every mode. The first run's HVX and HMX power-up (3011) was 8.4 to
8.6 ms without a vote and 3.9 to 4.9 ms with one.

The Googlebook, profiled the same way at 28 to 32 °C: the accelerator's
time was 503.8 to 552.0 ms a run without a vote (27 runs) and 168.9 to
171.7 ms with any of them (54). Its bench, best of 5, in rounds whose
order turned each time, ran beside f, the vote to be kept (DCVS's config
alone, up for each run and down after):

| vote                           | best of 5, s                      | system s         |
| ------------------------------ | --------------------------------- | ---------------- |
| f                              | 4.158, 4.158, 4.158, 4.155, 4.169 | 0.97 to 1.55     |
| run                            | 4.152, 4.164                      | 2.82, 2.10       |
| nopoll                         | 4.191, 4.191                      | 1.96, 1.32       |
| open                           | 4.050, 4.057, 4.046               | 0.90, 1.55, 0.87 |
| opennp: open's vote, polling 0 | 4.070, 4.055, 4.060               | 0.93, 0.93, 1.48 |
| off                            | 12.354, 12.343                    | 1.11, 1.71       |

open is the vote burst() cast, now destroyed at the free; on 11-03 it
took the same graph 4.073 and 4.079 s (the Googlebook's section above).
f, timing its own vote calls in another trial build, took 4.165, 4.154
and 4.153 s: 230 calls a bench, 46 a pass, in 212.3 to 215.2 ms, 0.93 ms
each. With the graph's read timed as well (test.wav, three rounds), a
vote for the read read it in 0.752 to 0.759 s, and no vote in 0.771 to
0.796.

So, without the vote, the Googlebook's accelerator took three times as
long over each run: its DCVS left the clocks low through a run, where
the S24's had them up (its runs took the same with a vote and without).
Neither the latency nor the polling made a run faster on either phone.
A run takes over 100 ms, and the polling waits 9999 us at most. What
the polling cost in system time:

- on the S24, 0.6 to 1.6 s a bench (run, open and destroy against nopoll
  and off, round by round);
- on the Googlebook, 0.6 and 1.4 s in run's two rounds against f's,
  and nothing to its spread in open's against opennp's.

Held from the open, the vote took the Googlebook 0.10 to 0.11 s less
than f (open and opennp against f, 2.4 to 2.6%). f's own calls are
0.04 s of that, and the rest is in the runs. On the S24, open and run
differ by less than a round's spread. An app keeps its transcriber for
the process's life (pktito-app's Transcriber.model: "loaded once for the
process"), so a vote held to the free would keep the DSP from sleeping
through all of the app's idle time. f lets the vote go after each run.

Taken: f, as core/qnn.c's vote(). Not taken: a vote for the read, which
saves 0.03 s of the Googlebook's 0.78.

Profiling slows each run by more than the vote does. Profiled, f's bench
took the Googlebook 11.32 s, best of 5. The time QNN gives the HTP's
side of each call (3002) was then 382 to 492 ms after the first run,
where the accelerator's was 164.1 to 171.6 ms. A profile's times are for
comparing with other profiles.

Not run: f's bench on the S24, which left adb before it (its wireless
debugging port refused connections). The nearest is run: f's vote with
the latency and the polling added.

## A header's ids and names, on the NPU

2026-11-04, on the S24, with graph 2's v2 SM8650 file and the CLI at
d2ccda1 (b2dac6c's header checks). The full model ran test.wav with the
file's header patched in turn. Each run was checked against the CPU's
encoder (PKTITO_QNN_CHECK=1), with QNN's log on (PKTITO_QNN_LOG=1).
check_head holds the names and the ids to being there and all different,
and nothing more, so each patched header passes it, and QNN has the ids:

| header                         | QNN                                                           | words                       |
| ------------------------------ | ------------------------------------------------------------- | --------------------------- |
| as compiled (ids 1 to 5, 3398) | ran each chunk                                                | the NPU's, 3 chunks checked |
| input 0's id 99                | "Expected Tensor ID: 1 not found in user-provided tensors"    | the CPU's                   |
| inputs 1's and 2's ids swapped | "Tensor ID 2 dim 1 mismatch. 1536 != 768"                     | the CPU's                   |
| inputs 3's and 4's ids swapped | "Tensor ID 4 batch (dim[0]) incompatible. (1, 384)"           | the CPU's                   |
| the output's id 99             | "Expected Tensor ID: 3398 not found in user-provided tensors" | the CPU's                   |
| every name another (x0 to x5)  | ran each chunk                                                | the NPU's, 3 chunks checked |

QNN holds each run's tensors to the binary's own ids and shapes, and
refuses any run where they differ. It did so at every chunk, and the
full model's encoder ran each chunk instead, as core/encoder.c has it,
so the transcript was the CPU's. QNN matches nothing by the names.

The slim model, with the binary's first byte (offset 616) set from 0 to
0xFF: QNN read the binary. With its first 32 bytes set to 0xFF, QNN
refused it at the read: "Serializer failed to get context blob", error 30001.

## Hypothesis: the turn's tables from vmath, for graph 3

tables() makes the position tables a graph holds from libm's sin and cos
of j w, in double, each rounded once to float. CLAUDE.md has core/vmath.h's
functions as the only transcendentals that reach a transcript, because
libm's are not the same bits on every platform. core/encoder.c takes the
CPU's angles as float32(pos \* freq) and gives them to pk_sincosf
(libpktito's review, 2026-11-04). With tables made through pk_sincosf
from the CPU's angles, a graph compiled on any PC would hold the same
tables, and they would be the CPU's.

The tables are a graph's constants, so the change waits for the next
change to the graph: graph 3, a new "pktito-qnn-graph" format, and a new
compile for every chip, as graph 2 had. To be measured then:

- the tables' bits against graph 2's: the entries that differ, and by how
  much;
- test.wav's cosines against the CPU's encoder;
- AliceBench's words against the CPU's, on the S24 (graph 2: 17 to 20 of
  27,114 on v2).

Expected: the float tables differ from graph 2's wherever float32(j w)
rounds j w, by up to 1.5e-5 at j = 383, and at far fewer entries in the
NPU's fp16. No word should move by more than graph 2's own spread across
the three chips.

## QNN's timeout signal, on the Googlebook

2026-11-04, on the Googlebook, with graph 2's v2 file, the full model
and test.wav's three chunks. The question: could a time limit on
graphExecute end a hang like the S24's in chapter 5 ("The depthwise wide
and P13: kept", above)? There, the process waited in
fastrpc_wait_for_completion, uninterruptible, until the phone was
rebooted. A trial build of bd53ef8 passed each graphExecute a QnnSignal
made with QNN_SIGNAL_CONFIG_OPTION_TIMEOUT, at the given time a run:

| timeout                                         | graphExecute returned                                              | words                        |
| ----------------------------------------------- | ------------------------------------------------------------------ | ---------------------------- |
| none                                            | 0, in 172.9 to 180.1 ms                                            | the NPU's                    |
| 1 s, 200 ms                                     | 0, in 173.2 to 180.5 ms                                            | the NPU's                    |
| 150 ms                                          | QNN_GRAPH_ERROR_TIMED_OUT (6033), in 154.2 to 164.9 ms             | the CPU's                    |
| 100 ms                                          | 6033, in 103.7 to 114.9 ms                                         | the CPU's                    |
| 50 ms                                           | 6033, in 53.5 to 62.1 ms                                           | the CPU's                    |
| 50 ms on one chunk of three, none on the others | 6033 on that chunk; the others 0, in 172.9 to 180.1 ms             | that chunk the CPU encoder's |
| 10 us                                           | 6033 ("Graph timed out. Timeout value (us) 10"), in 2.0 to 10.8 ms | the CPU's                    |

A run started at once after a timed-out one, with no signal, took 172.6
to 176.1 ms, the normal time. So the DSP stops the aborted graph, 4 to
15 ms after the deadline, and the context goes on as before. A signal
that doesn't fire costs nothing measurable.

The thread that calls graphExecute waits in the kernel itself. Sampled
through chapter 1's bench, the main thread was in
fastrpc_wait_for_response in 20 of 20 samples, and QNN's own threads
waited in poll, futex and fastrpc too. So the timeout's return depends
on the DSP answering the abort, and it would likely not free a thread
stuck as the S24's was. That can't be tried without a hang.

Not taken: a time limit in the engine. It can only bound an NPU that
still answers. In the app's slim edition, which has no encoder of its
own to go on with, a run that times out fails the transcription, and the
app then takes the NPU as refused for that build over a passing slowdown
(libpktito, 2026-11-04). A hang is the app's to watch for.

# 2026-11-04 - the Galaxy S26 Ultra's V81

|         |                                                                                                                                                                                                      |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| machine | the 9950X for the compiles (QNN 2.51.0.260929's x86_64 prepare library); the Galaxy S26 Ultra (SM-S948U, Android 16, ro.soc.model SM8850) in Samsung's Remote Test Lab for the runs, through the app |
| code    | qnn 88f5a6d for the graphs, as the app's other six; pktito-app's android-s26 979cf09                                                                                                                 |

The S26's QNN describes its chip as PKTITO_QNN_TARGET=87,81,8,1,0: SoC
model 87, which is the SM8850 in QAIRT 2.51's SoC table, V81, 8 MB of
VTCM (the S25's), DLBC, and no signed PD. Its two graphs, compiled for
that chip on the 9950X from the full file, two at a time:

| model | bytes       | SHA-256                                                          | compile, wall | peak RSS (GNU time) |
| ----- | ----------- | ---------------------------------------------------------------- | ------------- | ------------------- |
| v2    | 714,412,648 | 08388c7f4b9cbb562330111ed6e0b026d00512168cee589b04578f50673b1aff | 95.12 s       | 4.32 GiB            |
| v3    | 714,412,648 | 68eb09b667c7f9f6f36bb7eb7812945078150d8ff9984c61ce169d1bdfe99eab | 95.12 s       | 4.33 GiB            |

In the PC's x86 emulation they give test.wav the transcript that the
SM8650's, SM8750's and X1E80100's graphs give. Through the app, each
time from the decoded audio to the last segment:

| model | graph opened   | chapter 1 (640.5 s)                                                | AliceBench as one file (10731 s) |
| ----- | -------------- | ------------------------------------------------------------------ | -------------------------------- |
| v2    | 0.62 s, 0.83 s | 2.0 to 2.1 s in three runs, the first 2.038 s (314x); 111 segments | 37.3 s (288x), 1709 segments     |
| v3    | 0.53 s         | 2.074 s (309x); 115 segments                                       | 38.2 s (281x), 1904 segments     |

The file is alice-wonderland-ch01-12.flac, AliceBench's twelve chapters
joined (176,226,818 bytes, 10731.43 s). In chapter 1, v2's words are the
S25's and v3's the Googlebook's; the book's were compared with no other
device. The battery was at 30.0 °C before v2's book run and 35.5 °C
after.

## An x86 build compiles for another chip again

2026-11-05. beb62a8 refused PKTITO_QNN_TARGET where QNN describes an NPU
of its own, for a phone told to compile for another chip. QNN 2.51's x86
HTP library describes the device it emulates, 30,68,4,0,0, so on the
9950X 3adfae8's pktito-c (QNN=1) refused every compile, in 0.08 s ("...
and this one has an NPU"), and the graphs above came from 88f5a6d, from
before the check. With the check left out of an x86 build (core/qnn.c,
open_device), 3adfae8's code compiled the S26's two graphs on the 9950X,
one at a time, in 88.22 and 89.16 s (4.36 and 4.41 GiB). Each is 714,412,648
bytes, and QNN's binary in each (the file less its header) is the 88f5a6d
build's byte for byte: v2 8172f441fbec5a17..., v3 a31c73589cc8f27e.... The
headers differ in the pktito commit alone, and test.wav's transcripts are
the same.

## Windows on Snapdragon: the Googlebook's graph, unchanged

2026-11-05, snappy (Snapdragon X Elite X1E-78-100, Hexagon V73, Windows
11 build 26300, NPU driver 30.0.220.3000, on mains power). pktito-c built
by llvm-mingw 20260922 for aarch64-w64-mingw32 with QNN=1 against QNN
2.51's aarch64-windows-msvc libraries: QnnHtp.dll loaded by LoadLibraryExA
from the exe's folder, beside QnnSystem.dll, QnnHtpV73Stub.dll and
hexagon-v73's libQnnHtpV73Skel.so and libqnnhtpv73.cat. The graph is the
X1E80100 one made for the Googlebook (v2, 714,523,240 bytes, sha256
dab2d4d69028caba...), with the v2 slim file; it opened on Windows as it
is, run over SSH.

Chapter 1 (640.5 s), runs taken in turn, the encoder on the NPU against
the CPU's neon-dp with 12 threads (the times pktito-c -v gives, VAD to
last segment):

|     | run 1           | run 2           | CPU time of the whole process |
| --- | --------------- | --------------- | ----------------------------- |
| NPU | 4.02 s (159.5x) | 4.14 s (154.9x) | 2.41 s in 5.19 s              |
| CPU | 5.59 s (114.5x) | 4.78 s (134.0x) | 52.92 s in 4.94 s             |

The NPU's transcript is the same in both runs, and so is the CPU's; of
their 2,194 words one differs, "Why," for "Why". The NPU is 1.2 to 1.4
times the CPU's speed here, and takes a twenty-second of its CPU time.
Not measured: power on battery, v3, AliceBench, the Windows app.

## Windows on Snapdragon: the X2 Elite's two graphs, untried on one

2026-11-05, the 9950X. QAIRT 2.51's SoC table gives the X2 Elite Extreme
(SC8480XP) as SoC model 88, V81, and Windows its toolchain; ExecuTorch's
SoC table (backends/qualcomm/serialization/qc_schema.py) gives it 8 MB of
VTCM, and the X1E80100 (SC8380XP) 60, V73, 8 MB, which is what snappy's
QNN describes. DLBC and the signed PD are snappy's, 1 and 1. No X2 has run
either graph. Compiled with PKTITO_QNN_TARGET=88,81,8,1,1 from the full
file by bbdcf60's x86 pktito-c (gcc 15.2.0, QNN=1), one at a time:

| model | bytes       | SHA-256                                                          | compile, wall | peak RSS (GNU time) |
| ----- | ----------- | ---------------------------------------------------------------- | ------------- | ------------------- |
| v2    | 714,445,416 | 726db8ecde7934510ab0193f52de93601c64506edf1804a240612a823e90d811 | 87.82 s       | 4.32 GiB            |
| v3    | 714,445,416 | 237a5a6c51abfcb969310d494f70b2663c568a43174b70b9fd3f6d624c6caa7a | 87.53 s       | 4.34 GiB            |

The same build compiled the X1E80100's v2 graph (60,73,8,1,1) again
first, in 68.83 s: QNN's binary in it (the file less its 616-byte header)
is the catalog's (dab2d4d69028caba...) byte for byte. In the PC's x86
emulation the X2's v2 graph gives test.wav the X1E80100's graph's
transcript.

## Windows on Snapdragon: where QNN's files go, and the CPU's i8mm

2026-11-08, snappy (Snapdragon X Elite X1E-78-100, Hexagon V73, Windows
11 build 26300.9818, on mains power). The npu branch's pktito-c and
examples, built in MSYS2's CLANGARM64 shell (clang 23.1.5) with QNN=1
against QAIRT 2.51.0.260929; the catalog's X1E80100 v2 graph
(dab2d4d69028caba...) and the v2 slim file, over SSH. examples/transcribe
(static) ran from a folder of its own, the current folder another, with
ADSP_LIBRARY_PATH unset but where a row sets it; QAIRT's files from
lib/aarch64-windows-msvc and lib/hexagon-v73/unsigned:

| beside the program                           | elsewhere                                      | test.wav                                                               |
| -------------------------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------- |
| QnnHtp.dll, QnnSystem.dll, QnnHtpV73Stub.dll |                                                | refused: "Can't read future blob. Newest blob version supported: 3.3." |
| those, libQnnHtpV73Skel.so, libqnnhtpv73.cat |                                                | the CLI's transcript                                                   |
| QnnHtp.dll, QnnSystem.dll, QnnHtpV73Stub.dll | the skel and .cat in the current folder        | the CLI's transcript                                                   |
| QnnHtp.dll, QnnSystem.dll, QnnHtpV73Stub.dll | ADSP_LIBRARY_PATH QAIRT's hexagon-v73/unsigned | the CLI's transcript                                                   |
| all five but the .cat                        |                                                | refused, as without the skel                                           |
| all five but QnnSystem.dll                   |                                                | the CLI's transcript                                                   |

So pktito needs four of the five, and without 2.51's skel and its .cat
where the DSP's loader looks, the DSP runs its driver's own older QNN,
which can't read 2.51's graph. Python 3.15.2 and Node 26.13.0, loading
libpktito.dll with QNN's four beside it, the current folder another and
nothing on PATH, gave the transcript too: the skel is found beside
QnnHtp.dll and its stub, wherever the program is.

The `DSP_INFO UNSUPPORTED_KEY: 49` (50, 51) lines on stderr, three a call
and two calls a run, are the driver's: the string is in its HTP folder's
QnnSystemInfo.dll, which QnnHtp.dll loads to learn the chip
("SoCServiceUtils/QnnSystemInfo not loaded" is its message without it),
and in none of QAIRT's files. The chip it learns names no SoC,
2147483647,73,8,1,1, as on 11-05.

With the CPU's backend neon-i8mm now, where 11-05's was neon-dp, chapter 1
(640.5 s), runs taken in turn, the times pktito-c -v gives, and the
process's CPU time as PowerShell's TotalProcessorTime gives it:

|     | run 1           | run 2           | CPU time of the whole process        |
| --- | --------------- | --------------- | ------------------------------------ |
| NPU | 4.16 s (154.1x) | 4.21 s (152.3x) | 2.22 s in 5.33 s; 2.44 s in 5.54 s   |
| CPU | 4.66 s (137.4x) | 4.74 s (135.0x) | 50.14 s in 4.81 s; 51.14 s in 4.90 s |

The transcripts are 11-05's: each the same as its kind's other run, and of
their 2,194 words one differs, "Why" on the NPU for the CPU's "Why,". The
NPU is now 1.1 times the CPU's speed, from 1.2 to 1.4, and takes a
twenty-first to a twenty-third of its CPU time. An NPU run spends 1.2 to
1.3 s before the VAD starts, most of it the graph's load, a CPU run 0.15 s.
test.wav's minute: 0.52 s on the NPU and 0.55 s on the CPU by pktito-c -v,
and by bench's best of 5, 0.507 s and 0.361 s.
