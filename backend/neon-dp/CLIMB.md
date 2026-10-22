# backend/neon-dp: the climb

Dated records, appended; rejected hypotheses stay. A row's numbers are true
of the board, compiler and tree it names.

## 2026-10-20 - the baseline, on an M4 Pro (no Pi 5 up yet)

Apple M4 Pro, 8 performance and 4 efficiency cores, macOS 27, Apple clang
21.0, `-mcpu=apple-m1 -O3 -ffp-contract=off`. `./pkito bench` on Silero's
test.wav (60 s) whole, `--max-chunk 60`: 750 encoder frames, 232 tokens,
best of five.

| tree | threads | best | real time |
|---|---|---|---|
| 3bbbf3b, first SDOT kernel (8x4 tile, no column blocking) | 12 | 0.543 s | 110.6x |
| 55e6d97, column blocks of 192 KB | 4 | 1.008 s | 59.5x |
| | 8 | 0.512 s | 117.3x |
| | 12 | 0.523 s | 114.7x |

Column blocking moved nothing here (0.543 s at 12 threads before, 0.523 s
after, inside the spread): the M4's 16 MB L2 per cluster holds a 750-column
product's activations (3 MB at 4096 inputs) whole. It went in for the
12900HK and the A53, whose L2s do not (backend/avx2/CLIMB.md,
backend/neon/CLIMB.md). The efficiency cores add nothing past eight
threads: dynamic dealing gives them items, and they finish them late.

The stage split at eight threads (`make PROFILE=1`, 0.509 s):

| stage | ms | share |
|---|---|---|
| feed-forward (4 int8 products a layer, SiLU) | 234.1 | 44.3% |
| attention, float (scores, positions, softmax, context) | 114.3 | 21.7% |
| q k v and the positions' projection | 71.5 | 13.5% |
| convolution module's two products | 41.7 | 7.9% |
| decoder | 16.2 | 3.1% |
| attention output | 14.7 | 2.8% |
| subsampling | 13.0 | 2.5% |
| mel | 7.2 | 1.4% |
| quantize, layer norms, GLU and depthwise, positions | 15.3 | 2.9% |

The feed-forward is 302 G multiply-adds in 234 ms: 1.29 T a second over
eight cores, 161 G a core, about 56% of a P core's SDOT peak (four 128-bit
pipes at ~4.5 GHz, 16 multiply-adds an SDOT: 288 G). The float attention is
42 G fused multiply-adds in 114 ms, 46 G a core, about 64% of FMLA's peak
(72 G). So the int8 path has the more room, and the attention, at a fifth
of the time at 750 frames, is the largest stage that is not int8.

Open for the A76, once a Pi 5 is up: the tile (8 rows x 4 columns, 16
int32 and 8 float accumulators) was shaped for the M4's 32 registers and
four pipes and has not been measured against a 16x2 or 8x3 on a core with
two; PK_Q8_BLOCK_BYTES (192 KB) against the A76's 512 KB L2 and a 2 MB L3
shared by four; and whether the float attention's share on the A76 makes an
int8 attention worth its rounding.

## 2026-10-20 - the Pi 5 baseline

Raspberry Pi 5, 4 x Cortex-A76 at 2.4 GHz (performance governor, 60 C under
load, never throttled), 4 GB, Debian, gcc 14.2, `-march=armv8.2-a+dotprod
-mtune=cortex-a76 -O3 -ffp-contract=off`. Tree 02f692f plus the profile's
repack split. check-c passes: every kernel ref's bits, both transcripts of
test.wav byte for byte. Four threads.

| test.wav | best of 5 | real time |
|---|---|---|
| whole (750 frames) | 4.934 s | 12.2x |
| 30 s chunks (3) | 4.371 s | 13.7x, and 0.069 s of VAD |

The stage split, whole (`make PROFILE=1`):

| stage | ms | share |
|---|---|---|
| feed-forward | 1969 | 39.8% |
| attention products (scores, positions, softmax, context) | 891 | 18.0% |
| q k v and the positions' projection | 544 | 11.0% |
| conv pw1 pw2 | 338 | 6.8% |
| subsampling | 330 | 6.7% |
| attention repack | 285 | 5.8% |
| quantize | 163 | 3.3% |
| decoder | 128 | 2.6% |
| attention out | 122 | 2.5% |
| glu + depthwise | 80 | 1.6% |
| layer norms | 77 | 1.6% |
| mel, positions | 26 | 0.5% |

The feed-forward is 302 G multiply-adds in 1.97 s: 38 G a core, 16 a cycle,
half of SDOT's 32 (two 128-bit pipes). The attention's products run at
about 3.7 FMAs a cycle of FMLA's 8. Three things stand out against the M4,
whose split this resembles otherwise:

* the repack is 5.8%, for moving 18 MB a layer: K's and P's panels are a
  transpose, written a float at a time with each store a cache line apart
  (core/encoder.c, repack_item);
* the subsampling is 6.7%, where its arithmetic (7 G int8 multiply-adds and
  0.3 G float) should take about 80 ms: its 3x3 convolutions run channel
  group outermost, so a block's 2.3 MB of output is walked 64 times;
* quantize is 3.3%, for one pass over each product's input.

Each is a kernel or a loop order; none changes a bit.

## 2026-10-20 - the attention's repack

Same Pi 5, compiler and flags; four threads, best of five, check-c passing
for each row (every transcript byte for byte, so none changes a bit).

| tree | whole | 30 s chunks | repack | attention |
|---|---|---|---|---|
| fde7b8b, the baseline above | 4.934 s | 4.371 s | 285 ms | 891 ms |
| K's, V's and P's panels written a dimension's 16 keys at a time | 4.863 s | | 192 ms | |
| no repack: V into panels by its product, K and P as written, each block's scores and position products transposed instead (rejected) | 5.064 s | 4.221 s | 0 | 1293 ms |
| V into panels by its product; K and P transposed 16 keys at a time; the queries plus biases made a block at a time in att_item | 4.741 s | 4.169 s | 103 ms | 867 ms |

The rejected row moved the transposes from K and P (T x dk a head) to the
scores (T x 16 a block, T/16 blocks a head: T squared over 16 a head), and
its st and bt buffers, 48 KB each at 750 frames, left L1 for every block.
It won at 30 s chunks, where T is a third, and lost whole. The last row
keeps what both got right: V's product writes the panel layout the context
product reads (one pk_k_gemm_q8 a 16-row panel, columns 16 floats apart),
and the two transposes that remain are written as runs. The q k v stage
did not move (544 ms either way), so writing V's panels in place is free.
Memory fell by two [T][D] arrays (qu and qv, now 2 x 64 x dk a thread).

Now (whole, `make PROFILE=1`, 4.754 s): feed-forward 1976 ms (41.5%),
attention 867 (18.2%), q k v pos 544 (11.4%), subsampling 343 (7.2%), conv
pw1 pw2 338 (7.1%), quantize 161 (3.4%), decoder 126 (2.6%), attention out
123 (2.6%), repack 103 (2.2%). Next: the subsampling's loop order.

## 2026-10-20 - the subsampling

Same Pi 5, compiler and flags; four threads, best of five, check-c passing
for each row.

| tree | whole | 30 s chunks | subsampling |
|---|---|---|---|
| b55ce56 | 4.741 s | 4.169 s | 343 ms |
| pk_k_conv3x3s2 a row at a time, 4 channels by 8 columns at once | 4.601 s | 4.026 s | 197 ms |
| and the first stage made a window at a time, the 3 rows the second's next output row reads | 4.537 s | | 115 ms |
| windows for 2, 3 and 4 output rows instead (rejected) | 4.547, 4.567, 4.585 s | | 129, 150, 165 ms |
| every stage a row at a time, each keeping the 3 rows the next reads | 4.505 s | 3.920 s | 90 ms |

An instrumented build (not committed) timed each stage's calls inside the
items, summed over the threads, a run:

| ms of thread time | conv0 | dw1 | pw1 | dw2 | pw2 |
|---|---|---|---|---|---|
| b55ce56, 4 threads | 510 | 562 | 130 | 64 | 32 |
| the new kernel, 4 threads | 229 | 261 | 141 | 39 | 32 |
| the new kernel, 1 thread | 79 | 69 | 92 | 12 | 22 |
| first stage in windows, 4 threads | 82 | 78 | 142 | 30 | 28 |
| first stage in windows, 1 thread | 70 | 44 | 91 | 11 | 22 |

The old kernel ran a channel group outermost over the block: one chain of
nine FMAs at a time, four cycles a link, and the block's whole input (2.3
MB for dw1) read again for every group. The new one keeps eight chains in
flight and works a row at a time, which more than halved both
convolutions. Then the one-thread row showed what was left: the same
stages took two to three times as long on four threads as on one. Each
thread's block of the first stage was 2.3 MB and of the next two 557 KB,
none of which stays in a 512 KB L2, so four threads were waiting on
memory. Made a row at a time, each stage keeps only the three rows the
next reads (192 KB for the first, 96 KB for the third) and moves the one
it still needs down the window. Windows wider than one output row lost
each time they grew. The thread buffers fell from about 3.5 MB to 0.6 MB
a thread.

The M4 Pro (eight threads) went from 12.6 to 10.2 ms in this stage, and
the 12900HK (avx2-vnni, six threads) from 19.5 to 17.5 ms; neither L2
was overflowing, so only the kernel's chains helped there.

## 2026-10-20 - the depthwise convolution, and the column blocks

Same Pi 5. pk_k_dwconv (the convolution module's, nine taps over time)
now takes 64 frames and 64 channels at a time and, in that, four channels
by eight frames at once, eight chains of FMAs where there was one, with
the taps tap-major on the stack instead of gathered at every tap. Its
stage (with the GLU) went from 80 to 72 ms and the minute from 4.505 to
4.496 s: the stage reads and writes three [T][D] arrays of 3 MB a layer,
none of which stays in cache, so it waits on memory more than on FMAs.

PK_Q8_BLOCK_BYTES, the activations an int8 product holds per column
block, measured on the whole minute: 64 KB 4.507 s, 96 KB 4.502, 128 KB
4.497, 192 KB 4.500 and 4.512, 256 KB 4.494. Nothing outside the spread,
so it stays at 192 KB. A 64-row slice's weights stay in L2 at every size
here, and the activations stream past them either way.

q8bench on one A76 (`./tests/q8bench 750 5 64`): the feed-forward up
product 45.4 GMAC/s, down 40.4, the attention's 45.5: 18.9 multiply-adds
a cycle at 2.4 GHz. Every accumulator's block of 32 inputs is eight SDOTs
and three more vector instructions on the same two pipes (the zeroing,
SCVTF, and the FMA into the float sum), so 23.3 a cycle is this format's
ceiling on the A76; the kernel is at 81% of it alone, and at about 16 a
cycle with four threads in the encoder.

## 2026-10-20 - the attention's float products: gcc's first scheduling pass

Same Pi 5. The attention stage split by an instrumented build (ms of
thread time a run, four threads; one thread gives the same within 2%, so
none of it waits on memory): scores 1008, position products 1055, context
1015, softmax with the position add 348, queries plus biases 72. The
three products are about 14 G FMA lanes each, so 5.7 lanes a cycle of
FMLA's 8.

gcc 14.2 spills in f32_tile: its scheduling before register allocation
(-fschedule-insns, on at -O3 with -fsched-pressure) hoists all sixteen
weight vectors of a step ahead of the sixteen accumulators, and the 4x4
tile's loop stores and reloads them. Built without that pass:

| build | feed-forward | attention | minute |
|---|---|---|---|
| -O3 | 1977 ms | 870 ms | 4.501-4.510 s |
| -O3 -fno-schedule-insns, everything | 2270 ms | 728 ms | 4.812-4.819 s |
| -O3, the inputs two at a time in f32_tile (no spill, but register moves at every back edge; rejected) | 1975 ms | 921 ms | 4.555 s |
| -O3, the pass off for the float products only (pragma) | 1978 ms | 728 ms | 4.358 s |

The int8 tile needs the pass and the float tile is hurt by it, so a
gcc-only pragma turns it off around the float products: their 4x4 loop is
then 64 FMLAs and 20 loads with nothing else, about 86% of FMLA's peak.
The VAD's products are among them (0.069 -> 0.059 s for test.wav's). 30 s
chunks 3.920 -> 3.847 s. The one-column tile (MR 16, the VAD's step)
still keeps its accumulators on the stack; it is a small part of a small
stage.

## 2026-10-20 - the feed-forward's SiLU folded into its quantize

Same Pi 5. Quantize does not scale with threads: 235 ms on one, 145 on
two, 163 on four, where the feed-forward's int8 products scale 3.77 times
on four (7.44 s, 3.73, 1.98). Most of what quantize reads is the
feed-forward's hidden activations, [T][4096] floats, 12 MB at a minute,
which a SiLU pass first read and wrote back in place. Now one pass takes
a frame at a time through a 16 KB row of thread scratch, SiLU then
quantize, and the array is read once and never written:

| | feed-forward | quantize | minute | 30 s chunks |
|---|---|---|---|---|
| d6afe66 | 1978 ms | 163 ms | 4.358 s | 3.847 s |
| SiLU in the quantize pass | 1827 ms | 213 ms | 4.244-4.248 s | 3.791 s |

(The SiLU's arithmetic moved from the feed-forward's row of the profile
to quantize's.) The 12900HK and the M4 Pro did not move, inside their
spreads: their caches and memory keep up either way.

## 2026-10-20 - the layer norms into their quantize

Same Pi 5. Four of a layer's five layer norms (both feed-forwards', the
attention's, the convolution module's) wrote [T][1024] floats, 3 MB at a
minute, that only the quantize after them read. Each now goes a frame at
a time through the same row of thread scratch as the SiLU, norm then
quantize; the last norm, whose output is the residual stream, stays.

| | layer norms | quantize | minute | 30 s chunks |
|---|---|---|---|---|
| dfec31c | 77 ms | 213 ms | 4.244-4.248 s | 3.791 s |
| norms in the quantize passes | 13 ms | 219 ms | 4.186-4.189 s | 3.737 s |

(The four norms' arithmetic is in quantize's row now.) The 12900HK and
the M4 Pro did not move outside their spreads.

## 2026-10-20 - backend/neon's A53 climb merged, and a block of 32 unrolled

The A53 climb (backend/neon/CLIMB.md) changed the int8 product's shared
code: the column-block loop moved into q8_cols, and q8_tile took a range
of blocks and a partial-sum buffer for ARMv8.0's chunks of 1024 inputs.
For SDOT the tile stays 8 x 4, the column blocks 192 KB, and the chunks are
compiled out. Same Pi 5, merged tree:

| | minute | 30 s chunks |
|---|---|---|
| 9793aab | 4.186-4.189 s | 3.737 s |
| merged | 4.155-4.162 s | 3.714-3.718 s |

gemmbench, one A76 at 2.4 GHz, T 750: feed-forward up 19.2 MAC/cycle, down
17.6, the attention's 19.0, conv pw1 19.1, all four 18.5; the L1-resident
case 18.9.

Rejected, the encoder's blocks of 32 inputs unrolled (q8_tile told at
compile time that qb is 32, the main tiles only): before the merge, with
q8bench, feed-forward down 40.4 -> 43.7 GMAC/s and the others 45.4 ->
44.9; on the merged tree, with gemmbench, all four shapes 44.5 -> 43.1
GMAC/s and the L1-resident case 45.4 -> 42.6. The tile stays as it is.

## 2026-10-20 - every core of the M4 Pro

The thread default is now one a physical core of every kind
(backend/avx2/CLIMB.md has the 12900HK's and a Core 3 304's case). The
M4 Pro, interleaved, three passes, the minute whole: 8 threads (its
performance cores) 0.498, 0.500, 0.509 s; 10 threads 0.495, 0.497, 0.513;
12 (every core) 0.488, 0.489, 0.495. The efficiency cores add about 2%
now, where on 55e6d97 they took 0.523 s against 0.512. The Pi 5's four
cores are one kind, so its default does not move.

## 2026-10-20 - one binary for the family: the backend picked at run time

neon-dp and neon in one build, each with its own instruction set, the
tree around them at ARMv8.0 (with the board's -mtune); backend/kernels.c
runs the first the CPU can. The Pi 5 picks neon-dp: 4.161-4.163 s whole,
3.709-3.717 s in 30 s chunks (4.150 and 3.710 before). The Pi 3B+'s build,
copied to the Pi 5 as it was, picks neon-dp too: 4.116 s, and the v3
transcript of test.wav byte for byte; the M4 Pro's 0.484 s, and
`PKITO_KERNELS=neon` 1.052 s (1.047 as a build of its own).

## 2026-10-20 - pkito is now pktito

The project took its intended name, pktito (a little parakeet): the
binary, library, header, converter and PKTITO_KERNELS. The records above
keep the name they were written under.

## 2026-10-21 - a non-finite quantizer block is zero

A `.pkf` stores finite floats, but an extreme finite weight can overflow a
product before its activation is quantized. Narrowing the resulting NaN to
`int8_t` was undefined in ref. Every backend now notices a NaN or infinity
while finding the block maximum and writes a zero block instead. The ordinary
finite path keeps ref's bits; kcheck's new non-finite case matches ref on both
ARM backends too (1,663 cases each).

On the M4 Pro, v2's one-minute whole recording, 12 threads, best of five in
interleaved old/new runs: neon-dp 0.498 and 0.512 s before, 0.505 and 0.520 s
with the check; neon 1.073 and 1.111 s before, 1.103 and 1.122 s with it. The
roughly 1-2% cost is kept at the boundary where a hostile model otherwise
reaches undefined behaviour.

## 2026-10-22 - the resampler as a kernel, four blocks a vector (hypothesis)

core/resample.c's dot products ran in core, a chain of fused
multiply-adds an output (FMADD on AArch64: core has the instruction),
about 34 taps a phase from 44.1 kHz and 38 from 48. `pktito bench` on
test.wav resampled, its resample line (new, best of five), three rounds
on the M4 Pro: 9.2 to 10.0 ms at 44.1 kHz, 10.6 to 10.7 ms at 48, about
1.3 cycles a tap: the out-of-order core runs neighbouring outputs'
chains at once.

As a kernel, pk_k_resample, a vector holds four consecutive blocks at
the same phase (backend/neon serves neon and neon-dp alike): the tap
broadcast, and the four inputs, orig floats apart, four LD1s to a lane;
each lane's FMLA is the scalar chain's, so the bits are ref's. Two
phases at once keep two chains in flight. The loads bound it, four a
step for four outputs, three a cycle: about 0.35 cycles a tap, 3 ms.
Kept if kcheck's resample cases are ref's bits and the line falls past
its spread.

Result, kept, short of the hypothesis: kcheck's 114 resample cases (1,777
in all) are ref's bits on neon-dp and neon, and the VAD's probabilities and
both transcripts of test.wav at 44.1 and 48 kHz the bytes of the tree
before, under neon-dp and ref. The line, three rounds: 4.5 to 4.6 ms at
44.1 kHz (9.2 to 10.0 before), 4.5 at 48 (10.6 to 10.7), 2.1 and 2.4
times, not the 3 ms hoped for: the four lane loads of a step, each waiting
on the vector before it, are the step.

## 2026-10-22 - the GB10's little cores (a measurement)

NVIDIA GB10 (a DGX Spark): 10 Cortex-X925 at 3.9 GHz (CPUs 5-9 and 15-19)
and 10 Cortex-A725 at 2.8 GHz (CPUs 0-4 and 10-14), Ubuntu 24.04, kernel
7.0.0-1019-nvidia, GCC 13.3, the Makefile's flags. Tree 3626bc0. `./pktito
bench` on test.wav in its default 30 s chunks (three), best of five; a vLLM
worker resident and idle, load 0.13. check-c passes; the backend picked is
neon-dp.

| threads | on | best | real time |
|---|---|---|---|
| 20, the default (a core of every kind each) | any | 0.711-0.719 s | 83.4-84.3x |
| 16 | any | 0.896 s | 67.0x |
| 12 | any | 0.645 s | 93.0x |
| 10 | any | 0.442 s | 135.8x |
| 8 | any | 0.497 s | 120.7x |
| 5 | any | 0.715 s | 84.0x |
| 10 | the X925s (taskset) | 0.437 s | 137.4x |
| 8 | the X925s | 0.474 s | 126.7x |
| 5 | the X925s | 0.688 s | 87.2x |
| 10 | the A725s | 1.267 s | 47.4x |

The default gives up 1.6x. Ten threads, unpinned, land on the X925s and
run as fast as pinned there. An A725 does a third of an X925's work here
(47.4x against 137.4x, ten of each), where the kernel's cpu_capacity has
718 against 997. The pool deals its items as they are taken, but pk_gemm
cuts a product into whole rounds for equal threads, 16 to 48 items: an
A725's last item holds the job while the X925s wait. A share in proportion
to each core's speed would have the twenty make up to 137.4 + 47.4 = 185x.

The M4 Pro on the same tree and bench (in use while measured): 12 threads
0.477-0.478 s, 125.5-125.8x; 8 0.457-0.458 s, 131.0-131.3x; 10 0.471 s;
6 0.582 s. On the minute whole, 2026-10-20 above found the 12 a little the
faster; in 30 s chunks the 8 are, by 4%.
