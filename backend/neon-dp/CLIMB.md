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
| 2b035b3, first SDOT kernel (8x4 tile, no column blocking) | 12 | 0.543 s | 110.6x |
| 2997d4f, column blocks of 192 KB | 4 | 1.008 s | 59.5x |
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
-mtune=cortex-a76 -O3 -ffp-contract=off`. Tree 9083e1c plus the profile's
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
| 34ecaaa, the baseline above | 4.934 s | 4.371 s | 285 ms | 891 ms |
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
| c00013c | 4.741 s | 4.169 s | 343 ms |
| pk_k_conv3x3s2 a row at a time, 4 channels by 8 columns at once | 4.601 s | 4.026 s | 197 ms |
| and the first stage made a window at a time, the 3 rows the second's next output row reads | 4.537 s | | 115 ms |
| windows for 2, 3 and 4 output rows instead (rejected) | 4.547, 4.567, 4.585 s | | 129, 150, 165 ms |
| every stage a row at a time, each keeping the 3 rows the next reads | 4.505 s | 3.920 s | 90 ms |

An instrumented build (not committed) timed each stage's calls inside the
items, summed over the threads, a run:

| ms of thread time | conv0 | dw1 | pw1 | dw2 | pw2 |
|---|---|---|---|---|---|
| c00013c, 4 threads | 510 | 562 | 130 | 64 | 32 |
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
| b325cd9 | 1978 ms | 163 ms | 4.358 s | 3.847 s |
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
| cfaf43a | 77 ms | 213 ms | 4.244-4.248 s | 3.791 s |
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
| 1aab324 | 4.186-4.189 s | 3.737 s |
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
now, where on 2997d4f they took 0.523 s against 0.512. The Pi 5's four
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
7.0.0-1019-nvidia, GCC 13.3, the Makefile's flags. Tree 57d6ccb. `./pktito
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

## 2026-10-22 - the GB10: the decoder on four threads (hypothesis)

`make PROFILE=1`, the bench above, tree c51b433, the stages that move:

| stage | 10 threads | 20 threads |
|---|---|---|
| decoder | 16.9 ms | 183.2 ms |
| quantize | 53.2 ms | 105.8 ms |
| q k v pos | 59.1 ms | 84.1 ms |
| glu + depthwise | 15.4 ms | 24.1 ms |
| layer norms | 3.1 ms | 11.4 ms |
| attention out | 11.9 ms | 16.5 ms |
| feed-forward | 186.6 ms | 193.6 ms |
| bench | 0.437 s | 0.662 s |

The decoder's products past PK_DEC_SPLIT (2048 rows: the LSTM's 2560-row
gate products, four a token) go to every worker: a job of a few
microseconds for each of twenty, half of them A725s, and woken each time,
since the joint's per-frame work between tokens outlasts their spin. With
every decoder product on the calling thread (PK_DEC_SPLIT past them all,
measured): the decoder 29.6 ms at 20 threads and 28.7 ms at 10, bench
0.564 and 0.448 s. A product wants a few threads, not one and not twenty.

H1: the pool takes a count of active threads, pk_pool_set_active. A job
runs on the first k, the caller one of them; the others sleep on a
condition variable of their own, neither woken by the job nor waited for.
The job's width travels in the generation word with its number, so a
worker learns from one load that there is a job and whether it is in it,
and cannot pair one job's number with the next one's width (pinfer's
core/pool.c does the same). pk_decode runs on four. Predict on the GB10:
the decoder about 15 ms at both 10 and 20 threads, bench at 20 threads
0.662 -> about 0.50 s and at 10 about 0.43 s, the encoder's stages as they
were. The M4 Pro's twelve and the 12900HK's fourteen gain their decoder's
share at most (16 ms there, 3%); the Pi 3's four do not move.

Result, kept. `make PROFILE=1` on the GB10: the decoder 19.9-23.2 ms a run
at 20 threads (183.2 before) and 17.3-20.3 at 10 (16.9); on two threads
26-28 ms at either count, on eight 24-33 at 20 and 17-20 at 10, so four.
`./pktito bench`, the trees built as they ship, interleaved, three rounds:

| board, threads | before (c51b433) | after |
|---|---|---|
| GB10, 20 (its default) | 0.719-0.800 s | 0.573-0.580 s |
| GB10, 10 (the X925s) | 0.435-0.438 s | 0.441-0.447 s |
| 12900HK, 14 (its default) | 0.835-0.840 s | 0.835-0.838 s |
| M4 Pro, 12 (its default; in use) | 0.515-0.529 s | 0.498-0.522 s |

The GB10's default is 1.25-1.39x faster, short of the 0.50 s predicted:
the other small jobs (quantize, the norms) still wait on the A725s. Ten
X925s lose 1.5%, their decoder 3 ms slower on four of them than on ten.
The 12900HK does not move, and the M4 Pro gains 1-3%; the Pi 3 and the
Pi 5, four threads in all, run as before. check-c passes, and check-tsan
on six threads, after one race it found: a worker in no job read the
pool's quit flag with nothing ordering the read before pk_pool_free's
write, which is now atomic.

## 2026-10-22 - the GB10: frame passes in items of a few frames (hypothesis)

After H1, `make PROFILE=1` at 20 threads against 10: quantize 115.9 ms
against 55.6, q k v pos 88.0 against 59.5, feed-forward 200.0 against
186.7, glu + depthwise 27.0 against 15.7, layer norms 12.4 against 3.3,
conv 40.6 against 33.8, attention out 17.4 against 12.0; bench 0.588 s
against 0.438.

The quantize passes, the norms and the GLU cut their T frames into items
of 16: at T 250 (test.wav's 20 s chunks) that is 16 items, so an A725
that takes one ends the job, three X925 items late. In X925 item-times, a
job of N items at 20 threads takes N / 13.3 and then the slowest A725's
remaining item, near 3 when items are large; ten X925s take N / 10. Twenty
threads win once N passes about 100 items, or items an eighth of the
size.

H2: a pass over T frames cuts them into at least eight items for each
active thread, and at most 16 frames an item as now: 2 frames at 20
threads (125 items), 4 at 10, 3 on the 12900HK's 14, 8 on the Pi 3's four.
Predict on the GB10 at 20 threads: quantize 116 -> about 55 ms, the norms
12 -> 4, glu + depthwise 27 -> 18, bench 0.58 -> about 0.50 s; at 10
threads, and on the Pi 3, no move past the spread. The products' items are
H3's.

Result, rejected. check-c passes (the bits cannot move: each frame is
quantized, normed and gated alone). `./pktito bench` against 61d7288,
interleaved, three rounds: 20 threads 0.574-0.584 s against 0.563-0.591,
10 threads 0.437-0.443 against 0.438-0.441. `make PROFILE=1` at 20
threads: quantize 109.9 ms (115.9 in 16 items), the norms 12.3 (12.4),
glu + depthwise 26.9 (27.0). In 125 items a quantize pass still takes
twice its time on the ten X925s, about 190 us against 93, so its items'
size was never what held it: the clocks hold at 3.9 and 2.8 GHz under
either load (scaling_cur_freq, sampled through the runs, performance
governor), and the likelier cost is the memory system's, every pass
reading what the one before left in every worker's cache, from cores a
cluster away. The products do not gain from the A725s either (feed-forward
200 ms at 20 threads against 188 at 10).

So on the GB10 the A725s slow every stage, where the 12900HK's E-cores
speed its run by 3% (above, and backend/avx2/CLIMB.md) and the Core 3
304's carry it (backend/avx2/CLIMB.md, 2026-10-20: one P-core alone at a
quarter of its five cores' speed). A core's speed does not tell them
apart: an A725 does 0.34 of an X925's work, an Alder Lake E-core 0.37 of a
P-core's, a Darkmont E-core about 0.65 of a Cougar Cove's.

## 2026-10-22 - every core, or the fast kind: measured on the encoder's layers (hypothesis)

Every core or the fast kind alone is a question for each chip, and a
core's speed does not answer it (above). The GB10 runs fastest on its ten
X925s (0.437 s against 0.58 on all twenty, after H1); the M4 Pro on its
eight performance cores (by 4% in 30 s chunks); a Snapdragon 7c, two A76s
and six A55s, on its A76s (1.7-2.4x, pinfer's core/SCHED.md); the 12900HK
on all fourteen (by 3%); the Core 3 304 on all five (4x).

H3: where the threads are pktito's choice and the cores are of two kinds,
the first recording long enough to time (100 encoder frames, 8 s) runs its
encoder's layers on every core and on the fast kind by turns, two layers
each, four times over: the layers are the same shapes, so their times
compare. The transcriber keeps the fast kind if its layers took 3% less,
and every core otherwise; the narrower setting runs through pk_pool_set_active,
with the other workers asleep, and the decoder's four come out of it. The
fast kind is what the OS says: arm64 Linux's cpu_capacity, else cpufreq's
maximum, within 10% of the highest; macOS's perflevel0; Windows'
EfficiencyClass. Threads given (-t) are kept as given. Predict: the GB10
picks its X925s, 0.58 -> about 0.44 s; the M4 Pro its eight, a few percent;
the 7c its two A76s; the 12900HK and the 304 every core, unmoved; the Pis,
one kind, untouched.

Result, kept. `./pktito bench`, each tree's default threads, interleaved,
three rounds, against 61d7288 (H1):

| board | before | after | the choice |
|---|---|---|---|
| GB10 | 0.570-0.581 s, 20 threads | 0.437-0.443 s | 10, the X925s, every run and every `-v` |
| 12900HK | 0.836-0.840 s, 14 | 0.838-0.841 s | 14, every core, every run |
| M4 Pro (in use) | | 0.466 s once | 8 in bench, 12 in one `-v`: 4% apart, near the 3% |

The GB10's default is 1.31x faster than H1's and 1.62x the 0.711-0.719 s
it began the day at, as fast as its X925s pinned. The 12900HK keeps every
core and does not move. The M4 Pro's two settings are as close as its
measurement, and it takes either. The Pis are one kind and never time.
check-c passes on the M4 Pro, the GB10 and the 12900HK; check-tsan now
also runs the default threads, which on the M4 Pro switch the pool between
twelve and eight as the first recording is timed, with no race.

The Snapdragon 7c (armlet: two Cortex-A76, six A55, Windows 11), through
the app's Arm64 build of b93bc51, `--transcribe` on test.wav: the default
keeps all eight, 6.17-6.56 s, its layers timed against the A76s'. With the
process held to the A76s (affinity 0xC0) it takes 7.95-8.70 s, to the A55s
14.4 s, so eight is the right choice; Windows names the A76s
(EfficiencyClass 1, the A55s 0). pinfer's 1.7-2.4x on the A76s is a decode
of matrix-vector products; the encoder's matrix products keep the A55s
busy enough to pay. Each transcript is the x64 build's, byte for byte.

## 2026-10-24 - the int8 product on I8MM's SMMLA (hypothesis)

ARMv8.6's I8MM (optional from 8.2) adds SMMLA: a 2x8 matrix of signed
bytes by another's transpose into a 2x2 of exact int32 sums, 32 MACs an
instruction where SDOT's lane form does 16. Neoverse V1, V2 and V3
(Graviton 3, 4 and 5), the X2 and later X cores with the A710 and later
(the GB10's X925 and A725), and Apple's M2 and later have it; the A76 (Pi
5) and the M1 do not. Wherever SMMLA issues as often as SDOT, the inner
loop's multiplies halve.

The file's layout gives SMMLA's weight operand with one ZIP. A group is
four inputs by 16 rows, each row's four bytes together; the group after it
holds the next four inputs. ZIP1 at 32 bits of a vector of group k/4
(rows 4j..4j+3) with the same vector of group k/4 + 1 makes rows 4j and
4j+1, eight inputs each, and ZIP2 rows 4j+2 and 4j+3: SMMLA's B for two
rows. The activations' A is two columns' eight bytes, x[c] + k and
x[c+1] + k, one 64-bit load each. No repacking of the file: its pages stay
the page cache's, shared, and every other backend reads the same bytes.

The tile: a half panel's 8 rows by NC columns, as SDOT's, 8 inputs a step.
A step loads the half panel's 4 vectors of each of the two groups and
ZIPs them into 4 B operands, then for each pair of columns loads A and
runs 4 SMMLAs (rows 0-1, 2-3, 4-5, 6-7). At NC = 4: 8 SMMLAs and 4 ZIPs
for 256 MACs, against 16 SDOTs. A block's end takes each accumulator
[c r, c r+1, c+1 r, c+1 r+1] apart into the columns' row vectors with
ZIP1 and ZIP2 at 64 bits, 4 a column pair, and they go into the float
chain by the same convert and fused multiply-add as the SDOT tile's.

Bits: the integer sums are exact, in any order, and the chain is ref's,
lane for lane. An odd last column and the decoder's single column take
the SDOT tile.

What it should do: if SMMLA issues as often as SDOT (not checked here
against any core's guide; this measures it), the product's multiplies run
2x faster, the ZIPs and the float side taking back some of it. Predict the product 1.4 to 1.8x on one core and `pktito bench` 15 to
30% on c7g, c8g and c9g, less on the M4 Pro, whose SMMLA throughput is
not published. If SMMLA issues at half SDOT's rate on a core, the tile is
level or slower there, and the order must not put it first on that core.

Built as a third variant of backend/neon, backend/neon-i8mm: neon-dp
with -march=armv8.2-a+dotprod+i8mm (Apple: -mcpu=apple-m2), needing
asimddp and i8mm. core/os.c learns i8mm: HWCAP2_I8MM on Linux,
hw.optional.arm.FEAT_I8MM on macOS, PF_ARM_V82_I8MM_INSTRUCTIONS_AVAILABLE
(66, MinGW's winnt.h) on Windows. Kept if kcheck and check-c are ref's
bits on a Graviton and the M4 Pro, and the bench moves past its spread;
first in the order only where it wins.

## 2026-10-24 - the int8 product on I8MM's SMMLA (result: rejected)

Built as the hypothesis above has it, with two changes the first version
measured into: sixteen inputs a step (two columns' 16 bytes, ZIP1 and
ZIP2 at 64 bits, make A for both halves, where two 8-byte loads and an
insert made it for one), and the float chains run in SMMLA's order,
[c r, c r+1, c+1 r, c+1 r+1], each lane its column's d, taken apart into
the columns' rows once at the tile's end: the same fused multiply-add for
every output, ref's chain. kcheck, 1777 cases, ref's bits on neon-i8mm,
neon-dp and neon, and check-c passes, on a c8g (Neoverse V2, gcc 13.3),
the M4 Pro (Apple clang) and the GB10 (gcc 13.3); every transcript below
is ref's. The code is the commit before this entry's ("neon-i8mm: the
int8 product on SMMLA, as measured"); this one takes it out.

The tile's shape, on a c8g.large (one V2 core, two threads), `pktito
bench` on test.wav, interleaved with neon-dp's 15.4 to 15.5x: eight
inputs a step with the chains taken apart every block, 14.4x; sixteen a
step, 15.9 to 16.0x; the chains in SMMLA's order, 16.1 to 16.2x; six
columns a tile in place of four, 16.0x.

EC2, us-east-1, each instance fresh, Ubuntu 24.04, one binary built on
the c8g holding all three, three interleaved rounds of `pktito bench`
(best of five, the default threads), then q8bench on one thread, the
1024 x 1024 attention product at T 750 in pk_gemm's 64-row slices,
GMAC/s; the M4 Pro (in use, load 3 to 4) and the GB10 (idle) by
pktito-mac, three interleaved rounds:

| board | core | neon-dp | neon-i8mm | | q8bench |
|---|---|---|---|---|---|
| c6g.xlarge | Neoverse N1 | 18.9x | (no I8MM: passed over) | | |
| c7g.xlarge | Neoverse V1 | 24.5x | 26.3x | +7.3% | 81.3 to 85.8 |
| c8g.xlarge | Neoverse V2 | 30.7x | 32.0x | +4.2% | 103.8 to 110.3 |
| c9g.xlarge | Neoverse V3 | 44.3x | 45.9x | +3.6% | 126.5 to 133.1 |
| GB10, -t 10 | X925 and A725 | 136.3-137.9x | 138.3-139.8x | +1.4% | X925 202-204 to 209, A725 61.0 to 63.9 |
| GB10, its X925s alone | X925 | 135.6-136.9x | 137.1-137.9x | +0.8% | |
| GB10, -t 20 | X925 and A725 | 105.0-108.1x | 104.7-107.4x | level | |
| M4 Pro | M4 | 131.9-134.1x | 113.4-115.7x | -12 to -15% | 182-186 to 148 |

(six columns on the M4 Pro: 117.0 and 118.4x, the product 155.)

mmrate2.c, ns an instruction with 1, 4, 8 and 16 independent chains, one
thread (1 is the latency; 16 the issue rate where the latency allows):

| core | SDOT | SMMLA |
|---|---|---|
| Neoverse V2 (2.8 GHz) | 0.36, 0.17, 0.11, 0.097 | 0.36, 0.17, 0.10, 0.097 |
| X925 (GB10) | 0.43, 0.12, 0.066, 0.047 | 0.44, 0.12, 0.065, 0.047 |
| A725 (GB10) | 0.357, 0.179, 0.179, 0.179 | the same |
| M4 Pro P-core (~4.5 GHz) | 0.67-0.72, 0.17, 0.087, 0.066 | 1.334, 0.333, 0.17, 0.131-0.135 |

Where SMMLA issues as SDOT does (V2, X925, A725) it does twice the
multiply-adds an instruction, but the tile does not live on the
multiply: each SMMLA operand costs a ZIP that SDOT's lane form does not
(0.75 a SMMLA at 4 columns by 8 rows), and 32-input blocks put a convert
and a fused multiply-add on every accumulator every four SMMLAs, so the
SIMD instructions a MAC fall by about 8% and the product moves 3 to 6%.
Wider tiles to share the ZIPs run out of the 32 registers, as six
columns did. On the M4 SMMLA issues at half SDOT's rate with twice its
latency: the same multiply-adds a cycle, and the ZIPs all cost.

Rejected. Not by the M4 alone - leaving Apple out would be a platform
condition, not a list of models - but by what is left: 1 to 7% on the
cores where it can win, a third NEON backend to keep and read for it,
and two places it would run unmeasured and could lose (Linux on Apple's
cores; Snapdragon X's Oryon, which has I8MM, under Windows). A core
whose SMMLA is the limit - wider than SDOT, or a kernel whose blocks are
longer than 32 - is where to look again. Untried, and the question for
whoever does: the same tile timed with its ZIPs taken out (wrong sums,
every other instruction as it is), which bounds what weights already in
SMMLA's order - a second copy of the model laid out for it, or a file
format of its own, with the page cache's sharing lost - could buy over
SDOT on these cores.

Seen in passing: without -t, the encoder's choice between every core and
the fast kind (core/encoder.c, its layers 3% faster to keep the fast
kind) took the GB10's twenty threads in two of three neon-i8mm runs,
101.7x and 110.5x against 138.8x at its ten X925s, where neon-dp took
the ten every time; on the M4 Pro neon-i8mm went between 8 and 12, and
neon-dp held at 8. A kernel that moves the balance between the two kinds
of core can move that choice, and the 3% margin gave way here: worth a
look when one does.

## 2026-10-24 - the program is pktito-c (a note)

The command-line program the Makefile builds, `./pktito` in the rows
above, is `./pktito-c` from here on, so that it is not taken for the
app, pktito; the same program, the same output. The rows above keep the
name they were measured under.

## 2026-10-24 - the int8 product on SME2's SMOPA (a measurement: not taken up)

The M4 Pro reports FEAT_SME2 with a 512-bit streaming vector, an SME unit
to each of its two performance clusters. Before any kernel, how much of
the unit the q8 format leaves: microbenchmarks only, Apple clang 21,
-mcpu=apple-m4, the operands in registers (no loads), one thread unless
said, the M4 in use (load average 1.7 to 8). Their sources are kept with
the session's notes, not in the tree.

| loop | int8 multiply-adds a second |
|---|---|
| SDOT, 16 chains, 1 / 2 / 4 / 8 threads | 240-259 G / 497-504 G / 908-910 G / 1713-1804 G |
| SMOPA (16 x 16 int32, 1024 a SMOPA), 4 tiles, 1 / 2 / 4 / 8 threads | 1984 G / 1957 G / 2254-2364 G / 3812-3829 G |
| SMOPA, 8 a 32-input block, each tile then drained: rows to Z four at a time, SCVTF, FMLA, zeroed | 498 G |
| the same with SME2's multi-vector forms (vgx4 array reads, SCVTF x4, FMLA vgx4 into a float tile), one scale vector or one a row | 620-633 G |
| one tile filled while the other is drained (double-buffered), vgx4 | 728-729 G |

One thread fills a unit (2.0 T); the second cluster's makes 3.8 T with
threads on both, about 2.2 times eight cores of SDOT. But the format's
scale on every 32 inputs ends each tile's int32 sum after 8 SMOPAs, and
the drain to float costs about two thirds of the unit however it is
written here: the best, double-buffered, 0.73 T a unit, 1.46 T for two,
is about what eight cores of SDOT give the product now (q8bench's
182-186 G a core), before a single load. So SME2 is not taken up for the
q8 format as it is. What would change it is a longer integer sum - a
block of 128 inputs or more, or a scale a row - which is a quantization
choice for tools/quantcheck.py, not a kernel's.

## 2026-10-24 - pk_gemm's step, 256 rows where they make two rounds (measurement)

core/encoder.c's slices of 256 rows where a product's rows / 256 is at
least twice the threads (backend/avx2/CLIMB.md, pk_gemm's step), which at
four threads moves only the feed-forward's up product: Pi 5, neon-dp: 14.4-14.7x against 14.4-14.8x,
interleaved, test.wav, best of five. Level.

The M4 Pro (pktito-mac's measurement, 03:51, the machine in use, load 1.6
to 2.0): at its eight threads the up product takes the step (16 slices,
twice eight), test.wav interleaved, 47f0093 134.9, 134.4, 133.9x against
bec30f3 133.6, 133.9, 133.4x: -0.5%, at the edge of the spread, so level
or a hair under; not acted on unless a quiet run repeats it.

## 2026-10-24 - Graviton5 against Zen 5, stage by stage (a measurement)

Why a Graviton costs two to three times what a c8a does per hour of
audio (AliceBench on EC2): is one stage behind, or the core? EC2,
us-east-1, each instance fresh, Ubuntu 24.04, gcc 13.3, the tree at
ab737a1 built on the instance, plain and `make PROFILE=1`; `pktito-c
bench` on test.wav, best of five. A c9g.xlarge (Neoverse V3, 3.3 GHz,
4 cores) on neon-dp: 44.4, 44.5, 44.5x at four threads, 12.4x at one. A
c8a.xlarge (Zen 5, 4 cores) on avx512-vnni: 133.6, 133.2, 133.0x, 36.8x
at one.

The PROFILE build at one thread, ms a run:

| stage | c9g (neon-dp) | c8a (avx512-vnni) | c9g / c8a |
|---|---|---|---|
| feed-forward | 2508.2 | 825.6 | 3.04 |
| q k v pos | 761.7 | 260.6 | 2.92 |
| conv pw1 pw2 | 460.8 | 150.9 | 3.05 |
| attention (float) | 437.4 | 150.6 | 2.90 |
| quantize | 259.0 | 62.9 | 4.12 |
| attention out | 157.8 | 50.2 | 3.14 |
| subsampling | 126.3 | 46.3 | 2.73 |
| glu + depthwise | 54.3 | 23.0 | 2.36 |
| decoder | 39.0 | 25.5 | 1.53 |
| mel | 17.5 | 18.4 | 0.95 |
| the run | 4854 | 1631 | 2.98 |

q8bench, one thread, the 1024 x 1024 attention product at T 750: 126.7
GMAC/s on the V3, 407.0 on Zen 5 (the feed-forward's up 125.1 and
405.3). Four threads over one: 3.6 times on both.

The int8 products and the float attention are three times behind alike,
and each stage's share of the run is the same on both to a point or two
(the feed-forward 51.6% and 50.2%): no kernel lags, the core is
narrower. If the V3 issues four SDOTs a cycle as the V2 does (mmrate2.c,
above; not measured on the V3), its ceiling at 3.3 GHz is about 211
GMAC/s, and the product is at about 60% of it, where it is on the V2.
The quantizer is the one stage out of line, 4.1 times to the run's 3.0:
brought into line it would give about 70 ms of the 4854, 1.5%. What
would move the Arm products more is the longer integer sum the SMMLA and
SMOPA entries above name, a quantization choice.

## 2026-10-25 - SMMLA on Snapdragon X's Oryon (a measurement: the open case closed)

The i8mm entry above left Snapdragon X's Oryon, which has I8MM, as a core
it could have won or lost on unmeasured. snappy: an ASUS Vivobook S 15,
X1E-78-100 (12 first-generation Oryon cores, 3.4 GHz), Windows 11 Home on
AC in Best performance; its ID registers (Windows' "CP 40xx" values)
give DP and I8MM, no SVE and no SME. mmrate2.c as above, built by
pktito-9950x with llvm-mingw 20261020 (clang 23.1.4), `-O2
-march=armv8.6-a+i8mm -static -lpthread`, three runs, within 1%:

| core | SDOT | SMMLA |
|---|---|---|
| Oryon (X1E-78-100) | 0.887-0.892, 0.221-0.223, 0.110-0.111, 0.073-0.074 | 1.178-1.197, 0.294-0.295, 0.147-0.148, 0.147-0.148 |

SMMLA issues at half SDOT's rate with a longer latency, as on the M4: the
same multiply-adds a cycle, and the ZIPs all cost, where neon-i8mm was 12
to 15% slower. Not a core for it either; the rejection stands, with no
core left where it could have won by more than the Gravitons' 1 to 7%.
The same laptop's AliceBench on neon-dp, 0.9.2's signed pktito-cli: 51.0x
(pktito-app ALICEBENCH.md).

## 2026-10-25 - a Snapdragon X Elite's cores under Android, counted by thread siblings (a measurement)

A Dell XPS Googlebook, X1E-80-100: 12 first-generation Oryon cores in
three clusters of four, cpu0-3 to 3.42 GHz (cpu_capacity 870), cpu4-7 and
cpu8-11 to 4.01 GHz (1022, 1024); Android 17 (API 37), its 6.12 GKI kernel,
the WALT governor, 16 GB. pk_os_cores counted distinct (package, core_id)
pairs, and the kernel numbers core_id 0 to 3 within each cluster, all in
package 0, as Linux does for any device tree's cpu-map without a socket
node (drivers/base/arch_topology.c, parse_cluster): 4 cores, so 4 threads.
Each of its CPUs is its own thread_siblings_list, so counting the CPUs
that head theirs gives 12; the fast kind is cpu4-11, 8, which the
encoder's layers time against all 12 as before. The two counts read from
sysfs, old and new: the Googlebook 4 and 12, the 9950X (32 threads, under
WSL) 16 and 16, the Pi 5 4 and 4, the Pi 3B+ 4 and 4. An RK3588's cpu-map
(rk3588-base.dtsi) has clusters of 4, 2 and 2, which the old count makes 4
of 8; not measured.

`make adb` (2a3d597), NDK r30's clang 21.0.0 for API 24, linked to the
device's bionic: check-c passes there, kcheck every one of ref's bits on
neon-dp and neon. `pktito-c bench` (test.wav through v2, best of 5), the
two builds interleaved, four rounds, on AC with the laptop in use (Chrome
playing media, 0.4 GB of 15.5 GB free), zone0 42 to 68 C, in seconds:

| run | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| before, default: 4 threads | 1.029 | 1.507 | 1.095 | 1.256 |
| after, default: 8 each time | 0.677 | 1.243 | 0.876 | 0.665 |
| after, -t 4 | 1.086 | 2.098 | 1.741 | 1.310 |
| after, -t 8 | 0.694 | 1.231 | 1.223 | 1.205 |
| after, -t 12 | 0.665 | 0.917 | 1.183 | 1.186 |

The default is 1.2 to 1.9x faster in every round. Four rounds of a static
build of the same change earlier the same day, the first two moving from
battery to AC: the default 1.03-1.42 s before, 0.55-1.01 s after (8, 8, 12
and 12 threads); -t 8 0.65-0.77 s, -t 12 0.61-0.63 s. make adb's own bench
of the change: 0.580 s, 103.4x, on 12.

Given -t, 12 beat 8 in all eight rounds, by 2 to 26%, yet the default kept
8 in six of its nine runs. Its timing is two layers at a time while the
laptop's other work comes and goes, and the 3% rule needs a quiet board to
judge; left as it is until one is measured quiet.

## 2026-10-27 - every core, or the fast kind: each by its fastest pair, decided on the second recording (hypothesis)

H3 (2026-10-22, above) sums the sixteen layers it times, eight pairs a
side, on the first recording of 8 s or more, keeps the fast kind if its
sum is 3% less, and decides once. Twice since, it has kept the slower
side: the GB10 under neon-i8mm took its twenty threads in two of three
runs, 101.7x and 110.5x against 138.8x on its ten X925s (2026-10-24), and
the Googlebook kept 8 in six of nine runs where, given -t, 12 beat 8 in
all eight rounds by 2 to 26% (2026-10-25). Two things in what it times can
do that.

A sum carries every pair that other work slowed, and the Googlebook was
playing media. Other work only ever adds time, which is why pinfer's race
between its pool's settings judges each by its fastest sample (pinfer's
core/SCHED.md, 2026-10).

And the first recording is the first pass over the model's int8 weights,
which nothing reads before it: the loader reads the floats, to check each
is finite, and none of a layer's 25 MB of int8. Every page of a layer is
mapped in while that layer is timed - about 1,500 of macOS's 16 KB pages -
by the kernel, whose work need not divide among the threads as the
arithmetic does, nor the same way on both sides.

H4: each side is judged by its fastest pair; the race runs on the first
recording of 8 s or more, as before, and again on the next, and the
second verdict stands, the first holding in between. The pairs and the 3%
margin are H3's. Predict, on the M4 Pro (in use), with each race's pairs
logged from inside: the first recording's pairs slower than the same
audio's when its pages are warm, and the second verdict steadier from run
to run than the first; the default level with the better of -t 8 and
-t 12, which H3 found 4% apart. On the GB10 its ten X925s every run, as
H3 had them under neon-dp; on the Googlebook 12. The 12900HK and armlet
unmoved, and the Pis, one kind, never race. Its cost is a second
recording whose sixteen raced layers run half on the slower side: on the
GB10, where every core is 1.32 times the X925s' time, about a tenth of
that recording's encoder.

## 2026-10-27 - every core, or the fast kind: each by its fastest pair, decided on the second recording (result: kept, on the M4 Pro)

The M4 Pro alone: of the boards the rule was written for, none was up.
In use throughout (load 4 rising to 8.4), Apple clang 21, test.wav
through v2: three chunks, 250, 348 and 150 encoder frames.

From inside, a build that logged each race's eight pairs. Set to race
every chunk of every pass, so that the first race's audio was raced again
warm: the first process, after the mac had let the model's pages go (180
MB free), timed the first chunk's pairs at 35.5-45.6 ms against 11.2-12.0
in its four warm races, 3.0 to 3.9 times, with every core 15% ahead where
warm the fast kind was 2-4% ahead - its 50 MB a pair at about 1.9 GB/s,
the disk's rate. Two more processes, the file cached: the first race's
first pair 14.4 and 15.0 ms, the rest 12.0-12.4, against 11.2-11.9 warm;
and the second race's first pair, at 348 frames, where the buffers had
just grown, 20.2 ms against 15.9-16.2. A race's first pair is always
every core's, so a sum charges every core with what a recording sets up
first: both cached processes summed to 8 in both races, where the fastest
pairs were within the margin (12.12-12.27 against 11.96-12.08 ms, and
15.93-15.97 against 16.13-16.23), which keeps every core. Warm, the fast
kind's lead shrinks as a recording grows: 5-8% at 150 frames, 2.4-4.7% at
250, none at 348.

`./pktito-c bench`, six rounds, the order reversed every other round,
best of 5 and the threads each run kept:

| run | threads | best of 5, s |
|---|---|---|
| H3 (c42738d), default | 8, 8, 8, 12, 12, 12 | 0.474, 0.480, 0.484, 0.484, 0.485, 0.485 |
| H4, default | 12 in all six | 0.460, 0.463, 0.473, 0.486, 0.490, 0.486 |
| -t 8 | | 0.443, 0.615, 0.496, 0.508, 0.513, 0.513 |
| -t 12 | | 0.494, 0.497, 0.482, 0.482, 0.484, 0.485 |

-t 12 beat -t 8 in five rounds of six, the more as the machine got busier,
and H4's default is level with it. Five more logged processes, normally
run, the machine busier still (the 348-frame pairs 17-31 ms): the second
race kept 12 in all five, the first 12, 12, 12, 12 and 8. In eleven
processes H4 kept 12 every time; H3's verdict, the first race's sum, was
12 in seven and 8 in four. check-c and check-tsan pass on the M4 Pro
(check-tsan's default-thread run switches the pool through both races).

Kept. The first race times the disk or the page tables as much as the
arithmetic, as predicted; the fastest pair drops the first pair's setup
and the passes other work slowed; and the second verdict holds where the
first moved. Still to measure: the GB10 (where H3 took its twenty threads
under neon-i8mm), the Googlebook (where it kept 8), the 12900HK and armlet.

## 2026-10-27 - the power-throttling opt-out and H4's race on armlet (a measurement)

armlet: Snapdragon 7c, six A55s (Windows' EfficiencyClass 0, lp0-5) and
two A76s (class 1, lp6-7), Windows 11 on Arm, on AC. Run by the
pktito-windows session over ssh, so in session 0 with no window, the
case that sent snappy's pool onto its four class-0 cores (core/os.c).
The builds were 98a54d1 (before the opt-out), c42738d (with it) and
79af521 (with it and H4), cross-built with llvm-mingw 20260922 (clang
23.1.2, aarch64-w64-mingw32, static). Every build's --json of test.wav is
test.v2.json byte for byte, on neon-dp. Before every run, a check that
no other process had used more than 0.5 s of CPU in 3 s. Five rounds,
the order of each pair reversed every other round. Best of 5, s,
before / after:

| run | 1 | 2 | 3 | 4 | 5 | median |
|---|---|---|---|---|---|---|
| default (8, every run) | 6.289 / 6.368 | 5.778 / 6.267 | 6.937 / 6.065 | 5.702 / 5.181 | 6.993 / 5.284 | 6.289 / 6.065 |
| -t 8 | 6.324 / 6.409 | 5.271 / 5.977 | 6.010 / 5.835 | 6.522 / 5.346 | 5.418 / 6.948 | 6.010 / 5.977 |
| -t 2 | 7.260 / 8.537 | 7.249 / 7.187 | 8.119 / 8.118 | 7.261 / 8.147 | 7.926 / 8.134 | 7.261 / 8.134 |

The spread from run to run is about 13% either way, drifting slowly,
and the best of 5 does not take it out. The -t 2 times fall near 7.2 or
8.1 s under both builds, and two extra pairs went the other way (8.229 /
8.004 and 7.976 / 7.176). Each core's busy share, from 1 s samples in one
extra run of each: -t 8 put the A55s at 77-79% before and 85-88% after,
and the A76s at 98-99% in both; -t 2 put both threads on the A76s
(99-100%) under both builds, the A55s at 0-7%.

So the opt-out moves nothing past the spread here. In session 0, Windows
gave the 7c's two threads its A76s by itself, throttled or not: its
classes name the right cores, and its heuristics did not hold the process
to the class-0 ones as they held snappy's. The opt-out is for snappy's
case, still unmeasured. -t 2 never beat -t 8 (the best -t 2 was 7.187 s,
the worst -t 8 6.948), so 2026-10-22's verdict of eight stands, though
those runs were over ssh too.

H4's race, default threads, c42738d / 79af521: 6.811 / 5.974, 5.794 /
5.329, 6.803 / 5.199, 5.642 / 5.228, 6.407 / 7.070; medians 6.407 / 5.329.
Both kept 8 in all ten runs, as predicted. Both rules race only within
bench's first pass, the one that also maps the model in, and passes two
to five do the same work under both, so the second's lower median is the
board's spread (paired sd about 0.8 s at five pairs), not the rule.

## 2026-10-27 - H4 on the GB10 (result: rejected), and the middle two pairs (hypothesis)

The GB10 came up after H4 was kept: ten X925s and ten A725s, the
performance governor, gcc 13.3, a vLLM container idle beside it.
`./pktito-c bench`, test.wav through v2, six rounds, the order reversed
every other round:

| run | threads | best of 5, s |
|---|---|---|
| H3 (c42738d), default | 10 in all six | 0.438, 0.438, 0.441, 0.443, 0.448, 0.447 |
| H4 (79af521), default | 10, 20, 10, 10, 20, 10 | 0.441, 0.576, 0.445, 0.445, 0.567, 0.446 |
| -t 10 | | 0.436, 0.438, 0.448, 0.440, 0.440, 0.444 |
| -t 20 | | 0.586, 0.578, 0.603, 0.600, 0.561, 0.577 |

H4 took the twenty in two runs of six and lost 30% in each, where H3
kept the ten every time. Ten processes of the logging build, both races
each: the second race's fastest pairs had the X925s 0.4 to 12% ahead, and
within the 3% margin once, where a steady pass's layers have them 14 to
46% ahead. Timed a layer at a time:

| frames | -t 10, a layer | -t 20, a layer | in the race, the fast kind's | in the race, every core's |
|---|---|---|---|---|
| 150 | 3.4-4.7 ms | 5.4-7.5 | 3.4-5.2 | 4.6-8.9 |
| 250 | 5.6-6.4 | 6.4-8.6 | 5.6-7.5 | 5.9-8.1 |
| 348 | 7.7-9.1 | 8.4-10.7 | 7.6-9.1 | 7.8-10.4 |

The pool's threads are not pinned, and a switch leaves them where they
were. From twenty to ten, the ten that stay are on both kinds of core
and take some layers to come onto the X925s: even a -t 10 pass's first
four layers take 5.9-6.4 ms against 5.6 after, coming from the decoder's
four threads. From ten to twenty, the first ten are on the X925s already,
placed better than twenty threads are left to themselves: every core's
layers in the race ran as fast as 5.9 ms at 250 frames, where -t 20 ran
6.4 to 8.6. Both put the fast kind behind its true time, and the fastest
pair takes each side's luckiest pair, which for every core is its best
placement. H3's sum carried the slow first pair, always every core's,
and that offset the switches here; on the Googlebook it was the other way.

pinfer's race pins its threads, so its spread is other work's and its
fastest sample is its true one. pktito's spread is also its own threads'
placement, and that is part of a side's time.

H5: each side by its four pairs without its fastest and its slowest - the
middle two - so one slow pair (the first pair's setup, a burst of other
work) drops out, and so does a side's luckiest placement. The second race
decides, as in H4. Predict: the GB10 keeps its ten in every run (in the
ten logged processes the middle two had the X925s 6 to 13% ahead in every
second race); the M4 Pro keeps 12 (the middle two pick 12 in all seven of
its logged second races); armlet's eight, which won its races by far more
than any statistic of them moves, not rerun.

## 2026-10-27 - H5 on the GB10 (result: rejected), and every layer raced (hypothesis)

H5 as above, sixteen layers and the middle two of each side's four
pairs, on the GB10: six rounds against H3, H5 kept the ten in all six
(0.438-0.450 s, -t 10 0.437-0.447, -t 20 0.562-0.592), and H3 took the
twenty in one (0.587 s); ten more H5 runs kept ten in nine. Logged, 55
second races, the X925s' middle two against every core's: 0.856 to 0.976,
three of them past 0.97, which keeps every core. At 348 frames a side's
pairs vary by about 9% either way, and two pairs a side do not hold a
true lead of 7 to 10% clear of a 3% margin.

H6: race every layer, in pairs, as many as make the same number on each
side - all 24 of a 0.6B model's, six pairs a side - and judge each side
by its pairs without its fastest and its slowest, the middle four. The
second race decides and the margin is 3%, as in H5. Measured before the
change, in a build that raced all 24 layers: 25 second races put the
X925s' middle four at 0.886 to 0.940 of every core's, and the first 16
layers' middle two of the same races at 0.876 to 0.957. Predict: the
GB10 keeps its ten in every run, the M4 Pro 12, the Tab S7 FE 4 (-t 4
beat -t 8 by 3 to 8% in five rounds there, and H3 kept 4 in all five).

## 2026-10-27 - H6, every layer raced and the middle four pairs (result: rejected, on the RG477V)

`./pktito-c bench`, test.wav through v2, six rounds, the order reversed
every other round, best of 5 in seconds and the threads each run kept.

The GB10 (gcc 13.3):

| run | threads | best of 5, s |
|---|---|---|
| H3 (c42738d), default | 10, 10, 20, 10, 10, 10 | 0.435, 0.442, 0.561, 0.447, 0.443, 0.449 |
| H6, default | 10 in all six | 0.444, 0.440, 0.445, 0.452, 0.448, 0.448 |
| -t 10 | | 0.438, 0.443, 0.446, 0.447, 0.450, 0.447 |
| -t 20 | | 0.573, 0.588, 0.592, 0.571, 0.577, 0.585 |

Thirty more default runs of each: H6 kept the ten in all thirty, H3 in 26.
Over the day's runs on the GB10, H3 took the twenty in 6 of 48, each about
30% slower (0.56-0.59 s against 0.44), and H6 in none of 36.

The M4 Pro (Apple clang 21, in use, load 2.4 rising to 8.9):

| run | threads | best of 5, s |
|---|---|---|
| H3 (c42738d), default | 12, 8, 8, 12, 12, 8 | 0.456, 0.467, 0.474, 0.486, 0.487, 0.510 |
| H6, default | 12 in all six | 0.454, 0.453, 0.471, 0.484, 0.487, 0.487 |
| -t 8 | | 0.442, 0.455, 0.485, 0.500, 0.509, 0.510 |
| -t 12 | | 0.453, 0.453, 0.478, 0.480, 0.490, 0.488 |

Twenty more H6 runs kept 12 in all twenty. -t 12 led -t 8 in four rounds
of six, the more as the machine got busier, and H6's default is level with
it.

The Tab S7 FE (Snapdragon 778G: four A55s, four A78s; Android 14, NDK
r30's clang 21, on AC): H6 kept 4 in five runs of five, as H3 did, where
-t 4 had beaten -t 8 by 3 to 8% in every round. Its best of 5 trailed H3's
in four rounds (8.7, 7.8, 7.1 and 1.9% behind, 1.5% ahead in the fifth),
both on four threads after the first pass, which holds both rules' races.

The RG477V (MediaTek Dimensity 8300: four A510s, cpu_capacity 221, and
four A715s, 996-1024; Android 14, on battery). Both builds' --json is
test.v2.json. Five rounds, the order reversed every other round:

| run | threads | best of 5, s |
|---|---|---|
| H3 (c42738d), default | 4, 8, 8, 4, 8 | 3.089, 3.613, 3.682, 3.299, 3.520 |
| H6, default | 4, 8, 4, 8, 4 | 3.283, 3.533, 3.507, 3.769, 3.554 |
| H3's build, -t 4 | | 3.240, 3.240, 3.372, 3.178, 3.306 |
| H6's build, -t 4 | | 3.421, 3.116, 3.451, 3.423, 3.443 |
| -t 8 | | 3.619, 3.372, 3.549, 3.740, 3.808 |

Ten more default runs of each: H6 took 8 in eight, H3 in two. Over
fifteen, H6 kept the four in five and H3 in ten, where -t 4 beat -t 8 in
every round, by 3 to 15%. At -t 4, where neither build's race runs, H6's
build trailed H3's in four rounds and led in one: the board's spread, not
separable here from the binaries'.

Timed a layer at a time, in a scratch build: in every mode, the -t 4 and
-t 8 passes as much as the races, about one layer in seven took 90 to 260
ms where its neighbours took 20 to 60 - the board stopping the threads
every few hundred ms, other work's time. Steady, the four A715s took 35, 50
and 21 ms a layer at 250, 348 and 150 frames, all eight cores 39, 54 and
24.5: the fast kind 7 to 14% ahead. Fifteen pair races put the fast
kind's middle four at 0.66 to 1.35 of every core's: two stalls in a
side's six pairs outweigh the one the trim drops. Raced instead in two
halves, the first twelve layers on every core and the last twelve on the
fast kind, each side by the median of its last eight: 0.80 to 1.00 over
fifteen races, fourteen of them keeping the four; 0.89-0.94 at 250 frames
against 0.90 steady, 0.92-1.00 at 348 against 0.93, 0.80-0.86 at 150
against 0.86. Stalls are one or two of a half's layers, which a median of
eight leaves out, and a half has no turns to switch between: every
core's twelve layers follow every core's passes, as they would in a
steady pass, and the fast kind's first four give its threads time to
come onto its cores (the GB10's took about four, above).

## 2026-10-27 - every core, or the fast kind: raced in two halves, once (hypothesis)

H7: the second recording long enough to time races; the first runs on
every core as the transcriber starts, its pass over the weights mapping
their pages in. The race runs the first half of the layers on every core
and the second half on the fast kind, and judges each side by the median
of its layers after its first four: eight a side on a 0.6B model. The
fast kind is kept if its median is 3% less. Every layer races, so a first
race's verdict would steer none of its recording's layers, which is why
there is one race and not two; the pool is on every core until it, so the
every-core half runs as a steady every-core pass does. Predict: the RG477V
keeps its four in nearly every run (its closest halves were at 348 frames,
where test.wav's second chunk falls, 0.92 to 1.00); the GB10 its ten in
every run; the Tab S7 FE its four; the M4 Pro level with the better of -t
8 and -t 12. Its cost is twelve layers of one recording on the slower
side, where H6's was twenty-four.

## 2026-10-27 - H7 (result: rejected), and turns judged by their layers' medians (hypothesis)

H7, six rounds on the GB10:

| run | threads | best of 5, s |
|---|---|---|
| H3 (c42738d), default | 20, 20, 10, 20, 10, 10 | 0.569, 0.561, 0.439, 0.594, 0.447, 0.443 |
| H7, default | 10 in all six | 0.435, 0.437, 0.442, 0.442, 0.443, 0.443 |
| -t 10 | | 0.437, 0.447, 0.447, 0.445, 0.443, 0.440 |
| -t 20 | | 0.582, 0.592, 0.562, 0.593, 0.587, 0.568 |

Thirty more of each: H7 kept the ten in 27, H3 in 26. Thirty logged races,
all at 348 frames, put the X925s' median at 0.856 to 1.005 of every
core's (median 0.919), four of them past 0.97. The M4 Pro, six rounds: H7
kept 12 in all six and twenty more, level with -t 12 (0.453-0.509 against
0.460-0.502); H3 took 8 in three; -t 12 led -t 8 in five rounds by 3 to
4.6%. The RG477V, now at 28 C throughout (22-27 in H6's runs): H7 kept the
four in nine runs of fifteen, H3 in four, where -t 4 led -t 8 by 4 to 6%
in four rounds and trailed by 3.4% in one.

H7 does without the turns' bias but times the two sides apart, and the
board's drift between halves lands on one side: at 348 frames, where the
X925s lead by about 10%, its ratio spread over 0.856-1.005, where H6's
turns held the same races to 0.886-0.940. What undid H6 on the RG477V was
its stalls, one layer in seven, which a pair statistic cannot drop. A
median of each side's single layers can: computed from the logged turns,
it kept the four in 14 of the RG477V's 15 races, and the ten in 5 of the
GB10's 6. The misses were at 0.977 and 0.982, inside the 3% margin, which
favours every core. On the GB10, the RG477V and the Tab S7 FE, every core
lost wherever its race was close; on the M4 Pro, where every core wins,
the second races read 1.00 to 1.06.

H8: turns of two layers over every layer, as in H6; each side by the
median of its own layers; whichever is faster kept, with no margin; one
race, on the second long recording, as in H7. Predict: the GB10 keeps its
ten in every run; the RG477V its four in most (the medians of its logged
turns put the fast kind at 0.81 to 0.98); the M4 Pro is level with the
better of -t 8 and -t 12.

## 2026-10-27 - H8, turns judged by their layers' medians (result: kept)

`./pktito-c bench`, test.wav through v2, best of 5 in seconds and the
threads each run kept, the order reversed every other round; then more
default runs of each, threads only. Every build's --json of test.wav is
test.v2.json.

The GB10 (gcc 13.3), six rounds:

| run | threads | best of 5, s |
|---|---|---|
| H3 (c42738d), default | 10, 10, 10, 10, 20, 10 | 0.435, 0.436, 0.438, 0.441, 0.587, 0.438 |
| H8, default | 10 in all six | 0.436, 0.440, 0.436, 0.446, 0.440, 0.440 |
| -t 10 | | 0.433, 0.432, 0.441, 0.435, 0.443, 0.440 |
| -t 20 | | 0.573, 0.567, 0.575, 0.571, 0.590, 0.573 |

Thirty more of each: H8 kept the ten in 29, H3 in 24. In all, 35 of 36
and 29 of 36.

The M4 Pro (Apple clang 21, in use, load 2.4 rising to 11), six rounds:

| run | threads | best of 5, s |
|---|---|---|
| H3 (c42738d), default | 8, 8, 8, 12, 12, 12 | 0.443, 0.466, 0.470, 0.472, 0.475, 0.477 |
| H8, default | 8, then 12 in five | 0.442, 0.451, 0.463, 0.472, 0.474, 0.476 |
| -t 8 | | 0.442, 0.450, 0.485, 0.490, 0.502, 0.502 |
| -t 12 | | 0.450, 0.450, 0.473, 0.475, 0.478, 0.476 |

-t 8 led by 1.8% in the first round, where H8 took 8, and was level in
the second; -t 12 led in the last four, by 2.5 to 5.2%, as the load rose.
H8's default was level with or ahead of the faster count in every round.
Thirty more of each: H8 took 12 in all thirty, H3 8 in 29.

The RG477V (on battery, 28-29 C throughout), five rounds:

| run | threads | best of 5, s |
|---|---|---|
| H3 (c42738d), default | 4, 4, 8, 4, 8 | 3.625, 3.562, 3.422, 3.800, 3.645 |
| H8, default | 4 in all five | 3.339, 3.636, 3.571, 3.935, 3.577 |
| H3's build, -t 4 | | 3.326, 3.327, 3.961, 3.729, 3.638 |
| H8's build, -t 4 | | 3.554, 3.549, 3.602, 3.789, 3.560 |
| -t 8 | | 3.958, 3.789, 3.949, 3.819, 3.789 |

-t 4 led -t 8 in every round, by 2.4 to 16%. Ten more of each: H8 kept
the four in six, H3 in eight; over fifteen, eleven each. H3 had kept the
four in 10 of 15 in H6's runs and 4 of 15 in H7's: on this board the count
moves more from one session to the next than between the two rules.

The Googlebook (2026-10-25, above; on AC, load 2.7 before the runs),
five rounds:

| run | threads | best of 5, s |
|---|---|---|
| H3 (c42738d), default | 8 in all five | 0.619, 0.782, 0.665, 0.757, 0.639 |
| H8, default | 12 in all five | 0.583, 0.580, 0.579, 0.809, 0.882 |
| H3's build, -t 8 | | 0.644, 0.655, 0.813, 0.640, 0.692 |
| H8's build, -t 8 | | 0.658, 0.788, 0.650, 0.756, 0.648 |
| -t 12 | | 0.602, 0.582, 0.594, 0.762, 0.823 |

Ten more of each: H8 took 12 in all ten, H3 8 in all ten. -t 12 led by
6.5 to 11% in the first three rounds and -t 8 by 16 and 21% in the last
two: every twelve-thread run took 0.58-0.60 s in rounds 1-3 and 0.76-0.88
in 4-5, the eight-thread runs 0.62-0.81 throughout, while the battery
went from 28 to 33 C. Each rule was right in the rounds that favoured its
side, H8 in three and H3 in two; the board's answer changed within
minutes, and a race run once, in a process's first seconds, follows
neither.

The Galaxy S25 Ultra (SM-S938U1, Snapdragon 8 Elite: six Oryon cores at
cpu_capacity 765, two at 1024; NDK r30's clang 21, on battery, 29 C
rising to 34), five rounds:

| run | threads | best of 5, s |
|---|---|---|
| H3 (c42738d), default | 8 in all five | 1.643, 1.705, 1.251, 1.930, 1.494 |
| H8, default | 8 in all five | 1.336, 1.606, 1.789, 1.624, 1.755 |
| H3's build, -t 2 | | 2.145, 4.547, 2.179, 2.223, 2.390 |
| H8's build, -t 2 | | 3.818, 4.091, 2.447, 2.204, 2.794 |
| -t 8 | | 1.687, 1.217, 1.231, 1.176, 2.297 |

Ten more of each: both kept the eight in all ten. -t 8 led -t 2 in every
round, by 4 to 70%, on a board whose -t 8 runs alone spread from 1.18 to
2.30 s.

snappy (the next entry), by the pktito-windows session: H8 cross-built as
the others, its --json test.v2.json; `bench` of long10.wav, three rounds
against 79af521 (H4), the order alternating. Both kept 12 in all six
runs, where H3 had kept 8. Best of 5: H8 4.191, 4.887, 4.763; H4 4.890,
4.661 and 8.425, a run of 43 s where the others took about 25, every
quiet check passed. Level.

Kept. H8 chose the faster side wherever one side led steadily: the
GB10's X925s in 35 of 36 runs, one short of the prediction, snappy's
twelve and the S25's eight in every run, and on the M4 Pro, where the
load moved the answer between 8 and 12, a default level with or ahead of
the faster count in every round. H3 did as well on the S25 and the
RG477V, and worse on the GB10, the M4 Pro and snappy. Two boards stay
open. The RG477V's four A715s lead its eight cores by about 7% at the
race's 348 frames, and the board stops threads for three to ten times a
layer's time about one layer in seven: a median of twelve layers a side
kept the four in 11 of 15 runs. The Googlebook's answer changed within
minutes, which a race run once cannot follow. The race costs one
recording's twelve layers on the slower side: on the S25, twelve layers
on two cores. check-c passes on the 9950X (gcc 15.2), the GB10 (gcc 13.3)
and the M4 Pro (Apple clang 21), check-tsan on the GB10 and the M4 Pro.

## 2026-10-27 - the power-throttling opt-out on snappy (a measurement)

snappy: a Snapdragon X Elite, X1E78100, twelve identical Oryon cores,
which Windows classes 0 (cpus 0-3) and 1 (cpus 4-11); Windows 11 Home
26200, on AC, Best performance. Run by the pktito-windows session over
ssh, so in session 0, the session locked and its screen off throughout.
The builds were armlet's (above), each one's --json of test.wav
test.v2.json byte for byte. `bench -m v2.pkf long10.wav`, test.wav ten
times over (600 s), so that each timed pass outlasts the few seconds
after which Windows packs a throttled process; default threads, three
rounds with the order rotating and a fourth that read each busy thread's
ThreadPowerThrottling 5 s in. Before each run, a check that no other
process had used 1.0 s of CPU in 3 s; every cpu's busy share sampled once
a second.

| build | threads | best of 5, s | cpus busy, 0-3 / 4-11 | the busy threads |
|---|---|---|---|---|
| 98a54d1, before the opt-out | 8 | 9.01, 11.40, 11.36, 8.92 | 3.9 / 0.1 | all 8 system-managed (ControlMask 0) |
| c42738d, the opt-out, H3's race | 8 | 5.79, 5.96, 6.24, 5.85 | 2.0 / 5.8-5.9 | all 8 ControlMask 1, StateMask 0 |
| 79af521, H4's race | 12 | 4.94, 4.44, 4.73, 4.69 | 3.6 / 7.3-7.4 | all 12 opted out |

About 3 s into each pass, Windows had all of 98a54d1's pool on cpus 0-3.
The opt-out spread c42738d's over the twelve, 1.72 times faster (medians
10.19 and 5.91 s), and H4's race kept twelve threads where H3's kept the
eight Windows classes 1, a further 1.25 times (4.71 s). 98a54d1 predates
both the workers' opt-out and the caller's, so these runs and armlet's
compare none against all. Every run here was in session 0 with the
session locked and the screen off, so which of those put the process
under EcoQoS is not separated; core/os.c says what was measured.

## 2026-10-27 - the opt-out on snappy with the screen on (a measurement)

By the pktito-windows session, as in the row above but with Peter's
session unlocked and the screen held on: over ssh, so in session 0; a
process in his session asking for ES_DISPLAY_REQUIRED alone, pktito's
window closed; on AC (98%, Best performance), snappy warm from an
8-minute window test just before. One round of `bench -m v2.pkf
long10.wav`, default threads, a quiet check before each run:

| build | threads | best of 5, s | cpus busy, 0-3 / 4-11 | the busy threads |
|---|---|---|---|---|
| 98a54d1, before the opt-out | 8 | 8.992 | 3.9 / 0.1 | all 8 system-managed |
| c42738d, the opt-out | 8 | 6.412 | 2.0 / 5.8 | all 8 opted out |

The packing waits for neither the lock nor the screen: with the session
unlocked and the screen on, Windows held the windowless process's pool
to cpus 0-3 as before, and the opt-out spread it, 1.40 times faster in
this round. In the same window pktito-cli 0.9.2, without an opt-out,
stayed packed (a median of 11.30 s over four rounds of Alice chapter 1)
and 0.9.3, whose process opts out, used every core (5.57 s). In Peter's
desktop session the app, focused or minimised, never packed under
either version, on AC.

## 2026-10-27 - the day's builds, by their commits on main (a note)

The rows above name builds 98a54d1, c42738d and 79af521, the commits they
were cut from before the branch was rebased onto main. On main they are
1eb53f8, df52223 and 19dd301: the same code, but for main's make adb and
pk_wav_open_file beneath them, which change nothing these rows time.

## 2026-10-27 - snappy on battery (a measurement)

By the pktito-windows session, in the hold of the screen-on entry above
(Peter's session unlocked, the screen held on by ES_DISPLAY_REQUIRED
alone, Normal priority) but unplugged, in Windows' default power mode on
battery (Balanced, no DC overlay set). svchost and Defender were busy and
every quiet check failed, so the times are noisy; each is the run's own,
not a best.

pktito's app: six copies of Alice chapter 1 (1:04:04 of audio, v2)
handed to a fresh window that was then focused or minimised, two rounds
with the order alternating, timed from launch to all done; the cpus' busy
shares after the first 3 s.

| app | window | s | cpus busy, 0-3 / 4-11 |
|---|---|---|---|
| 0.9.2 | focused | 74.8, 85.9 | 3.4 / 2.8, 3.5 / 2.4 |
| 0.9.2 | minimised | 357.4, 400.2 | 3.9 / 0.1, 3.9 / 0.0 |
| 0.9.3, its process opted out | focused | 70.3, 69.5 | 3.4 / 3.0, 3.4 / 3.1 |
| 0.9.3, its process opted out | minimised | 75.2, 75.1 | 3.4 / 2.8, 3.4 / 2.9 |

On AC the app never packed, focused or minimised (the screen-on entry).
On battery 0.9.2's minimised window did, onto cpus 0-3, and took 4.7
times as long as focused (757.6 s over the two rounds against 160.7).
0.9.3's process-level opt-out, its masks 1 and 0 through each job, took
that away: minimised within 8% of focused (150.3 s against 139.8).

Over ssh in the same hold, pktito-cli on chapter 1: 0.9.2 packed (4.0 /
0.0) and took 51.92 s, 4.6 times its 11.30 s median on AC with the screen
on, in one run whose quiet check had failed (svchost 2.55 s of CPU in
3 s). 0.9.3 took 9.89 and 12.19 s, against 5.57 s on AC, its 12 threads
on cpus 0-7 alone: 84, 87, 82, 80, 73, 72, 69 and 68% busy in the first
run, cpus 8-11 at 0-1% in both. Windows had parked the third cluster;
the app's unpacked runs were as narrow, about 6.3 of the 12 cpus busy.
Not measured: the threads' opt-out alone on battery (98a54d1 against
c42738d, as on AC above), and fewer threads than 12 with cpus 8-11
parked. The logs are on snappy, in `C:\pktito-test\window` (`window\ac`
for AC) and `C:\pktito-test\ab-093\runs` (`bat*`).

## 2026-10-27 - the int8 product's loops, 32 bytes along (a measurement)

FCVTNS in the quantizer (backend/neon/CLIMB.md, its hypothesis) measured
on two Pi 5s, eight interleaved rounds each, test.wav through v2, best
of 5 at 4 threads: under gcc (12.2 on pi5c, 14.2 on pi5a) -0.32% and
-0.28%, under clang 21 +5.24% and +4.81%. The control, 1120 bytes of
uncalled code ahead of core/cli.c's, moved what follows it by 1152 bytes
and the runs by +0.07% and -0.26% under clang, +0.05% and -0.23% under
gcc. In the clang build FCVTNS's instructions were as meant: four FCVTNS
where base has four FRINTNs and four FCVTZSs. And k_quantize shrank by
16 bytes, which moved k_gemm_q8 behind it. clang 21 aligns a function to
16 bytes and a loop to 32 where that costs at most 16
(`.p2align 5, , 16`), gcc 15 the same (`.p2align 5,,15`), in a .text
that is 32-aligned. Re-padded at its new place, every SDOT loop in
k_gemm_q8 sat 32 bytes along modulo 64, the same instructions; the
control's 1152 bytes are a multiple of 64.

The same code moved by exactly 32 bytes, by linking a 32-byte, 32-aligned
object ahead of backend/neon-dp/kernels.o: pi5c, six interleaved rounds
as above, static builds from clang 21.1.8 and gcc 15.2 (cross-compiled,
-mtune=cortex-a76), each build's --json test.v2.json byte for byte, not
throttled.

| build | clang 21 | gcc 15 |
|---|---|---|
| base | 3.845 s | 3.725 s |
| base, 32 bytes along | +6.21% (+5.67 to +6.46) | +7.91% (+7.81 to +7.95) |
| FCVTNS | +5.08% (+4.33 to +5.74) | +0.01% (-0.08 to +0.13) |
| FCVTNS, 32 bytes along | -0.08% (-0.41 to +0.08) | -0.01% (-0.11 to +0.19) |

gcc's tile is a clone, q8_cols.constprop.0. FCVTNS moved it 16 bytes,
which gcc's re-padding absorbed. Base moved 32 bytes cost 7.9%. Under
clang, FCVTNS 32 bytes along put its loops back where base's are, and the
5% went with them. `make PROFILE=1` on the clang pair, three runs each,
medians:

| stage | base, ms | FCVTNS, ms | |
|---|---|---|---|
| quantize | 218.6 | 203.2 | -7.0% |
| feed-forward | 1889.5 | 2054.7 | +8.7% |
| q k v pos | 576.9 | 600.3 | +4.1% |
| attention out | 118.5 | 129.6 | +9.4% |
| conv pw1 pw2 | 349.4 | 371.0 | +6.2% |
| attention | 295.1 | 295.0 | 0.0% |
| decoder | 140.3 | 138.8 | -1.1% |

FCVTNS's own effect is the quantize stage's 15 ms, 0.4% of the run; the
rest is the int8 products, the four stages that are 76% of it. So the
place a compiler gives the product's loops is worth 6 to 8% of a Pi 5
run under either compiler, and it moves with code nothing in the product
calls. Every function at a 64-byte boundary (-falign-functions=64) pins
that much: the product was then the same bytes under base, FCVTNS and
the softmax change. It does not pin what the product's own code does to
its loops. The activations' stride (backend/neon/CLIMB.md) changes only
the tile's setup, and put the 4-column loop at 36 modulo 64, where base
has it at 4. Each compiler, at the place it gave the product: clang
3.845 s, gcc 15 3.725 s.

## 2026-10-27 - the SDOT tile's block loop in assembly, at a 64-byte line (hypothesis)

Every product in the encoder is the 4-column SDOT tile at 32-input
blocks. Both compilers make it a loop over blocks around a loop of two
16-input chunks, since qb is a run-time value. gcc 15 also spills a
counter to the stack in the block loop and works out each chunk's
addresses again. The change writes that block loop in inline assembly.
Per block: the two chunks written out, 16 weight vectors in 8 LDPs, 8
activation loads, 64 SDOTs by lane over the 8 int accumulators in gcc's
order (each weight vector against the four columns in turn); then 8
SCVTFs, the four columns' scales into one vector's lanes, and 8 FMLAs by
element. The loop's head is at a 64-byte line (`.p2align 6`).

The bits are ref's. A block's integer sums are exact, its sum is exact
as a float (32 products of at most 128 * 127, under 2^24), and an FMLA
by element is the fused multiply-add vfmaq_f32 is, block after block,
ref's chain. The rest stays C: tiles of 1 to 3 columns, other block
sizes (the decoder's), and each output's scale, bias and add. A64's
instructions are all 4 bytes, so the loop is the same bytes and the same
place under every compiler, whichever registers each gives the pointers.
Inline rather than a .S file leaves the function's prologue, CFI, BTI
landing pads and each platform's symbol and section rules to the
compiler. GNU inline asm is already in the tree (core/os.c,
core/pool.c), so no compiler is newly required.

Predict: kcheck's bits unchanged. On pi5c, gcc 15 and clang 21 within
the spread of each other, and both at gcc 15's 3.725 s or faster: the
product stages a few % faster for the chunks written out and the spill
gone. The loop's head at 32 modulo 64 (a pad for the measurement only)
slower by a measurable share if the A76's sensitivity is the line, and
within the spread if it was the chunk loop's branch. No loss on the M4
Pro or the GB10's X925s, which run the same tile.

## 2026-10-27 - the SDOT block loop in assembly (result: slower as written), and its scales a register each (hypothesis)

The block loop above (scratch tree; kcheck ref's bits on the M4 Pro,
1779 cases, and a copy with one FMLA's lane wrong fails a 4-column qb-32
case, so the cases reach it), pi5c, six interleaved rounds, test.wav
through v2, best of 5 at 4 threads, each build's --json test.v2.json byte
for byte, not throttled:

| build | clang 21 | gcc 15 |
|---|---|---|
| base | 3.845 s | 3.726 s |
| the loop in assembly | 3.886 s, +1.13% (+0.96 to +1.25) | 3.893 s, +4.51% (+4.21 to +4.56) |
| the same, its head 32 bytes along | 3.884 s, +1.07% | 3.896 s, +4.56% |

It took the compiler out of it: the two builds were within 0.2% of each
other, and the head 32 bytes along changed nothing. But it is slower than
both compilers' C. One difference from both: it gathers the four
columns' scales into one register's lanes (LD1 {v12.s}[c]), and each lane
insert reads the register the last one wrote, a chain of four loads a
block carried into the next. gcc 15 and clang 21 load each scale into a
register of its own (LDR s) and FMLA by its lane 0.

Hypothesis: the same loop with the scales in v12 to v15, one LDR each.
The chain goes, and the loop reaches gcc 15's 3.726 s or better under
both compilers; the bits are the same, each scale the same float.

## 2026-10-27 - the SDOT block loop in assembly, its scales a register each (result: kept)

The loop of the entry before, with each column's scale loaded into one
register of its own (LDR s12 to s15) and each FMLA taking lane 0 of its
own. kcheck ref's bits on every board below under each of its
compilers, and check-c's transcripts ref's on the M4 Pro (Apple clang),
the GB10 (gcc 13.3 and clang 18.1) and pi5c (its gcc 12.2). Interleaved
rounds, test.wav through v2, best of 5 at the default threads, each
build's --json test.v2.json byte for byte:

| board | compiler | base | the loop in assembly | |
|---|---|---|---|---|
| pi5c, Pi 5, 4 threads, 6 rounds | clang 21.1 | 3.842 s | 3.657 s | -4.82% (-4.94 to -4.69) |
| | gcc 15.2 | 3.726 s | 3.671 s | -1.46% (-1.56 to -1.31) |
| GB10, 10 X925s, 6 rounds | clang 18.1 | 0.426 s | 0.429 s | +0.82% (-0.94 to +1.17) |
| | gcc 13.3 | 0.441 s | 0.431 s | -2.27% (-3.17 to -1.37) |
| M4 Pro, 12 threads, 6 rounds | Apple clang 21.0 | 0.456 s | 0.444 s | -2.23% (-3.36 to +1.73) |

pi5c's builds were cross-compiled static (clang 21.1.8, gcc 15.2); the
GB10's and the M4 Pro's were built on the board. The M4 Pro was in use (a
load of 3, Chrome and the Claude app about 1.3 cores), and its one round
the other way was +1.7%. The chain was the loss: with it gone the loop
beats both compilers' C on the A76, and the two builds of it are within
0.4% of each other (clang's 3.657 s, gcc's 3.671 s) where the C tile's
were 3.2% apart; on the GB10 the gap between gcc and clang went from 3.5%
to 0.5%. The head 32 bytes along was measured on the first version only,
where it changed nothing. The loop stays at its 64-byte line, which
costs at most 15 NOPs a tile.

Kept: the 4-column, 32-input-block tile of neon-dp runs this loop; tiles
of 1 to 3 columns and other block sizes keep the C. FCVTNS, the softmax
at sixteen and the activations' stride (backend/neon/CLIMB.md) are to be
measured again on top of it.
