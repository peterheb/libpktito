# backend/neon-dp: the climb

Dated records, appended; rejected hypotheses stay. A row's numbers are true
of the board, compiler and tree it names.

## 2026-10-20 - the baseline, on an M4 Pro (no Pi 5 up yet)

Apple M4 Pro, 8 performance and 4 efficiency cores, macOS 27, Apple clang
21.0, `-mcpu=apple-m1 -O3 -ffp-contract=off`. `./pkito bench` on Silero's
test.wav (60 s) whole, `--max-chunk 60`: 750 encoder frames, 232 tokens,
best of five.

| tree                                                      | threads | best    | real time |
| --------------------------------------------------------- | ------- | ------- | --------- |
| 2b035b3, first SDOT kernel (8x4 tile, no column blocking) | 12      | 0.543 s | 110.6x    |
| 2997d4f, column blocks of 192 KB                          | 4       | 1.008 s | 59.5x     |
|                                                           | 8       | 0.512 s | 117.3x    |
|                                                           | 12      | 0.523 s | 114.7x    |

Column blocking moved nothing here (0.543 s at 12 threads before, 0.523 s
after, inside the spread): the M4's 16 MB L2 per cluster holds a 750-column
product's activations (3 MB at 4096 inputs) whole. It went in for the
12900HK and the A53, whose L2s do not (backend/avx2/CLIMB.md,
backend/neon/CLIMB.md). The efficiency cores add nothing past eight
threads: dynamic dealing gives them items, and they finish them late.

The stage split at eight threads (`make PROFILE=1`, 0.509 s):

| stage                                                  | ms    | share |
| ------------------------------------------------------ | ----- | ----- |
| feed-forward (4 int8 products a layer, SiLU)           | 234.1 | 44.3% |
| attention, float (scores, positions, softmax, context) | 114.3 | 21.7% |
| q k v and the positions' projection                    | 71.5  | 13.5% |
| convolution module's two products                      | 41.7  | 7.9%  |
| decoder                                                | 16.2  | 3.1%  |
| attention output                                       | 14.7  | 2.8%  |
| subsampling                                            | 13.0  | 2.5%  |
| mel                                                    | 7.2   | 1.4%  |
| quantize, layer norms, GLU and depthwise, positions    | 15.3  | 2.9%  |

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

| test.wav           | best of 5 | real time                 |
| ------------------ | --------- | ------------------------- |
| whole (750 frames) | 4.934 s   | 12.2x                     |
| 30 s chunks (3)    | 4.371 s   | 13.7x, and 0.069 s of VAD |

The stage split, whole (`make PROFILE=1`):

| stage                                                    | ms   | share |
| -------------------------------------------------------- | ---- | ----- |
| feed-forward                                             | 1969 | 39.8% |
| attention products (scores, positions, softmax, context) | 891  | 18.0% |
| q k v and the positions' projection                      | 544  | 11.0% |
| conv pw1 pw2                                             | 338  | 6.8%  |
| subsampling                                              | 330  | 6.7%  |
| attention repack                                         | 285  | 5.8%  |
| quantize                                                 | 163  | 3.3%  |
| decoder                                                  | 128  | 2.6%  |
| attention out                                            | 122  | 2.5%  |
| glu + depthwise                                          | 80   | 1.6%  |
| layer norms                                              | 77   | 1.6%  |
| mel, positions                                           | 26   | 0.5%  |

The feed-forward is 302 G multiply-adds in 1.97 s: 38 G a core, 16 a cycle,
half of SDOT's 32 (two 128-bit pipes). The attention's products run at
about 3.7 FMAs a cycle of FMLA's 8. Three things stand out against the M4,
whose split this resembles otherwise:

- the repack is 5.8%, for moving 18 MB a layer: K's and P's panels are a
  transpose, written a float at a time with each store a cache line apart
  (core/encoder.c, repack_item);
- the subsampling is 6.7%, where its arithmetic (7 G int8 multiply-adds and
  0.3 G float) should take about 80 ms: its 3x3 convolutions run channel
  group outermost, so a block's 2.3 MB of output is walked 64 times;
- quantize is 3.3%, for one pass over each product's input.

Each is a kernel or a loop order; none changes a bit.

## 2026-10-20 - the attention's repack

Same Pi 5, compiler and flags; four threads, best of five, check-c passing
for each row (every transcript byte for byte, so none changes a bit).

| tree                                                                                                                                 | whole   | 30 s chunks | repack | attention |
| ------------------------------------------------------------------------------------------------------------------------------------ | ------- | ----------- | ------ | --------- |
| 34ecaaa, the baseline above                                                                                                          | 4.934 s | 4.371 s     | 285 ms | 891 ms    |
| K's, V's and P's panels written a dimension's 16 keys at a time                                                                      | 4.863 s |             | 192 ms |           |
| no repack: V into panels by its product, K and P as written, each block's scores and position products transposed instead (rejected) | 5.064 s | 4.221 s     | 0      | 1293 ms   |
| V into panels by its product; K and P transposed 16 keys at a time; the queries plus biases made a block at a time in att_item       | 4.741 s | 4.169 s     | 103 ms | 867 ms    |

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

| tree                                                                                       | whole                 | 30 s chunks | subsampling      |
| ------------------------------------------------------------------------------------------ | --------------------- | ----------- | ---------------- |
| c00013c                                                                                    | 4.741 s               | 4.169 s     | 343 ms           |
| pk_k_conv3x3s2 a row at a time, 4 channels by 8 columns at once                            | 4.601 s               | 4.026 s     | 197 ms           |
| and the first stage made a window at a time, the 3 rows the second's next output row reads | 4.537 s               |             | 115 ms           |
| windows for 2, 3 and 4 output rows instead (rejected)                                      | 4.547, 4.567, 4.585 s |             | 129, 150, 165 ms |
| every stage a row at a time, each keeping the 3 rows the next reads                        | 4.505 s               | 3.920 s     | 90 ms            |

An instrumented build (not committed) timed each stage's calls inside the
items, summed over the threads, a run:

| ms of thread time                 | conv0 | dw1 | pw1 | dw2 | pw2 |
| --------------------------------- | ----- | --- | --- | --- | --- |
| c00013c, 4 threads                | 510   | 562 | 130 | 64  | 32  |
| the new kernel, 4 threads         | 229   | 261 | 141 | 39  | 32  |
| the new kernel, 1 thread          | 79    | 69  | 92  | 12  | 22  |
| first stage in windows, 4 threads | 82    | 78  | 142 | 30  | 28  |
| first stage in windows, 1 thread  | 70    | 44  | 91  | 11  | 22  |

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

| build                                                                                                 | feed-forward | attention | minute        |
| ----------------------------------------------------------------------------------------------------- | ------------ | --------- | ------------- |
| -O3                                                                                                   | 1977 ms      | 870 ms    | 4.501-4.510 s |
| -O3 -fno-schedule-insns, everything                                                                   | 2270 ms      | 728 ms    | 4.812-4.819 s |
| -O3, the inputs two at a time in f32_tile (no spill, but register moves at every back edge; rejected) | 1975 ms      | 921 ms    | 4.555 s       |
| -O3, the pass off for the float products only (pragma)                                                | 1978 ms      | 728 ms    | 4.358 s       |

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

|                           | feed-forward | quantize | minute        | 30 s chunks |
| ------------------------- | ------------ | -------- | ------------- | ----------- |
| b325cd9                   | 1978 ms      | 163 ms   | 4.358 s       | 3.847 s     |
| SiLU in the quantize pass | 1827 ms      | 213 ms   | 4.244-4.248 s | 3.791 s     |

(The SiLU's arithmetic moved from the feed-forward's row of the profile
to quantize's.) The 12900HK and the M4 Pro did not move, inside their
spreads: their caches and memory keep up either way.

## 2026-10-20 - the layer norms into their quantize

Same Pi 5. Four of a layer's five layer norms (both feed-forwards', the
attention's, the convolution module's) wrote [T][1024] floats, 3 MB at a
minute, that only the quantize after them read. Each now goes a frame at
a time through the same row of thread scratch as the SiLU, norm then
quantize; the last norm, whose output is the residual stream, stays.

|                              | layer norms | quantize | minute        | 30 s chunks |
| ---------------------------- | ----------- | -------- | ------------- | ----------- |
| cfaf43a                      | 77 ms       | 213 ms   | 4.244-4.248 s | 3.791 s     |
| norms in the quantize passes | 13 ms       | 219 ms   | 4.186-4.189 s | 3.737 s     |

(The four norms' arithmetic is in quantize's row now.) The 12900HK and
the M4 Pro did not move outside their spreads.

## 2026-10-20 - backend/neon's A53 climb merged, and a block of 32 unrolled

The A53 climb (backend/neon/CLIMB.md) changed the int8 product's shared
code: the column-block loop moved into q8_cols, and q8_tile took a range
of blocks and a partial-sum buffer for ARMv8.0's chunks of 1024 inputs.
For SDOT the tile stays 8 x 4, the column blocks 192 KB, and the chunks are
compiled out. Same Pi 5, merged tree:

|         | minute        | 30 s chunks   |
| ------- | ------------- | ------------- |
| 1aab324 | 4.186-4.189 s | 3.737 s       |
| merged  | 4.155-4.162 s | 3.714-3.718 s |

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

| threads                                     | on                  | best          | real time  |
| ------------------------------------------- | ------------------- | ------------- | ---------- |
| 20, the default (a core of every kind each) | any                 | 0.711-0.719 s | 83.4-84.3x |
| 16                                          | any                 | 0.896 s       | 67.0x      |
| 12                                          | any                 | 0.645 s       | 93.0x      |
| 10                                          | any                 | 0.442 s       | 135.8x     |
| 8                                           | any                 | 0.497 s       | 120.7x     |
| 5                                           | any                 | 0.715 s       | 84.0x      |
| 10                                          | the X925s (taskset) | 0.437 s       | 137.4x     |
| 8                                           | the X925s           | 0.474 s       | 126.7x     |
| 5                                           | the X925s           | 0.688 s       | 87.2x      |
| 10                                          | the A725s           | 1.267 s       | 47.4x      |

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

| stage           | 10 threads | 20 threads |
| --------------- | ---------- | ---------- |
| decoder         | 16.9 ms    | 183.2 ms   |
| quantize        | 53.2 ms    | 105.8 ms   |
| q k v pos       | 59.1 ms    | 84.1 ms    |
| glu + depthwise | 15.4 ms    | 24.1 ms    |
| layer norms     | 3.1 ms     | 11.4 ms    |
| attention out   | 11.9 ms    | 16.5 ms    |
| feed-forward    | 186.6 ms   | 193.6 ms   |
| bench           | 0.437 s    | 0.662 s    |

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

| board, threads                   | before (c51b433) | after         |
| -------------------------------- | ---------------- | ------------- |
| GB10, 20 (its default)           | 0.719-0.800 s    | 0.573-0.580 s |
| GB10, 10 (the X925s)             | 0.435-0.438 s    | 0.441-0.447 s |
| 12900HK, 14 (its default)        | 0.835-0.840 s    | 0.835-0.838 s |
| M4 Pro, 12 (its default; in use) | 0.515-0.529 s    | 0.498-0.522 s |

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

| board           | before                    | after         | the choice                                        |
| --------------- | ------------------------- | ------------- | ------------------------------------------------- |
| GB10            | 0.570-0.581 s, 20 threads | 0.437-0.443 s | 10, the X925s, every run and every `-v`           |
| 12900HK         | 0.836-0.840 s, 14         | 0.838-0.841 s | 14, every core, every run                         |
| M4 Pro (in use) |                           | 0.466 s once  | 8 in bench, 12 in one `-v`: 4% apart, near the 3% |

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
(the GB10's X925 and A725), and Apple's M2 and later have it; the A76 (Pi 5) and the M1 do not. Wherever SMMLA issues as often as SDOT, the inner
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

| board                 | core          | neon-dp      | neon-i8mm              |             | q8bench                                |
| --------------------- | ------------- | ------------ | ---------------------- | ----------- | -------------------------------------- |
| c6g.xlarge            | Neoverse N1   | 18.9x        | (no I8MM: passed over) |             |                                        |
| c7g.xlarge            | Neoverse V1   | 24.5x        | 26.3x                  | +7.3%       | 81.3 to 85.8                           |
| c8g.xlarge            | Neoverse V2   | 30.7x        | 32.0x                  | +4.2%       | 103.8 to 110.3                         |
| c9g.xlarge            | Neoverse V3   | 44.3x        | 45.9x                  | +3.6%       | 126.5 to 133.1                         |
| GB10, -t 10           | X925 and A725 | 136.3-137.9x | 138.3-139.8x           | +1.4%       | X925 202-204 to 209, A725 61.0 to 63.9 |
| GB10, its X925s alone | X925          | 135.6-136.9x | 137.1-137.9x           | +0.8%       |                                        |
| GB10, -t 20           | X925 and A725 | 105.0-108.1x | 104.7-107.4x           | level       |                                        |
| M4 Pro                | M4            | 131.9-134.1x | 113.4-115.7x           | -12 to -15% | 182-186 to 148                         |

(six columns on the M4 Pro: 117.0 and 118.4x, the product 155.)

mmrate2.c, ns an instruction with 1, 4, 8 and 16 independent chains, one
thread (1 is the latency; 16 the issue rate where the latency allows):

| core                     | SDOT                          | SMMLA                           |
| ------------------------ | ----------------------------- | ------------------------------- |
| Neoverse V2 (2.8 GHz)    | 0.36, 0.17, 0.11, 0.097       | 0.36, 0.17, 0.10, 0.097         |
| X925 (GB10)              | 0.43, 0.12, 0.066, 0.047      | 0.44, 0.12, 0.065, 0.047        |
| A725 (GB10)              | 0.357, 0.179, 0.179, 0.179    | the same                        |
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

| loop                                                                                                                             | int8 multiply-adds a second                     |
| -------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| SDOT, 16 chains, 1 / 2 / 4 / 8 threads                                                                                           | 240-259 G / 497-504 G / 908-910 G / 1713-1804 G |
| SMOPA (16 x 16 int32, 1024 a SMOPA), 4 tiles, 1 / 2 / 4 / 8 threads                                                              | 1984 G / 1957 G / 2254-2364 G / 3812-3829 G     |
| SMOPA, 8 a 32-input block, each tile then drained: rows to Z four at a time, SCVTF, FMLA, zeroed                                 | 498 G                                           |
| the same with SME2's multi-vector forms (vgx4 array reads, SCVTF x4, FMLA vgx4 into a float tile), one scale vector or one a row | 620-633 G                                       |
| one tile filled while the other is drained (double-buffered), vgx4                                                               | 728-729 G                                       |

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

| stage             | c9g (neon-dp) | c8a (avx512-vnni) | c9g / c8a |
| ----------------- | ------------- | ----------------- | --------- |
| feed-forward      | 2508.2        | 825.6             | 3.04      |
| q k v pos         | 761.7         | 260.6             | 2.92      |
| conv pw1 pw2      | 460.8         | 150.9             | 3.05      |
| attention (float) | 437.4         | 150.6             | 2.90      |
| quantize          | 259.0         | 62.9              | 4.12      |
| attention out     | 157.8         | 50.2              | 3.14      |
| subsampling       | 126.3         | 46.3              | 2.73      |
| glu + depthwise   | 54.3          | 23.0              | 2.36      |
| decoder           | 39.0          | 25.5              | 1.53      |
| mel               | 17.5          | 18.4              | 0.95      |
| the run           | 4854          | 1631              | 2.98      |

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

| core               | SDOT                                               | SMMLA                                              |
| ------------------ | -------------------------------------------------- | -------------------------------------------------- |
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

| run                         | 1     | 2     | 3     | 4     |
| --------------------------- | ----- | ----- | ----- | ----- |
| before, default: 4 threads  | 1.029 | 1.507 | 1.095 | 1.256 |
| after, default: 8 each time | 0.677 | 1.243 | 0.876 | 0.665 |
| after, -t 4                 | 1.086 | 2.098 | 1.741 | 1.310 |
| after, -t 8                 | 0.694 | 1.231 | 1.223 | 1.205 |
| after, -t 12                | 0.665 | 0.917 | 1.183 | 1.186 |

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

| run                   | threads             | best of 5, s                             |
| --------------------- | ------------------- | ---------------------------------------- |
| H3 (c42738d), default | 8, 8, 8, 12, 12, 12 | 0.474, 0.480, 0.484, 0.484, 0.485, 0.485 |
| H4, default           | 12 in all six       | 0.460, 0.463, 0.473, 0.486, 0.490, 0.486 |
| -t 8                  |                     | 0.443, 0.615, 0.496, 0.508, 0.513, 0.513 |
| -t 12                 |                     | 0.494, 0.497, 0.482, 0.482, 0.484, 0.485 |

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

| run                    | 1             | 2             | 3             | 4             | 5             | median        |
| ---------------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| default (8, every run) | 6.289 / 6.368 | 5.778 / 6.267 | 6.937 / 6.065 | 5.702 / 5.181 | 6.993 / 5.284 | 6.289 / 6.065 |
| -t 8                   | 6.324 / 6.409 | 5.271 / 5.977 | 6.010 / 5.835 | 6.522 / 5.346 | 5.418 / 6.948 | 6.010 / 5.977 |
| -t 2                   | 7.260 / 8.537 | 7.249 / 7.187 | 8.119 / 8.118 | 7.261 / 8.147 | 7.926 / 8.134 | 7.261 / 8.134 |

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

| run                   | threads                | best of 5, s                             |
| --------------------- | ---------------------- | ---------------------------------------- |
| H3 (c42738d), default | 10 in all six          | 0.438, 0.438, 0.441, 0.443, 0.448, 0.447 |
| H4 (79af521), default | 10, 20, 10, 10, 20, 10 | 0.441, 0.576, 0.445, 0.445, 0.567, 0.446 |
| -t 10                 |                        | 0.436, 0.438, 0.448, 0.440, 0.440, 0.444 |
| -t 20                 |                        | 0.586, 0.578, 0.603, 0.600, 0.561, 0.577 |

H4 took the twenty in two runs of six and lost 30% in each, where H3
kept the ten every time. Ten processes of the logging build, both races
each: the second race's fastest pairs had the X925s 0.4 to 12% ahead, and
within the 3% margin once, where a steady pass's layers have them 14 to
46% ahead. Timed a layer at a time:

| frames | -t 10, a layer | -t 20, a layer | in the race, the fast kind's | in the race, every core's |
| ------ | -------------- | -------------- | ---------------------------- | ------------------------- |
| 150    | 3.4-4.7 ms     | 5.4-7.5        | 3.4-5.2                      | 4.6-8.9                   |
| 250    | 5.6-6.4        | 6.4-8.6        | 5.6-7.5                      | 5.9-8.1                   |
| 348    | 7.7-9.1        | 8.4-10.7       | 7.6-9.1                      | 7.8-10.4                  |

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

| run                   | threads                | best of 5, s                             |
| --------------------- | ---------------------- | ---------------------------------------- |
| H3 (c42738d), default | 10, 10, 20, 10, 10, 10 | 0.435, 0.442, 0.561, 0.447, 0.443, 0.449 |
| H6, default           | 10 in all six          | 0.444, 0.440, 0.445, 0.452, 0.448, 0.448 |
| -t 10                 |                        | 0.438, 0.443, 0.446, 0.447, 0.450, 0.447 |
| -t 20                 |                        | 0.573, 0.588, 0.592, 0.571, 0.577, 0.585 |

Thirty more default runs of each: H6 kept the ten in all thirty, H3 in 26.
Over the day's runs on the GB10, H3 took the twenty in 6 of 48, each about
30% slower (0.56-0.59 s against 0.44), and H6 in none of 36.

The M4 Pro (Apple clang 21, in use, load 2.4 rising to 8.9):

| run                   | threads             | best of 5, s                             |
| --------------------- | ------------------- | ---------------------------------------- |
| H3 (c42738d), default | 12, 8, 8, 12, 12, 8 | 0.456, 0.467, 0.474, 0.486, 0.487, 0.510 |
| H6, default           | 12 in all six       | 0.454, 0.453, 0.471, 0.484, 0.487, 0.487 |
| -t 8                  |                     | 0.442, 0.455, 0.485, 0.500, 0.509, 0.510 |
| -t 12                 |                     | 0.453, 0.453, 0.478, 0.480, 0.490, 0.488 |

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

| run                   | threads       | best of 5, s                      |
| --------------------- | ------------- | --------------------------------- |
| H3 (c42738d), default | 4, 8, 8, 4, 8 | 3.089, 3.613, 3.682, 3.299, 3.520 |
| H6, default           | 4, 8, 4, 8, 4 | 3.283, 3.533, 3.507, 3.769, 3.554 |
| H3's build, -t 4      |               | 3.240, 3.240, 3.372, 3.178, 3.306 |
| H6's build, -t 4      |               | 3.421, 3.116, 3.451, 3.423, 3.443 |
| -t 8                  |               | 3.619, 3.372, 3.549, 3.740, 3.808 |

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

| run                   | threads                | best of 5, s                             |
| --------------------- | ---------------------- | ---------------------------------------- |
| H3 (c42738d), default | 20, 20, 10, 20, 10, 10 | 0.569, 0.561, 0.439, 0.594, 0.447, 0.443 |
| H7, default           | 10 in all six          | 0.435, 0.437, 0.442, 0.442, 0.443, 0.443 |
| -t 10                 |                        | 0.437, 0.447, 0.447, 0.445, 0.443, 0.440 |
| -t 20                 |                        | 0.582, 0.592, 0.562, 0.593, 0.587, 0.568 |

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

| run                   | threads                | best of 5, s                             |
| --------------------- | ---------------------- | ---------------------------------------- |
| H3 (c42738d), default | 10, 10, 10, 10, 20, 10 | 0.435, 0.436, 0.438, 0.441, 0.587, 0.438 |
| H8, default           | 10 in all six          | 0.436, 0.440, 0.436, 0.446, 0.440, 0.440 |
| -t 10                 |                        | 0.433, 0.432, 0.441, 0.435, 0.443, 0.440 |
| -t 20                 |                        | 0.573, 0.567, 0.575, 0.571, 0.590, 0.573 |

Thirty more of each: H8 kept the ten in 29, H3 in 24. In all, 35 of 36
and 29 of 36.

The M4 Pro (Apple clang 21, in use, load 2.4 rising to 11), six rounds:

| run                   | threads             | best of 5, s                             |
| --------------------- | ------------------- | ---------------------------------------- |
| H3 (c42738d), default | 8, 8, 8, 12, 12, 12 | 0.443, 0.466, 0.470, 0.472, 0.475, 0.477 |
| H8, default           | 8, then 12 in five  | 0.442, 0.451, 0.463, 0.472, 0.474, 0.476 |
| -t 8                  |                     | 0.442, 0.450, 0.485, 0.490, 0.502, 0.502 |
| -t 12                 |                     | 0.450, 0.450, 0.473, 0.475, 0.478, 0.476 |

-t 8 led by 1.8% in the first round, where H8 took 8, and was level in
the second; -t 12 led in the last four, by 2.5 to 5.2%, as the load rose.
H8's default was level with or ahead of the faster count in every round.
Thirty more of each: H8 took 12 in all thirty, H3 8 in 29.

The RG477V (on battery, 28-29 C throughout), five rounds:

| run                   | threads       | best of 5, s                      |
| --------------------- | ------------- | --------------------------------- |
| H3 (c42738d), default | 4, 4, 8, 4, 8 | 3.625, 3.562, 3.422, 3.800, 3.645 |
| H8, default           | 4 in all five | 3.339, 3.636, 3.571, 3.935, 3.577 |
| H3's build, -t 4      |               | 3.326, 3.327, 3.961, 3.729, 3.638 |
| H8's build, -t 4      |               | 3.554, 3.549, 3.602, 3.789, 3.560 |
| -t 8                  |               | 3.958, 3.789, 3.949, 3.819, 3.789 |

-t 4 led -t 8 in every round, by 2.4 to 16%. Ten more of each: H8 kept
the four in six, H3 in eight; over fifteen, eleven each. H3 had kept the
four in 10 of 15 in H6's runs and 4 of 15 in H7's: on this board the count
moves more from one session to the next than between the two rules.

The Googlebook (2026-10-25, above; on AC, load 2.7 before the runs),
five rounds:

| run                   | threads        | best of 5, s                      |
| --------------------- | -------------- | --------------------------------- |
| H3 (c42738d), default | 8 in all five  | 0.619, 0.782, 0.665, 0.757, 0.639 |
| H8, default           | 12 in all five | 0.583, 0.580, 0.579, 0.809, 0.882 |
| H3's build, -t 8      |                | 0.644, 0.655, 0.813, 0.640, 0.692 |
| H8's build, -t 8      |                | 0.658, 0.788, 0.650, 0.756, 0.648 |
| -t 12                 |                | 0.602, 0.582, 0.594, 0.762, 0.823 |

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

| run                   | threads       | best of 5, s                      |
| --------------------- | ------------- | --------------------------------- |
| H3 (c42738d), default | 8 in all five | 1.643, 1.705, 1.251, 1.930, 1.494 |
| H8, default           | 8 in all five | 1.336, 1.606, 1.789, 1.624, 1.755 |
| H3's build, -t 2      |               | 2.145, 4.547, 2.179, 2.223, 2.390 |
| H8's build, -t 2      |               | 3.818, 4.091, 2.447, 2.204, 2.794 |
| -t 8                  |               | 1.687, 1.217, 1.231, 1.176, 2.297 |

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

| build                           | threads | best of 5, s             | cpus busy, 0-3 / 4-11 | the busy threads                     |
| ------------------------------- | ------- | ------------------------ | --------------------- | ------------------------------------ |
| 98a54d1, before the opt-out     | 8       | 9.01, 11.40, 11.36, 8.92 | 3.9 / 0.1             | all 8 system-managed (ControlMask 0) |
| c42738d, the opt-out, H3's race | 8       | 5.79, 5.96, 6.24, 5.85   | 2.0 / 5.8-5.9         | all 8 ControlMask 1, StateMask 0     |
| 79af521, H4's race              | 12      | 4.94, 4.44, 4.73, 4.69   | 3.6 / 7.3-7.4         | all 12 opted out                     |

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

| build                       | threads | best of 5, s | cpus busy, 0-3 / 4-11 | the busy threads     |
| --------------------------- | ------- | ------------ | --------------------- | -------------------- |
| 98a54d1, before the opt-out | 8       | 8.992        | 3.9 / 0.1             | all 8 system-managed |
| c42738d, the opt-out        | 8       | 6.412        | 2.0 / 5.8             | all 8 opted out      |

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

| app                          | window    | s            | cpus busy, 0-3 / 4-11 |
| ---------------------------- | --------- | ------------ | --------------------- |
| 0.9.2                        | focused   | 74.8, 85.9   | 3.4 / 2.8, 3.5 / 2.4  |
| 0.9.2                        | minimised | 357.4, 400.2 | 3.9 / 0.1, 3.9 / 0.0  |
| 0.9.3, its process opted out | focused   | 70.3, 69.5   | 3.4 / 3.0, 3.4 / 3.1  |
| 0.9.3, its process opted out | minimised | 75.2, 75.1   | 3.4 / 2.8, 3.4 / 2.9  |

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

| build                  | clang 21                | gcc 15                  |
| ---------------------- | ----------------------- | ----------------------- |
| base                   | 3.845 s                 | 3.725 s                 |
| base, 32 bytes along   | +6.21% (+5.67 to +6.46) | +7.91% (+7.81 to +7.95) |
| FCVTNS                 | +5.08% (+4.33 to +5.74) | +0.01% (-0.08 to +0.13) |
| FCVTNS, 32 bytes along | -0.08% (-0.41 to +0.08) | -0.01% (-0.11 to +0.19) |

gcc's tile is a clone, q8_cols.constprop.0. FCVTNS moved it 16 bytes,
which gcc's re-padding absorbed. Base moved 32 bytes cost 7.9%. Under
clang, FCVTNS 32 bytes along put its loops back where base's are, and the
5% went with them. `make PROFILE=1` on the clang pair, three runs each,
medians:

| stage         | base, ms | FCVTNS, ms |       |
| ------------- | -------- | ---------- | ----- |
| quantize      | 218.6    | 203.2      | -7.0% |
| feed-forward  | 1889.5   | 2054.7     | +8.7% |
| q k v pos     | 576.9    | 600.3      | +4.1% |
| attention out | 118.5    | 129.6      | +9.4% |
| conv pw1 pw2  | 349.4    | 371.0      | +6.2% |
| attention     | 295.1    | 295.0      | 0.0%  |
| decoder       | 140.3    | 138.8      | -1.1% |

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
as a float (32 products of at most 128 \* 127, under 2^24), and an FMLA
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

| build                             | clang 21                         | gcc 15                           |
| --------------------------------- | -------------------------------- | -------------------------------- |
| base                              | 3.845 s                          | 3.726 s                          |
| the loop in assembly              | 3.886 s, +1.13% (+0.96 to +1.25) | 3.893 s, +4.51% (+4.21 to +4.56) |
| the same, its head 32 bytes along | 3.884 s, +1.07%                  | 3.896 s, +4.56%                  |

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

| board                           | compiler         | base    | the loop in assembly |                         |
| ------------------------------- | ---------------- | ------- | -------------------- | ----------------------- |
| pi5c, Pi 5, 4 threads, 6 rounds | clang 21.1       | 3.842 s | 3.657 s              | -4.82% (-4.94 to -4.69) |
|                                 | gcc 15.2         | 3.726 s | 3.671 s              | -1.46% (-1.56 to -1.31) |
| GB10, 10 X925s, 6 rounds        | clang 18.1       | 0.426 s | 0.429 s              | +0.82% (-0.94 to +1.17) |
|                                 | gcc 13.3         | 0.441 s | 0.431 s              | -2.27% (-3.17 to -1.37) |
| M4 Pro, 12 threads, 6 rounds    | Apple clang 21.0 | 0.456 s | 0.444 s              | -2.23% (-3.36 to +1.73) |

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

## 2026-10-27 - the SDOT block loop on Windows, without its line (a build fix)

f13f54a did not build for Windows on Arm64: llvm-mingw's clang 23.1.4,
for aarch64-w64-mingw32, stopped on backend/neon-dp/kernels.c with
"fatal error: error in backend: Failed to evaluate function length in
SEH unwind info". Windows on Arm64 keeps an unwind record for every
function that saves registers or moves the stack, and the record holds
the function's length, which clang 23.1 does not work out across an
alignment inside the function. A test file shows it is any alignment: a
loop under -falign-loops=16, 32 or 64 fails the same way, and so does an
inline .p2align 3. A leaf function, which has no record, takes a
.p2align 6 inside it. -fno-unwind-tables builds, by dropping every
record, which a debugger's or a crash dump's walk through the kernels
needs: not that.

So on Windows the loop is written without its .p2align 6: the same
instructions, the head where the linker puts it, the object's .text
being aligned to 4 there - 16 bytes into a 64-byte line in pktito-c.exe,
on a line in tests/kcheck.exe. Elsewhere nothing changed: the neon and
neon-dp objects for Linux, under gcc 15.2 and clang 21.1, are byte for
byte f13f54a's.

Checked on armlet (Snapdragon 7c, Windows 11 on Arm), the programs
cross-built with llvm-mingw clang 23.1.4: kcheck ref's bits on neon-dp
and neon, 1779 cases each; check-c's other programs pass; test.wav's VAD
probabilities and its v2 and v3 transcripts are the files' bytes;
examples/transcribe gives the CLI's transcript; and a model and a WAV
named past ASCII go through the CLI's arguments. armlet has no shell, so
check-c's commands ran from a batch file, and the outputs were compared
on the 9950X. make check-c passes on pi5c (gcc 12.2).

Not measured: what the head's place costs on Windows. On a Pi 5 the
first version's head 32 bytes along changed nothing (above); the kept
version's was not tried. armlet's AliceBench runs spread by 13% today,
so a few percent there takes many interleaved rounds. A loop that saves
no registers and moves no stack - out of line, a leaf, clear of v8 to
v15 - would keep its line on Windows as well, with 24 registers to this
loop's 32: the weights two at a time and the four scales through two.
That is a different loop, to be measured as one.

## 2026-10-27 - the SDOT block loop in assembly on Android (a measurement, android-9950x's)

ebce152 against bf5abaa (the loop's parent, the C tile), measured by the
android-9950x session: NDK r30's clang 21 for API 24, both trees from git
archive, test.wav through v2, best of 5 at the default threads, 8
interleaved rounds with 45 s of rest before each; every build's --json
byte for byte the same.

| device                     | SoC                | the loop in assembly  |                                                                                                                                                        |
| -------------------------- | ------------------ | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Galaxy Tab S7 FE (SM-T733) | Snapdragon 778G    | -0.50% (-1.0 to -0.0) | base 3.216-3.243 s, new 3.205-3.231 s                                                                                                                  |
| RG477V                     | Dimensity 8300     | -4.8% (-10.9 to +4.5) | noisy; the fastest of each 3.161 s and 3.096 s (-2.1%)                                                                                                 |
| Galaxy S25 Ultra           | Snapdragon 8 Elite | no result             | 1.10 s or 1.6 to 2.2 s in both builds, which looks like where the threads were placed, not the kernel; the fastest of each 1.096 s and 1.104 s (+0.7%) |

Faster or level where a result could be had, as on the boards above.

## 2026-10-27 - the SDOT block loop in assembly on Android, the S25 Ultra on AC and a Googlebook (a measurement, android-9950x's)

The same pair as the entry above, ebce152 against bf5abaa, by the same
method and session: NDK r30's clang 21 for API 24, test.wav through v2,
best of 5 at the default threads, 8 interleaved rounds with 45 s of rest
before each, every build's --json byte for byte the same. Both devices on
AC, and each bench logged the phone's wake state: every one Awake.

| device                       | SoC                | the loop in assembly   |                                                                                                                                                                           |
| ---------------------------- | ------------------ | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Galaxy S25 Ultra (SM-S938U1) | Snapdragon 8 Elite | -6.5% (-14.8 to +59.3) | faster in 7 rounds of 8 (-2.5 to -14.8); round 7's +59.3% is one bench of the new build at 1.913 s; else base 1.194-1.377 s, new 1.138-1.204 s; the fastest of each -4.7% |
| XPS Googlebook               | arm64, 12 cores    | -0.86% (-2.1 to +1.0)  | faster in 7 rounds of 8; base 0.520-0.532 s, new 0.516-0.525 s; the fastest of each -0.8%                                                                                 |

The S25 Ultra's first run, 1.1 s or 1.6 to 2.2 s in both builds, was the
phone asleep on battery, not where the threads were placed. Awake, the
loop is faster there, and on the Googlebook, as on every other device
measured.

## 2026-10-28 - the int8 product on SME2's SMOPA with its loads, a scale a row (a measurement)

The 2026-10-24 entry measured SMOPA on registers. This one gives it its
loads, for the products that could take SMOPA's integer sums whole:
activations with one scale a column (qb = n), which tools/quantcheck.py
finds cost nothing measurable for q, k, v and the positions (README,
"Why int8"): paired against the engine's blocks of 32, dlogp +0.003 (t
+1.1) on v2 over LibriSpeech's 200 and -0.004 (t -1.1) on v3 over
FLEURS' 360. The engine itself built so (a scratch build of 33c5e7c,
tests/wercheck.py) keeps the same 4 of 200 and 45 of 360 transcripts
different from float32's as 33c5e7c, at WER 1.49% and 4.38% against
1.51% and 4.42%.

The probe, kept with the session's notes and not in the tree: the QKV
stage of a 20 s chunk, 24 layers of q, k and v (1024 x 1024 at T 256)
and the positions (1024 x 1024 at 512), each product with its own
weights so that they stream as the encoder's do; four ZA tiles, 32 rows
by 32 columns a step; the weights' panels as the file holds them (16
rows by 4 inputs, 64 bytes: SMOPA's operand as it is), the activations
packed 16 columns by 4 inputs (0.02 ms for 256 columns of 1024); each
tile drained once, at the end (MOVA vgx4, SCVTF x4, the column's scale,
the row's, the bias). Every output is ref's bits: 0 of 262,144
different from ref's formula. Apple clang 21, the M4 Pro in use (load
1.9 to 5.4), best of 7, multiply-adds a second:

| threads                                                    | the stage                                | SMOPA on registers alone |
| ---------------------------------------------------------- | ---------------------------------------- | ------------------------ |
| 1                                                          | 1331-1358 G                              | 2005-2009 G              |
| 2                                                          | 2497-2533 G                              | 4004-4012 G              |
| 4, from a shared counter                                   | 2647-2914 G (a static split: 1135)       | 1223-1250 G              |
| 8, from a shared counter                                   | 2298-2809 G (a static split: 1440, 2027) | 1424-1984 G              |
| 8: 1 SMOPA thread, the rest the tree's pk_k_gemm_q8 (SDOT) | 1765-1766 G                              |                          |
| 8: 2, 3 or 4 SMOPA threads, the rest SDOT                  | 2447-2466 G                              |                          |
| 8, all SDOT at blocks of 32 (33c5e7c's numerics)           | 1342-1430 G                              |                          |
| 8, all SDOT at a scale a row                               | 1262-1315 G                              |                          |

With its loads one thread gets two thirds of the unit's peak. The K loop
in assembly, the next four groups' loads issued ahead of this four's
SMOPAs, ran level with clang's (1331 against 1351 G), so the loads'
latency is not what is lost. Two threads used both clusters' units (4.0
T on registers). At four and eight, threads in streaming mode ran 3 to
12 times slower than the fastest (8.2 ms against 25 to 103 on
registers), sharing a unit or on the E cores', and a shared counter
takes most of that back. SDOT threads beside SMOPA's add nothing. So
SMOPA takes these products in about half the time eight cores of SDOT
take now, and the QKV stage, 77 ms of a 475 ms run at 8 threads (16.2%,
PROFILE, 2026-10-28), would give back about 7 to 8% of an M4 Pro run.

Not taken up yet. A scale a row needs every backend's tile fast at a
whole row first. Through this file's C tile an M4 core does 180-194 G
at a row, against 188 at 32 in the same code and 193-212 in the
assembly (q8bench, 1024 x 1024 at 256 columns): each accumulator is one
chain of SDOTs the row's length, where shorter blocks let the next
block's chains overlap, and two chains a sum would take it back at the
same bits. Then the SME2 variant, and a place to pack the activations: a
call takes a slice of rows and every column, so packing in the kernel
would pack them again for each slice.

## 2026-11-02 - SVE on the GB10's cores: its vector length and its instructions' rates (a measurement)

Nothing in the tree uses SVE. The GB10's cores have it: /proc/cpuinfo on
spark-a80d lists sve, sve2, svei8mm, svebf16 and i8mm on all twenty. Its
vector length is 16 bytes on both kinds, by PR_SVE_GET_VL and by RDVL on
CPU 5 (an X925) and CPU 0 (an A725), so SVE is NEON's width there, and
anything it gives must come from its instructions' forms or rates.

The probe (svrate, kept with the session's notes, not in the tree): each
instruction in C independent chains, the chains interleaved and unrolled
to at least 96 instructions a loop, pinned to one core, best of 7; the
clock from a chain of dependent adds, 3.889 GHz on the X925 and 2.800 on
the A725 (the performance governor, the vLLM worker idle). Ubuntu 24.04,
kernel 7.0.0-1019-nvidia; the loops are assembly, the same bytes under gcc
13.3 and clang 18.1. Cycles an instruction, one chain (the latency) and
the best rate a cycle:

| instruction                                               | X925, 1 chain | 8    | 16   | best a cycle | A725, 1 chain | best a cycle         |
| --------------------------------------------------------- | ------------- | ---- | ---- | ------------ | ------------- | -------------------- |
| SDOT by element; SVE SDOT indexed; SDOT vector, both      | 1.06          | 0.27 | 0.19 | 5.16         | 1.00          | 2.00                 |
| SMMLA, NEON and SVE                                       | 1.06          | 0.27 | 0.19 | 5.16         | 1.00          | 2.00                 |
| FMLA by element or vector; SVE FMLA indexed or predicated | 2.00          | 0.25 | 0.20 | 5.16         | 2.00          | 2.00                 |
| SCVTF, NEON and SVE                                       | 4.00          | 0.62 | 0.51 | 2.00         | 4.00          | 0.50                 |
| FSUB, FADD, ZIP1 (NEON and SVE), ADD, ADDP                | 2.00          | 0.25 | 0.19 | 5.16         | 2.00          | 2.00                 |
| MOVI #0, MOV, SVE DUP of #0                               |               |      |      | 5.16         |               | 2.00 (MOVI #0: 4.00) |
| LDR q, d or s, LD1R; SVE LD1B, LD1W, LD1RQB, LD1RW, LDR z |               |      |      | 4.00         |               | 2.7 to 3.0           |
| LDP q                                                     |               |      |      | 2.00         |               | 1.00                 |
| SVE gather, LD1W of 4 words / LD1D of 2                   |               |      |      | 0.72 / 1.00  |               | 0.27 / 0.41          |

Every SVE form runs at its NEON form's rate, to the third figure, on both
cores. The SDOT chain's 1.06 cycles is the accumulator forwarded within a
pipe; eight chains reach 3.66 a cycle and sixteen 5.16, and the 4-column
SDOT tile has eight accumulators. The A725 takes MOVI #0 for nothing, any
other immediate at a pipe's cost.

Mixed into sixteen SDOT chains on the X925, cycles a SDOT (0.19 alone):
eight MOVI, MOV, FMLA, ZIP1 or FSUB beside them 0.29, each a slot of the
pipes'; eight SCVTF 0.46, each 0.54 cycles, 2.7 slots; eight LDR q (or
four LDP q, or eight SVE LD1B) 0.17 to 0.18, the SDOTs then at 5.6 to 5.8
a cycle; eight LD1 {.d}[1] 0.25; four gathers 0.36. On the A725 (0.50
alone): FMLA, ZIP1, FSUB or MOV 0.75; SCVTF 1.00, two slots; MOVI #0 0.50.

What SVE has that NEON lacks and these cores run for less is MOVPRFX: a
copy fused into the destructive instruction after it, which makes SDOT,
SMMLA and FMLA constructive. Cycles a pair, sixteen independent pairs of
a register's start and one instruction into it:

| pair                     | X925 | A725 |
| ------------------------ | ---- | ---- |
| MOVI #0 + SDOT           | 0.39 | 0.50 |
| MOVPRFX + SDOT (indexed) | 0.20 | 1.00 |
| MOVPRFX + SMMLA          | 0.20 | 0.50 |
| MOV + SMMLA              | 0.39 | 1.00 |
| MOVPRFX + FMLA (indexed) | 0.20 | 1.00 |

The X925 fuses MOVPRFX with all three; the A725 with SMMLA alone, and
there a MOVI #0 is free anyway. A gather costs four to seven contiguous
loads, and predication has no tails to take here: every kernel's widths
are multiples of 16. So at 128 bits SVE's arithmetic is NEON's, and of
its forms only MOVPRFX could pay, a slot a block for each accumulator
started. The two other things the table shows are NEON's: SCVTF costs the
X925 as much as 2.7 SDOTs, and its eight chains hold the SDOT tile below
its sixteen's rate.

## 2026-11-02 - the int8 tile's block loop in L1: SDOT, and SMMLA with its weights in SMMLA's order (a probe)

The 2026-10-24 entry left open what weights already in SMMLA's order
could buy over SDOT, with the ZIPs out of the loop. The probe (qk, kept
with the session's notes): the tile's block loop, 8 rows by 4 columns,
written out as assembly by a generator in each form below, on one
L1-resident buffer (1024 inputs: the half panel's 8 KB, the columns' 1 KB
each), every output checked bit for bit against ref's chain (acc =
fmaf((float)s, d, acc)) before it is timed; three processes on each core,
within 0.3%. Cycles a 1024 multiply-adds (a block of 32 inputs):

| loop                                                               | X925        | A725        |
| ------------------------------------------------------------------ | ----------- | ----------- |
| the tree's SDOT loop (q8_blocks4_qb32)                             | 19.29-19.33 | 41.77-41.85 |
| the same, MOVI and SCVTF as a MOV of M and an FSUB (below)         | 19.05       | 44.18       |
| the same, MOVI and SCVTF as MOVPRFX of M and an FSUB               | 17.52-17.54 | 46.25-46.29 |
| SMMLA, weights in its order, A by ZIP; MOVI and SCVTF              | 12.88       | 31.88       |
| the same; MOVPRFX of M and FSUB                                    | 12.15-12.16 | 30.06-30.07 |
| SMMLA, A by LDR d and LD1 {.d}[1]; MOVI and SCVTF                  | 14.11-14.15 | 33.31-33.32 |
| the same; MOV of M and FSUB                                        | 14.40-14.42 | 33.65       |
| the same; MOVPRFX of M and FSUB                                    | 13.15-13.19 | 30.45       |
| A by ZIP, MOVPRFX; each block's scales into vectors in the loop    | 12.51-12.52 | 30.12-30.14 |
| the same, the weights ZIPped in the loop from the file's order     | 15.38-15.43 | 37.13-37.15 |
| the same as the last but one at blocks of 64 (the fast mode)       | 10.77-10.78 | 25.11-25.12 |
| the same, MOVI and SCVTF                                           | 11.15-11.18 | 26.35-26.37 |
| the half panel's weights put in SMMLA's order, a block's 256 bytes | 8.11        | 12.42       |

The SDOT loop's 19.3 cycles in L1 is q8bench's at T 256 in 64-row slices
(207-210 GMAC/s on one X925 under gcc 13.3 and clang 18.1, 19.1 cycles):
the product is bound by its instructions, not its memory. At about 5.2
slots a cycle they come to 19.5 cycles: 64 SDOTs, 8 MOVIs, 8 SCVTFs at 2.7
and 8 FMLAs.

The SMMLA loop: a k-step of 8 inputs is four row pairs' B (two LDPs),
each two columns' A from their 16 bytes ZIPped at 64 bits (one ZIP a
k-step and pair), and 8 SMMLAs into accumulators m[p][j], [c r, c r+1,
c+1 r, c+1 r+1]: 32 SMMLAs a block where SDOT takes 64. A block's sums
start at M = 0x4B400000 (MOVPRFX, the first SMMLA into each), so that a
sum s is held as M + s, which as a float is 12582912 + s exactly while |s|
< 2^22 (a block of 64 inputs is at most 64 x 127 x 128 = 1,040,384); an
FSUB of 12582912.0f leaves s exactly, the float SCVTF would give, +0.0 for 0. Then one FMLA by a vector of the block's scales, [d_c d_c d_c+1
d_c+1], two LD1Rs and a ZIP1 a pair: each lane is one output's ref chain,
and the floats are taken apart into columns once, at the tile's end. Of
the 1.54 times, MOVPRFX is 5 to 6% (12.88 to 12.15, 31.88 to 30.06), the
conversion without it a little worse than SCVTF (its MOV costs the slot
MOVPRFX saves), and the rest SMMLA on weights in its order: in the
file's order, ZIPped in the loop, the tile keeps 1.26x on the X925 and
1.13x on the A725, the earlier tile's ZIPs and SCVTFs having taken the
rest. A's ZIP beats LDR d and LD1 {.d}[1] on both cores.

## 2026-11-02 - neon-sve: the int8 product on SMMLA, each half panel's weights put in its order a call at a time (hypothesis)

A third variant of backend/neon, backend/neon-sve: neon-dp's code, and
its 4-column tile at blocks of 32 or 64 inputs on the SMMLA loop of the
probe above. Each half panel's weights are put in SMMLA's order once a
call (a thread's buffer, 8 rows of up to 16384 inputs, 128 KB), and every
4-column tile of the call's columns runs the loop from that copy; tiles of
1 to 3 columns, other block sizes (the decoder's) and calls of fewer than
8 columns keep neon-dp's SDOT tile. The copy is per half panel and column
block, so the variant's column block is 1 MB, a call's columns at any
thread count here (at ten threads pk_gemm's calls are 64 rows by about 60
columns; at one, 256 rows by 250): each half panel is copied once a call,
0.65 of a tile's time on the X925, about 4% of a call at ten threads and
1% at one. The model file is read as it is, its pages the page cache's.

Its needs are asimddp, sve, i8mm and svei8mm (HWCAP_SVE, HWCAP2_I8MM and
HWCAP2_SVEI8MM; core/os.c learns them on Linux, and elsewhere answers no,
so neon-dp runs). SVE is what the loop's MOVPRFX uses, and it is also
where SMMLA is known to issue as SDOT does: the cores without it, Apple's
M cores (no SVE outside streaming mode) and Snapdragon X's Oryon, issue
SMMLA at half SDOT's rate (2026-10-24 and 2026-10-25 above), and the
variant passes them by a feature they lack. Its C is compiled as
neon-dp's, with neon-dp's flags; the loop's assembly asks for SVE and
I8MM itself (.arch_extension), so that no other kernel's code differs
from neon-dp's by a compiler's choice to use SVE, and the bench compares
the int8 product alone. Its first SMMLA into each accumulator is SVE's
(after MOVPRFX), the rest NEON's: at any vector length the low 128 bits
are the sums, since NEON's writes zero what is above them. Measured at
128 bits only.

Bits: the integer sums are exact in any order, the conversion is exact,
and each lane's FMLA is the fused multiply-add of ref's chain, block
after block. tests/kcheck holds it on the GB10 under both compilers.

Predict, on one X925 (q8bench, T 256, 64-row slices): blocks of 32 from
207-210 GMAC/s to 290-315 under both compilers, blocks of 64 from 224-231
to 340-370; on an A725, 65 to 85-90. `pktito-c bench`, test.wav through
v2, best of 5: at the default (the ten X925s), where `make PROFILE=1` puts
the int8 products at 74% of 0.386 s (feed-forward 46.8%, q k v pos 15.8%,
conv 8.3%, attention out 3.1%), the run 15 to 22% faster, 0.30 to 0.33 s;
on one X925 (taskset -c 5, -t 1), where they are 84% of 2.995 s, 20 to
28% faster, 2.15 to 2.4 s; the fast mode by about as much. Kept if kcheck
and check-c are ref's bits on the GB10 under gcc 13.3 and clang 18.1 and
the bench moves past its spread under both, at the default threads and at
one.

## 2026-11-02 - neon-sve: the int8 product on SMMLA, each half panel's weights put in its order a call at a time (result: kept)

Built as the hypothesis has it. backend/neon-sve/kernels.c includes
backend/neon's with PK_NEON_DOTPROD and PK_NEON_SVE; there q8_mm4 is the
probe's loop (A by ZIP, MOVPRFX of M and an FSUB, each block's scales into
vectors in the loop) at blocks of 32 and 64, mm_copy the copy, and the
column block 1 MB (not measured against other sizes). core/os.c learns
sve, i8mm and svei8mm from the auxiliary vector; the Makefile builds
neon-sve where the compiler's assembler takes a MOVPRFX and SVE's SMMLA
(its probe is those two instructions), and backend/kernels.c puts it
first.

Bits: kcheck, 1906 cases, ref's bits on neon-sve, neon-dp and neon on
spark-a80d under gcc 13.3 and clang 18.1, and check-c passes under both:
test.wav's v2 and v3 transcripts ref's, every token and time, in the
default and the fast mode, neon-sve picked. Two mutations of the loop
each fail kcheck: the last FMLA given columns 0 and 1's scales (cases at
blocks of 32, n 32 to 4096, T 8 to 26), and the loop at 64 a pair of
k-steps short (cases at blocks of 64, n 64 to 256). The neon and neon-dp
objects are the parent's bytes (gcc 15.2's cross build, compared by
objdump); the tree builds with llvm-mingw's clang 23.1 for Windows on ARM
(neon-sve built there, never picked: core/os.c answers no for sve) and
with NDK r30's clang for Android.

q8bench on one X925 (taskset -c 5), T 256 in 64-row slices, GMAC/s:

| product                 | qb  | gcc 13.3, neon-dp | neon-sve | clang 18.1, neon-dp | neon-sve |
| ----------------------- | --- | ----------------- | -------- | ------------------- | -------- |
| ff up 4096 x 1024       | 32  | 207.0             | 292.1    | 207.8               | 296.1    |
|                         | 64  | 223.8             | 333.9    | 228.6               | 331.9    |
| ff down 1024 x 4096     | 32  | 193.9             | 289.5    | 193.9               | 291.4    |
|                         | 64  | 200.3             | 329.6    | 201.0               | 332.9    |
| attention q 1024 x 1024 | 32  | 209.2             | 295.7    | 210.2               | 300.3    |
|                         | 64  | 226.3             | 333.9    | 231.2               | 342.0    |

On one A725 (CPU 0, gcc): ff up 65.4 to 88.9 at 32 and 71.2 to 103.7 at
64, ff down 65.4 to 91.7 and 71.5 to 109.2, attention q 64.8 to 88.9 and
71.1 to 106.0. Predicted 290-315 at 32 and 340-370 at 64 on the X925, 85
to 90 on the A725: in range at 32, 2 to 3% short at 64, the A725 above.

By columns, q8bench on one X925 (gcc), a call of all T columns,
neon-sve's GMAC/s against neon-dp's, at blocks of 32 / 64:

| T   | ff up          | ff down        | attention q    |
| --- | -------------- | -------------- | -------------- |
| 4   | +0.6 / +0.2%   | +3.0 / -0.2%   | -0.7 / +0.8%   |
| 8   | -0.2 / +3.3%   | -6.8 / -4.1%   | +5.7 / +8.3%   |
| 12  | +11.1 / +16.3% | +7.6 / +13.8%  | +15.2 / +20.1% |
| 16  | +18.9 / +24.2% | +16.8 / +24.3% | +21.4 / +27.7% |
| 24  | +26.2 / +31.9% | +28.7 / +37.8% | +28.7 / +36.5% |
| 60  | +34.0 / +41.6% | +42.9 / +54.9% | +35.4 / +42.6% |
| 128 | +39.7 / +46.2% | +49.5 / +64.6% | +40.3 / +47.3% |
| 256 | +41.1 / +47.7% | +47.4 / +64.5% | +41.2 / +48.4% |

At T 4 both run SDOT (fewer than PK_Q8_MM_COLS, 8). At 8 two tiles share
the copy, and the down product's 32 KB half panels lose where the 8 KB
ones gain; from 12 every product gains. pk_gemm's calls have all T
columns below 96, and above it 48 or more but for a product's last call,
which on test.wav's chunks is 10 at ten threads: 8 to 11 columns is a
recording under a second or such a last call.

`pktito-c bench` on spark-a80d, test.wav through v2, best of 5,
interleaved rounds with the order reversed every other round, against the
parent (d2c70f0) built the same way; the vLLM worker idle throughout,
about 3% of one core in every run of both builds (its CPU ticks, logged
for each). The paired change within each round, its median and range:

| run                                     | compiler   | base                     | neon-sve      | change                  |
| --------------------------------------- | ---------- | ------------------------ | ------------- | ----------------------- |
| default (ten X925s), 6 rounds           | gcc 13.3   | 0.383-0.392 s            | 0.324-0.332 s | -15.4% (-17.1 to -14.0) |
|                                         | clang 18.1 | 0.380-0.392 s, and 0.489 | 0.319-0.329 s | -16.8% (-32.9 to -13.9) |
| one X925 (taskset -c 5, -t 1), 6 rounds | gcc 13.3   | 2.990-2.994 s            | 2.276-2.293 s | -23.7% (-24.0 to -23.3) |
|                                         | clang 18.1 | 2.953-2.958 s            | 2.238-2.257 s | -24.1% (-24.3 to -23.6) |
| --fast, default, 4 rounds               | gcc 13.3   | 0.372-0.387 s            | 0.303-0.311 s | -19.4% (-21.1 to -18.3) |
|                                         | clang 18.1 | 0.370-0.376 s            | 0.300-0.307 s | -19.1% (-20.0 to -17.0) |
| --fast, one X925, 4 rounds              | gcc 13.3   | 2.844-2.851 s            | 2.082-2.101 s | -26.7% (-26.8 to -26.2) |
|                                         | clang 18.1 | 2.791-2.794 s            | 2.042-2.055 s | -26.7% (-26.8 to -26.4) |
| -t 20, 4 rounds                         | gcc 13.3   | 0.506-0.519 s            | 0.445-0.468 s | -11.2% (-13.1 to -7.7)  |
|                                         | clang 18.1 | 0.496-0.509 s            | 0.442-0.457 s | -11.5% (-12.5 to -7.9)  |
| one A725 (taskset -c 0, -t 1), 3 rounds | gcc 13.3   | 9.153-9.163 s            | 7.274-7.280 s | -20.6% (-20.6 to -20.5) |

The base's 0.489 under clang is a run whose race took the twenty threads;
neon-sve kept the ten in all twelve default runs, and the fast mode both
builds in all sixteen. In every row the two builds' ranges are apart by
more than either's spread (that 0.489 aside), under both compilers. gcc
and clang are within 2% of each other on either build (the loop is the
same bytes; the rest is each compiler's C). Predicted 15 to 22% at the
default and 20 to 28% on one X925: the default at the bottom of its
range, one thread in the middle of its.

Where it went, `make PROFILE=1` (gcc), three interleaved runs of each,
the median ms a run:

| stage         | default, base | neon-sve |        | one X925, base | neon-sve |        |
| ------------- | ------------- | -------- | ------ | -------------- | -------- | ------ |
| feed-forward  | 191.3         | 155.7    | -18.6% | 1661.1         | 1211.2   | -27.1% |
| q k v pos     | 64.9          | 52.0     | -19.9% | 478.9          | 344.3    | -28.1% |
| conv pw1 pw2  | 34.1          | 27.0     | -20.8% | 282.4          | 202.9    | -28.2% |
| attention out | 12.4          | 10.1     | -18.5% | 95.1           | 69.0     | -27.4% |
| subsampling   | 10.8          | 10.1     | -6.5%  | 79.8           | 70.6     | -11.5% |
| decoder       | 5.1           | 3.7      |        | 33.8           | 33.9     |        |
| the rest      | 91.7          | 87.2     |        | 363.7          | 363.8    |        |
| sum           | 410.3         | 345.8    | -15.7% | 2994.8         | 2295.7   | -23.3% |

(the rest: mel, positions, layer norms, quantize, attention, glu and
depthwise, none of them changed; at ten threads they move by the pool's
timing.) The four int8 stages fall 27.4% on one X925, a little under
q8bench's 29 to 33% at 256 columns, and 19.1% at ten threads.

The difference is the copy. At ten threads pk_gemm cuts a chunk's frames
(about 250 of test.wav's) into five calls of columns (48 to 60, a last
one fewer), and each call copies every half panel: five copies a product
where one thread makes one. The hypothesis put the copy at 4% of a call
at ten threads from its 8.1 cycles a block in L1. On spark-9eba, `make PROFILE=1` (gcc),
four interleaved rounds at the default, best of 5:

| build                                                                                                                          | best of 5     | the int8 stages, median ms |
| ------------------------------------------------------------------------------------------------------------------------------ | ------------- | -------------------------- |
| neon-sve                                                                                                                       | 0.331-0.337 s | 249.5                      |
| the copy made once a thread and never again (wrong sums: every 4-column tile reads a stale buffer, and none reads the weights) | 0.291-0.297 s | 215.7                      |
| a prefetch of the next panel's 16 rows (PRFM PLDL2KEEP, all of them before the panel's first tile)                             | 0.328-0.336 s |                            |
| a prefetch 2 KB ahead in the copy (PLDL1STRM)                                                                                  | 0.330-0.337 s |                            |

So the copy, with the read of the weights from memory it makes, costs
12% of a ten-thread run, three times its cost in L1, and neither prefetch
took it back. The copy reads a half panel 2.4 times as fast as the SDOT
tile it replaces (8.1 cycles a block against 19.3) and has no work of its
own to overlap the misses with, where the SDOT tile's first pass over the
half panel computes as it reads. Why the prefetches did not help is not
measured, and both were crude: the first issues up to 1024 PRFMs at once,
more than a core holds in flight, and the second runs about 30 cycles
ahead of its loads, short of a miss to DRAM.

Kept: neon-sve is first in the order, picked where Linux reports sve,
i8mm and svei8mm. Unmeasured, and where it is picked: Neoverse V1 (its
SVE 256 bits: the loop's MOVPRFX and first SMMLA then work on 256, the
upper half zeros), V2, V3 and N2 (c7g, c8g and c9g, Cobalt 100, Axion),
and Android's cores with SVE and I8MM where the kernel reports them, such
as the RG477V's Dimensity 8300 (A715 and A510). Apple's M cores and
Snapdragon X's Oryon have no SVE and keep neon-dp; Windows on ARM and
macOS answer no for sve.

## 2026-11-02 - neon-sve: each half panel's first tile makes the copy as it multiplies (hypothesis)

The entry above leaves 12% of a ten-thread run in the copy: a pass over
each half panel's weights with nothing to overlap its misses, five times
a product at ten threads. The probe's loop that ZIPs B in from the
file's order (15.4 cycles a block on the X925, 37.1 on the A725, against
SDOT's 19.3 and 41.8) is the SDOT tile's shape again, a first pass that
computes as it reads: so let each half panel's first 4-column tile run
that loop and store each k-step's four B vectors into the copy (two STPs
a k-step), and the call's other tiles read the copy as now. mm_copy goes.
The copy is then written at the first tile's pace and read from L1, and
its separate pass, the 8.1 cycles a block and the burst of misses, goes
with it. A tile from the file's order costs a fifth less than SDOT's in
L1, so a call of 4 to 7 columns gains too: PK_Q8_MM_COLS falls to 4.

Bits: the loop's sums and floats are the same operations on the same
bytes in the same order (the ZIPs only move them), so the same bits;
kcheck on the GB10 under both compilers, with a mutation of the new
loads.

Predict, test.wav through v2, best of 5: at the default (ten X925s), half
to two thirds of the copy's 12% back, 0.324-0.332 s to 0.305-0.315 under
both compilers; on one X925, where a call's 62 tiles share one copy,
level to 1% faster; the fast mode as the default. q8bench on one X925:
level at T 256, the T 8 down product's loss (-6.8% at blocks of 32) gone,
and T 4 10 to 20% over SDOT. Kept if the default moves past its spread
under both compilers and one thread does not lose.

## 2026-11-02 - neon-sve: each half panel's first tile makes the copy as it multiplies (result: kept, on the short recording)

Built as the hypothesis has it: MM_LDBF loads a k-step's B in the file's
order (two LDPs), ZIPs it and stores it into the copy (two STPs); q8_mm4
takes the copy's address and runs that loop for each half panel's first
4-column tile, the call's other 4-column tiles reading the copy; mm_copy
is gone, and PK_Q8_MM_COLS is 4. kcheck, 1906 cases, ref's bits on
neon-sve, neon-dp and neon on spark-a80d under gcc 13.3 and clang 18.1,
and check-c passes under both. Two mutations each fail kcheck: the copy's
second STP with its two registers swapped, so that only the tiles after
the first are wrong (cases at T 8 to 26), and a ZIP1 of the wrong group
(cases from T 6).

q8bench on one X925 (gcc), GMAC/s against the commit before (4e6a9c5),
whose T 4 is SDOT's tile, at blocks of 32 / 64:

| T   | ff up          | ff down        | attention q    |
| --- | -------------- | -------------- | -------------- |
| 4   | +14.1 / +11.8% | -1.2 / +3.9%   | +8.6 / +15.6%  |
| 8   | +18.3 / +20.5% | +25.0 / +23.4% | +20.0 / +21.8% |
| 12  | +13.3 / +15.2% | +18.5 / +17.7% | +13.1 / +15.7% |
| 60  | +2.6 / +3.4%   | +3.5 / +2.1%   | +2.5 / +2.7%   |
| 256 | -0.6 / +0.8%   | +0.4 / -1.6%   | +0.3 / +1.4%   |

At T 8 the down product, -6.8% against neon-dp before, is now 15.6% over
it (205.3 GMAC/s to 177.6).

`pktito-c bench`, best of 5, interleaved rounds as before, against
4e6a9c5; rounds where the vLLM worker took more than twice its median CPU
ticks are dropped (two rounds at the default, where it was serving and
both builds took twenty threads, and two one-thread runs). spark-9eba has
no clang. The clip is test.wav's first 1.5 s (24000 samples, "And says,
How do I get to Dublin?", 9 tokens):

| run                           | board, compiler        | rounds | 4e6a9c5       | the first tile's copy | change, median (range) | faster in |
| ----------------------------- | ---------------------- | ------ | ------------- | --------------------- | ---------------------- | --------- |
| test.wav, default (ten X925s) | spark-a80d, gcc 13.3   | 10     | 0.323-0.332 s | 0.318-0.327 s         | -2.3% (-3.9 to +1.2)   | 9         |
|                               | spark-a80d, clang 18.1 | 18     | 0.320-0.334 s | 0.315-0.326 s         | -2.5% (-4.0 to +0.3)   | 17        |
|                               | spark-9eba, gcc 13.3   | 8      | 0.326-0.336 s | 0.322-0.333 s         | -0.8% (-2.7 to +0.6)   | 6         |
| test.wav --fast, default      | spark-9eba, gcc 13.3   | 8      | 0.305-0.319 s | 0.299-0.311 s         | -2.7% (-3.5 to +0.7)   | 7         |
|                               | spark-a80d, clang 18.1 | 8      | 0.304-0.310 s | 0.296-0.304 s         | -1.8% (-3.3 to -0.7)   | 8         |
| test.wav, one X925            | spark-a80d, gcc 13.3   | 4      | 2.281-2.291 s | 2.283-2.295 s         | +0.0% (-0.1 to +0.4)   | 2         |
|                               | spark-a80d, clang 18.1 | 6      | 2.237-2.256 s | 2.237-2.248 s         | -0.0% (-0.6 to +0.1)   | 3         |
| the clip, one X925            | spark-a80d, gcc 13.3   | 4      | 0.078 s       | 0.076-0.077 s         | -2.6% (-2.6 to -1.3)   | 4         |
|                               | spark-9eba, gcc 13.3   | 4      | 0.077-0.078 s | 0.075-0.076 s         | -1.9% (-2.6 to -1.3)   | 4         |
|                               | spark-a80d, clang 18.1 | 8      | 0.077-0.078 s | 0.075 s               | -2.6% (-3.8 to -2.6)   | 8         |

The clip at the default threads is not in the table: both builds took
twenty threads there and ran 21 to 36 ms, the bench's millisecond as
wide as any change.

Predicted 0.305-0.315 s at the default; it is 0.315-0.333, 1 to 2.5%
faster in the median and faster in 32 of 36 rounds, but within its
spread (about 3%) on both boards and under both compilers. The
hypothesis asked for the default past its spread, and it is not. One
thread on test.wav is level, as predicted. Where a call has few columns
the change is past its spread: 2 to 2.6% on the 1.5 s clip at one thread,
every round on both boards under both compilers, and 13 to 25% on
q8bench's calls of 8 to 12 columns. So the copy's own pass was the
smaller part of the 12% the entry above put on it. The rest, about 9%
to the build that never copies (0.291-0.297 s on spark-9eba, a PROFILE
build, whose 4-column tiles read no weights at all), is as far as these
builds tell the first tile's reading of the weights from beyond L1, five
times a product at ten threads, which the SDOT tile pays too and no form
of the copy avoids. Fewer, longer column calls would read each panel
fewer times: a question for pk_gemm's split, which every backend shares,
not taken up here.

Kept: it is faster or level on every bench here, by more than its spread
where a call has few columns (a recording of a few seconds, or a
product's last call at ten threads), and MM_LDBF's eight instructions
take the place of mm_copy's loop. The default's 1 to 2.5% alone would
not have kept it; whoever weighs it differently can take it back to
4e6a9c5 with nothing else changing.

## 2026-11-02 - neon-sve as it stands against neon-dp, end to end (a measurement)

6381086's code against d2c70f0 (main, neon-dp picked), spark-a80d,
test.wav through v2, best of 5, four interleaved rounds, the vLLM worker
idle; both builds' default kept the ten X925s in every run:

| run      | compiler   | d2c70f0       | 6381086       | change, median (range)  |
| -------- | ---------- | ------------- | ------------- | ----------------------- |
| default  | gcc 13.3   | 0.388-0.398 s | 0.320-0.324 s | -17.5% (-19.4 to -17.1) |
|          | clang 18.1 | 0.385-0.390 s | 0.316-0.324 s | -17.0% (-18.6 to -16.8) |
| one X925 | gcc 13.3   | 2.992-2.999 s | 2.288-2.292 s | -23.6% (-23.6 to -23.5) |
|          | clang 18.1 | 2.952-2.955 s | 2.239-2.249 s | -24.0% (-24.2 to -23.9) |

A minute of speech in 0.32 s at the default, 187 times real time, where
neon-dp takes 0.39, 154 times.

## 2026-11-02 - neon-sve on Graviton2 to 5 (a measurement, pktito-windows's)

The cores the entries above left unmeasured: EC2 c6g, c7g, c8g and c9g
at .4xlarge, 16 cores each, Ubuntu 24.04 and kernel 7.0, gcc 13.3 and
clang 18.1. The runs were pktito-windows's, one script an instance
(graviton-sve.sh), with nothing else on the instance. 6381086's code
(646de79) was run against its parent d2c70f0, with neon-dp picked. The
bench was test.wav through v2, best of 5, rotated a place each round, 6
rounds at the default threads and 4 on one core (taskset -c 1, -t 1).

The bits:

- kcheck, 1906 cases, gives ref's bits on neon-sve, neon-dp and neon
  under both compilers on c7g, c8g and c9g, and on neon-dp and neon on
  c6g. There neon-sve is skipped: "this CPU has no i8mm".
- check-c passes on all four under both compilers.
- On c7g, Graviton3 (Neoverse V1, SVE at 256 bits), both hold at 256 bits
  and again with the length set to 128 (PR_SVE_SET_VL, inherited across
  exec). That is the first run of the loop at a length above 128 bits:
  its first SMMLA into each accumulator is SVE's, at the machine's
  length.

The change against d2c70f0, median (range). Every SVE row is faster in
every round:

| instance       | core, SVE             | run      | gcc 13.3                | clang 18.1              |
| -------------- | --------------------- | -------- | ----------------------- | ----------------------- |
| c6g, Graviton2 | Neoverse N1, none     | default  | +0.1% (+0.0 to +0.1)    | +0.1% (-0.1 to +0.1)    |
|                |                       | one core | +0.0% (-0.1 to +0.0)    | -0.1% (-0.1 to -0.0)    |
| c7g, Graviton3 | Neoverse V1, 256 bits | default  | -11.7% (-11.9 to -11.6) | -12.0% (-12.1 to -11.8) |
|                |                       | one core | -11.9% (-12.1 to -11.8) | -12.0% (-12.1 to -11.5) |
|                | at 128 bits           | default  | -15.6% (-15.9 to -15.6) | -15.9% (-16.2 to -15.7) |
|                |                       | one core | -13.6% (-13.8 to -13.5) | -13.7% (-13.7 to -13.6) |
| c8g, Graviton4 | Neoverse V2, 128 bits | default  | -12.7% (-12.8 to -12.6) | -12.5% (-12.7 to -12.5) |
|                |                       | one core | -17.2% (-17.3 to -16.9) | -17.6% (-17.7 to -17.6) |
| c9g, Graviton5 | Neoverse V3, 128 bits | default  | -13.3% (-13.6 to -13.3) | -14.0% (-14.0 to -13.7) |
|                |                       | one core | -15.8% (-15.9 to -15.6) | -15.6% (-15.7 to -15.6) |

c6g is the control: both builds run neon-dp there. The fastest of each,
under clang:

- c7g: 0.617 -> 0.543 s at the default, and 0.518 at 128 bits. On one
  core, 7.015 -> 6.171 s, and 6.054 at 128 bits.
- c8g: 0.479 -> 0.419 s at the default. On one core, 5.624 -> 4.630 s.
- c9g: 0.335 -> 0.289 s at the default. On one core, 4.555 -> 3.842 s.

gcc's times are within 1.7% of clang's, and the two compilers' changes
within 0.7 points of each other.

At 256 bits c7g is 4.5% slower than at 128 at the default, and 2% slower
on one core, in every round under both compilers. Where that goes is not
measured. The loop's SVE instructions, MOVPRFX and the first SMMLA into
each accumulator, are the only ones whose work grows with the length,
though the upper half is zeros. A loop with no SVE instruction would
run at 128 bits whatever the length. Setting the length is not the
library's to do: it belongs to the caller's thread, which runs a share
of every job.

So neon-sve is faster wherever it is picked, on every SVE core measured
so far: the GB10's X925 and A725, and Neoverse V1, V2 and V3. Not yet
measured: N2 (Cobalt 100), and Android's cores where the kernel reports
SVE. None of the four Android devices at hand reports it: the S24 Ultra
(whose kernel hides its cores' SVE2), the S25 Ultra and the Googlebook
(Oryon), and a moto g7.

## 2026-11-03 - neon-sve against neon-dp with the lanes (a measurement)

sve-climb rebased onto main's 7012b87 (the lanes and Silero's pool):
aae1fdf's code on spark-a80d, test.wav through v2, best of 5, at the
default, which for test.wav is two lanes of 10 on all 20 cores. neon-sve
as picked and neon-dp by PKTITO_KERNELS ran back to back in each of 6
rounds, with the vLLM worker idle (16 to 18 of its CPU ticks a pair of
runs):

| compiler   | neon-dp       | neon-sve      | change, median (range)  |
| ---------- | ------------- | ------------- | ----------------------- |
| gcc 13.3   | 0.348-0.356 s | 0.307-0.312 s | -12.4% (-13.5 to -11.5) |
| clang 18.1 | 0.346-0.352 s | 0.301-0.306 s | -12.7% (-14.0 to -12.1) |

Before the lanes (2026-11-02 above, the ten X925s) it was 0.388-0.398
against 0.320-0.324 s under gcc, -17.5%. Across the two sittings the
lanes took neon-dp's run from 0.39 to 0.35 s and neon-sve's from 0.32 to
0.31, so the margin narrowed from 17% to 12 to 13%.

Seen in passing, not paired with the rows above: on the ten X925s alone
(`taskset -c 5-9,15-19`, `-t 10`, two lanes of 5), neon-sve ran test.wav in
0.284-0.300 s, under the default's 0.307-0.312, and neon-dp in
0.354-0.369, over its 0.348-0.356. With neon-sve's product, the A725s'
share of the lanes may cost more than it gives on a minute of speech. That
is a question for the lanes' choice of threads, not measured here.

kcheck (1906 cases) gives ref's bits and check-c passes on spark-a80d
under gcc 13.3 and clang 18.1 (neon-sve, neon-dp and neon), on the M4 Pro
under Apple clang 21 (neon-dp and neon; neon-sve skipped), on the Pi 5
(gcc 14.2, the same) and on the Pi 3B+ (gcc 14.2, neon; the other two
skipped).

## 2026-11-05 - neon-sve on Windows: SVE and I8MM asked of the system (hypothesis)

core/os.c's Windows cpu_has answers asimddp alone, so neon-sve, which
needs asimddp, sve, i8mm and svei8mm, is never picked there. Windows
answers the other three through IsProcessorFeaturePresent:
PF_ARM_SVE_INSTRUCTIONS_AVAILABLE (46), PF_ARM_V82_I8MM_INSTRUCTIONS_AVAILABLE
(66) and PF_ARM_SVE_I8MM_INSTRUCTIONS_AVAILABLE (57), Microsoft Learn's
numbers for that call. rtxspark, an NVIDIA RTX Spark N1X (the GB10's ten
Cortex-X925 and ten Cortex-A725) under Windows 11 build 28000, reported all
three on 2026-11-03. Expected, if Windows runs the SVE loop as Linux does:
kcheck gives ref's bits under neon-sve there, and test.wav through v2
takes about 12% less time than under neon-dp, as on spark-a80d with the
lanes (2026-11-03 above). A Windows that answers no keeps neon-dp.

## 2026-11-05 - neon-sve on Windows: SVE and I8MM asked of the system (result: kept)

731cc3b with core/os.c's Windows cpu_has answering sve, i8mm and svei8mm
through IsProcessorFeaturePresent (46, 66, 57), cross-built on the 9950X
by llvm-mingw's clang 23.1.2 (UCRT) and run on rtxspark: Windows 11 build
28000.3231 says yes to all three, and neon-sve is picked.
tests/check-c.ps1 passes there with v2 (the app's copy; no v3 on the
machine, skipped): kcheck (1906 cases) gives ref's bits under neon-sve,
neon-dp and neon, and test.wav's v2 transcript under neon-sve is ref's,
every token and time.

test.wav through v2, best of 5, at the default of 20 threads, neon-sve as
picked and neon-dp by PKTITO_KERNELS back to back in each of 6 rounds, the
order alternating, the laptop idle before (1-2% CPU):

| compiler     | neon-dp       | neon-sve      | change, median (range)  |
| ------------ | ------------- | ------------- | ----------------------- |
| clang 23.1.2 | 0.319-0.346 s | 0.283-0.296 s | -12.4% (-15.3 to -10.3) |

Both slowed across the rounds as the laptop warmed (neon-dp 0.319 to
0.346 s, neon-sve 0.283 to 0.293), and the margin held: spark-a80d's
-12.4% (gcc 13.3) and -12.7% (clang 18.1) under Linux with the lanes
(2026-11-03 above).

## 2026-11-06 - SMMLA on the Galaxy S24's and S25's cores (a measurement)

Both phones' kernels report DP, I8MM and BF16 and no SVE (/proc/cpuinfo's
Features), so neon-sve is never picked there and their int8 product is
neon-dp's. Whether a NEON SMMLA loop could serve them turns on the
2026-10-24 entry's question, whether SMMLA issues as SDOT does. mmrate2.c
as above, built by NDK r30's clang 21 (`-O2 -march=armv8.6-a+i8mm`; each
loop holds its 1, 4, 8 or 16 SDOTs or SMMLAs and no others), run by
android-9950x through adb, pinned to one CPU at a time with taskset, three
runs a CPU, the first of which warms the clock and is left out: the Galaxy
S24 Ultra (SM-S928U1, SM8650: two A520s, five A720s and an X4) on AC at
27 C, and the Galaxy S25 Ultra (SM-S938U1, SM8750: eight second-generation
Oryon cores, every CPU part 0x001) on battery at 27 C. ns an instruction
with 1, 4, 8 and 16 chains, and a run's SMMLA over its SDOT at 16:

| core                                    | SDOT                                               | SMMLA                                              | SMMLA / SDOT |
| --------------------------------------- | -------------------------------------------------- | -------------------------------------------------- | ------------ |
| A520 (S24 cpu0-1)                       | 1.799-1.817, 0.447-0.471, 0.223-0.232, 0.224-0.232 | 1.784-1.826, 0.445-0.450, 0.279-0.280, 0.223-0.225 | 0.96-1.00    |
| A720 (S24 cpu2-4)                       | 0.319-0.389, 0.160-0.161, 0.160-0.161, 0.160       | 0.318-0.322, 0.160-0.161, 0.160-0.161, 0.160       | 1.00         |
| A720 (S24 cpu5-6, their clock low)      | 0.847-0.851, 0.425-0.427, 0.424-0.425, 0.425       | 0.847-0.853, 0.425-0.426, 0.424-0.425, 0.424-0.425 | 1.00         |
| X4 (S24 cpu7)                           | 0.361-0.415, 0.127-0.156, 0.082-0.090, 0.074       | 0.296-0.299, 0.074-0.147, 0.082-0.095, 0.074       | 1.00         |
| Oryon (S25 cpu0-5)                      | 0.899-1.214, 0.214-0.298, 0.142-0.197, 0.144-0.206 | 0.854-1.165, 0.214-0.296, 0.143-0.178, 0.143-0.186 | 0.90-1.05    |
| Oryon (S25 cpu6-7)                      | 0.785-0.986, 0.196-0.197, 0.088-0.089, 0.056-0.057 | 0.748-0.753, 0.197, 0.089-0.090, 0.056             | 0.98-1.00    |
| Oryon (snappy's X1E-78-100, 2026-10-25) | 0.887-0.892, 0.221-0.223, 0.110-0.111, 0.073-0.074 | 1.178-1.197, 0.294-0.295, 0.147-0.148, 0.147-0.148 | 2.0          |

SMMLA issues at SDOT's rate on every one of these cores, with the same
latency or less: twice SDOT's multiply-adds a cycle, as on the V2, X925
and A725 (2026-10-24), and not as on snappy's first-generation Oryon,
where it issues at half. The A520 and the X4 need more than 8 chains to
reach it (the A520's SMMLA 0.279-0.280 at 8, its SDOT 0.223-0.232). The
clocks read before the runs are not the runs' own, so the ratio, both
instructions in one run, is the measurement; the S25's cpu0-5 moved clock
within runs (2.75 to 3.53 GHz read), which spreads theirs.

What these cores could take is neon-sve's int8 loop without its one SVE
instruction: SMMLA on weights in its order, a block in 12.88 cycles to
SDOT's 19.3 on the X925 and 31.88 to 41.8 on the A725 without MOVPRFX (the
2026-11-02 probe). Untried. Without SVE as the gate it needs another way
to keep off the cores where SMMLA issues at half: Apple's M cores and the
first-generation Oryon (2026-10-24, 2026-10-25). The outputs are kept with
the session's notes.

## 2026-11-06 - neon-i8mm: neon-sve's int8 loop without SVE, for the cores with I8MM and none (hypothesis)

The S24's and S25's cores issue SMMLA at SDOT's rate (the entry above) and
report I8MM but no SVE, so they run neon-dp's SDOT loop. A backend
neon-i8mm would give them neon-sve's loop with its one SVE instruction
taken out. A block's first SMMLA into each accumulator becomes a MOV of M
(v12) and NEON's SMMLA, where neon-sve has MOVPRFX and SVE's SMMLA. That
is the 2026-11-02 probe's form: 12.88 cycles a block of 32 inputs on the
X925 and 31.88 on the A725, against neon-sve's 12.15 and 30.06 and the
SDOT loop's 19.3 and 41.8. The loop asks for I8MM itself
(.arch_extension), as neon-sve's asks for SVE and I8MM, and the file
takes neon-dp's flags. Its sums are the same integers in the same lanes
as neon-sve's, so its bits are ref's.

I8MM alone does not make the gate. Apple's M cores that have it and the
first-generation Oryon issue SMMLA at half SDOT's rate (2026-10-24,
2026-10-25), where this loop would lose to neon-dp's, and a list of CPU
parts would miss the cores to come. So neon-i8mm also needs a
measurement. When the backends are picked, a probe of its own times 16
chains of SMMLA against 16 of SDOT on the calling thread, for a few
hundred microseconds. It passes neon-i8mm by where SMMLA takes over 1.5
times SDOT's time an instruction: the phones' cores measured 0.96 to
1.05, snappy's X1E-78-100 2.0. The probe is backend code, so core/ stays
without intrinsics: backend/kernels.h's table gets an optional `usable`,
which pk_kernels_pick asks after a backend's needs. neon-i8mm goes after
neon-sve and before neon-dp, so a core with SVE keeps neon-sve.

Expected: on the S24 and S25, pktito-c bench under neon-i8mm takes about
10 to 15% less time than under PKTITO_KERNELS=neon-dp at the default
threads. That is neon-sve's gain on the X925 less MOVPRFX's part of it.
Less on the A520s, which reach SMMLA's rate only past 8 chains, where the
loop has 8 accumulators. Kept if:

- tests/kcheck gives ref's bits under neon-i8mm: under qemu-aarch64
  here, then on both phones (android-9950x);
- pktito-c bench moves by more than the run-to-run spread on both phones,
  under two compilers (two NDKs' clang);
- the probe takes neon-i8mm on both phones and passes it by on snappy.

Also to be measured, outside the decision: Graviton3 (c7g),
PKTITO_KERNELS=neon-i8mm against neon-sve at VL 256. The V1 runs neon-sve
4.5% slower at 256 than at 128 (2026-11-02, the Graviton entry), and the
NEON loop has no instruction whose speed the VL can move.

## 2026-11-07 - neon-i8mm: neon-sve's int8 loop without SVE, for the cores with I8MM and none (result: kept)

e7b432d, as the hypothesis above has it. kcheck, 1906 cases, gives ref's
bits under neon-i8mm, neon-dp and neon:

- under qemu-aarch64 here, with aarch64-linux-gnu-gcc 15.2, NDK r30's
  clang 21.0.0 and NDK r27d's clang 18.0.4;
- on both phones under both NDKs (android-9950x, through adb), where
  check-c passes and test.wav's v2 and v3 transcripts under neon-i8mm,
  picked, are ref's;
- on snappy, with llvm-mingw's clang 23.1.2 (kcheck alone);
- on the M4 mini, with Apple clang 21.0.0 (macOS asked for I8MM, below),
  where check-c passes and test.wav's v2 transcript under neon-i8mm,
  named, is ref's;
- on Graviton3, under gcc 13.3 and clang 18.1, at SVE's 256 bits and 128
  (below).

test.wav through v2, pktito-c bench (best of 5) at the default 8 threads.
The probe's pick and PKTITO_KERNELS=neon-dp ran back to back in each of 6
rounds, the order alternating. The phones were on AC, at 29.6 to 31.3 C
when each run began:

| phone     | compiler      | neon-dp       | neon-i8mm     | change, median (range)  |
| --------- | ------------- | ------------- | ------------- | ----------------------- |
| S24 Ultra | r30 clang 21  | 1.955-2.251 s | 1.869-1.965 s | -12.5% (-12.7 to -4.4)  |
| S24 Ultra | r27d clang 18 | 2.139-2.455 s | 1.883-2.154 s | -12.4% (-16.6 to -11.4) |
| S25 Ultra | r30 clang 21  | 1.471-1.828 s | 1.129-1.451 s | -21.5% (-30.9 to -19.5) |
| S25 Ultra | r27d clang 18 | 1.612-1.808 s | 1.254-1.441 s | -21.5% (-29.1 to -19.9) |

The probe took neon-i8mm on both phones in every round. The S24 is in the
expected 10 to 15%, and the S25's second-generation Oryon is past it.

snappy (X1E-78-100, Windows 11, on AC and idle). The probe passes
neon-i8mm by, so neon-dp is picked, as the hypothesis wanted. But
neon-i8mm, named, was the faster in every pair:

- 12 threads (the default), 3 rounds: neon-dp 0.397-0.445 s, neon-i8mm
  0.357-0.400 s, -10.1% (-10.3 to -9.3);
- one thread, 3 rounds: 3.981-3.992 s against 3.456-3.458 s, -13.3%
  (-13.4 to -13.2).

The premise that the loop would lose at half SDOT's rate does not hold on
the first-generation Oryon.

The M4 mini (base M4, macOS 26.4.1), from a scratch copy whose core/os.c
answers i8mm from hw.optional.arm.FEAT_I8MM. The tree's macOS cpu_has asks
only for the dot product, so neon-i8mm is never picked on a Mac or named
there. The probe passes neon-i8mm by, and here it is the slower:

- 10 threads (the default), 6 rounds: neon-dp 0.576-0.580 s, neon-i8mm
  0.623-0.627 s, +8.0% (+7.6 to +8.7);
- one thread, 4 rounds: 3.100-3.102 s against 3.398 s, +9.5% (+9.5 to
  +9.6).

Why the two differ, when mmrate2 has SMMLA at half SDOT's rate on both
(2.0)? mix.c times 16 independent SMMLA or SDOT chains a round with K
independent other instructions mixed in, in cycles from a chain of
dependent adds. Best of 7, three runs within 1% on each. The M4 mini held
4.42 GHz and snappy 3.41 to 3.42 throughout (mix.c: Apple clang 21.0.0
and llvm-mingw's clang 23.1.2, -O2):

| a round                      | M4                 | snappy           |
| ---------------------------- | ------------------ | ---------------- |
| 16 SMMLA                     | 8.9                | 8.0              |
| 16 SMMLA + 8 / 16 / 32 ZIP1  | 10.9 / 12.2 / 16.3 | 8.0 / 8.0 / 13.4 |
| 16 SMMLA + 16 FSUB           | 12.2-12.3          | 8.0              |
| 16 SMMLA + 16 MOV or MOVI #0 | 8.8-9.0            | 8.0-8.1          |
| 16 SDOT                      | 4.3-4.4            | 4.0              |
| 16 SDOT + 16 ZIP1 or SCVTF   | 8.1-8.2            | 8.2-8.3          |
| 16 SDOT + 16 MOVI #0         | 4.3-4.5            | 8.0              |
| 16 ZIP1, FSUB or SCVTF alone | 4.0                | 4.0              |
| 16 MOV alone                 | 2.0                | 2.25             |
| 16 MOVI #0 alone             | 2.0                | 4.0              |

On the M4 an SMMLA takes two of the four SIMD issue slots, so whatever is
mixed in adds what it takes alone. MOV and MOVI #0 are free there (8 a
cycle alone, handled at rename). On snappy's Oryon an SMMLA takes one slot
on two of the four pipes, and the other two run up to 16 other
instructions a round beside them for nothing. MOVI #0 takes a slot there.
A block of 32 inputs counts as follows:

- the M4: the SMMLA tile's 32 SMMLAs take 64 slots and its 10 ZIPs, 8
  FSUBs and 8 FMLAs 26, 90 in all; the SDOT tile's 64 SDOTs, 8 SCVTFs and
  8 FMLAs take 80, its MOVIs free. That is 12.5% more, where the M4
  measured +8 to +10% on the whole.
- Oryon: the SMMLA tile's 32 SMMLAs take 16 cycles on their two pipes,
  with the other 26 beside them; the SDOT tile's 88 slots, its MOVIs
  included, take 22. That is about a quarter less, where snappy measured
  -10 to -13% on the whole.

So the probe measures the one thing the two cores share, SMMLA's rate
alone. What parts them is what the rest of the tile can issue beside it.
A gate that times the two tiles themselves, in L1 on the calling thread,
would take neon-i8mm on snappy and keep it off the M4 by measurement, with
no platform's system left unasked. That is untried, a hypothesis of its
own (PKTITO-15 on the release board).

Graviton3 (c7g.4xlarge, 16 Neoverse V1 cores, Ubuntu 24.04), outside the
decision. Linux's default vector length there is 256 bits, and 128 was
set through /proc/sys/abi/sve_default_vector_length. Under gcc 13.3 and
clang 18.1, at 256 and at 128:

- check-c passes, kcheck giving ref's bits under all four NEON backends;
- test.wav's v2 transcript is ref's under neon-sve, neon-i8mm and neon-dp;
- the pick is neon-sve.

pktito-c bench, best of 5, the four in turn, the order rotated each
round: 6 rounds at the default 16 threads, 4 on one core (taskset):

| threads | compiler   | neon-sve, 256 | neon-sve, 128 | neon-i8mm     | neon-dp       |
| ------- | ---------- | ------------- | ------------- | ------------- | ------------- |
| 16      | gcc 13.3   | 0.543-0.545 s | 0.518-0.520 s | 0.541-0.542 s | 0.623-0.626 s |
| 16      | clang 18.1 | 0.539-0.544 s | 0.513-0.515 s | 0.535-0.540 s | 0.619-0.621 s |
| 1       | gcc 13.3   | 6.289-6.294 s | 6.183-6.189 s | 6.379-6.387 s | 7.137-7.145 s |
| 1       | clang 18.1 | 6.227-6.238 s | 6.112-6.123 s | 6.318-6.342 s | 7.068-7.074 s |

Paired by round, under gcc and clang:

- neon-i8mm against neon-sve at 256: -0.5 and -0.8% at 16 threads, +1.5
  and +1.5% on one. Level.
- neon-i8mm against neon-sve at 128: +4.4 and +4.6%, +3.2 and +3.4%.
- neon-sve at 128 against 256: -4.8 and -5.1%, -1.7 and -1.9%, the
  2026-11-02 entry's 4.5%.

So neon-i8mm does not take back the V1's loss at 256 bits. neon-sve at
128 is the V1's fastest, by 4.4 to 5.1% at its threads. Getting it would
mean neon-sve's threads at 128 bits (prctl's PR_SVE_SET_VL), a change of
its own (PKTITO-16).

Kept: -12.4 to -12.5% on the S24 and -21.5% on the S25 under both
compilers, at ref's bits. The gate is right on the M4, but for a reason
that does not tell it from snappy.

## 2026-11-07 - neon-i8mm's gate: the two tiles timed, not SMMLA's rate (hypothesis)

The entry above found the gate wrong on snappy. Its probe of SMMLA's rate
sees 2.0 on snappy's Oryon and on the M4 alike, and passes neon-i8mm by on
both. But neon-i8mm is 10 to 13% faster on the one and 8 to 10% slower on
the other. mix.c says why: what parts them is whether the tile's other
instructions issue beside SMMLA, and a rate alone cannot see that.

This probe measures the thing decided instead: the two 4-column tiles
themselves, timed on the calling thread when the backends are picked.

- q8_mm4 runs on a half panel's copy in SMMLA's order, and neon-dp's SDOT
  loop on the file's order.
- Each takes 16 blocks of 32 inputs, on zeros: 8, 4 and 2 KB, in L1.
- 64 calls make a run, the two alternate, and the best of 5 is taken.
- neon-i8mm runs where the SMMLA tile is the faster.

Its time is about the old probe's, a few tenths of a millisecond. The
kernels are unchanged, so every bit is.

macOS's cpu_has learns i8mm (hw.optional.arm.FEAT_I8MM). Apple's cores are
then kept off by the probe's measurement, not by a system left unasked,
and the neon-sve comment that leans on that goes.

Expected, from mix.c's counts: the SMMLA tile is 12.5% slower than SDOT's
on the M4 and about a quarter faster on snappy. It is faster on the S24's
and S25's cores, where SMMLA issues at SDOT's rate. So:

- snappy picks neon-i8mm, and its bench runs the 10 to 13% faster that the
  entry above measured with it named;
- the M4 keeps neon-dp, on its P cores and on its E cores (taskpolicy -b),
  which mix.c has not measured;
- the phones keep neon-i8mm on each kind of core, pinned: the S24's A520,
  A720 and X4, and the S25's two clusters.

Kept if the verdicts are those on every core named, and snappy's pick
moves past its spread against neon-dp. kcheck and check-c must pass as
before on each.
