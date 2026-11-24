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

## 2026-11-07 - neon-sve at 128 bits where a core's SVE is wider (hypothesis)

Graviton3's V1 runs SVE at 256 bits by Linux's default. There neon-sve's
loop is 4.8 to 5.1% slower at 16 threads, and 1.7 to 1.9% slower on one,
than at 128 (the neon-i8mm result above, and the 2026-11-02 Graviton
entry's 4.5%). The bits are the same: the loop's sums are its low 128
bits at any length.

Linux sets a thread's vector length (prctl's PR_SVE_SET_VL), so the
library can run its own threads at 128 without changing the machine's.
pk_os_unthrottle already readies the threads that run the library's work.
It would also set 128 where the length is longer, and pk_os_rethrottle
would put the caller's thread back. A pool worker keeps 128; the caller's
thread has it for the calls that pk_os_unthrottle wraps. That costs two
prctl calls a pk_asr_run on such a core, one where SVE is 128 bits
already, and none without SVE.

Expected on a c7g.4xlarge at the default 256: against main, -4.8 to -5.1%
at 16 threads and -1.7 to -1.9% on one. That is what main gives at 128 set
by the sysctl. Nothing changes on cores whose SVE is 128 bits (the GB10,
Graviton4 and 5) or that have none.

Kept if the bench moves past its spread under gcc and clang, and kcheck
and check-c pass at 256 and at 128.

## 2026-11-07 - neon-sve at 128 bits where a core's SVE is wider (result: kept)

55fcdcc, as the hypothesis above has it. On Linux, pk_os_unthrottle sets
the calling thread's SVE length to 128 bits where it is longer, and
pk_os_rethrottle puts it back. A pool worker keeps 128; the caller's
thread has it for the call.

Graviton3 (c7g.4xlarge, 16 Neoverse V1 cores, Ubuntu 24.04), 1c7dfc7 and
the change, each under gcc 13.3 and clang 18.1. With the length at
Linux's default 256 bits and at 128 (the sysctl):

- check-c passes, kcheck giving ref's bits under all four NEON backends;
- test.wav's v2 transcript under neon-sve, picked, is ref's.

pktito-c bench (best of 5), the four in turn, the order rotated each
round: 6 rounds at the default 16 threads, 4 on one core (taskset):

| threads | compiler   | 1c7dfc7, 256  | change, 256   | 1c7dfc7, 128  | change, 128   |
| ------- | ---------- | ------------- | ------------- | ------------- | ------------- |
| 16      | gcc 13.3   | 0.543-0.545 s | 0.517-0.518 s | 0.517-0.520 s | 0.517-0.520 s |
| 16      | clang 18.1 | 0.538-0.541 s | 0.512-0.514 s | 0.513-0.514 s | 0.512-0.515 s |
| 1       | gcc 13.3   | 6.272-6.288 s | 6.159-6.168 s | 6.154-6.172 s | 6.166-6.184 s |
| 1       | clang 18.1 | 6.202-6.220 s | 6.095-6.120 s | 6.092-6.131 s | 6.090-6.095 s |

Paired by round, under gcc and clang:

- the change against 1c7dfc7 at 256: -4.8 and -4.8% at 16 threads (-5.2
  to -4.6 in all), -1.9 and -1.8% on one (-2.1 to -1.3);
- the change at 256 against 1c7dfc7 at 128: +0.0 and -0.1% at 16 threads,
  -0.1 and +0.1% on one. The change gives what the sysctl gave;
- at 128, where the length is only asked, the change against 1c7dfc7:
  +0.0 and -0.1%, +0.2 and -0.1%. The one prctl a call costs nothing seen.

The Windows and plain-C builds give pk_os_throttle's new field as well:
os.c compiles without a warning under llvm-mingw's clang 23.1.2 (arm64 and
x64) and mingw-w64 gcc 13. With two fields in their initializers, -Wextra
gave three warnings.

Kept: -4.8% at 16 threads and -1.8 to -1.9% on one under both compilers,
past spreads of 0.6% and 0.3%, at ref's bits.

## 2026-11-07 - neon-i8mm's probe on memory it has written (hypothesis)

The phones' first timings of the probe above disagree with each other.
Three builds of the same loops, timed the same way (SDOT then SMMLA a run,
the best of 5 of each), read three ratios on each core, pinned:

| core           | tilens, its main's loops | tilens, the probe's function | tilens2, 8 cold processes |
| -------------- | ------------------------ | ---------------------------- | ------------------------- |
| S24, A520 cpu0 | 0.580-0.646              | 1.061, 1.092                 | 0.802-0.874               |
| S24, A720 cpu2 | 0.427-0.433              | 0.551, 0.559                 | 0.555-0.575               |
| S24, A720 cpu5 | 0.427-0.432              | 0.543, 0.554                 | 0.567-0.584               |
| S24, X4 cpu7   | 0.576-0.583              | 0.504, 0.505                 | 0.516-0.525               |
| S25, cpu0      | 0.484-0.492              | 0.675, 0.684                 | 0.720-0.739               |
| S25, cpu6      | 0.347-0.365              | 0.702, 0.706                 | 0.601-0.655               |

tilens's two columns come from the same two processes. Each build is
steady from run to run, so it is the build that moves the ratio. On the
A520 one build would keep neon-i8mm off. tilens2's other estimator, the
probe's now, reads 0.805 to 0.888 there.

The loops are the same instructions in every build, their heads on a
64-byte line. What differs is where their data sits. The arrays are
static zeros the probe never writes, and Linux maps each such page, read
before it is written, to its one page of zeros. So every 4 KB page of the
arrays is the same memory, and each 64-byte line of it is read through as
many addresses as the arrays have pages.

The A720's L1 data cache chooses a line's set by virtual address bits 6
to 12 or 13 (4 ways of 32 or 64 KB), where a 4 KB page fixes only bits 0
to 11. It holds a line in one set at a time ("VIPT behaving as PIPT", its
manual), so a line read through two addresses in two sets moves between
them. Which lines move, and how often a call, follows from where each
binary's linker put the arrays.

What production reads is never that page: the weights come from the
model file, and the activations and the copy are written for each call.

The change: the probe's 14.6 KB go on the stack of the thread it times,
written with zeros before the timing, so that each of its pages is memory
of its own, as production's are. pk_os_fast_cores already holds 16 KB
there on Linux.

Expected: written, the ratios no longer move with the arrays' place, and
every phone core reads below 1. tilens3 times backend/neon/kernels.c's
own probe, with the change, on its data in five places:

- static zeros never written, at a page's start and 2 KB along;
- the same two, written first;
- on the stack, as the probe now runs.

Kept if the written arrays read alike in both places, every phone core
reads below 1 cold, and pktito-c's pick is neon-i8mm on each phone core
alone and unpinned. The M4 mini (default and background QoS) and snappy
must keep their verdicts, and kcheck and check-c must pass as before.

## 2026-11-07 - neon-i8mm's probe on the fastest cores, from a thread of its own (hypothesis)

The entry above's kit, tilens3, cold. Each core ran 6 processes a mode,
interleaved a second apart (the phones through android-9950x, on AC and
idle; the M4 mini on AC):

| core                        | zeros, a page's start | zeros, 2 KB along | written, at the start | written, along | on the stack |
| --------------------------- | --------------------- | ----------------- | --------------------- | -------------- | ------------ |
| S24, A520 cpu0              | 0.800-0.908           | 0.953-1.038       | 1.159-1.169           | 1.162-1.180    | 1.157-1.171  |
| S24, A720 cpu2              | 0.566-0.573           | 0.562-0.570       | 0.806-0.807           | 0.806-0.809    | 0.800-0.809  |
| S24, A720 cpu5              | 0.563-0.571           | 0.542-0.579       | 0.802-0.808           | 0.807-0.808    | 0.805-0.808  |
| S24, X4 cpu7                | 0.510-0.513           | 0.504-0.513       | 0.838-0.841           | 0.838-0.843    | 0.838-0.844  |
| S25, cpu0                   | 0.699-0.710           | 0.710-0.722       | 0.730-0.733           | 0.731-0.732    | 0.731-0.734  |
| S25, cpu6                   | 0.751-0.769           | 0.728-0.746       | 0.728-0.740           | 0.721-0.731    | 0.730-0.733  |
| M4 mini, background QoS (E) | 1.081-1.082           | 1.079-1.082       | 1.080-1.082           | 1.080-1.087    | 1.081-1.082  |
| M4 mini, after 0.3 s (P)    | 1.112-1.123           | 1.118-1.126       | 1.118-1.127           | 1.115-1.126    | 1.113-1.123  |

Written, the ratios no longer move with the arrays' place: the zero page
had moved them. The M4's are the same in every mode. But the expectation
that every phone core reads below 1 was wrong. The A520 reads 1.16 in
every process, its SDOT tile 14% the faster, which the zero page had
hidden. The S24's A720s and X4 read 0.80 to 0.84, so its cores of two
kinds disagree. The S25's are unmoved, at 0.72 to 0.74.

Unpinned on the S24, the stack read the A520's 1.17 in 1 process of 6.
The pick goes where the loading thread happens to be. On an A520 it keeps
neon-dp, and the S24's whole run at 8 threads is 12.5% slower on that
(the neon-i8mm entry above).

The proposal: the probe's answer is the process's, and the fast cores do
most of a run's work, so ask it there. On Linux the probe runs on a thread
of its own, held by sched_setaffinity to the fastest kind of core the
process may use (the cores pk_os_fast_cores counts), and the caller waits
for it. The caller's own thread is left as it was. Setting its mask and
back would leave it a mask of its own, the cores it had at that moment,
which Linux keeps (6.2 and later) through the process's later cpusets:
Android moves an app between them as it goes to the background and back.

- Where the cores are of one kind, the probe runs on the calling thread,
  as now.
- A process that may run only on the A520s (taskset, or Android's
  background cpuset) probes there, and keeps neon-dp.
- Elsewhere than Linux, the calling thread, as now: the M4's two kinds
  agree.

Expected: unpinned, both phones pick neon-i8mm every time. Pinned to one
core, the pick is that core's: neon-dp on the A520, neon-i8mm on the
A720s, the X4 and the S25's cores. Nobody has run the whole model on the
A520 alone. One thread there should run neon-dp faster than neon-i8mm, if
the probe's tiles in L1 stand for its run; if not, that case is a known
miss, and recorded.

Kept if the picks are those, and check-c passes on both phones, the M4
mini, snappy and the 9950X.

## 2026-11-07 - neon-i8mm's gate: the two tiles timed, not SMMLA's rate (result: kept)

a38ee90, the three hypotheses above: the two tiles timed; on memory the
probe has written, on its stack; on the fastest kind of core, from a
thread of its own. Three more changes came from the first measurements.

The clock. macOS's CLOCK_MONOTONIC steps by a microsecond, where its
CLOCK_MONOTONIC_RAW and CLOCK_UPTIME_RAW step by 41 ns (the M4 mini, macOS
26.4.1). The probe's sections take 4 to 5 us on an M4 P core, so its
ratios there came out as 5/4 and 6/5. pk_os_now now reads CLOCK_UPTIME_RAW
on macOS, which stops in sleep as Linux's CLOCK_MONOTONIC does.

The order. As the first hypothesis had it (SDOT then SMMLA a run, the best
of 5 of each), the probe read up to 10% low and 7% high on a cold M5 core.
A clock that moves within its 0.1 to 0.25 ms favours whichever tile runs
second. The probe now runs SDOT, SMMLA, SMMLA, SDOT a run, so that a clock
rising or falling through the four slows both alike, and takes the median
of 5 runs' ratios, past a run that a migration or an interrupt cut into.
12 cold processes of each, interleaved a second apart (tilens2, which
includes backend/neon/kernels.c; static zeros, which macOS and Windows
read as written data: the second hypothesis's table, and snappy's below):

| board, thread            | SDOT, SMMLA: best of 5 | SDOT, SMMLA, SMMLA, SDOT: median of 5 | its time     |
| ------------------------ | ---------------------- | ------------------------------------- | ------------ |
| M4 mini, default QoS     | 1.072-1.091            | 1.044-1.070                           | 0.38-0.81 ms |
| M4 mini, background QoS  | 1.053-1.082            | 1.062-1.065                           | 0.46-0.81 ms |
| M5 Ultra, default QoS    | 1.059-1.248            | 1.173-1.190                           | 0.24-0.47 ms |
| M5 Ultra, background QoS | 1.073-1.260            | 1.174-1.187                           | 0.38-0.55 ms |
| snappy                   | 0.797-0.855            | 0.788-0.816                           | 0.13 ms      |

Once a process. pk_model_load's check of PKTITO_KERNELS made the pick and
dropped it, and the kernels made it again at their first call: in a
pktito-c run on the M4 mini, the probe ran eleven times, ten of them at
once, one on each of the first job's threads, each keeping its own
verdict. On cores of two kinds those could differ. The first
pk_model_load's choice is now the one kept (backend/kernels.c), so the
probe runs once.

The fastest cores. pk_os_on_fast_cores (core/os.c) on a GB10, from a test
that prints where its function runs: unpinned, on a thread held to the ten
X925s (CPUs 5-9 and 15-19), the caller's mask as it was; under taskset -c
0-6, on the two X925s; under taskset -c 0-4, the A725s alone, on the
calling thread.

On the phones the fastest kind, by cpu_capacity, is the S24's X4 alone
(cpu7 at 1024, its A720s 897 and 842, its A520s 369) and the S25's two
4.47 GHz cores (cpu6-7 at 1024, cpu0-5 765). pktito-c info's pick, from
the tree with the change (android-9950x, ~/i8mm-phone-logs/s24-probe- and
s25-probe-5c480a7.txt):

- unpinned, 5 runs on each phone: neon-i8mm every time;
- the S24 under taskset, 3 runs a core: neon-dp on the A520 (cpu0), and
  neon-i8mm on cpu2, cpu5 and cpu7;
- the S25 under taskset: neon-i8mm on cpu0 and cpu6.

The whole model on one of the S24's cores, one thread, best of 5, in one
round of neon-dp, neon-i8mm, neon-i8mm, neon-dp, on AC after cooling to
30.2 C (s24-onecore-5c480a7.txt):

| S24 core    | neon-dp          | neon-i8mm        | first pair | the round |
| ----------- | ---------------- | ---------------- | ---------- | --------- |
| A520 (cpu0) | 23.629, 23.620 s | 24.784, 24.769 s | +4.9%      | +4.9%     |
| X4 (cpu7)   | 4.927, 5.356 s   | 4.663, 4.747 s   | -5.4%      | -8.5%     |

The A520's verdict holds for the model: neon-dp is the faster there, by
about a third of its tile's 16%. The X4's second neon-dp run began at
31.5 C, the round's warmest, so the first pair's -5.4% is the safer
figure.

The verdicts:

- snappy (X1E-78-100, Windows 11, on AC) picks neon-i8mm, in 5 runs of 5
  of pktito-c info. tilens3, 6 cold processes a mode, reads 0.796 to
  0.822 on static zeros, 0.807 to 0.815 on written data and 0.801 to
  0.816 on the stack, in 0.12 to 0.22 ms: Windows reads its never-written
  zeros as written ones. pktito-c bench, best of 5, on the tree before
  the second and third hypotheses (a38ee90's kernels, and the same pick):
  with a 45 s rest before each run, in 3 rounds of neon-dp, neon-i8mm,
  neon-i8mm, neon-dp at the default 12 threads, neon-dp 0.395-0.402 s and
  neon-i8mm 0.356-0.369 s, -8.2, -10.3 and -9.9% by round. On one thread,
  3 rounds: 4.003-4.016 s against 3.479-3.510 s, -12.5 to -13.1%. Back to
  back without the rest, the second run of each pair at 12 threads slowed
  as the laptop heated, whichever came second.
- The M4 mini keeps neon-dp. With the probe's data written, it reads 1.079
  to 1.087 on its E cores (default and background QoS) and 1.113 to 1.127
  on its P cores (after 0.3 s of work), the same in every mode.
- The M5 Ultra keeps neon-dp: 1.173 to 1.190 cold, as above.
- The S24 picks neon-i8mm, on its X4: at its 8 threads 12.5% faster than
  neon-dp (the neon-i8mm entry above). A process held to its A520s keeps
  neon-dp, 4.9% faster there.
- The S25 picks neon-i8mm on every core: 21.5% faster at 8 threads (the
  same entry).

check-c passes with no warning on the 9950X, the M4 mini (Apple clang
21.0.0), pi5a (gcc 14.2) and both phones (android-9950x, NDK r27d's clang
18.0.4: kcheck's 1906 cases at ref's bits under neon-i8mm, neon-dp and
neon, and test.wav's v2 and v3 transcripts ref's). check-c.ps1 passes on
rtxspark and snappy (llvm-mingw's clang 23.1.2). make check passes on
the 9950X, at the WER of main's. The tree cross-compiles with no warning
under aarch64-linux-gnu-gcc 15.2, NDK r27d's clang 18.0.4 and
llvm-mingw's clang 23.1.2.

Kept: neon-i8mm where its tile is the faster on the fastest cores a
process may use: snappy (8 to 10% at 12 threads, 13% on one), the S24
(12.5% at its 8 threads) and the S25 (21.5%). neon-dp where SDOT's tile
is: the M4, the M5, and the S24 held to its A520s (4.9%). The probe runs
once a process, in under a millisecond, and every backend gives ref's
bits.

## 2026-11-08 - A76 assembly worktree (hypotheses)

Starting at 5b85618 on branch neon-dp-a76-asm, measured on pi5gui:
Cortex-A76 r4p1, four cores, 2.4 GHz, SDOT present and no i8mm. The
existing qb32 inline assembly is the control; qb64 and the float products
are still intrinsics. Compare native GCC 14.2 with cross-built GCC 15.2
and Clang 21.1.8, all at -O3 -mtune=cortex-a76 -ffp-contract=off, with
-march=armv8.2-a+dotprod only in the SDOT backend. The cross builds are
static. Use a temporary performance governor, restore it after each batch,
run only one benchmark process at a time, and check throttling.

H1: write qb64's block loop in assembly as qb32's, then overlap weight
loads and block conversions with independent SDOT work. The eight integer
and eight float accumulators fit without spilling. Keep each block's
SCVTF and FMLA in ref's order; no quantization or arithmetic changes.

H2: compare 8x4, 4x8 and 8x6 tiles. More columns amortize weight loads;
the larger tiles may instead lose to register pressure, activation working
sets or tails. Measure both cache-resident and encoder-shaped products,
one and four threads, qb32 and qb64, before full transcription.

H3: write the float 16-row x 4-column tile's ordered FMLA chains in
assembly based on the fastest compiler output. Explicit streaming of
weight vectors can avoid compiler spills and fix loop placement. Preserve
bias addition (including +0 when absent), ReLU and arbitrary n tails.

Keep only changes that pass kcheck and check-c and improve interleaved
whole-transcription measurements outside their spread under both GCC and
Clang. Append rejected schedules too. New A76-specific code will be an
explicit build choice; the portable family backend remains available.

H4: replace SCVTF in the bounded qb32/qb64 integer sums with an exact
biased-float representation. Arm's Cortex-A76 Software Optimization Guide
(https://documentation-service.arm.com/static/5ed4bd67ca06a95ce53f917d)
lists Q-form F32 SCVTF at four cycles latency, one per two cycles on V0;
SDOT and FMLA can use both vector pipes. Initialize the integer sums to
0x4b400000 (the bits of 12582912.0f) instead of zero. A sum s over at most
64 products of int8 weights and [-127,127] activations has |s| <= 1040384,
well inside 2^22. Across this range float spacing around 12582912 is 1,
so reinterpret(0x4b400000 + s) - 12582912.0f is exactly (float)s, with
an exact subtraction. The SDOT adds are still exact int32 additions and
the ordered FMLAs are unchanged. One constant vector replaces a spare
weight register in the streaming schedule. Predict: the conversion
bottleneck becomes a two-pipe FSUB, saving several cycles per block.
Require bitwise ref checks, including extreme sums and cancellation.

## 2026-11-08 - compiler output and the tile search (results)

pi5gui has 64 KB L1D and 512 KB L2 per A76, and 2 MB shared L3. GCC
14.2.0-19 is native; GCC 15.2.0 and Clang 21.1.8 are cross compilers
with static glibc linking. All use -O3, -mtune=cortex-a76,
-ffp-contract=off and -fno-math-errno; the SDOT objects alone use
-march=armv8.2-a+dotprod. Neither compiler is allowed to generate i8mm.

The starting code is already stronger than pure intrinsics at qb32.
Disabling its inline assembly in scratch builds gave v3's minute in
3.827 s under GCC 15 and 3.943 s under Clang 21, against 3.775 and 3.754
for the existing backend in the initial runs. GCC's general 8x4 tile
recomputes group addresses inside a run-time chunk loop, moves an integer
accumulator at the loop edge and spills scalar state around the loops.
Clang groups many operations by output, retaining different dependency
chains. Both have a second loop over chunks for qb64. The existing qb32
assembly removes that compiler variation, but not its half-panel access
pattern. A cold native-GCC timing was unusually slow; a later native
profile took 3.787 s, with the same stage split as GCC 15. That first
sample is not evidence of a compiler regression.

Standalone assembly probes first reproduced the compiler's integer and
float chains, checked every tile against a scalar FMA reference, and
counted user-space CPU cycles with perf_event_open under sudo, pinned to
CPU 3. Three timings per variant, minimum, 30,000 repetitions at n=1024
and 20,000 at n=4096. These probes use ordinary SCVTF, before H4:

| tile / weight schedule   | qb32 cycles/block, n=1024 |   qb64 | qb32, n=4096 |   qb64 |
| ------------------------ | ------------------------: | -----: | -----------: | -----: |
| 8x4, preload four groups |                     50.51 |  85.32 |        53.49 |  92.52 |
| 8x4, stream two groups   |                     49.64 |  84.16 |        52.61 |  90.93 |
| 4x8, stream two groups   |                     51.08 |  84.51 |        64.58 | 114.25 |
| 8x6, stream one group    |                     75.66 | 128.00 |        77.57 | 133.23 |
| 16x2, stream two groups  |                     49.26 |  81.57 |        49.93 |  83.60 |

8x6 does 1.5 times the work per block: its cycles cannot be compared
without that factor. It offered no convincing throughput gain. 4x8 was
competitive only with small operands. The full 16x2 panel was the best
at 4096 inputs, and therefore advanced to encoder-shaped measurements.
An early 16x2 probe had an incomplete weight load and failed its reference
check; all table entries above are from the corrected, passing probe.

The first standalone 8x4 implementation saved registers once per run of
columns, included the output finish, and specialized qb64. It improved
the small qb64 case but did not reliably improve normal transcription.
Changing post-increment activation loads to offsets did not help. Loading
all four groups before their dots also lost. Explicit L1 prefetch at
512, 1024 and 2048 bytes ahead made streaming shapes slower, even when
the tiny L1 case improved. Moving the aligned block head 32 bytes had no
useful gain. Interleaving final dots with SCVTF improved this prototype's
L1 case slightly, but did not rescue it as the final tile.

The full-panel tile and exact biased conversion were complementary.
`gemmbench 250 1 2400 0 QB`, GCC 15, the four encoder shapes summed,
and its separately timed L1 case:

| implementation                            | qb32 all shapes, GMAC/s | qb32 L1 | qb64 all shapes | qb64 L1 |
| ----------------------------------------- | ----------------------: | ------: | --------------: | ------: |
| existing neon-dp                          |                   46.32 |   45.90 |           50.68 |   52.35 |
| 16x2 with SCVTF                           |                   48.63 |   48.68 |           57.65 |   58.39 |
| 16x2 with biased conversion, offset loads |                   51.19 |   51.36 |           59.13 |   60.15 |
| same, post-increment loads                |                   51.08 |   51.38 |           59.73 |   60.12 |
| same, explicit prefetch                   |                   48.94 |   51.20 |           56.71 |   59.94 |

Post-increment loads were retained: level at qb32, slightly better for
qb64. A final local scheduling comparison tried subtracting the bias
immediately after each last SDOT, then subtracting and FMA immediately.
Their L1 rates were 51.11/59.65 and 50.64/59.34 GMAC/s (qb32/qb64),
against 51.36/60.07 for the retained grouped finish. Both were rejected.

The float 16x4 tile was also written as standalone assembly, with ordered
FMLAs and explicit input tails, bias and ReLU. It passed kcheck and 1,656
prototype guard cases, including in-place bias. Its attention stage was
293.2 ms, essentially the existing code's 290-298 ms, with no useful
whole-run gain. H3 is rejected; float kernels remain the existing C.

Two interleaved rounds of v3's minute, four threads, best of five each,
confirmed the 16x2 candidate before final integration:

| implementation             | normal, seconds | fast, seconds |
| -------------------------- | --------------- | ------------- |
| existing GCC 15 neon-dp    | 3.872, 3.874    | 3.637, 3.631  |
| 16x2, offset loads         | 3.561, 3.553    | 3.230, 3.231  |
| 16x2, post-increment loads | 3.557, 3.558    | 3.228, 3.224  |

No throttling was reported. The integrated backend is explicitly named
neon-dp-a76 rather than replacing other SDOT cores' tile. Its FPCR check
retains the old tile under non-default rounding: the biased subtraction
would otherwise produce -0 for a zero integer sum under FE_DOWNWARD.
The normal path hoists the bias constant outside the column-pair loop,
removes unused prototype pointers, and has no stack access in the block
loop. Final same-binary comparisons and complete checks follow below.

## 2026-11-08 - integrated A76 backend, final comparisons and validation

The final comparison builds neon-dp-a76, neon-dp and neon into the same
binary. `tests/a76bench.sh` alternates the first two, reverses their order
on every other round, and runs v3 on tests/data/test.wav, four threads,
default chunking. Each sample is the best of five 60-second transcriptions;
the table gives the median of those samples and their full range in
seconds. VAD time is reported separately by the CLI, not included here.
There are three rounds for GCC 15 and Clang 21, two for native GCC 14.
The CPU governor was temporarily performance at 2.4 GHz, then restored to
ondemand. Every observation reported 2,400,000 kHz and throttled=0x0.

| compiler | mode   | existing neon-dp, median (range) | neon-dp-a76, median (range) | time reduction |
| -------- | ------ | -------------------------------: | --------------------------: | -------------: |
| gcc15    | normal |             3.8720 (3.870-3.878) |        3.5500 (3.549-3.558) |          8.32% |
| gcc15    | fast   |             3.6340 (3.627-3.635) |        3.2260 (3.216-3.235) |         11.23% |
| clang21  | normal |             3.8530 (3.850-3.855) |        3.5290 (3.523-3.532) |          8.41% |
| clang21  | fast   |             3.6070 (3.598-3.613) |        3.2130 (3.212-3.214) |         10.92% |
| gcc14    | normal |             3.8845 (3.884-3.885) |        3.5420 (3.540-3.544) |          8.82% |
| gcc14    | fast   |             3.5870 (3.586-3.588) |        3.2235 (3.221-3.226) |         10.13% |

All six improvements exceed their observed spread. The Clang build is
slightly fastest in this batch. Cross-compiler absolute times also include
the different toolchain/runtime builds; the same-binary comparisons are
what establish the assembly gain. No claim is made about other cores,
thread counts, recordings or chunk sizes.

Validation on pi5gui, GCC 14.2.0, GCC 15.2.0 and Clang 21.1.8:

- The complete C suite passes with all three backends. kcheck checks
  1,906 cases per backend, including v3's actual weights, against ref's
  bits. VAD probabilities and both v2 and v3 transcripts, in normal and
  fast mode, match every reference token and timestamp.
- The new a76check passes 379 guarded cases per compiler: extreme sums,
  cancellation, zero scales, bias and residual combinations, odd columns,
  unaligned output strides and untouched padding, column-block boundaries,
  maximum input width, empty batches and all directed rounding modes.
- Native GCC's shared library passes the same 379 guard cases and kcheck's
  1,892 model-free cases per backend. The normal suite also exercises the
  static library through examples/transcribe.
- GCC 15 and Clang 21 emit byte-identical assembly .text (SHA-256
  39e1c8d5b5a7374b60d3c341c1e8bf9a85a8624006bf5e762f63f5413e792d66).
  There are no code relocations or i8mm instructions; q8_32 and q8_64
  occupy 812 and 1,148 bytes. Both have hidden ELF symbols and unwind CFI.
- The existing ThreadSanitizer target compiles and links with the new
  assembly dependency. Its runtime was not run; no thread-pool code
  changed. Python/NeMo WER validation was not rerun: the kernel definition
  and every checked result remain bit-identical, and this is not a merge.

The retained changes are the full-panel 16x2 schedule, dedicated qb32 and
qb64 loops, exact biased conversion, post-increment activation loads and
one register save per run of column pairs. Other kernels and the ordinary
family selection retain their existing implementations. This is an
explicit `BACKEND=neon-dp-a76` experiment, ready for further A76 climbs.

## 2026-11-08 - neon-dp-a76 in every AArch64 build, picked by a probe (hypothesis)

The A76's tile above, on main at 76d4cef, which since 5b85618 has the
lanes, neon-sve, neon-i8mm and neon-i8mm's probe. Two changes, neither to
the loop's instructions:

- Inline assembly in backend/neon/kernels.c, as the 4-column tile's loop
  is, in place of q8.S: the same instructions in the same order and the
  same vector registers, the general registers the compiler's, and the
  callee-saved ones saved by q8_cols's prologue once a column block. An
  ELF .S needed Makefile rules of its own and would need Mach-O's and
  COFF's symbols and unwind records (Windows's are SEH, not CFI); inline,
  every AArch64 build has it - Linux, Android, macOS, and Windows under
  llvm-mingw, where the loop's head goes without its line, as the
  4-column loop's does.
- In the family build, between neon-i8mm and neon-dp, behind a probe as
  neon-i8mm is: the A76's two pairs against the 4-column tile's two half
  panels, a panel's 16 rows by 4 columns over 32 blocks of 32 inputs (16
  KB of weights, in L1), each with its finish; SDOT's, the A76's, the
  A76's, SDOT's a run, the median of 5 runs' ratios, on the fastest cores
  the process may use. PKTITO_KERNELS=neon-dp names the old tile, to
  compare.

Predict, test.wav through v3, best of 5, the two backends in one binary by
turns:

- H1, a Pi 5 (pi5a; its gcc 14.2, and gcc 15.2 and clang 21.1 built here):
  q8.S's gain, 8 to 9% at 4 threads and 10 to 11% in the fast mode, and
  the probe picks the A76's tile.
- H2, Graviton2 (c6g, Neoverse N1: the A76's server sibling, no I8MM, on
  neon-dp at 18.9x, 2026-11-02): 5 to 11% at 4 threads, the probe picks
  it.
- H3, armlet (Snapdragon 7c, Windows: two A76s, six A55s): the probe, on
  the A76s, picks it. The A55 issues in order and its SDOT rate is not the
  A76's; whether the tile helps or hurts there is unknown, and a run on
  all eight cores gains less than a run on the two.
- H4, the M4 (neon-dp: its probe passes neon-i8mm by): unknown. Its SIMD
  pipes are four, not the A76's two, so SCVTF may cost it nothing extra;
  within 3% either way, and the probe goes with whichever is faster.
- H5, Graviton3, 4 and 5 (Neoverse V1 to V3): neon-sve, before both, stays
  their pick, so nothing changes; neon-dp-a76 against neon-dp, named, is
  recorded for what the tile does on wider cores.
- Every probe's verdict is its machine's end-to-end winner; a machine where
  it is not is a reason to change the probe.
- tests/q8check, a76check's cases for every backend of a build, at ref's
  bits on every board it runs on.

## 2026-11-08 - a backend for each kind of core, picked by its own gemm_q8 (hypothesis)

The surveys of the int8 product alone (tests/q8bench on one core,
tests/gemmbench over a kind; each backend named, in order and reversed)
give each kind of core a different best tile:

| core           | neon-i8mm | neon-dp | neon-dp-a76 | best        |
| -------------- | --------- | ------- | ----------- | ----------- |
| Pi 5's A76     | -         | 47.5    | 54.4        | neon-dp-a76 |
| armlet's A76   | -         | 47.6    | 53.4        | neon-dp-a76 |
| armlet's A55   | -         | 8.80    | 8.99        | level       |
| the S24's X4   | 157.2     | 126.1   | 114.8       | neon-i8mm   |
| the S24's A720 | 99.7      | 76.3    | 77.1        | neon-i8mm   |
| the S24's A520 | 28.4      | 29.4    | 33.3        | neon-dp-a76 |

(GMAC/s on one thread: the mean of q8bench's six encoder rows, its three
shapes at T 750 in blocks of 32 and of 64, over both orders; the S24's
from android-9950x, ~/i8mm-phone-logs/s24-coresurvey-02ffafb-r30.txt;
02ffafb, gcc 15.2, llvm-mingw's clang 23.1.2 and r30's clang 21.0.0).
A process takes one
backend, picked on its fastest kind, so the S24's A520s run SMMLA, 17%
slower there than the A76's tile. And the probes, each a tile timed in L1,
miss on the in-order cores: the A76's reads 1.142 on armlet's A55s, where
the backend is 2.6% faster end to end, and keeps neon-dp on the A520.

Every backend computes ref's bits in every kernel, so threads on different
kinds of core may each run their own and no transcript can change. Three
changes:

- The kinds. core/os.c groups the cores the process may use: on Linux by
  each CPU's MIDR (implementer and part; /sys/.../regs/identification),
  else by cpu_capacity; on Windows by EfficiencyClass. macOS does not say
  which core a thread is on, so its cores stay one kind. pk_os_kind() is
  the calling thread's, from sched_getcpu or GetCurrentProcessorNumberEx.
- The probe. backend/kernels.c times each candidate's own gemm_q8, not a
  tile, on a slice as core/encoder.c's pk_gemm makes them: 64 rows of
  1024 inputs at T 64, blocks of 32, on memory it has written (64 KB of
  weights, past the A55's and A520's L1), by turns against the backend the
  needs alone would give (A, B, B, A, the median of 5 runs' ratios), on a
  thread held to each kind. The fastest kind's pick is the process's, as
  now; another kind takes another backend only where that one is at least
  3% the faster there. The candidates are the backends with a probe
  (neon-i8mm, neon-dp-a76), which now say only that they compete.
- The dispatch. pk_k_gemm_q8 takes the table of the kind its thread is on
  when called; every other kernel takes the process's. A call is a 64-row
  slice of a product, milliseconds on an A520, so a lookup a call is
  nothing; a thread that moves mid-call finishes with the tile it began.

Predict:

- H1, the probe: each kind's pick is its q8bench winner above: neon-dp-a76
  on the A76s and the A520, neon-i8mm on the X4 and A720, and on armlet's
  A55s the process's (neon-dp-a76, level). The M4 and M5 keep neon-dp.
  Under 5 ms a process on the S24 (three kinds), under 2 ms on a Pi 5.
- H2, the S24 at its 8 threads: its A520s on the A76's tile take 1 to 3%
  off a v3 run of test.wav; armlet unchanged (both kinds on neon-dp-a76);
  every board of one kind unchanged.
- H3, the bits: tests/q8check and kcheck at ref's bits for every backend
  as before, and a transcript made with a backend drawn at random for
  every gemm_q8 call ref's, token for token (a new check).
- H4, the cost: no bench of a board of one kind moves, and the S24's
  lookups cost under 0.1% of its run.

## 2026-11-08 - the probe on all of a kind's cores at once (hypothesis)

The probe above, built with T 128 (its result entry follows when it is
measured), times one core of each kind alone, and on two kinds that
reading and the kind's own end-to-end A/B disagree. The S24's two A520s,
held to test.wav at two threads (taskset 03, v2, android-9950x): neon-i8mm
14.95 to 15.08 s, neon-dp-a76 15.29, neon-dp 15.64 to 15.86, where one
A520 alone reads neon-dp-a76 0.945 to 0.968 of neon-dp's gemm_q8 and
neon-i8mm 1.120. armlet's A55s run test.wav 2.6% the faster on the A76's
tile, six at once, where one alone reads it 1.013 to 1.016.

A scratch tool (shapepar.c, kept with the run's logs) times the probe's
slice on several cores at once, each with its own weights and all of them
the same activations, every sample begun on all of them together by a
spin barrier. On armlet at 64 x 1024, T 128, blocks of 32: the six A55s
read neon-dp-a76 0.977 together against 1.006 for one of them alone in the
same run, the two A76s 0.975 against 0.925.

The change: pk_os_on_kind(j, fn, arg) runs fn(arg, i, n) on n of kind j's
cores at once (up to 16), a thread held to each, on Linux and Windows,
and on the calling thread on macOS, which holds no thread to a core. A
board of one kind is timed on all its cores too. Each thread has its own
weights and output; the activations are shared; each sample starts on
all threads at a barrier, with as many calls on each as thread 0's take.
Each thread's ratio is the median of 5 runs as before, and the kind's is
the median of its threads'. The pick and its 3% margin are unchanged.

Predict:

- H1: the S24's A520 pair reads neon-i8mm under neon-dp-a76, as its
  end-to-end A/B ranks them, and takes neon-i8mm; its X4 (one core) and
  A720s keep neon-i8mm.
- H2: armlet's two A76s read neon-dp-a76 about 0.975 and the A55s about
  0.977, both inside the 3% margin, so armlet takes neon-dp throughout;
  its 8-thread A/B (neon-dp, neon-dp-a76, the pick) says what the margin
  costs there. pi5a's four A76s keep neon-dp-a76.
- H3: the bits as before: kcheck, q8check, and check-c's transcripts with
  PKTITO_KERNELS=mix.
- H4: the probe's time a process grows by a thread's start per core (on
  the order of 50 us each), from 4.2 to 4.8 ms on pi5a.

## 2026-11-08 - a backend for each kind of core (result: rejected), the probe on all of the fastest cores at once (result: kept)

Measured on the per-kind code (a76235e, 691d3dd, 2ae0966 and the
concurrent probe after them):

- H1 of the probe on all of a kind's cores at once held on the S24: its
  X4 and A720s took neon-i8mm 5/5 (0.866 to 0.877, 0.832 to 0.834), and
  its A520 pair read noisily, within 5% either way, the 3% margin keeping
  kind 0's neon-i8mm in four runs of five. H2 did not: armlet's two A76s
  read neon-dp-a76 0.910 to 0.914 together, not 0.975, and its A55s 0.971
  to 0.984, so both kinds took neon-dp-a76, 5/5. pi5a read 0.909 to 0.916
  on its four cores at once and took neon-dp-a76 15/15 (gcc 14, gcc 15,
  clang 21), timed in 4.5 to 5.0 ms (H4). H3: kcheck, q8check and the
  mix transcripts held on pi5a, armlet and the S24.
- What a backend for each kind bought end to end: nothing measured. The
  S24 at 8 threads came out in two clusters (1.27 s and 1.50 to 1.67 s)
  whichever kernel its A520s ran; the A520 pair alone ran neon-i8mm
  14.936 s against neon-dp-a76's 15.302, 2.4%, and that pair does a
  small share of an 8-thread run. armlet's 8-thread A/B was too noisy to
  separate (means neon-dp 6.650 s, neon-dp-a76 6.385, the per-kind pick
  6.327).

So the per-kind dispatch (pk_os_kind, a lookup every 16th gemm_q8 call,
the kinds' names, PKTITO_KERNELS=mix) is taken out, and the probe that
times all of the fastest cores at once is kept: pk_os_on_fast_cores(fn,
arg), on Linux the cores of pk_os_fast_cores's speed, or every core where
sysfs gives no speeds; on Windows the highest EfficiencyClass's, a thread
a physical core. The process runs the fastest cores' pick on every core
(4e8c1d0).

## 2026-11-08 - neon-dp-a76, one pick on the fastest cores (results)

On 4e8c1d0. check-c passed by the pick and, where it says, with
PKTITO_KERNELS=neon-dp-a76: x86 (the 9950X, gcc 15); pi5a (gcc 14);
the M4 mini; armlet (llvm-mingw clang 23.1.2, tests/check-c.ps1); c6g,
c7g, c8g and c9g, each under gcc 13 and clang 18, both ways; the Galaxy
S25 (make adb, NDK r30 and r27d, android-9950x). No build warned. check-tsan passed on a c6g (gcc 13, `setarch -R`; on pi5a
TSan's runtime refuses the kernel's 47-bit address space), where the
probe's threads run.

The probe's pick, `pktito-c info`:

| machine                    | probe                                                     | pick             |
| -------------------------- | --------------------------------------------------------- | ---------------- |
| pi5a (A76)                 | neon-dp-a76 0.910-0.913 on 4 cores                        | neon-dp-a76, 3/3 |
| c6g (Graviton2, N1)        | neon-dp-a76 0.917-0.919 on 4 cores                        | neon-dp-a76, 5/5 |
| c7g, c8g, c9g (V1, V2, V3) | none: neon-sve competes with nothing                      | neon-sve, 5/5    |
| S25 (2 prime cores)        | neon-i8mm 0.722-0.756, neon-dp-a76 0.949-0.956 on 2 cores | neon-i8mm, 5/5   |
| M4 mini                    | neon-i8mm 1.119, neon-dp-a76 0.976-0.977                  | neon-dp, 3/3     |
| armlet (7c, 2 A76 + 6 A55) | neon-dp-a76 1.085-1.165 on 2 cores                        | neon-dp, 8/8     |

pktito-c bench, test.wav, the default threads, medians of 5 rounds in
rotated order (EC2: v3; the boards: v2):

| machine                 | compiler | neon-dp normal / fast | neon-dp-a76 normal / fast | change                   |
| ----------------------- | -------- | --------------------- | ------------------------- | ------------------------ |
| pi5a                    | gcc 14   | 3.917                 | 3.567                     | -8.9%                    |
| c6g                     | gcc 13   | 3.153 / 2.754         | 2.931 / 2.600             | -7.0% / -5.6%            |
| c6g                     | clang 18 | 3.130 / 2.848         | 2.914 / 2.583             | -6.9% / -9.3%            |
| c6g, 1 thread, 3 rounds | gcc 13   | 12.476                | 11.562                    | -7.3%                    |
| c7g                     | gcc 13   | 2.368 / 2.205         | 2.450 / 2.243             | (neon-sve 1.937 / 1.723) |
| c8g                     | gcc 13   | 1.858 / 1.759         | 1.984 / 1.840             | (neon-sve 1.645 / 1.498) |
| c9g                     | gcc 13   | 1.265 / 1.207         | 1.357 / 1.261             | (neon-sve 1.075 / 0.979) |

On the V cores the A76's tile is 3.5 to 7.3% the slower, 12.9% on one
c9g core (5.206 s against 4.620), which is what its probe is for; they
take neon-sve before it in any case.

armlet: in the morning its two A76s read the tile at 0.910 to 0.914 at
once; at 21:58 they read 1.085 to 1.165, and test.wav held to them (`start
/affinity C0`, 2 threads) agreed, neon-dp 7.314 s against neon-dp-a76's
7.782. At 8 threads, two rounds each, neon-dp 6.558 and 5.404 s,
neon-dp-a76 4.993 and 5.304: too noisy to rank. The board changed between
the two readings, and the probe read it each time; where the tile is not
3% the faster, a process runs neon-dp, as main does.

## 2026-11-23 - the M4 Pro as 4332856 stands (a measurement)

Before anything is changed: 4332856, Apple clang 21.0.0, macOS 27.0.1,
the M4 Pro mini (8 P cores in two clusters of 4, 4 E cores), v2,
test.wav (60.0 s in chunks of 28.6, 28.0 and 3.2 s, 232 tokens). The
machine is in use, and through this day shared with another session's
builds; these were taken while it was idle.

`pktito-c bench` at its default, 12 threads in two lanes of 6: best of 5
from 0.397 to 0.447 s over the morning's runs, 134 to 151x. One run's
best of 5 to the next moves by about 5% here, so this day's comparisons
are rounds in rotated order, each binary once a round, and the median of
the rounds (ab.py, with the session's notes).

| threads | test.wav's three chunks | one chunk of 60 s (--max-chunk 60) |
| ------- | ----------------------- | ---------------------------------- |
| 1       | 3.141 s                 | 3.644 s                            |
| 2       | 1.627                   |                                    |
| 4       | 0.833                   |                                    |
| 6       | 0.566                   |                                    |
| 8       | 0.467                   | 0.545                              |
| 10      | 0.430                   |                                    |
| 12      | 0.427                   | 0.496                              |

Eight threads are 6.7 times one, and the four E cores add 9% to them.

The stages (make PROFILE=1, which runs one lane), ms a run:

| stage                             | 12 threads, 438 ms | 8 threads, 445 ms |
| --------------------------------- | ------------------ | ----------------- |
| feed-forward                      | 241.4 (51.1%)      | 253.6 (54.0%)     |
| q k v and the positions           | 80.6 (17.1%)       | 77.4              |
| pw1 and pw2                       | 45.8 (9.7%)        | 44.4              |
| attention (float) and its softmax | 44.7 (9.5%)        | 45.2              |
| attention out                     | 16.5 (3.5%)        | 16.1              |
| quantize                          | 13.8               |                   |
| subsampling                       | 11.4               |                   |
| GLU and the depthwise             | 9.7                |                   |
| decoder                           | 3.4                |                   |
| layer norms, mel, positions       | 2.5, 1.8, 0.7      |                   |

The int8 products are 82% of a run: about 470 G multiply-adds (24
layers, each two feed-forwards of 4096 x 1024 and 1024 x 4096, q k v
and the positions, the attention's output, pw1 and pw2) in 385 ms, 1.23
T a second from twelve threads.

tests/q8bench, one thread, T 750, G multiply-adds a second at blocks of
32 / 64:

| the product                    | neon-dp       | neon-dp-a76   | neon-i8mm |
| ------------------------------ | ------------- | ------------- | --------- |
| feed-forward up, 4096 x 1024   | 188.0 / 202.3 | 193.5 / 194.7 | 168.6     |
| feed-forward down, 1024 x 4096 | 181.0 / 196.5 | 191.5 / 190.6 | 168.6     |
| attention q, 1024 x 1024       | 186.6 / 201.7 | 187.2 / 196.8 | 169.0     |

The assembly block loop is 64 SDOT, 8 SCVTF and 8 FMLA for 1024
multiply-adds, 80 instructions on four SIMD pipes: 51 multiply-adds a
cycle at best, and 188 G is about 42 at 4.5 GHz. The tile has little
left; what the run leaves is between the tiles: eight cores of 188 G are
1.50 T and four E cores more, against the 1.23 T the products get.

## 2026-11-23 - the pool's spin on the M4 Pro: 6 us against none, 62 and 620 (a measurement)

1eb53f8 left it open: "Whether 6 us outlasts the M4's gaps between a
layer's jobs ... is not measured here." PK_SPIN is 20000 polls, 6.2 us
here (tests/spinbench). 4332856 built with others (make OPT="-O3
-DPK_SPIN=..." over a copy whose pool.c takes the define), bench in
rounds in rotated order, min / median / max in seconds and the median
against the tree's own binary's in the same rounds:

| PK_SPIN          | test.wav, 12 threads in two lanes | one 60 s chunk, 8 threads        |
| ---------------- | --------------------------------- | -------------------------------- |
| 0                | 0.407 / 0.415 / 0.418, +1.2% (6)  | 0.526 / 0.529 / 0.531, +1.9% (5) |
| 2000 (0.6 us)    | 0.407 / 0.411 / 0.414, +0.2% (6)  |                                  |
| 20000, rebuilt   | 0.418 / 0.436 / 0.438, +0.7% (7)  |                                  |
| 200000 (62 us)   | 0.426 / 0.442 / 0.445, +2.1% (7)  | 0.515 / 0.516 / 0.518, -0.6% (5) |
|                  | 0.403 / 0.417 / 0.425, +1.6% (6)  |                                  |
| 2000000 (620 us) | 0.460 / 0.486 / 0.489, +12.2% (7) | 0.515 / 0.516 / 0.517, -0.6% (5) |

(rounds in brackets; the tree's binary read 0.413 / 0.433 / 0.447 in the
seven, 0.399 / 0.410 / 0.414 in the six and 0.518 / 0.519 / 0.520 in the
five.) No spin at all costs 1 to 2%. A longer one costs where twelve
threads share eight P cores, a spinning thread holding a P core while a
working one sits on an E core, and buys 0.6% at most where every thread
has a P core. 6 us stays. A build given its define through CFLAGS= had
no NEON backend at all (the Makefile adds each backend's flags and
-DPK*HAVE* to CFLAGS) and ran ref at 20 s; OPT= is the way.

## 2026-11-23 - SMOPA at the encoder's blocks of 32, with its loads, its packing and its drain, a cluster's threads sharing its unit (a measurement)

The 2026-10-24 entry put SMOPA at blocks of 32 at 0.73 T a unit on
registers, "about what eight cores of SDOT give the product now", and
left it; the 2026-10-28 one gave it loads at a scale a row and found
SDOT threads beside it adding nothing. Neither ran the engine's own
arithmetic with loads, and neither asked what a unit gives when every
thread of its cluster is on it. This does: a probe (smeq8.c, with the
session's notes, not in the tree) that is pk_k_gemm_q8 as kernels.h
defines it. A panel's groups are SMOPA's first operand as the file holds
them; 16 columns' bytes are packed to its second with NEON (16 columns
by 4 inputs a group, their 16 scales a block beside them; 8 us for 256
columns of 1024, 28 to 32 GB/s); block b's 8 SMOPAs go to int tile b % 2
while block b - 1's is drained into a float tile (MOVA vgx4, SCVTF, one
FMLA vgx4 by the block's 16 scales, ZERO), so each output's chain is
ref's fused multiply-adds in ref's order; the row scales, bias and add
come out of the float tile a column at a time. 0 outputs of 480 shapes'
differ from pk_kernels_ref.gemm_q8's (64 x 1024, 16 x 4096, 128 x 256,
32 x 32 and 48 x 64; 1 to 64 columns; blocks of 32 and 64; bias and add
both ways; a block of zeros and one outlier among the activations).

4332856's build for SDOT (pk_kernels_neon_dp.gemm_q8), Apple clang 21,
-O2 -mcpu=apple-m4 -ffp-contract=off for the probe, the M4 Pro mini, T
256, blocks of 32, G multiply-adds a second. The machine is shared with
another session's builds today: the one-thread rows were taken at a load
average of 6 to 10 and falling with SDOT reading its usual 186 to 196,
the rest as that session's check-c ended; to be taken again quiet.

One thread:

| the product                             | SMOPA, packed in the call | packed once | SDOT    |
| --------------------------------------- | ------------------------- | ----------- | ------- |
| feed-forward up, 4096 x 1024, one call  | 780                       | 762-775     | 190     |
| feed-forward down, 1024 x 4096          | 801                       | 788-810     | 186     |
| attention q, 1024 x 1024                | 785                       | 796         | 196     |
| the same in slices of 64 rows           | 544 / 566 / 574           | 753-795     | 187-190 |
| in slices of 256 rows                   | 727 / 734 / 735           | 758-795     | 187-190 |
| at blocks of 64 (the fast mode), a call | 992 / 1007 / 960          |             | 202-210 |

Packed once, an item of 64 rows by 16 columns runs as fast as a whole
matrix (753 to 773 G against 762 to 775): entering streaming mode a call
costs nothing that shows. Packed in the call, a slice of 64 rows packs
its 256 columns for 64 rows' work, and gives up 30%.

Threads, feed-forward up, items of 64 rows by 48 columns from one
counter, packed once, each thread's CPU and rate read in the best of 15
runs (pthread_cpu_number_np: 0 to 3 are the E cores, 4 to 7 and 8 to 11
the two P clusters):

| threads         | G a second | each thread's                                                            |
| --------------- | ---------- | ------------------------------------------------------------------------ |
| 8 SDOT          | 1298-1454  |                                                                          |
| 1 SMOPA         | 784        |                                                                          |
| 2 SMOPA         | 1335       | 673 and 664, a cluster each                                              |
| 3 SMOPA         | 1768       | 700 alone in its cluster; 537 and 537 sharing the other's unit           |
| 4 SMOPA         | 2136       | 541 and 541 in one cluster, 530 and 531 in the other                     |
| 6 SMOPA         | 2266       | 528 and 527 in one cluster; 302 to 306 each for four in the other        |
| 8 SMOPA         | 2571       | 394 each for three in one cluster, 304 to 307 for four; 187 on an E core |
| 4 SMOPA, 4 SDOT | 1957       | SMOPA's four in one cluster at 305 to 308, SDOT's in the other at 184    |
| 2 SMOPA, 6 SDOT | 1869-2108  | one SMOPA thread on an E core in the slower run                          |

A unit is not one thread's: its cluster's threads share it, and it gives
more the more of them feed it, 780 G to one, 1070 to two, 1180 to
three, 1220 to four, where four cores of SDOT give 736 to 750. The
2026-10-28 threads "3 to 12 times slower than the fastest" on registers
are not seen with the loads and the drain: four threads on a unit run
level, 302 to 306 each. Every P thread on SMOPA is 2.4 to 2.6 T against
SDOT's 1.3 to 1.45 T, wherever the system puts the threads, which macOS
gives a process no say in. The best here would be two threads a cluster
on SMOPA and two on SDOT, 1070 + 2 x 184 = 1438 G a cluster, 18% over
four on SMOPA; it needs each thread to know its cluster. The E cluster's
unit gave its one thread 187 G, an E core's SDOT tile 29 to 44
(tests/q8bench under `taskpolicy -b`, the machine busy).

## 2026-11-23 - neon-sme: the int8 product on SMOPA, every thread's, at ref's bits (hypothesis)

The engine's int8 products are about 470 G multiply-adds a run of
test.wav (24 layers, each two feed-forwards of 4096 x 1024 and 1024 x
4096, q k v and the positions, the attention's output, pw1 and pw2), 385
ms of a 438 ms run at 12 threads (make PROFILE=1, 4332856): 1.23 T a
second, 82% of the run. The entry above has the M4 Pro's two P clusters'
units at 2.4 to 2.6 T on the same arithmetic with all eight P threads on
them, 1.8 times eight cores of SDOT.

So: a backend, neon-sme, that is neon-dp with pk_k_gemm_q8 on SMOPA as
the probe has it, for a CPU with SME2 at a streaming vector of 512 bits
(the weights' 64-byte group is then SMOPA's operand whole). It competes
(kernels.h), so backend/kernels.c's probe, every one of the fastest
cores at once on a slice of 64 rows, takes it only where it measures 3%
the faster so, which is the case that decides it: a unit shared by its
cluster. Expected there 300 against 184 G a thread, 0.6 of neon-dp's
time.

- The activations are packed in the call, 16 columns at a time into a
  thread's buffer (16 x n bytes: 16 KB at 1024, 64 KB at 4096), as
  amx-int8 copies a short group: the backend alone knows of it, and
  core/ is untouched. One thread gives up 30% to it in slices of 64 rows
  (544 to 574 G against 753 to 795); a thread sharing a unit with three
  others packs on its own core while the unit serves them, so less is
  expected to show at 8 threads. Packing once a product (a place in
  pk_act, the quantizer's to fill) is the next step if it does.
- The last columns of a call, fewer than 16: padded with zeros and
  multiplied as 16, or left to SDOT's tiles, whichever the rates say (16
  columns at a shared unit's 305 G cost what 9.7 do at SDOT's 184).
  pk_gemm's calls are whole twelves of columns, 48 to 60 at 12 threads.
  The decoder's products are one column a call and stay SDOT's.
- Blocks of any length a multiple of 4 that divides n; n to 4096, the
  model's widest (wider goes to SDOT, as amx-int8's short group does).
- Darwin only for now: it is measured on the M4 Pro alone, and the
  streaming vector length and the feature are asked of sysctl
  (hw.optional.arm.FEAT_SME2, hw.optional.arm.sme_max_svl_b). Linux and
  Windows get it when a board with SME2 is measured.

Expected end to end: the int8 stages from 385 ms toward 200 at 12
threads, test.wav 0.42 s toward 0.25 to 0.27; more at one thread (780 G
against 188, a 3.1 s run toward 1.3). Kept if `pktito-c bench` moves by
more than the 5% a run moves here, in rounds in rotated order against
4332856, with tests/kcheck and check-c passing and the transcript's
tokens the same. What may not carry over from the probe: the lanes (two
of 6 threads, each lane's threads on both clusters' units), the E cores'
threads on their own unit (187 G for one; four not measured), and
whether the system keeps eight streaming threads spread four a cluster.

## 2026-11-23 - neon-sme: kept, test.wav 0.428 -> 0.291 s at the default; the probe's tables again, quiet (result)

backend/neon-sme (PK_NEON_SME: backend/neon/kernels.c's k_gemm_q8 on
SMOPA as the hypothesis has it; "sme2" in core/os.c's cpu_has on Darwin;
first in pk_kernels_built, competing). Apple clang 21.0.0, the M4 Pro
mini, v2, test.wav, the other session stopped for these (load average 3
to 4 before the first).

tests/kcheck: neon-sme's 1752 cases are ref's bits, and `make check-c`
passes with it the backend the Mac picks (v2 and v3 and the fast mode's
transcripts ref's every token and time, the diarizer's and the stream's
with them). `pktito-c info`: "probe neon-sme 0.355, neon-i8mm 1.123,
neon-dp-a76 0.977 of neon-dp's time, timed in 5.3 ms".

`pktito-c bench` in rounds in rotated order against 4332856's build, min
/ median / max in seconds, 232 tokens in every run:

|                                          | 4332856, neon-dp      | neon-sme              |        |
| ---------------------------------------- | --------------------- | --------------------- | ------ |
| the default, 12 threads in two lanes (7) | 0.395 / 0.428 / 0.445 | 0.289 / 0.291 / 0.293 | -32.0% |
| the same an hour before, shared (5)      | 0.391 / 0.428 / 0.449 | 0.290 / 0.291 / 0.296 | -32.0% |
| -t 8 (5)                                 | 0.533 / 0.537 / 0.539 | 0.349 / 0.351 / 0.351 | -34.6% |
| one chunk of 60 s, 12 threads (5)        | 0.518 / 0.520 / 0.533 | 0.392 / 0.393 / 0.399 | -24.4% |

The same binary with PKTITO_KERNELS=neon-dp read 0.406 / 0.421 / 0.449
in the shared rounds, 1.6% under 4332856's: the build is the same but
for the new backend. 4332856 at -t 8 by itself read 0.454, 0.478 and
0.489 a minute after its 0.537 in the rounds, and 0.467 in the morning:
eight threads are wherever macOS puts them among twelve cores, and its
rows move more than the default's. 206 times real time at the default.

neon-sme by threads, test.wav's three chunks, best of 5 (4332856's in
the entry above): 1: 1.364 s (3.141), 2: 0.705, 4: 0.578, 6: 0.468, 8:
0.348, 10: 0.316, 12: 0.291. One thread is 2.3 times neon-dp's; four are
barely over two, where macOS has the four on one cluster and one unit.

The stages, make PROFILE=1 (one lane), ms a run, neon-dp forced in the
same build beside it:

| stage                             | 12 threads: neon-dp | neon-sme | 8 threads: neon-dp | neon-sme |
| --------------------------------- | ------------------- | -------- | ------------------ | -------- |
| the run                           | 453                 | 375      | 470                | 401      |
| feed-forward                      | 237.5               | 179.6    | 256.7              | 187.8    |
| q k v and the positions           | 77.8                | 66.4     | 77.2               | 70.4     |
| pw1 and pw2                       | 44.4                | 33.7     | 45.7               | 35.8     |
| attention out                     | 15.6                | 12.3     | 16.1               | 13.3     |
| attention (float) and its softmax | 43.5                | 42.8     | 47.5               | 55.1     |
| quantize                          | 11.9                | 13.8     | 6.4                | 10.8     |
| GLU and the depthwise             | 9.3                 | 10.5     | 6.5                | 10.1     |

The int8 stages are 375 -> 292 ms at 12 threads in one lane, not the
200 the hypothesis expected. A timer around each item's kernel and its
tile's SiLU and quantize (a copy of core/passes.c, with the session's
notes) says where the rest is: as jobs the products run at 1.45 T
(feed-forward up), 1.94 T (down) and 1.60 T (the rest) at 12 threads,
1.49, 1.70 and 1.49 T at 8, where the probe's threads, never off the
units, get 2.43 to 2.61 T. A thread in the kernel runs at the probe's
rate (289 G at 8 threads, the probe's 305 less its packing), but it is
in the kernel 65 to 81% of a job: 15 to 18% of feed-forward up is the
tile's SiLU and quantize on its core, and the rest is packing and the
wait for the job's last item. A unit with no thread at it is idle, as
a core never is for its own thread, and a job's threads leave it
together: at 8 threads feed-forward up is 16 items of 256 rows by every
column, two a thread, so the eight run SMOPA at once, then their tiles
at once. Two lanes cover some of that, each lane's threads at the units
while the other's are elsewhere (0.291 s against one lane's 0.375).

So kept, at -32%, with what the probe promised not all had. Open:

- a unit's idle time: fewer threads a unit and SDOT on the cores for the
  rest (the next entry), smaller items, packing once a product;
- PK_SME_SHORT, the columns under which a call's last few are SDOT's
  (10, from the rates above, not tuned in the engine);
- a second compiler: this Mac has Apple clang alone, and neon-sme builds
  on macOS alone.

The probe's tables again with the machine quiet (17:57, the other
session stopped), T 256, G multiply-adds a second. One thread:

| the product                            | SMOPA, packed in the call | SDOT |
| -------------------------------------- | ------------------------- | ---- |
| feed-forward up, 4096 x 1024, one call | 804                       | 190  |
| feed-forward down, 1024 x 4096         | 804                       | 187  |
| attention q, 1024 x 1024               | 790                       | 190  |
| the three in slices of 64 rows         | 511 / 578 / 575           |      |
| in slices of 256 rows                  | 735 / 745 / 735           |      |
| at blocks of 64, a call                | 1013 / 1013 / 993         | 205  |

Threads, feed-forward up, items of 64 rows by 48 columns, packed once,
best of 15 runs and each thread's rate in it:

| threads         | G a second | each thread's                                                   |
| --------------- | ---------- | --------------------------------------------------------------- |
| 8 SDOT          | 1460       | 183 to 184                                                      |
| 12 SDOT         | 1627       | 180 to 183 on the P cores, 54 to 63 on the E cores              |
| 1 SMOPA         | 786        |                                                                 |
| 2 SMOPA         | 1472       | 730 and 742, a cluster each (the median run, both in one: 1093) |
| 3 SMOPA         | 1760       | 686 alone in its cluster, 538 and 537 sharing the other's       |
| 4 SMOPA         | 1915       | 542 and 546 sharing, 644 alone, 188 alone on the E cluster's    |
| 6 SMOPA         | 2295       | 302 to 307 for four on one unit, 541 and 544 on the other       |
| 8 SMOPA         | 2430       | 305 to 307                                                      |
| 12 SMOPA        | 2608       | 305 to 307 on the P cores, 58 to 61 on the E cores              |
| 4 SMOPA, 4 SDOT | 1953       | SMOPA's four on one unit at 304 to 309, SDOT's at 184           |
| 2 SMOPA, 6 SDOT | 2110       | SMOPA's two on one unit at 529 and 534, SDOT's 175 to 184       |

The shared run's figures stand: a unit gives one thread 786 G, two 1070
to 1090, four 1220 to 1230, and four cores of SDOT 736. The E cluster's
unit gives one thread 188 G and four 58 to 61 each, about 240 together,
where four E cores of SDOT give 54 to 63 each: level. The weights being
out of the caches costs SMOPA nothing that shows: 30 matrices of 4096 x
1024 taken in turn ran 2321 G on eight threads against 2333 for one
taken again and again (SDOT 1454 against 1463), in the shared hour.

## 2026-11-23 - neon-sme: a unit for two or three of its cluster's threads and SDOT for the rest, small items, the columns packed once a product (hypothesis)

The result above leaves the units short of threads for part of every
job, and the probe's table says a unit does not want all four of its
cluster's anyway: it gives 1070 to 1090 G to two threads, 1180 to three
and 1220 to four, and a core off the unit has its own SDOT, 184 G. Two
on SMOPA and two on SDOT are 1438 G a cluster, three and one 1364, where
four on SMOPA are 1220: +18% and +12%, and whatever a thread on SDOT
does it does without waiting at the unit.

A thread cannot be given a cluster on macOS, but it can ask where it is
(pthread_cpu_number_np; the clusters' sizes from sysctl's
hw.perflevelN.cpusperl2). So: a count for each cluster of its threads in
SMOPA, and a call takes SMOPA where its cluster's count is under K and
SDOT's tiles where it is not. Both are ref's bits, so the transcript
cannot depend on who took what. The probe with each thread choosing so
an item (AUTO=K, the machine shared, to be taken again quiet): 12
threads 2855 G at K = 2 and 2884 at K = 3 against 2566 with every
thread on SMOPA.

What it needs with it, each to be measured in a copy before the tree
changes:

1. Smaller items. A thread on SDOT takes an item about three times as
   long as one with half a unit, and a job waits for its last item. In a
   lane of 6 threads feed-forward up is 48 items of 256 rows by about
   128 columns, 33.5 M multiply-adds: 182 us on SDOT in a job of about 300. Items of 64 rows by 48 columns are 17 us there.
2. The columns packed once a product. At 64 rows an item packs its
   columns for a quarter of the work 256 rows gave them: one thread
   loses 30 to 36% to packing in the call there (511 to 578 G against
   786 to 804), and with K threads at a unit nothing hides it. A
   product's 16 columns packed by the first item that needs them, for
   every item after (core/passes.c, a place for them beside the
   activations), is 18 KB a frame a layer, 0.33 GB a run, where packing
   in the call is 7 GB.

Expected: the int8 stages from 1.45 to 1.94 T as jobs toward 2.4 T and
over; test.wav at the default from 0.291 s toward 0.25. Kept, each
part, if bench moves by more than neon-sme's own spread (0.289 to 0.293
in 7 rounds), kcheck and check-c passing.

## 2026-11-23 - neon-sme: the columns packed once a product kept, 0.292 -> 0.263 s; a unit's threads limited, and small items, rejected (result)

Each part first in a copy of ba52667 with switches (PK_SME_K: a call
takes SMOPA where fewer than K threads of its cluster are in it, by
pthread_cpu_number_np and sysctl's hw.perflevelN.cpusperl2, else SDOT's
tiles; the item's rows and columns; a product's sixteens of columns
packed by the first item to need them), the machine quiet, v2, test.wav
at the default, 5 rounds in rotated order, medians, 232 tokens in every
run:

|                                   | every thread on SMOPA | K = 3        | K = 2 |
| --------------------------------- | --------------------- | ------------ | ----- |
| ba52667                           | 0.289 to 0.292        | 0.293        | 0.307 |
| items of 64 rows by 48 columns    |                       |              | 0.308 |
| packed once                       | 0.270, 0.273          | 0.266, 0.271 | 0.277 |
| packed once, items of 64 by 48    | 0.284                 | 0.276        | 0.285 |
| packed once, slices of 128 rows   | 0.277                 | 0.280        |       |
| packed once, slices of 256 rows   | 0.293                 | 0.297        |       |
| packed once, chunks of 96 columns | 0.271                 | 0.273        |       |
| packed once, -t 8                 | 0.321                 | 0.321        | 0.338 |
| packed once, one chunk of 60 s    | 0.373                 | 0.372        | 0.378 |

**A unit's threads limited: rejected.** The probe, quiet, each thread
choosing so an item (items of 64 by 48 packed once, G a second, best of
15):

| threads | K = 1 | K = 2 | K = 3 | every thread |
| ------- | ----- | ----- | ----- | ------------ |
| 8       | 1977  | 2662  | 2674  | 2413         |
| 12      | 2231  | 2968  | 2871  | 2605         |

+10 to 14% there, and in the engine K = 3 is inside the rounds' spread
(0.264 to 0.277 against 0.269 to 0.274) and K = 2 costs 3 to 6%. The
probe's threads are at the kernel every moment, and the limit moves the
fourth to its own SDOT; the engine's are in their tiles' SiLU and
quantize or waiting on a job's last item for a fifth to a third of a
job, a unit there is short of threads more often than it has one too
many, and a thread on SDOT holds its item three times as long with the
job waiting on it. Not built into the tree. It may be what a larger
cluster needs: sysctl has an M5 Ultra's 36 cores in clusters of six and
a base M4's E cores in one of six, where six threads on a unit would
share the 1.22 T that four do here and six cores of SDOT are about 1.1
T. Neither is measured; macOS's probe is one thread's and would take
neon-sme there on 0.36.

**Small items: rejected.** 64 rows by 48 columns cost 4 to 5% with the
columns packed once (0.284 against 0.270), and slices of 128 and 256
rows for every product 2 and 8%. pk_gemm_on's shapes stay.

**The columns packed once a product: kept.** kernels.h has a packed
order as a thing a backend may have (pk_k_packs, pk_k_act_pack, and
pk_k_gemm_q8_packed: pk_k_gemm_q8 given the columns that way as well);
neon-sme's is SMOPA's operand. core/passes.c packs a product's sixteens
of columns into a place of the transcriber's (pk_packed, carved with its
other arrays: 1.6 MB and a fifth for 30 s), each by the first item that
needs it, the items after reading it; a chunk of columns is then whole
sixteens. The V product's panels read them too: a panel was a call of 16
rows that packed every column for itself. The diarizer's products pack
in the call as before (pk_gemm_on's NULL).

tests/kcheck: neon-sme's cases are 2003 to 2033 with
pk_k_gemm_q8_packed's, every column and a run of columns from a
sixteen's first, all ref's bits; q8check's 385 at guard pages; `make
check-c` passes. A build with -fsanitize=address,undefined and one with
-fsanitize=thread (PKTITO_KERNELS=neon-sme there: under it the probe
takes neon-dp-a76) ran kcheck and test.wav at the default and as one
chunk at 8 threads with nothing reported and the transcript's bytes the
same.

`pktito-c bench`, rounds in rotated order, medians, the machine quiet:

|                                                       | 4332856 | ba52667 | packed once |                |
| ----------------------------------------------------- | ------- | ------- | ----------- | -------------- |
| the default, 12 threads in two lanes (7)              | 0.424   | 0.292   | 0.263       | -38.0%, -9.9%  |
| -t 8 (5)                                              | 0.541   | 0.348   | 0.301       | -44.4%, -13.5% |
| one chunk of 60 s (5)                                 | 0.538   | 0.397   | 0.364       | -32.3%, -8.3%  |
| v3, the default (3), 220 tokens                       | 0.453   |         | 0.265       | -41.5%         |
| the fast mode, the default (3)                        | 0.421   |         | 0.232       | -44.9%         |
| test.wav ten times, 600 s, 21 chunks (3), 2329 tokens | 4.662   | 2.867   | 2.593       | -44.4%, -9.6%  |

(min to max at the default: 0.393 to 0.452, 0.290 to 0.294, 0.258 to
0.265.) 228 times real time on test.wav, 231 on the ten minutes, where
4332856 is 141 and 129. By threads, best of 5: 1: 1.270 s, 2: 0.682, 4:
0.496, 6: 0.402, 8: 0.300, 10: 0.273, 12: 0.259; and in the copy 14:
0.272 and 16: 0.291 against its 12's 0.270, so a thread a core stays.

The stages, make PROFILE=1 (one lane), ms a run:

| stage                             | 12 threads: neon-dp | neon-sme | 8 threads: neon-dp | neon-sme |
| --------------------------------- | ------------------- | -------- | ------------------ | -------- |
| the run                           | 433                 | 335      | 483                | 362      |
| feed-forward                      | 229.2               | 163.0    | 262.5              | 170.5    |
| q k v and the positions           | 76.7                | 54.7     | 81.1               | 57.1     |
| pw1 and pw2                       | 43.3                | 29.0     | 47.6               | 30.2     |
| attention out                     | 15.4                | 10.8     | 16.9               | 11.0     |
| attention (float) and its softmax | 42.6                | 43.2     | 47.1               | 55.3     |
| quantize                          | 13.1                | 15.3     | 8.1                | 11.2     |
| subsampling                       | 10.7                | 10.0     | 12.3               | 12.0     |
| GLU and the depthwise             | 9.4                 | 10.7     | 7.3                | 9.6      |

The int8 products are 258 ms of the lane's 335 at 12 threads, 1.83 T a
second where neon-dp's are 1.29 T and the probe's ceiling is 2.6: the
units still wait on their threads' other work, less than they did.

PK_SME_SHORT, the columns under which a call's last few are SDOT's: 4,
10 and 16 (never padded) read 0.262, 0.261 and 0.264 s at the default
and 0.373, 0.372 and 0.377 as one chunk. 10 stays.

Set aside without building: FMOPA for the attention's float products. A
run is 21.8 G fused multiply-adds of them (4096 T^2 a layer), done with
the softmax in 43 ms by twelve threads, 0.5 T a second or more. FMOPA is
256 multiply-adds an instruction where SMOPA is 1024: at SMOPA's 0.77 G
instructions a second a unit that is 0.2 T, 0.4 T for both, under what
the cores' FMLA gives now, and taken from the int8 products.

Open: the diarizer's products packed once (a pk_packed sized for its
activations); a unit's limit on a cluster of six; a second compiler; and
the units' idle time, a third of their ceiling still.

## 2026-11-23 - the probe's reading for neon-sme, later the same day (a measurement)

`pktito-c info` after the afternoon's rounds: "probe neon-sme 0.395 to
0.397" from ba52667's binary, which read 0.355 at its build, and 0.400
to 0.402 from 2dfc43f's, six runs each. The same binary moves by a
tenth with the machine's state; 2dfc43f's call that packs its own
columns is 1% behind ba52667's (k_act_pack reads a column of zeros by a
mask where sme_pack had 4096 zeros to read). Either is far under the
0.97 the probe asks. backend/kernels.c's comment says 0.36 to 0.40.

## 2026-11-23 - neon-sme on an M5 Ultra and a base M4: the fleet's rounds (a measurement), and a unit's limit again (hypothesis)

95bc7c6 against 4332856 on two more of Apple's chips (the fleet's check:
v2, test.wav, Apple clang 21, best of 5 a round, each machine quiet):

|                                 | 4332856, neon-dp | 95bc7c6, neon-sme | 95bc7c6, neon-dp forced |
| ------------------------------- | ---------------- | ----------------- | ----------------------- |
| M5 Ultra, 36 threads (8 rounds) | 0.183 to 0.185   | 0.202 to 0.206    | 0.179 to 0.180          |
| M5 Ultra, -t 12                 | 0.276 to 0.285   | 0.205 to 0.207    |                         |
| base M4, 10 threads (4 P + 6 E) | 0.576 to 0.582   | 0.418 to 0.422    |                         |

The base M4 gains 27%. The M5 Ultra loses 10% at its default: its 36
cores are six clusters of six (sysctl's cpusperl2), its neon-sme is the
same 0.205 s at 12 threads and at 36, and neon-dp, which every core adds
to, passes it at 36. The probe that took neon-sme there is one thread's
(macOS gives no affinity to time a cluster's threads with), reading the
unit a thread has to itself. A pick that is the slower on a machine is a
defect of 95bc7c6.

Hypothesis. A unit gives what it gives however many threads share it
(on the M4 Pro 0.79 T to one, 1.07 to two, 1.18 to three, 1.22 to four),
so the fourth to sixth threads of a cluster add nothing on SMOPA and
would add a core's SDOT each (0.18 T here). The limit that was inside
the spread on the M4 Pro's clusters of four (above: K = 3 0.266 and
0.271 against 0.270 and 0.273) should be worth a third of the product on
a cluster of six: K threads of a cluster in SMOPA at once, counted by
cluster, the others on SDOT's tiles for that call. And against what cost
K = 2 here, a thread on SDOT holding an item three times as long with
the job waiting on it: a call in pieces of 48 columns, each piece asking
for the unit again, so that the thread is on SMOPA from the moment a
place is free. Expected: the M5 Ultra at 36 threads under neon-dp's
0.180 s, the M4 Pro no slower than 0.263, and the base M4's six E cores
(their unit 0.19 T to one thread, 0.24 to four; their SDOT 0.06 each)
better with one or two on the unit than with six. To be measured on all
three before the tree changes.

## 2026-11-23 - neon-sme: a unit two threads' at a time, kept; an M5 Ultra's ten minutes 1.188 -> 0.933 s where 95bc7c6 took 1.261 (result)

In a copy of 95bc7c6 with switches (K threads of a cluster in SMOPA at
once, counted by the cluster pthread_cpu_number_np's CPU is in; a
refused call's SDOT in pieces of P columns, asking again after each),
`pktito-c bench`, v2, rounds in rotated order, medians, each machine
quiet. "Ten minutes" is test.wav ten times over, 600 s in 21 chunks,
which fills an M5 Ultra's nine lanes where test.wav's three chunks fill
three; 232 and 2329 tokens in every run.

M5 Ultra, 36 threads (six clusters of six), Apple clang 21:

|                          | test.wav | one chunk of 60 s | ten minutes | -t 24 | -t 12 |
| ------------------------ | -------- | ----------------- | ----------- | ----- | ----- |
| neon-dp, forced          | 0.184    | 0.229             | 1.203       | 1.756 | 2.863 |
| every thread on its unit | 0.209    | 0.248             | 1.265       | 1.364 | 2.000 |
| K = 1, P = 48            |          |                   | 0.968       | 1.317 | 1.926 |
| K = 2, the call whole    | 0.180    | 0.218             | 0.964       | 1.236 | 1.985 |
| K = 2, P = 48            | 0.176    | 0.209             | 0.943       | 1.253 | 1.986 |
| K = 2, P = 16            | 0.173    | 0.205             | 0.938       | 1.200 | 1.997 |
| K = 3, P = 48            | 0.179    | 0.214             | 1.006       | 1.310 | 1.990 |
| K = 4, the call whole    | 0.189    |                   | 1.089       |       |       |

(-t 24 and -t 12 are the ten minutes; 4332856 itself reads as the
copy's neon-dp, 0.184 and 1.201.) Base M4, 10 threads (a cluster of
four P and one of six E), Apple clang 21:

|                          | test.wav | ten minutes | test.wav, -t 4 |
| ------------------------ | -------- | ----------- | -------------- |
| 4332856, neon-dp         | 0.580    | 5.957       | 0.887          |
| every thread on its unit | 0.419    | 4.216       | 0.577          |
| K = 1, P = 16            | 0.415    | 4.118       |                |
| K = 2, the call whole    | 0.381    |             |                |
| K = 2, P = 16            | 0.375    | 3.772       | 0.526          |
| K = 2, P = 48            | 0.380    | 3.805       |                |
| K = 3, P = 16            | 0.379    | 3.841       | 0.540          |
| P 3 and E 1, P = 16      | 0.376    | 3.756       |                |
| P 2 and E 1, P = 16      | 0.377    |             |                |

M4 Pro, 12 threads (two clusters of four P, one of four E), against
95bc7c6's 0.259 and 2.556:

|                         | test.wav | ten minutes | one chunk | -t 8  |
| ----------------------- | -------- | ----------- | --------- | ----- |
| every thread (the copy) | 0.264    | 2.578       | 0.358     | 0.306 |
| K = 2, the call whole   | 0.268    |             |           |       |
| K = 2, P = 16           | 0.254    | 2.544       | 0.336     | 0.305 |
| K = 2, P = 48           | 0.262    | 2.535       |           |       |
| K = 3, P = 16           | 0.251    | 2.524       | 0.342     | 0.303 |
| K = 3, P = 48           | 0.254    | 2.502       |           |       |

(95bc7c6 reads 0.355 as one chunk and 0.302 at -t 8.) Two a unit is the fastest on the clusters of six, by 3 to 7% over one
and over three, and on the base M4; on the M4 Pro two and three are
inside each other's rounds (0.249 to 0.257 and 0.249 to 0.252). One
number for every cluster: one thread on the E cores' unit with two or
three on the P cores' bought under 1% on the base M4 (the table) and
cost 2 to 3% on the M4 Pro (0.263 against 0.256 with three on the P
units, 0.269 against 0.263 with two, in the pieces' first form below).

What did not hold of the hypothesis, and what else was tried:

- Pieces as first written, every call's columns in pieces of 48 each
  asking for the unit, cost 2 to 5% on the M5 Ultra at 12 threads, where
  the limit seldom refuses (the ten minutes 2.118 s against 2.021): each piece is a call
  of sme_cols, a start and stop of streaming mode and a ZA to clear. A
  call that has the unit now keeps it to its last column and only a
  refused call works in pieces; the tables above are that form, and at
  12 threads it is the unlimited build's 1.99 to 2.00.
- A floor under the limit, a call never refused while fewer than twice
  the units' number are in SMOPA on the whole machine, whichever
  clusters they are on: slower at every count (M5 Ultra, ten minutes, 36
  threads 1.006 against 0.956; test.wav 0.186 against 0.175). Rejected.
- The size of the gain on test.wav: 5% on the M5 Ultra, not the third
  the kernel's rates promised, because its three chunks give 36 threads
  three lanes and the longest chunk's twelve threads set the time. The
  ten minutes, nine lanes, gain 22%.

In the tree: core/os.c's pk_os_cluster (macOS: sysctl's perflevels, a
level's CPUs in runs of its cpusperl2, the last level's first; the
device tree's logical-cluster-id agrees on the three machines, the M5
Ultra's levels alternating by die, 12 M, 6 P, 12 M, 6 P, in runs of
six), and in backend/neon/kernels.c PK_SME_THREADS 2 and PK_SME_PIECE
16, a count of each cluster's calls in SMOPA, a line a cluster.
pk_k_gemm_q8_packed takes the unit at the first column it has a place
and gives SDOT pieces until then; pk_k_gemm_q8, which is a call of
sme_cols for each of its packings already, asks at each.

tests/kcheck has the product from a pool of 8 at once, 48 calls of each
of three shapes, whole and packed turn about, each held to ref's
(neon-sme 2177 cases; the other backends run them too, 1896). A copy
with the packed call's SDOT piece reading the first column's scales
fails 15 of them, one with the other call's SDOT step written a float
late 27 to 30. `make check-c` passes on the M4 Pro, the base M4 and the
M5 Ultra, the transcript's bytes those of 95bc7c6.

The tree's build, rounds in rotated order, medians (min to max):

|                                    | 4332856                | 95bc7c6                | a unit two threads'    |        |
| ---------------------------------- | ---------------------- | ---------------------- | ---------------------- | ------ |
| M5 Ultra, test.wav, 36 threads (5) | 0.183 (0.182 to 0.185) | 0.207 (0.207 to 0.208) | 0.173 (0.172 to 0.175) | -5.5%  |
| M5 Ultra, one chunk of 60 s (3)    | 0.229                  | 0.248                  | 0.206                  | -10.0% |
| M5 Ultra, ten minutes (3)          | 1.188 (1.180 to 1.194) | 1.261                  | 0.933 (0.931 to 0.935) | -21.5% |
| M5 Ultra, ten minutes, -t 24 (2)   | 1.740                  | 1.483                  | 1.192                  | -31.5% |
| M5 Ultra, ten minutes, -t 12 (2)   | 2.870                  | 1.986                  | 1.999                  | -30.3% |
| M5 Ultra, test.wav, -t 24 (3)      | 0.196                  | 0.172                  | 0.154                  | -21.4% |
| M5 Ultra, test.wav, -t 12 (3)      | 0.286                  | 0.206                  | 0.208                  | -27.3% |
| base M4, test.wav, 10 threads (5)  | 0.583 (0.576 to 0.584) | 0.421 (0.418 to 0.423) | 0.377 (0.376 to 0.378) | -35.3% |
| base M4, one chunk of 60 s (3)     | 0.679                  | 0.555                  | 0.487                  | -28.3% |
| base M4, ten minutes (2)           | 5.906                  | 4.204                  | 3.754                  | -36.4% |
| base M4, test.wav, -t 4 (3)        | 0.888                  | 0.579                  | 0.525                  | -40.9% |
| M4 Pro, test.wav, 12 threads (7)   |                        | 0.259 (0.257 to 0.263) | 0.255 (0.252 to 0.258) | -1.5%  |
| M4 Pro, one chunk of 60 s (5)      |                        | 0.367 (0.364 to 0.368) | 0.354 (0.349 to 0.359) | -3.5%  |
| M4 Pro, ten minutes (3)            |                        | 2.589 (2.577 to 2.615) | 2.533 (2.531 to 2.565) | -2.2%  |
| M4 Pro, test.wav, -t 8 (5)         |                        | 0.303 (0.301 to 0.307) | 0.307 (0.306 to 0.310) | +1.3%  |

(The last column is against 4332856 on the M5 Ultra and the base M4, and
against 95bc7c6 on the M4 Pro, whose 4332856 is 0.424 and 4.662 above;
the M4 Pro's 95bc7c6 is 2dfc43f's binary, the same code.)
The M5 Ultra's ten minutes are 643 times real time where neon-dp's are 505. `pktito-c info` reads the probe at 0.411 there and 0.383 on the
base M4; the pick is neon-sme on all three and the faster on all three.

The stages on the M4 Pro, make PROFILE=1, the minute whole at 8 threads,
medians of 5, ms a run: feed-forward 148.1, attention 127.9, q k v and
the positions 49.0, pw1 and pw2 26.6, decoder 17.9, subsampling 9.9,
attention out 9.3, GLU and the depthwise 7.5, quantize 7.2, mel 1.8,
norms 1.1; 95bc7c6's code reads 158.7 to 159.9, 129.4 to 129.9 and 50.5
to 50.8 for the first three. The int8 products are 233 ms, 2.0 T a
second. (The stages of 2dfc43f's entry above are of the three chunks at
the default; README's "Where the time goes" is of the minute whole, and
its shares after 2c09c24 were the chunks'. They are the whole's now.)

Open: a level whose clusters differ in size from another's on a chip of
two dies would be numbered wrong by pk_os_cluster (none of the three);
the E cores' unit, where one thread and two measured alike; the
diarizer's products packed once; a second compiler.

## 2026-11-23 - 4332856 against dd2faed on Graviton2 to 5, gcc 13.3 and clang 18.1: the same seconds (a measurement)

What the Macs above could not say: whether the branch costs a board with
no SME anything. Its core packs a product's columns only for a backend
that has `act_pack`, which on Linux is none, so the expectation was no
change. EC2 c6g, c7g, c8g and c9g, an xlarge each (4 cores), Ubuntu
24.04, both trees built on the instance under each compiler, v2,
tests/data/test.wav, rounds in rotated order.

- `make check-c` passes on dd2faed on all four under both compilers, and
  on c7g again with SVE at 128 bits. kcheck's shared calls run there on
  every backend the build holds: neon-sve, neon-i8mm, neon-dp-a76,
  neon-dp and neon on c7g, c8g and c9g, the last three on c6g.
- The transcript's bytes are one on the sixteen builds, and the M4 Pro's.
- The pick is 4332856's on each: neon-dp-a76 on c6g, neon-sve on the
  others.

Medians, seconds, 4332856 / dd2faed:

|                        |           | gcc 13.3        | clang 18.1      |
| ---------------------- | --------- | --------------- | --------------- |
| c6g, N1, neon-dp-a76   | 4 threads | 2.908 / 2.912   | 2.888 / 2.888   |
|                        | one core  | 11.487 / 11.522 | 11.482 / 11.463 |
| c6g, neon-dp by name   | 4 threads | 3.141 / 3.142   | 3.115 / 3.118   |
| c7g, V1, neon-sve, 256 | 4 threads | 1.944 / 1.944   | 1.927 / 1.925   |
|                        | one core  | 6.363 / 6.376   | 6.306 / 6.302   |
| c7g, neon-sve, 128     | 4 threads | 1.935 / 1.936   | 1.918 / 1.919   |
| c7g, neon-dp by name   | 4 threads | 2.380 / 2.380   | 2.362 / 2.361   |
| c8g, V2, neon-sve      | 4 threads | 1.621 / 1.623   | 1.610 / 1.610   |
|                        | one core  | 5.260 / 5.261   | 5.228 / 5.229   |
| c8g, neon-dp by name   | 4 threads | 1.840 / 1.841   | 1.824 / 1.824   |
| c9g, V3, neon-sve      | 4 threads | 1.069 / 1.068   | 1.057 / 1.056   |
|                        | one core  | 3.872 / 3.873   | 3.863 / 3.865   |
| c9g, neon-dp by name   | 4 threads | 1.261 / 1.261   | 1.246 / 1.247   |

(Five rounds at 4 threads, three on one core, `taskset -c 2 -t 1`, and
at 128 bits.) Every row is within 0.3%, the fast mode's sixteen within
0.2% (three rounds each). The widest is c6g's one core, +0.3% under gcc
and -0.2% under clang, each outside its three rounds' range and of
opposite sign: a build's layout, not the change.

The files: results/m4climb-<type>/ in the bench bucket (sysinfo, both
check-c logs, the rounds, and the instance's own log, which traces the
batch's script line by line).

## 2026-11-23 - neon-sme: the float products on FMOPA, at ref's bits with a tile that holds a NaN done again by FMLA (hypothesis)

pktito-mac's reading of the diarizer on the M4 Pro, ten minutes (test.wav
ten times over), single runs with the machine in use: neon-dp 2.297 s,
neon-sme at dd2faed 2.088, 9% where v2's speech gains 35. The attention's
float products (pk_k_gemm_f32: the scores, and the values' weighted sum)
did not move, 863 to 853 ms, and are 40% of the run. Here, make PROFILE=1
at dd2faed, the same file at 12 threads: 2.119 s, attention 856.9 ms, ff
and GELU 735.8, q k v 292.1, attention out 91.8, norm and quantize 81.9.

FMOPA (SME's, single precision, non-widening; hw.optional.arm.SME_F32F32
on the M4 Pro) adds the outer product of two vectors of 16 floats to a
tile: ZA[i][j] += zn[i] zm[j], 256 fused multiply-adds an instruction
where FMLA's is 4. ref's product is one fmaf chain an output in k order,
and a panel's 16 rows at one k are 16 floats in a row as core lays them
out, so they are zm as they stand; zn is 16 columns' x at that k, which
is a 16 by n transpose.

Its bits. The Arm ARM (D22.6, floating-point behaviors for instructions
that target the SME ZA array): such an instruction does a fused
multiply-add an element with no rounding of the product, honours FPCR's
FZ, RMode, AH and FIZ, and generates the default NaN as if DN were 1.
~/pktito-share/sme/fmopa_sem.c on the M4 Pro: 512,000 elements of chains
of 64 are fmaf's; a subnormal operand, result and addend, a sum that
cancels to a subnormal, the zeros' signs, an overflow, inf times 0 and
inf less inf are fmaf's; a NaN with a payload, in either operand or the
addend, comes out 7fc00000 where fmaf keeps the payload (and FPCR is 0
inside streaming mode as outside). A NaN stays one down a chain, so an
output is a NaN here exactly where it is one in ref, and every other
output is ref's bits: a block whose tiles hold a NaN is done again by
FMLA. The tiles are read four rows at a time and folded by FMAX on the
way to memory, which finds one.

~/pktito-share/sme/fmopa_rate.c, the attention's shapes, 16 columns a
call, against backend/ref's bits (400 shapes with subnormals, and with
NaNs and infinities put in: 89 differ with no test, none with it) and
neon-dp's FMLA tile. One thread, G multiply-adds a second, two runs, the
machine in use:

|                                  | FMLA     | FMOPA, x turned by NEON | x turned in ZA |
| -------------------------------- | -------- | ----------------------- | -------------- |
| diarizer, scores: 688 rows, n 64 | 56 to 64 | 470 to 604              | 604 to 650     |
| diarizer, values: 64 rows, n 684 | 58 to 63 | 300 to 306              | 467 to 600     |
| v2, scores: 384 rows, n 128      | 58 to 63 | 510 to 674              | 590 to 755     |
| v2, positions: 400 rows, n 128   | 63 to 65 | 504 to 615              | 678 to 702     |
| v2, values: 128 rows, n 382      | 58 to 63 | 391 to 400              | 537 to 722     |
| v2, values of a minute: n 750    | 61 to 63 | 434 to 461              | 696 to 709     |

Four tiles of a panel each take one k's FMOPAs, a group's four chains
apart, and x's 16 columns are turned in ZA itself before them (16 rows
loaded across a tile, 16 columns stored down it), which is 32
instructions for 256 floats where NEON's 4 by 4 transposes are 256: the
values' product, one group of four panels to a turn of 684, is half as
fast with NEON's. Threads at once (x turned by NEON, the system placing
them), scores and values: 12 on FMLA 434 and 409 G; 2 on FMOPA and 10 on
FMLA 1396 and 827; 4 and 8, 1593 and 1028.

The change: backend/neon's pk_k_gemm_f32, in the neon-sme build, takes a
call's columns in sixteens on FMOPA where the unit has a place
(sme_enter, the place SMOPA's threads take) and by FMLA where it has not,
and its last columns as SMOPA's are (sme_takes). The turned columns are
the thread's own, 64 KB for an n of 1024 (a chunk of 82 s is 1024
frames); a longer product is FMLA's. A bias is added and ReLU taken on
the way out of the tile, in SVE as ref writes them; a call whose bias is
its out is FMLA's, since a block done again would read what it wrote.

Expected: the diarizer's attention, 857 ms of 2119, to under 500 if the
products are two thirds of it (the softmax between them is the rest and
does not move), so the ten minutes to about 1.75 s; v2's attention a
quarter to a third faster where it is 128 ms of the minute whole at 8
threads, test.wav 0.255 s to about 0.24. kcheck's float cases take
columns of 16 and more with every tile count, a NaN's payload and sign,
and the shared calls that q8's have. A unit's threads for FMOPA to be
measured: 2 is SMOPA's.

## 2026-11-23 - neon-sme: the float products on FMOPA (result: kept; the diarizer's ten minutes 2.278 to 1.716 s on the M4 Pro, v2's 2.505 to 2.265)

As the hypothesis has it, with three things it did not: where a call's
last columns go, what the other builds compile to, and the 0.0f.

The code. backend/neon's k_gemm_f32 is as it was and is neon-sme's FMLA;
k_gemm_f32_sme, the kernel in that build alone, gives a call's columns to
sme_f32_cols in blocks of 16 from the first at which the unit has a place
and to k_gemm_f32 until then. A first form renamed the NEON kernel and
gave every build a kernel that called it: Apple clang's neon-dp came out
with another frame and other registers (1446 lines of its disassembly),
which is every other board's attention moved by no one's measurement. As
it stands the text of neon, neon-dp, neon-dp-a76 and neon-i8mm is
dd2faed's byte for byte under Apple clang 21 (otool -tv), and gcc's
no-schedule-insns region closes after neon-sme's kernel, with nothing
between for the others.

A tile's row goes to memory through ref's last line: the bias or 0.0f
added, then FMAXNM with zero for a ReLU. The 0.0f is not nothing: a chain
of products that underflow from below ends at -0, and ref's sum makes it
+0.

Bits. tests/kcheck has 348 float cases more (neon-sme 2525, the others
2244, which run them too): every count of columns a block can be left
with (2, 3, 15, 16, a seventeenth alone, 35) by every count of panels a
group can (1, 3, 4, 5, 11) by n of 1, 17, 64 and 130; a NaN with a sign and a payload in
x, in the matrix and in the bias; an infinity by a zero; chains that end
at -0; the bias as out, with a NaN and without; 16 columns past each out
that no call may write, the cases there were given them too; and 48
calls at once on a pool of 8 at the
attention's two shapes and one that leaves columns and panels over.
Nine mutations of the kernel, each in a copy:

| mutation                               | kcheck                                |
| -------------------------------------- | ------------------------------------- |
| a block with a NaN not done again      | fails, the NaN in x and in the matrix |
| a NaN's block counted as finished      | fails, the same                       |
| no 0.0f added where there is no bias   | fails, the chains that end at -0      |
| the turned columns one buffer for all  | fails, 30 of the calls on 8 threads   |
| a call whose bias is its out taken     | fails, the bias as out with a NaN     |
| FMAXNM left out                        | fails, the ReLU's cases               |
| a last block's 16 rows all written     | fails, the columns past out           |
| a last column alone given a block more | fails, the columns past out           |
| x's load not masked past n             | passes: it changes no value           |

The last reads up to 15 floats past a column's n and stores none of
them. ~/pktito-share/sme/guard.c ends x at a page no one may read: the
kernel as it stands reads nothing past it at five shapes, neon-dp's
neither, and the mutation faults. make check-c passes. pktito-c's output
is dd2faed's on five runs (md5): v2 and v3 on test.wav, v2 on the ten
minutes, `speakers` on the ten minutes, v2 with `--speakers`.

A call's last columns. ~/pktito-share/sme/fshort.c, ns a call by its
columns, medians, one thread: FMLA; FMOPA with the unit to itself; FMOPA
with one other thread at FMOPA on the same cluster (pk_os_cluster of
both, read around each sample):

|                 | FMLA: 1 | 2    | 4    | 16    | FMOPA alone: 1 | 4    | 16   | shared: 4 | 16   |
| --------------- | ------- | ---- | ---- | ----- | -------------- | ---- | ---- | --------- | ---- |
| 688 rows, n 64  | 1805    | 2184 | 3056 | 12158 | 915            | 972  | 1474 | 1750      | 2474 |
| 64 rows, n 684  | 1513    | 2669 | 3056 | 12211 | 1095           | 1139 | 1263 | 1847      | 2105 |
| 384 rows, n 128 | 2004    | 2724 | 3415 | 13647 | 1004           | 1046 | 1353 | 1800      | 2294 |
| 128 rows, n 375 | 1749    | 2869 | 3348 | 13353 | 1060           | 1106 | 1235 | 1803      | 2000 |
| 64 rows, n 96   | 135     | 323  | 425  | 1701  | 208            | 213  | 276  | 328       | 417  |
| 16 rows, n 48   | 27      | 38   | 50   | 210   | 146            | 149  | 152  |           | 249  |

A block costs FMOPA nearly the same whatever its columns, and a thread
that shares a unit gets 0.55 to 0.62 of it. One column on a shared unit
is a tie with FMLA's (1660 to 1790 ns in the few samples there were),
two are FMOPA's, 1750 to 1850 ns against 2184 to 2869, so PK_SME_FSHORT
is 2 and not sme_takes' rule, which is SMOPA's against SDOT. Entering
streaming mode and making ZA its own is about 150 ns a call, what the 16
by 48 row shows: the VAD's batched products, 64 rows by 128 and more,
are past it, and its one column a call as it streams stays FMLA's.

Threads at once with x turned in ZA, as the kernel turns it (the
hypothesis's were with NEON's), fmopa_rate.c, G multiply-adds a second,
the system placing the threads:

|                  | 1 at FMOPA | 2    | 4    | 2, and 10 at FMLA | 4 and 8 | 12 at FMLA |
| ---------------- | ---------- | ---- | ---- | ----------------- | ------- | ---------- |
| diarizer, scores | 622        | 1250 | 1503 | 1477              | 1664    | 469        |
| diarizer, values | 581        | 1160 | 1387 | 1448              | 1607    | 467        |
| v2, values       | 656        | 1335 | 1622 | 1455              | 1733    | 447        |

A unit's threads. An experiment build with the count in the environment,
interleaved medians of 3 (5 for the minute), 12 threads:

| threads of a unit at FMOPA | diarizer, ten minutes | v2, ten minutes | v2, the minute whole |
| -------------------------- | --------------------- | --------------- | -------------------- |
| 1                          | 1.792                 | 2.467           | 0.303                |
| 2                          | 1.701                 | 2.308           | 0.298                |
| 3                          | 1.689                 | 2.330           | 0.293                |
| 4                          | 1.689                 | 2.316           | 0.294                |
| none (FMLA)                | 2.207                 |                 |                      |

2 to 4 are inside one another's rounds, so it stays SMOPA's 2 and one
count for both products. The M4 Pro's clusters are of four; a cluster of
six (the M5 Ultra) is still to be read.

Timing: the M4 Pro, Apple clang 21, the machine in use, dd2faed's binary
and this one turn about (ab.py), medians and each side's least and most:

|                                   | rounds | dd2faed                | FMOPA                  |        |
| --------------------------------- | ------ | ---------------------- | ---------------------- | ------ |
| diarizer, ten minutes, 12 threads | 5      | 2.278 (2.163 to 2.313) | 1.716 (1.701 to 1.792) | -24.7% |
| diarizer, test.wav, 1 thread      | 3      | 0.946 (0.941 to 0.951) | 0.615 (0.605 to 0.632) | -35.0% |
| v2, test.wav, 12 threads          | 7      | 0.258 (0.254 to 0.265) | 0.241 (0.237 to 0.245) | -6.6%  |
| v2, ten minutes, 12 threads       | 5      | 2.505 (2.464 to 2.540) | 2.265 (2.236 to 2.320) | -9.6%  |
| v2, the minute whole, 12 threads  | 5      | 0.344 (0.341 to 0.351) | 0.292 (0.286 to 0.300) | -15.1% |
| v2, the minute whole, 8 threads   | 5      | 0.403 (0.402 to 0.404) | 0.319 (0.318 to 0.320) | -20.8% |
| v2, test.wav, 1 thread            | 3      | 1.261 (1.258 to 1.262) | 1.020 (1.006 to 1.034) | -19.1% |
| v3, test.wav, 12 threads          | 5      | 0.261 (0.256 to 0.266) | 0.246 (0.244 to 0.247) | -5.7%  |

make PROFILE=1, the diarizer's ten minutes: attention 856.9 to 555.6 ms
a run, not the under 500 expected. A copy with clocks in att_item, its
thread-seconds over a bench: on FMLA (PKTITO_KERNELS=neon-dp) the scores
20.2, the softmax 8.9, the values 19.7; here 8.0, 11.4 and 8.9. The
products are 2.4 times faster with two threads a cluster at the unit and
the rest on FMLA, and the softmax, which did not move, is now the largest
of the three. v2's minute whole at 8 threads: attention 127.8
to 46.2 ms of 405 and 321.

What the diarizer's run is now (PROFILE, 1.833 s): ff and GELU 765 ms,
attention 556, q k v 306, attention out 96, norm and quantize 87. Its
int8 products, the first, third and fourth of those, still pack their
columns a call (pk_k_gemm_q8), which costs one thread a third of SMOPA's
rate (2026-11-23, above); that is next.

neon-sme is built on macOS alone and Apple clang 21 is the one compiler
here: no second to bench under. The base M4 and the M5 Ultra follow.

## 2026-11-23 - neon-sme: the diarizer's int8 products from columns packed once (hypothesis)

core/diar.c gives pk_gemm_on no pk_packed, so on neon-sme each of a
product's items packs the columns it reads (pk_k_gemm_q8): a
feed-forward's up product, 2048 rows in items of 64, packs each sixteen
of its 684 columns 32 times, and one thread packing a call does 511 to
578 G multiply-adds a second where from packed columns it does 786 to
804 (2026-11-23, above). The recognizer's products have had theirs
packed once since that entry, 0.292 to 0.263 s on test.wav.

After c7960e4 the diarizer's ten minutes at 12 threads are (PROFILE,
1.833 s): ff and GELU 765 ms, attention 556, q k v 306, attention out
96, norm and quantize 87. The first, third and fourth are these
products, 1167 ms, 64% of the run.

The change: a pk_packed of the diarizer's own, the size of the largest
of its products' inputs (the feed-forward's hidden, a step's 688 frames
of 2048: 1.4 MB, and 0.18 MB of scales; nothing where the backend does
not pack), given to every pk_gemm_on and pk_gemm_panels_on in
core/diar.c. No kernel changes, and no bit: pk_k_gemm_q8_packed is held
to ref by kcheck from the same columns.

Expected: what the recognizer got of its int8 stages, a tenth to a
seventh of 1167 ms, so the ten minutes 1.716 to about 1.58 s. Less if
the threads at the unit are a smaller share here: two of a cluster's
are SMOPA's and the rest SDOT's, which packs nothing.

## 2026-11-23 - neon-sme: the diarizer's int8 products from columns packed once (result: kept; the ten minutes 1.708 to 1.659 s, one thread's minute 0.593 to 0.544)

core/diar.c as the hypothesis has it: a pk_packed carved with the
diarizer's other arrays, the size of the largest of its products' inputs
in sixteens of columns (688 by 2048 for Nemotron 3: 1.41 MB, 0.18 MB of
scales, 2.8 KB of states), 0 where the backend does not pack, and given
to its eleven products. Peak memory footprint of `speakers` on the ten
minutes, 58.97 to 59.64 MB.

M4 Pro, the machine in use, c7960e4's binary and this one turn about,
medians and each side's least and most:

|                                   | rounds | c7960e4                | packed once            |       |
| --------------------------------- | ------ | ---------------------- | ---------------------- | ----- |
| diarizer, ten minutes, 12 threads | 5      | 1.708 (1.694 to 1.780) | 1.659 (1.603 to 1.682) | -2.9% |
| diarizer, test.wav, 12 threads    | 5      | 0.162 (0.161 to 0.164) | 0.155 (0.146 to 0.158) | -4.3% |
| diarizer, test.wav, 1 thread      | 3      | 0.593 (0.593 to 0.594) | 0.544 (0.543 to 0.544) | -8.3% |
| v2, test.wav, 12 threads          | 3      | 0.243 (0.243 to 0.245) | 0.244 (0.243 to 0.244) |       |

make PROFILE=1, one build after the other, ms a run:

|               | ten minutes, 12 threads | test.wav, 1 thread |
| ------------- | ----------------------- | ------------------ |
| q k v         | 287.8 to 251.9          | 85.0 to 50.7       |
| attention out | 90.8 to 85.2            | 20.7 to 19.2       |
| ff and GELU   | 720.6 to 681.4          | 279.1 to 265.8     |

81 ms of 1099 at 12 threads, 7%, where a tenth to a seventh was
expected: two threads of a cluster are at the unit and the rest are
SDOT's, which never packed. One thread shows what the packing was: q, k
and v, three products from one input that each packed it, 40% less.

It shows the feed-forward too. At one thread q, k, v and the attention's
output, 1.05 M multiply-adds a frame, are 70 ms; the feed-forward's two
products are 2.1 M, so about 140 of its 266 ms by that rate, and the
other 125, a quarter of the run, would be the tile's GELU and the
hidden's quantizing, NEON's both. With the softmax, the largest part of
the attention's 176 ms, the float kernels that are not products are now
a third of the diarizer.

`speakers` on the ten minutes and on test.wav, at 12 threads and at 3,
`--speakers` with v2 and `probs` are c7960e4's (md5), and `probs` is
PKTITO_KERNELS=ref's. make check-c and make check-tsan pass.

## 2026-11-23 - the diarizer on every core, or the fast kind: the encoder's race at its second step (hypothesis)

The diarizer runs on every core (pk_diar_new, pk_os_cores) and has never
been timed against fewer. `./pktito-c bench -t N -m
models/nemotron-3-diarization.pkf` on the ten minutes, best of 5, two
rounds each; the GB10's are pktito-mac's (dd2faed, gcc 13.3, neon-sve,
three rounds, its server idle beside):

| board                                            | threads: seconds                                                                                          |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| M5 Ultra (12 and 24 cores of two kinds), ab7cc7f | 6: 1.795, 1.794; 12: 1.261, 1.260; 18: 1.418, 1.405; 24: 1.571, 1.559; 30: 1.694, 1.693; 36: 1.889, 1.888 |
| GB10 (10 X925, 10 A725), dd2faed                 | 10: 1.875, 1.885, 1.904; 20: 2.502, 2.559, 2.469                                                          |
| M4 Pro (8 and 4), in use, ab7cc7f                | 4: 2.576, 2.580; 6: 2.193, 2.191; 8: 1.760, 1.797; 10: 1.596, 1.593; 12: 1.554, 1.713                     |
| base M4 (4 and 6), c7960e4                       | 4: 3.251, 3.336; 6: 2.768, 2.796; 8: 2.633, 2.678; 10: 2.399, 2.422                                       |

The M5 Ultra's default, 36 threads, is half again its twelve fast cores'
time, and the GB10's twenty a third more than its ten: what the encoder
showed on the GB10 and its race settled (2026-10-27, H8). The two M4s are
fastest on every core. On the M5 Ultra (make PROFILE=1, ms a run at 12, 24
and 36 threads) the small passes grow with the threads and the large ones
do not shrink: norm + quantize 77.9, 234.5, 362.1; q k v 171.1, 286.3,
388.0; rotation 25.3, 73.2, 114.4; attention out 60.4, 103.9, 135.7;
attention 412.6, 361.1, 353.6; ff + GELU 470.9, 442.4, 462.5.

Where the time goes, on the M4 Pro: core/pool.c with counters in a
scratch copy (ab7cc7f; `speakers` of the ten minutes, one run each), and
the polls before a worker parks, PK_SPIN, read from the environment:

|                                             | 12 threads, 20000 polls (the tree) | 12, 2,000,000 | 8, 20000       | 8, 2,000,000 |
| ------------------------------------------- | ---------------------------------- | ------------- | -------------- | ------------ |
| the 8901 passes, wall, ms                   | 1715                               | 3117          | 1872           | 1713         |
| their items' time, summed over threads, s   | 14.39                              | 16.38         | 11.76          | 11.33        |
| the caller waiting after its last item, ms  | 288                                | 1740          | 383            | 283          |
| threads that took an item, a pass           | 10.0                               | 11.3          | 7.9            | 8.0          |
| workers in a pass that came from parked     | 66937 of 97911                     | 3808          | 29729 of 62307 | 187          |
| a pass's start to a worker's look, mean, us | 29.1                               | 16.6          | 8.0            | 0.5          |

One thread's whole run is 5.4 s, and twelve threads' items take 14.4 s:
the threads past a cluster's two at the SME unit run SDOT and FMLA, and
four of them are efficiency cores. A pass waits for its slowest item.
With every worker awake at a pass's start (2,000,000 polls, 0.6 ms) the
efficiency cores take a twelfth of the items each and the caller waits
1.74 s of the 3.1 for them; the tree's 20000 polls are 6 us here
(core/pool.c, PK_SPIN), two workers in three start a pass parked, and the
slow to wake take fewer items, which is how twelve threads come out level
with eight. `bench` at 12 threads and 20000, 200,000, 2,000,000 and
20,000,000 polls: 1.712 and 1.674 s, 2.050 and 1.956, 2.642 and 2.617,
2.952 and 3.003. Polling longer is rejected for a pool of two kinds; on
the fast kind alone it gave 8% (1872 to 1713 ms), a lead for a pool that
knows which it is on.

Hypothesis: the diarizer takes the encoder's race, by H8's rule, as the
encoder runs it: with no thread count asked for and cores of two kinds
(pk_os_fast_cores under pk_os_cores), the second step of 100 frames or
more runs its layers in turns of two, on every core and on the fast kind
alone, 28 of Nemotron 3's 31 layers, and the diarizer keeps whichever's
median layer took less. The first step is the first pass over the
weights' pages, as the encoder's first recording is. The race moves to
core/pool.c as pk_pool_race, which core/encoder.c's layers and
core/diar.c's both call, so that the rule is written once; the encoder's
turns, medians and counts are as they were. A step's bits are the same on
any threads (core/diar.c), so nothing but the time can change.

Predict, the ten minutes at the default threads: the M5 Ultra 1.889 to
about 1.28 s (its twelve, less the half step raced on 36); the GB10 2.6
to about 1.9; the M4 Pro and the base M4 level with the better of their
two counts in every round, the M4 Pro's race reading either way (8 and 12
are 5 to 10% apart, and the machine is in use). The recognizer's race
keeps what it kept: 12 on the M4 Pro.

## 2026-11-24 - the diarizer's race in the encoder's turns of two layers (result: rejected, on the GB10)

Built as the hypothesis has it, in the working tree and not committed:
pk_pool_race in core/pool.c, called by core/encoder.c's layers and by
core/diar.c's on the second step. `./pktito-c bench` of the diarizer over
the ten minutes, best of 5 in seconds and the threads each default run
kept; every board's `speakers` and `probs` were ab7cc7f's (md5), and the
recognizer kept what it kept before (12 on the M4 Pro, 10 on the base M4,
36 on the M5 Ultra, 20 on the GB10).

| board                  | default: threads kept  | default, s                               | the fast kind, -t  | every core, -t     |
| ---------------------- | ---------------------- | ---------------------------------------- | ------------------ | ------------------ |
| M5 Ultra, six rounds   | 12 in all six          | 1.249 to 1.256                           | 12: 1.258 to 1.265 | 36: 1.884 to 1.898 |
| base M4, nine runs     | 10 in all nine         | 2.151 to 2.164                           | 4: 2.803 to 2.826  | 10: 2.152 to 2.156 |
| M4 Pro, in use, twelve | 12 in eleven, 8 in one | 1.567 to 1.725                           | 8: 1.697 to 1.888  | 12: 1.538 to 1.726 |
| GB10 (gcc 13.3), six   | 10, 20, 10, 10, 20, 10 | 1.885, 2.441, 1.915, 1.919, 2.518, 1.926 | 10: 1.906 to 1.943 | 20: 2.469 to 2.565 |

The GB10 kept its twenty in two races of six, and with each raced layer's
time logged in four more of eight. Those eight races' medians, every
core's layers and the fast kind's, ms: 3.850 and 3.661, 3.804 and 3.904,
4.104 and 3.822, 3.771 and 3.777, 3.914 and 3.495, 3.691 and 3.724, 3.716
and 3.384, 3.927 and 4.055. The fast kind's are 0.89 to 1.03 of every
core's, where its whole run is 0.77 (1.92 s of 2.50). The encoder's turns
read 0.886 to 0.940 on this board for a run at 0.76 (2026-10-27, H6's
races in H7's entry), and H8 kept the ten in 35 of 36; the diarizer's
read too near 1 to decide on.

Why, on the GB10: whole steps by turns on 20 threads and on 10
(pk_pool_set_active before a step's layers), each layer's time in ms, the
first twelve layers and the last three, three runs alike:

| step | threads | layers 1 to 12                                              | 29 to 31       | the 31, ms |
| ---- | ------- | ----------------------------------------------------------- | -------------- | ---------- |
| 2    | 20      | 8.66 3.88 3.66 3.79 3.30 3.73 3.48 3.47 2.76 2.95 3.24 3.00 | 3.02 3.02 3.37 | 105.3      |
| 3    | 10      | 4.04 3.73 3.41 3.85 4.02 3.83 4.39 3.74 3.88 3.12 2.92 3.17 | 2.63 2.63 2.60 | 93.8       |
| 4    | 20      | 3.93 3.70 4.28 4.26 3.54 2.81 3.08 3.60 3.14 3.00 2.97 3.51 | 3.17 2.91 3.00 | 105.2      |
| 5    | 10      | 3.00 2.78 2.63 2.60 2.60 2.72 2.73 2.58 2.62 2.62 2.62 2.61 | 2.62 2.61 2.57 | 82.7       |
| 6    | 20      | 4.85 4.54 3.96 4.42 4.00 3.40 3.93 3.98 4.24 4.29 3.78 3.09 | 4.58 3.72 3.14 | 112.7      |
| 7    | 10      | 2.96 2.78 2.63 2.57 2.62 2.58 2.56 2.62 2.62 2.56 2.59 2.58 | 2.62 2.61 2.60 | 81.1       |

When ten of the twenty threads go to sleep the other ten stay where they
were, some on A725s, and Linux takes nine to twelve layers, about 40 ms,
to have them on the X925s the first time and one to eight after; a turn of
two layers, 7 ms, is over before that. The step's 31 layers on ten threads
took less than on twenty in every pair of the three runs, the first
included (93.8 to 100.8 ms against 100.6 to 112.7; 81.1 to 88.4 after).
On the Macs a turn of two was enough.

The pool's counters on the M5 Ultra (ab7cc7f, `speakers` of the ten
minutes), as the M4 Pro's above:

|                                             | 12 threads     | 36 threads       |
| ------------------------------------------- | -------------- | ---------------- |
| the 8901 passes, wall, ms                   | 1240           | 1890             |
| their items' time, summed over threads, s   | 12.21          | 15.13            |
| the caller waiting after its last item, ms  | 119            | 804              |
| threads that took an item, a pass           | 11.0           | 15.1             |
| workers in a pass that came from parked     | 37169 of 97911 | 277769 of 311535 |
| a pass's start to a worker's look, mean, us | 10.5           | 96.0             |

## 2026-11-24 - the diarizer's race in turns of a step (result: kept; the M5 Ultra's ten minutes 1.812 to 1.254 s, the GB10's 2.497 to 1.929)

The rule is H8's with a step for a turn, written in core/diar.c's layers;
core/pool.c and core/encoder.c are as they were. With no thread count
asked for and cores of two kinds, the first four full steps (tmax frames:
the same shapes) after the diarizer's first step run their layers on
every core, the fast kind, every core, the fast kind, each layer timed,
and the diarizer keeps whichever's 62 layers have the lesser median.
A recording that ends before the four leaves the rest to the next one the
diarizer hears; pk_diar_threads says the count kept. Expected before the
runs: the GB10 keeps its ten every time, the M5 Ultra its twelve, the M4s
every core; a first run pays two steps on the slower side.

`./pktito-c bench` of the diarizer over the ten minutes, best of 5 in
seconds, ab7cc7f's binary beside it at the default threads:

| board                          | default: threads kept | default, s     | ab7cc7f, default, s | the fast kind, -t  | every core, -t     |
| ------------------------------ | --------------------- | -------------- | ------------------- | ------------------ | ------------------ |
| M5 Ultra, six rounds           | 12 in all six         | 1.249 to 1.257 | 1.807 to 1.815      | 12: 1.199 to 1.268 | 36: 1.887 to 1.896 |
| GB10 (gcc 13.3), twelve rounds | 10 in all twelve      | 1.916 to 1.990 | 2.398 to 2.559      | 10: 1.920 to 1.931 | 20: 2.446 to 2.584 |
| base M4, six rounds            | 10 in all six         | 2.150 to 2.171 | 2.164 to 2.170      | 4: 2.806 to 2.816  | 10: 2.160 to 2.176 |
| M4 Pro, in use, six rounds     | 12 in all six         | 1.621 to 1.693 |                     | 8: 1.696 to 1.870  | 12: 1.534 to 1.691 |

Medians: the M5 Ultra 1.812 to 1.254 s (-30.8%, five rounds turn about),
the GB10 2.497 to 1.929 (-22.7%, twelve), the base M4 2.167 and 2.167.
The M5 Ultra's test.wav, best of 5, 0.211 to 0.120 s. One process over
the ten minutes, `time pktito-c speakers`, four runs each: the M5 Ultra
1.86 to 1.38 s, the GB10 2.52 to 2.57 down to 2.04 to 2.08, the base M4
2.20 to 2.21 up to 2.26 to 2.28, its two steps on four cores. One process
over test.wav, which has one full step and so finishes no race: the M5
Ultra 0.24 s before and after.

The recognizer is as it was: an M5 Ultra's v2 keeps 36 threads, test.wav
0.172 to 0.174 s before and 0.171 to 0.174 after, the ten minutes 0.916
and 0.914.
`speakers` and `probs` are ab7cc7f's on the four boards (md5), and make
check-c passes on each (Apple clang 21; the GB10's gcc 13.3). make
check-tsan now runs the tiny diarizer on the default threads as well as
on six, where this Mac's eight and four take the turns, and passes.

Not measured: a phone. Leads left where they are. Polling longer before a
worker parks gave 8% on the M4 Pro's fast kind alone and cost 55% on every
core (2026-11-23, above). A second thread a core on x86, from
pktito-mac's rounds of the ten minutes: an i9-12900HK took 4.31 to 4.33 s
on its six P-cores, 4.29 to 4.30 on its 14 cores, the default, and 3.96 to
4.02 on 20 threads, 7% less; a 9950X under WSL2 1.090 to 1.094 on 8, 1.002
to 1.011 on its 16, and 2.45 to 2.70 on 32. Neither has a slow kind that
costs it, and a third side for the i9's 7% would be a race of its own.

## 2026-11-24 - c7960e4 and ab7cc7f on the base M4, the M5 Ultra and the other boards (a measurement)

The two commits before the race, each against the one before it, binaries
turn about, medians in seconds; the ten minutes is test.wav ten times
over. Every board's transcripts, `speakers` and `probs` are the commit
before's (md5), and make check-c passes on each: neon-sme's 2525 kcheck
cases on the three Macs (Apple clang 21), and c7960e4's 2244 a backend on
an i9-12900HK (gcc 15.2: avx2-vnni, avx2), a Pi 5 (gcc 14.2: neon-dp-a76,
neon-dp, neon) and a GB10 (gcc 13.3 and clang 18.1: neon-sve, neon-i8mm,
neon-dp-a76, neon-dp, neon).

FMOPA, dd2faed to c7960e4:

|                                 | base M4 (10 threads) | M5 Ultra, 36 threads | M5 Ultra, -t 12 | M5 Ultra, -t 24 |
| ------------------------------- | -------------------- | -------------------- | --------------- | --------------- |
| diarizer, the ten minutes       | 2.940 to 2.425       | 1.940 to 1.875       | 1.608 to 1.435  | 1.747 to 1.645  |
| v2, test.wav                    | 0.377 to 0.356       | 0.175 to 0.174       |                 |                 |
| v2, the ten minutes             | 3.770 to 3.476       | 0.934 to 0.914       | 1.993 to 1.771  | 1.196 to 1.212  |
| v2, the minute whole            | 0.486 to 0.410       | 0.205 to 0.196       |                 |                 |
| v2, the minute whole, 4 threads | 0.527 to 0.477       |                      |                 |                 |
| v2, the minute whole, 1 thread  | 1.262 to 1.016       |                      |                 |                 |

The M5 Ultra's one row the wrong way, v2's ten minutes at 24 threads, is
inside its rounds: 1.176 to 1.223 before, 1.136 to 1.217 after.

A unit's threads at FMOPA where a cluster is six cores (the M5 Ultra, 36
threads; c7960e4 with the count read from the environment, the SMOPA
count left at 2):

| threads a unit | diarizer, the ten minutes | v2, the ten minutes | v2, the minute whole |
| -------------- | ------------------------- | ------------------- | -------------------- |
| 2 (the tree)   | 1.870                     | 0.915               | 0.195                |
| 1              | 1.898                     | 0.928               | 0.198                |
| 3              | 1.865                     | 0.908               | 0.196                |
| 4              | 1.868                     | 0.912               | 0.196                |
| 6              | 1.895                     | 0.914               | 0.199                |
| none (FMLA)    | 1.942                     | 0.935               | 0.207                |

Two, three and four are within 0.8% of each other on every row, as on the
M4 Pro's clusters of four: PK_SME_THREADS stays 2 for both products.

The diarizer's columns packed once, c7960e4 to ab7cc7f, the diarizer
alone:

|                          | base M4        | M5 Ultra       |
| ------------------------ | -------------- | -------------- |
| the ten minutes, default | 2.436 to 2.168 | 1.871 to 1.818 |
| the ten minutes, -t 4    | 3.346 to 2.810 |                |
| the ten minutes, -t 12   |                | 1.437 to 1.260 |
| the ten minutes, -t 24   |                | 1.646 to 1.577 |
| test.wav, 1 thread       | 0.631 to 0.555 | 0.592 to 0.523 |

11 to 16% on the base M4 and 12% on the M5 Ultra's twelve, against the M4
Pro's 2.9% at twelve threads and 8.3% at one. Why the M4 Pro's twelve gain
the least was not measured.

Rejected by a probe, with no kernel written: the GELU and the softmax's
exp in streaming-mode SVE, sixteen lanes on the unit. A degree-7 Horner
polynomial with exp's rounding, conversion and scaling over 16384 floats
(svefma.c, a scratch program, not in the tree; M4 Pro, best of 200): NEON
at four vectors a step 0.18 to 0.24 ns a value, streaming SVE 0.67 at
one, two and four vectors a step. FMLA alone in eight chains: 33.8 G
multiply-adds a second in NEON, 15.4 in streaming SVE. The unit pays for
the outer products and nothing else.

## 2026-11-24 - the diarizer's scores in int8, the mean key off (hypothesis)

Of the diarizer's products the attention's two are the float ones, and
the attention is 30.1% of the ten minutes on the M4 Pro's twelve threads
(make PROFILE=1 at 299d03a: 517 ms of 1.72 s) and 36.1% of test.wav on
neon-dp at one (518 ms of 1.43 s). docs/DIARIZATION.md, "The attention in
int8" (2026-11-23, tools/diarquant.py's qk8), measured the first of the
two, Q Kᵀ, rounded as an int8 product would round it, and it passed: each
head's mean key over the step's frames taken off its keys, which moves a
query's scores by one amount that its softmax undoes; the queries as
activations in blocks of 32 of a head's 64 dimensions; the keys as
weights, a scale a key. In C that is pk_k_gemm_q8 with a head's keys as a
PK_Q8 matrix of a step's frames by 64, built each layer.

A probe first (qkprobe.c, a scratch program, not in the tree): a layer's
scores at a full step, 684 frames and 8 heads, one thread of the M4 Pro,
best of 30, in ms. Float is the keys into pk_k_gemm_f32's panels and the
product; int8 is the queries quantized, the keys' mean taken off, the
keys quantized and laid in PK_Q8's panels, and pk_k_gemm_q8. The softmax
that follows either is beside them.

|            | float: panels, product | int8: queries, keys, product | softmax |
| ---------- | ---------------------- | ---------------------------- | ------- |
| `neon-sme` | 0.059, 0.499           | 0.049, 0.090, 0.477          | 1.79    |
| `neon-dp`  | 0.061, 4.043           | 0.052, 0.101, 1.397          | 1.84    |
| `neon`     | 0.063, 4.080           | 0.052, 0.100, 3.738          | 1.79    |

The product is 64 inputs wide, two blocks, so each output's two
conversions and fused multiply-adds weigh as much as its sums: SDOT is
2.9 times FMLA here where the int8 product is four times the float one a
multiply-add, and on the unit SMOPA's sixteen outer products with their
two rounds of ZA's rows are level with FMOPA's 64.

Expected: on a board whose float product is FMLA, 2.55 ms less a layer,
a tenth of the diarizer at one thread (neon-dp's test.wav is 1.43 s); on
an SME Mac nothing on a unit's two threads (0.06 ms more a layer of 7.9)
and FMLA's gain on the cluster's others, so a few percent at the default
threads; x86 is not probed, and VNNI against FMA should stand as SDOT
against FMLA. Kept only if the diarizer is no slower on an SME Mac, a
NEON board without SME and an x86 one, and faster on one of them: this
changes ref's bits, every board's, and its rows (`pktito-c probs`,
diarquant.py's rows:DIR) are then measured against float32 before it
merges. P·V stays float: the same section finds it the less safe half,
and the two together a step past the rule.

## 2026-11-24 - the diarizer's scores in int8 (result: kept, 3 to 8% at the default threads; an SME unit's one thread 1 to 2% slower)

core/diar.c: the rotation's items quantize the queries they have just
turned, a head's 64 dimensions in two blocks; keys_item, a head an item,
takes the head's mean key off its keys (a float32 chain a dimension in
frame order, divided by the frames), quantizes each key as one block and
lays its bytes in PK_Q8's panels; att_item's scores are pk_k_gemm_q8. K's
float panels, 1.41 MB at a full step, give way to 0.77 MB of int8 operands
and their scales. pk_k_gemm_q8 takes its inputs in 32s, so a diarizer's
heads are whole blocks now (core/model.c asks dk % 32 where it asked 16;
Nemotron 3's are 64, and the tiny diarizer is regenerated with two heads
of 32 where it had four of 16).

The bits: `probs` of conv2.wav is one md5 from ref and from every backend
run, at its default threads and the M4 Pro's at three - neon-sme, neon-dp and neon on the M4 Pro, neon-sme
on a base M4 and an M5 Ultra, neon-sve on a GB10 under gcc and clang,
neon-dp-a76 on a Pi 5, avx2-vnni on an i9-12900HK - and another than the
float scores' on each. conv2's stretches of speech move at one boundary,
48.820 to 48.830 s (tests/data/conv2.speakers.txt, rewritten by ref), and
its words keep their speakers (conv2.v2.json and the three beside it are
unchanged). tests/kcheck has the scores' shapes, 688 rows by 64 inputs
for 16 columns and for 12: 2250 cases a backend, neon-sme's 2533. make
check-c passes on the seven boards and compilers of the table,
check-sanitize and check-tsan on the M4 Pro.

`pktito-c bench` of the diarizer, float scores (299d03a) to int8, the two
binaries turn about, medians in seconds of 3 to 6 rounds (the Pi 5's ten
minutes one round):

|                        | backend, threads   | test.wav               | test.wav, 1 thread     | the ten minutes        |
| ---------------------- | ------------------ | ---------------------- | ---------------------- | ---------------------- |
| Pi 5, gcc 14.2         | neon-dp-a76, 4     | 1.504 to 1.400 (-6.9%) | 5.554 to 5.163 (-7.0%) | 17.18 to 15.78 (-8.1%) |
| GB10, gcc 13.3         | neon-sve, 10 of 20 | 0.168 to 0.160 (-4.8%) | 1.147 to 1.059 (-7.7%) | 1.911 to 1.795 (-6.1%) |
| GB10, clang 18.1       | neon-sve, 10 of 20 | 0.168 to 0.159 (-5.4%) | 1.110 to 1.034 (-6.8%) | 1.908 to 1.786 (-6.4%) |
| i9-12900HK, gcc 15.2   | avx2-vnni, 14      | 0.388 to 0.370 (-4.6%) | 1.272 to 1.172 (-7.9%) | 4.283 to 4.045 (-5.5%) |
| base M4                | neon-sme, 10       | 0.196 to 0.187 (-4.6%) | 0.555 to 0.561 (+1.1%) | 2.163 to 2.054 (-5.0%) |
| M5 Ultra               | neon-sme, 12 of 36 | 0.121 to 0.116 (-4.1%) | 0.522 to 0.528 (+1.1%) | 1.254 to 1.212 (-3.3%) |
| M4 Pro, in use         | neon-sme, 12       |                        | 0.540 to 0.550 (+1.9%) | 1.645 to 1.611 (-2.1%) |
| M4 Pro, PKTITO_KERNELS | neon-dp, 12        |                        | 1.433 to 1.306 (-8.9%) | 2.531 to 2.287 (-9.6%) |

The M5 Ultra at -t 12 is 1.265 to 1.226 and at -t 36 1.894 to 1.871. The
M4 Pro's twelve are level, not faster: its rounds ran 1.574 to 1.800 s
before and 1.559 to 1.623 after.

Against the hypothesis: the FMLA and FMA boards gain 7 to 9% at one
thread where a tenth was expected, and 5 to 8% at their default threads.
On SME the one thread loses 1.1 to 1.9% where 0.4% was expected, cleanly
(the base M4's rounds 0.555 to 0.555 against 0.561 to 0.562): the scores
are level on the unit and their operands' rounding is more work than K's
panels were. The Macs gain at the default threads because a unit is two
threads' at a time (PK_SME_THREADS) and a cluster's others had the float
product on FMLA: the base M4's ten 5.0%, the M5 Ultra's twelve 3.3%, its
36 only 1.2%. A machine whose every thread has a place at a unit, two a
cluster or fewer, would lose the one thread's 1 to 2%; none of these is
one at its default.

Kept on that: a gain on every board without SME, and nothing slower at
the default threads. Not measured: the C's rows against float32 (the
study's are diarquant.py's rounding), a phone, the 9950X's avx512-vnni,
and a Pi 3B+, whose `neon` the probe puts at 3.74 ms a layer against
4.08.

Leads, neither taken: the queries with one scale a head, a block of 64,
would halve SMOPA's drains and every backend's conversions, and is a
question for diarquant.py first; and the softmax is now the attention's
largest part on a unit, 1.8 ms a layer on one thread beside the scores'
0.5.

## 2026-11-24 - the queries with one scale a head: the probe (a measurement; the lead is small)

The entry above left a lead it had not timed: a head's queries as one
block of 64, a conversion and a fused multiply-add an output where two
blocks have two. The probe again at 3276708 with the queries' block a
macro (qkprobe64.c, a scratch program, not in the tree), a layer's scores
on one thread of the M4 Pro, best of 30, in ms:

|            | product, blocks of 32 | product, one block of 64 | float product, with K's panels |
| ---------- | --------------------- | ------------------------ | ------------------------------ |
| `neon-sme` | 0.474                 | 0.444                    | 0.498 + 0.058                  |
| `neon-dp`  | 1.368                 | 1.289                    | 3.985 + 0.059                  |
| `neon`     | 3.750                 | 3.568                    | 4.047 + 0.061                  |

5 to 6% of the product, with the operands' rounding the same 0.14 ms
beside it. On the unit the int8 scores would be 0.58 ms against float's
0.56 where they are 0.61 now: the one thread's loss halved, not gone. On
SDOT it is 0.08 ms of a layer's 21 or so, 0.4% of the diarizer. A rounding
of its own to score against float32 for that is not worth its run, and
the one above stays as measured.

## 2026-11-24 - the int8 scores: the C's rows against float32, and the 9950X (a measurement)

What the result above left unmeasured, from the session on the 9950X
(gcc 15.2, avx512-vnni, 16 threads). `pktito-c probs` of 299d03a and of
3276708 over docs/DIARIZATION.md's 25 recordings, scored by diarquant.py
(`--configs rows:DIR`) against the same float32 runs as its tables:

|                                       | AMI, float scores | AMI, int8 scores | synthetic, float scores | synthetic, int8 scores |
| ------------------------------------- | ----------------- | ---------------- | ----------------------- | ---------------------- |
| labels changed                        | 0.086%            | 0.092%           | 0.053%                  | 0.039%                 |
| DER against float32's                 | 0.91%             | 0.94%            | 0.47%                   | 0.37%                  |
| DER against the truth, from float32's | +0.065 (t 1.6)    | +0.061 (t 1.6)   | -0.071 (t -1.0)         | -0.021 (t -0.8)        |
| words to another speaker              | 264               | 299              | 17                      | 0                      |
| words not the true speaker's          | 837               | 848              | 112                     | 129                    |

float32's own words not the true speaker's are 855 and 129. On AMI the C
is where it was, and on the synthetic set conv5's 17 moved words and its
speaker fewer are gone. The whole tables, with dBCE and the speakers'
changes, are in docs/DIARIZATION.md, "The attention in int8".

The 9950X's time over the ten minutes at its default 16 threads, five
rounds turn about: 0.989 to 0.999 s with float scores, 0.947 to 0.962
with int8, 4.3% less, the int8 the faster in every pair.

Still not measured: a phone and a Pi 3B+.

## 2026-11-24 - the int8 scores' rows: the entry above scored the held rows (a correction, with the model's own)

The rows scored in the entry above were `pktito-c probs`'s as it prints
them now, which since 19ad40b (2026-11-22) are pk_diar_run_held's: the
short speakers' hold, processing on top of the model
(docs/SHORT_SPEAKERS.md). docs/DIARIZATION.md's "The C, measured the same
way" of 2026-11-09 scored the model's own rows, and the synthetic set's
line with float scores did not match it: 0.053% of labels changed above,
0.040% there. The session on the 9950X bisected that on conv8 to 19ad40b,
and with `--no-hold` 299d03a's rows are 2026-11-09's byte for byte on all
25 recordings. The model's own rows, `probs --no-hold`, scored the same
way:

|                                       | AMI, float scores | AMI, int8 scores | synthetic, float scores | synthetic, int8 scores |
| ------------------------------------- | ----------------- | ---------------- | ----------------------- | ---------------------- |
| labels changed                        | 0.086%            | 0.092%           | 0.040%                  | 0.025%                 |
| DER against float32's                 | 0.91%             | 0.93%            | 0.35%                   | 0.23%                  |
| DER against the truth, from float32's | +0.064 (t 1.6)    | +0.061 (t 1.6)   | -0.091 (t -1.4)         | -0.006 (t -0.3)        |
| words to another speaker              | 264               | 299              | 17                      | 0                      |
| words not the true speaker's          | 837               | 848              | 112                     | 129                    |

The reading is the entry above's: on AMI the C is where it was, on the
synthetic set nearer float32. docs/DIARIZATION.md, "The attention in
int8", has both sets of rows, each named.

On the M4 Pro (neon-sme), 7fbd322 (the commit 2026-11-09 scored), 4ba872e
(the one before the hold) and 299d03a print one md5 for conv2.wav, for it
with 30 s of a third voice after it, and for it at 44.1 kHz, with
`--no-hold` and without: the model's own rows did not move in between,
and none of the three is a recording the hold changes. Rows scored for a
rounding are `probs --no-hold`'s.
