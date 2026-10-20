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
