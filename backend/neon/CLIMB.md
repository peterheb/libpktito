# backend/neon: the climb

Dated records of what was measured on which board, append-only: a later
section may supersede an earlier one, but the earlier one stays, rejected
hypotheses included.

# 2026-10-20 - the A53 int8 product (Pi 3B+)

## The board and the build

| | |
|---|---|
| board | Raspberry Pi 3B+, 4 x Cortex-A53 at 1.4 GHz (cpufreq `ondemand`, 1400 MHz under load), 905 MB |
| caches | 32 KB L1D a core, 512 KB L2 shared |
| compiler | gcc 14.2.0 (Debian 14.2.0-19) |
| flags | `-O3 -march=armv8-a -mtune=cortex-a53 -std=c11 -ffp-contract=off` (the Makefile's, backend neon) |
| model | parakeet-tdt-0.6b-v2.pkf |
| recording | Silero's test.wav, 60 s, which `--max-chunk 30` cuts into 3 chunks |
| tools | `./pkito bench` (best of 5 transcriptions), `make PROFILE=1` for the stage split, `tests/gemmbench` (the int8 product alone on the encoder's four shapes, T = 250, the engine's 64-row slices) |

The board has hit its soft temperature limit before (`vcgencmd get_throttled`
0x80000) and a bench runs four cores flat out for five minutes, so every run
here records the temperature and the ARM clock around it.

## What the A53 issues

pinfer's measurement of this same board (`../pinfer/backend/neon/CLIMB.md`,
"What the A53 actually issues", tests/isabench): a 128-bit vector op of any
kind (SMULL, SMLAL2, ADDP, FMLA, SCVTF, DUP) issues once a cycle; SADALP
once every two; a 128-bit load once every two (L1D hands over 8 bytes a
cycle), `ldp q, q` once every four; a load and a vector op do not dual-issue;
post-increment addressing costs a cycle a load. Those prices are used below
and re-measured where a decision rests on one.

## The baseline (main at b07c50d)

The int8 product (`q8_tile`, ARMv8.0 path): a tile of 8 rows by 4 columns,
a pair of groups (8 inputs) at a time as SMULL / SMLAL into 16 bits and
SADALP into 32, each row's sum in two lanes until the block's end.

| | 1 thread | 4 threads |
|---|---|---|
| gemmbench, all four shapes | 1075.9 ms, 1.91 MAC/cycle | 309.7 ms, 1.66 MAC/cycle/core |
| ff up 4096x1024 | 1.97 | 1.86 |
| ff down 1024x4096 | 1.82 | **1.41** |
| q k v o 1024x1024 | 1.98 | 1.86 |
| conv pw1 2048x1024 | 1.97 | 1.86 |

`./pkito bench` (PROFILE=1 build, main's commit message): 58.4 s, 1.0x real
time; feed-forward 57%, q k v pos 15%, conv pw1 pw2 9%, attention 5%,
subsampling 5%, decoder 2%.

The disassembly of the 4-column tile's loop (objdump, gcc 14.2): per 256
MACs (8 inputs x 8 rows x 4 columns), 16 SMULL, 16 SMLAL, 16 SADALP, 8 DUP
(the activation's four bytes, loaded as `ldr s`, then replicated), 4
`mov d, v.d[1]` (the high half of a weight vector, extracted so that a
64-bit SMULL can take it), 4 weight loads (two `ldr q`, one `ldp q`), 8
activation loads, and **4 spill loads and 4 spill stores of accumulators**:
the tile wants 16 int32x4 accumulators, 8 float ones, 4 weight vectors and
the temporaries, more than 32 registers. At the prices above that is about
32 + 32 (SADALP) + 8 + 4 + 8 + 8 + 16 (spills) = 108 cycles, 2.4 MAC/cycle,
against 1.97 measured.

## H1 - no spills: fewer columns a tile, and SMULL2 on replicated activations

Stated before the edit. Two changes to the ARMv8.0 tile, measured together
and then apart if they land:

- **A tile of 8 rows by 2 or 3 columns**, not 4: 8 or 12 int accumulators,
  4 or 6 float, 4 weight vectors, 4 product temporaries, 2 activation
  vectors - inside 32 registers, so no spills.
- **The activation's four bytes replicated across 16** (`vld1q_dup_s32`,
  LD1R) and the high halves of the weight vectors taken by SMULL2/SMLAL2
  (`vmull_high_s8`), which removes the four `mov d, v.d[1]` and the DUPs.

Priced: per 8 inputs, 8 rows and 2 columns (128 MACs), 4 weight loads (8
cycles), 4 activation loads (4), 2 x (4 multiplies + 4 SMLAL... ) = 2 x (8 +
4 SADALP at 2) = 32: **44 cycles, 2.9 MAC/cycle**; at 3 columns, 62 cycles
for 192 MACs, 3.1. Predicted: gemmbench single-thread +40 to +55%, the bench
-25 to -35% (the int8 product is about 80% of it).

### H1, measured

`./pkito bench` baseline, the release build, two runs: **58.681 s and
58.496 s**, a spread of 0.3%. The board went from 52 to 69 C over the two
and read 1200 MHz at the end: four cores flat out for five minutes reach
the 60 C soft limit, where the firmware takes the A53s from 1.4 to 1.2 GHz.
From here every run starts below 55 C (`pkito-a53-cool.sh` on the board),
and gemmbench, a second or so a shape, mostly runs before the limit.

gemmbench, GMAC/s (the MAC/cycle column read the idle clock at the start of
these runs, so it is left out; GMAC/s is the measure):

| | 1 thread | 4 threads |
|---|---|---|
| baseline, 4 columns, spilling | 2.68 | 9.31 |
| 2 columns, LD1R-less replication, SMULL2 | 2.78 (+3.7%) | 10.00 (+7.4%) |
| **3 columns**, the same | **3.03 (+13.1%)** | **10.94 (+17.5%)** |

Far short of the +40-55% priced. The 3-column loop (objdump) has no spill
left: per 192 MACs, 12 SMULL/SMULL2, 12 SMLAL/SMLAL2, 12 SADALP, 6 `ldr s`
+ 6 `dup v.4s, v.s[0]` for the activations (gcc did not fold memcpy +
vdupq_n_s32 into LD1R), `ldr q`, `ldr q`, `ldp q` for the weights, 8
integer ops. At pinfer's prices that is 24 + 24 + 6 + 6 + 8 = 68 cycles,
and it runs in 88. The SADALPs are bunched at the loop's end, six in a
row, where nothing can pair with them.

Kept: 3 columns. A tile reads its half panel's weights once, so 3 columns
read them a third more often than 4 and a third less often than 2; the
4-thread gain being the larger says the four cores' shared L2 and DRAM are
part of the cost. The ff down shape (4096 inputs, whose column blocks are
only 48 wide) is the bandwidth-bound one: 1.82 -> 2.04 MAC/cycle at 1
thread, 1.41 -> 1.67 at 4 (the clock read 1400 MHz for these rows).

## H2 - the activations by LD1R

Stated before the edit. `vld1q_dup_s32` on a `may_alias` int32 (a GCC
and clang extension that makes the int8 buffer's four bytes a legal int32
read) instead of memcpy + `vdupq_n_s32`, so that gcc emits one LD1R where
it emits `ldr s` + `dup`: 6 of the 3-column loop's ~88 cycles. Predicted
gemmbench +3 to +7% at 1 thread.

### H2, measured: rejected

gcc 14 emits `ldr s` + `dup` for `vld1q_dup_s32` too (objdump: no LD1R in
the function), and gemmbench did not move (3.03 GMAC/s at 1 thread). Forced
with an inline-asm LD1R (18 of them in the function):

| | 1 thread | 4 threads |
|---|---|---|
| `ldr s` + `dup` (gcc) | 3.03 | 10.87 |
| `ld1r {v.4s}` (asm) | 2.70 (-11%) | 9.29 (-15%) |

LD1R is the slower way to replicate four bytes on this core, and the asm
block also takes the loads out of gcc's scheduling. The memcpy form stays.

## H3 - the column block for four A53s sharing 512 KB of L2

Stated before the edit. At 4 threads the ff down shape (4096 inputs) runs
at 1.65 MAC/cycle a core against 2.05 alone; the other shapes lose 4%.
PK_Q8_BLOCK_BYTES = 192 KB makes a block 48 columns of 4096; four threads,
each on its own 64-row slice and its own column block at any moment, want
up to 768 KB of activations in a 512 KB L2, besides the weights streaming
through it. A smaller block fits four but reads each slice's weights more
often (250/cb times a call). Measured over 48, 96, 128 and 384 KB at 4
threads; predicted that 96 KB (4 x 96 = 384 KB) gains ff down 5-15% at 4
threads and costs little alone.

### H3, measured: kept at 48 KB

gemmbench at 4 threads, GMAC/s (two sweeps, run on a cool board each):

| block | ff down | all four | | block | ff down | all four |
|---|---|---|---|---|---|---|
| 16 KB | 5.83 | 8.69 | | 96 KB | 9.17 | 10.86 |
| 24 KB | 6.17 | 9.02 | | 128 KB | 9.58 | 11.08 |
| 32 KB | 9.31 | 10.96 | | 192 KB (was) | 9.45, 9.15 | 10.97, 10.84 |
| **48 KB** | **10.31, 9.88** | **11.45, 11.25** | | 384 KB | 8.97 | 10.79 |
| 64 KB | 9.55 | 11.06 | | | | |

and alone, ff down 2.85 GMAC/s at 192 KB, 2.94 at 48, 2.89 at 32. The
repeats put the spread at about 4%; 48 KB is +8% on ff down at four threads
in both sweeps, and the three 1024-input shapes do not care. Below 32 KB a
block of 4-8 columns reads each slice's weights too often. Not the reason
predicted: 96 KB, which fits four blocks in the L2, is no better than 192.
At 48 KB a 4096-input block is 12 columns, four 3-column tiles, and a half
panel's 32 KB of weights is read into L1 once for those four; the 1024-input
shapes' blocks are 48 columns either way. neon-dp keeps 192 KB (measured on
the M4 Pro in main's commit; an A76 has not been asked).

### H1 + H3 together: the bench

`./pkito bench`, release build, each run started below 55 C:

| | run 1 | run 2 |
|---|---|---|
| baseline (b07c50d) | 58.681 s | 58.496 s |
| 3-column tile + 48 KB blocks | **51.801 s** | **51.793 s** |

-11.6%, against a spread of 0.3%. Kept.

## Where the int8 product stands (after H1 + H3)

gprof of a one-thread transcription (`-pg`, 181 s): pk_k_gemm_q8 87.6%,
pk_k_gemm_f32 4.5% (the attention, the VAD), pk_k_conv3x3s2 2.0%,
pk_k_silu 1.7%, pk_k_quantize 0.9%, pk_k_dwconv 0.7%, the attention's
repack 0.6%, pk_k_softmax 0.5%, the rest below 0.4% each. The four-thread
profile's 5% for the subsampling is its pointwise convolutions' and its
projection's int8 products, not its 3x3 convolutions.

gemmbench now also runs the kernel with everything in L1 (64 rows of 256
inputs, 48 columns, 400 calls): **2.40 MAC/cycle**, against 2.25 on the ff
up shape and 2.10 on ff down at one thread. The 3-column loop priced at
pinfer's rates, its per-block epilogue included (2 ADDP, 2 SCVTF, 2 FMLA by
element a column every four iterations, already minimal in gcc's output),
is about 74 cycles per 192 MACs, 2.6 MAC/cycle: the instruction stream runs
at 92% of its price, and the memory system takes another 7% alone and about
11% more at four threads. What is left is in the instruction count.

## H4 - eight activation bytes a load

Stated before the edit. Per column and pair of groups the activations are
`ldr s` + `dup`, twice: four issue slots. One `ldr d` (`vld1_s8`) and two
DUPs by element (`vdupq_lane_s32` lanes 0 and 1) are three. That is 3 of the
~74 cycles of a 3-column iteration: predicted gemmbench +3 to +5% at one
thread, L1-resident case included.

### H4, measured: rejected

gemmbench at one thread 3.07 -> 2.84 GMAC/s (-7.5%), L1-resident 2.40 ->
2.27 MAC/cycle; four threads 11.25 -> 10.13. gcc 14 took the 8-byte load
apart again - two `ldr s` and two `dup` per column, as before - and added
address arithmetic for the second load (objdump: six more integer ops an
iteration). The instruction order is gcc's to choose in C, and it chose
worse; only an asm loop would hold it to the form priced.

## The four-thread profile after H1 + H3

`make PROFILE=1`, 51.74 s: feed-forward 29.04 s (56.0%), q k v pos 7.67
(14.8%), conv pw1 pw2 4.66 (9.0%), attention 3.02 (5.8%), subsampling 2.82
(5.4%), attention out 1.59 (3.1%), decoder 1.02 (2.0%), quantize 0.77,
glu + depthwise 0.47, mel 0.44, layer norms 0.35. The int8 products run at
10.4 GMAC/s in the feed-forward (the 4096-input ff down among them) and
11.3-11.7 in the others.

## H5 - prefetch the weight stream

Stated before the edit. pinfer measured PRFM nearly free on this core (it
takes its AGU slot and never stalls). A half panel's weights are 32 bytes
of every 64-byte group, a stride of 64 through the panel, which the A53's
prefetcher may or may not follow ahead of an in-order core that stalls on
every miss. A `prfm pldl1keep` (`__builtin_prefetch(g + D, 0, 3)`) every
pair of groups, D bytes ahead, for D = 256 and 512. Predicted: ff down at
four threads +3 to +8%, the 1024-input shapes flat.

### H5, measured: rejected

gemmbench, GMAC/s, all four shapes (two passes; 0 is no prefetch):

| distance | 4 threads | | 1 thread | L1-resident |
|---|---|---|---|---|
| 0 | 11.03, 11.05 | | 3.08 | 2.40 MAC/cycle |
| 256 B | 10.68, 10.76 | | | |
| 512 B | 10.64, 10.79 | | 2.98 | 2.30 |

-3% everywhere, the L1-resident case included, where there is nothing to
fetch: a PRFM an iteration costs this loop more than its issue slot.

The four-thread repeats of one configuration have drifted by 4-8% between
sessions (ff down at 48 KB: 10.31, 9.88, 9.46, 9.46, 10.68, 9.79), so a
four-thread conclusion here wants an effect larger than that, or the bench.

## Why the 4096-input product loses at four threads

Two candidates, the second measured to be the one:

1. *Four threads thrash the shared L2 with four column blocks.* gemmbench
   can deal its items column-block-major, every thread on the same block of
   activations at once (which core/encoder.c's pk_gemm does not do). ff down
   at four threads, GMAC/s, blocks of 0 (rows only, as now), 24, 48, 96
   columns: 10.68 / 9.70 / 10.25 / 10.10, and again 9.79 / 10.29 / 9.75 /
   9.70 - nothing outside the drift. Rejected, and with it the idea of
   changing pk_gemm's dealing for this.
2. *A half panel of 4096 inputs does not fit L1.* Its 32 bytes of every
   64-byte group span 64 KB of cache lines, twice the A53's L1D, so every
   3-column tile streams them in again from the L2 the four cores share; at
   1024 inputs they are 16 KB and stay. Rows fixed at 1024, MAC/cycle a core:

   | inputs | 1 thread | 2 threads | 4 threads |
   |---|---|---|---|
   | 1024 | 2.24 | 2.22 | 2.17 |
   | 2048 | 2.13 | 2.14 | 2.01 |
   | 3072 | 2.08 | 2.07 | 1.88 |
   | 4096 | 2.10 | 2.04 | 1.74 |

   Flat alone, falling with width at four threads: the L2's bandwidth,
   shared, against a weight stream that should have been in L1.

## H6 - the inputs in chunks of 1024

Stated before the edit. For a product of more than 1024 inputs (the
ff down, 4096) with columns to reuse its weights over, take the inputs in
chunks of 1024 (32 blocks), outermost within a column block: each panel's
chunk is 16 KB of lines, used by both its halves and every tile of the
block from L1. Each output's chain of block FMAs is the same chain in the
same order, its float accumulator kept in thread-local memory between
chunks (stored and reloaded exactly), so the bits cannot change; a chunk
of activations is 1 KB a column, so a column block can be 48 columns
again. Predicted: ff down at four threads +15 to +25% (towards the 2.17 of
1024 inputs), alone +3 to +7%; the bench -3 to -5%.

### H6, measured: kept

gemmbench, GMAC/s (MAC/cycle a core at 1400 MHz):

| | 1 thread | 4 threads |
|---|---|---|
| ff down, before (the six repeats above) | 2.94-2.96 | 9.46-10.68 |
| **ff down, chunks of 1024** | 2.96 (2.11) | **11.41 (2.04)** |
| all four shapes | 3.07 (2.19) | 11.89 (2.12) |

and the width sweep at four threads, MAC/cycle a core, 1024 / 2048 / 3072
/ 4096 inputs: 2.15 / 2.07 / 2.07 / 2.04, where it was 2.17 / 2.01 / 1.88 /
1.74. Alone nothing moved, as a second thread's absence would have it.
`./pkito bench`: **49.164 s and 49.171 s**, from 51.80: -5.1%. gcc 14's
-Warray-bounds read the whole product's null partial-sum pointer as an
array of no floats and warned on the stores it never makes; the tiles of
a whole product get a real 32-float buffer they do not touch.

## The instruction order, asked of the core

A probe in the manner of pinfer's tests/isabench (inline asm, sixteen-wide
independence where it matters, L1-resident, one core, 1400 MHz; it lives
outside the tree, as the CLIMB rows are what it was for):

| loop | cycles | MAC/cycle |
|---|---|---|
| `smlal v.4s, v.4h, v.h[i]` x16, the int16 by-element form | 33.2 per 32 | 3.86 |
| an int16 panel: 8 rows x 5 columns x 8 inputs, its loads included | 230.5 | 2.78 |
| this file's int8 loop, 8 rows x 3 columns x 8 inputs, gcc 14's order | 156.0 per 2 | **2.46** |
| the same instructions, each column's SADALPs interleaved with the next column's activation loads and DUPs | 143.9 | **2.67** |
| SADALPs and multiplies alternating, two product sets | 147.9 | 2.60 |
| interleaved as above with one product set (the drains read the products before the next column's multiplies overwrite them, in program order, which an in-order core makes safe) | 143.9 | **2.67** |

The by-element SMLAL issues once a cycle, so an int16 form with the
weights re-laid out [k][row] (into L1, once a half panel and chunk) would
need no drain at all: 2.78 against 2.46 raw, but with the re-layout to pay
and a second weight path to keep. The order alone is +8.5% on the same
instructions.

## H7 - the 3-column loop in asm, in the measured order

Stated before the edit. The 8-input step of the ARMv8.0 3-column tile as
one inline-asm block, in the interleaved one-set order above: 12
accumulators, 4 weights, 4 products and 4 activation registers, which
leaves the tile's 6 float accumulators their registers. Integer sums only,
so ref's bits cannot change. Predicted: gemmbench +5 to +8% at one thread
(the epilogue and the memory system are unchanged), the bench -4 to -6%.

### H7, measured: rejected

| | 1 thread | 4 threads | L1-resident |
|---|---|---|---|
| gcc 14's C (b9fd42c) | 3.07 GMAC/s | 11.89 | 2.40 MAC/cycle |
| asm, one 8-input step a statement | 2.69 | 10.40 | 2.07 |
| asm, 32 inputs a pair of statements, immediate offsets | 2.86 | 10.99 | 2.22 |
| the same, the block's first drains SADDLP (no zeroing) | 3.06 | 11.72 | 2.38 |

At best level with gcc. The probe's "gcc order" row was this file's loop as
read from one objdump, not its schedule in place: in place, with its
integer work interleaved and dual-issued, gcc's loop must run near 2.55
raw, so the probe's +8.5% was against the wrong baseline. Two things on
the way: gcc allows an asm 30 operands and counts a "+w" twice (clang
counted otherwise and took the first version), so the temporaries became
named caller-saved registers; and C's 4095-character string limit splits
32 inputs into two statements. The asm is gone; what it showed is H8.

## H8 - no zeroing at a block's start, in C

Stated before the edit. The asm's last row gained 7% (2.86 -> 3.06) from
writing its accumulators with SADDLP at a block's first step instead of
zeroing twelve of them (`mov v.16b` from a zero register, objdump) and
accumulating. gcc's C does the same twelve moves every block (every four
steps at qb = 32). Peeled in C: the first step of a block drains with
vpaddlq_s16 (SADDLP), the rest with vpadalq_s16. Predicted: gemmbench +3
to +6% at one thread.

### H8, measured: kept

| | 1 thread | 4 threads | L1-resident |
|---|---|---|---|
| b9fd42c | 3.07 GMAC/s | 11.89 | 2.40 MAC/cycle |
| **first step SADDLP** | **3.23 (+5.2%)** | **12.43 (+4.5%)** | 2.45 |

ff down gained most alone, 2.96 -> 3.18 (its blocks are the same 32 inputs,
its tiles more of them). MAC/cycle a core: 2.31 alone, 2.22 at four.
`./pkito bench`: **47.319 s and 47.328 s**, from 49.17: -3.8%.

## The profile after H8, and the 3x3 convolutions

`make PROFILE=1`, 47.40 s: feed-forward 25.16 s (53.0%), q k v pos 7.42
(15.6%), conv pw1 pw2 4.52 (9.5%), attention 3.02 (6.4%), subsampling 2.73
(5.8%), attention out 1.54 (3.3%), decoder 1.02 (2.1%), quantize 0.77,
glu + depthwise 0.47, mel 0.44, layer norms 0.35. The int8 products run at
12.0-12.7 GMAC/s in every stage.

pk_k_conv3x3s2 alone (a probe outside the tree), one 20 s chunk's shapes,
one thread:

| | ms | GFMA/s | FMA/cycle |
|---|---|---|---|
| conv0, 2000x128 -> 1000x64x256, one input channel | 465.8 | 0.32 | 0.23 |
| dw1, 1000x64x256 -> 500x32x256 | 700.7 | 0.05 | **0.04** |
| dw2, 500x32x256 -> 250x16x256 | 165.8 | 0.06 | 0.04 |

1.33 s a chunk, 4 s of the bench's one-thread equivalent. The kernel loops
channels outermost: for each group of four it walks the whole image, its
input at a stride of the channel count (1 KB), using 16 bytes of each
64-byte line it fetches, through an image (dw1's: 64 MB) far past every
cache. pk_k_dwconv (the conformer's depthwise convolution) builds each tap's
four weights from strided scalars, for every row and channel group.

## H9 - the convolutions' loops turned over

Stated before the edit. pk_k_conv3x3s2: positions outer, channels inner,
the taps' weights transposed tap-major once a call (9 x 256 floats, on the
stack, 256 channels at a time), so that a position's input is read along
its channels as one stream and the weights from L1; for the one-channel
conv0, a position's nine inputs loaded once for all its channels.
pk_k_dwconv: its block of channels' taps transposed once a call, as backend/
avx2 already does. Each output's chain of taps is ref's chain in ref's
order, so the bits cannot move. Predicted: the probe's three shapes from
1.33 s to under 0.15 s a chunk, the bench -1.5 to -2.5%.

### H9, measured once, then stopped

The edit (positions outer, 16 channels a pass, the taps tap-major on the
stack) held ref's bits: kcheck neon on the Pi, neon and neon-dp on the Mac.
The probe's three shapes went from 1.33 s to 0.45 s a chunk: conv0 188 ms
(0.56 FMA/cycle), dw1 213 ms (0.12), dw2 48 ms (0.14). That is well short of
the predicted 0.15 s. In the probe's whole-image form each stage now waits
on its 64 MB of output or input rather than on its multiplies. One
`./pkito bench` gave **45.619 s**, -3.6% from 47.32, beyond the predicted
-1.5 to -2.5%.

It was stopped there, before the second run, and reverted, because main had
rewritten both kernels since this branch's base:

- d4b368b: pk_k_conv3x3s2 a row of output at a time in eight FMA chains,
  the weights tap-major, and core/encoder.c's sub_item streaming every
  subsampling stage through 3-row windows that stay in L2.
- b05af83: pk_k_dwconv 64 frames by 64 channels at a time, the taps
  tap-major.

**The convolutions are main's work.** Nothing of H9 is in this branch's
code.

## Where the climb ends

| | main at b07c50d | e870ae1 |
|---|---|---|
| `./pkito bench`, 60 s in 3 chunks (`--max-chunk 30`) | 58.681 s, 58.496 s | **47.319 s, 47.328 s (-19.3%)** |
| gemmbench, all four shapes, 1 thread | 2.68 GMAC/s, 1.91 MAC/cycle | 3.23, **2.31** |
| gemmbench, all four shapes, 4 threads | 9.31 GMAC/s, 1.66 MAC/cycle a core | 12.43, **2.22** |
| gemmbench, L1-resident (64 x 256, T 48) | - | 2.45 MAC/cycle |

The stage split, before and after:

- **Before:** main's b07c50d, its commit message, 58.4 s: feed-forward 57%,
  q k v pos 15%, conv pw1 pw2 9%, attention 5%, subsampling 5%,
  decoder 2%.
- **After:** above, 47.40 s, the feed-forward at 53%.

Kept:

- H1, a 3-column tile with SMULL2 on replicated activations: no spills.
- H3, 48 KB column blocks.
- H1 and H3 together: -11.6%, 7ca815b.
- H6, inputs in chunks of 1024 within a column block: -5.1%, b9fd42c.
- H8, SADDLP at a block's first step: -3.8%, e870ae1.

Rejected:

| Hypothesis | Result |
|---|---|
| H2, LD1R | -11% |
| H4, eight activation bytes a load | -7.5%: gcc split it back into `ldr s` |
| H5, prefetching the weight stream | -3%, even in L1 |
| Column-block-major dealing (a gemmbench prototype) | no change, so the pool was left alone |
| H7, the loop in asm in the measured order | at best level with gcc |
| H9 | not rejected: stopped, see above |

The tile now runs at 2.45 MAC/cycle in L1, about 93% of what its
instruction stream is priced at. A column's 8-input step costs about 24
cycles for 64 MACs:

- 8 multiplies;
- 4 SADALP at 2 cycles each;
- 2 `ldr s` and 2 DUP;
- a third of the 4 weight loads at 2 cycles each;
- a quarter of the block end.

Each change left in this form has been measured at or below gcc's code:
LD1R, a wider activation load, prefetching, and hand-ordered asm. Going
further needs fewer instructions per MAC. On ARMv8.0 that means the int16
by-element SMLAL: no SADALP, 3.86 MAC/cycle issued in the probe, 2.78 on an
int16 panel. It needs either a weight layout that kernels.h does not have
or widening on the fly, which spends most of the gain. Int16 weights would
not fit a 1 GB board.

Not measured here: the attention stage runs at roughly 0.85 FMA a cycle a
core. That is a count of its three products at three 20 s chunks: 14.3 G
multiply-adds in 3.02 s on four cores. f32_tile is priced at 2.46 (20
loads and 64 FMLA a step). Main's b325cd9, which turns gcc's first
scheduling pass off around the float products, has not been measured on
the A53.

## 2026-10-20 - on main after the merge: the float products' gcc pragma, measured here

Main's b325cd9 turned gcc's first scheduling pass off around the float
products, for the A76's spills (backend/neon-dp/CLIMB.md). On this Pi 3B+,
the merged tree (with main's convolution rewrites and fused quantizes),
`make PROFILE=1`, the default bench, each run after the SoC cooled below
55 C:

| | attention | minute (3 chunks) |
|---|---|---|
| the pass off (the pragma) | 2312 ms | 44.960 s |
| the pass on | 2035 ms | 44.618 s |

The A53 runs in order, and the scheduled code with its spills is faster
there than the unscheduled code without them. The pragma now applies to
neon-dp only. The merged tree with the pragma: 45.001 and 44.984 s in
three chunks, 48.863 s whole, from 47.3 s on e870ae1; peak RSS 652 MB, and
692 MB whole. Without it, the change that makes it neon-dp's: 44.586 and
44.636 s in three chunks, 48.116 s whole, attention 2038 ms; check-c
passing.

## 2026-10-20 - one binary for the family: the backend picked at run time

The Pi 3B+ builds neon-dp too now, and skips it at run time (no dot
product): its own build picks neon, 44.575 s in three chunks (44.586
before). The Pi 5's build, copied here as it was, picks neon and writes the
v2 transcript of test.wav byte for byte; `PKITO_KERNELS=neon-dp` is refused
with the reason.

## 2026-10-20 - pkito is now pktito

The project took its intended name, pktito (a little parakeet): the
binary, library, header, converter and PKTITO_KERNELS. The records above
keep the name they were written under.

## 2026-10-21 - a non-finite quantizer block is zero

The same check as neon-dp's, in this backend's k_quantize; its record, and
the M4 Pro's numbers for both backends, are in neon-dp/CLIMB.md. kcheck's
1,663 cases are ref's bits on the M4 Pro.

## 2026-10-22 - the resampler as a kernel, four blocks a vector (hypothesis)

core/resample.c's dot products ran in core, one chain of FMADDs an
output, about 34 taps a phase from 44.1 kHz and 38 from 48. `pktito
bench` on the first 20 s of test.wav resampled, its resample line (new,
best of five), two rounds on the Pi 3B+: 47.8 ms at 44.1 kHz, 53.2 ms at
48, about 5.8 cycles a tap at 1.4 GHz, the in-order A53 waiting on each
FMADD.

As a kernel, pk_k_resample (backend/neon-dp/CLIMB.md has its shape):
four blocks a vector, the inputs four LD1s to a lane, an FMLA a step for
four outputs, two phases at once against the FMLA's latency. A step of
four loads and an FMLA, perhaps 6 cycles on the A53, is 1.5 cycles a tap:
12 to 15 ms at 44.1 kHz. Kept if kcheck's resample cases are ref's bits
and the line falls past its spread.

Result, kept, short of the hypothesis: kcheck's resample cases ref's bits
and check-c passing on the Pi 3B+. The line: 23.1 to 23.2 ms at 44.1 kHz
(47.8 before), 30.5 at 48 (53.2), 2.1 and 1.7 times, not 4: the A53 runs
in order, and each LD1 to a lane waits on the vector before it. At 48 kHz
a block is three inputs, so LD3 would load a step's four lanes and the
next two steps' with one instruction: a next hypothesis, which would buy
little here, a minute of 48 kHz audio resampling in 0.09 s against the
48 s of transcribing it.

## 2026-10-23 - the RK3399 keeps its six cores, and should

The Rock Pi 4B (RK3399: two Cortex-A72 at 1.8 GHz, four A53 at 1.4 GHz;
Ubuntu 20.04, kernel 4.4, gcc 9.4), pktito d093d1a, its first run there:
check-c passes, kcheck's 1,777 neon cases ref's bits (neon-dp skipped,
no asimddp). `pktito bench` on test.wav with v2, best of five: the default
keeps all six, 33.6 s (1.8x real time), and -t 6 takes 31.2 s. Held to the
A72s (taskset 4,5, -t 2) it takes 51.5 s, to the A53s (taskset 0-3, -t 4)
57.7 s, so six is the right choice, by 1.6 times over the A72s, as on the
Snapdragon 7c (neon-dp/CLIMB.md). The board ran hot: 75.6 °C before the
held runs, 87.5 to 90.0 °C after them (thermal_zone0), so each figure is
the hot board's; the A72s' would close the gap only if their clock was
held down by more than a third.

## 2026-10-24 - the program is pktito-c (a note)

The command-line program the Makefile builds, `./pktito` in the rows
above, is `./pktito-c` from here on, so that it is not taken for the
app, pktito; the same program, the same output. The rows above keep the
name they were measured under.

## 2026-10-24 - pk_gemm's step, 256 rows where they make two rounds (measurement)

core/encoder.c's slices of 256 rows where a product's rows / 256 is at
least twice the threads (backend/avx2/CLIMB.md, pk_gemm's step), which at
four threads moves only the feed-forward's up product: Pi 3B+, neon: 44.70 s against 44.72 s (1.3x),
interleaved, test.wav, best of five. Level.

## 2026-10-25 - the Cortex-A35 (RK3326), 2.3 times the A53's time at its clock (measurement)

The Anbernic RG351M handheld (Rockchip RK3326: four Cortex-A35 at
1.416 GHz, 983 MB; ArchR 20260709, kernel 6.12), whose features (fp asimd
aes pmull sha1 sha2 crc32) are this backend's baseline. AliceBench's
chapter 1 (640.5 s) with v2, four threads, once each:

- pktito.net's pktito-cli 0.9.2 for Linux arm64 (gcc 11.4): 1282.81 s,
  0.50x real time.
- This repository's program as the Pi 3B+ built it on 2026-10-22
  (gcc 14.2): 1260.35 s transcribing and 14.47 s of VAD, 0.51x. 1.7% under
  the first, so the compiler is not the gap.

The clock read 1416 MHz in 48 of the first run's 52 readings, 30 s apart,
and 1008 MHz in the other 4, when the SoC reached its first trip point,
85 °C (84.6 and 85.0 °C at the two runs' hottest).

The Pi 3B+ ran the same pktito-cli binary over the same chapter that
evening, the first of its AliceBench run (ALICEBENCH.md in pktito-app):
558.30 s, 1.15x, at 1.4 GHz throughout (47.8 to 63.9 °C, under its 70 °C
soft limit). At about the same clock, the A35 takes 2.30 times the A53's
time.

Why is not measured. tests/gemmbench on the A35 would say whether it is
the int8 product (the feed-forward was 53% of the A53's time after the
climb), against the A53's 2.31 MAC/cycle on all four shapes. No climb for
it: a kernel changed for the A35 has to leave the A53 and the A72, which
run the same code, no slower, and the A53's tile is at about 93% of its
instruction stream's price ("Where the climb ends"). A reordering cannot
win back a unit that does less a cycle. A climb would have a case only if
gemmbench found one instruction the A35 runs far slower than the A53 that
the tile could do without. The board's full AliceBench stopped a few
minutes in, when its battery ran out under load, so ALICEBENCH.md has no
row for it.

## 2026-10-27 - the quantizer's rounding in one instruction, FCVTNS (hypothesis)

k_quantize rounds each scaled input with FRINTN and converts it with
FCVTZS. FCVTNS, the conversion that rounds to nearest with ties to even
itself, is one instruction for the two, and gives the same integers
here: the block's values are finite (a block with a NaN or an infinity
is zero before this loop) and at most 127 in magnitude once scaled, so
neither conversion saturates, and a value in (-0.5, 0) becomes 0 either
way (FRINTN's -0.0, truncated). pinfer's A76 table has FRINTN and FCVTZS
at 2.0 cycles each (relayed by pktito-mac). The quantizing loop is, a 16
inputs, 4 loads, 4 FMULs, 4 FRINTNs, 4 FCVTZSs, 6 narrows and a store,
after the block's max pass of about 20 more: FCVTNS takes 4 of about 43
instructions away. The quantize stage was 3.3% of a Pi 5 run on
2026-10-20 (backend/neon-dp/CLIMB.md); the A72 and the A53 are not
profiled since.

Predict: kcheck's bits unchanged; `make PROFILE=1`'s quantize stage 5 to
10% faster on the A72, the A53 and the A76; a run 0.1 to 0.4% faster,
which is about a quiet Pi's spread between best-of-5s, so it is kept only
if interleaved rounds put it past that under both compilers on a board:
its gcc and clang 21, cross-built here.

## 2026-10-27 - the softmax's exponentials sixteen at a time (hypothesis)

k_softmax's second loop takes one vector of four exponentials a step:
v_exp is a chain of about 15 dependent instructions, then the vector is
added into one of four accumulators. The A72 and the A76 run the next
steps' chains under this one's out of order, as far as their windows
reach; the in-order A53 runs what the compiler put next, which is this
chain's next instruction, waiting on the last. Four vectors a step give
it four chains to interleave. Each element still goes to lane i % 16 and
each lane takes its elements in order, so the sums, and the bits, are
ref's. pinfer's softmax went from 35.8 to 30.5 ms with four exponentials
in flight (relayed by pktito-mac); pktito's softmax, with the position
add before it, was 1.9% of a Pi 5 run's thread time on 2026-10-20
(348 ms, backend/neon-dp/CLIMB.md).

Predict: kcheck's bits unchanged; the attention stage faster by up to a
sixth of its softmax's time on the A53, by less on the A72 and the A76;
a run 0.1 to 0.3% faster on the A53 and within the spread elsewhere.
Kept only on a bench past the spread, as above.

## 2026-10-27 - the activations' columns 64 bytes apart past their width (hypothesis)

The int8 product's activations are column after column, column t at
q + t * n. On ARMv8.0 a 4096-input product, the feed-forward's down
projection (half its multiply-adds), is taken 1024 inputs a chunk (H6),
three columns a tile, and a tile reads its three columns' chunks in step,
4096 bytes apart: the same offset in their 4 KB pages. The A72's L1D is
32 KB in two ways, 16 KB a way, its set picked by address bits 6 to 13,
of which 12 and 13 come from wherever the page landed; a chunk's 16 KB of
weights fill about a way's worth of lines. Where two or three of a tile's
columns sit in pages that agree on bits 12 and 13, their lines contend
for one way, and with three random pages that is five tiles in eight.
With the columns n + 64 bytes apart, a tile's columns start a line apart
and never share a set. pinfer packed its activations at the width + 64
on a Pi 400 (PN_XQ_PAD) and took 1.9 to 3.7% off prefill where the
hidden width was 2048, neutral on a Pi 5 and the A53 (its
backend/neon-dp/CLIMB.md, 2026-09-24). Neither the A53 (four ways) nor
the A76 (64 KB in four ways, and 16 KB pages on the Pi 5's kernel, which
make bits 12 and 13 the address's own) has the A72's conflict. pi4 and
pi3 run 4 KB pages, no transparent huge pages. Only addresses move, so
the bits are ref's.

The change: pk_act carries the columns' stride, n + 64, which every
backend's quantizer writes and every int8 product reads, and the buffers
are sized for it. Measured first in a build of the core, ref and the NEON
backends, which is all an AArch64 build compiles. Predict, on the A72:
the feed-forward stage (PROFILE=1) 2 to 5% faster, and a run 1 to 2%
faster, past the spread under both compilers. On the A53 and the A76,
within the spread. If it holds, the AVX2 backends take the stride too
(they copy their columns each call, so only the copy's reads move), and
the 9950X is benched for no loss before it is kept.

## 2026-10-27 - FCVTNS, and the softmax's exponentials sixteen at a time (results: not kept)

The two hypotheses above, each against its base under the same compiler,
beside a layout control: 1120 bytes of code nothing calls at the head of
core/cli.c, which moves every object linked after it 32 bytes modulo 64.
kcheck ref's bits for every build on its board, and each build's test.wav
transcript test.v2.json's bytes. Interleaved rounds, the order turned
each round, test.wav through v2, best of 5 at the default threads; each
variant's ratio to its base a round, median (range):

| board | compiler | FCVTNS | softmax16 | control |
|---|---|---|---|---|
| pi4, A72, 6 rounds | gcc 14.2 | -0.07% (-0.37 to +0.22) | -0.08% (-0.34 to +0.02) | -0.01% (-0.22 to +0.39) |
| | clang 21.1 | -0.06% (-0.54 to +0.66) | -0.23% (-0.46 to +0.14) | +0.10% (-0.19 to +0.32) |
| pi3, A53, 3 rounds | gcc 14.2 | +0.07% (-1.88 to +0.18) | -0.09% (-1.14 to +0.12) | +0.18% (-1.22 to +0.35) |
| | clang 21.1 | -0.11% (-0.33 to +0.53) | +0.66% (-0.52 to +1.28) | +0.94% (-0.20 to +1.99) |
| pi5c, A76, on ebce152, 6 rounds | clang 21.1 | -0.38% (-1.12 to -0.22) | -0.10% (-0.49 to +0.03) | -0.03% (-0.38 to +0.16) |
| | gcc 15.2 | -0.18% (-0.43 to +0.19) | -0.23% (-0.54 to -0.19) | -0.25% (-0.43 to 0.00) |

pi4's and pi3's gcc builds were the boards' own (Debian 13), their clang
builds cross-compiled with -mtune for the core; pi5c's were all
cross-compiled static, on ebce152, where the 4-column SDOT tile's loop is
assembly at a 64-byte line (backend/neon-dp/CLIMB.md). pi3 reached its
soft temperature limit (throttled=0x80008), and its runs drifted 7%
slower over the rounds, which the rounds' turned order spreads over every
build. On pi5c before ebce152, a build of each with the product's loops
put 32 bytes along read FCVTNS at -0.08% under clang and -0.01% under gcc
against base the same (backend/neon-dp/CLIMB.md, the int8 product's loops
32 bytes along), where PROFILE=1 had its quantize stage 7% faster.

Neither moves a run past its control on any board under both compilers.
FCVTNS under clang on the A76 comes nearest, below base in all six
rounds, but at the edge of the control's range, and level under gcc 15;
the softmax at sixteen is nowhere past it, on the A53 it was written for
included. The quantize stage is faster by what the hypothesis said, and
that is about 0.4% of a run, under the boards' spread. Not kept.

## 2026-10-27 - the activations' columns 64 bytes apart past their width (results: not kept)

The hypothesis above against its base under the same compiler, beside the
layout control of the entry above, by the same method. kcheck ref's bits
for every build on its board (2259 cases with the stride, whose quantizer
and product cases take the columns packed or 64 bytes apart), and each
build's test.wav transcript test.v2.json's bytes. All cross-compiled
static on ebce152 with -mtune for the core; interleaved rounds, the order
turned each round, best of 5 at the default threads; each build's ratio
to base a round, median (range):

| board | compiler | stride | control |
|---|---|---|---|
| pi4, A72, 6 rounds | gcc 15.2 | -0.60% (-0.86 to -0.51) | +0.01% (-0.17 to +0.14) |
| | clang 21.1 | -0.21% (-0.33 to -0.10) | +0.06% (-0.11 to +0.28) |
| pi5c, A76, 6 rounds | gcc 15.2 | -0.31% (-0.60 to +0.03) | -0.25% (-0.43 to 0.00) |
| | clang 21.1 | -0.26% (-0.84 to -0.08) | -0.03% (-0.38 to +0.16) |

Both boards throttled=0x0. PROFILE=1 builds of base and stride on pi4,
one bench each: the feed-forward stage 11679 to 11563 ms a run under gcc
15 (-1.0%) and 13115 to 13035 ms under clang 21 (-0.6%), every other
stage within 0.6% either way. The A53 was not measured.

On the A72 the stride is below base in every round under both compilers,
past the control's range under gcc and touching it under clang, but a
third of what was predicted: a run 0.2 to 0.6% faster where 1 to 2% was,
the feed-forward 0.6 to 1.0% where 2 to 5% was. On the A76 it is within
the control's range under both, as predicted.

Where the rest went, by the cache's arithmetic (not measured): the stride
keeps a tile's columns out of one set at any one step, which is the
conflict the hypothesis counted. But over a tile's 1024 inputs, columns
whose pages agree on bits 12 and 13 still pass through the same sets, a
line apart in time, and a two-way set that takes two activation lines
between two uses of its weight line has lost the weight line, at the same
step or not. And the chunk's weights conflict among themselves: a panel's
16 KB for a chunk lie in four 4 KB pages, and unless all four differ in
bits 12 and 13 (24 draws in 256), two of them fill both ways of 64 sets,
where every activation line evicts a weight. The A53's four ways hold
both; the A72's two hold one.

Not kept: past the spread on one core of the three, at a third of the
prediction there, where keeping it means every backend taking the stride,
the AMX path's tile loads included, and a no-loss bench on each x86
family. If the A72's chunking is measured, the stride is worth measuring
again with it.

## 2026-10-28 - the release builds' compiler and tuning, arm64 (a measurement)

0.9.4's arm64 .deb is cross-built in an Ubuntu 22.04 container by its gcc
11.4, and a cross build's TUNE is empty (the Makefile reads the core's part
number only on the core), so the .deb is generic: `-march=armv8-a` for neon
and `-march=armv8.2-a+dotprod` for neon-dp, no -mtune. Measured here against
the same container's gcc 12.3 and clang 14.0, apt.llvm.org's clang 21.1.8
and this machine's cross gcc 15.2, and gcc 12, gcc 15 and clang 21 with
-mtune for the core (on the whole tree, as a native build on that core has
it), on be5c4ae's tree.
Every binary static; kcheck ref's bits for each build on its board (1779
cases on the ARMv8.0 cores; neon and neon-dp on the others), and each
build's test.wav transcript test.v2.json's bytes. Each build at two
placements, 0 and 32 bytes at the head of each NEON backend object
(through -include), but the Pi 3's tuning set at 0 only. Interleaved
rounds, the order turned each round, test.wav through v2, best of 5 at the
default threads; a build's time a round is its placements' mean, and the
tables give its ratio to the base a round, median (range).

The GB10 ran the compilers twice: 6 rounds of the eight builds, and again
among the tuned builds, where the script's turn, `(3i + r - 1) mod n`,
made each round of 18 builds three cycles of six; in that set each build
has six runs over the rounds, its figure is their median, and the runs
where the engine took 20 threads (two, 0.549 and 0.567 s) are left out. Two
rounds of the first set took 20 threads for gcc 12's pad 32 (0.552 and
0.556 s), which its range carries. The vLLM worker on it was idle (GPU
0%) through both sets.

pi3 reached its soft temperature limit in both sets (throttled=0x80000):
its first round of each ran partly at 1.4 GHz, and its ranges carry it.

The compiler, against the .deb's gcc 11.4, every build generic:

| board, rounds | backend, threads | gcc 11.4 | gcc 12.3 | clang 14.0 | clang 21.1 |
|---|---|---|---|---|---|
| pi3, A53, 3 | neon, 4 | 51.386 s | -5.69% (-5.71 to -5.32) | +54.98% (+54.29 to +61.80) | +6.84% (+6.53 to +12.22) |
| pi4, A72, 3 | neon, 4 | 23.589 s | -1.09% (-1.12 to -0.95) | +46.55% (+46.36 to +46.90) | +6.60% (+6.37 to +6.67) |
| pi5c, A76, 6 | neon-dp, 4 | 3.670 s | -0.28% (-0.38 to -0.20) | -0.61% (-0.80 to -0.54) | -0.42% (-0.57 to -0.31) |
| gb10, X925, 6 | neon-dp, 10 | 0.433 s | -0.29% (-1.14 to +14.00) | -1.32% (-2.18 to -0.46) | -1.45% (-2.18 to -0.35) |
| gb10, the second set | | 0.4350 s | -0.86% | -1.03% | -1.21% |

The tuning, against gcc 12.3 generic:

| board, rounds | gcc 12 a53 | gcc 12 a72 | gcc 12 a76 | gcc 15.2 generic | gcc 15 a53 | gcc 15 a72 | gcc 15 a76 | gcc 15 x925 | clang 21 | clang 21 tuned |
|---|---|---|---|---|---|---|---|---|---|---|
| pi3, A53, 3 | -0.22% (-0.24 to +0.17) | +3.30% (+2.93 to +3.67) | | +0.48% (-0.02 to +4.09) | +0.39% (+0.10 to +5.24) | +3.84% (+3.59 to +9.85) | | | +13.77% (+13.75 to +20.98) | a53: +9.65% (+9.40 to +16.52) |
| pi4, A72, 3 | -1.45% (-1.48 to -1.42) | -9.16% (-9.17 to -8.93) | | -0.67% (-0.88 to -0.65) | -0.87% (-0.95 to -0.84) | -8.56% (-8.71 to -8.55) | | | | |
| pi5c, A76, 6 | | | -0.01% (-0.20 to +0.05) | -0.08% (-0.10 to +0.01) | | | +0.20% (+0.11 to +0.31) | +0.33% (+0.16 to +0.45) | -0.26% (-0.33 to -0.19) | a76: +0.01% (-0.12 to +0.18) |
| gb10, 6 runs a build | | | +0.46% | +0.35% | | | +0.29% | +0.52% | -0.35% | a76: +0.29% |

gb10's runs of one build spread over 1 to 2%, so nothing there is past
its noise; pi5c's spread between placements is at most 0.25%, pi4's 0.12%.

On the dotprod cores, then, no compiler or tuning moves a run by more
than 1.5%. On the ARMv8.0 cores clang is 7 to 14% behind gcc, and gcc 12
is ahead of gcc 11 by 5.7% on the A53 and 1.1% on the A72. Tuning for the
A72 is worth 9% on the A72 and costs 3.3 to 3.8% on the A53; tuning for
the A53 is within 0.5% of generic on the A53 and 0.2 to 1.5% ahead of it
on the A72.

Where the A72's 9% is. PROFILE=1 builds of gcc 15 generic and tuned for
the A72 on pi4, two runs of each: the int8 products' stages 8.6 to 10.7%
faster tuned (feed-forward 12768 to 11675 ms a run, q k v 4005 to 3578,
attention out 817 to 731, conv 2415 to 2171), subsampling 4.9%, and
attention 6.1% slower (943 to 1001 ms); every other stage within 1.5%.
The ARMv8.0 tile's step loop, three columns of 8 inputs, is the same 59
instructions under every tuning, and only their order differs: gcc's
generic order is its A53 order (the same instructions in the same places,
the registers renamed), every load and DUP, then the 12 SMULL, the 12
SMLAL and the 12 SADALP; the A72 order interleaves the three. Both loops
copied into a probe (the loads on one 4 KB buffer, w3 held still, the
same instructions otherwise), each with its head 0, 8 and 32 bytes past a
64-byte line, on pi4 at 1.8 GHz: the generic order takes 44.04 cycles a
step and the A72 order 36.53, at every one of the three heads.

What the A72 issues of the tile's instructions, operands in registers,
all independent (16 a pass, unrolled 8 times): SMULL, SMULL2, SMLAL,
SADALP, SADDLP and DUP from a w register once a cycle each; DUP by element
and a 4s ADD 1.49 a cycle; mixed, SMULL and SADALP 8 + 8 1.46 a cycle,
2 : 1 1.23, SMULL with DUP by element or ADD 1.35, SADALP with DUP by
element 1.32, SMULL with a 128-bit load 1.49, SADALP with a 32-bit load
1.99. So the multiplies take one pipe and the SADALPs share it with them
some of the time; how the A72 assigns them was not worked out.

So the release build gets the A53's order, in which the A72 takes 21%
longer than in its own, and neither tuning is right for both cores in one
binary.
For the .deb: gcc 12, generic. The loop's order is a climb of its own
(the next entry).

## 2026-10-28 - the ARMv8.0 tile's blocks at 32 inputs as one loop of assembly (hypothesis)

The entry above found that the release build runs the tile's step in the
A53's order, where the A72 takes 44.04 cycles a step against its own
order's 36.53, and that the A53 takes 3% longer in the A72's order than
in its own; one binary carries only one. A loop of assembly has the same
order under every compiler and tuning, so the question is which order,
and whether one order is near the best on both cores. Measured with
probes (scratch, not the tree): the loop's instructions written out by a
generator from a pattern, on one buffer, timed on pi4 (A72, 1.8 GHz) and
pi3 (A53, 1.4 GHz; each batch started below 56 C and ran on one core;
get_throttled read 0x80000 around each, the soft limit reached earlier in
the day and not engaged, and the runs of one order agree within 0.1%).

What the A53 issues, by the probes of the entry above: SMULL, SMULL2,
SMLAL, SADDLP, DUP by element and a 4s ADD once a cycle and never two
together; SADALP, Q or D form, once every two cycles; DUP from a w
register about twice a cycle. With a vector instruction, a 128-bit load
or a 32-bit load takes its own cycles (SMULL and ldr q 8 + 8 in 24
cycles, SADALP and ldr s in 24): the A53 pairs a vector load with no
vector instruction. A step's price on it is then about its vector
instructions' cycles plus its vector loads': 24 multiplies, 12 SADALP at
two, the 4 weight vectors' two LDPs, and the activations.

The step, 8 inputs by 3 columns (192 multiply-adds), as units: a unit is
one column's 8 multiplies (SMULL, SMULL2 against the low and high halves
of the first group's two weight vectors, then SMLAL, SMLAL2 of the
second's) into 4 int16 sums, whose 4 SADALP are placed in the next unit
among its multiplies, while it loads the next unit's activation words
into the other of two register pairs. A unit's order is a pattern of its
slots, M (the next multiply), S (the next SADALP of the unit before) and
the loads. On one 4 KB buffer, its addresses fixed, cycles a step:

| order | A72 | A53 |
|---|---|---|
| gcc's generic (the .deb's) | 44.04 | 67.42 |
| gcc's A72 | 36.53 | 69.44 |
| column by column, each column's SADALP after its multiplies | 48.05 | 72.46 |
| M M S four times, words by ldr s + DUP | 35.31 | 68.94 |
| hill-climbed on the A72, single moves, ldr s + DUP (`S M S a b S M M S M M M A M M B`) | 31.86 | 68.93 |
| M M S, the words by LD1R | 37.81 | 62.91 |
| hill-climbed on the A72, LD1R (`S M M S M S M a b M M M M S`) | 33.03 | 62.92 |
| M M S, the words by ldr d and two DUPs by element | 35.82 | 65.93 |

The A53 takes every interleaving of M and S at the same price (eleven
orders measured at 68.93 to 68.95 with ldr s + DUP, four at 62.90 to
62.92 with LD1R): what it pays for is instructions, a DUP a cycle that
LD1R does not need. The A72 is the other way: the order moves it from
48 cycles to 32, and LD1R costs it.

A block is 12 units, then each column's int32 pairs added (ADDP), made
floats (SCVTF) and added to its floats times the block's scale (FMLA by
element): the C tile's operations on the same values in the same order,
so the same bits. Registers: 4 weight, 4 activation, 6 int16 sums (unit
u's j-th in v8 + (j - u) mod 6, which every order here allows when each
unit's j-th SMULL comes after its j-1-th SADALP of the unit before, and
which comes back to the same registers every 12 units), 12 int32 and 6
float sums: 32, the finish in place (the block's scale loaded into an
int32 sum its ADDP has read). Each block's finish is spread over the
next block's first three units, and the block's last unit loads the
next block's first weights and words, so the loop runs a head (block
0's first three units), nb - 1 rounds of units 3-11 and the next block's
0-2, and a tail (the last block's units 3-11 and its finish): no load
past the tile's last block. Timed as k_gemm_q8 calls it, 32 blocks a
call (a 1024-input chunk: the half panel's 8 KB of weights and the
columns' 1 KB each, scales of 1.0), cycles a block:

| loop | A72, six processes | against the last | A53, three | against the last |
|---|---|---|---|---|
| words by ldr s + DUP, post-indexed (`S M S a b S M M S M M M A M M B`) | 159.8 to 179.2 | +7.7% (+6.1 to +8.0) | 303.7 to 307.8 | +11.8% (+11.6 to +11.8) |
| words by LD1R, post-indexed (`S M M S M S M a b M M M M S`) | 174.6 to 195.4 | +17.6% (+16.2 to +18.3) | 279.3 to 283.8 | +2.8% (+2.8 to +2.9) |
| words by ldr d at offsets, climbed on the A53 (`L M S M S M S M S M A M M M B`) | 159.0 to 176.6 | +7.0% (+4.9 to +7.4) | 270.6 to 274.8 | -0.4% (-0.4 to -0.3) |
| words by ldr d at offsets, climbed on the A72 (`S M M S S M S M L M M A M M B`) | 149.7 to 165.6 | +0.7% (-1.5 to +1.1) | 281.7 to 285.8 | +3.7% (+3.6 to +3.7) |
| words by ldr d at offsets, `L S M M S S M S M M M A M M B` | 148.1 to 167.5 | | 271.6 to 275.8 | |

(L the ldr d of the next unit's two words into its b register, A and B
the DUPs of its lanes 0 and 1 into a and b.) One binary of the five, run
six times on pi4 and three on pi3; each loop's ratio to the last a
process, median (range). The A72's figures move by up to 13% from one
process to the next with the pages the buffer gets (its two-way L1: the
conflicts the entry of 2026-10-27 on the activations' stride counted),
the loops together; the A53's by 1.5%. The climbs ran on this harness,
each batch beside its starting order in one process.

So the loop takes the last order: level with the A72's climbed order on
the A72 and within 0.4% of the A53's climbed order on the A53, where
LD1R is 2.8% behind it, and 17.6% on the A72. LD1R has no offset form,
so its loop steps a pointer on every load, as the ldr s loop here does;
ldr d takes immediate offsets, and its loop steps the four pointers once
a block. Whether the post-indexing or the memory puts the A72's ldr s
loop 7.7% behind in the block, where its order led in the step, was not
separated.

Predicted, against the C tile under gcc 12 generic (the .deb's): the
tile's block from about 196 cycles (the step's 44.0 four times and the
finish) to 148 on the A72, and from about 295 (67.4 four times and the
finish) to 273 on the A53. Today's
profiles put the order-sensitive part of the A72's int8 stages at about
60% (the A72 tuning's 17% in the step moved them 9.2%), so on pi4 the
int8 stages 13 to 16% faster and a run 11 to 14%, and against a tree
tuned for the A72 (TUNE from its part number) 3 to 6%. On pi3, a run 4
to 7% faster. On the cores with dotprod nothing changes: they run
neon-dp. Measured by tests/kcheck (ref's bits on every case: the
3-column tile at 32 inputs whole, and k_gemm_q8's 1024-input chunks
first, inner and last, n from 32 to 4096, nb from 1), on pi3, pi4,
pi5c and the M4 Pro (each runs the neon backend too); then interleaved
rounds of pktito-c, base against the loop, under gcc 12 and clang 21 at
two placements, on pi4 and pi3, and PROFILE=1 for the stages.

## 2026-10-28 - the ARMv8.0 tile's blocks at 32 inputs as one loop of assembly (results: kept)

The hypothesis above against be5c4ae (base) under the same compiler. Both
trees cross-built static and generic in the Ubuntu 22.04 container, by
its gcc 12.3 and by apt.llvm.org's clang 21.1.8, each at two placements
(0 and 32 bytes at the head of each NEON backend object); on pi4 also
be5c4ae under gcc 15.2 with -mtune=cortex-a72 on the whole tree, the
tuning a Pi 4's own build has. kcheck's 1779 cases ref's bits on pi3
and pi4 under both compilers, on pi5c (neon and neon-dp) under both, on
the M4 Pro under Apple clang 21, and on armlet (Windows 11 on Arm,
llvm-mingw's clang 23.1.4), where all of check-c's commands passed;
every build's test.wav transcript test.v2.json's bytes, and with
PKTITO_KERNELS=neon on pi5c, the M4 Pro and armlet too. kcheck fails
with either of two one-register faults in the finish, an FMLA reading
the wrong register for its scale, once in the loop and once in the tail.

The product alone, gemmbench at T 250 (the encoder's four shapes, as
pk_gemm calls them), two rounds, pi3's from 45 C, below its limit: the
four shapes' time, and in brackets MAC a cycle a core at cpufreq's
clock:

| board | compiler | threads | base | loop | change |
|---|---|---|---|---|---|
| pi4, A72, 1.8 GHz | gcc 12.3 | 1 | 480.7 ms (3.33) | 362.6 ms (4.42) | -24.6% |
| | | 4 | 123.9 ms (3.23) | 96.7 ms (4.14) | -22.0% |
| | clang 21.1 | 1 | 523.8 ms (3.05) | 367.8 ms (4.36) | -29.8% |
| | | 4 | 135.8 ms (2.95) | 97.4 ms (4.12) | -28.3% |
| pi3, A53, 1.4 GHz | gcc 12.3 | 1 | 903.0 ms (2.28) | 842.5 ms (2.44) | -6.7% |
| | | 4 | 235.8 ms (2.19) | 222.3 ms (2.32) | -5.8% |
| | clang 21.1 | 1 | 1072.5 ms (1.92) | 849.6 ms (2.42) | -20.8% |
| | | 4 | 278.5 ms (1.85) | 222.4 ms (2.31) | -20.2% |

In L1 (64 rows by 256 inputs, T 48, one thread), a block's 768
multiply-adds take the A72 212 to 217 cycles under gcc and 233 to 235
under clang before, 159 to 167 under both with the loop, where the
probe's loop alone took 148 to 168; the A53 312 under gcc and 366 to 367
under clang before, 290 to 293 under both with the loop, where the
probe's took 272 to 276.

Whole runs: interleaved rounds, the order turned each round, test.wav
through v2, best of 5 at the default threads (4); a build's time a
round is its placements' mean, and its ratio to base a round, median
(range):

| board, rounds | compiler | base | loop | change |
|---|---|---|---|---|
| pi4, A72, 3 | gcc 12.3 | 23.425 s | 18.926 s | -19.17% (-19.34 to -19.09) |
| | clang 21.1 | 25.242 s | 18.909 s | -25.09% (-25.13 to -25.01) |
| | gcc 15.2, -mtune=cortex-a72 | 21.398 s | | the loop under gcc 12 against it: -11.55% (-11.79 to -11.45) |
| pi3, A53, 3 | gcc 12.3 | 48.484 s | 46.654 s | -3.32% (-3.78 to -2.77) |
| | clang 21.1 | 54.819 s | 46.847 s | -14.50% (-14.74 to -14.03) |

pi4 throttled=0x0 through every run, 60 to 69 C, its placements within
0.25% of each other. The loop runs level under the two compilers, on
both boards (18.926 and 18.909 s, 46.654 and 46.847 s). make check-c
with the loop passes on pi4 (Debian's gcc 14.2, which the Makefile tunes
for the A72; its bench 18.788 s) and on pi3 (the same gcc, tuned for the
A53; 43.544 s).

PROFILE=1 builds under gcc 12, two runs of each, ms a run:

| stage | pi4 base | pi4 loop | change | pi3 base | pi3 loop | change |
|---|---|---|---|---|---|---|
| feed-forward | 12990 | 10146 | -21.9% | 26472 | 25423 | -4.0% |
| q k v pos | 3992 | 3089 | -22.6% | 7990 | 7709 | -3.5% |
| attention out | 817 | 635 | -22.2% | 1659 | 1604 | -3.4% |
| conv pw1 pw2 | 2412 | 1882 | -22.0% | 4857 | 4664 | -4.0% |
| subsampling | 600 | 538 | -10.3% | 1299 | 1277 | -1.7% |
| run, best of 5 | 23.42 s | 18.94 s | -19.1% | 48.23 s | 46.69 s | -3.2% |

(the subsampling's two pointwise convolutions are int8 products). Every
other stage within 2.8% on pi4 and 2.4% on pi3.

pi3 ran at its soft temperature limit from the first round's third
build on: 70 C on this board (its firmware's temp_soft_limit=70),
throttled=0x80000, 69 to 70 C after every bench, the core clock at 400
MHz throughout. There the firmware holds the temperature by the ARM
clock, between 1.2 and 1.4 GHz, so a time is the loop's cycles and the
clock together. After the rounds, three pairs of five single runs, base
and the loop under gcc 12 alternating, the ARM clock read every half
second (vcgencmd measure_clock), each run's transcription time against
its samples' mean clock:

| | base | loop | change |
|---|---|---|---|
| a run at 1400 MHz throughout (each's first three) | 46.40 to 46.45 s | 44.15 to 44.31 s | -4.7% |
| cycles a run, time by mean clock, all fifteen, median (range) | 64.96 G (63.95 to 65.87) | 61.47 G (60.62 to 61.87) | -5.4% |
| mean clock at the limit (pairs 2 and 3), median | 1288 MHz | 1251 MHz | -2.9% |

So the loop takes 5.4% of a run's cycles at every clock, and 4.7% of its
time at 1.4 GHz; held at its limit, the board ran the loop's runs at a
clock about 3% lower, and the rounds' 3.3% is what was left. That the
loop draws more power a cycle, which the limit answers with the clock,
is the likely reading; the power was not measured.

Against the prediction. The block, from about 196 cycles to 148 on the
A72 (-24.5%), held: gemmbench -24.6% at one thread, -24.4% in L1. The
stages went past it: predicted 13 to 16% for the int8 stages and 11 to
14% for a run, measured 22.1% and 19.2%. The prediction took 60% of the
int8 stages to move with the step's order, from the A72 tuning's 9.2% on
them over its 17% on the probe's step; the loop's 22% on them against
its 24.5% says they move nearly whole, and so that the A72 order took
less off the tile than off the probe (not measured apart). Against the
tree tuned for the A72, predicted 3 to 6%, measured 11.6%. On pi3,
predicted 4 to 7% a run: 4.7% at its full clock, 3.3% held at its soft
limit.

Kept: past the spread on both boards under both compilers. The release
build is generic, and with the loop it runs the A72 11.6% ahead of the
C tile tuned for it, so no -mtune is missed. An NDK build for an ARMv8.0
phone, which only clang makes, takes the clang rows. The C step stays
for tiles of fewer than three columns and for blocks other than 32.

The entry before the hypothesis said the .deb is cross-built. It is
built on the GB10, natively, in an arm64 Ubuntu 22.04 container by its
gcc; TUNE is empty there too, since the Makefile tunes only for the A53,
A72 and A76 (0xd03, 0xd08, 0xd0b) and the GB10's cores are 0xd85 and
0xd87. jammy's arm64 gcc-12 and gcc-11 (12.3.0-1ubuntu1~22.04.3,
11.4.0-1ubuntu1~22.04.3) make the same objects as the cross compilers
these entries measured (the same versions, cross1): all 37 of the tree's
objects, static and PIC, on be5c4ae and with the loop, the same under
objdump -dr, at the release's flags. So the figures stand for the .deb.

## 2026-10-28 - the loop under the NDK (a check)

3720a1b built by NDK r30's clang 21 (API 24) without a warning, and make
adb on the Tab S7 FE (Snapdragon 778G, whose cores run neon-dp) passes
check-c: kcheck's 1779 cases ref's bits on neon-dp and on neon, test.wav
ref's under v2 and v3 on every token and time, and with
PKTITO_KERNELS=neon test.wav's v2 --json test.v2.json's bytes. Run by
the android-9950x session. neon's bench there was 7.453 s at 8 threads
and neon-dp's 3.639 s; the C tile was not timed beside it, so this is a
check of the bits, not of the speed. No device on hand runs neon by
default: all four have dotprod.
