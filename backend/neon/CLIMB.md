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

## The baseline (main at 166e169)

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
| baseline (166e169) | 58.681 s | 58.496 s |
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
| gcc 14's C (a01a01f) | 3.07 GMAC/s | 11.89 | 2.40 MAC/cycle |
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
| a01a01f | 3.07 GMAC/s | 11.89 | 2.40 MAC/cycle |
| **first step SADDLP** | **3.23 (+5.2%)** | **12.43 (+4.5%)** | 2.45 |

ff down gained most alone, 2.96 -> 3.18 (its blocks are the same 32 inputs,
its tiles more of them). MAC/cycle a core: 2.31 alone, 2.22 at four.
`./pkito bench`: **47.319 s and 47.328 s**, from 49.17: -3.8%.
