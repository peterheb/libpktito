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
