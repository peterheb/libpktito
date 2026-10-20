# backend/avx2 and avx2-vnni: the climb

Dated records. Append; do not rewrite old rows; rejected hypotheses stay.

# 2026-10-20: the board, the baseline, and what limits it

## The board

| | |
|---|---|
| CPU | Intel Core i9-12900HK (Alder Lake-H): 6 P-cores (Golden Cove, 2 threads each, CPUs 0-11, 4.9-5.0 GHz, 1.25 MB L2 each) + 8 E-cores (Gracemont, CPUs 12-19, 3.8 GHz, 2 MB L2 per four) |
| L3 | 24 MB, shared |
| power | PL1 = PL2 = 45 W (`/sys/class/powercap/intel-rapl:0`), governor powersave (HWP) |
| OS, compiler | Ubuntu, kernel 7.0, gcc 15.2.0 |
| flags | avx2: `-O3 -mavx2 -mfma -std=c11 -ffp-contract=off`; avx2-vnni adds `-mavxvnni` |
| bench | `./pkito bench -t N --max-chunk 60 test.wav` (Silero's 60 s test.wav whole: T = 750 encoder frames), best of 5 per run; the spread is over 5 runs |

## Baseline (main at 166e169)

`x86bench.sh`: 5 runs of `pkito bench`, each best of 5.

| build | threads | min | median | max |
|---|---|---|---|---|
| avx2-vnni | 1 | 5.460 | 5.512 | 5.597 |
| avx2-vnni | 6 | 1.652 | 1.656 | 1.658 |
| avx2 | 1 | 7.706 | 7.709 | 7.712 |
| avx2 | 6 | 2.259 | 2.260 | 2.261 |

So the spread is 0.4% at 6 threads and 2.5% at one (one core's turbo wanders
between 4.5 and 5.0 GHz). Profile (`make PROFILE=1`), avx2-vnni, one thread:
feed-forward 50.0%, q k v pos 15.5%, float attention 15.3%, conv pw1/pw2 9.3%,
attention out 3.4%, subsampling 2.1%, decoder 1.8%, the rest under 1.2% each.
At 6 threads every stage is ~3.3x faster than at one.

## What limits it

**Power, then instructions.** turbostat (PkgWatt, Bzy_MHz over the bench):

| threads | package | busy MHz | bench |
|---|---|---|---|
| 1 | 44.1 W | 4537 | 5.51 s |
| 6 | 45.0 W | 3081 | 1.655 s |
| 12 | 44.6 W | 2734 | 1.862 s |

The package is at its 45 W limit with one thread already. Six threads run at
3.08 GHz, and more threads (hyperthreads, E-cores) only spread the same 45 W
thinner. perf stat (cpu_core cycles and instructions over the whole bench,
five runs):

| threads | core GHz | core instructions | atom instructions | IPC |
|---|---|---|---|---|
| 1 | 4.94 | 727e9 | 0.15e9 | 5.30 |
| 6 | 3.07 | 729e9 | 0.06e9 | 4.88 |
| 12 | | 916e9 | 355e9 | |
| 20 | | 834e9 | 551e9 | |

At 12 and 20 threads the instruction count grows 74-90%: the pool's workers
spin between jobs, and on a power-capped chip a spinning core's watts are
taken from the working ones. On this board time is energy, and energy is
instructions: **the int8 product's instructions per MAC is the lever**, at
one thread and at six.

**The int8 product alone** (`tests/q8bench`, one thread, taskset to a P-core,
T = 750, blocks of 32):

| build | 4096x1024 | 1024x4096 | 1024x1024 | decoder 2560x640, T 1 |
|---|---|---|---|---|
| avx2-vnni | 113.4 GMAC/s | 112.2 | 113.6 | 19.3 |
| avx2 | 73.3 | 73.7 | 73.4 | 47.6 |

At ~4.9 GHz that is 23 MAC/cycle for VNNI against a peak of 64 (two
VPDPBUSD a cycle, 32 MACs each) and 15 for VPMADDUBSW, at IPC 5.05 and 4.55:
the core issues near its width and most of what it issues is not a multiply.
The VNNI inner loop gcc emits, per group of 4 inputs of an 8-row, 4-column
tile (perf annotate, objdump):

    vpbroadcastd x4   activations          (load port)
    vpxor x4          +128 on each         (ALU)
    vpdpbusd x4       the 128 MACs         (ALU p0/p1)
    vmovdqu x1        weights              (load port)
    vmovdqa x8        accumulators copied in and out of the loop (!)
    mov/lea/shl/add/cmp/jb x7

28 instructions for 128 MACs. The eight register copies are gcc keeping the
tile's accumulator array live across the k loop; the four XORs are the
activation offset applied per broadcast, where one XOR of the weight vector
would serve every column. And the decoder's T = 1 product runs at a sixth of
the encoder's rate on VNNI, a third of avx2's: a call of more than 64 rows
takes each half panel's block sums of the weights in a pass of its own, which
at one column is as much work as the product.

## H1, before the edit: VNNI with the offset on the weights

Put the +128 on the weights instead (XOR the weight vector once a group, for
every column of the tile) and take 128 x the block's sum of the *activations*
back off, which depends on the column and the block but not on the row: one
sum a column a block, taken once a call over T x n bytes, not over the
weights. That removes the weight-sum pass (the decoder at T = 1) and the
per-broadcast XOR. Write the tile with its accumulators unrolled so gcc keeps
them in registers, 16 rows (a whole panel, two vectors) by 3 columns: per
group 2 weight loads, 2 XORs, 3 broadcasts, 6 VPDPBUSD, 13 instructions for
192 MACs, in 15 of the 16 ymm registers. The bits cannot move (the integer
sums are exact either way). Predict q8bench 113 -> ~200 GMAC/s on the encoder
shapes, the decoder 19 -> 60+, and bench at 6 threads 1.656 -> ~1.25 s.

**H1, amended before measuring.** The column sums are taken per call, and a
call is a 64-row slice of the product (core/encoder.c's pk_gemm), so each
slice takes all T columns' sums again: at T 750 and 4096 inputs about 768K
instructions, ~15% of a slice's product, most of it the horizontal add of
each block's sum. The two offsets cost differently with the shape: the
weights' block sums are 1/T of the product, the activations' 1/rows. So two
paths: at T >= 16, the activations offset - XORed once a column block into a
thread-local copy, 192 KB, so that the inner loop has no XOR at all - and the
weights' block sums taken a panel at a time (1/cb of the product, 2% at 4096
inputs); below 16 columns (the decoder), the weights offset as H1 states,
with the column sums. Both on the 16-row, 3-column tile: per group 2 weight
loads, 3 broadcasts and 6 VPDPBUSD (11 instructions for 192 MACs) at T >= 16.

**H1, measured.** tests/kcheck: ref's bits on avx2 and avx2-vnni. q8bench,
one P-core, VNNI: 113.4 -> 142.4 GMAC/s on 4096x1024 (139.5 on 1024x4096,
142.5 on 1024x1024); the decoder's T = 1 product 19.3 -> 63.1 GMAC/s. Short
of the ~200 predicted, and objdump says why: gcc still copies all twelve
accumulators in and out of the inner loop (12 VMOVDQA a group) and spends 9
scalar instructions a group on the two addresses - 32 instructions for 192
MACs, 6 MAC an instruction, at IPC ~5 the 30 MAC/cycle measured.

## H2, before the edit: unroll the block

The k loop's trip count is a runtime qb, and gcc does not coalesce the
accumulators across a loop it cannot unroll. With blocks of 32 (every encoder
product), make the block's eight groups straight-line code at constant
offsets from one weight pointer and three activation pointers: no copies,
no per-group address arithmetic, 11 instructions a group. The decoder's
qb = n keeps the loop. Predict q8bench ~142 -> ~200 GMAC/s.

**H2, measured.** tests/kcheck: ref's bits (avx2-vnni; avx2 untouched).
q8bench, one P-core: 142.4 -> 228.8 GMAC/s on 4096x1024, 213.1 on 1024x4096,
228.7 on 1024x1024: about 47 MAC/cycle at ~4.85 GHz, 73% of VPDPBUSD's 64.
The decoder's qb = n path keeps its loop: 64.3. Kept. The engine:

| build | threads | min | median | max | was (median) |
|---|---|---|---|---|---|
| avx2-vnni | 1 | 3.359 | 3.419 | 3.447 | 5.512 |
| avx2-vnni | 6 | 1.056 | 1.057 | 1.062 | 1.656 |

-36% at six threads. The profile at six moves the float attention to 24.1%
of the run (257.6 ms), behind the feed-forward's 42.1% (449.3 ms).

## H3, before the edit: the VPMADDUBSW path, unrolled, |w| hoisted

The avx2 tile takes, per column and group, |x| (VPSIGNB x, x) and w sign(x)
(VPSIGNB w, x), then VPMADDUBSW, VPMADDWD and VPADDD: five ALU ops for 32
MACs, three of them a second VPSIGNB's worth of waste. Turn it around: |w|
(VPABSB) once a weight vector, for every column, and x sign(w) (VPSIGNB x, w)
a column - four ALU ops a column a group, one shared - and unroll the block
of 32 as H2 did (gcc is copying the accumulators here too). Golden Cove
issues VPSIGNB, VPMADDUBSW and VPMADDWD on ports 0 and 1 only, so a group's
ceiling is about 1.6 cycles a column, 20 MAC/cycle, from 15. Predict q8bench
avx2 73 -> ~90 GMAC/s, bench at six threads 2.26 -> ~1.9 s.

**H3, measured.** tests/kcheck: ref's bits, avx2 and avx2-vnni. q8bench, one
P-core, avx2: 73.3 -> 89.9 GMAC/s on 4096x1024 (90.2, 90.0 on the other two),
the decoder's T = 1 product 47.6 -> 52.1: 18.6 MAC/cycle against the ~20
that ports 0 and 1 allow it. As predicted. Kept. The engine:

| build | threads | min | median | max | was (median) |
|---|---|---|---|---|---|
| avx2 | 1 | 6.544 | 6.640 | 6.711 | 7.709 |
| avx2 | 6 | 1.994 | 1.997 | 2.002 | 2.260 |

(A side effect worth writing down: with |w| as the unsigned factor a weight
of -128 multiplies correctly, VPABSB leaving it 0x80 = 128 unsigned, where
the old form's VPSIGNB w, x wrapped it. No converted file has one.)

## Threads (avx2-vnni after H3)

`pkito bench -t N`, best of 5, one run each; scheduler's placement unless
taskset is given:

| threads | where | s |
|---|---|---|
| 4 | | 1.241 |
| 5 | | 1.188 |
| **6** | | **1.062** |
| 6 | taskset one a P-core (0,2,4,6,8,10) | 1.062 |
| 7 | | 1.206 |
| 8 | | 1.192 |
| 10 | | 1.223 |
| 12 | | 1.183 |
| 12 | taskset P-cores and their siblings (0-11) | 1.110 |
| 14 | (the default: pk_os_cores(), every physical core) | 1.194 |
| 14 | taskset one a core, P and E | 1.191 |
| 20 | | 1.199 |
| 8 | taskset E-cores only (12-19) | 2.010 |

Six threads, one a P-core, is the best by 11-15%, and the default (14) is
12% slower than it. A seventh thread lands on an E-core or a sibling: it
takes power from the six (the package is at 45 W already), and an E-core's
item is two to three times a P-core's, so whenever it holds the last item of
a job - six hundred jobs a run - five P-cores wait for it. The M4 Pro said
the same with less to lose (`neon-dp`, 8 performance threads 0.512 s, 12
with the efficiency cores 0.523 s). **The default should be the physical
cores of the fastest kind**: on Linux the cores whose cpufreq maximum is
within 10% of the fastest's (6 here; every core on a Pi, whose cores are one
kind), on macOS hw.perflevel0.physicalcpu. H4.

**Slice size** (core/encoder.c's pk_gemm, rows a pool item), six threads,
three runs each: 32 rows 1.087-1.089 s, **64 rows 1.057-1.059**, 128 rows
1.146-1.148, 256 rows 1.166-1.169. Each call copies its column blocks'
activations (the XOR, H1) and takes its panels' weight sums - 3-5% of a
64-row slice in q8bench's sliced mode - so smaller is dearer; larger leaves
too few items for six threads (1024 rows in 128s is 8 items: a second round
of two). 64 stays.

## H4: the default thread count, the fastest kind of core (core/os.c)

pk_os_cores() counts the physical cores whose cpufreq maximum is within 10%
of the fastest's (sysfs; every core where cpufreq says nothing), and on macOS
hw.perflevel0.physicalcpu before hw.physicalcpu. Here it now answers 6 (the
P-cores at 4.9 and 5.0 GHz; the E-cores' 3.8 is 76% of the fastest), the M4
Pro 8 (its performance cores; 12 before). `pkito bench` with no -t, three
runs: 1.057, 1.062, 1.056 s, against 1.194 at the old default of 14: -11%.
A Pi's cores are one kind, so it keeps all four.
