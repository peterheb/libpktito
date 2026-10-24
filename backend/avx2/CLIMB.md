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

## Baseline (main at b07c50d)

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

## The float attention and what is left (after H4, six threads)

Profile: feed-forward 42.2%, float attention 24.2%, q k v pos 13.3%, conv
pw1/pw2 7.9%, attention out 2.9%, subsampling 2.3%, mel 2.1% (serial),
decoder 1.7% (serial), quantize 1.6%, the rest under 1%. turbostat: 45.08 W,
busy 3042 MHz - the clock did not rise; the cap binds as before and the work
a cycle is what moved. The attention's float products alone (a throwaway
f32 bench, one P-core): scores 752 x 128 x 16 66.7 GMAC/s, positions 65.8,
context 128 x 750 x 16 67.9 - 13.8 MAC/cycle against the 16 that two FMA
ports allow, 87%. The int8 tile is at 47 against the ~51 that its per-block
float work allows (8 VPDPBUSD, a VCVTDQ2PS and a VFMADD an accumulator a
block on ports 0 and 1), 92%. Both are near what their arithmetic allows;
past here the gain is in how the work is spread.

## H5, before the edit: split each product by columns as well as rows

pk_gemm deals a product out in 64-row slices: 16 of them for every
1024-row matrix (q, k, v, o, pw2, both ff downs, pos - 60% of the int8 work
at T 750), which six threads take in 2.67 rounds, the last a third idle:
~11% of those jobs, ~5% of the run. Cut each slice's columns into chunks too,
enough that a job has at least four items a thread (and a chunk at least 48
columns): 16 x 3 = 48 items, 8 rounds, none short. Each call then copies
(H1's XOR) only its own columns, so the copying does not grow; the panels'
weight sums are taken once a call instead of once a slice, about 0.4% more.
Bit-exact by construction: every output is still one call's, with ref's
arithmetic. On four or eight threads 16 slices were already whole rounds,
so they should not move. Predict six threads 1.057 -> ~1.01 s.

**H5, measured.** The rule as written ("four items a thread") would have
chosen 32 items, which six threads also take with a third of the last round
idle; what matters is how full the last round is. So pk_gemm picks the
number of column chunks, 1 to 8 (chunks of whole 12-column tiles, at least
48 columns), whose last round is least idle: 16 slices x 3 chunks on six
threads, and 1 chunk wherever the slices are whole rounds already (four and
eight threads at 1024 rows, any count at 4096). `--json` output (text and
every token's time) identical at 1 and 6 threads and to main's build.

| build | threads | min | median | max | was (median) |
|---|---|---|---|---|---|
| avx2-vnni | 6 | 1.015 | 1.017 | 1.021 | 1.057 |
| avx2 | 6 | 1.890 | 1.891 | 1.937 | 1.997 |

and avx2-vnni at 1, 4 and 8 threads unchanged (3.39-3.41, 1.244-1.247,
1.200-1.203 s); at 5, 1.188 -> 1.113-1.117. Kept.

## H6 (rejected): the weights offset at every T

Would drop the activations' copy (H1) for an XOR of the two weight vectors a
group, which Golden Cove can issue on port 5, idle in this tile, and the
columns' sums. q8bench, one P-core, sliced as pk_gemm calls it: 205.1,
191.3, 204.7 GMAC/s against the copy's 221.2, 212.1, 221.3; whole matrices
223.2, 207.7, 222.4 against 229.5, 213.5, 229.3. The column sums' horizontal
adds cost more than the copy does. Rejected; the copy stays for T >= 16.

## (rejected on paper) VPMADDWD on sign-extended int16 for avx2

VPMADDWD multiplies 16-bit pairs with no saturation, so the activations
could be widened once a call and the weights a group (VPMOVSXBW, port 5) and
the sign trick dropped. Per group of four inputs and four columns that is 8
VPMADDWD (ports 0, 1), 2 VPMOVSXBW (5) and 8 VPADDD (0, 1, 5): 18 ALU uops on
three ports, 6 cycles, the same ~21 MAC/cycle as the VPMADDUBSW form's 17
uops (12 of them on ports 0 and 1). No gain to measure; not written.

## Where this climb ends (H1-H5 kept)

tests/kcheck: ref's bits on avx2 and avx2-vnni. `--json` output (text and
token times, test.wav whole) and the default 30 s chunked text are main's,
byte for byte, from both builds. pkito bench, test.wav whole, 5 runs:

| build | threads | before (median) | after (min / median / max) |
|---|---|---|---|
| avx2-vnni | 1 | 5.512 | 3.375 / 3.420 / 3.470 |
| avx2-vnni | 6 | 1.656 | 1.014 / 1.018 / 1.021 |
| avx2-vnni | default | 1.194 (14 threads) | 1.018 (6: H4) |
| avx2 | 1 | 7.709 | 6.762 / 6.766 / 6.798 |
| avx2 | 6 | 2.260 | 1.891 / 1.896 / 1.938 |

The int8 product alone (q8bench, one P-core at ~4.85 GHz, encoder shapes):
avx2-vnni 113 -> 229 GMAC/s (23 -> 47 MAC/cycle, 92% of what its per-block
float work allows), avx2 73 -> 90 (15 -> 18.6, ~93% of its ports' 20). The
profile at six threads, avx2-vnni: feed-forward 42.3%, float attention 25.2%
(its products at 87% of the FMA ports' rate), q k v pos 12.6%, conv 7.5%,
attention out 2.8%, subsampling 2.4%, mel 2.2% and decoder 1.8% (both on one
thread), quantize 1.7%.

Left for core/, measured but not changed here: the mel (22 ms of 1018 at six
threads) runs on the calling thread while five wait, and could be split by
frames; the attention's repack is one item a head, 8 items for 6 threads.
Each would be 1-2%.

## 2026-10-20 - threads again, on today's tree; and a Core 3 304 on Windows

The finding above, that the P-cores alone beat every core, no longer
holds. `./pkito bench --max-chunk 60` (the minute whole), tree 78c27e1,
avx2-vnni, the same 12900HK, two passes over the counts in turn (each
count's best of five):

| threads | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 |
|---|---|---|---|---|---|---|---|---|
| pass 1 | 0.965 | 1.147 | 0.925 | 0.958 | 0.910 | 1.119 | 1.103 | 0.926 |
| pass 2 | 0.959 | 1.140 | 0.923 | 0.958 | 0.906 | 1.115 | 1.101 | 0.913 |

(10 and 12 are from three passes on b05af83, whose other counts were
within 10 ms of these.) Every core, 14, is the fastest; 8, 16 and 18 are the
slowest by 20%, the same in every pass. A new machine makes the case
plainer: an Intel Core 3 304 (Wildcat Lake: one Cougar Cove P-core, four
Darkmont E-cores, gcc's -march=native is pantherlake, AVX-VNNI and
AVX-VNNI-INT8), Windows 11, built with MSYS2's UCRT64 gcc 16.2:

| threads | 1 (P) | 2 | 3 | 4 | 5 (every core) |
|---|---|---|---|---|---|
| best of 5 | 5.019, 5.193 | 3.159 | 2.090 | 1.801, 1.877 | 1.387, 1.409 |

The rule that chose the fast kind alone gave the 304 one thread, at a
quarter of its speed. core/os.c now takes one thread a physical core of
every kind: the 12900HK's default 14 runs the minute in 0.909-0.921 s
whole and 0.828-0.829 s in 30 s chunks (from 0.960 and 0.861 on 6), the
304's default 5 in 1.392-1.405 s and 1.290-1.312 s. check-c passes on
both, Windows included (every kernel ref's bits, both transcripts byte for
byte).

Not adopted, measured on the 304: workers pinned one a core (pinfer's
pool, core/pool.c and core/SCHED.md there, pins worker k to the k-th
fastest core and deals each a share of what is left by its measured
speed). A scratch build here pinned the calling thread to the P-core (cpu
0, EfficiencyClass 1) and worker k to cpu k, the E-cores, with
SetThreadAffinityMask; three interleaved passes, five threads: the minute
whole 1.408, 1.394, 1.393 s pinned against 1.419, 1.438, 1.401 unpinned;
30 s chunks 1.289, 1.436, 1.296 against 1.299, 1.306, 1.303. About 1%,
inside the spread, so the pool keeps its plain counter.

## 2026-10-20 - a Ryzen 9 9950X: the EVEX VPDPBUSD runs, and both encodings agree

Measured by another session on Peter's Ryzen 9 9950X (Zen 5, 16 cores, 32
threads, AVX-VNNI and AVX512-VNNI both), Linux under WSL2, gcc 15.2, tree
719ff08, with no other load (1-minute load under 2). WSL2 shows the real
topology, so the default is 16 threads. check-c passed on all three builds:
every kernel ref's bits (kcheck, 1525 cases), both transcripts of test.wav
byte for byte. The second build is AVX512-VNNI's EVEX VPDPBUSD at 256
bits (`BFLAGS="-mavx2 -mfma -mavx512vnni -mavx512vl"`; every one of
kernels.o's 219 VPDPBUSDs 62-prefixed), the encoding that had not run
before.

| build | default (16) whole | -t 1 | -t 8 | -t 32 | 30 s chunks | q8bench, 1 thread (ff up / down / att q) |
|---|---|---|---|---|---|---|
| avx2-vnni, VEX (the Makefile's choice) | 0.328 s | 2.656 | 0.380 | 0.355 | 0.310 | 257.4 / 252.0 / 260.5 GMAC/s |
| avx2-vnni, EVEX | 0.332 | 2.628 | 0.379 | 0.319 | 0.303 | 260.0 / 256.3 / 265.2 |
| avx2 | 0.615 | 4.817 | 0.681 | 0.551 | 0.604 | 117.9 / 118.1 / 119.3 |

The two encodings are level within 1-2%, inside the spread (a recheck of
the VEX build's default gave 0.334 and 0.333; -t 32 wanders 0.350-0.374).
VNNI over avx2 is 1.85x end to end and 2.2x in the product alone.

The same machine under native Windows, cross-built with Ubuntu's mingw-w64
gcc 13 (msvcrt, winpthreads, static): check-c passed with every line the
same; 16 threads 0.298-0.303 s whole and 0.294 s in 30 s chunks, about 10%
faster than under WSL2. That toolchain's msvcrt printf did not know %zu (46
-Wformat warnings), so the Makefile's Windows branch now defines
__USE_MINGW_ANSI_STDIO; under MSYS2's UCRT64 the output is the same bytes
with it (check-c on the Core 3 304).

## 2026-10-20 - one binary for the family: the backend picked at run time

A build now holds avx2-vnni, avx512-vnni and avx2 (and ref), each compiled
with its own instruction set, the rest of the tree with x86-64's baseline,
and backend/kernels.c runs the first the CPU can (CLAUDE.md, 4). The core
lost -mavx2 -mfma and nothing measured: the 12900HK picks avx2-vnni and
runs the minute whole in 0.909-0.912 s and in 30 s chunks in 0.829-0.830
(0.909-0.921 and 0.828-0.829 before), and `PKITO_KERNELS=avx2` 1.676 s
(1.682 as a build of its own); the Core 3 304, Windows, 1.393-1.402 and
1.307-1.313 (1.392-1.405, 1.290-1.312). kcheck tests every backend the CPU
can run and skips the rest with a line: avx512-vnni on both.

## 2026-10-20 - 74b5f78's portable builds on a Zen 4 and a Zen 5

Measured by another session on Peter's machines, idle, the v2 model,
best of five; the binaries built here (sha256 checked): pkito.exe on the
Core 3 304 with MSYS2's UCRT64 gcc 16.2, the Linux one on the 12900HK with
gcc 15.2.

A Ryzen 7 PRO 8840HS (Zen 4: AVX512-VNNI, no AVX-VNNI), Windows 11, 8
threads, the exe with no PKITO_KERNELS: it runs avx512-vnni, the minute
whole 0.607 s and in 30 s chunks 0.537 s, and writes test.wav's v3
transcript byte for byte. `PKITO_KERNELS=avx2-vnni` is refused, exit 1;
`PKITO_KERNELS=avx2` 1.196 and 1.120 s. The AVX-VNNI build of 719ff08 died
there on its first VPDPBUSD.

A Ryzen 9 9950X (Zen 5, both encodings), 16 threads, whole and in chunks:

| | avx2-vnni (picked) | avx512-vnni | avx2 |
|---|---|---|---|
| Windows, the exe | 0.283, 0.235 s | 0.282, 0.241 | 0.524, 0.498 |
| WSL2, the Linux binary | 0.323, 0.296 | 0.322, 0.283 | 0.595, 0.552 |

Both write the v3 transcript byte for byte; built from source there,
check-c passes with all three backends in kcheck. The two encodings stay
level. The UCRT64 gcc 16.2 exe beats 719ff08's cross-built gcc 13 one by
about 20% natively (0.235 against 0.291 s in chunks).

## 2026-10-20 - pkito is now pktito

The project took its intended name, pktito (a little parakeet): the
binary, library, header, converter and PKTITO_KERNELS. The records above
keep the name they were written under.

## 2026-10-21 - a non-finite quantizer block is zero

The check neon-dp/CLIMB.md describes, in this backend's k_quantize: an
unordered compare and an OR beside each max of the block's maximum, and a
zero block when either finds a NaN or an infinity. kcheck's non-finite
case and the rest, 1,663 cases, are ref's bits on avx2 and avx2-vnni (the
12900HK).

On the 12900HK, avx2-vnni, 14 threads, test.wav's minute whole, best of
five, main and the check in five interleaved rounds: 0.829 to 0.835 s
before, 0.831 to 0.834 s with it, inside the spread. A PROFILE=1 build
puts the quantize stage at 43.8 to 44.0 ms a run before and 44.8 to 44.9 ms
with the check, three rounds each: about 1 ms, 0.1% of the run.

## 2026-10-22 - the resampler as a kernel, eight blocks a vector (hypothesis)

core/resample.c's dot products ran in core, at x86's baseline: since
ec448c8 each fused multiply-add of them is pk_fmaf_soft, one chain an
output, about 34 taps a phase from 44.1 kHz and 38 from 48. `pktito bench`
on test.wav resampled, its resample line (new, best of five), three
rounds on the 12900HK: 76.6 to 76.7 ms at 44.1 kHz, 85.5 to 85.6 ms at 48
(glibc's fmaf, before ec448c8, about 36 ms at 44.1).

As a kernel, pk_k_resample, a vector holds eight consecutive blocks at
the same phase: the tap broadcast to the eight, and their inputs, orig
floats apart, one VGATHERDPS. Each lane's FMAs are the scalar chain's,
so the bits are ref's. The gather, about one in five cycles on Golden
Cove, bounds a step of eight taps: 34 taps for each of 120,000 groups of
eight outputs a minute, about 20 M cycles, 4 to 5 ms at 44.1 kHz. A
refill's blocks that make no eight run one at a time on the FMA
instruction, and the history buffer grows to about 32 blocks a refill so
that they are few. Kept if kcheck's resample cases are ref's bits and
the line falls past its spread.

Result, kept: kcheck's 114 resample cases (1,777 in all) are ref's bits on
avx2 and avx2-vnni, and check-c passes, on the 12900HK. The resample line,
three rounds: 4.4 to 4.5 ms at 44.1 kHz (76.6 to 76.7 before), 4.1 to 4.3
ms at 48 (85.5 to 85.6), 17 and 20 times, inside the gather's 4 to 5 ms;
the transcription's 0.83 s as it was. On the M4, the VAD's probabilities
and both transcripts of test.wav at 44.1 and 48 kHz are the bytes of the
tree before, under neon-dp and under ref.

## 2026-10-22 - the E-cores' share (a measurement)

The 12900HK as above, Ubuntu 26.04, kernel 7.0.0-38, GCC 15.2. Tree
57d6ccb. `./pktito bench` on test.wav in its default 30 s chunks (three),
best of five, load 0.00; backend avx2-vnni.

| threads | on | best | real time |
|---|---|---|---|
| 14, the default (a core of every kind each) | any | 0.836-0.839 s | 71.5-71.8x |
| 20 | any | 0.912 s | 65.8x |
| 12 | any | 0.888 s | 67.5x |
| 8 | any | 1.050 s | 57.2x |
| 6 | any | 0.867 s | 69.2x |
| 12 | the P-cores, CPUs 0-11 (taskset) | 0.860 s | 69.8x |
| 6 | the P-cores | 0.864 s | 69.5x |
| 8 | the E-cores, CPUs 12-19 | 1.772 s | 33.9x |

The default is still the fastest here, but by 3% over the P-cores alone,
where the eight E-cores alone do half their work (33.9x against 69.5x): a
share in proportion to each core's speed would make up to 103x. The GB10
loses far more to the same dealing (backend/neon-dp/CLIMB.md, the same
date).
