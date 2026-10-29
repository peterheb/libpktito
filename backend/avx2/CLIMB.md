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

## 2026-10-24 - avx512-vnni's int8 product in zmm (hypothesis)

The avx512-vnni build is avx2-vnni with VPDPBUSD in its EVEX encoding at
256 bits: the tile is AVX2's, 16 rows in two ymm by up to three columns,
sized for 16 registers (15 in use). AVX-512 has twice the width and twice
the registers, and nothing uses either.

A panel of 16 rows is exactly one zmm of int32 sums, and a group of four
inputs for those rows exactly the 64 weight bytes one zmm load takes. So
in zmm a group is one load and, for each column, one VPDPBUSD with the
column's four activation bytes as an embedded {1to16} broadcast, where
the ymm tile takes two loads and two VPDPBUSDs a column. With 32
registers the tile can hold NC columns at two zmm each (the block's
integer sums and the float chain) beside the weight vector: NC up to 8,
so the weights a load brings are used by up to eight columns, not three.

What it should do, by core:
- Zen 5 (the 9950X here; c8a's EPYC 9R45) executes 512-bit VPDPBUSD at
  full width. The ymm tile ran 257-265 GMAC/s on one 9950X core (above,
  2026-10-20), about 46 MAC a cycle against 64 for two ymm VPDPBUSD a
  cycle; zmm doubles the ceiling. The int8 product is about three
  quarters of a run (the 12900HK profile above: feed-forward, q k v pos,
  conv pw, attention out), so q8bench up to 1.7x and `pktito bench` 20
  to 35% faster.
- Zen 4 (c7a's EPYC 9R14, the 8840HS) splits a zmm operation into two
  256-bit halves: the same MACs a cycle, fewer instructions and loads.
  Level, or a few percent.
- Sapphire and Granite Rapids (c7i, c8i): 512-bit integer multiplies on
  two ports where they have the second FMA unit, with a lower clock under
  heavy 512-bit work. Unknown; measured, not guessed.
- Ice Lake (c6i): the largest 512-bit clock penalty of these. Unknown.

Bits: every lane's int32 sums are exact integers over the same blocks,
and each block goes into its lane's float chain by the same fused
multiply-add, scale, bias and add as ref's: kcheck must be ref's bits.
The avx512-vnni backend's needs gain avx512f (CPUID 7.0 EBX bit 16 with
the ZMM state; core/os.c answers 1 for a name it doesn't know, so it has
to learn this one).

Kept if kcheck and check-c pass and q8bench and `pktito bench` move past
their spread on the 9950X. If they do, it is measured on c8a, c7a, c8i,
c7i and c6i (AliceBench, the c8a.xlarge-class sizes), and avx512-vnni goes
ahead of avx2-vnni in backend/kernels.c's order only if no CPU with both
runs slower on it.

## 2026-10-24 - the int8 product on AMX (hypothesis)

Sapphire Rapids and Granite Rapids (c7i, c8i) have AMX: eight tile
registers of up to 16 rows of 64 bytes, and TDPBSSD, which multiplies a
tile of signed bytes by another into a tile of exact int32 sums. The
AliceBench sweep (2026-10-23) put these Xeons at about 14x a core against
Zen 5's 19x, on VPDPBUSD at about 60 MAC a cycle a core.

The file's layout is already AMX's. TDPBSSD's B operand is K/4 rows of 64
bytes, each four inputs for 16 outputs: a panel's group. So a block of 32
inputs of one panel, 512 contiguous bytes, is a B tile of 8 rows as it
lies in the file. A is 16 columns' 32 bytes of the block, rows n apart in
the activations, and C is 16 columns by the panel's 16 rows. Signed by
signed, so no +128 offset and no correction.

Bits: each block's C is its exact integer sums. Stored to L1 and read back
a column at a time, each goes into that column's float chain by the same
convert and fused multiply-add as the zmm tile's: ref's chain, lane for
lane. Columns past the last whole 16, and products of fewer than 16
columns (the decoder), take the zmm tile.

What it should do: a block of one panel by 16 columns is 8,192 MACs. If
TDPBSSD at K = 32 takes 16 cycles (a row of A a cycle), that is 512 MAC a
cycle against the zmm tile's 60. The float side is then the limit: 16
loads, 16 converts and 16 FMAs a block, about 16 cycles on two ports,
level with the tile. So the product 3 to 4x on one core, and `pktito
bench` 50 to 100% faster on c7i and c8i, if the store, reload and
convert overlap the next block's TDPBSSD (two C tiles, alternating).

Built as a fourth variant of this file, backend/amx-int8: the avx512-vnni
build with -mamx-tile -mamx-int8, needing avx512f, avx512vnni, avx512vl,
amx_tile and amx_int8. core/os.c learns the last two (CPUID 7.0 EDX bits
24 and 25, XCR0 bits 17 and 18) and, on Linux, asks once a process for
the tile data (ARCH_REQ_XCOMP_PERM): refused, the answer is no and
avx512-vnni runs. Each thread loads the tile configuration once.

Kept if kcheck and check-c are ref's bits on a c7i and the bench moves past
its spread there and on a c8i; it goes first in the order only where it
wins.

## 2026-10-24 - avx512-vnni's int8 product in zmm (result: kept)

The zmm tile as the hypothesis above has it: a panel one register, a
group one load and one VPDPBUSD a column, up to PK_Q8_NC columns (8) a
tile. kcheck, 1777 cases, ref's bits on avx2, avx2-vnni and avx512-vnni,
and check-c passes, on the 9950X (gcc 15.2); test.wav's v2 and v3 through
avx512-vnni are ref's, there and natively on Windows (MSYS2 gcc 16.2).

Measured on EC2, us-east-1, each instance fresh, Ubuntu 24.04, binaries
built here: main (a331da1) against this tree, interleaved, three rounds
of `pktito bench` on test.wav (best of five each, the default threads),
then q8bench on one thread. Every transcript of every build and backend,
38 files, ref's bytes. avx512-vnni's medians:

| instance | CPU | threads | main | zmm | | zmm over avx2-vnni |
|---|---|---|---|---|---|---|
| c8a.xlarge | EPYC 9R45, Zen 5 | 4 | 82.5x | 119.1x | +44% | +47% |
| c8a.4xlarge | EPYC 9R45, Zen 5 | 16 | 226.1x | 287.8x | +27% | +29% |
| c7i.xlarge | Xeon 8488C, Sapphire Rapids | 2 | 25.7x | 32.8x | +28% | +28% |
| c8i.xlarge | Xeon 6975P-C, Granite Rapids | 2 | 29.6x | 35.4x | +20% | +20% |
| c8i.4xlarge | Xeon 6975P-C, Granite Rapids | 8 | 108.6x | 126.2x | +16% | +18% |
| c7a.xlarge | EPYC 9R14, Zen 4 | 4 | 63.6x | 67.6x | +6% | (none) |
| c6i.xlarge | Xeon 8375C, Ice Lake | 2 | 24.1x | 23.6x | -2% | (none) |

q8bench, feed-forward up, one thread, GMAC/s, main and zmm: Zen 5 224 and
394, Sapphire Rapids 152 and 212, Granite Rapids 167 and 227, Zen 4 178
and 195, Ice Lake 137 and 131. The ranges: within 1% but on the 16-thread
c8a (main 221-232, zmm 278-290) and c8i.4xlarge (zmm 121-131).

The tile's width, swept on four of them (the same method; PK_Q8_NC 4, 6, 8
and 12, medians):

| | main | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|
| c8a.xlarge | 82.1 | 116.1 | 116.8 | 118.4 | 108.5 |
| c7a.xlarge | 63.6 | 67.3 | 67.0 | 67.9 | 55.3 |
| c8i.xlarge | 29.7 | 34.3 | 35.1 | 35.2 | 33.6 |
| c6i.xlarge | 23.4 | 21.5 | 22.0 | 22.2 | 19.7 |

Eight is best or level everywhere; twelve spills. Ice Lake is the one
loss, and not the tile's: its product is faster in zmm at eight columns
on one thread (164 GMAC/s against main's 137), and the whole run on two
threads slower by 2 to 6% (this sweep's spread there is the widest, 22.1
to 27.8), which is its clock under 512-bit work slowing the rest of the
run. No narrower tile removes it; a ymm product chosen for Ice Lake by
model would be the model list CLAUDE.md, 4, rules out. Kept, with Ice
Lake's 2 to 6% as its cost; Ice Lake has no other VNNI to fall back to.

## 2026-10-24 - the int8 product on AMX (result: kept, first in the order)

As the hypothesis above has it, two C tiles in turn: block b's TDPBSSD
runs while block b-1's sums go to L1 and into the float chains, 16 columns
at a time with every panel of pk_gemm's slice under them. kcheck, 1777
cases, ref's bits on amx-int8, and check-c passes, on a c7i (Xeon 8488C,
gcc 13.3 on Ubuntu 24.04); every transcript below is ref's bytes, through
either backend.

The loop's shape, on a c7i.large (one core, two threads), `pktito bench`
on test.wav: avx512-vnni's zmm 19.1 to 19.7x; two C tiles 27.6 to 27.7x;
four C tiles in turn, to wait out more of TDPBSSD's latency, 27.1 to
27.3x, the two interleaved in one sitting (an earlier sitting had four
tiles at 30.4 to 30.9x and two at 28.3x, which the interleaved rounds did
not reproduce; the box was slower by about 3% across the board by then);
panels outer, 26.6 to 27.5x; the columns cut into blocks of 64, 28.0 to
28.9x. Two tiles, columns outer: the simplest, and level or ahead.

Measured on EC2, us-east-1, each instance fresh, Ubuntu 24.04, one binary
built here holding both backends: three interleaved rounds of `pktito
bench` on test.wav (best of five each, the default threads), then
q8bench on one thread, the 1024 x 1024 attention product at T 750, whole
and in pk_gemm's 64-row slices:

| instance | CPU | threads | avx512-vnni | amx-int8 | | q8bench, whole | in slices |
|---|---|---|---|---|---|---|---|
| c7i.xlarge | Xeon 8488C, Sapphire Rapids | 2 | 37.7x | 46.8x | +24% | 250 to 429 | 236 to 343 |
| c8i.xlarge | Xeon 6975P-C, Granite Rapids | 2 | 35.2x | 44.0x | +25% | 223 to 351 | 207 to 296 |
| c8i.4xlarge | Xeon 6975P-C, Granite Rapids | 8 | 135.6x | 168.9x | +25% | 253 to 448 | 226 to 370 |

(medians; every round within 1% of its median; q8bench in GMAC/s.)

Short of the hypothesis: the product 1.5 to 1.7x, not 3 to 4x, and the
run 25%, not 50 to 100%. 429 GMAC/s at about 3 GHz is some 57 cycles a
block of 8,192 MACs, where the hypothesis counted 16 for TDPBSSD and 16
for the float side overlapping it. Where the other 25 go - A's 16 rows
loaded n bytes apart, C's kilobyte stored and read back, the chain's
dependence through f - is the next climb's question, not answered here.
Kept: 24 to 25% on both Xeons with AMX, past a spread under 1%, and the
order puts amx-int8 first. Not measured on Windows, which builds it
(MSYS2's gcc has -mamx-int8) and has no such CPU here: amx_permitted
asks only Linux, and on Windows the tiles' XCR0 bits are the whole check,
which a Windows Xeon should confirm before the order is trusted there.

## 2026-10-24 - the zmm product on the 9950X (a measurement)

The Ryzen 9 9950X under Windows 11, MSYS2's gcc 16.2, main (a331da1)
against f0232d2, three interleaved rounds of `pktito bench` on test.wav,
16 threads, run twice (the second after another job's build had shared
the first); medians of the second, seconds whole and in 30 s chunks:

| | main | f0232d2 |
|---|---|---|
| avx512-vnni | 0.288, 0.274 | 0.228, 0.217 (+26%) |
| avx2-vnni | 0.287, 0.283 | 0.292, 0.274 |
| avx2 | 0.546, 0.541 | 0.534, 0.529 |

The first run agreed within 3%. avx2-vnni and avx2 are the same code in
both and move within their spread (200 to 224x across rounds); the zmm
product is the one change, ref's bits from every build and backend.

## 2026-10-24 - amx-int8 on Windows (a measurement)

The AMX result above left Windows unmeasured: amx_permitted asks only
Linux, and elsewhere XCR0's tile bits decide. EC2 c7i.large (Xeon 8488C,
one core, two threads), fresh, Windows Server 2025 (10.0.26100) and 2022
(10.0.20348), the engine at ae34ec1 built on the 9950X with MSYS2's gcc
16.2 (UCRT and KERNEL32 only), run from user data: kcheck, 1770 cases,
ref's bits on amx-int8, avx512-vnni, avx2-vnni and avx2, on both; pktito
picks amx-int8 itself, and test.wav's v2 through it, forced to amx-int8
and forced to avx512-vnni, is ref's bytes. Windows hands a thread the
tile data on its first use without being asked, as XCR0 says it will.

`pktito bench` on test.wav, one thread (Windows counts the one core), two
interleaved rounds: Server 2025, amx-int8 21.1x and 21.1x against
avx512-vnni's 17.7x and 17.8x; Server 2022, 21.2x and 21.2x against
17.5x and 17.6x: 19 to 21% on both. q8bench, three repetitions in 64-row
slices, Server 2022: feed-forward up 212 to 275 GMAC/s, down 191 to 265,
attention q 214 to 268. On Server 2025 the same run put ff up at 159 to
273 and the other two level (155 to 162, 181 to 176); not repeated, and
its bench, the longer measure, moved as 2022's did.

## 2026-10-24 - the program is pktito-c (a note)

The command-line program the Makefile builds, `./pktito` in the rows
above, is `./pktito-c` from here on, so that it is not taken for the
app, pktito; the same program, the same output. The rows above keep the
name they were measured under.

## 2026-10-24 - AMX: where a block's 57 cycles go, and a pipelined loop (hypothesis)

The amx-int8 result above left 25 of some 57 cycles a block unexplained.
Measured on the c7i.large (Xeon 8488C), with standalone loops of one
instruction kind under `perf stat` (cycles, exe.amx_busy,
uops_dispatched.port_*), all data in L1:

- TDPBSSD takes 16 cycles at K = 64, 32 or 16 alike (exe.amx_busy 16.0 an
  instruction), and 16 back to back on one C tile: a block of 32 inputs
  is half the unit's peak, 512 MAC a cycle, and the dependence through C
  costs nothing. At M = 8 or 12 rows it takes 8 or 12.
- TILELOADD of A (16 rows of 32 bytes, n apart) takes 9.5 cycles alone,
  of B (8 rows of 64) 6.8, the two together 20 a block.
- TILESTORED of C takes 17 cycles and 43 uops: 16 store-address, 16
  store-data and 19 on port 5.
- Beside a TDPBSSD, 32 zmm FMAs or converts take 32 cycles, not 16: the
  unit leaves one 512-bit FP port. 32 ymm FMAs and 32 zmm integer adds take
  16, as without it. In the kernel's loop every convert and FMA is
  dispatched to port 0 (32 a block); port 5 is the tile store's.

So a block's convert and FMA, 32 on port 0, set a floor near 36 cycles
(32 and the tile instructions' 4 there), and the loop as it stands runs
57: it stores each block's C right after its TDPBSSD, so the store waits
out the product and the next block's tile loads wait behind the store
(a TDPBSSD and a store of its own C, 29 cycles a pair; of the C before it,
19). The block loop alone, data in L1, one panel: 57.5 cycles a block;
with B from L2 (four panels in turn) 61.8; the feed-forward's down shape
(n 4096) 70.0.

The change: a software pipeline over the blocks. Step b loads block b's A
and B into one of two tile pairs (0 and 1, 4 and 5), multiplies into one
of two C tiles, stores block b-1's C (its product issued a step before),
and runs block b-3's chains from a ring of four buffers. The same loop in
the standalone test: 45.7, 48.9 and 62.0 cycles a block (chains two blocks
behind: 49.0, 53.3, 65.5; four: 45.3, 48.8, 66.4). Predicted: q8bench in
slices of 64 on the c7i.large from 391 to about 480 GMAC/s on the
attention's shape and from 344 to about 380 on the feed-forward's down;
`pktito bench`, whose int8 products are 64% of a run there (PROFILE=1,
26.5x on one thread), 10 to 12% faster. The bits are untouched: each
block's sums go into the chains in block order, as now.

## 2026-10-24 - AMX's pipelined loop (result: kept)

The pipeline as the hypothesis above has it, chains three blocks behind,
with two things the first build showed. gcc 15 keeps an array of the 16
chains on the stack, a load and a store more a column a block; it did so
with main's loop too, which gcc 13 keeps in registers: so the chains are
16 named variables. And with nb a variable the 16 scales' offsets m nb
take as many registers, some of them moved through xmm; so the steady
state is four steps a turn (every tile and buffer a constant), and nb a
constant for the encoder's 32 and 128, a variable for other multiples of
4. Blocks that are no multiple of 4 (none of the model's) take the zmm
tile.

kcheck, 1777 cases, ref's bits on amx-int8, avx512-vnni, avx2-vnni and
avx2 and check-c passes, on the c7i.large (gcc 13.3) and, without AMX, on
the 9950X (gcc 15.2); test.wav's v2 and v3 through amx-int8 are ref's
bytes from both compilers' builds.

On the c7i.large, one sitting, main (ae34ec1) and this tree each built
there with gcc 13.3 and here with gcc 15.2, three interleaved rounds of
q8bench in slices of 64 (GMAC/s: feed-forward up, down, attention) and
then of `pktito bench` on test.wav (one thread, best of five):

| | q8bench, medians | bench |
|---|---|---|
| main, gcc 13 | 363, 348, 381 | 24.8, 26.0, 25.8x |
| this, gcc 13 | 443, 420, 459 (+22, +21, +20%) | 28.9, 28.9, 28.5x (+12%) |
| main, gcc 15 | 306, 297, 299 | 22.8, 22.9, 22.7x |
| this, gcc 15 | 471, 429, 458 (+54, +44, +53%) | 28.3, 28.5, 28.6x (+25%) |

The rounds' ranges: main gcc 13 q8bench 359-384, 347-353, 369-387; this
gcc 13 429-449, 405-430, 421-478; the bench ranges as listed. Past the
spread everywhere. The prediction was the attention's shape to about 480
and the down to about 380, and the run 10 to 12%: the attention's shape
fell short (459), the down did better (420), and the run moved 12%. The
EC2 rows of the entry before were gcc 15 builds, the spilling loop.

nb a variable for every shape against the constants, the same sitting,
gcc 13: q8bench 375-414, 376-388, 386-416 against 445-470, 417-424,
448-464; bench 26.7 to 26.9x against 28.5 to 28.8x. Kept, the constants
with it.

The floor is still the port: 32 converts and FMAs a block on port 0
beside the tile store's 19 uops on port 5, near 36 cycles a block on L1
data, where the loop now runs 46 (49 with B from L2, 62 on the down
shape, whose 16 rows of A, 4096 bytes apart, fall in one L1 set).

## 2026-10-24 - AMX: the down shape's A copied off one L1 set (hypothesis)

At n = 4096 (the feed-forward's down product, a third of the encoder's
int8 MACs) a tile's 16 rows of A are 4096 bytes apart: all in one set of
the 12-way L1, which each TILELOADD thrashes, so every row comes from L2
on both blocks that share its line. The pipelined loop alone, data as in
q8bench's slices (four panels of B in turn): 61.9 cycles a block at a
stride of 4096, 53.5 at 4160.

The change: for n = 4096, each group of 16 columns' activations copied
first into a thread's buffer 4160 bytes a row (66.5 KB), and the panels'
tiles loaded from there. The copy is 1024 loads and stores a group, about
4% of the four panels' work in a 64-row slice. Predicted: q8bench's down
shape 9 to 10% faster on the c7i.large, `pktito bench` 2 to 3%. The copy
moves bytes and nothing else; the bits are untouched.

## 2026-10-24 - AMX: the down shape's A copied off one L1 set (result: rejected)

As the hypothesis above has it, the 16 columns' activations copied to a
4160-byte stride before each group's panels, for n = 4096 alone. kcheck
ref's bits on amx-int8 and test.wav v2 and v3 ref's bytes, on the
c7i.large (gcc 13.3). Against the pipelined loop (49acfe4), one sitting,
three interleaved rounds: q8bench in slices, the down shape, 375 to 378
GMAC/s against 439 to 448 (the other two shapes level); `pktito bench`
29.0 to 29.1x against 29.9 to 30.0x. Slower, not 9 to 10% faster.

The standalone loop's A was in L2; in a slice of the real product it is
not. The activations of a minute at n = 4096 are 3 MB, past the 2 MB L2,
so every 64-row slice reads each group's 64 KB from L3 again, and the
copy does that as a phase of its own (about 5,300 cycles a group, 5 a
line), where the tile loads of the uncopied loop take it while the TMUL
and the chains run. Rejected; a copy would have to be pipelined with the
group before it to pay, for at most the 7 to 10% between the down shape
and the up.

## 2026-10-24 - AMX: the next group's A prefetched in the last panel (hypothesis)

q8bench on the c7i.large with the pipelined loop, in slices of 64, 128
and 256 rows and whole: 468-487, 492-514 and 542-585 GMAC/s on the
feed-forward's up shape (the attention's 484-491, 518-525, 555-588). A
64-row slice is four panels a group of 16 columns, and the group's first
panel takes its A from L2, 16 lines a TILELOADD; fitting a + X/128 and
a + X/512 cycles a block to the two ends gives about 43 a block and some
1,600 more a group, the first panel's blocks near twice the others'.
pk_gemm's step at 256 instead of 64 moved `pktito bench` there from 29.5-
29.9x to 30.7-31.6x (amx-int8) and 19.1x to 19.9-20.0x (avx512-vnni), one
thread: a seam change, for Peter, not made here.

Inside the kernel: while a slice's last panel of one group runs, its
steps prefetch the next group's A into L1, the 16 rows' lines of each
pair of blocks (eight lines a block, 16 KB at n = 1024, beside the
panel's 16 KB of B). Predicted: q8bench in slices of 64 most of the way
to the 256-row figure on the n = 1024 shapes, 10 to 15%, and `pktito
bench` 3 to 4%. At n = 4096 a group's A, 64 KB, is past L1; the same
prefetch there should do little.

## 2026-10-24 - AMX: the next group's A prefetched in the last panel (result: rejected)

As the hypothesis above has it: 16 PREFETCHT0s every other block of a
group's last panel, the next group's A at constant offsets. kcheck ref's
bits on amx-int8, test.wav v2 and v3 ref's bytes (c7i.large, gcc 13.3).
Against 49acfe4, one sitting, three interleaved rounds: q8bench in slices
of 64, 452-476, 409-430 and 478-527 GMAC/s against 476-493, 431-445 and
465-479; `pktito bench` 28.8 to 29.1x against 29.5 to 29.8x. Slower.
Rejected. The slice's effect is not the zmm tail's either: at T 736, no
tail, the up shape ran 499-504 GMAC/s in slices of 64 and 554-585 in
slices of 256. What the 256-row slice buys stays open; the step is
pk_gemm's.

## 2026-10-24 - AMX: a short group of columns padded to 16 (hypothesis)

Columns past the last whole 16 take the zmm tile, which at a few columns
is slow: q8bench in slices of 64 on the c7i.large ran 493-522 GMAC/s at T
240, 455-469 at T 250 and 499-532 at T 256 (the up, down and attention
shapes alike). A minute of test.wav is three chunks of about 250 frames:
a tail of 10 columns, 4% of them, near 15% of the product's time; at T 750
the 14 cost 8% (T 736: 499-504, T 750: 462-466 on the up shape).

The change: the last, short group's activations and scales copied into
16-column buffers of the thread (the scales' extra columns zero, so that
their chains stay 0 and the converts and FMAs on them see no denormal),
the group run as any other into a 16-column scratch, and only its real
columns written out. For n up to 4096 (the model's largest; a buffer of
64 KB); past that the zmm tile, as now. Predicted: q8bench at T 250 up
to near T 256's rate on the columns that count, 8% faster, and `pktito
bench` on test.wav 4 to 5% on the c7i.large. The real columns' sums and
chains are the same instructions on the same bytes: the bits are
untouched.

## 2026-10-24 - AMX: a short group of columns padded to 16 (result: kept)

As the hypothesis above has it, for n up to 4096; a call of fewer than 16
columns takes the same path. kcheck ref's bits on all four x86 backends
and check-c, on the c7i.large (gcc 13.3); kcheck and test.wav v2 and v3
from a gcc 15.2 build there too.

Against 49acfe4 on the c7i.large, `pktito bench` on test.wav in two sets
of interleaved rounds, one sitting: 29.9, 29.9, 30.0x against 30.3, 30.6,
30.3x; then 29.5, 29.8, 29.6, 29.8x against 30.6, 30.3, 30.6, 30.5x. Every
round of it above every round without; the medians 29.8x and 30.5x,
+2.3%, about half the 4 to 5% predicted. q8bench in slices is too loose
on this box to place it more closely (T 250: +2 to 3% on the medians,
inside a 5% spread). What the padding costs is the copy of the short
group's activations, every call: a lone group of 10 columns (T 10, the
up shape) ran 173-190 us padded against 157-169 for 16 whole ones, and
211-222 through the zmm tile. Reconfiguring the tiles to the short
group's rows instead would skip the copy, at about 200 cycles a
LDTILECFG (measured) twice a call; not tried.

## 2026-10-24 - the zmm product on Zen 5: where it stands, and its copy (hypothesis)

The Ryzen 9 9950X here (WSL2, gcc 15.2), one thread, the core near 5.55
GHz (a chain of dependent adds against the TSC). Inline-asm loops of the
zmm tile's instructions:

- VPDPBUSD zmm issues two a cycle; eight independent accumulators, the
  tile's NC = 8, leave it latency-bound at 1.75 a cycle (sixteen: 2.04).
- A block of eight groups for eight columns (64 VPDPBUSDs, the broadcasts
  from memory) runs 1.73 a cycle; with the block's float work (eight
  subtracts, converts and FMAs) 1.56. Beside the 64, 16 converts or 16
  subtracts cost nothing and 16 FMAs about 5 cycles: the FMA shares the
  VPDPBUSD pipes, the convert does not. Two sets of accumulators (even
  and odd groups) gain nothing once the float work is in: about 41 cycles
  a block either way, near (64 + 8) / 2 = 36 with the loads beside them.
  The weights-offset form with embedded broadcasts (the T < 16 path) runs
  the same block as fast; an XOR of each broadcast, to offset the
  activations in the loop, 11% slower.

q8bench on avx512-vnni: in pk_gemm's 64-row slices 433-438, 417-425 and
439-445 GMAC/s (up, down, attention), whole 475-483, 450-453, 476-478;
1.56 VPDPBUSDs a cycle would be some 555. A build that skips the XOR copy
of the activations (wrong sums, the same instructions otherwise) ran
460-469, 445-449 and 470-472 in slices: the copy, made per call over the
whole column block before any panel, is 6-7% of the product.

The change: the copy made during the first panel, a tile ahead: the next
tile's columns XORed into the buffer, in zmm, while this tile's VPDPBUSDs
run, so that it overlaps work on other pipes and lands in L1 just before
it is read. Predicted: q8bench in slices most of the way to the no-copy
build, 5 to 6%, and `pktito bench` on the 9950X 3 to 4% (16 threads).
The bytes copied are the same; the bits are untouched.

## 2026-10-24 - amx-int8's pipelined loop and padded group on EC2 (measurement)

main (ae34ec1) against d1bda18 (the pipelined loop and the padded short
group), each one binary built here with gcc 15.2 holding both backends,
on fresh us-east-1 instances (Ubuntu 24.04), one sitting each: three
interleaved rounds of `pktito bench` on test.wav (best of five, the
default two threads), then q8bench once on one thread in pk_gemm's
64-row slices. Every transcript, from either binary through either
backend, is ref's bytes (test.v2.json).

| instance | backend | main | d1bda18 | |
|---|---|---|---|---|
| c7i.xlarge, Xeon 8488C | amx-int8 | 38.4, 38.6, 38.3x | 47.5, 48.1, 48.3x | +25% |
| | avx512-vnni | 33.9, 34.0, 33.9x | 33.7, 34.0, 33.9x | level |
| c8i.xlarge, Xeon 6975P-C | amx-int8 | 43.2, 43.0, 42.8x | 56.3, 56.1, 56.4x | +31% |
| | avx512-vnni | 34.8, 35.0, 34.7x | 34.8, 34.9, 34.7x | level |

q8bench on amx-int8, GMAC/s (up, down, attention):

| instance | T | main | d1bda18 |
|---|---|---|---|
| c7i.xlarge | 750 | 258, 276, 277 | 412, 426, 426 |
| | 250 | 250, 286, 293 | 399, 440, 422 |
| c8i.xlarge | 750 | 288, 285, 293 | 463, 438, 479 |
| | 250 | 282, 288, 302 | 474, 455, 499 |

The product 1.44 to 1.68 times main's; avx512-vnni's, whose code the two
share, within 4% of it on every shape (single runs). Both builds are gcc
15's, under which main's loop spills its chains (the pipelined loop's
entry: 22.8x against gcc 13's 25.8x on the c7i.large); against a gcc 13
build of main the gain would be nearer that entry's +12% and the padded
group's +2.3%. Across sittings the instances differ: this c7i.xlarge ran
main's amx-int8 at 38.4x and avx512-vnni at 33.9x, against 46.8x and 37.7x
for the same loop in the first AMX entry's sitting; the c8i.xlarge agrees
with its row there within 2%. Only the rows of one sitting compare.

## 2026-10-24 - the zmm product's offset copy made a tile ahead (result: rejected)

As the hypothesis above has it: the first panel of a column block XORs
the next tile's columns into the buffer, in zmm, before running its own
tile, where the copy was a pass of its own before any panel. kcheck ref's
bits on all four x86 backends, on the 9950X (gcc 15.2, AMX skipped) and
the c7i.large (gcc 13.3); test.wav v2 and v3 ref's bytes through
avx512-vnni, avx2-vnni and avx2 here and amx-int8 and avx512-vnni there.

On the 9950X, avx512-vnni, one sitting (06:36 to 06:38 UTC), five
interleaved rounds against d1bda18's code: q8bench in slices of 64 on one
pinned core, GMAC/s (up, down, attention), at T 750 425-437, 418-423 and
443-448 against 434-438, 421-424 and 439-442; at T 250 447-453, 422-428
and 450-459 against 443-452, 421-423 and 451-458. `pktito bench` on
test.wav, 16 threads, 252.5 to 260.4x against 250.0 to 263.1x (medians
255.9x and 255.6x); one pinned thread 38.2 to 38.4x against 38.2 to
38.5x. On the c7i.large, avx512-vnni, three rounds: q8bench at T 750
225-228, 207-211 and 234-238 against 225-232, 206-210 and 229-231;
`pktito bench`, one thread, 19.2x against 19.1 to 19.2x.

Level everywhere, where 5 to 6% was predicted on the product: made beside
the VPDPBUSDs, the copy's loads and stores cost what they cost as a pass
of their own. What would take the 6-7% out is making the copy less
often: the quantizer writing the offset bytes, or pk_gemm's slices of 256
rows, the copy a quarter as often (on the c7i.large, avx512-vnni, one
thread: 19.1x to 19.9-20.0x, the AMX entries above); both outside the int8
product. Rejected; the code is as before.

## 2026-10-24 - the float kernels in zmm (measured, a case for Peter)

CLAUDE.md makes the int8 product the only kernel that differs inside a
family, so this is a measurement for that rule's owner, not a change. A
prototype outside the tree, on d1bda18's code, gave the two AVX-512
builds (avx512-vnni, amx-int8) zmm forms of four kernels, each lane doing
what a ymm lane does in the same order:

- the float product: two panels of 16 rows by eight columns, 16 chains,
  where the ymm tile is one panel by four columns, 8 chains (T = 1 keeps
  the ymm tile);
- the quantizer: a block's 32 floats as two zmm, its bytes by VPMOVSDB;
- the layer norm: its 16 lanes in one zmm where ymm has two;
- SiLU.

kcheck, 1777 cases, ref's bits on all four x86 backends, on the 9950X
(gcc 15.2) and the c7i.large (gcc 13.3); test.wav v2 and v3 ref's bytes
through avx512-vnni and amx-int8, and v2 on the two EC2 instances. One
binary each, the forms picked by an environment variable read once (the
prototype's, for the A/B alone), interleaved rounds, one sitting a box;
`pktito bench` on test.wav, best of five:

| box | backend | threads | ymm | the zmm product | all four in zmm |
|---|---|---|---|---|---|
| 9950X | avx512-vnni | 1 | 38.3-38.5x | 41.3-41.5x (+8%) | 41.8-42.0x (+9%) |
| | | 16 | 252.0-260.1x | 259.6-266.0x (+2%) | 259.6-266.0x (+2%) |
| c7i.large | amx-int8 | 1 | 31.7-31.9x | 34.2-34.9x (+9%) | 33.6-35.7x (+11%) |
| | avx512-vnni | 1 | 19.2-19.3x | 19.8-19.9x (+4%) | 19.7-20.3x (+5%) |
| c7i.xlarge | amx-int8 | 2 | 55.3-56.0x | | 60.2-60.8x (+8%) |
| | | 1 | 31.1-31.6x | | 34.1-34.6x (+10%) |
| | avx512-vnni | 2 | 36.8-37.3x | | 38.1-38.6x (+3%) |
| | | 1 | 19.4-19.6x | | 20.4x (+5%) |
| c8i.xlarge | amx-int8 | 2 | 56.9-57.3x | | 61.4-61.5x (+7%) |
| | | 1 | 32.8-33.2x | | 35.8x (+8%) |
| | avx512-vnni | 2 | 35.4-35.9x | | 36.7-37.1x (+4%) |
| | | 1 | 19.1x | | 19.7-19.9x (+4%) |

(9950X and c7i.large: five and three rounds, the ymm column the
prototype's own ymm form, the percentages on the medians. The EC2 rows:
us-east-1, Ubuntu 24.04, gcc 13.3 builds from the c7i.large, three
rounds, the ymm column d1bda18's own binary; the prototype's ymm form ran
0.3 to 1.4% below it there.) A second sitting on the 9950X, three rounds:
one thread 38.3-38.6x against 41.7-41.9x, 16 threads 262.1-269.5x against
267.4-271.0x, zmm ahead in every pair.

PROFILE=1, one thread: the attention stage, which holds a query block's
three float products, 233 to 125 ms on the 9950X and 386-391 to 236-240
ms on the c7i.large (amx-int8); the quantize stage, which holds the layer
norms and SiLU, 78 to 62 ms and 165 to 134 ms. The product's tile on the
c7i.large (amx-int8, the attention stage): two panels by eight columns
237-259 ms, two by four and one by eight (8 chains, as many as the ymm
tile's) 280-284 ms; not tuned past that.

Not measured: Ice Lake, whose clock falls under 512-bit work (the zmm
product's entry: 2 to 6% at two threads); the softmax, GLU and the
depthwise convolution, stages of 1 to 2%. On these CPUs at one and two
threads it is the largest step left outside the int8 product, larger than
the AMX padded group and the zmm copy would have been together.

## 2026-10-24 - pk_gemm's step, 64 rows or 256 (measured, for Peter)

The AMX entries above have 256-row slices faster on the c7i.large at one
thread (amx-int8 29.5-29.9x to 30.7-31.6x, avx512-vnni 19.1x to
19.9-20.0x): each call's own work, the zmm product's offset copy, the
panels' weight sums, AMX's first panel of a group with its A from L2,
made a quarter as often. The slice-size rows above (the 12900HK, six
threads) have 64 best, 256 rows 1.166-1.169 s against 1.057-1.059: too few
items for the threads.

On the 9950X, avx512-vnni, d1bda18's code, gcc 15.2, interleaved rounds:
256 everywhere, one pinned thread, 39.7-39.9x against 38.3-38.6x (+3%);
16 threads 242.3-249.9x against 252.4-258.8x (-3 to -4%), two sittings.
256 only where a product's 256-row slices are at least the threads
(rows / 256 >= threads), 64 elsewhere: one thread 39.9x (the same code
path), 16 threads 248.7-255.8x against 255.7-257.7x (-1.5%: at 16
threads the feed-forward's up product is then one round of 16 slices,
where 64 rows make four). On EC2 the same rule, gcc 13.3, three rounds,
against d1bda18 (the float kernels' table above, the same sitting):

| instance | backend | 2 threads | 1 thread |
|---|---|---|---|
| c7i.xlarge | amx-int8 | 59.5-60.6x (+7%) | 32.5-33.4x (+7%) |
| | avx512-vnni | 38.6-38.7x (+4%) | 20.2-20.4x (+4%) |
| c8i.xlarge | amx-int8 | 60.1-61.1x (+6%) | 33.8-34.2x (+3%) |
| | avx512-vnni | 37.3-37.6x (+5%) | 19.8x (+4%) |

and on the c7i.large, amx-int8, one thread, with the float kernels in zmm
as well: 37.4-37.5x, against 31.9-32.3x for d1bda18, 33.1-34.2x for the
step alone and 34.1-35.3x for the float kernels alone; the two add. A
rule asking for two rounds (rows / 256 >= 2 x threads) would be this at one
and two threads and today's code at 16. core/encoder.c's, outside the
int8 product: not made here. The bits do not depend on the split, every
output being one call's.

## 2026-10-24 - pk_gemm's step: 256 rows where they make two rounds (hypothesis)

Peter's call on the entry above: take the step. The form taken is the
two-round rule, 256-row slices where a product's rows / 256 is at least
twice the threads and 64 elsewhere, which is the measured rule's code
path at one and two threads (every product here has rows / 256 >= 4) and
today's at 16, so the rows above carry over: +3 to 7% at one and two
threads on the 9950X, c7i and c8i, level at 16. It is core/encoder.c's,
so it reaches every board, and the ones it changes that nothing above
measured are those with three to eight threads, where only the
feed-forward's up product (4096 rows: 16 slices) moves: the 12900HK, the
Pi 5 and the Pi 3B+ at their four. The 12900HK's slice rows above have
64 best at six threads with 256 everywhere; its up product alone should
cost less than that. Expected: within the spread on those three, kept
if none falls by more than it. The bits do not depend on the split.

## 2026-10-24 - pk_gemm's step: 256 rows where they make two rounds (result: kept)

As the hypothesis above. On the boards it reaches that nothing above
measured, base (47f0093) against it, interleaved rounds, `pktito-c bench`
on test.wav, best of five:

| board | backend | threads | 47f0093 | 256 where two rounds |
|---|---|---|---|---|
| 12900HK, gcc 15.2 | avx2-vnni | 14 (its default) | 72.5-73.0x | 72.6-73.0x |
| | | 6 | 69.3-69.6x | 70.0-70.1x (+0.7%) |
| Pi 5, gcc 14.2 | neon-dp | 4 | 14.4-14.7x | 14.4-14.8x |
| Pi 3B+, gcc 14.2 | neon | 4 | 44.70 s (1.3x) | 44.72 s |

(12900HK five rounds at 14, three at 6; the Pi 5 five; the Pi 3B+ two
base and one of the change, each 2.5 minutes.) Level within the spread
where only the up product moves, a little ahead at six threads; the one-
and two-thread rows above stand. With the float kernels in zmm (the next
entry) the two together, on the 9950X and EC2, are there.

## 2026-10-24 - the float kernels in zmm (result: kept)

The prototype of the entry above, its A/B switches taken out: in the
AVX-512 builds the float product (two panels of 16 rows by eight
columns), the quantizer, the layer norm and SiLU in zmm. CLAUDE.md's line
now lets a variant replace a kernel where it measures faster, each to
ref's bits. Both changes against 47f0093, the same binary pair, three
interleaved rounds, test.wav, best of five; every transcript ref's, and
check-c passing with amx-int8 on the two Xeons:

| box | backend | threads | 47f0093 | both |
|---|---|---|---|---|
| 9950X, gcc 15.2 | avx512-vnni | 16 | 262.8-267.2x | 266.6-269.9x |
| | | 1 | 38.5-38.7x | 43.3-43.4x (+12%) |
| c7i.xlarge, gcc 13.3 | amx-int8 | 2 | 57.3-57.7x | 66.1-67.1x (+16%) |
| | | 1 | 32.4-32.5x | 37.7-38.0x (+17%) |
| | avx512-vnni | 2 | 37.9x | 41.3-41.5x (+9%) |
| | | 1 | 19.9-20.0x | 21.9-22.0x (+10%) |
| c8i.xlarge, gcc 13.3 | amx-int8 | 2 | 56.7-57.0x | 64.3-64.9x (+13%) |
| | | 1 | 32.4-32.5x | 35.5-36.0x (+10%) |
| | avx512-vnni | 2 | 35.3-35.8x | 38.5-39.0x (+9%) |
| | | 1 | 18.8x | 20.3x (+8%) |

(EC2: us-east-1, Ubuntu 24.04, fresh instances, both trees built there.)
Not measured: Ice Lake (c6i), whose clock falls under 512-bit work.

## 2026-10-24 - the x86 backends under two compilers (hypothesis)

CLAUDE.md now asks a tuned kernel to be benched under two compilers where
the board has them. Applied to the x86 kernels as they stand (ee893c3):
on the 9950X, gcc 15.2 against clang 21.1 (Ubuntu's, both), and gcc 14.3
beside them; avx512-vnni and avx2-vnni through PKTITO_KERNELS, `pktito-c
bench` on test.wav at one thread and 16, and q8bench at T 750 on one
thread; three interleaved rounds, one sitting. Expected: within the spread
everywhere, the int8 products being intrinsics with their schedules
written out, except where a loop holds more live vectors than registers
(the zmm product's PK_Q8_NC columns, the zmm float tile's 16 chains),
where one compiler's spill would show first. A backend that differs by
more than the spread is a climb of its own: why, then C both agree on.
amx-int8 needs a Xeon: gcc 13.3 against gcc 15 on one instance, one
sitting, separately.

## 2026-10-24 - the x86 backends under two compilers (result: a climb)

The hypothesis above, on the 9950X, ee893c3, one sitting, three
interleaved rounds (test.wav best of five; q8bench T 750, 64-row slices,
GMAC/s up/down/attention/decoder):

| backend | measure | gcc 15.2 | gcc 14.3 | clang 21.1 |
|---|---|---|---|---|
| avx512-vnni | bench, 1 thread | 43.3-43.7x | 43.5-43.7x | 45.6-45.7x |
| | bench, 16 | 269.3-274.5x | 270.6-271.7x | 269.6-273.0x |
| | q8bench | 424-427/415-416/431-440/94-95 | 435-441/416-422/441-446/95-96 | 458-460/431-437/463-466/103-106 |
| avx2-vnni | bench, 1 thread | 26.8-26.9x | 27.0x | 27.2x |
| | bench, 16 | 200.3-211.4x | 199.0-201.6x | 206.3-210.8x |
| | q8bench | 253-256/250-254/259-261/93-94 | 261-262/254-255/263-267/94 | 259-262/252-257/263-267/100-102 |

avx512-vnni's product is 7 to 8% behind under gcc, a whole run 5%; the
decoder's one-column product (T 1) 7 to 10% under gcc on both. A climb,
by the rule.

## 2026-10-24 - the zmm product's scales block-major for gcc (hypothesis)

objdump of q8_zmm's eight-column block loop (qb 32): clang 147
instructions where the activations are read in place (the broadcast
folded into VPDPBUSD as {1to16}) and 192 from the offset copy (the
broadcast first: the unsigned operand cannot be the folded one), 12
stack references; gcc 15.2 262 to 277 with 15 to 29, its eight scale
pointers a->d + (t0 + c) * nb in xmm27-31 and on the stack (vmovq rax,
xmmN before each FMA), and the block count and two offsets kept in
memory (add DWORD PTR [rsp+...]). Hypothesis: with the copy's tile's
scales laid out block-major beside the copied activations, a block's
eight at one pointer and constant offsets, gcc has the registers clang
uses, its loop comes to clang's length, and its product to clang's
GMAC/s (+5 to 8%), clang's level; the bits are the same, each scale the
same float broadcast. The decoder's T 1 path, which reads in place, is
a separate question.

## 2026-10-24 - the zmm product's scales block-major for gcc (result: half the gap, not kept as is)

As the hypothesis above: each tile's scales copied block-major beside the
offset copy (~/ec2-bench/x86climb/dsc-blockmajor.patch), kcheck ref's
bits under both compilers. gcc's loop lost its eight scale pointers (its
FMAs now DWORD BCST [rcx-0x20] to [rcx-0x4]) and went from 262-277
instructions and 22 to 29 stack references to 254 and 15; it still keeps
two running offsets in memory. Three interleaved rounds, avx512-vnni,
q8bench T 750 in 64-row slices (up/down/attention) and test.wav at one
thread:

| build | q8bench | bench, 1 thread |
|---|---|---|
| gcc 15.2, ee893c3 | 430-440 / 422-424 / 442 | 43.6-43.7x |
| gcc 15.2, scales block-major | 441-447 / 420-423 / 447-450 | 44.5-44.8x (+2.3%) |
| clang 21.1, ee893c3 | 458-463 / 431-436 / 466-467 | 45.5-45.7x |
| clang 21.1, scales block-major | 452-455 / 428-434 / 460-464 (-1.5%) | 45.7-45.9x |

Half of gcc's 5% closed, clang's product 1.5% down (the scales' copy, a
call's T x nb floats) with its run level. By the two-compiler rule not
finished: kept out. Next, the copy's activations interleaved by tile too
(a group's NC columns' four bytes adjacent), so that a block reads from
one activation pointer and one weight pointer at constant offsets under
either compiler; its cost is a transposing copy a call, which 64-row
slices at 16 threads (four panels a call) would feel most.

## 2026-10-27 - libpktito.so: the zmm product's thread-local copy under gcc (a measurement)

`make shared` builds the library as a shared object, for other languages
(examples/README.md), from the same sources compiled again with -fPIC.
./pktito-c's cli.o linked against it, against ./pktito-c: the 9950X under
WSL2 (1-minute load under 2, this bench's own), tree 226fbf8 with the
shared target, avx512-vnni, test.wav, best of 5, three runs each (s):

| build | 16 threads | 1 thread |
|---|---|---|
| gcc 15.2, ./pktito-c | 0.224-0.225 | 1.385-1.393 |
| gcc 15.2, libpktito.so | 0.232-0.240 (+3.5 to 7%) | 1.418-1.420 (+2%) |
| gcc 15.2, libpktito.so, -ftls-model=initial-exec | 0.225-0.227 | 1.386-1.388 |
| clang 21.1, ./pktito-c | 0.215-0.216 | 1.323-1.326 |
| clang 21.1, libpktito.so | 0.219-0.221 | 1.314-1.323 |

The gap is q8_zmm's offset copy, `static _Thread_local xoff`. In an
executable a thread-local is an offset from %fs; in a shared object it is
a call to __tls_get_addr, and gcc makes it at 13 places in q8_zmm, the
uses in PK_Q8_TILES' cases in the tile loop, where clang makes it once
(objdump -dr of backend/avx512-vnni/kernels.pic.o). -ftls-model=
initial-exec reads it at an offset again and closes the gap, but a
library built so does not load by dlopen, which is how ctypes, koffi and
P/Invoke load one: Python's ctypes.CDLL refuses it, "cannot allocate
memory in static TLS block" (glibc 2.43), the library's thread-locals
being 975 KB (each backend's offset copy, 320 KB of it the zmm
product's) against the static TLS glibc keeps spare for a library loaded
late. -ftls-model=local-dynamic, and the address copied to a
`uint8_t *const` at the top of q8_zmm, leave gcc's 13 calls: it holds the
address invariant and folds it back into each use.

Next: the copy's address from a function gcc does not inline, taken once
a call, should bring gcc's shared library to its executable's speed and
leave the executable where it is (a call a q8_zmm call). A kernel change:
its hypothesis here first, and benched under both compilers.

## 2026-10-27 - the offset copies' addresses taken once a call (hypothesis)

As the Next above: q8_zmm's offset copy, and the ymm product's (avx2-vnni's
k_gemm_q8, where gcc makes 4 calls to __tls_get_addr), each moved into a
function gcc does not inline that returns its address, taken once a call.
Expected: gcc's libpktito.so level with its ./pktito-c, 3.5 to 7% faster
at 16 threads and 2% at one on avx512-vnni, and avx2-vnni faster in kind;
clang's unchanged; both compilers' ./pktito-c unchanged within the spread,
the change there being one call a product call against its tile loops.
The bits cannot move: the copy is the same bytes at the same kind of
address. Benched as the measurement above, base and new interleaved,
avx512-vnni and avx2-vnni (PKTITO_KERNELS), 16 threads and one, gcc 15.2
and clang 21.1. amx-int8's own thread-locals (ready, xshort, dshort) stay
as they are: this board has no AMX to run them, though its build takes
the new q8_zmm with the others.

## 2026-10-27 - the offset copies' addresses taken once a call (result: kept)

As the hypothesis: zmm_xoff and ymm_xoff, noinline, each return its
product's thread-local copy, taken once a product call. gcc's calls to
__tls_get_addr in the shared objects went from 13 in q8_zmm and 4 in
avx2-vnni's k_gemm_q8 to the one in each function; clang's stayed at one.
kcheck: every case ref's bits under both compilers (1779 each, avx2,
avx2-vnni, avx512-vnni); libpktito.so's test.wav transcripts ref's, v2
and v3, on avx512-vnni and avx2-vnni under both. The 9950X under WSL2, the
other sessions' jobs held, base (da42a7b) and new interleaved, test.wav,
best of 5, three rounds (s):

| kernel | threads | build | base | new | |
|---|---|---|---|---|---|
| avx512-vnni | 16 | gcc, libpktito.so | 0.238-0.241 | 0.224-0.227 | -5.9% |
| avx512-vnni | 16 | gcc, ./pktito-c | 0.219-0.225 | 0.225-0.227 | (below) |
| avx512-vnni | 16 | clang, libpktito.so | 0.220-0.222 | 0.220-0.222 | level |
| avx512-vnni | 16 | clang, ./pktito-c | 0.218-0.222 | 0.213-0.222 | level |
| avx512-vnni | 1 | gcc, libpktito.so | 1.414-1.418 | 1.386-1.387 | -2.0% |
| avx512-vnni | 1 | gcc, ./pktito-c | 1.381-1.386 | 1.383-1.385 | level |
| avx512-vnni | 1 | clang, libpktito.so | 1.315-1.316 | 1.313-1.323 | level |
| avx512-vnni | 1 | clang, ./pktito-c | 1.317-1.320 | 1.311-1.321 | level |
| avx2-vnni | 16 | gcc, libpktito.so | 0.315-0.319 | 0.303-0.306 | -3.8% |
| avx2-vnni | 16 | gcc, ./pktito-c | 0.297-0.303 | 0.299-0.307 | level |
| avx2-vnni | 16 | clang, libpktito.so | 0.298-0.301 | 0.299-0.305 | level |
| avx2-vnni | 16 | clang, ./pktito-c | 0.294-0.300 | 0.296-0.302 | level |
| avx2-vnni | 1 | gcc, libpktito.so | 2.291-2.302 | 2.237-2.248 | -2.4% |
| avx2-vnni | 1 | gcc, ./pktito-c | 2.238-2.245 | 2.238-2.244 | level |
| avx2-vnni | 1 | clang, libpktito.so | 2.212-2.217 | 2.215-2.217 | level |
| avx2-vnni | 1 | clang, ./pktito-c | 2.212-2.214 | 2.210-2.215 | level |

gcc's ./pktito-c on avx512-vnni at 16 threads read +2.7% by its minimum,
one base run of 0.219 against pairs otherwise 0.4% apart; ten more
interleaved rounds of it: base 0.223 / 0.225 / 0.227 (min, median, max),
new 0.221 / 0.226 / 0.227, and avx2-vnni's 0.296 / 0.302 / 0.304 against
0.298 / 0.302 / 0.305: level. So gcc's shared library is level with its
executable, as clang's was, and neither executable moved. Kept.

## 2026-10-27 - the day's builds, by their commits on main (a note)

The base above, da42a7b, is the hypothesis's commit as it stood on
226fbf8, before the branch was rebased onto main; on main it is 70a3891,
with main's commits of 2026-10-25 to 27 and the threads branch's (the
encoder's race, Windows' power throttling, PK_SPIN) beneath it, which
these measurements predate.

## 2026-10-27 - the release builds' compiler, x86 (a measurement)

0.9.4's x86 builds were gcc's: the Linux .debs gcc 11.4 (Ubuntu 22.04's,
for its glibc), the app's Windows x64 engine MSYS2's gcc 16.2. Each is
measured here against what its build could use instead, on be5c4ae's tree
(whose x86 objects are ebce152's under gcc and clang 21; clang 14 needs
be5c4ae to link at all). Every binary static; each one's test.wav
transcript is test.v2.json's bytes, and kcheck gives ref's bits under
every compiler on every backend each CPU has (1779 cases). Each
compiler's binary was built at four placements, 0, 16, 32 and 48 bytes
of NOPs at the head of each x86 backend object (a top-level .skip,
through -include), so that a compiler's figure is not one layout's.
Interleaved rounds, the order turned each round, test.wav through v2,
best of 5 at the default threads. A compiler's time a round is its four
placements' mean; the tables give its ratio to the base compiler a
round, median (range), and the spread is the largest over the compilers
of the slowest placement's median over the fastest's.

Linux: EC2 xlarge instances in us-east-1b, Ubuntu 24.04, one instance
each, the binaries from an Ubuntu 22.04 container: its gcc 11.4.0 and
12.3.0 and clang 14.0.0, the toolchain PPA's gcc 13.4.0 and
apt.llvm.org's clang 21.1.8. The threads are the cores: two on Intel's
xlarge and on AMD's with SMT, four on the c7a and the c8a.

| instance, rounds | CPU | backend, threads | gcc 11 | gcc 12 | gcc 13 | clang 14 | clang 21 | spread |
|---|---|---|---|---|---|---|---|---|
| c5a, 6 | EPYC 7R32, Zen 2 | avx2, 2 | 6.381 s | -0.5% (-0.6 to -0.2) | -0.3% (-0.5 to -0.2) | -4.6% (-4.9 to -4.5) | -4.5% (-4.7 to -4.2) | 0.3% |
| c6a, 6 | EPYC 7R13, Zen 3 | avx2, 2 | 4.219 s | -1.0% (-1.5 to -0.5) | -1.0% (-1.3 to -0.7) | -3.0% (-3.2 to -2.7) | -5.2% (-5.5 to -5.1) | 0.4% |
| c7a, 6 | EPYC 9R14, Zen 4 | avx512-vnni, 4 | 0.866 s | +0.5% (+0.4 to +0.5) | -0.1% (-0.3 to -0.1) | -1.2% (-1.3 to -1.1) | -1.8% (-1.9 to -1.7) | 0.3% |
| c8a, 6 | EPYC 9R45, Zen 5 | avx512-vnni, 4 | 0.457 s | -0.9% (-1.0 to -0.8) | +0.4% (+0.3 to +0.5) | -2.7% (-2.8 to -2.5) | -4.3% (-4.5 to -4.1) | 1.1% |
| m5zn, 6 | Xeon 8252C, Cascade Lake | avx512-vnni, 2 | 1.921 s | -0.5% (-0.6 to -0.2) | +0.1% (0.0 to +0.3) | +1.7% (+1.6 to +2.2) | -2.7% (-3.1 to -2.2) | 0.3% |
| c6i, 6 | Xeon 8375C, Ice Lake | avx512-vnni, 2 | 2.448 s | -0.9% (-1.4 to -0.6) | +0.2% (0.0 to +0.5) | +0.2% (-0.1 to +0.6) | -2.6% (-2.9 to -2.3) | 0.4% |
| c7i, 6 | Xeon 8488C, Sapphire Rapids | amx-int8, 2 | 1.127 s | -0.5% (-0.8 to 0.0) | 0.0% (-0.3 to 0.0) | +0.8% (+0.3 to +1.4) | +0.4% (+0.4 to +1.0) | 0.5% |
| c8i, 6 | Xeon 6975P-C, Granite Rapids | amx-int8, 2 | 0.913 s | +0.1% (-0.3 to +0.2) | 0.0% (-0.5 to +0.4) | +2.5% (+1.8 to +2.9) | +2.1% (+1.7 to +2.5) | 0.8% |

clang 21 is ahead everywhere but amx-int8, by 1.8 to 5.2%; gcc 12 and 13
are within about 1% of gcc 11; clang 14 runs from 4.6% ahead to 2.5%
behind. On the two AMX Xeons each compiler's binary was linked with the
other's backend/amx-int8 object (pads 0 and 32, 8 rounds, a fresh c8i
and c7i, every transcript test.v2.json's bytes):

| build | c8i | c7i |
|---|---|---|
| gcc 11 | 0.912 s | 0.944 s |
| gcc 11, clang 21's amx-int8 | +2.80% (+2.02 to +3.51) | +1.51% (+0.53 to +2.60) |
| clang 21, gcc 11's amx-int8 | -0.58% (-1.14 to +0.05) | -0.58% (-0.95 to +0.21) |
| clang 21 | +2.02% (+1.59 to +2.41) | +0.69% (+0.26 to +1.55) |

The loss is clang's amx-int8 object, and the rest of clang's build is
ahead there too. PROFILE=1 on the c8i, two runs of each: under clang 21
the int8 products' stages are 2.2 to 3.1% slower (feed-forward, q k v,
attention out, conv), attention's float kernels 4.0%, and subsampling,
layer norms and the depthwise convolution 3 to 6% faster. Its loop of
four AMX steps is gcc's to within three instructions (159 against
gcc 11's 162), with the same 20 tile operations in the same order, the
same 64 reads of the C buffers and no spill; where the time goes is not
found.

Windows x64: the 9950X (Windows 11, the other sessions' jobs held),
MSYS2 UCRT64's gcc 16.2.0 against llvm-mingw's clang 23.1.4, each the
whole toolchain (llvm-mingw's lld and runtime with its clang), at the
same four placements, 6 rounds at 16 threads, each backend through
PKTITO_KERNELS:

| backend | gcc 16.2 | clang 23.1 | clang against gcc | spread, gcc / clang |
|---|---|---|---|---|
| avx512-vnni | 0.2165 s | 0.2120 s | -1.99% (-3.87 to -0.80) | 2.6% / 4.1% |
| avx2-vnni | 0.2910 s | 0.2850 s | -2.42% (-3.32 to -1.32) | 1.4% / 1.2% |
| avx2 | 0.5500 s | 0.5000 s | -10.25% (-13.42 to -8.81) | 1.5% / 3.8% |

At one thread (PROFILE=1, one run of each) clang is 5.4%, 2.2% and 9.3%
ahead, in the int8 products' stages: avx2's feed-forward, q k v,
attention out and conv each 9 to 10%. The app built both ways
(pktito-9950x; its route b is the whole of llvm-mingw, as here) ran
AliceBench 3.5% faster on the 9950X under clang, its transcripts byte for
byte gcc's.

So on these CPUs clang's int8 products are the faster, amx-int8's aside,
and gcc 16.2's avx2 product is 10% behind clang 23's. By the
two-compiler rule each is a climb of its own: why, then code both
compilers agree on - for a product's block loop, what neon-dp's 4-column
tile now is, assembly (backend/neon-dp/CLIMB.md). 0.9.4 takes clang for
both (pktito-mac's call): llvm-mingw for Windows x64, apt.llvm.org's
clang 21 for the Linux x86_64 .deb, amx-int8 included.

The app's own build of it, the 0.9.4 candidate's amd64 .deb (pktito-app
23d9b73, its engine 69e8242's under clang 21.1.8, all of it): on a c7i
(Xeon 8488C) its pktito-cli's --json of test.wav is the same 9292 bytes
at the default backend, amx-int8, and under PKTITO_KERNELS ref, avx2,
avx2-vnni, avx512-vnni and amx-int8, and they are the bytes it writes on
the 9950X under every backend there and 0.9.3's .deb (gcc 11.4) wrote
(pktito-9950x's check).

## 2026-10-28 - the release builds' compiler, x86: an i7-10875H (a measurement)

The Windows x64 binaries of the entry above (be5c4ae's tree, MSYS2's
gcc 16.2.0 and llvm-mingw's clang 23.1.4, each at the four placements)
on a laptop's i7-10875H (Comet Lake, 8 cores, 16 threads; Windows 11 on
AC, the Balanced plan, the clock steady at about 2.4 GHz), pktito-
windows' sitting of 2026-10-28: 6 interleaved rounds, test.wav through
v2, best of 5, every run avx2 at its default 8 threads. Every binary's
transcript is test.v2.json's bytes. clang against gcc a round: -12.3%
(-13.5 to -10.0), gcc's time 3.65 s; the spread 1.5% (clang's; gcc's
1.1%). The 9950X's avx2 row above is -10.25%: clang's avx2 product is
ahead on both. The laptop's other processes took 4.7 of its 16 threads'
time throughout (2.0 to 8.0), alike in both compilers' rounds.

## 2026-10-28 - the passes between the products, made by the products' items (hypothesis)

On the 9950X a run at 16 threads is no faster than at 8. cb12bab, clang
21.1.8, avx512-vnni, test.wav (60 s in three chunks), plain builds, three
interleaved rounds, best of 5: Linux (WSL) 0.218-0.316 s at 8 threads,
0.222-0.232 at 16, 0.283-0.647 at 32; Windows 0.205-0.206, 0.204-0.227 and
0.229-0.260. A PROFILE=1 build of the same tree that also counts each
stage's pool jobs, two rounds at each of 1, 2, 4, 8 and 16 threads on
each OS (the other sessions' jobs held), Linux, the two rounds' means:

| stage | 1 thread | 8 | 16 | jobs | a job at 1 | at 8 | at 16 |
|---|---|---|---|---|---|---|---|
| norm+quantize | 14.1 ms | 3.0 | 6.0 | 288 | 48.9 us | 10.6 | 21.1 |
| silu+quantize | 37.1 | 6.7 | 10.9 | 144 | 257.6 | 46.0 | 75.7 |
| quantize, attention and conv out | 3.3 | 0.9 | 2.7 | 144 | 23.3 | 6.1 | 18.9 |
| attention repack | 19.0 | 3.9 | 8.8 | 72 | 265.0 | 53.8 | 122.7 |
| glu | 5.0 | 1.1 | 2.6 | 72 | 68.8 | 15.1 | 36.5 |
| depthwise | 15.2 | 2.5 | 2.8 | 72 | 211.6 | 35.3 | 38.8 |
| layer norms | 1.9 | 0.6 | 1.0 | 72 | 25.9 | 7.8 | 14.3 |
| feed-forward | 665.8 | 118.2 | 96.7 | 288 | | | |
| q k v pos | 206.4 | 40.0 | 35.2 | 288 | | | |
| decoder | 22.1 | 11.4 | 12.4 | 946 | | | |
| the run, best of 5 | 1.317 s | 0.218 | 0.222 | | | | |

The products still gain from 8 threads to 16; the passes between them
lose, every job slower at 16 threads than at 8 in its own time, with the
same work: the depthwise convolution's by a tenth, the rest's 1.6 to 3.1
times (the repack's 54 us a job to 123). Windows' rounds are the same.
The pool's own cost is not it: a job of one empty item a thread takes
0.76 us at 8 threads, 2.05 at 16 and 4.46 at 32, and 16 items of 20 us
22.47 us at 16 (a bench of core/pool.c alone, best of 5 batches), about
3.5 ms of a run's 1,750 jobs at 16.

What these passes share is that each reads what the product before it
wrote, all of it, split the other way: a product's items are slices of
its output's rows (pk_gemm), a pass's are blocks of frames, heads or
channels, so every item reads lines most of the other threads wrote. At
16 threads half of those are on the other CCD, the lines in its caches,
reached over Infinity Fabric; at 8, if the scheduler keeps them on one
CCD (Windows prefers the faster CCD's cores, and WSL's vCPUs are threads
Windows schedules), none are. The sitting will check which cores 8
threads run on.

The hypothesis: three of the passes read only what one item of the work
before them made, and can be made by that item, from its own core's
cache, with nothing to wait for and no lines from the other threads:

- the feed-forward's SiLU and quantize, by the up product's items, on
  their tile, a frame at a time: a slice of 64 or 256 of the hidden's
  rows is 2 or 8 whole blocks of 32. The hidden's int8 then needs an
  array of its own, since the up product reads a->act while its items
  write the hidden: Te (F + F/8) bytes, 3.5 MB at a minute.
- the attention's repack, by the K and position products' items: each
  transposes its tile into the panels, the last column chunk's item
  writing the padding.
- the convolution module's output quantize, by dw_item: its 64 channels
  of a frame are 2 whole blocks.

Not here: the attention's output quantize, which att_item could make
only where a head's dimensions are whole blocks of 32 (Parakeet's are
128, but the tiny model's, tools/pktito_tools/tiny.py, 16); the layer
norms (a frame's norm needs all of its rows, which every item of the
product before wrote), the GLU (its gate is D rows from its value, in
another item's slice; with pw1's items made of a slice and its gate's
slice it could be the product's, a split of its own for an entry of its
own), and the depthwise convolution, which reads the GLU's output across
frames.

No bits change: SiLU is element-wise, the quantizer's blocks are
independent (a column's blocks quantized a part at a time are the bytes
of the whole column's), the repack is a copy, and kcheck holds every
backend's quantize and SiLU to ref's at these lengths (n of 32 to 1280,
T of 1 to 20; SiLU at 1 to 3000).

Expected on the 9950X, Linux: at 16 threads the three passes' 21.1 ms a
run (10.9 + 8.8 and half of 2.7) fall to about their one thread's work
over 16 (57.8 / 16 = 3.6 ms), plus the epilogues' calls (each kernel
call takes 32 to 256 inputs), so the run 11 to 17 ms faster (5 to 8%)
and then ahead of 8 threads. At 8 threads 11.1 ms (6.7 + 3.9 + 0.45) to
about 7.2, 1.5 to 2%. At one thread level or a little ahead: SiLU's copy
through a row of scratch is gone, and the repack reads its tile from L2
where it read its whole array from L3. Every board runs this code
(core/encoder.c); on the others, level or ahead. Kept if the 9950X's
16-thread run moves by more than the spread on Linux and Windows, and no
board falls by more than its own.

## 2026-10-28 - the passes between the products, made by the products' items (result: kept)

The bits, on the 9950X: make check-c (kcheck, every x86 variant);
test.wav's --json through v2 and v3 the bytes of test.v2.json and
test.v3.json at 15 thread counts from 1 to 32; the tiny model's --json
and --stream, and v2's and v3's --stream, the bytes of f6bc689's at 8
thread counts; and make check-tsan clean. make check-c also passes on
pi5c (gcc 12.2: kcheck's neon-dp and neon) and pi4 (gcc 14.2: neon, both
models' transcripts).

The SiLU as the hypothesis had it, in place on the up product's tile and
quantized from there, lost on every Pi. Static builds of f6bc689 and of
it, gcc 12.3 and clang 21.1.8, test.wav at the default threads, best of
5, interleaved rounds (five; three on pi4), the medians:

| board | gcc 12 | clang 21 |
|---|---|---|
| pi5c (Pi 5, 4 GB) | +1.29% | +0.90% |
| pi5-8gb (Pi 5, 8 GB) | +4.60% | +4.55% |
| pi4 (Pi 400) | +0.95% | +0.71% |

and pi3, in cycles a run (each run's transcription time times the ARM
clock's mean over it, sampled every half second, the board at its 70 C
limit; the best of a build's five in each of three pairs), +0.07% (-0.29
to +0.46). PROFILE=1 (gcc 12, two runs a build) put it in the
feed-forward: +231.7 ms a run on pi5c for the 173.0 the quantize gave
back, +433.9 for 185.7 on pi5-8gb. An item of the up product at four
threads is 256 rows of all of a chunk's frames (250 at 20 s), a 250 KB
tile made while the item streams the slice's 256 KB of weights; by the
time it reaches a frame's SiLU most of the tile has left a Pi 5's 512 KB
of L2, written back, and the SiLU in place dirtied those lines again, so
they went out twice. The form measured from here on, and the one in the
tree, takes a frame of the tile through a row of thread scratch as the
pass did (copy, SiLU, quantize), so that the tile is only read.

On the 9950X (the box quiet, pinfer's jobs held; Linux in WSL, clang
21.1.8 and gcc 15.2; Windows, llvm-mingw clang 23.1.4), interleaved
rounds, four at 16 and 8 threads and three at 1, 2 and 4, best of 5,
f6bc689 against this, the medians (in brackets f6bc689's rounds' range
over its median, then the paired rounds' changes):

| threads | Linux clang 21 | Linux gcc 15 | Windows clang 23 |
|---|---|---|---|
| 16 | 0.221 to 0.212 s, -3.85% (0.9%; -1.83 to -4.52) | 0.228 to 0.216, -5.05% (0.4%; -3.96 to -5.70) | 0.205 to 0.193, -6.10% (2.9%; -4.37 to -7.69) |
| 8 | 0.230 to 0.220, -4.57% (9.6%; +1.86 to -7.59) | 0.230 to 0.230, 0.00% (7.0%; +2.17 to -6.50) | 0.208 to 0.206, -0.72% (2.9%; +1.46 to -2.37) |
| 4 | 0.376 to 0.377, +0.27% (0.5%) | | |
| 2 | 0.690 to 0.691, +0.14% (0.6%) | | |
| 1 | 1.319 to 1.330, +0.83% (0.3%; +0.23 to +0.83) | | 1.291 to 1.305, +1.08% (0.5%; +0.93 to +1.08) |

and the in-place form at 16 threads -3.40% (Linux, clang) and -5.12%
(Windows), at 8 +0.43% and -1.20%. The other boards, the same way as the
Pis above (the M4 Pro: Apple clang 21, eight rounds, the Mac in use and
its rounds drifting from 0.43 to 0.49 s, so paired; storage1, the
12900HK: gcc 15.2 and clang 21.1.8 static, six rounds; the GB10: its ten
X925s, six rounds, GLM's vLLM idle on the GPU throughout):

| board | threads | compiler | f6bc689 | this | change | f6bc689's spread |
|---|---|---|---|---|---|---|
| GB10 | 10 | gcc 12 | 0.432 s | 0.399 s | -7.64% | 2.5% |
| | | clang 21 | 0.430 | 0.398 | -7.55% | 2.3% |
| pi5c | 4 | gcc 12 | 3.661 | 3.629 | -0.87% | 0.14% |
| | | clang 21 | 3.656 | 3.618 | -1.04% | 0.30% |
| pi5-8gb | 4 | gcc 12 | 4.190 | 4.158 | -0.76% | 0.57% |
| | | clang 21 | 4.177 | 4.127 | -1.20% | 1.3% |
| pi4 | 4 | gcc 12 | 18.759 | 18.752 | -0.04% | 0.42% |
| | | clang 21 | 18.729 | 18.690 | -0.21% | 0.29% |
| pi3 | 4 | gcc 12 | cycles | | +0.03% (pairs -0.32 to +0.36) | |
| M4 Pro | 8 | Apple clang 21 | 0.479 | 0.480 | +0.21% (paired 0.00) | drift |
| | 12 | | 0.469 | 0.468 | -0.32% (paired -0.95) | |
| | default (12) | | 0.469 | 0.465 | -0.85% (paired -0.65) | |
| storage1 | 14 | clang 21 | 0.819 | 0.819 | 0.00% | 1.0% |
| | | gcc 15 | 0.825 | 0.824 | -0.06% | 1.0% |

PROFILE=1, each stage's change in ms a run (two runs a build; the
9950X's clang builds on Linux at 16, 8 and 1 thread and Windows at 16
and 8; the boards' at their default threads, the M4 Pro's at 8):

| stage | L16 | W16 | L8 | W8 | L1 | GB10 | pi5c | pi5-8gb | pi4 | M4 Pro | storage1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| quantize | -12.2 | -11.9 | -6.5 | -6.9 | -39.0 | -37.9 | -171.6 | -191.7 | -497.9 | -15.6 | -32.6 |
| feed-forward | +6.0 | +8.5 | +1.5 | +5.8 | +35.1 | +14.6 | +163.6 | +180.3 | +529.8 | +12.7 | +26.2 |
| q k v pos | +3.5 | +4.8 | +3.9 | +4.1 | +25.7 | +3.8 | +58.7 | +103.7 | +185.0 | +2.4 | +7.0 |
| attention repack | -8.9 | -8.8 | -3.8 | -4.0 | -19.5 | -15.5 | -83.4 | -149.1 | -151.9 | -3.4 | -8.1 |
| glu + depthwise | +1.0 | +1.0 | +1.0 | +0.9 | +2.5 | +0.8 | +9.5 | +5.2 | +33.6 | +0.7 | +2.5 |
| the run | -13.5 | -7.5 | +3.5 | +3.5 | +6.5 | -32.0 | -28.5 | -41.0 | +118.5 | -1.5 | -5.5 |

Which logical processors ran the threads (the hypervisor's % Total Run
Time for each, typeperf, over three benches; LPs 0-15 the first CCD): at
8 threads Windows put 417 points of 459 on the first CCD and WSL 476 of
525; at 16, 498 of 827 and 572 of 887.

So what a pass costs made by the items is its work less its traffic, and
the 9950X's traffic is between its CCDs. At 16 threads, half on each
CCD, the feed-forward took on 49 to 71% of the quantize stage's fall
(the convolution module's quantize, in it, went to glu + depthwise) and
the K and position products 39 to 55% of the repack's, and the run is 4
to 6% faster: the hypothesis' 5 to 8% at its low end, 9 to 12 ms where
it expected 11 to 17. Sixteen threads now run ahead of eight (Linux
0.212 s against 0.220, Windows 0.193 against 0.206), where f6bc689's
were level with eight on Windows. At 8 threads, all on one CCD, the
passes had found their lines in its L3, and made by the items they cost
about what they did (84 and 103% on Windows): level, not the 1.5 to 2%
expected. So on the boards whose cores share a last-level cache: the
feed-forward took on 80 to 95% of the quantize's fall on pi5c, pi5-8gb,
storage1 and the M4 Pro, and the products 70 to 86% of the repack (pi4
106 and 122%: its four cores share 1 MB of L2 with the products'
weights; its bench rounds are level). On the GB10, whose ten X925s are
five in each of its two clusters, each with an L3 of its own (cpus 5-9
with 8 MB, 15-19 with 16 MB; sysfs), 39% and 25%, and the run 7.6%
faster.

At one thread the change costs 0.7 to 1.1% on the 9950X (+6.5 ms in the
profile), where the hypothesis expected level or a little ahead: the
feed-forward took on 90% of the quantize's fall, but the products 132%
of the repack, 25.7 ms for its 19.5. The repack walked one head's
dimensions down 16 frames at a time; panels_tile works out each row's
head and dimension with a division by dk, for every row of every
16-column group of the tile, and loops over a panel's columns between
bounds it computes, so no compiler unrolls that loop. Whether that is
the third more is the next entry's question.

Peak RSS, storage1, test.wav at 14 threads (GNU time's maximum resident
set): 670,764 KB to 672,852, the hidden's int8 array (Te F 9/8 bytes).

Kept: the 9950X's 16-thread runs moved by more than their spread on
Linux (-3.85% under clang, -5.05% under gcc; spreads 0.9 and 0.4%) and
Windows (-6.10%; 2.9%), the GB10's by 7.6%, and no board fell at its
default threads by more than its own spread. The 9950X at one thread,
0.7 to 1.1% behind, is the next entry's.

## 2026-10-28 - the panels' rows a head at a time (hypothesis)

At one thread on the 9950X the K and position products' items take 25.7
ms a run more making the panels than the products did alone, where the
repack pass they replace took 19.5 (the entry above). For every row of
every 16-column group of its tile, panels_tile finds the row's head and
dimension as r / dk and r % dk, a division by a value known only at run
time, and copies the group's columns in a loop whose bounds it computes
and whose every column it tests for padding. The repack walked one
head's dimensions down a group of 16 frames, no division, in a loop of
16. The hypothesis: the head and dimension stepped along the rows (one
division a group, for the tile's first row), and a group whose 16
columns are all the tile's and all keys or positions (every group but
the tile's edges and the padding) copied by a loop of 16 with no test,
make the panels cost the products no more than the repack cost at one
thread: the 9950X's one-thread run back to f6bc689's (1.319 s under
clang on Linux, 1.291 on Windows), and the 16-thread run level with the
entry above's or a little ahead. No bits change: the same copy, in the
same order. Measured on the 9950X at 1 and 16 threads under clang and
gcc on Linux and clang on Windows, and on pi5c, against the entry
above's tree.

## 2026-10-28 - the panels' rows a head at a time (result: kept)

No bits change: test.wav's --json through v2 and v3 the bytes of
test.v2.json and test.v3.json at 9 thread counts from 1 to 32, the tiny
model's --json and --stream and v2's --stream the bytes of f6bc689's at
8, and make check-c on the 9950X.

On the 9950X (the box quiet, pinfer's jobs held), three interleaved
rounds at one thread and four at 16, best of 5, against d648b27 (the
entry above's tree) and, at one thread, f6bc689:

| threads | tree | Linux clang 21 | Linux gcc 15 | Windows clang 23 |
|---|---|---|---|---|
| 1 | f6bc689 | 1.323 s (1.7%) | | 1.292 (0.4%) |
| | d648b27 | 1.324 (0.4%) | 1.397 (0.2%) | 1.304 (0.3%) |
| | this | 1.321 (0.5%) | 1.388 (0.9%) | 1.291 (0.5%) |
| 16 | d648b27 | 0.213 (2.8%) | 0.214 (2.8%) | 0.190 (5.3%) |
| | this | 0.209 (3.3%) | 0.215 (2.3%) | 0.190 (1.6%) |

(the medians, each with its rounds' range over the median). Against
d648b27 at one thread: Windows -1.00% (pairs -0.77 to -1.00), back to
f6bc689's 1.291 s from the +0.93% that d648b27 cost it (+1.08% in the
entry above); Linux -0.23% under clang (pairs 0.00 to -0.45) and -0.64%
under gcc (pairs -0.14 to -0.79). d648b27's one-thread cost on Linux did
not show in this sitting, +0.08% against f6bc689 where the entry above
had +0.83%. At 16 threads level: -1.88%, +0.47% and 0.00%, inside the
spreads.

pi5c (static gcc 12.3 and clang 21.1.8, 4 threads, five rounds): +0.03%
(pairs -0.33 to +0.06) and +0.49% (pairs -0.11 to +1.19), spreads 0.8 to
1.0%: level, PROFILE's q k v pos +3.0 ms of 587. storage1 (the 12900HK,
clang 21 and gcc 15 static, five rounds): at one thread -0.14% and
-0.07%, inside spreads of 3.0 to 5.6% (one thread there lands on a P-core
or an E-core), at 14 threads -0.61% (pairs -0.85 to -0.12) and -0.24%;
PROFILE's q k v pos -0.6 ms at either.

So what the change takes away is the 9950X's: on the 12900HK the panels
cost about the same both ways, and whether the division or the loop's
padding test was the 9950X's third more is not separated. Kept: the
9950X's one-thread runs moved by more than the spread on Windows (-1.00%
against spreads of 0.3 and 0.5%), every Linux pair at one thread is
level or faster, and no board fell by more than its own spread.

## 2026-10-28 - a model's load out of the power throttling (a note)

pktito-windows, in the i7-10875H sitting above, read each busy thread's
ThreadPowerThrottling 4 s into one of its bits checks (pktito-c --json,
test.wav): only the calling thread was busy, 3.8 s of CPU, system-managed,
and no pool worker had started yet; in a trial two minutes before, the
same program had its 8 workers out of the throttling by 4 s. Untimed.
In pktito-c a pool starts with pk_asr_new, right after pk_model_load, so
that start was still loading. The library took the caller's thread out
for a pk_asr_run and a pk_vad_probs (core/os.c), not for a load.

pk_model_load and pk_model_from_bytes now take it out for the call, and
pktito-c takes its own thread out for the whole run: the audio's read
and resampling, a stream's VAD a frame at a time and the output were on
a system-managed thread too. On the 9950X (Windows 11 x64, llvm-mingw
clang 23), a probe that loads v2 twenty times while a second thread
reads the loading thread's two masks as fast as it can:

| build | the thread before | during the loads | after | best load |
|---|---|---|---|---|
| HEAD (2b34cd3) | 0/0, system-managed | 0/0 | 0/0 | 3.43 ms |
| | 1/1, EcoQoS | 1/1 | 1/1 | 3.37 |
| this | 0/0 | 1/0, out | 0/0 | 3.41 |
| | 1/1 | 1/0 | 1/1 | 3.43 |

(ControlMask/StateMask; the poller also saw each thread's own state
between the loads.) With the file in the page cache the checks take
3.4 ms, under EcoQoS or not: the 9950X has no cores Windows calls
efficient, so EcoQoS moves only its clock, and 3.4 ms is far from the
3.8 s seen. What that start spent its time on is not known; neither a
load from a file not yet cached nor a chip with efficient cores, where
EcoQoS moves threads, was measured.

## 2026-10-28 - each chunk's decoder beside the next chunk's encoder (hypothesis)

pk_asr_run takes one chunk at a time: its features, its encoder, then its
decoder, whose search runs a token's LSTM and a step's joint one after
another, the widest products split over four threads (PK_DEC_THREADS)
while the others wait. test.wav is three chunks (20.0, 27.8 and 12.0 s;
71, 117 and 44 tokens), and the decoder's part of a run, from the
PROFILE=1 runs of the entries above (d648b27's tree; the decoder is the
same at 25ca5e6):

| board | threads | the decoder, ms a run | of the run |
|---|---|---|---|
| 9950X, Linux, clang 21 | 16 | 11.7-11.9 | 5.1-5.4% of 0.208-0.212 s |
| | 8 | 10.9-11.0 | 4.7% |
| | 1 | 22.1 (cb12bab) | 1.7% |
| 9950X, Windows, clang 23 | 16 | 10.2 | 4.7% |
| GB10, its X925s | 10 | 16.3-17.7 | 3.9-4.1% |
| M4 Pro | 8 | 18.7 | 3.8-3.9% |
| 12900HK (storage1) | 14 | 25.1-25.5 | 3.1% |
| | 1 | 34.2-36.2 | 1.2% |
| Pi 5 (pi5c) | 4 | 139.5-144.2 | 3.8-4.0% |
| Pi 4 | 4 | 551-554 | 2.9% |

A chunk's search starts from a zero state and reads nothing of the
encoder's but the joint's encoder half of its own frames, which one
product makes for every frame before the search starts: Te x J floats,
0.96 MB for 30 s. So chunk i's search can run while chunk i+1's
features and encoder do, if the two chunks' joint halves are kept apart.

The hypothesis: pk_asr_run_chunks(a, x, chunks, n, fn, arg) runs the
loop itself. Chunk 0's encoder; then for each chunk i, its search on the
pool's last thread beside chunk i+1's features, encoder and joint half
on the others, and fn(arg, i, its tokens) once both are done; the last
chunk's search as now, on four threads. Beside the encoder the search
splits nothing, its products whole on its own thread. That thread is one
of the encoder's, except where the transcriber runs on its fast kind
alone (a GB10's X925s, an M4 Pro's P-cores where the race keeps them), and
there it is an idle one. The encoder's race (encoder.c, layers) waits for
the search first, so that each side has the threads it names. pk_asr_run
stays as it is; pktito-c's transcription and bench take the new call,
and --stream keeps pk_asr_run, its next stretch not yet heard.

No bits change: a product computes every output element whole in one
item, so the search's arithmetic is the same on one thread as split over
four, and its input is its own chunk's alone.

Expected on test.wav: the searches of chunks 0 and 1, 188 of the 232
tokens (81%), hide behind the encoders of chunks 1 and 2, which take 4
to 14 times as long as the search does on one thread (on the 9950X at 16
threads, about 97 ms against 6.8, and 42 against 11.1). What is lost is
one of the encoder's threads for the search's one-thread time, where
that thread is the encoder's: 22.1 x 0.81 / 16 = 1.1 ms on the 9950X at
16 threads and 2.2 at 8. So on the 9950X at 16 threads 0.81 x 11.8 - 1.1
= 8.5 ms (4%), at 8 6.7 ms (3%), on Windows at 16 about 3.7%; the GB10,
and an M4 Pro on its P-cores, the whole 81%, 3 to 3.4%, if the search's
thread lands on a core the encoder isn't using; the 12900HK at 14
threads 18.5 ms (2.3%); the Pis at 4 threads, where the search on one
thread may take 1.5 to 2 times its four threads' time and a quarter of
the encoder goes for it, 1.5 to 1.9% (Pi 5) and 1.2 to 1.5% (Pi 4). At
one thread level: the search runs after the next chunk's encoder instead
of before it, the same work. A longer recording hides nearly all of its
decoder: test.wav ten times over, in about 25 chunks, 96%.

Risks: beside the encoder, which streams 600 MB of weights a chunk, the
search reads the LSTM's 6.5 MB a token from farther out than its own
22 ms has them, and takes some of the encoder's bandwidth; on the GB10
the scheduler may put the search on an X925 and an encoder thread on an
A725. The memory is a second joint half, 0.96 MB with 30 s chunks.

Kept if the bench moves by more than its spread on the 9950X under two
compilers and on the boards, with none slower by more than its own; the
bits checked as for the fused passes (make check-c; test.wav's --json at
every thread count from 1 to 32; --stream; make check-tsan).

## 2026-10-28 - each chunk's decoder beside the next chunk's encoder (result: kept)

The bits, on the 9950X: test.wav's --json through v2 and v3 the bytes of
test.v2.json and test.v3.json at every thread count from 1 to 32;
--stream through both the bytes of be4b58a's at 1, 2, 3, 8 and 16
threads; long10.wav's --json (test.wav ten times over: 600 s, 22 chunks,
2303 tokens) the bytes of be4b58a's; make check-c on Linux (gcc 15) and
Windows (llvm-mingw; v2), make check-tsan clean, and the TSan build at
1, 2 and 3 threads giving test.v2.json's bytes. On the GB10 (gcc 13.3)
make check-c with both models, make check-tsan clean, and the TSan build
at 1, 2, 3, 10 and 20 threads the same bytes. Every board's builds gave
test.v2.json's bytes before their rounds.

The 9950X, in pinfer's quiet window (13:37-13:42): be4b58a (base) and
this tree (new), test.wav through v2, best of 5, four interleaved rounds
at 16 and 8 threads and three at one; long10.wav two rounds at 16:

| | threads | base | new | new against base, a round |
|---|---|---|---|---|
| Linux, clang 21 | 16 | 0.212 s (1.4%) | 0.205 (1.0%) | -3.53% (-4.23 to -1.90) |
| | 8 | 0.218 (11.0%) | 0.210 (6.7%) | -3.66% (-12.55 to +3.72) |
| | 1 | 1.322 (0.2%) | 1.319 (0.4%) | -0.30% (-0.38 to +0.08) |
| Linux, gcc 15 | 16 | 0.212 (2.8%) | 0.209 (0.5%) | -1.88% (-4.13 to -1.42) |
| | 8 | 0.245 (3.7%) | 0.217 (0.5%) | -11.01% (-12.50 to -9.21) |
| | 1 | 1.388 (0.5%) | 1.390 (0.6%) | +0.14% (-0.79 to +0.29) |
| Windows, clang 23 | 16 | 0.197 (3.1%) | 0.185 (3.2%) | -5.60% (-7.61 to -1.57) |
| | 8 | 0.206 (3.9%) | 0.202 (3.5%) | -3.15% (-4.69 to +0.49) |
| | 1 | 1.296 (0.2%) | 1.294 (0.2%) | -0.15% (-0.15 to -0.08) |
| long10.wav, Linux, clang 21 | 16 | 2.090 (0.2%) | 2.014 (0.3%) | -3.61% (-3.87 to -3.35) |
| long10.wav, Windows | 16 | 1.936 (0.7%) | 1.867 (0.2%) | -3.59% (-3.86 to -3.32) |

(the medians, each with its rounds' range over the median; a round's
change is new's time over base's in that round, their median and range.)
PROFILE=1 builds, two runs each turned, the stages' means, ms a run (gcc
15's in a second window, 13:50):

| | threads | the decoder, base and new | the mel and encoder stages, new less base | the run |
|---|---|---|---|---|
| Linux, clang 21 | 16 | 11.8, 2.8 | -1.8 | 210.5 to 203.5 (-3.3%) |
| Linux, gcc 15 | 16 | 11.8, 2.8 | -1.1 | 217.0 to 210.5 (-3.0%) |
| Windows | 16 | 10.7, 2.2 | +1.6 | 193.0 to 188.0 (-2.6%) |
| Linux, clang 21 | 8 | 10.9, 2.5 | +19.7 | 217.5 to 211.5 (-2.8%) |
| Linux, gcc 15 | 8 | 11.0, 2.6 | +21.0 | 242.5 to 240.5 (-0.8%) |
| Windows | 8 | 9.8, 2.1 | +5.6 | 205.5 to 204.0 (-0.7%) |

The boards, static generic builds as before (arm64: Ubuntu 22.04's
aarch64 gcc 12.3 and clang 21.1.8; storage1: gcc 15.2 and clang 21.1.8;
the M4 Pro its own Apple clang 21 builds), test.wav at the default
threads, interleaved rounds, best of 5:

| board | threads | compiler | base | new | new against base, a round |
|---|---|---|---|---|---|
| GB10, 8 rounds | 10 of 20 | gcc 12 | 0.397 (3.8%) | 0.391 (2.0%) | -2.01% (-3.45 to -1.26) |
| | | clang 21 | 0.395 (2.3%) | 0.384 (3.9%) | -2.77% (-5.06 to -1.53) |
| | long10.wav, 3 | gcc 12 | 4.019 (0.2%) | 3.917 (1.2%) | -2.47% (-2.73 to -1.42) |
| M4 Pro, 6 rounds, in use | 12, the race's | Apple clang 21 | 0.459 (7.0%) | 0.449 (8.2%) | -2.58% (-3.38 to -1.55) |
| | 12 | | 0.463 (7.6%) | 0.449 (8.0%) | -2.57% (-3.44 to -1.62) |
| | 8 | | 0.477 (10.9%) | 0.471 (11.3%) | -1.66% (-2.57 to -1.05) |
| 12900HK (storage1), 6 | 14 | gcc 15 | 0.822 (0.9%) | 0.813 (1.1%) | -1.27% (-1.83 to -1.09) |
| | | clang 21 | 0.817 (1.0%) | 0.806 (0.4%) | -1.22% (-1.71 to -0.98) |
| | long10.wav, 2 | gcc 15 | 8.134 (0.6%) | 8.018 (0.0%) | -1.42% (-1.69 to -1.15) |
| Pi 5 (pi5c), 6 | 4 | gcc 12 | 3.641 (0.3%) | 3.632 (0.3%) | -0.16% (-0.60 to -0.05) |
| | | clang 21 | 3.630 (0.3%) | 3.617 (0.5%) | -0.36% (-0.58 to +0.06) |
| Pi 4, 4 | 4 | gcc 12 | 18.827 (0.7%) | 18.681 (0.8%) | -0.72% (-1.16 to -0.16) |
| | | clang 21 | 18.776 (1.1%) | 18.655 (0.4%) | -0.63% (-1.28 to -0.59) |
| Pi 3, 3 | 4 | gcc 12 | 49.345 (13.0%) | 49.071 (13.0%) | -0.56% (-0.56 to -0.55) |
| | | clang 21 | 49.543 (4.7%) | 49.320 (0.9%) | -0.19% (-0.53 to +3.68) |
| | 15 pairs, single runs | gcc 12 | 50.57 (0.9%) | 50.16 (1.4%) | -0.79% (-1.42 to +0.26), a pair |

PROFILE=1 (gcc 12 on the arm64 boards, clang 21 on storage1, Apple clang
on the M4 Pro), two runs each, the Pi 3's one:

| board | threads | the decoder, base and new | the mel and encoder stages, new less base | the run |
|---|---|---|---|---|
| GB10 | 10 of 20 | 17.4, 3.9 | +2.4 | 401.5 to 390.5 (-2.7%) |
| M4 Pro | 12 | 18.8, 4.2 | +4.6 | 468.5 to 456.5 (-2.6%) |
| | 8 | 18.8, 4.2 | +8.0 | 478.0 to 472.5 (-1.2%) |
| 12900HK | 14 | 25.2, 6.3 | +8.2 | 811.0 to 803.5 (-0.9%) |
| Pi 5 | 4 | 138.4, 28.6 | +122.2, the subsampling +69.0 | 3644.5 to 3630.0 (-0.4%) |
| Pi 4 | 4 | 555.1, 121.6 | +314.3, the subsampling +193.3 | 18835.0 to 18779.5 (-0.3%) |
| Pi 3 | 4 | 1120.6, 250.1 | +581.2, the subsampling +330.1 | 49478.0 to 49198.0 (-0.6%) |

pi5-8gb's rounds are not here: its runs took 4.69 to 5.75 s where they
took 4.15 to 4.20 this morning, with spreads of 7 to 19% and rounds
from -14% to +10%, nothing else running and the board not throttled.

pi3 reached its soft temperature limit in the first round (70 C on this
board, throttled=0x80008), where its firmware holds the ARM clock at 1.2
GHz; the round's first builds ran partly at 1.4 GHz, which the ranges
carry. After the rounds, fifteen pairs of single runs (-v's
transcription time), base and new under gcc 12, the pair's order turned
each time, the ARM clock read every half second through each
(vcgencmd measure_clock, 94 to 96 samples a run): every sample read
1200 MHz, so the times are the cycles' measure as well. New was faster
in 14 of the 15 pairs, -0.75% at the mean (sd 0.43).

What the change takes away is the same everywhere: the decoder's wait
falls to a fifth or a quarter of what it was, the last chunk's 44 tokens
(19%) and the joint halves. What it costs is where the pool's last thread
is one the encoder had, and where the search's reads (the LSTM's 6.5 MB
a token, from DRAM once a layer's weights have passed through the
caches) share a memory path with the encoder's. The GB10's last thread
is one its X925s leave idle (its race keeps 10 of 20), and its encoder
lost 2.4 ms; the 9950X at 16 threads, across two CCDs, lost none. At 8
threads, on one CCD, its encoder lost 5.6 ms on Windows and 20 under
WSL, two thirds of it in the feed-forward, the products most bound by
bandwidth. The M4 Pro at 8 threads, its P-cores, lost nearly twice what
it did at 12. A Pi 5's four cores give the search one of the encoder's,
and its encoder lost more than the 110 ms the decoder gave back, most of
it in the subsampling, the stage beside which each search starts; its
best of 5 is 0.2 to 0.4% ahead and its stages' means 0.3% behind: level.
A Pi 4's encoder lost 314 ms of the 434 its decoder gave back, 193 of
them in the subsampling, and every round was faster, by 0.6 to 0.7% at
the medians, inside spreads of 0.4 to 1.1%. A Pi 3's lost 581 of 871
ms, 330 in the subsampling, and at its held clock it is 0.8% ahead. At
one thread level on every system, as expected. Longer recordings keep
the gain: long10.wav 3.6% on the 9950X, 2.5% on the GB10, 1.4% on the
12900HK.

Against the hypothesis: the 9950X at 16 threads 1.9 to 5.6% where it
expected 4% (3.7% on Windows); at 8, 0.7 to 2.8% by PROFILE where it
expected 3%. The gcc build's -11.01% at 8 threads is placement, not the
change: in the PROFILE window both gcc builds took about 0.24 s at 8
threads, the decoder's 8.4 ms taken away in each and 21 given back by
the encoder. The GB10 2.0 to 2.8% (3 to 3.4% expected), the M4 Pro 1.7
to 2.6% (3%), the 12900HK 1.2 to 1.3% (2.3%), the Pis 0.2 to 0.8% (1.2 to
1.9%): the estimate counted a thread's share of the encoder for the
search's time, and left the memory the two share to its risks.

Kept: at 16 threads on the 9950X the runs moved by more than their
spreads under clang on both systems (Linux -3.53% against spreads of 1.0
and 1.4%, Windows -5.60% against 3.1 and 3.2%), and under gcc every
round was faster, by 1.42% or more; long10.wav by 3.6% against spreads
under 1%; the 12900HK's by more than its spreads; every round faster on
the GB10 and the M4 Pro; the Pis level or a little ahead; and nothing
slower by more than its spread.

## 2026-10-28 - blocks of 64 at the caller's choice: a fast mode (hypothesis)

The encoder's activations are int8 in blocks of 32 (README, "Why int8"):
every block's integer sum ends in a convert and a fused multiply-add, and
every block costs the quantizer an absolute maximum and two divisions.
Blocks of 64 halve both, and move transcripts further from float32's,
which is why the default stays 32.

What they cost, measured before this (2026-10-28). tools/quantcheck.py,
the encoder's activations at 64 paired against the engine's 32: v2 over
LibriSpeech's 200 recordings dlogp +0.002 (t +0.8), 3 transcripts
different; v3 over FLEURS' 360 -0.006 (t -1.2), 26 different; the
encoder's error 4.1% against 3.6% and 5.8% against 5.3%. The engine
itself, core/encoder.c's QB at 64 on 33c5e7c (a scratch build), through
tests/wercheck.py against quantcheck's float32:

| | v2: WER | differ from float32 | v3: WER | differ from float32 (WER between) |
|---|---|---|---|---|
| blocks of 32 (33c5e7c) | 1.51% | 4 of 200 | 4.42% | 45 of 360 (0.35%) |
| blocks of 64 | 1.47% | 7 | 4.43% | 49 (0.50%) |

What they buy: the same scratch build against 33c5e7c, test.wav through
v2, best of 5 at the default threads, four interleaved rounds, the order
turned, each base build's --json test.v2.json's bytes:

| board | threads | compiler | 33c5e7c | blocks of 64 | a round's change, median (range) |
|---|---|---|---|---|---|
| 9950X, Linux | 16 | gcc 15 | 0.209 s | 0.198 | -5.5% (-5.7 to -3.9) |
| | | clang 21 | 0.204 | 0.1945 | -4.4% (-5.9 to -3.4) |
| 12900HK (storage1) | 14 | gcc 15 | 0.809 | 0.764 | -5.6% (-6.2 to -5.1) |
| | | clang 21 | 0.8055 | 0.7595 | -5.8% (-6.0 to -5.0) |
| GB10 | 10 | gcc 13 | 0.388 | 0.372 | -3.7% (-4.6 to -3.1) |
| | | clang 18 | 0.3815 | 0.3655 | -4.6% (-4.7 to -2.7) |
| M4 Pro, in use | 12 | Apple clang 21 | 0.422 | 0.401 | -5.0% (-5.4 to -4.1) |
| Pi 5 (pi5c) | 4 | gcc 12 | 3.628 | 3.440 | -5.2% (-5.4 to -5.0) |

And the int8 product alone: q8bench on one core (taskset), the
encoder's three shapes (4096 x 1024, 1024 x 4096, 1024 x 1024) at 256
columns in 64-row slices, best of 5, three interleaved rounds, blocks of
64 against 32, the range over the shapes' medians:

| core | the tile at 32 | at 64 | 64 against 32 |
|---|---|---|---|
| Zen 5 (9950X) | zmm, unrolled (as now) | the same, unrolled | +8 to +12% (gcc 15), +5 to +7.5% (clang 21) |
| Zen 5, PKTITO_KERNELS=avx2 | VPMADDUBSW, unrolled | the same | -5 to -6% (gcc 15), +3 to +4% (clang 21) |
| Golden Cove (12900HK) | AVX-VNNI, unrolled | the same | +1 to +2% (gcc 15), -0.7 to +1.1% (clang 21) |
| M4 Pro P-core | SDOT, the assembly | the C tile | +8 to +13% (Apple clang 21) |
| X925 (GB10) | | | +3.5 to +10% (gcc 13, clang 18) |
| A76 (Pi 5) | | | +5 to +7% (gcc 12) |
| A72 (Pi 4) | ARMv8.0, the assembly | the C tile | -4 to -8% (gcc 14); the C tile at 64 against its own 32, +8 to +10% |
| A53 (Pi 3) | | | -2 to 0% (gcc 14); the C tile's, +4 to +5% |

The hypothesis: a fast mode, pk_asr_set_fast_mode(a, 1) and pktito-c's
--fast, for a caller who wants the speed more than float32's nearest
transcripts. Off by default, and the default's bits are 33c5e7c's. In it
the encoder's activations are in blocks of 64 wherever a product's input
is whole blocks of 64 (every one of Parakeet's; the tiny model's 32
subsampling channels keep 32), and the decoder's stay a row each.
core/encoder.c's QB becomes a choice the transcriber makes from that.
The slices the fused passes quantize, 64 and 256 of a product's rows,
and the depthwise items' 64 channels are whole blocks of 64 already.
Each x86 tile's unrolled block of 32 becomes a macro, and a block of 64
two of it, which is what the table measured; NEON with dotprod takes
blocks of 64 through its C tile, as measured; the ARMv8.0 loop at 64 is
backend/neon/CLIMB.md's next entry, and until it the A72 and the A53 run
the C tile at 64, slower than their default. AMX's pipeline takes blocks
of 32 only, and at 64 the amx-int8 build runs its zmm tile, unmeasured.
Every backend computes ref's bits in each mode: kcheck learns blocks of
64, and make check-c checks --fast's test.wav against ref's
(tests/data/test.v2.fast.json and test.v3.fast.json).

Predicted: --fast against the default as the whole-run table above,
within a point; the default against 33c5e7c level, its bits the same.
The VPMADDUBSW tile's -6% under gcc is to be looked at before the
commit. Kept if --fast is faster than the default by more than the
spread on every board with a dotprod or a VNNI tile, under both
compilers, and the default is level with 33c5e7c.

## 2026-10-28 - the VPMADDUBSW tile's sums kept a chain under gcc (hypothesis)

The fast mode's entry above found the avx2 tile 5 to 6% slower at blocks
of 64 than at 32 under gcc 15, and 3 to 4% faster under clang 21. Its
code (gcc -O3 -S, k_gemm_q8, the four-column tile's block): every
group's VPMADDWD product goes to the stack and the block's 32 (64) are
summed at its end as a tree, 40 stack references in a block of 32 and
104 in a block of 64 - the same in gcc 11.4, 14.3 and 15.2, and with
-fno-tree-reassoc - where clang keeps each column's sum a chain in a
register and has 0 and 2. The sums are int32 and wrap, so every order
gives the same bits, and gcc is free to pick its own; at 32 it costs
gcc's tile, too: q8bench on a Zen 5 core, avx2, 120 GMAC/s under gcc 15
against clang's 130 (the same entry's run).

The hypothesis: an empty asm on each column's sum after each group
(`__asm__("" : "+x"(s[c]))`), which leaves nothing in the instruction
stream but makes the sum a value gcc has to hold in a register there,
so that the block's adds stay the chain they are written as. In a
scratch copy gcc 11, 14 and 15 then have no stack reference in any of
the tile's blocks and 190 instructions in the four-column block of 32
(from 209) and 382 in the block of 64 (from 429); clang's instructions
are the ones it had. The bits are the same: kcheck. Predicted, avx2
forced on the 9950X and on the 12900HK: q8bench under gcc 15 to within
2% of clang's at 32 and at 64, clang's level; a whole run at the
default threads, gcc's faster by the product's gain in the products'
share of the run (make PROFILE=1), clang's level; and the fast mode
then faster than the default under both. Kept if gcc's product and run
move by more than the spread and clang's do not fall. The Zen 2 and
Zen 3 instances, where clang 21's avx2 build was 4.5 to 5.2% ahead of
gcc 11's (the release builds' entry, 2026-10-27), are the machines this
is for, and are not measured here.
