# pktito

NVIDIA's Parakeet TDT speech recognizers (0.6B v2 English, v3 multilingual)
in C11, for anything with a C compiler, written to be read. The reader is
someone putting speech-to-text on a small device - a Pi, an old laptop, a
board with no network - or learning how a transducer runs, with the C
library and nothing else. NeMo already exists for everyone with Python,
PyTorch and a GPU; pktito is the version you can read end to end, that gives
NeMo's transcripts, and that gets as much out of the board as the silicon
allows.

## What decides a change

1. **Does NeMo agree?** The answer is NeMo's transcript. The float64
   reference (tools/pktito_tools/reference.py) gives NeMo's token ids and
   timestamps; the engine's int8 numerics were chosen by measurement against
   float32 (README, "Why int8") and every backend computes the same bits as
   `ref` on every machine. A change that moves a transcript off that, or a
   bit between backends, is a bug, whatever it buys.
2. **Can a learner follow it?** The tree reads top down: core/encoder.c and
   core/decoder.c are the model, core/mel.c the features, backend/ref the
   arithmetic. Code outside the tuned backends is written for a reader first,
   and a clever form has to buy something measurable.
3. **Is it the most the hardware can give?** A tuned backend's kernels.c is
   the one place allowed to be dense, on measurements (see Measuring).
4. **Does it hand the user a decision they can't make well?** Don't. The
   backend is picked by the CPU at run time from every one the build holds
   for its family (one binary per family: a copy that lands on another
   machine of it still runs, at that machine's best), the numerics by
   measurement, the thread count from the cores, the chunking by the VAD. A flag exists only for what the
   user knows and the engine cannot: their file, their model, their
   language's model.
5. **Does it need anything?** The C library, libm and pthreads. No network,
   no allocation the caller cannot bound. Conversion is offline Python whose
   only dependency is numpy.

## Code

- C11. `core/` never contains an intrinsic; everything that touches a weight
  goes through backend/kernels.h, and backend/ref is the definition of
  correct. The other backends match it bit for bit on every kernel - not to
  a tolerance - which tests/kcheck holds them to; keep it so.
- `-ffp-contract=off` everywhere: a fused multiply-add is written as `fmaf`
  where it is meant, and nowhere else. core/vmath.h's functions are the only
  transcendentals that reach a transcript; libm's `exp`, `log`, `sin` are not
  the same bits on every platform.
- backend/neon serves neon and neon-dp, backend/avx2 serves avx2, avx2-vnni
  and avx512-vnni (each variant a file that includes the first with a macro,
  compiled with its own instruction set; backend/kernels.c picks among
  them). Inside a family a variant replaces a kernel only where it measures
  faster: the int8 product in each, and in the AVX-512 builds the float
  kernels in zmm - each to ref's bits, like every kernel.
- Comments explain *why* and carry the number that justified it, and say
  where the number came from (a CLIMB.md row, a check), because a bare
  figure goes stale when the code under it changes.
- Comments and commit messages describe the code and its measurements as
  they stand. A reader has the tree, not the conversation that produced it.
- backend/*/CLIMB.md are dated records. Append to them; do not rewrite old
  rows, and keep the rejected ones.
- The model file and WAV parsing follow docs/MEMORY_SAFETY.md; a format
  change extends its validator and its corpus in the same change.

## Measuring

- A kernel change: the hypothesis in the backend's CLIMB.md before the edit,
  tests/kcheck passing, and kept only if `./pktito-c bench` moves by more than
  the run-to-run spread on a quiet board (`make remote HOST=` runs both
  there). `make PROFILE=1` says which stage moved.
- A change to arithmetic - a rounding, an accumulation order, a block size,
  a kernel's definition - changes `ref`'s bits, so it reruns
  tests/wercheck.py over tools/corpus.py's LibriSpeech and FLEURS sets and
  records the WER and the transcripts that changed against the run before.
- A quantization choice comes from tools/quantcheck.py: paired dlogp, the
  transcripts that change, and WER, against float32 on the same recordings.
- A finding is true of the board, compiler and code it was measured on.
  Before relying on one after any of those changed, measure again.
- A tuned kernel's change is benched under two compilers where the board
  has them (two gcc majors, or gcc and clang), and one that wins under one
  and loses under the other is not finished: gcc 15 spilled a loop gcc 13
  kept in registers (backend/avx2/CLIMB.md, amx-int8's pipelined loop).

## Checking

- `make check-c` before every commit, on every backend touched, on its board.
- `make check` where Python and NeMo's checkpoints are, before every merge.
- `make check-sanitize` for any change to core/model.c or core/wav.c.

## Working

- Commit at milestones, with the numbers in the message.
- The boards: an i9-12900HK (AVX2, AVX-VNNI), a Pi 3B+ (Cortex-A53,
  905 MB), a Pi 5 (Cortex-A76, 4 GB; pi5a), a DGX Spark (GB10: ten
  Cortex-X925 and ten Cortex-A725; a vLLM worker lives there too), a
  Snapdragon 7c laptop (Windows 11 on ARM: two Cortex-A76, six A55; 7.6 GB;
  armlet), and the M4 Pro here, the one machine that is also in use while
  measuring. Their logins and addresses are in CLAUDE.local.md, which is
  not committed. One job at a time on a board when measuring.
- When a request comes with a reason, weigh the reason. If it points
  somewhere other than the literal ask, say so.
