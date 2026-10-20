# pkito

NVIDIA's Parakeet TDT speech recognizers (0.6B v2 English, v3 multilingual)
in C11, for anything with a C compiler, written to be read. The reader is
someone putting speech-to-text on a small device - a Pi, an old laptop, a
board with no network - or learning how a transducer runs, with the C
library and nothing else. NeMo already exists for everyone with Python,
PyTorch and a GPU; pkito is the version you can read end to end, that gives
NeMo's transcripts, and that gets as much out of the board as the silicon
allows.

## What decides a change

1. **Does NeMo agree?** The answer is NeMo's transcript. The float64
   reference (tools/pkito_tools/reference.py) gives NeMo's token ids and
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
   backend is picked by `make`, the numerics by measurement, the thread count
   from the cores, the chunking by the VAD. A flag exists only for what the
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
- backend/neon serves neon and neon-dp, backend/avx2 serves avx2 and
  avx2-vnni (the second of each pair includes the first with a macro). The
  int8 product is the only kernel that differs inside a pair.
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
  tests/kcheck passing, and kept only if `./pkito bench` moves by more than
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

## Checking

- `make check-c` before every commit, on every backend touched, on its board.
- `make check` where Python and NeMo's checkpoints are, before every merge.
- `make check-sanitize` for any change to core/model.c or core/wav.c.

## Working

- Commit at milestones, with the numbers in the message.
- The boards: an i9-12900HK (AVX2, AVX-VNNI) at phebert@192.168.86.119, a
  Pi 3B+ (Cortex-A53, 905 MB) at phebert@192.168.86.155, a Pi 5 (Cortex-A76,
  4 GB) at phebert@192.168.86.79, and the M4 Pro here, the one machine that
  is also in use while measuring. One job at a time on a board when
  measuring.
- When a request comes with a reason, weigh the reason. If it points
  somewhere other than the literal ask, say so.
