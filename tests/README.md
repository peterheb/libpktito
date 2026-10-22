# The checks

Each check holds one claim the README makes to the thing it is a claim
about. `make check-c` runs the ones that need only a compiler and a model;
`make check-python` the ones that compare with sentencepiece, Silero's
Python, torchaudio or the corpora, each skipped with a line when what it
needs is not there.

| check | in | holds |
|---|---|---|
| `mathcheck` | check-c (`--all`: check-math) | core/vmath.h's exp, tanh, sigmoid and log to the double result over every float, its sin and cos over the positional encoding's angles, and sqrt(x * x) == \|x\| wherever x * x is normal |
| `kcheck` | check-c | every kernel of every backend in the build that the CPU can run to backend/ref's bits (the rest skipped with a line), on random and awkward data and on the model's real matrices, and ref's int8 product to its definition |
| `pkfcheck` | check-c, check-sanitize | the model file's validator and binder on tests/data/tiny.pkf and 29,104 malformed versions of it, and the whole engine on the tiny model (docs/MEMORY_SAFETY.md) |
| `wavcheck` | check-c, check-sanitize | the WAV reader on every prefix of a valid file and each field of fmt wrong |
| `capcheck` | check-c | core/captions.c to docs/caption-spec.md's cases (a cue's layout and times, each boundary's costs, graphemes, the 4 + 5 it finds where greedy packing finds 8 + 1), and its dynamic program to the best of every partition of 3000 random transcripts |
| test.wav | check-c | `./pktito probs` against tests/data/test.probs.txt (vadito's bits) and `./pktito --json` against test.v2.json and test.v3.json (ref's transcripts, every token and time), byte for byte |
| `examples/transcribe` | check-c | the library alone gives the CLI's transcript |
| `check-tsan` | check-tsan | the pool and every pass split over it: no race on six threads |
| `tokcheck` | check-python | core/tokens.c against sentencepiece's decode_pieces and NeMo's punctuation rule, every piece alone and 3000 random sequences, for each model |
| `segcheck` | check-python | core/segment.c's two state machines against Silero's Python on 3000 generated probability sequences, segments to the sample and events to the frame |
| `rscheck` | check-python | core/resample.c against torchaudio's Resample on thirteen rate pairs |
| `wercheck.py` | check-python | ./pktito's WER and transcripts over tools/corpus.py's LibriSpeech and FLEURS sets against float32's (tools/quantcheck.py's) |
| `q8bench` | `make benches` | the int8 product alone, one thread, on the encoder's shapes and the decoder's, for a kernel climb (backend/avx2/CLIMB.md) |
| `gemmbench` | `make benches` | the same over threads, sliced and dealt as pk_gemm deals it, in MAC a cycle at the core's clock (backend/neon/CLIMB.md) |

tests/data/test.wav is Silero's own test recording (MIT, from the
silero-vad repository), a minute of English lecture; test.probs.txt is the
VAD's probabilities for it as vadito writes them; test.v2.json and
test.v3.json are `ref`'s transcripts of it. tiny.pkf is `pktito-tools tiny`'s
output, a 1.2 MB model with random weights and every tensor, param and
piece type.

When a change to arithmetic is meant - a rounding, an order, a kernel's
definition - test.v2.json and test.v3.json are regenerated from `ref`
(`./pktito --json -m MODEL tests/data/test.wav`), and the change carries its
`wercheck.py` numbers against the run before (CLAUDE.md, "Measuring").
