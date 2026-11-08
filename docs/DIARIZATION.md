# Who spoke when

pktito writes what was said. An interview or a podcast also needs who said
it: Speaker 1, Speaker 2, each word given to the voice that said it. This is
the study of doing that in pktito - which of NVIDIA's diarizers, how close to
NeMo it can be held, what it costs on the boards, how its output meets
Parakeet's words, and in what order to build it. It was written on
2026-11-08 on a Ryzen 9 9950X under WSL2, which is not one of the lab
boards. Nothing in it is in the engine yet; the prototype is Python:

- `tools/pktito_tools/sortformer.py`: NVIDIA's Streaming Sortformer v2 in
  float64 numpy, NeMo 3.0's arithmetic, what reference.py is to Parakeet.
  `tools/pktito-tools diar model.nemo talk.wav` prints its RTTM.
- `tools/pktito_tools/speakers.py`: words to speakers, NeMo's rule and the
  one alternative measured below.
- `tools/pktito_tools/reference.py`: the three things the diarizer's encoder
  has and Parakeet's lacks - a bias on its matrices, xscaling, features that
  are not normalized - each off for Parakeet, whose features, encoder output
  and tokens on the first 12 s of test.wav are unchanged bit for bit.

A number here is one of three kinds and says which. Measured: on this
machine, in this study. Published: a model card's, read on 2026-11-08.
Inferred: reasoned from the other two, and to be measured before anything
relies on it.

## In short

- **The model: Streaming Sortformer v2** (nvidia/diar_streaming_sortformer_4spk-v2,
  CC-BY-4.0, 117.7M parameters, up to four speakers). Its network is the
  FastConformer pktito already runs - 17 layers of 512 where Parakeet has 24
  of 1024 - and a small Transformer after it; the released NeMo runs it; and
  the float64 reference gives NeMo's labels on seven recordings up to an
  hour long, all but 9 of 469,856. Its successor, Nemotron 3 Diarization
  (eight speakers, a third less DER on NVIDIA's own benchmarks), needs a
  NeMo that is not released; the plan builds everything around the network
  so that it carries over.
- **The cost: about half of Parakeet's time again** (inferred from pktito's
  own encoder measured at the diarizer's shapes, 47 ms a step on the 9950X):
  a Pi 5 would transcribe and label an hour in 5 to 6 minutes, a Pi 3B+ in
  63 to 72. Memory: a 120 MB file mapped beside the recognizer's and 20 to
  40 MB of work.
- **The merge: NeMo's rule**, each word to the speaker most probable over
  its frames. On synthetic conversations of two to four speakers, 5 minutes
  to an hour: 15 of 14,017 words to the wrong speaker. A fallback for words
  the diarizer heard no one say takes that to 4, and 20 changes of speaker
  that were not there to 4, but departs from NeMo: a question for Peter.
- **The plan:** the reference kept and the numerics measured, as Parakeet's
  were; C on backend/ref with a .pkf of its own; the tuned backends, which
  need no new kernel; Nemotron 3 when NeMo releases it.

## 1. The candidates

NVIDIA's diarizers for NeMo on Hugging Face (its API searched for them,
2026-11-08), each from its card, configuration and file listing:

| | released | licence | parameters | speakers | network | a long recording |
|---|---|---|---|---|---|---|
| diar_sortformer_4spk-v1 | 2024-12-09 | CC-BY-NC-4.0 | 123M | 4 | 18 FastConformer layers, 18 Transformer layers | whole, at once: "around 12 minutes" on a 48 GB GPU |
| diar_streaming_sortformer_4spk-v2 | 2025-06-04 | CC-BY-4.0 | 117.7M | 4 | 17 FastConformer layers of 512, 18 Transformer layers of 192 | streamed, a speaker cache: "several hours" |
| diar_streaming_sortformer_4spk-v2.1 | 2025-10-22 | NVIDIA Open Model License | 117M | 4 | v2's (inferred: its .nemo is v2's size to the byte, 471,367,680) | as v2 |
| Nemotron-3-Diarization | 2026-09-01 | OpenMDW-1.1 | 99.2M | 8 | 31 Transformer layers of 512 with rotary positions, 8 feature frames stacked into one, an upsampler to 10 ms | streamed, a speaker cache |
| TitaNet-large and clustering | 2022-07-15 | CC-BY-4.0 | 23M | any | speaker embeddings over VAD segments, spectral clustering | yes |

And one that is not a diarizer: multitalker-parakeet-streaming-0.6b-v1
(2025-10-15, NVIDIA Open Model License) is a recognizer that writes each
speaker's words on its own, taking Sortformer's output as an input. It would
replace Parakeet rather than sit beside it.

Published DER (missed speech, false alarm and confusion, percent of speech;
overlap included). Each card measures with its own protocol - its
reference labels, its post-processing - so compare within a table, not
across:

v2.1's card, the 30.4 s configuration, no post-processing (v1's from its own
card, likewise):

| | DIHARD III, 1-4 speakers | DIHARD III, all | CALLHOME, 2 | CALLHOME, 4 | CALLHOME, all | AMI, far microphone | NOTSOFAR1, far, all |
|---|---|---|---|---|---|---|---|
| v1 | 16.28 | | 6.49 | 14.14 | | | |
| v2 | 14.63 | 19.68 | 6.27 | 12.30 | 10.50 | 28.56 | 33.43 |
| v2.1 | 14.84 | 19.49 | 5.65 | 12.33 | 10.10 | 17.80 | 27.07 |

Nemotron 3's card, the 30.4 s configuration, v2.1 measured again beside it:

| | DIHARD III, 1-4 speakers | DIHARD III, all | CALLHOME, 2 | CALLHOME, 4 | CALLHOME, all | AMI, far microphone | NOTSOFAR1, far, all |
|---|---|---|---|---|---|---|---|
| v2.1 | 13.98 | 19.09 | 5.68 | 12.36 | 10.32 | 21.42 | 30.49 |
| Nemotron 3 | 9.13 | 12.73 | 5.98 | 11.03 | 9.10 | 11.14 | 11.00 |

TitaNet's card publishes DER only with the true speech segments given (an
"oracle VAD", 1.2 to 6.7%), which no recording comes with, so it is not in
the tables. As a pipeline it is a VAD, embeddings at several scales and an
eigendecomposition, with no single NeMo answer to hold C to; it is what one
would use for more than eight speakers, and nothing here needs that.

v1 is out on its licence. NVIDIA also publishes q8_0 GGUF files of v2 (147
MB) and Nemotron 3 (107 MB), and Nemotron 3's card runs it with
NeMo-Speech.cpp, NVIDIA's ggml runtime in C++ (Apache-2.0, begun
2026-07-15): the models run on a CPU without PyTorch, but not in C11 on the C
library alone, and not with the same bits on every backend.

## 2. Which, and why

**v2**, for reasons in CLAUDE.md's order:

1. **NeMo agrees with it today.** NeMo 3.0.0 (2026-08-07, still the latest
   release on PyPI and GitHub on 2026-11-08) runs v2, and the reference
   gives its labels (section 4). Nemotron 3's attention is rotary, and NeMo
   3.0.0's TransformerEncoder has none (its attention models are abs_pos,
   rel_pos and no_pos; "rope" and RotaryPositionalEncoding are on NeMo's
   main branch only), nor a learned silence frame in the speaker cache,
   which Nemotron 3's weights carry (`silence_embeds`) and only NeMo's main
   branch reads (`use_learnable_sil_emb`). Its card's `pip install
   'nemo-toolkit[asr]'` installs 3.0.0. Holding pktito to an unreleased
   branch is holding it to a moving target.
2. **A reader already knows most of it.** The 17 layers are core/encoder.c's
   FastConformer with a bias on each matrix and the input scaled; what is
   new is an 18-layer Transformer of the textbook kind (post-LN, ReLU), a
   two-layer head, and the speaker cache, a page of plain C.
3. **Its licence is Parakeet's**, CC-BY-4.0.
4. **Its limits are known**: four speakers, and section 8 shows what a fifth
   does.

What v2 gives up is Nemotron 3's: eight speakers, and accuracy where
recordings are hard - DIHARD III 12.73 against 19.09, far-field meetings
(NOTSOFAR1) 11.00 against 30.49. Two to four people on a telephone line are
close (CALLHOME: 5.98 against 5.68 for two, 11.03 against 12.36 for four).
v2.1 is v2's code with other weights, much better in meetings, under a
licence Peter would have to read. So the plan keeps the network a module:
the cache, the merge, the outputs, the file and its checks are the same for
all three, and Nemotron 3's network replaces v2's when NeMo releases it.

## 3. What the engine would compute

Streaming Sortformer v2 as NeMo 3.0 runs it (SortformerEncLabelModel's
forward_streaming and SortformerModules' streaming_update and
_compress_spkcache, at batch size one; the configuration and the tensors
read from the checkpoint):

- **features**: Parakeet's log-mel (128 mels, 25 ms Hann windows every 10
  ms) without its per-feature normalization (`normalize: NA`). A frame's
  features then depend on its own samples alone, and a recording can be
  featurized a chunk at a time.
- **pre-encode**: Parakeet's subsampling (three stride-2 convolutions of 256
  channels, to 80 ms frames, and a matrix to 512), a chunk at a time: 340
  frames (27.2 s) with 1 frame of left context and 40 of right, each chunk's
  edges padded as its own.
- **a step**: the speaker cache (at most 188 frames), the FIFO (at most 40)
  and the chunk with its context (at most 381) - at most 609 frames -
  through 17 FastConformer layers (512 wide, 8 heads of 64, feed-forward
  2048, depthwise kernel 9, relative positions; every matrix but the
  positions' with a bias; the input first scaled by the square root of 512),
  a matrix to 192, an 18-layer Transformer (post-LN, x = LN(x + attention(x))
  then x = LN(x + ff(x)); 8 heads of 24, a ReLU feed-forward of 768, no
  positions of its own), and a head (ReLU, 192 x 192, ReLU, 192 x 4,
  sigmoid): four probabilities every 80 ms, one a speaker.
- **the update**: the chunk's frames join the FIFO; when it overflows, its
  oldest 300 or more join the cache, which is then cut back to 188: three
  frames a speaker of the recording's mean silence, and the 176 best-scored
  frames, a frame's score for a speaker being how much likelier it is that
  speaker alone than not. Frames that are not the speaker's speech never
  qualify; each speaker's best 33 are raised a lot and its next 33 a little,
  so that every speaker keeps frames and none takes them all; the newest
  frames get a little extra.

The speakers come out in the order they first spoke, column 0 whoever spoke
first, and the cache is what keeps that order over an hour: each step
attends to frames of every speaker heard so far. The lower latencies are the
same computation with shorter chunks: the card's 1.04 s configuration takes
chunks of 6 frames, 7 of right context and a FIFO of 188.

## 4. The reference against NeMo

Measured: sortformer.py against NeMo 3.0.0's own forward() (torch 2.14.1,
CPU, float32), the 30.4 s configuration. A label is a probability over one
half; the conversations are section 8's.

| | speakers | length | steps | most apart, first chunk | anywhere | labels not NeMo's | DER against NeMo's labels |
|---|---|---|---|---|---|---|---|
| test.wav | 1 | 60 s | 3 | 1.5e-6 | 1.6e-6 | 0 of 3,000 | 0 |
| conv2 | 2 | 5.0 min | 12 | 8.5e-7 | 1.9e-6 | 0 of 15,016 | 0 |
| conv3 | 3 | 10.2 min | 23 | 4.8e-7 | 6.3e-3 | 0 of 30,608 | 0 |
| conv4 | 4 | 10.1 min | 23 | 9.4e-7 | 2.7e-6 | 0 of 30,260 | 0 |
| conv5 | 5 | 10.2 min | 23 | 3.8e-7 | 0.094 | 3 of 30,560 | 0.05% |
| conv2h | 2 | 60.1 min | 133 | 6.4e-7 | 0.016 | 1 of 180,304 | 0.003% |
| conv3h | 3 | 60.0 min | 133 | 5.9e-7 | 0.038 | 5 of 180,108 | 0.01% |

Every step here ends in a cut of the cache. In the first chunk the
reference is within 1.5e-6 of NeMo, as reference.py is of Parakeet's encoder
(2e-6). After a cut the two can part: the cut keeps the 188 best-scored
frames, and a score that float32 rounds one way and float64 the other
changes which frame is kept, after which every later step attends to a
slightly different cache. The differences stay small - at most 0.094, on the
recording with one speaker too many - and moved 9 labels in 469,856. The
discontinuity is the model's: the engine will differ from NeMo's float32 as
float64 does, more with int8. So the diarizer is held to NeMo by its labels,
DER and the words that change speaker, as the transcripts are held by WER
and the transcripts that change, not by a tolerance on its probabilities.

## 5. Ties

The cut is torch.topk over every speaker's scores, and on a CPU torch
returns equal values in an order of its own. Measured with torch 2.14.1: the
3 largest of 10 zeros are indices 6, 7 and 8; the 188 of 1,000 zeros start
at 501; the 188 of 40,000 are the lowest. Where scores tie, NeMo's cache is
whatever torch's partition did, and that could change with torch's version
or the length. The reference keeps the lower index (a stable sort), and C
would do the same, with a comparator that orders equal scores by their
index, since qsort is not stable. That also keeps every backend's cache the
same: the probabilities are ref's bits on every backend, and the cut is
plain C after them.

It has not mattered: no tie decided any of the 350 cuts above (the
reference counts them). A score is continuous in probabilities that never
saturated in NeMo's float32 output (the largest was 0.99995), so a tie needs
a coincidence. The infinities tie by design and harmlessly: a silence pad's
+inf is always kept, and a disqualified frame's -inf, if kept, becomes a
silence frame whichever it was.

## 6. Into pktito

### Kernels: none new for v2

Each operation and the kernel that already has it (backend/kernels.h):

- every matrix: `pk_k_gemm_q8`, which takes a bias and adds into a residual
  stream. Rows come in 16s and inputs in 32s: the FastConformer's (512, 1024,
  2048) and the Transformer's (192, 768) fit; the head's last matrix (4
  rows) is padded to 16 at conversion.
- the attention's products: `pk_k_gemm_f32`. Its rows come in 16s too, and
  the Transformer's heads are 24 wide, so the product with the values has 24
  rows: core/ lays each head's values out as 32, the last 8 zero, and drops
  the 8 outputs they make; no sum that is kept changes.
- NeMo's Transformer divides its queries and keys each by the fourth root of
  24; the converter folds that into their matrices and `pk_k_softmax` takes
  a scale of 1.
- layer norm (512 and 192, both multiples of 16), softmax, SiLU, GLU, the
  depthwise convolution (512 channels) and the subsampling's (256): the
  kernels Parakeet's encoder uses.
- ReLU is exact in any order, so it is a loop in core/.
- the head's sigmoid and the cut's logarithms: core/vmath.h's `pk_sigmoidf`
  and `pk_logf`, so that a cut decides the same on every machine.

Nemotron 3 would add GELU, which needs erf (a core/vmath.h function, and
its sweep in tests/mathcheck), the rotation of rotary positions (sines and
cosines: a table the converter writes, or `pk_sincosf`), the stacking of 8
feature frames (a reshape before a matrix, 1024 to 512), and its upsampler
(a convolution over 3 frames, 192 to 1536: a matrix).

### Code

- core/encoder.c's FastConformer takes its weights and shape from a struct
  both models fill, its bias pointers NULL and its scale 1 for Parakeet, so
  that the recognizer computes the same bits (check-c's test.v2.json and
  test.v3.json hold it to that).
- core/mel.c: the features without the normalization, a chunk at a time.
- core/diar.c (new): a chunk's pre-encode, the step - encoder, projection,
  Transformer, head - and the cache, written to be read beside sortformer.py.
- core/speakers.c (new): a word's speaker from its frames, and the turns.
  Samples to frames in integers, every float sum in one order.
- core/pktito.h: a diarizer, `pk_diar_new(model, threads)`, `pk_diar_feed`
  (samples in; a chunk's probabilities out once its right context has
  come), `pk_diar_free`; and `pk_speakers` (words and probabilities in, a
  speaker a word out).
- the CLI: section 8's outputs.

### The model file

A .pkf of its own, "sortformer" its architecture:

- **params**: the shapes (layers, widths, heads, kernel, channels, the
  Transformer's, the speakers), the scale, and the cache's numbers - chunk,
  contexts, FIFO, cache, update period, silence frames, thresholds, boosts -
  for the 30.4 s configuration.
- **tensors**: the FastConformer's under Parakeet's names, each bias as
  `<name>.b` (sortformer.py's `tensors()` names them so already); the
  projection, the Transformer's `t<i>.`, the head.
- **no vocabulary**: the validator requires 1 to 2^20 pieces today; for this
  architecture it would require none.
- **metadata**: the source, the licence with its attribution, the
  converter's version, as Parakeet's files have.

It is separate because 120 MB is a lot to download and map for someone who
never wants speakers, and because one diarizer serves both recognizers (it
hears voices, not a language; its training data include Mandarin and
CALLHOME's several languages).

docs/MEMORY_SAFETY.md's contract, in the same change: `pk_pkf_validate`
takes the second architecture; the binder reads every param with its range
(speakers 1 to 8, a cache of at least speakers x (silence frames + 1),
thresholds in (0, 1), every float finite) and derives every tensor's shape
from them; `pktito-tools tiny` writes a tiny random-weight diarizer, and
tests/pkfcheck.c's corpus gains its flips and truncations; and
`make check-sanitize` runs, core/model.c having changed.

### Conversion

sortformer.py's `tensors()` is the converter's first half, numpy alone (the
.nemo read by nemo.py): BatchNorm folded into the depthwise convolution and
its bias, the feed-forwards' one half into their second matrices. The second
half is canon.py's: int8 panels, a scale a row. Folding the input's scale
into the subsampling's last matrix is optional, and measured like any other
rounding.

### int8 or float

Decided as README's "Why int8" decided Parakeet's: tools/quantcheck.py's
method, on the diarizer. Run the reference in float32 with each matrix, and
the activations entering it, rounded each way, over real conversations, and
report against float32:

- the labels that differ, and DER between the two (as "transcripts not
  float32's" and "WER between them");
- a paired measure finer than labels: the mean binary cross-entropy of the
  rounded model's probabilities against float32's labels, less float32's own
  (the diarizer's dlogp);
- DER against the corpus's labels, and the words that change speaker;
- each twice: with the cache choosing its own frames, as it will, and with
  float32's choices imposed, to tell the network's rounding from the
  cache's choices parting.

What int8 costs here is not known. Parakeet's encoder lost nothing
measurable to int8 in blocks of 32, and this one is the same kind of
network; but a diarizer's output is cut at one half and fed back into its own
cache.

The corpus cannot be tools/corpus.py's LibriSpeech, a speaker a recording.
Real conversations with reference labels that can be downloaded are AMI's
meetings and VoxConverse's debates and news (inferred: neither was fetched
here, and their licences and speaker counts are to check); CALLHOME and
DIHARD, behind NVIDIA's tables, are the LDC's and not free.

## 7. What it costs

### Time

Measured on the 9950X (16 threads, `avx512-vnni`, gcc 15.2, `make
PROFILE=1`, `pktito-c bench`, best of five): pktito's own encoder with v2's
shapes and random weights (17 layers of 512, 8 heads, feed-forward 2048,
256 subsampling channels) over 609 frames, one step's input, takes 47 ms, of
which features 3.1 ms and subsampling 2.8 ms; Parakeet v2 takes 228 ms for
test.wav's minute whole on the same build. By this study's count of their
multiply-adds, that is 1.8 trillion a second at the diarizer's shapes
against 2.3 at Parakeet's.

A minute is 60 / 27.2 = 2.2 steps, 104 ms of encoder. The Transformer,
projection and head are 8% of a step's multiply-adds (7.5 of 91 billion) and
some more of its time, their matrices being small. By multiply-adds alone
the diarizer is 0.40 of Parakeet (202 billion a minute against 507 in 30 s
chunks); by the measured encoder, about half (inferred from here on):

| | Parakeet v2 (README) | the diarizer | both |
|---|---|---|---|
| Ryzen 9 9950X | 0.217 s a minute | 0.11 to 0.13 s | 0.33 to 0.35 s: about 175x real time |
| Raspberry Pi 5 | 3.71 s | 1.5 to 2.2 s | 5.2 to 5.9 s: an hour in 5 to 6 minutes |
| Raspberry Pi 3B+ | 44.6 s | 18 to 27 s | 63 to 72 s: an hour in 63 to 72 minutes |

So a Pi 3B+ would label a podcast a little slower than it plays. For scale,
NeMo itself on this machine (float32, 16 threads) took 50 s for an hour, and
3.9 GB (measured).

The latency is paid in compute: every step encodes the cache again, so
shorter chunks pay for it more often. By multiply-adds, the 10 s
configuration costs 1.8 times the 30.4 s one and the 1.04 s configuration 33
times, which on a Pi 5 is about a minute of work a minute of audio
(inferred). A file wants 30.4 s.

Nemotron 3 costs about the same: 183 billion multiply-adds a minute, 0.36 of
Parakeet (its 31 layers each do less than a FastConformer layer - one
feed-forward, no convolution - and its cache is 264 frames).

### Memory

Inferred, from sizes:

- the file: about 120 MB (117.7M weights at a byte each and a float scale a
  row; NVIDIA's q8_0 GGUF of v2 is 147 MB), mapped as the recognizer's is.
- a step's work at 609 frames: a feed-forward's 609 x 2048 floats (5 MB), a
  head's attention scores, 609 x 609 (1.5 MB a thread), the chunk's features
  (3,048 x 128 floats, 1.6 MB), the cache and the FIFO (0.5 MB); and the
  probabilities, 16 bytes every 80 ms (0.7 MB an hour). 20 to 40 MB, with the
  threads'.
- A Pi 3B+ peaks at 652 MB with v2 alone (README); with the diarizer's 150
  MB or so beside it, about 800 of its 905. Diarizing the recording first and
  transcribing it after lets the kernel drop one model's pages for the
  other's, both being clean mappings of their files. To measure on the
  board: peak memory and major faults over an hour.

One finding beside the point but in a podcast's way: core/cli.c's
`read_audio` holds the whole recording as floats, at least 230 MB an hour,
and `chunks_of` copies it again, padded, for the VAD: at least 460 MB at the
peak. With the 624 MB model, an hour-long file needs more than a Pi 3B+'s
905 MB already, diarizer or not, and the model's pages would be read from
the card again and again (inferred from the code; not run on the board).
Samples kept as 16-bit, or the VAD run as the file is read, would fix it.

## 8. From frames to words

### NeMo's rule

NeMo's speaker-tagged transcript (SpeakerTaggedASR in NeMo 3.0's
multispk_transcribe_utils: perform_offline_stt_spk and
get_word_dict_content_offline) gives each word one speaker. Each speaker's
probability is averaged over the word's frames and the frame before them (a
transducer emits a token a little after it is heard), each average raised to
at least 0.01, and the largest wins, the first of equals. A turn - NeMo's
"sentence" - is a run of words with one speaker. Nothing is smoothed.

Where two people talk at once, Parakeet writes one stream of words, usually
the louder voice's, and each word still goes to one speaker, the likeliest
over its frames. The diarizer hears both; the merge can say so (another
speaker's average over one half too), but the transcript holds only what
Parakeet heard. A speaker change in mid-sentence is a turn that starts in
mid-sentence: the merge is a word at a time.

pktito would compute the same in integers where it can: a word's span in
samples (its chunk's first sample, plus its 80 ms encoder frames) to the
diarizer's 80 ms frames counted from the recording's start; the averages in
one order; so the same speakers on every machine. One edge NeMo gets wrong:
a word of no frames at the last frame averages from frame n_spk - 1 to the
end (`diar_pred_out.shape[1] - 1`, the speakers' axis where the frames' is
meant); pktito would take the last frame.

### Measured, on synthetic conversations

Six conversations made from LibriSpeech chapters, each speaker a reader:
turns of a median 5 s, gaps of 0.15 to 1 s, one turn in seven overlapping
the last by 0.2 to 0.8 s, the first speaker a host twice as likely to speak
next as each guest; the reference is Silero's speech segments within each
turn. Read speech from distinct readers on good microphones is far easier
than a podcast - section 1's tables say how real conversations go - so this
measures the pipeline, not the model. NeMo's v2 at 30.4 s, pktito-c's
transcripts (v2):

| | speakers | length | words | DER (miss, false alarm, confusion) | with a 0.25 s collar | words to the wrong speaker: NeMo's rule | the fallback | turns: true, NeMo's rule (changes not there), the fallback (changes not there) |
|---|---|---|---|---|---|---|---|---|
| conv2 | 2 | 5.0 min | 407 | 3.80 (2.45, 1.35, 0.00) | 0.00 | 1 | 0 | 23, 25 (2), 23 (0) |
| conv3 | 3 | 10.2 min | 832 | 4.12 (1.02, 3.10, 0.00) | 0.00 | 2 | 0 | 31, 34 (3), 31 (0) |
| conv4 | 4 | 10.1 min | 778 | 5.21 (1.28, 3.94, 0.00) | 0.18 | 1 | 0 | 39, 39 (1), 39 (0) |
| conv2h | 2 | 60.1 min | 6,465 | 4.27 (1.48, 2.79, 0.00) | 0.00 | 7 | 3 | 276, 280 (10), 274 (3) |
| conv3h | 3 | 60.0 min | 5,535 | 4.57 (1.44, 3.13, 0.00) | 0.00 | 4 | 1 | 221, 222 (4), 221 (1) |
| conv5 | 5 | 10.2 min | 1,320 | 22.45 (4.41, 2.85, 15.18) | 19.19 | 270 | 263 | |

What it shows:

- Up to four speakers and an hour long, v2 never took one speaker for
  another (confusion 0.00); its misses and false alarms are where its idea
  of where speech starts and ends differs from Silero's, gone with a
  quarter-second collar.
- A fifth speaker has nowhere to go: conv5's confusion is 15%, and a fifth
  of its words go to the wrong speaker.
- 916 of the 14,017 words in the two-to-four-speaker conversations (6.5%)
  have no speaker over one half, most of them a turn's first or last word.
  NeMo's rule still gives most of them the right speaker, the likeliest of
  the unlikely. Of the 15 it gets wrong, 11 are words where every average is
  under 0.01, which go to speaker 0 (whoever spoke first: here the host),
  and read as a word or two of the host's inside someone else's sentence:
  20 changes of speaker that were not there, in 2.4 hours.
- The fallback (speakers.py's "near": a word whose averages are all under
  one half takes the speaker heard nearest it within 0.5 s, else the
  previous word's) took the 15 wrong words to 4 and the 20 changes to 4. It
  departs from NeMo, so it is a question (section 12).

### What the user sees

- **the text**: a paragraph a turn, led by `Speaker 1:`, the speakers
  numbered from 1 in the order they first spoke.
- **--json**: each word gains `speaker`; a top-level `turns` (speaker,
  start, end, first word, how many words); `meta` gains `speakers` (how many
  were heard) and the diarizer's model. The chunks stay as they are.
- **--srt and --vtt**: the caption spec already has the rule
  (docs/caption-spec.md 4.3, `speaker_change_requires_cut`): a change
  between two known speakers is a cut. core/captions.c leaves speakers out
  today; with them, `pk_capword` gains a speaker and every change is a
  required cut. Where the cut falls inside words said over each other, the
  spec stops for an editor; pktito's profile makes a forced cue instead, as
  it does with its other conflicts. WebVTT names the voice, `<v Speaker 1>`
  at the start of each cue. SRT has no markup for it; `Speaker 1: ` before a
  turn's first cue is the usual practice.
- **one flag, --speakers**: whether to spend half again the time on who
  said what is the user's to know and the engine's not. How many speakers
  there are is not a flag; the model finds up to four. The diarizer's .pkf
  is found in models/ as the recognizer's is.
- **--stream** is left out of the first stages: the 10 s configuration
  costs about twice the 30.4 s one, and a line would wait 10 s for its
  speaker.

## 9. The plan

Each stage ends at a check that holds it there.

0. **This study** (done): the float64 reference against NeMo 3.0.0, the
   merge, the measurements above.
1. **The reference kept, the numerics chosen.** tests/diarcheck.py: the
   reference against NeMo on a handful of recordings, labels and DER, in
   `make check-python` and skipped where NeMo is not. The quantcheck of
   section 6 over a corpus of real conversations, its table in the README
   beside "Why int8". The converter and the .pkf: the validator, the binder,
   docs/MEMORY_SAFETY.md, a tiny diarizer for pkfcheck's corpus. Done when the
   file loads with every byte checked and its numerics have their
   measurement.
2. **C on backend/ref.** core/encoder.c taking either model's weights,
   core/diar.c, core/speakers.c, the features unnormalized, the API, the
   CLI's --speakers and its outputs. A two-speaker recording in tests/data
   (two LibriSpeech readers, a minute) with the --json ref writes for it,
   which check-c holds every backend to byte for byte, as it holds
   test.v2.json; DER and the words that change speaker against the
   reference's over the corpus; check-sanitize. Done when every backend
   writes ref's bytes and C's DER is the reference's within int8's measured
   cost.
3. **The tuned backends.** v2 adds no kernel, so this is measuring: `make
   PROFILE=1` on each board, the Transformer's 192-wide matrices and 24-wide
   heads the likely candidates for a CLIMB.md hypothesis; then a Pi 5 and a
   Pi 3B+ over an hour-long recording: time, peak memory, major faults.
4. **Nemotron 3, once NeMo releases it.** Its reference first - a pre-LN
   Transformer with rotary positions, GELU, the stacked features, the
   upsampler, the learned silence frame, eight speakers - held to that
   release; then its encoder in C beside the FastConformer, and erf in
   core/vmath.h. The cache, the merge, the outputs and the checks stay.

## 10. Risks

- **The labels are NeMo's, the bits cannot be.** A cut is discontinuous, so
  float64 already moves 9 labels in 469,856, and int8 will move more. If it
  moves many more, some matrices stay float, and pay for it.
- **Synthetic conversations flatter.** Real ones have music, laughter,
  cross-talk, a guest on a phone line, two voices alike. NVIDIA's tables
  for v2 - DER 5 to 12% on telephone calls of two to four speakers, 15% on
  DIHARD III's mix of domains, 20 to 29% in meetings - are closer to what a
  reader will see.
- **More than four speakers** with v2: a fifth of a five-speaker
  conversation's words went to the wrong speaker.
- **Very long recordings**: v2's card warns that they "may degrade"; the two
  synthetic hours here did not, with two and three speakers.
- **Licences**: v2's is CC-BY-4.0; v2.1's and Nemotron 3's are not, and
  are Peter's to read.
- **The Pi 3B+'s memory**: tight with both models, and past its limit for an
  hour-long file today because of the CLI's buffers.
- **NeMo's tie order** is torch's and unspecified. It decided nothing here.

## 11. What would make me recommend against it

- If int8 cannot hold v2's labels near float32's and float makes the
  diarizer slower than the recognizer: a Pi 3B+ would then take two hours
  or more for an hour, and the labels would cost more than the words.
- If the recordings Peter has in mind mostly have five speakers or more and
  NeMo has not released rotary attention: v2 would put a fifth of their
  words on the wrong speaker. Better to wait for Nemotron 3.
- If v2 confuses the speakers of real podcasts often enough that a reader
  would trust its labels less than none. The reference can measure that on
  any recording before a line of C is written.
- If a Pi 3B+ must label hour-long recordings and cannot hold both models
  even once the CLI keeps its samples as 16-bit.

## 12. Questions for Peter

1. **v2 now, or wait for Nemotron 3?** I would build stages 1 and 2 on v2
   now, since all but the network carries over, and decide what ships when
   NeMo's next release is out.
2. **v2.1's licence** (NVIDIA Open Model License): acceptable? Same code,
   much better in meetings.
3. **NeMo's rule, or the fallback for unheard words?** I would keep NeMo's
   rule as the definition and propose the fallback again after measuring it
   on real recordings.
4. **The output**: `Speaker 1:` in the text and in SRT, `<v Speaker 1>` in
   WebVTT, numbered in the order the voices arrive?
5. **A separate .pkf for the diarizer**, one for both recognizers, as I would
   do it?
6. **The corpus** for the numerics: AMI and VoxConverse, or recordings you
   have?
7. **Live speaker labels** (--stream): later, or never?
8. **The CLI's whole-recording buffers**: fix them before stage 3, so that a
   Pi 3B+ can take an hour?

## 13. What was verified, and what was inferred

Verified in this study:

- each model's licence, size, date, speaker count and published DER: its
  Hugging Face card, configuration and files, read on 2026-11-08; v2's
  architecture from its checkpoint's configuration and tensors;
- NeMo 3.0.0 is the latest release (PyPI and GitHub) and has no rotary
  attention (its source), which NeMo's main branch has, with the learned
  silence frame; Nemotron 3's tensors (its safetensors header: 99,226,504
  numbers, 31 layers, the 1024-to-512 stacking projection, the 192-to-1536
  upsampler over 3 frames, a silence embedding);
- the reference's agreement with NeMo (section 4) and torch's order for ties
  (section 5);
- NeMo's DER and the words' speakers on the synthetic conversations, and
  NeMo's rule for words (its source) (section 8);
- pktito's encoder at v2's shapes and Parakeet's minute, on one build
  (section 7);
- the CLI's whole-recording buffers (its source);
- reference.py's change leaves Parakeet's features, encoder output and
  tokens on the first 12 s of test.wav bit for bit, and `make check-c`
  passes on this machine.

Inferred, to be measured:

- the diarizer's whole time in C, the Transformer's share in it, and every
  number for the Pi 5 and the Pi 3B+, time or memory (the multiply-adds of
  section 7 are this study's own count, from the shapes);
- what int8 costs the diarizer;
- v2.1's network being v2's;
- what Nemotron 3 needs beyond its tensors (from its card, its
  configuration and NeMo main's source; it was not run);
- the size of the .pkf;
- the corpora's licences.
