# Who spoke when

pktito writes what was said. An interview or a podcast also needs who said
it: Speaker 1, Speaker 2, each word given to the voice that said it. This is
the study of doing that in pktito - which of NVIDIA's diarizers, how close to
NeMo it can be held, what it costs on the boards, how its output meets
Parakeet's words, and in what order to build it. It was written on
2026-11-08 on a Ryzen 9 9950X under WSL2, which is not one of the lab
boards. Nothing in it is in the engine yet; the prototype is Python:

- `tools/pktito_tools/nemotron3.py`: NVIDIA's Nemotron 3 Diarization in
  float64 numpy, NeMo's arithmetic at a pinned commit, what reference.py is
  to Parakeet. `tools/pktito-tools diar Nemotron-3-Diarization.nemo
  talk.wav` prints its RTTM.
- `tools/pktito_tools/sortformer.py`: the speaker cache that both of
  NVIDIA's streaming diarizers keep, and the network of Streaming Sortformer
  v2, the alternative measured here; `pktito-tools diar` runs it when given
  v2's checkpoint.
- `tools/pktito_tools/speakers.py`: words to speakers, NeMo's rule and the
  one alternative measured below.
- `tools/pktito_tools/reference.py`: the three things v2's encoder has and
  Parakeet's lacks - a bias on its matrices, xscaling, features that are not
  normalized - each off for Parakeet, whose features, encoder output and
  tokens on the first 12 s of test.wav are unchanged bit for bit.

A number here is one of three kinds and says which. Measured: on this
machine, in this study. Published: a model card's, read on 2026-11-08.
Inferred: reasoned from the other two, and to be measured before anything
relies on it.

## In short

- **The model: Nemotron 3 Diarization** (nvidia/Nemotron-3-Diarization,
  OpenMDW-1.1, 99.2M parameters, up to eight speakers), the one Peter would
  ship. It is a Streaming Sortformer like v2 - the same speaker cache, kept
  by the same rules - with a network of its own: 31 pre-LN Transformer
  layers of 512 with rotary positions over feature frames stacked eight at a
  time, and an upsampler that gives each speaker a probability every 10 ms.
  NeMo runs it from its repository at a pinned commit (NVIDIA-NeMo/Speech
  cd61689, 2026-11-06), and Hugging Face Transformers runs it with code of
  its own (pinned at 0f7a876). The two agree to 5e-6 on ten recordings once
  Transformers takes the checkpoint's mel tables (but in the last 80 ms of
  five, which an extra frame of Transformers' moves by up to 0.015, no label
  changed), and the float64 reference gives NeMo's labels on seven
  recordings up to an hour long, all but 888 of 7.5 million - 874 of them on
  the one conversation where the model hears one voice as three.
- **The cost: v2's, under half of Parakeet's** (measured on pktito's own
  kernels at its shapes, 45 ms a step on the 9950X; inferred for the
  boards): a Pi 5 would transcribe and label an hour in 5 to 5.5 minutes, a
  Pi 3B+ in 61 to 66. Memory: a 100 MB file beside the recognizer's, 20 to
  40 MB of work.
- **The merge: NeMo's rule**, each word to the speaker most probable over
  its frames, on the probabilities averaged to 80 ms as NeMo's own pipeline
  takes them. On synthetic conversations of two to four speakers, 8 of
  14,017 words went to the wrong speaker (v2: 15); of six and eight
  speakers, 11 of 3,590 (v2, which has four, 1,569). A fallback for words
  no one was heard saying takes the 8 to 0, but departs from NeMo: a
  question for Peter.
- **The new C:** no new matrix kernel. GELU (erf in core/vmath.h, and an
  elementwise kernel beside SiLU's), the rotation of the attention's
  queries and keys (a table of sines and cosines and a loop in core/), the
  stacking (no copy: eight feature frames are already one row of 1024), the
  upsampler (one matrix over three frames side by side), the learned
  silence frame (a tensor), and the cache.
- **The licence: OpenMDW-1.1**, from its text: any use, change and
  distribution, if a copy of the licence and the model's notices of origin
  go with what is distributed; the rights end for whoever sues claiming the
  model infringes; nothing on the outputs. Peter accepted it on 2026-11-08.
- **The plan:** the reference kept, held to NeMo at its commit and to
  Transformers, and the numerics measured, as Parakeet's were; C on
  backend/ref with a .pkf of its own; GELU's kernel in the tuned backends;
  then the boards.

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

**Nemotron 3**, which Peter has chosen ("it is definitely the one that I
would want to ship"), and which answers CLAUDE.md's questions in order:

1. **NeMo agrees with it, and so does a second implementation.** NeMo 3.0.0,
   the latest release (2026-08-07), cannot load the checkpoint (measured:
   "self_attention_model='rope' is not supported"). NeMo's repository can:
   NVIDIA-NeMo/Speech at cd61689d41b1962d3e1957ae8faf28bf8bee0e3b
   (2026-11-06) restores it and runs the card's configuration (measured),
   and a pinned commit is as fixed as a release - nemotron3.py names the
   commit whose arithmetic it is. An earlier version of this study held that
   an unreleased NeMo could not be checked against; it can, at a commit.
   Hugging Face Transformers implements the model again, with code of its
   own (AutoModelForAudioFrameClassification, pinned at
   0f7a8768733c448751f25abcb4b6b3ffef829cf5, 2026-11-08), and the two agree
   with each other to float32's rounding once Transformers uses the
   checkpoint's mel tables (section 4). v2 has one reference; this has two.
2. **A reader can follow it.** Its network is a textbook pre-LN
   Transformer - LayerNorm, attention, LayerNorm, feed-forward - with rotary
   positions: less to read than v2's FastConformer with a second Transformer
   after it. The cache is the same page of C.
3. **It gets more from the same silicon.** It costs what v2 costs (section
   7) and buys eight speakers and, on NVIDIA's benchmarks, much less error
   where recordings are hard: DIHARD III 12.73 against v2.1's 19.09,
   far-field meetings (NOTSOFAR1) 11.00 against 30.49 (published). On this
   study's synthetic conversations, up to eight speakers, it took one
   speaker for another only on the one where it heard a voice as three
   (section 8).
4. **Its licence is permissive** (below).

What it gives up: v2's encoder is the FastConformer that core/encoder.c
already runs, and Nemotron 3's is new C, though on the same kernels (section
6). And where it is unsure of a voice it can split it: it heard one of the
nine conversations' readers as three speakers (section 8). v2 stays in this
study as the measured alternative; nothing in the plan needs it.

### The licence: OpenMDW-1.1, from its text

The card says the model "is governed by the OpenMDW License Agreement,
version 1.1" and links openmdw.ai/license/1-1 (the Linux Foundation's
OpenMDW; read 2026-11-08). The model's repository has no LICENSE or NOTICE
file of its own. The agreement:

- **grants**, free of charge, permission "to deal in the Model Materials
  without restriction", under every copyright, patent, database and trade
  secret right in them; the Model Materials are the model (architecture and
  parameters) and the data, documentation and software provided with it.
  The phrase names no exception, so converting, quantizing, shipping and
  selling it are within it (my reading). It is accepted by exercising those
  rights.
- **requires**, of anyone who distributes any portion, that the
  distribution keep "a copy of this agreement" and "all copyright notices
  and other notices of origin included in the Model Materials that are
  applicable". A .pkf made from the checkpoint and offered for download
  would carry the agreement's text and the model's origin
  (nvidia/Nemotron-3-Diarization, NVIDIA) in its metadata, and the download
  page the same. It asks for no other attribution and no notice of changes,
  both of which CC-BY-4.0, Parakeet's licence, asks for.
- **ends** the rights of anyone who brings, keeps up or joins a suit
  claiming that the Model Materials infringe a patent or copyright, unless
  in answer to one brought against them first.
- **imposes nothing** on "any outputs generated by using the Model
  Materials": the labels, the transcripts.
- **disclaims** every warranty and liability, and makes the user "solely
  responsible" for clearing other people's rights in the materials and
  their use. The card lists among its training data recordings "Licensed
  under agreement" from a third party.

Peter accepted those terms on 2026-11-08: "sufficiently open and
commercial-safe for our users" (section 12).

## 3. What the engine would compute

Nemotron 3 Diarization as NeMo runs it at cd61689
(SortformerEncLabelModel's forward_streaming with its TransformerEncoder,
SortformerModules' streaming_update and _compress_spkcache), at batch size
one; the configuration and the tensors read from the checkpoint:

- **features**: Parakeet's log-mel computation (128 mels; a 512-point FFT of
  25 ms Hann windows every 10 ms; preemphasis 0.97; power; log(x + 2^-24))
  without its normalization (`normalize: NA`), with the checkpoint's own
  filterbank and window (below).
- **pre-encode**: each 8 feature frames (80 ms) side by side, 1024 numbers,
  through a matrix to 512 (no bias); the last group padded with zeros.
- **a step**: the speaker cache (at most 264 frames), the FIFO (at most 40)
  and the chunk (340 frames, 27.2 s) with its right context (40) - at most
  684 frames - through a LayerNorm; 31 layers of x += attention(LN x), x +=
  ff(LN x) - 8 heads of 64, the queries and keys turned by rotary positions
  counted from the step's first frame (theta 10000, dimensions i and i + 32
  as a pair), a feed-forward 512 to 2048 to 512 with torch's exact GELU, a
  bias on the attention's output and the feed-forward's matrices only; a
  LayerNorm; a matrix to 192; an upsampler, a convolution over 3 frames from
  192 channels to 1536 read as eight frames of 192, 10 ms each; and a head
  (ReLU, 192 x 192, ReLU, 192 x 8, sigmoid): eight probabilities every 10
  ms, one a speaker.
- **the update**: v2's (sortformer.py's Stream). The chunk's frames join the
  FIFO; when it overflows, its oldest 300 or more join the cache, which is
  then cut back to 264: one frame a speaker of silence - a frame the model
  learned (`learnable_sil_emb`), where v2 keeps the running mean of what it
  called silence - and the 256 best-scored, each 80 ms frame scored on the
  mean of its eight 10 ms probabilities. Each speaker's best 24 are raised
  a lot and its next 24 a little; otherwise as v2's below.

The speakers come out in the order they first spoke, column 0 whoever spoke
first, and the cache is what keeps that order over an hour: each step
attends to frames of every speaker heard so far. The card's lower latencies
(1.04, 0.64 and 0.32 s) are the same computation with chunks of 9, 6 and 3
frames, right contexts of 4, 2 and 1, a FIFO of 264 and an update period of
222.

### Its features: Parakeet's computation, not Parakeet's numbers

Nemotron 3's preprocessor configuration is Parakeet's in every field but the
normalization, as v2's is (measured: the three checkpoints' configurations
compared). Its filterbank and window are Parakeet v2's with each value
rounded to bfloat16 (measured: equal bit for bit to Parakeet's rounded by
torch). NeMo computes with the stored tables, so the features differ from
Parakeet's - on test.wav by up to 1.76 and on average 0.0089 in the log
(measured, reference.features with each pair of tables), the most in bands
that hold little energy.

So one mel computation serves both models - core/mel.c with the model's
tables and a switch for the normalization, as reference.features has - but
one pass of it cannot feed both. The recognizer wants Parakeet's tables,
normalized over each VAD chunk; the diarizer wants its own, unnormalized,
over the whole recording. What feeding the diarizer Parakeet's features
would do is measured, since it is what Transformers does (section 4): 2,189
of 7.5 million labels not NeMo's on the seven recordings, nearly all on the
five-speaker conversation. And there is little to save: the features are
3.7 ms of a minute on the 9950X (pktito's mel stage, measured).

### v2, the alternative

Streaming Sortformer v2 as NeMo 3.0 runs it, which sortformer.py computes:
the same features with Parakeet's own tables, unnormalized; Parakeet's
subsampling a chunk at a time (340 frames with 1 of left context and 40 of
right, each chunk's edges padded as its own); a step of at most 609 frames
(cache 188, FIFO 40, chunk 381) through 17 FastConformer layers (512 wide,
8 heads of 64, feed-forward 2048, depthwise kernel 9, relative positions,
every matrix but the positions' with a bias, the input first scaled by the
square root of 512), a matrix to 192, an 18-layer post-LN Transformer (8
heads of 24, a ReLU feed-forward of 768, no positions of its own) and a
head: four probabilities every 80 ms. Its update keeps three frames a
speaker of the recording's mean silence and the 176 best-scored; each
speaker's best 33 are raised a lot and its next 33 a little. In both, a
frame's score for a speaker is how much likelier it is that speaker alone
than not; frames that are not the speaker's speech never qualify, nor,
once the speaker has enough frames of its own, frames shared with another;
the newest frames get a little extra.

## 4. The references

### NeMo against Transformers

Measured: Transformers' Nemotron3DiarizationForAudioFrameClassification at
0f7a876 against NeMo's SortformerEncLabelModel at cd61689 (torch 2.14.1,
CPU, float32), in the card's 30.4 s configuration with its 264-frame cache,
which is also the checkpoint's own and what Transformers' offline mode
uses; the seven recordings of v2's table below. A label is whether one
speaker's probability is over one half in one 10 ms frame: eight a frame.

| | speakers | length | steps | as shipped: most apart, first chunk | anywhere | labels not NeMo's | with the checkpoint's tables: first chunk | anywhere | labels not NeMo's |
|---|---|---|---|---|---|---|---|---|---|
| test.wav | 1 | 60 s | 3 | 0.0043 | 0.015 | 1 of 48,000 | 6.4e-7 | 0.015 (6.4e-7 but the last 80 ms) | 0 |
| conv2 | 2 | 5.0 min | 12 | 0.0052 | 0.024 | 1 of 240,240 | 1.2e-6 | 1.4e-6 | 0 |
| conv3 | 3 | 10.2 min | 23 | 0.0025 | 0.023 | 19 of 489,680 | 8.3e-7 | 2.9e-6 | 0 |
| conv4 | 4 | 10.1 min | 23 | 0.0033 | 0.076 | 17 of 484,144 | 1.5e-6 | 2.1e-6 | 0 |
| conv5 | 5 | 10.2 min | 23 | 0.0021 | 0.79 | 2,096 of 488,960 | 7.7e-7 | 0.0049 (1.8e-6 but the last 80 ms) | 0 |
| conv2h | 2 | 60.1 min | 133 | 0.0060 | 0.018 | 18 of 2,884,808 | 6.0e-7 | 2.1e-6 | 0 |
| conv3h | 3 | 60.0 min | 133 | 0.0043 | 0.052 | 37 of 2,881,688 | 1.2e-6 | 4.2e-6 | 0 |

Two things part them, and past float32's rounding nothing else:

- **The mel tables.** Transformers' feature extractor computes the
  filterbank (librosa's) and the Hann window afresh, in float32, unless its
  `mel_filters_dtype` says otherwise - its documentation names the case
  ("Some checkpoints keep these constants in a reduced precision") - and
  this checkpoint's processor_config.json does not say. Those are
  Parakeet's tables, not the checkpoint's. With `mel_filters_dtype` set to
  "bfloat16", its features equal NeMo's bit for bit (test.wav), and its
  probabilities agree with NeMo's to 1.5e-6 in the first chunk and 4.8e-6
  everywhere but the end of a recording, without a label different, on
  these seven and on the 6- and 8-speaker conversations of section 8.
- **The last 80 ms of some recordings.** Transformers makes one feature
  frame more than it counts, masked, and where a recording's frames are a
  multiple of 8 that frame becomes an encoder frame of its own; the
  encoder's output for it, not zeroed, reaches the upsampler's convolution
  of the last real frame. NeMo zeroes it (forward_infer's `mask_padding`).
  The last 8 frames of test.wav and four others differ by up to 0.015; no
  label changed.

So the two implementations compute the same thing, and Transformers as
shipped differs from NeMo by its tables. The reference follows NeMo: the
checkpoint's tables, no extra frame.

### The float64 reference against them

Measured: nemotron3.py against NeMo at cd61689, as above. Against
Transformers with the checkpoint's tables, the same labels differ, frame
for frame.

| | speakers | length | steps | most apart, first chunk | anywhere | labels not NeMo's | DER against NeMo's labels |
|---|---|---|---|---|---|---|---|
| test.wav | 1 | 60 s | 3 | 9.7e-7 | 4.0e-6 | 0 of 48,000 | 0 |
| conv2 | 2 | 5.0 min | 12 | 4.1e-6 | 0.015 | 0 of 240,240 | 0 |
| conv3 | 3 | 10.2 min | 23 | 3.0e-6 | 0.0044 | 0 of 489,680 | 0 |
| conv4 | 4 | 10.1 min | 23 | 2.3e-6 | 1.2e-5 | 0 of 484,144 | 0 |
| conv5 | 5 | 10.2 min | 23 | 1.8e-6 | 0.27 | 874 of 488,960 | 1.66% |
| conv2h | 2 | 60.1 min | 133 | 2.1e-6 | 0.021 | 4 of 2,884,808 | 0.001% |
| conv3h | 3 | 60.0 min | 133 | 4.8e-6 | 0.048 | 10 of 2,881,688 | 0.003% |

And on section 8's larger conversations: conv6 (6 speakers) 1.2e-5 and no
label, conv8 (8) 9.1e-6 and none, conv8b (8, 20 minutes) 0.016 and 4.

In the first chunk the reference is within 4.8e-6 of NeMo, as v2's is
(1.5e-6) and reference.py is of Parakeet's encoder (2e-6). After a cut the
two can part, and the cuts are close calls: the gap between the last score
the cut kept and the best it dropped was under 1e-5 at 22 of conv2's and
conv5's 35 cache updates, and under 1e-7 at 5 (measured, the reference's
scores), which is where float32's rounding and float64's choose
differently. Most such partings mend at the next cut, which chooses the same
cache again: conv2's steps 2 to 4 differ from NeMo's by up to 0.02 and the
rest agree to 1e-5 again, as do conv2h's step 21, conv3h's 11, 12 and 23,
and conv8b's 29. On conv5 they do not: there the model hears the host as
three speakers (section 8), and how much of the host's speech also lands in
a second column depends on the cuts - 874 labels, all of them that.

The two float32 implementations, meanwhile, made the same choice at every
cut of the ten recordings: Transformers with the checkpoint's tables stays
within 5e-6 of NeMo at every step, but for the last 80 ms of five
recordings (measured). They compute the same float32 numbers from the same
features, and so, it seems, round a close call the same way (inferred).

This is v2's finding again: the diarizer is held to NeMo by its labels, DER
and the words that change speaker, as the transcripts are held by WER and
the transcripts that change, not by a tolerance on its probabilities. The
engine will differ from NeMo's float32 as float64 does, and more with int8.

### v2's reference against NeMo 3.0.0

Measured: sortformer.py against NeMo 3.0.0's own forward() (torch 2.14.1,
CPU, float32), the 30.4 s configuration. A label here is one of four
speakers in one 80 ms frame.

| | speakers | length | steps | most apart, first chunk | anywhere | labels not NeMo's | DER against NeMo's labels |
|---|---|---|---|---|---|---|---|
| test.wav | 1 | 60 s | 3 | 1.5e-6 | 1.6e-6 | 0 of 3,000 | 0 |
| conv2 | 2 | 5.0 min | 12 | 8.5e-7 | 1.9e-6 | 0 of 15,016 | 0 |
| conv3 | 3 | 10.2 min | 23 | 4.8e-7 | 6.3e-3 | 0 of 30,608 | 0 |
| conv4 | 4 | 10.1 min | 23 | 9.4e-7 | 2.7e-6 | 0 of 30,260 | 0 |
| conv5 | 5 | 10.2 min | 23 | 3.8e-7 | 0.094 | 3 of 30,560 | 0.05% |
| conv2h | 2 | 60.1 min | 133 | 6.4e-7 | 0.016 | 1 of 180,304 | 0.003% |
| conv3h | 3 | 60.0 min | 133 | 5.9e-7 | 0.038 | 5 of 180,108 | 0.01% |

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

No exact tie decided any of the reference's cuts - v2's at its 350 steps on
seven recordings, Nemotron 3's at its 441 on ten (the reference counts
them). The infinities tie by design and
harmlessly: a silence pad's +inf is always kept, and a disqualified frame's
-inf, if kept, becomes a silence frame whichever it was. What does decide
cuts is the near tie of section 4, a gap below float32's rounding, which no
ordering rule removes: C in float32, or int8, will choose as its own
arithmetic rounds.

## 6. Into pktito

### What Nemotron 3 needs in C

Each operation and what has it (backend/kernels.h), or what is new:

- **every matrix**: `pk_k_gemm_q8`, which takes a bias and adds into a
  residual stream. Rows come in 16s and inputs in 32s, and all fit: the
  stacking (1024 to 512), the queries, keys and values (512 to 512 each, or
  1536 at once), the attention's output, the feed-forward (512 to 2048 and
  back), the projection (512 to 192), the upsampler (576 to 1536) and the
  head (192 to 192; 192 to 8, padded to 16 at conversion).
- **the attention's products**: `pk_k_gemm_f32`, as core/encoder.c lays
  them out, without the position half that Parakeet's needs: heads of 64,
  no padding, `pk_k_softmax` with a scale of 1/8.
- **the rotation (new)**: the queries and keys, after their products and
  before K is laid out as panels, each pair (x_i, x_i+32) of a head turned
  to (x_i cos - x_i+32 sin, x_i+32 cos + x_i sin) by the frame's angle: a
  loop in core/, the same bits on every backend. The angles are the position (0
  to 683) times 10000^(-i/32), a float32 product as NeMo forms it, and their
  sines and cosines a table made once with core/vmath.h's `pk_sincosf`,
  whose sweep in tests/mathcheck would take these 21,888 angles.
- **GELU (new)**: torch's exact one, x times the normal distribution at x,
  so erf: `pk_erff` in core/vmath.h with its sweep in tests/mathcheck, and,
  since it runs over every feed-forward's 2048 x 684 values, a kernel
  beside SiLU's that applies it and feeds the quantization in one pass, as
  encoder.c's SiLU does: `pk_k_gelu`, in ref and every tuned backend, held
  bit for bit by tests/kcheck. numpy has no erf either; nemotron3.py's
  follows Abramowitz and Stegun, within 1.4e-14 of the C library's.
- **the stacking**: no copy. The features are [T][128], so eight frames are
  already one row of 1024; the matrix reads them with a stride of 1024, the
  last group from a zeroed tail.
- **the upsampler**: one product over each frame with the one before and
  the one after it, 576 numbers. Quantized in blocks of 32, a frame of 192
  is six whole blocks, so the 576-wide int8 rows are the three frames' int8
  rows side by side, each frame quantized once.
- **the learned silence frame**: 512 floats in the file, put in the cache's
  silence slots; simpler than v2's running mean.
- **layer norm** (512), the head's ReLU and sigmoid (`pk_sigmoidf`), the
  cut's logarithms (`pk_logf`): what Parakeet and v2 use. The cut scores
  80 ms frames, each the mean of its eight 10 ms probabilities: a sum in
  order and a multiplication by 1/8, exact.

v2 would need no new kernel at all: its FastConformer is Parakeet's with
biases and a scale; its Transformer's 24-wide heads are laid out as 32 for
`pk_k_gemm_f32`; its queries and keys each divided by the fourth root of 24
fold into their matrices.

### Code

- core/diar.c (new): the stacking, the encoder's layers on encoder.c's
  passes (pk_gemm, pk_quantize, the attention's panels), the projection, the
  upsampler, the head, and the cache - written to be read beside
  nemotron3.py and sortformer.py's Stream.
- core/speakers.c (new): a word's speaker from its frames, and the turns.
  Samples to frames in integers, every float sum in one order.
- core/mel.c: the features with the model's tables, without the
  normalization, a chunk at a time.
- core/vmath.h and the backends: `pk_erff`, `pk_k_gelu`.
- core/pktito.h: a diarizer, `pk_diar_new(model, threads)`, `pk_diar_feed`
  (samples in; a chunk's probabilities out once its right context has
  come), `pk_diar_free`; and `pk_speakers` (words and probabilities in, a
  speaker a word out).
- the CLI: section 8's outputs.

### The model file

A .pkf of its own, "sortformer" its architecture:

- **params**: the network (Nemotron 3's; v2's if it were ever wanted), the
  shapes (layers, width, heads, feed-forward, stacking, the projection's
  and the upsampler's widths, the speakers), the rotary base, and the
  cache's numbers - chunk, contexts, FIFO, cache, update period, silence
  frames, thresholds, boosts - for the 30.4 s configuration.
- **tensors**: nemotron3.py's `tensors()` names them: the mel tables, the
  stacking matrix, each layer's under Parakeet's names (`l<i>.att.q`,
  `l<i>.ff.up`, each bias `<name>.b`), the projection, the upsampler as a
  matrix over three frames side by side, the head, the silence frame.
- **no vocabulary**: the validator requires 1 to 2^20 pieces today; for this
  architecture it would require none.
- **metadata**: the source, the converter's version, and OpenMDW-1.1's text
  with the model's origin, which the licence requires to go with the file.

It is separate because 100 MB is a lot to download and map for someone who
never wants speakers, and because one diarizer serves both recognizers (it
hears voices, not a language).

docs/MEMORY_SAFETY.md's contract, in the same change: `pk_pkf_validate`
takes the second architecture; the binder reads every param with its range
(speakers 1 to 8, a cache of at least speakers x (silence frames + 1),
thresholds in (0, 1), every float finite) and derives every tensor's shape
from them; `pktito-tools tiny` writes a tiny random-weight diarizer, and
tests/pkfcheck.c's corpus gains its flips and truncations; and
`make check-sanitize` runs, core/model.c having changed.

### Conversion

nemotron3.py's `tensors()` is the converter's first half, numpy alone (the
.nemo read by nemo.py): NeMo's fused query, key and value matrix split in
three, the upsampler's taps laid side by side, the training-only heads left
out. The second half is canon.py's: int8 panels, a scale a row.

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
measurable to int8 in blocks of 32; this is a plainer network of the same
kind, but its stacking matrix reads log-mels directly, and its output is cut
at one half and fed back into its own cache, whose choices section 4 shows
turn on gaps of 1e-7.

The corpus cannot be tools/corpus.py's LibriSpeech, a speaker a recording,
and for DER it should be recordings the model was not trained on. The card
lists among its training data AMI's training and development meetings,
DiPCo's development set and VoxConverse's development and test sets, so of
the conversations with reference labels that can be downloaded, AMI's
evaluation meetings and DiPCo's evaluation set are clean and VoxConverse is
not (inferred from the card: nothing was fetched here, and their licences
and speaker counts are to check). CALLHOME and DIHARD, behind NVIDIA's
tables, are the LDC's and not free.

## 7. What it costs

### Time

Measured on the 9950X (16 threads, `avx512-vnni`, gcc 15.2): one 684-frame
step of Nemotron 3 takes 44.4 to 45.0 ms on pktito's own passes and kernels
(three runs, each the best of 7). The program that times it (n3bench.c, a
scratch program, not in the tree) runs the step core/ would run, with
random weights: core/encoder.c's `pk_gemm` and `pk_quantize`, the attention
as encoder.c's att_item does it without the position half,
`pk_k_layernorm`, `pk_k_softmax`, the rotation as a plain loop, and
`pk_k_silu` timed where `pk_k_gelu` would be. Its
stages run at the encoder's own rates: the feed-forward at 3.0 trillion
multiply-adds a second and the attention at 1.3, where `pktito-c bench`'s
profile of pktito's encoder at v2's shapes gives 2.6 to 3.2 and 1.2.

| a step (the run of 44.6 ms) | ms | |
|---|---|---|
| feed-forward | 14.8 | 33% |
| attention | 11.6 | 26% |
| queries, keys, values | 7.3 | 16% |
| attention out | 2.9 | 6% |
| GELU (as SiLU) into int8 | 2.7 | 6% |
| repacking K | 1.7 | 4% |
| layer norms into int8 | 1.5 | 3% |
| rotation | 1.4 | 3% |
| stacking, projection, upsampler, head | 0.7 | 2% |

The cache is 264 of the step's 684 frames, so every step pays to attend to
it again, and the attention's share grows with it. v2, measured the same
way: pktito's encoder at its shapes over 609 frames, 47 ms (`pktito-c
bench`, features and subsampling included), and its Transformer and head in
the scratch program, 7.8 to 8.2 ms: 55 ms a step. Parakeet v2 takes 225 ms
for test.wav's minute in the same session.

A minute is 60 / 27.2 = 2.2 steps: 98 ms, and 3.7 ms of features, 0.10 s a
minute for Nemotron 3, 0.46 of Parakeet's; v2's 0.12 s, 0.54. By this
study's count of multiply-adds, Nemotron 3 is 182 billion a minute, 0.36 of
Parakeet's 507, v2 202, 0.40. For the boards (inferred from here on):

| | Parakeet v2 (README) | Nemotron 3 | both |
|---|---|---|---|
| Ryzen 9 9950X | 0.217 s a minute | 0.10 s (measured) | 0.32 s: about 190x real time |
| Raspberry Pi 5 | 3.71 s | 1.3 to 1.7 s | 5.0 to 5.4 s: an hour in 5 to 5.5 minutes |
| Raspberry Pi 3B+ | 44.6 s | 16 to 21 s | 61 to 66 s: an hour in 61 to 66 minutes |

The boards' ranges run from the multiply-adds' share of Parakeet (0.36) to
the 9950X's measured share (0.46); v2's were 1.5 to 2.2 s and 18 to 27 s. So
a Pi 3B+ would transcribe and label a podcast a little slower than it
plays. For scale,
NeMo itself on this machine (float32, 16 threads) took 57 s for an hour and
3.5 GB, Transformers 30 s and 4.1 GB; NeMo 3.0.0 ran v2's hour in 50 s and
3.9 GB (measured).

The latency is paid in compute: every step encodes the cache again, so
shorter chunks pay for it more often. The card's 1.04 s configuration
(chunks of 9 frames, 541 a step) is 5.2 trillion multiply-adds a minute, 29
times the 30.4 s one: about 3 s a minute on the 9950X and 40 to 50 s on a Pi
5 (inferred). A file wants 30.4 s; live labels are a desktop's.

### Memory

Inferred, from sizes:

- the file: about 100 MB (99.2M weights at a byte each and a float scale a
  row; NVIDIA's q8_0 GGUF is 107 MB), mapped as the recognizer's is.
- a step's work at 684 frames: a feed-forward's 684 x 2048 floats (5.6 MB),
  the upsampled frames and the head's (5,472 x 192 floats, 4.2 MB each), a
  block of attention scores a thread, the chunk's features (3,040 x 128
  floats, 1.6 MB), the cache and the FIFO (0.6 MB); and the probabilities,
  eight floats every 10 ms (11.5 MB an hour; their 80 ms means, 1.4 MB). 20
  to 40 MB, with the threads'.
- A Pi 3B+ peaks at 652 MB with v2 alone (README); with the diarizer's 130
  MB or so beside it, about 780 of its 905. Diarizing the recording first
  and transcribing it after lets the kernel drop one model's pages for the
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

NeMo's speaker-tagged transcript (SpeakerTaggedASR in NeMo's
multispk_transcribe_utils: perform_offline_stt_spk and
get_word_dict_content_offline, the same at 3.0.0 and at cd61689) gives each
word one speaker. Each speaker's probability is averaged over the word's
frames and the frame before them (a transducer emits a token a little after
it is heard), each average raised to at least 0.01, and the largest wins,
the first of equals. A turn - NeMo's "sentence" - is a run of words with one
speaker. Nothing is smoothed. The rule considers the first
`max_num_of_spks` speakers of its own configuration and zeroes the rest
before choosing, and the multitalker example's configuration sets 4: with
Nemotron 3 it has to be 8, as it was for every number below, or no word can
go to a fifth speaker.

The frames are 80 ms. NeMo's pipeline sets a diarizer's output to the
recognizer's encoder frame before it merges (configure_diar_streaming at
cd61689, which its multitalker example calls with the ASR encoder's
subsampling of 8), so Nemotron 3's 10 ms probabilities reach the rule
averaged eight at a time (downsample_preds), and the rule is v2's. The 10 ms
frames are for where speech starts and ends: the RTTM, and the captions'
cuts.

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

Nine conversations made from LibriSpeech test-clean's chapters, each
speaker a reader: turns of a median 5 s, gaps of 0.15 to 1 s, one turn in
seven overlapping the last by 0.2 to 0.8 s, the first speaker a host twice
as likely to speak next as each guest; the reference is Silero's speech
segments within each turn. Read speech from distinct readers on good
microphones is far easier than a podcast - section 1's tables say how real
conversations go - and Nemotron 3 was trained on mixtures made from
LibriSpeech's 960 training hours (its card), other readers but this kind of
recording, so this measures the pipeline, not the model. NeMo's output in
the 30.4 s configuration (Nemotron 3 at cd61689, v2 at 3.0.0), DER on a 10
ms grid, pktito-c's transcripts (Parakeet v2):

Nemotron 3:

| | speakers | length | words | DER (miss, false alarm, confusion) | with a 0.25 s collar | words to the wrong speaker: NeMo's rule | the fallback | turns: true, NeMo's rule (changes not there), the fallback (changes not there) |
|---|---|---|---|---|---|---|---|---|
| conv2 | 2 | 5.0 min | 407 | 5.28 (3.11, 2.18, 0.00) | 0.11 | 0 | 0 | 23, 23 (0), 23 (0) |
| conv3 | 3 | 10.2 min | 832 | 4.92 (2.35, 2.56, 0.00) | 0.22 | 1 | 0 | 31, 33 (2), 31 (0) |
| conv4 | 4 | 10.1 min | 778 | 6.11 (2.13, 3.97, 0.01) | 0.16 | 0 | 0 | 39, 39 (0), 39 (0) |
| conv2h | 2 | 60.1 min | 6,465 | 4.70 (2.62, 2.08, 0.00) | 0.09 | 3 | 0 | 276, 282 (10), 276 (4) |
| conv3h | 3 | 60.0 min | 5,535 | 6.37 (1.92, 4.45, 0.00) | 0.41 | 4 | 0 | 221, 227 (7), 221 (0) |
| conv5 | 5 | 10.2 min | 1,320 | 11.40 (3.58, 2.51, 5.32) | 7.40 | 110 | 109 | 46, 50 (4), 48 (2) |
| conv6 | 6 | 10.1 min | 1,001 | 6.25 (2.63, 3.62, 0.00) | 0.37 | 0 | 0 | 50, 50 (0), 50 (0) |
| conv8 | 8 | 10.1 min | 905 | 6.21 (2.22, 3.99, 0.00) | 0.65 | 1 | 0 | 40, 42 (2), 40 (0) |
| conv8b | 8 | 20.1 min | 1,684 | 4.91 (2.20, 2.71, 0.00) | 0.23 | 10 | 7 | 79, 77 (1), 77 (0) |

v2:

| | speakers | length | words | DER (miss, false alarm, confusion) | with a 0.25 s collar | words to the wrong speaker: NeMo's rule | the fallback | turns: true, NeMo's rule (changes not there), the fallback (changes not there) |
|---|---|---|---|---|---|---|---|---|
| conv2 | 2 | 5.0 min | 407 | 3.80 (2.45, 1.35, 0.00) | 0.00 | 1 | 0 | 23, 25 (2), 23 (0) |
| conv3 | 3 | 10.2 min | 832 | 4.12 (1.02, 3.10, 0.00) | 0.00 | 2 | 0 | 31, 34 (3), 31 (0) |
| conv4 | 4 | 10.1 min | 778 | 5.21 (1.28, 3.94, 0.00) | 0.18 | 1 | 0 | 39, 39 (1), 39 (0) |
| conv2h | 2 | 60.1 min | 6,465 | 4.27 (1.48, 2.79, 0.00) | 0.00 | 7 | 3 | 276, 280 (10), 274 (3) |
| conv3h | 3 | 60.0 min | 5,535 | 4.57 (1.44, 3.13, 0.00) | 0.00 | 4 | 1 | 221, 222 (4), 221 (1) |
| conv5 | 5 | 10.2 min | 1,320 | 22.45 (4.41, 2.85, 15.18) | 19.19 | 270 | 263 | 46, 48 (2), 46 (5) |
| conv6 | 6 | 10.1 min | 1,001 | 27.25 (3.97, 1.95, 21.33) | 21.79 | 194 | 194 | 50, 50 (6), 45 (2) |
| conv8 | 8 | 10.1 min | 905 | 44.15 (17.16, 1.45, 25.53) | 40.09 | 363 | 363 | 40, 54 (19), 35 (5) |
| conv8b | 8 | 20.1 min | 1,684 | 57.75 (8.39, 1.73, 47.64) | 56.69 | 1,012 | 1,010 | 79, 90 (30), 80 (19) |

What it shows:

- Up to eight speakers and an hour long, Nemotron 3 never took one speaker
  for another (confusion at most 0.01%), but on conv5. Its misses and false
  alarms are where its idea of where speech starts and ends differs from
  Silero's; a quarter-second collar leaves 0.1 to 0.7%, a little more than
  v2's 0 to 0.2%.
- conv5 is the exception, and not for its count: Nemotron 3 heard its host
  as three speakers (110.7 s in one column, 16.4 and 11.9 s in two more,
  measured) and the other four as four. 110 of its 1,320 words went to the
  wrong speaker, where v2, which has nowhere to put a fifth, put 270.
- v2 has four columns: at six speakers a fifth of the words went to the
  wrong speaker, at eight 40 to 60%.
- 760 of the 14,017 words in the two-to-four-speaker conversations (5.4%;
  v2 916) have no speaker over one half, most of them a turn's first or last
  word. NeMo's rule still gives most of them the right speaker. All 8 it
  gets wrong are words where every average is under 0.01, which go to
  speaker 0 (whoever spoke first: here the host), and read as a word or two
  of the host's inside someone else's sentence: 19 changes of speaker that
  were not there, in 2.4 hours (v2: 20).
- The fallback (speakers.py's "near": a word whose averages are all under
  one half takes the speaker heard nearest it within 0.5 s, else the
  previous word's) took those 8 to 0 and the 19 changes to 4. It departs
  from NeMo, so it is a question (section 12).
- conv8b's 10 are seven words that Parakeet wrote twice over at one instant
  at a change of speaker ("of the power of the power of the power of the
  family") and three unheard ones: a recognizer's error, scored here.

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
  there are is not a flag; the model finds up to eight. The diarizer's .pkf
  is found in models/ as the recognizer's is.
- **--stream** is left out of the first stages: the 1.04 s configuration
  costs 29 times the 30.4 s one, and a line would wait half a minute for its
  speaker in the cheap one.

## 9. The plan

Each stage ends at a check that holds it there.

0. **This study** (done): the float64 reference against NeMo at cd61689 and
   against Transformers at 0f7a876, and the two against each other; v2's
   against NeMo 3.0.0; the merge; the costs.
1. **The reference kept, the numerics chosen.** tests/diarcheck.py: the
   reference against NeMo at its commit and against Transformers (with the
   checkpoint's tables) on a handful of recordings, labels and DER, in `make
   check-python` and skipped where they are not installed; nemotron3.py's
   `_erfc` against the C library's. The quantcheck of section 6 over a
   corpus of real conversations, its table in the README beside "Why int8".
   The converter and the .pkf: the validator, the binder,
   docs/MEMORY_SAFETY.md, a tiny diarizer for pkfcheck's corpus. Done when
   the file loads with every byte checked and its numerics have their
   measurement.
2. **C on backend/ref.** `pk_erff` and its mathcheck sweep, `pk_k_gelu` in
   ref, the rotary table's angles in mathcheck's sweep of `pk_sincosf`;
   core/diar.c, core/speakers.c, the features with the model's tables and
   unnormalized, the API, the CLI's --speakers and its outputs. A
   two-speaker recording in tests/data (two LibriSpeech readers, a minute)
   with the --json ref writes for it, which check-c holds every backend to
   byte for byte, as it holds test.v2.json; DER and the words that change
   speaker against the reference's over the corpus; check-sanitize. Done
   when every backend writes ref's bytes and C's DER is the reference's
   within int8's measured cost.
3. **The tuned backends.** `pk_k_gelu` in each, held to ref by kcheck, its
   hypothesis and measurement in the backend's CLIMB.md; then `make
   PROFILE=1` on each board, the attention over a 684-frame step the likely
   next candidate; then a Pi 5 and a Pi 3B+ over an hour-long recording:
   time, peak memory, major faults.
4. **When NeMo releases a version that runs the model**, or moves the
   arithmetic: the reference checked against it, and nemotron3.py's commit
   moved with a measurement, as any change to the arithmetic would be.

## 10. Risks

- **The labels are NeMo's, the bits cannot be.** A cut is decided by gaps
  below float32's rounding, so float64 already moves 888 labels in 7.5
  million, and int8 will move more. If it moves many more, some matrices
  stay float, and pay for it.
- **A voice split in three.** On one conversation of nine the model heard
  one reader as three speakers; real recordings, with a voice that changes
  between a laugh and a quote, may do that more. Measure it on real
  conversations before relying on the speaker count.
- **Synthetic conversations flatter**, the more so as Nemotron 3 was
  trained on mixtures of LibriSpeech readers. Real ones have music,
  laughter, cross-talk, a guest on a phone line, two voices alike. NVIDIA's
  tables
  for Nemotron 3 - DER 6 to 11% on telephone calls of two to four speakers,
  13% on DIHARD III's mix of domains, 11% in meetings - are closer to what a
  reader will see.
- **More than eight speakers**: nothing measured; the ninth has nowhere to
  go, as v2's fifth had.
- **The reference's NeMo is a commit**: a later NeMo may change the
  arithmetic (it has: 3.0.0 cannot load the model). The commit is in
  nemotron3.py; a move to another is a change to the arithmetic, measured.
- **Transformers as shipped is not NeMo**: its processor computes the
  features with Parakeet's tables. Anyone holding pktito to Transformers
  must set `mel_filters_dtype`, as diarcheck would.
- **The licence's terms**: a copy of OpenMDW-1.1 and the notices of origin
  must go with every distributed .pkf; the rights end on a patent or
  copyright suit over the model.
- **The Pi 3B+'s memory**: tight with both models, and past its limit for an
  hour-long file today because of the CLI's buffers.

## 11. What would make me recommend against it

- If int8 cannot hold Nemotron 3's labels near float32's and float makes the
  diarizer slower than the recognizer: a Pi 3B+ would then take two hours
  or more for an hour, and the labels would cost more than the words.
- If Nemotron 3 splits real speakers often - the conv5 failure on real
  podcasts - so that a reader would trust its labels less than none. The
  reference can measure that on any recording before a line of C is
  written.
- If a Pi 3B+ must label hour-long recordings and cannot hold both models
  even once the CLI keeps its samples as 16-bit.

## 12. Questions for Peter

1. **OpenMDW-1.1's terms** (section 2): decided 2026-11-08. Peter accepted
   them as "sufficiently open and commercial-safe for our users"; the
   licence's text and the model's origin go in the .pkf's metadata and on
   the download page, as it requires.
2. **NeMo's rule, or the fallback for unheard words?** I would keep NeMo's
   rule as the definition and propose the fallback again after measuring it
   on real recordings.
3. **The output**: `Speaker 1:` in the text and in SRT, `<v Speaker 1>` in
   WebVTT, numbered in the order the voices arrive?
4. **A separate .pkf for the diarizer**, one for both recognizers, as I would
   do it?
5. **The corpus** for the numerics: AMI's evaluation meetings (VoxConverse
   is in the model's training data), or recordings you have?
6. **Live speaker labels** (--stream): later, or never?
7. **The CLI's whole-recording buffers**: fix them before stage 3, so that a
   Pi 3B+ can take an hour?
8. **Transformers' two differences from NeMo** (section 4, the mel tables
   and the last 80 ms): report them upstream?

## 13. What was verified, and what was inferred

Verified in this study:

- each model's licence, size, date, speaker count and published DER: its
  Hugging Face card, configuration and files, read on 2026-11-08; v2's and
  Nemotron 3's architectures from their checkpoints' configurations and
  tensors, and Nemotron 3's from NeMo's and Transformers' sources at the
  pinned commits;
- the licence's terms: OpenMDW-1.1's text at openmdw.ai, read 2026-11-08;
  the model's repository has no LICENSE or NOTICE file (its file listing);
- NeMo 3.0.0 cannot load Nemotron 3; NVIDIA-NeMo/Speech at
  cd61689d41b1962d3e1957ae8faf28bf8bee0e3b can, and Transformers at
  0f7a8768733c448751f25abcb4b6b3ffef829cf5 runs it (torch 2.14.1, CPU).
  Nemotron-3-Diarization.nemo's sha256 is
  867c53f552998f772e5b5e5c082962ae85ee7ca5669c2bc17d7f615133d4e96d,
  model.safetensors'
  c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6;
- Nemotron 3's tables are Parakeet's rounded to bfloat16, NeMo's features
  are Transformers' with those tables bit for bit, and Transformers as
  shipped computes Parakeet's (test.wav);
- the two implementations against each other, and the reference against
  each, on ten recordings (section 4), the cuts' gaps on two, and torch's
  order for ties (section 5); v2's reference against NeMo 3.0.0;
- nemotron3.py's erf against the C library's, and sortformer.py's v2 output
  on test.wav unchanged bit for bit by the cache's move to serve both
  models (at the same BLAS threading; the float64 sums' order follows it);
- NeMo's DER and the words' speakers on nine synthetic conversations, for
  both models, and NeMo's rule and 80 ms frames for words (its source)
  (section 8);
- a Nemotron 3 step and v2's Transformer on pktito's kernels, pktito's
  encoder at v2's shapes, and Parakeet's minute, on one build (section 7);
- the CLI's whole-recording buffers (its source);
- reference.py's change leaves Parakeet's features, encoder output and
  tokens on the first 12 s of test.wav bit for bit, and `make check-c`
  passes on this machine.

Inferred, to be measured:

- the diarizer's whole time in C, and every number for the Pi 5 and the Pi
  3B+, time or memory (the multiply-adds of section 7 are this study's own
  count, from the shapes; the step's time is pktito's kernels in a scratch
  program, not core/diar.c);
- GELU's cost as SiLU's;
- what int8 costs the diarizer;
- why the two float32 implementations make the same close calls;
- v2.1's network being v2's;
- the size of the .pkf;
- the corpora's licences.
