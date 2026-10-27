# Fuzzing

The fuzzers are a development check built with Clang's libFuzzer, ASan and
UBSan, on Linux or macOS (`wav_bytes` needs POSIX's `fmemopen`).  They add
no dependency to libpktito itself.

```
make fuzz                         # build and make the seed corpora
make fuzz-replay                  # every input in the corpora once
make fuzz-smoke FUZZ_TIME=30      # every target for about 30 seconds
make fuzz-run FUZZ=wav_bytes FUZZ_TIME=3600
```

Their live corpora are under `build/fuzz/corpus`, which git ignores and
`make clean` and `make remote` leave alone.  A crash is
reproduced with the command libFuzzer prints and minimized with
`-minimize_crash=1`.  Its fix also gets a focused case in the ordinary check
that owns the rule; a parser finding belongs in `pkfcheck.c` or `wavcheck.c`,
not only in an opaque corpus file.

## A box that fuzzes for good

`tests/fuzz/campaign.sh` runs every target at once in hour-long cycles.
Each target gets its share of the workers. What crashed is replayed and
grouped by its sanitizer's kind and first three frames in the tree. The
first input of each group goes to `findings/` and the rest are counted.
Each corpus is minimized with `-merge=1`, and `STATUS` is written. Its
defaults suit 16 threads and 32 GB; its head lists what can be changed.
On Linux, a user unit keeps it running:

```
# ~/.config/systemd/user/pktito-fuzz.service
[Unit]
Description=pktito fuzzing
[Service]
ExecStart=%h/libpktito/tests/fuzz/campaign.sh
Environment=UPDATE=1
Nice=10
Restart=on-failure
RestartSec=60
[Install]
WantedBy=default.target
```

`systemctl --user enable --now pktito-fuzz`, then `loginctl enable-linger`
so it runs without a login. `cat ~/pktito-fuzz/STATUS` shows it. A finding
is reproduced with `build/fuzz/<target> findings/<id>/input`.

## Targets

- `pkf_bytes` sends arbitrary bytes through `pk_pkf_validate` and, once they
  are byte-valid, through the semantic model binder.  A model the binder
  accepts is inspected through the public metadata and vocabulary API.  Its
  4 MB input limit holds the complete 1.2 MB tiny model while keeping
  mutations cheap; the format's size arithmetic itself remains covered by
  `pkfcheck.c`'s synthetic maximum fields.
- `wav_bytes` opens the same WAV or AIFF bytes three times.  Mono reads of one
  and 4096 frames must be bit-identical, and interleaved reads of seven frames
  must contain the same frames and average to the mono result.  The target
  includes core/wav.c and hands its parser an `fmemopen` stream, so the
  library keeps one way in and no iteration writes a file.
- `resample` makes bounded rate pairs and arbitrary float bit patterns.  The
  output sent in one call and in fuzz-selected pieces must have the same bits,
  the documented room must bound every call, and the final count must be
  `ceil(to * input / from)`.
- `state` selects probabilities and segmentation parameters, a valid sorted
  speech layout and audio, or caption words and timings.  It checks capacity
  independence, ordered and bounded output, speech coverage, and caption cue
  coverage.  Invalid pointers, undersized caller buffers and unsorted speech
  are outside the public contracts and are not generated.
- `audio_pipeline` loads the synthetic tiny model once and interprets every
  remaining four bytes as an exact float bit pattern.  It checks ASR token,
  text and word contracts; VAD batch and frame-at-a-time identity; segment and
  chunk bounds; and, on selected inputs, transcribes every derived chunk.
- `model_semantic` treats its first bytes as typed patches to the tiny model:
  arbitrary q8 weights, finite float tensors, bounded durations and
  `max_symbols`, and same-size ASCII changes to pieces and metadata.  The
  patch language preserves the file layout so most inputs reach inference;
  the remaining bytes are exact float audio samples.  This is deliberately a
  slower target than `pkf_bytes`.

The fast targets link only `backend/ref`: their coverage does not change with
the host CPU, and failures reproduce anywhere.  Native vector instructions
are checked separately against ref's bits.
