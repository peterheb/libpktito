# Examples

The library from the outside. transcribe.c is a whole program against it in
a page: a WAV or AIFF file to its transcript, cut at its silences. The others
are the same program in another language, and print what it prints:

|        |                      | through      | links        | needs                            |
| ------ | -------------------- | ------------ | ------------ | -------------------------------- |
| C      | transcribe.c         |              | libpktito.a  | `make examples/transcribe`       |
| Python | python/transcribe.py | ctypes       | libpktito.so | Python 3.8, its standard library |
| Go     | go/main.go           | cgo          | libpktito.a  | Go 1.21 and a C compiler         |
| Rust   | rust/src/main.rs     | `extern "C"` | libpktito.a  | Rust 1.73, no crates             |
| Node   | node/transcribe.mjs  | koffi 3      | libpktito.so | Node, and `npm install` there    |

Go and Rust link the archive, `make libpktito.a`, into one binary that needs
nothing beside it but the model, as ./pktito-c does. cargo relinks for a
new archive (rust/build.rs says it's an input); `go build` relinks for
Go's own files alone, so after a new libpktito.a its binary is deleted
first (go/transcribe, transcribe.exe on Windows), as check.sh does. Python and Node can
only load a shared library: `make shared` builds it,
libpktito.so, libpktito.dylib on macOS, libpktito.dll on Windows (MSYS2, as
for the CLI, with winpthreads in it, so it needs nothing beside it), and an
example finds it at the top of the tree, or where PKTITO_LIB says. npm 11
holds koffi's install script for approval; it only builds koffi from
source where no prebuilt package (`@koromix/koffi-<system>`) fits, and the
example runs without it.

These are examples of the wiring, not packages: nothing here is published
or versioned, and none is a binding kept to core/pktito.h. Each declares the
functions it calls; a program of yours declares the ones it calls the same
way. What every one has to get right:

- **Each type as the header has it.** A function's result, every argument,
  a struct's fields in order. `size_t`, `int64_t` and `long` are not one
  another's size on every system (`long` is 32 bits on Windows), and a
  foreign-function layer that takes a pointer result for an int keeps half
  of it.
- **Handles are opaque.** `pk_model *`, `pk_asr *`, `pk_vad *` and
  `pk_wav *` are held and handed back, never looked into, and freed with
  their own `pk_*_free`, or `pk_wav_close`.
- **Memory is the caller's.** The library fills the arrays it is given,
  at the sizes pktito.h states: samples, probabilities, segments, chunks,
  text. The tokens `pk_asr_run` points at are the transcriber's, and last
  until its next call. `pk_asr_run_chunks`, which takes a recording's
  chunks at once and runs each one's decoder beside the next one's
  encoder, and `pk_asr_transcribe`, the high-level call that does a whole
  recording's VAD, chunks and retries (README, "Transcribing a
  recording"), hand each chunk's to a callback instead, which these
  examples leave out: they show the low-level blocks.
- **A recording is a `pk_audio`**, its length and a function the library
  calls to read a stretch of it. `pk_chunks` takes one; these examples make
  it with `pk_audio_floats` over the samples they hold, read in place, so
  the samples must outlive it, and Go pins them for as long
  (`runtime.Pinner`), since the struct keeps their address past the call
  that made it. A reader of one's own is a callback, as
  `pk_asr_transcribe`'s are, called from the library's threads but never
  two at once.
- **Text is UTF-8**, NUL-terminated, in a buffer the caller sized; so are
  paths, on every system, Windows too.
- **Threads.** A model may be shared by any number of threads; a
  transcriber belongs to one at a time and runs its own pool, so a call
  into it is a long one, and holds whatever the language gives the call:
  ctypes lets go of Python's lock for it, and a cgo call holds only its
  goroutine, but a koffi call holds Node's one thread, which a server gives
  back with koffi's async form.

`make check-ffi` runs each example whose language is here on test.wav and
holds its transcript to that of `./pktito-c --no-context --no-retry`, word
for word (the examples write a chunk a line, the CLI a sentence), as `make
check-c` holds transcribe.c to it; one whose language is not is skipped
with a line. It is not part of
`make check`.

## On an NPU

Each example takes a third argument, an NPU's encoder, and hands it to
`pk_asr_use_encoder`: a compiled Core ML model (a `.mlmodelc` folder) for
Apple's Neural Engine, or a QNN graph (`.qnn`) for a Snapdragon's Hexagon
NPU (README, "The Neural Engine and NPUs"). The library is built for it,
`make ANE=1` or `make QNN=1`, and the model is the slim file, which leaves
its encoder to the NPU; `./pktito-c models` names both for this machine
and prints the commands that fetch them. On a Mac, with Apple's Command
Line Tools and nothing else:

```sh
make ANE=1 pktito-c libpktito.a shared examples/transcribe
./pktito-c models          # then the lines it prints, in models/
./examples/transcribe models/parakeet-tdt-0.6b-v2-slim.pkf talk.wav models/parakeet-tdt-0.6b-v2-encoder.mlmodelc
python3 examples/python/transcribe.py models/parakeet-tdt-0.6b-v2-slim.pkf talk.wav models/parakeet-tdt-0.6b-v2-encoder.mlmodelc
```

and the same three arguments for the others. What differs from the CPU's:

- **Linking.** An archive built with ANE=1 calls Core ML and Foundation.
  go/main.go names both on macOS, which always has them, and a binary of
  an archive built without loads them unused (cgo refuses the
  `-Wl,-dead_strip_dylibs` that would leave them out). rust/build.rs
  names them when build.opts, which make writes beside the archive, says
  ANE=1. The shared library links them itself, so Python and Node need
  nothing more.
- **The first run.** The first time a program runs the encoder, Core ML
  compiles it for the Mac's Neural Engine, 9 s on an M4 mini, and keeps
  the result, about 0.6 GB of disk for v2's, in
  /Library/Caches/com.apple.aned, which only the system clears. pktito-c,
  python3 and node are each a program to it, and so is each example's
  binary; one rebuilt in place keeps its copy. From then on test.wav's
  minute takes 0.3 s.
- **A slim model without its encoder** runs nothing: `pk_asr_run`
  returns -1, and each example says so.
- **A build without ANE=1** refuses the encoder ("this pktito was built
  without ANE=1") and the slim file ("a slim model file, whose encoder is
  an NPU's").

`make check-ffi ANE=1 MODEL=models/parakeet-tdt-0.6b-v2-slim.pkf
ENCODER=models/parakeet-tdt-0.6b-v2-encoder.mlmodelc` holds every example,
examples/transcribe with them, to `./pktito-c --npu`'s transcript. Each
gave it on an M4 Mac mini (macOS 26.4, the Command Line Tools' clang 21)
on 2026-11-08: Apple's python3 3.9.6, Go 1.27.3, Rust 1.99.1, and Node
26.13.0 with koffi 3.3.4.

On Windows on a Snapdragon, in MSYS2's CLANGARM64 shell, with QAIRT's
headers in ~/qairt and QNN's four files at the top of the tree, as the
README's lines put them ("The Neural Engine and NPUs"):

```sh
make QNN=1 QNN_SDK=~/qairt/2.51.0.260929 pktito-c libpktito.a shared examples/transcribe
./pktito-c models          # then the lines it prints, in models/
python examples/python/transcribe.py models/parakeet-tdt-0.6b-v2-slim.pkf talk.wav models/parakeet-tdt-0.6b-v2-x1e80100.qnn
PATH=$PWD:$PATH ./examples/transcribe models/parakeet-tdt-0.6b-v2-slim.pkf talk.wav models/parakeet-tdt-0.6b-v2-x1e80100.qnn
```

What differs from the Mac's:

- **QNN's files.** The library loads QnnHtp.dll from beside the file its
  code is in, and QnnHtp.dll the stub, skel and .cat from beside itself.
  For Python and Node that file is libpktito.dll, at the top of the tree
  with QNN's four. C, Go and Rust link libpktito.a into a program of
  their own, in examples/, examples/go and examples/rust/target/release,
  which finds them on PATH: the top of the tree on it, as above, or a
  folder that holds the four. So does what `go run` and `cargo run` build
  elsewhere. The current folder is never looked in. Where QnnHtp.dll
  isn't found the error names the program it looked beside
  ("QnnHtp.dll is not beside ...\examples\go\transcribe.exe or on PATH").
- **Static binaries.** MSYS2's toolchains link their own DLLs
  (winpthreads, libunwind) by default, which only its shell finds:
  go/main.go links with `-static` on Windows, and
  rust/.cargo/config.toml asks for `crt-static`, so each binary runs from
  cmd or PowerShell, as pktito-c.exe does.
- **No first run.** Nothing is compiled or kept between runs: each loads
  the graph onto the NPU, 1.2 s for v2's 0.7 GB, and test.wav's minute
  then takes 0.52 s.
- **stderr** carries the driver's `DSP_INFO UNSUPPORTED_KEY` lines
  (README); stdout, the transcript, doesn't.

`make check-ffi QNN=1 QNN_SDK=~/qairt/2.51.0.260929
MODEL=models/parakeet-tdt-0.6b-v2-slim.pkf
ENCODER=models/parakeet-tdt-0.6b-v2-x1e80100.qnn` holds them the same way,
with the top of the tree on PATH for C, Go and Rust. Each gave
`./pktito-c --npu`'s transcript on a Snapdragon X Elite X1E-78-100
(Windows 11 build 26300, MSYS2's CLANGARM64 clang 23.1.5, QAIRT
2.51.0.260929) on 2026-11-08: python.org's embeddable Python 3.15.2, Go
1.27.3, Rust 1.99.1 (aarch64-pc-windows-gnullvm), and Node 26.13.0 with
koffi 3.3.4.
