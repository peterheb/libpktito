# Examples

The library from the outside. transcribe.c is a whole program against it in
a page: a WAV or AIFF file to its transcript, cut at its silences. The others
are the same program in another language, and print what it prints:

| | | through | links | needs |
|---|---|---|---|---|
| C | transcribe.c | | libpktito.a | `make examples/transcribe` |
| Python | python/transcribe.py | ctypes | libpktito.so | Python 3.8, its standard library |
| Go | go/main.go | cgo | libpktito.a | Go 1.21 and a C compiler |
| Rust | rust/src/main.rs | `extern "C"` | libpktito.a | Rust 1.73, no crates |
| Node | node/transcribe.mjs | koffi 3 | libpktito.so | Node, and `npm install` there |

Go and Rust link the archive, `make libpktito.a`, into one binary that needs
nothing beside it but the model, as ./pktito-c does. Python and Node can
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
  encoder, hands each chunk's to a callback instead, which these examples
  leave out.
- **Text is UTF-8**, NUL-terminated, in a buffer the caller sized; so are
  paths, on every system, Windows too.
- **Threads.** A model may be shared by any number of threads; a
  transcriber belongs to one at a time and runs its own pool, so a call
  into it is a long one, and holds whatever the language gives the call:
  ctypes lets go of Python's lock for it, and a cgo call holds only its
  goroutine, but a koffi call holds Node's one thread, which a server gives
  back with koffi's async form.

`make check-ffi` runs each example whose language is here on test.wav and
holds its transcript to ./pktito-c's, as `make check-c` holds transcribe.c
to it; one whose language is not is skipped with a line. It is not part of
`make check`.
