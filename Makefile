# pktito
#
#   make                          ./pktito-c for this machine's family, every backend in it:
#                                 neon-dp and neon on AArch64; amx-int8, avx512-vnni, avx2-vnni
#                                 and avx2 on x86-64; and ref. The CPU picks one when it runs.
#   make BACKEND=<name>           that backend alone, with ref:
#                                 neon     ARMv8.0 NEON: Pi 3 and Zero 2 W (A53), Pi 4 (A72)
#                                 neon-dp  ARMv8.2 NEON + SDOT: Pi 5 (A76), RK3588, Apple M
#                                 avx2     x86-64 AVX2 + FMA
#                                 avx2-vnni    and AVX-VNNI: Alder Lake and later, Zen 5
#                                 avx512-vnni  and AVX512-VNNI in zmm: Ice Lake, Zen 4 and 5
#                                 amx-int8     and AMX's int8 tiles: Sapphire and Granite Rapids
#                                 ref      scalar C, any host
#   PKTITO_KERNELS=<name> ./pktito-c  runs that backend, of those built, to compare them
#   make libpktito.a               the library alone (examples/transcribe.c uses it)
#   make shared                   the library as libpktito.so (.dylib on macOS, .dll on
#                                 Windows), for other languages: examples/README.md
#   make models                   models/*.pkf from the checkpoints (Python with numpy)
#   make check                    the C half and, where it can run, the Python half
#   make check-c                  what needs only a compiler and a model: runs on any board
#   make check-sanitize           the two file parsers under ASan and UBSan
#   make check-math               every float through exp, sigmoid, tanh and log
#   make check-ffi                the examples in other languages, where they are installed
#   make fuzz                     coverage-guided harnesses (Clang libFuzzer)
#   make fuzz-smoke               a short sanitizer-backed run of every harness
#   make remote HOST=pi           rsync, make, check-c and bench over ssh
#   make adb NDK=<ndk>            the same on an Android device over adb, cross-built here
#
# No dependency beyond the C library, libm and pthreads.

CC ?= cc

# The backends. A build holds every one of the machine's family that the
# compiler can make - neon-dp and neon on AArch64; amx-int8, avx512-vnni,
# avx2-vnni and avx2 on x86-64 - each compiled with its own instruction
# set, and ref; the rest of the tree takes the family's baseline. Which runs
# is the CPU's to say, at run time (backend/kernels.c), so one binary serves
# its family: built on an Alder Lake it runs on a Zen 4, built on a Pi 5 on
# a Pi 3.
# BACKEND=<name> builds that one alone, with ref.
UNAME_M := $(shell uname -m)
UNAME_S := $(shell uname -s)
ifneq ($(filter aarch64 arm64,$(UNAME_M)),)
  FAMILY := neon-dp neon
else ifneq ($(filter x86_64 amd64,$(UNAME_M)),)
  FAMILY := amx-int8 avx512-vnni avx2-vnni avx2
endif

# Each backend's instruction set, and tuning for the core the build is on
# where gcc can be told which: it resolves -mtune=native to nothing on an
# A53, so the core is read from /proc/cpuinfo. Tuning orders instructions;
# it adds none a CPU might lack. backend/avx2 makes avx2-vnni or
# avx512-vnni from what its compiler is told: AVX-VNNI's VEX VPDPBUSD in
# ymm, or AVX512-VNNI's in zmm.
CPU_PART := $(shell grep -m1 'CPU part' /proc/cpuinfo 2>/dev/null | awk '{print $$NF}')
TUNE_0xd03 := cortex-a53
TUNE_0xd08 := cortex-a72
TUNE_0xd0b := cortex-a76
TUNE := $(if $(TUNE_$(CPU_PART)),-mtune=$(TUNE_$(CPU_PART)))
ISA_neon := -march=armv8-a $(TUNE)
ifeq ($(UNAME_S),Darwin)
  ISA_neon-dp := -mcpu=apple-m1
else
  ISA_neon-dp := -march=armv8.2-a+dotprod $(TUNE)
endif
ISA_avx2 := -mavx2 -mfma
ISA_avx2-vnni := -mavx2 -mfma -mavxvnni
ISA_avx512-vnni := -mavx2 -mfma -mavx512f -mavx512vnni -mavx512vl
ISA_amx-int8 := $(ISA_avx512-vnni) -mamx-tile -mamx-int8

can_build = $(shell echo 'int x;' | $(CC) $(ISA_$(1)) -x c -c -o /dev/null - 2>/dev/null && echo $(1))
BACKENDS := $(filter-out ref,$(if $(BACKEND),$(BACKEND),$(foreach b,$(FAMILY),$(call can_build,$(b)))))

# -ffp-contract=off is not a style choice. Every backend computes ref's bits
# (backend/kernels.h), and that holds only if no compiler fuses a multiply
# and an add that the code wrote apart: a contraction gcc makes on one target
# and not another would round differently there. Fusion is spelt out, with
# fmaf, where it is meant. -fno-math-errno makes sqrtf and sqrt the CPU's
# square root, the same correctly rounded bits, rather than a call into
# libm for errno's sake: glibc 2.43's sqrtf is a GLIBC_2.43 symbol, and with
# it gone a build on Ubuntu 26.04 needs GLIBC_2.34 (check-c there, v2 and v3
# transcripts unchanged).
# Windows builds under MSYS2 (UCRT64) or another MinGW-w64: pthreads from
# winpthreads, linked in, so that pktito-c.exe runs outside the shell it was
# built in with nothing beside it; and MinGW's own printf, since a
# toolchain on the old msvcrt runtime has none that knows %zu (UCRT's does,
# and its output is the same bytes either way: check-c on the Core 3 304).
# shell32 is a system DLL, for CommandLineToArgvW: the CLI's arguments as
# UTF-8 (core/cli.c, utf8_args). gcc and clang for MinGW link it by
# default; it is named so that no toolchain has to.
ifeq ($(UNAME_S),Darwin)
  PLATFLAGS := -D_DARWIN_C_SOURCE
else ifneq ($(filter MINGW% MSYS% UCRT% CLANG%,$(UNAME_S)),)
  PLATFLAGS := -D_WIN32_WINNT=0x0A00 -D__USE_MINGW_ANSI_STDIO=1
  PLATLIBS := -static -lshell32
else
  PLATFLAGS := -D_DEFAULT_SOURCE -D_POSIX_C_SOURCE=200809L
endif
# Android's C library, bionic, holds pthreads itself and has no libpthread
# to link; the NDK's clang names its target (-dumpmachine) as Android.
ifneq ($(findstring android,$(shell $(CC) -dumpmachine 2>/dev/null)),)
  PTHREAD :=
else
  PTHREAD := -lpthread
endif
OPT     ?= -O3
CFLAGS  ?= $(OPT) $(TUNE) -std=c11 -ffp-contract=off -fno-math-errno -Wall -Wextra -Wpedantic
LDLIBS  ?= -lm $(PTHREAD) $(PLATLIBS)

# make PROFILE=1: `pktito-c bench` also prints what each stage costs. Timing a
# stage at a time costs a little; quote bench without it.
ifeq ($(PROFILE),1)
  CFLAGS += -DPK_PROFILE
endif

INC := -Icore -Ibackend
LIB_SRC := core/model.c core/os.c core/pool.c core/mel.c core/encoder.c core/decoder.c \
           core/tokens.c core/asr.c core/vad.c core/segment.c core/wav.c core/resample.c core/captions.c \
           core/language.c
BACK_OBJ := $(foreach b,$(BACKENDS) ref,backend/$(b)/kernels.o) backend/kernels.o
LIB_OBJ := $(LIB_SRC:.c=.o) $(BACK_OBJ)
HAVE := $(foreach b,$(BACKENDS),-DPK_HAVE_$(subst -,_,$(b)))
$(foreach b,$(BACKENDS),$(eval backend/$(b)/kernels.o backend/$(b)/kernels.pic.o: CFLAGS += $(ISA_$(b))))
backend/kernels.o backend/kernels.pic.o: CFLAGS += $(HAVE)

.DEFAULT_GOAL := pktito-c

pktito-c: core/cli.o $(LIB_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

libpktito.a: $(LIB_OBJ)
	$(AR) rcs $@ $^

# The library as a shared object, for a language that loads one: Python's
# ctypes, Node's koffi, C#'s P/Invoke (examples/README.md). Its objects are
# compiled again, beside the others as *.pic.o, so that ./pktito-c and
# libpktito.a stay what they were: with -fPIC, which ELF needs and macOS has
# by default, and without it for a DLL, which Windows relocates as it loads
# and which takes in winpthreads as pktito-c.exe does. Every name in the
# library is pk_'s, so it exports them all, as libpktito.a does; core/pktito.h,
# wav.h and resample.h say which are the interface. Built by gcc, on x86-64
# it runs 2% behind ./pktito-c on one thread and 3.5 to 7% on sixteen (the
# 9950X), clang's level: gcc reaches the zmm product's thread-local copy
# through a call to __tls_get_addr at each use in its tile loop, which no
# flag cures that dlopen allows (backend/avx2/CLIMB.md, 2026-10-27).
ifeq ($(UNAME_S),Darwin)
  SHLIB   := libpktito.dylib
  SHFLAGS := -dynamiclib -install_name @rpath/$(SHLIB)
  PIC     := -fPIC
else ifneq ($(filter MINGW% MSYS% UCRT% CLANG%,$(UNAME_S)),)
  SHLIB   := libpktito.dll
  SHFLAGS := -shared
else
  SHLIB   := libpktito.so
  SHFLAGS := -shared
  PIC     := -fPIC
endif
SH_OBJ := $(LIB_OBJ:.o=.pic.o)

shared: $(SHLIB)
$(SHLIB): $(SH_OBJ)
	$(CC) $(CFLAGS) $(SHFLAGS) -o $@ $^ $(LDLIBS)

%.pic.o: %.c
	$(CC) $(CFLAGS) $(PIC) $(PLATFLAGS) $(INC) -MMD -MP -c $< -o $@

# The library from the outside: examples/transcribe.c is a page of it.
examples/transcribe: examples/transcribe.c libpktito.a
	$(CC) $(CFLAGS) -Icore -o $@ examples/transcribe.c libpktito.a $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) $(PLATFLAGS) $(INC) -MMD -MP -c $< -o $@

-include $(LIB_OBJ:.o=.d) $(SH_OBJ:.o=.d) core/cli.d

# ---------------------------------------------------------------- models ----

# The checkpoints, from HuggingFace, and Silero's VAD from its repository at
# v6.2.3; the converter needs Python with numpy and nothing else.
PYTHON ?= $(firstword $(wildcard .venv/bin/python) $(shell command -v python3))
SRC    ?= models/src
VAD    ?= $(SRC)/silero-vad/src/silero_vad/data/silero_vad.onnx
models: models/parakeet-tdt-0.6b-v2.pkf models/parakeet-tdt-0.6b-v3.pkf
models/parakeet-tdt-0.6b-v2.pkf: $(SRC)/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo
models/parakeet-tdt-0.6b-v3.pkf: $(SRC)/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo
models/%.pkf:
	$(PYTHON) tools/pktito-tools convert $< $(VAD) $@

# ---------------------------------------------------------------- tests ----

MODEL ?= models/parakeet-tdt-0.6b-v2.pkf
TINY  := tests/data/tiny.pkf
TESTS := tests/kcheck tests/mathcheck tests/pkfcheck tests/wavcheck tests/capcheck tests/langcheck tests/chunkcheck \
         tests/pathcheck
BENCHES := tests/q8bench tests/gemmbench tests/spinbench

# kcheck holds every backend of the build that this CPU can run to ref, bit
# for bit, in one binary.
tests/kcheck: tests/kcheck.c $(LIB_OBJ)
	$(CC) $(CFLAGS) $(PLATFLAGS) $(INC) -o $@ tests/kcheck.c $(LIB_OBJ) $(LDLIBS)
tests/mathcheck: tests/mathcheck.c core/vmath.h
	$(CC) $(CFLAGS) $(INC) -o $@ tests/mathcheck.c $(LDLIBS)
tests/pkfcheck: tests/pkfcheck.c $(LIB_OBJ)
	$(CC) $(CFLAGS) $(PLATFLAGS) $(INC) -o $@ tests/pkfcheck.c $(LIB_OBJ) $(LDLIBS)
tests/chunkcheck: tests/chunkcheck.c $(LIB_OBJ)
	$(CC) $(CFLAGS) $(PLATFLAGS) $(INC) -o $@ tests/chunkcheck.c $(LIB_OBJ) $(LDLIBS)
tests/pathcheck: tests/pathcheck.c $(LIB_OBJ)
	$(CC) $(CFLAGS) $(PLATFLAGS) $(INC) -o $@ tests/pathcheck.c $(LIB_OBJ) $(LDLIBS)
tests/wavcheck: tests/wavcheck.c core/wav.o core/os.o
	$(CC) $(CFLAGS) $(PLATFLAGS) $(INC) -o $@ tests/wavcheck.c core/wav.o core/os.o $(LDLIBS)
# capcheck includes core/captions.c itself, for its static parts, and
# langcheck core/language.c
tests/capcheck: tests/capcheck.c core/captions.c core/language.c core/langwords.h core/pktito.h
	$(CC) $(CFLAGS) $(INC) -o $@ tests/capcheck.c core/language.c $(LDLIBS)
tests/langcheck: tests/langcheck.c core/language.c core/langwords.h core/pktito.h
	$(CC) $(CFLAGS) $(INC) -o $@ tests/langcheck.c $(LDLIBS)

# The int8 product alone, for the kernel climbs, and not part of any check:
# q8bench on one thread (backend/avx2/CLIMB.md's numbers), gemmbench over
# threads dealt as the pool deals them, with MAC/cycle (backend/neon's).
tests/q8bench: tests/q8bench.c $(BACK_OBJ) core/os.o
	$(CC) $(CFLAGS) $(PLATFLAGS) $(INC) -o $@ tests/q8bench.c $(BACK_OBJ) core/os.o $(LDLIBS)
tests/gemmbench: tests/gemmbench.c $(BACK_OBJ) core/os.o
	$(CC) $(CFLAGS) $(PLATFLAGS) $(INC) -o $@ tests/gemmbench.c $(BACK_OBJ) core/os.o $(LDLIBS)
# spinbench includes core/pool.c, for the worker's spin: how long PK_SPIN lasts
tests/spinbench: tests/spinbench.c core/pool.c core/pool.h core/os.o
	$(CC) $(CFLAGS) $(PLATFLAGS) $(INC) -o $@ tests/spinbench.c core/os.o $(LDLIBS)
benches: $(BENCHES)

# check-c needs a compiler and nothing else but MODEL, so it runs on any
# board: the math library; every kernel against ref; the model file's
# validator on its malformed corpus and the whole engine on the tiny model
# (tests/data/tiny.pkf, `pktito-tools tiny` writes it); the WAV reader on its
# own; a WAV and a model under a name past ASCII, through the library and
# through the CLI's arguments (a path is UTF-8 everywhere); the CLI on the tiny
# model's 35 s recording, which it cuts into chunks; and Silero's test.wav
# through ./pktito-c. Its VAD probabilities must be
# tests/data/test.probs.txt exactly - vadito's bits, which vadito holds to
# onnxruntime - and its transcripts, every token and its time, the bytes in
# test.v2.json and test.v3.json, which ref wrote: every backend on every
# machine computes the same ones (backend/kernels.h), and this is where a
# board that does not would say so. A model file that is not there is
# skipped, with a line. The commands are tests/check-c.sh, so that a board
# without make, Android over adb, runs the same ones.
check-c: pktito-c $(TESTS) examples/transcribe
	@MODEL=$(MODEL) sh tests/check-c.sh

# The half that needs Python and NeMo's checkpoints to compare with: the
# detokenizer against sentencepiece (whose package NeMo brings; PYNEMO is a
# Python that has it), and, where tools/corpus.py's sets are, the WER over
# them against float32's transcripts from tools/quantcheck.py. Each is
# skipped with a line when what it needs is not here.
PYNEMO ?= $(firstword $(wildcard .venv-nemo/bin/python) $(PYTHON))
SILERO ?= $(firstword $(wildcard $(SRC)/silero-vad-master) $(wildcard $(SRC)/silero-vad))
PYTESTS := tests/tokcheck tests/segcheck tests/rscheck
tests/tokcheck: tests/tokcheck.c core/model.o core/os.o core/tokens.o $(BACK_OBJ)
	$(CC) $(CFLAGS) $(PLATFLAGS) $(INC) -o $@ tests/tokcheck.c core/model.o core/os.o core/tokens.o \
	  $(BACK_OBJ) $(LDLIBS)
tests/segcheck: tests/segcheck.c core/segment.o
	$(CC) $(CFLAGS) $(INC) -o $@ tests/segcheck.c core/segment.o $(LDLIBS)
tests/rscheck: tests/rscheck.c core/resample.o core/os.o $(BACK_OBJ)
	$(CC) $(CFLAGS) $(PLATFLAGS) $(INC) -o $@ tests/rscheck.c core/resample.o core/os.o $(BACK_OBJ) $(LDLIBS)
check-python: pktito-c $(PYTESTS)
	@if [ -n "$(SILERO)" ]; then SILERO=$(SILERO) $(PYNEMO) -W ignore tests/segcheck.py tests/segcheck || exit 1; \
	else echo "segcheck: no silero-vad checkout in $(SRC) - skipped"; fi
	@$(PYNEMO) -W ignore tests/rscheck.py tests/rscheck || exit 1
	@for v in v2 v3; do n=$(SRC)/parakeet-tdt-0.6b-$$v/parakeet-tdt-0.6b-$$v.nemo; \
	  f=models/parakeet-tdt-0.6b-$$v.pkf; \
	  if [ -f $$n ] && [ -f $$f ]; then $(PYNEMO) tests/tokcheck.py tests/tokcheck $$n $$f || exit 1; \
	  else echo "tokcheck: no $$n or $$f - skipped"; fi; done
	@if [ -d models/corpus/ls200 ] && [ -f models/corpus/qc-v2-ls200.json ]; then \
	  $(PYTHON) tests/wercheck.py ./pktito-c models/parakeet-tdt-0.6b-v2.pkf models/corpus/ls200 \
	    --against models/corpus/qc-v2-ls200.json || exit 1; \
	else echo "wercheck: no models/corpus/ls200 - skipped"; fi
	@if [ -d models/corpus/fleurs-de_de ] && [ -f models/corpus/qc-v3-fleurs.json ]; then \
	  $(PYTHON) tests/wercheck.py ./pktito-c models/parakeet-tdt-0.6b-v3.pkf models/corpus/fleurs-*_* \
	    --against models/corpus/qc-v3-fleurs.json || exit 1; \
	else echo "wercheck: no models/corpus/fleurs-* - skipped"; fi

check: check-c check-python

# The two parsers of untrusted bytes and the engine behind them, under ASan
# and UBSan, on the same corpora, against backend/ref. Apart from check-c
# because not every board's compiler ships the sanitizer runtimes.
SAN := -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer
check-sanitize:
	$(CC) $(SAN) -std=c11 -ffp-contract=off -Wall -Wextra $(PLATFLAGS) $(INC) -o tests/pkfcheck-san \
	  tests/pkfcheck.c $(LIB_SRC) backend/ref/kernels.c backend/kernels.c $(LDLIBS)
	$(CC) $(SAN) -std=c11 -ffp-contract=off -Wall -Wextra $(PLATFLAGS) $(INC) -o tests/wavcheck-san \
	  tests/wavcheck.c core/wav.c core/os.c $(LDLIBS)
	ASAN_OPTIONS=detect_leaks=0 ./tests/pkfcheck-san $(TINY)
	ASAN_OPTIONS=detect_leaks=0 ./tests/wavcheck-san tests/wavcheck.tmp

# The thread pool and every pass split over it, under ThreadSanitizer: a
# transcription and a stream on six threads, which must report nothing.
TSAN := -O1 -g -fsanitize=thread -std=c11 -ffp-contract=off $(PLATFLAGS) $(INC)
check-tsan:
	$(foreach b,$(BACKENDS) ref,$(CC) $(TSAN) $(ISA_$(b)) -c backend/$(b)/kernels.c -o tests/tsan-$(b).o &&) true
	$(CC) $(TSAN) $(HAVE) -o tests/pktito-tsan core/cli.c $(LIB_SRC) backend/kernels.c \
	  $(foreach b,$(BACKENDS) ref,tests/tsan-$(b).o) $(LDLIBS)
	TSAN_OPTIONS=halt_on_error=1 ./tests/pktito-tsan -t 6 -m $(MODEL) tests/data/test.wav > /dev/null
	TSAN_OPTIONS=halt_on_error=1 ./tests/pktito-tsan -t 6 --stream -m $(MODEL) tests/data/test.wav > /dev/null
	@# and on the default threads, which on a chip of two kinds of core switch
	@# between every core and the fast kind as the first recording is timed
	TSAN_OPTIONS=halt_on_error=1 ./tests/pktito-tsan -m $(MODEL) tests/data/test.wav > /dev/null
	@echo "check-tsan: no race on six threads, nor on the default"

check-math: tests/mathcheck
	./tests/mathcheck --all

# The examples in other languages (examples/README.md), through the shared
# library: each whose language is here must print test.wav's transcript as
# ./pktito-c does, as check-c holds examples/transcribe to it, and one whose
# language is not is skipped with a line. Apart from check, since they are
# examples of the wiring, not the library: a language missing or changed on
# a board is no reason to hold up a commit.
check-ffi: pktito-c $(SHLIB)
	@if [ -n "$(PYTHON)" ]; then \
	  $(PYTHON) examples/python/transcribe.py $(MODEL) tests/data/test.wav | tr -d '\r' | cut -d' ' -f3- > tests/ffi.tmp; \
	  ./pktito-c -m $(MODEL) tests/data/test.wav | cmp -s - tests/ffi.tmp \
	    && echo "examples/python: the CLI's transcript, through ctypes" \
	    || { echo "examples/python: FAIL - not the CLI's transcript"; rm -f tests/ffi.tmp; exit 1; }; \
	else echo "examples/python: no python3 - skipped"; fi; rm -f tests/ffi.tmp

# --------------------------------------------------------------- fuzzing ----

# Fuzzing is a development tool, not a library dependency.  The targets use
# backend/ref so that their coverage and failures do not depend on the CPU
# running them; tests/kcheck holds the native kernels to ref's bits.
FUZZ_CC      ?= clang
FUZZ_DIR     := build/fuzz
FUZZ_CORPUS  := $(FUZZ_DIR)/corpus
FUZZ_SAN     := -O1 -g -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all \
                -fno-omit-frame-pointer
FUZZ_CFLAGS  := $(FUZZ_SAN) -std=c11 -ffp-contract=off -fno-math-errno -Wall -Wextra -Wpedantic \
                $(PLATFLAGS) $(INC) -Itests/fuzz
FUZZ_NAMES   := pkf_bytes wav_bytes resample state audio_pipeline model_semantic
FUZZ_BINS    := $(addprefix $(FUZZ_DIR)/,$(FUZZ_NAMES))
FUZZ_KERNELS := backend/ref/kernels.c backend/kernels.c
FUZZ_REF     := $(FUZZ_KERNELS) core/os.c
# Every header, for each target: one clang call compiles and links a target,
# and -MMD keeps only its last source's dependencies.
FUZZ_HDRS    := $(wildcard core/*.h backend/*.h backend/*/*.h tests/fuzz/*.h)
FUZZ_TIME    ?= 5
FUZZ_MAX_pkf_bytes := 4194304
FUZZ_MAX_wav_bytes := 1048576
FUZZ_MAX_resample  := 8192
FUZZ_MAX_state     := 65536
FUZZ_MAX_audio_pipeline := 131073
FUZZ_MAX_model_semantic := 8192
# target, its corpus and its input limit: what every run of one starts with,
# with any ASAN_OPTIONS the caller set taking precedence
FUZZ_ONE = ASAN_OPTIONS=detect_leaks=1$${ASAN_OPTIONS:+:$$ASAN_OPTIONS} $(FUZZ_DIR)/$(1) $(FUZZ_CORPUS)/$(1) -max_len=$(FUZZ_MAX_$(1))

$(FUZZ_DIR):
	mkdir -p $@

$(FUZZ_DIR)/pkf_bytes: tests/fuzz/pkf_bytes.c $(FUZZ_HDRS) core/model.c core/tokens.c $(FUZZ_REF) | $(FUZZ_DIR)
	$(FUZZ_CC) $(FUZZ_CFLAGS) -o $@ tests/fuzz/pkf_bytes.c core/model.c core/tokens.c $(FUZZ_REF) $(LDLIBS)

# wav_bytes includes core/wav.c, to reach its parser without a path
$(FUZZ_DIR)/wav_bytes: tests/fuzz/wav_bytes.c $(FUZZ_HDRS) core/wav.c core/os.c | $(FUZZ_DIR)
	$(FUZZ_CC) $(FUZZ_CFLAGS) -o $@ tests/fuzz/wav_bytes.c core/os.c $(LDLIBS)

$(FUZZ_DIR)/resample: tests/fuzz/resample.c $(FUZZ_HDRS) core/resample.c core/resample.h $(FUZZ_REF) | $(FUZZ_DIR)
	$(FUZZ_CC) $(FUZZ_CFLAGS) -o $@ tests/fuzz/resample.c core/resample.c $(FUZZ_REF) $(LDLIBS)

$(addprefix $(FUZZ_DIR)/,state audio_pipeline model_semantic): $(FUZZ_DIR)/%: tests/fuzz/%.c $(FUZZ_HDRS) \
                $(LIB_SRC) $(FUZZ_KERNELS) | $(FUZZ_DIR)
	$(FUZZ_CC) $(FUZZ_CFLAGS) -o $@ $< $(LIB_SRC) $(FUZZ_KERNELS) $(LDLIBS)

$(FUZZ_DIR)/seeds: tests/fuzz/seeds.c | $(FUZZ_DIR)
	$(CC) -O2 -std=c11 -Wall -Wextra -Wpedantic -o $@ $<

fuzz-corpus: $(FUZZ_DIR)/seeds
	mkdir -p $(foreach f,$(FUZZ_NAMES),$(FUZZ_CORPUS)/$(f))
	cp $(TINY) $(FUZZ_CORPUS)/pkf_bytes/tiny.pkf
	$(FUZZ_DIR)/seeds $(FUZZ_CORPUS)

fuzz: $(FUZZ_BINS) fuzz-corpus

fuzz-smoke: fuzz
	@set -e; $(foreach f,$(FUZZ_NAMES),echo "fuzz-smoke: $(f)"; \
	  $(call FUZZ_ONE,$(f)) -max_total_time=$(FUZZ_TIME) -timeout=10 -rss_limit_mb=2048 \
	  -print_final_stats=1 -verbosity=0;)

fuzz-run: fuzz
	@[ -n "$(FUZZ)" ] || { echo "make fuzz-run FUZZ=wav_bytes [FUZZ_TIME=3600]"; exit 1; }
	$(call FUZZ_ONE,$(FUZZ)) -max_total_time=$(FUZZ_TIME) -timeout=10 -rss_limit_mb=2048 -print_final_stats=1

# a target's input limit, for tests/fuzz/campaign.sh
fuzz-max:
	@echo $(FUZZ_MAX_$(FUZZ))

fuzz-replay: fuzz
	@set -e; $(foreach f,$(FUZZ_NAMES),echo "fuzz-replay: $(f)"; $(call FUZZ_ONE,$(f)) -runs=0 -verbosity=0;)

# The loop a board gets: the tree to HOST:$(RDIR) (the models too, rsync
# sends only what changed), then the build, the compiler half of the checks
# and a bench, there. BENCH names the recording; Silero's test.wav, which
# the tree carries, is the default.
HOST  ?=
RDIR  ?= pktito
BENCH ?= tests/data/test.wav
remote:
	@[ -n "$(HOST)" ] || { echo "make remote HOST=user@board"; exit 1; }
	rsync -a --delete --exclude .venv --exclude .venv-nemo --exclude .git --exclude '*.o' \
	  --exclude '*.d' --exclude /pktito-c --exclude /pktito --exclude 'models/src' --exclude 'models/corpus' \
	  --exclude '/tests/*check' --exclude /build ./ $(HOST):$(RDIR)/
	ssh $(HOST) 'cd $(RDIR) && make -s clean && make -s -j4 $(if $(BACKEND),BACKEND=$(BACKEND)) check-c MODEL=$(MODEL) && ./pktito-c bench -m $(MODEL) $(BENCH)'

# The same loop for an Android device, over adb: the tree cross-built here
# with the NDK's clang for the device's ABI, the programs, test data and
# models pushed, and check-c and a bench run there (tests/adb.sh says how).
# NDK is the NDK's folder; ANDROID_SERIAL picks the device of several.
NDK ?=
adb:
	@NDK="$(NDK)" TESTS="$(TESTS)" MODEL="$(MODEL)" BENCH="$(BENCH)" BACKEND="$(BACKEND)" sh tests/adb.sh

.PHONY: models shared check-c check-sanitize check-math check-ffi fuzz fuzz-corpus fuzz-smoke fuzz-run fuzz-max fuzz-replay remote adb clean benches

clean:
	rm -f pktito-c pktito libpktito.a libpktito.so libpktito.dylib libpktito.dll examples/transcribe
	rm -f core/*.o core/*.d backend/*.o backend/*.d backend/*/*.o backend/*/*.d tests/*.o tests/*.d $(TESTS) $(PYTESTS) $(BENCHES)
	rm -f tests/*-san tests/pktito-tsan tests/tsan-*.o tests/*.tmp tests/pkfcheck.wav tests/pkfcheck.json
	rm -rf tests/*.dSYM tests/pathcheck-* build/android-*
	rm -f $(FUZZ_BINS) $(FUZZ_DIR)/seeds   # not the corpora a campaign grew
