# pkito
#
#   make                          ./pkito for this machine (backend detected)
#   make BACKEND=ref              scalar C, any host
#   make BACKEND=neon             AArch64 NEON, ARMv8.0: Pi 3 and Zero 2 W (A53), Pi 4 (A72)
#   make BACKEND=neon-dp          ARMv8.2 NEON + SDOT: Pi 5 (A76), RK3588, Apple M
#   make BACKEND=avx2             x86-64 AVX2 + FMA
#   make BACKEND=avx2-vnni        x86-64 AVX2 + FMA + AVX-VNNI: Alder Lake and later, Zen 5
#   make libpkito.a               the library alone (examples/transcribe.c uses it)
#   make models                   models/*.pkf from the checkpoints (Python with numpy)
#   make check                    the C half and, where it can run, the Python half
#   make check-c                  what needs only a compiler and a model: runs on any board
#   make check-sanitize           the two file parsers under ASan and UBSan
#   make check-math               every float through exp, sigmoid, tanh and log
#   make remote HOST=pi           rsync, make, check-c and bench over ssh
#
# No dependency beyond the C library, libm and pthreads.

CC ?= cc

# Backend detection: the compiler is asked what it targets natively. AArch64
# with the dot-product instructions gets neon-dp, without them neon; x86-64
# with AVX-VNNI gets avx2-vnni, with AVX2 and FMA avx2; anything else ref.
# A compiler too old to know its own core says nothing, and then Linux's
# /proc/cpuinfo answers (pinfer's Makefile has the story).
UNAME_M := $(shell uname -m)
UNAME_S := $(shell uname -s)
ifneq ($(filter aarch64 arm64,$(UNAME_M)),)
  DOTPROD := $(shell $(CC) -mcpu=native -dM -E - </dev/null 2>/dev/null | grep -c __ARM_FEATURE_DOTPROD)
  ifneq ($(DOTPROD),1)
    DOTPROD := $(shell grep -m1 -cw asimddp /proc/cpuinfo 2>/dev/null)
  endif
  ifeq ($(UNAME_S),Darwin)
    DOTPROD := 1
  endif
  ifeq ($(DOTPROD),1)
    BACKEND ?= neon-dp
  else
    BACKEND ?= neon
  endif
else ifneq ($(filter x86_64 amd64,$(UNAME_M)),)
  X86 := $(shell $(CC) -march=native -dM -E - </dev/null 2>/dev/null \
                 | grep -cE '^\#define (__AVX2__|__FMA__|__AVXVNNI__) ')
  ifeq ($(X86),3)
    BACKEND ?= avx2-vnni
  else ifeq ($(X86),2)
    BACKEND ?= avx2
  else
    BACKEND ?= ref
  endif
else
  BACKEND ?= ref
endif

# Per-backend code generation. Each pins its ISA, so a build is the same
# instructions on every machine that can run it, and tunes for the core it
# is built on where gcc can be told which (it resolves -mtune=native to
# nothing on an A53, so the core is read from /proc/cpuinfo).
CPU_PART := $(shell grep -m1 'CPU part' /proc/cpuinfo 2>/dev/null | awk '{print $$NF}')
TUNE_0xd03 := cortex-a53
TUNE_0xd08 := cortex-a72
TUNE_0xd0b := cortex-a76
TUNE := $(TUNE_$(CPU_PART))
ifeq ($(BACKEND),neon)
  BFLAGS := -march=armv8-a $(if $(TUNE),-mtune=$(TUNE))
else ifeq ($(BACKEND),neon-dp)
  ifeq ($(UNAME_S),Darwin)
    BFLAGS := -mcpu=apple-m1
  else
    BFLAGS := -march=armv8.2-a+dotprod $(if $(TUNE),-mtune=$(TUNE))
  endif
else ifeq ($(BACKEND),avx2)
  BFLAGS := -mavx2 -mfma
else ifeq ($(BACKEND),avx2-vnni)
  BFLAGS := -mavx2 -mfma -mavxvnni
else
  BFLAGS :=
endif

# -ffp-contract=off is not a style choice. Every backend computes ref's bits
# (backend/kernels.h), and that holds only if no compiler fuses a multiply
# and an add that the code wrote apart: a contraction gcc makes on one target
# and not another would round differently there. Fusion is spelt out, with
# fmaf, where it is meant.
ifeq ($(UNAME_S),Darwin)
  PLATFLAGS := -D_DARWIN_C_SOURCE
else
  PLATFLAGS := -D_DEFAULT_SOURCE -D_POSIX_C_SOURCE=200809L
endif
OPT     ?= -O3
CFLAGS  ?= $(OPT) $(BFLAGS) -std=c11 -ffp-contract=off -Wall -Wextra -Wpedantic
LDLIBS  ?= -lm -lpthread

# make PROFILE=1: `pkito bench` also prints what each stage costs. Timing a
# stage at a time costs a little; quote bench without it.
ifeq ($(PROFILE),1)
  CFLAGS += -DPK_PROFILE
endif

INC := -Icore -Ibackend
LIB_SRC := core/model.c core/os.c core/pool.c core/mel.c core/encoder.c core/decoder.c \
           core/tokens.c core/asr.c core/vad.c core/segment.c core/wav.c core/resample.c
BACK_SRC := $(wildcard backend/$(BACKEND)/*.c)
LIB_OBJ := $(LIB_SRC:.c=.o) $(BACK_SRC:.c=.o)

.DEFAULT_GOAL := pkito

pkito: core/cli.o $(LIB_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

libpkito.a: $(LIB_OBJ)
	$(AR) rcs $@ $^

%.o: %.c
	$(CC) $(CFLAGS) $(PLATFLAGS) $(INC) -MMD -MP -c $< -o $@

-include $(LIB_OBJ:.o=.d) core/cli.d

# ---------------------------------------------------------------- models ----

# The checkpoints, from HuggingFace, and Silero's VAD from its repository at
# v6.2.3; the converter needs Python with numpy and nothing else.
PYTHON ?= $(firstword $(wildcard .venv/bin/python) $(shell command -v python3))
SRC    ?= models/src
VAD    ?= $(SRC)/silero-vad/src/silero_vad/data/silero_vad.onnx
models: models/parakeet-tdt-0.6b-v2.pkf models/parakeet-tdt-0.6b-v3.pkf
models/%.pkf: $(SRC)/%/%.nemo
	$(PYTHON) tools/pkito-tools convert $< $(VAD) $@

# ---------------------------------------------------------------- tests ----

MODEL ?= models/parakeet-tdt-0.6b-v2.pkf
TESTS := tests/kcheck tests/mathcheck

# tests/kref.o is backend/ref under other names, built with this backend's
# flags, so that kcheck holds the backend to ref bit for bit in one binary.
tests/kcheck: tests/kcheck.c tests/kref.o $(LIB_OBJ)
	$(CC) $(CFLAGS) $(PLATFLAGS) $(INC) -o $@ tests/kcheck.c tests/kref.o $(LIB_OBJ) $(LDLIBS)
tests/mathcheck: tests/mathcheck.c core/vmath.h
	$(CC) $(CFLAGS) $(INC) -o $@ tests/mathcheck.c $(LDLIBS)

# check-c needs a compiler and a model and nothing else, so it runs on any
# board: the math library, every kernel against ref, and Silero's test.wav
# through ./pkito. Its VAD probabilities must be tests/data/test.probs.txt
# exactly - vadito's bits, which vadito holds to onnxruntime - and its
# transcripts, every token and its time, the bytes in test.v2.json and
# test.v3.json, which ref wrote: every backend on every machine computes the
# same ones (backend/kernels.h), and this is where a board that does not
# would say so. A model file that is not there is skipped, with a line.
check-c: pkito $(TESTS)
	./tests/mathcheck
	./tests/kcheck $(MODEL)
	@./pkito probs -m $(MODEL) tests/data/test.wav | cmp -s - tests/data/test.probs.txt \
	  && echo "test.wav: VAD probabilities are vadito's, bit for bit" \
	  || { echo "test.wav: FAIL - VAD probabilities differ from tests/data/test.probs.txt"; exit 1; }
	@for v in v2 v3; do f=models/parakeet-tdt-0.6b-$$v.pkf; \
	  if [ ! -f $$f ]; then echo "test.wav: no $$f - skipped"; continue; fi; \
	  ./pkito --json -m $$f tests/data/test.wav | cmp -s - tests/data/test.$$v.json \
	    && echo "test.wav: $$v transcript is ref's, every token and time" \
	    || { echo "test.wav: FAIL - $$v transcript differs from tests/data/test.$$v.json"; exit 1; }; done

check-math: tests/mathcheck
	./tests/mathcheck --all

# The loop a board gets: the tree to HOST:$(RDIR) (the models too, rsync
# sends only what changed), then the build, the compiler half of the checks
# and a bench, there. BENCH names the recording; Silero's test.wav, from a
# vadito checkout beside the tree, is the default.
HOST  ?=
RDIR  ?= pkito
BENCH ?= ../vadito/tests/data/test.wav
remote:
	@[ -n "$(HOST)" ] || { echo "make remote HOST=user@board"; exit 1; }
	rsync -a --delete --exclude .venv --exclude .venv-nemo --exclude .git --exclude '*.o' \
	  --exclude '*.d' --exclude /pkito --exclude 'models/src' --exclude 'models/corpus' \
	  --exclude '/tests/*check' ./ $(HOST):$(RDIR)/
	ssh $(HOST) 'cd $(RDIR) && make -s clean && make -s -j4 $(if $(BACKEND),BACKEND=$(BACKEND)) check-c MODEL=$(MODEL) && ./pkito bench -m $(MODEL) $(BENCH)'

.PHONY: models check-c check-math remote clean

clean:
	rm -f pkito libpkito.a core/*.o core/*.d backend/*/*.o backend/*/*.d tests/*.o tests/*.d $(TESTS)
