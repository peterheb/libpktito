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

.PHONY: models clean

clean:
	rm -f pkito libpkito.a core/*.o core/*.d backend/*/*.o backend/*/*.d tests/*.o tests/*.d
