CC = cc
CFLAGS_BASE = -std=c11 -O3 -Wall -Wextra -fopenmp
LDFLAGS = -lm
WINCC = x86_64-w64-mingw32-gcc

# Architecture detection.
HAS_AVX2 := $(shell echo | $(CC) -mavx2 -m64 -E -x c - 2>/dev/null | wc -l)
IS_AARCH64 := $(shell uname -m 2>/dev/null | grep -c aarch64)

ifeq ($(KERNELS),pure)
  KERNEL_SRC = kernels_pure.c
  CFLAGS = $(CFLAGS_BASE)
  ARCH = pure
else ifeq ($(KERNELS),avx2)
  KERNEL_SRC = kernels.c
  CFLAGS = $(CFLAGS_BASE) -mavx2 -mfma
  ARCH = avx2
else ifeq ($(KERNELS),neon)
  KERNEL_SRC = kernels_neon.c
  CFLAGS = $(CFLAGS_BASE) -mcpu=cortex-a72
  ARCH = neon
else ifneq ($(HAS_AVX2),0)
  KERNEL_SRC = kernels.c
  CFLAGS = $(CFLAGS_BASE) -march=native
  ARCH = native
else ifneq ($(IS_AARCH64),0)
  KERNEL_SRC = kernels_neon.c
  CFLAGS = $(CFLAGS_BASE) -mcpu=cortex-a72
  ARCH = neon
else
  KERNEL_SRC = kernels_pure.c
  CFLAGS = $(CFLAGS_BASE)
  ARCH = pure
endif

SRCS = tokenizer.c model.c $(KERNEL_SRC) transformer.c generate.c main.c

.PHONY: all run win64 test test-neon profile clean info

all: run

run: $(SRCS) gemma4.h
	@echo "Building [$(ARCH)] kernels: $(KERNEL_SRC)"
	$(CC) $(CFLAGS) $(SRCS) -o run $(LDFLAGS)

win64: $(SRCS) win.c win.h gemma4.h
	$(WINCC) $(CFLAGS) -static -D_WIN32 $(SRCS) win.c -o run.exe $(LDFLAGS) -lshell32

# Correctness test: NEON matmul vs scalar reference (aarch64 only).
test-neon:
	$(CC) $(CFLAGS_BASE) -mcpu=cortex-a72 -O2 test_neon_matmul.c kernels_neon.c -o test_neon_matmul $(LDFLAGS)
	./test_neon_matmul

# Correctness test: int4 matmul vs float reference.
# test-int4-pure builds the scalar reference; test-int4-neon builds the NEON kernel.
test-int4-pure:
	$(CC) $(CFLAGS_BASE) -O2 test_int4.c kernels_pure.c -o test_int4_pure $(LDFLAGS)
	./test_int4_pure

test-int4-neon:
	$(CC) $(CFLAGS_BASE) -mcpu=cortex-a72 -O2 test_int4.c kernels_neon.c -o test_int4_neon $(LDFLAGS)
	./test_int4_neon

# Run all available tests.
test:
	@echo "Running tests..."
	$(MAKE) test-int4-pure
	@if [ "$(ARCH)" = "neon" ]; then \
		$(MAKE) test-neon; \
		$(MAKE) test-int4-neon; \
	fi

info:
	@echo "CC       = $(CC)"
	@echo "CFLAGS   = $(CFLAGS)"
	@echo "KERNELS  = $(KERNEL_SRC)"
	@echo "ARCH     = $(ARCH)"

# Profile build: per-component timing in forward() and logits().
profile:
	$(CC) $(CFLAGS) tokenizer.c model.c $(KERNEL_SRC) transformer.c generate.c profile.c profile_main.c -o run_profile $(LDFLAGS)
	./run_profile $(ARGS)

clean:
	rm -f run run.exe run_profile test_neon_matmul test_kernels_pure
