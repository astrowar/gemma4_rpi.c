# Refactoring Plan: gemma4.c → Multi-file Project

## Goal

Split the single 701-line `gemma4.c` into cohesive source files grouped by
functionality, while keeping the exact same behavior, ABI, and build output.

## File Layout

```
gemma4.c/
├── gemma4.h             # Shared types, constants, and cross-module declarations
├── tokenizer.c          # Tokenizer types + BPE encode/decode
├── model.c              # Model loading, memory mapping, tensor offset resolution
├── kernels.c            # AVX2/AVX-512 int8 kernels: matmul, quantize, attention, GELU
├── kernels_avx_int4.c   # AVX2 int4 matmul (linked with kernels.c on x86)
├── kernels_neon.c       # ARM NEON int8 + int4 kernels (aarch64)
├── kernels_pure.c       # Portable scalar fallback for all kernels
├── transformer.c        # Forward pass: embedding, layernorms, attention, MLP, logits
├── generate.c           # Sampling, prefill, generation loop, benchmark
├── main.c               # CLI argument parsing, model loading, entry point
├── win.c                # (unchanged) Windows mmap compatibility layer
├── win.h                # (unchanged) Windows compatibility header
└── Makefile             # (updated) builds all .c files into the `run` binary
```

The kernel files are mutually exclusive at build time: the `Makefile` selects
exactly one of `kernels.c` (+ `kernels_avx_int4.c`), `kernels_neon.c`, or
`kernels_pure.c` based on the target architecture (or an explicit
`KERNELS=` override). Each provides the same public kernel interface, so the
rest of the program is kernel-agnostic.

## Function / Type Mapping

### `gemma4.h` — Shared header

Everything that more than one module needs to see:

| Item | Kind | Used by |
|---|---|---|
| `NUM_LAYERS`, `HIDDEN_SIZE`, `VOCAB_SIZE`, `MAX_CONTEXT`, `SLIDING_WINDOW`, `BATCH_SIZE` | macros | all |
| `Tensor` | struct | kernels, transformer, model |
| `LayerWeights` | struct | transformer |
| `ModelWeights` | struct | transformer, model |
| `InferenceState` | struct | transformer, generate, main |
| `Model` | struct | model, main, generate |
| `Tokenizer` | struct | tokenizer, generate, main |
| `VocabEntry`, `LookupEntry` | structs | tokenizer |
| `_Static_assert` ABI check | declaration | (stays in header so every TU verifies) |
| `thread_count()` | inline fn | transformer, generate |

### `tokenizer.c` — Tokenizer

| Function | Description |
|---|---|
| `compare_lookup_keys()` | bsearch comparator for 8-byte lookup keys |
| `apply_bpe_merges()` | Iteratively applies highest-priority BPE merges |
| `tokenize()` | UTF-8 → vocabulary pieces → BPE → token IDs (with `<bos>`) |
| `token_text()` | Token ID → decoded string |

### `model.c` — Model loading

| Function | Description |
|---|---|
| `model_load()` | Opens the `.bin` file, `mmap`s it, validates magic, resolves tensor file-offsets to pointers |
| `model_unload()` | `munmap`s the model |

(These two are extracted from `main()` to keep the entry point thin.)

### `kernels.c` — AVX2 / AVX-512 int8 kernels

| Function | Description |
|---|---|
| `matmul_block()` (AVX512VNNI) | 16-row int8×int8 dot-product block, 8 rows per VNNI iteration |
| `matmul_block()` (AVX2) | 16-row int8×int8 dot-product block, 4 rows per AVX2 iteration |
| `matmul_int8()` | OpenMP-parallel driver over `matmul_block` |
| `quantize()` | Dynamic int8 quantization of float activations in groups of 64 |
| `quantize_int4()` | Same, but groups of 32 to match int4 weight scale granularity |
| `attention_scores()` | SIMD dot-product of query against all cached keys |
| `weighted_value_sum()` | SIMD weighted sum of value vectors by attention probabilities |
| `geglu()` | Table-lookup GELU × up-projection (gated linear unit) |

### `kernels_avx_int4.c` — AVX2 int4 matmul

Linked alongside `kernels.c` on x86. Provides `matmul_int4` only; the rest of
the kernel interface comes from `kernels.c`.

| Function | Description |
|---|---|
| `unpack_row_16()` | Split one 16-byte int4 row into even/odd signed int8 vectors (nibble − 8) |
| `deinterleave_32()` | Split 32 sequential int8 activations into even/odd 16-element vectors |
| `dot_16()` | 16×16 int8 dot product via `abs`/`sign`/`maddubs`/`madd_epi16` |
| `matmul_int4_block_1x16()` | GEMV: 1 input row × 16 output rows (decode) |
| `matmul_int4_block_2x16()` | GEMM: 2 input rows × 16 output rows (prefill), shared weight loads |
| `matmul_int4()` | OpenMP-parallel driver over the block kernels |

### `kernels_neon.c` — ARM NEON kernels (aarch64)

Base AdvSIMD only (no dotprod/fp16), so it runs on Cortex-A72. Provides the
full kernel interface for both int8 and int4. See `NEON_OPTIMIZATIONS.md`.

| Function | Description |
|---|---|
| `matmul_int8()` | int8 GEMV (1×16) / GEMM (2×16) via `vmull_s8` + `vpadalq_s16` |
| `matmul_int4()` | int4 GEMV / GEMM; widens nibbles to int8 then reuses the int8 MAC path |
| `quantize()` / `quantize_int4()` | Dynamic int8 quantization (groups of 64 / 32) |
| `attention_scores()` / `weighted_value_sum()` / `geglu()` | NEON attention and GELU |

### `kernels_pure.c` — portable scalar fallback

No SIMD intrinsics; the reference implementation selected when AVX2/NEON are
unavailable. Provides the same interface as the other kernel files, including
`matmul_int8`, `matmul_int4`, `quantize`, `quantize_int4`, and the attention
primitives.

### `transformer.c` — Model forward pass

| Function | Description |
|---|---|
| `embedding()` / `embedding_int4()` | Packed embedding lookup with dequantization (int8 group 64 / int4 group 32) |
| `rmsnorm()` | RMS normalization (optional learned weight) |
| `add_and_scale()` | Residual add with learned scale |
| `apply_rope()` | Rotary position embedding on query/key vectors |
| `softmax()` | Online (single-pass) softmax |
| `attention()` | Full attention: QKV projections, KV cache update, causal attention, output projection |
| `forward()` | One full forward pass through all 35 layers (OpenMP team) |
| `logits()` | Final norm + embedding-matrix matmul + tanh soft-cap → vocab logits |

`forward()` dispatches each linear layer and quantization step to the kernel
matching the model's weight format via `matmul_dispatch()` and
`quantize_dispatch()` (int8 vs int4), so the layer code is format-agnostic.

### `generate.c` — Sampling & generation loop

| Function | Description |
|---|---|
| `rng_state` | Global xorshift64* state |
| `random_uniform()` | xorshift64* → [0, 1) float |
| `sample()` | Greedy or top-64 / 95%-mass temperature sampling |
| `prefill()` | Chunked forward pass over the prompt (optional logit dump) |
| `generate()` | Tokenize prompt → prefill → autoregressive decode loop |
| `time_seconds()` | Monotonic clock (POSIX `clock_gettime` / Win `QueryPerformanceCounter`) |
| `benchmark()` | Prefill + decode throughput measurement |

### `main.c` — Entry point

| Function | Description |
|---|---|
| `main()` | CLI parsing, model load, dispatch to `generate()` or `benchmark()`, cleanup |

## Makefile Changes

The `Makefile` auto-detects the target and selects the kernel sources:

```make
CC = cc
CFLAGS_BASE = -std=c11 -O3 -Wall -Wextra -fopenmp
LDFLAGS = -lm

# x86 with AVX2:  KERNEL_SRC = kernels.c kernels_avx_int4.c   (-march=native)
# aarch64:        KERNEL_SRC = kernels_neon.c                 (-mcpu=cortex-a72)
# otherwise:      KERNEL_SRC = kernels_pure.c
# override with:  KERNELS=pure | avx2 | neon

SRCS = tokenizer.c model.c $(KERNEL_SRC) transformer.c generate.c main.c

run: $(SRCS) gemma4.h
	$(CC) $(CFLAGS) $(SRCS) -o run $(LDFLAGS)
```

On Windows the `win64` target adds `win.c` to `SRCS`. Test targets
(`test-int4-pure`, `test-int4-neon`, `test-int4-avx`) build `test_int4.c`
against a single kernel file and compare it to a float reference.

## Key Design Decisions

1. **`gemma4.h` is the single source of truth for types.** No `.c` file
   redefines a struct; they all `#include "gemma4.h"`.
2. **`thread_count()` stays `static inline` in the header** because it is
   called from both `transformer.c` and `generate.c` and must not be
   duplicated across TUs.
3. **`rng_state` lives in `generate.c`** as a file-scope global. `main.c`
   sets it via a small `seed_rng()` helper to avoid exposing the global.
4. **`model_load` / `model_unload`** wrap the mmap logic currently inlined
   in `main()`, keeping the entry point to ~40 lines.
5. **The original split was purely structural** — same algorithms, same SIMD
   intrinsics, same OpenMP pragmas, same ABI. The int4 weight format was added
   later as a new feature (new exporter mode, new kernels, new dispatch), not
   part of the original refactor.
6. **`win.c` / `win.h` are untouched.** They already form a separate
   compatibility layer.

## Verification

After refactoring:

```bash
make clean && make
./run -t 0 -n 256 "Why is the sky blue?"     # must match previous output
./run -m ./gemma4-E2B-int8.bin --bench 512 128  # must match previous throughput
```

Kernel correctness is checked separately against a float reference:

```bash
make test   # runs test-int4-pure, and test-int4-avx / test-int4-neon as applicable
```
