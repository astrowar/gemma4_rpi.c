# Refactoring Plan: gemma4.c → Multi-file Project

## Goal

Split the single 701-line `gemma4.c` into cohesive source files grouped by
functionality, while keeping the exact same behavior, ABI, and build output.

## File Layout

```
gemma4.c/
├── gemma4.h        # Shared types, constants, and cross-module declarations
├── tokenizer.c     # Tokenizer types + BPE encode/decode
├── model.c         # Model loading, memory mapping, tensor offset resolution
├── kernels.c       # SIMD kernels: matmul, quantize, attention primitives, GELU
├── transformer.c   # Forward pass: embedding, layernorms, attention, MLP, logits
├── generate.c      # Sampling, prefill, generation loop, benchmark
├── main.c          # CLI argument parsing, model loading, entry point
├── win.c           # (unchanged) Windows mmap compatibility layer
├── win.h           # (unchanged) Windows compatibility header
└── Makefile        # (updated) builds all .c files into the `run` binary
```

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

### `kernels.c` — SIMD / numeric kernels

| Function | Description |
|---|---|
| `matmul_block()` (AVX512VNNI) | 16-row int8×int8 dot-product block, 8 rows per VNNI iteration |
| `matmul_block()` (AVX2) | 16-row int8×int8 dot-product block, 4 rows per AVX2 iteration |
| `matmul_int8()` | OpenMP-parallel driver over `matmul_block` |
| `quantize()` | Dynamic int8 quantization of float activations in groups of 64 |
| `attention_scores()` | SIMD dot-product of query against all cached keys |
| `weighted_value_sum()` | SIMD weighted sum of value vectors by attention probabilities |
| `geglu()` | Table-lookup GELU × up-projection (gated linear unit) |

### `transformer.c` — Model forward pass

| Function | Description |
|---|---|
| `embedding()` | Packed int8 embedding lookup with dequantization |
| `rmsnorm()` | RMS normalization (optional learned weight) |
| `add_and_scale()` | Residual add with learned scale |
| `apply_rope()` | Rotary position embedding on query/key vectors |
| `softmax()` | Online (single-pass) softmax |
| `attention()` | Full attention: QKV projections, KV cache update, causal attention, output projection |
| `forward()` | One full forward pass through all 35 layers (OpenMP team) |
| `logits()` | Final norm + embedding-matrix matmul + tanh soft-cap → vocab logits |

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

```make
CC = cc
CFLAGS = -std=c11 -O3 -Wall -Wextra -march=native -fopenmp
LDFLAGS = -lm

SRCS = tokenizer.c model.c kernels.c transformer.c generate.c main.c

run: $(SRCS) gemma4.h
	$(CC) $(CFLAGS) $(SRCS) -o run $(LDFLAGS)
```

On Windows the `win64` target adds `win.c` to `SRCS`.

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
5. **No behavior changes.** The refactoring is purely structural: same
   algorithms, same SIMD intrinsics, same OpenMP pragmas, same ABI.
6. **`win.c` / `win.h` are untouched.** They already form a separate
   compatibility layer.

## Verification

After refactoring:

```bash
make clean && make
./run -t 0 -n 256 "Why is the sky blue?"     # must match previous output
./run -m ./gemma4-E2B-int8.bin --bench 512 128  # must match previous throughput
```
