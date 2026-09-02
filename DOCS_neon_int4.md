# NEON int4 Kernel — Design & Benchmarks

## File: `kernels_neon_int4.c`

ARM NEON (AdvSIMD) implementation of `matmul_int4` for aarch64.
Compiled with `-mcpu=cortex-a72` (base NEON, no SDOT/FP16 vector).
Linked alongside `kernels_neon_int8.c` for NEON builds (see Makefile).

## Weight layout

```
data[block*16*(width/2) + group*16*16 + row*16 + byte]
scales[(block*groups + group)*16 + row]
```

- 2 nibbles per byte: low = even input, high = odd input
- Zero point 8: signed value = nibble − 8 → range [−8, +7]
- One fp16 scale per (group, row); group size 32
- A 16-byte NEON load covers 1 row × 32 inputs (vs 4 rows × 4 inputs for int8)

## Kernel dispatch

| Rows | Kernel | Use case |
|------|--------|----------|
| 1    | `matmul_int4_block_1x16` | Decode (GEMV) |
| 2–3  | `matmul_int4_block_2x16` | Small prefill batches |
| ≥ 4  | `matmul_int4_block_4x16` | Prefill (amortizes weight unpack) |

All kernels use `#pragma omp for schedule(static)` over 16-row output blocks.

## Key optimizations

### Scale cache hoisting

All three kernels pre-convert the 16×groups fp16 weight scales to fp32
**once per output_block** (stored in a `float32x4_t scale_cache[groups][4]`
array) instead of converting them on every group iteration. This eliminates
redundant `fp16_to_f32` calls from the hot loop.

### int4 unpack (`int4_row_to_s8x2`)

```
16-byte load → 32 packed nibbles
  ├─ AND 0x0f        → 16 low nibbles  (even inputs)
  └─ SHR 4           → 16 high nibbles (odd inputs)
       │
       ├─ VZIP1 → int8x16 (even, zero-point subtracted)
       └─ VZIP2 → int8x16 (odd,  zero-point subtracted)
```

6 instructions total (1 load, 2 bitwise, 2 zip, 2 subtract). The zero-point
constant (`vdupq_n_s8(8)`) is hoisted by the compiler.

### Dot product

Uses `vmull_s8` + `vpaddlq_s16` + `vaddvq_s32` (base AdvSIMD, no SDOT).
Each 16-element int8 dot is ~6 instructions; a full 32-value dot is ~12.

### 4×16 kernel (prefill)

- Weight unpacking amortized across 4 input rows (1 unpack → 4 dot products)
- Processes 4 output rows at a time: 4 scalar dots packed into one `int32x4_t`
- Single `vfmaq_f32` per (input_row, quartet) for the rescale+accumulate
- Tail rows (1–3 remaining) fall back to the 2×16 kernel

### Prefetch

`__builtin_prefetch` one group ahead (256 bytes) in all kernels to hide
L1 miss latency on the weight stream.

## Performance

### Raspberry Pi 4 (Cortex-A72, 4 cores, NEON)

Measured with `OMP_NUM_THREADS=4`, model `gemma4-E2B-int4.bin` (2707.8 MB):

| Metric | Before optimization | After optimization | Change |
|--------|:-:|:-:|:-:|
| pp32 (benchmark) | 5.50 tok/s | **7.77 tok/s** | **+41%** |
| tg32 (benchmark) | 2.58 tok/s | **2.75 tok/s** | +6.6% |
| Prefill (16 tok, real prompt) | 2.6 tok/s | **7.5 tok/s** | **+188%** |
| Decode (10 tok, real prompt) | 1.03 tok/s | **2.73 tok/s** | **+165%** |

Comparison with int8 on the same hardware:

| Model | pp32 (tok/s) | tg32 (tok/s) |
|-------|:-:|:-:|
| int8  | 11.78 | 1.54 |
| int4 (optimized) | 7.77 | 2.75 |

The int4 kernel now reaches 66% of int8 prefill throughput and **178% of int8
decode throughput** (int4 decode benefits from half the memory bandwidth).

### Intel Xeon E5-2690 v4 (Haswell, 14C/28T, AVX2)

For reference (see `DOCS_avx_int4.md`):

| Model | Prefill (tok/s) | Decode (tok/s) |
|-------|:-:|:-:|
| int8  | 14.78 | 3.46 |
| int4 (2-row YMM) | 5.46 | 3.23 |

## Test

```bash
make test-int4-neon   # builds test_int4.c + kernels_neon_int4.c, compares vs float reference
```

All 5 shapes (hidden, MLP up/down, KV) must report `max_abs_diff < 0.01`.
