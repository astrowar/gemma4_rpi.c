# AVX2 int4 Kernel — Design & Bug Notes

## File: `kernels_avx_int4.c`

SIMD-accelerated `matmul_int4` for x86 (AVX2). Linked alongside
`kernels_avx_int8.c` for AVX2/native builds (see Makefile). The scalar
fallback lives in `kernels_pure_int4.c`; the NEON equivalent in
`kernels_neon_int4.c`.

## Weight layout

```
data[block*16*(width/2) + group*16*16 + row*16 + byte]
scales[(block*groups + group)*16 + row]
```

- 2 nibbles per byte: low = even input, high = odd input
- Zero point 8: signed value = nibble − 8  →  range [−8, +7]
- One fp16 scale per (group, row); group size 32

## Pipeline (per 32-byte weight pair)

```
packed 32 bytes (two 16-byte rows)
  ├─ & 0x0f            → 32 low nibbles
  └─ & 0xf0, srli16    → 32 high nibbles
       │
       └─ − 8 (zero point)
            │
            ├─ even = [row0: w0,w2,…,w30 | row1: w0,w2,…,w30]   (256-bit)
            └─ odd  = [row0: w1,w3,…,w31 | row1: w1,w3,…,w31]   (256-bit)
```

The 32 sequential int8 activations are deinterleaved into 128-bit even/odd
vectors, then broadcast into both 128-bit halves of the YMM. One
`dot_32_2rows()` call produces two 32-term dot products simultaneously
(one per 128-bit half), halving the loop iterations from 16 to 8 per group.

## Deinterleave (the bug we hit)

`PSHUFB` mask semantics: each mask byte `m` selects source byte `m & 0x0f`.
If the high bit of `m` is set, the result is **zero**.

```c
// WRONG — 0 selects byte 0, does NOT zero the lane:
_mm_setr_epi8(0,2,4,6,8,10,12,14, 0,0,0,0,0,0,0,0)

// CORRECT — 0x80 zeroes the lane:
_mm_setr_epi8(0,2,4,6,8,10,12,14,
              (char)0x80,(char)0x80,(char)0x80,(char)0x80,
              (char)0x80,(char)0x80,(char)0x80,(char)0x80)
```

After the shuffle, the 8 valid bytes sit in positions 0–7 and positions 8–15
are zero. `_mm_unpacklo_epi64(lo_e, hi_e)` then packs the two 8-byte halves
into one 128-bit register with all 16 values dense.

The original code used `_mm256_permute4x64_epi64` + `_mm256_castsi256_si128`
to combine the halves. That works but is heavier (YMM temp + VPERMQ) than
the single `UNPCKLPD`-equivalent `_mm_unpacklo_epi64`.

## Dot product (2 rows per YMM)

```c
// Precomputed once per group (outside the row loop):
AIE = broadcast(abs(ie))   // 256-bit |x_even|
AIO = broadcast(abs(io))   // 256-bit |x_odd|

// Per 2-row iteration:
swe = sign(we, ie)         // 256-bit PSIGNB
swo = sign(wo, io)         // 256-bit PSIGNB
pe  = maddubs(AIE, swe)    // 256-bit PMADDUBSW → 16 × int16
po  = maddubs(AIO, swo)    // 256-bit PMADDUBSW → 16 × int16
se  = madd_epi16(pe, 1)    // 256-bit PMADDWD → 8 × int32
so  = madd_epi16(po, 1)    // 256-bit PMADDWD → 8 × int32
sum = se + so             // 256-bit PADDD (even + odd before reduction)
sum = hadd_epi32(sum) × 2 // 256-bit PHADDD (independent per 128-bit half)
dot0 = extract low 128    // row r
dot1 = extract high 128   // row r+1
```

Key optimizations over the original 1-row kernel:
- `abs(input)` computed **once** per group, not 16× (was inside the row loop)
- Even + odd summed **before** the horizontal reduction (2× PHADDD instead of 4×)
- Two output rows per 256-bit load (8 iterations instead of 16 per group)

Why `abs(x) * sign(w,x)` instead of `abs(w) * sign(x,w)`:
`PSIGNB(x, w)` with `x = −128` cannot produce `+128` (not representable in
int8), so the product would be wrong. Swapping the operands puts the
unrepresentable value on the unsigned side where `0x80` is read as 128.

## GEMV vs GEMM

- **rows == 1** (decode): `matmul_int4_block_1x16` — one input row, 16 outputs.
- **rows >= 2** (prefill): `matmul_int4_block_2x16` — pairs of input rows
  share weight loads; an odd trailing row is handled with `has1 == 0`.

Both use `#pragma omp for` over output blocks (16 rows each).

## Performance

### Intel Xeon E5-2690 v4 (Haswell, 14C/28T, AVX2)

| Model | Prefill (tok/s) | Decode (tok/s) |
|-------|:-:|:-:|
| int8  | 14.78 | 3.46 |
| int4 (1-row, before) | 3.6 | 2.0 |
| **int4 (2-row YMM)** | **5.46** | **3.23** |

Speedup from the 2-row YMM microkernel: **+52% prefill**, **+61% decode**.
The int4 kernel now reaches 73% of int8 prefill throughput and 93% of int8
decode throughput.

### AMD Ryzen 7 7700 (Zen 4, 8C/16T, AVX2 + AVX-512)

| Model | Prefill (tok/s) | Decode (tok/s) |
|-------|:-:|:-:|
| int8  | 632.84 ± 6.07 | 25.01 ± 0.07 |
| int4  | TBD | TBD |

(See `README.md` for the full benchmark methodology: 12 timed runs after
one discarded warmup, fastest thread count.)

### Remaining optimisation opportunities

Tracked in `optimization_int4.md`:
- Persistent FP32 scales (avoid per-call fp16→fp32 conversion)
- 4-output SIMD float rescale (`CVTDQ2PS` + `MULPS` + `ADDPS`)
- 2×2 input×output kernel for prefill

## Test

```bash
make test-int4-avx   # builds test_int4.c + kernels_avx_int4.c, compares vs float reference
```

All 5 shapes (hidden, MLP up/down, KV) must report `max_abs_diff < 0.01`.
