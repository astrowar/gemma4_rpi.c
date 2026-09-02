# AVX2 int4 Kernel — Design & Bug Notes

## File: `kernels_avx_int4.c`

SIMD-accelerated `matmul_int4` for x86 (AVX2). Linked alongside `kernels.c`
for AVX2/native builds (see Makefile). The scalar fallback lives in
`kernels_pure.c`; the NEON equivalent in `kernels_neon.c`.

## Weight layout

```
data[block*16*(width/2) + group*16*16 + row*16 + byte]
scales[(block*groups + group)*16 + row]
```

- 2 nibbles per byte: low = even input, high = odd input
- Zero point 8: signed value = nibble − 8  →  range [−8, +7]
- One fp16 scale per (group, row); group size 32

## Pipeline (per 16-byte weight row)

```
packed 16 bytes
  ├─ & 0x0f          → 16 low nibbles  (inputs 0,2,4,…,30)
  └─ & 0xf0, srli16  → 16 high nibbles (inputs 1,3,5,…,31)
       │
       └─ − 8 (zero point)
            │
            ├─ even = [w0, w2, …, w30]   (128-bit)
            └─ odd  = [w1, w3, …, w31]   (128-bit)
```

The 32 sequential int8 activations are deinterleaved the same way so that
`dot_16(ie, we) + dot_16(io, wo)` equals the full 32-term dot product.

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

## Dot product

```c
ax = abs(x)          // PABSB  — |x|, safe for x = −128 (gives 0x80 = 128 unsigned)
sw = sign(w, x)      // PSIGNB — w, −w, or 0 depending on sign of x
p16 = maddubs(ax, sw) // PMADDUBSW — unsigned × signed → int16 pairs
p32 = madd_epi16(p16, 1)
p32 = hadd_epi32(p32, p32)  × 2
result = extract p32
```

Why `abs(x) * sign(w,x)` instead of `abs(w) * sign(x,w)`:
`PSIGNB(x, w)` with `x = −128` cannot produce `+128` (not representable in
int8), so the product would be wrong. Swapping the operands puts the
unrepresentable value on the unsigned side where `0x80` is read as 128.

## GEMV vs GEMM

- **rows == 1** (decode): `matmul_int4_block_1x16` — one input row, 16 outputs.
- **rows >= 2** (prefill): `matmul_int4_block_2x16` — pairs of input rows
  share weight loads; odd trailing row is handled with `has_row1 == 0`.

Both use `#pragma omp for` over output blocks (16 rows each).

## Performance (Gemma 4 E2B, x86_64 AVX2)

| Model | Prefill (tok/s) | Decode (tok/s) |
|-------|----------------|----------------|
| int8  | 14.6           | 3.7            |
| int4  | 3.6            | 2.0            |

The int4 unpack (7 SIMD ops per 16 bytes) adds overhead that the halved
memory bandwidth does not yet offset at this model size. Further optimisation
opportunities: process 4 rows × 8 inputs per 256-bit load (like the NEON
kernel), or use VNNI if AVX-512 is available.

## Test

```bash
make test-int4-avx   # builds test_int4.c + kernels_avx_int4.c, compares vs float reference
```

All 5 shapes (hidden, MLP up/down, KV) must report `max_abs_diff < 0.01`.
