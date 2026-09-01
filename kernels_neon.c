// ARM NEON kernel implementations for gemma4.c.
// Compiled only on aarch64 (see Makefile KERNELS=neon).
//
// Uses base NEON (AdvSIMD) only — no dotprod/fp16 — so it runs on Cortex-A72
// and older cores. The int8 GEMM exploits the [block][group][chunk][row][4]
// packed layout directly: each 16-byte NEON load covers 4 output rows × 4
// inputs. Two dedicated kernels:
//   - 1×16 (GEMV) for rows==1 (decode): minimal registers, no redundant work
//   - 2×16 (GEMM) for rows>=2 (prefill): weight loads shared between 2 tokens
// Accumulation uses vmull_s8 + vpadalq_s16 (pairwise widening accumulate)
// with a single vpaddq_s32 at the end of each 64-wide group.

#if !defined(__aarch64__)
#error "kernels_neon.c requires an aarch64 target. Use kernels_pure.c for other architectures."
#endif

#include "gemma4.h"
#include <arm_neon.h>

// Converts a packed half-precision (fp16) value to float via bit manipulation.
// No FP16 hardware or ldexpf needed — works on Cortex-A72.
static inline __attribute__((always_inline))
float fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            // FP16 subnormal.
            int p = 31 - __builtin_clz(mant);
            uint32_t e = (uint32_t)(p + 103);
            uint32_t frac = (mant << (23 - p)) & 0x007fffffu;
            bits = sign | (e << 23) | frac;
        }
    }
    else if (exp == 31) {
        // Inf / NaN
        bits = sign | 0x7f800000u | (mant << 13);
        if (mant) bits |= 0x00400000u;
    }
    else {
        // FP16 normal → FP32 normal
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }

    float result;
    __builtin_memcpy(&result, &bits, sizeof(result));
    return result;
}

// Reads a single packed int8 weight from the block-interleaved layout.
// Layout (matches the exporter and the scalar kernels_pure.c):
//   data[block*16*width + group*16*64 + chunk*16*4 + row*4 + offset]
static inline int8_t packed_weight(const Tensor *weight, int output, int input) {
    const int block_rows = 16;
    int width = weight->shape[1];
    int block = output / block_rows;
    int row = output % block_rows;
    int group = input / 64;
    int chunk = input % 64 / 4;
    int offset = input % 4;
    const int8_t *data = (const int8_t *)weight->data;
    return data[(size_t)block * block_rows * width
                + (size_t)group * block_rows * 64
                + chunk * block_rows * 4 + row * 4 + offset];
}

// Reads the fp16 scale for a given output row and input group.
static inline float weight_scale(const Tensor *weight, int output, int input_group) {
    int groups = weight->shape[1] / 64;
    int block = output / 16;
    int row = output % 16;
    return fp16_to_f32(weight->scales[((size_t)block * groups + input_group) * 16 + row]);
}

// Duplicates 4 bytes into all four 4-byte lanes of an int8x16 register:
//   x0 x1 x2 x3  →  x0 x1 x2 x3  x0 x1 x2 x3  x0 x1 x2 x3  x0 x1 x2 x3
static inline __attribute__((always_inline))
int8x16_t duplicate_4_s8(const int8_t *p) {
    uint32_t x;
    __builtin_memcpy(&x, p, sizeof(x));
    return vreinterpretq_s8_u32(vdupq_n_u32(x));
}

// Widening multiply-accumulate for 4 rows × 4 cols.
//
// w layout (one 16-byte load from packed weights):
//   row0: w00 w01 w02 w03
//   row1: w10 w11 w12 w13
//   row2: w20 w21 w22 w23
//   row3: w30 w31 w32 w33
//
// x layout (duplicated input): x0 x1 x2 x3 (×4)
//
// Accumulates into acc_lo (rows 0-1) and acc_hi (rows 2-3) using
// vpadalq_s16 (pairwise widening add-accumulate, base AdvSIMD).
// After all 16 chunks: vpaddq_s32(acc_lo, acc_hi) gives 4 dot products.
static inline __attribute__((always_inline))
void dot_4rows_4cols_acc(int32x4_t *acc_lo, int32x4_t *acc_hi,
                         int8x16_t w, int8x16_t x) {
    int16x8_t lo = vmull_s8(vget_low_s8(w), vget_low_s8(x));
    int16x8_t hi = vmull_high_s8(w, x);
    *acc_lo = vpadalq_s16(*acc_lo, lo);
    *acc_hi = vpadalq_s16(*acc_hi, hi);
}

// GEMV kernel: 1 input row × 16 output rows (decode, rows==1).
// Minimal register pressure: 8 int32 accumulators + 4 float accumulators.
static inline __attribute__((always_inline)) void matmul_block_1x16(
    float *output, const int8_t *input_q, const float *input_scales,
    const Tensor *weight, size_t output_block) {
    const int block_rows = 16;
    const int width = weight->shape[1];
    const int groups = width / 64;

    const int8_t *input_row = input_q;
    const float *input_scale_row = input_scales;
    const int8_t *packed_weights =
        (const int8_t *)weight->data + output_block * block_rows * width;

    // Convert all weight scales once.
    float32x4_t scale_cache[groups * 4];
    for (int group = 0; group < groups; group++) {
        const uint16_t *ws =
            weight->scales + ((output_block * groups + group) * 16);
        scale_cache[group * 4 + 0] = (float32x4_t) {
            fp16_to_f32(ws[0]),  fp16_to_f32(ws[1]),
            fp16_to_f32(ws[2]),  fp16_to_f32(ws[3]) };
        scale_cache[group * 4 + 1] = (float32x4_t) {
            fp16_to_f32(ws[4]),  fp16_to_f32(ws[5]),
            fp16_to_f32(ws[6]),  fp16_to_f32(ws[7]) };
        scale_cache[group * 4 + 2] = (float32x4_t) {
            fp16_to_f32(ws[8]),  fp16_to_f32(ws[9]),
            fp16_to_f32(ws[10]), fp16_to_f32(ws[11]) };
        scale_cache[group * 4 + 3] = (float32x4_t) {
            fp16_to_f32(ws[12]), fp16_to_f32(ws[13]),
            fp16_to_f32(ws[14]), fp16_to_f32(ws[15]) };
    }

    // Float accumulators: each holds 4 output rows.
    float32x4_t r0 = vdupq_n_f32(0.0f);
    float32x4_t r1 = vdupq_n_f32(0.0f);
    float32x4_t r2 = vdupq_n_f32(0.0f);
    float32x4_t r3 = vdupq_n_f32(0.0f);

    for (int group = 0; group < groups; group++) {
        const int8_t *in = input_row + group * 64;
        const int8_t *wg = packed_weights + (size_t)group * 16 * 64;

        // Prefetch next group's weights into L1 while processing current.
        if (group + 1 < groups)
            __builtin_prefetch(packed_weights + (size_t)(group + 1) * 16 * 64, 0, 1);

        // Int32 partial accumulators (lo = rows 0-1, hi = rows 2-3 per quartet).
        int32x4_t a0l = vdupq_n_s32(0), a0h = vdupq_n_s32(0);
        int32x4_t a1l = vdupq_n_s32(0), a1h = vdupq_n_s32(0);
        int32x4_t a2l = vdupq_n_s32(0), a2h = vdupq_n_s32(0);
        int32x4_t a3l = vdupq_n_s32(0), a3h = vdupq_n_s32(0);

        // 64 inputs = 16 chunks of 4.
        for (int chunk = 0; chunk < 16; chunk++) {
            int8x16_t x = duplicate_4_s8(in + chunk * 4);
            const int8_t *wc = wg + chunk * 64;

            // Prefetch next chunk (64 bytes ahead).
            if (chunk + 1 < 16)
                __builtin_prefetch(wc + 64, 0, 3);

            int8x16_t w;
            w = vld1q_s8(wc + 0);
            dot_4rows_4cols_acc(&a0l, &a0h, w, x);
            w = vld1q_s8(wc + 16);
            dot_4rows_4cols_acc(&a1l, &a1h, w, x);
            w = vld1q_s8(wc + 32);
            dot_4rows_4cols_acc(&a2l, &a2h, w, x);
            w = vld1q_s8(wc + 48);
            dot_4rows_4cols_acc(&a3l, &a3h, w, x);
        }

        // Rescale and accumulate.
        float s = input_scale_row[group];
        r0 = vaddq_f32(r0, vmulq_f32(
            vmulq_n_f32(vcvtq_f32_s32(vpaddq_s32(a0l, a0h)), s),
            scale_cache[group * 4 + 0]));
        r1 = vaddq_f32(r1, vmulq_f32(
            vmulq_n_f32(vcvtq_f32_s32(vpaddq_s32(a1l, a1h)), s),
            scale_cache[group * 4 + 1]));
        r2 = vaddq_f32(r2, vmulq_f32(
            vmulq_n_f32(vcvtq_f32_s32(vpaddq_s32(a2l, a2h)), s),
            scale_cache[group * 4 + 2]));
        r3 = vaddq_f32(r3, vmulq_f32(
            vmulq_n_f32(vcvtq_f32_s32(vpaddq_s32(a3l, a3h)), s),
            scale_cache[group * 4 + 3]));
    }

    float *dst = output + output_block * 16;
    vst1q_f32(dst + 0,  r0);
    vst1q_f32(dst + 4,  r1);
    vst1q_f32(dst + 8,  r2);
    vst1q_f32(dst + 12, r3);
}

// GEMM kernel: 2 input rows × 16 output rows (prefill, rows>=2).
// Weight loads are shared between the two input rows.
static inline __attribute__((always_inline)) void matmul_block_2x16(
    float *output, const int8_t *input_q, const float *input_scales,
    const Tensor *weight, size_t rows, size_t output_block) {
    const int block_rows = 16;
    const int width = weight->shape[1];
    const int groups = width / 64;

    const int8_t *packed_weights =
        (const int8_t *)weight->data + output_block * block_rows * width;

    // Convert all weight scales once per output_block.
    float32x4_t scale_cache[groups * 4];
    for (int group = 0; group < groups; group++) {
        const uint16_t *ws =
            weight->scales + ((output_block * groups + group) * 16);
        scale_cache[group * 4 + 0] = (float32x4_t) {
            fp16_to_f32(ws[0]),  fp16_to_f32(ws[1]),
            fp16_to_f32(ws[2]),  fp16_to_f32(ws[3]) };
        scale_cache[group * 4 + 1] = (float32x4_t) {
            fp16_to_f32(ws[4]),  fp16_to_f32(ws[5]),
            fp16_to_f32(ws[6]),  fp16_to_f32(ws[7]) };
        scale_cache[group * 4 + 2] = (float32x4_t) {
            fp16_to_f32(ws[8]),  fp16_to_f32(ws[9]),
            fp16_to_f32(ws[10]), fp16_to_f32(ws[11]) };
        scale_cache[group * 4 + 3] = (float32x4_t) {
            fp16_to_f32(ws[12]), fp16_to_f32(ws[13]),
            fp16_to_f32(ws[14]), fp16_to_f32(ws[15]) };
    }

    // Process 2 input rows at a time (shared weight loads).
    for (size_t row0 = 0; row0 < rows; row0 += 2) {
        const int8_t *in0 = input_q + row0 * width;
        const float *sc0 = input_scales + row0 * groups;
        const int has_row1 = (row0 + 1 < rows);
        const int8_t *in1 = has_row1 ? input_q + (row0 + 1) * width : in0;
        const float *sc1 = has_row1 ? input_scales + (row0 + 1) * groups : sc0;

        // Float accumulators: each holds 4 output rows.
        float32x4_t r0_0 = vdupq_n_f32(0.0f), r0_1 = vdupq_n_f32(0.0f);
        float32x4_t r0_2 = vdupq_n_f32(0.0f), r0_3 = vdupq_n_f32(0.0f);
        float32x4_t r1_0 = vdupq_n_f32(0.0f), r1_1 = vdupq_n_f32(0.0f);
        float32x4_t r1_2 = vdupq_n_f32(0.0f), r1_3 = vdupq_n_f32(0.0f);

        for (int group = 0; group < groups; group++) {
            const int8_t *g0 = in0 + group * 64;
            const int8_t *g1 = in1 + group * 64;
            const int8_t *wg = packed_weights + (size_t)group * 16 * 64;

            // Prefetch next group's weights into L1 while processing current.
            if (group + 1 < groups)
                __builtin_prefetch(packed_weights + (size_t)(group + 1) * 16 * 64, 0, 1);

            // Int32 partial accumulators (lo = rows 0-1, hi = rows 2-3 per quartet).
            int32x4_t a0_0l = vdupq_n_s32(0), a0_0h = vdupq_n_s32(0);
            int32x4_t a0_1l = vdupq_n_s32(0), a0_1h = vdupq_n_s32(0);
            int32x4_t a0_2l = vdupq_n_s32(0), a0_2h = vdupq_n_s32(0);
            int32x4_t a0_3l = vdupq_n_s32(0), a0_3h = vdupq_n_s32(0);
            int32x4_t a1_0l = vdupq_n_s32(0), a1_0h = vdupq_n_s32(0);
            int32x4_t a1_1l = vdupq_n_s32(0), a1_1h = vdupq_n_s32(0);
            int32x4_t a1_2l = vdupq_n_s32(0), a1_2h = vdupq_n_s32(0);
            int32x4_t a1_3l = vdupq_n_s32(0), a1_3h = vdupq_n_s32(0);

            // 64 inputs = 16 chunks of 4.
            for (int chunk = 0; chunk < 16; chunk++) {
                int8x16_t x0 = duplicate_4_s8(g0 + chunk * 4);
                int8x16_t x1 = duplicate_4_s8(g1 + chunk * 4);
                const int8_t *wc = wg + chunk * 64;

                // Prefetch next chunk (64 bytes ahead).
                if (chunk + 1 < 16)
                    __builtin_prefetch(wc + 64, 0, 3);

                // Single weight register, reused for both rows.
                int8x16_t w;
                w = vld1q_s8(wc + 0);
                dot_4rows_4cols_acc(&a0_0l, &a0_0h, w, x0);
                dot_4rows_4cols_acc(&a1_0l, &a1_0h, w, x1);

                w = vld1q_s8(wc + 16);
                dot_4rows_4cols_acc(&a0_1l, &a0_1h, w, x0);
                dot_4rows_4cols_acc(&a1_1l, &a1_1h, w, x1);

                w = vld1q_s8(wc + 32);
                dot_4rows_4cols_acc(&a0_2l, &a0_2h, w, x0);
                dot_4rows_4cols_acc(&a1_2l, &a1_2h, w, x1);

                w = vld1q_s8(wc + 48);
                dot_4rows_4cols_acc(&a0_3l, &a0_3h, w, x0);
                dot_4rows_4cols_acc(&a1_3l, &a1_3h, w, x1);
            }

            // Finalize: pairwise add lo+hi → 4 dot products per row.
            float s0 = sc0[group];
            float s1 = sc1[group];

            r0_0 = vaddq_f32(r0_0, vmulq_f32(
                vmulq_n_f32(vcvtq_f32_s32(vpaddq_s32(a0_0l, a0_0h)), s0),
                scale_cache[group * 4 + 0]));
            r0_1 = vaddq_f32(r0_1, vmulq_f32(
                vmulq_n_f32(vcvtq_f32_s32(vpaddq_s32(a0_1l, a0_1h)), s0),
                scale_cache[group * 4 + 1]));
            r0_2 = vaddq_f32(r0_2, vmulq_f32(
                vmulq_n_f32(vcvtq_f32_s32(vpaddq_s32(a0_2l, a0_2h)), s0),
                scale_cache[group * 4 + 2]));
            r0_3 = vaddq_f32(r0_3, vmulq_f32(
                vmulq_n_f32(vcvtq_f32_s32(vpaddq_s32(a0_3l, a0_3h)), s0),
                scale_cache[group * 4 + 3]));

            r1_0 = vaddq_f32(r1_0, vmulq_f32(
                vmulq_n_f32(vcvtq_f32_s32(vpaddq_s32(a1_0l, a1_0h)), s1),
                scale_cache[group * 4 + 0]));
            r1_1 = vaddq_f32(r1_1, vmulq_f32(
                vmulq_n_f32(vcvtq_f32_s32(vpaddq_s32(a1_1l, a1_1h)), s1),
                scale_cache[group * 4 + 1]));
            r1_2 = vaddq_f32(r1_2, vmulq_f32(
                vmulq_n_f32(vcvtq_f32_s32(vpaddq_s32(a1_2l, a1_2h)), s1),
                scale_cache[group * 4 + 2]));
            r1_3 = vaddq_f32(r1_3, vmulq_f32(
                vmulq_n_f32(vcvtq_f32_s32(vpaddq_s32(a1_3l, a1_3h)), s1),
                scale_cache[group * 4 + 3]));
        }

        float *dst0 = output + row0 * weight->shape[0] + output_block * 16;
        vst1q_f32(dst0 + 0,  r0_0);
        vst1q_f32(dst0 + 4,  r0_1);
        vst1q_f32(dst0 + 8,  r0_2);
        vst1q_f32(dst0 + 12, r0_3);

        if (has_row1) {
            float *dst1 = output + (row0 + 1) * weight->shape[0] + output_block * 16;
            vst1q_f32(dst1 + 0,  r1_0);
            vst1q_f32(dst1 + 4,  r1_1);
            vst1q_f32(dst1 + 8,  r1_2);
            vst1q_f32(dst1 + 12, r1_3);
        }
    }
}

void matmul_int8(float *output, const int8_t *input_q, const float *input_scales,
                 const Tensor *weight, size_t rows) {
    const size_t block_rows = 16;
    #pragma omp for schedule(static)
    for (size_t output_block = 0; output_block < (size_t)weight->shape[0] / block_rows; output_block++) {
        if (rows == 1)
            matmul_block_1x16(output, input_q, input_scales, weight, output_block);
        else
            matmul_block_2x16(output, input_q, input_scales, weight, rows, output_block);
    }
}

// ----------------------------------------------------------------------------
// int4 weight kernels.
//
// Weights are packed two 4-bit values per byte:
//   data[block*16*width + group*16*32 + row*16 + byte]
// A 16-byte NEON load of one row's group therefore holds 32 signed int4
// weights (low nibble = even input, high nibble = odd input), with 8 as the
// zero point. Each (group, row) carries one fp16 scale, laid out
//   scales[(block*groups + group)*16 + row].
//
// The int4 values are widened to int8 (nibble - 8) and accumulated with the
// same vmull_s8/vpadalq_s16 machinery as the int8 path, so the integer dot
// product is exact; only the rescale step differs (scale every 32 inputs).

// Widens one 16-byte int4 row (32 values) to a signed int8x16 register.
static inline __attribute__((always_inline))
int8x16_t int4_row_to_s8(const uint8_t *row) {
    uint8x16_t packed = vld1q_u8(row);
    uint8x16_t low = vandq_u8(packed, vdupq_n_u8(0x0f));
    uint8x16_t high = vrshlq_u8(packed, vdupq_n_u8(4));
    int8x16_t lo8 = vreinterpretq_s8_u8(vsubq_u8(low, vdupq_n_u8(8)));
    int8x16_t hi8 = vreinterpretq_s8_u8(vsubq_u8(high, vdupq_n_u8(8)));
    return vcombine_s8(vget_low_s8(lo8), vget_low_s8(hi8));
}

// GEMV: 1 input row x 16 output rows (decode).
static inline __attribute__((always_inline)) void matmul_int4_block_1x16(
    float *output, const int8_t *input_q, const float *input_scales,
    const Tensor *weight, size_t output_block) {
    const int block_rows = 16;
    const int width = weight->shape[1];
    const int groups = width / 32;

    const int8_t *input_row = input_q;
    const float *input_scale_row = input_scales;
    // Block stride is 16 rows x (width/2) bytes (int4 packs 2 weights/byte).
    const uint8_t *packed_weights =
        (const uint8_t *)weight->data + output_block * block_rows * (width / 2);

    float32x4_t scale_cache[groups * 4];
    for (int group = 0; group < groups; group++) {
        const uint16_t *ws = weight->scales + ((output_block * groups + group) * 16);
        scale_cache[group * 4 + 0] = (float32x4_t) {
            fp16_to_f32(ws[0]),  fp16_to_f32(ws[1]),
            fp16_to_f32(ws[2]),  fp16_to_f32(ws[3]) };
        scale_cache[group * 4 + 1] = (float32x4_t) {
            fp16_to_f32(ws[4]),  fp16_to_f32(ws[5]),
            fp16_to_f32(ws[6]),  fp16_to_f32(ws[7]) };
        scale_cache[group * 4 + 2] = (float32x4_t) {
            fp16_to_f32(ws[8]),  fp16_to_f32(ws[9]),
            fp16_to_f32(ws[10]), fp16_to_f32(ws[11]) };
        scale_cache[group * 4 + 3] = (float32x4_t) {
            fp16_to_f32(ws[12]), fp16_to_f32(ws[13]),
            fp16_to_f32(ws[14]), fp16_to_f32(ws[15]) };
    }

    float32x4_t r0 = vdupq_n_f32(0.0f);
    float32x4_t r1 = vdupq_n_f32(0.0f);
    float32x4_t r2 = vdupq_n_f32(0.0f);
    float32x4_t r3 = vdupq_n_f32(0.0f);

    for (int group = 0; group < groups; group++) {
        const int8_t *in = input_row + group * 32;
        const uint8_t *wg = packed_weights + (size_t)group * 16 * 16;

        if (group + 1 < groups)
            __builtin_prefetch(packed_weights + (size_t)(group + 1) * 16 * 16, 0, 1);

        int32x4_t a0l = vdupq_n_s32(0), a0h = vdupq_n_s32(0);
        int32x4_t a1l = vdupq_n_s32(0), a1h = vdupq_n_s32(0);
        int32x4_t a2l = vdupq_n_s32(0), a2h = vdupq_n_s32(0);
        int32x4_t a3l = vdupq_n_s32(0), a3h = vdupq_n_s32(0);

        // 32 inputs = 8 chunks of 4.
        for (int chunk = 0; chunk < 8; chunk++) {
            int8x16_t x = duplicate_4_s8(in + chunk * 4);
            const uint8_t *wc = wg + chunk * 64;

            if (chunk + 1 < 8)
                __builtin_prefetch(wc + 64, 0, 3);

            int8x16_t w;
            w = int4_row_to_s8(wc + 0);
            dot_4rows_4cols_acc(&a0l, &a0h, w, x);
            w = int4_row_to_s8(wc + 16);
            dot_4rows_4cols_acc(&a1l, &a1h, w, x);
            w = int4_row_to_s8(wc + 32);
            dot_4rows_4cols_acc(&a2l, &a2h, w, x);
            w = int4_row_to_s8(wc + 48);
            dot_4rows_4cols_acc(&a3l, &a3h, w, x);
        }

        float s = input_scale_row[group];
        r0 = vaddq_f32(r0, vmulq_f32(
            vmulq_n_f32(vcvtq_f32_s32(vpaddq_s32(a0l, a0h)), s),
            scale_cache[group * 4 + 0]));
        r1 = vaddq_f32(r1, vmulq_f32(
            vmulq_n_f32(vcvtq_f32_s32(vpaddq_s32(a1l, a1h)), s),
            scale_cache[group * 4 + 1]));
        r2 = vaddq_f32(r2, vmulq_f32(
            vmulq_n_f32(vcvtq_f32_s32(vpaddq_s32(a2l, a2h)), s),
            scale_cache[group * 4 + 2]));
        r3 = vaddq_f32(r3, vmulq_f32(
            vmulq_n_f32(vcvtq_f32_s32(vpaddq_s32(a3l, a3h)), s),
            scale_cache[group * 4 + 3]));
    }

    float *dst = output + output_block * 16;
    vst1q_f32(dst + 0,  r0);
    vst1q_f32(dst + 4,  r1);
    vst1q_f32(dst + 8,  r2);
    vst1q_f32(dst + 12, r3);
}

// GEMM: 2 input rows x 16 output rows (prefill). Weight loads shared.
static inline __attribute__((always_inline)) void matmul_int4_block_2x16(
    float *output, const int8_t *input_q, const float *input_scales,
    const Tensor *weight, size_t rows, size_t output_block) {
    const int block_rows = 16;
    const int width = weight->shape[1];
    const int groups = width / 32;

    // Block stride is 16 rows x (width/2) bytes (int4 packs 2 weights/byte).
    const uint8_t *packed_weights =
        (const uint8_t *)weight->data + output_block * block_rows * (width / 2);

    float32x4_t scale_cache[groups * 4];
    for (int group = 0; group < groups; group++) {
        const uint16_t *ws = weight->scales + ((output_block * groups + group) * 16);
        scale_cache[group * 4 + 0] = (float32x4_t) {
            fp16_to_f32(ws[0]),  fp16_to_f32(ws[1]),
            fp16_to_f32(ws[2]),  fp16_to_f32(ws[3]) };
        scale_cache[group * 4 + 1] = (float32x4_t) {
            fp16_to_f32(ws[4]),  fp16_to_f32(ws[5]),
            fp16_to_f32(ws[6]),  fp16_to_f32(ws[7]) };
        scale_cache[group * 4 + 2] = (float32x4_t) {
            fp16_to_f32(ws[8]),  fp16_to_f32(ws[9]),
            fp16_to_f32(ws[10]), fp16_to_f32(ws[11]) };
        scale_cache[group * 4 + 3] = (float32x4_t) {
            fp16_to_f32(ws[12]), fp16_to_f32(ws[13]),
            fp16_to_f32(ws[14]), fp16_to_f32(ws[15]) };
    }

    for (size_t row0 = 0; row0 < rows; row0 += 2) {
        const int8_t *in0 = input_q + row0 * width;
        const float *sc0 = input_scales + row0 * groups;
        const int has_row1 = (row0 + 1 < rows);
        const int8_t *in1 = has_row1 ? input_q + (row0 + 1) * width : in0;
        const float *sc1 = has_row1 ? input_scales + (row0 + 1) * groups : sc0;

        float32x4_t r0_0 = vdupq_n_f32(0.0f), r0_1 = vdupq_n_f32(0.0f);
        float32x4_t r0_2 = vdupq_n_f32(0.0f), r0_3 = vdupq_n_f32(0.0f);
        float32x4_t r1_0 = vdupq_n_f32(0.0f), r1_1 = vdupq_n_f32(0.0f);
        float32x4_t r1_2 = vdupq_n_f32(0.0f), r1_3 = vdupq_n_f32(0.0f);

        for (int group = 0; group < groups; group++) {
            const int8_t *g0 = in0 + group * 32;
            const int8_t *g1 = in1 + group * 32;
            const uint8_t *wg = packed_weights + (size_t)group * 16 * 16;

            if (group + 1 < groups)
                __builtin_prefetch(packed_weights + (size_t)(group + 1) * 16 * 16, 0, 1);

            int32x4_t a0_0l = vdupq_n_s32(0), a0_0h = vdupq_n_s32(0);
            int32x4_t a0_1l = vdupq_n_s32(0), a0_1h = vdupq_n_s32(0);
            int32x4_t a0_2l = vdupq_n_s32(0), a0_2h = vdupq_n_s32(0);
            int32x4_t a0_3l = vdupq_n_s32(0), a0_3h = vdupq_n_s32(0);
            int32x4_t a1_0l = vdupq_n_s32(0), a1_0h = vdupq_n_s32(0);
            int32x4_t a1_1l = vdupq_n_s32(0), a1_1h = vdupq_n_s32(0);
            int32x4_t a1_2l = vdupq_n_s32(0), a1_2h = vdupq_n_s32(0);
            int32x4_t a1_3l = vdupq_n_s32(0), a1_3h = vdupq_n_s32(0);

            for (int chunk = 0; chunk < 8; chunk++) {
                int8x16_t x0 = duplicate_4_s8(g0 + chunk * 4);
                int8x16_t x1 = duplicate_4_s8(g1 + chunk * 4);
                const uint8_t *wc = wg + chunk * 64;

                if (chunk + 1 < 8)
                    __builtin_prefetch(wc + 64, 0, 3);

                int8x16_t w;
                w = int4_row_to_s8(wc + 0);
                dot_4rows_4cols_acc(&a0_0l, &a0_0h, w, x0);
                dot_4rows_4cols_acc(&a1_0l, &a1_0h, w, x1);

                w = int4_row_to_s8(wc + 16);
                dot_4rows_4cols_acc(&a0_1l, &a0_1h, w, x0);
                dot_4rows_4cols_acc(&a1_1l, &a1_1h, w, x1);

                w = int4_row_to_s8(wc + 32);
                dot_4rows_4cols_acc(&a0_2l, &a0_2h, w, x0);
                dot_4rows_4cols_acc(&a1_2l, &a1_2h, w, x1);

                w = int4_row_to_s8(wc + 48);
                dot_4rows_4cols_acc(&a0_3l, &a0_3h, w, x0);
                dot_4rows_4cols_acc(&a1_3l, &a1_3h, w, x1);
            }

            float s0 = sc0[group];
            float s1 = sc1[group];

            r0_0 = vaddq_f32(r0_0, vmulq_f32(
                vmulq_n_f32(vcvtq_f32_s32(vpaddq_s32(a0_0l, a0_0h)), s0),
                scale_cache[group * 4 + 0]));
            r0_1 = vaddq_f32(r0_1, vmulq_f32(
                vmulq_n_f32(vcvtq_f32_s32(vpaddq_s32(a0_1l, a0_1h)), s0),
                scale_cache[group * 4 + 1]));
            r0_2 = vaddq_f32(r0_2, vmulq_f32(
                vmulq_n_f32(vcvtq_f32_s32(vpaddq_s32(a0_2l, a0_2h)), s0),
                scale_cache[group * 4 + 2]));
            r0_3 = vaddq_f32(r0_3, vmulq_f32(
                vmulq_n_f32(vcvtq_f32_s32(vpaddq_s32(a0_3l, a0_3h)), s0),
                scale_cache[group * 4 + 3]));

            r1_0 = vaddq_f32(r1_0, vmulq_f32(
                vmulq_n_f32(vcvtq_f32_s32(vpaddq_s32(a1_0l, a1_0h)), s1),
                scale_cache[group * 4 + 0]));
            r1_1 = vaddq_f32(r1_1, vmulq_f32(
                vmulq_n_f32(vcvtq_f32_s32(vpaddq_s32(a1_1l, a1_1h)), s1),
                scale_cache[group * 4 + 1]));
            r1_2 = vaddq_f32(r1_2, vmulq_f32(
                vmulq_n_f32(vcvtq_f32_s32(vpaddq_s32(a1_2l, a1_2h)), s1),
                scale_cache[group * 4 + 2]));
            r1_3 = vaddq_f32(r1_3, vmulq_f32(
                vmulq_n_f32(vcvtq_f32_s32(vpaddq_s32(a1_3l, a1_3h)), s1),
                scale_cache[group * 4 + 3]));
        }

        float *dst0 = output + row0 * weight->shape[0] + output_block * 16;
        vst1q_f32(dst0 + 0,  r0_0);
        vst1q_f32(dst0 + 4,  r0_1);
        vst1q_f32(dst0 + 8,  r0_2);
        vst1q_f32(dst0 + 12, r0_3);

        if (has_row1) {
            float *dst1 = output + (row0 + 1) * weight->shape[0] + output_block * 16;
            vst1q_f32(dst1 + 0,  r1_0);
            vst1q_f32(dst1 + 4,  r1_1);
            vst1q_f32(dst1 + 8,  r1_2);
            vst1q_f32(dst1 + 12, r1_3);
        }
    }
}

void matmul_int4(float *output, const int8_t *input_q, const float *input_scales,
                 const Tensor *weight, size_t rows) {
    const size_t block_rows = 16;
    #pragma omp for schedule(static)
    for (size_t output_block = 0; output_block < (size_t)weight->shape[0] / block_rows; output_block++) {
        if (rows == 1)
            matmul_int4_block_1x16(output, input_q, input_scales, weight, output_block);
        else
            matmul_int4_block_2x16(output, input_q, input_scales, weight, rows, output_block);
    }
}

// Converts each input row to int8 in groups of 64 values with a float scale recording each group's magnitude.
void quantize(int8_t *quantized, float *scales, const float *input, size_t rows, size_t width) {
    #pragma omp for schedule(static)
    for (size_t group_index = 0; group_index < rows * (width / 64); group_index++) {
        const float *group = input + group_index * 64;

        // NEON: find max absolute value (4-wide).
        float32x4_t m0 = vdupq_n_f32(0.0f);
        float32x4_t m1 = vdupq_n_f32(0.0f);
        float32x4_t m2 = vdupq_n_f32(0.0f);
        float32x4_t m3 = vdupq_n_f32(0.0f);
        for (int j = 0; j < 64; j += 16) {
            m0 = vmaxq_f32(m0, vabsq_f32(vld1q_f32(group + j + 0)));
            m1 = vmaxq_f32(m1, vabsq_f32(vld1q_f32(group + j + 4)));
            m2 = vmaxq_f32(m2, vabsq_f32(vld1q_f32(group + j + 8)));
            m3 = vmaxq_f32(m3, vabsq_f32(vld1q_f32(group + j + 12)));
        }
        float32x4_t m = vmaxq_f32(vmaxq_f32(m0, m1), vmaxq_f32(m2, m3));
        float max_abs = vmaxvq_f32(m);

        float scale = max_abs / 127.0f;
        float inverse_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

        // NEON: multiply, round to int32, narrow to int8 (8 at a time).
        for (int j = 0; j < 64; j += 8) {
            float32x4_t v0 = vmulq_n_f32(vld1q_f32(group + j), inverse_scale);
            float32x4_t v1 = vmulq_n_f32(vld1q_f32(group + j + 4), inverse_scale);
            int32x4_t q0 = vcvtnq_s32_f32(v0);
            int32x4_t q1 = vcvtnq_s32_f32(v1);
            int16x4_t q016 = vqmovn_s32(q0);
            int16x4_t q116 = vqmovn_s32(q1);
            int8x8_t q8 = vqmovn_s16(vcombine_s16(q016, q116));
            vst1_s8(quantized + group_index * 64 + j, q8);
        }
        scales[group_index] = scale;
    }
}

// int4 variant: quantize activations in groups of 32 to match the int4 weight scale granularity.
void quantize_int4(int8_t *quantized, float *scales, const float *input, size_t rows, size_t width) {
    #pragma omp for schedule(static)
    for (size_t group_index = 0; group_index < rows * (width / 32); group_index++) {
        const float *group = input + group_index * 32;

        // NEON: find max absolute value (4-wide).
        float32x4_t m0 = vdupq_n_f32(0.0f);
        float32x4_t m1 = vdupq_n_f32(0.0f);
        float32x4_t m2 = vdupq_n_f32(0.0f);
        float32x4_t m3 = vdupq_n_f32(0.0f);
        for (int j = 0; j < 32; j += 16) {
            m0 = vmaxq_f32(m0, vabsq_f32(vld1q_f32(group + j + 0)));
            m1 = vmaxq_f32(m1, vabsq_f32(vld1q_f32(group + j + 4)));
            m2 = vmaxq_f32(m2, vabsq_f32(vld1q_f32(group + j + 8)));
            m3 = vmaxq_f32(m3, vabsq_f32(vld1q_f32(group + j + 12)));
        }
        float32x4_t m = vmaxq_f32(vmaxq_f32(m0, m1), vmaxq_f32(m2, m3));
        float max_abs = vmaxvq_f32(m);

        float scale = max_abs / 127.0f;
        float inverse_scale = scale > 0.0f ? 1.0f / scale : 0.0f;

        // NEON: multiply, round to int32, narrow to int8 (8 at a time).
        for (int j = 0; j < 32; j += 8) {
            float32x4_t v0 = vmulq_n_f32(vld1q_f32(group + j), inverse_scale);
            float32x4_t v1 = vmulq_n_f32(vld1q_f32(group + j + 4), inverse_scale);
            int32x4_t q0 = vcvtnq_s32_f32(v0);
            int32x4_t q1 = vcvtnq_s32_f32(v1);
            int16x4_t q016 = vqmovn_s32(q0);
            int16x4_t q116 = vqmovn_s32(q1);
            int8x8_t q8 = vqmovn_s16(vcombine_s16(q016, q116));
            vst1_s8(quantized + group_index * 32 + j, q8);
        }
        scales[group_index] = scale;
    }
}

void attention_scores(float *scores, const float *query, const float *key_cache,
        int first_key, int num_keys, int cache_mask, int head_dim) {
    for (int key_index = 0; key_index < num_keys; key_index++) {
        const float *key = key_cache + ((first_key + key_index) & cache_mask) * head_dim;
        float32x4_t acc = vdupq_n_f32(0.0f);
        int j = 0;
        for (; j + 3 < head_dim; j += 4)
            acc = vmlaq_f32(acc, vld1q_f32(query + j), vld1q_f32(key + j));
        float sum = vaddvq_f32(acc);
        for (; j < head_dim; j++)
            sum += query[j] * key[j];
        scores[key_index] = sum;
    }
}

void weighted_value_sum(float *output, const float *probabilities, const float *value_cache,
        int first_key, int num_keys, int cache_mask, int head_dim) {
    for (int j = 0; j + 3 < head_dim; j += 4) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (int key_index = 0; key_index < num_keys; key_index++) {
            const float *value = value_cache + ((first_key + key_index) & cache_mask) * head_dim + j;
            acc = vmlaq_f32(acc, vdupq_n_f32(probabilities[key_index]), vld1q_f32(value));
        }
        vst1q_f32(output + j, acc);
    }
    for (int j = (head_dim / 4) * 4; j < head_dim; j++) {
        float sum = 0.0f;
        for (int key_index = 0; key_index < num_keys; key_index++)
            sum += probabilities[key_index] * value_cache[((first_key + key_index) & cache_mask) * head_dim + j];
        output[j] = sum;
    }
}

// Approximates GELU from the exported lookup table and multiplies it by the up projection to produce the MLP's gated activation.
void geglu(float *gate, const float *up, int rows, int width, int up_stride, const Tensor *gelu_table) {
    const float *table = (const float *)gelu_table->data;
    const int table_size = gelu_table->shape[0];
    const float lower = (float)gelu_table->shape[1];
    const float upper = (float)gelu_table->shape[2];
    const float scale = (float)(table_size - 1) / (upper - lower);
    #pragma omp for collapse(2) schedule(static)
    for (int row = 0; row < rows; row++) {
        for (int i = 0; i < width; i++) {
            float x = gate[row * width + i];
            if (x <= lower) {
                x = table[0];
            } else if (!(x >= upper)) {
                float position = (x - lower) * scale;
                int index = (int)position;
                float fraction = position - (float)index;
                x = table[index] + fraction * (table[index + 1] - table[index]);
            }
            gate[row * width + i] = x * up[row * up_stride + i];
        }
    }
}
