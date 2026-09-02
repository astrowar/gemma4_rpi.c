// ARM NEON int4 kernel implementations for gemma4.c.
// Compiled only on aarch64 (see Makefile KERNELS=neon).
//
// Weights are packed two 4-bit values per byte:
//   data[block*16*(width/2) + group*16*16 + row*16 + byte]
// A 16-byte NEON load of one row's group therefore holds 32 signed int4
// weights (low nibble = even input, high nibble = odd input), with 8 as the
// zero point. Each (group, row) carries one fp16 scale, laid out
//   scales[(block*groups + group)*16 + row].
//
// The int4 values are widened to int8 (nibble - 8) and accumulated with the
// same vmull_s8/vpadalq_s16 machinery as the int8 path, so the integer dot
// product is exact; only the rescale step differs (scale every 32 inputs).

#if !defined(__aarch64__)
#error "kernels_neon_int4.c requires an aarch64 target."
#endif

#include "gemma4.h"
#include <arm_neon.h>

// ----------------------------------------------------------------------------
// fp16 → float

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
            int p = 31 - __builtin_clz(mant);
            uint32_t e = (uint32_t)(p + 103);
            uint32_t frac = (mant << (23 - p)) & 0x007fffffu;
            bits = sign | (e << 23) | frac;
        }
    }
    else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
        if (mant) bits |= 0x00400000u;
    }
    else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }

    float result;
    __builtin_memcpy(&result, &bits, sizeof(result));
    return result;
}

// ----------------------------------------------------------------------------
// Shared helpers (same as kernels_neon_int8.c)

// Duplicates 4 bytes into all four 4-byte lanes of an int8x16 register.
static inline __attribute__((always_inline))
int8x16_t duplicate_4_s8(const int8_t *p) {
    uint32_t x;
    __builtin_memcpy(&x, p, sizeof(x));
    return vreinterpretq_s8_u32(vdupq_n_u32(x));
}

// Widening multiply-accumulate for 4 rows × 4 cols.
static inline __attribute__((always_inline))
void dot_4rows_4cols_acc(int32x4_t *acc_lo, int32x4_t *acc_hi,
                         int8x16_t w, int8x16_t x) {
    int16x8_t lo = vmull_s8(vget_low_s8(w), vget_low_s8(x));
    int16x8_t hi = vmull_high_s8(w, x);
    *acc_lo = vpadalq_s16(*acc_lo, lo);
    *acc_hi = vpadalq_s16(*acc_hi, hi);
}

// ----------------------------------------------------------------------------
// int4-specific helpers

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

// ----------------------------------------------------------------------------
// GEMV: 1 input row x 16 output rows (decode).

static inline __attribute__((always_inline)) void matmul_int4_block_1x16(
    float *output, const int8_t *input_q, const float *input_scales,
    const Tensor *weight, size_t output_block) {
    const int block_rows = 16;
    const int width = weight->shape[1];
    const int groups = width / 32;

    const int8_t *input_row = input_q;
    const float *input_scale_row = input_scales;
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

// ----------------------------------------------------------------------------
// GEMM: 2 input rows x 16 output rows (prefill). Weight loads shared.

static inline __attribute__((always_inline)) void matmul_int4_block_2x16(
    float *output, const int8_t *input_q, const float *input_scales,
    const Tensor *weight, size_t rows, size_t output_block) {
    const int block_rows = 16;
    const int width = weight->shape[1];
    const int groups = width / 32;

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

// ----------------------------------------------------------------------------
// Entry point.

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
