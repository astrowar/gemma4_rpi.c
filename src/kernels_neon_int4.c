// ARM NEON int4 kernel implementations for gemma4.c.
// Compiled only on aarch64 (see Makefile KERNELS=neon).
//
// Weights are packed two 4-bit values per byte:
//   data[block*16*(width/2) + group*16*16 + row*16 + byte]
// A 16-byte NEON load of one row's group holds 32 signed int4
// weights (low nibble = even input, high nibble = odd input), with 8 as the
// zero point. Each (group, row) carries one fp16 scale, laid out
//   scales[(block*groups + group)*16 + row].
//
// Unlike the int8 path (where a 16-byte load covers 4 rows x 4 inputs),
// a 16-byte int4 load covers 1 row x 32 inputs. The kernel iterates over
// rows and uses NEON for the integer dot product.
//
// Optimizations:
//   - vzip1q/vzip2q for int4 unpack (fewer instructions)
//   - SDOT (ARMv8.2+) when available, falling back to vmull/vpaddl
//   - Vector FP16→FP32 conversion when FP16 arithmetic is available
//   - FMA for the rescale/accumulate step
//   - 4x16 GEMM for prefill (rows>=4) to amortize weight unpacking

#if !defined(__aarch64__)
#error "kernels_neon_int4.c requires an aarch64 target."
#endif

#include "gemma4.h"
#include <arm_neon.h>

// ----------------------------------------------------------------------------
// fp16 → float (scalar fallback)

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

// Convert 16 fp16 scales to 4 float32x4 vectors (one per 4 rows).
static inline __attribute__((always_inline))
void fp16_scales_to_f32(const uint16_t *ws, float32x4_t *out) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    float16x8_t h0 = vreinterpretq_f16_u16(vld1q_u16(ws));
    float16x8_t h1 = vreinterpretq_f16_u16(vld1q_u16(ws + 8));
    out[0] = vcvt_f32_f16(vget_low_f16(h0));
    out[1] = vcvt_f32_f16(vget_high_f16(h0));
    out[2] = vcvt_f32_f16(vget_low_f16(h1));
    out[3] = vcvt_f32_f16(vget_high_f16(h1));
#else
    out[0] = (float32x4_t) { fp16_to_f32(ws[0]),  fp16_to_f32(ws[1]),
                             fp16_to_f32(ws[2]),  fp16_to_f32(ws[3]) };
    out[1] = (float32x4_t) { fp16_to_f32(ws[4]),  fp16_to_f32(ws[5]),
                             fp16_to_f32(ws[6]),  fp16_to_f32(ws[7]) };
    out[2] = (float32x4_t) { fp16_to_f32(ws[8]),  fp16_to_f32(ws[9]),
                             fp16_to_f32(ws[10]), fp16_to_f32(ws[11]) };
    out[3] = (float32x4_t) { fp16_to_f32(ws[12]), fp16_to_f32(ws[13]),
                             fp16_to_f32(ws[14]), fp16_to_f32(ws[15]) };
#endif
}

// ----------------------------------------------------------------------------
// int4 unpack: widen one 16-byte row (32 values) to two int8x16 registers.
// Uses vzip1q/vzip2q to interleave even/odd nibbles.

static inline __attribute__((always_inline))
void int4_row_to_s8x2(const uint8_t *row, int8x16_t *out_lo, int8x16_t *out_hi) {
    uint8x16_t packed = vld1q_u8(row);
    uint8x16_t low  = vandq_u8(packed, vdupq_n_u8(0x0f));
    uint8x16_t high = vshrq_n_u8(packed, 4);

    uint8x16_t z0 = vzip1q_u8(low, high);
    uint8x16_t z1 = vzip2q_u8(low, high);

    int8x16_t zp = vdupq_n_s8(8);
    *out_lo = vsubq_s8(vreinterpretq_s8_u8(z0), zp);
    *out_hi = vsubq_s8(vreinterpretq_s8_u8(z1), zp);
}

// ----------------------------------------------------------------------------
// Dot product of two int8x16 vectors → int32.

static inline __attribute__((always_inline))
int32_t dot16_s8(int8x16_t x, int8x16_t w) {
#if defined(__ARM_FEATURE_DOTPROD)
    int32x4_t acc = vdupq_n_s32(0);
    acc = vdotq_s32(acc, x, w);
    return vaddvq_s32(acc);
#else
    int16x8_t p0 = vmull_s8(vget_low_s8(x), vget_low_s8(w));
    int16x8_t p1 = vmull_high_s8(x, w);
    int32x4_t s = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
    return vaddvq_s32(s);
#endif
}

// Full 32-value dot: unpack one int4 row and dot against two int8x16 inputs.
static inline __attribute__((always_inline))
int32_t dot32_int4(const uint8_t *row, int8x16_t x0, int8x16_t x1) {
    int8x16_t w0, w1;
    int4_row_to_s8x2(row, &w0, &w1);
    return dot16_s8(x0, w0) + dot16_s8(x1, w1);
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

    // Convert all weight scales once per output_block.
    float32x4_t scale_cache[groups][4];
    for (int group = 0; group < groups; group++) {
        const uint16_t *ws = weight->scales + ((output_block * groups + group) * 16);
        fp16_scales_to_f32(ws, scale_cache[group]);
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

        int8x16_t x0 = vld1q_s8(in + 0);
        int8x16_t x1 = vld1q_s8(in + 16);

        float s = input_scale_row[group];

        int32_t dots[16];
        for (int row = 0; row < 16; row++)
            dots[row] = dot32_int4(wg + row * 16, x0, x1);

        float32x4_t d0 = vcvtq_f32_s32(vld1q_s32(dots + 0));
        float32x4_t d1 = vcvtq_f32_s32(vld1q_s32(dots + 4));
        float32x4_t d2 = vcvtq_f32_s32(vld1q_s32(dots + 8));
        float32x4_t d3 = vcvtq_f32_s32(vld1q_s32(dots + 12));

        r0 = vfmaq_f32(r0, d0, vmulq_n_f32(scale_cache[group][0], s));
        r1 = vfmaq_f32(r1, d1, vmulq_n_f32(scale_cache[group][1], s));
        r2 = vfmaq_f32(r2, d2, vmulq_n_f32(scale_cache[group][2], s));
        r3 = vfmaq_f32(r3, d3, vmulq_n_f32(scale_cache[group][3], s));
    }

    float *dst = output + output_block * 16;
    vst1q_f32(dst + 0,  r0);
    vst1q_f32(dst + 4,  r1);
    vst1q_f32(dst + 8,  r2);
    vst1q_f32(dst + 12, r3);
}

// ----------------------------------------------------------------------------
// GEMM: 2 input rows x 16 output rows (prefill, rows 2-3).

static inline __attribute__((always_inline)) void matmul_int4_block_2x16(
    float *output, const int8_t *input_q, const float *input_scales,
    const Tensor *weight, size_t rows, size_t output_block) {
    const int block_rows = 16;
    const int width = weight->shape[1];
    const int groups = width / 32;

    const uint8_t *packed_weights =
        (const uint8_t *)weight->data + output_block * block_rows * (width / 2);

    // Convert all weight scales once per output_block.
    float32x4_t scale_cache[groups][4];
    for (int group = 0; group < groups; group++) {
        const uint16_t *ws = weight->scales + ((output_block * groups + group) * 16);
        fp16_scales_to_f32(ws, scale_cache[group]);
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

            int8x16_t x00 = vld1q_s8(g0 + 0);
            int8x16_t x01 = vld1q_s8(g0 + 16);
            int8x16_t x10, x11;
            if (has_row1) {
                x10 = vld1q_s8(g1 + 0);
                x11 = vld1q_s8(g1 + 16);
            }

            float s0 = sc0[group];
            float s1 = sc1[group];

            int32_t d0[16];
            int32_t d1[16] = {0};
            for (int row = 0; row < 16; row++) {
                const uint8_t *wrow = wg + row * 16;
                d0[row] = dot32_int4(wrow, x00, x01);
                if (has_row1)
                    d1[row] = dot32_int4(wrow, x10, x11);
            }

            float32x4_t dd0_0 = vcvtq_f32_s32(vld1q_s32(d0 + 0));
            float32x4_t dd0_1 = vcvtq_f32_s32(vld1q_s32(d0 + 4));
            float32x4_t dd0_2 = vcvtq_f32_s32(vld1q_s32(d0 + 8));
            float32x4_t dd0_3 = vcvtq_f32_s32(vld1q_s32(d0 + 12));

            r0_0 = vfmaq_f32(r0_0, dd0_0, vmulq_n_f32(scale_cache[group][0], s0));
            r0_1 = vfmaq_f32(r0_1, dd0_1, vmulq_n_f32(scale_cache[group][1], s0));
            r0_2 = vfmaq_f32(r0_2, dd0_2, vmulq_n_f32(scale_cache[group][2], s0));
            r0_3 = vfmaq_f32(r0_3, dd0_3, vmulq_n_f32(scale_cache[group][3], s0));

            if (has_row1) {
                float32x4_t dd1_0 = vcvtq_f32_s32(vld1q_s32(d1 + 0));
                float32x4_t dd1_1 = vcvtq_f32_s32(vld1q_s32(d1 + 4));
                float32x4_t dd1_2 = vcvtq_f32_s32(vld1q_s32(d1 + 8));
                float32x4_t dd1_3 = vcvtq_f32_s32(vld1q_s32(d1 + 12));

                r1_0 = vfmaq_f32(r1_0, dd1_0, vmulq_n_f32(scale_cache[group][0], s1));
                r1_1 = vfmaq_f32(r1_1, dd1_1, vmulq_n_f32(scale_cache[group][1], s1));
                r1_2 = vfmaq_f32(r1_2, dd1_2, vmulq_n_f32(scale_cache[group][2], s1));
                r1_3 = vfmaq_f32(r1_3, dd1_3, vmulq_n_f32(scale_cache[group][3], s1));
            }
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
// GEMM: 4 input rows x 16 output rows (prefill, rows >= 4).
// Weight unpacking is amortized across 4 input rows.
// Uses int32x4_t accumulators (4 rows per vector) to minimize FP ops.

static inline __attribute__((always_inline)) void matmul_int4_block_4x16(
    float *output, const int8_t *input_q, const float *input_scales,
    const Tensor *weight, size_t rows, size_t output_block) {
    const int block_rows = 16;
    const int width = weight->shape[1];
    const int groups = width / 32;

    const uint8_t *packed_weights =
        (const uint8_t *)weight->data + output_block * block_rows * (width / 2);

    // Convert all weight scales once per output_block (16 rows x groups).
    // Layout: scale_cache[group][4] = 4 float32x4 vectors (one per 4 rows).
    float32x4_t scale_cache[groups][4];
    for (int group = 0; group < groups; group++) {
        const uint16_t *ws = weight->scales + ((output_block * groups + group) * 16);
        fp16_scales_to_f32(ws, scale_cache[group]);
    }

    for (size_t row0 = 0; row0 + 4 <= rows; row0 += 4) {
        const int8_t *in[4] = {
            input_q + row0 * width,
            input_q + (row0 + 1) * width,
            input_q + (row0 + 2) * width,
            input_q + (row0 + 3) * width
        };
        const float *sc[4] = {
            input_scales + row0 * groups,
            input_scales + (row0 + 1) * groups,
            input_scales + (row0 + 2) * groups,
            input_scales + (row0 + 3) * groups
        };

        // Float accumulators: 4 per input row, each holding 4 output rows.
        float32x4_t r0[4] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                               vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };
        float32x4_t r1[4] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                               vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };
        float32x4_t r2[4] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                               vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };
        float32x4_t r3[4] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                               vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };

        for (int group = 0; group < groups; group++) {
            const uint8_t *wg = packed_weights + (size_t)group * 16 * 16;

            if (group + 1 < groups)
                __builtin_prefetch(packed_weights + (size_t)(group + 1) * 16 * 16, 0, 1);

            int8x16_t x[4][2];
            for (int i = 0; i < 4; i++) {
                x[i][0] = vld1q_s8(in[i] + group * 32);
                x[i][1] = vld1q_s8(in[i] + group * 32 + 16);
            }

            // Process 4 output rows at a time using int32x4 accumulators.
            for (int r = 0; r < 16; r += 4) {
                // Unpack 4 weight rows.
                int8x16_t w0[4], w1[4];
                for (int i = 0; i < 4; i++)
                    int4_row_to_s8x2(wg + (r + i) * 16, &w0[i], &w1[i]);

                // Compute dot products: int32x4 per input row (4 output rows each).
                int32x4_t d0 = vdupq_n_s32(0);
                int32x4_t d1 = vdupq_n_s32(0);
                int32x4_t d2 = vdupq_n_s32(0);
                int32x4_t d3 = vdupq_n_s32(0);

                for (int i = 0; i < 4; i++) {
                    // dot16_s8 returns scalar; we need vector dot across 4 rows.
                    // Each w0[i]/w1[i] is one row's weights; x[i][0]/x[i][1] is one input.
                    // We compute 4 scalar dots and pack them.
                    int32_t dots[4];
                    for (int j = 0; j < 4; j++)
                        dots[j] = dot16_s8(x[i][0], w0[j]) + dot16_s8(x[i][1], w1[j]);
                    int32x4_t dv = vld1q_s32(dots);

                    if (i == 0) d0 = dv;
                    else if (i == 1) d1 = dv;
                    else if (i == 2) d2 = dv;
                    else d3 = dv;
                }

                // Rescale and accumulate using pre-computed scales.
                float s0 = sc[0][group], s1 = sc[1][group];
                float s2 = sc[2][group], s3 = sc[3][group];

                r0[r/4] = vfmaq_f32(r0[r/4], vcvtq_f32_s32(d0), vmulq_n_f32(scale_cache[group][r/4], s0));
                r1[r/4] = vfmaq_f32(r1[r/4], vcvtq_f32_s32(d1), vmulq_n_f32(scale_cache[group][r/4], s1));
                r2[r/4] = vfmaq_f32(r2[r/4], vcvtq_f32_s32(d2), vmulq_n_f32(scale_cache[group][r/4], s2));
                r3[r/4] = vfmaq_f32(r3[r/4], vcvtq_f32_s32(d3), vmulq_n_f32(scale_cache[group][r/4], s3));
            }
        }

        // Store results.
        for (int i = 0; i < 4; i++) {
            float *dst = output + (row0 + i) * weight->shape[0] + output_block * 16;
            float32x4_t *ri = (i == 0) ? r0 : (i == 1) ? r1 : (i == 2) ? r2 : r3;
            vst1q_f32(dst + 0,  ri[0]);
            vst1q_f32(dst + 4,  ri[1]);
            vst1q_f32(dst + 8,  ri[2]);
            vst1q_f32(dst + 12, ri[3]);
        }
    }

    // Tail: handle remaining rows (1-3) with the 2x16 kernel.
    size_t tail_start = (rows / 4) * 4;
    if (tail_start < rows) {
        for (size_t row = tail_start; row < rows; row++) {
            float *dst = output + row * weight->shape[0] + output_block * 16;
            for (int i = 0; i < 16; i++) dst[i] = 0.0f;
        }
        const int8_t *tail_in = input_q + tail_start * width;
        const float *tail_sc = input_scales + tail_start * groups;
        size_t tail_rows = rows - tail_start;

        for (size_t row0 = 0; row0 < tail_rows; row0 += 2) {
            const int has_row1 = (row0 + 1 < tail_rows);
            const int8_t *in0 = tail_in + row0 * width;
            const float *sc0 = tail_sc + row0 * groups;
            const int8_t *in1 = has_row1 ? tail_in + (row0 + 1) * width : in0;
            const float *sc1 = has_row1 ? tail_sc + (row0 + 1) * groups : sc0;

            float32x4_t r0_0 = vdupq_n_f32(0.0f), r0_1 = vdupq_n_f32(0.0f);
            float32x4_t r0_2 = vdupq_n_f32(0.0f), r0_3 = vdupq_n_f32(0.0f);
            float32x4_t r1_0 = vdupq_n_f32(0.0f), r1_1 = vdupq_n_f32(0.0f);
            float32x4_t r1_2 = vdupq_n_f32(0.0f), r1_3 = vdupq_n_f32(0.0f);

            for (int group = 0; group < groups; group++) {
                const uint8_t *wg = packed_weights + (size_t)group * 16 * 16;
                int8x16_t x00 = vld1q_s8(in0 + group * 32);
                int8x16_t x01 = vld1q_s8(in0 + group * 32 + 16);
                int8x16_t x10, x11;
                if (has_row1) {
                    x10 = vld1q_s8(in1 + group * 32);
                    x11 = vld1q_s8(in1 + group * 32 + 16);
                }

                float s0 = sc0[group];
                float s1 = sc1[group];

                for (int r = 0; r < 16; r += 4) {
                    int32_t d0[4], d1[4] = {0,0,0,0};
                    for (int i = 0; i < 4; i++) {
                        const uint8_t *wrow = wg + (r + i) * 16;
                        int8x16_t w0, w1;
                        int4_row_to_s8x2(wrow, &w0, &w1);
                        d0[i] = dot16_s8(x00, w0) + dot16_s8(x01, w1);
                        if (has_row1)
                            d1[i] = dot16_s8(x10, w0) + dot16_s8(x11, w1);
                    }

                    float32x4_t dd0 = vcvtq_f32_s32(vld1q_s32(d0));
                    r0_0 = vfmaq_f32(r0_0, dd0, vmulq_n_f32(scale_cache[group][0], s0));
                    r0_1 = vfmaq_f32(r0_1, dd0, vmulq_n_f32(scale_cache[group][1], s0));
                    r0_2 = vfmaq_f32(r0_2, dd0, vmulq_n_f32(scale_cache[group][2], s0));
                    r0_3 = vfmaq_f32(r0_3, dd0, vmulq_n_f32(scale_cache[group][3], s0));

                    if (has_row1) {
                        float32x4_t dd1 = vcvtq_f32_s32(vld1q_s32(d1));
                        r1_0 = vfmaq_f32(r1_0, dd1, vmulq_n_f32(scale_cache[group][0], s1));
                        r1_1 = vfmaq_f32(r1_1, dd1, vmulq_n_f32(scale_cache[group][1], s1));
                        r1_2 = vfmaq_f32(r1_2, dd1, vmulq_n_f32(scale_cache[group][2], s1));
                        r1_3 = vfmaq_f32(r1_3, dd1, vmulq_n_f32(scale_cache[group][3], s1));
                    }
                }
            }

            float *dst0 = output + (tail_start + row0) * weight->shape[0] + output_block * 16;
            vst1q_f32(dst0 + 0,  r0_0);
            vst1q_f32(dst0 + 4,  r0_1);
            vst1q_f32(dst0 + 8,  r0_2);
            vst1q_f32(dst0 + 12, r0_3);

            if (has_row1) {
                float *dst1 = output + (tail_start + row0 + 1) * weight->shape[0] + output_block * 16;
                vst1q_f32(dst1 + 0,  r1_0);
                vst1q_f32(dst1 + 4,  r1_1);
                vst1q_f32(dst1 + 8,  r1_2);
                vst1q_f32(dst1 + 12, r1_3);
            }
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
        else if (rows >= 4)
            matmul_int4_block_4x16(output, input_q, input_scales, weight, rows, output_block);
        else
            matmul_int4_block_2x16(output, input_q, input_scales, weight, rows, output_block);
    }
}
