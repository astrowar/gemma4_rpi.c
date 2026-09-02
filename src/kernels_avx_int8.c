// AVX2 / AVX-512 VNNI int8 kernel implementations.
// Only compiled when the target supports AVX2 (see Makefile).
// For portable scalar fallback, see kernels_pure_int8.c.

#if !defined(__AVX2__)
#error "kernels_avx_int8.c requires -mavx2 (or -march=native on an AVX2 CPU). Use kernels_pure_int8.c for scalar targets."
#endif
#if !defined(__FMA__)
#error "kernels_avx_int8.c requires -mfma for _mm256_fmadd_ps."
#endif
#if !defined(__F16C__)
#error "kernels_avx_int8.c requires -mf16c for _mm256_cvtph_ps."
#endif

#include "gemma4.h"
#include <assert.h>

// ============================================================================
// AVX2 matmul: int8 weights x int8 activations -> float output
// ============================================================================

// Single multiply-accumulate step: acc += Σ(abs_w[i] * sign(x[i], w[i]))
// Uses maddubs to pair adjacent bytes into int16 products, then madd_epi16
// to sum four int16 pairs into one int32 lane.
static inline __attribute__((always_inline))
__m256i dot4_acc(__m256i acc, __m256i x, __m256i w, __m256i abs_w, __m256i ones) {
    __m256i sx = _mm256_sign_epi8(x, w);
    __m256i p16 = _mm256_maddubs_epi16(abs_w, sx);
    __m256i p32 = _mm256_madd_epi16(p16, ones);
    return _mm256_add_epi32(acc, p32);
}

// Fast path for decode (rows == 1): single input row, no arrays, no inner row loop.
// Both halves (outputs 0-7 and 8-15) share a single broadcast of each input
// chunk, halving the activation loads compared to a naive two-pass approach.
static inline __attribute__((always_inline)) void matmul_block_1x16(
    float *output, const int8_t *input_q, const float *input_scales,
    const Tensor *weight, size_t output_block) {
    const size_t block_rows = 16;
    const size_t width = (size_t)weight->shape[1];
    const size_t groups = width / 64;
    const int8_t *pw = (const int8_t *)weight->data + output_block * block_rows * width;
    const __m256i ones = _mm256_set1_epi16(1);

    __m256 result0 = _mm256_setzero_ps();
    __m256 result1 = _mm256_setzero_ps();

    for (size_t group = 0; group < groups; group++) {
        __m256i dot0 = _mm256_setzero_si256();
        __m256i dot1 = _mm256_setzero_si256();

        for (int chunk = 0; chunk < 16; chunk++) {
            __m256i x = _mm256_broadcastd_epi32(
                _mm_loadu_si32(input_q + group * 64 + chunk * 4));

            const int8_t *base = pw + group * 1024 + chunk * 64;
            __m256i w0 = _mm256_loadu_si256((const __m256i *)(base));
            __m256i w1 = _mm256_loadu_si256((const __m256i *)(base + 32));

            dot0 = dot4_acc(dot0, x, w0, _mm256_abs_epi8(w0), ones);
            dot1 = dot4_acc(dot1, x, w1, _mm256_abs_epi8(w1), ones);
        }

        __m256 ws0 = _mm256_cvtph_ps(_mm_loadu_si128(
            (const __m128i *)(weight->scales + (output_block * groups + group) * block_rows)));
        __m256 ws1 = _mm256_cvtph_ps(_mm_loadu_si128(
            (const __m128i *)(weight->scales + (output_block * groups + group) * block_rows + 8)));
        result0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot0),
                                  _mm256_mul_ps(ws0, _mm256_set1_ps(input_scales[group])),
                                  result0);
        result1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot1),
                                  _mm256_mul_ps(ws1, _mm256_set1_ps(input_scales[group])),
                                  result1);
    }

    _mm256_storeu_ps(output + output_block * block_rows, result0);
    _mm256_storeu_ps(output + output_block * block_rows + 8, result1);
}

// Full 4x16 block: all four rows active. No conditional branches in the
// inner loop, giving the compiler maximum freedom for scheduling.
static inline __attribute__((always_inline)) void matmul_block_4x16_full(
    float *output, const int8_t *input_q, const float *input_scales,
    const Tensor *weight, size_t row_start, size_t output_block) {
    const size_t block_rows = 16;
    const size_t width = (size_t)weight->shape[1];
    const size_t groups = width / 64;
    const int8_t *pw = (const int8_t *)weight->data + output_block * block_rows * width;
    const __m256i ones = _mm256_set1_epi16(1);

    for (int half = 0; half < 2; half++) {
        __m256 r0 = _mm256_setzero_ps();
        __m256 r1 = _mm256_setzero_ps();
        __m256 r2 = _mm256_setzero_ps();
        __m256 r3 = _mm256_setzero_ps();

        for (size_t group = 0; group < groups; group++) {
            __m256i d0 = _mm256_setzero_si256();
            __m256i d1 = _mm256_setzero_si256();
            __m256i d2 = _mm256_setzero_si256();
            __m256i d3 = _mm256_setzero_si256();

            for (int chunk = 0; chunk < 16; chunk++) {
                __m256i w = _mm256_loadu_si256(
                    (const __m256i *)(pw + group * 1024 + chunk * 64 + half * 32));
                __m256i wm = _mm256_abs_epi8(w);

                __m256i x0 = _mm256_broadcastd_epi32(
                    _mm_loadu_si32(input_q + (row_start + 0) * width + group * 64 + chunk * 4));
                d0 = dot4_acc(d0, x0, w, wm, ones);
                __m256i x1 = _mm256_broadcastd_epi32(
                    _mm_loadu_si32(input_q + (row_start + 1) * width + group * 64 + chunk * 4));
                d1 = dot4_acc(d1, x1, w, wm, ones);
                __m256i x2 = _mm256_broadcastd_epi32(
                    _mm_loadu_si32(input_q + (row_start + 2) * width + group * 64 + chunk * 4));
                d2 = dot4_acc(d2, x2, w, wm, ones);
                __m256i x3 = _mm256_broadcastd_epi32(
                    _mm_loadu_si32(input_q + (row_start + 3) * width + group * 64 + chunk * 4));
                d3 = dot4_acc(d3, x3, w, wm, ones);
            }

            __m256 ws = _mm256_cvtph_ps(_mm_loadu_si128(
                (const __m128i *)(weight->scales + (output_block * groups + group) * block_rows + half * 8)));
            r0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d0),
                                 _mm256_mul_ps(ws, _mm256_set1_ps(input_scales[(row_start + 0) * groups + group])), r0);
            r1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d1),
                                 _mm256_mul_ps(ws, _mm256_set1_ps(input_scales[(row_start + 1) * groups + group])), r1);
            r2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d2),
                                 _mm256_mul_ps(ws, _mm256_set1_ps(input_scales[(row_start + 2) * groups + group])), r2);
            r3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d3),
                                 _mm256_mul_ps(ws, _mm256_set1_ps(input_scales[(row_start + 3) * groups + group])), r3);
        }

        _mm256_storeu_ps(output + (row_start + 0) * weight->shape[0] + output_block * block_rows + half * 8, r0);
        _mm256_storeu_ps(output + (row_start + 1) * weight->shape[0] + output_block * block_rows + half * 8, r1);
        _mm256_storeu_ps(output + (row_start + 2) * weight->shape[0] + output_block * block_rows + half * 8, r2);
        _mm256_storeu_ps(output + (row_start + 3) * weight->shape[0] + output_block * block_rows + half * 8, r3);
    }
}

// Tail for 4x16: fewer than 4 active rows.
static inline __attribute__((always_inline)) void matmul_block_4x16_tail(
    float *output, const int8_t *input_q, const float *input_scales,
    const Tensor *weight, size_t row_start, size_t active, size_t output_block) {
    const size_t block_rows = 16;
    const size_t width = (size_t)weight->shape[1];
    const size_t groups = width / 64;
    const int8_t *pw = (const int8_t *)weight->data + output_block * block_rows * width;
    const __m256i ones = _mm256_set1_epi16(1);

    for (int half = 0; half < 2; half++) {
        __m256 r0 = _mm256_setzero_ps();
        __m256 r1 = _mm256_setzero_ps();
        __m256 r2 = _mm256_setzero_ps();
        __m256 r3 = _mm256_setzero_ps();

        for (size_t group = 0; group < groups; group++) {
            __m256i d0 = _mm256_setzero_si256();
            __m256i d1 = _mm256_setzero_si256();
            __m256i d2 = _mm256_setzero_si256();
            __m256i d3 = _mm256_setzero_si256();

            for (int chunk = 0; chunk < 16; chunk++) {
                __m256i w = _mm256_loadu_si256(
                    (const __m256i *)(pw + group * 1024 + chunk * 64 + half * 32));
                __m256i wm = _mm256_abs_epi8(w);

                __m256i x0 = _mm256_broadcastd_epi32(
                    _mm_loadu_si32(input_q + (row_start + 0) * width + group * 64 + chunk * 4));
                d0 = dot4_acc(d0, x0, w, wm, ones);

                if (active > 1) {
                    __m256i x1 = _mm256_broadcastd_epi32(
                        _mm_loadu_si32(input_q + (row_start + 1) * width + group * 64 + chunk * 4));
                    d1 = dot4_acc(d1, x1, w, wm, ones);
                }
                if (active > 2) {
                    __m256i x2 = _mm256_broadcastd_epi32(
                        _mm_loadu_si32(input_q + (row_start + 2) * width + group * 64 + chunk * 4));
                    d2 = dot4_acc(d2, x2, w, wm, ones);
                }
                if (active > 3) {
                    __m256i x3 = _mm256_broadcastd_epi32(
                        _mm_loadu_si32(input_q + (row_start + 3) * width + group * 64 + chunk * 4));
                    d3 = dot4_acc(d3, x3, w, wm, ones);
                }
            }

            __m256 ws = _mm256_cvtph_ps(_mm_loadu_si128(
                (const __m128i *)(weight->scales + (output_block * groups + group) * block_rows + half * 8)));
            r0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d0),
                                 _mm256_mul_ps(ws, _mm256_set1_ps(input_scales[(row_start + 0) * groups + group])), r0);
            if (active > 1)
                r1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d1),
                                     _mm256_mul_ps(ws, _mm256_set1_ps(input_scales[(row_start + 1) * groups + group])), r1);
            if (active > 2)
                r2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d2),
                                     _mm256_mul_ps(ws, _mm256_set1_ps(input_scales[(row_start + 2) * groups + group])), r2);
            if (active > 3)
                r3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d3),
                                     _mm256_mul_ps(ws, _mm256_set1_ps(input_scales[(row_start + 3) * groups + group])), r3);
        }

        _mm256_storeu_ps(output + (row_start + 0) * weight->shape[0] + output_block * block_rows + half * 8, r0);
        if (active > 1)
            _mm256_storeu_ps(output + (row_start + 1) * weight->shape[0] + output_block * block_rows + half * 8, r1);
        if (active > 2)
            _mm256_storeu_ps(output + (row_start + 2) * weight->shape[0] + output_block * block_rows + half * 8, r2);
        if (active > 3)
            _mm256_storeu_ps(output + (row_start + 3) * weight->shape[0] + output_block * block_rows + half * 8, r3);
    }
}

// ============================================================================
// AVX-512 VNNI matmul: int8 weights x int8 activations -> float output
// ============================================================================

#if defined(__AVX512VNNI__)

// Full 8x16 block: eight input rows processed simultaneously.
// Correction (128 * sum_w) is computed once per group since it depends only
// on the weights, not on the input tokens.
static inline __attribute__((always_inline)) void matmul_block_vnni_8x16(
    float *output, const int8_t *input_q, const float *input_scales,
    const Tensor *weight, size_t output_block) {
    const size_t block_rows = 16;
    const size_t width = (size_t)weight->shape[1];
    const size_t groups = width / 64;
    const int8_t *pw = (const int8_t *)weight->data + output_block * block_rows * width;
    const __m512i offset = _mm512_set1_epi8(-128);

    __m512 r0 = _mm512_setzero_ps();
    __m512 r1 = _mm512_setzero_ps();
    __m512 r2 = _mm512_setzero_ps();
    __m512 r3 = _mm512_setzero_ps();
    __m512 r4 = _mm512_setzero_ps();
    __m512 r5 = _mm512_setzero_ps();
    __m512 r6 = _mm512_setzero_ps();
    __m512 r7 = _mm512_setzero_ps();

    for (size_t group = 0; group < groups; group++) {
        __m512i d0 = _mm512_setzero_si512();
        __m512i d1 = _mm512_setzero_si512();
        __m512i d2 = _mm512_setzero_si512();
        __m512i d3 = _mm512_setzero_si512();
        __m512i d4 = _mm512_setzero_si512();
        __m512i d5 = _mm512_setzero_si512();
        __m512i d6 = _mm512_setzero_si512();
        __m512i d7 = _mm512_setzero_si512();
        __m512i correction = _mm512_setzero_si512();

        for (int chunk = 0; chunk < 16; chunk++) {
            __m512i w = _mm512_loadu_si512(
                (const void *)(pw + group * 1024 + chunk * 64));
            correction = _mm512_dpbusd_epi32(correction, offset, w);

            __m512i x0 = _mm512_broadcastd_epi32(
                _mm_loadu_si32(input_q + 0 * width + group * 64 + chunk * 4));
            d0 = _mm512_dpbusd_epi32(d0, _mm512_xor_si512(x0, offset), w);
            __m512i x1 = _mm512_broadcastd_epi32(
                _mm_loadu_si32(input_q + 1 * width + group * 64 + chunk * 4));
            d1 = _mm512_dpbusd_epi32(d1, _mm512_xor_si512(x1, offset), w);
            __m512i x2 = _mm512_broadcastd_epi32(
                _mm_loadu_si32(input_q + 2 * width + group * 64 + chunk * 4));
            d2 = _mm512_dpbusd_epi32(d2, _mm512_xor_si512(x2, offset), w);
            __m512i x3 = _mm512_broadcastd_epi32(
                _mm_loadu_si32(input_q + 3 * width + group * 64 + chunk * 4));
            d3 = _mm512_dpbusd_epi32(d3, _mm512_xor_si512(x3, offset), w);
            __m512i x4 = _mm512_broadcastd_epi32(
                _mm_loadu_si32(input_q + 4 * width + group * 64 + chunk * 4));
            d4 = _mm512_dpbusd_epi32(d4, _mm512_xor_si512(x4, offset), w);
            __m512i x5 = _mm512_broadcastd_epi32(
                _mm_loadu_si32(input_q + 5 * width + group * 64 + chunk * 4));
            d5 = _mm512_dpbusd_epi32(d5, _mm512_xor_si512(x5, offset), w);
            __m512i x6 = _mm512_broadcastd_epi32(
                _mm_loadu_si32(input_q + 6 * width + group * 64 + chunk * 4));
            d6 = _mm512_dpbusd_epi32(d6, _mm512_xor_si512(x6, offset), w);
            __m512i x7 = _mm512_broadcastd_epi32(
                _mm_loadu_si32(input_q + 7 * width + group * 64 + chunk * 4));
            d7 = _mm512_dpbusd_epi32(d7, _mm512_xor_si512(x7, offset), w);
        }

        __m512 ws = _mm512_cvtph_ps(_mm256_loadu_si256(
            (const __m256i *)(weight->scales + (output_block * groups + group) * block_rows)));
        r0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(d0, correction)),
                             _mm512_mul_ps(ws, _mm512_set1_ps(input_scales[0 * groups + group])), r0);
        r1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(d1, correction)),
                             _mm512_mul_ps(ws, _mm512_set1_ps(input_scales[1 * groups + group])), r1);
        r2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(d2, correction)),
                             _mm512_mul_ps(ws, _mm512_set1_ps(input_scales[2 * groups + group])), r2);
        r3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(d3, correction)),
                             _mm512_mul_ps(ws, _mm512_set1_ps(input_scales[3 * groups + group])), r3);
        r4 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(d4, correction)),
                             _mm512_mul_ps(ws, _mm512_set1_ps(input_scales[4 * groups + group])), r4);
        r5 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(d5, correction)),
                             _mm512_mul_ps(ws, _mm512_set1_ps(input_scales[5 * groups + group])), r5);
        r6 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(d6, correction)),
                             _mm512_mul_ps(ws, _mm512_set1_ps(input_scales[6 * groups + group])), r6);
        r7 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(d7, correction)),
                             _mm512_mul_ps(ws, _mm512_set1_ps(input_scales[7 * groups + group])), r7);
    }

    _mm512_storeu_ps(output + 0 * weight->shape[0] + output_block * block_rows, r0);
    _mm512_storeu_ps(output + 1 * weight->shape[0] + output_block * block_rows, r1);
    _mm512_storeu_ps(output + 2 * weight->shape[0] + output_block * block_rows, r2);
    _mm512_storeu_ps(output + 3 * weight->shape[0] + output_block * block_rows, r3);
    _mm512_storeu_ps(output + 4 * weight->shape[0] + output_block * block_rows, r4);
    _mm512_storeu_ps(output + 5 * weight->shape[0] + output_block * block_rows, r5);
    _mm512_storeu_ps(output + 6 * weight->shape[0] + output_block * block_rows, r6);
    _mm512_storeu_ps(output + 7 * weight->shape[0] + output_block * block_rows, r7);
}

// Tail handler for VNNI: fewer than 8 active rows.
static inline __attribute__((always_inline)) void matmul_block_vnni_tail(
    float *output, const int8_t *input_q, const float *input_scales,
    const Tensor *weight, size_t row_start, size_t active_rows, size_t output_block) {
    const size_t block_rows = 16;
    const size_t width = (size_t)weight->shape[1];
    const size_t groups = width / 64;
    const int8_t *pw = (const int8_t *)weight->data + output_block * block_rows * width;
    const __m512i offset = _mm512_set1_epi8(-128);

    __m512 result[8] = {0};
    for (size_t group = 0; group < groups; group++) {
        __m512i dot[8] = {0};
        __m512i correction = _mm512_setzero_si512();
        for (int chunk = 0; chunk < 16; chunk++) {
            __m512i w = _mm512_loadu_si512(
                (const void *)(pw + group * 1024 + chunk * 64));
            correction = _mm512_dpbusd_epi32(correction, offset, w);
            for (size_t row = 0; row < active_rows; row++) {
                __m512i x = _mm512_broadcastd_epi32(
                    _mm_loadu_si32(input_q + (row_start + row) * width + group * 64 + chunk * 4));
                dot[row] = _mm512_dpbusd_epi32(dot[row], _mm512_xor_si512(x, offset), w);
            }
        }
        __m512 ws = _mm512_cvtph_ps(_mm256_loadu_si256(
            (const __m256i *)(weight->scales + (output_block * groups + group) * block_rows)));
        for (size_t row = 0; row < active_rows; row++)
            result[row] = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_sub_epi32(dot[row], correction)),
                _mm512_mul_ps(ws, _mm512_set1_ps(input_scales[(row_start + row) * groups + group])),
                result[row]);
    }
    for (size_t row = 0; row < active_rows; row++)
        _mm512_storeu_ps(output + (row_start + row) * weight->shape[0] + output_block * block_rows, result[row]);
}

// VNNI dispatch: 8-row full blocks + tail.
static inline __attribute__((always_inline)) void matmul_block(
    float *output, const int8_t *input_q, const float *input_scales,
    const Tensor *weight, size_t rows, size_t output_block) {
    size_t row_start = 0;
    for (; row_start + 8 <= rows; row_start += 8)
        matmul_block_vnni_8x16(output + row_start * weight->shape[0],
                               input_q + row_start * weight->shape[1],
                               input_scales + row_start * (weight->shape[1] / 64),
                               weight, output_block);
    if (row_start < rows)
        matmul_block_vnni_tail(output, input_q, input_scales, weight, row_start, rows - row_start, output_block);
}

#else // !__AVX512VNNI__ — AVX2 dispatch

// AVX2 dispatch: 1x16 for decode, 4x16 full + tail for prefill.
static inline __attribute__((always_inline)) void matmul_block(
    float *output, const int8_t *input_q, const float *input_scales,
    const Tensor *weight, size_t rows, size_t output_block) {
    if (rows == 1) {
        matmul_block_1x16(output, input_q, input_scales, weight, output_block);
    } else {
        size_t row_start = 0;
        for (; row_start + 4 <= rows; row_start += 4)
            matmul_block_4x16_full(output, input_q, input_scales, weight, row_start, output_block);
        if (row_start < rows)
            matmul_block_4x16_tail(output, input_q, input_scales, weight, row_start, rows - row_start, output_block);
    }
}

#endif // __AVX512VNNI__

// ============================================================================
// Public entry point
// ============================================================================

void matmul_int8(float *output, const int8_t *input_q, const float *input_scales,
                 const Tensor *weight, size_t rows) {
    const size_t block_rows = 16;
    #pragma omp for schedule(static)
    for (size_t output_block = 0; output_block < (size_t)weight->shape[0] / block_rows; output_block++)
        matmul_block(output, input_q, input_scales, weight, rows, output_block);
}

// matmul_int4 lives in kernels_avx_int4.c.

// ============================================================================
// Quantization (AVX2-vectorized)
// ============================================================================

// Converts 8 floats to int8 in one shot. Uses VROUNDPS (SSE4.1, implied by
// AVX2) for round-to-nearest-even matching rintf(), then PACKS for signed
// narrowing. The upper 4 int16 lanes of PACKS are a duplicate of the lower
// 4, but we only store the first 8 bytes (STOREL).
static inline __attribute__((always_inline))
void quantize_8(const float *src, int8_t *dst, float inverse_scale) {
    __m256 inv = _mm256_set1_ps(inverse_scale);
    __m256 f = _mm256_mul_ps(_mm256_loadu_ps(src), inv);
    __m256i q32 = _mm256_cvtps_epi32(f);
    __m128i lo = _mm256_castsi256_si128(q32);
    __m128i hi = _mm256_extracti128_si256(q32, 1);
    __m128i q16 = _mm_packs_epi32(lo, hi);
    __m128i q8 = _mm_packs_epi16(q16, q16);
    q8 = _mm_max_epi8(q8, _mm_set1_epi8(-127));
    q8 = _mm_min_epi8(q8, _mm_set1_epi8(127));
    _mm_storel_epi64((__m128i *)dst, q8);
}

// Converts each input row to int8 in groups of 64 values with a float scale
// recording each group's magnitude.
void quantize(int8_t *quantized, float *scales, const float *input, size_t rows, size_t width) {
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    #pragma omp for schedule(static)
    for (size_t group_index = 0; group_index < rows * (width / 64); group_index++) {
        const float *group = input + group_index * 64;
        int8_t *out = quantized + group_index * 64;

        // Pass 1: find max absolute value.
        __m256 m0 = _mm256_setzero_ps();
        __m256 m1 = _mm256_setzero_ps();
        for (int j = 0; j < 64; j += 16) {
            __m256 a = _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(group + j));
            __m256 b = _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(group + j + 8));
            m0 = _mm256_max_ps(m0, a);
            m1 = _mm256_max_ps(m1, b);
        }
        __m256 m = _mm256_max_ps(m0, m1);
        __m128 lo = _mm256_castps256_ps128(m);
        __m128 hi = _mm256_extractf128_ps(m, 1);
        __m128 mx = _mm_max_ps(lo, hi);
        mx = _mm_max_ps(mx, _mm_movehl_ps(mx, mx));
        float max_abs = _mm_cvtss_f32(_mm_max_ss(mx, _mm_movehdup_ps(mx)));

        float scale = max_abs / 127.0f;
        float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        scales[group_index] = scale;

        // Pass 2: quantize 64 values (8 x YMM).
        for (int j = 0; j < 64; j += 8)
            quantize_8(group + j, out + j, inv);
    }
}

// int4 variant: quantize activations in groups of 32 to match the int4 weight
// scale granularity. Same int8 output, just smaller groups.
void quantize_int4(int8_t *quantized, float *scales, const float *input, size_t rows, size_t width) {
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    #pragma omp for schedule(static)
    for (size_t group_index = 0; group_index < rows * (width / 32); group_index++) {
        const float *group = input + group_index * 32;
        int8_t *out = quantized + group_index * 32;

        // Pass 1: max abs over 32 values (4 x YMM).
        __m256 m0 = _mm256_setzero_ps();
        __m256 m1 = _mm256_setzero_ps();
        for (int j = 0; j < 32; j += 16) {
            __m256 a = _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(group + j));
            __m256 b = _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(group + j + 8));
            m0 = _mm256_max_ps(m0, a);
            m1 = _mm256_max_ps(m1, b);
        }
        __m256 m = _mm256_max_ps(m0, m1);
        __m128 lo = _mm256_castps256_ps128(m);
        __m128 hi = _mm256_extractf128_ps(m, 1);
        __m128 mx = _mm_max_ps(lo, hi);
        mx = _mm_max_ps(mx, _mm_movehl_ps(mx, mx));
        float max_abs = _mm_cvtss_f32(_mm_max_ss(mx, _mm_movehdup_ps(mx)));

        float scale = max_abs / 127.0f;
        float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        scales[group_index] = scale;

        // Pass 2: quantize 32 values (4 x YMM).
        for (int j = 0; j < 32; j += 8)
            quantize_8(group + j, out + j, inv);
    }
}

// ============================================================================
// Attention (unchanged logic, alignment asserts added)
// ============================================================================

void attention_scores(float *scores, const float *query, const float *key_cache,
        int first_key, int num_keys, int cache_mask, int head_dim) {
    assert((head_dim & 15) == 0 && "head_dim must be a multiple of 16");
    for (int key_index = 0; key_index < num_keys; key_index++) {
        const float *key = key_cache + ((first_key + key_index) & cache_mask) * head_dim;
        __m256 sum0 = _mm256_setzero_ps(), sum1 = _mm256_setzero_ps();
        for (int j = 0; j < head_dim; j += 16) {
            sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(query + j), _mm256_loadu_ps(key + j), sum0);
            sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(query + j + 8), _mm256_loadu_ps(key + j + 8), sum1);
        }
        __m256 sum8 = _mm256_add_ps(sum0, sum1);
        __m128 sum4 = _mm_add_ps(_mm256_castps256_ps128(sum8), _mm256_extractf128_ps(sum8, 1));
        sum4 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
        scores[key_index] = _mm_cvtss_f32(_mm_add_ss(sum4, _mm_movehdup_ps(sum4)));
    }
}

void weighted_value_sum(float *output, const float *probabilities, const float *value_cache,
        int first_key, int num_keys, int cache_mask, int head_dim) {
    assert((head_dim & 63) == 0 && "head_dim must be a multiple of 64");
    for (int j = 0; j < head_dim; j += 64) {
        __m256 sum[8] = {0};
        for (int key_index = 0; key_index < num_keys; key_index++) {
            const float *value = value_cache + ((first_key + key_index) & cache_mask) * head_dim + j;
            __m256 probability = _mm256_set1_ps(probabilities[key_index]);
            for (int u = 0; u < 8; u++)
                sum[u] = _mm256_fmadd_ps(probability, _mm256_loadu_ps(value + u * 8), sum[u]);
        }
        for (int u = 0; u < 8; u++) _mm256_storeu_ps(output + j + u * 8, sum[u]);
    }
}

// ============================================================================
// GELU (unchanged)
// ============================================================================

// Approximates GELU from the exported lookup table and multiplies it by the up
// projection to produce the MLP's gated activation.
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
