// AVX2 int4 weight kernels for gemma4.c.
//
// Weights are packed two 4-bit values per byte:
//   data[block*16*(width/2) + group*16*16 + row*16 + byte]
// Low nibble = even input, high nibble = odd input. 8 is the zero point.
// One fp16 scale per (group, row): scales[(block*groups + group)*16 + row].
//
// Strategy: for each 16-byte weight row, extract the 16 low nibbles (even
// inputs) and 16 high nibbles (odd inputs) as signed int8 vectors. Then
// compute the dot product as two 16-element dot products against
// deinterleaved input halves. Uses maddubs/madd_epi16 for the MACs.
//
// The microkernel processes two output rows at a time: one 256-bit load
// fetches weight rows r and r+1, and the input vectors are broadcast into
// both 128-bit halves of the YMM. The even/odd contributions are summed
// before the horizontal reduction, and abs(input) is computed once per
// group instead of once per output row.
//
// This file provides matmul_int4 only. All other kernels come from kernels_avx_int8.c.

#if !defined(__AVX2__)
#error "kernels_avx_int4.c requires -mavx2 (or -march=native on an AVX2 CPU)."
#endif

#include "gemma4.h"

// ----------------------------------------------------------------------------
// fp16 → float

static inline __attribute__((always_inline))
float fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            int p = 31 - __builtin_clz(mant);
            bits = sign | ((uint32_t)(p + 103) << 23) | ((mant << (23 - p)) & 0x007fffffu);
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
// Deinterleave 32 sequential int8 values into even/odd 16-element vectors.
//   input: [x0, x1, x2, ..., x31]
//   even:  [x0, x2, x4, ..., x30]
//   odd:   [x1, x3, x5, ..., x31]

static inline __attribute__((always_inline))
void deinterleave_32(const int8_t *in, __m128i *even, __m128i *odd) {
    __m128i lo = _mm_loadu_si128((const __m128i *)(in));
    __m128i hi = _mm_loadu_si128((const __m128i *)(in + 16));
    // pshufb: mask byte 0x80+idx selects byte idx; 0x80 alone zeroes the lane.
    __m128i me = _mm_setr_epi8(0,2,4,6,8,10,12,14,
                               (char)0x80,(char)0x80,(char)0x80,(char)0x80,
                               (char)0x80,(char)0x80,(char)0x80,(char)0x80);
    __m128i mo = _mm_setr_epi8(1,3,5,7,9,11,13,15,
                               (char)0x80,(char)0x80,(char)0x80,(char)0x80,
                               (char)0x80,(char)0x80,(char)0x80,(char)0x80);
    __m128i lo_e = _mm_shuffle_epi8(lo, me); // [x0,x2,...,x14, 0,...]
    __m128i hi_e = _mm_shuffle_epi8(hi, me); // [x16,x18,...,x30, 0,...]
    *even = _mm_unpacklo_epi64(lo_e, hi_e);  // [x0..x14, x16..x30]
    __m128i lo_o = _mm_shuffle_epi8(lo, mo); // [x1,x3,...,x15, 0,...]
    __m128i hi_o = _mm_shuffle_epi8(hi, mo); // [x17,x19,...,x31, 0,...]
    *odd = _mm_unpacklo_epi64(lo_o, hi_o);   // [x1..x15, x17..x31]
}

// ----------------------------------------------------------------------------
// Unpack two consecutive 16-byte weight rows (32 bytes) into even/odd
// 256-bit signed int8 vectors. Low 128 bits = row r, high 128 bits = row r+1.
//   even = [w0, w2, ..., w30] per row  (low nibbles - 8)
//   odd  = [w1, w3, ..., w31] per row  (high nibbles - 8)

static inline __attribute__((always_inline))
void unpack_rows_2x16(const uint8_t *p, __m256i *even, __m256i *odd) {
    __m256i packed = _mm256_loadu_si256((const __m256i *)p);
    const __m256i mask = _mm256_set1_epi8(0x0f);
    const __m256i eight = _mm256_set1_epi8(8);
    __m256i lo = _mm256_and_si256(packed, mask);
    __m256i hi = _mm256_and_si256(
        _mm256_srli_epi16(packed, 4), mask);
    *even = _mm256_sub_epi8(lo, eight);
    *odd  = _mm256_sub_epi8(hi, eight);
}

// ----------------------------------------------------------------------------
// Dot product of two output rows at once:
//   low 128 bits  = dot for row r
//   high 128 bits = dot for row r+1
// The even and odd contributions are summed before the horizontal reduction,
// and the input magnitudes (aie/aio) are precomputed by the caller.

static inline __attribute__((always_inline))
__m256i dot_32_2rows(
    __m256i ie, __m256i io,
    __m256i aie, __m256i aio,
    __m256i we, __m256i wo) {
    __m256i swe = _mm256_sign_epi8(we, ie);
    __m256i swo = _mm256_sign_epi8(wo, io);

    __m256i pe = _mm256_maddubs_epi16(aie, swe);
    __m256i po = _mm256_maddubs_epi16(aio, swo);

    const __m256i ones = _mm256_set1_epi16(1);

    __m256i se = _mm256_madd_epi16(pe, ones);
    __m256i so = _mm256_madd_epi16(po, ones);

    // Sum even + odd before the horizontal reduction. Each 128-bit half
    // stays independent, so the two output rows reduce separately.
    __m256i sum = _mm256_add_epi32(se, so);
    sum = _mm256_hadd_epi32(sum, sum);
    sum = _mm256_hadd_epi32(sum, sum);
    return sum;
}

// ----------------------------------------------------------------------------
// GEMV: 1 input row × 16 output rows (decode).

static inline __attribute__((always_inline)) void matmul_int4_block_1x16(
    float *output, const int8_t *input_q, const float *input_scales,
    const Tensor *weight, size_t output_block) {
    const int width = weight->shape[1];
    const int groups = width / 32;
    const uint8_t *pw = (const uint8_t *)weight->data
                       + output_block * 16 * (width / 2);

    float scale_cache[groups * 16];
    for (int g = 0; g < groups; g++) {
        const uint16_t *ws = weight->scales + ((output_block * groups + g) * 16);
        for (int r = 0; r < 16; r++)
            scale_cache[g * 16 + r] = fp16_to_f32(ws[r]);
    }

    float result[16] = {0};

    for (int g = 0; g < groups; g++) {
        const int8_t *in = input_q + g * 32;
        const uint8_t *wg = pw + (size_t)g * 256;

        if (g + 1 < groups)
            __builtin_prefetch(pw + (size_t)(g + 1) * 256, 0, 1);

        __m128i ie, io;
        deinterleave_32(in, &ie, &io);

        // Broadcast the input into both 128-bit halves of the YMM so one
        // microkernel iteration handles two output rows.
        __m256i IE  = _mm256_broadcastsi128_si256(ie);
        __m256i IO  = _mm256_broadcastsi128_si256(io);
        __m256i AIE = _mm256_broadcastsi128_si256(_mm_abs_epi8(ie));
        __m256i AIO = _mm256_broadcastsi128_si256(_mm_abs_epi8(io));

        for (int r = 0; r < 16; r += 2) {
            __m256i we, wo;
            unpack_rows_2x16(wg + r * 16, &we, &wo);

            __m256i d = dot_32_2rows(IE, IO, AIE, AIO, we, wo);

            int32_t dot0 = _mm_cvtsi128_si32(_mm256_castsi256_si128(d));
            int32_t dot1 = _mm_cvtsi128_si32(_mm256_extracti128_si256(d, 1));

            result[r]     += (float)dot0 * input_scales[g] * scale_cache[g * 16 + r];
            result[r + 1] += (float)dot1 * input_scales[g] * scale_cache[g * 16 + r + 1];
        }
    }

    for (int r = 0; r < 16; r++)
        output[output_block * 16 + r] = result[r];
}

// ----------------------------------------------------------------------------
// GEMM: 2 input rows × 16 output rows (prefill).

static inline __attribute__((always_inline)) void matmul_int4_block_2x16(
    float *output, const int8_t *input_q, const float *input_scales,
    const Tensor *weight, size_t rows, size_t output_block) {
    const int width = weight->shape[1];
    const int groups = width / 32;
    const uint8_t *pw = (const uint8_t *)weight->data
                       + output_block * 16 * (width / 2);

    float scale_cache[groups * 16];
    for (int g = 0; g < groups; g++) {
        const uint16_t *ws = weight->scales + ((output_block * groups + g) * 16);
        for (int r = 0; r < 16; r++)
            scale_cache[g * 16 + r] = fp16_to_f32(ws[r]);
    }

    for (size_t row0 = 0; row0 < rows; row0 += 2) {
        const int8_t *in0 = input_q + row0 * width;
        const float *sc0 = input_scales + row0 * groups;
        int has1 = (row0 + 1 < rows);
        const int8_t *in1 = has1 ? input_q + (row0 + 1) * width : in0;
        const float *sc1 = has1 ? input_scales + (row0 + 1) * groups : sc0;

        float r0[16] = {0}, r1[16] = {0};

        for (int g = 0; g < groups; g++) {
            const int8_t *g0 = in0 + g * 32;
            const int8_t *g1 = in1 + g * 32;
            const uint8_t *wg = pw + (size_t)g * 256;

            if (g + 1 < groups)
                __builtin_prefetch(pw + (size_t)(g + 1) * 256, 0, 1);

            __m128i e0, o0, e1, o1;
            deinterleave_32(g0, &e0, &o0);
            if (has1) deinterleave_32(g1, &e1, &o1);

            // Broadcast each input row into a YMM (two output rows at a time).
            __m256i IE0 = _mm256_broadcastsi128_si256(e0);
            __m256i IO0 = _mm256_broadcastsi128_si256(o0);
            __m256i AIE0 = _mm256_broadcastsi128_si256(_mm_abs_epi8(e0));
            __m256i AIO0 = _mm256_broadcastsi128_si256(_mm_abs_epi8(o0));

            for (int r = 0; r + 1 < 16; r += 2) {
                __m256i we, wo;
                unpack_rows_2x16(wg + r * 16, &we, &wo);

                __m256i d = dot_32_2rows(IE0, IO0, AIE0, AIO0, we, wo);

                int32_t dot0 = _mm_cvtsi128_si32(_mm256_castsi256_si128(d));
                int32_t dot1 = _mm_cvtsi128_si32(_mm256_extracti128_si256(d, 1));

                r0[r]     += (float)dot0 * sc0[g] * scale_cache[g * 16 + r];
                r0[r + 1] += (float)dot1 * sc0[g] * scale_cache[g * 16 + r + 1];

                if (has1) {
                    __m256i IE1 = _mm256_broadcastsi128_si256(e1);
                    __m256i IO1 = _mm256_broadcastsi128_si256(o1);
                    __m256i AIE1 = _mm256_broadcastsi128_si256(_mm_abs_epi8(e1));
                    __m256i AIO1 = _mm256_broadcastsi128_si256(_mm_abs_epi8(o1));

                    __m256i d1 = dot_32_2rows(IE1, IO1, AIE1, AIO1, we, wo);

                    int32_t dd0 = _mm_cvtsi128_si32(_mm256_castsi256_si128(d1));
                    int32_t dd1 = _mm_cvtsi128_si32(_mm256_extracti128_si256(d1, 1));

                    r1[r]     += (float)dd0 * sc1[g] * scale_cache[g * 16 + r];
                    r1[r + 1] += (float)dd1 * sc1[g] * scale_cache[g * 16 + r + 1];
                }
            }

        }

        float *dst0 = output + row0 * weight->shape[0] + output_block * 16;
        for (int r = 0; r < 16; r++) dst0[r] = r0[r];
        if (has1) {
            float *dst1 = output + (row0 + 1) * weight->shape[0] + output_block * 16;
            for (int r = 0; r < 16; r++) dst1[r] = r1[r];
        }
    }
}

// ----------------------------------------------------------------------------
// Entry point.

void matmul_int4(float *output, const int8_t *input_q, const float *input_scales,
                 const Tensor *weight, size_t rows) {
    #pragma omp for schedule(static)
    for (size_t ob = 0; ob < (size_t)weight->shape[0] / 16; ob++) {
        if (rows == 1)
            matmul_int4_block_1x16(output, input_q, input_scales, weight, ob);
        else
            matmul_int4_block_2x16(output, input_q, input_scales, weight, rows, ob);
    }
}
