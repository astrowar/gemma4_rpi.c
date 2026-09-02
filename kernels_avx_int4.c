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
// This file provides matmul_int4 only. All other kernels come from kernels.c.

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
// Unpack 16 bytes of int4 into two 128-bit signed int8 vectors.
//   even = [w0, w2, w4, ..., w30]  (low nibbles - 8)
//   odd  = [w1, w3, w5, ..., w31]  (high nibbles - 8)

static inline __attribute__((always_inline))
void unpack_row_16(const uint8_t *p, __m128i *even, __m128i *odd) {
    __m128i packed = _mm_loadu_si128((const __m128i *)p);
    __m128i lo = _mm_and_si128(packed, _mm_set1_epi8(0x0f));
    // High nibbles: mask upper 4 bits of each byte, then shift 16-bit lanes
    // right by 4. This packs [hi0, hi1, ..., hi15] densely.
    __m128i hi = _mm_srli_epi16(_mm_and_si128(packed, _mm_set1_epi8((int)0xf0)), 4);
    __m128i eight = _mm_set1_epi8(8);
    *even = _mm_sub_epi8(lo, eight);
    *odd  = _mm_sub_epi8(hi, eight);
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
// Dot product: 16 int8 × 16 int8 → int32.

static inline __attribute__((always_inline))
int32_t dot_16(const __m128i x, const __m128i w) {
    // |x| as unsigned × sign(w,x) as signed = x*w.
    // Handles x=-128 correctly (PABSB gives 0x80, PMADDUBSW reads as 128).
    __m128i ax = _mm_abs_epi8(x);
    __m128i sw = _mm_sign_epi8(w, x);
    __m128i p16 = _mm_maddubs_epi16(ax, sw);
    __m128i p32 = _mm_madd_epi16(p16, _mm_set1_epi16(1));
    p32 = _mm_hadd_epi32(p32, p32);
    p32 = _mm_hadd_epi32(p32, p32);
    return _mm_cvtsi128_si32(p32);
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

        for (int r = 0; r < 16; r++) {
            __m128i we, wo;
            unpack_row_16(wg + r * 16, &we, &wo);
            int32_t dot = dot_16(ie, we) + dot_16(io, wo);
            result[r] += (float)dot * input_scales[g] * scale_cache[g * 16 + r];
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

            for (int r = 0; r < 16; r++) {
                __m128i we, wo;
                unpack_row_16(wg + r * 16, &we, &wo);
                r0[r] += (float)(dot_16(e0, we) + dot_16(o0, wo))
                        * sc0[g] * scale_cache[g * 16 + r];
                if (has1)
                    r1[r] += (float)(dot_16(e1, we) + dot_16(o1, wo))
                            * sc1[g] * scale_cache[g * 16 + r];
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
