// AVX2 / AVX-512 VNNI kernel implementations.
// Only compiled when the target supports AVX2 (see Makefile).
// For portable scalar fallback, see kernels_pure.c.

#if !defined(__AVX2__)
#error "kernels.c requires -mavx2 (or -march=native on an AVX2 CPU). Use kernels_pure.c for scalar targets."
#endif

#include "gemma4.h"

// Multiplies dynamically quantized activations by packed int8 weights in blocks of 16
// output rows. VNNI handles eight input rows at once while AVX2 handles four. Both
// accumulate integer dot products before restoring float values with their scales.
#if defined(__AVX512VNNI__)
static inline __attribute__((always_inline)) void matmul_block(
    float *output, const int8_t *input_q, const float *input_scales,
    const Tensor *weight, size_t rows, size_t output_block) {
    const size_t block_rows = 16;
    size_t groups_per_row = (size_t)weight->shape[1] / 64;
    const int8_t *packed_weights = (const int8_t *)weight->data + output_block * block_rows * weight->shape[1];
    for (size_t row_start = 0; row_start < rows; row_start += 8) {
        size_t active_rows = rows - row_start < 8 ? rows - row_start : 8;
        __m512 result[8] = {0};
        for (size_t group = 0; group < groups_per_row; group++) {
            __m512i dot[8] = {0};
            __m512i correction = _mm512_setzero_si512();
            __m512i offset_bytes = _mm512_set1_epi8(-128); // Flip signed activations into the unsigned range required by VNNI.
            for (int chunk = 0; chunk < 16; chunk++) {
                __m512i weight_values = _mm512_load_si512((const __m512i *)(packed_weights + group * 1024 + chunk * 64));
                correction = _mm512_dpbusd_epi32(correction, offset_bytes, weight_values);
                for (size_t row = 0; row < active_rows; row++) {
                    __m512i input_values = _mm512_broadcastd_epi32(
                        _mm_loadu_si32(input_q + (row_start + row) * weight->shape[1] + group * 64 + chunk * 4));
                    input_values = _mm512_xor_si512(input_values, offset_bytes);
                    dot[row] = _mm512_dpbusd_epi32(dot[row], input_values, weight_values);
                }
            }
            __m512 weight_scales = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(weight->scales + (output_block * groups_per_row + group) * block_rows)));
            for (size_t row = 0; row < active_rows; row++)
                result[row] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(dot[row], correction)), _mm512_mul_ps(weight_scales, _mm512_set1_ps(input_scales[(row_start + row) * groups_per_row + group])), result[row]);
        }
        for (size_t row = 0; row < active_rows; row++)
            _mm512_storeu_ps(output + (row_start + row) * weight->shape[0] + output_block * block_rows, result[row]);
    }
}
#else
static inline __attribute__((always_inline)) void matmul_block(
    float *output, const int8_t *input_q, const float *input_scales,
    const Tensor *weight, size_t rows, size_t output_block) {
    const size_t block_rows = 16;
    size_t groups_per_row = (size_t)weight->shape[1] / 64;
    const int8_t *packed_weights = (const int8_t *)weight->data + output_block * block_rows * weight->shape[1];
    for (size_t row_start = 0; row_start < rows; row_start += 4) {
        size_t active_rows = rows - row_start < 4 ? rows - row_start : 4;
        for (int half = 0; half < 2; half++) {
            __m256 result[4] = {0};
            for (size_t group = 0; group < groups_per_row; group++) {
                __m256i dot[4] = {0};
                for (int chunk = 0; chunk < 16; chunk++) {
                    __m256i weight_values = _mm256_loadu_si256((const __m256i *)(packed_weights + group * 1024 + chunk * 64 + half * 32));
                    __m256i weight_magnitudes = _mm256_abs_epi8(weight_values);
                    for (size_t row = 0; row < active_rows; row++) {
                        __m256i input_values = _mm256_broadcastd_epi32(
                            _mm_loadu_si32(input_q + (row_start + row) * weight->shape[1] + group * 64 + chunk * 4));
                        dot[row] = _mm256_add_epi32(dot[row], _mm256_madd_epi16(_mm256_maddubs_epi16(weight_magnitudes, _mm256_sign_epi8(input_values, weight_values)), _mm256_set1_epi16(1)));
                    }
                }
                __m256 weight_scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(weight->scales + (output_block * groups_per_row + group) * block_rows + half * 8)));
                for (size_t row = 0; row < active_rows; row++)
                    result[row] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot[row]), _mm256_mul_ps(weight_scales, _mm256_set1_ps(input_scales[(row_start + row) * groups_per_row + group])), result[row]);
            }
            for (size_t row = 0; row < active_rows; row++)
                _mm256_storeu_ps(output + (row_start + row) * weight->shape[0] + output_block * block_rows + half * 8, result[row]);
        }
    }
}
#endif

void matmul_int8(float *output, const int8_t *input_q, const float *input_scales,
                 const Tensor *weight, size_t rows) {
    const size_t block_rows = 16;
    #pragma omp for schedule(static)
    for (size_t output_block = 0; output_block < (size_t)weight->shape[0] / block_rows; output_block++)
        matmul_block(output, input_q, input_scales, weight, rows, output_block);
}

// int4 weights against int8 activations. The AVX2 path does not yet have a
// SIMD int4 kernel, so this uses the portable scalar reference (same layout
// and math as kernels_pure.c). Correct but slower than the NEON int4 path.
static inline float fp16_to_f32_k(const uint16_t value) {
    int sign = value >> 15;
    int exponent = (value >> 10) & 31;
    int fraction = value & 1023;
    float result;
    if (exponent == 0) result = ldexpf((float)fraction, -24);
    else if (exponent == 31) result = fraction ? NAN : INFINITY;
    else result = ldexpf((float)(1024 + fraction), exponent - 25);
    return sign ? -result : result;
}

void matmul_int4(float *output, const int8_t *input, const float *input_scales,
                 const Tensor *weight, size_t rows) {
    const int block_rows = 16;
    int outputs = weight->shape[0];
    int inputs = weight->shape[1];
    int groups = inputs / 32;
    const uint8_t *data = (const uint8_t *)weight->data;
    #pragma omp for collapse(2) schedule(static)
    for (size_t row = 0; row < rows; row++) {
        for (int j = 0; j < outputs; j++) {
            int block = j / block_rows;
            int r = j % block_rows;
            float sum = 0.0f;
            for (int group = 0; group < groups; group++) {
                int dot = 0;
                // Block stride is 16 rows x (inputs/2) bytes (int4 packs 2/byte).
                const uint8_t *wg = data + (size_t)block * block_rows * (inputs / 2)
                                    + (size_t)group * block_rows * 16 + r * 16;
                for (int k = 0; k < 32; k++) {
                    int input_index = group * 32 + k;
                    uint8_t packed = wg[k / 2];
                    int nibble = (packed >> ((k % 2) * 4)) & 0xf;
                    dot += (int)input[row * inputs + input_index] * (nibble - 8);
                }
                float scale = fp16_to_f32_k(weight->scales[((size_t)block * groups + group) * 16 + r]);
                sum += (float)dot * input_scales[row * groups + group] * scale;
            }
            output[row * outputs + j] = sum;
        }
    }
}

// Converts each input row to int8 in groups of 64 values with a float scale recording each group's magnitude.
void quantize(int8_t *quantized, float *scales, const float *input, size_t rows, size_t width) {
    #pragma omp for schedule(static)
    for (size_t group_index = 0; group_index < rows * (width / 64); group_index++) {
        const float *group = input + group_index * 64;
        float max_abs = 0.0f;
        for (int j = 0; j < 64; j++) {
            float value = fabsf(group[j]);
            if (value > max_abs) max_abs = value;
        }
        float scale = max_abs / 127.0f;
        float inverse_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
        for (int j = 0; j < 64; j++) quantized[group_index * 64 + j] = (int8_t)rintf(group[j] * inverse_scale);
        scales[group_index] = scale;
    }
}

// int4 variant: quantize activations in groups of 32 to match the int4 weight scale granularity.
void quantize_int4(int8_t *quantized, float *scales, const float *input, size_t rows, size_t width) {
    #pragma omp for schedule(static)
    for (size_t group_index = 0; group_index < rows * (width / 32); group_index++) {
        const float *group = input + group_index * 32;
        float max_abs = 0.0f;
        for (int j = 0; j < 32; j++) {
            float value = fabsf(group[j]);
            if (value > max_abs) max_abs = value;
        }
        float scale = max_abs / 127.0f;
        float inverse_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
        for (int j = 0; j < 32; j++) quantized[group_index * 32 + j] = (int8_t)rintf(group[j] * inverse_scale);
        scales[group_index] = scale;
    }
}

void attention_scores(float *scores, const float *query, const float *key_cache,
        int first_key, int num_keys, int cache_mask, int head_dim) {
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
