// Pure C (scalar) int8 kernel implementations for gemma4.c.
//
// No SIMD intrinsics, no architecture-specific code.
// Designed for portability to any platform with a C11 compiler.
//
// In the main build this is selected by the Makefile when AVX2 is unavailable.
// Provides: quantize, quantize_int4, matmul_int8, attention_scores,
//           weighted_value_sum, geglu
//
// The int4 matmul lives in kernels_pure_int4.c.
// The other kernels (embedding, rmsnorm, add_and_scale, apply_rope, softmax)
// live in transformer.c and are already portable.

#include "gemma4.h"

// ----------------------------------------------------------------------------
// fp16 conversion (also used by matmul_int8 via weight_scale)

float fp16_to_float(uint16_t value) {
    int sign = value >> 15;
    int exponent = (value >> 10) & 31;
    int fraction = value & 1023;
    float result;
    if (exponent == 0) result = ldexpf((float)fraction, -24);
    else if (exponent == 31) result = fraction ? NAN : INFINITY;
    else result = ldexpf((float)(1024 + fraction), exponent - 25);
    return sign ? -result : result;
}

// ----------------------------------------------------------------------------
// Packed weight access

int8_t packed_weight(const Tensor *weight, int output, int input) {
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

float weight_scale(const Tensor *weight, int output, int input_group) {
    int groups = weight->shape[1] / 64;
    int block = output / 16;
    int row = output % 16;
    return fp16_to_float(weight->scales[((size_t)block * groups + input_group) * 16 + row]);
}

// ----------------------------------------------------------------------------
// Quantization

void quantize(int8_t *output, float *scales, const float *input, size_t rows, size_t width) {
    #pragma omp for schedule(static)
    for (size_t group = 0; group < rows * (width / 64); group++) {
        float max_abs = 0.0f;
        for (int k = 0; k < 64; k++) {
            float value = fabsf(input[group * 64 + k]);
            if (value > max_abs) max_abs = value;
        }
        float scale = max_abs / 127.0f;
        float inverse_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
        for (int k = 0; k < 64; k++) {
            output[group * 64 + k] = (int8_t)rintf(input[group * 64 + k] * inverse_scale);
        }
        scales[group] = scale;
    }
}

// int4 variant: quantize activations in groups of 32 to match the int4 weight scale granularity.
void quantize_int4(int8_t *output, float *scales, const float *input, size_t rows, size_t width) {
    #pragma omp for schedule(static)
    for (size_t group = 0; group < rows * (width / 32); group++) {
        float max_abs = 0.0f;
        for (int k = 0; k < 32; k++) {
            float value = fabsf(input[group * 32 + k]);
            if (value > max_abs) max_abs = value;
        }
        float scale = max_abs / 127.0f;
        float inverse_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
        for (int k = 0; k < 32; k++) {
            output[group * 32 + k] = (int8_t)rintf(input[group * 32 + k] * inverse_scale);
        }
        scales[group] = scale;
    }
}

// ----------------------------------------------------------------------------
// int8 matrix multiply

void matmul_int8(float *output, const int8_t *input, const float *input_scales,
                 const Tensor *weight, size_t rows) {
    int outputs = weight->shape[0];
    int inputs = weight->shape[1];
    int groups = inputs / 64;
    #pragma omp for collapse(2) schedule(static)
    for (size_t row = 0; row < rows; row++) {
        for (int j = 0; j < outputs; j++) {
            float sum = 0.0f;
            for (int group = 0; group < groups; group++) {
                int dot = 0;
                for (int k = 0; k < 64; k++) {
                    int input_index = group * 64 + k;
                    dot += (int)input[row * inputs + input_index]
                           * (int)packed_weight(weight, j, input_index);
                }
                sum += (float)dot * input_scales[row * groups + group]
                       * weight_scale(weight, j, group);
            }
            output[row * outputs + j] = sum;
        }
    }
}

// ----------------------------------------------------------------------------
// Gated GELU (table-interpolated)

void geglu(float *gate, const float *up, int rows, int width, int up_stride,
           const Tensor *gelu_table) {
    const float *table = (const float *)gelu_table->data;
    int table_size = gelu_table->shape[0];
    float lower = (float)gelu_table->shape[1];
    float upper = (float)gelu_table->shape[2];
    float scale = (float)(table_size - 1) / (upper - lower);
    #pragma omp for collapse(2) schedule(static)
    for (int row = 0; row < rows; row++) {
        for (int i = 0; i < width; i++) {
            float x = gate[row * width + i];
            if (x <= lower) x = table[0];
            else if (x < upper) {
                float position = (x - lower) * scale;
                int index = (int)position;
                float fraction = position - (float)index;
                x = table[index] + fraction * (table[index + 1] - table[index]);
            }
            gate[row * width + i] = x * up[row * up_stride + i];
        }
    }
}

// ----------------------------------------------------------------------------
// Attention: Q·K^T scores

void attention_scores(float *scores, const float *query, const float *key_cache,
                      int first_key, int num_keys, int cache_mask, int head_dim) {
    for (int key_index = 0; key_index < num_keys; key_index++) {
        const float *key = key_cache + ((first_key + key_index) & cache_mask) * head_dim;
        float dot = 0.0f;
        for (int j = 0; j < head_dim; j++) dot += query[j] * key[j];
        scores[key_index] = dot;
    }
}

// ----------------------------------------------------------------------------
// Attention: weighted value sum

void weighted_value_sum(float *output, const float *probabilities, const float *value_cache,
                        int first_key, int num_keys, int cache_mask, int head_dim) {
    for (int j = 0; j < head_dim; j++) {
        float sum = 0.0f;
        for (int key_index = 0; key_index < num_keys; key_index++) {
            const float *value = value_cache
                + ((first_key + key_index) & cache_mask) * head_dim;
            sum += probabilities[key_index] * value[j];
        }
        output[j] = sum;
    }
}
