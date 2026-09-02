// Pure C (scalar) int4 kernel implementations for gemma4.c.
//
// No SIMD intrinsics, no architecture-specific code.
// Provides: matmul_int4
//
// Weights are packed two 4-bit values per byte in the layout
//   data[block*16*(width/2) + group*16*16 + row*16 + byte]
// where byte holds input 2*byte (low nibble) and 2*byte+1 (high nibble), and
// one fp16 scale is stored per (group, row) at
//   scales[(block*groups + group)*16 + row].
// Nibbles are stored unsigned (0..15); 8 is the zero point, so the signed
// weight value is (nibble - 8).

#include "gemma4.h"

// ----------------------------------------------------------------------------
// fp16 → float (same as kernels_pure_int8.c)

static inline float fp16_to_float(uint16_t value) {
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
// Packed int4 weight access

static inline int int4_weight(const Tensor *weight, int output, int input) {
    const int block_rows = 16;
    int width = weight->shape[1];
    int block = output / block_rows;
    int row = output % block_rows;
    int group = input / 32;
    int byte = input % 32 / 2;
    int bit = (input % 2) * 4;
    const uint8_t *data = (const uint8_t *)weight->data;
    // Layout: data[block*16*(width/2) + group*16*16 + row*16 + byte].
    // Block stride is 16 rows x (width/2) bytes (int4 packs 2 weights/byte).
    // Within a block, each row's group is 16 bytes (32 nibbles), so the group
    // stride is 16*16 = 256 bytes (16 rows x 16 bytes).
    uint8_t packed = data[(size_t)block * block_rows * (width / 2)
                          + (size_t)group * block_rows * 16
                          + row * 16 + byte];
    return (int)((packed >> bit) & 0xf) - 8;
}

static inline float int4_weight_scale(const Tensor *weight, int output, int input_group) {
    int groups = weight->shape[1] / 32;
    int block = output / 16;
    int row = output % 16;
    return fp16_to_float(weight->scales[((size_t)block * groups + input_group) * 16 + row]);
}

// ----------------------------------------------------------------------------
// int4 matrix multiply (scalar reference)

void matmul_int4(float *output, const int8_t *input, const float *input_scales,
                 const Tensor *weight, size_t rows) {
    int outputs = weight->shape[0];
    int inputs = weight->shape[1];
    int groups = inputs / 32;
    #pragma omp for collapse(2) schedule(static)
    for (size_t row = 0; row < rows; row++) {
        for (int j = 0; j < outputs; j++) {
            float sum = 0.0f;
            for (int group = 0; group < groups; group++) {
                int dot = 0;
                for (int k = 0; k < 32; k++) {
                    int input_index = group * 32 + k;
                    dot += (int)input[row * inputs + input_index] * int4_weight(weight, j, input_index);
                }
                sum += (float)dot * input_scales[row * groups + group]
                       * int4_weight_scale(weight, j, group);
            }
            output[row * outputs + j] = sum;
        }
    }
}
