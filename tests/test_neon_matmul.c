// Standalone correctness test for the NEON int8 matmul against the scalar reference.
// Builds synthetic packed weights + scales and random quantized inputs, runs both
// matmul_int8 implementations, and reports the max absolute difference.
//
// Build (aarch64):
//   cc -std=c11 -O2 -Wall -Wextra -fopenmp test_neon_matmul.c kernels_neon_int8.c -o test_neon_matmul -lm
// Run:
//   ./test_neon_matmul

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gemma4.h"

// The NEON matmul (from kernels_neon_int8.c).
extern void matmul_int8(float *output, const int8_t *input_q, const float *input_scales,
                        const Tensor *weight, size_t rows);

// ----------------------------------------------------------------------------
// Scalar reference (copied from kernels_pure_int8.c)

static float fp16_to_f32(uint16_t value) {
    int sign = value >> 15;
    int exponent = (value >> 10) & 31;
    int fraction = value & 1023;
    float result;
    if (exponent == 0) result = ldexpf((float)fraction, -24);
    else if (exponent == 31) result = fraction ? NAN : INFINITY;
    else result = ldexpf((float)(1024 + fraction), exponent - 25);
    return sign ? -result : result;
}

static int8_t packed_weight(const Tensor *weight, int output, int input) {
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

static float weight_scale(const Tensor *weight, int output, int input_group) {
    int groups = weight->shape[1] / 64;
    int block = output / 16;
    int row = output % 16;
    return fp16_to_f32(weight->scales[((size_t)block * groups + input_group) * 16 + row]);
}

static void matmul_int8_ref(float *output, const int8_t *input, const float *input_scales,
                            const Tensor *weight, size_t rows) {
    int outputs = weight->shape[0];
    int inputs = weight->shape[1];
    int groups = inputs / 64;
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

// Deterministic pseudo-random in [-127, 127].
static int8_t rand_int8(unsigned *state) {
    *state = *state * 1103515245u + 12345u;
    return (int8_t)(((*state >> 16) & 0xff) - 128 + 1); // [-127, 127]
}

// Build a float16 bit pattern for a given float (round-to-nearest, no denormals needed here).
static uint16_t float_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exponent = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mantissa = x & 0x7fffff;
    if (exponent <= 0) return (uint16_t)sign; // flush to signed zero (scales are small)
    if (exponent >= 31) return (uint16_t)(sign | 0x7c00);
    uint32_t rounded = mantissa + 0x0fff + ((mantissa >> 13) & 1);
    uint16_t h = (uint16_t)(sign | ((uint32_t)exponent << 10) | (rounded >> 13));
    return h;
}

int main(void) {
    // Shapes matching gemma4.c linear layers: outputs=1536 (HIDDEN), inputs=1536.
    const int outputs = 1536;
    const int inputs = 1536;
    const int groups = inputs / 64;
    const size_t rows = 4; // a few tokens

    size_t weight_bytes = (size_t)outputs * inputs;
    size_t scale_count = (size_t)(outputs / 16) * groups * 16;

    int8_t *wdata = malloc(weight_bytes);
    uint16_t *wscales = malloc(scale_count * sizeof(uint16_t));
    int8_t *input_q = malloc(rows * inputs);
    float *input_scales = malloc(rows * groups * sizeof(float));
    float *out_neon = malloc(rows * outputs * sizeof(float));
    float *out_ref = malloc(rows * outputs * sizeof(float));

    unsigned state = 12345;
    for (size_t i = 0; i < weight_bytes; i++) wdata[i] = rand_int8(&state);
    for (size_t i = 0; i < scale_count; i++) {
        // Scales in a realistic range: 0.001 .. 1.0
        float s = 0.001f + ((float)((state = state * 1103515245u + 12345u, (state >> 16) & 0xffff)) / 65535.0f);
        wscales[i] = float_to_fp16(s);
    }
    for (size_t i = 0; i < rows * inputs; i++) input_q[i] = rand_int8(&state);
    for (size_t i = 0; i < rows * groups; i++)
        input_scales[i] = 0.01f + ((float)((state = state * 1103515245u + 12345u, (state >> 16) & 0xffff)) / 65535.0f);

    Tensor weight = { .data = wdata, .scales = wscales, .shape = {outputs, inputs, 0, 0} };

    matmul_int8(out_neon, input_q, input_scales, &weight, rows);
    matmul_int8_ref(out_ref, input_q, input_scales, &weight, rows);

    float max_diff = 0.0f;
    size_t worst = 0;
    for (size_t i = 0; i < rows * outputs; i++) {
        float d = fabsf(out_neon[i] - out_ref[i]);
        if (d > max_diff) { max_diff = d; worst = i; }
    }
    printf("rows=%zu outputs=%d inputs=%d\n", rows, outputs, inputs);
    printf("max_abs_diff = %.9f  (at [%zu]: neon=%.6f ref=%.6f)\n",
           max_diff, worst, out_neon[worst], out_ref[worst]);
    // Relative check: values are O(1000), so a tolerance of 1e-2 absolute is generous.
    if (max_diff < 1e-2f) {
        printf("PASS: NEON matches scalar reference.\n");
        return 0;
    }
    printf("FAIL: NEON diverges from scalar reference.\n");
    // Dump a few entries to help debug.
    for (size_t i = 0; i < 8; i++)
        printf("  [%zu] neon=%.6f ref=%.6f\n", i, out_neon[i], out_ref[i]);
    return 1;
}
