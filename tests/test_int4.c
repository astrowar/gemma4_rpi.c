// Standalone correctness test for the int4 matmul kernels.
//
// Builds a synthetic int4 weight tensor (packed two 4-bit values per byte,
// one fp16 scale per 32 inputs) plus random int8 activations, then compares
// the kernel under test against a direct float reference that dequantizes the
// exact same packed weights.
//
// The same source is compiled against either kernels_pure_int4.c (scalar reference)
// or kernels_neon_int4.c (NEON) via the Makefile, so one test validates both.
//
// Build (aarch64, NEON):
//   cc -std=c11 -O2 -Wall -Wextra -fopenmp test_int4.c kernels_neon_int4.c -o test_int4 -lm
// Run:
//   ./test_int4

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gemma4.h"

// The kernel under test (provided by whichever kernels_*.c is linked).
extern void matmul_int4(float *output, const int8_t *input_q, const float *input_scales,
                        const Tensor *weight, size_t rows);

// ----------------------------------------------------------------------------
// fp16 helpers (match the exporter's round-to-nearest encoding)

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

static uint16_t float_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exponent = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mantissa = x & 0x7fffff;
    if (exponent <= 0) return (uint16_t)sign;
    if (exponent >= 31) return (uint16_t)(sign | 0x7c00);
    uint32_t rounded = mantissa + 0x0fff + ((mantissa >> 13) & 1);
    return (uint16_t)(sign | ((uint32_t)exponent << 10) | (rounded >> 13));
}

// ----------------------------------------------------------------------------
// Deterministic PRNG (LCG)

static unsigned lcg(unsigned *state) {
    *state = *state * 1103515245u + 12345u;
    return *state;
}

// ----------------------------------------------------------------------------
// Build one synthetic int4 weight tensor and run the comparison.

static int run_case(const char *name, int outputs, int inputs, size_t rows) {
    const int group = 32; // int4: one fp16 scale per 32 inputs
    int groups = inputs / group;

    size_t weight_bytes = (size_t)outputs * inputs / 2;
    // One scale per (output row, input group): outputs * groups.
    size_t scale_count = (size_t)outputs * groups;

    uint8_t *wdata = malloc(weight_bytes);
    uint16_t *wscales = malloc(scale_count * sizeof(uint16_t));
    int8_t *input_q = malloc(rows * (size_t)inputs);
    float *input_scales = malloc(rows * (size_t)groups * sizeof(float));
    float *out_kernel = malloc(rows * (size_t)outputs * sizeof(float));
    float *out_ref = malloc(rows * (size_t)outputs * sizeof(float));

    unsigned state = 0x1234abcd;
    // Fill packed weights with random nibbles (0..15).
    for (size_t i = 0; i < weight_bytes; i++) wdata[i] = (uint8_t)(lcg(&state) & 0xff);
    // Realistic scales in [0.001, 1.0), stored as fp16.
    for (size_t i = 0; i < scale_count; i++) {
        float s = 0.001f + (float)((lcg(&state) >> 16) & 0xffff) / 65535.0f;
        wscales[i] = float_to_fp16(s);
    }
    // int8 activations in [-127, 127] and float activation scales.
    for (size_t i = 0; i < rows * (size_t)inputs; i++)
        input_q[i] = (int8_t)(((lcg(&state) >> 16) & 0xff) - 128 + 1);
    for (size_t i = 0; i < rows * (size_t)groups; i++)
        input_scales[i] = 0.01f + (float)((lcg(&state) >> 16) & 0xffff) / 65535.0f;

    // Heap-allocated so OpenMP worker threads (which read weight->shape) do not
    // race on a stack-local struct.
    Tensor *weight = malloc(sizeof(Tensor));
    weight->data = wdata;
    weight->scales = wscales;
    weight->shape[0] = outputs;
    weight->shape[1] = inputs;
    weight->shape[2] = 0;
    weight->shape[3] = 0;

    matmul_int4(out_kernel, input_q, input_scales, weight, rows);

    // Float reference: dequantize the exact packed weights and accumulate.
    const int block_rows = 16;
    for (size_t row = 0; row < rows; row++) {
        for (int j = 0; j < outputs; j++) {
            int block = j / block_rows;
            int r = j % block_rows;
            float sum = 0.0f;
            for (int g = 0; g < groups; g++) {
                int dot = 0;
                // Block stride is 16 rows x (inputs/2) bytes (int4 packs 2/byte).
                const uint8_t *wg = wdata + (size_t)block * block_rows * (inputs / 2)
                                    + (size_t)g * block_rows * 16 + r * 16;
                for (int k = 0; k < group; k++) {
                    uint8_t packed = wg[k / 2];
                    int nibble = (packed >> ((k % 2) * 4)) & 0xf;
                    dot += (int)input_q[row * inputs + g * group + k] * (nibble - 8);
                }
                float scale = fp16_to_f32(wscales[((size_t)block * groups + g) * 16 + r]);
                sum += (float)dot * input_scales[row * groups + g] * scale;
            }
            out_ref[row * outputs + j] = sum;
        }
    }

    float max_diff = 0.0f;
    size_t worst = 0;
    for (size_t i = 0; i < rows * (size_t)outputs; i++) {
        float d = fabsf(out_kernel[i] - out_ref[i]);
        if (d > max_diff) { max_diff = d; worst = i; }
    }
    printf("%s: rows=%zu outputs=%d inputs=%d  max_abs_diff=%.9f (at [%zu]: kernel=%.6f ref=%.6f)\n",
           name, rows, outputs, inputs, max_diff, worst, out_kernel[worst], out_ref[worst]);

    // Values are O(1000); a 1e-2 absolute tolerance is generous for fp32 accumulation.
    int ok = max_diff < 1e-2f;
    printf("  %s\n", ok ? "PASS" : "FAIL");

    free(wdata); free(wscales); free(input_q); free(input_scales);
    free(out_kernel); free(out_ref); free(weight);
    return ok;
}

int main(void) {
    int failures = 0;
    // Shapes matching gemma4.c linear layers.
    failures += !run_case("hidden (1536x1536, rows=1)", 1536, 1536, 1);   // decode GEMV
    failures += !run_case("hidden (1536x1536, rows=4)", 1536, 1536, 4);   // prefill GEMM
    failures += !run_case("mlp  (6144x1536, rows=1)", 6144, 1536, 1);
    failures += !run_case("mlp  (1536x6144, rows=2)", 1536, 6144, 2);
    failures += !run_case("kv   (256x1536, rows=1)", 256, 1536, 1);
    if (failures) {
        printf("FAIL: %d case(s) diverged from the float reference.\n", failures);
        return 1;
    }
    printf("PASS: all int4 matmul cases match the float reference.\n");
    return 0;
}
