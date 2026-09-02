// Validation: compares kernels_pure_int8.c (scalar) against kernels_avx_int8.c (AVX2/AVX-512)
// using real model weights.
//
// Build: cc -std=c11 -O3 -Wall -Wextra -march=native -fopenmp
//        test_kernels_pure.c kernels_avx_int8.c kernels_pure_int8_renamed.c -o test_kernels_pure -lm
// Run:   ./test_kernels_pure ./gemma4-E2B-int8.bin

#include <fcntl.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <omp.h>
#include "gemma4.h"

// Declarations for the pure C kernels (renamed with _pure suffix)
extern float fp16_to_float_pure(uint16_t value);
extern int8_t packed_weight_pure(const Tensor *weight, int output, int input);
extern float weight_scale_pure(const Tensor *weight, int output, int input_group);
extern void quantize_pure(int8_t *output, float *scales, const float *input, size_t rows, int width);
extern void matmul_int8_pure(float *output, const int8_t *input, const float *input_scales,
                             const Tensor *weight, size_t rows);
extern void embedding_pure(float *output, const Tensor *table, const int *tokens,
                           size_t token_count, float multiplier);
extern void rmsnorm_pure(float *output, const float *input, const Tensor *weight,
                         int width, float epsilon, size_t rows);
extern void add_and_scale_pure(float *output, const float *addend, size_t count, float scale);
extern void apply_rope_pure(const Tensor *cosines, const Tensor *sines, float *vectors,
                            int num_heads, int head_dim, int start_pos, size_t token_count);
extern void softmax_pure(float *values, int count);
extern void geglu_pure(float *gate, const float *up, int rows, int width, int up_stride,
                       const Tensor *gelu_table);
extern void attention_scores_pure(float *scores, const float *query, const float *key_cache,
                                  int first_key, int num_keys, int cache_mask, int head_dim);
extern void weighted_value_sum_pure(float *output, const float *probabilities, const float *value_cache,
                                    int first_key, int num_keys, int cache_mask, int head_dim);

// ----------------------------------------------------------------------------

static int check_close(const char *name, const float *a, const float *b, size_t count, float tol) {
    float max_diff = 0.0f;
    size_t worst = 0;
    for (size_t i = 0; i < count; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) { max_diff = diff; worst = i; }
    }
    if (max_diff > tol) {
        fprintf(stderr, "FAIL %s: max_diff=%.9f at [%zu] (pure=%.6f simd=%.6f)\n",
                name, max_diff, worst, a[worst], b[worst]);
        return 1;
    }
    printf("  OK  %-24s max_diff=%.9f\n", name, max_diff);
    return 0;
}

static int check_int8(const char *name, const int8_t *a, const int8_t *b, size_t count) {
    int mismatches = 0;
    for (size_t i = 0; i < count; i++) {
        if (a[i] != b[i]) {
            if (mismatches < 3)
                fprintf(stderr, "  %s: [%zu] pure=%d simd=%d\n", name, i, a[i], b[i]);
            mismatches++;
        }
    }
    if (mismatches) {
        fprintf(stderr, "FAIL %s: %d/%zu mismatches\n", name, mismatches, count);
        return 1;
    }
    printf("  OK  %-24s exact match (%zu elements)\n", name, count);
    return 0;
}

// ----------------------------------------------------------------------------
int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "gemma4-E2B-int8.bin";

    int fd = open(model_path, O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st)) { perror(model_path); return 1; }
    Model *model = mmap(NULL, (size_t)st.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    close(fd);
    if (model == MAP_FAILED) { perror("mmap"); return 1; }
    if (memcmp(model->magic, "MOG", 4) != 0) { fprintf(stderr, "bad model file\n"); return 1; }
    Tensor *tensors = (Tensor *)&model->weights;
    for (size_t i = 0; i < sizeof(model->weights) / sizeof(*tensors); i++) {
        tensors[i].data = tensors[i].data ? (void *)((uint8_t *)model + (uintptr_t)tensors[i].data) : NULL;
        tensors[i].scales = tensors[i].scales ? (uint16_t *)((uint8_t *)model + (uintptr_t)tensors[i].scales) : NULL;
    }

    printf("Validating pure C kernels against AVX2/AVX-512 kernels\n");
    printf("Model: %s (%.1f MB)\n\n", model_path, (double)st.st_size / (1024 * 1024));

    int failures = 0;
    Tensor *w = &model->weights.layers[0].q_proj;
    int rows = 4;
    int width = w->shape[1];
    int outputs = w->shape[0];
    int groups = width / 64;

    float *input = calloc((size_t)rows * width, sizeof(float));
    int8_t *q_pure = malloc((size_t)rows * width);
    int8_t *q_simd = malloc((size_t)rows * width);
    float *scales_pure = malloc((size_t)rows * groups * sizeof(float));
    float *scales_simd = malloc((size_t)rows * groups * sizeof(float));
    float *out_pure = malloc((size_t)rows * outputs * sizeof(float));
    float *out_simd = malloc((size_t)rows * outputs * sizeof(float));

    srand(42);
    for (size_t i = 0; i < (size_t)rows * width; i++)
        input[i] = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;

    // 1: quantize
    printf("[quantize]\n");
    quantize_pure(q_pure, scales_pure, input, rows, width);
    quantize(q_simd, scales_simd, input, rows, width);
    failures += check_int8("quantize int8", q_pure, q_simd, (size_t)rows * width);
    failures += check_close("quantize scales", scales_pure, scales_simd, (size_t)rows * groups, 1e-7f);

    // 2: matmul_int8
    printf("[matmul_int8]\n");
    matmul_int8_pure(out_pure, q_pure, scales_pure, w, rows);
    matmul_int8(out_simd, q_simd, scales_simd, w, rows);
    failures += check_close("matmul_int8", out_pure, out_simd, (size_t)rows * outputs, 1e-3f);

    // 3: embedding
    printf("[embedding]\n");
    {
        Tensor *embed = &model->weights.embed;
        int e_width = embed->shape[1];
        int e_rows = 4;
        int test_tokens[4] = {100, 2000, 50000, 200000};
        float *emb = malloc((size_t)e_rows * e_width * sizeof(float));
        embedding_pure(emb, embed, test_tokens, e_rows, sqrtf((float)HIDDEN_SIZE));
        int finite = 1;
        for (int i = 0; i < e_rows * e_width; i++)
            if (!isfinite(emb[i])) { finite = 0; break; }
        if (finite) printf("  OK  %-24s all %d values finite\n", "embedding", e_rows * e_width);
        else { printf("  FAIL embedding: non-finite values\n"); failures++; }
        free(emb);
    }

    // 4: rmsnorm
    printf("[rmsnorm]\n");
    {
        Tensor *norm_w = &model->weights.layers[0].input_layernorm;
        int n_width = HIDDEN_SIZE, n_rows = 2;
        float *in = malloc((size_t)n_rows * n_width * sizeof(float));
        float *o_pure = malloc((size_t)n_rows * n_width * sizeof(float));
        float *o_ref = malloc((size_t)n_rows * n_width * sizeof(float));
        for (int i = 0; i < n_rows * n_width; i++)
            in[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        rmsnorm_pure(o_pure, in, norm_w, n_width, 1e-6f, n_rows);
        const float *weights = (const float *)norm_w->data;
        for (int r = 0; r < n_rows; r++) {
            float ss = 0.0f;
            for (int i = 0; i < n_width; i++) ss += in[r * n_width + i] * in[r * n_width + i];
            float inv = 1.0f / sqrtf(ss / (float)n_width + 1e-6f);
            for (int i = 0; i < n_width; i++)
                o_ref[r * n_width + i] = weights[i] * inv * in[r * n_width + i];
        }
        failures += check_close("rmsnorm", o_pure, o_ref, (size_t)n_rows * n_width, 1e-6f);
        free(in); free(o_pure); free(o_ref);
    }

    // 5: add_and_scale
    printf("[add_and_scale]\n");
    {
        size_t count = 1024;
        float *a = malloc(count * sizeof(float));
        float *b = malloc(count * sizeof(float));
        float *a_pure = malloc(count * sizeof(float));
        float *a_ref = malloc(count * sizeof(float));
        for (size_t i = 0; i < count; i++) {
            a[i] = ((float)rand() / RAND_MAX - 0.5f);
            b[i] = ((float)rand() / RAND_MAX - 0.5f);
            a_pure[i] = a[i];
            a_ref[i] = a[i];
        }
        add_and_scale_pure(a_pure, b, count, 0.7f);
        for (size_t i = 0; i < count; i++) a_ref[i] = (a_ref[i] + b[i]) * 0.7f;
        failures += check_close("add_and_scale", a_pure, a_ref, count, 1e-7f);
        free(a); free(b); free(a_pure); free(a_ref);
    }

    // 6: apply_rope
    printf("[apply_rope]\n");
    {
        Tensor *cos_t = &model->weights.layers[0].rope_cos;
        Tensor *sin_t = &model->weights.layers[0].rope_sin;
        int head_dim = model->weights.layers[0].q_norm.shape[0];
        int num_heads = 1, token_count = 2;
        float *v_pure = malloc((size_t)token_count * num_heads * head_dim * sizeof(float));
        float *v_orig = malloc((size_t)token_count * num_heads * head_dim * sizeof(float));
        for (int i = 0; i < token_count * num_heads * head_dim; i++)
            v_pure[i] = v_orig[i] = ((float)rand() / RAND_MAX - 0.5f);
        apply_rope_pure(cos_t, sin_t, v_pure, num_heads, head_dim, 0, token_count);
        int finite = 1;
        for (int i = 0; i < token_count * num_heads * head_dim; i++)
            if (!isfinite(v_pure[i])) { finite = 0; break; }
        if (finite) printf("  OK  %-24s all values finite\n", "apply_rope");
        else { printf("  FAIL apply_rope: non-finite\n"); failures++; }
        free(v_pure); free(v_orig);
    }

    // 7: softmax
    printf("[softmax]\n");
    {
        int count = 64;
        float *s_pure = malloc(count * sizeof(float));
        float *s_ref = malloc(count * sizeof(float));
        for (int i = 0; i < count; i++)
            s_pure[i] = s_ref[i] = ((float)rand() / RAND_MAX - 0.5f) * 10.0f;
        softmax_pure(s_pure, count);
        float mx = s_ref[0];
        for (int i = 1; i < count; i++) if (s_ref[i] > mx) mx = s_ref[i];
        float sum = 0.0f;
        for (int i = 0; i < count; i++) sum += s_ref[i] = expf(s_ref[i] - mx);
        for (int i = 0; i < count; i++) s_ref[i] /= sum;
        failures += check_close("softmax", s_pure, s_ref, count, 1e-6f);
        free(s_pure); free(s_ref);
    }

    // 8: geglu
    printf("[geglu]\n");
    {
        Tensor *gelu = &model->weights.gelu_table;
        int g_rows = 2, g_width = 256;
        float *gate_pure = malloc((size_t)g_rows * g_width * sizeof(float));
        float *gate_ref = malloc((size_t)g_rows * g_width * sizeof(float));
        float *up = malloc((size_t)g_rows * g_width * sizeof(float));
        for (int i = 0; i < g_rows * g_width; i++) {
            gate_pure[i] = gate_ref[i] = ((float)rand() / RAND_MAX - 0.5f) * 6.0f;
            up[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        }
        geglu_pure(gate_pure, up, g_rows, g_width, g_width, gelu);
        const float *table = (const float *)gelu->data;
        int table_size = gelu->shape[0];
        float lower = (float)gelu->shape[1], upper = (float)gelu->shape[2];
        float scale = (float)(table_size - 1) / (upper - lower);
        for (int r = 0; r < g_rows; r++)
            for (int i = 0; i < g_width; i++) {
                float x = gate_ref[r * g_width + i];
                if (x <= lower) x = table[0];
                else if (x < upper) {
                    float pos = (x - lower) * scale;
                    int idx = (int)pos;
                    float frac = pos - (float)idx;
                    x = table[idx] + frac * (table[idx + 1] - table[idx]);
                }
                gate_ref[r * g_width + i] = x * up[r * g_width + i];
            }
        failures += check_close("geglu", gate_pure, gate_ref, (size_t)g_rows * g_width, 1e-6f);
        free(gate_pure); free(gate_ref); free(up);
    }

    // 9: attention_scores (pure vs AVX2)
    printf("[attention_scores]\n");
    {
        int head_dim = model->weights.layers[0].q_norm.shape[0];
        int num_keys = 16;
        int cache_len = SLIDING_WINDOW + BATCH_SIZE;
        int cache_mask = cache_len - 1;
        float *query = malloc(head_dim * sizeof(float));
        float *key_cache = malloc((size_t)cache_len * head_dim * sizeof(float));
        float *sc_pure = malloc(num_keys * sizeof(float));
        float *sc_simd = malloc(num_keys * sizeof(float));
        for (int i = 0; i < head_dim; i++) query[i] = ((float)rand() / RAND_MAX - 0.5f);
        for (int i = 0; i < cache_len * head_dim; i++) key_cache[i] = ((float)rand() / RAND_MAX - 0.5f);
        attention_scores_pure(sc_pure, query, key_cache, 0, num_keys, cache_mask, head_dim);
        attention_scores(sc_simd, query, key_cache, 0, num_keys, cache_mask, head_dim);
        failures += check_close("attention_scores", sc_pure, sc_simd, num_keys, 1e-5f);
        free(query); free(key_cache); free(sc_pure); free(sc_simd);
    }

    // 10: weighted_value_sum (pure vs AVX2)
    printf("[weighted_value_sum]\n");
    {
        int head_dim = model->weights.layers[0].q_norm.shape[0];
        int num_keys = 16;
        int cache_len = SLIDING_WINDOW + BATCH_SIZE;
        int cache_mask = cache_len - 1;
        float *probs = malloc(num_keys * sizeof(float));
        float *val_cache = malloc((size_t)cache_len * head_dim * sizeof(float));
        float *o_pure = malloc(head_dim * sizeof(float));
        float *o_simd = malloc(head_dim * sizeof(float));
        for (int i = 0; i < num_keys; i++) probs[i] = (float)rand() / RAND_MAX;
        float psum = 0; for (int i = 0; i < num_keys; i++) psum += probs[i];
        for (int i = 0; i < num_keys; i++) probs[i] /= psum;
        for (int i = 0; i < cache_len * head_dim; i++) val_cache[i] = ((float)rand() / RAND_MAX - 0.5f);
        weighted_value_sum_pure(o_pure, probs, val_cache, 0, num_keys, cache_mask, head_dim);
        weighted_value_sum(o_simd, probs, val_cache, 0, num_keys, cache_mask, head_dim);
        failures += check_close("weighted_value_sum", o_pure, o_simd, head_dim, 1e-5f);
        free(probs); free(val_cache); free(o_pure); free(o_simd);
    }

    // 11: fp16_to_float
    printf("[fp16_to_float]\n");
    {
        int ok = 1;
        uint16_t test_vals[] = {0x0000, 0x0001, 0x3C00, 0x3C01, 0x7BFF, 0x7C00, 0x7C01, 0x7FFF, 0xFC00, 0xFBFF};
        float expected[] = {0.0f, 5.9604645e-8f, 1.0f, 1.0009766f, 65504.0f, INFINITY, NAN, NAN, -INFINITY, -65504.0f};
        for (int i = 0; i < 10; i++) {
            float r = fp16_to_float_pure(test_vals[i]);
            if (isnan(expected[i])) {
                if (!isnan(r)) { printf("  FAIL fp16_to_float[0x%04X]: got %f, expected NaN\n", test_vals[i], r); ok = 0; }
            } else if (isinf(expected[i])) {
                if (!isinf(r) || (r > 0) != (expected[i] > 0)) { printf("  FAIL fp16_to_float[0x%04X]: got %f, expected inf\n", test_vals[i], r); ok = 0; }
            } else if (fabsf(r - expected[i]) > 1e-4f * (fabsf(expected[i]) + 1e-8f)) {
                printf("  FAIL fp16_to_float[0x%04X]: got %f, expected %f\n", test_vals[i], r, expected[i]);
                ok = 0;
            }
        }
        if (ok) printf("  OK  %-24s all 10 test vectors pass\n", "fp16_to_float");
        else failures++;
    }

    // 12: packed_weight
    printf("[packed_weight]\n");
    {
        int ok = 1;
        const int8_t *data = (const int8_t *)w->data;
        int width_w = w->shape[1];
        for (int test = 0; test < 20; test++) {
            int out = (rand() % 16) + (rand() % (outputs / 16)) * 16;
            int inp = rand() % width_w;
            int block = out / 16, row = out % 16;
            int group = inp / 64, chunk = inp % 64 / 4, offset = inp % 4;
            int8_t expected = data[(size_t)block * 16 * width_w + (size_t)group * 16 * 64
                                   + chunk * 16 * 4 + row * 4 + offset];
            int8_t got = packed_weight_pure(w, out, inp);
            if (got != expected) {
                printf("  FAIL packed_weight[%d,%d]: got %d expected %d\n", out, inp, got, expected);
                ok = 0;
            }
        }
        if (ok) printf("  OK  %-24s 20 random lookups match\n", "packed_weight");
        else failures++;
    }

    free(input); free(q_pure); free(q_simd);
    free(scales_pure); free(scales_simd);
    free(out_pure); free(out_simd);
    munmap(model, (size_t)st.st_size);

    printf("\n%s: %d failure(s)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
