// Pure C (scalar) kernel declarations for gemma4.c.
// No SIMD intrinsics, no architecture-specific code.
// Designed for portability to any platform with a C11 compiler.
// Implementation: kernels_pure_int8.c + kernels_pure_int4.c

#ifndef KERNELS_PURE_H
#define KERNELS_PURE_H

#include <stddef.h>
#include <stdint.h>

// Model constants (must match the exported model layout)
#define NUM_LAYERS 35
#define HIDDEN_SIZE 1536
#define VOCAB_SIZE 262144
#define MAX_CONTEXT 131072
#define SLIDING_WINDOW 512
#define BATCH_SIZE 512

// ----------------------------------------------------------------------------
// Types

typedef struct {
    void *data;
    uint16_t *scales;
    int shape[4];
} Tensor;

typedef struct {
    Tensor input_layernorm;
    Tensor layer_scalar;
    Tensor pre_ffn_layernorm;
    Tensor post_attn_layernorm;
    Tensor post_ffn_layernorm;
    Tensor post_per_layer_input_norm;
    Tensor per_layer_input_gate;
    Tensor per_layer_projection;
    Tensor q_norm;
    Tensor k_norm;
    Tensor q_proj;
    Tensor k_proj;
    Tensor v_proj;
    Tensor o_proj;
    Tensor gate_proj;
    Tensor up_proj;
    Tensor down_proj;
    Tensor rope_cos;
    Tensor rope_sin;
} LayerWeights;

typedef struct {
    Tensor embed;
    Tensor embed_per_layer;
    LayerWeights layers[NUM_LAYERS];
    Tensor norm;
    Tensor per_layer_model_projection;
    Tensor per_layer_projection_norm;
    Tensor gelu_table;
} ModelWeights;

typedef struct {
    float residual[BATCH_SIZE * HIDDEN_SIZE];
    float hidden[BATCH_SIZE * 8 * HIDDEN_SIZE];
    float auxiliary[BATCH_SIZE * 8 * HIDDEN_SIZE];
    int8_t quantized[BATCH_SIZE * 8 * HIDDEN_SIZE];
    float activation_scales[BATCH_SIZE * 8 * HIDDEN_SIZE / 64];
    float per_layer_inputs[BATCH_SIZE * NUM_LAYERS * 256];
    float sliding_cache[3][4][2 * (SLIDING_WINDOW + BATCH_SIZE) * 256];
    float full_cache[3][2 * MAX_CONTEXT * 512];
    int token_ids[MAX_CONTEXT];
} InferenceState;

// ----------------------------------------------------------------------------
// Kernel functions (all pure C, no intrinsics)

// Converts a packed half-precision (fp16) value to float.
float fp16_to_float(uint16_t value);

// Reads a single packed int8 weight from the block-interleaved layout.
int8_t packed_weight(const Tensor *weight, int output, int input);

// Reads the fp16 scale for a given output row and input group.
float weight_scale(const Tensor *weight, int output, int input_group);

// Quantizes float activations to int8 in groups of 64, storing per-group scales.
void quantize(int8_t *output, float *scales, const float *input, size_t rows, size_t width);

// int8 x int8 matrix multiply with per-group rescaling.
// output[rows x weight->shape[0]] = input[rows x weight->shape[1]] * weight^T
void matmul_int8(float *output, const int8_t *input, const float *input_scales,
                 const Tensor *weight, size_t rows);

// Looks up embedding rows from the packed int8 table and dequantizes.
void embedding(float *output, const Tensor *table, const int *tokens,
               size_t token_count, float multiplier);

// Root-mean-square normalization over each row.
void rmsnorm(float *output, const float *input, const Tensor *weight,
             int width, float epsilon, size_t rows);

// output[i] = (output[i] + addend[i]) * scale
void add_and_scale(float *output, const float *addend, size_t count, float scale);

// Applies rotary position embeddings to query/key vectors.
void apply_rope(const Tensor *cosines, const Tensor *sines, float *vectors,
                int num_heads, int head_dim, int start_pos, size_t token_count);

// In-place softmax over a single row.
void softmax(float *values, int count);

// GELU (table-interpolated) * up-projection element-wise.
void geglu(float *gate, const float *up, int rows, int width, int up_stride,
           const Tensor *gelu_table);

// Dot product of query against each key in the cache ring.
void attention_scores(float *scores, const float *query, const float *key_cache,
                      int first_key, int num_keys, int cache_mask, int head_dim);

// Weighted sum of value vectors using attention probabilities.
void weighted_value_sum(float *output, const float *probabilities, const float *value_cache,
                        int first_key, int num_keys, int cache_mask, int head_dim);

#endif // KERNELS_PURE_H
