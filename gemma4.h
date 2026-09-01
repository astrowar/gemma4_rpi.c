#ifndef GEMMA4_H
#define GEMMA4_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include "win.h"
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

#include <omp.h>

// SIMD intrinsics are only needed by the AVX2/AVX-512 kernel path.
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#include <immintrin.h>
#endif

#define NUM_LAYERS 35
#define HIDDEN_SIZE 1536
#define VOCAB_SIZE 262144
#define MAX_CONTEXT 131072
#define SLIDING_WINDOW 512
#define BATCH_SIZE 512

// ----------------------------------------------------------------------------
// Tokenizer types

typedef struct {
    char key[8]; // Holds one UTF-8 piece or a pair of 32-bit token IDs.
    int result, rank;
} LookupEntry;

typedef struct {
    char token[94]; // The longest vocabulary piece is 93 bytes plus the null terminator.
    int id;
} VocabEntry;

typedef struct {
    int merge_count;
    int encode_vocab_count;
    int special_count;
    char decoded_tokens[VOCAB_SIZE][94];
    VocabEntry specials[256]; // Reserves space for the checkpoint's 24 special tokens.
    LookupEntry encode_vocab[32768]; // Reserves space for 19,249 directly encoded pieces.
    LookupEntry merges[514906]; // This checkpoint contains 514,906 merge rules.
} Tokenizer;

// ----------------------------------------------------------------------------
// Model types

// data and scales begin as file offsets and become pointers after the model is memory-mapped.
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
    float residual[BATCH_SIZE * HIDDEN_SIZE];                           // Carries each token's hidden state through all 35 layers.
    float hidden[BATCH_SIZE * 8 * HIDDEN_SIZE];                         // Reused for intermediate results and sized for the largest 12,288-value MLP output.
    float auxiliary[BATCH_SIZE * 8 * HIDDEN_SIZE];                      // Holds a second intermediate when attention or the MLP needs two results at once.
    int8_t quantized[BATCH_SIZE * 8 * HIDDEN_SIZE];                     // Holds the current linear input after dynamic int8 quantization.
    float activation_scales[BATCH_SIZE * 8 * HIDDEN_SIZE / 64];         // Stores one float scale for every 64 quantized values.
    float per_layer_inputs[BATCH_SIZE * NUM_LAYERS * 256];              // Stores one 256-value conditioning vector for every token and layer.
    float sliding_cache[3][4][2 * (SLIDING_WINDOW + BATCH_SIZE) * 256]; // Keeps the previous window and current batch for twelve sliding KV caches.
    float full_cache[3][2 * MAX_CONTEXT * 512];                         // Holds the complete context for three 512-wide full-attention KV caches.
    int token_ids[MAX_CONTEXT];                                         // Holds the tokenized prompt before prefill.
} InferenceState;

typedef struct {
    char magic[4];
    Tokenizer tokenizer;
    ModelWeights weights;
} Model;

// Verifies that the compiler laid out the memory-mapped model exactly as the exporter expects.
_Static_assert(sizeof(int) == 4 && sizeof(float) == 4 && sizeof(void *) == 8 && sizeof(VocabEntry) == 100 && offsetof(VocabEntry, id) == 96 && sizeof(LookupEntry) == 16 && sizeof(Tokenizer) == 33429932 && sizeof(Tensor) == 32 && sizeof(ModelWeights) == 21472 && sizeof(Model) == 33451408 && offsetof(Model, weights) == 33429936 && BATCH_SIZE == SLIDING_WINDOW && !((SLIDING_WINDOW + BATCH_SIZE) & (SLIDING_WINDOW + BATCH_SIZE - 1)) && !(MAX_CONTEXT & (MAX_CONTEXT - 1)), "MOG ABI mismatch");

// ----------------------------------------------------------------------------
// Shared helpers

// Uses one OpenMP thread per physical core because each core already uses SIMD, unless OMP_NUM_THREADS overrides it.
static inline int thread_count(void) {
#if defined(__x86_64__) || defined(__i386__)
    unsigned int eax, ebx, ecx, edx;
    __cpuid_count(0xB, 0, eax, ebx, ecx, edx);
    int logical_cpus = omp_get_num_procs();
    int threads_per_core = ebx & 0xffff;
    return getenv("OMP_NUM_THREADS") ? omp_get_max_threads() : logical_cpus / (threads_per_core ? threads_per_core : 1);
#else
    return getenv("OMP_NUM_THREADS") ? omp_get_max_threads() : omp_get_num_procs();
#endif
}

// ----------------------------------------------------------------------------
// tokenizer.c

int tokenize(const Tokenizer *tokenizer, const char *segments[3], int *tokens, int capacity);
const char *token_text(const Tokenizer *tokenizer, int token);

// ----------------------------------------------------------------------------
// model.c

Model *model_load(const char *path);
void model_unload(Model *model, size_t size);

// ----------------------------------------------------------------------------
// kernels.c

void matmul_int8(float *output, const int8_t *input_q, const float *input_scales,
                 const Tensor *weight, size_t rows);
void quantize(int8_t *quantized, float *scales, const float *input, size_t rows, size_t width);
void attention_scores(float *scores, const float *query, const float *key_cache,
                      int first_key, int num_keys, int cache_mask, int head_dim);
void weighted_value_sum(float *output, const float *probabilities, const float *value_cache,
                        int first_key, int num_keys, int cache_mask, int head_dim);
void geglu(float *gate, const float *up, int rows, int width, int up_stride, const Tensor *gelu_table);

// ----------------------------------------------------------------------------
// transformer.c

void forward(Model *model, InferenceState *state, const int *tokens, size_t token_count, int start_pos);
float *logits(Model *model, InferenceState *state, size_t token);

// ----------------------------------------------------------------------------
// generate.c

void seed_rng(void);
void prefill(Model *model, InferenceState *state, const int *tokens, int token_count, int dump_logits);
void generate(Model *model, InferenceState *state, const char *prompt,
              int max_new_tokens, float temperature, int dump_logits, int stats);
void benchmark(Model *model, InferenceState *state, int prefill_tokens, int generated_tokens);

#endif // GEMMA4_H
