// Profiling build: instruments forward() and logits() with per-component timers.
// Uses omp critical for thread-safe accumulation.
// Compile: make KERNELS=neon profile ARGS="-n 4"

#include "gemma4.h"
#include <time.h>

// Functions defined in transformer.c / generate.c (not in gemma4.h).
extern void rmsnorm(float *output, const float *input, const Tensor *weight, int width, float epsilon, size_t row_count);
extern void add_and_scale(float *output, const float *addend, size_t count, float scale);
extern void apply_rope(const Tensor *cosines, const Tensor *sines, float *vectors, int num_heads, int head_dim, int start_pos, size_t token_count);
extern void softmax(float *values, int count);
extern int sample(float *logits, int vocab_size, float temperature);

// Dispatchers from transformer.c — route to int8 or int4 based on model->quant.
extern void embedding_dispatch(Model *model, float *output, const Tensor *table, const int *tokens, size_t token_count, float multiplier);
extern void quantize_dispatch(Model *model, int8_t *output, float *scales, const float *input, size_t rows, size_t width);
extern void matmul_dispatch(Model *model, float *output, const int8_t *input_q, const float *input_scales, const Tensor *weight, size_t rows);

// --- Accumulators (thread-safe via omp critical) ---
static double t_embedding, t_rmsnorm, t_quantize, t_matmul, t_attention,
              t_softmax, t_geglu, t_add_scale, t_rope, t_other;
static int prof_layer_count;

static inline double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

// Timed section macro: measures wall-clock, adds to accumulator under critical.
#define TIMED(acc, code) do { \
    double _t0 = now(); \
    code; \
    double _dt = now() - _t0; \
    _Pragma("omp critical") { (acc) += _dt; } \
} while(0)

void prof_forward(Model *model, InferenceState *state, const int *tokens,
                 size_t token_count, int start_pos) {
    int per_layer_width = model->weights.per_layer_projection_norm.shape[0];
    #pragma omp parallel num_threads(thread_count())
    {
        float scores[(size_t)start_pos + token_count];

        TIMED(t_embedding,
            embedding_dispatch(model, state->residual, &model->weights.embed, tokens, token_count, sqrtf((float)HIDDEN_SIZE)));

        TIMED(t_quantize,
            quantize_dispatch(model, state->quantized, state->activation_scales, state->residual, token_count, HIDDEN_SIZE));

        TIMED(t_matmul,
            matmul_dispatch(model, state->per_layer_inputs, state->quantized, state->activation_scales, &model->weights.per_layer_model_projection, token_count));

        TIMED(t_rmsnorm,
            rmsnorm(state->per_layer_inputs, state->per_layer_inputs, &model->weights.per_layer_projection_norm, per_layer_width, 1e-6f * HIDDEN_SIZE, token_count * NUM_LAYERS));

        TIMED(t_embedding,
            embedding_dispatch(model, state->hidden, &model->weights.embed_per_layer, tokens, token_count, sqrtf((float)per_layer_width)));

        TIMED(t_add_scale,
            add_and_scale(state->per_layer_inputs, state->hidden, token_count * NUM_LAYERS * per_layer_width, 1.0f / sqrtf(2.0f)));

        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            LayerWeights *weights = &model->weights.layers[layer];

            TIMED(t_rmsnorm,
                rmsnorm(state->hidden, state->residual, &weights->input_layernorm, HIDDEN_SIZE, 1e-6f, token_count));

            // --- Attention ---
            {
                const LayerWeights *lw = weights;
                int full_attention = layer % 5 == 4;
                int cache_len = full_attention ? MAX_CONTEXT : SLIDING_WINDOW + BATCH_SIZE;
                int cache_mask = cache_len - 1;
                int head_dim = lw->q_norm.shape[0];
                int query_width = lw->q_proj.shape[0];
                int cache_owner = layer;
                while (!model->weights.layers[cache_owner].k_proj.data || (cache_owner % 5 == 4) != full_attention) cache_owner--;
                float *key_cache = full_attention ? state->full_cache[cache_owner / 5] : state->sliding_cache[cache_owner / 5][cache_owner % 5];
                float *value_cache = key_cache + (size_t)cache_len * head_dim;

                TIMED(t_quantize,
                    quantize_dispatch(model, state->quantized, state->activation_scales, state->hidden, token_count, lw->q_proj.shape[1]));

                TIMED(t_matmul,
                    matmul_dispatch(model, state->auxiliary, state->quantized, state->activation_scales, &lw->q_proj, token_count));

                TIMED(t_rmsnorm,
                    rmsnorm(state->auxiliary, state->auxiliary, &lw->q_norm, head_dim, 1e-6f, token_count * (query_width / head_dim)));

                TIMED(t_rope,
                    apply_rope(&lw->rope_cos, &lw->rope_sin, state->auxiliary, query_width / head_dim, head_dim, start_pos, token_count));

                if (lw->k_proj.data) {
                    float *new_keys = key_cache + ((size_t)start_pos & cache_mask) * head_dim;
                    float *new_values = value_cache + ((size_t)start_pos & cache_mask) * head_dim;

                    TIMED(t_matmul,
                        matmul_dispatch(model, new_keys, state->quantized, state->activation_scales, &lw->k_proj, token_count));

                    TIMED(t_matmul,
                        matmul_dispatch(model, new_values, state->quantized, state->activation_scales, &lw->v_proj, token_count));

                    TIMED(t_rmsnorm,
                        rmsnorm(new_keys, new_keys, &lw->k_norm, head_dim, 1e-6f, token_count));

                    TIMED(t_rmsnorm,
                        rmsnorm(new_values, new_values, NULL, head_dim, 1e-6f, token_count));

                    TIMED(t_rope,
                        apply_rope(&lw->rope_cos, &lw->rope_sin, new_keys, 1, head_dim, start_pos, token_count));
                }

                #pragma omp for collapse(2) schedule(dynamic, 1)
                for (size_t head = 0; head < (size_t)(query_width / head_dim); head++) {
                    for (size_t token = 0; token < token_count; token++) {
                        int first_key = !full_attention && start_pos + (int)token + 1 > SLIDING_WINDOW ? start_pos + (int)token + 1 - SLIDING_WINDOW : 0;
                        int num_keys = start_pos + (int)token + 1 - first_key;
                        float *head_output = state->hidden + token * query_width + head * head_dim;
                        const float *query = state->auxiliary + token * query_width + head * head_dim;

                        TIMED(t_attention,
                            attention_scores(scores, query, key_cache, first_key, num_keys, cache_mask, head_dim));

                        TIMED(t_softmax,
                            softmax(scores, num_keys));

                        TIMED(t_attention,
                            weighted_value_sum(head_output, scores, value_cache, first_key, num_keys, cache_mask, head_dim));
                    }
                }

                TIMED(t_quantize,
                    quantize_dispatch(model, state->quantized, state->activation_scales, state->hidden, token_count, lw->o_proj.shape[1]));

                TIMED(t_matmul,
                    matmul_dispatch(model, state->hidden, state->quantized, state->activation_scales, &lw->o_proj, token_count));
            }

            TIMED(t_rmsnorm,
                rmsnorm(state->hidden, state->hidden, &weights->post_attn_layernorm, HIDDEN_SIZE, 1e-6f, token_count));

            TIMED(t_add_scale,
                add_and_scale(state->residual, state->hidden, token_count * HIDDEN_SIZE, 1.0f));

            // --- FFN ---
            TIMED(t_rmsnorm,
                rmsnorm(state->hidden, state->residual, &weights->pre_ffn_layernorm, HIDDEN_SIZE, 1e-6f, token_count));

            TIMED(t_quantize,
                quantize_dispatch(model, state->quantized, state->activation_scales, state->hidden, token_count, weights->gate_proj.shape[1]));

            TIMED(t_matmul,
                matmul_dispatch(model, state->hidden, state->quantized, state->activation_scales, &weights->gate_proj, token_count));

            TIMED(t_matmul,
                matmul_dispatch(model, state->auxiliary, state->quantized, state->activation_scales, &weights->up_proj, token_count));

            TIMED(t_geglu,
                geglu(state->hidden, state->auxiliary, token_count, weights->gate_proj.shape[0], weights->gate_proj.shape[0], &model->weights.gelu_table));

            TIMED(t_quantize,
                quantize_dispatch(model, state->quantized, state->activation_scales, state->hidden, token_count, weights->down_proj.shape[1]));

            TIMED(t_matmul,
                matmul_dispatch(model, state->hidden, state->quantized, state->activation_scales, &weights->down_proj, token_count));

            TIMED(t_rmsnorm,
                rmsnorm(state->hidden, state->hidden, &weights->post_ffn_layernorm, HIDDEN_SIZE, 1e-6f, token_count));

            TIMED(t_add_scale,
                add_and_scale(state->residual, state->hidden, token_count * HIDDEN_SIZE, 1.0f));

            // --- Per-layer projection ---
            TIMED(t_quantize,
                quantize_dispatch(model, state->quantized, state->activation_scales, state->residual, token_count, HIDDEN_SIZE));

            TIMED(t_matmul,
                matmul_dispatch(model, state->hidden, state->quantized, state->activation_scales, &weights->per_layer_input_gate, token_count));

            TIMED(t_geglu,
                geglu(state->hidden, state->per_layer_inputs + layer * per_layer_width, token_count, per_layer_width, NUM_LAYERS * per_layer_width, &model->weights.gelu_table));

            TIMED(t_quantize,
                quantize_dispatch(model, state->quantized, state->activation_scales, state->hidden, token_count, weights->per_layer_projection.shape[1]));

            TIMED(t_matmul,
                matmul_dispatch(model, state->hidden, state->quantized, state->activation_scales, &weights->per_layer_projection, token_count));

            TIMED(t_rmsnorm,
                rmsnorm(state->hidden, state->hidden, &weights->post_per_layer_input_norm, HIDDEN_SIZE, 1e-6f, token_count));

            TIMED(t_add_scale,
                add_and_scale(state->residual, state->hidden, token_count * HIDDEN_SIZE, ((float *)weights->layer_scalar.data)[0]));

            _Pragma("omp critical") { prof_layer_count++; }
        }
    }
}

float *prof_logits(Model *model, InferenceState *state, size_t token) {
    #pragma omp parallel num_threads(thread_count())
    {
        TIMED(t_rmsnorm,
            rmsnorm(state->hidden, state->residual + token * HIDDEN_SIZE, &model->weights.norm, HIDDEN_SIZE, 1e-6f, 1));

        TIMED(t_quantize,
            quantize_dispatch(model, state->quantized, state->activation_scales, state->hidden, 1, HIDDEN_SIZE));

        TIMED(t_matmul,
            matmul_dispatch(model, state->hidden, state->quantized, state->activation_scales, &model->weights.embed, 1));

        {
            double _t0 = now();
            #pragma omp for schedule(static)
            for (int i = 0; i < VOCAB_SIZE; i++) state->hidden[i] = 30.0f * tanhf(state->hidden[i] / 30.0f);
            double _dt = now() - _t0;
            _Pragma("omp critical") { t_other += _dt; }
        }
    }
    return state->hidden;
}

void prof_generate(Model *model, InferenceState *state, const char *prompt,
                  int max_new_tokens, float temperature) {
    Tokenizer *tokenizer = &model->tokenizer;
    const char *segments[3] = {"<|turn>user\n", prompt, "<turn|>\n<|turn>model\n"};
    int prompt_tokens = tokenize(tokenizer, segments, state->token_ids, MAX_CONTEXT);
    if (prompt_tokens < 0) { fprintf(stderr, "prompt too long\n"); exit(1); }

    double t_pf_start = now();
    for (int position = 0; position < prompt_tokens; position += BATCH_SIZE) {
        int chunk = prompt_tokens - position < BATCH_SIZE ? prompt_tokens - position : BATCH_SIZE;
        prof_forward(model, state, state->token_ids + position, chunk, position);
    }
    double t_pf_end = now();

    int end = prompt_tokens + max_new_tokens;
    if (end > MAX_CONTEXT) end = MAX_CONTEXT;
    int generated = 0;
    double t_dec_start = now();
    for (int position = prompt_tokens; position < end; position++) {
        float *logits_out = prof_logits(model, state, position == prompt_tokens ? (prompt_tokens - 1) % BATCH_SIZE : 0);
        int next_token = sample(logits_out, VOCAB_SIZE, temperature);
        if (next_token == 1 || next_token == 106) break;
        fputs(token_text(tokenizer, next_token), stdout);
        fflush(stdout);
        prof_forward(model, state, &next_token, 1, position);
        generated++;
    }
    double t_dec_end = now();

    double total = t_pf_end - t_pf_start + (t_dec_end - t_dec_start);
    int nthreads = thread_count();
    fprintf(stderr, "\n=== PROFILE (%d prefill + %d decode tokens, %.2fs wall, %d threads) ===\n",
            prompt_tokens, generated, total, nthreads);
    fprintf(stderr, "  %-20s %8.3f ms  %5.1f%%\n", "matmul_int8",      t_matmul    * 1000 / nthreads, 100.0 * t_matmul    / (total * nthreads));
    fprintf(stderr, "  %-20s %8.3f ms  %5.1f%%\n", "quantize",         t_quantize  * 1000 / nthreads, 100.0 * t_quantize  / (total * nthreads));
    fprintf(stderr, "  %-20s %8.3f ms  %5.1f%%\n", "rmsnorm",          t_rmsnorm   * 1000 / nthreads, 100.0 * t_rmsnorm   / (total * nthreads));
    fprintf(stderr, "  %-20s %8.3f ms  %5.1f%%\n", "attention_scores", t_attention * 1000 / nthreads, 100.0 * t_attention / (total * nthreads));
    fprintf(stderr, "  %-20s %8.3f ms  %5.1f%%\n", "softmax",          t_softmax   * 1000 / nthreads, 100.0 * t_softmax   / (total * nthreads));
    fprintf(stderr, "  %-20s %8.3f ms  %5.1f%%\n", "geglu",            t_geglu     * 1000 / nthreads, 100.0 * t_geglu     / (total * nthreads));
    fprintf(stderr, "  %-20s %8.3f ms  %5.1f%%\n", "embedding",        t_embedding * 1000 / nthreads, 100.0 * t_embedding / (total * nthreads));
    fprintf(stderr, "  %-20s %8.3f ms  %5.1f%%\n", "add_and_scale",    t_add_scale * 1000 / nthreads, 100.0 * t_add_scale / (total * nthreads));
    fprintf(stderr, "  %-20s %8.3f ms  %5.1f%%\n", "rope",             t_rope      * 1000 / nthreads, 100.0 * t_rope      / (total * nthreads));
    fprintf(stderr, "  %-20s %8.3f ms  %5.1f%%\n", "tanh (logits)",    t_other     * 1000 / nthreads, 100.0 * t_other     / (total * nthreads));
    double sum = (t_matmul + t_quantize + t_rmsnorm + t_attention + t_softmax +
                  t_geglu + t_embedding + t_add_scale + t_rope + t_other) / nthreads;
    fprintf(stderr, "  %-20s %8.3f ms  %5.1f%%\n", "SUM (measured)",   sum       * 1000, 100.0 * sum / total);
    fprintf(stderr, "  %-20s %8.3f ms  %5.1f%%\n", "wall clock",       total     * 1000, 100.0);
    fprintf(stderr, "  %-20s %8.3f ms  %5.1f%%\n", "unaccounted",      (total - sum) * 1000, 100.0 * (total - sum) / total);
    fprintf(stderr, "  layers: %d\n", prof_layer_count);
    fprintf(stderr, "\n  prefill: %d tokens in %.2fs (%.1f tok/s)\n", prompt_tokens, t_pf_end - t_pf_start,
            (t_pf_end - t_pf_start) > 0 ? (double)prompt_tokens / (t_pf_end - t_pf_start) : 0);
    fprintf(stderr, "  decode:  %d tokens in %.2fs (%.2f tok/s)\n", generated, t_dec_end - t_dec_start,
            (t_dec_end - t_dec_start) > 0 ? (double)generated / (t_dec_end - t_dec_start) : 0);
}
