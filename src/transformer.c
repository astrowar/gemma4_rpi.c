#include "gemma4.h"

// Portable fp16 → float conversion (replaces the _cvtsh_ss SSE intrinsic).
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

// Dispatches a linear layer to the kernel matching the model's weight format.
// Activations are always dynamically int8-quantized; only the stored weights
// differ between int8 and int4.
void matmul_dispatch(Model *model, float *output, const int8_t *input_q,
                                   const float *input_scales, const Tensor *weight, size_t rows) {
    if (model->quant == QUANT_INT4) matmul_int4(output, input_q, input_scales, weight, rows);
    else matmul_int8(output, input_q, input_scales, weight, rows);
}

// Looks up packed embedding rows and dequantizes them directly without materializing the full embedding table.
// Supports both int8 (group 64) and int4 (group 32) weight formats.
void embedding(float *output, const Tensor *table, const int *tokens, size_t token_count, float multiplier) {
    const int block_rows = 16;
    int width = table->shape[1];
    int groups = width / 64;
    #pragma omp for schedule(static)
    for (size_t token = 0; token < token_count; token++) {
        size_t block = (size_t)(tokens[token] / block_rows);
        int row = tokens[token] % block_rows;
        float *vector = output + token * width;
        const int8_t *block_data = (const int8_t *)table->data + block * block_rows * width;
        const uint16_t *block_scales = table->scales + block * groups * block_rows;
        for (size_t group_index = 0; group_index < (size_t)groups; group_index++) {
            const int8_t *group = block_data + group_index * block_rows * 64;
            float scale = fp16_to_f32(block_scales[group_index * block_rows + row]) * multiplier;
            for (int j = 0; j < 64; j++) {
                int chunk = j / 4;
                int offset = j % 4;
                vector[group_index * 64 + j] = (float)group[chunk * block_rows * 4 + row * 4 + offset] * scale;
            }
        }
    }
}

// int4 variant: embeddings are packed two 4-bit values per byte with group 32.
void embedding_int4(float *output, const Tensor *table, const int *tokens, size_t token_count, float multiplier) {
    const int block_rows = 16;
    int width = table->shape[1];
    int groups = width / 32;
    #pragma omp for schedule(static)
    for (size_t token = 0; token < token_count; token++) {
        size_t block = (size_t)(tokens[token] / block_rows);
        int row = tokens[token] % block_rows;
        float *vector = output + token * width;
        const uint8_t *block_data = (const uint8_t *)table->data + block * block_rows * (width / 2);
        const uint16_t *block_scales = table->scales + block * groups * block_rows;
        for (size_t group_index = 0; group_index < (size_t)groups; group_index++) {
            const uint8_t *group = block_data + group_index * block_rows * 16;
            float scale = fp16_to_f32(block_scales[group_index * block_rows + row]) * multiplier;
            for (int j = 0; j < 32; j++) {
                int byte = j / 2;
                int bit = (j % 2) * 4;
                int nibble = (group[row * 16 + byte] >> bit) & 0xf;
                vector[group_index * 32 + j] = (float)(nibble - 8) * scale;
            }
        }
    }
}

// Dispatches embedding lookup to the kernel matching the model's weight format.
void embedding_dispatch(Model *model, float *output, const Tensor *table,
                                      const int *tokens, size_t token_count, float multiplier) {
    if (model->quant == QUANT_INT4)
        embedding_int4(output, table, tokens, token_count, multiplier);
    else
        embedding(output, table, tokens, token_count, multiplier);
}

// Dispatches activation quantization to match the weight scale granularity.
// int8 weights use group 64; int4 weights use group 32.
void quantize_dispatch(Model *model, int8_t *output, float *scales,
                                     const float *input, size_t rows, size_t width) {
    if (model->quant == QUANT_INT4)
        quantize_int4(output, scales, input, rows, width);
    else
        quantize(output, scales, input, rows, width);
}

void rmsnorm(float *output, const float *input, const Tensor *weight, int width, float epsilon, size_t row_count) {
    const float *weights = weight ? (const float *)weight->data : NULL;
    #pragma omp for schedule(static)
    for (size_t row = 0; row < row_count; row++) {
        const float *input_row = input + row * width;
        float *output_row = output + row * width;
        float sum_squares = 0.0f;
        for (int i = 0; i < width; i++)
            sum_squares += input_row[i] * input_row[i];
        float inverse_rms = 1.0f / sqrtf(sum_squares / (float)width + epsilon);
        for (int i = 0; i < width; i++)
            output_row[i] = (weights ? weights[i] : 1.0f) * (inverse_rms * input_row[i]);
    }
}

void add_and_scale(float *output, const float *addend, size_t count, float scale) {
    #pragma omp for schedule(static)
    for (size_t i = 0; i < count; i++) output[i] = (output[i] + addend[i]) * scale;
}

// Rotates pairs of query or key channels using each position's sine and cosine values so attention can distinguish token order.
void apply_rope(const Tensor *cosines, const Tensor *sines, float *vectors,
                int num_heads, int head_dim, int start_pos, size_t token_count) {
    int pairs = cosines->shape[1];
    #pragma omp for schedule(static)
    for (size_t token = 0; token < token_count; token++) {
        const float *cosine = (float *)cosines->data + (start_pos + token) * pairs;
        const float *sine = (float *)sines->data + (start_pos + token) * pairs;
        for (size_t head = 0; head < (size_t)num_heads; head++) {
            float *vector = vectors + (token * num_heads + head) * head_dim;
            for (int j = 0; j < pairs; j++) {
                float first = vector[j];
                float second = vector[j + head_dim / 2];
                vector[j] = first * cosine[j] - second * sine[j];
                vector[j + head_dim / 2] = second * cosine[j] + first * sine[j];
            }
        }
    }
}

void softmax(float *values, int count) {
    float max = values[0], sum = 1.0f;
    for (int i = 1; i < count; i++) {
        if (values[i] > max) { sum = sum * expf(max - values[i]) + 1.0f; max = values[i]; } // Rescale the sum when a new maximum appears so expf() stays in range.
        else sum += expf(values[i] - max);
    }
    for (int i = 0; i < count; i++) values[i] = expf(values[i] - max) / sum;
}

// Builds queries, updates the KV cache, and computes causal attention over 512 tokens or the full context while shared layers reuse the latest compatible cache.
void attention(Model *model, InferenceState *state, const LayerWeights *layers, int layer,
               int start_pos, size_t token_count, float *scores) {
    const LayerWeights *weights = &layers[layer];
    int full_attention = layer % 5 == 4; // Every fifth layer uses full attention.
    int cache_len = full_attention ? MAX_CONTEXT : SLIDING_WINDOW + BATCH_SIZE;
    int cache_mask = cache_len - 1; // Both cache lengths are powers of two, so masking wraps positions without division.
    int head_dim = weights->q_norm.shape[0];
    int query_width = weights->q_proj.shape[0];
    int cache_owner = layer;
    while (!layers[cache_owner].k_proj.data || (cache_owner % 5 == 4) != full_attention) cache_owner--; // Shared layers reuse the latest cache of the same attention type.
    float *key_cache = full_attention ? state->full_cache[cache_owner / 5] : state->sliding_cache[cache_owner / 5][cache_owner % 5];
    float *value_cache = key_cache + (size_t)cache_len * head_dim;

    quantize_dispatch(model, state->quantized, state->activation_scales, state->hidden, token_count, weights->q_proj.shape[1]);
    matmul_dispatch(model, state->auxiliary, state->quantized, state->activation_scales, &weights->q_proj, token_count);
    rmsnorm(state->auxiliary, state->auxiliary, &weights->q_norm, head_dim, 1e-6f, token_count * (query_width / head_dim));
    apply_rope(&weights->rope_cos, &weights->rope_sin, state->auxiliary, query_width / head_dim, head_dim, start_pos, token_count);

    if (weights->k_proj.data) {
        float *new_keys = key_cache + ((size_t)start_pos & cache_mask) * head_dim;
        float *new_values = value_cache + ((size_t)start_pos & cache_mask) * head_dim;
        matmul_dispatch(model, new_keys, state->quantized, state->activation_scales, &weights->k_proj, token_count);
        matmul_dispatch(model, new_values, state->quantized, state->activation_scales, &weights->v_proj, token_count);
        rmsnorm(new_keys, new_keys, &weights->k_norm, head_dim, 1e-6f, token_count);
        rmsnorm(new_values, new_values, NULL, head_dim, 1e-6f, token_count); // Value vectors are normalized without a learned weight.
        apply_rope(&weights->rope_cos, &weights->rope_sin, new_keys, 1, head_dim, start_pos, token_count);
    }

    #pragma omp for collapse(2) schedule(dynamic, 1)
    for (size_t head = 0; head < (size_t)(query_width / head_dim); head++) {
        for (size_t token = 0; token < token_count; token++) {
            int first_key = !full_attention && start_pos + (int)token + 1 > SLIDING_WINDOW ? start_pos + (int)token + 1 - SLIDING_WINDOW : 0;
            int num_keys = start_pos + (int)token + 1 - first_key;
            float *head_output = state->hidden + token * query_width + head * head_dim;
            const float *query = state->auxiliary + token * query_width + head * head_dim;
            attention_scores(scores, query, key_cache, first_key, num_keys, cache_mask, head_dim);
            softmax(scores, num_keys);
            weighted_value_sum(head_output, scores, value_cache, first_key, num_keys, cache_mask, head_dim);
        }
    }

    quantize_dispatch(model, state->quantized, state->activation_scales, state->hidden, token_count, weights->o_proj.shape[1]);
    matmul_dispatch(model, state->hidden, state->quantized, state->activation_scales, &weights->o_proj, token_count);
}

void forward(Model *model, InferenceState *state, const int *tokens, size_t token_count, int start_pos) {
    int per_layer_width = model->weights.per_layer_projection_norm.shape[0];
    // One OpenMP team stays alive for the full forward pass while each kernel divides its own loop.
    #pragma omp parallel num_threads(thread_count())
    {
    float scores[(size_t)start_pos + token_count]; // Each thread needs private scratch large enough for every visible key.
    embedding_dispatch(model, state->residual, &model->weights.embed, tokens, token_count, sqrtf((float)HIDDEN_SIZE));

    // Replace audio placeholder embeddings with pre-computed soft tokens.
    if (state->audio_embeds && state->audio_count > 0) {
        for (size_t i = 0; i < token_count; i++) {
            int abs_pos = start_pos + (int)i;
            if (tokens[i] == AUDIO_TOKEN_ID && abs_pos >= state->audio_start && abs_pos < state->audio_start + state->audio_count) {
                int embed_idx = abs_pos - state->audio_start;
                memcpy(state->residual + i * HIDDEN_SIZE,
                       state->audio_embeds + (size_t)embed_idx * HIDDEN_SIZE,
                       HIDDEN_SIZE * sizeof(float));
            }
        }
    }

    // Build the token-conditioned input that each transformer layer will receive.
    quantize_dispatch(model, state->quantized, state->activation_scales, state->residual, token_count, HIDDEN_SIZE);
    matmul_dispatch(model, state->per_layer_inputs, state->quantized, state->activation_scales, &model->weights.per_layer_model_projection, token_count);
    rmsnorm(state->per_layer_inputs, state->per_layer_inputs, &model->weights.per_layer_projection_norm, per_layer_width, 1e-6f * HIDDEN_SIZE, token_count * NUM_LAYERS);

    embedding_dispatch(model, state->hidden, &model->weights.embed_per_layer, tokens, token_count, sqrtf((float)per_layer_width));
    add_and_scale(state->per_layer_inputs, state->hidden, token_count * NUM_LAYERS * per_layer_width,
               1.0f / sqrtf(2.0f)); // Keeps the variance of the combined input stable.

    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        LayerWeights *weights = &model->weights.layers[layer];

        rmsnorm(state->hidden, state->residual, &weights->input_layernorm, HIDDEN_SIZE, 1e-6f, token_count);
        attention(model, state, model->weights.layers, layer, start_pos, token_count, scores);
        rmsnorm(state->hidden, state->hidden, &weights->post_attn_layernorm, HIDDEN_SIZE, 1e-6f, token_count);
        add_and_scale(state->residual, state->hidden, token_count * HIDDEN_SIZE, 1.0f);

        rmsnorm(state->hidden, state->residual, &weights->pre_ffn_layernorm, HIDDEN_SIZE, 1e-6f, token_count);
        quantize_dispatch(model, state->quantized, state->activation_scales, state->hidden, token_count, weights->gate_proj.shape[1]);
        matmul_dispatch(model, state->hidden, state->quantized, state->activation_scales, &weights->gate_proj, token_count);
        matmul_dispatch(model, state->auxiliary, state->quantized, state->activation_scales, &weights->up_proj, token_count);
        geglu(state->hidden, state->auxiliary, token_count, weights->gate_proj.shape[0], weights->gate_proj.shape[0], &model->weights.gelu_table);
        quantize_dispatch(model, state->quantized, state->activation_scales, state->hidden, token_count, weights->down_proj.shape[1]);
        matmul_dispatch(model, state->hidden, state->quantized, state->activation_scales, &weights->down_proj, token_count);
        rmsnorm(state->hidden, state->hidden, &weights->post_ffn_layernorm, HIDDEN_SIZE, 1e-6f, token_count);
        add_and_scale(state->residual, state->hidden, token_count * HIDDEN_SIZE, 1.0f);
        // Gate and project this layer's conditioning input before adding it to the residual stream with a learned scale.
        quantize_dispatch(model, state->quantized, state->activation_scales, state->residual, token_count, HIDDEN_SIZE);
        matmul_dispatch(model, state->hidden, state->quantized, state->activation_scales, &weights->per_layer_input_gate, token_count);
        geglu(state->hidden, state->per_layer_inputs + layer * per_layer_width, token_count, per_layer_width, NUM_LAYERS * per_layer_width, &model->weights.gelu_table);
        quantize_dispatch(model, state->quantized, state->activation_scales, state->hidden, token_count, weights->per_layer_projection.shape[1]);
        matmul_dispatch(model, state->hidden, state->quantized, state->activation_scales, &weights->per_layer_projection, token_count);
        rmsnorm(state->hidden, state->hidden, &weights->post_per_layer_input_norm, HIDDEN_SIZE, 1e-6f, token_count);
        add_and_scale(state->residual, state->hidden, token_count * HIDDEN_SIZE,
                   ((float *)weights->layer_scalar.data)[0]);
    }
    }
}

// Reuses the embedding matrix to turn the final token representation into vocabulary logits, then applies Gemma's tanh soft cap.
float *logits(Model *model, InferenceState *state, size_t token) {
    #pragma omp parallel num_threads(thread_count())
    {
        rmsnorm(state->hidden, state->residual + token * HIDDEN_SIZE, &model->weights.norm, HIDDEN_SIZE, 1e-6f, 1);
        quantize_dispatch(model, state->quantized, state->activation_scales, state->hidden, 1, HIDDEN_SIZE);
        matmul_dispatch(model, state->hidden, state->quantized, state->activation_scales, &model->weights.embed, 1);
        #pragma omp for schedule(static)
        for (int i = 0; i < VOCAB_SIZE; i++) state->hidden[i] = 30.0f * tanhf(state->hidden[i] / 30.0f);
    }
    return state->hidden;
}
