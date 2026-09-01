#include "gemma4.h"

unsigned long long rng_state = 42;

double time_seconds(void) {
#ifdef _WIN32
    static double frequency;
    LARGE_INTEGER ticks;
    if (!frequency) {
        QueryPerformanceFrequency(&ticks);
        frequency = (double)ticks.QuadPart;
    }
    QueryPerformanceCounter(&ticks);
    return (double)ticks.QuadPart / frequency;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
#endif
}

void seed_rng(void) {
    rng_state = (unsigned long long)(time_seconds() * 1e9);
}

float random_uniform(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (float)((rng_state * 0x2545F4914F6CDD1DULL) >> 40) / 16777216.0f;
}

int sample(float *logits, int vocab_size, float temperature) {
    if (temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < vocab_size; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    struct { float score; int token; } top[64]; // Sampling considers only the 64 highest logits.
    for (int i = 0; i < 64; i++) top[i].score = -INFINITY;
    for (int token = 0; token < vocab_size; token++) {
        if (logits[token] <= top[63].score) continue;
        int i = 63;
        while (i > 0 && logits[token] > top[i - 1].score) { top[i] = top[i - 1]; i--; }
        top[i].score = logits[token]; top[i].token = token;
    }
    float sum = 0.0f, max = top[0].score / temperature;
    for (int i = 0; i < 64; i++) sum += top[i].score = expf(top[i].score / temperature - max);
    float mass = 0.0f;
    int count = 0;
    while (mass < 0.95f * sum) mass += top[count++].score; // Keep the smallest prefix containing 95% of the top-64 probability mass.
    float threshold = random_uniform() * mass;
    for (int i = 0; i < count; i++)
        if ((threshold -= top[i].score) <= 0.0f) return top[i].token;
    return top[count - 1].token;
}

void prefill(Model *model, InferenceState *state, const int *tokens, int token_count, int dump_logits) {
    for (int position = 0; position < token_count; position += BATCH_SIZE) {
        int chunk = token_count - position < BATCH_SIZE ? token_count - position : BATCH_SIZE;
        forward(model, state, tokens + position, chunk, position);
        if (dump_logits) {
            for (int i = 0; i < chunk; i++) {
                fwrite(logits(model, state, i), sizeof(float), VOCAB_SIZE, stdout);
            }
        }
    }
}

void generate(Model *model, InferenceState *state, const char *prompt,
              int max_new_tokens, float temperature, int dump_logits, int stats) {
    Tokenizer *tokenizer = &model->tokenizer;
    int styled = !dump_logits && isatty(STDOUT_FILENO);

    if (max_new_tokens < 0) {
        fprintf(stderr, "-n must be non-negative\n");
        exit(1);
    }
    const char *segments[3] = {dump_logits ? "" : "<|turn>user\n", prompt,
                               dump_logits ? "" : "<turn|>\n<|turn>model\n"};
    int prompt_tokens = tokenize(tokenizer, segments, state->token_ids, MAX_CONTEXT);
    if (prompt_tokens < 0) {
        fprintf(stderr, "prompt exceeds the %d-token context limit\n", MAX_CONTEXT);
        exit(1);
    }

    if (styled) {
        fputs("\n\033[2;36m────────────────────────────────\033[0m\n", stdout);
        fflush(stdout);
    }

    double t_prefill_start = time_seconds();
    prefill(model, state, state->token_ids, prompt_tokens, dump_logits);
    double t_prefill_end = time_seconds();
    if (dump_logits) return;

    int end = prompt_tokens + max_new_tokens;
    if (end > MAX_CONTEXT || end < prompt_tokens) end = MAX_CONTEXT;
    int generated = 0;
    double t_decode_start = time_seconds();
    for (int position = prompt_tokens; position < end; position++) {
        int next_token = sample(logits(model, state, position == prompt_tokens ? (prompt_tokens - 1) % BATCH_SIZE : 0), VOCAB_SIZE, temperature);
        if (next_token == 1 || next_token == 106) break; // Stop at <eos> or <turn|>.

        fputs(token_text(tokenizer, next_token), stdout);
        fflush(stdout);
        forward(model, state, &next_token, 1, position);
        generated++;
    }
    double t_decode_end = time_seconds();
    putchar('\n');

    if (stats) {
        double prefill_time = t_prefill_end - t_prefill_start;
        double decode_time = t_decode_end - t_decode_start;
        fprintf(stderr, "\n[stats] prefill: %d tokens in %.2fs (%.1f tok/s)\n",
                prompt_tokens, prefill_time,
                prefill_time > 0 ? (double)prompt_tokens / prefill_time : 0.0);
        fprintf(stderr, "[stats] decode:  %d tokens in %.2fs (%.2f tok/s)\n",
                generated, decode_time,
                decode_time > 0 ? (double)generated / decode_time : 0.0);
    }
}

void benchmark(Model *model, InferenceState *state, int prefill_tokens, int generated_tokens) {
    if (prefill_tokens > 0) {
        for (int i = 0; i < prefill_tokens; i++)
            state->token_ids[i] = 2 + i % 1000;
        double start = time_seconds();
        prefill(model, state, state->token_ids, prefill_tokens, 0);
        printf("pp%d %.2f tok/s\n", prefill_tokens, (double)prefill_tokens / (time_seconds() - start));
    }
    if (generated_tokens > 0) {
        int token = 2;
        double start = time_seconds();
        for (int position = prefill_tokens; position < prefill_tokens + generated_tokens; position++) {
            forward(model, state, &token, 1, position);
            token = sample(logits(model, state, 0), VOCAB_SIZE, 0.0f);
        }
        printf("tg%d@d%d %.2f tok/s\n", generated_tokens, prefill_tokens, (double)generated_tokens / (time_seconds() - start));
    }
}
