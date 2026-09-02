// Profile main: loads model, runs prof_generate with per-component timing.
#include "gemma4.h"
#include <time.h>

// Declared in profile.c
void prof_generate(Model *model, InferenceState *state, const char *prompt,
                  int max_new_tokens, float temperature);

int main(int argc, char **argv) {
    const char *model_path = "gemma4-E2B-int8.bin";
    const char *prompt = "What is the capital of France?";
    float temperature = 0.0f;
    int max_new_tokens = 8;

    for (int i = 1; i < argc; i++) {
        if ((!strcmp(argv[i], "-m") || !strcmp(argv[i], "-t")) && i + 1 < argc) {
            if (!strcmp(argv[i], "-m")) model_path = argv[++i];
            else temperature = atof(argv[++i]);
        }
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) max_new_tokens = atoi(argv[++i]);
        else prompt = argv[i];
    }

    fprintf(stderr, "Loading model: %s\n", model_path);
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
    fprintf(stderr, "Model loaded: %.1f MB\n", (double)st.st_size / (1024.0 * 1024.0));
    InferenceState *state = calloc(1, sizeof(*state));

    seed_rng();
    omp_init();
    prof_generate(model, state, prompt, max_new_tokens, temperature);
    free(state);
    munmap(model, (size_t)st.st_size);
    return 0;
}
