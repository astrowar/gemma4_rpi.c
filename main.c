#include "gemma4.h"

int main(int argc, char **argv) {
#ifdef _WIN32
    argv_utf8(&argc, &argv);
#endif
    const char *model_path = "gemma4-E2B-int8.bin";
    const char *prompt = "Why is the sky blue?";
    float temperature = 1.0f;
    int max_new_tokens = 1024;
    int benchmark_mode = 0, dump_logits = 0, stats = 0, prefill_tokens = 0, generated_tokens = 256;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) temperature = atof(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) max_new_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bench")) {
            benchmark_mode = 1;
            if (i + 1 < argc) prefill_tokens = atoi(argv[++i]);
            if (i + 1 < argc) generated_tokens = atoi(argv[++i]);
        }
        else if (!strcmp(argv[i], "--dump-logits")) dump_logits = 1;
        else if (!strcmp(argv[i], "--stats")) stats = 1;
        else prompt = argv[i];
    }
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    if (dump_logits) _setmode(_fileno(stdout), _O_BINARY);
#endif
    fprintf(stderr, "Loading model: %s\n", model_path);
    int fd = open(model_path, O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st)) { perror(model_path); return 1; }
    Model *model = mmap(NULL, (size_t)st.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    close(fd);
    if (model == MAP_FAILED) { perror("mmap"); return 1; }

    if (memcmp(model->magic, "MOG", 4) != 0) { fprintf(stderr, "bad model file\n"); return 1; }
    // Old int8 files predate the quant field and read back as 0; treat that as int8.
    if (model->quant == 0) model->quant = QUANT_INT8;
    if (model->quant != QUANT_INT8 && model->quant != QUANT_INT4) {
        fprintf(stderr, "unsupported quantization mode %d\n", (int)model->quant);
        munmap(model, (size_t)st.st_size);
        return 1;
    }
    Tensor *tensors = (Tensor *)&model->weights;
    for (size_t i = 0; i < sizeof(model->weights) / sizeof(*tensors); i++) {
        tensors[i].data = tensors[i].data ? (void *)((uint8_t *)model + (uintptr_t)tensors[i].data) : NULL;
        tensors[i].scales = tensors[i].scales ? (uint16_t *)((uint8_t *)model + (uintptr_t)tensors[i].scales) : NULL;
    }
    fprintf(stderr, "Model loaded: %.1f MB (%s weights)\n",
            (double)st.st_size / (1024.0 * 1024.0),
            model->quant == QUANT_INT4 ? "int4" : "int8");
    InferenceState *state = calloc(1, sizeof(*state));

    seed_rng();
    if (benchmark_mode) benchmark(model, state, prefill_tokens, generated_tokens);
    else generate(model, state, prompt, max_new_tokens, temperature, dump_logits, stats);
    free(state);
    munmap(model, (size_t)st.st_size);
    return 0;
}
