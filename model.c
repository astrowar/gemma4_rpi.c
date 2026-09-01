#include "gemma4.h"

// Opens the model file, memory-maps it, validates the magic bytes, and resolves
// tensor file offsets into real pointers.
Model *model_load(const char *path) {
    int fd = open(path, O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st)) { perror(path); return NULL; }
    Model *model = mmap(NULL, (size_t)st.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    close(fd);
    if (model == MAP_FAILED) { perror("mmap"); return NULL; }

    if (memcmp(model->magic, "MOG", 4) != 0) {
        fprintf(stderr, "bad model file\n");
        munmap(model, (size_t)st.st_size);
        return NULL;
    }

    Tensor *tensors = (Tensor *)&model->weights;
    for (size_t i = 0; i < sizeof(model->weights) / sizeof(*tensors); i++) {
        tensors[i].data = tensors[i].data ? (void *)((uint8_t *)model + (uintptr_t)tensors[i].data) : NULL;
        tensors[i].scales = tensors[i].scales ? (uint16_t *)((uint8_t *)model + (uintptr_t)tensors[i].scales) : NULL;
    }
    return model;
}

void model_unload(Model *model, size_t size) {
    munmap(model, size);
}
