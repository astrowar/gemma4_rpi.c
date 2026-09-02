# gemma4_rpi.c

Gemma 4 E2B CPU inference in pure C.

> **Fork** of [ryanssenn/gemma4.c](https://github.com/ryanssenn/gemma4.c) — optimized for Raspberry Pi (aarch64/NEON) with int4 kernel support.

An educational project made to understand how LLM inference works. The full inference path is implemented without external libraries. Matrix weights can be stored as int8 (group of 64) or int4 (group of 32, two 4-bit values per byte); inputs to linear layers are dynamically quantized to int8 in either case.

<img width="800" height="339" alt="demo" src="https://github.com/user-attachments/assets/2de47c35-ee34-473e-9164-635c82377267" />

## Benchmark

Measured on an AMD Ryzen 7 7700 using the default native build.

| Implementation | Prefill (pp512) | Decode (tg128@d512) |
| -------------- | --------------: | ------------------: |
| gemma4.c (int8) | 632.84 ± 6.07 tok/s | 25.01 ± 0.07 tok/s |
| llama.cpp (Q8_0) | 262.16 ± 1.19 tok/s | 23.19 ± 0.03 tok/s |

Results are the mean ± sample standard deviation over 12 timed runs after one discarded warmup, with each runtime using its fastest tested thread count. Both were built natively for CPU and dynamically quantize matrix inputs to int8.

```bash
./run -m ./gemma4-E2B-int8.bin --bench 512 128
```

## Quick start

You need an OpenMP-capable C compiler and `make`. On x86, AVX2 is required for the SIMD kernels (otherwise the build falls back to portable scalar code). The int8 model takes about 5.0 GB (4.7 GiB) of disk space and the int4 model about 2.7 GB (2.6 GiB); 8 GB of RAM is recommended.

Clone the repository and download a ready-to-run model:

```bash
git clone https://github.com/astrowar/gemma4_rpi.c
cd gemma4_rpi.c
python3 -m pip install -U huggingface_hub
hf download QmogAI/gemma4-e2b-int8 gemma4-E2B-int8.bin --local-dir .
# or the smaller int4 variant:
hf download QmogAI/gemma4-e2b-int4 gemma4-E2B-int4.bin --local-dir .
```

On Linux:

```bash
make
./run -t 0 -n 256 "Why is the sky blue?"
# to use the int4 model:
./run -m ./gemma4-E2B-int4.bin -t 0 -n 256 "Why is the sky blue?"
```

On Windows, use a MinGW-w64 environment that provides `gcc`, OpenMP, and `make`:

```powershell
make win64 WINCC=gcc
.\run.exe -t 0 -n 256 "Why is the sky blue?"
```

### Kernel selection

The `Makefile` picks the kernel implementation automatically: AVX2/AVX-512 on x86, NEON on aarch64, and portable scalar code elsewhere. You can override it with `KERNELS=pure`, `KERNELS=avx2`, or `KERNELS=neon`. See `make info` to print the resolved configuration.

| Kernels | Files | Weights |
| ------- | ----- | ------- |
| `avx2` / `native` | `kernels_avx_int8.c` + `kernels_avx_int4.c` | int8, int4 |
| `neon` | `kernels_neon_int8.c` + `kernels_neon_int4.c` | int8, int4 |
| `pure` | `kernels_pure_int8.c` + `kernels_pure_int4.c` | int8, int4 |

## Options

- `-m` sets the model path. The default is `gemma4-E2B-int8.bin`.
- `-t` sets the temperature. The default is `1.0`. Use `0` for greedy decoding.
- `-n` sets the maximum number of tokens to generate. The default is `1,024`.
- `--bench` measures prefill and decode throughput.
- `--dump-logits` writes prompt logits as float32 binary data.

## Model

The C runtime cannot read the original checkpoint directly. `exporter.py` takes the tokenizer and language-model weights from the Hugging Face checkpoint and writes them in the exact layout used by `gemma4.c`.

Matrix weights are stored with FP16 scales in one of two modes:

- **int8** (default): one int8 per weight, one scale per group of 64 inputs. The resulting file is about 5.0 GB (4.7 GiB).
- **int4**: two 4-bit values per byte (zero point 8, range [−8, +7]), one scale per group of 32 inputs. The resulting file is about 2.7 GB (2.6 GiB).

In both modes, inputs to linear layers are dynamically quantized to int8 (in groups of 64 for int8, 32 for int4) while the rest of the activations remain float32.

To create a model file yourself:

```bash
python3 -m pip install -r requirements.txt
python3 exporter.py /path/to/gemma-4-E2B-it-qat-q4_0-unquantized -o ./gemma4-E2B-int8.bin
# or the int4 variant:
python3 exporter.py /path/to/gemma-4-E2B-it-qat-q4_0-unquantized --quant int4 -o ./gemma4-E2B-int4.bin
```

Python is only needed to export the model or run numerical validation. Once the `.bin` file exists, inference runs entirely through the C program.

## Numerical validation

The implementation is validated against the Hugging Face Transformers reference implementation running Google's unquantized [Gemma 4 E2B QAT checkpoint](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized) in BF16. Both implementations process the complete Éva Gauthier article from the WikiText-103 validation split under teacher forcing. The passage contains 2,387 text tokens, and the added BOS token brings the comparison to 2,388 model positions.

| Metric | Result |
| ------ | -----: |
| Top-1 agreement | 2,305 / 2,388 (96.5%) |
| Mean KL divergence | 0.005207 |

An exact match is not expected because gemma4.c uses int8 matrix weights and linear inputs while the reference runs the unquantized checkpoint in BF16. The output distributions nevertheless remain closely aligned.

Build the C runtime first (`make` or `make win64`). The complete validation peaks at about 10 GiB of RAM:

```bash
python3 validation.py
```

Pass a smaller token count for a quicker check:

```bash
python3 validation.py 64
```

`validation.py` uses the runtime's `--dump-logits` flag to collect float32 logits after each prompt position. The flag can also be used directly when comparing gemma4.c with another implementation:

```bash
./run -m ./gemma4-E2B-int8.bin --dump-logits "Why is the sky blue?" > logits.bin
```

## Repository contents

- `gemma4.h` — shared types, constants, and cross-module declarations.
- `tokenizer.c` — BPE encode/decode.
- `model.c` — model loading, memory mapping, tensor offset resolution.
- `kernels_avx_int8.c` — AVX2/AVX-512 int8 matmul, quantize, attention, GELU.
- `kernels_avx_int4.c` — AVX2 int4 matmul.
- `kernels_neon_int8.c` — ARM NEON int8 matmul, quantize, attention, GELU.
- `kernels_neon_int4.c` — ARM NEON int4 matmul.
- `kernels_pure_int8.c` — portable scalar int8 fallback.
- `kernels_pure_int4.c` — portable scalar int4 fallback.
- `transformer.c` — forward pass: embedding, layernorms, attention, MLP, logits.
- `generate.c` — sampling, prefill, generation loop, benchmark.
- `main.c` — CLI argument parsing, model loading, entry point.
- `exporter.py` converts the original checkpoint into the binary layout read by the C runtime.
- `validation.py` compares the runtime's logits with Hugging Face Transformers.
- `validation.txt` contains the WikiText-103 passage used for numerical validation.
- `win.c` and `win.h` provide the small Windows memory-mapping compatibility layer.
- `Makefile` builds the Linux or Windows executable.
