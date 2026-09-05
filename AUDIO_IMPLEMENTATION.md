# Gemma 4 E2B — Audio Processing Implementation

## Overview

Pure C audio encoder for the Gemma 4 E2B text model. Takes a WAV file, runs it through
a 12-layer conformer audio encoder, and injects soft tokens into the text model's
input embeddings at `AUDIO_TOKEN_ID` (258881) positions.

```
WAV (16kHz mono)
  → Mel spectrogram (128 bins, 10ms hop)
  → SSCP (2× Conv2d stride-2 + Linear)
  → 12× Conformer layers (FFN → Attn → LConv1d → FFN)
  → output_proj (1024→1536, int8) + bias
  → RMSNorm(no scale) + Linear(1536→1536, float32)
  → Soft tokens [seq_len, 1536]
  → Replace AUDIO_TOKEN_ID positions in inputs_embeds
  → Text model continues as normal
```

## Architecture Parameters (E2B)

| Parameter | Value |
|-----------|-------|
| Conformer layers | 12 |
| Hidden size | 1024 |
| Attention heads | 8 |
| Head dim | 128 |
| FFN intermediate | 4096 |
| Output projection | 1536 |
| Mel bins | 128 |
| FFT length | 512 |
| Frame length | 320 (20ms) |
| Hop length | 160 (10ms) |
| Chunked attention | chunk=12, past=12, future=0, context=24 |
| Soft-cap on logits | 50.0 |
| Depthwise conv1d kernel | 5 |
| SSCP: Conv2d 1→128 (k=3, s=2) + LN + ReLU | |
| SSCP: Conv2d 128→32 (k=3, s=2) + LN + ReLU | |
| SSCP: Linear(1024→1024) | |
| Max audio duration | ~25.6s (4096 mel frames → 1024 soft tokens) |
| 4× temporal reduction | 10ms → 40ms per soft token |

## File Structure

```
gemma4.c/
├── audio.h                  # Struct definitions (AudioModel, AudioLayer, etc.)
├── src/audio.c              # Full audio encoder implementation (~890 lines)
├── src/transformer.c        # Audio embedding injection in forward()
├── src/generate.c           # generate_audio() — builds prompt, prefill, decode
├── src/main.c               # CLI flags: -a <wav>, -A <audio_model.bin>
├── Makefile                 # Includes src/audio.c in SRCS
├── tools/exporter.py        # --audio-out flag exports audio weights
├── tools/validate_audio.py  # Validation script (C vs PyTorch)
├── gemma4-E2B-int8.bin      # Text model (4.8 GB)
└── gemma4-E2B-int8-audio.bin # Audio model (345 MB, MOGA magic)
```

## Binary Format

### Header (raw struct dump, 147,264 bytes)

The file is a raw binary of the `AudioModel` C struct followed by tensor data.
The C loader mmaps the file and patches tensor offsets in-place.

```
Offset  0:    char magic[4]       "MOGA"
Offset  4:    int quant           8 (int8)
Offset  8:    int version         1
Offset  12:   float hann[512]     320-sample Hann + 192 zeros (for FFT-512)
Offset 2060:  float mel[257][128] HTK mel filterbank (0-8kHz)
Offset 133644: float mean[128]    per-bin mean (from config, or 0)
Offset 134156: float std[128]     per-bin stddev (from config, or 1)
Offset 134672: AudioSSCP          (192 bytes)
Offset 134864: AudioLayer[12]     (12 × 1024 bytes)
Offset 147152: Tensor output_proj [1536, 1024] int8
Offset 147184: Tensor output_bias [1536] float32
Offset 147216: Tensor embed_proj  [1536, 1536] float32
Offset 147248: int boa_token_id   256000
Offset 147252: int eoa_token_id   258883
Offset 147256: int audio_token_id 258881
Offset 147264: [tensor data region]
```

### Tensor (32 bytes)

```c
typedef struct {
    uint64_t data;      // file offset → patched to pointer
    uint64_t scales;    // file offset for int8 scales → patched to pointer
    int shape[4];       // dimensions
} Tensor;
```

### ClippableLinear (64 bytes)

```c
typedef struct {
    Tensor weight;           // int8 packed (32 bytes)
    uint64_t _pad1;          // MUST be 0 (loader stepper reads as .data)
    uint64_t _pad2;          // MUST be 0 (loader stepper reads as .scales)
    float input_min, input_max, output_min, output_max;  // at offset 48
} ClippableLinear;
```

**Critical:** The loader steps through all 32-byte slots from `sscp` to `boa_token_id`.
At offset +32 within a ClippableLinear, it reads `_pad1`/`_pad2` as `.data`/`.scales`.
These must be zero so the null-check skips them. The clip floats at offset +48 are
read as `.shape[4]` (ignored by the loader).

### Tensor Data Region

After the struct, all tensor data is stored sequentially with 64-byte alignment:
- int8 packed weights (same layout as text model: block-rows, groups)
- uint16 scales (one per 32 elements for int8)
- float32 arrays (convs, norms, relative_k_proj, etc.)

## C Implementation Details

### Loading (`audio_load`)

1. `mmap(PROT_READ|PROT_WRITE, MAP_PRIVATE)` the entire file
2. Check magic "MOGA"
3. Iterate all 32-byte slots from `&model->sscp` to `&model->boa_token_id`
4. For each slot: if `.data != 0`, patch to `base + offset`; same for `.scales`

### Mel Spectrogram (`audio_mel_spectrogram`)

Matches `transformers.Gemma4AudioFeatureExtractor._extract_spectrogram`:

- Semicausal padding: `frame_len/2` zeros at start
- Frame: 320 samples, Hop: 160, FFT: 512 (zero-padded to 512)
- Periodic Hann window applied to 320 samples
- In-place radix-2 FFT (iterative, bit-reversal)
- **Magnitude** spectrum: `sqrt(re² + im²)` (not power)
- Mel filterbank: slope-based triangular (matches `transformers.audio_utils.mel_filter_bank`)
- `log(mel + 1e-3)` (additive floor, not clamp)
- Per-bin normalize: `(log_val - mean) / std` (defaults: mean=0, std=1 → no-op)

### SSCP (`audio_sscp`)

- Input: mel `[T, 128]` → treated as image `[H=T, W=128]`
- Conv2d(1→128, k=3, s=2, pad=1) → LayerNorm(128) → ReLU
- Conv2d(128→32, k=3, s=2, pad=1) → LayerNorm(32) → ReLU
- Permute `[C, H, W]` → `[H, W*C]` (reshape: `[H2, 32*32=1024]`)
- Linear(1024→1024) via int8 matmul
- Output: `[H2, 1024]` where H2 ≈ T/4

### Conformer Layer (×12)

```
FFN1 (residual=0.5):
  pre_norm → ffw1 [1024→4096] → SiLU → ffw2 [4096→1024] → post_norm * 0.5 + residual

Attention (chunked local):
  norm_pre_attn → Q,K,V proj [1024→1024] int8
  Scale Q by q_scale * softplus(per_dim_scale), K by k_scale
  Build block contexts (chunk=12, past=12): ctx_k, ctx_v [blocks, 8, 24, 128]
  Relative position: sin/cos → relative_k_proj [1024×1024] float32 → [13, 8, 128]
  Scores: Q·K + Q·rel_K → softcap(50) → softmax → ·V
  Post proj [1024→1024] int8
  norm_post_attn + residual

LConv1d:
  pre_norm → linear_start [1024→2048] → GLU(SiLU) → causal depthwise_conv(k=5)
  → conv_norm → SiLU → linear_end [1024→1024] → residual

FFN2 (residual=0.5): same as FFN1

Final: norm_out (RMSNorm with scale)
```

### Output Projection + Embedder

```
output_proj: [seq, 1024] → int8 matmul → [seq, 1536] + bias
RMSNorm(no scale): [seq, 1536]
embed_proj: [seq, 1536] @ [1536, 1536]^T float32 → [seq, 1536] soft tokens
```

### Text Model Integration

In `transformer.c forward()`:
- `state->audio_embeds` holds the soft tokens
- For each token position with ID == 258881, replace embedding with the
  corresponding soft token vector

In `generate.c generate_audio()`:
- Builds prompt: `<BOA>` + N×`<AUDIO_TOKEN>` + `<EOA>` + user_prompt
- N = number of soft tokens from encoder
- Runs prefill + decode as normal

## Export (tools/exporter.py)

```bash
python3 tools/exporter.py <checkpoint> --audio-out ./gemma4-E2B-int8-audio.bin --quant int8
```

Process:
1. Load config, validate audio params
2. Open safetensors checkpoint
3. Quantize all 2D linear weights to int8 (block-128, group-64)
4. Pack conv/norm weights as flat float32
5. Compute Hann window (320-sample periodic, zero-padded to 512)
6. Compute HTK mel filterbank (257×128) using slope-based triangular construction
   matching `transformers.audio_utils.mel_filter_bank` (not bin-index quantization)
7. Read per_bin_mean/stddev from `audio_preprocessor_config.json` (or default 0/1)
8. Build 147,264-byte struct with all tensor offsets
9. Write struct + tensor data to file

## Token IDs

| Token | ID | Purpose |
|-------|----|---------|
| BOA | 256000 | Begin of Audio |
| AUDIO_TOKEN | 258881 | Placeholder replaced by soft token |
| EOA | 258883 | End of Audio |

## WAV Loading

- Supports: 16-bit PCM mono, 32-bit float mono
- Resampling: linear interpolation to 16kHz (handles 44100Hz, 22050Hz, etc.)
- Max: ~25.6 seconds (4096 mel frames)

## Usage

```bash
./run -m gemma4-E2B-int8.bin \
      -A gemma4-E2B-int8-audio.bin \
      -a /path/to/audio.wav \
      -t 0 -n 128 \
      "What did the person say in this audio?"
```

## Current Status

### Working
- [x] Binary format (MOGA) with correct struct layout
- [x] C loader with 32-byte tensor stepping
- [x] Exporter produces valid binary (1172 MB float32, 272 tensors)
- [x] WAV loading (strict: 16kHz mono, 16-bit PCM or 32-bit float, no resampling)
- [x] Mel spectrogram (verified: max diff 0.002 vs PyTorch, cos=0.99999988)
- [x] SSCP convs + LayerNorm + ReLU (verified: max diff 0.004 vs PyTorch, cos=1.0)
- [x] SSCP input_proj (float32, verified: max diff 0.001, cos=1.0)
- [x] Conformer FFN1 (float32, verified: max diff 0.002, cos=1.00000012)
- [ ] Conformer attention (chunked local + rel_shift) — **IN PROGRESS** (cos=0.707)
- [ ] Conformer LConv1d
- [ ] Conformer FFN2
- [ ] Output projection + multimodal embedder
- [ ] Audio embedding injection in text model
- [x] No ASan errors (heap/stack overflow fixed)
- [x] End-to-end run: 6s WAV → 151 soft tokens → text generation (output quality broken)

### Mel Spectrogram Fix (2026-09-04)

Three bugs caused the ~12.7 max-diff mismatch vs PyTorch:

1. **Mel filterbank formula** (exporter.py): The old code used bin-index
   quantization with a wrong scaling factor (`floor(hz * 258 / sr)`), which
   compressed the entire filterbank into 0–4 kHz. Fixed to use the same
   slope-based triangular construction as `transformers.audio_utils.mel_filter_bank`:
   `fft_freqs = linspace(0, sr/2, 257)`, slopes computed in frequency space.

2. **Power vs Magnitude** (audio.c): The C code used `re² + im²` (power),
   but transformers uses `np.abs(stft)` = `sqrt(re² + im²)` (magnitude).
   Fixed to use `sqrtf(...)`.

3. **Log floor** (audio.c): The C code used `log(max(val, 1e-3))` (clamp),
   but transformers uses `log(val + 1e-3)` (additive floor). Fixed to `logf(sum + 1e-3f)`.

4. **Resampling mismatch** (audio.c): The C resampler (linear interp) produced
   different samples than Python's `np.interp`, causing per-frame drift.
   Fixed by requiring 16kHz mono WAV input (no resampling in C).

### SSCP input_proj int8 issue (2026-09-04)

The SSCP pre-linear features (after 2× conv + LN + ReLU) have a skewed
distribution: max=25.2, 50% zeros, mean|val|=0.91. Per-group-64 int8
quantization of this input produces cos=0.924 vs float32 (43% relative error),
which is too large for the model to tolerate.

Fix: store input_proj weight as float32 and use float32 matmul for this layer.

### Conformer layer fixes (2026-09-04)

Multiple bugs found while validating against PyTorch reference
(`tools/reference_encoder.py`):

1. **FFN residual aliasing** (audio.c): The old code wrote the residual into
   `buf` (which is `state->ffn_out`), then immediately called `audio_linear`
   which overwrote `buf` with the ffw1 output — destroying the residual.
   Fixed by using a separate `mid` buffer for the FFN intermediate and
   reading the residual from `input` directly.

2. **rel_shift indexing** (audio.c): The old code used
   `rel_idx = c - AUDIO_PAST_HORIZON + pos_in_block`. The correct formula
   (matching PyTorch's `_rel_shift`) is `rel_idx = c - pos_in_block`.

3. **Out-of-range relative positions** (audio.c): When `rel_idx` is out of
   `[0, CTX_SIZE//2]`, the old code set the attention score to -1e9 (killing
   the softmax probability). The correct behavior is to simply add 0 (no
   relative position bias for that context slot).

4. **Int8 weight error too large for attention**: Per-group-64 int8
   quantization of audio hidden states produces ~3% cosine error in Q/K/V
   projections. This is tolerable for FFN (cos=0.993) but destroys attention
   (cos=0.74) because scores are sensitive to small input perturbations.
   Fixed by storing ALL audio linear weights as float32 (binary: 345→1172 MB).

### Known Issues
- [ ] **Conformer attention** produces cos=0.707 vs PyTorch — bug still present
  - FFN1 is perfect (cos=1.0), so the bug is isolated to the attention path
  - Q/K/V projections verified correct (Q cos=0.972 with int8, should be ~1.0 now)
  - Suspected: context window construction or rel_shift logic still has an error
- [ ] **Conformer LConv1d** — not yet validated
- [ ] **End-to-end output quality** — needs re-testing after attention fix

### Next Steps
1. ~~Fix mel spectrogram to match PyTorch exactly~~ ✓ (done 2026-09-04)
2. ~~Verify per_bin_mean/stddev from the actual preprocessor config~~ ✓ (not in config → defaults to 0/1, no-op)
3. ~~Compare SSCP output with int8 vs float32 reference~~ ✓ (done: int8 cos=0.92, need float32)
4. ~~Switch SSCP input_proj to float32 matmul~~ ✓ (done 2026-09-04)
5. ~~Fix FFN residual aliasing~~ ✓ (done 2026-09-04)
6. ~~Fix rel_shift indexing + out-of-range handling~~ ✓ (done 2026-09-04)
7. ~~Switch all audio linears to float32~~ ✓ (done 2026-09-04, binary 1172 MB)
8. **Debug conformer attention** — compare C vs PT attention output
   - Dump matrix_ac, matrix_bd, attn_weights from both C and PyTorch
   - Check context window construction (pos = b*CHUNK + c - PAST_HORIZON)
9. Validate LConv1d + FFN2 + output_proj
10. Once encoder matches, test full pipeline with known transcriptions

## Debug Tools

```bash
# Build with AddressSanitizer
cc -std=c11 -g -O1 -fopenmp -I. -march=native -fsanitize=address \
   src/*.c -o run_asan -lm -fsanitize=address

# Dump intermediate tensors
AUDIO_DEBUG=1 ./run -m ... -a ... -A ... -t 0 -n 1 "test"
# Produces:
#   /tmp/c_mel.bin           mel spectrogram [T, 128]
#   /tmp/c_sscp.bin          SSCP output [T/4, 1024]
#   /tmp/c_sscp_prelin.bin   SSCP pre-linear features [T/4, 1024]
#   /tmp/c_layer0.bin        conformer layer 0 output [T/4, 1024]
#   /tmp/c_l0_ffn1.bin       layer 0 FFN1 output [T/4, 1024]
#   /tmp/c_l0_postattn.bin   layer 0 post-attention [T/4, 1024]
#   /tmp/c_l0_lconv.bin      layer 0 LConv1d output [T/4, 1024]
#   /tmp/c_attn_q.bin        layer 0 scaled Q [T/4, 1024]
#   /tmp/c_soft_tokens.bin   final soft tokens [T/4, 1536]

# PyTorch reference: mel + SSCP
python3 tools/validate_audio.py /path/to/audio.wav
# Produces: /tmp/mel_torch.pt, /tmp/sscp_ref.pt

# PyTorch reference: full conformer encoder + output_proj
python3 tools/reference_encoder.py /tmp/sscp_ref.pt
# Produces: /tmp/encoder_ref.pt [T/4, 1536], /tmp/layer0_ref.pt [T/4, 1024]
```

## Key Constants

```c
#define AUDIO_HIDDEN     1024
#define AUDIO_HEADS      8
#define AUDIO_HEAD_DIM   128
#define AUDIO_LAYERS     12
#define AUDIO_FFN        4096
#define AUDIO_OUTPUT     1536
#define AUDIO_MEL_BINS   128
#define AUDIO_FFT_LEN    512
#define AUDIO_FRAME_LEN  320
#define AUDIO_HOP_LEN    160
#define AUDIO_MAX_TOKENS 750
#define AUDIO_CHUNK      12
#define AUDIO_PAST_HORIZON 12
#define AUDIO_CONTEXT    24
#define AUDIO_CONV_KERNEL 5
#define AUDIO_TOKEN_ID   258881
#define AUDIO_BOA_ID     256000
#define AUDIO_EOA_ID     258883
```
