# Gemma 4 E2B — Audio Processing Implementation

## Overview

Pure C audio encoder for the Gemma 4 E2B text model. Takes a WAV file, runs it through
a 12-layer conformer audio encoder, and injects soft tokens into the text model's
input embeddings at `AUDIO_TOKEN_ID` (258881) positions.

The official token sequence (matching `transformers` pipeline) is:

```
<bos>  <|turn>user\n  [text instruction]  N×<|audio|>  <turn|>\n<|turn>model\n
```

i.e. the text instruction comes **before** the audio soft tokens, and no BOA/EOA
wrapping tokens are used.

```
WAV (16kHz mono)
  → Mel spectrogram (128 bins, 10ms hop)
  → SSCP (2× Conv2d stride-2 + Linear)
  → 12× Conformer layers (FFN → Attn → LConv1d → FFN)
  → output_proj (1024→1536, float32) + bias
  → RMSNorm(no scale) + Linear(1536→1536, float32)
  → Soft tokens [seq_len, 1536]
  → Replace AUDIO_TOKEN_ID positions in inputs_embeds (after text tokens)
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
├── src/audio.c              # Full audio encoder implementation (~1015 lines)
├── src/transformer.c        # Audio embedding injection in forward()
├── src/generate.c           # generate_audio() — builds prompt, prefill, decode
├── src/main.c               # CLI flags: -a <wav>, -A <audio_model.bin>
├── Makefile                 # Includes src/audio.c in SRCS
├── tools/exporter.py        # --audio-out flag exports audio weights
├── tools/validate_audio.py  # Validation script (mel + SSCP: C vs PyTorch)
├── tools/reference_encoder.py  # Full 12-layer conformer PyTorch reference
├── gemma4-E2B-int8.bin      # Text model (4.8 GB)
└── gemma4-E2B-int8-audio.bin # Audio model (1172 MB float32, MOGA magic)
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
5. Dequantize any int8 tensors (`.scales != NULL`) to float32 via
   `dequant_weight_f32()` — currently all linears are stored as float32 in the
   binary, so this path is not exercised

### Matrix Multiplication (`audio_linear_f64`)

All audio linear layers in the conformer loop use **full double-precision**
computation. Weights are stored as float32 in the binary but upcast to double
at the point of multiplication:

```c
for (int j = 0; j < out_dim; j++) {
    double sum = 0.0;
    for (int k = 0; k < in_dim; k++)
        sum += input[i*in_dim + k] * (double)w[j*in_dim + k];
    output[i*out_dim + j] = sum;
}
```

OpenMP-parallelized across rows. All intermediate activations (RMSNorm, SiLU,
softmax, GLU, etc.) also use double precision. This eliminates the chaotic
divergence that float32 intermediates caused across 12 layers. Result: cos=1.0
vs PyTorch at every layer.

The SSCP stage (mel → conv → input_proj) remains float32 since it's only 2
layers and matches PyTorch to <0.005 abs diff.

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
- Linear(1024→1024) via float32 matmul
- Output: `[H2, 1024]` where H2 ≈ T/4

### Conformer Layer (×12)

```
FFN1 (residual=0.5):
  pre_norm → ffw1 [1024→4096] → SiLU → ffw2 [4096→1024] → post_norm * 0.5 + residual

Attention (chunked local):
  norm_pre_attn → Q,K,V proj [1024→1024] float32
  Scale Q by q_scale * softplus(per_dim_scale), K by k_scale
  Build block contexts (chunk=12, past=12): ctx_k, ctx_v [blocks, 8, 24, 128]
  Relative position: sin/cos → relative_k_proj [1024×1024] float32 → [13, 8, 128]
  Scores: Q·K + Q·rel_K → softcap(50) → softmax → ·V
  Post proj [1024→1024] float32
  norm_post_attn + residual

LConv1d:
  pre_norm → linear_start [1024→2048] → GLU(a*sigmoid(b)) → causal depthwise_conv(k=5)
  → conv_norm → SiLU → linear_end [1024→1024] → residual

FFN2 (residual=0.5): same as FFN1

Final: norm_out (RMSNorm with scale)
```

### Output Projection + Embedder

```
output_proj: [seq, 1024] → float32 matmul → [seq, 1536] + bias
RMSNorm(no scale): [seq, 1536]
embed_proj: [seq, 1536] @ [1536, 1536]^T float32 → [seq, 1536] soft tokens
```

### Text Model Integration

In `transformer.c forward()`:
- `state->audio_embeds` holds the soft tokens
- For each token position with ID == 258881, replace embedding with the
  corresponding soft token vector

In `generate.c generate_audio()` — **current (incorrect) sequence:**
```
<|turn>user\n  <|audio>  N×<|audio|>  <audio|>  user_prompt  <turn|>\n<|turn>model\n
 [105][2364][107] [256000]  [258881×N]  [258883]  [text...]    [106][107][105][4368][107]
```

**Correct sequence (matching `transformers` pipeline):**
```
<bos>  <|turn>user\n  user_prompt  N×<|audio|>  <turn|>\n<|turn>model\n
 [2]   [105][2364][107]  [text...]    [258881×N]   [106][107][105][4368][107]
```

Key differences from the C implementation (see "Token Sequence Divergence" below):
- `<bos>` (token 2) is **required** at position 0
- **No** BOA (256000) or EOA (258883) tokens — transformers uses only N × 258881
- **Order is text → audio**, not audio → text
- The instruction text precedes the audio placeholders in the user turn

## Export (tools/exporter.py)

```bash
python3 tools/exporter.py <checkpoint> --audio-out ./gemma4-E2B-int8-audio.bin --quant int8
```

Process:
1. Load config, validate audio params
2. Open safetensors checkpoint
3. Store all 2D linear weights as float32 (int8 was too lossy for audio)
4. Pack conv/norm weights as flat float32
5. Compute Hann window (320-sample periodic, zero-padded to 512)
6. Compute HTK mel filterbank (257×128) using slope-based triangular construction
   matching `transformers.audio_utils.mel_filter_bank` (not bin-index quantization)
7. Read per_bin_mean/stddev from `audio_preprocessor_config.json` (or default 0/1)
8. Build 147,264-byte struct with all tensor offsets
9. Write struct + tensor data to file

Note: the `--quant` flag is accepted for CLI compatibility but currently all
audio weights are exported as float32 regardless.

## Token IDs

| Token | ID | Purpose | Used by `transformers` pipeline |
|-------|----|---------|-------------------------------|
| BOA `<\|audio\>` | 256000 | Begin of Audio marker | **No** — not emitted by the chat template |
| AUDIO_TOKEN `<\|audio\|>` | 258881 | Placeholder replaced by soft token | **Yes** — the only audio token used |
| EOA `<audio\|>` | 258883 | End of Audio marker | **No** — not emitted by the chat template |

The official `transformers` pipeline (see `apply_chat_template` with `{"type": "audio"}`
content parts) emits **only** N × token 258881 to represent the audio. No BOA/EOA wrapping.
The model is trained on this format. See "Token Sequence Divergence" for details.

## WAV Loading

- Supports: 16-bit PCM mono, 32-bit float mono
- **Requires 16kHz** — no resampling in C (avoids mismatch with PyTorch's `np.interp`)
- Resample externally before passing: `ffmpeg -i in.wav -ar 16000 -ac 1 -sample_fmt f32 out.wav`
- Max: ~25.6 seconds (4096 mel frames)

## Usage

```bash
# 16kHz mono WAV required
# NOTE: The prompt is the text instruction; audio tokens are appended after it.
# The official Gemma 4 ASR prompt:
PROMPT="Transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:
* Only output the transcription, with no newlines.
* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."

./run -m gemma4-E2B-int8.bin \
      -A gemma4-E2B-int8-audio.bin \
      -a /path/to/audio_16k.wav \
      -t 0 -n 128 \
      "$PROMPT"

# Resample if needed
ffmpeg -i /path/to/audio.wav -ar 16000 -ac 1 /tmp/audio_16k.wav
```

> **Status (2026-09-05):** The `generate_audio()` function currently places
> audio tokens **before** the prompt text and wraps them with BOA/EOA, which
> does not match the official pipeline. The usage above shows the *intended*
> behavior after the token sequence fix (Next Steps #12).

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
- [x] Conformer attention (chunked local + rel_shift) — **VERIFIED CORRECT** (cos=1.0, see below)
- [x] Conformer LConv1d
- [x] Conformer FFN2
- [x] Output projection + multimodal embedder
- [x] Audio embedding injection in text model
- [x] No ASan errors (heap/stack overflow fixed)
- [x] End-to-end encoder matches PyTorch (cos=1.0 at all layers + output)
- [ ] **Token sequence in `generate_audio()`** — wrong order (audio before text)
  and includes BOA/EOA not present in the official pipeline. **Blocking issue**
  for end-to-end transcription. See "Token Sequence Divergence" below.
- [ ] Speech transcription quality — int8 activation quantization of soft tokens
  degrades quality even after token sequence fix (soft tokens are 50× larger
  than text embeddings; per-group-64 int8 zeros 1% of values)

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
   *Superseded:* after switching to float32, the remaining "attention bug"
   turned out to be a permute error in the reference script (see below).

5. **GLU activation in LConv1d** (2026-09-05): The C code computed
   `silu(a) * b` (i.e., `a * sigmoid(a) * b`) but PyTorch's
   `F.glu(x, dim=-1)` computes `a * sigmoid(b)` where `a` and `b` are the
   two halves of the input. This systematic error at every layer was the
   **primary cause** of the 12-layer divergence (even with float64
   accumulation, the wrong activation function produces wrong outputs).
   Fixed to `a / (1.0 + exp(-b))`.

### Reference Encoder Bug (2026-09-04)

The apparent "attention cos=0.707" mismatch was a **false alarm** caused by a bug
in `tools/reference_encoder.py`, not in the C code.

**The bug:** After computing `attn_out = attn_weights @ v_5d` with shape
`[1, H, nb, C, D]`, the reference script used:

```python
# WRONG: permute(2, 0, 1, 3) gives [C, H, nb, D] → reshapes heads before blocks
attn_out = attn_out.squeeze(0).permute(2, 0, 1, 3).reshape(num_blocks * CHUNK, -1)
```

This interleaves heads within each chunk position, producing a `[T, H*D]` layout
where dimension order is `[head0_d0..127, head1_d0..127, ...]` grouped by
**position first, then head** — but the `reshape` treats the first axis as the
row index, so it actually produces rows in `[C, H, nb]` order rather than the
required `[nb, C, H]` order.

**The fix:**

```python
# CORRECT: permute(1, 2, 0, 3) gives [nb, C, H, D] → [nb*C, H*D]
attn_out = attn_out.squeeze(0).permute(1, 2, 0, 3).reshape(num_blocks * CHUNK, -1)
```

This matches the C code's layout where row `t = block * CHUNK + pos_in_block`
contains all 8 heads' 128-dim outputs concatenated:
`[h0_d0..127, h1_d0..127, ..., h7_d0..127]`.

**Verification after fix:**

| Step | max_diff | cosine |
|------|----------|--------|
| Q (scaled) | 0.0000 | 1.000000 |
| K context (block 0) | 0.0005 | 1.000000 |
| V context (block 0) | 0.0003 | 1.000000 |
| matrix_ac (Q·K) | 0.0046 | 0.99999988 |
| Pre-softcap scores | 0.0045 | 1.000000 |
| Softmax weights | 0.0000077 | 1.00000024 |
| Pre-post-proj (attn·V) | 0.0020 | 1.000000 |
| Post-attention (norm+res) | 0.0020 | 1.000000 |
| **Layer 0 final output** | **0.0034** | **0.999999** |

The C conformer layer is **bit-exact equivalent** to PyTorch within float32
rounding. The remaining layer-0 diff of ~0.003 is pure float32 accumulation
order (C scalar loop vs PyTorch SIMD FMA).

### Float32 Precision Compounding (2026-09-04) — **RESOLVED**

Despite each conformer layer matching PyTorch to ~0.1% relative error, the
**12-layer cascade** amplified differences due to nonlinear operations
(RMSNorm, SiLU, tanh, softmax) acting on slightly different inputs each time:

| Layers | cosine vs PyTorch | max_diff |
|--------|-------------------|----------|
| 1 (layer 0) | 0.998 | 0.003 |
| 12 (full) | 0.017 | 36 |

**Root causes (both fixed 2026-09-05):**

1. **GLU activation bug** (LConv1d): The C code used `silu(a) * b`
   (= `a*sigmoid(a)*b`) but PyTorch's `F.glu(x, dim=-1)` computes
   `a * sigmoid(b)`. This was the **primary** source of divergence — a
   systematic error at every layer, not a precision issue.

2. **Float32 storage of intermediates**: Even after fixing the GLU bug,
   storing intermediate activations as float32 (while using double
   accumulation in matmul) was still insufficient. The fix: compute ALL
   intermediate values (activations, norms, attention scores, softmax
   probabilities) in **full double precision**. Weights remain float32 in
   the binary but are upcast to double at use.

**Result after both fixes (2026-09-05):**

| Stage | cosine vs PyTorch | max_diff |
|-------|-------------------|----------|
| Layer 0 | 1.000000 | 0.005 |
| Layer 1–11 | 1.000000 | < 0.009 |
| output_proj | 1.000000 | 0.006 |
| soft_tokens (final) | 0.9999999992 | 0.005 |

The remaining diff (~0.005) is pure float32 rounding in the final output
storage (the API returns float32 soft tokens to the text model).

### Token Sequence Divergence (2026-09-05)

**Root cause of complete transcription failure** (model responds "I cannot
process audio" or generates unrelated text). Discovered by comparing the
token sequence built by `generate_audio()` against the sequence produced
by the official `transformers` pipeline.

**Official `transformers` pipeline token sequence** (from `apply_chat_template`):

```
<bos>  <|turn>user\n  "Transcribe the following..."  <|audio|>×N  <turn|>\n<|turn>model\n
 [2]   [105] [2364] [107]   [text tokens 5183 17038 ...]    [258881×N]  [106][107][105][4368][107]
```

**C implementation `generate_audio()` token sequence:**

```
<|turn>user\n  <|audio>  <|audio|>×N  <audio|>  "What is in this audio?"  <turn|>\n<|turn>model\n
 [105][2364][107] [256000]  [258881×N]  [258883]   [text tokens 5183 ...]    [106][107][105][4368][107]
```

**Three differences (in order of impact):**

1. **Audio/text order is reversed** (primary cause of failure)
   - Official: text instruction → audio tokens
   - C code: audio tokens → text instruction
   - The model was trained with the instruction preceding the audio, so it
     uses the text as context for interpreting the audio soft tokens. With
     the order reversed, the model sees 151+ audio soft tokens before any
     instruction, producing incoherent output.

2. **BOA/EOA tokens not in training data**
   - The C code wraps audio with `<\|audio\>` (256000) and `<audio\|>` (258883).
   - The official pipeline never emits these — it uses only N × 258881.
   - These tokens act as unexpected noise in the attention pattern.

3. **Missing `<bos>` token**
   - The official pipeline starts with `<bos>` (token 2) from the chat template.
   - The C code starts directly with `<\|turn\>user\n`.
   - Less impactful than 1 and 2, but changes positional encoding alignment.

**Reference (official docs):**
- https://ai.google.dev/gemma/docs/capabilities/audio
- Chat template: `google/gemma-4-E2B-it` → `tokenizer_config.json` → `chat_template`
- Audio token emission: `{%- elif item.get('type') in ['audio', 'input_audio'] -%}{{- '<|audio|>' -}}`

**Fix (pending):**
Rewrite `generate_audio()` in `src/generate.c` to match the official sequence:

```c
// Correct order: BOS + turn header + text prompt + N×AUDIO + turn footer
// 1. "<bos>" (token 2)
// 2. "<|turn>user\n"
// 3. user_prompt (tokenized text)
// 4. N × AUDIO_TOKEN_ID (258881)
// 5. "<turn|>\n<|turn>model\n"
// No BOA/EOA tokens.
```

### Known Issues
- [ ] **Token sequence mismatch** — `generate_audio()` uses the wrong token
  order (audio before text) and includes BOA/EOA tokens that the official
  pipeline never emits. This is the **primary cause** of the model's failure
  to transcribe. See "Token Sequence Divergence" above.
- [ ] **Int8 activation quantization of soft tokens** — the audio soft tokens
  have ~50× the dynamic range of regular text embeddings (abs_max 31.4 vs 0.63).
  Per-group-64 int8 quantization (scale = max_abs/127) zeros 1.06% of values
  and introduces ~3.4% mean relative error that compounds through 28 layers.
  The `transformers` pipeline runs the text model in float32/bfloat16 where
  this issue doesn't exist. May require float32 bypass for audio token rows,
  or a QAT model calibrated for the wider activation range.

### Next Steps
1. ~~Fix mel spectrogram to match PyTorch exactly~~ ✓ (done 2026-09-04)
2. ~~Verify per_bin_mean/stddev from the actual preprocessor config~~ ✓ (not in config → defaults to 0/1, no-op)
3. ~~Compare SSCP output with int8 vs float32 reference~~ ✓ (done: int8 cos=0.92, need float32)
4. ~~Switch SSCP input_proj to float32 matmul~~ ✓ (done 2026-09-04)
5. ~~Fix FFN residual aliasing~~ ✓ (done 2026-09-04)
6. ~~Fix rel_shift indexing + out-of-range handling~~ ✓ (done 2026-09-04)
7. ~~Switch all audio linears to float32~~ ✓ (done 2026-09-04, binary 1172 MB)
8. ~~Debug conformer attention~~ ✓ (done 2026-09-04: C code was correct;
   the "cos=0.707" was a permute bug in `reference_encoder.py`)
9. ~~Validate LConv1d + FFN2 + output_proj~~ ✓ (done 2026-09-04)
10. ~~Fix GLU activation in LConv1d~~ ✓ (done 2026-09-05: `silu(a)*b` → `a*sigmoid(b)`)
11. ~~Switch conformer loop to float64 intermediates~~ ✓ (done 2026-09-05:
    cos=1.0 at all 12 layers + output, vs PyTorch)
12. **Fix token sequence in `generate_audio()`** (2026-09-05) — align with
    official pipeline: `<bos>` + `<\|turn\>user\n` + **text first** + N×258881
    + `<turn\|>\n<\|turn\>model\n`. Remove BOA/EOA tokens. This is the
    **blocking issue** for end-to-end transcription.
13. **Address int8 quantization of audio soft tokens** — options:
    - Bypass `quantize_dispatch` for audio token rows (use float32 matmul path)
    - Use QAT int4 model trained with audio (if available)
    - Scale soft tokens to match text embedding range before injection
14. **Optimize** — the float64 conformer is ~5× slower than float32. Options:
    - Use float32 for the first N layers (where error is still small) and
      float64 only for the last M layers
    - Use float32 with Kahan summation or pairwise summation in matmul
    - Profile to find which layers are most sensitive and only use float64 there
15. **Test with actual speech audio** to validate end-to-end transcription
    quality (current test audio is music)

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
#   /tmp/c_attn_k_ctx_b0.bin layer 0 K context block 0 [8*24, 128]
#   /tmp/c_attn_v_ctx_b0.bin layer 0 V context block 0 [8*24, 128]
#   /tmp/c_attn_matrix_ac.bin  layer 0 Q·K scores [T/4, 8*24]
#   /tmp/c_attn_pre_cap.bin    layer 0 pre-softcap (ac+bd) [T/4, 8*24]
#   /tmp/c_attn_weights.bin    layer 0 softmax weights [T/4, 8*24]
#   /tmp/c_attn_rel_k.bin      relative key projections [13, 1024]
#   /tmp/c_attn_prepost.bin    layer 0 weighted-V output [T/4, 1024]
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
#define AUDIO_TOKEN_ID   258881   // <|audio|> — the only audio token used by transformers
#define AUDIO_BOA_ID     256000   // <|audio>  — NOT used by the official pipeline (unused)
#define AUDIO_EOA_ID     258883   // <audio|>  — NOT used by the official pipeline (unused)
```
