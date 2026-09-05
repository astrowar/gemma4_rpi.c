#ifndef GEMMA4_AUDIO_H
#define GEMMA4_AUDIO_H

#include "gemma4.h"

// ----------------------------------------------------------------------------
// Audio encoder dimensions (Gemma 4 E2B)

#define AUDIO_HIDDEN     1024
#define AUDIO_HEADS      8
#define AUDIO_HEAD_DIM   128   // AUDIO_HIDDEN / AUDIO_HEADS
#define AUDIO_LAYERS     12
#define AUDIO_FFN        4096  // 4x hidden
#define AUDIO_OUTPUT     1536  // output_proj dims (= text hidden_size for E2B)
#define AUDIO_MEL_BINS   128
#define AUDIO_FFT_LEN    512
#define AUDIO_FRAME_LEN  320   // 20ms at 16kHz
#define AUDIO_HOP_LEN    160   // 10ms at 16kHz
#define AUDIO_MAX_TOKENS 750   // max soft tokens per segment

// Chunked attention parameters
#define AUDIO_CHUNK      12
#define AUDIO_PAST_HORIZON 12  // context_left - 1
#define AUDIO_FUTURE_HORIZON 0
#define AUDIO_CONTEXT    24    // chunk + past + future

// Convolution parameters
#define AUDIO_CONV_KERNEL 5    // depthwise conv1d kernel size
#define AUDIO_CONV2D_K    3    // subsample conv2d kernel size

// ----------------------------------------------------------------------------
// Clippable linear: 4 float32 bounds + one int8 weight tensor.
// 64 bytes: the 32-byte loader stepper reads bytes 32-47 as .data/.scales,
// so those MUST be zero. Clip bounds live in bytes 48-63 (the "shape" slot).
typedef struct {
    Tensor weight;      // int8 packed (same layout as text)
    uint64_t _pad1;     // must be 0 (stepper .data → NULL skip)
    uint64_t _pad2;     // must be 0 (stepper .scales → NULL skip)
    float input_min, input_max, output_min, output_max;
} ClippableLinear;

// ----------------------------------------------------------------------------
// Sub-sample convolution projection
typedef struct {
    Tensor conv0_weight;  // [128, 1, 3, 3] float32
    Tensor norm0_weight;  // [128] float32 (LayerNorm, no bias)
    Tensor conv1_weight;  // [32, 128, 3, 3] float32
    Tensor norm1_weight;  // [32] float32
    ClippableLinear input_proj; // [1024, 1024] int8
} AudioSSCP;

// ----------------------------------------------------------------------------
// Feed forward block
typedef struct {
    ClippableLinear ffw1;  // [4096, 1024] int8
    ClippableLinear ffw2;  // [1024, 4096] int8
    Tensor pre_norm;       // [1024] float32 RMSNorm
    Tensor post_norm;      // [1024] float32 RMSNorm
} AudioFFN;

// ----------------------------------------------------------------------------
// Chunked local attention
typedef struct {
    ClippableLinear q_proj;      // [1024, 1024] int8
    ClippableLinear k_proj;      // [1024, 1024] int8
    ClippableLinear v_proj;      // [1024, 1024] int8
    ClippableLinear post;        // [1024, 1024] int8
    Tensor relative_k_proj;      // [1024, 1024] float32 (NOT clippable)
    Tensor per_dim_scale;        // [128] float32 (softplus applied at runtime)
} AudioAttention;

// ----------------------------------------------------------------------------
// Light causal convolution 1d
typedef struct {
    ClippableLinear linear_start;  // [2048, 1024] int8 (GLU)
    ClippableLinear linear_end;    // [1024, 1024] int8
    Tensor depthwise_conv;         // [1024, 1, 5] float32
    Tensor pre_norm;               // [1024] float32 RMSNorm
    Tensor conv_norm;              // [1024] float32 RMSNorm
} AudioLConv1d;

// ----------------------------------------------------------------------------
// One conformer layer
typedef struct {
    AudioFFN ffn1;
    AudioAttention attn;
    AudioLConv1d lconv;
    AudioFFN ffn2;
    Tensor norm_pre_attn;    // [1024] float32
    Tensor norm_post_attn;   // [1024] float32
    Tensor norm_out;         // [1024] float32
} AudioLayer;

// ----------------------------------------------------------------------------
// Complete audio model
typedef struct AudioModel {
    char magic[4];           // "MOGA"
    int quant;               // QuantMode (8 = int8)
    int version;             // format version (1)

    // Precomputed feature extraction tables
    float hann_window[AUDIO_FFT_LEN];                    // [512]
    float mel_filters[AUDIO_FFT_LEN / 2 + 1][AUDIO_MEL_BINS]; // [257, 128]
    float per_bin_mean[AUDIO_MEL_BINS];                  // [128]
    float per_bin_stddev[AUDIO_MEL_BINS];                // [128]

    // Model weights
    AudioSSCP sscp;
    AudioLayer layers[AUDIO_LAYERS];
    Tensor output_proj;    // [1536, 1024] int8 (with bias)
    Tensor output_bias;    // [1536] float32

    // Multimodal embedder (audio -> text hidden)
    Tensor embed_proj;     // [1536, 1536] float32 (RMSNorm has no scale)

    // Token IDs for audio placeholder substitution
    int boa_token_id;
    int eoa_token_id;
    int audio_token_id;
} AudioModel;

// Verifies the audio model ABI matches the exporter.
_Static_assert(sizeof(Tensor) == 32 &&
    sizeof(ClippableLinear) == 64 &&
    sizeof(AudioSSCP) == 32 * 4 + 64 &&
    sizeof(AudioFFN) == 64 * 2 + 32 * 2 &&
    sizeof(AudioAttention) == 64 * 4 + 32 * 2 &&
    sizeof(AudioLConv1d) == 64 * 2 + 32 * 3 &&
    sizeof(AudioLayer) == sizeof(AudioFFN) * 2 + sizeof(AudioAttention) + sizeof(AudioLConv1d) + 32 * 3,
    "Audio ABI mismatch");

// ----------------------------------------------------------------------------
// Audio inference state
typedef struct {
    float *mel_spec;       // [max_frames, 128] log-mel spectrogram
    float *hidden;         // [seq_len, 1024] audio encoder hidden states
    float *ffn_out;        // [seq_len, 4096] FFN intermediate
    float *attn_q;         // [seq_len, 8, 128] query states
    float *attn_k;         // [seq_len, 8, 128] key states
    float *attn_v;         // [seq_len, 8, 128] value states
    float *attn_ctx_k;     // [num_blocks, 8, 24, 128] block-context keys
    float *attn_ctx_v;     // [num_blocks, 8, 24, 128] block-context values
    float *attn_scores;    // [seq_len, 8, 24] attention scores
    float *attn_out;       // [seq_len, 8, 128] attention output
    float *lconv_buf;      // [seq_len, 2048] GLU buffer
    float *rel_pos;        // [13, 1024] relative position embeddings
    float *rel_k;          // [13, 8, 128] relative key projections
    int seq_len;           // actual number of frames after SSCP
    int max_frames;        // allocated mel spectrogram size
} AudioState;

// ----------------------------------------------------------------------------
// Audio API

AudioModel *audio_load(const char *path);
void audio_unload(AudioModel *model, size_t size);

AudioState *audio_state_alloc(int max_samples);
void audio_state_free(AudioState *state);

// Compute log-mel spectrogram from raw 16kHz mono float samples.
// Returns the number of mel frames.
int audio_mel_spectrogram(AudioModel *model, const float *samples, int num_samples,
                          float *output);

// Run the full audio encoder. Writes soft tokens to `output` [seq_len, 1536].
// Returns the number of soft tokens produced.
int audio_encode(AudioModel *model, AudioState *state, const float *mel_spec,
                 int num_mel_frames, float *output);

// Load a WAV file (16-bit PCM or 32-bit float, mono).
// Returns a malloc'd float array; caller must free().
float *wav_load(const char *path, int *out_num_samples, int *out_sample_rate);

#endif // GEMMA4_AUDIO_H
