// Pure C audio encoder for Gemma 4 E2B.
//
// Implements the full audio inference path:
//   mel spectrogram -> SSCP -> 12 conformer layers -> output projection
//   -> multimodal embedder -> soft tokens for the text model.
//
// Uses the existing int8 matmul/quantize kernels from the main build for
// all 2D linear layers. Convolutions and norms are scalar float32.

#include "gemma4.h"
#include "audio.h"
#include <string.h>

static float fp16_to_f32(uint16_t v) {
    int sign = v >> 15;
    int exp = (v >> 10) & 31;
    int frac = v & 1023;
    float r;
    if (exp == 0) r = ldexpf((float)frac, -24);
    else if (exp == 31) r = frac ? NAN : INFINITY;
    else r = ldexpf((float)(1024 + frac), exp - 25);
    return sign ? -r : r;
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ----------------------------------------------------------------------------
// Loading

// Dequantize int8 packed weight to float32 row-major
static float *dequant_weight_f32(const Tensor *w) {
    int outputs = w->shape[0];
    int inputs = w->shape[1];
    float *out = malloc((size_t)outputs * inputs * sizeof(float));
    for (int o = 0; o < outputs; o++) {
        for (int i = 0; i < inputs; i++) {
            int block = o / 16, row = o % 16;
            int group = i / 64, chunk = (i % 64) / 4, off = i % 4;
            const int8_t *data = (const int8_t *)w->data;
            int8_t val = data[(size_t)block * 16 * inputs
                             + (size_t)group * 16 * 64
                             + (size_t)chunk * 16 * 4 + row * 4 + off];
            int groups = inputs / 64;
            float scale = fp16_to_f32(w->scales[((size_t)block * groups + group) * 16 + row]);
            out[(size_t)o * inputs + i] = (float)val * scale;
        }
    }
    return out;
}

AudioModel *audio_load(const char *path) {
    int fd = open(path, O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st)) { perror(path); return NULL; }
    AudioModel *model = mmap(NULL, (size_t)st.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    close(fd);
    if (model == MAP_FAILED) { perror("mmap"); return NULL; }

    if (memcmp(model->magic, "MOGA", 4) != 0) {
        fprintf(stderr, "bad audio file (expected MOGA magic)\n");
        munmap(model, (size_t)st.st_size);
        return NULL;
    }

    // Resolve all tensor file offsets to pointers.
    uint8_t *base = (uint8_t *)model;
    Tensor *all_tensors = (Tensor *)&model->sscp;
    size_t num_tensors = ((uint8_t *)&model->boa_token_id - (uint8_t *)&model->sscp) / sizeof(Tensor);
    for (size_t i = 0; i < num_tensors; i++) {
        all_tensors[i].data = all_tensors[i].data ? (void *)(base + (uintptr_t)all_tensors[i].data) : NULL;
        all_tensors[i].scales = all_tensors[i].scales ? (uint16_t *)(base + (uintptr_t)all_tensors[i].scales) : NULL;
    }

    // Dequantize all int8 weights to float32 (int8 too lossy for audio hidden states)
    for (size_t i = 0; i < num_tensors; i++) {
        if (all_tensors[i].data && all_tensors[i].scales) {
            all_tensors[i].data = dequant_weight_f32(&all_tensors[i]);
        }
    }
    return model;
}

void audio_unload(AudioModel *model, size_t size) {
    munmap(model, size);
}

// ----------------------------------------------------------------------------
// State allocation

#define MAX_MEL_FRAMES 4096  // ~25.6 seconds at 10ms hop

AudioState *audio_state_alloc(int max_samples) {
    (void)max_samples;
    AudioState *state = calloc(1, sizeof(AudioState));
    int max_frames = MAX_MEL_FRAMES;
    int max_seq = (max_frames + 1) / 4;  // after 4x subsampling

    state->max_frames = max_frames;
    state->mel_spec = malloc((size_t)max_frames * AUDIO_MEL_BINS * sizeof(float));
    state->hidden = malloc((size_t)max_seq * AUDIO_HIDDEN * sizeof(float));
    state->ffn_out = malloc((size_t)max_seq * AUDIO_FFN * sizeof(float));
    state->attn_q = malloc((size_t)max_seq * AUDIO_HEADS * AUDIO_HEAD_DIM * sizeof(float));
    state->attn_k = malloc((size_t)max_seq * AUDIO_HEADS * AUDIO_HEAD_DIM * sizeof(float));
    state->attn_v = malloc((size_t)max_seq * AUDIO_HEADS * AUDIO_HEAD_DIM * sizeof(float));
    int max_blocks = (max_seq + AUDIO_CHUNK - 1) / AUDIO_CHUNK;
    state->attn_ctx_k = malloc((size_t)max_blocks * AUDIO_HEADS * AUDIO_CONTEXT * AUDIO_HEAD_DIM * sizeof(float));
    state->attn_ctx_v = malloc((size_t)max_blocks * AUDIO_HEADS * AUDIO_CONTEXT * AUDIO_HEAD_DIM * sizeof(float));
    state->attn_scores = malloc((size_t)max_seq * AUDIO_HEADS * AUDIO_CONTEXT * sizeof(float));
    state->attn_out = malloc((size_t)max_seq * AUDIO_HEADS * AUDIO_HEAD_DIM * sizeof(float));
    state->lconv_buf = malloc((size_t)max_seq * (AUDIO_HIDDEN * 2) * sizeof(float));
    state->rel_pos = malloc((AUDIO_CONTEXT / 2 + 1) * AUDIO_HIDDEN * sizeof(float));
    state->rel_k = malloc((AUDIO_CONTEXT / 2 + 1) * AUDIO_HEADS * AUDIO_HEAD_DIM * sizeof(float));
    return state;
}

void audio_state_free(AudioState *state) {
    if (!state) return;
    free(state->mel_spec);
    free(state->hidden);
    free(state->ffn_out);
    free(state->attn_q);
    free(state->attn_k);
    free(state->attn_v);
    free(state->attn_ctx_k);
    free(state->attn_ctx_v);
    free(state->attn_scores);
    free(state->attn_out);
    free(state->lconv_buf);
    free(state->rel_pos);
    free(state->rel_k);
    free(state);
}

// ----------------------------------------------------------------------------
// FFT (radix-2, iterative, in-place)

static void fft_radix2(float *re, float *im, int n) {
    // Bit-reversal permutation
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j |= bit;
        if (i < j) {
            float tr = re[i]; re[i] = re[j]; re[j] = tr;
            float ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
    // Butterfly
    for (int len = 2; len <= n; len <<= 1) {
        float angle = -2.0f * M_PI / len;
        float wlen_re = cosf(angle), wlen_im = sinf(angle);
        for (int i = 0; i < n; i += len) {
            float w_re = 1.0f, w_im = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                int a = i + j, b = i + j + len / 2;
                float u_re = re[a], u_im = im[a];
                float v_re = re[b] * w_re - im[b] * w_im;
                float v_im = re[b] * w_im + im[b] * w_re;
                re[a] = u_re + v_re; im[a] = u_im + v_im;
                re[b] = u_re - v_re; im[b] = u_im - v_im;
                float nw_re = w_re * wlen_re - w_im * wlen_im;
                w_im = w_re * wlen_im + w_im * wlen_re;
                w_re = nw_re;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Mel spectrogram

int audio_mel_spectrogram(AudioModel *model, const float *samples, int num_samples,
                          float *output) {
    const float *hann = model->hann_window;
    const float (*mel_filters)[AUDIO_MEL_BINS] = model->mel_filters;
    const float *bin_mean = model->per_bin_mean;
    const float *bin_std = model->per_bin_stddev;

    // Pad: semicausal (frame_length/2 zeros at start)
    int pad = AUDIO_FRAME_LEN / 2;
    int total = num_samples + pad;
    int num_frames = (total - (AUDIO_FRAME_LEN + 1)) / AUDIO_HOP_LEN + 1;
    if (num_frames <= 0) return 0;
    if (num_frames > MAX_MEL_FRAMES) num_frames = MAX_MEL_FRAMES;

    float *buf_re = malloc((size_t)AUDIO_FFT_LEN * sizeof(float));
    float *buf_im = malloc((size_t)AUDIO_FFT_LEN * sizeof(float));
    float *power = malloc((size_t)(AUDIO_FFT_LEN / 2 + 1) * sizeof(float));

    for (int f = 0; f < num_frames; f++) {
        int offset = f * AUDIO_HOP_LEN - pad;
        // Fill FFT buffer
        memset(buf_re, 0, AUDIO_FFT_LEN * sizeof(float));
        memset(buf_im, 0, AUDIO_FFT_LEN * sizeof(float));
        for (int i = 0; i < AUDIO_FRAME_LEN && i < AUDIO_FFT_LEN; i++) {
            int idx = offset + i;
            float sample = (idx >= 0 && idx < num_samples) ? samples[idx] : 0.0f;
            buf_re[i] = sample * hann[i];
        }
        // Debug: dump windowed frame 215
        if (getenv("AUDIO_DEBUG") && f == 215) {
            FILE *fp = fopen("/tmp/c_fft_input_215.bin", "wb");
            fwrite(buf_re, sizeof(float), AUDIO_FRAME_LEN, fp);
            fclose(fp);
        }
        // FFT
        fft_radix2(buf_re, buf_im, AUDIO_FFT_LEN);
        // Magnitude spectrum (matches transformers: np.abs(stft))
        for (int k = 0; k <= AUDIO_FFT_LEN / 2; k++) {
            power[k] = sqrtf(buf_re[k] * buf_re[k] + buf_im[k] * buf_im[k]);
        }
        // Mel filterbank
        for (int m = 0; m < AUDIO_MEL_BINS; m++) {
            float sum = 0.0f;
            for (int k = 0; k <= AUDIO_FFT_LEN / 2; k++) {
                sum += power[k] * mel_filters[k][m];
            }
            // Log with additive floor (matches transformers: log(mel + floor))
            float log_val = logf(sum + 1e-3f);
            // Per-bin normalization
            output[(size_t)f * AUDIO_MEL_BINS + m] = (log_val - bin_mean[m]) / bin_std[m];
        }
    }
    free(buf_re);
    free(buf_im);
    free(power);
    return num_frames;
}

// ----------------------------------------------------------------------------
// Helpers: RMSNorm, LayerNorm, activations

static void rmsnorm(float *output, const float *input, const float *weight,
                    int n, int dim) {
    // Batched RMSNorm: n rows, dim columns. weight is [dim] or NULL.
    for (int i = 0; i < n; i++) {
        const float *x = input + (size_t)i * dim;
        float *o = output + (size_t)i * dim;
        float sum = 0.0f;
        for (int j = 0; j < dim; j++) sum += x[j] * x[j];
        float inv_rms = 1.0f / sqrtf(sum / dim + 1e-6f);
        if (weight) {
            for (int j = 0; j < dim; j++) o[j] = x[j] * inv_rms * weight[j];
        } else {
            for (int j = 0; j < dim; j++) o[j] = x[j] * inv_rms;
        }
    }
}

static void layernorm2d(float *output, const float *input, const float *weight,
                        int c, int h, int w, float eps) {
    // LayerNorm over channels (c) for 2D feature map [c, h, w]
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            // Compute mean and variance over c channels
            float mean = 0.0f;
            for (int ch = 0; ch < c; ch++)
                mean += input[(size_t)ch * h * w + (size_t)y * w + x];
            mean /= c;
            float var = 0.0f;
            for (int ch = 0; ch < c; ch++) {
                float d = input[(size_t)ch * h * w + (size_t)y * w + x] - mean;
                var += d * d;
            }
            var = var / c + eps;
            float inv_std = 1.0f / sqrtf(var);
            for (int ch = 0; ch < c; ch++) {
                float val = input[(size_t)ch * h * w + (size_t)y * w + x] - mean;
                output[(size_t)ch * h * w + (size_t)y * w + x] =
                    val * inv_std * weight[ch];
            }
        }
    }
}

static float silu(float x) {
    return x / (1.0f + expf(-x));
}

static inline float softcap(float x, float cap) {
    return cap * tanhf(x / cap);
}

static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// ----------------------------------------------------------------------------
// Int8 linear layer (uses existing matmul_int8 from the build)

static void audio_linear(float *output, const float *input, int rows,
                         const ClippableLinear *layer, int in_dim, int out_dim) {
    // Float32 matmul: output[rows, out_dim] = input[rows, in_dim] @ weight[out_dim, in_dim]^T
    const float *w = (const float *)layer->weight.data;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < out_dim; j++) {
            float sum = 0.0f;
            for (int k = 0; k < in_dim; k++)
                sum += input[(size_t)i * in_dim + k] * w[(size_t)j * in_dim + k];
            output[(size_t)i * out_dim + j] = sum;
        }
    }
}

// ----------------------------------------------------------------------------
// Relative positional encoding

static void compute_rel_pos(AudioModel *model, float *rel_pos) {
    (void)model;
    int num_timescales = AUDIO_HIDDEN / 2;
    float log_inc = logf(10000.0f) / (num_timescales - 1);
    int num_positions = AUDIO_CONTEXT / 2 + 1;  // 13

    for (int p = 0; p < num_positions; p++) {
        int pos = num_positions - 1 - p;  // position_ids = [12, 11, ..., 0]
        for (int i = 0; i < num_timescales; i++) {
            float inv_ts = expf(-log_inc * i);
            float angle = pos * inv_ts;
            rel_pos[(size_t)p * AUDIO_HIDDEN + i] = sinf(angle);
            rel_pos[(size_t)p * AUDIO_HIDDEN + num_timescales + i] = cosf(angle);
        }
    }
}

// ----------------------------------------------------------------------------
// Sub-sample convolution projection

static void audio_sscp(AudioModel *model, const float *mel, int num_frames,
                       float *output, int *out_frames) {
    const AudioSSCP *sscp = &model->sscp;
    int H0 = num_frames, W0 = AUDIO_MEL_BINS;  // [1, H, W] input
    // Conv2d stride-2: output_h = (H + 2*pad - kernel) / stride + 1
    int H1 = (H0 + 2 - AUDIO_CONV2D_K) / 2 + 1;
    int W1 = (W0 + 2 - AUDIO_CONV2D_K) / 2 + 1;
    int H2 = (H1 + 2 - AUDIO_CONV2D_K) / 2 + 1;
    int W2 = (W1 + 2 - AUDIO_CONV2D_K) / 2 + 1;

    const int C1 = 128, C2 = 32;

    static float *buf1 = NULL, *buf2 = NULL, *buf3 = NULL;
    static size_t cap1 = 0, cap2 = 0, cap3 = 0;
    size_t sz1 = (size_t)C1 * H1 * W1;
    size_t sz2 = (size_t)C2 * H2 * W2;
    if (cap1 < sz1) { free(buf1); buf1 = malloc(sz1 * sizeof(float)); cap1 = sz1; }
    if (cap2 < sz1) { free(buf2); buf2 = malloc(sz1 * sizeof(float)); cap2 = sz1; }  // needs sz1 for layer0 LN
    size_t sz3 = (size_t)H2 * W2 * C2;
    if (cap3 < sz3) { free(buf3); buf3 = malloc(sz3 * sizeof(float)); cap3 = sz3; }

    // Layer 0: Conv2d(1->128, k=3, s=2, pad=1)
    const float *w0 = (const float *)sscp->conv0_weight.data;  // [128, 1, 3, 3]
    for (int oc = 0; oc < C1; oc++) {
        for (int oy = 0; oy < H1; oy++) {
            for (int ox = 0; ox < W1; ox++) {
                int iy = oy * 2 - 1;  // pad=1
                int ix = ox * 2 - 1;
                float sum = 0.0f;
                for (int ky = 0; ky < 3; ky++) {
                    for (int kx = 0; kx < 3; kx++) {
                        int y = iy + ky, x = ix + kx;
                        if (y >= 0 && y < H0 && x >= 0 && x < W0)
                            sum += w0[oc * 9 + ky * 3 + kx] * mel[(size_t)y * W0 + x];
                    }
                }
                buf1[(size_t)oc * H1 * W1 + (size_t)oy * W1 + ox] = sum;
            }
        }
    }
    // LayerNorm over channels (128), then ReLU
    if (getenv("AUDIO_DEBUG")) {
        FILE *fp = fopen("/tmp/c_conv0.bin", "wb");
        int hdr[3] = {C1, H1, W1};
        fwrite(hdr, sizeof(int), 3, fp);
        fwrite(buf1, sizeof(float), sz1, fp);
        fclose(fp);
    }
    layernorm2d(buf2, buf1, (const float *)sscp->norm0_weight.data, C1, H1, W1, 1e-6f);
    for (size_t i = 0; i < sz1; i++) buf2[i] = buf2[i] > 0 ? buf2[i] : 0.0f;

    // Layer 1: Conv2d(128->32, k=3, s=2, pad=1)
    const float *w1 = (const float *)sscp->conv1_weight.data;  // [32, 128, 3, 3]
    for (int oc = 0; oc < C2; oc++) {
        for (int oy = 0; oy < H2; oy++) {
            for (int ox = 0; ox < W2; ox++) {
                int iy = oy * 2 - 1;
                int ix = ox * 2 - 1;
                float sum = 0.0f;
                for (int ic = 0; ic < C1; ic++) {
                    for (int ky = 0; ky < 3; ky++) {
                        for (int kx = 0; kx < 3; kx++) {
                            int y = iy + ky, x = ix + kx;
                            if (y >= 0 && y < H1 && x >= 0 && x < W1)
                                sum += w1[((size_t)oc * C1 + ic) * 9 + ky * 3 + kx] *
                                       buf2[(size_t)ic * H1 * W1 + (size_t)y * W1 + x];
                        }
                    }
                }
                buf1[(size_t)oc * H2 * W2 + (size_t)oy * W2 + ox] = sum;
            }
        }
    }
    // LayerNorm over channels (32), then ReLU
    layernorm2d(buf2, buf1, (const float *)sscp->norm1_weight.data, C2, H2, W2, 1e-6f);
    for (size_t i = 0; i < sz2; i++) buf2[i] = buf2[i] > 0 ? buf2[i] : 0.0f;

    // Reshape to [H2, W2*C2] = [H2, 32] then project to [H2, 1024]
    // Actually: permute to [batch, seq=H2, feat=W2*C2=32*8=256? No...]
    // Wait: after conv1, shape is [batch, C2=32, H2, W2]. Permute to [batch, H2, W2, C2]
    // then reshape to [batch, H2, W2*C2]. W2 = (W1+2-3)/2+1 = ((W0+2-3)/2+1+2-3)/2+1
    // W0=128: W1=(128+2-3)/2+1=64, W2=(64+2-3)/2+1=32
    // So feat = W2*C2 = 32*32 = 1024. That matches input_proj [1024, 1024].
    int feat = W2 * C2;  // should be 1024

    // Reshape: [H2, C2, W2] -> [H2, W2*C2] (interleave)
    for (int h = 0; h < H2; h++) {
        for (int c = 0; c < C2; c++) {
            for (int w = 0; w < W2; w++) {
                buf3[(size_t)h * feat + (size_t)w * C2 + c] =
                    buf2[(size_t)c * H2 * W2 + (size_t)h * W2 + w];
            }
        }
    }
    // Debug: dump pre-linear features (after conv1+LN+ReLU+reshape)
    if (getenv("AUDIO_DEBUG")) {
        FILE *fp = fopen("/tmp/c_sscp_prelin.bin", "wb");
        int hdr[2] = {H2, feat};
        fwrite(hdr, sizeof(int), 2, fp);
        fwrite(buf3, sizeof(float), (size_t)H2 * feat, fp);
        fclose(fp);
    }
    // Linear projection (float32 matmul)
    audio_linear(output, buf3, H2, &sscp->input_proj, feat, AUDIO_HIDDEN);
    *out_frames = H2;
}

// ----------------------------------------------------------------------------
// Feed forward block

static void audio_ffn(AudioState *state, AudioFFN *ffn, const float *input,
                      float *output, int seq_len, float residual_weight) {
    // Pre-norm + ffw1 + SiLU + ffw2
    float *normed = malloc((size_t)seq_len * AUDIO_HIDDEN * sizeof(float));
    float *mid = malloc((size_t)seq_len * AUDIO_FFN * sizeof(float));
    rmsnorm(normed, input, (const float *)ffn->pre_norm.data, seq_len, AUDIO_HIDDEN);
    audio_linear(mid, normed, seq_len, &ffn->ffw1, AUDIO_HIDDEN, AUDIO_FFN);
    for (int i = 0; i < seq_len * AUDIO_FFN; i++) mid[i] = silu(mid[i]);
    audio_linear(output, mid, seq_len, &ffn->ffw2, AUDIO_FFN, AUDIO_HIDDEN);

    // Post-norm * scale + residual (residual is `input`, a separate buffer)
    rmsnorm(output, output, (const float *)ffn->post_norm.data, seq_len, AUDIO_HIDDEN);
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < AUDIO_HIDDEN; j++)
            output[(size_t)i * AUDIO_HIDDEN + j] =
                output[(size_t)i * AUDIO_HIDDEN + j] * residual_weight +
                input[(size_t)i * AUDIO_HIDDEN + j];
    free(normed);
    free(mid);
}

// ----------------------------------------------------------------------------
// Chunked local attention

static void audio_attention(AudioState *state, AudioModel *model, int layer_idx,
                            const float *input, float *output, int seq_len) {
    AudioAttention *attn = &model->layers[layer_idx].attn;
    float *q = state->attn_q;  // [seq, heads, dim]
    float *k = state->attn_k;
    float *v = state->attn_v;

    // Project Q, K, V
    // Apply per_dim_scale to Q
    float softplus_scale[AUDIO_HEAD_DIM];
    for (int i = 0; i < AUDIO_HEAD_DIM; i++) {
        float x = ((const float *)attn->per_dim_scale.data)[i];
        softplus_scale[i] = log1pf(expf(x));  // softplus(x) = log(1+exp(x))
    }

    audio_linear(q, input, seq_len, &attn->q_proj, AUDIO_HIDDEN, AUDIO_HIDDEN);
    audio_linear(k, input, seq_len, &attn->k_proj, AUDIO_HIDDEN, AUDIO_HIDDEN);
    audio_linear(v, input, seq_len, &attn->v_proj, AUDIO_HIDDEN, AUDIO_HIDDEN);

    // Scale Q by q_scale * softplus(per_dim_scale), K by k_scale
    const float q_scale = 1.0f / sqrtf(AUDIO_HEAD_DIM) / logf(2.0f);
    const float k_scale = log1pf(expf(1.0f)) / logf(2.0f);  // log(1+e)/log(2)

    for (int t = 0; t < seq_len; t++) {
        for (int h = 0; h < AUDIO_HEADS; h++) {
            for (int d = 0; d < AUDIO_HEAD_DIM; d++) {
                size_t idx = ((size_t)t * AUDIO_HEADS + h) * AUDIO_HEAD_DIM + d;
                q[idx] *= q_scale * softplus_scale[d];
                k[idx] *= k_scale;
            }
        }
    }

    // Debug: dump scaled Q
    if (getenv("AUDIO_DEBUG") && layer_idx == 0) {
        FILE *fp = fopen("/tmp/c_attn_q.bin", "wb");
        int hdr[2] = {seq_len, AUDIO_HIDDEN};
        fwrite(hdr, sizeof(int), 2, fp);
        fwrite(q, sizeof(float), (size_t)seq_len * AUDIO_HIDDEN, fp);
        fclose(fp);
    }

    // Build block contexts for K and V
    int num_blocks = (seq_len + AUDIO_CHUNK - 1) / AUDIO_CHUNK;
    int padded_len = num_blocks * AUDIO_CHUNK;
    float *ctx_k = state->attn_ctx_k;  // [blocks, heads, context, dim]
    float *ctx_v = state->attn_ctx_v;

    for (int h = 0; h < AUDIO_HEADS; h++) {
        for (int b = 0; b < num_blocks; b++) {
            for (int c = 0; c < AUDIO_CONTEXT; c++) {
                int pos = b * AUDIO_CHUNK + c - AUDIO_PAST_HORIZON;
                for (int d = 0; d < AUDIO_HEAD_DIM; d++) {
                    float kv = 0.0f;
                    if (pos >= 0 && pos < seq_len) {
                        size_t idx = ((size_t)pos * AUDIO_HEADS + h) * AUDIO_HEAD_DIM + d;
                        kv = k[idx];
                    }
                    ctx_k[((size_t)b * AUDIO_HEADS + h) * AUDIO_CONTEXT * AUDIO_HEAD_DIM
                          + (size_t)c * AUDIO_HEAD_DIM + d] = kv;
                    kv = 0.0f;
                    if (pos >= 0 && pos < seq_len) {
                        size_t idx = ((size_t)pos * AUDIO_HEADS + h) * AUDIO_HEAD_DIM + d;
                        kv = v[idx];
                    }
                    ctx_v[((size_t)b * AUDIO_HEADS + h) * AUDIO_CONTEXT * AUDIO_HEAD_DIM
                          + (size_t)c * AUDIO_HEAD_DIM + d] = kv;
                }
            }
        }
    }

    // Relative position encoding + projection
    compute_rel_pos(model, state->rel_pos);
    // relative_k_proj: [13, 1024] -> [13, 1024] (float32, not quantized)
    const float *rel_w = (const float *)attn->relative_k_proj.data;
    // rel_k_proj weight shape: [1024, 1024] row-major (out_dim x in_dim)
    int num_positions = AUDIO_CONTEXT / 2 + 1;  // 13
    for (int p = 0; p < num_positions; p++) {
        for (int o = 0; o < AUDIO_HIDDEN; o++) {
            float sum = 0.0f;
            const float *rp = state->rel_pos + (size_t)p * AUDIO_HIDDEN;
            for (int i = 0; i < AUDIO_HIDDEN; i++)
                sum += rel_w[(size_t)o * AUDIO_HIDDEN + i] * rp[i];
            // Reshape to [heads, dim]
            state->rel_k[(size_t)p * AUDIO_HIDDEN + o] = sum;
        }
    }

    // Compute attention scores for each query position
    float *scores = state->attn_scores;  // [seq, heads, context]
    float *attn_out = state->attn_out;   // [seq, heads, dim]

    for (int t = 0; t < seq_len; t++) {
        int block = t / AUDIO_CHUNK;
        int pos_in_block = t % AUDIO_CHUNK;
        for (int h = 0; h < AUDIO_HEADS; h++) {
            const float *q_t = q + ((size_t)t * AUDIO_HEADS + h) * AUDIO_HEAD_DIM;
            const float *ctx_k_h = ctx_k + ((size_t)block * AUDIO_HEADS + h) * AUDIO_CONTEXT * AUDIO_HEAD_DIM;
            const float *ctx_v_h = ctx_v + ((size_t)block * AUDIO_HEADS + h) * AUDIO_CONTEXT * AUDIO_HEAD_DIM;

            // matrix_ac: q[t] @ ctx_k[context positions]
            for (int c = 0; c < AUDIO_CONTEXT; c++) {
                float dot = 0.0f;
                const float *k_c = ctx_k_h + (size_t)c * AUDIO_HEAD_DIM;
                for (int d = 0; d < AUDIO_HEAD_DIM; d++)
                    dot += q_t[d] * k_c[d];
                scores[((size_t)t * AUDIO_HEADS + h) * AUDIO_CONTEXT + c] = dot;
            }

            // matrix_bd: relative position contribution (zero for out-of-range)
            // rel_shift: position p in block, context c → rel_k index = c - p
            for (int c = 0; c < AUDIO_CONTEXT; c++) {
                int rel_idx = c - pos_in_block;
                if (rel_idx >= 0 && rel_idx < num_positions) {
                    float dot = 0.0f;
                    const float *rk = state->rel_k + (size_t)rel_idx * AUDIO_HIDDEN + h * AUDIO_HEAD_DIM;
                    for (int d = 0; d < AUDIO_HEAD_DIM; d++)
                        dot += q_t[d] * rk[d];
                    scores[((size_t)t * AUDIO_HEADS + h) * AUDIO_CONTEXT + c] += dot;
                }
            }

            // Soft cap
            for (int c = 0; c < AUDIO_CONTEXT; c++)
                scores[((size_t)t * AUDIO_HEADS + h) * AUDIO_CONTEXT + c] =
                    softcap(scores[((size_t)t * AUDIO_HEADS + h) * AUDIO_CONTEXT + c], 50.0f);

            // Softmax
            float max_s = -1e30f;
            for (int c = 0; c < AUDIO_CONTEXT; c++)
                if (scores[((size_t)t * AUDIO_HEADS + h) * AUDIO_CONTEXT + c] > max_s)
                    max_s = scores[((size_t)t * AUDIO_HEADS + h) * AUDIO_CONTEXT + c];
            float sum_exp = 0.0f;
            for (int c = 0; c < AUDIO_CONTEXT; c++) {
                float s = scores[((size_t)t * AUDIO_HEADS + h) * AUDIO_CONTEXT + c];
                if (s < -1e8f) { s = 0.0f; }
                else s = expf(s - max_s);
                scores[((size_t)t * AUDIO_HEADS + h) * AUDIO_CONTEXT + c] = s;
                sum_exp += s;
            }
            for (int c = 0; c < AUDIO_CONTEXT; c++)
                scores[((size_t)t * AUDIO_HEADS + h) * AUDIO_CONTEXT + c] /= sum_exp;

            // Weighted sum of values
            float *out = attn_out + ((size_t)t * AUDIO_HEADS + h) * AUDIO_HEAD_DIM;
            for (int d = 0; d < AUDIO_HEAD_DIM; d++) {
                float sum = 0.0f;
                for (int c = 0; c < AUDIO_CONTEXT; c++)
                    sum += scores[((size_t)t * AUDIO_HEADS + h) * AUDIO_CONTEXT + c] *
                           ctx_v_h[(size_t)c * AUDIO_HEAD_DIM + d];
                out[d] = sum;
            }
        }
    }

    // Reshape back to [seq, hidden] and apply post projection
    audio_linear(output, attn_out, seq_len, &attn->post, AUDIO_HIDDEN, AUDIO_HIDDEN);
    (void)padded_len;
}

// ----------------------------------------------------------------------------
// Light causal convolution 1d

static void audio_lconv1d(AudioState *state, AudioModel *model, int layer_idx,
                          const float *input, float *output, int seq_len) {
    AudioLConv1d *lc = &model->layers[layer_idx].lconv;
    float *buf = state->lconv_buf;  // [seq, 2048]

    // Pre-norm
    float *normed = malloc((size_t)seq_len * AUDIO_HIDDEN * sizeof(float));
    rmsnorm(normed, input, (const float *)lc->pre_norm.data, seq_len, AUDIO_HIDDEN);

    // linear_start: [1024] -> [2048]
    audio_linear(buf, normed, seq_len, &lc->linear_start, AUDIO_HIDDEN, AUDIO_HIDDEN * 2);

    // GLU: split in half, multiply
    float *glu_out = malloc((size_t)seq_len * AUDIO_HIDDEN * sizeof(float));
    for (int t = 0; t < seq_len; t++) {
        for (int d = 0; d < AUDIO_HIDDEN; d++) {
            float a = buf[(size_t)t * AUDIO_HIDDEN * 2 + d];
            float b = buf[(size_t)t * AUDIO_HIDDEN * 2 + AUDIO_HIDDEN + d];
            glu_out[(size_t)t * AUDIO_HIDDEN + d] = silu(a) * b;
        }
    }
    free(normed);

    // Causal depthwise conv1d (kernel=5, left_pad=4)
    const float *dw_w = (const float *)lc->depthwise_conv.data;  // [1024, 1, 5]
    for (int t = 0; t < seq_len; t++) {
        for (int ch = 0; ch < AUDIO_HIDDEN; ch++) {
            float sum = 0.0f;
            for (int k = 0; k < AUDIO_CONV_KERNEL; k++) {
                int src = t - (AUDIO_CONV_KERNEL - 1 - k);
                if (src >= 0 && src < seq_len)
                    sum += dw_w[(size_t)ch * AUDIO_CONV_KERNEL + k] *
                           glu_out[(size_t)src * AUDIO_HIDDEN + ch];
            }
            output[(size_t)t * AUDIO_HIDDEN + ch] = sum;
        }
    }
    free(glu_out);

    // Conv norm + SiLU
    float *post = malloc((size_t)seq_len * AUDIO_HIDDEN * sizeof(float));
    rmsnorm(post, output, (const float *)lc->conv_norm.data, seq_len, AUDIO_HIDDEN);
    for (int i = 0; i < seq_len * AUDIO_HIDDEN; i++) post[i] = silu(post[i]);

    // linear_end + residual
    audio_linear(output, post, seq_len, &lc->linear_end, AUDIO_HIDDEN, AUDIO_HIDDEN);
    for (int t = 0; t < seq_len; t++)
        for (int d = 0; d < AUDIO_HIDDEN; d++)
            output[(size_t)t * AUDIO_HIDDEN + d] += input[(size_t)t * AUDIO_HIDDEN + d];
    free(post);
}

// ----------------------------------------------------------------------------
// Full audio encoder

int audio_encode(AudioModel *model, AudioState *state, const float *mel_spec,
                 int num_mel_frames, float *output) {
    // SSCP: mel [frames, 128] -> hidden [seq_len, 1024]
    int seq_len = 0;
    audio_sscp(model, mel_spec, num_mel_frames, state->hidden, &seq_len);
    state->seq_len = seq_len;

    if (getenv("AUDIO_DEBUG")) {
        FILE *fp = fopen("/tmp/c_sscp.bin", "wb");
        int hdr[2] = {seq_len, AUDIO_HIDDEN};
        fwrite(hdr, sizeof(int), 2, fp);
        fwrite(state->hidden, sizeof(float), (size_t)seq_len * AUDIO_HIDDEN, fp);
        fclose(fp);
    }

    float *cur = state->hidden;
    float *next = state->ffn_out;  // reuse as scratch

    // 12 conformer layers
    for (int layer = 0; layer < AUDIO_LAYERS; layer++) {
        AudioLayer *al = &model->layers[layer];

        // FFN1 (residual_weight = 0.5)
        audio_ffn(state, &al->ffn1, cur, next, seq_len, 0.5f);

        if (getenv("AUDIO_DEBUG") && layer == 0) {
            FILE *fp = fopen("/tmp/c_l0_ffn1.bin", "wb");
            int hdr[2] = {seq_len, AUDIO_HIDDEN};
            fwrite(hdr, sizeof(int), 2, fp);
            fwrite(next, sizeof(float), (size_t)seq_len * AUDIO_HIDDEN, fp);
            fclose(fp);
        }

        // Norm pre-attn
        float *normed = malloc((size_t)seq_len * AUDIO_HIDDEN * sizeof(float));
        rmsnorm(normed, next, (const float *)al->norm_pre_attn.data, seq_len, AUDIO_HIDDEN);

        // Attention
        float *attn_buf = malloc((size_t)seq_len * AUDIO_HIDDEN * sizeof(float));
        audio_attention(state, model, layer, normed, attn_buf, seq_len);

        // Norm post-attn + residual
        rmsnorm(normed, attn_buf, (const float *)al->norm_post_attn.data, seq_len, AUDIO_HIDDEN);
        for (int t = 0; t < seq_len; t++)
            for (int d = 0; d < AUDIO_HIDDEN; d++)
                next[(size_t)t * AUDIO_HIDDEN + d] += normed[(size_t)t * AUDIO_HIDDEN + d];

        if (getenv("AUDIO_DEBUG") && layer == 0) {
            FILE *fp = fopen("/tmp/c_l0_postattn.bin", "wb");
            int hdr[2] = {seq_len, AUDIO_HIDDEN};
            fwrite(hdr, sizeof(int), 2, fp);
            fwrite(next, sizeof(float), (size_t)seq_len * AUDIO_HIDDEN, fp);
            fclose(fp);
        }

        // LConv1d
        audio_lconv1d(state, model, layer, next, attn_buf, seq_len);

        if (getenv("AUDIO_DEBUG") && layer == 0) {
            FILE *fp = fopen("/tmp/c_l0_lconv.bin", "wb");
            int hdr[2] = {seq_len, AUDIO_HIDDEN};
            fwrite(hdr, sizeof(int), 2, fp);
            fwrite(attn_buf, sizeof(float), (size_t)seq_len * AUDIO_HIDDEN, fp);
            fclose(fp);
        }

        // FFN2 (residual_weight = 0.5)
        audio_ffn(state, &al->ffn2, attn_buf, next, seq_len, 0.5f);

        // Final norm
        rmsnorm(cur, next, (const float *)al->norm_out.data, seq_len, AUDIO_HIDDEN);

        // Debug: dump after each layer
        if (getenv("AUDIO_DEBUG") && layer == 0) {
            FILE *fp = fopen("/tmp/c_layer0.bin", "wb");
            int hdr[2] = {seq_len, AUDIO_HIDDEN};
            fwrite(hdr, sizeof(int), 2, fp);
            fwrite(cur, sizeof(float), (size_t)seq_len * AUDIO_HIDDEN, fp);
            fclose(fp);
        }

        free(normed);
        free(attn_buf);
    }

    // Output projection: [seq, 1024] -> [seq, 1536] (float32 + bias)
    {
        const float *w = (const float *)model->output_proj.data;
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < seq_len; t++) {
            for (int d = 0; d < AUDIO_OUTPUT; d++) {
                float sum = 0.0f;
                for (int k = 0; k < AUDIO_HIDDEN; k++)
                    sum += cur[(size_t)t * AUDIO_HIDDEN + k] * w[(size_t)d * AUDIO_HIDDEN + k];
                output[(size_t)t * AUDIO_OUTPUT + d] = sum;
            }
        }
        // Add bias
        const float *bias = (const float *)model->output_bias.data;
        for (int t = 0; t < seq_len; t++)
            for (int d = 0; d < AUDIO_OUTPUT; d++)
                output[(size_t)t * AUDIO_OUTPUT + d] += bias[d];
    }

    // Multimodal embedder: RMSNorm(no scale) + Linear(1536->1536, float32)
    {
        float *normed = malloc((size_t)seq_len * AUDIO_OUTPUT * sizeof(float));
        rmsnorm(normed, output, NULL, seq_len, AUDIO_OUTPUT);
        // Float32 matrix multiply: [seq, 1536] @ [1536, 1536]^T
        const float *W = (const float *)model->embed_proj.data;
        float *buf = malloc((size_t)seq_len * AUDIO_OUTPUT * sizeof(float));
        for (int t = 0; t < seq_len; t++) {
            for (int o = 0; o < AUDIO_OUTPUT; o++) {
                float sum = 0.0f;
                for (int i = 0; i < AUDIO_OUTPUT; i++)
                    sum += W[(size_t)o * AUDIO_OUTPUT + i] * normed[(size_t)t * AUDIO_OUTPUT + i];
                buf[(size_t)t * AUDIO_OUTPUT + o] = sum;
            }
        }
        memcpy(output, buf, (size_t)seq_len * AUDIO_OUTPUT * sizeof(float));
        free(normed);
        free(buf);
    }

    return seq_len;
}

// ----------------------------------------------------------------------------
// WAV file reader: requires 16kHz mono, 16-bit PCM or 32-bit float
// No resampling — the caller must pre-process the WAV.

float *wav_load(const char *path, int *out_num_samples, int *out_sample_rate) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }

    // Read RIFF header
    char header[12];
    if (fread(header, 1, 12, f) != 12 || memcmp(header, "RIFF", 4) != 0 ||
        memcmp(header + 8, "WAVE", 4) != 0) {
        fprintf(stderr, "not a WAV file: %s\n", path);
        fclose(f);
        return NULL;
    }

    // Read chunks
    int audio_format = 0, num_channels = 0, sample_rate = 0, bits_per_sample = 0;
    int data_size = 0;
    uint8_t *data = NULL;

    while (fread(header, 1, 8, f) == 8) {
        uint32_t chunk_size;
        memcpy(&chunk_size, header + 4, 4);
        if (memcmp(header, "fmt ", 4) == 0) {
            uint8_t fmt[16];
            if (fread(fmt, 1, 16, f) != 16) break;
            audio_format = fmt[0] | (fmt[1] << 8);
            num_channels = fmt[2] | (fmt[3] << 8);
            sample_rate = fmt[4] | (fmt[5] << 8) | (fmt[6] << 16) | (fmt[7] << 24);
            bits_per_sample = fmt[14] | (fmt[15] << 8);
            if (chunk_size > 16) fseek(f, chunk_size - 16, SEEK_CUR);
        } else if (memcmp(header, "data", 4) == 0) {
            data_size = (int)chunk_size;
            data = malloc(data_size);
            if (!data) break;
            if (fread(data, 1, data_size, f) != (size_t)data_size) { free(data); break; }
            break;
        } else {
            fseek(f, chunk_size + (chunk_size & 1), SEEK_CUR);
        }
    }

    fclose(f);
    if (!data) { fprintf(stderr, "no data chunk in WAV\n"); return NULL; }

    // Strict format validation
    if (num_channels != 1) {
        fprintf(stderr, "WAV must be mono (got %d channels)\n", num_channels);
        free(data);
        return NULL;
    }
    if (sample_rate != 16000) {
        fprintf(stderr, "WAV must be 16000 Hz (got %d Hz) — resample before passing\n", sample_rate);
        free(data);
        return NULL;
    }
    if (!((bits_per_sample == 16 && audio_format == 1) ||
          (bits_per_sample == 32 && (audio_format == 3 || audio_format == 1)))) {
        fprintf(stderr, "WAV must be 16-bit PCM or 32-bit float (got %d-bit, format=%d)\n",
                bits_per_sample, audio_format);
        free(data);
        return NULL;
    }

    int num_samples;
    float *samples;
    if (bits_per_sample == 16) {
        num_samples = data_size / 2;
        samples = malloc((size_t)num_samples * sizeof(float));
        const int16_t *pcm = (const int16_t *)data;
        for (int i = 0; i < num_samples; i++)
            samples[i] = (float)pcm[i] / 32768.0f;
    } else {
        num_samples = data_size / 4;
        samples = malloc((size_t)num_samples * sizeof(float));
        memcpy(samples, data, data_size);
    }
    free(data);

    *out_num_samples = num_samples;
    *out_sample_rate = sample_rate;
    return samples;
}

// ----------------------------------------------------------------------------
// High-level: load WAV, run encoder, store soft tokens in InferenceState

void audio_encode_into(AudioModel *audio, InferenceState *state, const char *wav_path) {
    int num_samples = 0, sample_rate = 0;
    float *samples = wav_load(wav_path, &num_samples, &sample_rate);
    if (!samples) return;

    double duration_s = (double)num_samples / 16000.0;
    fprintf(stderr, "Audio: %s (%.1fs, %d samples @ %d Hz)\n", wav_path, duration_s, num_samples, sample_rate);

    // Compute mel spectrogram
    AudioState *astate = audio_state_alloc(num_samples);
    int num_mel_frames = audio_mel_spectrogram(audio, samples, num_samples, astate->mel_spec);
    fprintf(stderr, "Mel spectrogram: %d frames\n", num_mel_frames);
    free(samples);

    // Debug dump
    if (getenv("AUDIO_DEBUG")) {
        FILE *fp = fopen("/tmp/c_mel.bin", "wb");
        int hdr[2] = {num_mel_frames, AUDIO_MEL_BINS};
        fwrite(hdr, sizeof(int), 2, fp);
        fwrite(astate->mel_spec, sizeof(float), (size_t)num_mel_frames * AUDIO_MEL_BINS, fp);
        fclose(fp);
    }

    // Run encoder: output is [seq_len, 1536]
    float *soft_tokens = malloc((size_t)num_mel_frames * AUDIO_OUTPUT * sizeof(float));
    int seq_len = audio_encode(audio, astate, astate->mel_spec, num_mel_frames, soft_tokens);
    fprintf(stderr, "Audio encoder: %d soft tokens\n", seq_len);

    if (getenv("AUDIO_DEBUG")) {
        FILE *fp = fopen("/tmp/c_soft_tokens.bin", "wb");
        int hdr[2] = {seq_len, AUDIO_OUTPUT};
        fwrite(hdr, sizeof(int), 2, fp);
        fwrite(soft_tokens, sizeof(float), (size_t)seq_len * AUDIO_OUTPUT, fp);
        fclose(fp);
    }

    // Project to text hidden size: the output is already [seq_len, 1536] = HIDDEN_SIZE for E2B
    // Store in state
    if (state->audio_embeds) free(state->audio_embeds);
    state->audio_embeds = soft_tokens;
    state->audio_count = seq_len;
    state->audio_start = -1;  // Will be set by generate_audio() when building the prompt

    audio_state_free(astate);
}
