#!/usr/bin/env python3
"""Standalone PyTorch reference for the Gemma4 audio conformer encoder + output_proj.

Usage: python3 tools/reference_encoder.py [sscp_input.pt]
  Default input: /tmp/sscp_ref.pt
  Output: /tmp/encoder_ref.pt (final soft tokens [T, 1536])
"""
import sys, math, torch, torch.nn.functional as F
from safetensors import safe_open

CKPT = '/home/astro/ml/gemma4_rt/.cache/huggingface/models--google--gemma-4-E2B-it-qat-q4_0-unquantized/snapshots/6befbaca7398925921802abd1f277b495b78b738'
# Audio config
H = 1024          # hidden_size
HEADS = 8
HEAD_DIM = 128
L = 12            # num_layers
FFN = 4096        # ffn hidden
OUT_DIM = 1536    # output_proj_dims
CHUNK = 12        # attention_chunk_size
CTX_LEFT = 13     # attention_context_left
CTX_RIGHT = 0     # attention_context_right
SOFTCAP = 50.0    # attention_logit_cap
RES_W = 0.5       # residual_weight
EPS = 1e-6        # rms_norm_eps
CONV_K = 5        # conv_kernel_size
MAX_PAST = CTX_LEFT - 1   # 12
MAX_FUT = CTX_RIGHT       # 0
CTX_SIZE = CHUNK + MAX_PAST + MAX_FUT  # 24


def rms_norm(x, weight, eps=EPS):
    """RMSNorm: x * (mean(x^2) + eps)^(-0.5) * weight"""
    ms = x.pow(2).mean(-1, keepdim=True) + eps
    return x * ms.pow(-0.5) * weight


def linear(x, w):
    """F.linear with no bias: x @ w.T"""
    return F.linear(x, w)


def causal_depthwise_conv(x, w):
    """Causal depthwise conv1d with left-padding.
    x: [T, C]  (seq, channels)
    w: [C, 1, K]  (depthwise kernel)
    left_pad = K - 1
    """
    K = w.shape[2]
    left_pad = K - 1
    x = x.unsqueeze(0).permute(0, 2, 1)  # [1, C, T]
    x = F.pad(x, (left_pad, 0))
    out = F.conv1d(x, w, groups=w.shape[0])  # [1, C, T]
    return out.permute(0, 2, 1).squeeze(0)  # [T, C]


def audio_ffn(hs, w1, w2, pre_ln, post_ln, res_w=RES_W):
    """Feed-forward: residual + pre_ln → linear1 → SiLU → linear2 → post_ln → *res_w + residual"""
    residual = hs
    hs = rms_norm(hs, pre_ln)
    hs = linear(hs, w1)
    hs = F.silu(hs)
    hs = linear(hs, w2)
    hs = rms_norm(hs, post_ln) * res_w
    hs = hs + residual
    return hs


def audio_lconv1d(hs, w_start, w_end, w_conv, pre_ln, conv_norm):
    """Light Conv1d: residual + pre_ln → linear_start → GLU → causal_dw_conv → conv_norm → SiLU → linear_end + residual"""
    residual = hs
    hs = rms_norm(hs, pre_ln)
    hs = linear(hs, w_start)
    hs = F.glu(hs, dim=-1)
    hs = causal_depthwise_conv(hs, w_conv)
    hs = rms_norm(hs, conv_norm)
    hs = F.silu(hs)
    hs = linear(hs, w_end)
    hs = hs + residual
    return hs


def audio_attention(hs, wq, wk, wv, wpost, wrel, per_dim_scale, pos_embed):
    """Chunked local attention with relative position bias.
    hs: [T, H]
    pos_embed: [CTX_SIZE//2 + 1, H]  (sin/cos concatenated)
    """
    T = hs.shape[0]
    q_scale = (HEAD_DIM ** -0.5) / math.log(2)
    k_scale = math.log(1 + math.e) / math.log(2)

    # Project: [T, HEADS, HEAD_DIM]
    q = linear(hs, wq).float().view(T, HEADS, HEAD_DIM)
    k = linear(hs, wk).float().view(T, HEADS, HEAD_DIM)
    v = linear(hs, wv).float().view(T, HEADS, HEAD_DIM)

    # Scale
    q = q * q_scale * F.softplus(per_dim_scale.float())
    k = k * k_scale

    # Blocks: [num_blocks, CHUNK, H, D]
    num_blocks = (T + CHUNK - 1) // CHUNK
    pad = num_blocks * CHUNK - T
    q_padded = F.pad(q, (0, 0, 0, 0, 0, pad))
    q_blocks = q_padded.view(num_blocks, CHUNK, HEADS, HEAD_DIM)

    # Context windows for K/V: pad left=MAX_PAST, right=MAX_FUT+CHUNK-1
    # unfold(0, CTX_SIZE, CHUNK) on [N, H, D] gives [nb, H, D, CTX]
    k_pad = F.pad(k, (0, 0, 0, 0, MAX_PAST, MAX_FUT + CHUNK - 1))
    k_ctx = k_pad.unfold(0, CTX_SIZE, CHUNK).transpose(2, 3)  # [nb, H, CTX, D]
    v_pad = F.pad(v, (0, 0, 0, 0, MAX_PAST, MAX_FUT + CHUNK - 1))
    v_ctx = v_pad.unfold(0, CTX_SIZE, CHUNK).transpose(2, 3)  # [nb, H, CTX, D]

    # Relative keys: pos_embed [13, 1024] → linear → [13, 8*128] → [13, 8, 128] → permute → [8, 128, 13]
    rel_k = linear(pos_embed.float(), wrel)  # [13, 1024]
    rel_k = rel_k.view(-1, HEADS, HEAD_DIM)  # [13, 8, 128]
    rel_k = rel_k.permute(1, 2, 0)  # [8, 128, 13]

    # 5D layout: [1, H, nb, C, D]
    q_5d = q_blocks.permute(2, 0, 1, 3).unsqueeze(0)  # [1, H, nb, C, D]
    k_5d = k_ctx.permute(1, 0, 2, 3).unsqueeze(0)     # [1, H, nb, CTX, D]
    v_5d = v_ctx.permute(1, 0, 2, 3).unsqueeze(0)     # [1, H, nb, CTX, D]

    # matrix_ac: [1, H, nb, C, D] @ [1, H, nb, D, CTX] → [1, H, nb, C, CTX]
    matrix_ac = q_5d @ k_5d.permute(0, 1, 2, 4, 3)

    # matrix_bd: rel_shift trick
    q_flat = q_5d.reshape(1, HEADS, -1, HEAD_DIM)  # [1, H, nb*C, D]
    matrix_bd = q_flat @ rel_k.unsqueeze(0)  # [1, H, nb*C, D] @ [1, H, D, 13] → [1, H, nb*C, 13]
    matrix_bd = matrix_bd.reshape(1, HEADS, num_blocks, CHUNK, -1)
    # rel_shift: pad to CTX_SIZE+1, reshape, slice, reshape
    ctx_half_plus_1 = matrix_bd.shape[-1]  # 13
    matrix_bd = F.pad(matrix_bd, (0, CTX_SIZE + 1 - ctx_half_plus_1))  # pad to 25
    matrix_bd = matrix_bd.view(1, HEADS, num_blocks, CHUNK * (CTX_SIZE + 1))
    matrix_bd = matrix_bd[..., :CHUNK * CTX_SIZE]  # slice to CHUNK*24
    matrix_bd = matrix_bd.view(1, HEADS, num_blocks, CHUNK, CTX_SIZE)

    # Combine + softcap
    attn_weights = matrix_ac + matrix_bd
    attn_weights = attn_weights / SOFTCAP
    attn_weights = torch.tanh(attn_weights)
    attn_weights = attn_weights * SOFTCAP

    # Softmax + attention
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
    attn_out = attn_weights @ v_5d  # [1, H, nb, C, CTX] @ [1, H, nb, CTX, D] → [1, H, nb, C, D]

    # Reshape back: [H, nb, C, D] → permute → [nb, C, H, D] → [nb*C, H*D]
    attn_out = attn_out.squeeze(0).permute(1, 2, 0, 3).reshape(num_blocks * CHUNK, -1)
    attn_out = attn_out[:T].contiguous()

    # Post projection
    attn_out = linear(attn_out, wpost)
    return attn_out


def main():
    sscp_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/sscp_ref.pt'
    sscp = torch.load(sscp_path, map_location='cpu').float()  # [T, 1024]
    T = sscp.shape[0]
    print(f'SSCP input: {sscp.shape}')

    # Relative position encoding
    ctx_half = CTX_SIZE // 2  # 12
    num_timescales = H // 2  # 512
    log_increment = math.log(10000.0 / 1.0) / max(num_timescales - 1, 1)
    inv_timescales = 1.0 * torch.exp(torch.arange(num_timescales, dtype=torch.float32) * -log_increment)
    position_ids = torch.arange(ctx_half, -1, -1, dtype=torch.float32).unsqueeze(1)  # [13, 1]
    scaled_time = position_ids * inv_timescales.unsqueeze(0)
    pos_embed = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)  # [13, 1024]
    print(f'pos_embed: {pos_embed.shape}')

    with safe_open(f'{CKPT}/model.safetensors', framework='pt') as f:
        ap = 'model.audio_tower.'
        hs = sscp
        for i in range(L):
            lp = f'{ap}layers.{i}.'
            # FF1
            hs = audio_ffn(hs,
                           f.get_tensor(lp + 'feed_forward1.ffw_layer_1.linear.weight').float(),
                           f.get_tensor(lp + 'feed_forward1.ffw_layer_2.linear.weight').float(),
                           f.get_tensor(lp + 'feed_forward1.pre_layer_norm.weight').float(),
                           f.get_tensor(lp + 'feed_forward1.post_layer_norm.weight').float())
            residual = hs
            # Attention
            hs_normed = rms_norm(hs, f.get_tensor(lp + 'norm_pre_attn.weight').float())
            hs = audio_attention(hs_normed,
                                 f.get_tensor(lp + 'self_attn.q_proj.linear.weight').float(),
                                 f.get_tensor(lp + 'self_attn.k_proj.linear.weight').float(),
                                 f.get_tensor(lp + 'self_attn.v_proj.linear.weight').float(),
                                 f.get_tensor(lp + 'self_attn.post.linear.weight').float(),
                                 f.get_tensor(lp + 'self_attn.relative_k_proj.weight').float(),
                                 f.get_tensor(lp + 'self_attn.per_dim_scale').float(),
                                 pos_embed)
            hs = rms_norm(hs, f.get_tensor(lp + 'norm_post_attn.weight').float()) + residual
            # Light Conv1d
            hs = audio_lconv1d(hs,
                               f.get_tensor(lp + 'lconv1d.linear_start.linear.weight').float(),
                               f.get_tensor(lp + 'lconv1d.linear_end.linear.weight').float(),
                               f.get_tensor(lp + 'lconv1d.depthwise_conv1d.weight').float(),
                               f.get_tensor(lp + 'lconv1d.pre_layer_norm.weight').float(),
                               f.get_tensor(lp + 'lconv1d.conv_norm.weight').float())
            # FF2
            hs = audio_ffn(hs,
                           f.get_tensor(lp + 'feed_forward2.ffw_layer_1.linear.weight').float(),
                           f.get_tensor(lp + 'feed_forward2.ffw_layer_2.linear.weight').float(),
                           f.get_tensor(lp + 'feed_forward2.pre_layer_norm.weight').float(),
                           f.get_tensor(lp + 'feed_forward2.post_layer_norm.weight').float())
            # Norm out
            hs = rms_norm(hs, f.get_tensor(lp + 'norm_out.weight').float())
            print(f'Layer {i}: mean={hs.mean():.4f}, std={hs.std():.4f}, max={hs.abs().max():.4f}')
            if i == 0:
                torch.save(hs, '/tmp/layer0_ref.pt')

        # Output projection (has bias)
        out_w = f.get_tensor(ap + 'output_proj.weight').float()
        out_b = f.get_tensor(ap + 'output_proj.bias').float()
        output = F.linear(hs, out_w, out_b)
        print(f'Output: {output.shape}, mean={output.mean():.4f}, std={output.std():.4f}')

    torch.save(output, '/tmp/encoder_ref.pt')
    print(f'Saved /tmp/encoder_ref.pt')


if __name__ == '__main__':
    main()
