#!/usr/bin/env python3
"""Validate the C audio encoder against PyTorch reference.

Compares mel spectrogram, SSCP output, and (optionally) full encoder output.
"""
import sys, os, math, struct, json
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

CKPT = '/home/astro/ml/gemma4_rt/.cache/huggingface/models--google--gemma-4-E2B-it-qat-q4_0-unquantized/snapshots/6befbaca7398925921802abd1f277b495b78b738'
AUDIO_BIN = '/home/astro/ml/gemma4_rt/gemma4.c/gemma4-E2B-int8-audio.bin'
WAV = sys.argv[1] if len(sys.argv) > 1 else '/home/astro/whats_you_name.wav'


def load_wav_16k(path):
    """Load WAV and resample to 16kHz mono float32."""
    with open(path, 'rb') as f:
        data = f.read()
    # Parse RIFF
    assert data[:4] == b'RIFF' and data[8:12] == b'WAVE'
    pos = 12
    fmt = None
    audio_data = None
    while pos < len(data):
        chunk_id = data[pos:pos+4]
        chunk_size = struct.unpack('<I', data[pos+4:pos+8])[0]
        if chunk_id == b'fmt ':
            audio_format, channels, sr, _, _, bits = struct.unpack('<HHIIHH', data[pos+8:pos+24])
            fmt = (audio_format, channels, sr, bits)
        elif chunk_id == b'data':
            audio_data = data[pos+8:pos+8+chunk_size]
            break
        pos += 8 + chunk_size + (chunk_size & 1)
    assert fmt is not None, "no fmt chunk"
    assert audio_data is not None, "no data chunk"
    audio_format, channels, sr, bits = fmt
    assert channels == 1, f"mono only, got {channels}"
    if bits == 16 and audio_format == 1:
        samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    elif bits == 32 and audio_format == 3:
        samples = np.frombuffer(audio_data, dtype=np.float32)
    else:
        raise ValueError(f"unsupported: {bits}-bit fmt {audio_format}")
    # Resample to 16kHz
    if sr != 16000:
        n_new = int(len(samples) * 16000 / sr)
        t_old = np.arange(len(samples)) / sr
        t_new = np.arange(n_new) / 16000
        samples = np.interp(t_new, t_old, samples).astype(np.float32)
        sr = 16000
    return samples, sr


def compute_mel(samples):
    """Compute log-mel spectrogram matching transformers' Gemma4AudioFeatureExtractor."""
    fft_len, frame_len, hop_len = 512, 320, 160
    num_bins = 128
    sr = 16000

    hann = torch.hann_window(frame_len, periodic=True)
    x = torch.from_numpy(samples).float()

    # Semicausal padding
    pad = frame_len // 2
    x = F.pad(x.unsqueeze(0).unsqueeze(0), (pad, 0))
    # frame_size_for_unfold = frame_length + 1 (matches transformers)
    frame_size_for_unfold = frame_len + 1
    num_frames = (x.shape[-1] - frame_size_for_unfold) // hop_len + 1
    # Unfold to 321, drop last sample → 320 (matches: frames_to_process[..., :-1] when preemphasis=0)
    frames = x.unfold(-1, frame_size_for_unfold, hop_len)[..., :frame_len]
    windows = frames * hann  # hann is [320]
    spec = torch.fft.rfft(windows, n=fft_len)
    # Magnitude (matches transformers: np.abs(stft)), NOT power
    magnitude = torch.sqrt(spec.real**2 + spec.imag**2)  # [1, num_frames, 257]

    # HTK mel filterbank using transformers' slope-based construction
    def hz_to_mel(hz): return 2595.0 * math.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    mel_freqs = np.linspace(hz_to_mel(0), hz_to_mel(8000), num_bins + 2)
    filter_freqs = np.array([mel_to_hz(m) for m in mel_freqs])
    fft_freqs = np.linspace(0, sr // 2, fft_len // 2 + 1)
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    filters = np.maximum(0.0, np.minimum(down_slopes, up_slopes)).astype(np.float32)
    mel_fb = torch.from_numpy(filters)  # [257, 128]

    mel = magnitude.squeeze(0) @ mel_fb  # [num_frames, 128]
    # Additive floor (matches transformers: log(mel + floor))
    mel = torch.log(mel + 1e-3)
    return mel  # [num_frames, 128]


def read_audio_bin_header(path):
    """Read the inlined tables from the audio binary."""
    with open(path, 'rb') as f:
        data = f.read(147264)  # just the struct
    magic = data[0:4]
    assert magic == b'MOGA', f"bad magic: {magic}"
    hann = np.frombuffer(data[12:12+512*4], dtype=np.float32)
    mel_f = np.frombuffer(data[2060:2060+257*128*4], dtype=np.float32).reshape(257, 128)
    return hann, mel_f


def sscp_torch(mel_spec, w0, n0, w1, n1, wp):
    """Run SSCP in torch for reference."""
    x = mel_spec.unsqueeze(0).unsqueeze(0)  # [1, 1, T, F]
    x = F.conv2d(x, w0, padding=1, stride=2)
    # LayerNorm over channels (128) — need channels last
    x = x.permute(0, 2, 3, 1)  # [1, H, W, C]
    x = F.layer_norm(x, (128,), weight=n0)
    x = F.relu(x)
    x = x.permute(0, 3, 1, 2)  # [1, C, H, W]
    x = F.conv2d(x, w1, padding=1, stride=2)
    x = x.permute(0, 2, 3, 1)  # [1, H, W, C]
    x = F.layer_norm(x, (32,), weight=n1)
    x = F.relu(x)
    B, H, W, C = x.shape
    x = x.reshape(B, H, W * C)
    h = x @ wp.T
    return h.squeeze(0)


def compare(name, a, b, atol=1e-3):
    """Compare two arrays, print stats."""
    a, b = a.numpy() if isinstance(a, torch.Tensor) else a, \
           b.numpy() if isinstance(b, torch.Tensor) else b
    diff = np.abs(a - b)
    max_d = diff.max()
    mean_d = diff.mean()
    cos = (a.ravel() @ b.ravel()) / (np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()) + 1e-8)
    status = "OK" if max_d < atol else "MISMATCH"
    print(f"  {name}: max={max_d:.6f} mean={mean_d:.6f} cos={cos:.6f} [{status}]")
    return max_d < atol


def main():
    print(f"Loading WAV: {WAV}")
    samples, sr = load_wav_16k(WAV)
    print(f"  {len(samples)} samples @ {sr} Hz ({len(samples)/sr:.1f}s)")

    # Load reference tables from the audio binary
    print("Loading audio binary tables...")
    bin_hann, bin_mel_fb = read_audio_bin_header(AUDIO_BIN)

    # Compute mel spectrogram (torch reference)
    print("Computing mel spectrogram (torch)...")
    mel_torch = compute_mel(samples)  # [T, 128]
    print(f"  {mel_torch.shape}")

    # Compute mel spectrogram (C-style: using the binary's tables)
    print("Computing mel spectrogram (C-style tables)...")
    fft_len, frame_len, hop_len = 512, 320, 160
    x = torch.from_numpy(samples).float()
    hann_t = torch.from_numpy(bin_hann)
    pad = frame_len // 2
    x = F.pad(x.unsqueeze(0).unsqueeze(0), (pad, 0))
    frame_size_for_unfold = frame_len + 1
    num_frames = (x.shape[-1] - frame_size_for_unfold) // hop_len + 1
    # C code uses only the first 320 of the 512 stored window samples
    hann_320 = hann_t[:frame_len]
    frames = x.unfold(-1, frame_size_for_unfold, hop_len)[..., :frame_len]
    windows = frames * hann_320
    spec = torch.fft.rfft(windows, n=fft_len)
    # Magnitude (matches C code and transformers)
    magnitude = torch.sqrt(spec.real**2 + spec.imag**2).squeeze(0)  # [T, 257]
    mel_fb_t = torch.from_numpy(bin_mel_fb)
    mel_c = magnitude @ mel_fb_t
    mel_c = torch.log(mel_c + 1e-3)
    # Per-bin normalization (from binary tables; defaults to 0/1 = no-op)
    with open(AUDIO_BIN, 'rb') as bf:
        bf.seek(133644)
        bin_mean = np.frombuffer(bf.read(512), dtype=np.float32)
        bf.seek(134156)
        bin_std = np.frombuffer(bf.read(512), dtype=np.float32)
    mel_c = (mel_c - torch.from_numpy(bin_mean)) / torch.from_numpy(bin_std)
    print(f"  {mel_c.shape}")

    print("\n=== Mel spectrogram comparison ===")
    # The C code uses the mel_filters from the binary, which should match
    # if the filterbank computation is the same. Compare torch vs C-style.
    n = min(mel_torch.shape[0], mel_c.shape[0])
    compare("mel (first 100 frames)", mel_torch[:100], mel_c[:100], atol=0.1)

    # Load SSCP weights and compute reference
    print("\n=== SSCP comparison ===")
    with safe_open(f'{CKPT}/model.safetensors', framework='pt') as f:
        ap = 'model.audio_tower.'
        w0 = f.get_tensor(ap + 'subsample_conv_projection.layer0.conv.weight').float()
        n0 = f.get_tensor(ap + 'subsample_conv_projection.layer0.norm.weight').float()
        w1 = f.get_tensor(ap + 'subsample_conv_projection.layer1.conv.weight').float()
        n1 = f.get_tensor(ap + 'subsample_conv_projection.layer1.norm.weight').float()
        wp = f.get_tensor(ap + 'subsample_conv_projection.input_proj_linear.weight').float()

    # Reference: use mel_torch [T, 128] (no per-bin norm)
    sscp_ref = sscp_torch(mel_torch.squeeze(0), w0, n0, w1, n1, wp)
    print(f"  SSCP ref: {sscp_ref.shape}")

    # For C comparison, we need to dump the C SSCP output.
    # We'll add a debug flag to the C code later; for now just save reference.
    torch.save(sscp_ref, '/tmp/sscp_ref.pt')
    torch.save(mel_torch, '/tmp/mel_torch.pt')
    print("\nSaved /tmp/sscp_ref.pt and /tmp/mel_torch.pt")
    print("Done.")


if __name__ == '__main__':
    main()
