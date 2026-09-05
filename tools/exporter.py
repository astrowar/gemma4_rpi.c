#!/usr/bin/env python3
"""Export Gemma 4 E2B to the binary consumed by gemma4.c."""

import argparse
import json
import math
import os
from math import inf
from pathlib import Path
import struct
import tempfile

import numpy as np
import safetensors
from tqdm import tqdm


N_LAYERS = 35
N_KV_LAYERS = 15
TOKEN_MAX = 94
VOCAB_MAX = 262144
BYTE_TOKEN_BASE = 238
SPECIAL_MAX = 256
ENCODE_VOCAB_MAX = 32768
MERGES_MAX = 514906
GROUP_SIZE = 64
BLOCK_ROWS = 16
BLOCK_WIDTH = 4

TOKENIZER_SPAN = 33_429_936  # 4-byte quant field + 33_429_932-byte Tokenizer
# data, scales, shape[4] — matches C Tensor (32 bytes)
TENSOR_RECORD = struct.Struct("<QQ4i")
LOOKUP_RECORD = struct.Struct("<8sii")
TEXT_TENSOR_SPAN = 55_008
HEADER_SIZE = 4  # MOG\0
GELU_TABLE_N = 4096
GELU_TABLE_LO = -8.0
GELU_TABLE_HI = 8.0

cursor = 0


def align64(value):
    return (value + 63) & ~63


def read_json(root, name):
    with (root / name).open(encoding="utf-8") as file:
        return json.load(file)


def tensor(weights, path, synthetic=None, quant_mode=8):
    global cursor
    shape = tuple(synthetic.shape) if synthetic is not None else tuple(
        int(n) for n in weights.get_slice(path).get_shape()
    )
    if not 1 <= len(shape) <= 2 or any(size <= 0 for size in shape):
        raise ValueError(f"unsupported shape for {path}: {shape}")
    quantized = synthetic is None and len(shape) == 2
    group = 64 if quant_mode == 8 else 32
    if quantized and (shape[0] % BLOCK_ROWS or shape[1] % group):
        raise ValueError(f"cannot pack {path} with shape {shape}")
    count = math.prod(shape)
    result = {
        "path": path, "shape": shape, "synthetic": synthetic,
        "data": cursor, "scales": 0, "quant_mode": quant_mode,
    }
    if quant_mode == 8:
        cursor = align64(cursor + count * (1 if quantized else 4))
        if quantized:
            result["scales"] = cursor
            cursor = align64(cursor + count // group * 2)
    elif quant_mode == 4:
        # int4: 2 weights per byte -> count // 2 bytes for quantized tensors.
        cursor = align64(cursor + (count // 2 if quantized else count * 4))
        if quantized:
            result["scales"] = cursor
            cursor = align64(cursor + count // group * 2)
    else:
        raise ValueError(f"unsupported quant mode {quant_mode}")
    return result


def gelu_table():
    x = np.linspace(GELU_TABLE_LO, GELU_TABLE_HI, GELU_TABLE_N, dtype=np.float32)
    return (0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))).astype(np.float32)


def rope_tables(config):
    def table(head_dim, fraction, theta):
        pairs = int(head_dim * fraction) // 2
        exponent = 2 * np.arange(pairs, dtype=np.float32) / np.float32(head_dim)
        frequency = np.float32(1) / np.power(np.float32(theta), exponent)
        angles = np.arange(config["max_position_embeddings"], dtype=np.float32)[:, None] * frequency
        cosine = np.cos(angles)
        np.sin(angles, out=angles)
        return cosine, angles

    sliding = table(config["head_dim"], 1,
                    config["rope_parameters"]["sliding_attention"]["rope_theta"])
    full = config["rope_parameters"]["full_attention"]
    global_ = table(config["global_head_dim"], full["partial_rotary_factor"], full["rope_theta"])
    return *sliding, *global_


def text_tensors(weights, config, quant_mode=8):
    tables = rope_tables(config)
    rope_cache = {}

    def shared_rope(index):
        if index not in rope_cache:
            rope_cache[index] = tensor(weights, None, synthetic=tables[index])
        return rope_cache[index]

    slots = [
        tensor(weights, "model.language_model.embed_tokens.weight", quant_mode=quant_mode),
        tensor(weights, "model.language_model.embed_tokens_per_layer.weight", quant_mode=quant_mode),
    ]
    for layer in range(N_LAYERS):
        p = f"model.language_model.layers.{layer}."
        shared_kv = layer < N_KV_LAYERS
        rope_index = 0 if config["layer_types"][layer] == "sliding_attention" else 2
        layer_slots = [
            tensor(weights, p + "input_layernorm.weight"), tensor(weights, p + "layer_scalar"),
            tensor(weights, p + "pre_feedforward_layernorm.weight"),
            tensor(weights, p + "post_attention_layernorm.weight"),
            tensor(weights, p + "post_feedforward_layernorm.weight"),
            tensor(weights, p + "post_per_layer_input_norm.weight"),
            tensor(weights, p + "per_layer_input_gate.weight", quant_mode=quant_mode),
            tensor(weights, p + "per_layer_projection.weight", quant_mode=quant_mode),
            tensor(weights, p + "self_attn.q_norm.weight"),
            tensor(weights, p + "self_attn.k_norm.weight") if shared_kv else None,
            tensor(weights, p + "self_attn.q_proj.weight", quant_mode=quant_mode),
            tensor(weights, p + "self_attn.k_proj.weight", quant_mode=quant_mode) if shared_kv else None,
            tensor(weights, p + "self_attn.v_proj.weight", quant_mode=quant_mode) if shared_kv else None,
            tensor(weights, p + "self_attn.o_proj.weight", quant_mode=quant_mode),
            tensor(weights, p + "mlp.gate_proj.weight", quant_mode=quant_mode),
            tensor(weights, p + "mlp.up_proj.weight", quant_mode=quant_mode),
            tensor(weights, p + "mlp.down_proj.weight", quant_mode=quant_mode),
            shared_rope(rope_index), shared_rope(rope_index + 1),
        ]
        intermediate_size = 6144 if layer < N_KV_LAYERS else 12288
        if (layer_slots[14]["shape"] != (intermediate_size, 1536)
                or layer_slots[15]["shape"] != (intermediate_size, 1536)
                or layer_slots[16]["shape"] != (1536, intermediate_size)):
            raise ValueError(f"unexpected MLP shapes in layer {layer}")
        if shared_kv:
            head_width = 512 if rope_index == 2 else 256
            if (layer_slots[11]["shape"] != (head_width, 1536)
                    or layer_slots[12]["shape"] != (head_width, 1536)):
                raise ValueError(f"unexpected K/V shapes in layer {layer}")
        slots += layer_slots
    gelu = tensor(weights, None, synthetic=gelu_table())
    gelu["shape"] = (GELU_TABLE_N, int(GELU_TABLE_LO), int(GELU_TABLE_HI))
    slots += [
        tensor(weights, "model.language_model.norm.weight"),
        tensor(weights, "model.language_model.per_layer_model_projection.weight", quant_mode=quant_mode),
        tensor(weights, "model.language_model.per_layer_projection_norm.weight"),
        gelu,
    ]
    return slots


def validate_text_config(model):
    if model.get("model_type") != "gemma4":
        raise ValueError("checkpoint is not Gemma 4")
    text = model["text_config"]
    if ((text["hidden_size"], text["vocab_size"], text["num_hidden_layers"],
            text["num_hidden_layers"] - text["num_kv_shared_layers"],
            text["num_key_value_heads"], text["head_dim"],
            text["global_head_dim"], text["intermediate_size"],
            text["use_double_wide_mlp"]) !=
            (1536, VOCAB_MAX, N_LAYERS, N_KV_LAYERS, 1, 256, 512, 6144, True)):
        raise ValueError("expected Gemma 4 E2B")
    if (text["max_position_embeddings"], text["sliding_window"],
            text["hidden_size_per_layer_input"]) != (131072, 512, 256):
        raise ValueError("expected Gemma 4 E2B text configuration")
    if (np.float32(text["rms_norm_eps"]) != np.float32(float.fromhex("0x1.0c6f7ap-20"))
            or np.float32(text["final_logit_softcapping"]) != np.float32(30.0)):
        raise ValueError("expected Gemma 4 E2B text configuration")
    layer_types = ["sliding_attention"] * 4 + ["full_attention"]
    if text["layer_types"] != layer_types * 7:
        raise ValueError("expected Gemma 4 E2B attention sequence")
    return text


def put_string(buffer, offset, width, value):
    encoded = value.encode("utf-8")
    if b"\0" in encoded or len(encoded) >= width:
        raise ValueError(f"token cannot fit char[{width}]")
    buffer[offset:offset + len(encoded)] = encoded


def decoded_token(token_id, token, special_ids):
    if token_id in special_ids:
        payload = b""
    elif len(token) == 6 and token[:3] == "<0x" and token[5] == ">" \
            and all(char in "0123456789abcdefABCDEF" for char in token[3:5]):
        payload = bytes([int(token[3:5], 16)])
    else:
        payload = token.replace("▁", " ").encode("utf-8")
        if b"\0" in payload:
            raise ValueError(f"token cannot fit char[{TOKEN_MAX}]")
    if len(payload) >= TOKEN_MAX:
        raise ValueError(f"token cannot fit char[{TOKEN_MAX}]")
    return payload


def compile_tokenizer(data):
    raw_vocab = data["model"]["vocab"]
    raw_merges = data["model"]["merges"]
    added = data["added_tokens"]
    vocab = dict(raw_vocab)
    for token in added:
        content, token_id = token["content"], token["id"]
        if content in vocab and vocab[content] != token_id:
            raise ValueError(f"conflicting token ID for {content!r}")
        vocab[content] = token_id
    ids = [None] * (max(vocab.values()) + 1)
    for token, token_id in vocab.items():
        if token_id < 0 or token_id >= VOCAB_MAX or ids[token_id] is not None:
            raise ValueError("token IDs must be unique and dense")
        ids[token_id] = token
    if len(ids) != VOCAB_MAX or any(token is None for token in ids):
        raise ValueError("token IDs must be unique and dense")

    merges, results, pairs = [], set(), set()
    for rank, merge in enumerate(raw_merges):
        left, right = merge if isinstance(merge, list) else merge.split(" ", 1)
        pair = vocab[left], vocab[right]
        if pair in pairs or left + right not in vocab:
            raise ValueError(f"invalid merge {merge!r}")
        pairs.add(pair)
        results.add(left + right)
        merges.append((struct.pack("<2i", *pair), vocab[left + right], rank))
    specials = sorted(
        ((token["id"], token["content"]) for token in added if token["special"]),
        key=lambda item: (-len(item[1].encode()), item[1].encode()),
    )
    special_ids = {token_id for token_id, _ in specials}
    special_tokens = {token for _, token in specials}
    for byte in range(256):
        token = f"<0x{byte:02X}>"
        if (vocab.get(token) != BYTE_TOKEN_BASE + byte
                or token in special_tokens or token in results):
            raise ValueError("byte tokens must be contiguous non-special non-merge entries")
    encode = [(token.encode().ljust(8, b"\0"), token_id, 0)
              for token_id, token in enumerate(ids)
              if token not in special_tokens and token not in results
              and len(token.encode()) <= 4]
    if len(specials) > SPECIAL_MAX or len(encode) > ENCODE_VOCAB_MAX \
            or len(merges) > MERGES_MAX:
        raise ValueError("tokenizer exceeds fixed MOG limits")

    buffer = bytearray(TOKENIZER_SPAN)
    struct.pack_into("<3i", buffer, 0, len(merges), len(encode), len(specials))
    for token_id, token in enumerate(ids):
        payload = decoded_token(token_id, token, special_ids)
        offset = 12 + token_id * TOKEN_MAX
        buffer[offset:offset + len(payload)] = payload
    special_offset = 12 + VOCAB_MAX * TOKEN_MAX
    encode_offset = special_offset + SPECIAL_MAX * 100
    merge_offset = encode_offset + ENCODE_VOCAB_MAX * LOOKUP_RECORD.size
    for index, (token_id, token) in enumerate(specials):
        put_string(buffer, special_offset + index * 100, TOKEN_MAX, token)
        struct.pack_into("<i", buffer, special_offset + index * 100 + 96, token_id)
    for base, entries in ((encode_offset, encode), (merge_offset, merges)):
        for index, entry in enumerate(sorted(entries)):
            LOOKUP_RECORD.pack_into(buffer, base + index * LOOKUP_RECORD.size, *entry)
    return bytes(buffer)


def tensor_record(tensor):
    shape = tensor["shape"] + (0,) * (4 - len(tensor["shape"]))
    return TENSOR_RECORD.pack(tensor["data"], tensor["scales"], *shape)


def quantize(array, quant_mode=8):
    group = 64 if quant_mode == 8 else 32
    groups = array.reshape(-1, group)
    # Round the scale to its fp16 storage precision first, then quantize against the rounded value so the runtime dequantizes with the exact scale used here. Floor at the smallest normal fp16.
    if quant_mode == 8:
        scales = np.maximum(np.abs(groups).max(axis=1) / 127.0, 6.104e-05).astype("<f2")
        wide = scales.astype("<f4")
        values = np.rint(groups / wide[:, None]).clip(-127, 127).astype(np.int8)
        return values, scales
    # int4: symmetric range [-8, 7] (zero point 8), scale = max_abs / 8.
    scales = np.maximum(np.abs(groups).max(axis=1) / 8.0, 6.104e-05).astype("<f2")
    wide = scales.astype("<f4")
    values = np.rint(groups / wide[:, None]).clip(-8, 7).astype(np.int8)
    return values, scales


def read_weight(weights, path, rows=None):
    tensor = weights.get_tensor(path) if rows is None else weights.get_slice(path)[rows]
    return tensor.float().contiguous().numpy()


def write_tensor(output, weights, tensor, progress):
    if tensor["synthetic"] is not None:
        arrays = [np.asarray(tensor["synthetic"], dtype="<f4", order="C")]
    elif len(tensor["shape"]) == 2:
        total_rows, columns = tensor["shape"]
        chunk_rows = max(1, (16 * 1024 * 1024) // columns)
        chunk_rows = max(BLOCK_ROWS, chunk_rows - chunk_rows % BLOCK_ROWS)
        arrays = (read_weight(weights, tensor["path"],
                              slice(start, min(total_rows, start + chunk_rows)))
                  for start in range(0, total_rows, chunk_rows))
    else:
        arrays = [read_weight(weights, tensor["path"])]

    data_at, scales_at = tensor["data"], tensor["scales"]
    for array in arrays:
        array = np.asarray(array, dtype="<f4", order="C")
        if not scales_at:
            output.seek(data_at)
            output.write(array.tobytes())
            data_at += array.nbytes
            progress.update(array.size)
            continue

        values, scales = quantize(array, tensor["quant_mode"])
        rows, columns = array.shape
        blocks, groups = rows // BLOCK_ROWS, columns // (64 if tensor["quant_mode"] == 8 else 32)
        if tensor["quant_mode"] == 8:
            values = values.reshape(rows, columns).reshape(
                blocks, BLOCK_ROWS, groups, GROUP_SIZE // BLOCK_WIDTH, BLOCK_WIDTH
            ).transpose(0, 2, 3, 1, 4).reshape(-1)
            packed = values.tobytes()
        else:
            # int4: pack two 4-bit values per byte. Each row's group is 32
            # values -> 16 bytes; low nibble = even input, high = odd.
            # Layout: [block][group][row][16 bytes] -> group stride is
            # BLOCK_ROWS*16 (16 rows x 16 bytes), matching the runtime kernels.
            packed_values = values.reshape(rows, columns).reshape(
                blocks, BLOCK_ROWS, groups, 16, 2
            ).transpose(0, 2, 1, 3, 4)
            low = (packed_values[..., 0::2] + 8).astype(np.uint8)
            high = (packed_values[..., 1::2] + 8).astype(np.uint8)
            packed = (low | (high << 4)).reshape(-1).tobytes()
        scales = scales.reshape(blocks, BLOCK_ROWS, groups).transpose(0, 2, 1).reshape(-1)
        output.seek(data_at)
        output.write(packed)
        data_at += len(packed)
        output.seek(scales_at)
        output.write(scales.tobytes())
        scales_at += scales.nbytes
        progress.update(array.size)


def export(checkpoint_path, output_path, quant_mode=8):
    global cursor
    if quant_mode not in (8, 4):
        raise ValueError(f"unsupported quant mode {quant_mode}")
    checkpoint_path = checkpoint_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading checkpoint (exporting {quant_mode}-bit weights)...")
    model = read_json(checkpoint_path, "config.json")
    tokenizer = compile_tokenizer(read_json(checkpoint_path, "tokenizer.json"))
    text_config = validate_text_config(model)

    weights_path = checkpoint_path / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError("checkpoint is missing model.safetensors")
    cursor = align64(HEADER_SIZE + TOKENIZER_SPAN + TEXT_TENSOR_SPAN)
    with safetensors.safe_open(weights_path, framework="pt") as weights:
        text = text_tensors(weights, text_config, quant_mode)
        file_size = cursor

        with tempfile.TemporaryDirectory(
            prefix=f".{output_path.name}.", dir=output_path.parent,
        ) as directory:
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as output:
                temporary_path = Path(output.name)
                output.truncate(file_size)
                output.write(b"MOG\0")
                output.write(struct.pack("<i", quant_mode))
                output.write(tokenizer)
                for item in text:
                    record = tensor_record(item) if item else bytes(TENSOR_RECORD.size)
                    output.write(record)

                seen = set()
                items = []
                for item in text:
                    if item is None or item["data"] in seen:
                        continue
                    seen.add(item["data"])
                    items.append(item)
                total = sum(math.prod(item["shape"]) for item in items)
                with tqdm(total=total, desc="Quantizing weights", unit="weight", unit_scale=True) as progress:
                    for item in items:
                        write_tensor(output, weights, item, progress)
                print("Finalizing model file...")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, output_path)

    text_count = len({item["data"] for item in text if item})
    print(f"Done {output_path} ({file_size / 1024**3:.2f} GiB): {text_count} text tensors")


# ----------------------------------------------------------------------------
# Audio export

AUDIO_MAGIC = b"MOGA"
AUDIO_LAYERS = 12
AUDIO_HIDDEN = 1024
AUDIO_HEADS = 8
AUDIO_HEAD_DIM = 128
AUDIO_FFN = 4096
AUDIO_OUTPUT = 1536
AUDIO_MEL_BINS = 128
AUDIO_FFT_LEN = 512
AUDIO_FRAME_LEN = 320
AUDIO_HOP_LEN = 160
AUDIO_CONTEXT = 24  # chunk(12) + past(12) + future(0)
AUDIO_NUM_POSITIONS = AUDIO_CONTEXT // 2 + 1  # 13


def compute_hann_window(n):
    """Periodic Hann window: w[k] = 0.5 - 0.5 * cos(2*pi*k / n)"""
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)


def compute_mel_filterbank(num_fft_bins, num_mels, min_freq, max_freq, sr, mel_scale="htk"):
    """HTK mel filterbank matching transformers' mel_filter_bank.

    Uses the same slope-based construction as transformers.audio_utils._create_triangular_filter_bank:
    fft_freqs = linspace(0, sr/2, num_fft_bins)
    triangular slopes computed in frequency space.
    """
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    # Center points of the triangular mel filters (edges for num_mels triangles)
    mel_min = hz_to_mel(min_freq)
    mel_max = hz_to_mel(max_freq)
    mel_freqs = np.linspace(mel_min, mel_max, num_mels + 2)
    filter_freqs = mel_to_hz(mel_freqs)  # [num_mels + 2] in Hz

    # Frequencies of FFT bins in Hz
    fft_freqs = np.linspace(0, sr // 2, num_fft_bins)

    # Triangular filter construction (same as transformers)
    filter_diff = np.diff(filter_freqs)  # [num_mels]
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)  # [num_fft_bins, num_mels+2]
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]  # [num_fft_bins, num_mels]
    up_slopes = slopes[:, 2:] / filter_diff[1:]  # [num_fft_bins, num_mels]
    filters = np.maximum(0.0, np.minimum(down_slopes, up_slopes)).astype(np.float32)
    return filters


def tensor_flat(weights, path):
    """Read an N-D tensor and store it as a flat float32 synthetic."""
    arr = weights.get_tensor(path).float().numpy().ravel().astype(np.float32)
    return tensor(weights, None, synthetic=arr)


def validate_audio_config(config):
    audio = config.get("audio_config")
    if audio is None:
        raise ValueError("checkpoint has no audio_config")
    if audio["hidden_size"] != AUDIO_HIDDEN:
        raise ValueError(f"expected audio hidden_size={AUDIO_HIDDEN}, got {audio['hidden_size']}")
    if audio["num_hidden_layers"] != AUDIO_LAYERS:
        raise ValueError(f"expected audio num_hidden_layers={AUDIO_LAYERS}, got {audio['num_hidden_layers']}")
    return audio


def _get_clip(weights, prefix):
    """Read 4 clip-bound floats from checkpoint."""
    return np.array([
        weights.get_tensor(prefix + "input_min").item(),
        weights.get_tensor(prefix + "input_max").item(),
        weights.get_tensor(prefix + "output_min").item(),
        weights.get_tensor(prefix + "output_max").item(),
    ], dtype=np.float32)


def export_audio(checkpoint_path, output_path, quant_mode=8):
    """Export audio encoder to a binary that is a raw AudioModel struct + tensor data."""
    checkpoint_path = checkpoint_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = read_json(checkpoint_path, "config.json")
    validate_audio_config(config)
    for preproc_name in ["audio_preprocessor_config.json", "preprocessor_config.json"]:
        preproc_path = checkpoint_path / preproc_name
        if preproc_path.is_file():
            preproc = read_json(checkpoint_path, preproc_name)
            if "per_bin_mean" in preproc:
                config.setdefault("audio_config", {})["per_bin_mean"] = preproc["per_bin_mean"]
            if "per_bin_stddev" in preproc:
                config.setdefault("audio_config", {})["per_bin_stddev"] = preproc["per_bin_stddev"]
            break
    audio_cfg = config.get("audio_config", config)
    boa_token_id = config.get("boa_token_id", 256000)
    eoa_token_id = config.get("eoa_token_index", config.get("eoa_token_id", 258883))
    audio_token_id = config.get("audio_token_id", 258881)

    # Struct layout constants (verified against C sizeof/offsetof)
    STRUCT_SIZE = 147264
    OFF_MAGIC = 0
    OFF_QUANT = 4
    OFF_VERSION = 8
    OFF_HANN = 12          # 512 floats
    OFF_MEL = 2060         # 257*128 floats
    OFF_MEAN = 133644      # 128 floats
    OFF_STD = 134156       # 128 floats
    OFF_SSCP = 134672      # AudioSSCP (192 bytes = 6 slots)
    OFF_LAYERS = 134864    # 12 x AudioLayer (1024 bytes = 32 slots each)
    OFF_OUT_PROJ = 147152  # Tensor (32 bytes)
    OFF_OUT_BIAS = 147184  # Tensor (32 bytes)
    OFF_EMBED_PROJ = 147216 # Tensor (32 bytes)
    OFF_BOA = 147248
    OFF_EOA = 147252
    OFF_AUDIO_TOK = 147256

    # Each Tensor is 32 bytes: <qq4i> (data_u64, scales_u64, shape[4]_i32)
    # Each ClippableLinear is 64 bytes: Tensor(32) + 4*float(16) + pad(16)
    # The 32-byte stepper visits: slot 0 = Tensor, slot 1 = clip floats + pad

    def tensor_entry(data_off, scales_off, shape):
        """Pack a 32-byte Tensor: data, scales, shape[4]."""
        s = list(shape) + [0] * (4 - len(shape))
        return struct.pack("<qq4i", data_off, scales_off, *s)

    def clip_entry(bounds):
        """Pack 16 bytes: 4 clip floats + 16 pad (the part after Tensor in ClippableLinear)."""
        return struct.pack("<4f16x", *bounds)

    with safetensors.safe_open(checkpoint_path / "model.safetensors", framework="pt") as weights:
        ap = "model.audio_tower."

        # Phase 1: collect all tensor data (quantize now)
        # Returns list of (shape, data_bytes, scales_bytes) for real tensors
        tensor_blobs = []  # (shape, packed_data, scales)

        def add_linear(path, clip_prefix=None):
            """Add a 2D int8 linear. Returns (shape, data, scales)."""
            arr = np.asarray(read_weight(weights, path), dtype="<f4", order="C")
            values, scales = quantize(arr, quant_mode)
            rows, cols = arr.shape
            groups = cols // (64 if quant_mode == 8 else 32)
            blocks = rows // BLOCK_ROWS
            values_packed = values.reshape(rows, cols).reshape(
                blocks, BLOCK_ROWS, groups, GROUP_SIZE // BLOCK_WIDTH, BLOCK_WIDTH
            ).transpose(0, 2, 3, 1, 4).reshape(-1)
            tensor_blobs.append(((rows, cols), values_packed.tobytes(), scales.tobytes()))
            return (rows, cols), values_packed.size, scales.nbytes

        def add_float(path, shape=None):
            """Add a float32 tensor (1D or N-D). Returns (shape, size)."""
            arr = np.asarray(read_weight(weights, path), dtype="<f4", order="C")
            if shape is None:
                shape = arr.shape
            flat = arr.ravel().astype("<f4")
            tensor_blobs.append((tuple(shape), flat.tobytes(), b""))
            return tuple(shape), flat.size * 4

        # Phase 2: build the struct as a bytearray
        buf = bytearray(STRUCT_SIZE)

        # Header
        buf[OFF_MAGIC:OFF_MAGIC+4] = b"MOGA"
        buf[OFF_QUANT:OFF_QUANT+4] = struct.pack("<i", quant_mode)
        buf[OFF_VERSION:OFF_VERSION+4] = struct.pack("<i", 1)

        # Inlined tables
        # C code applies hann[i] for i < AUDIO_FRAME_LEN (320), rest is zero-pad
        hann_320 = compute_hann_window(320).astype("<f4")
        hann_512 = np.zeros(512, dtype=np.float32)
        hann_512[:320] = hann_320
        buf[OFF_HANN:OFF_HANN+512*4] = hann_512.tobytes()
        mel_fb = compute_mel_filterbank(257, AUDIO_MEL_BINS, 0, 8000, 16000).astype("<f4")
        buf[OFF_MEL:OFF_MEL+257*128*4] = mel_fb.reshape(-1).tobytes()
        per_bin_mean = audio_cfg.get("per_bin_mean", [0.0]*AUDIO_MEL_BINS)
        per_bin_stddev = audio_cfg.get("per_bin_stddev", [1.0]*AUDIO_MEL_BINS)
        buf[OFF_MEAN:OFF_MEAN+128*4] = np.array(per_bin_mean, dtype="<f4").tobytes()
        buf[OFF_STD:OFF_STD+128*4] = np.array(per_bin_stddev, dtype="<f4").tobytes()

        # We'll fill in Tensor fields once we know data offsets.
        # For now, record where each Tensor goes in the struct.
        tensor_offsets = []  # list of (struct_offset, blob_index, is_clippable, clip_prefix)

        def sscp_off(field_off):
            return OFF_SSCP + field_off

        def layer_off(layer_idx, field_off):
            return OFF_LAYERS + layer_idx * 1024 + field_off

        # --- SSCP (6 slots) ---
        # slot 0: conv0_weight [128,1,3,3] float
        shape, sz = add_float(ap + "subsample_conv_projection.layer0.conv.weight", (128,1,3,3))
        tensor_offsets.append((sscp_off(0), len(tensor_blobs)-1, False, None))
        # slot 1: norm0_weight [128] float
        shape, sz = add_float(ap + "subsample_conv_projection.layer0.norm.weight", (128,))
        tensor_offsets.append((sscp_off(32), len(tensor_blobs)-1, False, None))
        # slot 2: conv1_weight [32,128,3,3] float
        shape, sz = add_float(ap + "subsample_conv_projection.layer1.conv.weight", (32,128,3,3))
        tensor_offsets.append((sscp_off(64), len(tensor_blobs)-1, False, None))
        # slot 3: norm1_weight [32] float
        shape, sz = add_float(ap + "subsample_conv_projection.layer1.norm.weight", (32,))
        tensor_offsets.append((sscp_off(96), len(tensor_blobs)-1, False, None))
        # slot 4: input_proj [1024,1024] float32 (int8 too lossy for skewed ReLU input)
        shape, sz = add_float(ap + "subsample_conv_projection.input_proj_linear.weight")
        tensor_offsets.append((sscp_off(128), len(tensor_blobs)-1, False, None))
        # slot 5: clip bounds (unused for float32)
        buf[sscp_off(176):sscp_off(192)] = struct.pack("<4f", -inf, inf, -inf, inf)

        # --- 12 layers ---
        for i in range(AUDIO_LAYERS):
            lp = ap + f"layers.{i}."
            lo = layer_off(i, 0)

            # ffn1 (6 slots: 0,32,64,96,128,160) — float32 for accuracy
            shape, sz = add_float(lp + "feed_forward1.ffw_layer_1.linear.weight")
            tensor_offsets.append((lo + 0, len(tensor_blobs)-1, False, None))
            buf[lo+48:lo+64] = struct.pack("<4f", *_get_clip(weights, lp+"feed_forward1.ffw_layer_1."))
            shape, sz = add_float(lp + "feed_forward1.ffw_layer_2.linear.weight")
            tensor_offsets.append((lo + 64, len(tensor_blobs)-1, False, None))
            buf[lo+112:lo+128] = struct.pack("<4f", *_get_clip(weights, lp+"feed_forward1.ffw_layer_2."))
            shape, sz = add_float(lp + "feed_forward1.pre_layer_norm.weight", (1024,))
            tensor_offsets.append((lo + 128, len(tensor_blobs)-1, False, None))
            shape, sz = add_float(lp + "feed_forward1.post_layer_norm.weight", (1024,))
            tensor_offsets.append((lo + 160, len(tensor_blobs)-1, False, None))

            # attn (10 slots: 192-447) — float32 for accuracy
            for j, pn in enumerate(["q_proj", "k_proj", "v_proj", "post"]):
                shape, sz = add_float(lp + f"self_attn.{pn}.linear.weight")
                tensor_offsets.append((lo + 192 + j*64, len(tensor_blobs)-1, False, None))
                buf[lo+192+j*64+48 : lo+192+j*64+64] = struct.pack(
                    "<4f", *_get_clip(weights, lp+f"self_attn.{pn}."))
            shape, sz = add_float(lp + "self_attn.relative_k_proj.weight", (1024,1024))
            tensor_offsets.append((lo + 448, len(tensor_blobs)-1, False, None))
            shape, sz = add_float(lp + "self_attn.per_dim_scale", (128,))
            tensor_offsets.append((lo + 480, len(tensor_blobs)-1, False, None))

            # lconv (7 slots: 512-735) — float32 for accuracy
            shape, sz = add_float(lp + "lconv1d.linear_start.linear.weight")
            tensor_offsets.append((lo + 512, len(tensor_blobs)-1, False, None))
            buf[lo+512+48:lo+512+64] = struct.pack("<4f", *_get_clip(weights, lp+"lconv1d.linear_start."))
            shape, sz = add_float(lp + "lconv1d.linear_end.linear.weight")
            tensor_offsets.append((lo + 576, len(tensor_blobs)-1, False, None))
            buf[lo+576+48:lo+576+64] = struct.pack("<4f", *_get_clip(weights, lp+"lconv1d.linear_end."))
            shape, sz = add_float(lp + "lconv1d.depthwise_conv1d.weight", (1024,1,5))
            tensor_offsets.append((lo + 640, len(tensor_blobs)-1, False, None))
            shape, sz = add_float(lp + "lconv1d.pre_layer_norm.weight", (1024,))
            tensor_offsets.append((lo + 672, len(tensor_blobs)-1, False, None))
            shape, sz = add_float(lp + "lconv1d.conv_norm.weight", (1024,))
            tensor_offsets.append((lo + 704, len(tensor_blobs)-1, False, None))

            # ffn2 (6 slots: 736-927) — float32 for accuracy
            shape, sz = add_float(lp + "feed_forward2.ffw_layer_1.linear.weight")
            tensor_offsets.append((lo + 736, len(tensor_blobs)-1, False, None))
            buf[lo+736+48:lo+736+64] = struct.pack("<4f", *_get_clip(weights, lp+"feed_forward2.ffw_layer_1."))
            shape, sz = add_float(lp + "feed_forward2.ffw_layer_2.linear.weight")
            tensor_offsets.append((lo + 800, len(tensor_blobs)-1, False, None))
            buf[lo+800+48:lo+800+64] = struct.pack("<4f", *_get_clip(weights, lp+"feed_forward2.ffw_layer_2."))
            shape, sz = add_float(lp + "feed_forward2.pre_layer_norm.weight", (1024,))
            tensor_offsets.append((lo + 864, len(tensor_blobs)-1, False, None))
            shape, sz = add_float(lp + "feed_forward2.post_layer_norm.weight", (1024,))
            tensor_offsets.append((lo + 896, len(tensor_blobs)-1, False, None))

            # layer norms (3 slots: 928,960,992)
            shape, sz = add_float(lp + "norm_pre_attn.weight", (1024,))
            tensor_offsets.append((lo + 928, len(tensor_blobs)-1, False, None))
            shape, sz = add_float(lp + "norm_post_attn.weight", (1024,))
            tensor_offsets.append((lo + 960, len(tensor_blobs)-1, False, None))
            shape, sz = add_float(lp + "norm_out.weight", (1024,))
            tensor_offsets.append((lo + 992, len(tensor_blobs)-1, False, None))

        # --- Output projection (float32) ---
        shape, sz = add_float(ap + "output_proj.weight")
        tensor_offsets.append((OFF_OUT_PROJ, len(tensor_blobs)-1, False, None))
        shape, sz = add_float(ap + "output_proj.bias", (1536,))
        tensor_offsets.append((OFF_OUT_BIAS, len(tensor_blobs)-1, False, None))

        # --- Embed projection ---
        shape, sz = add_float("model.embed_audio.embedding_projection.weight", (1536,1536))
        tensor_offsets.append((OFF_EMBED_PROJ, len(tensor_blobs)-1, False, None))

        # Token IDs
        buf[OFF_BOA:OFF_BOA+4] = struct.pack("<i", boa_token_id)
        buf[OFF_EOA:OFF_EOA+4] = struct.pack("<i", eoa_token_id)
        buf[OFF_AUDIO_TOK:OFF_AUDIO_TOK+4] = struct.pack("<i", audio_token_id)

        # Phase 3: calculate data offsets and patch Tensor fields
        data_cursor = STRUCT_SIZE
        for blob_shape, data_bytes, scales_bytes in tensor_blobs:
            data_cursor = align64(data_cursor)

        for struct_off, blob_idx, _, _ in tensor_offsets:
            shape, data_bytes, scales_bytes = tensor_blobs[blob_idx]
            data_off = align64(data_cursor)
            data_cursor = align64(data_off + len(data_bytes))
            scales_off = 0
            if scales_bytes:
                scales_off = align64(data_cursor)
                data_cursor = align64(scales_off + len(scales_bytes))
            buf[struct_off:struct_off+32] = tensor_entry(data_off, scales_off, shape)

        file_size = align64(data_cursor)

        # Phase 4: write the file
        with tempfile.TemporaryDirectory(prefix=f".{output_path.name}.", dir=output_path.parent) as directory:
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as output:
                temp_path = Path(output.name)
                output.truncate(file_size)

                # Struct
                output.seek(0)
                output.write(bytes(buf))

                # Tensor data
                data_cursor = STRUCT_SIZE
                for blob_shape, data_bytes, scales_bytes in tensor_blobs:
                    data_off = align64(data_cursor)
                    output.seek(data_off)
                    output.write(data_bytes)
                    data_cursor = align64(data_off + len(data_bytes))
                    if scales_bytes:
                        scales_off = align64(data_cursor)
                        output.seek(scales_off)
                        output.write(scales_bytes)
                        data_cursor = align64(scales_off + len(scales_bytes))

                output.flush()
                os.fsync(output.fileno())
            os.replace(temp_path, output_path)

    print(f"Done {output_path} ({file_size / 1024**2:.1f} MiB): {len(tensor_blobs)} tensors")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=None)
    parser.add_argument("--quant", choices=("int8", "int4"), default="int8",
                        help="weight quantization mode (default: int8)")
    parser.add_argument("--audio-out", type=Path, default=None,
                        help="export audio encoder weights to a separate file")
    args = parser.parse_args()
    if not args.out and not args.audio_out:
        parser.error("at least one of -o/--out or --audio-out is required")
    if args.out:
        export(args.checkpoint, args.out, 4 if args.quant == "int4" else 8)
    if args.audio_out:
        export_audio(args.checkpoint, args.audio_out, 4 if args.quant == "int4" else 8)


if __name__ == "__main__":
    main()
