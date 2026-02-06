"""
Convert a GGUF model file to GdsLLM per-layer binary format.

Keeps block-quantized data as-is (Q4_0, Q8_0) for GPU dequantization.
Norm tensors (f32 in GGUF) are converted to f16.

Usage:
    python -m gdsllm.tools.convert_gguf \
        --gguf /path/to/model.gguf \
        --output-dir /path/to/gdsllm_model
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType, GGML_QUANT_SIZES, dequantize


ALIGNMENT = 4096

# GGUF tensor name → GdsLLM per-layer suffix
GGUF_LAYER_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}

# GdsLLM layer tensor order (must match LAYER_TENSOR_ORDER in convert_weights.py)
LAYER_TENSOR_ORDER = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]

# GGUF special tensor names → GdsLLM names
GGUF_SPECIAL_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}

# Supported GGUF quantization types for native block loading
SUPPORTED_BLOCK_TYPES = {
    GGMLQuantizationType.Q4_0: "q4_0",
    GGMLQuantizationType.Q8_0: "q8_0",
}

# Types that we convert to fp16 (norms, small tensors)
CONVERT_TO_FP16_TYPES = {
    GGMLQuantizationType.F32,
    GGMLQuantizationType.F16,
}


def align_up(value: int, alignment: int) -> int:
    return math.ceil(value / alignment) * alignment


def write_aligned(f, data: bytes) -> tuple:
    """Write data with 4KB alignment padding. Returns (offset, padded_size)."""
    offset = f.tell()
    f.write(data)
    raw_size = len(data)
    padded_size = align_up(raw_size, ALIGNMENT)
    padding = padded_size - raw_size
    if padding > 0:
        f.write(b"\x00" * padding)
    return offset, padded_size


def get_gguf_metadata(reader: GGUFReader) -> dict:
    """Extract model config from GGUF metadata fields."""
    from gguf import GGUFValueType

    def get_field(key, default=None):
        try:
            field = reader.get_field(key)
            if field is None:
                return default
            # String fields: decode raw uint8 bytes
            if field.types and field.types[0] == GGUFValueType.STRING:
                vals = field.parts[field.data[0]]
                return bytes(vals).decode("utf-8")
            # Array fields: return count of data entries (e.g. token list)
            if field.types and field.types[0] == GGUFValueType.ARRAY:
                return len(field.data)
            # Scalar fields
            vals = field.parts[field.data[0]]
            if len(vals) == 1:
                return vals[0].item() if hasattr(vals[0], 'item') else vals[0]
            return vals.tolist()
        except (KeyError, IndexError):
            return default

    arch = get_field("general.architecture", "llama")

    # vocab_size: prefer explicit field, fall back to token list count
    vocab_size = get_field(f"{arch}.vocab_size")
    if vocab_size is None:
        vocab_size = get_field("tokenizer.ggml.tokens", 32000)

    config = {
        "vocab_size": vocab_size,
        "hidden_size": get_field(f"{arch}.embedding_length"),
        "intermediate_size": get_field(f"{arch}.feed_forward_length"),
        "num_layers": get_field(f"{arch}.block_count"),
        "num_heads": get_field(f"{arch}.attention.head_count"),
        "num_kv_heads": get_field(f"{arch}.attention.head_count_kv"),
        "rms_norm_eps": get_field(f"{arch}.attention.layer_norm_rms_epsilon", 1e-5),
        "rope_theta": get_field(f"{arch}.rope.freq_base", 10000.0),
        "max_seq_len": get_field(f"{arch}.context_length", 4096),
    }

    # Compute head_dim
    if config["num_heads"] and config["hidden_size"]:
        config["head_dim"] = config["hidden_size"] // config["num_heads"]

    # RoPE scaling
    rope_scale_type = get_field(f"{arch}.rope.scaling.type")
    if rope_scale_type:
        config["rope_scaling"] = {
            "rope_type": rope_scale_type,
            "factor": get_field(f"{arch}.rope.scaling.factor", 1.0),
            "original_max_position_embeddings": get_field(
                f"{arch}.rope.scaling.original_context_length", 8192
            ),
            "low_freq_factor": get_field(f"{arch}.rope.scaling.low_freq_factor", 1.0),
            "high_freq_factor": get_field(f"{arch}.rope.scaling.high_freq_factor", 4.0),
        }

    return config


def tensor_to_bytes(tensor_data, tensor_type) -> tuple:
    """Convert GGUF tensor data to bytes for writing.

    For block-quantized types (Q4_0, Q8_0): returns raw block bytes as-is.
    For F32/F16: converts to fp16 bytes.
    For Q6_K: dequantizes on CPU to fp16 (used for lm_head/output.weight).

    Returns: (raw_bytes, dtype_str, quant_str)
    """
    if tensor_type in SUPPORTED_BLOCK_TYPES:
        # Keep block data as-is
        raw_bytes = bytes(tensor_data)
        quant_str = SUPPORTED_BLOCK_TYPES[tensor_type]
        dtype_str = f"gguf_{quant_str}"
        return raw_bytes, dtype_str, quant_str
    elif tensor_type in CONVERT_TO_FP16_TYPES:
        # Convert to fp16
        if tensor_type == GGMLQuantizationType.F32:
            arr = np.frombuffer(tensor_data, dtype=np.float32)
            arr = arr.astype(np.float16)
        else:  # F16
            arr = np.frombuffer(tensor_data, dtype=np.float16)
        raw_bytes = arr.tobytes()
        return raw_bytes, "float16", None
    else:
        # Unsupported block type (Q6_K, Q5_K, etc.): dequant to fp16 via gguf library
        print(f"    (dequanting {tensor_type.name} to fp16 on CPU)")
        arr = dequantize(tensor_data, tensor_type)
        arr = arr.astype(np.float16)
        raw_bytes = arr.tobytes()
        return raw_bytes, "float16", None


def extract_tokenizer(reader: GGUFReader, output_dir: str):
    """Extract tokenizer from GGUF metadata and save as HF-compatible files.

    Builds tokenizer.json (HF fast tokenizer format) and tokenizer_config.json
    from the GGUF tokenizer.ggml.* fields.
    """
    from gguf import GGUFValueType

    def get_string_field(key):
        field = reader.get_field(key)
        if field is None:
            return None
        if field.types and field.types[0] == GGUFValueType.STRING:
            return bytes(field.parts[field.data[0]]).decode("utf-8")
        return None

    def get_int_field(key):
        field = reader.get_field(key)
        if field is None:
            return None
        vals = field.parts[field.data[0]]
        return vals[0].item() if len(vals) == 1 else None

    def get_string_array(key):
        field = reader.get_field(key)
        if field is None:
            return []
        return [bytes(field.parts[idx]).decode("utf-8") for idx in field.data]

    def get_int_array(key):
        field = reader.get_field(key)
        if field is None:
            return []
        return [field.parts[idx][0].item() for idx in field.data]

    model_type = get_string_field("tokenizer.ggml.model")  # "llama", "gpt2", etc.
    tokens = get_string_array("tokenizer.ggml.tokens")
    token_types = get_int_array("tokenizer.ggml.token_type")
    merges = get_string_array("tokenizer.ggml.merges")
    bos_id = get_int_field("tokenizer.ggml.bos_token_id")
    eos_id = get_int_field("tokenizer.ggml.eos_token_id")
    pad_id = get_int_field("tokenizer.ggml.padding_token_id")
    chat_template = get_string_field("tokenizer.chat_template")
    pre_type = get_string_field("tokenizer.ggml.pre")

    if not tokens:
        print("Warning: no tokenizer data found in GGUF, skipping tokenizer extraction")
        return

    # Build vocab: token string -> id
    vocab = {tok: i for i, tok in enumerate(tokens)}

    # Identify special/added tokens (type 3 = control, type 4 = user-defined)
    added_tokens = []
    for i, tok in enumerate(tokens):
        tt = token_types[i] if i < len(token_types) else 1
        if tt == 3 or tt == 4:  # control or user-defined
            added_tokens.append({
                "id": i,
                "content": tok,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            })

    # Build tokenizer.json (HF fast tokenizer format)
    is_bpe = model_type in ("gpt2",) or merges
    is_spm = model_type in ("llama",) and not merges

    # Pre-tokenizer regex patterns for known tokenizer types
    PRE_TOKENIZER_PATTERNS = {
        "llama-bpe": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    }

    if is_bpe:
        # Build pre_tokenizer based on tokenizer.ggml.pre
        regex_pattern = PRE_TOKENIZER_PATTERNS.get(pre_type)
        if regex_pattern:
            pre_tokenizer = {
                "type": "Sequence",
                "pretokenizers": [
                    {
                        "type": "Split",
                        "pattern": {"Regex": regex_pattern},
                        "behavior": "Isolated",
                        "invert": False,
                    },
                    {
                        "type": "ByteLevel",
                        "add_prefix_space": False,
                        "trim_offsets": True,
                        "use_regex": False,
                    },
                ],
            }
        else:
            pre_tokenizer = {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": True,
                "use_regex": True,
            }

        tokenizer_json = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": added_tokens,
            "normalizer": None,
            "pre_tokenizer": pre_tokenizer,
            "post_processor": None,
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": True,
                "trim_offsets": True,
                "use_regex": True,
            },
            "model": {
                "type": "BPE",
                "dropout": None,
                "unk_token": None,
                "continuing_subword_prefix": None,
                "end_of_word_suffix": None,
                "fuse_unk": False,
                "byte_fallback": True,
                "vocab": vocab,
                "merges": merges,
            },
        }
    else:
        # SentencePiece / Unigram fallback — write raw token list
        # HF can still load this with the right tokenizer_config
        scores = []
        scores_field = reader.get_field("tokenizer.ggml.scores")
        if scores_field:
            scores = [scores_field.parts[idx][0].item() for idx in scores_field.data]

        tokenizer_json = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": added_tokens,
            "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "\u2581"},
            "pre_tokenizer": None,
            "post_processor": None,
            "decoder": {"type": "Replace", "pattern": {"String": "\u2581"}, "content": " "},
            "model": {
                "type": "Unigram",
                "unk_id": 0,
                "vocab": [[tok, scores[i] if i < len(scores) else 0.0]
                          for i, tok in enumerate(tokens)],
            },
        }

    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    with open(tokenizer_path, "w") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False)
    print(f"Wrote tokenizer to {tokenizer_path}")

    # Build tokenizer_config.json
    bos_token = tokens[bos_id] if bos_id is not None and bos_id < len(tokens) else None
    eos_token = tokens[eos_id] if eos_id is not None and eos_id < len(tokens) else None
    pad_token = tokens[pad_id] if pad_id is not None and pad_id < len(tokens) else None

    tokenizer_config = {
        "model_type": "llama",
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    if bos_token:
        tokenizer_config["bos_token"] = bos_token
    if eos_token:
        tokenizer_config["eos_token"] = eos_token
    if pad_token:
        tokenizer_config["pad_token"] = pad_token
    if chat_template:
        tokenizer_config["chat_template"] = chat_template

    config_path = os.path.join(output_dir, "tokenizer_config.json")
    with open(config_path, "w") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
    print(f"Wrote tokenizer config to {config_path}")


def build_tensor_map(reader: GGUFReader) -> dict:
    """Build a map from GGUF tensor name → ReaderTensor."""
    return {t.name: t for t in reader.tensors}


def convert_layer(
    layer_idx: int,
    tensor_map: dict,
    output_dir: str,
) -> dict:
    """Convert one transformer layer to a GdsLLM .bin file."""
    filename = f"layer_{layer_idx:03d}.bin"
    filepath = os.path.join(output_dir, filename)
    tensor_metas = []

    with open(filepath, "wb") as f:
        for gdsllm_suffix in LAYER_TENSOR_ORDER:
            # Find the GGUF tensor name for this layer
            gguf_suffix = None
            for gguf_key, gds_key in GGUF_LAYER_MAP.items():
                if gds_key == gdsllm_suffix:
                    gguf_suffix = gguf_key
                    break

            gguf_name = f"blk.{layer_idx}.{gguf_suffix}"
            tensor = tensor_map[gguf_name]

            raw_bytes, dtype_str, quant_str = tensor_to_bytes(
                tensor.data, tensor.tensor_type
            )
            offset, padded_size = write_aligned(f, raw_bytes)

            # GGUF shapes are reversed vs PyTorch convention
            # GGUF: [cols, rows] for 2D, PyTorch: [rows, cols]
            shape = list(reversed(tensor.shape.tolist()))

            meta = {
                "name": gdsllm_suffix,
                "shape": shape,
                "dtype": dtype_str,
                "offset": offset,
                "size_bytes": len(raw_bytes),
                "padded_size": padded_size,
                "quant": quant_str,
            }
            if quant_str in ("q4_0", "q8_0"):
                block_size, _ = GGML_QUANT_SIZES[tensor.tensor_type]
                meta["block_size"] = block_size

            tensor_metas.append(meta)

        total_size = f.tell()

    return {
        "index": layer_idx,
        "path": filename,
        "size_bytes": total_size,
        "tensors": tensor_metas,
    }


def convert_special(
    gguf_name: str,
    gdsllm_name: str,
    tensor_map: dict,
    output_filename: str,
    output_dir: str,
) -> dict:
    """Convert a special tensor (embed, norm, lm_head) to .bin file."""
    if gguf_name not in tensor_map:
        return None

    tensor = tensor_map[gguf_name]
    raw_bytes, dtype_str, quant_str = tensor_to_bytes(
        tensor.data, tensor.tensor_type
    )

    filepath = os.path.join(output_dir, output_filename)
    with open(filepath, "wb") as f:
        offset, padded_size = write_aligned(f, raw_bytes)
        total_size = f.tell()

    shape = list(reversed(tensor.shape.tolist()))

    meta = {
        "name": gdsllm_name,
        "shape": shape,
        "dtype": dtype_str,
        "offset": offset,
        "size_bytes": len(raw_bytes),
        "padded_size": padded_size,
        "quant": quant_str,
    }
    if quant_str in ("q4_0", "q8_0"):
        block_size, _ = GGML_QUANT_SIZES[tensor.tensor_type]
        meta["block_size"] = block_size

    return {
        "path": output_filename,
        "size_bytes": total_size,
        "tensors": [meta],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert GGUF model to GdsLLM format"
    )
    parser.add_argument(
        "--gguf",
        required=True,
        help="Path to GGUF model file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for converted .bin files and metadata.json",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.gguf):
        print(f"Error: GGUF file not found: {args.gguf}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Parse GGUF
    print(f"Reading GGUF: {args.gguf}")
    reader = GGUFReader(args.gguf)
    tensor_map = build_tensor_map(reader)
    print(f"  Found {len(tensor_map)} tensors")

    # Detect quantization type from first layer weight
    first_weight = tensor_map.get("blk.0.attn_q.weight")
    if first_weight is None:
        print("Error: cannot find blk.0.attn_q.weight in GGUF")
        sys.exit(1)

    gguf_quant_type = first_weight.tensor_type
    if gguf_quant_type not in SUPPORTED_BLOCK_TYPES:
        print(f"Error: unsupported GGUF quantization type: {gguf_quant_type.name}")
        print(f"Supported: {[t.name for t in SUPPORTED_BLOCK_TYPES]}")
        sys.exit(1)

    quant_name = SUPPORTED_BLOCK_TYPES[gguf_quant_type]
    print(f"  Quantization: {gguf_quant_type.name} ({quant_name})")

    # Step 2: Extract config
    config = get_gguf_metadata(reader)
    print(f"  Model: {config['num_layers']} layers, "
          f"hidden_size={config['hidden_size']}, "
          f"vocab_size={config['vocab_size']}, "
          f"num_heads={config['num_heads']}, "
          f"num_kv_heads={config['num_kv_heads']}")

    # Step 3: Convert specials
    print("Converting embed_tokens...")
    embed_meta = convert_special(
        "token_embd.weight", "model.embed_tokens.weight",
        tensor_map, "embed_tokens.bin", args.output_dir,
    )

    print("Converting final_norm...")
    norm_meta = convert_special(
        "output_norm.weight", "model.norm.weight",
        tensor_map, "final_norm.bin", args.output_dir,
    )

    lm_head_meta = None
    lm_head_tied = True
    if "output.weight" in tensor_map:
        print("Converting lm_head...")
        lm_head_meta = convert_special(
            "output.weight", "lm_head.weight",
            tensor_map, "lm_head.bin", args.output_dir,
        )
        lm_head_tied = False
    else:
        print("lm_head tied to embed_tokens (weight sharing)")

    # Step 4: Convert layers
    layer_metas = []
    for i in range(config["num_layers"]):
        print(f"Converting layer {i}/{config['num_layers'] - 1}...")
        layer_meta = convert_layer(i, tensor_map, args.output_dir)
        layer_metas.append(layer_meta)

    # Step 5: Write metadata
    metadata = {
        "model": "llama",
        "dtype": quant_name,
        "quantization": quant_name,
        "alignment": ALIGNMENT,
        **config,
        "lm_head_tied": lm_head_tied,
        "files": {
            "embed_tokens": embed_meta,
            "final_norm": norm_meta,
            "lm_head": lm_head_meta,
            "layers": layer_metas,
        },
    }

    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote metadata to {metadata_path}")

    # Step 6: Extract tokenizer
    print("Extracting tokenizer...")
    extract_tokenizer(reader, args.output_dir)

    # Summary
    total_size = sum(lm["size_bytes"] for lm in layer_metas)
    total_size += embed_meta["size_bytes"] + norm_meta["size_bytes"]
    if lm_head_meta:
        total_size += lm_head_meta["size_bytes"]
    print(f"\nConversion complete:")
    print(f"  {len(layer_metas)} layer files + specials")
    print(f"  Total size: {total_size / (1024**3):.2f} GB")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
