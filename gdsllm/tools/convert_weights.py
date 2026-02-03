"""
Convert HuggingFace LLaMA safetensors weights to GdsLLM flat binary format.

For each transformer layer, produces a single .bin file containing all weight
tensors concatenated in a fixed order, with 4KB alignment padding.

Also produces embed_tokens.bin, final_norm.bin, lm_head.bin, and metadata.json.

Usage:
    python -m gdsllm.tools.convert_weights \
        --model-dir /path/to/llama-2-7b-hf \
        --output-dir /path/to/gdsllm/model
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open


# Alignment requirement for O_DIRECT / GDS (4KB)
ALIGNMENT = 4096

# Fixed order of tensors within each transformer layer .bin file
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


def align_up(value: int, alignment: int) -> int:
    """Round up value to the next multiple of alignment."""
    return math.ceil(value / alignment) * alignment


def get_safetensors_files(model_dir: str) -> list[str]:
    """Find all safetensors files in the model directory."""
    model_path = Path(model_dir)
    files = sorted(model_path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(
            f"No .safetensors files found in {model_dir}"
        )
    return [str(f) for f in files]


def build_tensor_index(safetensors_files: list[str]) -> dict[str, str]:
    """Build a mapping from tensor name -> safetensors file path."""
    index = {}
    for filepath in safetensors_files:
        with safe_open(filepath, framework="numpy") as f:
            for key in f.keys():
                index[key] = filepath
    return index


def read_tensor(
    tensor_index: dict[str, str], tensor_name: str
) -> np.ndarray:
    """Read a single tensor from the safetensors files as float16 numpy array."""
    if tensor_name not in tensor_index:
        raise KeyError(f"Tensor '{tensor_name}' not found in safetensors files")
    filepath = tensor_index[tensor_name]
    with safe_open(filepath, framework="numpy") as f:
        tensor = f.get_tensor(tensor_name)
    # Ensure float16 and C-contiguous
    if tensor.dtype != np.float16:
        tensor = tensor.astype(np.float16)
    if not tensor.flags["C_CONTIGUOUS"]:
        tensor = np.ascontiguousarray(tensor)
    return tensor


def write_aligned(f, data: bytes) -> tuple[int, int]:
    """Write data to file with 4KB alignment padding. Returns (offset, padded_size)."""
    offset = f.tell()
    f.write(data)
    raw_size = len(data)
    padded_size = align_up(raw_size, ALIGNMENT)
    padding = padded_size - raw_size
    if padding > 0:
        f.write(b"\x00" * padding)
    return offset, padded_size


def convert_layer(
    layer_idx: int,
    tensor_index: dict[str, str],
    output_dir: str,
    model_prefix: str = "model.layers",
) -> dict:
    """Convert one transformer layer to a flat .bin file.

    Returns metadata dict for this layer.
    """
    filename = f"layer_{layer_idx:03d}.bin"
    filepath = os.path.join(output_dir, filename)
    tensor_metas = []

    with open(filepath, "wb") as f:
        for tensor_suffix in LAYER_TENSOR_ORDER:
            full_name = f"{model_prefix}.{layer_idx}.{tensor_suffix}"
            tensor = read_tensor(tensor_index, full_name)
            raw_bytes = tensor.tobytes()
            offset, padded_size = write_aligned(f, raw_bytes)

            tensor_metas.append(
                {
                    "name": tensor_suffix,
                    "shape": list(tensor.shape),
                    "dtype": "float16",
                    "offset": offset,
                    "size_bytes": len(raw_bytes),
                    "padded_size": padded_size,
                }
            )

        total_size = f.tell()

    return {
        "index": layer_idx,
        "path": filename,
        "size_bytes": total_size,
        "tensors": tensor_metas,
    }


def convert_special(
    tensor_names: list[str],
    tensor_index: dict[str, str],
    output_filename: str,
    output_dir: str,
) -> dict:
    """Convert special tensors (embeddings, norm, lm_head) to a flat .bin file.

    Returns metadata dict.
    """
    filepath = os.path.join(output_dir, output_filename)
    tensor_metas = []

    with open(filepath, "wb") as f:
        for full_name in tensor_names:
            tensor = read_tensor(tensor_index, full_name)
            raw_bytes = tensor.tobytes()
            offset, padded_size = write_aligned(f, raw_bytes)

            # Use short name (last part after the last dot-separated prefix)
            short_name = full_name.split(".")[-1]
            if short_name == "weight":
                # Use parent.weight for clarity
                parts = full_name.split(".")
                short_name = ".".join(parts[-2:])

            tensor_metas.append(
                {
                    "name": full_name,
                    "shape": list(tensor.shape),
                    "dtype": "float16",
                    "offset": offset,
                    "size_bytes": len(raw_bytes),
                    "padded_size": padded_size,
                }
            )

        total_size = f.tell()

    return {
        "path": output_filename,
        "size_bytes": total_size,
        "tensors": tensor_metas,
    }


def detect_model_config(tensor_index: dict[str, str]) -> dict:
    """Detect model configuration from tensor shapes."""
    # Read embed_tokens to get vocab_size and hidden_size
    embed = read_tensor(tensor_index, "model.embed_tokens.weight")
    vocab_size, hidden_size = embed.shape

    # Read gate_proj to get intermediate_size
    gate = read_tensor(tensor_index, "model.layers.0.mlp.gate_proj.weight")
    intermediate_size = gate.shape[0]

    # Read q_proj to get num_heads
    q_proj = read_tensor(tensor_index, "model.layers.0.self_attn.q_proj.weight")
    q_size = q_proj.shape[0]

    # Count layers
    num_layers = 0
    while f"model.layers.{num_layers}.input_layernorm.weight" in tensor_index:
        num_layers += 1

    # Detect num_heads and num_kv_heads from q_proj and k_proj shapes
    k_proj = read_tensor(tensor_index, "model.layers.0.self_attn.k_proj.weight")
    k_size = k_proj.shape[0]
    head_dim = hidden_size // (q_size // hidden_size) if q_size >= hidden_size else 128
    # For standard LLaMA: q_size == hidden_size, so num_heads = hidden_size / head_dim
    # For GQA models: k_size < q_size
    num_heads = q_size // (hidden_size // (q_size // hidden_size)) if q_size >= hidden_size else 32
    num_kv_heads = k_size // (hidden_size // num_heads)

    # Simpler calculation
    head_dim = hidden_size // (q_size // hidden_size)
    num_heads = q_size // head_dim
    num_kv_heads = k_size // head_dim

    config = {
        "vocab_size": int(vocab_size),
        "hidden_size": int(hidden_size),
        "intermediate_size": int(intermediate_size),
        "num_layers": num_layers,
        "num_heads": int(num_heads),
        "num_kv_heads": int(num_kv_heads),
        "head_dim": int(head_dim),
        "rms_norm_eps": 1e-5,  # LLaMA-2 default; LLaMA-1 uses 1e-6
        "max_seq_len": 4096,  # LLaMA-2 default
        "rope_theta": 10000.0,
    }
    return config


def try_read_hf_config(model_dir: str, config: dict) -> dict:
    """Try to read config.json from HuggingFace model dir to fill in exact values."""
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return config
    with open(config_path, "r") as f:
        hf_config = json.load(f)

    # Map HF config keys to our config keys
    hf_to_gds = {
        "num_attention_heads": "num_heads",
        "num_key_value_heads": "num_kv_heads",
        "hidden_size": "hidden_size",
        "intermediate_size": "intermediate_size",
        "num_hidden_layers": "num_layers",
        "vocab_size": "vocab_size",
        "rms_norm_eps": "rms_norm_eps",
        "rope_theta": "rope_theta",
    }
    for hf_key, gds_key in hf_to_gds.items():
        if hf_key in hf_config:
            config[gds_key] = hf_config[hf_key]

    if "max_position_embeddings" in hf_config:
        config["max_seq_len"] = hf_config["max_position_embeddings"]

    # Recompute head_dim from authoritative values
    if config["num_heads"] > 0:
        config["head_dim"] = config["hidden_size"] // config["num_heads"]

    return config


def verify_layer(
    layer_meta: dict,
    tensor_index: dict[str, str],
    output_dir: str,
    layer_idx: int,
    model_prefix: str = "model.layers",
) -> bool:
    """Verify a converted layer file against original safetensors."""
    filepath = os.path.join(output_dir, layer_meta["path"])
    with open(filepath, "rb") as f:
        for tmeta in layer_meta["tensors"]:
            full_name = f"{model_prefix}.{layer_idx}.{tmeta['name']}"
            original = read_tensor(tensor_index, full_name)
            f.seek(tmeta["offset"])
            converted = np.frombuffer(
                f.read(tmeta["size_bytes"]), dtype=np.float16
            ).reshape(tmeta["shape"])
            if not np.array_equal(original, converted):
                print(f"  MISMATCH: {full_name}")
                return False
    return True


def verify_special(
    file_meta: dict,
    tensor_index: dict[str, str],
    output_dir: str,
) -> bool:
    """Verify a special file against original safetensors."""
    filepath = os.path.join(output_dir, file_meta["path"])
    with open(filepath, "rb") as f:
        for tmeta in file_meta["tensors"]:
            original = read_tensor(tensor_index, tmeta["name"])
            f.seek(tmeta["offset"])
            converted = np.frombuffer(
                f.read(tmeta["size_bytes"]), dtype=np.float16
            ).reshape(tmeta["shape"])
            if not np.array_equal(original, converted):
                print(f"  MISMATCH: {tmeta['name']}")
                return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace LLaMA safetensors to GdsLLM format"
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to HuggingFace model directory containing .safetensors files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for converted .bin files and metadata.json",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip verification step after conversion",
    )
    args = parser.parse_args()

    # Validate input
    if not os.path.isdir(args.model_dir):
        print(f"Error: model directory not found: {args.model_dir}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Index all tensors
    print("Indexing safetensors files...")
    safetensors_files = get_safetensors_files(args.model_dir)
    print(f"  Found {len(safetensors_files)} safetensors file(s)")
    tensor_index = build_tensor_index(safetensors_files)
    print(f"  Indexed {len(tensor_index)} tensors")

    # Step 2: Detect model configuration
    print("Detecting model configuration...")
    config = detect_model_config(tensor_index)
    config = try_read_hf_config(args.model_dir, config)
    print(f"  Model: {config['num_layers']} layers, "
          f"hidden_size={config['hidden_size']}, "
          f"vocab_size={config['vocab_size']}, "
          f"num_heads={config['num_heads']}, "
          f"num_kv_heads={config['num_kv_heads']}")

    # Step 3: Convert special files
    print("Converting embed_tokens...")
    embed_meta = convert_special(
        ["model.embed_tokens.weight"],
        tensor_index, "embed_tokens.bin", args.output_dir,
    )

    print("Converting final_norm...")
    norm_meta = convert_special(
        ["model.norm.weight"],
        tensor_index, "final_norm.bin", args.output_dir,
    )

    # lm_head: some models tie lm_head to embed_tokens
    if "lm_head.weight" in tensor_index:
        print("Converting lm_head...")
        lm_head_meta = convert_special(
            ["lm_head.weight"],
            tensor_index, "lm_head.bin", args.output_dir,
        )
        lm_head_tied = False
    else:
        print("lm_head tied to embed_tokens (weight sharing)")
        lm_head_meta = None
        lm_head_tied = True

    # Step 4: Convert transformer layers
    layer_metas = []
    for i in range(config["num_layers"]):
        print(f"Converting layer {i}/{config['num_layers'] - 1}...")
        layer_meta = convert_layer(i, tensor_index, args.output_dir)
        layer_metas.append(layer_meta)

    # Step 5: Write metadata.json
    metadata = {
        "model": "llama",
        "dtype": "float16",
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

    # Step 6: Verify
    if not args.skip_verify:
        print("Verifying converted files...")
        ok = True

        if not verify_special(embed_meta, tensor_index, args.output_dir):
            ok = False
        if not verify_special(norm_meta, tensor_index, args.output_dir):
            ok = False
        if lm_head_meta and not verify_special(lm_head_meta, tensor_index, args.output_dir):
            ok = False

        for i, layer_meta in enumerate(layer_metas):
            if not verify_layer(layer_meta, tensor_index, args.output_dir, i):
                ok = False
            if (i + 1) % 8 == 0:
                print(f"  Verified {i + 1}/{config['num_layers']} layers")

        if ok:
            print("Verification PASSED: all tensors match.")
        else:
            print("Verification FAILED: some tensors do not match!")
            sys.exit(1)

    # Summary
    total_size = sum(lm["size_bytes"] for lm in layer_metas)
    total_size += embed_meta["size_bytes"] + norm_meta["size_bytes"]
    if lm_head_meta:
        total_size += lm_head_meta["size_bytes"]
    print(f"\nConversion complete:")
    print(f"  {len(layer_metas)} layer files + 3 special files")
    print(f"  Total size: {total_size / (1024**3):.2f} GB")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
