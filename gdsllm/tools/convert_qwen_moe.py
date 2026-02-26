"""
Convert Qwen3.5-MoE safetensors weights to GdsLLM flat binary format.

Handles:
  - Hybrid attention: full attention (every 4th layer) + linear attention (Mamba SSM)
  - MoE MLP: 512 experts + shared expert on all layers
  - 4-bit affine quantization: packed uint32 weights + bf16 scales + bf16 biases
  - 3D expert weight tensors [num_experts, intermediate, hidden]
  - Tensor prefix stripping: language_model.model. -> model.

Environment variables:
    GDSLLM_MODEL_ROOT  -- Default output directory

Usage:
    python -m gdsllm.tools.convert_qwen_moe \\
        --model-dir /path/to/Qwen3.5-397B-A17B-SWAN-4bit \\
        --output-dir /path/to/gdsllm/model
"""

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open


# Alignment requirement for O_DIRECT / GDS (4KB)
ALIGNMENT = 4096

# Prefix to strip from safetensors tensor names
HF_PREFIX = "language_model."

# Tensors for full attention layers (ordered)
FULL_ATTN_TENSORS = [
    "input_layernorm.weight",
    # Q projection (affine Q4: weight + scales + biases)
    "self_attn.q_proj.weight",
    "self_attn.q_proj.scales",
    "self_attn.q_proj.biases",
    # K projection
    "self_attn.k_proj.weight",
    "self_attn.k_proj.scales",
    "self_attn.k_proj.biases",
    # V projection
    "self_attn.v_proj.weight",
    "self_attn.v_proj.scales",
    "self_attn.v_proj.biases",
    # O projection
    "self_attn.o_proj.weight",
    "self_attn.o_proj.scales",
    "self_attn.o_proj.biases",
    # Per-head norms
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
]

# Tensors for linear attention (Mamba SSM) layers (ordered)
LINEAR_ATTN_TENSORS = [
    "input_layernorm.weight",
    # in_proj_a (affine Q4)
    "linear_attn.in_proj_a.weight",
    "linear_attn.in_proj_a.scales",
    "linear_attn.in_proj_a.biases",
    # in_proj_b
    "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_b.scales",
    "linear_attn.in_proj_b.biases",
    # in_proj_qkv
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_qkv.scales",
    "linear_attn.in_proj_qkv.biases",
    # in_proj_z
    "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_z.scales",
    "linear_attn.in_proj_z.biases",
    # Non-quantized SSM params
    "linear_attn.conv1d.weight",
    "linear_attn.A_log",
    "linear_attn.dt_bias",
    "linear_attn.norm.weight",
    # out_proj (affine Q4)
    "linear_attn.out_proj.weight",
    "linear_attn.out_proj.scales",
    "linear_attn.out_proj.biases",
]

# MoE MLP tensors (shared by all layers, appended after attn)
MOE_MLP_TENSORS = [
    "post_attention_layernorm.weight",
    # Router (bf16, not quantized)
    "mlp.gate.weight",
    # Switch MLP experts (3D affine Q4: [512, intermediate, hidden])
    "mlp.switch_mlp.gate_proj.weight",
    "mlp.switch_mlp.gate_proj.scales",
    "mlp.switch_mlp.gate_proj.biases",
    "mlp.switch_mlp.up_proj.weight",
    "mlp.switch_mlp.up_proj.scales",
    "mlp.switch_mlp.up_proj.biases",
    "mlp.switch_mlp.down_proj.weight",
    "mlp.switch_mlp.down_proj.scales",
    "mlp.switch_mlp.down_proj.biases",
    # Shared expert (2D affine Q4)
    "mlp.shared_expert.gate_proj.weight",
    "mlp.shared_expert.gate_proj.scales",
    "mlp.shared_expert.gate_proj.biases",
    "mlp.shared_expert.up_proj.weight",
    "mlp.shared_expert.up_proj.scales",
    "mlp.shared_expert.up_proj.biases",
    "mlp.shared_expert.down_proj.weight",
    "mlp.shared_expert.down_proj.scales",
    "mlp.shared_expert.down_proj.biases",
    # Shared expert gate (bf16, not quantized)
    "mlp.shared_expert_gate.weight",
]


def align_up(value: int, alignment: int) -> int:
    return math.ceil(value / alignment) * alignment


def get_safetensors_files(model_dir: str) -> list[str]:
    files = sorted(Path(model_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")
    return [str(f) for f in files]


def build_tensor_index(safetensors_files: list[str]) -> dict[str, str]:
    """Build mapping from tensor name -> safetensors file path."""
    index = {}
    for filepath in safetensors_files:
        with safe_open(filepath, framework="pt") as f:
            for key in f.keys():
                index[key] = filepath
    return index


def strip_prefix(name: str) -> str:
    """Strip 'language_model.' prefix if present."""
    if name.startswith(HF_PREFIX):
        return name[len(HF_PREFIX):]
    return name


def read_tensor_raw(tensor_index: dict[str, str], name: str) -> torch.Tensor:
    """Read a tensor from safetensors, keeping original dtype."""
    if name not in tensor_index:
        raise KeyError(f"Tensor '{name}' not found in safetensors files")
    with safe_open(tensor_index[name], framework="pt") as f:
        return f.get_tensor(name)


def write_aligned(f, data: bytes) -> tuple[int, int]:
    """Write data with 4KB alignment padding. Returns (offset, padded_size)."""
    offset = f.tell()
    f.write(data)
    raw_size = len(data)
    padded_size = align_up(raw_size, ALIGNMENT)
    padding = padded_size - raw_size
    if padding > 0:
        f.write(b"\x00" * padding)
    return offset, padded_size


def is_affine_q4(tensor: torch.Tensor) -> bool:
    """Check if tensor is packed affine Q4 (uint32)."""
    return tensor.dtype == torch.uint32


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Convert tensor to contiguous bytes, converting bf16->fp16."""
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float16)
    return tensor.contiguous().numpy().tobytes()


def write_tensor_entry(f, tensor: torch.Tensor, name: str) -> dict:
    """Write a single tensor and return its metadata entry."""
    raw_bytes = tensor_to_bytes(tensor)
    offset, padded_size = write_aligned(f, raw_bytes)

    dtype_str = {
        torch.float16: "float16",
        torch.bfloat16: "float16",  # converted above
        torch.uint32: "uint32",
        torch.int8: "int8",
    }.get(tensor.dtype, str(tensor.dtype))

    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": dtype_str,
        "offset": offset,
        "size_bytes": len(raw_bytes),
        "padded_size": padded_size,
        "quant": None,
    }


def write_affine_q4_group(f, weight: torch.Tensor, scales: torch.Tensor,
                           biases: torch.Tensor, name_prefix: str,
                           group_size: int) -> list[dict]:
    """Write an affine Q4 weight + scales + biases and return metadata entries.

    The weight is packed uint32, scales and biases are bf16/fp16.
    Returns 3 metadata entries (weight, scales, biases) with the weight entry
    containing quant="affine_q4" and references to scale/bias offsets.
    """
    entries = []

    # Write packed weight (uint32)
    w_bytes = weight.contiguous().numpy().tobytes()
    w_offset, w_padded = write_aligned(f, w_bytes)

    # Write scales (bf16 -> fp16)
    s_tensor = scales.to(torch.float16) if scales.dtype == torch.bfloat16 else scales
    s_bytes = s_tensor.contiguous().numpy().tobytes()
    s_offset, s_padded = write_aligned(f, s_bytes)

    # Write biases (bf16 -> fp16)
    b_tensor = biases.to(torch.float16) if biases.dtype == torch.bfloat16 else biases
    b_bytes = b_tensor.contiguous().numpy().tobytes()
    b_offset, b_padded = write_aligned(f, b_bytes)

    # Compute original (unpacked) weight shape
    # uint32 packs 8 x 4-bit values, so real_in_dim = packed_in_dim * 8
    orig_shape = list(weight.shape)
    orig_shape[-1] = orig_shape[-1] * 8

    # Weight entry with quant metadata
    entries.append({
        "name": f"{name_prefix}.weight",
        "shape": orig_shape,
        "dtype": "uint32",
        "offset": w_offset,
        "size_bytes": len(w_bytes),
        "padded_size": w_padded,
        "quant": "affine_q4",
        "group_size": group_size,
        "scale_shape": list(s_tensor.shape),
        "scale_offset": s_offset,
        "scale_size_bytes": len(s_bytes),
        "scale_padded_size": s_padded,
        "bias_shape": list(b_tensor.shape),
        "bias_offset": b_offset,
        "bias_size_bytes": len(b_bytes),
        "bias_padded_size": b_padded,
    })

    # Also record scales and biases as separate entries for completeness
    entries.append({
        "name": f"{name_prefix}.scales",
        "shape": list(s_tensor.shape),
        "dtype": "float16",
        "offset": s_offset,
        "size_bytes": len(s_bytes),
        "padded_size": s_padded,
        "quant": None,
    })

    entries.append({
        "name": f"{name_prefix}.biases",
        "shape": list(b_tensor.shape),
        "dtype": "float16",
        "offset": b_offset,
        "size_bytes": len(b_bytes),
        "padded_size": b_padded,
        "quant": None,
    })

    return entries


def convert_layer(layer_idx: int, layer_type: str, tensor_index: dict[str, str],
                  output_dir: str, group_size: int) -> dict:
    """Convert one Qwen3.5-MoE layer to a flat .bin file."""
    filename = f"layer_{layer_idx:03d}.bin"
    filepath = os.path.join(output_dir, filename)

    # Select attention tensors based on layer type
    if layer_type == "full_attention":
        attn_tensors = FULL_ATTN_TENSORS
    else:
        attn_tensors = LINEAR_ATTN_TENSORS

    all_tensor_names = attn_tensors + MOE_MLP_TENSORS
    tensor_metas = []
    hf_prefix = f"{HF_PREFIX}model.layers.{layer_idx}."

    with open(filepath, "wb") as f:
        i = 0
        while i < len(all_tensor_names):
            suffix = all_tensor_names[i]
            hf_name = f"{hf_prefix}{suffix}"

            # Check if this is an affine Q4 group (weight + scales + biases)
            if (i + 2 < len(all_tensor_names)
                    and all_tensor_names[i + 1].endswith(".scales")
                    and all_tensor_names[i + 2].endswith(".biases")):
                # Read the weight and check if it's uint32 (packed Q4)
                weight = read_tensor_raw(tensor_index, hf_name)
                if is_affine_q4(weight):
                    scales_name = f"{hf_prefix}{all_tensor_names[i + 1]}"
                    biases_name = f"{hf_prefix}{all_tensor_names[i + 2]}"
                    scales = read_tensor_raw(tensor_index, scales_name)
                    biases = read_tensor_raw(tensor_index, biases_name)

                    # Strip suffix to get the projection name
                    proj_name = suffix.rsplit(".weight", 1)[0]
                    entries = write_affine_q4_group(
                        f, weight, scales, biases, proj_name, group_size)
                    tensor_metas.extend(entries)
                    i += 3
                    continue
                else:
                    # Not actually quantized, write as normal tensor
                    entry = write_tensor_entry(f, weight, suffix)
                    tensor_metas.append(entry)
                    i += 1
                    continue

            # Normal tensor (norms, routers, SSM params)
            tensor = read_tensor_raw(tensor_index, hf_name)
            entry = write_tensor_entry(f, tensor, suffix)
            tensor_metas.append(entry)
            i += 1

        total_size = f.tell()

    return {
        "index": layer_idx,
        "path": filename,
        "size_bytes": total_size,
        "layer_type": layer_type,
        "tensors": tensor_metas,
    }


def convert_special(tensor_names: list[tuple[str, str]], tensor_index: dict[str, str],
                    output_filename: str, output_dir: str) -> dict:
    """Convert special tensors (embed, norm, lm_head) to a flat .bin file.

    tensor_names: list of (hf_name, gdsllm_name) pairs.
    """
    filepath = os.path.join(output_dir, output_filename)
    tensor_metas = []

    with open(filepath, "wb") as f:
        for hf_name, gds_name in tensor_names:
            tensor = read_tensor_raw(tensor_index, hf_name)
            entry = write_tensor_entry(f, tensor, gds_name)
            tensor_metas.append(entry)
        total_size = f.tell()

    return {
        "path": output_filename,
        "size_bytes": total_size,
        "tensors": tensor_metas,
    }


def detect_layer_types(hf_config: dict) -> list[str]:
    """Determine layer types from HF config."""
    text_config = hf_config.get("text_config", hf_config)
    layer_types = text_config.get("layer_types", [])
    if not layer_types:
        # Fallback: every 4th layer is full attention
        num_layers = text_config.get("num_hidden_layers", 60)
        full_interval = text_config.get("full_attention_interval", 4)
        layer_types = []
        for i in range(num_layers):
            if (i + 1) % full_interval == 0:
                layer_types.append("full_attention")
            else:
                layer_types.append("linear_attention")
    return layer_types


def read_hf_config(model_dir: str) -> dict:
    """Read and parse config.json from HuggingFace model directory."""
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    with open(config_path) as f:
        return json.load(f)


def build_gdsllm_config(hf_config: dict, layer_types: list[str]) -> dict:
    """Build GdsLLM metadata config from HF config."""
    text = hf_config.get("text_config", hf_config)
    quant = hf_config.get("quantization_config", hf_config.get("quantization", {}))

    full_attn_layers = [i for i, t in enumerate(layer_types) if t == "full_attention"]

    config = {
        "model": "qwen3.5-moe",
        "dtype": "affine_q4",
        "quantization": "affine_q4",
        "alignment": ALIGNMENT,
        "vocab_size": text.get("vocab_size", 248320),
        "hidden_size": text.get("hidden_size", 4096),
        "num_layers": text.get("num_hidden_layers", 60),
        "num_heads": text.get("num_attention_heads", 32),
        "num_kv_heads": text.get("num_key_value_heads", 2),
        "head_dim": text.get("head_dim", 256),
        "rms_norm_eps": text.get("rms_norm_eps", 1e-6),
        "rope_theta": text.get("rope_parameters", {}).get("rope_theta",
                      text.get("rope_theta", 10000000.0)),
        "max_seq_len": min(text.get("max_position_embeddings", 262144), 32768),
        "partial_rotary_factor": text.get("rope_parameters", {}).get("partial_rotary_factor",
                                 text.get("partial_rotary_factor", 0.25)),
        # MoE config
        "num_experts": text.get("num_experts", 512),
        "num_experts_per_token": text.get("num_experts_per_tok", 10),
        "moe_intermediate_size": text.get("moe_intermediate_size", 1024),
        "shared_expert_intermediate_size": text.get("shared_expert_intermediate_size", 1024),
        # Layer types
        "full_attn_layers": full_attn_layers,
        "layer_types": layer_types,
        "attn_output_gate": text.get("attn_output_gate", True),
        # SSM config
        "linear_key_head_dim": text.get("linear_key_head_dim", 128),
        "linear_value_head_dim": text.get("linear_value_head_dim", 128),
        "linear_num_key_heads": text.get("linear_num_key_heads", 16),
        "linear_num_value_heads": text.get("linear_num_value_heads", 64),
        "linear_conv_kernel_dim": text.get("linear_conv_kernel_dim", 4),
        # Quantization
        "group_size": quant.get("group_size", 128),
        "quant_bits": quant.get("bits", 4),
        # Tied embeddings
        "lm_head_tied": hf_config.get("tie_word_embeddings", False),
    }
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen3.5-MoE safetensors to GdsLLM format"
    )
    parser.add_argument(
        "--model-dir", required=True,
        help="Path to HuggingFace model directory containing .safetensors files",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("GDSLLM_MODEL_ROOT"),
        help="Output directory (default: GDSLLM_MODEL_ROOT env var)",
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip verification step after conversion",
    )
    args = parser.parse_args()

    if not args.output_dir:
        print("Error: No output directory specified.\n"
              "Pass --output-dir or set GDSLLM_MODEL_ROOT env var.")
        sys.exit(1)

    if not os.path.isdir(args.model_dir):
        print(f"Error: model directory not found: {args.model_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Read HF config
    print("Reading model configuration...")
    hf_config = read_hf_config(args.model_dir)
    layer_types = detect_layer_types(hf_config)
    config = build_gdsllm_config(hf_config, layer_types)
    group_size = config["group_size"]

    num_full = sum(1 for t in layer_types if t == "full_attention")
    num_linear = sum(1 for t in layer_types if t == "linear_attention")
    print(f"  Model: qwen3.5-moe")
    print(f"  Layers: {config['num_layers']} ({num_full} full attn + {num_linear} linear attn)")
    print(f"  Hidden: {config['hidden_size']}, Heads: {config['num_heads']}, KV heads: {config['num_kv_heads']}")
    print(f"  MoE: {config['num_experts']} experts, top-{config['num_experts_per_token']}")
    print(f"  Quantization: affine Q4 (group_size={group_size})")

    # Step 2: Index safetensors
    print("Indexing safetensors files...")
    safetensors_files = get_safetensors_files(args.model_dir)
    print(f"  Found {len(safetensors_files)} safetensors file(s)")
    tensor_index = build_tensor_index(safetensors_files)
    print(f"  Indexed {len(tensor_index)} tensors")

    # Step 3: Convert special files
    print("Converting embed_tokens...")
    embed_meta = convert_special(
        [("language_model.model.embed_tokens.weight", "model.embed_tokens.weight")],
        tensor_index, "embed_tokens.bin", args.output_dir,
    )

    print("Converting final_norm...")
    norm_meta = convert_special(
        [("language_model.model.norm.weight", "model.norm.weight")],
        tensor_index, "final_norm.bin", args.output_dir,
    )

    if not config["lm_head_tied"]:
        print("Converting lm_head...")
        lm_head_meta = convert_special(
            [("language_model.lm_head.weight", "lm_head.weight")],
            tensor_index, "lm_head.bin", args.output_dir,
        )
    else:
        print("lm_head tied to embed_tokens (weight sharing)")
        lm_head_meta = None

    # Step 4: Convert layers
    layer_metas = []
    for i in range(config["num_layers"]):
        lt = layer_types[i]
        lt_short = "full" if lt == "full_attention" else "linear"
        print(f"Converting layer {i}/{config['num_layers'] - 1} ({lt_short})...")
        layer_meta = convert_layer(i, lt, tensor_index, args.output_dir, group_size)
        layer_metas.append(layer_meta)

    # Step 5: Write metadata.json
    metadata = {
        **config,
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

    # Step 6: Copy tokenizer files
    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
        "special_tokens_map.json", "added_tokens.json", "vocab.json",
        "merges.txt", "chat_template.jinja",
    ]
    copied = 0
    for tf in tokenizer_files:
        src = os.path.join(args.model_dir, tf)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(args.output_dir, tf))
            copied += 1
    if copied:
        print(f"Copied {copied} tokenizer files to {args.output_dir}")

    # Summary
    total_size = sum(lm["size_bytes"] for lm in layer_metas)
    total_size += embed_meta["size_bytes"] + norm_meta["size_bytes"]
    if lm_head_meta:
        total_size += lm_head_meta["size_bytes"]
    print(f"\nConversion complete:")
    print(f"  {len(layer_metas)} layer files + special files")
    print(f"  Total size: {total_size / (1024**3):.2f} GB")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
