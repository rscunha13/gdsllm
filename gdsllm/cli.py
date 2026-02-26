"""
GdsLLM — Command Line Interface

Commands:
    gdsllm serve  — Start the API server
    gdsllm stop   — Stop the API server
    gdsllm run    — Interactive chat in terminal
    gdsllm pull   — Download and convert a HuggingFace model
    gdsllm rm     — Delete a local model
    gdsllm list   — List available local models
    gdsllm show   — Show model details

Environment variables:
    GDSLLM_MODEL_ROOT  — Root directory containing converted GdsLLM model subdirs
    GDSLLM_HF_CACHE    — Directory for HuggingFace model downloads
    HUGGINGFACE_HUB_TOKEN — HuggingFace authentication token
"""

import argparse
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path


PID_DIR = os.path.expanduser("~/.gdsllm")
PID_FILE = os.path.join(PID_DIR, "server.pid")


def load_dotenv():
    """Load .env file from the project root."""
    current = Path(__file__).resolve().parent
    for _ in range(5):
        env_file = current / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("\"'")
                    if key and key not in os.environ:
                        os.environ[key] = value
            return
        current = current.parent


def _model_root() -> str:
    """Get the model root directory from env var."""
    return os.environ.get("GDSLLM_MODEL_ROOT", "")


def _hf_cache() -> str:
    """Get the HuggingFace cache directory from env var."""
    return os.environ.get("GDSLLM_HF_CACHE", "")


def _resolve_model_dir(model_dir: str) -> str:
    """Resolve a model name or path to an absolute directory.

    If model_dir is already an existing directory, return it.
    Otherwise, try to find it as a subdirectory of GDSLLM_MODEL_ROOT.
    """
    if os.path.isdir(model_dir):
        return os.path.abspath(model_dir)

    # Try resolving against model root
    root = _model_root()
    if root:
        candidate = os.path.join(root, model_dir)
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)

    # Return as-is (will fail later with a clear error)
    return model_dir


def _check_nvme_warning(path: str):
    """Warn if path is not on an NVMe device."""
    try:
        path = os.path.realpath(path)
        best_mount = ""
        best_dev = ""
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                dev, mount = parts[0], parts[1]
                mount = mount.encode("raw_unicode_escape").decode("unicode_escape")
                if path.startswith(mount) and len(mount) > len(best_mount):
                    best_mount = mount
                    best_dev = dev
        if best_dev and "nvme" not in os.path.basename(best_dev):
            print(
                f"Warning: {path} is not on an NVMe device ({best_dev}).\n"
                f"  GPUDirect Storage requires NVMe for optimal performance.",
                file=sys.stderr,
            )
    except OSError:
        pass


def _dir_size(path: str) -> int:
    """Calculate total size of all files in a directory."""
    total = 0
    for root, _, files in os.walk(path):
        for fname in files:
            total += os.path.getsize(os.path.join(root, fname))
    return total


# ─── serve ───────────────────────────────────────────────────────────────────


def cmd_serve(args):
    """Start the GdsLLM API server."""
    import uvicorn
    from gdsllm.server.app import create_app

    model_dir = _resolve_model_dir(args.model_dir)
    if not os.path.isdir(model_dir):
        print(f"Error: Model directory not found: {model_dir}", file=sys.stderr)
        sys.exit(1)

    model_root = args.model_root or _model_root()

    app = create_app(
        model_dir=model_dir,
        preload=args.preload,
        model_root=model_root,
    )

    # Write PID file
    os.makedirs(PID_DIR, exist_ok=True)
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    auth_enabled = bool(os.environ.get("GDSLLM_API_TOKEN", "").strip())

    print(f"Starting GdsLLM server on {args.host}:{args.port}")
    print(f"Model: {model_dir}")
    print(f"Model root: {model_root or '(not set)'}")
    print(f"Preload: {args.preload}")
    print(f"Auth: {'enabled (Bearer token)' if auth_enabled else 'disabled (open access)'}")
    print()
    print(f"API docs: http://{args.host}:{args.port}/docs")
    print(f"Ollama API: http://{args.host}:{args.port}/api/")
    print(f"OpenAI API: http://{args.host}:{args.port}/v1/")

    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
        )
    finally:
        if os.path.isfile(PID_FILE):
            os.remove(PID_FILE)


# ─── stop ────────────────────────────────────────────────────────────────────


def cmd_stop(args):
    """Stop the GdsLLM API server."""
    if not os.path.isfile(PID_FILE):
        print("No running GdsLLM server found.", file=sys.stderr)
        sys.exit(1)

    with open(PID_FILE) as f:
        pid = int(f.read().strip())

    # Check if process is alive
    try:
        os.kill(pid, 0)
    except OSError:
        print(f"Server process {pid} is not running (stale PID file).", file=sys.stderr)
        os.remove(PID_FILE)
        sys.exit(1)

    print(f"Stopping GdsLLM server (PID {pid})...")
    os.kill(pid, signal.SIGTERM)

    # Wait for process to exit
    for _ in range(30):
        time.sleep(0.1)
        try:
            os.kill(pid, 0)
        except OSError:
            break

    # Verify it's dead
    try:
        os.kill(pid, 0)
        print(f"Warning: process {pid} still running after SIGTERM", file=sys.stderr)
    except OSError:
        print("Server stopped.")

    if os.path.isfile(PID_FILE):
        os.remove(PID_FILE)


# ─── run ─────────────────────────────────────────────────────────────────────


def cmd_run(args):
    """Interactive chat in the terminal."""
    from gdsllm.engine import InferenceEngine

    model_dir = _resolve_model_dir(args.model_dir)
    if not os.path.isdir(model_dir):
        print(f"Error: Model directory not found: {model_dir}", file=sys.stderr)
        root = _model_root()
        if not root:
            print("Hint: set GDSLLM_MODEL_ROOT to resolve model names automatically", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {model_dir}...")
    engine = InferenceEngine(model_dir, preload=args.preload)
    info = engine.model_info
    print(
        f"Model loaded: {info['name']} "
        f"({info['parameter_size']}, {info['quantization']})"
    )
    if args.prompt:
        # Non-interactive: single prompt, print response, exit
        messages = [{"role": "user", "content": args.prompt}]
        try:
            for event in engine.generate_chat(
                messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repeat_penalty=args.repeat_penalty,
            ):
                print(event.text, end="", flush=True)
            print()
        finally:
            engine.shutdown()
        return

    print("Type your message (Ctrl+D or 'exit' to quit)\n")

    try:
        while True:
            try:
                prompt = input(">>> ")
            except EOFError:
                break

            if prompt.strip().lower() in ("exit", "quit", "/bye"):
                break

            if not prompt.strip():
                continue

            messages = [{"role": "user", "content": prompt}]
            for event in engine.generate_chat(
                messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repeat_penalty=args.repeat_penalty,
            ):
                print(event.text, end="", flush=True)

            print()  # newline after generation
    except KeyboardInterrupt:
        print()
    finally:
        engine.shutdown()
        print("Goodbye!")


# ─── pull ────────────────────────────────────────────────────────────────────


def _detect_architecture(hf_path: str) -> str:
    """Detect model architecture from downloaded HuggingFace config.json.

    Returns "qwen3.5-moe" or "llama".
    """
    config_path = os.path.join(hf_path, "config.json")
    if not os.path.isfile(config_path):
        return "llama"

    with open(config_path) as f:
        config = json.load(f)

    # Check for Qwen3.5-MoE indicators
    text_config = config.get("text_config", config)

    # Has layer_types with linear_attention → Qwen3.5-MoE hybrid
    layer_types = text_config.get("layer_types", [])
    if any(t == "linear_attention" for t in layer_types):
        return "qwen3.5-moe"

    # Has num_experts + linear attention config keys
    if (text_config.get("num_experts") and
            text_config.get("linear_key_head_dim")):
        return "qwen3.5-moe"

    return "llama"


def _pull_convert_llama(hf_path: str, output_dir: str, quantize: str):
    """Convert a LLaMA-family model to GdsLLM format."""
    from gdsllm.tools.convert_weights import (
        ALIGNMENT,
        build_tensor_index,
        convert_layer,
        convert_special,
        detect_model_config,
        get_safetensors_files,
        try_read_hf_config,
    )

    print("Indexing safetensors files...")
    safetensors_files = get_safetensors_files(hf_path)
    print(f"  Found {len(safetensors_files)} safetensors file(s)")
    tensor_index = build_tensor_index(safetensors_files)
    print(f"  Indexed {len(tensor_index)} tensors")

    print("Detecting model configuration...")
    config = detect_model_config(tensor_index)
    config = try_read_hf_config(hf_path, config)
    print(f"  {config['num_layers']} layers, "
          f"hidden_size={config['hidden_size']}, "
          f"vocab_size={config['vocab_size']}")

    # Convert special files
    print("Converting embed_tokens...")
    embed_meta = convert_special(
        ["model.embed_tokens.weight"],
        tensor_index, "embed_tokens.bin", output_dir,
        quantize=quantize,
    )

    print("Converting final_norm...")
    norm_meta = convert_special(
        ["model.norm.weight"],
        tensor_index, "final_norm.bin", output_dir,
        quantize=quantize, force_fp16=True,
    )

    if "lm_head.weight" in tensor_index:
        print("Converting lm_head...")
        lm_head_meta = convert_special(
            ["lm_head.weight"],
            tensor_index, "lm_head.bin", output_dir,
            quantize=quantize,
        )
        lm_head_tied = False
    else:
        print("lm_head tied to embed_tokens (weight sharing)")
        lm_head_meta = None
        lm_head_tied = True

    layer_metas = []
    for i in range(config["num_layers"]):
        print(f"Converting layer {i}/{config['num_layers'] - 1}...")
        layer_meta = convert_layer(
            i, tensor_index, output_dir, quantize=quantize,
        )
        layer_metas.append(layer_meta)

    dtype_label = "int8" if quantize == "int8" else "float16"
    metadata = {
        "model": "llama",
        "dtype": dtype_label,
        "quantization": quantize,
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

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Copy tokenizer files
    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json",
        "tokenizer.model", "special_tokens_map.json",
    ]
    copied = 0
    for tf in tokenizer_files:
        src = os.path.join(hf_path, tf)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, tf))
            copied += 1
    if copied:
        print(f"Copied {copied} tokenizer files")


def _pull_convert_qwen_moe(hf_path: str, output_dir: str):
    """Convert a Qwen3.5-MoE model to GdsLLM format."""
    from gdsllm.tools.convert_qwen_moe import (
        build_gdsllm_config,
        build_tensor_index,
        convert_layer,
        convert_special,
        detect_layer_types,
        get_safetensors_files,
        read_hf_config,
    )

    print("Reading model configuration...")
    hf_config = read_hf_config(hf_path)
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

    print("Indexing safetensors files...")
    safetensors_files = get_safetensors_files(hf_path)
    print(f"  Found {len(safetensors_files)} safetensors file(s)")
    tensor_index = build_tensor_index(safetensors_files)
    print(f"  Indexed {len(tensor_index)} tensors")

    # Convert special files
    print("Converting embed_tokens...")
    embed_meta = convert_special(
        [("language_model.model.embed_tokens.weight", "model.embed_tokens.weight")],
        tensor_index, "embed_tokens.bin", output_dir,
    )

    print("Converting final_norm...")
    norm_meta = convert_special(
        [("language_model.model.norm.weight", "model.norm.weight")],
        tensor_index, "final_norm.bin", output_dir,
    )

    if not config["lm_head_tied"]:
        print("Converting lm_head...")
        lm_head_meta = convert_special(
            [("language_model.lm_head.weight", "lm_head.weight")],
            tensor_index, "lm_head.bin", output_dir,
        )
    else:
        print("lm_head tied to embed_tokens (weight sharing)")
        lm_head_meta = None

    # Convert layers
    layer_metas = []
    for i in range(config["num_layers"]):
        lt = layer_types[i]
        lt_short = "full" if lt == "full_attention" else "linear"
        print(f"Converting layer {i}/{config['num_layers'] - 1} ({lt_short})...")
        layer_meta = convert_layer(i, lt, tensor_index, output_dir, group_size)
        layer_metas.append(layer_meta)

    # Write metadata
    metadata = {
        **config,
        "files": {
            "embed_tokens": embed_meta,
            "final_norm": norm_meta,
            "lm_head": lm_head_meta,
            "layers": layer_metas,
        },
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote metadata to {metadata_path}")

    # Copy tokenizer files (including Qwen-specific files)
    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
        "special_tokens_map.json", "added_tokens.json", "vocab.json",
        "merges.txt", "chat_template.jinja",
    ]
    copied = 0
    for tf in tokenizer_files:
        src = os.path.join(hf_path, tf)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, tf))
            copied += 1
    if copied:
        print(f"Copied {copied} tokenizer files")


def cmd_pull(args):
    """Download a HuggingFace model and convert to GdsLLM format."""
    from huggingface_hub import snapshot_download

    model_root = _model_root()
    hf_cache = _hf_cache()
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

    if not model_root:
        print("Error: GDSLLM_MODEL_ROOT is not set.", file=sys.stderr)
        print("Set it in .env or as an environment variable.", file=sys.stderr)
        sys.exit(1)
    _check_nvme_warning(model_root)
    if not hf_cache:
        print("Error: GDSLLM_HF_CACHE is not set.", file=sys.stderr)
        print("Set it in .env or as an environment variable.", file=sys.stderr)
        sys.exit(1)
    if not token:
        print("Error: HUGGINGFACE_HUB_TOKEN is not set.", file=sys.stderr)
        print("Set it in .env or as an environment variable.", file=sys.stderr)
        sys.exit(1)

    # Derive model name from HF ID (e.g. "meta-llama/Llama-2-7b-hf" -> "Llama-2-7b-hf")
    model_name = args.model.split("/")[-1]
    output_dir = os.path.join(model_root, model_name)

    # Check if already converted
    meta_path = os.path.join(output_dir, "metadata.json")
    if os.path.isfile(meta_path) and not args.force:
        size_gb = _dir_size(output_dir) / 1e9
        print(f"Model '{model_name}' already exists ({size_gb:.1f} GB).")
        try:
            answer = input("Re-download and convert? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if answer != "y":
            return

    # ── Step 1: Download from HuggingFace ──

    download_dir = os.path.join(hf_cache, model_name)
    print(f"Downloading {args.model} to {download_dir}")
    print("This may take a while for large models...\n")

    hf_path = snapshot_download(
        repo_id=args.model,
        local_dir=download_dir,
        token=token,
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "*.model",         # sentencepiece tokenizer
            "*.jinja",         # chat template (Qwen)
            "tokenizer*",
            "vocab.*",
            "merges.txt",
        ],
        ignore_patterns=[
            "*.bin",           # skip pytorch_model.bin
            "*.h5",
            "*.ot",
            "*.msgpack",
            "training_args*",
        ],
    )
    dl_size = _dir_size(hf_path) / (1024**3)
    print(f"Download complete: {hf_path} ({dl_size:.2f} GB)\n")

    # ── Step 2: Detect architecture and convert ──

    arch = _detect_architecture(hf_path)
    print(f"Detected architecture: {arch}")
    print(f"Converting to GdsLLM format in {output_dir}\n")
    os.makedirs(output_dir, exist_ok=True)

    if arch == "qwen3.5-moe":
        _pull_convert_qwen_moe(hf_path, output_dir)
    else:
        _pull_convert_llama(hf_path, output_dir, args.quantize)

    # Summary
    total_size = _dir_size(output_dir)
    print(f"\nModel ready: {model_name}")
    print(f"  Size: {total_size / (1024**3):.2f} GB")
    print(f"  Path: {output_dir}")
    print(f"\nRun with:")
    print(f"  gdsllm run {model_name}")


# ─── rm ──────────────────────────────────────────────────────────────────────


def cmd_rm(args):
    """Delete a local model."""
    model_root = _model_root()
    if not model_root:
        print("Error: GDSLLM_MODEL_ROOT is not set.", file=sys.stderr)
        sys.exit(1)

    model_dir = _resolve_model_dir(args.model_name)
    if not os.path.isdir(model_dir):
        print(f"Error: Model not found: {args.model_name}", file=sys.stderr)
        sys.exit(1)

    meta_path = os.path.join(model_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        print(f"Error: Not a GdsLLM model directory (no metadata.json): {model_dir}", file=sys.stderr)
        sys.exit(1)

    model_name = os.path.basename(os.path.abspath(model_dir))
    size_gb = _dir_size(model_dir) / 1e9

    try:
        answer = input(f"Delete {model_name} ({size_gb:.1f} GB)? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if answer != "y":
        return

    shutil.rmtree(model_dir)
    print(f"Deleted {model_name}")

    # Check for cached HF download
    hf_cache = _hf_cache()
    if hf_cache:
        hf_dir = os.path.join(hf_cache, model_name)
        if os.path.isdir(hf_dir):
            hf_size_gb = _dir_size(hf_dir) / 1e9
            try:
                answer = input(
                    f"Also delete HF cache for {model_name} ({hf_size_gb:.1f} GB)? [y/N] "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if answer == "y":
                shutil.rmtree(hf_dir)
                print(f"Deleted HF cache: {hf_dir}")


# ─── list ────────────────────────────────────────────────────────────────────


def cmd_list(args):
    """List available local models."""
    model_root = args.model_root or _model_root()
    if not model_root:
        print(
            "Error: --model-root is required (or set GDSLLM_MODEL_ROOT)",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.isdir(model_root):
        print(f"Error: Directory not found: {model_root}", file=sys.stderr)
        sys.exit(1)

    models = []
    for entry in sorted(os.listdir(model_root)):
        meta_path = os.path.join(model_root, entry, "metadata.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)

            # Calculate size
            total_size = 0
            model_dir = os.path.join(model_root, entry)
            for root, _, files in os.walk(model_dir):
                for fname in files:
                    total_size += os.path.getsize(os.path.join(root, fname))

            quant = meta.get("quantization", "none")
            hidden = meta.get("hidden_size", 0)
            layers = meta.get("num_layers", 0)
            params = 12 * hidden * hidden * layers
            if params > 50e9:
                param_str = f"{params / 1e9:.0f}B"
            elif params > 1e9:
                param_str = f"{params / 1e9:.1f}B"
            else:
                param_str = f"{params / 1e6:.0f}M"

            models.append({
                "name": entry,
                "params": param_str,
                "quant": quant.upper() if quant != "none" else "FP16",
                "layers": layers,
                "size_gb": total_size / 1e9,
            })
        except (json.JSONDecodeError, OSError):
            continue

    if not models:
        print(f"No models found in {model_root}")
        return

    # Print table
    print(f"{'NAME':<40} {'PARAMS':>8} {'QUANT':>8} {'LAYERS':>8} {'SIZE':>10}")
    print("-" * 78)
    for m in models:
        print(
            f"{m['name']:<40} {m['params']:>8} {m['quant']:>8} "
            f"{m['layers']:>8} {m['size_gb']:>9.1f}G"
        )


# ─── show ────────────────────────────────────────────────────────────────────


def cmd_show(args):
    """Show model details."""
    model_dir = _resolve_model_dir(args.model_dir)
    if not os.path.isdir(model_dir):
        print(f"Error: Model directory not found: {model_dir}", file=sys.stderr)
        root = _model_root()
        if not root:
            print("Hint: set GDSLLM_MODEL_ROOT to resolve model names automatically", file=sys.stderr)
        sys.exit(1)

    meta_path = os.path.join(model_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        print(f"Error: No metadata.json in {model_dir}", file=sys.stderr)
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    name = os.path.basename(os.path.abspath(model_dir))
    quant = meta.get("quantization", "none")
    hidden = meta.get("hidden_size", 0)
    layers = meta.get("num_layers", 0)
    heads = meta.get("num_heads", 0)
    kv_heads = meta.get("num_kv_heads", 0)
    vocab = meta.get("vocab_size", 0)
    max_seq = meta.get("max_seq_len", 0)

    # Size
    total_size = _dir_size(model_dir)

    family = meta.get("model", "llama")

    print(f"  Model:          {name}")
    print(f"  Family:         {family}")
    print(f"  Quantization:   {quant.upper() if quant != 'none' else 'FP16'}")
    print(f"  Hidden size:    {hidden}")
    print(f"  Layers:         {layers}")
    print(f"  Attention:      {heads} heads ({kv_heads} KV heads)")
    print(f"  Vocab size:     {vocab}")
    print(f"  Max seq len:    {max_seq}")

    # MoE info
    num_experts = meta.get("num_experts")
    if num_experts:
        top_k = meta.get("num_experts_per_token", "?")
        print(f"  MoE:            {num_experts} experts (top-{top_k})")

    # Layer type breakdown
    layer_types = meta.get("layer_types", [])
    if layer_types:
        num_full = sum(1 for t in layer_types if t == "full_attention")
        num_linear = len(layer_types) - num_full
        print(f"  Layer types:    {num_full} full attn + {num_linear} linear attn")

    print(f"  Size on disk:   {total_size / 1e9:.2f} GB")


# ─── main ────────────────────────────────────────────────────────────────────


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="gdsllm",
        description="GdsLLM — LLM inference with NVMe-to-VRAM weight streaming",
        epilog=(
            "Environment variables:\n"
            "  GDSLLM_MODEL_ROOT      Root directory for converted GdsLLM models\n"
            "  GDSLLM_HF_CACHE        Directory for HuggingFace model downloads\n"
            "  HUGGINGFACE_HUB_TOKEN   HuggingFace authentication token\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # serve
    p_serve = subparsers.add_parser("serve", help="Start the API server")
    p_serve.add_argument(
        "model_dir",
        help="Model name or path (resolved against GDSLLM_MODEL_ROOT)",
    )
    p_serve.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    p_serve.add_argument("--preload", action="store_true", help="Preload layers into VRAM")
    p_serve.add_argument("--model-root", default=None, help="Override GDSLLM_MODEL_ROOT")

    # stop
    subparsers.add_parser("stop", help="Stop the API server")

    # run
    p_run = subparsers.add_parser("run", help="Interactive chat in terminal")
    p_run.add_argument(
        "model_dir",
        help="Model name or path (resolved against GDSLLM_MODEL_ROOT)",
    )
    p_run.add_argument("--max-tokens", type=int, default=256, help="Max tokens per response")
    p_run.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    p_run.add_argument("--top-p", type=float, default=1.0, help="Top-p (nucleus) sampling (default: 1.0)")
    p_run.add_argument("--top-k", type=int, default=0, help="Top-k sampling, 0=disabled (default: 0)")
    p_run.add_argument("--repeat-penalty", type=float, default=1.0, help="Repeat penalty, 1.0=disabled (default: 1.0)")
    p_run.add_argument("--preload", action="store_true", default=True, help="Preload layers into VRAM")
    p_run.add_argument("--prompt", type=str, default=None, help="Single prompt (non-interactive mode, exits after generation)")

    # pull
    p_pull = subparsers.add_parser("pull", help="Download and convert a HuggingFace model")
    p_pull.add_argument(
        "model",
        help="HuggingFace model ID (e.g. meta-llama/Llama-2-7b-hf)",
    )
    p_pull.add_argument(
        "--quantize", choices=["none", "int8"], default="none",
        help="Weight quantization (default: none = fp16)",
    )
    p_pull.add_argument(
        "--force", action="store_true",
        help="Re-download and convert even if model already exists",
    )

    # rm
    p_rm = subparsers.add_parser("rm", help="Delete a local model")
    p_rm.add_argument(
        "model_name",
        help="Model name or path to delete",
    )

    # list
    p_list = subparsers.add_parser("list", help="List available local models")
    p_list.add_argument("--model-root", default=None, help="Override GDSLLM_MODEL_ROOT")

    # show
    p_show = subparsers.add_parser("show", help="Show model details")
    p_show.add_argument(
        "model_dir",
        help="Model name or path (resolved against GDSLLM_MODEL_ROOT)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "serve": cmd_serve,
        "stop": cmd_stop,
        "run": cmd_run,
        "pull": cmd_pull,
        "rm": cmd_rm,
        "list": cmd_list,
        "show": cmd_show,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
