"""
Download a LLaMA model from HuggingFace Hub.

Authenticates via HUGGINGFACE_HUB_TOKEN environment variable.
Downloads only the safetensors weights, tokenizer, and config.

Usage:
    export HUGGINGFACE_HUB_TOKEN="hf_..."
    python -m gdsllm.tools.download_model \
        --model meta-llama/Llama-2-7b-hf \
        --output-dir /mnt/SSD2TB/AIModels/llama-2-7b-hf
"""

import argparse
import os
from pathlib import Path
import sys

from huggingface_hub import snapshot_download


def load_dotenv():
    """Load .env file from the project root."""
    # Walk up from this file to find .env
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


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Download a LLaMA model from HuggingFace Hub"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace model ID (default: meta-llama/Llama-2-7b-hf)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Local directory to save the model",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace token (default: reads from .env or HUGGINGFACE_HUB_TOKEN env var)",
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print(
            "Error: No HuggingFace token found.\n"
            "Set HUGGINGFACE_HUB_TOKEN env var or pass --token.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Downloading {args.model} to {args.output_dir}")
    print("This may take a while for large models...\n")

    path = snapshot_download(
        repo_id=args.model,
        local_dir=args.output_dir,
        token=token,
        # Only download what we need: safetensors, config, tokenizer
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "*.model",         # sentencepiece tokenizer
            "tokenizer*",
        ],
        ignore_patterns=[
            "*.bin",           # skip pytorch_model.bin (we use safetensors)
            "*.h5",
            "*.ot",
            "*.msgpack",
            "training_args*",
        ],
    )

    print(f"\nDownload complete: {path}")

    # List downloaded files
    total_size = 0
    for root, _, files in os.walk(path):
        for f in sorted(files):
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            total_size += size
            rel = os.path.relpath(fpath, path)
            print(f"  {rel:50s} {size / (1024**2):8.1f} MB")

    print(f"\nTotal: {total_size / (1024**3):.2f} GB")
    print(f"\nNext step: convert weights with:")
    print(f"  python -m gdsllm.tools.convert_weights \\")
    print(f"      --model-dir {path} \\")
    print(f"      --output-dir /mnt/SSD2TB/gdsllm_model")


if __name__ == "__main__":
    main()
