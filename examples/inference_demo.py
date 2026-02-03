"""
GdsLLM — End-to-End Inference Demo

Generates text from a prompt using a LLaMA model with weights
streamed from NVMe directly to GPU VRAM via GPUDirect Storage.

Usage:
    python examples/inference_demo.py \
        --model-dir /mnt/SSD2TB/gdsllm_model \
        --tokenizer /path/to/tokenizer.model \
        --prompt "The meaning of life is"

    # With VRAM caching (preload all layers):
    python examples/inference_demo.py \
        --model-dir /mnt/SSD2TB/gdsllm_model \
        --tokenizer /path/to/tokenizer.model \
        --prompt "The meaning of life is" \
        --preload
"""

import argparse
import os
import time

import torch
from transformers import AutoTokenizer

from gdsllm.runtime.scheduler import SimpleScheduler, CachedScheduler


def print_diagnostics(scheduler):
    """Print GDS and system diagnostics."""
    print("\n=== GdsLLM Diagnostics ===")
    print(f"GDS driver open: {scheduler.weights.is_driver_open()}")
    print(f"Model layers: {scheduler.weights.num_layers}")
    print(f"Config: {scheduler.config}")

    # Cache stats
    if hasattr(scheduler, "cache") and scheduler.cache is not None:
        stats = scheduler.cache.stats
        print(f"Cache: {stats['cached_layers']}/{stats['total_layers']} layers, "
              f"{stats['cached_specials']} specials, "
              f"{stats['vram_used_mb']:.0f}/{stats['vram_budget_mb']:.0f} MB")

    # GPU info
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        vram_alloc = torch.cuda.memory_allocated(0) / (1024**2)
        print(f"GPU: {gpu} ({vram_total:.1f} GB VRAM)")
        print(f"VRAM allocated: {vram_alloc:.1f} MB")

    # RSS (resident memory)
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_mb = int(line.split()[1]) / 1024
                    print(f"System RAM (RSS): {rss_mb:.1f} MB")
                    break
    except OSError:
        pass
    print("==========================\n")


def generate(
    scheduler,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
):
    """Auto-regressive text generation."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    seq_len = input_ids.shape[1]

    mode = "cached" if hasattr(scheduler, "cache") else "streaming"
    print(f"Prompt: {prompt}")
    print(f"Input tokens: {seq_len}")
    print(f"Mode: {mode}")
    print(f"Generating (max {max_new_tokens} tokens)...\n")

    generated = input_ids[0].tolist()
    token_times = []

    for step in range(max_new_tokens):
        t0 = time.time()

        # Full forward pass (no KV cache)
        input_tensor = torch.tensor([generated], dtype=torch.long)
        logits = scheduler.forward(input_tensor)

        # Sample next token
        next_logits = logits[0, -1, :]
        if temperature <= 0:
            next_token = next_logits.argmax().item()
        else:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

        generated.append(next_token)
        elapsed = time.time() - t0
        token_times.append(elapsed)

        # Decode and print
        decoded = tokenizer.decode([next_token])
        vram_mb = torch.cuda.memory_allocated(0) / (1024**2)
        print(
            f"  [{step+1:3d}] token={next_token:6d} "
            f"'{decoded}' "
            f"({elapsed:.2f}s, VRAM: {vram_mb:.0f}MB)"
        )

        if next_token == tokenizer.eos_token_id:
            break

    output_text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"\n--- Output ---\n{output_text}\n")

    # Summary
    if token_times:
        print(f"--- Timing ---")
        print(f"  First token:  {token_times[0]:.2f}s")
        if len(token_times) > 1:
            avg_rest = sum(token_times[1:]) / len(token_times[1:])
            print(f"  Avg (rest):   {avg_rest:.2f}s")
            print(f"  Speedup:      {token_times[0] / avg_rest:.1f}x" if avg_rest > 0 else "")
        print(f"  Total:        {sum(token_times):.2f}s for {len(token_times)} tokens")

    # Cache stats
    if hasattr(scheduler, "cache") and scheduler.cache is not None:
        stats = scheduler.cache.stats
        print(f"\n--- Cache Stats ---")
        print(f"  Hits: {stats['hits']}, Misses: {stats['misses']}")
        print(f"  Cached: {stats['cached_layers']}/{stats['total_layers']} layers")
        print(f"  VRAM used: {stats['vram_used_mb']:.0f} MB")

    return output_text


def main():
    parser = argparse.ArgumentParser(
        description="GdsLLM inference demo — NVMe-to-VRAM weight streaming"
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to converted GdsLLM weights directory",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Path to HuggingFace tokenizer (model directory or tokenizer name)",
    )
    parser.add_argument(
        "--prompt",
        default="The meaning of life is",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload all layers into VRAM (uses CachedScheduler)",
    )
    parser.add_argument(
        "--vram-reserve",
        type=int,
        default=1024,
        help="VRAM to reserve for activations in MB (default: 1024)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Print GDS and system diagnostics",
    )
    args = parser.parse_args()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Choose scheduler
    if args.preload:
        print("Using CachedScheduler with VRAM preload...")
        ctx = CachedScheduler(
            args.model_dir,
            preload=True,
            vram_reserve_mb=args.vram_reserve,
        )
    else:
        print("Using SimpleScheduler (load/free per layer)...")
        ctx = SimpleScheduler(args.model_dir)

    t_preload = time.time()
    with ctx as scheduler:
        preload_time = time.time() - t_preload
        if args.preload:
            print(f"Preload complete in {preload_time:.2f}s")

        if args.verify:
            print_diagnostics(scheduler)

        generate(
            scheduler,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )


if __name__ == "__main__":
    main()
