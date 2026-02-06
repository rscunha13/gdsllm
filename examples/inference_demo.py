"""
GdsLLM — End-to-End Inference Demo

Generates text from a prompt using a LLaMA model with weights
streamed from NVMe directly to GPU VRAM via GPUDirect Storage.

Usage:
    # Chat mode (default for instruct models — applies chat template):
    python examples/inference_demo.py \
        --model-dir /mnt/SSD2TB/gdsllm_model \
        --prompt "What is the meaning of life?" \
        --preload

    # Completion mode (raw text, no chat template):
    python examples/inference_demo.py \
        --model-dir /mnt/SSD2TB/gdsllm_model \
        --prompt "The meaning of life is" \
        --preload --no-chat
"""

import argparse
import json
import os
import time

import torch
from transformers import AutoTokenizer

from gdsllm.runtime.scheduler import SimpleScheduler, CachedScheduler, estimate_activation_vram
from gdsllm.runtime.kv_cache import KVCache


def print_diagnostics(scheduler):
    """Print GDS and system diagnostics."""
    print("\n=== GdsLLM Diagnostics ===")
    print(f"GDS driver open: {scheduler.weights.is_driver_open()}")
    print(f"Model layers: {scheduler.weights.num_layers}")
    dtype = scheduler.config.get("dtype", "float16")
    quant = scheduler.config.get("quantization", "none")
    print(f"Weight dtype: {dtype} (quantization: {quant})")
    print(f"Config: {scheduler.config}")

    # Cache stats
    if hasattr(scheduler, "cache") and scheduler.cache is not None:
        stats = scheduler.cache.stats
        print(f"Cache: {stats['cached_layers']}/{stats['total_layers']} layers, "
              f"{stats['cached_specials']} specials, "
              f"{stats['vram_used_mb']:.0f} MB used, {stats['vram_free_mb']:.0f} MB free")

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


def _sample(logits: torch.Tensor, temperature: float) -> int:
    """Sample next token from logits."""
    next_logits = logits[0, -1, :]
    if temperature <= 0:
        return next_logits.argmax().item()
    probs = torch.softmax(next_logits / temperature, dim=-1)
    return torch.multinomial(probs, 1).item()


def _get_stop_token_ids(tokenizer):
    """Collect all token IDs that should trigger end of generation."""
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    # Check for additional stop tokens in added_tokens (e.g. <|end_of_text|>)
    for tok_name in ("<|end_of_text|>", "<|eot_id|>", "</s>"):
        tid = tokenizer.convert_tokens_to_ids(tok_name)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop_ids.add(tid)
    return stop_ids


def generate(
    scheduler,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    use_kv_cache: bool = True,
    chat: bool = True,
):
    """Auto-regressive text generation with optional KV cache."""
    # Apply chat template if available and requested
    if chat and hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        input_ids = tokenizer.encode(formatted, return_tensors="pt", add_special_tokens=False)
        print(f"Chat mode: on (template applied, {input_ids.shape[1]} prompt tokens)")
    else:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        if chat:
            print("Chat mode: off (no chat template found)")
        else:
            print("Chat mode: off")

    prompt_len = input_ids.shape[1]
    config = scheduler.config
    stop_ids = _get_stop_token_ids(tokenizer)

    mode = "cached" if hasattr(scheduler, "cache") else "streaming"
    print(f"Prompt: {prompt}")
    print(f"Input tokens: {prompt_len}")
    print(f"Mode: {mode}, KV cache: {'on' if use_kv_cache else 'off'}")
    print(f"Stop tokens: {stop_ids}")

    if use_kv_cache:
        return _generate_with_kv_cache(
            scheduler, tokenizer, input_ids, config,
            max_new_tokens, temperature, stop_ids,
        )
    else:
        return _generate_no_kv_cache(
            scheduler, tokenizer, input_ids,
            max_new_tokens, temperature, stop_ids,
        )


def _generate_with_kv_cache(
    scheduler, tokenizer, input_ids, config,
    max_new_tokens, temperature, stop_ids,
):
    """Two-phase generation: prefill + decode."""
    prompt_len = input_ids.shape[1]
    max_total = prompt_len + max_new_tokens

    # Allocate KV cache
    kv = KVCache(
        num_layers=config["num_layers"],
        num_kv_heads=config["num_kv_heads"],
        head_dim=config["head_dim"],
        max_seq_len=max_total,
    )
    kv_mb = KVCache.estimate_vram_bytes(
        config["num_layers"], config["num_kv_heads"],
        config["head_dim"], max_total,
    ) / (1024 * 1024)
    print(f"KV cache: {kv_mb:.0f} MB for max_seq_len={max_total}")
    print(f"Generating (max {max_new_tokens} tokens)...\n")

    generated = input_ids[0].tolist()
    token_times = []

    # Phase 1: Prefill — process entire prompt
    t0 = time.time()
    logits = scheduler.forward(input_ids, kv_cache=kv, start_pos=0)
    next_token = _sample(logits, temperature)
    generated.append(next_token)
    prefill_time = time.time() - t0
    token_times.append(prefill_time)

    decoded = tokenizer.decode([next_token])
    print(f"  [  1] token={next_token:6d} '{decoded}' "
          f"(prefill {prefill_time:.2f}s, {prompt_len} tokens)")

    if next_token in stop_ids:
        return _finish(tokenizer, generated, token_times, scheduler, prompt_len)

    # Phase 2: Decode — one token at a time
    for step in range(1, max_new_tokens):
        t0 = time.time()

        input_tensor = torch.tensor([[next_token]], dtype=torch.long)
        logits = scheduler.forward(input_tensor, kv_cache=kv, start_pos=kv.current_seq_len)
        next_token = _sample(logits, temperature)
        generated.append(next_token)

        elapsed = time.time() - t0
        token_times.append(elapsed)

        decoded = tokenizer.decode([next_token])
        vram_mb = torch.cuda.memory_allocated(0) / (1024**2)
        print(f"  [{step+1:3d}] token={next_token:6d} '{decoded}' "
              f"({elapsed:.3f}s, VRAM: {vram_mb:.0f}MB)")

        if next_token in stop_ids:
            break

    return _finish(tokenizer, generated, token_times, scheduler, prompt_len)


def _generate_no_kv_cache(
    scheduler, tokenizer, input_ids,
    max_new_tokens, temperature, stop_ids,
):
    """Full recompute per token (original behavior, no KV cache)."""
    print(f"Generating (max {max_new_tokens} tokens)...\n")

    generated = input_ids[0].tolist()
    token_times = []

    for step in range(max_new_tokens):
        t0 = time.time()

        input_tensor = torch.tensor([generated], dtype=torch.long)
        logits = scheduler.forward(input_tensor)
        next_token = _sample(logits, temperature)
        generated.append(next_token)

        elapsed = time.time() - t0
        token_times.append(elapsed)

        decoded = tokenizer.decode([next_token])
        vram_mb = torch.cuda.memory_allocated(0) / (1024**2)
        print(f"  [{step+1:3d}] token={next_token:6d} '{decoded}' "
              f"({elapsed:.2f}s, VRAM: {vram_mb:.0f}MB)")

        if next_token in stop_ids:
            break

    return _finish(tokenizer, generated, token_times, scheduler, prompt_len=None)


def _finish(tokenizer, generated, token_times, scheduler, prompt_len):
    """Print output text, timing summary, and cache stats."""
    output_text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"\n--- Output ---\n{output_text}\n")

    if token_times:
        print("--- Timing ---")
        if prompt_len is not None and len(token_times) > 1:
            # KV cache mode: first entry is prefill, rest are decode
            print(f"  Prefill:      {token_times[0]:.2f}s ({prompt_len} tokens)")
            decode_times = token_times[1:]
            if decode_times:
                avg = sum(decode_times) / len(decode_times)
                print(f"  Decode avg:   {avg:.3f}s/token ({1/avg:.1f} tok/s)")
        else:
            print(f"  First token:  {token_times[0]:.2f}s")
            if len(token_times) > 1:
                avg_rest = sum(token_times[1:]) / len(token_times[1:])
                print(f"  Avg (rest):   {avg_rest:.2f}s")
                if avg_rest > 0:
                    print(f"  Speedup:      {token_times[0] / avg_rest:.1f}x")
        print(f"  Total:        {sum(token_times):.2f}s for {len(token_times)} tokens")

    # Cache stats
    if hasattr(scheduler, "cache") and scheduler.cache is not None:
        stats = scheduler.cache.stats
        print(f"\n--- Cache Stats ---")
        print(f"  Hits: {stats['hits']}, Misses: {stats['misses']}, Evictions: {stats['evictions']}")
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
        "--no-kv-cache",
        action="store_true",
        help="Disable KV cache (full recompute per token, for comparison)",
    )
    parser.add_argument(
        "--no-chat",
        action="store_true",
        help="Disable chat template (raw completion mode)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Print GDS and system diagnostics",
    )
    args = parser.parse_args()

    # Load tokenizer from model directory
    print(f"Loading tokenizer from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    # Estimate max sequence length for VRAM budgeting
    use_chat = not args.no_chat and hasattr(tokenizer, "chat_template") and tokenizer.chat_template
    if use_chat:
        # Account for chat template overhead in token count
        messages = [{"role": "user", "content": args.prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompt_tokens = len(tokenizer.encode(formatted, add_special_tokens=False))
    else:
        prompt_tokens = len(tokenizer.encode(args.prompt))
    max_seq_len = prompt_tokens + args.max_tokens

    # Choose scheduler
    if args.preload:
        reserve_mb = estimate_activation_vram(
            CachedScheduler(args.model_dir, preload=False).config, max_seq_len
        ) // (1024 * 1024)
        print(f"Using CachedScheduler (max_seq_len={max_seq_len}, activation_reserve={reserve_mb} MB)...")
        ctx = CachedScheduler(
            args.model_dir,
            preload=True,
            max_seq_len=max_seq_len,
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
            use_kv_cache=not args.no_kv_cache,
            chat=not args.no_chat,
        )


if __name__ == "__main__":
    main()
