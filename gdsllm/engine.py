"""
GdsLLM — Inference Engine

Wraps the scheduler, tokenizer, and generation loop into a reusable
class that can be consumed by the API server, CLI, or directly.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Generator, Optional

import torch
from transformers import AutoTokenizer

logger = logging.getLogger("gdsllm.engine")

from gdsllm.runtime.scheduler import (
    CachedScheduler,
    SimpleScheduler,
    estimate_activation_vram,
)
from gdsllm.runtime.kv_cache import KVCache


@dataclass
class TokenEvent:
    """A single event yielded during token generation."""

    token_id: int = 0
    text: str = ""
    done: bool = False
    done_reason: Optional[str] = None  # "stop", "length"
    # Timing — populated on the final event (done=True)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prefill_duration_ns: int = 0
    eval_duration_ns: int = 0
    total_duration_ns: int = 0


def _sample(logits: torch.Tensor, temperature: float) -> int:
    """Sample next token from logits."""
    next_logits = logits[0, -1, :]
    if temperature <= 0:
        return next_logits.argmax().item()
    probs = torch.softmax(next_logits / temperature, dim=-1)
    return torch.multinomial(probs, 1).item()


def _get_stop_token_ids(tokenizer) -> set:
    """Collect all token IDs that should trigger end of generation."""
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    for tok_name in ("<|end_of_text|>", "<|eot_id|>", "</s>"):
        tid = tokenizer.convert_tokens_to_ids(tok_name)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop_ids.add(tid)
    return stop_ids


def check_nvme(path: str) -> bool:
    """Check if the given path resides on an NVMe device.

    Reads /proc/mounts to find the mount point, then checks if the
    underlying block device is NVMe (e.g. /dev/nvme0n1p1).

    Returns True if on NVMe, False otherwise. Logs a warning if not.
    """
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
                # Unescape octal sequences in mount paths (e.g. \040 for space)
                mount = mount.encode("raw_unicode_escape").decode("unicode_escape")
                if path.startswith(mount) and len(mount) > len(best_mount):
                    best_mount = mount
                    best_dev = dev
        if not best_dev:
            return True  # can't determine, assume ok
        # NVMe devices are /dev/nvme*
        is_nvme = "nvme" in os.path.basename(best_dev)
        if not is_nvme:
            logger.warning(
                f"Model directory is NOT on an NVMe device (found {best_dev}). "
                f"GPUDirect Storage requires NVMe for optimal performance. "
                f"Inference will fall back to a slower I/O path."
            )
        return is_nvme
    except OSError:
        return True  # can't check, assume ok


class InferenceEngine:
    """Manages a loaded model and exposes token generation.

    Usage:
        engine = InferenceEngine("/path/to/model", preload=True)
        for event in engine.generate("Hello world"):
            print(event.text, end="", flush=True)
        engine.shutdown()
    """

    def __init__(self, model_dir: str, preload: bool = True):
        self.model_dir = os.path.abspath(model_dir)
        self.model_name = os.path.basename(self.model_dir)

        check_nvme(self.model_dir)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.stop_ids = _get_stop_token_ids(self.tokenizer)

        # Create scheduler
        if preload:
            self._scheduler = CachedScheduler(
                self.model_dir, preload=False,
            )
            # We'll preload after entering context
            self._preload = True
        else:
            self._scheduler = SimpleScheduler(self.model_dir)
            self._preload = False

        self.config = self._scheduler.config
        self._context_entered = False
        self._enter_context()

    def _enter_context(self):
        """Enter the scheduler context (init GDS, preload layers)."""
        self._scheduler.__enter__()
        self._context_entered = True

        if self._preload and isinstance(self._scheduler, CachedScheduler):
            # Estimate activation VRAM for a reasonable max_seq_len
            max_seq_len = min(self.config.get("max_seq_len", 4096), 4096)
            min_free = estimate_activation_vram(self.config, max_seq_len)
            self._scheduler.cache.preload(min_free=min_free)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        chat: bool = True,
    ) -> Generator[TokenEvent, None, None]:
        """Generate tokens from a text prompt.

        Yields TokenEvent for each generated token. The final event
        has done=True with timing information.
        """
        # Apply chat template if available and requested
        if chat and hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            input_ids = self.tokenizer.encode(
                formatted, return_tensors="pt", add_special_tokens=False,
            )
        else:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        yield from self._generate_tokens(input_ids, max_tokens, temperature)

    def generate_chat(
        self,
        messages: list[dict],
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> Generator[TokenEvent, None, None]:
        """Generate tokens from a chat message list.

        Messages should be [{"role": "user", "content": "..."}].
        Applies the chat template automatically.
        """
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            input_ids = self.tokenizer.encode(
                formatted, return_tensors="pt", add_special_tokens=False,
            )
        else:
            # Fallback: concatenate messages
            text = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
            )
            text += "\nassistant: "
            input_ids = self.tokenizer.encode(text, return_tensors="pt")

        yield from self._generate_tokens(input_ids, max_tokens, temperature)

    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
        temperature: float,
    ) -> Generator[TokenEvent, None, None]:
        """Two-phase generation: prefill + decode with KV cache."""
        prompt_len = input_ids.shape[1]
        max_total = prompt_len + max_tokens

        # Allocate KV cache
        kv = KVCache(
            num_layers=self.config["num_layers"],
            num_kv_heads=self.config["num_kv_heads"],
            head_dim=self.config["head_dim"],
            max_seq_len=max_total,
        )

        t_start = time.perf_counter_ns()
        completion_tokens = 0

        # Phase 1: Prefill
        logits = self._scheduler.forward(input_ids, kv_cache=kv, start_pos=0)
        next_token = _sample(logits, temperature)
        t_prefill = time.perf_counter_ns()
        prefill_ns = t_prefill - t_start
        completion_tokens += 1

        decoded = self.tokenizer.decode([next_token])

        if next_token in self.stop_ids:
            yield TokenEvent(
                token_id=next_token, text=decoded, done=True,
                done_reason="stop",
                prompt_tokens=prompt_len,
                completion_tokens=completion_tokens,
                prefill_duration_ns=prefill_ns,
                eval_duration_ns=0,
                total_duration_ns=time.perf_counter_ns() - t_start,
            )
            return

        yield TokenEvent(token_id=next_token, text=decoded)

        # Phase 2: Decode — one token at a time
        for step in range(1, max_tokens):
            input_tensor = torch.tensor([[next_token]], dtype=torch.long)
            logits = self._scheduler.forward(
                input_tensor, kv_cache=kv, start_pos=kv.current_seq_len,
            )
            next_token = _sample(logits, temperature)
            completion_tokens += 1

            decoded = self.tokenizer.decode([next_token])

            is_stop = next_token in self.stop_ids
            is_last = step == max_tokens - 1

            if is_stop or is_last:
                t_end = time.perf_counter_ns()
                yield TokenEvent(
                    token_id=next_token,
                    text="" if is_stop else decoded,
                    done=True,
                    done_reason="stop" if is_stop else "length",
                    prompt_tokens=prompt_len,
                    completion_tokens=completion_tokens,
                    prefill_duration_ns=prefill_ns,
                    eval_duration_ns=t_end - t_prefill,
                    total_duration_ns=t_end - t_start,
                )
                return

            yield TokenEvent(token_id=next_token, text=decoded)

    @property
    def model_info(self) -> dict:
        """Return model metadata for API responses."""
        # Calculate total model size from layer files
        total_size = 0
        for layer in self._scheduler.weights.layers:
            total_size += layer.size_bytes
        for name, special in self._scheduler.weights.special.items():
            if special is not None:
                total_size += special.size_bytes

        return {
            "name": self.model_name,
            "model_dir": self.model_dir,
            "family": "llama",
            "parameter_size": self._guess_param_size(),
            "quantization": self.config.get("quantization", "none"),
            "hidden_size": self.config.get("hidden_size"),
            "num_layers": self.config.get("num_layers"),
            "num_heads": self.config.get("num_heads"),
            "vocab_size": self.config.get("vocab_size"),
            "max_seq_len": self.config.get("max_seq_len"),
            "size_bytes": total_size,
        }

    def _guess_param_size(self) -> str:
        """Guess parameter count from hidden_size and num_layers."""
        hidden = self.config.get("hidden_size", 0)
        layers = self.config.get("num_layers", 0)
        # Rough estimate: params ~ 12 * hidden^2 * layers
        params = 12 * hidden * hidden * layers
        if params > 50e9:
            return f"{params / 1e9:.0f}B"
        elif params > 1e9:
            return f"{params / 1e9:.1f}B"
        elif params > 1e6:
            return f"{params / 1e6:.0f}M"
        return "unknown"

    def shutdown(self):
        """Clean up scheduler, free VRAM."""
        if self._context_entered:
            self._scheduler.__exit__(None, None, None)
            self._context_entered = False
