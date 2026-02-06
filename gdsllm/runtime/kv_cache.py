"""
GdsLLM — KV Cache for autoregressive LLaMA inference.

Pre-allocates K and V tensors for all layers.  During generation the
scheduler calls ``update()`` once per layer to store the new K/V and
retrieve the full cached history, then ``advance()`` once per forward
pass (after all layers) to bump the position counter.
"""

from typing import Tuple

import torch


class KVCache:
    """Pre-allocated KV cache living in VRAM.

    Shape per tensor: (num_layers, batch, num_kv_heads, max_seq_len, head_dim)
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cuda:0"),
    ):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.seq_pos = 0

        self.k_cache = torch.zeros(
            num_layers, batch_size, num_kv_heads, max_seq_len, head_dim,
            dtype=dtype, device=device,
        )
        self.v_cache = torch.zeros(
            num_layers, batch_size, num_kv_heads, max_seq_len, head_dim,
            dtype=dtype, device=device,
        )

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Store new K/V and return full cached K/V up to current position.

        Args:
            layer_idx: Transformer layer index.
            k: (batch, num_kv_heads, new_tokens, head_dim) — post-RoPE keys.
            v: (batch, num_kv_heads, new_tokens, head_dim) — values.

        Returns:
            (cached_k, cached_v) each (batch, num_kv_heads, seq_pos+new, head_dim).
        """
        new_tokens = k.shape[2]
        start = self.seq_pos
        end = start + new_tokens

        self.k_cache[layer_idx, :, :, start:end, :] = k
        self.v_cache[layer_idx, :, :, start:end, :] = v

        return (
            self.k_cache[layer_idx, :, :, :end, :],
            self.v_cache[layer_idx, :, :, :end, :],
        )

    def advance(self, num_tokens: int):
        """Advance position counter.  Call once per forward pass, after all layers."""
        self.seq_pos += num_tokens

    def reset(self):
        """Reset for a new sequence (no memory re-allocation)."""
        self.seq_pos = 0

    @property
    def current_seq_len(self) -> int:
        return self.seq_pos

    @staticmethod
    def estimate_vram_bytes(
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        batch_size: int = 1,
        bytes_per_element: int = 2,
    ) -> int:
        """Estimate total VRAM in bytes for the KV cache."""
        return (
            2 * num_layers * batch_size * num_kv_heads
            * max_seq_len * head_dim * bytes_per_element
        )
