"""
GdsLLM — Hybrid State Cache for Qwen3.5-MoE.

Combines KV cache (for full attention layers) with SSM recurrent state
(for linear attention / Mamba layers). The two cache types live in VRAM
and are indexed by layer_idx.

Full attention layers (15 of 60): standard KV cache
Linear attention layers (45 of 60): conv state + SSM hidden state
"""

from typing import Dict, List, Optional, Tuple

import torch


class HybridCache:
    """Pre-allocated hybrid cache for Qwen3.5-MoE inference.

    KV cache for full attention layers:
        k_cache/v_cache: (num_full_layers, batch, num_kv_heads, max_seq_len, head_dim)

    SSM state for linear attention layers:
        conv_state: (num_linear_layers, batch, d_inner, kernel_size-1)
        ssm_state: (num_linear_layers, batch, state_dim, bc_dim)
    """

    def __init__(
        self,
        layer_types: List[str],
        # Full attention params
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        # Common
        batch_size: int = 1,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cuda:0"),
        # Legacy (ignored, SSM state is now lazily populated)
        **_kwargs,
    ):
        self.layer_types = layer_types
        self.num_layers = len(layer_types)
        self.max_seq_len = max_seq_len
        self.seq_pos = 0

        # Map layer_idx -> cache index for each type
        self._full_indices: Dict[int, int] = {}
        self._linear_indices: Dict[int, int] = {}
        full_count = 0
        linear_count = 0
        for i, lt in enumerate(layer_types):
            if lt == "full_attention":
                self._full_indices[i] = full_count
                full_count += 1
            else:
                self._linear_indices[i] = linear_count
                linear_count += 1

        self.num_full = full_count
        self.num_linear = linear_count

        # KV cache for full attention layers
        if full_count > 0:
            self.k_cache = torch.zeros(
                full_count, batch_size, num_kv_heads, max_seq_len, head_dim,
                dtype=dtype, device=device,
            )
            self.v_cache = torch.zeros(
                full_count, batch_size, num_kv_heads, max_seq_len, head_dim,
                dtype=dtype, device=device,
            )
        else:
            self.k_cache = None
            self.v_cache = None

        # SSM state for linear attention layers (gated delta rule)
        # Stored as a plain dict since state shapes vary and are
        # populated lazily during the first forward pass.
        # Keys: "conv_{layer_idx}" -> (batch, 12288, kernel-1)
        #        "ssm_{layer_idx}" -> (batch, num_v_heads, k_dim, v_dim)
        self.ssm_state: Dict[str, torch.Tensor] = {}

        # Keep legacy attributes for compatibility
        self.conv_state = None

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Store K/V and return full cached history (full attention layers)."""
        cache_idx = self._full_indices[layer_idx]
        new_tokens = k.shape[2]
        start = self.seq_pos
        end = start + new_tokens

        self.k_cache[cache_idx, :, :, start:end, :] = k
        self.v_cache[cache_idx, :, :, start:end, :] = v

        return (
            self.k_cache[cache_idx, :, :, :end, :],
            self.v_cache[cache_idx, :, :, :end, :],
        )

    def get_ssm_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get the SSM state dict (shared across all linear attention layers).

        The dict is populated lazily by linear_attention() during forward pass.
        Keys: "conv_{layer_idx}", "ssm_{layer_idx}"
        """
        return self.ssm_state

    def advance(self, num_tokens: int):
        """Advance position counter (call once after all layers)."""
        self.seq_pos += num_tokens

    def reset(self):
        """Reset for a new sequence."""
        self.seq_pos = 0
        if self.k_cache is not None:
            self.k_cache.zero_()
            self.v_cache.zero_()
        self.ssm_state.clear()

    @property
    def current_seq_len(self) -> int:
        return self.seq_pos

    def is_full_attention(self, layer_idx: int) -> bool:
        return layer_idx in self._full_indices

    @staticmethod
    def estimate_vram_bytes(
        layer_types: List[str],
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        batch_size: int = 1,
        num_v_heads: int = 64,
        k_head_dim: int = 128,
        v_head_dim: int = 128,
        conv_dim: int = 12288,
        conv_kernel_size: int = 4,
    ) -> int:
        """Estimate total VRAM for the hybrid cache."""
        num_full = sum(1 for t in layer_types if t == "full_attention")
        num_linear = len(layer_types) - num_full

        # KV cache: 2 * num_full * batch * kv_heads * max_seq * head_dim * 2 bytes
        kv_bytes = 2 * num_full * batch_size * num_kv_heads * max_seq_len * head_dim * 2

        # Conv state: num_linear * batch * conv_dim * (kernel-1) * 2 bytes
        conv_bytes = num_linear * batch_size * conv_dim * (conv_kernel_size - 1) * 2

        # SSM state: num_linear * batch * num_v_heads * k_dim * v_dim * 4 bytes (float32)
        ssm_bytes = num_linear * batch_size * num_v_heads * k_head_dim * v_head_dim * 4

        return kv_bytes + conv_bytes + ssm_bytes
