"""
GdsLLM — Schedulers for layer-by-layer LLaMA inference.

SimpleScheduler: loads/frees each layer per forward pass (minimal VRAM).
CachedScheduler: caches layers in VRAM for fast subsequent passes.
"""

import torch
import torch.nn.functional as F
from typing import Optional

from gdsllm.runtime.torch_bridge import ModelWeights, LayerCache
from gdsllm.runtime.llama import (
    rms_norm,
    precompute_freqs_cis,
    transformer_block,
)


class SimpleScheduler:
    """Layer-by-layer LLaMA inference with GDS weight streaming."""

    def __init__(self, model_dir: str, device: str = "cuda:0"):
        self.weights = ModelWeights(model_dir)
        self.device = torch.device(device)
        self.config = self.weights.config

        # Precompute RoPE frequencies (small, lives in VRAM permanently)
        self.freqs_cis = precompute_freqs_cis(
            head_dim=self.config["head_dim"],
            seq_len=self.config["max_seq_len"],
            theta=self.config.get("rope_theta", 10000.0),
            device=self.device,
        )

    def __enter__(self):
        self.weights.init_gds()
        return self

    def __exit__(self, *args):
        self.weights.shutdown_gds()

    @torch.inference_mode()
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Full forward pass through the LLaMA model.

        Args:
            token_ids: (1, seq_len) long tensor

        Returns:
            logits: (1, seq_len, vocab_size) float tensor on CUDA
        """
        batch, seq_len = token_ids.shape
        assert batch == 1, "MVP supports batch=1 only"
        assert seq_len <= self.config["max_seq_len"], (
            f"Sequence length {seq_len} exceeds max {self.config['max_seq_len']}"
        )

        num_heads = self.config["num_heads"]
        num_kv_heads = self.config["num_kv_heads"]
        rms_norm_eps = self.config["rms_norm_eps"]

        # Step 1: Token embeddings
        embed_tensors, embed_buf = self.weights.load_special("embed_tokens")
        h = F.embedding(
            token_ids.to(self.device),
            embed_tensors["model.embed_tokens.weight"],
        )
        del embed_tensors, embed_buf
        torch.cuda.empty_cache()

        # Step 2: Causal mask
        mask = torch.full(
            (seq_len, seq_len), float("-inf"), device=self.device
        )
        mask = torch.triu(mask, diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)

        # Step 3: RoPE frequencies for this sequence
        freqs = self.freqs_cis[:seq_len]

        # Step 4: Transformer layers
        for layer_idx in range(self.weights.num_layers):
            layer_tensors, layer_buf = self.weights.load_layer(layer_idx)
            h = transformer_block(
                h, layer_tensors, freqs, mask,
                num_heads, num_kv_heads, rms_norm_eps,
            )
            del layer_tensors, layer_buf
            torch.cuda.empty_cache()

        # Step 5: Final norm
        norm_tensors, norm_buf = self.weights.load_special("final_norm")
        h = rms_norm(h, norm_tensors["model.norm.weight"], rms_norm_eps)
        del norm_tensors, norm_buf
        torch.cuda.empty_cache()

        # Step 6: LM head
        if self.weights.special["lm_head"] is not None:
            lm_tensors, lm_buf = self.weights.load_special("lm_head")
            logits = F.linear(h, lm_tensors["lm_head.weight"])
            del lm_tensors, lm_buf
        else:
            # Tied weights: reload embed_tokens for projection
            embed_tensors, embed_buf = self.weights.load_special("embed_tokens")
            logits = F.linear(h, embed_tensors["model.embed_tokens.weight"])
            del embed_tensors, embed_buf
        torch.cuda.empty_cache()

        return logits


class CachedScheduler:
    """LLaMA inference with VRAM-cached layers.

    Loads layers via GDS on first access and keeps them resident in VRAM.
    Subsequent forward passes reuse cached layers with zero I/O.
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda:0",
        preload: bool = True,
        vram_reserve_mb: int = 1024,
    ):
        self.weights = ModelWeights(model_dir)
        self.device = torch.device(device)
        self.config = self.weights.config
        self._preload = preload
        self._vram_reserve_mb = vram_reserve_mb
        self.cache: Optional[LayerCache] = None

        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(
            head_dim=self.config["head_dim"],
            seq_len=self.config["max_seq_len"],
            theta=self.config.get("rope_theta", 10000.0),
            device=self.device,
        )

    def __enter__(self):
        self.weights.init_gds()
        self.cache = LayerCache(
            self.weights,
            vram_reserve_mb=self._vram_reserve_mb,
        )
        if self._preload:
            self.cache.preload()
        return self

    def __exit__(self, *args):
        if self.cache:
            self.cache.evict_all()
        self.weights.shutdown_gds()

    @torch.inference_mode()
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Full forward pass using cached layers."""
        batch, seq_len = token_ids.shape
        assert batch == 1, "MVP supports batch=1 only"
        assert seq_len <= self.config["max_seq_len"]

        num_heads = self.config["num_heads"]
        num_kv_heads = self.config["num_kv_heads"]
        rms_norm_eps = self.config["rms_norm_eps"]

        # Step 1: Token embeddings
        embed_tensors = self.cache.get_special("embed_tokens")
        h = F.embedding(
            token_ids.to(self.device),
            embed_tensors["model.embed_tokens.weight"],
        )

        # Step 2: Causal mask
        mask = torch.full(
            (seq_len, seq_len), float("-inf"), device=self.device
        )
        mask = torch.triu(mask, diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)

        # Step 3: RoPE frequencies
        freqs = self.freqs_cis[:seq_len]

        # Step 4: Transformer layers (cached — no load/free per call)
        for layer_idx in range(self.weights.num_layers):
            layer_tensors = self.cache.get_layer(layer_idx)
            h = transformer_block(
                h, layer_tensors, freqs, mask,
                num_heads, num_kv_heads, rms_norm_eps,
            )

        # Step 5: Final norm
        norm_tensors = self.cache.get_special("final_norm")
        h = rms_norm(h, norm_tensors["model.norm.weight"], rms_norm_eps)

        # Step 6: LM head
        if self.weights.special["lm_head"] is not None:
            lm_tensors = self.cache.get_special("lm_head")
            logits = F.linear(h, lm_tensors["lm_head.weight"])
        else:
            embed_tensors = self.cache.get_special("embed_tokens")
            logits = F.linear(h, embed_tensors["model.embed_tokens.weight"])

        return logits
