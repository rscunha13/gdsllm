"""
GdsLLM — Scheduler for Qwen3.5-MoE inference.

Layer-by-layer forward pass with GDS weight streaming, using:
- Two-phase loading: attention+router first, then selective expert loading
- HybridCache for KV (full attention) + SSM (linear attention) state
- Affine Q4 dequantization
- MoE MLP with top-k routing (only 10 of 512 experts loaded per token)
"""

import torch
import torch.nn.functional as F
from typing import Optional

from gdsllm.runtime.torch_bridge import ModelWeights, LayerCache
from gdsllm.runtime.qwen_moe import (
    rms_norm,
    precompute_freqs_cis,
    qwen_moe_attention,
    moe_route,
    moe_mlp_selective,
    _dequant_affine_q4,
    affine_q4_linear,
)
from gdsllm.runtime.hybrid_cache import HybridCache


def _embed_tokens(token_ids, tensors, scratch=None):
    """Embedding lookup with affine Q4 support."""
    w = tensors["model.embed_tokens.weight"]
    if isinstance(w, dict) and w.get("quant_type") == "affine_q4":
        w_fp16 = _dequant_affine_q4(w, scratch)
        h = F.embedding(token_ids, w_fp16)
        del w_fp16
    elif isinstance(w, dict) and "gguf_type" in w:
        from gdsllm.runtime.llama import _dequant_gguf
        w_fp16 = _dequant_gguf(w, scratch)
        h = F.embedding(token_ids, w_fp16)
        del w_fp16
    else:
        h = F.embedding(token_ids, w)
    return h


def _lm_head_proj(h, tensors, key="lm_head.weight", scratch=None):
    """LM head projection with affine Q4 support."""
    w = tensors[key]
    if isinstance(w, dict) and w.get("quant_type") == "affine_q4":
        w = _dequant_affine_q4(w, scratch)
    elif isinstance(w, dict) and "gguf_type" in w:
        from gdsllm.runtime.llama import _dequant_gguf
        w = _dequant_gguf(w, scratch)
    return F.linear(h, w)


class QwenMoEScheduler:
    """Layer-by-layer Qwen3.5-MoE inference with two-phase GDS streaming.

    Phase 1: Load attention + norms + router + shared expert (~100 MB)
    Phase 2: After routing, load only the selected experts (~65 MB)
    Total per-layer: ~165 MB instead of 3.3 GB
    """

    def __init__(self, model_dir: str, device: str = "cuda:0"):
        self.weights = ModelWeights(model_dir)
        self.device = torch.device(device)
        self.config = self.weights.config
        self._scratch = None

        # Layer types from metadata
        self.layer_types = self.config.get("layer_types", [])
        if not self.layer_types:
            num_layers = self.config["num_layers"]
            self.layer_types = [
                "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                for i in range(num_layers)
            ]

        # Precompute RoPE frequencies for full attention layers
        self.freqs_cis = precompute_freqs_cis(
            head_dim=self.config["head_dim"],
            seq_len=min(self.config.get("max_seq_len", 32768), 32768),
            theta=self.config.get("rope_theta", 10000000.0),
            device=self.device,
            partial_rotary_factor=self.config.get("partial_rotary_factor", 0.25),
        )

    def __enter__(self):
        self.weights.init_gds()
        self._scratch = self._alloc_scratch()
        return self

    def __exit__(self, *args):
        self.weights.shutdown_gds()

    def _alloc_scratch(self):
        """Allocate reusable scratch buffer for affine Q4 dequant.

        Sized for the largest single-expert tensor (not all 512).
        """
        max_elements = 0
        for layer in self.weights.layers:
            for t in layer.tensors:
                if t.quant == "affine_q4":
                    elems = 1
                    for s in t.shape:
                        elems *= s
                    # For 3D MoE tensors, scratch only needs 1 expert
                    if len(t.shape) == 3:
                        elems = elems // t.shape[0]
                    max_elements = max(max_elements, elems)
        if max_elements == 0:
            return None
        return torch.empty(max_elements, dtype=torch.float16, device=self.device)

    def create_cache(self, max_seq_len: int = 4096) -> HybridCache:
        """Create a hybrid KV+SSM cache."""
        config = self.config
        linear_key_heads = config.get("linear_num_key_heads", 16)
        linear_key_dim = config.get("linear_key_head_dim", 128)
        linear_value_heads = config.get("linear_num_value_heads", 64)
        linear_value_dim = config.get("linear_value_head_dim", 128)
        linear_hidden_dim = linear_key_heads * linear_key_dim
        linear_state_dim = linear_hidden_dim
        linear_bc_dim = linear_value_heads * linear_value_dim // linear_hidden_dim

        return HybridCache(
            layer_types=self.layer_types,
            num_kv_heads=config["num_kv_heads"],
            head_dim=config["head_dim"],
            max_seq_len=max_seq_len,
            linear_hidden_dim=linear_hidden_dim,
            linear_state_dim=linear_state_dim,
            linear_bc_dim=max(linear_bc_dim, 1),
            conv_kernel_size=config.get("linear_conv_kernel_dim", 4),
            device=self.device,
        )

    @torch.inference_mode()
    def forward(
        self,
        token_ids: torch.Tensor,
        cache: Optional[HybridCache] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Full forward pass with two-phase layer loading.

        Args:
            token_ids: (1, seq_len) long tensor
            cache: optional HybridCache for incremental decoding
            start_pos: position offset for RoPE

        Returns:
            logits: (1, seq_len, vocab_size) float tensor
        """
        batch, seq_len = token_ids.shape
        assert batch == 1, "batch=1 only"

        config = self.config
        scratch = self._scratch

        # Step 1: Token embeddings
        embed_tensors, embed_buf = self.weights.load_special("embed_tokens")
        h = _embed_tokens(token_ids.to(self.device), embed_tensors)
        del embed_tensors, embed_buf
        torch.cuda.empty_cache()

        # Step 2: Causal mask (only for full attention layers)
        if cache is not None and start_pos > 0:
            mask = None
        else:
            total_len = start_pos + seq_len
            mask = torch.full(
                (seq_len, total_len), float("-inf"), device=self.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1)
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Step 3: RoPE frequencies
        freqs = self.freqs_cis[start_pos: start_pos + seq_len]
        ssm_state = cache.get_ssm_state_dict() if cache is not None else {}

        # Step 4: Transformer layers (two-phase per layer)
        for layer_idx in range(self.weights.num_layers):
            layer_type = self.layer_types[layer_idx]

            # Phase 1: Load attention + norms + router + shared expert (~100 MB)
            layer_tensors, layer_bufs = self.weights.load_layer_partial(
                layer_idx, exclude_prefixes=("mlp.switch_mlp.",)
            )

            # Attention with residual
            h = qwen_moe_attention(
                h, layer_tensors,
                layer_type=layer_type,
                freqs_cis=freqs,
                mask=mask if layer_type == "full_attention" else None,
                config=config,
                kv_cache=cache,
                ssm_state=ssm_state,
                layer_idx=layer_idx,
                scratch=scratch,
            )

            # Routing decision
            rms_norm_eps = config["rms_norm_eps"]
            h_normed = rms_norm(
                h, layer_tensors["post_attention_layernorm.weight"], rms_norm_eps
            )
            top_k_indices, top_k_weights, unique_experts = moe_route(
                h_normed, layer_tensors,
                num_experts_per_token=config["num_experts_per_token"],
                scratch=scratch,
            )

            # Phase 2: Load only selected experts (~65 MB for 10 experts)
            expert_tensors, expert_bufs = self.weights.load_experts(
                layer_idx, unique_experts
            )

            # MoE MLP with selective experts (adds residual)
            mlp_out = moe_mlp_selective(
                h_normed, layer_tensors, expert_tensors,
                top_k_indices, top_k_weights,
                num_experts_per_token=config["num_experts_per_token"],
                scratch=scratch,
            )
            # Clamp after residual to prevent fp16 overflow (-inf)
            h = (h + mlp_out).clamp(min=-65504.0, max=65504.0)
            torch.cuda.synchronize()

            del layer_tensors, layer_bufs, expert_tensors, expert_bufs

        # Advance cache
        if cache is not None:
            cache.advance(seq_len)

        # Step 5: Final norm
        norm_tensors, norm_buf = self.weights.load_special("final_norm")
        h = rms_norm(h, norm_tensors["model.norm.weight"], config["rms_norm_eps"])
        torch.cuda.synchronize()
        del norm_tensors, norm_buf
        torch.cuda.empty_cache()

        # Step 6: LM head
        if self.weights.special["lm_head"] is not None:
            lm_tensors, lm_buf = self.weights.load_special("lm_head")
            logits = _lm_head_proj(h, lm_tensors, "lm_head.weight")
            torch.cuda.synchronize()
            del lm_tensors, lm_buf
        else:
            embed_tensors, embed_buf = self.weights.load_special("embed_tokens")
            logits = _lm_head_proj(h, embed_tensors, "model.embed_tokens.weight")
            torch.cuda.synchronize()
            del embed_tensors, embed_buf
        torch.cuda.empty_cache()

        return logits


class QwenMoECachedScheduler:
    """Qwen3.5-MoE with adaptive VRAM caching + two-phase expert loading.

    Caches the partial layer data (attention+router+shared) across tokens.
    Expert loading is always on-demand since it depends on routing decisions.
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda:0",
        preload: bool = True,
        max_seq_len: Optional[int] = None,
    ):
        self.weights = ModelWeights(model_dir)
        self.device = torch.device(device)
        self.config = self.weights.config
        self._preload = preload
        self._max_seq_len = max_seq_len or min(
            self.config.get("max_seq_len", 32768), 32768
        )
        self.layer_cache: Optional[LayerCache] = None
        self._scratch = None

        self.layer_types = self.config.get("layer_types", [])
        if not self.layer_types:
            num_layers = self.config["num_layers"]
            self.layer_types = [
                "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                for i in range(num_layers)
            ]

        self.freqs_cis = precompute_freqs_cis(
            head_dim=self.config["head_dim"],
            seq_len=self._max_seq_len,
            theta=self.config.get("rope_theta", 10000000.0),
            device=torch.device(device),
            partial_rotary_factor=self.config.get("partial_rotary_factor", 0.25),
        )

    def __enter__(self):
        self.weights.init_gds()
        self._scratch = self._alloc_scratch()
        # Note: LayerCache uses load_layer() which loads full layers (3GB+).
        # For MoE, we skip LayerCache and use load_layer_partial() directly.
        # Specials (embed_tokens, final_norm, lm_head) are still cached.
        self.layer_cache = LayerCache(self.weights)
        if self._preload:
            # Only preload specials, not layers (too large for MoE)
            for name in ("embed_tokens", "final_norm", "lm_head"):
                if self.weights.special.get(name) is not None:
                    self.layer_cache.get_special(name, min_free=512 * 1024 * 1024)
        return self

    def __exit__(self, *args):
        if self.layer_cache:
            self.layer_cache.evict_all()
        self.weights.shutdown_gds()

    def _alloc_scratch(self):
        """Allocate scratch sized for single expert, not all 512."""
        max_elements = 0
        for layer in self.weights.layers:
            for t in layer.tensors:
                if t.quant == "affine_q4":
                    elems = 1
                    for s in t.shape:
                        elems *= s
                    if len(t.shape) == 3:
                        elems = elems // t.shape[0]
                    max_elements = max(max_elements, elems)
        if max_elements == 0:
            return None
        return torch.empty(max_elements, dtype=torch.float16, device=self.device)

    def create_cache(self, max_seq_len: int = 4096) -> HybridCache:
        config = self.config
        linear_key_heads = config.get("linear_num_key_heads", 16)
        linear_key_dim = config.get("linear_key_head_dim", 128)
        linear_value_heads = config.get("linear_num_value_heads", 64)
        linear_value_dim = config.get("linear_value_head_dim", 128)
        linear_hidden_dim = linear_key_heads * linear_key_dim
        linear_state_dim = linear_hidden_dim
        linear_bc_dim = linear_value_heads * linear_value_dim // linear_hidden_dim

        return HybridCache(
            layer_types=self.layer_types,
            num_kv_heads=config["num_kv_heads"],
            head_dim=config["head_dim"],
            max_seq_len=max_seq_len,
            linear_hidden_dim=linear_hidden_dim,
            linear_state_dim=linear_state_dim,
            linear_bc_dim=max(linear_bc_dim, 1),
            conv_kernel_size=config.get("linear_conv_kernel_dim", 4),
            device=self.device,
        )

    @torch.inference_mode()
    def forward(
        self,
        token_ids: torch.Tensor,
        cache: Optional[HybridCache] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        batch, seq_len = token_ids.shape
        assert batch == 1

        config = self.config
        scratch = self._scratch

        # Embeddings (cached)
        embed_tensors = self.layer_cache.get_special("embed_tokens")
        h = _embed_tokens(token_ids.to(self.device), embed_tensors)

        # Causal mask
        if cache is not None and start_pos > 0:
            mask = None
        else:
            total_len = start_pos + seq_len
            mask = torch.full(
                (seq_len, total_len), float("-inf"), device=self.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1)
            mask = mask.unsqueeze(0).unsqueeze(0)

        freqs = self.freqs_cis[start_pos: start_pos + seq_len]
        ssm_state = cache.get_ssm_state_dict() if cache is not None else {}

        # Use scratch only for decode (seq_len=1) where fused GEMV doesn't need it.
        # During prefill, scratch=None to avoid buffer reuse issues.
        effective_scratch = scratch if seq_len == 1 else None

        # Transformer layers (two-phase per layer)
        for layer_idx in range(self.weights.num_layers):
            layer_type = self.layer_types[layer_idx]

            # Phase 1: Load partial layer (attention + norms + router + shared)
            layer_tensors, layer_bufs = self.weights.load_layer_partial(
                layer_idx, exclude_prefixes=("mlp.switch_mlp.",)
            )

            # Attention with residual
            h = qwen_moe_attention(
                h, layer_tensors,
                layer_type=layer_type,
                freqs_cis=freqs,
                mask=mask if layer_type == "full_attention" else None,
                config=config,
                kv_cache=cache,
                ssm_state=ssm_state,
                layer_idx=layer_idx,
                scratch=effective_scratch,
            )

            # Routing
            rms_norm_eps = config["rms_norm_eps"]
            h_normed = rms_norm(
                h, layer_tensors["post_attention_layernorm.weight"], rms_norm_eps
            )
            top_k_indices, top_k_weights, unique_experts = moe_route(
                h_normed, layer_tensors,
                num_experts_per_token=config["num_experts_per_token"],
                scratch=effective_scratch,
            )

            # Phase 2: Load selected experts
            expert_tensors, expert_bufs = self.weights.load_experts(
                layer_idx, unique_experts
            )

            # MoE MLP
            mlp_out = moe_mlp_selective(
                h_normed, layer_tensors, expert_tensors,
                top_k_indices, top_k_weights,
                num_experts_per_token=config["num_experts_per_token"],
                scratch=effective_scratch,
            )
            # Clamp after residual to prevent fp16 overflow (-inf)
            h = (h + mlp_out).clamp(min=-65504.0, max=65504.0)
            torch.cuda.synchronize()

            del layer_tensors, layer_bufs, expert_tensors, expert_bufs

        if cache is not None:
            cache.advance(seq_len)

        # Final norm (cached)
        norm_tensors = self.layer_cache.get_special("final_norm")
        h = rms_norm(h, norm_tensors["model.norm.weight"], config["rms_norm_eps"])

        # LM head (cached)
        if self.weights.special["lm_head"] is not None:
            lm_tensors = self.layer_cache.get_special("lm_head")
            logits = _lm_head_proj(h, lm_tensors, "lm_head.weight")
        else:
            embed_tensors = self.layer_cache.get_special("embed_tokens")
            logits = _lm_head_proj(h, embed_tensors, "model.embed_tokens.weight")

        return logits
