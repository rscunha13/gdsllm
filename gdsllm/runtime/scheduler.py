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
    dequant_linear,
    _dequant_gguf,
)


def _embed_tokens(token_ids, tensors, scratch=None):
    """Embedding lookup with INT8 and GGUF dequantization support.

    For INT8, only dequantizes the rows needed (avoids 4 GB float32 temp
    for full vocab embedding table).
    For GGUF block-quantized, dequants the full table then does lookup
    (block format can't do per-row extraction).
    """
    w = tensors["model.embed_tokens.weight"]
    if isinstance(w, dict) and "gguf_type" in w:
        # GGUF: must dequant full table, then lookup
        w_fp16 = _dequant_gguf(w, scratch)
        h = F.embedding(token_ids, w_fp16)
        del w_fp16
    elif w.dtype == torch.int8:
        scale = tensors["model.embed_tokens.scale"]  # [1, hidden]
        # Only extract and dequantize the needed rows
        rows_int8 = F.embedding(token_ids, w)  # [B, S, hidden] int8
        h = (rows_int8.float() * scale).half()  # dequant only selected rows
    else:
        h = F.embedding(token_ids, w)
    return h


def _lm_head_proj(h, tensors, key="lm_head.weight", scratch=None):
    """LM head projection with INT8 and GGUF dequantization support.

    For large vocab models (70B+), dequantizes in chunks to avoid
    allocating a massive float32 temp (e.g. 128256 * 8192 * 4 = 4 GB).
    For GGUF block-quantized, dequants via CUDA kernel to fp16.
    """
    w = tensors[key]
    if isinstance(w, dict) and "gguf_type" in w:
        w = _dequant_gguf(w, scratch)
    elif w.dtype == torch.int8:
        scale_key = key.rsplit(".weight", 1)[0] + ".scale"
        scale = tensors[scale_key]
        # Chunked dequant for large matrices (>512 MB float32 temp)
        if w.shape[0] * w.shape[1] * 4 > 512 * 1024 * 1024:
            chunk_size = 16384
            chunks = []
            for i in range(0, w.shape[0], chunk_size):
                w_chunk = (w[i:i + chunk_size].float() * scale).half()
                chunks.append(F.linear(h, w_chunk))
                del w_chunk
            return torch.cat(chunks, dim=-1)
        w = (w.float() * scale).half()
    return F.linear(h, w)


def _alloc_dequant_scratch(weights: ModelWeights, device: torch.device):
    """Allocate a reusable fp16 scratch buffer for GGUF dequantization.

    Sized for the largest block-quantized weight across transformer layers
    (not specials — embed/lm_head are called once per pass, not worth the
    VRAM cost of a 2 GB scratch for vocab-sized weights).
    Returns None if the model has no GGUF-quantized layer weights.
    """
    max_elements = 0
    for layer in weights.layers:
        for t in layer.tensors:
            if t.quant in ("q4_0", "q8_0"):
                elems = 1
                for s in t.shape:
                    elems *= s
                max_elements = max(max_elements, elems)
    if max_elements == 0:
        return None
    return torch.empty(max_elements, dtype=torch.float16, device=device)


class SimpleScheduler:
    """Layer-by-layer LLaMA inference with GDS weight streaming."""

    def __init__(self, model_dir: str, device: str = "cuda:0"):
        self.weights = ModelWeights(model_dir)
        self.device = torch.device(device)
        self.config = self.weights.config
        self._scratch = None

        # Precompute RoPE frequencies (small, lives in VRAM permanently)
        self.freqs_cis = precompute_freqs_cis(
            head_dim=self.config["head_dim"],
            seq_len=self.config["max_seq_len"],
            theta=self.config.get("rope_theta", 10000.0),
            device=self.device,
            rope_scaling=self.config.get("rope_scaling"),
        )

    def __enter__(self):
        self.weights.init_gds()
        self._scratch = _alloc_dequant_scratch(self.weights, self.device)
        return self

    def __exit__(self, *args):
        self.weights.shutdown_gds()

    @torch.inference_mode()
    def forward(
        self,
        token_ids: torch.Tensor,
        kv_cache=None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Full forward pass through the LLaMA model.

        Args:
            token_ids: (1, seq_len) long tensor
            kv_cache: optional KVCache for incremental decoding
            start_pos: position offset for RoPE (0 for prefill, seq_pos for decode)

        Returns:
            logits: (1, seq_len, vocab_size) float tensor on CUDA
        """
        batch, seq_len = token_ids.shape
        assert batch == 1, "MVP supports batch=1 only"
        assert start_pos + seq_len <= self.config["max_seq_len"], (
            f"Position {start_pos + seq_len} exceeds max {self.config['max_seq_len']}"
        )

        num_heads = self.config["num_heads"]
        num_kv_heads = self.config["num_kv_heads"]
        rms_norm_eps = self.config["rms_norm_eps"]

        scratch = self._scratch

        # Step 1: Token embeddings (no scratch — vocab weights too large)
        embed_tensors, embed_buf = self.weights.load_special("embed_tokens")
        h = _embed_tokens(token_ids.to(self.device), embed_tensors)
        del embed_tensors, embed_buf
        torch.cuda.empty_cache()

        # Step 2: Causal mask
        if kv_cache is not None and start_pos > 0:
            # Decode: single token attends to all cached positions — no mask needed
            mask = None
        else:
            # Prefill or no-cache: standard causal mask
            total_len = start_pos + seq_len
            mask = torch.full(
                (seq_len, total_len), float("-inf"), device=self.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1)
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Step 3: RoPE frequencies at correct positions
        freqs = self.freqs_cis[start_pos : start_pos + seq_len]

        # Step 4: Transformer layers (scratch reused across all dequant calls)
        for layer_idx in range(self.weights.num_layers):
            layer_tensors, layer_buf = self.weights.load_layer(layer_idx)
            h = transformer_block(
                h, layer_tensors, freqs, mask,
                num_heads, num_kv_heads, rms_norm_eps,
                kv_cache=kv_cache, layer_idx=layer_idx,
                scratch=scratch,
            )
            del layer_tensors, layer_buf
            torch.cuda.empty_cache()

        # Advance KV cache after all layers
        if kv_cache is not None:
            kv_cache.advance(seq_len)

        # Step 5: Final norm
        norm_tensors, norm_buf = self.weights.load_special("final_norm")
        h = rms_norm(h, norm_tensors["model.norm.weight"], rms_norm_eps)
        del norm_tensors, norm_buf
        torch.cuda.empty_cache()

        # Step 6: LM head (no scratch — vocab weights too large)
        if self.weights.special["lm_head"] is not None:
            lm_tensors, lm_buf = self.weights.load_special("lm_head")
            logits = _lm_head_proj(h, lm_tensors, "lm_head.weight")
            del lm_tensors, lm_buf
        else:
            embed_tensors, embed_buf = self.weights.load_special("embed_tokens")
            logits = _lm_head_proj(h, embed_tensors, "model.embed_tokens.weight")
            del embed_tensors, embed_buf
        torch.cuda.empty_cache()

        return logits


def estimate_activation_vram(
    config: dict, max_seq_len: int, use_kv_cache: bool = False,
) -> int:
    """Estimate peak VRAM needed for activations at a given sequence length.

    Without KV cache the peak occurs during attention softmax, which holds
    multiple copies of the (num_heads, S, S) score matrix simultaneously.
    With KV cache in decode mode, scores are (num_heads, 1, S) — much smaller.

    For INT8 quantized models, dequantization creates a temporary float32 copy
    of each weight matrix. The largest single dequant is the lm_head
    (vocab_size * hidden_size * 4 bytes).

    Returns:
        Estimated bytes needed.
    """
    S = max_seq_len
    num_heads = config["num_heads"]
    head_dim = config["head_dim"]
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    vocab_size = config.get("vocab_size", 32000)

    if use_kv_cache:
        # Decode: scores shape is (num_heads, 1, S), not (num_heads, S, S)
        attn_peak = num_heads * S * 4 * 3
    else:
        attn_peak = num_heads * S * S * 4 * 3

    qkv_bytes = 3 * num_heads * S * head_dim * 2
    mlp_bytes = 2 * hidden_size * intermediate_size * 2

    # Dequantization scratch: largest weight dequantized during forward pass
    # lm_head [vocab_size, hidden_size] or mlp projections [intermediate, hidden]
    quantization = config.get("quantization", "none")
    if quantization == "int8":
        # float32 temp + half copy of largest weight (lm_head or embed)
        largest_weight = max(vocab_size * hidden_size, intermediate_size * hidden_size)
        dequant_scratch = largest_weight * (4 + 2)  # float32 + half
    elif quantization in ("q4_0", "q8_0"):
        # GGUF: layer dequant uses pre-allocated scratch (no extra cost),
        # but embed_tokens / lm_head still allocate one-shot fp16 output
        dequant_scratch = vocab_size * hidden_size * 2  # fp16 embed/lm_head
    else:
        dequant_scratch = 0

    fragmentation = 512 * 1024 * 1024
    return int(attn_peak + qkv_bytes + mlp_bytes + dequant_scratch + fragmentation)


class CachedScheduler:
    """LLaMA inference with adaptive VRAM-cached layers.

    Loads layers via GDS on first access and keeps them resident in VRAM.
    Uses actual free VRAM to decide caching — adapts automatically as
    activation memory grows with sequence length.
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
        self._max_seq_len = max_seq_len or self.config["max_seq_len"]
        self.cache: Optional[LayerCache] = None
        self._scratch = None

        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(
            head_dim=self.config["head_dim"],
            seq_len=self.config["max_seq_len"],
            theta=self.config.get("rope_theta", 10000.0),
            device=self.device,
            rope_scaling=self.config.get("rope_scaling"),
        )

    def __enter__(self):
        self.weights.init_gds()
        self._scratch = _alloc_dequant_scratch(self.weights, self.device)
        self.cache = LayerCache(self.weights)
        if self._preload:
            # Reserve VRAM for activations at max_seq_len during preload
            min_free = estimate_activation_vram(self.config, self._max_seq_len)
            self.cache.preload(min_free=min_free)
        return self

    def __exit__(self, *args):
        if self.cache:
            self.cache.evict_all()
        self.weights.shutdown_gds()

    @torch.inference_mode()
    def forward(
        self,
        token_ids: torch.Tensor,
        kv_cache=None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Full forward pass using cached layers.

        Args:
            token_ids: (1, seq_len) for prefill or (1, 1) for decode
            kv_cache: optional KVCache for incremental decoding
            start_pos: position offset for RoPE
        """
        batch, seq_len = token_ids.shape
        assert batch == 1, "MVP supports batch=1 only"
        assert start_pos + seq_len <= self.config["max_seq_len"]

        num_heads = self.config["num_heads"]
        num_kv_heads = self.config["num_kv_heads"]
        rms_norm_eps = self.config["rms_norm_eps"]

        scratch = self._scratch

        # Step 1: Token embeddings (no scratch — vocab weights too large)
        embed_tensors = self.cache.get_special("embed_tokens")
        h = _embed_tokens(token_ids.to(self.device), embed_tensors)

        # Step 2: Causal mask
        if kv_cache is not None and start_pos > 0:
            mask = None
        else:
            total_len = start_pos + seq_len
            mask = torch.full(
                (seq_len, total_len), float("-inf"), device=self.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1)
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Step 3: RoPE frequencies at correct positions
        freqs = self.freqs_cis[start_pos : start_pos + seq_len]

        # Step 4: Compute activation VRAM needed.
        # With KV cache in decode mode, activation peak is O(S) not O(S²).
        effective_seq = start_pos + seq_len
        use_kv = kv_cache is not None and start_pos > 0
        activation_need = estimate_activation_vram(
            self.config, effective_seq, use_kv_cache=use_kv,
        )
        layer_buf = max(l.size_bytes for l in self.weights.layers)
        self.cache.ensure_free_vram(activation_need + layer_buf)

        # Step 5: Transformer layers (scratch reused across all dequant calls)
        for layer_idx in range(self.weights.num_layers):
            layer_tensors = self.cache.get_layer(
                layer_idx, min_free=activation_need
            )
            h = transformer_block(
                h, layer_tensors, freqs, mask,
                num_heads, num_kv_heads, rms_norm_eps,
                kv_cache=kv_cache, layer_idx=layer_idx,
                scratch=scratch,
            )

        # Advance KV cache after all layers
        if kv_cache is not None:
            kv_cache.advance(seq_len)

        # Free temp handle from last non-cached layer before norm/lm_head
        self.cache.release_temp()

        # Step 6: Final norm
        norm_tensors = self.cache.get_special("final_norm")
        h = rms_norm(h, norm_tensors["model.norm.weight"], rms_norm_eps)

        # Step 7: LM head (no scratch — vocab weights too large)
        if self.weights.special["lm_head"] is not None:
            lm_tensors = self.cache.get_special("lm_head")
            logits = _lm_head_proj(h, lm_tensors, "lm_head.weight")
        else:
            embed_tensors = self.cache.get_special("embed_tokens")
            logits = _lm_head_proj(h, embed_tensors, "model.embed_tokens.weight")

        return logits
