"""
GdsLLM — LLaMA Forward Pass

Manual implementation of the LLaMA architecture using functional PyTorch
operations. Operates on externally-provided weight tensors (no nn.Module).

Supports both LLaMA-1 and LLaMA-2 (standard MHA and GQA).
"""

import math

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """RMSNorm as used in LLaMA."""
    input_dtype = x.dtype
    x = x.float()
    norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * norm).to(input_dtype) * weight


def _apply_llama3_rope_scaling(
    freqs: torch.Tensor, rope_scaling: dict,
) -> torch.Tensor:
    """Apply Llama 3.x frequency-dependent RoPE scaling.

    Low-frequency components (long wavelengths) get scaled down by `factor`,
    high-frequency components (short wavelengths) are kept unchanged, and
    mid-range frequencies are smoothly interpolated between the two.
    """
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
    high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    new_freqs = []
    for freq in freqs.tolist():
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)  # high freq: no scaling
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / factor)  # low freq: full scaling
        else:
            # smooth interpolation
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    head_dim: int, seq_len: int, theta: float = 10000.0,
    device: torch.device = None, rope_scaling: Optional[dict] = None,
) -> torch.Tensor:
    """Precompute the RoPE complex frequency tensor.

    Supports standard RoPE and Llama 3.x frequency-dependent scaling.
    """
    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    if rope_scaling is not None:
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", ""))
        if rope_type in ("llama3", "llama3.1"):
            freqs = _apply_llama3_rope_scaling(freqs, rope_scaling)
    t = torch.arange(seq_len, device=device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors.

    Args:
        xq: (batch, seq_len, num_heads, head_dim)
        xk: (batch, seq_len, num_kv_heads, head_dim)
        freqs_cis: (seq_len, head_dim // 2) complex
    """
    # Reshape last dim to pairs for complex view
    xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)

    xq_c = torch.view_as_complex(xq_r)
    xk_c = torch.view_as_complex(xk_r)

    # freqs_cis: (seq_len, head_dim//2) -> (1, seq_len, 1, head_dim//2)
    freqs = freqs_cis.unsqueeze(0).unsqueeze(2)

    xq_out = torch.view_as_real(xq_c * freqs).flatten(-2)
    xk_out = torch.view_as_real(xk_c * freqs).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def _dequant_gguf(meta: dict, scratch: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Dequantize a GGUF block-quantized tensor on GPU.

    Args:
        meta: dict with keys 'buffer', 'shape', 'gguf_type', 'offset'
        scratch: optional pre-allocated fp16 buffer to write into
    Returns:
        fp16 tensor (view into scratch if provided)
    """
    from gdsllm.runtime import gds_io_ext
    return gds_io_ext.dequant_gguf(
        meta["buffer"], meta["shape"], meta["gguf_type"], meta["offset"],
        scratch,
    )


def dequant_linear(
    x: torch.Tensor, weights: Dict[str, torch.Tensor], name: str,
    scratch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """F.linear with on-the-fly dequantization (INT8, GGUF Q4_0/Q8_0).

    If the weight is a GGUF dict, dequantizes block data via CUDA kernel.
    If the weight is int8, dequantizes using the per-channel scale before matmul.
    If the weight is fp16, behaves like a normal F.linear.
    """
    w = weights[f"{name}.weight"]
    if isinstance(w, dict) and "gguf_type" in w:
        w = _dequant_gguf(w, scratch)
    elif w.dtype == torch.int8:
        scale = weights[f"{name}.scale"]
        w = w.float() * scale  # broadcast [out, in] * [1, in]
        w = w.half()
    return F.linear(x, w)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match the number of query heads (for GQA).

    Args:
        x: (batch, num_kv_heads, seq_len, head_dim)
        n_rep: number of times to repeat
    """
    if n_rep == 1:
        return x
    batch, n_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)


def attention(
    x: torch.Tensor,
    weights: Dict[str, torch.Tensor],
    freqs_cis: torch.Tensor,
    mask: Optional[torch.Tensor],
    num_heads: int,
    num_kv_heads: int,
    kv_cache=None,
    layer_idx: int = 0,
    scratch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Multi-head self-attention (supports GQA and optional KV cache).

    Args:
        x: (batch, seq_len, hidden_size)
        weights: dict with q_proj.weight, k_proj.weight, v_proj.weight, o_proj.weight
        freqs_cis: (seq_len, head_dim // 2) complex
        mask: (1, 1, seq_len, total_len) or None
        num_heads: number of query heads
        num_kv_heads: number of KV heads
        kv_cache: optional KVCache instance for incremental decoding
        layer_idx: transformer layer index (used with kv_cache)
        scratch: optional pre-allocated fp16 buffer for dequantization
    """
    batch, seq_len, hidden_size = x.shape
    head_dim = hidden_size // num_heads
    n_rep = num_heads // num_kv_heads

    # Project Q, K, V
    xq = dequant_linear(x, weights, "self_attn.q_proj", scratch)
    xk = dequant_linear(x, weights, "self_attn.k_proj", scratch)
    xv = dequant_linear(x, weights, "self_attn.v_proj", scratch)

    # Reshape for multi-head: (batch, seq, num_heads, head_dim)
    xq = xq.view(batch, seq_len, num_heads, head_dim)
    xk = xk.view(batch, seq_len, num_kv_heads, head_dim)
    xv = xv.view(batch, seq_len, num_kv_heads, head_dim)

    # Apply RoPE
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

    # Transpose to (batch, heads, seq, head_dim)
    xq = xq.transpose(1, 2)
    xk = xk.transpose(1, 2)
    xv = xv.transpose(1, 2)

    # KV cache: store current K/V and retrieve full history
    if kv_cache is not None:
        xk, xv = kv_cache.update(layer_idx, xk, xv)

    # Repeat KV for GQA
    xk = repeat_kv(xk, n_rep)
    xv = repeat_kv(xv, n_rep)

    # Scaled dot-product attention
    scores = torch.matmul(xq, xk.transpose(-2, -1)) / (head_dim ** 0.5)
    if mask is not None:
        scores = scores + mask
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    output = torch.matmul(scores, xv)

    # Reshape and project output
    output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
    return dequant_linear(output, weights, "self_attn.o_proj", scratch)


def mlp(
    x: torch.Tensor, weights: Dict[str, torch.Tensor],
    scratch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """SwiGLU MLP as used in LLaMA."""
    gate = dequant_linear(x, weights, "mlp.gate_proj", scratch)
    up = dequant_linear(x, weights, "mlp.up_proj", scratch)
    return dequant_linear(F.silu(gate) * up, weights, "mlp.down_proj", scratch)


def transformer_block(
    x: torch.Tensor,
    weights: Dict[str, torch.Tensor],
    freqs_cis: torch.Tensor,
    mask: Optional[torch.Tensor],
    num_heads: int,
    num_kv_heads: int,
    rms_norm_eps: float = 1e-5,
    kv_cache=None,
    layer_idx: int = 0,
    scratch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One transformer layer (pre-norm architecture)."""
    # Self-attention with residual
    h = rms_norm(x, weights["input_layernorm.weight"], rms_norm_eps)
    h = attention(
        h, weights, freqs_cis, mask, num_heads, num_kv_heads,
        kv_cache=kv_cache, layer_idx=layer_idx, scratch=scratch,
    )
    x = x + h

    # MLP with residual
    h = rms_norm(x, weights["post_attention_layernorm.weight"], rms_norm_eps)
    h = mlp(h, weights, scratch)
    x = x + h

    return x
