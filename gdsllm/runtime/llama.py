"""
GdsLLM — LLaMA Forward Pass

Manual implementation of the LLaMA architecture using functional PyTorch
operations. Operates on externally-provided weight tensors (no nn.Module).

Supports both LLaMA-1 and LLaMA-2 (standard MHA and GQA).
"""

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


def precompute_freqs_cis(
    head_dim: int, seq_len: int, theta: float = 10000.0,
    device: torch.device = None,
) -> torch.Tensor:
    """Precompute the RoPE complex frequency tensor."""
    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
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
) -> torch.Tensor:
    """Multi-head self-attention (supports GQA).

    Args:
        x: (batch, seq_len, hidden_size)
        weights: dict with q_proj.weight, k_proj.weight, v_proj.weight, o_proj.weight
        freqs_cis: (seq_len, head_dim // 2) complex
        mask: (1, 1, seq_len, seq_len) or None
        num_heads: number of query heads
        num_kv_heads: number of KV heads
    """
    batch, seq_len, hidden_size = x.shape
    head_dim = hidden_size // num_heads
    n_rep = num_heads // num_kv_heads

    # Project Q, K, V
    xq = F.linear(x, weights["self_attn.q_proj.weight"])
    xk = F.linear(x, weights["self_attn.k_proj.weight"])
    xv = F.linear(x, weights["self_attn.v_proj.weight"])

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
    return F.linear(output, weights["self_attn.o_proj.weight"])


def mlp(x: torch.Tensor, weights: Dict[str, torch.Tensor]) -> torch.Tensor:
    """SwiGLU MLP as used in LLaMA."""
    gate = F.linear(x, weights["mlp.gate_proj.weight"])
    up = F.linear(x, weights["mlp.up_proj.weight"])
    return F.linear(F.silu(gate) * up, weights["mlp.down_proj.weight"])


def transformer_block(
    x: torch.Tensor,
    weights: Dict[str, torch.Tensor],
    freqs_cis: torch.Tensor,
    mask: Optional[torch.Tensor],
    num_heads: int,
    num_kv_heads: int,
    rms_norm_eps: float = 1e-5,
) -> torch.Tensor:
    """One transformer layer (pre-norm architecture)."""
    # Self-attention with residual
    h = rms_norm(x, weights["input_layernorm.weight"], rms_norm_eps)
    h = attention(h, weights, freqs_cis, mask, num_heads, num_kv_heads)
    x = x + h

    # MLP with residual
    h = rms_norm(x, weights["post_attention_layernorm.weight"], rms_norm_eps)
    h = mlp(h, weights)
    x = x + h

    return x
