"""
GdsLLM — Qwen3.5-MoE Forward Pass

Hybrid architecture: full attention (15 layers) + linear attention / Mamba SSM
(45 layers), with MoE MLP (512 experts + shared expert) on every layer.

All weights are affine Q4 quantized: W_fp = nibble * scale + bias.
Functional implementation (no nn.Module) for GDS weight streaming.
"""

import math

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ─── Affine Q4 dequant helpers ───────────────────────────────────────────

def _dequant_affine_q4(meta: dict, scratch: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Dequantize an affine Q4 tensor on GPU.

    Args:
        meta: dict with keys 'buffer', 'shape', 'quant_type', 'weight_offset',
              'scale_offset', 'bias_offset', 'group_size'
        scratch: optional pre-allocated fp16 buffer
    Returns:
        fp16 tensor
    """
    from gdsllm.runtime import gds_io_ext
    return gds_io_ext.dequant_affine_q4(
        meta["buffer"], meta["shape"],
        meta["weight_offset"], meta["scale_offset"], meta["bias_offset"],
        meta["group_size"], scratch,
    )


def _fused_affine_q4_gemv(x: torch.Tensor, meta: dict) -> torch.Tensor:
    """Fused affine Q4 dequant + GEMV for decode (single token).

    Args:
        x: (batch, 1, in_dim) fp16 input
        meta: affine_q4 metadata dict
    Returns:
        (batch, 1, out_dim) fp16 output
    """
    from gdsllm.runtime import gds_io_ext
    out = gds_io_ext.fused_affine_q4_gemv(
        meta["buffer"], x.view(-1),
        meta["shape"],
        meta["weight_offset"], meta["scale_offset"], meta["bias_offset"],
        meta["group_size"],
    )
    # out is [1, out_dim] -> [1, 1, out_dim]
    return out.unsqueeze(0)


def affine_q4_linear(
    x: torch.Tensor, weights: Dict, name: str,
    scratch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """F.linear with affine Q4 dequantization.

    For decode (seq_len=1), uses fused dequant+GEMV kernel.
    For prefill (seq_len>1), dequants to fp16 then uses cuBLAS GEMM.
    For non-quantized fp16 weights, uses standard F.linear.
    """
    w = weights[f"{name}.weight"]
    if isinstance(w, dict) and w.get("quant_type") == "affine_q4":
        if x.shape[1] == 1:
            return _fused_affine_q4_gemv(x, w)
        w = _dequant_affine_q4(w, scratch)
    return F.linear(x, w)


# ─── Shared ops ──────────────────────────────────────────────────────────

def rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    input_dtype = x.dtype
    x = x.float()
    norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * norm).to(input_dtype) * weight


def precompute_freqs_cis(
    head_dim: int, seq_len: int, theta: float = 10000000.0,
    device: torch.device = None, partial_rotary_factor: float = 1.0,
) -> torch.Tensor:
    """Precompute RoPE frequencies for Qwen3.5-MoE.

    With partial_rotary_factor < 1.0, only part of the head dimensions
    get RoPE. Returns complex tensor of shape (seq_len, rot_dim//2).
    """
    rot_dim = int(head_dim * partial_rotary_factor)
    freqs = 1.0 / (
        theta ** (torch.arange(0, rot_dim, 2, device=device).float() / rot_dim)
    )
    t = torch.arange(seq_len, device=device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_partial_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor,
    head_dim: int, partial_rotary_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to partial dimensions of Q and K.

    Args:
        xq: (batch, seq_len, num_heads, head_dim)
        xk: (batch, seq_len, num_kv_heads, head_dim)
        freqs_cis: (seq_len, rot_dim//2) complex
    """
    rot_dim = int(head_dim * partial_rotary_factor)

    if rot_dim == head_dim:
        # Full rotary — same as standard RoPE
        xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)
        xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)
        xq_c = torch.view_as_complex(xq_r)
        xk_c = torch.view_as_complex(xk_r)
        freqs = freqs_cis.unsqueeze(0).unsqueeze(2)
        xq_out = torch.view_as_real(xq_c * freqs).flatten(-2)
        xk_out = torch.view_as_real(xk_c * freqs).flatten(-2)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    # Partial rotary: split into rotary and pass-through parts
    xq_rot, xq_pass = xq[..., :rot_dim], xq[..., rot_dim:]
    xk_rot, xk_pass = xk[..., :rot_dim], xk[..., rot_dim:]

    xq_r = xq_rot.float().reshape(*xq_rot.shape[:-1], -1, 2)
    xk_r = xk_rot.float().reshape(*xk_rot.shape[:-1], -1, 2)
    xq_c = torch.view_as_complex(xq_r)
    xk_c = torch.view_as_complex(xk_r)
    freqs = freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_rot = torch.view_as_real(xq_c * freqs).flatten(-2).type_as(xq)
    xk_rot = torch.view_as_real(xk_c * freqs).flatten(-2).type_as(xk)

    return (
        torch.cat([xq_rot, xq_pass], dim=-1),
        torch.cat([xk_rot, xk_pass], dim=-1),
    )


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    batch, n_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)


# ─── Full Attention (15 layers) ──────────────────────────────────────────

def full_attention(
    x: torch.Tensor,
    weights: Dict,
    freqs_cis: torch.Tensor,
    mask: Optional[torch.Tensor],
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    partial_rotary_factor: float,
    rms_norm_eps: float,
    kv_cache=None,
    layer_idx: int = 0,
    scratch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Full attention with QK-norm, partial RoPE, and output gate (Qwen3.5-MoE).

    q_proj outputs num_heads * head_dim * 2: first half is Q, second half is gate.
    attn_output = attn_output * sigmoid(gate)
    """
    batch, seq_len, _ = x.shape
    n_rep = num_heads // num_kv_heads

    xq_gate = affine_q4_linear(x, weights, "self_attn.q_proj", scratch)
    xk = affine_q4_linear(x, weights, "self_attn.k_proj", scratch)
    xv = affine_q4_linear(x, weights, "self_attn.v_proj", scratch)

    # Split Q projection into query and gate: each (batch, seq, num_heads, head_dim)
    xq_gate = xq_gate.view(batch, seq_len, num_heads, head_dim * 2)
    xq, gate = xq_gate.chunk(2, dim=-1)
    xk = xk.view(batch, seq_len, num_kv_heads, head_dim)
    xv = xv.view(batch, seq_len, num_kv_heads, head_dim)

    # QK-norm (RMSNorm per head)
    xq = rms_norm(xq, weights["self_attn.q_norm.weight"], rms_norm_eps)
    xk = rms_norm(xk, weights["self_attn.k_norm.weight"], rms_norm_eps)

    # Partial RoPE
    xq, xk = apply_partial_rotary_emb(
        xq, xk, freqs_cis, head_dim, partial_rotary_factor
    )

    # Transpose to (batch, heads, seq, head_dim)
    xq = xq.transpose(1, 2)
    xk = xk.transpose(1, 2)
    xv = xv.transpose(1, 2)

    # KV cache
    if kv_cache is not None:
        xk, xv = kv_cache.update(layer_idx, xk, xv)

    # GQA
    xk = repeat_kv(xk, n_rep)
    xv = repeat_kv(xv, n_rep)

    # Scaled dot-product attention
    scores = torch.matmul(xq, xk.transpose(-2, -1)) / (head_dim ** 0.5)
    if mask is not None:
        scores = scores + mask
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    output = torch.matmul(scores, xv)

    # Apply output gate: sigmoid(gate) * attn_output
    output = output.transpose(1, 2)  # (batch, seq, heads, head_dim)
    output = output * torch.sigmoid(gate)
    output = output.contiguous().view(batch, seq_len, -1)
    return affine_q4_linear(output, weights, "self_attn.o_proj", scratch)


# ─── Linear Attention / Mamba SSM (45 layers) ────────────────────────────

def _l2_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2 normalize along last dimension."""
    return x / (x.norm(dim=-1, keepdim=True).clamp(min=eps))


def linear_attention(
    x: torch.Tensor,
    weights: Dict,
    ssm_state: Optional[dict] = None,
    layer_idx: int = 0,
    scratch: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
) -> torch.Tensor:
    """Gated Delta Rule linear attention for Qwen3.5-MoE.

    Architecture:
    1. Project x → Q+K+V (conv1d + SiLU), Z (gate), a (decay), b (write gate)
    2. Gated delta rule recurrence with associative state S[64, 128, 128]
    3. Output = rms_norm(attn_out) * silu(z) → out_proj

    Config dims:
        num_key_heads=16, key_head_dim=128 → Q,K = 2048 each
        num_value_heads=64, value_head_dim=128 → V = 8192
        Q+K+V = 12288, Z = 8192
        a = 64 (decay per v_head), b = 64 (write gate per v_head)
    """
    batch, seq_len, hidden_size = x.shape

    num_k_heads = config.get("linear_num_key_heads", 16)
    k_head_dim = config.get("linear_key_head_dim", 128)
    num_v_heads = config.get("linear_num_value_heads", 64)
    v_head_dim = config.get("linear_value_head_dim", 128)
    heads_per_group = num_v_heads // num_k_heads  # 4

    # Step 1: Input projections
    qkv = affine_q4_linear(x, weights, "linear_attn.in_proj_qkv", scratch)
    z = affine_q4_linear(x, weights, "linear_attn.in_proj_z", scratch)
    a = affine_q4_linear(x, weights, "linear_attn.in_proj_a", scratch)
    b = affine_q4_linear(x, weights, "linear_attn.in_proj_b", scratch)

    # Step 2: Depthwise causal conv1d on QKV
    # conv_w stored as [12288, 4, 1] in Mamba convention, need [12288, 1, 4]
    conv_w = weights["linear_attn.conv1d.weight"]
    if isinstance(conv_w, dict):
        conv_w = _dequant_affine_q4(conv_w, scratch)

    conv_w = conv_w.reshape(conv_w.shape[0], conv_w.shape[1], conv_w.shape[2])
    # Transpose from [channels, kernel_size, 1] to [channels, 1, kernel_size]
    if conv_w.shape[2] == 1 and conv_w.shape[1] > 1:
        conv_w = conv_w.transpose(1, 2)  # [12288, 1, 4]

    d_conv = qkv.shape[-1]  # 12288
    conv_kernel_size = conv_w.shape[-1]  # 4
    conv_input = qkv.transpose(1, 2)  # (batch, 12288, seq_len)

    # Handle conv state for incremental decode
    if ssm_state is not None and f"conv_{layer_idx}" in ssm_state:
        conv_state = ssm_state[f"conv_{layer_idx}"]
        conv_input = torch.cat([conv_state, conv_input], dim=-1)
    else:
        # First forward: left-pad with zeros to preserve sequence length
        conv_input = F.pad(conv_input, (conv_kernel_size - 1, 0))

    qkv_conv = F.conv1d(conv_input, conv_w, groups=d_conv, padding=0)
    if qkv_conv.shape[-1] > seq_len:
        qkv_conv = qkv_conv[..., -seq_len:]

    # Save conv state
    if ssm_state is not None:
        ssm_state[f"conv_{layer_idx}"] = conv_input[..., -(conv_kernel_size - 1):]

    qkv_conv = F.silu(qkv_conv.transpose(1, 2))  # (batch, seq_len, 12288)

    # Step 3: Split Q, K, V
    q_dim = num_k_heads * k_head_dim   # 2048
    k_dim = num_k_heads * k_head_dim   # 2048
    v_dim = num_v_heads * v_head_dim   # 8192

    q, k, v = qkv_conv.split([q_dim, k_dim, v_dim], dim=-1)
    q = q.view(batch, seq_len, num_k_heads, k_head_dim)
    k = k.view(batch, seq_len, num_k_heads, k_head_dim)
    v = v.view(batch, seq_len, num_v_heads, v_head_dim)

    # Step 4: Expand Q, K to match V heads (GQA-style: 16 → 64)
    q = q.repeat_interleave(heads_per_group, dim=2)  # (B, S, 64, 128)
    k = k.repeat_interleave(heads_per_group, dim=2)  # (B, S, 64, 128)

    # Step 5: Compute decay g and write gate beta
    A_log = weights["linear_attn.A_log"]
    if isinstance(A_log, dict):
        A_log = _dequant_affine_q4(A_log, scratch)
    dt_bias = weights["linear_attn.dt_bias"]
    if isinstance(dt_bias, dict):
        dt_bias = _dequant_affine_q4(dt_bias, scratch)

    # g = -exp(A_log) * softplus(a + dt_bias)  → per-step decay
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())
    # beta = sigmoid(b) → write gate
    beta = torch.sigmoid(b.float())

    # Step 6: Gated Delta Rule recurrence
    # State S: [batch, num_v_heads, k_head_dim, v_head_dim] = [B, 64, 128, 128]
    if ssm_state is not None and f"ssm_{layer_idx}" in ssm_state:
        S = ssm_state[f"ssm_{layer_idx}"]
    else:
        S = torch.zeros(batch, num_v_heads, k_head_dim, v_head_dim,
                        device=x.device, dtype=torch.float32)

    outputs = []
    for t in range(seq_len):
        q_t = _l2_norm(q[:, t].float())  # (B, 64, 128)
        k_t = _l2_norm(k[:, t].float())  # (B, 64, 128)
        v_t = v[:, t].float()             # (B, 64, 128)
        g_t = g[:, t]                     # (B, 64)
        beta_t = beta[:, t]               # (B, 64)

        # Scale query
        q_t = q_t * (k_head_dim ** -0.5)

        # Decay the state
        S = S * torch.exp(g_t).unsqueeze(-1).unsqueeze(-1)  # (B, 64, 128, 128)

        # Read from state: kv_mem = S @ k  → (B, 64, 128)
        kv_mem = torch.einsum("bhkv,bhk->bhv", S, k_t)

        # Delta update: error correction
        delta = (v_t - kv_mem) * beta_t.unsqueeze(-1)  # (B, 64, 128)

        # Write to state: S += outer(k, delta)
        S = S + torch.einsum("bhk,bhv->bhkv", k_t, delta)

        # Read output with query
        y_t = torch.einsum("bhkv,bhk->bhv", S, q_t)  # (B, 64, 128)
        outputs.append(y_t)

    if ssm_state is not None:
        ssm_state[f"ssm_{layer_idx}"] = S

    y = torch.stack(outputs, dim=1)  # (B, seq_len, 64, 128)

    # Step 7: Gated RMSNorm
    norm_w = weights["linear_attn.norm.weight"]
    if isinstance(norm_w, dict):
        norm_w = _dequant_affine_q4(norm_w, scratch)

    # Reshape z to match: (B, seq_len, 64, 128)
    z = z.view(batch, seq_len, num_v_heads, v_head_dim)

    # RMSNorm per-vector on last dim (128), then gate
    y = y.to(x.dtype)
    y = rms_norm(y, norm_w) * F.silu(z)

    # Step 8: Flatten and output projection
    y = y.reshape(batch, seq_len, v_dim)  # (B, seq_len, 8192)
    return affine_q4_linear(y, weights, "linear_attn.out_proj", scratch)


# ─── MoE MLP ────────────────────────────────────────────────────────────

def moe_route(
    x: torch.Tensor,
    weights: Dict,
    num_experts_per_token: int,
    scratch: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """Compute top-k expert routing.

    Args:
        x: (batch, seq_len, hidden_size) — post-norm hidden states
        weights: layer weight tensors (must contain "mlp.gate.weight")
        num_experts_per_token: top-k experts to select

    Returns:
        top_k_indices: (batch, seq_len, k)
        top_k_weights: (batch, seq_len, k)
        unique_experts: sorted list of unique expert indices
    """
    router_w = weights["mlp.gate.weight"]
    if isinstance(router_w, dict):
        router_w = _dequant_affine_q4(router_w, scratch)
    router_logits = F.linear(x, router_w)

    top_k_logits, top_k_indices = torch.topk(
        router_logits, num_experts_per_token, dim=-1
    )
    top_k_weights = F.softmax(top_k_logits.float(), dim=-1).type_as(x)

    unique_experts = top_k_indices.unique().tolist()

    return top_k_indices, top_k_weights, unique_experts


def moe_mlp_selective(
    x: torch.Tensor,
    weights: Dict,
    expert_tensors: Dict[str, dict],
    top_k_indices: torch.Tensor,
    top_k_weights: torch.Tensor,
    num_experts_per_token: int,
    scratch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mixture-of-Experts MLP with selectively loaded experts.

    Instead of loading all 512 experts, only the experts in expert_tensors
    are available. Each expert is an individual 2D affine_q4 tensor loaded
    from specific byte ranges of the layer file.

    Args:
        x: (batch, seq_len, hidden_size) — post-norm hidden states
        weights: layer weight tensors (shared expert + gate)
        expert_tensors: dict from load_experts(), maps "gate_proj.{id}" ->
                        affine_q4 metadata dict
        top_k_indices: (batch, seq_len, k) expert indices
        top_k_weights: (batch, seq_len, k) softmax routing weights
        num_experts_per_token: k
    """
    batch, seq_len, hidden_size = x.shape
    x_flat = x.view(-1, hidden_size)
    num_tokens = x_flat.shape[0]

    expert_output = torch.zeros(num_tokens, hidden_size,
                                device=x.device, dtype=x.dtype)

    indices_flat = top_k_indices.view(num_tokens, num_experts_per_token)
    weights_flat = top_k_weights.view(num_tokens, num_experts_per_token)

    # For decode (seq_len=1, 1 token), compute each selected expert individually
    # using fused GEMV. For prefill, dequant + matmul.
    is_decode = (num_tokens == 1)

    for k in range(num_experts_per_token):
        expert_ids = indices_flat[:, k]   # (num_tokens,)
        expert_w = weights_flat[:, k]     # (num_tokens,)

        # In decode mode, there's exactly 1 token so 1 expert per k-slot
        if is_decode:
            eidx = expert_ids[0].item()
            gate_meta = expert_tensors[f"gate_proj.{eidx}"]
            up_meta = expert_tensors[f"up_proj.{eidx}"]
            down_meta = expert_tensors[f"down_proj.{eidx}"]

            # Fused dequant+GEMV for each projection
            gate_out = _fused_affine_q4_gemv(x_flat, gate_meta)
            up_out = _fused_affine_q4_gemv(x_flat, up_meta)
            intermediate = F.silu(gate_out) * up_out
            down_out = _fused_affine_q4_gemv(intermediate, down_meta)

            expert_output += expert_w.unsqueeze(-1) * down_out.view(num_tokens, -1)
        else:
            # Prefill: group tokens by expert, dequant + matmul
            unique_ids = expert_ids.unique()
            for eid_t in unique_ids:
                eid = eid_t.item()
                token_mask = (expert_ids == eid)
                x_sel = x_flat[token_mask]  # (n_sel, hidden)

                gate_w = _dequant_affine_q4(expert_tensors[f"gate_proj.{eid}"], scratch)
                up_w = _dequant_affine_q4(expert_tensors[f"up_proj.{eid}"], scratch)
                down_w = _dequant_affine_q4(expert_tensors[f"down_proj.{eid}"], scratch)

                gate_out = F.linear(x_sel, gate_w)
                up_out = F.linear(x_sel, up_w)
                intermediate = F.silu(gate_out) * up_out
                down_out = F.linear(intermediate, down_w)

                expert_output[token_mask] += expert_w[token_mask].unsqueeze(-1) * down_out

    expert_output = expert_output.view(batch, seq_len, hidden_size)

    # Shared expert (always active, loaded in partial layer)
    shared_gate = affine_q4_linear(x, weights, "mlp.shared_expert.gate_proj", scratch)
    shared_up = affine_q4_linear(x, weights, "mlp.shared_expert.up_proj", scratch)
    shared_intermediate = F.silu(shared_gate) * shared_up
    shared_out = affine_q4_linear(
        shared_intermediate, weights, "mlp.shared_expert.down_proj", scratch
    )

    # Shared expert gate
    shared_gate_w = weights["mlp.shared_expert_gate.weight"]
    if isinstance(shared_gate_w, dict):
        shared_gate_w = _dequant_affine_q4(shared_gate_w, scratch)
    shared_weight = torch.sigmoid(F.linear(x, shared_gate_w))
    shared_out = shared_weight * shared_out

    result = expert_output + shared_out
    # Clamp to fp16 safe range to prevent -inf from overflow in MoE MLP
    return result.clamp(min=-65504.0, max=65504.0)


# ─── Transformer Block (split into attention + MLP) ──────────────────────

def qwen_moe_attention(
    x: torch.Tensor,
    weights: Dict,
    layer_type: str,
    freqs_cis: torch.Tensor,
    mask: Optional[torch.Tensor],
    config: dict,
    kv_cache=None,
    ssm_state: Optional[dict] = None,
    layer_idx: int = 0,
    scratch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Attention half of a Qwen3.5-MoE block (with residual).

    Returns x + attention(norm(x)).
    The caller should then compute routing and MoE MLP separately.
    """
    rms_norm_eps = config["rms_norm_eps"]

    h = rms_norm(x, weights["input_layernorm.weight"], rms_norm_eps)

    if layer_type == "full_attention":
        h = full_attention(
            h, weights, freqs_cis, mask,
            num_heads=config["num_heads"],
            num_kv_heads=config["num_kv_heads"],
            head_dim=config["head_dim"],
            partial_rotary_factor=config["partial_rotary_factor"],
            rms_norm_eps=rms_norm_eps,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            scratch=scratch,
        )
    else:
        h = linear_attention(
            h, weights,
            ssm_state=ssm_state,
            layer_idx=layer_idx,
            scratch=scratch,
            config=config,
        )

    # Clamp after residual to prevent fp16 overflow (-inf)
    return (x + h).clamp(min=-65504.0, max=65504.0)
