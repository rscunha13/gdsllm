/*
 * GdsLLM — Affine 4-bit Dequantization Kernel Declarations
 *
 * CUDA kernels for affine Q4 quantized weights (SWAN/Qwen3.5-MoE format).
 * Format: packed uint32 weights + fp16 per-group scales + fp16 per-group biases
 * Dequant: W_fp = nibble * scale + bias  (group_size=128)
 *
 * Memory layout per weight matrix [out_dim, in_dim]:
 *   weight: [out_dim, in_dim/8] uint32 (8 nibbles packed per uint32, LSB-first)
 *   scales: [out_dim, in_dim/group_size] float16
 *   biases: [out_dim, in_dim/group_size] float16
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// Dequantize affine Q4 packed data to fp16.
// weight:     packed uint32 data on GPU [out_dim, in_dim/8]
// scales:     fp16 per-group scales on GPU [out_dim, in_dim/group_size]
// biases:     fp16 per-group biases on GPU [out_dim, in_dim/group_size]
// dst:        output fp16 buffer [out_dim, in_dim]
// num_elements: total output elements (out_dim * in_dim)
// in_dim:     number of columns (must be multiple of group_size)
// group_size: number of weights sharing one scale+bias (typically 128)
void launch_dequant_affine_q4(
    const uint32_t* weight, const __half* scales, const __half* biases,
    __half* dst, size_t num_elements, int in_dim, int group_size,
    cudaStream_t stream = 0);

// Fused affine Q4 dequant + GEMV: y = dequant(W_aq4) @ x
// For decode (batch=1, seq_len=1) only.
// weight: [out_dim, in_dim/8] packed uint32
// scales: [out_dim, in_dim/group_size] fp16
// biases: [out_dim, in_dim/group_size] fp16
// x:      [in_dim] fp16 input vector
// y:      [out_dim] fp16 output vector
void launch_fused_affine_q4_gemv(
    const uint32_t* weight, const __half* scales, const __half* biases,
    const __half* x, __half* y,
    int out_dim, int in_dim, int group_size, cudaStream_t stream);
