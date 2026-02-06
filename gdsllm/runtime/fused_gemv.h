/*
 * GdsLLM — Fused Dequant+GEMV Kernel Declarations
 *
 * Fused kernels that dequantize GGUF block-quantized weight matrices
 * and perform matrix-vector multiplication in a single pass, avoiding
 * the fp16 intermediate allocation.
 *
 * For decode (batch=1, seq_len=1) only. Prefill uses separate
 * dequant + cuBLAS GEMM.
 */

#ifndef GDSLLM_FUSED_GEMV_H
#define GDSLLM_FUSED_GEMV_H

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>

// Fused Q4_0 dequant + GEMV: y = dequant(W_q4) @ x
// weight: [out_dim, in_dim] stored as Q4_0 blocks (18 bytes per 32 weights)
// x: [in_dim] fp16 input vector
// y: [out_dim] fp16 output vector
void launch_fused_q4_0_gemv(
    const void* weight, const __half* x, __half* y,
    int out_dim, int in_dim, cudaStream_t stream);

// Fused Q8_0 dequant + GEMV: y = dequant(W_q8) @ x
// weight: [out_dim, in_dim] stored as Q8_0 blocks (34 bytes per 32 weights)
void launch_fused_q8_0_gemv(
    const void* weight, const __half* x, __half* y,
    int out_dim, int in_dim, cudaStream_t stream);

#endif // GDSLLM_FUSED_GEMV_H