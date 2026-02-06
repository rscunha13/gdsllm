/*
 * GdsLLM — GGUF Block Dequantization Kernel Declarations
 *
 * CUDA kernels that dequantize GGUF block-quantized data (Q4_0, Q8_0)
 * from raw GPU memory into fp16 tensors.
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// Dequantize Q8_0 blocks to fp16.
// src: raw Q8_0 block data on GPU (34 bytes per block of 32 weights)
// dst: output fp16 buffer (num_elements * sizeof(half))
// num_blocks: total number of Q8_0 blocks
void launch_dequant_q8_0(
    const void* src, __half* dst, size_t num_blocks, cudaStream_t stream = 0);

// Dequantize Q4_0 blocks to fp16.
// src: raw Q4_0 block data on GPU (18 bytes per block of 32 weights)
// dst: output fp16 buffer (num_elements * sizeof(half))
// num_blocks: total number of Q4_0 blocks
void launch_dequant_q4_0(
    const void* src, __half* dst, size_t num_blocks, cudaStream_t stream = 0);
