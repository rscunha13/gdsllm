/*
 * GdsLLM — GGUF Block Dequantization CUDA Kernels
 *
 * Dequantizes Q8_0 and Q4_0 block-quantized data (as stored in GGUF files)
 * from raw GPU memory into fp16 tensors. One thread per block of 32 weights.
 *
 * Block formats (matching llama.cpp / GGML):
 *
 * Q8_0: 34 bytes per block of 32 weights
 *   - half  scale      (2 bytes)
 *   - int8  qs[32]     (32 bytes)
 *   - dequant: output[i] = scale * qs[i]
 *
 * Q4_0: 18 bytes per block of 32 weights
 *   - half  scale      (2 bytes)
 *   - uint8 qs[16]     (16 bytes, 2 nibbles packed per byte)
 *   - dequant: output[i] = scale * (nibble - 8)
 *     low nibble  = qs[i/2] & 0x0F  (for even i)
 *     high nibble = qs[i/2] >> 4     (for odd i within each byte)
 */

#include <cstdint>

#include "gguf_dequant.h"

// ─── Q8_0 ─────────────────────────────────────────────────────────────────

// Q8_0 block: 2 bytes scale + 32 bytes quants = 34 bytes
#define Q8_0_BLOCK_SIZE 32
#define Q8_0_BYTES_PER_BLOCK 34

__global__ void kernel_dequant_q8_0(
    const uint8_t* __restrict__ src,
    __half* __restrict__ dst,
    size_t num_blocks
) {
    size_t block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks) return;

    // Pointer to this Q8_0 block
    const uint8_t* block_ptr = src + block_idx * Q8_0_BYTES_PER_BLOCK;

    // First 2 bytes: fp16 scale
    __half scale = *reinterpret_cast<const __half*>(block_ptr);

    // Next 32 bytes: int8 quantized weights
    const int8_t* qs = reinterpret_cast<const int8_t*>(block_ptr + 2);

    // Output location: 32 fp16 values per block
    __half* out = dst + block_idx * Q8_0_BLOCK_SIZE;

    float scale_f = __half2float(scale);

    #pragma unroll
    for (int i = 0; i < Q8_0_BLOCK_SIZE; i++) {
        out[i] = __float2half(scale_f * static_cast<float>(qs[i]));
    }
}

void launch_dequant_q8_0(
    const void* src, __half* dst, size_t num_blocks, cudaStream_t stream
) {
    if (num_blocks == 0) return;
    const int threads = 256;
    const int blocks = (num_blocks + threads - 1) / threads;
    kernel_dequant_q8_0<<<blocks, threads, 0, stream>>>(
        static_cast<const uint8_t*>(src), dst, num_blocks);
}

// ─── Q4_0 ─────────────────────────────────────────────────────────────────

// Q4_0 block: 2 bytes scale + 16 bytes quants = 18 bytes
#define Q4_0_BLOCK_SIZE 32
#define Q4_0_BYTES_PER_BLOCK 18

__global__ void kernel_dequant_q4_0(
    const uint8_t* __restrict__ src,
    __half* __restrict__ dst,
    size_t num_blocks
) {
    size_t block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks) return;

    // Pointer to this Q4_0 block
    const uint8_t* block_ptr = src + block_idx * Q4_0_BYTES_PER_BLOCK;

    // First 2 bytes: fp16 scale
    __half scale = *reinterpret_cast<const __half*>(block_ptr);

    // Next 16 bytes: packed 4-bit quants (2 per byte)
    const uint8_t* qs = block_ptr + 2;

    // Output location: 32 fp16 values per block
    __half* out = dst + block_idx * Q4_0_BLOCK_SIZE;

    float scale_f = __half2float(scale);

    // Each byte holds 2 weights: low nibble first, high nibble second
    // GGML Q4_0: values are unsigned 0..15, centered by subtracting 8
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        uint8_t byte = qs[j];
        int lo = (byte & 0x0F);
        int hi = (byte >> 4);
        out[j]      = __float2half(scale_f * static_cast<float>(lo - 8));
        out[j + 16] = __float2half(scale_f * static_cast<float>(hi - 8));
    }
}

void launch_dequant_q4_0(
    const void* src, __half* dst, size_t num_blocks, cudaStream_t stream
) {
    if (num_blocks == 0) return;
    const int threads = 256;
    const int blocks = (num_blocks + threads - 1) / threads;
    kernel_dequant_q4_0<<<blocks, threads, 0, stream>>>(
        static_cast<const uint8_t*>(src), dst, num_blocks);
}
