/*
 * GdsLLM — Fused Dequant+GEMV CUDA Kernels
 *
 * Performs matrix-vector multiplication directly on GGUF block-quantized
 * weights without materializing an fp16 intermediate. Each CUDA block
 * computes one output element by:
 *   1. Loading the input vector into shared memory
 *   2. Each thread dequantizes a subset of Q4_0/Q8_0 blocks for one row
 *   3. Accumulates dot product with input in float registers
 *   4. Block-level parallel reduction via warp shuffle + shared memory
 *
 * This eliminates the largest bottleneck: writing then re-reading a
 * 134 MB fp16 intermediate per weight matrix (for 70B Q4_0).
 */

#include <cstdint>
#include <cstdio>

#include "fused_gemv.h"

// ─── Q4_0 Fused GEMV ─────────────────────────────────────────────────────

#define Q4_0_BLOCK_SIZE 32
#define Q4_0_BYTES_PER_BLOCK 18
#define GEMV_THREADS 256

__global__ void kernel_fused_q4_0_gemv(
    const uint8_t* __restrict__ weight,
    const __half* __restrict__ x,
    __half* __restrict__ y,
    int out_dim,
    int in_dim
) {
    int row = blockIdx.x;
    if (row >= out_dim) return;

    int blocks_per_row = in_dim / Q4_0_BLOCK_SIZE;

    // Load input vector into shared memory (16 KB for in_dim=8192)
    extern __shared__ __half s_x[];
    for (int i = threadIdx.x; i < in_dim; i += GEMV_THREADS) {
        s_x[i] = x[i];
    }
    __syncthreads();

    // Each thread accumulates dot product over its assigned Q4_0 blocks
    float acc = 0.0f;

    // Pointer to this row's Q4_0 blocks
    const uint8_t* row_ptr = weight + (size_t)row * blocks_per_row * Q4_0_BYTES_PER_BLOCK;

    for (int b = threadIdx.x; b < blocks_per_row; b += GEMV_THREADS) {
        const uint8_t* block_ptr = row_ptr + (size_t)b * Q4_0_BYTES_PER_BLOCK;

        // Read scale (fp16)
        float scale = __half2float(*reinterpret_cast<const __half*>(block_ptr));

        // Read 16 packed bytes (32 nibbles)
        const uint8_t* qs = block_ptr + 2;
        int col_base = b * Q4_0_BLOCK_SIZE;

        // Dequant and dot product — 32 weights per block
        // Q4_0 layout: low nibbles are indices 0..15, high nibbles are 16..31
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            uint8_t byte = qs[j];
            int lo = (byte & 0x0F) - 8;
            int hi = (byte >> 4) - 8;
            acc += scale * lo * __half2float(s_x[col_base + j]);
            acc += scale * hi * __half2float(s_x[col_base + j + 16]);
        }
    }

    // Block-level reduction: warp shuffle first, then shared memory
    // Reuse shared memory for reduction (after input vector is no longer needed)
    __shared__ float s_reduce[GEMV_THREADS / 32];  // one slot per warp

    // Warp-level reduction
    unsigned mask = 0xFFFFFFFF;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(mask, acc, offset);
    }

    // First thread in each warp writes to shared memory
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) {
        s_reduce[warp_id] = acc;
    }
    __syncthreads();

    // First warp reduces across warps
    if (warp_id == 0) {
        int num_warps = GEMV_THREADS / 32;
        acc = (lane_id < num_warps) ? s_reduce[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = num_warps / 2; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(mask, acc, offset);
        }
        if (lane_id == 0) {
            y[row] = __float2half(acc);
        }
    }
}

void launch_fused_q4_0_gemv(
    const void* weight, const __half* x, __half* y,
    int out_dim, int in_dim, cudaStream_t stream
) {
    if (out_dim == 0 || in_dim == 0) return;

    // Shared memory: input vector (in_dim * 2 bytes)
    size_t smem_bytes = in_dim * sizeof(__half);

    // Ensure shared memory fits (48 KB default, request more if needed)
    int max_smem = 0;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    if (smem_bytes + 32 > 48 * 1024 && smem_bytes + 32 <= (size_t)max_smem) {
        cudaFuncSetAttribute(
            kernel_fused_q4_0_gemv,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes);
    }

    kernel_fused_q4_0_gemv<<<out_dim, GEMV_THREADS, smem_bytes, stream>>>(
        static_cast<const uint8_t*>(weight), x, y, out_dim, in_dim);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "fused_q4_0_gemv launch error: %s (out_dim=%d, in_dim=%d, smem=%zu)\n",
                cudaGetErrorString(err), out_dim, in_dim, smem_bytes);
    }
}

// ─── Q8_0 Fused GEMV ─────────────────────────────────────────────────────

#define Q8_0_BLOCK_SIZE 32
#define Q8_0_BYTES_PER_BLOCK 34

__global__ void kernel_fused_q8_0_gemv(
    const uint8_t* __restrict__ weight,
    const __half* __restrict__ x,
    __half* __restrict__ y,
    int out_dim,
    int in_dim
) {
    int row = blockIdx.x;
    if (row >= out_dim) return;

    int blocks_per_row = in_dim / Q8_0_BLOCK_SIZE;

    // Load input vector into shared memory
    extern __shared__ __half s_x[];
    for (int i = threadIdx.x; i < in_dim; i += GEMV_THREADS) {
        s_x[i] = x[i];
    }
    __syncthreads();

    float acc = 0.0f;

    const uint8_t* row_ptr = weight + (size_t)row * blocks_per_row * Q8_0_BYTES_PER_BLOCK;

    for (int b = threadIdx.x; b < blocks_per_row; b += GEMV_THREADS) {
        const uint8_t* block_ptr = row_ptr + (size_t)b * Q8_0_BYTES_PER_BLOCK;

        float scale = __half2float(*reinterpret_cast<const __half*>(block_ptr));
        const int8_t* qs = reinterpret_cast<const int8_t*>(block_ptr + 2);
        int col_base = b * Q8_0_BLOCK_SIZE;

        #pragma unroll
        for (int j = 0; j < Q8_0_BLOCK_SIZE; j++) {
            acc += scale * static_cast<float>(qs[j]) * __half2float(s_x[col_base + j]);
        }
    }

    // Block-level reduction
    __shared__ float s_reduce[GEMV_THREADS / 32];

    unsigned mask = 0xFFFFFFFF;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(mask, acc, offset);
    }

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) {
        s_reduce[warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = GEMV_THREADS / 32;
        acc = (lane_id < num_warps) ? s_reduce[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = num_warps / 2; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(mask, acc, offset);
        }
        if (lane_id == 0) {
            y[row] = __float2half(acc);
        }
    }
}

void launch_fused_q8_0_gemv(
    const void* weight, const __half* x, __half* y,
    int out_dim, int in_dim, cudaStream_t stream
) {
    if (out_dim == 0 || in_dim == 0) return;

    size_t smem_bytes = in_dim * sizeof(__half);

    int max_smem = 0;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    if (smem_bytes + 32 > 48 * 1024 && smem_bytes + 32 <= (size_t)max_smem) {
        cudaFuncSetAttribute(
            kernel_fused_q8_0_gemv,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes);
    }

    kernel_fused_q8_0_gemv<<<out_dim, GEMV_THREADS, smem_bytes, stream>>>(
        static_cast<const uint8_t*>(weight), x, y, out_dim, in_dim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "fused_q8_0_gemv launch error: %s (out_dim=%d, in_dim=%d, smem=%zu)\n",
                cudaGetErrorString(err), out_dim, in_dim, smem_bytes);
    }
}