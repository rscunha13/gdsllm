/*
 * GdsLLM — Affine 4-bit Dequantization CUDA Kernels
 *
 * Dequantizes affine Q4 weights (SWAN/Qwen3.5-MoE format) from packed
 * uint32 + fp16 scales + fp16 biases into fp16 tensors.
 *
 * Affine Q4 format:
 *   - Each uint32 packs 8 x 4-bit unsigned values (LSB-first)
 *   - nibble[i] = (packed >> (i*4)) & 0xF
 *   - Dequantization: W_fp = nibble * scale + bias
 *   - Per-group: 128 weights share one scale and one bias (both fp16)
 *
 * Also includes a fused dequant+GEMV kernel for decode (batch=1).
 */

#include <cstdint>
#include <cstdio>

#include "affine_q4_dequant.h"

// ─── Affine Q4 Dequant ──────────────────────────────────────────────────

// One thread processes one group of 128 weights (16 uint32 values)
#define AQ4_GROUP_SIZE_DEFAULT 128
#define AQ4_NIBBLES_PER_U32 8
#define AQ4_DEQUANT_THREADS 256

__global__ void kernel_dequant_affine_q4(
    const uint32_t* __restrict__ weight,
    const __half*   __restrict__ scales,
    const __half*   __restrict__ biases,
    __half*         __restrict__ dst,
    size_t num_groups,
    int group_size
) {
    size_t group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx >= num_groups) return;

    float scale = __half2float(scales[group_idx]);
    float bias  = __half2float(biases[group_idx]);

    // Number of uint32 words per group
    int words_per_group = group_size / AQ4_NIBBLES_PER_U32;

    // Pointers into this group
    const uint32_t* w_ptr = weight + group_idx * words_per_group;
    __half* out = dst + group_idx * group_size;

    // Unpack each uint32 into 8 fp16 values
    for (int w = 0; w < words_per_group; w++) {
        uint32_t packed = w_ptr[w];
        #pragma unroll
        for (int n = 0; n < AQ4_NIBBLES_PER_U32; n++) {
            int nibble = (packed >> (n * 4)) & 0xF;
            out[w * AQ4_NIBBLES_PER_U32 + n] =
                __float2half(static_cast<float>(nibble) * scale + bias);
        }
    }
}

void launch_dequant_affine_q4(
    const uint32_t* weight, const __half* scales, const __half* biases,
    __half* dst, size_t num_elements, int in_dim, int group_size,
    cudaStream_t stream
) {
    if (num_elements == 0) return;

    size_t num_groups = num_elements / group_size;
    const int threads = AQ4_DEQUANT_THREADS;
    const int blocks = (num_groups + threads - 1) / threads;
    kernel_dequant_affine_q4<<<blocks, threads, 0, stream>>>(
        weight, scales, biases, dst, num_groups, group_size);
}

// ─── Affine Q4 Fused Dequant + GEMV ────────────────────────────────────

#define GEMV_THREADS 256

__global__ void kernel_fused_affine_q4_gemv(
    const uint32_t* __restrict__ weight,
    const __half*   __restrict__ scales,
    const __half*   __restrict__ biases,
    const __half*   __restrict__ x,
    __half*         __restrict__ y,
    int out_dim,
    int in_dim,
    int group_size
) {
    int row = blockIdx.x;
    if (row >= out_dim) return;

    int groups_per_row = in_dim / group_size;
    int words_per_group = group_size / AQ4_NIBBLES_PER_U32;
    int words_per_row = in_dim / AQ4_NIBBLES_PER_U32;

    // Load input vector into shared memory
    extern __shared__ __half s_x[];
    for (int i = threadIdx.x; i < in_dim; i += GEMV_THREADS) {
        s_x[i] = x[i];
    }
    __syncthreads();

    // Each thread accumulates over assigned groups
    float acc = 0.0f;

    // Row pointers
    const uint32_t* row_weight = weight + (size_t)row * words_per_row;
    const __half* row_scales = scales + (size_t)row * groups_per_row;
    const __half* row_biases = biases + (size_t)row * groups_per_row;

    for (int g = threadIdx.x; g < groups_per_row; g += GEMV_THREADS) {
        float scale = __half2float(row_scales[g]);
        float bias  = __half2float(row_biases[g]);

        const uint32_t* g_weight = row_weight + g * words_per_group;
        int col_base = g * group_size;

        for (int w = 0; w < words_per_group; w++) {
            uint32_t packed = g_weight[w];
            int col = col_base + w * AQ4_NIBBLES_PER_U32;

            #pragma unroll
            for (int n = 0; n < AQ4_NIBBLES_PER_U32; n++) {
                int nibble = (packed >> (n * 4)) & 0xF;
                float val = static_cast<float>(nibble) * scale + bias;
                acc += val * __half2float(s_x[col + n]);
            }
        }
    }

    // Block-level reduction: warp shuffle then shared memory
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

void launch_fused_affine_q4_gemv(
    const uint32_t* weight, const __half* scales, const __half* biases,
    const __half* x, __half* y,
    int out_dim, int in_dim, int group_size, cudaStream_t stream
) {
    if (out_dim == 0 || in_dim == 0) return;

    size_t smem_bytes = in_dim * sizeof(__half);

    int max_smem = 0;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    if (smem_bytes + 32 > 48 * 1024 && smem_bytes + 32 <= (size_t)max_smem) {
        cudaFuncSetAttribute(
            kernel_fused_affine_q4_gemv,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes);
    }

    kernel_fused_affine_q4_gemv<<<out_dim, GEMV_THREADS, smem_bytes, stream>>>(
        weight, scales, biases, x, y, out_dim, in_dim, group_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "fused_affine_q4_gemv launch error: %s "
                "(out_dim=%d, in_dim=%d, group_size=%d, smem=%zu)\n",
                cudaGetErrorString(err), out_dim, in_dim, group_size, smem_bytes);
    }
}
