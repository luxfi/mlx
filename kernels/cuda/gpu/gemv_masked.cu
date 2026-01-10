// Copyright (c) 2024-2026 Lux Industries Inc. All Rights Reserved.
// PROPRIETARY AND CONFIDENTIAL - NO LICENSE GRANTED
// Contact: licensing@luxindustries.xyz
//
// Masked GEMV (General Matrix-Vector Multiplication) - CUDA Implementation
// Optimized for sparse attention patterns in transformer architectures.
// Supports various mask types: causal, block-sparse, sliding window.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace lux {
namespace cuda {
namespace gemv {

// ============================================================================
// Configuration
// ============================================================================

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int TILE_M = 4;       // Rows per thread
constexpr int TILE_K = 128;     // Columns processed per iteration

// Mask types for sparse attention
enum class MaskType : uint32_t {
    NONE = 0,           // No masking (dense GEMV)
    CAUSAL = 1,         // Lower triangular (causal attention)
    BLOCK_SPARSE = 2,   // Block-sparse pattern
    SLIDING_WINDOW = 3, // Sliding window attention
    CUSTOM = 4          // Custom mask buffer
};

// ============================================================================
// Mask Utilities
// ============================================================================

// Check if position should be masked (returns true if valid, false if masked)
__device__ __forceinline__
bool is_unmasked_causal(int row, int col) {
    return col <= row;
}

__device__ __forceinline__
bool is_unmasked_sliding(int row, int col, int window_size) {
    int diff = row - col;
    return diff >= 0 && diff < window_size;
}

__device__ __forceinline__
bool is_unmasked_block_sparse(
    int row, int col,
    const uint32_t* block_mask,  // Bitmap for block sparsity
    int block_size,
    int blocks_per_row
) {
    int block_row = row / block_size;
    int block_col = col / block_size;
    int block_idx = block_row * blocks_per_row + block_col;
    int word_idx = block_idx / 32;
    int bit_idx = block_idx % 32;
    return (block_mask[word_idx] >> bit_idx) & 1;
}

// ============================================================================
// FP32 Masked GEMV Kernels
// ============================================================================

// Dense GEMV: y = alpha * A * x + beta * y
// A is M x K, x is K x 1, y is M x 1
__global__ void gemv_f32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K,
    float alpha,
    float beta
) {
    const int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (row >= M) return;

    float sum = 0.0f;
    const float* A_row = A + row * K;

    #pragma unroll 4
    for (int k = 0; k < K; k++) {
        sum += A_row[k] * x[k];
    }

    if (beta == 0.0f) {
        y[row] = alpha * sum;
    } else {
        y[row] = alpha * sum + beta * y[row];
    }
}

// Causal masked GEMV (lower triangular)
__global__ void gemv_causal_f32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K,
    float alpha,
    float beta
) {
    const int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (row >= M) return;

    float sum = 0.0f;
    const float* A_row = A + row * K;

    // Only sum up to and including current row (causal mask)
    int max_k = min(row + 1, K);

    #pragma unroll 4
    for (int k = 0; k < max_k; k++) {
        sum += A_row[k] * x[k];
    }

    if (beta == 0.0f) {
        y[row] = alpha * sum;
    } else {
        y[row] = alpha * sum + beta * y[row];
    }
}

// Sliding window masked GEMV
__global__ void gemv_sliding_f32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K,
    int window_size,
    float alpha,
    float beta
) {
    const int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (row >= M) return;

    float sum = 0.0f;
    const float* A_row = A + row * K;

    // Only sum within sliding window
    int k_start = max(0, row - window_size + 1);
    int k_end = min(row + 1, K);

    for (int k = k_start; k < k_end; k++) {
        sum += A_row[k] * x[k];
    }

    if (beta == 0.0f) {
        y[row] = alpha * sum;
    } else {
        y[row] = alpha * sum + beta * y[row];
    }
}

// Block-sparse masked GEMV
__global__ void gemv_block_sparse_f32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K,
    const uint32_t* __restrict__ block_mask,
    int block_size,
    float alpha,
    float beta
) {
    const int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (row >= M) return;

    float sum = 0.0f;
    const float* A_row = A + row * K;

    int block_row = row / block_size;
    int blocks_per_row = (K + block_size - 1) / block_size;

    // Iterate over blocks
    for (int b = 0; b < blocks_per_row; b++) {
        int block_idx = block_row * blocks_per_row + b;
        int word_idx = block_idx / 32;
        int bit_idx = block_idx % 32;

        if (!((block_mask[word_idx] >> bit_idx) & 1)) continue;

        // Process block
        int k_start = b * block_size;
        int k_end = min(k_start + block_size, K);

        for (int k = k_start; k < k_end; k++) {
            sum += A_row[k] * x[k];
        }
    }

    if (beta == 0.0f) {
        y[row] = alpha * sum;
    } else {
        y[row] = alpha * sum + beta * y[row];
    }
}

// Custom mask buffer GEMV
__global__ void gemv_custom_mask_f32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    const uint8_t* __restrict__ mask,  // M x K mask (1 = valid, 0 = masked)
    int M,
    int K,
    float alpha,
    float beta
) {
    const int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (row >= M) return;

    float sum = 0.0f;
    const float* A_row = A + row * K;
    const uint8_t* mask_row = mask + row * K;

    for (int k = 0; k < K; k++) {
        if (mask_row[k]) {
            sum += A_row[k] * x[k];
        }
    }

    if (beta == 0.0f) {
        y[row] = alpha * sum;
    } else {
        y[row] = alpha * sum + beta * y[row];
    }
}

// ============================================================================
// FP16 Masked GEMV Kernels
// ============================================================================

// Dense GEMV with FP16 (uses FP32 accumulation)
__global__ void gemv_f16_kernel(
    const half* __restrict__ A,
    const half* __restrict__ x,
    half* __restrict__ y,
    int M,
    int K,
    float alpha,
    float beta
) {
    const int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (row >= M) return;

    float sum = 0.0f;
    const half* A_row = A + row * K;

    // Use half2 for vectorized loads when K is even
    int k = 0;
    if (K >= 2) {
        const half2* A_row2 = (const half2*)A_row;
        const half2* x2 = (const half2*)x;
        int K2 = K / 2;

        for (; k < K2; k++) {
            half2 a = A_row2[k];
            half2 b = x2[k];
            sum += __half2float(a.x) * __half2float(b.x);
            sum += __half2float(a.y) * __half2float(b.y);
        }
        k *= 2;
    }

    // Handle remaining elements
    for (; k < K; k++) {
        sum += __half2float(A_row[k]) * __half2float(x[k]);
    }

    if (beta == 0.0f) {
        y[row] = __float2half(alpha * sum);
    } else {
        y[row] = __float2half(alpha * sum + beta * __half2float(y[row]));
    }
}

// Causal masked GEMV with FP16
__global__ void gemv_causal_f16_kernel(
    const half* __restrict__ A,
    const half* __restrict__ x,
    half* __restrict__ y,
    int M,
    int K,
    float alpha,
    float beta
) {
    const int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (row >= M) return;

    float sum = 0.0f;
    const half* A_row = A + row * K;

    int max_k = min(row + 1, K);

    for (int k = 0; k < max_k; k++) {
        sum += __half2float(A_row[k]) * __half2float(x[k]);
    }

    if (beta == 0.0f) {
        y[row] = __float2half(alpha * sum);
    } else {
        y[row] = __float2half(alpha * sum + beta * __half2float(y[row]));
    }
}

// ============================================================================
// Batched Masked GEMV
// ============================================================================

// Batched dense GEMV: Y[b] = alpha * A[b] * X[b] + beta * Y[b]
__global__ void gemv_batched_f32_kernel(
    const float* const* __restrict__ A_batch,
    const float* const* __restrict__ x_batch,
    float* const* __restrict__ y_batch,
    int M,
    int K,
    int batch_size,
    float alpha,
    float beta
) {
    const int batch_idx = blockIdx.y;
    const int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (batch_idx >= batch_size || row >= M) return;

    const float* A = A_batch[batch_idx];
    const float* x = x_batch[batch_idx];
    float* y = y_batch[batch_idx];

    float sum = 0.0f;
    const float* A_row = A + row * K;

    #pragma unroll 4
    for (int k = 0; k < K; k++) {
        sum += A_row[k] * x[k];
    }

    if (beta == 0.0f) {
        y[row] = alpha * sum;
    } else {
        y[row] = alpha * sum + beta * y[row];
    }
}

// Batched causal masked GEMV
__global__ void gemv_batched_causal_f32_kernel(
    const float* const* __restrict__ A_batch,
    const float* const* __restrict__ x_batch,
    float* const* __restrict__ y_batch,
    int M,
    int K,
    int batch_size,
    float alpha,
    float beta
) {
    const int batch_idx = blockIdx.y;
    const int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (batch_idx >= batch_size || row >= M) return;

    const float* A = A_batch[batch_idx];
    const float* x = x_batch[batch_idx];
    float* y = y_batch[batch_idx];

    float sum = 0.0f;
    const float* A_row = A + row * K;

    int max_k = min(row + 1, K);

    #pragma unroll 4
    for (int k = 0; k < max_k; k++) {
        sum += A_row[k] * x[k];
    }

    if (beta == 0.0f) {
        y[row] = alpha * sum;
    } else {
        y[row] = alpha * sum + beta * y[row];
    }
}

// ============================================================================
// Strided Masked GEMV (for multi-head attention)
// ============================================================================

// Strided GEMV with causal mask for multi-head attention
// A is [num_heads, M, K], x is [num_heads, K], y is [num_heads, M]
__global__ void gemv_strided_causal_f32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K,
    int num_heads,
    int stride_A_head,  // Stride between heads in A
    int stride_x_head,  // Stride between heads in x
    int stride_y_head,  // Stride between heads in y
    float alpha,
    float beta
) {
    const int head = blockIdx.y;
    const int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (head >= num_heads || row >= M) return;

    const float* A_head = A + head * stride_A_head;
    const float* x_head = x + head * stride_x_head;
    float* y_head = y + head * stride_y_head;

    float sum = 0.0f;
    const float* A_row = A_head + row * K;

    int max_k = min(row + 1, K);

    #pragma unroll 4
    for (int k = 0; k < max_k; k++) {
        sum += A_row[k] * x_head[k];
    }

    if (beta == 0.0f) {
        y_head[row] = alpha * sum;
    } else {
        y_head[row] = alpha * sum + beta * y_head[row];
    }
}

// ============================================================================
// Warp-Optimized GEMV (for small K)
// ============================================================================

// Warp-level reduction for GEMV
__device__ __forceinline__
float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp-optimized GEMV for K <= 32 (one warp per row)
__global__ void gemv_warp_f32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K,
    float alpha,
    float beta
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= M) return;

    const float* A_row = A + warp_id * K;

    float val = 0.0f;
    if (lane_id < K) {
        val = A_row[lane_id] * x[lane_id];
    }

    // Warp reduction
    float sum = warp_reduce_sum(val);

    // First lane writes result
    if (lane_id == 0) {
        if (beta == 0.0f) {
            y[warp_id] = alpha * sum;
        } else {
            y[warp_id] = alpha * sum + beta * y[warp_id];
        }
    }
}

// ============================================================================
// Host API (C Interface)
// ============================================================================

} // namespace gemv
} // namespace cuda
} // namespace lux

extern "C" {

using namespace lux::cuda::gemv;

void lux_cuda_gemv_f32(
    const float* A,
    const float* x,
    float* y,
    int M,
    int K,
    float alpha,
    float beta,
    uint32_t mask_type,
    const void* mask_data,
    int mask_param,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    switch ((MaskType)mask_type) {
        case MaskType::NONE:
            gemv_f32_kernel<<<grid, block, 0, stream>>>(
                A, x, y, M, K, alpha, beta
            );
            break;

        case MaskType::CAUSAL:
            gemv_causal_f32_kernel<<<grid, block, 0, stream>>>(
                A, x, y, M, K, alpha, beta
            );
            break;

        case MaskType::SLIDING_WINDOW:
            gemv_sliding_f32_kernel<<<grid, block, 0, stream>>>(
                A, x, y, M, K, mask_param, alpha, beta
            );
            break;

        case MaskType::BLOCK_SPARSE:
            gemv_block_sparse_f32_kernel<<<grid, block, 0, stream>>>(
                A, x, y, M, K, (const uint32_t*)mask_data, mask_param, alpha, beta
            );
            break;

        case MaskType::CUSTOM:
            gemv_custom_mask_f32_kernel<<<grid, block, 0, stream>>>(
                A, x, y, (const uint8_t*)mask_data, M, K, alpha, beta
            );
            break;
    }
}

void lux_cuda_gemv_f16(
    const void* A,
    const void* x,
    void* y,
    int M,
    int K,
    float alpha,
    float beta,
    uint32_t mask_type,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    switch ((MaskType)mask_type) {
        case MaskType::NONE:
            gemv_f16_kernel<<<grid, block, 0, stream>>>(
                (const half*)A, (const half*)x, (half*)y, M, K, alpha, beta
            );
            break;

        case MaskType::CAUSAL:
            gemv_causal_f16_kernel<<<grid, block, 0, stream>>>(
                (const half*)A, (const half*)x, (half*)y, M, K, alpha, beta
            );
            break;

        default:
            // Other mask types not yet implemented for FP16
            break;
    }
}

void lux_cuda_gemv_batched_f32(
    const float* const* A_batch,
    const float* const* x_batch,
    float* const* y_batch,
    int M,
    int K,
    int batch_size,
    float alpha,
    float beta,
    uint32_t mask_type,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size);

    switch ((MaskType)mask_type) {
        case MaskType::NONE:
            gemv_batched_f32_kernel<<<grid, block, 0, stream>>>(
                A_batch, x_batch, y_batch, M, K, batch_size, alpha, beta
            );
            break;

        case MaskType::CAUSAL:
            gemv_batched_causal_f32_kernel<<<grid, block, 0, stream>>>(
                A_batch, x_batch, y_batch, M, K, batch_size, alpha, beta
            );
            break;

        default:
            break;
    }
}

void lux_cuda_gemv_strided_f32(
    const float* A,
    const float* x,
    float* y,
    int M,
    int K,
    int num_heads,
    int stride_A,
    int stride_x,
    int stride_y,
    float alpha,
    float beta,
    uint32_t mask_type,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, num_heads);

    if ((MaskType)mask_type == MaskType::CAUSAL) {
        gemv_strided_causal_f32_kernel<<<grid, block, 0, stream>>>(
            A, x, y, M, K, num_heads, stride_A, stride_x, stride_y, alpha, beta
        );
    }
}

void lux_cuda_gemv_warp_f32(
    const float* A,
    const float* x,
    float* y,
    int M,
    int K,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    // One warp per row
    int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    int num_blocks = (M + warps_per_block - 1) / warps_per_block;

    dim3 block(BLOCK_SIZE);
    dim3 grid(num_blocks);

    gemv_warp_f32_kernel<<<grid, block, 0, stream>>>(
        A, x, y, M, K, alpha, beta
    );
}

} // extern "C"
