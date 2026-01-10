// Copyright 2025 Lux Industries. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Steel Masked GEMM Kernels - GEMM operations with various masking patterns
// Implements causal masks, attention masks, block-sparse masks for transformers

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

// ============================================================================
// Configuration
// ============================================================================

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define BLOCK_SIZE 256

// Mask types
#define MASK_NONE 0
#define MASK_CAUSAL 1
#define MASK_BIDIRECTIONAL 2
#define MASK_BLOCK_SPARSE 3
#define MASK_SLIDING_WINDOW 4
#define MASK_CUSTOM 5

// ============================================================================
// Device helper functions
// ============================================================================

__device__ __forceinline__
bool is_causal_valid(int row, int col, int offset = 0)
{
    return col <= row + offset;
}

__device__ __forceinline__
bool is_sliding_window_valid(int row, int col, int window_size)
{
    return (col <= row) && (col >= row - window_size + 1);
}

__device__ __forceinline__
bool is_block_sparse_valid(int row, int col, int block_size, const int* pattern)
{
    int block_row = row / block_size;
    int block_col = col / block_size;
    // Pattern is a bitfield or 2D array indicating which blocks are active
    // For simplicity, we use a linear pattern where pattern[block_row] is a bitfield
    return (pattern[block_row] >> block_col) & 1;
}

// ============================================================================
// Causal Masked GEMM - Lower triangular mask for autoregressive attention
// ============================================================================

extern "C" __global__
void steel_gemm_causal_mask_kernel(
    const float* __restrict__ A,   // [M, K] - typically Q
    const float* __restrict__ B,   // [K, N] - typically K^T
    float* __restrict__ C,         // [M, N]
    int M, int N, int K,
    float alpha, float beta,
    float mask_value,              // Value for masked positions (e.g., -inf for softmax)
    int causal_offset)             // Offset for causal mask (0 = strict causal)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;

    const int row_in_tile = tx / (TILE_N / 4);
    const int col_in_tile = (tx % (TILE_N / 4)) * 4;

    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    const int c_row = bx * TILE_M + row_in_tile;
    const int c_col = by * TILE_N + col_in_tile;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Check if entire tile is masked (optimization)
    const int tile_row_start = bx * TILE_M;
    const int tile_col_end = (by + 1) * TILE_N - 1;

    // If entire tile is below the causal line, skip computation
    if (tile_col_end > tile_row_start + TILE_M - 1 + causal_offset) {
        // Could be partially or fully masked
    }

    for (int k_tile = 0; k_tile < (K + TILE_K - 1) / TILE_K; k_tile++) {
        // Load A tile
        for (int i = tx; i < TILE_M * TILE_K; i += BLOCK_SIZE) {
            int local_m = i / TILE_K;
            int local_k = i % TILE_K;
            int global_m = bx * TILE_M + local_m;
            int global_k = k_tile * TILE_K + local_k;

            float val = 0.0f;
            if (global_m < M && global_k < K) {
                val = A[global_m * K + global_k];
            }
            As[local_m][local_k] = val;
        }

        // Load B tile
        for (int i = tx; i < TILE_K * TILE_N; i += BLOCK_SIZE) {
            int local_k = i / TILE_N;
            int local_n = i % TILE_N;
            int global_k = k_tile * TILE_K + local_k;
            int global_n = by * TILE_N + local_n;

            float val = 0.0f;
            if (global_k < K && global_n < N) {
                val = B[global_k * N + global_n];
            }
            Bs[local_k][local_n] = val;
        }

        __syncthreads();

        // Compute
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            float a_val = As[row_in_tile][k];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                acc[j] += a_val * Bs[k][col_in_tile + j];
            }
        }

        __syncthreads();
    }

    // Write output with causal masking
    if (c_row < M) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int global_col = c_col + j;
            if (global_col < N) {
                float result;
                if (is_causal_valid(c_row, global_col, causal_offset)) {
                    result = alpha * acc[j];
                    if (beta != 0.0f) {
                        result += beta * C[c_row * N + global_col];
                    }
                } else {
                    result = mask_value;
                }
                C[c_row * N + global_col] = result;
            }
        }
    }
}

// ============================================================================
// Sliding Window Masked GEMM - For local attention patterns
// ============================================================================

extern "C" __global__
void steel_gemm_sliding_window_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta,
    float mask_value,
    int window_size)              // Size of the sliding window
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;

    const int row_in_tile = tx / (TILE_N / 4);
    const int col_in_tile = (tx % (TILE_N / 4)) * 4;

    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    const int c_row = bx * TILE_M + row_in_tile;
    const int c_col = by * TILE_N + col_in_tile;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k_tile = 0; k_tile < (K + TILE_K - 1) / TILE_K; k_tile++) {
        for (int i = tx; i < TILE_M * TILE_K; i += BLOCK_SIZE) {
            int local_m = i / TILE_K;
            int local_k = i % TILE_K;
            int global_m = bx * TILE_M + local_m;
            int global_k = k_tile * TILE_K + local_k;

            As[local_m][local_k] = (global_m < M && global_k < K) ?
                                   A[global_m * K + global_k] : 0.0f;
        }

        for (int i = tx; i < TILE_K * TILE_N; i += BLOCK_SIZE) {
            int local_k = i / TILE_N;
            int local_n = i % TILE_N;
            int global_k = k_tile * TILE_K + local_k;
            int global_n = by * TILE_N + local_n;

            Bs[local_k][local_n] = (global_k < K && global_n < N) ?
                                   B[global_k * N + global_n] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            float a_val = As[row_in_tile][k];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                acc[j] += a_val * Bs[k][col_in_tile + j];
            }
        }

        __syncthreads();
    }

    if (c_row < M) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int global_col = c_col + j;
            if (global_col < N) {
                float result;
                if (is_sliding_window_valid(c_row, global_col, window_size)) {
                    result = alpha * acc[j];
                    if (beta != 0.0f) {
                        result += beta * C[c_row * N + global_col];
                    }
                } else {
                    result = mask_value;
                }
                C[c_row * N + global_col] = result;
            }
        }
    }
}

// ============================================================================
// Custom Mask GEMM - User-provided mask tensor
// ============================================================================

extern "C" __global__
void steel_gemm_custom_mask_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const bool* __restrict__ mask,  // [M, N] boolean mask (true = keep, false = mask)
    int M, int N, int K,
    float alpha, float beta,
    float mask_value)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;

    const int row_in_tile = tx / (TILE_N / 4);
    const int col_in_tile = (tx % (TILE_N / 4)) * 4;

    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    const int c_row = bx * TILE_M + row_in_tile;
    const int c_col = by * TILE_N + col_in_tile;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k_tile = 0; k_tile < (K + TILE_K - 1) / TILE_K; k_tile++) {
        for (int i = tx; i < TILE_M * TILE_K; i += BLOCK_SIZE) {
            int local_m = i / TILE_K;
            int local_k = i % TILE_K;
            int global_m = bx * TILE_M + local_m;
            int global_k = k_tile * TILE_K + local_k;

            As[local_m][local_k] = (global_m < M && global_k < K) ?
                                   A[global_m * K + global_k] : 0.0f;
        }

        for (int i = tx; i < TILE_K * TILE_N; i += BLOCK_SIZE) {
            int local_k = i / TILE_N;
            int local_n = i % TILE_N;
            int global_k = k_tile * TILE_K + local_k;
            int global_n = by * TILE_N + local_n;

            Bs[local_k][local_n] = (global_k < K && global_n < N) ?
                                   B[global_k * N + global_n] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            float a_val = As[row_in_tile][k];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                acc[j] += a_val * Bs[k][col_in_tile + j];
            }
        }

        __syncthreads();
    }

    if (c_row < M) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int global_col = c_col + j;
            if (global_col < N) {
                bool keep = mask[c_row * N + global_col];
                float result;
                if (keep) {
                    result = alpha * acc[j];
                    if (beta != 0.0f) {
                        result += beta * C[c_row * N + global_col];
                    }
                } else {
                    result = mask_value;
                }
                C[c_row * N + global_col] = result;
            }
        }
    }
}

// ============================================================================
// Block Sparse Masked GEMM - For efficient sparse attention patterns
// ============================================================================

extern "C" __global__
void steel_gemm_block_sparse_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int* __restrict__ block_pattern,  // [num_block_rows] - bitfield per row
    int M, int N, int K,
    float alpha, float beta,
    float mask_value,
    int sparse_block_size)         // Size of sparse blocks (e.g., 64)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;

    // Check if this block is in the sparse pattern
    int block_row = (bx * TILE_M) / sparse_block_size;
    int block_col = (by * TILE_N) / sparse_block_size;

    bool block_active = (block_pattern[block_row] >> block_col) & 1;

    const int row_in_tile = tx / (TILE_N / 4);
    const int col_in_tile = (tx % (TILE_N / 4)) * 4;

    const int c_row = bx * TILE_M + row_in_tile;
    const int c_col = by * TILE_N + col_in_tile;

    // If block is not active, fill with mask_value
    if (!block_active) {
        if (c_row < M) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                if (c_col + j < N) {
                    C[c_row * N + c_col + j] = mask_value;
                }
            }
        }
        return;
    }

    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k_tile = 0; k_tile < (K + TILE_K - 1) / TILE_K; k_tile++) {
        for (int i = tx; i < TILE_M * TILE_K; i += BLOCK_SIZE) {
            int local_m = i / TILE_K;
            int local_k = i % TILE_K;
            int global_m = bx * TILE_M + local_m;
            int global_k = k_tile * TILE_K + local_k;

            As[local_m][local_k] = (global_m < M && global_k < K) ?
                                   A[global_m * K + global_k] : 0.0f;
        }

        for (int i = tx; i < TILE_K * TILE_N; i += BLOCK_SIZE) {
            int local_k = i / TILE_N;
            int local_n = i % TILE_N;
            int global_k = k_tile * TILE_K + local_k;
            int global_n = by * TILE_N + local_n;

            Bs[local_k][local_n] = (global_k < K && global_n < N) ?
                                   B[global_k * N + global_n] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            float a_val = As[row_in_tile][k];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                acc[j] += a_val * Bs[k][col_in_tile + j];
            }
        }

        __syncthreads();
    }

    if (c_row < M) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (c_col + j < N) {
                float result = alpha * acc[j];
                if (beta != 0.0f) {
                    result += beta * C[c_row * N + c_col + j];
                }
                C[c_row * N + c_col + j] = result;
            }
        }
    }
}

// ============================================================================
// Attention Score Masked GEMM with Online Softmax Preparation
// ============================================================================

extern "C" __global__
void steel_gemm_attention_masked_kernel(
    const float* __restrict__ Q,    // [batch, heads, seq_q, head_dim]
    const float* __restrict__ K,    // [batch, heads, seq_k, head_dim]
    float* __restrict__ scores,     // [batch, heads, seq_q, seq_k]
    float* __restrict__ row_max,    // [batch, heads, seq_q] - for online softmax
    float* __restrict__ row_sum,    // [batch, heads, seq_q] - for online softmax
    int batch, int heads,
    int seq_q, int seq_k, int head_dim,
    float scale,
    int mask_type,                  // MASK_CAUSAL, MASK_SLIDING_WINDOW, etc.
    int mask_param)                 // causal_offset or window_size
{
    const int batch_idx = blockIdx.z / heads;
    const int head_idx = blockIdx.z % heads;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;

    const int row_in_tile = tx / (TILE_N / 4);
    const int col_in_tile = (tx % (TILE_N / 4)) * 4;

    __shared__ float Qs[TILE_M][TILE_K + 1];
    __shared__ float Ks[TILE_K][TILE_N + 1];
    __shared__ float tile_max[TILE_M];
    __shared__ float tile_sum[TILE_M];

    const int q_row = bx * TILE_M + row_in_tile;
    const int k_col = by * TILE_N + col_in_tile;

    // Base pointers for this batch/head
    const int qk_offset = (batch_idx * heads + head_idx) * seq_q * head_dim;
    const int kv_offset = (batch_idx * heads + head_idx) * seq_k * head_dim;
    const int out_offset = (batch_idx * heads + head_idx) * seq_q * seq_k;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Initialize tile max/sum
    if (tx < TILE_M) {
        tile_max[tx] = -FLT_MAX;
        tile_sum[tx] = 0.0f;
    }
    __syncthreads();

    // Compute Q @ K^T
    for (int k_tile = 0; k_tile < (head_dim + TILE_K - 1) / TILE_K; k_tile++) {
        // Load Q tile
        for (int i = tx; i < TILE_M * TILE_K; i += BLOCK_SIZE) {
            int local_m = i / TILE_K;
            int local_k = i % TILE_K;
            int global_m = bx * TILE_M + local_m;
            int global_k = k_tile * TILE_K + local_k;

            float val = 0.0f;
            if (global_m < seq_q && global_k < head_dim) {
                val = Q[qk_offset + global_m * head_dim + global_k];
            }
            Qs[local_m][local_k] = val;
        }

        // Load K tile (transposed access)
        for (int i = tx; i < TILE_K * TILE_N; i += BLOCK_SIZE) {
            int local_k = i / TILE_N;
            int local_n = i % TILE_N;
            int global_k = k_tile * TILE_K + local_k;
            int global_n = by * TILE_N + local_n;

            float val = 0.0f;
            if (global_k < head_dim && global_n < seq_k) {
                val = K[kv_offset + global_n * head_dim + global_k];
            }
            Ks[local_k][local_n] = val;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            float q_val = Qs[row_in_tile][k];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                acc[j] += q_val * Ks[k][col_in_tile + j];
            }
        }

        __syncthreads();
    }

    // Apply scale and mask
    float masked_vals[4];
    float local_max = -FLT_MAX;

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int global_col = k_col + j;
        bool valid = false;

        if (global_col < seq_k && q_row < seq_q) {
            switch (mask_type) {
                case MASK_CAUSAL:
                    valid = is_causal_valid(q_row, global_col, mask_param);
                    break;
                case MASK_SLIDING_WINDOW:
                    valid = is_sliding_window_valid(q_row, global_col, mask_param);
                    break;
                default:
                    valid = true;
                    break;
            }
        }

        if (valid) {
            masked_vals[j] = acc[j] * scale;
            local_max = fmaxf(local_max, masked_vals[j]);
        } else {
            masked_vals[j] = -FLT_MAX;
        }
    }

    // Warp-level reduction for max
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }

    // Update shared tile max
    if ((tx % 32) == 0) {
        atomicMax((int*)&tile_max[row_in_tile], __float_as_int(local_max));
    }
    __syncthreads();

    // Compute exp and sum
    float row_max_val = tile_max[row_in_tile];
    float local_sum = 0.0f;

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        if (masked_vals[j] > -FLT_MAX) {
            masked_vals[j] = expf(masked_vals[j] - row_max_val);
            local_sum += masked_vals[j];
        } else {
            masked_vals[j] = 0.0f;
        }
    }

    // Warp-level sum reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if ((tx % 32) == 0) {
        atomicAdd(&tile_sum[row_in_tile], local_sum);
    }
    __syncthreads();

    // Write scores and online softmax state
    if (q_row < seq_q) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (k_col + j < seq_k) {
                scores[out_offset + q_row * seq_k + k_col + j] = masked_vals[j];
            }
        }

        // First thread in row writes max/sum
        if (col_in_tile == 0 && by == 0) {
            int max_sum_offset = (batch_idx * heads + head_idx) * seq_q + q_row;
            row_max[max_sum_offset] = tile_max[row_in_tile];
            row_sum[max_sum_offset] = tile_sum[row_in_tile];
        }
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_steel_gemm_causal_mask(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta,
    float mask_value, int causal_offset,
    cudaStream_t stream)
{
    dim3 block(BLOCK_SIZE);
    dim3 grid(
        (M + TILE_M - 1) / TILE_M,
        (N + TILE_N - 1) / TILE_N
    );

    steel_gemm_causal_mask_kernel<<<grid, block, 0, stream>>>(
        A, B, C, M, N, K, alpha, beta, mask_value, causal_offset
    );

    return cudaGetLastError();
}

int lux_cuda_steel_gemm_sliding_window(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta,
    float mask_value, int window_size,
    cudaStream_t stream)
{
    dim3 block(BLOCK_SIZE);
    dim3 grid(
        (M + TILE_M - 1) / TILE_M,
        (N + TILE_N - 1) / TILE_N
    );

    steel_gemm_sliding_window_kernel<<<grid, block, 0, stream>>>(
        A, B, C, M, N, K, alpha, beta, mask_value, window_size
    );

    return cudaGetLastError();
}

int lux_cuda_steel_gemm_custom_mask(
    const float* A, const float* B, float* C,
    const bool* mask,
    int M, int N, int K,
    float alpha, float beta,
    float mask_value,
    cudaStream_t stream)
{
    dim3 block(BLOCK_SIZE);
    dim3 grid(
        (M + TILE_M - 1) / TILE_M,
        (N + TILE_N - 1) / TILE_N
    );

    steel_gemm_custom_mask_kernel<<<grid, block, 0, stream>>>(
        A, B, C, mask, M, N, K, alpha, beta, mask_value
    );

    return cudaGetLastError();
}

int lux_cuda_steel_gemm_block_sparse(
    const float* A, const float* B, float* C,
    const int* block_pattern,
    int M, int N, int K,
    float alpha, float beta,
    float mask_value, int sparse_block_size,
    cudaStream_t stream)
{
    dim3 block(BLOCK_SIZE);
    dim3 grid(
        (M + TILE_M - 1) / TILE_M,
        (N + TILE_N - 1) / TILE_N
    );

    steel_gemm_block_sparse_kernel<<<grid, block, 0, stream>>>(
        A, B, C, block_pattern, M, N, K, alpha, beta, mask_value, sparse_block_size
    );

    return cudaGetLastError();
}

int lux_cuda_steel_gemm_attention_masked(
    const float* Q, const float* K, float* scores,
    float* row_max, float* row_sum,
    int batch, int heads,
    int seq_q, int seq_k, int head_dim,
    float scale,
    int mask_type, int mask_param,
    cudaStream_t stream)
{
    dim3 block(BLOCK_SIZE);
    dim3 grid(
        (seq_q + TILE_M - 1) / TILE_M,
        (seq_k + TILE_N - 1) / TILE_N,
        batch * heads
    );

    steel_gemm_attention_masked_kernel<<<grid, block, 0, stream>>>(
        Q, K, scores, row_max, row_sum,
        batch, heads, seq_q, seq_k, head_dim,
        scale, mask_type, mask_param
    );

    return cudaGetLastError();
}

}  // extern "C"
