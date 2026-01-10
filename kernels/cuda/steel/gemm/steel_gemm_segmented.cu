// Copyright 2025 Lux Industries. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Steel Segmented GEMM Kernels - Batched and segmented matrix multiplication
// Implements variable-size batched GEMM, grouped GEMM, and ragged tensor operations

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// Configuration
// ============================================================================

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define BLOCK_SIZE 256

// ============================================================================
// Batched GEMM - Fixed size batches
// ============================================================================

extern "C" __global__
void steel_gemm_batched_kernel(
    const float* __restrict__ A,      // [batch, M, K]
    const float* __restrict__ B,      // [batch, K, N]
    float* __restrict__ C,            // [batch, M, N]
    int batch_size,
    int M, int N, int K,
    float alpha, float beta)
{
    const int batch_idx = blockIdx.z;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const int row_in_tile = tx / (TILE_N / 4);
    const int col_in_tile = (tx % (TILE_N / 4)) * 4;

    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    const int c_row = bx * TILE_M + row_in_tile;
    const int c_col = by * TILE_N + col_in_tile;

    // Base offsets for this batch
    const int a_batch_offset = batch_idx * M * K;
    const int b_batch_offset = batch_idx * K * N;
    const int c_batch_offset = batch_idx * M * N;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k_tile = 0; k_tile < (K + TILE_K - 1) / TILE_K; k_tile++) {
        // Load A tile
        for (int i = tx; i < TILE_M * TILE_K; i += BLOCK_SIZE) {
            int local_m = i / TILE_K;
            int local_k = i % TILE_K;
            int global_m = bx * TILE_M + local_m;
            int global_k = k_tile * TILE_K + local_k;

            As[local_m][local_k] = (global_m < M && global_k < K) ?
                A[a_batch_offset + global_m * K + global_k] : 0.0f;
        }

        // Load B tile
        for (int i = tx; i < TILE_K * TILE_N; i += BLOCK_SIZE) {
            int local_k = i / TILE_N;
            int local_n = i % TILE_N;
            int global_k = k_tile * TILE_K + local_k;
            int global_n = by * TILE_N + local_n;

            Bs[local_k][local_n] = (global_k < K && global_n < N) ?
                B[b_batch_offset + global_k * N + global_n] : 0.0f;
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

    // Write output
    if (c_row < M) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (c_col + j < N) {
                float result = alpha * acc[j];
                if (beta != 0.0f) {
                    result += beta * C[c_batch_offset + c_row * N + c_col + j];
                }
                C[c_batch_offset + c_row * N + c_col + j] = result;
            }
        }
    }
}

// ============================================================================
// Strided Batched GEMM - Batches with custom strides
// ============================================================================

extern "C" __global__
void steel_gemm_strided_batched_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M, int N, int K,
    long long stride_a,           // Stride between A matrices
    long long stride_b,           // Stride between B matrices
    long long stride_c,           // Stride between C matrices
    float alpha, float beta)
{
    const int batch_idx = blockIdx.z;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const int row_in_tile = tx / (TILE_N / 4);
    const int col_in_tile = (tx % (TILE_N / 4)) * 4;

    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    const int c_row = bx * TILE_M + row_in_tile;
    const int c_col = by * TILE_N + col_in_tile;

    // Strided offsets
    const float* A_batch = A + batch_idx * stride_a;
    const float* B_batch = B + batch_idx * stride_b;
    float* C_batch = C + batch_idx * stride_c;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k_tile = 0; k_tile < (K + TILE_K - 1) / TILE_K; k_tile++) {
        for (int i = tx; i < TILE_M * TILE_K; i += BLOCK_SIZE) {
            int local_m = i / TILE_K;
            int local_k = i % TILE_K;
            int global_m = bx * TILE_M + local_m;
            int global_k = k_tile * TILE_K + local_k;

            As[local_m][local_k] = (global_m < M && global_k < K) ?
                A_batch[global_m * K + global_k] : 0.0f;
        }

        for (int i = tx; i < TILE_K * TILE_N; i += BLOCK_SIZE) {
            int local_k = i / TILE_N;
            int local_n = i % TILE_N;
            int global_k = k_tile * TILE_K + local_k;
            int global_n = by * TILE_N + local_n;

            Bs[local_k][local_n] = (global_k < K && global_n < N) ?
                B_batch[global_k * N + global_n] : 0.0f;
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
                    result += beta * C_batch[c_row * N + c_col + j];
                }
                C_batch[c_row * N + c_col + j] = result;
            }
        }
    }
}

// ============================================================================
// Variable Size Batched GEMM - Each batch can have different M, N, K
// ============================================================================

extern "C" __global__
void steel_gemm_variable_batched_kernel(
    const float* const* __restrict__ A_ptrs,   // [batch] array of pointers
    const float* const* __restrict__ B_ptrs,   // [batch] array of pointers
    float* const* __restrict__ C_ptrs,         // [batch] array of pointers
    const int* __restrict__ M_array,           // [batch] M dimensions
    const int* __restrict__ N_array,           // [batch] N dimensions
    const int* __restrict__ K_array,           // [batch] K dimensions
    int batch_size,
    float alpha, float beta)
{
    const int batch_idx = blockIdx.z;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;

    if (batch_idx >= batch_size) return;

    // Get dimensions for this batch
    const int M = M_array[batch_idx];
    const int N = N_array[batch_idx];
    const int K = K_array[batch_idx];

    const int row_in_tile = tx / (TILE_N / 4);
    const int col_in_tile = (tx % (TILE_N / 4)) * 4;

    const int c_row = bx * TILE_M + row_in_tile;
    const int c_col = by * TILE_N + col_in_tile;

    // Early exit if this block is out of bounds for this batch
    if (bx * TILE_M >= M || by * TILE_N >= N) return;

    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    const float* A = A_ptrs[batch_idx];
    const float* B = B_ptrs[batch_idx];
    float* C = C_ptrs[batch_idx];

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
// Grouped GEMM - Multiple independent GEMMs with different sizes
// ============================================================================

extern "C" __global__
void steel_gemm_grouped_kernel(
    const float* __restrict__ A,        // Packed A matrices
    const float* __restrict__ B,        // Packed B matrices
    float* __restrict__ C,              // Packed C matrices
    const int* __restrict__ group_offsets_a,  // [num_groups + 1]
    const int* __restrict__ group_offsets_b,  // [num_groups + 1]
    const int* __restrict__ group_offsets_c,  // [num_groups + 1]
    const int* __restrict__ M_array,    // [num_groups]
    const int* __restrict__ N_array,    // [num_groups]
    const int* __restrict__ K_array,    // [num_groups]
    int num_groups,
    float alpha, float beta)
{
    // Find which group this block belongs to
    const int global_block_idx = blockIdx.z;

    // Binary search to find group
    int group_idx = 0;
    int cumulative_blocks = 0;

    for (int g = 0; g < num_groups; g++) {
        int M_g = M_array[g];
        int N_g = N_array[g];
        int blocks_in_group = ((M_g + TILE_M - 1) / TILE_M) * ((N_g + TILE_N - 1) / TILE_N);

        if (global_block_idx < cumulative_blocks + blocks_in_group) {
            group_idx = g;
            break;
        }
        cumulative_blocks += blocks_in_group;
    }

    const int M = M_array[group_idx];
    const int N = N_array[group_idx];
    const int K = K_array[group_idx];

    // Get block indices within this group
    int local_block_idx = global_block_idx - cumulative_blocks;
    int blocks_n = (N + TILE_N - 1) / TILE_N;
    int bx = local_block_idx / blocks_n;
    int by = local_block_idx % blocks_n;
    int tx = threadIdx.x;

    const int row_in_tile = tx / (TILE_N / 4);
    const int col_in_tile = (tx % (TILE_N / 4)) * 4;

    const int c_row = bx * TILE_M + row_in_tile;
    const int c_col = by * TILE_N + col_in_tile;

    if (c_row >= M || c_col >= N) return;

    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    // Get pointers for this group
    const float* A_g = A + group_offsets_a[group_idx];
    const float* B_g = B + group_offsets_b[group_idx];
    float* C_g = C + group_offsets_c[group_idx];

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k_tile = 0; k_tile < (K + TILE_K - 1) / TILE_K; k_tile++) {
        for (int i = tx; i < TILE_M * TILE_K; i += BLOCK_SIZE) {
            int local_m = i / TILE_K;
            int local_k = i % TILE_K;
            int global_m = bx * TILE_M + local_m;
            int global_k = k_tile * TILE_K + local_k;

            As[local_m][local_k] = (global_m < M && global_k < K) ?
                A_g[global_m * K + global_k] : 0.0f;
        }

        for (int i = tx; i < TILE_K * TILE_N; i += BLOCK_SIZE) {
            int local_k = i / TILE_N;
            int local_n = i % TILE_N;
            int global_k = k_tile * TILE_K + local_k;
            int global_n = by * TILE_N + local_n;

            Bs[local_k][local_n] = (global_k < K && global_n < N) ?
                B_g[global_k * N + global_n] : 0.0f;
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
                    result += beta * C_g[c_row * N + c_col + j];
                }
                C_g[c_row * N + c_col + j] = result;
            }
        }
    }
}

// ============================================================================
// Segmented Reduction GEMM - GEMM with segment-wise accumulation
// ============================================================================

extern "C" __global__
void steel_gemm_segmented_reduce_kernel(
    const float* __restrict__ A,         // [total_rows, K]
    const float* __restrict__ B,         // [K, N]
    float* __restrict__ C,               // [num_segments, N]
    const int* __restrict__ segment_ids, // [total_rows] - segment ID for each row
    int total_rows, int N, int K,
    int num_segments,
    float alpha)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;

    const int row_in_tile = tx / (TILE_N / 4);
    const int col_in_tile = (tx % (TILE_N / 4)) * 4;

    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];
    __shared__ int seg_ids[TILE_M];

    const int a_row = bx * TILE_M + row_in_tile;
    const int c_col = by * TILE_N + col_in_tile;

    // Cache segment IDs
    if (tx < TILE_M) {
        int global_row = bx * TILE_M + tx;
        seg_ids[tx] = (global_row < total_rows) ? segment_ids[global_row] : -1;
    }
    __syncthreads();

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k_tile = 0; k_tile < (K + TILE_K - 1) / TILE_K; k_tile++) {
        for (int i = tx; i < TILE_M * TILE_K; i += BLOCK_SIZE) {
            int local_m = i / TILE_K;
            int local_k = i % TILE_K;
            int global_m = bx * TILE_M + local_m;
            int global_k = k_tile * TILE_K + local_k;

            As[local_m][local_k] = (global_m < total_rows && global_k < K) ?
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

    // Atomic add to segment output
    if (a_row < total_rows) {
        int seg_id = seg_ids[row_in_tile];
        if (seg_id >= 0 && seg_id < num_segments) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                if (c_col + j < N) {
                    atomicAdd(&C[seg_id * N + c_col + j], alpha * acc[j]);
                }
            }
        }
    }
}

// ============================================================================
// Ragged Batched GEMM - For variable length sequences
// ============================================================================

extern "C" __global__
void steel_gemm_ragged_kernel(
    const float* __restrict__ A,        // [total_seq_len, K]
    const float* __restrict__ B,        // [K, N]
    float* __restrict__ C,              // [total_seq_len, N]
    const int* __restrict__ seq_offsets, // [batch + 1] - cumulative sequence lengths
    int batch_size, int N, int K,
    float alpha, float beta)
{
    const int batch_idx = blockIdx.z;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;

    if (batch_idx >= batch_size) return;

    // Get sequence bounds for this batch
    const int seq_start = seq_offsets[batch_idx];
    const int seq_end = seq_offsets[batch_idx + 1];
    const int seq_len = seq_end - seq_start;

    const int row_in_tile = tx / (TILE_N / 4);
    const int col_in_tile = (tx % (TILE_N / 4)) * 4;

    const int local_row = bx * TILE_M + row_in_tile;
    const int global_row = seq_start + local_row;
    const int c_col = by * TILE_N + col_in_tile;

    // Early exit if out of bounds
    if (local_row >= seq_len) return;

    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k_tile = 0; k_tile < (K + TILE_K - 1) / TILE_K; k_tile++) {
        for (int i = tx; i < TILE_M * TILE_K; i += BLOCK_SIZE) {
            int local_m = i / TILE_K;
            int local_k = i % TILE_K;
            int global_m = seq_start + bx * TILE_M + local_m;
            int global_k = k_tile * TILE_K + local_k;

            float val = 0.0f;
            if (bx * TILE_M + local_m < seq_len && global_k < K) {
                val = A[global_m * K + global_k];
            }
            As[local_m][local_k] = val;
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

    if (local_row < seq_len) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (c_col + j < N) {
                float result = alpha * acc[j];
                if (beta != 0.0f) {
                    result += beta * C[global_row * N + c_col + j];
                }
                C[global_row * N + c_col + j] = result;
            }
        }
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_steel_gemm_batched(
    const float* A, const float* B, float* C,
    int batch_size, int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    dim3 block(BLOCK_SIZE);
    dim3 grid(
        (M + TILE_M - 1) / TILE_M,
        (N + TILE_N - 1) / TILE_N,
        batch_size
    );

    steel_gemm_batched_kernel<<<grid, block, 0, stream>>>(
        A, B, C, batch_size, M, N, K, alpha, beta
    );

    return cudaGetLastError();
}

int lux_cuda_steel_gemm_strided_batched(
    const float* A, const float* B, float* C,
    int batch_size, int M, int N, int K,
    long long stride_a, long long stride_b, long long stride_c,
    float alpha, float beta,
    cudaStream_t stream)
{
    dim3 block(BLOCK_SIZE);
    dim3 grid(
        (M + TILE_M - 1) / TILE_M,
        (N + TILE_N - 1) / TILE_N,
        batch_size
    );

    steel_gemm_strided_batched_kernel<<<grid, block, 0, stream>>>(
        A, B, C, batch_size, M, N, K,
        stride_a, stride_b, stride_c,
        alpha, beta
    );

    return cudaGetLastError();
}

int lux_cuda_steel_gemm_variable_batched(
    const float* const* A_ptrs, const float* const* B_ptrs, float* const* C_ptrs,
    const int* M_array, const int* N_array, const int* K_array,
    int batch_size,
    float alpha, float beta,
    cudaStream_t stream)
{
    // Launch with maximum possible grid size
    // Each batch element handles its own bounds checking
    int max_M = 0, max_N = 0;
    // Note: In production, these should be passed in or computed on GPU
    // For simplicity, we use a reasonable max
    max_M = 4096;  // Assume max dimensions
    max_N = 4096;

    dim3 block(BLOCK_SIZE);
    dim3 grid(
        (max_M + TILE_M - 1) / TILE_M,
        (max_N + TILE_N - 1) / TILE_N,
        batch_size
    );

    steel_gemm_variable_batched_kernel<<<grid, block, 0, stream>>>(
        A_ptrs, B_ptrs, C_ptrs,
        M_array, N_array, K_array,
        batch_size, alpha, beta
    );

    return cudaGetLastError();
}

int lux_cuda_steel_gemm_grouped(
    const float* A, const float* B, float* C,
    const int* group_offsets_a, const int* group_offsets_b, const int* group_offsets_c,
    const int* M_array, const int* N_array, const int* K_array,
    int num_groups, int total_blocks,
    float alpha, float beta,
    cudaStream_t stream)
{
    dim3 block(BLOCK_SIZE);
    dim3 grid(1, 1, total_blocks);  // Flat grid, each block finds its group

    steel_gemm_grouped_kernel<<<grid, block, 0, stream>>>(
        A, B, C,
        group_offsets_a, group_offsets_b, group_offsets_c,
        M_array, N_array, K_array,
        num_groups, alpha, beta
    );

    return cudaGetLastError();
}

int lux_cuda_steel_gemm_segmented_reduce(
    const float* A, const float* B, float* C,
    const int* segment_ids,
    int total_rows, int N, int K,
    int num_segments,
    float alpha,
    cudaStream_t stream)
{
    // Zero output first (segment reduction uses atomicAdd)
    cudaMemsetAsync(C, 0, num_segments * N * sizeof(float), stream);

    dim3 block(BLOCK_SIZE);
    dim3 grid(
        (total_rows + TILE_M - 1) / TILE_M,
        (N + TILE_N - 1) / TILE_N
    );

    steel_gemm_segmented_reduce_kernel<<<grid, block, 0, stream>>>(
        A, B, C, segment_ids,
        total_rows, N, K, num_segments,
        alpha
    );

    return cudaGetLastError();
}

int lux_cuda_steel_gemm_ragged(
    const float* A, const float* B, float* C,
    const int* seq_offsets,
    int batch_size, int max_seq_len, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    dim3 block(BLOCK_SIZE);
    dim3 grid(
        (max_seq_len + TILE_M - 1) / TILE_M,
        (N + TILE_N - 1) / TILE_N,
        batch_size
    );

    steel_gemm_ragged_kernel<<<grid, block, 0, stream>>>(
        A, B, C, seq_offsets,
        batch_size, N, K,
        alpha, beta
    );

    return cudaGetLastError();
}

}  // extern "C"
