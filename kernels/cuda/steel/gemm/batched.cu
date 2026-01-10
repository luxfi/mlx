// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Batched GEMM CUDA Kernels
// Efficient batched matrix multiplication for transformer workloads

#include <cstdint>
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
// Batched GEMM (Strided)
// ============================================================================

// C[b] = alpha * A[b] @ B[b] + beta * C[b]
// Tensors have uniform strides between batches

extern "C" __global__
void steel_batched_gemm_kernel(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t batch_size,
    uint32_t stride_a,
    uint32_t stride_b,
    uint32_t stride_c
) {
    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    uint32_t batch = blockIdx.z;
    uint32_t bx = blockIdx.x;
    uint32_t by = blockIdx.y;
    uint32_t tid = threadIdx.x;

    if (batch >= batch_size) return;

    // Batch offsets
    const float* A_batch = A + batch * stride_a;
    const float* B_batch = B + batch * stride_b;
    float* C_batch = C + batch * stride_c;

    // Thread block tile
    uint32_t row_base = by * TILE_M;
    uint32_t col_base = bx * TILE_N;

    // Thread tile (each thread computes 4x4 output)
    uint32_t thread_row = (tid / (TILE_N / 4)) * 4;
    uint32_t thread_col = (tid % (TILE_N / 4)) * 4;

    // Accumulators
    float acc[4][4] = {{0.0f}};

    // K dimension tiles
    uint32_t num_k_tiles = (K + TILE_K - 1) / TILE_K;

    for (uint32_t kt = 0; kt < num_k_tiles; kt++) {
        uint32_t k_start = kt * TILE_K;

        // Load A tile
        for (uint32_t i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
            uint32_t ti = i / TILE_K;
            uint32_t tk = i % TILE_K;
            uint32_t a_row = row_base + ti;
            uint32_t a_col = k_start + tk;

            As[ti][tk] = (a_row < M && a_col < K) ?
                A_batch[a_row * K + a_col] : 0.0f;
        }

        // Load B tile
        for (uint32_t i = tid; i < TILE_K * TILE_N; i += blockDim.x) {
            uint32_t tk = i / TILE_N;
            uint32_t tj = i % TILE_N;
            uint32_t b_row = k_start + tk;
            uint32_t b_col = col_base + tj;

            Bs[tk][tj] = (b_row < K && b_col < N) ?
                B_batch[b_row * N + b_col] : 0.0f;
        }
        __syncthreads();

        // Compute
        #pragma unroll
        for (uint32_t k = 0; k < TILE_K; k++) {
            float a_frag[4], b_frag[4];

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                a_frag[i] = As[thread_row + i][k];
                b_frag[i] = Bs[k][thread_col + i];
            }

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    acc[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }
        __syncthreads();
    }

    // Write results
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t row = row_base + thread_row + i;
        if (row >= M) continue;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t col = col_base + thread_col + j;
            if (col >= N) continue;

            float result = alpha * acc[i][j];
            if (beta != 0.0f) {
                result += beta * C_batch[row * N + col];
            }
            C_batch[row * N + col] = result;
        }
    }
}

// ============================================================================
// Batched GEMM with Variable Batch Strides
// ============================================================================

// Each batch can have different pointers (array of pointers)

extern "C" __global__
void steel_batched_gemm_array_kernel(
    float* const* __restrict__ C_array,
    const float* const* __restrict__ A_array,
    const float* const* __restrict__ B_array,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t batch_size
) {
    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    uint32_t batch = blockIdx.z;
    uint32_t bx = blockIdx.x;
    uint32_t by = blockIdx.y;
    uint32_t tid = threadIdx.x;

    if (batch >= batch_size) return;

    // Get batch pointers
    const float* A_batch = A_array[batch];
    const float* B_batch = B_array[batch];
    float* C_batch = C_array[batch];

    uint32_t row_base = by * TILE_M;
    uint32_t col_base = bx * TILE_N;

    uint32_t thread_row = (tid / (TILE_N / 4)) * 4;
    uint32_t thread_col = (tid % (TILE_N / 4)) * 4;

    float acc[4][4] = {{0.0f}};

    uint32_t num_k_tiles = (K + TILE_K - 1) / TILE_K;

    for (uint32_t kt = 0; kt < num_k_tiles; kt++) {
        uint32_t k_start = kt * TILE_K;

        for (uint32_t i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
            uint32_t ti = i / TILE_K;
            uint32_t tk = i % TILE_K;
            uint32_t a_row = row_base + ti;
            uint32_t a_col = k_start + tk;

            As[ti][tk] = (a_row < M && a_col < K) ?
                A_batch[a_row * K + a_col] : 0.0f;
        }

        for (uint32_t i = tid; i < TILE_K * TILE_N; i += blockDim.x) {
            uint32_t tk = i / TILE_N;
            uint32_t tj = i % TILE_N;
            uint32_t b_row = k_start + tk;
            uint32_t b_col = col_base + tj;

            Bs[tk][tj] = (b_row < K && b_col < N) ?
                B_batch[b_row * N + b_col] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (uint32_t k = 0; k < TILE_K; k++) {
            float a_frag[4], b_frag[4];

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                a_frag[i] = As[thread_row + i][k];
                b_frag[i] = Bs[k][thread_col + i];
            }

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    acc[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t row = row_base + thread_row + i;
        if (row >= M) continue;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t col = col_base + thread_col + j;
            if (col >= N) continue;

            float result = alpha * acc[i][j];
            if (beta != 0.0f) {
                result += beta * C_batch[row * N + col];
            }
            C_batch[row * N + col] = result;
        }
    }
}

// ============================================================================
// Batched GEMM with Different Sizes per Batch
// ============================================================================

// For variable-size batched operations (padding handled externally)

extern "C" __global__
void steel_batched_gemm_variable_kernel(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const uint32_t* __restrict__ M_array,  // [batch_size]
    const uint32_t* __restrict__ N_array,  // [batch_size]
    const uint32_t* __restrict__ K_array,  // [batch_size]
    const uint32_t* __restrict__ offset_a, // [batch_size]
    const uint32_t* __restrict__ offset_b, // [batch_size]
    const uint32_t* __restrict__ offset_c, // [batch_size]
    float alpha,
    float beta,
    uint32_t batch_size,
    uint32_t max_M,
    uint32_t max_N
) {
    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    uint32_t batch = blockIdx.z;
    uint32_t bx = blockIdx.x;
    uint32_t by = blockIdx.y;
    uint32_t tid = threadIdx.x;

    if (batch >= batch_size) return;

    // Get batch dimensions
    uint32_t M = M_array[batch];
    uint32_t N = N_array[batch];
    uint32_t K = K_array[batch];

    // Get batch pointers
    const float* A_batch = A + offset_a[batch];
    const float* B_batch = B + offset_b[batch];
    float* C_batch = C + offset_c[batch];

    uint32_t row_base = by * TILE_M;
    uint32_t col_base = bx * TILE_N;

    // Early exit if this block is outside matrix bounds
    if (row_base >= M || col_base >= N) return;

    uint32_t thread_row = (tid / (TILE_N / 4)) * 4;
    uint32_t thread_col = (tid % (TILE_N / 4)) * 4;

    float acc[4][4] = {{0.0f}};

    uint32_t num_k_tiles = (K + TILE_K - 1) / TILE_K;

    for (uint32_t kt = 0; kt < num_k_tiles; kt++) {
        uint32_t k_start = kt * TILE_K;

        for (uint32_t i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
            uint32_t ti = i / TILE_K;
            uint32_t tk = i % TILE_K;
            uint32_t a_row = row_base + ti;
            uint32_t a_col = k_start + tk;

            As[ti][tk] = (a_row < M && a_col < K) ?
                A_batch[a_row * K + a_col] : 0.0f;
        }

        for (uint32_t i = tid; i < TILE_K * TILE_N; i += blockDim.x) {
            uint32_t tk = i / TILE_N;
            uint32_t tj = i % TILE_N;
            uint32_t b_row = k_start + tk;
            uint32_t b_col = col_base + tj;

            Bs[tk][tj] = (b_row < K && b_col < N) ?
                B_batch[b_row * N + b_col] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (uint32_t k = 0; k < TILE_K; k++) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    acc[i][j] += As[thread_row + i][k] * Bs[k][thread_col + j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t row = row_base + thread_row + i;
        if (row >= M) continue;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t col = col_base + thread_col + j;
            if (col >= N) continue;

            float result = alpha * acc[i][j];
            if (beta != 0.0f) {
                result += beta * C_batch[row * N + col];
            }
            C_batch[row * N + col] = result;
        }
    }
}

// ============================================================================
// Batched GEMM + Transpose Combinations
// ============================================================================

// C = alpha * A^T @ B + beta * C

extern "C" __global__
void steel_batched_gemm_tn_kernel(
    float* __restrict__ C,
    const float* __restrict__ A,       // [batch, K, M] stored, accessed as A^T
    const float* __restrict__ B,       // [batch, K, N]
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t batch_size,
    uint32_t stride_a,
    uint32_t stride_b,
    uint32_t stride_c
) {
    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    uint32_t batch = blockIdx.z;
    uint32_t bx = blockIdx.x;
    uint32_t by = blockIdx.y;
    uint32_t tid = threadIdx.x;

    if (batch >= batch_size) return;

    const float* A_batch = A + batch * stride_a;
    const float* B_batch = B + batch * stride_b;
    float* C_batch = C + batch * stride_c;

    uint32_t row_base = by * TILE_M;
    uint32_t col_base = bx * TILE_N;

    uint32_t thread_row = (tid / (TILE_N / 4)) * 4;
    uint32_t thread_col = (tid % (TILE_N / 4)) * 4;

    float acc[4][4] = {{0.0f}};

    uint32_t num_k_tiles = (K + TILE_K - 1) / TILE_K;

    for (uint32_t kt = 0; kt < num_k_tiles; kt++) {
        uint32_t k_start = kt * TILE_K;

        // Load A^T (stored as [K, M], access as [M, K])
        for (uint32_t i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
            uint32_t ti = i / TILE_K;
            uint32_t tk = i % TILE_K;
            uint32_t a_row = row_base + ti;  // M dimension
            uint32_t a_col = k_start + tk;   // K dimension

            // A is stored [K, M], so access [k, m]
            As[ti][tk] = (a_row < M && a_col < K) ?
                A_batch[a_col * M + a_row] : 0.0f;
        }

        for (uint32_t i = tid; i < TILE_K * TILE_N; i += blockDim.x) {
            uint32_t tk = i / TILE_N;
            uint32_t tj = i % TILE_N;
            uint32_t b_row = k_start + tk;
            uint32_t b_col = col_base + tj;

            Bs[tk][tj] = (b_row < K && b_col < N) ?
                B_batch[b_row * N + b_col] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (uint32_t k = 0; k < TILE_K; k++) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    acc[i][j] += As[thread_row + i][k] * Bs[k][thread_col + j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t row = row_base + thread_row + i;
        if (row >= M) continue;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t col = col_base + thread_col + j;
            if (col >= N) continue;

            float result = alpha * acc[i][j];
            if (beta != 0.0f) {
                result += beta * C_batch[row * N + col];
            }
            C_batch[row * N + col] = result;
        }
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_steel_batched_gemm(
    void* C,
    const void* A,
    const void* B,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t batch_size,
    uint32_t stride_a,
    uint32_t stride_b,
    uint32_t stride_c,
    cudaStream_t stream
) {
    dim3 blocks((N + TILE_N - 1) / TILE_N,
                (M + TILE_M - 1) / TILE_M,
                batch_size);
    uint32_t threads = BLOCK_SIZE;

    steel_batched_gemm_kernel<<<blocks, threads, 0, stream>>>(
        (float*)C,
        (const float*)A,
        (const float*)B,
        alpha, beta,
        M, N, K,
        batch_size, stride_a, stride_b, stride_c
    );

    return cudaGetLastError();
}

int lux_cuda_steel_batched_gemm_array(
    void* const* C_array,
    const void* const* A_array,
    const void* const* B_array,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t batch_size,
    cudaStream_t stream
) {
    dim3 blocks((N + TILE_N - 1) / TILE_N,
                (M + TILE_M - 1) / TILE_M,
                batch_size);
    uint32_t threads = BLOCK_SIZE;

    steel_batched_gemm_array_kernel<<<blocks, threads, 0, stream>>>(
        (float* const*)C_array,
        (const float* const*)A_array,
        (const float* const*)B_array,
        alpha, beta,
        M, N, K, batch_size
    );

    return cudaGetLastError();
}

int lux_cuda_steel_batched_gemm_tn(
    void* C,
    const void* A,
    const void* B,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t batch_size,
    uint32_t stride_a,
    uint32_t stride_b,
    uint32_t stride_c,
    cudaStream_t stream
) {
    dim3 blocks((N + TILE_N - 1) / TILE_N,
                (M + TILE_M - 1) / TILE_M,
                batch_size);
    uint32_t threads = BLOCK_SIZE;

    steel_batched_gemm_tn_kernel<<<blocks, threads, 0, stream>>>(
        (float*)C,
        (const float*)A,
        (const float*)B,
        alpha, beta,
        M, N, K,
        batch_size, stride_a, stride_b, stride_c
    );

    return cudaGetLastError();
}

}  // extern "C"
