// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Steel Gather GEMM CUDA Kernels
// GEMM with gather/scatter operations for sparse and indexed computation

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// Gather GEMM Configuration
// ============================================================================

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define BLOCK_SIZE 256

// ============================================================================
// Gather A GEMM
// C = A[indices_a] @ B, where indices_a selects rows from A
// ============================================================================

extern "C" __global__
void steel_gemm_gather_a_kernel(
    float* __restrict__ C,
    const float* __restrict__ A,         // [total_rows_a, K]
    const float* __restrict__ B,         // [K, N]
    const int32_t* __restrict__ indices, // [M] indices into A
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t total_rows_a
) {
    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    uint32_t bx = blockIdx.x;
    uint32_t by = blockIdx.y;
    uint32_t tid = threadIdx.x;

    uint32_t thread_row = (tid / (TILE_N / 4)) * 4;
    uint32_t thread_col = (tid % (TILE_N / 4)) * 4;

    uint32_t row_base = by * TILE_M;
    uint32_t col_base = bx * TILE_N;

    float acc[4][4] = {{0.0f}};

    uint32_t num_k_tiles = (K + TILE_K - 1) / TILE_K;

    for (uint32_t kt = 0; kt < num_k_tiles; kt++) {
        uint32_t k_start = kt * TILE_K;

        // Load A tile with gather (indexed row access)
        for (uint32_t i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
            uint32_t ti = i / TILE_K;
            uint32_t tk = i % TILE_K;
            uint32_t logical_row = row_base + ti;
            uint32_t a_col = k_start + tk;

            if (logical_row < M && a_col < K) {
                int32_t physical_row = indices[logical_row];
                if (physical_row >= 0 && physical_row < (int32_t)total_rows_a) {
                    As[ti][tk] = A[physical_row * K + a_col];
                } else {
                    As[ti][tk] = 0.0f;
                }
            } else {
                As[ti][tk] = 0.0f;
            }
        }

        // Load B tile normally
        for (uint32_t i = tid; i < TILE_K * TILE_N; i += blockDim.x) {
            uint32_t tk = i / TILE_N;
            uint32_t tj = i % TILE_N;
            uint32_t b_row = k_start + tk;
            uint32_t b_col = col_base + tj;

            Bs[tk][tj] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
        }
        __syncthreads();

        // Compute
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

    // Write output
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t row = row_base + thread_row + i;
        if (row >= M) continue;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t col = col_base + thread_col + j;
            if (col >= N) continue;

            C[row * N + col] = acc[i][j];
        }
    }
}

// ============================================================================
// Gather B GEMM
// C = A @ B[indices_b], where indices_b selects columns from B
// ============================================================================

extern "C" __global__
void steel_gemm_gather_b_kernel(
    float* __restrict__ C,
    const float* __restrict__ A,         // [M, K]
    const float* __restrict__ B,         // [K, total_cols_b]
    const int32_t* __restrict__ indices, // [N] indices into B columns
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t total_cols_b
) {
    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    uint32_t bx = blockIdx.x;
    uint32_t by = blockIdx.y;
    uint32_t tid = threadIdx.x;

    uint32_t thread_row = (tid / (TILE_N / 4)) * 4;
    uint32_t thread_col = (tid % (TILE_N / 4)) * 4;

    uint32_t row_base = by * TILE_M;
    uint32_t col_base = bx * TILE_N;

    float acc[4][4] = {{0.0f}};

    uint32_t num_k_tiles = (K + TILE_K - 1) / TILE_K;

    for (uint32_t kt = 0; kt < num_k_tiles; kt++) {
        uint32_t k_start = kt * TILE_K;

        // Load A tile normally
        for (uint32_t i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
            uint32_t ti = i / TILE_K;
            uint32_t tk = i % TILE_K;
            uint32_t a_row = row_base + ti;
            uint32_t a_col = k_start + tk;

            As[ti][tk] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        }

        // Load B tile with gather (indexed column access)
        for (uint32_t i = tid; i < TILE_K * TILE_N; i += blockDim.x) {
            uint32_t tk = i / TILE_N;
            uint32_t tj = i % TILE_N;
            uint32_t b_row = k_start + tk;
            uint32_t logical_col = col_base + tj;

            if (b_row < K && logical_col < N) {
                int32_t physical_col = indices[logical_col];
                if (physical_col >= 0 && physical_col < (int32_t)total_cols_b) {
                    Bs[tk][tj] = B[b_row * total_cols_b + physical_col];
                } else {
                    Bs[tk][tj] = 0.0f;
                }
            } else {
                Bs[tk][tj] = 0.0f;
            }
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

            C[row * N + col] = acc[i][j];
        }
    }
}

// ============================================================================
// Scatter Add GEMM
// C[indices] += A @ B, scatter results to indexed positions
// ============================================================================

extern "C" __global__
void steel_gemm_scatter_kernel(
    float* __restrict__ C,               // [total_rows_c, N]
    const float* __restrict__ A,         // [M, K]
    const float* __restrict__ B,         // [K, N]
    const int32_t* __restrict__ indices, // [M] indices for output rows
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t total_rows_c
) {
    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    uint32_t bx = blockIdx.x;
    uint32_t by = blockIdx.y;
    uint32_t tid = threadIdx.x;

    uint32_t thread_row = (tid / (TILE_N / 4)) * 4;
    uint32_t thread_col = (tid % (TILE_N / 4)) * 4;

    uint32_t row_base = by * TILE_M;
    uint32_t col_base = bx * TILE_N;

    float acc[4][4] = {{0.0f}};

    uint32_t num_k_tiles = (K + TILE_K - 1) / TILE_K;

    for (uint32_t kt = 0; kt < num_k_tiles; kt++) {
        uint32_t k_start = kt * TILE_K;

        for (uint32_t i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
            uint32_t ti = i / TILE_K;
            uint32_t tk = i % TILE_K;
            uint32_t a_row = row_base + ti;
            uint32_t a_col = k_start + tk;

            As[ti][tk] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        }

        for (uint32_t i = tid; i < TILE_K * TILE_N; i += blockDim.x) {
            uint32_t tk = i / TILE_N;
            uint32_t tj = i % TILE_N;
            uint32_t b_row = k_start + tk;
            uint32_t b_col = col_base + tj;

            Bs[tk][tj] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
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

    // Scatter write with atomic add
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t logical_row = row_base + thread_row + i;
        if (logical_row >= M) continue;

        int32_t physical_row = indices[logical_row];
        if (physical_row < 0 || physical_row >= (int32_t)total_rows_c) continue;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t col = col_base + thread_col + j;
            if (col >= N) continue;

            atomicAdd(&C[physical_row * N + col], acc[i][j]);
        }
    }
}

// ============================================================================
// Embedding GEMM (Lookup Table + Linear)
// Output = Embedding[indices] @ W
// ============================================================================

extern "C" __global__
void steel_embedding_gemm_kernel(
    float* __restrict__ output,
    const float* __restrict__ embedding,  // [vocab_size, embed_dim]
    const float* __restrict__ W,          // [embed_dim, out_dim]
    const int32_t* __restrict__ indices,  // [batch_size]
    uint32_t batch_size,
    uint32_t embed_dim,
    uint32_t out_dim,
    uint32_t vocab_size
) {
    __shared__ float Es[TILE_M][TILE_K + 1];  // Embedding lookup results
    __shared__ float Ws[TILE_K][TILE_N + 1];

    uint32_t bx = blockIdx.x;
    uint32_t by = blockIdx.y;
    uint32_t tid = threadIdx.x;

    uint32_t thread_row = (tid / (TILE_N / 4)) * 4;
    uint32_t thread_col = (tid % (TILE_N / 4)) * 4;

    uint32_t row_base = by * TILE_M;
    uint32_t col_base = bx * TILE_N;

    float acc[4][4] = {{0.0f}};

    uint32_t num_k_tiles = (embed_dim + TILE_K - 1) / TILE_K;

    for (uint32_t kt = 0; kt < num_k_tiles; kt++) {
        uint32_t k_start = kt * TILE_K;

        // Load embedding lookup results
        for (uint32_t i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
            uint32_t ti = i / TILE_K;
            uint32_t tk = i % TILE_K;
            uint32_t batch_idx = row_base + ti;
            uint32_t embed_col = k_start + tk;

            if (batch_idx < batch_size && embed_col < embed_dim) {
                int32_t token_id = indices[batch_idx];
                if (token_id >= 0 && token_id < (int32_t)vocab_size) {
                    Es[ti][tk] = embedding[token_id * embed_dim + embed_col];
                } else {
                    Es[ti][tk] = 0.0f;
                }
            } else {
                Es[ti][tk] = 0.0f;
            }
        }

        // Load W tile
        for (uint32_t i = tid; i < TILE_K * TILE_N; i += blockDim.x) {
            uint32_t tk = i / TILE_N;
            uint32_t tj = i % TILE_N;
            uint32_t w_row = k_start + tk;
            uint32_t w_col = col_base + tj;

            Ws[tk][tj] = (w_row < embed_dim && w_col < out_dim) ? W[w_row * out_dim + w_col] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (uint32_t k = 0; k < TILE_K; k++) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    acc[i][j] += Es[thread_row + i][k] * Ws[k][thread_col + j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t row = row_base + thread_row + i;
        if (row >= batch_size) continue;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t col = col_base + thread_col + j;
            if (col >= out_dim) continue;

            output[row * out_dim + col] = acc[i][j];
        }
    }
}

// ============================================================================
// Batched Index GEMM
// C[b] = A[batch_indices_a[b]] @ B[batch_indices_b[b]]
// ============================================================================

extern "C" __global__
void steel_batched_index_gemm_kernel(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const int32_t* __restrict__ batch_indices_a,
    const int32_t* __restrict__ batch_indices_b,
    uint32_t batch_size,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    int64_t stride_a,
    int64_t stride_b,
    int64_t stride_c
) {
    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    uint32_t batch = blockIdx.z;
    uint32_t bx = blockIdx.x;
    uint32_t by = blockIdx.y;
    uint32_t tid = threadIdx.x;

    if (batch >= batch_size) return;

    // Get actual batch indices
    int32_t a_batch = batch_indices_a ? batch_indices_a[batch] : batch;
    int32_t b_batch = batch_indices_b ? batch_indices_b[batch] : batch;

    const float* A_batch = A + a_batch * stride_a;
    const float* B_batch = B + b_batch * stride_b;
    float* C_batch = C + batch * stride_c;

    uint32_t thread_row = (tid / (TILE_N / 4)) * 4;
    uint32_t thread_col = (tid % (TILE_N / 4)) * 4;

    uint32_t row_base = by * TILE_M;
    uint32_t col_base = bx * TILE_N;

    float acc[4][4] = {{0.0f}};

    uint32_t num_k_tiles = (K + TILE_K - 1) / TILE_K;

    for (uint32_t kt = 0; kt < num_k_tiles; kt++) {
        uint32_t k_start = kt * TILE_K;

        for (uint32_t i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
            uint32_t ti = i / TILE_K;
            uint32_t tk = i % TILE_K;
            uint32_t a_row = row_base + ti;
            uint32_t a_col = k_start + tk;

            As[ti][tk] = (a_row < M && a_col < K) ? A_batch[a_row * K + a_col] : 0.0f;
        }

        for (uint32_t i = tid; i < TILE_K * TILE_N; i += blockDim.x) {
            uint32_t tk = i / TILE_N;
            uint32_t tj = i % TILE_N;
            uint32_t b_row = k_start + tk;
            uint32_t b_col = col_base + tj;

            Bs[tk][tj] = (b_row < K && b_col < N) ? B_batch[b_row * N + b_col] : 0.0f;
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

            C_batch[row * N + col] = acc[i][j];
        }
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_steel_gemm_gather_a(
    void* C,
    const void* A,
    const void* B,
    const int32_t* indices,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t total_rows_a,
    cudaStream_t stream
) {
    dim3 blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    uint32_t threads = BLOCK_SIZE;

    steel_gemm_gather_a_kernel<<<blocks, threads, 0, stream>>>(
        (float*)C,
        (const float*)A,
        (const float*)B,
        indices,
        M, N, K, total_rows_a
    );

    return cudaGetLastError();
}

int lux_cuda_steel_gemm_gather_b(
    void* C,
    const void* A,
    const void* B,
    const int32_t* indices,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t total_cols_b,
    cudaStream_t stream
) {
    dim3 blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    uint32_t threads = BLOCK_SIZE;

    steel_gemm_gather_b_kernel<<<blocks, threads, 0, stream>>>(
        (float*)C,
        (const float*)A,
        (const float*)B,
        indices,
        M, N, K, total_cols_b
    );

    return cudaGetLastError();
}

int lux_cuda_steel_gemm_scatter(
    void* C,
    const void* A,
    const void* B,
    const int32_t* indices,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t total_rows_c,
    cudaStream_t stream
) {
    dim3 blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    uint32_t threads = BLOCK_SIZE;

    steel_gemm_scatter_kernel<<<blocks, threads, 0, stream>>>(
        (float*)C,
        (const float*)A,
        (const float*)B,
        indices,
        M, N, K, total_rows_c
    );

    return cudaGetLastError();
}

int lux_cuda_steel_embedding_gemm(
    void* output,
    const void* embedding,
    const void* W,
    const int32_t* indices,
    uint32_t batch_size,
    uint32_t embed_dim,
    uint32_t out_dim,
    uint32_t vocab_size,
    cudaStream_t stream
) {
    dim3 blocks((out_dim + TILE_N - 1) / TILE_N, (batch_size + TILE_M - 1) / TILE_M);
    uint32_t threads = BLOCK_SIZE;

    steel_embedding_gemm_kernel<<<blocks, threads, 0, stream>>>(
        (float*)output,
        (const float*)embedding,
        (const float*)W,
        indices,
        batch_size, embed_dim, out_dim, vocab_size
    );

    return cudaGetLastError();
}

int lux_cuda_steel_batched_index_gemm(
    void* C,
    const void* A,
    const void* B,
    const int32_t* batch_indices_a,
    const int32_t* batch_indices_b,
    uint32_t batch_size,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    int64_t stride_a,
    int64_t stride_b,
    int64_t stride_c,
    cudaStream_t stream
) {
    dim3 blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M, batch_size);
    uint32_t threads = BLOCK_SIZE;

    steel_batched_index_gemm_kernel<<<blocks, threads, 0, stream>>>(
        (float*)C,
        (const float*)A,
        (const float*)B,
        batch_indices_a,
        batch_indices_b,
        batch_size, M, N, K,
        stride_a, stride_b, stride_c
    );

    return cudaGetLastError();
}

}  // extern "C"
