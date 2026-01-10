// Copyright 2025 Lux Industries. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Steel GEMM Gather NAX Kernels - Non-blocking Asynchronous eXecution gather operations
// Implements pipelined gather GEMM with software prefetching and multi-stage buffering

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// Configuration
// ============================================================================

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define BLOCK_SIZE 256
#define NAX_STAGES 3

// ============================================================================
// NAX Gather A Kernel - Pipelined row gathering from A matrix
// ============================================================================

extern "C" __global__
void steel_gemm_gather_a_nax_kernel(
    const float* __restrict__ A,      // [K, M] or [M, K]
    const float* __restrict__ B,      // [K, N]
    float* __restrict__ C,            // [num_indices, N]
    const int* __restrict__ indices,  // [num_indices] - rows to gather from A
    int M, int N, int K,
    int num_indices,
    float alpha, float beta,
    bool A_transposed)
{
    // Block indices
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;

    // Thread position within tile
    const int row_in_tile = tx / (TILE_N / 4);
    const int col_in_tile = (tx % (TILE_N / 4)) * 4;

    // Triple-buffered shared memory for NAX pipelining
    __shared__ float As[NAX_STAGES][TILE_M][TILE_K + 1];
    __shared__ float Bs[NAX_STAGES][TILE_K][TILE_N + 1];

    // Output row in C
    const int c_row = bx * TILE_M + row_in_tile;
    const int c_col = by * TILE_N + col_in_tile;

    // Register accumulator
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Get the actual A row index from indices array
    int a_row = -1;
    if (c_row < num_indices) {
        a_row = indices[c_row];
    }

    // Number of K tiles
    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // Prefetch first stages
    #pragma unroll
    for (int stage = 0; stage < NAX_STAGES - 1 && stage < num_k_tiles; stage++) {
        const int k_offset = stage * TILE_K;

        // Load A tile with gather
        for (int i = tx; i < TILE_M * TILE_K; i += BLOCK_SIZE) {
            int local_m = i / TILE_K;
            int local_k = i % TILE_K;
            int global_m = bx * TILE_M + local_m;
            int global_k = k_offset + local_k;

            float val = 0.0f;
            if (global_m < num_indices && global_k < K) {
                int actual_row = indices[global_m];
                if (actual_row >= 0 && actual_row < M) {
                    if (A_transposed) {
                        val = A[global_k * M + actual_row];
                    } else {
                        val = A[actual_row * K + global_k];
                    }
                }
            }
            As[stage][local_m][local_k] = val;
        }

        // Load B tile
        for (int i = tx; i < TILE_K * TILE_N; i += BLOCK_SIZE) {
            int local_k = i / TILE_N;
            int local_n = i % TILE_N;
            int global_k = k_offset + local_k;
            int global_n = by * TILE_N + local_n;

            float val = 0.0f;
            if (global_k < K && global_n < N) {
                val = B[global_k * N + global_n];
            }
            Bs[stage][local_k][local_n] = val;
        }
    }

    __syncthreads();

    // Main NAX loop
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int curr_stage = k_tile % NAX_STAGES;
        const int next_stage = (k_tile + NAX_STAGES - 1) % NAX_STAGES;
        const int prefetch_tile = k_tile + NAX_STAGES - 1;

        // Prefetch next stage if available
        if (prefetch_tile < num_k_tiles) {
            const int k_offset = prefetch_tile * TILE_K;

            for (int i = tx; i < TILE_M * TILE_K; i += BLOCK_SIZE) {
                int local_m = i / TILE_K;
                int local_k = i % TILE_K;
                int global_m = bx * TILE_M + local_m;
                int global_k = k_offset + local_k;

                float val = 0.0f;
                if (global_m < num_indices && global_k < K) {
                    int actual_row = indices[global_m];
                    if (actual_row >= 0 && actual_row < M) {
                        if (A_transposed) {
                            val = A[global_k * M + actual_row];
                        } else {
                            val = A[actual_row * K + global_k];
                        }
                    }
                }
                As[next_stage][local_m][local_k] = val;
            }

            for (int i = tx; i < TILE_K * TILE_N; i += BLOCK_SIZE) {
                int local_k = i / TILE_N;
                int local_n = i % TILE_N;
                int global_k = k_offset + local_k;
                int global_n = by * TILE_N + local_n;

                float val = 0.0f;
                if (global_k < K && global_n < N) {
                    val = B[global_k * N + global_n];
                }
                Bs[next_stage][local_k][local_n] = val;
            }
        }

        // Compute on current stage
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            float a_val = As[curr_stage][row_in_tile][k];

            #pragma unroll
            for (int j = 0; j < 4; j++) {
                acc[j] += a_val * Bs[curr_stage][k][col_in_tile + j];
            }
        }

        __syncthreads();
    }

    // Write output
    if (c_row < num_indices) {
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
// NAX Gather B Kernel - Pipelined column gathering from B matrix
// ============================================================================

extern "C" __global__
void steel_gemm_gather_b_nax_kernel(
    const float* __restrict__ A,      // [M, K]
    const float* __restrict__ B,      // [K, N] or [N, K]
    float* __restrict__ C,            // [M, num_indices]
    const int* __restrict__ indices,  // [num_indices] - columns to gather from B
    int M, int N, int K,
    int num_indices,
    float alpha, float beta,
    bool B_transposed)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;

    const int row_in_tile = tx / (TILE_N / 4);
    const int col_in_tile = (tx % (TILE_N / 4)) * 4;

    __shared__ float As[NAX_STAGES][TILE_M][TILE_K + 1];
    __shared__ float Bs[NAX_STAGES][TILE_K][TILE_N + 1];

    const int c_row = bx * TILE_M + row_in_tile;
    const int c_col = by * TILE_N + col_in_tile;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // Prefetch initial stages
    #pragma unroll
    for (int stage = 0; stage < NAX_STAGES - 1 && stage < num_k_tiles; stage++) {
        const int k_offset = stage * TILE_K;

        // Load A tile
        for (int i = tx; i < TILE_M * TILE_K; i += BLOCK_SIZE) {
            int local_m = i / TILE_K;
            int local_k = i % TILE_K;
            int global_m = bx * TILE_M + local_m;
            int global_k = k_offset + local_k;

            float val = 0.0f;
            if (global_m < M && global_k < K) {
                val = A[global_m * K + global_k];
            }
            As[stage][local_m][local_k] = val;
        }

        // Load B tile with gather
        for (int i = tx; i < TILE_K * TILE_N; i += BLOCK_SIZE) {
            int local_k = i / TILE_N;
            int local_n = i % TILE_N;
            int global_k = k_offset + local_k;
            int global_n = by * TILE_N + local_n;

            float val = 0.0f;
            if (global_k < K && global_n < num_indices) {
                int actual_col = indices[global_n];
                if (actual_col >= 0 && actual_col < N) {
                    if (B_transposed) {
                        val = B[actual_col * K + global_k];
                    } else {
                        val = B[global_k * N + actual_col];
                    }
                }
            }
            Bs[stage][local_k][local_n] = val;
        }
    }

    __syncthreads();

    // Main NAX loop
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int curr_stage = k_tile % NAX_STAGES;
        const int next_stage = (k_tile + NAX_STAGES - 1) % NAX_STAGES;
        const int prefetch_tile = k_tile + NAX_STAGES - 1;

        if (prefetch_tile < num_k_tiles) {
            const int k_offset = prefetch_tile * TILE_K;

            for (int i = tx; i < TILE_M * TILE_K; i += BLOCK_SIZE) {
                int local_m = i / TILE_K;
                int local_k = i % TILE_K;
                int global_m = bx * TILE_M + local_m;
                int global_k = k_offset + local_k;

                float val = 0.0f;
                if (global_m < M && global_k < K) {
                    val = A[global_m * K + global_k];
                }
                As[next_stage][local_m][local_k] = val;
            }

            for (int i = tx; i < TILE_K * TILE_N; i += BLOCK_SIZE) {
                int local_k = i / TILE_N;
                int local_n = i % TILE_N;
                int global_k = k_offset + local_k;
                int global_n = by * TILE_N + local_n;

                float val = 0.0f;
                if (global_k < K && global_n < num_indices) {
                    int actual_col = indices[global_n];
                    if (actual_col >= 0 && actual_col < N) {
                        if (B_transposed) {
                            val = B[actual_col * K + global_k];
                        } else {
                            val = B[global_k * N + actual_col];
                        }
                    }
                }
                Bs[next_stage][local_k][local_n] = val;
            }
        }

        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            float a_val = As[curr_stage][row_in_tile][k];

            #pragma unroll
            for (int j = 0; j < 4; j++) {
                acc[j] += a_val * Bs[curr_stage][k][col_in_tile + j];
            }
        }

        __syncthreads();
    }

    // Write output
    if (c_row < M) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (c_col + j < num_indices) {
                float result = alpha * acc[j];
                if (beta != 0.0f) {
                    result += beta * C[c_row * num_indices + c_col + j];
                }
                C[c_row * num_indices + c_col + j] = result;
            }
        }
    }
}

// ============================================================================
// NAX Sparse Gather GEMM - Both A and B use index arrays
// ============================================================================

extern "C" __global__
void steel_gemm_sparse_gather_nax_kernel(
    const float* __restrict__ A,       // [M, K]
    const float* __restrict__ B,       // [K, N]
    float* __restrict__ C,             // [num_rows, num_cols]
    const int* __restrict__ row_idx,   // [num_rows]
    const int* __restrict__ col_idx,   // [num_cols]
    int M, int N, int K,
    int num_rows, int num_cols,
    float alpha, float beta)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;

    const int row_in_tile = tx / (TILE_N / 4);
    const int col_in_tile = (tx % (TILE_N / 4)) * 4;

    __shared__ float As[NAX_STAGES][TILE_M][TILE_K + 1];
    __shared__ float Bs[NAX_STAGES][TILE_K][TILE_N + 1];
    __shared__ int row_cache[TILE_M];
    __shared__ int col_cache[TILE_N];

    const int c_row = bx * TILE_M + row_in_tile;
    const int c_col = by * TILE_N + col_in_tile;

    // Cache row and column indices
    if (tx < TILE_M) {
        int global_row = bx * TILE_M + tx;
        row_cache[tx] = (global_row < num_rows) ? row_idx[global_row] : -1;
    }
    if (tx < TILE_N) {
        int global_col = by * TILE_N + tx;
        col_cache[tx] = (global_col < num_cols) ? col_idx[global_col] : -1;
    }
    __syncthreads();

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // Prefetch
    #pragma unroll
    for (int stage = 0; stage < NAX_STAGES - 1 && stage < num_k_tiles; stage++) {
        const int k_offset = stage * TILE_K;

        for (int i = tx; i < TILE_M * TILE_K; i += BLOCK_SIZE) {
            int local_m = i / TILE_K;
            int local_k = i % TILE_K;
            int global_k = k_offset + local_k;

            float val = 0.0f;
            int actual_row = row_cache[local_m];
            if (actual_row >= 0 && actual_row < M && global_k < K) {
                val = A[actual_row * K + global_k];
            }
            As[stage][local_m][local_k] = val;
        }

        for (int i = tx; i < TILE_K * TILE_N; i += BLOCK_SIZE) {
            int local_k = i / TILE_N;
            int local_n = i % TILE_N;
            int global_k = k_offset + local_k;

            float val = 0.0f;
            int actual_col = col_cache[local_n];
            if (global_k < K && actual_col >= 0 && actual_col < N) {
                val = B[global_k * N + actual_col];
            }
            Bs[stage][local_k][local_n] = val;
        }
    }

    __syncthreads();

    // Main loop
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int curr_stage = k_tile % NAX_STAGES;
        const int next_stage = (k_tile + NAX_STAGES - 1) % NAX_STAGES;
        const int prefetch_tile = k_tile + NAX_STAGES - 1;

        if (prefetch_tile < num_k_tiles) {
            const int k_offset = prefetch_tile * TILE_K;

            for (int i = tx; i < TILE_M * TILE_K; i += BLOCK_SIZE) {
                int local_m = i / TILE_K;
                int local_k = i % TILE_K;
                int global_k = k_offset + local_k;

                float val = 0.0f;
                int actual_row = row_cache[local_m];
                if (actual_row >= 0 && actual_row < M && global_k < K) {
                    val = A[actual_row * K + global_k];
                }
                As[next_stage][local_m][local_k] = val;
            }

            for (int i = tx; i < TILE_K * TILE_N; i += BLOCK_SIZE) {
                int local_k = i / TILE_N;
                int local_n = i % TILE_N;
                int global_k = k_offset + local_k;

                float val = 0.0f;
                int actual_col = col_cache[local_n];
                if (global_k < K && actual_col >= 0 && actual_col < N) {
                    val = B[global_k * N + actual_col];
                }
                Bs[next_stage][local_k][local_n] = val;
            }
        }

        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            float a_val = As[curr_stage][row_in_tile][k];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                acc[j] += a_val * Bs[curr_stage][k][col_in_tile + j];
            }
        }

        __syncthreads();
    }

    if (c_row < num_rows) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (c_col + j < num_cols) {
                float result = alpha * acc[j];
                if (beta != 0.0f) {
                    result += beta * C[c_row * num_cols + c_col + j];
                }
                C[c_row * num_cols + c_col + j] = result;
            }
        }
    }
}

// ============================================================================
// NAX Embedding Gather GEMM - Optimized for embedding lookups
// ============================================================================

extern "C" __global__
void steel_embedding_gather_nax_kernel(
    const float* __restrict__ embeddings,  // [vocab_size, embed_dim]
    const float* __restrict__ weights,     // [embed_dim, hidden_dim]
    float* __restrict__ output,            // [batch_size, hidden_dim]
    const int* __restrict__ token_ids,     // [batch_size]
    int vocab_size, int embed_dim, int hidden_dim,
    int batch_size,
    float alpha)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;

    const int row_in_tile = tx / (TILE_N / 4);
    const int col_in_tile = (tx % (TILE_N / 4)) * 4;

    __shared__ float Es[NAX_STAGES][TILE_M][TILE_K + 1];  // Embedding rows
    __shared__ float Ws[NAX_STAGES][TILE_K][TILE_N + 1];  // Weight columns
    __shared__ int token_cache[TILE_M];

    const int out_row = bx * TILE_M + row_in_tile;
    const int out_col = by * TILE_N + col_in_tile;

    // Cache token IDs
    if (tx < TILE_M) {
        int global_row = bx * TILE_M + tx;
        token_cache[tx] = (global_row < batch_size) ? token_ids[global_row] : -1;
    }
    __syncthreads();

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const int num_k_tiles = (embed_dim + TILE_K - 1) / TILE_K;

    // Prefetch
    #pragma unroll
    for (int stage = 0; stage < NAX_STAGES - 1 && stage < num_k_tiles; stage++) {
        const int k_offset = stage * TILE_K;

        for (int i = tx; i < TILE_M * TILE_K; i += BLOCK_SIZE) {
            int local_m = i / TILE_K;
            int local_k = i % TILE_K;
            int global_k = k_offset + local_k;

            float val = 0.0f;
            int tok_id = token_cache[local_m];
            if (tok_id >= 0 && tok_id < vocab_size && global_k < embed_dim) {
                val = embeddings[tok_id * embed_dim + global_k];
            }
            Es[stage][local_m][local_k] = val;
        }

        for (int i = tx; i < TILE_K * TILE_N; i += BLOCK_SIZE) {
            int local_k = i / TILE_N;
            int local_n = i % TILE_N;
            int global_k = k_offset + local_k;
            int global_n = by * TILE_N + local_n;

            float val = 0.0f;
            if (global_k < embed_dim && global_n < hidden_dim) {
                val = weights[global_k * hidden_dim + global_n];
            }
            Ws[stage][local_k][local_n] = val;
        }
    }

    __syncthreads();

    // Main loop
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int curr_stage = k_tile % NAX_STAGES;
        const int next_stage = (k_tile + NAX_STAGES - 1) % NAX_STAGES;
        const int prefetch_tile = k_tile + NAX_STAGES - 1;

        if (prefetch_tile < num_k_tiles) {
            const int k_offset = prefetch_tile * TILE_K;

            for (int i = tx; i < TILE_M * TILE_K; i += BLOCK_SIZE) {
                int local_m = i / TILE_K;
                int local_k = i % TILE_K;
                int global_k = k_offset + local_k;

                float val = 0.0f;
                int tok_id = token_cache[local_m];
                if (tok_id >= 0 && tok_id < vocab_size && global_k < embed_dim) {
                    val = embeddings[tok_id * embed_dim + global_k];
                }
                Es[next_stage][local_m][local_k] = val;
            }

            for (int i = tx; i < TILE_K * TILE_N; i += BLOCK_SIZE) {
                int local_k = i / TILE_N;
                int local_n = i % TILE_N;
                int global_k = k_offset + local_k;
                int global_n = by * TILE_N + local_n;

                float val = 0.0f;
                if (global_k < embed_dim && global_n < hidden_dim) {
                    val = weights[global_k * hidden_dim + global_n];
                }
                Ws[next_stage][local_k][local_n] = val;
            }
        }

        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            float e_val = Es[curr_stage][row_in_tile][k];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                acc[j] += e_val * Ws[curr_stage][k][col_in_tile + j];
            }
        }

        __syncthreads();
    }

    if (out_row < batch_size) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (out_col + j < hidden_dim) {
                output[out_row * hidden_dim + out_col + j] = alpha * acc[j];
            }
        }
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_steel_gemm_gather_a_nax(
    const float* A, const float* B, float* C,
    const int* indices,
    int M, int N, int K, int num_indices,
    float alpha, float beta,
    bool A_transposed,
    cudaStream_t stream)
{
    dim3 block(BLOCK_SIZE);
    dim3 grid(
        (num_indices + TILE_M - 1) / TILE_M,
        (N + TILE_N - 1) / TILE_N
    );

    steel_gemm_gather_a_nax_kernel<<<grid, block, 0, stream>>>(
        A, B, C, indices, M, N, K, num_indices,
        alpha, beta, A_transposed
    );

    return cudaGetLastError();
}

int lux_cuda_steel_gemm_gather_b_nax(
    const float* A, const float* B, float* C,
    const int* indices,
    int M, int N, int K, int num_indices,
    float alpha, float beta,
    bool B_transposed,
    cudaStream_t stream)
{
    dim3 block(BLOCK_SIZE);
    dim3 grid(
        (M + TILE_M - 1) / TILE_M,
        (num_indices + TILE_N - 1) / TILE_N
    );

    steel_gemm_gather_b_nax_kernel<<<grid, block, 0, stream>>>(
        A, B, C, indices, M, N, K, num_indices,
        alpha, beta, B_transposed
    );

    return cudaGetLastError();
}

int lux_cuda_steel_gemm_sparse_gather_nax(
    const float* A, const float* B, float* C,
    const int* row_indices, const int* col_indices,
    int M, int N, int K,
    int num_rows, int num_cols,
    float alpha, float beta,
    cudaStream_t stream)
{
    dim3 block(BLOCK_SIZE);
    dim3 grid(
        (num_rows + TILE_M - 1) / TILE_M,
        (num_cols + TILE_N - 1) / TILE_N
    );

    steel_gemm_sparse_gather_nax_kernel<<<grid, block, 0, stream>>>(
        A, B, C, row_indices, col_indices,
        M, N, K, num_rows, num_cols,
        alpha, beta
    );

    return cudaGetLastError();
}

int lux_cuda_steel_embedding_gather_nax(
    const float* embeddings, const float* weights, float* output,
    const int* token_ids,
    int vocab_size, int embed_dim, int hidden_dim,
    int batch_size,
    float alpha,
    cudaStream_t stream)
{
    dim3 block(BLOCK_SIZE);
    dim3 grid(
        (batch_size + TILE_M - 1) / TILE_M,
        (hidden_dim + TILE_N - 1) / TILE_N
    );

    steel_embedding_gather_nax_kernel<<<grid, block, 0, stream>>>(
        embeddings, weights, output, token_ids,
        vocab_size, embed_dim, hidden_dim, batch_size,
        alpha
    );

    return cudaGetLastError();
}

}  // extern "C"
