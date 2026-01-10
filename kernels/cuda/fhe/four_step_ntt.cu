// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file four_step_ntt.cu
 * @brief Four-step NTT algorithm for large polynomial transforms
 *
 * Implements the four-step (or six-step) NTT algorithm for transforms
 * where N > shared memory capacity. The algorithm factorizes N = N1 * N2
 * and performs:
 * 1. N1 column NTTs of size N2
 * 2. Twiddle factor multiplication
 * 3. N2 row NTTs of size N1
 * 4. Final transpose (if needed)
 *
 * This approach enables efficient NTT for N = 16384, 32768, 65536, etc.
 * where single-kernel shared memory NTT would exceed GPU limits.
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// ============================================================================
// Constants and Configuration
// ============================================================================

#define FOUR_STEP_BLOCK_SIZE 256
#define FOUR_STEP_TILE_DIM 32
#define MAX_SHARED_LOG_N 12  // Max log_n for shared memory NTT (4096 elements)

// ============================================================================
// Device Helper Functions
// ============================================================================

/**
 * @brief Barrett multiplication: (a * b) mod q
 */
__device__ __forceinline__ uint64_t barrett_mul_4step(
    uint64_t a,
    uint64_t b,
    uint64_t q,
    uint64_t mu
) {
    unsigned __int128 prod = (unsigned __int128)a * b;
    uint64_t lo = (uint64_t)prod;
    uint64_t hi = (uint64_t)(prod >> 64);

    unsigned __int128 tmp = (unsigned __int128)hi * mu;
    uint64_t q_approx = (uint64_t)(tmp >> 64);

    uint64_t result = lo - q_approx * q;

    // Correction
    if (result >= q) result -= q;
    if (result >= q) result -= q;

    return result;
}

/**
 * @brief Modular addition with single correction
 */
__device__ __forceinline__ uint64_t mod_add_4step(uint64_t a, uint64_t b, uint64_t q) {
    uint64_t sum = a + b;
    return (sum >= q) ? (sum - q) : sum;
}

/**
 * @brief Modular subtraction with correction
 */
__device__ __forceinline__ uint64_t mod_sub_4step(uint64_t a, uint64_t b, uint64_t q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

/**
 * @brief Bit reversal for index
 */
__device__ __forceinline__ uint32_t bit_reverse_4step(uint32_t x, uint32_t log_n) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < log_n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

// ============================================================================
// Shared Memory NTT for Sub-transforms
// ============================================================================

/**
 * @brief Shared memory NTT for sub-transform (Cooley-Tukey)
 * @param shared Shared memory array
 * @param n Size of transform
 * @param log_n Log2 of n
 * @param twiddles Twiddle factors (bit-reversed)
 * @param q Modulus
 * @param mu Barrett parameter
 */
__device__ void shared_ntt_forward_4step(
    uint64_t* shared,
    uint32_t n,
    uint32_t log_n,
    const uint64_t* twiddles,
    uint64_t q,
    uint64_t mu
) {
    uint32_t tid = threadIdx.x;

    // Cooley-Tukey butterfly
    for (uint32_t stage = 0; stage < log_n; stage++) {
        uint32_t m = 1u << (stage + 1);
        uint32_t half_m = m >> 1;

        __syncthreads();

        for (uint32_t k = tid; k < n / 2; k += blockDim.x) {
            uint32_t j = k % half_m;
            uint32_t i = (k / half_m) * m + j;

            uint64_t w = twiddles[half_m + j];
            uint64_t u = shared[i];
            uint64_t t = barrett_mul_4step(w, shared[i + half_m], q, mu);

            shared[i] = mod_add_4step(u, t, q);
            shared[i + half_m] = mod_sub_4step(u, t, q);
        }
    }

    __syncthreads();
}

/**
 * @brief Shared memory inverse NTT (Gentleman-Sande)
 */
__device__ void shared_ntt_inverse_4step(
    uint64_t* shared,
    uint32_t n,
    uint32_t log_n,
    const uint64_t* inv_twiddles,
    uint64_t n_inv,
    uint64_t q,
    uint64_t mu
) {
    uint32_t tid = threadIdx.x;

    // Gentleman-Sande butterfly
    for (int stage = log_n - 1; stage >= 0; stage--) {
        uint32_t m = 1u << (stage + 1);
        uint32_t half_m = m >> 1;

        __syncthreads();

        for (uint32_t k = tid; k < n / 2; k += blockDim.x) {
            uint32_t j = k % half_m;
            uint32_t i = (k / half_m) * m + j;

            uint64_t w = inv_twiddles[half_m + j];
            uint64_t u = shared[i];
            uint64_t v = shared[i + half_m];

            shared[i] = mod_add_4step(u, v, q);
            uint64_t diff = mod_sub_4step(u, v, q);
            shared[i + half_m] = barrett_mul_4step(diff, w, q, mu);
        }
    }

    __syncthreads();

    // Scale by n_inv
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        shared[i] = barrett_mul_4step(shared[i], n_inv, q, mu);
    }

    __syncthreads();
}

// ============================================================================
// Four-Step NTT Kernels
// ============================================================================

/**
 * @brief Step 1: Column NTTs (N1 transforms of size N2)
 *
 * For N = N1 * N2, process N1 columns, each of size N2.
 * Input: row-major matrix [N1 x N2]
 * Each block handles one column.
 */
extern "C" __global__ void four_step_ntt_columns_kernel(
    uint64_t* data,
    const uint64_t* twiddles_n2,  // Twiddles for size N2
    uint32_t n1,                   // Number of columns
    uint32_t n2,                   // Column size
    uint32_t log_n2,
    uint64_t q,
    uint64_t mu
) {
    extern __shared__ uint64_t shared[];

    uint32_t col = blockIdx.x;
    if (col >= n1) return;

    uint32_t tid = threadIdx.x;

    // Load column into shared memory (strided access)
    for (uint32_t i = tid; i < n2; i += blockDim.x) {
        shared[i] = data[i * n1 + col];
    }

    // NTT on column
    shared_ntt_forward_4step(shared, n2, log_n2, twiddles_n2, q, mu);

    // Store back (strided)
    for (uint32_t i = tid; i < n2; i += blockDim.x) {
        data[i * n1 + col] = shared[i];
    }
}

/**
 * @brief Step 2: Twiddle factor multiplication
 *
 * Multiply element (i, j) by w^(i*j) where w is primitive N-th root.
 * These are the "diagonal" twiddle factors.
 */
extern "C" __global__ void four_step_ntt_twiddle_kernel(
    uint64_t* data,
    const uint64_t* diag_twiddles,  // w^(i*j) precomputed
    uint32_t n1,
    uint32_t n2,
    uint64_t q,
    uint64_t mu
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = n1 * n2;

    if (idx >= total) return;

    uint32_t i = idx / n1;  // Row
    uint32_t j = idx % n1;  // Column

    // Twiddle index = i * j (mod N for negacyclic)
    uint64_t twiddle = diag_twiddles[idx];

    data[idx] = barrett_mul_4step(data[idx], twiddle, q, mu);
}

/**
 * @brief Step 3: Row NTTs (N2 transforms of size N1)
 *
 * Process N2 rows, each of size N1.
 * Each block handles one row.
 */
extern "C" __global__ void four_step_ntt_rows_kernel(
    uint64_t* data,
    const uint64_t* twiddles_n1,  // Twiddles for size N1
    uint32_t n1,                   // Row size
    uint32_t n2,                   // Number of rows
    uint32_t log_n1,
    uint64_t q,
    uint64_t mu
) {
    extern __shared__ uint64_t shared[];

    uint32_t row = blockIdx.x;
    if (row >= n2) return;

    uint32_t tid = threadIdx.x;
    uint64_t* row_ptr = data + row * n1;

    // Load row into shared memory (contiguous)
    for (uint32_t i = tid; i < n1; i += blockDim.x) {
        shared[i] = row_ptr[i];
    }

    // NTT on row
    shared_ntt_forward_4step(shared, n1, log_n1, twiddles_n1, q, mu);

    // Store back
    for (uint32_t i = tid; i < n1; i += blockDim.x) {
        row_ptr[i] = shared[i];
    }
}

/**
 * @brief Transpose kernel for final reordering
 *
 * Converts row-major [N2 x N1] to row-major [N1 x N2].
 * Uses shared memory tiles to coalesce global memory access.
 */
extern "C" __global__ void four_step_ntt_transpose_kernel(
    const uint64_t* input,
    uint64_t* output,
    uint32_t n1,  // Output rows (input cols)
    uint32_t n2   // Output cols (input rows)
) {
    __shared__ uint64_t tile[FOUR_STEP_TILE_DIM][FOUR_STEP_TILE_DIM + 1];  // +1 to avoid bank conflicts

    uint32_t x = blockIdx.x * FOUR_STEP_TILE_DIM + threadIdx.x;
    uint32_t y = blockIdx.y * FOUR_STEP_TILE_DIM + threadIdx.y;

    // Load tile from input [n2 x n1] (rows x cols)
    if (x < n1 && y < n2) {
        tile[threadIdx.y][threadIdx.x] = input[y * n1 + x];
    }

    __syncthreads();

    // Write transposed tile to output [n1 x n2]
    uint32_t out_x = blockIdx.y * FOUR_STEP_TILE_DIM + threadIdx.x;
    uint32_t out_y = blockIdx.x * FOUR_STEP_TILE_DIM + threadIdx.y;

    if (out_x < n2 && out_y < n1) {
        output[out_y * n2 + out_x] = tile[threadIdx.x][threadIdx.y];
    }
}

// ============================================================================
// Inverse Four-Step NTT Kernels
// ============================================================================

/**
 * @brief Inverse Step 1: Row inverse NTTs
 */
extern "C" __global__ void four_step_intt_rows_kernel(
    uint64_t* data,
    const uint64_t* inv_twiddles_n1,
    uint64_t n1_inv,
    uint32_t n1,
    uint32_t n2,
    uint32_t log_n1,
    uint64_t q,
    uint64_t mu
) {
    extern __shared__ uint64_t shared[];

    uint32_t row = blockIdx.x;
    if (row >= n2) return;

    uint32_t tid = threadIdx.x;
    uint64_t* row_ptr = data + row * n1;

    // Load row
    for (uint32_t i = tid; i < n1; i += blockDim.x) {
        shared[i] = row_ptr[i];
    }

    // Inverse NTT
    shared_ntt_inverse_4step(shared, n1, log_n1, inv_twiddles_n1, n1_inv, q, mu);

    // Store
    for (uint32_t i = tid; i < n1; i += blockDim.x) {
        row_ptr[i] = shared[i];
    }
}

/**
 * @brief Inverse Step 2: Inverse twiddle multiplication
 */
extern "C" __global__ void four_step_intt_twiddle_kernel(
    uint64_t* data,
    const uint64_t* inv_diag_twiddles,  // w^(-i*j)
    uint32_t n1,
    uint32_t n2,
    uint64_t q,
    uint64_t mu
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = n1 * n2;

    if (idx >= total) return;

    uint64_t twiddle = inv_diag_twiddles[idx];
    data[idx] = barrett_mul_4step(data[idx], twiddle, q, mu);
}

/**
 * @brief Inverse Step 3: Column inverse NTTs
 */
extern "C" __global__ void four_step_intt_columns_kernel(
    uint64_t* data,
    const uint64_t* inv_twiddles_n2,
    uint64_t n2_inv,
    uint32_t n1,
    uint32_t n2,
    uint32_t log_n2,
    uint64_t q,
    uint64_t mu
) {
    extern __shared__ uint64_t shared[];

    uint32_t col = blockIdx.x;
    if (col >= n1) return;

    uint32_t tid = threadIdx.x;

    // Load column
    for (uint32_t i = tid; i < n2; i += blockDim.x) {
        shared[i] = data[i * n1 + col];
    }

    // Inverse NTT
    shared_ntt_inverse_4step(shared, n2, log_n2, inv_twiddles_n2, n2_inv, q, mu);

    // Store
    for (uint32_t i = tid; i < n2; i += blockDim.x) {
        data[i * n1 + col] = shared[i];
    }
}

// ============================================================================
// Batch Processing Kernels
// ============================================================================

/**
 * @brief Batched column NTTs for multiple polynomials
 */
extern "C" __global__ void four_step_ntt_columns_batch_kernel(
    uint64_t* data,
    const uint64_t* twiddles_n2,
    uint32_t n1,
    uint32_t n2,
    uint32_t log_n2,
    uint32_t batch_count,
    uint64_t q,
    uint64_t mu
) {
    extern __shared__ uint64_t shared[];

    uint32_t col = blockIdx.x % n1;
    uint32_t batch = blockIdx.x / n1;

    if (batch >= batch_count) return;

    uint32_t tid = threadIdx.x;
    uint64_t* poly = data + batch * n1 * n2;

    // Load column
    for (uint32_t i = tid; i < n2; i += blockDim.x) {
        shared[i] = poly[i * n1 + col];
    }

    shared_ntt_forward_4step(shared, n2, log_n2, twiddles_n2, q, mu);

    // Store
    for (uint32_t i = tid; i < n2; i += blockDim.x) {
        poly[i * n1 + col] = shared[i];
    }
}

/**
 * @brief Batched row NTTs for multiple polynomials
 */
extern "C" __global__ void four_step_ntt_rows_batch_kernel(
    uint64_t* data,
    const uint64_t* twiddles_n1,
    uint32_t n1,
    uint32_t n2,
    uint32_t log_n1,
    uint32_t batch_count,
    uint64_t q,
    uint64_t mu
) {
    extern __shared__ uint64_t shared[];

    uint32_t row = blockIdx.x % n2;
    uint32_t batch = blockIdx.x / n2;

    if (batch >= batch_count) return;

    uint32_t tid = threadIdx.x;
    uint64_t* row_ptr = data + batch * n1 * n2 + row * n1;

    // Load row
    for (uint32_t i = tid; i < n1; i += blockDim.x) {
        shared[i] = row_ptr[i];
    }

    shared_ntt_forward_4step(shared, n1, log_n1, twiddles_n1, q, mu);

    // Store
    for (uint32_t i = tid; i < n1; i += blockDim.x) {
        row_ptr[i] = shared[i];
    }
}

/**
 * @brief Batched twiddle multiplication
 */
extern "C" __global__ void four_step_ntt_twiddle_batch_kernel(
    uint64_t* data,
    const uint64_t* diag_twiddles,
    uint32_t n,
    uint32_t batch_count,
    uint64_t q,
    uint64_t mu
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t batch = idx / n;
    uint32_t pos = idx % n;

    if (batch >= batch_count) return;

    uint64_t* poly = data + batch * n;
    poly[pos] = barrett_mul_4step(poly[pos], diag_twiddles[pos], q, mu);
}

// ============================================================================
// Fused Four-Step NTT (Single Launch)
// ============================================================================

/**
 * @brief Configuration for four-step NTT
 */
struct FourStepNTTConfig {
    uint32_t n;           // Total size
    uint32_t n1;          // Factorization N = N1 * N2
    uint32_t n2;
    uint32_t log_n1;
    uint32_t log_n2;
    uint64_t q;           // Modulus
    uint64_t mu;          // Barrett parameter
    uint64_t n1_inv;      // N1^(-1) mod q
    uint64_t n2_inv;      // N2^(-1) mod q
};

/**
 * @brief Compute optimal factorization N = N1 * N2
 *
 * Chooses N1 and N2 such that both fit in shared memory
 * and minimize global memory transactions.
 */
__host__ void compute_four_step_factorization(
    uint32_t n,
    uint32_t max_shared_size,
    uint32_t* n1_out,
    uint32_t* n2_out
) {
    // Find sqrt(N) rounded to power of 2
    uint32_t log_n = 0;
    uint32_t temp = n;
    while (temp > 1) {
        log_n++;
        temp >>= 1;
    }

    uint32_t log_n1 = log_n / 2;
    uint32_t log_n2 = log_n - log_n1;

    // Ensure both fit in shared memory
    uint32_t max_log = 0;
    temp = max_shared_size / sizeof(uint64_t);
    while (temp > 1) {
        max_log++;
        temp >>= 1;
    }

    if (log_n1 > max_log) {
        log_n1 = max_log;
        log_n2 = log_n - log_n1;
    }
    if (log_n2 > max_log) {
        log_n2 = max_log;
        log_n1 = log_n - log_n2;
    }

    *n1_out = 1u << log_n1;
    *n2_out = 1u << log_n2;
}

// ============================================================================
// C API Functions
// ============================================================================

extern "C" {

/**
 * @brief Perform four-step forward NTT
 */
cudaError_t lux_cuda_four_step_ntt_forward(
    uint64_t* d_data,
    const uint64_t* d_twiddles_n1,
    const uint64_t* d_twiddles_n2,
    const uint64_t* d_diag_twiddles,
    uint32_t n,
    uint32_t n1,
    uint32_t n2,
    uint32_t log_n1,
    uint32_t log_n2,
    uint64_t q,
    uint64_t mu,
    cudaStream_t stream
) {
    // Step 1: Column NTTs
    size_t shared_size_n2 = n2 * sizeof(uint64_t);
    four_step_ntt_columns_kernel<<<n1, FOUR_STEP_BLOCK_SIZE, shared_size_n2, stream>>>(
        d_data, d_twiddles_n2, n1, n2, log_n2, q, mu
    );

    // Step 2: Twiddle multiplication
    uint32_t twiddle_blocks = (n + FOUR_STEP_BLOCK_SIZE - 1) / FOUR_STEP_BLOCK_SIZE;
    four_step_ntt_twiddle_kernel<<<twiddle_blocks, FOUR_STEP_BLOCK_SIZE, 0, stream>>>(
        d_data, d_diag_twiddles, n1, n2, q, mu
    );

    // Step 3: Row NTTs
    size_t shared_size_n1 = n1 * sizeof(uint64_t);
    four_step_ntt_rows_kernel<<<n2, FOUR_STEP_BLOCK_SIZE, shared_size_n1, stream>>>(
        d_data, d_twiddles_n1, n1, n2, log_n1, q, mu
    );

    return cudaGetLastError();
}

/**
 * @brief Perform four-step inverse NTT
 */
cudaError_t lux_cuda_four_step_ntt_inverse(
    uint64_t* d_data,
    const uint64_t* d_inv_twiddles_n1,
    const uint64_t* d_inv_twiddles_n2,
    const uint64_t* d_inv_diag_twiddles,
    uint32_t n,
    uint32_t n1,
    uint32_t n2,
    uint32_t log_n1,
    uint32_t log_n2,
    uint64_t n1_inv,
    uint64_t n2_inv,
    uint64_t q,
    uint64_t mu,
    cudaStream_t stream
) {
    // Step 1: Row inverse NTTs
    size_t shared_size_n1 = n1 * sizeof(uint64_t);
    four_step_intt_rows_kernel<<<n2, FOUR_STEP_BLOCK_SIZE, shared_size_n1, stream>>>(
        d_data, d_inv_twiddles_n1, n1_inv, n1, n2, log_n1, q, mu
    );

    // Step 2: Inverse twiddle multiplication
    uint32_t twiddle_blocks = (n + FOUR_STEP_BLOCK_SIZE - 1) / FOUR_STEP_BLOCK_SIZE;
    four_step_intt_twiddle_kernel<<<twiddle_blocks, FOUR_STEP_BLOCK_SIZE, 0, stream>>>(
        d_data, d_inv_diag_twiddles, n1, n2, q, mu
    );

    // Step 3: Column inverse NTTs
    size_t shared_size_n2 = n2 * sizeof(uint64_t);
    four_step_intt_columns_kernel<<<n1, FOUR_STEP_BLOCK_SIZE, shared_size_n2, stream>>>(
        d_data, d_inv_twiddles_n2, n2_inv, n1, n2, log_n2, q, mu
    );

    return cudaGetLastError();
}

/**
 * @brief Batched four-step forward NTT
 */
cudaError_t lux_cuda_four_step_ntt_forward_batch(
    uint64_t* d_data,
    const uint64_t* d_twiddles_n1,
    const uint64_t* d_twiddles_n2,
    const uint64_t* d_diag_twiddles,
    uint32_t n,
    uint32_t n1,
    uint32_t n2,
    uint32_t log_n1,
    uint32_t log_n2,
    uint32_t batch_count,
    uint64_t q,
    uint64_t mu,
    cudaStream_t stream
) {
    uint32_t total_n = n * batch_count;

    // Batched column NTTs
    size_t shared_size_n2 = n2 * sizeof(uint64_t);
    four_step_ntt_columns_batch_kernel<<<n1 * batch_count, FOUR_STEP_BLOCK_SIZE, shared_size_n2, stream>>>(
        d_data, d_twiddles_n2, n1, n2, log_n2, batch_count, q, mu
    );

    // Batched twiddle multiplication
    uint32_t twiddle_blocks = (total_n + FOUR_STEP_BLOCK_SIZE - 1) / FOUR_STEP_BLOCK_SIZE;
    four_step_ntt_twiddle_batch_kernel<<<twiddle_blocks, FOUR_STEP_BLOCK_SIZE, 0, stream>>>(
        d_data, d_diag_twiddles, n, batch_count, q, mu
    );

    // Batched row NTTs
    size_t shared_size_n1 = n1 * sizeof(uint64_t);
    four_step_ntt_rows_batch_kernel<<<n2 * batch_count, FOUR_STEP_BLOCK_SIZE, shared_size_n1, stream>>>(
        d_data, d_twiddles_n1, n1, n2, log_n1, batch_count, q, mu
    );

    return cudaGetLastError();
}

/**
 * @brief Transpose data for final output ordering
 */
cudaError_t lux_cuda_four_step_transpose(
    const uint64_t* d_input,
    uint64_t* d_output,
    uint32_t n1,
    uint32_t n2,
    cudaStream_t stream
) {
    dim3 block(FOUR_STEP_TILE_DIM, FOUR_STEP_TILE_DIM);
    dim3 grid(
        (n1 + FOUR_STEP_TILE_DIM - 1) / FOUR_STEP_TILE_DIM,
        (n2 + FOUR_STEP_TILE_DIM - 1) / FOUR_STEP_TILE_DIM
    );

    four_step_ntt_transpose_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, n1, n2
    );

    return cudaGetLastError();
}

/**
 * @brief Get optimal factorization for given N
 */
void lux_cuda_four_step_get_factorization(
    uint32_t n,
    uint32_t max_shared_bytes,
    uint32_t* n1_out,
    uint32_t* n2_out
) {
    compute_four_step_factorization(n, max_shared_bytes, n1_out, n2_out);
}

/**
 * @brief Query shared memory requirement
 */
size_t lux_cuda_four_step_shared_memory_requirement(
    uint32_t n1,
    uint32_t n2
) {
    return (n1 > n2 ? n1 : n2) * sizeof(uint64_t);
}

}  // extern "C"
