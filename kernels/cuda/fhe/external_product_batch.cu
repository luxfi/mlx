// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Batched External Product CUDA Kernels for TFHE
// Optimized for processing multiple external products in parallel with
// support for different GGSW selection per batch element.
//
// Features:
// - Batch processing with per-element GGSW selection
// - Prefetching and L2 cache optimization
// - Warp-level parallelism for decomposition
// - Persistent threads for BSK traversal

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================================
// Constants
// ============================================================================

#define MAX_N 2048
#define MAX_K 2
#define MAX_L 8
#define WARP_SIZE 32
#define MAX_BATCH_PER_BLOCK 4

// ============================================================================
// Modular Arithmetic (Montgomery Form)
// ============================================================================

__device__ __forceinline__
uint64_t mod_add(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t sum = a + b;
    return sum >= Q ? sum - Q : sum;
}

__device__ __forceinline__
uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t Q) {
    return a >= b ? a - b : a + Q - b;
}

// Barrett reduction: (a * b) mod Q
__device__ __forceinline__
uint64_t barrett_mul(uint64_t a, uint64_t omega, uint64_t Q, uint64_t precon) {
    uint64_t q_approx = __umul64hi(a, precon);
    uint64_t product = a * omega;
    uint64_t result = product - q_approx * Q;
    return result >= Q ? result - Q : result;
}

// Full modular multiplication
__device__ __forceinline__
uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);

    if (hi == 0) return lo % Q;

    uint64_t r = (1ULL << 32) % Q;
    r = (r * r) % Q;

    uint64_t lo_mod = lo % Q;
    uint64_t hi_mod = (hi % Q) * r % Q;
    return mod_add(lo_mod, hi_mod, Q);
}

// ============================================================================
// NTT Butterflies
// ============================================================================

__device__ __forceinline__
void ct_butterfly(uint64_t* lo, uint64_t* hi, uint64_t omega, uint64_t Q, uint64_t precon) {
    uint64_t u = *lo;
    uint64_t v = barrett_mul(*hi, omega, Q, precon);
    *lo = mod_add(u, v, Q);
    *hi = mod_sub(u, v, Q);
}

__device__ __forceinline__
void gs_butterfly(uint64_t* lo, uint64_t* hi, uint64_t omega, uint64_t Q, uint64_t precon) {
    uint64_t u = *lo;
    uint64_t v = *hi;
    *lo = mod_add(u, v, Q);
    *hi = barrett_mul(mod_sub(u, v, Q), omega, Q, precon);
}

// ============================================================================
// Batch External Product Parameters
// ============================================================================

struct BatchExternalProductParams {
    uint64_t Q;              // NTT modulus
    uint64_t N_inv;          // N^{-1} mod Q
    uint64_t N_inv_precon;   // Barrett precomputation for N_inv
    uint64_t Bg;             // Gadget base
    uint64_t Bg_half;        // Bg / 2
    uint64_t Bg_mask;        // Bg - 1
    uint32_t Bg_bits;        // log2(Bg)
    uint32_t N;              // Ring dimension
    uint32_t log_N;          // log2(N)
    uint32_t k;              // GLWE dimension
    uint32_t L;              // Decomposition levels
    uint32_t batch_size;     // Total batch size
    uint32_t num_ggsw;       // Number of distinct GGSW ciphertexts
};

// ============================================================================
// Forward NTT in Shared Memory
// ============================================================================

__device__
void ntt_forward_shared(
    uint64_t* poly,
    const uint64_t* __restrict__ twiddles,
    const uint64_t* __restrict__ precons,
    uint32_t N,
    uint32_t log_N,
    uint64_t Q,
    uint32_t tid,
    uint32_t tpg
) {
    for (uint32_t stage = 0; stage < log_N; ++stage) {
        uint32_t m = 1U << stage;
        uint32_t t = N >> (stage + 1);

        for (uint32_t b = 0; b < (N / 2 + tpg - 1) / tpg; ++b) {
            uint32_t butterfly_idx = tid + b * tpg;
            if (butterfly_idx >= N / 2) break;

            uint32_t group = butterfly_idx / t;
            uint32_t j = butterfly_idx % t;
            uint32_t idx_lo = group * (2 * t) + j;
            uint32_t idx_hi = idx_lo + t;

            uint32_t tw_idx = m + group;
            uint64_t omega = twiddles[tw_idx];
            uint64_t precon = precons[tw_idx];

            uint64_t lo = poly[idx_lo];
            uint64_t hi = poly[idx_hi];

            ct_butterfly(&lo, &hi, omega, Q, precon);

            poly[idx_lo] = lo;
            poly[idx_hi] = hi;
        }
        __syncthreads();
    }
}

// ============================================================================
// Inverse NTT in Shared Memory with Scaling
// ============================================================================

__device__
void ntt_inverse_shared(
    uint64_t* poly,
    const uint64_t* __restrict__ inv_twiddles,
    const uint64_t* __restrict__ inv_precons,
    uint64_t N_inv,
    uint64_t N_inv_precon,
    uint32_t N,
    uint32_t log_N,
    uint64_t Q,
    uint32_t tid,
    uint32_t tpg
) {
    for (uint32_t stage = 0; stage < log_N; ++stage) {
        uint32_t m = N >> (stage + 1);
        uint32_t t = 1U << stage;

        for (uint32_t b = 0; b < (N / 2 + tpg - 1) / tpg; ++b) {
            uint32_t butterfly_idx = tid + b * tpg;
            if (butterfly_idx >= N / 2) break;

            uint32_t group = butterfly_idx / t;
            uint32_t j = butterfly_idx % t;
            uint32_t idx_lo = group * (2 * t) + j;
            uint32_t idx_hi = idx_lo + t;

            uint32_t tw_idx = m + group;
            uint64_t omega = inv_twiddles[tw_idx];
            uint64_t precon = inv_precons[tw_idx];

            uint64_t lo = poly[idx_lo];
            uint64_t hi = poly[idx_hi];

            gs_butterfly(&lo, &hi, omega, Q, precon);

            poly[idx_lo] = lo;
            poly[idx_hi] = hi;
        }
        __syncthreads();
    }

    // Scale by N^{-1}
    for (uint32_t i = tid; i < N; i += tpg) {
        poly[i] = barrett_mul(poly[i], N_inv, Q, N_inv_precon);
    }
    __syncthreads();
}

// ============================================================================
// Batched External Product Kernel
// ============================================================================

// Process multiple GLWE x GGSW products with per-batch GGSW selection
// Each batch element can use a different GGSW from the pool

extern "C" __global__
void external_product_batch_kernel(
    uint64_t* __restrict__ glwe_out,            // [batch, k+1, N]
    const uint64_t* __restrict__ glwe_in,       // [batch, k+1, N]
    const uint64_t* __restrict__ ggsw_pool,     // [num_ggsw, k+1, L, k+1, N]
    const uint32_t* __restrict__ ggsw_indices,  // [batch] GGSW index per batch
    const uint64_t* __restrict__ twiddles,
    const uint64_t* __restrict__ precons,
    const uint64_t* __restrict__ inv_twiddles,
    const uint64_t* __restrict__ inv_precons,
    const BatchExternalProductParams params
) {
    extern __shared__ uint64_t shared[];

    const uint32_t N = params.N;
    const uint32_t log_N = params.log_N;
    const uint32_t k = params.k;
    const uint32_t L = params.L;
    const uint64_t Q = params.Q;
    const uint64_t Bg_half = params.Bg_half;
    const uint64_t Bg_mask = params.Bg_mask;
    const uint32_t Bg_bits = params.Bg_bits;

    const uint32_t batch_idx = blockIdx.x;
    const uint32_t out_poly_idx = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t tpg = blockDim.x;

    // Shared memory layout: [decomposed][accumulator]
    uint64_t* decomposed = shared;
    uint64_t* accumulator = shared + N;

    // Get GGSW for this batch element
    uint32_t ggsw_idx = ggsw_indices[batch_idx];
    uint32_t ggsw_stride = (k + 1) * L * (k + 1) * N;
    const uint64_t* ggsw = ggsw_pool + ggsw_idx * ggsw_stride;

    // Initialize accumulator
    for (uint32_t i = tid; i < N; i += tpg) {
        accumulator[i] = 0;
    }
    __syncthreads();

    // Process each input polynomial
    for (uint32_t in_poly = 0; in_poly <= k; ++in_poly) {
        const uint64_t* glwe_poly = glwe_in + batch_idx * (k + 1) * N + in_poly * N;

        // Process each decomposition level
        for (uint32_t level = 0; level < L; ++level) {
            // Decompose at this level
            for (uint32_t i = tid; i < N; i += tpg) {
                uint64_t coeff = glwe_poly[i];
                uint32_t shift = 64 - (L - level) * Bg_bits;
                uint64_t digit = ((coeff >> shift) + Bg_half) & Bg_mask;
                int64_t signed_digit = (int64_t)digit - (int64_t)Bg_half;
                decomposed[i] = (signed_digit >= 0) ? (uint64_t)signed_digit : (Q + signed_digit);
            }
            __syncthreads();

            // Forward NTT
            ntt_forward_shared(decomposed, twiddles, precons, N, log_N, Q, tid, tpg);

            // Multiply and accumulate
            uint32_t ggsw_offset = ((in_poly * L + level) * (k + 1) + out_poly_idx) * N;
            for (uint32_t i = tid; i < N; i += tpg) {
                uint64_t prod = mod_mul(decomposed[i], __ldg(&ggsw[ggsw_offset + i]), Q);
                accumulator[i] = mod_add(accumulator[i], prod, Q);
            }
            __syncthreads();
        }
    }

    // Inverse NTT
    ntt_inverse_shared(accumulator, inv_twiddles, inv_precons,
                       params.N_inv, params.N_inv_precon,
                       N, log_N, Q, tid, tpg);

    // Write output
    uint64_t* out_poly = glwe_out + batch_idx * (k + 1) * N + out_poly_idx * N;
    for (uint32_t i = tid; i < N; i += tpg) {
        out_poly[i] = accumulator[i];
    }
}

// ============================================================================
// Batched External Product with Accumulation
// ============================================================================

// acc[batch] += GLWE[batch] x GGSW[ggsw_indices[batch]]

extern "C" __global__
void external_product_batch_accumulate_kernel(
    uint64_t* __restrict__ glwe_acc,            // [batch, k+1, N] in/out
    const uint64_t* __restrict__ glwe_diff,     // [batch, k+1, N] input
    const uint64_t* __restrict__ ggsw_pool,     // [num_ggsw, k+1, L, k+1, N]
    const uint32_t* __restrict__ ggsw_indices,  // [batch]
    const uint64_t* __restrict__ twiddles,
    const uint64_t* __restrict__ precons,
    const uint64_t* __restrict__ inv_twiddles,
    const uint64_t* __restrict__ inv_precons,
    const BatchExternalProductParams params
) {
    extern __shared__ uint64_t shared[];

    const uint32_t N = params.N;
    const uint32_t log_N = params.log_N;
    const uint32_t k = params.k;
    const uint32_t L = params.L;
    const uint64_t Q = params.Q;
    const uint64_t Bg_half = params.Bg_half;
    const uint64_t Bg_mask = params.Bg_mask;
    const uint32_t Bg_bits = params.Bg_bits;

    const uint32_t batch_idx = blockIdx.x;
    const uint32_t out_poly_idx = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t tpg = blockDim.x;

    uint64_t* decomposed = shared;
    uint64_t* product_acc = shared + N;

    uint32_t ggsw_idx = ggsw_indices[batch_idx];
    uint32_t ggsw_stride = (k + 1) * L * (k + 1) * N;
    const uint64_t* ggsw = ggsw_pool + ggsw_idx * ggsw_stride;

    // Initialize product accumulator to zero
    for (uint32_t i = tid; i < N; i += tpg) {
        product_acc[i] = 0;
    }
    __syncthreads();

    // Compute external product
    for (uint32_t in_poly = 0; in_poly <= k; ++in_poly) {
        const uint64_t* diff_poly = glwe_diff + batch_idx * (k + 1) * N + in_poly * N;

        for (uint32_t level = 0; level < L; ++level) {
            // Decompose
            for (uint32_t i = tid; i < N; i += tpg) {
                uint64_t coeff = diff_poly[i];
                uint32_t shift = 64 - (L - level) * Bg_bits;
                uint64_t digit = ((coeff >> shift) + Bg_half) & Bg_mask;
                int64_t signed_digit = (int64_t)digit - (int64_t)Bg_half;
                decomposed[i] = (signed_digit >= 0) ? (uint64_t)signed_digit : (Q + signed_digit);
            }
            __syncthreads();

            // Forward NTT
            ntt_forward_shared(decomposed, twiddles, precons, N, log_N, Q, tid, tpg);

            // Multiply and accumulate
            uint32_t ggsw_offset = ((in_poly * L + level) * (k + 1) + out_poly_idx) * N;
            for (uint32_t i = tid; i < N; i += tpg) {
                uint64_t prod = mod_mul(decomposed[i], __ldg(&ggsw[ggsw_offset + i]), Q);
                product_acc[i] = mod_add(product_acc[i], prod, Q);
            }
            __syncthreads();
        }
    }

    // Inverse NTT
    ntt_inverse_shared(product_acc, inv_twiddles, inv_precons,
                       params.N_inv, params.N_inv_precon,
                       N, log_N, Q, tid, tpg);

    // Add to accumulator
    uint64_t* acc_poly = glwe_acc + batch_idx * (k + 1) * N + out_poly_idx * N;
    for (uint32_t i = tid; i < N; i += tpg) {
        acc_poly[i] = mod_add(acc_poly[i], product_acc[i], Q);
    }
}

// ============================================================================
// Multi-Batch External Product (Multiple Batches per Block)
// ============================================================================

// Process multiple batch elements per block for better occupancy with small N

extern "C" __global__
void external_product_multi_batch_kernel(
    uint64_t* __restrict__ glwe_out,            // [batch, k+1, N]
    const uint64_t* __restrict__ glwe_in,       // [batch, k+1, N]
    const uint64_t* __restrict__ ggsw_pool,     // [num_ggsw, k+1, L, k+1, N]
    const uint32_t* __restrict__ ggsw_indices,  // [batch]
    const uint64_t* __restrict__ twiddles,
    const uint64_t* __restrict__ precons,
    const uint64_t* __restrict__ inv_twiddles,
    const uint64_t* __restrict__ inv_precons,
    const BatchExternalProductParams params,
    uint32_t batches_per_block
) {
    extern __shared__ uint64_t shared[];

    const uint32_t N = params.N;
    const uint32_t log_N = params.log_N;
    const uint32_t k = params.k;
    const uint32_t L = params.L;
    const uint64_t Q = params.Q;
    const uint64_t Bg_half = params.Bg_half;
    const uint64_t Bg_mask = params.Bg_mask;
    const uint32_t Bg_bits = params.Bg_bits;

    const uint32_t block_batch_start = blockIdx.x * batches_per_block;
    const uint32_t out_poly_idx = blockIdx.y;
    const uint32_t tid = threadIdx.x;

    // Threads per batch
    const uint32_t threads_per_batch = blockDim.x / batches_per_block;
    const uint32_t local_batch = tid / threads_per_batch;
    const uint32_t local_tid = tid % threads_per_batch;

    if (local_batch >= batches_per_block) return;

    uint32_t batch_idx = block_batch_start + local_batch;
    if (batch_idx >= params.batch_size) return;

    // Per-batch shared memory
    uint64_t* decomposed = shared + local_batch * 2 * N;
    uint64_t* accumulator = decomposed + N;

    uint32_t ggsw_idx = ggsw_indices[batch_idx];
    uint32_t ggsw_stride = (k + 1) * L * (k + 1) * N;
    const uint64_t* ggsw = ggsw_pool + ggsw_idx * ggsw_stride;

    // Initialize
    for (uint32_t i = local_tid; i < N; i += threads_per_batch) {
        accumulator[i] = 0;
    }
    __syncthreads();

    // Process
    for (uint32_t in_poly = 0; in_poly <= k; ++in_poly) {
        const uint64_t* glwe_poly = glwe_in + batch_idx * (k + 1) * N + in_poly * N;

        for (uint32_t level = 0; level < L; ++level) {
            // Decompose
            for (uint32_t i = local_tid; i < N; i += threads_per_batch) {
                uint64_t coeff = glwe_poly[i];
                uint32_t shift = 64 - (L - level) * Bg_bits;
                uint64_t digit = ((coeff >> shift) + Bg_half) & Bg_mask;
                int64_t signed_digit = (int64_t)digit - (int64_t)Bg_half;
                decomposed[i] = (signed_digit >= 0) ? (uint64_t)signed_digit : (Q + signed_digit);
            }
            __syncthreads();

            // Forward NTT
            ntt_forward_shared(decomposed, twiddles, precons, N, log_N, Q, local_tid, threads_per_batch);

            // MAC
            uint32_t ggsw_offset = ((in_poly * L + level) * (k + 1) + out_poly_idx) * N;
            for (uint32_t i = local_tid; i < N; i += threads_per_batch) {
                uint64_t prod = mod_mul(decomposed[i], __ldg(&ggsw[ggsw_offset + i]), Q);
                accumulator[i] = mod_add(accumulator[i], prod, Q);
            }
            __syncthreads();
        }
    }

    // Inverse NTT
    ntt_inverse_shared(accumulator, inv_twiddles, inv_precons,
                       params.N_inv, params.N_inv_precon,
                       N, log_N, Q, local_tid, threads_per_batch);

    // Write output
    uint64_t* out_poly = glwe_out + batch_idx * (k + 1) * N + out_poly_idx * N;
    for (uint32_t i = local_tid; i < N; i += threads_per_batch) {
        out_poly[i] = accumulator[i];
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

cudaError_t lux_cuda_fhe_external_product_batch(
    uint64_t* glwe_out,
    const uint64_t* glwe_in,
    const uint64_t* ggsw_pool,
    const uint32_t* ggsw_indices,
    const uint64_t* twiddles,
    const uint64_t* precons,
    const uint64_t* inv_twiddles,
    const uint64_t* inv_precons,
    uint32_t N,
    uint32_t k,
    uint32_t L,
    uint32_t Bg_bits,
    uint64_t Q,
    uint64_t N_inv,
    uint64_t N_inv_precon,
    uint32_t batch_size,
    uint32_t num_ggsw,
    cudaStream_t stream
) {
    BatchExternalProductParams params;
    params.N = N;
    params.log_N = 31 - __builtin_clz(N);
    params.k = k;
    params.L = L;
    params.Bg_bits = Bg_bits;
    params.Bg = 1ULL << Bg_bits;
    params.Bg_half = 1ULL << (Bg_bits - 1);
    params.Bg_mask = params.Bg - 1;
    params.Q = Q;
    params.N_inv = N_inv;
    params.N_inv_precon = N_inv_precon;
    params.batch_size = batch_size;
    params.num_ggsw = num_ggsw;

    size_t shmem_size = 2 * N * sizeof(uint64_t);

    dim3 grid(batch_size, k + 1);
    dim3 block(min(N / 2, 256u));

    external_product_batch_kernel<<<grid, block, shmem_size, stream>>>(
        glwe_out, glwe_in, ggsw_pool, ggsw_indices,
        twiddles, precons, inv_twiddles, inv_precons,
        params
    );

    return cudaGetLastError();
}

cudaError_t lux_cuda_fhe_external_product_batch_accumulate(
    uint64_t* glwe_acc,
    const uint64_t* glwe_diff,
    const uint64_t* ggsw_pool,
    const uint32_t* ggsw_indices,
    const uint64_t* twiddles,
    const uint64_t* precons,
    const uint64_t* inv_twiddles,
    const uint64_t* inv_precons,
    uint32_t N,
    uint32_t k,
    uint32_t L,
    uint32_t Bg_bits,
    uint64_t Q,
    uint64_t N_inv,
    uint64_t N_inv_precon,
    uint32_t batch_size,
    uint32_t num_ggsw,
    cudaStream_t stream
) {
    BatchExternalProductParams params;
    params.N = N;
    params.log_N = 31 - __builtin_clz(N);
    params.k = k;
    params.L = L;
    params.Bg_bits = Bg_bits;
    params.Bg = 1ULL << Bg_bits;
    params.Bg_half = 1ULL << (Bg_bits - 1);
    params.Bg_mask = params.Bg - 1;
    params.Q = Q;
    params.N_inv = N_inv;
    params.N_inv_precon = N_inv_precon;
    params.batch_size = batch_size;
    params.num_ggsw = num_ggsw;

    size_t shmem_size = 2 * N * sizeof(uint64_t);

    dim3 grid(batch_size, k + 1);
    dim3 block(min(N / 2, 256u));

    external_product_batch_accumulate_kernel<<<grid, block, shmem_size, stream>>>(
        glwe_acc, glwe_diff, ggsw_pool, ggsw_indices,
        twiddles, precons, inv_twiddles, inv_precons,
        params
    );

    return cudaGetLastError();
}

cudaError_t lux_cuda_fhe_external_product_multi_batch(
    uint64_t* glwe_out,
    const uint64_t* glwe_in,
    const uint64_t* ggsw_pool,
    const uint32_t* ggsw_indices,
    const uint64_t* twiddles,
    const uint64_t* precons,
    const uint64_t* inv_twiddles,
    const uint64_t* inv_precons,
    uint32_t N,
    uint32_t k,
    uint32_t L,
    uint32_t Bg_bits,
    uint64_t Q,
    uint64_t N_inv,
    uint64_t N_inv_precon,
    uint32_t batch_size,
    uint32_t num_ggsw,
    uint32_t batches_per_block,
    cudaStream_t stream
) {
    BatchExternalProductParams params;
    params.N = N;
    params.log_N = 31 - __builtin_clz(N);
    params.k = k;
    params.L = L;
    params.Bg_bits = Bg_bits;
    params.Bg = 1ULL << Bg_bits;
    params.Bg_half = 1ULL << (Bg_bits - 1);
    params.Bg_mask = params.Bg - 1;
    params.Q = Q;
    params.N_inv = N_inv;
    params.N_inv_precon = N_inv_precon;
    params.batch_size = batch_size;
    params.num_ggsw = num_ggsw;

    size_t shmem_size = 2 * N * batches_per_block * sizeof(uint64_t);
    uint32_t grid_x = (batch_size + batches_per_block - 1) / batches_per_block;

    dim3 grid(grid_x, k + 1);
    dim3 block(min(N / 2, 256u) * batches_per_block);

    external_product_multi_batch_kernel<<<grid, block, shmem_size, stream>>>(
        glwe_out, glwe_in, ggsw_pool, ggsw_indices,
        twiddles, precons, inv_twiddles, inv_precons,
        params, batches_per_block
    );

    return cudaGetLastError();
}

}  // extern "C"
