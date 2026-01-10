// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// NTT (Number Theoretic Transform) CUDA Kernels for FHE
// Ported from Metal implementation - supports Cooley-Tukey (forward) and Gentleman-Sande (inverse)

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================================
// Constants and Configuration
// ============================================================================

#define MAX_SHARED_TWIDDLES 4096  // 32KB / 8 bytes per uint64_t
#define MAX_SHARED_POLY     4096  // Maximum N for fused kernel
#define WARP_SIZE           32

// ============================================================================
// NTT Parameters Structure (must match host)
// ============================================================================

struct NTTParams {
    uint64_t Q;            // Prime modulus (< 2^62)
    uint64_t mu;           // Barrett constant: floor(2^64 / Q)
    uint64_t N_inv;        // N^{-1} mod Q
    uint64_t N_inv_precon; // floor(2^64 * N_inv / Q)
    uint32_t N;            // Ring dimension (power of 2)
    uint32_t log_N;        // log2(N)
    uint32_t stage;        // Current stage (staged dispatch only)
    uint32_t batch;        // Batch size
};

// ============================================================================
// Modular Arithmetic Primitives
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

// Barrett reduction: computes (a * b) mod Q
// Requires precon = floor(2^64 * b / Q) precomputed
__device__ __forceinline__
uint64_t barrett_mul(uint64_t a, uint64_t omega, uint64_t Q, uint64_t precon) {
    // q_approx = floor(a * precon / 2^64) = high 64 bits of (a * precon)
    uint64_t q_approx = __umul64hi(a, precon);

    // result = a * omega - q_approx * Q
    uint64_t product = a * omega;
    uint64_t result = product - q_approx * Q;

    // Final reduction: result may be in [0, 2Q)
    return result >= Q ? result - Q : result;
}

// Full modular multiplication without precomputation
__device__ __forceinline__
uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);

    if (hi == 0) {
        return lo % Q;
    }

    // 2^64 mod Q = ((2^32 mod Q)^2) mod Q
    uint64_t r = (1ULL << 32) % Q;
    r = (r * r) % Q;

    // result = (lo mod Q + (hi mod Q) * r) mod Q
    uint64_t lo_mod = lo % Q;
    uint64_t hi_mod = (hi % Q) * r % Q;
    return mod_add(lo_mod, hi_mod, Q);
}

// ============================================================================
// Cooley-Tukey Butterfly (Forward NTT)
// ============================================================================

__device__ __forceinline__
void ct_butterfly(uint64_t* lo, uint64_t* hi, uint64_t omega, uint64_t Q, uint64_t precon) {
    uint64_t u = *lo;
    uint64_t v = barrett_mul(*hi, omega, Q, precon);
    *lo = mod_add(u, v, Q);
    *hi = mod_sub(u, v, Q);
}

// ============================================================================
// Gentleman-Sande Butterfly (Inverse NTT)
// ============================================================================

__device__ __forceinline__
void gs_butterfly(uint64_t* lo, uint64_t* hi, uint64_t omega, uint64_t Q, uint64_t precon) {
    uint64_t u = *lo;
    uint64_t v = *hi;
    *lo = mod_add(u, v, Q);
    *hi = barrett_mul(mod_sub(u, v, Q), omega, Q, precon);
}

// ============================================================================
// Staged NTT Kernels (for large N or when shared memory insufficient)
// ============================================================================

// Forward NTT - Single Stage (Cooley-Tukey)
extern "C" __global__
void ntt_forward_stage(
    uint64_t* __restrict__ data,
    const uint64_t* __restrict__ twiddles,
    const uint64_t* __restrict__ precons,
    const NTTParams params
) {
    const uint32_t N = params.N;
    const uint64_t Q = params.Q;
    const uint32_t stage = params.stage;
    const uint32_t batch_idx = blockIdx.y;

    // Butterfly index within this stage
    const uint32_t butterfly_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_butterflies = N >> 1;

    if (butterfly_idx >= total_butterflies) return;

    // Compute m = 2^stage, t = N / (2*m) = 2^(log_N - stage - 1)
    const uint32_t m = 1U << stage;
    const uint32_t t = N >> (stage + 1);

    // Map butterfly_idx to (group, position within group)
    const uint32_t group = butterfly_idx / t;
    const uint32_t j = butterfly_idx % t;

    // Compute indices
    const uint32_t idx_lo = group * (2 * t) + j;
    const uint32_t idx_hi = idx_lo + t;

    // Data offset for this batch
    uint64_t* poly = data + batch_idx * N;

    // Load twiddle factor (OpenFHE layout: twiddles[m + group])
    const uint32_t tw_idx = m + group;
    uint64_t omega = twiddles[tw_idx];
    uint64_t precon = precons[tw_idx];

    // Load values
    uint64_t lo = poly[idx_lo];
    uint64_t hi = poly[idx_hi];

    // Perform Cooley-Tukey butterfly
    ct_butterfly(&lo, &hi, omega, Q, precon);

    // Store results
    poly[idx_lo] = lo;
    poly[idx_hi] = hi;
}

// Inverse NTT - Single Stage (Gentleman-Sande)
extern "C" __global__
void ntt_inverse_stage(
    uint64_t* __restrict__ data,
    const uint64_t* __restrict__ inv_twiddles,
    const uint64_t* __restrict__ inv_precons,
    const NTTParams params
) {
    const uint32_t N = params.N;
    const uint64_t Q = params.Q;
    const uint32_t stage = params.stage;
    const uint32_t batch_idx = blockIdx.y;

    const uint32_t butterfly_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_butterflies = N >> 1;

    if (butterfly_idx >= total_butterflies) return;

    // For inverse: m starts at N/2 and decreases
    // stage 0 => m = N/2, stage log_N-1 => m = 1
    const uint32_t m = N >> (stage + 1);
    const uint32_t t = 1U << stage;

    const uint32_t group = butterfly_idx / t;
    const uint32_t j = butterfly_idx % t;

    const uint32_t idx_lo = group * (2 * t) + j;
    const uint32_t idx_hi = idx_lo + t;

    uint64_t* poly = data + batch_idx * N;

    const uint32_t tw_idx = m + group;
    uint64_t omega = inv_twiddles[tw_idx];
    uint64_t precon = inv_precons[tw_idx];

    uint64_t lo = poly[idx_lo];
    uint64_t hi = poly[idx_hi];

    gs_butterfly(&lo, &hi, omega, Q, precon);

    poly[idx_lo] = lo;
    poly[idx_hi] = hi;
}

// ============================================================================
// Fused NTT Kernels (all stages in shared memory, N <= 4096)
// ============================================================================

extern "C" __global__
void ntt_forward_fused(
    uint64_t* __restrict__ data,
    const uint64_t* __restrict__ twiddles,
    const uint64_t* __restrict__ precons,
    const NTTParams params
) {
    extern __shared__ uint64_t shared[];
    uint64_t* poly_shared = shared;
    uint64_t* tw_shared = shared + params.N;

    const uint32_t N = params.N;
    const uint32_t log_N = params.log_N;
    const uint64_t Q = params.Q;
    const uint32_t batch_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t tpg = blockDim.x;

    // Global data pointer for this batch
    uint64_t* poly = data + batch_idx * N;

    // Cooperative load: polynomial to shared memory
    for (uint32_t i = tid; i < N; i += tpg) {
        poly_shared[i] = poly[i];
    }

    // Cooperative load: twiddles to shared memory
    for (uint32_t i = tid; i < N; i += tpg) {
        tw_shared[i] = twiddles[i];
    }
    __syncthreads();

    // Precons loaded on-demand via texture cache (__ldg)

    // Process all log_N stages
    for (uint32_t stage = 0; stage < log_N; ++stage) {
        const uint32_t m = 1U << stage;
        const uint32_t t = N >> (stage + 1);
        const uint32_t butterflies_per_thread = (N / 2 + tpg - 1) / tpg;

        for (uint32_t b = 0; b < butterflies_per_thread; ++b) {
            uint32_t butterfly_idx = tid + b * tpg;
            if (butterfly_idx >= N / 2) break;

            uint32_t group = butterfly_idx / t;
            uint32_t j = butterfly_idx % t;
            uint32_t idx_lo = group * (2 * t) + j;
            uint32_t idx_hi = idx_lo + t;

            uint32_t tw_idx = m + group;
            uint64_t omega = tw_shared[tw_idx];
            uint64_t precon = __ldg(&precons[tw_idx]);

            uint64_t lo = poly_shared[idx_lo];
            uint64_t hi = poly_shared[idx_hi];

            ct_butterfly(&lo, &hi, omega, Q, precon);

            poly_shared[idx_lo] = lo;
            poly_shared[idx_hi] = hi;
        }
        __syncthreads();
    }

    // Write back to global memory
    for (uint32_t i = tid; i < N; i += tpg) {
        poly[i] = poly_shared[i];
    }
}

extern "C" __global__
void ntt_inverse_fused(
    uint64_t* __restrict__ data,
    const uint64_t* __restrict__ inv_twiddles,
    const uint64_t* __restrict__ inv_precons,
    const NTTParams params
) {
    extern __shared__ uint64_t shared[];
    uint64_t* poly_shared = shared;
    uint64_t* tw_shared = shared + params.N;

    const uint32_t N = params.N;
    const uint32_t log_N = params.log_N;
    const uint64_t Q = params.Q;
    const uint64_t N_inv = params.N_inv;
    const uint64_t N_inv_precon = params.N_inv_precon;
    const uint32_t batch_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t tpg = blockDim.x;

    uint64_t* poly = data + batch_idx * N;

    // Cooperative load
    for (uint32_t i = tid; i < N; i += tpg) {
        poly_shared[i] = poly[i];
    }
    for (uint32_t i = tid; i < N; i += tpg) {
        tw_shared[i] = inv_twiddles[i];
    }
    __syncthreads();

    // Process all log_N stages (reverse direction)
    for (uint32_t stage = 0; stage < log_N; ++stage) {
        const uint32_t m = N >> (stage + 1);
        const uint32_t t = 1U << stage;
        const uint32_t butterflies_per_thread = (N / 2 + tpg - 1) / tpg;

        for (uint32_t b = 0; b < butterflies_per_thread; ++b) {
            uint32_t butterfly_idx = tid + b * tpg;
            if (butterfly_idx >= N / 2) break;

            uint32_t group = butterfly_idx / t;
            uint32_t j = butterfly_idx % t;
            uint32_t idx_lo = group * (2 * t) + j;
            uint32_t idx_hi = idx_lo + t;

            uint32_t tw_idx = m + group;
            uint64_t omega = tw_shared[tw_idx];
            uint64_t precon = __ldg(&inv_precons[tw_idx]);

            uint64_t lo = poly_shared[idx_lo];
            uint64_t hi = poly_shared[idx_hi];

            gs_butterfly(&lo, &hi, omega, Q, precon);

            poly_shared[idx_lo] = lo;
            poly_shared[idx_hi] = hi;
        }
        __syncthreads();
    }

    // Scale by N^{-1} and write back
    for (uint32_t i = tid; i < N; i += tpg) {
        poly[i] = barrett_mul(poly_shared[i], N_inv, Q, N_inv_precon);
    }
}

// ============================================================================
// Four-Step NTT (for N > 4096)
// ============================================================================

#define TILE_SIZE 16

// Row NTT + diagonal twiddle + transposed write (fused)
extern "C" __global__
void ntt_four_step_row_fused(
    uint64_t* __restrict__ data,
    const uint64_t* __restrict__ row_twiddles,
    const uint64_t* __restrict__ row_precons,
    const uint64_t* __restrict__ diag_twiddles,
    const uint64_t* __restrict__ diag_precons,
    const uint32_t n1,         // Row size
    const uint32_t n2,         // Number of rows
    const uint32_t log_n1,
    const uint64_t Q,
    const uint32_t batch_idx
) {
    extern __shared__ uint64_t row_shared[];

    const uint32_t row_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t tpg = blockDim.x;

    uint64_t* batch_data = data + batch_idx * n1 * n2;

    // Load row to shared memory
    for (uint32_t i = tid; i < n1; i += tpg) {
        row_shared[i] = batch_data[row_idx * n1 + i];
    }
    __syncthreads();

    // Perform n1-point NTT on this row
    for (uint32_t stage = 0; stage < log_n1; ++stage) {
        const uint32_t m = 1U << stage;
        const uint32_t t = n1 >> (stage + 1);

        for (uint32_t b = 0; b < (n1 / 2 + tpg - 1) / tpg; ++b) {
            uint32_t butterfly_idx = tid + b * tpg;
            if (butterfly_idx >= n1 / 2) break;

            uint32_t group = butterfly_idx / t;
            uint32_t j = butterfly_idx % t;
            uint32_t idx_lo = group * (2 * t) + j;
            uint32_t idx_hi = idx_lo + t;

            uint32_t tw_idx = m + group;
            uint64_t omega = row_twiddles[tw_idx];
            uint64_t precon = row_precons[tw_idx];

            uint64_t lo = row_shared[idx_lo];
            uint64_t hi = row_shared[idx_hi];

            ct_butterfly(&lo, &hi, omega, Q, precon);

            row_shared[idx_lo] = lo;
            row_shared[idx_hi] = hi;
        }
        __syncthreads();
    }

    // Apply diagonal twiddles and write transposed
    for (uint32_t col = tid; col < n1; col += tpg) {
        uint32_t diag_idx = row_idx * n1 + col;
        uint64_t val = row_shared[col];
        val = barrett_mul(val, diag_twiddles[diag_idx], Q, diag_precons[diag_idx]);
        // Transposed write: column-major output
        batch_data[col * n2 + row_idx] = val;
    }
}

// Column NTT (after transpose)
extern "C" __global__
void ntt_four_step_col(
    uint64_t* __restrict__ data,
    const uint64_t* __restrict__ col_twiddles,
    const uint64_t* __restrict__ col_precons,
    const uint32_t n1,         // Now number of rows (was columns)
    const uint32_t n2,         // Column size
    const uint32_t log_n2,
    const uint64_t Q,
    const uint32_t batch_idx
) {
    extern __shared__ uint64_t col_shared[];

    const uint32_t col_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t tpg = blockDim.x;

    uint64_t* batch_data = data + batch_idx * n1 * n2;

    // Load column to shared memory (now contiguous after transpose)
    for (uint32_t i = tid; i < n2; i += tpg) {
        col_shared[i] = batch_data[col_idx * n2 + i];
    }
    __syncthreads();

    // Perform n2-point NTT on this column
    for (uint32_t stage = 0; stage < log_n2; ++stage) {
        const uint32_t m = 1U << stage;
        const uint32_t t = n2 >> (stage + 1);

        for (uint32_t b = 0; b < (n2 / 2 + tpg - 1) / tpg; ++b) {
            uint32_t butterfly_idx = tid + b * tpg;
            if (butterfly_idx >= n2 / 2) break;

            uint32_t group = butterfly_idx / t;
            uint32_t j = butterfly_idx % t;
            uint32_t idx_lo = group * (2 * t) + j;
            uint32_t idx_hi = idx_lo + t;

            uint32_t tw_idx = m + group;
            uint64_t omega = col_twiddles[tw_idx];
            uint64_t precon = col_precons[tw_idx];

            uint64_t lo = col_shared[idx_lo];
            uint64_t hi = col_shared[idx_hi];

            ct_butterfly(&lo, &hi, omega, Q, precon);

            col_shared[idx_lo] = lo;
            col_shared[idx_hi] = hi;
        }
        __syncthreads();
    }

    // Write back (still in transposed layout)
    for (uint32_t i = tid; i < n2; i += tpg) {
        batch_data[col_idx * n2 + i] = col_shared[i];
    }
}

// ============================================================================
// Utility Kernels
// ============================================================================

// Pointwise modular multiplication of two polynomials
extern "C" __global__
void ntt_pointwise_mul(
    uint64_t* __restrict__ result,
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    const uint64_t Q,
    const uint32_t N,
    const uint32_t batch_size
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = N * batch_size;

    if (idx < total) {
        result[idx] = mod_mul(a[idx], b[idx], Q);
    }
}

// Scale polynomial by constant (for INTT final step)
extern "C" __global__
void ntt_scale(
    uint64_t* __restrict__ data,
    const uint64_t scale,
    const uint64_t scale_precon,
    const uint64_t Q,
    const uint32_t N,
    const uint32_t batch_size
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = N * batch_size;

    if (idx < total) {
        data[idx] = barrett_mul(data[idx], scale, Q, scale_precon);
    }
}
