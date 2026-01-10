// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// TFHE External Product CUDA Kernel
// Implements GLWE x GGSW -> GLWE external product, the core operation for
// programmable bootstrapping and CMux gates.
//
// The external product computes: GLWE * GGSW = GLWE
// Where GGSW is the gadget-based encryption of a bit, enabling CMux selection.

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================================
// Constants
// ============================================================================

#define MAX_N 2048
#define MAX_K 2
#define MAX_L 8
#define WARP_SIZE 32

// ============================================================================
// Modular Arithmetic (Torus mod 2^64)
// ============================================================================

__device__ __forceinline__
uint64_t torus_add(uint64_t a, uint64_t b) {
    return a + b;  // Natural overflow gives mod 2^64
}

__device__ __forceinline__
uint64_t torus_sub(uint64_t a, uint64_t b) {
    return a - b;  // Natural underflow gives mod 2^64
}

__device__ __forceinline__
uint64_t torus_neg(uint64_t a) {
    return (~a) + 1;  // Two's complement negation
}

// Modular arithmetic for prime modulus (NTT domain)
__device__ __forceinline__
uint64_t mod_add(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t sum = a + b;
    return sum >= Q ? sum - Q : sum;
}

__device__ __forceinline__
uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t Q) {
    return a >= b ? a - b : a + Q - b;
}

// Barrett multiplication: (a * b) mod Q
// Requires precon = floor(2^64 * b / Q) precomputed for twiddles
__device__ __forceinline__
uint64_t barrett_mul(uint64_t a, uint64_t omega, uint64_t Q, uint64_t precon) {
    uint64_t q_approx = __umul64hi(a, precon);
    uint64_t product = a * omega;
    uint64_t result = product - q_approx * Q;
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

    uint64_t lo_mod = lo % Q;
    uint64_t hi_mod = (hi % Q) * r % Q;
    return mod_add(lo_mod, hi_mod, Q);
}

// ============================================================================
// Gadget Decomposition
// ============================================================================

// Parameters for gadget decomposition
struct GadgetParams {
    uint64_t Bg;          // Gadget base (typically 2^Bg_bits)
    uint64_t Bg_half;     // Bg / 2 for signed decomposition
    uint64_t Bg_mask;     // Bg - 1 (for fast modulo when Bg is power of 2)
    uint32_t Bg_bits;     // log2(Bg)
    uint32_t L;           // Number of decomposition levels
};

// Signed gadget decomposition into L digits
// Each digit is in [-Bg/2, Bg/2) and represents coefficient at level l
__device__ __forceinline__
void gadget_decompose_signed(
    uint64_t coeff,
    int64_t* digits,
    const GadgetParams& params
) {
    // Shift to align with decomposition levels
    // For Torus mod 2^64: we decompose the most significant bits
    uint32_t total_shift = 64 - params.L * params.Bg_bits;
    uint64_t val = (coeff >> total_shift) + params.Bg_half;

    for (uint32_t l = 0; l < params.L; ++l) {
        // Extract digit in [0, Bg)
        uint64_t digit = val & params.Bg_mask;

        // Center around 0: map [0, Bg) to [-Bg/2, Bg/2)
        int64_t signed_digit = (int64_t)digit - (int64_t)params.Bg_half;
        digits[l] = signed_digit;

        // Prepare for next level
        val >>= params.Bg_bits;

        // Propagate carry if needed (for cleaner decomposition)
        if (l + 1 < params.L && signed_digit < 0) {
            val += 1;  // Carry
        }
    }
}

// Alternative: Unsigned decomposition for NTT-friendly processing
__device__ __forceinline__
void gadget_decompose_unsigned(
    uint64_t coeff,
    uint64_t* digits,
    const GadgetParams& params,
    uint64_t Q
) {
    uint32_t total_shift = 64 - params.L * params.Bg_bits;
    uint64_t val = (coeff >> total_shift);

    for (uint32_t l = 0; l < params.L; ++l) {
        uint64_t digit = val & params.Bg_mask;

        // Round to nearest
        if (digit > params.Bg_half) {
            // Negative value: store as Q - (Bg - digit)
            digits[l] = Q - (params.Bg - digit);
            val = (val >> params.Bg_bits) + 1;  // Carry
        } else {
            digits[l] = digit;
            val >>= params.Bg_bits;
        }
    }
}

// ============================================================================
// NTT Operations (in-shared-memory)
// ============================================================================

// Cooley-Tukey butterfly for forward NTT
__device__ __forceinline__
void ct_butterfly(uint64_t* lo, uint64_t* hi, uint64_t omega, uint64_t Q, uint64_t precon) {
    uint64_t u = *lo;
    uint64_t v = barrett_mul(*hi, omega, Q, precon);
    *lo = mod_add(u, v, Q);
    *hi = mod_sub(u, v, Q);
}

// Gentleman-Sande butterfly for inverse NTT
__device__ __forceinline__
void gs_butterfly(uint64_t* lo, uint64_t* hi, uint64_t omega, uint64_t Q, uint64_t precon) {
    uint64_t u = *lo;
    uint64_t v = *hi;
    *lo = mod_add(u, v, Q);
    *hi = barrett_mul(mod_sub(u, v, Q), omega, Q, precon);
}

// Forward NTT in shared memory
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

// Inverse NTT in shared memory with N^-1 scaling
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
// External Product Parameters
// ============================================================================

struct ExternalProductParams {
    uint64_t Q;              // NTT modulus
    uint64_t N_inv;          // N^{-1} mod Q
    uint64_t N_inv_precon;   // Barrett precomputation for N_inv
    uint32_t N;              // Ring dimension
    uint32_t log_N;          // log2(N)
    uint32_t k;              // GLWE dimension (typically 1)
    uint32_t L;              // Decomposition levels
    uint32_t Bg_bits;        // Gadget base bits
    uint64_t Bg;             // Gadget base
    uint64_t Bg_half;        // Bg / 2
    uint64_t Bg_mask;        // Bg - 1
    uint32_t batch_size;     // Number of external products
};

// ============================================================================
// External Product Kernel: GLWE x GGSW -> GLWE
// ============================================================================

// Layout of GGSW: [k+1][L][k+1][N] in NTT domain
// For each row (input component, decomposition level), we have (k+1) polynomials

extern "C" __global__
void tfhe_external_product(
    uint64_t* __restrict__ glwe_out,            // [batch, k+1, N] output GLWE
    const uint64_t* __restrict__ glwe_in,       // [batch, k+1, N] input GLWE
    const uint64_t* __restrict__ ggsw,          // [batch or 1, k+1, L, k+1, N] GGSW in NTT
    const uint64_t* __restrict__ twiddles,      // [N] forward twiddles
    const uint64_t* __restrict__ precons,       // [N] Barrett precomputations
    const uint64_t* __restrict__ inv_twiddles,  // [N] inverse twiddles
    const uint64_t* __restrict__ inv_precons,   // [N] inverse precomputations
    const ExternalProductParams params,
    bool ggsw_broadcast                          // If true, use same GGSW for all batch
) {
    extern __shared__ uint64_t shared[];

    const uint32_t N = params.N;
    const uint32_t log_N = params.log_N;
    const uint32_t k = params.k;
    const uint32_t L = params.L;
    const uint64_t Q = params.Q;

    const uint32_t batch_idx = blockIdx.x;
    const uint32_t out_poly_idx = blockIdx.y;  // Which output polynomial (0..k)
    const uint32_t tid = threadIdx.x;
    const uint32_t tpg = blockDim.x;

    // Shared memory layout:
    // [decomposed_poly][accumulator][work_poly]
    uint64_t* decomposed = shared;                    // [N]
    uint64_t* accumulator = shared + N;               // [N]
    uint64_t* work = shared + 2 * N;                  // [N]

    // GGSW pointer (handle broadcast)
    const uint64_t* ggsw_batch = ggsw_broadcast ? ggsw : (ggsw + batch_idx * (k + 1) * L * (k + 1) * N);

    // Initialize accumulator to zero
    for (uint32_t i = tid; i < N; i += tpg) {
        accumulator[i] = 0;
    }
    __syncthreads();

    // Gadget parameters
    GadgetParams gadget_params;
    gadget_params.Bg = params.Bg;
    gadget_params.Bg_half = params.Bg_half;
    gadget_params.Bg_mask = params.Bg_mask;
    gadget_params.Bg_bits = params.Bg_bits;
    gadget_params.L = L;

    // For each input polynomial (in_poly = 0..k)
    for (uint32_t in_poly = 0; in_poly <= k; ++in_poly) {
        // Load input GLWE polynomial
        const uint64_t* glwe_poly = glwe_in + batch_idx * (k + 1) * N + in_poly * N;

        // For each decomposition level
        for (uint32_t level = 0; level < L; ++level) {
            // Step 1: Decompose the polynomial coefficient-wise
            for (uint32_t i = tid; i < N; i += tpg) {
                uint64_t coeff = glwe_poly[i];

                // Extract digit at this level
                uint32_t total_shift = 64 - (L - level) * params.Bg_bits;
                uint64_t val = (coeff >> total_shift);
                uint64_t digit = (val + params.Bg_half) & params.Bg_mask;

                // Convert to centered representation in [0, Q)
                int64_t signed_digit = (int64_t)digit - (int64_t)params.Bg_half;
                decomposed[i] = (signed_digit >= 0) ? (uint64_t)signed_digit : (Q + signed_digit);
            }
            __syncthreads();

            // Step 2: Forward NTT of decomposed polynomial
            ntt_forward_shared(decomposed, twiddles, precons, N, log_N, Q, tid, tpg);

            // Step 3: Pointwise multiply with GGSW and accumulate
            // GGSW layout: ggsw[in_poly][level][out_poly][coeff]
            uint32_t ggsw_offset = ((in_poly * L + level) * (k + 1) + out_poly_idx) * N;

            for (uint32_t i = tid; i < N; i += tpg) {
                uint64_t d = decomposed[i];
                uint64_t g = __ldg(&ggsw_batch[ggsw_offset + i]);
                uint64_t prod = mod_mul(d, g, Q);
                accumulator[i] = mod_add(accumulator[i], prod, Q);
            }
            __syncthreads();
        }
    }

    // Step 4: Inverse NTT of accumulator
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
// External Product with Accumulation: acc += GLWE x GGSW
// ============================================================================

// Used for CMux: acc = d0 + GGSW * (d1 - d0)
// This version adds the result to existing accumulator

extern "C" __global__
void tfhe_external_product_accumulate(
    uint64_t* __restrict__ glwe_acc,            // [batch, k+1, N] accumulator (in/out)
    const uint64_t* __restrict__ glwe_diff,     // [batch, k+1, N] difference (d1 - d0)
    const uint64_t* __restrict__ ggsw,          // [batch or 1, k+1, L, k+1, N] GGSW
    const uint64_t* __restrict__ twiddles,
    const uint64_t* __restrict__ precons,
    const uint64_t* __restrict__ inv_twiddles,
    const uint64_t* __restrict__ inv_precons,
    const ExternalProductParams params,
    bool ggsw_broadcast
) {
    extern __shared__ uint64_t shared[];

    const uint32_t N = params.N;
    const uint32_t log_N = params.log_N;
    const uint32_t k = params.k;
    const uint32_t L = params.L;
    const uint64_t Q = params.Q;

    const uint32_t batch_idx = blockIdx.x;
    const uint32_t out_poly_idx = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t tpg = blockDim.x;

    uint64_t* decomposed = shared;
    uint64_t* product_acc = shared + N;

    const uint64_t* ggsw_batch = ggsw_broadcast ? ggsw : (ggsw + batch_idx * (k + 1) * L * (k + 1) * N);

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
                uint32_t total_shift = 64 - (L - level) * params.Bg_bits;
                uint64_t val = (coeff >> total_shift);
                uint64_t digit = (val + params.Bg_half) & params.Bg_mask;
                int64_t signed_digit = (int64_t)digit - (int64_t)params.Bg_half;
                decomposed[i] = (signed_digit >= 0) ? (uint64_t)signed_digit : (Q + signed_digit);
            }
            __syncthreads();

            // Forward NTT
            ntt_forward_shared(decomposed, twiddles, precons, N, log_N, Q, tid, tpg);

            // Multiply and accumulate
            uint32_t ggsw_offset = ((in_poly * L + level) * (k + 1) + out_poly_idx) * N;
            for (uint32_t i = tid; i < N; i += tpg) {
                uint64_t prod = mod_mul(decomposed[i], __ldg(&ggsw_batch[ggsw_offset + i]), Q);
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
// Batched External Product (multiple GGSW, parallel)
// ============================================================================

extern "C" __global__
void tfhe_external_product_batch(
    uint64_t* __restrict__ glwe_out,            // [batch, k+1, N]
    const uint64_t* __restrict__ glwe_in,       // [batch, k+1, N]
    const uint64_t* __restrict__ ggsw_array,    // [num_ggsw, k+1, L, k+1, N]
    const uint32_t* __restrict__ ggsw_indices,  // [batch] which GGSW to use
    const uint64_t* __restrict__ twiddles,
    const uint64_t* __restrict__ precons,
    const uint64_t* __restrict__ inv_twiddles,
    const uint64_t* __restrict__ inv_precons,
    const ExternalProductParams params
) {
    extern __shared__ uint64_t shared[];

    const uint32_t N = params.N;
    const uint32_t log_N = params.log_N;
    const uint32_t k = params.k;
    const uint32_t L = params.L;
    const uint64_t Q = params.Q;

    const uint32_t batch_idx = blockIdx.x;
    const uint32_t out_poly_idx = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t tpg = blockDim.x;

    uint64_t* decomposed = shared;
    uint64_t* accumulator = shared + N;

    // Get GGSW for this batch element
    uint32_t ggsw_idx = ggsw_indices[batch_idx];
    const uint64_t* ggsw = ggsw_array + ggsw_idx * (k + 1) * L * (k + 1) * N;

    // Initialize accumulator
    for (uint32_t i = tid; i < N; i += tpg) {
        accumulator[i] = 0;
    }
    __syncthreads();

    // External product computation
    for (uint32_t in_poly = 0; in_poly <= k; ++in_poly) {
        const uint64_t* glwe_poly = glwe_in + batch_idx * (k + 1) * N + in_poly * N;

        for (uint32_t level = 0; level < L; ++level) {
            // Decompose
            for (uint32_t i = tid; i < N; i += tpg) {
                uint64_t coeff = glwe_poly[i];
                uint32_t total_shift = 64 - (L - level) * params.Bg_bits;
                uint64_t digit = ((coeff >> total_shift) + params.Bg_half) & params.Bg_mask;
                int64_t signed_digit = (int64_t)digit - (int64_t)params.Bg_half;
                decomposed[i] = (signed_digit >= 0) ? (uint64_t)signed_digit : (Q + signed_digit);
            }
            __syncthreads();

            ntt_forward_shared(decomposed, twiddles, precons, N, log_N, Q, tid, tpg);

            uint32_t ggsw_offset = ((in_poly * L + level) * (k + 1) + out_poly_idx) * N;
            for (uint32_t i = tid; i < N; i += tpg) {
                uint64_t prod = mod_mul(decomposed[i], __ldg(&ggsw[ggsw_offset + i]), Q);
                accumulator[i] = mod_add(accumulator[i], prod, Q);
            }
            __syncthreads();
        }
    }

    ntt_inverse_shared(accumulator, inv_twiddles, inv_precons,
                       params.N_inv, params.N_inv_precon,
                       N, log_N, Q, tid, tpg);

    uint64_t* out_poly = glwe_out + batch_idx * (k + 1) * N + out_poly_idx * N;
    for (uint32_t i = tid; i < N; i += tpg) {
        out_poly[i] = accumulator[i];
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

// External product: glwe_out = glwe_in * ggsw
cudaError_t lux_cuda_tfhe_external_product(
    uint64_t* glwe_out,
    const uint64_t* glwe_in,
    const uint64_t* ggsw,
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
    bool ggsw_broadcast,
    cudaStream_t stream
) {
    ExternalProductParams params;
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

    // Shared memory: 3 * N * sizeof(uint64_t)
    size_t shmem_size = 3 * N * sizeof(uint64_t);

    // Launch: one block per (batch, output polynomial)
    dim3 grid(batch_size, k + 1);
    dim3 block(min(N / 2, 256u));

    tfhe_external_product<<<grid, block, shmem_size, stream>>>(
        glwe_out, glwe_in, ggsw,
        twiddles, precons, inv_twiddles, inv_precons,
        params, ggsw_broadcast
    );

    return cudaGetLastError();
}

// External product with accumulation: acc += diff * ggsw
cudaError_t lux_cuda_tfhe_external_product_accumulate(
    uint64_t* glwe_acc,
    const uint64_t* glwe_diff,
    const uint64_t* ggsw,
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
    bool ggsw_broadcast,
    cudaStream_t stream
) {
    ExternalProductParams params;
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

    size_t shmem_size = 2 * N * sizeof(uint64_t);

    dim3 grid(batch_size, k + 1);
    dim3 block(min(N / 2, 256u));

    tfhe_external_product_accumulate<<<grid, block, shmem_size, stream>>>(
        glwe_acc, glwe_diff, ggsw,
        twiddles, precons, inv_twiddles, inv_precons,
        params, ggsw_broadcast
    );

    return cudaGetLastError();
}

// Batched external product with index selection
cudaError_t lux_cuda_tfhe_external_product_batch(
    uint64_t* glwe_out,
    const uint64_t* glwe_in,
    const uint64_t* ggsw_array,
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
    cudaStream_t stream
) {
    ExternalProductParams params;
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

    size_t shmem_size = 2 * N * sizeof(uint64_t);

    dim3 grid(batch_size, k + 1);
    dim3 block(min(N / 2, 256u));

    tfhe_external_product_batch<<<grid, block, shmem_size, stream>>>(
        glwe_out, glwe_in, ggsw_array, ggsw_indices,
        twiddles, precons, inv_twiddles, inv_precons,
        params
    );

    return cudaGetLastError();
}

}  // extern "C"
