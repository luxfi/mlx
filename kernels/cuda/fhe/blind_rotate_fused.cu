// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// TFHE Blind Rotation CUDA Kernel
// Implements the full bootstrapping critical path with fused operations

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================================
// Constants
// ============================================================================

#define MAX_N 1024
#define MAX_L 4
#define WARP_SIZE 32

// ============================================================================
// Modular Arithmetic (same as NTT kernels)
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

__device__ __forceinline__
uint64_t barrett_mul(uint64_t a, uint64_t omega, uint64_t Q, uint64_t precon) {
    uint64_t q_approx = __umul64hi(a, precon);
    uint64_t product = a * omega;
    uint64_t result = product - q_approx * Q;
    return result >= Q ? result - Q : result;
}

// ============================================================================
// Butterfly Operations
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
// NTT in Shared Memory
// ============================================================================

__device__
void ntt_forward_shared(
    uint64_t* shared_poly,
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

            uint64_t lo = shared_poly[idx_lo];
            uint64_t hi = shared_poly[idx_hi];

            ct_butterfly(&lo, &hi, omega, Q, precon);

            shared_poly[idx_lo] = lo;
            shared_poly[idx_hi] = hi;
        }
        __syncthreads();
    }
}

__device__
void ntt_inverse_shared(
    uint64_t* shared_poly,
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

            uint64_t lo = shared_poly[idx_lo];
            uint64_t hi = shared_poly[idx_hi];

            gs_butterfly(&lo, &hi, omega, Q, precon);

            shared_poly[idx_lo] = lo;
            shared_poly[idx_hi] = hi;
        }
        __syncthreads();
    }

    // Scale by N^{-1}
    for (uint32_t i = tid; i < N; i += tpg) {
        shared_poly[i] = barrett_mul(shared_poly[i], N_inv, Q, N_inv_precon);
    }
    __syncthreads();
}

// ============================================================================
// Negacyclic Rotation
// ============================================================================

// Rotate polynomial by k positions in Z_Q[X]/(X^N + 1)
// X^N = -1, so rotation wraps with negation
__device__
void negacyclic_rotate(
    uint64_t* dst,
    const uint64_t* src,
    int32_t k,
    uint32_t N,
    uint64_t Q,
    uint32_t tid,
    uint32_t tpg
) {
    // Normalize k to [0, 2N)
    k = ((k % (2 * (int32_t)N)) + 2 * (int32_t)N) % (2 * (int32_t)N);

    for (uint32_t i = tid; i < N; i += tpg) {
        int32_t src_idx = (int32_t)i - k;
        bool negate = false;

        // Handle wrap-around with negation
        while (src_idx < 0) {
            src_idx += N;
            negate = !negate;
        }
        while (src_idx >= (int32_t)N) {
            src_idx -= N;
            negate = !negate;
        }

        uint64_t val = src[src_idx];
        dst[i] = negate ? mod_sub(0, val, Q) : val;
    }
}

// ============================================================================
// Gadget Decomposition
// ============================================================================

// Decompose coefficient into L digits for RGSW multiplication
__device__ __forceinline__
void gadget_decompose(
    uint64_t coeff,
    uint64_t* digits,
    uint32_t L,
    uint64_t Bg,      // Gadget base
    uint64_t Bg_half, // Bg / 2 for rounding
    uint64_t Q
) {
    // Signed decomposition with rounding
    uint64_t val = coeff;

    for (uint32_t l = 0; l < L; ++l) {
        uint64_t digit = val % Bg;

        // Center the digit: if digit > Bg/2, subtract Bg and carry
        if (digit > Bg_half) {
            digit = digit - Bg;
            val = val + Bg;  // Borrow
        }

        // Store as positive value mod Q
        digits[l] = (digit >= 0) ? digit : mod_sub(0, (uint64_t)(-((int64_t)digit)), Q);
        val = val / Bg;
    }
}

// ============================================================================
// External Product: acc += (diff) * RGSW
// ============================================================================

__device__
void external_product_accumulate(
    uint64_t* acc_c0,           // [N] accumulator component 0
    uint64_t* acc_c1,           // [N] accumulator component 1
    const uint64_t* diff_c0,    // [N] (rotated - acc) component 0
    const uint64_t* diff_c1,    // [N] (rotated - acc) component 1
    const uint64_t* __restrict__ rgsw,  // [2, L, 2, N] RGSW ciphertext
    uint64_t* work,             // [N] workspace
    const uint64_t* __restrict__ twiddles,
    const uint64_t* __restrict__ precons,
    const uint64_t* __restrict__ inv_twiddles,
    const uint64_t* __restrict__ inv_precons,
    uint64_t N_inv,
    uint64_t N_inv_precon,
    uint32_t N,
    uint32_t log_N,
    uint32_t L,
    uint64_t Q,
    uint64_t Bg,
    uint64_t Bg_half,
    uint32_t tid,
    uint32_t tpg
) {
    // Temporary accumulators for external product result
    // We'll accumulate in shared memory
    extern __shared__ uint64_t ext_shared[];
    uint64_t* result_c0 = ext_shared;
    uint64_t* result_c1 = ext_shared + N;

    // Zero the result accumulators
    for (uint32_t i = tid; i < N; i += tpg) {
        result_c0[i] = 0;
        result_c1[i] = 0;
    }
    __syncthreads();

    // Process both components (c0, c1) of diff
    for (uint32_t comp = 0; comp < 2; ++comp) {
        const uint64_t* diff = (comp == 0) ? diff_c0 : diff_c1;

        // For each decomposition level
        for (uint32_t l = 0; l < L; ++l) {
            // Decompose and transform to NTT domain
            for (uint32_t i = tid; i < N; i += tpg) {
                uint64_t digits[MAX_L];
                gadget_decompose(diff[i], digits, L, Bg, Bg_half, Q);
                work[i] = digits[l];
            }
            __syncthreads();

            // Forward NTT of decomposed polynomial
            ntt_forward_shared(work, twiddles, precons, N, log_N, Q, tid, tpg);

            // Pointwise multiply with RGSW and accumulate
            // RGSW layout: rgsw[comp][l][out_comp][coeff]
            uint32_t rgsw_offset_0 = (comp * L * 2 + l * 2 + 0) * N;
            uint32_t rgsw_offset_1 = (comp * L * 2 + l * 2 + 1) * N;

            for (uint32_t i = tid; i < N; i += tpg) {
                uint64_t w = work[i];
                uint64_t r0 = __ldg(&rgsw[rgsw_offset_0 + i]);
                uint64_t r1 = __ldg(&rgsw[rgsw_offset_1 + i]);

                // Multiply and accumulate (still in NTT domain)
                result_c0[i] = mod_add(result_c0[i], barrett_mul(w, r0, Q, precons[i]), Q);
                result_c1[i] = mod_add(result_c1[i], barrett_mul(w, r1, Q, precons[i]), Q);
            }
            __syncthreads();
        }
    }

    // INTT the results
    ntt_inverse_shared(result_c0, inv_twiddles, inv_precons, N_inv, N_inv_precon, N, log_N, Q, tid, tpg);
    ntt_inverse_shared(result_c1, inv_twiddles, inv_precons, N_inv, N_inv_precon, N, log_N, Q, tid, tpg);

    // Add to accumulator
    for (uint32_t i = tid; i < N; i += tpg) {
        acc_c0[i] = mod_add(acc_c0[i], result_c0[i], Q);
        acc_c1[i] = mod_add(acc_c1[i], result_c1[i], Q);
    }
    __syncthreads();
}

// ============================================================================
// Blind Rotation Parameters
// ============================================================================

struct BlindRotateParams {
    uint64_t Q;              // Ring modulus
    uint64_t Bg;             // Gadget base
    uint64_t Bg_half;        // Bg / 2
    uint64_t N_inv;          // N^{-1} mod Q
    uint64_t N_inv_precon;   // Barrett precomputation for N_inv
    uint32_t N;              // Ring dimension
    uint32_t log_N;          // log2(N)
    uint32_t n;              // LWE dimension
    uint32_t L;              // Decomposition levels
    uint32_t batch_size;     // Number of bootstraps
};

// ============================================================================
// Main Blind Rotation Kernel
// ============================================================================

extern "C" __global__
void blind_rotate_fused(
    uint64_t* __restrict__ acc_out,          // [batch, 2, N] output RLWE
    const uint64_t* __restrict__ lwe_in,     // [batch, n+1] input LWE (a[0..n-1], b)
    const uint64_t* __restrict__ bsk,        // [n, 2, L, 2, N] bootstrap key
    const uint64_t* __restrict__ test_poly,  // [N] test polynomial
    const uint64_t* __restrict__ twiddles,
    const uint64_t* __restrict__ precons,
    const uint64_t* __restrict__ inv_twiddles,
    const uint64_t* __restrict__ inv_precons,
    const BlindRotateParams params
) {
    // Shared memory layout
    extern __shared__ uint64_t shared[];

    const uint32_t N = params.N;
    const uint32_t n = params.n;
    const uint32_t L = params.L;
    const uint64_t Q = params.Q;
    const uint64_t Bg = params.Bg;
    const uint64_t Bg_half = params.Bg_half;

    uint64_t* acc_c0 = shared;           // [N]
    uint64_t* acc_c1 = shared + N;       // [N]
    uint64_t* rot_c0 = shared + 2 * N;   // [N]
    uint64_t* rot_c1 = shared + 3 * N;   // [N]
    uint64_t* diff_c0 = shared + 4 * N;  // [N]
    uint64_t* diff_c1 = shared + 5 * N;  // [N]
    uint64_t* work = shared + 6 * N;     // [N]

    const uint32_t batch_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t tpg = blockDim.x;

    // LWE ciphertext for this batch
    const uint64_t* lwe = lwe_in + batch_idx * (n + 1);
    int32_t b = (int32_t)(lwe[n] % (2 * N));  // b is the last element

    // Initialize accumulator: acc = (0, X^{-b} * testPoly)
    for (uint32_t i = tid; i < N; i += tpg) {
        acc_c0[i] = 0;
    }
    __syncthreads();

    negacyclic_rotate(acc_c1, test_poly, -b, N, Q, tid, tpg);
    __syncthreads();

    // Main bootstrapping loop
    for (uint32_t j = 0; j < n; ++j) {
        int32_t a_j = (int32_t)(lwe[j] % (2 * N));

        // Skip if a[j] == 0 (no rotation needed)
        if (a_j == 0) continue;

        // Compute rotated = X^{a[j]} * acc
        negacyclic_rotate(rot_c0, acc_c0, a_j, N, Q, tid, tpg);
        __syncthreads();
        negacyclic_rotate(rot_c1, acc_c1, a_j, N, Q, tid, tpg);
        __syncthreads();

        // Compute diff = rotated - acc
        for (uint32_t i = tid; i < N; i += tpg) {
            diff_c0[i] = mod_sub(rot_c0[i], acc_c0[i], Q);
            diff_c1[i] = mod_sub(rot_c1[i], acc_c1[i], Q);
        }
        __syncthreads();

        // Bootstrap key for this index: bsk[j]
        const uint64_t* bsk_j = bsk + j * (2 * L * 2 * N);

        // acc += ExternalProduct(diff, bsk[j])
        external_product_accumulate(
            acc_c0, acc_c1,
            diff_c0, diff_c1,
            bsk_j, work,
            twiddles, precons,
            inv_twiddles, inv_precons,
            params.N_inv, params.N_inv_precon,
            N, params.log_N, L, Q, Bg, Bg_half,
            tid, tpg
        );
    }

    // Write output
    uint64_t* out = acc_out + batch_idx * 2 * N;
    for (uint32_t i = tid; i < N; i += tpg) {
        out[i] = acc_c0[i];
        out[N + i] = acc_c1[i];
    }
}

// ============================================================================
// Simplified Kernel for Testing
// ============================================================================

extern "C" __global__
void blind_rotate_single_step(
    uint64_t* __restrict__ acc,              // [2, N] accumulator
    const uint64_t* __restrict__ bsk_j,      // [2, L, 2, N] single BSK entry
    int32_t rotation,                        // Rotation amount
    const uint64_t* __restrict__ twiddles,
    const uint64_t* __restrict__ precons,
    const uint64_t* __restrict__ inv_twiddles,
    const uint64_t* __restrict__ inv_precons,
    const BlindRotateParams params
) {
    extern __shared__ uint64_t shared[];

    const uint32_t N = params.N;
    const uint32_t L = params.L;
    const uint64_t Q = params.Q;

    uint64_t* acc_c0 = shared;
    uint64_t* acc_c1 = shared + N;
    uint64_t* rot_c0 = shared + 2 * N;
    uint64_t* rot_c1 = shared + 3 * N;
    uint64_t* diff_c0 = shared + 4 * N;
    uint64_t* diff_c1 = shared + 5 * N;
    uint64_t* work = shared + 6 * N;

    const uint32_t tid = threadIdx.x;
    const uint32_t tpg = blockDim.x;

    // Load accumulator to shared
    for (uint32_t i = tid; i < N; i += tpg) {
        acc_c0[i] = acc[i];
        acc_c1[i] = acc[N + i];
    }
    __syncthreads();

    if (rotation != 0) {
        // Rotate
        negacyclic_rotate(rot_c0, acc_c0, rotation, N, Q, tid, tpg);
        __syncthreads();
        negacyclic_rotate(rot_c1, acc_c1, rotation, N, Q, tid, tpg);
        __syncthreads();

        // Diff
        for (uint32_t i = tid; i < N; i += tpg) {
            diff_c0[i] = mod_sub(rot_c0[i], acc_c0[i], Q);
            diff_c1[i] = mod_sub(rot_c1[i], acc_c1[i], Q);
        }
        __syncthreads();

        // External product
        external_product_accumulate(
            acc_c0, acc_c1,
            diff_c0, diff_c1,
            bsk_j, work,
            twiddles, precons,
            inv_twiddles, inv_precons,
            params.N_inv, params.N_inv_precon,
            N, params.log_N, L, Q,
            params.Bg, params.Bg_half,
            tid, tpg
        );
    }

    // Write back
    for (uint32_t i = tid; i < N; i += tpg) {
        acc[i] = acc_c0[i];
        acc[N + i] = acc_c1[i];
    }
}
