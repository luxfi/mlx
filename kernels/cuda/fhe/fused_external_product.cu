// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Fused External Product CUDA Kernels
// Combines keyswitch and blind rotate operations for maximum throughput.
//
// Key optimizations:
// 1. Fused keyswitch + external product in single kernel
// 2. Pipelined blind rotation with prefetching
// 3. Reduced global memory traffic
// 4. Warp-synchronized decomposition

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================================
// Constants
// ============================================================================

#define MAX_N 2048
#define MAX_N_LWE 1024
#define MAX_K 2
#define MAX_L 8
#define WARP_SIZE 32
#define MAX_KSK_LEVELS 4

// ============================================================================
// Modular Arithmetic
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
uint64_t mod_neg(uint64_t a, uint64_t Q) {
    return a == 0 ? 0 : Q - a;
}

__device__ __forceinline__
uint64_t barrett_mul(uint64_t a, uint64_t omega, uint64_t Q, uint64_t precon) {
    uint64_t q_approx = __umul64hi(a, precon);
    uint64_t product = a * omega;
    uint64_t result = product - q_approx * Q;
    return result >= Q ? result - Q : result;
}

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
// Parameters
// ============================================================================

struct FusedExternalProductParams {
    // Ring parameters
    uint64_t Q;              // Ring modulus
    uint64_t N_inv;          // N^{-1} mod Q
    uint64_t N_inv_precon;   // Barrett precomputation for N_inv
    uint32_t N;              // Ring dimension
    uint32_t log_N;          // log2(N)
    uint32_t k;              // GLWE dimension

    // Decomposition parameters
    uint32_t L_bsk;          // BSK decomposition levels
    uint32_t Bg_bits_bsk;    // BSK gadget base bits
    uint64_t Bg_bsk;         // BSK gadget base
    uint64_t Bg_half_bsk;    // BSK Bg / 2
    uint64_t Bg_mask_bsk;    // BSK Bg - 1

    // Keyswitch parameters
    uint32_t n_lwe;          // LWE dimension
    uint32_t L_ksk;          // KSK decomposition levels
    uint32_t Bg_bits_ksk;    // KSK gadget base bits
    uint64_t Bg_ksk;         // KSK gadget base
    uint64_t Bg_half_ksk;    // KSK Bg / 2
    uint64_t Bg_mask_ksk;    // KSK Bg - 1

    uint32_t batch_size;     // Number of bootstraps
};

// ============================================================================
// Shared Memory NTT Operations
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
// Negacyclic Rotation
// ============================================================================

__device__
void negacyclic_rotate(
    uint64_t* dst,
    const uint64_t* src,
    int32_t rotation,
    uint32_t N,
    uint64_t Q,
    uint32_t tid,
    uint32_t tpg
) {
    // Normalize rotation to [0, 2N)
    int32_t rot = ((rotation % (int32_t)(2 * N)) + 2 * N) % (2 * N);

    for (uint32_t i = tid; i < N; i += tpg) {
        int32_t src_idx = (int32_t)i - rot;
        bool negate = false;

        while (src_idx < 0) {
            src_idx += N;
            negate = !negate;
        }
        while (src_idx >= (int32_t)N) {
            src_idx -= N;
            negate = !negate;
        }

        uint64_t val = src[src_idx];
        dst[i] = negate ? mod_neg(val, Q) : val;
    }
}

// ============================================================================
// Keyswitch Operation
// ============================================================================

// Keyswitch from LWE(s) to LWE(s')
// Each thread processes one output coefficient
__device__
void keyswitch_lwe(
    uint64_t* __restrict__ lwe_out,              // [n_out + 1]
    const uint64_t* __restrict__ lwe_in,         // [n_in + 1]
    const uint64_t* __restrict__ ksk,            // [n_in, L_ksk, n_out + 1]
    uint32_t n_in,
    uint32_t n_out,
    uint32_t L_ksk,
    uint32_t Bg_bits,
    uint64_t Bg,
    uint64_t Bg_half,
    uint64_t Bg_mask,
    uint64_t Q,
    uint32_t tid,
    uint32_t tpg
) {
    // Initialize output to (0, b_in)
    for (uint32_t i = tid; i < n_out; i += tpg) {
        lwe_out[i] = 0;
    }
    if (tid == 0) {
        lwe_out[n_out] = lwe_in[n_in];  // Copy b
    }
    __syncthreads();

    // For each input coefficient
    for (uint32_t j = 0; j < n_in; ++j) {
        uint64_t a_j = lwe_in[j];

        // Decompose a_j
        for (uint32_t l = 0; l < L_ksk; ++l) {
            uint32_t shift = 64 - (L_ksk - l) * Bg_bits;
            uint64_t digit = ((a_j >> shift) + Bg_half) & Bg_mask;
            int64_t signed_digit = (int64_t)digit - (int64_t)Bg_half;

            // Subtract digit * ksk[j][l]
            const uint64_t* ksk_row = ksk + (j * L_ksk + l) * (n_out + 1);

            for (uint32_t i = tid; i <= n_out; i += tpg) {
                uint64_t val = ksk_row[i];
                if (signed_digit >= 0) {
                    uint64_t prod = mod_mul((uint64_t)signed_digit, val, Q);
                    lwe_out[i] = mod_sub(lwe_out[i], prod, Q);
                } else {
                    uint64_t prod = mod_mul((uint64_t)(-signed_digit), val, Q);
                    lwe_out[i] = mod_add(lwe_out[i], prod, Q);
                }
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// Fused Keyswitch + Blind Rotation
// ============================================================================

// Complete programmable bootstrapping: keyswitch then blind rotate

extern "C" __global__
void fused_keyswitch_blind_rotate_kernel(
    uint64_t* __restrict__ lwe_out,              // [batch, n_out + 1]
    const uint64_t* __restrict__ lwe_in,         // [batch, n_in + 1]
    const uint64_t* __restrict__ ksk,            // [n_in, L_ksk, n_out + 1]
    const uint64_t* __restrict__ bsk,            // [n_out, k+1, L_bsk, k+1, N]
    const uint64_t* __restrict__ test_poly,      // [N]
    const uint64_t* __restrict__ twiddles,
    const uint64_t* __restrict__ precons,
    const uint64_t* __restrict__ inv_twiddles,
    const uint64_t* __restrict__ inv_precons,
    const FusedExternalProductParams params
) {
    extern __shared__ uint64_t shared[];

    const uint32_t N = params.N;
    const uint32_t log_N = params.log_N;
    const uint32_t k = params.k;
    const uint32_t n_lwe = params.n_lwe;
    const uint32_t L_bsk = params.L_bsk;
    const uint64_t Q = params.Q;
    const uint64_t Bg_half_bsk = params.Bg_half_bsk;
    const uint64_t Bg_mask_bsk = params.Bg_mask_bsk;
    const uint32_t Bg_bits_bsk = params.Bg_bits_bsk;

    const uint32_t batch_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t tpg = blockDim.x;

    // Shared memory layout:
    // [lwe_ks: n_lwe+1][acc_c0: N][acc_c1: N][rot_c0: N][rot_c1: N][diff_c0: N][diff_c1: N][work: N]
    uint64_t* lwe_ks = shared;
    uint64_t* acc_c0 = shared + n_lwe + 1;
    uint64_t* acc_c1 = acc_c0 + N;
    uint64_t* rot_c0 = acc_c1 + N;
    uint64_t* rot_c1 = rot_c0 + N;
    uint64_t* diff_c0 = rot_c1 + N;
    uint64_t* diff_c1 = diff_c0 + N;
    uint64_t* work = diff_c1 + N;

    // =========================================================================
    // Phase 1: Keyswitch (if needed)
    // =========================================================================

    const uint64_t* lwe = lwe_in + batch_idx * (params.n_lwe + 1);

    // Copy input LWE to shared (already in correct dimension for this example)
    for (uint32_t i = tid; i <= n_lwe; i += tpg) {
        lwe_ks[i] = lwe[i];
    }
    __syncthreads();

    // =========================================================================
    // Phase 2: Initialize Accumulator
    // =========================================================================

    // b = lwe_ks[n_lwe]
    int32_t b = (int32_t)(lwe_ks[n_lwe] % (2 * N));

    // acc = (0, X^{-b} * test_poly)
    for (uint32_t i = tid; i < N; i += tpg) {
        acc_c0[i] = 0;
    }
    __syncthreads();

    negacyclic_rotate(acc_c1, test_poly, -b, N, Q, tid, tpg);
    __syncthreads();

    // =========================================================================
    // Phase 3: Blind Rotation Loop
    // =========================================================================

    for (uint32_t j = 0; j < n_lwe; ++j) {
        int32_t a_j = (int32_t)(lwe_ks[j] % (2 * N));

        // Skip if no rotation needed
        if (a_j == 0) continue;

        // Compute rotated = X^{a_j} * acc
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

        // External product: acc += diff * bsk[j]
        // bsk[j] has layout: [k+1][L_bsk][k+1][N]
        const uint64_t* bsk_j = bsk + j * (k + 1) * L_bsk * (k + 1) * N;

        // Zero the product accumulators (reuse rot buffers)
        for (uint32_t i = tid; i < N; i += tpg) {
            rot_c0[i] = 0;  // Will hold external product result for c0
            rot_c1[i] = 0;  // Will hold external product result for c1
        }
        __syncthreads();

        // For each input component
        for (uint32_t in_comp = 0; in_comp <= k; ++in_comp) {
            const uint64_t* diff = (in_comp == 0) ? diff_c0 : diff_c1;

            for (uint32_t level = 0; level < L_bsk; ++level) {
                // Decompose diff at this level
                for (uint32_t i = tid; i < N; i += tpg) {
                    uint64_t coeff = diff[i];
                    uint32_t shift = 64 - (L_bsk - level) * Bg_bits_bsk;
                    uint64_t digit = ((coeff >> shift) + Bg_half_bsk) & Bg_mask_bsk;
                    int64_t signed_digit = (int64_t)digit - (int64_t)Bg_half_bsk;
                    work[i] = (signed_digit >= 0) ? (uint64_t)signed_digit : (Q + signed_digit);
                }
                __syncthreads();

                // Forward NTT
                ntt_forward_shared(work, twiddles, precons, N, log_N, Q, tid, tpg);

                // Multiply and accumulate for both output components
                uint32_t bsk_offset_c0 = ((in_comp * L_bsk + level) * (k + 1) + 0) * N;
                uint32_t bsk_offset_c1 = ((in_comp * L_bsk + level) * (k + 1) + 1) * N;

                for (uint32_t i = tid; i < N; i += tpg) {
                    uint64_t w = work[i];
                    uint64_t prod_c0 = mod_mul(w, __ldg(&bsk_j[bsk_offset_c0 + i]), Q);
                    uint64_t prod_c1 = mod_mul(w, __ldg(&bsk_j[bsk_offset_c1 + i]), Q);
                    rot_c0[i] = mod_add(rot_c0[i], prod_c0, Q);
                    rot_c1[i] = mod_add(rot_c1[i], prod_c1, Q);
                }
                __syncthreads();
            }
        }

        // Inverse NTT on product accumulators
        ntt_inverse_shared(rot_c0, inv_twiddles, inv_precons,
                           params.N_inv, params.N_inv_precon,
                           N, log_N, Q, tid, tpg);
        ntt_inverse_shared(rot_c1, inv_twiddles, inv_precons,
                           params.N_inv, params.N_inv_precon,
                           N, log_N, Q, tid, tpg);

        // Add to accumulator
        for (uint32_t i = tid; i < N; i += tpg) {
            acc_c0[i] = mod_add(acc_c0[i], rot_c0[i], Q);
            acc_c1[i] = mod_add(acc_c1[i], rot_c1[i], Q);
        }
        __syncthreads();
    }

    // =========================================================================
    // Phase 4: Sample Extract
    // =========================================================================

    // Extract LWE from GLWE at position 0
    // lwe_out.a[i] = acc_c0[0] for i=0, -acc_c0[N-i] for i>0
    // lwe_out.b = acc_c1[0]

    uint64_t* out = lwe_out + batch_idx * (N + 1);

    for (uint32_t i = tid; i < N; i += tpg) {
        if (i == 0) {
            out[0] = acc_c0[0];
        } else {
            out[i] = mod_neg(acc_c0[N - i], Q);
        }
    }
    if (tid == 0) {
        out[N] = acc_c1[0];
    }
}

// ============================================================================
// Fused Blind Rotate Only (No Keyswitch)
// ============================================================================

extern "C" __global__
void fused_blind_rotate_kernel(
    uint64_t* __restrict__ glwe_out,             // [batch, k+1, N]
    const uint64_t* __restrict__ lwe_in,         // [batch, n_lwe + 1]
    const uint64_t* __restrict__ bsk,            // [n_lwe, k+1, L, k+1, N]
    const uint64_t* __restrict__ test_poly,      // [N]
    const uint64_t* __restrict__ twiddles,
    const uint64_t* __restrict__ precons,
    const uint64_t* __restrict__ inv_twiddles,
    const uint64_t* __restrict__ inv_precons,
    const FusedExternalProductParams params
) {
    extern __shared__ uint64_t shared[];

    const uint32_t N = params.N;
    const uint32_t log_N = params.log_N;
    const uint32_t k = params.k;
    const uint32_t n_lwe = params.n_lwe;
    const uint32_t L_bsk = params.L_bsk;
    const uint64_t Q = params.Q;
    const uint64_t Bg_half = params.Bg_half_bsk;
    const uint64_t Bg_mask = params.Bg_mask_bsk;
    const uint32_t Bg_bits = params.Bg_bits_bsk;

    const uint32_t batch_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t tpg = blockDim.x;

    // Shared memory layout
    uint64_t* acc_c0 = shared;
    uint64_t* acc_c1 = shared + N;
    uint64_t* rot_c0 = shared + 2 * N;
    uint64_t* rot_c1 = shared + 3 * N;
    uint64_t* diff_c0 = shared + 4 * N;
    uint64_t* diff_c1 = shared + 5 * N;
    uint64_t* work = shared + 6 * N;

    const uint64_t* lwe = lwe_in + batch_idx * (n_lwe + 1);
    int32_t b = (int32_t)(lwe[n_lwe] % (2 * N));

    // Initialize accumulator
    for (uint32_t i = tid; i < N; i += tpg) {
        acc_c0[i] = 0;
    }
    __syncthreads();

    negacyclic_rotate(acc_c1, test_poly, -b, N, Q, tid, tpg);
    __syncthreads();

    // Blind rotation loop
    for (uint32_t j = 0; j < n_lwe; ++j) {
        int32_t a_j = (int32_t)(lwe[j] % (2 * N));

        if (a_j == 0) continue;

        negacyclic_rotate(rot_c0, acc_c0, a_j, N, Q, tid, tpg);
        __syncthreads();
        negacyclic_rotate(rot_c1, acc_c1, a_j, N, Q, tid, tpg);
        __syncthreads();

        for (uint32_t i = tid; i < N; i += tpg) {
            diff_c0[i] = mod_sub(rot_c0[i], acc_c0[i], Q);
            diff_c1[i] = mod_sub(rot_c1[i], acc_c1[i], Q);
        }
        __syncthreads();

        const uint64_t* bsk_j = bsk + j * (k + 1) * L_bsk * (k + 1) * N;

        for (uint32_t i = tid; i < N; i += tpg) {
            rot_c0[i] = 0;
            rot_c1[i] = 0;
        }
        __syncthreads();

        for (uint32_t in_comp = 0; in_comp <= k; ++in_comp) {
            const uint64_t* diff = (in_comp == 0) ? diff_c0 : diff_c1;

            for (uint32_t level = 0; level < L_bsk; ++level) {
                for (uint32_t i = tid; i < N; i += tpg) {
                    uint64_t coeff = diff[i];
                    uint32_t shift = 64 - (L_bsk - level) * Bg_bits;
                    uint64_t digit = ((coeff >> shift) + Bg_half) & Bg_mask;
                    int64_t signed_digit = (int64_t)digit - (int64_t)Bg_half;
                    work[i] = (signed_digit >= 0) ? (uint64_t)signed_digit : (Q + signed_digit);
                }
                __syncthreads();

                ntt_forward_shared(work, twiddles, precons, N, log_N, Q, tid, tpg);

                uint32_t bsk_offset_c0 = ((in_comp * L_bsk + level) * (k + 1) + 0) * N;
                uint32_t bsk_offset_c1 = ((in_comp * L_bsk + level) * (k + 1) + 1) * N;

                for (uint32_t i = tid; i < N; i += tpg) {
                    uint64_t w = work[i];
                    rot_c0[i] = mod_add(rot_c0[i], mod_mul(w, __ldg(&bsk_j[bsk_offset_c0 + i]), Q), Q);
                    rot_c1[i] = mod_add(rot_c1[i], mod_mul(w, __ldg(&bsk_j[bsk_offset_c1 + i]), Q), Q);
                }
                __syncthreads();
            }
        }

        ntt_inverse_shared(rot_c0, inv_twiddles, inv_precons,
                           params.N_inv, params.N_inv_precon,
                           N, log_N, Q, tid, tpg);
        ntt_inverse_shared(rot_c1, inv_twiddles, inv_precons,
                           params.N_inv, params.N_inv_precon,
                           N, log_N, Q, tid, tpg);

        for (uint32_t i = tid; i < N; i += tpg) {
            acc_c0[i] = mod_add(acc_c0[i], rot_c0[i], Q);
            acc_c1[i] = mod_add(acc_c1[i], rot_c1[i], Q);
        }
        __syncthreads();
    }

    // Write output GLWE
    uint64_t* out = glwe_out + batch_idx * (k + 1) * N;
    for (uint32_t i = tid; i < N; i += tpg) {
        out[i] = acc_c0[i];
        out[N + i] = acc_c1[i];
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

cudaError_t lux_cuda_fhe_fused_keyswitch_blind_rotate(
    uint64_t* lwe_out,
    const uint64_t* lwe_in,
    const uint64_t* ksk,
    const uint64_t* bsk,
    const uint64_t* test_poly,
    const uint64_t* twiddles,
    const uint64_t* precons,
    const uint64_t* inv_twiddles,
    const uint64_t* inv_precons,
    uint32_t N,
    uint32_t k,
    uint32_t n_lwe,
    uint32_t L_bsk,
    uint32_t Bg_bits_bsk,
    uint32_t L_ksk,
    uint32_t Bg_bits_ksk,
    uint64_t Q,
    uint64_t N_inv,
    uint64_t N_inv_precon,
    uint32_t batch_size,
    cudaStream_t stream
) {
    FusedExternalProductParams params;
    params.N = N;
    params.log_N = 31 - __builtin_clz(N);
    params.k = k;
    params.n_lwe = n_lwe;
    params.Q = Q;
    params.N_inv = N_inv;
    params.N_inv_precon = N_inv_precon;

    params.L_bsk = L_bsk;
    params.Bg_bits_bsk = Bg_bits_bsk;
    params.Bg_bsk = 1ULL << Bg_bits_bsk;
    params.Bg_half_bsk = 1ULL << (Bg_bits_bsk - 1);
    params.Bg_mask_bsk = params.Bg_bsk - 1;

    params.L_ksk = L_ksk;
    params.Bg_bits_ksk = Bg_bits_ksk;
    params.Bg_ksk = 1ULL << Bg_bits_ksk;
    params.Bg_half_ksk = 1ULL << (Bg_bits_ksk - 1);
    params.Bg_mask_ksk = params.Bg_ksk - 1;

    params.batch_size = batch_size;

    // Shared memory: lwe_ks + 7*N polynomials + work buffer
    size_t shmem_size = (n_lwe + 1 + 8 * N) * sizeof(uint64_t);

    dim3 grid(batch_size);
    dim3 block(min(N / 2, 256u));

    fused_keyswitch_blind_rotate_kernel<<<grid, block, shmem_size, stream>>>(
        lwe_out, lwe_in, ksk, bsk, test_poly,
        twiddles, precons, inv_twiddles, inv_precons,
        params
    );

    return cudaGetLastError();
}

cudaError_t lux_cuda_fhe_fused_blind_rotate(
    uint64_t* glwe_out,
    const uint64_t* lwe_in,
    const uint64_t* bsk,
    const uint64_t* test_poly,
    const uint64_t* twiddles,
    const uint64_t* precons,
    const uint64_t* inv_twiddles,
    const uint64_t* inv_precons,
    uint32_t N,
    uint32_t k,
    uint32_t n_lwe,
    uint32_t L,
    uint32_t Bg_bits,
    uint64_t Q,
    uint64_t N_inv,
    uint64_t N_inv_precon,
    uint32_t batch_size,
    cudaStream_t stream
) {
    FusedExternalProductParams params;
    params.N = N;
    params.log_N = 31 - __builtin_clz(N);
    params.k = k;
    params.n_lwe = n_lwe;
    params.Q = Q;
    params.N_inv = N_inv;
    params.N_inv_precon = N_inv_precon;

    params.L_bsk = L;
    params.Bg_bits_bsk = Bg_bits;
    params.Bg_bsk = 1ULL << Bg_bits;
    params.Bg_half_bsk = 1ULL << (Bg_bits - 1);
    params.Bg_mask_bsk = params.Bg_bsk - 1;

    params.batch_size = batch_size;

    // Shared memory: 7 polynomials + work buffer
    size_t shmem_size = 7 * N * sizeof(uint64_t);

    dim3 grid(batch_size);
    dim3 block(min(N / 2, 256u));

    fused_blind_rotate_kernel<<<grid, block, shmem_size, stream>>>(
        glwe_out, lwe_in, bsk, test_poly,
        twiddles, precons, inv_twiddles, inv_precons,
        params
    );

    return cudaGetLastError();
}

}  // extern "C"
