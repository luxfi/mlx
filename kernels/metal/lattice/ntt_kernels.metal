// =============================================================================
// Lux Lattice - Metal NTT Kernels
// =============================================================================
//
// GPU-accelerated Number Theoretic Transform for polynomial ring operations.
// Derived from Lux FHE's OpenFHE-compatible implementation.
//
// Features:
// - Barrett modular multiplication with precomputed constants
// - Cooley-Tukey forward NTT (decimation-in-time)
// - Gentleman-Sande inverse NTT (decimation-in-frequency)
// - Fused single-kernel NTT for N <= 4096 (fits in shared memory)
// - Batch processing for multiple polynomials
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// NTT Parameters Structure (matches host struct)
// =============================================================================

struct NTTParams {
    uint64_t Q;            // Prime modulus
    uint64_t mu;           // Barrett constant: floor(2^64 / Q)
    uint64_t N_inv;        // N^{-1} mod Q
    uint64_t N_inv_precon; // Barrett precomputation for N_inv
    uint32_t N;            // Ring dimension (power of 2)
    uint32_t log_N;        // log2(N)
    uint32_t stage;        // Current stage (for staged dispatch)
    uint32_t batch;        // Batch size
};

// =============================================================================
// Barrett Modular Arithmetic
// =============================================================================

// Barrett multiplication: (a * b) mod Q
// Requires precon = floor(2^64 * b / Q)
inline uint64_t barrett_mul(uint64_t a, uint64_t b, uint64_t Q, uint64_t precon) {
    uint64_t q_approx = metal::mulhi(a, precon);
    uint64_t result = a * b - q_approx * Q;
    return result >= Q ? result - Q : result;
}

// Modular addition: (a + b) mod Q
inline uint64_t mod_add(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t sum = a + b;
    return sum >= Q ? sum - Q : sum;
}

// Modular subtraction: (a - b) mod Q
inline uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t Q) {
    return a >= b ? a - b : a + Q - b;
}

// =============================================================================
// Cooley-Tukey Butterfly (Forward NTT)
// =============================================================================

inline void ct_butterfly(device uint64_t* data,
                         uint32_t idx_lo, uint32_t idx_hi,
                         uint64_t omega, uint64_t precon,
                         uint64_t Q) {
    uint64_t lo_val = data[idx_lo];
    uint64_t hi_val = data[idx_hi];
    uint64_t omega_hi = barrett_mul(hi_val, omega, Q, precon);
    data[idx_lo] = mod_add(lo_val, omega_hi, Q);
    data[idx_hi] = mod_sub(lo_val, omega_hi, Q);
}

// =============================================================================
// Gentleman-Sande Butterfly (Inverse NTT)
// =============================================================================

inline void gs_butterfly(device uint64_t* data,
                         uint32_t idx_lo, uint32_t idx_hi,
                         uint64_t omega, uint64_t precon,
                         uint64_t Q) {
    uint64_t lo_val = data[idx_lo];
    uint64_t hi_val = data[idx_hi];
    uint64_t sum = mod_add(lo_val, hi_val, Q);
    uint64_t diff = mod_sub(lo_val, hi_val, Q);
    data[idx_lo] = sum;
    data[idx_hi] = barrett_mul(diff, omega, Q, precon);
}

// =============================================================================
// Staged NTT Kernels (for large N > 4096)
// =============================================================================

kernel void ntt_forward_stage(
    device uint64_t* data [[buffer(0)]],
    constant uint64_t* twiddles [[buffer(1)]],
    constant uint64_t* precon [[buffer(2)]],
    constant NTTParams& params [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = tid.y;
    uint32_t butterfly_idx = tid.x;

    if (batch_idx >= params.batch) return;

    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint32_t stage = params.stage;

    uint32_t m = 1u << stage;
    uint32_t t = N >> (stage + 1);
    uint32_t num_butterflies = N >> 1;

    if (butterfly_idx >= num_butterflies) return;

    uint32_t i = butterfly_idx / t;
    uint32_t j = butterfly_idx % t;
    uint32_t idx_lo = (i << (params.log_N - stage)) + j;
    uint32_t idx_hi = idx_lo + t;

    uint32_t tw_idx = m + i;
    device uint64_t* poly = data + batch_idx * N;

    ct_butterfly(poly, idx_lo, idx_hi, twiddles[tw_idx], precon[tw_idx], Q);
}

kernel void ntt_inverse_stage(
    device uint64_t* data [[buffer(0)]],
    constant uint64_t* inv_twiddles [[buffer(1)]],
    constant uint64_t* inv_precon [[buffer(2)]],
    constant NTTParams& params [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = tid.y;
    uint32_t butterfly_idx = tid.x;

    if (batch_idx >= params.batch) return;

    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint32_t stage = params.stage;

    uint32_t m = N >> (stage + 1);
    uint32_t t = 1u << stage;
    uint32_t num_butterflies = N >> 1;

    if (butterfly_idx >= num_butterflies) return;

    uint32_t i = butterfly_idx / t;
    uint32_t j = butterfly_idx % t;
    uint32_t idx_lo = (i << (stage + 1)) + j;
    uint32_t idx_hi = idx_lo + t;

    uint32_t tw_idx = m + i;
    device uint64_t* poly = data + batch_idx * N;

    gs_butterfly(poly, idx_lo, idx_hi, inv_twiddles[tw_idx], inv_precon[tw_idx], Q);
}

// Scale by N^{-1} after inverse NTT
kernel void ntt_scale_inverse(
    device uint64_t* data [[buffer(0)]],
    constant NTTParams& params [[buffer(1)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = tid.y;
    uint32_t coeff_idx = tid.x;

    if (batch_idx >= params.batch || coeff_idx >= params.N) return;

    uint32_t idx = batch_idx * params.N + coeff_idx;
    data[idx] = barrett_mul(data[idx], params.N_inv, params.Q, params.N_inv_precon);
}

// =============================================================================
// Fused NTT Kernels (for N <= 4096, all stages in shared memory)
// =============================================================================

kernel void ntt_forward_fused(
    device uint64_t* data [[buffer(0)]],
    constant uint64_t* twiddles [[buffer(1)]],
    constant uint64_t* precon [[buffer(2)]],
    constant NTTParams& params [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint tpg [[threads_per_threadgroup]],
    threadgroup uint64_t* shared [[threadgroup(0)]]
) {
    uint32_t batch_idx = gid;
    if (batch_idx >= params.batch) return;

    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint32_t log_N = params.log_N;

    device uint64_t* poly = data + batch_idx * N;

    // Load data to shared memory
    for (uint32_t i = tid; i < N; i += tpg) {
        shared[i] = poly[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Perform all NTT stages in shared memory
    for (uint32_t s = 0; s < log_N; ++s) {
        uint32_t m = 1u << s;
        uint32_t t = N >> (s + 1);

        for (uint32_t butterfly = tid; butterfly < N/2; butterfly += tpg) {
            uint32_t i = butterfly / t;
            uint32_t j = butterfly % t;
            uint32_t idx_lo = (i << (log_N - s)) + j;
            uint32_t idx_hi = idx_lo + t;

            uint32_t tw_idx = m + i;
            uint64_t omega = twiddles[tw_idx];
            uint64_t pc = precon[tw_idx];

            uint64_t lo = shared[idx_lo];
            uint64_t hi = shared[idx_hi];
            uint64_t omega_hi = barrett_mul(hi, omega, Q, pc);

            shared[idx_lo] = mod_add(lo, omega_hi, Q);
            shared[idx_hi] = mod_sub(lo, omega_hi, Q);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back to global memory
    for (uint32_t i = tid; i < N; i += tpg) {
        poly[i] = shared[i];
    }
}

kernel void ntt_inverse_fused(
    device uint64_t* data [[buffer(0)]],
    constant uint64_t* inv_twiddles [[buffer(1)]],
    constant uint64_t* inv_precon [[buffer(2)]],
    constant NTTParams& params [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint tpg [[threads_per_threadgroup]],
    threadgroup uint64_t* shared [[threadgroup(0)]]
) {
    uint32_t batch_idx = gid;
    if (batch_idx >= params.batch) return;

    uint32_t N = params.N;
    uint64_t Q = params.Q;
    uint32_t log_N = params.log_N;

    device uint64_t* poly = data + batch_idx * N;

    // Load data to shared memory
    for (uint32_t i = tid; i < N; i += tpg) {
        shared[i] = poly[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Perform all inverse NTT stages in shared memory
    for (uint32_t s = 0; s < log_N; ++s) {
        uint32_t m = N >> (s + 1);
        uint32_t t = 1u << s;

        for (uint32_t butterfly = tid; butterfly < N/2; butterfly += tpg) {
            uint32_t i = butterfly / t;
            uint32_t j = butterfly % t;
            uint32_t idx_lo = (i << (s + 1)) + j;
            uint32_t idx_hi = idx_lo + t;

            uint32_t tw_idx = m + i;
            uint64_t omega = inv_twiddles[tw_idx];
            uint64_t pc = inv_precon[tw_idx];

            uint64_t lo = shared[idx_lo];
            uint64_t hi = shared[idx_hi];
            uint64_t sum = mod_add(lo, hi, Q);
            uint64_t diff = mod_sub(lo, hi, Q);

            shared[idx_lo] = sum;
            shared[idx_hi] = barrett_mul(diff, omega, Q, pc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Scale by N^{-1} and write back
    for (uint32_t i = tid; i < N; i += tpg) {
        poly[i] = barrett_mul(shared[i], params.N_inv, Q, params.N_inv_precon);
    }
}

// =============================================================================
// Pointwise Multiplication Kernel
// =============================================================================

kernel void ntt_pointwise_mul(
    device uint64_t* result [[buffer(0)]],
    constant uint64_t* a [[buffer(1)]],
    constant uint64_t* b [[buffer(2)]],
    constant NTTParams& params [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = tid.y;
    uint32_t coeff_idx = tid.x;

    if (batch_idx >= params.batch || coeff_idx >= params.N) return;

    uint32_t idx = batch_idx * params.N + coeff_idx;
    uint64_t Q = params.Q;

    // Simple modular multiplication (no precomputation needed for single mul)
    uint64_t av = a[idx];
    uint64_t bv = b[idx];

    // Use mulhi for 128-bit multiply
    uint64_t lo = av * bv;
    uint64_t hi = metal::mulhi(av, bv);

    if (hi == 0) {
        result[idx] = lo % Q;
    } else {
        // Full reduction for large products
        uint64_t two32_mod_q = (uint64_t(1) << 32) % Q;
        uint64_t two64_mod_q = (two32_mod_q * two32_mod_q) % Q;
        result[idx] = (lo % Q + (hi % Q) * two64_mod_q % Q) % Q;
    }
}
