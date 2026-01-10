// =============================================================================
// ML-DSA Batch Verification - Metal Compute Shaders
// =============================================================================
//
// GPU-accelerated ML-DSA (FIPS 204) signature verification on Apple Silicon.
// Uses NTT from luxcpp/lattice for polynomial operations.
//
// ML-DSA Parameters (65 = NIST Level 3):
//   n = 256 (polynomial degree)
//   q = 8380417 (prime modulus)
//   k = 6 (rows in A matrix)
//   l = 5 (columns in A matrix)
//
// Key sizes:
//   Public key: 1952 bytes
//   Signature: 3309 bytes
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// ML-DSA-65 Parameters
// =============================================================================

constant uint32_t MLDSA_N = 256;        // Polynomial degree
constant uint64_t MLDSA_Q = 8380417;    // Prime modulus
constant uint32_t MLDSA_K = 6;          // A matrix rows
constant uint32_t MLDSA_L = 5;          // A matrix columns
constant uint32_t MLDSA_LOG_N = 8;      // log2(N)

// Montgomery constants for q = 8380417
constant uint64_t MLDSA_R2 = 2365951;   // R^2 mod q
constant uint64_t MLDSA_QINV = 58728449; // -q^{-1} mod 2^32

// NTT primitive root
constant uint32_t MLDSA_ROOT = 1753;    // 2N-th root of unity mod q

// =============================================================================
// NTT Parameters Structure
// =============================================================================

struct MLDSAParams {
    uint32_t batch_size;    // Number of signatures in batch
    uint32_t stage;         // Current NTT stage (for staged execution)
    uint32_t mode;          // 0=forward NTT, 1=inverse NTT
    uint32_t pad;           // Alignment padding
};

// =============================================================================
// Barrett Reduction for ML-DSA
// =============================================================================

// Barrett reduction: compute a mod q where a < q^2
inline uint32_t barrett_reduce(uint64_t a) {
    // For q = 8380417, we use the approximation q ≈ 2^23
    // mu = floor(2^46 / q) = 8396807
    const uint64_t mu = 8396807;
    
    uint64_t q_approx = (a * mu) >> 46;
    uint32_t result = uint32_t(a - q_approx * MLDSA_Q);
    
    // Final reduction
    if (result >= MLDSA_Q) {
        result -= MLDSA_Q;
    }
    return result;
}

// Montgomery multiplication for ML-DSA
inline uint32_t mont_mul(uint32_t a, uint32_t b) {
    uint64_t prod = uint64_t(a) * uint64_t(b);
    uint32_t m = uint32_t(prod) * uint32_t(MLDSA_QINV);
    uint64_t t = prod + uint64_t(m) * MLDSA_Q;
    uint32_t result = uint32_t(t >> 32);
    return result >= MLDSA_Q ? result - MLDSA_Q : result;
}

// Modular addition
inline uint32_t mod_add(uint32_t a, uint32_t b) {
    uint32_t sum = a + b;
    return sum >= MLDSA_Q ? sum - MLDSA_Q : sum;
}

// Modular subtraction
inline uint32_t mod_sub(uint32_t a, uint32_t b) {
    return a >= b ? a - b : a + MLDSA_Q - b;
}

// =============================================================================
// Precomputed Twiddle Factors (loaded from buffer in actual impl)
// =============================================================================

// Generate twiddle factors on the fly (for simplicity)
// In production, these would be precomputed
inline uint32_t get_twiddle(uint32_t idx, constant uint32_t* twiddles) {
    return twiddles[idx];
}

// =============================================================================
// NTT Butterfly Operations
// =============================================================================

// Cooley-Tukey butterfly for forward NTT
inline void ct_butterfly(thread uint32_t& lo, thread uint32_t& hi,
                         uint32_t omega) {
    uint32_t omega_hi = mont_mul(hi, omega);
    uint32_t new_lo = mod_add(lo, omega_hi);
    uint32_t new_hi = mod_sub(lo, omega_hi);
    lo = new_lo;
    hi = new_hi;
}

// Gentleman-Sande butterfly for inverse NTT
inline void gs_butterfly(thread uint32_t& lo, thread uint32_t& hi,
                         uint32_t omega) {
    uint32_t sum = mod_add(lo, hi);
    uint32_t diff = mod_sub(lo, hi);
    lo = sum;
    hi = mont_mul(diff, omega);
}

// =============================================================================
// Batch NTT Kernel - Forward Transform
// =============================================================================

kernel void mldsa_ntt_forward(
    device uint32_t* polys [[buffer(0)]],           // Batch of polynomials (modified in-place)
    constant uint32_t* twiddles [[buffer(1)]],      // Precomputed twiddle factors
    constant MLDSAParams& params [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = tid.y;
    uint32_t butterfly_idx = tid.x;
    
    if (batch_idx >= params.batch_size) return;
    
    uint32_t stage = params.stage;
    uint32_t m = 1u << stage;
    uint32_t t = MLDSA_N >> (stage + 1);
    uint32_t num_butterflies = MLDSA_N >> 1;
    
    if (butterfly_idx >= num_butterflies) return;
    
    uint32_t i = butterfly_idx / t;
    uint32_t j = butterfly_idx % t;
    uint32_t idx_lo = (i << (MLDSA_LOG_N - stage)) + j;
    uint32_t idx_hi = idx_lo + t;
    
    device uint32_t* poly = polys + batch_idx * MLDSA_N;
    uint32_t omega = twiddles[m + i];
    
    uint32_t lo = poly[idx_lo];
    uint32_t hi = poly[idx_hi];
    ct_butterfly(lo, hi, omega);
    poly[idx_lo] = lo;
    poly[idx_hi] = hi;
}

// =============================================================================
// Batch NTT Kernel - Inverse Transform
// =============================================================================

kernel void mldsa_ntt_inverse(
    device uint32_t* polys [[buffer(0)]],
    constant uint32_t* inv_twiddles [[buffer(1)]],
    constant MLDSAParams& params [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = tid.y;
    uint32_t butterfly_idx = tid.x;
    
    if (batch_idx >= params.batch_size) return;
    
    uint32_t stage = params.stage;
    uint32_t m = MLDSA_N >> (stage + 1);
    uint32_t t = 1u << stage;
    uint32_t num_butterflies = MLDSA_N >> 1;
    
    if (butterfly_idx >= num_butterflies) return;
    
    uint32_t i = butterfly_idx / t;
    uint32_t j = butterfly_idx % t;
    uint32_t idx_lo = (i << (stage + 1)) + j;
    uint32_t idx_hi = idx_lo + t;
    
    device uint32_t* poly = polys + batch_idx * MLDSA_N;
    uint32_t omega = inv_twiddles[m + i];
    
    uint32_t lo = poly[idx_lo];
    uint32_t hi = poly[idx_hi];
    gs_butterfly(lo, hi, omega);
    poly[idx_lo] = lo;
    poly[idx_hi] = hi;
}

// =============================================================================
// Fused NTT for Small Batches (fits in threadgroup memory)
// =============================================================================

kernel void mldsa_ntt_forward_fused(
    device uint32_t* polys [[buffer(0)]],
    constant uint32_t* twiddles [[buffer(1)]],
    constant MLDSAParams& params [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint tpg [[threads_per_threadgroup]],
    threadgroup uint32_t* shared [[threadgroup(0)]]
) {
    uint32_t batch_idx = gid;
    if (batch_idx >= params.batch_size) return;
    
    device uint32_t* poly = polys + batch_idx * MLDSA_N;
    
    // Load polynomial to shared memory
    for (uint32_t i = tid; i < MLDSA_N; i += tpg) {
        shared[i] = poly[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Perform all NTT stages
    for (uint32_t s = 0; s < MLDSA_LOG_N; ++s) {
        uint32_t m = 1u << s;
        uint32_t t = MLDSA_N >> (s + 1);
        
        for (uint32_t butterfly = tid; butterfly < MLDSA_N/2; butterfly += tpg) {
            uint32_t i = butterfly / t;
            uint32_t j = butterfly % t;
            uint32_t idx_lo = (i << (MLDSA_LOG_N - s)) + j;
            uint32_t idx_hi = idx_lo + t;
            
            uint32_t omega = twiddles[m + i];
            uint32_t lo = shared[idx_lo];
            uint32_t hi = shared[idx_hi];
            
            ct_butterfly(lo, hi, omega);
            shared[idx_lo] = lo;
            shared[idx_hi] = hi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write back
    for (uint32_t i = tid; i < MLDSA_N; i += tpg) {
        poly[i] = shared[i];
    }
}

// =============================================================================
// Polynomial Arithmetic Kernels
// =============================================================================

// Pointwise multiplication in NTT domain
kernel void mldsa_poly_mul_ntt(
    device uint32_t* result [[buffer(0)]],
    constant uint32_t* a [[buffer(1)]],
    constant uint32_t* b [[buffer(2)]],
    constant MLDSAParams& params [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = tid.y;
    uint32_t coeff_idx = tid.x;
    
    if (batch_idx >= params.batch_size || coeff_idx >= MLDSA_N) return;
    
    uint32_t idx = batch_idx * MLDSA_N + coeff_idx;
    result[idx] = mont_mul(a[idx], b[idx]);
}

// Polynomial addition
kernel void mldsa_poly_add(
    device uint32_t* result [[buffer(0)]],
    constant uint32_t* a [[buffer(1)]],
    constant uint32_t* b [[buffer(2)]],
    constant MLDSAParams& params [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = tid.y;
    uint32_t coeff_idx = tid.x;
    
    if (batch_idx >= params.batch_size || coeff_idx >= MLDSA_N) return;
    
    uint32_t idx = batch_idx * MLDSA_N + coeff_idx;
    result[idx] = mod_add(a[idx], b[idx]);
}

// Polynomial subtraction
kernel void mldsa_poly_sub(
    device uint32_t* result [[buffer(0)]],
    constant uint32_t* a [[buffer(1)]],
    constant uint32_t* b [[buffer(2)]],
    constant MLDSAParams& params [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = tid.y;
    uint32_t coeff_idx = tid.x;
    
    if (batch_idx >= params.batch_size || coeff_idx >= MLDSA_N) return;
    
    uint32_t idx = batch_idx * MLDSA_N + coeff_idx;
    result[idx] = mod_sub(a[idx], b[idx]);
}

// =============================================================================
// ML-DSA Verification Core Kernel
// =============================================================================

// Hint expansion: use hint to recover w1
// This is the core of ML-DSA verification
kernel void mldsa_verify_hint(
    device uint32_t* w1_prime [[buffer(0)]],   // Output: reconstructed w1
    constant uint32_t* w_approx [[buffer(1)]],  // Input: approximate w
    constant uint8_t* hints [[buffer(2)]],      // Input: hint bits
    constant MLDSAParams& params [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = tid.y;
    uint32_t coeff_idx = tid.x;
    
    if (batch_idx >= params.batch_size || coeff_idx >= MLDSA_N * MLDSA_K) return;
    
    uint32_t idx = batch_idx * MLDSA_N * MLDSA_K + coeff_idx;
    uint32_t hint_byte_idx = idx / 8;
    uint32_t hint_bit_idx = idx % 8;
    
    uint8_t hint = (hints[hint_byte_idx] >> hint_bit_idx) & 1;
    uint32_t w = w_approx[idx];
    
    // UseHint algorithm from FIPS 204
    // Simplified: if hint is set, adjust w1 by 1
    int32_t w1 = int32_t(w) >> 19;  // HighBits
    if (hint) {
        w1 = (w1 + 1) % 16;  // Adjust based on hint
    }
    
    w1_prime[idx] = uint32_t(w1 & 0xF);
}

// =============================================================================
// Batch Verification Orchestration
// =============================================================================

// Parameters for verification
struct VerifyParams {
    uint32_t batch_size;
    uint32_t num_valid;     // Output: count of valid signatures
    uint32_t reserved[2];
};

// Check if verification passed for a signature
kernel void mldsa_verify_check(
    device uint32_t* results [[buffer(0)]],     // Output: 1=valid, 0=invalid
    constant uint32_t* c_computed [[buffer(1)]], // Computed challenge
    constant uint32_t* c_expected [[buffer(2)]], // Expected challenge from signature
    constant VerifyParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.batch_size) return;
    
    // Compare challenge hashes (32 coefficients)
    bool match = true;
    for (uint32_t i = 0; i < 32 && match; ++i) {
        if (c_computed[tid * 32 + i] != c_expected[tid * 32 + i]) {
            match = false;
        }
    }
    
    results[tid] = match ? 1 : 0;
}

// =============================================================================
// Matrix-Vector Multiplication for ML-DSA
// =============================================================================

// Compute A * s where A is k x l matrix of polynomials in NTT domain
kernel void mldsa_matrix_vec_mul(
    device uint32_t* result [[buffer(0)]],      // Output: k polynomials
    constant uint32_t* A [[buffer(1)]],          // k*l polynomials (NTT domain)
    constant uint32_t* s [[buffer(2)]],          // l polynomials (NTT domain)
    constant MLDSAParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = tid.z;
    uint32_t row_idx = tid.y;      // Which row of A (0 to k-1)
    uint32_t coeff_idx = tid.x;    // Which coefficient (0 to N-1)
    
    if (batch_idx >= params.batch_size || row_idx >= MLDSA_K || coeff_idx >= MLDSA_N) return;
    
    // Accumulate A[row, col] * s[col] for all columns
    uint32_t acc = 0;
    for (uint32_t col = 0; col < MLDSA_L; ++col) {
        uint32_t A_idx = (row_idx * MLDSA_L + col) * MLDSA_N + coeff_idx;
        uint32_t s_idx = batch_idx * MLDSA_L * MLDSA_N + col * MLDSA_N + coeff_idx;
        
        uint32_t prod = mont_mul(A[A_idx], s[s_idx]);
        acc = mod_add(acc, prod);
    }
    
    uint32_t result_idx = batch_idx * MLDSA_K * MLDSA_N + row_idx * MLDSA_N + coeff_idx;
    result[result_idx] = acc;
}
