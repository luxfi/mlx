// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Polynomial Multiplication for ML-DSA (Dilithium) and ML-KEM (Kyber)
// - Coefficient-wise multiplication in NTT domain
// - Batch operations for vector polynomials
// - Fused NTT multiply for polynomial convolution
//
// Part of the Lux Network GPU acceleration library

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Constants
// ============================================================================

constant uint32_t DILITHIUM_Q = 8380417;
constant uint32_t DILITHIUM_QINV = 58728449;

constant uint32_t KYBER_Q = 3329;
constant uint32_t KYBER_QINV = 62209;

// ============================================================================
// Montgomery Multiplication (same as ntt_negacyclic.metal)
// ============================================================================

METAL_FUNC int32_t montgomery_reduce_dilithium(int64_t a) {
    int32_t t = (int32_t)a * (int32_t)DILITHIUM_QINV;
    return (int32_t)((a - (int64_t)t * (int64_t)DILITHIUM_Q) >> 32);
}

METAL_FUNC int32_t mont_mul_dilithium(int32_t a, int32_t b) {
    return montgomery_reduce_dilithium((int64_t)a * (int64_t)b);
}

METAL_FUNC int16_t montgomery_reduce_kyber(int32_t a) {
    int16_t t = (int16_t)a * (int16_t)KYBER_QINV;
    return (int16_t)((a - (int32_t)t * (int32_t)KYBER_Q) >> 16);
}

METAL_FUNC int16_t mont_mul_kyber(int16_t a, int16_t b) {
    return montgomery_reduce_kyber((int32_t)a * (int32_t)b);
}

// Conditional reduction
METAL_FUNC int32_t cond_sub_dilithium(int32_t a) {
    a += (a >> 31) & (int32_t)DILITHIUM_Q;
    a -= DILITHIUM_Q;
    a += (a >> 31) & (int32_t)DILITHIUM_Q;
    return a;
}

METAL_FUNC int16_t cond_sub_kyber(int16_t a) {
    a += (a >> 15) & (int16_t)KYBER_Q;
    a -= KYBER_Q;
    a += (a >> 15) & (int16_t)KYBER_Q;
    return a;
}

// Barrett reduction for Kyber (must be before first use)
METAL_FUNC int16_t barrett_reduce_kyber(int16_t a) {
    const int32_t v = 20159;  // Barrett constant: floor(2^26 / q) + 1
    int16_t t = (int16_t)((v * (int32_t)a + (1 << 25)) >> 26);
    t *= KYBER_Q;
    return a - t;
}

// ============================================================================
// Coefficient-wise Multiplication in NTT Domain
// ============================================================================

// Dilithium: result[i] = a[i] * b[i] mod q (in Montgomery form)
kernel void poly_pointwise_mul_dilithium(
    device int32_t* result [[buffer(0)]],
    device const int32_t* a [[buffer(1)]],
    device const int32_t* b [[buffer(2)]],
    constant uint32_t& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    result[gid] = mont_mul_dilithium(a[gid], b[gid]);
}

// Kyber: result[i] = a[i] * b[i] mod q
kernel void poly_pointwise_mul_kyber(
    device int16_t* result [[buffer(0)]],
    device const int16_t* a [[buffer(1)]],
    device const int16_t* b [[buffer(2)]],
    constant uint32_t& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    result[gid] = mont_mul_kyber(a[gid], b[gid]);
}

// ============================================================================
// Coefficient-wise Multiply-Accumulate (MAC) in NTT Domain
// ============================================================================

// Dilithium: result[i] += a[i] * b[i] mod q
kernel void poly_pointwise_mac_dilithium(
    device int32_t* result [[buffer(0)]],
    device const int32_t* a [[buffer(1)]],
    device const int32_t* b [[buffer(2)]],
    constant uint32_t& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    int32_t prod = mont_mul_dilithium(a[gid], b[gid]);
    int32_t sum = result[gid] + prod;
    // Lazy reduction: keep in [-2q, 2q]
    result[gid] = sum;
}

// Kyber: result[i] += a[i] * b[i] mod q
kernel void poly_pointwise_mac_kyber(
    device int16_t* result [[buffer(0)]],
    device const int16_t* a [[buffer(1)]],
    device const int16_t* b [[buffer(2)]],
    constant uint32_t& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    int16_t prod = mont_mul_kyber(a[gid], b[gid]);
    int16_t sum = result[gid] + prod;
    result[gid] = sum;
}

// ============================================================================
// Vector Polynomial Multiplication (Batch Operations)
// ============================================================================

// Dilithium: Multiply k polynomials element-wise and accumulate
// result = sum_{i=0}^{k-1} a[i] * b[i] (in NTT domain)
kernel void poly_vector_mul_acc_dilithium(
    device int32_t* result [[buffer(0)]],
    device const int32_t* a [[buffer(1)]],      // k * n coefficients
    device const int32_t* b [[buffer(2)]],      // k * n coefficients
    constant uint32_t& n [[buffer(3)]],         // polynomial degree (256)
    constant uint32_t& k [[buffer(4)]],         // vector dimension
    constant uint32_t& batch [[buffer(5)]],     // number of result polynomials
    uint gid [[thread_position_in_grid]]
) {
    uint32_t total = batch * n;
    if (gid >= total) return;
    
    uint32_t batch_idx = gid / n;
    uint32_t coef_idx = gid % n;
    
    int32_t acc = 0;
    
    for (uint32_t i = 0; i < k; i++) {
        uint32_t a_idx = batch_idx * k * n + i * n + coef_idx;
        uint32_t b_idx = i * n + coef_idx;
        acc += mont_mul_dilithium(a[a_idx], b[b_idx]);
    }
    
    result[gid] = acc;
}

// Kyber: Vector inner product in NTT domain
kernel void poly_vector_mul_acc_kyber(
    device int16_t* result [[buffer(0)]],
    device const int16_t* a [[buffer(1)]],
    device const int16_t* b [[buffer(2)]],
    constant uint32_t& n [[buffer(3)]],
    constant uint32_t& k [[buffer(4)]],
    constant uint32_t& batch [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint32_t total = batch * n;
    if (gid >= total) return;
    
    uint32_t batch_idx = gid / n;
    uint32_t coef_idx = gid % n;
    
    int16_t acc = 0;
    
    for (uint32_t i = 0; i < k; i++) {
        uint32_t a_idx = batch_idx * k * n + i * n + coef_idx;
        uint32_t b_idx = i * n + coef_idx;
        acc += mont_mul_kyber(a[a_idx], b[b_idx]);
    }
    
    result[gid] = barrett_reduce_kyber(acc);
}

// Note: barrett_reduce_kyber is declared earlier in this file

// ============================================================================
// Polynomial Addition and Subtraction
// ============================================================================

kernel void poly_add_dilithium(
    device int32_t* result [[buffer(0)]],
    device const int32_t* a [[buffer(1)]],
    device const int32_t* b [[buffer(2)]],
    constant uint32_t& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    result[gid] = a[gid] + b[gid];
}

kernel void poly_add_kyber(
    device int16_t* result [[buffer(0)]],
    device const int16_t* a [[buffer(1)]],
    device const int16_t* b [[buffer(2)]],
    constant uint32_t& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    result[gid] = a[gid] + b[gid];
}

kernel void poly_sub_dilithium(
    device int32_t* result [[buffer(0)]],
    device const int32_t* a [[buffer(1)]],
    device const int32_t* b [[buffer(2)]],
    constant uint32_t& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    result[gid] = a[gid] - b[gid];
}

kernel void poly_sub_kyber(
    device int16_t* result [[buffer(0)]],
    device const int16_t* a [[buffer(1)]],
    device const int16_t* b [[buffer(2)]],
    constant uint32_t& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    result[gid] = a[gid] - b[gid];
}

// ============================================================================
// Polynomial Reduction
// ============================================================================

kernel void poly_reduce_dilithium(
    device int32_t* data [[buffer(0)]],
    constant uint32_t& size [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    data[gid] = cond_sub_dilithium(data[gid]);
}

kernel void poly_reduce_kyber(
    device int16_t* data [[buffer(0)]],
    constant uint32_t& size [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    data[gid] = cond_sub_kyber(data[gid]);
}

// ============================================================================
// Scalar Multiplication
// ============================================================================

kernel void poly_scalar_mul_dilithium(
    device int32_t* result [[buffer(0)]],
    device const int32_t* a [[buffer(1)]],
    constant int32_t& scalar [[buffer(2)]],
    constant uint32_t& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    result[gid] = mont_mul_dilithium(a[gid], scalar);
}

kernel void poly_scalar_mul_kyber(
    device int16_t* result [[buffer(0)]],
    device const int16_t* a [[buffer(1)]],
    constant int16_t& scalar [[buffer(2)]],
    constant uint32_t& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    result[gid] = mont_mul_kyber(a[gid], scalar);
}

// ============================================================================
// Kyber-Specific: Basemul (NTT multiplication with special structure)
// ============================================================================

// Kyber uses a special NTT structure with degree-2 polynomial multiplication
// for each pair of coefficients. This is the basemul operation.
kernel void kyber_basemul(
    device int16_t* result [[buffer(0)]],
    device const int16_t* a [[buffer(1)]],
    device const int16_t* b [[buffer(2)]],
    device const int16_t* zetas [[buffer(3)]],  // Precomputed zeta values
    constant uint32_t& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    // Each thread handles one pair of coefficients
    uint32_t pair_count = n / 2;
    if (gid >= pair_count) return;
    
    uint32_t idx = gid * 2;
    
    int16_t a0 = a[idx];
    int16_t a1 = a[idx + 1];
    int16_t b0 = b[idx];
    int16_t b1 = b[idx + 1];
    int16_t zeta = zetas[64 + gid];  // Zeta for this pair
    
    // Compute (a0 + a1*X) * (b0 + b1*X) mod (X^2 - zeta)
    // = (a0*b0 + a1*b1*zeta) + (a0*b1 + a1*b0)*X
    
    int16_t r0 = mont_mul_kyber(a0, b0);
    int16_t t = mont_mul_kyber(a1, b1);
    t = mont_mul_kyber(t, zeta);
    r0 = r0 + t;
    
    int16_t r1 = mont_mul_kyber(a0, b1);
    t = mont_mul_kyber(a1, b0);
    r1 = r1 + t;
    
    result[idx] = r0;
    result[idx + 1] = r1;
}

// ============================================================================
// Dilithium-Specific: Power2Round and Decompose helper
// ============================================================================

// Power2Round: decompose a = a1*2^D + a0 where |a0| <= 2^(D-1)
kernel void dilithium_power2round(
    device int32_t* a1 [[buffer(0)]],      // Output: high bits
    device int32_t* a0 [[buffer(1)]],      // Output: low bits
    device const int32_t* a [[buffer(2)]], // Input
    constant uint32_t& size [[buffer(3)]],
    constant uint32_t& d [[buffer(4)]],    // D parameter (13 for Dilithium)
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    int32_t val = a[gid];
    
    // Ensure val is in [0, q)
    val = cond_sub_dilithium(val);
    
    // a0 = centered remainder of a mod 2^D
    int32_t mask = (1 << d) - 1;
    int32_t a0_val = val & mask;
    
    // Center a0 to [-2^(D-1), 2^(D-1)]
    if (a0_val > (1 << (d - 1))) {
        a0_val -= (1 << d);
    }
    
    // a1 = (a - a0) / 2^D
    int32_t a1_val = (val - a0_val) >> d;
    
    a1[gid] = a1_val;
    a0[gid] = a0_val;
}

// ============================================================================
// Batch Polynomial Multiplication (for matrix operations)
// ============================================================================

// Compute C[i][j] = sum_k A[i][k] * B[k][j] for polynomial matrices
// where all polynomials are in NTT domain
kernel void poly_matrix_mul_ntt_dilithium(
    device int32_t* C [[buffer(0)]],           // Output: m x p matrix of polynomials
    device const int32_t* A [[buffer(1)]],     // Input: m x k matrix
    device const int32_t* B [[buffer(2)]],     // Input: k x p matrix
    constant uint32_t& n [[buffer(3)]],        // Polynomial degree (256)
    constant uint32_t& m [[buffer(4)]],        // Matrix dimension m
    constant uint32_t& k_dim [[buffer(5)]],    // Matrix dimension k
    constant uint32_t& p [[buffer(6)]],        // Matrix dimension p
    uint3 gid [[thread_position_in_grid]]
) {
    // gid.x = coefficient index (0 to n-1)
    // gid.y = output matrix index (flattened i*p + j)
    
    if (gid.x >= n) return;
    
    uint32_t out_idx = gid.y;
    if (out_idx >= m * p) return;
    
    uint32_t i = out_idx / p;
    uint32_t j = out_idx % p;
    uint32_t coef = gid.x;
    
    int32_t acc = 0;
    
    for (uint32_t k = 0; k < k_dim; k++) {
        uint32_t a_poly_idx = i * k_dim + k;
        uint32_t b_poly_idx = k * p + j;
        
        int32_t a_coef = A[a_poly_idx * n + coef];
        int32_t b_coef = B[b_poly_idx * n + coef];
        
        acc += mont_mul_dilithium(a_coef, b_coef);
    }
    
    C[out_idx * n + coef] = acc;
}

kernel void poly_matrix_mul_ntt_kyber(
    device int16_t* C [[buffer(0)]],
    device const int16_t* A [[buffer(1)]],
    device const int16_t* B [[buffer(2)]],
    constant uint32_t& n [[buffer(3)]],
    constant uint32_t& m [[buffer(4)]],
    constant uint32_t& k_dim [[buffer(5)]],
    constant uint32_t& p [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= n) return;
    
    uint32_t out_idx = gid.y;
    if (out_idx >= m * p) return;
    
    uint32_t i = out_idx / p;
    uint32_t j = out_idx % p;
    uint32_t coef = gid.x;
    
    int16_t acc = 0;
    
    for (uint32_t k = 0; k < k_dim; k++) {
        uint32_t a_poly_idx = i * k_dim + k;
        uint32_t b_poly_idx = k * p + j;
        
        int16_t a_coef = A[a_poly_idx * n + coef];
        int16_t b_coef = B[b_poly_idx * n + coef];
        
        acc += mont_mul_kyber(a_coef, b_coef);
    }
    
    C[out_idx * n + coef] = barrett_reduce_kyber(acc);
}
