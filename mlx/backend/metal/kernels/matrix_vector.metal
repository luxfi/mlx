// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Matrix-Vector Multiplication for ML-DSA (Dilithium) and ML-KEM (Kyber)
// - Matrix-vector multiplication for A*s (key generation, signing)
// - Parallel row processing for large matrices
// - NTT domain operations for efficiency
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

// Standard polynomial degree for both schemes
constant uint32_t N = 256;

// ============================================================================
// Montgomery Multiplication
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

// ============================================================================
// Barrett Reduction
// ============================================================================

METAL_FUNC int16_t barrett_reduce_kyber(int16_t a) {
    constant int32_t v = 20159;
    int16_t t = (int16_t)((v * (int32_t)a + (1 << 25)) >> 26);
    t *= KYBER_Q;
    return a - t;
}

METAL_FUNC int32_t barrett_reduce_dilithium(int32_t a) {
    constant int64_t v = 8396807;
    int32_t t = (int32_t)((v * (int64_t)a) >> 46);
    t *= DILITHIUM_Q;
    return a - t;
}

// ============================================================================
// Matrix-Vector Multiplication Parameters
// ============================================================================

struct MatVecParams {
    uint32_t n;        // Polynomial degree (256)
    uint32_t k;        // Number of rows in matrix A
    uint32_t l;        // Number of columns in matrix A (or same as k for Kyber)
    uint32_t batch;    // Batch size for parallel operations
};

// ============================================================================
// Dilithium Matrix-Vector Multiplication
// ============================================================================

// Compute t = A*s where A is k×l matrix and s is l-vector of polynomials
// All polynomials are in NTT domain
// Output: t is k-vector of polynomials
kernel void matrix_vector_mul_dilithium(
    device int32_t* t [[buffer(0)]],              // Output: k polynomials
    device const int32_t* A [[buffer(1)]],        // Matrix: k × l polynomials
    device const int32_t* s [[buffer(2)]],        // Vector: l polynomials
    constant MatVecParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint32_t n = params.n;
    uint32_t k = params.k;
    uint32_t l = params.l;
    
    // gid.x = coefficient index (0 to n-1)
    // gid.y = row index (0 to k-1)
    
    if (gid.x >= n || gid.y >= k) return;
    
    uint32_t coef_idx = gid.x;
    uint32_t row = gid.y;
    
    int32_t acc = 0;
    
    // Inner product of row `row` of A with vector s
    for (uint32_t col = 0; col < l; col++) {
        // A[row][col] polynomial, coefficient coef_idx
        uint32_t a_idx = (row * l + col) * n + coef_idx;
        // s[col] polynomial, coefficient coef_idx
        uint32_t s_idx = col * n + coef_idx;
        
        acc += mont_mul_dilithium(A[a_idx], s[s_idx]);
    }
    
    // t[row] polynomial, coefficient coef_idx
    uint32_t t_idx = row * n + coef_idx;
    t[t_idx] = acc;
}

// Compute t = A*s + e (add error vector)
kernel void matrix_vector_mul_add_dilithium(
    device int32_t* t [[buffer(0)]],
    device const int32_t* A [[buffer(1)]],
    device const int32_t* s [[buffer(2)]],
    device const int32_t* e [[buffer(3)]],        // Error vector: k polynomials
    constant MatVecParams& params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint32_t n = params.n;
    uint32_t k = params.k;
    uint32_t l = params.l;
    
    if (gid.x >= n || gid.y >= k) return;
    
    uint32_t coef_idx = gid.x;
    uint32_t row = gid.y;
    
    int32_t acc = 0;
    
    for (uint32_t col = 0; col < l; col++) {
        uint32_t a_idx = (row * l + col) * n + coef_idx;
        uint32_t s_idx = col * n + coef_idx;
        acc += mont_mul_dilithium(A[a_idx], s[s_idx]);
    }
    
    // Add error
    uint32_t e_idx = row * n + coef_idx;
    acc += e[e_idx];
    
    uint32_t t_idx = row * n + coef_idx;
    t[t_idx] = acc;
}

// ============================================================================
// Kyber Matrix-Vector Multiplication
// ============================================================================

// Compute t = A*s where A is k×k matrix and s is k-vector
kernel void matrix_vector_mul_kyber(
    device int16_t* t [[buffer(0)]],
    device const int16_t* A [[buffer(1)]],
    device const int16_t* s [[buffer(2)]],
    constant MatVecParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint32_t n = params.n;
    uint32_t k = params.k;
    
    if (gid.x >= n || gid.y >= k) return;
    
    uint32_t coef_idx = gid.x;
    uint32_t row = gid.y;
    
    int16_t acc = 0;
    
    for (uint32_t col = 0; col < k; col++) {
        uint32_t a_idx = (row * k + col) * n + coef_idx;
        uint32_t s_idx = col * n + coef_idx;
        acc += mont_mul_kyber(A[a_idx], s[s_idx]);
    }
    
    uint32_t t_idx = row * n + coef_idx;
    t[t_idx] = barrett_reduce_kyber(acc);
}

// Compute t = A*s + e
kernel void matrix_vector_mul_add_kyber(
    device int16_t* t [[buffer(0)]],
    device const int16_t* A [[buffer(1)]],
    device const int16_t* s [[buffer(2)]],
    device const int16_t* e [[buffer(3)]],
    constant MatVecParams& params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint32_t n = params.n;
    uint32_t k = params.k;
    
    if (gid.x >= n || gid.y >= k) return;
    
    uint32_t coef_idx = gid.x;
    uint32_t row = gid.y;
    
    int16_t acc = 0;
    
    for (uint32_t col = 0; col < k; col++) {
        uint32_t a_idx = (row * k + col) * n + coef_idx;
        uint32_t s_idx = col * n + coef_idx;
        acc += mont_mul_kyber(A[a_idx], s[s_idx]);
    }
    
    uint32_t e_idx = row * n + coef_idx;
    acc += e[e_idx];
    
    uint32_t t_idx = row * n + coef_idx;
    t[t_idx] = barrett_reduce_kyber(acc);
}

// ============================================================================
// Transpose Matrix Multiplication: t = A^T * s
// ============================================================================

// Dilithium: Compute t = A^T * s (transpose multiply)
// Used in verification: w = A^T * z
kernel void matrix_transpose_vector_mul_dilithium(
    device int32_t* t [[buffer(0)]],              // Output: l polynomials
    device const int32_t* A [[buffer(1)]],        // Matrix: k × l polynomials
    device const int32_t* s [[buffer(2)]],        // Vector: k polynomials
    constant MatVecParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint32_t n = params.n;
    uint32_t k = params.k;
    uint32_t l = params.l;
    
    // gid.x = coefficient index
    // gid.y = column index (output vector element)
    
    if (gid.x >= n || gid.y >= l) return;
    
    uint32_t coef_idx = gid.x;
    uint32_t col = gid.y;
    
    int32_t acc = 0;
    
    // Sum over rows
    for (uint32_t row = 0; row < k; row++) {
        // A^T[col][row] = A[row][col]
        uint32_t a_idx = (row * l + col) * n + coef_idx;
        uint32_t s_idx = row * n + coef_idx;
        acc += mont_mul_dilithium(A[a_idx], s[s_idx]);
    }
    
    uint32_t t_idx = col * n + coef_idx;
    t[t_idx] = acc;
}

// Kyber transpose multiply
kernel void matrix_transpose_vector_mul_kyber(
    device int16_t* t [[buffer(0)]],
    device const int16_t* A [[buffer(1)]],
    device const int16_t* s [[buffer(2)]],
    constant MatVecParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint32_t n = params.n;
    uint32_t k = params.k;
    
    if (gid.x >= n || gid.y >= k) return;
    
    uint32_t coef_idx = gid.x;
    uint32_t col = gid.y;
    
    int16_t acc = 0;
    
    for (uint32_t row = 0; row < k; row++) {
        uint32_t a_idx = (row * k + col) * n + coef_idx;
        uint32_t s_idx = row * n + coef_idx;
        acc += mont_mul_kyber(A[a_idx], s[s_idx]);
    }
    
    uint32_t t_idx = col * n + coef_idx;
    t[t_idx] = barrett_reduce_kyber(acc);
}

// ============================================================================
// Vector Inner Product
// ============================================================================

// Compute scalar polynomial: result = sum(a[i] * b[i])
kernel void vector_inner_product_dilithium(
    device int32_t* result [[buffer(0)]],         // Output: 1 polynomial
    device const int32_t* a [[buffer(1)]],        // Vector: k polynomials
    device const int32_t* b [[buffer(2)]],        // Vector: k polynomials
    constant MatVecParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint32_t n = params.n;
    uint32_t k = params.k;
    
    if (gid >= n) return;
    
    int32_t acc = 0;
    
    for (uint32_t i = 0; i < k; i++) {
        uint32_t idx = i * n + gid;
        acc += mont_mul_dilithium(a[idx], b[idx]);
    }
    
    result[gid] = acc;
}

kernel void vector_inner_product_kyber(
    device int16_t* result [[buffer(0)]],
    device const int16_t* a [[buffer(1)]],
    device const int16_t* b [[buffer(2)]],
    constant MatVecParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint32_t n = params.n;
    uint32_t k = params.k;
    
    if (gid >= n) return;
    
    int16_t acc = 0;
    
    for (uint32_t i = 0; i < k; i++) {
        uint32_t idx = i * n + gid;
        acc += mont_mul_kyber(a[idx], b[idx]);
    }
    
    result[gid] = barrett_reduce_kyber(acc);
}

// ============================================================================
// Vector Addition/Subtraction
// ============================================================================

kernel void vector_add_dilithium(
    device int32_t* result [[buffer(0)]],
    device const int32_t* a [[buffer(1)]],
    device const int32_t* b [[buffer(2)]],
    constant uint32_t& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    result[gid] = a[gid] + b[gid];
}

kernel void vector_sub_dilithium(
    device int32_t* result [[buffer(0)]],
    device const int32_t* a [[buffer(1)]],
    device const int32_t* b [[buffer(2)]],
    constant uint32_t& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    result[gid] = a[gid] - b[gid];
}

kernel void vector_add_kyber(
    device int16_t* result [[buffer(0)]],
    device const int16_t* a [[buffer(1)]],
    device const int16_t* b [[buffer(2)]],
    constant uint32_t& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    result[gid] = a[gid] + b[gid];
}

kernel void vector_sub_kyber(
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
// Fused Operations for Key Generation
// ============================================================================

// Dilithium key generation: t = A*s1 + s2
kernel void keygen_compute_t_dilithium(
    device int32_t* t [[buffer(0)]],
    device const int32_t* A [[buffer(1)]],
    device const int32_t* s1 [[buffer(2)]],
    device const int32_t* s2 [[buffer(3)]],
    constant MatVecParams& params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint32_t n = params.n;
    uint32_t k = params.k;
    uint32_t l = params.l;
    
    if (gid.x >= n || gid.y >= k) return;
    
    uint32_t coef_idx = gid.x;
    uint32_t row = gid.y;
    
    int32_t acc = 0;
    
    // Compute A * s1
    for (uint32_t col = 0; col < l; col++) {
        uint32_t a_idx = (row * l + col) * n + coef_idx;
        uint32_t s1_idx = col * n + coef_idx;
        acc += mont_mul_dilithium(A[a_idx], s1[s1_idx]);
    }
    
    // Add s2
    uint32_t s2_idx = row * n + coef_idx;
    acc += s2[s2_idx];
    
    uint32_t t_idx = row * n + coef_idx;
    t[t_idx] = acc;
}

// Kyber key generation: t = A*s + e
kernel void keygen_compute_t_kyber(
    device int16_t* t [[buffer(0)]],
    device const int16_t* A [[buffer(1)]],
    device const int16_t* s [[buffer(2)]],
    device const int16_t* e [[buffer(3)]],
    constant MatVecParams& params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint32_t n = params.n;
    uint32_t k = params.k;
    
    if (gid.x >= n || gid.y >= k) return;
    
    uint32_t coef_idx = gid.x;
    uint32_t row = gid.y;
    
    int16_t acc = 0;
    
    for (uint32_t col = 0; col < k; col++) {
        uint32_t a_idx = (row * k + col) * n + coef_idx;
        uint32_t s_idx = col * n + coef_idx;
        acc += mont_mul_kyber(A[a_idx], s[s_idx]);
    }
    
    uint32_t e_idx = row * n + coef_idx;
    acc = barrett_reduce_kyber(acc) + e[e_idx];
    
    uint32_t t_idx = row * n + coef_idx;
    t[t_idx] = barrett_reduce_kyber(acc);
}

// ============================================================================
// Fused Operations for Signing (Dilithium)
// ============================================================================

// Compute w = A*y for signing
kernel void sign_compute_w_dilithium(
    device int32_t* w [[buffer(0)]],
    device const int32_t* A [[buffer(1)]],
    device const int32_t* y [[buffer(2)]],
    constant MatVecParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint32_t n = params.n;
    uint32_t k = params.k;
    uint32_t l = params.l;
    
    if (gid.x >= n || gid.y >= k) return;
    
    uint32_t coef_idx = gid.x;
    uint32_t row = gid.y;
    
    int32_t acc = 0;
    
    for (uint32_t col = 0; col < l; col++) {
        uint32_t a_idx = (row * l + col) * n + coef_idx;
        uint32_t y_idx = col * n + coef_idx;
        acc += mont_mul_dilithium(A[a_idx], y[y_idx]);
    }
    
    uint32_t w_idx = row * n + coef_idx;
    w[w_idx] = barrett_reduce_dilithium(acc);
}

// Compute z = y + c*s1 for signing
kernel void sign_compute_z_dilithium(
    device int32_t* z [[buffer(0)]],
    device const int32_t* y [[buffer(1)]],
    device const int32_t* c [[buffer(2)]],        // Challenge (1 polynomial)
    device const int32_t* s1 [[buffer(3)]],       // Secret (l polynomials)
    constant MatVecParams& params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint32_t n = params.n;
    uint32_t l = params.l;
    
    // gid.x = coefficient index
    // gid.y = polynomial index (0 to l-1)
    
    if (gid.x >= n || gid.y >= l) return;
    
    uint32_t coef_idx = gid.x;
    uint32_t poly_idx = gid.y;
    
    // c * s1[poly_idx] (pointwise in NTT domain)
    uint32_t idx = poly_idx * n + coef_idx;
    int32_t cs = mont_mul_dilithium(c[coef_idx], s1[idx]);
    
    // z = y + c*s
    z[idx] = y[idx] + cs;
}

// ============================================================================
// Fused Operations for Encryption/Decryption (Kyber)
// ============================================================================

// Kyber encryption: u = A^T*r + e1
kernel void encrypt_compute_u_kyber(
    device int16_t* u [[buffer(0)]],
    device const int16_t* A [[buffer(1)]],        // Matrix A (k × k)
    device const int16_t* r [[buffer(2)]],        // Random vector (k polynomials)
    device const int16_t* e1 [[buffer(3)]],       // Error vector (k polynomials)
    constant MatVecParams& params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint32_t n = params.n;
    uint32_t k = params.k;
    
    if (gid.x >= n || gid.y >= k) return;
    
    uint32_t coef_idx = gid.x;
    uint32_t col = gid.y;
    
    int16_t acc = 0;
    
    // A^T[col][row] = A[row][col], sum over rows
    for (uint32_t row = 0; row < k; row++) {
        uint32_t a_idx = (row * k + col) * n + coef_idx;
        uint32_t r_idx = row * n + coef_idx;
        acc += mont_mul_kyber(A[a_idx], r[r_idx]);
    }
    
    uint32_t e1_idx = col * n + coef_idx;
    acc = barrett_reduce_kyber(acc) + e1[e1_idx];
    
    uint32_t u_idx = col * n + coef_idx;
    u[u_idx] = barrett_reduce_kyber(acc);
}

// Kyber encryption: v = t^T*r + e2 + m
kernel void encrypt_compute_v_kyber(
    device int16_t* v [[buffer(0)]],
    device const int16_t* t [[buffer(1)]],        // Public key (k polynomials)
    device const int16_t* r [[buffer(2)]],        // Random vector (k polynomials)
    device const int16_t* e2 [[buffer(3)]],       // Error (1 polynomial)
    device const int16_t* m [[buffer(4)]],        // Message (1 polynomial, encoded)
    constant MatVecParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint32_t n = params.n;
    uint32_t k = params.k;
    
    if (gid >= n) return;
    
    int16_t acc = 0;
    
    // Inner product t^T * r
    for (uint32_t i = 0; i < k; i++) {
        uint32_t idx = i * n + gid;
        acc += mont_mul_kyber(t[idx], r[idx]);
    }
    
    // Add e2 and m
    acc = barrett_reduce_kyber(acc) + e2[gid] + m[gid];
    
    v[gid] = barrett_reduce_kyber(acc);
}

// Kyber decryption: m' = v - s^T*u
kernel void decrypt_compute_m_kyber(
    device int16_t* mp [[buffer(0)]],
    device const int16_t* v [[buffer(1)]],        // Ciphertext component
    device const int16_t* s [[buffer(2)]],        // Secret key (k polynomials)
    device const int16_t* u [[buffer(3)]],        // Ciphertext component (k polynomials)
    constant MatVecParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint32_t n = params.n;
    uint32_t k = params.k;
    
    if (gid >= n) return;
    
    int16_t acc = 0;
    
    // Inner product s^T * u
    for (uint32_t i = 0; i < k; i++) {
        uint32_t idx = i * n + gid;
        acc += mont_mul_kyber(s[idx], u[idx]);
    }
    
    // m' = v - s^T*u
    acc = barrett_reduce_kyber(acc);
    mp[gid] = v[gid] - acc;
}
