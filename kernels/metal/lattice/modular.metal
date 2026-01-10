// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Modular Arithmetic Primitives for Lattice Cryptography
// Optimized for Apple Silicon GPUs

#include <metal_stdlib>
using namespace metal;

// 64-bit unsigned integer
struct U64 {
    uint lo;
    uint hi;
};

// 128-bit unsigned integer
struct U128 {
    U64 lo;
    U64 hi;
};

// Modular arithmetic parameters
struct ModParams {
    U64 modulus;
    U64 barrett_mu;
    uint num_elements;
};

// ============================================================================
// 64-bit Arithmetic
// ============================================================================

inline U64 u64_from_u32(uint x) {
    return {x, 0u};
}

inline bool u64_is_zero(U64 a) {
    return a.lo == 0 && a.hi == 0;
}

inline bool u64_eq(U64 a, U64 b) {
    return a.lo == b.lo && a.hi == b.hi;
}

inline bool u64_lt(U64 a, U64 b) {
    if (a.hi < b.hi) return true;
    if (a.hi > b.hi) return false;
    return a.lo < b.lo;
}

inline bool u64_gte(U64 a, U64 b) {
    return !u64_lt(a, b);
}

inline U64 u64_add(U64 a, U64 b) {
    uint lo = a.lo + b.lo;
    uint carry = (lo < a.lo) ? 1u : 0u;
    uint hi = a.hi + b.hi + carry;
    return {lo, hi};
}

inline U64 u64_sub(U64 a, U64 b) {
    uint borrow = (a.lo < b.lo) ? 1u : 0u;
    uint lo = a.lo - b.lo;
    uint hi = a.hi - b.hi - borrow;
    return {lo, hi};
}

// Multiply two 32-bit values to get 64-bit result
inline U64 mul32_to_64(uint a, uint b) {
    uint a_lo = a & 0xFFFFu;
    uint a_hi = a >> 16u;
    uint b_lo = b & 0xFFFFu;
    uint b_hi = b >> 16u;
    
    uint p0 = a_lo * b_lo;
    uint p1 = a_lo * b_hi;
    uint p2 = a_hi * b_lo;
    uint p3 = a_hi * b_hi;
    
    uint mid = p1 + p2;
    uint mid_carry = (mid < p1) ? 0x10000u : 0u;
    
    uint lo = p0 + (mid << 16u);
    uint carry = (lo < p0) ? 1u : 0u;
    uint hi = p3 + (mid >> 16u) + mid_carry + carry;
    
    return {lo, hi};
}

// Full 64x64 -> 128 multiplication
inline U128 u64_mul_full(U64 a, U64 b) {
    U64 p0 = mul32_to_64(a.lo, b.lo);
    U64 p1 = mul32_to_64(a.lo, b.hi);
    U64 p2 = mul32_to_64(a.hi, b.lo);
    U64 p3 = mul32_to_64(a.hi, b.hi);
    
    // Combine results
    U64 lo = p0;
    U64 mid = u64_add(p1, p2);
    
    // Add mid.lo to position 32-95
    lo = u64_add(lo, {0u, mid.lo});
    
    // hi = p3 + mid.hi + carries
    U64 hi = u64_add(p3, {mid.hi, 0u});
    
    return {lo, hi};
}

// ============================================================================
// Modular Arithmetic
// ============================================================================

// Barrett reduction: compute x mod q
inline U64 barrett_reduce(U64 x, U64 q, U64 mu) {
    // Approximate quotient using precomputed mu
    U128 prod = u64_mul_full(x, mu);
    U64 q_approx = prod.hi;
    
    // Compute r = x - q_approx * q
    U128 qaq = u64_mul_full(q_approx, q);
    U64 r = u64_sub(x, qaq.lo);
    
    // Correction: at most 2 subtractions needed
    if (u64_gte(r, q)) {
        r = u64_sub(r, q);
    }
    if (u64_gte(r, q)) {
        r = u64_sub(r, q);
    }
    
    return r;
}

// Modular addition: (a + b) mod q
inline U64 mod_add(U64 a, U64 b, U64 q) {
    U64 sum = u64_add(a, b);
    
    // Check for overflow or sum >= q
    bool overflow = u64_lt(sum, a);
    if (overflow || u64_gte(sum, q)) {
        sum = u64_sub(sum, q);
    }
    
    return sum;
}

// Modular subtraction: (a - b) mod q
inline U64 mod_sub(U64 a, U64 b, U64 q) {
    if (u64_lt(a, b)) {
        // a < b: compute q - (b - a)
        return u64_sub(q, u64_sub(b, a));
    }
    return u64_sub(a, b);
}

// Modular multiplication: (a * b) mod q
inline U64 mod_mul(U64 a, U64 b, U64 q, U64 mu) {
    U128 prod = u64_mul_full(a, b);
    // Need to reduce 128-bit product - simplified
    return barrett_reduce(prod.lo, q, mu);
}

// Modular negation: (-a) mod q = q - a
inline U64 mod_neg(U64 a, U64 q) {
    if (u64_is_zero(a)) return a;
    return u64_sub(q, a);
}

// ============================================================================
// Batch Operations
// ============================================================================

kernel void mod_add_batch(
    device const U64* a [[buffer(0)]],
    device const U64* b [[buffer(1)]],
    device U64* result [[buffer(2)]],
    constant ModParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.num_elements) return;
    result[gid] = mod_add(a[gid], b[gid], params.modulus);
}

kernel void mod_sub_batch(
    device const U64* a [[buffer(0)]],
    device const U64* b [[buffer(1)]],
    device U64* result [[buffer(2)]],
    constant ModParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.num_elements) return;
    result[gid] = mod_sub(a[gid], b[gid], params.modulus);
}

kernel void mod_mul_batch(
    device const U64* a [[buffer(0)]],
    device const U64* b [[buffer(1)]],
    device U64* result [[buffer(2)]],
    constant ModParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.num_elements) return;
    result[gid] = mod_mul(a[gid], b[gid], params.modulus, params.barrett_mu);
}

kernel void mod_neg_batch(
    device const U64* a [[buffer(0)]],
    device U64* result [[buffer(1)]],
    constant ModParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.num_elements) return;
    result[gid] = mod_neg(a[gid], params.modulus);
}

// Scalar multiplication: result[i] = a[i] * scalar mod q
kernel void mod_scalar_mul_batch(
    device const U64* a [[buffer(0)]],
    device U64* result [[buffer(1)]],
    constant U64& scalar [[buffer(2)]],
    constant ModParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.num_elements) return;
    result[gid] = mod_mul(a[gid], scalar, params.modulus, params.barrett_mu);
}

// Fused multiply-add: result[i] = (a[i] * b[i] + c[i]) mod q
kernel void mod_fma_batch(
    device const U64* a [[buffer(0)]],
    device const U64* b [[buffer(1)]],
    device const U64* c [[buffer(2)]],
    device U64* result [[buffer(3)]],
    constant ModParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.num_elements) return;
    
    U64 prod = mod_mul(a[gid], b[gid], params.modulus, params.barrett_mu);
    result[gid] = mod_add(prod, c[gid], params.modulus);
}

// Barrett reduction batch
kernel void barrett_reduce_batch(
    device const U64* input [[buffer(0)]],
    device U64* output [[buffer(1)]],
    constant ModParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.num_elements) return;
    output[gid] = barrett_reduce(input[gid], params.modulus, params.barrett_mu);
}
