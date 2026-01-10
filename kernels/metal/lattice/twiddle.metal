// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Twiddle Factor Generation for NTT/FFT
// Optimized for Apple Silicon GPUs

#include <metal_stdlib>
using namespace metal;

struct U64 {
    uint lo;
    uint hi;
};

struct TwiddleParams {
    uint N;
    uint log_N;
    uint modulus_idx;
    uint num_moduli;
    uint direction;  // 0 = forward, 1 = inverse
};

struct RnsModulus {
    U64 q;
    U64 root;
    U64 root_inv;
    U64 n_inv;
    U64 barrett_mu;
};

// ============================================================================
// 64-bit Arithmetic
// ============================================================================

inline U64 u64_zero() { return {0u, 0u}; }
inline U64 u64_one() { return {1u, 0u}; }

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

// ============================================================================
// Modular Arithmetic
// ============================================================================

inline U64 barrett_reduce(U64 x, U64 q, U64 mu) {
    // Simplified Barrett reduction
    // Full implementation needs 128-bit intermediate
    if (u64_gte(x, q)) {
        x = u64_sub(x, q);
    }
    if (u64_gte(x, q)) {
        x = u64_sub(x, q);
    }
    return x;
}

inline U64 mod_mul(U64 a, U64 b, U64 q, U64 mu) {
    // Simplified - full implementation needs proper Montgomery
    U64 prod = mul32_to_64(a.lo, b.lo);
    return barrett_reduce(prod, q, mu);
}

inline U64 mod_pow(U64 base, uint exp, U64 q, U64 mu) {
    U64 result = u64_one();
    U64 b = base;
    
    while (exp > 0) {
        if (exp & 1u) {
            result = mod_mul(result, b, q, mu);
        }
        b = mod_mul(b, b, q, mu);
        exp >>= 1u;
    }
    
    return result;
}

// ============================================================================
// Bit Reversal
// ============================================================================

inline uint bit_reverse(uint x, uint log_n) {
    uint result = 0;
    for (uint i = 0; i < log_n; i++) {
        result = (result << 1u) | (x & 1u);
        x >>= 1u;
    }
    return result;
}

// ============================================================================
// Twiddle Generation Kernels
// ============================================================================

kernel void generate_twiddles(
    device U64* twiddles [[buffer(0)]],
    device U64* twiddles_inv [[buffer(1)]],
    device const RnsModulus* moduli [[buffer(2)]],
    constant TwiddleParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint idx = gid;
    uint N = params.N;
    
    if (idx >= N) return;
    
    uint mod_idx = params.modulus_idx;
    U64 q = moduli[mod_idx].q;
    U64 mu = moduli[mod_idx].barrett_mu;
    U64 root = (params.direction == 0) ? moduli[mod_idx].root : moduli[mod_idx].root_inv;
    
    uint br_idx = bit_reverse(idx, params.log_N);
    
    U64 twiddle = mod_pow(root, br_idx, q, mu);
    
    uint output_offset = mod_idx * N;
    if (params.direction == 0) {
        twiddles[output_offset + idx] = twiddle;
    } else {
        twiddles_inv[output_offset + idx] = twiddle;
    }
}

kernel void generate_stage_twiddles(
    device U64* twiddles [[buffer(0)]],
    device const RnsModulus* moduli [[buffer(1)]],
    constant TwiddleParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint idx = gid.x;
    uint stage = gid.y;
    uint N = params.N;
    
    uint half_size = 1u << stage;
    if (idx >= half_size) return;
    
    uint mod_idx = params.modulus_idx;
    U64 q = moduli[mod_idx].q;
    U64 mu = moduli[mod_idx].barrett_mu;
    U64 root = moduli[mod_idx].root;
    
    uint exp = (idx * N) >> (stage + 1u);
    U64 twiddle = mod_pow(root, exp, q, mu);
    
    uint output_offset = mod_idx * N + (1u << stage) - 1u + idx;
    twiddles[output_offset] = twiddle;
}

kernel void generate_inverse_twiddles(
    device U64* twiddles_inv [[buffer(0)]],
    device const RnsModulus* moduli [[buffer(1)]],
    constant TwiddleParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint idx = gid;
    uint N = params.N;
    
    if (idx >= N) return;
    
    uint mod_idx = params.modulus_idx;
    U64 q = moduli[mod_idx].q;
    U64 mu = moduli[mod_idx].barrett_mu;
    U64 root_inv = moduli[mod_idx].root_inv;
    U64 n_inv = moduli[mod_idx].n_inv;
    
    uint br_idx = bit_reverse(idx, params.log_N);
    
    U64 twiddle = mod_pow(root_inv, br_idx, q, mu);
    twiddle = mod_mul(twiddle, n_inv, q, mu);
    
    uint output_offset = mod_idx * N;
    twiddles_inv[output_offset + idx] = twiddle;
}

kernel void generate_rns_twiddles(
    device U64* twiddles [[buffer(0)]],
    device const RnsModulus* moduli [[buffer(1)]],
    constant TwiddleParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint idx = gid.x;
    uint mod_idx = gid.y;
    uint N = params.N;
    
    if (idx >= N || mod_idx >= params.num_moduli) return;
    
    U64 q = moduli[mod_idx].q;
    U64 mu = moduli[mod_idx].barrett_mu;
    U64 root = moduli[mod_idx].root;
    
    uint br_idx = bit_reverse(idx, params.log_N);
    U64 twiddle = mod_pow(root, br_idx, q, mu);
    
    uint output_offset = mod_idx * N + idx;
    twiddles[output_offset] = twiddle;
}
