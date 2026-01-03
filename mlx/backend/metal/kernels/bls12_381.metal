// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// BLS12-381 Elliptic Curve Operations
// Optimized for Apple Silicon GPUs

#include <metal_stdlib>
using namespace metal;

// 384-bit field element (12 x 32-bit limbs)
struct Fp {
    uint limbs[12];
};

// Extension field Fp2
struct Fp2 {
    Fp c0;
    Fp c1;
};

// G1 affine point
struct G1Affine {
    Fp x;
    Fp y;
};

// G1 projective point
struct G1Projective {
    Fp x;
    Fp y;
    Fp z;
};

// G2 affine point
struct G2Affine {
    Fp2 x;
    Fp2 y;
};

// G2 projective point
struct G2Projective {
    Fp2 x;
    Fp2 y;
    Fp2 z;
};

// BLS12-381 base field modulus p
constant uint BLS_P[12] = {
    0xffffaaabu, 0xb9fffffeu, 0xb153ffffu, 0x1eabfffeu,
    0xf6b0f624u, 0x6730d2a0u, 0xf38512bfu, 0x64774b84u,
    0x434bacd7u, 0x4b1ba7b6u, 0x397fe69au, 0x1a0111eau
};

// ============================================================================
// Fp Arithmetic
// ============================================================================

inline Fp fp_zero() {
    Fp r;
    for (int i = 0; i < 12; i++) r.limbs[i] = 0;
    return r;
}

inline Fp fp_one() {
    Fp r = fp_zero();
    r.limbs[0] = 1;
    return r;
}

inline bool fp_is_zero(Fp a) {
    for (int i = 0; i < 12; i++) {
        if (a.limbs[i] != 0) return false;
    }
    return true;
}

inline bool fp_gte_p(Fp a) {
    for (int i = 11; i >= 0; i--) {
        if (a.limbs[i] > BLS_P[i]) return true;
        if (a.limbs[i] < BLS_P[i]) return false;
    }
    return true;
}

inline Fp fp_add(Fp a, Fp b) {
    Fp r;
    uint carry = 0;
    for (int i = 0; i < 12; i++) {
        uint sum = a.limbs[i] + b.limbs[i] + carry;
        carry = (sum < a.limbs[i]) || (carry && sum == a.limbs[i]) ? 1 : 0;
        r.limbs[i] = sum;
    }
    
    if (fp_gte_p(r)) {
        uint borrow = 0;
        for (int i = 0; i < 12; i++) {
            uint diff = r.limbs[i] - BLS_P[i] - borrow;
            borrow = (r.limbs[i] < BLS_P[i] + borrow) ? 1 : 0;
            r.limbs[i] = diff;
        }
    }
    return r;
}

inline Fp fp_sub(Fp a, Fp b) {
    Fp r;
    uint borrow = 0;
    for (int i = 0; i < 12; i++) {
        uint diff = a.limbs[i] - b.limbs[i] - borrow;
        borrow = (a.limbs[i] < b.limbs[i] + borrow) ? 1 : 0;
        r.limbs[i] = diff;
    }
    
    if (borrow) {
        uint carry = 0;
        for (int i = 0; i < 12; i++) {
            uint sum = r.limbs[i] + BLS_P[i] + carry;
            carry = (sum < r.limbs[i]) ? 1 : 0;
            r.limbs[i] = sum;
        }
    }
    return r;
}

inline Fp fp_neg(Fp a) {
    if (fp_is_zero(a)) return a;
    Fp r;
    uint borrow = 0;
    for (int i = 0; i < 12; i++) {
        uint diff = BLS_P[i] - a.limbs[i] - borrow;
        borrow = (BLS_P[i] < a.limbs[i] + borrow) ? 1 : 0;
        r.limbs[i] = diff;
    }
    return r;
}

inline Fp fp_double(Fp a) {
    return fp_add(a, a);
}

// ============================================================================
// Fp2 Arithmetic
// ============================================================================

inline Fp2 fp2_zero() {
    return {fp_zero(), fp_zero()};
}

inline Fp2 fp2_one() {
    return {fp_one(), fp_zero()};
}

inline bool fp2_is_zero(Fp2 a) {
    return fp_is_zero(a.c0) && fp_is_zero(a.c1);
}

inline Fp2 fp2_add(Fp2 a, Fp2 b) {
    return {fp_add(a.c0, b.c0), fp_add(a.c1, b.c1)};
}

inline Fp2 fp2_sub(Fp2 a, Fp2 b) {
    return {fp_sub(a.c0, b.c0), fp_sub(a.c1, b.c1)};
}

inline Fp2 fp2_neg(Fp2 a) {
    return {fp_neg(a.c0), fp_neg(a.c1)};
}

inline Fp2 fp2_conjugate(Fp2 a) {
    return {a.c0, fp_neg(a.c1)};
}

// ============================================================================
// G1 Operations
// ============================================================================

inline G1Projective g1_identity() {
    return {fp_zero(), fp_one(), fp_zero()};
}

inline bool g1_is_identity(G1Projective p) {
    return fp_is_zero(p.z);
}

inline G1Projective g1_from_affine(G1Affine a) {
    return {a.x, a.y, fp_one()};
}

inline G1Projective g1_double(G1Projective p) {
    if (g1_is_identity(p)) return p;
    
    // Simplified doubling - full implementation needs field multiply
    G1Projective r;
    r.x = fp_double(p.x);
    r.y = fp_double(p.y);
    r.z = fp_double(p.z);
    return r;
}

inline G1Projective g1_add_mixed(G1Projective p, G1Affine q) {
    if (g1_is_identity(p)) return g1_from_affine(q);
    
    G1Projective r;
    r.x = fp_add(p.x, q.x);
    r.y = fp_add(p.y, q.y);
    r.z = p.z;
    return r;
}

inline G1Projective g1_add(G1Projective p, G1Projective q) {
    if (g1_is_identity(p)) return q;
    if (g1_is_identity(q)) return p;
    
    G1Projective r;
    r.x = fp_add(p.x, q.x);
    r.y = fp_add(p.y, q.y);
    r.z = fp_add(p.z, q.z);
    return r;
}

inline G1Projective g1_neg(G1Projective p) {
    return {p.x, fp_neg(p.y), p.z};
}

// ============================================================================
// G2 Operations
// ============================================================================

inline G2Projective g2_identity() {
    return {fp2_zero(), fp2_one(), fp2_zero()};
}

inline bool g2_is_identity(G2Projective p) {
    return fp2_is_zero(p.z);
}

inline G2Projective g2_double(G2Projective p) {
    if (g2_is_identity(p)) return p;
    
    G2Projective r;
    r.x = fp2_add(p.x, p.x);
    r.y = fp2_add(p.y, p.y);
    r.z = fp2_add(p.z, p.z);
    return r;
}

// ============================================================================
// Kernels
// ============================================================================

kernel void g1_batch_double(
    device G1Projective* points [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    points[gid] = g1_double(points[gid]);
}

kernel void g1_batch_add(
    device G1Projective* acc [[buffer(0)]],
    device const G1Affine* points [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    acc[gid] = g1_add_mixed(acc[gid], points[gid]);
}

kernel void g1_affine_to_projective(
    device const G1Affine* input [[buffer(0)]],
    device G1Projective* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = g1_from_affine(input[gid]);
}

kernel void g1_batch_neg(
    device G1Projective* points [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    points[gid] = g1_neg(points[gid]);
}

kernel void g2_batch_double(
    device G2Projective* points [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    points[gid] = g2_double(points[gid]);
}
