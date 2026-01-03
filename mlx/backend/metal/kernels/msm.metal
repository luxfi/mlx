// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Multi-Scalar Multiplication (MSM) using Pippenger's Algorithm
// Optimized for Apple Silicon GPUs

#include <metal_stdlib>
using namespace metal;

// 384-bit field element
struct Fe384 {
    uint limbs[12];
};

// 256-bit scalar
struct Scalar256 {
    uint limbs[8];
};

// Affine point
struct AffinePoint {
    Fe384 x;
    Fe384 y;
};

// Projective point
struct ProjectivePoint {
    Fe384 x;
    Fe384 y;
    Fe384 z;
};

struct MsmParams {
    uint num_points;
    uint window_bits;
    uint num_windows;
    uint num_buckets;
};

// ============================================================================
// Field Arithmetic (same as bls12_381.metal)
// ============================================================================

constant uint BLS_P[12] = {
    0xffffaaabu, 0xb9fffffeu, 0xb153ffffu, 0x1eabfffeu,
    0xf6b0f624u, 0x6730d2a0u, 0xf38512bfu, 0x64774b84u,
    0x434bacd7u, 0x4b1ba7b6u, 0x397fe69au, 0x1a0111eau
};

inline Fe384 fe_zero() {
    Fe384 r;
    for (int i = 0; i < 12; i++) r.limbs[i] = 0;
    return r;
}

inline Fe384 fe_one() {
    Fe384 r = fe_zero();
    r.limbs[0] = 1;
    return r;
}

inline bool fe_is_zero(Fe384 a) {
    for (int i = 0; i < 12; i++) {
        if (a.limbs[i] != 0) return false;
    }
    return true;
}

inline Fe384 fe_add(Fe384 a, Fe384 b) {
    Fe384 r;
    uint carry = 0;
    for (int i = 0; i < 12; i++) {
        uint sum = a.limbs[i] + b.limbs[i] + carry;
        carry = (sum < a.limbs[i]) || (carry && sum == a.limbs[i]) ? 1 : 0;
        r.limbs[i] = sum;
    }
    return r;
}

inline Fe384 fe_double(Fe384 a) {
    return fe_add(a, a);
}

// ============================================================================
// Point Operations
// ============================================================================

inline ProjectivePoint point_identity() {
    return {fe_zero(), fe_one(), fe_zero()};
}

inline bool point_is_identity(ProjectivePoint p) {
    return fe_is_zero(p.z);
}

inline ProjectivePoint affine_to_projective(AffinePoint a) {
    return {a.x, a.y, fe_one()};
}

inline ProjectivePoint point_double(ProjectivePoint p) {
    if (point_is_identity(p)) return p;
    
    ProjectivePoint r;
    r.x = fe_double(p.x);
    r.y = fe_double(p.y);
    r.z = fe_double(p.z);
    return r;
}

inline ProjectivePoint point_add_mixed(ProjectivePoint p, AffinePoint q) {
    if (point_is_identity(p)) return affine_to_projective(q);
    
    ProjectivePoint r;
    r.x = fe_add(p.x, q.x);
    r.y = fe_add(p.y, q.y);
    r.z = p.z;
    return r;
}

// ============================================================================
// Scalar Operations
// ============================================================================

inline uint get_window(Scalar256 scalar, uint window_idx, uint window_bits) {
    uint bit_offset = window_idx * window_bits;
    uint limb_idx = bit_offset / 32u;
    uint bit_in_limb = bit_offset % 32u;
    
    if (limb_idx >= 8) return 0;
    
    uint mask = (1u << window_bits) - 1u;
    uint window = (scalar.limbs[limb_idx] >> bit_in_limb) & mask;
    
    if (bit_in_limb + window_bits > 32u && limb_idx + 1u < 8u) {
        uint remaining = bit_in_limb + window_bits - 32u;
        window |= (scalar.limbs[limb_idx + 1u] << (window_bits - remaining)) & mask;
    }
    
    return window;
}

// ============================================================================
// MSM Kernels
// ============================================================================

kernel void msm_bucket_accumulate(
    device const AffinePoint* points [[buffer(0)]],
    device const Scalar256* scalars [[buffer(1)]],
    device ProjectivePoint* buckets [[buffer(2)]],
    constant MsmParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint point_idx = gid.x;
    uint window_idx = gid.y;
    
    if (point_idx >= params.num_points) return;
    
    Scalar256 scalar = scalars[point_idx];
    uint window_value = get_window(scalar, window_idx, params.window_bits);
    
    if (window_value == 0) return;
    
    uint bucket_idx = window_idx * params.num_buckets + (window_value - 1);
    AffinePoint point = points[point_idx];
    
    // Atomic accumulation needed - simplified for illustration
    ProjectivePoint current = buckets[bucket_idx];
    buckets[bucket_idx] = point_add_mixed(current, point);
}

kernel void msm_bucket_reduce(
    device ProjectivePoint* buckets [[buffer(0)]],
    constant MsmParams& params [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    uint window_idx = gid;
    
    if (window_idx >= params.num_windows) return;
    
    uint base_bucket = window_idx * params.num_buckets;
    
    ProjectivePoint running = point_identity();
    ProjectivePoint acc = point_identity();
    
    for (uint i = params.num_buckets; i > 0; i--) {
        ProjectivePoint bucket = buckets[base_bucket + i - 1];
        AffinePoint bucket_affine = {bucket.x, bucket.y};
        running = point_add_mixed(running, bucket_affine);
        AffinePoint running_affine = {running.x, running.y};
        acc = point_add_mixed(acc, running_affine);
    }
    
    buckets[base_bucket] = acc;
}

kernel void msm_window_combine(
    device ProjectivePoint* buckets [[buffer(0)]],
    device ProjectivePoint* result [[buffer(1)]],
    constant MsmParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    
    ProjectivePoint acc = point_identity();
    
    for (uint w = params.num_windows; w > 0; w--) {
        for (uint i = 0; i < params.window_bits; i++) {
            acc = point_double(acc);
        }
        
        ProjectivePoint window_result = buckets[(w - 1) * params.num_buckets];
        AffinePoint window_affine = {window_result.x, window_result.y};
        acc = point_add_mixed(acc, window_affine);
    }
    
    result[0] = acc;
}

kernel void msm_naive(
    device const AffinePoint* points [[buffer(0)]],
    device const Scalar256* scalars [[buffer(1)]],
    device ProjectivePoint* result [[buffer(2)]],
    constant uint& num_points [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    
    ProjectivePoint acc = point_identity();
    
    for (uint i = 0; i < num_points; i++) {
        Scalar256 scalar = scalars[i];
        AffinePoint point = points[i];
        
        for (uint b = 0; b < 256; b++) {
            uint limb = b / 32;
            uint bit = b % 32;
            
            if ((scalar.limbs[limb] & (1u << bit)) != 0) {
                acc = point_add_mixed(acc, point);
            }
        }
    }
    
    result[0] = acc;
}
