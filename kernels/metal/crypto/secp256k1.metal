// =============================================================================
// secp256k1 GPU Kernels for Metal (Apple Silicon)
// =============================================================================
//
// GPU-accelerated secp256k1 operations using the GTable approach.
// Based on CudaBrainSecp optimization for ~20x speedup over double-and-add.
//
// GTable Structure:
// - 16 chunks × 65536 points = 1,048,576 precomputed points (~67MB)
// - Scalar multiplication: 16 table lookups + 15 point additions
// - Perfect for batch operations on Apple Silicon GPU
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// secp256k1 Field Constants
// =============================================================================
//
// Prime: p = 2^256 - 2^32 - 977
// = 0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F
//
// Order: n = 0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141
//
// Generator: G = (Gx, Gy) where:
// Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
// Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

// secp256k1 prime (little-endian limbs)
constant uint64_t SECP256K1_P[4] = {
    0xFFFFFFFEFFFFFC2FULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL
};

// secp256k1 order (little-endian limbs)
constant uint64_t SECP256K1_N[4] = {
    0xBFD25E8CD0364141ULL,
    0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL,
    0xFFFFFFFFFFFFFFFFULL
};

// Generator point Gx (little-endian)
constant uint64_t SECP256K1_GX[4] = {
    0x59F2815B16F81798ULL,
    0x029BFCDB2DCE28D9ULL,
    0x55A06295CE870B07ULL,
    0x79BE667EF9DCBBACULL
};

// Generator point Gy (little-endian)
constant uint64_t SECP256K1_GY[4] = {
    0x9C47D08FFB10D4B8ULL,
    0xFD17B448A6855419ULL,
    0x5DA4FBFC0E1108A8ULL,
    0x483ADA7726A3C465ULL
};

// =============================================================================
// Types
// =============================================================================

struct Fp256 {
    uint64_t limbs[4];
};

struct Scalar256 {
    uint64_t limbs[4];
};

struct AffinePoint {
    Fp256 x;
    Fp256 y;
    bool infinity;
};

struct JacobianPoint {
    Fp256 x;
    Fp256 y;
    Fp256 z;
};

// =============================================================================
// 256-bit Arithmetic Primitives
// =============================================================================

// Add with carry
inline uint64_t adc(uint64_t a, uint64_t b, thread uint64_t& carry) {
    uint64_t sum = a + b + carry;
    carry = (sum < a || (carry && sum == a)) ? 1 : 0;
    return sum;
}

// Subtract with borrow  
inline uint64_t sbb(uint64_t a, uint64_t b, thread uint64_t& borrow) {
    uint64_t diff = a - b - borrow;
    borrow = (a < b + borrow) ? 1 : 0;
    return diff;
}

// 256-bit addition
inline void add256(thread Fp256& c, Fp256 a, Fp256 b) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        c.limbs[i] = adc(a.limbs[i], b.limbs[i], carry);
    }
}

// 256-bit subtraction
inline void sub256(thread Fp256& c, Fp256 a, Fp256 b) {
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        c.limbs[i] = sbb(a.limbs[i], b.limbs[i], borrow);
    }
}

// Compare: return true if a >= b
inline bool gte256(Fp256 a, Fp256 b) {
    for (int i = 3; i >= 0; i--) {
        if (a.limbs[i] > b.limbs[i]) return true;
        if (a.limbs[i] < b.limbs[i]) return false;
    }
    return true;  // Equal
}

// Check if zero
inline bool is_zero256(Fp256 a) {
    return a.limbs[0] == 0 && a.limbs[1] == 0 && 
           a.limbs[2] == 0 && a.limbs[3] == 0;
}

// =============================================================================
// secp256k1 Field Arithmetic
// =============================================================================

// Modular reduction: c = a mod p
// Uses secp256k1's special form: p = 2^256 - 2^32 - 977
inline void fp_reduce(thread Fp256& c, Fp256 a) {
    // Check if a >= p
    Fp256 p = {{SECP256K1_P[0], SECP256K1_P[1], SECP256K1_P[2], SECP256K1_P[3]}};
    if (gte256(a, p)) {
        sub256(c, a, p);
    } else {
        c = a;
    }
}

// Field addition: c = (a + b) mod p
inline void fp_add(thread Fp256& c, Fp256 a, Fp256 b) {
    add256(c, a, b);
    Fp256 p = {{SECP256K1_P[0], SECP256K1_P[1], SECP256K1_P[2], SECP256K1_P[3]}};
    if (gte256(c, p)) {
        sub256(c, c, p);
    }
}

// Field subtraction: c = (a - b) mod p
inline void fp_sub(thread Fp256& c, Fp256 a, Fp256 b) {
    if (gte256(a, b)) {
        sub256(c, a, b);
    } else {
        Fp256 p = {{SECP256K1_P[0], SECP256K1_P[1], SECP256K1_P[2], SECP256K1_P[3]}};
        add256(c, a, p);
        sub256(c, c, b);
    }
}

// Field negation: c = -a mod p
inline void fp_neg(thread Fp256& c, Fp256 a) {
    if (is_zero256(a)) {
        c = a;
    } else {
        Fp256 p = {{SECP256K1_P[0], SECP256K1_P[1], SECP256K1_P[2], SECP256K1_P[3]}};
        sub256(c, p, a);
    }
}

// Field doubling: c = 2*a mod p (faster than add)
inline void fp_double(thread Fp256& c, Fp256 a) {
    fp_add(c, a, a);
}

// Field multiplication: c = a * b mod p
// Uses schoolbook multiplication with reduction optimized for secp256k1
inline void fp_mul(thread Fp256& c, Fp256 a, Fp256 b) {
    // Full 512-bit product
    uint64_t t[8] = {0};
    
    // Schoolbook multiplication
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            // 64x64 -> 128-bit multiplication
            uint64_t lo = a.limbs[i] * b.limbs[j];
            uint64_t hi = mulhi(a.limbs[i], b.limbs[j]);
            
            uint64_t sum = t[i+j] + lo;
            uint64_t c1 = (sum < t[i+j]) ? 1 : 0;
            sum += carry;
            c1 += (sum < carry) ? 1 : 0;
            t[i+j] = sum;
            carry = hi + c1;
        }
        t[i+4] = carry;
    }
    
    // Reduction using p = 2^256 - 2^32 - 977
    // t mod p = t[0..3] + (t[4..7] * 2^256) mod p
    //         = t[0..3] + t[4..7] * (2^32 + 977) mod p
    
    // First reduction round
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        // Add t[i+4] * (2^32 + 977) to t[i]
        uint64_t lo = t[i+4] * 0x1000003D1ULL;  // 2^32 + 977
        uint64_t hi = mulhi(t[i+4], 0x1000003D1ULL);
        
        uint64_t sum = t[i] + lo + carry;
        carry = (sum < t[i] || (carry && sum == t[i] + lo)) ? 1 : 0;
        carry += hi;
        t[i] = sum;
    }
    
    // Handle remaining carry
    while (carry > 0) {
        uint64_t lo = carry * 0x1000003D1ULL;
        uint64_t hi = mulhi(carry, 0x1000003D1ULL);
        
        uint64_t sum = t[0] + lo;
        uint64_t c1 = (sum < t[0]) ? 1 : 0;
        t[0] = sum;
        
        sum = t[1] + c1;
        c1 = (sum < t[1]) ? 1 : 0;
        t[1] = sum;
        
        sum = t[2] + c1;
        c1 = (sum < t[2]) ? 1 : 0;
        t[2] = sum;
        
        sum = t[3] + c1;
        c1 = (sum < t[3]) ? 1 : 0;
        t[3] = sum;
        
        carry = hi + c1;
    }
    
    // Final reduction if needed
    c.limbs[0] = t[0];
    c.limbs[1] = t[1];
    c.limbs[2] = t[2];
    c.limbs[3] = t[3];
    fp_reduce(c, c);
}

// Field squaring: c = a^2 mod p (slightly faster than mul)
inline void fp_sqr(thread Fp256& c, Fp256 a) {
    fp_mul(c, a, a);  // TODO: Optimize with dedicated squaring
}

// Field inversion: c = a^(-1) mod p using Fermat's little theorem
// a^(-1) = a^(p-2) mod p
inline void fp_inv(thread Fp256& c, Fp256 a) {
    // Exponent: p - 2 = 0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF 
    //                   FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2D
    
    Fp256 result = {{1, 0, 0, 0}};
    Fp256 base = a;
    
    // Binary exponentiation
    // Most bits are 1, so we optimize for that
    for (int limb = 0; limb < 4; limb++) {
        uint64_t exp_limb;
        if (limb == 0) {
            exp_limb = 0xFFFFFFFEFFFFFC2DULL;  // p[0] - 2
        } else {
            exp_limb = 0xFFFFFFFFFFFFFFFFULL;
        }
        
        for (int bit = 0; bit < 64; bit++) {
            if ((exp_limb >> bit) & 1) {
                fp_mul(result, result, base);
            }
            fp_sqr(base, base);
        }
    }
    
    c = result;
}

// =============================================================================
// Point Operations
// =============================================================================

// Convert Jacobian to Affine: (X, Y, Z) -> (X/Z^2, Y/Z^3)
inline AffinePoint jacobian_to_affine(JacobianPoint p) {
    if (is_zero256(p.z)) {
        AffinePoint inf;
        inf.infinity = true;
        return inf;
    }
    
    Fp256 z_inv, z_inv2, z_inv3;
    fp_inv(z_inv, p.z);
    fp_sqr(z_inv2, z_inv);
    fp_mul(z_inv3, z_inv2, z_inv);
    
    AffinePoint result;
    fp_mul(result.x, p.x, z_inv2);
    fp_mul(result.y, p.y, z_inv3);
    result.infinity = false;
    
    return result;
}

// Convert Affine to Jacobian
inline JacobianPoint affine_to_jacobian(AffinePoint p) {
    JacobianPoint result;
    if (p.infinity) {
        result.x = {{0, 0, 0, 0}};
        result.y = {{1, 0, 0, 0}};
        result.z = {{0, 0, 0, 0}};
    } else {
        result.x = p.x;
        result.y = p.y;
        result.z = {{1, 0, 0, 0}};
    }
    return result;
}

// Point doubling in Jacobian coordinates
// Formula from https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html
// secp256k1 has a = 0, so we use optimized formula
inline JacobianPoint point_double(JacobianPoint p) {
    if (is_zero256(p.z)) {
        return p;  // Point at infinity
    }
    
    Fp256 s, m, t, x3, y3, z3;
    Fp256 yy, yyyy, xx;
    
    // YY = Y^2
    fp_sqr(yy, p.y);
    
    // S = 4*X*YY
    fp_mul(s, p.x, yy);
    fp_double(s, s);
    fp_double(s, s);
    
    // M = 3*X^2 (a=0 for secp256k1)
    fp_sqr(xx, p.x);
    fp_add(m, xx, xx);
    fp_add(m, m, xx);
    
    // T = M^2 - 2*S
    fp_sqr(t, m);
    Fp256 two_s;
    fp_double(two_s, s);
    fp_sub(t, t, two_s);
    
    // X3 = T
    x3 = t;
    
    // Y3 = M*(S-T) - 8*YYYY
    fp_sqr(yyyy, yy);
    Fp256 eight_yyyy;
    fp_double(eight_yyyy, yyyy);
    fp_double(eight_yyyy, eight_yyyy);
    fp_double(eight_yyyy, eight_yyyy);
    
    Fp256 s_minus_t;
    fp_sub(s_minus_t, s, t);
    fp_mul(y3, m, s_minus_t);
    fp_sub(y3, y3, eight_yyyy);
    
    // Z3 = 2*Y*Z
    fp_mul(z3, p.y, p.z);
    fp_double(z3, z3);
    
    JacobianPoint result;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return result;
}

// Point addition: Jacobian + Affine -> Jacobian (mixed addition)
// More efficient when one point is affine
inline JacobianPoint point_add_mixed(JacobianPoint p, AffinePoint q) {
    if (q.infinity) {
        return p;
    }
    if (is_zero256(p.z)) {
        return affine_to_jacobian(q);
    }
    
    Fp256 z1z1, u2, s2, h, hh, hhh, r, v;
    
    // Z1Z1 = Z1^2
    fp_sqr(z1z1, p.z);
    
    // U2 = X2 * Z1Z1
    fp_mul(u2, q.x, z1z1);
    
    // S2 = Y2 * Z1 * Z1Z1
    Fp256 z1_z1z1;
    fp_mul(z1_z1z1, p.z, z1z1);
    fp_mul(s2, q.y, z1_z1z1);
    
    // H = U2 - X1
    fp_sub(h, u2, p.x);
    
    // HH = H^2
    fp_sqr(hh, h);
    
    // HHH = H * HH
    fp_mul(hhh, h, hh);
    
    // R = S2 - Y1
    fp_sub(r, s2, p.y);
    
    // V = X1 * HH
    fp_mul(v, p.x, hh);
    
    // X3 = R^2 - HHH - 2*V
    Fp256 x3, y3, z3;
    Fp256 rr, two_v;
    fp_sqr(rr, r);
    fp_double(two_v, v);
    fp_sub(x3, rr, hhh);
    fp_sub(x3, x3, two_v);
    
    // Y3 = R*(V - X3) - Y1*HHH
    Fp256 v_minus_x3, y1_hhh;
    fp_sub(v_minus_x3, v, x3);
    fp_mul(y3, r, v_minus_x3);
    fp_mul(y1_hhh, p.y, hhh);
    fp_sub(y3, y3, y1_hhh);
    
    // Z3 = Z1 * H
    fp_mul(z3, p.z, h);
    
    JacobianPoint result;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return result;
}

// Full Jacobian + Jacobian addition
inline JacobianPoint point_add(JacobianPoint p, JacobianPoint q) {
    if (is_zero256(p.z)) return q;
    if (is_zero256(q.z)) return p;
    
    Fp256 z1z1, z2z2, u1, u2, s1, s2, h, r;
    
    fp_sqr(z1z1, p.z);
    fp_sqr(z2z2, q.z);
    
    fp_mul(u1, p.x, z2z2);
    fp_mul(u2, q.x, z1z1);
    
    Fp256 z1_z1z1, z2_z2z2;
    fp_mul(z1_z1z1, p.z, z1z1);
    fp_mul(z2_z2z2, q.z, z2z2);
    
    fp_mul(s1, p.y, z2_z2z2);
    fp_mul(s2, q.y, z1_z1z1);
    
    fp_sub(h, u2, u1);
    fp_sub(r, s2, s1);
    
    // Check if P = Q (need doubling) or P = -Q (result is infinity)
    if (is_zero256(h)) {
        if (is_zero256(r)) {
            return point_double(p);
        }
        // P = -Q, return infinity
        JacobianPoint inf;
        inf.x = {{0, 0, 0, 0}};
        inf.y = {{1, 0, 0, 0}};
        inf.z = {{0, 0, 0, 0}};
        return inf;
    }
    
    Fp256 hh, hhh, v;
    fp_sqr(hh, h);
    fp_mul(hhh, h, hh);
    fp_mul(v, u1, hh);
    
    Fp256 x3, y3, z3;
    Fp256 rr, two_v;
    fp_sqr(rr, r);
    fp_double(two_v, v);
    fp_sub(x3, rr, hhh);
    fp_sub(x3, x3, two_v);
    
    Fp256 v_minus_x3, s1_hhh;
    fp_sub(v_minus_x3, v, x3);
    fp_mul(y3, r, v_minus_x3);
    fp_mul(s1_hhh, s1, hhh);
    fp_sub(y3, y3, s1_hhh);
    
    Fp256 z1_z2;
    fp_mul(z1_z2, p.z, q.z);
    fp_mul(z3, z1_z2, h);
    
    JacobianPoint result;
    result.x = x3;
    result.y = y3;
    result.z = z3;
    return result;
}

// =============================================================================
// GTable-based Scalar Multiplication
// =============================================================================
//
// The GTable stores precomputed multiples of G:
// table[i][j] = (j * 2^(16*i)) * G  for i in 0..15, j in 0..65535
//
// To compute k * G for 256-bit scalar k:
// 1. Split k into 16 chunks of 16 bits each: k = k[0] + k[1]*2^16 + ... + k[15]*2^240
// 2. Look up table[i][k[i]] for each chunk
// 3. Sum the 16 looked-up points
//
// This requires 16 lookups + 15 additions, vs 256 doublings + ~128 additions for double-and-add

// Extract 16-bit chunk from scalar
inline uint32_t get_scalar_chunk(Scalar256 s, uint32_t chunk_idx) {
    uint32_t bit_idx = chunk_idx * 16;
    uint32_t limb_idx = bit_idx / 64;
    uint32_t bit_offset = bit_idx % 64;
    
    uint64_t value = s.limbs[limb_idx] >> bit_offset;
    
    // Handle crossing limb boundary
    if (bit_offset > 48 && limb_idx < 3) {
        value |= s.limbs[limb_idx + 1] << (64 - bit_offset);
    }
    
    return value & 0xFFFF;
}

// GTable scalar multiplication kernel
// Each thread computes one scalar multiplication using the precomputed table
kernel void gtable_scalar_mul(
    device const AffinePoint* gtable [[buffer(0)]],  // 16 * 65536 precomputed points
    device const Scalar256* scalars [[buffer(1)]],
    device AffinePoint* results [[buffer(2)]],
    constant uint32_t& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    
    Scalar256 scalar = scalars[index];
    
    // Initialize accumulator to identity (will be set on first non-zero chunk)
    JacobianPoint acc;
    acc.x = {{0, 0, 0, 0}};
    acc.y = {{1, 0, 0, 0}};
    acc.z = {{0, 0, 0, 0}};
    bool started = false;
    
    // Process all 16 chunks
    for (uint32_t i = 0; i < 16; i++) {
        uint32_t chunk = get_scalar_chunk(scalar, i);
        
        if (chunk != 0) {
            // Look up point from table
            uint32_t table_idx = i * 65536 + (chunk - 1);  // -1 because we skip 0
            AffinePoint p = gtable[table_idx];
            
            if (started) {
                acc = point_add_mixed(acc, p);
            } else {
                acc = affine_to_jacobian(p);
                started = true;
            }
        }
    }
    
    // Convert to affine for output
    if (started) {
        results[index] = jacobian_to_affine(acc);
    } else {
        results[index].infinity = true;
    }
}

// =============================================================================
// Double-and-Add Scalar Multiplication (for arbitrary base points)
// =============================================================================

kernel void scalar_mul_general(
    device const AffinePoint* points [[buffer(0)]],
    device const Scalar256* scalars [[buffer(1)]],
    device AffinePoint* results [[buffer(2)]],
    constant uint32_t& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    
    AffinePoint base = points[index];
    Scalar256 scalar = scalars[index];
    
    if (base.infinity) {
        results[index].infinity = true;
        return;
    }
    
    JacobianPoint acc;
    acc.x = {{0, 0, 0, 0}};
    acc.y = {{1, 0, 0, 0}};
    acc.z = {{0, 0, 0, 0}};
    bool started = false;
    
    // Double-and-add from MSB
    for (int limb = 3; limb >= 0; limb--) {
        for (int bit = 63; bit >= 0; bit--) {
            if (started) {
                acc = point_double(acc);
            }
            
            if ((scalar.limbs[limb] >> bit) & 1) {
                if (started) {
                    acc = point_add_mixed(acc, base);
                } else {
                    acc = affine_to_jacobian(base);
                    started = true;
                }
            }
        }
    }
    
    if (started) {
        results[index] = jacobian_to_affine(acc);
    } else {
        results[index].infinity = true;
    }
}

// =============================================================================
// Point Addition Batch Kernel
// =============================================================================

kernel void point_add_batch(
    device const AffinePoint* a [[buffer(0)]],
    device const AffinePoint* b [[buffer(1)]],
    device AffinePoint* results [[buffer(2)]],
    constant uint32_t& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    
    AffinePoint pa = a[index];
    AffinePoint pb = b[index];
    
    if (pa.infinity) {
        results[index] = pb;
        return;
    }
    if (pb.infinity) {
        results[index] = pa;
        return;
    }
    
    JacobianPoint ja = affine_to_jacobian(pa);
    JacobianPoint sum = point_add_mixed(ja, pb);
    results[index] = jacobian_to_affine(sum);
}

// =============================================================================
// Point Doubling Batch Kernel
// =============================================================================

kernel void point_double_batch(
    device const AffinePoint* points [[buffer(0)]],
    device AffinePoint* results [[buffer(1)]],
    constant uint32_t& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    
    AffinePoint p = points[index];
    
    if (p.infinity) {
        results[index] = p;
        return;
    }
    
    JacobianPoint jp = affine_to_jacobian(p);
    JacobianPoint doubled = point_double(jp);
    results[index] = jacobian_to_affine(doubled);
}

// =============================================================================
// ECDSA Verification Kernel
// =============================================================================
//
// Verifies signature (r, s) on message hash z with public key Q:
// 1. Compute u1 = z * s^(-1) mod n
// 2. Compute u2 = r * s^(-1) mod n
// 3. Compute R = u1*G + u2*Q
// 4. Signature is valid if R.x mod n == r

kernel void ecdsa_verify_batch(
    device const uint8_t* messages [[buffer(0)]],  // 32 bytes each
    device const Scalar256* r_values [[buffer(1)]],
    device const Scalar256* s_values [[buffer(2)]],
    device const AffinePoint* public_keys [[buffer(3)]],
    device const AffinePoint* gtable [[buffer(4)]],  // For u1*G
    device int* results [[buffer(5)]],
    constant uint32_t& count [[buffer(6)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    
    // Load inputs
    Scalar256 r = r_values[index];
    Scalar256 s = s_values[index];
    AffinePoint Q = public_keys[index];
    
    // Load message hash as scalar z
    Scalar256 z;
    device const uint8_t* msg = messages + index * 32;
    // Convert bytes to scalar (big-endian to little-endian)
    for (int i = 0; i < 4; i++) {
        z.limbs[3-i] = 0;
        for (int j = 0; j < 8; j++) {
            z.limbs[3-i] = (z.limbs[3-i] << 8) | msg[i*8 + j];
        }
    }
    
    // TODO: Implement full ECDSA verification
    // This requires modular inversion in the scalar field (mod n)
    // and the two scalar multiplications u1*G and u2*Q
    
    // Placeholder: mark as valid for now (actual implementation would compute R.x)
    results[index] = 1;
}

// =============================================================================
// GTable Precomputation Kernel
// =============================================================================
//
// Computes the GTable for generator point G:
// table[i][j] = (j * 2^(16*i)) * G
//
// This is run once at initialization (~67MB output)

kernel void precompute_gtable(
    device AffinePoint* gtable [[buffer(0)]],
    constant AffinePoint& generator [[buffer(1)]],
    constant uint32_t& chunk_idx [[buffer(2)]],  // Which 16-bit chunk we're computing
    uint j [[thread_position_in_grid]]
) {
    if (j == 0 || j >= 65536) return;  // Skip j=0 (identity)
    
    // Compute scalar = j * 2^(16 * chunk_idx)
    Scalar256 scalar = {{0, 0, 0, 0}};
    uint32_t bit_offset = chunk_idx * 16;
    uint32_t limb_idx = bit_offset / 64;
    uint32_t limb_offset = bit_offset % 64;
    
    scalar.limbs[limb_idx] = (uint64_t)j << limb_offset;
    if (limb_offset > 48 && limb_idx < 3) {
        scalar.limbs[limb_idx + 1] = (uint64_t)j >> (64 - limb_offset);
    }
    
    // Compute scalar * G using double-and-add
    AffinePoint base = generator;
    JacobianPoint acc;
    acc.x = {{0, 0, 0, 0}};
    acc.y = {{1, 0, 0, 0}};
    acc.z = {{0, 0, 0, 0}};
    bool started = false;
    
    for (int limb = 3; limb >= 0; limb--) {
        for (int bit = 63; bit >= 0; bit--) {
            if (started) {
                acc = point_double(acc);
            }
            if ((scalar.limbs[limb] >> bit) & 1) {
                if (started) {
                    acc = point_add_mixed(acc, base);
                } else {
                    acc = affine_to_jacobian(base);
                    started = true;
                }
            }
        }
    }
    
    // Store in table (j-1 because we skip 0)
    uint32_t table_idx = chunk_idx * 65536 + (j - 1);
    gtable[table_idx] = jacobian_to_affine(acc);
}

// =============================================================================
// Keccak256 for Address Derivation
// =============================================================================

// Keccak-256 round constants
constant uint64_t KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
    0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

// Keccak rotation offsets
constant int KECCAK_R[25] = {
    0, 1, 62, 28, 27, 36, 44, 6, 55, 20,
    3, 10, 43, 25, 39, 41, 45, 15, 21, 8,
    18, 2, 61, 56, 14
};

inline uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

// Single Keccak-f[1600] permutation round
inline void keccak_round(thread uint64_t* state, uint64_t rc) {
    uint64_t C[5], D[5];
    
    // θ step
    for (int x = 0; x < 5; x++) {
        C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
    }
    for (int x = 0; x < 5; x++) {
        D[x] = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
    }
    for (int i = 0; i < 25; i++) {
        state[i] ^= D[i % 5];
    }
    
    // ρ and π steps
    uint64_t temp[25];
    for (int i = 0; i < 25; i++) {
        int x = i % 5;
        int y = i / 5;
        int new_x = y;
        int new_y = (2 * x + 3 * y) % 5;
        temp[new_y * 5 + new_x] = rotl64(state[i], KECCAK_R[i]);
    }
    
    // χ step
    for (int y = 0; y < 5; y++) {
        for (int x = 0; x < 5; x++) {
            int i = y * 5 + x;
            state[i] = temp[i] ^ ((~temp[y * 5 + (x + 1) % 5]) & temp[y * 5 + (x + 2) % 5]);
        }
    }
    
    // ι step
    state[0] ^= rc;
}

// Keccak-256 hash for deriving Ethereum address from public key
kernel void keccak256_batch(
    device const uint8_t* inputs [[buffer(0)]],  // 64 bytes each (uncompressed pubkey without 0x04)
    device uint8_t* outputs [[buffer(1)]],        // 32 bytes each
    constant uint32_t& count [[buffer(2)]],
    constant uint32_t& input_len [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    
    const device uint8_t* input = inputs + index * input_len;
    device uint8_t* output = outputs + index * 32;
    
    // Initialize state
    uint64_t state[25] = {0};
    
    // Absorb input (simplified for 64-byte input)
    // For public key: input is 64 bytes (X || Y coordinates)
    for (uint32_t i = 0; i < input_len && i < 136; i += 8) {
        uint64_t block = 0;
        for (int j = 0; j < 8 && i + j < input_len; j++) {
            block |= (uint64_t)input[i + j] << (j * 8);
        }
        state[i / 8] ^= block;
    }
    
    // Padding (0x01 ... 0x80 for Keccak-256)
    if (input_len < 136) {
        state[input_len / 8] ^= (uint64_t)0x01 << ((input_len % 8) * 8);
        state[16] ^= 0x8000000000000000ULL;  // rate = 136, so last block is index 16
    }
    
    // Keccak-f[1600]
    for (int round = 0; round < 24; round++) {
        keccak_round(state, KECCAK_RC[round]);
    }
    
    // Squeeze output (32 bytes)
    for (int i = 0; i < 4; i++) {
        uint64_t block = state[i];
        for (int j = 0; j < 8; j++) {
            output[i * 8 + j] = (block >> (j * 8)) & 0xFF;
        }
    }
}

// Derive Ethereum address from public key
// Address = keccak256(pubkey)[12:32]
kernel void derive_address_batch(
    device const AffinePoint* public_keys [[buffer(0)]],
    device uint8_t* addresses [[buffer(1)]],  // 20 bytes each
    constant uint32_t& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    
    AffinePoint pk = public_keys[index];
    
    if (pk.infinity) {
        // Invalid public key
        for (int i = 0; i < 20; i++) {
            addresses[index * 20 + i] = 0;
        }
        return;
    }
    
    // Serialize public key (64 bytes: X || Y, big-endian)
    uint8_t pubkey_bytes[64];
    for (int i = 0; i < 4; i++) {
        uint64_t x_limb = pk.x.limbs[3 - i];
        uint64_t y_limb = pk.y.limbs[3 - i];
        for (int j = 0; j < 8; j++) {
            pubkey_bytes[i * 8 + j] = (x_limb >> ((7 - j) * 8)) & 0xFF;
            pubkey_bytes[32 + i * 8 + j] = (y_limb >> ((7 - j) * 8)) & 0xFF;
        }
    }
    
    // Compute Keccak-256
    uint64_t state[25] = {0};
    
    // Absorb 64 bytes
    for (int i = 0; i < 8; i++) {
        uint64_t block = 0;
        for (int j = 0; j < 8; j++) {
            block |= (uint64_t)pubkey_bytes[i * 8 + j] << (j * 8);
        }
        state[i] ^= block;
    }
    
    // Padding
    state[8] ^= 0x01;
    state[16] ^= 0x8000000000000000ULL;
    
    // Keccak-f[1600]
    for (int round = 0; round < 24; round++) {
        keccak_round(state, KECCAK_RC[round]);
    }
    
    // Extract address (last 20 bytes of 32-byte hash)
    // Hash bytes 12-31 become address bytes 0-19
    device uint8_t* addr = addresses + index * 20;
    
    // Bytes 12-15 from state[1]
    addr[0] = (state[1] >> 32) & 0xFF;
    addr[1] = (state[1] >> 40) & 0xFF;
    addr[2] = (state[1] >> 48) & 0xFF;
    addr[3] = (state[1] >> 56) & 0xFF;
    
    // Bytes 16-23 from state[2]
    for (int j = 0; j < 8; j++) {
        addr[4 + j] = (state[2] >> (j * 8)) & 0xFF;
    }
    
    // Bytes 24-31 from state[3]
    for (int j = 0; j < 8; j++) {
        addr[12 + j] = (state[3] >> (j * 8)) & 0xFF;
    }
}
