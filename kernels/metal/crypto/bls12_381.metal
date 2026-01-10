// =============================================================================
// BLS12-381 Metal Compute Shaders
// =============================================================================
//
// GPU-accelerated elliptic curve operations for BLS12-381 on Apple Silicon.
// Implements G1 point operations for batch signature verification.
//
// BLS12-381 Parameters:
//   p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
//   r = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
//   G1: y^2 = x^3 + 4  over Fp
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// 384-bit Field Arithmetic (6 x 64-bit limbs)
// =============================================================================

// BLS12-381 base field prime p (6 limbs, little-endian)
constant uint64_t BLS_P[6] = {
    0xb9feffffffffaaab,
    0x1eabfffeb153ffff,
    0x6730d2a0f6b0f624,
    0x64774b84f38512bf,
    0x4b1ba7b6434bacd7,
    0x1a0111ea397fe69a
};

// Montgomery R^2 mod p (for converting to Montgomery form)
constant uint64_t BLS_R2[6] = {
    0xf4df1f341c341746,
    0x0a76e6a609d104f1,
    0x8de5476c4c95b6d5,
    0x67eb88a9939d83c0,
    0x9a793e85b519952d,
    0x11988fe592cae3aa
};

// Montgomery constant: -p^{-1} mod 2^64
constant uint64_t BLS_INV = 0x89f3fffcfffcfffd;

// Fp384 represented as 6 uint64 limbs
struct Fp384 {
    uint64_t limbs[6];
};

// G1 affine point
struct G1Affine {
    Fp384 x;
    Fp384 y;
    bool infinity;
};

// G1 projective point (Jacobian coordinates)
struct G1Projective {
    Fp384 x;
    Fp384 y;
    Fp384 z;
};

// =============================================================================
// Multi-precision Arithmetic
// =============================================================================

// Add with carry
inline uint64_t adc(uint64_t a, uint64_t b, thread uint64_t& carry) {
    uint64_t sum = a + b + carry;
    carry = (sum < a) || ((sum == a) && (b > 0 || carry > 0)) ? 1 : 0;
    // Simplified carry detection
    if (carry == 0) {
        if (sum < a || sum < b) carry = 1;
    }
    return sum;
}

// Subtract with borrow
inline uint64_t sbb(uint64_t a, uint64_t b, thread uint64_t& borrow) {
    uint64_t diff = a - b - borrow;
    borrow = (a < b + borrow) ? 1 : 0;
    return diff;
}

// 64x64 -> 128 bit multiplication (returns low and high parts)
inline void mul64(uint64_t a, uint64_t b, thread uint64_t& lo, thread uint64_t& hi) {
    lo = a * b;
    hi = mulhi(a, b);
}

// Compare: returns -1 if a < b, 0 if a == b, 1 if a > b
inline int fp384_cmp(thread const Fp384& a, constant uint64_t* b) {
    for (int i = 5; i >= 0; i--) {
        if (a.limbs[i] < b[i]) return -1;
        if (a.limbs[i] > b[i]) return 1;
    }
    return 0;
}

// Conditional subtraction: if a >= p, compute a - p
inline void fp384_reduce(thread Fp384& a) {
    if (fp384_cmp(a, BLS_P) >= 0) {
        uint64_t borrow = 0;
        for (int i = 0; i < 6; i++) {
            a.limbs[i] = sbb(a.limbs[i], BLS_P[i], borrow);
        }
    }
}

// Fp addition: c = a + b mod p
inline Fp384 fp384_add(thread const Fp384& a, thread const Fp384& b) {
    Fp384 c;
    uint64_t carry = 0;

    for (int i = 0; i < 6; i++) {
        uint64_t sum = a.limbs[i] + b.limbs[i] + carry;
        carry = (sum < a.limbs[i]) ? 1 : 0;
        c.limbs[i] = sum;
    }

    fp384_reduce(c);
    return c;
}

// Fp subtraction: c = a - b mod p
inline Fp384 fp384_sub(thread const Fp384& a, thread const Fp384& b) {
    Fp384 c;
    uint64_t borrow = 0;

    for (int i = 0; i < 6; i++) {
        c.limbs[i] = sbb(a.limbs[i], b.limbs[i], borrow);
    }

    // If underflow, add p
    if (borrow) {
        uint64_t carry = 0;
        for (int i = 0; i < 6; i++) {
            uint64_t sum = c.limbs[i] + BLS_P[i] + carry;
            carry = (sum < c.limbs[i]) ? 1 : 0;
            c.limbs[i] = sum;
        }
    }

    return c;
}

// Montgomery reduction: given T < p*R, compute T*R^{-1} mod p
inline Fp384 fp384_mont_reduce(thread uint64_t t[12]) {
    for (int i = 0; i < 6; i++) {
        uint64_t m = t[i] * BLS_INV;
        uint64_t carry = 0;

        for (int j = 0; j < 6; j++) {
            uint64_t lo, hi;
            mul64(m, BLS_P[j], lo, hi);

            uint64_t sum = t[i + j] + lo + carry;
            carry = (sum < lo) ? hi + 1 : hi;
            t[i + j] = sum;
        }

        // Propagate carry
        for (int j = i + 6; j < 12; j++) {
            uint64_t sum = t[j] + carry;
            carry = (sum < carry) ? 1 : 0;
            t[j] = sum;
            if (carry == 0) break;
        }
    }

    Fp384 result;
    for (int i = 0; i < 6; i++) {
        result.limbs[i] = t[i + 6];
    }

    fp384_reduce(result);
    return result;
}

// Montgomery multiplication: c = a * b * R^{-1} mod p
inline Fp384 fp384_mul(thread const Fp384& a, thread const Fp384& b) {
    uint64_t t[12] = {0};

    // Schoolbook multiplication
    for (int i = 0; i < 6; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 6; j++) {
            uint64_t lo, hi;
            mul64(a.limbs[i], b.limbs[j], lo, hi);

            uint64_t sum = t[i + j] + lo + carry;
            carry = (sum < lo) ? hi + 1 : hi;
            t[i + j] = sum;
        }
        t[i + 6] += carry;
    }

    return fp384_mont_reduce(t);
}

// Montgomery squaring (optimized)
inline Fp384 fp384_sqr(thread const Fp384& a) {
    return fp384_mul(a, a);
}

// Double: c = 2 * a mod p
inline Fp384 fp384_double(thread const Fp384& a) {
    Fp384 c;
    uint64_t carry = 0;

    for (int i = 0; i < 6; i++) {
        uint64_t sum = (a.limbs[i] << 1) | carry;
        carry = a.limbs[i] >> 63;
        c.limbs[i] = sum;
    }

    fp384_reduce(c);
    return c;
}

// Negate: c = -a mod p = p - a
inline Fp384 fp384_neg(thread const Fp384& a) {
    bool is_zero = true;
    for (int i = 0; i < 6; i++) {
        if (a.limbs[i] != 0) { is_zero = false; break; }
    }

    if (is_zero) return a;

    Fp384 c;
    uint64_t borrow = 0;
    for (int i = 0; i < 6; i++) {
        c.limbs[i] = sbb(BLS_P[i], a.limbs[i], borrow);
    }

    return c;
}

// Check if zero
inline bool fp384_is_zero(thread const Fp384& a) {
    for (int i = 0; i < 6; i++) {
        if (a.limbs[i] != 0) return false;
    }
    return true;
}

// =============================================================================
// G1 Point Arithmetic (Jacobian Projective Coordinates)
// =============================================================================

// Point at infinity (identity element)
inline G1Projective g1_identity() {
    G1Projective p;
    for (int i = 0; i < 6; i++) {
        p.x.limbs[i] = 0;
        p.y.limbs[i] = i == 0 ? 1 : 0;  // y = 1 in Montgomery form
        p.z.limbs[i] = 0;
    }
    return p;
}

// Check if point is at infinity (Z == 0)
inline bool g1_is_identity(thread const G1Projective& p) {
    return fp384_is_zero(p.z);
}

// Point doubling in Jacobian coordinates
// Formula: http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
inline G1Projective g1_double(thread const G1Projective& p) {
    if (g1_is_identity(p)) {
        return p;
    }

    // A = X1^2
    Fp384 a = fp384_sqr(p.x);
    // B = Y1^2
    Fp384 b = fp384_sqr(p.y);
    // C = B^2
    Fp384 c = fp384_sqr(b);

    // D = 2*((X1+B)^2 - A - C)
    Fp384 xb = fp384_add(p.x, b);
    Fp384 xb2 = fp384_sqr(xb);
    Fp384 d = fp384_sub(xb2, a);
    d = fp384_sub(d, c);
    d = fp384_double(d);

    // E = 3*A
    Fp384 e = fp384_add(a, a);
    e = fp384_add(e, a);

    // F = E^2
    Fp384 f = fp384_sqr(e);

    // X3 = F - 2*D
    Fp384 d2 = fp384_double(d);
    G1Projective result;
    result.x = fp384_sub(f, d2);

    // Y3 = E*(D - X3) - 8*C
    Fp384 dx3 = fp384_sub(d, result.x);
    Fp384 edx3 = fp384_mul(e, dx3);
    Fp384 c8 = fp384_double(c);
    c8 = fp384_double(c8);
    c8 = fp384_double(c8);
    result.y = fp384_sub(edx3, c8);

    // Z3 = 2*Y1*Z1
    Fp384 yz = fp384_mul(p.y, p.z);
    result.z = fp384_double(yz);

    return result;
}

// Mixed addition: R = P + Q where Q is affine (Z_Q = 1)
// More efficient when one point is in affine form
inline G1Projective g1_add_mixed(thread const G1Projective& p, thread const G1Affine& q) {
    if (q.infinity) return p;
    if (g1_is_identity(p)) {
        G1Projective r;
        r.x = q.x;
        r.y = q.y;
        // Z = 1 in Montgomery form
        for (int i = 0; i < 6; i++) r.z.limbs[i] = BLS_R2[i];  // R mod p
        return r;
    }

    // Z1Z1 = Z1^2
    Fp384 z1z1 = fp384_sqr(p.z);

    // U2 = X2*Z1Z1
    Fp384 u2 = fp384_mul(q.x, z1z1);

    // S2 = Y2*Z1*Z1Z1
    Fp384 s2 = fp384_mul(p.z, z1z1);
    s2 = fp384_mul(q.y, s2);

    // H = U2 - X1
    Fp384 h = fp384_sub(u2, p.x);

    // HH = H^2
    Fp384 hh = fp384_sqr(h);

    // I = 4*HH
    Fp384 i = fp384_double(hh);
    i = fp384_double(i);

    // J = H*I
    Fp384 j = fp384_mul(h, i);

    // r = 2*(S2 - Y1)
    Fp384 r = fp384_sub(s2, p.y);
    r = fp384_double(r);

    // V = X1*I
    Fp384 v = fp384_mul(p.x, i);

    // X3 = r^2 - J - 2*V
    Fp384 r2 = fp384_sqr(r);
    Fp384 v2 = fp384_double(v);
    G1Projective result;
    result.x = fp384_sub(r2, j);
    result.x = fp384_sub(result.x, v2);

    // Y3 = r*(V - X3) - 2*Y1*J
    Fp384 vx3 = fp384_sub(v, result.x);
    Fp384 rvx3 = fp384_mul(r, vx3);
    Fp384 y1j = fp384_mul(p.y, j);
    y1j = fp384_double(y1j);
    result.y = fp384_sub(rvx3, y1j);

    // Z3 = (Z1 + H)^2 - Z1Z1 - HH
    Fp384 zh = fp384_add(p.z, h);
    Fp384 zh2 = fp384_sqr(zh);
    result.z = fp384_sub(zh2, z1z1);
    result.z = fp384_sub(result.z, hh);

    return result;
}

// Copy projective point (for address space conversion)
inline G1Projective g1_copy(constant G1Projective& src) {
    G1Projective dst;
    for (int i = 0; i < 6; i++) {
        dst.x.limbs[i] = src.x.limbs[i];
        dst.y.limbs[i] = src.y.limbs[i];
        dst.z.limbs[i] = src.z.limbs[i];
    }
    return dst;
}

inline G1Projective g1_copy_tg(threadgroup G1Projective& src) {
    G1Projective dst;
    for (int i = 0; i < 6; i++) {
        dst.x.limbs[i] = src.x.limbs[i];
        dst.y.limbs[i] = src.y.limbs[i];
        dst.z.limbs[i] = src.z.limbs[i];
    }
    return dst;
}

// Full point addition: R = P + Q (both projective)
inline G1Projective g1_add(thread const G1Projective& p, thread const G1Projective& q) {
    if (g1_is_identity(p)) return q;
    if (g1_is_identity(q)) return p;

    // Z1Z1 = Z1^2, Z2Z2 = Z2^2
    Fp384 z1z1 = fp384_sqr(p.z);
    Fp384 z2z2 = fp384_sqr(q.z);

    // U1 = X1*Z2Z2, U2 = X2*Z1Z1
    Fp384 u1 = fp384_mul(p.x, z2z2);
    Fp384 u2 = fp384_mul(q.x, z1z1);

    // S1 = Y1*Z2*Z2Z2, S2 = Y2*Z1*Z1Z1
    Fp384 s1 = fp384_mul(p.y, q.z);
    s1 = fp384_mul(s1, z2z2);
    Fp384 s2 = fp384_mul(q.y, p.z);
    s2 = fp384_mul(s2, z1z1);

    // H = U2 - U1
    Fp384 h = fp384_sub(u2, u1);

    // Check if points are equal (H == 0 and S2 - S1 == 0)
    bool h_zero = fp384_is_zero(h);
    Fp384 s_diff = fp384_sub(s2, s1);
    bool s_zero = fp384_is_zero(s_diff);

    if (h_zero && s_zero) {
        return g1_double(p);  // P == Q, use doubling
    }
    if (h_zero) {
        return g1_identity();  // P == -Q, result is identity
    }

    // I = (2*H)^2
    Fp384 h2 = fp384_double(h);
    Fp384 i = fp384_sqr(h2);

    // J = H*I
    Fp384 j = fp384_mul(h, i);

    // r = 2*(S2 - S1)
    Fp384 r = fp384_double(s_diff);

    // V = U1*I
    Fp384 v = fp384_mul(u1, i);

    // X3 = r^2 - J - 2*V
    Fp384 r2 = fp384_sqr(r);
    Fp384 v2 = fp384_double(v);
    G1Projective result;
    result.x = fp384_sub(r2, j);
    result.x = fp384_sub(result.x, v2);

    // Y3 = r*(V - X3) - 2*S1*J
    Fp384 vx3 = fp384_sub(v, result.x);
    Fp384 rvx3 = fp384_mul(r, vx3);
    Fp384 s1j = fp384_mul(s1, j);
    s1j = fp384_double(s1j);
    result.y = fp384_sub(rvx3, s1j);

    // Z3 = ((Z1 + Z2)^2 - Z1Z1 - Z2Z2) * H
    Fp384 z12 = fp384_add(p.z, q.z);
    Fp384 z12_2 = fp384_sqr(z12);
    result.z = fp384_sub(z12_2, z1z1);
    result.z = fp384_sub(result.z, z2z2);
    result.z = fp384_mul(result.z, h);

    return result;
}

// Scalar multiplication using double-and-add
// scalar is 256 bits (4 x 64-bit limbs)
inline G1Projective g1_scalar_mul(thread const G1Projective& p, constant uint64_t* scalar) {
    G1Projective result = g1_identity();
    G1Projective base = p;

    // Process 256 bits
    for (int limb = 0; limb < 4; limb++) {
        uint64_t bits = scalar[limb];
        for (int bit = 0; bit < 64; bit++) {
            if (bits & 1) {
                result = g1_add(result, base);
            }
            base = g1_double(base);
            bits >>= 1;
        }
    }

    return result;
}

// =============================================================================
// Metal Compute Kernels
// =============================================================================

// Batch point addition kernel
// Adds pairs of points in parallel
kernel void g1_batch_add(
    device G1Projective* results [[buffer(0)]],
    constant G1Projective* points_a [[buffer(1)]],
    constant G1Projective* points_b [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;

    G1Projective a = points_a[tid];
    G1Projective b = points_b[tid];
    results[tid] = g1_add(a, b);
}

// Batch point doubling kernel
kernel void g1_batch_double(
    device G1Projective* results [[buffer(0)]],
    constant G1Projective* points [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;

    G1Projective p = points[tid];
    results[tid] = g1_double(p);
}

// Parallel scalar multiplication kernel
// Each thread computes one scalar multiplication
kernel void g1_batch_scalar_mul(
    device G1Projective* results [[buffer(0)]],
    constant G1Projective* points [[buffer(1)]],
    constant uint64_t* scalars [[buffer(2)]],  // 4 limbs per scalar
    constant uint& count [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;

    G1Projective p = points[tid];
    constant uint64_t* scalar = scalars + tid * 4;
    results[tid] = g1_scalar_mul(p, scalar);
}

// Multi-scalar multiplication (MSM) using bucket method
// Suitable for batch signature verification
// This is a simplified version; production would use Pippenger's algorithm
kernel void g1_msm_accumulate(
    device G1Projective* buckets [[buffer(0)]],
    constant G1Projective* points [[buffer(1)]],
    constant uint8_t* bucket_indices [[buffer(2)]],  // Which bucket each point goes to
    constant uint& num_points [[buffer(3)]],
    constant uint& num_buckets [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_points) return;

    uint bucket_idx = bucket_indices[tid];
    if (bucket_idx >= num_buckets) return;

    G1Projective p = points[tid];

    // Atomic-style accumulation (simplified - real impl needs proper sync)
    // This accumulates point into the appropriate bucket
    G1Projective current = buckets[bucket_idx];
    buckets[bucket_idx] = g1_add(current, p);
}

// Reduce buckets for MSM final result
kernel void g1_msm_reduce(
    device G1Projective* result [[buffer(0)]],
    constant G1Projective* buckets [[buffer(1)]],
    constant uint& num_buckets [[buffer(2)]],
    constant uint& window_bits [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;  // Single thread for final reduction

    G1Projective sum = g1_identity();
    G1Projective running = g1_identity();

    // Process buckets from highest to lowest
    for (int i = num_buckets - 1; i >= 0; i--) {
        G1Projective bucket = g1_copy(buckets[i]);
        running = g1_add(running, bucket);
        sum = g1_add(sum, running);
    }

    result[0] = sum;
}

// Batch signature verification helper
// Computes: sum_i (r_i * P_i) where P_i are public keys and r_i are random scalars
// Used for verifying aggregate signatures efficiently
kernel void bls_batch_verify_msm(
    device G1Projective* result [[buffer(0)]],
    constant G1Affine* public_keys [[buffer(1)]],
    constant uint64_t* random_scalars [[buffer(2)]],  // 4 limbs per scalar
    constant uint& count [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint threads_per_group [[threads_per_threadgroup]],
    threadgroup G1Projective* shared_mem [[threadgroup(0)]])
{
    G1Projective local_sum = g1_identity();

    // Each thread processes multiple points with stride
    uint total_threads = threads_per_group;
    for (uint i = tid; i < count; i += total_threads) {
        // Convert affine to projective
        G1Projective p;
        p.x = public_keys[i].x;
        p.y = public_keys[i].y;
        for (int j = 0; j < 6; j++) p.z.limbs[j] = BLS_R2[j];  // Z = 1

        if (public_keys[i].infinity) {
            p = g1_identity();
        }

        // Scalar multiplication
        constant uint64_t* scalar = random_scalars + i * 4;
        G1Projective scaled = g1_scalar_mul(p, scalar);

        // Accumulate
        local_sum = g1_add(local_sum, scaled);
    }

    // Store to shared memory
    shared_mem[tid] = local_sum;

    // Barrier
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = threads_per_group / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            G1Projective a = g1_copy_tg(shared_mem[tid]);
            G1Projective b = g1_copy_tg(shared_mem[tid + stride]);
            shared_mem[tid] = g1_add(a, b);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes final result
    if (tid == 0) {
        result[0] = g1_copy_tg(shared_mem[0]);
    }
}
