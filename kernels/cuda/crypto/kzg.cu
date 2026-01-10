// Copyright (c) 2024-2026 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// KZG Polynomial Commitment CUDA Kernels
// GPU-accelerated KZG polynomial commitments for EIP-4844 blobs.
// Uses BLS12-381 curve operations with optimized MSM via Pippenger's algorithm.
//
// KZG Parameters:
//   - Uses BLS12-381 G1/G2 for commitments and proofs
//   - Polynomial degree up to 4096 (blob elements)
//   - Trusted setup from Ethereum KZG ceremony
//
// Key Operations:
//   - Commitment: C = sum(coeffs[i] * g1_generators[i])
//   - Evaluation: p(z) = sum(coeffs[i] * z^i)
//   - Proof: W = sum(quotient[i] * g1_generators[i])
//   - Pairing check: e(C - [y]G1, G2) == e(W, [s-z]G2)
//
// References:
//   - EIP-4844: Shard Blob Transactions
//   - KZG Commitments paper (Kate, Zaverucha, Goldberg 2010)

#include <cstdint>
#include <cuda_runtime.h>

namespace lux {
namespace cuda {
namespace kzg {

// =============================================================================
// BLS12-381 Field Types (384-bit base field, 256-bit scalar field)
// =============================================================================

// Fp384: Base field element (6 x 64-bit limbs)
struct Fp384 {
    uint64_t limbs[6];
};

// Fr256: Scalar field element (4 x 64-bit limbs)
struct Fr256 {
    uint64_t limbs[4];
};

// G1 affine point
struct G1Affine {
    Fp384 x;
    Fp384 y;
    uint32_t infinity;  // Padded for alignment
    uint32_t _pad;
};

// G1 projective point (Jacobian coordinates)
struct G1Projective {
    Fp384 x;
    Fp384 y;
    Fp384 z;
};

// MSM parameters for Pippenger's algorithm
struct MsmParams {
    uint32_t num_points;
    uint32_t window_bits;
    uint32_t num_windows;
    uint32_t num_buckets;
};

// KZG parameters
struct KzgParams {
    uint32_t degree;        // Polynomial degree (max 4096 for blobs)
    uint32_t num_polys;     // Batch size
    uint32_t _pad[2];
};

// =============================================================================
// BLS12-381 Constants
// =============================================================================

// Base field modulus p
__constant__ uint64_t BLS_P[6] = {
    0xb9feffffffffaaabULL,
    0x1eabfffeb153ffffULL,
    0x6730d2a0f6b0f624ULL,
    0x64774b84f38512bfULL,
    0x4b1ba7b6434bacd7ULL,
    0x1a0111ea397fe69aULL
};

// Scalar field modulus r
__constant__ uint64_t BLS_R[4] = {
    0xffffffff00000001ULL,
    0x53bda402fffe5bfeULL,
    0x3339d80809a1d805ULL,
    0x73eda753299d7d48ULL
};

// Montgomery R^2 mod r (for Fr)
__constant__ uint64_t FR_R2[4] = {
    0xc999e990f3f29c6dULL,
    0x2b6cedcb87925c23ULL,
    0x05d314967254398fULL,
    0x0748d9d99f59ff11ULL
};

// Montgomery constant: -r^{-1} mod 2^64
__constant__ uint64_t FR_INV = 0xfffffffeffffffffULL;

// Montgomery R^2 mod p (for Fp)
__constant__ uint64_t FP_R2[6] = {
    0xf4df1f341c341746ULL,
    0x0a76e6a609d104f1ULL,
    0x8de5476c4c95b6d5ULL,
    0x67eb88a9939d83c0ULL,
    0x9a793e85b519952dULL,
    0x11988fe592cae3aaULL
};

// Montgomery constant: -p^{-1} mod 2^64
__constant__ uint64_t FP_INV = 0x89f3fffcfffcfffdULL;

// =============================================================================
// Multi-precision Arithmetic Primitives
// =============================================================================

__device__ __forceinline__
uint64_t adc(uint64_t a, uint64_t b, uint64_t& carry) {
    uint64_t result = a + carry;
    carry = (result < a) ? 1ULL : 0ULL;
    uint64_t sum = result + b;
    carry += (sum < result) ? 1ULL : 0ULL;
    return sum;
}

__device__ __forceinline__
uint64_t sbb(uint64_t a, uint64_t b, uint64_t& borrow) {
    uint64_t diff = a - borrow;
    borrow = (a < borrow) ? 1ULL : 0ULL;
    uint64_t result = diff - b;
    borrow += (diff < b) ? 1ULL : 0ULL;
    return result;
}

// 64x64 -> 128 bit multiply using PTX
__device__ __forceinline__
void mul64(uint64_t a, uint64_t b, uint64_t& lo, uint64_t& hi) {
    asm("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a), "l"(b));
    asm("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a), "l"(b));
}

// =============================================================================
// Fr256 (Scalar Field) Arithmetic
// =============================================================================

__device__ __forceinline__
Fr256 fr_zero() {
    Fr256 r;
    r.limbs[0] = 0; r.limbs[1] = 0;
    r.limbs[2] = 0; r.limbs[3] = 0;
    return r;
}

__device__ __forceinline__
Fr256 fr_one() {
    // Montgomery form of 1
    Fr256 r;
    r.limbs[0] = 0xfffffffe00000001ULL;
    r.limbs[1] = 0x53bda402fffe5bfeULL;
    r.limbs[2] = 0x09a1d80553bda402ULL;
    r.limbs[3] = 0x0000000000000000ULL;
    return r;
}

__device__ __forceinline__
bool fr_is_zero(const Fr256& a) {
    return a.limbs[0] == 0 && a.limbs[1] == 0 &&
           a.limbs[2] == 0 && a.limbs[3] == 0;
}

__device__
Fr256 fr_add(const Fr256& a, const Fr256& b) {
    Fr256 result;
    uint64_t carry = 0;

    result.limbs[0] = adc(a.limbs[0], b.limbs[0], carry);
    result.limbs[1] = adc(a.limbs[1], b.limbs[1], carry);
    result.limbs[2] = adc(a.limbs[2], b.limbs[2], carry);
    result.limbs[3] = adc(a.limbs[3], b.limbs[3], carry);

    // Conditional reduction
    uint64_t borrow = 0;
    Fr256 reduced;
    reduced.limbs[0] = sbb(result.limbs[0], BLS_R[0], borrow);
    reduced.limbs[1] = sbb(result.limbs[1], BLS_R[1], borrow);
    reduced.limbs[2] = sbb(result.limbs[2], BLS_R[2], borrow);
    reduced.limbs[3] = sbb(result.limbs[3], BLS_R[3], borrow);

    if (carry != 0 || borrow == 0) {
        return reduced;
    }
    return result;
}

__device__
Fr256 fr_sub(const Fr256& a, const Fr256& b) {
    Fr256 result;
    uint64_t borrow = 0;

    result.limbs[0] = sbb(a.limbs[0], b.limbs[0], borrow);
    result.limbs[1] = sbb(a.limbs[1], b.limbs[1], borrow);
    result.limbs[2] = sbb(a.limbs[2], b.limbs[2], borrow);
    result.limbs[3] = sbb(a.limbs[3], b.limbs[3], borrow);

    if (borrow != 0) {
        uint64_t carry = 0;
        result.limbs[0] = adc(result.limbs[0], BLS_R[0], carry);
        result.limbs[1] = adc(result.limbs[1], BLS_R[1], carry);
        result.limbs[2] = adc(result.limbs[2], BLS_R[2], carry);
        result.limbs[3] = adc(result.limbs[3], BLS_R[3], carry);
    }

    return result;
}

// Montgomery multiplication for Fr
__device__
Fr256 fr_mont_mul(const Fr256& a, const Fr256& b) {
    uint64_t t[8] = {0};

    // Schoolbook multiplication
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint64_t hi, lo;
            mul64(a.limbs[i], b.limbs[j], lo, hi);

            uint64_t sum = t[i + j] + lo + carry;
            carry = (sum < t[i + j]) ? 1ULL : 0ULL;
            carry += hi;
            t[i + j] = sum;
        }
        t[i + 4] = carry;
    }

    // Montgomery reduction
    for (int i = 0; i < 4; i++) {
        uint64_t m = t[i] * FR_INV;
        uint64_t carry = 0;

        for (int j = 0; j < 4; j++) {
            uint64_t hi, lo;
            mul64(m, BLS_R[j], lo, hi);

            uint64_t sum = t[i + j] + lo + carry;
            carry = (sum < t[i + j]) ? 1ULL : 0ULL;
            carry += hi;
            t[i + j] = sum;
        }

        for (int j = i + 4; j < 8 && carry != 0; j++) {
            uint64_t sum = t[j] + carry;
            carry = (sum < t[j]) ? 1ULL : 0ULL;
            t[j] = sum;
        }
    }

    // Result is in t[4..7]
    Fr256 result;
    result.limbs[0] = t[4];
    result.limbs[1] = t[5];
    result.limbs[2] = t[6];
    result.limbs[3] = t[7];

    // Final reduction
    uint64_t borrow = 0;
    Fr256 reduced;
    reduced.limbs[0] = sbb(result.limbs[0], BLS_R[0], borrow);
    reduced.limbs[1] = sbb(result.limbs[1], BLS_R[1], borrow);
    reduced.limbs[2] = sbb(result.limbs[2], BLS_R[2], borrow);
    reduced.limbs[3] = sbb(result.limbs[3], BLS_R[3], borrow);

    return (borrow == 0) ? reduced : result;
}

__device__ __forceinline__
Fr256 fr_square(const Fr256& a) {
    return fr_mont_mul(a, a);
}

// =============================================================================
// Fp384 (Base Field) Arithmetic
// =============================================================================

__device__ __forceinline__
Fp384 fp_zero() {
    Fp384 r;
    #pragma unroll
    for (int i = 0; i < 6; i++) r.limbs[i] = 0;
    return r;
}

__device__ __forceinline__
Fp384 fp_one() {
    // Montgomery form of 1
    Fp384 r;
    r.limbs[0] = 0x760900000002fffdULL;
    r.limbs[1] = 0xebf4000bc40c0002ULL;
    r.limbs[2] = 0x5f48985753c758baULL;
    r.limbs[3] = 0x77ce585370525745ULL;
    r.limbs[4] = 0x5c071a97a256ec6dULL;
    r.limbs[5] = 0x15f65ec3fa80e493ULL;
    return r;
}

__device__ __forceinline__
bool fp_is_zero(const Fp384& a) {
    for (int i = 0; i < 6; i++) {
        if (a.limbs[i] != 0) return false;
    }
    return true;
}

__device__
Fp384 fp_add(const Fp384& a, const Fp384& b) {
    Fp384 result;
    uint64_t carry = 0;

    #pragma unroll
    for (int i = 0; i < 6; i++) {
        result.limbs[i] = adc(a.limbs[i], b.limbs[i], carry);
    }

    // Conditional reduction
    uint64_t borrow = 0;
    Fp384 reduced;
    #pragma unroll
    for (int i = 0; i < 6; i++) {
        reduced.limbs[i] = sbb(result.limbs[i], BLS_P[i], borrow);
    }

    if (carry != 0 || borrow == 0) {
        return reduced;
    }
    return result;
}

__device__
Fp384 fp_sub(const Fp384& a, const Fp384& b) {
    Fp384 result;
    uint64_t borrow = 0;

    #pragma unroll
    for (int i = 0; i < 6; i++) {
        result.limbs[i] = sbb(a.limbs[i], b.limbs[i], borrow);
    }

    if (borrow != 0) {
        uint64_t carry = 0;
        #pragma unroll
        for (int i = 0; i < 6; i++) {
            result.limbs[i] = adc(result.limbs[i], BLS_P[i], carry);
        }
    }

    return result;
}

__device__ __forceinline__
Fp384 fp_neg(const Fp384& a) {
    if (fp_is_zero(a)) return a;

    Fp384 result;
    uint64_t borrow = 0;
    #pragma unroll
    for (int i = 0; i < 6; i++) {
        result.limbs[i] = sbb(BLS_P[i], a.limbs[i], borrow);
    }
    return result;
}

// Montgomery multiplication for Fp
__device__
Fp384 fp_mont_mul(const Fp384& a, const Fp384& b) {
    uint64_t t[12] = {0};

    // Schoolbook multiplication
    for (int i = 0; i < 6; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 6; j++) {
            uint64_t hi, lo;
            mul64(a.limbs[i], b.limbs[j], lo, hi);

            uint64_t sum = t[i + j] + lo + carry;
            carry = (sum < t[i + j]) ? 1ULL : 0ULL;
            carry += hi;
            t[i + j] = sum;
        }
        t[i + 6] = carry;
    }

    // Montgomery reduction
    for (int i = 0; i < 6; i++) {
        uint64_t m = t[i] * FP_INV;
        uint64_t carry = 0;

        for (int j = 0; j < 6; j++) {
            uint64_t hi, lo;
            mul64(m, BLS_P[j], lo, hi);

            uint64_t sum = t[i + j] + lo + carry;
            carry = (sum < t[i + j]) ? 1ULL : 0ULL;
            carry += hi;
            t[i + j] = sum;
        }

        for (int j = i + 6; j < 12 && carry != 0; j++) {
            uint64_t sum = t[j] + carry;
            carry = (sum < t[j]) ? 1ULL : 0ULL;
            t[j] = sum;
        }
    }

    // Result in t[6..11]
    Fp384 result;
    #pragma unroll
    for (int i = 0; i < 6; i++) {
        result.limbs[i] = t[i + 6];
    }

    // Final reduction
    uint64_t borrow = 0;
    Fp384 reduced;
    #pragma unroll
    for (int i = 0; i < 6; i++) {
        reduced.limbs[i] = sbb(result.limbs[i], BLS_P[i], borrow);
    }

    return (borrow == 0) ? reduced : result;
}

__device__ __forceinline__
Fp384 fp_square(const Fp384& a) {
    return fp_mont_mul(a, a);
}

// =============================================================================
// G1 Point Operations
// =============================================================================

__device__ __forceinline__
G1Projective g1_identity() {
    G1Projective p;
    p.x = fp_zero();
    p.y = fp_one();
    p.z = fp_zero();
    return p;
}

__device__ __forceinline__
bool g1_is_identity(const G1Projective& p) {
    return fp_is_zero(p.z);
}

// Convert affine to projective
__device__ __forceinline__
G1Projective g1_from_affine(const G1Affine& a) {
    G1Projective p;
    if (a.infinity) {
        return g1_identity();
    }
    p.x = a.x;
    p.y = a.y;
    p.z = fp_one();
    return p;
}

// Point doubling (optimized for a=0, b=4 in BLS12-381)
__device__
G1Projective g1_double(const G1Projective& p) {
    if (g1_is_identity(p)) return p;

    // A = X^2
    Fp384 A = fp_square(p.x);
    // B = Y^2
    Fp384 B = fp_square(p.y);
    // C = B^2
    Fp384 C = fp_square(B);

    // D = 2*((X+B)^2 - A - C)
    Fp384 xpb = fp_add(p.x, B);
    Fp384 xpb2 = fp_square(xpb);
    Fp384 D = fp_sub(fp_sub(xpb2, A), C);
    D = fp_add(D, D);

    // E = 3*A
    Fp384 E = fp_add(fp_add(A, A), A);

    // F = E^2
    Fp384 F = fp_square(E);

    // X3 = F - 2*D
    G1Projective result;
    Fp384 D2 = fp_add(D, D);
    result.x = fp_sub(F, D2);

    // Y3 = E*(D - X3) - 8*C
    Fp384 C8 = fp_add(C, C);
    C8 = fp_add(C8, C8);
    C8 = fp_add(C8, C8);
    result.y = fp_sub(fp_mont_mul(E, fp_sub(D, result.x)), C8);

    // Z3 = 2*Y*Z
    result.z = fp_mont_mul(p.y, p.z);
    result.z = fp_add(result.z, result.z);

    return result;
}

// Mixed addition: projective + affine (more efficient than full addition)
__device__
G1Projective g1_add_mixed(const G1Projective& p, const G1Affine& q) {
    if (q.infinity) return p;
    if (g1_is_identity(p)) return g1_from_affine(q);

    // Z1Z1 = Z1^2
    Fp384 z1z1 = fp_square(p.z);

    // U2 = X2 * Z1Z1
    Fp384 u2 = fp_mont_mul(q.x, z1z1);

    // S2 = Y2 * Z1 * Z1Z1
    Fp384 s2 = fp_mont_mul(fp_mont_mul(q.y, p.z), z1z1);

    // H = U2 - X1
    Fp384 h = fp_sub(u2, p.x);

    // R = S2 - Y1
    Fp384 r = fp_sub(s2, p.y);

    // Check for point doubling case
    if (fp_is_zero(h)) {
        if (fp_is_zero(r)) {
            // Points are equal, double
            return g1_double(p);
        }
        // Points are inverse, return identity
        return g1_identity();
    }

    // HH = H^2
    Fp384 hh = fp_square(h);

    // HHH = H * HH
    Fp384 hhh = fp_mont_mul(h, hh);

    // V = X1 * HH
    Fp384 v = fp_mont_mul(p.x, hh);

    // X3 = R^2 - HHH - 2*V
    G1Projective result;
    Fp384 r2 = fp_square(r);
    Fp384 v2 = fp_add(v, v);
    result.x = fp_sub(fp_sub(r2, hhh), v2);

    // Y3 = R * (V - X3) - Y1 * HHH
    result.y = fp_sub(fp_mont_mul(r, fp_sub(v, result.x)),
                      fp_mont_mul(p.y, hhh));

    // Z3 = Z1 * H
    result.z = fp_mont_mul(p.z, h);

    return result;
}

// Full addition: projective + projective
__device__
G1Projective g1_add(const G1Projective& p, const G1Projective& q) {
    if (g1_is_identity(p)) return q;
    if (g1_is_identity(q)) return p;

    // Z1Z1 = Z1^2, Z2Z2 = Z2^2
    Fp384 z1z1 = fp_square(p.z);
    Fp384 z2z2 = fp_square(q.z);

    // U1 = X1 * Z2Z2, U2 = X2 * Z1Z1
    Fp384 u1 = fp_mont_mul(p.x, z2z2);
    Fp384 u2 = fp_mont_mul(q.x, z1z1);

    // S1 = Y1 * Z2 * Z2Z2, S2 = Y2 * Z1 * Z1Z1
    Fp384 s1 = fp_mont_mul(fp_mont_mul(p.y, q.z), z2z2);
    Fp384 s2 = fp_mont_mul(fp_mont_mul(q.y, p.z), z1z1);

    // H = U2 - U1
    Fp384 h = fp_sub(u2, u1);

    // R = S2 - S1
    Fp384 r = fp_sub(s2, s1);

    if (fp_is_zero(h)) {
        if (fp_is_zero(r)) {
            return g1_double(p);
        }
        return g1_identity();
    }

    // HH, HHH, V
    Fp384 hh = fp_square(h);
    Fp384 hhh = fp_mont_mul(h, hh);
    Fp384 v = fp_mont_mul(u1, hh);

    // X3, Y3, Z3
    G1Projective result;
    Fp384 r2 = fp_square(r);
    Fp384 v2 = fp_add(v, v);
    result.x = fp_sub(fp_sub(r2, hhh), v2);
    result.y = fp_sub(fp_mont_mul(r, fp_sub(v, result.x)),
                      fp_mont_mul(s1, hhh));
    result.z = fp_mont_mul(fp_mont_mul(p.z, q.z), h);

    return result;
}

// Scalar multiplication using double-and-add
__device__
G1Projective g1_scalar_mul(const G1Projective& p, const Fr256& scalar) {
    G1Projective result = g1_identity();
    G1Projective temp = p;

    for (int i = 0; i < 4; i++) {
        uint64_t s = scalar.limbs[i];
        for (int j = 0; j < 64; j++) {
            if (s & 1ULL) {
                result = g1_add(result, temp);
            }
            temp = g1_double(temp);
            s >>= 1;
        }
    }

    return result;
}

// =============================================================================
// Polynomial Operations
// =============================================================================

// Evaluate polynomial at point z using Horner's method
// p(z) = c[0] + c[1]*z + c[2]*z^2 + ... + c[n]*z^n
__device__
Fr256 poly_evaluate(const Fr256* coeffs, uint32_t degree, const Fr256& z) {
    Fr256 result = fr_zero();

    for (int32_t i = (int32_t)degree; i >= 0; i--) {
        result = fr_mont_mul(result, z);
        result = fr_add(result, coeffs[i]);
    }

    return result;
}

// =============================================================================
// MSM Kernels (Pippenger's Algorithm)
// =============================================================================

// Extract window value from scalar
__device__ __forceinline__
uint32_t get_window(const Fr256& scalar, uint32_t window_idx, uint32_t window_bits) {
    uint32_t bit_offset = window_idx * window_bits;
    uint32_t limb_idx = bit_offset >> 6;
    uint32_t bit_in_limb = bit_offset & 63;

    if (limb_idx >= 4) return 0;

    uint64_t mask = (1ULL << window_bits) - 1ULL;
    uint64_t window = (scalar.limbs[limb_idx] >> bit_in_limb) & mask;

    if (bit_in_limb + window_bits > 64 && limb_idx + 1 < 4) {
        uint32_t remaining = bit_in_limb + window_bits - 64;
        window |= (scalar.limbs[limb_idx + 1] << (window_bits - remaining)) & mask;
    }

    return (uint32_t)window;
}

// Phase 1: Sort points into buckets based on scalar windows
__global__
void kzg_msm_bucket_sort(
    const G1Affine* __restrict__ bases,
    const Fr256* __restrict__ scalars,
    uint32_t* __restrict__ bucket_indices,
    uint32_t* __restrict__ point_indices,
    uint32_t num_points,
    uint32_t window_idx,
    uint32_t window_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    Fr256 scalar = scalars[idx];
    uint32_t window_val = get_window(scalar, window_idx, window_bits);

    bucket_indices[idx] = window_val;
    point_indices[idx] = idx;
}

// Phase 2: Accumulate points into buckets (per-thread local accumulation)
__global__
void kzg_msm_bucket_accumulate(
    const G1Affine* __restrict__ bases,
    const Fr256* __restrict__ scalars,
    G1Projective* __restrict__ buckets,
    uint32_t num_points,
    uint32_t window_idx,
    uint32_t window_bits,
    uint32_t num_buckets
) {
    // Shared memory for local bucket accumulation
    extern __shared__ G1Projective shared_buckets[];

    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint32_t block_size = blockDim.x;

    // Initialize shared buckets to identity
    for (uint32_t b = tid; b < num_buckets; b += block_size) {
        shared_buckets[b] = g1_identity();
    }
    __syncthreads();

    // Each thread processes multiple points
    uint32_t points_per_block = (num_points + gridDim.x - 1) / gridDim.x;
    uint32_t start = bid * points_per_block;
    uint32_t end = min(start + points_per_block, num_points);

    for (uint32_t i = start + tid; i < end; i += block_size) {
        Fr256 scalar = scalars[i];
        uint32_t window_val = get_window(scalar, window_idx, window_bits);

        if (window_val > 0) {
            G1Affine base = bases[i];
            uint32_t bucket_idx = window_val - 1;

            // Atomic-free local accumulation (each thread has its own iteration)
            // Final reduction done in separate kernel
            G1Projective old_val = shared_buckets[bucket_idx];
            shared_buckets[bucket_idx] = g1_add_mixed(old_val, base);
        }
    }
    __syncthreads();

    // Write shared buckets to global memory (one thread per bucket)
    for (uint32_t b = tid; b < num_buckets; b += block_size) {
        uint32_t global_bucket = bid * num_buckets + b;
        buckets[global_bucket] = shared_buckets[b];
    }
}

// Phase 3: Reduce buckets within each window using running sum method
__global__
void kzg_msm_bucket_reduce(
    G1Projective* __restrict__ buckets,
    G1Projective* __restrict__ window_results,
    uint32_t num_buckets,
    uint32_t num_blocks_per_window
) {
    uint32_t window_idx = blockIdx.x;

    // First, reduce partial bucket results from different blocks
    __shared__ G1Projective merged_buckets[1024]; // Max 2^10 buckets per window

    // Initialize
    for (uint32_t b = threadIdx.x; b < num_buckets; b += blockDim.x) {
        merged_buckets[b] = g1_identity();
    }
    __syncthreads();

    // Merge buckets from all blocks for this window
    for (uint32_t block = 0; block < num_blocks_per_window; block++) {
        uint32_t bucket_offset = (window_idx * num_blocks_per_window + block) * num_buckets;
        for (uint32_t b = threadIdx.x; b < num_buckets; b += blockDim.x) {
            merged_buckets[b] = g1_add(merged_buckets[b], buckets[bucket_offset + b]);
        }
        __syncthreads();
    }

    // Running sum reduction (single thread)
    if (threadIdx.x == 0) {
        G1Projective running = g1_identity();
        G1Projective sum = g1_identity();

        for (int32_t i = (int32_t)num_buckets - 1; i >= 0; i--) {
            running = g1_add(running, merged_buckets[i]);
            sum = g1_add(sum, running);
        }

        window_results[window_idx] = sum;
    }
}

// Phase 4: Combine window results
__global__
void kzg_msm_window_combine(
    const G1Projective* __restrict__ window_results,
    G1Projective* __restrict__ result,
    uint32_t num_windows,
    uint32_t window_bits
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    G1Projective acc = g1_identity();

    for (int32_t w = (int32_t)num_windows - 1; w >= 0; w--) {
        // Double by window_bits
        for (uint32_t i = 0; i < window_bits; i++) {
            acc = g1_double(acc);
        }

        // Add window result
        acc = g1_add(acc, window_results[w]);
    }

    result[0] = acc;
}

// =============================================================================
// KZG-Specific Kernels
// =============================================================================

// Compute polynomial quotient: q(x) = (p(x) - p(z)) / (x - z)
// Used for KZG proof generation
__global__
void kzg_compute_quotient(
    const Fr256* __restrict__ poly_coeffs,
    Fr256* __restrict__ quotient_coeffs,
    const Fr256* __restrict__ z_point,
    const Fr256* __restrict__ p_z_value,
    uint32_t degree
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0 || degree == 0) return;  // Single-thread computation for correctness

    Fr256 z = z_point[0];

    // Synthetic division: q[i] = p[i+1] + z * q[i+1]
    // Working backwards from highest degree coefficient
    Fr256 q[4096];  // Max blob degree
    q[degree - 1] = poly_coeffs[degree];

    for (int32_t i = (int32_t)degree - 2; i >= 0; i--) {
        q[i] = fr_add(poly_coeffs[i + 1], fr_mont_mul(z, q[i + 1]));
    }

    // Write quotient coefficients
    for (uint32_t i = 0; i < degree; i++) {
        quotient_coeffs[i] = q[i];
    }
}

// Parallel quotient computation using tree reduction
__global__
void kzg_compute_quotient_parallel(
    const Fr256* __restrict__ poly_coeffs,
    Fr256* __restrict__ quotient_coeffs,
    Fr256 z,
    uint32_t degree
) {
    extern __shared__ Fr256 shared_q[];

    uint32_t tid = threadIdx.x;
    uint32_t block_size = blockDim.x;

    // Initialize quotient coefficients
    if (tid < degree) {
        shared_q[tid] = (tid == degree - 1) ? poly_coeffs[degree] : fr_zero();
    }
    __syncthreads();

    // Parallel prefix computation (needs multiple passes)
    for (int32_t i = (int32_t)degree - 2; i >= 0; i--) {
        if (tid == 0) {
            shared_q[i] = fr_add(poly_coeffs[i + 1], fr_mont_mul(z, shared_q[i + 1]));
        }
        __syncthreads();
    }

    // Write output
    if (tid < degree) {
        quotient_coeffs[tid] = shared_q[tid];
    }
}

// Batch polynomial evaluation kernel
__global__
void kzg_batch_evaluate(
    const Fr256* __restrict__ coeffs,
    const Fr256* __restrict__ points,
    Fr256* __restrict__ results,
    uint32_t degree,
    uint32_t num_evals
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_evals) return;

    Fr256 z = points[idx];
    Fr256 result = fr_zero();

    // Horner's method
    for (int32_t i = (int32_t)degree; i >= 0; i--) {
        result = fr_mont_mul(result, z);
        result = fr_add(result, coeffs[i]);
    }

    results[idx] = result;
}

// Batch commitment generation (MSM wrapper)
__global__
void kzg_batch_commit_kernel(
    const G1Affine* __restrict__ srs_g1,
    const Fr256* __restrict__ poly_coeffs,
    G1Projective* __restrict__ partial_sums,
    uint32_t degree,
    uint32_t poly_idx
) {
    // Each block handles a portion of the polynomial
    extern __shared__ G1Projective block_sum[];

    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x + tid;

    // Initialize thread-local accumulator
    G1Projective local_sum = g1_identity();

    // Strided access for coalescing
    for (uint32_t i = gid; i <= degree; i += gridDim.x * blockDim.x) {
        Fr256 coeff = poly_coeffs[poly_idx * (degree + 1) + i];
        if (!fr_is_zero(coeff)) {
            G1Projective scaled = g1_scalar_mul(g1_from_affine(srs_g1[i]), coeff);
            local_sum = g1_add(local_sum, scaled);
        }
    }

    // Store in shared memory
    block_sum[tid] = local_sum;
    __syncthreads();

    // Reduction within block
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            block_sum[tid] = g1_add(block_sum[tid], block_sum[tid + s]);
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        partial_sums[poly_idx * gridDim.x + blockIdx.x] = block_sum[0];
    }
}

// Final reduction for batch commitments
__global__
void kzg_reduce_partial_sums(
    const G1Projective* __restrict__ partial_sums,
    G1Projective* __restrict__ commitments,
    uint32_t num_blocks,
    uint32_t num_polys
) {
    uint32_t poly_idx = blockIdx.x;
    if (poly_idx >= num_polys) return;

    G1Projective sum = g1_identity();

    for (uint32_t b = 0; b < num_blocks; b++) {
        sum = g1_add(sum, partial_sums[poly_idx * num_blocks + b]);
    }

    commitments[poly_idx] = sum;
}

// Prepare data for pairing check (actual pairing done on CPU)
__global__
void kzg_prepare_pairing(
    const G1Projective* __restrict__ commitment,
    const G1Projective* __restrict__ proof,
    const Fr256* __restrict__ z,
    const Fr256* __restrict__ y,
    const G1Affine* __restrict__ g1_gen,
    G1Projective* __restrict__ lhs_g1,   // C - [y]G1
    G1Projective* __restrict__ rhs_g1    // W (proof)
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // LHS = C - [y]*G1
    G1Projective y_g1 = g1_scalar_mul(g1_from_affine(g1_gen[0]), y[0]);
    G1Projective neg_y_g1;
    neg_y_g1.x = y_g1.x;
    neg_y_g1.y = fp_neg(y_g1.y);
    neg_y_g1.z = y_g1.z;

    lhs_g1[0] = g1_add(commitment[0], neg_y_g1);

    // RHS = W (the proof itself)
    rhs_g1[0] = proof[0];
}

} // namespace kzg
} // namespace cuda
} // namespace lux

// =============================================================================
// C API for Go CGO bindings
// =============================================================================

extern "C" {

using namespace lux::cuda::kzg;

// Compute KZG commitment: C = sum(coeffs[i] * g1[i])
int lux_cuda_kzg_commit(
    const void* srs_g1,      // G1Affine[degree+1] - trusted setup
    const void* coeffs,       // Fr256[degree+1] - polynomial coefficients
    void* commitment,         // G1Projective[1] - output commitment
    uint32_t degree,
    cudaStream_t stream
) {
    // Determine optimal parameters
    uint32_t window_bits = 12;
    if (degree < (1 << 10)) window_bits = 8;
    if (degree > (1 << 14)) window_bits = 14;

    uint32_t num_windows = (256 + window_bits - 1) / window_bits;
    uint32_t num_buckets = (1u << window_bits) - 1;

    // Allocate temporary buffers
    G1Projective* d_buckets;
    G1Projective* d_window_results;
    uint32_t num_blocks = (degree + 255) / 256;

    cudaMalloc(&d_buckets, num_blocks * num_buckets * sizeof(G1Projective));
    cudaMalloc(&d_window_results, num_windows * sizeof(G1Projective));

    // Run MSM phases for each window
    for (uint32_t w = 0; w < num_windows; w++) {
        dim3 block(256);
        dim3 grid(num_blocks);

        size_t shared_mem = num_buckets * sizeof(G1Projective);

        kzg_msm_bucket_accumulate<<<grid, block, shared_mem, stream>>>(
            (const G1Affine*)srs_g1,
            (const Fr256*)coeffs,
            d_buckets,
            degree + 1,
            w,
            window_bits,
            num_buckets
        );

        kzg_msm_bucket_reduce<<<1, 256, 0, stream>>>(
            d_buckets,
            d_window_results + w,
            num_buckets,
            num_blocks
        );
    }

    // Combine windows
    kzg_msm_window_combine<<<1, 1, 0, stream>>>(
        d_window_results,
        (G1Projective*)commitment,
        num_windows,
        window_bits
    );

    cudaFree(d_buckets);
    cudaFree(d_window_results);

    return cudaGetLastError();
}

// Evaluate polynomial at point z
int lux_cuda_kzg_evaluate(
    const void* coeffs,       // Fr256[degree+1]
    const void* z,            // Fr256[1]
    void* result,             // Fr256[1]
    uint32_t degree,
    cudaStream_t stream
) {
    kzg_batch_evaluate<<<1, 1, 0, stream>>>(
        (const Fr256*)coeffs,
        (const Fr256*)z,
        (Fr256*)result,
        degree,
        1
    );

    return cudaGetLastError();
}

// Compute quotient polynomial for proof generation
int lux_cuda_kzg_compute_quotient(
    const void* poly_coeffs,  // Fr256[degree+1]
    void* quotient_coeffs,    // Fr256[degree]
    const void* z,            // Fr256[1]
    const void* p_z,          // Fr256[1] - p(z) value
    uint32_t degree,
    cudaStream_t stream
) {
    kzg_compute_quotient<<<1, 1, 0, stream>>>(
        (const Fr256*)poly_coeffs,
        (Fr256*)quotient_coeffs,
        (const Fr256*)z,
        (const Fr256*)p_z,
        degree
    );

    return cudaGetLastError();
}

// Generate KZG proof: W = sum(quotient[i] * g1[i])
int lux_cuda_kzg_prove(
    const void* srs_g1,       // G1Affine[degree]
    const void* quotient,     // Fr256[degree]
    void* proof,              // G1Projective[1]
    uint32_t degree,
    cudaStream_t stream
) {
    // Proof is just MSM of quotient polynomial
    return lux_cuda_kzg_commit(srs_g1, quotient, proof, degree - 1, stream);
}

// Batch commit multiple polynomials
int lux_cuda_kzg_batch_commit(
    const void* srs_g1,       // G1Affine[degree+1]
    const void* coeffs,       // Fr256[(degree+1) * num_polys]
    void* commitments,        // G1Projective[num_polys]
    uint32_t degree,
    uint32_t num_polys,
    cudaStream_t stream
) {
    uint32_t num_blocks = (degree + 255) / 256;

    G1Projective* d_partial_sums;
    cudaMalloc(&d_partial_sums, num_polys * num_blocks * sizeof(G1Projective));

    dim3 block(256);
    dim3 grid(num_blocks);
    size_t shared_mem = 256 * sizeof(G1Projective);

    for (uint32_t p = 0; p < num_polys; p++) {
        kzg_batch_commit_kernel<<<grid, block, shared_mem, stream>>>(
            (const G1Affine*)srs_g1,
            (const Fr256*)coeffs,
            d_partial_sums,
            degree,
            p
        );
    }

    kzg_reduce_partial_sums<<<num_polys, 1, 0, stream>>>(
        d_partial_sums,
        (G1Projective*)commitments,
        num_blocks,
        num_polys
    );

    cudaFree(d_partial_sums);

    return cudaGetLastError();
}

// Batch evaluate polynomial at multiple points
int lux_cuda_kzg_batch_evaluate(
    const void* coeffs,       // Fr256[degree+1]
    const void* points,       // Fr256[num_points]
    void* results,            // Fr256[num_points]
    uint32_t degree,
    uint32_t num_points,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((num_points + block.x - 1) / block.x);

    kzg_batch_evaluate<<<grid, block, 0, stream>>>(
        (const Fr256*)coeffs,
        (const Fr256*)points,
        (Fr256*)results,
        degree,
        num_points
    );

    return cudaGetLastError();
}

// Prepare pairing inputs for verification (pairing done on CPU)
int lux_cuda_kzg_prepare_verify(
    const void* commitment,   // G1Projective[1]
    const void* proof,        // G1Projective[1]
    const void* z,            // Fr256[1]
    const void* y,            // Fr256[1]
    const void* g1_gen,       // G1Affine[1]
    void* lhs_g1,             // G1Projective[1] output
    void* rhs_g1,             // G1Projective[1] output
    cudaStream_t stream
) {
    kzg_prepare_pairing<<<1, 1, 0, stream>>>(
        (const G1Projective*)commitment,
        (const G1Projective*)proof,
        (const Fr256*)z,
        (const Fr256*)y,
        (const G1Affine*)g1_gen,
        (G1Projective*)lhs_g1,
        (G1Projective*)rhs_g1
    );

    return cudaGetLastError();
}

// MSM (multi-scalar multiplication) - general purpose
int lux_cuda_kzg_msm(
    const void* bases,        // G1Affine[count]
    const void* scalars,      // Fr256[count]
    void* result,             // G1Projective[1]
    uint32_t count,
    cudaStream_t stream
) {
    return lux_cuda_kzg_commit(bases, scalars, result, count - 1, stream);
}

} // extern "C"
