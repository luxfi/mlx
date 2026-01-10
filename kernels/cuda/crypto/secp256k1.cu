// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// secp256k1 GPU Acceleration - CUDA Kernel
// Implements GTable-based scalar multiplication for threshold ECDSA
// For NVIDIA GPUs (sm_60+ recommended)
//
// Curve: secp256k1 (Bitcoin/Ethereum)
// p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
// n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
// G = (0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
//      0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================================
// Configuration
// ============================================================================

#define GTABLE_CHUNKS       16      // 256 bits / 16 = 16 bits per chunk
#define GTABLE_CHUNK_SIZE   65536   // 2^16 = 65536 points per chunk
#define KECCAK_ROUNDS       24

// ============================================================================
// 256-bit Field Element (4 x 64-bit limbs)
// ============================================================================

struct Fp256 {
    uint64_t limbs[4];  // Little-endian: limbs[0] is LSB
};

// secp256k1 prime: p = 2^256 - 2^32 - 977
// p = 0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F
__constant__ uint64_t SECP256K1_P[4] = {
    0xFFFFFFFEFFFFFC2FULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL
};

// Curve order n
__constant__ uint64_t SECP256K1_N[4] = {
    0xBFD25E8CD0364141ULL,
    0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL,
    0xFFFFFFFFFFFFFFFFULL
};

// Generator point G
__constant__ uint64_t SECP256K1_GX[4] = {
    0x59F2815B16F81798ULL,
    0x029BFCDB2DCE28D9ULL,
    0x55A06295CE870B07ULL,
    0x79BE667EF9DCBBACULL
};

__constant__ uint64_t SECP256K1_GY[4] = {
    0x9C47D08FFB10D4B8ULL,
    0xFD17B448A6855419ULL,
    0x5DA4FBFC0E1108A8ULL,
    0x483ADA7726A3C465ULL
};

// Montgomery constant: R^2 mod p (for converting to Montgomery form)
__constant__ uint64_t SECP256K1_R2[4] = {
    0x0000000000000001ULL,
    0x0000000100000000ULL,
    0x0000000000000000ULL,
    0x0000000000000000ULL
};

// -p^{-1} mod 2^64
__constant__ uint64_t SECP256K1_INV = 0xD838091DD2253531ULL;

// ============================================================================
// Basic Operations
// ============================================================================

// Add with carry
__device__ __forceinline__
uint64_t adc(uint64_t a, uint64_t b, uint64_t& carry) {
    uint64_t sum = a + b + carry;
    carry = (sum < a || (carry && sum == a)) ? 1ULL : 0ULL;
    return sum;
}

// Subtract with borrow
__device__ __forceinline__
uint64_t sbb(uint64_t a, uint64_t b, uint64_t& borrow) {
    uint64_t diff = a - b - borrow;
    borrow = (a < b || (borrow && a == b)) ? 1ULL : 0ULL;
    return diff;
}

// Multiply and accumulate with carry
__device__ __forceinline__
void mac(uint64_t& lo, uint64_t& carry, uint64_t a, uint64_t b, uint64_t c) {
    uint64_t product_lo = a * b;
    uint64_t product_hi = __umul64hi(a, b);

    uint64_t sum = product_lo + c;
    uint64_t c1 = (sum < product_lo) ? 1ULL : 0ULL;

    sum += carry;
    uint64_t c2 = (sum < carry) ? 1ULL : 0ULL;

    lo = sum;
    carry = product_hi + c1 + c2;
}

// Check if zero
__device__
bool fp256_is_zero(const Fp256& a) {
    return (a.limbs[0] | a.limbs[1] | a.limbs[2] | a.limbs[3]) == 0;
}

// ============================================================================
// secp256k1 Field Arithmetic
// ============================================================================

// Field addition mod p
__device__
Fp256 fp256_add(const Fp256& a, const Fp256& b) {
    Fp256 result;
    uint64_t carry = 0;

    result.limbs[0] = adc(a.limbs[0], b.limbs[0], carry);
    result.limbs[1] = adc(a.limbs[1], b.limbs[1], carry);
    result.limbs[2] = adc(a.limbs[2], b.limbs[2], carry);
    result.limbs[3] = adc(a.limbs[3], b.limbs[3], carry);

    // Reduce if >= p
    uint64_t borrow = 0;
    Fp256 reduced;
    reduced.limbs[0] = sbb(result.limbs[0], SECP256K1_P[0], borrow);
    reduced.limbs[1] = sbb(result.limbs[1], SECP256K1_P[1], borrow);
    reduced.limbs[2] = sbb(result.limbs[2], SECP256K1_P[2], borrow);
    reduced.limbs[3] = sbb(result.limbs[3], SECP256K1_P[3], borrow);

    bool needs_reduce = (carry != 0) || (borrow == 0);
    if (needs_reduce) return reduced;
    return result;
}

// Field subtraction mod p
__device__
Fp256 fp256_sub(const Fp256& a, const Fp256& b) {
    Fp256 result;
    uint64_t borrow = 0;

    result.limbs[0] = sbb(a.limbs[0], b.limbs[0], borrow);
    result.limbs[1] = sbb(a.limbs[1], b.limbs[1], borrow);
    result.limbs[2] = sbb(a.limbs[2], b.limbs[2], borrow);
    result.limbs[3] = sbb(a.limbs[3], b.limbs[3], borrow);

    // Add p if underflow
    if (borrow) {
        uint64_t carry = 0;
        result.limbs[0] = adc(result.limbs[0], SECP256K1_P[0], carry);
        result.limbs[1] = adc(result.limbs[1], SECP256K1_P[1], carry);
        result.limbs[2] = adc(result.limbs[2], SECP256K1_P[2], carry);
        result.limbs[3] = adc(result.limbs[3], SECP256K1_P[3], carry);
    }

    return result;
}

// Double the field element
__device__
Fp256 fp256_double(const Fp256& a) {
    return fp256_add(a, a);
}

// Negate
__device__
Fp256 fp256_neg(const Fp256& a) {
    if (fp256_is_zero(a)) return a;
    Fp256 p = {{SECP256K1_P[0], SECP256K1_P[1], SECP256K1_P[2], SECP256K1_P[3]}};
    return fp256_sub(p, a);
}

// secp256k1 fast reduction using p = 2^256 - 2^32 - 977
// For a 512-bit product T, compute T mod p
__device__
Fp256 fp256_reduce_512(uint64_t t[8]) {
    // T = T_lo + T_hi * 2^256
    // 2^256 mod p = 2^32 + 977
    // So T mod p = T_lo + T_hi * (2^32 + 977)

    Fp256 lo = {{t[0], t[1], t[2], t[3]}};
    Fp256 hi = {{t[4], t[5], t[6], t[7]}};

    // Compute hi * 977
    uint64_t h977[5] = {0, 0, 0, 0, 0};
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        mac(h977[i], carry, hi.limbs[i], 977ULL, h977[i]);
    }
    h977[4] = carry;

    // Compute hi * 2^32 (shift left by 32 bits)
    uint64_t hi_shifted[5];
    hi_shifted[0] = hi.limbs[0] << 32;
    hi_shifted[1] = (hi.limbs[0] >> 32) | (hi.limbs[1] << 32);
    hi_shifted[2] = (hi.limbs[1] >> 32) | (hi.limbs[2] << 32);
    hi_shifted[3] = (hi.limbs[2] >> 32) | (hi.limbs[3] << 32);
    hi_shifted[4] = hi.limbs[3] >> 32;

    // Add hi_shifted + h977
    uint64_t reduction[5];
    carry = 0;
    for (int i = 0; i < 5; i++) {
        reduction[i] = adc(hi_shifted[i], h977[i], carry);
    }

    // Add lo + reduction[0..3]
    Fp256 result;
    carry = 0;
    result.limbs[0] = adc(lo.limbs[0], reduction[0], carry);
    result.limbs[1] = adc(lo.limbs[1], reduction[1], carry);
    result.limbs[2] = adc(lo.limbs[2], reduction[2], carry);
    result.limbs[3] = adc(lo.limbs[3], reduction[3], carry);

    // Handle overflow from reduction[4] and carry
    uint64_t overflow = reduction[4] + carry;

    // If overflow > 0, reduce again: overflow * (2^32 + 977)
    while (overflow > 0) {
        uint64_t correction_lo = overflow * 977ULL;
        uint64_t correction_hi = (overflow << 32);

        uint64_t c = 0;
        result.limbs[0] = adc(result.limbs[0], correction_lo, c);
        result.limbs[1] = adc(result.limbs[1], correction_hi, c);
        result.limbs[2] = adc(result.limbs[2], 0, c);
        result.limbs[3] = adc(result.limbs[3], 0, c);
        overflow = c;
    }

    // Final reduction if >= p
    uint64_t borrow = 0;
    Fp256 reduced;
    reduced.limbs[0] = sbb(result.limbs[0], SECP256K1_P[0], borrow);
    reduced.limbs[1] = sbb(result.limbs[1], SECP256K1_P[1], borrow);
    reduced.limbs[2] = sbb(result.limbs[2], SECP256K1_P[2], borrow);
    reduced.limbs[3] = sbb(result.limbs[3], SECP256K1_P[3], borrow);

    if (borrow == 0) return reduced;
    return result;
}

// Field multiplication mod p
__device__
Fp256 fp256_mul(const Fp256& a, const Fp256& b) {
    // Schoolbook multiplication -> 512-bit product
    uint64_t t[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            mac(t[i + j], carry, a.limbs[i], b.limbs[j], t[i + j]);
        }
        t[i + 4] = carry;
    }

    return fp256_reduce_512(t);
}

// Field squaring (optimized)
__device__
Fp256 fp256_sqr(const Fp256& a) {
    // Could be optimized with squaring-specific formula
    return fp256_mul(a, a);
}

// Modular inverse using Fermat's little theorem: a^{-1} = a^{p-2} mod p
__device__
Fp256 fp256_inv(const Fp256& a) {
    // p - 2 for secp256k1
    // Use square-and-multiply with optimized addition chain

    Fp256 result = {{1, 0, 0, 0}};
    Fp256 base = a;

    // Binary exponentiation with p-2
    // p-2 = 0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2D

    // Simplified: 256 iterations (not optimized)
    uint64_t exp[4] = {
        0xFFFFFFFEFFFFFC2DULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL
    };

    for (int limb = 0; limb < 4; limb++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((exp[limb] >> bit) & 1) {
                result = fp256_mul(result, base);
            }
            base = fp256_sqr(base);
        }
    }

    return result;
}

// ============================================================================
// Point Types
// ============================================================================

struct AffinePoint {
    Fp256 x;
    Fp256 y;
    int infinity;  // 0 = regular point, 1 = point at infinity
};

struct JacobianPoint {
    Fp256 x;
    Fp256 y;
    Fp256 z;
};

__device__
JacobianPoint jacobian_identity() {
    JacobianPoint p;
    p.x = {{1, 0, 0, 0}};
    p.y = {{1, 0, 0, 0}};
    p.z = {{0, 0, 0, 0}};
    return p;
}

__device__
bool jacobian_is_identity(const JacobianPoint& p) {
    return fp256_is_zero(p.z);
}

// ============================================================================
// Point Operations
// ============================================================================

// Point doubling in Jacobian coordinates (a=0 for secp256k1)
// Formula: dbl-2009-l (4M + 6S)
__device__
JacobianPoint point_double(const JacobianPoint& p) {
    if (jacobian_is_identity(p)) return p;

    Fp256 A = fp256_sqr(p.x);           // X^2
    Fp256 B = fp256_sqr(p.y);           // Y^2
    Fp256 C = fp256_sqr(B);             // Y^4

    // D = 2*((X+B)^2 - A - C)
    Fp256 xplusb = fp256_add(p.x, B);
    Fp256 xplusb_sq = fp256_sqr(xplusb);
    Fp256 D = fp256_sub(xplusb_sq, A);
    D = fp256_sub(D, C);
    D = fp256_double(D);

    // E = 3*A
    Fp256 E = fp256_add(A, fp256_double(A));

    // F = E^2
    Fp256 F = fp256_sqr(E);

    // X' = F - 2*D
    Fp256 X3 = fp256_sub(F, fp256_double(D));

    // Y' = E*(D - X') - 8*C
    Fp256 dmx = fp256_sub(D, X3);
    Fp256 edx = fp256_mul(E, dmx);
    Fp256 c8 = fp256_double(fp256_double(fp256_double(C)));
    Fp256 Y3 = fp256_sub(edx, c8);

    // Z' = 2*Y*Z
    Fp256 Z3 = fp256_mul(p.y, p.z);
    Z3 = fp256_double(Z3);

    JacobianPoint result = {X3, Y3, Z3};
    return result;
}

// Mixed addition: Jacobian + Affine -> Jacobian
// Formula: madd-2008-s (7M + 4S)
__device__
JacobianPoint point_add_mixed(const JacobianPoint& p, const AffinePoint& q) {
    if (q.infinity) return p;
    if (jacobian_is_identity(p)) {
        Fp256 one = {{1, 0, 0, 0}};
        return {q.x, q.y, one};
    }

    Fp256 Z1Z1 = fp256_sqr(p.z);
    Fp256 U2 = fp256_mul(q.x, Z1Z1);
    Fp256 S2 = fp256_mul(q.y, p.z);
    S2 = fp256_mul(S2, Z1Z1);

    Fp256 H = fp256_sub(U2, p.x);
    Fp256 HH = fp256_sqr(H);
    Fp256 I = fp256_double(fp256_double(HH));
    Fp256 J = fp256_mul(H, I);
    Fp256 r = fp256_sub(S2, p.y);
    r = fp256_double(r);
    Fp256 V = fp256_mul(p.x, I);

    // X' = r^2 - J - 2*V
    Fp256 r_sq = fp256_sqr(r);
    Fp256 X3 = fp256_sub(r_sq, J);
    X3 = fp256_sub(X3, fp256_double(V));

    // Y' = r*(V - X') - 2*Y1*J
    Fp256 vmx = fp256_sub(V, X3);
    Fp256 rvmx = fp256_mul(r, vmx);
    Fp256 y1j = fp256_mul(p.y, J);
    Fp256 Y3 = fp256_sub(rvmx, fp256_double(y1j));

    // Z' = (Z1+H)^2 - Z1Z1 - HH
    Fp256 zph = fp256_add(p.z, H);
    Fp256 zph_sq = fp256_sqr(zph);
    Fp256 Z3 = fp256_sub(zph_sq, Z1Z1);
    Z3 = fp256_sub(Z3, HH);

    JacobianPoint result = {X3, Y3, Z3};
    return result;
}

// Full Jacobian addition (12M + 4S)
__device__
JacobianPoint point_add(const JacobianPoint& p, const JacobianPoint& q) {
    if (jacobian_is_identity(p)) return q;
    if (jacobian_is_identity(q)) return p;

    Fp256 Z1Z1 = fp256_sqr(p.z);
    Fp256 Z2Z2 = fp256_sqr(q.z);
    Fp256 U1 = fp256_mul(p.x, Z2Z2);
    Fp256 U2 = fp256_mul(q.x, Z1Z1);
    Fp256 S1 = fp256_mul(p.y, q.z);
    S1 = fp256_mul(S1, Z2Z2);
    Fp256 S2 = fp256_mul(q.y, p.z);
    S2 = fp256_mul(S2, Z1Z1);

    Fp256 H = fp256_sub(U2, U1);
    Fp256 I = fp256_double(H);
    I = fp256_sqr(I);
    Fp256 J = fp256_mul(H, I);
    Fp256 r = fp256_sub(S2, S1);
    r = fp256_double(r);
    Fp256 V = fp256_mul(U1, I);

    Fp256 r_sq = fp256_sqr(r);
    Fp256 X3 = fp256_sub(r_sq, J);
    X3 = fp256_sub(X3, fp256_double(V));

    Fp256 vmx = fp256_sub(V, X3);
    Fp256 rvmx = fp256_mul(r, vmx);
    Fp256 s1j = fp256_mul(S1, J);
    Fp256 Y3 = fp256_sub(rvmx, fp256_double(s1j));

    Fp256 zsum = fp256_add(p.z, q.z);
    Fp256 zsum_sq = fp256_sqr(zsum);
    Fp256 Z3 = fp256_sub(zsum_sq, Z1Z1);
    Z3 = fp256_sub(Z3, Z2Z2);
    Z3 = fp256_mul(Z3, H);

    JacobianPoint result = {X3, Y3, Z3};
    return result;
}

// Jacobian to Affine conversion
__device__
AffinePoint jacobian_to_affine(const JacobianPoint& p) {
    if (jacobian_is_identity(p)) {
        AffinePoint identity;
        identity.x = {{0, 0, 0, 0}};
        identity.y = {{0, 0, 0, 0}};
        identity.infinity = 1;
        return identity;
    }

    Fp256 z_inv = fp256_inv(p.z);
    Fp256 z_inv_sq = fp256_sqr(z_inv);
    Fp256 z_inv_cubed = fp256_mul(z_inv_sq, z_inv);

    AffinePoint result;
    result.x = fp256_mul(p.x, z_inv_sq);
    result.y = fp256_mul(p.y, z_inv_cubed);
    result.infinity = 0;

    return result;
}

// ============================================================================
// Keccak256 (for Ethereum address derivation)
// ============================================================================

__constant__ uint64_t KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
    0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

__constant__ int KECCAK_ROTC[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
};

__constant__ int KECCAK_PILN[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
};

__device__ __forceinline__
uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

__device__
void keccak_f1600(uint64_t state[25]) {
    for (int round = 0; round < KECCAK_ROUNDS; round++) {
        // Theta
        uint64_t C[5], D[5];
        for (int x = 0; x < 5; x++) {
            C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        for (int x = 0; x < 5; x++) {
            D[x] = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
            for (int y = 0; y < 5; y++) {
                state[y * 5 + x] ^= D[x];
            }
        }

        // Rho and Pi
        uint64_t temp = state[1];
        for (int i = 0; i < 24; i++) {
            int j = KECCAK_PILN[i];
            uint64_t t = state[j];
            state[j] = rotl64(temp, KECCAK_ROTC[i]);
            temp = t;
        }

        // Chi
        for (int y = 0; y < 5; y++) {
            uint64_t row[5];
            for (int x = 0; x < 5; x++) {
                row[x] = state[y * 5 + x];
            }
            for (int x = 0; x < 5; x++) {
                state[y * 5 + x] = row[x] ^ ((~row[(x + 1) % 5]) & row[(x + 2) % 5]);
            }
        }

        // Iota
        state[0] ^= KECCAK_RC[round];
    }
}

// Keccak256 for 64-byte public key -> 32-byte hash
__device__
void keccak256_64bytes(const uint8_t* input, uint8_t* output) {
    uint64_t state[25] = {0};

    // Absorb 64 bytes (rate = 136 bytes for Keccak-256)
    for (int i = 0; i < 8; i++) {
        uint64_t word = 0;
        for (int j = 0; j < 8; j++) {
            word |= ((uint64_t)input[i * 8 + j]) << (j * 8);
        }
        state[i] = word;
    }

    // Padding (64 bytes + padding in 136-byte rate)
    state[8] ^= 0x01ULL;    // Start of padding
    state[16] ^= 0x8000000000000000ULL;  // End of padding (rate/8 - 1 = 16)

    // Permute
    keccak_f1600(state);

    // Squeeze 32 bytes
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            output[i * 8 + j] = (uint8_t)(state[i] >> (j * 8));
        }
    }
}

// ============================================================================
// GTable Scalar Multiplication Kernels
// ============================================================================

// Extract 16-bit chunk from 256-bit scalar
__device__ __forceinline__
uint32_t extract_chunk(const Fp256& scalar, uint32_t chunk_idx) {
    uint32_t bit_offset = chunk_idx * 16;
    uint32_t limb_idx = bit_offset / 64;
    uint32_t bit_in_limb = bit_offset % 64;

    uint64_t chunk = scalar.limbs[limb_idx] >> bit_in_limb;

    // Handle cross-limb chunk
    if (bit_in_limb + 16 > 64 && limb_idx + 1 < 4) {
        chunk |= scalar.limbs[limb_idx + 1] << (64 - bit_in_limb);
    }

    return (uint32_t)(chunk & 0xFFFFULL);
}

// GTable-based scalar multiplication
// k*G = sum_{i=0}^{15} gtable[i][chunk_i]
extern "C" __global__
void gtable_scalar_mul(
    const AffinePoint* __restrict__ gtable,  // [16][65536] precomputed points
    const Fp256* __restrict__ scalars,
    AffinePoint* __restrict__ results,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    Fp256 scalar = scalars[idx];
    JacobianPoint acc = jacobian_identity();

    // Process 16 chunks of 16 bits each
    for (uint32_t chunk_idx = 0; chunk_idx < GTABLE_CHUNKS; chunk_idx++) {
        uint32_t chunk_val = extract_chunk(scalar, chunk_idx);

        if (chunk_val != 0) {
            // Look up precomputed point
            uint32_t table_offset = chunk_idx * GTABLE_CHUNK_SIZE + chunk_val - 1;
            AffinePoint table_point = gtable[table_offset];

            // Add to accumulator
            acc = point_add_mixed(acc, table_point);
        }
    }

    // Convert to affine and store
    results[idx] = jacobian_to_affine(acc);
}

// ============================================================================
// Batch ECDSA Signature Verification
// ============================================================================

struct SignatureData {
    Fp256 r;       // r component
    Fp256 s;       // s component
    Fp256 e;       // message hash
    Fp256 pub_x;   // public key x
    Fp256 pub_y;   // public key y
};

// Scalar multiplication mod n (curve order)
__device__
Fp256 scalar_mul_mod_n(const Fp256& a, const Fp256& b) {
    // Similar to fp256_mul but with reduction mod n
    uint64_t t[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            mac(t[i + j], carry, a.limbs[i], b.limbs[j], t[i + j]);
        }
        t[i + 4] = carry;
    }

    // Reduction mod n (simplified - should use Barrett or similar)
    Fp256 result = {{t[0], t[1], t[2], t[3]}};
    return result;  // Placeholder
}

// Modular inverse mod n
__device__
Fp256 scalar_inv_mod_n(const Fp256& a) {
    // Use Fermat's little theorem: a^{-1} = a^{n-2} mod n
    // Simplified implementation
    Fp256 result = {{1, 0, 0, 0}};
    Fp256 base = a;

    uint64_t exp[4] = {
        SECP256K1_N[0] - 2,
        SECP256K1_N[1],
        SECP256K1_N[2],
        SECP256K1_N[3]
    };

    for (int limb = 0; limb < 4; limb++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((exp[limb] >> bit) & 1) {
                result = scalar_mul_mod_n(result, base);
            }
            base = scalar_mul_mod_n(base, base);
        }
    }

    return result;
}

extern "C" __global__
void batch_verify_ecdsa(
    const AffinePoint* __restrict__ gtable,
    const SignatureData* __restrict__ signatures,
    int* __restrict__ results,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    SignatureData sig = signatures[idx];

    // ECDSA verification:
    // w = s^{-1} mod n
    // u1 = e*w mod n
    // u2 = r*w mod n
    // R = u1*G + u2*P
    // verify R.x == r

    Fp256 w = scalar_inv_mod_n(sig.s);
    Fp256 u1 = scalar_mul_mod_n(sig.e, w);
    Fp256 u2 = scalar_mul_mod_n(sig.r, w);

    // Compute u1*G using GTable
    JacobianPoint acc1 = jacobian_identity();
    for (uint32_t chunk_idx = 0; chunk_idx < GTABLE_CHUNKS; chunk_idx++) {
        uint32_t chunk_val = extract_chunk(u1, chunk_idx);
        if (chunk_val != 0) {
            uint32_t table_offset = chunk_idx * GTABLE_CHUNK_SIZE + chunk_val - 1;
            acc1 = point_add_mixed(acc1, gtable[table_offset]);
        }
    }

    // Compute u2*P using double-and-add
    AffinePoint P = {sig.pub_x, sig.pub_y, 0};
    JacobianPoint acc2 = jacobian_identity();
    JacobianPoint base = {P.x, P.y, {{1, 0, 0, 0}}};

    for (int limb = 0; limb < 4; limb++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((u2.limbs[limb] >> bit) & 1) {
                acc2 = point_add(acc2, base);
            }
            base = point_double(base);
        }
    }

    // R = u1*G + u2*P
    JacobianPoint R = point_add(acc1, acc2);
    AffinePoint R_affine = jacobian_to_affine(R);

    // Verify R.x == r (simplified comparison)
    bool valid = (R_affine.x.limbs[0] == sig.r.limbs[0]) &&
                 (R_affine.x.limbs[1] == sig.r.limbs[1]) &&
                 (R_affine.x.limbs[2] == sig.r.limbs[2]) &&
                 (R_affine.x.limbs[3] == sig.r.limbs[3]);

    results[idx] = valid ? 1 : 0;
}

// ============================================================================
// Batch Address Derivation
// ============================================================================

extern "C" __global__
void batch_derive_address(
    const Fp256* __restrict__ pub_x,
    const Fp256* __restrict__ pub_y,
    uint8_t* __restrict__ addresses,  // 20 bytes per address
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // Serialize public key (64 bytes: x || y)
    uint8_t pubkey[64];
    for (int i = 0; i < 4; i++) {
        // x coordinate
        for (int j = 0; j < 8; j++) {
            pubkey[i * 8 + j] = (uint8_t)(pub_x[idx].limbs[i] >> (j * 8));
        }
        // y coordinate
        for (int j = 0; j < 8; j++) {
            pubkey[32 + i * 8 + j] = (uint8_t)(pub_y[idx].limbs[i] >> (j * 8));
        }
    }

    // Keccak256(pubkey)
    uint8_t hash[32];
    keccak256_64bytes(pubkey, hash);

    // Take last 20 bytes as Ethereum address
    for (int i = 0; i < 20; i++) {
        addresses[idx * 20 + i] = hash[12 + i];
    }
}

// ============================================================================
// GTable Precomputation Kernel
// ============================================================================

extern "C" __global__
void precompute_gtable(
    AffinePoint* __restrict__ gtable,  // [16][65536] output
    const AffinePoint* __restrict__ generator
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= GTABLE_CHUNKS * GTABLE_CHUNK_SIZE) return;

    uint32_t chunk_idx = idx / GTABLE_CHUNK_SIZE;
    uint32_t point_idx = idx % GTABLE_CHUNK_SIZE;

    // base = 2^(16*chunk_idx) * G
    Fp256 one = {{1, 0, 0, 0}};
    JacobianPoint base = {generator->x, generator->y, one};

    // Double base (16 * chunk_idx) times
    for (uint32_t i = 0; i < chunk_idx * 16; i++) {
        base = point_double(base);
    }

    // Compute (point_idx + 1) * base
    JacobianPoint result = base;
    for (uint32_t i = 0; i < point_idx; i++) {
        result = point_add(result, base);
    }

    gtable[idx] = jacobian_to_affine(result);
}

// ============================================================================
// Parallel Reduction for Point Aggregation
// ============================================================================

extern "C" __global__
void parallel_point_reduce(
    JacobianPoint* __restrict__ points,
    uint32_t n
) {
    extern __shared__ JacobianPoint shared[];

    uint32_t tid = threadIdx.x;
    uint32_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load two points per thread
    if (idx < n) {
        shared[tid] = points[idx];
    } else {
        shared[tid] = jacobian_identity();
    }

    if (idx + blockDim.x < n) {
        shared[tid] = point_add(shared[tid], points[idx + blockDim.x]);
    }

    __syncthreads();

    // Tree reduction
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = point_add(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        points[blockIdx.x] = shared[0];
    }
}

// ============================================================================
// Host-Side Helper Functions
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

// Initialize CUDA context and allocate GTable
cudaError_t secp256k1_cuda_init(
    AffinePoint** gtable_device,
    size_t* gtable_size
) {
    *gtable_size = GTABLE_CHUNKS * GTABLE_CHUNK_SIZE * sizeof(AffinePoint);
    return cudaMalloc((void**)gtable_device, *gtable_size);
}

// Free CUDA resources
cudaError_t secp256k1_cuda_cleanup(AffinePoint* gtable_device) {
    return cudaFree(gtable_device);
}

// Batch scalar multiplication
cudaError_t secp256k1_batch_scalar_mul(
    const AffinePoint* gtable_device,
    const Fp256* scalars_host,
    AffinePoint* results_host,
    uint32_t count
) {
    Fp256* scalars_device;
    AffinePoint* results_device;

    cudaMalloc(&scalars_device, count * sizeof(Fp256));
    cudaMalloc(&results_device, count * sizeof(AffinePoint));

    cudaMemcpy(scalars_device, scalars_host, count * sizeof(Fp256), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (count + threads - 1) / threads;

    gtable_scalar_mul<<<blocks, threads>>>(
        gtable_device, scalars_device, results_device, count
    );

    cudaMemcpy(results_host, results_device, count * sizeof(AffinePoint), cudaMemcpyDeviceToHost);

    cudaFree(scalars_device);
    cudaFree(results_device);

    return cudaGetLastError();
}

#ifdef __cplusplus
}
#endif
