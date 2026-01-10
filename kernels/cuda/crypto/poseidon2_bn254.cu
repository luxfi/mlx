// Copyright (c) 2024-2026 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Poseidon2 Hash Function - BN254 Scalar Field CUDA Implementation
// GPU-accelerated Poseidon2 hash function over BN254 scalar field.
// Optimized for ZK-SNARK and STARK proof systems.
//
// Poseidon2 Parameters (t=3, 2-to-1 compression):
//   - State width: t = 3
//   - Full rounds: RF = 8 (4 at start, 4 at end)
//   - Partial rounds: RP = 56
//   - S-box: x^5 (quintic)
//   - Field: BN254 scalar field (Fr)
//
// BN254 Scalar Field:
//   r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
//
// References:
//   - Poseidon2: https://eprint.iacr.org/2023/323
//   - Original Poseidon: https://eprint.iacr.org/2019/458
//   - BN254 curve specification

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

namespace lux {
namespace cuda {
namespace poseidon2 {

// =============================================================================
// BN254 Scalar Field Constants (254-bit prime, 4 x 64-bit limbs)
// =============================================================================

// BN254 scalar field modulus r (little-endian)
// r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
__constant__ uint64_t BN254_R[4] = {
    0x43E1F593F0000001ULL,  // limb 0
    0x2833E84879B97091ULL,  // limb 1
    0xB85045B68181585DULL,  // limb 2
    0x30644E72E131A029ULL   // limb 3
};

// Montgomery R^2 mod r for converting to Montgomery form
__constant__ uint64_t BN254_R2[4] = {
    0x1BB8E645AE216DA7ULL,
    0x53FE3AB1E35C59E3ULL,
    0x8C49833D53BB8085ULL,
    0x0216D0B17F4E44A5ULL
};

// Montgomery constant: -r^{-1} mod 2^64
__constant__ uint64_t BN254_R_INV = 0xC2E1F593EFFFFFFULL;

// Poseidon2 round constants (RF + RP rounds * t elements)
// Total: (8 + 56) * 3 = 192 field elements for full rounds
// Plus: 56 elements for partial rounds (only first element)
// Precomputed in Montgomery form
__constant__ uint64_t ROUND_CONSTANTS[4 * 248];  // 248 field elements * 4 limbs

// Poseidon2 MDS matrix for t=3 (3x3 matrix, 9 elements in Montgomery form)
// Using the optimized Poseidon2 matrix
__constant__ uint64_t MDS_MATRIX[4 * 9];

// Poseidon2 internal matrix for partial rounds (diagonal + low-rank)
__constant__ uint64_t INTERNAL_MATRIX_DIAG[4 * 3];  // Diagonal elements

// =============================================================================
// 256-bit Field Element (4 x 64-bit limbs, little-endian)
// =============================================================================

struct Fr {
    uint64_t limbs[4];
};

// =============================================================================
// Multi-precision Arithmetic Primitives
// =============================================================================

// Add with carry
__device__ __forceinline__ uint64_t adc(uint64_t a, uint64_t b, uint64_t& carry) {
    uint64_t result = a + carry;
    carry = (result < a) ? 1ULL : 0ULL;
    uint64_t sum = result + b;
    carry += (sum < result) ? 1ULL : 0ULL;
    return sum;
}

// Subtract with borrow
__device__ __forceinline__ uint64_t sbb(uint64_t a, uint64_t b, uint64_t& borrow) {
    uint64_t diff = a - borrow;
    borrow = (a < borrow) ? 1ULL : 0ULL;
    uint64_t result = diff - b;
    borrow += (diff < b) ? 1ULL : 0ULL;
    return result;
}

// 64x64 -> 128-bit multiply using PTX
__device__ __forceinline__ void mul64(uint64_t a, uint64_t b, uint64_t& lo, uint64_t& hi) {
    asm("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a), "l"(b));
    asm("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a), "l"(b));
}

// Multiply-add: a*b + c -> (lo, hi)
__device__ __forceinline__ void mac(uint64_t a, uint64_t b, uint64_t c,
                                     uint64_t& lo, uint64_t& hi) {
    uint64_t tlo, thi;
    mul64(a, b, tlo, thi);
    lo = tlo + c;
    hi = thi + (lo < tlo ? 1ULL : 0ULL);
}

// =============================================================================
// Field Arithmetic
// =============================================================================

// Zero element
__device__ __forceinline__ Fr fr_zero() {
    Fr r;
    r.limbs[0] = 0; r.limbs[1] = 0;
    r.limbs[2] = 0; r.limbs[3] = 0;
    return r;
}

// One in Montgomery form
__device__ __forceinline__ Fr fr_one() {
    Fr r;
    // R mod r (Montgomery form of 1)
    r.limbs[0] = 0xAC96341C4FFFFFFBULL;
    r.limbs[1] = 0x36FC76959F60CD29ULL;
    r.limbs[2] = 0x666EA36F7879462EULL;
    r.limbs[3] = 0x0E0A77C19A07DF2FULL;
    return r;
}

// Load constant from __constant__ memory
__device__ __forceinline__ Fr fr_load_constant(const uint64_t* ptr) {
    Fr r;
    r.limbs[0] = ptr[0];
    r.limbs[1] = ptr[1];
    r.limbs[2] = ptr[2];
    r.limbs[3] = ptr[3];
    return r;
}

// Check if a >= r
__device__ __forceinline__ bool fr_gte_modulus(const Fr& a) {
    for (int i = 3; i >= 0; i--) {
        if (a.limbs[i] > BN254_R[i]) return true;
        if (a.limbs[i] < BN254_R[i]) return false;
    }
    return true;  // Equal
}

// Reduce mod r if needed
__device__ __forceinline__ Fr fr_reduce(const Fr& a) {
    if (!fr_gte_modulus(a)) return a;

    Fr result;
    uint64_t borrow = 0;
    result.limbs[0] = sbb(a.limbs[0], BN254_R[0], borrow);
    result.limbs[1] = sbb(a.limbs[1], BN254_R[1], borrow);
    result.limbs[2] = sbb(a.limbs[2], BN254_R[2], borrow);
    result.limbs[3] = sbb(a.limbs[3], BN254_R[3], borrow);
    return result;
}

// Add two field elements
__device__ Fr fr_add(const Fr& a, const Fr& b) {
    Fr result;
    uint64_t carry = 0;

    result.limbs[0] = adc(a.limbs[0], b.limbs[0], carry);
    result.limbs[1] = adc(a.limbs[1], b.limbs[1], carry);
    result.limbs[2] = adc(a.limbs[2], b.limbs[2], carry);
    result.limbs[3] = adc(a.limbs[3], b.limbs[3], carry);

    // Reduce if overflow or >= r
    if (carry != 0 || fr_gte_modulus(result)) {
        uint64_t borrow = 0;
        result.limbs[0] = sbb(result.limbs[0], BN254_R[0], borrow);
        result.limbs[1] = sbb(result.limbs[1], BN254_R[1], borrow);
        result.limbs[2] = sbb(result.limbs[2], BN254_R[2], borrow);
        result.limbs[3] = sbb(result.limbs[3], BN254_R[3], borrow);
    }

    return result;
}

// Subtract two field elements
__device__ Fr fr_sub(const Fr& a, const Fr& b) {
    Fr result;
    uint64_t borrow = 0;

    result.limbs[0] = sbb(a.limbs[0], b.limbs[0], borrow);
    result.limbs[1] = sbb(a.limbs[1], b.limbs[1], borrow);
    result.limbs[2] = sbb(a.limbs[2], b.limbs[2], borrow);
    result.limbs[3] = sbb(a.limbs[3], b.limbs[3], borrow);

    // If underflow, add modulus back
    if (borrow != 0) {
        uint64_t carry = 0;
        result.limbs[0] = adc(result.limbs[0], BN254_R[0], carry);
        result.limbs[1] = adc(result.limbs[1], BN254_R[1], carry);
        result.limbs[2] = adc(result.limbs[2], BN254_R[2], carry);
        result.limbs[3] = adc(result.limbs[3], BN254_R[3], carry);
    }

    return result;
}

// Montgomery multiplication using CIOS (Coarsely Integrated Operand Scanning)
__device__ Fr fr_mont_mul(const Fr& a, const Fr& b) {
    uint64_t t[5] = {0, 0, 0, 0, 0};

    // Schoolbook multiplication with interleaved Montgomery reduction
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        // Multiply-accumulate column
        uint64_t carry = 0;
        uint64_t hi, lo;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            mac(a.limbs[j], b.limbs[i], t[j], lo, hi);
            t[j] = lo + carry;
            carry = hi + (t[j] < lo ? 1ULL : 0ULL);
        }
        t[4] += carry;

        // Montgomery reduction step
        uint64_t m = t[0] * BN254_R_INV;

        carry = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            mac(m, BN254_R[j], t[j], lo, hi);
            if (j > 0) {
                t[j-1] = lo + carry;
                carry = hi + (t[j-1] < lo ? 1ULL : 0ULL);
            } else {
                // First limb becomes zero, just propagate carry
                carry = hi + (lo != 0 ? 0ULL : 0ULL);
                uint64_t sum = lo + carry;
                carry = hi + (sum < lo ? 1ULL : 0ULL);
            }
        }
        t[3] = t[4] + carry;
        t[4] = 0;
    }

    // Final reduction
    Fr result;
    result.limbs[0] = t[0];
    result.limbs[1] = t[1];
    result.limbs[2] = t[2];
    result.limbs[3] = t[3];

    return fr_reduce(result);
}

// Optimized CIOS Montgomery multiplication
__device__ Fr fr_mont_mul_cios(const Fr& a, const Fr& b) {
    uint64_t t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0;
    uint64_t lo, hi, carry;

    // Iteration 0
    mul64(a.limbs[0], b.limbs[0], t0, t1);
    mul64(a.limbs[1], b.limbs[0], lo, hi);
    t1 += lo; carry = (t1 < lo) ? 1ULL : 0ULL;
    t2 = hi + carry;
    mul64(a.limbs[2], b.limbs[0], lo, hi);
    t2 += lo; carry = (t2 < lo) ? 1ULL : 0ULL;
    t3 = hi + carry;
    mul64(a.limbs[3], b.limbs[0], lo, hi);
    t3 += lo; carry = (t3 < lo) ? 1ULL : 0ULL;
    t4 = hi + carry;

    uint64_t m = t0 * BN254_R_INV;
    mul64(m, BN254_R[0], lo, hi);
    carry = (t0 + lo < t0) ? 1ULL : 0ULL;
    carry += hi;

    mul64(m, BN254_R[1], lo, hi);
    t0 = t1 + lo + carry;
    carry = ((t1 + lo) < t1 || t0 < carry) ? 1ULL : 0ULL;
    carry += hi;

    mul64(m, BN254_R[2], lo, hi);
    t1 = t2 + lo + carry;
    carry = ((t2 + lo) < t2 || t1 < carry) ? 1ULL : 0ULL;
    carry += hi;

    mul64(m, BN254_R[3], lo, hi);
    t2 = t3 + lo + carry;
    carry = ((t3 + lo) < t3 || t2 < carry) ? 1ULL : 0ULL;
    t3 = t4 + hi + carry;
    t4 = (t3 < t4 + hi + carry) ? 1ULL : 0ULL;

    // Iterations 1-3 (similar pattern)
    #pragma unroll
    for (int i = 1; i < 4; i++) {
        mul64(a.limbs[0], b.limbs[i], lo, hi);
        t0 += lo; carry = (t0 < lo) ? 1ULL : 0ULL;
        uint64_t tmp = hi + carry;

        mul64(a.limbs[1], b.limbs[i], lo, hi);
        t1 += lo; carry = (t1 < lo) ? 1ULL : 0ULL;
        t1 += tmp; carry += (t1 < tmp) ? 1ULL : 0ULL;
        tmp = hi + carry;

        mul64(a.limbs[2], b.limbs[i], lo, hi);
        t2 += lo; carry = (t2 < lo) ? 1ULL : 0ULL;
        t2 += tmp; carry += (t2 < tmp) ? 1ULL : 0ULL;
        tmp = hi + carry;

        mul64(a.limbs[3], b.limbs[i], lo, hi);
        t3 += lo; carry = (t3 < lo) ? 1ULL : 0ULL;
        t3 += tmp; carry += (t3 < tmp) ? 1ULL : 0ULL;
        t4 += hi + carry;

        m = t0 * BN254_R_INV;
        mul64(m, BN254_R[0], lo, hi);
        carry = (t0 + lo < t0) ? 1ULL : 0ULL;
        carry += hi;

        mul64(m, BN254_R[1], lo, hi);
        t0 = t1 + lo + carry;
        carry = ((t1 + lo) < t1 || t0 < carry) ? 1ULL : 0ULL;
        carry += hi;

        mul64(m, BN254_R[2], lo, hi);
        t1 = t2 + lo + carry;
        carry = ((t2 + lo) < t2 || t1 < carry) ? 1ULL : 0ULL;
        carry += hi;

        mul64(m, BN254_R[3], lo, hi);
        t2 = t3 + lo + carry;
        carry = ((t3 + lo) < t3 || t2 < carry) ? 1ULL : 0ULL;
        t3 = t4 + hi + carry;
        t4 = (t3 < t4 + hi + carry) ? 1ULL : 0ULL;
    }

    Fr result;
    result.limbs[0] = t0;
    result.limbs[1] = t1;
    result.limbs[2] = t2;
    result.limbs[3] = t3;

    return fr_reduce(result);
}

// Square (optimized)
__device__ __forceinline__ Fr fr_square(const Fr& a) {
    return fr_mont_mul_cios(a, a);
}

// Convert to Montgomery form: a * R^2 * R^{-1} = a * R
__device__ Fr fr_to_mont(const Fr& a) {
    Fr r2;
    r2.limbs[0] = BN254_R2[0];
    r2.limbs[1] = BN254_R2[1];
    r2.limbs[2] = BN254_R2[2];
    r2.limbs[3] = BN254_R2[3];
    return fr_mont_mul_cios(a, r2);
}

// Convert from Montgomery form: a * 1 * R^{-1} = a * R^{-1}
__device__ Fr fr_from_mont(const Fr& a) {
    Fr one;
    one.limbs[0] = 1;
    one.limbs[1] = 0;
    one.limbs[2] = 0;
    one.limbs[3] = 0;
    return fr_mont_mul_cios(a, one);
}

// =============================================================================
// Poseidon2 S-box: x^5
// =============================================================================

__device__ __forceinline__ Fr sbox(const Fr& x) {
    Fr x2 = fr_square(x);
    Fr x4 = fr_square(x2);
    return fr_mont_mul_cios(x, x4);
}

// =============================================================================
// Poseidon2 MDS Matrix Multiplication (t=3)
// =============================================================================

// External MDS matrix multiplication for full rounds
// Using optimized 3x3 circulant-like matrix
__device__ void mds_multiply(Fr& s0, Fr& s1, Fr& s2) {
    // Poseidon2 uses an efficient matrix structure
    // M_E = circ(2, 1, 1) for t=3
    // [2 1 1]
    // [1 2 1]
    // [1 1 2]

    Fr t0 = fr_add(s0, s1);
    t0 = fr_add(t0, s2);  // sum = s0 + s1 + s2

    Fr r0 = fr_add(s0, t0);  // 2*s0 + s1 + s2
    Fr r1 = fr_add(s1, t0);  // s0 + 2*s1 + s2
    Fr r2 = fr_add(s2, t0);  // s0 + s1 + 2*s2

    s0 = r0;
    s1 = r1;
    s2 = r2;
}

// Internal matrix multiplication for partial rounds
// Uses diagonal matrix with perturbation: M_I = D + v*w^T
// For efficiency, Poseidon2 uses M_I where only s0 is updated with full mixing
__device__ void internal_mds_multiply(Fr& s0, Fr& s1, Fr& s2) {
    // For partial rounds, use simpler internal diffusion
    // s0 = s0 + s1 + s2 (mix all into first element)
    // Others remain with minimal update for efficiency

    Fr sum = fr_add(s0, s1);
    sum = fr_add(sum, s2);

    // Update with diagonal multipliers from constant memory
    Fr d0 = fr_load_constant(&INTERNAL_MATRIX_DIAG[0]);
    Fr d1 = fr_load_constant(&INTERNAL_MATRIX_DIAG[4]);
    Fr d2 = fr_load_constant(&INTERNAL_MATRIX_DIAG[8]);

    s0 = fr_mont_mul_cios(sum, d0);
    s1 = fr_add(s1, fr_mont_mul_cios(sum, d1));
    s2 = fr_add(s2, fr_mont_mul_cios(sum, d2));
}

// =============================================================================
// Poseidon2 Permutation
// =============================================================================

// Full Poseidon2 permutation with t=3, RF=8, RP=56
__device__ void poseidon2_permutation(Fr& s0, Fr& s1, Fr& s2) {
    const int RF = 8;   // Full rounds (4 + 4)
    const int RP = 56;  // Partial rounds

    int rc_idx = 0;

    // Initial linear layer (external matrix)
    mds_multiply(s0, s1, s2);

    // First RF/2 = 4 full rounds
    #pragma unroll 4
    for (int r = 0; r < RF / 2; r++) {
        // Add round constants
        s0 = fr_add(s0, fr_load_constant(&ROUND_CONSTANTS[rc_idx * 4]));
        s1 = fr_add(s1, fr_load_constant(&ROUND_CONSTANTS[(rc_idx + 1) * 4]));
        s2 = fr_add(s2, fr_load_constant(&ROUND_CONSTANTS[(rc_idx + 2) * 4]));
        rc_idx += 3;

        // Full S-box layer
        s0 = sbox(s0);
        s1 = sbox(s1);
        s2 = sbox(s2);

        // External MDS
        mds_multiply(s0, s1, s2);
    }

    // RP = 56 partial rounds
    #pragma unroll 8
    for (int r = 0; r < RP; r++) {
        // Add round constant only to first element
        s0 = fr_add(s0, fr_load_constant(&ROUND_CONSTANTS[rc_idx * 4]));
        rc_idx++;

        // S-box only on first element
        s0 = sbox(s0);

        // Internal MDS (optimized for partial rounds)
        internal_mds_multiply(s0, s1, s2);
    }

    // Last RF/2 = 4 full rounds
    #pragma unroll 4
    for (int r = 0; r < RF / 2; r++) {
        // Add round constants
        s0 = fr_add(s0, fr_load_constant(&ROUND_CONSTANTS[rc_idx * 4]));
        s1 = fr_add(s1, fr_load_constant(&ROUND_CONSTANTS[(rc_idx + 1) * 4]));
        s2 = fr_add(s2, fr_load_constant(&ROUND_CONSTANTS[(rc_idx + 2) * 4]));
        rc_idx += 3;

        // Full S-box layer
        s0 = sbox(s0);
        s1 = sbox(s1);
        s2 = sbox(s2);

        // External MDS
        mds_multiply(s0, s1, s2);
    }
}

// =============================================================================
// Poseidon2 Hash Kernels
// =============================================================================

// 2-to-1 compression: hash(left, right) -> digest
__global__ void poseidon2_hash_2to1_kernel(
    const uint64_t* __restrict__ inputs,  // 2 field elements per hash (8 uint64 each)
    uint64_t* __restrict__ outputs,        // 1 field element per hash (4 uint64)
    uint32_t num_hashes
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_hashes) return;

    // Load inputs (already in Montgomery form expected)
    Fr s0 = fr_zero();  // Capacity element
    Fr s1, s2;

    const uint64_t* in_ptr = &inputs[idx * 8];
    s1.limbs[0] = in_ptr[0];
    s1.limbs[1] = in_ptr[1];
    s1.limbs[2] = in_ptr[2];
    s1.limbs[3] = in_ptr[3];
    s2.limbs[0] = in_ptr[4];
    s2.limbs[1] = in_ptr[5];
    s2.limbs[2] = in_ptr[6];
    s2.limbs[3] = in_ptr[7];

    // Apply permutation
    poseidon2_permutation(s0, s1, s2);

    // Output is s1 (first rate element after permutation)
    uint64_t* out_ptr = &outputs[idx * 4];
    out_ptr[0] = s1.limbs[0];
    out_ptr[1] = s1.limbs[1];
    out_ptr[2] = s1.limbs[2];
    out_ptr[3] = s1.limbs[3];
}

// Single element hash with domain separator
__global__ void poseidon2_hash_single_kernel(
    const uint64_t* __restrict__ inputs,   // 1 field element per hash
    const uint64_t* __restrict__ domain,   // Domain separator (1 field element)
    uint64_t* __restrict__ outputs,
    uint32_t num_hashes
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_hashes) return;

    // State: [domain, input, 0]
    Fr s0, s1, s2;
    s0.limbs[0] = domain[0];
    s0.limbs[1] = domain[1];
    s0.limbs[2] = domain[2];
    s0.limbs[3] = domain[3];

    const uint64_t* in_ptr = &inputs[idx * 4];
    s1.limbs[0] = in_ptr[0];
    s1.limbs[1] = in_ptr[1];
    s1.limbs[2] = in_ptr[2];
    s1.limbs[3] = in_ptr[3];

    s2 = fr_zero();

    poseidon2_permutation(s0, s1, s2);

    uint64_t* out_ptr = &outputs[idx * 4];
    out_ptr[0] = s1.limbs[0];
    out_ptr[1] = s1.limbs[1];
    out_ptr[2] = s1.limbs[2];
    out_ptr[3] = s1.limbs[3];
}

// =============================================================================
// Merkle Tree Hashing (Batch Mode)
// =============================================================================

// Hash one layer of Merkle tree: pairs of leaves -> parent nodes
__global__ void poseidon2_merkle_layer_kernel(
    const uint64_t* __restrict__ leaves,   // 2N field elements
    uint64_t* __restrict__ parents,        // N field elements
    uint32_t num_pairs
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    // Load left and right children
    Fr s0 = fr_zero();
    Fr s1, s2;

    const uint64_t* left = &leaves[idx * 8];
    const uint64_t* right = &leaves[idx * 8 + 4];

    s1.limbs[0] = left[0]; s1.limbs[1] = left[1];
    s1.limbs[2] = left[2]; s1.limbs[3] = left[3];
    s2.limbs[0] = right[0]; s2.limbs[1] = right[1];
    s2.limbs[2] = right[2]; s2.limbs[3] = right[3];

    poseidon2_permutation(s0, s1, s2);

    uint64_t* out = &parents[idx * 4];
    out[0] = s1.limbs[0]; out[1] = s1.limbs[1];
    out[2] = s1.limbs[2]; out[3] = s1.limbs[3];
}

// Build complete Merkle tree from leaves
__global__ void poseidon2_merkle_tree_kernel(
    uint64_t* __restrict__ tree,   // Tree storage: leaves at bottom, root at top
    uint32_t num_leaves,
    uint32_t layer_offset_in,      // Offset to current layer
    uint32_t layer_offset_out,     // Offset to parent layer
    uint32_t num_pairs
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    Fr s0 = fr_zero();
    Fr s1, s2;

    const uint64_t* left = &tree[(layer_offset_in + idx * 2) * 4];
    const uint64_t* right = &tree[(layer_offset_in + idx * 2 + 1) * 4];

    s1.limbs[0] = left[0]; s1.limbs[1] = left[1];
    s1.limbs[2] = left[2]; s1.limbs[3] = left[3];
    s2.limbs[0] = right[0]; s2.limbs[1] = right[1];
    s2.limbs[2] = right[2]; s2.limbs[3] = right[3];

    poseidon2_permutation(s0, s1, s2);

    uint64_t* out = &tree[(layer_offset_out + idx) * 4];
    out[0] = s1.limbs[0]; out[1] = s1.limbs[1];
    out[2] = s1.limbs[2]; out[3] = s1.limbs[3];
}

// =============================================================================
// Sponge Construction for Variable-Length Input
// =============================================================================

// Absorb phase: process input blocks into sponge state
__global__ void poseidon2_sponge_absorb_kernel(
    const uint64_t* __restrict__ inputs,   // Variable-length input (multiple of rate=2)
    uint64_t* __restrict__ states,         // Sponge states (3 field elements each)
    const uint32_t* __restrict__ lengths,  // Number of field elements per input
    uint32_t num_inputs
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_inputs) return;

    uint32_t len = lengths[idx];
    uint32_t num_blocks = (len + 1) / 2;  // Rate = 2

    // Load initial state
    Fr s0, s1, s2;
    s0 = fr_zero();
    s1 = fr_zero();
    s2 = fr_zero();

    // Calculate input offset (sum of previous lengths)
    uint32_t offset = 0;
    for (uint32_t i = 0; i < idx; i++) {
        offset += lengths[i];
    }

    // Absorb input blocks
    for (uint32_t b = 0; b < num_blocks; b++) {
        // XOR input into rate portion of state
        uint32_t elem_idx = b * 2;

        if (elem_idx < len) {
            Fr input1;
            const uint64_t* in1 = &inputs[(offset + elem_idx) * 4];
            input1.limbs[0] = in1[0]; input1.limbs[1] = in1[1];
            input1.limbs[2] = in1[2]; input1.limbs[3] = in1[3];
            s1 = fr_add(s1, input1);
        }

        if (elem_idx + 1 < len) {
            Fr input2;
            const uint64_t* in2 = &inputs[(offset + elem_idx + 1) * 4];
            input2.limbs[0] = in2[0]; input2.limbs[1] = in2[1];
            input2.limbs[2] = in2[2]; input2.limbs[3] = in2[3];
            s2 = fr_add(s2, input2);
        }

        // Apply permutation
        poseidon2_permutation(s0, s1, s2);
    }

    // Store final state
    uint64_t* state_out = &states[idx * 12];
    state_out[0] = s0.limbs[0]; state_out[1] = s0.limbs[1];
    state_out[2] = s0.limbs[2]; state_out[3] = s0.limbs[3];
    state_out[4] = s1.limbs[0]; state_out[5] = s1.limbs[1];
    state_out[6] = s1.limbs[2]; state_out[7] = s1.limbs[3];
    state_out[8] = s2.limbs[0]; state_out[9] = s2.limbs[1];
    state_out[10] = s2.limbs[2]; state_out[11] = s2.limbs[3];
}

// Squeeze phase: extract output from sponge state
__global__ void poseidon2_sponge_squeeze_kernel(
    const uint64_t* __restrict__ states,   // Sponge states
    uint64_t* __restrict__ outputs,        // Output field elements
    uint32_t num_outputs_per_state,        // Number of elements to squeeze
    uint32_t num_states
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states) return;

    // Load state
    Fr s0, s1, s2;
    const uint64_t* state_in = &states[idx * 12];
    s0.limbs[0] = state_in[0]; s0.limbs[1] = state_in[1];
    s0.limbs[2] = state_in[2]; s0.limbs[3] = state_in[3];
    s1.limbs[0] = state_in[4]; s1.limbs[1] = state_in[5];
    s1.limbs[2] = state_in[6]; s1.limbs[3] = state_in[7];
    s2.limbs[0] = state_in[8]; s2.limbs[1] = state_in[9];
    s2.limbs[2] = state_in[10]; s2.limbs[3] = state_in[11];

    uint64_t* out = &outputs[idx * num_outputs_per_state * 4];

    // Squeeze outputs
    for (uint32_t o = 0; o < num_outputs_per_state; o += 2) {
        // Output rate elements
        if (o < num_outputs_per_state) {
            out[o * 4 + 0] = s1.limbs[0];
            out[o * 4 + 1] = s1.limbs[1];
            out[o * 4 + 2] = s1.limbs[2];
            out[o * 4 + 3] = s1.limbs[3];
        }
        if (o + 1 < num_outputs_per_state) {
            out[(o + 1) * 4 + 0] = s2.limbs[0];
            out[(o + 1) * 4 + 1] = s2.limbs[1];
            out[(o + 1) * 4 + 2] = s2.limbs[2];
            out[(o + 1) * 4 + 3] = s2.limbs[3];
        }

        // Permute for next squeeze
        if (o + 2 < num_outputs_per_state) {
            poseidon2_permutation(s0, s1, s2);
        }
    }
}

// =============================================================================
// Round Constants Generation (Host-side utility)
// =============================================================================

// Grain LFSR for round constant generation (called on host)
struct GrainLFSR {
    uint8_t state[80];

    void init(uint8_t seed[80]) {
        memcpy(state, seed, 80);
    }

    uint8_t get_bit() {
        uint8_t new_bit = state[62] ^ state[51] ^ state[38] ^ state[23] ^ state[13] ^ state[0];
        for (int i = 0; i < 79; i++) {
            state[i] = state[i + 1];
        }
        state[79] = new_bit;
        return new_bit;
    }

    void get_bits(uint8_t* out, int n) {
        for (int i = 0; i < n; i++) {
            out[i] = get_bit();
        }
    }
};

} // namespace poseidon2
} // namespace cuda
} // namespace lux

// =============================================================================
// C API for Go/CGO Bindings
// =============================================================================

extern "C" {

using namespace lux::cuda::poseidon2;

// Initialize round constants (must be called before hashing)
int lux_cuda_poseidon2_init(
    const uint64_t* round_constants,   // 248 field elements * 4 limbs
    const uint64_t* mds_matrix,        // 9 field elements * 4 limbs
    const uint64_t* internal_diag      // 3 field elements * 4 limbs
) {
    cudaError_t err;

    err = cudaMemcpyToSymbol(ROUND_CONSTANTS, round_constants,
                             248 * 4 * sizeof(uint64_t));
    if (err != cudaSuccess) return err;

    err = cudaMemcpyToSymbol(MDS_MATRIX, mds_matrix,
                             9 * 4 * sizeof(uint64_t));
    if (err != cudaSuccess) return err;

    err = cudaMemcpyToSymbol(INTERNAL_MATRIX_DIAG, internal_diag,
                             3 * 4 * sizeof(uint64_t));
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

// 2-to-1 hash (compression function)
int lux_cuda_poseidon2_hash_2to1(
    const void* inputs,      // Device pointer: 2 field elements per hash
    void* outputs,           // Device pointer: 1 field element per hash
    uint32_t count,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((count + block.x - 1) / block.x);

    poseidon2_hash_2to1_kernel<<<grid, block, 0, stream>>>(
        (const uint64_t*)inputs,
        (uint64_t*)outputs,
        count
    );

    return cudaGetLastError();
}

// Single element hash with domain separator
int lux_cuda_poseidon2_hash_single(
    const void* inputs,      // Device pointer: 1 field element per hash
    const void* domain,      // Device pointer: 1 field element (domain separator)
    void* outputs,           // Device pointer: 1 field element per hash
    uint32_t count,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((count + block.x - 1) / block.x);

    poseidon2_hash_single_kernel<<<grid, block, 0, stream>>>(
        (const uint64_t*)inputs,
        (const uint64_t*)domain,
        (uint64_t*)outputs,
        count
    );

    return cudaGetLastError();
}

// Merkle tree layer hashing
int lux_cuda_poseidon2_merkle_layer(
    const void* leaves,      // Device pointer: 2N field elements
    void* parents,           // Device pointer: N field elements
    uint32_t num_pairs,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((num_pairs + block.x - 1) / block.x);

    poseidon2_merkle_layer_kernel<<<grid, block, 0, stream>>>(
        (const uint64_t*)leaves,
        (uint64_t*)parents,
        num_pairs
    );

    return cudaGetLastError();
}

// Build complete Merkle tree
int lux_cuda_poseidon2_merkle_tree(
    void* tree,              // Device pointer: complete tree storage
    uint32_t num_leaves,
    cudaStream_t stream
) {
    if (num_leaves == 0 || (num_leaves & (num_leaves - 1)) != 0) {
        return cudaErrorInvalidValue;  // Must be power of 2
    }

    dim3 block(256);

    // Build tree layer by layer
    uint32_t layer_size = num_leaves;
    uint32_t layer_offset_in = 0;
    uint32_t layer_offset_out = num_leaves;

    while (layer_size > 1) {
        uint32_t num_pairs = layer_size / 2;
        dim3 grid((num_pairs + block.x - 1) / block.x);

        poseidon2_merkle_tree_kernel<<<grid, block, 0, stream>>>(
            (uint64_t*)tree,
            num_leaves,
            layer_offset_in,
            layer_offset_out,
            num_pairs
        );

        layer_offset_in = layer_offset_out;
        layer_offset_out += num_pairs;
        layer_size = num_pairs;
    }

    return cudaGetLastError();
}

// Sponge-based variable-length hash
int lux_cuda_poseidon2_sponge(
    const void* inputs,          // Device pointer: concatenated field elements
    const uint32_t* lengths,     // Device pointer: length per input (in field elements)
    void* outputs,               // Device pointer: output field elements
    uint32_t num_inputs,
    uint32_t outputs_per_input,
    cudaStream_t stream
) {
    // Allocate temporary state storage
    uint64_t* d_states;
    cudaError_t err = cudaMalloc(&d_states, num_inputs * 12 * sizeof(uint64_t));
    if (err != cudaSuccess) return err;

    dim3 block(256);
    dim3 grid((num_inputs + block.x - 1) / block.x);

    // Absorb phase
    poseidon2_sponge_absorb_kernel<<<grid, block, 0, stream>>>(
        (const uint64_t*)inputs,
        d_states,
        lengths,
        num_inputs
    );

    // Squeeze phase
    poseidon2_sponge_squeeze_kernel<<<grid, block, 0, stream>>>(
        d_states,
        (uint64_t*)outputs,
        outputs_per_input,
        num_inputs
    );

    cudaFree(d_states);

    return cudaGetLastError();
}

// Batch permutation (for advanced use)
int lux_cuda_poseidon2_permutation(
    void* states,            // Device pointer: 3 field elements per state (in-place)
    uint32_t count,
    cudaStream_t stream
) {
    // Use 2-to-1 kernel structure but treat as raw permutation
    // State layout: [s0, s1, s2] repeated
    dim3 block(256);
    dim3 grid((count + block.x - 1) / block.x);

    // Note: This requires a separate kernel for raw permutation
    // For now, this is a placeholder - implement if needed

    return cudaGetLastError();
}

// Get field element size in bytes
uint32_t lux_cuda_poseidon2_element_size() {
    return 32;  // 4 x 64-bit limbs = 256 bits = 32 bytes
}

// Get rate (number of field elements absorbed per permutation)
uint32_t lux_cuda_poseidon2_rate() {
    return 2;  // t=3, capacity=1, rate=2
}

// Get capacity (security parameter)
uint32_t lux_cuda_poseidon2_capacity() {
    return 1;  // 1 field element = 254 bits of security
}

} // extern "C"
