// Copyright (c) 2024-2026 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// BN254 (alt_bn128) CUDA Kernels
// GPU-accelerated elliptic curve operations for BN254 on NVIDIA GPUs.
// Used for Pedersen commitments, PLONK verification, and Groth16 proofs.
//
// BN254 Parameters:
//   p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
//   r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
//   G1: y^2 = x^3 + 3  over Fp
//
// References:
//   - EIP-196, EIP-197 (Ethereum precompiles)
//   - Zcash BN-254 specification

#include <cstdint>
#include <cuda_runtime.h>

namespace lux {
namespace cuda {
namespace bn254 {

// =============================================================================
// 256-bit Field Arithmetic (4 x 64-bit limbs)
// =============================================================================

// BN254 base field prime p (4 limbs, little-endian)
__constant__ uint64_t BN254_P[4] = {
    0x3C208C16D87CFD47ULL,
    0x97816A916871CA8DULL,
    0xB85045B68181585DULL,
    0x30644E72E131A029ULL
};

// Montgomery R^2 mod p
__constant__ uint64_t BN254_R2[4] = {
    0xF32CFC5B538AFA89ULL,
    0xB5E71911D44501FBULL,
    0x47AB1EFF0A417FF6ULL,
    0x06D89F71CAB8351FULL
};

// Montgomery constant: -p^{-1} mod 2^64
__constant__ uint64_t BN254_INV = 0x87D20782E4866389ULL;

// Generator points (Montgomery form)
__constant__ uint64_t BN254_G1_X[4] = {
    0xD35D438DC58F0D9DULL,
    0x0A78EB28F5C70B3DULL,
    0x666EA36F7879462CULL,
    0x0E0A77C19A07DF2FULL
};

__constant__ uint64_t BN254_G1_Y[4] = {
    0xA6BA871B8B1E1B3AULL,
    0x14F1D651EB8E167BULL,
    0xCCDD46DEF0F28C58ULL,
    0x1C14EF83340FBE5EULL
};

// Fp256 represented as 4 uint64 limbs
struct Fp256 {
    uint64_t limbs[4];
};

// G1 affine point
struct G1Affine {
    Fp256 x;
    Fp256 y;
    bool infinity;
};

// G1 projective point (Jacobian coordinates)
struct G1Projective {
    Fp256 x;
    Fp256 y;
    Fp256 z;
};

// =============================================================================
// Multi-precision Arithmetic
// =============================================================================

__device__ __forceinline__ uint64_t adc(uint64_t a, uint64_t b, uint64_t& carry) {
    uint64_t result = a + carry;
    carry = (result < a) ? 1ULL : 0ULL;
    uint64_t sum = result + b;
    carry += (sum < result) ? 1ULL : 0ULL;
    return sum;
}

__device__ __forceinline__ uint64_t sbb(uint64_t a, uint64_t b, uint64_t& borrow) {
    uint64_t diff = a - borrow;
    borrow = (a < borrow) ? 1ULL : 0ULL;
    uint64_t result = diff - b;
    borrow += (diff < b) ? 1ULL : 0ULL;
    return result;
}

__device__ __forceinline__ void mul64(uint64_t a, uint64_t b, uint64_t& lo, uint64_t& hi) {
    // Use PTX for 64x64->128 multiply on NVIDIA GPUs
    asm("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a), "l"(b));
    asm("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a), "l"(b));
}

// Add two Fp256 values
__device__ Fp256 fp256_add(const Fp256& a, const Fp256& b) {
    Fp256 result;
    uint64_t carry = 0;

    result.limbs[0] = adc(a.limbs[0], b.limbs[0], carry);
    result.limbs[1] = adc(a.limbs[1], b.limbs[1], carry);
    result.limbs[2] = adc(a.limbs[2], b.limbs[2], carry);
    result.limbs[3] = adc(a.limbs[3], b.limbs[3], carry);

    // Reduce mod p if needed
    uint64_t borrow = 0;
    Fp256 reduced;
    reduced.limbs[0] = sbb(result.limbs[0], BN254_P[0], borrow);
    reduced.limbs[1] = sbb(result.limbs[1], BN254_P[1], borrow);
    reduced.limbs[2] = sbb(result.limbs[2], BN254_P[2], borrow);
    reduced.limbs[3] = sbb(result.limbs[3], BN254_P[3], borrow);

    // If no borrow, use reduced; otherwise use result
    if (carry != 0 || borrow == 0) {
        return reduced;
    }
    return result;
}

// Subtract two Fp256 values
__device__ Fp256 fp256_sub(const Fp256& a, const Fp256& b) {
    Fp256 result;
    uint64_t borrow = 0;

    result.limbs[0] = sbb(a.limbs[0], b.limbs[0], borrow);
    result.limbs[1] = sbb(a.limbs[1], b.limbs[1], borrow);
    result.limbs[2] = sbb(a.limbs[2], b.limbs[2], borrow);
    result.limbs[3] = sbb(a.limbs[3], b.limbs[3], borrow);

    // If borrow, add p back
    if (borrow != 0) {
        uint64_t carry = 0;
        result.limbs[0] = adc(result.limbs[0], BN254_P[0], carry);
        result.limbs[1] = adc(result.limbs[1], BN254_P[1], carry);
        result.limbs[2] = adc(result.limbs[2], BN254_P[2], carry);
        result.limbs[3] = adc(result.limbs[3], BN254_P[3], carry);
    }

    return result;
}

// Montgomery multiplication
__device__ Fp256 fp256_mont_mul(const Fp256& a, const Fp256& b) {
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
        uint64_t m;
        mul64(t[i], BN254_INV, m, m); // Only need low part
        m = t[i] * BN254_INV;

        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint64_t hi, lo;
            mul64(m, BN254_P[j], lo, hi);

            uint64_t sum = t[i + j] + lo + carry;
            carry = (sum < t[i + j]) ? 1ULL : 0ULL;
            carry += hi;
            t[i + j] = sum;
        }

        // Propagate carry
        for (int j = i + 4; j < 8 && carry != 0; j++) {
            uint64_t sum = t[j] + carry;
            carry = (sum < t[j]) ? 1ULL : 0ULL;
            t[j] = sum;
        }
    }

    // Result is in t[4..7]
    Fp256 result;
    result.limbs[0] = t[4];
    result.limbs[1] = t[5];
    result.limbs[2] = t[6];
    result.limbs[3] = t[7];

    // Final reduction
    uint64_t borrow = 0;
    Fp256 reduced;
    reduced.limbs[0] = sbb(result.limbs[0], BN254_P[0], borrow);
    reduced.limbs[1] = sbb(result.limbs[1], BN254_P[1], borrow);
    reduced.limbs[2] = sbb(result.limbs[2], BN254_P[2], borrow);
    reduced.limbs[3] = sbb(result.limbs[3], BN254_P[3], borrow);

    return (borrow == 0) ? reduced : result;
}

// =============================================================================
// G1 Point Operations
// =============================================================================

__device__ bool g1_is_zero(const G1Projective& p) {
    return p.z.limbs[0] == 0 && p.z.limbs[1] == 0 &&
           p.z.limbs[2] == 0 && p.z.limbs[3] == 0;
}

// Point doubling in Jacobian coordinates
__device__ G1Projective g1_double(const G1Projective& p) {
    if (g1_is_zero(p)) return p;

    // a = x^2
    Fp256 a = fp256_mont_mul(p.x, p.x);
    // b = y^2
    Fp256 b = fp256_mont_mul(p.y, p.y);
    // c = b^2
    Fp256 c = fp256_mont_mul(b, b);

    // d = 2*((x+b)^2 - a - c)
    Fp256 xpb = fp256_add(p.x, b);
    Fp256 xpb2 = fp256_mont_mul(xpb, xpb);
    Fp256 d = fp256_sub(fp256_sub(xpb2, a), c);
    d = fp256_add(d, d);

    // e = 3*a
    Fp256 e = fp256_add(fp256_add(a, a), a);

    // f = e^2
    Fp256 f = fp256_mont_mul(e, e);

    // x3 = f - 2*d
    G1Projective result;
    Fp256 d2 = fp256_add(d, d);
    result.x = fp256_sub(f, d2);

    // y3 = e*(d - x3) - 8*c
    Fp256 c8 = fp256_add(fp256_add(fp256_add(c, c), fp256_add(c, c)),
                         fp256_add(fp256_add(c, c), fp256_add(c, c)));
    result.y = fp256_sub(fp256_mont_mul(e, fp256_sub(d, result.x)), c8);

    // z3 = 2*y*z
    result.z = fp256_mont_mul(p.y, p.z);
    result.z = fp256_add(result.z, result.z);

    return result;
}

// Point addition in Jacobian coordinates
__device__ G1Projective g1_add(const G1Projective& p, const G1Projective& q) {
    if (g1_is_zero(p)) return q;
    if (g1_is_zero(q)) return p;

    // z1z1 = z1^2
    Fp256 z1z1 = fp256_mont_mul(p.z, p.z);
    // z2z2 = z2^2
    Fp256 z2z2 = fp256_mont_mul(q.z, q.z);

    // u1 = x1 * z2z2
    Fp256 u1 = fp256_mont_mul(p.x, z2z2);
    // u2 = x2 * z1z1
    Fp256 u2 = fp256_mont_mul(q.x, z1z1);

    // s1 = y1 * z2 * z2z2
    Fp256 s1 = fp256_mont_mul(fp256_mont_mul(p.y, q.z), z2z2);
    // s2 = y2 * z1 * z1z1
    Fp256 s2 = fp256_mont_mul(fp256_mont_mul(q.y, p.z), z1z1);

    // h = u2 - u1
    Fp256 h = fp256_sub(u2, u1);
    // r = s2 - s1
    Fp256 r = fp256_sub(s2, s1);

    // Check for doubling case
    bool h_zero = (h.limbs[0] == 0 && h.limbs[1] == 0 &&
                   h.limbs[2] == 0 && h.limbs[3] == 0);
    bool r_zero = (r.limbs[0] == 0 && r.limbs[1] == 0 &&
                   r.limbs[2] == 0 && r.limbs[3] == 0);

    if (h_zero && r_zero) {
        return g1_double(p);
    }

    // hh = h^2
    Fp256 hh = fp256_mont_mul(h, h);
    // hhh = h * hh
    Fp256 hhh = fp256_mont_mul(h, hh);

    // v = u1 * hh
    Fp256 v = fp256_mont_mul(u1, hh);

    // x3 = r^2 - hhh - 2*v
    G1Projective result;
    Fp256 r2 = fp256_mont_mul(r, r);
    Fp256 v2 = fp256_add(v, v);
    result.x = fp256_sub(fp256_sub(r2, hhh), v2);

    // y3 = r * (v - x3) - s1 * hhh
    result.y = fp256_sub(fp256_mont_mul(r, fp256_sub(v, result.x)),
                         fp256_mont_mul(s1, hhh));

    // z3 = z1 * z2 * h
    result.z = fp256_mont_mul(fp256_mont_mul(p.z, q.z), h);

    return result;
}

// Scalar multiplication using double-and-add
__device__ G1Projective g1_scalar_mul(const G1Projective& p, const uint64_t* scalar) {
    G1Projective result;
    result.x.limbs[0] = 0; result.x.limbs[1] = 0;
    result.x.limbs[2] = 0; result.x.limbs[3] = 0;
    result.y.limbs[0] = 1; result.y.limbs[1] = 0;
    result.y.limbs[2] = 0; result.y.limbs[3] = 0;
    result.z.limbs[0] = 0; result.z.limbs[1] = 0;
    result.z.limbs[2] = 0; result.z.limbs[3] = 0;

    G1Projective temp = p;

    for (int i = 0; i < 4; i++) {
        uint64_t s = scalar[i];
        for (int j = 0; j < 64; j++) {
            if (s & 1) {
                result = g1_add(result, temp);
            }
            temp = g1_double(temp);
            s >>= 1;
        }
    }

    return result;
}

// =============================================================================
// CUDA Kernels
// =============================================================================

__global__ void bn254_g1_add_kernel(
    const G1Projective* points_a,
    const G1Projective* points_b,
    G1Projective* results,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    results[idx] = g1_add(points_a[idx], points_b[idx]);
}

__global__ void bn254_g1_double_kernel(
    const G1Projective* points,
    G1Projective* results,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    results[idx] = g1_double(points[idx]);
}

__global__ void bn254_g1_scalar_mul_kernel(
    const G1Projective* points,
    const uint64_t* scalars, // 4 uint64 per scalar
    G1Projective* results,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    results[idx] = g1_scalar_mul(points[idx], &scalars[idx * 4]);
}

__global__ void bn254_batch_msm_kernel(
    const G1Affine* bases,
    const uint64_t* scalars,
    G1Projective* result,
    uint32_t count
) {
    // Shared memory for partial sums
    __shared__ G1Projective partial[256];

    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint32_t gid = bid * blockDim.x + tid;

    // Initialize partial sum to zero
    partial[tid].x.limbs[0] = 0;
    partial[tid].y.limbs[0] = 1;
    partial[tid].z.limbs[0] = 0;

    // Each thread processes some points
    uint32_t points_per_thread = (count + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x);
    uint32_t start = gid * points_per_thread;
    uint32_t end = min(start + points_per_thread, count);

    for (uint32_t i = start; i < end; i++) {
        G1Projective p;
        p.x = bases[i].x;
        p.y = bases[i].y;
        p.z.limbs[0] = 1; p.z.limbs[1] = 0;
        p.z.limbs[2] = 0; p.z.limbs[3] = 0;

        G1Projective sp = g1_scalar_mul(p, &scalars[i * 4]);
        partial[tid] = g1_add(partial[tid], sp);
    }

    __syncthreads();

    // Reduction within block
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial[tid] = g1_add(partial[tid], partial[tid + s]);
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        result[bid] = partial[0];
    }
}

} // namespace bn254
} // namespace cuda
} // namespace lux

// =============================================================================
// C API for Go CGO bindings
// =============================================================================

extern "C" {

int lux_cuda_bn254_g1_add(
    const void* points_a,
    const void* points_b,
    void* results,
    uint32_t count,
    cudaStream_t stream
) {
    using namespace lux::cuda::bn254;

    dim3 block(256);
    dim3 grid((count + block.x - 1) / block.x);

    bn254_g1_add_kernel<<<grid, block, 0, stream>>>(
        (const G1Projective*)points_a,
        (const G1Projective*)points_b,
        (G1Projective*)results,
        count
    );

    return cudaGetLastError();
}

int lux_cuda_bn254_g1_scalar_mul(
    const void* points,
    const void* scalars,
    void* results,
    uint32_t count,
    cudaStream_t stream
) {
    using namespace lux::cuda::bn254;

    dim3 block(256);
    dim3 grid((count + block.x - 1) / block.x);

    bn254_g1_scalar_mul_kernel<<<grid, block, 0, stream>>>(
        (const G1Projective*)points,
        (const uint64_t*)scalars,
        (G1Projective*)results,
        count
    );

    return cudaGetLastError();
}

int lux_cuda_bn254_msm(
    const void* bases,
    const void* scalars,
    void* result,
    uint32_t count,
    cudaStream_t stream
) {
    using namespace lux::cuda::bn254;

    // Allocate temporary buffer for partial sums
    G1Projective* d_partials;
    uint32_t num_blocks = (count + 255) / 256;
    cudaMalloc(&d_partials, num_blocks * sizeof(G1Projective));

    dim3 block(256);
    dim3 grid(num_blocks);

    bn254_batch_msm_kernel<<<grid, block, 0, stream>>>(
        (const G1Affine*)bases,
        (const uint64_t*)scalars,
        d_partials,
        count
    );

    // Final reduction on CPU (small number of partial sums)
    cudaMemcpyAsync(result, d_partials, sizeof(G1Projective), cudaMemcpyDeviceToHost, stream);

    cudaFree(d_partials);

    return cudaGetLastError();
}

} // extern "C"
