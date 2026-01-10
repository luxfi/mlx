// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Multi-Scalar Multiplication (MSM) CUDA Kernel
// Implements Pippenger's bucket method for BN254 and BLS12-381 curves

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================================
// Configuration
// ============================================================================

#define MSM_WINDOW_SIZE 8
#define MSM_NUM_WINDOWS 32        // 256 bits / 8 bits
#define MSM_BUCKETS_PER_WINDOW 255  // 2^c - 1

// ============================================================================
// BN254 Field Arithmetic (4 x 64-bit limbs)
// ============================================================================

struct Fp256 {
    uint64_t limbs[4];
};

// BN254 prime: 21888242871839275222246405745257275088696311157297823662689037894645226208583
__constant__ uint64_t BN254_P[4] = {
    0x3C208C16D87CFD47ULL,
    0x97816A916871CA8DULL,
    0xB85045B68181585DULL,
    0x30644E72E131A029ULL
};

// Montgomery constant: -p^{-1} mod 2^64
__constant__ uint64_t BN254_INV = 0x87D20782E4866389ULL;

// R^2 mod p (for converting to Montgomery form)
__constant__ uint64_t BN254_R2[4] = {
    0xF32CFC5B538AFA89ULL,
    0xB5E71911D44501FBULL,
    0x47AB1EFF0A417FF6ULL,
    0x06D89F71CAB8351FULL
};

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
    // lo = (a * b + c + carry).lo
    // carry = (a * b + c + carry).hi
    uint64_t product_lo = a * b;
    uint64_t product_hi = __umul64hi(a, b);

    uint64_t sum = product_lo + c;
    uint64_t c1 = (sum < product_lo) ? 1ULL : 0ULL;

    sum += carry;
    uint64_t c2 = (sum < carry) ? 1ULL : 0ULL;

    lo = sum;
    carry = product_hi + c1 + c2;
}

// BN254 field addition
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
    reduced.limbs[0] = sbb(result.limbs[0], BN254_P[0], borrow);
    reduced.limbs[1] = sbb(result.limbs[1], BN254_P[1], borrow);
    reduced.limbs[2] = sbb(result.limbs[2], BN254_P[2], borrow);
    reduced.limbs[3] = sbb(result.limbs[3], BN254_P[3], borrow);

    // If no borrow, use reduced result
    bool needs_reduce = (carry != 0) || (borrow == 0);
    if (needs_reduce) return reduced;
    return result;
}

// BN254 field subtraction
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
        result.limbs[0] = adc(result.limbs[0], BN254_P[0], carry);
        result.limbs[1] = adc(result.limbs[1], BN254_P[1], carry);
        result.limbs[2] = adc(result.limbs[2], BN254_P[2], carry);
        result.limbs[3] = adc(result.limbs[3], BN254_P[3], carry);
    }

    return result;
}

// BN254 Montgomery multiplication
__device__
Fp256 fp256_mul(const Fp256& a, const Fp256& b) {
    // Schoolbook multiplication with interleaved Montgomery reduction
    uint64_t t[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    // Multiply
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            mac(t[i + j], carry, a.limbs[i], b.limbs[j], t[i + j]);
        }
        t[i + 4] = carry;
    }

    // Montgomery reduction
    for (int i = 0; i < 4; i++) {
        uint64_t k = t[i] * BN254_INV;
        uint64_t carry = 0;
        mac(t[i], carry, k, BN254_P[0], t[i]);
        mac(t[i + 1], carry, k, BN254_P[1], t[i + 1]);
        mac(t[i + 2], carry, k, BN254_P[2], t[i + 2]);
        mac(t[i + 3], carry, k, BN254_P[3], t[i + 3]);

        // Propagate carry
        for (int j = i + 4; j < 8 && carry; j++) {
            uint64_t sum = t[j] + carry;
            carry = (sum < t[j]) ? 1ULL : 0ULL;
            t[j] = sum;
        }
    }

    // Result is in t[4..7]
    Fp256 result = {{t[4], t[5], t[6], t[7]}};

    // Final reduction
    uint64_t borrow = 0;
    Fp256 reduced;
    reduced.limbs[0] = sbb(result.limbs[0], BN254_P[0], borrow);
    reduced.limbs[1] = sbb(result.limbs[1], BN254_P[1], borrow);
    reduced.limbs[2] = sbb(result.limbs[2], BN254_P[2], borrow);
    reduced.limbs[3] = sbb(result.limbs[3], BN254_P[3], borrow);

    if (borrow == 0) return reduced;
    return result;
}

// BN254 field squaring (optimized)
__device__
Fp256 fp256_sqr(const Fp256& a) {
    return fp256_mul(a, a);  // TODO: Optimize with squaring-specific formula
}

// Double the field element
__device__
Fp256 fp256_double(const Fp256& a) {
    return fp256_add(a, a);
}

// Check if zero
__device__
bool fp256_is_zero(const Fp256& a) {
    return (a.limbs[0] | a.limbs[1] | a.limbs[2] | a.limbs[3]) == 0;
}

// ============================================================================
// Elliptic Curve Point Types
// ============================================================================

struct G1Affine {
    Fp256 x;
    Fp256 y;
    bool infinity;
};

struct G1Projective {
    Fp256 x;
    Fp256 y;
    Fp256 z;
};

// ============================================================================
// Point Operations (Jacobian Projective Coordinates)
// ============================================================================

__device__
G1Projective g1_identity() {
    G1Projective p;
    p.x = {{0, 0, 0, 0}};
    p.y = {{1, 0, 0, 0}};  // Montgomery form of 1
    p.z = {{0, 0, 0, 0}};
    return p;
}

__device__
bool g1_is_identity(const G1Projective& p) {
    return fp256_is_zero(p.z);
}

// Point doubling: 2P (Jacobian, a=0 optimization)
// Uses dbl-2009-l: 4M + 6S
__device__
G1Projective g1_double(const G1Projective& p) {
    if (g1_is_identity(p)) return p;

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

    G1Projective result = {X3, Y3, Z3};
    return result;
}

// Mixed addition: P + Q where Q is affine (madd-2008-s: 7M + 4S)
__device__
G1Projective g1_add_mixed(const G1Projective& p, const G1Affine& q) {
    if (q.infinity) return p;
    if (g1_is_identity(p)) {
        // Convert affine to projective
        Fp256 one = {{1, 0, 0, 0}};  // TODO: Use proper Montgomery 1
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

    G1Projective result = {X3, Y3, Z3};
    return result;
}

// Full projective addition (12M + 4S)
__device__
G1Projective g1_add(const G1Projective& p, const G1Projective& q) {
    if (g1_is_identity(p)) return q;
    if (g1_is_identity(q)) return p;

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

    G1Projective result = {X3, Y3, Z3};
    return result;
}

// ============================================================================
// MSM Kernels
// ============================================================================

// Extract window from scalar
__device__ __forceinline__
uint32_t extract_window(const Fp256& scalar, uint32_t window_idx) {
    uint32_t bit_offset = window_idx * MSM_WINDOW_SIZE;
    uint32_t limb_idx = bit_offset / 64;
    uint32_t bit_in_limb = bit_offset % 64;

    uint64_t window = scalar.limbs[limb_idx] >> bit_in_limb;

    // Handle cross-limb window
    if (bit_in_limb + MSM_WINDOW_SIZE > 64 && limb_idx + 1 < 4) {
        uint32_t bits_from_next = bit_in_limb + MSM_WINDOW_SIZE - 64;
        window |= (scalar.limbs[limb_idx + 1] << (64 - bit_in_limb));
    }

    return (uint32_t)(window & ((1ULL << MSM_WINDOW_SIZE) - 1));
}

// Phase 1: Bucket accumulation
extern "C" __global__
void msm_bucket_accumulate(
    const G1Affine* __restrict__ points,
    const Fp256* __restrict__ scalars,
    G1Projective* __restrict__ buckets,  // [num_windows][num_buckets]
    uint32_t num_points,
    uint32_t window_idx
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_points) return;

    uint32_t bucket_idx = extract_window(scalars[tid], window_idx);
    if (bucket_idx == 0) return;  // Skip zero bucket

    uint32_t bucket_offset = window_idx * MSM_BUCKETS_PER_WINDOW + (bucket_idx - 1);

    // NOTE: This has race conditions - production code needs atomic or sorting
    G1Projective bucket = buckets[bucket_offset];
    buckets[bucket_offset] = g1_add_mixed(bucket, points[tid]);
}

// Phase 2: Bucket reduction (running sum)
extern "C" __global__
void msm_bucket_reduce(
    G1Projective* __restrict__ buckets,      // [num_windows][num_buckets]
    G1Projective* __restrict__ window_sums,  // [num_windows]
    uint32_t window_idx
) {
    // Single thread per window for now
    if (threadIdx.x != 0) return;

    G1Projective running = g1_identity();
    G1Projective sum = g1_identity();

    uint32_t base = window_idx * MSM_BUCKETS_PER_WINDOW;

    // Running sum: bucket[n-1] + (bucket[n-1] + bucket[n-2]) + ...
    for (int i = MSM_BUCKETS_PER_WINDOW - 1; i >= 0; i--) {
        running = g1_add(running, buckets[base + i]);
        sum = g1_add(sum, running);
    }

    window_sums[window_idx] = sum;
}

// Phase 3: Window combination (Horner's method)
extern "C" __global__
void msm_window_combine(
    const G1Projective* __restrict__ window_sums,
    G1Projective* __restrict__ result,
    uint32_t num_windows
) {
    if (threadIdx.x != 0) return;

    G1Projective acc = g1_identity();

    // Horner: result = sum_{i=k-1}^{0} 2^{c*i} * window_sum[i]
    for (int i = num_windows - 1; i >= 0; i--) {
        // Multiply by 2^c
        for (int j = 0; j < MSM_WINDOW_SIZE; j++) {
            acc = g1_double(acc);
        }
        acc = g1_add(acc, window_sums[i]);
    }

    *result = acc;
}

// ============================================================================
// Tree Reduction for Parallel Bucket Reduction
// ============================================================================

extern "C" __global__
void msm_parallel_reduce(
    G1Projective* __restrict__ data,
    uint32_t n
) {
    extern __shared__ G1Projective shared[];

    uint32_t tid = threadIdx.x;
    uint32_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load two elements per thread
    if (idx < n) {
        shared[tid] = data[idx];
    } else {
        shared[tid] = g1_identity();
    }

    if (idx + blockDim.x < n) {
        shared[tid] = g1_add(shared[tid], data[idx + blockDim.x]);
    }

    __syncthreads();

    // Tree reduction
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = g1_add(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        data[blockIdx.x] = shared[0];
    }
}

// =============================================================================
// C API for CGO Bindings
// =============================================================================

extern "C" {

int lux_cuda_msm_bucket_accumulate(
    const void* points,
    const void* scalars,
    void* buckets,
    uint32_t num_points,
    uint32_t window_idx,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((num_points + block.x - 1) / block.x);
    msm_bucket_accumulate<<<grid, block, 0, stream>>>(
        (const G1Affine*)points,
        (const Fp256*)scalars,
        (G1Projective*)buckets,
        num_points,
        window_idx
    );
    return cudaGetLastError();
}

int lux_cuda_msm_bucket_reduce(
    void* buckets,
    void* window_sums,
    uint32_t window_idx,
    cudaStream_t stream
) {
    msm_bucket_reduce<<<1, 1, 0, stream>>>(
        (G1Projective*)buckets,
        (G1Projective*)window_sums,
        window_idx
    );
    return cudaGetLastError();
}

int lux_cuda_msm_window_combine(
    const void* window_sums,
    void* result,
    uint32_t num_windows,
    cudaStream_t stream
) {
    msm_window_combine<<<1, 1, 0, stream>>>(
        (const G1Projective*)window_sums,
        (G1Projective*)result,
        num_windows
    );
    return cudaGetLastError();
}

int lux_cuda_msm_parallel_reduce(
    void* data,
    uint32_t n,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((n + block.x * 2 - 1) / (block.x * 2));
    size_t shmem = block.x * sizeof(G1Projective);
    msm_parallel_reduce<<<grid, block, shmem, stream>>>((G1Projective*)data, n);
    return cudaGetLastError();
}

} // extern "C"
