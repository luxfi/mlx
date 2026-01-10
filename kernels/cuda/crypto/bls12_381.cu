// Copyright (c) 2024-2026 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// BLS12-381 Elliptic Curve Operations
// High-performance CUDA implementation

#include <cuda_runtime.h>
#include <stdint.h>

namespace lux {
namespace cuda {
namespace kernels {

// 384-bit field element (12 x 32-bit limbs)
struct Fp {
    uint32_t limbs[12];
};

// Fp2 = Fp[u] / (u^2 + 1)
struct Fp2 {
    Fp c0;
    Fp c1;
};

// G1 affine point
struct G1Affine {
    Fp x;
    Fp y;
};

// G1 projective (Jacobian) point
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
__constant__ uint32_t BLS_P[12] = {
    0xffffaaabu, 0xb9fffffeu, 0xb153ffffu, 0x1eabfffeu,
    0xf6b0f624u, 0x6730d2a0u, 0xf38512bfu, 0x64774b84u,
    0x434bacd7u, 0x4b1ba7b6u, 0x397fe69au, 0x1a0111eau
};

// Montgomery R^2 mod p
__constant__ uint32_t BLS_R2[12];

// n' for Montgomery reduction
__constant__ uint32_t BLS_N_PRIME;

// ============================================================================
// Fp Arithmetic
// ============================================================================

__device__ __forceinline__
Fp fp_zero() {
    Fp r;
    #pragma unroll
    for (int i = 0; i < 12; i++) r.limbs[i] = 0;
    return r;
}

__device__ __forceinline__
Fp fp_one() {
    Fp r = fp_zero();
    // Montgomery form of 1
    r.limbs[0] = 0x760900u;
    r.limbs[1] = 0x00000002u;
    return r;
}

__device__ __forceinline__
bool fp_is_zero(Fp a) {
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        if (a.limbs[i] != 0) return false;
    }
    return true;
}

// Fp addition with reduction
__device__
Fp fp_add(Fp a, Fp b) {
    Fp r;
    uint64_t carry = 0;
    
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        uint64_t sum = (uint64_t)a.limbs[i] + b.limbs[i] + carry;
        r.limbs[i] = (uint32_t)sum;
        carry = sum >> 32;
    }
    
    // Conditional subtraction of p
    bool gte = true;
    #pragma unroll
    for (int i = 11; i >= 0; i--) {
        if (r.limbs[i] < BLS_P[i]) { gte = false; break; }
        if (r.limbs[i] > BLS_P[i]) break;
    }
    
    if (gte || carry) {
        uint64_t borrow = 0;
        #pragma unroll
        for (int i = 0; i < 12; i++) {
            uint64_t diff = (uint64_t)r.limbs[i] - BLS_P[i] - borrow;
            r.limbs[i] = (uint32_t)diff;
            borrow = (diff >> 63) & 1;
        }
    }
    
    return r;
}

// Fp subtraction
__device__
Fp fp_sub(Fp a, Fp b) {
    Fp r;
    int64_t borrow = 0;
    
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        int64_t diff = (int64_t)a.limbs[i] - b.limbs[i] - borrow;
        r.limbs[i] = (uint32_t)diff;
        borrow = (diff < 0) ? 1 : 0;
    }
    
    if (borrow) {
        uint64_t carry = 0;
        #pragma unroll
        for (int i = 0; i < 12; i++) {
            uint64_t sum = (uint64_t)r.limbs[i] + BLS_P[i] + carry;
            r.limbs[i] = (uint32_t)sum;
            carry = sum >> 32;
        }
    }
    
    return r;
}

// Fp negation
__device__ __forceinline__
Fp fp_neg(Fp a) {
    if (fp_is_zero(a)) return a;
    
    Fp r;
    uint64_t borrow = 0;
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        uint64_t diff = (uint64_t)BLS_P[i] - a.limbs[i] - borrow;
        r.limbs[i] = (uint32_t)diff;
        borrow = (diff >> 63) & 1;
    }
    return r;
}

// Montgomery multiplication
__device__
Fp fp_mul(Fp a, Fp b) {
    // CIOS (Coarsely Integrated Operand Scanning) algorithm
    uint64_t t[13] = {0};
    
    for (int i = 0; i < 12; i++) {
        // Multiply
        uint64_t c = 0;
        for (int j = 0; j < 12; j++) {
            uint64_t uv = t[j] + (uint64_t)a.limbs[j] * b.limbs[i] + c;
            t[j] = uv & 0xFFFFFFFF;
            c = uv >> 32;
        }
        t[12] = c;
        
        // Montgomery reduction
        uint32_t m = (uint32_t)t[0] * BLS_N_PRIME;
        c = 0;
        for (int j = 0; j < 12; j++) {
            uint64_t uv = t[j] + (uint64_t)m * BLS_P[j] + c;
            if (j > 0) t[j-1] = uv & 0xFFFFFFFF;
            c = uv >> 32;
        }
        t[11] = t[12] + c;
        t[12] = 0;
    }
    
    Fp r;
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        r.limbs[i] = (uint32_t)t[i];
    }
    
    // Final reduction
    bool gte = true;
    for (int i = 11; i >= 0; i--) {
        if (r.limbs[i] < BLS_P[i]) { gte = false; break; }
        if (r.limbs[i] > BLS_P[i]) break;
    }
    if (gte) {
        r = fp_sub(r, *(Fp*)BLS_P);
    }
    
    return r;
}

// Fp squaring (optimized)
__device__
Fp fp_square(Fp a) {
    return fp_mul(a, a);  // Can be optimized further
}

// ============================================================================
// G1 Operations
// ============================================================================

__device__ __forceinline__
G1Projective g1_identity() {
    G1Projective p;
    p.x = fp_zero();
    p.y = fp_one();
    p.z = fp_zero();
    return p;
}

__device__ __forceinline__
bool g1_is_identity(G1Projective p) {
    return fp_is_zero(p.z);
}

// Point doubling: 2P
__device__
G1Projective g1_double(G1Projective p) {
    if (g1_is_identity(p)) return p;
    
    // Optimized doubling for a=0 (BLS12-381)
    Fp A = fp_square(p.x);
    Fp B = fp_square(p.y);
    Fp C = fp_square(B);
    
    // D = 2*((X+B)^2 - A - C)
    Fp D = fp_add(p.x, B);
    D = fp_square(D);
    D = fp_sub(D, A);
    D = fp_sub(D, C);
    D = fp_add(D, D);
    
    // E = 3*A
    Fp E = fp_add(A, fp_add(A, A));
    
    // F = E^2
    Fp F = fp_square(E);
    
    // X3 = F - 2*D
    G1Projective r;
    r.x = fp_sub(F, fp_add(D, D));
    
    // Y3 = E*(D - X3) - 8*C
    Fp C8 = fp_add(C, C);
    C8 = fp_add(C8, C8);
    C8 = fp_add(C8, C8);
    r.y = fp_sub(D, r.x);
    r.y = fp_mul(E, r.y);
    r.y = fp_sub(r.y, C8);
    
    // Z3 = 2*Y*Z
    r.z = fp_mul(p.y, p.z);
    r.z = fp_add(r.z, r.z);
    
    return r;
}

// ============================================================================
// Kernels
// ============================================================================

__global__
void g1_batch_double(G1Projective* points, uint32_t count) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    points[idx] = g1_double(points[idx]);
}

__global__
void g1_batch_add(G1Projective* acc, const G1Affine* points, uint32_t count) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    // acc[idx] = g1_add_mixed(acc[idx], points[idx]);
}

} // namespace kernels
} // namespace cuda
} // namespace lux

// =============================================================================
// C API for CGO Bindings
// =============================================================================

extern "C" {

int lux_cuda_bls12_381_g1_batch_double(
    void* points,
    uint32_t count,
    cudaStream_t stream
) {
    using namespace lux::cuda::kernels;
    dim3 block(256);
    dim3 grid((count + block.x - 1) / block.x);
    g1_batch_double<<<grid, block, 0, stream>>>((G1Projective*)points, count);
    return cudaGetLastError();
}

int lux_cuda_bls12_381_g1_batch_add(
    void* acc,
    const void* points,
    uint32_t count,
    cudaStream_t stream
) {
    using namespace lux::cuda::kernels;
    dim3 block(256);
    dim3 grid((count + block.x - 1) / block.x);
    g1_batch_add<<<grid, block, 0, stream>>>(
        (G1Projective*)acc, (const G1Affine*)points, count
    );
    return cudaGetLastError();
}

} // extern "C"
