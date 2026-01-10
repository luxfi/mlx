// Copyright (c) 2024-2026 Lux Industries Inc. All Rights Reserved.
// PROPRIETARY AND CONFIDENTIAL - NO LICENSE GRANTED
// Contact: licensing@luxindustries.xyz
//
// Negacyclic NTT for Lattice Cryptography
// Optimized for Dilithium (q=8380417) and Kyber (q=3329)
// Supports n=256, 512, 1024 with Montgomery arithmetic

#include <cuda_runtime.h>
#include <cstdint>

namespace lux {
namespace cuda {
namespace lattice {

// ============================================================================
// Prime Parameters
// ============================================================================

// Dilithium parameters
constexpr uint32_t DILITHIUM_Q = 8380417;          // q = 2^23 - 2^13 + 1
constexpr uint32_t DILITHIUM_N = 256;
constexpr uint32_t DILITHIUM_ROOT = 1753;          // Primitive 512-th root of unity
constexpr uint32_t DILITHIUM_MONT_R = 4193792;     // R = 2^32 mod q
constexpr uint32_t DILITHIUM_QINV = 58728449;      // q^{-1} mod 2^32

// Kyber parameters  
constexpr uint32_t KYBER_Q = 3329;
constexpr uint32_t KYBER_N = 256;
constexpr uint32_t KYBER_ROOT = 17;                // Primitive 512-th root of unity
constexpr uint32_t KYBER_MONT_R = 2285;            // R = 2^16 mod q
constexpr uint32_t KYBER_QINV = 62209;             // -q^{-1} mod 2^16

// Device constants
__constant__ uint32_t d_dilithium_zetas[256];      // Precomputed twiddles
__constant__ uint32_t d_dilithium_zetas_inv[256];
__constant__ uint32_t d_kyber_zetas[128];
__constant__ uint32_t d_kyber_zetas_inv[128];

// ============================================================================
// Montgomery Arithmetic for 32-bit Primes
// ============================================================================

// Dilithium Montgomery reduction: x * R^{-1} mod q
__device__ __forceinline__
int32_t dilithium_mont_reduce(int64_t x) {
    int32_t t = (int32_t)x * DILITHIUM_QINV;
    t = (int32_t)((x - (int64_t)t * DILITHIUM_Q) >> 32);
    return t;
}

// Kyber Montgomery reduction (16-bit based)
__device__ __forceinline__
int16_t kyber_mont_reduce(int32_t x) {
    int16_t t = (int16_t)((uint16_t)x * (uint16_t)KYBER_QINV);
    t = (int16_t)((x - (int32_t)t * KYBER_Q) >> 16);
    return t;
}

// Barrett reduction for Dilithium
__device__ __forceinline__
int32_t dilithium_barrett_reduce(int32_t x) {
    // Reduce x mod q to range (-q, q)
    const int32_t v = ((int64_t)x * 8396807 + (1LL << 31)) >> 32;
    return x - v * DILITHIUM_Q;
}

// Barrett reduction for Kyber
__device__ __forceinline__
int16_t kyber_barrett_reduce(int16_t x) {
    // Reduce x mod q to range (-q, q)
    const int16_t v = ((int32_t)x * 20159 + (1 << 25)) >> 26;
    return x - v * KYBER_Q;
}

// Conditional subtraction to normalize
__device__ __forceinline__
int32_t dilithium_caddq(int32_t x) {
    return x + ((x >> 31) & DILITHIUM_Q);
}

__device__ __forceinline__
int16_t kyber_caddq(int16_t x) {
    return x + ((x >> 15) & KYBER_Q);
}

// ============================================================================
// Cooley-Tukey Butterfly (CT) - Used in Forward NTT
// ============================================================================

__device__ __forceinline__
void dilithium_ct_butterfly(int32_t& a, int32_t& b, int32_t zeta) {
    int32_t t = dilithium_mont_reduce((int64_t)zeta * b);
    b = a - t;
    a = a + t;
}

__device__ __forceinline__
void kyber_ct_butterfly(int16_t& a, int16_t& b, int16_t zeta) {
    int16_t t = kyber_mont_reduce((int32_t)zeta * b);
    b = a - t;
    a = a + t;
}

// ============================================================================
// Gentleman-Sande Butterfly (GS) - Used in Inverse NTT
// ============================================================================

__device__ __forceinline__
void dilithium_gs_butterfly(int32_t& a, int32_t& b, int32_t zeta) {
    int32_t t = a;
    a = t + b;
    b = t - b;
    b = dilithium_mont_reduce((int64_t)zeta * b);
}

__device__ __forceinline__
void kyber_gs_butterfly(int16_t& a, int16_t& b, int16_t zeta) {
    int16_t t = a;
    a = kyber_barrett_reduce(t + b);
    b = t - b;
    b = kyber_mont_reduce((int32_t)zeta * b);
}

// ============================================================================
// Dilithium NTT Kernels
// ============================================================================

// In-place forward NTT for Dilithium (n=256)
__global__ void dilithium_ntt_forward_kernel(
    int32_t* __restrict__ poly,
    uint32_t batch_size
) {
    __shared__ int32_t s_poly[256];
    
    const uint32_t batch_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    int32_t* p = poly + batch_idx * 256;
    
    // Load to shared memory
    if (tid < 256) {
        s_poly[tid] = p[tid];
    }
    __syncthreads();
    
    // NTT stages: 8 layers for n=256
    uint32_t zeta_idx = 0;
    
    // Stage 1: m=128, len=1
    if (tid < 128) {
        int32_t zeta = d_dilithium_zetas[++zeta_idx];
        uint32_t j = tid;
        uint32_t k = tid + 128;
        dilithium_ct_butterfly(s_poly[j], s_poly[k], zeta);
    }
    __syncthreads();
    
    // Stage 2: m=64, len=2
    if (tid < 128) {
        uint32_t group = tid / 64;
        uint32_t in_group = tid % 64;
        int32_t zeta = d_dilithium_zetas[2 + group];
        uint32_t j = group * 128 + in_group;
        uint32_t k = j + 64;
        dilithium_ct_butterfly(s_poly[j], s_poly[k], zeta);
    }
    __syncthreads();
    
    // Stage 3: m=32, len=4
    if (tid < 128) {
        uint32_t group = tid / 32;
        uint32_t in_group = tid % 32;
        int32_t zeta = d_dilithium_zetas[4 + group];
        uint32_t j = group * 64 + in_group;
        uint32_t k = j + 32;
        dilithium_ct_butterfly(s_poly[j], s_poly[k], zeta);
    }
    __syncthreads();
    
    // Stage 4: m=16, len=8
    if (tid < 128) {
        uint32_t group = tid / 16;
        uint32_t in_group = tid % 16;
        int32_t zeta = d_dilithium_zetas[8 + group];
        uint32_t j = group * 32 + in_group;
        uint32_t k = j + 16;
        dilithium_ct_butterfly(s_poly[j], s_poly[k], zeta);
    }
    __syncthreads();
    
    // Stage 5: m=8, len=16
    if (tid < 128) {
        uint32_t group = tid / 8;
        uint32_t in_group = tid % 8;
        int32_t zeta = d_dilithium_zetas[16 + group];
        uint32_t j = group * 16 + in_group;
        uint32_t k = j + 8;
        dilithium_ct_butterfly(s_poly[j], s_poly[k], zeta);
    }
    __syncthreads();
    
    // Stage 6: m=4, len=32
    if (tid < 128) {
        uint32_t group = tid / 4;
        uint32_t in_group = tid % 4;
        int32_t zeta = d_dilithium_zetas[32 + group];
        uint32_t j = group * 8 + in_group;
        uint32_t k = j + 4;
        dilithium_ct_butterfly(s_poly[j], s_poly[k], zeta);
    }
    __syncthreads();
    
    // Stage 7: m=2, len=64
    if (tid < 128) {
        uint32_t group = tid / 2;
        uint32_t in_group = tid % 2;
        int32_t zeta = d_dilithium_zetas[64 + group];
        uint32_t j = group * 4 + in_group;
        uint32_t k = j + 2;
        dilithium_ct_butterfly(s_poly[j], s_poly[k], zeta);
    }
    __syncthreads();
    
    // Stage 8: m=1, len=128
    if (tid < 128) {
        int32_t zeta = d_dilithium_zetas[128 + tid];
        uint32_t j = tid * 2;
        uint32_t k = j + 1;
        dilithium_ct_butterfly(s_poly[j], s_poly[k], zeta);
    }
    __syncthreads();
    
    // Store results
    if (tid < 256) {
        p[tid] = s_poly[tid];
    }
}

// In-place inverse NTT for Dilithium
__global__ void dilithium_ntt_inverse_kernel(
    int32_t* __restrict__ poly,
    uint32_t batch_size
) {
    __shared__ int32_t s_poly[256];
    
    const uint32_t batch_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    int32_t* p = poly + batch_idx * 256;
    
    // Load to shared memory
    if (tid < 256) {
        s_poly[tid] = p[tid];
    }
    __syncthreads();
    
    // Inverse NTT: reverse order stages with GS butterflies
    
    // Stage 1: m=1, len=128 (reverse of forward stage 8)
    if (tid < 128) {
        int32_t zeta = d_dilithium_zetas_inv[128 + tid];
        uint32_t j = tid * 2;
        uint32_t k = j + 1;
        dilithium_gs_butterfly(s_poly[j], s_poly[k], zeta);
    }
    __syncthreads();
    
    // Stage 2: m=2, len=64
    if (tid < 128) {
        uint32_t group = tid / 2;
        uint32_t in_group = tid % 2;
        int32_t zeta = d_dilithium_zetas_inv[64 + group];
        uint32_t j = group * 4 + in_group;
        uint32_t k = j + 2;
        dilithium_gs_butterfly(s_poly[j], s_poly[k], zeta);
    }
    __syncthreads();
    
    // Stage 3: m=4, len=32
    if (tid < 128) {
        uint32_t group = tid / 4;
        uint32_t in_group = tid % 4;
        int32_t zeta = d_dilithium_zetas_inv[32 + group];
        uint32_t j = group * 8 + in_group;
        uint32_t k = j + 4;
        dilithium_gs_butterfly(s_poly[j], s_poly[k], zeta);
    }
    __syncthreads();
    
    // Stage 4: m=8, len=16
    if (tid < 128) {
        uint32_t group = tid / 8;
        uint32_t in_group = tid % 8;
        int32_t zeta = d_dilithium_zetas_inv[16 + group];
        uint32_t j = group * 16 + in_group;
        uint32_t k = j + 8;
        dilithium_gs_butterfly(s_poly[j], s_poly[k], zeta);
    }
    __syncthreads();
    
    // Stage 5: m=16, len=8
    if (tid < 128) {
        uint32_t group = tid / 16;
        uint32_t in_group = tid % 16;
        int32_t zeta = d_dilithium_zetas_inv[8 + group];
        uint32_t j = group * 32 + in_group;
        uint32_t k = j + 16;
        dilithium_gs_butterfly(s_poly[j], s_poly[k], zeta);
    }
    __syncthreads();
    
    // Stage 6: m=32, len=4
    if (tid < 128) {
        uint32_t group = tid / 32;
        uint32_t in_group = tid % 32;
        int32_t zeta = d_dilithium_zetas_inv[4 + group];
        uint32_t j = group * 64 + in_group;
        uint32_t k = j + 32;
        dilithium_gs_butterfly(s_poly[j], s_poly[k], zeta);
    }
    __syncthreads();
    
    // Stage 7: m=64, len=2
    if (tid < 128) {
        uint32_t group = tid / 64;
        uint32_t in_group = tid % 64;
        int32_t zeta = d_dilithium_zetas_inv[2 + group];
        uint32_t j = group * 128 + in_group;
        uint32_t k = j + 64;
        dilithium_gs_butterfly(s_poly[j], s_poly[k], zeta);
    }
    __syncthreads();
    
    // Stage 8: m=128, len=1 (and scale by n^{-1})
    if (tid < 128) {
        int32_t zeta = d_dilithium_zetas_inv[1];  // Root inverse
        uint32_t j = tid;
        uint32_t k = tid + 128;
        dilithium_gs_butterfly(s_poly[j], s_poly[k], zeta);
    }
    __syncthreads();
    
    // Scale by n^{-1} and normalize
    const int32_t ninv = 41978;  // 256^{-1} mod q in Montgomery form
    if (tid < 256) {
        int32_t val = dilithium_mont_reduce((int64_t)s_poly[tid] * ninv);
        p[tid] = dilithium_caddq(val);
    }
}

// ============================================================================
// Kyber NTT Kernels
// ============================================================================

// In-place forward NTT for Kyber (n=256, but NTT only to degree 128)
__global__ void kyber_ntt_forward_kernel(
    int16_t* __restrict__ poly,
    uint32_t batch_size
) {
    __shared__ int16_t s_poly[256];
    
    const uint32_t batch_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    int16_t* p = poly + batch_idx * 256;
    
    // Load to shared memory
    if (tid < 256) {
        s_poly[tid] = p[tid];
    }
    __syncthreads();
    
    // Kyber uses 7 layers of NTT (down to pairs)
    // zeta_idx starts at 1 (zeta[0] is unused)
    
    // Stage 1: len=128
    if (tid < 128) {
        int16_t zeta = d_kyber_zetas[1];
        kyber_ct_butterfly(s_poly[tid], s_poly[tid + 128], zeta);
    }
    __syncthreads();
    
    // Stage 2: len=64
    if (tid < 128) {
        uint32_t group = tid / 64;
        uint32_t in_group = tid % 64;
        int16_t zeta = d_kyber_zetas[2 + group];
        uint32_t j = group * 128 + in_group;
        kyber_ct_butterfly(s_poly[j], s_poly[j + 64], zeta);
    }
    __syncthreads();
    
    // Stage 3: len=32
    if (tid < 128) {
        uint32_t group = tid / 32;
        uint32_t in_group = tid % 32;
        int16_t zeta = d_kyber_zetas[4 + group];
        uint32_t j = group * 64 + in_group;
        kyber_ct_butterfly(s_poly[j], s_poly[j + 32], zeta);
    }
    __syncthreads();
    
    // Stage 4: len=16
    if (tid < 128) {
        uint32_t group = tid / 16;
        uint32_t in_group = tid % 16;
        int16_t zeta = d_kyber_zetas[8 + group];
        uint32_t j = group * 32 + in_group;
        kyber_ct_butterfly(s_poly[j], s_poly[j + 16], zeta);
    }
    __syncthreads();
    
    // Stage 5: len=8
    if (tid < 128) {
        uint32_t group = tid / 8;
        uint32_t in_group = tid % 8;
        int16_t zeta = d_kyber_zetas[16 + group];
        uint32_t j = group * 16 + in_group;
        kyber_ct_butterfly(s_poly[j], s_poly[j + 8], zeta);
    }
    __syncthreads();
    
    // Stage 6: len=4
    if (tid < 128) {
        uint32_t group = tid / 4;
        uint32_t in_group = tid % 4;
        int16_t zeta = d_kyber_zetas[32 + group];
        uint32_t j = group * 8 + in_group;
        kyber_ct_butterfly(s_poly[j], s_poly[j + 4], zeta);
    }
    __syncthreads();
    
    // Stage 7: len=2
    if (tid < 128) {
        uint32_t group = tid / 2;
        uint32_t in_group = tid % 2;
        int16_t zeta = d_kyber_zetas[64 + group];
        uint32_t j = group * 4 + in_group;
        kyber_ct_butterfly(s_poly[j], s_poly[j + 2], zeta);
    }
    __syncthreads();
    
    // Reduce and store
    if (tid < 256) {
        p[tid] = kyber_barrett_reduce(s_poly[tid]);
    }
}

// In-place inverse NTT for Kyber
__global__ void kyber_ntt_inverse_kernel(
    int16_t* __restrict__ poly,
    uint32_t batch_size
) {
    __shared__ int16_t s_poly[256];
    
    const uint32_t batch_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    int16_t* p = poly + batch_idx * 256;
    
    // Load
    if (tid < 256) {
        s_poly[tid] = p[tid];
    }
    __syncthreads();
    
    // Inverse NTT stages (reverse order with GS butterflies)
    
    // Stage 1: len=2
    if (tid < 128) {
        uint32_t group = tid / 2;
        uint32_t in_group = tid % 2;
        int16_t zeta = d_kyber_zetas_inv[64 + group];
        uint32_t j = group * 4 + in_group;
        kyber_gs_butterfly(s_poly[j], s_poly[j + 2], zeta);
    }
    __syncthreads();
    
    // Stage 2: len=4
    if (tid < 128) {
        uint32_t group = tid / 4;
        uint32_t in_group = tid % 4;
        int16_t zeta = d_kyber_zetas_inv[32 + group];
        uint32_t j = group * 8 + in_group;
        kyber_gs_butterfly(s_poly[j], s_poly[j + 4], zeta);
    }
    __syncthreads();
    
    // Stage 3: len=8
    if (tid < 128) {
        uint32_t group = tid / 8;
        uint32_t in_group = tid % 8;
        int16_t zeta = d_kyber_zetas_inv[16 + group];
        uint32_t j = group * 16 + in_group;
        kyber_gs_butterfly(s_poly[j], s_poly[j + 8], zeta);
    }
    __syncthreads();
    
    // Stage 4: len=16
    if (tid < 128) {
        uint32_t group = tid / 16;
        uint32_t in_group = tid % 16;
        int16_t zeta = d_kyber_zetas_inv[8 + group];
        uint32_t j = group * 32 + in_group;
        kyber_gs_butterfly(s_poly[j], s_poly[j + 16], zeta);
    }
    __syncthreads();
    
    // Stage 5: len=32
    if (tid < 128) {
        uint32_t group = tid / 32;
        uint32_t in_group = tid % 32;
        int16_t zeta = d_kyber_zetas_inv[4 + group];
        uint32_t j = group * 64 + in_group;
        kyber_gs_butterfly(s_poly[j], s_poly[j + 32], zeta);
    }
    __syncthreads();
    
    // Stage 6: len=64
    if (tid < 128) {
        uint32_t group = tid / 64;
        uint32_t in_group = tid % 64;
        int16_t zeta = d_kyber_zetas_inv[2 + group];
        uint32_t j = group * 128 + in_group;
        kyber_gs_butterfly(s_poly[j], s_poly[j + 64], zeta);
    }
    __syncthreads();
    
    // Stage 7: len=128 (final, with n^-1 scaling)
    if (tid < 128) {
        int16_t zeta = d_kyber_zetas_inv[1];
        kyber_gs_butterfly(s_poly[tid], s_poly[tid + 128], zeta);
    }
    __syncthreads();
    
    // Scale by n^{-1} = 3303 (256^{-1} mod 3329 in Montgomery form)
    const int16_t ninv = 1441;  // 128^{-1} * R mod q
    if (tid < 256) {
        int16_t val = kyber_mont_reduce((int32_t)s_poly[tid] * ninv);
        p[tid] = kyber_caddq(val);
    }
}

// ============================================================================
// Pointwise Multiplication in NTT Domain
// ============================================================================

// Dilithium pointwise multiplication
__global__ void dilithium_pointwise_mul_kernel(
    const int32_t* __restrict__ a,
    const int32_t* __restrict__ b,
    int32_t* __restrict__ c,
    uint32_t n,
    uint32_t batch_size
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = n * batch_size;
    
    if (idx >= total) return;
    
    c[idx] = dilithium_mont_reduce((int64_t)a[idx] * b[idx]);
}

// Kyber base multiplication (for pairs in NTT domain)
// Kyber NTT leaves pairs that need special handling
__global__ void kyber_basemul_kernel(
    const int16_t* __restrict__ a,
    const int16_t* __restrict__ b,
    int16_t* __restrict__ c,
    uint32_t batch_size
) {
    const uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_pairs = 128 * batch_size;
    
    if (pair_idx >= total_pairs) return;
    
    uint32_t batch = pair_idx / 128;
    uint32_t local_pair = pair_idx % 128;
    
    const int16_t* a_ptr = a + batch * 256 + local_pair * 2;
    const int16_t* b_ptr = b + batch * 256 + local_pair * 2;
    int16_t* c_ptr = c + batch * 256 + local_pair * 2;
    
    // Get zeta for this pair (gamma in original Kyber notation)
    int16_t zeta = d_kyber_zetas[64 + local_pair];
    
    // r[0] = a[0]*b[0] + a[1]*b[1]*zeta
    // r[1] = a[0]*b[1] + a[1]*b[0]
    int16_t a0 = a_ptr[0], a1 = a_ptr[1];
    int16_t b0 = b_ptr[0], b1 = b_ptr[1];
    
    int32_t t = (int32_t)a1 * b1;
    t = kyber_mont_reduce(t);
    t = (int32_t)t * zeta;
    t = kyber_mont_reduce(t);
    t = t + (int32_t)a0 * b0;
    c_ptr[0] = kyber_mont_reduce(t);
    
    t = (int32_t)a0 * b1 + (int32_t)a1 * b0;
    c_ptr[1] = kyber_mont_reduce(t);
}

// ============================================================================
// Batch Operations
// ============================================================================

// Batch add polynomials
__global__ void poly_add_kernel(
    const int32_t* __restrict__ a,
    const int32_t* __restrict__ b,
    int32_t* __restrict__ c,
    uint32_t n,
    uint32_t batch_size,
    uint32_t q
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = n * batch_size;
    
    if (idx >= total) return;
    
    int32_t sum = a[idx] + b[idx];
    // Reduce if needed
    if (sum >= (int32_t)q) sum -= q;
    if (sum < 0) sum += q;
    
    c[idx] = sum;
}

// Batch subtract polynomials  
__global__ void poly_sub_kernel(
    const int32_t* __restrict__ a,
    const int32_t* __restrict__ b,
    int32_t* __restrict__ c,
    uint32_t n,
    uint32_t batch_size,
    uint32_t q
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = n * batch_size;
    
    if (idx >= total) return;
    
    int32_t diff = a[idx] - b[idx];
    if (diff < 0) diff += q;
    
    c[idx] = diff;
}

// ============================================================================
// Twiddle Factor Generation
// ============================================================================

__global__ void generate_dilithium_zetas_kernel(
    uint32_t* __restrict__ zetas,
    uint32_t* __restrict__ zetas_inv
) {
    // Compute powers of the primitive root
    // Root of unity: zeta = 1753 (primitive 512th root)
    // For NTT: we need powers in bit-reversed order
    
    if (threadIdx.x != 0) return;
    
    const uint32_t q = DILITHIUM_Q;
    const uint32_t root = DILITHIUM_ROOT;
    
    // Compute root^{br(k)} for k = 0..255
    // Using standard bit-reversal for the NTT indices
    
    zetas[0] = 0;  // Unused
    
    uint32_t power = 1;
    for (uint32_t i = 0; i < 256; i++) {
        // Bit-reverse i in 8 bits
        uint32_t br = 0;
        uint32_t tmp = i;
        for (uint32_t j = 0; j < 8; j++) {
            br = (br << 1) | (tmp & 1);
            tmp >>= 1;
        }
        
        // Compute root^br mod q
        uint64_t val = 1;
        uint64_t base = root;
        uint32_t exp = br;
        while (exp > 0) {
            if (exp & 1) val = (val * base) % q;
            base = (base * base) % q;
            exp >>= 1;
        }
        
        zetas[i] = (uint32_t)val;
        
        // Compute inverse: val^{-1} = val^{q-2} mod q
        uint64_t inv = 1;
        base = val;
        exp = q - 2;
        while (exp > 0) {
            if (exp & 1) inv = (inv * base) % q;
            base = (base * base) % q;
            exp >>= 1;
        }
        zetas_inv[i] = (uint32_t)inv;
    }
}

// ============================================================================
// Host API
// ============================================================================

void dilithium_ntt_forward(
    int32_t* poly,
    uint32_t batch_size,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(batch_size);
    
    dilithium_ntt_forward_kernel<<<grid, block, 0, stream>>>(poly, batch_size);
}

void dilithium_ntt_inverse(
    int32_t* poly,
    uint32_t batch_size,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(batch_size);
    
    dilithium_ntt_inverse_kernel<<<grid, block, 0, stream>>>(poly, batch_size);
}

void kyber_ntt_forward(
    int16_t* poly,
    uint32_t batch_size,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(batch_size);
    
    kyber_ntt_forward_kernel<<<grid, block, 0, stream>>>(poly, batch_size);
}

void kyber_ntt_inverse(
    int16_t* poly,
    uint32_t batch_size,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(batch_size);
    
    kyber_ntt_inverse_kernel<<<grid, block, 0, stream>>>(poly, batch_size);
}

void dilithium_pointwise_mul(
    const int32_t* a,
    const int32_t* b,
    int32_t* c,
    uint32_t n,
    uint32_t batch_size,
    cudaStream_t stream
) {
    uint32_t total = n * batch_size;
    dim3 block(256);
    dim3 grid((total + 255) / 256);
    
    dilithium_pointwise_mul_kernel<<<grid, block, 0, stream>>>(a, b, c, n, batch_size);
}

void kyber_basemul(
    const int16_t* a,
    const int16_t* b,
    int16_t* c,
    uint32_t batch_size,
    cudaStream_t stream
) {
    uint32_t total_pairs = 128 * batch_size;
    dim3 block(256);
    dim3 grid((total_pairs + 255) / 256);
    
    kyber_basemul_kernel<<<grid, block, 0, stream>>>(a, b, c, batch_size);
}

void init_dilithium_zetas(cudaStream_t stream) {
    uint32_t* d_zetas;
    uint32_t* d_zetas_inv;
    
    cudaMalloc(&d_zetas, 256 * sizeof(uint32_t));
    cudaMalloc(&d_zetas_inv, 256 * sizeof(uint32_t));
    
    generate_dilithium_zetas_kernel<<<1, 1, 0, stream>>>(d_zetas, d_zetas_inv);
    
    cudaMemcpyToSymbol(d_dilithium_zetas, d_zetas, 256 * sizeof(uint32_t));
    cudaMemcpyToSymbol(d_dilithium_zetas_inv, d_zetas_inv, 256 * sizeof(uint32_t));
    
    cudaFree(d_zetas);
    cudaFree(d_zetas_inv);
}

} // namespace lattice
} // namespace cuda
} // namespace lux
