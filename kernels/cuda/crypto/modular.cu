// Copyright (c) 2024-2026 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Modular Arithmetic - High-Performance CUDA Implementation

#include <cuda_runtime.h>
#include <cstdint>

namespace lux {
namespace cuda {
namespace modular {

// ============================================================================
// Montgomery Arithmetic
// ============================================================================

// Compute Montgomery parameters
__device__ __forceinline__ uint64_t compute_m0_inv(uint64_t mod) {
    // Compute -m^-1 mod 2^64 using Newton iteration
    uint64_t inv = mod;
    for (int i = 0; i < 6; i++) {
        inv *= 2 - mod * inv;
    }
    return -inv;
}

// Montgomery multiplication (CIOS algorithm)
__device__ __forceinline__ uint64_t mont_mul_cios(
    uint64_t a, uint64_t b, uint64_t mod, uint64_t m0_inv
) {
    // For 64-bit modulus
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);
    
    uint64_t m = lo * m0_inv;
    uint64_t t_hi = __umul64hi(m, mod);
    
    // result = (a*b + m*mod) / 2^64
    uint64_t result = hi - t_hi;
    
    // Conditional subtraction
    if (hi < t_hi) {
        result += mod;
    }
    if (result >= mod) {
        result -= mod;
    }
    
    return result;
}

// Montgomery reduction
__device__ __forceinline__ uint64_t mont_reduce(
    uint64_t lo, uint64_t hi, uint64_t mod, uint64_t m0_inv
) {
    uint64_t m = lo * m0_inv;
    uint64_t t_hi = __umul64hi(m, mod);
    uint64_t t_lo = m * mod;
    
    // Add (hi, lo) + (t_hi, t_lo) and take high part
    uint64_t carry = (lo + t_lo < lo) ? 1 : 0;
    uint64_t result = hi + t_hi + carry;
    
    if (result >= mod || result < hi) {
        result -= mod;
    }
    return result;
}

// Convert to Montgomery form
__device__ __forceinline__ uint64_t to_mont(uint64_t a, uint64_t r2, uint64_t mod, uint64_t m0_inv) {
    return mont_mul_cios(a, r2, mod, m0_inv);
}

// Convert from Montgomery form
__device__ __forceinline__ uint64_t from_mont(uint64_t a, uint64_t mod, uint64_t m0_inv) {
    return mont_mul_cios(a, 1, mod, m0_inv);
}

// ============================================================================
// Barrett Reduction
// ============================================================================

// Barrett reduction for 64-bit
__device__ __forceinline__ uint64_t barrett_reduce(
    uint64_t a, uint64_t mod, uint64_t mu, uint32_t k
) {
    // q = floor(a * mu / 2^k)
    uint64_t q = __umul64hi(a, mu);
    if (k > 64) {
        q >>= (k - 64);
    }
    
    // r = a - q * mod
    uint64_t r = a - q * mod;
    
    // Correction
    if (r >= mod) r -= mod;
    
    return r;
}

// Barrett reduction for 128-bit product
__device__ __forceinline__ uint64_t barrett_reduce_wide(
    uint64_t lo, uint64_t hi, uint64_t mod, uint64_t mu
) {
    // Approximate quotient from high bits
    uint64_t q = __umul64hi(hi, mu);
    
    // r = (hi, lo) - q * mod
    uint64_t q_mod_lo = q * mod;
    uint64_t q_mod_hi = __umul64hi(q, mod);
    
    // Subtract
    uint64_t r = lo - q_mod_lo;
    if (lo < q_mod_lo) hi--;
    hi -= q_mod_hi;
    
    // Result should fit in 64 bits now
    while (hi > 0 || r >= mod) {
        r -= mod;
        if (r > lo) hi--;  // Borrow
    }
    
    return r;
}

// ============================================================================
// Basic Modular Operations
// ============================================================================

// Modular addition
__device__ __forceinline__ uint64_t mod_add(uint64_t a, uint64_t b, uint64_t mod) {
    uint64_t sum = a + b;
    if (sum < a || sum >= mod) {
        sum -= mod;
    }
    return sum;
}

// Modular subtraction
__device__ __forceinline__ uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t mod) {
    uint64_t diff = a - b;
    if (a < b) {
        diff += mod;
    }
    return diff;
}

// Modular negation
__device__ __forceinline__ uint64_t mod_neg(uint64_t a, uint64_t mod) {
    return (a == 0) ? 0 : (mod - a);
}

// Modular multiplication (using Barrett)
__device__ __forceinline__ uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t mod, uint64_t mu) {
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);
    return barrett_reduce_wide(lo, hi, mod, mu);
}

// Modular exponentiation (square-and-multiply)
__device__ uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t mod, uint64_t mu) {
    uint64_t result = 1;
    base = base % mod;
    
    while (exp > 0) {
        if (exp & 1) {
            result = mod_mul(result, base, mod, mu);
        }
        exp >>= 1;
        base = mod_mul(base, base, mod, mu);
    }
    return result;
}

// Modular inverse using Fermat's little theorem
__device__ uint64_t mod_inv(uint64_t a, uint64_t mod, uint64_t mu) {
    return mod_pow(a, mod - 2, mod, mu);
}

// ============================================================================
// Batch Kernels
// ============================================================================

// Batch modular addition
__global__ void batch_add_kernel(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ c,
    uint64_t mod,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    
    c[tid] = mod_add(a[tid], b[tid], mod);
}

// Batch modular subtraction
__global__ void batch_sub_kernel(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ c,
    uint64_t mod,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    
    c[tid] = mod_sub(a[tid], b[tid], mod);
}

// Batch modular multiplication
__global__ void batch_mul_kernel(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ c,
    uint64_t mod,
    uint64_t mu,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    
    c[tid] = mod_mul(a[tid], b[tid], mod, mu);
}

// Batch Montgomery multiplication
__global__ void batch_mont_mul_kernel(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ c,
    uint64_t mod,
    uint64_t m0_inv,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    
    c[tid] = mont_mul_cios(a[tid], b[tid], mod, m0_inv);
}

// Batch modular exponentiation
__global__ void batch_pow_kernel(
    const uint64_t* __restrict__ bases,
    const uint64_t* __restrict__ exps,
    uint64_t* __restrict__ results,
    uint64_t mod,
    uint64_t mu,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    
    results[tid] = mod_pow(bases[tid], exps[tid], mod, mu);
}

// Batch modular inverse
__global__ void batch_inv_kernel(
    const uint64_t* __restrict__ a,
    uint64_t* __restrict__ a_inv,
    uint64_t mod,
    uint64_t mu,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    
    a_inv[tid] = mod_inv(a[tid], mod, mu);
}

// Batch Montgomery conversion (to Montgomery form)
__global__ void batch_to_mont_kernel(
    const uint64_t* __restrict__ a,
    uint64_t* __restrict__ a_mont,
    uint64_t mod,
    uint64_t r2,
    uint64_t m0_inv,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    
    a_mont[tid] = to_mont(a[tid], r2, mod, m0_inv);
}

// Batch Montgomery conversion (from Montgomery form)
__global__ void batch_from_mont_kernel(
    const uint64_t* __restrict__ a_mont,
    uint64_t* __restrict__ a,
    uint64_t mod,
    uint64_t m0_inv,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    
    a[tid] = from_mont(a_mont[tid], mod, m0_inv);
}

// ============================================================================
// Host Functions
// ============================================================================

void batch_add(
    const uint64_t* a, const uint64_t* b, uint64_t* c,
    uint64_t mod, uint32_t n, cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    batch_add_kernel<<<grid, block, 0, stream>>>(a, b, c, mod, n);
}

void batch_sub(
    const uint64_t* a, const uint64_t* b, uint64_t* c,
    uint64_t mod, uint32_t n, cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    batch_sub_kernel<<<grid, block, 0, stream>>>(a, b, c, mod, n);
}

void batch_mul(
    const uint64_t* a, const uint64_t* b, uint64_t* c,
    uint64_t mod, uint64_t mu, uint32_t n, cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    batch_mul_kernel<<<grid, block, 0, stream>>>(a, b, c, mod, mu, n);
}

void batch_mont_mul(
    const uint64_t* a, const uint64_t* b, uint64_t* c,
    uint64_t mod, uint64_t m0_inv, uint32_t n, cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    batch_mont_mul_kernel<<<grid, block, 0, stream>>>(a, b, c, mod, m0_inv, n);
}

void batch_pow(
    const uint64_t* bases, const uint64_t* exps, uint64_t* results,
    uint64_t mod, uint64_t mu, uint32_t n, cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    batch_pow_kernel<<<grid, block, 0, stream>>>(bases, exps, results, mod, mu, n);
}

void batch_inv(
    const uint64_t* a, uint64_t* a_inv,
    uint64_t mod, uint64_t mu, uint32_t n, cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    batch_inv_kernel<<<grid, block, 0, stream>>>(a, a_inv, mod, mu, n);
}

void batch_to_mont(
    const uint64_t* a, uint64_t* a_mont,
    uint64_t mod, uint64_t r2, uint64_t m0_inv, uint32_t n, cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    batch_to_mont_kernel<<<grid, block, 0, stream>>>(a, a_mont, mod, r2, m0_inv, n);
}

void batch_from_mont(
    const uint64_t* a_mont, uint64_t* a,
    uint64_t mod, uint64_t m0_inv, uint32_t n, cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    batch_from_mont_kernel<<<grid, block, 0, stream>>>(a_mont, a, mod, m0_inv, n);
}

} // namespace modular
} // namespace cuda
} // namespace lux

// =============================================================================
// C API for CGO Bindings
// =============================================================================

extern "C" {

int lux_cuda_modular_batch_add(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* c,
    uint64_t mod,
    uint32_t n,
    cudaStream_t stream
) {
    lux::cuda::modular::batch_add(a, b, c, mod, n, stream);
    return cudaGetLastError();
}

int lux_cuda_modular_batch_sub(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* c,
    uint64_t mod,
    uint32_t n,
    cudaStream_t stream
) {
    lux::cuda::modular::batch_sub(a, b, c, mod, n, stream);
    return cudaGetLastError();
}

int lux_cuda_modular_batch_mul(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* c,
    uint64_t mod,
    uint64_t mu,
    uint32_t n,
    cudaStream_t stream
) {
    lux::cuda::modular::batch_mul(a, b, c, mod, mu, n, stream);
    return cudaGetLastError();
}

int lux_cuda_modular_batch_mont_mul(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* c,
    uint64_t mod,
    uint64_t m0_inv,
    uint32_t n,
    cudaStream_t stream
) {
    lux::cuda::modular::batch_mont_mul(a, b, c, mod, m0_inv, n, stream);
    return cudaGetLastError();
}

int lux_cuda_modular_batch_pow(
    const uint64_t* bases,
    const uint64_t* exps,
    uint64_t* results,
    uint64_t mod,
    uint64_t mu,
    uint32_t n,
    cudaStream_t stream
) {
    lux::cuda::modular::batch_pow(bases, exps, results, mod, mu, n, stream);
    return cudaGetLastError();
}

int lux_cuda_modular_batch_inv(
    const uint64_t* a,
    uint64_t* a_inv,
    uint64_t mod,
    uint64_t mu,
    uint32_t n,
    cudaStream_t stream
) {
    lux::cuda::modular::batch_inv(a, a_inv, mod, mu, n, stream);
    return cudaGetLastError();
}

int lux_cuda_modular_batch_to_mont(
    const uint64_t* a,
    uint64_t* a_mont,
    uint64_t mod,
    uint64_t r2,
    uint64_t m0_inv,
    uint32_t n,
    cudaStream_t stream
) {
    lux::cuda::modular::batch_to_mont(a, a_mont, mod, r2, m0_inv, n, stream);
    return cudaGetLastError();
}

int lux_cuda_modular_batch_from_mont(
    const uint64_t* a_mont,
    uint64_t* a,
    uint64_t mod,
    uint64_t m0_inv,
    uint32_t n,
    cudaStream_t stream
) {
    lux::cuda::modular::batch_from_mont(a_mont, a, mod, m0_inv, n, stream);
    return cudaGetLastError();
}

} // extern "C"
