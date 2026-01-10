// Copyright (c) 2024-2026 Lux Industries Inc. All Rights Reserved.
// PROPRIETARY AND CONFIDENTIAL - NO LICENSE GRANTED
// Contact: licensing@luxindustries.xyz

#include <cuda_runtime.h>
#include <cstdint>

namespace lux {
namespace cuda {
namespace twiddle {

// Common prime moduli for NTT
__constant__ uint64_t PRIMES[] = {
    0xFFFFFFFF00000001ULL,  // Goldilocks: 2^64 - 2^32 + 1
    0x7FFFFFFF80000001ULL,  // BN254 scalar field characteristic
    0x73eda753299d7d483ULL, // BLS12-381 scalar field characteristic
};

// Primitive roots for common primes
__constant__ uint64_t PRIMITIVE_ROOTS[] = {
    7ULL,   // Goldilocks
    5ULL,   // BN254
    7ULL,   // BLS12-381
};

// Montgomery multiplication
__device__ __forceinline__ uint64_t mont_mul(uint64_t a, uint64_t b, uint64_t mod, uint64_t mod_inv) {
    // Using __umul64hi for high 64 bits
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);
    
    uint64_t m = lo * mod_inv;
    uint64_t t = __umul64hi(m, mod);
    
    uint64_t result = hi - t;
    if (hi < t) result += mod;
    
    return result;
}

// Modular exponentiation
__device__ uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    base %= mod;
    
    while (exp > 0) {
        if (exp & 1) {
            result = __umul64hi(result, base);
            result = result % mod;  // Simplified, should use Barrett
        }
        exp >>= 1;
        base = __umul64hi(base, base);
        base = base % mod;
    }
    return result;
}

// Montgomery modular exponentiation
__device__ uint64_t mont_pow(uint64_t base, uint64_t exp, uint64_t mod, uint64_t mod_inv, uint64_t r2) {
    uint64_t result = r2;  // R mod m (1 in Montgomery form)
    base = mont_mul(base, r2, mod, mod_inv);  // Convert to Montgomery form
    
    while (exp > 0) {
        if (exp & 1) {
            result = mont_mul(result, base, mod, mod_inv);
        }
        exp >>= 1;
        base = mont_mul(base, base, mod, mod_inv);
    }
    return result;
}

// Generate twiddle factors for NTT
__global__ void generate_twiddle_kernel(
    uint64_t* __restrict__ twiddles,
    uint64_t* __restrict__ inv_twiddles,
    uint32_t n,
    uint64_t mod,
    uint64_t primitive_root
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    
    // omega = g^((mod-1)/n) where g is primitive root
    uint64_t exp = (mod - 1) / n;
    uint64_t omega = mod_pow(primitive_root, exp, mod);
    uint64_t omega_inv = mod_pow(omega, mod - 2, mod);  // Fermat's little theorem
    
    // twiddle[i] = omega^i
    twiddles[tid] = mod_pow(omega, tid, mod);
    inv_twiddles[tid] = mod_pow(omega_inv, tid, mod);
}

// Generate twiddle factors with bit-reversal
__global__ void generate_twiddle_bitrev_kernel(
    uint64_t* __restrict__ twiddles,
    uint64_t* __restrict__ inv_twiddles,
    uint32_t n,
    uint32_t log_n,
    uint64_t mod,
    uint64_t primitive_root
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    
    // Bit-reverse the index
    uint32_t rev = 0;
    uint32_t tmp = tid;
    for (uint32_t i = 0; i < log_n; i++) {
        rev = (rev << 1) | (tmp & 1);
        tmp >>= 1;
    }
    
    uint64_t exp = (mod - 1) / n;
    uint64_t omega = mod_pow(primitive_root, exp, mod);
    uint64_t omega_inv = mod_pow(omega, mod - 2, mod);
    
    twiddles[tid] = mod_pow(omega, rev, mod);
    inv_twiddles[tid] = mod_pow(omega_inv, rev, mod);
}

// Generate twiddle factors for each NTT stage
__global__ void generate_stage_twiddles_kernel(
    uint64_t* __restrict__ twiddles,    // Output: [log_n][n/2]
    uint32_t n,
    uint32_t log_n,
    uint64_t mod,
    uint64_t omega                      // n-th root of unity
) {
    uint32_t stage = blockIdx.y;
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    uint32_t half_n = n >> 1;
    if (tid >= half_n) return;
    
    // For stage s, the twiddle factors are omega^(j * 2^(log_n - 1 - s))
    uint32_t m = 1 << (stage + 1);
    uint32_t group_size = m >> 1;
    uint32_t group = tid / group_size;
    uint32_t j = tid % group_size;
    
    uint64_t exp = j * (n / m);
    uint64_t tw = mod_pow(omega, exp, mod);
    
    twiddles[stage * half_n + tid] = tw;
}

// Generate inverse twiddle factors with n^-1 scaling
__global__ void generate_inv_twiddles_scaled_kernel(
    uint64_t* __restrict__ inv_twiddles,
    const uint64_t* __restrict__ twiddles,
    uint32_t n,
    uint64_t mod,
    uint64_t n_inv                      // n^-1 mod m
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    
    // inv_twiddle[i] = twiddle[n-i] * n^-1 (with wrap: inv_twiddle[0] = n^-1)
    uint32_t src_idx = (tid == 0) ? 0 : (n - tid);
    
    // Multiply by n^-1
    __uint128_t prod = (__uint128_t)twiddles[src_idx] * n_inv;
    inv_twiddles[tid] = (uint64_t)(prod % mod);
}

// Precompute Barrett reduction constants
__global__ void precompute_barrett_kernel(
    uint64_t* __restrict__ mu,          // Barrett multiplier
    uint64_t* __restrict__ shift,       // Barrett shift
    uint64_t mod
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // mu = floor(2^128 / mod)
        // We compute this in two steps
        uint32_t k = 128;
        
        // Simple approximation for common primes
        *shift = k;
        
        // For Goldilocks prime: mu ≈ 2^64 + 2^32
        if (mod == 0xFFFFFFFF00000001ULL) {
            *mu = 0x100000001ULL;
        } else {
            // Generic case - simplified
            *mu = (uint64_t)(-1) / (mod >> 64) + 1;
        }
    }
}

// Host wrapper functions
void generate_twiddles(
    uint64_t* twiddles,
    uint64_t* inv_twiddles,
    uint32_t n,
    uint64_t mod,
    uint64_t primitive_root,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    generate_twiddle_kernel<<<grid, block, 0, stream>>>(
        twiddles, inv_twiddles, n, mod, primitive_root
    );
}

void generate_twiddles_bitrev(
    uint64_t* twiddles,
    uint64_t* inv_twiddles,
    uint32_t n,
    uint32_t log_n,
    uint64_t mod,
    uint64_t primitive_root,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    generate_twiddle_bitrev_kernel<<<grid, block, 0, stream>>>(
        twiddles, inv_twiddles, n, log_n, mod, primitive_root
    );
}

void generate_stage_twiddles(
    uint64_t* twiddles,
    uint32_t n,
    uint32_t log_n,
    uint64_t mod,
    uint64_t omega,
    cudaStream_t stream
) {
    uint32_t half_n = n >> 1;
    dim3 block(256);
    dim3 grid((half_n + block.x - 1) / block.x, log_n);
    generate_stage_twiddles_kernel<<<grid, block, 0, stream>>>(
        twiddles, n, log_n, mod, omega
    );
}

} // namespace twiddle
} // namespace cuda
} // namespace lux
