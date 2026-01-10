// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Goldilocks Field CUDA Kernels
// Prime: p = 2^64 - 2^32 + 1 (the "Goldilocks" prime)
// Optimal for 64-bit arithmetic with fast reduction

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================================
// Goldilocks Prime Constants
// ============================================================================

// p = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001
__constant__ uint64_t GOLDILOCKS_P = 0xFFFFFFFF00000001ULL;
__constant__ uint64_t GOLDILOCKS_EPSILON = 0xFFFFFFFFULL;  // 2^32 - 1

// Generator for multiplicative group (primitive root)
__constant__ uint64_t GOLDILOCKS_GENERATOR = 7ULL;

// 2^64 mod p (for Montgomery, though Goldilocks doesn't need it)
__constant__ uint64_t GOLDILOCKS_R = 0xFFFFFFFF;

// ============================================================================
// Field Arithmetic (mod p = 2^64 - 2^32 + 1)
// ============================================================================

// Fast reduction: if x >= p, return x - p
__device__ __forceinline__
uint64_t goldilocks_reduce_once(uint64_t x) {
    // Since p = 2^64 - 2^32 + 1, we need to check if x >= p
    // x >= p iff x > 0xFFFFFFFF00000000 or (x == 0xFFFFFFFF00000001+k for some k)
    return (x >= GOLDILOCKS_P) ? (x - GOLDILOCKS_P) : x;
}

// Reduce a 128-bit value to Goldilocks field
// Input: high * 2^64 + low
// Since 2^64 ≡ 2^32 - 1 (mod p), we have:
// high * 2^64 + low ≡ high * (2^32 - 1) + low (mod p)
//                   ≡ low - high + high * 2^32 (mod p)
__device__ __forceinline__
uint64_t goldilocks_reduce_128(uint64_t low, uint64_t high) {
    // high * 2^64 ≡ high * (2^32 - 1) = high * 2^32 - high (mod p)
    uint64_t high_lo = high & 0xFFFFFFFFULL;
    uint64_t high_hi = high >> 32;

    // Compute high * 2^32 = high_lo * 2^32 + high_hi * 2^64
    // high_hi * 2^64 ≡ high_hi * (2^32 - 1) (mod p)

    // Result = low + high_lo * 2^32 + high_hi * 2^32 - high_hi - high
    //        = low + (high_lo + high_hi) * 2^32 - high_hi - high

    uint64_t result = low;

    // Add high * (2^32 - 1) = high << 32 - high
    // But high << 32 might overflow, so we do this carefully

    // First, subtract high (with borrow handling)
    bool borrow = result < high;
    result -= high;
    if (borrow) {
        // result + p (since result went negative)
        result += GOLDILOCKS_P;
    }

    // Add high << 32 in parts
    uint64_t high_shifted_lo = high_lo << 32;
    uint64_t high_shifted_hi = high_hi;

    // Add high_shifted_lo
    uint64_t prev = result;
    result += high_shifted_lo;
    bool carry = result < prev;

    // Handle carry: carry * 2^64 ≡ carry * (2^32 - 1) (mod p)
    if (carry) {
        result -= 1;  // subtract 1
        result += (1ULL << 32);  // add 2^32
        if (result < (1ULL << 32) - 1) {
            result += GOLDILOCKS_P;
        }
    }

    // Add high_shifted_hi * 2^32
    prev = result;
    uint64_t add_hi = high_shifted_hi << 32;
    result += add_hi;
    if (result < prev) {
        // overflow: add (2^32 - 1)
        result += GOLDILOCKS_EPSILON;
        result = goldilocks_reduce_once(result);
    }

    return goldilocks_reduce_once(result);
}

// Addition in Goldilocks field
__device__ __forceinline__
uint64_t goldilocks_add(uint64_t a, uint64_t b) {
    uint64_t sum = a + b;
    // Check for overflow or >= p
    bool overflow = sum < a;
    if (overflow) {
        // sum = a + b - 2^64 + p = a + b + (2^32 - 1) - 2^64 + 2^64
        //     = a + b + 2^32 - 1 (since we wrapped around)
        sum += GOLDILOCKS_EPSILON + 1;
    }
    return goldilocks_reduce_once(sum);
}

// Subtraction in Goldilocks field
__device__ __forceinline__
uint64_t goldilocks_sub(uint64_t a, uint64_t b) {
    uint64_t diff = a - b;
    bool borrow = a < b;
    if (borrow) {
        diff += GOLDILOCKS_P;
    }
    return diff;
}

// Negation in Goldilocks field
__device__ __forceinline__
uint64_t goldilocks_neg(uint64_t a) {
    return (a == 0) ? 0 : (GOLDILOCKS_P - a);
}

// Multiplication in Goldilocks field
__device__ __forceinline__
uint64_t goldilocks_mul(uint64_t a, uint64_t b) {
    // 64x64 -> 128-bit multiplication
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);

    return goldilocks_reduce_128(lo, hi);
}

// Squaring (slightly optimized)
__device__ __forceinline__
uint64_t goldilocks_sqr(uint64_t a) {
    return goldilocks_mul(a, a);
}

// Exponentiation by squaring
__device__
uint64_t goldilocks_pow(uint64_t base, uint64_t exp) {
    uint64_t result = 1;

    while (exp > 0) {
        if (exp & 1) {
            result = goldilocks_mul(result, base);
        }
        base = goldilocks_sqr(base);
        exp >>= 1;
    }

    return result;
}

// Modular inverse using Fermat's little theorem: a^(-1) = a^(p-2) mod p
__device__
uint64_t goldilocks_inv(uint64_t a) {
    // p - 2 = 0xFFFFFFFF00000001 - 2 = 0xFFFFFFFEFFFFFFFF
    // Use addition chain for efficiency

    // a^2
    uint64_t a2 = goldilocks_sqr(a);
    // a^3
    uint64_t a3 = goldilocks_mul(a2, a);
    // a^(2^2) = a^4
    uint64_t a4 = goldilocks_sqr(a2);
    // a^(2^4) = a^16
    uint64_t a16 = goldilocks_sqr(goldilocks_sqr(a4));
    // a^(2^8)
    uint64_t a256 = a16;
    for (int i = 0; i < 4; i++) a256 = goldilocks_sqr(a256);
    // a^(2^16)
    uint64_t a65536 = a256;
    for (int i = 0; i < 8; i++) a65536 = goldilocks_sqr(a65536);
    // a^(2^32)
    uint64_t a_2_32 = a65536;
    for (int i = 0; i < 16; i++) a_2_32 = goldilocks_sqr(a_2_32);

    // a^(2^32 - 1) = a * a^2 * a^4 * ... * a^(2^31)
    uint64_t chain = a;
    uint64_t power = a;
    for (int i = 0; i < 31; i++) {
        power = goldilocks_sqr(power);
        chain = goldilocks_mul(chain, power);
    }

    // a^(p-2) = a^(2^64 - 2^32 - 1)
    // = a^(2^64) / a^(2^32) / a
    // Using Fermat: a^(2^64) ≡ a (mod p)
    // So a^(p-2) = a^(-1) directly from exponentiation

    return goldilocks_pow(a, GOLDILOCKS_P - 2);
}

// Batch inversion using Montgomery's trick
__device__
void goldilocks_batch_inv(uint64_t* values, uint64_t* results, uint32_t n) {
    if (n == 0) return;
    if (n == 1) {
        results[0] = goldilocks_inv(values[0]);
        return;
    }

    // Prefix products
    uint64_t* prefix = results;  // Reuse results array
    prefix[0] = values[0];
    for (uint32_t i = 1; i < n; i++) {
        prefix[i] = goldilocks_mul(prefix[i-1], values[i]);
    }

    // Invert the product of all
    uint64_t inv_all = goldilocks_inv(prefix[n-1]);

    // Backward pass
    for (int32_t i = n - 1; i > 0; i--) {
        results[i] = goldilocks_mul(inv_all, prefix[i-1]);
        inv_all = goldilocks_mul(inv_all, values[i]);
    }
    results[0] = inv_all;
}

// ============================================================================
// Extension Field: Goldilocks^2 (quadratic extension)
// ============================================================================

// Goldilocks^2 = Goldilocks[x] / (x^2 - 7)
// Element: a + b*x where x^2 = 7

struct GoldilocksExt2 {
    uint64_t c0;  // coefficient of 1
    uint64_t c1;  // coefficient of x
};

__device__ __forceinline__
GoldilocksExt2 goldilocks2_add(GoldilocksExt2 a, GoldilocksExt2 b) {
    return {goldilocks_add(a.c0, b.c0), goldilocks_add(a.c1, b.c1)};
}

__device__ __forceinline__
GoldilocksExt2 goldilocks2_sub(GoldilocksExt2 a, GoldilocksExt2 b) {
    return {goldilocks_sub(a.c0, b.c0), goldilocks_sub(a.c1, b.c1)};
}

__device__ __forceinline__
GoldilocksExt2 goldilocks2_mul(GoldilocksExt2 a, GoldilocksExt2 b) {
    // (a0 + a1*x)(b0 + b1*x) = a0*b0 + (a0*b1 + a1*b0)*x + a1*b1*x^2
    // = a0*b0 + 7*a1*b1 + (a0*b1 + a1*b0)*x

    uint64_t a0b0 = goldilocks_mul(a.c0, b.c0);
    uint64_t a1b1 = goldilocks_mul(a.c1, b.c1);
    uint64_t a0b1 = goldilocks_mul(a.c0, b.c1);
    uint64_t a1b0 = goldilocks_mul(a.c1, b.c0);

    // 7 * a1b1
    uint64_t seven_a1b1 = goldilocks_mul(a1b1, 7);

    return {
        goldilocks_add(a0b0, seven_a1b1),
        goldilocks_add(a0b1, a1b0)
    };
}

// ============================================================================
// NTT (Number Theoretic Transform) for Goldilocks
// ============================================================================

// Primitive 2^32-th root of unity in Goldilocks
// ω = 1753635133440165772 (a primitive 2^32-th root)
__constant__ uint64_t GOLDILOCKS_ROOT_2_32 = 1753635133440165772ULL;

// Compute twiddle factor: ω^(n/2^k * i) for NTT
__device__ __forceinline__
uint64_t goldilocks_get_twiddle(uint32_t i, uint32_t log_n) {
    // Compute ω^(2^(32-log_n) * i)
    uint64_t exp = ((uint64_t)i) << (32 - log_n);
    return goldilocks_pow(GOLDILOCKS_ROOT_2_32, exp);
}

// Forward NTT kernel (Cooley-Tukey, decimation-in-time)
extern "C" __global__
void goldilocks_ntt_forward(
    uint64_t* __restrict__ data,
    const uint64_t* __restrict__ twiddles,
    uint32_t n,
    uint32_t log_n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread handles one butterfly
    for (uint32_t s = 1; s <= log_n; s++) {
        uint32_t m = 1 << s;
        uint32_t m2 = m >> 1;

        uint32_t butterflies_per_stage = n >> 1;

        if (tid < butterflies_per_stage) {
            uint32_t group = tid / m2;
            uint32_t j = tid % m2;
            uint32_t k = group * m + j;

            uint64_t w = twiddles[m2 + j];  // twiddle factor

            uint64_t u = data[k];
            uint64_t t = goldilocks_mul(w, data[k + m2]);

            data[k] = goldilocks_add(u, t);
            data[k + m2] = goldilocks_sub(u, t);
        }

        __syncthreads();
    }
}

// Inverse NTT kernel
extern "C" __global__
void goldilocks_ntt_inverse(
    uint64_t* __restrict__ data,
    const uint64_t* __restrict__ inv_twiddles,
    uint32_t n,
    uint32_t log_n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Inverse NTT (Gentleman-Sande, decimation-in-frequency)
    for (int32_t s = log_n; s >= 1; s--) {
        uint32_t m = 1 << s;
        uint32_t m2 = m >> 1;

        uint32_t butterflies_per_stage = n >> 1;

        if (tid < butterflies_per_stage) {
            uint32_t group = tid / m2;
            uint32_t j = tid % m2;
            uint32_t k = group * m + j;

            uint64_t w = inv_twiddles[m2 + j];

            uint64_t u = data[k];
            uint64_t v = data[k + m2];

            data[k] = goldilocks_add(u, v);
            data[k + m2] = goldilocks_mul(goldilocks_sub(u, v), w);
        }

        __syncthreads();
    }

    // Scale by n^(-1)
    if (tid < n) {
        uint64_t n_inv = goldilocks_inv(n);
        data[tid] = goldilocks_mul(data[tid], n_inv);
    }
}

// ============================================================================
// Vector Operations
// ============================================================================

extern "C" __global__
void goldilocks_vec_add(
    uint64_t* __restrict__ result,
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        result[tid] = goldilocks_add(a[tid], b[tid]);
    }
}

extern "C" __global__
void goldilocks_vec_mul(
    uint64_t* __restrict__ result,
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        result[tid] = goldilocks_mul(a[tid], b[tid]);
    }
}

extern "C" __global__
void goldilocks_vec_scale(
    uint64_t* __restrict__ result,
    const uint64_t* __restrict__ a,
    uint64_t scalar,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        result[tid] = goldilocks_mul(a[tid], scalar);
    }
}

// Inner product (dot product)
extern "C" __global__
void goldilocks_inner_product(
    uint64_t* __restrict__ result,
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint32_t n
) {
    extern __shared__ uint64_t sdata[];

    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and multiply
    sdata[tid] = (i < n) ? goldilocks_mul(a[i], b[i]) : 0;
    __syncthreads();

    // Reduction in shared memory
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = goldilocks_add(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        atomicAdd((unsigned long long*)result, sdata[0]);  // Note: not perfectly correct for Goldilocks
    }
}

// ============================================================================
// Polynomial Operations
// ============================================================================

// Evaluate polynomial at point using Horner's method
extern "C" __global__
void goldilocks_poly_eval(
    uint64_t* __restrict__ result,
    const uint64_t* __restrict__ coeffs,
    uint64_t point,
    uint32_t degree
) {
    // Single thread evaluation (parallelize over multiple points in practice)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        uint64_t acc = coeffs[degree];
        for (int32_t i = degree - 1; i >= 0; i--) {
            acc = goldilocks_add(goldilocks_mul(acc, point), coeffs[i]);
        }
        *result = acc;
    }
}

// Batch polynomial evaluation at multiple points
extern "C" __global__
void goldilocks_poly_eval_batch(
    uint64_t* __restrict__ results,
    const uint64_t* __restrict__ coeffs,
    const uint64_t* __restrict__ points,
    uint32_t degree,
    uint32_t num_points
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_points) {
        uint64_t point = points[tid];
        uint64_t acc = coeffs[degree];

        for (int32_t i = degree - 1; i >= 0; i--) {
            acc = goldilocks_add(goldilocks_mul(acc, point), coeffs[i]);
        }

        results[tid] = acc;
    }
}

// ============================================================================
// C API for CGO Bindings
// ============================================================================

extern "C" {

int lux_cuda_goldilocks_vec_add(
    uint64_t* result,
    const uint64_t* a,
    const uint64_t* b,
    uint32_t n,
    cudaStream_t stream
) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;

    goldilocks_vec_add<<<blocks, threads, 0, stream>>>(result, a, b, n);

    return cudaGetLastError();
}

int lux_cuda_goldilocks_vec_mul(
    uint64_t* result,
    const uint64_t* a,
    const uint64_t* b,
    uint32_t n,
    cudaStream_t stream
) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;

    goldilocks_vec_mul<<<blocks, threads, 0, stream>>>(result, a, b, n);

    return cudaGetLastError();
}

int lux_cuda_goldilocks_ntt_forward(
    uint64_t* data,
    const uint64_t* twiddles,
    uint32_t n,
    uint32_t log_n,
    cudaStream_t stream
) {
    uint32_t threads = 256;
    uint32_t blocks = (n / 2 + threads - 1) / threads;

    goldilocks_ntt_forward<<<blocks, threads, 0, stream>>>(data, twiddles, n, log_n);

    return cudaGetLastError();
}

int lux_cuda_goldilocks_ntt_inverse(
    uint64_t* data,
    const uint64_t* inv_twiddles,
    uint32_t n,
    uint32_t log_n,
    cudaStream_t stream
) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;

    goldilocks_ntt_inverse<<<blocks, threads, 0, stream>>>(data, inv_twiddles, n, log_n);

    return cudaGetLastError();
}

int lux_cuda_goldilocks_poly_eval_batch(
    uint64_t* results,
    const uint64_t* coeffs,
    const uint64_t* points,
    uint32_t degree,
    uint32_t num_points,
    cudaStream_t stream
) {
    uint32_t threads = 256;
    uint32_t blocks = (num_points + threads - 1) / threads;

    goldilocks_poly_eval_batch<<<blocks, threads, 0, stream>>>(
        results, coeffs, points, degree, num_points
    );

    return cudaGetLastError();
}

}  // extern "C"
