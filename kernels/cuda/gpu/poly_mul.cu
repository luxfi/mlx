// Copyright (c) 2024-2026 Lux Industries Inc. All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Polynomial Multiplication - High-Performance CUDA Implementation
// Supports both schoolbook and NTT-based multiplication for lattice cryptography.
// Optimized for FHE and post-quantum signature schemes.
//
// Features:
//   - Schoolbook multiplication for small polynomials (N <= 64)
//   - NTT-based multiplication for large polynomials (N > 64)
//   - Montgomery form arithmetic for efficient modular operations
//   - Batch multiplication support for parallel processing
//   - Negacyclic convolution for ring R_q = Z_q[X]/(X^n + 1)

#include <cuda_runtime.h>
#include <cstdint>

namespace lux {
namespace cuda {
namespace poly_mul {

// ============================================================================
// Constants and Parameters
// ============================================================================

// Threshold for switching from schoolbook to NTT multiplication
constexpr uint32_t SCHOOLBOOK_THRESHOLD = 64;

// Maximum shared memory polynomial size
constexpr uint32_t MAX_SHMEM_SIZE = 4096;

// Common NTT-friendly primes for lattice crypto
constexpr uint64_t Q_DILITHIUM = 8380417ULL;      // 2^23 - 2^13 + 1
constexpr uint64_t Q_KYBER = 3329ULL;              // 13 * 256 + 1
constexpr uint64_t Q_GOLDILOCKS = 0xFFFFFFFF00000001ULL;

// Device constants
__constant__ uint64_t d_modulus;
__constant__ uint64_t d_modulus_inv;      // -q^{-1} mod 2^64 (Montgomery)
__constant__ uint64_t d_mont_r;           // 2^64 mod q
__constant__ uint64_t d_mont_r2;          // (2^64)^2 mod q
__constant__ uint64_t d_twiddles[8192];   // Forward NTT twiddles
__constant__ uint64_t d_twiddles_inv[8192]; // Inverse NTT twiddles
__constant__ uint64_t d_n_inv;            // n^{-1} mod q

// ============================================================================
// Montgomery Arithmetic
// ============================================================================

// Montgomery reduction: computes aR^{-1} mod q from aR mod q
__device__ __forceinline__
uint64_t mont_reduce_64(uint64_t lo, uint64_t hi, uint64_t q, uint64_t q_inv) {
    uint64_t m = lo * q_inv;
    uint64_t t = __umul64hi(m, q);
    uint64_t result = hi - t;
    if (hi < t) result += q;
    return result >= q ? result - q : result;
}

// Montgomery multiplication: computes abR^{-1} mod q
__device__ __forceinline__
uint64_t mont_mul_64(uint64_t a, uint64_t b, uint64_t q, uint64_t q_inv) {
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);
    return mont_reduce_64(lo, hi, q, q_inv);
}

// Modular addition
__device__ __forceinline__
uint64_t mod_add_64(uint64_t a, uint64_t b, uint64_t q) {
    uint64_t sum = a + b;
    return sum >= q ? sum - q : sum;
}

// Modular subtraction
__device__ __forceinline__
uint64_t mod_sub_64(uint64_t a, uint64_t b, uint64_t q) {
    return a >= b ? a - b : a + q - b;
}

// Modular negation
__device__ __forceinline__
uint64_t mod_neg_64(uint64_t a, uint64_t q) {
    return a == 0 ? 0 : q - a;
}

// 32-bit versions for smaller moduli (Kyber, Dilithium)
__device__ __forceinline__
uint32_t mont_reduce_32(uint64_t a, uint32_t q, uint32_t q_inv) {
    uint32_t m = (uint32_t)a * q_inv;
    uint64_t t = (uint64_t)m * q;
    uint32_t result = (uint32_t)((a - t) >> 32);
    return result;
}

__device__ __forceinline__
uint32_t mont_mul_32(uint32_t a, uint32_t b, uint32_t q, uint32_t q_inv) {
    return mont_reduce_32((uint64_t)a * b, q, q_inv);
}

__device__ __forceinline__
uint32_t mod_add_32(uint32_t a, uint32_t b, uint32_t q) {
    uint32_t sum = a + b;
    return sum >= q ? sum - q : sum;
}

__device__ __forceinline__
uint32_t mod_sub_32(uint32_t a, uint32_t b, uint32_t q) {
    return a >= b ? a - b : a + q - b;
}

// ============================================================================
// Schoolbook Polynomial Multiplication
// ============================================================================

// Schoolbook multiplication for small polynomials
// c(X) = a(X) * b(X) mod (X^n + 1) mod q
// Uses negacyclic convolution
__global__ void poly_mul_schoolbook_kernel(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ c,
    uint32_t n,
    uint64_t q,
    uint64_t q_inv
) {
    extern __shared__ uint64_t shmem[];
    uint64_t* s_a = shmem;
    uint64_t* s_b = shmem + n;

    const uint32_t tid = threadIdx.x;
    const uint32_t poly_idx = blockIdx.x;

    // Load polynomials to shared memory
    if (tid < n) {
        s_a[tid] = a[poly_idx * n + tid];
        s_b[tid] = b[poly_idx * n + tid];
    }
    __syncthreads();

    if (tid >= n) return;

    // Compute coefficient tid of result
    // c[k] = sum_{i+j=k} a[i]*b[j] - sum_{i+j=k+n} a[i]*b[j]
    // (negacyclic: X^n = -1)
    uint64_t sum = 0;

    for (uint32_t i = 0; i <= tid; i++) {
        uint64_t prod = mont_mul_64(s_a[i], s_b[tid - i], q, q_inv);
        sum = mod_add_64(sum, prod, q);
    }

    for (uint32_t i = tid + 1; i < n; i++) {
        uint64_t prod = mont_mul_64(s_a[i], s_b[n + tid - i], q, q_inv);
        sum = mod_sub_64(sum, prod, q);  // Subtract due to negacyclic
    }

    c[poly_idx * n + tid] = sum;
}

// 32-bit version for smaller moduli
__global__ void poly_mul_schoolbook_32_kernel(
    const uint32_t* __restrict__ a,
    const uint32_t* __restrict__ b,
    uint32_t* __restrict__ c,
    uint32_t n,
    uint32_t q,
    uint32_t q_inv
) {
    extern __shared__ uint32_t shmem32[];
    uint32_t* s_a = shmem32;
    uint32_t* s_b = shmem32 + n;

    const uint32_t tid = threadIdx.x;
    const uint32_t poly_idx = blockIdx.x;

    if (tid < n) {
        s_a[tid] = a[poly_idx * n + tid];
        s_b[tid] = b[poly_idx * n + tid];
    }
    __syncthreads();

    if (tid >= n) return;

    uint64_t sum = 0;

    // Positive terms: i + j = tid
    for (uint32_t i = 0; i <= tid; i++) {
        sum += (uint64_t)s_a[i] * s_b[tid - i];
    }

    // Negative terms: i + j = tid + n (wrap-around with negation)
    uint64_t neg_sum = 0;
    for (uint32_t i = tid + 1; i < n; i++) {
        neg_sum += (uint64_t)s_a[i] * s_b[n + tid - i];
    }

    // Reduce and combine
    sum = sum % q;
    neg_sum = neg_sum % q;

    uint32_t result = (sum >= neg_sum) ? (uint32_t)(sum - neg_sum) : (uint32_t)(sum + q - neg_sum);
    c[poly_idx * n + tid] = result;
}

// ============================================================================
// NTT-Based Polynomial Multiplication
// ============================================================================

// Cooley-Tukey butterfly for forward NTT
__device__ __forceinline__
void ct_butterfly_64(uint64_t& x0, uint64_t& x1, uint64_t w, uint64_t q, uint64_t q_inv) {
    uint64_t t = mont_mul_64(x1, w, q, q_inv);
    x1 = mod_sub_64(x0, t, q);
    x0 = mod_add_64(x0, t, q);
}

// Gentleman-Sande butterfly for inverse NTT
__device__ __forceinline__
void gs_butterfly_64(uint64_t& x0, uint64_t& x1, uint64_t w, uint64_t q, uint64_t q_inv) {
    uint64_t t = mod_sub_64(x0, x1, q);
    x0 = mod_add_64(x0, x1, q);
    x1 = mont_mul_64(t, w, q, q_inv);
}

// Forward NTT in shared memory
__device__
void ntt_forward_shared(uint64_t* poly, uint32_t n, uint32_t log_n,
                         uint64_t q, uint64_t q_inv) {
    uint32_t tid = threadIdx.x;

    for (uint32_t stage = 0; stage < log_n; stage++) {
        uint32_t half_size = 1U << stage;
        uint32_t group_size = n >> (stage + 1);

        for (uint32_t i = tid; i < n / 2; i += blockDim.x) {
            uint32_t group_id = i / half_size;
            uint32_t idx_in_group = i % half_size;

            uint32_t idx0 = group_id * (half_size << 1) + idx_in_group;
            uint32_t idx1 = idx0 + half_size;

            uint64_t w = d_twiddles[group_id + half_size];
            ct_butterfly_64(poly[idx0], poly[idx1], w, q, q_inv);
        }
        __syncthreads();
    }
}

// Inverse NTT in shared memory
__device__
void ntt_inverse_shared(uint64_t* poly, uint32_t n, uint32_t log_n,
                         uint64_t q, uint64_t q_inv, uint64_t n_inv) {
    uint32_t tid = threadIdx.x;

    for (int stage = log_n - 1; stage >= 0; stage--) {
        uint32_t half_size = 1U << stage;

        for (uint32_t i = tid; i < n / 2; i += blockDim.x) {
            uint32_t group_id = i / half_size;
            uint32_t idx_in_group = i % half_size;

            uint32_t idx0 = group_id * (half_size << 1) + idx_in_group;
            uint32_t idx1 = idx0 + half_size;

            uint64_t w = d_twiddles_inv[group_id + half_size];
            gs_butterfly_64(poly[idx0], poly[idx1], w, q, q_inv);
        }
        __syncthreads();
    }

    // Scale by n^{-1}
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        poly[i] = mont_mul_64(poly[i], n_inv, q, q_inv);
    }
    __syncthreads();
}

// NTT-based polynomial multiplication
// c = INTT(NTT(a) * NTT(b))
__global__ void poly_mul_ntt_kernel(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ c,
    uint32_t n,
    uint32_t log_n,
    uint64_t q,
    uint64_t q_inv,
    uint64_t n_inv
) {
    extern __shared__ uint64_t shmem[];
    uint64_t* s_a = shmem;
    uint64_t* s_b = shmem + n;

    const uint32_t tid = threadIdx.x;
    const uint32_t poly_idx = blockIdx.x;

    // Load polynomials
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        s_a[i] = a[poly_idx * n + i];
        s_b[i] = b[poly_idx * n + i];
    }
    __syncthreads();

    // Forward NTT on both polynomials
    ntt_forward_shared(s_a, n, log_n, q, q_inv);
    ntt_forward_shared(s_b, n, log_n, q, q_inv);

    // Pointwise multiplication in NTT domain
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        s_a[i] = mont_mul_64(s_a[i], s_b[i], q, q_inv);
    }
    __syncthreads();

    // Inverse NTT
    ntt_inverse_shared(s_a, n, log_n, q, q_inv, n_inv);

    // Store result
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        c[poly_idx * n + i] = s_a[i];
    }
}

// Pointwise multiplication (when inputs are already in NTT domain)
__global__ void poly_mul_pointwise_kernel(
    const uint64_t* __restrict__ a_ntt,
    const uint64_t* __restrict__ b_ntt,
    uint64_t* __restrict__ c_ntt,
    uint32_t n,
    uint64_t q,
    uint64_t q_inv
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    c_ntt[idx] = mont_mul_64(a_ntt[idx], b_ntt[idx], q, q_inv);
}

// ============================================================================
// Batch Polynomial Multiplication
// ============================================================================

// Batch schoolbook multiplication
__global__ void poly_mul_batch_schoolbook_kernel(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ c,
    uint32_t n,
    uint32_t batch_size,
    uint64_t q,
    uint64_t q_inv
) {
    extern __shared__ uint64_t shmem[];
    uint64_t* s_a = shmem;
    uint64_t* s_b = shmem + n;

    const uint32_t tid = threadIdx.x;
    const uint32_t batch_idx = blockIdx.x;

    if (batch_idx >= batch_size) return;

    // Load polynomials
    if (tid < n) {
        s_a[tid] = a[batch_idx * n + tid];
        s_b[tid] = b[batch_idx * n + tid];
    }
    __syncthreads();

    if (tid >= n) return;

    // Compute negacyclic convolution
    uint64_t sum = 0;

    for (uint32_t i = 0; i <= tid; i++) {
        uint64_t prod = mont_mul_64(s_a[i], s_b[tid - i], q, q_inv);
        sum = mod_add_64(sum, prod, q);
    }

    for (uint32_t i = tid + 1; i < n; i++) {
        uint64_t prod = mont_mul_64(s_a[i], s_b[n + tid - i], q, q_inv);
        sum = mod_sub_64(sum, prod, q);
    }

    c[batch_idx * n + tid] = sum;
}

// Batch NTT multiplication
__global__ void poly_mul_batch_ntt_kernel(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ c,
    uint32_t n,
    uint32_t log_n,
    uint32_t batch_size,
    uint64_t q,
    uint64_t q_inv,
    uint64_t n_inv
) {
    extern __shared__ uint64_t shmem[];
    uint64_t* s_a = shmem;
    uint64_t* s_b = shmem + n;

    const uint32_t tid = threadIdx.x;
    const uint32_t batch_idx = blockIdx.x;

    if (batch_idx >= batch_size) return;

    // Load polynomials
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        s_a[i] = a[batch_idx * n + i];
        s_b[i] = b[batch_idx * n + i];
    }
    __syncthreads();

    // Forward NTT
    ntt_forward_shared(s_a, n, log_n, q, q_inv);
    ntt_forward_shared(s_b, n, log_n, q, q_inv);

    // Pointwise multiply
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        s_a[i] = mont_mul_64(s_a[i], s_b[i], q, q_inv);
    }
    __syncthreads();

    // Inverse NTT
    ntt_inverse_shared(s_a, n, log_n, q, q_inv, n_inv);

    // Store
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        c[batch_idx * n + i] = s_a[i];
    }
}

// ============================================================================
// Fused Operations
// ============================================================================

// Multiply-accumulate: c = c + a * b
__global__ void poly_mul_acc_kernel(
    const uint64_t* __restrict__ a_ntt,
    const uint64_t* __restrict__ b_ntt,
    uint64_t* __restrict__ c_ntt,
    uint32_t n,
    uint64_t q,
    uint64_t q_inv
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint64_t prod = mont_mul_64(a_ntt[idx], b_ntt[idx], q, q_inv);
    c_ntt[idx] = mod_add_64(c_ntt[idx], prod, q);
}

// Scalar multiply: c = a * scalar
__global__ void poly_scalar_mul_kernel(
    const uint64_t* __restrict__ a,
    uint64_t scalar,
    uint64_t* __restrict__ c,
    uint32_t n,
    uint64_t q,
    uint64_t q_inv
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    c[idx] = mont_mul_64(a[idx], scalar, q, q_inv);
}

// ============================================================================
// Karatsuba Multiplication (for medium-sized polynomials)
// ============================================================================

// Karatsuba step: computes a*b using 3 multiplications instead of 4
// For polynomials split as a = a0 + a1*X^{n/2}, b = b0 + b1*X^{n/2}
// a*b = a0*b0 + (a0*b1 + a1*b0)*X^{n/2} + a1*b1*X^n
//     = a0*b0 + ((a0+a1)*(b0+b1) - a0*b0 - a1*b1)*X^{n/2} + a1*b1*X^n
__global__ void poly_mul_karatsuba_combine_kernel(
    const uint64_t* __restrict__ low,      // a0*b0
    const uint64_t* __restrict__ mid,      // (a0+a1)*(b0+b1)
    const uint64_t* __restrict__ high,     // a1*b1
    uint64_t* __restrict__ result,
    uint32_t half_n,
    uint64_t q
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t n = half_n * 2;

    if (idx >= n * 2 - 1) return;

    uint64_t sum = 0;

    // Low contribution [0, n-1]
    if (idx < n) {
        sum = mod_add_64(sum, low[idx], q);
    }

    // High contribution [n, 2n-2] -> shifted to [0, n-2]
    if (idx >= n && idx < 2 * n - 1) {
        // For negacyclic: subtract since X^n = -1
        sum = mod_sub_64(sum, high[idx - n], q);
    }

    // Middle contribution [n/2, 3n/2-1]
    if (idx >= half_n && idx < half_n + n) {
        uint64_t mid_val = mid[idx - half_n];
        // Subtract low and high contributions from mid
        if (idx - half_n < n) {
            mid_val = mod_sub_64(mid_val, low[idx - half_n], q);
        }
        if (idx - half_n < n - 1) {
            mid_val = mod_sub_64(mid_val, high[idx - half_n], q);
        }
        sum = mod_add_64(sum, mid_val, q);
    }

    result[idx % n] = sum;
}

// ============================================================================
// Host API (C Linkage)
// ============================================================================

} // namespace poly_mul
} // namespace cuda
} // namespace lux

#ifdef __cplusplus
extern "C" {
#endif

// Initialize polynomial multiplication context
cudaError_t lux_cuda_poly_mul_init(
    uint64_t modulus,
    uint64_t* twiddles,
    uint64_t* twiddles_inv,
    uint32_t max_n
) {
    // Copy modulus parameters
    cudaMemcpyToSymbol(lux::cuda::poly_mul::d_modulus, &modulus, sizeof(uint64_t));

    // Compute Montgomery parameters
    uint64_t r2 = 1;
    for (int i = 0; i < 128; i++) {
        r2 = (r2 << 1) % modulus;
    }
    cudaMemcpyToSymbol(lux::cuda::poly_mul::d_mont_r2, &r2, sizeof(uint64_t));

    // Compute -q^{-1} mod 2^64 using extended Euclidean algorithm
    uint64_t q_inv = 1;
    uint64_t t = modulus;
    for (int i = 0; i < 63; i++) {
        q_inv *= 2 - t * q_inv;
        t = t * t;
    }
    q_inv = -q_inv;  // We want -q^{-1}
    cudaMemcpyToSymbol(lux::cuda::poly_mul::d_modulus_inv, &q_inv, sizeof(uint64_t));

    // Copy twiddle factors
    if (twiddles && max_n <= 8192) {
        cudaMemcpyToSymbol(lux::cuda::poly_mul::d_twiddles, twiddles,
                           max_n * sizeof(uint64_t));
    }
    if (twiddles_inv && max_n <= 8192) {
        cudaMemcpyToSymbol(lux::cuda::poly_mul::d_twiddles_inv, twiddles_inv,
                           max_n * sizeof(uint64_t));
    }

    return cudaGetLastError();
}

// Polynomial multiplication (auto-selects algorithm)
cudaError_t lux_cuda_poly_mul(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* c,
    uint32_t n,
    uint64_t q,
    uint64_t q_inv,
    uint64_t n_inv,
    cudaStream_t stream
) {
    using namespace lux::cuda::poly_mul;

    if (n <= SCHOOLBOOK_THRESHOLD) {
        // Use schoolbook for small polynomials
        dim3 block(n);
        dim3 grid(1);
        size_t shmem = 2 * n * sizeof(uint64_t);

        poly_mul_schoolbook_kernel<<<grid, block, shmem, stream>>>(
            a, b, c, n, q, q_inv
        );
    } else {
        // Use NTT for large polynomials
        uint32_t log_n = 0;
        for (uint32_t t = n; t > 1; t >>= 1) log_n++;

        dim3 block(min(n / 2, 256U));
        dim3 grid(1);
        size_t shmem = 2 * n * sizeof(uint64_t);

        poly_mul_ntt_kernel<<<grid, block, shmem, stream>>>(
            a, b, c, n, log_n, q, q_inv, n_inv
        );
    }

    return cudaGetLastError();
}

// Batch polynomial multiplication
cudaError_t lux_cuda_poly_mul_batch(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* c,
    uint32_t n,
    uint32_t batch_size,
    uint64_t q,
    uint64_t q_inv,
    uint64_t n_inv,
    cudaStream_t stream
) {
    using namespace lux::cuda::poly_mul;

    if (n <= SCHOOLBOOK_THRESHOLD) {
        dim3 block(n);
        dim3 grid(batch_size);
        size_t shmem = 2 * n * sizeof(uint64_t);

        poly_mul_batch_schoolbook_kernel<<<grid, block, shmem, stream>>>(
            a, b, c, n, batch_size, q, q_inv
        );
    } else {
        uint32_t log_n = 0;
        for (uint32_t t = n; t > 1; t >>= 1) log_n++;

        dim3 block(min(n / 2, 256U));
        dim3 grid(batch_size);
        size_t shmem = 2 * n * sizeof(uint64_t);

        poly_mul_batch_ntt_kernel<<<grid, block, shmem, stream>>>(
            a, b, c, n, log_n, batch_size, q, q_inv, n_inv
        );
    }

    return cudaGetLastError();
}

// Pointwise multiplication (NTT domain)
cudaError_t lux_cuda_poly_mul_pointwise(
    const uint64_t* a_ntt,
    const uint64_t* b_ntt,
    uint64_t* c_ntt,
    uint32_t n,
    uint64_t q,
    uint64_t q_inv,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);

    lux::cuda::poly_mul::poly_mul_pointwise_kernel<<<grid, block, 0, stream>>>(
        a_ntt, b_ntt, c_ntt, n, q, q_inv
    );

    return cudaGetLastError();
}

// Multiply-accumulate (NTT domain): c += a * b
cudaError_t lux_cuda_poly_mul_acc(
    const uint64_t* a_ntt,
    const uint64_t* b_ntt,
    uint64_t* c_ntt,
    uint32_t n,
    uint64_t q,
    uint64_t q_inv,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);

    lux::cuda::poly_mul::poly_mul_acc_kernel<<<grid, block, 0, stream>>>(
        a_ntt, b_ntt, c_ntt, n, q, q_inv
    );

    return cudaGetLastError();
}

// Scalar multiplication
cudaError_t lux_cuda_poly_scalar_mul(
    const uint64_t* a,
    uint64_t scalar,
    uint64_t* c,
    uint32_t n,
    uint64_t q,
    uint64_t q_inv,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);

    lux::cuda::poly_mul::poly_scalar_mul_kernel<<<grid, block, 0, stream>>>(
        a, scalar, c, n, q, q_inv
    );

    return cudaGetLastError();
}

// Cleanup
cudaError_t lux_cuda_poly_mul_cleanup(void) {
    return cudaSuccess;
}

#ifdef __cplusplus
}
#endif
