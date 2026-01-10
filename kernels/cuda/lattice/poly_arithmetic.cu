// Copyright (c) 2024-2026 Lux Industries Inc. All Rights Reserved.
// PROPRIETARY AND CONFIDENTIAL - NO LICENSE GRANTED
// Contact: licensing@luxindustries.xyz
//
// Polynomial Arithmetic Operations - High-Performance CUDA Implementation
// Supports NTT-domain and coefficient-domain operations for FHE and lattice crypto.
// Optimized for batched polynomial processing with matrix operations.

#include <cuda_runtime.h>
#include <cstdint>

namespace lux {
namespace cuda {
namespace poly {

// ============================================================================
// Constants and Parameters
// ============================================================================

// Maximum shared memory polynomial size
constexpr uint32_t MAX_SHMEM_POLY = 4096;

// Device constants
__constant__ uint64_t d_modulus;
__constant__ uint64_t d_modulus_inv;      // -q^{-1} mod 2^64
__constant__ uint64_t d_barrett_mu;       // Barrett constant
__constant__ uint64_t d_mont_r2;          // R^2 mod q

// ============================================================================
// Modular Arithmetic Primitives
// ============================================================================

__device__ __forceinline__
uint64_t mod_add(uint64_t a, uint64_t b, uint64_t q) {
    uint64_t sum = a + b;
    return sum >= q ? sum - q : sum;
}

__device__ __forceinline__
uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t q) {
    return a >= b ? a - b : a + q - b;
}

__device__ __forceinline__
uint64_t mod_neg(uint64_t a, uint64_t q) {
    return a == 0 ? 0 : q - a;
}

__device__ __forceinline__
uint64_t mont_reduce(uint64_t lo, uint64_t hi, uint64_t q, uint64_t q_inv) {
    uint64_t m = lo * q_inv;
    uint64_t t = __umul64hi(m, q);
    uint64_t result = hi - t;
    if (hi < t) result += q;
    return result >= q ? result - q : result;
}

__device__ __forceinline__
uint64_t mont_mul(uint64_t a, uint64_t b, uint64_t q, uint64_t q_inv) {
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);
    return mont_reduce(lo, hi, q, q_inv);
}

__device__ __forceinline__
uint64_t barrett_reduce(uint64_t a, uint64_t q, uint64_t mu) {
    uint64_t quot = __umul64hi(a, mu);
    uint64_t rem = a - quot * q;
    return rem >= q ? rem - q : rem;
}

// ============================================================================
// Coefficient-wise Polynomial Operations
// ============================================================================

// Polynomial addition: c[i] = a[i] + b[i] mod q
__global__ void poly_add_kernel(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ c,
    uint32_t n,
    uint64_t q
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    c[idx] = mod_add(a[idx], b[idx], q);
}

// Polynomial subtraction: c[i] = a[i] - b[i] mod q
__global__ void poly_sub_kernel(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ c,
    uint32_t n,
    uint64_t q
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    c[idx] = mod_sub(a[idx], b[idx], q);
}

// Polynomial negation: c[i] = -a[i] mod q
__global__ void poly_neg_kernel(
    const uint64_t* __restrict__ a,
    uint64_t* __restrict__ c,
    uint32_t n,
    uint64_t q
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    c[idx] = mod_neg(a[idx], q);
}

// Scalar multiplication: c[i] = a[i] * scalar mod q
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
    
    c[idx] = mont_mul(a[idx], scalar, q, q_inv);
}

// ============================================================================
// NTT-Domain Polynomial Multiplication
// ============================================================================

// Pointwise multiplication in NTT domain: c[i] = a[i] * b[i] mod q
__global__ void poly_mul_ntt_kernel(
    const uint64_t* __restrict__ a_ntt,
    const uint64_t* __restrict__ b_ntt,
    uint64_t* __restrict__ c_ntt,
    uint32_t n,
    uint64_t q,
    uint64_t q_inv
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    c_ntt[idx] = mont_mul(a_ntt[idx], b_ntt[idx], q, q_inv);
}

// Fused multiply-add: c[i] = a[i] * b[i] + c[i] mod q
__global__ void poly_mul_add_ntt_kernel(
    const uint64_t* __restrict__ a_ntt,
    const uint64_t* __restrict__ b_ntt,
    uint64_t* __restrict__ c_ntt,
    uint32_t n,
    uint64_t q,
    uint64_t q_inv
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    uint64_t prod = mont_mul(a_ntt[idx], b_ntt[idx], q, q_inv);
    c_ntt[idx] = mod_add(c_ntt[idx], prod, q);
}

// Fused multiply-subtract: c[i] = c[i] - a[i] * b[i] mod q
__global__ void poly_mul_sub_ntt_kernel(
    const uint64_t* __restrict__ a_ntt,
    const uint64_t* __restrict__ b_ntt,
    uint64_t* __restrict__ c_ntt,
    uint32_t n,
    uint64_t q,
    uint64_t q_inv
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    uint64_t prod = mont_mul(a_ntt[idx], b_ntt[idx], q, q_inv);
    c_ntt[idx] = mod_sub(c_ntt[idx], prod, q);
}

// ============================================================================
// Batch Polynomial Operations
// ============================================================================

// Batch polynomial addition: c[batch][i] = a[batch][i] + b[batch][i]
__global__ void poly_batch_add_kernel(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ c,
    uint32_t n,
    uint32_t batch_size,
    uint64_t q
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = n * batch_size;
    if (idx >= total) return;
    
    c[idx] = mod_add(a[idx], b[idx], q);
}

// Batch polynomial multiplication in NTT domain
__global__ void poly_batch_mul_ntt_kernel(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ c,
    uint32_t n,
    uint32_t batch_size,
    uint64_t q,
    uint64_t q_inv
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = n * batch_size;
    if (idx >= total) return;
    
    c[idx] = mont_mul(a[idx], b[idx], q, q_inv);
}

// ============================================================================
// Matrix-Vector Polynomial Operations (for RLWE/RGSW)
// ============================================================================

// Matrix-vector multiply: c[i] = sum_j(A[i][j] * v[j])
// All polynomials in NTT domain
__global__ void poly_matrix_vec_mul_kernel(
    const uint64_t* __restrict__ A,         // Matrix [rows][cols][n]
    const uint64_t* __restrict__ v,         // Vector [cols][n]
    uint64_t* __restrict__ c,               // Output [rows][n]
    uint32_t rows,
    uint32_t cols,
    uint32_t n,
    uint64_t q,
    uint64_t q_inv
) {
    extern __shared__ uint64_t shmem[];
    
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t stride = blockDim.x;
    
    if (row >= rows) return;
    
    // Initialize accumulator in shared memory
    for (uint32_t i = tid; i < n; i += stride) {
        shmem[i] = 0;
    }
    __syncthreads();
    
    // Accumulate A[row][j] * v[j] for all j
    for (uint32_t col = 0; col < cols; col++) {
        for (uint32_t i = tid; i < n; i += stride) {
            uint64_t a_val = A[(row * cols + col) * n + i];
            uint64_t v_val = v[col * n + i];
            uint64_t prod = mont_mul(a_val, v_val, q, q_inv);
            shmem[i] = mod_add(shmem[i], prod, q);
        }
        __syncthreads();
    }
    
    // Write output
    for (uint32_t i = tid; i < n; i += stride) {
        c[row * n + i] = shmem[i];
    }
}

// Outer product: C[i][j] = a[i] * b[j] (element-wise in NTT domain)
__global__ void poly_outer_product_kernel(
    const uint64_t* __restrict__ a,         // Vector [rows][n]
    const uint64_t* __restrict__ b,         // Vector [cols][n]
    uint64_t* __restrict__ C,               // Matrix [rows][cols][n]
    uint32_t rows,
    uint32_t cols,
    uint32_t n,
    uint64_t q,
    uint64_t q_inv
) {
    const uint32_t row = blockIdx.y;
    const uint32_t col = blockIdx.z;
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= rows || col >= cols || idx >= n) return;
    
    uint64_t a_val = a[row * n + idx];
    uint64_t b_val = b[col * n + idx];
    C[(row * cols + col) * n + idx] = mont_mul(a_val, b_val, q, q_inv);
}

// ============================================================================
// Coefficient Reduction and Normalization
// ============================================================================

// Reduce coefficients to [0, q)
__global__ void poly_reduce_kernel(
    uint64_t* __restrict__ poly,
    uint32_t n,
    uint64_t q
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    uint64_t val = poly[idx];
    while (val >= q) val -= q;
    poly[idx] = val;
}

// Center reduction: coefficients to [-q/2, q/2)
__global__ void poly_center_reduce_kernel(
    const uint64_t* __restrict__ a,
    int64_t* __restrict__ c,
    uint32_t n,
    uint64_t q
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    uint64_t val = a[idx];
    int64_t half_q = (int64_t)(q >> 1);
    
    if ((int64_t)val > half_q) {
        c[idx] = (int64_t)val - (int64_t)q;
    } else {
        c[idx] = (int64_t)val;
    }
}

// Convert to Montgomery form
__global__ void poly_to_mont_kernel(
    const uint64_t* __restrict__ a,
    uint64_t* __restrict__ c,
    uint32_t n,
    uint64_t q,
    uint64_t q_inv,
    uint64_t r2
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    c[idx] = mont_mul(a[idx], r2, q, q_inv);
}

// Convert from Montgomery form
__global__ void poly_from_mont_kernel(
    const uint64_t* __restrict__ a,
    uint64_t* __restrict__ c,
    uint32_t n,
    uint64_t q,
    uint64_t q_inv
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Multiply by 1 (in non-Montgomery form)
    uint64_t lo = a[idx];
    uint64_t hi = 0;
    c[idx] = mont_reduce(lo, hi, q, q_inv);
}

// ============================================================================
// Vector Operations for Polynomial Vectors
// ============================================================================

// Inner product of polynomial vectors (result is a single polynomial)
__global__ void poly_inner_product_kernel(
    const uint64_t* __restrict__ a,         // [k][n]
    const uint64_t* __restrict__ b,         // [k][n]
    uint64_t* __restrict__ c,               // [n]
    uint32_t k,
    uint32_t n,
    uint64_t q,
    uint64_t q_inv
) {
    extern __shared__ uint64_t shmem[];
    
    const uint32_t coeff_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    
    if (coeff_idx >= n) return;
    
    // Each thread computes partial sum for a range of k
    uint64_t local_sum = 0;
    for (uint32_t i = tid; i < k; i += blockDim.x) {
        uint64_t prod = mont_mul(a[i * n + coeff_idx], b[i * n + coeff_idx], q, q_inv);
        local_sum = mod_add(local_sum, prod, q);
    }
    
    shmem[tid] = local_sum;
    __syncthreads();
    
    // Reduction
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shmem[tid] = mod_add(shmem[tid], shmem[tid + s], q);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        c[coeff_idx] = shmem[0];
    }
}

// Vector-scalar multiplication: c[i] = scalar * a[i] for each polynomial
__global__ void poly_vec_scalar_mul_kernel(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ scalars,   // [k] scalars (one per polynomial)
    uint64_t* __restrict__ c,
    uint32_t k,
    uint32_t n,
    uint64_t q,
    uint64_t q_inv
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = k * n;
    if (idx >= total) return;
    
    uint32_t poly_idx = idx / n;
    uint64_t scalar = scalars[poly_idx];
    c[idx] = mont_mul(a[idx], scalar, q, q_inv);
}

// ============================================================================
// Automorphism Operations (for RLWE key switching)
// ============================================================================

// Apply automorphism sigma_k: a(X) -> a(X^k)
__global__ void poly_automorphism_kernel(
    const uint64_t* __restrict__ a,
    uint64_t* __restrict__ c,
    uint32_t n,
    uint32_t k,                             // Automorphism index (odd)
    uint64_t q
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // For negacyclic: c[i] = a[(i*k) mod 2n] with appropriate sign
    uint32_t src_idx = ((uint64_t)idx * k) % (2 * n);
    bool negate = src_idx >= n;
    if (negate) src_idx -= n;
    
    uint64_t val = a[src_idx];
    c[idx] = negate ? mod_neg(val, q) : val;
}

// Inverse automorphism
__global__ void poly_automorphism_inv_kernel(
    const uint64_t* __restrict__ a,
    uint64_t* __restrict__ c,
    uint32_t n,
    uint32_t k_inv,                         // Inverse automorphism index
    uint64_t q
) {
    poly_automorphism_kernel<<<gridDim, blockDim>>>(a, c, n, k_inv, q);
}

// ============================================================================
// Rounding and Scaling Operations
// ============================================================================

// Scale and round: c[i] = round(a[i] * t / q) for rescaling
__global__ void poly_scale_round_kernel(
    const uint64_t* __restrict__ a,
    uint64_t* __restrict__ c,
    uint32_t n,
    uint64_t q,
    uint64_t t                              // Target modulus
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Compute (a[i] * t + q/2) / q using 128-bit arithmetic
    unsigned __int128 prod = (unsigned __int128)a[idx] * t;
    prod += q >> 1;  // Round
    uint64_t result = (uint64_t)(prod / q);
    
    c[idx] = result % t;
}

// Modulus switch: q1 -> q2
__global__ void poly_mod_switch_kernel(
    const uint64_t* __restrict__ a,
    uint64_t* __restrict__ c,
    uint32_t n,
    uint64_t q1,
    uint64_t q2
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    unsigned __int128 prod = (unsigned __int128)a[idx] * q2;
    prod += q1 >> 1;
    c[idx] = (uint64_t)(prod / q1);
}

// ============================================================================
// Host API
// ============================================================================

void poly_add(
    const uint64_t* a, const uint64_t* b, uint64_t* c,
    uint32_t n, uint64_t q, cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    poly_add_kernel<<<grid, block, 0, stream>>>(a, b, c, n, q);
}

void poly_sub(
    const uint64_t* a, const uint64_t* b, uint64_t* c,
    uint32_t n, uint64_t q, cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    poly_sub_kernel<<<grid, block, 0, stream>>>(a, b, c, n, q);
}

void poly_neg(
    const uint64_t* a, uint64_t* c,
    uint32_t n, uint64_t q, cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    poly_neg_kernel<<<grid, block, 0, stream>>>(a, c, n, q);
}

void poly_scalar_mul(
    const uint64_t* a, uint64_t scalar, uint64_t* c,
    uint32_t n, uint64_t q, uint64_t q_inv, cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    poly_scalar_mul_kernel<<<grid, block, 0, stream>>>(a, scalar, c, n, q, q_inv);
}

void poly_mul_ntt(
    const uint64_t* a, const uint64_t* b, uint64_t* c,
    uint32_t n, uint64_t q, uint64_t q_inv, cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    poly_mul_ntt_kernel<<<grid, block, 0, stream>>>(a, b, c, n, q, q_inv);
}

void poly_mul_add_ntt(
    const uint64_t* a, const uint64_t* b, uint64_t* c,
    uint32_t n, uint64_t q, uint64_t q_inv, cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    poly_mul_add_ntt_kernel<<<grid, block, 0, stream>>>(a, b, c, n, q, q_inv);
}

void poly_batch_add(
    const uint64_t* a, const uint64_t* b, uint64_t* c,
    uint32_t n, uint32_t batch_size, uint64_t q, cudaStream_t stream
) {
    uint32_t total = n * batch_size;
    dim3 block(256);
    dim3 grid((total + 255) / 256);
    poly_batch_add_kernel<<<grid, block, 0, stream>>>(a, b, c, n, batch_size, q);
}

void poly_batch_mul_ntt(
    const uint64_t* a, const uint64_t* b, uint64_t* c,
    uint32_t n, uint32_t batch_size, uint64_t q, uint64_t q_inv, cudaStream_t stream
) {
    uint32_t total = n * batch_size;
    dim3 block(256);
    dim3 grid((total + 255) / 256);
    poly_batch_mul_ntt_kernel<<<grid, block, 0, stream>>>(a, b, c, n, batch_size, q, q_inv);
}

void poly_matrix_vec_mul(
    const uint64_t* A, const uint64_t* v, uint64_t* c,
    uint32_t rows, uint32_t cols, uint32_t n,
    uint64_t q, uint64_t q_inv, cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(rows);
    size_t shmem = n * sizeof(uint64_t);
    poly_matrix_vec_mul_kernel<<<grid, block, shmem, stream>>>(A, v, c, rows, cols, n, q, q_inv);
}

void poly_inner_product(
    const uint64_t* a, const uint64_t* b, uint64_t* c,
    uint32_t k, uint32_t n,
    uint64_t q, uint64_t q_inv, cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(n);
    size_t shmem = 256 * sizeof(uint64_t);
    poly_inner_product_kernel<<<grid, block, shmem, stream>>>(a, b, c, k, n, q, q_inv);
}

void poly_to_mont(
    const uint64_t* a, uint64_t* c,
    uint32_t n, uint64_t q, uint64_t q_inv, uint64_t r2, cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    poly_to_mont_kernel<<<grid, block, 0, stream>>>(a, c, n, q, q_inv, r2);
}

void poly_from_mont(
    const uint64_t* a, uint64_t* c,
    uint32_t n, uint64_t q, uint64_t q_inv, cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    poly_from_mont_kernel<<<grid, block, 0, stream>>>(a, c, n, q, q_inv);
}

void poly_automorphism(
    const uint64_t* a, uint64_t* c,
    uint32_t n, uint32_t k, uint64_t q, cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    poly_automorphism_kernel<<<grid, block, 0, stream>>>(a, c, n, k, q);
}

} // namespace poly
} // namespace cuda
} // namespace lux
