// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Twiddle Factor Cache CUDA Kernels for FHE NTT
// Provides efficient caching and generation of NTT twiddle factors.
//
// Features:
// - Precomputation of twiddle factors for multiple ring sizes
// - Barrett precomputation for fast modular multiplication
// - Bit-reversed ordering support
// - L1/L2 cache-optimized access patterns
// - Multi-modulus support for RNS

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================================
// Constants
// ============================================================================

#define MAX_LOG_N 16           // Max N = 65536
#define MAX_N (1 << MAX_LOG_N)
#define WARP_SIZE 32

// ============================================================================
// Modular Arithmetic for Twiddle Generation
// ============================================================================

// Extended Euclidean algorithm for modular inverse
__device__ __host__
uint64_t mod_inverse(uint64_t a, uint64_t m) {
    if (m == 1) return 0;

    int64_t m0 = m, x0 = 0, x1 = 1;
    int64_t a_signed = a;

    while (a_signed > 1) {
        int64_t q = a_signed / m0;
        int64_t t = m0;
        m0 = a_signed % m0;
        a_signed = t;
        t = x0;
        x0 = x1 - q * x0;
        x1 = t;
    }

    if (x1 < 0) x1 += m;
    return (uint64_t)x1;
}

// Modular exponentiation
__device__ __host__
uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    base = base % mod;

    while (exp > 0) {
        if (exp & 1) {
            __uint128_t tmp = (__uint128_t)result * base;
            result = (uint64_t)(tmp % mod);
        }
        exp >>= 1;
        __uint128_t tmp = (__uint128_t)base * base;
        base = (uint64_t)(tmp % mod);
    }
    return result;
}

// Compute Barrett precomputation constant
// precon = floor(2^64 * omega / Q)
__device__ __host__
uint64_t compute_barrett_precon(uint64_t omega, uint64_t Q) {
    // We compute floor(2^64 * omega / Q) using 128-bit arithmetic
    __uint128_t numerator = ((__uint128_t)omega) << 64;
    return (uint64_t)(numerator / Q);
}

// ============================================================================
// Twiddle Factor Generation Parameters
// ============================================================================

struct TwiddleCacheParams {
    uint64_t Q;              // Prime modulus
    uint64_t omega;          // Primitive N-th root of unity
    uint64_t omega_inv;      // Inverse of omega
    uint64_t N_inv;          // N^{-1} mod Q
    uint32_t N;              // Ring dimension
    uint32_t log_N;          // log2(N)
};

// ============================================================================
// Twiddle Factor Generation Kernels
// ============================================================================

// Generate forward twiddle factors: omega^i for i = 0..N-1
extern "C" __global__
void generate_twiddles_kernel(
    uint64_t* __restrict__ twiddles,
    uint64_t* __restrict__ precons,
    uint64_t Q,
    uint64_t omega,
    uint32_t N
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        uint64_t tw = mod_pow(omega, tid, Q);
        twiddles[tid] = tw;
        precons[tid] = compute_barrett_precon(tw, Q);
    }
}

// Generate inverse twiddle factors: omega_inv^i for i = 0..N-1
extern "C" __global__
void generate_inv_twiddles_kernel(
    uint64_t* __restrict__ inv_twiddles,
    uint64_t* __restrict__ inv_precons,
    uint64_t Q,
    uint64_t omega_inv,
    uint32_t N
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        uint64_t tw = mod_pow(omega_inv, tid, Q);
        inv_twiddles[tid] = tw;
        inv_precons[tid] = compute_barrett_precon(tw, Q);
    }
}

// Generate twiddles in OpenFHE layout (stage-wise organization)
// twiddles[m + j] = omega^(bit_reverse(j, log_m)) for stage m
extern "C" __global__
void generate_twiddles_staged_kernel(
    uint64_t* __restrict__ twiddles,
    uint64_t* __restrict__ precons,
    uint64_t Q,
    uint64_t omega,
    uint32_t N,
    uint32_t log_N
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= N) return;

    // For OpenFHE layout: twiddles[m + j] where m = 2^stage, j < m
    // This gives stage-optimized access patterns

    // Determine stage and position within stage
    uint32_t m = 1;
    uint32_t pos = tid;

    // Find which "group" this index belongs to
    for (uint32_t s = 0; s < log_N; ++s) {
        m = 1 << s;
        if (tid < 2 * m) break;
    }

    if (tid < 2) {
        // Special case: indices 0 and 1
        twiddles[tid] = (tid == 0) ? 1 : omega;
        precons[tid] = compute_barrett_precon(twiddles[tid], Q);
        return;
    }

    // General case: compute exponent based on bit-reversal
    uint32_t exp = 0;
    uint32_t idx = tid - m;  // Position within this stage's twiddles

    // The exponent for twiddle at index m+j is (N/2m) * bit_reverse(j, log_m)
    uint32_t log_m = 31 - __clz(m);  // log2(m)
    uint32_t j = tid - m;

    // Bit reverse j with log_m bits
    uint32_t rev_j = 0;
    uint32_t tmp = j;
    for (uint32_t b = 0; b < log_m; ++b) {
        rev_j = (rev_j << 1) | (tmp & 1);
        tmp >>= 1;
    }

    exp = (N / (2 * m)) * rev_j;

    uint64_t tw = mod_pow(omega, exp, Q);
    twiddles[tid] = tw;
    precons[tid] = compute_barrett_precon(tw, Q);
}

// Generate twiddles with bit-reversed ordering
extern "C" __global__
void generate_twiddles_bitrev_kernel(
    uint64_t* __restrict__ twiddles,
    uint64_t* __restrict__ precons,
    uint64_t Q,
    uint64_t omega,
    uint32_t N,
    uint32_t log_N
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= N) return;

    // Bit-reverse the index
    uint32_t rev = 0;
    uint32_t tmp = tid;
    for (uint32_t i = 0; i < log_N; ++i) {
        rev = (rev << 1) | (tmp & 1);
        tmp >>= 1;
    }

    uint64_t tw = mod_pow(omega, rev, Q);
    twiddles[tid] = tw;
    precons[tid] = compute_barrett_precon(tw, Q);
}

// ============================================================================
// Multi-Modulus Twiddle Generation (for RNS)
// ============================================================================

// Generate twiddles for multiple moduli simultaneously
extern "C" __global__
void generate_twiddles_rns_kernel(
    uint64_t* __restrict__ twiddles,        // [num_moduli, N]
    uint64_t* __restrict__ precons,         // [num_moduli, N]
    const uint64_t* __restrict__ Q_array,   // [num_moduli]
    const uint64_t* __restrict__ omega_array, // [num_moduli]
    uint32_t N,
    uint32_t num_moduli
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = N * num_moduli;

    if (tid >= total) return;

    uint32_t mod_idx = tid / N;
    uint32_t coeff_idx = tid % N;

    uint64_t Q = Q_array[mod_idx];
    uint64_t omega = omega_array[mod_idx];

    uint64_t tw = mod_pow(omega, coeff_idx, Q);
    twiddles[tid] = tw;
    precons[tid] = compute_barrett_precon(tw, Q);
}

// ============================================================================
// Twiddle Cache Management
// ============================================================================

// Copy twiddles to constant memory (for frequently used sizes)
// This is typically called from host code after generation

// Prefetch twiddles to L2 cache
extern "C" __global__
void prefetch_twiddles_kernel(
    const uint64_t* __restrict__ twiddles,
    const uint64_t* __restrict__ precons,
    uint32_t N
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        // Force load into L2 cache
        uint64_t t = __ldg(&twiddles[tid]);
        uint64_t p = __ldg(&precons[tid]);

        // Prevent compiler from optimizing away the loads
        if (t == 0 && p == 0) {
            // Never executed, but compiler doesn't know
            twiddles[0] = 1;
        }
    }
}

// ============================================================================
// Diagonal Twiddle Generation (for Four-Step NTT)
// ============================================================================

// Generate diagonal twiddles for four-step NTT: omega^(i*j) for 0 <= i < n1, 0 <= j < n2
extern "C" __global__
void generate_diagonal_twiddles_kernel(
    uint64_t* __restrict__ diag_twiddles,   // [n1 * n2]
    uint64_t* __restrict__ diag_precons,    // [n1 * n2]
    uint64_t Q,
    uint64_t omega,
    uint32_t n1,
    uint32_t n2
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = n1 * n2;

    if (tid >= total) return;

    uint32_t i = tid / n2;  // Row index
    uint32_t j = tid % n2;  // Column index

    // Diagonal twiddle = omega^(i * j)
    uint64_t exp = ((uint64_t)i * j) % (n1 * n2);
    uint64_t tw = mod_pow(omega, exp, Q);

    diag_twiddles[tid] = tw;
    diag_precons[tid] = compute_barrett_precon(tw, Q);
}

// ============================================================================
// Negacyclic Twiddle Generation
// ============================================================================

// Generate negacyclic twiddles: psi^i where psi is a 2N-th root of unity
// Used for negacyclic convolution: X^N = -1
extern "C" __global__
void generate_negacyclic_twiddles_kernel(
    uint64_t* __restrict__ psi_powers,      // [N]
    uint64_t* __restrict__ psi_inv_powers,  // [N]
    uint64_t* __restrict__ psi_precons,     // [N]
    uint64_t* __restrict__ psi_inv_precons, // [N]
    uint64_t Q,
    uint64_t psi,           // 2N-th root of unity
    uint64_t psi_inv,       // Inverse of psi
    uint32_t N
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= N) return;

    uint64_t pw = mod_pow(psi, tid, Q);
    uint64_t pw_inv = mod_pow(psi_inv, tid, Q);

    psi_powers[tid] = pw;
    psi_inv_powers[tid] = pw_inv;
    psi_precons[tid] = compute_barrett_precon(pw, Q);
    psi_inv_precons[tid] = compute_barrett_precon(pw_inv, Q);
}

// ============================================================================
// Apply Negacyclic Pre/Post Processing
// ============================================================================

// Pre-multiply by psi powers before standard NTT
extern "C" __global__
void apply_negacyclic_pre_kernel(
    uint64_t* __restrict__ data,
    const uint64_t* __restrict__ psi_powers,
    const uint64_t* __restrict__ psi_precons,
    uint64_t Q,
    uint32_t N,
    uint32_t batch_size
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = N * batch_size;

    if (tid >= total) return;

    uint32_t coeff_idx = tid % N;
    uint64_t psi = psi_powers[coeff_idx];
    uint64_t precon = psi_precons[coeff_idx];

    // Barrett multiplication
    uint64_t a = data[tid];
    uint64_t q_approx = __umul64hi(a, precon);
    uint64_t product = a * psi;
    uint64_t result = product - q_approx * Q;

    data[tid] = result >= Q ? result - Q : result;
}

// Post-multiply by psi_inv powers after standard INTT
extern "C" __global__
void apply_negacyclic_post_kernel(
    uint64_t* __restrict__ data,
    const uint64_t* __restrict__ psi_inv_powers,
    const uint64_t* __restrict__ psi_inv_precons,
    uint64_t Q,
    uint32_t N,
    uint32_t batch_size
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = N * batch_size;

    if (tid >= total) return;

    uint32_t coeff_idx = tid % N;
    uint64_t psi_inv = psi_inv_powers[coeff_idx];
    uint64_t precon = psi_inv_precons[coeff_idx];

    uint64_t a = data[tid];
    uint64_t q_approx = __umul64hi(a, precon);
    uint64_t product = a * psi_inv;
    uint64_t result = product - q_approx * Q;

    data[tid] = result >= Q ? result - Q : result;
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

cudaError_t lux_cuda_fhe_generate_twiddles(
    uint64_t* twiddles,
    uint64_t* precons,
    uint64_t Q,
    uint64_t omega,
    uint32_t N,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((N + 255) / 256);

    generate_twiddles_kernel<<<grid, block, 0, stream>>>(
        twiddles, precons, Q, omega, N
    );

    return cudaGetLastError();
}

cudaError_t lux_cuda_fhe_generate_inv_twiddles(
    uint64_t* inv_twiddles,
    uint64_t* inv_precons,
    uint64_t Q,
    uint64_t omega_inv,
    uint32_t N,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((N + 255) / 256);

    generate_inv_twiddles_kernel<<<grid, block, 0, stream>>>(
        inv_twiddles, inv_precons, Q, omega_inv, N
    );

    return cudaGetLastError();
}

cudaError_t lux_cuda_fhe_generate_twiddles_staged(
    uint64_t* twiddles,
    uint64_t* precons,
    uint64_t Q,
    uint64_t omega,
    uint32_t N,
    cudaStream_t stream
) {
    uint32_t log_N = 31 - __builtin_clz(N);

    dim3 block(256);
    dim3 grid((N + 255) / 256);

    generate_twiddles_staged_kernel<<<grid, block, 0, stream>>>(
        twiddles, precons, Q, omega, N, log_N
    );

    return cudaGetLastError();
}

cudaError_t lux_cuda_fhe_generate_twiddles_bitrev(
    uint64_t* twiddles,
    uint64_t* precons,
    uint64_t Q,
    uint64_t omega,
    uint32_t N,
    cudaStream_t stream
) {
    uint32_t log_N = 31 - __builtin_clz(N);

    dim3 block(256);
    dim3 grid((N + 255) / 256);

    generate_twiddles_bitrev_kernel<<<grid, block, 0, stream>>>(
        twiddles, precons, Q, omega, N, log_N
    );

    return cudaGetLastError();
}

cudaError_t lux_cuda_fhe_generate_twiddles_rns(
    uint64_t* twiddles,
    uint64_t* precons,
    const uint64_t* Q_array,
    const uint64_t* omega_array,
    uint32_t N,
    uint32_t num_moduli,
    cudaStream_t stream
) {
    uint32_t total = N * num_moduli;

    dim3 block(256);
    dim3 grid((total + 255) / 256);

    generate_twiddles_rns_kernel<<<grid, block, 0, stream>>>(
        twiddles, precons, Q_array, omega_array, N, num_moduli
    );

    return cudaGetLastError();
}

cudaError_t lux_cuda_fhe_generate_diagonal_twiddles(
    uint64_t* diag_twiddles,
    uint64_t* diag_precons,
    uint64_t Q,
    uint64_t omega,
    uint32_t n1,
    uint32_t n2,
    cudaStream_t stream
) {
    uint32_t total = n1 * n2;

    dim3 block(256);
    dim3 grid((total + 255) / 256);

    generate_diagonal_twiddles_kernel<<<grid, block, 0, stream>>>(
        diag_twiddles, diag_precons, Q, omega, n1, n2
    );

    return cudaGetLastError();
}

cudaError_t lux_cuda_fhe_generate_negacyclic_twiddles(
    uint64_t* psi_powers,
    uint64_t* psi_inv_powers,
    uint64_t* psi_precons,
    uint64_t* psi_inv_precons,
    uint64_t Q,
    uint64_t psi,
    uint64_t psi_inv,
    uint32_t N,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((N + 255) / 256);

    generate_negacyclic_twiddles_kernel<<<grid, block, 0, stream>>>(
        psi_powers, psi_inv_powers, psi_precons, psi_inv_precons,
        Q, psi, psi_inv, N
    );

    return cudaGetLastError();
}

cudaError_t lux_cuda_fhe_prefetch_twiddles(
    const uint64_t* twiddles,
    const uint64_t* precons,
    uint32_t N,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((N + 255) / 256);

    prefetch_twiddles_kernel<<<grid, block, 0, stream>>>(twiddles, precons, N);

    return cudaGetLastError();
}

cudaError_t lux_cuda_fhe_apply_negacyclic_pre(
    uint64_t* data,
    const uint64_t* psi_powers,
    const uint64_t* psi_precons,
    uint64_t Q,
    uint32_t N,
    uint32_t batch_size,
    cudaStream_t stream
) {
    uint32_t total = N * batch_size;

    dim3 block(256);
    dim3 grid((total + 255) / 256);

    apply_negacyclic_pre_kernel<<<grid, block, 0, stream>>>(
        data, psi_powers, psi_precons, Q, N, batch_size
    );

    return cudaGetLastError();
}

cudaError_t lux_cuda_fhe_apply_negacyclic_post(
    uint64_t* data,
    const uint64_t* psi_inv_powers,
    const uint64_t* psi_inv_precons,
    uint64_t Q,
    uint32_t N,
    uint32_t batch_size,
    cudaStream_t stream
) {
    uint32_t total = N * batch_size;

    dim3 block(256);
    dim3 grid((total + 255) / 256);

    apply_negacyclic_post_kernel<<<grid, block, 0, stream>>>(
        data, psi_inv_powers, psi_inv_precons, Q, N, batch_size
    );

    return cudaGetLastError();
}

}  // extern "C"
