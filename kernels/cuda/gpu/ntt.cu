// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Number Theoretic Transform (NTT) CUDA Kernels
// Implements NTT for lattice-based cryptography (Kyber, Dilithium)
// Operations in Z_q where q is a prime with 2^k | (q-1) for roots of unity
//
// Common primes:
// - Kyber/Dilithium: q = 3329 (Kyber), q = 8380417 (Dilithium)
// - General: q = 12289 (NewHope), q = 7681

#include <cuda_runtime.h>
#include <cstdint>

// ============================================================================
// Configuration
// ============================================================================

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define MAX_NTT_SIZE 4096

// Default modulus (Dilithium: q = 8380417 = 2^23 - 2^13 + 1)
// This gives primitive 2^23-th root of unity
#define DILITHIUM_Q 8380417
#define DILITHIUM_Q_INV 58728449  // Montgomery constant

// Kyber modulus: q = 3329
#define KYBER_Q 3329
#define KYBER_MONT 2285  // 2^16 mod q
#define KYBER_QINV (-3327)  // q^(-1) mod 2^16

// ============================================================================
// Modular Arithmetic
// ============================================================================

// Barrett reduction: compute a mod q where a < q^2
__device__ __forceinline__
int32_t barrett_reduce(int64_t a, int32_t q, int64_t barrett_const) {
    int64_t t = (a * barrett_const) >> 32;
    int32_t r = (int32_t)(a - t * q);
    return r >= q ? r - q : (r < 0 ? r + q : r);
}

// Montgomery reduction: compute a * R^(-1) mod q
__device__ __forceinline__
int32_t montgomery_reduce(int64_t a, int32_t q, int32_t q_inv) {
    int32_t t = (int32_t)a * q_inv;
    int64_t m = (int64_t)t * q;
    int32_t r = (int32_t)((a - m) >> 32);
    return r < 0 ? r + q : r;
}

// Simple modular reduction for small values
__device__ __forceinline__
int32_t mod_reduce(int32_t a, int32_t q) {
    int32_t r = a % q;
    return r < 0 ? r + q : r;
}

// Modular addition
__device__ __forceinline__
int32_t mod_add(int32_t a, int32_t b, int32_t q) {
    int32_t r = a + b;
    return r >= q ? r - q : r;
}

// Modular subtraction
__device__ __forceinline__
int32_t mod_sub(int32_t a, int32_t b, int32_t q) {
    int32_t r = a - b;
    return r < 0 ? r + q : r;
}

// Modular multiplication with Montgomery
__device__ __forceinline__
int32_t mod_mul_mont(int32_t a, int32_t b, int32_t q, int32_t q_inv) {
    return montgomery_reduce((int64_t)a * b, q, q_inv);
}

// ============================================================================
// NTT Butterfly (Cooley-Tukey for forward, Gentleman-Sande for inverse)
// ============================================================================

// Forward NTT butterfly: Cooley-Tukey decimation-in-time
// a' = a + w*b
// b' = a - w*b
__device__ __forceinline__
void ntt_butterfly_ct(int32_t& a, int32_t& b, int32_t w, int32_t q, int32_t q_inv) {
    int32_t t = mod_mul_mont(w, b, q, q_inv);
    b = mod_sub(a, t, q);
    a = mod_add(a, t, q);
}

// Inverse NTT butterfly: Gentleman-Sande decimation-in-frequency
// a' = a + b
// b' = w * (a - b)
__device__ __forceinline__
void ntt_butterfly_gs(int32_t& a, int32_t& b, int32_t w, int32_t q, int32_t q_inv) {
    int32_t t = a;
    a = mod_add(a, b, q);
    b = mod_mul_mont(w, mod_sub(t, b, q), q, q_inv);
}

// ============================================================================
// Bit Reversal Permutation
// ============================================================================

__device__ __forceinline__
uint32_t bit_reverse_ntt(uint32_t x, int log2n) {
    uint32_t result = 0;
    for (int i = 0; i < log2n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

// ============================================================================
// Forward NTT Kernel
// Input in normal order, output in bit-reversed order
// ============================================================================

template <int N>
__global__ void lux_ntt_forward_fused_kernel(
    int32_t* __restrict__ output,
    const int32_t* __restrict__ input,
    const int32_t* __restrict__ twiddles,  // Precomputed twiddle factors
    int32_t q,
    int32_t q_inv,
    int batch
) {
    __shared__ int32_t smem[N];

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch) return;

    input += batch_idx * N;
    output += batch_idx * N;

    // Compute log2(N)
    int log2n = 0;
    for (int t = N; t > 1; t >>= 1) log2n++;

    // Load input to shared memory
    for (int i = tid; i < N; i += blockDim.x) {
        smem[i] = input[i];
    }
    __syncthreads();

    // NTT stages (Cooley-Tukey)
    int twiddle_idx = 0;
    for (int len = N / 2; len >= 1; len >>= 1) {
        for (int i = tid; i < N / 2; i += blockDim.x) {
            int group = i / len;
            int idx = i % len;
            int j = group * 2 * len + idx;

            int32_t w = twiddles[twiddle_idx + group];
            ntt_butterfly_ct(smem[j], smem[j + len], w, q, q_inv);
        }
        __syncthreads();
        twiddle_idx += N / (2 * len);
    }

    // Write output (bit-reversed order)
    for (int i = tid; i < N; i += blockDim.x) {
        int j = bit_reverse_ntt(i, log2n);
        output[j] = smem[i];
    }
}

// ============================================================================
// Inverse NTT Kernel
// Input in bit-reversed order, output in normal order
// ============================================================================

template <int N>
__global__ void lux_ntt_inverse_fused_kernel(
    int32_t* __restrict__ output,
    const int32_t* __restrict__ input,
    const int32_t* __restrict__ inv_twiddles,  // Inverse twiddle factors
    int32_t n_inv,      // N^(-1) mod q
    int32_t q,
    int32_t q_inv,
    int batch
) {
    __shared__ int32_t smem[N];

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch) return;

    input += batch_idx * N;
    output += batch_idx * N;

    int log2n = 0;
    for (int t = N; t > 1; t >>= 1) log2n++;

    // Load input (bit-reversed order)
    for (int i = tid; i < N; i += blockDim.x) {
        int j = bit_reverse_ntt(i, log2n);
        smem[j] = input[i];
    }
    __syncthreads();

    // Inverse NTT stages (Gentleman-Sande)
    int twiddle_idx = N - 2;
    for (int len = 1; len < N; len <<= 1) {
        for (int i = tid; i < N / 2; i += blockDim.x) {
            int group = i / len;
            int idx = i % len;
            int j = group * 2 * len + idx;

            int32_t w = inv_twiddles[twiddle_idx - (N / (2 * len) - 1) + group];
            ntt_butterfly_gs(smem[j], smem[j + len], w, q, q_inv);
        }
        __syncthreads();
        twiddle_idx -= N / (2 * len);
    }

    // Scale by N^(-1) and write output
    for (int i = tid; i < N; i += blockDim.x) {
        output[i] = mod_mul_mont(smem[i], n_inv, q, q_inv);
    }
}

// ============================================================================
// Staged NTT for large sizes (global memory between stages)
// ============================================================================

__global__ void lux_ntt_forward_stage_kernel(
    int32_t* __restrict__ data,
    const int32_t* __restrict__ twiddles,
    int32_t q,
    int32_t q_inv,
    int n,
    int len,          // Current butterfly size
    int twiddle_offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_butterflies = n / 2;

    if (idx >= total_butterflies) return;

    int group = idx / len;
    int j = idx % len;
    int i0 = group * 2 * len + j;
    int i1 = i0 + len;

    int32_t w = twiddles[twiddle_offset + group];

    int32_t a = data[i0];
    int32_t b = data[i1];

    ntt_butterfly_ct(a, b, w, q, q_inv);

    data[i0] = a;
    data[i1] = b;
}

__global__ void lux_ntt_inverse_stage_kernel(
    int32_t* __restrict__ data,
    const int32_t* __restrict__ inv_twiddles,
    int32_t q,
    int32_t q_inv,
    int n,
    int len,
    int twiddle_offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_butterflies = n / 2;

    if (idx >= total_butterflies) return;

    int group = idx / len;
    int j = idx % len;
    int i0 = group * 2 * len + j;
    int i1 = i0 + len;

    int32_t w = inv_twiddles[twiddle_offset + group];

    int32_t a = data[i0];
    int32_t b = data[i1];

    ntt_butterfly_gs(a, b, w, q, q_inv);

    data[i0] = a;
    data[i1] = b;
}

// ============================================================================
// NTT Pointwise Multiplication
// Component-wise multiplication in NTT domain: c = a * b
// ============================================================================

__global__ void lux_ntt_pointwise_mul_kernel(
    int32_t* __restrict__ output,
    const int32_t* __restrict__ a,
    const int32_t* __restrict__ b,
    int32_t q,
    int32_t q_inv,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    output[idx] = mod_mul_mont(a[idx], b[idx], q, q_inv);
}

// ============================================================================
// NTT Scaling (multiply all elements by a constant)
// ============================================================================

__global__ void lux_ntt_scale_kernel(
    int32_t* __restrict__ data,
    int32_t scale,
    int32_t q,
    int32_t q_inv,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    data[idx] = mod_mul_mont(data[idx], scale, q, q_inv);
}

// ============================================================================
// Bit Reversal Permutation Kernel
// ============================================================================

__global__ void lux_ntt_bit_reverse_kernel(
    int32_t* __restrict__ output,
    const int32_t* __restrict__ input,
    int n,
    int log2n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int rev = bit_reverse_ntt(idx, log2n);
    output[rev] = input[idx];
}

// ============================================================================
// NTT Polynomial Addition/Subtraction
// ============================================================================

__global__ void lux_ntt_add_kernel(
    int32_t* __restrict__ output,
    const int32_t* __restrict__ a,
    const int32_t* __restrict__ b,
    int32_t q,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    output[idx] = mod_add(a[idx], b[idx], q);
}

__global__ void lux_ntt_sub_kernel(
    int32_t* __restrict__ output,
    const int32_t* __restrict__ a,
    const int32_t* __restrict__ b,
    int32_t q,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    output[idx] = mod_sub(a[idx], b[idx], q);
}

// ============================================================================
// Kyber-specific NTT (q = 3329, n = 256)
// Uses Cooley-Tukey with 7 layers
// ============================================================================

__global__ void lux_ntt_kyber_forward_kernel(
    int16_t* __restrict__ output,
    const int16_t* __restrict__ input,
    const int16_t* __restrict__ zetas  // Precomputed twiddle factors
) {
    __shared__ int16_t smem[256];

    int poly_idx = blockIdx.x;
    int tid = threadIdx.x;

    input += poly_idx * 256;
    output += poly_idx * 256;

    // Load
    if (tid < 256) {
        smem[tid] = input[tid];
    }
    __syncthreads();

    // 7 layers of NTT
    int k = 1;
    for (int len = 128; len >= 2; len >>= 1) {
        for (int start = tid; start < 256; start += blockDim.x) {
            int group = start / (2 * len);
            int idx = start % (2 * len);
            if (idx < len) {
                int j = group * 2 * len + idx;
                int16_t zeta = zetas[k + group];

                int16_t t = (int16_t)(((int32_t)zeta * smem[j + len]) % KYBER_Q);
                smem[j + len] = (smem[j] - t + KYBER_Q) % KYBER_Q;
                smem[j] = (smem[j] + t) % KYBER_Q;
            }
        }
        __syncthreads();
        k += 256 / (2 * len);
    }

    // Store
    if (tid < 256) {
        output[tid] = smem[tid];
    }
}

// ============================================================================
// Dilithium-specific NTT (q = 8380417, n = 256)
// ============================================================================

__global__ void lux_ntt_dilithium_forward_kernel(
    int32_t* __restrict__ output,
    const int32_t* __restrict__ input,
    const int32_t* __restrict__ zetas
) {
    __shared__ int32_t smem[256];

    int poly_idx = blockIdx.x;
    int tid = threadIdx.x;

    input += poly_idx * 256;
    output += poly_idx * 256;

    if (tid < 256) {
        smem[tid] = input[tid];
    }
    __syncthreads();

    int k = 0;
    for (int len = 128; len >= 1; len >>= 1) {
        for (int start = tid; start < 256; start += blockDim.x) {
            int group = start / (2 * len);
            int idx = start % (2 * len);
            if (idx < len) {
                int j = group * 2 * len + idx;
                int32_t zeta = zetas[k + group];

                int64_t t = ((int64_t)zeta * smem[j + len]) % DILITHIUM_Q;
                smem[j + len] = mod_sub(smem[j], (int32_t)t, DILITHIUM_Q);
                smem[j] = mod_add(smem[j], (int32_t)t, DILITHIUM_Q);
            }
        }
        __syncthreads();
        k += 256 / (2 * len);
    }

    if (tid < 256) {
        output[tid] = smem[tid];
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_ntt_forward_fused(
    void* output,
    const void* input,
    const void* twiddles,
    int32_t q,
    int32_t q_inv,
    int n,
    int batch,
    cudaStream_t stream
) {
    int threads = min(256, n / 2);

    if (n == 256) {
        lux_ntt_forward_fused_kernel<256><<<batch, threads, 0, stream>>>(
            (int32_t*)output, (const int32_t*)input, (const int32_t*)twiddles,
            q, q_inv, batch
        );
    } else if (n == 512) {
        lux_ntt_forward_fused_kernel<512><<<batch, threads, 0, stream>>>(
            (int32_t*)output, (const int32_t*)input, (const int32_t*)twiddles,
            q, q_inv, batch
        );
    } else if (n == 1024) {
        lux_ntt_forward_fused_kernel<1024><<<batch, threads, 0, stream>>>(
            (int32_t*)output, (const int32_t*)input, (const int32_t*)twiddles,
            q, q_inv, batch
        );
    } else if (n == 2048) {
        lux_ntt_forward_fused_kernel<2048><<<batch, threads, 0, stream>>>(
            (int32_t*)output, (const int32_t*)input, (const int32_t*)twiddles,
            q, q_inv, batch
        );
    } else if (n == 4096) {
        lux_ntt_forward_fused_kernel<4096><<<batch, threads, 0, stream>>>(
            (int32_t*)output, (const int32_t*)input, (const int32_t*)twiddles,
            q, q_inv, batch
        );
    } else {
        return -1;  // Unsupported size
    }

    return cudaGetLastError();
}

int lux_cuda_ntt_inverse_fused(
    void* output,
    const void* input,
    const void* inv_twiddles,
    int32_t n_inv,
    int32_t q,
    int32_t q_inv,
    int n,
    int batch,
    cudaStream_t stream
) {
    int threads = min(256, n / 2);

    if (n == 256) {
        lux_ntt_inverse_fused_kernel<256><<<batch, threads, 0, stream>>>(
            (int32_t*)output, (const int32_t*)input, (const int32_t*)inv_twiddles,
            n_inv, q, q_inv, batch
        );
    } else if (n == 512) {
        lux_ntt_inverse_fused_kernel<512><<<batch, threads, 0, stream>>>(
            (int32_t*)output, (const int32_t*)input, (const int32_t*)inv_twiddles,
            n_inv, q, q_inv, batch
        );
    } else if (n == 1024) {
        lux_ntt_inverse_fused_kernel<1024><<<batch, threads, 0, stream>>>(
            (int32_t*)output, (const int32_t*)input, (const int32_t*)inv_twiddles,
            n_inv, q, q_inv, batch
        );
    } else if (n == 2048) {
        lux_ntt_inverse_fused_kernel<2048><<<batch, threads, 0, stream>>>(
            (int32_t*)output, (const int32_t*)input, (const int32_t*)inv_twiddles,
            n_inv, q, q_inv, batch
        );
    } else if (n == 4096) {
        lux_ntt_inverse_fused_kernel<4096><<<batch, threads, 0, stream>>>(
            (int32_t*)output, (const int32_t*)input, (const int32_t*)inv_twiddles,
            n_inv, q, q_inv, batch
        );
    } else {
        return -1;
    }

    return cudaGetLastError();
}

int lux_cuda_ntt_forward_stage(
    void* data,
    const void* twiddles,
    int32_t q,
    int32_t q_inv,
    int n,
    int len,
    int twiddle_offset,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n / 2 + threads - 1) / threads;

    lux_ntt_forward_stage_kernel<<<blocks, threads, 0, stream>>>(
        (int32_t*)data, (const int32_t*)twiddles, q, q_inv, n, len, twiddle_offset
    );

    return cudaGetLastError();
}

int lux_cuda_ntt_inverse_stage(
    void* data,
    const void* inv_twiddles,
    int32_t q,
    int32_t q_inv,
    int n,
    int len,
    int twiddle_offset,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n / 2 + threads - 1) / threads;

    lux_ntt_inverse_stage_kernel<<<blocks, threads, 0, stream>>>(
        (int32_t*)data, (const int32_t*)inv_twiddles, q, q_inv, n, len, twiddle_offset
    );

    return cudaGetLastError();
}

int lux_cuda_ntt_pointwise_mul(
    void* output,
    const void* a,
    const void* b,
    int32_t q,
    int32_t q_inv,
    int n,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    lux_ntt_pointwise_mul_kernel<<<blocks, threads, 0, stream>>>(
        (int32_t*)output, (const int32_t*)a, (const int32_t*)b, q, q_inv, n
    );

    return cudaGetLastError();
}

int lux_cuda_ntt_scale(
    void* data,
    int32_t scale,
    int32_t q,
    int32_t q_inv,
    int n,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    lux_ntt_scale_kernel<<<blocks, threads, 0, stream>>>(
        (int32_t*)data, scale, q, q_inv, n
    );

    return cudaGetLastError();
}

int lux_cuda_ntt_bit_reverse(
    void* output,
    const void* input,
    int n,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    int log2n = 0;
    for (int t = n; t > 1; t >>= 1) log2n++;

    lux_ntt_bit_reverse_kernel<<<blocks, threads, 0, stream>>>(
        (int32_t*)output, (const int32_t*)input, n, log2n
    );

    return cudaGetLastError();
}

int lux_cuda_ntt_add(
    void* output,
    const void* a,
    const void* b,
    int32_t q,
    int n,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    lux_ntt_add_kernel<<<blocks, threads, 0, stream>>>(
        (int32_t*)output, (const int32_t*)a, (const int32_t*)b, q, n
    );

    return cudaGetLastError();
}

int lux_cuda_ntt_sub(
    void* output,
    const void* a,
    const void* b,
    int32_t q,
    int n,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    lux_ntt_sub_kernel<<<blocks, threads, 0, stream>>>(
        (int32_t*)output, (const int32_t*)a, (const int32_t*)b, q, n
    );

    return cudaGetLastError();
}

int lux_cuda_ntt_kyber_forward(
    void* output,
    const void* input,
    const void* zetas,
    int batch,
    cudaStream_t stream
) {
    lux_ntt_kyber_forward_kernel<<<batch, 128, 0, stream>>>(
        (int16_t*)output, (const int16_t*)input, (const int16_t*)zetas
    );

    return cudaGetLastError();
}

int lux_cuda_ntt_dilithium_forward(
    void* output,
    const void* input,
    const void* zetas,
    int batch,
    cudaStream_t stream
) {
    lux_ntt_dilithium_forward_kernel<<<batch, 128, 0, stream>>>(
        (int32_t*)output, (const int32_t*)input, (const int32_t*)zetas
    );

    return cudaGetLastError();
}

}  // extern "C"
