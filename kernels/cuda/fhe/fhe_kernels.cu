// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// FHE Utility CUDA Kernels
// Common utility operations for FHE computations including:
// - Polynomial arithmetic
// - Modular arithmetic
// - Noise generation
// - Encoding/Decoding
// - Vector operations

#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// ============================================================================
// Constants
// ============================================================================

#define WARP_SIZE 32
#define MAX_N 4096

// ============================================================================
// Common FHE Parameters Structure
// ============================================================================

struct FHEParams {
    uint64_t Q;              // Ring modulus
    uint64_t N_inv;          // N^{-1} mod Q
    uint64_t N_inv_precon;   // Barrett precomputation for N_inv
    uint32_t N;              // Ring dimension (power of 2)
    uint32_t log_N;          // log2(N)
    uint32_t k;              // GLWE dimension
    uint32_t L;              // Decomposition levels
    uint32_t Bg_bits;        // Gadget base bits
};

// ============================================================================
// Modular Arithmetic Primitives
// ============================================================================

__device__ __forceinline__
uint64_t mod_add(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t sum = a + b;
    return sum >= Q ? sum - Q : sum;
}

__device__ __forceinline__
uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t Q) {
    return a >= b ? a - b : a + Q - b;
}

__device__ __forceinline__
uint64_t mod_neg(uint64_t a, uint64_t Q) {
    return a == 0 ? 0 : Q - a;
}

// Barrett reduction for modular multiplication
__device__ __forceinline__
uint64_t barrett_mul(uint64_t a, uint64_t b, uint64_t Q, uint64_t precon) {
    uint64_t q_approx = __umul64hi(a, precon);
    uint64_t product = a * b;
    uint64_t result = product - q_approx * Q;
    return result >= Q ? result - Q : result;
}

// Full modular multiplication without precomputation
__device__ __forceinline__
uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);

    if (hi == 0) return lo % Q;

    // 2^64 mod Q
    uint64_t r = (1ULL << 32) % Q;
    r = (r * r) % Q;

    uint64_t lo_mod = lo % Q;
    uint64_t hi_mod = (hi % Q) * r % Q;
    return mod_add(lo_mod, hi_mod, Q);
}

// ============================================================================
// Polynomial Addition/Subtraction
// ============================================================================

extern "C" __global__
void fhe_poly_add_kernel(
    uint64_t* __restrict__ result,
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t Q,
    uint32_t N,
    uint32_t batch_size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = N * batch_size;

    if (idx < total) {
        result[idx] = mod_add(a[idx], b[idx], Q);
    }
}

extern "C" __global__
void fhe_poly_sub_kernel(
    uint64_t* __restrict__ result,
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t Q,
    uint32_t N,
    uint32_t batch_size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = N * batch_size;

    if (idx < total) {
        result[idx] = mod_sub(a[idx], b[idx], Q);
    }
}

extern "C" __global__
void fhe_poly_neg_kernel(
    uint64_t* __restrict__ result,
    const uint64_t* __restrict__ a,
    uint64_t Q,
    uint32_t N,
    uint32_t batch_size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = N * batch_size;

    if (idx < total) {
        result[idx] = mod_neg(a[idx], Q);
    }
}

// ============================================================================
// Polynomial Scaling
// ============================================================================

extern "C" __global__
void fhe_poly_scale_kernel(
    uint64_t* __restrict__ result,
    const uint64_t* __restrict__ a,
    uint64_t scalar,
    uint64_t Q,
    uint32_t N,
    uint32_t batch_size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = N * batch_size;

    if (idx < total) {
        result[idx] = mod_mul(a[idx], scalar, Q);
    }
}

extern "C" __global__
void fhe_poly_scale_barrett_kernel(
    uint64_t* __restrict__ result,
    const uint64_t* __restrict__ a,
    uint64_t scalar,
    uint64_t scalar_precon,
    uint64_t Q,
    uint32_t N,
    uint32_t batch_size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = N * batch_size;

    if (idx < total) {
        result[idx] = barrett_mul(a[idx], scalar, Q, scalar_precon);
    }
}

// ============================================================================
// Pointwise Polynomial Multiplication (NTT Domain)
// ============================================================================

extern "C" __global__
void fhe_poly_pointwise_mul_kernel(
    uint64_t* __restrict__ result,
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t Q,
    uint32_t N,
    uint32_t batch_size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = N * batch_size;

    if (idx < total) {
        result[idx] = mod_mul(a[idx], b[idx], Q);
    }
}

// Fused multiply-add: result = a * b + c
extern "C" __global__
void fhe_poly_pointwise_mac_kernel(
    uint64_t* __restrict__ result,
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    const uint64_t* __restrict__ c,
    uint64_t Q,
    uint32_t N,
    uint32_t batch_size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = N * batch_size;

    if (idx < total) {
        uint64_t prod = mod_mul(a[idx], b[idx], Q);
        result[idx] = mod_add(prod, c[idx], Q);
    }
}

// ============================================================================
// GLWE Operations
// ============================================================================

// Add GLWE ciphertexts: result = a + b
extern "C" __global__
void fhe_glwe_add_kernel(
    uint64_t* __restrict__ result,
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t Q,
    uint32_t N,
    uint32_t k,
    uint32_t batch_size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t glwe_size = (k + 1) * N;
    uint32_t total = glwe_size * batch_size;

    if (idx < total) {
        result[idx] = mod_add(a[idx], b[idx], Q);
    }
}

// Subtract GLWE ciphertexts: result = a - b
extern "C" __global__
void fhe_glwe_sub_kernel(
    uint64_t* __restrict__ result,
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t Q,
    uint32_t N,
    uint32_t k,
    uint32_t batch_size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t glwe_size = (k + 1) * N;
    uint32_t total = glwe_size * batch_size;

    if (idx < total) {
        result[idx] = mod_sub(a[idx], b[idx], Q);
    }
}

// ============================================================================
// LWE Operations
// ============================================================================

// Add LWE ciphertexts
extern "C" __global__
void fhe_lwe_add_kernel(
    uint64_t* __restrict__ result,
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t Q,
    uint32_t n,
    uint32_t batch_size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t lwe_size = n + 1;
    uint32_t total = lwe_size * batch_size;

    if (idx < total) {
        result[idx] = mod_add(a[idx], b[idx], Q);
    }
}

// Subtract LWE ciphertexts
extern "C" __global__
void fhe_lwe_sub_kernel(
    uint64_t* __restrict__ result,
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t Q,
    uint32_t n,
    uint32_t batch_size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t lwe_size = n + 1;
    uint32_t total = lwe_size * batch_size;

    if (idx < total) {
        result[idx] = mod_sub(a[idx], b[idx], Q);
    }
}

// ============================================================================
// Negacyclic Rotation
// ============================================================================

// Rotate polynomial by k positions in Z_Q[X]/(X^N + 1)
extern "C" __global__
void fhe_negacyclic_rotate_kernel(
    uint64_t* __restrict__ result,
    const uint64_t* __restrict__ input,
    int32_t rotation,
    uint64_t Q,
    uint32_t N,
    uint32_t batch_size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t batch_idx = idx / N;
    uint32_t coeff_idx = idx % N;

    if (batch_idx >= batch_size) return;

    // Normalize rotation to [0, 2N)
    int32_t rot = ((rotation % (int32_t)(2 * N)) + 2 * N) % (2 * N);

    // Compute source index and sign
    int32_t src = (int32_t)coeff_idx - rot;
    bool negate = false;

    while (src < 0) {
        src += N;
        negate = !negate;
    }
    while (src >= (int32_t)N) {
        src -= N;
        negate = !negate;
    }

    uint64_t val = input[batch_idx * N + src];
    result[idx] = negate ? mod_neg(val, Q) : val;
}

// ============================================================================
// Encoding/Decoding
// ============================================================================

// Encode plaintext bits to Torus (for TFHE)
extern "C" __global__
void fhe_encode_bits_kernel(
    uint64_t* __restrict__ encoded,
    const uint32_t* __restrict__ plaintext,
    uint32_t num_bits,
    uint64_t delta  // Scaling factor (typically Q/2 for binary)
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_bits) {
        encoded[idx] = plaintext[idx] ? delta : 0;
    }
}

// Decode Torus to plaintext bits (rounding)
extern "C" __global__
void fhe_decode_bits_kernel(
    uint32_t* __restrict__ plaintext,
    const uint64_t* __restrict__ encoded,
    uint32_t num_bits,
    uint64_t Q,
    uint64_t threshold  // Q/4 for binary
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_bits) {
        uint64_t val = encoded[idx];
        // Normalize to [0, Q)
        val = val % Q;
        // Check if closer to 0 or Q/2
        plaintext[idx] = (val > threshold && val < Q - threshold) ? 1 : 0;
    }
}

// ============================================================================
// Noise Sampling (Gaussian)
// ============================================================================

// Initialize CURAND states
extern "C" __global__
void fhe_init_curand_kernel(
    curandState* __restrict__ states,
    uint64_t seed,
    uint32_t num_states
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_states) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Sample discrete Gaussian noise
extern "C" __global__
void fhe_sample_gaussian_kernel(
    int64_t* __restrict__ output,
    curandState* __restrict__ states,
    double sigma,
    uint32_t num_samples
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_samples) {
        curandState local_state = states[idx];

        // Box-Muller transform for Gaussian
        double u1 = curand_uniform_double(&local_state);
        double u2 = curand_uniform_double(&local_state);

        // Prevent log(0)
        if (u1 < 1e-10) u1 = 1e-10;

        double z = sigma * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        output[idx] = (int64_t)round(z);

        states[idx] = local_state;
    }
}

// Add Gaussian noise to polynomial
extern "C" __global__
void fhe_add_gaussian_noise_kernel(
    uint64_t* __restrict__ poly,
    curandState* __restrict__ states,
    double sigma,
    uint64_t Q,
    uint32_t N,
    uint32_t batch_size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = N * batch_size;

    if (idx < total) {
        curandState local_state = states[idx % N];

        double u1 = curand_uniform_double(&local_state);
        double u2 = curand_uniform_double(&local_state);

        if (u1 < 1e-10) u1 = 1e-10;

        double z = sigma * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        int64_t noise = (int64_t)round(z);

        // Add noise mod Q
        if (noise >= 0) {
            poly[idx] = mod_add(poly[idx], (uint64_t)noise, Q);
        } else {
            poly[idx] = mod_sub(poly[idx], (uint64_t)(-noise), Q);
        }

        states[idx % N] = local_state;
    }
}

// ============================================================================
// Modulus Switching
// ============================================================================

// Switch from modulus Q to modulus P
extern "C" __global__
void fhe_modulus_switch_kernel(
    uint64_t* __restrict__ output,
    const uint64_t* __restrict__ input,
    uint64_t Q,
    uint64_t P,
    uint32_t N,
    uint32_t batch_size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = N * batch_size;

    if (idx < total) {
        // Round(input * P / Q)
        __uint128_t tmp = (__uint128_t)input[idx] * P;
        uint64_t result = (uint64_t)((tmp + Q / 2) / Q);
        output[idx] = result % P;
    }
}

// ============================================================================
// Copy and Initialization
// ============================================================================

extern "C" __global__
void fhe_memset_kernel(
    uint64_t* __restrict__ data,
    uint64_t value,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        data[idx] = value;
    }
}

extern "C" __global__
void fhe_memcpy_kernel(
    uint64_t* __restrict__ dst,
    const uint64_t* __restrict__ src,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        dst[idx] = src[idx];
    }
}

// ============================================================================
// Reduction Operations
// ============================================================================

// Sum all elements (warp-level reduction)
extern "C" __global__
void fhe_reduce_sum_kernel(
    uint64_t* __restrict__ result,
    const uint64_t* __restrict__ input,
    uint64_t Q,
    uint32_t N
) {
    extern __shared__ uint64_t shared[];

    uint32_t tid = threadIdx.x;
    uint32_t idx = blockIdx.x * blockDim.x + tid;

    // Load to shared
    shared[tid] = (idx < N) ? input[idx] : 0;
    __syncthreads();

    // Tree reduction
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = mod_add(shared[tid], shared[tid + s], Q);
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = shared[0];
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

cudaError_t lux_cuda_fhe_poly_add(
    uint64_t* result,
    const uint64_t* a,
    const uint64_t* b,
    uint64_t Q,
    uint32_t N,
    uint32_t batch_size,
    cudaStream_t stream
) {
    uint32_t total = N * batch_size;
    dim3 block(256);
    dim3 grid((total + 255) / 256);

    fhe_poly_add_kernel<<<grid, block, 0, stream>>>(result, a, b, Q, N, batch_size);
    return cudaGetLastError();
}

cudaError_t lux_cuda_fhe_poly_sub(
    uint64_t* result,
    const uint64_t* a,
    const uint64_t* b,
    uint64_t Q,
    uint32_t N,
    uint32_t batch_size,
    cudaStream_t stream
) {
    uint32_t total = N * batch_size;
    dim3 block(256);
    dim3 grid((total + 255) / 256);

    fhe_poly_sub_kernel<<<grid, block, 0, stream>>>(result, a, b, Q, N, batch_size);
    return cudaGetLastError();
}

cudaError_t lux_cuda_fhe_poly_scale(
    uint64_t* result,
    const uint64_t* a,
    uint64_t scalar,
    uint64_t Q,
    uint32_t N,
    uint32_t batch_size,
    cudaStream_t stream
) {
    uint32_t total = N * batch_size;
    dim3 block(256);
    dim3 grid((total + 255) / 256);

    fhe_poly_scale_kernel<<<grid, block, 0, stream>>>(result, a, scalar, Q, N, batch_size);
    return cudaGetLastError();
}

cudaError_t lux_cuda_fhe_poly_pointwise_mul(
    uint64_t* result,
    const uint64_t* a,
    const uint64_t* b,
    uint64_t Q,
    uint32_t N,
    uint32_t batch_size,
    cudaStream_t stream
) {
    uint32_t total = N * batch_size;
    dim3 block(256);
    dim3 grid((total + 255) / 256);

    fhe_poly_pointwise_mul_kernel<<<grid, block, 0, stream>>>(result, a, b, Q, N, batch_size);
    return cudaGetLastError();
}

cudaError_t lux_cuda_fhe_negacyclic_rotate(
    uint64_t* result,
    const uint64_t* input,
    int32_t rotation,
    uint64_t Q,
    uint32_t N,
    uint32_t batch_size,
    cudaStream_t stream
) {
    uint32_t total = N * batch_size;
    dim3 block(256);
    dim3 grid((total + 255) / 256);

    fhe_negacyclic_rotate_kernel<<<grid, block, 0, stream>>>(
        result, input, rotation, Q, N, batch_size
    );
    return cudaGetLastError();
}

cudaError_t lux_cuda_fhe_init_curand(
    curandState* states,
    uint64_t seed,
    uint32_t num_states,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((num_states + 255) / 256);

    fhe_init_curand_kernel<<<grid, block, 0, stream>>>(states, seed, num_states);
    return cudaGetLastError();
}

cudaError_t lux_cuda_fhe_sample_gaussian(
    int64_t* output,
    curandState* states,
    double sigma,
    uint32_t num_samples,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((num_samples + 255) / 256);

    fhe_sample_gaussian_kernel<<<grid, block, 0, stream>>>(output, states, sigma, num_samples);
    return cudaGetLastError();
}

cudaError_t lux_cuda_fhe_modulus_switch(
    uint64_t* output,
    const uint64_t* input,
    uint64_t Q,
    uint64_t P,
    uint32_t N,
    uint32_t batch_size,
    cudaStream_t stream
) {
    uint32_t total = N * batch_size;
    dim3 block(256);
    dim3 grid((total + 255) / 256);

    fhe_modulus_switch_kernel<<<grid, block, 0, stream>>>(output, input, Q, P, N, batch_size);
    return cudaGetLastError();
}

}  // extern "C"
