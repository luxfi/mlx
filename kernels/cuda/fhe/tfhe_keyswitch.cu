// Copyright (c) 2024-2026 Lux Industries Inc. All Rights Reserved.
// PROPRIETARY AND CONFIDENTIAL - NO LICENSE GRANTED
// Contact: licensing@luxindustries.xyz
//
// TFHE LWE Key Switching - High-Performance CUDA Implementation
// Implements decomposition, key multiplication, and accumulation for
// switching between LWE keys of different dimensions.

#include <cuda_runtime.h>
#include <cstdint>

namespace lux {
namespace cuda {
namespace tfhe {

// ============================================================================
// Key Switching Parameters
// ============================================================================

struct KeySwitchParams {
    uint32_t n_in;           // Input LWE dimension
    uint32_t n_out;          // Output LWE dimension
    uint32_t l;              // Decomposition levels
    uint32_t base_log;       // Base log (bits per level)
    uint64_t q;              // Modulus
};

__constant__ KeySwitchParams d_ks_params;

// ============================================================================
// Modular Arithmetic
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
uint64_t mont_mul(uint64_t a, uint64_t b, uint64_t q, uint64_t q_inv) {
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);
    uint64_t m = lo * q_inv;
    uint64_t t = __umul64hi(m, q);
    uint64_t result = hi - t;
    if (hi < t) result += q;
    return result >= q ? result - q : result;
}

// ============================================================================
// Signed Decomposition
// ============================================================================

// Decompose a single value into l signed digits
// Returns digits in base 2^base_log using signed representation
__device__ __forceinline__
void signed_decompose(
    uint64_t val,
    int32_t* digits,
    uint32_t l,
    uint32_t base_log,
    uint64_t q
) {
    const uint64_t base = 1ULL << base_log;
    const uint64_t half_base = base >> 1;
    const uint64_t mask = base - 1;
    
    // Compute digits from MSB to LSB
    #pragma unroll
    for (uint32_t level = 0; level < l; level++) {
        uint32_t shift = 64 - (level + 1) * base_log;
        uint64_t digit = ((val >> shift) + half_base) & mask;
        digits[level] = (int32_t)digit - (int32_t)half_base;
    }
}

// Batch decomposition kernel
__global__ void decompose_lwe_kernel(
    const uint64_t* __restrict__ lwe_in,    // Input LWE mask [n_in]
    int32_t* __restrict__ decomposed,       // Output [n_in][l]
    uint32_t n_in,
    uint32_t l,
    uint32_t base_log,
    uint64_t q
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= n_in) return;
    
    uint64_t val = lwe_in[tid];
    
    const uint64_t base = 1ULL << base_log;
    const uint64_t half_base = base >> 1;
    const uint64_t mask = base - 1;
    
    #pragma unroll
    for (uint32_t level = 0; level < l; level++) {
        uint32_t shift = 64 - (level + 1) * base_log;
        uint64_t digit = ((val >> shift) + half_base) & mask;
        decomposed[tid * l + level] = (int32_t)digit - (int32_t)half_base;
    }
}

// ============================================================================
// Key Switching Core Kernels
// ============================================================================

// Key multiplication and accumulation
// For each output coefficient, sum over input coefficients and levels
__global__ void keyswitch_accumulate_kernel(
    const int32_t* __restrict__ decomposed,     // Decomposed input [n_in][l]
    const uint64_t* __restrict__ ksk,           // Key switching key [n_in][l][n_out+1]
    uint64_t* __restrict__ lwe_out,             // Output LWE [n_out+1]
    uint32_t n_in,
    uint32_t n_out,
    uint32_t l,
    uint64_t q
) {
    const uint32_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_idx > n_out) return;  // n_out+1 total outputs
    
    uint64_t acc = 0;
    
    // Sum over all input coefficients and decomposition levels
    for (uint32_t in_idx = 0; in_idx < n_in; in_idx++) {
        for (uint32_t level = 0; level < l; level++) {
            int32_t digit = decomposed[in_idx * l + level];
            
            if (digit == 0) continue;
            
            // KSK layout: [n_in][l][n_out+1]
            uint64_t ksk_val = ksk[(in_idx * l + level) * (n_out + 1) + out_idx];
            
            // Multiply by signed digit
            if (digit > 0) {
                uint64_t prod = ((uint64_t)digit * ksk_val) % q;
                acc = mod_add(acc, prod, q);
            } else {
                uint64_t prod = ((uint64_t)(-digit) * ksk_val) % q;
                acc = mod_sub(acc, prod, q);
            }
        }
    }
    
    lwe_out[out_idx] = acc;
}

// Optimized: shared memory accumulation with warp reduction
__global__ void keyswitch_accumulate_shared_kernel(
    const int32_t* __restrict__ decomposed,
    const uint64_t* __restrict__ ksk,
    uint64_t* __restrict__ lwe_out,
    uint32_t n_in,
    uint32_t n_out,
    uint32_t l,
    uint64_t q
) {
    extern __shared__ uint64_t shmem[];
    
    const uint32_t out_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t stride = blockDim.x;
    
    if (out_idx > n_out) return;
    
    // Each thread accumulates a portion
    uint64_t local_acc = 0;
    
    for (uint32_t i = tid; i < n_in * l; i += stride) {
        uint32_t in_idx = i / l;
        uint32_t level = i % l;
        
        int32_t digit = decomposed[in_idx * l + level];
        
        if (digit != 0) {
            uint64_t ksk_val = ksk[(in_idx * l + level) * (n_out + 1) + out_idx];
            
            if (digit > 0) {
                uint64_t prod = ((uint64_t)digit * ksk_val) % q;
                local_acc = mod_add(local_acc, prod, q);
            } else {
                uint64_t prod = ((uint64_t)(-digit) * ksk_val) % q;
                local_acc = mod_sub(local_acc, prod, q);
            }
        }
    }
    
    // Store to shared memory
    shmem[tid] = local_acc;
    __syncthreads();
    
    // Reduction in shared memory
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shmem[tid] = mod_add(shmem[tid], shmem[tid + s], q);
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        lwe_out[out_idx] = shmem[0];
    }
}

// Combined decomposition and key switching in single kernel
__global__ void keyswitch_fused_kernel(
    const uint64_t* __restrict__ lwe_in,    // Input LWE [n_in+1]
    const uint64_t* __restrict__ ksk,       // Key switching key [n_in][l][n_out+1]
    uint64_t* __restrict__ lwe_out,         // Output LWE [n_out+1]
    uint32_t n_in,
    uint32_t n_out,
    uint32_t l,
    uint32_t base_log,
    uint64_t q
) {
    extern __shared__ uint64_t shmem[];
    int32_t* s_decomposed = (int32_t*)shmem;  // [n_in][l]
    
    const uint32_t tid = threadIdx.x;
    const uint32_t stride = blockDim.x;
    
    // Step 1: Decompose all input coefficients (collaborative)
    const uint64_t base = 1ULL << base_log;
    const uint64_t half_base = base >> 1;
    const uint64_t mask = base - 1;
    
    for (uint32_t i = tid; i < n_in; i += stride) {
        uint64_t val = lwe_in[i];
        
        for (uint32_t level = 0; level < l; level++) {
            uint32_t shift = 64 - (level + 1) * base_log;
            uint64_t digit = ((val >> shift) + half_base) & mask;
            s_decomposed[i * l + level] = (int32_t)digit - (int32_t)half_base;
        }
    }
    __syncthreads();
    
    // Step 2: Key multiplication and accumulation
    // Each thread handles one output coefficient
    for (uint32_t out_idx = tid; out_idx <= n_out; out_idx += stride) {
        uint64_t acc = 0;
        
        if (out_idx == n_out) {
            // Body: copy from input
            acc = lwe_in[n_in];
        }
        
        for (uint32_t in_idx = 0; in_idx < n_in; in_idx++) {
            for (uint32_t level = 0; level < l; level++) {
                int32_t digit = s_decomposed[in_idx * l + level];
                
                if (digit != 0) {
                    uint64_t ksk_val = ksk[(in_idx * l + level) * (n_out + 1) + out_idx];
                    
                    if (digit > 0) {
                        uint64_t prod = ((uint64_t)digit * ksk_val) % q;
                        acc = mod_add(acc, prod, q);
                    } else {
                        uint64_t prod = ((uint64_t)(-digit) * ksk_val) % q;
                        acc = mod_sub(acc, prod, q);
                    }
                }
            }
        }
        
        lwe_out[out_idx] = acc;
    }
}

// ============================================================================
// Packing Key Switch: Multiple LWE -> Single RLWE
// ============================================================================

// Pack multiple LWE ciphertexts into a single RLWE ciphertext
__global__ void packing_keyswitch_kernel(
    const uint64_t* __restrict__ lwe_batch,     // Input: batch of LWE [batch_size][n_lwe+1]
    const uint64_t* __restrict__ pksk,          // Packing key [n_lwe][l][2][N]
    uint64_t* __restrict__ rlwe_out,            // Output RLWE [2][N]
    uint32_t batch_size,
    uint32_t n_lwe,
    uint32_t N,                                 // RLWE polynomial degree
    uint32_t l,
    uint32_t base_log,
    uint64_t q
) {
    extern __shared__ uint64_t shmem[];
    uint64_t* acc = shmem;  // [2][N]
    
    const uint32_t tid = threadIdx.x;
    const uint32_t stride = blockDim.x;
    
    // Initialize accumulator
    for (uint32_t i = tid; i < 2 * N; i += stride) {
        acc[i] = 0;
    }
    __syncthreads();
    
    const uint64_t base = 1ULL << base_log;
    const uint64_t half_base = base >> 1;
    const uint64_t mask = base - 1;
    
    // Process each LWE in the batch
    for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        const uint64_t* lwe = lwe_batch + batch_idx * (n_lwe + 1);
        
        // For each input coefficient
        for (uint32_t in_idx = 0; in_idx < n_lwe; in_idx++) {
            uint64_t val = lwe[in_idx];
            
            // Decompose
            for (uint32_t level = 0; level < l; level++) {
                uint32_t shift = 64 - (level + 1) * base_log;
                int32_t digit = (int32_t)(((val >> shift) + half_base) & mask) - (int32_t)half_base;
                
                if (digit == 0) continue;
                
                // Multiply by PKSK element and add to accumulator
                // PKSK layout: [n_lwe][l][2][N]
                for (uint32_t i = tid; i < 2 * N; i += stride) {
                    uint32_t poly_idx = i / N;
                    uint32_t coeff_idx = i % N;
                    
                    uint64_t pksk_val = pksk[((in_idx * l + level) * 2 + poly_idx) * N + coeff_idx];
                    
                    // Apply rotation for position in batch
                    // (Simplified - full impl would rotate polynomial)
                    
                    if (digit > 0) {
                        uint64_t prod = ((uint64_t)digit * pksk_val) % q;
                        acc[i] = mod_add(acc[i], prod, q);
                    } else {
                        uint64_t prod = ((uint64_t)(-digit) * pksk_val) % q;
                        acc[i] = mod_sub(acc[i], prod, q);
                    }
                }
                __syncthreads();
            }
        }
    }
    
    // Write output
    for (uint32_t i = tid; i < 2 * N; i += stride) {
        rlwe_out[i] = acc[i];
    }
}

// ============================================================================
// Private Functional Key Switch
// ============================================================================

// PFKS: Functional key switching for evaluating linear functions
__global__ void functional_keyswitch_kernel(
    const uint64_t* __restrict__ lwe_in,        // Input LWE [n_in+1]
    const uint64_t* __restrict__ fksk,          // Functional KSK [n_in][l][n_out+1]
    const int64_t* __restrict__ function_coeffs, // Function coefficients [n_in]
    uint64_t* __restrict__ lwe_out,             // Output LWE [n_out+1]
    uint32_t n_in,
    uint32_t n_out,
    uint32_t l,
    uint32_t base_log,
    uint64_t q
) {
    extern __shared__ uint64_t shmem[];
    int32_t* s_decomposed = (int32_t*)shmem;
    
    const uint32_t tid = threadIdx.x;
    const uint32_t stride = blockDim.x;
    
    const uint64_t base = 1ULL << base_log;
    const uint64_t half_base = base >> 1;
    const uint64_t mask = base - 1;
    
    // Decompose scaled input: a[i] * f[i]
    for (uint32_t i = tid; i < n_in; i += stride) {
        int64_t f_i = function_coeffs[i];
        uint64_t a_i = lwe_in[i];
        
        // Scale coefficient
        int64_t scaled = ((int64_t)a_i * f_i) % (int64_t)q;
        if (scaled < 0) scaled += q;
        
        // Decompose
        for (uint32_t level = 0; level < l; level++) {
            uint32_t shift = 64 - (level + 1) * base_log;
            uint64_t digit = (((uint64_t)scaled >> shift) + half_base) & mask;
            s_decomposed[i * l + level] = (int32_t)digit - (int32_t)half_base;
        }
    }
    __syncthreads();
    
    // Accumulate using FKSK
    for (uint32_t out_idx = tid; out_idx <= n_out; out_idx += stride) {
        uint64_t acc = 0;
        
        for (uint32_t in_idx = 0; in_idx < n_in; in_idx++) {
            for (uint32_t level = 0; level < l; level++) {
                int32_t digit = s_decomposed[in_idx * l + level];
                
                if (digit != 0) {
                    uint64_t fksk_val = fksk[(in_idx * l + level) * (n_out + 1) + out_idx];
                    
                    if (digit > 0) {
                        uint64_t prod = ((uint64_t)digit * fksk_val) % q;
                        acc = mod_add(acc, prod, q);
                    } else {
                        uint64_t prod = ((uint64_t)(-digit) * fksk_val) % q;
                        acc = mod_sub(acc, prod, q);
                    }
                }
            }
        }
        
        lwe_out[out_idx] = acc;
    }
}

// ============================================================================
// Batched Key Switching
// ============================================================================

// Process multiple LWE ciphertexts in parallel
__global__ void batch_keyswitch_kernel(
    const uint64_t* __restrict__ lwe_batch_in,  // Input [batch][n_in+1]
    const uint64_t* __restrict__ ksk,           // KSK [n_in][l][n_out+1]
    uint64_t* __restrict__ lwe_batch_out,       // Output [batch][n_out+1]
    uint32_t batch_size,
    uint32_t n_in,
    uint32_t n_out,
    uint32_t l,
    uint32_t base_log,
    uint64_t q
) {
    const uint32_t batch_idx = blockIdx.y;
    const uint32_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || out_idx > n_out) return;
    
    const uint64_t* lwe_in = lwe_batch_in + batch_idx * (n_in + 1);
    uint64_t* lwe_out = lwe_batch_out + batch_idx * (n_out + 1);
    
    const uint64_t base = 1ULL << base_log;
    const uint64_t half_base = base >> 1;
    const uint64_t mask = base - 1;
    
    uint64_t acc = 0;
    
    // Body passthrough
    if (out_idx == n_out) {
        acc = lwe_in[n_in];
    }
    
    // Accumulate over all inputs
    for (uint32_t in_idx = 0; in_idx < n_in; in_idx++) {
        uint64_t val = lwe_in[in_idx];
        
        for (uint32_t level = 0; level < l; level++) {
            uint32_t shift = 64 - (level + 1) * base_log;
            int32_t digit = (int32_t)(((val >> shift) + half_base) & mask) - (int32_t)half_base;
            
            if (digit != 0) {
                uint64_t ksk_val = ksk[(in_idx * l + level) * (n_out + 1) + out_idx];
                
                if (digit > 0) {
                    uint64_t prod = ((uint64_t)digit * ksk_val) % q;
                    acc = mod_add(acc, prod, q);
                } else {
                    uint64_t prod = ((uint64_t)(-digit) * ksk_val) % q;
                    acc = mod_sub(acc, prod, q);
                }
            }
        }
    }
    
    lwe_out[out_idx] = acc;
}

// ============================================================================
// Key Switching Key Generation Helper
// ============================================================================

// Generate key switching key element
// ksk[i][j] = s_out * decomp_j(-s_in[i]) + e
__global__ void generate_ksk_element_kernel(
    const uint64_t* __restrict__ s_in,      // Input secret key [n_in]
    const uint64_t* __restrict__ s_out,     // Output secret key [n_out]
    const uint64_t* __restrict__ random_a,  // Random masks [n_in][l][n_out]
    const int64_t* __restrict__ errors,     // Error samples [n_in][l]
    uint64_t* __restrict__ ksk,             // Output KSK [n_in][l][n_out+1]
    uint32_t n_in,
    uint32_t n_out,
    uint32_t l,
    uint32_t base_log,
    uint64_t q
) {
    const uint32_t in_idx = blockIdx.y;
    const uint32_t level = blockIdx.z;
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (in_idx >= n_in || level >= l) return;
    
    const size_t ksk_base = (in_idx * l + level) * (n_out + 1);
    
    if (tid < n_out) {
        // Mask coefficients: just copy random_a
        ksk[ksk_base + tid] = random_a[(in_idx * l + level) * n_out + tid];
    } else if (tid == n_out) {
        // Body coefficient: sum(a[j] * s_out[j]) + e + decomp_level(s_in[i])
        uint64_t body = 0;
        
        // Compute inner product <a, s_out>
        for (uint32_t j = 0; j < n_out; j++) {
            uint64_t prod = (random_a[(in_idx * l + level) * n_out + j] * s_out[j]) % q;
            body = mod_add(body, prod, q);
        }
        
        // Add error
        int64_t e = errors[in_idx * l + level];
        if (e >= 0) {
            body = mod_add(body, (uint64_t)e, q);
        } else {
            body = mod_sub(body, (uint64_t)(-e), q);
        }
        
        // Add scaled secret key coefficient
        // decomp_level(s_in[i]) = s_in[i] * B^{l-1-level}
        uint32_t shift = (l - 1 - level) * base_log;
        uint64_t scaled = (s_in[in_idx] << shift) % q;
        body = mod_add(body, scaled, q);
        
        ksk[ksk_base + n_out] = body;
    }
}

// ============================================================================
// Host API
// ============================================================================

void keyswitch(
    const uint64_t* lwe_in,
    const uint64_t* ksk,
    uint64_t* lwe_out,
    uint32_t n_in,
    uint32_t n_out,
    uint32_t l,
    uint32_t base_log,
    uint64_t q,
    cudaStream_t stream
) {
    // Allocate decomposed buffer
    int32_t* d_decomposed;
    cudaMalloc(&d_decomposed, n_in * l * sizeof(int32_t));
    
    // Decompose
    dim3 decomp_block(256);
    dim3 decomp_grid((n_in + 255) / 256);
    decompose_lwe_kernel<<<decomp_grid, decomp_block, 0, stream>>>(
        lwe_in, d_decomposed, n_in, l, base_log, q
    );
    
    // Key switching with shared memory
    dim3 ks_block(256);
    dim3 ks_grid(n_out + 1);
    size_t shmem_size = ks_block.x * sizeof(uint64_t);
    
    keyswitch_accumulate_shared_kernel<<<ks_grid, ks_block, shmem_size, stream>>>(
        d_decomposed, ksk, lwe_out, n_in, n_out, l, q
    );
    
    cudaFree(d_decomposed);
}

void keyswitch_fused(
    const uint64_t* lwe_in,
    const uint64_t* ksk,
    uint64_t* lwe_out,
    uint32_t n_in,
    uint32_t n_out,
    uint32_t l,
    uint32_t base_log,
    uint64_t q,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(1);
    size_t shmem_size = n_in * l * sizeof(int32_t);
    
    keyswitch_fused_kernel<<<grid, block, shmem_size, stream>>>(
        lwe_in, ksk, lwe_out, n_in, n_out, l, base_log, q
    );
}

void batch_keyswitch(
    const uint64_t* lwe_batch_in,
    const uint64_t* ksk,
    uint64_t* lwe_batch_out,
    uint32_t batch_size,
    uint32_t n_in,
    uint32_t n_out,
    uint32_t l,
    uint32_t base_log,
    uint64_t q,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((n_out + 256) / 256, batch_size);
    
    batch_keyswitch_kernel<<<grid, block, 0, stream>>>(
        lwe_batch_in, ksk, lwe_batch_out, batch_size, n_in, n_out, l, base_log, q
    );
}

void packing_keyswitch(
    const uint64_t* lwe_batch,
    const uint64_t* pksk,
    uint64_t* rlwe_out,
    uint32_t batch_size,
    uint32_t n_lwe,
    uint32_t N,
    uint32_t l,
    uint32_t base_log,
    uint64_t q,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(1);
    size_t shmem_size = 2 * N * sizeof(uint64_t);
    
    packing_keyswitch_kernel<<<grid, block, shmem_size, stream>>>(
        lwe_batch, pksk, rlwe_out, batch_size, n_lwe, N, l, base_log, q
    );
}

void functional_keyswitch(
    const uint64_t* lwe_in,
    const uint64_t* fksk,
    const int64_t* function_coeffs,
    uint64_t* lwe_out,
    uint32_t n_in,
    uint32_t n_out,
    uint32_t l,
    uint32_t base_log,
    uint64_t q,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(1);
    size_t shmem_size = n_in * l * sizeof(int32_t);
    
    functional_keyswitch_kernel<<<grid, block, shmem_size, stream>>>(
        lwe_in, fksk, function_coeffs, lwe_out, n_in, n_out, l, base_log, q
    );
}

} // namespace tfhe
} // namespace cuda
} // namespace lux
