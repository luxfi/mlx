// Copyright (c) 2024-2026 Lux Industries Inc. All Rights Reserved.
// PROPRIETARY AND CONFIDENTIAL - NO LICENSE GRANTED
// Contact: licensing@luxindustries.xyz

#include <cuda_runtime.h>
#include <cstdint>

namespace lux {
namespace cuda {
namespace fhe {

// TFHE parameters
constexpr uint32_t N = 1024;           // Polynomial degree
constexpr uint32_t K = 1;              // GLWE dimension
constexpr uint32_t L = 3;              // Decomposition levels
constexpr uint32_t BG_BITS = 8;        // Base log2
constexpr uint64_t BG = 1ULL << BG_BITS;
constexpr uint64_t HALF_BG = BG / 2;
constexpr uint64_t MASK = BG - 1;

// Montgomery parameters for modular arithmetic
constexpr uint64_t Q = 0xFFFFFFFF00000001ULL;  // Example modulus
constexpr uint64_t R = 0x100000000ULL;          // Montgomery R

// Signed decomposition
__device__ __forceinline__ int64_t signed_decomp_value(uint64_t val, uint32_t level) {
    uint32_t shift = 64 - (level + 1) * BG_BITS;
    int64_t digit = ((val >> shift) + HALF_BG) & MASK;
    return digit - HALF_BG;
}

// Polynomial multiplication via NTT (external)
extern __device__ void ntt_forward(uint64_t* poly, uint32_t n);
extern __device__ void ntt_inverse(uint64_t* poly, uint32_t n);
extern __device__ void ntt_mul(uint64_t* a, const uint64_t* b, uint32_t n);

// External product: GGSW × GLWE → GLWE
__global__ void external_product_kernel(
    const uint64_t* __restrict__ ggsw,      // GGSW ciphertext [K+1][L][K+1][N]
    const uint64_t* __restrict__ glwe_in,   // Input GLWE [K+1][N]
    uint64_t* __restrict__ glwe_out,        // Output GLWE [K+1][N]
    uint32_t n,
    uint32_t k,
    uint32_t l
) {
    __shared__ uint64_t decomposed[(K+1) * L * N];
    __shared__ uint64_t accumulator[(K+1) * N];
    
    uint32_t tid = threadIdx.x;
    uint32_t stride = blockDim.x;
    
    // Initialize accumulator to zero
    for (uint32_t i = tid; i < (k + 1) * n; i += stride) {
        accumulator[i] = 0;
    }
    __syncthreads();
    
    // Decompose GLWE coefficients
    for (uint32_t poly = 0; poly < k + 1; poly++) {
        for (uint32_t i = tid; i < n; i += stride) {
            uint64_t val = glwe_in[poly * n + i];
            for (uint32_t level = 0; level < l; level++) {
                int64_t digit = signed_decomp_value(val, level);
                // Store as unsigned for NTT
                decomposed[(poly * l + level) * n + i] = (digit >= 0) ? digit : (Q + digit);
            }
        }
    }
    __syncthreads();
    
    // Multiply and accumulate
    for (uint32_t row = 0; row < k + 1; row++) {
        for (uint32_t level = 0; level < l; level++) {
            // Get GGSW row pointer
            const uint64_t* ggsw_row = ggsw + ((row * l + level) * (k + 1)) * n;
            uint64_t* dec_poly = decomposed + (row * l + level) * n;
            
            // Apply NTT to decomposed polynomial
            // (Simplified - real impl uses shared memory NTT)
            
            for (uint32_t col = 0; col < k + 1; col++) {
                // Multiply decomposed × GGSW[row][level][col] and add to accumulator[col]
                for (uint32_t i = tid; i < n; i += stride) {
                    // Schoolbook multiply (simplified - real impl uses NTT)
                    uint64_t sum = 0;
                    for (uint32_t j = 0; j < n; j++) {
                        uint32_t idx = (i + n - j) % n;
                        int sign = (i < j) ? -1 : 1;
                        uint64_t prod = dec_poly[j] * ggsw_row[col * n + idx];
                        sum += (sign > 0) ? prod : (Q - prod);
                    }
                    atomicAdd((unsigned long long*)&accumulator[col * n + i], sum % Q);
                }
            }
        }
    }
    __syncthreads();
    
    // Write output
    for (uint32_t i = tid; i < (k + 1) * n; i += stride) {
        glwe_out[i] = accumulator[i] % Q;
    }
}

// Blind rotation kernel
// Rotates polynomial by encrypted amount using bootstrapping keys
__global__ void blind_rotate_kernel(
    uint64_t* __restrict__ acc,             // Accumulator GLWE [K+1][N]
    const uint64_t* __restrict__ bsk,       // Bootstrapping key [n][K+1][L][K+1][N]
    const uint64_t* __restrict__ lwe_a,     // LWE 'a' coefficients [n]
    uint32_t lwe_n,                         // LWE dimension
    uint32_t glwe_n,                        // GLWE polynomial size
    uint32_t glwe_k,                        // GLWE dimension
    uint32_t decomp_l                       // Decomposition levels
) {
    extern __shared__ uint64_t shmem[];
    
    uint64_t* temp_glwe = shmem;
    uint32_t tid = threadIdx.x;
    uint32_t stride = blockDim.x;
    
    // For each LWE coefficient, perform CMux
    for (uint32_t i = 0; i < lwe_n; i++) {
        uint64_t a_i = lwe_a[i];
        
        // Compute rotation amount
        uint32_t rotation = (uint32_t)((a_i * 2 * glwe_n) >> 64);
        
        if (rotation != 0) {
            // Rotate accumulator by 'rotation' positions (multiply by X^rotation)
            for (uint32_t poly = 0; poly < glwe_k + 1; poly++) {
                for (uint32_t j = tid; j < glwe_n; j += stride) {
                    uint32_t src_idx = (j + glwe_n - rotation) % glwe_n;
                    int sign = (j < rotation) ? -1 : 1;
                    uint64_t val = acc[poly * glwe_n + src_idx];
                    temp_glwe[poly * glwe_n + j] = (sign > 0) ? val : (Q - val);
                }
            }
            __syncthreads();
            
            // Copy back
            for (uint32_t j = tid; j < (glwe_k + 1) * glwe_n; j += stride) {
                acc[j] = temp_glwe[j];
            }
            __syncthreads();
            
            // CMux: acc = bsk[i] × (rotated - acc) + acc
            // This requires external product, simplified here
            const uint64_t* bsk_i = bsk + i * (glwe_k + 1) * decomp_l * (glwe_k + 1) * glwe_n;
            
            // Compute difference (rotated already in acc, need original)
            // In full impl, we'd store original and compute external_product
        }
    }
}

// Programmable bootstrapping
__global__ void programmable_bootstrap_kernel(
    const uint64_t* __restrict__ lwe_in,    // Input LWE ciphertext [n+1]
    uint64_t* __restrict__ lwe_out,         // Output LWE ciphertext [N+1]
    const uint64_t* __restrict__ bsk,       // Bootstrapping key
    const uint64_t* __restrict__ test_poly, // Test polynomial (LUT)
    uint32_t lwe_n,
    uint32_t glwe_n,
    uint32_t glwe_k,
    uint32_t decomp_l
) {
    extern __shared__ uint64_t shmem[];
    uint64_t* acc = shmem;  // Accumulator GLWE
    
    uint32_t tid = threadIdx.x;
    uint32_t stride = blockDim.x;
    
    // Initialize accumulator with rotated test polynomial
    // Rotation amount = round(b * 2N / q)
    uint64_t b = lwe_in[lwe_n];  // LWE body
    uint32_t b_tilde = (uint32_t)((b * 2 * glwe_n) >> 64);
    
    // acc = X^{-b_tilde} * test_poly (in body position)
    for (uint32_t i = tid; i < glwe_n; i += stride) {
        // Mask part is zero
        for (uint32_t k_idx = 0; k_idx < glwe_k; k_idx++) {
            acc[k_idx * glwe_n + i] = 0;
        }
        // Body part is rotated test polynomial
        uint32_t src = (i + b_tilde) % glwe_n;
        int sign = (i + b_tilde >= glwe_n) ? -1 : 1;
        acc[glwe_k * glwe_n + i] = (sign > 0) ? test_poly[src] : (Q - test_poly[src]);
    }
    __syncthreads();
    
    // Blind rotation
    // In full impl, call blind_rotate_kernel or inline the logic
    
    // Sample extract: extract LWE from GLWE
    // lwe_out[0..N-1] = -acc_body[N-1..0] (reversed and negated)
    // lwe_out[N] = acc_body[0]
    
    for (uint32_t i = tid; i < glwe_n; i += stride) {
        uint64_t val = acc[glwe_k * glwe_n + (glwe_n - 1 - i)];
        lwe_out[i] = Q - val;  // Negate
    }
    if (tid == 0) {
        lwe_out[glwe_n] = acc[glwe_k * glwe_n];
    }
}

// Host functions
void blind_rotate(
    uint64_t* acc,
    const uint64_t* bsk,
    const uint64_t* lwe_a,
    uint32_t lwe_n,
    uint32_t glwe_n,
    uint32_t glwe_k,
    uint32_t decomp_l,
    cudaStream_t stream
) {
    size_t shmem_size = (glwe_k + 1) * glwe_n * sizeof(uint64_t);
    dim3 block(256);
    dim3 grid(1);
    blind_rotate_kernel<<<grid, block, shmem_size, stream>>>(
        acc, bsk, lwe_a, lwe_n, glwe_n, glwe_k, decomp_l
    );
}

void external_product(
    const uint64_t* ggsw,
    const uint64_t* glwe_in,
    uint64_t* glwe_out,
    uint32_t n,
    uint32_t k,
    uint32_t l,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(1);
    external_product_kernel<<<grid, block, 0, stream>>>(ggsw, glwe_in, glwe_out, n, k, l);
}

void programmable_bootstrap(
    const uint64_t* lwe_in,
    uint64_t* lwe_out,
    const uint64_t* bsk,
    const uint64_t* test_poly,
    uint32_t lwe_n,
    uint32_t glwe_n,
    uint32_t glwe_k,
    uint32_t decomp_l,
    cudaStream_t stream
) {
    size_t shmem_size = (glwe_k + 1) * glwe_n * sizeof(uint64_t);
    dim3 block(256);
    dim3 grid(1);
    programmable_bootstrap_kernel<<<grid, block, shmem_size, stream>>>(
        lwe_in, lwe_out, bsk, test_poly, lwe_n, glwe_n, glwe_k, decomp_l
    );
}

} // namespace fhe
} // namespace cuda
} // namespace lux
