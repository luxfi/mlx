// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// TFHE Fused External Product CUDA Kernels
// Implements fused operations for maximum performance in bootstrapping.
//
// Fused operations combine multiple steps into single kernels:
// 1. Decomposition + NTT (DecompNTT)
// 2. NTT + Multiply + Accumulate (NTT-MAC)
// 3. Full External Product with Double-Buffering
// 4. CMux with Rotation (for blind rotation)
//
// These fused kernels reduce memory bandwidth and kernel launch overhead.

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================================
// Constants
// ============================================================================

#define MAX_N 2048
#define MAX_K 2
#define MAX_L 8
#define WARP_SIZE 32

// Tensor core tile sizes (for future tensor core support)
#define TENSOR_M 16
#define TENSOR_N 16
#define TENSOR_K 16

// ============================================================================
// Modular Arithmetic
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

// Barrett reduction
__device__ __forceinline__
uint64_t barrett_mul(uint64_t a, uint64_t omega, uint64_t Q, uint64_t precon) {
    uint64_t q_approx = __umul64hi(a, precon);
    uint64_t product = a * omega;
    uint64_t result = product - q_approx * Q;
    return result >= Q ? result - Q : result;
}

// Full modular multiplication
__device__ __forceinline__
uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);

    if (hi == 0) return lo % Q;

    uint64_t r = (1ULL << 32) % Q;
    r = (r * r) % Q;

    uint64_t lo_mod = lo % Q;
    uint64_t hi_mod = (hi % Q) * r % Q;
    return mod_add(lo_mod, hi_mod, Q);
}

// ============================================================================
// NTT Butterflies
// ============================================================================

__device__ __forceinline__
void ct_butterfly(uint64_t* lo, uint64_t* hi, uint64_t omega, uint64_t Q, uint64_t precon) {
    uint64_t u = *lo;
    uint64_t v = barrett_mul(*hi, omega, Q, precon);
    *lo = mod_add(u, v, Q);
    *hi = mod_sub(u, v, Q);
}

__device__ __forceinline__
void gs_butterfly(uint64_t* lo, uint64_t* hi, uint64_t omega, uint64_t Q, uint64_t precon) {
    uint64_t u = *lo;
    uint64_t v = *hi;
    *lo = mod_add(u, v, Q);
    *hi = barrett_mul(mod_sub(u, v, Q), omega, Q, precon);
}

// ============================================================================
// Fused Decomposition + NTT
// ============================================================================

// Parameters for fused operations
struct FusedParams {
    uint64_t Q;              // NTT modulus
    uint64_t N_inv;          // N^{-1} mod Q
    uint64_t N_inv_precon;   // Barrett precomputation
    uint64_t Bg;             // Gadget base
    uint64_t Bg_half;        // Bg / 2
    uint64_t Bg_mask;        // Bg - 1
    uint32_t Bg_bits;        // log2(Bg)
    uint32_t N;              // Ring dimension
    uint32_t log_N;          // log2(N)
    uint32_t k;              // GLWE dimension
    uint32_t L;              // Decomposition levels
};

// Decompose and forward NTT in one pass
// Input: polynomial in coefficient domain
// Output: L polynomials in NTT domain (one per decomposition level)
extern "C" __global__
void fused_decompose_ntt(
    uint64_t* __restrict__ ntt_decomposed,       // [L, N] output in NTT domain
    const uint64_t* __restrict__ poly_in,        // [N] input polynomial
    const uint64_t* __restrict__ twiddles,
    const uint64_t* __restrict__ precons,
    const FusedParams params
) {
    extern __shared__ uint64_t shared[];
    uint64_t* poly_shared = shared;  // [N]

    const uint32_t N = params.N;
    const uint32_t log_N = params.log_N;
    const uint32_t L = params.L;
    const uint64_t Q = params.Q;
    const uint64_t Bg_half = params.Bg_half;
    const uint64_t Bg_mask = params.Bg_mask;
    const uint32_t Bg_bits = params.Bg_bits;

    const uint32_t level = blockIdx.x;  // One block per level
    const uint32_t tid = threadIdx.x;
    const uint32_t tpg = blockDim.x;

    if (level >= L) return;

    // Step 1: Load and decompose to this level
    for (uint32_t i = tid; i < N; i += tpg) {
        uint64_t coeff = poly_in[i];

        // Extract digit at this level
        uint32_t shift = 64 - (L - level) * Bg_bits;
        uint64_t digit = ((coeff >> shift) + Bg_half) & Bg_mask;
        int64_t signed_digit = (int64_t)digit - (int64_t)Bg_half;

        // Convert to field element
        poly_shared[i] = (signed_digit >= 0) ? (uint64_t)signed_digit : (Q + signed_digit);
    }
    __syncthreads();

    // Step 2: Forward NTT in-place
    for (uint32_t stage = 0; stage < log_N; ++stage) {
        uint32_t m = 1U << stage;
        uint32_t t = N >> (stage + 1);

        for (uint32_t b = 0; b < (N / 2 + tpg - 1) / tpg; ++b) {
            uint32_t butterfly_idx = tid + b * tpg;
            if (butterfly_idx >= N / 2) break;

            uint32_t group = butterfly_idx / t;
            uint32_t j = butterfly_idx % t;
            uint32_t idx_lo = group * (2 * t) + j;
            uint32_t idx_hi = idx_lo + t;

            uint32_t tw_idx = m + group;
            uint64_t omega = twiddles[tw_idx];
            uint64_t precon = precons[tw_idx];

            uint64_t lo = poly_shared[idx_lo];
            uint64_t hi = poly_shared[idx_hi];
            ct_butterfly(&lo, &hi, omega, Q, precon);
            poly_shared[idx_lo] = lo;
            poly_shared[idx_hi] = hi;
        }
        __syncthreads();
    }

    // Step 3: Write output
    uint64_t* out = ntt_decomposed + level * N;
    for (uint32_t i = tid; i < N; i += tpg) {
        out[i] = poly_shared[i];
    }
}

// ============================================================================
// Fused NTT-Multiply-Accumulate
// ============================================================================

// Perform pointwise multiply-accumulate in NTT domain
// result[i] += sum_{l=0}^{L-1} decomposed[l][i] * ggsw[l][i]
extern "C" __global__
void fused_ntt_mac(
    uint64_t* __restrict__ result_ntt,           // [N] accumulator in NTT domain
    const uint64_t* __restrict__ decomposed_ntt, // [L, N] decomposed polys in NTT
    const uint64_t* __restrict__ ggsw_row,       // [L, N] GGSW row in NTT
    const FusedParams params
) {
    const uint32_t N = params.N;
    const uint32_t L = params.L;
    const uint64_t Q = params.Q;

    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    uint64_t acc = result_ntt[tid];

    // Accumulate over all decomposition levels
    #pragma unroll
    for (uint32_t l = 0; l < L; ++l) {
        uint64_t d = decomposed_ntt[l * N + tid];
        uint64_t g = ggsw_row[l * N + tid];
        uint64_t prod = mod_mul(d, g, Q);
        acc = mod_add(acc, prod, Q);
    }

    result_ntt[tid] = acc;
}

// ============================================================================
// Full Fused External Product
// ============================================================================

// Complete external product with all operations fused
// Uses double-buffering for decomposition levels
extern "C" __global__
void fused_external_product_full(
    uint64_t* __restrict__ glwe_out,             // [k+1, N] output GLWE
    const uint64_t* __restrict__ glwe_in,        // [k+1, N] input GLWE
    const uint64_t* __restrict__ ggsw,           // [k+1, L, k+1, N] GGSW in NTT
    const uint64_t* __restrict__ twiddles,
    const uint64_t* __restrict__ precons,
    const uint64_t* __restrict__ inv_twiddles,
    const uint64_t* __restrict__ inv_precons,
    const FusedParams params
) {
    extern __shared__ uint64_t shared[];

    const uint32_t N = params.N;
    const uint32_t log_N = params.log_N;
    const uint32_t k = params.k;
    const uint32_t L = params.L;
    const uint64_t Q = params.Q;
    const uint64_t Bg_half = params.Bg_half;
    const uint64_t Bg_mask = params.Bg_mask;
    const uint32_t Bg_bits = params.Bg_bits;

    const uint32_t out_poly = blockIdx.x;  // Which output polynomial
    const uint32_t tid = threadIdx.x;
    const uint32_t tpg = blockDim.x;

    // Shared memory layout: [buffer_a][buffer_b][accumulator]
    uint64_t* buffer_a = shared;
    uint64_t* buffer_b = shared + N;
    uint64_t* accumulator = shared + 2 * N;

    // Initialize accumulator to zero
    for (uint32_t i = tid; i < N; i += tpg) {
        accumulator[i] = 0;
    }
    __syncthreads();

    // Process each input polynomial
    for (uint32_t in_poly = 0; in_poly <= k; ++in_poly) {
        const uint64_t* input_poly = glwe_in + in_poly * N;

        // Double-buffered processing of decomposition levels
        for (uint32_t level = 0; level < L; level += 2) {
            // Process level in buffer_a
            for (uint32_t i = tid; i < N; i += tpg) {
                uint64_t coeff = input_poly[i];
                uint32_t shift = 64 - (L - level) * Bg_bits;
                uint64_t digit = ((coeff >> shift) + Bg_half) & Bg_mask;
                int64_t signed_digit = (int64_t)digit - (int64_t)Bg_half;
                buffer_a[i] = (signed_digit >= 0) ? (uint64_t)signed_digit : (Q + signed_digit);
            }

            // Process level+1 in buffer_b (if exists)
            if (level + 1 < L) {
                for (uint32_t i = tid; i < N; i += tpg) {
                    uint64_t coeff = input_poly[i];
                    uint32_t shift = 64 - (L - level - 1) * Bg_bits;
                    uint64_t digit = ((coeff >> shift) + Bg_half) & Bg_mask;
                    int64_t signed_digit = (int64_t)digit - (int64_t)Bg_half;
                    buffer_b[i] = (signed_digit >= 0) ? (uint64_t)signed_digit : (Q + signed_digit);
                }
            }
            __syncthreads();

            // NTT on buffer_a
            for (uint32_t stage = 0; stage < log_N; ++stage) {
                uint32_t m = 1U << stage;
                uint32_t t = N >> (stage + 1);

                for (uint32_t b = 0; b < (N / 2 + tpg - 1) / tpg; ++b) {
                    uint32_t butterfly_idx = tid + b * tpg;
                    if (butterfly_idx >= N / 2) break;

                    uint32_t group = butterfly_idx / t;
                    uint32_t j = butterfly_idx % t;
                    uint32_t idx_lo = group * (2 * t) + j;
                    uint32_t idx_hi = idx_lo + t;

                    uint64_t omega = twiddles[m + group];
                    uint64_t precon = precons[m + group];

                    uint64_t lo = buffer_a[idx_lo];
                    uint64_t hi = buffer_a[idx_hi];
                    ct_butterfly(&lo, &hi, omega, Q, precon);
                    buffer_a[idx_lo] = lo;
                    buffer_a[idx_hi] = hi;
                }
                __syncthreads();
            }

            // Multiply and accumulate for buffer_a
            uint32_t ggsw_offset = ((in_poly * L + level) * (k + 1) + out_poly) * N;
            for (uint32_t i = tid; i < N; i += tpg) {
                uint64_t prod = mod_mul(buffer_a[i], ggsw[ggsw_offset + i], Q);
                accumulator[i] = mod_add(accumulator[i], prod, Q);
            }
            __syncthreads();

            // NTT and accumulate for buffer_b (if exists)
            if (level + 1 < L) {
                for (uint32_t stage = 0; stage < log_N; ++stage) {
                    uint32_t m = 1U << stage;
                    uint32_t t = N >> (stage + 1);

                    for (uint32_t b = 0; b < (N / 2 + tpg - 1) / tpg; ++b) {
                        uint32_t butterfly_idx = tid + b * tpg;
                        if (butterfly_idx >= N / 2) break;

                        uint32_t group = butterfly_idx / t;
                        uint32_t j = butterfly_idx % t;
                        uint32_t idx_lo = group * (2 * t) + j;
                        uint32_t idx_hi = idx_lo + t;

                        uint64_t omega = twiddles[m + group];
                        uint64_t precon = precons[m + group];

                        uint64_t lo = buffer_b[idx_lo];
                        uint64_t hi = buffer_b[idx_hi];
                        ct_butterfly(&lo, &hi, omega, Q, precon);
                        buffer_b[idx_lo] = lo;
                        buffer_b[idx_hi] = hi;
                    }
                    __syncthreads();
                }

                ggsw_offset = ((in_poly * L + level + 1) * (k + 1) + out_poly) * N;
                for (uint32_t i = tid; i < N; i += tpg) {
                    uint64_t prod = mod_mul(buffer_b[i], ggsw[ggsw_offset + i], Q);
                    accumulator[i] = mod_add(accumulator[i], prod, Q);
                }
                __syncthreads();
            }
        }
    }

    // Inverse NTT on accumulator
    for (uint32_t stage = 0; stage < log_N; ++stage) {
        uint32_t m = N >> (stage + 1);
        uint32_t t = 1U << stage;

        for (uint32_t b = 0; b < (N / 2 + tpg - 1) / tpg; ++b) {
            uint32_t butterfly_idx = tid + b * tpg;
            if (butterfly_idx >= N / 2) break;

            uint32_t group = butterfly_idx / t;
            uint32_t j = butterfly_idx % t;
            uint32_t idx_lo = group * (2 * t) + j;
            uint32_t idx_hi = idx_lo + t;

            uint64_t omega = inv_twiddles[m + group];
            uint64_t precon = inv_precons[m + group];

            uint64_t lo = accumulator[idx_lo];
            uint64_t hi = accumulator[idx_hi];
            gs_butterfly(&lo, &hi, omega, Q, precon);
            accumulator[idx_lo] = lo;
            accumulator[idx_hi] = hi;
        }
        __syncthreads();
    }

    // Scale by N^{-1} and write output
    uint64_t* out = glwe_out + out_poly * N;
    for (uint32_t i = tid; i < N; i += tpg) {
        out[i] = barrett_mul(accumulator[i], params.N_inv, Q, params.N_inv_precon);
    }
}

// ============================================================================
// Fused CMux with Rotation (for Blind Rotation)
// ============================================================================

// CMux: acc = d0 + GGSW * (d1 - d0)
// With rotation: d1 = X^rotation * acc
// Combined: acc = acc + GGSW * (X^rotation * acc - acc)

extern "C" __global__
void fused_cmux_rotate(
    uint64_t* __restrict__ acc,                  // [k+1, N] accumulator (in/out)
    const uint64_t* __restrict__ ggsw,           // [k+1, L, k+1, N] GGSW in NTT
    int32_t rotation,                            // Negacyclic rotation amount
    const uint64_t* __restrict__ twiddles,
    const uint64_t* __restrict__ precons,
    const uint64_t* __restrict__ inv_twiddles,
    const uint64_t* __restrict__ inv_precons,
    const FusedParams params
) {
    extern __shared__ uint64_t shared[];

    const uint32_t N = params.N;
    const uint32_t log_N = params.log_N;
    const uint32_t k = params.k;
    const uint32_t L = params.L;
    const uint64_t Q = params.Q;
    const uint64_t Bg_half = params.Bg_half;
    const uint64_t Bg_mask = params.Bg_mask;
    const uint32_t Bg_bits = params.Bg_bits;

    const uint32_t out_poly = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t tpg = blockDim.x;

    // Shared memory: [acc_local][rotated][diff][product_acc]
    uint64_t* acc_local = shared;
    uint64_t* rotated = shared + N;
    uint64_t* diff = shared + 2 * N;
    uint64_t* product_acc = shared + 3 * N;

    // Load accumulator for this polynomial
    for (uint32_t i = tid; i < N; i += tpg) {
        acc_local[i] = acc[out_poly * N + i];
        product_acc[i] = 0;
    }
    __syncthreads();

    // Skip if no rotation
    if (rotation == 0) return;

    // Normalize rotation to [0, 2N)
    int32_t rot = ((rotation % (int32_t)(2 * N)) + 2 * N) % (2 * N);

    // Process each input polynomial
    for (uint32_t in_poly = 0; in_poly <= k; ++in_poly) {
        const uint64_t* acc_in = acc + in_poly * N;

        // Compute rotated = X^rotation * acc[in_poly]
        for (uint32_t i = tid; i < N; i += tpg) {
            int32_t src = (int32_t)i - rot;
            bool negate = false;

            while (src < 0) {
                src += N;
                negate = !negate;
            }
            while (src >= (int32_t)N) {
                src -= N;
                negate = !negate;
            }

            uint64_t val = acc_in[src];
            rotated[i] = negate ? mod_neg(val, Q) : val;
        }
        __syncthreads();

        // Compute diff = rotated - acc_local (for this in_poly)
        for (uint32_t i = tid; i < N; i += tpg) {
            diff[i] = mod_sub(rotated[i], acc_in[i], Q);
        }
        __syncthreads();

        // Decompose, NTT, multiply, accumulate for each level
        for (uint32_t level = 0; level < L; ++level) {
            // Decompose diff at this level
            for (uint32_t i = tid; i < N; i += tpg) {
                uint64_t coeff = diff[i];
                uint32_t shift = 64 - (L - level) * Bg_bits;
                uint64_t digit = ((coeff >> shift) + Bg_half) & Bg_mask;
                int64_t signed_digit = (int64_t)digit - (int64_t)Bg_half;
                rotated[i] = (signed_digit >= 0) ? (uint64_t)signed_digit : (Q + signed_digit);
            }
            __syncthreads();

            // Forward NTT on decomposed
            for (uint32_t stage = 0; stage < log_N; ++stage) {
                uint32_t m = 1U << stage;
                uint32_t t = N >> (stage + 1);

                for (uint32_t b = 0; b < (N / 2 + tpg - 1) / tpg; ++b) {
                    uint32_t butterfly_idx = tid + b * tpg;
                    if (butterfly_idx >= N / 2) break;

                    uint32_t group = butterfly_idx / t;
                    uint32_t j = butterfly_idx % t;
                    uint32_t idx_lo = group * (2 * t) + j;
                    uint32_t idx_hi = idx_lo + t;

                    uint64_t omega = twiddles[m + group];
                    uint64_t precon = precons[m + group];

                    uint64_t lo = rotated[idx_lo];
                    uint64_t hi = rotated[idx_hi];
                    ct_butterfly(&lo, &hi, omega, Q, precon);
                    rotated[idx_lo] = lo;
                    rotated[idx_hi] = hi;
                }
                __syncthreads();
            }

            // Multiply and accumulate
            uint32_t ggsw_offset = ((in_poly * L + level) * (k + 1) + out_poly) * N;
            for (uint32_t i = tid; i < N; i += tpg) {
                uint64_t prod = mod_mul(rotated[i], ggsw[ggsw_offset + i], Q);
                product_acc[i] = mod_add(product_acc[i], prod, Q);
            }
            __syncthreads();
        }
    }

    // Inverse NTT on product accumulator
    for (uint32_t stage = 0; stage < log_N; ++stage) {
        uint32_t m = N >> (stage + 1);
        uint32_t t = 1U << stage;

        for (uint32_t b = 0; b < (N / 2 + tpg - 1) / tpg; ++b) {
            uint32_t butterfly_idx = tid + b * tpg;
            if (butterfly_idx >= N / 2) break;

            uint32_t group = butterfly_idx / t;
            uint32_t j = butterfly_idx % t;
            uint32_t idx_lo = group * (2 * t) + j;
            uint32_t idx_hi = idx_lo + t;

            uint64_t omega = inv_twiddles[m + group];
            uint64_t precon = inv_precons[m + group];

            uint64_t lo = product_acc[idx_lo];
            uint64_t hi = product_acc[idx_hi];
            gs_butterfly(&lo, &hi, omega, Q, precon);
            product_acc[idx_lo] = lo;
            product_acc[idx_hi] = hi;
        }
        __syncthreads();
    }

    // Add scaled product to accumulator and write back
    for (uint32_t i = tid; i < N; i += tpg) {
        uint64_t scaled = barrett_mul(product_acc[i], params.N_inv, Q, params.N_inv_precon);
        acc[out_poly * N + i] = mod_add(acc_local[i], scaled, Q);
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

// Fused decompose + NTT
cudaError_t lux_cuda_fhe_fused_decompose_ntt(
    uint64_t* ntt_decomposed,
    const uint64_t* poly_in,
    const uint64_t* twiddles,
    const uint64_t* precons,
    uint32_t N,
    uint32_t L,
    uint32_t Bg_bits,
    uint64_t Q,
    cudaStream_t stream
) {
    FusedParams params;
    params.N = N;
    params.log_N = 31 - __builtin_clz(N);
    params.L = L;
    params.Bg_bits = Bg_bits;
    params.Bg = 1ULL << Bg_bits;
    params.Bg_half = 1ULL << (Bg_bits - 1);
    params.Bg_mask = params.Bg - 1;
    params.Q = Q;
    params.k = 0;  // Not used for this kernel

    size_t shmem_size = N * sizeof(uint64_t);
    dim3 grid(L);
    dim3 block(min(N / 2, 256u));

    fused_decompose_ntt<<<grid, block, shmem_size, stream>>>(
        ntt_decomposed, poly_in, twiddles, precons, params
    );

    return cudaGetLastError();
}

// Fused NTT multiply-accumulate
cudaError_t lux_cuda_fhe_fused_ntt_mac(
    uint64_t* result_ntt,
    const uint64_t* decomposed_ntt,
    const uint64_t* ggsw_row,
    uint32_t N,
    uint32_t L,
    uint64_t Q,
    cudaStream_t stream
) {
    FusedParams params;
    params.N = N;
    params.L = L;
    params.Q = Q;

    dim3 grid((N + 255) / 256);
    dim3 block(256);

    fused_ntt_mac<<<grid, block, 0, stream>>>(result_ntt, decomposed_ntt, ggsw_row, params);

    return cudaGetLastError();
}

// Full fused external product
cudaError_t lux_cuda_fhe_fused_external_product(
    uint64_t* glwe_out,
    const uint64_t* glwe_in,
    const uint64_t* ggsw,
    const uint64_t* twiddles,
    const uint64_t* precons,
    const uint64_t* inv_twiddles,
    const uint64_t* inv_precons,
    uint32_t N,
    uint32_t k,
    uint32_t L,
    uint32_t Bg_bits,
    uint64_t Q,
    uint64_t N_inv,
    uint64_t N_inv_precon,
    cudaStream_t stream
) {
    FusedParams params;
    params.N = N;
    params.log_N = 31 - __builtin_clz(N);
    params.k = k;
    params.L = L;
    params.Bg_bits = Bg_bits;
    params.Bg = 1ULL << Bg_bits;
    params.Bg_half = 1ULL << (Bg_bits - 1);
    params.Bg_mask = params.Bg - 1;
    params.Q = Q;
    params.N_inv = N_inv;
    params.N_inv_precon = N_inv_precon;

    size_t shmem_size = 3 * N * sizeof(uint64_t);
    dim3 grid(k + 1);
    dim3 block(min(N / 2, 256u));

    fused_external_product_full<<<grid, block, shmem_size, stream>>>(
        glwe_out, glwe_in, ggsw,
        twiddles, precons, inv_twiddles, inv_precons,
        params
    );

    return cudaGetLastError();
}

// Fused CMux with rotation
cudaError_t lux_cuda_fhe_fused_cmux_rotate(
    uint64_t* acc,
    const uint64_t* ggsw,
    int32_t rotation,
    const uint64_t* twiddles,
    const uint64_t* precons,
    const uint64_t* inv_twiddles,
    const uint64_t* inv_precons,
    uint32_t N,
    uint32_t k,
    uint32_t L,
    uint32_t Bg_bits,
    uint64_t Q,
    uint64_t N_inv,
    uint64_t N_inv_precon,
    cudaStream_t stream
) {
    FusedParams params;
    params.N = N;
    params.log_N = 31 - __builtin_clz(N);
    params.k = k;
    params.L = L;
    params.Bg_bits = Bg_bits;
    params.Bg = 1ULL << Bg_bits;
    params.Bg_half = 1ULL << (Bg_bits - 1);
    params.Bg_mask = params.Bg - 1;
    params.Q = Q;
    params.N_inv = N_inv;
    params.N_inv_precon = N_inv_precon;

    size_t shmem_size = 4 * N * sizeof(uint64_t);
    dim3 grid(k + 1);
    dim3 block(min(N / 2, 256u));

    fused_cmux_rotate<<<grid, block, shmem_size, stream>>>(
        acc, ggsw, rotation,
        twiddles, precons, inv_twiddles, inv_precons,
        params
    );

    return cudaGetLastError();
}

}  // extern "C"
