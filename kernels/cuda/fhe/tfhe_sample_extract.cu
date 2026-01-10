// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// TFHE Sample Extraction CUDA Kernels
// Implements extraction of LWE ciphertexts from GLWE ciphertexts.
//
// Sample extraction converts a polynomial ciphertext (GLWE/RLWE) to a scalar
// ciphertext (LWE) by extracting a specific coefficient. This is the final
// step of programmable bootstrapping.
//
// For negacyclic polynomial ring Z_q[X]/(X^N + 1):
//   Extract at position p from GLWE(a(X), b(X)) gives LWE(a', b')
//   where a'[i] = -a[p-i-1 mod N] with sign flip on wrap
//   and b' = b[p]

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================================
// Constants
// ============================================================================

#define MAX_N 2048
#define MAX_K 2
#define WARP_SIZE 32

// ============================================================================
// Modular Arithmetic (Torus mod 2^64)
// ============================================================================

__device__ __forceinline__
uint64_t torus_neg(uint64_t a) {
    return (~a) + 1;  // Two's complement negation
}

// For prime modulus (when working in NTT domain)
__device__ __forceinline__
uint64_t mod_neg(uint64_t a, uint64_t Q) {
    return a == 0 ? 0 : Q - a;
}

// ============================================================================
// Sample Extraction Parameters
// ============================================================================

struct SampleExtractParams {
    uint32_t N;           // GLWE polynomial degree
    uint32_t k;           // GLWE dimension (typically 1)
    uint32_t n_lwe;       // Output LWE dimension (N for standard, or smaller after key switch)
    uint64_t Q;           // Modulus (0 for Torus32/64)
    uint32_t batch_size;  // Number of extractions
};

// ============================================================================
// Sample Extract at Position 0 (Most Common Case)
// ============================================================================

// Standard sample extraction at position 0:
// From GLWE(a(X), b(X)) extracts LWE(a', b') where:
//   a'[i] = -a[N-1-i] for i = 0..N-1
//   b' = b[0]
//
// For k > 1: a' is concatenation of extractions from all mask polynomials

extern "C" __global__
void tfhe_sample_extract_0(
    uint64_t* __restrict__ lwe_out,         // [batch, k*N + 1] output LWE
    const uint64_t* __restrict__ glwe_in,   // [batch, k+1, N] input GLWE
    const SampleExtractParams params
) {
    const uint32_t N = params.N;
    const uint32_t k = params.k;
    const uint32_t batch_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t tpg = blockDim.x;

    const uint64_t* glwe = glwe_in + batch_idx * (k + 1) * N;
    uint64_t* lwe = lwe_out + batch_idx * (k * N + 1);

    // Extract mask coefficients: lwe[poly_idx * N + i] = -glwe[poly_idx][N-1-i]
    for (uint32_t poly_idx = 0; poly_idx < k; ++poly_idx) {
        const uint64_t* mask_poly = glwe + poly_idx * N;

        for (uint32_t i = tid; i < N; i += tpg) {
            uint64_t val = mask_poly[N - 1 - i];
            lwe[poly_idx * N + i] = torus_neg(val);
        }
    }

    // Extract body coefficient (first thread only)
    if (tid == 0) {
        lwe[k * N] = glwe[k * N];  // b' = b[0]
    }
}

// Version with prime modulus
extern "C" __global__
void tfhe_sample_extract_0_modq(
    uint64_t* __restrict__ lwe_out,
    const uint64_t* __restrict__ glwe_in,
    const SampleExtractParams params
) {
    const uint32_t N = params.N;
    const uint32_t k = params.k;
    const uint64_t Q = params.Q;
    const uint32_t batch_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t tpg = blockDim.x;

    const uint64_t* glwe = glwe_in + batch_idx * (k + 1) * N;
    uint64_t* lwe = lwe_out + batch_idx * (k * N + 1);

    for (uint32_t poly_idx = 0; poly_idx < k; ++poly_idx) {
        const uint64_t* mask_poly = glwe + poly_idx * N;

        for (uint32_t i = tid; i < N; i += tpg) {
            uint64_t val = mask_poly[N - 1 - i];
            lwe[poly_idx * N + i] = mod_neg(val, Q);
        }
    }

    if (tid == 0) {
        lwe[k * N] = glwe[k * N];
    }
}

// ============================================================================
// Sample Extract at Arbitrary Position
// ============================================================================

// Extract at position p (0 <= p < N):
// For negacyclic ring: X^N = -1
// a'[i] = coeff of X^{p-i-1} in a(X), with sign flip when index wraps around 0
//
// More precisely, for j = (p - i - 1) mod 2N:
//   if j < N: a'[i] = -a[j]
//   if j >= N: a'[i] = a[j - N]

extern "C" __global__
void tfhe_sample_extract_at(
    uint64_t* __restrict__ lwe_out,         // [batch, k*N + 1]
    const uint64_t* __restrict__ glwe_in,   // [batch, k+1, N]
    uint32_t position,                       // Extraction position (0..N-1)
    const SampleExtractParams params
) {
    const uint32_t N = params.N;
    const uint32_t k = params.k;
    const uint32_t batch_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t tpg = blockDim.x;

    const uint64_t* glwe = glwe_in + batch_idx * (k + 1) * N;
    uint64_t* lwe = lwe_out + batch_idx * (k * N + 1);

    // Normalize position
    int32_t p = (int32_t)(position % N);

    for (uint32_t poly_idx = 0; poly_idx < k; ++poly_idx) {
        const uint64_t* mask_poly = glwe + poly_idx * N;

        for (uint32_t i = tid; i < N; i += tpg) {
            // Compute source index: j = (p - i - 1) in Z/2NZ
            int32_t j = p - (int32_t)i - 1;

            // Normalize to [0, 2N)
            while (j < 0) j += 2 * N;
            while (j >= 2 * (int32_t)N) j -= 2 * N;

            // Determine sign and actual index
            bool negate = (j < (int32_t)N);  // Negate if in first half
            uint32_t src_idx = negate ? (uint32_t)j : (uint32_t)(j - N);

            uint64_t val = mask_poly[src_idx];
            lwe[poly_idx * N + i] = negate ? torus_neg(val) : val;
        }
    }

    // Body coefficient
    if (tid == 0) {
        lwe[k * N] = glwe[k * N + position];
    }
}

// Version with prime modulus
extern "C" __global__
void tfhe_sample_extract_at_modq(
    uint64_t* __restrict__ lwe_out,
    const uint64_t* __restrict__ glwe_in,
    uint32_t position,
    const SampleExtractParams params
) {
    const uint32_t N = params.N;
    const uint32_t k = params.k;
    const uint64_t Q = params.Q;
    const uint32_t batch_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t tpg = blockDim.x;

    const uint64_t* glwe = glwe_in + batch_idx * (k + 1) * N;
    uint64_t* lwe = lwe_out + batch_idx * (k * N + 1);

    int32_t p = (int32_t)(position % N);

    for (uint32_t poly_idx = 0; poly_idx < k; ++poly_idx) {
        const uint64_t* mask_poly = glwe + poly_idx * N;

        for (uint32_t i = tid; i < N; i += tpg) {
            int32_t j = p - (int32_t)i - 1;
            while (j < 0) j += 2 * N;
            while (j >= 2 * (int32_t)N) j -= 2 * N;

            bool negate = (j < (int32_t)N);
            uint32_t src_idx = negate ? (uint32_t)j : (uint32_t)(j - N);

            uint64_t val = mask_poly[src_idx];
            lwe[poly_idx * N + i] = negate ? mod_neg(val, Q) : val;
        }
    }

    if (tid == 0) {
        lwe[k * N] = glwe[k * N + position];
    }
}

// ============================================================================
// Batch Sample Extract (Multiple Positions)
// ============================================================================

// Extract from multiple positions, useful for SIMD-style operations
extern "C" __global__
void tfhe_sample_extract_batch(
    uint64_t* __restrict__ lwe_out,              // [batch, num_pos, k*N + 1]
    const uint64_t* __restrict__ glwe_in,        // [batch, k+1, N]
    const uint32_t* __restrict__ positions,      // [batch, num_pos] or [num_pos] if broadcast
    uint32_t num_positions,
    bool positions_broadcast,                     // If true, same positions for all batch
    const SampleExtractParams params
) {
    const uint32_t N = params.N;
    const uint32_t k = params.k;
    const uint32_t batch_idx = blockIdx.x;
    const uint32_t pos_idx = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t tpg = blockDim.x;

    if (pos_idx >= num_positions) return;

    // Get position
    uint32_t position;
    if (positions_broadcast) {
        position = positions[pos_idx];
    } else {
        position = positions[batch_idx * num_positions + pos_idx];
    }

    const uint64_t* glwe = glwe_in + batch_idx * (k + 1) * N;
    uint64_t* lwe = lwe_out + (batch_idx * num_positions + pos_idx) * (k * N + 1);

    int32_t p = (int32_t)(position % N);

    for (uint32_t poly_idx = 0; poly_idx < k; ++poly_idx) {
        const uint64_t* mask_poly = glwe + poly_idx * N;

        for (uint32_t i = tid; i < N; i += tpg) {
            int32_t j = p - (int32_t)i - 1;
            while (j < 0) j += 2 * N;
            while (j >= 2 * (int32_t)N) j -= 2 * N;

            bool negate = (j < (int32_t)N);
            uint32_t src_idx = negate ? (uint32_t)j : (uint32_t)(j - N);

            uint64_t val = mask_poly[src_idx];
            lwe[poly_idx * N + i] = negate ? torus_neg(val) : val;
        }
    }

    if (tid == 0) {
        lwe[k * N] = glwe[k * N + position];
    }
}

// ============================================================================
// Sample Extract with Key Switching (Fused)
// ============================================================================

// Key switch parameters for reducing LWE dimension
struct KeySwitchParams {
    uint32_t n_in;        // Input LWE dimension (k * N)
    uint32_t n_out;       // Output LWE dimension
    uint32_t L;           // Decomposition levels
    uint64_t Bg;          // Decomposition base
    uint64_t Bg_half;     // Bg / 2
    uint64_t Bg_mask;     // Bg - 1
    uint32_t Bg_bits;     // log2(Bg)
};

// Fused sample extract + key switch
// Reduces from N-dimensional LWE to n-dimensional LWE
extern "C" __global__
void tfhe_sample_extract_keyswitch(
    uint64_t* __restrict__ lwe_out,              // [batch, n_out + 1]
    const uint64_t* __restrict__ glwe_in,        // [batch, k+1, N]
    const uint64_t* __restrict__ ksk,            // [k*N, L, n_out + 1] key switching key
    uint32_t position,
    const SampleExtractParams se_params,
    const KeySwitchParams ks_params
) {
    extern __shared__ uint64_t shared[];

    const uint32_t N = se_params.N;
    const uint32_t k = se_params.k;
    const uint32_t n_in = ks_params.n_in;
    const uint32_t n_out = ks_params.n_out;
    const uint32_t L = ks_params.L;
    const uint64_t Bg = ks_params.Bg;
    const uint64_t Bg_half = ks_params.Bg_half;
    const uint64_t Bg_mask = ks_params.Bg_mask;
    const uint32_t Bg_bits = ks_params.Bg_bits;

    const uint32_t batch_idx = blockIdx.x;
    const uint32_t out_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (out_idx > n_out) return;  // n_out + 1 elements

    const uint64_t* glwe = glwe_in + batch_idx * (k + 1) * N;
    uint64_t* lwe = lwe_out + batch_idx * (n_out + 1);

    int32_t p = (int32_t)(position % N);

    // Initialize accumulator
    uint64_t acc = 0;

    // If this is the body coefficient
    if (out_idx == n_out) {
        acc = glwe[k * N + position];
    }

    // Key switching: accumulate over all input LWE coefficients
    for (uint32_t poly_idx = 0; poly_idx < k; ++poly_idx) {
        const uint64_t* mask_poly = glwe + poly_idx * N;

        for (uint32_t i = 0; i < N; ++i) {
            // Extract coefficient
            int32_t j = p - (int32_t)i - 1;
            while (j < 0) j += 2 * N;
            while (j >= 2 * (int32_t)N) j -= 2 * N;

            bool negate = (j < (int32_t)N);
            uint32_t src_idx = negate ? (uint32_t)j : (uint32_t)(j - N);

            uint64_t val = mask_poly[src_idx];
            uint64_t lwe_coeff = negate ? torus_neg(val) : val;

            // Decompose and apply KSK
            uint32_t lwe_idx = poly_idx * N + i;

            for (uint32_t l = 0; l < L; ++l) {
                // Extract digit
                uint32_t shift = 64 - (l + 1) * Bg_bits;
                uint64_t digit = ((lwe_coeff >> shift) + Bg_half) & Bg_mask;
                int64_t signed_digit = (int64_t)digit - (int64_t)Bg_half;

                if (signed_digit != 0) {
                    // KSK layout: ksk[lwe_idx][l][out_idx]
                    uint32_t ksk_offset = (lwe_idx * L + l) * (n_out + 1) + out_idx;
                    uint64_t ksk_val = __ldg(&ksk[ksk_offset]);

                    // Multiply digit by KSK value and accumulate
                    if (signed_digit > 0) {
                        acc -= (uint64_t)signed_digit * ksk_val;
                    } else {
                        acc += (uint64_t)(-signed_digit) * ksk_val;
                    }
                }
            }
        }
    }

    lwe[out_idx] = acc;
}

// ============================================================================
// Multi-Value Sample Extract (for WoPBS)
// ============================================================================

// Extract multiple values from multiple polynomials at once
// Used in WoP-PBS where we want to extract several LWE ciphertexts
extern "C" __global__
void tfhe_sample_extract_multi(
    uint64_t* __restrict__ lwe_out,              // [batch, num_glwe, num_pos, k*N + 1]
    const uint64_t* __restrict__ glwe_in,        // [batch, num_glwe, k+1, N]
    const uint32_t* __restrict__ positions,      // [num_pos] positions to extract
    uint32_t num_glwe,                           // Number of GLWE ciphertexts per batch
    uint32_t num_positions,                      // Number of positions to extract
    const SampleExtractParams params
) {
    const uint32_t N = params.N;
    const uint32_t k = params.k;
    const uint32_t batch_idx = blockIdx.x;
    const uint32_t glwe_idx = blockIdx.y;
    const uint32_t pos_idx = blockIdx.z;
    const uint32_t tid = threadIdx.x;
    const uint32_t tpg = blockDim.x;

    if (glwe_idx >= num_glwe || pos_idx >= num_positions) return;

    uint32_t position = positions[pos_idx];
    int32_t p = (int32_t)(position % N);

    const uint64_t* glwe = glwe_in + (batch_idx * num_glwe + glwe_idx) * (k + 1) * N;
    uint64_t* lwe = lwe_out + ((batch_idx * num_glwe + glwe_idx) * num_positions + pos_idx) * (k * N + 1);

    for (uint32_t poly_idx = 0; poly_idx < k; ++poly_idx) {
        const uint64_t* mask_poly = glwe + poly_idx * N;

        for (uint32_t i = tid; i < N; i += tpg) {
            int32_t j = p - (int32_t)i - 1;
            while (j < 0) j += 2 * N;
            while (j >= 2 * (int32_t)N) j -= 2 * N;

            bool negate = (j < (int32_t)N);
            uint32_t src_idx = negate ? (uint32_t)j : (uint32_t)(j - N);

            uint64_t val = mask_poly[src_idx];
            lwe[poly_idx * N + i] = negate ? torus_neg(val) : val;
        }
    }

    if (tid == 0) {
        lwe[k * N] = glwe[k * N + position];
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

// Sample extract at position 0
cudaError_t lux_cuda_tfhe_sample_extract_0(
    uint64_t* lwe_out,
    const uint64_t* glwe_in,
    uint32_t N,
    uint32_t k,
    uint32_t batch_size,
    cudaStream_t stream
) {
    SampleExtractParams params;
    params.N = N;
    params.k = k;
    params.n_lwe = k * N;
    params.Q = 0;  // Torus
    params.batch_size = batch_size;

    dim3 grid(batch_size);
    dim3 block(min(N, 256u));

    tfhe_sample_extract_0<<<grid, block, 0, stream>>>(lwe_out, glwe_in, params);

    return cudaGetLastError();
}

// Sample extract at arbitrary position
cudaError_t lux_cuda_tfhe_sample_extract_at(
    uint64_t* lwe_out,
    const uint64_t* glwe_in,
    uint32_t position,
    uint32_t N,
    uint32_t k,
    uint32_t batch_size,
    cudaStream_t stream
) {
    SampleExtractParams params;
    params.N = N;
    params.k = k;
    params.n_lwe = k * N;
    params.Q = 0;
    params.batch_size = batch_size;

    dim3 grid(batch_size);
    dim3 block(min(N, 256u));

    tfhe_sample_extract_at<<<grid, block, 0, stream>>>(lwe_out, glwe_in, position, params);

    return cudaGetLastError();
}

// Sample extract with prime modulus
cudaError_t lux_cuda_tfhe_sample_extract_modq(
    uint64_t* lwe_out,
    const uint64_t* glwe_in,
    uint32_t position,
    uint32_t N,
    uint32_t k,
    uint64_t Q,
    uint32_t batch_size,
    cudaStream_t stream
) {
    SampleExtractParams params;
    params.N = N;
    params.k = k;
    params.n_lwe = k * N;
    params.Q = Q;
    params.batch_size = batch_size;

    dim3 grid(batch_size);
    dim3 block(min(N, 256u));

    if (position == 0) {
        tfhe_sample_extract_0_modq<<<grid, block, 0, stream>>>(lwe_out, glwe_in, params);
    } else {
        tfhe_sample_extract_at_modq<<<grid, block, 0, stream>>>(lwe_out, glwe_in, position, params);
    }

    return cudaGetLastError();
}

// Batch sample extract at multiple positions
cudaError_t lux_cuda_tfhe_sample_extract_batch(
    uint64_t* lwe_out,
    const uint64_t* glwe_in,
    const uint32_t* positions,
    uint32_t num_positions,
    bool positions_broadcast,
    uint32_t N,
    uint32_t k,
    uint32_t batch_size,
    cudaStream_t stream
) {
    SampleExtractParams params;
    params.N = N;
    params.k = k;
    params.n_lwe = k * N;
    params.Q = 0;
    params.batch_size = batch_size;

    dim3 grid(batch_size, num_positions);
    dim3 block(min(N, 256u));

    tfhe_sample_extract_batch<<<grid, block, 0, stream>>>(
        lwe_out, glwe_in, positions, num_positions, positions_broadcast, params
    );

    return cudaGetLastError();
}

// Fused sample extract + key switch
cudaError_t lux_cuda_tfhe_sample_extract_keyswitch(
    uint64_t* lwe_out,
    const uint64_t* glwe_in,
    const uint64_t* ksk,
    uint32_t position,
    uint32_t N,
    uint32_t k,
    uint32_t n_out,
    uint32_t L,
    uint32_t Bg_bits,
    uint32_t batch_size,
    cudaStream_t stream
) {
    SampleExtractParams se_params;
    se_params.N = N;
    se_params.k = k;
    se_params.n_lwe = k * N;
    se_params.Q = 0;
    se_params.batch_size = batch_size;

    KeySwitchParams ks_params;
    ks_params.n_in = k * N;
    ks_params.n_out = n_out;
    ks_params.L = L;
    ks_params.Bg_bits = Bg_bits;
    ks_params.Bg = 1ULL << Bg_bits;
    ks_params.Bg_half = 1ULL << (Bg_bits - 1);
    ks_params.Bg_mask = ks_params.Bg - 1;

    dim3 grid(batch_size, (n_out + 256) / 256);
    dim3 block(256);

    tfhe_sample_extract_keyswitch<<<grid, block, 0, stream>>>(
        lwe_out, glwe_in, ksk, position, se_params, ks_params
    );

    return cudaGetLastError();
}

// Multi-value sample extract
cudaError_t lux_cuda_tfhe_sample_extract_multi(
    uint64_t* lwe_out,
    const uint64_t* glwe_in,
    const uint32_t* positions,
    uint32_t num_glwe,
    uint32_t num_positions,
    uint32_t N,
    uint32_t k,
    uint32_t batch_size,
    cudaStream_t stream
) {
    SampleExtractParams params;
    params.N = N;
    params.k = k;
    params.n_lwe = k * N;
    params.Q = 0;
    params.batch_size = batch_size;

    dim3 grid(batch_size, num_glwe, num_positions);
    dim3 block(min(N, 256u));

    tfhe_sample_extract_multi<<<grid, block, 0, stream>>>(
        lwe_out, glwe_in, positions, num_glwe, num_positions, params
    );

    return cudaGetLastError();
}

}  // extern "C"
