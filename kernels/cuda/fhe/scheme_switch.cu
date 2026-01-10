// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// FHE Scheme Switching CUDA Kernels
// CKKS <-> TFHE <-> BGV conversions via key switching

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================================
// Constants
// ============================================================================

#define MAX_DECOMP_LEVELS 8
#define MAX_DIM 4096

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
uint64_t barrett_mul(uint64_t a, uint64_t b, uint64_t Q, uint64_t mu) {
    uint64_t q_approx = __umul64hi(a, mu);
    uint64_t product = a * b;
    uint64_t result = product - q_approx * Q;
    return result >= Q ? result - Q : result;
}

// ============================================================================
// Digit Decomposition for Key Switching
// ============================================================================

// Decompose value into L digits in base Bg
__device__
void digit_decompose(
    uint64_t value,
    uint64_t* digits,
    uint32_t L,
    uint64_t Bg,
    uint64_t Bg_half,
    uint64_t Q
) {
    uint64_t val = value;

    for (uint32_t l = 0; l < L; ++l) {
        uint64_t digit = val % Bg;

        // Signed decomposition: center around 0
        if (digit > Bg_half) {
            digits[l] = mod_sub(0, Bg - digit, Q);
            val = val / Bg + 1;  // Carry
        } else {
            digits[l] = digit;
            val = val / Bg;
        }
    }
}

// ============================================================================
// Key Switching Parameters
// ============================================================================

struct KeySwitchParams {
    uint64_t Q;           // Ciphertext modulus
    uint64_t mu;          // Barrett constant
    uint64_t Bg;          // Decomposition base
    uint64_t Bg_half;     // Bg / 2
    uint32_t input_dim;   // Input dimension (e.g., N for RLWE, n for LWE)
    uint32_t output_dim;  // Output dimension
    uint32_t L;           // Number of decomposition levels
    uint32_t batch_size;  // Batch size
};

// ============================================================================
// Key Switching Kernel
// ============================================================================

// Key switch: transform ciphertext under old key to new key
// Output[j] = sum_{i=0}^{input_dim-1} sum_{l=0}^{L-1} digit_l(input[i]) * KSK[i][l][j]
extern "C" __global__
void key_switch(
    uint64_t* __restrict__ output,              // [batch, 2, output_dim]
    const uint64_t* __restrict__ input,         // [batch, input_dim]
    const uint64_t* __restrict__ ksk,           // [input_dim, L, 2, output_dim]
    const KeySwitchParams params
) {
    const uint32_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t comp = blockIdx.y;      // 0 or 1
    const uint32_t batch_idx = blockIdx.z;

    if (out_idx >= params.output_dim) return;

    const uint64_t Q = params.Q;
    const uint64_t Bg = params.Bg;
    const uint64_t Bg_half = params.Bg_half;
    const uint32_t L = params.L;
    const uint32_t input_dim = params.input_dim;
    const uint32_t output_dim = params.output_dim;

    uint64_t acc = 0;

    // Accumulate over all input dimensions and decomposition levels
    for (uint32_t i = 0; i < input_dim; ++i) {
        uint64_t val = input[batch_idx * input_dim + i];

        // Decompose into L digits
        uint64_t digits[MAX_DECOMP_LEVELS];
        digit_decompose(val, digits, L, Bg, Bg_half, Q);

        // Accumulate: digit_l * KSK[i][l][comp][out_idx]
        for (uint32_t l = 0; l < L; ++l) {
            // KSK layout: [input_dim, L, 2, output_dim]
            uint32_t ksk_offset = ((i * L + l) * 2 + comp) * output_dim + out_idx;
            uint64_t ksk_val = __ldg(&ksk[ksk_offset]);

            acc = mod_add(acc, barrett_mul(digits[l], ksk_val, Q, params.mu), Q);
        }
    }

    // Write output
    output[(batch_idx * 2 + comp) * output_dim + out_idx] = acc;
}

// ============================================================================
// CKKS to TFHE: Extract bits from CKKS ciphertext
// ============================================================================

struct BitExtractParams {
    uint64_t Q;
    uint64_t delta;       // Scaling factor
    uint32_t N;           // CKKS ring dimension
    uint32_t num_bits;    // Number of bits to extract
    uint32_t batch_size;
};

extern "C" __global__
void extract_bits(
    uint64_t* __restrict__ output_bits,         // [batch, num_bits, n+1] LWE ciphertexts
    const uint64_t* __restrict__ ckks_ct,       // [batch, 2, N] CKKS ciphertext
    const uint64_t* __restrict__ extraction_key,// Key for LWE extraction
    const BitExtractParams params
) {
    const uint32_t N = params.N;
    const uint32_t num_bits = params.num_bits;
    const uint32_t batch_idx = blockIdx.z;
    const uint32_t bit_idx = blockIdx.y;
    const uint32_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (coeff_idx >= N || bit_idx >= num_bits) return;

    const uint64_t Q = params.Q;
    const uint64_t delta = params.delta;

    // Extract coefficient
    uint64_t ct0 = ckks_ct[(batch_idx * 2 + 0) * N + coeff_idx];
    uint64_t ct1 = ckks_ct[(batch_idx * 2 + 1) * N + coeff_idx];

    // Scale and extract bit
    // bit = floor((ct0 / delta) >> bit_idx) mod 2
    uint64_t scaled = ct0 / delta;
    uint64_t bit = (scaled >> bit_idx) & 1;

    // Create LWE ciphertext for this bit
    // Output: (a, b) where a comes from extraction_key, b encodes the bit
    // Simplified: just output scaled coefficient per bit position
    uint32_t out_idx = (batch_idx * num_bits + bit_idx) * (N + 1) + coeff_idx;

    if (coeff_idx == 0) {
        // First element is the "b" part with the extracted bit
        output_bits[out_idx] = bit * (Q / 4);  // Encode in [0, Q/4] or [Q/2, 3Q/4]
    } else {
        // Rest are "a" coefficients (from key)
        output_bits[out_idx] = extraction_key[coeff_idx];
    }
}

// ============================================================================
// TFHE to CKKS: Pack bits into CKKS slots
// ============================================================================

struct BitPackParams {
    uint64_t Q;
    uint64_t delta;       // Scaling factor for CKKS
    uint32_t N;           // CKKS ring dimension
    uint32_t num_bits;    // Number of bits to pack
    uint32_t batch_size;
};

extern "C" __global__
void pack_bits(
    uint64_t* __restrict__ ckks_ct,             // [batch, 2, N] output CKKS
    const uint64_t* __restrict__ tfhe_bits,     // [batch, num_bits] decrypted bits (0 or 1)
    const uint64_t* __restrict__ packing_key,   // Packing helper
    const BitPackParams params
) {
    const uint32_t N = params.N;
    const uint32_t num_bits = params.num_bits;
    const uint32_t batch_idx = blockIdx.y;
    const uint32_t coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (coeff_idx >= N) return;

    const uint64_t Q = params.Q;
    const uint64_t delta = params.delta;

    // Pack bits into polynomial coefficient
    // coefficient = sum_{i=0}^{num_bits-1} bit_i * 2^i * delta
    uint64_t packed = 0;
    for (uint32_t i = 0; i < num_bits && i < 64; ++i) {
        uint64_t bit = tfhe_bits[batch_idx * num_bits + i];
        packed = mod_add(packed, barrett_mul(bit, delta * (1ULL << i), Q, params.delta), Q);
    }

    // Write to first slot, zero elsewhere (simplified)
    ckks_ct[(batch_idx * 2 + 0) * N + coeff_idx] = (coeff_idx == 0) ? packed : 0;
    ckks_ct[(batch_idx * 2 + 1) * N + coeff_idx] = 0;  // Noiseless
}

// ============================================================================
// Modulus Switching
// ============================================================================

struct ModSwitchParams {
    uint64_t Q_from;      // Source modulus
    uint64_t Q_to;        // Target modulus
    uint64_t Q_ratio_hi;  // High bits of Q_to / Q_from (for rounding)
    uint32_t size;        // Number of elements
    uint32_t batch_size;
};

extern "C" __global__
void modulus_switch(
    uint64_t* __restrict__ output,
    const uint64_t* __restrict__ input,
    const ModSwitchParams params
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = params.size * params.batch_size;

    if (idx >= total) return;

    uint64_t val = input[idx];
    const uint64_t Q_from = params.Q_from;
    const uint64_t Q_to = params.Q_to;

    // Scale: output = round(val * Q_to / Q_from)
    // Using: result = (val * Q_to + Q_from/2) / Q_from

    uint64_t scaled_lo = val * Q_to;
    uint64_t scaled_hi = __umul64hi(val, Q_to);

    // Add Q_from/2 for rounding
    uint64_t half = Q_from / 2;
    scaled_lo += half;
    if (scaled_lo < half) scaled_hi++;

    // Divide by Q_from (simplified - assumes Q_from is power of 2 or uses Barrett)
    // For general case, would need multi-precision division
    uint64_t result = scaled_lo / Q_from;

    output[idx] = result;
}

// ============================================================================
// BGV to CKKS Scale Adjustment
// ============================================================================

extern "C" __global__
void scale_adjust_bgv_to_ckks(
    uint64_t* __restrict__ output,
    const uint64_t* __restrict__ input,
    uint64_t scale_factor,
    uint64_t Q,
    uint64_t mu,
    uint32_t size
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Multiply by scale factor to convert discrete values to scaled representation
    output[idx] = barrett_mul(input[idx], scale_factor, Q, mu);
}

// ============================================================================
// CKKS to BGV Scale Adjustment (Round to nearest integer)
// ============================================================================

extern "C" __global__
void scale_adjust_ckks_to_bgv(
    uint64_t* __restrict__ output,
    const uint64_t* __restrict__ input,
    uint64_t scale_inv,    // 1/scale_factor mod Q
    uint64_t Q,
    uint64_t mu,
    uint32_t size
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Multiply by inverse scale to round to integer
    output[idx] = barrett_mul(input[idx], scale_inv, Q, mu);
}

// ============================================================================
// Fused Scheme Switch (Key Switch + Modulus Switch)
// ============================================================================

extern "C" __global__
void batch_scheme_switch_fused(
    uint64_t* __restrict__ output,
    const uint64_t* __restrict__ input,
    const uint64_t* __restrict__ ksk,
    const KeySwitchParams ks_params,
    const ModSwitchParams ms_params
) {
    extern __shared__ uint64_t shared[];

    const uint32_t out_idx = threadIdx.x;
    const uint32_t batch_idx = blockIdx.x;
    const uint32_t input_dim = ks_params.input_dim;
    const uint32_t output_dim = ks_params.output_dim;
    const uint32_t L = ks_params.L;

    if (out_idx >= output_dim) return;

    // Phase 1: Key switch
    uint64_t acc0 = 0, acc1 = 0;

    for (uint32_t i = 0; i < input_dim; ++i) {
        uint64_t val = input[batch_idx * input_dim + i];
        uint64_t digits[MAX_DECOMP_LEVELS];
        digit_decompose(val, digits, L, ks_params.Bg, ks_params.Bg_half, ks_params.Q);

        for (uint32_t l = 0; l < L; ++l) {
            uint32_t ksk_base = (i * L + l) * 2 * output_dim;
            acc0 = mod_add(acc0, barrett_mul(digits[l], ksk[ksk_base + out_idx], ks_params.Q, ks_params.mu), ks_params.Q);
            acc1 = mod_add(acc1, barrett_mul(digits[l], ksk[ksk_base + output_dim + out_idx], ks_params.Q, ks_params.mu), ks_params.Q);
        }
    }

    // Store in shared for modulus switch
    shared[out_idx] = acc0;
    shared[output_dim + out_idx] = acc1;

    __syncthreads();

    // Phase 2: Modulus switch
    uint64_t switched0 = (shared[out_idx] * ms_params.Q_to + ms_params.Q_from / 2) / ms_params.Q_from;
    uint64_t switched1 = (shared[output_dim + out_idx] * ms_params.Q_to + ms_params.Q_from / 2) / ms_params.Q_from;

    // Write output
    output[(batch_idx * 2 + 0) * output_dim + out_idx] = switched0;
    output[(batch_idx * 2 + 1) * output_dim + out_idx] = switched1;
}
