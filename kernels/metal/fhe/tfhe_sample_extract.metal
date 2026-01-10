// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// TFHE Sample Extraction: Extract LWE from GLWE
// After blind rotation, extract an LWE sample from the GLWE accumulator
// Optimized for Apple Silicon GPUs (M1/M2/M3/M4)
//
// Sample extraction is the final step of programmable bootstrapping,
// converting the rotated GLWE back to an LWE ciphertext.

#include <metal_stdlib>
#include "tfhe_uint128.h"

using namespace metal;

// ============================================================================
// Sample Extract Parameters
// ============================================================================

struct SampleExtractParams {
    uint N;           // GLWE polynomial degree (also output LWE dimension)
    uint k;           // GLWE dimension
    uint batch;       // Batch size
    uint coeff_idx;   // Which coefficient to extract (usually 0)
    ulong Q;          // Modulus
};

// ============================================================================
// 64-bit Modular Arithmetic
// ============================================================================

METAL_FUNC ulong mod_neg(ulong a, ulong Q) {
    return (a == 0) ? 0 : Q - a;
}

METAL_FUNC ulong mod_add(ulong a, ulong b, ulong Q) {
    ulong sum = a + b;
    return (sum >= Q || sum < a) ? sum - Q : sum;
}

METAL_FUNC ulong mod_sub(ulong a, ulong b, ulong Q) {
    return (a >= b) ? a - b : a + Q - b;
}

// ============================================================================
// Sample Extraction Theory
// ============================================================================
//
// Given GLWE ciphertext (a_0(X), ..., a_{k-1}(X), b(X)) in R_Q^{k+1}
// where R_Q = Z_Q[X]/(X^N + 1)
//
// We want to extract the constant term (coefficient 0) as an LWE sample.
//
// The GLWE decryption is:
//   m(X) = b(X) - sum_{i=0}^{k-1} a_i(X) * s_i(X)
//
// For coefficient j, this becomes:
//   m_j = b_j - sum_{i=0}^{k-1} sum_{l=0}^{N-1} a_i[l] * s_i[(j-l) mod N] * sign(j,l)
//
// where sign(j,l) = -1 if l > j (wraparound in X^N + 1), else 1
//
// Rearranging for LWE format (â, b̂) where â is length N*k:
//   For extracted LWE coefficient at position (i*N + l):
//     â[i*N + l] = -a_i[N-l] if l > 0, else a_i[0]
//     (with appropriate negation for the X^N + 1 structure)

// ============================================================================
// Sample Extract: Extract LWE Mask from GLWE
// ============================================================================

// Extract LWE mask coefficients from GLWE ciphertext
// Output: LWE mask of dimension N * k
kernel void sample_extract_mask(
    device const ulong* glwe [[buffer(0)]],      // GLWE ciphertext [(k+1)][N]
    device ulong* lwe_a [[buffer(1)]],           // Output LWE mask [N*k]
    constant SampleExtractParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid_vec [[thread_position_in_threadgroup]]
) {
    uint output_idx = gid.x;  // Index in output LWE mask [0, N*k)
    uint batch_idx = gid.y;
    
    uint N = params.N;
    uint k = params.k;
    uint extract_coeff = params.coeff_idx;  // Which coefficient to extract (usually 0)
    ulong Q = params.Q;
    
    if (output_idx >= N * k || batch_idx >= params.batch) return;
    
    // Determine which GLWE polynomial and coefficient this corresponds to
    uint poly_idx = output_idx / N;  // Which mask polynomial (0 to k-1)
    uint coeff_idx = output_idx % N; // Coefficient index within polynomial
    
    uint glwe_offset = batch_idx * (k + 1) * N;
    
    // For extracting coefficient 0:
    // LWE[i*N + j] = -GLWE_mask_i[N - j] for j > 0
    // LWE[i*N + 0] = GLWE_mask_i[0]
    //
    // This follows from the structure of polynomial multiplication mod (X^N + 1):
    // When we multiply a_i(X) * s_i(X) and look at coefficient 0,
    // we get sum_l a_i[l] * s_i[-l mod N] with appropriate signs.
    
    ulong val;
    if (coeff_idx == 0) {
        // Coefficient 0: direct copy
        val = glwe[glwe_offset + poly_idx * N];
    } else {
        // Coefficients 1 to N-1: reversed and negated
        // a[N - coeff_idx] with sign flip due to X^N = -1
        val = glwe[glwe_offset + poly_idx * N + (N - coeff_idx)];
        val = mod_neg(val, Q);
    }
    
    lwe_a[batch_idx * N * k + output_idx] = val;
}

// Extract LWE body from GLWE ciphertext
kernel void sample_extract_body(
    device const ulong* glwe [[buffer(0)]],      // GLWE ciphertext [(k+1)][N]
    device ulong* lwe_b [[buffer(1)]],           // Output LWE body [batch]
    constant SampleExtractParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint batch_idx = gid;
    
    uint N = params.N;
    uint k = params.k;
    uint extract_coeff = params.coeff_idx;
    
    if (batch_idx >= params.batch) return;
    
    // Body polynomial is the last one (index k)
    uint body_offset = batch_idx * (k + 1) * N + k * N;
    
    // Extract the requested coefficient (usually 0)
    lwe_b[batch_idx] = glwe[body_offset + extract_coeff];
}

// ============================================================================
// Fused Sample Extract (Mask + Body in One Kernel)
// ============================================================================

// Single kernel for full sample extraction
kernel void sample_extract_fused(
    device const ulong* glwe [[buffer(0)]],
    device ulong* lwe_out [[buffer(1)]],  // [batch][N*k + 1] (mask + body)
    constant SampleExtractParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid_vec [[thread_position_in_threadgroup]]
) {
    uint output_idx = gid.x;  // [0, N*k] where N*k is the body
    uint batch_idx = gid.y;
    
    uint N = params.N;
    uint k = params.k;
    uint extract_coeff = params.coeff_idx;
    ulong Q = params.Q;
    
    uint lwe_dim = N * k;
    
    if (output_idx > lwe_dim || batch_idx >= params.batch) return;
    
    uint glwe_offset = batch_idx * (k + 1) * N;
    uint out_stride = lwe_dim + 1;
    
    if (output_idx == lwe_dim) {
        // Body: coefficient 'extract_coeff' of body polynomial
        lwe_out[batch_idx * out_stride + output_idx] = 
            glwe[glwe_offset + k * N + extract_coeff];
    } else {
        // Mask coefficient
        uint poly_idx = output_idx / N;
        uint coeff_idx = output_idx % N;
        
        ulong val;
        if (coeff_idx == 0) {
            val = glwe[glwe_offset + poly_idx * N];
        } else {
            val = glwe[glwe_offset + poly_idx * N + (N - coeff_idx)];
            val = mod_neg(val, Q);
        }
        
        lwe_out[batch_idx * out_stride + output_idx] = val;
    }
}

// ============================================================================
// Sample Extract for Arbitrary Coefficient
// ============================================================================

// Extract any coefficient from the GLWE (not just coefficient 0)
// Useful for multi-output bootstrapping
kernel void sample_extract_arbitrary(
    device const ulong* glwe [[buffer(0)]],
    device ulong* lwe_out [[buffer(1)]],
    constant SampleExtractParams& params [[buffer(2)]],
    constant uint& target_coeff [[buffer(3)]],  // Which coefficient to extract
    uint3 gid [[thread_position_in_grid]],
    uint3 tid_vec [[thread_position_in_threadgroup]]
) {
    uint output_idx = gid.x;
    uint batch_idx = gid.y;
    
    uint N = params.N;
    uint k = params.k;
    ulong Q = params.Q;
    uint lwe_dim = N * k;
    
    if (output_idx > lwe_dim || batch_idx >= params.batch) return;
    
    uint glwe_offset = batch_idx * (k + 1) * N;
    uint out_stride = lwe_dim + 1;
    
    if (output_idx == lwe_dim) {
        // Body
        lwe_out[batch_idx * out_stride + output_idx] = 
            glwe[glwe_offset + k * N + target_coeff];
    } else {
        // Mask
        // For coefficient j, the extraction formula changes:
        // We're looking at the contribution to coefficient j from each s_i coefficient
        
        uint poly_idx = output_idx / N;
        uint l = output_idx % N;  // This is the secret key coefficient index
        
        // a_i[l] contributes to m[j] as: a_i[l] * s_i[(j-l) mod N] * sign
        // Rearranging: for LWE mask index (poly_idx*N + l), we need
        // the GLWE coefficient that when multiplied by s[l] gives contribution to m[target_coeff]
        
        // Index in polynomial: (target_coeff - l) mod N with appropriate sign
        uint src_idx;
        bool negate;
        
        if (l <= target_coeff) {
            src_idx = target_coeff - l;
            negate = false;
        } else {
            // Wraparound: index becomes N + (target_coeff - l) = N - (l - target_coeff)
            src_idx = N - (l - target_coeff);
            negate = true;  // Due to X^N = -1
        }
        
        ulong val = glwe[glwe_offset + poly_idx * N + src_idx];
        if (negate) {
            val = mod_neg(val, Q);
        }
        
        lwe_out[batch_idx * out_stride + output_idx] = val;
    }
}

// ============================================================================
// Multi-Coefficient Extraction
// ============================================================================

// Extract multiple coefficients at once (for packed bootstrapping)
// Outputs multiple LWE ciphertexts from a single GLWE
kernel void sample_extract_multi(
    device const ulong* glwe [[buffer(0)]],
    device ulong* lwe_batch [[buffer(1)]],  // [num_extracts][N*k + 1]
    constant SampleExtractParams& params [[buffer(2)]],
    device const uint* extract_indices [[buffer(3)]],  // Which coefficients to extract
    constant uint& num_extracts [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid_vec [[thread_position_in_threadgroup]]
) {
    uint output_idx = gid.x;
    uint extract_idx = gid.y;  // Which extraction we're doing
    uint batch_idx = gid.z;
    
    uint N = params.N;
    uint k = params.k;
    ulong Q = params.Q;
    uint lwe_dim = N * k;
    
    if (output_idx > lwe_dim || extract_idx >= num_extracts || batch_idx >= params.batch) return;
    
    uint target_coeff = extract_indices[extract_idx];
    uint glwe_offset = batch_idx * (k + 1) * N;
    uint out_stride = lwe_dim + 1;
    uint out_offset = (batch_idx * num_extracts + extract_idx) * out_stride;
    
    if (output_idx == lwe_dim) {
        // Body
        lwe_batch[out_offset + output_idx] = glwe[glwe_offset + k * N + target_coeff];
    } else {
        uint poly_idx = output_idx / N;
        uint l = output_idx % N;
        
        uint src_idx;
        bool negate;
        
        if (l <= target_coeff) {
            src_idx = target_coeff - l;
            negate = false;
        } else {
            src_idx = N - (l - target_coeff);
            negate = true;
        }
        
        ulong val = glwe[glwe_offset + poly_idx * N + src_idx];
        if (negate) {
            val = mod_neg(val, Q);
        }
        
        lwe_batch[out_offset + output_idx] = val;
    }
}

// ============================================================================
// Sample Extract with Immediate Key Switch
// ============================================================================

// Combines sample extraction with key switching in one pass
// Avoids materializing the large-dimension LWE
kernel void sample_extract_and_keyswitch(
    device const ulong* glwe [[buffer(0)]],            // Input GLWE [(k+1)][N]
    device const ulong* ksk [[buffer(1)]],             // Key switching key
    device ulong* lwe_out [[buffer(2)]],               // Output small LWE [n_out + 1]
    constant SampleExtractParams& params [[buffer(3)]],
    constant uint& n_out [[buffer(4)]],                // Output LWE dimension
    constant uint& ks_l [[buffer(5)]],                 // KS decomposition levels
    constant uint& ks_base_log [[buffer(6)]],          // KS base log
    uint3 gid [[thread_position_in_grid]],
    uint3 tid_vec [[thread_position_in_threadgroup]],
    uint3 tg_size_vec [[threads_per_threadgroup]],
    threadgroup ulong* shared [[threadgroup(0)]]
) {
    uint output_idx = gid.x;  // [0, n_out]
    uint batch_idx = gid.y;
    
    uint N = params.N;
    uint k = params.k;
    ulong Q = params.Q;
    uint n_large = N * k;  // Large LWE dimension after extraction
    
    if (output_idx > n_out || batch_idx >= params.batch) return;
    
    uint glwe_offset = batch_idx * (k + 1) * N;
    bool is_body = (output_idx == n_out);
    
    // For efficiency, we compute the key-switched output directly
    // without explicitly forming the large LWE
    
    ulong acc = is_body ? glwe[glwe_offset + k * N] : 0;  // Start with body if computing body
    
    uint ksk_stride = ks_l * (n_out + 1);
    uint Bg = 1u << ks_base_log;
    uint half_Bg = Bg >> 1;
    uint mask = Bg - 1;
    
    // Iterate over all large LWE coefficients (N*k of them)
    for (uint large_idx = 0; large_idx < n_large; large_idx++) {
        // Compute extracted LWE coefficient on the fly
        uint poly_idx = large_idx / N;
        uint coeff_idx = large_idx % N;
        
        ulong extracted;
        if (coeff_idx == 0) {
            extracted = glwe[glwe_offset + poly_idx * N];
        } else {
            extracted = glwe[glwe_offset + poly_idx * N + (N - coeff_idx)];
            extracted = mod_neg(extracted, Q);
        }
        
        // Apply key switching decomposition
        for (uint level = 0; level < ks_l; level++) {
            uint shift = 64u - (level + 1u) * ks_base_log;
            ulong digit_u = (extracted >> shift) & mask;
            long digit;
            
            if (digit_u >= half_Bg) {
                digit = long(digit_u) - long(Bg);
            } else {
                digit = long(digit_u);
            }
            
            if (digit == 0) continue;
            
            uint ksk_offset = large_idx * ksk_stride + level * (n_out + 1) + output_idx;
            ulong ksk_coeff = ksk[ksk_offset];
            
            uint128 prod = mul64x64_fast(ulong(digit > 0 ? digit : -digit), ksk_coeff);
            ulong prod_mod = mod_128_64(prod, Q);
            
            if (is_body) {
                if (digit > 0) {
                    acc = mod_sub(acc, prod_mod, Q);
                } else {
                    acc = mod_add(acc, prod_mod, Q);
                }
            } else {
                if (digit > 0) {
                    acc = mod_add(acc, prod_mod, Q);
                } else {
                    acc = mod_sub(acc, prod_mod, Q);
                }
            }
        }
    }
    
    // Negate mask at the end
    if (!is_body) {
        acc = mod_neg(acc, Q);
    }
    
    lwe_out[batch_idx * (n_out + 1) + output_idx] = acc;
}

// ============================================================================
// Vectorized Sample Extract (SIMD Optimization)
// ============================================================================

// Process 4 coefficients at a time using Metal SIMD operations
kernel void sample_extract_simd(
    device const ulong* glwe [[buffer(0)]],
    device ulong* lwe_out [[buffer(1)]],
    constant SampleExtractParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid_vec [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint base_idx = gid.x * 4;  // Process 4 at a time
    uint batch_idx = gid.y;
    
    uint N = params.N;
    uint k = params.k;
    ulong Q = params.Q;
    uint lwe_dim = N * k;
    
    if (batch_idx >= params.batch) return;
    
    uint glwe_offset = batch_idx * (k + 1) * N;
    uint out_stride = lwe_dim + 1;
    
    // Process up to 4 coefficients
    for (uint i = 0; i < 4 && base_idx + i <= lwe_dim; i++) {
        uint output_idx = base_idx + i;
        
        ulong val;
        if (output_idx == lwe_dim) {
            val = glwe[glwe_offset + k * N];
        } else {
            uint poly_idx = output_idx / N;
            uint coeff_idx = output_idx % N;
            
            if (coeff_idx == 0) {
                val = glwe[glwe_offset + poly_idx * N];
            } else {
                val = glwe[glwe_offset + poly_idx * N + (N - coeff_idx)];
                val = mod_neg(val, Q);
            }
        }
        
        lwe_out[batch_idx * out_stride + output_idx] = val;
    }
}
