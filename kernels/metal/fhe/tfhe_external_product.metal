// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// TFHE External Product: GGSW × GLWE → GLWE
// Core operation for blind rotation in programmable bootstrapping
// Optimized for Apple Silicon GPUs (M1/M2/M3/M4)
//
// The external product computes the encryption of (msg_GGSW * msg_GLWE)
// where msg_GGSW is typically a bit of the secret key.

#include <metal_stdlib>
#include "tfhe_uint128.h"

using namespace metal;

// ============================================================================
// External Product Parameters (passed via buffer, no compile-time defaults)
// ============================================================================

struct ExternalProductParams {
    uint N;           // Polynomial degree
    uint k;           // GLWE dimension (typically 1)
    uint l;           // Decomposition levels
    uint base_log;    // Bg = 2^base_log
    uint batch;       // Batch size
    ulong Q;          // Modulus
    ulong mu;         // Barrett reduction parameter
    uint ntt_mode;    // 0 = coefficient domain, 1 = NTT domain
};

// ============================================================================
// 64-bit Modular Arithmetic
// ============================================================================

METAL_FUNC ulong mod_mul(ulong a, ulong b, ulong Q) {
    uint128 prod = mul64x64_fast(a, b);
    return mod_128_64(prod, Q);
}

METAL_FUNC ulong mod_add(ulong a, ulong b, ulong Q) {
    ulong sum = a + b;
    return (sum >= Q || sum < a) ? sum - Q : sum;
}

METAL_FUNC ulong mod_sub(ulong a, ulong b, ulong Q) {
    return (a >= b) ? a - b : a + Q - b;
}

METAL_FUNC ulong mod_neg(ulong a, ulong Q) {
    return (a == 0) ? 0 : Q - a;
}

// ============================================================================
// Signed Gadget Decomposition
// ============================================================================

// Signed decomposition with carry propagation for minimal noise
// Decomposes x into l digits where each digit is in [-Bg/2, Bg/2)
struct DecompResult {
    long digits[8];  // Max 8 levels
};

METAL_FUNC DecompResult signed_decompose_full(ulong val, uint l, uint base_log) {
    DecompResult result;
    uint Bg = 1u << base_log;
    uint half_Bg = Bg >> 1;
    uint mask = Bg - 1;
    
    // Decompose from MSB to LSB with carry propagation
    ulong carry = 0;
    
    for (uint level = 0; level < l; level++) {
        uint shift = 64u - (level + 1u) * base_log;
        ulong digit = ((val >> shift) + carry) & mask;
        carry = 0;
        
        // Signed representation
        if (digit >= half_Bg) {
            result.digits[level] = long(digit) - long(Bg);
            carry = 1;  // Propagate carry to next level
        } else {
            result.digits[level] = long(digit);
        }
    }
    
    // Pad remaining levels with zeros
    for (uint level = l; level < 8; level++) {
        result.digits[level] = 0;
    }
    
    return result;
}

// Fast single-digit extraction without full decomposition
METAL_FUNC long signed_decomp_digit(ulong val, uint level, uint base_log) {
    uint Bg = 1u << base_log;
    uint half_Bg = Bg >> 1;
    uint mask = Bg - 1;
    
    uint shift = 64u - (level + 1u) * base_log;
    ulong digit = (val >> shift) & mask;
    
    if (digit >= half_Bg) {
        return long(digit) - long(Bg);
    }
    return long(digit);
}

// ============================================================================
// External Product in Coefficient Domain (Schoolbook)
// ============================================================================

// External product: result = GGSW × GLWE
// GGSW structure: (k+1) rows × l levels × (k+1) GLWE polynomials × N coefficients
// GLWE structure: (k+1) polynomials × N coefficients
//
// Algorithm:
// 1. Decompose each GLWE polynomial into l signed polynomials
// 2. For each output polynomial j:
//    result[j] = Σ_{i=0}^{k} Σ_{level=0}^{l-1} decomp[i][level] × GGSW[i][level][j]
kernel void external_product_coefficient(
    device const ulong* glwe_in [[buffer(0)]],     // Input GLWE [(k+1)][N]
    device const ulong* ggsw [[buffer(1)]],        // GGSW [(k+1)][l][(k+1)][N]
    device ulong* glwe_out [[buffer(2)]],          // Output GLWE [(k+1)][N]
    constant ExternalProductParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid_vec [[thread_position_in_threadgroup]],
    uint3 tg_size_vec [[threads_per_threadgroup]],
    threadgroup ulong* shared [[threadgroup(0)]]
) {
    uint coeff_idx = gid.x;
    uint out_poly = gid.y;  // Which output polynomial we're computing
    
    uint N = params.N;
    uint k = params.k;
    uint l = params.l;
    ulong Q = params.Q;
    
    if (coeff_idx >= N || out_poly > k) return;
    
    // Accumulator for this coefficient
    ulong acc = 0;
    
    // For each input polynomial
    for (uint in_poly = 0; in_poly <= k; in_poly++) {
        // Load GLWE coefficient to decompose
        ulong glwe_coeff = glwe_in[in_poly * N + coeff_idx];
        
        // For each decomposition level
        for (uint level = 0; level < l; level++) {
            // Get signed decomposition digit
            long digit = signed_decomp_digit(glwe_coeff, level, params.base_log);
            
            if (digit == 0) continue;  // Skip zero digits
            
            // GGSW coefficient at [in_poly][level][out_poly][coeff_idx]
            // Layout: row-major with in_poly as outermost
            uint ggsw_offset = ((in_poly * l + level) * (k + 1) + out_poly) * N + coeff_idx;
            ulong ggsw_coeff = ggsw[ggsw_offset];
            
            // Note: This is coefficient-wise multiplication, NOT polynomial multiplication
            // Full polynomial multiplication requires convolution (done in NTT domain)
            // This kernel assumes NTT-domain inputs for pointwise multiplication
            
            if (digit > 0) {
                ulong prod = mod_mul(ulong(digit), ggsw_coeff, Q);
                acc = mod_add(acc, prod, Q);
            } else {
                ulong prod = mod_mul(ulong(-digit), ggsw_coeff, Q);
                acc = mod_sub(acc, prod, Q);
            }
        }
    }
    
    glwe_out[out_poly * N + coeff_idx] = acc;
}

// ============================================================================
// External Product in NTT Domain (Production)
// ============================================================================

// NTT-accelerated external product
// Assumes all inputs are already in NTT domain
// Polynomial multiplication becomes pointwise multiplication in NTT domain
kernel void external_product_ntt(
    device const ulong* glwe_in_ntt [[buffer(0)]],  // GLWE in NTT domain [(k+1)][N]
    device const ulong* ggsw_ntt [[buffer(1)]],     // GGSW in NTT domain [(k+1)][l][(k+1)][N]
    device ulong* glwe_out_ntt [[buffer(2)]],       // Output GLWE in NTT domain [(k+1)][N]
    device ulong* decomp_ntt [[buffer(3)]],         // Workspace for decomposed polynomials
    constant ExternalProductParams& params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid_vec [[thread_position_in_threadgroup]],
    uint3 tg_size_vec [[threads_per_threadgroup]],
    threadgroup ulong* shared [[threadgroup(0)]]
) {
    uint coeff_idx = gid.x;
    uint out_poly = gid.y;
    
    uint N = params.N;
    uint k = params.k;
    uint l = params.l;
    ulong Q = params.Q;
    
    if (coeff_idx >= N || out_poly > k) return;
    
    // Initialize output accumulator
    ulong acc = 0;
    
    // Process each input polynomial and decomposition level
    for (uint in_poly = 0; in_poly <= k; in_poly++) {
        ulong glwe_coeff = glwe_in_ntt[in_poly * N + coeff_idx];
        
        for (uint level = 0; level < l; level++) {
            long digit = signed_decomp_digit(glwe_coeff, level, params.base_log);
            
            if (digit == 0) continue;
            
            uint ggsw_offset = ((in_poly * l + level) * (k + 1) + out_poly) * N + coeff_idx;
            ulong ggsw_coeff = ggsw_ntt[ggsw_offset];
            
            // Pointwise multiplication (polynomial mul in NTT domain)
            if (digit > 0) {
                acc = mod_add(acc, mod_mul(ulong(digit), ggsw_coeff, Q), Q);
            } else {
                acc = mod_sub(acc, mod_mul(ulong(-digit), ggsw_coeff, Q), Q);
            }
        }
    }
    
    glwe_out_ntt[out_poly * N + coeff_idx] = acc;
}

// ============================================================================
// Fused Decomposition + External Product (Optimized)
// ============================================================================

// Single-pass fused kernel that decomposes and multiplies together
// Uses shared memory to cache GLWE coefficients across levels
kernel void external_product_fused(
    device const ulong* glwe_in [[buffer(0)]],
    device const ulong* ggsw [[buffer(1)]],
    device ulong* glwe_out [[buffer(2)]],
    constant ExternalProductParams& params [[buffer(3)]],
    uint3 gid_vec [[thread_position_in_grid]],
    uint3 tid_vec [[thread_position_in_threadgroup]],
    uint3 tg_size_vec [[threads_per_threadgroup]],
    threadgroup ulong* shared [[threadgroup(0)]]
) {
    uint gid = gid_vec.x;
    uint tid = tid_vec.x;
    uint tg_size = tg_size_vec.x;
    uint N = params.N;
    uint k = params.k;
    uint l = params.l;
    ulong Q = params.Q;
    
    uint total_coeffs = (k + 1) * N;
    if (gid >= total_coeffs) return;
    
    uint out_poly = gid / N;
    uint coeff_idx = gid % N;
    
    // Cache input GLWE in shared memory
    threadgroup ulong* glwe_cache = shared;
    
    // Collaborative load of GLWE
    for (uint i = tid; i < total_coeffs; i += tg_size) {
        glwe_cache[i] = glwe_in[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Compute external product
    ulong acc = 0;
    
    for (uint in_poly = 0; in_poly <= k; in_poly++) {
        ulong glwe_coeff = glwe_cache[in_poly * N + coeff_idx];
        
        // Full decomposition with carry propagation for this coefficient
        DecompResult decomp = signed_decompose_full(glwe_coeff, l, params.base_log);
        
        for (uint level = 0; level < l; level++) {
            long digit = decomp.digits[level];
            
            if (digit == 0) continue;
            
            uint ggsw_offset = ((in_poly * l + level) * (k + 1) + out_poly) * N + coeff_idx;
            ulong ggsw_coeff = ggsw[ggsw_offset];
            
            if (digit > 0) {
                acc = mod_add(acc, mod_mul(ulong(digit), ggsw_coeff, Q), Q);
            } else {
                acc = mod_sub(acc, mod_mul(ulong(-digit), ggsw_coeff, Q), Q);
            }
        }
    }
    
    glwe_out[out_poly * N + coeff_idx] = acc;
}

// ============================================================================
// Batched External Product (Multiple GLWE × Same GGSW)
// ============================================================================

kernel void external_product_batch(
    device const ulong* glwe_batch [[buffer(0)]],  // [batch][(k+1)][N]
    device const ulong* ggsw [[buffer(1)]],        // [(k+1)][l][(k+1)][N]
    device ulong* output_batch [[buffer(2)]],      // [batch][(k+1)][N]
    constant ExternalProductParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid_vec [[thread_position_in_threadgroup]],
    uint3 tg_size_vec [[threads_per_threadgroup]]
) {
    uint coeff_idx = gid.x;
    uint out_poly = gid.y;
    uint batch_idx = gid.z;
    
    uint N = params.N;
    uint k = params.k;
    uint l = params.l;
    uint batch = params.batch;
    ulong Q = params.Q;
    
    if (coeff_idx >= N || out_poly > k || batch_idx >= batch) return;
    
    uint glwe_batch_stride = (k + 1) * N;
    uint glwe_offset = batch_idx * glwe_batch_stride;
    
    ulong acc = 0;
    
    for (uint in_poly = 0; in_poly <= k; in_poly++) {
        ulong glwe_coeff = glwe_batch[glwe_offset + in_poly * N + coeff_idx];
        
        for (uint level = 0; level < l; level++) {
            long digit = signed_decomp_digit(glwe_coeff, level, params.base_log);
            
            if (digit == 0) continue;
            
            uint ggsw_offset = ((in_poly * l + level) * (k + 1) + out_poly) * N + coeff_idx;
            ulong ggsw_coeff = ggsw[ggsw_offset];
            
            if (digit > 0) {
                acc = mod_add(acc, mod_mul(ulong(digit), ggsw_coeff, Q), Q);
            } else {
                acc = mod_sub(acc, mod_mul(ulong(-digit), ggsw_coeff, Q), Q);
            }
        }
    }
    
    output_batch[glwe_offset + out_poly * N + coeff_idx] = acc;
}

// ============================================================================
// CMux via External Product
// ============================================================================

// CMux(GGSW(b), d1, d0) = d0 + GGSW(b) × (d1 - d0)
// When b=0: result = d0
// When b=1: result = d1
kernel void cmux(
    device const ulong* glwe_d0 [[buffer(0)]],   // GLWE for bit=0
    device const ulong* glwe_d1 [[buffer(1)]],   // GLWE for bit=1
    device const ulong* ggsw_bit [[buffer(2)]],  // GGSW(secret_bit)
    device ulong* result [[buffer(3)]],           // Output GLWE
    constant ExternalProductParams& params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid_vec [[thread_position_in_threadgroup]],
    uint3 tg_size_vec [[threads_per_threadgroup]],
    threadgroup ulong* shared [[threadgroup(0)]]
) {
    uint coeff_idx = gid.x;
    uint out_poly = gid.y;
    
    uint N = params.N;
    uint k = params.k;
    uint l = params.l;
    ulong Q = params.Q;
    
    if (coeff_idx >= N || out_poly > k) return;
    
    // Compute d1 - d0 for all polynomials (use shared memory)
    threadgroup ulong* diff = shared;
    
    // Collaborative computation of d1 - d0
    for (uint p = 0; p <= k; p++) {
        uint idx = p * N + coeff_idx;
        ulong v0 = glwe_d0[idx];
        ulong v1 = glwe_d1[idx];
        diff[idx] = mod_sub(v1, v0, Q);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // External product: GGSW(bit) × (d1 - d0)
    ulong ext_prod = 0;
    
    for (uint in_poly = 0; in_poly <= k; in_poly++) {
        ulong diff_coeff = diff[in_poly * N + coeff_idx];
        
        for (uint level = 0; level < l; level++) {
            long digit = signed_decomp_digit(diff_coeff, level, params.base_log);
            
            if (digit == 0) continue;
            
            uint ggsw_offset = ((in_poly * l + level) * (k + 1) + out_poly) * N + coeff_idx;
            ulong ggsw_coeff = ggsw_bit[ggsw_offset];
            
            if (digit > 0) {
                ext_prod = mod_add(ext_prod, mod_mul(ulong(digit), ggsw_coeff, Q), Q);
            } else {
                ext_prod = mod_sub(ext_prod, mod_mul(ulong(-digit), ggsw_coeff, Q), Q);
            }
        }
    }
    
    // result = d0 + external_product
    ulong d0_coeff = glwe_d0[out_poly * N + coeff_idx];
    result[out_poly * N + coeff_idx] = mod_add(d0_coeff, ext_prod, Q);
}

// ============================================================================
// External Product with Streaming GGSW (Memory-Efficient)
// ============================================================================

// Process external product level-by-level to reduce memory bandwidth
// Useful when GGSW is too large to cache in shared memory
kernel void external_product_streaming(
    device const ulong* glwe_in [[buffer(0)]],
    device const ulong* ggsw [[buffer(1)]],
    device ulong* glwe_out [[buffer(2)]],
    device ulong* temp [[buffer(3)]],  // Temporary accumulator
    constant ExternalProductParams& params [[buffer(4)]],
    constant uint& level [[buffer(5)]],  // Current level being processed
    uint3 gid [[thread_position_in_grid]],
    uint3 tid_vec [[thread_position_in_threadgroup]],
    uint3 tg_size_vec [[threads_per_threadgroup]]
) {
    uint coeff_idx = gid.x;
    uint out_poly = gid.y;
    
    uint N = params.N;
    uint k = params.k;
    uint l = params.l;
    ulong Q = params.Q;
    
    if (coeff_idx >= N || out_poly > k) return;
    
    // Load current accumulator (or initialize to zero for level 0)
    ulong acc;
    if (level == 0) {
        acc = 0;
    } else {
        acc = temp[out_poly * N + coeff_idx];
    }
    
    // Process this level for all input polynomials
    for (uint in_poly = 0; in_poly <= k; in_poly++) {
        ulong glwe_coeff = glwe_in[in_poly * N + coeff_idx];
        long digit = signed_decomp_digit(glwe_coeff, level, params.base_log);
        
        if (digit != 0) {
            uint ggsw_offset = ((in_poly * l + level) * (k + 1) + out_poly) * N + coeff_idx;
            ulong ggsw_coeff = ggsw[ggsw_offset];
            
            if (digit > 0) {
                acc = mod_add(acc, mod_mul(ulong(digit), ggsw_coeff, Q), Q);
            } else {
                acc = mod_sub(acc, mod_mul(ulong(-digit), ggsw_coeff, Q), Q);
            }
        }
    }
    
    // Store to temp or final output
    if (level == l - 1) {
        glwe_out[out_poly * N + coeff_idx] = acc;
    } else {
        temp[out_poly * N + coeff_idx] = acc;
    }
}
