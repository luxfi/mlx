// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// TFHE LWE Key Switching Kernel
// Converts LWE ciphertext encrypted under one key to encryption under another key
// Optimized for Apple Silicon GPUs (M1/M2/M3/M4)
//
// Key switching is used after sample extraction to convert from large-dimension
// LWE (dimension N) back to small-dimension LWE (dimension n).

#include <metal_stdlib>
#include "tfhe_uint128.h"

using namespace metal;

// ============================================================================
// Key Switch Parameters (passed via buffer, no compile-time defaults)
// ============================================================================

struct KeySwitchParams {
    uint N_in;         // Input LWE dimension (typically N from GLWE, e.g., 1024)
    uint N_out;        // Output LWE dimension (typically n, e.g., 630)
    uint ks_l;         // Key switching decomposition levels
    uint ks_base_log;  // Bg = 2^ks_base_log
    uint batch;        // Batch size
    ulong Q;           // Modulus
    ulong mu;          // Barrett parameter
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
// Signed Gadget Decomposition for Key Switching
// ============================================================================

// Key switching uses its own decomposition parameters (typically smaller base)
METAL_FUNC long ks_signed_decomp_digit(ulong val, uint level, uint base_log) {
    uint Bg = 1u << base_log;
    uint half_Bg = Bg >> 1;
    uint mask = Bg - 1;
    
    // Extract digit from MSB position
    uint shift = 64u - (level + 1u) * base_log;
    ulong digit = (val >> shift) & mask;
    
    // Signed representation
    if (digit >= half_Bg) {
        return long(digit) - long(Bg);
    }
    return long(digit);
}

// ============================================================================
// LWE Key Switching (Basic)
// ============================================================================

// Key switching key structure:
// KSK[i][level] is an LWE encryption of (s_in[i] * Bg^(l-level-1) * 2^(64-l*base_log))
// where s_in is the input secret key
//
// Layout: ksk[N_in][ks_l][N_out + 1]
//   For each input coefficient i: ks_l LWE ciphertexts of dimension N_out

// Standard key switching kernel
// Input:  LWE(s_in) of dimension N_in
// Output: LWE(s_out) of dimension N_out
kernel void keyswitch_lwe(
    device const ulong* lwe_in_a [[buffer(0)]],    // Input LWE mask [batch][N_in]
    device const ulong* lwe_in_b [[buffer(1)]],    // Input LWE body [batch]
    device const ulong* ksk [[buffer(2)]],         // Key switching key [N_in][ks_l][N_out+1]
    device ulong* lwe_out_a [[buffer(3)]],         // Output LWE mask [batch][N_out]
    device ulong* lwe_out_b [[buffer(4)]],         // Output LWE body [batch]
    constant KeySwitchParams& params [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid_vec [[thread_position_in_threadgroup]],
    uint3 tg_size_vec [[threads_per_threadgroup]],
    threadgroup ulong* shared [[threadgroup(0)]]
) {
    uint coeff_idx = gid.x;  // Output coefficient index [0, N_out)
    uint batch_idx = gid.y;
    
    uint N_in = params.N_in;
    uint N_out = params.N_out;
    uint ks_l = params.ks_l;
    uint ks_base_log = params.ks_base_log;
    ulong Q = params.Q;
    
    if (coeff_idx >= N_out || batch_idx >= params.batch) return;
    
    // Compute the output coefficient by summing contributions from all input coefficients
    // lwe_out[j] = sum_{i=0}^{N_in-1} sum_{level=0}^{ks_l-1} decomp[i][level] * ksk[i][level][j]
    
    ulong acc = 0;
    uint ksk_stride = ks_l * (N_out + 1);  // Stride for each input coefficient
    
    for (uint i = 0; i < N_in; i++) {
        // Get input LWE coefficient to decompose
        ulong a_i = lwe_in_a[batch_idx * N_in + i];
        
        for (uint level = 0; level < ks_l; level++) {
            // Signed decomposition of a_i
            long digit = ks_signed_decomp_digit(a_i, level, ks_base_log);
            
            if (digit == 0) continue;
            
            // KSK coefficient: ksk[i][level][coeff_idx]
            uint ksk_offset = i * ksk_stride + level * (N_out + 1) + coeff_idx;
            ulong ksk_coeff = ksk[ksk_offset];
            
            // Accumulate: acc += digit * ksk_coeff
            if (digit > 0) {
                acc = mod_add(acc, mod_mul(ulong(digit), ksk_coeff, Q), Q);
            } else {
                acc = mod_sub(acc, mod_mul(ulong(-digit), ksk_coeff, Q), Q);
            }
        }
    }
    
    // Output mask coefficient: negate the accumulated value
    // (because we're subtracting from the original)
    lwe_out_a[batch_idx * N_out + coeff_idx] = mod_neg(acc, Q);
}

// Kernel for computing the body of key-switched LWE
kernel void keyswitch_lwe_body(
    device const ulong* lwe_in_a [[buffer(0)]],
    device const ulong* lwe_in_b [[buffer(1)]],
    device const ulong* ksk [[buffer(2)]],
    device ulong* lwe_out_b [[buffer(3)]],
    constant KeySwitchParams& params [[buffer(4)]],
    uint3 gid_vec [[thread_position_in_grid]],
    uint3 tid_vec [[thread_position_in_threadgroup]],
    uint3 tg_size_vec [[threads_per_threadgroup]],
    threadgroup ulong* shared [[threadgroup(0)]]
) {
    uint batch_idx = gid_vec.x;
    
    uint N_in = params.N_in;
    uint N_out = params.N_out;
    uint ks_l = params.ks_l;
    uint ks_base_log = params.ks_base_log;
    ulong Q = params.Q;
    
    if (batch_idx >= params.batch) return;
    
    // Start with input body
    ulong acc = lwe_in_b[batch_idx];
    
    uint ksk_stride = ks_l * (N_out + 1);
    uint body_offset = N_out;  // Body is at index N_out in each KSK LWE
    
    // Subtract contributions from key switching
    for (uint i = 0; i < N_in; i++) {
        ulong a_i = lwe_in_a[batch_idx * N_in + i];
        
        for (uint level = 0; level < ks_l; level++) {
            long digit = ks_signed_decomp_digit(a_i, level, ks_base_log);
            
            if (digit == 0) continue;
            
            // KSK body: ksk[i][level][N_out] (the body is at position N_out)
            uint ksk_offset = i * ksk_stride + level * (N_out + 1) + body_offset;
            ulong ksk_body = ksk[ksk_offset];
            
            if (digit > 0) {
                acc = mod_sub(acc, mod_mul(ulong(digit), ksk_body, Q), Q);
            } else {
                acc = mod_add(acc, mod_mul(ulong(-digit), ksk_body, Q), Q);
            }
        }
    }
    
    lwe_out_b[batch_idx] = acc;
}

// ============================================================================
// Fused Key Switching (Single Kernel)
// ============================================================================

// Computes both mask and body in one kernel launch
// Uses shared memory to cache decomposed digits
kernel void keyswitch_lwe_fused(
    device const ulong* lwe_in_a [[buffer(0)]],
    device const ulong* lwe_in_b [[buffer(1)]],
    device const ulong* ksk [[buffer(2)]],
    device ulong* lwe_out [[buffer(3)]],  // [batch][N_out + 1] (mask + body)
    constant KeySwitchParams& params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid_vec [[thread_position_in_threadgroup]],
    uint3 tg_size_vec [[threads_per_threadgroup]],
    threadgroup long* decomp_cache [[threadgroup(0)]]
) {
    uint tid = tid_vec.x;
    uint tg_size = tg_size_vec.x;
    uint output_idx = gid.x;  // [0, N_out] where N_out is the body
    uint batch_idx = gid.y;

    uint N_in = params.N_in;
    uint N_out = params.N_out;
    uint ks_l = params.ks_l;
    uint ks_base_log = params.ks_base_log;
    ulong Q = params.Q;
    
    if (output_idx > N_out || batch_idx >= params.batch) return;
    
    // Cache decomposition for all input coefficients
    // Layout: decomp_cache[i * ks_l + level]
    for (uint i = tid; i < N_in; i += tg_size) {
        ulong a_i = lwe_in_a[batch_idx * N_in + i];
        for (uint level = 0; level < ks_l; level++) {
            decomp_cache[i * ks_l + level] = ks_signed_decomp_digit(a_i, level, ks_base_log);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    ulong acc;
    bool is_body = (output_idx == N_out);
    
    if (is_body) {
        // Computing body: start with input body
        acc = lwe_in_b[batch_idx];
    } else {
        // Computing mask coefficient: start with zero
        acc = 0;
    }
    
    uint ksk_stride = ks_l * (N_out + 1);
    
    for (uint i = 0; i < N_in; i++) {
        for (uint level = 0; level < ks_l; level++) {
            long digit = decomp_cache[i * ks_l + level];
            
            if (digit == 0) continue;
            
            uint ksk_offset = i * ksk_stride + level * (N_out + 1) + output_idx;
            ulong ksk_coeff = ksk[ksk_offset];
            
            if (is_body) {
                // Body: subtract contribution
                if (digit > 0) {
                    acc = mod_sub(acc, mod_mul(ulong(digit), ksk_coeff, Q), Q);
                } else {
                    acc = mod_add(acc, mod_mul(ulong(-digit), ksk_coeff, Q), Q);
                }
            } else {
                // Mask: accumulate (will negate at the end)
                if (digit > 0) {
                    acc = mod_add(acc, mod_mul(ulong(digit), ksk_coeff, Q), Q);
                } else {
                    acc = mod_sub(acc, mod_mul(ulong(-digit), ksk_coeff, Q), Q);
                }
            }
        }
    }
    
    // Negate mask coefficients
    if (!is_body) {
        acc = mod_neg(acc, Q);
    }
    
    lwe_out[batch_idx * (N_out + 1) + output_idx] = acc;
}

// ============================================================================
// Parallel Key Switching (Per Input Coefficient)
// ============================================================================

// Parallelizes over input coefficients with reduction
// Better for large N_in with sufficient parallelism
// Note: Uses threadgroup reduction instead of atomics (not supported for ulong in Metal)
kernel void keyswitch_parallel_input(
    device const ulong* lwe_in_a [[buffer(0)]],
    device const ulong* lwe_in_b [[buffer(1)]],
    device const ulong* ksk [[buffer(2)]],
    device ulong* lwe_out [[buffer(3)]],  // Output accumulator
    constant KeySwitchParams& params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid_vec [[thread_position_in_threadgroup]],
    uint3 tg_size_vec [[threads_per_threadgroup]],
    threadgroup ulong* partial_sums [[threadgroup(0)]]
) {
    uint input_idx = gid.x;   // Input coefficient being processed
    uint output_idx = gid.y;  // Output coefficient being written
    uint batch_idx = gid.z;
    uint tid = tid_vec.x;
    uint tg_size = tg_size_vec.x;

    uint N_in = params.N_in;
    uint N_out = params.N_out;
    uint ks_l = params.ks_l;
    uint ks_base_log = params.ks_base_log;
    ulong Q = params.Q;

    ulong contrib = 0;

    if (input_idx < N_in && output_idx <= N_out && batch_idx < params.batch) {
        ulong a_i = lwe_in_a[batch_idx * N_in + input_idx];
        uint ksk_stride = ks_l * (N_out + 1);

        for (uint level = 0; level < ks_l; level++) {
            long digit = ks_signed_decomp_digit(a_i, level, ks_base_log);

            if (digit == 0) continue;

            uint ksk_offset = input_idx * ksk_stride + level * (N_out + 1) + output_idx;
            ulong ksk_coeff = ksk[ksk_offset];

            if (digit > 0) {
                contrib = mod_add(contrib, mod_mul(ulong(digit), ksk_coeff, Q), Q);
            } else {
                contrib = mod_sub(contrib, mod_mul(ulong(-digit), ksk_coeff, Q), Q);
            }
        }
    }

    // Store partial sum in threadgroup memory
    partial_sums[tid] = contrib;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction within threadgroup
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_sums[tid] = mod_add(partial_sums[tid], partial_sums[tid + stride], Q);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // First thread writes result
    if (tid == 0 && output_idx <= N_out && batch_idx < params.batch) {
        // Add to existing output (must be initialized to 0 before kernel)
        ulong old_val = lwe_out[batch_idx * (N_out + 1) + output_idx];
        lwe_out[batch_idx * (N_out + 1) + output_idx] = mod_add(old_val, partial_sums[0], Q);
    }
}

// ============================================================================
// Key Switching with Modulus Switching (Combined)
// ============================================================================

// Often key switching is combined with modulus switching
// This kernel performs both operations
kernel void keyswitch_modswitch(
    device const ulong* lwe_in_a [[buffer(0)]],
    device const ulong* lwe_in_b [[buffer(1)]],
    device const ulong* ksk [[buffer(2)]],
    device uint* lwe_out [[buffer(3)]],  // Smaller modulus output (32-bit)
    constant KeySwitchParams& params [[buffer(4)]],
    constant ulong& Q_out [[buffer(5)]],  // Output modulus (smaller)
    uint3 gid [[thread_position_in_grid]],
    uint3 tid_vec [[thread_position_in_threadgroup]],
    uint3 tg_size_vec [[threads_per_threadgroup]]
) {
    uint output_idx = gid.x;
    uint batch_idx = gid.y;
    
    uint N_in = params.N_in;
    uint N_out = params.N_out;
    uint ks_l = params.ks_l;
    uint ks_base_log = params.ks_base_log;
    ulong Q = params.Q;
    
    if (output_idx > N_out || batch_idx >= params.batch) return;
    
    bool is_body = (output_idx == N_out);
    ulong acc = is_body ? lwe_in_b[batch_idx] : 0;
    
    uint ksk_stride = ks_l * (N_out + 1);
    
    for (uint i = 0; i < N_in; i++) {
        ulong a_i = lwe_in_a[batch_idx * N_in + i];
        
        for (uint level = 0; level < ks_l; level++) {
            long digit = ks_signed_decomp_digit(a_i, level, ks_base_log);
            
            if (digit == 0) continue;
            
            uint ksk_offset = i * ksk_stride + level * (N_out + 1) + output_idx;
            ulong ksk_coeff = ksk[ksk_offset];
            
            if (is_body) {
                if (digit > 0) {
                    acc = mod_sub(acc, mod_mul(ulong(digit), ksk_coeff, Q), Q);
                } else {
                    acc = mod_add(acc, mod_mul(ulong(-digit), ksk_coeff, Q), Q);
                }
            } else {
                if (digit > 0) {
                    acc = mod_add(acc, mod_mul(ulong(digit), ksk_coeff, Q), Q);
                } else {
                    acc = mod_sub(acc, mod_mul(ulong(-digit), ksk_coeff, Q), Q);
                }
            }
        }
    }
    
    // Negate mask coefficients
    if (!is_body) {
        acc = mod_neg(acc, Q);
    }
    
    // Modulus switching: scale from Q to Q_out
    // out = round(acc * Q_out / Q)
    // Using 128-bit arithmetic for precision
    uint128 scaled = mul64x64_fast(acc, Q_out);
    // Add Q/2 for rounding
    scaled = scaled + uint128(Q >> 1);
    // Divide by Q (approximate using reciprocal multiplication)
    ulong result = mod_128_64(scaled, Q);
    
    // Store as 32-bit
    lwe_out[batch_idx * (N_out + 1) + output_idx] = uint(result);
}

// ============================================================================
// Batched Key Switching with Tiling
// ============================================================================

// Process multiple LWE ciphertexts with tiled memory access pattern
// Optimizes memory bandwidth by processing tiles of input coefficients
kernel void keyswitch_tiled(
    device const ulong* lwe_in_a [[buffer(0)]],
    device const ulong* lwe_in_b [[buffer(1)]],
    device const ulong* ksk [[buffer(2)]],
    device ulong* lwe_out [[buffer(3)]],
    constant KeySwitchParams& params [[buffer(4)]],
    constant uint& tile_size [[buffer(5)]],
    constant uint& tile_idx [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid_vec [[thread_position_in_threadgroup]],
    uint3 tg_size_vec [[threads_per_threadgroup]],
    threadgroup ulong* shared [[threadgroup(0)]]
) {
    uint tid = tid_vec.x;
    uint tg_size = tg_size_vec.x;
    uint output_idx = gid.x;
    uint batch_idx = gid.y;

    uint N_in = params.N_in;
    uint N_out = params.N_out;
    uint ks_l = params.ks_l;
    uint ks_base_log = params.ks_base_log;
    ulong Q = params.Q;
    
    if (output_idx > N_out || batch_idx >= params.batch) return;
    
    // Tile boundaries
    uint tile_start = tile_idx * tile_size;
    uint tile_end = min(tile_start + tile_size, N_in);
    
    // Cache input coefficients for this tile
    threadgroup ulong* input_cache = shared;
    for (uint i = tid; i < tile_end - tile_start; i += tg_size) {
        input_cache[i] = lwe_in_a[batch_idx * N_in + tile_start + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Load or initialize accumulator
    ulong acc;
    if (tile_idx == 0) {
        acc = (output_idx == N_out) ? lwe_in_b[batch_idx] : 0;
    } else {
        acc = lwe_out[batch_idx * (N_out + 1) + output_idx];
    }
    
    bool is_body = (output_idx == N_out);
    uint ksk_stride = ks_l * (N_out + 1);
    
    // Process tile
    for (uint i = 0; i < tile_end - tile_start; i++) {
        ulong a_i = input_cache[i];
        uint global_i = tile_start + i;
        
        for (uint level = 0; level < ks_l; level++) {
            long digit = ks_signed_decomp_digit(a_i, level, ks_base_log);
            
            if (digit == 0) continue;
            
            uint ksk_offset = global_i * ksk_stride + level * (N_out + 1) + output_idx;
            ulong ksk_coeff = ksk[ksk_offset];
            
            if (is_body) {
                if (digit > 0) {
                    acc = mod_sub(acc, mod_mul(ulong(digit), ksk_coeff, Q), Q);
                } else {
                    acc = mod_add(acc, mod_mul(ulong(-digit), ksk_coeff, Q), Q);
                }
            } else {
                if (digit > 0) {
                    acc = mod_add(acc, mod_mul(ulong(digit), ksk_coeff, Q), Q);
                } else {
                    acc = mod_sub(acc, mod_mul(ulong(-digit), ksk_coeff, Q), Q);
                }
            }
        }
    }
    
    // Store intermediate or final result
    // Final tile: negate mask coefficients
    if (tile_end == N_in && !is_body) {
        acc = mod_neg(acc, Q);
    }
    
    lwe_out[batch_idx * (N_out + 1) + output_idx] = acc;
}
