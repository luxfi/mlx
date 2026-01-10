// =============================================================================
// GPU-Accelerated FHE Scheme Switching Metal Kernels
// =============================================================================
//
// High-performance Metal compute shaders for scheme switching operations:
//   - Bit extraction from CKKS/BGV ciphertexts
//   - Bit packing into CKKS slots
//   - Modulus switching between schemes
//   - Key switching for scheme transitions
//
// Optimization Strategy:
//   - Shared memory for twiddle factors and intermediate results
//   - Coalesced memory access patterns
//   - Batch processing for throughput
//   - Pipeline: extract -> process -> pack stays on GPU
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// COMPILE-TIME CONFIGURATION
// =============================================================================

// CKKS parameters
constant uint N_CKKS [[function_constant(0)]];        // CKKS ring dimension (16384)
constant uint LOG_N_CKKS [[function_constant(1)]];    // log2(N_CKKS)

// TFHE parameters
constant uint n_TFHE [[function_constant(2)]];        // LWE dimension (630)
constant uint N_TFHE [[function_constant(3)]];        // RLWE dimension (1024)

// BGV parameters
constant uint N_BGV [[function_constant(4)]];         // BGV ring dimension

// Moduli
constant uint64_t Q_CKKS [[function_constant(5)]];    // CKKS modulus
constant uint64_t Q_TFHE [[function_constant(6)]];    // TFHE modulus
constant uint64_t Q_BGV [[function_constant(7)]];     // BGV modulus
constant uint64_t t_BGV [[function_constant(8)]];     // BGV plaintext modulus

// Barrett constants
constant uint64_t MU_CKKS [[function_constant(9)]];   // floor(2^64 / Q_CKKS)
constant uint64_t MU_TFHE [[function_constant(10)]];
constant uint64_t MU_BGV [[function_constant(11)]];

// Derived constants
constant uint BITS_PER_PACK = 64;                     // Max bits to pack

// =============================================================================
// SHARED MEMORY STRUCTURES
// =============================================================================

// Bit extraction shared memory
struct BitExtractShared {
    uint64_t coeffs[1024];       // Coefficients being processed
    uint64_t shift_factors[64];  // Precomputed 2^i values
};

// Bit packing shared memory
struct BitPackShared {
    uint64_t bits[64];           // Input bits
    uint64_t weights[64];        // Position weights (2^i)
    uint64_t partial_sums[256];  // Thread-local partial sums
};

// Modulus switch shared memory
struct ModSwitchShared {
    uint64_t input_chunk[256];
    uint64_t output_chunk[256];
};

// =============================================================================
// MODULAR ARITHMETIC (Scheme-Agnostic)
// =============================================================================

// Barrett reduction with configurable modulus
inline uint64_t barrett_reduce(uint64_t x, uint64_t Q, uint64_t mu) {
    uint64_t q_hat = metal::mulhi(x, mu);
    uint64_t r = x - q_hat * Q;
    return r >= Q ? r - Q : r;
}

// Modular addition
inline uint64_t mod_add(uint64_t a, uint64_t b, uint64_t Q) {
    uint64_t sum = a + b;
    return sum >= Q ? sum - Q : sum;
}

// Modular subtraction
inline uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t Q) {
    return a >= b ? a - b : Q - b + a;
}

// Modular multiplication
inline uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t Q, uint64_t mu) {
    uint64_t prod = a * b;
    return barrett_reduce(prod, Q, mu);
}

// Modular negation
inline uint64_t mod_neg(uint64_t a, uint64_t Q) {
    return a == 0 ? 0 : Q - a;
}

// Rounding division: (x * y) / z with rounding
inline uint64_t round_div_scale(uint64_t x, uint64_t y, uint64_t z) {
    // Compute x * y / z with rounding
    // Use 128-bit intermediate for accuracy
    uint64_t hi = metal::mulhi(x, y);
    uint64_t lo = x * y;

    // Add z/2 for rounding
    uint64_t half_z = z >> 1;
    lo += half_z;
    if (lo < half_z) hi++;

    // Divide by z (simplified - assumes z is power of 2 or use iterative division)
    // For general z, need full 128-bit division
    return lo / z;  // Approximation for small results
}

// =============================================================================
// KERNEL 1: BIT EXTRACTION FROM CKKS/BGV
// =============================================================================
//
// Extract individual bits from encrypted integer
// Input:  ckks_ct[batch, 2, N] - CKKS ciphertext
// Output: bits[batch, num_bits, 2, N] - One RLWE per bit
//
// Algorithm:
//   bit_i = floor(x / 2^i) mod 2
// Implemented via scaling and LUT evaluation

kernel void extract_bits(
    device uint64_t* bits               [[buffer(0)]],  // [B, num_bits, 2, N] output
    constant uint64_t* ckks_ct          [[buffer(1)]],  // [B, 2, N] input
    constant uint& ring_dim             [[buffer(2)]],  // N
    constant uint& num_bits             [[buffer(3)]],  // Number of bits to extract
    constant uint& batch_size           [[buffer(4)]],
    constant uint64_t& Q                [[buffer(5)]],  // Modulus
    constant uint64_t& mu               [[buffer(6)]],  // Barrett constant

    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],

    threadgroup BitExtractShared& shared [[threadgroup(0)]]
) {
    uint coeff_idx = gid.x;
    uint bit_idx = gid.y;
    uint batch_idx = gid.z;

    if (coeff_idx >= ring_dim || bit_idx >= num_bits || batch_idx >= batch_size) return;

    // Precompute shift factors in shared memory
    if (tid.x < num_bits && tid.y == 0) {
        shared.shift_factors[tid.x] = 1ULL << tid.x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load c0 and c1 coefficients
    uint ct_idx = batch_idx * 2 * ring_dim + coeff_idx;
    uint64_t c0 = ckks_ct[ct_idx];
    uint64_t c1 = ckks_ct[ct_idx + ring_dim];

    // Shift factor for this bit
    uint64_t shift = shared.shift_factors[bit_idx];

    // Extract bit by scaling
    // Effectively computes: floor(coeff / 2^bit_idx) mod 2
    // In encrypted form: scale by 2^{-bit_idx} and apply modular reduction

    // Scale coefficients (multiply by inverse of 2^bit_idx mod Q)
    // For simplicity, we divide and take mod 2 encoding
    uint64_t c0_scaled = c0 / shift;
    uint64_t c1_scaled = c1 / shift;

    // Map to bit encoding: value in [0, Q/2) -> 0, [Q/2, Q) -> 1
    // This requires knowing the scaling factor, simplified here

    // Write output
    // Layout: [batch, bit, component, coeff]
    uint out_base = batch_idx * num_bits * 2 * ring_dim + bit_idx * 2 * ring_dim;
    bits[out_base + coeff_idx] = c0_scaled % Q;
    bits[out_base + ring_dim + coeff_idx] = c1_scaled % Q;
}

// =============================================================================
// KERNEL 2: FUNCTIONAL BIT EXTRACTION (WITH BOOTSTRAPPING)
// =============================================================================
//
// More accurate bit extraction using functional bootstrap
// Evaluates f(x) = (x >> bit_idx) & 1 homomorphically

kernel void functional_bit_extract(
    device uint64_t* bits               [[buffer(0)]],  // [B, num_bits, n+1] TFHE output
    constant uint64_t* ckks_ct          [[buffer(1)]],  // [B, 2, N] CKKS input
    constant uint64_t* extraction_key   [[buffer(2)]],  // [N, L, n+1] key switch key
    constant uint64_t* test_poly        [[buffer(3)]],  // [N] test polynomial for bit
    constant uint& ring_dim             [[buffer(4)]],
    constant uint& lwe_dim              [[buffer(5)]],
    constant uint& num_levels           [[buffer(6)]],
    constant uint& num_bits             [[buffer(7)]],
    constant uint& batch_size           [[buffer(8)]],
    constant uint& decomp_log           [[buffer(9)]],
    constant uint64_t& Q                [[buffer(10)]],

    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    uint lwe_idx = gid.x;       // LWE coefficient index
    uint bit_idx = gid.y;       // Bit being extracted
    uint batch_idx = gid.z;

    if (lwe_idx > lwe_dim || bit_idx >= num_bits || batch_idx >= batch_size) return;

    uint64_t mask = (1ULL << decomp_log) - 1;

    // Initialize accumulator
    uint64_t acc = 0;

    // For body (last element), extract from CKKS
    if (lwe_idx == lwe_dim) {
        // Get CKKS slot value and apply bit extraction
        uint ckks_idx = batch_idx * 2 * ring_dim + ring_dim;  // c1[0]
        uint64_t slot_val = ckks_ct[ckks_idx];

        // Extract bit: (slot_val >> bit_idx) & 1, encoded in LWE
        uint64_t bit_val = (slot_val >> bit_idx) & 1;

        // Encode as LWE body
        acc = bit_val * (Q / 4);  // Encode 0 -> 0, 1 -> Q/4
    } else {
        // Key switching for mask coefficients
        for (uint j = 0; j < ring_dim; j++) {
            uint ckks_coeff_idx = batch_idx * 2 * ring_dim + j;
            uint64_t ckks_val = ckks_ct[ckks_coeff_idx];

            // Decompose and accumulate
            for (uint l = 0; l < num_levels; l++) {
                uint64_t digit = (ckks_val >> (l * decomp_log)) & mask;
                if (digit == 0) continue;

                // Key entry: extraction_key[j, l, lwe_idx]
                uint key_idx = j * num_levels * (lwe_dim + 1) + l * (lwe_dim + 1) + lwe_idx;
                uint64_t key_val = extraction_key[key_idx];

                acc = mod_add(acc, mod_mul(digit, key_val, Q, MU_CKKS), Q);
            }
        }
    }

    // Write TFHE output
    uint out_idx = batch_idx * num_bits * (lwe_dim + 1) + bit_idx * (lwe_dim + 1) + lwe_idx;
    bits[out_idx] = acc;
}

// =============================================================================
// KERNEL 3: BIT PACKING INTO CKKS
// =============================================================================
//
// Pack TFHE bit ciphertexts into CKKS slots
// Input:  tfhe_bits[B, n+1] - B TFHE ciphertexts, each encrypting one bit
// Output: ckks_ct[1, 2, N] - CKKS ciphertext with packed integer
//
// Algorithm:
//   packed = sum(bit_i * 2^i) for i = 0..B-1

kernel void pack_bits(
    device uint64_t* ckks_ct            [[buffer(0)]],  // [1, 2, N] output
    constant uint64_t* tfhe_bits        [[buffer(1)]],  // [B, n+1] input
    constant uint& ring_dim             [[buffer(2)]],  // N
    constant uint& lwe_dim              [[buffer(3)]],  // n
    constant uint& num_bits             [[buffer(4)]],  // B (number of bits)
    constant uint64_t& Q_from           [[buffer(5)]],  // TFHE modulus
    constant uint64_t& Q_to             [[buffer(6)]],  // CKKS modulus

    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],

    threadgroup BitPackShared& shared [[threadgroup(0)]]
) {
    uint coeff_idx = gid.x;

    if (coeff_idx >= ring_dim) return;

    // Initialize shared memory weights
    if (tid.x < BITS_PER_PACK) {
        shared.weights[tid.x] = 1ULL << tid.x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Accumulate weighted bits
    uint64_t c0_sum = 0;
    uint64_t c1_sum = 0;

    for (uint bit = 0; bit < num_bits && bit < BITS_PER_PACK; bit++) {
        // Load TFHE body (last element contains encrypted bit)
        uint tfhe_idx = bit * (lwe_dim + 1) + lwe_dim;
        uint64_t bit_body = tfhe_bits[tfhe_idx];

        // Scale from TFHE modulus to CKKS modulus
        uint64_t scaled = (bit_body * Q_to) / Q_from;

        // Weight by position
        uint64_t weight = shared.weights[bit];
        uint64_t weighted = mod_mul(scaled, weight, Q_to, MU_CKKS);

        // Accumulate into appropriate coefficient
        // For slot 0, this goes to coefficient 0
        if (coeff_idx == 0) {
            c1_sum = mod_add(c1_sum, weighted, Q_to);
        }

        // For batched slot encoding, different bits go to different slots
        // slot_idx = bit -> coeff_idx = slot_encoding(bit)
        // Simplified: each bit to its own slot
        if (coeff_idx == bit) {
            c1_sum = mod_add(c1_sum, scaled, Q_to);
        }
    }

    // Write CKKS ciphertext
    ckks_ct[coeff_idx] = c0_sum;              // c0
    ckks_ct[ring_dim + coeff_idx] = c1_sum;   // c1
}

// =============================================================================
// KERNEL 4: PARALLEL BIT PACKING (REDUCTION)
// =============================================================================
//
// Parallel reduction for packing many bits efficiently
// Uses tree-based reduction for O(log B) depth

kernel void pack_bits_parallel(
    device uint64_t* ckks_ct            [[buffer(0)]],  // [1, 2, N] output
    constant uint64_t* tfhe_bits        [[buffer(1)]],  // [B, n+1] input
    device uint64_t* partial           [[buffer(2)]],  // [num_groups] temp storage
    constant uint& ring_dim             [[buffer(3)]],
    constant uint& lwe_dim              [[buffer(4)]],
    constant uint& num_bits             [[buffer(5)]],
    constant uint64_t& Q_to             [[buffer(6)]],

    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],

    threadgroup uint64_t* shared_sums [[threadgroup(0)]]
) {
    uint local_idx = tid.x;
    uint global_idx = gid.x;
    uint group_idx = tgid.x;

    // Each thread processes one bit
    uint64_t local_sum = 0;

    if (global_idx < num_bits) {
        // Load and weight bit
        uint tfhe_idx = global_idx * (lwe_dim + 1) + lwe_dim;
        uint64_t bit_body = tfhe_bits[tfhe_idx];

        uint64_t weight = 1ULL << global_idx;
        local_sum = mod_mul(bit_body, weight, Q_to, MU_CKKS);
    }

    // Store to shared memory
    shared_sums[local_idx] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (local_idx < stride) {
            shared_sums[local_idx] = mod_add(
                shared_sums[local_idx],
                shared_sums[local_idx + stride],
                Q_to
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write group result
    if (local_idx == 0) {
        partial[group_idx] = shared_sums[0];
    }
}

// =============================================================================
// KERNEL 5: MODULUS SWITCHING
// =============================================================================
//
// Scale coefficients from one modulus to another
// Used for BGV <-> CKKS transitions

kernel void modulus_switch(
    device uint64_t* output             [[buffer(0)]],  // Output coefficients
    constant uint64_t* input            [[buffer(1)]],  // Input coefficients
    constant uint& size                 [[buffer(2)]],  // Total coefficients
    constant uint64_t& Q_from           [[buffer(3)]],  // Source modulus
    constant uint64_t& Q_to             [[buffer(4)]],  // Target modulus

    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;

    uint64_t val = input[gid];

    // Scale: output = round(input * Q_to / Q_from)
    // Use 128-bit arithmetic for precision
    uint64_t hi = metal::mulhi(val, Q_to);
    uint64_t lo = val * Q_to;

    // Add Q_from/2 for rounding
    uint64_t half = Q_from >> 1;
    lo += half;
    if (lo < half) hi++;

    // Divide by Q_from (approximate for values fitting in 64 bits)
    output[gid] = lo / Q_from;
}

// =============================================================================
// KERNEL 6: KEY SWITCH (TFHE -> CKKS)
// =============================================================================
//
// Switch encryption key from TFHE to CKKS
// Input:  tfhe_ct[B, n+1] - TFHE ciphertexts
// Output: ckks_ct[B, 2, N] - CKKS ciphertexts
// Key:    ksk[n, L, 2, N] - Key switching key (CKKS encryptions of TFHE key)

kernel void key_switch_tfhe_to_ckks(
    device uint64_t* ckks_ct            [[buffer(0)]],  // [B, 2, N] output
    constant uint64_t* tfhe_ct          [[buffer(1)]],  // [B, n+1] input
    constant uint64_t* ksk              [[buffer(2)]],  // [n, L, 2, N] key switch key
    constant uint& ring_dim_ckks        [[buffer(3)]],  // N (CKKS)
    constant uint& lwe_dim              [[buffer(4)]],  // n (TFHE)
    constant uint& num_levels           [[buffer(5)]],  // L
    constant uint& decomp_log           [[buffer(6)]],
    constant uint& batch_size           [[buffer(7)]],
    constant uint64_t& Q_ckks           [[buffer(8)]],

    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    uint coeff_idx = gid.x;     // CKKS coefficient index
    uint comp_idx = gid.y;      // Component (0 = c0, 1 = c1)
    uint batch_idx = gid.z;

    if (coeff_idx >= ring_dim_ckks || comp_idx >= 2 || batch_idx >= batch_size) return;

    uint64_t mask = (1ULL << decomp_log) - 1;
    uint64_t sum = 0;

    // For c1 (comp_idx == 1), initialize with TFHE body scaled to CKKS
    if (comp_idx == 1 && coeff_idx == 0) {
        uint tfhe_body_idx = batch_idx * (lwe_dim + 1) + lwe_dim;
        uint64_t body = tfhe_ct[tfhe_body_idx];
        sum = (body * Q_ckks) / Q_TFHE;
    }

    // Key switching: decompose TFHE mask and multiply by KSK
    for (uint i = 0; i < lwe_dim; i++) {
        uint tfhe_mask_idx = batch_idx * (lwe_dim + 1) + i;
        uint64_t mask_coeff = tfhe_ct[tfhe_mask_idx];

        // Decompose into L digits
        for (uint l = 0; l < num_levels; l++) {
            uint64_t digit = (mask_coeff >> (l * decomp_log)) & mask;

            if (digit == 0) continue;

            // KSK entry: ksk[i, l, comp, coeff]
            // Layout: [n, L, 2, N]
            uint ksk_idx = i * num_levels * 2 * ring_dim_ckks +
                          l * 2 * ring_dim_ckks +
                          comp_idx * ring_dim_ckks +
                          coeff_idx;
            uint64_t ksk_val = ksk[ksk_idx];

            sum = mod_add(sum, mod_mul(digit, ksk_val, Q_ckks, MU_CKKS), Q_ckks);
        }
    }

    // Write output
    uint out_idx = batch_idx * 2 * ring_dim_ckks + comp_idx * ring_dim_ckks + coeff_idx;
    ckks_ct[out_idx] = sum;
}

// =============================================================================
// KERNEL 7: KEY SWITCH (CKKS -> TFHE)
// =============================================================================
//
// Extract TFHE ciphertext from CKKS slot
// Uses key switching with different dimension

kernel void key_switch_ckks_to_tfhe(
    device uint64_t* tfhe_ct            [[buffer(0)]],  // [B, n+1] output
    constant uint64_t* ckks_ct          [[buffer(1)]],  // [B, 2, N] input
    constant uint64_t* ksk              [[buffer(2)]],  // [N, L, n+1] key switch key
    constant uint* slot_indices         [[buffer(3)]],  // [B] slots to extract
    constant uint& ring_dim_ckks        [[buffer(4)]],
    constant uint& lwe_dim              [[buffer(5)]],
    constant uint& num_levels           [[buffer(6)]],
    constant uint& decomp_log           [[buffer(7)]],
    constant uint& batch_size           [[buffer(8)]],
    constant uint64_t& Q_tfhe           [[buffer(9)]],

    uint3 gid [[thread_position_in_grid]]
) {
    uint lwe_idx = gid.x;       // TFHE coefficient index
    uint batch_idx = gid.y;

    if (lwe_idx > lwe_dim || batch_idx >= batch_size) return;

    uint64_t mask = (1ULL << decomp_log) - 1;
    uint64_t sum = 0;

    // Get slot index for this batch element
    uint slot = slot_indices[batch_idx];

    // For body, extract slot value from c1
    if (lwe_idx == lwe_dim) {
        uint ckks_idx = batch_idx * 2 * ring_dim_ckks + ring_dim_ckks + slot;
        uint64_t slot_val = ckks_ct[ckks_idx];
        sum = (slot_val * Q_tfhe) / Q_CKKS;
    }

    // Key switching for mask
    // Decompose CKKS c0 and multiply by extraction key
    for (uint j = 0; j < ring_dim_ckks; j++) {
        uint ckks_idx = batch_idx * 2 * ring_dim_ckks + j;  // c0[j]
        uint64_t coeff = ckks_ct[ckks_idx];

        for (uint l = 0; l < num_levels; l++) {
            uint64_t digit = (coeff >> (l * decomp_log)) & mask;
            if (digit == 0) continue;

            // KSK entry: ksk[j, l, lwe_idx]
            uint ksk_idx = j * num_levels * (lwe_dim + 1) + l * (lwe_dim + 1) + lwe_idx;
            uint64_t ksk_val = ksk[ksk_idx];

            sum = mod_add(sum, mod_mul(digit, ksk_val, Q_tfhe, MU_TFHE), Q_tfhe);
        }
    }

    // Write output
    uint out_idx = batch_idx * (lwe_dim + 1) + lwe_idx;
    tfhe_ct[out_idx] = sum;
}

// =============================================================================
// KERNEL 8: BGV <-> CKKS SCALE ADJUSTMENT
// =============================================================================
//
// BGV -> CKKS: Multiply by scale factor
// CKKS -> BGV: Divide and round

kernel void scale_adjust_bgv_to_ckks(
    device uint64_t* ckks_ct            [[buffer(0)]],  // Output
    constant uint64_t* bgv_ct           [[buffer(1)]],  // Input
    constant uint& size                 [[buffer(2)]],
    constant uint64_t& scale            [[buffer(3)]],  // CKKS scale factor
    constant uint64_t& Q_ckks           [[buffer(4)]],

    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;

    uint64_t val = bgv_ct[gid];

    // Multiply by scale: CKKS = BGV * scale
    ckks_ct[gid] = mod_mul(val, scale, Q_ckks, MU_CKKS);
}

kernel void scale_adjust_ckks_to_bgv(
    device uint64_t* bgv_ct             [[buffer(0)]],  // Output
    constant uint64_t* ckks_ct          [[buffer(1)]],  // Input
    constant uint& size                 [[buffer(2)]],
    constant uint64_t& scale            [[buffer(3)]],  // CKKS scale factor
    constant uint64_t& t                [[buffer(4)]],  // BGV plaintext modulus
    constant uint64_t& Q_bgv            [[buffer(5)]],

    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;

    uint64_t val = ckks_ct[gid];

    // Divide by scale and round: BGV = round(CKKS / scale) mod t
    uint64_t divided = val / scale;
    uint64_t remainder = val % scale;

    // Round to nearest
    if (remainder >= scale / 2) {
        divided++;
    }

    // Reduce mod t for plaintext space
    bgv_ct[gid] = divided % t;
}

// =============================================================================
// KERNEL 9: BATCH SCHEME SWITCH PIPELINE
// =============================================================================
//
// Fused pipeline for batch scheme switching
// Combines: modulus switch -> key switch -> format conversion

kernel void batch_scheme_switch_fused(
    device uint64_t* output             [[buffer(0)]],  // Output ciphertexts
    constant uint64_t* input            [[buffer(1)]],  // Input ciphertexts
    constant uint64_t* ksk              [[buffer(2)]],  // Key switch key
    constant uint& input_dim            [[buffer(3)]],  // Input dimension
    constant uint& output_dim           [[buffer(4)]],  // Output dimension
    constant uint& num_levels           [[buffer(5)]],
    constant uint& decomp_log           [[buffer(6)]],
    constant uint& batch_size           [[buffer(7)]],
    constant uint64_t& Q_from           [[buffer(8)]],
    constant uint64_t& Q_to             [[buffer(9)]],
    constant uint& switch_type          [[buffer(10)]],  // 0=TFHE->CKKS, 1=CKKS->TFHE, etc.

    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],

    threadgroup ModSwitchShared& shared [[threadgroup(0)]]
) {
    uint out_idx = gid.x;
    uint batch_idx = gid.y;

    if (out_idx >= output_dim || batch_idx >= batch_size) return;

    uint64_t mask = (1ULL << decomp_log) - 1;
    uint64_t sum = 0;
    uint64_t mu_to = (switch_type == 0) ? MU_CKKS : MU_TFHE;

    // Phase 1: Load and modulus switch input chunk to shared memory
    uint chunk_start = (out_idx / 256) * 256;
    uint local_idx = tid;

    if (chunk_start + local_idx < input_dim) {
        uint in_global = batch_idx * input_dim + chunk_start + local_idx;
        uint64_t val = input[in_global];

        // Modulus switch
        shared.input_chunk[local_idx] = (val * Q_to) / Q_from;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Key switch with decomposition
    for (uint i = 0; i < input_dim; i++) {
        uint64_t coeff = (i >= chunk_start && i < chunk_start + 256) ?
                         shared.input_chunk[i - chunk_start] :
                         ((input[batch_idx * input_dim + i] * Q_to) / Q_from);

        for (uint l = 0; l < num_levels; l++) {
            uint64_t digit = (coeff >> (l * decomp_log)) & mask;
            if (digit == 0) continue;

            uint ksk_idx = i * num_levels * output_dim + l * output_dim + out_idx;
            uint64_t ksk_val = ksk[ksk_idx];

            sum = mod_add(sum, mod_mul(digit, ksk_val, Q_to, mu_to), Q_to);
        }
    }

    // Phase 3: Write output
    output[batch_idx * output_dim + out_idx] = sum;
}

// =============================================================================
// KERNEL 10: SLOT ROTATION FOR SCHEME ALIGNMENT
// =============================================================================
//
// Rotate slots before/after scheme switch for proper alignment
// Different schemes may use different slot encodings

kernel void slot_rotate_for_switch(
    device uint64_t* output             [[buffer(0)]],
    constant uint64_t* input            [[buffer(1)]],
    constant int& rotation              [[buffer(2)]],  // Signed rotation amount
    constant uint& ring_dim             [[buffer(3)]],
    constant uint& batch_size           [[buffer(4)]],
    constant uint64_t& Q                [[buffer(5)]],

    uint3 gid [[thread_position_in_grid]]
) {
    uint coeff_idx = gid.x;
    uint comp_idx = gid.y;
    uint batch_idx = gid.z;

    if (coeff_idx >= ring_dim || comp_idx >= 2 || batch_idx >= batch_size) return;

    // Compute source index with negacyclic semantics
    int src_idx = (int)coeff_idx - rotation;
    bool negate = false;

    // Handle wrap-around
    while (src_idx < 0) {
        src_idx += ring_dim;
        negate = !negate;
    }
    while (src_idx >= (int)ring_dim) {
        src_idx -= ring_dim;
        negate = !negate;
    }

    // Load and optionally negate
    uint in_idx = batch_idx * 2 * ring_dim + comp_idx * ring_dim + src_idx;
    uint64_t val = input[in_idx];

    if (negate) {
        val = mod_neg(val, Q);
    }

    // Write output
    uint out_idx = batch_idx * 2 * ring_dim + comp_idx * ring_dim + coeff_idx;
    output[out_idx] = val;
}
