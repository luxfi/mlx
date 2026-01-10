// Poseidon2 Hash Function for STARK-Friendly Hashing
// Optimized for Goldilocks field (p = 2^64 - 2^32 + 1)
//
// Poseidon2 is a SNARK-friendly hash function with:
// - Low multiplicative complexity
// - Efficient GPU parallelization
// - Used for Merkle trees and Fiat-Shamir in STARKs

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Goldilocks Field Arithmetic (reused from goldilocks.metal)
// =============================================================================

constant uint64_t GOLDILOCKS_P = 0xFFFFFFFF00000001ULL;

inline uint64_t gl_reduce(uint64_t x) {
    return (x >= GOLDILOCKS_P) ? (x - GOLDILOCKS_P) : x;
}

inline uint64_t gl_add(uint64_t a, uint64_t b) {
    uint64_t sum = a + b;
    if (sum < a || sum >= GOLDILOCKS_P) {
        sum -= GOLDILOCKS_P;
    }
    return sum;
}

inline uint64_t gl_sub(uint64_t a, uint64_t b) {
    if (a >= b) return a - b;
    return GOLDILOCKS_P - (b - a);
}

inline void gl_mul128(uint64_t a, uint64_t b, thread uint64_t& hi, thread uint64_t& lo) {
    uint64_t a_lo = a & 0xFFFFFFFF;
    uint64_t a_hi = a >> 32;
    uint64_t b_lo = b & 0xFFFFFFFF;
    uint64_t b_hi = b >> 32;

    uint64_t p0 = a_lo * b_lo;
    uint64_t p1 = a_lo * b_hi;
    uint64_t p2 = a_hi * b_lo;
    uint64_t p3 = a_hi * b_hi;

    uint64_t mid = p1 + (p0 >> 32);
    uint64_t mid_lo = mid & 0xFFFFFFFF;
    uint64_t mid_hi = mid >> 32;

    mid_lo += p2;
    mid_hi += (mid_lo < p2) ? 1 : 0;
    mid_hi += (p2 >> 32);

    lo = (mid_lo << 32) | (p0 & 0xFFFFFFFF);
    hi = p3 + mid_hi;
}

inline uint64_t gl_reduce128(uint64_t hi, uint64_t lo) {
    uint64_t hi_shifted = hi << 32;
    uint64_t result = lo;
    result = gl_add(result, hi_shifted);
    result = gl_sub(result, hi);
    uint64_t hi_upper = hi >> 32;
    if (hi_upper > 0) {
        result = gl_sub(result, hi_upper << 32);
        result = gl_add(result, hi_upper);
    }
    return gl_reduce(result);
}

inline uint64_t gl_mul(uint64_t a, uint64_t b) {
    uint64_t hi, lo;
    gl_mul128(a, b, hi, lo);
    return gl_reduce128(hi, lo);
}

// =============================================================================
// Poseidon2 Parameters
// =============================================================================

// State width (t = 8 for Goldilocks Poseidon2)
constant uint32_t POSEIDON_WIDTH = 8;

// Number of full rounds (beginning + end)
constant uint32_t POSEIDON_FULL_ROUNDS = 8;  // 4 beginning + 4 end

// Number of partial rounds (middle)
constant uint32_t POSEIDON_PARTIAL_ROUNDS = 22;

// S-box exponent (x^7 for Goldilocks)
constant uint64_t POSEIDON_ALPHA = 7;

// Round constants (pre-computed for Goldilocks)
// These would normally be generated from a seed
constant uint64_t POSEIDON_RC[POSEIDON_WIDTH * (POSEIDON_FULL_ROUNDS + POSEIDON_PARTIAL_ROUNDS)] = {
    // Round constants would be filled with proper values from reference implementation
    // Placeholder values for structure
    0x1234567890abcdefULL, 0x234567890abcdef1ULL, 0x34567890abcdef12ULL, 0x4567890abcdef123ULL,
    0x567890abcdef1234ULL, 0x67890abcdef12345ULL, 0x7890abcdef123456ULL, 0x890abcdef1234567ULL,
    // ... more constants
};

// MDS matrix for linear layer (8x8 Cauchy matrix)
constant uint64_t POSEIDON_MDS[POSEIDON_WIDTH][POSEIDON_WIDTH] = {
    {1, 1, 1, 1, 1, 1, 1, 1},
    {1, 2, 3, 4, 5, 6, 7, 8},
    {1, 4, 9, 16, 25, 36, 49, 64},
    {1, 8, 27, 64, 125, 216, 343, 512},
    {1, 16, 81, 256, 625, 1296, 2401, 4096},
    {1, 32, 243, 1024, 3125, 7776, 16807, 32768},
    {1, 64, 729, 4096, 15625, 46656, 117649, 262144},
    {1, 128, 2187, 16384, 78125, 279936, 823543, 2097152}
};

// =============================================================================
// S-box: x^7 in Goldilocks field
// =============================================================================

inline uint64_t poseidon_sbox(uint64_t x) {
    uint64_t x2 = gl_mul(x, x);      // x^2
    uint64_t x4 = gl_mul(x2, x2);    // x^4
    uint64_t x3 = gl_mul(x2, x);     // x^3
    return gl_mul(x4, x3);            // x^7
}

// =============================================================================
// Linear Layer: MDS Matrix Multiplication
// =============================================================================

inline void poseidon_mds(thread uint64_t* state) {
    uint64_t result[POSEIDON_WIDTH];

    for (uint32_t i = 0; i < POSEIDON_WIDTH; i++) {
        result[i] = 0;
        for (uint32_t j = 0; j < POSEIDON_WIDTH; j++) {
            result[i] = gl_add(result[i], gl_mul(POSEIDON_MDS[i][j], state[j]));
        }
    }

    for (uint32_t i = 0; i < POSEIDON_WIDTH; i++) {
        state[i] = result[i];
    }
}

// Poseidon2 optimized internal linear layer
inline void poseidon2_internal_linear(thread uint64_t* state) {
    // Poseidon2 uses a simpler internal matrix for partial rounds
    // M_I = I + diag(d) where d is a fixed diagonal

    // First, compute sum of all elements
    uint64_t sum = 0;
    for (uint32_t i = 0; i < POSEIDON_WIDTH; i++) {
        sum = gl_add(sum, state[i]);
    }

    // Apply: state[i] = state[i] + sum + d_i * state[i]
    // For simplicity, using d_i = 1 (actual values depend on security analysis)
    for (uint32_t i = 0; i < POSEIDON_WIDTH; i++) {
        state[i] = gl_add(state[i], sum);
    }
}

// =============================================================================
// Poseidon2 Permutation
// =============================================================================

inline void poseidon2_permutation(thread uint64_t* state) {
    uint32_t rc_idx = 0;

    // Beginning full rounds (4 rounds)
    for (uint32_t r = 0; r < POSEIDON_FULL_ROUNDS / 2; r++) {
        // Add round constants
        for (uint32_t i = 0; i < POSEIDON_WIDTH; i++) {
            state[i] = gl_add(state[i], POSEIDON_RC[rc_idx++]);
        }

        // S-box on all elements
        for (uint32_t i = 0; i < POSEIDON_WIDTH; i++) {
            state[i] = poseidon_sbox(state[i]);
        }

        // MDS matrix
        poseidon_mds(state);
    }

    // Partial rounds (22 rounds)
    for (uint32_t r = 0; r < POSEIDON_PARTIAL_ROUNDS; r++) {
        // Add round constant only to first element
        state[0] = gl_add(state[0], POSEIDON_RC[rc_idx++]);

        // S-box only on first element
        state[0] = poseidon_sbox(state[0]);

        // Internal linear layer (faster than full MDS)
        poseidon2_internal_linear(state);
    }

    // Ending full rounds (4 rounds)
    for (uint32_t r = 0; r < POSEIDON_FULL_ROUNDS / 2; r++) {
        // Add round constants
        for (uint32_t i = 0; i < POSEIDON_WIDTH; i++) {
            state[i] = gl_add(state[i], POSEIDON_RC[rc_idx++]);
        }

        // S-box on all elements
        for (uint32_t i = 0; i < POSEIDON_WIDTH; i++) {
            state[i] = poseidon_sbox(state[i]);
        }

        // MDS matrix
        poseidon_mds(state);
    }
}

// =============================================================================
// Sponge Construction
// =============================================================================

// Hash arbitrary input to single field element
kernel void poseidon_hash(
    device const uint64_t* input [[buffer(0)]],
    device uint64_t* output [[buffer(1)]],
    constant uint32_t& input_len [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    // Each thread hashes one input block
    uint32_t offset = index * (POSEIDON_WIDTH - 1);  // Rate = width - 1

    if (offset >= input_len) return;

    // Initialize state
    uint64_t state[POSEIDON_WIDTH] = {0};

    // Absorb phase
    uint32_t remaining = input_len - offset;
    uint32_t to_absorb = min(remaining, POSEIDON_WIDTH - 1);

    for (uint32_t i = 0; i < to_absorb; i++) {
        state[i] = input[offset + i];
    }

    // Domain separation / padding
    if (to_absorb < POSEIDON_WIDTH - 1) {
        state[to_absorb] = 1;  // Padding
    }

    // Permutation
    poseidon2_permutation(state);

    // Output first element (squeeze phase)
    output[index] = state[0];
}

// Hash pair for Merkle tree (2-to-1 compression)
kernel void poseidon_hash_pair(
    device const uint64_t* left [[buffer(0)]],
    device const uint64_t* right [[buffer(1)]],
    device uint64_t* output [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    // Initialize state with inputs
    uint64_t state[POSEIDON_WIDTH] = {0};
    state[0] = left[index];
    state[1] = right[index];

    // Domain separation for 2-to-1 hash
    state[POSEIDON_WIDTH - 1] = 2;

    // Permutation
    poseidon2_permutation(state);

    // Output
    output[index] = state[0];
}

// =============================================================================
// Merkle Tree Construction
// =============================================================================

// Build one layer of Merkle tree
kernel void poseidon_merkle_layer(
    device const uint64_t* current_layer [[buffer(0)]],
    device uint64_t* next_layer [[buffer(1)]],
    constant uint32_t& current_size [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= current_size / 2) return;

    uint64_t left = current_layer[2 * index];
    uint64_t right = current_layer[2 * index + 1];

    // Initialize state
    uint64_t state[POSEIDON_WIDTH] = {0};
    state[0] = left;
    state[1] = right;
    state[POSEIDON_WIDTH - 1] = 2;  // Domain separation

    // Permutation
    poseidon2_permutation(state);

    next_layer[index] = state[0];
}

// Batch hash for multiple independent Merkle tree constructions
kernel void poseidon_batch_merkle_layer(
    device const uint64_t* leaves [[buffer(0)]],
    device uint64_t* parents [[buffer(1)]],
    constant uint32_t& num_pairs [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= num_pairs) return;

    uint64_t left = leaves[2 * index];
    uint64_t right = leaves[2 * index + 1];

    uint64_t state[POSEIDON_WIDTH] = {0};
    state[0] = left;
    state[1] = right;
    state[POSEIDON_WIDTH - 1] = 2;

    poseidon2_permutation(state);

    parents[index] = state[0];
}

// Verify Merkle proof (single proof, multiple in parallel)
kernel void poseidon_verify_merkle_proof(
    device const uint64_t* leaf [[buffer(0)]],
    device const uint64_t* path [[buffer(1)]],
    device const uint32_t* path_indices [[buffer(2)]],  // 0 = left, 1 = right
    device const uint64_t* expected_root [[buffer(3)]],
    device uint32_t* result [[buffer(4)]],  // 1 = valid, 0 = invalid
    constant uint32_t& path_len [[buffer(5)]],
    uint proof_idx [[thread_position_in_grid]]
) {
    uint64_t current = leaf[proof_idx];

    for (uint32_t i = 0; i < path_len; i++) {
        uint64_t sibling = path[proof_idx * path_len + i];
        uint32_t idx = path_indices[proof_idx * path_len + i];

        uint64_t left = (idx == 0) ? current : sibling;
        uint64_t right = (idx == 0) ? sibling : current;

        uint64_t state[POSEIDON_WIDTH] = {0};
        state[0] = left;
        state[1] = right;
        state[POSEIDON_WIDTH - 1] = 2;

        poseidon2_permutation(state);

        current = state[0];
    }

    result[proof_idx] = (current == expected_root[proof_idx]) ? 1 : 0;
}

// =============================================================================
// Transcript (Fiat-Shamir)
// =============================================================================

// Add data to Fiat-Shamir transcript and squeeze challenge
kernel void poseidon_fiat_shamir(
    device const uint64_t* transcript_state [[buffer(0)]],
    device const uint64_t* new_data [[buffer(1)]],
    device uint64_t* updated_state [[buffer(2)]],
    device uint64_t* challenge [[buffer(3)]],
    constant uint32_t& data_len [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    if (index != 0) return;  // Single thread operation

    // Load current state
    uint64_t state[POSEIDON_WIDTH];
    for (uint32_t i = 0; i < POSEIDON_WIDTH; i++) {
        state[i] = transcript_state[i];
    }

    // Absorb new data
    uint32_t absorbed = 0;
    while (absorbed < data_len) {
        uint32_t rate = POSEIDON_WIDTH - 1;
        uint32_t to_absorb = min(data_len - absorbed, rate);

        for (uint32_t i = 0; i < to_absorb; i++) {
            state[i] = gl_add(state[i], new_data[absorbed + i]);
        }
        absorbed += to_absorb;

        poseidon2_permutation(state);
    }

    // Output updated state
    for (uint32_t i = 0; i < POSEIDON_WIDTH; i++) {
        updated_state[i] = state[i];
    }

    // Squeeze challenge
    *challenge = state[0];
}
