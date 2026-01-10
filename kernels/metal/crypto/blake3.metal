// =============================================================================
// Blake3 Metal Compute Shaders
// =============================================================================
//
// GPU-accelerated Blake3 hash function on Apple Silicon.
// Implements batch hashing for high-throughput applications.
//
// Blake3 Parameters:
//   Block size: 64 bytes
//   Output size: 256 bits (default), extensible to arbitrary length
//   Rounds: 7 per compression
//
// Reference: https://github.com/BLAKE3-team/BLAKE3-specs
//
// Copyright (C) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Blake3 Constants
// =============================================================================

// Initial state (IV from SHA-256)
constant uint32_t BLAKE3_IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

// Message permutation schedule
constant uint8_t MSG_PERMUTATION[16] = {
    2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8
};

// Block length for chunk chaining
constant uint32_t BLOCK_LEN = 64;
constant uint32_t CHUNK_LEN = 1024;

// Domain separation flags
constant uint32_t CHUNK_START = 1 << 0;
constant uint32_t CHUNK_END = 1 << 1;
constant uint32_t PARENT = 1 << 2;
constant uint32_t ROOT = 1 << 3;

// =============================================================================
// Blake3 State
// =============================================================================

struct Blake3State {
    uint32_t cv[8];          // Chaining value
    uint64_t chunk_counter;
    uint8_t block[64];
    uint8_t block_len;
    uint8_t blocks_compressed;
    uint8_t flags;
};

struct Blake3Output {
    uint32_t hash[8];        // 256-bit output
};

// =============================================================================
// Rotation and Mixing Functions
// =============================================================================

inline uint32_t rotr32(uint32_t x, uint8_t n) {
    return (x >> n) | (x << (32 - n));
}

// G function - quarter round
inline void g(thread uint32_t& a, thread uint32_t& b, thread uint32_t& c, thread uint32_t& d,
              uint32_t mx, uint32_t my) {
    a = a + b + mx;
    d = rotr32(d ^ a, 16);
    c = c + d;
    b = rotr32(b ^ c, 12);
    a = a + b + my;
    d = rotr32(d ^ a, 8);
    c = c + d;
    b = rotr32(b ^ c, 7);
}

// =============================================================================
// Compression Function
// =============================================================================

inline void compress(thread uint32_t state[16],
                     thread const uint32_t cv[8],
                     thread const uint32_t block_words[16],
                     uint64_t counter,
                     uint32_t block_len,
                     uint32_t flags) {
    // Initialize state
    for (int i = 0; i < 8; i++) {
        state[i] = cv[i];
    }
    state[8] = BLAKE3_IV[0];
    state[9] = BLAKE3_IV[1];
    state[10] = BLAKE3_IV[2];
    state[11] = BLAKE3_IV[3];
    state[12] = (uint32_t)counter;
    state[13] = (uint32_t)(counter >> 32);
    state[14] = block_len;
    state[15] = flags;

    // Message schedule
    uint32_t m[16];
    for (int i = 0; i < 16; i++) {
        m[i] = block_words[i];
    }

    // 7 rounds
    for (int round = 0; round < 7; round++) {
        // Column step
        g(state[0], state[4], state[8],  state[12], m[0],  m[1]);
        g(state[1], state[5], state[9],  state[13], m[2],  m[3]);
        g(state[2], state[6], state[10], state[14], m[4],  m[5]);
        g(state[3], state[7], state[11], state[15], m[6],  m[7]);

        // Diagonal step
        g(state[0], state[5], state[10], state[15], m[8],  m[9]);
        g(state[1], state[6], state[11], state[12], m[10], m[11]);
        g(state[2], state[7], state[8],  state[13], m[12], m[13]);
        g(state[3], state[4], state[9],  state[14], m[14], m[15]);

        // Permute message for next round
        uint32_t temp[16];
        for (int i = 0; i < 16; i++) {
            temp[i] = m[MSG_PERMUTATION[i]];
        }
        for (int i = 0; i < 16; i++) {
            m[i] = temp[i];
        }
    }

    // Finalize (XOR with input chaining value)
    for (int i = 0; i < 8; i++) {
        state[i] ^= state[i + 8];
        state[i + 8] ^= cv[i];
    }
}

// =============================================================================
// Hash Single Block Kernel
// =============================================================================

kernel void blake3_hash_block(
    device const uint8_t* input [[buffer(0)]],
    device uint32_t* output [[buffer(1)]],
    device const uint32_t* input_lengths [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    uint32_t len = input_lengths[index];
    uint32_t offset = index * 64; // Assuming 64-byte aligned inputs

    // Load block words (little-endian)
    uint32_t block_words[16];
    for (int i = 0; i < 16; i++) {
        uint32_t word = 0;
        for (int j = 0; j < 4; j++) {
            uint32_t byte_idx = offset + i * 4 + j;
            if (i * 4 + j < len) {
                word |= ((uint32_t)input[byte_idx]) << (j * 8);
            }
        }
        block_words[i] = word;
    }

    // Compress with IV
    uint32_t state[16];
    uint32_t cv[8];
    for (int i = 0; i < 8; i++) {
        cv[i] = BLAKE3_IV[i];
    }

    uint32_t flags = CHUNK_START | CHUNK_END | ROOT;
    compress(state, cv, block_words, 0, len, flags);

    // Output first 8 words (256 bits)
    uint32_t out_offset = index * 8;
    for (int i = 0; i < 8; i++) {
        output[out_offset + i] = state[i];
    }
}

// =============================================================================
// Batch Hash Kernel
// =============================================================================

kernel void blake3_batch_hash(
    device const uint8_t* inputs [[buffer(0)]],
    device uint32_t* outputs [[buffer(1)]],
    device const uint32_t* offsets [[buffer(2)]],
    device const uint32_t* lengths [[buffer(3)]],
    uint batch_idx [[thread_position_in_grid]]
) {
    uint32_t offset = offsets[batch_idx];
    uint32_t len = lengths[batch_idx];

    // Initialize chaining value
    uint32_t cv[8];
    for (int i = 0; i < 8; i++) {
        cv[i] = BLAKE3_IV[i];
    }

    // Process full blocks
    uint64_t chunk_counter = 0;
    uint32_t bytes_processed = 0;

    while (bytes_processed < len) {
        uint32_t block_len = min(64u, len - bytes_processed);

        // Load block
        uint32_t block_words[16] = {0};
        for (uint32_t i = 0; i < block_len; i++) {
            uint32_t word_idx = i / 4;
            uint32_t byte_pos = i % 4;
            block_words[word_idx] |= ((uint32_t)inputs[offset + bytes_processed + i]) << (byte_pos * 8);
        }

        // Determine flags
        uint32_t flags = 0;
        if (bytes_processed == 0) flags |= CHUNK_START;
        if (bytes_processed + block_len >= len) flags |= CHUNK_END | ROOT;

        // Compress
        uint32_t state[16];
        compress(state, cv, block_words, chunk_counter, block_len, flags);

        // Update chaining value
        for (int i = 0; i < 8; i++) {
            cv[i] = state[i];
        }

        bytes_processed += block_len;
        if ((bytes_processed % CHUNK_LEN) == 0) {
            chunk_counter++;
        }
    }

    // Output
    uint32_t out_offset = batch_idx * 8;
    for (int i = 0; i < 8; i++) {
        outputs[out_offset + i] = cv[i];
    }
}

// =============================================================================
// Merkle Tree Root Kernel
// =============================================================================

// Compute parent node from two child hashes
kernel void blake3_merge_nodes(
    device const uint32_t* left_hashes [[buffer(0)]],
    device const uint32_t* right_hashes [[buffer(1)]],
    device uint32_t* parent_hashes [[buffer(2)]],
    constant uint32_t& num_pairs [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= num_pairs) return;

    // Load left and right child hashes
    uint32_t block_words[16];
    for (int i = 0; i < 8; i++) {
        block_words[i] = left_hashes[index * 8 + i];
        block_words[i + 8] = right_hashes[index * 8 + i];
    }

    // Compress with PARENT flag
    uint32_t state[16];
    uint32_t cv[8];
    for (int i = 0; i < 8; i++) {
        cv[i] = BLAKE3_IV[i];
    }

    uint32_t flags = PARENT;
    compress(state, cv, block_words, 0, 64, flags);

    // Output parent hash
    uint32_t out_offset = index * 8;
    for (int i = 0; i < 8; i++) {
        parent_hashes[out_offset + i] = state[i];
    }
}

// =============================================================================
// XOF (Extendable Output Function) Kernel
// =============================================================================

kernel void blake3_xof(
    device const uint8_t* input [[buffer(0)]],
    device uint8_t* output [[buffer(1)]],
    constant uint32_t& input_len [[buffer(2)]],
    constant uint32_t& output_len [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    // Each thread generates 64 bytes of output
    uint32_t out_offset = index * 64;
    if (out_offset >= output_len) return;

    // First, compute the base hash
    uint32_t cv[8];
    for (int i = 0; i < 8; i++) {
        cv[i] = BLAKE3_IV[i];
    }

    // Load and hash input (simplified for short inputs)
    uint32_t block_words[16] = {0};
    for (uint32_t i = 0; i < min(64u, input_len); i++) {
        uint32_t word_idx = i / 4;
        uint32_t byte_pos = i % 4;
        block_words[word_idx] |= ((uint32_t)input[i]) << (byte_pos * 8);
    }

    uint32_t state[16];
    uint32_t flags = CHUNK_START | CHUNK_END | ROOT;
    compress(state, cv, block_words, 0, input_len, flags);

    // XOF: use counter to extend output
    uint32_t xof_state[16];
    uint32_t xof_cv[8];
    for (int i = 0; i < 8; i++) {
        xof_cv[i] = state[i];
    }

    // Extend using counter = block index
    uint64_t counter = index;
    uint32_t zero_block[16] = {0};
    compress(xof_state, xof_cv, zero_block, counter, 0, ROOT);

    // Output 64 bytes
    uint32_t bytes_to_write = min(64u, output_len - out_offset);
    for (uint32_t i = 0; i < bytes_to_write; i++) {
        uint32_t word_idx = i / 4;
        uint32_t byte_pos = i % 4;
        output[out_offset + i] = (uint8_t)(xof_state[word_idx] >> (byte_pos * 8));
    }
}

// =============================================================================
// Enhanced Merkle Tree Operations
// =============================================================================

// Build one layer of Merkle tree from contiguous array
// Input: current_layer has `layer_size` hashes (each 8 uint32_t = 32 bytes)
// Output: next_layer has layer_size/2 parent hashes
kernel void blake3_merkle_layer(
    device const uint32_t* current_layer [[buffer(0)]],
    device uint32_t* next_layer [[buffer(1)]],
    constant uint32_t& layer_size [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= layer_size / 2) return;

    // Load left child (8 words at position 2*index)
    uint32_t left_offset = 2 * index * 8;
    uint32_t block_words[16];
    for (int i = 0; i < 8; i++) {
        block_words[i] = current_layer[left_offset + i];
    }

    // Load right child (8 words at position 2*index + 1)
    uint32_t right_offset = (2 * index + 1) * 8;
    for (int i = 0; i < 8; i++) {
        block_words[i + 8] = current_layer[right_offset + i];
    }

    // Compress with PARENT flag
    uint32_t state[16];
    uint32_t cv[8];
    for (int i = 0; i < 8; i++) {
        cv[i] = BLAKE3_IV[i];
    }

    compress(state, cv, block_words, 0, 64, PARENT);

    // Write parent hash
    uint32_t out_offset = index * 8;
    for (int i = 0; i < 8; i++) {
        next_layer[out_offset + i] = state[i];
    }
}

// Hash leaves (raw data) to first layer of Merkle tree
// Each thread hashes one leaf of fixed size
kernel void blake3_hash_leaves(
    device const uint8_t* leaf_data [[buffer(0)]],
    device uint32_t* leaf_hashes [[buffer(1)]],
    constant uint32_t& leaf_size [[buffer(2)]],  // Size of each leaf in bytes
    constant uint32_t& num_leaves [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= num_leaves) return;

    uint32_t offset = index * leaf_size;

    // Initialize chaining value
    uint32_t cv[8];
    for (int i = 0; i < 8; i++) {
        cv[i] = BLAKE3_IV[i];
    }

    // Process leaf data in chunks
    uint32_t bytes_processed = 0;
    uint64_t chunk_counter = 0;

    while (bytes_processed < leaf_size) {
        uint32_t block_len = min(64u, leaf_size - bytes_processed);

        // Load block
        uint32_t block_words[16] = {0};
        for (uint32_t i = 0; i < block_len; i++) {
            uint32_t word_idx = i / 4;
            uint32_t byte_pos = i % 4;
            block_words[word_idx] |= ((uint32_t)leaf_data[offset + bytes_processed + i]) << (byte_pos * 8);
        }

        // Determine flags
        uint32_t flags = 0;
        if (bytes_processed == 0) flags |= CHUNK_START;
        if (bytes_processed + block_len >= leaf_size) flags |= CHUNK_END;

        // Compress
        uint32_t state[16];
        compress(state, cv, block_words, chunk_counter, block_len, flags);

        // Update chaining value
        for (int i = 0; i < 8; i++) {
            cv[i] = state[i];
        }

        bytes_processed += block_len;
    }

    // Write leaf hash
    uint32_t out_offset = index * 8;
    for (int i = 0; i < 8; i++) {
        leaf_hashes[out_offset + i] = cv[i];
    }
}

// Verify a Merkle proof
// Computes root from leaf and sibling path, compares to expected
kernel void blake3_verify_merkle_proof(
    device const uint32_t* leaf_hash [[buffer(0)]],      // Leaf hash (8 words per proof)
    device const uint32_t* sibling_path [[buffer(1)]],   // Siblings (8 words each)
    device const uint32_t* path_indices [[buffer(2)]],   // 0=left, 1=right for each level
    device const uint32_t* expected_root [[buffer(3)]],  // Expected root (8 words per proof)
    device uint32_t* results [[buffer(4)]],              // 1=valid, 0=invalid
    constant uint32_t& path_len [[buffer(5)]],
    uint proof_idx [[thread_position_in_grid]]
) {
    // Load leaf hash
    uint32_t current[8];
    uint32_t leaf_offset = proof_idx * 8;
    for (int i = 0; i < 8; i++) {
        current[i] = leaf_hash[leaf_offset + i];
    }

    // Traverse up the tree
    for (uint32_t level = 0; level < path_len; level++) {
        uint32_t sibling_offset = (proof_idx * path_len + level) * 8;
        uint32_t idx = path_indices[proof_idx * path_len + level];

        // Prepare block: [left, right] based on index
        uint32_t block_words[16];
        if (idx == 0) {
            // Current is left child
            for (int i = 0; i < 8; i++) {
                block_words[i] = current[i];
                block_words[i + 8] = sibling_path[sibling_offset + i];
            }
        } else {
            // Current is right child
            for (int i = 0; i < 8; i++) {
                block_words[i] = sibling_path[sibling_offset + i];
                block_words[i + 8] = current[i];
            }
        }

        // Compress
        uint32_t cv[8];
        for (int i = 0; i < 8; i++) {
            cv[i] = BLAKE3_IV[i];
        }

        uint32_t state[16];
        compress(state, cv, block_words, 0, 64, PARENT);

        // Update current
        for (int i = 0; i < 8; i++) {
            current[i] = state[i];
        }
    }

    // Compare with expected root
    uint32_t root_offset = proof_idx * 8;
    bool valid = true;
    for (int i = 0; i < 8; i++) {
        if (current[i] != expected_root[root_offset + i]) {
            valid = false;
            break;
        }
    }

    results[proof_idx] = valid ? 1 : 0;
}

// Batch Merkle proof verification (multiple proofs in parallel)
kernel void blake3_batch_verify_proofs(
    device const uint32_t* leaf_hashes [[buffer(0)]],    // All leaf hashes
    device const uint32_t* all_siblings [[buffer(1)]],   // All sibling paths
    device const uint32_t* all_indices [[buffer(2)]],    // All path indices
    device const uint32_t* expected_root [[buffer(3)]],  // Single root for all
    device uint32_t* results [[buffer(4)]],              // Per-proof results
    constant uint32_t& path_len [[buffer(5)]],
    constant uint32_t& num_proofs [[buffer(6)]],
    uint proof_idx [[thread_position_in_grid]]
) {
    if (proof_idx >= num_proofs) return;

    // Load leaf hash for this proof
    uint32_t current[8];
    uint32_t leaf_offset = proof_idx * 8;
    for (int i = 0; i < 8; i++) {
        current[i] = leaf_hashes[leaf_offset + i];
    }

    // Traverse path
    for (uint32_t level = 0; level < path_len; level++) {
        uint32_t sibling_offset = (proof_idx * path_len + level) * 8;
        uint32_t idx = all_indices[proof_idx * path_len + level];

        uint32_t block_words[16];
        if (idx == 0) {
            for (int i = 0; i < 8; i++) {
                block_words[i] = current[i];
                block_words[i + 8] = all_siblings[sibling_offset + i];
            }
        } else {
            for (int i = 0; i < 8; i++) {
                block_words[i] = all_siblings[sibling_offset + i];
                block_words[i + 8] = current[i];
            }
        }

        uint32_t cv[8];
        for (int i = 0; i < 8; i++) {
            cv[i] = BLAKE3_IV[i];
        }

        uint32_t state[16];
        compress(state, cv, block_words, 0, 64, PARENT);

        for (int i = 0; i < 8; i++) {
            current[i] = state[i];
        }
    }

    // Compare with shared root
    bool valid = true;
    for (int i = 0; i < 8; i++) {
        if (current[i] != expected_root[i]) {
            valid = false;
            break;
        }
    }

    results[proof_idx] = valid ? 1 : 0;
}

// KDF (Key Derivation Function) using Blake3
kernel void blake3_derive_key(
    device const uint8_t* context [[buffer(0)]],    // Context string
    device const uint8_t* key_material [[buffer(1)]],
    device uint8_t* derived_key [[buffer(2)]],
    constant uint32_t& context_len [[buffer(3)]],
    constant uint32_t& key_len [[buffer(4)]],
    constant uint32_t& output_len [[buffer(5)]],
    uint index [[thread_position_in_grid]]
) {
    if (index != 0) return;  // Single-threaded for simplicity

    // Step 1: Hash context to get derive_key_context
    uint32_t cv[8];
    for (int i = 0; i < 8; i++) {
        cv[i] = BLAKE3_IV[i];
    }

    // Hash context (simplified for short contexts)
    uint32_t context_words[16] = {0};
    for (uint32_t i = 0; i < min(64u, context_len); i++) {
        uint32_t word_idx = i / 4;
        uint32_t byte_pos = i % 4;
        context_words[word_idx] |= ((uint32_t)context[i]) << (byte_pos * 8);
    }

    uint32_t state[16];
    compress(state, cv, context_words, 0, context_len, CHUNK_START | CHUNK_END);

    // Use state as new IV
    uint32_t key_cv[8];
    for (int i = 0; i < 8; i++) {
        key_cv[i] = state[i];
    }

    // Step 2: Hash key material with derive_key_context
    uint32_t key_words[16] = {0};
    for (uint32_t i = 0; i < min(64u, key_len); i++) {
        uint32_t word_idx = i / 4;
        uint32_t byte_pos = i % 4;
        key_words[word_idx] |= ((uint32_t)key_material[i]) << (byte_pos * 8);
    }

    compress(state, key_cv, key_words, 0, key_len, CHUNK_START | CHUNK_END | ROOT);

    // Output derived key
    for (uint32_t i = 0; i < min(32u, output_len); i++) {
        uint32_t word_idx = i / 4;
        uint32_t byte_pos = i % 4;
        derived_key[i] = (uint8_t)(state[word_idx] >> (byte_pos * 8));
    }
}
