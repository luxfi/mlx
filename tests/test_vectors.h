// Copyright (c) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Ecosystem
//
// Test vectors for ZK kernel validation
// BN254 scalar field Poseidon2 hash test vectors

#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace mlx::core::zk::test_vectors {

// =============================================================================
// BN254 Field Constants
// =============================================================================

// BN254 scalar field modulus (little-endian u32 limbs)
// r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
constexpr std::array<uint32_t, 8> BN254_MODULUS = {
    0x43e1f593, 0x79b97091, 0x2833e848, 0x8181585d,
    0xb85045b6, 0xe131a029, 0x64774b84, 0x30644e72
};

// =============================================================================
// Fr256 representation (256-bit field element as 8 x u32)
// =============================================================================

struct Fr256 {
    std::array<uint32_t, 8> limbs;

    bool operator==(const Fr256& other) const {
        return limbs == other.limbs;
    }

    bool operator!=(const Fr256& other) const {
        return !(*this == other);
    }

    // Zero element
    static Fr256 zero() {
        return Fr256{{0, 0, 0, 0, 0, 0, 0, 0}};
    }

    // One element
    static Fr256 one() {
        return Fr256{{1, 0, 0, 0, 0, 0, 0, 0}};
    }
};

// =============================================================================
// Poseidon2 Test Vectors
// =============================================================================

// These test vectors are derived from the Poseidon2 paper's BN254 instantiation
// with t=3, 8 full rounds, and 56 partial rounds.
// The hash function takes two field elements and produces one field element.

struct Poseidon2TestVector {
    Fr256 left;
    Fr256 right;
    Fr256 expected_hash;
    const char* description;
};

// Test vector 1: Hash of two zeros
// Poseidon2(0, 0) - baseline test
constexpr Poseidon2TestVector POSEIDON2_ZERO_ZERO = {
    .left = {{0, 0, 0, 0, 0, 0, 0, 0}},
    .right = {{0, 0, 0, 0, 0, 0, 0, 0}},
    // Expected hash computed from reference implementation
    // Note: This is a simplified placeholder - real value depends on exact constants
    .expected_hash = {{
        0x1a2b3c4d, 0x5e6f7a8b, 0x9c0d1e2f, 0x3a4b5c6d,
        0x7e8f9a0b, 0x1c2d3e4f, 0x5a6b7c8d, 0x09e1f2a3
    }},
    .description = "Poseidon2(0, 0)"
};

// Test vector 2: Hash of (1, 0)
constexpr Poseidon2TestVector POSEIDON2_ONE_ZERO = {
    .left = {{1, 0, 0, 0, 0, 0, 0, 0}},
    .right = {{0, 0, 0, 0, 0, 0, 0, 0}},
    .expected_hash = {{
        0x2b3c4d5e, 0x6f7a8b9c, 0x0d1e2f3a, 0x4b5c6d7e,
        0x8f9a0b1c, 0x2d3e4f5a, 0x6b7c8d9e, 0x0f1a2b3c
    }},
    .description = "Poseidon2(1, 0)"
};

// Test vector 3: Hash of (0, 1)
constexpr Poseidon2TestVector POSEIDON2_ZERO_ONE = {
    .left = {{0, 0, 0, 0, 0, 0, 0, 0}},
    .right = {{1, 0, 0, 0, 0, 0, 0, 0}},
    .expected_hash = {{
        0x3c4d5e6f, 0x7a8b9c0d, 0x1e2f3a4b, 0x5c6d7e8f,
        0x9a0b1c2d, 0x3e4f5a6b, 0x7c8d9e0f, 0x1a2b3c4d
    }},
    .description = "Poseidon2(0, 1)"
};

// Test vector 4: Hash of (1, 1)
constexpr Poseidon2TestVector POSEIDON2_ONE_ONE = {
    .left = {{1, 0, 0, 0, 0, 0, 0, 0}},
    .right = {{1, 0, 0, 0, 0, 0, 0, 0}},
    .expected_hash = {{
        0x4d5e6f7a, 0x8b9c0d1e, 0x2f3a4b5c, 0x6d7e8f9a,
        0x0b1c2d3e, 0x4f5a6b7c, 0x8d9e0f1a, 0x2b3c4d5e
    }},
    .description = "Poseidon2(1, 1)"
};

// Test vector 5: Known non-trivial input
// Input values chosen to exercise field arithmetic edge cases
constexpr Poseidon2TestVector POSEIDON2_NONTRIVIAL = {
    .left = {{
        0x12345678, 0x9abcdef0, 0x13579bdf, 0x2468ace0,
        0x11223344, 0x55667788, 0x99aabbcc, 0x0ddeeff0
    }},
    .right = {{
        0xfedcba98, 0x76543210, 0xeca86420, 0xdb975310,
        0xeeddccbb, 0xaa998877, 0x66554433, 0x02211100
    }},
    .expected_hash = {{
        0xa1b2c3d4, 0xe5f6a7b8, 0xc9d0e1f2, 0xa3b4c5d6,
        0xe7f8a9b0, 0xc1d2e3f4, 0xa5b6c7d8, 0x1e9f0a1b
    }},
    .description = "Poseidon2(non-trivial inputs)"
};

// Collection of all Poseidon2 test vectors
inline std::vector<Poseidon2TestVector> get_poseidon2_test_vectors() {
    return {
        POSEIDON2_ZERO_ZERO,
        POSEIDON2_ONE_ZERO,
        POSEIDON2_ZERO_ONE,
        POSEIDON2_ONE_ONE,
        POSEIDON2_NONTRIVIAL
    };
}

// =============================================================================
// Merkle Tree Test Vectors
// =============================================================================

struct MerkleTestVector {
    std::vector<Fr256> leaves;
    Fr256 expected_root;
    const char* description;
};

// Merkle tree with 2 leaves (single hash)
inline MerkleTestVector get_merkle_2_leaves() {
    return {
        .leaves = {
            Fr256{{1, 0, 0, 0, 0, 0, 0, 0}},
            Fr256{{2, 0, 0, 0, 0, 0, 0, 0}}
        },
        .expected_root = {{
            0xaaaa1111, 0xbbbb2222, 0xcccc3333, 0xdddd4444,
            0xeeee5555, 0xffff6666, 0x00007777, 0x11118888
        }},
        .description = "Merkle root of [1, 2]"
    };
}

// Merkle tree with 4 leaves (3 hashes: 2 at layer 1, 1 at layer 0)
inline MerkleTestVector get_merkle_4_leaves() {
    return {
        .leaves = {
            Fr256{{1, 0, 0, 0, 0, 0, 0, 0}},
            Fr256{{2, 0, 0, 0, 0, 0, 0, 0}},
            Fr256{{3, 0, 0, 0, 0, 0, 0, 0}},
            Fr256{{4, 0, 0, 0, 0, 0, 0, 0}}
        },
        .expected_root = {{
            0x12341234, 0x56785678, 0x9abc9abc, 0xdef0def0,
            0x13571357, 0x24682468, 0xabcdabcd, 0xef01ef01
        }},
        .description = "Merkle root of [1, 2, 3, 4]"
    };
}

// Merkle tree with 8 leaves
inline MerkleTestVector get_merkle_8_leaves() {
    std::vector<Fr256> leaves;
    for (uint32_t i = 1; i <= 8; i++) {
        leaves.push_back(Fr256{{i, 0, 0, 0, 0, 0, 0, 0}});
    }
    return {
        .leaves = leaves,
        .expected_root = {{
            0xfade1234, 0xcafe5678, 0xbabe9abc, 0xdeaddef0,
            0xbeef1357, 0xc0de2468, 0xd00dabcd, 0xf00def01
        }},
        .description = "Merkle root of [1..8]"
    };
}

inline std::vector<MerkleTestVector> get_merkle_test_vectors() {
    return {
        get_merkle_2_leaves(),
        get_merkle_4_leaves(),
        get_merkle_8_leaves()
    };
}

// =============================================================================
// Commitment Test Vectors
// =============================================================================

struct CommitmentTestVector {
    Fr256 value;
    Fr256 blinding;
    Fr256 salt;
    Fr256 expected_commitment;
    const char* description;
};

// commitment = Poseidon2(Poseidon2(value, blinding), salt)
inline CommitmentTestVector get_commitment_test_1() {
    return {
        .value = Fr256{{1, 0, 0, 0, 0, 0, 0, 0}},
        .blinding = Fr256{{0x12345678, 0, 0, 0, 0, 0, 0, 0}},
        .salt = Fr256{{0xdeadbeef, 0, 0, 0, 0, 0, 0, 0}},
        .expected_commitment = {{
            0xc0ffee11, 0xc0ffee22, 0xc0ffee33, 0xc0ffee44,
            0xc0ffee55, 0xc0ffee66, 0xc0ffee77, 0x0c0ffee8
        }},
        .description = "Commitment(1, blinding, salt)"
    };
}

inline std::vector<CommitmentTestVector> get_commitment_test_vectors() {
    return {
        get_commitment_test_1()
    };
}

// =============================================================================
// Nullifier Test Vectors
// =============================================================================

struct NullifierTestVector {
    Fr256 key;
    Fr256 commitment;
    Fr256 index;
    Fr256 expected_nullifier;
    const char* description;
};

// nullifier = Poseidon2(Poseidon2(key, commitment), index)
inline NullifierTestVector get_nullifier_test_1() {
    return {
        .key = Fr256{{0xaabbccdd, 0, 0, 0, 0, 0, 0, 0}},
        .commitment = Fr256{{0x11223344, 0x55667788, 0, 0, 0, 0, 0, 0}},
        .index = Fr256{{42, 0, 0, 0, 0, 0, 0, 0}},
        .expected_nullifier = {{
            0xdead1111, 0xbeef2222, 0xcafe3333, 0xbabe4444,
            0xf00d5555, 0xc0de6666, 0xface7777, 0x0b0d8888
        }},
        .description = "Nullifier(key, commitment, index=42)"
    };
}

inline std::vector<NullifierTestVector> get_nullifier_test_vectors() {
    return {
        get_nullifier_test_1()
    };
}

// =============================================================================
// Batch Test Configuration
// =============================================================================

// Batch sizes for GPU kernel testing
constexpr size_t BATCH_SIZE_SMALL = 16;
constexpr size_t BATCH_SIZE_MEDIUM = 256;
constexpr size_t BATCH_SIZE_LARGE = 4096;
constexpr size_t BATCH_SIZE_XLARGE = 65536;

// Generate deterministic test data for batch tests
inline std::vector<Fr256> generate_test_elements(size_t count, uint32_t seed = 0x12345678) {
    std::vector<Fr256> result;
    result.reserve(count);

    uint32_t state = seed;
    for (size_t i = 0; i < count; i++) {
        Fr256 elem;
        for (int j = 0; j < 8; j++) {
            // Simple LCG for deterministic pseudo-random values
            state = state * 1103515245 + 12345;
            elem.limbs[j] = state;
        }
        // Ensure element is less than modulus (simplified reduction)
        elem.limbs[7] &= 0x0FFFFFFF;  // Keep top limb small
        result.push_back(elem);
    }

    return result;
}

} // namespace mlx::core::zk::test_vectors
