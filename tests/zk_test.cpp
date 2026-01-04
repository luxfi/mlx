// Copyright (c) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Ecosystem
//
// ZK Kernel Tests: Validate WGSL/GPU kernels match CPU reference implementation
//
// Test strategy:
// 1. Run CPU reference implementation on known test vectors
// 2. Run WebGPU (WGSL) implementation on same inputs
// 3. Compare results and report mismatches

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "mlx/mlx.h"

// Include ZK module if available
#ifdef MLX_BUILD_ZK
#include "mlx/zk/zk.h"
#endif

#include "test_vectors.h"

using namespace mlx::core;
using namespace mlx::core::zk::test_vectors;

// =============================================================================
// Utility Functions
// =============================================================================

namespace {

// Convert Fr256 to string for error reporting
std::string fr256_to_hex(const Fr256& elem) {
    std::ostringstream oss;
    oss << "0x";
    for (int i = 7; i >= 0; i--) {
        oss << std::hex << std::setfill('0') << std::setw(8) << elem.limbs[i];
    }
    return oss.str();
}

// Compare two Fr256 elements
bool fr256_equal(const Fr256& a, const Fr256& b) {
    return std::memcmp(a.limbs.data(), b.limbs.data(), sizeof(a.limbs)) == 0;
}

// Convert Fr256 to MLX array (shape: [4] with uint64 type matching zk.h)
array fr256_to_array(const Fr256& elem) {
    // Pack 8 x u32 into 4 x u64 for mlx::core::zk API
    std::vector<uint64_t> data(4);
    for (int i = 0; i < 4; i++) {
        data[i] = ((uint64_t)elem.limbs[i*2 + 1] << 32) | elem.limbs[i*2];
    }
    return array(data.data(), {4}, uint64);
}

// Convert MLX array back to Fr256
Fr256 array_to_fr256(const array& arr) {
    eval(arr);
    auto data = arr.data<uint64_t>();
    Fr256 result;
    for (int i = 0; i < 4; i++) {
        result.limbs[i*2] = (uint32_t)(data[i] & 0xFFFFFFFF);
        result.limbs[i*2 + 1] = (uint32_t)(data[i] >> 32);
    }
    return result;
}

// Convert batch of Fr256 to MLX array (shape: [N, 4])
array fr256_batch_to_array(const std::vector<Fr256>& elems) {
    std::vector<uint64_t> data(elems.size() * 4);
    for (size_t i = 0; i < elems.size(); i++) {
        for (int j = 0; j < 4; j++) {
            data[i * 4 + j] = ((uint64_t)elems[i].limbs[j*2 + 1] << 32) | elems[i].limbs[j*2];
        }
    }
    return array(data.data(), {(int)elems.size(), 4}, uint64);
}

// Convert MLX array batch back to Fr256 vector
std::vector<Fr256> array_to_fr256_batch(const array& arr) {
    eval(arr);
    auto data = arr.data<uint64_t>();
    size_t n = arr.shape(0);
    std::vector<Fr256> result(n);
    for (size_t i = 0; i < n; i++) {
        for (int j = 0; j < 4; j++) {
            result[i].limbs[j*2] = (uint32_t)(data[i * 4 + j] & 0xFFFFFFFF);
            result[i].limbs[j*2 + 1] = (uint32_t)(data[i * 4 + j] >> 32);
        }
    }
    return result;
}

} // anonymous namespace

// =============================================================================
// CPU Reference Implementation
// =============================================================================

namespace cpu_reference {

// BN254 scalar field modulus (little-endian u32)
constexpr std::array<uint32_t, 8> MODULUS = {
    0x43e1f593, 0x79b97091, 0x2833e848, 0x8181585d,
    0xb85045b6, 0xe131a029, 0x64774b84, 0x30644e72
};

// Add two u32 with carry
inline std::pair<uint32_t, uint32_t> add_with_carry(uint32_t a, uint32_t b, uint32_t carry_in) {
    uint64_t sum = (uint64_t)a + (uint64_t)b + (uint64_t)carry_in;
    return {(uint32_t)sum, (uint32_t)(sum >> 32)};
}

// Multiply two u32 to get (lo, hi)
inline std::pair<uint32_t, uint32_t> mul_u32(uint32_t a, uint32_t b) {
    uint64_t product = (uint64_t)a * (uint64_t)b;
    return {(uint32_t)product, (uint32_t)(product >> 32)};
}

// Field addition mod p
Fr256 fr_add(const Fr256& a, const Fr256& b) {
    Fr256 result;
    uint32_t carry = 0;

    for (int i = 0; i < 8; i++) {
        auto [sum, c] = add_with_carry(a.limbs[i], b.limbs[i], carry);
        result.limbs[i] = sum;
        carry = c;
    }

    // Check if >= modulus and reduce
    bool ge_mod = carry > 0;
    if (!ge_mod) {
        for (int i = 7; i >= 0; i--) {
            if (result.limbs[i] < MODULUS[i]) {
                ge_mod = false;
                break;
            }
            if (result.limbs[i] > MODULUS[i]) {
                ge_mod = true;
                break;
            }
        }
    }

    if (ge_mod) {
        uint32_t borrow = 0;
        for (int i = 0; i < 8; i++) {
            int64_t diff = (int64_t)result.limbs[i] - (int64_t)MODULUS[i] - (int64_t)borrow;
            if (diff < 0) {
                result.limbs[i] = (uint32_t)(diff + 0x100000000LL);
                borrow = 1;
            } else {
                result.limbs[i] = (uint32_t)diff;
                borrow = 0;
            }
        }
    }

    return result;
}

// Field reduction if >= modulus
Fr256 fr_reduce(const Fr256& a) {
    bool ge_mod = true;
    for (int i = 7; i >= 0; i--) {
        if (a.limbs[i] < MODULUS[i]) {
            ge_mod = false;
            break;
        }
        if (a.limbs[i] > MODULUS[i]) {
            break;
        }
    }

    if (!ge_mod) {
        return a;
    }

    Fr256 result;
    uint32_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        int64_t diff = (int64_t)a.limbs[i] - (int64_t)MODULUS[i] - (int64_t)borrow;
        if (diff < 0) {
            result.limbs[i] = (uint32_t)(diff + 0x100000000LL);
            borrow = 1;
        } else {
            result.limbs[i] = (uint32_t)diff;
            borrow = 0;
        }
    }

    return result;
}

// Field multiplication (simplified schoolbook)
Fr256 fr_mul(const Fr256& a, const Fr256& b) {
    Fr256 result;
    for (int i = 0; i < 8; i++) {
        result.limbs[i] = 0;
    }

    for (int i = 0; i < 8; i++) {
        uint32_t carry = 0;
        for (int j = 0; j < 8; j++) {
            int idx = i + j;
            if (idx < 8) {
                auto [prod_lo, prod_hi] = mul_u32(a.limbs[i], b.limbs[j]);
                auto [sum1, c1] = add_with_carry(result.limbs[idx], prod_lo, carry);
                auto [sum2, c2] = add_with_carry(sum1, 0, prod_hi);
                result.limbs[idx] = sum2;
                carry = c1 + c2;
            }
        }
    }

    return fr_reduce(result);
}

// Field squaring
Fr256 fr_square(const Fr256& a) {
    return fr_mul(a, a);
}

// S-box: x^5
Fr256 sbox(const Fr256& x) {
    Fr256 x2 = fr_square(x);
    Fr256 x4 = fr_square(x2);
    return fr_mul(x4, x);
}

// MDS matrix multiplication for t=3
// Using circulant matrix [[2,1,1],[1,2,1],[1,1,2]]
void mds_multiply(std::array<Fr256, 3>& state) {
    Fr256 s0 = state[0];
    Fr256 s1 = state[1];
    Fr256 s2 = state[2];

    Fr256 sum = fr_add(fr_add(s0, s1), s2);
    state[0] = fr_add(sum, s0);  // 2*s0 + s1 + s2
    state[1] = fr_add(sum, s1);  // s0 + 2*s1 + s2
    state[2] = fr_add(sum, s2);  // s0 + s1 + 2*s2
}

// Poseidon2 hash (CPU reference)
Fr256 poseidon2_hash(const Fr256& left, const Fr256& right) {
    constexpr int ROUNDS_F = 8;   // Full rounds
    constexpr int ROUNDS_P = 56;  // Partial rounds

    // Initialize state: [left, right, 0]
    std::array<Fr256, 3> state;
    state[0] = left;
    state[1] = right;
    state[2] = Fr256::zero();

    // Full rounds (first half)
    for (int r = 0; r < ROUNDS_F / 2; r++) {
        // Add round constants (simplified: XOR index into limb[0])
        for (int i = 0; i < 3; i++) {
            state[i].limbs[0] ^= (r * 3 + i);
        }
        // S-box on all elements
        state[0] = sbox(state[0]);
        state[1] = sbox(state[1]);
        state[2] = sbox(state[2]);
        // MDS
        mds_multiply(state);
    }

    // Partial rounds (S-box only on first element)
    for (int r = 0; r < ROUNDS_P; r++) {
        state[0].limbs[0] ^= (r + 100);
        state[0] = sbox(state[0]);
        mds_multiply(state);
    }

    // Full rounds (second half)
    for (int r = 0; r < ROUNDS_F / 2; r++) {
        for (int i = 0; i < 3; i++) {
            state[i].limbs[0] ^= (r * 3 + i + 200);
        }
        state[0] = sbox(state[0]);
        state[1] = sbox(state[1]);
        state[2] = sbox(state[2]);
        mds_multiply(state);
    }

    return state[0];
}

// Merkle layer: hash pairs of nodes
std::vector<Fr256> merkle_layer(const std::vector<Fr256>& nodes) {
    std::vector<Fr256> parents;
    parents.reserve(nodes.size() / 2);

    for (size_t i = 0; i < nodes.size(); i += 2) {
        parents.push_back(poseidon2_hash(nodes[i], nodes[i + 1]));
    }

    return parents;
}

// Merkle root computation
Fr256 merkle_root(const std::vector<Fr256>& leaves) {
    if (leaves.size() == 1) {
        return leaves[0];
    }

    std::vector<Fr256> current = leaves;
    while (current.size() > 1) {
        current = merkle_layer(current);
    }

    return current[0];
}

// Commitment: H(H(value, blinding), salt)
Fr256 commitment(const Fr256& value, const Fr256& blinding, const Fr256& salt) {
    Fr256 inner = poseidon2_hash(value, blinding);
    return poseidon2_hash(inner, salt);
}

// Nullifier: H(H(key, commitment), index)
Fr256 nullifier(const Fr256& key, const Fr256& commit, const Fr256& index) {
    Fr256 inner = poseidon2_hash(key, commit);
    return poseidon2_hash(inner, index);
}

} // namespace cpu_reference

// =============================================================================
// Test Cases: CPU Reference Implementation Validation
// =============================================================================

TEST_SUITE("ZK CPU Reference") {

TEST_CASE("Poseidon2 basic properties") {
    // Test 1: Hash of zeros should be deterministic
    Fr256 zero = Fr256::zero();
    Fr256 hash1 = cpu_reference::poseidon2_hash(zero, zero);
    Fr256 hash2 = cpu_reference::poseidon2_hash(zero, zero);
    CHECK(fr256_equal(hash1, hash2));

    // Test 2: Different inputs should produce different outputs
    Fr256 one = Fr256::one();
    Fr256 hash_zero_zero = cpu_reference::poseidon2_hash(zero, zero);
    Fr256 hash_one_zero = cpu_reference::poseidon2_hash(one, zero);
    Fr256 hash_zero_one = cpu_reference::poseidon2_hash(zero, one);
    Fr256 hash_one_one = cpu_reference::poseidon2_hash(one, one);

    CHECK_FALSE(fr256_equal(hash_zero_zero, hash_one_zero));
    CHECK_FALSE(fr256_equal(hash_zero_zero, hash_zero_one));
    CHECK_FALSE(fr256_equal(hash_one_zero, hash_zero_one));
    CHECK_FALSE(fr256_equal(hash_one_zero, hash_one_one));

    // Test 3: Order matters (non-commutative)
    CHECK_FALSE(fr256_equal(hash_one_zero, hash_zero_one));
}

TEST_CASE("Poseidon2 known test vectors") {
    // Note: These tests verify consistency, not cryptographic correctness
    // Real deployment needs test vectors from a verified reference

    auto vectors = get_poseidon2_test_vectors();

    SUBCASE("Hash(0, 0)") {
        Fr256 result = cpu_reference::poseidon2_hash(
            vectors[0].left, vectors[0].right);
        // Output should be non-zero
        CHECK_FALSE(fr256_equal(result, Fr256::zero()));
        MESSAGE("Hash(0,0) = ", fr256_to_hex(result));
    }

    SUBCASE("Hash(1, 0)") {
        Fr256 result = cpu_reference::poseidon2_hash(
            vectors[1].left, vectors[1].right);
        CHECK_FALSE(fr256_equal(result, Fr256::zero()));
        MESSAGE("Hash(1,0) = ", fr256_to_hex(result));
    }

    SUBCASE("Hash(0, 1)") {
        Fr256 result = cpu_reference::poseidon2_hash(
            vectors[2].left, vectors[2].right);
        CHECK_FALSE(fr256_equal(result, Fr256::zero()));
        MESSAGE("Hash(0,1) = ", fr256_to_hex(result));
    }
}

TEST_CASE("Merkle tree basic properties") {
    // Test with power-of-2 leaf counts

    SUBCASE("2 leaves") {
        std::vector<Fr256> leaves = {Fr256::one(), Fr256::one()};
        leaves[1].limbs[0] = 2;

        Fr256 root = cpu_reference::merkle_root(leaves);
        Fr256 expected = cpu_reference::poseidon2_hash(leaves[0], leaves[1]);
        CHECK(fr256_equal(root, expected));
    }

    SUBCASE("4 leaves") {
        std::vector<Fr256> leaves;
        for (uint32_t i = 1; i <= 4; i++) {
            Fr256 elem = Fr256::zero();
            elem.limbs[0] = i;
            leaves.push_back(elem);
        }

        Fr256 root = cpu_reference::merkle_root(leaves);

        // Manual computation
        Fr256 h01 = cpu_reference::poseidon2_hash(leaves[0], leaves[1]);
        Fr256 h23 = cpu_reference::poseidon2_hash(leaves[2], leaves[3]);
        Fr256 expected = cpu_reference::poseidon2_hash(h01, h23);

        CHECK(fr256_equal(root, expected));
    }

    SUBCASE("8 leaves") {
        std::vector<Fr256> leaves;
        for (uint32_t i = 1; i <= 8; i++) {
            Fr256 elem = Fr256::zero();
            elem.limbs[0] = i;
            leaves.push_back(elem);
        }

        Fr256 root = cpu_reference::merkle_root(leaves);
        CHECK_FALSE(fr256_equal(root, Fr256::zero()));
        MESSAGE("Merkle root(1..8) = ", fr256_to_hex(root));
    }
}

TEST_CASE("Commitment and nullifier") {
    Fr256 value = Fr256::one();
    Fr256 blinding = Fr256::zero();
    blinding.limbs[0] = 0x12345678;
    Fr256 salt = Fr256::zero();
    salt.limbs[0] = 0xdeadbeef;

    Fr256 commit = cpu_reference::commitment(value, blinding, salt);

    // Verify commitment is deterministic
    Fr256 commit2 = cpu_reference::commitment(value, blinding, salt);
    CHECK(fr256_equal(commit, commit2));

    // Verify commitment changes with different inputs
    Fr256 different_blinding = blinding;
    different_blinding.limbs[0] ^= 1;
    Fr256 different_commit = cpu_reference::commitment(value, different_blinding, salt);
    CHECK_FALSE(fr256_equal(commit, different_commit));

    // Test nullifier
    Fr256 key = Fr256::zero();
    key.limbs[0] = 0xaabbccdd;
    Fr256 index = Fr256::zero();
    index.limbs[0] = 42;

    Fr256 null = cpu_reference::nullifier(key, commit, index);
    CHECK_FALSE(fr256_equal(null, Fr256::zero()));
}

TEST_CASE("Field arithmetic edge cases") {
    // Test modulus value (which reduces to 0)
    Fr256 mod_value;
    for (int i = 0; i < 8; i++) {
        mod_value.limbs[i] = cpu_reference::MODULUS[i];
    }

    Fr256 one = Fr256::one();
    Fr256 zero = Fr256::zero();

    // Test: modulus + 1 should reduce to 1
    Fr256 sum = cpu_reference::fr_add(mod_value, one);
    CHECK(fr256_equal(sum, Fr256::one()));

    // Test: modulus should reduce to 0
    Fr256 reduced = cpu_reference::fr_reduce(mod_value);
    CHECK(fr256_equal(reduced, Fr256::zero()));

    // Test multiplication by zero
    Fr256 prod = cpu_reference::fr_mul(one, zero);
    CHECK(fr256_equal(prod, Fr256::zero()));

    // Test multiplication by one (identity)
    Fr256 test_val;
    test_val.limbs[0] = 0x12345678;
    test_val.limbs[1] = 0xabcdef01;
    Fr256 prod_one = cpu_reference::fr_mul(test_val, one);
    CHECK(fr256_equal(prod_one, test_val));

    // Test: 1 * 1 = 1
    Fr256 one_squared = cpu_reference::fr_mul(one, one);
    CHECK(fr256_equal(one_squared, one));
}

} // TEST_SUITE("ZK CPU Reference")

// =============================================================================
// Test Cases: WebGPU/WGSL Implementation (when available)
// =============================================================================

#ifdef MLX_BUILD_ZK

TEST_SUITE("ZK GPU Kernels") {

TEST_CASE("GPU backend availability") {
    bool gpu = zk::gpu_available();
    MESSAGE("GPU ZK backend available: ", gpu ? "yes" : "no");
    MESSAGE("Active backend: ", zk::get_backend_name());
}

TEST_CASE("Poseidon2 CPU vs GPU") {
    auto vectors = get_poseidon2_test_vectors();

    for (const auto& vec : vectors) {
        SUBCASE(vec.description) {
            // CPU reference
            Fr256 cpu_result = cpu_reference::poseidon2_hash(vec.left, vec.right);

            // GPU (via MLX zk module)
            array left_arr = fr256_to_array(vec.left);
            array right_arr = fr256_to_array(vec.right);
            array gpu_result_arr = zk::poseidon2_hash(left_arr, right_arr);
            Fr256 gpu_result = array_to_fr256(gpu_result_arr);

            // Compare
            bool match = fr256_equal(cpu_result, gpu_result);
            if (!match) {
                MESSAGE("MISMATCH for ", vec.description);
                MESSAGE("  CPU:  ", fr256_to_hex(cpu_result));
                MESSAGE("  GPU:  ", fr256_to_hex(gpu_result));
            }
            CHECK(match);
        }
    }
}

TEST_CASE("Poseidon2 batch CPU vs GPU") {
    // Generate batch test data
    constexpr size_t batch_size = BATCH_SIZE_MEDIUM;
    auto left_elems = generate_test_elements(batch_size, 0x11111111);
    auto right_elems = generate_test_elements(batch_size, 0x22222222);

    // CPU batch computation
    std::vector<Fr256> cpu_results;
    cpu_results.reserve(batch_size);
    for (size_t i = 0; i < batch_size; i++) {
        cpu_results.push_back(
            cpu_reference::poseidon2_hash(left_elems[i], right_elems[i]));
    }

    // GPU batch computation
    array left_batch = fr256_batch_to_array(left_elems);
    array right_batch = fr256_batch_to_array(right_elems);
    array gpu_batch = zk::poseidon2_batch_hash(left_batch, right_batch);
    std::vector<Fr256> gpu_results = array_to_fr256_batch(gpu_batch);

    // Compare all results
    size_t mismatches = 0;
    for (size_t i = 0; i < batch_size; i++) {
        if (!fr256_equal(cpu_results[i], gpu_results[i])) {
            mismatches++;
            if (mismatches <= 5) {
                MESSAGE("Mismatch at index ", i);
                MESSAGE("  CPU: ", fr256_to_hex(cpu_results[i]));
                MESSAGE("  GPU: ", fr256_to_hex(gpu_results[i]));
            }
        }
    }

    MESSAGE("Batch size: ", batch_size, ", Mismatches: ", mismatches);
    CHECK(mismatches == 0);
}

TEST_CASE("Merkle root CPU vs GPU") {
    auto merkle_vectors = get_merkle_test_vectors();

    for (const auto& vec : merkle_vectors) {
        SUBCASE(vec.description) {
            // CPU reference
            Fr256 cpu_root = cpu_reference::merkle_root(vec.leaves);

            // GPU - need to handle the squeezed array carefully
            array leaves_arr = fr256_batch_to_array(vec.leaves);
            array gpu_root_arr = zk::merkle_root(leaves_arr);

            // Evaluate and ensure shape is [1, 4] for conversion
            gpu_root_arr = reshape(gpu_root_arr, {-1});  // Flatten first
            eval(gpu_root_arr);
            REQUIRE(gpu_root_arr.size() == 4);

            // Read the 4 uint64 values directly
            auto data = gpu_root_arr.data<uint64_t>();
            Fr256 gpu_root;
            for (int i = 0; i < 4; i++) {
                gpu_root.limbs[i*2] = (uint32_t)(data[i] & 0xFFFFFFFF);
                gpu_root.limbs[i*2 + 1] = (uint32_t)(data[i] >> 32);
            }

            bool match = fr256_equal(cpu_root, gpu_root);
            if (!match) {
                MESSAGE("MISMATCH for ", vec.description);
                MESSAGE("  CPU root: ", fr256_to_hex(cpu_root));
                MESSAGE("  GPU root: ", fr256_to_hex(gpu_root));
            }
            CHECK(match);
        }
    }
}

TEST_CASE("Merkle batch CPU vs GPU") {
    // Test with larger trees
    for (size_t leaf_count : {16, 64, 256, 1024}) {
        SUBCASE(("Merkle tree with " + std::to_string(leaf_count) + " leaves").c_str()) {
            auto leaves = generate_test_elements(leaf_count, 0x33333333);

            // CPU
            Fr256 cpu_root = cpu_reference::merkle_root(leaves);

            // GPU
            array leaves_arr = fr256_batch_to_array(leaves);
            array gpu_root_arr = zk::merkle_root(leaves_arr);
            eval(gpu_root_arr);

            // Handle squeezed array (shape [4] instead of [1, 4])
            if (gpu_root_arr.ndim() == 1) {
                gpu_root_arr = reshape(gpu_root_arr, {1, 4});
            }

            std::vector<Fr256> gpu_roots = array_to_fr256_batch(gpu_root_arr);
            REQUIRE(gpu_roots.size() >= 1);
            Fr256 gpu_root = gpu_roots[0];

            bool match = fr256_equal(cpu_root, gpu_root);
            CHECK(match);

            if (!match) {
                MESSAGE("Leaf count: ", leaf_count);
                MESSAGE("  CPU: ", fr256_to_hex(cpu_root));
                MESSAGE("  GPU: ", fr256_to_hex(gpu_root));
            }
        }
    }
}

TEST_CASE("Batch commitment CPU vs GPU") {
    constexpr size_t batch_size = BATCH_SIZE_SMALL;

    auto values = generate_test_elements(batch_size, 0x44444444);
    auto blindings = generate_test_elements(batch_size, 0x55555555);
    auto salts = generate_test_elements(batch_size, 0x66666666);

    // CPU batch
    std::vector<Fr256> cpu_results;
    cpu_results.reserve(batch_size);
    for (size_t i = 0; i < batch_size; i++) {
        cpu_results.push_back(
            cpu_reference::commitment(values[i], blindings[i], salts[i]));
    }

    // GPU batch
    array values_arr = fr256_batch_to_array(values);
    array blindings_arr = fr256_batch_to_array(blindings);
    array salts_arr = fr256_batch_to_array(salts);
    array gpu_batch = zk::batch_commitment(values_arr, blindings_arr, salts_arr);
    std::vector<Fr256> gpu_results = array_to_fr256_batch(gpu_batch);

    // Compare
    size_t mismatches = 0;
    for (size_t i = 0; i < batch_size; i++) {
        if (!fr256_equal(cpu_results[i], gpu_results[i])) {
            mismatches++;
        }
    }

    CHECK(mismatches == 0);
    MESSAGE("Commitment batch: ", batch_size, " items, ", mismatches, " mismatches");
}

TEST_CASE("Batch nullifier CPU vs GPU") {
    constexpr size_t batch_size = BATCH_SIZE_SMALL;

    auto keys = generate_test_elements(batch_size, 0x77777777);
    auto commitments = generate_test_elements(batch_size, 0x88888888);
    auto indices = generate_test_elements(batch_size, 0x99999999);

    // CPU batch
    std::vector<Fr256> cpu_results;
    cpu_results.reserve(batch_size);
    for (size_t i = 0; i < batch_size; i++) {
        cpu_results.push_back(
            cpu_reference::nullifier(keys[i], commitments[i], indices[i]));
    }

    // GPU batch
    array keys_arr = fr256_batch_to_array(keys);
    array commits_arr = fr256_batch_to_array(commitments);
    array indices_arr = fr256_batch_to_array(indices);
    array gpu_batch = zk::batch_nullifier(keys_arr, commits_arr, indices_arr);
    std::vector<Fr256> gpu_results = array_to_fr256_batch(gpu_batch);

    // Compare
    size_t mismatches = 0;
    for (size_t i = 0; i < batch_size; i++) {
        if (!fr256_equal(cpu_results[i], gpu_results[i])) {
            mismatches++;
        }
    }

    CHECK(mismatches == 0);
    MESSAGE("Nullifier batch: ", batch_size, " items, ", mismatches, " mismatches");
}

TEST_CASE("GPU performance scaling") {
    // Test that GPU handles large batches correctly
    for (size_t batch_size : {64, 256, 1024, 4096}) {
        SUBCASE(("Batch size " + std::to_string(batch_size)).c_str()) {
            auto left = generate_test_elements(batch_size, 0xAAAAAAAA);
            auto right = generate_test_elements(batch_size, 0xBBBBBBBB);

            array left_arr = fr256_batch_to_array(left);
            array right_arr = fr256_batch_to_array(right);

            // Just verify no errors occur
            array result = zk::poseidon2_batch_hash(left_arr, right_arr);
            eval(result);

            CHECK(result.shape(0) == (int)batch_size);
            CHECK(result.shape(1) == 4);
        }
    }
}

} // TEST_SUITE("ZK GPU Kernels")

#else // !MLX_BUILD_ZK

TEST_CASE("ZK module not built") {
    MESSAGE("ZK module not enabled. Skipping GPU kernel tests.");
    MESSAGE("Build with MLX_BUILD_ZK=ON to enable ZK tests.");
}

#endif // MLX_BUILD_ZK

// =============================================================================
// Test Cases: WebGPU Direct (using gpu:: namespace)
// =============================================================================

#ifdef LUX_HAVE_WEBGPU

#include "mlx/backend/webgpu/gpu.hpp"

TEST_SUITE("ZK WebGPU Direct") {

// Poseidon2 WGSL shader code (simplified for testing)
const char* POSEIDON2_WGSL = R"(
struct Fr256 {
    limbs: array<u32, 8>
}

@group(0) @binding(0) var<storage, read> input_left: array<Fr256>;
@group(0) @binding(1) var<storage, read> input_right: array<Fr256>;
@group(0) @binding(2) var<storage, read_write> output: array<Fr256>;

fn add_with_carry(a: u32, b: u32, carry_in: u32) -> vec2<u32> {
    let sum_lo = a + b + carry_in;
    var carry_out = 0u;
    if (sum_lo < a || (carry_in > 0u && sum_lo <= a)) {
        carry_out = 1u;
    }
    return vec2<u32>(sum_lo, carry_out);
}

fn fr_add(a: Fr256, b: Fr256) -> Fr256 {
    var result: Fr256;
    var carry: u32 = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let sum = add_with_carry(a.limbs[i], b.limbs[i], carry);
        result.limbs[i] = sum.x;
        carry = sum.y;
    }
    return result;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&input_left)) {
        return;
    }
    // Simplified: just add left + right for basic testing
    output[idx] = fr_add(input_left[idx], input_right[idx]);
}
)";

TEST_CASE("WebGPU context creation") {
    gpu::Context ctx = gpu::createContext();
    CHECK(ctx.device != nullptr);
    CHECK(ctx.queue != nullptr);
    MESSAGE("WebGPU context created successfully");
}

TEST_CASE("WebGPU simple kernel execution") {
    gpu::Context ctx = gpu::createContext();
    REQUIRE(ctx.device != nullptr);

    constexpr size_t N = 256;
    constexpr size_t elem_size = 8 * sizeof(uint32_t);  // Fr256 = 8 x u32

    // Create test data
    std::vector<uint32_t> left_data(N * 8, 0);
    std::vector<uint32_t> right_data(N * 8, 0);

    // Initialize with simple values
    for (size_t i = 0; i < N; i++) {
        left_data[i * 8] = i;      // First limb = index
        right_data[i * 8] = 1;     // First limb = 1
    }

    // Create GPU tensors
    gpu::Tensor left = gpu::createTensor(ctx, {N * 8}, gpu::ki32,
        reinterpret_cast<const int32_t*>(left_data.data()));
    gpu::Tensor right = gpu::createTensor(ctx, {N * 8}, gpu::ki32,
        reinterpret_cast<const int32_t*>(right_data.data()));
    gpu::Tensor output = gpu::createTensor(ctx, {N * 8}, gpu::ki32);

    // Create and run kernel
    gpu::KernelCode code{POSEIDON2_WGSL, 256, gpu::ki32};
    code.entryPoint = "main";

    auto kernel = gpu::createKernel(
        ctx, code,
        gpu::Bindings<3>{left, right, output},
        {gpu::cdiv(N, 256), 1, 1}
    );

    std::promise<void> promise;
    auto future = promise.get_future();
    gpu::dispatchKernel(ctx, kernel, promise);
    gpu::wait(ctx, future);

    // Read back results
    std::vector<uint32_t> result_data(N * 8);
    gpu::toCPU(ctx, output, result_data.data(), N * elem_size);

    // Verify: result[i].limbs[0] should be i + 1
    size_t errors = 0;
    for (size_t i = 0; i < N; i++) {
        uint32_t expected = i + 1;
        uint32_t actual = result_data[i * 8];
        if (actual != expected) {
            errors++;
            if (errors <= 5) {
                MESSAGE("Index ", i, ": expected ", expected, ", got ", actual);
            }
        }
    }

    CHECK(errors == 0);
    MESSAGE("WebGPU kernel test: ", N, " elements, ", errors, " errors");
}

} // TEST_SUITE("ZK WebGPU Direct")

#endif // LUX_HAVE_WEBGPU
