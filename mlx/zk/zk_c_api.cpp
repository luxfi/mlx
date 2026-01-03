// Copyright © 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0
//
// C API Implementation for Zero-Knowledge Operations

#include "mlx/zk/zk_c_api.h"
#include "mlx/zk/zk.h"
#include "mlx/array.h"

#include <cstring>
#include <vector>

using namespace mlx::core;

// =============================================================================
// Context Management
// =============================================================================

int zk_init(void) {
    // MLX initializes automatically
    return ZK_SUCCESS;
}

void zk_cleanup(void) {
    // Nothing to clean up - MLX handles this
}

bool zk_gpu_available(void) {
    return zk::gpu_available();
}

const char* zk_get_backend(void) {
    return zk::get_backend_name();
}

int zk_get_threshold(int op_type) {
    return zk::get_threshold(op_type);
}

// =============================================================================
// Helper: Convert Fr256 arrays to/from MLX arrays
// =============================================================================

namespace {

// Fr256* (N elements) -> array [N, 4] of uint64
array fr256_to_array(const Fr256* data, uint32_t count) {
    std::vector<int64_t> shape = {static_cast<int64_t>(count), 4};
    return array(reinterpret_cast<const uint64_t*>(data), shape, uint64);
}

// array [N, 4] -> Fr256* (caller allocates)
void array_to_fr256(Fr256* out, const array& arr) {
    auto data = arr.data<uint64_t>();
    size_t count = arr.size() / 4;
    std::memcpy(out, data, count * sizeof(Fr256));
}

// array [4] -> Fr256*
void array_to_fr256_single(Fr256* out, const array& arr) {
    auto data = arr.data<uint64_t>();
    std::memcpy(out, data, sizeof(Fr256));
}

} // anonymous namespace

// =============================================================================
// Poseidon2 Operations
// =============================================================================

int zk_poseidon2_hash(
    Fr256* out,
    const Fr256* left,
    const Fr256* right,
    uint32_t count) {

    if (!out || !left || !right || count == 0) {
        return ZK_ERROR_INVALID_ARG;
    }

    try {
        auto left_arr = fr256_to_array(left, count);
        auto right_arr = fr256_to_array(right, count);

        auto result = zk::poseidon2_batch_hash(left_arr, right_arr);
        eval(result);

        array_to_fr256(out, result);
        return ZK_SUCCESS;
    } catch (...) {
        return ZK_ERROR_ALLOC;
    }
}

int zk_merkle_layer(
    Fr256* out,
    const Fr256* nodes,
    uint32_t count) {

    if (!out || !nodes || count == 0 || count % 2 != 0) {
        return ZK_ERROR_INVALID_ARG;
    }

    try {
        auto nodes_arr = fr256_to_array(nodes, count);
        auto result = zk::merkle_layer(nodes_arr);
        eval(result);

        array_to_fr256(out, result);
        return ZK_SUCCESS;
    } catch (...) {
        return ZK_ERROR_ALLOC;
    }
}

int zk_merkle_root(
    Fr256* out,
    const Fr256* leaves,
    uint32_t count) {

    if (!out || !leaves || count == 0) {
        return ZK_ERROR_INVALID_ARG;
    }

    // Check power of 2
    if ((count & (count - 1)) != 0) {
        return ZK_ERROR_SIZE;
    }

    try {
        auto leaves_arr = fr256_to_array(leaves, count);
        auto result = zk::merkle_root(leaves_arr);
        eval(result);

        array_to_fr256_single(out, result);
        return ZK_SUCCESS;
    } catch (...) {
        return ZK_ERROR_ALLOC;
    }
}

int zk_merkle_tree(
    Fr256* out,
    const Fr256* leaves,
    uint32_t count) {

    if (!out || !leaves || count == 0) {
        return ZK_ERROR_INVALID_ARG;
    }

    // Check power of 2
    if ((count & (count - 1)) != 0) {
        return ZK_ERROR_SIZE;
    }

    try {
        auto leaves_arr = fr256_to_array(leaves, count);
        auto result = zk::merkle_tree(leaves_arr);
        eval(result);

        array_to_fr256(out, result);
        return ZK_SUCCESS;
    } catch (...) {
        return ZK_ERROR_ALLOC;
    }
}

int zk_batch_commitment(
    Fr256* out,
    const Fr256* values,
    const Fr256* blindings,
    const Fr256* salts,
    uint32_t count) {

    if (!out || !values || !blindings || !salts || count == 0) {
        return ZK_ERROR_INVALID_ARG;
    }

    try {
        auto values_arr = fr256_to_array(values, count);
        auto blindings_arr = fr256_to_array(blindings, count);
        auto salts_arr = fr256_to_array(salts, count);

        auto result = zk::batch_commitment(values_arr, blindings_arr, salts_arr);
        eval(result);

        array_to_fr256(out, result);
        return ZK_SUCCESS;
    } catch (...) {
        return ZK_ERROR_ALLOC;
    }
}

int zk_batch_nullifier(
    Fr256* out,
    const Fr256* keys,
    const Fr256* commitments,
    const Fr256* indices,
    uint32_t count) {

    if (!out || !keys || !commitments || !indices || count == 0) {
        return ZK_ERROR_INVALID_ARG;
    }

    try {
        auto keys_arr = fr256_to_array(keys, count);
        auto commitments_arr = fr256_to_array(commitments, count);
        auto indices_arr = fr256_to_array(indices, count);

        auto result = zk::batch_nullifier(keys_arr, commitments_arr, indices_arr);
        eval(result);

        array_to_fr256(out, result);
        return ZK_SUCCESS;
    } catch (...) {
        return ZK_ERROR_ALLOC;
    }
}
