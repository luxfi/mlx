// Copyright (c) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Ecosystem
//
// Unified ZK API - dispatches to best available backend:
//   CUDA (MLX) > Metal (MLX) > WebGPU (gpu.cpp/Dawn) > CPU

#ifndef LUX_GPU_ZK_H
#define LUX_GPU_ZK_H

#include "backend.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Fr256 - 256-bit field element (BN254 scalar field)
// =============================================================================

typedef struct {
    uint64_t limbs[4];  // Little-endian limbs
} LuxFr256;

// =============================================================================
// ZK Context - manages backend selection and state
// =============================================================================

typedef struct LuxZKContext LuxZKContext;

// Create context with auto-selected backend (CUDA > Metal > WebGPU > CPU)
LuxZKContext* lux_zk_create(void);

// Create context with specific backend
LuxZKContext* lux_zk_create_with_backend(LuxBackend backend);

// Destroy context
void lux_zk_destroy(LuxZKContext* ctx);

// Get active backend
LuxBackend lux_zk_backend(const LuxZKContext* ctx);

// Get backend name
const char* lux_zk_backend_name(const LuxZKContext* ctx);

// Check if GPU acceleration is active
bool lux_zk_gpu_enabled(const LuxZKContext* ctx);

// =============================================================================
// Poseidon2 Hash Operations
// =============================================================================

// Single hash: H(left, right) -> out
int lux_zk_poseidon2_hash(
    LuxZKContext* ctx,
    LuxFr256* out,
    const LuxFr256* left,
    const LuxFr256* right
);

// Batch hash: H(left[i], right[i]) -> out[i] for i in [0, count)
int lux_zk_poseidon2_batch_hash(
    LuxZKContext* ctx,
    LuxFr256* out,
    const LuxFr256* left,
    const LuxFr256* right,
    size_t count
);

// =============================================================================
// Merkle Tree Operations
// =============================================================================

// Compute one layer: parents[i] = H(nodes[2i], nodes[2i+1])
// out_count = node_count / 2
int lux_zk_merkle_layer(
    LuxZKContext* ctx,
    LuxFr256* out,
    const LuxFr256* nodes,
    size_t node_count
);

// Compute Merkle root from leaves
int lux_zk_merkle_root(
    LuxZKContext* ctx,
    LuxFr256* root,
    const LuxFr256* leaves,
    size_t leaf_count
);

// Build complete Merkle tree, returns internal nodes (n-1 for n leaves)
// tree must have space for (leaf_count - 1) elements
int lux_zk_merkle_tree(
    LuxZKContext* ctx,
    LuxFr256* tree,
    const LuxFr256* leaves,
    size_t leaf_count
);

// =============================================================================
// Commitment and Nullifier Operations
// =============================================================================

// Batch commitment: H(H(value, blinding), salt)
int lux_zk_batch_commitment(
    LuxZKContext* ctx,
    LuxFr256* out,
    const LuxFr256* values,
    const LuxFr256* blindings,
    const LuxFr256* salts,
    size_t count
);

// Batch nullifier: H(H(key, commitment), index)
int lux_zk_batch_nullifier(
    LuxZKContext* ctx,
    LuxFr256* out,
    const LuxFr256* keys,
    const LuxFr256* commitments,
    const LuxFr256* indices,
    size_t count
);

// =============================================================================
// Threshold Constants (for routing decisions)
// =============================================================================

#define LUX_ZK_THRESHOLD_POSEIDON2   64
#define LUX_ZK_THRESHOLD_MERKLE      128
#define LUX_ZK_THRESHOLD_MSM         256
#define LUX_ZK_THRESHOLD_COMMITMENT  128

// =============================================================================
// Global Context (convenience)
// =============================================================================

// Get global shared context (auto-created on first use)
LuxZKContext* lux_zk_global(void);

// Convenience macros using global context
#define lux_poseidon2_hash(out, l, r) \
    lux_zk_poseidon2_hash(lux_zk_global(), out, l, r)

#define lux_poseidon2_batch(out, l, r, n) \
    lux_zk_poseidon2_batch_hash(lux_zk_global(), out, l, r, n)

#define lux_merkle_root(root, leaves, n) \
    lux_zk_merkle_root(lux_zk_global(), root, leaves, n)

#ifdef __cplusplus
}
#endif

#endif // LUX_GPU_ZK_H
