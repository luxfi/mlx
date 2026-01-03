// Copyright © 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0
//
// C API for Zero-Knowledge Operations
// Used by Go bindings via CGO

#ifndef MLX_ZK_C_API_H
#define MLX_ZK_C_API_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Context Management
// =============================================================================

/**
 * Initialize ZK context.
 * @return 0 on success, negative on error
 */
int zk_init(void);

/**
 * Cleanup ZK context.
 */
void zk_cleanup(void);

/**
 * Check if GPU is available.
 * @return true if Metal or CUDA available
 */
bool zk_gpu_available(void);

/**
 * Get backend name.
 * @return "Metal", "CUDA", or "CPU"
 */
const char* zk_get_backend(void);

/**
 * Get threshold for operation type.
 * @param op_type 1=Poseidon2, 2=Merkle, 3=MSM, 4=Commitment, 5=FRI
 * @return Threshold count
 */
int zk_get_threshold(int op_type);

// =============================================================================
// Field Element Type (256-bit, 4 x 64-bit limbs)
// =============================================================================

typedef struct {
    uint64_t limbs[4];
} Fr256;

// =============================================================================
// Poseidon2 Operations
// =============================================================================

/**
 * Batch Poseidon2 hash.
 * @param out Output hashes (count elements)
 * @param left Left inputs (count elements)
 * @param right Right inputs (count elements)
 * @param count Number of hash operations
 * @return 0 on success
 */
int zk_poseidon2_hash(
    Fr256* out,
    const Fr256* left,
    const Fr256* right,
    uint32_t count);

/**
 * Compute Merkle layer.
 * @param out Output parent nodes (count/2 elements)
 * @param nodes Current layer nodes (count elements, must be even)
 * @param count Node count
 * @return 0 on success
 */
int zk_merkle_layer(
    Fr256* out,
    const Fr256* nodes,
    uint32_t count);

/**
 * Compute Merkle root.
 * @param out Output root (single element)
 * @param leaves Input leaves (count elements, must be power of 2)
 * @param count Leaf count
 * @return 0 on success
 */
int zk_merkle_root(
    Fr256* out,
    const Fr256* leaves,
    uint32_t count);

/**
 * Build complete Merkle tree.
 * @param out Output internal nodes (count-1 elements)
 * @param leaves Input leaves (count elements, must be power of 2)
 * @param count Leaf count
 * @return 0 on success
 */
int zk_merkle_tree(
    Fr256* out,
    const Fr256* leaves,
    uint32_t count);

/**
 * Batch commitment.
 * @param out Output commitments (count elements)
 * @param values Input values (count elements)
 * @param blindings Blinding factors (count elements)
 * @param salts Salt values (count elements)
 * @param count Number of commitments
 * @return 0 on success
 */
int zk_batch_commitment(
    Fr256* out,
    const Fr256* values,
    const Fr256* blindings,
    const Fr256* salts,
    uint32_t count);

/**
 * Batch nullifier.
 * @param out Output nullifiers (count elements)
 * @param keys Nullifier keys (count elements)
 * @param commitments Note commitments (count elements)
 * @param indices Leaf indices (count elements)
 * @param count Number of nullifiers
 * @return 0 on success
 */
int zk_batch_nullifier(
    Fr256* out,
    const Fr256* keys,
    const Fr256* commitments,
    const Fr256* indices,
    uint32_t count);

// =============================================================================
// Error Codes
// =============================================================================

#define ZK_SUCCESS 0
#define ZK_ERROR_INVALID_ARG -1
#define ZK_ERROR_SIZE -2
#define ZK_ERROR_ALLOC -3
#define ZK_ERROR_NOT_IMPL -4

// =============================================================================
// Threshold Constants
// =============================================================================

#define ZK_THRESHOLD_POSEIDON2 64
#define ZK_THRESHOLD_MERKLE 128
#define ZK_THRESHOLD_MSM 256
#define ZK_THRESHOLD_COMMITMENT 128
#define ZK_THRESHOLD_FRI 512

#ifdef __cplusplus
}
#endif

#endif // MLX_ZK_C_API_H
