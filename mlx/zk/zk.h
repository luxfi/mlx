// Copyright © 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Zero-Knowledge Cryptographic Operations for MLX
//
// GPU-accelerated ZK primitives with automatic backend selection:
// - Poseidon2 hash (BN254 scalar field)
// - Multi-scalar multiplication (MSM)
// - Merkle tree operations
// - Commitment/nullifier generation

#pragma once

#include "mlx/array.h"
#include "mlx/ops.h"

namespace mlx::core::zk {

// =============================================================================
// Field Constants (BN254 Scalar Field)
// =============================================================================

// BN254 scalar field modulus: r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
constexpr uint64_t BN254_R[4] = {
    0x43e1f593f0000001ULL,
    0x2833e84879b97091ULL,
    0xb85045b68181585dULL,
    0x30644e72e131a029ULL
};

// =============================================================================
// Threshold Constants (batch sizes for GPU efficiency)
// =============================================================================

constexpr int THRESHOLD_POSEIDON2 = 64;    // GPU faster above this
constexpr int THRESHOLD_MERKLE = 128;      // GPU faster for Merkle layers
constexpr int THRESHOLD_MSM = 256;         // GPU faster for MSM
constexpr int THRESHOLD_COMMITMENT = 128;  // GPU faster for commitments
constexpr int THRESHOLD_FRI = 512;         // GPU faster for FRI folding

// =============================================================================
// Poseidon2 Hash Operations
// =============================================================================

/**
 * Poseidon2 hash of two field elements (2-to-1 compression).
 * Uses BN254 scalar field with optimized round constants.
 *
 * @param left Left input (shape: [..., 4] for 4 limbs)
 * @param right Right input (shape: [..., 4] for 4 limbs)
 * @param s Stream for async execution
 * @return Hash output (shape: [..., 4])
 */
array poseidon2_hash(
    const array& left,
    const array& right,
    StreamOrDevice s = {});

/**
 * Batch Poseidon2 hash of pairs.
 * Optimized for large batches on GPU.
 *
 * @param left Left inputs (shape: [N, 4])
 * @param right Right inputs (shape: [N, 4])
 * @param s Stream for async execution
 * @return Hash outputs (shape: [N, 4])
 */
array poseidon2_batch_hash(
    const array& left,
    const array& right,
    StreamOrDevice s = {});

// =============================================================================
// Merkle Tree Operations
// =============================================================================

/**
 * Compute one layer of Poseidon2 Merkle tree.
 *
 * @param nodes Current layer nodes (shape: [N, 4], N must be even)
 * @param s Stream for async execution
 * @return Parent nodes (shape: [N/2, 4])
 */
array merkle_layer(
    const array& nodes,
    StreamOrDevice s = {});

/**
 * Compute Merkle root from leaves.
 *
 * @param leaves Leaf nodes (shape: [N, 4], N must be power of 2)
 * @param s Stream for async execution
 * @return Merkle root (shape: [4])
 */
array merkle_root(
    const array& leaves,
    StreamOrDevice s = {});

/**
 * Build complete Merkle tree and return all internal nodes.
 *
 * @param leaves Leaf nodes (shape: [N, 4], N must be power of 2)
 * @param s Stream for async execution
 * @return All internal nodes including root (shape: [N-1, 4])
 */
array merkle_tree(
    const array& leaves,
    StreamOrDevice s = {});

// =============================================================================
// Commitment Operations
// =============================================================================

/**
 * Compute Poseidon2-based commitments.
 * commitment = Poseidon2(Poseidon2(value, blinding), salt)
 *
 * @param values Values to commit (shape: [N, 4])
 * @param blindings Blinding factors (shape: [N, 4])
 * @param salts Salt values (shape: [N, 4])
 * @param s Stream for async execution
 * @return Commitments (shape: [N, 4])
 */
array batch_commitment(
    const array& values,
    const array& blindings,
    const array& salts,
    StreamOrDevice s = {});

/**
 * Compute Poseidon2-based nullifiers.
 * nullifier = Poseidon2(Poseidon2(key, commitment), index)
 *
 * @param keys Nullifier keys (shape: [N, 4])
 * @param commitments Note commitments (shape: [N, 4])
 * @param indices Leaf indices (shape: [N, 4])
 * @param s Stream for async execution
 * @return Nullifiers (shape: [N, 4])
 */
array batch_nullifier(
    const array& keys,
    const array& commitments,
    const array& indices,
    StreamOrDevice s = {});

// =============================================================================
// Multi-Scalar Multiplication (MSM)
// =============================================================================

/**
 * Multi-scalar multiplication on BN254 G1.
 * result = sum_i (scalars[i] * points[i])
 *
 * @param points G1 affine points (shape: [N, 2, 4] for x,y coords)
 * @param scalars Scalar field elements (shape: [N, 4])
 * @param s Stream for async execution
 * @return Result point in projective coords (shape: [3, 4] for x,y,z)
 */
array msm(
    const array& points,
    const array& scalars,
    StreamOrDevice s = {});

/**
 * Batch scalar multiplication on BN254 G1.
 * results[i] = scalars[i] * points[i]
 *
 * @param points G1 affine points (shape: [N, 2, 4])
 * @param scalars Scalar field elements (shape: [N, 4])
 * @param s Stream for async execution
 * @return Result points in projective coords (shape: [N, 3, 4])
 */
array batch_scalar_mul(
    const array& points,
    const array& scalars,
    StreamOrDevice s = {});

// =============================================================================
// Goldilocks Field Operations (for STARK FRI)
// =============================================================================

/**
 * FRI folding layer.
 * Folds evaluation points for FRI protocol.
 *
 * @param evals Current layer evaluations (shape: [N])
 * @param alpha Folding challenge (scalar)
 * @param omega_inv Inverse of subgroup generator (scalar)
 * @param s Stream for async execution
 * @return Folded evaluations (shape: [N/2])
 */
array fri_fold_layer(
    const array& evals,
    uint64_t alpha,
    uint64_t omega_inv,
    StreamOrDevice s = {});

/**
 * Batch Goldilocks field multiplication.
 *
 * @param a First operands (shape: [N])
 * @param b Second operands (shape: [N])
 * @param s Stream for async execution
 * @return Products (shape: [N])
 */
array goldilocks_mul(
    const array& a,
    const array& b,
    StreamOrDevice s = {});

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Check if GPU ZK operations are available.
 * @return true if Metal or CUDA backend is available
 */
bool gpu_available();

/**
 * Get the active backend name.
 * @return "Metal", "CUDA", or "CPU"
 */
const char* get_backend_name();

/**
 * Get recommended threshold for an operation type.
 * @param op_type 1=Poseidon2, 2=Merkle, 3=MSM, 4=Commitment, 5=FRI
 * @return Threshold count
 */
int get_threshold(int op_type);

} // namespace mlx::core::zk
