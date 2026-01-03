// Copyright (c) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Ecosystem
//
// WebGPU ZK operations wrapper using gpu.cpp

#ifndef LUX_ZK_WEBGPU_HPP
#define LUX_ZK_WEBGPU_HPP

#include "../../gpu.hpp"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace lux {
namespace zk {
namespace webgpu {

// =============================================================================
// Fr256 Type (matches MLX version)
// =============================================================================

struct Fr256 {
    std::array<uint64_t, 4> limbs;

    Fr256() : limbs{0, 0, 0, 0} {}
    Fr256(uint64_t v) : limbs{v, 0, 0, 0} {}

    bool operator==(const Fr256& other) const {
        return limbs == other.limbs;
    }
};

// =============================================================================
// WGSL Kernel Sources
// =============================================================================

// Embedded WGSL kernels (loaded at compile time or runtime)
extern const char* kPoseidon2WGSL;
extern const char* kMerkleWGSL;

// =============================================================================
// WebGPU ZK Context
// =============================================================================

class ZKContext {
public:
    ZKContext();
    ~ZKContext();

    // Check if WebGPU is available
    bool available() const { return ctx_ != nullptr; }
    const char* device_name() const;

    // Poseidon2 operations
    Fr256 poseidon2_hash(const Fr256& left, const Fr256& right);
    std::vector<Fr256> poseidon2_batch_hash(
        const std::vector<Fr256>& left,
        const std::vector<Fr256>& right);

    // Merkle tree operations
    std::vector<Fr256> merkle_layer(const std::vector<Fr256>& nodes);
    Fr256 merkle_root(const std::vector<Fr256>& leaves);
    std::vector<Fr256> merkle_tree(const std::vector<Fr256>& leaves);

    // Commitment and nullifier
    std::vector<Fr256> batch_commitment(
        const std::vector<Fr256>& values,
        const std::vector<Fr256>& blindings,
        const std::vector<Fr256>& salts);

    std::vector<Fr256> batch_nullifier(
        const std::vector<Fr256>& keys,
        const std::vector<Fr256>& commitments,
        const std::vector<Fr256>& indices);

private:
    std::unique_ptr<gpu::Context> ctx_;

    // Cached kernels
    std::unique_ptr<gpu::Kernel> poseidon2_kernel_;
    std::unique_ptr<gpu::Kernel> merkle_layer_kernel_;
    std::unique_ptr<gpu::Kernel> commitment_kernel_;
    std::unique_ptr<gpu::Kernel> nullifier_kernel_;

    void ensure_poseidon2_kernel();
    void ensure_merkle_layer_kernel();
    void ensure_commitment_kernel();
    void ensure_nullifier_kernel();
};

// =============================================================================
// Global Instance
// =============================================================================

ZKContext& get_zk_context();

// =============================================================================
// Convenience Functions
// =============================================================================

inline bool available() {
    return get_zk_context().available();
}

inline Fr256 poseidon2_hash(const Fr256& left, const Fr256& right) {
    return get_zk_context().poseidon2_hash(left, right);
}

inline std::vector<Fr256> poseidon2_batch_hash(
    const std::vector<Fr256>& left,
    const std::vector<Fr256>& right) {
    return get_zk_context().poseidon2_batch_hash(left, right);
}

inline Fr256 merkle_root(const std::vector<Fr256>& leaves) {
    return get_zk_context().merkle_root(leaves);
}

inline std::vector<Fr256> merkle_tree(const std::vector<Fr256>& leaves) {
    return get_zk_context().merkle_tree(leaves);
}

} // namespace webgpu
} // namespace zk
} // namespace lux

#endif // LUX_ZK_WEBGPU_HPP
