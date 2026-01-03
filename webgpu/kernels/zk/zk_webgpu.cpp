// Copyright (c) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Ecosystem
//
// WebGPU ZK operations implementation using gpu.cpp

#include "zk_webgpu.hpp"
#include <future>
#include <stdexcept>

namespace lux {
namespace zk {
namespace webgpu {

// =============================================================================
// Embedded WGSL Sources
// =============================================================================

const char* kPoseidon2WGSL = R"(
// Poseidon2 batch hash kernel
// See poseidon2.wgsl for full implementation

struct Fr256 {
    limbs: array<u32, 8>
}

@group(0) @binding(0) var<storage, read> input_left: array<Fr256>;
@group(0) @binding(1) var<storage, read> input_right: array<Fr256>;
@group(0) @binding(2) var<storage, read_write> output: array<Fr256>;

const FR_MODULUS: array<u32, 8> = array<u32, 8>(
    0x43e1f593u, 0x79b97091u, 0x2833e848u, 0x8181585du,
    0xb85045b6u, 0xe131a029u, 0x64774b84u, 0x30644e72u
);

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

fn mul_u32(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;
    let ll = a_lo * b_lo;
    let lh = a_lo * b_hi;
    let hl = a_hi * b_lo;
    let hh = a_hi * b_hi;
    let mid = lh + hl;
    let lo = ll + ((mid & 0xFFFFu) << 16u);
    var hi = hh + (mid >> 16u);
    if (lo < ll) { hi = hi + 1u; }
    if (mid < lh) { hi = hi + 0x10000u; }
    return vec2<u32>(lo, hi);
}

fn fr_mul(a: Fr256, b: Fr256) -> Fr256 {
    var result: Fr256;
    for (var i = 0u; i < 8u; i = i + 1u) { result.limbs[i] = 0u; }
    for (var i = 0u; i < 8u; i = i + 1u) {
        var carry: u32 = 0u;
        for (var j = 0u; j < 8u; j = j + 1u) {
            let idx = i + j;
            if (idx < 8u) {
                let prod = mul_u32(a.limbs[i], b.limbs[j]);
                let sum1 = add_with_carry(result.limbs[idx], prod.x, carry);
                let sum2 = add_with_carry(sum1.x, 0u, prod.y);
                result.limbs[idx] = sum2.x;
                carry = sum1.y + sum2.y;
            }
        }
    }
    return result;
}

fn fr_square(a: Fr256) -> Fr256 { return fr_mul(a, a); }

fn sbox(x: Fr256) -> Fr256 {
    let x2 = fr_square(x);
    let x4 = fr_square(x2);
    return fr_mul(x4, x);
}

fn mds_multiply(state: array<Fr256, 3>) -> array<Fr256, 3> {
    var result: array<Fr256, 3>;
    let sum = fr_add(fr_add(state[0], state[1]), state[2]);
    result[0] = fr_add(sum, state[0]);
    result[1] = fr_add(sum, state[1]);
    result[2] = fr_add(sum, state[2]);
    return result;
}

fn poseidon2_hash(left: Fr256, right: Fr256) -> Fr256 {
    var state: array<Fr256, 3>;
    state[0] = left;
    state[1] = right;
    for (var i = 0u; i < 8u; i = i + 1u) { state[2].limbs[i] = 0u; }

    // 4 full rounds
    for (var r = 0u; r < 4u; r = r + 1u) {
        for (var i = 0u; i < 3u; i = i + 1u) {
            state[i].limbs[0] = state[i].limbs[0] ^ (r * 3u + i);
        }
        state[0] = sbox(state[0]);
        state[1] = sbox(state[1]);
        state[2] = sbox(state[2]);
        state = mds_multiply(state);
    }

    // 56 partial rounds
    for (var r = 0u; r < 56u; r = r + 1u) {
        state[0].limbs[0] = state[0].limbs[0] ^ (r + 100u);
        state[0] = sbox(state[0]);
        state = mds_multiply(state);
    }

    // 4 full rounds
    for (var r = 0u; r < 4u; r = r + 1u) {
        for (var i = 0u; i < 3u; i = i + 1u) {
            state[i].limbs[0] = state[i].limbs[0] ^ (r * 3u + i + 200u);
        }
        state[0] = sbox(state[0]);
        state[1] = sbox(state[1]);
        state[2] = sbox(state[2]);
        state = mds_multiply(state);
    }

    return state[0];
}

@compute @workgroup_size({{workgroupSize}})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&input_left)) { return; }
    output[idx] = poseidon2_hash(input_left[idx], input_right[idx]);
}
)";

const char* kMerkleWGSL = R"(
// Merkle layer kernel - hashes pairs of nodes

struct Fr256 {
    limbs: array<u32, 8>
}

@group(0) @binding(0) var<storage, read> nodes: array<Fr256>;
@group(0) @binding(1) var<storage, read_write> parents: array<Fr256>;

// ... (same field arithmetic as poseidon2)

@compute @workgroup_size({{workgroupSize}})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let parent_count = arrayLength(&parents);
    if (idx >= parent_count) { return; }

    let left_idx = idx * 2u;
    let right_idx = left_idx + 1u;
    parents[idx] = poseidon2_hash(nodes[left_idx], nodes[right_idx]);
}
)";

// =============================================================================
// ZKContext Implementation
// =============================================================================

ZKContext::ZKContext() {
    try {
        ctx_ = std::make_unique<gpu::Context>(gpu::createContext());
    } catch (const std::exception& e) {
        ctx_ = nullptr;
    }
}

ZKContext::~ZKContext() = default;

const char* ZKContext::device_name() const {
    // TODO: get from gpu::Context
    return "WebGPU";
}

Fr256 ZKContext::poseidon2_hash(const Fr256& left, const Fr256& right) {
    std::vector<Fr256> l = {left};
    std::vector<Fr256> r = {right};
    auto results = poseidon2_batch_hash(l, r);
    return results.empty() ? Fr256{} : results[0];
}

std::vector<Fr256> ZKContext::poseidon2_batch_hash(
    const std::vector<Fr256>& left,
    const std::vector<Fr256>& right) {

    if (!ctx_ || left.size() != right.size()) {
        return {};
    }

    const size_t n = left.size();
    const size_t bytes = n * sizeof(Fr256);

    // Create GPU tensors
    gpu::Tensor t_left = gpu::createTensor(*ctx_, gpu::Shape{n * 4}, gpu::kf32,
        reinterpret_cast<const float*>(left.data()));
    gpu::Tensor t_right = gpu::createTensor(*ctx_, gpu::Shape{n * 4}, gpu::kf32,
        reinterpret_cast<const float*>(right.data()));
    gpu::Tensor t_output = gpu::createTensor(*ctx_, gpu::Shape{n * 4}, gpu::kf32);

    // Create and dispatch kernel
    gpu::KernelCode code{kPoseidon2WGSL, 256, gpu::kf32};
    gpu::Kernel kernel = gpu::createKernel(*ctx_, code,
        gpu::Bindings<3>{t_left, t_right, t_output},
        {gpu::cdiv(n, 256), 1, 1});

    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    gpu::dispatchKernel(*ctx_, kernel, promise);
    gpu::wait(*ctx_, future);

    // Copy results back
    std::vector<Fr256> results(n);
    gpu::toCPU(*ctx_, t_output, results.data(), bytes);

    return results;
}

std::vector<Fr256> ZKContext::merkle_layer(const std::vector<Fr256>& nodes) {
    if (!ctx_ || nodes.size() < 2 || (nodes.size() & 1)) {
        return {};
    }

    const size_t parent_count = nodes.size() / 2;
    std::vector<Fr256> left, right;
    left.reserve(parent_count);
    right.reserve(parent_count);

    for (size_t i = 0; i < nodes.size(); i += 2) {
        left.push_back(nodes[i]);
        right.push_back(nodes[i + 1]);
    }

    return poseidon2_batch_hash(left, right);
}

Fr256 ZKContext::merkle_root(const std::vector<Fr256>& leaves) {
    if (leaves.empty()) return Fr256{};
    if (leaves.size() == 1) return leaves[0];

    std::vector<Fr256> current = leaves;

    // Pad to power of 2 if needed
    size_t n = current.size();
    size_t pow2 = 1;
    while (pow2 < n) pow2 <<= 1;
    while (current.size() < pow2) {
        current.push_back(Fr256{});
    }

    // Build tree bottom-up
    while (current.size() > 1) {
        current = merkle_layer(current);
    }

    return current[0];
}

std::vector<Fr256> ZKContext::merkle_tree(const std::vector<Fr256>& leaves) {
    if (leaves.size() < 2) return {};

    std::vector<Fr256> current = leaves;

    // Pad to power of 2
    size_t n = current.size();
    size_t pow2 = 1;
    while (pow2 < n) pow2 <<= 1;
    while (current.size() < pow2) {
        current.push_back(Fr256{});
    }

    std::vector<Fr256> tree;
    tree.reserve(current.size() - 1);

    while (current.size() > 1) {
        current = merkle_layer(current);
        for (const auto& node : current) {
            tree.push_back(node);
        }
    }

    return tree;
}

std::vector<Fr256> ZKContext::batch_commitment(
    const std::vector<Fr256>& values,
    const std::vector<Fr256>& blindings,
    const std::vector<Fr256>& salts) {

    if (values.size() != blindings.size() || values.size() != salts.size()) {
        return {};
    }

    // H(H(value, blinding), salt)
    auto inner = poseidon2_batch_hash(values, blindings);
    return poseidon2_batch_hash(inner, salts);
}

std::vector<Fr256> ZKContext::batch_nullifier(
    const std::vector<Fr256>& keys,
    const std::vector<Fr256>& commitments,
    const std::vector<Fr256>& indices) {

    if (keys.size() != commitments.size() || keys.size() != indices.size()) {
        return {};
    }

    // H(H(key, commitment), index)
    auto inner = poseidon2_batch_hash(keys, commitments);
    return poseidon2_batch_hash(inner, indices);
}

// =============================================================================
// Global Instance
// =============================================================================

ZKContext& get_zk_context() {
    static ZKContext ctx;
    return ctx;
}

} // namespace webgpu
} // namespace zk
} // namespace lux
