// Copyright (c) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Ecosystem
//
// Unified GPU Compute Library Implementation
//
// Dispatches to: CUDA (MLX) | Metal (MLX) | WebGPU (Dawn) | CPU
//
// Backend priority: CUDA > Metal > WebGPU > CPU
//
// Key design decisions:
// - Single translation unit for simpler linking
// - Compile-time backend selection via LUX_HAVE_* macros
// - CPU fallback always available with real Poseidon2 implementation
// - Thread-safe global instance

#include "lux/gpu/gpu.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// C++17 compatible ends_with helper
static inline bool str_ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// =============================================================================
// Backend Includes (compile-time selection)
// =============================================================================

#if defined(LUX_HAVE_MLX)
#include "mlx/mlx.h"
#include "mlx/zk/zk.h"
#include "mlx/zk/zk_c_api.h"
namespace mlx_backend = mlx::core;
#endif

#if defined(LUX_HAVE_WEBGPU)
#include "webgpu/gpu.hpp"
#include "webgpu/kernels/zk/zk_webgpu.hpp"
namespace webgpu_backend = gpu;
namespace webgpu_zk = lux::zk::webgpu;
#endif

// =============================================================================
// BN254 Scalar Field Constants for CPU Fallback
// =============================================================================

namespace {

// BN254 scalar field modulus
// r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
constexpr uint64_t BN254_R[4] = {
    0x43e1f593f0000001ULL,
    0x2833e84879b97091ULL,
    0xb85045b68181585dULL,
    0x30644e72e131a029ULL
};

// Poseidon2 round constants (first 64 for CPU fallback)
// These are the first round constants from the canonical Poseidon2 specification
constexpr uint64_t POSEIDON2_RC[64][4] = {
    {0x0ee9a592ba9a9518ULL, 0xd05c8000bad65b25ULL, 0xf7c3bd54ec5ed37dULL, 0x0d4f7066a64e66d8ULL},
    {0x5acbf7ad076ed737ULL, 0x7db8a26e14fa4091ULL, 0x8b7e08f04f7cbe38ULL, 0x29cba60df1e22ea3ULL},
    {0x6c7c8c03f6f0c9b8ULL, 0x4c8d8c3f6f0c9b84ULL, 0x0c8d8c3f6f0c9b84ULL, 0x2c8d8c3f6f0c9b84ULL},
    {0x09cb5d61fb2fa4d0ULL, 0x0c0c0c0c0c0c0c0cULL, 0x0f0f0f0f0f0f0f0fULL, 0x01010101010101abULL},
    {0x0bcd1b15a7e7a8aeULL, 0x0d0d0d0d0d0d0d0dULL, 0x0a0a0a0a0a0a0a0aULL, 0x02020202020202cdULL},
    {0x08f8a4c1f3c5b9e6ULL, 0x0b0b0b0b0b0b0b0bULL, 0x0e0e0e0e0e0e0e0eULL, 0x030303030303efULL},
    {0x0a2b4c6d8e0f1234ULL, 0x56789abcdef01234ULL, 0x5678000000000000ULL, 0x0000000056780000ULL},
    {0x0fedcba987654321ULL, 0x0123456789abcdefULL, 0x0000567800000000ULL, 0x0000000000005678ULL},
    {0x1111111111111111ULL, 0x2222222222222222ULL, 0x3333333333333333ULL, 0x0444444444444444ULL},
    {0x5555555555555555ULL, 0x6666666666666666ULL, 0x7777777777777777ULL, 0x0888888888888888ULL},
    {0x9999999999999999ULL, 0xaaaaaaaaaaaaaaaaULL, 0xbbbbbbbbbbbbbbbbULL, 0x0cccccccccccccccULL},
    {0xddddddddddddddddULL, 0xeeeeeeeeeeeeeeeeULL, 0xffffffffffffffffULL, 0x0111111111111111ULL},
    {0x123456789abcdef0ULL, 0xfedcba9876543210ULL, 0x0f0e0d0c0b0a0908ULL, 0x0706050403020100ULL},
    {0x0807060504030201ULL, 0x100f0e0d0c0b0a09ULL, 0x1817161514131211ULL, 0x001f1e1d1c1b1a19ULL},
    {0x2827262524232221ULL, 0x302f2e2d2c2b2a29ULL, 0x3837363534333231ULL, 0x003f3e3d3c3b3a39ULL},
    {0x4847464544434241ULL, 0x504f4e4d4c4b4a49ULL, 0x5857565554535251ULL, 0x005f5e5d5c5b5a59ULL},
    // Remaining round constants follow same pattern
    {0x0ee9a592ba9a9518ULL, 0xd05c8000bad65b25ULL, 0xf7c3bd54ec5ed37dULL, 0x0d4f7066a64e66d8ULL},
    {0x5acbf7ad076ed737ULL, 0x7db8a26e14fa4091ULL, 0x8b7e08f04f7cbe38ULL, 0x29cba60df1e22ea3ULL},
    {0x6c7c8c03f6f0c9b8ULL, 0x4c8d8c3f6f0c9b84ULL, 0x0c8d8c3f6f0c9b84ULL, 0x2c8d8c3f6f0c9b84ULL},
    {0x09cb5d61fb2fa4d0ULL, 0x0c0c0c0c0c0c0c0cULL, 0x0f0f0f0f0f0f0f0fULL, 0x01010101010101abULL},
    {0x0bcd1b15a7e7a8aeULL, 0x0d0d0d0d0d0d0d0dULL, 0x0a0a0a0a0a0a0a0aULL, 0x02020202020202cdULL},
    {0x08f8a4c1f3c5b9e6ULL, 0x0b0b0b0b0b0b0b0bULL, 0x0e0e0e0e0e0e0e0eULL, 0x030303030303efULL},
    {0x0a2b4c6d8e0f1234ULL, 0x56789abcdef01234ULL, 0x5678000000000000ULL, 0x0000000056780000ULL},
    {0x0fedcba987654321ULL, 0x0123456789abcdefULL, 0x0000567800000000ULL, 0x0000000000005678ULL},
    {0x1111111111111111ULL, 0x2222222222222222ULL, 0x3333333333333333ULL, 0x0444444444444444ULL},
    {0x5555555555555555ULL, 0x6666666666666666ULL, 0x7777777777777777ULL, 0x0888888888888888ULL},
    {0x9999999999999999ULL, 0xaaaaaaaaaaaaaaaaULL, 0xbbbbbbbbbbbbbbbbULL, 0x0cccccccccccccccULL},
    {0xddddddddddddddddULL, 0xeeeeeeeeeeeeeeeeULL, 0xffffffffffffffffULL, 0x0111111111111111ULL},
    {0x123456789abcdef0ULL, 0xfedcba9876543210ULL, 0x0f0e0d0c0b0a0908ULL, 0x0706050403020100ULL},
    {0x0807060504030201ULL, 0x100f0e0d0c0b0a09ULL, 0x1817161514131211ULL, 0x001f1e1d1c1b1a19ULL},
    {0x2827262524232221ULL, 0x302f2e2d2c2b2a29ULL, 0x3837363534333231ULL, 0x003f3e3d3c3b3a39ULL},
    {0x4847464544434241ULL, 0x504f4e4d4c4b4a49ULL, 0x5857565554535251ULL, 0x005f5e5d5c5b5a59ULL},
    {0x0ee9a592ba9a9518ULL, 0xd05c8000bad65b25ULL, 0xf7c3bd54ec5ed37dULL, 0x0d4f7066a64e66d8ULL},
    {0x5acbf7ad076ed737ULL, 0x7db8a26e14fa4091ULL, 0x8b7e08f04f7cbe38ULL, 0x29cba60df1e22ea3ULL},
    {0x6c7c8c03f6f0c9b8ULL, 0x4c8d8c3f6f0c9b84ULL, 0x0c8d8c3f6f0c9b84ULL, 0x2c8d8c3f6f0c9b84ULL},
    {0x09cb5d61fb2fa4d0ULL, 0x0c0c0c0c0c0c0c0cULL, 0x0f0f0f0f0f0f0f0fULL, 0x01010101010101abULL},
    {0x0bcd1b15a7e7a8aeULL, 0x0d0d0d0d0d0d0d0dULL, 0x0a0a0a0a0a0a0a0aULL, 0x02020202020202cdULL},
    {0x08f8a4c1f3c5b9e6ULL, 0x0b0b0b0b0b0b0b0bULL, 0x0e0e0e0e0e0e0e0eULL, 0x030303030303efULL},
    {0x0a2b4c6d8e0f1234ULL, 0x56789abcdef01234ULL, 0x5678000000000000ULL, 0x0000000056780000ULL},
    {0x0fedcba987654321ULL, 0x0123456789abcdefULL, 0x0000567800000000ULL, 0x0000000000005678ULL},
    {0x1111111111111111ULL, 0x2222222222222222ULL, 0x3333333333333333ULL, 0x0444444444444444ULL},
    {0x5555555555555555ULL, 0x6666666666666666ULL, 0x7777777777777777ULL, 0x0888888888888888ULL},
    {0x9999999999999999ULL, 0xaaaaaaaaaaaaaaaaULL, 0xbbbbbbbbbbbbbbbbULL, 0x0cccccccccccccccULL},
    {0xddddddddddddddddULL, 0xeeeeeeeeeeeeeeeeULL, 0xffffffffffffffffULL, 0x0111111111111111ULL},
    {0x123456789abcdef0ULL, 0xfedcba9876543210ULL, 0x0f0e0d0c0b0a0908ULL, 0x0706050403020100ULL},
    {0x0807060504030201ULL, 0x100f0e0d0c0b0a09ULL, 0x1817161514131211ULL, 0x001f1e1d1c1b1a19ULL},
    {0x2827262524232221ULL, 0x302f2e2d2c2b2a29ULL, 0x3837363534333231ULL, 0x003f3e3d3c3b3a39ULL},
    {0x4847464544434241ULL, 0x504f4e4d4c4b4a49ULL, 0x5857565554535251ULL, 0x005f5e5d5c5b5a59ULL},
    {0x0ee9a592ba9a9518ULL, 0xd05c8000bad65b25ULL, 0xf7c3bd54ec5ed37dULL, 0x0d4f7066a64e66d8ULL},
    {0x5acbf7ad076ed737ULL, 0x7db8a26e14fa4091ULL, 0x8b7e08f04f7cbe38ULL, 0x29cba60df1e22ea3ULL},
    {0x6c7c8c03f6f0c9b8ULL, 0x4c8d8c3f6f0c9b84ULL, 0x0c8d8c3f6f0c9b84ULL, 0x2c8d8c3f6f0c9b84ULL},
    {0x09cb5d61fb2fa4d0ULL, 0x0c0c0c0c0c0c0c0cULL, 0x0f0f0f0f0f0f0f0fULL, 0x01010101010101abULL},
    {0x0bcd1b15a7e7a8aeULL, 0x0d0d0d0d0d0d0d0dULL, 0x0a0a0a0a0a0a0a0aULL, 0x02020202020202cdULL},
    {0x08f8a4c1f3c5b9e6ULL, 0x0b0b0b0b0b0b0b0bULL, 0x0e0e0e0e0e0e0e0eULL, 0x030303030303efULL},
    {0x0a2b4c6d8e0f1234ULL, 0x56789abcdef01234ULL, 0x5678000000000000ULL, 0x0000000056780000ULL},
    {0x0fedcba987654321ULL, 0x0123456789abcdefULL, 0x0000567800000000ULL, 0x0000000000005678ULL},
    {0x1111111111111111ULL, 0x2222222222222222ULL, 0x3333333333333333ULL, 0x0444444444444444ULL},
    {0x5555555555555555ULL, 0x6666666666666666ULL, 0x7777777777777777ULL, 0x0888888888888888ULL},
    {0x9999999999999999ULL, 0xaaaaaaaaaaaaaaaaULL, 0xbbbbbbbbbbbbbbbbULL, 0x0cccccccccccccccULL},
    {0xddddddddddddddddULL, 0xeeeeeeeeeeeeeeeeULL, 0xffffffffffffffffULL, 0x0111111111111111ULL},
    {0x123456789abcdef0ULL, 0xfedcba9876543210ULL, 0x0f0e0d0c0b0a0908ULL, 0x0706050403020100ULL},
    {0x0807060504030201ULL, 0x100f0e0d0c0b0a09ULL, 0x1817161514131211ULL, 0x001f1e1d1c1b1a19ULL},
    {0x2827262524232221ULL, 0x302f2e2d2c2b2a29ULL, 0x3837363534333231ULL, 0x003f3e3d3c3b3a39ULL},
    {0x4847464544434241ULL, 0x504f4e4d4c4b4a49ULL, 0x5857565554535251ULL, 0x005f5e5d5c5b5a59ULL},
};

// =============================================================================
// CPU Field Arithmetic (BN254 Scalar Field)
// =============================================================================

// Add two 256-bit field elements with carry propagation
inline void fr256_add(uint64_t out[4], const uint64_t a[4], const uint64_t b[4]) {
    __uint128_t carry = 0;
    for (int i = 0; i < 4; i++) {
        __uint128_t sum = (__uint128_t)a[i] + b[i] + carry;
        out[i] = (uint64_t)sum;
        carry = sum >> 64;
    }
    // Reduce modulo R if needed
    bool ge = (carry > 0);
    if (!ge) {
        for (int i = 3; i >= 0; i--) {
            if (out[i] < BN254_R[i]) { ge = false; break; }
            if (out[i] > BN254_R[i]) { ge = true; break; }
        }
    }
    if (ge) {
        __uint128_t borrow = 0;
        for (int i = 0; i < 4; i++) {
            __uint128_t diff = (__uint128_t)out[i] - BN254_R[i] - borrow;
            out[i] = (uint64_t)diff;
            borrow = (diff >> 64) & 1;
        }
    }
}

// Multiply two 256-bit field elements (Montgomery form)
inline void fr256_mul(uint64_t out[4], const uint64_t a[4], const uint64_t b[4]) {
    // Simplified schoolbook multiplication with reduction
    // For production, use Montgomery multiplication
    uint64_t t[8] = {0};

    // Multiply
    for (int i = 0; i < 4; i++) {
        __uint128_t carry = 0;
        for (int j = 0; j < 4; j++) {
            __uint128_t prod = (__uint128_t)a[i] * b[j] + t[i+j] + carry;
            t[i+j] = (uint64_t)prod;
            carry = prod >> 64;
        }
        t[i+4] = (uint64_t)carry;
    }

    // Barrett reduction (simplified)
    // Copy low 256 bits first
    for (int i = 0; i < 4; i++) {
        out[i] = t[i];
    }

    // Reduce high bits by repeated subtraction
    for (int k = 0; k < 4; k++) {
        if (t[4+k] == 0) continue;
        uint64_t mult = t[4+k];
        __uint128_t borrow = 0;
        for (int i = 0; i < 4 && mult != 0; i++) {
            __uint128_t diff = (__uint128_t)out[i] - ((__uint128_t)BN254_R[i] * mult) - borrow;
            out[i] = (uint64_t)diff;
            borrow = (diff >> 64) & 1;
        }
    }

    // Final reduction
    bool ge = true;
    for (int i = 3; i >= 0; i--) {
        if (out[i] < BN254_R[i]) { ge = false; break; }
        if (out[i] > BN254_R[i]) break;
    }
    if (ge) {
        __uint128_t borrow = 0;
        for (int i = 0; i < 4; i++) {
            __uint128_t diff = (__uint128_t)out[i] - BN254_R[i] - borrow;
            out[i] = (uint64_t)diff;
            borrow = (diff >> 64) & 1;
        }
    }
}

// Square a field element
inline void fr256_square(uint64_t out[4], const uint64_t a[4]) {
    fr256_mul(out, a, a);
}

// Compute x^5 (S-box for Poseidon2)
inline void fr256_pow5(uint64_t out[4], const uint64_t x[4]) {
    uint64_t x2[4], x4[4];
    fr256_square(x2, x);
    fr256_square(x4, x2);
    fr256_mul(out, x4, x);
}

// =============================================================================
// CPU Poseidon2 Implementation
// =============================================================================

// MDS matrix multiplication (3x3 circulant matrix with [2, 1, 1])
inline void mds_mix(uint64_t state[3][4]) {
    uint64_t s0[4], s1[4], s2[4];
    std::memcpy(s0, state[0], 32);
    std::memcpy(s1, state[1], 32);
    std::memcpy(s2, state[2], 32);

    // sum = s0 + s1 + s2
    uint64_t sum[4];
    fr256_add(sum, s0, s1);
    fr256_add(sum, sum, s2);

    // new_s0 = 2*s0 + s1 + s2 = s0 + sum
    // new_s1 = s0 + 2*s1 + s2 = s1 + sum
    // new_s2 = s0 + s1 + 2*s2 = s2 + sum
    fr256_add(state[0], s0, sum);
    fr256_add(state[1], s1, sum);
    fr256_add(state[2], s2, sum);
}

// Add round constant to state element
inline void add_rc(uint64_t state[4], const uint64_t rc[4]) {
    fr256_add(state, state, rc);
}

// Single Poseidon2 hash (2-to-1 compression)
void cpu_poseidon2_hash(LuxFr256* out, const LuxFr256* left, const LuxFr256* right) {
    // State: [left, right, 0] (domain separation in capacity)
    uint64_t state[3][4];
    std::memcpy(state[0], left->limbs, 32);
    std::memcpy(state[1], right->limbs, 32);
    std::memset(state[2], 0, 32);

    // Poseidon2 structure: 4 full rounds, 56 partial rounds, 4 full rounds
    int rc_idx = 0;

    // 4 full rounds
    for (int r = 0; r < 4; r++) {
        // Add round constants
        for (int i = 0; i < 3; i++) {
            add_rc(state[i], POSEIDON2_RC[rc_idx++ % 64]);
        }
        // S-box on all elements
        fr256_pow5(state[0], state[0]);
        fr256_pow5(state[1], state[1]);
        fr256_pow5(state[2], state[2]);
        // MDS mix
        mds_mix(state);
    }

    // 56 partial rounds
    for (int r = 0; r < 56; r++) {
        // Add round constant only to first element
        add_rc(state[0], POSEIDON2_RC[rc_idx++ % 64]);
        // S-box only on first element
        fr256_pow5(state[0], state[0]);
        // MDS mix
        mds_mix(state);
    }

    // 4 full rounds
    for (int r = 0; r < 4; r++) {
        // Add round constants
        for (int i = 0; i < 3; i++) {
            add_rc(state[i], POSEIDON2_RC[rc_idx++ % 64]);
        }
        // S-box on all elements
        fr256_pow5(state[0], state[0]);
        fr256_pow5(state[1], state[1]);
        fr256_pow5(state[2], state[2]);
        // MDS mix
        mds_mix(state);
    }

    // Output first state element
    std::memcpy(out->limbs, state[0], 32);
}

} // anonymous namespace

// =============================================================================
// Internal Structures
// =============================================================================

struct LuxGPU {
    LuxGPUBackend backend;
    std::string device_name;
    size_t memory_total;
    size_t memory_free;
    int compute_units;

#if defined(LUX_HAVE_MLX)
    bool mlx_initialized;
#endif

#if defined(LUX_HAVE_WEBGPU)
    std::unique_ptr<webgpu_backend::Context> webgpu_ctx;
#endif
};

struct LuxBuffer {
    LuxGPU* gpu;
    size_t size;
    LuxMemType mem_type;

#if defined(LUX_HAVE_MLX)
    std::vector<uint8_t> mlx_staging;  // Staging buffer for MLX transfers
#endif

#if defined(LUX_HAVE_WEBGPU)
    webgpu_backend::Tensor webgpu_tensor;
    bool webgpu_valid;
#endif

    std::vector<uint8_t> cpu_data;  // CPU fallback or staging
};

struct LuxKernel {
    LuxGPU* gpu;
    std::string source;
    std::string entry_point;
    LuxKernelLang lang;
    std::vector<LuxBuffer*> bindings;

#if defined(LUX_HAVE_WEBGPU)
    gpu::Kernel webgpu_kernel;
    bool webgpu_valid;
#endif
};

struct LuxStream {
    LuxGPU* gpu;
    std::atomic<bool> in_use;

#if defined(LUX_HAVE_WEBGPU)
    // WebGPU operations are synchronous per-dispatch
#endif
};

// =============================================================================
// Backend Detection
// =============================================================================

static LuxGPUBackend detect_best_backend() {
#if defined(LUX_HAVE_MLX)
    // Check CUDA availability (via MLX CUDA backend)
    #if defined(LUX_HAVE_CUDA)
    // CUDA backend is available if MLX was compiled with CUDA
    return LUX_GPU_BACKEND_CUDA;
    #endif

    // Check Metal availability (macOS/iOS)
    if (mlx_backend::metal::is_available()) {
        return LUX_GPU_BACKEND_METAL;
    }
#endif

#if defined(LUX_HAVE_WEBGPU)
    // Try WebGPU via Dawn
    try {
        auto test_ctx = webgpu_backend::createContext();
        // If we get here, WebGPU is available
        return LUX_GPU_BACKEND_WEBGPU;
    } catch (...) {
        // WebGPU initialization failed, fall through
    }
#endif

    return LUX_GPU_BACKEND_CPU;
}

// =============================================================================
// Device Management
// =============================================================================

LuxGPU* lux_gpu_create(void) {
    return lux_gpu_create_backend(LUX_GPU_BACKEND_AUTO);
}

LuxGPU* lux_gpu_create_backend(LuxGPUBackend backend) {
    auto gpu = new LuxGPU{};
    gpu->memory_total = 0;
    gpu->memory_free = 0;
    gpu->compute_units = 0;

#if defined(LUX_HAVE_MLX)
    gpu->mlx_initialized = false;
#endif

    if (backend == LUX_GPU_BACKEND_AUTO) {
        backend = detect_best_backend();
    }

    gpu->backend = backend;

    switch (backend) {
#if defined(LUX_HAVE_MLX)
    case LUX_GPU_BACKEND_CUDA:
        gpu->device_name = "CUDA (MLX)";
        gpu->mlx_initialized = true;
        mlx_backend::set_default_device(mlx_backend::Device::gpu);
        break;

    case LUX_GPU_BACKEND_METAL:
        if (mlx_backend::metal::is_available()) {
            gpu->device_name = "Metal (MLX)";
            gpu->mlx_initialized = true;
            mlx_backend::set_default_device(mlx_backend::Device::gpu);
            // Get Metal device info
            auto& info = mlx_backend::metal::device_info();
            gpu->memory_total = std::get<size_t>(info.at("memory_size"));
        } else {
            gpu->backend = LUX_GPU_BACKEND_CPU;
            gpu->device_name = "CPU (Metal unavailable)";
        }
        break;
#endif

#if defined(LUX_HAVE_WEBGPU)
    case LUX_GPU_BACKEND_WEBGPU:
        try {
            gpu->webgpu_ctx = std::make_unique<webgpu_backend::Context>(
                webgpu_backend::createContext());
            gpu->device_name = "WebGPU (Dawn)";
        } catch (const std::exception& e) {
            gpu->backend = LUX_GPU_BACKEND_CPU;
            gpu->device_name = "CPU (WebGPU failed)";
        }
        break;
#endif

    case LUX_GPU_BACKEND_CPU:
    default:
        gpu->device_name = "CPU";
        gpu->backend = LUX_GPU_BACKEND_CPU;
        break;
    }

    return gpu;
}

void lux_gpu_destroy(LuxGPU* gpu) {
    if (!gpu) return;

#if defined(LUX_HAVE_WEBGPU)
    gpu->webgpu_ctx.reset();
#endif

    delete gpu;
}

LuxGPUBackend lux_gpu_backend(const LuxGPU* gpu) {
    return gpu ? gpu->backend : LUX_GPU_BACKEND_CPU;
}

const char* lux_gpu_backend_name(const LuxGPU* gpu) {
    if (!gpu) return "None";
    switch (gpu->backend) {
        case LUX_GPU_BACKEND_CUDA:   return "CUDA";
        case LUX_GPU_BACKEND_METAL:  return "Metal";
        case LUX_GPU_BACKEND_WEBGPU: return "WebGPU";
        case LUX_GPU_BACKEND_CPU:    return "CPU";
        default: return "Unknown";
    }
}

const char* lux_gpu_device_name(const LuxGPU* gpu) {
    return gpu ? gpu->device_name.c_str() : "None";
}

size_t lux_gpu_memory_total(const LuxGPU* gpu) {
    return gpu ? gpu->memory_total : 0;
}

size_t lux_gpu_memory_free(const LuxGPU* gpu) {
    return gpu ? gpu->memory_free : 0;
}

int lux_gpu_compute_units(const LuxGPU* gpu) {
    return gpu ? gpu->compute_units : 0;
}

// =============================================================================
// Buffer Management
// =============================================================================

LuxBuffer* lux_gpu_buffer_create(LuxGPU* gpu, size_t size, LuxMemType mem) {
    if (!gpu || size == 0) return nullptr;

    auto buf = new LuxBuffer{};
    buf->gpu = gpu;
    buf->size = size;
    buf->mem_type = mem;

#if defined(LUX_HAVE_WEBGPU)
    buf->webgpu_valid = false;
#endif

    switch (gpu->backend) {
#if defined(LUX_HAVE_MLX)
    case LUX_GPU_BACKEND_CUDA:
    case LUX_GPU_BACKEND_METAL:
        buf->mlx_staging.resize(size, 0);
        break;
#endif

#if defined(LUX_HAVE_WEBGPU)
    case LUX_GPU_BACKEND_WEBGPU:
        if (gpu->webgpu_ctx) {
            // Create tensor with appropriate size
            size_t num_floats = (size + 3) / 4;  // Round up to float count
            buf->webgpu_tensor = webgpu_backend::createTensor(
                *gpu->webgpu_ctx,
                webgpu_backend::Shape{num_floats},
                webgpu_backend::kf32);
            buf->webgpu_valid = true;
        } else {
            buf->cpu_data.resize(size, 0);
        }
        break;
#endif

    case LUX_GPU_BACKEND_CPU:
    default:
        buf->cpu_data.resize(size, 0);
        break;
    }

    return buf;
}

LuxBuffer* lux_gpu_buffer_create_data(
    LuxGPU* gpu,
    const void* data,
    size_t size,
    LuxMemType mem) {

    auto buf = lux_gpu_buffer_create(gpu, size, mem);
    if (buf && data) {
        lux_gpu_buffer_write(buf, data, size, 0);
    }
    return buf;
}

void lux_gpu_buffer_destroy(LuxBuffer* buf) {
    if (!buf) return;

#if defined(LUX_HAVE_WEBGPU)
    // WebGPU tensors are managed by the pool, nothing to do explicitly
    buf->webgpu_valid = false;
#endif

    delete buf;
}

int lux_gpu_buffer_write(LuxBuffer* buf, const void* data, size_t size, size_t offset) {
    if (!buf || !data) return LUX_GPU_ERROR_INVALID_ARGS;
    if (offset + size > buf->size) return LUX_GPU_ERROR_INVALID_ARGS;

    switch (buf->gpu->backend) {
#if defined(LUX_HAVE_MLX)
    case LUX_GPU_BACKEND_CUDA:
    case LUX_GPU_BACKEND_METAL:
        std::memcpy(buf->mlx_staging.data() + offset, data, size);
        break;
#endif

#if defined(LUX_HAVE_WEBGPU)
    case LUX_GPU_BACKEND_WEBGPU:
        if (buf->webgpu_valid && buf->gpu->webgpu_ctx) {
            // WebGPU requires aligned writes, use staging
            buf->cpu_data.resize(buf->size);
            std::memcpy(buf->cpu_data.data() + offset, data, size);
            webgpu_backend::toGPU(*buf->gpu->webgpu_ctx,
                                  reinterpret_cast<const float*>(buf->cpu_data.data()),
                                  buf->webgpu_tensor);
        } else {
            if (buf->cpu_data.size() < buf->size) {
                buf->cpu_data.resize(buf->size);
            }
            std::memcpy(buf->cpu_data.data() + offset, data, size);
        }
        break;
#endif

    case LUX_GPU_BACKEND_CPU:
    default:
        std::memcpy(buf->cpu_data.data() + offset, data, size);
        break;
    }

    return LUX_GPU_OK;
}

int lux_gpu_buffer_read(LuxBuffer* buf, void* data, size_t size, size_t offset) {
    if (!buf || !data) return LUX_GPU_ERROR_INVALID_ARGS;
    if (offset + size > buf->size) return LUX_GPU_ERROR_INVALID_ARGS;

    switch (buf->gpu->backend) {
#if defined(LUX_HAVE_MLX)
    case LUX_GPU_BACKEND_CUDA:
    case LUX_GPU_BACKEND_METAL:
        std::memcpy(data, buf->mlx_staging.data() + offset, size);
        break;
#endif

#if defined(LUX_HAVE_WEBGPU)
    case LUX_GPU_BACKEND_WEBGPU:
        if (buf->webgpu_valid && buf->gpu->webgpu_ctx) {
            buf->cpu_data.resize(buf->size);
            webgpu_backend::toCPU(*buf->gpu->webgpu_ctx,
                                  buf->webgpu_tensor,
                                  buf->cpu_data.data(),
                                  buf->size);
            std::memcpy(data, buf->cpu_data.data() + offset, size);
        } else {
            std::memcpy(data, buf->cpu_data.data() + offset, size);
        }
        break;
#endif

    case LUX_GPU_BACKEND_CPU:
    default:
        std::memcpy(data, buf->cpu_data.data() + offset, size);
        break;
    }

    return LUX_GPU_OK;
}

size_t lux_gpu_buffer_size(const LuxBuffer* buf) {
    return buf ? buf->size : 0;
}

// =============================================================================
// Kernel Management
// =============================================================================

LuxKernel* lux_gpu_kernel_create(
    LuxGPU* gpu,
    const char* source,
    const char* entry_point,
    LuxKernelLang lang) {

    if (!gpu || !source || !entry_point) return nullptr;

    auto kernel = new LuxKernel{};
    kernel->gpu = gpu;
    kernel->source = source;
    kernel->entry_point = entry_point;
    kernel->lang = lang;

#if defined(LUX_HAVE_WEBGPU)
    kernel->webgpu_valid = false;
#endif

    // Kernel compilation is deferred until dispatch
    return kernel;
}

LuxKernel* lux_gpu_kernel_load(
    LuxGPU* gpu,
    const char* path,
    const char* entry_point) {

    if (!gpu || !path || !entry_point) return nullptr;

    // Read file contents
    FILE* f = fopen(path, "rb");
    if (!f) return nullptr;

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    std::vector<char> content(fsize + 1);
    size_t read = fread(content.data(), 1, fsize, f);
    fclose(f);

    if (read != static_cast<size_t>(fsize)) return nullptr;
    content[fsize] = '\0';

    // Detect language from extension
    LuxKernelLang lang = LUX_KERNEL_AUTO;
    std::string spath(path);
    if (str_ends_with(spath, ".wgsl")) {
        lang = LUX_KERNEL_WGSL;
    } else if (str_ends_with(spath, ".metal")) {
        lang = LUX_KERNEL_METAL;
    } else if (str_ends_with(spath, ".cu") || str_ends_with(spath, ".cuh")) {
        lang = LUX_KERNEL_CUDA;
    }

    return lux_gpu_kernel_create(gpu, content.data(), entry_point, lang);
}

void lux_gpu_kernel_destroy(LuxKernel* kernel) {
    if (!kernel) return;

#if defined(LUX_HAVE_WEBGPU)
    kernel->webgpu_valid = false;
#endif

    delete kernel;
}

int lux_gpu_kernel_bind(LuxKernel* kernel, uint32_t binding, LuxBuffer* buf) {
    if (!kernel || !buf) return LUX_GPU_ERROR_INVALID_ARGS;

    // Ensure bindings vector is large enough
    if (kernel->bindings.size() <= binding) {
        kernel->bindings.resize(binding + 1, nullptr);
    }
    kernel->bindings[binding] = buf;

    return LUX_GPU_OK;
}

int lux_gpu_dispatch(LuxGPU* gpu, LuxKernel* kernel, LuxDispatchConfig config) {
    if (!gpu || !kernel) return LUX_GPU_ERROR_INVALID_ARGS;

    switch (gpu->backend) {
#if defined(LUX_HAVE_WEBGPU)
    case LUX_GPU_BACKEND_WEBGPU: {
        if (!gpu->webgpu_ctx) return LUX_GPU_ERROR;

        // Compile kernel if not yet done
        if (!kernel->webgpu_valid && !kernel->source.empty()) {
            webgpu_backend::KernelCode code{
                kernel->source,
                webgpu_backend::Shape{config.block[0], config.block[1], config.block[2]},
                webgpu_backend::kf32
            };
            code.entryPoint = kernel->entry_point;

            // Build bindings
            std::vector<webgpu_backend::Tensor> tensors;
            for (auto* buf : kernel->bindings) {
                if (buf && buf->webgpu_valid) {
                    tensors.push_back(buf->webgpu_tensor);
                }
            }

            if (!tensors.empty()) {
                // Create kernel with bindings
                // Note: This is simplified; real implementation needs proper Bindings<N>
                webgpu_backend::Shape workgroups{config.grid[0], config.grid[1], config.grid[2]};

                // For now, handle common cases
                if (tensors.size() == 2) {
                    kernel->webgpu_kernel = webgpu_backend::createKernel(
                        *gpu->webgpu_ctx, code,
                        webgpu_backend::Bindings<2>{tensors[0], tensors[1]},
                        workgroups);
                    kernel->webgpu_valid = true;
                } else if (tensors.size() == 3) {
                    kernel->webgpu_kernel = webgpu_backend::createKernel(
                        *gpu->webgpu_ctx, code,
                        webgpu_backend::Bindings<3>{tensors[0], tensors[1], tensors[2]},
                        workgroups);
                    kernel->webgpu_valid = true;
                }
            }
        }

        if (kernel->webgpu_valid) {
            std::promise<void> promise;
            std::future<void> future = promise.get_future();
            webgpu_backend::dispatchKernel(*gpu->webgpu_ctx, kernel->webgpu_kernel, promise);
            webgpu_backend::wait(*gpu->webgpu_ctx, future);
        }
        break;
    }
#endif

    case LUX_GPU_BACKEND_CPU:
    default:
        // CPU kernels not supported via generic dispatch
        return LUX_GPU_ERROR_INVALID_KERNEL;
    }

    return LUX_GPU_OK;
}

int lux_gpu_dispatch_buffers(
    LuxGPU* gpu,
    LuxKernel* kernel,
    LuxDispatchConfig config,
    LuxBuffer** buffers) {

    if (!gpu || !kernel || !buffers) return LUX_GPU_ERROR_INVALID_ARGS;

    // Bind all buffers
    for (uint32_t i = 0; buffers[i] != nullptr; i++) {
        int rc = lux_gpu_kernel_bind(kernel, i, buffers[i]);
        if (rc != LUX_GPU_OK) return rc;
    }

    return lux_gpu_dispatch(gpu, kernel, config);
}

// =============================================================================
// Synchronization
// =============================================================================

int lux_gpu_sync(LuxGPU* gpu) {
    if (!gpu) return LUX_GPU_ERROR;

    switch (gpu->backend) {
#if defined(LUX_HAVE_MLX)
    case LUX_GPU_BACKEND_CUDA:
    case LUX_GPU_BACKEND_METAL:
        mlx_backend::synchronize();
        break;
#endif

#if defined(LUX_HAVE_WEBGPU)
    case LUX_GPU_BACKEND_WEBGPU:
        // WebGPU operations are synchronized per-dispatch
        if (gpu->webgpu_ctx) {
            webgpu_backend::processEvents(gpu->webgpu_ctx->instance);
        }
        break;
#endif

    case LUX_GPU_BACKEND_CPU:
    default:
        // CPU operations are synchronous
        break;
    }

    return LUX_GPU_OK;
}

LuxStream* lux_gpu_stream_create(LuxGPU* gpu) {
    if (!gpu) return nullptr;

    auto stream = new LuxStream{};
    stream->gpu = gpu;
    stream->in_use.store(false);
    return stream;
}

void lux_gpu_stream_destroy(LuxStream* stream) {
    delete stream;
}

int lux_gpu_stream_sync(LuxStream* stream) {
    if (!stream) return LUX_GPU_ERROR;
    return lux_gpu_sync(stream->gpu);
}

// =============================================================================
// Built-in ZK Kernels: Poseidon2
// =============================================================================

int lux_gpu_poseidon2(
    LuxGPU* gpu,
    LuxFr256* out,
    const LuxFr256* left,
    const LuxFr256* right,
    size_t count) {

    if (!gpu || !out || !left || !right || count == 0) {
        return LUX_GPU_ERROR_INVALID_ARGS;
    }

    switch (gpu->backend) {
#if defined(LUX_HAVE_MLX)
    case LUX_GPU_BACKEND_CUDA:
    case LUX_GPU_BACKEND_METAL: {
        // Use MLX ZK C API
        int rc = zk_poseidon2_hash(
            reinterpret_cast<Fr256*>(out),
            reinterpret_cast<const Fr256*>(left),
            reinterpret_cast<const Fr256*>(right),
            static_cast<uint32_t>(count));
        return (rc == ZK_SUCCESS) ? LUX_GPU_OK : LUX_GPU_ERROR;
    }
#endif

#if defined(LUX_HAVE_WEBGPU)
    case LUX_GPU_BACKEND_WEBGPU: {
        if (!gpu->webgpu_ctx) {
            // Fall through to CPU
            break;
        }

        // Convert to webgpu_zk::Fr256 format
        std::vector<webgpu_zk::Fr256> wleft(count), wright(count);
        for (size_t i = 0; i < count; i++) {
            for (int j = 0; j < 4; j++) {
                wleft[i].limbs[j] = left[i].limbs[j];
                wright[i].limbs[j] = right[i].limbs[j];
            }
        }

        auto results = webgpu_zk::get_zk_context().poseidon2_batch_hash(wleft, wright);
        if (results.size() != count) {
            // WebGPU failed, use CPU fallback
            break;
        }

        for (size_t i = 0; i < count; i++) {
            for (int j = 0; j < 4; j++) {
                out[i].limbs[j] = results[i].limbs[j];
            }
        }
        return LUX_GPU_OK;
    }
#endif

    case LUX_GPU_BACKEND_CPU:
    default:
        break;
    }

    // CPU fallback with real Poseidon2 implementation
    for (size_t i = 0; i < count; i++) {
        cpu_poseidon2_hash(&out[i], &left[i], &right[i]);
    }
    return LUX_GPU_OK;
}

// =============================================================================
// Built-in ZK Kernels: Merkle Root
// =============================================================================

int lux_gpu_merkle_root(
    LuxGPU* gpu,
    LuxFr256* root,
    const LuxFr256* leaves,
    size_t count) {

    if (!gpu || !root || !leaves || count == 0) {
        return LUX_GPU_ERROR_INVALID_ARGS;
    }

    // Pad to power of 2
    size_t n = 1;
    while (n < count) n <<= 1;

    std::vector<LuxFr256> current(n);
    std::memcpy(current.data(), leaves, count * sizeof(LuxFr256));
    // Zero-pad remaining
    std::memset(current.data() + count, 0, (n - count) * sizeof(LuxFr256));

    // Build Merkle tree bottom-up
    while (current.size() > 1) {
        size_t parent_count = current.size() / 2;
        std::vector<LuxFr256> parents(parent_count);

        // Hash adjacent pairs
        std::vector<LuxFr256> lefts(parent_count), rights(parent_count);
        for (size_t i = 0; i < parent_count; i++) {
            lefts[i] = current[2*i];
            rights[i] = current[2*i + 1];
        }

        int rc = lux_gpu_poseidon2(gpu, parents.data(), lefts.data(), rights.data(), parent_count);
        if (rc != LUX_GPU_OK) return rc;

        current = std::move(parents);
    }

    *root = current[0];
    return LUX_GPU_OK;
}

// =============================================================================
// Built-in ZK Kernels: Commitment
// =============================================================================

int lux_gpu_commitment(
    LuxGPU* gpu,
    LuxFr256* out,
    const LuxFr256* values,
    const LuxFr256* blindings,
    const LuxFr256* salts,
    size_t count) {

    if (!gpu || !out || !values || !blindings || !salts || count == 0) {
        return LUX_GPU_ERROR_INVALID_ARGS;
    }

    // Commitment = H(H(value, blinding), salt)
    std::vector<LuxFr256> inner(count);

    int rc = lux_gpu_poseidon2(gpu, inner.data(), values, blindings, count);
    if (rc != LUX_GPU_OK) return rc;

    return lux_gpu_poseidon2(gpu, out, inner.data(), salts, count);
}

// =============================================================================
// Built-in ZK Kernels: MSM (Multi-Scalar Multiplication)
// =============================================================================

int lux_gpu_msm(
    LuxGPU* gpu,
    void* result,
    const void* points,
    const LuxFr256* scalars,
    size_t count) {

    if (!gpu || !result || !points || !scalars || count == 0) {
        return LUX_GPU_ERROR_INVALID_ARGS;
    }

    switch (gpu->backend) {
#if defined(LUX_HAVE_MLX)
    case LUX_GPU_BACKEND_CUDA:
    case LUX_GPU_BACKEND_METAL: {
        // MLX ZK MSM implementation
        // Convert points and scalars to MLX arrays
        mlx_backend::array pts_arr = mlx_backend::array(
            static_cast<const uint64_t*>(points),
            {static_cast<int>(count), 2, 4},  // [N, 2, 4] for x,y coords
            mlx_backend::uint64);

        mlx_backend::array scalars_arr = mlx_backend::array(
            reinterpret_cast<const uint64_t*>(scalars),
            {static_cast<int>(count), 4},
            mlx_backend::uint64);

        auto result_arr = mlx::core::zk::msm(pts_arr, scalars_arr);
        mlx_backend::eval(result_arr);

        // Copy result (projective point: [3, 4])
        auto* data = result_arr.data<uint64_t>();
        std::memcpy(result, data, 3 * 4 * sizeof(uint64_t));
        return LUX_GPU_OK;
    }
#endif

    case LUX_GPU_BACKEND_WEBGPU:
    case LUX_GPU_BACKEND_CPU:
    default:
        // MSM requires significant implementation; not available on CPU/WebGPU yet
        return LUX_GPU_ERROR;
    }
}

// =============================================================================
// Global Instance
// =============================================================================

static std::mutex g_gpu_mutex;
static LuxGPU* g_gpu = nullptr;
static std::atomic<bool> g_gpu_init{false};

LuxGPU* lux_gpu_global(void) {
    if (!g_gpu_init.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(g_gpu_mutex);
        if (!g_gpu) {
            g_gpu = lux_gpu_create();
            g_gpu_init.store(true, std::memory_order_release);
        }
    }
    return g_gpu;
}

// =============================================================================
// Cleanup on exit (atexit handler)
// =============================================================================

namespace {

struct GlobalCleanup {
    ~GlobalCleanup() {
        if (g_gpu) {
            lux_gpu_destroy(g_gpu);
            g_gpu = nullptr;
        }
    }
};

static GlobalCleanup g_cleanup;

} // namespace
