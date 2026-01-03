// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Kernel Registry - Unified backend selection for GPU compute
//
// Selection logic:
//   1. Native override (Metal/CUDA) if available
//   2. WebGPU/WGSL if supported
//   3. CPU fallback
//
// Usage:
//   auto kernel = KernelRegistry::get("ntt_forward");
//   kernel->dispatch(params);

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace lux {
namespace gpu {

// ============================================================================
// Backend Types
// ============================================================================

enum class Backend {
    CPU,      // Fallback, always available
    Metal,    // Apple Silicon (native)
    CUDA,     // NVIDIA (native)
    WebGPU,   // Portable (Dawn/wgpu -> Metal/Vulkan/D3D12)
    Auto      // Select best available
};

inline const char* backend_name(Backend b) {
    switch (b) {
        case Backend::CPU:    return "CPU";
        case Backend::Metal:  return "Metal";
        case Backend::CUDA:   return "CUDA";
        case Backend::WebGPU: return "WebGPU";
        case Backend::Auto:   return "Auto";
    }
    return "Unknown";
}

// ============================================================================
// Kernel Capabilities
// ============================================================================

struct KernelCapabilities {
    bool supports_f16 = false;
    bool supports_f64 = false;
    bool supports_atomic64 = false;
    size_t max_workgroup_size = 256;
    size_t max_shared_memory = 32768;
    std::vector<std::string> extensions;
};

// ============================================================================
// Kernel Interface
// ============================================================================

struct KernelParams {
    void* data;
    size_t size;
    size_t batch;
    uint64_t modulus;
    uint64_t mu;  // Barrett constant
    void* extra;  // Kernel-specific data (twiddles, etc.)
};

class IKernel {
public:
    virtual ~IKernel() = default;
    
    virtual const std::string& name() const = 0;
    virtual Backend backend() const = 0;
    virtual void dispatch(const KernelParams& params) = 0;
    virtual void dispatch_async(const KernelParams& params) = 0;
    virtual void sync() = 0;
};

using KernelPtr = std::shared_ptr<IKernel>;

// ============================================================================
// Kernel Implementation Base
// ============================================================================

template <Backend B>
class KernelImpl : public IKernel {
protected:
    std::string name_;
    
public:
    explicit KernelImpl(const std::string& name) : name_(name) {}
    
    const std::string& name() const override { return name_; }
    Backend backend() const override { return B; }
    void sync() override {}
};

// ============================================================================
// Kernel Registry
// ============================================================================

class KernelRegistry {
public:
    using Factory = std::function<KernelPtr(Backend)>;
    
    static KernelRegistry& instance() {
        static KernelRegistry registry;
        return registry;
    }
    
    // Register a kernel factory
    void register_kernel(const std::string& name, Factory factory) {
        factories_[name] = std::move(factory);
    }
    
    // Get kernel with backend preference
    KernelPtr get(const std::string& name, Backend preferred = Backend::Auto) {
        auto it = factories_.find(name);
        if (it == factories_.end()) {
            return nullptr;
        }
        
        Backend backend = preferred;
        if (backend == Backend::Auto) {
            backend = select_best_backend();
        }
        
        return it->second(backend);
    }
    
    // Query available backends
    static std::vector<Backend> available_backends() {
        std::vector<Backend> backends;
        backends.push_back(Backend::CPU);
        
#ifdef MLX_BUILD_METAL
        backends.push_back(Backend::Metal);
#endif

#ifdef MLX_BUILD_CUDA  
        backends.push_back(Backend::CUDA);
#endif

#ifdef MLX_BUILD_WEBGPU
        backends.push_back(Backend::WebGPU);
#endif
        
        return backends;
    }
    
    // Check if a specific backend is available
    static bool has_backend(Backend b) {
        switch (b) {
            case Backend::CPU: return true;
#ifdef MLX_BUILD_METAL
            case Backend::Metal: return true;
#endif
#ifdef MLX_BUILD_CUDA
            case Backend::CUDA: return true;
#endif
#ifdef MLX_BUILD_WEBGPU
            case Backend::WebGPU: return true;
#endif
            default: return false;
        }
    }
    
    // Get capabilities for a backend
    static KernelCapabilities capabilities(Backend b);
    
private:
    KernelRegistry() = default;
    
    Backend select_best_backend() {
#ifdef MLX_BUILD_METAL
        return Backend::Metal;  // Prefer Metal on Apple
#elif defined(MLX_BUILD_CUDA)
        return Backend::CUDA;   // Prefer CUDA on NVIDIA
#elif defined(MLX_BUILD_WEBGPU)
        return Backend::WebGPU; // Fallback to WebGPU
#else
        return Backend::CPU;    // CPU fallback
#endif
    }
    
    std::unordered_map<std::string, Factory> factories_;
};

// ============================================================================
// Registration Helpers
// ============================================================================

#define REGISTER_KERNEL(name, factory) \
    static bool _kernel_##name##_registered = []() { \
        KernelRegistry::instance().register_kernel(#name, factory); \
        return true; \
    }()

// ============================================================================
// Crypto Ops Interface (unified API for FHE/lattice operations)
// ============================================================================

namespace cryptoops {

// NTT operations
void ntt_forward(void* data, size_t N, size_t batch, uint64_t Q, uint64_t mu,
                 const void* twiddles, Backend backend = Backend::Auto);

void ntt_inverse(void* data, size_t N, size_t batch, uint64_t Q, uint64_t mu,
                 uint64_t inv_N, const void* inv_twiddles, 
                 Backend backend = Backend::Auto);

void ntt_pointwise_mul(void* result, const void* a, const void* b,
                       size_t size, uint64_t Q, uint64_t mu,
                       Backend backend = Backend::Auto);

// Modular arithmetic
void mod_add(void* result, const void* a, const void* b,
             size_t size, uint64_t Q, Backend backend = Backend::Auto);

void mod_sub(void* result, const void* a, const void* b,
             size_t size, uint64_t Q, Backend backend = Backend::Auto);

void mod_mul(void* result, const void* a, const void* b,
             size_t size, uint64_t Q, uint64_t mu,
             Backend backend = Backend::Auto);

void mod_mul_add(void* acc, const void* a, const void* b,
                 size_t size, uint64_t Q, uint64_t mu,
                 Backend backend = Backend::Auto);

}  // namespace cryptoops

}  // namespace gpu
}  // namespace lux
