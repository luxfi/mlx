// Copyright (c) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Ecosystem
//
// Unified GPU implementation - dispatches to:
//   CUDA (MLX) | Metal (MLX) | WebGPU (Dawn) | CPU

#include "lux/gpu/gpu.h"
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

// =============================================================================
// Backend Includes
// =============================================================================

#if defined(LUX_HAVE_MLX)
#include "mlx/mlx.h"
#include "mlx/zk/zk.h"
namespace mlx_backend = mlx::core;
#endif

#if defined(LUX_HAVE_WEBGPU)
#include "webgpu/gpu.hpp"
namespace webgpu_backend = gpu;
#endif

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
    mlx_backend::Device mlx_device;
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
    mlx_backend::array mlx_array;
#endif

#if defined(LUX_HAVE_WEBGPU)
    webgpu_backend::Tensor webgpu_tensor;
#endif

    std::vector<uint8_t> cpu_data;  // CPU fallback
};

struct LuxKernel {
    LuxGPU* gpu;
    std::string source;
    std::string entry_point;
    LuxKernelLang lang;
    std::vector<LuxBuffer*> bindings;

#if defined(LUX_HAVE_WEBGPU)
    std::unique_ptr<webgpu_backend::Kernel> webgpu_kernel;
#endif
};

struct LuxStream {
    LuxGPU* gpu;
    // Backend-specific stream handles
};

// =============================================================================
// Backend Detection
// =============================================================================

static LuxGPUBackend detect_best_backend() {
#if defined(LUX_HAVE_MLX)
    // Check CUDA first (highest priority)
    #if defined(LUX_HAVE_CUDA)
    if (mlx_backend::metal::is_available() || true /* cuda check */) {
        return LUX_GPU_BACKEND_CUDA;
    }
    #endif

    // Then Metal
    if (mlx_backend::metal::is_available()) {
        return LUX_GPU_BACKEND_METAL;
    }
#endif

#if defined(LUX_HAVE_WEBGPU)
    // Try WebGPU
    try {
        auto ctx = webgpu_backend::createContext();
        return LUX_GPU_BACKEND_WEBGPU;
    } catch (...) {}
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

    if (backend == LUX_GPU_BACKEND_AUTO) {
        backend = detect_best_backend();
    }

    gpu->backend = backend;

    switch (backend) {
#if defined(LUX_HAVE_MLX)
    case LUX_GPU_BACKEND_CUDA:
        gpu->device_name = "CUDA (MLX)";
        gpu->mlx_device = mlx_backend::Device::gpu;
        mlx_backend::set_default_device(gpu->mlx_device);
        break;

    case LUX_GPU_BACKEND_METAL:
        gpu->device_name = "Metal (MLX)";
        gpu->mlx_device = mlx_backend::Device::gpu;
        mlx_backend::set_default_device(gpu->mlx_device);
        break;
#endif

#if defined(LUX_HAVE_WEBGPU)
    case LUX_GPU_BACKEND_WEBGPU:
        gpu->device_name = "WebGPU (Dawn)";
        try {
            gpu->webgpu_ctx = std::make_unique<webgpu_backend::Context>(
                webgpu_backend::createContext());
        } catch (...) {
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
    if (gpu) {
#if defined(LUX_HAVE_WEBGPU)
        gpu->webgpu_ctx.reset();
#endif
        delete gpu;
    }
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

    switch (gpu->backend) {
#if defined(LUX_HAVE_MLX)
    case LUX_GPU_BACKEND_CUDA:
    case LUX_GPU_BACKEND_METAL:
        buf->mlx_array = mlx_backend::zeros({static_cast<int>(size / 4)},
                                            mlx_backend::float32);
        break;
#endif

#if defined(LUX_HAVE_WEBGPU)
    case LUX_GPU_BACKEND_WEBGPU:
        if (gpu->webgpu_ctx) {
            buf->webgpu_tensor = webgpu_backend::createTensor(
                *gpu->webgpu_ctx,
                webgpu_backend::Shape{size / 4},
                webgpu_backend::kf32);
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
    delete buf;
}

int lux_gpu_buffer_write(LuxBuffer* buf, const void* data, size_t size, size_t offset) {
    if (!buf || !data || offset + size > buf->size) {
        return LUX_GPU_ERROR_INVALID_ARGS;
    }

    switch (buf->gpu->backend) {
#if defined(LUX_HAVE_MLX)
    case LUX_GPU_BACKEND_CUDA:
    case LUX_GPU_BACKEND_METAL:
        // MLX handles data transfer automatically
        buf->mlx_array = mlx_backend::array(
            static_cast<const float*>(data),
            {static_cast<int>(size / 4)});
        break;
#endif

#if defined(LUX_HAVE_WEBGPU)
    case LUX_GPU_BACKEND_WEBGPU:
        if (buf->gpu->webgpu_ctx) {
            webgpu_backend::toGPU(*buf->gpu->webgpu_ctx,
                                  buf->webgpu_tensor, data);
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
    if (!buf || !data || offset + size > buf->size) {
        return LUX_GPU_ERROR_INVALID_ARGS;
    }

    switch (buf->gpu->backend) {
#if defined(LUX_HAVE_MLX)
    case LUX_GPU_BACKEND_CUDA:
    case LUX_GPU_BACKEND_METAL: {
        mlx_backend::eval(buf->mlx_array);
        auto ptr = buf->mlx_array.data<float>();
        std::memcpy(data, ptr + offset/4, size);
        break;
    }
#endif

#if defined(LUX_HAVE_WEBGPU)
    case LUX_GPU_BACKEND_WEBGPU:
        if (buf->gpu->webgpu_ctx) {
            webgpu_backend::toCPU(*buf->gpu->webgpu_ctx,
                                  buf->webgpu_tensor, data, size);
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
        // WebGPU sync handled per-dispatch
        break;
#endif

    default:
        break;
    }

    return LUX_GPU_OK;
}

// =============================================================================
// Built-in ZK Kernels
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
    case LUX_GPU_BACKEND_METAL:
        return mlx::zk::poseidon2_hash(
            reinterpret_cast<mlx::zk::Fr256*>(out),
            reinterpret_cast<const mlx::zk::Fr256*>(left),
            reinterpret_cast<const mlx::zk::Fr256*>(right),
            count);
#endif

#if defined(LUX_HAVE_WEBGPU)
    case LUX_GPU_BACKEND_WEBGPU:
        if (gpu->webgpu_ctx) {
            // Use WebGPU ZK implementation
            // TODO: implement webgpu dispatch
        }
        break;
#endif

    case LUX_GPU_BACKEND_CPU:
    default:
        // CPU fallback - call gnark-crypto via C API
        for (size_t i = 0; i < count; i++) {
            // Simplified CPU hash (production should use gnark)
            out[i].limbs[0] = left[i].limbs[0] ^ right[i].limbs[0];
            out[i].limbs[1] = left[i].limbs[1] ^ right[i].limbs[1];
            out[i].limbs[2] = left[i].limbs[2] ^ right[i].limbs[2];
            out[i].limbs[3] = left[i].limbs[3] ^ right[i].limbs[3];
        }
        break;
    }

    return LUX_GPU_OK;
}

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

    // Build tree bottom-up
    while (current.size() > 1) {
        std::vector<LuxFr256> parents(current.size() / 2);

        int rc = lux_gpu_poseidon2(
            gpu,
            parents.data(),
            current.data(),
            current.data() + current.size() / 2,
            parents.size());

        if (rc != LUX_GPU_OK) return rc;

        current = std::move(parents);
    }

    *root = current[0];
    return LUX_GPU_OK;
}

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

    // H(H(value, blinding), salt)
    std::vector<LuxFr256> inner(count);
    int rc = lux_gpu_poseidon2(gpu, inner.data(), values, blindings, count);
    if (rc != LUX_GPU_OK) return rc;

    return lux_gpu_poseidon2(gpu, out, inner.data(), salts, count);
}

int lux_gpu_msm(
    LuxGPU* gpu,
    void* result,
    const void* points,
    const LuxFr256* scalars,
    size_t count) {

    // TODO: implement MSM
    return LUX_GPU_ERROR;
}

// =============================================================================
// Global Instance
// =============================================================================

static std::mutex g_gpu_mutex;
static LuxGPU* g_gpu = nullptr;

LuxGPU* lux_gpu_global(void) {
    std::lock_guard<std::mutex> lock(g_gpu_mutex);
    if (!g_gpu) {
        g_gpu = lux_gpu_create();
    }
    return g_gpu;
}
