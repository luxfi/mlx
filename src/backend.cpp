// Copyright (c) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Ecosystem

#include "lux/gpu/backend.h"
#include <cstring>
#include <mutex>

// =============================================================================
// Platform Detection
// =============================================================================

#if defined(__APPLE__)
#define LUX_PLATFORM_APPLE 1
#include <TargetConditionals.h>
#endif

#if defined(__linux__) && defined(__x86_64__)
#define LUX_PLATFORM_LINUX_X64 1
#endif

#if defined(_WIN32)
#define LUX_PLATFORM_WINDOWS 1
#endif

// =============================================================================
// Backend State
// =============================================================================

namespace {

struct BackendState {
    LuxBackend active_backend = LUX_BACKEND_NONE;
    bool initialized = false;
    std::mutex mutex;
};

BackendState& state() {
    static BackendState s;
    return s;
}

void ensure_initialized() {
    auto& s = state();
    if (s.initialized) return;

    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.initialized) return;

    // Auto-select best backend on first use
    s.active_backend = lux_backend_best();
    s.initialized = true;
}

} // namespace

// =============================================================================
// CUDA Detection
// =============================================================================

#if defined(LUX_HAVE_CUDA)
#include <cuda_runtime.h>

bool lux_cuda_available(void) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

int lux_cuda_device_count(void) {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

static bool cuda_get_info(LuxDeviceInfo* info) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return false;

    info->backend = LUX_BACKEND_CUDA;
    info->device_type = prop.integrated ? LUX_DEVICE_TYPE_GPU_INTEGRATED
                                        : LUX_DEVICE_TYPE_GPU_DISCRETE;
    info->memory_bytes = prop.totalGlobalMem;
    info->compute_units = prop.multiProcessorCount;
    info->max_workgroup_size = prop.maxThreadsPerBlock;

    info->capabilities = LUX_CAP_SHARED_MEM | LUX_CAP_SUBGROUPS;
    if (prop.major >= 6) info->capabilities |= LUX_CAP_FP16;
    if (prop.major >= 2) info->capabilities |= LUX_CAP_FP64;

    strncpy(info->name, prop.name, sizeof(info->name) - 1);
    strncpy(info->vendor, "NVIDIA", sizeof(info->vendor) - 1);
    return true;
}

#else

bool lux_cuda_available(void) { return false; }
int lux_cuda_device_count(void) { return 0; }
static bool cuda_get_info(LuxDeviceInfo*) { return false; }

#endif

// =============================================================================
// Metal Detection (Apple platforms)
// =============================================================================

#if defined(LUX_PLATFORM_APPLE)

// MLX handles Metal internally
extern "C" bool mlx_metal_available(void);
extern "C" bool mlx_metal_device_info(LuxDeviceInfo* info);

bool lux_metal_available(void) {
#if defined(LUX_HAVE_MLX)
    return mlx_metal_available();
#else
    return false;
#endif
}

static bool metal_get_info(LuxDeviceInfo* info) {
#if defined(LUX_HAVE_MLX)
    return mlx_metal_device_info(info);
#else
    return false;
#endif
}

#else

bool lux_metal_available(void) { return false; }
static bool metal_get_info(LuxDeviceInfo*) { return false; }

#endif

// =============================================================================
// WebGPU Detection (via Dawn)
// =============================================================================

#if defined(LUX_HAVE_WEBGPU)
// gpu.cpp / Dawn integration
extern "C" bool webgpu_available(void);
extern "C" bool webgpu_device_info(LuxDeviceInfo* info);

bool lux_webgpu_available(void) {
    return webgpu_available();
}

static bool webgpu_get_info(LuxDeviceInfo* info) {
    return webgpu_device_info(info);
}

#else

bool lux_webgpu_available(void) { return false; }
static bool webgpu_get_info(LuxDeviceInfo*) { return false; }

#endif

// =============================================================================
// CPU Optimized Libs Detection
// =============================================================================

bool lux_cpu_blst_available(void) {
#if defined(LUX_HAVE_BLST)
    return true;
#else
    return false;
#endif
}

bool lux_cpu_gnark_available(void) {
    // gnark-crypto is Go, accessed via CGO from lux/gpu
    return false; // C++ side doesn't use it directly
}

bool lux_cpu_asm_available(void) {
#if defined(__x86_64__) || defined(__aarch64__)
    return true; // Platform has optimized assembly paths
#else
    return false;
#endif
}

// =============================================================================
// Backend Selection
// =============================================================================

bool lux_backend_available(LuxBackend backend) {
    switch (backend) {
        case LUX_BACKEND_CUDA:     return lux_cuda_available();
        case LUX_BACKEND_METAL:    return lux_metal_available();
        case LUX_BACKEND_WEBGPU:   return lux_webgpu_available();
        case LUX_BACKEND_CPU_OPT:  return lux_cpu_blst_available() || lux_cpu_asm_available();
        case LUX_BACKEND_CPU_PURE: return true; // Always available
        default: return false;
    }
}

LuxBackend lux_backend_best(void) {
    // Priority: CUDA > Metal > WebGPU > CPU-Opt > CPU-Pure
    if (lux_cuda_available())   return LUX_BACKEND_CUDA;
    if (lux_metal_available())  return LUX_BACKEND_METAL;
    if (lux_webgpu_available()) return LUX_BACKEND_WEBGPU;
    if (lux_cpu_blst_available() || lux_cpu_asm_available())
        return LUX_BACKEND_CPU_OPT;
    return LUX_BACKEND_CPU_PURE;
}

bool lux_backend_device_info(LuxBackend backend, LuxDeviceInfo* info) {
    if (!info) return false;
    memset(info, 0, sizeof(*info));

    switch (backend) {
        case LUX_BACKEND_CUDA:   return cuda_get_info(info);
        case LUX_BACKEND_METAL:  return metal_get_info(info);
        case LUX_BACKEND_WEBGPU: return webgpu_get_info(info);
        case LUX_BACKEND_CPU_OPT:
        case LUX_BACKEND_CPU_PURE:
            info->backend = backend;
            info->device_type = LUX_DEVICE_TYPE_CPU;
            info->capabilities = LUX_CAP_FP64;
            strncpy(info->name, "CPU", sizeof(info->name) - 1);
            strncpy(info->vendor, "Host", sizeof(info->vendor) - 1);
            return true;
        default:
            return false;
    }
}

const char* lux_backend_name(LuxBackend backend) {
    switch (backend) {
        case LUX_BACKEND_CUDA:     return "CUDA";
        case LUX_BACKEND_METAL:    return "Metal";
        case LUX_BACKEND_WEBGPU:   return "WebGPU";
        case LUX_BACKEND_CPU_OPT:  return "CPU-Optimized";
        case LUX_BACKEND_CPU_PURE: return "CPU-Pure";
        default: return "Unknown";
    }
}

bool lux_backend_set(LuxBackend backend) {
    if (!lux_backend_available(backend)) return false;

    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.active_backend = backend;
    s.initialized = true;
    return true;
}

LuxBackend lux_backend_get(void) {
    ensure_initialized();
    return state().active_backend;
}

void lux_backend_reset(void) {
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.active_backend = lux_backend_best();
}
