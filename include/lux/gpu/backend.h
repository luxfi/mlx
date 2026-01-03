// Copyright (c) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Ecosystem

#ifndef LUX_GPU_BACKEND_H
#define LUX_GPU_BACKEND_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Backend Priority: CUDA > Metal > WebGPU > CPU-Optimized > CPU-Pure
// =============================================================================

typedef enum {
    LUX_BACKEND_NONE = 0,

    // GPU backends (high performance)
    LUX_BACKEND_CUDA = 1,      // NVIDIA CUDA (highest priority)
    LUX_BACKEND_METAL = 2,     // Apple Metal (via MLX)
    LUX_BACKEND_WEBGPU = 3,    // WebGPU/WGSL (portable, via Dawn)

    // CPU backends
    LUX_BACKEND_CPU_OPT = 10,  // CPU with optimized libs (blst, asm)
    LUX_BACKEND_CPU_PURE = 11, // Pure C/C++ fallback

    LUX_BACKEND_COUNT
} LuxBackend;

typedef enum {
    LUX_DEVICE_TYPE_UNKNOWN = 0,
    LUX_DEVICE_TYPE_CPU = 1,
    LUX_DEVICE_TYPE_GPU_DISCRETE = 2,
    LUX_DEVICE_TYPE_GPU_INTEGRATED = 3,
} LuxDeviceType;

// Backend capability flags
typedef enum {
    LUX_CAP_NONE = 0,
    LUX_CAP_FP16 = (1 << 0),           // Half-precision float
    LUX_CAP_FP64 = (1 << 1),           // Double precision
    LUX_CAP_INT64_ATOMICS = (1 << 2),  // 64-bit atomic ops
    LUX_CAP_SUBGROUPS = (1 << 3),      // Subgroup/warp operations
    LUX_CAP_SHARED_MEM = (1 << 4),     // Shared/local memory
    LUX_CAP_ASYNC_COPY = (1 << 5),     // Async memory copy
} LuxCapability;

// Device information
typedef struct {
    LuxBackend backend;
    LuxDeviceType device_type;
    uint32_t capabilities;        // Bitmask of LuxCapability
    uint64_t memory_bytes;        // Available device memory
    uint32_t compute_units;       // SMs, CUs, or cores
    uint32_t max_workgroup_size;  // Max threads per workgroup
    char name[256];               // Device name string
    char vendor[64];              // Vendor string
} LuxDeviceInfo;

// =============================================================================
// Backend Detection & Selection
// =============================================================================

// Check if a specific backend is available
bool lux_backend_available(LuxBackend backend);

// Get the best available backend (follows priority order)
LuxBackend lux_backend_best(void);

// Get device info for a backend
bool lux_backend_device_info(LuxBackend backend, LuxDeviceInfo* info);

// Get human-readable backend name
const char* lux_backend_name(LuxBackend backend);

// =============================================================================
// Backend-Specific Availability Checks
// =============================================================================

// CUDA availability (checks for NVIDIA driver + compatible GPU)
bool lux_cuda_available(void);
int lux_cuda_device_count(void);

// Metal availability (macOS/iOS only)
bool lux_metal_available(void);

// WebGPU availability (via Dawn)
bool lux_webgpu_available(void);

// CPU optimized libs availability
bool lux_cpu_blst_available(void);      // blst for BLS12-381
bool lux_cpu_gnark_available(void);     // gnark-crypto bindings
bool lux_cpu_asm_available(void);       // Platform-specific assembly

// =============================================================================
// Runtime Backend Selection
// =============================================================================

// Force a specific backend (returns false if unavailable)
bool lux_backend_set(LuxBackend backend);

// Get currently active backend
LuxBackend lux_backend_get(void);

// Reset to auto-selection (best available)
void lux_backend_reset(void);

#ifdef __cplusplus
}
#endif

#endif // LUX_GPU_BACKEND_H
