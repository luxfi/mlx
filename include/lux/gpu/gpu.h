// Copyright (c) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Ecosystem
//
// Unified GPU Compute Library
// Single API across: CUDA | Metal (MLX) | WebGPU (Dawn) | CPU
//
// Usage:
//   LuxGPU* gpu = lux_gpu_create();
//   LuxBuffer* buf = lux_gpu_buffer_create(gpu, size, LUX_MEM_DEVICE);
//   lux_gpu_dispatch(gpu, kernel, ...);
//   lux_gpu_sync(gpu);

#ifndef LUX_GPU_H
#define LUX_GPU_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Backend Types
// =============================================================================

typedef enum {
    LUX_GPU_BACKEND_AUTO = 0,    // Auto-select best: CUDA > Metal > WebGPU > CPU
    LUX_GPU_BACKEND_CUDA,        // NVIDIA CUDA (via MLX or native)
    LUX_GPU_BACKEND_METAL,       // Apple Metal (via MLX)
    LUX_GPU_BACKEND_WEBGPU,      // WebGPU/WGSL (via Dawn)
    LUX_GPU_BACKEND_CPU,         // CPU fallback
} LuxGPUBackend;

typedef enum {
    LUX_GPU_OK = 0,
    LUX_GPU_ERROR = -1,
    LUX_GPU_ERROR_NO_DEVICE = -2,
    LUX_GPU_ERROR_OUT_OF_MEMORY = -3,
    LUX_GPU_ERROR_INVALID_KERNEL = -4,
    LUX_GPU_ERROR_INVALID_ARGS = -5,
} LuxGPUError;

typedef enum {
    LUX_MEM_HOST = 0,            // CPU-accessible memory
    LUX_MEM_DEVICE = 1,          // GPU-only memory
    LUX_MEM_SHARED = 2,          // Unified/shared memory (if available)
} LuxMemType;

typedef enum {
    LUX_DTYPE_F32 = 0,
    LUX_DTYPE_F16 = 1,
    LUX_DTYPE_BF16 = 2,
    LUX_DTYPE_I32 = 3,
    LUX_DTYPE_U32 = 4,
    LUX_DTYPE_I64 = 5,
    LUX_DTYPE_U64 = 6,
} LuxDType;

// =============================================================================
// Opaque Handle Types
// =============================================================================

typedef struct LuxGPU LuxGPU;
typedef struct LuxBuffer LuxBuffer;
typedef struct LuxKernel LuxKernel;
typedef struct LuxStream LuxStream;

// =============================================================================
// Device Management
// =============================================================================

// Create GPU context with auto-selected backend
LuxGPU* lux_gpu_create(void);

// Create with specific backend
LuxGPU* lux_gpu_create_backend(LuxGPUBackend backend);

// Destroy context
void lux_gpu_destroy(LuxGPU* gpu);

// Query active backend
LuxGPUBackend lux_gpu_backend(const LuxGPU* gpu);
const char* lux_gpu_backend_name(const LuxGPU* gpu);

// Device info
const char* lux_gpu_device_name(const LuxGPU* gpu);
size_t lux_gpu_memory_total(const LuxGPU* gpu);
size_t lux_gpu_memory_free(const LuxGPU* gpu);
int lux_gpu_compute_units(const LuxGPU* gpu);

// =============================================================================
// Buffer Management
// =============================================================================

// Create buffer
LuxBuffer* lux_gpu_buffer_create(LuxGPU* gpu, size_t size, LuxMemType mem);

// Create buffer with initial data
LuxBuffer* lux_gpu_buffer_create_data(
    LuxGPU* gpu,
    const void* data,
    size_t size,
    LuxMemType mem
);

// Destroy buffer
void lux_gpu_buffer_destroy(LuxBuffer* buf);

// Copy data to buffer
int lux_gpu_buffer_write(LuxBuffer* buf, const void* data, size_t size, size_t offset);

// Copy data from buffer
int lux_gpu_buffer_read(LuxBuffer* buf, void* data, size_t size, size_t offset);

// Get buffer size
size_t lux_gpu_buffer_size(const LuxBuffer* buf);

// =============================================================================
// Kernel Management
// =============================================================================

// Kernel source types
typedef enum {
    LUX_KERNEL_WGSL = 0,         // WebGPU Shading Language (portable)
    LUX_KERNEL_METAL = 1,        // Metal Shading Language
    LUX_KERNEL_CUDA = 2,         // CUDA C++
    LUX_KERNEL_AUTO = 3,         // Auto-select based on backend
} LuxKernelLang;

// Create kernel from source
LuxKernel* lux_gpu_kernel_create(
    LuxGPU* gpu,
    const char* source,
    const char* entry_point,
    LuxKernelLang lang
);

// Create kernel from file
LuxKernel* lux_gpu_kernel_load(
    LuxGPU* gpu,
    const char* path,
    const char* entry_point
);

// Destroy kernel
void lux_gpu_kernel_destroy(LuxKernel* kernel);

// =============================================================================
// Kernel Dispatch
// =============================================================================

// Dispatch configuration
typedef struct {
    uint32_t grid[3];            // Grid dimensions (workgroups)
    uint32_t block[3];           // Block dimensions (threads per workgroup)
} LuxDispatchConfig;

// Bind buffers to kernel
int lux_gpu_kernel_bind(LuxKernel* kernel, uint32_t binding, LuxBuffer* buf);

// Dispatch kernel
int lux_gpu_dispatch(LuxGPU* gpu, LuxKernel* kernel, LuxDispatchConfig config);

// Dispatch with inline bindings (NULL-terminated buffer list)
int lux_gpu_dispatch_buffers(
    LuxGPU* gpu,
    LuxKernel* kernel,
    LuxDispatchConfig config,
    LuxBuffer** buffers
);

// =============================================================================
// Synchronization
// =============================================================================

// Wait for all operations to complete
int lux_gpu_sync(LuxGPU* gpu);

// Create async stream
LuxStream* lux_gpu_stream_create(LuxGPU* gpu);
void lux_gpu_stream_destroy(LuxStream* stream);
int lux_gpu_stream_sync(LuxStream* stream);

// =============================================================================
// Built-in Kernels (ZK Crypto)
// =============================================================================

// Fr256 field element (BN254 scalar field)
typedef struct { uint64_t limbs[4]; } LuxFr256;

// Poseidon2 hash
int lux_gpu_poseidon2(
    LuxGPU* gpu,
    LuxFr256* out,
    const LuxFr256* left,
    const LuxFr256* right,
    size_t count
);

// Merkle tree root
int lux_gpu_merkle_root(
    LuxGPU* gpu,
    LuxFr256* root,
    const LuxFr256* leaves,
    size_t count
);

// Batch commitment
int lux_gpu_commitment(
    LuxGPU* gpu,
    LuxFr256* out,
    const LuxFr256* values,
    const LuxFr256* blindings,
    const LuxFr256* salts,
    size_t count
);

// MSM (multi-scalar multiplication)
int lux_gpu_msm(
    LuxGPU* gpu,
    void* result,           // Output point
    const void* points,     // Base points
    const LuxFr256* scalars,
    size_t count
);

// =============================================================================
// Global Instance
// =============================================================================

// Get/create global GPU context
LuxGPU* lux_gpu_global(void);

// =============================================================================
// Helper Macros
// =============================================================================

#define LUX_DISPATCH_1D(n) \
    (LuxDispatchConfig){ {((n) + 255) / 256, 1, 1}, {256, 1, 1} }

#define LUX_DISPATCH_2D(x, y) \
    (LuxDispatchConfig){ {((x) + 15) / 16, ((y) + 15) / 16, 1}, {16, 16, 1} }

#ifdef __cplusplus
}
#endif

#endif // LUX_GPU_H
