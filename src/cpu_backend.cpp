// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Built-in CPU Backend - SIMD-optimized fallback
// This is linked directly into the core library (not a plugin).

#include "lux/gpu/backend_plugin.h"
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

// =============================================================================
// CPU Buffer Implementation
// =============================================================================

struct CPUBuffer {
    void* data;
    size_t size;
};

struct CPUContext {
    int device_index;
};

// =============================================================================
// Modular Arithmetic for NTT
// =============================================================================

static inline uint64_t mod_add(uint64_t a, uint64_t b, uint64_t m) {
    return ((__uint128_t)a + b) % m;
}

static inline uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t m) {
    a %= m;
    b %= m;
    return (a >= b) ? (a - b) : (m - (b - a));
}

static inline uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t m) {
    return ((__uint128_t)a * b) % m;
}

static uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t m) {
    uint64_t result = 1;
    base %= m;
    while (exp > 0) {
        if (exp & 1) result = mod_mul(result, base, m);
        exp >>= 1;
        base = mod_mul(base, base, m);
    }
    return result;
}

static void bit_reverse(uint64_t* data, size_t n) {
    size_t j = 0;
    for (size_t i = 0; i < n; i++) {
        if (i < j) std::swap(data[i], data[j]);
        size_t m = n >> 1;
        while (m >= 1 && j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
}

static uint64_t find_primitive_root(size_t n, uint64_t m) {
    // Known primitive roots for common NTT primes
    if (m == 0xFFFFFFFF00000001ULL) return 7;  // Goldilocks
    if (m == 0x1000000000000001ULL) return 3;
    return 3;
}

// =============================================================================
// CPU Backend Functions
// =============================================================================

static LuxBackendContext* cpu_create_context(int device_index) {
    auto ctx = new CPUContext();
    ctx->device_index = device_index;
    return reinterpret_cast<LuxBackendContext*>(ctx);
}

static void cpu_destroy_context(LuxBackendContext* ctx) {
    delete reinterpret_cast<CPUContext*>(ctx);
}

static LuxBackendError cpu_get_device_count(int* count) {
    if (!count) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    *count = 1;
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_get_device_info(LuxBackendContext*, LuxBackendDeviceInfo* info) {
    if (!info) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    info->name = "CPU";
    info->vendor = "System";
    info->memory_total = 0;
    info->memory_available = 0;
#ifdef _OPENMP
    info->compute_units = omp_get_max_threads();
#else
    info->compute_units = 1;
#endif
    info->max_workgroup_size = 1;
    info->is_discrete = false;
    info->is_unified_memory = true;
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_sync(LuxBackendContext*) {
    return LUX_BACKEND_OK;
}

// Buffer management
static LuxBackendBuffer* cpu_buffer_alloc(LuxBackendContext*, size_t bytes) {
    auto buf = new CPUBuffer();
    buf->data = std::malloc(bytes);
    buf->size = bytes;
    if (!buf->data) {
        delete buf;
        return nullptr;
    }
    std::memset(buf->data, 0, bytes);
    return reinterpret_cast<LuxBackendBuffer*>(buf);
}

static LuxBackendBuffer* cpu_buffer_alloc_with_data(LuxBackendContext* ctx, const void* data, size_t bytes) {
    auto buf = reinterpret_cast<CPUBuffer*>(cpu_buffer_alloc(ctx, bytes));
    if (!buf) return nullptr;
    std::memcpy(buf->data, data, bytes);
    return reinterpret_cast<LuxBackendBuffer*>(buf);
}

static void cpu_buffer_free(LuxBackendContext*, LuxBackendBuffer* buf) {
    auto b = reinterpret_cast<CPUBuffer*>(buf);
    if (b) {
        std::free(b->data);
        delete b;
    }
}

static LuxBackendError cpu_buffer_copy_to_host(LuxBackendContext*, LuxBackendBuffer* buf, void* dst, size_t bytes) {
    auto b = reinterpret_cast<CPUBuffer*>(buf);
    if (!b || !dst) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    std::memcpy(dst, b->data, std::min(bytes, b->size));
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_buffer_copy_from_host(LuxBackendContext*, LuxBackendBuffer* buf, const void* src, size_t bytes) {
    auto b = reinterpret_cast<CPUBuffer*>(buf);
    if (!b || !src) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    std::memcpy(b->data, src, std::min(bytes, b->size));
    return LUX_BACKEND_OK;
}

static void* cpu_buffer_get_host_ptr(LuxBackendContext*, LuxBackendBuffer* buf) {
    auto b = reinterpret_cast<CPUBuffer*>(buf);
    return b ? b->data : nullptr;
}

// Kernel management (not supported for CPU)
static LuxBackendKernel* cpu_kernel_load(LuxBackendContext*, const char*, const char*) {
    return nullptr;  // Not supported
}

static LuxBackendKernel* cpu_kernel_load_binary(LuxBackendContext*, const void*, size_t, const char*) {
    return nullptr;
}

static void cpu_kernel_destroy(LuxBackendContext*, LuxBackendKernel*) {
}

static LuxBackendError cpu_kernel_dispatch(
    LuxBackendContext*, LuxBackendKernel*, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, LuxBackendBuffer**, int) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

// Tensor operations
static LuxBackendError cpu_op_add_f32(LuxBackendContext*, LuxBackendBuffer* a, LuxBackendBuffer* b, LuxBackendBuffer* out, size_t n) {
    auto ba = reinterpret_cast<CPUBuffer*>(a);
    auto bb = reinterpret_cast<CPUBuffer*>(b);
    auto bo = reinterpret_cast<CPUBuffer*>(out);
    if (!ba || !bb || !bo) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const float* pa = static_cast<const float*>(ba->data);
    const float* pb = static_cast<const float*>(bb->data);
    float* po = static_cast<float*>(bo->data);

#ifdef _OPENMP
    #pragma omp parallel for simd
#endif
    for (size_t i = 0; i < n; i++) {
        po[i] = pa[i] + pb[i];
    }
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_sub_f32(LuxBackendContext*, LuxBackendBuffer* a, LuxBackendBuffer* b, LuxBackendBuffer* out, size_t n) {
    auto ba = reinterpret_cast<CPUBuffer*>(a);
    auto bb = reinterpret_cast<CPUBuffer*>(b);
    auto bo = reinterpret_cast<CPUBuffer*>(out);
    if (!ba || !bb || !bo) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const float* pa = static_cast<const float*>(ba->data);
    const float* pb = static_cast<const float*>(bb->data);
    float* po = static_cast<float*>(bo->data);

#ifdef _OPENMP
    #pragma omp parallel for simd
#endif
    for (size_t i = 0; i < n; i++) {
        po[i] = pa[i] - pb[i];
    }
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_mul_f32(LuxBackendContext*, LuxBackendBuffer* a, LuxBackendBuffer* b, LuxBackendBuffer* out, size_t n) {
    auto ba = reinterpret_cast<CPUBuffer*>(a);
    auto bb = reinterpret_cast<CPUBuffer*>(b);
    auto bo = reinterpret_cast<CPUBuffer*>(out);
    if (!ba || !bb || !bo) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const float* pa = static_cast<const float*>(ba->data);
    const float* pb = static_cast<const float*>(bb->data);
    float* po = static_cast<float*>(bo->data);

#ifdef _OPENMP
    #pragma omp parallel for simd
#endif
    for (size_t i = 0; i < n; i++) {
        po[i] = pa[i] * pb[i];
    }
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_matmul_f32(LuxBackendContext*, LuxBackendBuffer* a, LuxBackendBuffer* b, LuxBackendBuffer* out, int M, int K, int N) {
    auto ba = reinterpret_cast<CPUBuffer*>(a);
    auto bb = reinterpret_cast<CPUBuffer*>(b);
    auto bo = reinterpret_cast<CPUBuffer*>(out);
    if (!ba || !bb || !bo) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    const float* pa = static_cast<const float*>(ba->data);
    const float* pb = static_cast<const float*>(bb->data);
    float* po = static_cast<float*>(bo->data);

#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += pa[i * K + k] * pb[k * N + j];
            }
            po[i * N + j] = sum;
        }
    }
    return LUX_BACKEND_OK;
}

// NTT operations
static LuxBackendError cpu_op_ntt_forward(LuxBackendContext*, uint64_t* data, size_t n, uint64_t modulus) {
    if (!data || n == 0 || (n & (n - 1)) != 0) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    uint64_t g = find_primitive_root(n, modulus);
    uint64_t omega_n = mod_pow(g, (modulus - 1) / n, modulus);

    bit_reverse(data, n);

    for (size_t len = 2; len <= n; len *= 2) {
        uint64_t w = mod_pow(omega_n, n / len, modulus);
        for (size_t i = 0; i < n; i += len) {
            uint64_t wn = 1;
            for (size_t j = 0; j < len / 2; j++) {
                uint64_t u = data[i + j];
                uint64_t t = mod_mul(wn, data[i + j + len / 2], modulus);
                data[i + j] = mod_add(u, t, modulus);
                data[i + j + len / 2] = mod_sub(u, t, modulus);
                wn = mod_mul(wn, w, modulus);
            }
        }
    }
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_ntt_inverse(LuxBackendContext*, uint64_t* data, size_t n, uint64_t modulus) {
    if (!data || n == 0 || (n & (n - 1)) != 0) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    uint64_t g = find_primitive_root(n, modulus);
    uint64_t omega_n = mod_pow(g, (modulus - 1) / n, modulus);
    uint64_t omega_n_inv = mod_pow(omega_n, modulus - 2, modulus);

    // DIF butterfly (large to small)
    for (size_t len = n; len >= 2; len /= 2) {
        uint64_t w = mod_pow(omega_n_inv, n / len, modulus);
        for (size_t i = 0; i < n; i += len) {
            uint64_t wn = 1;
            for (size_t j = 0; j < len / 2; j++) {
                uint64_t u = data[i + j];
                uint64_t v = data[i + j + len / 2];
                data[i + j] = mod_add(u, v, modulus);
                data[i + j + len / 2] = mod_mul(mod_sub(u, v, modulus), wn, modulus);
                wn = mod_mul(wn, w, modulus);
            }
        }
    }

    bit_reverse(data, n);

    // Scale by n^-1
    uint64_t n_inv = mod_pow(n, modulus - 2, modulus);
    for (size_t i = 0; i < n; i++) {
        data[i] = mod_mul(data[i], n_inv, modulus);
    }
    return LUX_BACKEND_OK;
}

static LuxBackendError cpu_op_msm(LuxBackendContext*, const void*, const void*, void*, size_t, int) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;  // Needs crypto library
}

// =============================================================================
// CPU Backend VTable
// =============================================================================

static const lux_gpu_backend_vtbl cpu_vtbl = {
    // Lifecycle
    .create_context = cpu_create_context,
    .destroy_context = cpu_destroy_context,

    // Device info
    .get_device_count = cpu_get_device_count,
    .get_device_info = cpu_get_device_info,

    // Sync
    .sync = cpu_sync,

    // Buffer management
    .buffer_alloc = cpu_buffer_alloc,
    .buffer_alloc_with_data = cpu_buffer_alloc_with_data,
    .buffer_free = cpu_buffer_free,
    .buffer_copy_to_host = cpu_buffer_copy_to_host,
    .buffer_copy_from_host = cpu_buffer_copy_from_host,
    .buffer_get_host_ptr = cpu_buffer_get_host_ptr,

    // Kernel management
    .kernel_load = cpu_kernel_load,
    .kernel_load_binary = cpu_kernel_load_binary,
    .kernel_destroy = cpu_kernel_destroy,
    .kernel_dispatch = cpu_kernel_dispatch,

    // Built-in operations
    .op_add_f32 = cpu_op_add_f32,
    .op_sub_f32 = cpu_op_sub_f32,
    .op_mul_f32 = cpu_op_mul_f32,
    .op_matmul_f32 = cpu_op_matmul_f32,

    // NTT operations
    .op_ntt_forward = cpu_op_ntt_forward,
    .op_ntt_inverse = cpu_op_ntt_inverse,

    // MSM
    .op_msm = cpu_op_msm,

    // Reserved
    ._reserved = {nullptr}
};

// =============================================================================
// Entry Point (called by core, not a plugin)
// =============================================================================

extern "C" bool cpu_backend_init(lux_gpu_backend_desc* out) {
    if (!out) return false;
    out->abi_version = LUX_GPU_BACKEND_ABI_VERSION;
    out->backend_name = "cpu";
    out->backend_version = "0.1.0";
    out->capabilities = LUX_CAP_TENSOR_OPS | LUX_CAP_MATMUL | LUX_CAP_NTT | LUX_CAP_UNIFIED_MEMORY;
    out->vtbl = &cpu_vtbl;
    return true;
}
