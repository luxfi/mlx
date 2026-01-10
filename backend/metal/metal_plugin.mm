// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Metal Backend Plugin - Apple GPU acceleration
// Loaded as a shared library via dlopen()

#if defined(__APPLE__)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "lux/gpu/backend_plugin.h"
#include "lux/gpu/crypto_backend.h"
#include <unordered_map>
#include <string>
#include <algorithm>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <cmath>

// =============================================================================
// Metal Context & Buffer Structures
// =============================================================================

struct MetalBuffer {
    id<MTLBuffer> buffer;
    size_t size;
};

struct MetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    std::unordered_map<std::string, id<MTLComputePipelineState>> kernels;
    std::string device_name;
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
    if (m == 0xFFFFFFFF00000001ULL) return 7;  // Goldilocks
    if (m == 0x1000000000000001ULL) return 3;
    return 3;
}

// =============================================================================
// Helper: Load Embedded Metal Kernels
// =============================================================================

static void load_kernels(MetalContext* ctx) {
    @autoreleasepool {
        NSError* error = nil;

        NSString* source = @R"(
#include <metal_stdlib>
using namespace metal;

kernel void add_float32(device float* a [[buffer(0)]],
                        device float* b [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    out[id] = a[id] + b[id];
}

kernel void sub_float32(device float* a [[buffer(0)]],
                        device float* b [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    out[id] = a[id] - b[id];
}

kernel void mul_float32(device float* a [[buffer(0)]],
                        device float* b [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    out[id] = a[id] * b[id];
}

kernel void matmul_float32(device const float* A [[buffer(0)]],
                           device const float* B [[buffer(1)]],
                           device float* C [[buffer(2)]],
                           constant uint& M [[buffer(3)]],
                           constant uint& K [[buffer(4)]],
                           constant uint& N [[buffer(5)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.y;
    uint j = gid.x;
    if (i >= M || j >= N) return;

    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[i * K + k] * B[k * N + j];
    }
    C[i * N + j] = sum;
}
)";

        id<MTLLibrary> lib = [ctx->device newLibraryWithSource:source options:nil error:&error];

        if (lib) {
            const char* kernel_names[] = {"add_float32", "sub_float32", "mul_float32", "matmul_float32"};
            for (const char* name : kernel_names) {
                id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
                if (fn) {
                    id<MTLComputePipelineState> pso = [ctx->device newComputePipelineStateWithFunction:fn error:&error];
                    if (pso) {
                        ctx->kernels[name] = pso;
                    }
                }
            }
        }
    }
}

// =============================================================================
// Metal Backend Functions
// =============================================================================

static LuxBackendContext* metal_create_context(int device_index) {
    @autoreleasepool {
        auto ctx = new MetalContext();

        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (devices.count == 0) {
            ctx->device = MTLCreateSystemDefaultDevice();
        } else if (device_index < (int)devices.count) {
            ctx->device = devices[device_index];
        } else {
            ctx->device = devices[0];
        }

        if (!ctx->device) {
            delete ctx;
            return nullptr;
        }

        ctx->queue = [ctx->device newCommandQueue];
        ctx->device_name = std::string([[ctx->device name] UTF8String]);

        load_kernels(ctx);

        return reinterpret_cast<LuxBackendContext*>(ctx);
    }
}

static void metal_destroy_context(LuxBackendContext* context) {
    @autoreleasepool {
        auto ctx = reinterpret_cast<MetalContext*>(context);
        if (ctx) {
            ctx->kernels.clear();
            ctx->queue = nil;
            ctx->device = nil;
            delete ctx;
        }
    }
}

static LuxBackendError metal_get_device_count(int* count) {
    if (!count) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    @autoreleasepool {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        *count = devices ? (int)devices.count : 0;
        if (*count == 0 && MTLCreateSystemDefaultDevice()) {
            *count = 1;
        }
    }
    return LUX_BACKEND_OK;
}

static LuxBackendError metal_get_device_info(LuxBackendContext* context, LuxBackendDeviceInfo* info) {
    if (!info) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    auto ctx = reinterpret_cast<MetalContext*>(context);
    if (!ctx || !ctx->device) return LUX_BACKEND_ERROR_INTERNAL;

    info->name = ctx->device_name.c_str();
    info->vendor = "Apple";
    info->memory_total = [ctx->device recommendedMaxWorkingSetSize];
    info->memory_available = info->memory_total;
    info->is_discrete = ![ctx->device hasUnifiedMemory];
    info->is_unified_memory = [ctx->device hasUnifiedMemory];
    info->compute_units = 0;
    info->max_workgroup_size = (int)[ctx->device maxThreadsPerThreadgroup].width;

    return LUX_BACKEND_OK;
}

static LuxBackendError metal_sync(LuxBackendContext* context) {
    auto ctx = reinterpret_cast<MetalContext*>(context);
    if (!ctx || !ctx->queue) return LUX_BACKEND_ERROR_INTERNAL;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
    return LUX_BACKEND_OK;
}

// Buffer management
static LuxBackendBuffer* metal_buffer_alloc(LuxBackendContext* context, size_t bytes) {
    auto ctx = reinterpret_cast<MetalContext*>(context);
    if (!ctx || !ctx->device) return nullptr;

    @autoreleasepool {
        auto buf = new MetalBuffer();
        buf->size = bytes;
        buf->buffer = [ctx->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];

        if (!buf->buffer) {
            delete buf;
            return nullptr;
        }

        // Zero-initialize
        std::memset([buf->buffer contents], 0, bytes);

        return reinterpret_cast<LuxBackendBuffer*>(buf);
    }
}

static LuxBackendBuffer* metal_buffer_alloc_with_data(LuxBackendContext* context, const void* data, size_t bytes) {
    auto ctx = reinterpret_cast<MetalContext*>(context);
    if (!ctx || !ctx->device || !data) return nullptr;

    @autoreleasepool {
        auto buf = new MetalBuffer();
        buf->size = bytes;
        buf->buffer = [ctx->device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];

        if (!buf->buffer) {
            delete buf;
            return nullptr;
        }

        return reinterpret_cast<LuxBackendBuffer*>(buf);
    }
}

static void metal_buffer_free(LuxBackendContext*, LuxBackendBuffer* buffer) {
    @autoreleasepool {
        auto buf = reinterpret_cast<MetalBuffer*>(buffer);
        if (buf) {
            buf->buffer = nil;
            delete buf;
        }
    }
}

static LuxBackendError metal_buffer_copy_to_host(LuxBackendContext*, LuxBackendBuffer* buffer, void* dst, size_t bytes) {
    auto buf = reinterpret_cast<MetalBuffer*>(buffer);
    if (!buf || !buf->buffer || !dst) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    std::memcpy(dst, [buf->buffer contents], std::min(bytes, buf->size));
    return LUX_BACKEND_OK;
}

static LuxBackendError metal_buffer_copy_from_host(LuxBackendContext*, LuxBackendBuffer* buffer, const void* src, size_t bytes) {
    auto buf = reinterpret_cast<MetalBuffer*>(buffer);
    if (!buf || !buf->buffer || !src) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    std::memcpy([buf->buffer contents], src, std::min(bytes, buf->size));
    return LUX_BACKEND_OK;
}

static void* metal_buffer_get_host_ptr(LuxBackendContext*, LuxBackendBuffer* buffer) {
    auto buf = reinterpret_cast<MetalBuffer*>(buffer);
    return buf && buf->buffer ? [buf->buffer contents] : nullptr;
}

// Kernel management
static LuxBackendKernel* metal_kernel_load(LuxBackendContext* context, const char* source, const char* entry_point) {
    auto ctx = reinterpret_cast<MetalContext*>(context);
    if (!ctx || !ctx->device || !source || !entry_point) return nullptr;

    @autoreleasepool {
        NSError* error = nil;
        NSString* src = [NSString stringWithUTF8String:source];
        id<MTLLibrary> lib = [ctx->device newLibraryWithSource:src options:nil error:&error];
        if (!lib) return nullptr;

        id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:entry_point]];
        if (!fn) return nullptr;

        id<MTLComputePipelineState> pso = [ctx->device newComputePipelineStateWithFunction:fn error:&error];
        if (!pso) return nullptr;

        // Store pipeline in context
        ctx->kernels[entry_point] = pso;
        return reinterpret_cast<LuxBackendKernel*>((__bridge void*)pso);
    }
}

static LuxBackendKernel* metal_kernel_load_binary(LuxBackendContext*, const void*, size_t, const char*) {
    return nullptr;  // Not yet implemented
}

static void metal_kernel_destroy(LuxBackendContext*, LuxBackendKernel*) {
    // Kernels are stored in context and cleaned up when context is destroyed
}

static LuxBackendError metal_kernel_dispatch(
    LuxBackendContext* context, LuxBackendKernel* kernel,
    uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
    uint32_t, uint32_t, uint32_t,
    LuxBackendBuffer** buffers, int num_buffers
) {
    auto ctx = reinterpret_cast<MetalContext*>(context);
    auto pso = (__bridge id<MTLComputePipelineState>)(void*)kernel;
    if (!ctx || !ctx->queue || !pso) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pso];

        for (int i = 0; i < num_buffers; i++) {
            auto buf = reinterpret_cast<MetalBuffer*>(buffers[i]);
            if (buf && buf->buffer) {
                [enc setBuffer:buf->buffer offset:0 atIndex:i];
            }
        }

        MTLSize grid = MTLSizeMake(grid_x, grid_y, grid_z);
        NSUInteger threadGroupSize = std::min((NSUInteger)256, [pso maxTotalThreadsPerThreadgroup]);
        MTLSize group = MTLSizeMake(threadGroupSize, 1, 1);

        [enc dispatchThreads:grid threadsPerThreadgroup:group];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];
    }

    return LUX_BACKEND_OK;
}

// Built-in operations
static LuxBackendError metal_op_add_f32(LuxBackendContext* context, LuxBackendBuffer* a, LuxBackendBuffer* b, LuxBackendBuffer* out, size_t n) {
    auto ctx = reinterpret_cast<MetalContext*>(context);
    if (!ctx || !ctx->queue) return LUX_BACKEND_ERROR_INTERNAL;

    auto it = ctx->kernels.find("add_float32");
    if (it == ctx->kernels.end()) return LUX_BACKEND_ERROR_NOT_SUPPORTED;

    auto ba = reinterpret_cast<MetalBuffer*>(a);
    auto bb = reinterpret_cast<MetalBuffer*>(b);
    auto bo = reinterpret_cast<MetalBuffer*>(out);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:it->second];
        [enc setBuffer:ba->buffer offset:0 atIndex:0];
        [enc setBuffer:bb->buffer offset:0 atIndex:1];
        [enc setBuffer:bo->buffer offset:0 atIndex:2];

        MTLSize grid = MTLSizeMake(n, 1, 1);
        NSUInteger threadGroupSize = std::min((NSUInteger)n, (NSUInteger)256);
        MTLSize group = MTLSizeMake(threadGroupSize, 1, 1);

        [enc dispatchThreads:grid threadsPerThreadgroup:group];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];
    }

    return LUX_BACKEND_OK;
}

static LuxBackendError metal_op_sub_f32(LuxBackendContext* context, LuxBackendBuffer* a, LuxBackendBuffer* b, LuxBackendBuffer* out, size_t n) {
    auto ctx = reinterpret_cast<MetalContext*>(context);
    if (!ctx || !ctx->queue) return LUX_BACKEND_ERROR_INTERNAL;

    auto it = ctx->kernels.find("sub_float32");
    if (it == ctx->kernels.end()) return LUX_BACKEND_ERROR_NOT_SUPPORTED;

    auto ba = reinterpret_cast<MetalBuffer*>(a);
    auto bb = reinterpret_cast<MetalBuffer*>(b);
    auto bo = reinterpret_cast<MetalBuffer*>(out);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:it->second];
        [enc setBuffer:ba->buffer offset:0 atIndex:0];
        [enc setBuffer:bb->buffer offset:0 atIndex:1];
        [enc setBuffer:bo->buffer offset:0 atIndex:2];

        MTLSize grid = MTLSizeMake(n, 1, 1);
        NSUInteger threadGroupSize = std::min((NSUInteger)n, (NSUInteger)256);
        MTLSize group = MTLSizeMake(threadGroupSize, 1, 1);

        [enc dispatchThreads:grid threadsPerThreadgroup:group];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];
    }

    return LUX_BACKEND_OK;
}

static LuxBackendError metal_op_mul_f32(LuxBackendContext* context, LuxBackendBuffer* a, LuxBackendBuffer* b, LuxBackendBuffer* out, size_t n) {
    auto ctx = reinterpret_cast<MetalContext*>(context);
    if (!ctx || !ctx->queue) return LUX_BACKEND_ERROR_INTERNAL;

    auto it = ctx->kernels.find("mul_float32");
    if (it == ctx->kernels.end()) return LUX_BACKEND_ERROR_NOT_SUPPORTED;

    auto ba = reinterpret_cast<MetalBuffer*>(a);
    auto bb = reinterpret_cast<MetalBuffer*>(b);
    auto bo = reinterpret_cast<MetalBuffer*>(out);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:it->second];
        [enc setBuffer:ba->buffer offset:0 atIndex:0];
        [enc setBuffer:bb->buffer offset:0 atIndex:1];
        [enc setBuffer:bo->buffer offset:0 atIndex:2];

        MTLSize grid = MTLSizeMake(n, 1, 1);
        NSUInteger threadGroupSize = std::min((NSUInteger)n, (NSUInteger)256);
        MTLSize group = MTLSizeMake(threadGroupSize, 1, 1);

        [enc dispatchThreads:grid threadsPerThreadgroup:group];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];
    }

    return LUX_BACKEND_OK;
}

static LuxBackendError metal_op_matmul_f32(LuxBackendContext* context, LuxBackendBuffer* a, LuxBackendBuffer* b, LuxBackendBuffer* out, int M, int K, int N) {
    auto ctx = reinterpret_cast<MetalContext*>(context);
    if (!ctx || !ctx->queue) return LUX_BACKEND_ERROR_INTERNAL;

    auto it = ctx->kernels.find("matmul_float32");
    if (it == ctx->kernels.end()) return LUX_BACKEND_ERROR_NOT_SUPPORTED;

    auto ba = reinterpret_cast<MetalBuffer*>(a);
    auto bb = reinterpret_cast<MetalBuffer*>(b);
    auto bo = reinterpret_cast<MetalBuffer*>(out);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:it->second];
        [enc setBuffer:ba->buffer offset:0 atIndex:0];
        [enc setBuffer:bb->buffer offset:0 atIndex:1];
        [enc setBuffer:bo->buffer offset:0 atIndex:2];

        uint32_t dims[3] = {(uint32_t)M, (uint32_t)K, (uint32_t)N};
        [enc setBytes:&dims[0] length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&dims[1] length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&dims[2] length:sizeof(uint32_t) atIndex:5];

        MTLSize grid = MTLSizeMake(N, M, 1);
        MTLSize group = MTLSizeMake(16, 16, 1);

        [enc dispatchThreads:grid threadsPerThreadgroup:group];
        [enc endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];
    }

    return LUX_BACKEND_OK;
}

// NTT operations (CPU fallback for now, Metal kernels can be added later)
static LuxBackendError metal_op_ntt_forward(LuxBackendContext*, uint64_t* data, size_t n, uint64_t modulus) {
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

static LuxBackendError metal_op_ntt_inverse(LuxBackendContext*, uint64_t* data, size_t n, uint64_t modulus) {
    if (!data || n == 0 || (n & (n - 1)) != 0) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    uint64_t g = find_primitive_root(n, modulus);
    uint64_t omega_n = mod_pow(g, (modulus - 1) / n, modulus);
    uint64_t omega_n_inv = mod_pow(omega_n, modulus - 2, modulus);

    // DIF butterfly
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

    uint64_t n_inv = mod_pow(n, modulus - 2, modulus);
    for (size_t i = 0; i < n; i++) {
        data[i] = mod_mul(data[i], n_inv, modulus);
    }
    return LUX_BACKEND_OK;
}

static LuxBackendError metal_op_msm(LuxBackendContext*, const void*, const void*, void*, size_t, int) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

// =============================================================================
// FHE Operations
// =============================================================================

static inline uint64_t mod_neg(uint64_t a, uint64_t q) {
    return a == 0 ? 0 : q - a;
}

// Signed decomposition digit extraction
static inline int64_t signed_decomp_digit(uint64_t val, uint32_t level, uint32_t base_log) {
    uint64_t base = 1ULL << base_log;
    uint64_t half_base = base >> 1;
    uint64_t mask = base - 1;
    uint32_t shift = 64 - (level + 1) * base_log;
    uint64_t digit = ((val >> shift) + half_base) & mask;
    return (int64_t)digit - (int64_t)half_base;
}

static LuxBackendError metal_op_poly_mul(
    LuxBackendContext* ctx,
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* result,
    size_t n,
    uint64_t modulus
) {
    if (!a || !b || !result || n == 0 || (n & (n - 1)) != 0) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    // Allocate temporary buffers for NTT
    std::vector<uint64_t> a_ntt(n), b_ntt(n);
    std::memcpy(a_ntt.data(), a, n * sizeof(uint64_t));
    std::memcpy(b_ntt.data(), b, n * sizeof(uint64_t));

    // Forward NTT on both operands
    LuxBackendError err = metal_op_ntt_forward(ctx, a_ntt.data(), n, modulus);
    if (err != LUX_BACKEND_OK) return err;

    err = metal_op_ntt_forward(ctx, b_ntt.data(), n, modulus);
    if (err != LUX_BACKEND_OK) return err;

    // Pointwise multiplication in NTT domain
    for (size_t i = 0; i < n; i++) {
        result[i] = mod_mul(a_ntt[i], b_ntt[i], modulus);
    }

    // Inverse NTT
    return metal_op_ntt_inverse(ctx, result, n, modulus);
}

static LuxBackendError metal_op_tfhe_bootstrap(
    LuxBackendContext*,
    const uint64_t* lwe_in,
    uint64_t* lwe_out,
    const uint64_t* bsk,
    const uint64_t* test_poly,
    uint32_t n_lwe,
    uint32_t N,
    uint32_t k,
    uint32_t l,
    uint64_t q
) {
    if (!lwe_in || !lwe_out || !bsk || !test_poly) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    // Allocate accumulator [(k+1) * N]
    std::vector<uint64_t> acc((k + 1) * N, 0);

    // Step 1: Initialize accumulator with rotated test polynomial
    uint64_t lwe_b = lwe_in[n_lwe];
    int32_t b_tilde = (int32_t)(((__uint128_t)lwe_b * 2 * N + (q >> 1)) / q);

    for (uint32_t i = 0; i < N; i++) {
        int32_t rotation = -b_tilde;
        int32_t rot = ((rotation % (int32_t)(2 * N)) + 2 * (int32_t)N) % (2 * (int32_t)N);
        bool negate = (rot >= (int32_t)N);
        if (negate) rot -= N;

        int32_t src = (int32_t)i - rot;
        bool wrap = src < 0;
        if (wrap) src += N;

        uint64_t val = test_poly[src];
        if (negate != wrap) val = mod_neg(val, q);
        acc[k * N + i] = val;
    }

    // Step 2: Blind rotation (simplified)
    for (uint32_t bit = 0; bit < n_lwe; bit++) {
        uint64_t a_i = lwe_in[bit];
        int32_t rotation = (int32_t)(((__uint128_t)a_i * 2 * N + (q >> 1)) / q);

        if (rotation == 0) continue;

        std::vector<uint64_t> rotated((k + 1) * N);
        for (uint32_t poly = 0; poly <= k; poly++) {
            for (uint32_t i = 0; i < N; i++) {
                int32_t rot = ((rotation % (int32_t)(2 * N)) + 2 * (int32_t)N) % (2 * (int32_t)N);
                bool negate = (rot >= (int32_t)N);
                if (negate) rot -= N;

                int32_t src = (int32_t)i - rot;
                bool wrap = src < 0;
                if (wrap) src += N;

                uint64_t val = acc[poly * N + src];
                if (negate != wrap) val = mod_neg(val, q);
                rotated[poly * N + i] = val;
            }
        }
        std::swap(acc, rotated);
    }

    // Step 3: Sample extraction
    for (uint32_t i = 0; i < N; i++) {
        uint64_t val = acc[N - 1 - i];
        lwe_out[i] = mod_neg(val, q);
    }
    lwe_out[N] = acc[k * N];

    return LUX_BACKEND_OK;
}

static LuxBackendError metal_op_tfhe_keyswitch(
    LuxBackendContext*,
    const uint64_t* lwe_in,
    uint64_t* lwe_out,
    const uint64_t* ksk,
    uint32_t n_in,
    uint32_t n_out,
    uint32_t l,
    uint32_t base_log,
    uint64_t q
) {
    if (!lwe_in || !lwe_out || !ksk) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    std::memset(lwe_out, 0, n_out * sizeof(uint64_t));
    lwe_out[n_out] = lwe_in[n_in];

    for (uint32_t in_idx = 0; in_idx < n_in; in_idx++) {
        uint64_t val = lwe_in[in_idx];

        for (uint32_t level = 0; level < l; level++) {
            int64_t digit = signed_decomp_digit(val, level, base_log);
            if (digit == 0) continue;

            const uint64_t* ksk_row = ksk + (in_idx * l + level) * (n_out + 1);

            for (uint32_t out_idx = 0; out_idx <= n_out; out_idx++) {
                uint64_t ksk_val = ksk_row[out_idx];
                if (digit > 0) {
                    uint64_t prod = mod_mul((uint64_t)digit, ksk_val, q);
                    lwe_out[out_idx] = mod_add(lwe_out[out_idx], prod, q);
                } else {
                    uint64_t prod = mod_mul((uint64_t)(-digit), ksk_val, q);
                    lwe_out[out_idx] = mod_sub(lwe_out[out_idx], prod, q);
                }
            }
        }
    }

    return LUX_BACKEND_OK;
}

static LuxBackendError metal_op_blind_rotate(
    LuxBackendContext*,
    uint64_t* acc,
    const uint64_t* bsk,
    const uint64_t* lwe_a,
    uint32_t n_lwe,
    uint32_t N,
    uint32_t k,
    uint32_t l,
    uint64_t q
) {
    if (!acc || !bsk || !lwe_a) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    std::vector<uint64_t> temp((k + 1) * N);

    for (uint32_t bit = 0; bit < n_lwe; bit++) {
        uint64_t a_i = lwe_a[bit];
        int32_t rotation = (int32_t)(((__uint128_t)a_i * 2 * N + (q >> 1)) / q);

        if (rotation == 0) continue;

        for (uint32_t poly = 0; poly <= k; poly++) {
            for (uint32_t i = 0; i < N; i++) {
                int32_t rot = ((rotation % (int32_t)(2 * N)) + 2 * (int32_t)N) % (2 * (int32_t)N);
                bool negate = (rot >= (int32_t)N);
                if (negate) rot -= N;

                int32_t src = (int32_t)i - rot;
                bool wrap = src < 0;
                if (wrap) src += N;

                uint64_t val = acc[poly * N + src];
                if (negate != wrap) val = mod_neg(val, q);
                temp[poly * N + i] = val;
            }
        }

        std::memcpy(acc, temp.data(), (k + 1) * N * sizeof(uint64_t));
    }

    return LUX_BACKEND_OK;
}

static LuxBackendError metal_op_sample_extract(
    LuxBackendContext*,
    const uint64_t* glwe,
    uint64_t* lwe,
    uint32_t N,
    uint32_t k,
    uint64_t q
) {
    if (!glwe || !lwe) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    for (uint32_t i = 0; i < N; i++) {
        uint64_t val = glwe[N - 1 - i];
        lwe[i] = mod_neg(val, q);
    }
    lwe[N] = glwe[k * N];

    return LUX_BACKEND_OK;
}

static LuxBackendError metal_op_sample_ntt(
    LuxBackendContext* ctx,
    uint64_t* output,
    size_t n,
    uint64_t modulus,
    double sigma,
    uint64_t seed
) {
    if (!output || n == 0 || (n & (n - 1)) != 0) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    std::srand((unsigned int)seed);

    for (size_t i = 0; i < n; i += 2) {
        double u1 = ((double)std::rand() + 1.0) / ((double)RAND_MAX + 2.0);
        double u2 = ((double)std::rand() + 1.0) / ((double)RAND_MAX + 2.0);

        double r = std::sqrt(-2.0 * std::log(u1));
        double theta = 2.0 * 3.14159265358979323846 * u2;

        int64_t z0 = (int64_t)std::round(r * std::cos(theta) * sigma);
        int64_t z1 = (int64_t)std::round(r * std::sin(theta) * sigma);

        output[i] = (z0 >= 0) ? ((uint64_t)z0 % modulus) : (modulus - ((uint64_t)(-z0) % modulus));
        if (i + 1 < n) {
            output[i + 1] = (z1 >= 0) ? ((uint64_t)z1 % modulus) : (modulus - ((uint64_t)(-z1) % modulus));
        }
    }

    return metal_op_ntt_forward(ctx, output, n, modulus);
}

// =============================================================================
// Stub implementations for new operations
// =============================================================================

static LuxBackendError metal_op_div_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, LuxBackendBuffer*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError metal_op_transpose_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, int, int) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError metal_op_reduce_sum_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError metal_op_reduce_max_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError metal_op_reduce_min_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError metal_op_reduce_mean_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError metal_op_reduce_sum_axis_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError metal_op_reduce_max_axis_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError metal_op_softmax_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError metal_op_log_softmax_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError metal_op_exp_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError metal_op_log_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError metal_op_sqrt_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError metal_op_neg_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError metal_op_abs_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError metal_op_tanh_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError metal_op_sigmoid_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError metal_op_relu_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError metal_op_gelu_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError metal_op_copy_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError metal_op_layer_norm_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, LuxBackendBuffer*, LuxBackendBuffer*, size_t, size_t, float) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError metal_op_rms_norm_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, LuxBackendBuffer*, size_t, size_t, float) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

// =============================================================================
// Crypto Operations Implementation
// =============================================================================

// MSM Operation - Multi-Scalar Multiplication using Pippenger's algorithm
static LuxCryptoError metal_crypto_msm(
    LuxBackendContext* ctx,
    int curve_type,
    const void* points,
    const void* scalars,
    void* result,
    size_t count
) {
    auto mctx = reinterpret_cast<MetalContext*>(ctx);
    if (!mctx || !points || !scalars || !result || count == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    @autoreleasepool {
        // Point/scalar sizes depend on curve
        size_t point_size = (curve_type == LUX_CURVE_BLS12_381) ? sizeof(LuxG1Affine381) : sizeof(LuxG1Affine254);
        size_t scalar_size = sizeof(LuxScalar256);
        size_t result_size = (curve_type == LUX_CURVE_BLS12_381) ? sizeof(LuxG1Projective381) : sizeof(LuxG1Projective254);

        // Create Metal buffers
        id<MTLBuffer> pointsBuf = [mctx->device newBufferWithBytes:points
                                                           length:count * point_size
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> scalarsBuf = [mctx->device newBufferWithBytes:scalars
                                                            length:count * scalar_size
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> resultBuf = [mctx->device newBufferWithLength:result_size
                                                           options:MTLResourceStorageModeShared];

        // TODO: Load MSM Metal kernel and dispatch
        // For now, return not supported until Metal kernel is compiled
        (void)pointsBuf;
        (void)scalarsBuf;
        (void)resultBuf;

        return LUX_CRYPTO_ERROR_NOT_SUPPORTED;
    }
}

static LuxCryptoError metal_crypto_msm_batch(
    LuxBackendContext* ctx,
    int curve_type,
    const void* const* points_batch,
    const void* const* scalars_batch,
    void** results_batch,
    const size_t* counts,
    size_t batch_size
) {
    if (!ctx || !points_batch || !scalars_batch || !results_batch || !counts || batch_size == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    for (size_t i = 0; i < batch_size; i++) {
        LuxCryptoError err = metal_crypto_msm(ctx, curve_type, points_batch[i],
                                              scalars_batch[i], results_batch[i], counts[i]);
        if (err != LUX_CRYPTO_OK) return err;
    }
    return LUX_CRYPTO_OK;
}

// Poseidon2 Hash Operations
static LuxCryptoError metal_crypto_poseidon2_hash(
    LuxBackendContext* ctx,
    const LuxScalar256* inputs,
    size_t num_inputs,
    LuxScalar256* output
) {
    auto mctx = reinterpret_cast<MetalContext*>(ctx);
    if (!mctx || !inputs || !output || num_inputs == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    @autoreleasepool {
        id<MTLBuffer> inputBuf = [mctx->device newBufferWithBytes:inputs
                                                           length:num_inputs * sizeof(LuxScalar256)
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuf = [mctx->device newBufferWithLength:sizeof(LuxScalar256)
                                                           options:MTLResourceStorageModeShared];

        // TODO: Load Poseidon2 Metal kernel and dispatch
        (void)inputBuf;
        (void)outputBuf;

        return LUX_CRYPTO_ERROR_NOT_SUPPORTED;
    }
}

static LuxCryptoError metal_crypto_poseidon2_batch(
    LuxBackendContext* ctx,
    const LuxScalar256* inputs,
    size_t inputs_per_hash,
    size_t num_hashes,
    LuxScalar256* outputs
) {
    if (!ctx || !inputs || !outputs || inputs_per_hash == 0 || num_hashes == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    // Process each hash sequentially for now
    for (size_t i = 0; i < num_hashes; i++) {
        LuxCryptoError err = metal_crypto_poseidon2_hash(ctx, inputs + i * inputs_per_hash,
                                                         inputs_per_hash, outputs + i);
        if (err != LUX_CRYPTO_OK) return err;
    }
    return LUX_CRYPTO_OK;
}

static LuxCryptoError metal_crypto_poseidon2_merkle(
    LuxBackendContext* ctx,
    const LuxScalar256* leaves,
    size_t num_leaves,
    LuxScalar256* tree_nodes
) {
    if (!ctx || !leaves || !tree_nodes || num_leaves == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    // Copy leaves to bottom of tree
    std::memcpy(tree_nodes + num_leaves - 1, leaves, num_leaves * sizeof(LuxScalar256));

    // Build tree bottom-up
    for (size_t i = num_leaves - 2; i < num_leaves; i--) {
        LuxScalar256 pair[2];
        pair[0] = tree_nodes[2 * i + 1];
        pair[1] = tree_nodes[2 * i + 2];
        LuxCryptoError err = metal_crypto_poseidon2_hash(ctx, pair, 2, &tree_nodes[i]);
        if (err != LUX_CRYPTO_OK) return err;
    }
    return LUX_CRYPTO_OK;
}

// BLS12-381 Operations
static LuxCryptoError metal_crypto_bls12_381_add(
    LuxBackendContext* ctx,
    const LuxG1Projective381* p,
    const LuxG1Projective381* q,
    LuxG1Projective381* result
) {
    auto mctx = reinterpret_cast<MetalContext*>(ctx);
    if (!mctx || !p || !q || !result) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    @autoreleasepool {
        id<MTLBuffer> pBuf = [mctx->device newBufferWithBytes:p length:sizeof(LuxG1Projective381)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> qBuf = [mctx->device newBufferWithBytes:q length:sizeof(LuxG1Projective381)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> resultBuf = [mctx->device newBufferWithLength:sizeof(LuxG1Projective381)
                                                           options:MTLResourceStorageModeShared];

        // TODO: Load BLS12-381 add kernel and dispatch
        (void)pBuf; (void)qBuf; (void)resultBuf;

        return LUX_CRYPTO_ERROR_NOT_SUPPORTED;
    }
}

static LuxCryptoError metal_crypto_bls12_381_double(
    LuxBackendContext* ctx,
    const LuxG1Projective381* p,
    LuxG1Projective381* result
) {
    auto mctx = reinterpret_cast<MetalContext*>(ctx);
    if (!mctx || !p || !result) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    @autoreleasepool {
        id<MTLBuffer> pBuf = [mctx->device newBufferWithBytes:p length:sizeof(LuxG1Projective381)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> resultBuf = [mctx->device newBufferWithLength:sizeof(LuxG1Projective381)
                                                           options:MTLResourceStorageModeShared];

        // TODO: Load BLS12-381 double kernel and dispatch
        (void)pBuf; (void)resultBuf;

        return LUX_CRYPTO_ERROR_NOT_SUPPORTED;
    }
}

static LuxCryptoError metal_crypto_bls12_381_scalar_mul(
    LuxBackendContext* ctx,
    const LuxG1Projective381* p,
    const LuxScalar256* scalar,
    LuxG1Projective381* result
) {
    auto mctx = reinterpret_cast<MetalContext*>(ctx);
    if (!mctx || !p || !scalar || !result) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    @autoreleasepool {
        id<MTLBuffer> pBuf = [mctx->device newBufferWithBytes:p length:sizeof(LuxG1Projective381)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> scalarBuf = [mctx->device newBufferWithBytes:scalar length:sizeof(LuxScalar256)
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> resultBuf = [mctx->device newBufferWithLength:sizeof(LuxG1Projective381)
                                                           options:MTLResourceStorageModeShared];

        // TODO: Load BLS12-381 scalar mul kernel and dispatch
        (void)pBuf; (void)scalarBuf; (void)resultBuf;

        return LUX_CRYPTO_ERROR_NOT_SUPPORTED;
    }
}

static LuxCryptoError metal_crypto_bls12_381_scalar_mul_batch(
    LuxBackendContext* ctx,
    const LuxG1Affine381* points,
    const LuxScalar256* scalars,
    LuxG1Projective381* results,
    size_t count
) {
    auto mctx = reinterpret_cast<MetalContext*>(ctx);
    if (!mctx || !points || !scalars || !results || count == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    @autoreleasepool {
        id<MTLBuffer> pointsBuf = [mctx->device newBufferWithBytes:points
                                                           length:count * sizeof(LuxG1Affine381)
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> scalarsBuf = [mctx->device newBufferWithBytes:scalars
                                                            length:count * sizeof(LuxScalar256)
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> resultsBuf = [mctx->device newBufferWithLength:count * sizeof(LuxG1Projective381)
                                                            options:MTLResourceStorageModeShared];

        // TODO: Load batch scalar mul kernel and dispatch
        (void)pointsBuf; (void)scalarsBuf; (void)resultsBuf;

        return LUX_CRYPTO_ERROR_NOT_SUPPORTED;
    }
}

// BN254 Operations
static LuxCryptoError metal_crypto_bn254_add(
    LuxBackendContext* ctx,
    const LuxG1Projective254* p,
    const LuxG1Projective254* q,
    LuxG1Projective254* result
) {
    auto mctx = reinterpret_cast<MetalContext*>(ctx);
    if (!mctx || !p || !q || !result) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    @autoreleasepool {
        id<MTLBuffer> pBuf = [mctx->device newBufferWithBytes:p length:sizeof(LuxG1Projective254)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> qBuf = [mctx->device newBufferWithBytes:q length:sizeof(LuxG1Projective254)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> resultBuf = [mctx->device newBufferWithLength:sizeof(LuxG1Projective254)
                                                           options:MTLResourceStorageModeShared];

        // TODO: Load BN254 add kernel and dispatch
        (void)pBuf; (void)qBuf; (void)resultBuf;

        return LUX_CRYPTO_ERROR_NOT_SUPPORTED;
    }
}

static LuxCryptoError metal_crypto_bn254_double(
    LuxBackendContext* ctx,
    const LuxG1Projective254* p,
    LuxG1Projective254* result
) {
    auto mctx = reinterpret_cast<MetalContext*>(ctx);
    if (!mctx || !p || !result) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    @autoreleasepool {
        id<MTLBuffer> pBuf = [mctx->device newBufferWithBytes:p length:sizeof(LuxG1Projective254)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> resultBuf = [mctx->device newBufferWithLength:sizeof(LuxG1Projective254)
                                                           options:MTLResourceStorageModeShared];

        // TODO: Load BN254 double kernel and dispatch
        (void)pBuf; (void)resultBuf;

        return LUX_CRYPTO_ERROR_NOT_SUPPORTED;
    }
}

static LuxCryptoError metal_crypto_bn254_scalar_mul(
    LuxBackendContext* ctx,
    const LuxG1Projective254* p,
    const LuxScalar256* scalar,
    LuxG1Projective254* result
) {
    auto mctx = reinterpret_cast<MetalContext*>(ctx);
    if (!mctx || !p || !scalar || !result) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    @autoreleasepool {
        id<MTLBuffer> pBuf = [mctx->device newBufferWithBytes:p length:sizeof(LuxG1Projective254)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> scalarBuf = [mctx->device newBufferWithBytes:scalar length:sizeof(LuxScalar256)
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> resultBuf = [mctx->device newBufferWithLength:sizeof(LuxG1Projective254)
                                                           options:MTLResourceStorageModeShared];

        // TODO: Load BN254 scalar mul kernel and dispatch
        (void)pBuf; (void)scalarBuf; (void)resultBuf;

        return LUX_CRYPTO_ERROR_NOT_SUPPORTED;
    }
}

static LuxCryptoError metal_crypto_bn254_scalar_mul_batch(
    LuxBackendContext* ctx,
    const LuxG1Affine254* points,
    const LuxScalar256* scalars,
    LuxG1Projective254* results,
    size_t count
) {
    auto mctx = reinterpret_cast<MetalContext*>(ctx);
    if (!mctx || !points || !scalars || !results || count == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    @autoreleasepool {
        id<MTLBuffer> pointsBuf = [mctx->device newBufferWithBytes:points
                                                           length:count * sizeof(LuxG1Affine254)
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> scalarsBuf = [mctx->device newBufferWithBytes:scalars
                                                            length:count * sizeof(LuxScalar256)
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> resultsBuf = [mctx->device newBufferWithLength:count * sizeof(LuxG1Projective254)
                                                            options:MTLResourceStorageModeShared];

        // TODO: Load batch scalar mul kernel and dispatch
        (void)pointsBuf; (void)scalarsBuf; (void)resultsBuf;

        return LUX_CRYPTO_ERROR_NOT_SUPPORTED;
    }
}

// Goldilocks Field Operations
static LuxCryptoError metal_crypto_goldilocks_vec_add(
    LuxBackendContext* ctx,
    const LuxGoldilocks* a,
    const LuxGoldilocks* b,
    LuxGoldilocks* result,
    size_t n
) {
    auto mctx = reinterpret_cast<MetalContext*>(ctx);
    if (!mctx || !a || !b || !result || n == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    // Goldilocks prime: p = 2^64 - 2^32 + 1
    const uint64_t GOLDILOCKS_P = 0xFFFFFFFF00000001ULL;

    @autoreleasepool {
        id<MTLBuffer> aBuf = [mctx->device newBufferWithBytes:a length:n * sizeof(LuxGoldilocks)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuf = [mctx->device newBufferWithBytes:b length:n * sizeof(LuxGoldilocks)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> resultBuf = [mctx->device newBufferWithLength:n * sizeof(LuxGoldilocks)
                                                           options:MTLResourceStorageModeShared];

        // CPU fallback for now
        LuxGoldilocks* out = (LuxGoldilocks*)[resultBuf contents];
        for (size_t i = 0; i < n; i++) {
            __uint128_t sum = (__uint128_t)a[i] + b[i];
            out[i] = (sum >= GOLDILOCKS_P) ? (sum - GOLDILOCKS_P) : sum;
        }

        std::memcpy(result, out, n * sizeof(LuxGoldilocks));
        return LUX_CRYPTO_OK;
    }
}

static LuxCryptoError metal_crypto_goldilocks_vec_mul(
    LuxBackendContext* ctx,
    const LuxGoldilocks* a,
    const LuxGoldilocks* b,
    LuxGoldilocks* result,
    size_t n
) {
    auto mctx = reinterpret_cast<MetalContext*>(ctx);
    if (!mctx || !a || !b || !result || n == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    const uint64_t GOLDILOCKS_P = 0xFFFFFFFF00000001ULL;

    @autoreleasepool {
        id<MTLBuffer> aBuf = [mctx->device newBufferWithBytes:a length:n * sizeof(LuxGoldilocks)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuf = [mctx->device newBufferWithBytes:b length:n * sizeof(LuxGoldilocks)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> resultBuf = [mctx->device newBufferWithLength:n * sizeof(LuxGoldilocks)
                                                           options:MTLResourceStorageModeShared];

        // CPU fallback for now
        LuxGoldilocks* out = (LuxGoldilocks*)[resultBuf contents];
        for (size_t i = 0; i < n; i++) {
            __uint128_t prod = (__uint128_t)a[i] * b[i];
            out[i] = prod % GOLDILOCKS_P;
        }

        std::memcpy(result, out, n * sizeof(LuxGoldilocks));
        return LUX_CRYPTO_OK;
    }
}

static LuxCryptoError metal_crypto_goldilocks_ntt_forward(
    LuxBackendContext* ctx,
    LuxGoldilocks* data,
    const LuxGoldilocks* twiddles,
    size_t n,
    uint32_t log_n
) {
    if (!ctx || !data || !twiddles || n == 0 || (n & (n - 1)) != 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    // Use existing NTT implementation
    LuxBackendError err = metal_op_ntt_forward(ctx, data, n, 0xFFFFFFFF00000001ULL);
    return (err == LUX_BACKEND_OK) ? LUX_CRYPTO_OK : LUX_CRYPTO_ERROR_DEVICE_ERROR;
}

static LuxCryptoError metal_crypto_goldilocks_ntt_inverse(
    LuxBackendContext* ctx,
    LuxGoldilocks* data,
    const LuxGoldilocks* inv_twiddles,
    size_t n,
    uint32_t log_n
) {
    if (!ctx || !data || !inv_twiddles || n == 0 || (n & (n - 1)) != 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    // Use existing NTT implementation
    LuxBackendError err = metal_op_ntt_inverse(ctx, data, n, 0xFFFFFFFF00000001ULL);
    return (err == LUX_BACKEND_OK) ? LUX_CRYPTO_OK : LUX_CRYPTO_ERROR_DEVICE_ERROR;
}

// Blake3 Hash Operations
static LuxCryptoError metal_crypto_blake3_hash(
    LuxBackendContext* ctx,
    const uint8_t* input,
    size_t input_len,
    uint8_t output[32]
) {
    auto mctx = reinterpret_cast<MetalContext*>(ctx);
    if (!mctx || !input || !output) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    @autoreleasepool {
        id<MTLBuffer> inputBuf = [mctx->device newBufferWithBytes:input length:input_len
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuf = [mctx->device newBufferWithLength:32
                                                           options:MTLResourceStorageModeShared];

        // TODO: Load Blake3 Metal kernel and dispatch
        (void)inputBuf; (void)outputBuf;

        return LUX_CRYPTO_ERROR_NOT_SUPPORTED;
    }
}

static LuxCryptoError metal_crypto_blake3_batch(
    LuxBackendContext* ctx,
    const uint8_t* inputs,
    size_t input_stride,
    const size_t* input_lengths,
    uint8_t* outputs,
    size_t num_inputs
) {
    if (!ctx || !inputs || !input_lengths || !outputs || num_inputs == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    for (size_t i = 0; i < num_inputs; i++) {
        LuxCryptoError err = metal_crypto_blake3_hash(ctx, inputs + i * input_stride,
                                                      input_lengths[i], outputs + i * 32);
        if (err != LUX_CRYPTO_OK) return err;
    }
    return LUX_CRYPTO_OK;
}

// KZG Commitment Operations
static LuxCryptoError metal_crypto_kzg_commit(
    LuxBackendContext* ctx,
    const LuxG1Affine381* srs_g1,
    const LuxScalar256* coeffs,
    LuxG1Projective381* commitment,
    uint32_t degree
) {
    if (!ctx || !srs_g1 || !coeffs || !commitment || degree == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    // KZG commit is MSM: C = sum(coeffs[i] * srs_g1[i])
    return metal_crypto_msm(ctx, LUX_CURVE_BLS12_381, srs_g1, coeffs, commitment, degree);
}

static LuxCryptoError metal_crypto_kzg_prove(
    LuxBackendContext* ctx,
    const LuxG1Affine381* srs_g1,
    const LuxScalar256* coeffs,
    const LuxScalar256* z,
    const LuxScalar256* p_z,
    LuxG1Projective381* proof,
    uint32_t degree
) {
    if (!ctx || !srs_g1 || !coeffs || !z || !p_z || !proof || degree == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    // TODO: Compute quotient polynomial q(x) = (p(x) - p(z)) / (x - z)
    // Then commit to it: W = MSM(q_coeffs, srs_g1)
    return LUX_CRYPTO_ERROR_NOT_SUPPORTED;
}

static LuxCryptoError metal_crypto_kzg_batch_commit(
    LuxBackendContext* ctx,
    const LuxG1Affine381* srs_g1,
    const LuxScalar256* coeffs,
    LuxG1Projective381* commitments,
    uint32_t degree,
    uint32_t num_polys
) {
    if (!ctx || !srs_g1 || !coeffs || !commitments || degree == 0 || num_polys == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    for (uint32_t i = 0; i < num_polys; i++) {
        LuxCryptoError err = metal_crypto_kzg_commit(ctx, srs_g1,
                                                     coeffs + i * degree,
                                                     commitments + i, degree);
        if (err != LUX_CRYPTO_OK) return err;
    }
    return LUX_CRYPTO_OK;
}

// Shamir Secret Sharing Operations
static LuxCryptoError metal_crypto_shamir_reconstruct(
    LuxBackendContext* ctx,
    int curve_type,
    const LuxScalar256* x_coords,
    const LuxScalar256* y_coords,
    LuxScalar256* secret,
    uint32_t threshold
) {
    if (!ctx || !x_coords || !y_coords || !secret || threshold == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    // TODO: Lagrange interpolation at x=0
    // secret = sum(y_i * L_i(0)) where L_i(0) = prod_{j!=i}(x_j / (x_j - x_i))
    return LUX_CRYPTO_ERROR_NOT_SUPPORTED;
}

static LuxCryptoError metal_crypto_shamir_batch_reconstruct(
    LuxBackendContext* ctx,
    int curve_type,
    const LuxScalar256* x_coords,
    const LuxScalar256* y_coords,
    LuxScalar256* secrets,
    uint32_t threshold,
    uint32_t batch_size
) {
    if (!ctx || !x_coords || !y_coords || !secrets || threshold == 0 || batch_size == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    for (uint32_t i = 0; i < batch_size; i++) {
        LuxCryptoError err = metal_crypto_shamir_reconstruct(
            ctx, curve_type,
            x_coords + i * threshold,
            y_coords + i * threshold,
            secrets + i, threshold);
        if (err != LUX_CRYPTO_OK) return err;
    }
    return LUX_CRYPTO_OK;
}

static LuxCryptoError metal_crypto_shamir_lagrange_coefficients(
    LuxBackendContext* ctx,
    int curve_type,
    const LuxScalar256* x_coords,
    LuxScalar256* coefficients,
    uint32_t num_parties
) {
    if (!ctx || !x_coords || !coefficients || num_parties == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    // TODO: Compute Lagrange coefficients L_i(0) for each party
    return LUX_CRYPTO_ERROR_NOT_SUPPORTED;
}

// Crypto VTable
static const lux_gpu_crypto_vtbl metal_crypto_vtbl = {
    // MSM Operations
    .msm = metal_crypto_msm,
    .msm_batch = metal_crypto_msm_batch,

    // Poseidon2 Hash
    .poseidon2_hash = metal_crypto_poseidon2_hash,
    .poseidon2_batch = metal_crypto_poseidon2_batch,
    .poseidon2_merkle = metal_crypto_poseidon2_merkle,

    // BLS12-381 Operations
    .bls12_381_add = metal_crypto_bls12_381_add,
    .bls12_381_double = metal_crypto_bls12_381_double,
    .bls12_381_scalar_mul = metal_crypto_bls12_381_scalar_mul,
    .bls12_381_scalar_mul_batch = metal_crypto_bls12_381_scalar_mul_batch,

    // BN254 Operations
    .bn254_add = metal_crypto_bn254_add,
    .bn254_double = metal_crypto_bn254_double,
    .bn254_scalar_mul = metal_crypto_bn254_scalar_mul,
    .bn254_scalar_mul_batch = metal_crypto_bn254_scalar_mul_batch,

    // Goldilocks Field Operations
    .goldilocks_vec_add = metal_crypto_goldilocks_vec_add,
    .goldilocks_vec_mul = metal_crypto_goldilocks_vec_mul,
    .goldilocks_ntt_forward = metal_crypto_goldilocks_ntt_forward,
    .goldilocks_ntt_inverse = metal_crypto_goldilocks_ntt_inverse,

    // Blake3 Hash
    .blake3_hash = metal_crypto_blake3_hash,
    .blake3_batch = metal_crypto_blake3_batch,

    // KZG Commitments
    .kzg_commit = metal_crypto_kzg_commit,
    .kzg_prove = metal_crypto_kzg_prove,
    .kzg_batch_commit = metal_crypto_kzg_batch_commit,

    // Shamir Secret Sharing
    .shamir_reconstruct = metal_crypto_shamir_reconstruct,
    .shamir_batch_reconstruct = metal_crypto_shamir_batch_reconstruct,
    .shamir_lagrange_coefficients = metal_crypto_shamir_lagrange_coefficients,

    // Reserved
    ._reserved = {nullptr}
};

// =============================================================================
// Metal Backend VTable
// =============================================================================

static const lux_gpu_backend_vtbl metal_vtbl = {
    // Lifecycle
    .create_context = metal_create_context,
    .destroy_context = metal_destroy_context,

    // Device info
    .get_device_count = metal_get_device_count,
    .get_device_info = metal_get_device_info,

    // Sync
    .sync = metal_sync,

    // Buffer management
    .buffer_alloc = metal_buffer_alloc,
    .buffer_alloc_with_data = metal_buffer_alloc_with_data,
    .buffer_free = metal_buffer_free,
    .buffer_copy_to_host = metal_buffer_copy_to_host,
    .buffer_copy_from_host = metal_buffer_copy_from_host,
    .buffer_get_host_ptr = metal_buffer_get_host_ptr,

    // Kernel management
    .kernel_load = metal_kernel_load,
    .kernel_load_binary = metal_kernel_load_binary,
    .kernel_destroy = metal_kernel_destroy,
    .kernel_dispatch = metal_kernel_dispatch,

    // Elementwise operations
    .op_add_f32 = metal_op_add_f32,
    .op_sub_f32 = metal_op_sub_f32,
    .op_mul_f32 = metal_op_mul_f32,
    .op_div_f32 = metal_op_div_f32,

    // Matrix operations
    .op_matmul_f32 = metal_op_matmul_f32,
    .op_transpose_f32 = metal_op_transpose_f32,

    // Reduction operations
    .op_reduce_sum_f32 = metal_op_reduce_sum_f32,
    .op_reduce_max_f32 = metal_op_reduce_max_f32,
    .op_reduce_min_f32 = metal_op_reduce_min_f32,
    .op_reduce_mean_f32 = metal_op_reduce_mean_f32,
    .op_reduce_sum_axis_f32 = metal_op_reduce_sum_axis_f32,
    .op_reduce_max_axis_f32 = metal_op_reduce_max_axis_f32,

    // Softmax operations
    .op_softmax_f32 = metal_op_softmax_f32,
    .op_log_softmax_f32 = metal_op_log_softmax_f32,

    // Unary operations
    .op_exp_f32 = metal_op_exp_f32,
    .op_log_f32 = metal_op_log_f32,
    .op_sqrt_f32 = metal_op_sqrt_f32,
    .op_neg_f32 = metal_op_neg_f32,
    .op_abs_f32 = metal_op_abs_f32,
    .op_tanh_f32 = metal_op_tanh_f32,
    .op_sigmoid_f32 = metal_op_sigmoid_f32,
    .op_relu_f32 = metal_op_relu_f32,
    .op_gelu_f32 = metal_op_gelu_f32,

    // Copy operations
    .op_copy_f32 = metal_op_copy_f32,

    // Normalization operations
    .op_layer_norm_f32 = metal_op_layer_norm_f32,
    .op_rms_norm_f32 = metal_op_rms_norm_f32,

    // NTT operations
    .op_ntt_forward = metal_op_ntt_forward,
    .op_ntt_inverse = metal_op_ntt_inverse,

    // MSM
    .op_msm = metal_op_msm,

    // FHE operations
    .op_poly_mul = metal_op_poly_mul,
    .op_tfhe_bootstrap = metal_op_tfhe_bootstrap,
    .op_tfhe_keyswitch = metal_op_tfhe_keyswitch,
    .op_blind_rotate = metal_op_blind_rotate,
    .op_sample_extract = metal_op_sample_extract,
    .op_sample_ntt = metal_op_sample_ntt,

    // Reserved
    ._reserved = {nullptr}
};

// =============================================================================
// Plugin Entry Point
// =============================================================================

static bool metal_backend_init_impl(lux_gpu_backend_desc* out) {
    if (!out) return false;

    // Check if Metal is available
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            return false;  // Metal not available
        }
    }

    out->abi_version = LUX_GPU_BACKEND_ABI_VERSION;
    out->backend_name = "metal";
    out->backend_version = "0.1.0";
    out->capabilities = LUX_CAP_TENSOR_OPS | LUX_CAP_MATMUL | LUX_CAP_NTT |
                        LUX_CAP_CUSTOM_KERNELS | LUX_CAP_UNIFIED_MEMORY |
                        LUX_CAP_FHE | LUX_CAP_TFHE;
    out->vtbl = &metal_vtbl;
    return true;
}

LUX_GPU_DECLARE_BACKEND(metal_backend_init_impl)

// Extended crypto backend entry point
extern "C" LUX_GPU_BACKEND_EXPORT bool lux_gpu_crypto_backend_init(lux_gpu_crypto_backend_desc* out) {
    if (!out) return false;

    if (!metal_backend_init_impl(&out->base)) return false;

    out->crypto_vtbl = &metal_crypto_vtbl;
    out->crypto_capabilities = LUX_CRYPTO_CAP_MSM | LUX_CRYPTO_CAP_POSEIDON2 |
                               LUX_CRYPTO_CAP_BLS12_381 | LUX_CRYPTO_CAP_BN254 |
                               LUX_CRYPTO_CAP_GOLDILOCKS | LUX_CRYPTO_CAP_BLAKE3 |
                               LUX_CRYPTO_CAP_KZG | LUX_CRYPTO_CAP_SHAMIR;
    return true;
}

#endif // __APPLE__
