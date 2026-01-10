// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Metal Backend - Apple GPU acceleration

#if defined(__APPLE__) && defined(LUX_HAS_METAL)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../../src/backend.hpp"
#include <unordered_map>
#include <string>

class MetalBackend : public Backend {
public:
    explicit MetalBackend(int device_index = 0) {
        @autoreleasepool {
            NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
            if (devices.count == 0) {
                device_ = MTLCreateSystemDefaultDevice();
            } else if (device_index < (int)devices.count) {
                device_ = devices[device_index];
            } else {
                device_ = devices[0];
            }

            if (device_) {
                queue_ = [device_ newCommandQueue];
                load_kernels();
            }
        }
    }

    ~MetalBackend() override = default;

    LuxBackend type() const override { return LUX_BACKEND_METAL; }
    const char* name() const override { return "metal"; }

    LuxError sync() override {
        @autoreleasepool {
            if (!queue_) return LUX_ERROR_BACKEND_NOT_AVAILABLE;
            id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
            [cmd commit];
            [cmd waitUntilCompleted];
            return LUX_OK;
        }
    }

    LuxError device_info(LuxDeviceInfo* info) override {
        if (!info || !device_) return LUX_ERROR_INVALID_ARGUMENT;

        info->backend = LUX_BACKEND_METAL;
        info->index = 0;
        device_name_ = std::string([[device_ name] UTF8String]);
        info->name = device_name_.c_str();
        info->vendor = "Apple";
        info->memory_total = [device_ recommendedMaxWorkingSetSize];
        info->memory_available = info->memory_total;
        info->is_discrete = ![device_ hasUnifiedMemory];
        info->is_unified_memory = [device_ hasUnifiedMemory];
        info->compute_units = 0;
        info->max_workgroup_size = (int)[device_ maxThreadsPerThreadgroup].width;

        return LUX_OK;
    }

    LuxTensor* zeros(const int64_t* shape, int ndim, LuxDtype dtype) override {
        auto t = new LuxTensor();
        t->shape.assign(shape, shape + ndim);
        t->dtype = dtype;
        size_t bytes = t->size() * t->element_size();
        t->data.resize(bytes, 0);

        @autoreleasepool {
            t->device_ptr = (__bridge_retained void*)[device_ newBufferWithBytes:t->data.data()
                                                                          length:bytes
                                                                         options:MTLResourceStorageModeShared];
        }
        return t;
    }

    LuxTensor* ones(const int64_t* shape, int ndim, LuxDtype dtype) override {
        return full(shape, ndim, dtype, 1.0);
    }

    LuxTensor* full(const int64_t* shape, int ndim, LuxDtype dtype, double value) override {
        auto t = zeros(shape, ndim, dtype);
        if (dtype == LUX_FLOAT32) {
            float v = static_cast<float>(value);
            float* ptr = reinterpret_cast<float*>(t->data.data());
            for (int64_t i = 0; i < t->size(); i++) ptr[i] = v;
            sync_to_device(t);
        }
        return t;
    }

    LuxTensor* from_data(const void* data, const int64_t* shape, int ndim, LuxDtype dtype) override {
        auto t = new LuxTensor();
        t->shape.assign(shape, shape + ndim);
        t->dtype = dtype;
        size_t bytes = t->size() * t->element_size();
        t->data.resize(bytes);
        std::memcpy(t->data.data(), data, bytes);

        @autoreleasepool {
            t->device_ptr = (__bridge_retained void*)[device_ newBufferWithBytes:data
                                                                          length:bytes
                                                                         options:MTLResourceStorageModeShared];
        }
        return t;
    }

    void sync_to_device(LuxTensor* t) override {
        if (t && t->device_ptr) {
            @autoreleasepool {
                id<MTLBuffer> buf = (__bridge id<MTLBuffer>)t->device_ptr;
                memcpy([buf contents], t->data.data(), t->data.size());
            }
        }
    }

    void destroy_tensor(LuxTensor* t) override {
        if (t && t->device_ptr) {
            @autoreleasepool {
                id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)t->device_ptr;
                buf = nil;
            }
        }
        delete t;
    }

    LuxTensor* add(LuxTensor* a, LuxTensor* b) override {
        return binary_op(a, b, "add_float32");
    }

    LuxTensor* sub(LuxTensor* a, LuxTensor* b) override {
        return binary_op(a, b, "sub_float32");
    }

    LuxTensor* mul(LuxTensor* a, LuxTensor* b) override {
        return binary_op(a, b, "mul_float32");
    }

    LuxTensor* matmul(LuxTensor* a, LuxTensor* b) override {
        if (a->shape.size() != 2 || b->shape.size() != 2) return nullptr;
        if (a->shape[1] != b->shape[0]) return nullptr;
        if (a->dtype != LUX_FLOAT32) return nullptr;

        int64_t M = a->shape[0];
        int64_t K = a->shape[1];
        int64_t N = b->shape[1];

        int64_t out_shape[2] = {M, N};
        auto out = zeros(out_shape, 2, a->dtype);

        // CPU fallback matmul (Metal compute would use custom kernel)
        const float* pa = reinterpret_cast<const float*>(a->data.data());
        const float* pb = reinterpret_cast<const float*>(b->data.data());
        float* po = reinterpret_cast<float*>(out->data.data());

        for (int64_t i = 0; i < M; i++) {
            for (int64_t j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int64_t k = 0; k < K; k++) {
                    sum += pa[i * K + k] * pb[k * N + j];
                }
                po[i * N + j] = sum;
            }
        }

        // Update GPU buffer
        @autoreleasepool {
            id<MTLBuffer> buf = (__bridge id<MTLBuffer>)out->device_ptr;
            memcpy([buf contents], out->data.data(), out->data.size());
        }

        return out;
    }

    LuxError bls_verify_batch(const uint8_t* const*, const size_t*,
                              const uint8_t* const*, const size_t*,
                              const uint8_t* const*, const size_t*,
                              int, bool*) override {
        return LUX_ERROR_NOT_SUPPORTED;
    }

    LuxError ntt_forward(uint64_t* data, size_t n, uint64_t modulus) override {
        if (!data || n == 0 || (n & (n - 1)) != 0) return LUX_ERROR_INVALID_ARGUMENT;

        // CPU implementation (Metal kernel would be faster for large n)
        uint64_t g = find_primitive_root(n, modulus);
        if (g == 0) return LUX_ERROR_INVALID_ARGUMENT;

        // omega_n = g^((p-1)/n) is the principal n-th root of unity
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
        return LUX_OK;
    }

    LuxError ntt_inverse(uint64_t* data, size_t n, uint64_t modulus) override {
        if (!data || n == 0 || (n & (n - 1)) != 0) return LUX_ERROR_INVALID_ARGUMENT;

        // Compute inverse n-th root of unity
        uint64_t g = find_primitive_root(n, modulus);
        uint64_t omega_n = mod_pow(g, (modulus - 1) / n, modulus);
        uint64_t omega_n_inv = mod_pow(omega_n, modulus - 2, modulus);

        // Decimation-in-frequency butterfly (large to small)
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

        // Bit-reversal at end
        bit_reverse(data, n);

        // Scale by n^-1
        uint64_t n_inv = mod_pow(n, modulus - 2, modulus);
        for (size_t i = 0; i < n; i++) {
            data[i] = mod_mul(data[i], n_inv, modulus);
        }
        return LUX_OK;
    }

    bool is_available() const { return device_ != nil; }

private:
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    std::unordered_map<std::string, id<MTLComputePipelineState>> kernels_;
    std::string device_name_;

    void load_kernels() {
        @autoreleasepool {
            NSError* error = nil;

            // Compile embedded kernels
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
)";
            id<MTLLibrary> lib = [device_ newLibraryWithSource:source options:nil error:&error];

            if (lib) {
                const char* kernel_names[] = {"add_float32", "sub_float32", "mul_float32"};
                for (const char* name : kernel_names) {
                    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
                    if (fn) {
                        id<MTLComputePipelineState> pso = [device_ newComputePipelineStateWithFunction:fn error:&error];
                        if (pso) {
                            kernels_[name] = pso;
                        }
                    }
                }
            }
        }
    }

    LuxTensor* binary_op(LuxTensor* a, LuxTensor* b, const std::string& kernel_name) {
        if (a->shape != b->shape || a->dtype != b->dtype) return nullptr;
        if (a->dtype != LUX_FLOAT32) return nullptr;

        auto it = kernels_.find(kernel_name);
        auto out = zeros(a->shape.data(), static_cast<int>(a->shape.size()), a->dtype);

        if (it == kernels_.end()) {
            // CPU fallback
            const float* pa = reinterpret_cast<const float*>(a->data.data());
            const float* pb = reinterpret_cast<const float*>(b->data.data());
            float* po = reinterpret_cast<float*>(out->data.data());
            int64_t n = a->size();

            if (kernel_name == "add_float32") {
                for (int64_t i = 0; i < n; i++) po[i] = pa[i] + pb[i];
            } else if (kernel_name == "sub_float32") {
                for (int64_t i = 0; i < n; i++) po[i] = pa[i] - pb[i];
            } else if (kernel_name == "mul_float32") {
                for (int64_t i = 0; i < n; i++) po[i] = pa[i] * pb[i];
            }
            return out;
        }

        @autoreleasepool {
            id<MTLBuffer> buf_a = (__bridge id<MTLBuffer>)a->device_ptr;
            id<MTLBuffer> buf_b = (__bridge id<MTLBuffer>)b->device_ptr;
            id<MTLBuffer> buf_out = (__bridge id<MTLBuffer>)out->device_ptr;

            id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

            [enc setComputePipelineState:it->second];
            [enc setBuffer:buf_a offset:0 atIndex:0];
            [enc setBuffer:buf_b offset:0 atIndex:1];
            [enc setBuffer:buf_out offset:0 atIndex:2];

            NSUInteger n = static_cast<NSUInteger>(a->size());
            MTLSize grid = MTLSizeMake(n, 1, 1);
            NSUInteger threadGroupSize = MIN(n, 256);
            MTLSize group = MTLSizeMake(threadGroupSize, 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:group];
            [enc endEncoding];

            [cmd commit];
            [cmd waitUntilCompleted];

            // Sync back to host
            memcpy(out->data.data(), [buf_out contents], out->data.size());
        }

        return out;
    }

    static uint64_t mod_add(uint64_t a, uint64_t b, uint64_t m) {
        return ((__uint128_t)a + b) % m;
    }

    static uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t m) {
        a %= m;
        b %= m;
        return (a >= b) ? (a - b) : (m - (b - a));
    }

    static uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t m) {
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
};

// Factory function
std::unique_ptr<Backend> create_metal_backend(int device_index) {
    auto backend = std::make_unique<MetalBackend>(device_index);
    return backend->is_available() ? std::move(backend) : nullptr;
}

#endif // __APPLE__ && LUX_HAS_METAL
