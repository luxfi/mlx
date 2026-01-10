// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Internal Backend interface header

#ifndef LUX_GPU_BACKEND_HPP
#define LUX_GPU_BACKEND_HPP

#include "lux/gpu.h"
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Tensor implementation
struct LuxTensor {
    std::vector<int64_t> shape;
    LuxDtype dtype;
    std::vector<uint8_t> data;  // Host data
    void* device_ptr = nullptr; // Device-specific pointer

    int64_t size() const {
        int64_t s = 1;
        for (auto d : shape) s *= d;
        return s;
    }

    size_t element_size() const {
        switch (dtype) {
            case LUX_FLOAT32: case LUX_INT32: case LUX_UINT32: return 4;
            case LUX_FLOAT16: case LUX_BFLOAT16: return 2;
            case LUX_INT64: case LUX_UINT64: return 8;
            case LUX_BOOL: return 1;
            default: return 0;
        }
    }
};

// Backend interface (implemented by each backend)
class Backend {
public:
    virtual ~Backend() = default;
    virtual LuxBackend type() const = 0;
    virtual const char* name() const = 0;
    virtual LuxError sync() = 0;
    virtual LuxError device_info(LuxDeviceInfo* info) = 0;

    // Tensor operations
    virtual LuxTensor* zeros(const int64_t* shape, int ndim, LuxDtype dtype) = 0;
    virtual LuxTensor* ones(const int64_t* shape, int ndim, LuxDtype dtype) = 0;
    virtual LuxTensor* full(const int64_t* shape, int ndim, LuxDtype dtype, double value) = 0;
    virtual LuxTensor* from_data(const void* data, const int64_t* shape, int ndim, LuxDtype dtype) = 0;
    virtual void sync_to_device(LuxTensor* t) = 0;  // Copy host data to device
    virtual void destroy_tensor(LuxTensor* t) = 0;

    virtual LuxTensor* add(LuxTensor* a, LuxTensor* b) = 0;
    virtual LuxTensor* sub(LuxTensor* a, LuxTensor* b) = 0;
    virtual LuxTensor* mul(LuxTensor* a, LuxTensor* b) = 0;
    virtual LuxTensor* matmul(LuxTensor* a, LuxTensor* b) = 0;

    // Crypto
    virtual LuxError bls_verify_batch(const uint8_t* const* sigs, const size_t* sig_lens,
                                      const uint8_t* const* msgs, const size_t* msg_lens,
                                      const uint8_t* const* pubkeys, const size_t* pubkey_lens,
                                      int count, bool* results) = 0;

    // NTT
    virtual LuxError ntt_forward(uint64_t* data, size_t n, uint64_t modulus) = 0;
    virtual LuxError ntt_inverse(uint64_t* data, size_t n, uint64_t modulus) = 0;
};

// GPU context implementation
struct LuxGPU {
    std::unique_ptr<Backend> backend;
    std::string last_error;
    std::mutex mutex;

    LuxGPU() = default;
    ~LuxGPU() = default;

    void set_error(const char* msg) {
        std::lock_guard<std::mutex> lock(mutex);
        last_error = msg ? msg : "";
    }
};

// Factory functions (implemented by each backend)
std::unique_ptr<Backend> create_cpu_backend();

#ifdef LUX_HAS_METAL
std::unique_ptr<Backend> create_metal_backend(int device_index);
#endif

#ifdef LUX_HAS_CUDA
std::unique_ptr<Backend> create_cuda_backend(int device_index);
#endif

#ifdef LUX_HAS_DAWN
std::unique_ptr<Backend> create_dawn_backend(int device_index);
#endif

#endif // LUX_GPU_BACKEND_HPP
