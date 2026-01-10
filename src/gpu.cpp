// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco

#include "backend.hpp"
#include <cstring>

// =============================================================================
// Backend Detection
// =============================================================================

static bool check_metal_available() {
#ifdef LUX_HAS_METAL
    return true;  // Actual check in metal_backend.mm
#else
    return false;
#endif
}

static bool check_cuda_available() {
#ifdef LUX_HAS_CUDA
    return true;  // Actual check in cuda_backend.cpp
#else
    return false;
#endif
}

static bool check_dawn_available() {
#ifdef LUX_HAS_DAWN
    return true;  // Actual check in dawn_backend.cpp
#else
    return false;
#endif
}

static std::unique_ptr<Backend> create_backend(LuxBackend type, int device_index) {
    switch (type) {
#ifdef LUX_HAS_METAL
        case LUX_BACKEND_METAL:
            return create_metal_backend(device_index);
#endif
#ifdef LUX_HAS_CUDA
        case LUX_BACKEND_CUDA:
            return create_cuda_backend(device_index);
#endif
#ifdef LUX_HAS_DAWN
        case LUX_BACKEND_DAWN:
            return create_dawn_backend(device_index);
#endif
        case LUX_BACKEND_CPU:
        default:
            return create_cpu_backend();
    }
}

static LuxBackend auto_detect_backend() {
#ifdef LUX_HAS_METAL
    if (check_metal_available()) return LUX_BACKEND_METAL;
#endif
#ifdef LUX_HAS_CUDA
    if (check_cuda_available()) return LUX_BACKEND_CUDA;
#endif
#ifdef LUX_HAS_DAWN
    if (check_dawn_available()) return LUX_BACKEND_DAWN;
#endif
    return LUX_BACKEND_CPU;
}

// =============================================================================
// C API Implementation
// =============================================================================

extern "C" {

LuxGPU* lux_gpu_create(void) {
    return lux_gpu_create_with_backend(LUX_BACKEND_AUTO);
}

LuxGPU* lux_gpu_create_with_backend(LuxBackend backend) {
    return lux_gpu_create_with_device(backend, 0);
}

LuxGPU* lux_gpu_create_with_device(LuxBackend backend, int device_index) {
    auto gpu = new LuxGPU();

    LuxBackend actual = (backend == LUX_BACKEND_AUTO) ? auto_detect_backend() : backend;
    gpu->backend = create_backend(actual, device_index);

    if (!gpu->backend) {
        gpu->set_error("Failed to create backend");
        // Fall back to CPU
        gpu->backend = create_cpu_backend();
    }

    return gpu;
}

void lux_gpu_destroy(LuxGPU* gpu) {
    delete gpu;
}

LuxBackend lux_gpu_backend(LuxGPU* gpu) {
    return gpu && gpu->backend ? gpu->backend->type() : LUX_BACKEND_CPU;
}

const char* lux_gpu_backend_name(LuxGPU* gpu) {
    return gpu && gpu->backend ? gpu->backend->name() : "cpu";
}

LuxError lux_gpu_set_backend(LuxGPU* gpu, LuxBackend backend) {
    if (!gpu) return LUX_ERROR_INVALID_ARGUMENT;

    LuxBackend actual = (backend == LUX_BACKEND_AUTO) ? auto_detect_backend() : backend;

    // Check availability
    switch (actual) {
        case LUX_BACKEND_METAL:
            if (!check_metal_available()) {
                gpu->set_error("Metal backend not available");
                return LUX_ERROR_BACKEND_NOT_AVAILABLE;
            }
            break;
        case LUX_BACKEND_CUDA:
            if (!check_cuda_available()) {
                gpu->set_error("CUDA backend not available");
                return LUX_ERROR_BACKEND_NOT_AVAILABLE;
            }
            break;
        case LUX_BACKEND_DAWN:
            if (!check_dawn_available()) {
                gpu->set_error("Dawn backend not available");
                return LUX_ERROR_BACKEND_NOT_AVAILABLE;
            }
            break;
        default:
            break;
    }

    gpu->backend = create_backend(actual, 0);
    return gpu->backend ? LUX_OK : LUX_ERROR_BACKEND_NOT_AVAILABLE;
}

LuxError lux_gpu_device_info(LuxGPU* gpu, LuxDeviceInfo* info) {
    if (!gpu || !gpu->backend || !info) return LUX_ERROR_INVALID_ARGUMENT;
    return gpu->backend->device_info(info);
}

LuxError lux_gpu_sync(LuxGPU* gpu) {
    if (!gpu || !gpu->backend) return LUX_ERROR_INVALID_ARGUMENT;
    return gpu->backend->sync();
}

const char* lux_gpu_error(LuxGPU* gpu) {
    return gpu ? gpu->last_error.c_str() : "null gpu";
}

// Backend query
int lux_backend_count(void) {
    int count = 1;  // CPU always available
#ifdef LUX_HAS_METAL
    if (check_metal_available()) count++;
#endif
#ifdef LUX_HAS_CUDA
    if (check_cuda_available()) count++;
#endif
#ifdef LUX_HAS_DAWN
    if (check_dawn_available()) count++;
#endif
    return count;
}

bool lux_backend_available(LuxBackend backend) {
    switch (backend) {
        case LUX_BACKEND_CPU: return true;
        case LUX_BACKEND_METAL: return check_metal_available();
        case LUX_BACKEND_CUDA: return check_cuda_available();
        case LUX_BACKEND_DAWN: return check_dawn_available();
        default: return false;
    }
}

const char* lux_backend_name(LuxBackend backend) {
    switch (backend) {
        case LUX_BACKEND_AUTO: return "auto";
        case LUX_BACKEND_CPU: return "cpu";
        case LUX_BACKEND_METAL: return "metal";
        case LUX_BACKEND_CUDA: return "cuda";
        case LUX_BACKEND_DAWN: return "dawn";
        default: return "unknown";
    }
}

// Tensor operations
LuxTensor* lux_tensor_zeros(LuxGPU* gpu, const int64_t* shape, int ndim, LuxDtype dtype) {
    if (!gpu || !gpu->backend || !shape || ndim <= 0) return nullptr;
    return gpu->backend->zeros(shape, ndim, dtype);
}

LuxTensor* lux_tensor_ones(LuxGPU* gpu, const int64_t* shape, int ndim, LuxDtype dtype) {
    if (!gpu || !gpu->backend || !shape || ndim <= 0) return nullptr;
    return gpu->backend->ones(shape, ndim, dtype);
}

LuxTensor* lux_tensor_full(LuxGPU* gpu, const int64_t* shape, int ndim, LuxDtype dtype, double value) {
    if (!gpu || !gpu->backend || !shape || ndim <= 0) return nullptr;
    return gpu->backend->full(shape, ndim, dtype, value);
}

LuxTensor* lux_tensor_from_data(LuxGPU* gpu, const void* data, const int64_t* shape, int ndim, LuxDtype dtype) {
    if (!gpu || !gpu->backend || !data || !shape || ndim <= 0) return nullptr;
    return gpu->backend->from_data(data, shape, ndim, dtype);
}

void lux_tensor_destroy(LuxTensor* tensor) {
    delete tensor;
}

int lux_tensor_ndim(LuxTensor* tensor) {
    return tensor ? static_cast<int>(tensor->shape.size()) : 0;
}

int64_t lux_tensor_shape(LuxTensor* tensor, int dim) {
    return (tensor && dim >= 0 && dim < static_cast<int>(tensor->shape.size()))
        ? tensor->shape[dim] : 0;
}

int64_t lux_tensor_size(LuxTensor* tensor) {
    return tensor ? tensor->size() : 0;
}

LuxDtype lux_tensor_dtype(LuxTensor* tensor) {
    return tensor ? tensor->dtype : LUX_FLOAT32;
}

LuxError lux_tensor_to_host(LuxTensor* tensor, void* data, size_t size) {
    if (!tensor || !data) return LUX_ERROR_INVALID_ARGUMENT;
    size_t bytes = tensor->size() * tensor->element_size();
    if (size < bytes) return LUX_ERROR_INVALID_ARGUMENT;
    std::memcpy(data, tensor->data.data(), bytes);
    return LUX_OK;
}

LuxTensor* lux_tensor_add(LuxGPU* gpu, LuxTensor* a, LuxTensor* b) {
    if (!gpu || !gpu->backend || !a || !b) return nullptr;
    return gpu->backend->add(a, b);
}

LuxTensor* lux_tensor_sub(LuxGPU* gpu, LuxTensor* a, LuxTensor* b) {
    if (!gpu || !gpu->backend || !a || !b) return nullptr;
    return gpu->backend->sub(a, b);
}

LuxTensor* lux_tensor_mul(LuxGPU* gpu, LuxTensor* a, LuxTensor* b) {
    if (!gpu || !gpu->backend || !a || !b) return nullptr;
    return gpu->backend->mul(a, b);
}

LuxTensor* lux_tensor_matmul(LuxGPU* gpu, LuxTensor* a, LuxTensor* b) {
    if (!gpu || !gpu->backend || !a || !b) return nullptr;
    return gpu->backend->matmul(a, b);
}

// NTT
LuxError lux_ntt_forward(LuxGPU* gpu, uint64_t* data, size_t n, uint64_t modulus) {
    if (!gpu || !gpu->backend || !data) return LUX_ERROR_INVALID_ARGUMENT;
    return gpu->backend->ntt_forward(data, n, modulus);
}

LuxError lux_ntt_inverse(LuxGPU* gpu, uint64_t* data, size_t n, uint64_t modulus) {
    if (!gpu || !gpu->backend || !data) return LUX_ERROR_INVALID_ARGUMENT;
    return gpu->backend->ntt_inverse(data, n, modulus);
}

} // extern "C"
