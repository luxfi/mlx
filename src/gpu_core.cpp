// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Core GPU Library - Plugin-based backend management

#include "lux/gpu.h"
#include "lux/gpu/backend_plugin.h"
#include "plugin_loader.hpp"
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

// =============================================================================
// Built-in CPU backend declaration
// =============================================================================

extern "C" bool cpu_backend_init(lux_gpu_backend_desc* out);

// =============================================================================
// GPU Context Implementation
// =============================================================================

struct LuxGPU {
    std::string backend_name;
    const lux_gpu_backend_vtbl* vtbl = nullptr;
    LuxBackendContext* ctx = nullptr;
    std::string last_error;
    std::mutex mutex;

    ~LuxGPU() {
        if (ctx && vtbl && vtbl->destroy_context) {
            vtbl->destroy_context(ctx);
        }
    }

    void set_error(const char* msg) {
        std::lock_guard<std::mutex> lock(mutex);
        last_error = msg ? msg : "";
    }
};

// =============================================================================
// Tensor wrapper (bridges plugin buffers to public API)
// =============================================================================

struct LuxTensor {
    std::vector<int64_t> shape;
    LuxDtype dtype;
    std::vector<uint8_t> host_data;
    LuxBackendBuffer* device_buffer = nullptr;
    const lux_gpu_backend_vtbl* vtbl = nullptr;
    LuxBackendContext* ctx = nullptr;

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

    ~LuxTensor() {
        if (device_buffer && vtbl && vtbl->buffer_free && ctx) {
            vtbl->buffer_free(ctx, device_buffer);
        }
    }
};

// =============================================================================
// Global initialization
// =============================================================================

static std::once_flag g_init_flag;
static lux_gpu_backend_desc g_cpu_backend = {};

static void global_init() {
    // Initialize built-in CPU backend
    cpu_backend_init(&g_cpu_backend);

    // Scan for plugins in all search paths
    auto& loader = lux::gpu::PluginLoader::instance();

    // Try to load each backend (will search all paths)
    loader.load_backend("metal");
    loader.load_backend("cuda");
    loader.load_backend("webgpu");
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
    std::call_once(g_init_flag, global_init);

    auto gpu = new LuxGPU();
    auto& loader = lux::gpu::PluginLoader::instance();

    const lux_gpu_backend_vtbl* vtbl = nullptr;
    std::string name;

    if (backend == LUX_BACKEND_AUTO) {
        // Try to find the best available backend
        if (auto* best = loader.get_best_backend()) {
            vtbl = best->desc.vtbl;
            name = best->name;
        }
        // Fall back to CPU
        if (!vtbl) {
            vtbl = g_cpu_backend.vtbl;
            name = "cpu";
        }
    } else {
        // Specific backend requested
        switch (backend) {
            case LUX_BACKEND_CPU:
                vtbl = g_cpu_backend.vtbl;
                name = "cpu";
                break;

            case LUX_BACKEND_METAL:
                if (!loader.is_available("metal")) {
                    loader.load_backend("metal");
                }
                if (auto* b = loader.get_backend("metal")) {
                    vtbl = b->desc.vtbl;
                    name = "metal";
                }
                break;

            case LUX_BACKEND_CUDA:
                if (!loader.is_available("cuda")) {
                    loader.load_backend("cuda");
                }
                if (auto* b = loader.get_backend("cuda")) {
                    vtbl = b->desc.vtbl;
                    name = "cuda";
                }
                break;

            case LUX_BACKEND_DAWN:
                if (!loader.is_available("webgpu")) {
                    loader.load_backend("webgpu");
                }
                if (auto* b = loader.get_backend("webgpu")) {
                    vtbl = b->desc.vtbl;
                    name = "webgpu";
                }
                break;

            default:
                break;
        }
    }

    if (!vtbl) {
        // Fall back to CPU
        vtbl = g_cpu_backend.vtbl;
        name = "cpu";
    }

    gpu->vtbl = vtbl;
    gpu->backend_name = name;

    // Create context
    if (vtbl && vtbl->create_context) {
        gpu->ctx = vtbl->create_context(device_index);
        if (!gpu->ctx) {
            gpu->set_error("Failed to create backend context");
            // Fall back to CPU
            gpu->vtbl = g_cpu_backend.vtbl;
            gpu->backend_name = "cpu";
            gpu->ctx = g_cpu_backend.vtbl->create_context(0);
        }
    }

    return gpu;
}

void lux_gpu_destroy(LuxGPU* gpu) {
    delete gpu;
}

LuxBackend lux_gpu_backend(LuxGPU* gpu) {
    if (!gpu) return LUX_BACKEND_CPU;
    if (gpu->backend_name == "cpu") return LUX_BACKEND_CPU;
    if (gpu->backend_name == "metal") return LUX_BACKEND_METAL;
    if (gpu->backend_name == "cuda") return LUX_BACKEND_CUDA;
    if (gpu->backend_name == "webgpu") return LUX_BACKEND_DAWN;
    return LUX_BACKEND_CPU;
}

const char* lux_gpu_backend_name(LuxGPU* gpu) {
    return gpu ? gpu->backend_name.c_str() : "cpu";
}

LuxError lux_gpu_set_backend(LuxGPU* gpu, LuxBackend backend) {
    if (!gpu) return LUX_ERROR_INVALID_ARGUMENT;

    // Create new context for requested backend
    auto* new_gpu = lux_gpu_create_with_backend(backend);
    if (!new_gpu || new_gpu->backend_name == "cpu" && backend != LUX_BACKEND_CPU && backend != LUX_BACKEND_AUTO) {
        delete new_gpu;
        return LUX_ERROR_BACKEND_NOT_AVAILABLE;
    }

    // Swap internals
    std::swap(gpu->vtbl, new_gpu->vtbl);
    std::swap(gpu->ctx, new_gpu->ctx);
    std::swap(gpu->backend_name, new_gpu->backend_name);

    delete new_gpu;
    return LUX_OK;
}

LuxError lux_gpu_device_info(LuxGPU* gpu, LuxDeviceInfo* info) {
    if (!gpu || !gpu->vtbl || !info) return LUX_ERROR_INVALID_ARGUMENT;

    LuxBackendDeviceInfo binfo = {};
    LuxBackendError err = gpu->vtbl->get_device_info(gpu->ctx, &binfo);
    if (err != LUX_BACKEND_OK) return static_cast<LuxError>(err);

    info->backend = lux_gpu_backend(gpu);
    info->index = 0;
    info->name = binfo.name;
    info->vendor = binfo.vendor;
    info->memory_total = binfo.memory_total;
    info->memory_available = binfo.memory_available;
    info->compute_units = binfo.compute_units;
    info->max_workgroup_size = binfo.max_workgroup_size;
    info->is_discrete = binfo.is_discrete;
    info->is_unified_memory = binfo.is_unified_memory;

    return LUX_OK;
}

LuxError lux_gpu_sync(LuxGPU* gpu) {
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    return static_cast<LuxError>(gpu->vtbl->sync(gpu->ctx));
}

const char* lux_gpu_error(LuxGPU* gpu) {
    return gpu ? gpu->last_error.c_str() : "null gpu";
}

// Backend query
int lux_backend_count(void) {
    std::call_once(g_init_flag, global_init);
    return static_cast<int>(lux::gpu::PluginLoader::instance().available_backends().size());
}

bool lux_backend_available(LuxBackend backend) {
    std::call_once(g_init_flag, global_init);
    auto& loader = lux::gpu::PluginLoader::instance();

    switch (backend) {
        case LUX_BACKEND_CPU: return true;
        case LUX_BACKEND_METAL: return loader.is_available("metal") || loader.load_backend("metal");
        case LUX_BACKEND_CUDA: return loader.is_available("cuda") || loader.load_backend("cuda");
        case LUX_BACKEND_DAWN: return loader.is_available("webgpu") || loader.load_backend("webgpu");
        default: return false;
    }
}

const char* lux_backend_name(LuxBackend backend) {
    switch (backend) {
        case LUX_BACKEND_AUTO: return "auto";
        case LUX_BACKEND_CPU: return "cpu";
        case LUX_BACKEND_METAL: return "metal";
        case LUX_BACKEND_CUDA: return "cuda";
        case LUX_BACKEND_DAWN: return "webgpu";
        default: return "unknown";
    }
}

// =============================================================================
// Tensor Operations
// =============================================================================

LuxTensor* lux_tensor_zeros(LuxGPU* gpu, const int64_t* shape, int ndim, LuxDtype dtype) {
    if (!gpu || !gpu->vtbl || !shape || ndim <= 0) return nullptr;

    auto t = new LuxTensor();
    t->shape.assign(shape, shape + ndim);
    t->dtype = dtype;
    t->vtbl = gpu->vtbl;
    t->ctx = gpu->ctx;

    size_t bytes = t->size() * t->element_size();
    t->host_data.resize(bytes, 0);

    if (gpu->vtbl->buffer_alloc_with_data) {
        t->device_buffer = gpu->vtbl->buffer_alloc_with_data(gpu->ctx, t->host_data.data(), bytes);
    } else if (gpu->vtbl->buffer_alloc) {
        t->device_buffer = gpu->vtbl->buffer_alloc(gpu->ctx, bytes);
    }

    return t;
}

LuxTensor* lux_tensor_ones(LuxGPU* gpu, const int64_t* shape, int ndim, LuxDtype dtype) {
    return lux_tensor_full(gpu, shape, ndim, dtype, 1.0);
}

LuxTensor* lux_tensor_full(LuxGPU* gpu, const int64_t* shape, int ndim, LuxDtype dtype, double value) {
    if (!gpu || !gpu->vtbl || !shape || ndim <= 0) return nullptr;

    auto t = new LuxTensor();
    t->shape.assign(shape, shape + ndim);
    t->dtype = dtype;
    t->vtbl = gpu->vtbl;
    t->ctx = gpu->ctx;

    size_t bytes = t->size() * t->element_size();
    t->host_data.resize(bytes);

    // Fill host data
    if (dtype == LUX_FLOAT32) {
        float v = static_cast<float>(value);
        float* ptr = reinterpret_cast<float*>(t->host_data.data());
        for (int64_t i = 0; i < t->size(); i++) ptr[i] = v;
    }

    // Copy to device
    if (gpu->vtbl->buffer_alloc_with_data) {
        t->device_buffer = gpu->vtbl->buffer_alloc_with_data(gpu->ctx, t->host_data.data(), bytes);
    }

    return t;
}

LuxTensor* lux_tensor_from_data(LuxGPU* gpu, const void* data, const int64_t* shape, int ndim, LuxDtype dtype) {
    if (!gpu || !gpu->vtbl || !data || !shape || ndim <= 0) return nullptr;

    auto t = new LuxTensor();
    t->shape.assign(shape, shape + ndim);
    t->dtype = dtype;
    t->vtbl = gpu->vtbl;
    t->ctx = gpu->ctx;

    size_t bytes = t->size() * t->element_size();
    t->host_data.resize(bytes);
    std::memcpy(t->host_data.data(), data, bytes);

    if (gpu->vtbl->buffer_alloc_with_data) {
        t->device_buffer = gpu->vtbl->buffer_alloc_with_data(gpu->ctx, data, bytes);
    }

    return t;
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

    // If we have device buffer, sync from it
    if (tensor->device_buffer && tensor->vtbl && tensor->vtbl->buffer_copy_to_host) {
        LuxBackendError err = tensor->vtbl->buffer_copy_to_host(
            tensor->ctx, tensor->device_buffer, data, bytes
        );
        return static_cast<LuxError>(err);
    }

    // Otherwise copy from host data
    std::memcpy(data, tensor->host_data.data(), bytes);
    return LUX_OK;
}

// Binary operations helper
static LuxTensor* binary_op(LuxGPU* gpu, LuxTensor* a, LuxTensor* b,
                            LuxBackendError (*op)(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, LuxBackendBuffer*, size_t)) {
    if (!gpu || !a || !b || a->shape != b->shape) return nullptr;

    auto out = lux_tensor_zeros(gpu, a->shape.data(), static_cast<int>(a->shape.size()), a->dtype);
    if (!out) return nullptr;

    if (op && a->device_buffer && b->device_buffer && out->device_buffer) {
        LuxBackendError err = op(gpu->ctx, a->device_buffer, b->device_buffer, out->device_buffer, a->size());
        if (err != LUX_BACKEND_OK) {
            delete out;
            return nullptr;
        }
    }

    return out;
}

LuxTensor* lux_tensor_add(LuxGPU* gpu, LuxTensor* a, LuxTensor* b) {
    if (!gpu || !gpu->vtbl) return nullptr;
    return binary_op(gpu, a, b, gpu->vtbl->op_add_f32);
}

LuxTensor* lux_tensor_sub(LuxGPU* gpu, LuxTensor* a, LuxTensor* b) {
    if (!gpu || !gpu->vtbl) return nullptr;
    return binary_op(gpu, a, b, gpu->vtbl->op_sub_f32);
}

LuxTensor* lux_tensor_mul(LuxGPU* gpu, LuxTensor* a, LuxTensor* b) {
    if (!gpu || !gpu->vtbl) return nullptr;
    return binary_op(gpu, a, b, gpu->vtbl->op_mul_f32);
}

LuxTensor* lux_tensor_div(LuxGPU* gpu, LuxTensor* a, LuxTensor* b) {
    if (!gpu || !gpu->vtbl || !gpu->vtbl->op_div_f32) return nullptr;
    return binary_op(gpu, a, b, gpu->vtbl->op_div_f32);
}

LuxTensor* lux_tensor_matmul(LuxGPU* gpu, LuxTensor* a, LuxTensor* b) {
    if (!gpu || !gpu->vtbl || !a || !b) return nullptr;
    if (a->shape.size() != 2 || b->shape.size() != 2) return nullptr;
    if (a->shape[1] != b->shape[0]) return nullptr;

    int M = static_cast<int>(a->shape[0]);
    int K = static_cast<int>(a->shape[1]);
    int N = static_cast<int>(b->shape[1]);

    int64_t out_shape[2] = {M, N};
    auto out = lux_tensor_zeros(gpu, out_shape, 2, a->dtype);
    if (!out) return nullptr;

    if (gpu->vtbl->op_matmul_f32 && a->device_buffer && b->device_buffer && out->device_buffer) {
        LuxBackendError err = gpu->vtbl->op_matmul_f32(
            gpu->ctx, a->device_buffer, b->device_buffer, out->device_buffer, M, K, N
        );
        if (err != LUX_BACKEND_OK) {
            delete out;
            return nullptr;
        }
    }

    return out;
}

// =============================================================================
// Unary operations helper
// =============================================================================

static LuxTensor* unary_op(LuxGPU* gpu, LuxTensor* t,
                           LuxBackendError (*op)(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t)) {
    if (!gpu || !gpu->vtbl || !t || !op) return nullptr;

    auto out = lux_tensor_zeros(gpu, t->shape.data(), static_cast<int>(t->shape.size()), t->dtype);
    if (!out) return nullptr;

    if (t->device_buffer && out->device_buffer) {
        LuxBackendError err = op(gpu->ctx, t->device_buffer, out->device_buffer, t->size());
        if (err != LUX_BACKEND_OK) {
            delete out;
            return nullptr;
        }
    }

    return out;
}

LuxTensor* lux_tensor_neg(LuxGPU* gpu, LuxTensor* t) {
    if (!gpu || !gpu->vtbl || !gpu->vtbl->op_neg_f32) return nullptr;
    return unary_op(gpu, t, gpu->vtbl->op_neg_f32);
}

LuxTensor* lux_tensor_exp(LuxGPU* gpu, LuxTensor* t) {
    if (!gpu || !gpu->vtbl || !gpu->vtbl->op_exp_f32) return nullptr;
    return unary_op(gpu, t, gpu->vtbl->op_exp_f32);
}

LuxTensor* lux_tensor_log(LuxGPU* gpu, LuxTensor* t) {
    if (!gpu || !gpu->vtbl || !gpu->vtbl->op_log_f32) return nullptr;
    return unary_op(gpu, t, gpu->vtbl->op_log_f32);
}

LuxTensor* lux_tensor_sqrt(LuxGPU* gpu, LuxTensor* t) {
    if (!gpu || !gpu->vtbl || !gpu->vtbl->op_sqrt_f32) return nullptr;
    return unary_op(gpu, t, gpu->vtbl->op_sqrt_f32);
}

LuxTensor* lux_tensor_abs(LuxGPU* gpu, LuxTensor* t) {
    if (!gpu || !gpu->vtbl || !gpu->vtbl->op_abs_f32) return nullptr;
    return unary_op(gpu, t, gpu->vtbl->op_abs_f32);
}

LuxTensor* lux_tensor_tanh(LuxGPU* gpu, LuxTensor* t) {
    if (!gpu || !gpu->vtbl || !gpu->vtbl->op_tanh_f32) return nullptr;
    return unary_op(gpu, t, gpu->vtbl->op_tanh_f32);
}

LuxTensor* lux_tensor_sigmoid(LuxGPU* gpu, LuxTensor* t) {
    if (!gpu || !gpu->vtbl || !gpu->vtbl->op_sigmoid_f32) return nullptr;
    return unary_op(gpu, t, gpu->vtbl->op_sigmoid_f32);
}

LuxTensor* lux_tensor_relu(LuxGPU* gpu, LuxTensor* t) {
    if (!gpu || !gpu->vtbl || !gpu->vtbl->op_relu_f32) return nullptr;
    return unary_op(gpu, t, gpu->vtbl->op_relu_f32);
}

LuxTensor* lux_tensor_gelu(LuxGPU* gpu, LuxTensor* t) {
    if (!gpu || !gpu->vtbl || !gpu->vtbl->op_gelu_f32) return nullptr;
    return unary_op(gpu, t, gpu->vtbl->op_gelu_f32);
}

// =============================================================================
// Scalar reductions
// =============================================================================

float lux_tensor_reduce_sum(LuxGPU* gpu, LuxTensor* t) {
    if (!gpu || !gpu->vtbl || !t || !gpu->vtbl->op_reduce_sum_f32) return 0.0f;

    int64_t one = 1;
    auto out = lux_tensor_zeros(gpu, &one, 1, LUX_FLOAT32);
    if (!out) return 0.0f;

    if (t->device_buffer && out->device_buffer) {
        gpu->vtbl->op_reduce_sum_f32(gpu->ctx, t->device_buffer, out->device_buffer, t->size());
    }

    float result = 0.0f;
    lux_tensor_to_host(out, &result, sizeof(float));
    lux_tensor_destroy(out);
    return result;
}

float lux_tensor_reduce_max(LuxGPU* gpu, LuxTensor* t) {
    if (!gpu || !gpu->vtbl || !t || !gpu->vtbl->op_reduce_max_f32) return 0.0f;

    int64_t one = 1;
    auto out = lux_tensor_zeros(gpu, &one, 1, LUX_FLOAT32);
    if (!out) return 0.0f;

    if (t->device_buffer && out->device_buffer) {
        gpu->vtbl->op_reduce_max_f32(gpu->ctx, t->device_buffer, out->device_buffer, t->size());
    }

    float result = 0.0f;
    lux_tensor_to_host(out, &result, sizeof(float));
    lux_tensor_destroy(out);
    return result;
}

float lux_tensor_reduce_min(LuxGPU* gpu, LuxTensor* t) {
    if (!gpu || !gpu->vtbl || !t || !gpu->vtbl->op_reduce_min_f32) return 0.0f;

    int64_t one = 1;
    auto out = lux_tensor_zeros(gpu, &one, 1, LUX_FLOAT32);
    if (!out) return 0.0f;

    if (t->device_buffer && out->device_buffer) {
        gpu->vtbl->op_reduce_min_f32(gpu->ctx, t->device_buffer, out->device_buffer, t->size());
    }

    float result = 0.0f;
    lux_tensor_to_host(out, &result, sizeof(float));
    lux_tensor_destroy(out);
    return result;
}

float lux_tensor_reduce_mean(LuxGPU* gpu, LuxTensor* t) {
    if (!gpu || !gpu->vtbl || !t || !gpu->vtbl->op_reduce_mean_f32) return 0.0f;

    int64_t one = 1;
    auto out = lux_tensor_zeros(gpu, &one, 1, LUX_FLOAT32);
    if (!out) return 0.0f;

    if (t->device_buffer && out->device_buffer) {
        gpu->vtbl->op_reduce_mean_f32(gpu->ctx, t->device_buffer, out->device_buffer, t->size());
    }

    float result = 0.0f;
    lux_tensor_to_host(out, &result, sizeof(float));
    lux_tensor_destroy(out);
    return result;
}

// Reductions along axes (stubs - not yet fully implemented)
LuxTensor* lux_tensor_sum(LuxGPU* gpu, LuxTensor* t, const int* axes, int naxes) {
    (void)gpu; (void)axes; (void)naxes;
    // Return copy of input for now - proper reduction not yet implemented
    if (!t) return nullptr;
    return lux_tensor_from_data(gpu, t->host_data.data(), t->shape.data(),
                                 static_cast<int>(t->shape.size()), t->dtype);
}

LuxTensor* lux_tensor_mean(LuxGPU* gpu, LuxTensor* t, const int* axes, int naxes) {
    (void)gpu; (void)axes; (void)naxes;
    // Return copy of input for now - proper reduction not yet implemented
    if (!t) return nullptr;
    return lux_tensor_from_data(gpu, t->host_data.data(), t->shape.data(),
                                 static_cast<int>(t->shape.size()), t->dtype);
}

LuxTensor* lux_tensor_max(LuxGPU* gpu, LuxTensor* t, const int* axes, int naxes) {
    (void)gpu; (void)axes; (void)naxes;
    if (!t) return nullptr;
    return lux_tensor_from_data(gpu, t->host_data.data(), t->shape.data(),
                                 static_cast<int>(t->shape.size()), t->dtype);
}

LuxTensor* lux_tensor_min(LuxGPU* gpu, LuxTensor* t, const int* axes, int naxes) {
    (void)gpu; (void)axes; (void)naxes;
    if (!t) return nullptr;
    return lux_tensor_from_data(gpu, t->host_data.data(), t->shape.data(),
                                 static_cast<int>(t->shape.size()), t->dtype);
}

// =============================================================================
// Softmax and normalization
// =============================================================================

LuxTensor* lux_tensor_softmax(LuxGPU* gpu, LuxTensor* t, int axis) {
    if (!gpu || !gpu->vtbl || !t || !gpu->vtbl->op_softmax_f32) return nullptr;
    if (t->shape.size() < 1) return nullptr;

    // For now, only support last axis softmax
    (void)axis;
    size_t cols = static_cast<size_t>(t->shape.back());
    size_t rows = static_cast<size_t>(t->size() / cols);

    auto out = lux_tensor_zeros(gpu, t->shape.data(), static_cast<int>(t->shape.size()), t->dtype);
    if (!out) return nullptr;

    if (t->device_buffer && out->device_buffer) {
        LuxBackendError err = gpu->vtbl->op_softmax_f32(gpu->ctx, t->device_buffer, out->device_buffer, rows, cols);
        if (err != LUX_BACKEND_OK) {
            delete out;
            return nullptr;
        }
    }

    return out;
}

LuxTensor* lux_tensor_log_softmax(LuxGPU* gpu, LuxTensor* t, int axis) {
    if (!gpu || !gpu->vtbl || !t || !gpu->vtbl->op_log_softmax_f32) return nullptr;
    if (t->shape.size() < 1) return nullptr;

    (void)axis;
    size_t cols = static_cast<size_t>(t->shape.back());
    size_t rows = static_cast<size_t>(t->size() / cols);

    auto out = lux_tensor_zeros(gpu, t->shape.data(), static_cast<int>(t->shape.size()), t->dtype);
    if (!out) return nullptr;

    if (t->device_buffer && out->device_buffer) {
        LuxBackendError err = gpu->vtbl->op_log_softmax_f32(gpu->ctx, t->device_buffer, out->device_buffer, rows, cols);
        if (err != LUX_BACKEND_OK) {
            delete out;
            return nullptr;
        }
    }

    return out;
}

LuxTensor* lux_tensor_layer_norm(LuxGPU* gpu, LuxTensor* t, LuxTensor* gamma, LuxTensor* beta, float eps) {
    if (!gpu || !gpu->vtbl || !t || !gpu->vtbl->op_layer_norm_f32) return nullptr;
    if (t->shape.size() < 1) return nullptr;

    size_t dim = static_cast<size_t>(t->shape.back());
    size_t batch_size = static_cast<size_t>(t->size() / dim);

    auto out = lux_tensor_zeros(gpu, t->shape.data(), static_cast<int>(t->shape.size()), t->dtype);
    if (!out) return nullptr;

    LuxBackendBuffer* gamma_buf = gamma ? gamma->device_buffer : nullptr;
    LuxBackendBuffer* beta_buf = beta ? beta->device_buffer : nullptr;

    if (t->device_buffer && out->device_buffer) {
        LuxBackendError err = gpu->vtbl->op_layer_norm_f32(
            gpu->ctx, t->device_buffer, out->device_buffer,
            gamma_buf, beta_buf, batch_size, dim, eps
        );
        if (err != LUX_BACKEND_OK) {
            delete out;
            return nullptr;
        }
    }

    return out;
}

LuxTensor* lux_tensor_rms_norm(LuxGPU* gpu, LuxTensor* t, LuxTensor* weight, float eps) {
    if (!gpu || !gpu->vtbl || !t || !gpu->vtbl->op_rms_norm_f32) return nullptr;
    if (t->shape.size() < 1) return nullptr;

    size_t dim = static_cast<size_t>(t->shape.back());
    size_t batch_size = static_cast<size_t>(t->size() / dim);

    auto out = lux_tensor_zeros(gpu, t->shape.data(), static_cast<int>(t->shape.size()), t->dtype);
    if (!out) return nullptr;

    LuxBackendBuffer* weight_buf = weight ? weight->device_buffer : nullptr;

    if (t->device_buffer && out->device_buffer) {
        LuxBackendError err = gpu->vtbl->op_rms_norm_f32(
            gpu->ctx, t->device_buffer, out->device_buffer,
            weight_buf, batch_size, dim, eps
        );
        if (err != LUX_BACKEND_OK) {
            delete out;
            return nullptr;
        }
    }

    return out;
}

// =============================================================================
// Transpose and copy
// =============================================================================

LuxTensor* lux_tensor_transpose(LuxGPU* gpu, LuxTensor* t) {
    if (!gpu || !gpu->vtbl || !t || !gpu->vtbl->op_transpose_f32) return nullptr;
    if (t->shape.size() != 2) return nullptr;

    int rows = static_cast<int>(t->shape[0]);
    int cols = static_cast<int>(t->shape[1]);

    int64_t out_shape[2] = {cols, rows};
    auto out = lux_tensor_zeros(gpu, out_shape, 2, t->dtype);
    if (!out) return nullptr;

    if (t->device_buffer && out->device_buffer) {
        LuxBackendError err = gpu->vtbl->op_transpose_f32(gpu->ctx, t->device_buffer, out->device_buffer, rows, cols);
        if (err != LUX_BACKEND_OK) {
            delete out;
            return nullptr;
        }
    }

    return out;
}

LuxTensor* lux_tensor_copy(LuxGPU* gpu, LuxTensor* t) {
    if (!gpu || !gpu->vtbl || !t || !gpu->vtbl->op_copy_f32) return nullptr;

    auto out = lux_tensor_zeros(gpu, t->shape.data(), static_cast<int>(t->shape.size()), t->dtype);
    if (!out) return nullptr;

    if (t->device_buffer && out->device_buffer) {
        LuxBackendError err = gpu->vtbl->op_copy_f32(gpu->ctx, t->device_buffer, out->device_buffer, t->size());
        if (err != LUX_BACKEND_OK) {
            delete out;
            return nullptr;
        }
    }

    return out;
}

// Stream/Event Management (stubs)
LuxStream* lux_stream_create(LuxGPU* gpu) {
    (void)gpu;
    return nullptr;  // Not yet implemented
}

void lux_stream_destroy(LuxStream* stream) {
    (void)stream;
}

LuxError lux_stream_sync(LuxStream* stream) {
    (void)stream;
    return LUX_OK;
}

LuxEvent* lux_event_create(LuxGPU* gpu) {
    (void)gpu;
    return nullptr;
}

void lux_event_destroy(LuxEvent* event) {
    (void)event;
}

LuxError lux_event_record(LuxEvent* event, LuxStream* stream) {
    (void)event; (void)stream;
    return LUX_OK;
}

LuxError lux_event_wait(LuxEvent* event, LuxStream* stream) {
    (void)event; (void)stream;
    return LUX_OK;
}

float lux_event_elapsed(LuxEvent* start, LuxEvent* end) {
    (void)start; (void)end;
    return 0.0f;
}

// NTT Operations
LuxError lux_ntt_forward(LuxGPU* gpu, uint64_t* data, size_t n, uint64_t modulus) {
    if (!gpu || !gpu->vtbl || !data) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_ntt_forward) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_ntt_forward(gpu->ctx, data, n, modulus));
}

LuxError lux_ntt_inverse(LuxGPU* gpu, uint64_t* data, size_t n, uint64_t modulus) {
    if (!gpu || !gpu->vtbl || !data) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_ntt_inverse) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_ntt_inverse(gpu->ctx, data, n, modulus));
}

LuxError lux_ntt_batch(LuxGPU* gpu, uint64_t** polys, size_t count, size_t n, uint64_t modulus) {
    if (!gpu || !gpu->vtbl || !polys) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_ntt_forward) return LUX_ERROR_NOT_SUPPORTED;

    // Process each polynomial in sequence
    for (size_t i = 0; i < count; i++) {
        LuxBackendError err = gpu->vtbl->op_ntt_forward(gpu->ctx, polys[i], n, modulus);
        if (err != LUX_BACKEND_OK) return static_cast<LuxError>(err);
    }
    return LUX_OK;
}

// =============================================================================
// Polynomial Arithmetic
// =============================================================================

LuxError lux_poly_mul(LuxGPU* gpu, const uint64_t* a, const uint64_t* b, uint64_t* result, size_t n, uint64_t modulus) {
    if (!gpu || !gpu->vtbl || !a || !b || !result) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_poly_mul) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_poly_mul(gpu->ctx, a, b, result, n, modulus));
}

// =============================================================================
// TFHE Operations
// =============================================================================

LuxError lux_tfhe_bootstrap(LuxGPU* gpu,
                            const uint64_t* lwe_in, uint64_t* lwe_out,
                            const uint64_t* bsk, const uint64_t* test_poly,
                            uint32_t n_lwe, uint32_t N, uint32_t k, uint32_t l, uint64_t q) {
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!lwe_in || !lwe_out || !bsk || !test_poly) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_tfhe_bootstrap) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_tfhe_bootstrap(gpu->ctx, lwe_in, lwe_out, bsk, test_poly, n_lwe, N, k, l, q));
}

LuxError lux_tfhe_keyswitch(LuxGPU* gpu,
                            const uint64_t* lwe_in, uint64_t* lwe_out,
                            const uint64_t* ksk,
                            uint32_t n_in, uint32_t n_out, uint32_t l, uint32_t base_log, uint64_t q) {
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!lwe_in || !lwe_out || !ksk) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_tfhe_keyswitch) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_tfhe_keyswitch(gpu->ctx, lwe_in, lwe_out, ksk, n_in, n_out, l, base_log, q));
}

LuxError lux_blind_rotate(LuxGPU* gpu,
                          uint64_t* acc, const uint64_t* bsk, const uint64_t* lwe_a,
                          uint32_t n_lwe, uint32_t N, uint32_t k, uint32_t l, uint64_t q) {
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!acc || !bsk || !lwe_a) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_blind_rotate) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_blind_rotate(gpu->ctx, acc, bsk, lwe_a, n_lwe, N, k, l, q));
}

// =============================================================================
// Crypto: Hash Functions
// =============================================================================

LuxError lux_poseidon2_hash(LuxGPU* gpu, const uint64_t* inputs, uint64_t* outputs, size_t rate, size_t num_hashes) {
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!inputs || !outputs) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_poseidon2_hash) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_poseidon2_hash(gpu->ctx, inputs, outputs, rate, num_hashes));
}

LuxError lux_blake3_hash(LuxGPU* gpu, const uint8_t* inputs, uint8_t* outputs, const size_t* input_lens, size_t num_hashes) {
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!inputs || !outputs || !input_lens) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_blake3_hash) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_blake3_hash(gpu->ctx, inputs, outputs, input_lens, num_hashes));
}

// =============================================================================
// Crypto: MSM
// =============================================================================

LuxError lux_msm(LuxGPU* gpu, const void* scalars, const void* points, void* result, size_t count, LuxCurve curve) {
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!scalars || !points || !result) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_msm) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_msm(gpu->ctx, scalars, points, result, count, static_cast<int>(curve)));
}

// =============================================================================
// Crypto: BLS12-381 Curve
// =============================================================================

LuxError lux_bls12_381_add(LuxGPU* gpu, const void* a, const void* b, void* out, size_t count, bool is_g2) {
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!a || !b || !out) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_bls12_381_add) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_bls12_381_add(gpu->ctx, a, b, out, count, is_g2));
}

LuxError lux_bls12_381_mul(LuxGPU* gpu, const void* points, const void* scalars, void* out, size_t count, bool is_g2) {
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!points || !scalars || !out) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_bls12_381_mul) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_bls12_381_mul(gpu->ctx, points, scalars, out, count, is_g2));
}

LuxError lux_bls12_381_pairing(LuxGPU* gpu, const void* g1_points, const void* g2_points, void* out, size_t count) {
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!g1_points || !g2_points || !out) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_bls12_381_pairing) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_bls12_381_pairing(gpu->ctx, g1_points, g2_points, out, count));
}

// =============================================================================
// Crypto: BN254 Curve
// =============================================================================

LuxError lux_bn254_add(LuxGPU* gpu, const void* a, const void* b, void* out, size_t count, bool is_g2) {
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!a || !b || !out) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_bn254_add) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_bn254_add(gpu->ctx, a, b, out, count, is_g2));
}

LuxError lux_bn254_mul(LuxGPU* gpu, const void* points, const void* scalars, void* out, size_t count, bool is_g2) {
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!points || !scalars || !out) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_bn254_mul) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_bn254_mul(gpu->ctx, points, scalars, out, count, is_g2));
}

// =============================================================================
// Crypto: KZG Polynomial Commitments
// =============================================================================

LuxError lux_kzg_commit(LuxGPU* gpu, const void* coeffs, const void* srs, void* commitment, size_t degree, LuxCurve curve) {
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!coeffs || !srs || !commitment) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_kzg_commit) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_kzg_commit(gpu->ctx, coeffs, srs, commitment, degree, static_cast<int>(curve)));
}

LuxError lux_kzg_open(LuxGPU* gpu, const void* coeffs, const void* srs, const void* point, void* proof, size_t degree, LuxCurve curve) {
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!coeffs || !srs || !point || !proof) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_kzg_open) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_kzg_open(gpu->ctx, coeffs, srs, point, proof, degree, static_cast<int>(curve)));
}

LuxError lux_kzg_verify(LuxGPU* gpu, const void* commitment, const void* proof, const void* point, const void* value, const void* srs_g2, bool* result, LuxCurve curve) {
    if (!gpu || !gpu->vtbl) return LUX_ERROR_INVALID_ARGUMENT;
    if (!commitment || !proof || !point || !value || !srs_g2 || !result) return LUX_ERROR_INVALID_ARGUMENT;
    if (!gpu->vtbl->op_kzg_verify) return LUX_ERROR_NOT_SUPPORTED;
    return static_cast<LuxError>(gpu->vtbl->op_kzg_verify(gpu->ctx, commitment, proof, point, value, srs_g2, result, static_cast<int>(curve)));
}

// =============================================================================
// Stub implementations for high-level BLS functions (require full crypto lib)
// =============================================================================

LuxError lux_bls_verify(LuxGPU* gpu,
                        const uint8_t* sig, size_t sig_len,
                        const uint8_t* msg, size_t msg_len,
                        const uint8_t* pubkey, size_t pubkey_len,
                        bool* result) {
    (void)gpu; (void)sig; (void)sig_len; (void)msg; (void)msg_len;
    (void)pubkey; (void)pubkey_len; (void)result;
    return LUX_ERROR_NOT_SUPPORTED;
}

LuxError lux_bls_verify_batch(LuxGPU* gpu,
                              const uint8_t* const* sigs, const size_t* sig_lens,
                              const uint8_t* const* msgs, const size_t* msg_lens,
                              const uint8_t* const* pubkeys, const size_t* pubkey_lens,
                              int count, bool* results) {
    (void)gpu; (void)sigs; (void)sig_lens; (void)msgs; (void)msg_lens;
    (void)pubkeys; (void)pubkey_lens; (void)count; (void)results;
    return LUX_ERROR_NOT_SUPPORTED;
}

LuxError lux_bls_aggregate(LuxGPU* gpu,
                           const uint8_t* const* sigs, const size_t* sig_lens,
                           int count, uint8_t* out, size_t* out_len) {
    (void)gpu; (void)sigs; (void)sig_lens; (void)count; (void)out; (void)out_len;
    return LUX_ERROR_NOT_SUPPORTED;
}

} // extern "C"
