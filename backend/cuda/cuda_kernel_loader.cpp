// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// CUDA Kernel Loader - Load PTX/CUBIN and cache compiled modules

#include "lux/gpu/kernel_loader.h"
#include <unordered_map>
#include <string>
#include <mutex>
#include <vector>
#include <cstring>
#include <fstream>

#ifdef LUX_GPU_BUILD_CUDA

#include <cuda.h>

namespace lux::gpu::cuda {

// =============================================================================
// CUDA Kernel Implementation
// =============================================================================

struct CudaKernel {
    CUmodule module;
    CUfunction function;
    std::string entry_point;
    int max_threads_per_block;
    int shared_size_bytes;
    int const_size_bytes;
    int local_size_bytes;
    int num_regs;
};

// =============================================================================
// CUDA Kernel Cache
// =============================================================================

class CudaKernelCache {
public:
    static CudaKernelCache& instance() {
        static CudaKernelCache cache;
        return cache;
    }

    // Load from PTX source
    CudaKernel* load_ptx(
        CUcontext context,
        const char* ptx_source,
        size_t ptx_len,
        const char* entry_point,
        const char* compile_opts
    ) {
        CUmodule module;
        CUresult result;

        // JIT compile options
        std::vector<CUjit_option> options;
        std::vector<void*> option_values;

        // Enable fast math by default
        options.push_back(CU_JIT_FAST_COMPILE);
        option_values.push_back(reinterpret_cast<void*>(1));

        // Log buffer for errors
        char log_buffer[4096] = {0};
        options.push_back(CU_JIT_ERROR_LOG_BUFFER);
        option_values.push_back(log_buffer);
        options.push_back(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES);
        option_values.push_back(reinterpret_cast<void*>(sizeof(log_buffer)));

        // Load module from PTX
        result = cuModuleLoadDataEx(
            &module,
            ptx_source,
            static_cast<unsigned int>(options.size()),
            options.data(),
            option_values.data()
        );

        if (result != CUDA_SUCCESS) {
            const char* err_str;
            cuGetErrorString(result, &err_str);
            fprintf(stderr, "CudaKernelCache: PTX load failed: %s\n", err_str);
            if (log_buffer[0]) {
                fprintf(stderr, "JIT log: %s\n", log_buffer);
            }
            return nullptr;
        }

        // Get function
        CUfunction function;
        result = cuModuleGetFunction(&function, module, entry_point);
        if (result != CUDA_SUCCESS) {
            fprintf(stderr, "CudaKernelCache: Function '%s' not found\n", entry_point);
            cuModuleUnload(module);
            return nullptr;
        }

        // Get function attributes
        int max_threads = 0;
        int shared_size = 0;
        int const_size = 0;
        int local_size = 0;
        int num_regs = 0;

        cuFuncGetAttribute(&max_threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function);
        cuFuncGetAttribute(&shared_size, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function);
        cuFuncGetAttribute(&const_size, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, function);
        cuFuncGetAttribute(&local_size, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function);
        cuFuncGetAttribute(&num_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, function);

        auto kernel = new CudaKernel();
        kernel->module = module;
        kernel->function = function;
        kernel->entry_point = entry_point;
        kernel->max_threads_per_block = max_threads;
        kernel->shared_size_bytes = shared_size;
        kernel->const_size_bytes = const_size;
        kernel->local_size_bytes = local_size;
        kernel->num_regs = num_regs;

        return kernel;
    }

    // Load from CUBIN binary
    CudaKernel* load_cubin(
        CUcontext context,
        const void* cubin,
        size_t cubin_len,
        const char* entry_point
    ) {
        CUmodule module;
        CUresult result = cuModuleLoadData(&module, cubin);

        if (result != CUDA_SUCCESS) {
            const char* err_str;
            cuGetErrorString(result, &err_str);
            fprintf(stderr, "CudaKernelCache: CUBIN load failed: %s\n", err_str);
            return nullptr;
        }

        CUfunction function;
        result = cuModuleGetFunction(&function, module, entry_point);
        if (result != CUDA_SUCCESS) {
            fprintf(stderr, "CudaKernelCache: Function '%s' not found\n", entry_point);
            cuModuleUnload(module);
            return nullptr;
        }

        int max_threads = 0;
        cuFuncGetAttribute(&max_threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function);

        auto kernel = new CudaKernel();
        kernel->module = module;
        kernel->function = function;
        kernel->entry_point = entry_point;
        kernel->max_threads_per_block = max_threads;

        return kernel;
    }

    // Load from file (PTX or CUBIN)
    CudaKernel* load_file(
        CUcontext context,
        const char* path,
        const char* entry_point
    ) {
        // Read file
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) {
            fprintf(stderr, "CudaKernelCache: Failed to open file: %s\n", path);
            return nullptr;
        }

        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> content(size + 1);
        if (!file.read(content.data(), size)) {
            fprintf(stderr, "CudaKernelCache: Failed to read file: %s\n", path);
            return nullptr;
        }
        content[size] = '\0';

        // Determine type by extension
        std::string path_str(path);
        if (path_str.size() > 4 && path_str.substr(path_str.size() - 4) == ".ptx") {
            return load_ptx(context, content.data(), size, entry_point, nullptr);
        } else {
            return load_cubin(context, content.data(), size, entry_point);
        }
    }

    void destroy(CudaKernel* kernel) {
        if (kernel) {
            if (kernel->module) {
                cuModuleUnload(kernel->module);
            }
            delete kernel;
        }
    }

private:
    CudaKernelCache() = default;
    std::mutex mutex_;
};

} // namespace lux::gpu::cuda

// =============================================================================
// C API Implementation for CUDA
// =============================================================================

extern "C" {

using namespace lux::gpu::cuda;

LuxKernel* lux_cuda_kernel_load_ptx(
    void* context,
    const char* ptx_source,
    size_t ptx_len,
    const char* entry_point,
    const char* compile_opts
) {
    if (!context || !ptx_source || !entry_point) return nullptr;

    CUcontext cu_context = reinterpret_cast<CUcontext>(context);
    auto kernel = CudaKernelCache::instance().load_ptx(
        cu_context, ptx_source, ptx_len, entry_point, compile_opts
    );
    return reinterpret_cast<LuxKernel*>(kernel);
}

LuxKernel* lux_cuda_kernel_load_cubin(
    void* context,
    const void* cubin,
    size_t cubin_len,
    const char* entry_point
) {
    if (!context || !cubin || !entry_point) return nullptr;

    CUcontext cu_context = reinterpret_cast<CUcontext>(context);
    auto kernel = CudaKernelCache::instance().load_cubin(
        cu_context, cubin, cubin_len, entry_point
    );
    return reinterpret_cast<LuxKernel*>(kernel);
}

LuxKernel* lux_cuda_kernel_load_file(
    void* context,
    const char* path,
    const char* entry_point
) {
    if (!context || !path || !entry_point) return nullptr;

    CUcontext cu_context = reinterpret_cast<CUcontext>(context);
    auto kernel = CudaKernelCache::instance().load_file(cu_context, path, entry_point);
    return reinterpret_cast<LuxKernel*>(kernel);
}

void lux_cuda_kernel_destroy(LuxKernel* kernel) {
    CudaKernelCache::instance().destroy(reinterpret_cast<CudaKernel*>(kernel));
}

const char* lux_cuda_kernel_entry_point(LuxKernel* kernel) {
    auto k = reinterpret_cast<CudaKernel*>(kernel);
    return k ? k->entry_point.c_str() : nullptr;
}

// Get CUfunction for dispatch
void* lux_cuda_kernel_function(LuxKernel* kernel) {
    auto k = reinterpret_cast<CudaKernel*>(kernel);
    return k ? reinterpret_cast<void*>(k->function) : nullptr;
}

// Get max threads per block
int lux_cuda_kernel_max_threads(LuxKernel* kernel) {
    auto k = reinterpret_cast<CudaKernel*>(kernel);
    return k ? k->max_threads_per_block : 0;
}

// Get register count
int lux_cuda_kernel_num_regs(LuxKernel* kernel) {
    auto k = reinterpret_cast<CudaKernel*>(kernel);
    return k ? k->num_regs : 0;
}

// Get shared memory size
int lux_cuda_kernel_shared_size(LuxKernel* kernel) {
    auto k = reinterpret_cast<CudaKernel*>(kernel);
    return k ? k->shared_size_bytes : 0;
}

} // extern "C"

// =============================================================================
// CUDA Kernel Manager - High-level interface
// =============================================================================

namespace lux::gpu::cuda {

class CudaKernelManager {
public:
    CudaKernelManager(CUcontext context) : context_(context) {
        cache_ = lux_kernel_cache_create();
    }

    ~CudaKernelManager() {
        // Clear cache (kernels destroyed individually)
        lux_kernel_cache_clear(cache_);
        lux_kernel_cache_destroy(cache_);
    }

    // Get or load kernel by name
    CudaKernel* get_kernel(const char* name, uint32_t dtype = 0, uint32_t size_hint = 0) {
        LuxKernelVariant variant = { name, dtype, size_hint, 0 };

        // Check cache
        LuxKernel* cached = lux_kernel_cache_get(cache_, &variant);
        if (cached) {
            return reinterpret_cast<CudaKernel*>(cached);
        }

        // Find in registry
        const LuxKernelRegistry* registry = lux_kernel_registry_get("cuda");
        if (!registry) return nullptr;

        const LuxEmbeddedKernel* embedded = lux_kernel_registry_find(registry, name);
        if (!embedded) return nullptr;

        // Load kernel
        CudaKernel* kernel = nullptr;
        if (embedded->binary && embedded->binary_len > 0) {
            // Pre-compiled CUBIN
            kernel = CudaKernelCache::instance().load_cubin(
                context_, embedded->binary, embedded->binary_len, embedded->entry_point
            );
        } else if (embedded->source && embedded->source_len > 0) {
            // PTX source (JIT compiled)
            kernel = CudaKernelCache::instance().load_ptx(
                context_, embedded->source, embedded->source_len, embedded->entry_point, nullptr
            );
        }

        if (kernel) {
            lux_kernel_cache_put(cache_, &variant, reinterpret_cast<LuxKernel*>(kernel));
        }

        return kernel;
    }

    // Preload all embedded kernels
    void preload_all() {
        const LuxKernelRegistry* registry = lux_kernel_registry_get("cuda");
        if (!registry) return;

        for (size_t i = 0; i < registry->count; ++i) {
            get_kernel(registry->kernels[i].name);
        }
    }

private:
    CUcontext context_;
    LuxKernelCache* cache_;
};

} // namespace lux::gpu::cuda

#else // !LUX_GPU_BUILD_CUDA

// Stub implementations when CUDA is not available
extern "C" {

LuxKernel* lux_cuda_kernel_load_ptx(void*, const char*, size_t, const char*, const char*) {
    return nullptr;
}

LuxKernel* lux_cuda_kernel_load_cubin(void*, const void*, size_t, const char*) {
    return nullptr;
}

LuxKernel* lux_cuda_kernel_load_file(void*, const char*, const char*) {
    return nullptr;
}

void lux_cuda_kernel_destroy(LuxKernel*) {}

const char* lux_cuda_kernel_entry_point(LuxKernel*) {
    return nullptr;
}

void* lux_cuda_kernel_function(LuxKernel*) {
    return nullptr;
}

int lux_cuda_kernel_max_threads(LuxKernel*) {
    return 0;
}

int lux_cuda_kernel_num_regs(LuxKernel*) {
    return 0;
}

int lux_cuda_kernel_shared_size(LuxKernel*) {
    return 0;
}

} // extern "C"

#endif // LUX_GPU_BUILD_CUDA
