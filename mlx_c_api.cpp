// MLX C API Implementation
// This provides the C interface to the C++ MLX library

#include "mlx_c_api.h"
#include "mlx/mlx.h"
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <stdexcept>

#ifdef __APPLE__
#include <TargetConditionals.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#if TARGET_OS_MAC && TARGET_CPU_ARM64
#define HAS_METAL 1
#endif
#endif

#ifdef __linux__
#include <unistd.h>
#ifdef __CUDACC__
#define HAS_CUDA 1
#endif
#endif

#ifdef _WIN32
#include <windows.h>
#endif

// Backend detection
extern "C" {

bool mlx_has_metal() {
#ifdef HAS_METAL
    return mlx::core::metal::is_available();
#else
    return false;
#endif
}

bool mlx_has_cuda() {
#ifdef HAS_CUDA
    // CUDA is not yet available in this version
    return false;
#else
    return false;
#endif
}

char* mlx_get_metal_device_name() {
#ifdef HAS_METAL
    if (mlx::core::metal::is_available()) {
        const char* name = "Apple M-series GPU";
        char* result = (char*)malloc(strlen(name) + 1);
        strcpy(result, name);
        return result;
    }
#endif
    const char* name = "No Metal Support";
    char* result = (char*)malloc(strlen(name) + 1);
    strcpy(result, name);
    return result;
}

char* mlx_get_cuda_device_name() {
#ifdef HAS_CUDA
    const char* name = "NVIDIA GPU";
    char* result = (char*)malloc(strlen(name) + 1);
    strcpy(result, name);
    return result;
#endif
    const char* name = "No CUDA Support";
    char* result = (char*)malloc(strlen(name) + 1);
    strcpy(result, name);
    return result;
}

size_t mlx_get_metal_memory() {
#ifdef HAS_METAL
    // On Apple Silicon, unified memory equals system memory
    return mlx_get_system_memory();
#else
    return 0;
#endif
}

size_t mlx_get_cuda_memory() {
#ifdef HAS_CUDA
    // Would query CUDA device for actual memory
    // For now, return a reasonable default
    return 8ULL * 1024 * 1024 * 1024; // 8GB default
#else
    return 0;
#endif
}

size_t mlx_get_system_memory() {
#ifdef __APPLE__
    // Get actual system memory on macOS
    int64_t memsize = 0;
    size_t len = sizeof(memsize);
    if (sysctlbyname("hw.memsize", &memsize, &len, NULL, 0) == 0) {
        return (size_t)memsize;
    }
    return 16ULL * 1024 * 1024 * 1024; // Fallback
#elif defined(__linux__)
    // Get system memory on Linux
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        return (size_t)pages * (size_t)page_size;
    }
    return 16ULL * 1024 * 1024 * 1024; // Fallback
#elif defined(_WIN32)
    // Get system memory on Windows
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return (size_t)status.ullTotalPhys;
    }
    return 16ULL * 1024 * 1024 * 1024; // Fallback
#else
    return 16ULL * 1024 * 1024 * 1024; // Default fallback
#endif
}

} // extern "C"

// Helper: Convert C dtype to MLX dtype (internal, C++ linkage)
static mlx::core::Dtype to_mlx_dtype(int dtype) {
    switch (dtype) {
        case 0: return mlx::core::float32;
        case 1: return mlx::core::float64;
        case 2: return mlx::core::int32;
        case 3: return mlx::core::int64;
        case 4: return mlx::core::bool_;
        default: return mlx::core::float32;
    }
}

extern "C" {

// Array operations using real MLX library
void* mlx_zeros(int* shape, int ndim, int dtype) {
    try {
        mlx::core::Shape shape_vec(shape, shape + ndim);
        mlx::core::Dtype mlx_dtype = to_mlx_dtype(dtype);
        mlx::core::array arr = mlx::core::zeros(shape_vec, mlx_dtype);
        return new mlx::core::array(std::move(arr));
    } catch (const std::exception& e) {
        // Log error if needed
        return nullptr;
    }
}

void* mlx_ones(int* shape, int ndim, int dtype) {
    try {
        mlx::core::Shape shape_vec(shape, shape + ndim);
        mlx::core::Dtype mlx_dtype = to_mlx_dtype(dtype);
        mlx::core::array arr = mlx::core::ones(shape_vec, mlx_dtype);
        return new mlx::core::array(std::move(arr));
    } catch (const std::exception& e) {
        return nullptr;
    }
}

void* mlx_random(int* shape, int ndim, int dtype) {
    try {
        mlx::core::Shape shape_vec(shape, shape + ndim);
        mlx::core::Dtype mlx_dtype = to_mlx_dtype(dtype);
        // Use random::uniform for random values
        mlx::core::array arr = mlx::core::random::uniform(
            0.0f, 1.0f, shape_vec, mlx_dtype);
        return new mlx::core::array(std::move(arr));
    } catch (const std::exception& e) {
        return nullptr;
    }
}

void* mlx_arange(double start, double stop, double step) {
    try {
        mlx::core::array arr = mlx::core::arange(start, stop, step);
        return new mlx::core::array(std::move(arr));
    } catch (const std::exception& e) {
        return nullptr;
    }
}

void* mlx_add(void* a, void* b) {
    try {
        auto* arr_a = static_cast<mlx::core::array*>(a);
        auto* arr_b = static_cast<mlx::core::array*>(b);
        mlx::core::array result = mlx::core::add(*arr_a, *arr_b);
        return new mlx::core::array(std::move(result));
    } catch (const std::exception& e) {
        return nullptr;
    }
}

void* mlx_multiply(void* a, void* b) {
    try {
        auto* arr_a = static_cast<mlx::core::array*>(a);
        auto* arr_b = static_cast<mlx::core::array*>(b);
        mlx::core::array result = mlx::core::multiply(*arr_a, *arr_b);
        return new mlx::core::array(std::move(result));
    } catch (const std::exception& e) {
        return nullptr;
    }
}

void* mlx_matmul(void* a, void* b) {
    try {
        auto* arr_a = static_cast<mlx::core::array*>(a);
        auto* arr_b = static_cast<mlx::core::array*>(b);
        mlx::core::array result = mlx::core::matmul(*arr_a, *arr_b);
        return new mlx::core::array(std::move(result));
    } catch (const std::exception& e) {
        return nullptr;
    }
}

void* mlx_sum(void* array, int* axis, int naxis) {
    try {
        auto* arr = static_cast<mlx::core::array*>(array);
        mlx::core::array result = (naxis == 0) 
            ? mlx::core::sum(*arr)
            : mlx::core::sum(*arr, std::vector<int>(axis, axis + naxis));
        return new mlx::core::array(std::move(result));
    } catch (const std::exception& e) {
        return nullptr;
    }
}

void* mlx_mean(void* array, int* axis, int naxis) {
    try {
        auto* arr = static_cast<mlx::core::array*>(array);
        mlx::core::array result = (naxis == 0)
            ? mlx::core::mean(*arr)
            : mlx::core::mean(*arr, std::vector<int>(axis, axis + naxis));
        return new mlx::core::array(std::move(result));
    } catch (const std::exception& e) {
        return nullptr;
    }
}

void mlx_eval(void* arrays[], int count) {
    try {
        std::vector<mlx::core::array> arr_vec;
        for (int i = 0; i < count; i++) {
            if (arrays[i]) {
                auto* arr = static_cast<mlx::core::array*>(arrays[i]);
                arr_vec.push_back(*arr);
            }
        }
        mlx::core::eval(arr_vec);
    } catch (const std::exception& e) {
        // Log error if needed
    }
}

void mlx_synchronize() {
    try {
        mlx::core::synchronize();
    } catch (const std::exception& e) {
        // Log error if needed
    }
}

void* mlx_new_stream() {
    try {
        mlx::core::Device default_device(mlx::core::Device::gpu, 0);
        mlx::core::Stream stream = mlx::core::new_stream(default_device);
        return new mlx::core::Stream(std::move(stream));
    } catch (const std::exception& e) {
        return nullptr;
    }
}

void mlx_free_stream(void* stream) {
    if (stream) {
        auto* s = static_cast<mlx::core::Stream*>(stream);
        delete s;
    }
}

void mlx_free_array(void* array) {
    if (array) {
        auto* arr = static_cast<mlx::core::array*>(array);
        delete arr;
    }
}

} // extern "C"

// Create array from data slice
void* mlx_from_slice(float* data, int data_len, int* shape, int ndim, int dtype) {
    try {
        mlx::core::Shape shape_vec(shape, shape + ndim);
        mlx::core::Dtype mlx_dtype = to_mlx_dtype(dtype);
        
        // Create array from data
        std::vector<float> data_vec(data, data + data_len);
        mlx::core::array arr(data_vec.data(), shape_vec, mlx_dtype);
        
        return new mlx::core::array(std::move(arr));
    } catch (const std::exception& e) {
        return nullptr;
    }
}

// Element-wise maximum
void* mlx_maximum(void* a, void* b) {
    try {
        auto* arr_a = static_cast<mlx::core::array*>(a);
        auto* arr_b = static_cast<mlx::core::array*>(b);
        mlx::core::array result = mlx::core::maximum(*arr_a, *arr_b);
        return new mlx::core::array(std::move(result));
    } catch (const std::exception& e) {
        return nullptr;
    }
}

