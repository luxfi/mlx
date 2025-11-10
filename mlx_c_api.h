// MLX C API Header for CGO bindings
// This provides a C interface to the C++ MLX library

#ifndef MLX_C_API_H
#define MLX_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

// Check for backend support
bool mlx_has_metal();
bool mlx_has_cuda();

// Device information
char* mlx_get_metal_device_name();
char* mlx_get_cuda_device_name();
size_t mlx_get_metal_memory();
size_t mlx_get_cuda_memory();
size_t mlx_get_system_memory();

// Array operations
void* mlx_zeros(int* shape, int ndim, int dtype);
void* mlx_ones(int* shape, int ndim, int dtype);
void* mlx_random(int* shape, int ndim, int dtype);
void* mlx_arange(double start, double stop, double step);

// Binary operations
void* mlx_add(void* a, void* b);
void* mlx_multiply(void* a, void* b);
void* mlx_matmul(void* a, void* b);

// Reduction operations
void* mlx_sum(void* array, int* axis, int naxis);
void* mlx_mean(void* array, int* axis, int naxis);

// Evaluation and synchronization
void mlx_eval(void* arrays[], int count);
void mlx_synchronize();

// Stream management
void* mlx_new_stream();
void mlx_free_stream(void* stream);

// Memory management
void mlx_free_array(void* array);

// Array creation from data
void* mlx_from_slice(float* data, int data_len, int* shape, int ndim, int dtype);

// Element-wise maximum
void* mlx_maximum(void* a, void* b);

#ifdef __cplusplus
}
#endif

#endif // MLX_C_API_H

