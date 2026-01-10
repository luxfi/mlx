// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// CUDA Tensor Operations API
// Declares C API for calling optimized CUDA kernels from the backend plugin

#ifndef LUX_GPU_CUDA_TENSOR_OPS_H
#define LUX_GPU_CUDA_TENSOR_OPS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declare CUDA stream type to avoid header dependency
typedef struct CUstream_st* cudaStream_t;

// =============================================================================
// Elementwise Binary Operations
// =============================================================================

int lux_cuda_add_f32(
    void* output,
    const void* a,
    const void* b,
    size_t n,
    cudaStream_t stream
);

int lux_cuda_sub_f32(
    void* output,
    const void* a,
    const void* b,
    size_t n,
    cudaStream_t stream
);

int lux_cuda_mul_f32(
    void* output,
    const void* a,
    const void* b,
    size_t n,
    cudaStream_t stream
);

int lux_cuda_div_f32(
    void* output,
    const void* a,
    const void* b,
    size_t n,
    cudaStream_t stream
);

// =============================================================================
// Matrix Operations
// =============================================================================

// Tiled GEMM: C = A @ B
// A: [M x K], B: [K x N], C: [M x N]
int lux_cuda_matmul_f32(
    void* c,
    const void* a,
    const void* b,
    int M,
    int K,
    int N,
    cudaStream_t stream
);

// Transpose: B = A^T
// A: [rows x cols], B: [cols x rows]
int lux_cuda_transpose_f32(
    void* output,
    const void* input,
    int rows,
    int cols,
    cudaStream_t stream
);

// =============================================================================
// Reduction Operations
// =============================================================================

// Full array reductions (n elements -> 1 scalar)
int lux_cuda_reduce_sum_float(
    void* output,
    const void* input,
    uint64_t n,
    cudaStream_t stream
);

int lux_cuda_reduce_max_float(
    void* output,
    const void* input,
    uint64_t n,
    cudaStream_t stream
);

int lux_cuda_reduce_min_float(
    void* output,
    const void* input,
    uint64_t n,
    cudaStream_t stream
);

int lux_cuda_reduce_mean_float(
    void* output,
    const void* input,
    uint64_t n,
    cudaStream_t stream
);

// Axis reductions (outer_size x inner_size -> outer_size)
int lux_cuda_reduce_sum_axis_float(
    void* output,
    const void* input,
    uint64_t outer_size,
    uint64_t inner_size,
    cudaStream_t stream
);

int lux_cuda_reduce_max_axis_float(
    void* output,
    const void* input,
    uint64_t outer_size,
    uint64_t inner_size,
    cudaStream_t stream
);

// =============================================================================
// Softmax Operations
// =============================================================================

int lux_cuda_softmax_f32(
    void* output,
    const void* input,
    uint32_t batch_size,
    uint32_t dim,
    cudaStream_t stream
);

int lux_cuda_log_softmax_f32(
    void* output,
    const void* input,
    uint32_t batch_size,
    uint32_t dim,
    cudaStream_t stream
);

// =============================================================================
// Unary Operations
// =============================================================================

int lux_cuda_exp_f32(
    void* output,
    const void* input,
    size_t n,
    cudaStream_t stream
);

int lux_cuda_log_f32(
    void* output,
    const void* input,
    size_t n,
    cudaStream_t stream
);

int lux_cuda_sqrt_f32(
    void* output,
    const void* input,
    size_t n,
    cudaStream_t stream
);

int lux_cuda_neg_f32(
    void* output,
    const void* input,
    size_t n,
    cudaStream_t stream
);

int lux_cuda_abs_f32(
    void* output,
    const void* input,
    size_t n,
    cudaStream_t stream
);

int lux_cuda_tanh_f32(
    void* output,
    const void* input,
    size_t n,
    cudaStream_t stream
);

int lux_cuda_sigmoid_f32(
    void* output,
    const void* input,
    size_t n,
    cudaStream_t stream
);

int lux_cuda_relu_f32(
    void* output,
    const void* input,
    size_t n,
    cudaStream_t stream
);

int lux_cuda_gelu_f32(
    void* output,
    const void* input,
    size_t n,
    cudaStream_t stream
);

// =============================================================================
// Copy Operations
// =============================================================================

int lux_cuda_copy_f32(
    void* dst,
    const void* src,
    size_t n,
    cudaStream_t stream
);

// =============================================================================
// Normalization Operations
// =============================================================================

int lux_cuda_layer_norm_f32(
    void* output,
    const void* input,
    const void* gamma,
    const void* beta,
    uint32_t batch_size,
    uint32_t dim,
    float eps,
    cudaStream_t stream
);

int lux_cuda_rms_norm_f32(
    void* output,
    const void* input,
    const void* weight,
    uint32_t batch_size,
    uint32_t dim,
    float eps,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // LUX_GPU_CUDA_TENSOR_OPS_H
