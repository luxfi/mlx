// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Ternary CUDA Kernels
// Ternary operations (where/select, clamp, etc.)
// Supports float, half, int32, int64, and boolean types with broadcasting

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// ============================================================================
// Configuration
// ============================================================================

#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4

// ============================================================================
// Where/Select Kernels - Float
// ============================================================================

// where(condition, x, y): Select x where condition is true, y otherwise
extern "C" __global__
void where_float_kernel(
    float* __restrict__ output,
    const bool* __restrict__ condition,
    const float* __restrict__ x,
    const float* __restrict__ y,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            output[elem_idx] = condition[elem_idx] ? x[elem_idx] : y[elem_idx];
        }
    }
}

// where with scalar x
extern "C" __global__
void where_scalar_x_float_kernel(
    float* __restrict__ output,
    const bool* __restrict__ condition,
    float x,
    const float* __restrict__ y,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            output[elem_idx] = condition[elem_idx] ? x : y[elem_idx];
        }
    }
}

// where with scalar y
extern "C" __global__
void where_scalar_y_float_kernel(
    float* __restrict__ output,
    const bool* __restrict__ condition,
    const float* __restrict__ x,
    float y,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            output[elem_idx] = condition[elem_idx] ? x[elem_idx] : y;
        }
    }
}

// where with both scalars
extern "C" __global__
void where_scalar_xy_float_kernel(
    float* __restrict__ output,
    const bool* __restrict__ condition,
    float x,
    float y,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            output[elem_idx] = condition[elem_idx] ? x : y;
        }
    }
}

// ============================================================================
// Where/Select Kernels - Half
// ============================================================================

extern "C" __global__
void where_half_kernel(
    __half* __restrict__ output,
    const bool* __restrict__ condition,
    const __half* __restrict__ x,
    const __half* __restrict__ y,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            output[elem_idx] = condition[elem_idx] ? x[elem_idx] : y[elem_idx];
        }
    }
}

// ============================================================================
// Where/Select Kernels - Int32
// ============================================================================

extern "C" __global__
void where_int32_kernel(
    int32_t* __restrict__ output,
    const bool* __restrict__ condition,
    const int32_t* __restrict__ x,
    const int32_t* __restrict__ y,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            output[elem_idx] = condition[elem_idx] ? x[elem_idx] : y[elem_idx];
        }
    }
}

extern "C" __global__
void where_scalar_int32_kernel(
    int32_t* __restrict__ output,
    const bool* __restrict__ condition,
    int32_t x,
    int32_t y,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            output[elem_idx] = condition[elem_idx] ? x : y;
        }
    }
}

// ============================================================================
// Clamp Kernels - Float
// ============================================================================

// clamp(x, min, max): Constrain x to [min, max]
extern "C" __global__
void clamp_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ min_vals,
    const float* __restrict__ max_vals,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float val = input[elem_idx];
            float min_v = min_vals[elem_idx];
            float max_v = max_vals[elem_idx];
            output[elem_idx] = fminf(fmaxf(val, min_v), max_v);
        }
    }
}

// clamp with scalar bounds
extern "C" __global__
void clamp_scalar_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    float min_val,
    float max_val,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float val = input[elem_idx];
            output[elem_idx] = fminf(fmaxf(val, min_val), max_val);
        }
    }
}

// clamp_min only
extern "C" __global__
void clamp_min_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    float min_val,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            output[elem_idx] = fmaxf(input[elem_idx], min_val);
        }
    }
}

// clamp_max only
extern "C" __global__
void clamp_max_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    float max_val,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            output[elem_idx] = fminf(input[elem_idx], max_val);
        }
    }
}

// ============================================================================
// Clamp Kernels - Half
// ============================================================================

extern "C" __global__
void clamp_scalar_half_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    float min_val,
    float max_val,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float val = __half2float(input[elem_idx]);
            output[elem_idx] = __float2half(fminf(fmaxf(val, min_val), max_val));
        }
    }
}

// ============================================================================
// Clamp Kernels - Int32
// ============================================================================

extern "C" __global__
void clamp_scalar_int32_kernel(
    int32_t* __restrict__ output,
    const int32_t* __restrict__ input,
    int32_t min_val,
    int32_t max_val,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            int32_t val = input[elem_idx];
            output[elem_idx] = min(max(val, min_val), max_val);
        }
    }
}

// ============================================================================
// Lerp (Linear Interpolation) Kernels
// ============================================================================

// lerp(a, b, t) = a + t * (b - a) = (1 - t) * a + t * b
extern "C" __global__
void lerp_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ t,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float va = a[elem_idx];
            float vb = b[elem_idx];
            float vt = t[elem_idx];
            output[elem_idx] = fmaf(vt, vb - va, va);  // a + t * (b - a)
        }
    }
}

// lerp with scalar t
extern "C" __global__
void lerp_scalar_t_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ a,
    const float* __restrict__ b,
    float t,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float va = a[elem_idx];
            float vb = b[elem_idx];
            output[elem_idx] = fmaf(t, vb - va, va);
        }
    }
}

// ============================================================================
// Addcmul/Addcdiv (Fused Multiply-Add variants)
// ============================================================================

// addcmul: out = a + value * b * c
extern "C" __global__
void addcmul_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    float value,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            output[elem_idx] = a[elem_idx] + value * b[elem_idx] * c[elem_idx];
        }
    }
}

// addcdiv: out = a + value * b / c
extern "C" __global__
void addcdiv_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    float value,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float vc = c[elem_idx];
            if (vc != 0.0f) {
                output[elem_idx] = a[elem_idx] + value * b[elem_idx] / vc;
            } else {
                output[elem_idx] = a[elem_idx];  // Handle division by zero
            }
        }
    }
}

// ============================================================================
// Broadcasting Where Kernels
// ============================================================================

// Where with broadcasting (general N-D)
extern "C" __global__
void where_broadcast_float_kernel(
    float* __restrict__ output,
    const bool* __restrict__ condition,
    const float* __restrict__ x,
    const float* __restrict__ y,
    const int64_t* __restrict__ cond_strides,
    const int64_t* __restrict__ x_strides,
    const int64_t* __restrict__ y_strides,
    const uint64_t* __restrict__ shape,
    uint32_t ndim,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Convert linear index to multi-dimensional indices
    uint64_t remaining = idx;
    int64_t cond_offset = 0;
    int64_t x_offset = 0;
    int64_t y_offset = 0;

    for (int d = ndim - 1; d >= 0; d--) {
        uint64_t dim_idx = remaining % shape[d];
        remaining /= shape[d];
        cond_offset += dim_idx * cond_strides[d];
        x_offset += dim_idx * x_strides[d];
        y_offset += dim_idx * y_strides[d];
    }

    output[idx] = condition[cond_offset] ? x[x_offset] : y[y_offset];
}

// ============================================================================
// Nan-handling Select
// ============================================================================

// nanwhere: Like where but handles NaN
extern "C" __global__
void nan_to_num_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    float nan_val,
    float posinf_val,
    float neginf_val,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float val = input[elem_idx];
            if (isnan(val)) {
                output[elem_idx] = nan_val;
            } else if (isinf(val)) {
                output[elem_idx] = (val > 0) ? posinf_val : neginf_val;
            } else {
                output[elem_idx] = val;
            }
        }
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_where_float(
    void* output,
    const void* condition,
    const void* x,
    const void* y,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    where_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)output, (const bool*)condition, (const float*)x, (const float*)y, n
    );

    return cudaGetLastError();
}

int lux_cuda_where_scalar_float(
    void* output,
    const void* condition,
    float x,
    float y,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    where_scalar_xy_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)output, (const bool*)condition, x, y, n
    );

    return cudaGetLastError();
}

int lux_cuda_where_half(
    void* output,
    const void* condition,
    const void* x,
    const void* y,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    where_half_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (__half*)output, (const bool*)condition, (const __half*)x, (const __half*)y, n
    );

    return cudaGetLastError();
}

int lux_cuda_where_int32(
    void* output,
    const void* condition,
    const void* x,
    const void* y,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    where_int32_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (int32_t*)output, (const bool*)condition, (const int32_t*)x, (const int32_t*)y, n
    );

    return cudaGetLastError();
}

int lux_cuda_clamp_float(
    void* output,
    const void* input,
    float min_val,
    float max_val,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    clamp_scalar_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)output, (const float*)input, min_val, max_val, n
    );

    return cudaGetLastError();
}

int lux_cuda_clamp_half(
    void* output,
    const void* input,
    float min_val,
    float max_val,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    clamp_scalar_half_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (__half*)output, (const __half*)input, min_val, max_val, n
    );

    return cudaGetLastError();
}

int lux_cuda_clamp_int32(
    void* output,
    const void* input,
    int32_t min_val,
    int32_t max_val,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    clamp_scalar_int32_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (int32_t*)output, (const int32_t*)input, min_val, max_val, n
    );

    return cudaGetLastError();
}

int lux_cuda_lerp_float(
    void* output,
    const void* a,
    const void* b,
    float t,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    lerp_scalar_t_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)output, (const float*)a, (const float*)b, t, n
    );

    return cudaGetLastError();
}

int lux_cuda_addcmul_float(
    void* output,
    const void* a,
    const void* b,
    const void* c,
    float value,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    addcmul_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)output, (const float*)a, (const float*)b, (const float*)c, value, n
    );

    return cudaGetLastError();
}

int lux_cuda_addcdiv_float(
    void* output,
    const void* a,
    const void* b,
    const void* c,
    float value,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    addcdiv_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)output, (const float*)a, (const float*)b, (const float*)c, value, n
    );

    return cudaGetLastError();
}

int lux_cuda_nan_to_num_float(
    void* output,
    const void* input,
    float nan_val,
    float posinf_val,
    float neginf_val,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    nan_to_num_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)output, (const float*)input, nan_val, posinf_val, neginf_val, n
    );

    return cudaGetLastError();
}

}  // extern "C"
