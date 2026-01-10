// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Arange CUDA Kernels
// Generate sequential values [0, 1, 2, ..., n-1] with optional start, step, and stop
// Supports float, half, int32, and int64 types

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// ============================================================================
// Configuration
// ============================================================================

#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4

// ============================================================================
// Arange Kernels - Float32
// ============================================================================

// Generate [start, start+step, start+2*step, ...]
extern "C" __global__
void arange_float_kernel(
    float* __restrict__ out,
    float start,
    float step,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    // Process multiple elements per thread for better memory coalescing
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx * ELEMENTS_PER_THREAD + i;
        if (elem_idx < n) {
            out[elem_idx] = start + (float)elem_idx * step;
        }
    }
}

// Simple arange: [0, 1, 2, ..., n-1]
extern "C" __global__
void arange_simple_float_kernel(
    float* __restrict__ out,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx * ELEMENTS_PER_THREAD + i;
        if (elem_idx < n) {
            out[elem_idx] = (float)elem_idx;
        }
    }
}

// ============================================================================
// Arange Kernels - Float16 (Half)
// ============================================================================

extern "C" __global__
void arange_half_kernel(
    __half* __restrict__ out,
    float start,
    float step,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx * ELEMENTS_PER_THREAD + i;
        if (elem_idx < n) {
            out[elem_idx] = __float2half(start + (float)elem_idx * step);
        }
    }
}

extern "C" __global__
void arange_simple_half_kernel(
    __half* __restrict__ out,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx * ELEMENTS_PER_THREAD + i;
        if (elem_idx < n) {
            out[elem_idx] = __float2half((float)elem_idx);
        }
    }
}

// ============================================================================
// Arange Kernels - Int32
// ============================================================================

extern "C" __global__
void arange_int32_kernel(
    int32_t* __restrict__ out,
    int32_t start,
    int32_t step,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx * ELEMENTS_PER_THREAD + i;
        if (elem_idx < n) {
            out[elem_idx] = start + (int32_t)elem_idx * step;
        }
    }
}

extern "C" __global__
void arange_simple_int32_kernel(
    int32_t* __restrict__ out,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx * ELEMENTS_PER_THREAD + i;
        if (elem_idx < n) {
            out[elem_idx] = (int32_t)elem_idx;
        }
    }
}

// ============================================================================
// Arange Kernels - Int64
// ============================================================================

extern "C" __global__
void arange_int64_kernel(
    int64_t* __restrict__ out,
    int64_t start,
    int64_t step,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx * ELEMENTS_PER_THREAD + i;
        if (elem_idx < n) {
            out[elem_idx] = start + (int64_t)elem_idx * step;
        }
    }
}

extern "C" __global__
void arange_simple_int64_kernel(
    int64_t* __restrict__ out,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx * ELEMENTS_PER_THREAD + i;
        if (elem_idx < n) {
            out[elem_idx] = (int64_t)elem_idx;
        }
    }
}

// ============================================================================
// Linspace Kernels - Generate evenly spaced values
// ============================================================================

extern "C" __global__
void linspace_float_kernel(
    float* __restrict__ out,
    float start,
    float stop,
    uint64_t n
) {
    if (n <= 1) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            out[0] = start;
        }
        return;
    }

    float step = (stop - start) / (float)(n - 1);
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx * ELEMENTS_PER_THREAD + i;
        if (elem_idx < n) {
            // Use formula to avoid accumulated error
            out[elem_idx] = start + (float)elem_idx * step;
        }
    }

    // Ensure last element is exactly stop
    if (idx == 0 && threadIdx.x == 0) {
        out[n - 1] = stop;
    }
}

extern "C" __global__
void linspace_half_kernel(
    __half* __restrict__ out,
    float start,
    float stop,
    uint64_t n
) {
    if (n <= 1) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            out[0] = __float2half(start);
        }
        return;
    }

    float step = (stop - start) / (float)(n - 1);
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx * ELEMENTS_PER_THREAD + i;
        if (elem_idx < n) {
            out[elem_idx] = __float2half(start + (float)elem_idx * step);
        }
    }

    if (idx == 0 && threadIdx.x == 0) {
        out[n - 1] = __float2half(stop);
    }
}

// ============================================================================
// Logspace Kernels - Generate logarithmically spaced values
// ============================================================================

extern "C" __global__
void logspace_float_kernel(
    float* __restrict__ out,
    float start,      // log10(first_value)
    float stop,       // log10(last_value)
    float base,       // Base of logarithm (usually 10)
    uint64_t n
) {
    if (n <= 1) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            out[0] = powf(base, start);
        }
        return;
    }

    float step = (stop - start) / (float)(n - 1);
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx * ELEMENTS_PER_THREAD + i;
        if (elem_idx < n) {
            float exp = start + (float)elem_idx * step;
            out[elem_idx] = powf(base, exp);
        }
    }
}

// ============================================================================
// Meshgrid Helper - Generate indices for 2D grid
// ============================================================================

extern "C" __global__
void meshgrid_x_float_kernel(
    float* __restrict__ out,
    float start,
    float step,
    uint32_t nx,
    uint32_t ny
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)nx * ny;

    if (idx < total) {
        uint32_t x = idx % nx;
        out[idx] = start + (float)x * step;
    }
}

extern "C" __global__
void meshgrid_y_float_kernel(
    float* __restrict__ out,
    float start,
    float step,
    uint32_t nx,
    uint32_t ny
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)nx * ny;

    if (idx < total) {
        uint32_t y = idx / nx;
        out[idx] = start + (float)y * step;
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_arange_float(
    void* out,
    float start,
    float step,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    arange_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)out, start, step, n
    );

    return cudaGetLastError();
}

int lux_cuda_arange_simple_float(
    void* out,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    arange_simple_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)out, n
    );

    return cudaGetLastError();
}

int lux_cuda_arange_half(
    void* out,
    float start,
    float step,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    arange_half_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (__half*)out, start, step, n
    );

    return cudaGetLastError();
}

int lux_cuda_arange_int32(
    void* out,
    int32_t start,
    int32_t step,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    arange_int32_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (int32_t*)out, start, step, n
    );

    return cudaGetLastError();
}

int lux_cuda_arange_int64(
    void* out,
    int64_t start,
    int64_t step,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    arange_int64_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (int64_t*)out, start, step, n
    );

    return cudaGetLastError();
}

int lux_cuda_linspace_float(
    void* out,
    float start,
    float stop,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    linspace_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)out, start, stop, n
    );

    return cudaGetLastError();
}

int lux_cuda_linspace_half(
    void* out,
    float start,
    float stop,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    linspace_half_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (__half*)out, start, stop, n
    );

    return cudaGetLastError();
}

int lux_cuda_logspace_float(
    void* out,
    float start,
    float stop,
    float base,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    logspace_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)out, start, stop, base, n
    );

    return cudaGetLastError();
}

int lux_cuda_meshgrid_float(
    void* out_x,
    void* out_y,
    float x_start,
    float x_step,
    float y_start,
    float y_step,
    uint32_t nx,
    uint32_t ny,
    cudaStream_t stream
) {
    uint64_t total = (uint64_t)nx * ny;
    if (total == 0) return 0;

    uint64_t num_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    meshgrid_x_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)out_x, x_start, x_step, nx, ny
    );

    meshgrid_y_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)out_y, y_start, y_step, nx, ny
    );

    return cudaGetLastError();
}

}  // extern "C"
