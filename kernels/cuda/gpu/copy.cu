// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Copy CUDA Kernels
// Memory copy operations with striding, broadcasting, and type conversion
// Supports contiguous, strided, and general N-dimensional copies

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// ============================================================================
// Configuration
// ============================================================================

#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4
#define MAX_DIMS 8

// ============================================================================
// Contiguous Copy Kernels
// ============================================================================

// Simple contiguous copy (memcpy-like but kernel-based)
extern "C" __global__
void copy_contiguous_kernel(
    void* __restrict__ dst,
    const void* __restrict__ src,
    uint64_t n_bytes
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;

    // Copy 4 bytes at a time using uint32
    if (idx + 3 < n_bytes) {
        ((uint32_t*)dst)[idx / 4] = ((const uint32_t*)src)[idx / 4];
    } else {
        // Handle tail
        for (uint64_t i = idx; i < n_bytes && i < idx + 4; i++) {
            ((uint8_t*)dst)[i] = ((const uint8_t*)src)[i];
        }
    }
}

// Contiguous copy for float
extern "C" __global__
void copy_float_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            dst[elem_idx] = src[elem_idx];
        }
    }
}

// Contiguous copy for half
extern "C" __global__
void copy_half_kernel(
    __half* __restrict__ dst,
    const __half* __restrict__ src,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            dst[elem_idx] = src[elem_idx];
        }
    }
}

// ============================================================================
// Type Conversion Copies
// ============================================================================

// Float to Half conversion
extern "C" __global__
void copy_float_to_half_kernel(
    __half* __restrict__ dst,
    const float* __restrict__ src,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            dst[elem_idx] = __float2half(src[elem_idx]);
        }
    }
}

// Half to Float conversion
extern "C" __global__
void copy_half_to_float_kernel(
    float* __restrict__ dst,
    const __half* __restrict__ src,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            dst[elem_idx] = __half2float(src[elem_idx]);
        }
    }
}

// Int32 to Float conversion
extern "C" __global__
void copy_int32_to_float_kernel(
    float* __restrict__ dst,
    const int32_t* __restrict__ src,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            dst[elem_idx] = (float)src[elem_idx];
        }
    }
}

// Float to Int32 conversion
extern "C" __global__
void copy_float_to_int32_kernel(
    int32_t* __restrict__ dst,
    const float* __restrict__ src,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            dst[elem_idx] = (int32_t)src[elem_idx];
        }
    }
}

// ============================================================================
// Strided Copy Kernels
// ============================================================================

// 1D strided copy
extern "C" __global__
void copy_strided_1d_float_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int64_t dst_stride,
    int64_t src_stride,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        dst[idx * dst_stride] = src[idx * src_stride];
    }
}

// 2D strided copy
extern "C" __global__
void copy_strided_2d_float_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int64_t dst_stride0,
    int64_t dst_stride1,
    int64_t src_stride0,
    int64_t src_stride1,
    uint32_t dim0,
    uint32_t dim1
) {
    uint32_t i = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < dim0 && j < dim1) {
        int64_t dst_idx = i * dst_stride0 + j * dst_stride1;
        int64_t src_idx = i * src_stride0 + j * src_stride1;
        dst[dst_idx] = src[src_idx];
    }
}

// 3D strided copy
extern "C" __global__
void copy_strided_3d_float_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int64_t dst_stride0,
    int64_t dst_stride1,
    int64_t dst_stride2,
    int64_t src_stride0,
    int64_t src_stride1,
    int64_t src_stride2,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2
) {
    uint32_t k = blockIdx.z;
    uint32_t i = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < dim0 && i < dim1 && j < dim2) {
        int64_t dst_idx = k * dst_stride0 + i * dst_stride1 + j * dst_stride2;
        int64_t src_idx = k * src_stride0 + i * src_stride1 + j * src_stride2;
        dst[dst_idx] = src[src_idx];
    }
}

// ============================================================================
// General N-D Strided Copy
// ============================================================================

extern "C" __global__
void copy_strided_nd_float_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    const int64_t* __restrict__ dst_strides,
    const int64_t* __restrict__ src_strides,
    const uint64_t* __restrict__ shape,
    uint32_t ndim,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Convert linear index to multi-dimensional indices
    uint64_t remaining = idx;
    int64_t dst_offset = 0;
    int64_t src_offset = 0;

    for (int d = ndim - 1; d >= 0; d--) {
        uint64_t dim_idx = remaining % shape[d];
        remaining /= shape[d];
        dst_offset += dim_idx * dst_strides[d];
        src_offset += dim_idx * src_strides[d];
    }

    dst[dst_offset] = src[src_offset];
}

// ============================================================================
// Broadcasting Copy Kernels
// ============================================================================

// Broadcast scalar to array
extern "C" __global__
void broadcast_scalar_float_kernel(
    float* __restrict__ dst,
    float value,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            dst[elem_idx] = value;
        }
    }
}

// Broadcast row vector to 2D
extern "C" __global__
void broadcast_row_float_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,  // Shape (N,)
    uint32_t M,
    uint32_t N
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)M * N;

    if (idx < total) {
        uint32_t col = idx % N;
        dst[idx] = src[col];
    }
}

// Broadcast column vector to 2D
extern "C" __global__
void broadcast_col_float_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,  // Shape (M,) or (M, 1)
    uint32_t M,
    uint32_t N
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)M * N;

    if (idx < total) {
        uint32_t row = idx / N;
        dst[idx] = src[row];
    }
}

// General broadcast with stride computation
extern "C" __global__
void broadcast_nd_float_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    const int64_t* __restrict__ src_strides,  // May have zeros for broadcast dims
    const uint64_t* __restrict__ shape,
    uint32_t ndim,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Convert linear index to source offset
    uint64_t remaining = idx;
    int64_t src_offset = 0;

    for (int d = ndim - 1; d >= 0; d--) {
        uint64_t dim_idx = remaining % shape[d];
        remaining /= shape[d];
        src_offset += dim_idx * src_strides[d];
    }

    dst[idx] = src[src_offset];
}

// ============================================================================
// Gather/Scatter Operations
// ============================================================================

// Gather: dst[i] = src[indices[i]]
extern "C" __global__
void gather_float_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    const int64_t* __restrict__ indices,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int64_t src_idx = indices[idx];
        dst[idx] = src[src_idx];
    }
}

// Scatter: dst[indices[i]] = src[i]
extern "C" __global__
void scatter_float_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    const int64_t* __restrict__ indices,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int64_t dst_idx = indices[idx];
        dst[dst_idx] = src[idx];
    }
}

// Gather along axis
extern "C" __global__
void gather_axis_float_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    const int64_t* __restrict__ indices,
    uint64_t outer_size,
    uint64_t gather_size,
    uint64_t inner_size,
    uint64_t src_axis_size
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = outer_size * gather_size * inner_size;

    if (idx < total) {
        uint64_t inner_idx = idx % inner_size;
        uint64_t gather_idx = (idx / inner_size) % gather_size;
        uint64_t outer_idx = idx / (inner_size * gather_size);

        int64_t src_gather_idx = indices[gather_idx];
        uint64_t src_idx = outer_idx * src_axis_size * inner_size +
                          src_gather_idx * inner_size +
                          inner_idx;

        dst[idx] = src[src_idx];
    }
}

// ============================================================================
// Transpose Operations
// ============================================================================

// 2D transpose
extern "C" __global__
void transpose_2d_float_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    uint32_t rows,
    uint32_t cols
) {
    __shared__ float tile[32][33];  // +1 to avoid bank conflicts

    uint32_t x = blockIdx.x * 32 + threadIdx.x;
    uint32_t y = blockIdx.y * 32 + threadIdx.y;

    // Load tile
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = src[y * cols + x];
    }
    __syncthreads();

    // Write transposed
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;

    if (x < rows && y < cols) {
        dst[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// General permute (dimension reordering)
extern "C" __global__
void permute_float_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    const int64_t* __restrict__ dst_strides,
    const int64_t* __restrict__ src_strides,
    const uint64_t* __restrict__ shape,  // Shape of output
    const uint32_t* __restrict__ perm,   // Permutation indices
    uint32_t ndim,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Convert dst linear index to multi-dim, then to src linear
    uint64_t remaining = idx;
    int64_t src_offset = 0;

    for (int d = ndim - 1; d >= 0; d--) {
        uint64_t dim_idx = remaining % shape[d];
        remaining /= shape[d];
        // perm[d] gives the source dimension
        src_offset += dim_idx * src_strides[perm[d]];
    }

    dst[idx] = src[src_offset];
}

// ============================================================================
// Slice/Copy Operations
// ============================================================================

// Copy a slice from src to dst (contiguous destination)
extern "C" __global__
void copy_slice_float_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int64_t src_offset,
    int64_t src_stride,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        dst[idx] = src[src_offset + idx * src_stride];
    }
}

// Fill region with constant
extern "C" __global__
void fill_float_kernel(
    float* __restrict__ dst,
    float value,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            dst[elem_idx] = value;
        }
    }
}

// Fill with zeros
extern "C" __global__
void zero_float_kernel(
    float* __restrict__ dst,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            dst[elem_idx] = 0.0f;
        }
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_copy_float(
    void* dst,
    const void* src,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    copy_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)dst, (const float*)src, n
    );

    return cudaGetLastError();
}

int lux_cuda_copy_half(
    void* dst,
    const void* src,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    copy_half_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (__half*)dst, (const __half*)src, n
    );

    return cudaGetLastError();
}

int lux_cuda_copy_float_to_half(
    void* dst,
    const void* src,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    copy_float_to_half_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (__half*)dst, (const float*)src, n
    );

    return cudaGetLastError();
}

int lux_cuda_copy_half_to_float(
    void* dst,
    const void* src,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    copy_half_to_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)dst, (const __half*)src, n
    );

    return cudaGetLastError();
}

int lux_cuda_copy_strided_2d_float(
    void* dst,
    const void* src,
    int64_t dst_stride0,
    int64_t dst_stride1,
    int64_t src_stride0,
    int64_t src_stride1,
    uint32_t dim0,
    uint32_t dim1,
    cudaStream_t stream
) {
    if (dim0 == 0 || dim1 == 0) return 0;

    dim3 block(16, 16);
    dim3 grid((dim1 + block.x - 1) / block.x, (dim0 + block.y - 1) / block.y);

    copy_strided_2d_float_kernel<<<grid, block, 0, stream>>>(
        (float*)dst, (const float*)src,
        dst_stride0, dst_stride1, src_stride0, src_stride1,
        dim0, dim1
    );

    return cudaGetLastError();
}

int lux_cuda_copy_strided_nd_float(
    void* dst,
    const void* src,
    const int64_t* dst_strides,
    const int64_t* src_strides,
    const uint64_t* shape,
    uint32_t ndim,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    copy_strided_nd_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)dst, (const float*)src,
        dst_strides, src_strides, shape, ndim, n
    );

    return cudaGetLastError();
}

int lux_cuda_broadcast_scalar_float(
    void* dst,
    float value,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    broadcast_scalar_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)dst, value, n
    );

    return cudaGetLastError();
}

int lux_cuda_gather_float(
    void* dst,
    const void* src,
    const int64_t* indices,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    gather_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)dst, (const float*)src, indices, n
    );

    return cudaGetLastError();
}

int lux_cuda_scatter_float(
    void* dst,
    const void* src,
    const int64_t* indices,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    scatter_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)dst, (const float*)src, indices, n
    );

    return cudaGetLastError();
}

int lux_cuda_transpose_2d_float(
    void* dst,
    const void* src,
    uint32_t rows,
    uint32_t cols,
    cudaStream_t stream
) {
    if (rows == 0 || cols == 0) return 0;

    dim3 block(32, 32);
    dim3 grid((cols + 31) / 32, (rows + 31) / 32);

    transpose_2d_float_kernel<<<grid, block, 0, stream>>>(
        (float*)dst, (const float*)src, rows, cols
    );

    return cudaGetLastError();
}

int lux_cuda_fill_float(
    void* dst,
    float value,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    fill_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)dst, value, n
    );

    return cudaGetLastError();
}

int lux_cuda_zero_float(
    void* dst,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    zero_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)dst, n
    );

    return cudaGetLastError();
}

}  // extern "C"
