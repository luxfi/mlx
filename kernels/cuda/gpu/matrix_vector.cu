// Copyright (c) 2024-2026 Lux Industries Inc. All Rights Reserved.
// PROPRIETARY AND CONFIDENTIAL - NO LICENSE GRANTED
// Contact: licensing@luxindustries.xyz
//
// General Matrix-Vector Multiplication - CUDA Implementation
// Optimized kernels for various matrix-vector operations including
// GEMV, GEMM row reduction, and specialized transpose variants.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace lux {
namespace cuda {
namespace matvec {

// ============================================================================
// Configuration
// ============================================================================

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE_X = 256;
constexpr int BLOCK_SIZE_Y = 4;
constexpr int TILE_K = 128;

// Operation types
enum class MatVecOp : uint32_t {
    MV = 0,      // y = A * x
    MTV = 1,     // y = A^T * x (transpose)
    MVC = 2,     // y = A^H * x (conjugate transpose, for complex)
    MVS = 3      // y = A * x + s * y (with scalar update)
};

// ============================================================================
// Utility Functions
// ============================================================================

__device__ __forceinline__
float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__
float block_reduce_sum(float val) {
    __shared__ float shared[WARP_SIZE];

    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;

    if (wid == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

// ============================================================================
// GEMV: y = alpha * A * x + beta * y
// ============================================================================

// Row-major GEMV (standard layout)
__global__ void gemv_row_major_f32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int N,
    float alpha,
    float beta
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M) return;

    float sum = 0.0f;

    #pragma unroll 8
    for (int j = 0; j < N; j++) {
        sum += A[row * N + j] * x[j];
    }

    if (beta == 0.0f) {
        y[row] = alpha * sum;
    } else {
        y[row] = alpha * sum + beta * y[row];
    }
}

// Column-major GEMV (Fortran layout)
__global__ void gemv_col_major_f32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int N,
    int lda,
    float alpha,
    float beta
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M) return;

    float sum = 0.0f;

    for (int j = 0; j < N; j++) {
        sum += A[j * lda + row] * x[j];
    }

    if (beta == 0.0f) {
        y[row] = alpha * sum;
    } else {
        y[row] = alpha * sum + beta * y[row];
    }
}

// ============================================================================
// GEMV Transpose: y = alpha * A^T * x + beta * y
// ============================================================================

// Transpose GEMV using parallel reduction
__global__ void gemvt_f32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int N,
    float alpha,
    float beta
) {
    extern __shared__ float sdata[];

    const int col = blockIdx.x;
    const int tid = threadIdx.x;

    if (col >= N) return;

    // Each thread computes partial sum for column
    float sum = 0.0f;
    for (int row = tid; row < M; row += blockDim.x) {
        sum += A[row * N + col] * x[row];
    }

    sdata[tid] = sum;
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (beta == 0.0f) {
            y[col] = alpha * sdata[0];
        } else {
            y[col] = alpha * sdata[0] + beta * y[col];
        }
    }
}

// ============================================================================
// Tiled GEMV (for large matrices)
// ============================================================================

// Tiled row-major GEMV with shared memory
__global__ void gemv_tiled_f32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int N,
    float alpha,
    float beta
) {
    __shared__ float sx[TILE_K];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= M) return;

    float sum = 0.0f;

    // Process in tiles
    for (int tile = 0; tile < N; tile += TILE_K) {
        // Cooperatively load x tile to shared memory
        if (tid < TILE_K && tile + tid < N) {
            sx[tid] = x[tile + tid];
        }
        __syncthreads();

        // Each thread processes a portion of the tile
        int tile_end = min(TILE_K, N - tile);
        for (int k = tid; k < tile_end; k += blockDim.x) {
            sum += A[row * N + tile + k] * sx[k];
        }
        __syncthreads();
    }

    // Reduce within block
    sum = block_reduce_sum(sum);

    if (tid == 0) {
        if (beta == 0.0f) {
            y[row] = alpha * sum;
        } else {
            y[row] = alpha * sum + beta * y[row];
        }
    }
}

// ============================================================================
// Vectorized GEMV (using float4)
// ============================================================================

__global__ void gemv_vectorized_f32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int N,
    float alpha,
    float beta
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M) return;

    float sum = 0.0f;
    const float* A_row = A + row * N;

    // Process 4 elements at a time
    int j = 0;
    for (; j + 3 < N; j += 4) {
        float4 a = *((float4*)(A_row + j));
        float4 b = *((float4*)(x + j));
        sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }

    // Handle remainder
    for (; j < N; j++) {
        sum += A_row[j] * x[j];
    }

    if (beta == 0.0f) {
        y[row] = alpha * sum;
    } else {
        y[row] = alpha * sum + beta * y[row];
    }
}

// ============================================================================
// FP16 GEMV
// ============================================================================

__global__ void gemv_f16_kernel(
    const half* __restrict__ A,
    const half* __restrict__ x,
    half* __restrict__ y,
    int M,
    int N,
    float alpha,
    float beta
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M) return;

    float sum = 0.0f;
    const half* A_row = A + row * N;

    // Process 2 elements at a time
    int j = 0;
    for (; j + 1 < N; j += 2) {
        half2 a = *((half2*)(A_row + j));
        half2 b = *((half2*)(x + j));
        sum += __half2float(a.x) * __half2float(b.x);
        sum += __half2float(a.y) * __half2float(b.y);
    }

    // Handle remainder
    for (; j < N; j++) {
        sum += __half2float(A_row[j]) * __half2float(x[j]);
    }

    if (beta == 0.0f) {
        y[row] = __float2half(alpha * sum);
    } else {
        y[row] = __float2half(alpha * sum + beta * __half2float(y[row]));
    }
}

// ============================================================================
// Batched GEMV
// ============================================================================

// Strided batched GEMV
__global__ void gemv_strided_batched_f32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int N,
    int batch_size,
    int stride_A,
    int stride_x,
    int stride_y,
    float alpha,
    float beta
) {
    const int batch = blockIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch >= batch_size || row >= M) return;

    const float* A_b = A + batch * stride_A;
    const float* x_b = x + batch * stride_x;
    float* y_b = y + batch * stride_y;

    float sum = 0.0f;
    const float* A_row = A_b + row * N;

    #pragma unroll 8
    for (int j = 0; j < N; j++) {
        sum += A_row[j] * x_b[j];
    }

    if (beta == 0.0f) {
        y_b[row] = alpha * sum;
    } else {
        y_b[row] = alpha * sum + beta * y_b[row];
    }
}

// Array of pointers batched GEMV
__global__ void gemv_batched_f32_kernel(
    const float* const* __restrict__ A,
    const float* const* __restrict__ x,
    float* const* __restrict__ y,
    int M,
    int N,
    int batch_size,
    float alpha,
    float beta
) {
    const int batch = blockIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch >= batch_size || row >= M) return;

    const float* A_b = A[batch];
    const float* x_b = x[batch];
    float* y_b = y[batch];

    float sum = 0.0f;
    const float* A_row = A_b + row * N;

    #pragma unroll 8
    for (int j = 0; j < N; j++) {
        sum += A_row[j] * x_b[j];
    }

    if (beta == 0.0f) {
        y_b[row] = alpha * sum;
    } else {
        y_b[row] = alpha * sum + beta * y_b[row];
    }
}

// ============================================================================
// Symmetric GEMV (for symmetric/Hermitian matrices)
// ============================================================================

// Symmetric GEMV (uses only upper or lower triangle)
__global__ void gemv_symmetric_f32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int N,
    bool upper,
    float alpha,
    float beta
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N) return;

    float sum = 0.0f;

    if (upper) {
        // Upper triangular: A[i,j] for j >= i
        // Use A[j,i] for j < i (symmetry)
        for (int j = 0; j < N; j++) {
            float a_ij = (j >= row) ? A[row * N + j] : A[j * N + row];
            sum += a_ij * x[j];
        }
    } else {
        // Lower triangular: A[i,j] for j <= i
        // Use A[j,i] for j > i (symmetry)
        for (int j = 0; j < N; j++) {
            float a_ij = (j <= row) ? A[row * N + j] : A[j * N + row];
            sum += a_ij * x[j];
        }
    }

    if (beta == 0.0f) {
        y[row] = alpha * sum;
    } else {
        y[row] = alpha * sum + beta * y[row];
    }
}

// ============================================================================
// Band Matrix GEMV
// ============================================================================

// Banded GEMV (for matrices with limited bandwidth)
__global__ void gemv_banded_f32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int N,
    int kl,     // Number of sub-diagonals
    int ku,     // Number of super-diagonals
    int lda,    // Leading dimension of banded storage
    float alpha,
    float beta
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M) return;

    float sum = 0.0f;

    // Band storage: A[ku + i - j, j] for max(0, j-ku) <= i <= min(M-1, j+kl)
    int j_start = max(0, row - kl);
    int j_end = min(N, row + ku + 1);

    for (int j = j_start; j < j_end; j++) {
        int band_row = ku + row - j;
        sum += A[band_row * N + j] * x[j];
    }

    if (beta == 0.0f) {
        y[row] = alpha * sum;
    } else {
        y[row] = alpha * sum + beta * y[row];
    }
}

// ============================================================================
// Sparse GEMV (CSR format)
// ============================================================================

// CSR sparse GEMV
__global__ void gemv_csr_f32_kernel(
    const float* __restrict__ values,
    const int* __restrict__ col_indices,
    const int* __restrict__ row_ptr,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    float alpha,
    float beta
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M) return;

    float sum = 0.0f;
    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];

    for (int j = row_start; j < row_end; j++) {
        sum += values[j] * x[col_indices[j]];
    }

    if (beta == 0.0f) {
        y[row] = alpha * sum;
    } else {
        y[row] = alpha * sum + beta * y[row];
    }
}

// ============================================================================
// Matrix-Vector Add/Scale Operations
// ============================================================================

// y = alpha * x + y
__global__ void axpy_f32_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N,
    float alpha
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;

    y[idx] = alpha * x[idx] + y[idx];
}

// y = alpha * x
__global__ void scal_f32_kernel(
    float* __restrict__ x,
    int N,
    float alpha
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;

    x[idx] *= alpha;
}

// Dot product: result = x . y
__global__ void dot_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ result,
    int N
) {
    extern __shared__ float sdata[];

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        sum += x[i] * y[i];
    }

    sdata[tid] = sum;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// ============================================================================
// Host API (C Interface)
// ============================================================================

} // namespace matvec
} // namespace cuda
} // namespace lux

extern "C" {

using namespace lux::cuda::matvec;

void lux_cuda_gemv_f32(
    const float* A,
    const float* x,
    float* y,
    int M,
    int N,
    float alpha,
    float beta,
    bool row_major,
    int lda,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE_X);
    dim3 grid((M + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X);

    if (row_major) {
        gemv_row_major_f32_kernel<<<grid, block, 0, stream>>>(
            A, x, y, M, N, alpha, beta
        );
    } else {
        gemv_col_major_f32_kernel<<<grid, block, 0, stream>>>(
            A, x, y, M, N, lda, alpha, beta
        );
    }
}

void lux_cuda_gemvt_f32(
    const float* A,
    const float* x,
    float* y,
    int M,
    int N,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE_X);
    dim3 grid(N);
    size_t shmem = BLOCK_SIZE_X * sizeof(float);

    gemvt_f32_kernel<<<grid, block, shmem, stream>>>(
        A, x, y, M, N, alpha, beta
    );
}

void lux_cuda_gemv_tiled_f32(
    const float* A,
    const float* x,
    float* y,
    int M,
    int N,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE_X);
    dim3 grid(M);

    gemv_tiled_f32_kernel<<<grid, block, 0, stream>>>(
        A, x, y, M, N, alpha, beta
    );
}

void lux_cuda_gemv_vectorized_f32(
    const float* A,
    const float* x,
    float* y,
    int M,
    int N,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE_X);
    dim3 grid((M + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X);

    gemv_vectorized_f32_kernel<<<grid, block, 0, stream>>>(
        A, x, y, M, N, alpha, beta
    );
}

void lux_cuda_gemv_f16(
    const void* A,
    const void* x,
    void* y,
    int M,
    int N,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE_X);
    dim3 grid((M + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X);

    gemv_f16_kernel<<<grid, block, 0, stream>>>(
        (const half*)A, (const half*)x, (half*)y, M, N, alpha, beta
    );
}

void lux_cuda_gemv_strided_batched_f32(
    const float* A,
    const float* x,
    float* y,
    int M,
    int N,
    int batch_size,
    int stride_A,
    int stride_x,
    int stride_y,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE_X);
    dim3 grid((M + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, batch_size);

    gemv_strided_batched_f32_kernel<<<grid, block, 0, stream>>>(
        A, x, y, M, N, batch_size, stride_A, stride_x, stride_y, alpha, beta
    );
}

void lux_cuda_gemv_batched_f32(
    const float* const* A,
    const float* const* x,
    float* const* y,
    int M,
    int N,
    int batch_size,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE_X);
    dim3 grid((M + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, batch_size);

    gemv_batched_f32_kernel<<<grid, block, 0, stream>>>(
        A, x, y, M, N, batch_size, alpha, beta
    );
}

void lux_cuda_gemv_symmetric_f32(
    const float* A,
    const float* x,
    float* y,
    int N,
    bool upper,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE_X);
    dim3 grid((N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X);

    gemv_symmetric_f32_kernel<<<grid, block, 0, stream>>>(
        A, x, y, N, upper, alpha, beta
    );
}

void lux_cuda_gemv_banded_f32(
    const float* A,
    const float* x,
    float* y,
    int M,
    int N,
    int kl,
    int ku,
    int lda,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE_X);
    dim3 grid((M + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X);

    gemv_banded_f32_kernel<<<grid, block, 0, stream>>>(
        A, x, y, M, N, kl, ku, lda, alpha, beta
    );
}

void lux_cuda_gemv_csr_f32(
    const float* values,
    const int* col_indices,
    const int* row_ptr,
    const float* x,
    float* y,
    int M,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE_X);
    dim3 grid((M + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X);

    gemv_csr_f32_kernel<<<grid, block, 0, stream>>>(
        values, col_indices, row_ptr, x, y, M, alpha, beta
    );
}

void lux_cuda_axpy_f32(
    const float* x,
    float* y,
    int N,
    float alpha,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE_X);
    dim3 grid((N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X);

    axpy_f32_kernel<<<grid, block, 0, stream>>>(x, y, N, alpha);
}

void lux_cuda_scal_f32(
    float* x,
    int N,
    float alpha,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE_X);
    dim3 grid((N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X);

    scal_f32_kernel<<<grid, block, 0, stream>>>(x, N, alpha);
}

void lux_cuda_dot_f32(
    const float* x,
    const float* y,
    float* result,
    int N,
    cudaStream_t stream
) {
    cudaMemsetAsync(result, 0, sizeof(float), stream);

    dim3 block(BLOCK_SIZE_X);
    dim3 grid(min(256, (N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X));
    size_t shmem = BLOCK_SIZE_X * sizeof(float);

    dot_f32_kernel<<<grid, block, shmem, stream>>>(x, y, result, N);
}

} // extern "C"
