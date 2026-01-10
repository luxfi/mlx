// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// GEMV CUDA Kernels
// Matrix-vector multiplication: y = alpha * A * x + beta * y
// Supports both row-major (GEMV) and column-major (GEMV_T) layouts

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// ============================================================================
// Configuration
// ============================================================================

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define TILE_SIZE 128

// ============================================================================
// Utility: Warp and Block Reductions
// ============================================================================

__device__ __forceinline__
float warp_reduce_sum_f32(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__
float block_reduce_sum_f32(float val) {
    __shared__ float shared[WARP_SIZE];

    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum_f32(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;

    if (wid == 0) {
        val = warp_reduce_sum_f32(val);
    }

    return val;
}

// ============================================================================
// GEMV: y = alpha * A * x + beta * y (Row-major)
// A is M x N, x is N, y is M
// ============================================================================

extern "C" __global__
void lux_gemv_f32_kernel(
    float* __restrict__ y,
    const float* __restrict__ A,
    const float* __restrict__ x,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N
) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (row >= M) return;

    const float* A_row = A + row * N;

    // Each thread computes partial dot product
    float sum = 0.0f;

    // Vectorized load: 4 elements at a time
    uint32_t j = tid * 4;
    for (; j + 3 < N; j += blockDim.x * 4) {
        float4 a = *((float4*)(A_row + j));
        float4 b = *((float4*)(x + j));
        sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }

    // Handle remainder
    for (uint32_t k = j; k < N; k += blockDim.x) {
        if (k + tid < N) {
            sum += A_row[k + tid] * x[k + tid];
        }
    }

    // Block reduction
    sum = block_reduce_sum_f32(sum);

    if (tid == 0) {
        if (beta == 0.0f) {
            y[row] = alpha * sum;
        } else {
            y[row] = alpha * sum + beta * y[row];
        }
    }
}

// ============================================================================
// GEMV Simple: One thread per row (for small N)
// ============================================================================

extern "C" __global__
void lux_gemv_simple_f32_kernel(
    float* __restrict__ y,
    const float* __restrict__ A,
    const float* __restrict__ x,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M) return;

    const float* A_row = A + row * N;
    float sum = 0.0f;

    #pragma unroll 8
    for (uint32_t j = 0; j < N; j++) {
        sum += A_row[j] * x[j];
    }

    if (beta == 0.0f) {
        y[row] = alpha * sum;
    } else {
        y[row] = alpha * sum + beta * y[row];
    }
}

// ============================================================================
// GEMV_T: y = alpha * A^T * x + beta * y (Transpose)
// A is M x N, x is M, y is N
// ============================================================================

extern "C" __global__
void lux_gemvt_f32_kernel(
    float* __restrict__ y,
    const float* __restrict__ A,
    const float* __restrict__ x,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N
) {
    extern __shared__ float sdata[];

    uint32_t col = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (col >= N) return;

    // Each thread computes partial sum for column
    float sum = 0.0f;
    for (uint32_t row = tid; row < M; row += blockDim.x) {
        sum += A[row * N + col] * x[row];
    }

    sdata[tid] = sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
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
// GEMV with Shared Memory Tiling
// ============================================================================

extern "C" __global__
void lux_gemv_tiled_f32_kernel(
    float* __restrict__ y,
    const float* __restrict__ A,
    const float* __restrict__ x,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N
) {
    __shared__ float sx[TILE_SIZE];

    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (row >= M) return;

    const float* A_row = A + row * N;
    float sum = 0.0f;

    // Process x in tiles
    for (uint32_t tile = 0; tile < N; tile += TILE_SIZE) {
        // Cooperatively load x tile to shared memory
        uint32_t load_idx = tile + tid;
        if (tid < TILE_SIZE && load_idx < N) {
            sx[tid] = x[load_idx];
        }
        __syncthreads();

        // Compute partial dot product for this tile
        uint32_t tile_end = min(TILE_SIZE, N - tile);
        for (uint32_t k = tid; k < tile_end; k += blockDim.x) {
            sum += A_row[tile + k] * sx[k];
        }
        __syncthreads();
    }

    // Reduce within block
    sum = block_reduce_sum_f32(sum);

    if (tid == 0) {
        if (beta == 0.0f) {
            y[row] = alpha * sum;
        } else {
            y[row] = alpha * sum + beta * y[row];
        }
    }
}

// ============================================================================
// FP16 GEMV
// ============================================================================

extern "C" __global__
void lux_gemv_f16_kernel(
    __half* __restrict__ y,
    const __half* __restrict__ A,
    const __half* __restrict__ x,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N
) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (row >= M) return;

    const __half* A_row = A + row * N;

    // Accumulate in FP32 for precision
    float sum = 0.0f;

    // Process 2 elements at a time using half2
    uint32_t j = tid * 2;
    for (; j + 1 < N; j += blockDim.x * 2) {
        half2 a = *((half2*)(A_row + j));
        half2 b = *((half2*)(x + j));
        sum += __half2float(a.x) * __half2float(b.x);
        sum += __half2float(a.y) * __half2float(b.y);
    }

    // Handle remainder
    if (j < N && (j - tid * 2) + tid < N) {
        uint32_t idx = (j - tid * 2) + tid;
        if (idx < N) {
            sum += __half2float(A_row[idx]) * __half2float(x[idx]);
        }
    }

    // Block reduction
    sum = block_reduce_sum_f32(sum);

    if (tid == 0) {
        if (beta == 0.0f) {
            y[row] = __float2half(alpha * sum);
        } else {
            y[row] = __float2half(alpha * sum + beta * __half2float(y[row]));
        }
    }
}

// ============================================================================
// FP16 GEMV_T
// ============================================================================

extern "C" __global__
void lux_gemvt_f16_kernel(
    __half* __restrict__ y,
    const __half* __restrict__ A,
    const __half* __restrict__ x,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N
) {
    extern __shared__ float sdata[];

    uint32_t col = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (col >= N) return;

    float sum = 0.0f;
    for (uint32_t row = tid; row < M; row += blockDim.x) {
        sum += __half2float(A[row * N + col]) * __half2float(x[row]);
    }

    sdata[tid] = sum;
    __syncthreads();

    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (beta == 0.0f) {
            y[col] = __float2half(alpha * sdata[0]);
        } else {
            y[col] = __float2half(alpha * sdata[0] + beta * __half2float(y[col]));
        }
    }
}

// ============================================================================
// Batched GEMV
// ============================================================================

extern "C" __global__
void lux_gemv_batched_f32_kernel(
    float* __restrict__ y,
    const float* __restrict__ A,
    const float* __restrict__ x,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N,
    uint32_t batch_size,
    int64_t stride_A,
    int64_t stride_x,
    int64_t stride_y
) {
    uint32_t batch = blockIdx.y;
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (batch >= batch_size || row >= M) return;

    const float* A_batch = A + batch * stride_A + row * N;
    const float* x_batch = x + batch * stride_x;
    float* y_batch = y + batch * stride_y;

    float sum = 0.0f;
    for (uint32_t j = tid; j < N; j += blockDim.x) {
        sum += A_batch[j] * x_batch[j];
    }

    sum = block_reduce_sum_f32(sum);

    if (tid == 0) {
        if (beta == 0.0f) {
            y_batch[row] = alpha * sum;
        } else {
            y_batch[row] = alpha * sum + beta * y_batch[row];
        }
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_gemv_f32(
    void* y,
    const void* A,
    const void* x,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N,
    cudaStream_t stream
) {
    if (N <= 256) {
        // Use simple kernel for small N
        uint32_t threads = BLOCK_SIZE;
        uint32_t blocks = (M + threads - 1) / threads;

        lux_gemv_simple_f32_kernel<<<blocks, threads, 0, stream>>>(
            (float*)y, (const float*)A, (const float*)x,
            alpha, beta, M, N
        );
    } else {
        // Use tiled kernel for large N
        uint32_t threads = min(BLOCK_SIZE, N);

        lux_gemv_tiled_f32_kernel<<<M, threads, 0, stream>>>(
            (float*)y, (const float*)A, (const float*)x,
            alpha, beta, M, N
        );
    }

    return cudaGetLastError();
}

int lux_cuda_gemvt_f32(
    void* y,
    const void* A,
    const void* x,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N,
    cudaStream_t stream
) {
    uint32_t threads = min(BLOCK_SIZE, M);
    size_t shared_size = threads * sizeof(float);

    lux_gemvt_f32_kernel<<<N, threads, shared_size, stream>>>(
        (float*)y, (const float*)A, (const float*)x,
        alpha, beta, M, N
    );

    return cudaGetLastError();
}

int lux_cuda_gemv_f16(
    void* y,
    const void* A,
    const void* x,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N,
    cudaStream_t stream
) {
    uint32_t threads = min(BLOCK_SIZE, N);

    lux_gemv_f16_kernel<<<M, threads, 0, stream>>>(
        (__half*)y, (const __half*)A, (const __half*)x,
        alpha, beta, M, N
    );

    return cudaGetLastError();
}

int lux_cuda_gemvt_f16(
    void* y,
    const void* A,
    const void* x,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N,
    cudaStream_t stream
) {
    uint32_t threads = min(BLOCK_SIZE, M);
    size_t shared_size = threads * sizeof(float);

    lux_gemvt_f16_kernel<<<N, threads, shared_size, stream>>>(
        (__half*)y, (const __half*)A, (const __half*)x,
        alpha, beta, M, N
    );

    return cudaGetLastError();
}

int lux_cuda_gemv_batched_f32(
    void* y,
    const void* A,
    const void* x,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N,
    uint32_t batch_size,
    int64_t stride_A,
    int64_t stride_x,
    int64_t stride_y,
    cudaStream_t stream
) {
    dim3 blocks(M, batch_size);
    uint32_t threads = min(BLOCK_SIZE, N);

    lux_gemv_batched_f32_kernel<<<blocks, threads, 0, stream>>>(
        (float*)y, (const float*)A, (const float*)x,
        alpha, beta, M, N, batch_size,
        stride_A, stride_x, stride_y
    );

    return cudaGetLastError();
}

}  // extern "C"
