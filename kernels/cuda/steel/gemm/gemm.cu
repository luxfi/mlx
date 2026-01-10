// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Steel GEMM CUDA Kernels
// High-performance matrix multiplication

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// GEMM Configuration
// ============================================================================

#define TILE_M 128
#define TILE_N 128
#define TILE_K 8
#define BLOCK_SIZE 256

// ============================================================================
// Basic GEMM (C = alpha * A @ B + beta * C)
// ============================================================================

extern "C" __global__
void steel_gemm_kernel(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N,
    uint32_t K
) {
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    uint32_t bx = blockIdx.x;
    uint32_t by = blockIdx.y;
    uint32_t tx = threadIdx.x % 16;
    uint32_t ty = threadIdx.x / 16;

    uint32_t row = by * TILE_M + ty;
    uint32_t col = bx * TILE_N + tx;

    float acc = 0.0f;

    for (uint32_t t = 0; t < (K + TILE_K - 1) / TILE_K; t++) {
        // Load A tile
        for (uint32_t i = threadIdx.x; i < TILE_M * TILE_K; i += blockDim.x) {
            uint32_t ti = i / TILE_K;
            uint32_t tk = i % TILE_K;
            uint32_t a_row = by * TILE_M + ti;
            uint32_t a_col = t * TILE_K + tk;

            if (a_row < M && a_col < K) {
                As[ti][tk] = A[a_row * K + a_col];
            } else {
                As[ti][tk] = 0.0f;
            }
        }

        // Load B tile
        for (uint32_t i = threadIdx.x; i < TILE_K * TILE_N; i += blockDim.x) {
            uint32_t tk = i / TILE_N;
            uint32_t tj = i % TILE_N;
            uint32_t b_row = t * TILE_K + tk;
            uint32_t b_col = bx * TILE_N + tj;

            if (b_row < K && b_col < N) {
                Bs[tk][tj] = B[b_row * N + b_col];
            } else {
                Bs[tk][tj] = 0.0f;
            }
        }

        __syncthreads();

        // Compute partial product
        #pragma unroll
        for (uint32_t k = 0; k < TILE_K; k++) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        if (beta != 0.0f) {
            C[row * N + col] = alpha * acc + beta * C[row * N + col];
        } else {
            C[row * N + col] = alpha * acc;
        }
    }
}

// ============================================================================
// GEMM with Transpose Options
// ============================================================================

extern "C" __global__
void steel_gemm_tn_kernel(
    float* __restrict__ C,
    const float* __restrict__ A,  // Transposed: [K, M]
    const float* __restrict__ B,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N,
    uint32_t K
) {
    __shared__ float As[TILE_K][TILE_M];
    __shared__ float Bs[TILE_K][TILE_N];

    uint32_t bx = blockIdx.x;
    uint32_t by = blockIdx.y;
    uint32_t tx = threadIdx.x % 16;
    uint32_t ty = threadIdx.x / 16;

    uint32_t row = by * TILE_M + ty;
    uint32_t col = bx * TILE_N + tx;

    float acc = 0.0f;

    for (uint32_t t = 0; t < (K + TILE_K - 1) / TILE_K; t++) {
        // Load A^T tile (reading A as K x M)
        for (uint32_t i = threadIdx.x; i < TILE_K * TILE_M; i += blockDim.x) {
            uint32_t tk = i / TILE_M;
            uint32_t ti = i % TILE_M;
            uint32_t a_row = t * TILE_K + tk;
            uint32_t a_col = by * TILE_M + ti;

            if (a_row < K && a_col < M) {
                As[tk][ti] = A[a_row * M + a_col];
            } else {
                As[tk][ti] = 0.0f;
            }
        }

        // Load B tile
        for (uint32_t i = threadIdx.x; i < TILE_K * TILE_N; i += blockDim.x) {
            uint32_t tk = i / TILE_N;
            uint32_t tj = i % TILE_N;
            uint32_t b_row = t * TILE_K + tk;
            uint32_t b_col = bx * TILE_N + tj;

            if (b_row < K && b_col < N) {
                Bs[tk][tj] = B[b_row * N + b_col];
            } else {
                Bs[tk][tj] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (uint32_t k = 0; k < TILE_K; k++) {
            acc += As[k][ty] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        if (beta != 0.0f) {
            C[row * N + col] = alpha * acc + beta * C[row * N + col];
        } else {
            C[row * N + col] = alpha * acc;
        }
    }
}

// ============================================================================
// FP16 GEMM
// ============================================================================

extern "C" __global__
void steel_gemm_fp16_kernel(
    __half* __restrict__ C,
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N,
    uint32_t K
) {
    __shared__ __half As[TILE_M][TILE_K];
    __shared__ __half Bs[TILE_K][TILE_N];

    uint32_t bx = blockIdx.x;
    uint32_t by = blockIdx.y;
    uint32_t tx = threadIdx.x % 16;
    uint32_t ty = threadIdx.x / 16;

    uint32_t row = by * TILE_M + ty;
    uint32_t col = bx * TILE_N + tx;

    float acc = 0.0f;  // Accumulate in FP32

    for (uint32_t t = 0; t < (K + TILE_K - 1) / TILE_K; t++) {
        // Load tiles
        for (uint32_t i = threadIdx.x; i < TILE_M * TILE_K; i += blockDim.x) {
            uint32_t ti = i / TILE_K;
            uint32_t tk = i % TILE_K;
            uint32_t a_row = by * TILE_M + ti;
            uint32_t a_col = t * TILE_K + tk;

            if (a_row < M && a_col < K) {
                As[ti][tk] = A[a_row * K + a_col];
            } else {
                As[ti][tk] = __float2half(0.0f);
            }
        }

        for (uint32_t i = threadIdx.x; i < TILE_K * TILE_N; i += blockDim.x) {
            uint32_t tk = i / TILE_N;
            uint32_t tj = i % TILE_N;
            uint32_t b_row = t * TILE_K + tk;
            uint32_t b_col = bx * TILE_N + tj;

            if (b_row < K && b_col < N) {
                Bs[tk][tj] = B[b_row * N + b_col];
            } else {
                Bs[tk][tj] = __float2half(0.0f);
            }
        }

        __syncthreads();

        #pragma unroll
        for (uint32_t k = 0; k < TILE_K; k++) {
            acc += __half2float(As[ty][k]) * __half2float(Bs[k][tx]);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        if (beta != 0.0f) {
            C[row * N + col] = __float2half(alpha * acc + beta * __half2float(C[row * N + col]));
        } else {
            C[row * N + col] = __float2half(alpha * acc);
        }
    }
}

// ============================================================================
// GEMV (Matrix-Vector Multiplication)
// ============================================================================

extern "C" __global__
void steel_gemv_kernel(
    float* __restrict__ y,
    const float* __restrict__ A,
    const float* __restrict__ x,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N
) {
    extern __shared__ float smem[];

    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (row >= M) return;

    const float* A_row = A + row * N;

    // Parallel dot product
    float partial = 0.0f;
    for (uint32_t j = tid; j < N; j += blockDim.x) {
        partial += A_row[j] * x[j];
    }

    smem[tid] = partial;
    __syncthreads();

    // Reduction
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (beta != 0.0f) {
            y[row] = alpha * smem[0] + beta * y[row];
        } else {
            y[row] = alpha * smem[0];
        }
    }
}

// ============================================================================
// Strided Batched GEMM Helper
// ============================================================================

extern "C" __global__
void steel_gemm_strided_kernel(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    int64_t stride_a,
    int64_t stride_b,
    int64_t stride_c,
    uint32_t batch_size
) {
    uint32_t batch = blockIdx.z;
    if (batch >= batch_size) return;

    const float* A_batch = A + batch * stride_a;
    const float* B_batch = B + batch * stride_b;
    float* C_batch = C + batch * stride_c;

    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    uint32_t bx = blockIdx.x;
    uint32_t by = blockIdx.y;
    uint32_t tx = threadIdx.x % 16;
    uint32_t ty = threadIdx.x / 16;

    uint32_t row = by * TILE_M + ty;
    uint32_t col = bx * TILE_N + tx;

    float acc = 0.0f;

    for (uint32_t t = 0; t < (K + TILE_K - 1) / TILE_K; t++) {
        for (uint32_t i = threadIdx.x; i < TILE_M * TILE_K; i += blockDim.x) {
            uint32_t ti = i / TILE_K;
            uint32_t tk = i % TILE_K;
            uint32_t a_row = by * TILE_M + ti;
            uint32_t a_col = t * TILE_K + tk;

            if (a_row < M && a_col < K) {
                As[ti][tk] = A_batch[a_row * K + a_col];
            } else {
                As[ti][tk] = 0.0f;
            }
        }

        for (uint32_t i = threadIdx.x; i < TILE_K * TILE_N; i += blockDim.x) {
            uint32_t tk = i / TILE_N;
            uint32_t tj = i % TILE_N;
            uint32_t b_row = t * TILE_K + tk;
            uint32_t b_col = bx * TILE_N + tj;

            if (b_row < K && b_col < N) {
                Bs[tk][tj] = B_batch[b_row * N + b_col];
            } else {
                Bs[tk][tj] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (uint32_t k = 0; k < TILE_K; k++) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        if (beta != 0.0f) {
            C_batch[row * N + col] = alpha * acc + beta * C_batch[row * N + col];
        } else {
            C_batch[row * N + col] = alpha * acc;
        }
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_steel_gemm(
    void* C,
    const void* A,
    const void* B,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    cudaStream_t stream
) {
    dim3 blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    uint32_t threads = BLOCK_SIZE;

    steel_gemm_kernel<<<blocks, threads, 0, stream>>>(
        (float*)C,
        (const float*)A,
        (const float*)B,
        alpha, beta,
        M, N, K
    );

    return cudaGetLastError();
}

int lux_cuda_steel_gemm_tn(
    void* C,
    const void* A,
    const void* B,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    cudaStream_t stream
) {
    dim3 blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    uint32_t threads = BLOCK_SIZE;

    steel_gemm_tn_kernel<<<blocks, threads, 0, stream>>>(
        (float*)C,
        (const float*)A,
        (const float*)B,
        alpha, beta,
        M, N, K
    );

    return cudaGetLastError();
}

int lux_cuda_steel_gemm_fp16(
    void* C,
    const void* A,
    const void* B,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    cudaStream_t stream
) {
    dim3 blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    uint32_t threads = BLOCK_SIZE;

    steel_gemm_fp16_kernel<<<blocks, threads, 0, stream>>>(
        (__half*)C,
        (const __half*)A,
        (const __half*)B,
        alpha, beta,
        M, N, K
    );

    return cudaGetLastError();
}

int lux_cuda_steel_gemv(
    void* y,
    const void* A,
    const void* x,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N,
    cudaStream_t stream
) {
    uint32_t threads = 256;
    size_t shared_size = threads * sizeof(float);

    steel_gemv_kernel<<<M, threads, shared_size, stream>>>(
        (float*)y,
        (const float*)A,
        (const float*)x,
        alpha, beta,
        M, N
    );

    return cudaGetLastError();
}

int lux_cuda_steel_gemm_strided_batched(
    void* C,
    const void* A,
    const void* B,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    int64_t stride_a,
    int64_t stride_b,
    int64_t stride_c,
    uint32_t batch_size,
    cudaStream_t stream
) {
    dim3 blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M, batch_size);
    uint32_t threads = BLOCK_SIZE;

    steel_gemm_strided_kernel<<<blocks, threads, 0, stream>>>(
        (float*)C,
        (const float*)A,
        (const float*)B,
        alpha, beta,
        M, N, K,
        stride_a, stride_b, stride_c,
        batch_size
    );

    return cudaGetLastError();
}

}  // extern "C"
