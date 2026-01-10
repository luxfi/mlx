// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// CUDA Tensor Operations
// Optimized GPU kernels for ML tensor operations

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cfloat>
#include <cmath>

// =============================================================================
// Configuration
// =============================================================================

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define TILE_SIZE 16  // For tiled matmul

// =============================================================================
// Elementwise Binary Kernels
// =============================================================================

extern "C" __global__
void lux_add_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ a,
    const float* __restrict__ b,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] + b[idx];
    }
}

extern "C" __global__
void lux_sub_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ a,
    const float* __restrict__ b,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] - b[idx];
    }
}

extern "C" __global__
void lux_mul_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ a,
    const float* __restrict__ b,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] * b[idx];
    }
}

extern "C" __global__
void lux_div_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ a,
    const float* __restrict__ b,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] / b[idx];
    }
}

// Vectorized versions for better memory throughput
extern "C" __global__
void lux_add_f32_vec4_kernel(
    float4* __restrict__ output,
    const float4* __restrict__ a,
    const float4* __restrict__ b,
    size_t n4
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 va = a[idx];
        float4 vb = b[idx];
        output[idx] = make_float4(
            va.x + vb.x,
            va.y + vb.y,
            va.z + vb.z,
            va.w + vb.w
        );
    }
}

extern "C" __global__
void lux_mul_f32_vec4_kernel(
    float4* __restrict__ output,
    const float4* __restrict__ a,
    const float4* __restrict__ b,
    size_t n4
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 va = a[idx];
        float4 vb = b[idx];
        output[idx] = make_float4(
            va.x * vb.x,
            va.y * vb.y,
            va.z * vb.z,
            va.w * vb.w
        );
    }
}

// =============================================================================
// Tiled Matrix Multiplication
// =============================================================================

// GEMM with shared memory tiling
// C[M,N] = A[M,K] @ B[K,N]
extern "C" __global__
void lux_matmul_tiled_f32_kernel(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    int M, int K, int N
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile of A into shared memory
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile of B into shared memory
        int b_row = t * TILE_SIZE + ty;
        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// =============================================================================
// Matrix Transpose
// =============================================================================

// Tiled transpose with bank conflict avoidance
extern "C" __global__
void lux_transpose_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int rows, int cols
) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Read into shared memory (coalesced read)
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }

    __syncthreads();

    // Transposed coordinates
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    // Write from shared memory (coalesced write)
    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// =============================================================================
// Unary Operations
// =============================================================================

extern "C" __global__
void lux_exp_f32_kernel(float* out, const float* in, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = expf(in[idx]);
}

extern "C" __global__
void lux_log_f32_kernel(float* out, const float* in, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = logf(in[idx]);
}

extern "C" __global__
void lux_sqrt_f32_kernel(float* out, const float* in, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = sqrtf(in[idx]);
}

extern "C" __global__
void lux_neg_f32_kernel(float* out, const float* in, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = -in[idx];
}

extern "C" __global__
void lux_abs_f32_kernel(float* out, const float* in, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fabsf(in[idx]);
}

extern "C" __global__
void lux_tanh_f32_kernel(float* out, const float* in, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = tanhf(in[idx]);
}

extern "C" __global__
void lux_sigmoid_f32_kernel(float* out, const float* in, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = 1.0f / (1.0f + expf(-in[idx]));
}

extern "C" __global__
void lux_relu_f32_kernel(float* out, const float* in, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fmaxf(0.0f, in[idx]);
}

// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
extern "C" __global__
void lux_gelu_f32_kernel(float* out, const float* in, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        float x3 = x * x * x;
        // sqrt(2/pi) = 0.7978845608
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        out[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

// =============================================================================
// Copy
// =============================================================================

extern "C" __global__
void lux_copy_f32_kernel(float* dst, const float* src, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = src[idx];
}

// =============================================================================
// Layer Normalization
// =============================================================================

__device__ __forceinline__
float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
extern "C" __global__
void lux_layer_norm_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    uint32_t batch_size,
    uint32_t dim,
    float eps
) {
    __shared__ float s_mean;
    __shared__ float s_var;
    __shared__ float warp_data[WARP_SIZE];

    uint32_t batch_idx = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const float* x = input + batch_idx * dim;
    float* y = output + batch_idx * dim;

    // Compute mean
    float local_sum = 0.0f;
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        local_sum += x[i];
    }

    local_sum = warp_reduce_sum(local_sum);
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    if (lane == 0) warp_data[wid] = local_sum;
    __syncthreads();

    if (wid == 0) {
        local_sum = (lane < blockDim.x / WARP_SIZE) ? warp_data[lane] : 0.0f;
        local_sum = warp_reduce_sum(local_sum);
        if (lane == 0) s_mean = local_sum / dim;
    }
    __syncthreads();

    float mean = s_mean;

    // Compute variance
    float local_var = 0.0f;
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        float diff = x[i] - mean;
        local_var += diff * diff;
    }

    local_var = warp_reduce_sum(local_var);
    if (lane == 0) warp_data[wid] = local_var;
    __syncthreads();

    if (wid == 0) {
        local_var = (lane < blockDim.x / WARP_SIZE) ? warp_data[lane] : 0.0f;
        local_var = warp_reduce_sum(local_var);
        if (lane == 0) s_var = local_var / dim;
    }
    __syncthreads();

    float inv_std = rsqrtf(s_var + eps);

    // Normalize and scale
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        float normalized = (x[i] - mean) * inv_std;
        y[i] = normalized * gamma[i] + beta[i];
    }
}

// =============================================================================
// RMS Normalization
// =============================================================================

// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
extern "C" __global__
void lux_rms_norm_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    uint32_t batch_size,
    uint32_t dim,
    float eps
) {
    __shared__ float s_rms;
    __shared__ float warp_data[WARP_SIZE];

    uint32_t batch_idx = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const float* x = input + batch_idx * dim;
    float* y = output + batch_idx * dim;

    // Compute sum of squares
    float local_sum_sq = 0.0f;
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        float val = x[i];
        local_sum_sq += val * val;
    }

    local_sum_sq = warp_reduce_sum(local_sum_sq);
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    if (lane == 0) warp_data[wid] = local_sum_sq;
    __syncthreads();

    if (wid == 0) {
        local_sum_sq = (lane < blockDim.x / WARP_SIZE) ? warp_data[lane] : 0.0f;
        local_sum_sq = warp_reduce_sum(local_sum_sq);
        if (lane == 0) s_rms = rsqrtf(local_sum_sq / dim + eps);
    }
    __syncthreads();

    float rms_scale = s_rms;

    // Scale
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        y[i] = x[i] * rms_scale * weight[i];
    }
}

// =============================================================================
// C API
// =============================================================================

extern "C" {

int lux_cuda_add_f32(void* output, const void* a, const void* b, size_t n, cudaStream_t stream) {
    if (n == 0) return 0;

    // Use vectorized version when aligned
    if (n % 4 == 0 && (uintptr_t)output % 16 == 0 && (uintptr_t)a % 16 == 0 && (uintptr_t)b % 16 == 0) {
        size_t n4 = n / 4;
        size_t num_blocks = (n4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        lux_add_f32_vec4_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            (float4*)output, (const float4*)a, (const float4*)b, n4
        );
    } else {
        size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        lux_add_f32_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            (float*)output, (const float*)a, (const float*)b, n
        );
    }
    return cudaGetLastError();
}

int lux_cuda_sub_f32(void* output, const void* a, const void* b, size_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    lux_sub_f32_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)output, (const float*)a, (const float*)b, n
    );
    return cudaGetLastError();
}

int lux_cuda_mul_f32(void* output, const void* a, const void* b, size_t n, cudaStream_t stream) {
    if (n == 0) return 0;

    if (n % 4 == 0 && (uintptr_t)output % 16 == 0 && (uintptr_t)a % 16 == 0 && (uintptr_t)b % 16 == 0) {
        size_t n4 = n / 4;
        size_t num_blocks = (n4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        lux_mul_f32_vec4_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            (float4*)output, (const float4*)a, (const float4*)b, n4
        );
    } else {
        size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        lux_mul_f32_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            (float*)output, (const float*)a, (const float*)b, n
        );
    }
    return cudaGetLastError();
}

int lux_cuda_div_f32(void* output, const void* a, const void* b, size_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    lux_div_f32_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)output, (const float*)a, (const float*)b, n
    );
    return cudaGetLastError();
}

int lux_cuda_matmul_f32(void* c, const void* a, const void* b, int M, int K, int N, cudaStream_t stream) {
    if (M == 0 || K == 0 || N == 0) return 0;

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    lux_matmul_tiled_f32_kernel<<<grid, block, 0, stream>>>(
        (float*)c, (const float*)a, (const float*)b, M, K, N
    );
    return cudaGetLastError();
}

int lux_cuda_transpose_f32(void* output, const void* input, int rows, int cols, cudaStream_t stream) {
    if (rows == 0 || cols == 0) return 0;

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);

    lux_transpose_f32_kernel<<<grid, block, 0, stream>>>(
        (float*)output, (const float*)input, rows, cols
    );
    return cudaGetLastError();
}

// Unary operations
int lux_cuda_exp_f32(void* out, const void* in, size_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    lux_exp_f32_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>((float*)out, (const float*)in, n);
    return cudaGetLastError();
}

int lux_cuda_log_f32(void* out, const void* in, size_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    lux_log_f32_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>((float*)out, (const float*)in, n);
    return cudaGetLastError();
}

int lux_cuda_sqrt_f32(void* out, const void* in, size_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    lux_sqrt_f32_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>((float*)out, (const float*)in, n);
    return cudaGetLastError();
}

int lux_cuda_neg_f32(void* out, const void* in, size_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    lux_neg_f32_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>((float*)out, (const float*)in, n);
    return cudaGetLastError();
}

int lux_cuda_abs_f32(void* out, const void* in, size_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    lux_abs_f32_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>((float*)out, (const float*)in, n);
    return cudaGetLastError();
}

int lux_cuda_tanh_f32(void* out, const void* in, size_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    lux_tanh_f32_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>((float*)out, (const float*)in, n);
    return cudaGetLastError();
}

int lux_cuda_sigmoid_f32(void* out, const void* in, size_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    lux_sigmoid_f32_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>((float*)out, (const float*)in, n);
    return cudaGetLastError();
}

int lux_cuda_relu_f32(void* out, const void* in, size_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    lux_relu_f32_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>((float*)out, (const float*)in, n);
    return cudaGetLastError();
}

int lux_cuda_gelu_f32(void* out, const void* in, size_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    lux_gelu_f32_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>((float*)out, (const float*)in, n);
    return cudaGetLastError();
}

int lux_cuda_copy_f32(void* dst, const void* src, size_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    return cudaMemcpyAsync(dst, src, n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
}

int lux_cuda_layer_norm_f32(
    void* output, const void* input, const void* gamma, const void* beta,
    uint32_t batch_size, uint32_t dim, float eps, cudaStream_t stream
) {
    if (batch_size == 0 || dim == 0) return 0;
    uint32_t threads = BLOCK_SIZE;
    lux_layer_norm_f32_kernel<<<batch_size, threads, 0, stream>>>(
        (float*)output, (const float*)input, (const float*)gamma, (const float*)beta,
        batch_size, dim, eps
    );
    return cudaGetLastError();
}

int lux_cuda_rms_norm_f32(
    void* output, const void* input, const void* weight,
    uint32_t batch_size, uint32_t dim, float eps, cudaStream_t stream
) {
    if (batch_size == 0 || dim == 0) return 0;
    uint32_t threads = BLOCK_SIZE;
    lux_rms_norm_f32_kernel<<<batch_size, threads, 0, stream>>>(
        (float*)output, (const float*)input, (const float*)weight,
        batch_size, dim, eps
    );
    return cudaGetLastError();
}

}  // extern "C"
