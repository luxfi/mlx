// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Log-Sum-Exp CUDA Kernels
// Numerically stable computation of log(sum(exp(x)))
// Uses the identity: logsumexp(x) = max(x) + log(sum(exp(x - max(x))))

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// ============================================================================
// Configuration
// ============================================================================

#define WARP_SIZE 32
#define BLOCK_SIZE 256

// ============================================================================
// Utility: Warp Reductions
// ============================================================================

__device__ __forceinline__
float warp_reduce_max_f32(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__
float warp_reduce_sum_f32(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// LogSumExp over last dimension (most common case)
// Input: [batch_size, dim], Output: [batch_size]
// ============================================================================

extern "C" __global__
void lux_logsumexp_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint32_t batch_size,
    uint32_t dim
) {
    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ float warp_max[WARP_SIZE];
    __shared__ float warp_sum[WARP_SIZE];

    uint32_t batch_idx = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const float* x = input + batch_idx * dim;

    // Step 1: Find maximum
    float local_max = -INFINITY;
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }

    // Warp reduction for max
    local_max = warp_reduce_max_f32(local_max);

    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;

    if (lane == 0) {
        warp_max[wid] = local_max;
    }
    __syncthreads();

    // Final max reduction by first warp
    if (wid == 0) {
        local_max = (lane < blockDim.x / WARP_SIZE) ? warp_max[lane] : -INFINITY;
        local_max = warp_reduce_max_f32(local_max);

        if (lane == 0) {
            s_max = local_max;
        }
    }
    __syncthreads();

    float max_val = s_max;

    // Step 2: Compute sum of exp(x - max)
    float local_sum = 0.0f;
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        local_sum += expf(x[i] - max_val);
    }

    // Warp reduction for sum
    local_sum = warp_reduce_sum_f32(local_sum);

    if (lane == 0) {
        warp_sum[wid] = local_sum;
    }
    __syncthreads();

    if (wid == 0) {
        local_sum = (lane < blockDim.x / WARP_SIZE) ? warp_sum[lane] : 0.0f;
        local_sum = warp_reduce_sum_f32(local_sum);

        if (lane == 0) {
            s_sum = local_sum;
        }
    }
    __syncthreads();

    // Step 3: Compute result
    if (tid == 0) {
        output[batch_idx] = max_val + logf(s_sum);
    }
}

// ============================================================================
// LogSumExp with keepdim (output same shape as input but reduced)
// ============================================================================

extern "C" __global__
void lux_logsumexp_keepdim_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint32_t outer_size,    // Product of dimensions before reduce_dim
    uint32_t reduce_size,   // Size of dimension being reduced
    uint32_t inner_size     // Product of dimensions after reduce_dim
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_outputs = outer_size * inner_size;

    if (idx >= total_outputs) return;

    uint32_t outer_idx = idx / inner_size;
    uint32_t inner_idx = idx % inner_size;

    // Compute offset into input
    uint32_t base_offset = outer_idx * reduce_size * inner_size + inner_idx;

    // Find max
    float max_val = -INFINITY;
    for (uint32_t i = 0; i < reduce_size; i++) {
        float val = input[base_offset + i * inner_size];
        max_val = fmaxf(max_val, val);
    }

    // Compute sum of exp(x - max)
    float sum = 0.0f;
    for (uint32_t i = 0; i < reduce_size; i++) {
        float val = input[base_offset + i * inner_size];
        sum += expf(val - max_val);
    }

    output[idx] = max_val + logf(sum);
}

// ============================================================================
// LogSumExp FP16
// ============================================================================

extern "C" __global__
void lux_logsumexp_f16_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    uint32_t batch_size,
    uint32_t dim
) {
    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ float warp_max[WARP_SIZE];
    __shared__ float warp_sum[WARP_SIZE];

    uint32_t batch_idx = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const __half* x = input + batch_idx * dim;

    // Compute in FP32 for numerical stability
    float local_max = -INFINITY;
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        local_max = fmaxf(local_max, __half2float(x[i]));
    }

    local_max = warp_reduce_max_f32(local_max);

    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;

    if (lane == 0) {
        warp_max[wid] = local_max;
    }
    __syncthreads();

    if (wid == 0) {
        local_max = (lane < blockDim.x / WARP_SIZE) ? warp_max[lane] : -INFINITY;
        local_max = warp_reduce_max_f32(local_max);
        if (lane == 0) s_max = local_max;
    }
    __syncthreads();

    float max_val = s_max;

    float local_sum = 0.0f;
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        local_sum += expf(__half2float(x[i]) - max_val);
    }

    local_sum = warp_reduce_sum_f32(local_sum);

    if (lane == 0) {
        warp_sum[wid] = local_sum;
    }
    __syncthreads();

    if (wid == 0) {
        local_sum = (lane < blockDim.x / WARP_SIZE) ? warp_sum[lane] : 0.0f;
        local_sum = warp_reduce_sum_f32(local_sum);
        if (lane == 0) s_sum = local_sum;
    }
    __syncthreads();

    if (tid == 0) {
        output[batch_idx] = __float2half(max_val + logf(s_sum));
    }
}

// ============================================================================
// Online LogSumExp (Streaming/Incremental)
// Useful for attention mechanisms
// ============================================================================

extern "C" __global__
void lux_logsumexp_online_f32_kernel(
    float* __restrict__ output,       // [batch_size]
    float* __restrict__ max_out,      // [batch_size] - running max
    float* __restrict__ sum_out,      // [batch_size] - running sum
    const float* __restrict__ input,  // [batch_size, chunk_size]
    uint32_t batch_size,
    uint32_t chunk_size,
    bool is_first_chunk
) {
    __shared__ float s_max;
    __shared__ float s_sum;

    uint32_t batch_idx = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const float* x = input + batch_idx * chunk_size;

    // Load previous state
    float prev_max = is_first_chunk ? -INFINITY : max_out[batch_idx];
    float prev_sum = is_first_chunk ? 0.0f : sum_out[batch_idx];

    // Find max in this chunk
    float local_max = -INFINITY;
    for (uint32_t i = tid; i < chunk_size; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }

    local_max = warp_reduce_max_f32(local_max);

    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    __shared__ float warp_vals[WARP_SIZE];

    if (lane == 0) warp_vals[wid] = local_max;
    __syncthreads();

    if (wid == 0) {
        local_max = (lane < blockDim.x / WARP_SIZE) ? warp_vals[lane] : -INFINITY;
        local_max = warp_reduce_max_f32(local_max);
        if (lane == 0) s_max = local_max;
    }
    __syncthreads();

    float chunk_max = s_max;
    float new_max = fmaxf(prev_max, chunk_max);

    // Compute sum with rescaling
    float local_sum = 0.0f;
    for (uint32_t i = tid; i < chunk_size; i += blockDim.x) {
        local_sum += expf(x[i] - new_max);
    }

    local_sum = warp_reduce_sum_f32(local_sum);

    if (lane == 0) warp_vals[wid] = local_sum;
    __syncthreads();

    if (wid == 0) {
        local_sum = (lane < blockDim.x / WARP_SIZE) ? warp_vals[lane] : 0.0f;
        local_sum = warp_reduce_sum_f32(local_sum);
        if (lane == 0) s_sum = local_sum;
    }
    __syncthreads();

    if (tid == 0) {
        // Rescale previous sum and add new sum
        float rescaled_prev = prev_sum * expf(prev_max - new_max);
        float new_sum = rescaled_prev + s_sum;

        max_out[batch_idx] = new_max;
        sum_out[batch_idx] = new_sum;
        output[batch_idx] = new_max + logf(new_sum);
    }
}

// ============================================================================
// Softmax using LogSumExp (Fused for efficiency)
// ============================================================================

extern "C" __global__
void lux_softmax_via_logsumexp_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint32_t batch_size,
    uint32_t dim
) {
    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ float warp_max[WARP_SIZE];
    __shared__ float warp_sum[WARP_SIZE];

    uint32_t batch_idx = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const float* x = input + batch_idx * dim;
    float* y = output + batch_idx * dim;

    // Find max
    float local_max = -INFINITY;
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }

    local_max = warp_reduce_max_f32(local_max);

    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;

    if (lane == 0) warp_max[wid] = local_max;
    __syncthreads();

    if (wid == 0) {
        local_max = (lane < blockDim.x / WARP_SIZE) ? warp_max[lane] : -INFINITY;
        local_max = warp_reduce_max_f32(local_max);
        if (lane == 0) s_max = local_max;
    }
    __syncthreads();

    float max_val = s_max;

    // Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        float exp_val = expf(x[i] - max_val);
        y[i] = exp_val;  // Store intermediate
        local_sum += exp_val;
    }

    local_sum = warp_reduce_sum_f32(local_sum);

    if (lane == 0) warp_sum[wid] = local_sum;
    __syncthreads();

    if (wid == 0) {
        local_sum = (lane < blockDim.x / WARP_SIZE) ? warp_sum[lane] : 0.0f;
        local_sum = warp_reduce_sum_f32(local_sum);
        if (lane == 0) s_sum = local_sum;
    }
    __syncthreads();

    float inv_sum = 1.0f / s_sum;

    // Normalize
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        y[i] *= inv_sum;
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_logsumexp_f32(
    void* output,
    const void* input,
    uint32_t batch_size,
    uint32_t dim,
    cudaStream_t stream
) {
    uint32_t threads = min(BLOCK_SIZE, dim);

    lux_logsumexp_f32_kernel<<<batch_size, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        batch_size,
        dim
    );

    return cudaGetLastError();
}

int lux_cuda_logsumexp_f16(
    void* output,
    const void* input,
    uint32_t batch_size,
    uint32_t dim,
    cudaStream_t stream
) {
    uint32_t threads = min(BLOCK_SIZE, dim);

    lux_logsumexp_f16_kernel<<<batch_size, threads, 0, stream>>>(
        (__half*)output,
        (const __half*)input,
        batch_size,
        dim
    );

    return cudaGetLastError();
}

int lux_cuda_logsumexp_keepdim_f32(
    void* output,
    const void* input,
    uint32_t outer_size,
    uint32_t reduce_size,
    uint32_t inner_size,
    cudaStream_t stream
) {
    uint32_t total_outputs = outer_size * inner_size;
    uint32_t threads = BLOCK_SIZE;
    uint32_t blocks = (total_outputs + threads - 1) / threads;

    lux_logsumexp_keepdim_f32_kernel<<<blocks, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        outer_size,
        reduce_size,
        inner_size
    );

    return cudaGetLastError();
}

int lux_cuda_logsumexp_online_f32(
    void* output,
    void* max_out,
    void* sum_out,
    const void* input,
    uint32_t batch_size,
    uint32_t chunk_size,
    bool is_first_chunk,
    cudaStream_t stream
) {
    uint32_t threads = min(BLOCK_SIZE, chunk_size);

    lux_logsumexp_online_f32_kernel<<<batch_size, threads, 0, stream>>>(
        (float*)output,
        (float*)max_out,
        (float*)sum_out,
        (const float*)input,
        batch_size,
        chunk_size,
        is_first_chunk
    );

    return cudaGetLastError();
}

int lux_cuda_softmax_via_logsumexp_f32(
    void* output,
    const void* input,
    uint32_t batch_size,
    uint32_t dim,
    cudaStream_t stream
) {
    uint32_t threads = min(BLOCK_SIZE, dim);

    lux_softmax_via_logsumexp_f32_kernel<<<batch_size, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        batch_size,
        dim
    );

    return cudaGetLastError();
}

}  // extern "C"
