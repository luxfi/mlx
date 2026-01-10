// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Softmax CUDA Kernels
// Numerically stable softmax: softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
// Implements online softmax algorithm for single-pass computation

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
// Online Softmax State (for single-pass algorithm)
// ============================================================================

struct OnlineSoftmaxState {
    float max_val;
    float sum;
};

__device__ __forceinline__
OnlineSoftmaxState online_softmax_init() {
    return {-INFINITY, 0.0f};
}

__device__ __forceinline__
OnlineSoftmaxState online_softmax_update(OnlineSoftmaxState state, float x) {
    float new_max = fmaxf(state.max_val, x);
    float new_sum = state.sum * expf(state.max_val - new_max) + expf(x - new_max);
    return {new_max, new_sum};
}

__device__ __forceinline__
OnlineSoftmaxState online_softmax_merge(OnlineSoftmaxState a, OnlineSoftmaxState b) {
    float new_max = fmaxf(a.max_val, b.max_val);
    float new_sum = a.sum * expf(a.max_val - new_max) + b.sum * expf(b.max_val - new_max);
    return {new_max, new_sum};
}

__device__ __forceinline__
OnlineSoftmaxState warp_reduce_online_softmax(OnlineSoftmaxState state) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        OnlineSoftmaxState other;
        other.max_val = __shfl_down_sync(0xffffffff, state.max_val, offset);
        other.sum = __shfl_down_sync(0xffffffff, state.sum, offset);
        state = online_softmax_merge(state, other);
    }
    return state;
}

// ============================================================================
// Softmax Forward (FP32) - Two-pass algorithm
// Input/Output: [batch_size, dim]
// ============================================================================

extern "C" __global__
void lux_softmax_f32_kernel(
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

    // Pass 1: Find maximum
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

    if (wid == 0) {
        local_max = (lane < blockDim.x / WARP_SIZE) ? warp_max[lane] : -INFINITY;
        local_max = warp_reduce_max_f32(local_max);
        if (lane == 0) s_max = local_max;
    }
    __syncthreads();

    float max_val = s_max;

    // Pass 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        float exp_val = expf(x[i] - max_val);
        y[i] = exp_val;  // Store intermediate
        local_sum += exp_val;
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
// Softmax Forward (FP32) - Online algorithm (single pass for reading)
// More efficient for large dim when memory bandwidth is the bottleneck
// ============================================================================

extern "C" __global__
void lux_softmax_online_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint32_t batch_size,
    uint32_t dim
) {
    __shared__ OnlineSoftmaxState warp_states[WARP_SIZE];
    __shared__ float s_max;
    __shared__ float s_sum;

    uint32_t batch_idx = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const float* x = input + batch_idx * dim;
    float* y = output + batch_idx * dim;

    // Online: Compute max and sum in single pass
    OnlineSoftmaxState local_state = online_softmax_init();
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        local_state = online_softmax_update(local_state, x[i]);
    }

    // Warp reduction
    local_state = warp_reduce_online_softmax(local_state);

    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;

    if (lane == 0) {
        warp_states[wid] = local_state;
    }
    __syncthreads();

    // Final reduction by first warp
    if (wid == 0) {
        local_state = (lane < blockDim.x / WARP_SIZE) ? warp_states[lane] : online_softmax_init();
        local_state = warp_reduce_online_softmax(local_state);

        if (lane == 0) {
            s_max = local_state.max_val;
            s_sum = local_state.sum;
        }
    }
    __syncthreads();

    float max_val = s_max;
    float inv_sum = 1.0f / s_sum;

    // Second pass: Write normalized values
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        y[i] = expf(x[i] - max_val) * inv_sum;
    }
}

// ============================================================================
// Softmax Forward (FP16)
// ============================================================================

extern "C" __global__
void lux_softmax_f16_kernel(
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
    __half* y = output + batch_idx * dim;

    // Compute in FP32 for numerical stability
    float local_max = -INFINITY;
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        local_max = fmaxf(local_max, __half2float(x[i]));
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

    float local_sum = 0.0f;
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        local_sum += expf(__half2float(x[i]) - max_val);
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

    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        y[i] = __float2half(expf(__half2float(x[i]) - max_val) * inv_sum);
    }
}

// ============================================================================
// Softmax In-Place
// ============================================================================

extern "C" __global__
void lux_softmax_inplace_f32_kernel(
    float* __restrict__ data,
    uint32_t batch_size,
    uint32_t dim
) {
    __shared__ float s_max;
    __shared__ float s_sum;

    uint32_t batch_idx = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    float* x = data + batch_idx * dim;

    // Find max
    float local_max = -INFINITY;
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }

    local_max = warp_reduce_max_f32(local_max);

    int lane = tid % WARP_SIZE;
    __shared__ float warp_vals[WARP_SIZE];

    if (lane == 0) warp_vals[tid / WARP_SIZE] = local_max;
    __syncthreads();

    if (tid / WARP_SIZE == 0) {
        local_max = (lane < blockDim.x / WARP_SIZE) ? warp_vals[lane] : -INFINITY;
        local_max = warp_reduce_max_f32(local_max);
        if (lane == 0) s_max = local_max;
    }
    __syncthreads();

    float max_val = s_max;

    // Compute exp and sum
    float local_sum = 0.0f;
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        float exp_val = expf(x[i] - max_val);
        x[i] = exp_val;
        local_sum += exp_val;
    }

    local_sum = warp_reduce_sum_f32(local_sum);

    if (lane == 0) warp_vals[tid / WARP_SIZE] = local_sum;
    __syncthreads();

    if (tid / WARP_SIZE == 0) {
        local_sum = (lane < blockDim.x / WARP_SIZE) ? warp_vals[lane] : 0.0f;
        local_sum = warp_reduce_sum_f32(local_sum);
        if (lane == 0) s_sum = local_sum;
    }
    __syncthreads();

    float inv_sum = 1.0f / s_sum;

    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        x[i] *= inv_sum;
    }
}

// ============================================================================
// Log Softmax (numerically stable)
// log_softmax(x)_i = x_i - max(x) - log(sum(exp(x - max(x))))
// ============================================================================

extern "C" __global__
void lux_log_softmax_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint32_t batch_size,
    uint32_t dim
) {
    __shared__ float s_max;
    __shared__ float s_log_sum;
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

    // Compute sum
    float local_sum = 0.0f;
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        local_sum += expf(x[i] - max_val);
    }

    local_sum = warp_reduce_sum_f32(local_sum);

    if (lane == 0) warp_sum[wid] = local_sum;
    __syncthreads();

    if (wid == 0) {
        local_sum = (lane < blockDim.x / WARP_SIZE) ? warp_sum[lane] : 0.0f;
        local_sum = warp_reduce_sum_f32(local_sum);
        if (lane == 0) s_log_sum = logf(local_sum);
    }
    __syncthreads();

    float log_sum = s_log_sum;

    // Compute log softmax
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        y[i] = x[i] - max_val - log_sum;
    }
}

// ============================================================================
// Softmax with Temperature
// ============================================================================

extern "C" __global__
void lux_softmax_temperature_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    float temperature,
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

    float inv_temp = 1.0f / temperature;

    // Find max (after temperature scaling)
    float local_max = -INFINITY;
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i] * inv_temp);
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

    float local_sum = 0.0f;
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        float exp_val = expf(x[i] * inv_temp - max_val);
        y[i] = exp_val;
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

    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        y[i] *= inv_sum;
    }
}

// ============================================================================
// Softmax Backward
// dy/dx_i = softmax_i * (delta_ij - softmax_j) * dy_j
// Simplified: dx = softmax * (dy - sum(softmax * dy))
// ============================================================================

extern "C" __global__
void lux_softmax_backward_f32_kernel(
    float* __restrict__ dx,
    const float* __restrict__ dy,
    const float* __restrict__ softmax_output,
    uint32_t batch_size,
    uint32_t dim
) {
    __shared__ float s_dot;
    __shared__ float warp_sum[WARP_SIZE];

    uint32_t batch_idx = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const float* p = softmax_output + batch_idx * dim;  // softmax output
    const float* dy_row = dy + batch_idx * dim;
    float* dx_row = dx + batch_idx * dim;

    // Compute dot product: sum(p * dy)
    float local_dot = 0.0f;
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        local_dot += p[i] * dy_row[i];
    }

    local_dot = warp_reduce_sum_f32(local_dot);

    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;

    if (lane == 0) warp_sum[wid] = local_dot;
    __syncthreads();

    if (wid == 0) {
        local_dot = (lane < blockDim.x / WARP_SIZE) ? warp_sum[lane] : 0.0f;
        local_dot = warp_reduce_sum_f32(local_dot);
        if (lane == 0) s_dot = local_dot;
    }
    __syncthreads();

    float dot = s_dot;

    // dx = p * (dy - dot)
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        dx_row[i] = p[i] * (dy_row[i] - dot);
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_softmax_f32(
    void* output,
    const void* input,
    uint32_t batch_size,
    uint32_t dim,
    cudaStream_t stream
) {
    uint32_t threads = min(BLOCK_SIZE, dim);

    lux_softmax_f32_kernel<<<batch_size, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        batch_size,
        dim
    );

    return cudaGetLastError();
}

int lux_cuda_softmax_online_f32(
    void* output,
    const void* input,
    uint32_t batch_size,
    uint32_t dim,
    cudaStream_t stream
) {
    uint32_t threads = min(BLOCK_SIZE, dim);

    lux_softmax_online_f32_kernel<<<batch_size, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        batch_size,
        dim
    );

    return cudaGetLastError();
}

int lux_cuda_softmax_f16(
    void* output,
    const void* input,
    uint32_t batch_size,
    uint32_t dim,
    cudaStream_t stream
) {
    uint32_t threads = min(BLOCK_SIZE, dim);

    lux_softmax_f16_kernel<<<batch_size, threads, 0, stream>>>(
        (__half*)output,
        (const __half*)input,
        batch_size,
        dim
    );

    return cudaGetLastError();
}

int lux_cuda_softmax_inplace_f32(
    void* data,
    uint32_t batch_size,
    uint32_t dim,
    cudaStream_t stream
) {
    uint32_t threads = min(BLOCK_SIZE, dim);

    lux_softmax_inplace_f32_kernel<<<batch_size, threads, 0, stream>>>(
        (float*)data,
        batch_size,
        dim
    );

    return cudaGetLastError();
}

int lux_cuda_log_softmax_f32(
    void* output,
    const void* input,
    uint32_t batch_size,
    uint32_t dim,
    cudaStream_t stream
) {
    uint32_t threads = min(BLOCK_SIZE, dim);

    lux_log_softmax_f32_kernel<<<batch_size, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        batch_size,
        dim
    );

    return cudaGetLastError();
}

int lux_cuda_softmax_temperature_f32(
    void* output,
    const void* input,
    float temperature,
    uint32_t batch_size,
    uint32_t dim,
    cudaStream_t stream
) {
    uint32_t threads = min(BLOCK_SIZE, dim);

    lux_softmax_temperature_f32_kernel<<<batch_size, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        temperature,
        batch_size,
        dim
    );

    return cudaGetLastError();
}

int lux_cuda_softmax_backward_f32(
    void* dx,
    const void* dy,
    const void* softmax_output,
    uint32_t batch_size,
    uint32_t dim,
    cudaStream_t stream
) {
    uint32_t threads = min(BLOCK_SIZE, dim);

    lux_softmax_backward_f32_kernel<<<batch_size, threads, 0, stream>>>(
        (float*)dx,
        (const float*)dy,
        (const float*)softmax_output,
        batch_size,
        dim
    );

    return cudaGetLastError();
}

}  // extern "C"
