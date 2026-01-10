// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Layer Normalization CUDA Kernels
// Implements: y = (x - mean) / sqrt(var + eps) * gamma + beta
// Optimized with online Welford algorithm for numerical stability

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// ============================================================================
// Configuration
// ============================================================================

#define WARP_SIZE 32
#define BLOCK_SIZE 256

// ============================================================================
// Utility: Welford Online Algorithm for Mean and Variance
// ============================================================================

struct WelfordState {
    float mean;
    float m2;     // Sum of squared differences
    float count;
};

__device__ __forceinline__
WelfordState welford_init() {
    return {0.0f, 0.0f, 0.0f};
}

__device__ __forceinline__
WelfordState welford_update(WelfordState state, float x) {
    state.count += 1.0f;
    float delta = x - state.mean;
    state.mean += delta / state.count;
    float delta2 = x - state.mean;
    state.m2 += delta * delta2;
    return state;
}

__device__ __forceinline__
WelfordState welford_merge(WelfordState a, WelfordState b) {
    if (b.count == 0.0f) return a;
    if (a.count == 0.0f) return b;

    float count = a.count + b.count;
    float delta = b.mean - a.mean;
    float mean = a.mean + delta * b.count / count;
    float m2 = a.m2 + b.m2 + delta * delta * a.count * b.count / count;

    return {mean, m2, count};
}

__device__ __forceinline__
WelfordState warp_reduce_welford(WelfordState state) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        WelfordState other;
        other.mean = __shfl_down_sync(0xffffffff, state.mean, offset);
        other.m2 = __shfl_down_sync(0xffffffff, state.m2, offset);
        other.count = __shfl_down_sync(0xffffffff, state.count, offset);
        state = welford_merge(state, other);
    }
    return state;
}

// ============================================================================
// Layer Normalization Forward (FP32)
// ============================================================================

extern "C" __global__
void lux_layer_norm_f32_kernel(
    float* __restrict__ output,
    float* __restrict__ mean_out,      // Optional: save mean for backward
    float* __restrict__ rstd_out,      // Optional: save rstd for backward
    const float* __restrict__ input,
    const float* __restrict__ gamma,   // Scale parameter [D]
    const float* __restrict__ beta,    // Shift parameter [D]
    float eps,
    uint32_t batch_size,
    uint32_t hidden_size
) {
    __shared__ float s_mean;
    __shared__ float s_rstd;
    __shared__ WelfordState s_welford[WARP_SIZE];

    uint32_t batch_idx = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const float* x = input + batch_idx * hidden_size;
    float* y = output + batch_idx * hidden_size;

    // Step 1: Compute mean and variance using Welford's algorithm
    WelfordState local_state = welford_init();

    for (uint32_t i = tid; i < hidden_size; i += blockDim.x) {
        local_state = welford_update(local_state, x[i]);
    }

    // Warp reduction
    local_state = warp_reduce_welford(local_state);

    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;

    if (lane == 0) {
        s_welford[wid] = local_state;
    }
    __syncthreads();

    // Final reduction by first warp
    if (wid == 0) {
        local_state = (lane < blockDim.x / WARP_SIZE) ? s_welford[lane] : welford_init();
        local_state = warp_reduce_welford(local_state);

        if (lane == 0) {
            s_mean = local_state.mean;
            float variance = local_state.m2 / local_state.count;
            s_rstd = rsqrtf(variance + eps);

            // Optionally save mean and rstd for backward pass
            if (mean_out != nullptr) {
                mean_out[batch_idx] = s_mean;
            }
            if (rstd_out != nullptr) {
                rstd_out[batch_idx] = s_rstd;
            }
        }
    }
    __syncthreads();

    float mean = s_mean;
    float rstd = s_rstd;

    // Step 2: Normalize and apply affine transformation
    for (uint32_t i = tid; i < hidden_size; i += blockDim.x) {
        float normalized = (x[i] - mean) * rstd;

        if (gamma != nullptr && beta != nullptr) {
            y[i] = normalized * gamma[i] + beta[i];
        } else if (gamma != nullptr) {
            y[i] = normalized * gamma[i];
        } else {
            y[i] = normalized;
        }
    }
}

// ============================================================================
// Layer Normalization Forward (FP16)
// ============================================================================

extern "C" __global__
void lux_layer_norm_f16_kernel(
    __half* __restrict__ output,
    float* __restrict__ mean_out,
    float* __restrict__ rstd_out,
    const __half* __restrict__ input,
    const __half* __restrict__ gamma,
    const __half* __restrict__ beta,
    float eps,
    uint32_t batch_size,
    uint32_t hidden_size
) {
    __shared__ float s_mean;
    __shared__ float s_rstd;
    __shared__ WelfordState s_welford[WARP_SIZE];

    uint32_t batch_idx = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const __half* x = input + batch_idx * hidden_size;
    __half* y = output + batch_idx * hidden_size;

    // Compute in FP32 for numerical stability
    WelfordState local_state = welford_init();

    for (uint32_t i = tid; i < hidden_size; i += blockDim.x) {
        local_state = welford_update(local_state, __half2float(x[i]));
    }

    local_state = warp_reduce_welford(local_state);

    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;

    if (lane == 0) {
        s_welford[wid] = local_state;
    }
    __syncthreads();

    if (wid == 0) {
        local_state = (lane < blockDim.x / WARP_SIZE) ? s_welford[lane] : welford_init();
        local_state = warp_reduce_welford(local_state);

        if (lane == 0) {
            s_mean = local_state.mean;
            float variance = local_state.m2 / local_state.count;
            s_rstd = rsqrtf(variance + eps);

            if (mean_out != nullptr) mean_out[batch_idx] = s_mean;
            if (rstd_out != nullptr) rstd_out[batch_idx] = s_rstd;
        }
    }
    __syncthreads();

    float mean = s_mean;
    float rstd = s_rstd;

    for (uint32_t i = tid; i < hidden_size; i += blockDim.x) {
        float normalized = (__half2float(x[i]) - mean) * rstd;

        if (gamma != nullptr && beta != nullptr) {
            y[i] = __float2half(normalized * __half2float(gamma[i]) + __half2float(beta[i]));
        } else if (gamma != nullptr) {
            y[i] = __float2half(normalized * __half2float(gamma[i]));
        } else {
            y[i] = __float2half(normalized);
        }
    }
}

// ============================================================================
// Layer Normalization Backward
// ============================================================================

extern "C" __global__
void lux_layer_norm_backward_f32_kernel(
    float* __restrict__ dx,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    const float* __restrict__ dy,
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    uint32_t batch_size,
    uint32_t hidden_size
) {
    extern __shared__ float smem[];

    uint32_t batch_idx = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    float* s_dgamma = smem;
    float* s_dbeta = smem + hidden_size;

    const float* dy_row = dy + batch_idx * hidden_size;
    const float* x_row = x + batch_idx * hidden_size;
    float* dx_row = dx + batch_idx * hidden_size;

    float m = mean[batch_idx];
    float r = rstd[batch_idx];

    // Compute intermediate values
    float sum_dy = 0.0f;
    float sum_dy_xhat = 0.0f;

    for (uint32_t i = tid; i < hidden_size; i += blockDim.x) {
        float xhat = (x_row[i] - m) * r;
        float dy_val = dy_row[i] * (gamma != nullptr ? gamma[i] : 1.0f);

        sum_dy += dy_val;
        sum_dy_xhat += dy_val * xhat;

        // Accumulate dgamma and dbeta
        if (dgamma != nullptr) {
            atomicAdd(&dgamma[i], dy_row[i] * xhat);
        }
        if (dbeta != nullptr) {
            atomicAdd(&dbeta[i], dy_row[i]);
        }
    }

    // Reduce sums across block
    __shared__ float s_sum_dy;
    __shared__ float s_sum_dy_xhat;

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum_dy += __shfl_down_sync(0xffffffff, sum_dy, offset);
        sum_dy_xhat += __shfl_down_sync(0xffffffff, sum_dy_xhat, offset);
    }

    if (tid % WARP_SIZE == 0) {
        atomicAdd(&s_sum_dy, sum_dy);
        atomicAdd(&s_sum_dy_xhat, sum_dy_xhat);
    }
    __syncthreads();

    float inv_n = 1.0f / (float)hidden_size;

    // Compute dx
    for (uint32_t i = tid; i < hidden_size; i += blockDim.x) {
        float xhat = (x_row[i] - m) * r;
        float dy_val = dy_row[i] * (gamma != nullptr ? gamma[i] : 1.0f);

        dx_row[i] = r * (dy_val - inv_n * (s_sum_dy + xhat * s_sum_dy_xhat));
    }
}

// ============================================================================
// Batched Layer Normalization (Process multiple rows per block)
// ============================================================================

extern "C" __global__
void lux_layer_norm_batched_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float eps,
    uint32_t total_rows,
    uint32_t hidden_size,
    uint32_t rows_per_block
) {
    uint32_t block_row_start = blockIdx.x * rows_per_block;
    uint32_t tid = threadIdx.x;

    for (uint32_t r = 0; r < rows_per_block; r++) {
        uint32_t row_idx = block_row_start + r;
        if (row_idx >= total_rows) break;

        const float* x = input + row_idx * hidden_size;
        float* y = output + row_idx * hidden_size;

        // Compute mean
        float sum = 0.0f;
        for (uint32_t i = tid; i < hidden_size; i += blockDim.x) {
            sum += x[i];
        }

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        __shared__ float s_mean;
        if (tid == 0) {
            s_mean = sum / (float)hidden_size;
        }
        __syncthreads();

        float mean = s_mean;

        // Compute variance
        float var_sum = 0.0f;
        for (uint32_t i = tid; i < hidden_size; i += blockDim.x) {
            float diff = x[i] - mean;
            var_sum += diff * diff;
        }

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
        }

        __shared__ float s_rstd;
        if (tid == 0) {
            s_rstd = rsqrtf(var_sum / (float)hidden_size + eps);
        }
        __syncthreads();

        float rstd = s_rstd;

        // Normalize
        for (uint32_t i = tid; i < hidden_size; i += blockDim.x) {
            float normalized = (x[i] - mean) * rstd;
            if (gamma != nullptr && beta != nullptr) {
                y[i] = normalized * gamma[i] + beta[i];
            } else {
                y[i] = normalized;
            }
        }
        __syncthreads();
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_layer_norm_f32(
    void* output,
    void* mean_out,
    void* rstd_out,
    const void* input,
    const void* gamma,
    const void* beta,
    float eps,
    uint32_t batch_size,
    uint32_t hidden_size,
    cudaStream_t stream
) {
    uint32_t threads = min(BLOCK_SIZE, hidden_size);

    lux_layer_norm_f32_kernel<<<batch_size, threads, 0, stream>>>(
        (float*)output,
        (float*)mean_out,
        (float*)rstd_out,
        (const float*)input,
        (const float*)gamma,
        (const float*)beta,
        eps,
        batch_size,
        hidden_size
    );

    return cudaGetLastError();
}

int lux_cuda_layer_norm_f16(
    void* output,
    void* mean_out,
    void* rstd_out,
    const void* input,
    const void* gamma,
    const void* beta,
    float eps,
    uint32_t batch_size,
    uint32_t hidden_size,
    cudaStream_t stream
) {
    uint32_t threads = min(BLOCK_SIZE, hidden_size);

    lux_layer_norm_f16_kernel<<<batch_size, threads, 0, stream>>>(
        (__half*)output,
        (float*)mean_out,
        (float*)rstd_out,
        (const __half*)input,
        (const __half*)gamma,
        (const __half*)beta,
        eps,
        batch_size,
        hidden_size
    );

    return cudaGetLastError();
}

int lux_cuda_layer_norm_backward_f32(
    void* dx,
    void* dgamma,
    void* dbeta,
    const void* dy,
    const void* x,
    const void* gamma,
    const void* mean,
    const void* rstd,
    uint32_t batch_size,
    uint32_t hidden_size,
    cudaStream_t stream
) {
    // Initialize dgamma and dbeta to zero
    if (dgamma != nullptr) {
        cudaMemsetAsync(dgamma, 0, hidden_size * sizeof(float), stream);
    }
    if (dbeta != nullptr) {
        cudaMemsetAsync(dbeta, 0, hidden_size * sizeof(float), stream);
    }

    uint32_t threads = min(BLOCK_SIZE, hidden_size);
    size_t shared_size = 2 * hidden_size * sizeof(float);

    lux_layer_norm_backward_f32_kernel<<<batch_size, threads, shared_size, stream>>>(
        (float*)dx,
        (float*)dgamma,
        (float*)dbeta,
        (const float*)dy,
        (const float*)x,
        (const float*)gamma,
        (const float*)mean,
        (const float*)rstd,
        batch_size,
        hidden_size
    );

    return cudaGetLastError();
}

}  // extern "C"
