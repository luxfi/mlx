// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// RMS Normalization CUDA Kernels
// Implements: y = x * rsqrt(mean(x^2) + eps) * gamma
// Used in LLaMA, PaLM, and other modern transformer architectures

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
// RMS Normalization Forward (FP32)
// ============================================================================

extern "C" __global__
void lux_rms_norm_f32_kernel(
    float* __restrict__ output,
    float* __restrict__ rstd_out,      // Optional: save rstd for backward
    const float* __restrict__ input,
    const float* __restrict__ gamma,   // Scale parameter [hidden_size]
    float eps,
    uint32_t batch_size,
    uint32_t hidden_size
) {
    __shared__ float s_rstd;

    uint32_t batch_idx = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const float* x = input + batch_idx * hidden_size;
    float* y = output + batch_idx * hidden_size;

    // Step 1: Compute sum of squares
    float sum_sq = 0.0f;

    for (uint32_t i = tid; i < hidden_size; i += blockDim.x) {
        float val = x[i];
        sum_sq += val * val;
    }

    // Block reduction
    sum_sq = block_reduce_sum_f32(sum_sq);

    if (tid == 0) {
        float mean_sq = sum_sq / (float)hidden_size;
        s_rstd = rsqrtf(mean_sq + eps);

        if (rstd_out != nullptr) {
            rstd_out[batch_idx] = s_rstd;
        }
    }
    __syncthreads();

    float rstd = s_rstd;

    // Step 2: Normalize and apply scale
    for (uint32_t i = tid; i < hidden_size; i += blockDim.x) {
        float normalized = x[i] * rstd;

        if (gamma != nullptr) {
            y[i] = normalized * gamma[i];
        } else {
            y[i] = normalized;
        }
    }
}

// ============================================================================
// RMS Normalization Forward (FP16)
// ============================================================================

extern "C" __global__
void lux_rms_norm_f16_kernel(
    __half* __restrict__ output,
    float* __restrict__ rstd_out,
    const __half* __restrict__ input,
    const __half* __restrict__ gamma,
    float eps,
    uint32_t batch_size,
    uint32_t hidden_size
) {
    __shared__ float s_rstd;

    uint32_t batch_idx = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const __half* x = input + batch_idx * hidden_size;
    __half* y = output + batch_idx * hidden_size;

    // Compute in FP32 for numerical stability
    float sum_sq = 0.0f;

    // Use vectorized loads when possible
    for (uint32_t i = tid; i < hidden_size; i += blockDim.x) {
        float val = __half2float(x[i]);
        sum_sq += val * val;
    }

    sum_sq = block_reduce_sum_f32(sum_sq);

    if (tid == 0) {
        float mean_sq = sum_sq / (float)hidden_size;
        s_rstd = rsqrtf(mean_sq + eps);

        if (rstd_out != nullptr) {
            rstd_out[batch_idx] = s_rstd;
        }
    }
    __syncthreads();

    float rstd = s_rstd;

    for (uint32_t i = tid; i < hidden_size; i += blockDim.x) {
        float normalized = __half2float(x[i]) * rstd;

        if (gamma != nullptr) {
            y[i] = __float2half(normalized * __half2float(gamma[i]));
        } else {
            y[i] = __float2half(normalized);
        }
    }
}

// ============================================================================
// RMS Normalization with Residual Add (Fused)
// output = RMSNorm(input + residual) * gamma
// ============================================================================

extern "C" __global__
void lux_rms_norm_residual_f32_kernel(
    float* __restrict__ output,
    float* __restrict__ residual_out,  // Updated residual for next layer
    const float* __restrict__ input,
    const float* __restrict__ residual,
    const float* __restrict__ gamma,
    float eps,
    uint32_t batch_size,
    uint32_t hidden_size
) {
    __shared__ float s_rstd;

    uint32_t batch_idx = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const float* x = input + batch_idx * hidden_size;
    const float* r = residual + batch_idx * hidden_size;
    float* y = output + batch_idx * hidden_size;
    float* r_out = residual_out + batch_idx * hidden_size;

    // Step 1: Add residual and compute sum of squares
    float sum_sq = 0.0f;

    for (uint32_t i = tid; i < hidden_size; i += blockDim.x) {
        float val = x[i] + r[i];
        r_out[i] = val;  // Store updated residual
        sum_sq += val * val;
    }

    sum_sq = block_reduce_sum_f32(sum_sq);

    if (tid == 0) {
        float mean_sq = sum_sq / (float)hidden_size;
        s_rstd = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    float rstd = s_rstd;

    // Step 2: Normalize
    for (uint32_t i = tid; i < hidden_size; i += blockDim.x) {
        float normalized = r_out[i] * rstd;

        if (gamma != nullptr) {
            y[i] = normalized * gamma[i];
        } else {
            y[i] = normalized;
        }
    }
}

// ============================================================================
// RMS Normalization Backward
// ============================================================================

extern "C" __global__
void lux_rms_norm_backward_f32_kernel(
    float* __restrict__ dx,
    float* __restrict__ dgamma,
    const float* __restrict__ dy,
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ rstd,
    uint32_t batch_size,
    uint32_t hidden_size
) {
    __shared__ float s_c1;  // sum(dy * x * gamma)

    uint32_t batch_idx = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const float* dy_row = dy + batch_idx * hidden_size;
    const float* x_row = x + batch_idx * hidden_size;
    float* dx_row = dx + batch_idx * hidden_size;

    float r = rstd[batch_idx];

    // Compute sum for gradient
    float c1_local = 0.0f;

    for (uint32_t i = tid; i < hidden_size; i += blockDim.x) {
        float g = (gamma != nullptr) ? gamma[i] : 1.0f;
        float dy_val = dy_row[i] * g;
        c1_local += dy_val * x_row[i];

        // Accumulate dgamma
        if (dgamma != nullptr) {
            atomicAdd(&dgamma[i], dy_row[i] * x_row[i] * r);
        }
    }

    c1_local = block_reduce_sum_f32(c1_local);

    if (tid == 0) {
        s_c1 = c1_local * r * r / (float)hidden_size;
    }
    __syncthreads();

    float c1 = s_c1;

    // Compute dx
    for (uint32_t i = tid; i < hidden_size; i += blockDim.x) {
        float g = (gamma != nullptr) ? gamma[i] : 1.0f;
        float dy_val = dy_row[i] * g;

        dx_row[i] = r * (dy_val - x_row[i] * c1);
    }
}

// ============================================================================
// Vectorized RMS Norm (using float4 for coalesced access)
// ============================================================================

extern "C" __global__
void lux_rms_norm_vectorized_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    float eps,
    uint32_t batch_size,
    uint32_t hidden_size  // Must be divisible by 4
) {
    __shared__ float s_rstd;

    uint32_t batch_idx = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    uint32_t vec_size = hidden_size / 4;
    const float4* x = (const float4*)(input + batch_idx * hidden_size);
    float4* y = (float4*)(output + batch_idx * hidden_size);
    const float4* g = (const float4*)gamma;

    float sum_sq = 0.0f;

    for (uint32_t i = tid; i < vec_size; i += blockDim.x) {
        float4 v = x[i];
        sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    sum_sq = block_reduce_sum_f32(sum_sq);

    if (tid == 0) {
        float mean_sq = sum_sq / (float)hidden_size;
        s_rstd = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    float rstd = s_rstd;

    for (uint32_t i = tid; i < vec_size; i += blockDim.x) {
        float4 v = x[i];
        float4 result;

        if (gamma != nullptr) {
            float4 gv = g[i];
            result.x = v.x * rstd * gv.x;
            result.y = v.y * rstd * gv.y;
            result.z = v.z * rstd * gv.z;
            result.w = v.w * rstd * gv.w;
        } else {
            result.x = v.x * rstd;
            result.y = v.y * rstd;
            result.z = v.z * rstd;
            result.w = v.w * rstd;
        }

        y[i] = result;
    }
}

// ============================================================================
// Batched RMS Norm (Multiple rows per block for small hidden_size)
// ============================================================================

extern "C" __global__
void lux_rms_norm_batched_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ gamma,
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

        // Compute sum of squares
        float sum_sq = 0.0f;
        for (uint32_t i = tid; i < hidden_size; i += blockDim.x) {
            sum_sq += x[i] * x[i];
        }

        sum_sq = block_reduce_sum_f32(sum_sq);

        __shared__ float s_rstd;
        if (tid == 0) {
            s_rstd = rsqrtf(sum_sq / (float)hidden_size + eps);
        }
        __syncthreads();

        float rstd = s_rstd;

        // Normalize
        for (uint32_t i = tid; i < hidden_size; i += blockDim.x) {
            float normalized = x[i] * rstd;
            if (gamma != nullptr) {
                y[i] = normalized * gamma[i];
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

int lux_cuda_rms_norm_f32(
    void* output,
    void* rstd_out,
    const void* input,
    const void* gamma,
    float eps,
    uint32_t batch_size,
    uint32_t hidden_size,
    cudaStream_t stream
) {
    uint32_t threads = min(BLOCK_SIZE, hidden_size);

    // Use vectorized kernel for large hidden sizes divisible by 4
    if (hidden_size >= 256 && hidden_size % 4 == 0 && rstd_out == nullptr) {
        lux_rms_norm_vectorized_f32_kernel<<<batch_size, threads, 0, stream>>>(
            (float*)output,
            (const float*)input,
            (const float*)gamma,
            eps,
            batch_size,
            hidden_size
        );
    } else {
        lux_rms_norm_f32_kernel<<<batch_size, threads, 0, stream>>>(
            (float*)output,
            (float*)rstd_out,
            (const float*)input,
            (const float*)gamma,
            eps,
            batch_size,
            hidden_size
        );
    }

    return cudaGetLastError();
}

int lux_cuda_rms_norm_f16(
    void* output,
    void* rstd_out,
    const void* input,
    const void* gamma,
    float eps,
    uint32_t batch_size,
    uint32_t hidden_size,
    cudaStream_t stream
) {
    uint32_t threads = min(BLOCK_SIZE, hidden_size);

    lux_rms_norm_f16_kernel<<<batch_size, threads, 0, stream>>>(
        (__half*)output,
        (float*)rstd_out,
        (const __half*)input,
        (const __half*)gamma,
        eps,
        batch_size,
        hidden_size
    );

    return cudaGetLastError();
}

int lux_cuda_rms_norm_residual_f32(
    void* output,
    void* residual_out,
    const void* input,
    const void* residual,
    const void* gamma,
    float eps,
    uint32_t batch_size,
    uint32_t hidden_size,
    cudaStream_t stream
) {
    uint32_t threads = min(BLOCK_SIZE, hidden_size);

    lux_rms_norm_residual_f32_kernel<<<batch_size, threads, 0, stream>>>(
        (float*)output,
        (float*)residual_out,
        (const float*)input,
        (const float*)residual,
        (const float*)gamma,
        eps,
        batch_size,
        hidden_size
    );

    return cudaGetLastError();
}

int lux_cuda_rms_norm_backward_f32(
    void* dx,
    void* dgamma,
    const void* dy,
    const void* x,
    const void* gamma,
    const void* rstd,
    uint32_t batch_size,
    uint32_t hidden_size,
    cudaStream_t stream
) {
    if (dgamma != nullptr) {
        cudaMemsetAsync(dgamma, 0, hidden_size * sizeof(float), stream);
    }

    uint32_t threads = min(BLOCK_SIZE, hidden_size);

    lux_rms_norm_backward_f32_kernel<<<batch_size, threads, 0, stream>>>(
        (float*)dx,
        (float*)dgamma,
        (const float*)dy,
        (const float*)x,
        (const float*)gamma,
        (const float*)rstd,
        batch_size,
        hidden_size
    );

    return cudaGetLastError();
}

}  // extern "C"
