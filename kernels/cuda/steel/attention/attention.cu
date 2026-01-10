// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Steel Attention CUDA Kernels
// High-performance multi-head attention implementation

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// Attention Configuration
// ============================================================================

#define WARP_SIZE 32
#define MAX_SEQ_LEN 8192
#define MAX_HEAD_DIM 128

// ============================================================================
// Standard Multi-Head Attention
// ============================================================================

// Compute Q @ K^T / sqrt(d) -> softmax -> @ V
extern "C" __global__
void steel_attention_forward_kernel(
    float* __restrict__ output,       // [B, H, S, D]
    const float* __restrict__ Q,      // [B, H, S, D]
    const float* __restrict__ K,      // [B, H, S, D]
    const float* __restrict__ V,      // [B, H, S, D]
    const float* __restrict__ mask,   // [B, 1, S, S] or nullptr
    float scale,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim
) {
    extern __shared__ float smem[];

    uint32_t b = blockIdx.z / num_heads;
    uint32_t h = blockIdx.z % num_heads;
    uint32_t q_idx = blockIdx.y;
    uint32_t tid = threadIdx.x;

    if (b >= batch_size || q_idx >= seq_len) return;

    float* scores = smem;
    float* output_row = smem + seq_len;

    // Base pointers for this head
    const float* Q_head = Q + (b * num_heads + h) * seq_len * head_dim;
    const float* K_head = K + (b * num_heads + h) * seq_len * head_dim;
    const float* V_head = V + (b * num_heads + h) * seq_len * head_dim;
    const float* Q_row = Q_head + q_idx * head_dim;

    // Step 1: Compute attention scores (Q @ K^T)
    for (uint32_t k_idx = tid; k_idx < seq_len; k_idx += blockDim.x) {
        const float* K_row = K_head + k_idx * head_dim;
        float dot = 0.0f;

        for (uint32_t d = 0; d < head_dim; d++) {
            dot += Q_row[d] * K_row[d];
        }

        dot *= scale;

        // Apply mask if present
        if (mask != nullptr) {
            float m = mask[b * seq_len * seq_len + q_idx * seq_len + k_idx];
            if (m == 0.0f) {
                dot = -1e9f;  // Large negative for softmax
            }
        }

        scores[k_idx] = dot;
    }
    __syncthreads();

    // Step 2: Softmax
    // Find max
    float max_val = -1e10f;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        max_val = fmaxf(max_val, scores[i]);
    }

    // Warp reduction for max
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }

    __shared__ float block_max;
    if (tid % WARP_SIZE == 0) {
        atomicMax((int*)&block_max, __float_as_int(max_val));
    }
    __syncthreads();
    max_val = block_max;

    // Compute exp and sum
    float sum = 0.0f;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_val);
        sum += scores[i];
    }

    // Warp reduction for sum
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float block_sum;
    if (tid == 0) block_sum = 0.0f;
    __syncthreads();
    if (tid % WARP_SIZE == 0) atomicAdd(&block_sum, sum);
    __syncthreads();

    // Normalize
    float inv_sum = 1.0f / block_sum;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        scores[i] *= inv_sum;
    }
    __syncthreads();

    // Step 3: Compute output (scores @ V)
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;

        for (uint32_t k = 0; k < seq_len; k++) {
            acc += scores[k] * V_head[k * head_dim + d];
        }

        output_row[d] = acc;
    }
    __syncthreads();

    // Write output
    float* out_ptr = output + (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim;
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        out_ptr[d] = output_row[d];
    }
}

// ============================================================================
// FP16 Attention
// ============================================================================

extern "C" __global__
void steel_attention_forward_fp16_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    const __half* __restrict__ mask,
    float scale,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim
) {
    extern __shared__ float smem[];

    uint32_t b = blockIdx.z / num_heads;
    uint32_t h = blockIdx.z % num_heads;
    uint32_t q_idx = blockIdx.y;
    uint32_t tid = threadIdx.x;

    if (b >= batch_size || q_idx >= seq_len) return;

    float* scores = smem;

    const __half* Q_head = Q + (b * num_heads + h) * seq_len * head_dim;
    const __half* K_head = K + (b * num_heads + h) * seq_len * head_dim;
    const __half* V_head = V + (b * num_heads + h) * seq_len * head_dim;
    const __half* Q_row = Q_head + q_idx * head_dim;

    // Compute attention scores
    for (uint32_t k_idx = tid; k_idx < seq_len; k_idx += blockDim.x) {
        const __half* K_row = K_head + k_idx * head_dim;
        float dot = 0.0f;

        for (uint32_t d = 0; d < head_dim; d++) {
            dot += __half2float(Q_row[d]) * __half2float(K_row[d]);
        }

        dot *= scale;

        if (mask != nullptr) {
            float m = __half2float(mask[b * seq_len * seq_len + q_idx * seq_len + k_idx]);
            if (m == 0.0f) dot = -1e9f;
        }

        scores[k_idx] = dot;
    }
    __syncthreads();

    // Softmax (in FP32 for stability)
    float max_val = -1e10f;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        max_val = fmaxf(max_val, scores[i]);
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }

    __shared__ float block_max;
    if (tid % WARP_SIZE == 0) atomicMax((int*)&block_max, __float_as_int(max_val));
    __syncthreads();
    max_val = block_max;

    float sum = 0.0f;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_val);
        sum += scores[i];
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float block_sum;
    if (tid == 0) block_sum = 0.0f;
    __syncthreads();
    if (tid % WARP_SIZE == 0) atomicAdd(&block_sum, sum);
    __syncthreads();

    float inv_sum = 1.0f / block_sum;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        scores[i] *= inv_sum;
    }
    __syncthreads();

    // Compute output
    __half* out_ptr = output + (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim;
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < seq_len; k++) {
            acc += scores[k] * __half2float(V_head[k * head_dim + d]);
        }
        out_ptr[d] = __float2half(acc);
    }
}

// ============================================================================
// Causal (Decoder) Attention
// ============================================================================

extern "C" __global__
void steel_causal_attention_kernel(
    float* __restrict__ output,
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float scale,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim
) {
    extern __shared__ float smem[];

    uint32_t b = blockIdx.z / num_heads;
    uint32_t h = blockIdx.z % num_heads;
    uint32_t q_idx = blockIdx.y;
    uint32_t tid = threadIdx.x;

    if (b >= batch_size || q_idx >= seq_len) return;

    float* scores = smem;
    uint32_t valid_len = q_idx + 1;  // Causal: can only attend to positions <= q_idx

    const float* Q_head = Q + (b * num_heads + h) * seq_len * head_dim;
    const float* K_head = K + (b * num_heads + h) * seq_len * head_dim;
    const float* V_head = V + (b * num_heads + h) * seq_len * head_dim;
    const float* Q_row = Q_head + q_idx * head_dim;

    // Compute scores only for valid positions
    for (uint32_t k_idx = tid; k_idx < valid_len; k_idx += blockDim.x) {
        const float* K_row = K_head + k_idx * head_dim;
        float dot = 0.0f;

        for (uint32_t d = 0; d < head_dim; d++) {
            dot += Q_row[d] * K_row[d];
        }

        scores[k_idx] = dot * scale;
    }

    // Mask out future positions
    for (uint32_t k_idx = tid + valid_len; k_idx < seq_len; k_idx += blockDim.x) {
        scores[k_idx] = -1e9f;
    }
    __syncthreads();

    // Softmax over valid_len
    float max_val = -1e10f;
    for (uint32_t i = tid; i < valid_len; i += blockDim.x) {
        max_val = fmaxf(max_val, scores[i]);
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }

    __shared__ float block_max;
    if (tid % WARP_SIZE == 0) atomicMax((int*)&block_max, __float_as_int(max_val));
    __syncthreads();
    max_val = block_max;

    float sum = 0.0f;
    for (uint32_t i = tid; i < valid_len; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_val);
        sum += scores[i];
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float block_sum;
    if (tid == 0) block_sum = 0.0f;
    __syncthreads();
    if (tid % WARP_SIZE == 0) atomicAdd(&block_sum, sum);
    __syncthreads();

    float inv_sum = 1.0f / block_sum;
    for (uint32_t i = tid; i < valid_len; i += blockDim.x) {
        scores[i] *= inv_sum;
    }
    __syncthreads();

    // Compute output
    float* out_ptr = output + (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim;
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < valid_len; k++) {
            acc += scores[k] * V_head[k * head_dim + d];
        }
        out_ptr[d] = acc;
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_steel_attention_forward(
    void* output,
    const void* Q,
    const void* K,
    const void* V,
    const void* mask,
    float scale,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    cudaStream_t stream
) {
    dim3 blocks(1, seq_len, batch_size * num_heads);
    uint32_t threads = min(256u, seq_len);
    size_t shared_size = (seq_len + head_dim) * sizeof(float);

    steel_attention_forward_kernel<<<blocks, threads, shared_size, stream>>>(
        (float*)output,
        (const float*)Q,
        (const float*)K,
        (const float*)V,
        (const float*)mask,
        scale,
        batch_size, num_heads, seq_len, head_dim
    );

    return cudaGetLastError();
}

int lux_cuda_steel_attention_forward_fp16(
    void* output,
    const void* Q,
    const void* K,
    const void* V,
    const void* mask,
    float scale,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    cudaStream_t stream
) {
    dim3 blocks(1, seq_len, batch_size * num_heads);
    uint32_t threads = min(256u, seq_len);
    size_t shared_size = seq_len * sizeof(float);

    steel_attention_forward_fp16_kernel<<<blocks, threads, shared_size, stream>>>(
        (__half*)output,
        (const __half*)Q,
        (const __half*)K,
        (const __half*)V,
        (const __half*)mask,
        scale,
        batch_size, num_heads, seq_len, head_dim
    );

    return cudaGetLastError();
}

int lux_cuda_steel_causal_attention(
    void* output,
    const void* Q,
    const void* K,
    const void* V,
    float scale,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    cudaStream_t stream
) {
    dim3 blocks(1, seq_len, batch_size * num_heads);
    uint32_t threads = min(256u, seq_len);
    size_t shared_size = seq_len * sizeof(float);

    steel_causal_attention_kernel<<<blocks, threads, shared_size, stream>>>(
        (float*)output,
        (const float*)Q,
        (const float*)K,
        (const float*)V,
        scale,
        batch_size, num_heads, seq_len, head_dim
    );

    return cudaGetLastError();
}

}  // extern "C"
