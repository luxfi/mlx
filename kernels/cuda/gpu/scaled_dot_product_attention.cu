// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Scaled Dot-Product Attention CUDA Kernels
// Fused implementation of: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
// Supports causal masking, GQA/MQA, and Flash Attention-style memory efficiency

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// ============================================================================
// Configuration
// ============================================================================

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define TILE_Q 64     // Query tile size
#define TILE_KV 64    // Key/Value tile size

// ============================================================================
// Utility Functions
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

// Online softmax state for Flash Attention
struct OnlineState {
    float max_val;
    float sum;
    float output;  // Accumulator for weighted sum
};

__device__ __forceinline__
OnlineState online_init() {
    return {-INFINITY, 0.0f, 0.0f};
}

// ============================================================================
// Basic SDPA (for small sequence lengths, materializes attention matrix)
// Q, K, V: [batch, num_heads, seq_len, head_dim]
// ============================================================================

extern "C" __global__
void lux_sdpa_basic_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ mask,  // Optional: [seq_len, seq_len] or nullptr
    float scale,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    bool is_causal
) {
    extern __shared__ float smem[];

    // Block handles one query position for one (batch, head)
    uint32_t bh = blockIdx.z;  // batch * num_heads + head
    uint32_t b = bh / num_heads;
    uint32_t h = bh % num_heads;
    uint32_t q_idx = blockIdx.y;
    uint32_t tid = threadIdx.x;

    if (b >= batch_size || q_idx >= seq_len) return;

    float* scores = smem;  // [seq_len]

    // Pointers for this (batch, head)
    const float* Q_ptr = Q + (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim;
    const float* K_ptr = K + (b * num_heads + h) * seq_len * head_dim;
    const float* V_ptr = V + (b * num_heads + h) * seq_len * head_dim;
    float* O_ptr = output + (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim;

    // Step 1: Compute Q @ K^T * scale
    for (uint32_t k_idx = tid; k_idx < seq_len; k_idx += blockDim.x) {
        // Causal mask: only attend to positions <= q_idx
        if (is_causal && k_idx > q_idx) {
            scores[k_idx] = -INFINITY;
            continue;
        }

        const float* K_row = K_ptr + k_idx * head_dim;
        float dot = 0.0f;

        #pragma unroll 4
        for (uint32_t d = 0; d < head_dim; d++) {
            dot += Q_ptr[d] * K_row[d];
        }

        dot *= scale;

        // Apply external mask if provided
        if (mask != nullptr && mask[q_idx * seq_len + k_idx] == 0.0f) {
            dot = -INFINITY;
        }

        scores[k_idx] = dot;
    }
    __syncthreads();

    // Step 2: Softmax
    // Find max
    float local_max = -INFINITY;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        local_max = fmaxf(local_max, scores[i]);
    }

    local_max = warp_reduce_max_f32(local_max);

    __shared__ float block_max;
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    __shared__ float warp_vals[WARP_SIZE];

    if (lane == 0) warp_vals[wid] = local_max;
    __syncthreads();

    if (wid == 0) {
        local_max = (lane < blockDim.x / WARP_SIZE) ? warp_vals[lane] : -INFINITY;
        local_max = warp_reduce_max_f32(local_max);
        if (lane == 0) block_max = local_max;
    }
    __syncthreads();

    float max_val = block_max;

    // Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_val);
        local_sum += scores[i];
    }

    local_sum = warp_reduce_sum_f32(local_sum);

    __shared__ float block_sum;
    if (lane == 0) warp_vals[wid] = local_sum;
    __syncthreads();

    if (wid == 0) {
        local_sum = (lane < blockDim.x / WARP_SIZE) ? warp_vals[lane] : 0.0f;
        local_sum = warp_reduce_sum_f32(local_sum);
        if (lane == 0) block_sum = local_sum;
    }
    __syncthreads();

    float inv_sum = 1.0f / block_sum;

    // Normalize
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        scores[i] *= inv_sum;
    }
    __syncthreads();

    // Step 3: Compute attention @ V
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;

        for (uint32_t k = 0; k < seq_len; k++) {
            acc += scores[k] * V_ptr[k * head_dim + d];
        }

        O_ptr[d] = acc;
    }
}

// ============================================================================
// Fused SDPA with Online Softmax (Memory Efficient)
// Processes KV in tiles without materializing full attention matrix
// ============================================================================

extern "C" __global__
void lux_sdpa_fused_f32_kernel(
    float* __restrict__ output,
    float* __restrict__ lse,          // Optional: log-sum-exp for backward
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float scale,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    bool is_causal
) {
    // Each block handles one query position
    uint32_t bh = blockIdx.z;
    uint32_t b = bh / num_heads;
    uint32_t h = bh % num_heads;
    uint32_t q_idx = blockIdx.y;
    uint32_t tid = threadIdx.x;

    if (b >= batch_size || q_idx >= seq_len) return;

    // Pointers
    const float* Q_ptr = Q + (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim;
    const float* K_base = K + (b * num_heads + h) * seq_len * head_dim;
    const float* V_base = V + (b * num_heads + h) * seq_len * head_dim;
    float* O_ptr = output + (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim;

    // Thread-local accumulators (per head dimension)
    float O_acc[16];  // Assuming head_dim <= 256, each thread handles up to 16
    uint32_t num_acc = (head_dim + blockDim.x - 1) / blockDim.x;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        O_acc[i] = 0.0f;
    }

    float m_prev = -INFINITY;
    float l_prev = 0.0f;

    // Determine KV range based on causality
    uint32_t kv_end = is_causal ? (q_idx + 1) : seq_len;

    // Process KV in tiles
    for (uint32_t kv_start = 0; kv_start < kv_end; kv_start += TILE_KV) {
        uint32_t kv_tile_end = min(kv_start + TILE_KV, kv_end);

        // Compute scores and online softmax for this tile
        float tile_max = -INFINITY;
        float tile_sum = 0.0f;

        // First pass: find max and compute exp
        for (uint32_t k_idx = kv_start; k_idx < kv_tile_end; k_idx++) {
            const float* K_row = K_base + k_idx * head_dim;

            // Compute Q @ K^T
            float dot = 0.0f;
            for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                dot += Q_ptr[d] * K_row[d];
            }

            // Reduce dot product across threads
            dot = warp_reduce_sum_f32(dot);

            // Only lane 0 has the full dot product
            __shared__ float s_dot;
            if (tid == 0) {
                s_dot = dot * scale;
            }
            __syncthreads();

            float score = s_dot;
            tile_max = fmaxf(tile_max, score);
        }

        // Broadcast tile_max
        __shared__ float s_tile_max;
        if (tid == 0) s_tile_max = tile_max;
        __syncthreads();
        tile_max = s_tile_max;

        // Update running statistics
        float m_new = fmaxf(m_prev, tile_max);

        // Second pass: accumulate with rescaling
        for (uint32_t k_idx = kv_start; k_idx < kv_tile_end; k_idx++) {
            const float* K_row = K_base + k_idx * head_dim;
            const float* V_row = V_base + k_idx * head_dim;

            // Recompute score (could cache in smem for efficiency)
            float dot = 0.0f;
            for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
                dot += Q_ptr[d] * K_row[d];
            }
            dot = warp_reduce_sum_f32(dot);

            __shared__ float s_score;
            if (tid == 0) s_score = dot * scale;
            __syncthreads();

            float p = expf(s_score - m_new);
            tile_sum += p;

            // Accumulate V weighted by attention
            for (uint32_t i = 0; i < num_acc && tid + i * blockDim.x < head_dim; i++) {
                uint32_t d = tid + i * blockDim.x;
                O_acc[i] += p * V_row[d];
            }
        }

        // Rescale previous accumulator
        float rescale = expf(m_prev - m_new);
        for (uint32_t i = 0; i < num_acc; i++) {
            O_acc[i] *= rescale;
        }

        // Update running sum
        l_prev = l_prev * rescale + tile_sum;
        m_prev = m_new;
    }

    // Final normalization and output
    float inv_l = 1.0f / l_prev;
    for (uint32_t i = 0; i < num_acc && tid + i * blockDim.x < head_dim; i++) {
        uint32_t d = tid + i * blockDim.x;
        O_ptr[d] = O_acc[i] * inv_l;
    }

    // Optionally save LSE for backward
    if (lse != nullptr && tid == 0) {
        lse[(b * num_heads + h) * seq_len + q_idx] = m_prev + logf(l_prev);
    }
}

// ============================================================================
// SDPA with GQA (Grouped Query Attention)
// num_heads > num_kv_heads, multiple Q heads share same KV head
// ============================================================================

extern "C" __global__
void lux_sdpa_gqa_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float scale,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    bool is_causal
) {
    extern __shared__ float smem[];

    uint32_t b = blockIdx.z / num_heads;
    uint32_t h = blockIdx.z % num_heads;
    uint32_t q_idx = blockIdx.y;
    uint32_t tid = threadIdx.x;

    if (b >= batch_size || q_idx >= seq_len) return;

    // Map Q head to KV head
    uint32_t kv_h = h / (num_heads / num_kv_heads);

    float* scores = smem;

    // Q pointer uses h, KV pointers use kv_h
    const float* Q_ptr = Q + (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim;
    const float* K_ptr = K + (b * num_kv_heads + kv_h) * seq_len * head_dim;
    const float* V_ptr = V + (b * num_kv_heads + kv_h) * seq_len * head_dim;
    float* O_ptr = output + (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim;

    uint32_t kv_end = is_causal ? (q_idx + 1) : seq_len;

    // Compute attention scores
    for (uint32_t k_idx = tid; k_idx < seq_len; k_idx += blockDim.x) {
        if (k_idx >= kv_end) {
            scores[k_idx] = -INFINITY;
            continue;
        }

        const float* K_row = K_ptr + k_idx * head_dim;
        float dot = 0.0f;

        for (uint32_t d = 0; d < head_dim; d++) {
            dot += Q_ptr[d] * K_row[d];
        }

        scores[k_idx] = dot * scale;
    }
    __syncthreads();

    // Softmax
    float local_max = -INFINITY;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        local_max = fmaxf(local_max, scores[i]);
    }
    local_max = warp_reduce_max_f32(local_max);

    __shared__ float s_max, s_sum;
    __shared__ float warp_vals[WARP_SIZE];
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;

    if (lane == 0) warp_vals[wid] = local_max;
    __syncthreads();

    if (wid == 0) {
        local_max = (lane < blockDim.x / WARP_SIZE) ? warp_vals[lane] : -INFINITY;
        local_max = warp_reduce_max_f32(local_max);
        if (lane == 0) s_max = local_max;
    }
    __syncthreads();

    float max_val = s_max;
    float local_sum = 0.0f;

    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_val);
        local_sum += scores[i];
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

    float inv_sum = 1.0f / s_sum;

    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        scores[i] *= inv_sum;
    }
    __syncthreads();

    // Compute output
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < kv_end; k++) {
            acc += scores[k] * V_ptr[k * head_dim + d];
        }
        O_ptr[d] = acc;
    }
}

// ============================================================================
// SDPA FP16
// ============================================================================

extern "C" __global__
void lux_sdpa_f16_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    float scale,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    bool is_causal
) {
    extern __shared__ float smem[];

    uint32_t b = blockIdx.z / num_heads;
    uint32_t h = blockIdx.z % num_heads;
    uint32_t q_idx = blockIdx.y;
    uint32_t tid = threadIdx.x;

    if (b >= batch_size || q_idx >= seq_len) return;

    float* scores = smem;

    const __half* Q_ptr = Q + (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim;
    const __half* K_ptr = K + (b * num_heads + h) * seq_len * head_dim;
    const __half* V_ptr = V + (b * num_heads + h) * seq_len * head_dim;
    __half* O_ptr = output + (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim;

    uint32_t kv_end = is_causal ? (q_idx + 1) : seq_len;

    // Compute scores in FP32
    for (uint32_t k_idx = tid; k_idx < seq_len; k_idx += blockDim.x) {
        if (k_idx >= kv_end) {
            scores[k_idx] = -INFINITY;
            continue;
        }

        const __half* K_row = K_ptr + k_idx * head_dim;
        float dot = 0.0f;

        for (uint32_t d = 0; d < head_dim; d++) {
            dot += __half2float(Q_ptr[d]) * __half2float(K_row[d]);
        }

        scores[k_idx] = dot * scale;
    }
    __syncthreads();

    // Softmax in FP32
    float local_max = -INFINITY;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        local_max = fmaxf(local_max, scores[i]);
    }
    local_max = warp_reduce_max_f32(local_max);

    __shared__ float s_max, s_sum;
    __shared__ float warp_vals[WARP_SIZE];
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;

    if (lane == 0) warp_vals[wid] = local_max;
    __syncthreads();

    if (wid == 0) {
        local_max = (lane < blockDim.x / WARP_SIZE) ? warp_vals[lane] : -INFINITY;
        local_max = warp_reduce_max_f32(local_max);
        if (lane == 0) s_max = local_max;
    }
    __syncthreads();

    float max_val = s_max;
    float local_sum = 0.0f;

    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_val);
        local_sum += scores[i];
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

    float inv_sum = 1.0f / s_sum;

    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        scores[i] *= inv_sum;
    }
    __syncthreads();

    // Output in FP16
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < kv_end; k++) {
            acc += scores[k] * __half2float(V_ptr[k * head_dim + d]);
        }
        O_ptr[d] = __float2half(acc);
    }
}

// ============================================================================
// SDPA with Dropout (Training)
// ============================================================================

extern "C" __global__
void lux_sdpa_dropout_f32_kernel(
    float* __restrict__ output,
    float* __restrict__ dropout_mask,  // [batch, heads, seq, seq]
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float scale,
    float dropout_prob,
    uint64_t seed,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    bool is_causal
) {
    extern __shared__ float smem[];

    uint32_t b = blockIdx.z / num_heads;
    uint32_t h = blockIdx.z % num_heads;
    uint32_t q_idx = blockIdx.y;
    uint32_t tid = threadIdx.x;

    if (b >= batch_size || q_idx >= seq_len) return;

    float* scores = smem;

    const float* Q_ptr = Q + (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim;
    const float* K_ptr = K + (b * num_heads + h) * seq_len * head_dim;
    const float* V_ptr = V + (b * num_heads + h) * seq_len * head_dim;
    float* O_ptr = output + (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim;
    float* mask_ptr = dropout_mask + ((b * num_heads + h) * seq_len + q_idx) * seq_len;

    uint32_t kv_end = is_causal ? (q_idx + 1) : seq_len;

    // Compute attention scores with softmax
    for (uint32_t k_idx = tid; k_idx < seq_len; k_idx += blockDim.x) {
        if (k_idx >= kv_end) {
            scores[k_idx] = -INFINITY;
            continue;
        }

        const float* K_row = K_ptr + k_idx * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) {
            dot += Q_ptr[d] * K_row[d];
        }
        scores[k_idx] = dot * scale;
    }
    __syncthreads();

    // Standard softmax
    float local_max = -INFINITY;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        local_max = fmaxf(local_max, scores[i]);
    }
    local_max = warp_reduce_max_f32(local_max);

    __shared__ float s_max, s_sum;
    __shared__ float warp_vals[WARP_SIZE];
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;

    if (lane == 0) warp_vals[wid] = local_max;
    __syncthreads();

    if (wid == 0) {
        local_max = (lane < blockDim.x / WARP_SIZE) ? warp_vals[lane] : -INFINITY;
        local_max = warp_reduce_max_f32(local_max);
        if (lane == 0) s_max = local_max;
    }
    __syncthreads();

    float max_val = s_max;
    float local_sum = 0.0f;

    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_val);
        local_sum += scores[i];
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

    float inv_sum = 1.0f / s_sum;

    // Apply dropout with random mask
    float scale_factor = 1.0f / (1.0f - dropout_prob);
    uint64_t global_idx = (uint64_t)((b * num_heads + h) * seq_len + q_idx) * seq_len;

    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        float attn = scores[i] * inv_sum;

        // Simple LCG random number generator
        uint64_t rand_state = seed ^ (global_idx + i);
        rand_state = rand_state * 6364136223846793005ULL + 1442695040888963407ULL;
        float rand_val = (rand_state >> 33) * (1.0f / 2147483648.0f);

        float mask = (rand_val >= dropout_prob) ? 1.0f : 0.0f;
        scores[i] = attn * mask * scale_factor;
        mask_ptr[i] = mask;  // Save mask for backward
    }
    __syncthreads();

    // Compute output
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < kv_end; k++) {
            acc += scores[k] * V_ptr[k * head_dim + d];
        }
        O_ptr[d] = acc;
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_sdpa_f32(
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
    bool is_causal,
    cudaStream_t stream
) {
    dim3 blocks(1, seq_len, batch_size * num_heads);
    uint32_t threads = min((uint32_t)BLOCK_SIZE, seq_len);
    size_t shared_size = seq_len * sizeof(float);

    lux_sdpa_basic_f32_kernel<<<blocks, threads, shared_size, stream>>>(
        (float*)output,
        (const float*)Q,
        (const float*)K,
        (const float*)V,
        (const float*)mask,
        scale,
        batch_size, num_heads, seq_len, head_dim,
        is_causal
    );

    return cudaGetLastError();
}

int lux_cuda_sdpa_fused_f32(
    void* output,
    void* lse,
    const void* Q,
    const void* K,
    const void* V,
    float scale,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    bool is_causal,
    cudaStream_t stream
) {
    dim3 blocks(1, seq_len, batch_size * num_heads);
    uint32_t threads = min((uint32_t)BLOCK_SIZE, head_dim);

    lux_sdpa_fused_f32_kernel<<<blocks, threads, 0, stream>>>(
        (float*)output,
        (float*)lse,
        (const float*)Q,
        (const float*)K,
        (const float*)V,
        scale,
        batch_size, num_heads, seq_len, head_dim,
        is_causal
    );

    return cudaGetLastError();
}

int lux_cuda_sdpa_gqa_f32(
    void* output,
    const void* Q,
    const void* K,
    const void* V,
    float scale,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    bool is_causal,
    cudaStream_t stream
) {
    dim3 blocks(1, seq_len, batch_size * num_heads);
    uint32_t threads = min((uint32_t)BLOCK_SIZE, seq_len);
    size_t shared_size = seq_len * sizeof(float);

    lux_sdpa_gqa_f32_kernel<<<blocks, threads, shared_size, stream>>>(
        (float*)output,
        (const float*)Q,
        (const float*)K,
        (const float*)V,
        scale,
        batch_size, num_heads, num_kv_heads, seq_len, head_dim,
        is_causal
    );

    return cudaGetLastError();
}

int lux_cuda_sdpa_f16(
    void* output,
    const void* Q,
    const void* K,
    const void* V,
    float scale,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    bool is_causal,
    cudaStream_t stream
) {
    dim3 blocks(1, seq_len, batch_size * num_heads);
    uint32_t threads = min((uint32_t)BLOCK_SIZE, seq_len);
    size_t shared_size = seq_len * sizeof(float);

    lux_sdpa_f16_kernel<<<blocks, threads, shared_size, stream>>>(
        (__half*)output,
        (const __half*)Q,
        (const __half*)K,
        (const __half*)V,
        scale,
        batch_size, num_heads, seq_len, head_dim,
        is_causal
    );

    return cudaGetLastError();
}

int lux_cuda_sdpa_dropout_f32(
    void* output,
    void* dropout_mask,
    const void* Q,
    const void* K,
    const void* V,
    float scale,
    float dropout_prob,
    uint64_t seed,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    bool is_causal,
    cudaStream_t stream
) {
    dim3 blocks(1, seq_len, batch_size * num_heads);
    uint32_t threads = min((uint32_t)BLOCK_SIZE, seq_len);
    size_t shared_size = seq_len * sizeof(float);

    lux_sdpa_dropout_f32_kernel<<<blocks, threads, shared_size, stream>>>(
        (float*)output,
        (float*)dropout_mask,
        (const float*)Q,
        (const float*)K,
        (const float*)V,
        scale,
        dropout_prob,
        seed,
        batch_size, num_heads, seq_len, head_dim,
        is_causal
    );

    return cudaGetLastError();
}

}  // extern "C"
