// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Split-K Attention CUDA Kernels
// Parallelizes attention computation across the K dimension for better GPU utilization

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// Split-K Attention Configuration
// ============================================================================

#define WARP_SIZE 32
#define BLOCK_Q 32
#define BLOCK_K 64
#define BLOCK_SIZE 256

// ============================================================================
// Split-K Attention Forward
// ============================================================================

// Split the KV sequence into chunks, compute partial attention in parallel,
// then reduce. Useful when seq_len >> num_heads for better parallelism.

extern "C" __global__
void splitk_attention_partial_kernel(
    float* __restrict__ partial_output,    // [B, H, num_splits, S_q, D]
    float* __restrict__ partial_lse,       // [B, H, num_splits, S_q]
    const float* __restrict__ Q,           // [B, H, S_q, D]
    const float* __restrict__ K,           // [B, H, S_kv, D]
    const float* __restrict__ V,           // [B, H, S_kv, D]
    const float* __restrict__ mask,        // [B, 1, S_q, S_kv] or nullptr
    float scale,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len_q,
    uint32_t seq_len_kv,
    uint32_t head_dim,
    uint32_t num_splits
) {
    extern __shared__ float smem[];

    float* K_tile = smem;
    float* V_tile = smem + BLOCK_K * head_dim;
    float* scores = smem + 2 * BLOCK_K * head_dim;

    uint32_t b = blockIdx.z / (num_heads * num_splits);
    uint32_t remainder = blockIdx.z % (num_heads * num_splits);
    uint32_t h = remainder / num_splits;
    uint32_t split_idx = remainder % num_splits;

    uint32_t q_idx = blockIdx.y;
    uint32_t tid = threadIdx.x;

    if (b >= batch_size || q_idx >= seq_len_q) return;

    // KV range for this split
    uint32_t kv_per_split = (seq_len_kv + num_splits - 1) / num_splits;
    uint32_t kv_start = split_idx * kv_per_split;
    uint32_t kv_end = min(kv_start + kv_per_split, seq_len_kv);

    if (kv_start >= seq_len_kv) {
        // This split has no work
        if (tid == 0) {
            uint32_t out_idx = ((b * num_heads + h) * num_splits + split_idx) * seq_len_q + q_idx;
            partial_lse[out_idx] = -INFINITY;
        }
        return;
    }

    // Base pointers
    const float* Q_row = Q + (b * num_heads + h) * seq_len_q * head_dim + q_idx * head_dim;
    const float* K_base = K + (b * num_heads + h) * seq_len_kv * head_dim;
    const float* V_base = V + (b * num_heads + h) * seq_len_kv * head_dim;

    // Initialize accumulator
    float O_acc[64];  // Assume max head_dim = 64
    float m_i = -INFINITY;
    float l_i = 0.0f;

    for (uint32_t d = 0; d < head_dim; d++) {
        O_acc[d] = 0.0f;
    }

    // Process KV range in tiles
    uint32_t num_kv_tiles = (kv_end - kv_start + BLOCK_K - 1) / BLOCK_K;

    for (uint32_t kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        uint32_t k_start = kv_start + kv_tile * BLOCK_K;
        uint32_t k_end_tile = min(k_start + BLOCK_K, kv_end);
        uint32_t tile_len = k_end_tile - k_start;

        // Load K tile
        for (uint32_t i = tid; i < tile_len * head_dim; i += blockDim.x) {
            uint32_t k_local = i / head_dim;
            uint32_t d = i % head_dim;
            K_tile[k_local * head_dim + d] = K_base[(k_start + k_local) * head_dim + d];
        }

        // Load V tile
        for (uint32_t i = tid; i < tile_len * head_dim; i += blockDim.x) {
            uint32_t v_local = i / head_dim;
            uint32_t d = i % head_dim;
            V_tile[v_local * head_dim + d] = V_base[(k_start + v_local) * head_dim + d];
        }
        __syncthreads();

        // Compute attention scores
        for (uint32_t k_local = tid; k_local < tile_len; k_local += blockDim.x) {
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) {
                dot += Q_row[d] * K_tile[k_local * head_dim + d];
            }
            dot *= scale;

            // Apply mask if present
            if (mask != nullptr) {
                uint32_t k_idx = k_start + k_local;
                float m = mask[b * seq_len_q * seq_len_kv + q_idx * seq_len_kv + k_idx];
                if (m == 0.0f) {
                    dot = -INFINITY;
                }
            }

            scores[k_local] = dot;
        }
        __syncthreads();

        // Online softmax update (single thread for simplicity - production would parallelize)
        if (tid == 0) {
            // Find max in this tile
            float m_ij = -INFINITY;
            for (uint32_t k_local = 0; k_local < tile_len; k_local++) {
                m_ij = fmaxf(m_ij, scores[k_local]);
            }

            // New max
            float m_new = fmaxf(m_i, m_ij);

            // Rescale previous sum
            float l_rescale = expf(m_i - m_new);

            // Compute new sum
            float l_new = l_i * l_rescale;
            for (uint32_t k_local = 0; k_local < tile_len; k_local++) {
                l_new += expf(scores[k_local] - m_new);
            }

            // Rescale output accumulator
            for (uint32_t d = 0; d < head_dim; d++) {
                O_acc[d] *= l_rescale;
            }

            // Accumulate new contribution
            for (uint32_t k_local = 0; k_local < tile_len; k_local++) {
                float p = expf(scores[k_local] - m_new);
                for (uint32_t d = 0; d < head_dim; d++) {
                    O_acc[d] += p * V_tile[k_local * head_dim + d];
                }
            }

            m_i = m_new;
            l_i = l_new;
        }
        __syncthreads();
    }

    // Write partial output and LSE
    if (tid == 0) {
        uint32_t out_base = ((b * num_heads + h) * num_splits + split_idx) * seq_len_q * head_dim +
                           q_idx * head_dim;
        uint32_t lse_idx = ((b * num_heads + h) * num_splits + split_idx) * seq_len_q + q_idx;

        for (uint32_t d = 0; d < head_dim; d++) {
            partial_output[out_base + d] = O_acc[d] / l_i;
        }
        partial_lse[lse_idx] = m_i + logf(l_i);
    }
}

// ============================================================================
// Split-K Reduction Kernel
// ============================================================================

extern "C" __global__
void splitk_attention_reduce_kernel(
    float* __restrict__ output,            // [B, H, S_q, D]
    const float* __restrict__ partial_output,  // [B, H, num_splits, S_q, D]
    const float* __restrict__ partial_lse,     // [B, H, num_splits, S_q]
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len_q,
    uint32_t head_dim,
    uint32_t num_splits
) {
    uint32_t b = blockIdx.z / num_heads;
    uint32_t h = blockIdx.z % num_heads;
    uint32_t q_idx = blockIdx.y;
    uint32_t tid = threadIdx.x;

    if (b >= batch_size || q_idx >= seq_len_q) return;

    // Find global max LSE
    float global_max = -INFINITY;
    for (uint32_t s = 0; s < num_splits; s++) {
        uint32_t lse_idx = ((b * num_heads + h) * num_splits + s) * seq_len_q + q_idx;
        global_max = fmaxf(global_max, partial_lse[lse_idx]);
    }

    // Compute global sum
    float global_sum = 0.0f;
    for (uint32_t s = 0; s < num_splits; s++) {
        uint32_t lse_idx = ((b * num_heads + h) * num_splits + s) * seq_len_q + q_idx;
        float lse = partial_lse[lse_idx];
        if (lse > -INFINITY) {
            global_sum += expf(lse - global_max);
        }
    }

    // Combine outputs
    float* out = output + (b * num_heads + h) * seq_len_q * head_dim + q_idx * head_dim;

    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;

        for (uint32_t s = 0; s < num_splits; s++) {
            uint32_t lse_idx = ((b * num_heads + h) * num_splits + s) * seq_len_q + q_idx;
            float lse = partial_lse[lse_idx];

            if (lse > -INFINITY) {
                float weight = expf(lse - global_max) / global_sum;
                uint32_t part_idx = ((b * num_heads + h) * num_splits + s) * seq_len_q * head_dim +
                                   q_idx * head_dim + d;
                acc += weight * partial_output[part_idx];
            }
        }

        out[d] = acc;
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_splitk_attention(
    void* output,
    void* partial_output,
    void* partial_lse,
    const void* Q,
    const void* K,
    const void* V,
    const void* mask,
    float scale,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len_q,
    uint32_t seq_len_kv,
    uint32_t head_dim,
    uint32_t num_splits,
    cudaStream_t stream
) {
    // Phase 1: Compute partial attention
    dim3 blocks1(1, seq_len_q, batch_size * num_heads * num_splits);
    uint32_t threads = BLOCK_SIZE;
    size_t shared_size = (2 * BLOCK_K * head_dim + BLOCK_K) * sizeof(float);

    splitk_attention_partial_kernel<<<blocks1, threads, shared_size, stream>>>(
        (float*)partial_output,
        (float*)partial_lse,
        (const float*)Q,
        (const float*)K,
        (const float*)V,
        (const float*)mask,
        scale,
        batch_size, num_heads, seq_len_q, seq_len_kv, head_dim, num_splits
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // Phase 2: Reduce partials
    dim3 blocks2(1, seq_len_q, batch_size * num_heads);

    splitk_attention_reduce_kernel<<<blocks2, threads, 0, stream>>>(
        (float*)output,
        (const float*)partial_output,
        (const float*)partial_lse,
        batch_size, num_heads, seq_len_q, head_dim, num_splits
    );

    return cudaGetLastError();
}

}  // extern "C"
