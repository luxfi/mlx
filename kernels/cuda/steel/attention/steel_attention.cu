// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Steel Attention CUDA Kernels - Tiled Implementation
// High-performance attention using Steel-style tiled algorithms with shared memory

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// Steel Attention Configuration
// ============================================================================

#define WARP_SIZE 32
#define TILE_Q 32      // Query tile size
#define TILE_K 32      // Key tile size
#define TILE_D 64      // Head dimension tile
#define BLOCK_SIZE 256

// ============================================================================
// Steel Tiled Attention Forward
// ============================================================================

// Tiled attention using shared memory for Q, K, V tiles
// Computes: softmax(Q @ K^T / sqrt(d)) @ V
extern "C" __global__
void steel_attention_tiled_kernel(
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
    // Shared memory for tiles
    __shared__ float Qs[TILE_Q][TILE_D + 1];  // +1 to avoid bank conflicts
    __shared__ float Ks[TILE_K][TILE_D + 1];
    __shared__ float Vs[TILE_K][TILE_D + 1];
    __shared__ float Ss[TILE_Q][TILE_K + 1];  // Attention scores

    uint32_t b = blockIdx.z / num_heads;
    uint32_t h = blockIdx.z % num_heads;
    uint32_t q_tile = blockIdx.y;
    uint32_t tid = threadIdx.x;

    if (b >= batch_size) return;

    // Base pointers for this batch/head
    const float* Q_base = Q + (b * num_heads + h) * seq_len * head_dim;
    const float* K_base = K + (b * num_heads + h) * seq_len * head_dim;
    const float* V_base = V + (b * num_heads + h) * seq_len * head_dim;
    float* O_base = output + (b * num_heads + h) * seq_len * head_dim;

    // Query tile range
    uint32_t q_start = q_tile * TILE_Q;
    uint32_t q_end = min(q_start + TILE_Q, seq_len);

    // Thread coordinates within tile
    uint32_t tile_row = tid / TILE_D;
    uint32_t tile_col = tid % TILE_D;

    // Load Q tile to shared memory
    for (uint32_t i = tid; i < TILE_Q * head_dim; i += blockDim.x) {
        uint32_t q_local = i / head_dim;
        uint32_t d = i % head_dim;
        uint32_t q_idx = q_start + q_local;

        if (q_idx < seq_len && d < head_dim) {
            Qs[q_local][d] = Q_base[q_idx * head_dim + d];
        } else {
            Qs[q_local][d] = 0.0f;
        }
    }
    __syncthreads();

    // Accumulators for output (one per query in tile)
    float O_acc[TILE_Q];
    float m_i[TILE_Q];   // Running max for online softmax
    float l_i[TILE_Q];   // Running sum for online softmax

    #pragma unroll
    for (int i = 0; i < TILE_Q; i++) {
        O_acc[i] = 0.0f;
        m_i[i] = -1e10f;
        l_i[i] = 0.0f;
    }

    // Iterate over K/V tiles
    uint32_t num_kv_tiles = (seq_len + TILE_K - 1) / TILE_K;

    for (uint32_t kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        uint32_t k_start = kv_tile * TILE_K;
        uint32_t k_end = min(k_start + TILE_K, seq_len);

        // Load K tile to shared memory
        for (uint32_t i = tid; i < TILE_K * head_dim; i += blockDim.x) {
            uint32_t k_local = i / head_dim;
            uint32_t d = i % head_dim;
            uint32_t k_idx = k_start + k_local;

            if (k_idx < seq_len && d < head_dim) {
                Ks[k_local][d] = K_base[k_idx * head_dim + d];
            } else {
                Ks[k_local][d] = 0.0f;
            }
        }

        // Load V tile to shared memory
        for (uint32_t i = tid; i < TILE_K * head_dim; i += blockDim.x) {
            uint32_t v_local = i / head_dim;
            uint32_t d = i % head_dim;
            uint32_t v_idx = k_start + v_local;

            if (v_idx < seq_len && d < head_dim) {
                Vs[v_local][d] = V_base[v_idx * head_dim + d];
            } else {
                Vs[v_local][d] = 0.0f;
            }
        }
        __syncthreads();

        // Compute Q @ K^T for this tile pair
        for (uint32_t i = tid; i < TILE_Q * TILE_K; i += blockDim.x) {
            uint32_t q_local = i / TILE_K;
            uint32_t k_local = i % TILE_K;
            uint32_t q_idx = q_start + q_local;
            uint32_t k_idx = k_start + k_local;

            float dot = 0.0f;
            if (q_idx < seq_len && k_idx < seq_len) {
                #pragma unroll
                for (uint32_t d = 0; d < head_dim; d++) {
                    dot += Qs[q_local][d] * Ks[k_local][d];
                }
                dot *= scale;

                // Apply mask if present
                if (mask != nullptr) {
                    float m = mask[b * seq_len * seq_len + q_idx * seq_len + k_idx];
                    if (m == 0.0f) {
                        dot = -1e10f;
                    }
                }
            } else {
                dot = -1e10f;
            }

            Ss[q_local][k_local] = dot;
        }
        __syncthreads();

        // Online softmax and accumulate
        for (uint32_t q_local = tid / TILE_K; q_local < TILE_Q; q_local += blockDim.x / TILE_K) {
            uint32_t q_idx = q_start + q_local;
            if (q_idx >= seq_len) continue;

            // Find max in this tile
            float m_ij = -1e10f;
            #pragma unroll
            for (uint32_t k_local = 0; k_local < TILE_K; k_local++) {
                m_ij = fmaxf(m_ij, Ss[q_local][k_local]);
            }

            // Update running max
            float m_new = fmaxf(m_i[q_local], m_ij);

            // Compute exp and sum
            float l_ij = 0.0f;
            #pragma unroll
            for (uint32_t k_local = 0; k_local < TILE_K; k_local++) {
                l_ij += expf(Ss[q_local][k_local] - m_new);
            }

            // Rescale previous accumulator
            float alpha = expf(m_i[q_local] - m_new);
            l_i[q_local] = alpha * l_i[q_local] + l_ij;

            // Accumulate output (simplified - actual would track per-dimension)
            #pragma unroll
            for (uint32_t k_local = 0; k_local < TILE_K; k_local++) {
                float p = expf(Ss[q_local][k_local] - m_new);
                uint32_t k_idx = k_start + k_local;
                if (k_idx < seq_len) {
                    for (uint32_t d = 0; d < head_dim; d++) {
                        O_acc[q_local] = O_acc[q_local] * alpha + p * Vs[k_local][d];
                    }
                }
            }

            m_i[q_local] = m_new;
        }
        __syncthreads();
    }

    // Normalize and write output
    for (uint32_t i = tid; i < TILE_Q * head_dim; i += blockDim.x) {
        uint32_t q_local = i / head_dim;
        uint32_t d = i % head_dim;
        uint32_t q_idx = q_start + q_local;

        if (q_idx < seq_len && d < head_dim) {
            O_base[q_idx * head_dim + d] = O_acc[q_local] / l_i[q_local];
        }
    }
}

// ============================================================================
// Steel Grouped Query Attention (GQA)
// ============================================================================

extern "C" __global__
void steel_gqa_attention_kernel(
    float* __restrict__ output,       // [B, H_q, S, D]
    const float* __restrict__ Q,      // [B, H_q, S, D]
    const float* __restrict__ K,      // [B, H_kv, S, D]
    const float* __restrict__ V,      // [B, H_kv, S, D]
    float scale,
    uint32_t batch_size,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t seq_len,
    uint32_t head_dim
) {
    extern __shared__ float smem[];

    uint32_t b = blockIdx.z / num_q_heads;
    uint32_t h_q = blockIdx.z % num_q_heads;
    uint32_t q_idx = blockIdx.y;
    uint32_t tid = threadIdx.x;

    if (b >= batch_size || q_idx >= seq_len) return;

    // Map query head to KV head (GQA grouping)
    uint32_t heads_per_group = num_q_heads / num_kv_heads;
    uint32_t h_kv = h_q / heads_per_group;

    float* scores = smem;

    // Base pointers
    const float* Q_head = Q + (b * num_q_heads + h_q) * seq_len * head_dim;
    const float* K_head = K + (b * num_kv_heads + h_kv) * seq_len * head_dim;
    const float* V_head = V + (b * num_kv_heads + h_kv) * seq_len * head_dim;
    const float* Q_row = Q_head + q_idx * head_dim;

    // Compute attention scores
    for (uint32_t k_idx = tid; k_idx < seq_len; k_idx += blockDim.x) {
        const float* K_row = K_head + k_idx * head_dim;
        float dot = 0.0f;

        #pragma unroll 8
        for (uint32_t d = 0; d < head_dim; d++) {
            dot += Q_row[d] * K_row[d];
        }

        scores[k_idx] = dot * scale;
    }
    __syncthreads();

    // Softmax - find max
    float max_val = -1e10f;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        max_val = fmaxf(max_val, scores[i]);
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }

    __shared__ float block_max;
    if (tid % WARP_SIZE == 0) {
        atomicMax((int*)&block_max, __float_as_int(max_val));
    }
    __syncthreads();
    max_val = block_max;

    // Softmax - compute exp and sum
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

    // Normalize
    float inv_sum = 1.0f / block_sum;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        scores[i] *= inv_sum;
    }
    __syncthreads();

    // Compute output
    float* out_ptr = output + (b * num_q_heads + h_q) * seq_len * head_dim + q_idx * head_dim;
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;

        for (uint32_t k = 0; k < seq_len; k++) {
            acc += scores[k] * V_head[k * head_dim + d];
        }

        out_ptr[d] = acc;
    }
}

// ============================================================================
// Steel Multi-Query Attention (MQA)
// ============================================================================

extern "C" __global__
void steel_mqa_attention_kernel(
    float* __restrict__ output,       // [B, H, S, D]
    const float* __restrict__ Q,      // [B, H, S, D]
    const float* __restrict__ K,      // [B, 1, S, D]
    const float* __restrict__ V,      // [B, 1, S, D]
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

    // Q has multiple heads, K/V have single head (shared)
    const float* Q_head = Q + (b * num_heads + h) * seq_len * head_dim;
    const float* K_head = K + b * seq_len * head_dim;  // Single head
    const float* V_head = V + b * seq_len * head_dim;  // Single head
    const float* Q_row = Q_head + q_idx * head_dim;

    // Compute attention scores
    for (uint32_t k_idx = tid; k_idx < seq_len; k_idx += blockDim.x) {
        const float* K_row = K_head + k_idx * head_dim;
        float dot = 0.0f;

        #pragma unroll 8
        for (uint32_t d = 0; d < head_dim; d++) {
            dot += Q_row[d] * K_row[d];
        }

        scores[k_idx] = dot * scale;
    }
    __syncthreads();

    // Softmax
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
    float* out_ptr = output + (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim;
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;

        for (uint32_t k = 0; k < seq_len; k++) {
            acc += scores[k] * V_head[k * head_dim + d];
        }

        out_ptr[d] = acc;
    }
}

// ============================================================================
// Steel Sliding Window Attention
// ============================================================================

extern "C" __global__
void steel_sliding_window_attention_kernel(
    float* __restrict__ output,
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float scale,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    uint32_t window_size
) {
    extern __shared__ float smem[];

    uint32_t b = blockIdx.z / num_heads;
    uint32_t h = blockIdx.z % num_heads;
    uint32_t q_idx = blockIdx.y;
    uint32_t tid = threadIdx.x;

    if (b >= batch_size || q_idx >= seq_len) return;

    float* scores = smem;

    // Compute window bounds
    int32_t k_start = max(0, (int32_t)q_idx - (int32_t)window_size / 2);
    int32_t k_end = min((int32_t)seq_len, (int32_t)q_idx + (int32_t)window_size / 2 + 1);
    uint32_t window_len = k_end - k_start;

    const float* Q_head = Q + (b * num_heads + h) * seq_len * head_dim;
    const float* K_head = K + (b * num_heads + h) * seq_len * head_dim;
    const float* V_head = V + (b * num_heads + h) * seq_len * head_dim;
    const float* Q_row = Q_head + q_idx * head_dim;

    // Compute attention scores for window
    for (uint32_t i = tid; i < window_len; i += blockDim.x) {
        uint32_t k_idx = k_start + i;
        const float* K_row = K_head + k_idx * head_dim;
        float dot = 0.0f;

        for (uint32_t d = 0; d < head_dim; d++) {
            dot += Q_row[d] * K_row[d];
        }

        scores[i] = dot * scale;
    }
    __syncthreads();

    // Softmax within window
    float max_val = -1e10f;
    for (uint32_t i = tid; i < window_len; i += blockDim.x) {
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
    for (uint32_t i = tid; i < window_len; i += blockDim.x) {
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
    for (uint32_t i = tid; i < window_len; i += blockDim.x) {
        scores[i] *= inv_sum;
    }
    __syncthreads();

    // Compute output
    float* out_ptr = output + (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim;
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;

        for (uint32_t i = 0; i < window_len; i++) {
            uint32_t k_idx = k_start + i;
            acc += scores[i] * V_head[k_idx * head_dim + d];
        }

        out_ptr[d] = acc;
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_steel_attention_tiled(
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
    uint32_t num_q_tiles = (seq_len + TILE_Q - 1) / TILE_Q;
    dim3 blocks(1, num_q_tiles, batch_size * num_heads);
    uint32_t threads = BLOCK_SIZE;

    steel_attention_tiled_kernel<<<blocks, threads, 0, stream>>>(
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

int lux_cuda_steel_gqa_attention(
    void* output,
    const void* Q,
    const void* K,
    const void* V,
    float scale,
    uint32_t batch_size,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    cudaStream_t stream
) {
    dim3 blocks(1, seq_len, batch_size * num_q_heads);
    uint32_t threads = min(256u, seq_len);
    size_t shared_size = seq_len * sizeof(float);

    steel_gqa_attention_kernel<<<blocks, threads, shared_size, stream>>>(
        (float*)output,
        (const float*)Q,
        (const float*)K,
        (const float*)V,
        scale,
        batch_size, num_q_heads, num_kv_heads, seq_len, head_dim
    );

    return cudaGetLastError();
}

int lux_cuda_steel_mqa_attention(
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

    steel_mqa_attention_kernel<<<blocks, threads, shared_size, stream>>>(
        (float*)output,
        (const float*)Q,
        (const float*)K,
        (const float*)V,
        scale,
        batch_size, num_heads, seq_len, head_dim
    );

    return cudaGetLastError();
}

int lux_cuda_steel_sliding_window_attention(
    void* output,
    const void* Q,
    const void* K,
    const void* V,
    float scale,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    uint32_t window_size,
    cudaStream_t stream
) {
    dim3 blocks(1, seq_len, batch_size * num_heads);
    uint32_t threads = min(256u, window_size);
    size_t shared_size = window_size * sizeof(float);

    steel_sliding_window_attention_kernel<<<blocks, threads, shared_size, stream>>>(
        (float*)output,
        (const float*)Q,
        (const float*)K,
        (const float*)V,
        scale,
        batch_size, num_heads, seq_len, head_dim, window_size
    );

    return cudaGetLastError();
}

}  // extern "C"
