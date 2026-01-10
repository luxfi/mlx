// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Flash Attention CUDA Kernels
// Memory-efficient attention with O(N) memory instead of O(N^2)

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// Flash Attention Configuration
// ============================================================================

#define WARP_SIZE 32
#define BLOCK_M 64   // Query tile size
#define BLOCK_N 64   // Key/Value tile size
#define BLOCK_K 64   // Head dimension tile (if needed)

// ============================================================================
// Flash Attention Forward Pass
// ============================================================================

// Tiled attention with online softmax
// Key insight: maintain running max and sum for stable softmax without materializing full attention matrix
extern "C" __global__
void flash_attention_forward_kernel(
    float* __restrict__ output,         // [B, H, S, D]
    float* __restrict__ lse,            // [B, H, S] log-sum-exp for backward
    const float* __restrict__ Q,        // [B, H, S, D]
    const float* __restrict__ K,        // [B, H, S, D]
    const float* __restrict__ V,        // [B, H, S, D]
    float scale,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    bool is_causal
) {
    extern __shared__ float smem[];

    // Shared memory layout:
    // [0, BLOCK_N * head_dim): K tile
    // [BLOCK_N * head_dim, BLOCK_N * head_dim + BLOCK_N * head_dim): V tile
    // [2 * BLOCK_N * head_dim, 2 * BLOCK_N * head_dim + BLOCK_M * BLOCK_N): Scores

    float* K_tile = smem;
    float* V_tile = smem + BLOCK_N * head_dim;
    float* S_tile = smem + 2 * BLOCK_N * head_dim;

    uint32_t b = blockIdx.z / num_heads;
    uint32_t h = blockIdx.z % num_heads;
    uint32_t q_tile_idx = blockIdx.y;

    uint32_t tid = threadIdx.x;
    uint32_t warp_id = tid / WARP_SIZE;
    uint32_t lane_id = tid % WARP_SIZE;

    // Base pointers
    const float* Q_base = Q + (b * num_heads + h) * seq_len * head_dim;
    const float* K_base = K + (b * num_heads + h) * seq_len * head_dim;
    const float* V_base = V + (b * num_heads + h) * seq_len * head_dim;
    float* O_base = output + (b * num_heads + h) * seq_len * head_dim;

    // Query tile range
    uint32_t q_start = q_tile_idx * BLOCK_M;
    uint32_t q_end = min(q_start + BLOCK_M, seq_len);

    // Initialize output accumulator and running statistics
    float O_acc[BLOCK_M];
    float m_prev[BLOCK_M];  // Running max
    float l_prev[BLOCK_M];  // Running sum

    for (int i = 0; i < BLOCK_M; i++) {
        O_acc[i] = 0.0f;
        m_prev[i] = -INFINITY;
        l_prev[i] = 0.0f;
    }

    // Number of KV tiles
    uint32_t num_kv_tiles = (seq_len + BLOCK_N - 1) / BLOCK_N;
    uint32_t kv_tile_end = is_causal ? min(num_kv_tiles, (q_start + BLOCK_M + BLOCK_N - 1) / BLOCK_N) : num_kv_tiles;

    // Iterate over KV tiles
    for (uint32_t kv_tile = 0; kv_tile < kv_tile_end; kv_tile++) {
        uint32_t k_start = kv_tile * BLOCK_N;
        uint32_t k_end = min(k_start + BLOCK_N, seq_len);

        // Load K tile to shared memory
        for (uint32_t i = tid; i < BLOCK_N * head_dim; i += blockDim.x) {
            uint32_t k_idx = k_start + i / head_dim;
            uint32_t d_idx = i % head_dim;
            if (k_idx < seq_len) {
                K_tile[i] = K_base[k_idx * head_dim + d_idx];
            } else {
                K_tile[i] = 0.0f;
            }
        }

        // Load V tile to shared memory
        for (uint32_t i = tid; i < BLOCK_N * head_dim; i += blockDim.x) {
            uint32_t v_idx = k_start + i / head_dim;
            uint32_t d_idx = i % head_dim;
            if (v_idx < seq_len) {
                V_tile[i] = V_base[v_idx * head_dim + d_idx];
            } else {
                V_tile[i] = 0.0f;
            }
        }
        __syncthreads();

        // Compute attention scores for this tile: S = Q @ K^T * scale
        for (uint32_t q_local = tid / BLOCK_N; q_local < BLOCK_M; q_local += blockDim.x / BLOCK_N) {
            uint32_t q_idx = q_start + q_local;
            if (q_idx >= seq_len) continue;

            const float* Q_row = Q_base + q_idx * head_dim;

            for (uint32_t k_local = tid % BLOCK_N; k_local < BLOCK_N; k_local += 1) {
                uint32_t k_idx = k_start + k_local;

                // Causal mask
                if (is_causal && k_idx > q_idx) {
                    S_tile[q_local * BLOCK_N + k_local] = -INFINITY;
                } else if (k_idx < seq_len) {
                    float dot = 0.0f;
                    for (uint32_t d = 0; d < head_dim; d++) {
                        dot += Q_row[d] * K_tile[k_local * head_dim + d];
                    }
                    S_tile[q_local * BLOCK_N + k_local] = dot * scale;
                } else {
                    S_tile[q_local * BLOCK_N + k_local] = -INFINITY;
                }
            }
        }
        __syncthreads();

        // Online softmax and accumulate output
        for (uint32_t q_local = tid; q_local < BLOCK_M; q_local += blockDim.x) {
            uint32_t q_idx = q_start + q_local;
            if (q_idx >= seq_len) continue;

            // Find new max
            float m_new = m_prev[q_local];
            for (uint32_t k_local = 0; k_local < BLOCK_N; k_local++) {
                m_new = fmaxf(m_new, S_tile[q_local * BLOCK_N + k_local]);
            }

            // Compute new sum with rescaling
            float l_new = l_prev[q_local] * expf(m_prev[q_local] - m_new);

            for (uint32_t k_local = 0; k_local < BLOCK_N; k_local++) {
                float p = expf(S_tile[q_local * BLOCK_N + k_local] - m_new);
                l_new += p;
            }

            // Rescale previous output accumulator
            float rescale = expf(m_prev[q_local] - m_new) * l_prev[q_local] / l_new;
            O_acc[q_local] *= rescale;

            // Accumulate new contribution
            for (uint32_t k_local = 0; k_local < BLOCK_N; k_local++) {
                float p = expf(S_tile[q_local * BLOCK_N + k_local] - m_new) / l_new;

                for (uint32_t d = 0; d < head_dim; d++) {
                    O_acc[q_local] += p * V_tile[k_local * head_dim + d];
                }
            }

            m_prev[q_local] = m_new;
            l_prev[q_local] = l_new;
        }
        __syncthreads();
    }

    // Write output and LSE
    for (uint32_t q_local = tid; q_local < BLOCK_M; q_local += blockDim.x) {
        uint32_t q_idx = q_start + q_local;
        if (q_idx < seq_len) {
            for (uint32_t d = 0; d < head_dim; d++) {
                O_base[q_idx * head_dim + d] = O_acc[q_local];
            }

            if (lse != nullptr) {
                lse[(b * num_heads + h) * seq_len + q_idx] = m_prev[q_local] + logf(l_prev[q_local]);
            }
        }
    }
}

// ============================================================================
// Flash Attention FP16 (Tensor Core Optimized)
// ============================================================================

extern "C" __global__
void flash_attention_forward_fp16_kernel(
    __half* __restrict__ output,
    float* __restrict__ lse,
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
    // Similar structure to FP32 but using half precision
    // In production, this would use WMMA/MMA instructions for tensor cores

    extern __shared__ __half smem_half[];

    uint32_t b = blockIdx.z / num_heads;
    uint32_t h = blockIdx.z % num_heads;
    uint32_t q_tile_idx = blockIdx.y;
    uint32_t tid = threadIdx.x;

    const __half* Q_base = Q + (b * num_heads + h) * seq_len * head_dim;
    const __half* K_base = K + (b * num_heads + h) * seq_len * head_dim;
    const __half* V_base = V + (b * num_heads + h) * seq_len * head_dim;
    __half* O_base = output + (b * num_heads + h) * seq_len * head_dim;

    uint32_t q_start = q_tile_idx * BLOCK_M;

    // FP32 accumulators for numerical stability
    float O_acc = 0.0f;
    float m_prev = -INFINITY;
    float l_prev = 0.0f;

    uint32_t num_kv_tiles = (seq_len + BLOCK_N - 1) / BLOCK_N;

    for (uint32_t kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        uint32_t k_start = kv_tile * BLOCK_N;

        // Load and compute (simplified)
        for (uint32_t q_local = tid; q_local < BLOCK_M && q_start + q_local < seq_len; q_local += blockDim.x) {
            uint32_t q_idx = q_start + q_local;
            const __half* Q_row = Q_base + q_idx * head_dim;

            float m_new = m_prev;

            for (uint32_t k_local = 0; k_local < BLOCK_N && k_start + k_local < seq_len; k_local++) {
                uint32_t k_idx = k_start + k_local;

                if (is_causal && k_idx > q_idx) continue;

                const __half* K_row = K_base + k_idx * head_dim;

                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) {
                    dot += __half2float(Q_row[d]) * __half2float(K_row[d]);
                }

                float score = dot * scale;
                m_new = fmaxf(m_new, score);
            }

            float l_new = l_prev * expf(m_prev - m_new);

            for (uint32_t k_local = 0; k_local < BLOCK_N && k_start + k_local < seq_len; k_local++) {
                uint32_t k_idx = k_start + k_local;

                if (is_causal && k_idx > q_idx) continue;

                const __half* K_row = K_base + k_idx * head_dim;
                const __half* V_row = V_base + k_idx * head_dim;

                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) {
                    dot += __half2float(Q_row[d]) * __half2float(K_row[d]);
                }

                float p = expf(dot * scale - m_new);
                l_new += p;

                for (uint32_t d = 0; d < head_dim; d++) {
                    O_acc = O_acc * expf(m_prev - m_new) + p * __half2float(V_row[d]);
                }
            }

            m_prev = m_new;
            l_prev = l_new;
        }
    }

    // Write output
    for (uint32_t q_local = tid; q_local < BLOCK_M && q_start + q_local < seq_len; q_local += blockDim.x) {
        uint32_t q_idx = q_start + q_local;

        for (uint32_t d = 0; d < head_dim; d++) {
            O_base[q_idx * head_dim + d] = __float2half(O_acc / l_prev);
        }

        if (lse != nullptr) {
            lse[(b * num_heads + h) * seq_len + q_idx] = m_prev + logf(l_prev);
        }
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_flash_attention_forward(
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
    uint32_t num_q_tiles = (seq_len + BLOCK_M - 1) / BLOCK_M;
    dim3 blocks(1, num_q_tiles, batch_size * num_heads);
    uint32_t threads = 256;
    size_t shared_size = (2 * BLOCK_N * head_dim + BLOCK_M * BLOCK_N) * sizeof(float);

    flash_attention_forward_kernel<<<blocks, threads, shared_size, stream>>>(
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

int lux_cuda_flash_attention_forward_fp16(
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
    uint32_t num_q_tiles = (seq_len + BLOCK_M - 1) / BLOCK_M;
    dim3 blocks(1, num_q_tiles, batch_size * num_heads);
    uint32_t threads = 256;
    size_t shared_size = (2 * BLOCK_N * head_dim) * sizeof(__half);

    flash_attention_forward_fp16_kernel<<<blocks, threads, shared_size, stream>>>(
        (__half*)output,
        (float*)lse,
        (const __half*)Q,
        (const __half*)K,
        (const __half*)V,
        scale,
        batch_size, num_heads, seq_len, head_dim,
        is_causal
    );

    return cudaGetLastError();
}

}  // extern "C"
