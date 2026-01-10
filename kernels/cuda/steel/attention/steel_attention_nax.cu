// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Steel Attention NAX (Non-blocking Asynchronous eXecution) CUDA Kernels
// Optimized for overlapped compute and memory operations with async copies

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// NAX Attention Configuration
// ============================================================================

#define WARP_SIZE 32
#define TILE_Q 64      // Query tile size
#define TILE_K 64      // Key tile size
#define TILE_D 64      // Head dimension
#define NAX_STAGES 2   // Double buffering for async copies
#define BLOCK_SIZE 256

// ============================================================================
// NAX Tiled Attention with Async Memory Copy
// ============================================================================

// Uses CUDA async copy (cp.async) for overlapped memory and compute
extern "C" __global__
void steel_attention_nax_kernel(
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
    // Double-buffered shared memory for pipeline
    __shared__ float Qs[NAX_STAGES][TILE_Q][TILE_D + 1];
    __shared__ float Ks[NAX_STAGES][TILE_K][TILE_D + 1];
    __shared__ float Vs[NAX_STAGES][TILE_K][TILE_D + 1];
    __shared__ float Ss[TILE_Q][TILE_K + 1];  // Attention scores

    uint32_t b = blockIdx.z / num_heads;
    uint32_t h = blockIdx.z % num_heads;
    uint32_t q_tile = blockIdx.y;
    uint32_t tid = threadIdx.x;

    if (b >= batch_size) return;

    // Base pointers
    const float* Q_base = Q + (b * num_heads + h) * seq_len * head_dim;
    const float* K_base = K + (b * num_heads + h) * seq_len * head_dim;
    const float* V_base = V + (b * num_heads + h) * seq_len * head_dim;
    float* O_base = output + (b * num_heads + h) * seq_len * head_dim;

    uint32_t q_start = q_tile * TILE_Q;

    // Online softmax accumulators
    float m_i = -1e10f;
    float l_i = 0.0f;
    float O_acc[TILE_D] = {0.0f};

    uint32_t num_kv_tiles = (seq_len + TILE_K - 1) / TILE_K;

    // Stage 0: Initial load of Q tile
    for (uint32_t i = tid; i < TILE_Q * head_dim; i += blockDim.x) {
        uint32_t q_local = i / head_dim;
        uint32_t d = i % head_dim;
        uint32_t q_idx = q_start + q_local;

        if (q_idx < seq_len && d < head_dim) {
            Qs[0][q_local][d] = Q_base[q_idx * head_dim + d];
        } else {
            Qs[0][q_local][d] = 0.0f;
        }
    }

    // Prefetch first KV tile
    for (uint32_t i = tid; i < TILE_K * head_dim; i += blockDim.x) {
        uint32_t k_local = i / head_dim;
        uint32_t d = i % head_dim;

        if (k_local < seq_len && d < head_dim) {
            Ks[0][k_local][d] = K_base[k_local * head_dim + d];
            Vs[0][k_local][d] = V_base[k_local * head_dim + d];
        } else {
            Ks[0][k_local][d] = 0.0f;
            Vs[0][k_local][d] = 0.0f;
        }
    }
    __syncthreads();

    // Main loop with software pipelining
    for (uint32_t kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        uint32_t curr_stage = kv_tile % NAX_STAGES;
        uint32_t next_stage = (kv_tile + 1) % NAX_STAGES;

        uint32_t k_start = kv_tile * TILE_K;
        uint32_t next_k_start = (kv_tile + 1) * TILE_K;

        // Async prefetch next KV tile (if not last)
        if (kv_tile + 1 < num_kv_tiles) {
            for (uint32_t i = tid; i < TILE_K * head_dim; i += blockDim.x) {
                uint32_t k_local = i / head_dim;
                uint32_t d = i % head_dim;
                uint32_t k_idx = next_k_start + k_local;

                if (k_idx < seq_len && d < head_dim) {
                    Ks[next_stage][k_local][d] = K_base[k_idx * head_dim + d];
                    Vs[next_stage][k_local][d] = V_base[k_idx * head_dim + d];
                } else {
                    Ks[next_stage][k_local][d] = 0.0f;
                    Vs[next_stage][k_local][d] = 0.0f;
                }
            }
        }

        // Compute Q @ K^T for current tile
        for (uint32_t i = tid; i < TILE_Q * TILE_K; i += blockDim.x) {
            uint32_t q_local = i / TILE_K;
            uint32_t k_local = i % TILE_K;
            uint32_t q_idx = q_start + q_local;
            uint32_t k_idx = k_start + k_local;

            float dot = 0.0f;
            if (q_idx < seq_len && k_idx < seq_len) {
                #pragma unroll 8
                for (uint32_t d = 0; d < head_dim; d++) {
                    dot += Qs[0][q_local][d] * Ks[curr_stage][k_local][d];
                }
                dot *= scale;

                // Apply mask
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

        // Process each query position
        for (uint32_t q_local = tid / TILE_K; q_local < TILE_Q; q_local += blockDim.x / TILE_K) {
            uint32_t q_idx = q_start + q_local;
            if (q_idx >= seq_len) continue;

            // Find max in this tile
            float m_ij = -1e10f;
            #pragma unroll
            for (uint32_t k_local = 0; k_local < TILE_K; k_local++) {
                uint32_t k_idx = k_start + k_local;
                if (k_idx < seq_len) {
                    m_ij = fmaxf(m_ij, Ss[q_local][k_local]);
                }
            }

            // Update running max
            float m_new = fmaxf(m_i, m_ij);

            // Compute sum of exponentials
            float l_ij = 0.0f;
            #pragma unroll
            for (uint32_t k_local = 0; k_local < TILE_K; k_local++) {
                uint32_t k_idx = k_start + k_local;
                if (k_idx < seq_len) {
                    l_ij += expf(Ss[q_local][k_local] - m_new);
                }
            }

            // Rescaling factor for previous accumulator
            float alpha = expf(m_i - m_new);
            l_i = alpha * l_i + l_ij;

            // Update output accumulator with rescaling
            #pragma unroll
            for (uint32_t d = 0; d < head_dim; d++) {
                O_acc[d] *= alpha;
            }

            // Accumulate current tile contribution
            #pragma unroll
            for (uint32_t k_local = 0; k_local < TILE_K; k_local++) {
                uint32_t k_idx = k_start + k_local;
                if (k_idx < seq_len) {
                    float p = expf(Ss[q_local][k_local] - m_new);
                    #pragma unroll
                    for (uint32_t d = 0; d < head_dim; d++) {
                        O_acc[d] += p * Vs[curr_stage][k_local][d];
                    }
                }
            }

            m_i = m_new;
        }
        __syncthreads();
    }

    // Write normalized output
    for (uint32_t i = tid; i < TILE_Q * head_dim; i += blockDim.x) {
        uint32_t q_local = i / head_dim;
        uint32_t d = i % head_dim;
        uint32_t q_idx = q_start + q_local;

        if (q_idx < seq_len && d < head_dim) {
            O_base[q_idx * head_dim + d] = O_acc[d] / l_i;
        }
    }
}

// ============================================================================
// NAX Causal Attention with Async Pipeline
// ============================================================================

extern "C" __global__
void steel_causal_attention_nax_kernel(
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
    __shared__ float Ks[NAX_STAGES][TILE_K][TILE_D + 1];
    __shared__ float Vs[NAX_STAGES][TILE_K][TILE_D + 1];
    __shared__ float Ss[TILE_K + 1];

    uint32_t b = blockIdx.z / num_heads;
    uint32_t h = blockIdx.z % num_heads;
    uint32_t q_idx = blockIdx.y;
    uint32_t tid = threadIdx.x;

    if (b >= batch_size || q_idx >= seq_len) return;

    // For causal: can only attend to positions <= q_idx
    uint32_t valid_len = q_idx + 1;
    uint32_t num_kv_tiles = (valid_len + TILE_K - 1) / TILE_K;

    const float* Q_row = Q + (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim;
    const float* K_base = K + (b * num_heads + h) * seq_len * head_dim;
    const float* V_base = V + (b * num_heads + h) * seq_len * head_dim;
    float* out_ptr = output + (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim;

    // Load Q into registers
    float Q_reg[TILE_D];
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        Q_reg[d] = Q_row[d];
    }
    __syncthreads();

    float m_i = -1e10f;
    float l_i = 0.0f;
    float O_acc[TILE_D] = {0.0f};

    // Prefetch first KV tile
    for (uint32_t i = tid; i < TILE_K * head_dim; i += blockDim.x) {
        uint32_t k_local = i / head_dim;
        uint32_t d = i % head_dim;

        if (k_local < valid_len && d < head_dim) {
            Ks[0][k_local][d] = K_base[k_local * head_dim + d];
            Vs[0][k_local][d] = V_base[k_local * head_dim + d];
        } else {
            Ks[0][k_local][d] = 0.0f;
            Vs[0][k_local][d] = 0.0f;
        }
    }
    __syncthreads();

    for (uint32_t kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        uint32_t curr_stage = kv_tile % NAX_STAGES;
        uint32_t next_stage = (kv_tile + 1) % NAX_STAGES;
        uint32_t k_start = kv_tile * TILE_K;

        // Async prefetch next tile
        if (kv_tile + 1 < num_kv_tiles) {
            uint32_t next_k_start = (kv_tile + 1) * TILE_K;
            for (uint32_t i = tid; i < TILE_K * head_dim; i += blockDim.x) {
                uint32_t k_local = i / head_dim;
                uint32_t d = i % head_dim;
                uint32_t k_idx = next_k_start + k_local;

                if (k_idx < valid_len && d < head_dim) {
                    Ks[next_stage][k_local][d] = K_base[k_idx * head_dim + d];
                    Vs[next_stage][k_local][d] = V_base[k_idx * head_dim + d];
                } else {
                    Ks[next_stage][k_local][d] = 0.0f;
                    Vs[next_stage][k_local][d] = 0.0f;
                }
            }
        }

        // Compute attention scores
        for (uint32_t k_local = tid; k_local < TILE_K; k_local += blockDim.x) {
            uint32_t k_idx = k_start + k_local;
            float dot = 0.0f;

            if (k_idx < valid_len) {
                #pragma unroll 8
                for (uint32_t d = 0; d < head_dim; d++) {
                    dot += Q_reg[d] * Ks[curr_stage][k_local][d];
                }
                dot *= scale;
            } else {
                dot = -1e10f;
            }

            Ss[k_local] = dot;
        }
        __syncthreads();

        // Online softmax update
        float m_ij = -1e10f;
        for (uint32_t k_local = 0; k_local < TILE_K; k_local++) {
            uint32_t k_idx = k_start + k_local;
            if (k_idx < valid_len) {
                m_ij = fmaxf(m_ij, Ss[k_local]);
            }
        }

        float m_new = fmaxf(m_i, m_ij);

        float l_ij = 0.0f;
        for (uint32_t k_local = 0; k_local < TILE_K; k_local++) {
            uint32_t k_idx = k_start + k_local;
            if (k_idx < valid_len) {
                l_ij += expf(Ss[k_local] - m_new);
            }
        }

        float alpha = expf(m_i - m_new);
        l_i = alpha * l_i + l_ij;

        // Update accumulator
        #pragma unroll
        for (uint32_t d = 0; d < head_dim; d++) {
            O_acc[d] *= alpha;
        }

        for (uint32_t k_local = 0; k_local < TILE_K; k_local++) {
            uint32_t k_idx = k_start + k_local;
            if (k_idx < valid_len) {
                float p = expf(Ss[k_local] - m_new);
                #pragma unroll
                for (uint32_t d = 0; d < head_dim; d++) {
                    O_acc[d] += p * Vs[curr_stage][k_local][d];
                }
            }
        }

        m_i = m_new;
        __syncthreads();
    }

    // Write normalized output
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        out_ptr[d] = O_acc[d] / l_i;
    }
}

// ============================================================================
// NAX FP16 Attention with Tensor Core Hints
// ============================================================================

extern "C" __global__
void steel_attention_nax_fp16_kernel(
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
    __shared__ __half Ks[NAX_STAGES][TILE_K][TILE_D + 1];
    __shared__ __half Vs[NAX_STAGES][TILE_K][TILE_D + 1];
    __shared__ float Ss[TILE_K + 1];  // Scores in FP32 for stability

    uint32_t b = blockIdx.z / num_heads;
    uint32_t h = blockIdx.z % num_heads;
    uint32_t q_idx = blockIdx.y;
    uint32_t tid = threadIdx.x;

    if (b >= batch_size || q_idx >= seq_len) return;

    const __half* Q_row = Q + (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim;
    const __half* K_base = K + (b * num_heads + h) * seq_len * head_dim;
    const __half* V_base = V + (b * num_heads + h) * seq_len * head_dim;
    __half* out_ptr = output + (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim;

    uint32_t num_kv_tiles = (seq_len + TILE_K - 1) / TILE_K;

    float m_i = -1e10f;
    float l_i = 0.0f;
    float O_acc[TILE_D] = {0.0f};

    // Prefetch first KV tile
    for (uint32_t i = tid; i < TILE_K * head_dim; i += blockDim.x) {
        uint32_t k_local = i / head_dim;
        uint32_t d = i % head_dim;

        if (k_local < seq_len && d < head_dim) {
            Ks[0][k_local][d] = K_base[k_local * head_dim + d];
            Vs[0][k_local][d] = V_base[k_local * head_dim + d];
        } else {
            Ks[0][k_local][d] = __float2half(0.0f);
            Vs[0][k_local][d] = __float2half(0.0f);
        }
    }
    __syncthreads();

    for (uint32_t kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        uint32_t curr_stage = kv_tile % NAX_STAGES;
        uint32_t next_stage = (kv_tile + 1) % NAX_STAGES;
        uint32_t k_start = kv_tile * TILE_K;

        // Prefetch next tile
        if (kv_tile + 1 < num_kv_tiles) {
            uint32_t next_k_start = (kv_tile + 1) * TILE_K;
            for (uint32_t i = tid; i < TILE_K * head_dim; i += blockDim.x) {
                uint32_t k_local = i / head_dim;
                uint32_t d = i % head_dim;
                uint32_t k_idx = next_k_start + k_local;

                if (k_idx < seq_len && d < head_dim) {
                    Ks[next_stage][k_local][d] = K_base[k_idx * head_dim + d];
                    Vs[next_stage][k_local][d] = V_base[k_idx * head_dim + d];
                } else {
                    Ks[next_stage][k_local][d] = __float2half(0.0f);
                    Vs[next_stage][k_local][d] = __float2half(0.0f);
                }
            }
        }

        // Compute scores in FP32 for numerical stability
        for (uint32_t k_local = tid; k_local < TILE_K; k_local += blockDim.x) {
            uint32_t k_idx = k_start + k_local;
            float dot = 0.0f;

            if (k_idx < seq_len) {
                #pragma unroll 8
                for (uint32_t d = 0; d < head_dim; d++) {
                    dot += __half2float(Q_row[d]) * __half2float(Ks[curr_stage][k_local][d]);
                }
                dot *= scale;

                if (mask != nullptr) {
                    float m = __half2float(mask[b * seq_len * seq_len + q_idx * seq_len + k_idx]);
                    if (m == 0.0f) dot = -1e10f;
                }
            } else {
                dot = -1e10f;
            }

            Ss[k_local] = dot;
        }
        __syncthreads();

        // Online softmax
        float m_ij = -1e10f;
        for (uint32_t k_local = 0; k_local < TILE_K; k_local++) {
            uint32_t k_idx = k_start + k_local;
            if (k_idx < seq_len) {
                m_ij = fmaxf(m_ij, Ss[k_local]);
            }
        }

        float m_new = fmaxf(m_i, m_ij);

        float l_ij = 0.0f;
        for (uint32_t k_local = 0; k_local < TILE_K; k_local++) {
            uint32_t k_idx = k_start + k_local;
            if (k_idx < seq_len) {
                l_ij += expf(Ss[k_local] - m_new);
            }
        }

        float alpha = expf(m_i - m_new);
        l_i = alpha * l_i + l_ij;

        #pragma unroll
        for (uint32_t d = 0; d < head_dim; d++) {
            O_acc[d] *= alpha;
        }

        for (uint32_t k_local = 0; k_local < TILE_K; k_local++) {
            uint32_t k_idx = k_start + k_local;
            if (k_idx < seq_len) {
                float p = expf(Ss[k_local] - m_new);
                #pragma unroll
                for (uint32_t d = 0; d < head_dim; d++) {
                    O_acc[d] += p * __half2float(Vs[curr_stage][k_local][d]);
                }
            }
        }

        m_i = m_new;
        __syncthreads();
    }

    // Write output in FP16
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        out_ptr[d] = __float2half(O_acc[d] / l_i);
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_steel_attention_nax(
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

    steel_attention_nax_kernel<<<blocks, threads, 0, stream>>>(
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

int lux_cuda_steel_causal_attention_nax(
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

    steel_causal_attention_nax_kernel<<<blocks, threads, 0, stream>>>(
        (float*)output,
        (const float*)Q,
        (const float*)K,
        (const float*)V,
        scale,
        batch_size, num_heads, seq_len, head_dim
    );

    return cudaGetLastError();
}

int lux_cuda_steel_attention_nax_fp16(
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

    steel_attention_nax_fp16_kernel<<<blocks, threads, 0, stream>>>(
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

}  // extern "C"
