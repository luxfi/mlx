// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Rotary Position Embedding (RoPE) CUDA Kernels
// Implements the rotary position embedding from "RoFormer: Enhanced Transformer
// with Rotary Position Embedding" (Su et al., 2021)
//
// For each pair of dimensions (2i, 2i+1):
//   x'[2i]   = x[2i] * cos(theta) - x[2i+1] * sin(theta)
//   x'[2i+1] = x[2i] * sin(theta) + x[2i+1] * cos(theta)
// where theta = position * (base ^ (-2i/d))

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// ============================================================================
// Configuration
// ============================================================================

#define WARP_SIZE 32
#define BLOCK_SIZE 256

// ============================================================================
// RoPE Forward (FP32)
// Input shape: [batch_size, seq_len, num_heads, head_dim]
// ============================================================================

extern "C" __global__
void lux_rope_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ cos_cache,  // [max_seq_len, head_dim/2]
    const float* __restrict__ sin_cache,  // [max_seq_len, head_dim/2]
    uint32_t batch_size,
    uint32_t seq_len,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t offset  // Position offset for KV cache
) {
    // Grid: (num_heads, seq_len, batch_size)
    uint32_t h = blockIdx.x;
    uint32_t s = blockIdx.y;
    uint32_t b = blockIdx.z;
    uint32_t tid = threadIdx.x;

    if (b >= batch_size || s >= seq_len || h >= num_heads) return;

    uint32_t pos = s + offset;  // Actual position in sequence
    uint32_t half_dim = head_dim / 2;

    // Input/output pointer for this position
    const float* x = input + ((b * seq_len + s) * num_heads + h) * head_dim;
    float* y = output + ((b * seq_len + s) * num_heads + h) * head_dim;

    // Each thread handles one pair of dimensions
    for (uint32_t i = tid; i < half_dim; i += blockDim.x) {
        float cos_val = cos_cache[pos * half_dim + i];
        float sin_val = sin_cache[pos * half_dim + i];

        float x0 = x[i];
        float x1 = x[i + half_dim];

        // Apply rotation
        y[i] = x0 * cos_val - x1 * sin_val;
        y[i + half_dim] = x0 * sin_val + x1 * cos_val;
    }
}

// ============================================================================
// RoPE Forward with Interleaved Layout
// Some implementations interleave dimensions: [x0, x1, x0, x1, ...]
// ============================================================================

extern "C" __global__
void lux_rope_interleaved_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    uint32_t batch_size,
    uint32_t seq_len,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t offset
) {
    uint32_t h = blockIdx.x;
    uint32_t s = blockIdx.y;
    uint32_t b = blockIdx.z;
    uint32_t tid = threadIdx.x;

    if (b >= batch_size || s >= seq_len || h >= num_heads) return;

    uint32_t pos = s + offset;
    uint32_t half_dim = head_dim / 2;

    const float* x = input + ((b * seq_len + s) * num_heads + h) * head_dim;
    float* y = output + ((b * seq_len + s) * num_heads + h) * head_dim;

    for (uint32_t i = tid; i < half_dim; i += blockDim.x) {
        float cos_val = cos_cache[pos * half_dim + i];
        float sin_val = sin_cache[pos * half_dim + i];

        // Interleaved layout: pairs are adjacent
        uint32_t idx0 = 2 * i;
        uint32_t idx1 = 2 * i + 1;

        float x0 = x[idx0];
        float x1 = x[idx1];

        y[idx0] = x0 * cos_val - x1 * sin_val;
        y[idx1] = x0 * sin_val + x1 * cos_val;
    }
}

// ============================================================================
// RoPE Forward (FP16)
// ============================================================================

extern "C" __global__
void lux_rope_f16_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    uint32_t batch_size,
    uint32_t seq_len,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t offset
) {
    uint32_t h = blockIdx.x;
    uint32_t s = blockIdx.y;
    uint32_t b = blockIdx.z;
    uint32_t tid = threadIdx.x;

    if (b >= batch_size || s >= seq_len || h >= num_heads) return;

    uint32_t pos = s + offset;
    uint32_t half_dim = head_dim / 2;

    const __half* x = input + ((b * seq_len + s) * num_heads + h) * head_dim;
    __half* y = output + ((b * seq_len + s) * num_heads + h) * head_dim;

    for (uint32_t i = tid; i < half_dim; i += blockDim.x) {
        float cos_val = cos_cache[pos * half_dim + i];
        float sin_val = sin_cache[pos * half_dim + i];

        float x0 = __half2float(x[i]);
        float x1 = __half2float(x[i + half_dim]);

        y[i] = __float2half(x0 * cos_val - x1 * sin_val);
        y[i + half_dim] = __float2half(x0 * sin_val + x1 * cos_val);
    }
}

// ============================================================================
// Compute RoPE Frequencies (Precompute cache)
// freq[i] = base ^ (-2i/d) for i in [0, d/2)
// ============================================================================

extern "C" __global__
void lux_rope_compute_freqs_kernel(
    float* __restrict__ cos_cache,
    float* __restrict__ sin_cache,
    float base,
    uint32_t max_seq_len,
    uint32_t head_dim
) {
    uint32_t pos = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (pos >= max_seq_len) return;

    uint32_t half_dim = head_dim / 2;

    for (uint32_t i = tid; i < half_dim; i += blockDim.x) {
        // theta = pos * base^(-2i/d)
        float freq = 1.0f / powf(base, (float)(2 * i) / (float)head_dim);
        float theta = (float)pos * freq;

        cos_cache[pos * half_dim + i] = cosf(theta);
        sin_cache[pos * half_dim + i] = sinf(theta);
    }
}

// ============================================================================
// RoPE with NTK-Aware Scaling (Extended context)
// Uses scaled base: base_scaled = base * ((scale * n / n_orig) - (scale - 1)) ^ (d / (d - 2))
// ============================================================================

extern "C" __global__
void lux_rope_ntk_scaled_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    float base,
    float scale,
    uint32_t batch_size,
    uint32_t seq_len,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t offset,
    uint32_t original_max_len
) {
    uint32_t h = blockIdx.x;
    uint32_t s = blockIdx.y;
    uint32_t b = blockIdx.z;
    uint32_t tid = threadIdx.x;

    if (b >= batch_size || s >= seq_len || h >= num_heads) return;

    uint32_t pos = s + offset;
    uint32_t half_dim = head_dim / 2;

    // NTK-aware scaled base
    float ntk_scale = scale * (float)(seq_len + offset) / (float)original_max_len;
    float base_scaled = base * powf(ntk_scale - (scale - 1.0f), (float)head_dim / (float)(head_dim - 2));

    const float* x = input + ((b * seq_len + s) * num_heads + h) * head_dim;
    float* y = output + ((b * seq_len + s) * num_heads + h) * head_dim;

    for (uint32_t i = tid; i < half_dim; i += blockDim.x) {
        float freq = 1.0f / powf(base_scaled, (float)(2 * i) / (float)head_dim);
        float theta = (float)pos * freq;
        float cos_val = cosf(theta);
        float sin_val = sinf(theta);

        float x0 = x[i];
        float x1 = x[i + half_dim];

        y[i] = x0 * cos_val - x1 * sin_val;
        y[i + half_dim] = x0 * sin_val + x1 * cos_val;
    }
}

// ============================================================================
// RoPE with YaRN (Yet another RoPE extensioN)
// Combines NTK interpolation with attention scaling
// ============================================================================

extern "C" __global__
void lux_rope_yarn_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    float base,
    float scale,
    float mscale,           // Magnitude scale
    float mscale_all_dim,   // Per-dimension scale
    uint32_t batch_size,
    uint32_t seq_len,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t offset,
    uint32_t original_max_len,
    float alpha,            // Interpolation alpha
    float beta              // Interpolation beta
) {
    uint32_t h = blockIdx.x;
    uint32_t s = blockIdx.y;
    uint32_t b = blockIdx.z;
    uint32_t tid = threadIdx.x;

    if (b >= batch_size || s >= seq_len || h >= num_heads) return;

    uint32_t pos = s + offset;
    uint32_t half_dim = head_dim / 2;

    const float* x = input + ((b * seq_len + s) * num_heads + h) * head_dim;
    float* y = output + ((b * seq_len + s) * num_heads + h) * head_dim;

    for (uint32_t i = tid; i < half_dim; i += blockDim.x) {
        // Compute dimension-specific scaling
        float dim_ratio = (float)(2 * i) / (float)head_dim;

        // Ramp function for smooth interpolation
        float ramp = fminf(1.0f, fmaxf(0.0f, (dim_ratio * head_dim - alpha) / (beta - alpha)));

        // Interpolated frequency
        float freq_base = 1.0f / powf(base, dim_ratio);
        float freq_scaled = freq_base / scale;
        float freq = freq_base * (1.0f - ramp) + freq_scaled * ramp;

        float theta = (float)pos * freq;
        float cos_val = cosf(theta) * mscale * mscale_all_dim;
        float sin_val = sinf(theta) * mscale * mscale_all_dim;

        float x0 = x[i];
        float x1 = x[i + half_dim];

        y[i] = x0 * cos_val - x1 * sin_val;
        y[i + half_dim] = x0 * sin_val + x1 * cos_val;
    }
}

// ============================================================================
// RoPE In-Place (for KV cache updates)
// ============================================================================

extern "C" __global__
void lux_rope_inplace_f32_kernel(
    float* __restrict__ x,            // Modified in-place
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    uint32_t batch_size,
    uint32_t seq_len,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t offset
) {
    uint32_t h = blockIdx.x;
    uint32_t s = blockIdx.y;
    uint32_t b = blockIdx.z;
    uint32_t tid = threadIdx.x;

    if (b >= batch_size || s >= seq_len || h >= num_heads) return;

    uint32_t pos = s + offset;
    uint32_t half_dim = head_dim / 2;

    float* ptr = x + ((b * seq_len + s) * num_heads + h) * head_dim;

    for (uint32_t i = tid; i < half_dim; i += blockDim.x) {
        float cos_val = cos_cache[pos * half_dim + i];
        float sin_val = sin_cache[pos * half_dim + i];

        float x0 = ptr[i];
        float x1 = ptr[i + half_dim];

        ptr[i] = x0 * cos_val - x1 * sin_val;
        ptr[i + half_dim] = x0 * sin_val + x1 * cos_val;
    }
}

// ============================================================================
// Fused Q and K RoPE (common in attention)
// ============================================================================

extern "C" __global__
void lux_rope_qk_f32_kernel(
    float* __restrict__ Q_out,
    float* __restrict__ K_out,
    const float* __restrict__ Q_in,
    const float* __restrict__ K_in,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    uint32_t batch_size,
    uint32_t seq_len,
    uint32_t num_heads,
    uint32_t num_kv_heads,  // For GQA/MQA
    uint32_t head_dim,
    uint32_t offset
) {
    uint32_t s = blockIdx.x;
    uint32_t b = blockIdx.y;
    uint32_t tid = threadIdx.x;

    if (b >= batch_size || s >= seq_len) return;

    uint32_t pos = s + offset;
    uint32_t half_dim = head_dim / 2;

    // Apply RoPE to Q (all heads)
    for (uint32_t h = 0; h < num_heads; h++) {
        const float* q_in = Q_in + ((b * seq_len + s) * num_heads + h) * head_dim;
        float* q_out = Q_out + ((b * seq_len + s) * num_heads + h) * head_dim;

        for (uint32_t i = tid; i < half_dim; i += blockDim.x) {
            float cos_val = cos_cache[pos * half_dim + i];
            float sin_val = sin_cache[pos * half_dim + i];

            float x0 = q_in[i];
            float x1 = q_in[i + half_dim];

            q_out[i] = x0 * cos_val - x1 * sin_val;
            q_out[i + half_dim] = x0 * sin_val + x1 * cos_val;
        }
    }

    // Apply RoPE to K (KV heads only)
    for (uint32_t h = 0; h < num_kv_heads; h++) {
        const float* k_in = K_in + ((b * seq_len + s) * num_kv_heads + h) * head_dim;
        float* k_out = K_out + ((b * seq_len + s) * num_kv_heads + h) * head_dim;

        for (uint32_t i = tid; i < half_dim; i += blockDim.x) {
            float cos_val = cos_cache[pos * half_dim + i];
            float sin_val = sin_cache[pos * half_dim + i];

            float x0 = k_in[i];
            float x1 = k_in[i + half_dim];

            k_out[i] = x0 * cos_val - x1 * sin_val;
            k_out[i + half_dim] = x0 * sin_val + x1 * cos_val;
        }
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_rope_f32(
    void* output,
    const void* input,
    const void* cos_cache,
    const void* sin_cache,
    uint32_t batch_size,
    uint32_t seq_len,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t offset,
    cudaStream_t stream
) {
    dim3 blocks(num_heads, seq_len, batch_size);
    uint32_t threads = min((uint32_t)BLOCK_SIZE, head_dim / 2);

    lux_rope_f32_kernel<<<blocks, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        (const float*)cos_cache,
        (const float*)sin_cache,
        batch_size, seq_len, num_heads, head_dim, offset
    );

    return cudaGetLastError();
}

int lux_cuda_rope_interleaved_f32(
    void* output,
    const void* input,
    const void* cos_cache,
    const void* sin_cache,
    uint32_t batch_size,
    uint32_t seq_len,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t offset,
    cudaStream_t stream
) {
    dim3 blocks(num_heads, seq_len, batch_size);
    uint32_t threads = min((uint32_t)BLOCK_SIZE, head_dim / 2);

    lux_rope_interleaved_f32_kernel<<<blocks, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        (const float*)cos_cache,
        (const float*)sin_cache,
        batch_size, seq_len, num_heads, head_dim, offset
    );

    return cudaGetLastError();
}

int lux_cuda_rope_f16(
    void* output,
    const void* input,
    const void* cos_cache,
    const void* sin_cache,
    uint32_t batch_size,
    uint32_t seq_len,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t offset,
    cudaStream_t stream
) {
    dim3 blocks(num_heads, seq_len, batch_size);
    uint32_t threads = min((uint32_t)BLOCK_SIZE, head_dim / 2);

    lux_rope_f16_kernel<<<blocks, threads, 0, stream>>>(
        (__half*)output,
        (const __half*)input,
        (const float*)cos_cache,
        (const float*)sin_cache,
        batch_size, seq_len, num_heads, head_dim, offset
    );

    return cudaGetLastError();
}

int lux_cuda_rope_compute_freqs(
    void* cos_cache,
    void* sin_cache,
    float base,
    uint32_t max_seq_len,
    uint32_t head_dim,
    cudaStream_t stream
) {
    uint32_t threads = min((uint32_t)BLOCK_SIZE, head_dim / 2);

    lux_rope_compute_freqs_kernel<<<max_seq_len, threads, 0, stream>>>(
        (float*)cos_cache,
        (float*)sin_cache,
        base,
        max_seq_len,
        head_dim
    );

    return cudaGetLastError();
}

int lux_cuda_rope_inplace_f32(
    void* x,
    const void* cos_cache,
    const void* sin_cache,
    uint32_t batch_size,
    uint32_t seq_len,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t offset,
    cudaStream_t stream
) {
    dim3 blocks(num_heads, seq_len, batch_size);
    uint32_t threads = min((uint32_t)BLOCK_SIZE, head_dim / 2);

    lux_rope_inplace_f32_kernel<<<blocks, threads, 0, stream>>>(
        (float*)x,
        (const float*)cos_cache,
        (const float*)sin_cache,
        batch_size, seq_len, num_heads, head_dim, offset
    );

    return cudaGetLastError();
}

int lux_cuda_rope_qk_f32(
    void* Q_out,
    void* K_out,
    const void* Q_in,
    const void* K_in,
    const void* cos_cache,
    const void* sin_cache,
    uint32_t batch_size,
    uint32_t seq_len,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t offset,
    cudaStream_t stream
) {
    dim3 blocks(seq_len, batch_size);
    uint32_t threads = min((uint32_t)BLOCK_SIZE, head_dim / 2);

    lux_rope_qk_f32_kernel<<<blocks, threads, 0, stream>>>(
        (float*)Q_out,
        (float*)K_out,
        (const float*)Q_in,
        (const float*)K_in,
        (const float*)cos_cache,
        (const float*)sin_cache,
        batch_size, seq_len, num_heads, num_kv_heads, head_dim, offset
    );

    return cudaGetLastError();
}

}  // extern "C"
