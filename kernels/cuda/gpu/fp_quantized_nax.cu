// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// FP8/FP16 Quantized Neural Attention Operations - CUDA Implementation
// NAX (Neural Attention eXtension) variant with floating-point quantization.
// Optimized for transformer inference with FP8/FP16 mixed precision.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace lux {
namespace cuda {
namespace fp_nax {

// ============================================================================
// Configuration
// ============================================================================

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

// ============================================================================
// FP8 Conversion Utilities (E4M3 format)
// ============================================================================

constexpr int FP8_E4M3_BIAS = 7;
constexpr float FP8_E4M3_MAX = 448.0f;

__device__ __forceinline__
uint8_t float_to_fp8_e4m3(float val) {
    if (isnan(val)) return 0x7F;
    val = fminf(fmaxf(val, -FP8_E4M3_MAX), FP8_E4M3_MAX);

    uint32_t bits = __float_as_uint(val);
    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127;
    uint32_t frac = bits & 0x7FFFFF;

    if (exp == -127 && frac == 0) return (uint8_t)(sign << 7);

    int32_t new_exp = exp + FP8_E4M3_BIAS;
    if (new_exp <= 0) return (uint8_t)(sign << 7);
    if (new_exp >= 15) {
        new_exp = 15;
        frac = 0x700000;
    }

    uint32_t mant = (frac + 0x80000) >> 20;
    if (mant >= 8) {
        mant = 0;
        new_exp++;
        if (new_exp >= 15) { new_exp = 15; mant = 7; }
    }

    return (uint8_t)((sign << 7) | (new_exp << 3) | mant);
}

__device__ __forceinline__
float fp8_e4m3_to_float(uint8_t val) {
    uint32_t sign = (val >> 7) & 0x1;
    uint32_t exp = (val >> 3) & 0xF;
    uint32_t mant = val & 0x7;

    if (exp == 0 && mant == 0) return sign ? -0.0f : 0.0f;
    if (val == 0x7F || val == 0xFF) return nanf("");

    int32_t new_exp = exp - FP8_E4M3_BIAS + 127;
    uint32_t new_mant = mant << 20;
    uint32_t bits = (sign << 31) | (new_exp << 23) | new_mant;
    return __uint_as_float(bits);
}

// ============================================================================
// Quantization Parameters for FP NAX
// ============================================================================

struct FPQuantParams {
    float scale;
    float inv_scale;  // Pre-computed 1.0f / scale
};

struct FPNAXConfig {
    uint32_t batch_size;
    uint32_t num_heads;
    uint32_t seq_len;
    uint32_t head_dim;
    float temperature;
    float q_scale;
    float k_scale;
    float v_scale;
    float out_scale;
};

// ============================================================================
// FP16 Quantized Attention Score Computation
// ============================================================================

extern "C" __global__
void fp16_attention_scores_kernel(
    __half* __restrict__ scores,      // [batch, heads, seq_q, seq_k]
    const __half* __restrict__ Q,     // [batch, heads, seq_q, head_dim]
    const __half* __restrict__ K,     // [batch, heads, seq_k, head_dim]
    float scale,                       // 1/sqrt(head_dim) * temperature
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_q,
    uint32_t seq_k,
    uint32_t head_dim
) {
    uint32_t b = blockIdx.z / num_heads;
    uint32_t h = blockIdx.z % num_heads;
    uint32_t q_idx = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t k_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch_size || q_idx >= seq_q || k_idx >= seq_k) return;

    const __half* q_ptr = Q + (b * num_heads + h) * seq_q * head_dim + q_idx * head_dim;
    const __half* k_ptr = K + (b * num_heads + h) * seq_k * head_dim + k_idx * head_dim;

    float acc = 0.0f;

    // Vectorized dot product using half2
    uint32_t d = 0;
    for (; d + 1 < head_dim; d += 2) {
        half2 q2 = *((half2*)(q_ptr + d));
        half2 k2 = *((half2*)(k_ptr + d));
        acc += __half2float(q2.x) * __half2float(k2.x);
        acc += __half2float(q2.y) * __half2float(k2.y);
    }
    // Handle remaining element
    if (d < head_dim) {
        acc += __half2float(q_ptr[d]) * __half2float(k_ptr[d]);
    }

    acc *= scale;

    uint32_t out_idx = (b * num_heads + h) * seq_q * seq_k + q_idx * seq_k + k_idx;
    scores[out_idx] = __float2half(acc);
}

// ============================================================================
// FP8 Quantized Attention Score Computation
// ============================================================================

extern "C" __global__
void fp8_attention_scores_kernel(
    float* __restrict__ scores,       // [batch, heads, seq_q, seq_k] FP32 for accumulation
    const uint8_t* __restrict__ Q,    // [batch, heads, seq_q, head_dim] FP8 E4M3
    const uint8_t* __restrict__ K,    // [batch, heads, seq_k, head_dim] FP8 E4M3
    float q_scale,
    float k_scale,
    float attn_scale,                  // 1/sqrt(head_dim)
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_q,
    uint32_t seq_k,
    uint32_t head_dim
) {
    uint32_t b = blockIdx.z / num_heads;
    uint32_t h = blockIdx.z % num_heads;
    uint32_t q_idx = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t k_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch_size || q_idx >= seq_q || k_idx >= seq_k) return;

    const uint8_t* q_ptr = Q + (b * num_heads + h) * seq_q * head_dim + q_idx * head_dim;
    const uint8_t* k_ptr = K + (b * num_heads + h) * seq_k * head_dim + k_idx * head_dim;

    float acc = 0.0f;

    for (uint32_t d = 0; d < head_dim; d++) {
        float q = fp8_e4m3_to_float(q_ptr[d]) * q_scale;
        float k = fp8_e4m3_to_float(k_ptr[d]) * k_scale;
        acc += q * k;
    }

    acc *= attn_scale;

    uint32_t out_idx = (b * num_heads + h) * seq_q * seq_k + q_idx * seq_k + k_idx;
    scores[out_idx] = acc;
}

// ============================================================================
// FP16 Softmax with FP32 Accumulation
// ============================================================================

extern "C" __global__
void fp16_softmax_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    uint32_t batch_size,
    uint32_t seq_len
) {
    extern __shared__ float sdata[];

    uint32_t batch = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (batch >= batch_size) return;

    const __half* row = input + batch * seq_len;
    __half* out_row = output + batch * seq_len;

    // Find max (FP32 accumulation)
    float max_val = -1e10f;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        max_val = fmaxf(max_val, __half2float(row[i]));
    }

    sdata[tid] = max_val;
    __syncthreads();

    // Reduce max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();

    // Compute exp and sum
    float sum = 0.0f;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        float val = expf(__half2float(row[i]) - max_val);
        sdata[tid + (i / blockDim.x) * blockDim.x] = val;
        sum += val;
    }

    // Store partial sums for reduction
    __shared__ float sum_shared[32];
    float thread_sum = sum;

    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    if (tid % 32 == 0) sum_shared[tid / 32] = thread_sum;
    __syncthreads();

    // Final reduction
    if (tid < 32) {
        thread_sum = (tid < blockDim.x / 32) ? sum_shared[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
        if (tid == 0) sum_shared[0] = thread_sum;
    }
    __syncthreads();
    sum = sum_shared[0];

    // Normalize and write output
    float inv_sum = 1.0f / sum;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        float val = expf(__half2float(row[i]) - max_val) * inv_sum;
        out_row[i] = __float2half(val);
    }
}

// ============================================================================
// FP8 Attention Output (Probs @ V)
// ============================================================================

extern "C" __global__
void fp8_attention_output_kernel(
    __half* __restrict__ output,      // [batch, heads, seq_q, head_dim]
    const float* __restrict__ probs,  // [batch, heads, seq_q, seq_k]
    const uint8_t* __restrict__ V,    // [batch, heads, seq_k, head_dim]
    float v_scale,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_q,
    uint32_t seq_k,
    uint32_t head_dim
) {
    uint32_t b = blockIdx.z / num_heads;
    uint32_t h = blockIdx.z % num_heads;
    uint32_t q_idx = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t d_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch_size || q_idx >= seq_q || d_idx >= head_dim) return;

    const float* prob_row = probs + (b * num_heads + h) * seq_q * seq_k + q_idx * seq_k;
    const uint8_t* v_base = V + (b * num_heads + h) * seq_k * head_dim;

    float acc = 0.0f;

    for (uint32_t k = 0; k < seq_k; k++) {
        float p = prob_row[k];
        float v = fp8_e4m3_to_float(v_base[k * head_dim + d_idx]) * v_scale;
        acc += p * v;
    }

    uint32_t out_idx = (b * num_heads + h) * seq_q * head_dim + q_idx * head_dim + d_idx;
    output[out_idx] = __float2half(acc);
}

// ============================================================================
// FP16 Attention Output
// ============================================================================

extern "C" __global__
void fp16_attention_output_kernel(
    __half* __restrict__ output,      // [batch, heads, seq_q, head_dim]
    const __half* __restrict__ probs, // [batch, heads, seq_q, seq_k]
    const __half* __restrict__ V,     // [batch, heads, seq_k, head_dim]
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_q,
    uint32_t seq_k,
    uint32_t head_dim
) {
    uint32_t b = blockIdx.z / num_heads;
    uint32_t h = blockIdx.z % num_heads;
    uint32_t q_idx = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t d_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch_size || q_idx >= seq_q || d_idx >= head_dim) return;

    const __half* prob_row = probs + (b * num_heads + h) * seq_q * seq_k + q_idx * seq_k;
    const __half* v_base = V + (b * num_heads + h) * seq_k * head_dim;

    float acc = 0.0f;

    for (uint32_t k = 0; k < seq_k; k++) {
        float p = __half2float(prob_row[k]);
        float v = __half2float(v_base[k * head_dim + d_idx]);
        acc += p * v;
    }

    uint32_t out_idx = (b * num_heads + h) * seq_q * head_dim + q_idx * head_dim + d_idx;
    output[out_idx] = __float2half(acc);
}

// ============================================================================
// Fused FP16 NAX Attention
// ============================================================================

extern "C" __global__
void fp16_nax_fused_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    float scale,                       // 1/sqrt(head_dim)
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim
) {
    extern __shared__ float shared_mem[];

    uint32_t b = blockIdx.z / num_heads;
    uint32_t h = blockIdx.z % num_heads;
    uint32_t q_idx = blockIdx.y;

    if (b >= batch_size || q_idx >= seq_len) return;

    float* scores = shared_mem;
    uint32_t tid = threadIdx.x;

    const __half* q_row = Q + (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim;
    const __half* k_base = K + (b * num_heads + h) * seq_len * head_dim;
    const __half* v_base = V + (b * num_heads + h) * seq_len * head_dim;

    // Step 1: Compute Q @ K^T
    for (uint32_t k_idx = tid; k_idx < seq_len; k_idx += blockDim.x) {
        const __half* k_row = k_base + k_idx * head_dim;

        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) {
            dot += __half2float(q_row[d]) * __half2float(k_row[d]);
        }
        scores[k_idx] = dot * scale;
    }
    __syncthreads();

    // Step 2: Softmax
    // Find max
    float max_val = -1e10f;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        max_val = fmaxf(max_val, scores[i]);
    }

    __shared__ float reduce_buf[32];
    // Warp reduction for max
    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    if (tid % 32 == 0) reduce_buf[tid / 32] = max_val;
    __syncthreads();

    if (tid < 32) {
        max_val = (tid < (blockDim.x + 31) / 32) ? reduce_buf[tid] : -1e10f;
        for (int offset = 16; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        }
        if (tid == 0) reduce_buf[0] = max_val;
    }
    __syncthreads();
    max_val = reduce_buf[0];

    // Compute exp and sum
    float sum = 0.0f;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_val);
        sum += scores[i];
    }

    // Warp reduction for sum
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (tid % 32 == 0) reduce_buf[tid / 32] = sum;
    __syncthreads();

    if (tid < 32) {
        sum = (tid < (blockDim.x + 31) / 32) ? reduce_buf[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) reduce_buf[0] = sum;
    }
    __syncthreads();
    sum = reduce_buf[0];

    // Normalize
    float inv_sum = 1.0f / sum;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        scores[i] *= inv_sum;
    }
    __syncthreads();

    // Step 3: Compute Softmax @ V
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;

        for (uint32_t k = 0; k < seq_len; k++) {
            float v_val = __half2float(v_base[k * head_dim + d]);
            acc += scores[k] * v_val;
        }

        uint32_t out_idx = (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim + d;
        output[out_idx] = __float2half(acc);
    }
}

// ============================================================================
// Fused FP8 NAX Attention
// ============================================================================

extern "C" __global__
void fp8_nax_fused_kernel(
    __half* __restrict__ output,
    const uint8_t* __restrict__ Q,
    const uint8_t* __restrict__ K,
    const uint8_t* __restrict__ V,
    float q_scale,
    float k_scale,
    float v_scale,
    float attn_scale,                  // 1/sqrt(head_dim)
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim
) {
    extern __shared__ float shared_mem[];

    uint32_t b = blockIdx.z / num_heads;
    uint32_t h = blockIdx.z % num_heads;
    uint32_t q_idx = blockIdx.y;

    if (b >= batch_size || q_idx >= seq_len) return;

    float* scores = shared_mem;
    uint32_t tid = threadIdx.x;

    const uint8_t* q_row = Q + (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim;
    const uint8_t* k_base = K + (b * num_heads + h) * seq_len * head_dim;
    const uint8_t* v_base = V + (b * num_heads + h) * seq_len * head_dim;

    float combined_scale = q_scale * k_scale * attn_scale;

    // Step 1: Compute Q @ K^T
    for (uint32_t k_idx = tid; k_idx < seq_len; k_idx += blockDim.x) {
        const uint8_t* k_row = k_base + k_idx * head_dim;

        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) {
            float q = fp8_e4m3_to_float(q_row[d]);
            float k = fp8_e4m3_to_float(k_row[d]);
            dot += q * k;
        }
        scores[k_idx] = dot * combined_scale;
    }
    __syncthreads();

    // Step 2: Softmax
    __shared__ float reduce_buf[32];

    // Find max
    float max_val = -1e10f;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        max_val = fmaxf(max_val, scores[i]);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    if (tid % 32 == 0) reduce_buf[tid / 32] = max_val;
    __syncthreads();

    if (tid < 32) {
        max_val = (tid < (blockDim.x + 31) / 32) ? reduce_buf[tid] : -1e10f;
        for (int offset = 16; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        }
        if (tid == 0) reduce_buf[0] = max_val;
    }
    __syncthreads();
    max_val = reduce_buf[0];

    // Compute exp and sum
    float sum = 0.0f;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_val);
        sum += scores[i];
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (tid % 32 == 0) reduce_buf[tid / 32] = sum;
    __syncthreads();

    if (tid < 32) {
        sum = (tid < (blockDim.x + 31) / 32) ? reduce_buf[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) reduce_buf[0] = sum;
    }
    __syncthreads();
    sum = reduce_buf[0];

    // Normalize
    float inv_sum = 1.0f / sum;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        scores[i] *= inv_sum;
    }
    __syncthreads();

    // Step 3: Compute Softmax @ V
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;

        for (uint32_t k = 0; k < seq_len; k++) {
            float v_val = fp8_e4m3_to_float(v_base[k * head_dim + d]) * v_scale;
            acc += scores[k] * v_val;
        }

        uint32_t out_idx = (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim + d;
        output[out_idx] = __float2half(acc);
    }
}

// ============================================================================
// BF16 NAX Attention (for Ampere+ GPUs)
// ============================================================================

extern "C" __global__
void bf16_nax_fused_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    float scale,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim
) {
    extern __shared__ float shared_mem[];

    uint32_t b = blockIdx.z / num_heads;
    uint32_t h = blockIdx.z % num_heads;
    uint32_t q_idx = blockIdx.y;

    if (b >= batch_size || q_idx >= seq_len) return;

    float* scores = shared_mem;
    uint32_t tid = threadIdx.x;

    const __nv_bfloat16* q_row = Q + (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim;
    const __nv_bfloat16* k_base = K + (b * num_heads + h) * seq_len * head_dim;
    const __nv_bfloat16* v_base = V + (b * num_heads + h) * seq_len * head_dim;

    // Step 1: Compute Q @ K^T
    for (uint32_t k_idx = tid; k_idx < seq_len; k_idx += blockDim.x) {
        const __nv_bfloat16* k_row = k_base + k_idx * head_dim;

        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) {
            dot += __bfloat162float(q_row[d]) * __bfloat162float(k_row[d]);
        }
        scores[k_idx] = dot * scale;
    }
    __syncthreads();

    // Step 2: Softmax (same as FP16)
    __shared__ float reduce_buf[32];

    float max_val = -1e10f;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        max_val = fmaxf(max_val, scores[i]);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    if (tid % 32 == 0) reduce_buf[tid / 32] = max_val;
    __syncthreads();

    if (tid < 32) {
        max_val = (tid < (blockDim.x + 31) / 32) ? reduce_buf[tid] : -1e10f;
        for (int offset = 16; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        }
        if (tid == 0) reduce_buf[0] = max_val;
    }
    __syncthreads();
    max_val = reduce_buf[0];

    float sum = 0.0f;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_val);
        sum += scores[i];
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (tid % 32 == 0) reduce_buf[tid / 32] = sum;
    __syncthreads();

    if (tid < 32) {
        sum = (tid < (blockDim.x + 31) / 32) ? reduce_buf[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) reduce_buf[0] = sum;
    }
    __syncthreads();
    sum = reduce_buf[0];

    float inv_sum = 1.0f / sum;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        scores[i] *= inv_sum;
    }
    __syncthreads();

    // Step 3: Compute Softmax @ V
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;

        for (uint32_t k = 0; k < seq_len; k++) {
            float v_val = __bfloat162float(v_base[k * head_dim + d]);
            acc += scores[k] * v_val;
        }

        uint32_t out_idx = (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim + d;
        output[out_idx] = __float2bfloat16(acc);
    }
}

// ============================================================================
// Causal Masked Attention Variants
// ============================================================================

extern "C" __global__
void fp16_nax_causal_fused_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    float scale,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim
) {
    extern __shared__ float shared_mem[];

    uint32_t b = blockIdx.z / num_heads;
    uint32_t h = blockIdx.z % num_heads;
    uint32_t q_idx = blockIdx.y;

    if (b >= batch_size || q_idx >= seq_len) return;

    float* scores = shared_mem;
    uint32_t tid = threadIdx.x;

    const __half* q_row = Q + (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim;
    const __half* k_base = K + (b * num_heads + h) * seq_len * head_dim;
    const __half* v_base = V + (b * num_heads + h) * seq_len * head_dim;

    // Causal mask: only attend to positions <= q_idx
    uint32_t max_k = q_idx + 1;

    // Step 1: Compute Q @ K^T with causal mask
    for (uint32_t k_idx = tid; k_idx < seq_len; k_idx += blockDim.x) {
        if (k_idx >= max_k) {
            scores[k_idx] = -1e10f;  // Mask out future positions
            continue;
        }

        const __half* k_row = k_base + k_idx * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) {
            dot += __half2float(q_row[d]) * __half2float(k_row[d]);
        }
        scores[k_idx] = dot * scale;
    }
    __syncthreads();

    // Step 2: Softmax (only over valid positions)
    __shared__ float reduce_buf[32];

    float max_val = -1e10f;
    for (uint32_t i = tid; i < max_k; i += blockDim.x) {
        max_val = fmaxf(max_val, scores[i]);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    if (tid % 32 == 0) reduce_buf[tid / 32] = max_val;
    __syncthreads();

    if (tid < 32) {
        max_val = (tid < (blockDim.x + 31) / 32) ? reduce_buf[tid] : -1e10f;
        for (int offset = 16; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        }
        if (tid == 0) reduce_buf[0] = max_val;
    }
    __syncthreads();
    max_val = reduce_buf[0];

    float sum = 0.0f;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_val);
        if (i < max_k) sum += scores[i];
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (tid % 32 == 0) reduce_buf[tid / 32] = sum;
    __syncthreads();

    if (tid < 32) {
        sum = (tid < (blockDim.x + 31) / 32) ? reduce_buf[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) reduce_buf[0] = sum;
    }
    __syncthreads();
    sum = reduce_buf[0];

    float inv_sum = 1.0f / sum;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        scores[i] = (i < max_k) ? scores[i] * inv_sum : 0.0f;
    }
    __syncthreads();

    // Step 3: Compute Softmax @ V
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;

        for (uint32_t k = 0; k < max_k; k++) {
            float v_val = __half2float(v_base[k * head_dim + d]);
            acc += scores[k] * v_val;
        }

        uint32_t out_idx = (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim + d;
        output[out_idx] = __float2half(acc);
    }
}

// ============================================================================
// Host API (C Interface)
// ============================================================================

} // namespace fp_nax
} // namespace cuda
} // namespace lux

extern "C" {

using namespace lux::cuda::fp_nax;

// -------------------- FP16 NAX --------------------

int lux_cuda_fp16_attention_scores(
    void* scores,
    const void* Q,
    const void* K,
    float scale,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_q,
    uint32_t seq_k,
    uint32_t head_dim,
    cudaStream_t stream
) {
    dim3 threads(16, 16);
    dim3 blocks((seq_k + 15) / 16, (seq_q + 15) / 16, batch_size * num_heads);

    fp16_attention_scores_kernel<<<blocks, threads, 0, stream>>>(
        (__half*)scores, (const __half*)Q, (const __half*)K,
        scale, batch_size, num_heads, seq_q, seq_k, head_dim
    );

    return cudaGetLastError();
}

int lux_cuda_fp16_nax_fused(
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

    fp16_nax_fused_kernel<<<blocks, threads, shared_size, stream>>>(
        (__half*)output, (const __half*)Q, (const __half*)K, (const __half*)V,
        scale, batch_size, num_heads, seq_len, head_dim
    );

    return cudaGetLastError();
}

int lux_cuda_fp16_nax_causal_fused(
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

    fp16_nax_causal_fused_kernel<<<blocks, threads, shared_size, stream>>>(
        (__half*)output, (const __half*)Q, (const __half*)K, (const __half*)V,
        scale, batch_size, num_heads, seq_len, head_dim
    );

    return cudaGetLastError();
}

// -------------------- FP8 NAX --------------------

int lux_cuda_fp8_attention_scores(
    void* scores,
    const void* Q,
    const void* K,
    float q_scale,
    float k_scale,
    float attn_scale,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_q,
    uint32_t seq_k,
    uint32_t head_dim,
    cudaStream_t stream
) {
    dim3 threads(16, 16);
    dim3 blocks((seq_k + 15) / 16, (seq_q + 15) / 16, batch_size * num_heads);

    fp8_attention_scores_kernel<<<blocks, threads, 0, stream>>>(
        (float*)scores, (const uint8_t*)Q, (const uint8_t*)K,
        q_scale, k_scale, attn_scale,
        batch_size, num_heads, seq_q, seq_k, head_dim
    );

    return cudaGetLastError();
}

int lux_cuda_fp8_nax_fused(
    void* output,
    const void* Q,
    const void* K,
    const void* V,
    float q_scale,
    float k_scale,
    float v_scale,
    float attn_scale,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    cudaStream_t stream
) {
    dim3 blocks(1, seq_len, batch_size * num_heads);
    uint32_t threads = min(256u, seq_len);
    size_t shared_size = seq_len * sizeof(float);

    fp8_nax_fused_kernel<<<blocks, threads, shared_size, stream>>>(
        (__half*)output, (const uint8_t*)Q, (const uint8_t*)K, (const uint8_t*)V,
        q_scale, k_scale, v_scale, attn_scale,
        batch_size, num_heads, seq_len, head_dim
    );

    return cudaGetLastError();
}

int lux_cuda_fp8_attention_output(
    void* output,
    const void* probs,
    const void* V,
    float v_scale,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_q,
    uint32_t seq_k,
    uint32_t head_dim,
    cudaStream_t stream
) {
    dim3 threads(16, 16);
    dim3 blocks((head_dim + 15) / 16, (seq_q + 15) / 16, batch_size * num_heads);

    fp8_attention_output_kernel<<<blocks, threads, 0, stream>>>(
        (__half*)output, (const float*)probs, (const uint8_t*)V,
        v_scale, batch_size, num_heads, seq_q, seq_k, head_dim
    );

    return cudaGetLastError();
}

// -------------------- BF16 NAX --------------------

int lux_cuda_bf16_nax_fused(
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

    bf16_nax_fused_kernel<<<blocks, threads, shared_size, stream>>>(
        (__nv_bfloat16*)output,
        (const __nv_bfloat16*)Q,
        (const __nv_bfloat16*)K,
        (const __nv_bfloat16*)V,
        scale, batch_size, num_heads, seq_len, head_dim
    );

    return cudaGetLastError();
}

// -------------------- FP16 Softmax --------------------

int lux_cuda_fp16_softmax(
    void* output,
    const void* input,
    uint32_t batch_size,
    uint32_t seq_len,
    cudaStream_t stream
) {
    dim3 blocks(batch_size);
    uint32_t threads = min(256u, seq_len);
    size_t shared_size = threads * sizeof(float);

    fp16_softmax_kernel<<<blocks, threads, shared_size, stream>>>(
        (__half*)output, (const __half*)input, batch_size, seq_len
    );

    return cudaGetLastError();
}

// -------------------- FP16 Attention Output --------------------

int lux_cuda_fp16_attention_output(
    void* output,
    const void* probs,
    const void* V,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_q,
    uint32_t seq_k,
    uint32_t head_dim,
    cudaStream_t stream
) {
    dim3 threads(16, 16);
    dim3 blocks((head_dim + 15) / 16, (seq_q + 15) / 16, batch_size * num_heads);

    fp16_attention_output_kernel<<<blocks, threads, 0, stream>>>(
        (__half*)output, (const __half*)probs, (const __half*)V,
        batch_size, num_heads, seq_q, seq_k, head_dim
    );

    return cudaGetLastError();
}

} // extern "C"
