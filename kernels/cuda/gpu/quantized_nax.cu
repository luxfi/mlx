// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Quantized Neural Attention Operations CUDA Kernels
// INT8/INT4 quantized attention for efficient inference

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// Quantization Parameters
// ============================================================================

struct QuantParams {
    float scale;
    int32_t zero_point;
};

// ============================================================================
// INT8 Quantization/Dequantization
// ============================================================================

__device__ __forceinline__
int8_t quantize_int8(float val, float scale, int32_t zero_point) {
    int32_t q = __float2int_rn(val / scale) + zero_point;
    return (int8_t)max(-128, min(127, q));
}

__device__ __forceinline__
float dequantize_int8(int8_t val, float scale, int32_t zero_point) {
    return ((float)val - zero_point) * scale;
}

// ============================================================================
// INT4 Quantization (packed as 2 values per byte)
// ============================================================================

__device__ __forceinline__
uint8_t pack_int4(int8_t a, int8_t b) {
    return ((uint8_t)(a & 0xF)) | ((uint8_t)(b & 0xF) << 4);
}

__device__ __forceinline__
void unpack_int4(uint8_t packed, int8_t* a, int8_t* b) {
    *a = (int8_t)(packed & 0xF);
    *b = (int8_t)((packed >> 4) & 0xF);
    // Sign extend from 4 bits
    if (*a >= 8) *a -= 16;
    if (*b >= 8) *b -= 16;
}

// ============================================================================
// Quantized Matrix Multiplication (INT8)
// ============================================================================

extern "C" __global__
void quantized_matmul_int8_kernel(
    int32_t* __restrict__ C,          // Output accumulator (int32)
    const int8_t* __restrict__ A,     // [M, K] quantized
    const int8_t* __restrict__ B,     // [K, N] quantized
    uint32_t M, uint32_t N, uint32_t K
) {
    // Tile sizes
    const int TILE_M = 16;
    const int TILE_N = 16;
    const int TILE_K = 16;

    __shared__ int8_t As[TILE_M][TILE_K];
    __shared__ int8_t Bs[TILE_K][TILE_N];

    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    int32_t acc = 0;

    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; t++) {
        // Load A tile
        int a_col = t * TILE_K + threadIdx.x;
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0;
        }

        // Load B tile
        int b_row = t * TILE_K + threadIdx.y;
        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            acc += (int32_t)As[threadIdx.y][k] * (int32_t)Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ============================================================================
// Quantized Attention Score Computation
// ============================================================================

extern "C" __global__
void quantized_attention_scores_kernel(
    int32_t* __restrict__ scores,     // [batch, heads, seq_q, seq_k]
    const int8_t* __restrict__ Q,     // [batch, heads, seq_q, head_dim]
    const int8_t* __restrict__ K,     // [batch, heads, seq_k, head_dim]
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

    const int8_t* q_ptr = Q + (b * num_heads + h) * seq_q * head_dim + q_idx * head_dim;
    const int8_t* k_ptr = K + (b * num_heads + h) * seq_k * head_dim + k_idx * head_dim;

    int32_t score = 0;

    // Compute Q @ K^T for this position
    for (uint32_t d = 0; d < head_dim; d++) {
        score += (int32_t)q_ptr[d] * (int32_t)k_ptr[d];
    }

    uint32_t out_idx = (b * num_heads + h) * seq_q * seq_k + q_idx * seq_k + k_idx;
    scores[out_idx] = score;
}

// ============================================================================
// Dequantize + Softmax + Requantize
// ============================================================================

extern "C" __global__
void quantized_softmax_kernel(
    int8_t* __restrict__ output,
    const int32_t* __restrict__ input,
    float q_scale,
    float k_scale,
    float out_scale,
    int32_t out_zero_point,
    float temperature,
    uint32_t batch_size,
    uint32_t seq_len
) {
    extern __shared__ float shared[];

    uint32_t batch = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (batch >= batch_size) return;

    const int32_t* row = input + batch * seq_len;
    float* smax = shared;

    // Dequantize and find max
    float max_val = -1e10f;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        float val = (float)row[i] * q_scale * k_scale / temperature;
        smax[i] = val;
        max_val = fmaxf(max_val, val);
    }

    // Reduce max across threads
    __shared__ float block_max;
    if (tid == 0) block_max = -1e10f;
    __syncthreads();
    atomicMax((int*)&block_max, __float_as_int(max_val));
    __syncthreads();
    max_val = block_max;

    // Compute exp and sum
    float sum = 0.0f;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        smax[i] = expf(smax[i] - max_val);
        sum += smax[i];
    }

    // Reduce sum
    __shared__ float block_sum;
    if (tid == 0) block_sum = 0.0f;
    __syncthreads();
    atomicAdd(&block_sum, sum);
    __syncthreads();
    sum = block_sum;

    // Normalize and requantize
    int8_t* out_row = output + batch * seq_len;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        float prob = smax[i] / sum;
        out_row[i] = quantize_int8(prob, out_scale, out_zero_point);
    }
}

// ============================================================================
// Quantized Attention Output (INT8 x INT8 -> FP16)
// ============================================================================

extern "C" __global__
void quantized_attention_output_kernel(
    __half* __restrict__ output,      // [batch, heads, seq_q, head_dim]
    const int8_t* __restrict__ probs, // [batch, heads, seq_q, seq_k]
    const int8_t* __restrict__ V,     // [batch, heads, seq_k, head_dim]
    float prob_scale,
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

    const int8_t* prob_row = probs + (b * num_heads + h) * seq_q * seq_k + q_idx * seq_k;
    const int8_t* v_ptr = V + (b * num_heads + h) * seq_k * head_dim;

    float acc = 0.0f;

    for (uint32_t k = 0; k < seq_k; k++) {
        float p = (float)prob_row[k] * prob_scale;
        float v = (float)v_ptr[k * head_dim + d_idx] * v_scale;
        acc += p * v;
    }

    uint32_t out_idx = (b * num_heads + h) * seq_q * head_dim + q_idx * head_dim + d_idx;
    output[out_idx] = __float2half(acc);
}

// ============================================================================
// Full Quantized Attention (Fused)
// ============================================================================

extern "C" __global__
void quantized_nax_fused_kernel(
    __half* __restrict__ output,
    const int8_t* __restrict__ Q,
    const int8_t* __restrict__ K,
    const int8_t* __restrict__ V,
    const QuantParams* __restrict__ q_params,
    const QuantParams* __restrict__ k_params,
    const QuantParams* __restrict__ v_params,
    float temperature,
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
    float* v_acc = shared_mem + seq_len;

    uint32_t tid = threadIdx.x;

    const int8_t* q_row = Q + (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim;
    const int8_t* k_base = K + (b * num_heads + h) * seq_len * head_dim;
    const int8_t* v_base = V + (b * num_heads + h) * seq_len * head_dim;

    float scale_factor = q_params->scale * k_params->scale / sqrtf((float)head_dim) / temperature;

    // Step 1: Compute attention scores (Q @ K^T)
    for (uint32_t k_idx = tid; k_idx < seq_len; k_idx += blockDim.x) {
        int32_t dot = 0;
        const int8_t* k_row = k_base + k_idx * head_dim;

        for (uint32_t d = 0; d < head_dim; d++) {
            dot += (int32_t)q_row[d] * (int32_t)k_row[d];
        }

        scores[k_idx] = (float)dot * scale_factor;
    }
    __syncthreads();

    // Step 2: Softmax
    // Find max
    float max_val = -1e10f;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        max_val = fmaxf(max_val, scores[i]);
    }

    __shared__ float block_max;
    if (tid == 0) block_max = -1e10f;
    __syncthreads();

    // Warp reduction for max
    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    if (tid % 32 == 0) atomicMax((int*)&block_max, __float_as_int(max_val));
    __syncthreads();
    max_val = block_max;

    // Compute exp and sum
    float sum = 0.0f;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_val);
        sum += scores[i];
    }

    // Reduce sum
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float block_sum;
    if (tid == 0) block_sum = 0.0f;
    __syncthreads();
    if (tid % 32 == 0) atomicAdd(&block_sum, sum);
    __syncthreads();
    sum = block_sum;

    // Normalize
    float inv_sum = 1.0f / sum;
    for (uint32_t i = tid; i < seq_len; i += blockDim.x) {
        scores[i] *= inv_sum;
    }
    __syncthreads();

    // Step 3: Compute output (Softmax @ V)
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;

        for (uint32_t k = 0; k < seq_len; k++) {
            float v_val = (float)v_base[k * head_dim + d] * v_params->scale;
            acc += scores[k] * v_val;
        }

        uint32_t out_idx = (b * num_heads + h) * seq_len * head_dim + q_idx * head_dim + d;
        output[out_idx] = __float2half(acc);
    }
}

// ============================================================================
// C API for CGO Bindings
// ============================================================================

extern "C" {

int lux_cuda_quantized_matmul_int8(
    void* C,
    const void* A,
    const void* B,
    uint32_t M, uint32_t N, uint32_t K,
    cudaStream_t stream
) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    quantized_matmul_int8_kernel<<<blocks, threads, 0, stream>>>(
        (int32_t*)C,
        (const int8_t*)A,
        (const int8_t*)B,
        M, N, K
    );

    return cudaGetLastError();
}

int lux_cuda_quantized_attention_scores(
    void* scores,
    const void* Q,
    const void* K,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_q,
    uint32_t seq_k,
    uint32_t head_dim,
    cudaStream_t stream
) {
    dim3 threads(16, 16);
    dim3 blocks((seq_k + 15) / 16, (seq_q + 15) / 16, batch_size * num_heads);

    quantized_attention_scores_kernel<<<blocks, threads, 0, stream>>>(
        (int32_t*)scores,
        (const int8_t*)Q,
        (const int8_t*)K,
        batch_size, num_heads, seq_q, seq_k, head_dim
    );

    return cudaGetLastError();
}

int lux_cuda_quantized_nax_fused(
    void* output,
    const void* Q,
    const void* K,
    const void* V,
    const void* q_params,
    const void* k_params,
    const void* v_params,
    float temperature,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    cudaStream_t stream
) {
    dim3 blocks(1, seq_len, batch_size * num_heads);
    uint32_t threads = min(256u, seq_len);
    size_t shared_size = (seq_len + head_dim) * sizeof(float);

    quantized_nax_fused_kernel<<<blocks, threads, shared_size, stream>>>(
        (__half*)output,
        (const int8_t*)Q,
        (const int8_t*)K,
        (const int8_t*)V,
        (const QuantParams*)q_params,
        (const QuantParams*)k_params,
        (const QuantParams*)v_params,
        temperature,
        batch_size, num_heads, seq_len, head_dim
    );

    return cudaGetLastError();
}

}  // extern "C"
