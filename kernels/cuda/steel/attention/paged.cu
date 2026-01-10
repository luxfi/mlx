// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Paged Attention CUDA Kernels
// Efficient KV cache management for LLM inference (vLLM style)

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// Paged Attention Configuration
// ============================================================================

#define WARP_SIZE 32
#define BLOCK_SIZE 16          // Tokens per block in KV cache
#define MAX_CONTEXT_LEN 32768  // Maximum context length

// ============================================================================
// Paged Attention V1 (Simple version)
// ============================================================================

// Each sequence has a block table mapping logical blocks to physical blocks
// block_tables[seq_idx][logical_block_idx] = physical_block_idx
// k_cache[physical_block_idx][num_heads][block_size][head_dim]
// v_cache[physical_block_idx][num_heads][block_size][head_dim]

extern "C" __global__
void paged_attention_v1_kernel(
    float* __restrict__ output,           // [num_seqs, num_heads, head_dim]
    const float* __restrict__ query,      // [num_seqs, num_heads, head_dim]
    const float* __restrict__ key_cache,  // [num_blocks, num_heads, block_size, head_dim]
    const float* __restrict__ value_cache,// [num_blocks, num_heads, block_size, head_dim]
    const int32_t* __restrict__ block_tables,  // [num_seqs, max_blocks_per_seq]
    const int32_t* __restrict__ context_lens,  // [num_seqs]
    float scale,
    uint32_t num_seqs,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t block_size,
    uint32_t max_blocks_per_seq
) {
    extern __shared__ float smem[];

    uint32_t seq_idx = blockIdx.x;
    uint32_t head_idx = blockIdx.y;
    uint32_t tid = threadIdx.x;

    if (seq_idx >= num_seqs) return;

    int32_t context_len = context_lens[seq_idx];
    if (context_len <= 0) return;

    float* scores = smem;
    float* output_smem = smem + context_len;

    // Query for this sequence/head
    const float* q = query + seq_idx * num_heads * head_dim + head_idx * head_dim;
    float* out = output + seq_idx * num_heads * head_dim + head_idx * head_dim;

    // Block table for this sequence
    const int32_t* seq_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Step 1: Compute attention scores
    for (int32_t token_idx = tid; token_idx < context_len; token_idx += blockDim.x) {
        int32_t logical_block = token_idx / block_size;
        int32_t block_offset = token_idx % block_size;
        int32_t physical_block = seq_block_table[logical_block];

        // Key location in cache
        const float* k = key_cache +
            physical_block * num_heads * block_size * head_dim +
            head_idx * block_size * head_dim +
            block_offset * head_dim;

        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) {
            dot += q[d] * k[d];
        }

        scores[token_idx] = dot * scale;
    }
    __syncthreads();

    // Step 2: Softmax
    // Find max
    float max_val = -1e10f;
    for (int32_t i = tid; i < context_len; i += blockDim.x) {
        max_val = fmaxf(max_val, scores[i]);
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }

    __shared__ float block_max;
    if (tid % WARP_SIZE == 0) atomicMax((int*)&block_max, __float_as_int(max_val));
    __syncthreads();
    max_val = block_max;

    // Compute exp and sum
    float sum = 0.0f;
    for (int32_t i = tid; i < context_len; i += blockDim.x) {
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
    for (int32_t i = tid; i < context_len; i += blockDim.x) {
        scores[i] *= inv_sum;
    }
    __syncthreads();

    // Step 3: Compute output
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;

        for (int32_t token_idx = 0; token_idx < context_len; token_idx++) {
            int32_t logical_block = token_idx / block_size;
            int32_t block_offset = token_idx % block_size;
            int32_t physical_block = seq_block_table[logical_block];

            const float* v = value_cache +
                physical_block * num_heads * block_size * head_dim +
                head_idx * block_size * head_dim +
                block_offset * head_dim;

            acc += scores[token_idx] * v[d];
        }

        out[d] = acc;
    }
}

// ============================================================================
// Paged Attention V2 (Partitioned across blocks for long contexts)
// ============================================================================

// For very long contexts, partition the attention computation across multiple blocks
// Each block handles a subset of KV tokens, then results are reduced

extern "C" __global__
void paged_attention_v2_kernel(
    float* __restrict__ output,
    float* __restrict__ exp_sums,         // [num_seqs, num_heads, max_partitions]
    float* __restrict__ max_logits,       // [num_seqs, num_heads, max_partitions]
    const float* __restrict__ query,
    const float* __restrict__ key_cache,
    const float* __restrict__ value_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ context_lens,
    float scale,
    uint32_t num_seqs,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t block_size,
    uint32_t max_blocks_per_seq,
    uint32_t partition_size
) {
    extern __shared__ float smem[];

    uint32_t seq_idx = blockIdx.x;
    uint32_t head_idx = blockIdx.y;
    uint32_t partition_idx = blockIdx.z;
    uint32_t tid = threadIdx.x;

    if (seq_idx >= num_seqs) return;

    int32_t context_len = context_lens[seq_idx];
    if (context_len <= 0) return;

    // Partition range
    int32_t partition_start = partition_idx * partition_size;
    int32_t partition_end = min(partition_start + (int32_t)partition_size, context_len);

    if (partition_start >= context_len) return;

    int32_t partition_len = partition_end - partition_start;

    float* scores = smem;
    float* output_smem = smem + partition_len;

    const float* q = query + seq_idx * num_heads * head_dim + head_idx * head_dim;
    const int32_t* seq_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Compute scores for this partition
    for (int32_t i = tid; i < partition_len; i += blockDim.x) {
        int32_t token_idx = partition_start + i;
        int32_t logical_block = token_idx / block_size;
        int32_t block_offset = token_idx % block_size;
        int32_t physical_block = seq_block_table[logical_block];

        const float* k = key_cache +
            physical_block * num_heads * block_size * head_dim +
            head_idx * block_size * head_dim +
            block_offset * head_dim;

        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) {
            dot += q[d] * k[d];
        }

        scores[i] = dot * scale;
    }
    __syncthreads();

    // Local softmax
    float max_val = -1e10f;
    for (int32_t i = tid; i < partition_len; i += blockDim.x) {
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
    for (int32_t i = tid; i < partition_len; i += blockDim.x) {
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

    // Store partition statistics
    if (tid == 0) {
        uint32_t stat_idx = seq_idx * num_heads * gridDim.z + head_idx * gridDim.z + partition_idx;
        max_logits[stat_idx] = max_val;
        exp_sums[stat_idx] = block_sum;
    }

    // Compute weighted output for this partition
    float inv_sum = 1.0f / block_sum;
    for (int32_t i = tid; i < partition_len; i += blockDim.x) {
        scores[i] *= inv_sum;
    }
    __syncthreads();

    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;

        for (int32_t i = 0; i < partition_len; i++) {
            int32_t token_idx = partition_start + i;
            int32_t logical_block = token_idx / block_size;
            int32_t block_offset = token_idx % block_size;
            int32_t physical_block = seq_block_table[logical_block];

            const float* v = value_cache +
                physical_block * num_heads * block_size * head_dim +
                head_idx * block_size * head_dim +
                block_offset * head_dim;

            acc += scores[i] * v[d];
        }

        output_smem[d] = acc;
    }
    __syncthreads();

    // Write partition output (will be reduced later)
    float* part_out = output +
        (seq_idx * num_heads * gridDim.z + head_idx * gridDim.z + partition_idx) * head_dim;

    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        part_out[d] = output_smem[d];
    }
}

// Reduction kernel to combine partition outputs
extern "C" __global__
void paged_attention_reduce_kernel(
    float* __restrict__ output,
    const float* __restrict__ partition_outputs,
    const float* __restrict__ exp_sums,
    const float* __restrict__ max_logits,
    uint32_t num_seqs,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t num_partitions
) {
    uint32_t seq_idx = blockIdx.x;
    uint32_t head_idx = blockIdx.y;
    uint32_t tid = threadIdx.x;

    if (seq_idx >= num_seqs) return;

    // Find global max
    float global_max = -1e10f;
    for (uint32_t p = 0; p < num_partitions; p++) {
        uint32_t idx = seq_idx * num_heads * num_partitions + head_idx * num_partitions + p;
        global_max = fmaxf(global_max, max_logits[idx]);
    }

    // Compute rescaled sum
    float global_sum = 0.0f;
    for (uint32_t p = 0; p < num_partitions; p++) {
        uint32_t idx = seq_idx * num_heads * num_partitions + head_idx * num_partitions + p;
        global_sum += exp_sums[idx] * expf(max_logits[idx] - global_max);
    }

    // Combine outputs
    float* out = output + seq_idx * num_heads * head_dim + head_idx * head_dim;

    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;

        for (uint32_t p = 0; p < num_partitions; p++) {
            uint32_t stat_idx = seq_idx * num_heads * num_partitions + head_idx * num_partitions + p;
            const float* part_out = partition_outputs +
                (seq_idx * num_heads * num_partitions + head_idx * num_partitions + p) * head_dim;

            float weight = exp_sums[stat_idx] * expf(max_logits[stat_idx] - global_max) / global_sum;
            acc += weight * part_out[d];
        }

        out[d] = acc;
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_paged_attention_v1(
    void* output,
    const void* query,
    const void* key_cache,
    const void* value_cache,
    const int32_t* block_tables,
    const int32_t* context_lens,
    float scale,
    uint32_t num_seqs,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t block_size,
    uint32_t max_blocks_per_seq,
    uint32_t max_context_len,
    cudaStream_t stream
) {
    dim3 blocks(num_seqs, num_heads);
    uint32_t threads = 256;
    size_t shared_size = (max_context_len + head_dim) * sizeof(float);

    paged_attention_v1_kernel<<<blocks, threads, shared_size, stream>>>(
        (float*)output,
        (const float*)query,
        (const float*)key_cache,
        (const float*)value_cache,
        block_tables,
        context_lens,
        scale,
        num_seqs, num_heads, head_dim, block_size, max_blocks_per_seq
    );

    return cudaGetLastError();
}

int lux_cuda_paged_attention_v2(
    void* output,
    void* tmp_output,
    void* exp_sums,
    void* max_logits,
    const void* query,
    const void* key_cache,
    const void* value_cache,
    const int32_t* block_tables,
    const int32_t* context_lens,
    float scale,
    uint32_t num_seqs,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t block_size,
    uint32_t max_blocks_per_seq,
    uint32_t partition_size,
    uint32_t max_partitions,
    cudaStream_t stream
) {
    // Phase 1: Compute partitioned attention
    dim3 blocks1(num_seqs, num_heads, max_partitions);
    uint32_t threads = 256;
    size_t shared_size = (partition_size + head_dim) * sizeof(float);

    paged_attention_v2_kernel<<<blocks1, threads, shared_size, stream>>>(
        (float*)tmp_output,
        (float*)exp_sums,
        (float*)max_logits,
        (const float*)query,
        (const float*)key_cache,
        (const float*)value_cache,
        block_tables,
        context_lens,
        scale,
        num_seqs, num_heads, head_dim, block_size, max_blocks_per_seq, partition_size
    );

    // Phase 2: Reduce partitions
    dim3 blocks2(num_seqs, num_heads);

    paged_attention_reduce_kernel<<<blocks2, threads, 0, stream>>>(
        (float*)output,
        (const float*)tmp_output,
        (const float*)exp_sums,
        (const float*)max_logits,
        num_seqs, num_heads, head_dim, max_partitions
    );

    return cudaGetLastError();
}

}  // extern "C"
