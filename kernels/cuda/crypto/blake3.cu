// Copyright (c) 2024-2026 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// BLAKE3 Hash - High-Performance CUDA Implementation

#include <cuda_runtime.h>
#include <cstdint>

namespace lux {
namespace cuda {
namespace blake3 {

// BLAKE3 constants
__constant__ uint32_t IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

__constant__ uint32_t MSG_PERMUTATION[16] = {
    2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8
};

// Flags
constexpr uint32_t CHUNK_START = 1 << 0;
constexpr uint32_t CHUNK_END = 1 << 1;
constexpr uint32_t ROOT = 1 << 3;

// Rotate right
__device__ __forceinline__ uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

// G function - quarter round
__device__ __forceinline__ void g(
    uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d,
    uint32_t mx, uint32_t my
) {
    a = a + b + mx;
    d = rotr(d ^ a, 16);
    c = c + d;
    b = rotr(b ^ c, 12);
    a = a + b + my;
    d = rotr(d ^ a, 8);
    c = c + d;
    b = rotr(b ^ c, 7);
}

// Full round
__device__ void round(uint32_t state[16], const uint32_t m[16]) {
    // Column step
    g(state[0], state[4], state[8],  state[12], m[0],  m[1]);
    g(state[1], state[5], state[9],  state[13], m[2],  m[3]);
    g(state[2], state[6], state[10], state[14], m[4],  m[5]);
    g(state[3], state[7], state[11], state[15], m[6],  m[7]);
    
    // Diagonal step
    g(state[0], state[5], state[10], state[15], m[8],  m[9]);
    g(state[1], state[6], state[11], state[12], m[10], m[11]);
    g(state[2], state[7], state[8],  state[13], m[12], m[13]);
    g(state[3], state[4], state[9],  state[14], m[14], m[15]);
}

// Permute message schedule
__device__ void permute(uint32_t m[16]) {
    uint32_t permuted[16];
    for (int i = 0; i < 16; i++) {
        permuted[i] = m[MSG_PERMUTATION[i]];
    }
    for (int i = 0; i < 16; i++) {
        m[i] = permuted[i];
    }
}

// Compress function
__device__ void compress(
    uint32_t cv[8],
    const uint32_t block[16],
    uint64_t counter,
    uint32_t block_len,
    uint32_t flags
) {
    uint32_t state[16] = {
        cv[0], cv[1], cv[2], cv[3],
        cv[4], cv[5], cv[6], cv[7],
        IV[0], IV[1], IV[2], IV[3],
        (uint32_t)counter, (uint32_t)(counter >> 32),
        block_len, flags
    };
    
    uint32_t m[16];
    for (int i = 0; i < 16; i++) {
        m[i] = block[i];
    }
    
    // 7 rounds
    round(state, m); permute(m);
    round(state, m); permute(m);
    round(state, m); permute(m);
    round(state, m); permute(m);
    round(state, m); permute(m);
    round(state, m); permute(m);
    round(state, m);
    
    // Finalize
    for (int i = 0; i < 8; i++) {
        cv[i] = state[i] ^ state[i + 8];
    }
}

// Hash single chunk kernel
__global__ void hash_chunk_kernel(
    const uint8_t* __restrict__ input,
    uint32_t* __restrict__ output,
    uint32_t input_len,
    uint32_t flags
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize CV with IV
    uint32_t cv[8];
    for (int i = 0; i < 8; i++) {
        cv[i] = IV[i];
    }
    
    // Process single block (up to 64 bytes)
    uint32_t block[16] = {0};
    uint32_t offset = idx * 64;
    
    // Load input bytes into block words (little-endian)
    for (uint32_t i = 0; i < min(input_len, 64u); i += 4) {
        uint32_t word = 0;
        for (int j = 0; j < 4 && (i + j) < input_len; j++) {
            if (offset + i + j < input_len) {
                word |= ((uint32_t)input[offset + i + j]) << (j * 8);
            }
        }
        block[i / 4] = word;
    }
    
    // Compress with appropriate flags
    uint32_t chunk_flags = flags | CHUNK_START | CHUNK_END | ROOT;
    compress(cv, block, 0, min(input_len, 64u), chunk_flags);
    
    // Write output
    for (int i = 0; i < 8; i++) {
        output[idx * 8 + i] = cv[i];
    }
}

// Batch hash kernel - one thread per input
__global__ void hash_batch_kernel(
    const uint8_t* __restrict__ inputs,
    uint32_t* __restrict__ outputs,
    const uint32_t* __restrict__ input_lengths,
    uint32_t input_stride,
    uint32_t num_inputs
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_inputs) return;
    
    const uint8_t* input = inputs + idx * input_stride;
    uint32_t input_len = input_lengths[idx];
    uint32_t* output = outputs + idx * 8;
    
    // Initialize CV
    uint32_t cv[8];
    for (int i = 0; i < 8; i++) {
        cv[i] = IV[i];
    }
    
    uint64_t counter = 0;
    uint32_t remaining = input_len;
    
    while (remaining > 0) {
        // Load block
        uint32_t block[16] = {0};
        uint32_t block_len = min(remaining, 64u);
        uint32_t offset = input_len - remaining;
        
        for (uint32_t i = 0; i < block_len; i += 4) {
            uint32_t word = 0;
            for (int j = 0; j < 4 && (i + j) < block_len; j++) {
                word |= ((uint32_t)input[offset + i + j]) << (j * 8);
            }
            block[i / 4] = word;
        }
        
        // Determine flags
        uint32_t flags = 0;
        if (counter == 0) flags |= CHUNK_START;
        if (remaining <= 64) flags |= CHUNK_END | ROOT;
        
        compress(cv, block, counter, block_len, flags);
        
        counter++;
        remaining -= block_len;
    }
    
    // Write final hash
    for (int i = 0; i < 8; i++) {
        output[i] = cv[i];
    }
}

// Host functions
void hash(
    const uint8_t* input,
    uint32_t* output,
    uint32_t input_len,
    cudaStream_t stream
) {
    dim3 block(1);
    dim3 grid(1);
    hash_chunk_kernel<<<grid, block, 0, stream>>>(input, output, input_len, 0);
}

void hash_batch(
    const uint8_t* inputs,
    uint32_t* outputs,
    const uint32_t* input_lengths,
    uint32_t input_stride,
    uint32_t num_inputs,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((num_inputs + block.x - 1) / block.x);
    hash_batch_kernel<<<grid, block, 0, stream>>>(
        inputs, outputs, input_lengths, input_stride, num_inputs
    );
}

} // namespace blake3
} // namespace cuda
} // namespace lux

// =============================================================================
// C API for CGO Bindings
// =============================================================================

extern "C" {

int lux_cuda_blake3_hash(
    const uint8_t* input,
    uint32_t* output,
    uint32_t input_len,
    cudaStream_t stream
) {
    lux::cuda::blake3::hash(input, output, input_len, stream);
    return cudaGetLastError();
}

int lux_cuda_blake3_hash_batch(
    const uint8_t* inputs,
    uint32_t* outputs,
    const uint32_t* input_lengths,
    uint32_t input_stride,
    uint32_t num_inputs,
    cudaStream_t stream
) {
    lux::cuda::blake3::hash_batch(
        inputs, outputs, input_lengths, input_stride, num_inputs, stream
    );
    return cudaGetLastError();
}

} // extern "C"
