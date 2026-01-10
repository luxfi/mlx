// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// CUDA Kernels - Compiled kernel implementations
// This file includes all kernel sources for compilation into the plugin.
// PTX is generated at build time and embedded into the plugin.

#include <cuda_runtime.h>
#include <cstdint>

// =============================================================================
// Common Utilities
// =============================================================================

namespace lux {
namespace cuda {

// Thread-safe error checking
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) return err; \
} while(0)

// Grid/block size helpers
inline dim3 get_block_1d(size_t n, int block_size = 256) {
    return dim3(block_size);
}

inline dim3 get_grid_1d(size_t n, int block_size = 256) {
    return dim3((n + block_size - 1) / block_size);
}

// =============================================================================
// Tensor Operations
// =============================================================================

// Binary element-wise operations
__global__ void add_f32_kernel(const float* a, const float* b, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

__global__ void sub_f32_kernel(const float* a, const float* b, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] - b[idx];
}

__global__ void mul_f32_kernel(const float* a, const float* b, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] * b[idx];
}

__global__ void div_f32_kernel(const float* a, const float* b, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] / b[idx];
}

// Unary operations
__global__ void exp_f32_kernel(const float* in, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = expf(in[idx]);
}

__global__ void log_f32_kernel(const float* in, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = logf(in[idx]);
}

__global__ void sqrt_f32_kernel(const float* in, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = sqrtf(in[idx]);
}

__global__ void tanh_f32_kernel(const float* in, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = tanhf(in[idx]);
}

__global__ void sigmoid_f32_kernel(const float* in, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = 1.0f / (1.0f + expf(-in[idx]));
}

__global__ void relu_f32_kernel(const float* in, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fmaxf(0.0f, in[idx]);
}

__global__ void gelu_f32_kernel(const float* in, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        out[idx] = x * cdf;
    }
}

// =============================================================================
// Reduction Operations
// =============================================================================

template<int BLOCK_SIZE>
__global__ void reduce_sum_f32_kernel(const float* in, float* out, size_t n) {
    __shared__ float sdata[BLOCK_SIZE];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(out, sdata[0]);
}

template<int BLOCK_SIZE>
__global__ void reduce_max_f32_kernel(const float* in, float* out, size_t n) {
    __shared__ float sdata[BLOCK_SIZE];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? in[idx] : -INFINITY;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    // Use atomicMax with float reinterpretation (works for positive floats)
    if (tid == 0) {
        // For a full implementation, use a proper float atomicMax
        // This is simplified
    }
}

// =============================================================================
// Softmax
// =============================================================================

__global__ void softmax_f32_kernel(const float* in, float* out, size_t batch_size, size_t dim) {
    extern __shared__ float sdata[];

    size_t batch_idx = blockIdx.x;
    size_t tid = threadIdx.x;

    const float* row_in = in + batch_idx * dim;
    float* row_out = out + batch_idx * dim;

    // Find max for numerical stability
    float max_val = -INFINITY;
    for (size_t i = tid; i < dim; i += blockDim.x) {
        max_val = fmaxf(max_val, row_in[i]);
    }
    sdata[tid] = max_val;
    __syncthreads();

    // Reduce to find global max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();

    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (size_t i = tid; i < dim; i += blockDim.x) {
        float val = expf(row_in[i] - max_val);
        row_out[i] = val;
        sum += val;
    }
    sdata[tid] = sum;
    __syncthreads();

    // Reduce to find total sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    sum = sdata[0];

    // Normalize
    for (size_t i = tid; i < dim; i += blockDim.x) {
        row_out[i] /= sum;
    }
}

// =============================================================================
// Layer/RMS Normalization
// =============================================================================

__global__ void rms_norm_f32_kernel(const float* in, float* out, const float* weight,
                                     size_t batch_size, size_t dim, float eps) {
    extern __shared__ float sdata[];

    size_t batch_idx = blockIdx.x;
    size_t tid = threadIdx.x;

    const float* row_in = in + batch_idx * dim;
    float* row_out = out + batch_idx * dim;

    // Compute sum of squares
    float sq_sum = 0.0f;
    for (size_t i = tid; i < dim; i += blockDim.x) {
        float val = row_in[i];
        sq_sum += val * val;
    }
    sdata[tid] = sq_sum;
    __syncthreads();

    // Reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float rms = rsqrtf(sdata[0] / dim + eps);

    // Normalize and scale
    for (size_t i = tid; i < dim; i += blockDim.x) {
        row_out[i] = row_in[i] * rms * weight[i];
    }
}

// =============================================================================
// Matrix Multiplication (basic - for production use cuBLAS)
// =============================================================================

__global__ void matmul_f32_kernel(const float* A, const float* B, float* C,
                                   int M, int K, int N) {
    // Tiled matrix multiplication
    const int TILE_SIZE = 16;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        // Compute
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// =============================================================================
// NTT (Number Theoretic Transform)
// =============================================================================

// Modular arithmetic helpers
__device__ __forceinline__ uint64_t mod_add(uint64_t a, uint64_t b, uint64_t mod) {
    uint64_t sum = a + b;
    return (sum >= mod) ? (sum - mod) : sum;
}

__device__ __forceinline__ uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t mod) {
    return (a >= b) ? (a - b) : (mod - b + a);
}

__device__ __forceinline__ uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t mod) {
    // For 64-bit modulus, use __uint128_t if available or Montgomery multiplication
    unsigned __int128 prod = (unsigned __int128)a * b;
    return (uint64_t)(prod % mod);
}

__global__ void ntt_radix2_kernel(uint64_t* data, size_t n, uint64_t mod,
                                   const uint64_t* twiddles, int stage) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    size_t half = 1ULL << stage;
    size_t full = half << 1;
    size_t group = idx / half;
    size_t pos = idx % half;

    if (group * full + pos + half >= n) return;

    size_t i = group * full + pos;
    size_t j = i + half;

    uint64_t w = twiddles[pos * (n / full)];
    uint64_t u = data[i];
    uint64_t v = mod_mul(data[j], w, mod);

    data[i] = mod_add(u, v, mod);
    data[j] = mod_sub(u, v, mod);
}

// =============================================================================
// MSM (Multi-Scalar Multiplication) - Pippenger's Algorithm
// =============================================================================

// Simplified MSM kernel - full implementation requires elliptic curve operations
__global__ void msm_bucket_accumulate_kernel(const uint64_t* scalars,
                                              const void* points,
                                              void* buckets,
                                              size_t n,
                                              int window_bits,
                                              int window_idx) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Extract window from scalar
    uint64_t scalar = scalars[idx];
    int bucket_idx = (scalar >> (window_bits * window_idx)) & ((1 << window_bits) - 1);

    // Accumulate point into bucket (placeholder - needs EC point addition)
    // In production, this would perform actual elliptic curve point addition
}

// =============================================================================
// Poseidon2 Hash
// =============================================================================

// Poseidon2 constants (simplified - full implementation needs proper constants)
__constant__ uint64_t POSEIDON2_ROUND_CONSTANTS[64];
__constant__ uint64_t POSEIDON2_MDS_MATRIX[9];

__device__ void poseidon2_full_round(uint64_t* state, int round, uint64_t mod) {
    // Add round constants
    for (int i = 0; i < 3; i++) {
        state[i] = mod_add(state[i], POSEIDON2_ROUND_CONSTANTS[round * 3 + i], mod);
    }

    // S-box (x^5)
    for (int i = 0; i < 3; i++) {
        uint64_t x2 = mod_mul(state[i], state[i], mod);
        uint64_t x4 = mod_mul(x2, x2, mod);
        state[i] = mod_mul(x4, state[i], mod);
    }

    // MDS matrix multiplication
    uint64_t new_state[3];
    for (int i = 0; i < 3; i++) {
        new_state[i] = 0;
        for (int j = 0; j < 3; j++) {
            new_state[i] = mod_add(new_state[i],
                                    mod_mul(POSEIDON2_MDS_MATRIX[i * 3 + j], state[j], mod),
                                    mod);
        }
    }
    for (int i = 0; i < 3; i++) state[i] = new_state[i];
}

__global__ void poseidon2_hash_kernel(const uint64_t* inputs, uint64_t* outputs,
                                       size_t n, uint64_t mod) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint64_t state[3];
    state[0] = inputs[idx * 2];
    state[1] = inputs[idx * 2 + 1];
    state[2] = 0;

    // 8 full rounds (simplified)
    for (int r = 0; r < 8; r++) {
        poseidon2_full_round(state, r, mod);
    }

    outputs[idx] = state[0];
}

// =============================================================================
// Entry Points (C linkage for plugin)
// =============================================================================

} // namespace cuda
} // namespace lux

extern "C" {

cudaError_t lux_cuda_add_f32(const float* a, const float* b, float* out, size_t n, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    lux::cuda::add_f32_kernel<<<grid, block, 0, stream>>>(a, b, out, n);
    return cudaGetLastError();
}

cudaError_t lux_cuda_sub_f32(const float* a, const float* b, float* out, size_t n, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    lux::cuda::sub_f32_kernel<<<grid, block, 0, stream>>>(a, b, out, n);
    return cudaGetLastError();
}

cudaError_t lux_cuda_mul_f32(const float* a, const float* b, float* out, size_t n, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    lux::cuda::mul_f32_kernel<<<grid, block, 0, stream>>>(a, b, out, n);
    return cudaGetLastError();
}

cudaError_t lux_cuda_div_f32(const float* a, const float* b, float* out, size_t n, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    lux::cuda::div_f32_kernel<<<grid, block, 0, stream>>>(a, b, out, n);
    return cudaGetLastError();
}

cudaError_t lux_cuda_exp_f32(const float* in, float* out, size_t n, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    lux::cuda::exp_f32_kernel<<<grid, block, 0, stream>>>(in, out, n);
    return cudaGetLastError();
}

cudaError_t lux_cuda_relu_f32(const float* in, float* out, size_t n, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    lux::cuda::relu_f32_kernel<<<grid, block, 0, stream>>>(in, out, n);
    return cudaGetLastError();
}

cudaError_t lux_cuda_gelu_f32(const float* in, float* out, size_t n, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    lux::cuda::gelu_f32_kernel<<<grid, block, 0, stream>>>(in, out, n);
    return cudaGetLastError();
}

cudaError_t lux_cuda_softmax_f32(const float* in, float* out, size_t batch_size, size_t dim, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid(batch_size);
    size_t shared_mem = 256 * sizeof(float);
    lux::cuda::softmax_f32_kernel<<<grid, block, shared_mem, stream>>>(in, out, batch_size, dim);
    return cudaGetLastError();
}

cudaError_t lux_cuda_rms_norm_f32(const float* in, float* out, const float* weight,
                                   size_t batch_size, size_t dim, float eps, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid(batch_size);
    size_t shared_mem = 256 * sizeof(float);
    lux::cuda::rms_norm_f32_kernel<<<grid, block, shared_mem, stream>>>(in, out, weight, batch_size, dim, eps);
    return cudaGetLastError();
}

cudaError_t lux_cuda_matmul_f32(const float* A, const float* B, float* C,
                                 int M, int K, int N, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    lux::cuda::matmul_f32_kernel<<<grid, block, 0, stream>>>(A, B, C, M, K, N);
    return cudaGetLastError();
}

cudaError_t lux_cuda_ntt_forward(uint64_t* data, size_t n, uint64_t mod,
                                  const uint64_t* twiddles, cudaStream_t stream) {
    // Perform log2(n) stages of radix-2 NTT
    int log_n = 0;
    for (size_t temp = n; temp > 1; temp >>= 1) log_n++;

    dim3 block(256);
    for (int stage = 0; stage < log_n; stage++) {
        size_t work_items = n / 2;
        dim3 grid((work_items + 255) / 256);
        lux::cuda::ntt_radix2_kernel<<<grid, block, 0, stream>>>(data, n, mod, twiddles, stage);
    }
    return cudaGetLastError();
}

cudaError_t lux_cuda_poseidon2_hash(const uint64_t* inputs, uint64_t* outputs,
                                     size_t n, uint64_t mod, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    lux::cuda::poseidon2_hash_kernel<<<grid, block, 0, stream>>>(inputs, outputs, n, mod);
    return cudaGetLastError();
}

} // extern "C"
