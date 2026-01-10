// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file ntt_unified_memory.cu
 * @brief NTT implementation using CUDA Unified Memory (managed memory)
 *
 * This implementation leverages CUDA Unified Memory for:
 * 1. Automatic memory migration between CPU and GPU
 * 2. Simplified memory management for very large polynomials
 * 3. Support for polynomials larger than GPU memory (via demand paging)
 * 4. Easy CPU fallback for debugging and verification
 *
 * Key features:
 * - Prefetching hints for optimal performance
 * - Memory advice for access patterns
 * - Multi-GPU support with peer access
 * - Hybrid CPU-GPU execution for extreme sizes
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// Configuration
// ============================================================================

#define UM_BLOCK_SIZE 256
#define UM_PREFETCH_AHEAD 2       // Number of chunks to prefetch ahead
#define UM_MIN_GPU_ELEMENTS 1024  // Minimum elements for GPU execution
#define UM_CHUNK_SIZE 65536       // Chunk size for streaming

// ============================================================================
// Device Helper Functions
// ============================================================================

__device__ __forceinline__ uint64_t um_barrett_mul(
    uint64_t a,
    uint64_t b,
    uint64_t q,
    uint64_t mu
) {
    unsigned __int128 prod = (unsigned __int128)a * b;
    uint64_t lo = (uint64_t)prod;
    uint64_t hi = (uint64_t)(prod >> 64);

    unsigned __int128 tmp = (unsigned __int128)hi * mu;
    uint64_t q_approx = (uint64_t)(tmp >> 64);

    uint64_t result = lo - q_approx * q;
    if (result >= q) result -= q;
    if (result >= q) result -= q;

    return result;
}

__device__ __forceinline__ uint64_t um_mod_add(uint64_t a, uint64_t b, uint64_t q) {
    uint64_t sum = a + b;
    return (sum >= q) ? (sum - q) : sum;
}

__device__ __forceinline__ uint64_t um_mod_sub(uint64_t a, uint64_t b, uint64_t q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

// ============================================================================
// Host Helper Functions (CPU Barrett Multiplication)
// ============================================================================

__host__ inline uint64_t um_barrett_mul_host(
    uint64_t a,
    uint64_t b,
    uint64_t q,
    uint64_t mu
) {
    __uint128_t prod = (__uint128_t)a * b;
    uint64_t lo = (uint64_t)prod;
    uint64_t hi = (uint64_t)(prod >> 64);

    __uint128_t tmp = (__uint128_t)hi * mu;
    uint64_t q_approx = (uint64_t)(tmp >> 64);

    uint64_t result = lo - q_approx * q;
    if (result >= q) result -= q;
    if (result >= q) result -= q;

    return result;
}

__host__ inline uint64_t um_mod_add_host(uint64_t a, uint64_t b, uint64_t q) {
    uint64_t sum = a + b;
    return (sum >= q) ? (sum - q) : sum;
}

__host__ inline uint64_t um_mod_sub_host(uint64_t a, uint64_t b, uint64_t q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

// ============================================================================
// Unified Memory NTT Kernels
// ============================================================================

/**
 * @brief Single-stage NTT kernel for unified memory
 *
 * Processes one butterfly stage. Multiple launches perform full NTT.
 * This approach works well with unified memory's page migration.
 */
extern "C" __global__ void um_ntt_stage_kernel(
    uint64_t* data,
    const uint64_t* twiddles,
    uint32_t n,
    uint32_t stage,
    uint64_t q,
    uint64_t mu
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t num_butterflies = n / 2;

    if (idx >= num_butterflies) return;

    uint32_t m = 1u << (stage + 1);
    uint32_t half_m = m >> 1;

    uint32_t j = idx % half_m;
    uint32_t i = (idx / half_m) * m + j;

    uint64_t w = twiddles[half_m + j];
    uint64_t u = data[i];
    uint64_t t = um_barrett_mul(w, data[i + half_m], q, mu);

    data[i] = um_mod_add(u, t, q);
    data[i + half_m] = um_mod_sub(u, t, q);
}

/**
 * @brief Single-stage inverse NTT kernel
 */
extern "C" __global__ void um_intt_stage_kernel(
    uint64_t* data,
    const uint64_t* inv_twiddles,
    uint32_t n,
    uint32_t stage,
    uint64_t q,
    uint64_t mu
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t num_butterflies = n / 2;

    if (idx >= num_butterflies) return;

    uint32_t m = 1u << (stage + 1);
    uint32_t half_m = m >> 1;

    uint32_t j = idx % half_m;
    uint32_t i = (idx / half_m) * m + j;

    uint64_t w = inv_twiddles[half_m + j];
    uint64_t u = data[i];
    uint64_t v = data[i + half_m];

    data[i] = um_mod_add(u, v, q);
    uint64_t diff = um_mod_sub(u, v, q);
    data[i + half_m] = um_barrett_mul(diff, w, q, mu);
}

/**
 * @brief Scale by n^(-1) for inverse NTT
 */
extern "C" __global__ void um_scale_kernel(
    uint64_t* data,
    uint32_t n,
    uint64_t n_inv,
    uint64_t q,
    uint64_t mu
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    data[idx] = um_barrett_mul(data[idx], n_inv, q, mu);
}

/**
 * @brief Multi-stage NTT kernel with shared memory
 *
 * Processes multiple consecutive stages in one kernel to reduce
 * unified memory page migrations.
 */
extern "C" __global__ void um_ntt_multi_stage_kernel(
    uint64_t* data,
    const uint64_t* twiddles,
    uint32_t n,
    uint32_t start_stage,
    uint32_t end_stage,
    uint64_t q,
    uint64_t mu
) {
    extern __shared__ uint64_t shared[];

    uint32_t tid = threadIdx.x;
    uint32_t block_start = blockIdx.x * blockDim.x * 2;

    // Load data to shared memory (2 elements per thread)
    if (block_start + tid < n) {
        shared[tid] = data[block_start + tid];
    }
    if (block_start + blockDim.x + tid < n) {
        shared[blockDim.x + tid] = data[block_start + blockDim.x + tid];
    }

    __syncthreads();

    // Process stages
    uint32_t local_n = min(blockDim.x * 2, n - block_start);

    for (uint32_t stage = start_stage; stage <= end_stage; stage++) {
        uint32_t m = 1u << (stage + 1);
        uint32_t half_m = m >> 1;

        // Check if butterfly pairs are within this block
        uint32_t butterflies_per_block = local_n / 2;

        for (uint32_t k = tid; k < butterflies_per_block; k += blockDim.x) {
            uint32_t j = k % half_m;
            uint32_t local_i = (k / half_m) * m + j;

            if (local_i + half_m < local_n) {
                // Global twiddle index
                uint64_t w = twiddles[half_m + j];

                uint64_t u = shared[local_i];
                uint64_t t = um_barrett_mul(w, shared[local_i + half_m], q, mu);

                shared[local_i] = um_mod_add(u, t, q);
                shared[local_i + half_m] = um_mod_sub(u, t, q);
            }
        }

        __syncthreads();
    }

    // Store back
    if (block_start + tid < n) {
        data[block_start + tid] = shared[tid];
    }
    if (block_start + blockDim.x + tid < n) {
        data[block_start + blockDim.x + tid] = shared[blockDim.x + tid];
    }
}

/**
 * @brief Chunked NTT for streaming unified memory access
 *
 * Processes data in chunks to enable efficient prefetching.
 */
extern "C" __global__ void um_ntt_chunked_stage_kernel(
    uint64_t* data,
    const uint64_t* twiddles,
    uint32_t n,
    uint32_t stage,
    uint32_t chunk_start,
    uint32_t chunk_size,
    uint64_t q,
    uint64_t mu
) {
    uint32_t local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t chunk_butterflies = chunk_size / 2;

    if (local_idx >= chunk_butterflies) return;

    uint32_t global_idx = chunk_start / 2 + local_idx;
    uint32_t num_butterflies = n / 2;

    if (global_idx >= num_butterflies) return;

    uint32_t m = 1u << (stage + 1);
    uint32_t half_m = m >> 1;

    uint32_t j = global_idx % half_m;
    uint32_t i = (global_idx / half_m) * m + j;

    if (i + half_m >= n) return;

    uint64_t w = twiddles[half_m + j];
    uint64_t u = data[i];
    uint64_t t = um_barrett_mul(w, data[i + half_m], q, mu);

    data[i] = um_mod_add(u, t, q);
    data[i + half_m] = um_mod_sub(u, t, q);
}

// ============================================================================
// Batch NTT with Unified Memory
// ============================================================================

/**
 * @brief Batched single-stage NTT
 */
extern "C" __global__ void um_ntt_batch_stage_kernel(
    uint64_t* data,
    const uint64_t* twiddles,
    uint32_t n,
    uint32_t stage,
    uint32_t batch_count,
    uint64_t q,
    uint64_t mu
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_butterflies = (n / 2) * batch_count;

    if (idx >= total_butterflies) return;

    uint32_t batch = idx / (n / 2);
    uint32_t butterfly = idx % (n / 2);

    uint64_t* poly = data + batch * n;

    uint32_t m = 1u << (stage + 1);
    uint32_t half_m = m >> 1;

    uint32_t j = butterfly % half_m;
    uint32_t i = (butterfly / half_m) * m + j;

    uint64_t w = twiddles[half_m + j];
    uint64_t u = poly[i];
    uint64_t t = um_barrett_mul(w, poly[i + half_m], q, mu);

    poly[i] = um_mod_add(u, t, q);
    poly[i + half_m] = um_mod_sub(u, t, q);
}

// ============================================================================
// Memory Management Functions
// ============================================================================

/**
 * @brief Unified memory allocation info
 */
struct UMAllocInfo {
    void* ptr;
    size_t size;
    int device;
    bool prefetch_to_gpu;
};

/**
 * @brief Prefetch data to specific device
 */
extern "C" __host__ cudaError_t um_prefetch_async(
    void* ptr,
    size_t size,
    int device,
    cudaStream_t stream
) {
    return cudaMemPrefetchAsync(ptr, size, device, stream);
}

/**
 * @brief Set memory access hints
 */
extern "C" __host__ cudaError_t um_advise_access(
    void* ptr,
    size_t size,
    cudaMemoryAdvise advice,
    int device
) {
    return cudaMemAdvise(ptr, size, advice, device);
}

// ============================================================================
// CPU Fallback Implementation
// ============================================================================

/**
 * @brief CPU NTT implementation for verification and fallback
 */
void cpu_ntt_forward(
    uint64_t* data,
    const uint64_t* twiddles,
    uint32_t n,
    uint32_t log_n,
    uint64_t q,
    uint64_t mu
) {
    for (uint32_t stage = 0; stage < log_n; stage++) {
        uint32_t m = 1u << (stage + 1);
        uint32_t half_m = m >> 1;

        for (uint32_t k = 0; k < n / 2; k++) {
            uint32_t j = k % half_m;
            uint32_t i = (k / half_m) * m + j;

            uint64_t w = twiddles[half_m + j];
            uint64_t u = data[i];
            uint64_t t = um_barrett_mul_host(w, data[i + half_m], q, mu);

            data[i] = um_mod_add_host(u, t, q);
            data[i + half_m] = um_mod_sub_host(u, t, q);
        }
    }
}

/**
 * @brief CPU inverse NTT implementation
 */
void cpu_ntt_inverse(
    uint64_t* data,
    const uint64_t* inv_twiddles,
    uint64_t n_inv,
    uint32_t n,
    uint32_t log_n,
    uint64_t q,
    uint64_t mu
) {
    for (int stage = log_n - 1; stage >= 0; stage--) {
        uint32_t m = 1u << (stage + 1);
        uint32_t half_m = m >> 1;

        for (uint32_t k = 0; k < n / 2; k++) {
            uint32_t j = k % half_m;
            uint32_t i = (k / half_m) * m + j;

            uint64_t w = inv_twiddles[half_m + j];
            uint64_t u = data[i];
            uint64_t v = data[i + half_m];

            data[i] = um_mod_add_host(u, v, q);
            uint64_t diff = um_mod_sub_host(u, v, q);
            data[i + half_m] = um_barrett_mul_host(diff, w, q, mu);
        }
    }

    // Scale
    for (uint32_t i = 0; i < n; i++) {
        data[i] = um_barrett_mul_host(data[i], n_inv, q, mu);
    }
}

// ============================================================================
// C API Functions
// ============================================================================

extern "C" {

/**
 * @brief Allocate unified memory for NTT
 */
cudaError_t lux_cuda_um_allocate(
    uint64_t** ptr,
    size_t num_elements,
    int preferred_device
) {
    cudaError_t err = cudaMallocManaged(ptr, num_elements * sizeof(uint64_t));
    if (err != cudaSuccess) return err;

    // Set preferred location
    if (preferred_device >= 0) {
        err = cudaMemAdvise(*ptr, num_elements * sizeof(uint64_t),
                          cudaMemAdviseSetPreferredLocation, preferred_device);
        if (err != cudaSuccess) {
            cudaFree(*ptr);
            return err;
        }
    }

    return cudaSuccess;
}

/**
 * @brief Free unified memory
 */
cudaError_t lux_cuda_um_free(uint64_t* ptr) {
    return cudaFree(ptr);
}

/**
 * @brief Forward NTT using unified memory with automatic prefetching
 */
cudaError_t lux_cuda_um_ntt_forward(
    uint64_t* data,
    const uint64_t* twiddles,
    uint32_t n,
    uint32_t log_n,
    uint64_t q,
    uint64_t mu,
    int device,
    cudaStream_t stream
) {
    // Prefetch data to GPU
    size_t data_size = n * sizeof(uint64_t);
    size_t twiddle_size = n * sizeof(uint64_t);

    cudaMemPrefetchAsync((void*)data, data_size, device, stream);
    cudaMemPrefetchAsync((void*)twiddles, twiddle_size, device, stream);

    // Launch one kernel per stage
    uint32_t blocks = (n / 2 + UM_BLOCK_SIZE - 1) / UM_BLOCK_SIZE;

    for (uint32_t stage = 0; stage < log_n; stage++) {
        um_ntt_stage_kernel<<<blocks, UM_BLOCK_SIZE, 0, stream>>>(
            data, twiddles, n, stage, q, mu
        );
    }

    return cudaGetLastError();
}

/**
 * @brief Inverse NTT using unified memory
 */
cudaError_t lux_cuda_um_ntt_inverse(
    uint64_t* data,
    const uint64_t* inv_twiddles,
    uint64_t n_inv,
    uint32_t n,
    uint32_t log_n,
    uint64_t q,
    uint64_t mu,
    int device,
    cudaStream_t stream
) {
    size_t data_size = n * sizeof(uint64_t);
    cudaMemPrefetchAsync((void*)data, data_size, device, stream);
    cudaMemPrefetchAsync((void*)inv_twiddles, n * sizeof(uint64_t), device, stream);

    uint32_t blocks = (n / 2 + UM_BLOCK_SIZE - 1) / UM_BLOCK_SIZE;

    // Stages in reverse order
    for (int stage = log_n - 1; stage >= 0; stage--) {
        um_intt_stage_kernel<<<blocks, UM_BLOCK_SIZE, 0, stream>>>(
            data, inv_twiddles, n, (uint32_t)stage, q, mu
        );
    }

    // Scale
    uint32_t scale_blocks = (n + UM_BLOCK_SIZE - 1) / UM_BLOCK_SIZE;
    um_scale_kernel<<<scale_blocks, UM_BLOCK_SIZE, 0, stream>>>(
        data, n, n_inv, q, mu
    );

    return cudaGetLastError();
}

/**
 * @brief Batched NTT with unified memory
 */
cudaError_t lux_cuda_um_ntt_forward_batch(
    uint64_t* data,
    const uint64_t* twiddles,
    uint32_t n,
    uint32_t log_n,
    uint32_t batch_count,
    uint64_t q,
    uint64_t mu,
    int device,
    cudaStream_t stream
) {
    size_t total_size = (size_t)n * batch_count * sizeof(uint64_t);
    cudaMemPrefetchAsync((void*)data, total_size, device, stream);
    cudaMemPrefetchAsync((void*)twiddles, n * sizeof(uint64_t), device, stream);

    uint32_t total_butterflies = (n / 2) * batch_count;
    uint32_t blocks = (total_butterflies + UM_BLOCK_SIZE - 1) / UM_BLOCK_SIZE;

    for (uint32_t stage = 0; stage < log_n; stage++) {
        um_ntt_batch_stage_kernel<<<blocks, UM_BLOCK_SIZE, 0, stream>>>(
            data, twiddles, n, stage, batch_count, q, mu
        );
    }

    return cudaGetLastError();
}

/**
 * @brief Chunked NTT with explicit prefetching for large polynomials
 *
 * Suitable for polynomials larger than GPU memory.
 */
cudaError_t lux_cuda_um_ntt_forward_chunked(
    uint64_t* data,
    const uint64_t* twiddles,
    uint32_t n,
    uint32_t log_n,
    uint32_t chunk_size,
    uint64_t q,
    uint64_t mu,
    int device,
    cudaStream_t stream
) {
    // Prefetch twiddles (they're reused)
    cudaMemPrefetchAsync((void*)twiddles, n * sizeof(uint64_t), device, stream);

    uint32_t num_chunks = (n + chunk_size - 1) / chunk_size;

    for (uint32_t stage = 0; stage < log_n; stage++) {
        for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
            uint32_t chunk_start = chunk * chunk_size;
            uint32_t actual_chunk_size = min(chunk_size, n - chunk_start);

            // Prefetch current chunk
            cudaMemPrefetchAsync(
                (void*)(data + chunk_start),
                actual_chunk_size * sizeof(uint64_t),
                device,
                stream
            );

            // Prefetch next chunk (if exists)
            if (chunk + 1 < num_chunks) {
                uint32_t next_start = (chunk + 1) * chunk_size;
                uint32_t next_size = min(chunk_size, n - next_start);
                cudaMemPrefetchAsync(
                    (void*)(data + next_start),
                    next_size * sizeof(uint64_t),
                    device,
                    stream
                );
            }

            // Process chunk
            uint32_t blocks = (actual_chunk_size / 2 + UM_BLOCK_SIZE - 1) / UM_BLOCK_SIZE;
            um_ntt_chunked_stage_kernel<<<blocks, UM_BLOCK_SIZE, 0, stream>>>(
                data, twiddles, n, stage, chunk_start, actual_chunk_size, q, mu
            );
        }
    }

    return cudaGetLastError();
}

/**
 * @brief Hybrid CPU-GPU NTT for extreme polynomial sizes
 *
 * Uses CPU for global stages, GPU for local stages.
 */
cudaError_t lux_cuda_um_ntt_hybrid(
    uint64_t* data,
    const uint64_t* twiddles,
    uint32_t n,
    uint32_t log_n,
    uint32_t gpu_stages,  // Number of stages to run on GPU
    uint64_t q,
    uint64_t mu,
    int device,
    cudaStream_t stream
) {
    // CPU stages (large butterflies)
    uint32_t cpu_stages = log_n - gpu_stages;

    // Ensure data is on CPU for initial stages
    cudaMemPrefetchAsync((void*)data, n * sizeof(uint64_t), cudaCpuDeviceId, stream);
    cudaStreamSynchronize(stream);

    for (uint32_t stage = 0; stage < cpu_stages; stage++) {
        uint32_t m = 1u << (stage + 1);
        uint32_t half_m = m >> 1;

        #pragma omp parallel for
        for (uint32_t k = 0; k < n / 2; k++) {
            uint32_t j = k % half_m;
            uint32_t i = (k / half_m) * m + j;

            uint64_t w = twiddles[half_m + j];
            uint64_t u = data[i];
            uint64_t t = um_barrett_mul_host(w, data[i + half_m], q, mu);

            data[i] = um_mod_add_host(u, t, q);
            data[i + half_m] = um_mod_sub_host(u, t, q);
        }
    }

    // GPU stages (small butterflies, high parallelism)
    cudaMemPrefetchAsync((void*)data, n * sizeof(uint64_t), device, stream);
    cudaMemPrefetchAsync((void*)twiddles, n * sizeof(uint64_t), device, stream);

    uint32_t blocks = (n / 2 + UM_BLOCK_SIZE - 1) / UM_BLOCK_SIZE;

    for (uint32_t stage = cpu_stages; stage < log_n; stage++) {
        um_ntt_stage_kernel<<<blocks, UM_BLOCK_SIZE, 0, stream>>>(
            data, twiddles, n, stage, q, mu
        );
    }

    return cudaGetLastError();
}

/**
 * @brief Set memory advice for NTT data patterns
 */
cudaError_t lux_cuda_um_set_ntt_advice(
    uint64_t* data,
    size_t size,
    int device
) {
    cudaError_t err;

    // Advise that data will be accessed by GPU
    err = cudaMemAdvise(data, size, cudaMemAdviseSetAccessedBy, device);
    if (err != cudaSuccess) return err;

    // For multi-GPU, enable read duplication
    err = cudaMemAdvise(data, size, cudaMemAdviseSetReadMostly, device);
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

/**
 * @brief Verify GPU result against CPU reference
 */
bool lux_cuda_um_verify_ntt(
    const uint64_t* gpu_result,
    uint64_t* cpu_data,  // Will be modified
    const uint64_t* twiddles,
    uint32_t n,
    uint32_t log_n,
    uint64_t q,
    uint64_t mu
) {
    // Run CPU NTT
    cpu_ntt_forward(cpu_data, twiddles, n, log_n, q, mu);

    // Compare
    for (uint32_t i = 0; i < n; i++) {
        if (gpu_result[i] != cpu_data[i]) {
            fprintf(stderr, "NTT mismatch at index %u: GPU=%lu, CPU=%lu\n",
                    i, gpu_result[i], cpu_data[i]);
            return false;
        }
    }

    return true;
}

/**
 * @brief Query unified memory support
 */
bool lux_cuda_um_supported(int device) {
    int managed_memory;
    cudaDeviceGetAttribute(&managed_memory, cudaDevAttrManagedMemory, device);
    return managed_memory != 0;
}

/**
 * @brief Query concurrent managed access support
 */
bool lux_cuda_um_concurrent_access(int device) {
    int concurrent;
    cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, device);
    return concurrent != 0;
}

/**
 * @brief Get recommended chunk size based on GPU memory
 */
size_t lux_cuda_um_recommended_chunk_size(int device) {
    size_t free_mem, total_mem;
    cudaSetDevice(device);
    cudaMemGetInfo(&free_mem, &total_mem);

    // Use 1/4 of free memory as chunk size, minimum 64K elements
    size_t chunk_elements = free_mem / (4 * sizeof(uint64_t));
    chunk_elements = (chunk_elements < UM_CHUNK_SIZE) ? UM_CHUNK_SIZE : chunk_elements;

    // Round to power of 2
    size_t result = 1;
    while (result < chunk_elements) result <<= 1;
    result >>= 1;

    return result;
}

/**
 * @brief Multi-GPU NTT using unified memory
 *
 * Distributes work across multiple GPUs.
 */
cudaError_t lux_cuda_um_ntt_multi_gpu(
    uint64_t* data,
    const uint64_t* twiddles,
    uint32_t n,
    uint32_t log_n,
    int* devices,
    int num_devices,
    uint64_t q,
    uint64_t mu
) {
    if (num_devices <= 0) return cudaErrorInvalidValue;

    // Enable peer access between devices
    for (int i = 0; i < num_devices; i++) {
        for (int j = 0; j < num_devices; j++) {
            if (i != j) {
                cudaSetDevice(devices[i]);
                cudaDeviceEnablePeerAccess(devices[j], 0);
            }
        }
    }

    // Create streams for each device
    cudaStream_t* streams = (cudaStream_t*)malloc(num_devices * sizeof(cudaStream_t));
    for (int i = 0; i < num_devices; i++) {
        cudaSetDevice(devices[i]);
        cudaStreamCreate(&streams[i]);
    }

    // Partition data across devices
    uint32_t elements_per_device = n / num_devices;

    // For early stages (global butterflies), process on device 0
    // For later stages, distribute across devices

    uint32_t global_stages = log_n > 4 ? log_n - 4 : 0;

    // Global stages on device 0
    cudaSetDevice(devices[0]);
    cudaMemPrefetchAsync((void*)data, n * sizeof(uint64_t), devices[0], streams[0]);
    cudaMemPrefetchAsync((void*)twiddles, n * sizeof(uint64_t), devices[0], streams[0]);

    uint32_t blocks = (n / 2 + UM_BLOCK_SIZE - 1) / UM_BLOCK_SIZE;

    for (uint32_t stage = 0; stage < global_stages; stage++) {
        um_ntt_stage_kernel<<<blocks, UM_BLOCK_SIZE, 0, streams[0]>>>(
            data, twiddles, n, stage, q, mu
        );
    }

    cudaStreamSynchronize(streams[0]);

    // Local stages distributed across devices
    for (uint32_t stage = global_stages; stage < log_n; stage++) {
        for (int d = 0; d < num_devices; d++) {
            cudaSetDevice(devices[d]);

            uint32_t start = d * elements_per_device;
            uint32_t local_n = (d == num_devices - 1) ? (n - start) : elements_per_device;

            // Prefetch partition to device
            cudaMemPrefetchAsync((void*)(data + start), local_n * sizeof(uint64_t),
                               devices[d], streams[d]);

            uint32_t local_butterflies = local_n / 2;
            uint32_t local_blocks = (local_butterflies + UM_BLOCK_SIZE - 1) / UM_BLOCK_SIZE;

            // Note: This is simplified; proper implementation needs to handle
            // butterfly pairs that cross partition boundaries
            um_ntt_chunked_stage_kernel<<<local_blocks, UM_BLOCK_SIZE, 0, streams[d]>>>(
                data, twiddles, n, stage, start, local_n, q, mu
            );
        }

        // Sync all devices before next stage
        for (int d = 0; d < num_devices; d++) {
            cudaSetDevice(devices[d]);
            cudaStreamSynchronize(streams[d]);
        }
    }

    // Cleanup
    for (int i = 0; i < num_devices; i++) {
        cudaSetDevice(devices[i]);
        cudaStreamDestroy(streams[i]);
    }
    free(streams);

    return cudaGetLastError();
}

}  // extern "C"
