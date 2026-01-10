// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Scan CUDA Kernels
// Prefix scan/cumsum operations (inclusive and exclusive)
// Implements Blelloch scan algorithm for efficient parallel prefix operations

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// ============================================================================
// Configuration
// ============================================================================

#define BLOCK_SIZE 256
#define ELEMENTS_PER_BLOCK (BLOCK_SIZE * 2)  // Each thread processes 2 elements
#define WARP_SIZE 32

// ============================================================================
// Scan Operation Types
// ============================================================================

enum class ScanOp : uint32_t {
    SUM = 0,
    PROD = 1,
    MAX = 2,
    MIN = 3,
};

// ============================================================================
// Operation Helpers
// ============================================================================

__device__ __forceinline__
float scan_op_float(float a, float b, ScanOp op) {
    switch (op) {
        case ScanOp::SUM: return a + b;
        case ScanOp::PROD: return a * b;
        case ScanOp::MAX: return fmaxf(a, b);
        case ScanOp::MIN: return fminf(a, b);
        default: return a + b;
    }
}

__device__ __forceinline__
float scan_identity_float(ScanOp op) {
    switch (op) {
        case ScanOp::SUM: return 0.0f;
        case ScanOp::PROD: return 1.0f;
        case ScanOp::MAX: return -__FLT_MAX__;
        case ScanOp::MIN: return __FLT_MAX__;
        default: return 0.0f;
    }
}

__device__ __forceinline__
int32_t scan_op_int32(int32_t a, int32_t b, ScanOp op) {
    switch (op) {
        case ScanOp::SUM: return a + b;
        case ScanOp::PROD: return a * b;
        case ScanOp::MAX: return max(a, b);
        case ScanOp::MIN: return min(a, b);
        default: return a + b;
    }
}

__device__ __forceinline__
int32_t scan_identity_int32(ScanOp op) {
    switch (op) {
        case ScanOp::SUM: return 0;
        case ScanOp::PROD: return 1;
        case ScanOp::MAX: return INT32_MIN;
        case ScanOp::MIN: return INT32_MAX;
        default: return 0;
    }
}

// ============================================================================
// Warp Scan (Kogge-Stone)
// ============================================================================

__device__ __forceinline__
float warp_scan_inclusive_sum(float val) {
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float n = __shfl_up_sync(0xffffffff, val, offset);
        if ((threadIdx.x & 31) >= offset) {
            val += n;
        }
    }
    return val;
}

__device__ __forceinline__
float warp_scan_exclusive_sum(float val) {
    float inclusive = warp_scan_inclusive_sum(val);
    return inclusive - val;
}

// ============================================================================
// Block Scan (Blelloch algorithm)
// ============================================================================

// Inclusive scan within a block
__device__ __forceinline__
float block_scan_inclusive_sum(float val, float* shared) {
    int tid = threadIdx.x;

    // Store in shared memory
    shared[tid] = val;
    __syncthreads();

    // Up-sweep (reduce) phase
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            shared[index] += shared[index - stride];
        }
        __syncthreads();
    }

    // Down-sweep phase
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index + stride < blockDim.x) {
            shared[index + stride] += shared[index];
        }
        __syncthreads();
    }

    return shared[tid];
}

// ============================================================================
// Single Block Scan Kernels
// ============================================================================

// Inclusive cumsum for arrays that fit in one block
extern "C" __global__
void scan_inclusive_sum_float_single_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    __shared__ float shared[BLOCK_SIZE];

    uint32_t tid = threadIdx.x;
    float val = (tid < n) ? input[tid] : 0.0f;

    val = block_scan_inclusive_sum(val, shared);

    if (tid < n) {
        output[tid] = val;
    }
}

// Exclusive cumsum for arrays that fit in one block
extern "C" __global__
void scan_exclusive_sum_float_single_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    __shared__ float shared[BLOCK_SIZE];

    uint32_t tid = threadIdx.x;
    float val = (tid < n) ? input[tid] : 0.0f;

    // Store shifted input
    shared[tid] = (tid > 0 && tid < n) ? input[tid - 1] : 0.0f;
    __syncthreads();

    // Inclusive scan on shifted input gives exclusive scan
    val = shared[tid];
    val = block_scan_inclusive_sum(val, shared);

    if (tid < n) {
        output[tid] = val;
    }
}

// ============================================================================
// Multi-Block Scan Kernels (for large arrays)
// ============================================================================

// Phase 1: Scan within blocks, store block sums
extern "C" __global__
void scan_inclusive_sum_float_phase1_kernel(
    float* __restrict__ output,
    float* __restrict__ block_sums,
    const float* __restrict__ input,
    uint64_t n
) {
    __shared__ float shared[BLOCK_SIZE];

    uint64_t block_offset = (uint64_t)blockIdx.x * blockDim.x;
    uint64_t idx = block_offset + threadIdx.x;

    float val = (idx < n) ? input[idx] : 0.0f;

    val = block_scan_inclusive_sum(val, shared);

    if (idx < n) {
        output[idx] = val;
    }

    // Last thread stores block sum
    if (threadIdx.x == blockDim.x - 1) {
        block_sums[blockIdx.x] = val;
    }
}

// Phase 2: Scan block sums
extern "C" __global__
void scan_block_sums_kernel(
    float* __restrict__ block_sums,
    uint32_t num_blocks
) {
    __shared__ float shared[BLOCK_SIZE];

    uint32_t tid = threadIdx.x;
    float val = (tid < num_blocks) ? block_sums[tid] : 0.0f;

    val = block_scan_inclusive_sum(val, shared);

    if (tid < num_blocks) {
        block_sums[tid] = val;
    }
}

// Phase 3: Add block sums to get final result
extern "C" __global__
void scan_add_block_sums_kernel(
    float* __restrict__ output,
    const float* __restrict__ block_sums,
    uint64_t n
) {
    if (blockIdx.x == 0) return;  // First block doesn't need adjustment

    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        output[idx] += block_sums[blockIdx.x - 1];
    }
}

// ============================================================================
// Cumulative Sum (Cumsum) - Specialized
// ============================================================================

// Simple inclusive cumsum for contiguous float arrays
extern "C" __global__
void cumsum_inclusive_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    __shared__ float shared[ELEMENTS_PER_BLOCK];

    uint64_t block_offset = (uint64_t)blockIdx.x * ELEMENTS_PER_BLOCK;
    uint32_t tid = threadIdx.x;

    // Load two elements per thread
    uint64_t idx1 = block_offset + tid;
    uint64_t idx2 = block_offset + tid + BLOCK_SIZE;

    shared[tid] = (idx1 < n) ? input[idx1] : 0.0f;
    shared[tid + BLOCK_SIZE] = (idx2 < n) ? input[idx2] : 0.0f;
    __syncthreads();

    // Blelloch up-sweep
    int offset = 1;
    for (int d = ELEMENTS_PER_BLOCK >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            shared[bi] += shared[ai];
        }
        offset <<= 1;
    }

    // Clear last element for exclusive scan
    if (tid == 0) {
        shared[ELEMENTS_PER_BLOCK - 1] = 0;
    }

    // Down-sweep
    for (int d = 1; d < ELEMENTS_PER_BLOCK; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            float t = shared[ai];
            shared[ai] = shared[bi];
            shared[bi] += t;
        }
    }
    __syncthreads();

    // Convert exclusive to inclusive and write
    if (idx1 < n) {
        output[idx1] = shared[tid] + input[idx1];
    }
    if (idx2 < n) {
        output[idx2] = shared[tid + BLOCK_SIZE] + input[idx2];
    }
}

// ============================================================================
// Cumulative Product (Cumprod)
// ============================================================================

extern "C" __global__
void cumprod_inclusive_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    __shared__ float shared[BLOCK_SIZE];

    uint64_t block_offset = (uint64_t)blockIdx.x * blockDim.x;
    uint64_t idx = block_offset + threadIdx.x;
    uint32_t tid = threadIdx.x;

    float val = (idx < n) ? input[idx] : 1.0f;
    shared[tid] = val;
    __syncthreads();

    // Parallel scan with multiplication
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        float temp = (tid >= stride) ? shared[tid - stride] : 1.0f;
        __syncthreads();
        shared[tid] *= temp;
        __syncthreads();
    }

    if (idx < n) {
        output[idx] = shared[tid];
    }
}

// ============================================================================
// Cumulative Max/Min
// ============================================================================

extern "C" __global__
void cummax_inclusive_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    __shared__ float shared[BLOCK_SIZE];

    uint64_t block_offset = (uint64_t)blockIdx.x * blockDim.x;
    uint64_t idx = block_offset + threadIdx.x;
    uint32_t tid = threadIdx.x;

    float val = (idx < n) ? input[idx] : -__FLT_MAX__;
    shared[tid] = val;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        float temp = (tid >= stride) ? shared[tid - stride] : -__FLT_MAX__;
        __syncthreads();
        shared[tid] = fmaxf(shared[tid], temp);
        __syncthreads();
    }

    if (idx < n) {
        output[idx] = shared[tid];
    }
}

extern "C" __global__
void cummin_inclusive_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    __shared__ float shared[BLOCK_SIZE];

    uint64_t block_offset = (uint64_t)blockIdx.x * blockDim.x;
    uint64_t idx = block_offset + threadIdx.x;
    uint32_t tid = threadIdx.x;

    float val = (idx < n) ? input[idx] : __FLT_MAX__;
    shared[tid] = val;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        float temp = (tid >= stride) ? shared[tid - stride] : __FLT_MAX__;
        __syncthreads();
        shared[tid] = fminf(shared[tid], temp);
        __syncthreads();
    }

    if (idx < n) {
        output[idx] = shared[tid];
    }
}

// ============================================================================
// Axis Scan Kernels
// ============================================================================

// Cumsum along last axis
extern "C" __global__
void cumsum_axis_last_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t outer_size,
    uint64_t inner_size
) {
    uint64_t outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    const float* row_in = input + outer_idx * inner_size;
    float* row_out = output + outer_idx * inner_size;

    // Simple sequential scan per row (for small inner_size)
    float sum = 0.0f;
    for (uint64_t i = 0; i < inner_size; i++) {
        sum += row_in[i];
        row_out[i] = sum;
    }
}

// Parallel cumsum along last axis (for larger inner_size)
extern "C" __global__
void cumsum_axis_last_parallel_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t outer_size,
    uint64_t inner_size
) {
    extern __shared__ float shared[];

    uint64_t outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    const float* row_in = input + outer_idx * inner_size;
    float* row_out = output + outer_idx * inner_size;
    uint32_t tid = threadIdx.x;

    // Load into shared memory
    for (uint64_t i = tid; i < inner_size; i += blockDim.x) {
        shared[i] = row_in[i];
    }
    __syncthreads();

    // Hillis-Steele parallel scan
    for (int stride = 1; stride < inner_size; stride <<= 1) {
        float temp = (tid >= stride && tid < inner_size) ? shared[tid - stride] : 0.0f;
        __syncthreads();
        if (tid < inner_size) {
            shared[tid] += temp;
        }
        __syncthreads();
    }

    // Write output
    for (uint64_t i = tid; i < inner_size; i += blockDim.x) {
        row_out[i] = shared[i];
    }
}

// ============================================================================
// Int32 Scan Kernels
// ============================================================================

extern "C" __global__
void cumsum_inclusive_int32_kernel(
    int32_t* __restrict__ output,
    const int32_t* __restrict__ input,
    uint64_t n
) {
    __shared__ int32_t shared[BLOCK_SIZE];

    uint64_t block_offset = (uint64_t)blockIdx.x * blockDim.x;
    uint64_t idx = block_offset + threadIdx.x;
    uint32_t tid = threadIdx.x;

    int32_t val = (idx < n) ? input[idx] : 0;
    shared[tid] = val;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        int32_t temp = (tid >= stride) ? shared[tid - stride] : 0;
        __syncthreads();
        shared[tid] += temp;
        __syncthreads();
    }

    if (idx < n) {
        output[idx] = shared[tid];
    }
}

// ============================================================================
// Exclusive Scan Kernels
// ============================================================================

extern "C" __global__
void cumsum_exclusive_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    __shared__ float shared[BLOCK_SIZE];

    uint64_t block_offset = (uint64_t)blockIdx.x * blockDim.x;
    uint64_t idx = block_offset + threadIdx.x;
    uint32_t tid = threadIdx.x;

    // Shift input for exclusive scan
    float val = (idx > 0 && idx < n) ? input[idx - 1] : 0.0f;
    shared[tid] = val;
    __syncthreads();

    // Inclusive scan on shifted data
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        float temp = (tid >= stride) ? shared[tid - stride] : 0.0f;
        __syncthreads();
        shared[tid] += temp;
        __syncthreads();
    }

    if (idx < n) {
        output[idx] = shared[tid];
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_cumsum_inclusive_float(
    void* output,
    const void* input,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    if (n <= BLOCK_SIZE) {
        scan_inclusive_sum_float_single_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
            (float*)output, (const float*)input, n
        );
    } else {
        // Multi-block scan
        uint64_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        if (num_blocks <= BLOCK_SIZE) {
            float* block_sums;
            cudaMalloc(&block_sums, num_blocks * sizeof(float));

            scan_inclusive_sum_float_phase1_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                (float*)output, block_sums, (const float*)input, n
            );

            scan_block_sums_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
                block_sums, num_blocks
            );

            scan_add_block_sums_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                (float*)output, block_sums, n
            );

            cudaFree(block_sums);
        } else {
            // For very large arrays, use sequential blocks
            cumsum_inclusive_float_kernel<<<
                (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK,
                BLOCK_SIZE, 0, stream
            >>>((float*)output, (const float*)input, n);
        }
    }

    return cudaGetLastError();
}

int lux_cuda_cumsum_exclusive_float(
    void* output,
    const void* input,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    if (n <= BLOCK_SIZE) {
        scan_exclusive_sum_float_single_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
            (float*)output, (const float*)input, n
        );
    } else {
        uint64_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        cumsum_exclusive_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            (float*)output, (const float*)input, n
        );
    }

    return cudaGetLastError();
}

int lux_cuda_cumprod_inclusive_float(
    void* output,
    const void* input,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cumprod_inclusive_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)output, (const float*)input, n
    );

    return cudaGetLastError();
}

int lux_cuda_cummax_inclusive_float(
    void* output,
    const void* input,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cummax_inclusive_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)output, (const float*)input, n
    );

    return cudaGetLastError();
}

int lux_cuda_cummin_inclusive_float(
    void* output,
    const void* input,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cummin_inclusive_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)output, (const float*)input, n
    );

    return cudaGetLastError();
}

int lux_cuda_cumsum_axis_float(
    void* output,
    const void* input,
    uint64_t outer_size,
    uint64_t inner_size,
    cudaStream_t stream
) {
    if (outer_size == 0 || inner_size == 0) return 0;

    if (inner_size <= 1024) {
        // Use parallel kernel for reasonable sizes
        size_t shared_size = inner_size * sizeof(float);
        uint32_t threads = min((uint32_t)inner_size, (uint32_t)BLOCK_SIZE);

        cumsum_axis_last_parallel_float_kernel<<<outer_size, threads, shared_size, stream>>>(
            (float*)output, (const float*)input, outer_size, inner_size
        );
    } else {
        // Use sequential kernel for large inner dimensions
        cumsum_axis_last_float_kernel<<<outer_size, 1, 0, stream>>>(
            (float*)output, (const float*)input, outer_size, inner_size
        );
    }

    return cudaGetLastError();
}

int lux_cuda_cumsum_inclusive_int32(
    void* output,
    const void* input,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cumsum_inclusive_int32_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (int32_t*)output, (const int32_t*)input, n
    );

    return cudaGetLastError();
}

}  // extern "C"
