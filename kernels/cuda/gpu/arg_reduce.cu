// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Arg Reduce CUDA Kernels
// Argmin/argmax operations returning indices of min/max values
// Supports float, half, int32, and int64 types

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cfloat>

// ============================================================================
// Configuration
// ============================================================================

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// ============================================================================
// Helper Structures
// ============================================================================

template<typename T>
struct IndexValue {
    T value;
    uint64_t index;
};

// ============================================================================
// Warp Reduction Helpers
// ============================================================================

template<typename T>
__device__ __forceinline__
IndexValue<T> warp_argmax(IndexValue<T> val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        T other_val = __shfl_down_sync(0xffffffff, val.value, offset);
        uint64_t other_idx = __shfl_down_sync(0xffffffff, val.index, offset);

        if (other_val > val.value || (other_val == val.value && other_idx < val.index)) {
            val.value = other_val;
            val.index = other_idx;
        }
    }
    return val;
}

template<typename T>
__device__ __forceinline__
IndexValue<T> warp_argmin(IndexValue<T> val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        T other_val = __shfl_down_sync(0xffffffff, val.value, offset);
        uint64_t other_idx = __shfl_down_sync(0xffffffff, val.index, offset);

        if (other_val < val.value || (other_val == val.value && other_idx < val.index)) {
            val.value = other_val;
            val.index = other_idx;
        }
    }
    return val;
}

// ============================================================================
// Argmax Kernels - Float32
// ============================================================================

extern "C" __global__
void argmax_float_kernel(
    const float* __restrict__ input,
    uint64_t* __restrict__ output,
    uint64_t n
) {
    __shared__ float shared_vals[BLOCK_SIZE / WARP_SIZE];
    __shared__ uint64_t shared_idxs[BLOCK_SIZE / WARP_SIZE];

    uint64_t tid = threadIdx.x;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + tid;

    // Initialize with first valid value or -inf
    float my_val = -FLT_MAX;
    uint64_t my_idx = 0;

    // Grid-stride loop to handle large arrays
    for (uint64_t i = idx; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        float val = input[i];
        if (val > my_val || (val == my_val && i < my_idx)) {
            my_val = val;
            my_idx = i;
        }
    }

    // Warp reduction
    IndexValue<float> result = {my_val, my_idx};
    result = warp_argmax(result);

    // Write warp results to shared memory
    uint32_t lane = tid % WARP_SIZE;
    uint32_t warp_id = tid / WARP_SIZE;

    if (lane == 0) {
        shared_vals[warp_id] = result.value;
        shared_idxs[warp_id] = result.index;
    }
    __syncthreads();

    // Final reduction in first warp
    uint32_t num_warps = blockDim.x / WARP_SIZE;
    if (warp_id == 0) {
        result.value = (lane < num_warps) ? shared_vals[lane] : -FLT_MAX;
        result.index = (lane < num_warps) ? shared_idxs[lane] : 0;
        result = warp_argmax(result);

        if (lane == 0) {
            output[blockIdx.x] = result.index;
        }
    }
}

// Single-block argmax for final reduction
extern "C" __global__
void argmax_float_final_kernel(
    const float* __restrict__ input,
    const uint64_t* __restrict__ indices,
    uint64_t* __restrict__ output,
    uint64_t n
) {
    __shared__ float shared_vals[BLOCK_SIZE / WARP_SIZE];
    __shared__ uint64_t shared_idxs[BLOCK_SIZE / WARP_SIZE];

    uint64_t tid = threadIdx.x;

    float my_val = -FLT_MAX;
    uint64_t my_idx = 0;

    for (uint64_t i = tid; i < n; i += blockDim.x) {
        uint64_t original_idx = indices[i];
        float val = input[original_idx];
        if (val > my_val || (val == my_val && original_idx < my_idx)) {
            my_val = val;
            my_idx = original_idx;
        }
    }

    IndexValue<float> result = {my_val, my_idx};
    result = warp_argmax(result);

    uint32_t lane = tid % WARP_SIZE;
    uint32_t warp_id = tid / WARP_SIZE;

    if (lane == 0) {
        shared_vals[warp_id] = result.value;
        shared_idxs[warp_id] = result.index;
    }
    __syncthreads();

    uint32_t num_warps = blockDim.x / WARP_SIZE;
    if (warp_id == 0) {
        result.value = (lane < num_warps) ? shared_vals[lane] : -FLT_MAX;
        result.index = (lane < num_warps) ? shared_idxs[lane] : 0;
        result = warp_argmax(result);

        if (lane == 0) {
            output[0] = result.index;
        }
    }
}

// ============================================================================
// Argmin Kernels - Float32
// ============================================================================

extern "C" __global__
void argmin_float_kernel(
    const float* __restrict__ input,
    uint64_t* __restrict__ output,
    uint64_t n
) {
    __shared__ float shared_vals[BLOCK_SIZE / WARP_SIZE];
    __shared__ uint64_t shared_idxs[BLOCK_SIZE / WARP_SIZE];

    uint64_t tid = threadIdx.x;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + tid;

    float my_val = FLT_MAX;
    uint64_t my_idx = 0;

    for (uint64_t i = idx; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        float val = input[i];
        if (val < my_val || (val == my_val && i < my_idx)) {
            my_val = val;
            my_idx = i;
        }
    }

    IndexValue<float> result = {my_val, my_idx};
    result = warp_argmin(result);

    uint32_t lane = tid % WARP_SIZE;
    uint32_t warp_id = tid / WARP_SIZE;

    if (lane == 0) {
        shared_vals[warp_id] = result.value;
        shared_idxs[warp_id] = result.index;
    }
    __syncthreads();

    uint32_t num_warps = blockDim.x / WARP_SIZE;
    if (warp_id == 0) {
        result.value = (lane < num_warps) ? shared_vals[lane] : FLT_MAX;
        result.index = (lane < num_warps) ? shared_idxs[lane] : 0;
        result = warp_argmin(result);

        if (lane == 0) {
            output[blockIdx.x] = result.index;
        }
    }
}

extern "C" __global__
void argmin_float_final_kernel(
    const float* __restrict__ input,
    const uint64_t* __restrict__ indices,
    uint64_t* __restrict__ output,
    uint64_t n
) {
    __shared__ float shared_vals[BLOCK_SIZE / WARP_SIZE];
    __shared__ uint64_t shared_idxs[BLOCK_SIZE / WARP_SIZE];

    uint64_t tid = threadIdx.x;

    float my_val = FLT_MAX;
    uint64_t my_idx = 0;

    for (uint64_t i = tid; i < n; i += blockDim.x) {
        uint64_t original_idx = indices[i];
        float val = input[original_idx];
        if (val < my_val || (val == my_val && original_idx < my_idx)) {
            my_val = val;
            my_idx = original_idx;
        }
    }

    IndexValue<float> result = {my_val, my_idx};
    result = warp_argmin(result);

    uint32_t lane = tid % WARP_SIZE;
    uint32_t warp_id = tid / WARP_SIZE;

    if (lane == 0) {
        shared_vals[warp_id] = result.value;
        shared_idxs[warp_id] = result.index;
    }
    __syncthreads();

    uint32_t num_warps = blockDim.x / WARP_SIZE;
    if (warp_id == 0) {
        result.value = (lane < num_warps) ? shared_vals[lane] : FLT_MAX;
        result.index = (lane < num_warps) ? shared_idxs[lane] : 0;
        result = warp_argmin(result);

        if (lane == 0) {
            output[0] = result.index;
        }
    }
}

// ============================================================================
// Argmax/Argmin Along Axis - Float32
// ============================================================================

// Argmax along last axis (inner reduction)
extern "C" __global__
void argmax_axis_last_float_kernel(
    const float* __restrict__ input,
    uint64_t* __restrict__ output,
    uint64_t outer_size,
    uint64_t inner_size
) {
    extern __shared__ float smem[];
    float* shared_vals = smem;
    uint64_t* shared_idxs = (uint64_t*)(smem + blockDim.x);

    uint64_t outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    const float* row = input + outer_idx * inner_size;
    uint64_t tid = threadIdx.x;

    float my_val = -FLT_MAX;
    uint64_t my_idx = 0;

    for (uint64_t i = tid; i < inner_size; i += blockDim.x) {
        float val = row[i];
        if (val > my_val || (val == my_val && i < my_idx)) {
            my_val = val;
            my_idx = i;
        }
    }

    shared_vals[tid] = my_val;
    shared_idxs[tid] = my_idx;
    __syncthreads();

    // Block reduction
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float other_val = shared_vals[tid + s];
            uint64_t other_idx = shared_idxs[tid + s];
            if (other_val > shared_vals[tid] ||
                (other_val == shared_vals[tid] && other_idx < shared_idxs[tid])) {
                shared_vals[tid] = other_val;
                shared_idxs[tid] = other_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx] = shared_idxs[0];
    }
}

// Argmin along last axis
extern "C" __global__
void argmin_axis_last_float_kernel(
    const float* __restrict__ input,
    uint64_t* __restrict__ output,
    uint64_t outer_size,
    uint64_t inner_size
) {
    extern __shared__ float smem[];
    float* shared_vals = smem;
    uint64_t* shared_idxs = (uint64_t*)(smem + blockDim.x);

    uint64_t outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    const float* row = input + outer_idx * inner_size;
    uint64_t tid = threadIdx.x;

    float my_val = FLT_MAX;
    uint64_t my_idx = 0;

    for (uint64_t i = tid; i < inner_size; i += blockDim.x) {
        float val = row[i];
        if (val < my_val || (val == my_val && i < my_idx)) {
            my_val = val;
            my_idx = i;
        }
    }

    shared_vals[tid] = my_val;
    shared_idxs[tid] = my_idx;
    __syncthreads();

    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float other_val = shared_vals[tid + s];
            uint64_t other_idx = shared_idxs[tid + s];
            if (other_val < shared_vals[tid] ||
                (other_val == shared_vals[tid] && other_idx < shared_idxs[tid])) {
                shared_vals[tid] = other_val;
                shared_idxs[tid] = other_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx] = shared_idxs[0];
    }
}

// ============================================================================
// Int32 Kernels
// ============================================================================

extern "C" __global__
void argmax_int32_kernel(
    const int32_t* __restrict__ input,
    uint64_t* __restrict__ output,
    uint64_t n
) {
    __shared__ int32_t shared_vals[BLOCK_SIZE / WARP_SIZE];
    __shared__ uint64_t shared_idxs[BLOCK_SIZE / WARP_SIZE];

    uint64_t tid = threadIdx.x;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + tid;

    int32_t my_val = INT32_MIN;
    uint64_t my_idx = 0;

    for (uint64_t i = idx; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        int32_t val = input[i];
        if (val > my_val || (val == my_val && i < my_idx)) {
            my_val = val;
            my_idx = i;
        }
    }

    IndexValue<int32_t> result = {my_val, my_idx};
    result = warp_argmax(result);

    uint32_t lane = tid % WARP_SIZE;
    uint32_t warp_id = tid / WARP_SIZE;

    if (lane == 0) {
        shared_vals[warp_id] = result.value;
        shared_idxs[warp_id] = result.index;
    }
    __syncthreads();

    uint32_t num_warps = blockDim.x / WARP_SIZE;
    if (warp_id == 0) {
        result.value = (lane < num_warps) ? shared_vals[lane] : INT32_MIN;
        result.index = (lane < num_warps) ? shared_idxs[lane] : 0;
        result = warp_argmax(result);

        if (lane == 0) {
            output[blockIdx.x] = result.index;
        }
    }
}

extern "C" __global__
void argmin_int32_kernel(
    const int32_t* __restrict__ input,
    uint64_t* __restrict__ output,
    uint64_t n
) {
    __shared__ int32_t shared_vals[BLOCK_SIZE / WARP_SIZE];
    __shared__ uint64_t shared_idxs[BLOCK_SIZE / WARP_SIZE];

    uint64_t tid = threadIdx.x;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + tid;

    int32_t my_val = INT32_MAX;
    uint64_t my_idx = 0;

    for (uint64_t i = idx; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        int32_t val = input[i];
        if (val < my_val || (val == my_val && i < my_idx)) {
            my_val = val;
            my_idx = i;
        }
    }

    IndexValue<int32_t> result = {my_val, my_idx};
    result = warp_argmin(result);

    uint32_t lane = tid % WARP_SIZE;
    uint32_t warp_id = tid / WARP_SIZE;

    if (lane == 0) {
        shared_vals[warp_id] = result.value;
        shared_idxs[warp_id] = result.index;
    }
    __syncthreads();

    uint32_t num_warps = blockDim.x / WARP_SIZE;
    if (warp_id == 0) {
        result.value = (lane < num_warps) ? shared_vals[lane] : INT32_MAX;
        result.index = (lane < num_warps) ? shared_idxs[lane] : 0;
        result = warp_argmin(result);

        if (lane == 0) {
            output[blockIdx.x] = result.index;
        }
    }
}

// ============================================================================
// TopK Helper Kernels
// ============================================================================

// Partial argmax for topk - returns k largest indices
extern "C" __global__
void partial_argmax_float_kernel(
    const float* __restrict__ input,
    float* __restrict__ output_vals,
    uint64_t* __restrict__ output_idxs,
    uint64_t n,
    uint32_t k
) {
    extern __shared__ float shared[];
    float* s_vals = shared;
    uint64_t* s_idxs = (uint64_t*)(shared + k);

    uint64_t tid = threadIdx.x;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + tid;

    // Initialize shared memory with identity
    if (tid < k) {
        s_vals[tid] = -FLT_MAX;
        s_idxs[tid] = 0;
    }
    __syncthreads();

    // Each thread processes elements and maintains local top-k
    if (idx < n) {
        float val = input[idx];

        // Simple insertion into sorted top-k (small k assumed)
        for (uint32_t i = 0; i < k; i++) {
            if (val > s_vals[i]) {
                // Shift down and insert
                for (uint32_t j = k - 1; j > i; j--) {
                    atomicExch((float*)&s_vals[j], s_vals[j-1]);
                    atomicExch((unsigned long long*)&s_idxs[j], s_idxs[j-1]);
                }
                atomicExch((float*)&s_vals[i], val);
                atomicExch((unsigned long long*)&s_idxs[i], idx);
                break;
            }
        }
    }
    __syncthreads();

    // Write results
    if (tid < k) {
        output_vals[blockIdx.x * k + tid] = s_vals[tid];
        output_idxs[blockIdx.x * k + tid] = s_idxs[tid];
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_argmax_float(
    const void* input,
    void* output,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    // For small arrays, use single block
    if (n <= BLOCK_SIZE * 32) {
        argmax_float_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
            (const float*)input, (uint64_t*)output, n
        );
    } else {
        // Multi-block reduction
        uint64_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (num_blocks > 1024) num_blocks = 1024;

        // Allocate temporary buffer for block results
        uint64_t* temp;
        cudaMalloc(&temp, num_blocks * sizeof(uint64_t));

        argmax_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            (const float*)input, temp, n
        );

        argmax_float_final_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
            (const float*)input, temp, (uint64_t*)output, num_blocks
        );

        cudaFree(temp);
    }

    return cudaGetLastError();
}

int lux_cuda_argmin_float(
    const void* input,
    void* output,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    if (n <= BLOCK_SIZE * 32) {
        argmin_float_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
            (const float*)input, (uint64_t*)output, n
        );
    } else {
        uint64_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (num_blocks > 1024) num_blocks = 1024;

        uint64_t* temp;
        cudaMalloc(&temp, num_blocks * sizeof(uint64_t));

        argmin_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            (const float*)input, temp, n
        );

        argmin_float_final_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
            (const float*)input, temp, (uint64_t*)output, num_blocks
        );

        cudaFree(temp);
    }

    return cudaGetLastError();
}

int lux_cuda_argmax_axis_float(
    const void* input,
    void* output,
    uint64_t outer_size,
    uint64_t inner_size,
    cudaStream_t stream
) {
    if (outer_size == 0 || inner_size == 0) return 0;

    uint32_t threads = BLOCK_SIZE;
    size_t shared_size = threads * (sizeof(float) + sizeof(uint64_t));

    argmax_axis_last_float_kernel<<<outer_size, threads, shared_size, stream>>>(
        (const float*)input, (uint64_t*)output, outer_size, inner_size
    );

    return cudaGetLastError();
}

int lux_cuda_argmin_axis_float(
    const void* input,
    void* output,
    uint64_t outer_size,
    uint64_t inner_size,
    cudaStream_t stream
) {
    if (outer_size == 0 || inner_size == 0) return 0;

    uint32_t threads = BLOCK_SIZE;
    size_t shared_size = threads * (sizeof(float) + sizeof(uint64_t));

    argmin_axis_last_float_kernel<<<outer_size, threads, shared_size, stream>>>(
        (const float*)input, (uint64_t*)output, outer_size, inner_size
    );

    return cudaGetLastError();
}

int lux_cuda_argmax_int32(
    const void* input,
    void* output,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    argmax_int32_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        (const int32_t*)input, (uint64_t*)output, n
    );

    return cudaGetLastError();
}

int lux_cuda_argmin_int32(
    const void* input,
    void* output,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    argmin_int32_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        (const int32_t*)input, (uint64_t*)output, n
    );

    return cudaGetLastError();
}

}  // extern "C"
