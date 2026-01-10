// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Reduce CUDA Kernels
// General reduction operations (sum, prod, max, min, mean, all, any, etc.)
// Supports float, half, int32, int64, and boolean types

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
// Reduction Operation Types
// ============================================================================

enum class ReduceOp : uint32_t {
    SUM = 0,
    PROD = 1,
    MAX = 2,
    MIN = 3,
    MEAN = 4,
    ALL = 5,
    ANY = 6,
    L1_NORM = 7,
    L2_NORM = 8,
    COUNT_NONZERO = 9,
    VARIANCE = 10,
};

// ============================================================================
// Warp Reduction Helpers
// ============================================================================

__device__ __forceinline__
float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__
float warp_reduce_prod(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val *= __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__
float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__
float warp_reduce_min(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__
int32_t warp_reduce_sum_int32(int32_t val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__
bool warp_reduce_all(bool val) {
    return __all_sync(0xffffffff, val);
}

__device__ __forceinline__
bool warp_reduce_any(bool val) {
    return __any_sync(0xffffffff, val);
}

// ============================================================================
// Block Reduction Helpers
// ============================================================================

template<typename T>
__device__ __forceinline__
T block_reduce_sum(T val) {
    __shared__ T shared[BLOCK_SIZE / WARP_SIZE];

    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;

    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

__device__ __forceinline__
float block_reduce_max(float val) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    val = warp_reduce_max(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : -FLT_MAX;

    if (warp_id == 0) {
        val = warp_reduce_max(val);
    }

    return val;
}

__device__ __forceinline__
float block_reduce_min(float val) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    val = warp_reduce_min(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : FLT_MAX;

    if (warp_id == 0) {
        val = warp_reduce_min(val);
    }

    return val;
}

// ============================================================================
// Full Array Reduction Kernels - Float
// ============================================================================

extern "C" __global__
void reduce_sum_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    float sum = 0.0f;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-stride loop
    for (uint64_t i = idx; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        sum += input[i];
    }

    sum = block_reduce_sum(sum);

    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

extern "C" __global__
void reduce_prod_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    float prod = 1.0f;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (uint64_t i = idx; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        prod *= input[i];
    }

    // Warp reduction
    prod = warp_reduce_prod(prod);

    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane == 0) {
        shared[warp_id] = prod;
    }
    __syncthreads();

    if (warp_id == 0) {
        prod = (lane < blockDim.x / WARP_SIZE) ? shared[lane] : 1.0f;
        prod = warp_reduce_prod(prod);

        if (lane == 0) {
            // Use atomicCAS loop for float multiplication
            float old = *output;
            float assumed, new_val;
            do {
                assumed = old;
                new_val = assumed * prod;
                old = __int_as_float(atomicCAS((int*)output,
                    __float_as_int(assumed), __float_as_int(new_val)));
            } while (assumed != old);
        }
    }
}

extern "C" __global__
void reduce_max_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    float max_val = -FLT_MAX;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (uint64_t i = idx; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        max_val = fmaxf(max_val, input[i]);
    }

    max_val = block_reduce_max(max_val);

    if (threadIdx.x == 0) {
        // Atomic max for float
        int* output_int = (int*)output;
        int old = *output_int;
        int assumed;
        do {
            assumed = old;
            float old_val = __int_as_float(old);
            float new_val = fmaxf(old_val, max_val);
            old = atomicCAS(output_int, assumed, __float_as_int(new_val));
        } while (assumed != old);
    }
}

extern "C" __global__
void reduce_min_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    float min_val = FLT_MAX;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (uint64_t i = idx; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        min_val = fminf(min_val, input[i]);
    }

    min_val = block_reduce_min(min_val);

    if (threadIdx.x == 0) {
        int* output_int = (int*)output;
        int old = *output_int;
        int assumed;
        do {
            assumed = old;
            float old_val = __int_as_float(old);
            float new_val = fminf(old_val, min_val);
            old = atomicCAS(output_int, assumed, __float_as_int(new_val));
        } while (assumed != old);
    }
}

// ============================================================================
// Axis Reduction Kernels - Reduce along last axis
// ============================================================================

extern "C" __global__
void reduce_sum_axis_last_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t outer_size,
    uint64_t inner_size
) {
    extern __shared__ float smem[];

    uint64_t outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    const float* row = input + outer_idx * inner_size;
    uint64_t tid = threadIdx.x;

    float sum = 0.0f;
    for (uint64_t i = tid; i < inner_size; i += blockDim.x) {
        sum += row[i];
    }

    smem[tid] = sum;
    __syncthreads();

    // Block reduction
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx] = smem[0];
    }
}

extern "C" __global__
void reduce_max_axis_last_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t outer_size,
    uint64_t inner_size
) {
    extern __shared__ float smem[];

    uint64_t outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    const float* row = input + outer_idx * inner_size;
    uint64_t tid = threadIdx.x;

    float max_val = -FLT_MAX;
    for (uint64_t i = tid; i < inner_size; i += blockDim.x) {
        max_val = fmaxf(max_val, row[i]);
    }

    smem[tid] = max_val;
    __syncthreads();

    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx] = smem[0];
    }
}

extern "C" __global__
void reduce_min_axis_last_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t outer_size,
    uint64_t inner_size
) {
    extern __shared__ float smem[];

    uint64_t outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    const float* row = input + outer_idx * inner_size;
    uint64_t tid = threadIdx.x;

    float min_val = FLT_MAX;
    for (uint64_t i = tid; i < inner_size; i += blockDim.x) {
        min_val = fminf(min_val, row[i]);
    }

    smem[tid] = min_val;
    __syncthreads();

    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fminf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx] = smem[0];
    }
}

// ============================================================================
// Mean and Variance Kernels
// ============================================================================

extern "C" __global__
void reduce_mean_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    float sum = 0.0f;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (uint64_t i = idx; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        sum += input[i];
    }

    sum = block_reduce_sum(sum);

    if (threadIdx.x == 0) {
        atomicAdd(output, sum / (float)n);
    }
}

extern "C" __global__
void reduce_mean_axis_last_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t outer_size,
    uint64_t inner_size
) {
    extern __shared__ float smem[];

    uint64_t outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    const float* row = input + outer_idx * inner_size;
    uint64_t tid = threadIdx.x;

    float sum = 0.0f;
    for (uint64_t i = tid; i < inner_size; i += blockDim.x) {
        sum += row[i];
    }

    smem[tid] = sum;
    __syncthreads();

    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[outer_idx] = smem[0] / (float)inner_size;
    }
}

// Welford's online algorithm for variance
extern "C" __global__
void reduce_variance_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    float mean,
    uint64_t n
) {
    float sum_sq = 0.0f;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (uint64_t i = idx; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        float diff = input[i] - mean;
        sum_sq += diff * diff;
    }

    sum_sq = block_reduce_sum(sum_sq);

    if (threadIdx.x == 0) {
        atomicAdd(output, sum_sq / (float)n);
    }
}

// ============================================================================
// Norm Kernels
// ============================================================================

extern "C" __global__
void reduce_l1_norm_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    float sum = 0.0f;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (uint64_t i = idx; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        sum += fabsf(input[i]);
    }

    sum = block_reduce_sum(sum);

    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

extern "C" __global__
void reduce_l2_norm_sq_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    float sum = 0.0f;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (uint64_t i = idx; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        float val = input[i];
        sum += val * val;
    }

    sum = block_reduce_sum(sum);

    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

// ============================================================================
// Boolean Reductions
// ============================================================================

extern "C" __global__
void reduce_all_kernel(
    bool* __restrict__ output,
    const bool* __restrict__ input,
    uint64_t n
) {
    __shared__ bool shared[BLOCK_SIZE / WARP_SIZE];

    bool all_true = true;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (uint64_t i = idx; i < n && all_true; i += (uint64_t)gridDim.x * blockDim.x) {
        all_true = all_true && input[i];
    }

    all_true = warp_reduce_all(all_true);

    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane == 0) {
        shared[warp_id] = all_true;
    }
    __syncthreads();

    if (warp_id == 0) {
        all_true = (lane < blockDim.x / WARP_SIZE) ? shared[lane] : true;
        all_true = warp_reduce_all(all_true);

        if (lane == 0 && !all_true) {
            *output = false;
        }
    }
}

extern "C" __global__
void reduce_any_kernel(
    bool* __restrict__ output,
    const bool* __restrict__ input,
    uint64_t n
) {
    __shared__ bool shared[BLOCK_SIZE / WARP_SIZE];

    bool any_true = false;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (uint64_t i = idx; i < n && !any_true; i += (uint64_t)gridDim.x * blockDim.x) {
        any_true = any_true || input[i];
    }

    any_true = warp_reduce_any(any_true);

    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane == 0) {
        shared[warp_id] = any_true;
    }
    __syncthreads();

    if (warp_id == 0) {
        any_true = (lane < blockDim.x / WARP_SIZE) ? shared[lane] : false;
        any_true = warp_reduce_any(any_true);

        if (lane == 0 && any_true) {
            *output = true;
        }
    }
}

// ============================================================================
// Count Nonzero
// ============================================================================

extern "C" __global__
void reduce_count_nonzero_float_kernel(
    uint64_t* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    uint64_t count = 0;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (uint64_t i = idx; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        if (input[i] != 0.0f) {
            count++;
        }
    }

    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        count += __shfl_down_sync(0xffffffff, count, offset);
    }

    __shared__ uint64_t shared[BLOCK_SIZE / WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane == 0) {
        shared[warp_id] = count;
    }
    __syncthreads();

    if (warp_id == 0) {
        count = (lane < blockDim.x / WARP_SIZE) ? shared[lane] : 0;

        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            count += __shfl_down_sync(0xffffffff, count, offset);
        }

        if (lane == 0) {
            atomicAdd((unsigned long long*)output, count);
        }
    }
}

// ============================================================================
// Int32 Reductions
// ============================================================================

extern "C" __global__
void reduce_sum_int32_kernel(
    int64_t* __restrict__ output,
    const int32_t* __restrict__ input,
    uint64_t n
) {
    int64_t sum = 0;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (uint64_t i = idx; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        sum += input[i];
    }

    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ int64_t shared[BLOCK_SIZE / WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane < blockDim.x / WARP_SIZE) ? shared[lane] : 0;

        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0) {
            atomicAdd((unsigned long long*)output, sum);
        }
    }
}

extern "C" __global__
void reduce_max_int32_kernel(
    int32_t* __restrict__ output,
    const int32_t* __restrict__ input,
    uint64_t n
) {
    int32_t max_val = INT32_MIN;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    for (uint64_t i = idx; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        max_val = max(max_val, input[i]);
    }

    // Warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }

    __shared__ int32_t shared[BLOCK_SIZE / WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane == 0) {
        shared[warp_id] = max_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        max_val = (lane < blockDim.x / WARP_SIZE) ? shared[lane] : INT32_MIN;

        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        }

        if (lane == 0) {
            atomicMax(output, max_val);
        }
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_reduce_sum_float(
    void* output,
    const void* input,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    // Initialize output to 0
    cudaMemsetAsync(output, 0, sizeof(float), stream);

    uint64_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks > 1024) num_blocks = 1024;

    reduce_sum_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)output, (const float*)input, n
    );

    return cudaGetLastError();
}

int lux_cuda_reduce_prod_float(
    void* output,
    const void* input,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    float one = 1.0f;
    cudaMemcpyAsync(output, &one, sizeof(float), cudaMemcpyHostToDevice, stream);

    uint64_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks > 1024) num_blocks = 1024;

    reduce_prod_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)output, (const float*)input, n
    );

    return cudaGetLastError();
}

int lux_cuda_reduce_max_float(
    void* output,
    const void* input,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    float neg_inf = -FLT_MAX;
    cudaMemcpyAsync(output, &neg_inf, sizeof(float), cudaMemcpyHostToDevice, stream);

    uint64_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks > 1024) num_blocks = 1024;

    reduce_max_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)output, (const float*)input, n
    );

    return cudaGetLastError();
}

int lux_cuda_reduce_min_float(
    void* output,
    const void* input,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    float pos_inf = FLT_MAX;
    cudaMemcpyAsync(output, &pos_inf, sizeof(float), cudaMemcpyHostToDevice, stream);

    uint64_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks > 1024) num_blocks = 1024;

    reduce_min_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)output, (const float*)input, n
    );

    return cudaGetLastError();
}

int lux_cuda_reduce_mean_float(
    void* output,
    const void* input,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    cudaMemsetAsync(output, 0, sizeof(float), stream);

    uint64_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks > 1024) num_blocks = 1024;

    reduce_mean_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)output, (const float*)input, n
    );

    return cudaGetLastError();
}

int lux_cuda_reduce_sum_axis_float(
    void* output,
    const void* input,
    uint64_t outer_size,
    uint64_t inner_size,
    cudaStream_t stream
) {
    if (outer_size == 0 || inner_size == 0) return 0;

    uint32_t threads = BLOCK_SIZE;
    size_t shared_size = threads * sizeof(float);

    reduce_sum_axis_last_float_kernel<<<outer_size, threads, shared_size, stream>>>(
        (float*)output, (const float*)input, outer_size, inner_size
    );

    return cudaGetLastError();
}

int lux_cuda_reduce_max_axis_float(
    void* output,
    const void* input,
    uint64_t outer_size,
    uint64_t inner_size,
    cudaStream_t stream
) {
    if (outer_size == 0 || inner_size == 0) return 0;

    uint32_t threads = BLOCK_SIZE;
    size_t shared_size = threads * sizeof(float);

    reduce_max_axis_last_float_kernel<<<outer_size, threads, shared_size, stream>>>(
        (float*)output, (const float*)input, outer_size, inner_size
    );

    return cudaGetLastError();
}

int lux_cuda_reduce_l2_norm_float(
    void* output,
    const void* input,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    cudaMemsetAsync(output, 0, sizeof(float), stream);

    uint64_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks > 1024) num_blocks = 1024;

    reduce_l2_norm_sq_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)output, (const float*)input, n
    );

    // Note: Caller needs to take sqrt of output

    return cudaGetLastError();
}

int lux_cuda_reduce_all(
    void* output,
    const void* input,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    bool init_true = true;
    cudaMemcpyAsync(output, &init_true, sizeof(bool), cudaMemcpyHostToDevice, stream);

    uint64_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks > 1024) num_blocks = 1024;

    reduce_all_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (bool*)output, (const bool*)input, n
    );

    return cudaGetLastError();
}

int lux_cuda_reduce_any(
    void* output,
    const void* input,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    bool init_false = false;
    cudaMemcpyAsync(output, &init_false, sizeof(bool), cudaMemcpyHostToDevice, stream);

    uint64_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks > 1024) num_blocks = 1024;

    reduce_any_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (bool*)output, (const bool*)input, n
    );

    return cudaGetLastError();
}

int lux_cuda_reduce_count_nonzero_float(
    void* output,
    const void* input,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    cudaMemsetAsync(output, 0, sizeof(uint64_t), stream);

    uint64_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks > 1024) num_blocks = 1024;

    reduce_count_nonzero_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (uint64_t*)output, (const float*)input, n
    );

    return cudaGetLastError();
}

}  // extern "C"
