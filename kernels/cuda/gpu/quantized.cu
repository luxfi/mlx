// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// INT8/INT4 Quantization Operations - CUDA Implementation
// Provides symmetric and asymmetric quantization with per-tensor and per-channel scales.
// Optimized for neural network inference quantization workflows.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace lux {
namespace cuda {
namespace quantize {

// ============================================================================
// Configuration
// ============================================================================

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

// ============================================================================
// Quantization Mode Enums
// ============================================================================

enum class QuantMode : uint32_t {
    SYMMETRIC = 0,      // Zero-point is always 0
    ASYMMETRIC = 1      // Zero-point can be non-zero
};

enum class QuantGranularity : uint32_t {
    PER_TENSOR = 0,     // Single scale/zp for entire tensor
    PER_CHANNEL = 1,    // Scale/zp per output channel
    PER_GROUP = 2       // Scale/zp per group of elements
};

// ============================================================================
// Quantization Parameters
// ============================================================================

struct QuantParams {
    float scale;
    int32_t zero_point;
};

struct QuantParamsPerChannel {
    const float* scales;       // [num_channels]
    const int32_t* zero_points; // [num_channels]
    uint32_t num_channels;
    uint32_t channel_stride;   // Stride to next channel element
};

struct QuantParamsPerGroup {
    const float* scales;       // [num_groups]
    const int32_t* zero_points; // [num_groups]
    uint32_t group_size;
    uint32_t num_groups;
};

// ============================================================================
// INT8 Quantization Utilities
// ============================================================================

__device__ __forceinline__
int8_t quantize_int8_symmetric(float val, float scale) {
    int32_t q = __float2int_rn(val / scale);
    return (int8_t)max(-127, min(127, q));
}

__device__ __forceinline__
int8_t quantize_int8_asymmetric(float val, float scale, int32_t zero_point) {
    int32_t q = __float2int_rn(val / scale) + zero_point;
    return (int8_t)max(-128, min(127, q));
}

__device__ __forceinline__
float dequantize_int8_symmetric(int8_t val, float scale) {
    return (float)val * scale;
}

__device__ __forceinline__
float dequantize_int8_asymmetric(int8_t val, float scale, int32_t zero_point) {
    return ((float)val - (float)zero_point) * scale;
}

// ============================================================================
// UINT8 Quantization Utilities (for unsigned activations)
// ============================================================================

__device__ __forceinline__
uint8_t quantize_uint8(float val, float scale, int32_t zero_point) {
    int32_t q = __float2int_rn(val / scale) + zero_point;
    return (uint8_t)max(0, min(255, q));
}

__device__ __forceinline__
float dequantize_uint8(uint8_t val, float scale, int32_t zero_point) {
    return ((float)val - (float)zero_point) * scale;
}

// ============================================================================
// INT4 Quantization Utilities (packed 2 per byte)
// ============================================================================

__device__ __forceinline__
int8_t quantize_int4_symmetric(float val, float scale) {
    int32_t q = __float2int_rn(val / scale);
    return (int8_t)max(-7, min(7, q));
}

__device__ __forceinline__
int8_t quantize_int4_asymmetric(float val, float scale, int32_t zero_point) {
    int32_t q = __float2int_rn(val / scale) + zero_point;
    return (int8_t)max(-8, min(7, q));
}

__device__ __forceinline__
uint8_t pack_int4(int8_t low, int8_t high) {
    return ((uint8_t)(low & 0xF)) | ((uint8_t)(high & 0xF) << 4);
}

__device__ __forceinline__
void unpack_int4(uint8_t packed, int8_t* low, int8_t* high) {
    *low = (int8_t)(packed & 0xF);
    *high = (int8_t)((packed >> 4) & 0xF);
    // Sign extend from 4 bits
    if (*low >= 8) *low -= 16;
    if (*high >= 8) *high -= 16;
}

__device__ __forceinline__
float dequantize_int4_symmetric(int8_t val, float scale) {
    return (float)val * scale;
}

__device__ __forceinline__
float dequantize_int4_asymmetric(int8_t val, float scale, int32_t zero_point) {
    return ((float)val - (float)zero_point) * scale;
}

// ============================================================================
// Per-Tensor Quantization Kernels
// ============================================================================

// Quantize FP32 -> INT8 (symmetric, per-tensor)
extern "C" __global__
void quantize_f32_to_int8_symmetric_kernel(
    int8_t* __restrict__ output,
    const float* __restrict__ input,
    float scale,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    output[idx] = quantize_int8_symmetric(input[idx], scale);
}

// Quantize FP32 -> INT8 (asymmetric, per-tensor)
extern "C" __global__
void quantize_f32_to_int8_asymmetric_kernel(
    int8_t* __restrict__ output,
    const float* __restrict__ input,
    float scale,
    int32_t zero_point,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    output[idx] = quantize_int8_asymmetric(input[idx], scale, zero_point);
}

// Dequantize INT8 -> FP32 (symmetric, per-tensor)
extern "C" __global__
void dequantize_int8_to_f32_symmetric_kernel(
    float* __restrict__ output,
    const int8_t* __restrict__ input,
    float scale,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    output[idx] = dequantize_int8_symmetric(input[idx], scale);
}

// Dequantize INT8 -> FP32 (asymmetric, per-tensor)
extern "C" __global__
void dequantize_int8_to_f32_asymmetric_kernel(
    float* __restrict__ output,
    const int8_t* __restrict__ input,
    float scale,
    int32_t zero_point,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    output[idx] = dequantize_int8_asymmetric(input[idx], scale, zero_point);
}

// ============================================================================
// Per-Channel Quantization Kernels
// ============================================================================

// Quantize FP32 -> INT8 per-channel (row-major: [channels, elements_per_channel])
extern "C" __global__
void quantize_f32_to_int8_per_channel_kernel(
    int8_t* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ scales,
    const int32_t* __restrict__ zero_points,
    uint32_t num_channels,
    uint32_t elements_per_channel,
    bool symmetric
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = num_channels * elements_per_channel;

    if (idx >= total) return;

    uint32_t channel = idx / elements_per_channel;
    float scale = scales[channel];

    if (symmetric) {
        output[idx] = quantize_int8_symmetric(input[idx], scale);
    } else {
        int32_t zp = zero_points ? zero_points[channel] : 0;
        output[idx] = quantize_int8_asymmetric(input[idx], scale, zp);
    }
}

// Dequantize INT8 -> FP32 per-channel
extern "C" __global__
void dequantize_int8_to_f32_per_channel_kernel(
    float* __restrict__ output,
    const int8_t* __restrict__ input,
    const float* __restrict__ scales,
    const int32_t* __restrict__ zero_points,
    uint32_t num_channels,
    uint32_t elements_per_channel,
    bool symmetric
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = num_channels * elements_per_channel;

    if (idx >= total) return;

    uint32_t channel = idx / elements_per_channel;
    float scale = scales[channel];

    if (symmetric) {
        output[idx] = dequantize_int8_symmetric(input[idx], scale);
    } else {
        int32_t zp = zero_points ? zero_points[channel] : 0;
        output[idx] = dequantize_int8_asymmetric(input[idx], scale, zp);
    }
}

// ============================================================================
// Per-Group Quantization Kernels (for weight-only quantization)
// ============================================================================

// Quantize FP32 -> INT8 per-group
extern "C" __global__
void quantize_f32_to_int8_per_group_kernel(
    int8_t* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ scales,
    const int32_t* __restrict__ zero_points,
    uint32_t n,
    uint32_t group_size,
    bool symmetric
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    uint32_t group = idx / group_size;
    float scale = scales[group];

    if (symmetric) {
        output[idx] = quantize_int8_symmetric(input[idx], scale);
    } else {
        int32_t zp = zero_points ? zero_points[group] : 0;
        output[idx] = quantize_int8_asymmetric(input[idx], scale, zp);
    }
}

// ============================================================================
// INT4 Quantization Kernels
// ============================================================================

// Quantize FP32 -> INT4 packed (symmetric, per-tensor)
extern "C" __global__
void quantize_f32_to_int4_symmetric_kernel(
    uint8_t* __restrict__ output,   // Packed INT4 (2 per byte)
    const float* __restrict__ input,
    float scale,
    uint32_t n                       // Number of elements (must be even)
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t packed_idx = idx;
    uint32_t elem_idx = idx * 2;

    if (elem_idx >= n) return;

    int8_t low = quantize_int4_symmetric(input[elem_idx], scale);
    int8_t high = (elem_idx + 1 < n) ?
                  quantize_int4_symmetric(input[elem_idx + 1], scale) : 0;

    output[packed_idx] = pack_int4(low, high);
}

// Dequantize INT4 packed -> FP32 (symmetric, per-tensor)
extern "C" __global__
void dequantize_int4_to_f32_symmetric_kernel(
    float* __restrict__ output,
    const uint8_t* __restrict__ input,  // Packed INT4
    float scale,
    uint32_t n                           // Number of output elements
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t elem_idx = idx * 2;

    if (elem_idx >= n) return;

    int8_t low, high;
    unpack_int4(input[idx], &low, &high);

    output[elem_idx] = dequantize_int4_symmetric(low, scale);
    if (elem_idx + 1 < n) {
        output[elem_idx + 1] = dequantize_int4_symmetric(high, scale);
    }
}

// Quantize FP32 -> INT4 per-group (for LLM weight compression)
extern "C" __global__
void quantize_f32_to_int4_per_group_kernel(
    uint8_t* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ scales,
    uint32_t n,
    uint32_t group_size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t elem_idx = idx * 2;

    if (elem_idx >= n) return;

    uint32_t group_low = elem_idx / group_size;
    uint32_t group_high = (elem_idx + 1) / group_size;

    float scale_low = scales[group_low];
    float scale_high = scales[group_high];

    int8_t low = quantize_int4_symmetric(input[elem_idx], scale_low);
    int8_t high = (elem_idx + 1 < n) ?
                  quantize_int4_symmetric(input[elem_idx + 1], scale_high) : 0;

    output[idx] = pack_int4(low, high);
}

// ============================================================================
// FP16 to INT8/INT4 Quantization (common for inference)
// ============================================================================

// Quantize FP16 -> INT8 (symmetric, per-tensor)
extern "C" __global__
void quantize_f16_to_int8_symmetric_kernel(
    int8_t* __restrict__ output,
    const __half* __restrict__ input,
    float scale,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    float val = __half2float(input[idx]);
    output[idx] = quantize_int8_symmetric(val, scale);
}

// Dequantize INT8 -> FP16 (symmetric, per-tensor)
extern "C" __global__
void dequantize_int8_to_f16_symmetric_kernel(
    __half* __restrict__ output,
    const int8_t* __restrict__ input,
    float scale,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    float val = dequantize_int8_symmetric(input[idx], scale);
    output[idx] = __float2half(val);
}

// ============================================================================
// Calibration Kernels (for finding quantization parameters)
// ============================================================================

// Find min/max values for calibration
extern "C" __global__
void find_minmax_kernel(
    float* __restrict__ min_out,
    float* __restrict__ max_out,
    const float* __restrict__ input,
    uint32_t n
) {
    extern __shared__ float sdata[];
    float* smin = sdata;
    float* smax = sdata + blockDim.x;

    uint32_t tid = threadIdx.x;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize with extreme values
    float local_min = 1e10f;
    float local_max = -1e10f;

    // Grid-stride loop
    for (uint32_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        float val = input[i];
        local_min = fminf(local_min, val);
        local_max = fmaxf(local_max, val);
    }

    smin[tid] = local_min;
    smax[tid] = local_max;
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smin[tid] = fminf(smin[tid], smin[tid + s]);
            smax[tid] = fmaxf(smax[tid], smax[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMin((int*)min_out, __float_as_int(smin[0]));
        atomicMax((int*)max_out, __float_as_int(smax[0]));
    }
}

// Compute symmetric quantization scale from absmax
extern "C" __global__
void compute_scale_absmax_kernel(
    float* __restrict__ scale_out,
    const float* __restrict__ input,
    uint32_t n,
    float qmax  // e.g., 127 for INT8, 7 for INT4
) {
    extern __shared__ float sdata[];

    uint32_t tid = threadIdx.x;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    float local_absmax = 0.0f;

    for (uint32_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        local_absmax = fmaxf(local_absmax, fabsf(input[i]));
    }

    sdata[tid] = local_absmax;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        float absmax = sdata[0];
        float scale = absmax / qmax;
        // Ensure scale is not zero
        if (scale < 1e-10f) scale = 1e-10f;

        // Use atomicMax to find global absmax across blocks
        atomicMax((int*)scale_out, __float_as_int(scale));
    }
}

// ============================================================================
// Fused Quantized MatMul Support
// ============================================================================

// Quantized matrix multiplication accumulator structure
struct QuantMatMulAccum {
    int32_t* accumulator;  // INT32 accumulator buffer
    float scale_a;
    float scale_b;
    int32_t zp_a;
    int32_t zp_b;
};

// INT8 GEMM with INT32 accumulation (A: int8, B: int8, C: int32)
extern "C" __global__
void quantized_gemm_int8_kernel(
    int32_t* __restrict__ C,
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    uint32_t M, uint32_t N, uint32_t K
) {
    const int TILE = 16;

    __shared__ int8_t As[TILE][TILE];
    __shared__ int8_t Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    int32_t acc = 0;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // Load A tile
        int a_col = t * TILE + threadIdx.x;
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0;
        }

        // Load B tile
        int b_row = t * TILE + threadIdx.y;
        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            acc += (int32_t)As[threadIdx.y][k] * (int32_t)Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// Fused: INT8 GEMM -> Dequantize -> FP32 output
extern "C" __global__
void quantized_gemm_int8_dequant_kernel(
    float* __restrict__ C,
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    float scale_a,
    float scale_b,
    uint32_t M, uint32_t N, uint32_t K
) {
    const int TILE = 16;

    __shared__ int8_t As[TILE][TILE];
    __shared__ int8_t Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    int32_t acc = 0;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0;
        }

        int b_row = t * TILE + threadIdx.y;
        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            acc += (int32_t)As[threadIdx.y][k] * (int32_t)Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        // Dequantize: output = acc * scale_a * scale_b
        C[row * N + col] = (float)acc * scale_a * scale_b;
    }
}

// INT4 x INT8 GEMM for weight-only quantization (common in LLMs)
extern "C" __global__
void quantized_gemm_int4_int8_kernel(
    int32_t* __restrict__ C,
    const uint8_t* __restrict__ A,  // INT4 packed weights [M, K/2]
    const int8_t* __restrict__ B,   // INT8 activations [K, N]
    uint32_t M, uint32_t N, uint32_t K
) {
    const int TILE = 16;

    __shared__ int8_t As[TILE][TILE];
    __shared__ int8_t Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    int32_t acc = 0;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // Load and unpack A tile (INT4 -> INT8)
        int a_col = t * TILE + threadIdx.x;
        if (row < M && a_col < K) {
            int packed_idx = row * (K / 2) + (a_col / 2);
            int8_t low, high;
            unpack_int4(A[packed_idx], &low, &high);
            As[threadIdx.y][threadIdx.x] = (a_col % 2 == 0) ? low : high;
        } else {
            As[threadIdx.y][threadIdx.x] = 0;
        }

        // Load B tile
        int b_row = t * TILE + threadIdx.y;
        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            acc += (int32_t)As[threadIdx.y][k] * (int32_t)Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ============================================================================
// Vectorized Quantization (4 elements at a time)
// ============================================================================

extern "C" __global__
void quantize_f32_to_int8_vectorized_kernel(
    int8_t* __restrict__ output,
    const float* __restrict__ input,
    float scale,
    int32_t zero_point,
    uint32_t n,
    bool symmetric
) {
    uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx >= n) return;

    float4 in;
    if (idx + 3 < n) {
        in = *((float4*)(input + idx));
    } else {
        in.x = input[idx];
        in.y = (idx + 1 < n) ? input[idx + 1] : 0.0f;
        in.z = (idx + 2 < n) ? input[idx + 2] : 0.0f;
        in.w = 0.0f;
    }

    int8_t out[4];
    if (symmetric) {
        out[0] = quantize_int8_symmetric(in.x, scale);
        out[1] = quantize_int8_symmetric(in.y, scale);
        out[2] = quantize_int8_symmetric(in.z, scale);
        out[3] = quantize_int8_symmetric(in.w, scale);
    } else {
        out[0] = quantize_int8_asymmetric(in.x, scale, zero_point);
        out[1] = quantize_int8_asymmetric(in.y, scale, zero_point);
        out[2] = quantize_int8_asymmetric(in.z, scale, zero_point);
        out[3] = quantize_int8_asymmetric(in.w, scale, zero_point);
    }

    if (idx + 3 < n) {
        *((int32_t*)(output + idx)) = *((int32_t*)out);
    } else {
        output[idx] = out[0];
        if (idx + 1 < n) output[idx + 1] = out[1];
        if (idx + 2 < n) output[idx + 2] = out[2];
    }
}

// ============================================================================
// Host API (C Interface)
// ============================================================================

} // namespace quantize
} // namespace cuda
} // namespace lux

extern "C" {

using namespace lux::cuda::quantize;

// -------------------- INT8 Quantization --------------------

int lux_cuda_quantize_f32_to_int8(
    void* output,
    const void* input,
    float scale,
    int32_t zero_point,
    uint32_t n,
    bool symmetric,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    if (symmetric) {
        quantize_f32_to_int8_symmetric_kernel<<<grid, block, 0, stream>>>(
            (int8_t*)output, (const float*)input, scale, n
        );
    } else {
        quantize_f32_to_int8_asymmetric_kernel<<<grid, block, 0, stream>>>(
            (int8_t*)output, (const float*)input, scale, zero_point, n
        );
    }

    return cudaGetLastError();
}

int lux_cuda_dequantize_int8_to_f32(
    void* output,
    const void* input,
    float scale,
    int32_t zero_point,
    uint32_t n,
    bool symmetric,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    if (symmetric) {
        dequantize_int8_to_f32_symmetric_kernel<<<grid, block, 0, stream>>>(
            (float*)output, (const int8_t*)input, scale, n
        );
    } else {
        dequantize_int8_to_f32_asymmetric_kernel<<<grid, block, 0, stream>>>(
            (float*)output, (const int8_t*)input, scale, zero_point, n
        );
    }

    return cudaGetLastError();
}

// -------------------- Per-Channel Quantization --------------------

int lux_cuda_quantize_f32_to_int8_per_channel(
    void* output,
    const void* input,
    const float* scales,
    const int32_t* zero_points,
    uint32_t num_channels,
    uint32_t elements_per_channel,
    bool symmetric,
    cudaStream_t stream
) {
    uint32_t total = num_channels * elements_per_channel;
    dim3 block(BLOCK_SIZE);
    dim3 grid((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

    quantize_f32_to_int8_per_channel_kernel<<<grid, block, 0, stream>>>(
        (int8_t*)output, (const float*)input,
        scales, zero_points,
        num_channels, elements_per_channel, symmetric
    );

    return cudaGetLastError();
}

int lux_cuda_dequantize_int8_to_f32_per_channel(
    void* output,
    const void* input,
    const float* scales,
    const int32_t* zero_points,
    uint32_t num_channels,
    uint32_t elements_per_channel,
    bool symmetric,
    cudaStream_t stream
) {
    uint32_t total = num_channels * elements_per_channel;
    dim3 block(BLOCK_SIZE);
    dim3 grid((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

    dequantize_int8_to_f32_per_channel_kernel<<<grid, block, 0, stream>>>(
        (float*)output, (const int8_t*)input,
        scales, zero_points,
        num_channels, elements_per_channel, symmetric
    );

    return cudaGetLastError();
}

// -------------------- INT4 Quantization --------------------

int lux_cuda_quantize_f32_to_int4(
    void* output,
    const void* input,
    float scale,
    uint32_t n,
    cudaStream_t stream
) {
    uint32_t packed_n = (n + 1) / 2;
    dim3 block(BLOCK_SIZE);
    dim3 grid((packed_n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    quantize_f32_to_int4_symmetric_kernel<<<grid, block, 0, stream>>>(
        (uint8_t*)output, (const float*)input, scale, n
    );

    return cudaGetLastError();
}

int lux_cuda_dequantize_int4_to_f32(
    void* output,
    const void* input,
    float scale,
    uint32_t n,
    cudaStream_t stream
) {
    uint32_t packed_n = (n + 1) / 2;
    dim3 block(BLOCK_SIZE);
    dim3 grid((packed_n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    dequantize_int4_to_f32_symmetric_kernel<<<grid, block, 0, stream>>>(
        (float*)output, (const uint8_t*)input, scale, n
    );

    return cudaGetLastError();
}

int lux_cuda_quantize_f32_to_int4_per_group(
    void* output,
    const void* input,
    const float* scales,
    uint32_t n,
    uint32_t group_size,
    cudaStream_t stream
) {
    uint32_t packed_n = (n + 1) / 2;
    dim3 block(BLOCK_SIZE);
    dim3 grid((packed_n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    quantize_f32_to_int4_per_group_kernel<<<grid, block, 0, stream>>>(
        (uint8_t*)output, (const float*)input, scales, n, group_size
    );

    return cudaGetLastError();
}

// -------------------- FP16 <-> INT8 --------------------

int lux_cuda_quantize_f16_to_int8(
    void* output,
    const void* input,
    float scale,
    uint32_t n,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    quantize_f16_to_int8_symmetric_kernel<<<grid, block, 0, stream>>>(
        (int8_t*)output, (const __half*)input, scale, n
    );

    return cudaGetLastError();
}

int lux_cuda_dequantize_int8_to_f16(
    void* output,
    const void* input,
    float scale,
    uint32_t n,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    dequantize_int8_to_f16_symmetric_kernel<<<grid, block, 0, stream>>>(
        (__half*)output, (const int8_t*)input, scale, n
    );

    return cudaGetLastError();
}

// -------------------- Calibration --------------------

int lux_cuda_find_minmax(
    float* min_out,
    float* max_out,
    const void* input,
    uint32_t n,
    cudaStream_t stream
) {
    // Initialize with extreme values
    float init_min = 1e10f;
    float init_max = -1e10f;
    cudaMemcpyAsync(min_out, &init_min, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(max_out, &init_max, sizeof(float), cudaMemcpyHostToDevice, stream);

    dim3 block(BLOCK_SIZE);
    dim3 grid(min(256u, (n + BLOCK_SIZE - 1) / BLOCK_SIZE));
    size_t shmem = 2 * BLOCK_SIZE * sizeof(float);

    find_minmax_kernel<<<grid, block, shmem, stream>>>(
        min_out, max_out, (const float*)input, n
    );

    return cudaGetLastError();
}

int lux_cuda_compute_scale_absmax(
    float* scale_out,
    const void* input,
    uint32_t n,
    float qmax,
    cudaStream_t stream
) {
    float init = 0.0f;
    cudaMemcpyAsync(scale_out, &init, sizeof(float), cudaMemcpyHostToDevice, stream);

    dim3 block(BLOCK_SIZE);
    dim3 grid(min(256u, (n + BLOCK_SIZE - 1) / BLOCK_SIZE));
    size_t shmem = BLOCK_SIZE * sizeof(float);

    compute_scale_absmax_kernel<<<grid, block, shmem, stream>>>(
        scale_out, (const float*)input, n, qmax
    );

    return cudaGetLastError();
}

// -------------------- Quantized GEMM --------------------

int lux_cuda_quantized_gemm_int8(
    void* C,
    const void* A,
    const void* B,
    uint32_t M, uint32_t N, uint32_t K,
    cudaStream_t stream
) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    quantized_gemm_int8_kernel<<<blocks, threads, 0, stream>>>(
        (int32_t*)C, (const int8_t*)A, (const int8_t*)B, M, N, K
    );

    return cudaGetLastError();
}

int lux_cuda_quantized_gemm_int8_dequant(
    void* C,
    const void* A,
    const void* B,
    float scale_a,
    float scale_b,
    uint32_t M, uint32_t N, uint32_t K,
    cudaStream_t stream
) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    quantized_gemm_int8_dequant_kernel<<<blocks, threads, 0, stream>>>(
        (float*)C, (const int8_t*)A, (const int8_t*)B,
        scale_a, scale_b, M, N, K
    );

    return cudaGetLastError();
}

int lux_cuda_quantized_gemm_int4_int8(
    void* C,
    const void* A,
    const void* B,
    uint32_t M, uint32_t N, uint32_t K,
    cudaStream_t stream
) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    quantized_gemm_int4_int8_kernel<<<blocks, threads, 0, stream>>>(
        (int32_t*)C, (const uint8_t*)A, (const int8_t*)B, M, N, K
    );

    return cudaGetLastError();
}

// -------------------- Vectorized Quantization --------------------

int lux_cuda_quantize_f32_to_int8_vectorized(
    void* output,
    const void* input,
    float scale,
    int32_t zero_point,
    uint32_t n,
    bool symmetric,
    cudaStream_t stream
) {
    uint32_t n_vec = (n + 3) / 4;
    dim3 block(BLOCK_SIZE);
    dim3 grid((n_vec + BLOCK_SIZE - 1) / BLOCK_SIZE);

    quantize_f32_to_int8_vectorized_kernel<<<grid, block, 0, stream>>>(
        (int8_t*)output, (const float*)input, scale, zero_point, n, symmetric
    );

    return cudaGetLastError();
}

} // extern "C"
