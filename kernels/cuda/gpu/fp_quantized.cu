// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// FP8/FP16/BF16 Quantization Operations - CUDA Implementation
// Provides floating-point format conversions and quantization for neural network inference.
// Supports E4M3, E5M2 FP8 formats, FP16, and BF16.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace lux {
namespace cuda {
namespace fp_quantize {

// ============================================================================
// Configuration
// ============================================================================

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

// ============================================================================
// FP8 Format Definitions
// ============================================================================

// E4M3 format: 1 sign, 4 exponent, 3 mantissa bits
// Range: ~[-448, 448], precision ~0.125
// Bias: 7

// E5M2 format: 1 sign, 5 exponent, 2 mantissa bits
// Range: ~[-57344, 57344], precision ~0.25
// Bias: 15

enum class FP8Format : uint32_t {
    E4M3 = 0,   // 4-bit exponent, 3-bit mantissa (higher precision, lower range)
    E5M2 = 1    // 5-bit exponent, 2-bit mantissa (lower precision, higher range)
};

// ============================================================================
// FP8 E4M3 Conversion Utilities
// ============================================================================

// FP8 E4M3 constants
constexpr int FP8_E4M3_BIAS = 7;
constexpr int FP8_E4M3_MAX_EXP = 15;
constexpr float FP8_E4M3_MAX = 448.0f;
constexpr float FP8_E4M3_MIN = -448.0f;

__device__ __forceinline__
uint8_t float_to_fp8_e4m3(float val) {
    // Handle special cases
    if (isnan(val)) return 0x7F;  // NaN
    if (val > FP8_E4M3_MAX) val = FP8_E4M3_MAX;
    if (val < FP8_E4M3_MIN) val = FP8_E4M3_MIN;

    // Extract IEEE FP32 components
    uint32_t bits = __float_as_uint(val);
    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127;  // Unbias FP32
    uint32_t frac = bits & 0x7FFFFF;

    // Handle zero
    if (exp == -127 && frac == 0) {
        return (uint8_t)(sign << 7);
    }

    // Re-bias for E4M3 (bias = 7)
    int32_t new_exp = exp + FP8_E4M3_BIAS;

    // Handle underflow
    if (new_exp <= 0) {
        return (uint8_t)(sign << 7);  // Flush to zero
    }

    // Handle overflow
    if (new_exp >= FP8_E4M3_MAX_EXP) {
        new_exp = FP8_E4M3_MAX_EXP;
        frac = 0x700000;  // Max mantissa
    }

    // Round mantissa to 3 bits
    uint32_t mant = (frac + 0x80000) >> 20;  // Round to nearest
    if (mant >= 8) {
        mant = 0;
        new_exp++;
        if (new_exp >= FP8_E4M3_MAX_EXP) {
            new_exp = FP8_E4M3_MAX_EXP;
            mant = 7;
        }
    }

    return (uint8_t)((sign << 7) | (new_exp << 3) | mant);
}

__device__ __forceinline__
float fp8_e4m3_to_float(uint8_t val) {
    uint32_t sign = (val >> 7) & 0x1;
    uint32_t exp = (val >> 3) & 0xF;
    uint32_t mant = val & 0x7;

    // Handle zero
    if (exp == 0 && mant == 0) {
        return sign ? -0.0f : 0.0f;
    }

    // Handle NaN (E4M3 uses 0x7F for NaN)
    if (val == 0x7F || val == 0xFF) {
        return nanf("");
    }

    // Convert to FP32
    int32_t new_exp = exp - FP8_E4M3_BIAS + 127;
    uint32_t new_mant = mant << 20;

    uint32_t bits = (sign << 31) | (new_exp << 23) | new_mant;
    return __uint_as_float(bits);
}

// ============================================================================
// FP8 E5M2 Conversion Utilities
// ============================================================================

// FP8 E5M2 constants
constexpr int FP8_E5M2_BIAS = 15;
constexpr int FP8_E5M2_MAX_EXP = 31;
constexpr float FP8_E5M2_MAX = 57344.0f;
constexpr float FP8_E5M2_MIN = -57344.0f;

__device__ __forceinline__
uint8_t float_to_fp8_e5m2(float val) {
    // Handle special cases
    if (isnan(val)) return 0x7F;
    if (isinf(val)) return val > 0 ? 0x7C : 0xFC;
    if (val > FP8_E5M2_MAX) val = FP8_E5M2_MAX;
    if (val < FP8_E5M2_MIN) val = FP8_E5M2_MIN;

    uint32_t bits = __float_as_uint(val);
    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127;
    uint32_t frac = bits & 0x7FFFFF;

    // Handle zero
    if (exp == -127 && frac == 0) {
        return (uint8_t)(sign << 7);
    }

    // Re-bias for E5M2 (bias = 15)
    int32_t new_exp = exp + FP8_E5M2_BIAS;

    // Handle underflow
    if (new_exp <= 0) {
        return (uint8_t)(sign << 7);
    }

    // Handle overflow
    if (new_exp >= FP8_E5M2_MAX_EXP) {
        return (uint8_t)((sign << 7) | 0x7C);  // Infinity
    }

    // Round mantissa to 2 bits
    uint32_t mant = (frac + 0x100000) >> 21;
    if (mant >= 4) {
        mant = 0;
        new_exp++;
        if (new_exp >= FP8_E5M2_MAX_EXP) {
            return (uint8_t)((sign << 7) | 0x7C);
        }
    }

    return (uint8_t)((sign << 7) | (new_exp << 2) | mant);
}

__device__ __forceinline__
float fp8_e5m2_to_float(uint8_t val) {
    uint32_t sign = (val >> 7) & 0x1;
    uint32_t exp = (val >> 2) & 0x1F;
    uint32_t mant = val & 0x3;

    // Handle zero
    if (exp == 0 && mant == 0) {
        return sign ? -0.0f : 0.0f;
    }

    // Handle infinity
    if (exp == 31 && mant == 0) {
        return sign ? -INFINITY : INFINITY;
    }

    // Handle NaN
    if (exp == 31 && mant != 0) {
        return nanf("");
    }

    int32_t new_exp = exp - FP8_E5M2_BIAS + 127;
    uint32_t new_mant = mant << 21;

    uint32_t bits = (sign << 31) | (new_exp << 23) | new_mant;
    return __uint_as_float(bits);
}

// ============================================================================
// FP32 <-> FP8 Conversion Kernels
// ============================================================================

extern "C" __global__
void convert_f32_to_fp8_e4m3_kernel(
    uint8_t* __restrict__ output,
    const float* __restrict__ input,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    output[idx] = float_to_fp8_e4m3(input[idx]);
}

extern "C" __global__
void convert_fp8_e4m3_to_f32_kernel(
    float* __restrict__ output,
    const uint8_t* __restrict__ input,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    output[idx] = fp8_e4m3_to_float(input[idx]);
}

extern "C" __global__
void convert_f32_to_fp8_e5m2_kernel(
    uint8_t* __restrict__ output,
    const float* __restrict__ input,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    output[idx] = float_to_fp8_e5m2(input[idx]);
}

extern "C" __global__
void convert_fp8_e5m2_to_f32_kernel(
    float* __restrict__ output,
    const uint8_t* __restrict__ input,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    output[idx] = fp8_e5m2_to_float(input[idx]);
}

// ============================================================================
// FP16 <-> FP8 Conversion Kernels
// ============================================================================

extern "C" __global__
void convert_f16_to_fp8_e4m3_kernel(
    uint8_t* __restrict__ output,
    const __half* __restrict__ input,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    float val = __half2float(input[idx]);
    output[idx] = float_to_fp8_e4m3(val);
}

extern "C" __global__
void convert_fp8_e4m3_to_f16_kernel(
    __half* __restrict__ output,
    const uint8_t* __restrict__ input,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    float val = fp8_e4m3_to_float(input[idx]);
    output[idx] = __float2half(val);
}

extern "C" __global__
void convert_f16_to_fp8_e5m2_kernel(
    uint8_t* __restrict__ output,
    const __half* __restrict__ input,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    float val = __half2float(input[idx]);
    output[idx] = float_to_fp8_e5m2(val);
}

extern "C" __global__
void convert_fp8_e5m2_to_f16_kernel(
    __half* __restrict__ output,
    const uint8_t* __restrict__ input,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    float val = fp8_e5m2_to_float(input[idx]);
    output[idx] = __float2half(val);
}

// ============================================================================
// BF16 <-> FP8 Conversion Kernels
// ============================================================================

extern "C" __global__
void convert_bf16_to_fp8_e4m3_kernel(
    uint8_t* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    float val = __bfloat162float(input[idx]);
    output[idx] = float_to_fp8_e4m3(val);
}

extern "C" __global__
void convert_fp8_e4m3_to_bf16_kernel(
    __nv_bfloat16* __restrict__ output,
    const uint8_t* __restrict__ input,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    float val = fp8_e4m3_to_float(input[idx]);
    output[idx] = __float2bfloat16(val);
}

// ============================================================================
// FP32 <-> FP16 Conversion Kernels
// ============================================================================

extern "C" __global__
void convert_f32_to_f16_kernel(
    __half* __restrict__ output,
    const float* __restrict__ input,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    output[idx] = __float2half(input[idx]);
}

extern "C" __global__
void convert_f16_to_f32_kernel(
    float* __restrict__ output,
    const __half* __restrict__ input,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    output[idx] = __half2float(input[idx]);
}

// ============================================================================
// FP32 <-> BF16 Conversion Kernels
// ============================================================================

extern "C" __global__
void convert_f32_to_bf16_kernel(
    __nv_bfloat16* __restrict__ output,
    const float* __restrict__ input,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    output[idx] = __float2bfloat16(input[idx]);
}

extern "C" __global__
void convert_bf16_to_f32_kernel(
    float* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    output[idx] = __bfloat162float(input[idx]);
}

// ============================================================================
// FP16 <-> BF16 Conversion Kernels
// ============================================================================

extern "C" __global__
void convert_f16_to_bf16_kernel(
    __nv_bfloat16* __restrict__ output,
    const __half* __restrict__ input,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    float val = __half2float(input[idx]);
    output[idx] = __float2bfloat16(val);
}

extern "C" __global__
void convert_bf16_to_f16_kernel(
    __half* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    float val = __bfloat162float(input[idx]);
    output[idx] = __float2half(val);
}

// ============================================================================
// Scaled FP8 Conversion (with scaling factor)
// ============================================================================

extern "C" __global__
void convert_f32_to_fp8_e4m3_scaled_kernel(
    uint8_t* __restrict__ output,
    const float* __restrict__ input,
    float scale,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    float val = input[idx] * scale;
    output[idx] = float_to_fp8_e4m3(val);
}

extern "C" __global__
void convert_fp8_e4m3_to_f32_scaled_kernel(
    float* __restrict__ output,
    const uint8_t* __restrict__ input,
    float scale,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    float val = fp8_e4m3_to_float(input[idx]);
    output[idx] = val * scale;
}

// ============================================================================
// Vectorized FP8 Conversion (4 elements at a time)
// ============================================================================

extern "C" __global__
void convert_f32_to_fp8_e4m3_vectorized_kernel(
    uint8_t* __restrict__ output,
    const float* __restrict__ input,
    uint32_t n
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

    uint8_t out[4];
    out[0] = float_to_fp8_e4m3(in.x);
    out[1] = float_to_fp8_e4m3(in.y);
    out[2] = float_to_fp8_e4m3(in.z);
    out[3] = float_to_fp8_e4m3(in.w);

    if (idx + 3 < n) {
        *((uint32_t*)(output + idx)) = *((uint32_t*)out);
    } else {
        output[idx] = out[0];
        if (idx + 1 < n) output[idx + 1] = out[1];
        if (idx + 2 < n) output[idx + 2] = out[2];
    }
}

// ============================================================================
// FP8 GEMM Support (FP8 x FP8 -> FP16/FP32 accumulation)
// ============================================================================

// FP8 E4M3 GEMM with FP32 accumulation
extern "C" __global__
void fp8_gemm_e4m3_kernel(
    float* __restrict__ C,
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    float scale_a,
    float scale_b,
    uint32_t M, uint32_t N, uint32_t K
) {
    const int TILE = 16;

    __shared__ uint8_t As[TILE][TILE];
    __shared__ uint8_t Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

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
            float a = fp8_e4m3_to_float(As[threadIdx.y][k]) * scale_a;
            float b = fp8_e4m3_to_float(Bs[k][threadIdx.x]) * scale_b;
            acc += a * b;
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// FP8 E4M3 GEMM with FP16 output
extern "C" __global__
void fp8_gemm_e4m3_f16_kernel(
    __half* __restrict__ C,
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    float scale_a,
    float scale_b,
    uint32_t M, uint32_t N, uint32_t K
) {
    const int TILE = 16;

    __shared__ uint8_t As[TILE][TILE];
    __shared__ uint8_t Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

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
            float a = fp8_e4m3_to_float(As[threadIdx.y][k]) * scale_a;
            float b = fp8_e4m3_to_float(Bs[k][threadIdx.x]) * scale_b;
            acc += a * b;
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = __float2half(acc);
    }
}

// ============================================================================
// Mixed Precision GEMM (FP8 weights x FP16 activations)
// ============================================================================

extern "C" __global__
void fp8_f16_gemm_kernel(
    __half* __restrict__ C,
    const uint8_t* __restrict__ A,    // FP8 E4M3 weights [M, K]
    const __half* __restrict__ B,     // FP16 activations [K, N]
    float weight_scale,
    uint32_t M, uint32_t N, uint32_t K
) {
    const int TILE = 16;

    __shared__ uint8_t As[TILE][TILE];
    __shared__ __half Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

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
            Bs[threadIdx.y][threadIdx.x] = __float2half(0.0f);
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            float a = fp8_e4m3_to_float(As[threadIdx.y][k]) * weight_scale;
            float b = __half2float(Bs[k][threadIdx.x]);
            acc += a * b;
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = __float2half(acc);
    }
}

// ============================================================================
// Calibration for FP8 (find optimal scale)
// ============================================================================

extern "C" __global__
void compute_fp8_scale_kernel(
    float* __restrict__ scale_out,
    const float* __restrict__ input,
    uint32_t n,
    float fp8_max  // 448.0f for E4M3, 57344.0f for E5M2
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
        float scale = fp8_max / fmaxf(absmax, 1e-10f);
        atomicMax((int*)scale_out, __float_as_int(scale));
    }
}

// ============================================================================
// Host API (C Interface)
// ============================================================================

} // namespace fp_quantize
} // namespace cuda
} // namespace lux

extern "C" {

using namespace lux::cuda::fp_quantize;

// -------------------- FP32 <-> FP8 --------------------

int lux_cuda_convert_f32_to_fp8(
    void* output,
    const void* input,
    uint32_t n,
    uint32_t format,  // 0 = E4M3, 1 = E5M2
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    if (format == 0) {
        convert_f32_to_fp8_e4m3_kernel<<<grid, block, 0, stream>>>(
            (uint8_t*)output, (const float*)input, n
        );
    } else {
        convert_f32_to_fp8_e5m2_kernel<<<grid, block, 0, stream>>>(
            (uint8_t*)output, (const float*)input, n
        );
    }

    return cudaGetLastError();
}

int lux_cuda_convert_fp8_to_f32(
    void* output,
    const void* input,
    uint32_t n,
    uint32_t format,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    if (format == 0) {
        convert_fp8_e4m3_to_f32_kernel<<<grid, block, 0, stream>>>(
            (float*)output, (const uint8_t*)input, n
        );
    } else {
        convert_fp8_e5m2_to_f32_kernel<<<grid, block, 0, stream>>>(
            (float*)output, (const uint8_t*)input, n
        );
    }

    return cudaGetLastError();
}

// -------------------- FP16 <-> FP8 --------------------

int lux_cuda_convert_f16_to_fp8(
    void* output,
    const void* input,
    uint32_t n,
    uint32_t format,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    if (format == 0) {
        convert_f16_to_fp8_e4m3_kernel<<<grid, block, 0, stream>>>(
            (uint8_t*)output, (const __half*)input, n
        );
    } else {
        convert_f16_to_fp8_e5m2_kernel<<<grid, block, 0, stream>>>(
            (uint8_t*)output, (const __half*)input, n
        );
    }

    return cudaGetLastError();
}

int lux_cuda_convert_fp8_to_f16(
    void* output,
    const void* input,
    uint32_t n,
    uint32_t format,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    if (format == 0) {
        convert_fp8_e4m3_to_f16_kernel<<<grid, block, 0, stream>>>(
            (__half*)output, (const uint8_t*)input, n
        );
    } else {
        convert_fp8_e5m2_to_f16_kernel<<<grid, block, 0, stream>>>(
            (__half*)output, (const uint8_t*)input, n
        );
    }

    return cudaGetLastError();
}

// -------------------- BF16 <-> FP8 --------------------

int lux_cuda_convert_bf16_to_fp8(
    void* output,
    const void* input,
    uint32_t n,
    uint32_t format,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    convert_bf16_to_fp8_e4m3_kernel<<<grid, block, 0, stream>>>(
        (uint8_t*)output, (const __nv_bfloat16*)input, n
    );

    return cudaGetLastError();
}

int lux_cuda_convert_fp8_to_bf16(
    void* output,
    const void* input,
    uint32_t n,
    uint32_t format,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    convert_fp8_e4m3_to_bf16_kernel<<<grid, block, 0, stream>>>(
        (__nv_bfloat16*)output, (const uint8_t*)input, n
    );

    return cudaGetLastError();
}

// -------------------- FP32 <-> FP16/BF16 --------------------

int lux_cuda_convert_f32_to_f16(
    void* output,
    const void* input,
    uint32_t n,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    convert_f32_to_f16_kernel<<<grid, block, 0, stream>>>(
        (__half*)output, (const float*)input, n
    );

    return cudaGetLastError();
}

int lux_cuda_convert_f16_to_f32(
    void* output,
    const void* input,
    uint32_t n,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    convert_f16_to_f32_kernel<<<grid, block, 0, stream>>>(
        (float*)output, (const __half*)input, n
    );

    return cudaGetLastError();
}

int lux_cuda_convert_f32_to_bf16(
    void* output,
    const void* input,
    uint32_t n,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    convert_f32_to_bf16_kernel<<<grid, block, 0, stream>>>(
        (__nv_bfloat16*)output, (const float*)input, n
    );

    return cudaGetLastError();
}

int lux_cuda_convert_bf16_to_f32(
    void* output,
    const void* input,
    uint32_t n,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    convert_bf16_to_f32_kernel<<<grid, block, 0, stream>>>(
        (float*)output, (const __nv_bfloat16*)input, n
    );

    return cudaGetLastError();
}

// -------------------- Scaled FP8 --------------------

int lux_cuda_convert_f32_to_fp8_scaled(
    void* output,
    const void* input,
    float scale,
    uint32_t n,
    uint32_t format,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    convert_f32_to_fp8_e4m3_scaled_kernel<<<grid, block, 0, stream>>>(
        (uint8_t*)output, (const float*)input, scale, n
    );

    return cudaGetLastError();
}

int lux_cuda_convert_fp8_to_f32_scaled(
    void* output,
    const void* input,
    float scale,
    uint32_t n,
    uint32_t format,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    convert_fp8_e4m3_to_f32_scaled_kernel<<<grid, block, 0, stream>>>(
        (float*)output, (const uint8_t*)input, scale, n
    );

    return cudaGetLastError();
}

// -------------------- FP8 GEMM --------------------

int lux_cuda_fp8_gemm(
    void* C,
    const void* A,
    const void* B,
    float scale_a,
    float scale_b,
    uint32_t M, uint32_t N, uint32_t K,
    uint32_t output_type,  // 0 = FP32, 1 = FP16
    cudaStream_t stream
) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    if (output_type == 0) {
        fp8_gemm_e4m3_kernel<<<blocks, threads, 0, stream>>>(
            (float*)C, (const uint8_t*)A, (const uint8_t*)B,
            scale_a, scale_b, M, N, K
        );
    } else {
        fp8_gemm_e4m3_f16_kernel<<<blocks, threads, 0, stream>>>(
            (__half*)C, (const uint8_t*)A, (const uint8_t*)B,
            scale_a, scale_b, M, N, K
        );
    }

    return cudaGetLastError();
}

int lux_cuda_fp8_f16_gemm(
    void* C,
    const void* A,
    const void* B,
    float weight_scale,
    uint32_t M, uint32_t N, uint32_t K,
    cudaStream_t stream
) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    fp8_f16_gemm_kernel<<<blocks, threads, 0, stream>>>(
        (__half*)C, (const uint8_t*)A, (const __half*)B,
        weight_scale, M, N, K
    );

    return cudaGetLastError();
}

// -------------------- Calibration --------------------

int lux_cuda_compute_fp8_scale(
    float* scale_out,
    const void* input,
    uint32_t n,
    uint32_t format,
    cudaStream_t stream
) {
    float init = 0.0f;
    cudaMemcpyAsync(scale_out, &init, sizeof(float), cudaMemcpyHostToDevice, stream);

    dim3 block(BLOCK_SIZE);
    dim3 grid(min(256u, (n + BLOCK_SIZE - 1) / BLOCK_SIZE));
    size_t shmem = BLOCK_SIZE * sizeof(float);

    float fp8_max = (format == 0) ? FP8_E4M3_MAX : FP8_E5M2_MAX;

    compute_fp8_scale_kernel<<<grid, block, shmem, stream>>>(
        scale_out, (const float*)input, n, fp8_max
    );

    return cudaGetLastError();
}

} // extern "C"
