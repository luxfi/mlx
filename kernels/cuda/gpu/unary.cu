// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Unary CUDA Kernels
// Element-wise unary operations (exp, log, sin, cos, sqrt, abs, neg, etc.)
// Supports float, half, int32, and boolean types

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

// ============================================================================
// Configuration
// ============================================================================

#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4

// ============================================================================
// Unary Operation Types
// ============================================================================

enum class UnaryOp : uint32_t {
    // Basic
    NEG = 0,
    ABS = 1,
    SIGN = 2,
    // Exponential/Log
    EXP = 10,
    EXP2 = 11,
    EXPM1 = 12,
    LOG = 13,
    LOG2 = 14,
    LOG10 = 15,
    LOG1P = 16,
    // Power
    SQRT = 20,
    RSQRT = 21,
    CBRT = 22,
    SQUARE = 23,
    RECIPROCAL = 24,
    // Trigonometric
    SIN = 30,
    COS = 31,
    TAN = 32,
    ASIN = 33,
    ACOS = 34,
    ATAN = 35,
    // Hyperbolic
    SINH = 40,
    COSH = 41,
    TANH = 42,
    ASINH = 43,
    ACOSH = 44,
    ATANH = 45,
    // Rounding
    FLOOR = 50,
    CEIL = 51,
    ROUND = 52,
    TRUNC = 53,
    // Special
    ERF = 60,
    ERFC = 61,
    LGAMMA = 62,
    TGAMMA = 63,
    DIGAMMA = 64,
    // Neural Network
    SIGMOID = 70,
    RELU = 71,
    GELU = 72,
    SILU = 73,   // SiLU / Swish
    SOFTPLUS = 74,
    MISH = 75,
    // Logical
    NOT = 80,
    ISNAN = 81,
    ISINF = 82,
    ISFINITE = 83,
};

// ============================================================================
// Unary Operation Implementations - Float
// ============================================================================

__device__ __forceinline__
float unary_op_float(float x, UnaryOp op) {
    switch (op) {
        // Basic
        case UnaryOp::NEG: return -x;
        case UnaryOp::ABS: return fabsf(x);
        case UnaryOp::SIGN: return (x > 0.0f) - (x < 0.0f);
        // Exponential/Log
        case UnaryOp::EXP: return expf(x);
        case UnaryOp::EXP2: return exp2f(x);
        case UnaryOp::EXPM1: return expm1f(x);
        case UnaryOp::LOG: return logf(x);
        case UnaryOp::LOG2: return log2f(x);
        case UnaryOp::LOG10: return log10f(x);
        case UnaryOp::LOG1P: return log1pf(x);
        // Power
        case UnaryOp::SQRT: return sqrtf(x);
        case UnaryOp::RSQRT: return rsqrtf(x);
        case UnaryOp::CBRT: return cbrtf(x);
        case UnaryOp::SQUARE: return x * x;
        case UnaryOp::RECIPROCAL: return 1.0f / x;
        // Trigonometric
        case UnaryOp::SIN: return sinf(x);
        case UnaryOp::COS: return cosf(x);
        case UnaryOp::TAN: return tanf(x);
        case UnaryOp::ASIN: return asinf(x);
        case UnaryOp::ACOS: return acosf(x);
        case UnaryOp::ATAN: return atanf(x);
        // Hyperbolic
        case UnaryOp::SINH: return sinhf(x);
        case UnaryOp::COSH: return coshf(x);
        case UnaryOp::TANH: return tanhf(x);
        case UnaryOp::ASINH: return asinhf(x);
        case UnaryOp::ACOSH: return acoshf(x);
        case UnaryOp::ATANH: return atanhf(x);
        // Rounding
        case UnaryOp::FLOOR: return floorf(x);
        case UnaryOp::CEIL: return ceilf(x);
        case UnaryOp::ROUND: return roundf(x);
        case UnaryOp::TRUNC: return truncf(x);
        // Special
        case UnaryOp::ERF: return erff(x);
        case UnaryOp::ERFC: return erfcf(x);
        case UnaryOp::LGAMMA: return lgammaf(x);
        case UnaryOp::TGAMMA: return tgammaf(x);
        // Neural Network
        case UnaryOp::SIGMOID: return 1.0f / (1.0f + expf(-x));
        case UnaryOp::RELU: return fmaxf(0.0f, x);
        case UnaryOp::GELU: {
            // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
            return x * 0.5f * (1.0f + erff(x * 0.7071067811865476f));
        }
        case UnaryOp::SILU: return x / (1.0f + expf(-x));  // x * sigmoid(x)
        case UnaryOp::SOFTPLUS: return log1pf(expf(x));  // log(1 + exp(x))
        case UnaryOp::MISH: return x * tanhf(log1pf(expf(x)));  // x * tanh(softplus(x))
        default: return x;
    }
}

// ============================================================================
// General Unary Kernel - Float
// ============================================================================

extern "C" __global__
void unary_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n,
    uint32_t op
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            output[elem_idx] = unary_op_float(input[elem_idx], (UnaryOp)op);
        }
    }
}

// ============================================================================
// Specialized Fast Kernels - Float
// ============================================================================

extern "C" __global__
void neg_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            output[elem_idx] = -input[elem_idx];
        }
    }
}

extern "C" __global__
void abs_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            output[elem_idx] = fabsf(input[elem_idx]);
        }
    }
}

extern "C" __global__
void exp_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            output[elem_idx] = expf(input[elem_idx]);
        }
    }
}

extern "C" __global__
void log_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            output[elem_idx] = logf(input[elem_idx]);
        }
    }
}

extern "C" __global__
void sqrt_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            output[elem_idx] = sqrtf(input[elem_idx]);
        }
    }
}

extern "C" __global__
void rsqrt_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            output[elem_idx] = rsqrtf(input[elem_idx]);
        }
    }
}

extern "C" __global__
void sin_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            output[elem_idx] = sinf(input[elem_idx]);
        }
    }
}

extern "C" __global__
void cos_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            output[elem_idx] = cosf(input[elem_idx]);
        }
    }
}

extern "C" __global__
void tanh_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            output[elem_idx] = tanhf(input[elem_idx]);
        }
    }
}

// ============================================================================
// Neural Network Activations - Float
// ============================================================================

extern "C" __global__
void sigmoid_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float x = input[elem_idx];
            output[elem_idx] = 1.0f / (1.0f + expf(-x));
        }
    }
}

extern "C" __global__
void relu_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            output[elem_idx] = fmaxf(0.0f, input[elem_idx]);
        }
    }
}

extern "C" __global__
void leaky_relu_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    float negative_slope,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float x = input[elem_idx];
            output[elem_idx] = (x > 0.0f) ? x : (negative_slope * x);
        }
    }
}

extern "C" __global__
void gelu_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    const float sqrt2_inv = 0.7071067811865476f;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float x = input[elem_idx];
            output[elem_idx] = x * 0.5f * (1.0f + erff(x * sqrt2_inv));
        }
    }
}

// Fast GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
extern "C" __global__
void gelu_fast_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float x = input[elem_idx];
            float x3 = x * x * x;
            output[elem_idx] = 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + coeff * x3)));
        }
    }
}

extern "C" __global__
void silu_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float x = input[elem_idx];
            output[elem_idx] = x / (1.0f + expf(-x));
        }
    }
}

extern "C" __global__
void softplus_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    float beta,
    float threshold,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float x = input[elem_idx];
            float bx = beta * x;
            // Use linear approximation for large values to avoid overflow
            if (bx > threshold) {
                output[elem_idx] = x;
            } else {
                output[elem_idx] = log1pf(expf(bx)) / beta;
            }
        }
    }
}

extern "C" __global__
void elu_float_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    float alpha,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float x = input[elem_idx];
            output[elem_idx] = (x > 0.0f) ? x : (alpha * (expf(x) - 1.0f));
        }
    }
}

// ============================================================================
// Unary Kernels - Half
// ============================================================================

extern "C" __global__
void unary_half_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    uint64_t n,
    uint32_t op
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float x = __half2float(input[elem_idx]);
            output[elem_idx] = __float2half(unary_op_float(x, (UnaryOp)op));
        }
    }
}

extern "C" __global__
void relu_half_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float x = __half2float(input[elem_idx]);
            output[elem_idx] = __float2half(fmaxf(0.0f, x));
        }
    }
}

extern "C" __global__
void gelu_half_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    const float sqrt2_inv = 0.7071067811865476f;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float x = __half2float(input[elem_idx]);
            float result = x * 0.5f * (1.0f + erff(x * sqrt2_inv));
            output[elem_idx] = __float2half(result);
        }
    }
}

// ============================================================================
// Unary Kernels - Int32
// ============================================================================

extern "C" __global__
void neg_int32_kernel(
    int32_t* __restrict__ output,
    const int32_t* __restrict__ input,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            output[elem_idx] = -input[elem_idx];
        }
    }
}

extern "C" __global__
void abs_int32_kernel(
    int32_t* __restrict__ output,
    const int32_t* __restrict__ input,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            int32_t x = input[elem_idx];
            output[elem_idx] = (x >= 0) ? x : -x;
        }
    }
}

// ============================================================================
// Logical Unary - Boolean
// ============================================================================

extern "C" __global__
void not_kernel(
    bool* __restrict__ output,
    const bool* __restrict__ input,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            output[elem_idx] = !input[elem_idx];
        }
    }
}

// ============================================================================
// Predicate Kernels - Float to Bool
// ============================================================================

extern "C" __global__
void isnan_float_kernel(
    bool* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            output[elem_idx] = isnan(input[elem_idx]);
        }
    }
}

extern "C" __global__
void isinf_float_kernel(
    bool* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            output[elem_idx] = isinf(input[elem_idx]);
        }
    }
}

extern "C" __global__
void isfinite_float_kernel(
    bool* __restrict__ output,
    const float* __restrict__ input,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            output[elem_idx] = isfinite(input[elem_idx]);
        }
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_unary_float(
    void* output,
    const void* input,
    uint64_t n,
    uint32_t op,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    unary_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)output, (const float*)input, n, op
    );

    return cudaGetLastError();
}

int lux_cuda_unary_half(
    void* output,
    const void* input,
    uint64_t n,
    uint32_t op,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    unary_half_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (__half*)output, (const __half*)input, n, op
    );

    return cudaGetLastError();
}

// Specialized fast path functions
int lux_cuda_neg_float(void* output, const void* input, uint64_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;
    neg_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>((float*)output, (const float*)input, n);
    return cudaGetLastError();
}

int lux_cuda_abs_float(void* output, const void* input, uint64_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;
    abs_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>((float*)output, (const float*)input, n);
    return cudaGetLastError();
}

int lux_cuda_exp_float(void* output, const void* input, uint64_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;
    exp_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>((float*)output, (const float*)input, n);
    return cudaGetLastError();
}

int lux_cuda_log_float(void* output, const void* input, uint64_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;
    log_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>((float*)output, (const float*)input, n);
    return cudaGetLastError();
}

int lux_cuda_sqrt_float(void* output, const void* input, uint64_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;
    sqrt_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>((float*)output, (const float*)input, n);
    return cudaGetLastError();
}

int lux_cuda_sin_float(void* output, const void* input, uint64_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;
    sin_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>((float*)output, (const float*)input, n);
    return cudaGetLastError();
}

int lux_cuda_cos_float(void* output, const void* input, uint64_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;
    cos_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>((float*)output, (const float*)input, n);
    return cudaGetLastError();
}

int lux_cuda_tanh_float(void* output, const void* input, uint64_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;
    tanh_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>((float*)output, (const float*)input, n);
    return cudaGetLastError();
}

int lux_cuda_sigmoid_float(void* output, const void* input, uint64_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;
    sigmoid_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>((float*)output, (const float*)input, n);
    return cudaGetLastError();
}

int lux_cuda_relu_float(void* output, const void* input, uint64_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;
    relu_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>((float*)output, (const float*)input, n);
    return cudaGetLastError();
}

int lux_cuda_gelu_float(void* output, const void* input, uint64_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;
    gelu_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>((float*)output, (const float*)input, n);
    return cudaGetLastError();
}

int lux_cuda_gelu_fast_float(void* output, const void* input, uint64_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;
    gelu_fast_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>((float*)output, (const float*)input, n);
    return cudaGetLastError();
}

int lux_cuda_silu_float(void* output, const void* input, uint64_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;
    silu_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>((float*)output, (const float*)input, n);
    return cudaGetLastError();
}

int lux_cuda_leaky_relu_float(void* output, const void* input, float negative_slope, uint64_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;
    leaky_relu_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>((float*)output, (const float*)input, negative_slope, n);
    return cudaGetLastError();
}

int lux_cuda_elu_float(void* output, const void* input, float alpha, uint64_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;
    elu_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>((float*)output, (const float*)input, alpha, n);
    return cudaGetLastError();
}

int lux_cuda_isnan_float(void* output, const void* input, uint64_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;
    isnan_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>((bool*)output, (const float*)input, n);
    return cudaGetLastError();
}

int lux_cuda_isinf_float(void* output, const void* input, uint64_t n, cudaStream_t stream) {
    if (n == 0) return 0;
    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;
    isinf_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>((bool*)output, (const float*)input, n);
    return cudaGetLastError();
}

}  // extern "C"
