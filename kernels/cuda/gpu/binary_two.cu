// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Binary Two-Output CUDA Kernels
// Binary operations that produce two outputs (divmod, sincos, modf, frexp, etc.)
// Supports float, half, and integer types

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
// Two-Output Operation Types
// ============================================================================

enum class BinaryTwoOp : uint32_t {
    DIVMOD = 0,      // (quotient, remainder)
    SINCOS = 1,      // (sin, cos) - single input
    MODF = 2,        // (fractional, integral) - single input
    FREXP = 3,       // (mantissa, exponent) - single input
    REMQUO = 4,      // (remainder, quotient)
    MINMAX = 5,      // (min, max)
    SORT2 = 6,       // (smaller, larger)
};

// ============================================================================
// Divmod Kernels - Integer
// ============================================================================

extern "C" __global__
void divmod_int32_kernel(
    int32_t* __restrict__ quotient,
    int32_t* __restrict__ remainder,
    const int32_t* __restrict__ a,
    const int32_t* __restrict__ b,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            int32_t va = a[elem_idx];
            int32_t vb = b[elem_idx];
            if (vb != 0) {
                quotient[elem_idx] = va / vb;
                remainder[elem_idx] = va % vb;
            } else {
                quotient[elem_idx] = 0;
                remainder[elem_idx] = 0;
            }
        }
    }
}

extern "C" __global__
void divmod_int64_kernel(
    int64_t* __restrict__ quotient,
    int64_t* __restrict__ remainder,
    const int64_t* __restrict__ a,
    const int64_t* __restrict__ b,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            int64_t va = a[elem_idx];
            int64_t vb = b[elem_idx];
            if (vb != 0) {
                quotient[elem_idx] = va / vb;
                remainder[elem_idx] = va % vb;
            } else {
                quotient[elem_idx] = 0;
                remainder[elem_idx] = 0;
            }
        }
    }
}

// Divmod with scalar divisor
extern "C" __global__
void divmod_scalar_int32_kernel(
    int32_t* __restrict__ quotient,
    int32_t* __restrict__ remainder,
    const int32_t* __restrict__ a,
    int32_t b,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    if (b == 0) {
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            uint64_t elem_idx = idx + i * blockDim.x;
            if (elem_idx < n) {
                quotient[elem_idx] = 0;
                remainder[elem_idx] = 0;
            }
        }
        return;
    }

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            int32_t va = a[elem_idx];
            quotient[elem_idx] = va / b;
            remainder[elem_idx] = va % b;
        }
    }
}

// ============================================================================
// Sincos Kernels - Float
// ============================================================================

extern "C" __global__
void sincos_float_kernel(
    float* __restrict__ sin_out,
    float* __restrict__ cos_out,
    const float* __restrict__ input,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float val = input[elem_idx];
            float s, c;
            sincosf(val, &s, &c);
            sin_out[elem_idx] = s;
            cos_out[elem_idx] = c;
        }
    }
}

extern "C" __global__
void sincos_half_kernel(
    __half* __restrict__ sin_out,
    __half* __restrict__ cos_out,
    const __half* __restrict__ input,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float val = __half2float(input[elem_idx]);
            float s, c;
            sincosf(val, &s, &c);
            sin_out[elem_idx] = __float2half(s);
            cos_out[elem_idx] = __float2half(c);
        }
    }
}

// ============================================================================
// Modf Kernels - Float (split into fractional and integral parts)
// ============================================================================

extern "C" __global__
void modf_float_kernel(
    float* __restrict__ frac_out,
    float* __restrict__ int_out,
    const float* __restrict__ input,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float val = input[elem_idx];
            float int_part;
            float frac_part = modff(val, &int_part);
            frac_out[elem_idx] = frac_part;
            int_out[elem_idx] = int_part;
        }
    }
}

extern "C" __global__
void modf_half_kernel(
    __half* __restrict__ frac_out,
    __half* __restrict__ int_out,
    const __half* __restrict__ input,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float val = __half2float(input[elem_idx]);
            float int_part;
            float frac_part = modff(val, &int_part);
            frac_out[elem_idx] = __float2half(frac_part);
            int_out[elem_idx] = __float2half(int_part);
        }
    }
}

// ============================================================================
// Frexp Kernels - Float (split into mantissa and exponent)
// ============================================================================

extern "C" __global__
void frexp_float_kernel(
    float* __restrict__ mantissa_out,
    int32_t* __restrict__ exp_out,
    const float* __restrict__ input,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float val = input[elem_idx];
            int exp;
            float mantissa = frexpf(val, &exp);
            mantissa_out[elem_idx] = mantissa;
            exp_out[elem_idx] = exp;
        }
    }
}

// ============================================================================
// Remquo Kernels - Float (remainder with quotient bits)
// ============================================================================

extern "C" __global__
void remquo_float_kernel(
    float* __restrict__ rem_out,
    int32_t* __restrict__ quo_out,
    const float* __restrict__ a,
    const float* __restrict__ b,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float va = a[elem_idx];
            float vb = b[elem_idx];
            int quo;
            float rem = remquof(va, vb, &quo);
            rem_out[elem_idx] = rem;
            quo_out[elem_idx] = quo;
        }
    }
}

// ============================================================================
// MinMax Kernels - Compute both min and max simultaneously
// ============================================================================

extern "C" __global__
void minmax_float_kernel(
    float* __restrict__ min_out,
    float* __restrict__ max_out,
    const float* __restrict__ a,
    const float* __restrict__ b,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float va = a[elem_idx];
            float vb = b[elem_idx];
            min_out[elem_idx] = fminf(va, vb);
            max_out[elem_idx] = fmaxf(va, vb);
        }
    }
}

extern "C" __global__
void minmax_int32_kernel(
    int32_t* __restrict__ min_out,
    int32_t* __restrict__ max_out,
    const int32_t* __restrict__ a,
    const int32_t* __restrict__ b,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            int32_t va = a[elem_idx];
            int32_t vb = b[elem_idx];
            min_out[elem_idx] = va < vb ? va : vb;
            max_out[elem_idx] = va > vb ? va : vb;
        }
    }
}

// ============================================================================
// Sort2 Kernels - Sort pairs (useful for sorting networks)
// ============================================================================

extern "C" __global__
void sort2_float_kernel(
    float* __restrict__ smaller_out,
    float* __restrict__ larger_out,
    const float* __restrict__ a,
    const float* __restrict__ b,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float va = a[elem_idx];
            float vb = b[elem_idx];
            if (va <= vb) {
                smaller_out[elem_idx] = va;
                larger_out[elem_idx] = vb;
            } else {
                smaller_out[elem_idx] = vb;
                larger_out[elem_idx] = va;
            }
        }
    }
}

// ============================================================================
// Floor/Ceil Divide with Remainder
// ============================================================================

extern "C" __global__
void floordiv_mod_float_kernel(
    float* __restrict__ div_out,
    float* __restrict__ mod_out,
    const float* __restrict__ a,
    const float* __restrict__ b,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float va = a[elem_idx];
            float vb = b[elem_idx];
            float q = floorf(va / vb);
            float r = va - q * vb;
            div_out[elem_idx] = q;
            mod_out[elem_idx] = r;
        }
    }
}

// ============================================================================
// Polar to Cartesian (r, theta) -> (x, y)
// ============================================================================

extern "C" __global__
void polar_to_cartesian_float_kernel(
    float* __restrict__ x_out,
    float* __restrict__ y_out,
    const float* __restrict__ r,
    const float* __restrict__ theta,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float radius = r[elem_idx];
            float angle = theta[elem_idx];
            float s, c;
            sincosf(angle, &s, &c);
            x_out[elem_idx] = radius * c;
            y_out[elem_idx] = radius * s;
        }
    }
}

// ============================================================================
// Cartesian to Polar (x, y) -> (r, theta)
// ============================================================================

extern "C" __global__
void cartesian_to_polar_float_kernel(
    float* __restrict__ r_out,
    float* __restrict__ theta_out,
    const float* __restrict__ x,
    const float* __restrict__ y,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float vx = x[elem_idx];
            float vy = y[elem_idx];
            r_out[elem_idx] = hypotf(vx, vy);
            theta_out[elem_idx] = atan2f(vy, vx);
        }
    }
}

// ============================================================================
// Complex Operations (real, imag) pairs
// ============================================================================

// Complex multiply: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
extern "C" __global__
void complex_mul_float_kernel(
    float* __restrict__ real_out,
    float* __restrict__ imag_out,
    const float* __restrict__ real_a,
    const float* __restrict__ imag_a,
    const float* __restrict__ real_b,
    const float* __restrict__ imag_b,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float ar = real_a[elem_idx];
            float ai = imag_a[elem_idx];
            float br = real_b[elem_idx];
            float bi = imag_b[elem_idx];
            real_out[elem_idx] = ar * br - ai * bi;
            imag_out[elem_idx] = ar * bi + ai * br;
        }
    }
}

// Complex divide: (a + bi) / (c + di)
extern "C" __global__
void complex_div_float_kernel(
    float* __restrict__ real_out,
    float* __restrict__ imag_out,
    const float* __restrict__ real_a,
    const float* __restrict__ imag_a,
    const float* __restrict__ real_b,
    const float* __restrict__ imag_b,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float ar = real_a[elem_idx];
            float ai = imag_a[elem_idx];
            float br = real_b[elem_idx];
            float bi = imag_b[elem_idx];
            float denom = br * br + bi * bi;
            if (denom != 0.0f) {
                real_out[elem_idx] = (ar * br + ai * bi) / denom;
                imag_out[elem_idx] = (ai * br - ar * bi) / denom;
            } else {
                real_out[elem_idx] = 0.0f;
                imag_out[elem_idx] = 0.0f;
            }
        }
    }
}

// Complex to polar: (real, imag) -> (abs, angle)
extern "C" __global__
void complex_to_polar_float_kernel(
    float* __restrict__ abs_out,
    float* __restrict__ angle_out,
    const float* __restrict__ real,
    const float* __restrict__ imag,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float r = real[elem_idx];
            float im = imag[elem_idx];
            abs_out[elem_idx] = hypotf(r, im);
            angle_out[elem_idx] = atan2f(im, r);
        }
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_divmod_int32(
    void* quotient,
    void* remainder,
    const void* a,
    const void* b,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    divmod_int32_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (int32_t*)quotient, (int32_t*)remainder,
        (const int32_t*)a, (const int32_t*)b, n
    );

    return cudaGetLastError();
}

int lux_cuda_divmod_int64(
    void* quotient,
    void* remainder,
    const void* a,
    const void* b,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    divmod_int64_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (int64_t*)quotient, (int64_t*)remainder,
        (const int64_t*)a, (const int64_t*)b, n
    );

    return cudaGetLastError();
}

int lux_cuda_divmod_scalar_int32(
    void* quotient,
    void* remainder,
    const void* a,
    int32_t b,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    divmod_scalar_int32_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (int32_t*)quotient, (int32_t*)remainder,
        (const int32_t*)a, b, n
    );

    return cudaGetLastError();
}

int lux_cuda_sincos_float(
    void* sin_out,
    void* cos_out,
    const void* input,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    sincos_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)sin_out, (float*)cos_out, (const float*)input, n
    );

    return cudaGetLastError();
}

int lux_cuda_sincos_half(
    void* sin_out,
    void* cos_out,
    const void* input,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    sincos_half_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (__half*)sin_out, (__half*)cos_out, (const __half*)input, n
    );

    return cudaGetLastError();
}

int lux_cuda_modf_float(
    void* frac_out,
    void* int_out,
    const void* input,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    modf_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)frac_out, (float*)int_out, (const float*)input, n
    );

    return cudaGetLastError();
}

int lux_cuda_frexp_float(
    void* mantissa_out,
    void* exp_out,
    const void* input,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    frexp_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)mantissa_out, (int32_t*)exp_out, (const float*)input, n
    );

    return cudaGetLastError();
}

int lux_cuda_minmax_float(
    void* min_out,
    void* max_out,
    const void* a,
    const void* b,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    minmax_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)min_out, (float*)max_out,
        (const float*)a, (const float*)b, n
    );

    return cudaGetLastError();
}

int lux_cuda_polar_to_cartesian_float(
    void* x_out,
    void* y_out,
    const void* r,
    const void* theta,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    polar_to_cartesian_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)x_out, (float*)y_out,
        (const float*)r, (const float*)theta, n
    );

    return cudaGetLastError();
}

int lux_cuda_cartesian_to_polar_float(
    void* r_out,
    void* theta_out,
    const void* x,
    const void* y,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    cartesian_to_polar_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)r_out, (float*)theta_out,
        (const float*)x, (const float*)y, n
    );

    return cudaGetLastError();
}

int lux_cuda_complex_mul_float(
    void* real_out,
    void* imag_out,
    const void* real_a,
    const void* imag_a,
    const void* real_b,
    const void* imag_b,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    complex_mul_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)real_out, (float*)imag_out,
        (const float*)real_a, (const float*)imag_a,
        (const float*)real_b, (const float*)imag_b, n
    );

    return cudaGetLastError();
}

int lux_cuda_complex_div_float(
    void* real_out,
    void* imag_out,
    const void* real_a,
    const void* imag_a,
    const void* real_b,
    const void* imag_b,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    complex_div_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)real_out, (float*)imag_out,
        (const float*)real_a, (const float*)imag_a,
        (const float*)real_b, (const float*)imag_b, n
    );

    return cudaGetLastError();
}

}  // extern "C"
