// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Binary CUDA Kernels
// Element-wise binary operations (add, sub, mul, div, pow, mod, etc.)
// Supports float, half, int32, int64, and boolean types with broadcasting

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
// Binary Operation Types
// ============================================================================

enum class BinaryOp : uint32_t {
    ADD = 0,
    SUB = 1,
    MUL = 2,
    DIV = 3,
    POW = 4,
    MOD = 5,
    FMOD = 6,
    MAX = 7,
    MIN = 8,
    ATAN2 = 9,
    HYPOT = 10,
    COPYSIGN = 11,
    // Comparison ops
    EQ = 20,
    NE = 21,
    LT = 22,
    LE = 23,
    GT = 24,
    GE = 25,
    // Logical ops
    AND = 30,
    OR = 31,
    XOR = 32,
    // Bitwise ops (for integers)
    BIT_AND = 40,
    BIT_OR = 41,
    BIT_XOR = 42,
    BIT_LSHIFT = 43,
    BIT_RSHIFT = 44,
};

// ============================================================================
// Binary Operation Implementations - Float
// ============================================================================

__device__ __forceinline__
float binary_op_float(float a, float b, BinaryOp op) {
    switch (op) {
        case BinaryOp::ADD: return a + b;
        case BinaryOp::SUB: return a - b;
        case BinaryOp::MUL: return a * b;
        case BinaryOp::DIV: return a / b;
        case BinaryOp::POW: return powf(a, b);
        case BinaryOp::MOD: return fmodf(a, b);
        case BinaryOp::FMOD: return fmodf(a, b);
        case BinaryOp::MAX: return fmaxf(a, b);
        case BinaryOp::MIN: return fminf(a, b);
        case BinaryOp::ATAN2: return atan2f(a, b);
        case BinaryOp::HYPOT: return hypotf(a, b);
        case BinaryOp::COPYSIGN: return copysignf(a, b);
        default: return 0.0f;
    }
}

__device__ __forceinline__
bool binary_cmp_float(float a, float b, BinaryOp op) {
    switch (op) {
        case BinaryOp::EQ: return a == b;
        case BinaryOp::NE: return a != b;
        case BinaryOp::LT: return a < b;
        case BinaryOp::LE: return a <= b;
        case BinaryOp::GT: return a > b;
        case BinaryOp::GE: return a >= b;
        default: return false;
    }
}

// ============================================================================
// Binary Operation Implementations - Int32
// ============================================================================

__device__ __forceinline__
int32_t binary_op_int32(int32_t a, int32_t b, BinaryOp op) {
    switch (op) {
        case BinaryOp::ADD: return a + b;
        case BinaryOp::SUB: return a - b;
        case BinaryOp::MUL: return a * b;
        case BinaryOp::DIV: return b != 0 ? a / b : 0;
        case BinaryOp::MOD: return b != 0 ? a % b : 0;
        case BinaryOp::MAX: return a > b ? a : b;
        case BinaryOp::MIN: return a < b ? a : b;
        case BinaryOp::BIT_AND: return a & b;
        case BinaryOp::BIT_OR: return a | b;
        case BinaryOp::BIT_XOR: return a ^ b;
        case BinaryOp::BIT_LSHIFT: return a << b;
        case BinaryOp::BIT_RSHIFT: return a >> b;
        default: return 0;
    }
}

__device__ __forceinline__
bool binary_cmp_int32(int32_t a, int32_t b, BinaryOp op) {
    switch (op) {
        case BinaryOp::EQ: return a == b;
        case BinaryOp::NE: return a != b;
        case BinaryOp::LT: return a < b;
        case BinaryOp::LE: return a <= b;
        case BinaryOp::GT: return a > b;
        case BinaryOp::GE: return a >= b;
        default: return false;
    }
}

// ============================================================================
// Contiguous Binary Kernels - Float
// ============================================================================

extern "C" __global__
void binary_ss_float_kernel(
    float* __restrict__ out,
    const float* __restrict__ a,
    const float* __restrict__ b,
    uint64_t n,
    uint32_t op
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            out[elem_idx] = binary_op_float(a[elem_idx], b[elem_idx], (BinaryOp)op);
        }
    }
}

// Scalar-vector: a is scalar, b is vector
extern "C" __global__
void binary_sv_float_kernel(
    float* __restrict__ out,
    float a,
    const float* __restrict__ b,
    uint64_t n,
    uint32_t op
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            out[elem_idx] = binary_op_float(a, b[elem_idx], (BinaryOp)op);
        }
    }
}

// Vector-scalar: a is vector, b is scalar
extern "C" __global__
void binary_vs_float_kernel(
    float* __restrict__ out,
    const float* __restrict__ a,
    float b,
    uint64_t n,
    uint32_t op
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            out[elem_idx] = binary_op_float(a[elem_idx], b, (BinaryOp)op);
        }
    }
}

// ============================================================================
// Comparison Kernels - Float
// ============================================================================

extern "C" __global__
void binary_cmp_ss_float_kernel(
    bool* __restrict__ out,
    const float* __restrict__ a,
    const float* __restrict__ b,
    uint64_t n,
    uint32_t op
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            out[elem_idx] = binary_cmp_float(a[elem_idx], b[elem_idx], (BinaryOp)op);
        }
    }
}

extern "C" __global__
void binary_cmp_vs_float_kernel(
    bool* __restrict__ out,
    const float* __restrict__ a,
    float b,
    uint64_t n,
    uint32_t op
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            out[elem_idx] = binary_cmp_float(a[elem_idx], b, (BinaryOp)op);
        }
    }
}

// ============================================================================
// Contiguous Binary Kernels - Half
// ============================================================================

extern "C" __global__
void binary_ss_half_kernel(
    __half* __restrict__ out,
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    uint64_t n,
    uint32_t op
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float fa = __half2float(a[elem_idx]);
            float fb = __half2float(b[elem_idx]);
            out[elem_idx] = __float2half(binary_op_float(fa, fb, (BinaryOp)op));
        }
    }
}

extern "C" __global__
void binary_vs_half_kernel(
    __half* __restrict__ out,
    const __half* __restrict__ a,
    float b,
    uint64_t n,
    uint32_t op
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            float fa = __half2float(a[elem_idx]);
            out[elem_idx] = __float2half(binary_op_float(fa, b, (BinaryOp)op));
        }
    }
}

// ============================================================================
// Contiguous Binary Kernels - Int32
// ============================================================================

extern "C" __global__
void binary_ss_int32_kernel(
    int32_t* __restrict__ out,
    const int32_t* __restrict__ a,
    const int32_t* __restrict__ b,
    uint64_t n,
    uint32_t op
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            out[elem_idx] = binary_op_int32(a[elem_idx], b[elem_idx], (BinaryOp)op);
        }
    }
}

extern "C" __global__
void binary_vs_int32_kernel(
    int32_t* __restrict__ out,
    const int32_t* __restrict__ a,
    int32_t b,
    uint64_t n,
    uint32_t op
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            out[elem_idx] = binary_op_int32(a[elem_idx], b, (BinaryOp)op);
        }
    }
}

// ============================================================================
// Broadcasting Binary Kernels
// ============================================================================

// General broadcasting with strides
extern "C" __global__
void binary_broadcast_float_kernel(
    float* __restrict__ out,
    const float* __restrict__ a,
    const float* __restrict__ b,
    const int64_t* __restrict__ a_strides,
    const int64_t* __restrict__ b_strides,
    const int64_t* __restrict__ out_strides,
    const uint64_t* __restrict__ shape,
    uint32_t ndim,
    uint64_t n,
    uint32_t op
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Convert linear index to multi-dimensional indices
    uint64_t remaining = idx;
    uint64_t a_offset = 0;
    uint64_t b_offset = 0;

    for (int d = ndim - 1; d >= 0; d--) {
        uint64_t dim_idx = remaining % shape[d];
        remaining /= shape[d];
        a_offset += dim_idx * a_strides[d];
        b_offset += dim_idx * b_strides[d];
    }

    out[idx] = binary_op_float(a[a_offset], b[b_offset], (BinaryOp)op);
}

// Optimized 2D broadcast: (M, 1) op (1, N) -> (M, N)
extern "C" __global__
void binary_broadcast_2d_float_kernel(
    float* __restrict__ out,
    const float* __restrict__ a,  // Shape (M, 1) or (M,)
    const float* __restrict__ b,  // Shape (1, N) or (N,)
    uint32_t M,
    uint32_t N,
    uint32_t op
) {
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        out[row * N + col] = binary_op_float(a[row], b[col], (BinaryOp)op);
    }
}

// Row broadcast: (M, N) op (N,) -> (M, N)
extern "C" __global__
void binary_row_broadcast_float_kernel(
    float* __restrict__ out,
    const float* __restrict__ a,  // Shape (M, N)
    const float* __restrict__ b,  // Shape (N,)
    uint32_t M,
    uint32_t N,
    uint32_t op
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)M * N;

    if (idx < total) {
        uint32_t col = idx % N;
        out[idx] = binary_op_float(a[idx], b[col], (BinaryOp)op);
    }
}

// Col broadcast: (M, N) op (M, 1) -> (M, N)
extern "C" __global__
void binary_col_broadcast_float_kernel(
    float* __restrict__ out,
    const float* __restrict__ a,  // Shape (M, N)
    const float* __restrict__ b,  // Shape (M, 1) or (M,)
    uint32_t M,
    uint32_t N,
    uint32_t op
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)M * N;

    if (idx < total) {
        uint32_t row = idx / N;
        out[idx] = binary_op_float(a[idx], b[row], (BinaryOp)op);
    }
}

// ============================================================================
// Logical Operations - Boolean
// ============================================================================

extern "C" __global__
void binary_logical_kernel(
    bool* __restrict__ out,
    const bool* __restrict__ a,
    const bool* __restrict__ b,
    uint64_t n,
    uint32_t op
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            bool va = a[elem_idx];
            bool vb = b[elem_idx];
            bool result;
            switch ((BinaryOp)op) {
                case BinaryOp::AND: result = va && vb; break;
                case BinaryOp::OR: result = va || vb; break;
                case BinaryOp::XOR: result = va != vb; break;
                default: result = false;
            }
            out[elem_idx] = result;
        }
    }
}

// ============================================================================
// In-place Operations
// ============================================================================

extern "C" __global__
void binary_inplace_ss_float_kernel(
    float* __restrict__ a,
    const float* __restrict__ b,
    uint64_t n,
    uint32_t op
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            a[elem_idx] = binary_op_float(a[elem_idx], b[elem_idx], (BinaryOp)op);
        }
    }
}

extern "C" __global__
void binary_inplace_vs_float_kernel(
    float* __restrict__ a,
    float b,
    uint64_t n,
    uint32_t op
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            a[elem_idx] = binary_op_float(a[elem_idx], b, (BinaryOp)op);
        }
    }
}

// ============================================================================
// Specialized Fast Paths
// ============================================================================

// Fast add kernel
extern "C" __global__
void add_float_kernel(
    float* __restrict__ out,
    const float* __restrict__ a,
    const float* __restrict__ b,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            out[elem_idx] = a[elem_idx] + b[elem_idx];
        }
    }
}

// Fast multiply kernel
extern "C" __global__
void mul_float_kernel(
    float* __restrict__ out,
    const float* __restrict__ a,
    const float* __restrict__ b,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            out[elem_idx] = a[elem_idx] * b[elem_idx];
        }
    }
}

// Fused multiply-add: out = a * b + c
extern "C" __global__
void fma_float_kernel(
    float* __restrict__ out,
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    uint64_t n
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        uint64_t elem_idx = idx + i * blockDim.x;
        if (elem_idx < n) {
            out[elem_idx] = fmaf(a[elem_idx], b[elem_idx], c[elem_idx]);
        }
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_binary_float(
    void* out,
    const void* a,
    const void* b,
    uint64_t n,
    uint32_t op,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    binary_ss_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)out, (const float*)a, (const float*)b, n, op
    );

    return cudaGetLastError();
}

int lux_cuda_binary_scalar_float(
    void* out,
    const void* a,
    float b,
    uint64_t n,
    uint32_t op,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    binary_vs_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)out, (const float*)a, b, n, op
    );

    return cudaGetLastError();
}

int lux_cuda_binary_half(
    void* out,
    const void* a,
    const void* b,
    uint64_t n,
    uint32_t op,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    binary_ss_half_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (__half*)out, (const __half*)a, (const __half*)b, n, op
    );

    return cudaGetLastError();
}

int lux_cuda_binary_int32(
    void* out,
    const void* a,
    const void* b,
    uint64_t n,
    uint32_t op,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    binary_ss_int32_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (int32_t*)out, (const int32_t*)a, (const int32_t*)b, n, op
    );

    return cudaGetLastError();
}

int lux_cuda_compare_float(
    void* out,
    const void* a,
    const void* b,
    uint64_t n,
    uint32_t op,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    binary_cmp_ss_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (bool*)out, (const float*)a, (const float*)b, n, op
    );

    return cudaGetLastError();
}

int lux_cuda_add_float(
    void* out,
    const void* a,
    const void* b,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    add_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)out, (const float*)a, (const float*)b, n
    );

    return cudaGetLastError();
}

int lux_cuda_mul_float(
    void* out,
    const void* a,
    const void* b,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    mul_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)out, (const float*)a, (const float*)b, n
    );

    return cudaGetLastError();
}

int lux_cuda_fma_float(
    void* out,
    const void* a,
    const void* b,
    const void* c,
    uint64_t n,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    fma_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)out, (const float*)a, (const float*)b, (const float*)c, n
    );

    return cudaGetLastError();
}

int lux_cuda_binary_broadcast_float(
    void* out,
    const void* a,
    const void* b,
    const int64_t* a_strides,
    const int64_t* b_strides,
    const int64_t* out_strides,
    const uint64_t* shape,
    uint32_t ndim,
    uint64_t n,
    uint32_t op,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    binary_broadcast_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)out, (const float*)a, (const float*)b,
        a_strides, b_strides, out_strides, shape, ndim, n, op
    );

    return cudaGetLastError();
}

int lux_cuda_binary_row_broadcast_float(
    void* out,
    const void* a,
    const void* b,
    uint32_t M,
    uint32_t N,
    uint32_t op,
    cudaStream_t stream
) {
    uint64_t total = (uint64_t)M * N;
    if (total == 0) return 0;

    uint64_t num_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    binary_row_broadcast_float_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (float*)out, (const float*)a, (const float*)b, M, N, op
    );

    return cudaGetLastError();
}

int lux_cuda_logical(
    void* out,
    const void* a,
    const void* b,
    uint64_t n,
    uint32_t op,
    cudaStream_t stream
) {
    if (n == 0) return 0;

    uint64_t elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    uint64_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

    binary_logical_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        (bool*)out, (const bool*)a, (const bool*)b, n, op
    );

    return cudaGetLastError();
}

}  // extern "C"
