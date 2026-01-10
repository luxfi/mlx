// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Steel Fused GEMM NAX CUDA Kernels
// Non-blocking Asynchronous eXecution with pipelined memory operations

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// NAX Fused GEMM Configuration
// ============================================================================

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define NAX_STAGES 3    // Triple buffering for async pipeline
#define BLOCK_SIZE 256

// Epilogue types
#define EPILOGUE_NONE 0
#define EPILOGUE_BIAS 1
#define EPILOGUE_RELU 2
#define EPILOGUE_GELU 3
#define EPILOGUE_SILU 4
#define EPILOGUE_BIAS_RELU 5
#define EPILOGUE_BIAS_GELU 6
#define EPILOGUE_BIAS_SILU 7

// ============================================================================
// Activation Functions
// ============================================================================

__device__ __forceinline__ float relu_nax(float x) {
    return fmaxf(0.0f, x);
}

__device__ __forceinline__ float gelu_nax(float x) {
    const float c = 0.7978845608f;
    const float c2 = 0.044715f;
    float x3 = x * x * x;
    return x * 0.5f * (1.0f + tanhf(c * (x + c2 * x3)));
}

__device__ __forceinline__ float silu_nax(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float apply_activation_nax(float x, int activation_type) {
    switch (activation_type) {
        case EPILOGUE_RELU:
        case EPILOGUE_BIAS_RELU:
            return relu_nax(x);
        case EPILOGUE_GELU:
        case EPILOGUE_BIAS_GELU:
            return gelu_nax(x);
        case EPILOGUE_SILU:
        case EPILOGUE_BIAS_SILU:
            return silu_nax(x);
        default:
            return x;
    }
}

// ============================================================================
// NAX Fused GEMM with Software Pipelining
// ============================================================================

extern "C" __global__
void steel_gemm_fused_nax_kernel(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    int epilogue_type
) {
    // Triple-buffered shared memory for async pipeline
    __shared__ float As[NAX_STAGES][TILE_M][TILE_K + 1];
    __shared__ float Bs[NAX_STAGES][TILE_K][TILE_N + 1];

    uint32_t bx = blockIdx.x;
    uint32_t by = blockIdx.y;
    uint32_t tid = threadIdx.x;

    uint32_t thread_row = (tid / (TILE_N / 4)) * 4;
    uint32_t thread_col = (tid % (TILE_N / 4)) * 4;

    uint32_t row_base = by * TILE_M;
    uint32_t col_base = bx * TILE_N;

    // Accumulators
    float acc[4][4] = {{0.0f}};

    uint32_t num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // Prefill pipeline stages
    #pragma unroll
    for (int stage = 0; stage < min((int)NAX_STAGES, (int)num_k_tiles); stage++) {
        uint32_t k_start = stage * TILE_K;

        for (uint32_t i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
            uint32_t ti = i / TILE_K;
            uint32_t tk = i % TILE_K;
            uint32_t a_row = row_base + ti;
            uint32_t a_col = k_start + tk;

            As[stage][ti][tk] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        }

        for (uint32_t i = tid; i < TILE_K * TILE_N; i += blockDim.x) {
            uint32_t tk = i / TILE_N;
            uint32_t tj = i % TILE_N;
            uint32_t b_row = k_start + tk;
            uint32_t b_col = col_base + tj;

            Bs[stage][tk][tj] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
        }
    }
    __syncthreads();

    // Main loop with pipelining
    for (uint32_t kt = 0; kt < num_k_tiles; kt++) {
        uint32_t compute_stage = kt % NAX_STAGES;
        uint32_t load_stage = (kt + NAX_STAGES - 1) % NAX_STAGES;
        uint32_t next_kt = kt + NAX_STAGES;

        // Async load next tile if available
        if (next_kt < num_k_tiles) {
            uint32_t k_start = next_kt * TILE_K;

            for (uint32_t i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
                uint32_t ti = i / TILE_K;
                uint32_t tk = i % TILE_K;
                uint32_t a_row = row_base + ti;
                uint32_t a_col = k_start + tk;

                As[load_stage][ti][tk] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
            }

            for (uint32_t i = tid; i < TILE_K * TILE_N; i += blockDim.x) {
                uint32_t tk = i / TILE_N;
                uint32_t tj = i % TILE_N;
                uint32_t b_row = k_start + tk;
                uint32_t b_col = col_base + tj;

                Bs[load_stage][tk][tj] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
            }
        }

        // Compute on current stage
        #pragma unroll
        for (uint32_t k = 0; k < TILE_K; k++) {
            float a_frag[4];
            float b_frag[4];

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                a_frag[i] = As[compute_stage][thread_row + i][k];
                b_frag[i] = Bs[compute_stage][k][thread_col + i];
            }

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    acc[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }
        __syncthreads();
    }

    // Write results with fused epilogue
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t row = row_base + thread_row + i;
        if (row >= M) continue;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t col = col_base + thread_col + j;
            if (col >= N) continue;

            float result = alpha * acc[i][j];

            if (beta != 0.0f) {
                result += beta * C[row * N + col];
            }

            bool has_bias = (epilogue_type == EPILOGUE_BIAS ||
                           epilogue_type == EPILOGUE_BIAS_RELU ||
                           epilogue_type == EPILOGUE_BIAS_GELU ||
                           epilogue_type == EPILOGUE_BIAS_SILU);
            if (has_bias && bias != nullptr) {
                result += bias[col];
            }

            result = apply_activation_nax(result, epilogue_type);

            C[row * N + col] = result;
        }
    }
}

// ============================================================================
// NAX Fused GEMM + MLP Block (Gated Linear Unit variant)
// ============================================================================

// Fused: C = (A @ W1) * silu(A @ W2)  (GLU-style MLP)
extern "C" __global__
void steel_gemm_glu_nax_kernel(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ W1,
    const float* __restrict__ W2,
    uint32_t M,
    uint32_t N,
    uint32_t K
) {
    __shared__ float As[NAX_STAGES][TILE_M][TILE_K + 1];
    __shared__ float W1s[NAX_STAGES][TILE_K][TILE_N + 1];
    __shared__ float W2s[NAX_STAGES][TILE_K][TILE_N + 1];

    uint32_t bx = blockIdx.x;
    uint32_t by = blockIdx.y;
    uint32_t tid = threadIdx.x;

    uint32_t thread_row = (tid / (TILE_N / 4)) * 4;
    uint32_t thread_col = (tid % (TILE_N / 4)) * 4;

    uint32_t row_base = by * TILE_M;
    uint32_t col_base = bx * TILE_N;

    float acc1[4][4] = {{0.0f}};  // A @ W1
    float acc2[4][4] = {{0.0f}};  // A @ W2

    uint32_t num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // Prefill
    for (int stage = 0; stage < min((int)NAX_STAGES, (int)num_k_tiles); stage++) {
        uint32_t k_start = stage * TILE_K;

        for (uint32_t i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
            uint32_t ti = i / TILE_K;
            uint32_t tk = i % TILE_K;
            uint32_t a_row = row_base + ti;
            uint32_t a_col = k_start + tk;

            As[stage][ti][tk] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        }

        for (uint32_t i = tid; i < TILE_K * TILE_N; i += blockDim.x) {
            uint32_t tk = i / TILE_N;
            uint32_t tj = i % TILE_N;
            uint32_t b_row = k_start + tk;
            uint32_t b_col = col_base + tj;

            W1s[stage][tk][tj] = (b_row < K && b_col < N) ? W1[b_row * N + b_col] : 0.0f;
            W2s[stage][tk][tj] = (b_row < K && b_col < N) ? W2[b_row * N + b_col] : 0.0f;
        }
    }
    __syncthreads();

    // Main loop
    for (uint32_t kt = 0; kt < num_k_tiles; kt++) {
        uint32_t compute_stage = kt % NAX_STAGES;
        uint32_t load_stage = (kt + NAX_STAGES - 1) % NAX_STAGES;
        uint32_t next_kt = kt + NAX_STAGES;

        // Async load next tiles
        if (next_kt < num_k_tiles) {
            uint32_t k_start = next_kt * TILE_K;

            for (uint32_t i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
                uint32_t ti = i / TILE_K;
                uint32_t tk = i % TILE_K;
                uint32_t a_row = row_base + ti;
                uint32_t a_col = k_start + tk;

                As[load_stage][ti][tk] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
            }

            for (uint32_t i = tid; i < TILE_K * TILE_N; i += blockDim.x) {
                uint32_t tk = i / TILE_N;
                uint32_t tj = i % TILE_N;
                uint32_t b_row = k_start + tk;
                uint32_t b_col = col_base + tj;

                W1s[load_stage][tk][tj] = (b_row < K && b_col < N) ? W1[b_row * N + b_col] : 0.0f;
                W2s[load_stage][tk][tj] = (b_row < K && b_col < N) ? W2[b_row * N + b_col] : 0.0f;
            }
        }

        // Compute both GEMMs in parallel
        #pragma unroll
        for (uint32_t k = 0; k < TILE_K; k++) {
            float a_frag[4];

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                a_frag[i] = As[compute_stage][thread_row + i][k];
            }

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    acc1[i][j] += a_frag[i] * W1s[compute_stage][k][thread_col + j];
                    acc2[i][j] += a_frag[i] * W2s[compute_stage][k][thread_col + j];
                }
            }
        }
        __syncthreads();
    }

    // Write: C = acc1 * silu(acc2)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t row = row_base + thread_row + i;
        if (row >= M) continue;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t col = col_base + thread_col + j;
            if (col >= N) continue;

            float gate = silu_nax(acc2[i][j]);
            C[row * N + col] = acc1[i][j] * gate;
        }
    }
}

// ============================================================================
// NAX FP16 Fused GEMM
// ============================================================================

extern "C" __global__
void steel_gemm_fused_nax_fp16_kernel(
    __half* __restrict__ C,
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    const __half* __restrict__ bias,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    int epilogue_type
) {
    __shared__ __half As[NAX_STAGES][TILE_M][TILE_K + 1];
    __shared__ __half Bs[NAX_STAGES][TILE_K][TILE_N + 1];

    uint32_t bx = blockIdx.x;
    uint32_t by = blockIdx.y;
    uint32_t tid = threadIdx.x;

    uint32_t thread_row = (tid / (TILE_N / 4)) * 4;
    uint32_t thread_col = (tid % (TILE_N / 4)) * 4;

    uint32_t row_base = by * TILE_M;
    uint32_t col_base = bx * TILE_N;

    float acc[4][4] = {{0.0f}};

    uint32_t num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // Prefill
    for (int stage = 0; stage < min((int)NAX_STAGES, (int)num_k_tiles); stage++) {
        uint32_t k_start = stage * TILE_K;

        for (uint32_t i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
            uint32_t ti = i / TILE_K;
            uint32_t tk = i % TILE_K;
            uint32_t a_row = row_base + ti;
            uint32_t a_col = k_start + tk;

            As[stage][ti][tk] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : __float2half(0.0f);
        }

        for (uint32_t i = tid; i < TILE_K * TILE_N; i += blockDim.x) {
            uint32_t tk = i / TILE_N;
            uint32_t tj = i % TILE_N;
            uint32_t b_row = k_start + tk;
            uint32_t b_col = col_base + tj;

            Bs[stage][tk][tj] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : __float2half(0.0f);
        }
    }
    __syncthreads();

    // Main loop
    for (uint32_t kt = 0; kt < num_k_tiles; kt++) {
        uint32_t compute_stage = kt % NAX_STAGES;
        uint32_t load_stage = (kt + NAX_STAGES - 1) % NAX_STAGES;
        uint32_t next_kt = kt + NAX_STAGES;

        if (next_kt < num_k_tiles) {
            uint32_t k_start = next_kt * TILE_K;

            for (uint32_t i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
                uint32_t ti = i / TILE_K;
                uint32_t tk = i % TILE_K;
                uint32_t a_row = row_base + ti;
                uint32_t a_col = k_start + tk;

                As[load_stage][ti][tk] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : __float2half(0.0f);
            }

            for (uint32_t i = tid; i < TILE_K * TILE_N; i += blockDim.x) {
                uint32_t tk = i / TILE_N;
                uint32_t tj = i % TILE_N;
                uint32_t b_row = k_start + tk;
                uint32_t b_col = col_base + tj;

                Bs[load_stage][tk][tj] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : __float2half(0.0f);
            }
        }

        #pragma unroll
        for (uint32_t k = 0; k < TILE_K; k++) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    acc[i][j] += __half2float(As[compute_stage][thread_row + i][k]) *
                                __half2float(Bs[compute_stage][k][thread_col + j]);
                }
            }
        }
        __syncthreads();
    }

    // Write with epilogue
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t row = row_base + thread_row + i;
        if (row >= M) continue;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t col = col_base + thread_col + j;
            if (col >= N) continue;

            float result = alpha * acc[i][j];

            if (beta != 0.0f) {
                result += beta * __half2float(C[row * N + col]);
            }

            bool has_bias = (epilogue_type >= EPILOGUE_BIAS);
            if (has_bias && bias != nullptr) {
                result += __half2float(bias[col]);
            }

            result = apply_activation_nax(result, epilogue_type);

            C[row * N + col] = __float2half(result);
        }
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_steel_gemm_fused_nax(
    void* C,
    const void* A,
    const void* B,
    const void* bias,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    int epilogue_type,
    cudaStream_t stream
) {
    dim3 blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    uint32_t threads = BLOCK_SIZE;

    steel_gemm_fused_nax_kernel<<<blocks, threads, 0, stream>>>(
        (float*)C,
        (const float*)A,
        (const float*)B,
        (const float*)bias,
        alpha, beta,
        M, N, K,
        epilogue_type
    );

    return cudaGetLastError();
}

int lux_cuda_steel_gemm_glu_nax(
    void* C,
    const void* A,
    const void* W1,
    const void* W2,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    cudaStream_t stream
) {
    dim3 blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    uint32_t threads = BLOCK_SIZE;

    steel_gemm_glu_nax_kernel<<<blocks, threads, 0, stream>>>(
        (float*)C,
        (const float*)A,
        (const float*)W1,
        (const float*)W2,
        M, N, K
    );

    return cudaGetLastError();
}

int lux_cuda_steel_gemm_fused_nax_fp16(
    void* C,
    const void* A,
    const void* B,
    const void* bias,
    float alpha,
    float beta,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    int epilogue_type,
    cudaStream_t stream
) {
    dim3 blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    uint32_t threads = BLOCK_SIZE;

    steel_gemm_fused_nax_fp16_kernel<<<blocks, threads, 0, stream>>>(
        (__half*)C,
        (const __half*)A,
        (const __half*)B,
        (const __half*)bias,
        alpha, beta,
        M, N, K,
        epilogue_type
    );

    return cudaGetLastError();
}

}  // extern "C"
