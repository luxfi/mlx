// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Steel Convolution CUDA Kernels - Tiled Implementation
// High-performance convolution using shared memory tiling

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// Steel Convolution Configuration
// ============================================================================

#define TILE_OUT 16      // Output tile size (H and W)
#define TILE_IC 32       // Input channel tile
#define TILE_OC 32       // Output channel tile
#define BLOCK_SIZE 256
#define MAX_KERNEL_SIZE 7

// ============================================================================
// Steel Tiled Conv2D
// ============================================================================

// Tiled convolution with shared memory for input and weights
extern "C" __global__
void steel_conv2d_tiled_kernel(
    float* __restrict__ output,          // [N, C_out, H_out, W_out]
    const float* __restrict__ input,     // [N, C_in, H_in, W_in]
    const float* __restrict__ weight,    // [C_out, C_in, K_h, K_w]
    const float* __restrict__ bias,      // [C_out] or nullptr
    uint32_t batch_size,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t in_height,
    uint32_t in_width,
    uint32_t kernel_h,
    uint32_t kernel_w,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t pad_h,
    uint32_t pad_w,
    uint32_t dilation_h,
    uint32_t dilation_w
) {
    // Shared memory for input tile and weight tile
    __shared__ float input_tile[TILE_IC][TILE_OUT + MAX_KERNEL_SIZE - 1][TILE_OUT + MAX_KERNEL_SIZE - 1];
    __shared__ float weight_tile[TILE_OC][TILE_IC][MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];

    uint32_t out_height = (in_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    uint32_t out_width = (in_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    uint32_t n = blockIdx.z;
    uint32_t oc_tile = blockIdx.y;
    uint32_t h_tile = (blockIdx.x / ((out_width + TILE_OUT - 1) / TILE_OUT)) * TILE_OUT;
    uint32_t w_tile = (blockIdx.x % ((out_width + TILE_OUT - 1) / TILE_OUT)) * TILE_OUT;

    uint32_t tid = threadIdx.x;
    uint32_t ty = tid / TILE_OUT;
    uint32_t tx = tid % TILE_OUT;

    if (n >= batch_size) return;

    // Output indices
    uint32_t h_out = h_tile + ty;
    uint32_t w_out = w_tile + tx;
    uint32_t oc_start = oc_tile * TILE_OC;

    // Accumulator for output
    float acc[TILE_OC];
    #pragma unroll
    for (int i = 0; i < TILE_OC; i++) {
        acc[i] = 0.0f;
    }

    // Iterate over input channel tiles
    uint32_t num_ic_tiles = (in_channels + TILE_IC - 1) / TILE_IC;

    for (uint32_t ic_tile = 0; ic_tile < num_ic_tiles; ic_tile++) {
        uint32_t ic_start = ic_tile * TILE_IC;

        // Load input tile to shared memory (with halo for kernel)
        uint32_t input_tile_h = TILE_OUT * stride_h + (kernel_h - 1) * dilation_h;
        uint32_t input_tile_w = TILE_OUT * stride_w + (kernel_w - 1) * dilation_w;

        for (uint32_t i = tid; i < TILE_IC * input_tile_h * input_tile_w; i += blockDim.x) {
            uint32_t ic_local = i / (input_tile_h * input_tile_w);
            uint32_t spatial = i % (input_tile_h * input_tile_w);
            uint32_t ih_local = spatial / input_tile_w;
            uint32_t iw_local = spatial % input_tile_w;

            uint32_t ic = ic_start + ic_local;
            int32_t ih = (int32_t)(h_tile * stride_h) - (int32_t)pad_h + (int32_t)ih_local;
            int32_t iw = (int32_t)(w_tile * stride_w) - (int32_t)pad_w + (int32_t)iw_local;

            float val = 0.0f;
            if (ic < in_channels && ih >= 0 && ih < (int32_t)in_height &&
                iw >= 0 && iw < (int32_t)in_width) {
                val = input[n * in_channels * in_height * in_width +
                           ic * in_height * in_width +
                           ih * in_width + iw];
            }

            if (ic_local < TILE_IC && ih_local < input_tile_h && iw_local < input_tile_w) {
                input_tile[ic_local][ih_local][iw_local] = val;
            }
        }

        // Load weight tile to shared memory
        for (uint32_t i = tid; i < TILE_OC * TILE_IC * kernel_h * kernel_w; i += blockDim.x) {
            uint32_t oc_local = i / (TILE_IC * kernel_h * kernel_w);
            uint32_t rem = i % (TILE_IC * kernel_h * kernel_w);
            uint32_t ic_local = rem / (kernel_h * kernel_w);
            uint32_t k_idx = rem % (kernel_h * kernel_w);
            uint32_t kh = k_idx / kernel_w;
            uint32_t kw = k_idx % kernel_w;

            uint32_t oc = oc_start + oc_local;
            uint32_t ic = ic_start + ic_local;

            float val = 0.0f;
            if (oc < out_channels && ic < in_channels) {
                val = weight[oc * in_channels * kernel_h * kernel_w +
                            ic * kernel_h * kernel_w +
                            kh * kernel_w + kw];
            }

            if (oc_local < TILE_OC && ic_local < TILE_IC && kh < kernel_h && kw < kernel_w) {
                weight_tile[oc_local][ic_local][kh][kw] = val;
            }
        }
        __syncthreads();

        // Compute convolution for this channel tile
        if (h_out < out_height && w_out < out_width) {
            for (uint32_t oc_local = 0; oc_local < TILE_OC; oc_local++) {
                uint32_t oc = oc_start + oc_local;
                if (oc >= out_channels) break;

                for (uint32_t ic_local = 0; ic_local < TILE_IC; ic_local++) {
                    uint32_t ic = ic_start + ic_local;
                    if (ic >= in_channels) break;

                    #pragma unroll
                    for (uint32_t kh = 0; kh < kernel_h; kh++) {
                        #pragma unroll
                        for (uint32_t kw = 0; kw < kernel_w; kw++) {
                            uint32_t ih_local = ty * stride_h + kh * dilation_h;
                            uint32_t iw_local = tx * stride_w + kw * dilation_w;

                            acc[oc_local] += input_tile[ic_local][ih_local][iw_local] *
                                            weight_tile[oc_local][ic_local][kh][kw];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // Write output with bias
    if (h_out < out_height && w_out < out_width) {
        for (uint32_t oc_local = 0; oc_local < TILE_OC; oc_local++) {
            uint32_t oc = oc_start + oc_local;
            if (oc >= out_channels) break;

            float result = acc[oc_local];
            if (bias != nullptr) {
                result += bias[oc];
            }

            output[n * out_channels * out_height * out_width +
                   oc * out_height * out_width +
                   h_out * out_width + w_out] = result;
        }
    }
}

// ============================================================================
// Steel Winograd Conv2D (F(2,3) - 2x2 output tile, 3x3 filter)
// ============================================================================

// Winograd transformation matrices for F(2,3)
// G: Filter transform, B: Input transform, A: Output transform
extern "C" __global__
void steel_conv2d_winograd_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    uint32_t batch_size,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t in_height,
    uint32_t in_width,
    uint32_t pad_h,
    uint32_t pad_w
) {
    // Winograd F(2,3): 4x4 input tile -> 2x2 output tile for 3x3 kernel
    __shared__ float input_tiles[TILE_IC][4][4];
    __shared__ float weight_tiles[TILE_OC][TILE_IC][4][4];
    __shared__ float temp[4][4];

    uint32_t out_height = in_height + 2 * pad_h - 2;  // 3x3 kernel
    uint32_t out_width = in_width + 2 * pad_w - 2;

    // Tile indices (each tile produces 2x2 output)
    uint32_t n = blockIdx.z;
    uint32_t oc_tile = blockIdx.y;
    uint32_t spatial_tile = blockIdx.x;

    uint32_t tiles_w = (out_width + 1) / 2;
    uint32_t tile_h = (spatial_tile / tiles_w) * 2;
    uint32_t tile_w = (spatial_tile % tiles_w) * 2;

    uint32_t tid = threadIdx.x;
    uint32_t oc_start = oc_tile * TILE_OC;

    if (n >= batch_size) return;

    // Winograd transform matrices (F(2,3))
    const float B[4][4] = {
        {1, 0, -1, 0},
        {0, 1, 1, 0},
        {0, -1, 1, 0},
        {0, 1, 0, -1}
    };

    const float G[4][3] = {
        {1, 0, 0},
        {0.5f, 0.5f, 0.5f},
        {0.5f, -0.5f, 0.5f},
        {0, 0, 1}
    };

    const float A[2][4] = {
        {1, 1, 1, 0},
        {0, 1, -1, -1}
    };

    // Accumulator for transformed domain
    float M[TILE_OC][4][4];
    #pragma unroll
    for (int oc = 0; oc < TILE_OC; oc++) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                M[oc][i][j] = 0.0f;
            }
        }
    }

    uint32_t num_ic_tiles = (in_channels + TILE_IC - 1) / TILE_IC;

    for (uint32_t ic_tile = 0; ic_tile < num_ic_tiles; ic_tile++) {
        uint32_t ic_start = ic_tile * TILE_IC;

        // Load and transform input tile: d = B^T * input * B
        for (uint32_t ic_local = tid; ic_local < TILE_IC; ic_local += blockDim.x) {
            uint32_t ic = ic_start + ic_local;
            if (ic >= in_channels) continue;

            // Load 4x4 input tile
            float d[4][4];
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    int32_t ih = (int32_t)tile_h - (int32_t)pad_h + i;
                    int32_t iw = (int32_t)tile_w - (int32_t)pad_w + j;

                    if (ih >= 0 && ih < (int32_t)in_height &&
                        iw >= 0 && iw < (int32_t)in_width) {
                        d[i][j] = input[n * in_channels * in_height * in_width +
                                       ic * in_height * in_width +
                                       ih * in_width + iw];
                    } else {
                        d[i][j] = 0.0f;
                    }
                }
            }

            // Transform: temp = B^T * d
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    temp[i][j] = 0.0f;
                    for (int k = 0; k < 4; k++) {
                        temp[i][j] += B[k][i] * d[k][j];
                    }
                }
            }

            // Transform: input_tiles = temp * B
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    input_tiles[ic_local][i][j] = 0.0f;
                    for (int k = 0; k < 4; k++) {
                        input_tiles[ic_local][i][j] += temp[i][k] * B[k][j];
                    }
                }
            }
        }

        // Load and transform weights: g = G * weight * G^T
        for (uint32_t i = tid; i < TILE_OC * TILE_IC; i += blockDim.x) {
            uint32_t oc_local = i / TILE_IC;
            uint32_t ic_local = i % TILE_IC;
            uint32_t oc = oc_start + oc_local;
            uint32_t ic = ic_start + ic_local;

            if (oc >= out_channels || ic >= in_channels) continue;

            // Load 3x3 kernel
            float k[3][3];
            for (int ki = 0; ki < 3; ki++) {
                for (int kj = 0; kj < 3; kj++) {
                    k[ki][kj] = weight[oc * in_channels * 9 + ic * 9 + ki * 3 + kj];
                }
            }

            // Transform: temp = G * k
            float tmp[4][3];
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 3; j++) {
                    tmp[i][j] = 0.0f;
                    for (int kk = 0; kk < 3; kk++) {
                        tmp[i][j] += G[i][kk] * k[kk][j];
                    }
                }
            }

            // Transform: weight_tiles = temp * G^T
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    weight_tiles[oc_local][ic_local][i][j] = 0.0f;
                    for (int kk = 0; kk < 3; kk++) {
                        weight_tiles[oc_local][ic_local][i][j] += tmp[i][kk] * G[j][kk];
                    }
                }
            }
        }
        __syncthreads();

        // Element-wise multiply in transformed domain
        for (uint32_t oc_local = 0; oc_local < TILE_OC; oc_local++) {
            uint32_t oc = oc_start + oc_local;
            if (oc >= out_channels) break;

            for (uint32_t ic_local = 0; ic_local < TILE_IC; ic_local++) {
                uint32_t ic = ic_start + ic_local;
                if (ic >= in_channels) break;

                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        M[oc_local][i][j] += input_tiles[ic_local][i][j] *
                                            weight_tiles[oc_local][ic_local][i][j];
                    }
                }
            }
        }
        __syncthreads();
    }

    // Inverse transform and write output: output = A^T * M * A
    for (uint32_t oc_local = tid; oc_local < TILE_OC; oc_local += blockDim.x) {
        uint32_t oc = oc_start + oc_local;
        if (oc >= out_channels) continue;

        // temp = A^T * M
        float tmp[2][4];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 4; j++) {
                tmp[i][j] = 0.0f;
                for (int k = 0; k < 4; k++) {
                    tmp[i][j] += A[i][k] * M[oc_local][k][j];
                }
            }
        }

        // out = temp * A
        float out[2][2];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                out[i][j] = 0.0f;
                for (int k = 0; k < 4; k++) {
                    out[i][j] += tmp[i][k] * A[j][k];
                }

                if (bias != nullptr) {
                    out[i][j] += bias[oc];
                }
            }
        }

        // Write 2x2 output
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                uint32_t oh = tile_h + i;
                uint32_t ow = tile_w + j;

                if (oh < out_height && ow < out_width) {
                    output[n * out_channels * out_height * out_width +
                           oc * out_height * out_width +
                           oh * out_width + ow] = out[i][j];
                }
            }
        }
    }
}

// ============================================================================
// Steel Separable Conv2D (Depthwise + Pointwise fused)
// ============================================================================

extern "C" __global__
void steel_conv2d_separable_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ depthwise_weight,  // [C_in, 1, K_h, K_w]
    const float* __restrict__ pointwise_weight,  // [C_out, C_in, 1, 1]
    const float* __restrict__ bias,
    uint32_t batch_size,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t in_height,
    uint32_t in_width,
    uint32_t kernel_h,
    uint32_t kernel_w,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t pad_h,
    uint32_t pad_w
) {
    __shared__ float depthwise_output[TILE_IC][TILE_OUT][TILE_OUT];

    uint32_t out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
    uint32_t out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;

    uint32_t n = blockIdx.z;
    uint32_t oc_tile = blockIdx.y;
    uint32_t spatial_tile = blockIdx.x;

    uint32_t tiles_w = (out_width + TILE_OUT - 1) / TILE_OUT;
    uint32_t h_tile = (spatial_tile / tiles_w) * TILE_OUT;
    uint32_t w_tile = (spatial_tile % tiles_w) * TILE_OUT;

    uint32_t tid = threadIdx.x;
    uint32_t ty = tid / TILE_OUT;
    uint32_t tx = tid % TILE_OUT;

    if (n >= batch_size) return;

    uint32_t h_out = h_tile + ty;
    uint32_t w_out = w_tile + tx;
    uint32_t oc_start = oc_tile * TILE_OC;

    uint32_t num_ic_tiles = (in_channels + TILE_IC - 1) / TILE_IC;

    // Output accumulators
    float acc[TILE_OC] = {0.0f};

    for (uint32_t ic_tile = 0; ic_tile < num_ic_tiles; ic_tile++) {
        uint32_t ic_start = ic_tile * TILE_IC;

        // Stage 1: Depthwise convolution
        for (uint32_t ic_local = 0; ic_local < TILE_IC; ic_local++) {
            uint32_t ic = ic_start + ic_local;
            if (ic >= in_channels) {
                depthwise_output[ic_local][ty][tx] = 0.0f;
                continue;
            }

            float sum = 0.0f;
            if (h_out < out_height && w_out < out_width) {
                for (uint32_t kh = 0; kh < kernel_h; kh++) {
                    for (uint32_t kw = 0; kw < kernel_w; kw++) {
                        int32_t ih = (int32_t)(h_out * stride_h) - (int32_t)pad_h + (int32_t)kh;
                        int32_t iw = (int32_t)(w_out * stride_w) - (int32_t)pad_w + (int32_t)kw;

                        if (ih >= 0 && ih < (int32_t)in_height &&
                            iw >= 0 && iw < (int32_t)in_width) {
                            float in_val = input[n * in_channels * in_height * in_width +
                                                ic * in_height * in_width +
                                                ih * in_width + iw];
                            float wt_val = depthwise_weight[ic * kernel_h * kernel_w +
                                                          kh * kernel_w + kw];
                            sum += in_val * wt_val;
                        }
                    }
                }
            }

            depthwise_output[ic_local][ty][tx] = sum;
        }
        __syncthreads();

        // Stage 2: Pointwise convolution (1x1)
        if (h_out < out_height && w_out < out_width) {
            for (uint32_t oc_local = 0; oc_local < TILE_OC; oc_local++) {
                uint32_t oc = oc_start + oc_local;
                if (oc >= out_channels) break;

                for (uint32_t ic_local = 0; ic_local < TILE_IC; ic_local++) {
                    uint32_t ic = ic_start + ic_local;
                    if (ic >= in_channels) break;

                    acc[oc_local] += depthwise_output[ic_local][ty][tx] *
                                    pointwise_weight[oc * in_channels + ic];
                }
            }
        }
        __syncthreads();
    }

    // Write output
    if (h_out < out_height && w_out < out_width) {
        for (uint32_t oc_local = 0; oc_local < TILE_OC; oc_local++) {
            uint32_t oc = oc_start + oc_local;
            if (oc >= out_channels) break;

            float result = acc[oc_local];
            if (bias != nullptr) {
                result += bias[oc];
            }

            output[n * out_channels * out_height * out_width +
                   oc * out_height * out_width +
                   h_out * out_width + w_out] = result;
        }
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_steel_conv2d_tiled(
    void* output,
    const void* input,
    const void* weight,
    const void* bias,
    uint32_t batch_size,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t in_height,
    uint32_t in_width,
    uint32_t kernel_h,
    uint32_t kernel_w,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t pad_h,
    uint32_t pad_w,
    uint32_t dilation_h,
    uint32_t dilation_w,
    cudaStream_t stream
) {
    uint32_t out_height = (in_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    uint32_t out_width = (in_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    uint32_t num_h_tiles = (out_height + TILE_OUT - 1) / TILE_OUT;
    uint32_t num_w_tiles = (out_width + TILE_OUT - 1) / TILE_OUT;
    uint32_t num_oc_tiles = (out_channels + TILE_OC - 1) / TILE_OC;

    dim3 blocks(num_h_tiles * num_w_tiles, num_oc_tiles, batch_size);
    uint32_t threads = BLOCK_SIZE;

    steel_conv2d_tiled_kernel<<<blocks, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        (const float*)weight,
        (const float*)bias,
        batch_size, in_channels, out_channels,
        in_height, in_width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w
    );

    return cudaGetLastError();
}

int lux_cuda_steel_conv2d_winograd(
    void* output,
    const void* input,
    const void* weight,
    const void* bias,
    uint32_t batch_size,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t in_height,
    uint32_t in_width,
    uint32_t pad_h,
    uint32_t pad_w,
    cudaStream_t stream
) {
    uint32_t out_height = in_height + 2 * pad_h - 2;
    uint32_t out_width = in_width + 2 * pad_w - 2;

    uint32_t num_h_tiles = (out_height + 1) / 2;
    uint32_t num_w_tiles = (out_width + 1) / 2;
    uint32_t num_oc_tiles = (out_channels + TILE_OC - 1) / TILE_OC;

    dim3 blocks(num_h_tiles * num_w_tiles, num_oc_tiles, batch_size);
    uint32_t threads = BLOCK_SIZE;

    steel_conv2d_winograd_kernel<<<blocks, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        (const float*)weight,
        (const float*)bias,
        batch_size, in_channels, out_channels,
        in_height, in_width,
        pad_h, pad_w
    );

    return cudaGetLastError();
}

int lux_cuda_steel_conv2d_separable(
    void* output,
    const void* input,
    const void* depthwise_weight,
    const void* pointwise_weight,
    const void* bias,
    uint32_t batch_size,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t in_height,
    uint32_t in_width,
    uint32_t kernel_h,
    uint32_t kernel_w,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t pad_h,
    uint32_t pad_w,
    cudaStream_t stream
) {
    uint32_t out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
    uint32_t out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;

    uint32_t num_h_tiles = (out_height + TILE_OUT - 1) / TILE_OUT;
    uint32_t num_w_tiles = (out_width + TILE_OUT - 1) / TILE_OUT;
    uint32_t num_oc_tiles = (out_channels + TILE_OC - 1) / TILE_OC;

    dim3 blocks(num_h_tiles * num_w_tiles, num_oc_tiles, batch_size);
    uint32_t threads = BLOCK_SIZE;

    steel_conv2d_separable_kernel<<<blocks, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        (const float*)depthwise_weight,
        (const float*)pointwise_weight,
        (const float*)bias,
        batch_size, in_channels, out_channels,
        in_height, in_width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w
    );

    return cudaGetLastError();
}

}  // extern "C"
