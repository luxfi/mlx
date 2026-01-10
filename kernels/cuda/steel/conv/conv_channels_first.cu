// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Steel Convolution CUDA Kernels - NCHW (Channels First) Layout
// Optimized for memory coalescing with channels-first tensor format

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// Configuration
// ============================================================================

#define TILE_OUT_H 8
#define TILE_OUT_W 8
#define TILE_IC 16
#define TILE_OC 16
#define BLOCK_SIZE 256

// ============================================================================
// NCHW Convolution (Channels First)
// ============================================================================

// Input: [N, C_in, H_in, W_in] - standard PyTorch format
// Weight: [C_out, C_in, K_h, K_w]
// Output: [N, C_out, H_out, W_out]

extern "C" __global__
void steel_conv2d_nchw_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
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
    uint32_t pad_w,
    uint32_t dilation_h,
    uint32_t dilation_w
) {
    __shared__ float input_tile[TILE_IC][TILE_OUT_H + 6][TILE_OUT_W + 6];  // +6 for 7x7 max kernel
    __shared__ float weight_tile[TILE_OC][TILE_IC][7][7];

    uint32_t out_height = (in_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    uint32_t out_width = (in_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    uint32_t n = blockIdx.z;
    uint32_t oc_tile = blockIdx.y;
    uint32_t spatial_tile = blockIdx.x;

    uint32_t tiles_w = (out_width + TILE_OUT_W - 1) / TILE_OUT_W;
    uint32_t h_tile = (spatial_tile / tiles_w) * TILE_OUT_H;
    uint32_t w_tile = (spatial_tile % tiles_w) * TILE_OUT_W;

    uint32_t tid = threadIdx.x;
    uint32_t ty = tid / TILE_OUT_W;
    uint32_t tx = tid % TILE_OUT_W;

    if (n >= batch_size) return;

    uint32_t h_out = h_tile + ty;
    uint32_t w_out = w_tile + tx;
    uint32_t oc_start = oc_tile * TILE_OC;

    // Output accumulators
    float acc[TILE_OC];
    #pragma unroll
    for (int i = 0; i < TILE_OC; i++) {
        acc[i] = 0.0f;
    }

    // Number of input channel tiles
    uint32_t num_ic_tiles = (in_channels + TILE_IC - 1) / TILE_IC;

    for (uint32_t ic_tile = 0; ic_tile < num_ic_tiles; ic_tile++) {
        uint32_t ic_start = ic_tile * TILE_IC;

        // Load input tile with halo
        uint32_t in_tile_h = TILE_OUT_H * stride_h + (kernel_h - 1) * dilation_h;
        uint32_t in_tile_w = TILE_OUT_W * stride_w + (kernel_w - 1) * dilation_w;

        for (uint32_t i = tid; i < TILE_IC * in_tile_h * in_tile_w; i += blockDim.x) {
            uint32_t ic_local = i / (in_tile_h * in_tile_w);
            uint32_t spatial = i % (in_tile_h * in_tile_w);
            uint32_t ih_local = spatial / in_tile_w;
            uint32_t iw_local = spatial % in_tile_w;

            uint32_t ic = ic_start + ic_local;
            int32_t ih = (int32_t)(h_tile * stride_h) - (int32_t)pad_h + (int32_t)ih_local;
            int32_t iw = (int32_t)(w_tile * stride_w) - (int32_t)pad_w + (int32_t)iw_local;

            float val = 0.0f;
            if (ic < in_channels && ih >= 0 && ih < (int32_t)in_height &&
                iw >= 0 && iw < (int32_t)in_width) {
                // NCHW layout: input[n, c, h, w]
                val = input[((n * in_channels + ic) * in_height + ih) * in_width + iw];
            }

            if (ic_local < TILE_IC && ih_local < in_tile_h && iw_local < in_tile_w) {
                input_tile[ic_local][ih_local][iw_local] = val;
            }
        }

        // Load weight tile
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
            if (oc < out_channels && ic < in_channels && kh < kernel_h && kw < kernel_w) {
                // Weight layout: [C_out, C_in, K_h, K_w]
                val = weight[((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw];
            }

            if (oc_local < TILE_OC && ic_local < TILE_IC) {
                weight_tile[oc_local][ic_local][kh][kw] = val;
            }
        }
        __syncthreads();

        // Compute convolution
        if (h_out < out_height && w_out < out_width) {
            for (uint32_t oc_local = 0; oc_local < TILE_OC; oc_local++) {
                uint32_t oc = oc_start + oc_local;
                if (oc >= out_channels) break;

                for (uint32_t ic_local = 0; ic_local < TILE_IC; ic_local++) {
                    uint32_t ic = ic_start + ic_local;
                    if (ic >= in_channels) break;

                    for (uint32_t kh = 0; kh < kernel_h; kh++) {
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

    // Write output with bias (NCHW layout)
    if (h_out < out_height && w_out < out_width) {
        for (uint32_t oc_local = 0; oc_local < TILE_OC; oc_local++) {
            uint32_t oc = oc_start + oc_local;
            if (oc >= out_channels) break;

            float result = acc[oc_local];
            if (bias != nullptr) {
                result += bias[oc];
            }

            // NCHW layout: output[n, c, h, w]
            output[((n * out_channels + oc) * out_height + h_out) * out_width + w_out] = result;
        }
    }
}

// ============================================================================
// NCHW Convolution with ReLU Fusion
// ============================================================================

extern "C" __global__
void steel_conv2d_nchw_relu_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
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
    uint32_t out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
    uint32_t out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;

    uint32_t n = blockIdx.z / out_channels;
    uint32_t oc = blockIdx.z % out_channels;
    uint32_t out_spatial = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= batch_size || out_spatial >= out_height * out_width) return;

    uint32_t h_out = out_spatial / out_width;
    uint32_t w_out = out_spatial % out_width;

    float sum = 0.0f;

    for (uint32_t ic = 0; ic < in_channels; ic++) {
        for (uint32_t kh = 0; kh < kernel_h; kh++) {
            for (uint32_t kw = 0; kw < kernel_w; kw++) {
                int32_t ih = (int32_t)(h_out * stride_h + kh) - (int32_t)pad_h;
                int32_t iw = (int32_t)(w_out * stride_w + kw) - (int32_t)pad_w;

                if (ih >= 0 && ih < (int32_t)in_height &&
                    iw >= 0 && iw < (int32_t)in_width) {
                    float in_val = input[((n * in_channels + ic) * in_height + ih) * in_width + iw];
                    float w_val = weight[((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw];
                    sum += in_val * w_val;
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[oc];
    }

    // ReLU activation
    sum = fmaxf(0.0f, sum);

    output[((n * out_channels + oc) * out_height + h_out) * out_width + w_out] = sum;
}

// ============================================================================
// NCHW Transposed Convolution (Deconvolution)
// ============================================================================

extern "C" __global__
void steel_conv_transpose2d_nchw_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
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
    uint32_t pad_w,
    uint32_t output_pad_h,
    uint32_t output_pad_w
) {
    uint32_t out_height = (in_height - 1) * stride_h - 2 * pad_h + kernel_h + output_pad_h;
    uint32_t out_width = (in_width - 1) * stride_w - 2 * pad_w + kernel_w + output_pad_w;

    uint32_t n = blockIdx.z / out_channels;
    uint32_t oc = blockIdx.z % out_channels;
    uint32_t out_spatial = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= batch_size || out_spatial >= out_height * out_width) return;

    uint32_t h_out = out_spatial / out_width;
    uint32_t w_out = out_spatial % out_width;

    float sum = 0.0f;

    // Transposed convolution: for each output position, find contributing inputs
    for (uint32_t ic = 0; ic < in_channels; ic++) {
        for (uint32_t kh = 0; kh < kernel_h; kh++) {
            for (uint32_t kw = 0; kw < kernel_w; kw++) {
                // Calculate corresponding input position
                int32_t h_in_offset = (int32_t)h_out + (int32_t)pad_h - (int32_t)kh;
                int32_t w_in_offset = (int32_t)w_out + (int32_t)pad_w - (int32_t)kw;

                // Check if this input position contributes
                if (h_in_offset >= 0 && h_in_offset % stride_h == 0 &&
                    w_in_offset >= 0 && w_in_offset % stride_w == 0) {

                    int32_t ih = h_in_offset / stride_h;
                    int32_t iw = w_in_offset / stride_w;

                    if (ih < (int32_t)in_height && iw < (int32_t)in_width) {
                        float in_val = input[((n * in_channels + ic) * in_height + ih) * in_width + iw];
                        // Transposed weight: [C_in, C_out, K_h, K_w]
                        float w_val = weight[((ic * out_channels + oc) * kernel_h + kh) * kernel_w + kw];
                        sum += in_val * w_val;
                    }
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[oc];
    }

    output[((n * out_channels + oc) * out_height + h_out) * out_width + w_out] = sum;
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_steel_conv2d_nchw(
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

    uint32_t num_h_tiles = (out_height + TILE_OUT_H - 1) / TILE_OUT_H;
    uint32_t num_w_tiles = (out_width + TILE_OUT_W - 1) / TILE_OUT_W;
    uint32_t num_oc_tiles = (out_channels + TILE_OC - 1) / TILE_OC;

    dim3 blocks(num_h_tiles * num_w_tiles, num_oc_tiles, batch_size);
    uint32_t threads = BLOCK_SIZE;

    steel_conv2d_nchw_kernel<<<blocks, threads, 0, stream>>>(
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

int lux_cuda_steel_conv2d_nchw_relu(
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
    cudaStream_t stream
) {
    uint32_t out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
    uint32_t out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;
    uint32_t out_spatial = out_height * out_width;

    uint32_t threads = 256;
    uint32_t blocks_x = (out_spatial + threads - 1) / threads;

    dim3 blocks(blocks_x, 1, batch_size * out_channels);

    steel_conv2d_nchw_relu_kernel<<<blocks, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        (const float*)weight,
        (const float*)bias,
        batch_size, in_channels, out_channels,
        in_height, in_width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w
    );

    return cudaGetLastError();
}

int lux_cuda_steel_conv_transpose2d_nchw(
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
    uint32_t output_pad_h,
    uint32_t output_pad_w,
    cudaStream_t stream
) {
    uint32_t out_height = (in_height - 1) * stride_h - 2 * pad_h + kernel_h + output_pad_h;
    uint32_t out_width = (in_width - 1) * stride_w - 2 * pad_w + kernel_w + output_pad_w;
    uint32_t out_spatial = out_height * out_width;

    uint32_t threads = 256;
    uint32_t blocks_x = (out_spatial + threads - 1) / threads;

    dim3 blocks(blocks_x, 1, batch_size * out_channels);

    steel_conv_transpose2d_nchw_kernel<<<blocks, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        (const float*)weight,
        (const float*)bias,
        batch_size, in_channels, out_channels,
        in_height, in_width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        output_pad_h, output_pad_w
    );

    return cudaGetLastError();
}

}  // extern "C"
