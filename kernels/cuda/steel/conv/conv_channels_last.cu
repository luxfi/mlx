// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Steel Convolution CUDA Kernels - NHWC (Channels Last) Layout
// Optimized for memory coalescing with channels-last tensor format

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// Configuration
// ============================================================================

#define TILE_OUT_H 4
#define TILE_OUT_W 4
#define TILE_OC 32
#define BLOCK_SIZE 256

// ============================================================================
// NHWC Convolution (Channels Last)
// ============================================================================

// Input: [N, H_in, W_in, C_in] - TensorFlow/Keras default format
// Weight: [K_h, K_w, C_in, C_out] or [C_out, K_h, K_w, C_in]
// Output: [N, H_out, W_out, C_out]

extern "C" __global__
void steel_conv2d_nhwc_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,      // [K_h, K_w, C_in, C_out]
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
    __shared__ float weight_tile[7 * 7 * 16 * 32];  // [K_h * K_w * TILE_IC * TILE_OC]

    uint32_t out_height = (in_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    uint32_t out_width = (in_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    uint32_t n = blockIdx.z;
    uint32_t oc_tile = blockIdx.y;
    uint32_t spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t h_out = spatial_idx / out_width;
    uint32_t w_out = spatial_idx % out_width;

    if (n >= batch_size || h_out >= out_height || w_out >= out_width) return;

    uint32_t oc_start = oc_tile * TILE_OC;
    uint32_t tid = threadIdx.x;

    // Load weight tile cooperatively
    uint32_t weight_size = kernel_h * kernel_w * in_channels * min(TILE_OC, out_channels - oc_start);
    for (uint32_t i = tid; i < weight_size; i += blockDim.x) {
        uint32_t kh_kw_ic = i / TILE_OC;
        uint32_t oc_local = i % TILE_OC;

        uint32_t kh_kw = kh_kw_ic / in_channels;
        uint32_t ic = kh_kw_ic % in_channels;
        uint32_t kh = kh_kw / kernel_w;
        uint32_t kw = kh_kw % kernel_w;

        uint32_t oc = oc_start + oc_local;
        if (oc < out_channels) {
            // Weight layout: [K_h, K_w, C_in, C_out]
            weight_tile[i] = weight[((kh * kernel_w + kw) * in_channels + ic) * out_channels + oc];
        }
    }
    __syncthreads();

    // Compute convolution for this output position
    float acc[32];  // Max TILE_OC
    for (int i = 0; i < TILE_OC; i++) {
        acc[i] = 0.0f;
    }

    for (uint32_t kh = 0; kh < kernel_h; kh++) {
        for (uint32_t kw = 0; kw < kernel_w; kw++) {
            int32_t ih = (int32_t)(h_out * stride_h + kh * dilation_h) - (int32_t)pad_h;
            int32_t iw = (int32_t)(w_out * stride_w + kw * dilation_w) - (int32_t)pad_w;

            if (ih >= 0 && ih < (int32_t)in_height &&
                iw >= 0 && iw < (int32_t)in_width) {

                for (uint32_t ic = 0; ic < in_channels; ic++) {
                    // NHWC layout: input[n, h, w, c]
                    float in_val = input[((n * in_height + ih) * in_width + iw) * in_channels + ic];

                    uint32_t w_base = (kh * kernel_w + kw) * in_channels * TILE_OC + ic * TILE_OC;

                    #pragma unroll 4
                    for (uint32_t oc_local = 0; oc_local < TILE_OC; oc_local++) {
                        if (oc_start + oc_local < out_channels) {
                            acc[oc_local] += in_val * weight_tile[w_base + oc_local];
                        }
                    }
                }
            }
        }
    }

    // Write output with bias (NHWC layout)
    for (uint32_t oc_local = 0; oc_local < TILE_OC; oc_local++) {
        uint32_t oc = oc_start + oc_local;
        if (oc >= out_channels) break;

        float result = acc[oc_local];
        if (bias != nullptr) {
            result += bias[oc];
        }

        // NHWC layout: output[n, h, w, c]
        output[((n * out_height + h_out) * out_width + w_out) * out_channels + oc] = result;
    }
}

// ============================================================================
// NHWC 1x1 Convolution (Pointwise)
// ============================================================================

extern "C" __global__
void steel_conv2d_nhwc_1x1_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,      // [C_in, C_out]
    const float* __restrict__ bias,
    uint32_t batch_size,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t height,
    uint32_t width
) {
    __shared__ float weight_tile[256];  // 16 * 16

    uint32_t spatial_size = height * width;
    uint32_t total_elements = batch_size * spatial_size;

    uint32_t oc_tile = blockIdx.y;
    uint32_t spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t tid = threadIdx.x;

    if (spatial_idx >= total_elements) return;

    uint32_t n = spatial_idx / spatial_size;
    uint32_t hw = spatial_idx % spatial_size;

    uint32_t oc_start = oc_tile * 16;
    uint32_t num_oc = min(16u, out_channels - oc_start);

    // Process in chunks of input channels
    float acc[16] = {0.0f};

    for (uint32_t ic_chunk = 0; ic_chunk < in_channels; ic_chunk += 16) {
        uint32_t chunk_size = min(16u, in_channels - ic_chunk);

        // Load weight tile
        for (uint32_t i = tid; i < chunk_size * num_oc; i += blockDim.x) {
            uint32_t ic_local = i / num_oc;
            uint32_t oc_local = i % num_oc;
            weight_tile[ic_local * 16 + oc_local] =
                weight[(ic_chunk + ic_local) * out_channels + oc_start + oc_local];
        }
        __syncthreads();

        // Compute
        for (uint32_t ic_local = 0; ic_local < chunk_size; ic_local++) {
            float in_val = input[(n * spatial_size + hw) * in_channels + ic_chunk + ic_local];

            #pragma unroll
            for (uint32_t oc_local = 0; oc_local < 16; oc_local++) {
                if (oc_local < num_oc) {
                    acc[oc_local] += in_val * weight_tile[ic_local * 16 + oc_local];
                }
            }
        }
        __syncthreads();
    }

    // Write output
    for (uint32_t oc_local = 0; oc_local < num_oc; oc_local++) {
        float result = acc[oc_local];
        if (bias != nullptr) {
            result += bias[oc_start + oc_local];
        }
        output[(n * spatial_size + hw) * out_channels + oc_start + oc_local] = result;
    }
}

// ============================================================================
// NHWC Grouped Convolution
// ============================================================================

extern "C" __global__
void steel_conv2d_nhwc_grouped_kernel(
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
    uint32_t groups
) {
    uint32_t out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
    uint32_t out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;

    uint32_t in_channels_per_group = in_channels / groups;
    uint32_t out_channels_per_group = out_channels / groups;

    uint32_t n = blockIdx.z / groups;
    uint32_t g = blockIdx.z % groups;
    uint32_t spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t oc_local = blockIdx.y;

    if (n >= batch_size || spatial_idx >= out_height * out_width ||
        oc_local >= out_channels_per_group) return;

    uint32_t h_out = spatial_idx / out_width;
    uint32_t w_out = spatial_idx % out_width;
    uint32_t oc = g * out_channels_per_group + oc_local;

    float sum = 0.0f;

    for (uint32_t ic_local = 0; ic_local < in_channels_per_group; ic_local++) {
        uint32_t ic = g * in_channels_per_group + ic_local;

        for (uint32_t kh = 0; kh < kernel_h; kh++) {
            for (uint32_t kw = 0; kw < kernel_w; kw++) {
                int32_t ih = (int32_t)(h_out * stride_h + kh) - (int32_t)pad_h;
                int32_t iw = (int32_t)(w_out * stride_w + kw) - (int32_t)pad_w;

                if (ih >= 0 && ih < (int32_t)in_height &&
                    iw >= 0 && iw < (int32_t)in_width) {
                    float in_val = input[((n * in_height + ih) * in_width + iw) * in_channels + ic];
                    // Weight layout for grouped: [K_h, K_w, C_in/groups, C_out]
                    float w_val = weight[((kh * kernel_w + kw) * in_channels_per_group + ic_local) *
                                        out_channels + oc];
                    sum += in_val * w_val;
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[oc];
    }

    output[((n * out_height + h_out) * out_width + w_out) * out_channels + oc] = sum;
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_steel_conv2d_nhwc(
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

    uint32_t threads = 256;
    uint32_t spatial_blocks = (out_height * out_width + threads - 1) / threads;
    uint32_t oc_tiles = (out_channels + TILE_OC - 1) / TILE_OC;

    dim3 blocks(spatial_blocks, oc_tiles, batch_size);

    steel_conv2d_nhwc_kernel<<<blocks, threads, 0, stream>>>(
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

int lux_cuda_steel_conv2d_nhwc_1x1(
    void* output,
    const void* input,
    const void* weight,
    const void* bias,
    uint32_t batch_size,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t height,
    uint32_t width,
    cudaStream_t stream
) {
    uint32_t threads = 256;
    uint32_t spatial_blocks = (batch_size * height * width + threads - 1) / threads;
    uint32_t oc_tiles = (out_channels + 15) / 16;

    dim3 blocks(spatial_blocks, oc_tiles, 1);

    steel_conv2d_nhwc_1x1_kernel<<<blocks, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        (const float*)weight,
        (const float*)bias,
        batch_size, in_channels, out_channels, height, width
    );

    return cudaGetLastError();
}

int lux_cuda_steel_conv2d_nhwc_grouped(
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
    uint32_t groups,
    cudaStream_t stream
) {
    uint32_t out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
    uint32_t out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;
    uint32_t out_channels_per_group = out_channels / groups;

    uint32_t threads = 256;
    uint32_t spatial_blocks = (out_height * out_width + threads - 1) / threads;

    dim3 blocks(spatial_blocks, out_channels_per_group, batch_size * groups);

    steel_conv2d_nhwc_grouped_kernel<<<blocks, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        (const float*)weight,
        (const float*)bias,
        batch_size, in_channels, out_channels,
        in_height, in_width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        groups
    );

    return cudaGetLastError();
}

}  // extern "C"
