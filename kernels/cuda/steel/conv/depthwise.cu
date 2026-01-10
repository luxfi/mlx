// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Depthwise Convolution CUDA Kernels
// Efficient implementation for MobileNet-style depthwise separable convolutions

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// Configuration
// ============================================================================

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define TILE_SIZE 16

// ============================================================================
// Depthwise Conv2D (NCHW Layout)
// ============================================================================

// Each input channel is convolved independently with its own filter
// Weight shape: [C, 1, K_h, K_w] where C_out = C_in

extern "C" __global__
void steel_depthwise_conv2d_nchw_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    uint32_t batch_size,
    uint32_t channels,
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
    uint32_t out_height = (in_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    uint32_t out_width = (in_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    uint32_t n = blockIdx.z / channels;
    uint32_t c = blockIdx.z % channels;
    uint32_t out_spatial = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= batch_size || out_spatial >= out_height * out_width) return;

    uint32_t h_out = out_spatial / out_width;
    uint32_t w_out = out_spatial % out_width;

    float sum = 0.0f;

    // Load weight for this channel (small, keep in registers)
    for (uint32_t kh = 0; kh < kernel_h; kh++) {
        for (uint32_t kw = 0; kw < kernel_w; kw++) {
            int32_t ih = (int32_t)(h_out * stride_h + kh * dilation_h) - (int32_t)pad_h;
            int32_t iw = (int32_t)(w_out * stride_w + kw * dilation_w) - (int32_t)pad_w;

            if (ih >= 0 && ih < (int32_t)in_height &&
                iw >= 0 && iw < (int32_t)in_width) {
                // NCHW layout
                float in_val = input[((n * channels + c) * in_height + ih) * in_width + iw];
                float w_val = weight[(c * kernel_h + kh) * kernel_w + kw];
                sum += in_val * w_val;
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[c];
    }

    output[((n * channels + c) * out_height + h_out) * out_width + w_out] = sum;
}

// ============================================================================
// Depthwise Conv2D (NHWC Layout)
// ============================================================================

extern "C" __global__
void steel_depthwise_conv2d_nhwc_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,      // [K_h, K_w, C]
    const float* __restrict__ bias,
    uint32_t batch_size,
    uint32_t channels,
    uint32_t in_height,
    uint32_t in_width,
    uint32_t kernel_h,
    uint32_t kernel_w,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t pad_h,
    uint32_t pad_w
) {
    __shared__ float weight_smem[9 * 128];  // Up to 3x3 kernel, 128 channels

    uint32_t out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
    uint32_t out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;

    uint32_t n = blockIdx.z;
    uint32_t spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t c_tile = blockIdx.y;
    uint32_t tid = threadIdx.x;

    if (n >= batch_size || spatial_idx >= out_height * out_width) return;

    uint32_t h_out = spatial_idx / out_width;
    uint32_t w_out = spatial_idx % out_width;

    uint32_t c_start = c_tile * 128;
    uint32_t c_count = min(128u, channels - c_start);

    // Load weight tile to shared memory
    uint32_t weight_size = kernel_h * kernel_w * c_count;
    for (uint32_t i = tid; i < weight_size; i += blockDim.x) {
        uint32_t k_idx = i / c_count;
        uint32_t c_local = i % c_count;
        weight_smem[k_idx * 128 + c_local] = weight[k_idx * channels + c_start + c_local];
    }
    __syncthreads();

    // Compute depthwise conv for this position, all channels in tile
    for (uint32_t c_local = 0; c_local < c_count; c_local++) {
        uint32_t c = c_start + c_local;
        float sum = 0.0f;

        for (uint32_t kh = 0; kh < kernel_h; kh++) {
            for (uint32_t kw = 0; kw < kernel_w; kw++) {
                int32_t ih = (int32_t)(h_out * stride_h + kh) - (int32_t)pad_h;
                int32_t iw = (int32_t)(w_out * stride_w + kw) - (int32_t)pad_w;

                if (ih >= 0 && ih < (int32_t)in_height &&
                    iw >= 0 && iw < (int32_t)in_width) {
                    // NHWC layout
                    float in_val = input[((n * in_height + ih) * in_width + iw) * channels + c];
                    float w_val = weight_smem[(kh * kernel_w + kw) * 128 + c_local];
                    sum += in_val * w_val;
                }
            }
        }

        if (bias != nullptr) {
            sum += bias[c];
        }

        output[((n * out_height + h_out) * out_width + w_out) * channels + c] = sum;
    }
}

// ============================================================================
// Depthwise Conv2D with Channel Multiplier
// ============================================================================

// Expands channels: C_out = C_in * multiplier
extern "C" __global__
void steel_depthwise_conv2d_multiplier_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,      // [C_in, multiplier, K_h, K_w]
    const float* __restrict__ bias,        // [C_in * multiplier]
    uint32_t batch_size,
    uint32_t in_channels,
    uint32_t multiplier,
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
    uint32_t out_channels = in_channels * multiplier;

    uint32_t n = blockIdx.z / out_channels;
    uint32_t oc = blockIdx.z % out_channels;
    uint32_t out_spatial = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= batch_size || out_spatial >= out_height * out_width) return;

    uint32_t h_out = out_spatial / out_width;
    uint32_t w_out = out_spatial % out_width;

    // Map output channel to input channel
    uint32_t ic = oc / multiplier;
    uint32_t m = oc % multiplier;

    float sum = 0.0f;

    for (uint32_t kh = 0; kh < kernel_h; kh++) {
        for (uint32_t kw = 0; kw < kernel_w; kw++) {
            int32_t ih = (int32_t)(h_out * stride_h + kh) - (int32_t)pad_h;
            int32_t iw = (int32_t)(w_out * stride_w + kw) - (int32_t)pad_w;

            if (ih >= 0 && ih < (int32_t)in_height &&
                iw >= 0 && iw < (int32_t)in_width) {
                float in_val = input[((n * in_channels + ic) * in_height + ih) * in_width + iw];
                float w_val = weight[((ic * multiplier + m) * kernel_h + kh) * kernel_w + kw];
                sum += in_val * w_val;
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[oc];
    }

    output[((n * out_channels + oc) * out_height + h_out) * out_width + w_out] = sum;
}

// ============================================================================
// Fused Depthwise + Pointwise (Separable Convolution)
// ============================================================================

extern "C" __global__
void steel_separable_conv2d_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ depthwise_weight,  // [C_in, 1, K_h, K_w]
    const float* __restrict__ pointwise_weight,  // [C_out, C_in]
    const float* __restrict__ bias,              // [C_out]
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
    __shared__ float depthwise_out[16 * 16 * 32];  // [TILE_H * TILE_W * TILE_C]

    uint32_t out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
    uint32_t out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;

    uint32_t n = blockIdx.z;
    uint32_t oc_tile = blockIdx.y;
    uint32_t spatial_tile = blockIdx.x;
    uint32_t tid = threadIdx.x;

    uint32_t tiles_w = (out_width + 15) / 16;
    uint32_t h_tile = (spatial_tile / tiles_w) * 16;
    uint32_t w_tile = (spatial_tile % tiles_w) * 16;

    if (n >= batch_size) return;

    uint32_t ty = tid / 16;
    uint32_t tx = tid % 16;
    uint32_t h_out = h_tile + ty;
    uint32_t w_out = w_tile + tx;
    uint32_t oc_start = oc_tile * 32;

    // Stage 1: Depthwise convolution (store intermediate results)
    uint32_t num_ic_tiles = (in_channels + 31) / 32;

    float acc[32] = {0.0f};  // For pointwise accumulation

    for (uint32_t ic_tile = 0; ic_tile < num_ic_tiles; ic_tile++) {
        uint32_t ic_start = ic_tile * 32;
        uint32_t ic_count = min(32u, in_channels - ic_start);

        // Compute depthwise for this channel tile
        for (uint32_t ic_local = 0; ic_local < ic_count; ic_local++) {
            uint32_t ic = ic_start + ic_local;
            float dw_sum = 0.0f;

            if (h_out < out_height && w_out < out_width) {
                for (uint32_t kh = 0; kh < kernel_h; kh++) {
                    for (uint32_t kw = 0; kw < kernel_w; kw++) {
                        int32_t ih = (int32_t)(h_out * stride_h + kh) - (int32_t)pad_h;
                        int32_t iw = (int32_t)(w_out * stride_w + kw) - (int32_t)pad_w;

                        if (ih >= 0 && ih < (int32_t)in_height &&
                            iw >= 0 && iw < (int32_t)in_width) {
                            float in_val = input[((n * in_channels + ic) * in_height + ih) *
                                                in_width + iw];
                            float w_val = depthwise_weight[(ic * kernel_h + kh) * kernel_w + kw];
                            dw_sum += in_val * w_val;
                        }
                    }
                }
            }

            // Store for pointwise
            depthwise_out[(ty * 16 + tx) * 32 + ic_local] = dw_sum;
        }
        __syncthreads();

        // Stage 2: Pointwise convolution accumulation
        if (h_out < out_height && w_out < out_width) {
            for (uint32_t oc_local = 0; oc_local < 32 && oc_start + oc_local < out_channels; oc_local++) {
                uint32_t oc = oc_start + oc_local;

                for (uint32_t ic_local = 0; ic_local < ic_count; ic_local++) {
                    uint32_t ic = ic_start + ic_local;
                    float dw_val = depthwise_out[(ty * 16 + tx) * 32 + ic_local];
                    float pw_val = pointwise_weight[oc * in_channels + ic];
                    acc[oc_local] += dw_val * pw_val;
                }
            }
        }
        __syncthreads();
    }

    // Write output
    if (h_out < out_height && w_out < out_width) {
        for (uint32_t oc_local = 0; oc_local < 32 && oc_start + oc_local < out_channels; oc_local++) {
            uint32_t oc = oc_start + oc_local;
            float result = acc[oc_local];

            if (bias != nullptr) {
                result += bias[oc];
            }

            output[((n * out_channels + oc) * out_height + h_out) * out_width + w_out] = result;
        }
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_steel_depthwise_conv2d_nchw(
    void* output,
    const void* input,
    const void* weight,
    const void* bias,
    uint32_t batch_size,
    uint32_t channels,
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
    uint32_t out_spatial = out_height * out_width;

    uint32_t threads = BLOCK_SIZE;
    uint32_t spatial_blocks = (out_spatial + threads - 1) / threads;

    dim3 blocks(spatial_blocks, 1, batch_size * channels);

    steel_depthwise_conv2d_nchw_kernel<<<blocks, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        (const float*)weight,
        (const float*)bias,
        batch_size, channels,
        in_height, in_width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w
    );

    return cudaGetLastError();
}

int lux_cuda_steel_depthwise_conv2d_nhwc(
    void* output,
    const void* input,
    const void* weight,
    const void* bias,
    uint32_t batch_size,
    uint32_t channels,
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

    uint32_t threads = BLOCK_SIZE;
    uint32_t spatial_blocks = (out_spatial + threads - 1) / threads;
    uint32_t c_tiles = (channels + 127) / 128;

    dim3 blocks(spatial_blocks, c_tiles, batch_size);

    steel_depthwise_conv2d_nhwc_kernel<<<blocks, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        (const float*)weight,
        (const float*)bias,
        batch_size, channels,
        in_height, in_width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w
    );

    return cudaGetLastError();
}

int lux_cuda_steel_depthwise_conv2d_multiplier(
    void* output,
    const void* input,
    const void* weight,
    const void* bias,
    uint32_t batch_size,
    uint32_t in_channels,
    uint32_t multiplier,
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
    uint32_t out_channels = in_channels * multiplier;

    uint32_t threads = BLOCK_SIZE;
    uint32_t spatial_blocks = (out_spatial + threads - 1) / threads;

    dim3 blocks(spatial_blocks, 1, batch_size * out_channels);

    steel_depthwise_conv2d_multiplier_kernel<<<blocks, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        (const float*)weight,
        (const float*)bias,
        batch_size, in_channels, multiplier,
        in_height, in_width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w
    );

    return cudaGetLastError();
}

int lux_cuda_steel_separable_conv2d(
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

    uint32_t tiles_h = (out_height + 15) / 16;
    uint32_t tiles_w = (out_width + 15) / 16;
    uint32_t oc_tiles = (out_channels + 31) / 32;

    dim3 blocks(tiles_h * tiles_w, oc_tiles, batch_size);
    uint32_t threads = 256;

    steel_separable_conv2d_kernel<<<blocks, threads, 0, stream>>>(
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
