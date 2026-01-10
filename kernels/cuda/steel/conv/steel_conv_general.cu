// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Steel General Convolution CUDA Kernels
// Supports strided, grouped, dilated, and transposed convolutions

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// General Convolution Configuration
// ============================================================================

#define TILE_SIZE 16
#define BLOCK_SIZE 256

// ============================================================================
// General Strided/Dilated Conv2D
// ============================================================================

// Supports arbitrary stride, dilation, padding, and groups
extern "C" __global__
void steel_conv2d_general_kernel(
    float* __restrict__ output,          // [N, C_out, H_out, W_out]
    const float* __restrict__ input,     // [N, C_in, H_in, W_in]
    const float* __restrict__ weight,    // [C_out, C_in/groups, K_h, K_w]
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
    uint32_t dilation_w,
    uint32_t groups
) {
    uint32_t out_height = (in_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    uint32_t out_width = (in_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    uint32_t n = blockIdx.z / out_channels;
    uint32_t c_out = blockIdx.z % out_channels;
    uint32_t h_out = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t w_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= batch_size || h_out >= out_height || w_out >= out_width) return;

    // Group parameters
    uint32_t c_in_per_group = in_channels / groups;
    uint32_t c_out_per_group = out_channels / groups;
    uint32_t group = c_out / c_out_per_group;
    uint32_t c_in_start = group * c_in_per_group;

    float sum = 0.0f;

    // Convolution
    for (uint32_t c_in_local = 0; c_in_local < c_in_per_group; c_in_local++) {
        uint32_t c_in = c_in_start + c_in_local;

        for (uint32_t kh = 0; kh < kernel_h; kh++) {
            for (uint32_t kw = 0; kw < kernel_w; kw++) {
                int32_t h_in = (int32_t)(h_out * stride_h) - (int32_t)pad_h + (int32_t)(kh * dilation_h);
                int32_t w_in = (int32_t)(w_out * stride_w) - (int32_t)pad_w + (int32_t)(kw * dilation_w);

                if (h_in >= 0 && h_in < (int32_t)in_height &&
                    w_in >= 0 && w_in < (int32_t)in_width) {

                    float in_val = input[n * in_channels * in_height * in_width +
                                        c_in * in_height * in_width +
                                        h_in * in_width + w_in];

                    float wt_val = weight[c_out * c_in_per_group * kernel_h * kernel_w +
                                         c_in_local * kernel_h * kernel_w +
                                         kh * kernel_w + kw];

                    sum += in_val * wt_val;
                }
            }
        }
    }

    // Add bias
    if (bias != nullptr) {
        sum += bias[c_out];
    }

    output[n * out_channels * out_height * out_width +
           c_out * out_height * out_width +
           h_out * out_width + w_out] = sum;
}

// ============================================================================
// Transposed Convolution (Deconvolution)
// ============================================================================

extern "C" __global__
void steel_conv2d_transpose_kernel(
    float* __restrict__ output,          // [N, C_out, H_out, W_out]
    const float* __restrict__ input,     // [N, C_in, H_in, W_in]
    const float* __restrict__ weight,    // [C_in, C_out, K_h, K_w]
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
    uint32_t output_pad_h,
    uint32_t output_pad_w,
    uint32_t dilation_h,
    uint32_t dilation_w,
    uint32_t groups
) {
    uint32_t out_height = (in_height - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + output_pad_h + 1;
    uint32_t out_width = (in_width - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + output_pad_w + 1;

    uint32_t n = blockIdx.z / out_channels;
    uint32_t c_out = blockIdx.z % out_channels;
    uint32_t h_out = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t w_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= batch_size || h_out >= out_height || w_out >= out_width) return;

    // Group parameters
    uint32_t c_in_per_group = in_channels / groups;
    uint32_t c_out_per_group = out_channels / groups;
    uint32_t group = c_out / c_out_per_group;
    uint32_t c_in_start = group * c_in_per_group;

    float sum = 0.0f;

    // Transposed convolution: scatter from input to output
    for (uint32_t c_in_local = 0; c_in_local < c_in_per_group; c_in_local++) {
        uint32_t c_in = c_in_start + c_in_local;

        for (uint32_t kh = 0; kh < kernel_h; kh++) {
            for (uint32_t kw = 0; kw < kernel_w; kw++) {
                // Find which input position contributes to this output
                int32_t h_in_offset = (int32_t)h_out + (int32_t)pad_h - (int32_t)(kh * dilation_h);
                int32_t w_in_offset = (int32_t)w_out + (int32_t)pad_w - (int32_t)(kw * dilation_w);

                // Check if this is a valid strided position
                if (h_in_offset % stride_h != 0 || w_in_offset % stride_w != 0) continue;

                int32_t h_in = h_in_offset / stride_h;
                int32_t w_in = w_in_offset / stride_w;

                if (h_in >= 0 && h_in < (int32_t)in_height &&
                    w_in >= 0 && w_in < (int32_t)in_width) {

                    float in_val = input[n * in_channels * in_height * in_width +
                                        c_in * in_height * in_width +
                                        h_in * in_width + w_in];

                    // Weight layout for transpose: [C_in, C_out, K_h, K_w]
                    float wt_val = weight[c_in * c_out_per_group * kernel_h * kernel_w +
                                         (c_out % c_out_per_group) * kernel_h * kernel_w +
                                         kh * kernel_w + kw];

                    sum += in_val * wt_val;
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[c_out];
    }

    output[n * out_channels * out_height * out_width +
           c_out * out_height * out_width +
           h_out * out_width + w_out] = sum;
}

// ============================================================================
// Conv3D General (for video and volumetric data)
// ============================================================================

extern "C" __global__
void steel_conv3d_general_kernel(
    float* __restrict__ output,          // [N, C_out, D_out, H_out, W_out]
    const float* __restrict__ input,     // [N, C_in, D_in, H_in, W_in]
    const float* __restrict__ weight,    // [C_out, C_in/groups, K_d, K_h, K_w]
    const float* __restrict__ bias,
    uint32_t batch_size,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t in_depth,
    uint32_t in_height,
    uint32_t in_width,
    uint32_t kernel_d,
    uint32_t kernel_h,
    uint32_t kernel_w,
    uint32_t stride_d,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t pad_d,
    uint32_t pad_h,
    uint32_t pad_w,
    uint32_t groups
) {
    uint32_t out_depth = (in_depth + 2 * pad_d - kernel_d) / stride_d + 1;
    uint32_t out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
    uint32_t out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_outputs = batch_size * out_channels * out_depth * out_height * out_width;

    if (idx >= total_outputs) return;

    // Decode index
    uint32_t w_out = idx % out_width;
    uint32_t temp = idx / out_width;
    uint32_t h_out = temp % out_height;
    temp = temp / out_height;
    uint32_t d_out = temp % out_depth;
    temp = temp / out_depth;
    uint32_t c_out = temp % out_channels;
    uint32_t n = temp / out_channels;

    // Group parameters
    uint32_t c_in_per_group = in_channels / groups;
    uint32_t c_out_per_group = out_channels / groups;
    uint32_t group = c_out / c_out_per_group;
    uint32_t c_in_start = group * c_in_per_group;

    float sum = 0.0f;

    for (uint32_t c_in_local = 0; c_in_local < c_in_per_group; c_in_local++) {
        uint32_t c_in = c_in_start + c_in_local;

        for (uint32_t kd = 0; kd < kernel_d; kd++) {
            int32_t d_in = (int32_t)(d_out * stride_d) - (int32_t)pad_d + (int32_t)kd;
            if (d_in < 0 || d_in >= (int32_t)in_depth) continue;

            for (uint32_t kh = 0; kh < kernel_h; kh++) {
                int32_t h_in = (int32_t)(h_out * stride_h) - (int32_t)pad_h + (int32_t)kh;
                if (h_in < 0 || h_in >= (int32_t)in_height) continue;

                for (uint32_t kw = 0; kw < kernel_w; kw++) {
                    int32_t w_in = (int32_t)(w_out * stride_w) - (int32_t)pad_w + (int32_t)kw;
                    if (w_in < 0 || w_in >= (int32_t)in_width) continue;

                    float in_val = input[n * in_channels * in_depth * in_height * in_width +
                                        c_in * in_depth * in_height * in_width +
                                        d_in * in_height * in_width +
                                        h_in * in_width + w_in];

                    float wt_val = weight[c_out * c_in_per_group * kernel_d * kernel_h * kernel_w +
                                         c_in_local * kernel_d * kernel_h * kernel_w +
                                         kd * kernel_h * kernel_w +
                                         kh * kernel_w + kw];

                    sum += in_val * wt_val;
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[c_out];
    }

    output[n * out_channels * out_depth * out_height * out_width +
           c_out * out_depth * out_height * out_width +
           d_out * out_height * out_width +
           h_out * out_width + w_out] = sum;
}

// ============================================================================
// Conv1D General (for sequences and audio)
// ============================================================================

extern "C" __global__
void steel_conv1d_general_kernel(
    float* __restrict__ output,          // [N, C_out, L_out]
    const float* __restrict__ input,     // [N, C_in, L_in]
    const float* __restrict__ weight,    // [C_out, C_in/groups, K]
    const float* __restrict__ bias,
    uint32_t batch_size,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t in_length,
    uint32_t kernel_size,
    uint32_t stride,
    uint32_t padding,
    uint32_t dilation,
    uint32_t groups
) {
    uint32_t out_length = (in_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    uint32_t n = blockIdx.z / out_channels;
    uint32_t c_out = blockIdx.z % out_channels;
    uint32_t l_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= batch_size || l_out >= out_length) return;

    uint32_t c_in_per_group = in_channels / groups;
    uint32_t c_out_per_group = out_channels / groups;
    uint32_t group = c_out / c_out_per_group;
    uint32_t c_in_start = group * c_in_per_group;

    float sum = 0.0f;

    for (uint32_t c_in_local = 0; c_in_local < c_in_per_group; c_in_local++) {
        uint32_t c_in = c_in_start + c_in_local;

        for (uint32_t k = 0; k < kernel_size; k++) {
            int32_t l_in = (int32_t)(l_out * stride) - (int32_t)padding + (int32_t)(k * dilation);

            if (l_in >= 0 && l_in < (int32_t)in_length) {
                float in_val = input[n * in_channels * in_length + c_in * in_length + l_in];
                float wt_val = weight[c_out * c_in_per_group * kernel_size + c_in_local * kernel_size + k];
                sum += in_val * wt_val;
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[c_out];
    }

    output[n * out_channels * out_length + c_out * out_length + l_out] = sum;
}

// ============================================================================
// FP16 Variants
// ============================================================================

extern "C" __global__
void steel_conv2d_general_fp16_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
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
    uint32_t groups
) {
    uint32_t out_height = (in_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    uint32_t out_width = (in_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    uint32_t n = blockIdx.z / out_channels;
    uint32_t c_out = blockIdx.z % out_channels;
    uint32_t h_out = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t w_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= batch_size || h_out >= out_height || w_out >= out_width) return;

    uint32_t c_in_per_group = in_channels / groups;
    uint32_t c_out_per_group = out_channels / groups;
    uint32_t group = c_out / c_out_per_group;
    uint32_t c_in_start = group * c_in_per_group;

    float sum = 0.0f;  // Accumulate in FP32 for precision

    for (uint32_t c_in_local = 0; c_in_local < c_in_per_group; c_in_local++) {
        uint32_t c_in = c_in_start + c_in_local;

        for (uint32_t kh = 0; kh < kernel_h; kh++) {
            for (uint32_t kw = 0; kw < kernel_w; kw++) {
                int32_t h_in = (int32_t)(h_out * stride_h) - (int32_t)pad_h + (int32_t)(kh * dilation_h);
                int32_t w_in = (int32_t)(w_out * stride_w) - (int32_t)pad_w + (int32_t)(kw * dilation_w);

                if (h_in >= 0 && h_in < (int32_t)in_height &&
                    w_in >= 0 && w_in < (int32_t)in_width) {

                    float in_val = __half2float(input[n * in_channels * in_height * in_width +
                                                     c_in * in_height * in_width +
                                                     h_in * in_width + w_in]);

                    float wt_val = __half2float(weight[c_out * c_in_per_group * kernel_h * kernel_w +
                                                      c_in_local * kernel_h * kernel_w +
                                                      kh * kernel_w + kw]);

                    sum += in_val * wt_val;
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += __half2float(bias[c_out]);
    }

    output[n * out_channels * out_height * out_width +
           c_out * out_height * out_width +
           h_out * out_width + w_out] = __float2half(sum);
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_steel_conv2d_general(
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
    uint32_t groups,
    cudaStream_t stream
) {
    uint32_t out_height = (in_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    uint32_t out_width = (in_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (out_width + TILE_SIZE - 1) / TILE_SIZE,
        (out_height + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * out_channels
    );

    steel_conv2d_general_kernel<<<blocks, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        (const float*)weight,
        (const float*)bias,
        batch_size, in_channels, out_channels,
        in_height, in_width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        groups
    );

    return cudaGetLastError();
}

int lux_cuda_steel_conv2d_transpose(
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
    uint32_t dilation_h,
    uint32_t dilation_w,
    uint32_t groups,
    cudaStream_t stream
) {
    uint32_t out_height = (in_height - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + output_pad_h + 1;
    uint32_t out_width = (in_width - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + output_pad_w + 1;

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (out_width + TILE_SIZE - 1) / TILE_SIZE,
        (out_height + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * out_channels
    );

    steel_conv2d_transpose_kernel<<<blocks, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        (const float*)weight,
        (const float*)bias,
        batch_size, in_channels, out_channels,
        in_height, in_width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        output_pad_h, output_pad_w,
        dilation_h, dilation_w,
        groups
    );

    return cudaGetLastError();
}

int lux_cuda_steel_conv3d_general(
    void* output,
    const void* input,
    const void* weight,
    const void* bias,
    uint32_t batch_size,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t in_depth,
    uint32_t in_height,
    uint32_t in_width,
    uint32_t kernel_d,
    uint32_t kernel_h,
    uint32_t kernel_w,
    uint32_t stride_d,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t pad_d,
    uint32_t pad_h,
    uint32_t pad_w,
    uint32_t groups,
    cudaStream_t stream
) {
    uint32_t out_depth = (in_depth + 2 * pad_d - kernel_d) / stride_d + 1;
    uint32_t out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
    uint32_t out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;

    uint32_t total = batch_size * out_channels * out_depth * out_height * out_width;
    uint32_t threads = BLOCK_SIZE;
    uint32_t blocks = (total + threads - 1) / threads;

    steel_conv3d_general_kernel<<<blocks, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        (const float*)weight,
        (const float*)bias,
        batch_size, in_channels, out_channels,
        in_depth, in_height, in_width,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        groups
    );

    return cudaGetLastError();
}

int lux_cuda_steel_conv1d_general(
    void* output,
    const void* input,
    const void* weight,
    const void* bias,
    uint32_t batch_size,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t in_length,
    uint32_t kernel_size,
    uint32_t stride,
    uint32_t padding,
    uint32_t dilation,
    uint32_t groups,
    cudaStream_t stream
) {
    uint32_t out_length = (in_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    uint32_t threads = BLOCK_SIZE;
    dim3 blocks(
        (out_length + threads - 1) / threads,
        1,
        batch_size * out_channels
    );

    steel_conv1d_general_kernel<<<blocks, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        (const float*)weight,
        (const float*)bias,
        batch_size, in_channels, out_channels,
        in_length, kernel_size,
        stride, padding, dilation, groups
    );

    return cudaGetLastError();
}

int lux_cuda_steel_conv2d_general_fp16(
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
    uint32_t groups,
    cudaStream_t stream
) {
    uint32_t out_height = (in_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    uint32_t out_width = (in_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (out_width + TILE_SIZE - 1) / TILE_SIZE,
        (out_height + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * out_channels
    );

    steel_conv2d_general_fp16_kernel<<<blocks, threads, 0, stream>>>(
        (__half*)output,
        (const __half*)input,
        (const __half*)weight,
        (const __half*)bias,
        batch_size, in_channels, out_channels,
        in_height, in_width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        groups
    );

    return cudaGetLastError();
}

}  // extern "C"
