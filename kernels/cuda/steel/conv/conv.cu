// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Steel Convolution CUDA Kernels
// Optimized 2D convolution implementations

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// Convolution Configuration
// ============================================================================

#define TILE_SIZE 16
#define MAX_KERNEL_SIZE 7

// ============================================================================
// Direct Convolution (Im2col-free)
// ============================================================================

extern "C" __global__
void steel_conv2d_direct_kernel(
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
    uint32_t pad_w
) {
    uint32_t out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
    uint32_t out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;

    uint32_t n = blockIdx.z / out_channels;
    uint32_t c_out = blockIdx.z % out_channels;
    uint32_t h_out = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t w_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= batch_size || h_out >= out_height || w_out >= out_width) return;

    float sum = 0.0f;

    // Convolution
    for (uint32_t c_in = 0; c_in < in_channels; c_in++) {
        for (uint32_t kh = 0; kh < kernel_h; kh++) {
            for (uint32_t kw = 0; kw < kernel_w; kw++) {
                int32_t h_in = (int32_t)(h_out * stride_h) - (int32_t)pad_h + (int32_t)kh;
                int32_t w_in = (int32_t)(w_out * stride_w) - (int32_t)pad_w + (int32_t)kw;

                if (h_in >= 0 && h_in < (int32_t)in_height &&
                    w_in >= 0 && w_in < (int32_t)in_width) {

                    float in_val = input[n * in_channels * in_height * in_width +
                                        c_in * in_height * in_width +
                                        h_in * in_width + w_in];

                    float wt_val = weight[c_out * in_channels * kernel_h * kernel_w +
                                         c_in * kernel_h * kernel_w +
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
// Depthwise Convolution
// ============================================================================

extern "C" __global__
void steel_conv2d_depthwise_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,    // [C, 1, K_h, K_w]
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
    uint32_t out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
    uint32_t out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;

    uint32_t n = blockIdx.z / channels;
    uint32_t c = blockIdx.z % channels;
    uint32_t h_out = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t w_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= batch_size || h_out >= out_height || w_out >= out_width) return;

    float sum = 0.0f;

    for (uint32_t kh = 0; kh < kernel_h; kh++) {
        for (uint32_t kw = 0; kw < kernel_w; kw++) {
            int32_t h_in = (int32_t)(h_out * stride_h) - (int32_t)pad_h + (int32_t)kh;
            int32_t w_in = (int32_t)(w_out * stride_w) - (int32_t)pad_w + (int32_t)kw;

            if (h_in >= 0 && h_in < (int32_t)in_height &&
                w_in >= 0 && w_in < (int32_t)in_width) {

                float in_val = input[n * channels * in_height * in_width +
                                    c * in_height * in_width +
                                    h_in * in_width + w_in];

                float wt_val = weight[c * kernel_h * kernel_w + kh * kernel_w + kw];

                sum += in_val * wt_val;
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[c];
    }

    output[n * channels * out_height * out_width +
           c * out_height * out_width +
           h_out * out_width + w_out] = sum;
}

// ============================================================================
// 1x1 Convolution (Pointwise)
// ============================================================================

extern "C" __global__
void steel_conv2d_1x1_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,    // [C_out, C_in]
    const float* __restrict__ bias,
    uint32_t batch_size,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t height,
    uint32_t width
) {
    // Treat as batched matrix multiplication
    // Each spatial position is a vector of in_channels
    // Weight is [out_channels, in_channels]

    uint32_t spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t c_out = blockIdx.y;
    uint32_t n = blockIdx.z;

    if (n >= batch_size || spatial_idx >= height * width) return;

    const float* in_ptr = input + n * in_channels * height * width;

    float sum = 0.0f;

    for (uint32_t c_in = 0; c_in < in_channels; c_in++) {
        sum += in_ptr[c_in * height * width + spatial_idx] * weight[c_out * in_channels + c_in];
    }

    if (bias != nullptr) {
        sum += bias[c_out];
    }

    output[n * out_channels * height * width + c_out * height * width + spatial_idx] = sum;
}

// ============================================================================
// Im2Col Transformation (for GEMM-based convolution)
// ============================================================================

extern "C" __global__
void steel_im2col_kernel(
    float* __restrict__ col,             // [C_in * K_h * K_w, H_out * W_out]
    const float* __restrict__ input,     // [C_in, H_in, W_in]
    uint32_t in_channels,
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

    uint32_t col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t num_col_elements = in_channels * kernel_h * kernel_w * out_height * out_width;

    if (col_idx >= num_col_elements) return;

    // Decode index
    uint32_t spatial_idx = col_idx % (out_height * out_width);
    uint32_t kernel_idx = col_idx / (out_height * out_width);

    uint32_t h_out = spatial_idx / out_width;
    uint32_t w_out = spatial_idx % out_width;

    uint32_t c_in = kernel_idx / (kernel_h * kernel_w);
    uint32_t k_idx = kernel_idx % (kernel_h * kernel_w);
    uint32_t kh = k_idx / kernel_w;
    uint32_t kw = k_idx % kernel_w;

    int32_t h_in = (int32_t)(h_out * stride_h) - (int32_t)pad_h + (int32_t)kh;
    int32_t w_in = (int32_t)(w_out * stride_w) - (int32_t)pad_w + (int32_t)kw;

    float val = 0.0f;
    if (h_in >= 0 && h_in < (int32_t)in_height &&
        w_in >= 0 && w_in < (int32_t)in_width) {
        val = input[c_in * in_height * in_width + h_in * in_width + w_in];
    }

    col[kernel_idx * out_height * out_width + spatial_idx] = val;
}

// ============================================================================
// Col2Im Transformation (for backward pass)
// ============================================================================

extern "C" __global__
void steel_col2im_kernel(
    float* __restrict__ input_grad,
    const float* __restrict__ col_grad,
    uint32_t in_channels,
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

    uint32_t in_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t num_in_elements = in_channels * in_height * in_width;

    if (in_idx >= num_in_elements) return;

    uint32_t c_in = in_idx / (in_height * in_width);
    uint32_t spatial_idx = in_idx % (in_height * in_width);
    uint32_t h_in = spatial_idx / in_width;
    uint32_t w_in = spatial_idx % in_width;

    float grad_sum = 0.0f;

    // Find all output positions that contribute to this input
    for (uint32_t kh = 0; kh < kernel_h; kh++) {
        for (uint32_t kw = 0; kw < kernel_w; kw++) {
            int32_t h_out = ((int32_t)h_in + (int32_t)pad_h - (int32_t)kh);
            int32_t w_out = ((int32_t)w_in + (int32_t)pad_w - (int32_t)kw);

            if (h_out % stride_h == 0 && w_out % stride_w == 0) {
                h_out /= stride_h;
                w_out /= stride_w;

                if (h_out >= 0 && h_out < (int32_t)out_height &&
                    w_out >= 0 && w_out < (int32_t)out_width) {

                    uint32_t kernel_idx = c_in * kernel_h * kernel_w + kh * kernel_w + kw;
                    uint32_t col_idx = kernel_idx * out_height * out_width + h_out * out_width + w_out;

                    grad_sum += col_grad[col_idx];
                }
            }
        }
    }

    input_grad[in_idx] = grad_sum;
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_steel_conv2d(
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

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (out_width + TILE_SIZE - 1) / TILE_SIZE,
        (out_height + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * out_channels
    );

    steel_conv2d_direct_kernel<<<blocks, threads, 0, stream>>>(
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

int lux_cuda_steel_conv2d_depthwise(
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

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (out_width + TILE_SIZE - 1) / TILE_SIZE,
        (out_height + TILE_SIZE - 1) / TILE_SIZE,
        batch_size * channels
    );

    steel_conv2d_depthwise_kernel<<<blocks, threads, 0, stream>>>(
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

int lux_cuda_steel_im2col(
    void* col,
    const void* input,
    uint32_t in_channels,
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
    uint32_t num_elements = in_channels * kernel_h * kernel_w * out_height * out_width;

    uint32_t threads = 256;
    uint32_t blocks = (num_elements + threads - 1) / threads;

    steel_im2col_kernel<<<blocks, threads, 0, stream>>>(
        (float*)col,
        (const float*)input,
        in_channels, in_height, in_width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w
    );

    return cudaGetLastError();
}

}  // extern "C"
