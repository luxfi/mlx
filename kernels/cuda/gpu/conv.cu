// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Convolution CUDA Kernels
// Implements: depthwise conv, unfold/im2col, Winograd F(6,3)
// Optimized for ML inference workloads

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// ============================================================================
// Configuration
// ============================================================================

#define WARP_SIZE 32
#define BLOCK_SIZE 256

// ============================================================================
// Convolution Parameters (matches Metal MLXConvParams)
// ============================================================================

template <int N>
struct ConvParams {
    int32_t N_batch;           // Batch size
    int32_t C;                 // Input channels
    int32_t O;                 // Output channels
    int32_t iS[N];             // Input spatial dims
    int32_t oS[N];             // Output spatial dims
    int32_t wS[N];             // Kernel spatial dims
    int32_t str[N];            // Strides
    int32_t pad[N];            // Padding
    int32_t kdil[N];           // Kernel dilation
    int32_t idil[N];           // Input dilation
    int64_t in_strides[N + 2]; // Input strides (batch, spatial..., channel)
    int64_t wt_strides[N + 2]; // Weight strides
    int64_t out_strides[N + 2];// Output strides
    bool flip;                 // Flip kernel (for transposed conv)
};

// ============================================================================
// Naive Unfold N-D (im2col)
// Unfolds input into patches for matrix multiplication
// Input: [N, *spatial_dims, C] -> Output: [N * *oS, C * *wS]
// ============================================================================

template <typename T, int N_dims>
__global__ void lux_unfold_nd_kernel(
    T* __restrict__ out,
    const T* __restrict__ in,
    const ConvParams<N_dims>* params
) {
    // Compute filter size and output pixels
    int filter_size = params->C;
    for (int i = 0; i < N_dims; i++) {
        filter_size *= params->wS[i];
    }

    int out_pixels = 1;
    for (int i = 0; i < N_dims; i++) {
        out_pixels *= params->oS[i];
    }

    // gid.z: N * oS (batch and row in unfolded output)
    // gid.y: wS (filter location)
    // gid.x: C (channel)
    uint32_t gid_z = blockIdx.z * blockDim.z + threadIdx.z;
    uint32_t gid_y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t gid_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid_x >= (uint32_t)params->C) return;

    // Set output pointer
    out += gid_z * filter_size + gid_y * params->C;

    int n = gid_z / out_pixels;
    int oS = gid_z % out_pixels;
    int wS = gid_y;

    bool valid = n < params->N_batch;

    // Coordinates in input
    int is[N_dims];

    // Unroll dimensions
    int oS_temp = oS;
    int wS_temp = wS;
    for (int i = N_dims - 1; i >= 0; --i) {
        int os_ = oS_temp % params->oS[i];
        int ws_ = wS_temp % params->wS[i];

        ws_ = params->flip ? params->wS[i] - ws_ - 1 : ws_;

        int is_ = os_ * params->str[i] - params->pad[i] + ws_ * params->kdil[i];
        int is_max = 1 + params->idil[i] * (params->iS[i] - 1);

        valid = valid && (is_ >= 0) && (is_ < is_max) && (is_ % params->idil[i] == 0);

        is[i] = is_ / params->idil[i];

        oS_temp /= params->oS[i];
        wS_temp /= params->wS[i];
    }

    if (valid) {
        int64_t in_offset = n * params->in_strides[0];
        for (int i = 0; i < N_dims; ++i) {
            in_offset += is[i] * params->in_strides[i + 1];
        }
        out[gid_x] = in[in_offset + gid_x];
    } else {
        out[gid_x] = T(0);
    }
}

// ============================================================================
// Depthwise Convolution 1D
// Each output channel uses a single input channel (groups = C)
// ============================================================================

template <typename T>
__global__ void lux_depthwise_conv1d_kernel(
    T* __restrict__ out,
    const T* __restrict__ in,
    const T* __restrict__ w,
    const int64_t* strides,  // [batch_stride, spatial_stride, channel_stride]
    int32_t kernel_size,
    int32_t out_width,
    int32_t channels
) {
    // tid.z: batch
    // tid.y: output position
    // tid.x: channel
    uint32_t b = blockIdx.z;
    uint32_t pos = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;

    if (c >= (uint32_t)channels || pos >= (uint32_t)out_width) return;

    out += (b * out_width + pos) * channels + c;
    in += b * strides[0] + pos * strides[1] + c * strides[2];
    w += c * kernel_size;

    float acc = 0.0f;
    for (int i = 0; i < kernel_size; ++i) {
        acc += static_cast<float>(in[0]) * static_cast<float>(w[i]);
        in += strides[1];
    }
    *out = static_cast<T>(acc);
}

// ============================================================================
// Depthwise Convolution 2D
// Optimized with shared memory tiling
// ============================================================================

template <typename T, int TILE_H = 4, int TILE_W = 8, int TILE_C = 8>
__global__ void lux_depthwise_conv2d_kernel(
    T* __restrict__ out,
    const T* __restrict__ in,
    const T* __restrict__ wt,
    const ConvParams<2>* params,
    int32_t ker_h,
    int32_t ker_w,
    int32_t str_h,
    int32_t str_w,
    bool do_flip
) {
    // Shared memory for input tile
    constexpr int SPAN_H = TILE_H * 2 + 6;  // Assuming max kernel 7x7
    constexpr int SPAN_W = TILE_W * 2 + 6;
    __shared__ float ins[SPAN_H][SPAN_W][TILE_C];

    int n_tgblocks_h = params->oS[0] / TILE_H;
    int n = blockIdx.z / n_tgblocks_h;
    int tghid = blockIdx.z % n_tgblocks_h;
    int oh = tghid * TILE_H + threadIdx.z;
    int ow = blockIdx.y * TILE_W + threadIdx.y;
    int c = blockIdx.x * TILE_C + threadIdx.x;

    if (c >= params->C) return;

    // Load input tile to shared memory
    int tg_oh = tghid * TILE_H * str_h - params->pad[0];
    int tg_ow = blockIdx.y * TILE_W * str_w - params->pad[1];
    int tg_c = blockIdx.x * TILE_C;

    int thread_idx = threadIdx.z * TILE_W * TILE_C + threadIdx.y * TILE_C + threadIdx.x;
    int n_threads = TILE_H * TILE_W * TILE_C;

    // Cooperative loading
    int span_hw = (TILE_H * str_h + ker_h - 1) * (TILE_W * str_w + ker_w - 1);
    int span_w = TILE_W * str_w + ker_w - 1;

    for (int hw = thread_idx; hw < span_hw; hw += n_threads) {
        int h = hw / span_w;
        int w = hw % span_w;
        int ih = tg_oh + h;
        int iw = tg_ow + w;

        bool valid = (ih >= 0 && ih < params->iS[0] && iw >= 0 && iw < params->iS[1]);

        if (valid && threadIdx.x < TILE_C) {
            int64_t in_idx = n * params->in_strides[0] +
                            ih * params->in_strides[1] +
                            iw * params->in_strides[2] +
                            tg_c + threadIdx.x;
            ins[h][w][threadIdx.x] = static_cast<float>(in[in_idx]);
        } else if (threadIdx.x < TILE_C) {
            ins[h][w][threadIdx.x] = 0.0f;
        }
    }

    __syncthreads();

    if (oh >= params->oS[0] || ow >= params->oS[1]) return;

    // Compute convolution
    float acc = 0.0f;
    const T* wt_ptr = wt + c * ker_h * ker_w;

    for (int kh = 0; kh < ker_h; ++kh) {
        for (int kw = 0; kw < ker_w; ++kw) {
            int wt_h = do_flip ? ker_h - kh - 1 : kh;
            int wt_w = do_flip ? ker_w - kw - 1 : kw;

            int sh = threadIdx.z * str_h + kh;
            int sw = threadIdx.y * str_w + kw;

            acc += ins[sh][sw][threadIdx.x] * static_cast<float>(wt_ptr[wt_h * ker_w + wt_w]);
        }
    }

    // Write output
    int64_t out_idx = n * params->out_strides[0] +
                      oh * params->out_strides[1] +
                      ow * params->out_strides[2] + c;
    out[out_idx] = static_cast<T>(acc);
}

// ============================================================================
// Winograd F(6,3) Transforms for 2D Convolution
// Output tile: 6x6, Filter: 3x3, Input tile: 8x8
// ============================================================================

// Winograd transform matrices stored in constant memory
__constant__ float WINOGRAD_G[8][3] = {
    {1.00f, 0.00f, 0.00f},
    {-2.0f/9.0f, -2.0f/9.0f, -2.0f/9.0f},
    {-2.0f/9.0f, 2.0f/9.0f, -2.0f/9.0f},
    {1.0f/90.0f, 1.0f/45.0f, 2.0f/45.0f},
    {1.0f/90.0f, -1.0f/45.0f, 2.0f/45.0f},
    {32.0f/45.0f, 16.0f/45.0f, 8.0f/45.0f},
    {32.0f/45.0f, -16.0f/45.0f, 8.0f/45.0f},
    {0.00f, 0.00f, 1.00f}
};

__constant__ float WINOGRAD_BT[8][8] = {
    {1.00f, 0.00f, -5.25f, 0.00f, 5.25f, 0.00f, -1.00f, 0.00f},
    {0.00f, 1.00f, 1.00f, -4.25f, -4.25f, 1.00f, 1.00f, 0.00f},
    {0.00f, -1.00f, 1.00f, 4.25f, -4.25f, -1.00f, 1.00f, 0.00f},
    {0.00f, 0.50f, 0.25f, -2.50f, -1.25f, 2.00f, 1.00f, 0.00f},
    {0.00f, -0.50f, 0.25f, 2.50f, -1.25f, -2.00f, 1.00f, 0.00f},
    {0.00f, 2.00f, 4.00f, -2.50f, -5.00f, 0.50f, 1.00f, 0.00f},
    {0.00f, -2.00f, 4.00f, 2.50f, -5.00f, -0.50f, 1.00f, 0.00f},
    {0.00f, -1.00f, 0.00f, 5.25f, 0.00f, -5.25f, 0.00f, 1.00f}
};

__constant__ float WINOGRAD_AT[6][8] = {
    {1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 0.00f},
    {0.00f, 1.00f, -1.00f, 2.00f, -2.00f, 0.50f, -0.50f, 0.00f},
    {0.00f, 1.00f, 1.00f, 4.00f, 4.00f, 0.25f, 0.25f, 0.00f},
    {0.00f, 1.00f, -1.00f, 8.00f, -8.00f, 0.125f, -0.125f, 0.00f},
    {0.00f, 1.00f, 1.00f, 16.00f, 16.00f, 0.0625f, 0.0625f, 0.00f},
    {0.00f, 1.00f, -1.00f, 32.00f, -32.00f, 0.03125f, -0.03125f, 1.00f}
};

// ============================================================================
// Winograd Weight Transform: G * g * G^T
// Input: [O, 3, 3, C] -> Output: [8, 8, C, O]
// ============================================================================

template <typename T, int BC = 32>
__global__ void lux_winograd_weight_transform_kernel(
    T* __restrict__ wt_out,
    const T* __restrict__ wt_in,
    int32_t C,
    int32_t O
) {
    __shared__ float ws[3][3][BC];

    int ko = blockIdx.x * blockDim.y + threadIdx.y;  // Output channel
    int bc_base = blockIdx.y * BC;                    // Channel batch

    if (ko >= O) return;

    // Each warp transforms one output filter
    wt_in += ko * 9 * C;  // [O, 3, 3, C] layout

    // Process BC channels at a time
    for (int bc = 0; bc < BC && (bc_base + bc) < C; bc++) {
        int c = bc_base + bc;

        // Load 3x3 filter into shared memory
        if (threadIdx.x < 9) {
            int kh = threadIdx.x / 3;
            int kw = threadIdx.x % 3;
            ws[kh][kw][threadIdx.x % BC] = static_cast<float>(wt_in[kh * 3 * C + kw * C + c]);
        }
        __syncthreads();

        // Compute G * g
        float Gg[8][3];
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 3; j++) {
                Gg[i][j] = 0.0f;
                for (int k = 0; k < 3; k++) {
                    Gg[i][j] += WINOGRAD_G[i][k] * ws[k][j][threadIdx.x % BC];
                }
            }
        }

        // Compute (G * g) * G^T and store
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                float val = 0.0f;
                for (int k = 0; k < 3; k++) {
                    val += Gg[i][k] * WINOGRAD_G[j][k];  // G^T
                }
                // Output layout: [8, 8, C, O]
                wt_out[(i * 8 + j) * C * O + c * O + ko] = static_cast<T>(val);
            }
        }
        __syncthreads();
    }
}

// ============================================================================
// Winograd Input Transform: B^T * d * B
// Input: [N, H, W, C] -> Output: [8, 8, tiles, C]
// ============================================================================

template <typename T, int BC = 32>
__global__ void lux_winograd_input_transform_kernel(
    T* __restrict__ inp_out,
    const T* __restrict__ inp_in,
    const ConvParams<2>* params,
    int32_t tiles_h,
    int32_t tiles_w
) {
    __shared__ float ds[8][8][BC];

    int tile_idx = blockIdx.x;  // Linear tile index
    int n = tile_idx / (tiles_h * tiles_w);
    int tile_hw = tile_idx % (tiles_h * tiles_w);
    int tile_h = tile_hw / tiles_w;
    int tile_w = tile_hw % tiles_w;

    int bc_base = blockIdx.y * BC;

    // Input tile top-left corner
    int ih_base = tile_h * 6 - params->pad[0];
    int iw_base = tile_w * 6 - params->pad[1];

    int total_tiles = params->N_batch * tiles_h * tiles_w;

    for (int bc = 0; bc < BC && (bc_base + bc) < params->C; bc++) {
        int c = bc_base + bc;

        // Load 8x8 input tile
        if (threadIdx.x < 64) {
            int lh = threadIdx.x / 8;
            int lw = threadIdx.x % 8;
            int ih = ih_base + lh;
            int iw = iw_base + lw;

            float val = 0.0f;
            if (ih >= 0 && ih < params->iS[0] && iw >= 0 && iw < params->iS[1]) {
                int64_t idx = n * params->in_strides[0] +
                             ih * params->in_strides[1] +
                             iw * params->in_strides[2] + c;
                val = static_cast<float>(inp_in[idx]);
            }
            ds[lh][lw][threadIdx.x % BC] = val;
        }
        __syncthreads();

        // Compute B^T * d
        float Btd[8][8];
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                Btd[i][j] = 0.0f;
                for (int k = 0; k < 8; k++) {
                    Btd[i][j] += WINOGRAD_BT[i][k] * ds[k][j][threadIdx.x % BC];
                }
            }
        }

        // Compute (B^T * d) * B and store
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                float val = 0.0f;
                for (int k = 0; k < 8; k++) {
                    val += Btd[i][k] * WINOGRAD_BT[j][k];  // B = B^T^T
                }
                // Output layout: [8, 8, tiles, C]
                inp_out[(i * 8 + j) * total_tiles * params->C + tile_idx * params->C + c] =
                    static_cast<T>(val);
            }
        }
        __syncthreads();
    }
}

// ============================================================================
// Winograd Output Transform: A^T * m * A
// Input: [8, 8, tiles, O] -> Output: [N, H, W, O]
// ============================================================================

template <typename T, int BO = 32>
__global__ void lux_winograd_output_transform_kernel(
    T* __restrict__ out_out,
    const T* __restrict__ out_in,
    const ConvParams<2>* params,
    int32_t tiles_h,
    int32_t tiles_w
) {
    __shared__ float ms[8][8][BO];

    int tile_idx = blockIdx.x;
    int n = tile_idx / (tiles_h * tiles_w);
    int tile_hw = tile_idx % (tiles_h * tiles_w);
    int tile_h = tile_hw / tiles_w;
    int tile_w = tile_hw % tiles_w;

    int bo_base = blockIdx.y * BO;
    int total_tiles = params->N_batch * tiles_h * tiles_w;

    for (int bo = 0; bo < BO && (bo_base + bo) < params->O; bo++) {
        int o = bo_base + bo;

        // Load 8x8 Winograd domain tile
        if (threadIdx.x < 64) {
            int lh = threadIdx.x / 8;
            int lw = threadIdx.x % 8;
            int64_t idx = (lh * 8 + lw) * total_tiles * params->O + tile_idx * params->O + o;
            ms[lh][lw][threadIdx.x % BO] = static_cast<float>(out_in[idx]);
        }
        __syncthreads();

        // Compute A^T * m
        float Atm[6][8];
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 8; j++) {
                Atm[i][j] = 0.0f;
                for (int k = 0; k < 8; k++) {
                    Atm[i][j] += WINOGRAD_AT[i][k] * ms[k][j][threadIdx.x % BO];
                }
            }
        }

        // Compute (A^T * m) * A and store
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                int oh = tile_h * 6 + i;
                int ow = tile_w * 6 + j;

                if (oh < params->oS[0] && ow < params->oS[1]) {
                    float val = 0.0f;
                    for (int k = 0; k < 8; k++) {
                        val += Atm[i][k] * WINOGRAD_AT[j][k];
                    }
                    int64_t out_idx = n * params->out_strides[0] +
                                     oh * params->out_strides[1] +
                                     ow * params->out_strides[2] + o;
                    out_out[out_idx] = static_cast<T>(val);
                }
            }
        }
        __syncthreads();
    }
}

// ============================================================================
// Direct Convolution 2D (for small filters or when Winograd doesn't apply)
// ============================================================================

template <typename T>
__global__ void lux_conv2d_direct_kernel(
    T* __restrict__ out,
    const T* __restrict__ in,
    const T* __restrict__ wt,
    const ConvParams<2>* params
) {
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.z / params->O;
    int oc = blockIdx.z % params->O;

    if (oh >= params->oS[0] || ow >= params->oS[1]) return;

    float acc = 0.0f;

    for (int ic = 0; ic < params->C; ic++) {
        for (int kh = 0; kh < params->wS[0]; kh++) {
            for (int kw = 0; kw < params->wS[1]; kw++) {
                int ih = oh * params->str[0] - params->pad[0] + kh * params->kdil[0];
                int iw = ow * params->str[1] - params->pad[1] + kw * params->kdil[1];

                if (ih >= 0 && ih < params->iS[0] && iw >= 0 && iw < params->iS[1]) {
                    int64_t in_idx = n * params->in_strides[0] +
                                    ih * params->in_strides[1] +
                                    iw * params->in_strides[2] + ic;

                    int wt_kh = params->flip ? params->wS[0] - kh - 1 : kh;
                    int wt_kw = params->flip ? params->wS[1] - kw - 1 : kw;
                    int64_t wt_idx = oc * params->wt_strides[0] +
                                    wt_kh * params->wS[1] * params->C +
                                    wt_kw * params->C + ic;

                    acc += static_cast<float>(in[in_idx]) * static_cast<float>(wt[wt_idx]);
                }
            }
        }
    }

    int64_t out_idx = n * params->out_strides[0] +
                     oh * params->out_strides[1] +
                     ow * params->out_strides[2] + oc;
    out[out_idx] = static_cast<T>(acc);
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_unfold_1d_f32(
    void* output,
    const void* input,
    const void* params,
    uint32_t batch_out_size,
    uint32_t filter_size,
    uint32_t channels,
    cudaStream_t stream
) {
    dim3 blocks((channels + 31) / 32, filter_size, batch_out_size);
    dim3 threads(32, 1, 1);

    lux_unfold_nd_kernel<float, 1><<<blocks, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        (const ConvParams<1>*)params
    );

    return cudaGetLastError();
}

int lux_cuda_unfold_2d_f32(
    void* output,
    const void* input,
    const void* params,
    uint32_t batch_out_size,
    uint32_t filter_size,
    uint32_t channels,
    cudaStream_t stream
) {
    dim3 blocks((channels + 31) / 32, filter_size, batch_out_size);
    dim3 threads(32, 1, 1);

    lux_unfold_nd_kernel<float, 2><<<blocks, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        (const ConvParams<2>*)params
    );

    return cudaGetLastError();
}

int lux_cuda_depthwise_conv1d_f32(
    void* output,
    const void* input,
    const void* weights,
    const int64_t* strides,
    int32_t kernel_size,
    int32_t batch_size,
    int32_t out_width,
    int32_t channels,
    cudaStream_t stream
) {
    dim3 blocks((channels + 31) / 32, (out_width + 7) / 8, batch_size);
    dim3 threads(32, 8, 1);

    lux_depthwise_conv1d_kernel<float><<<blocks, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        (const float*)weights,
        strides,
        kernel_size,
        out_width,
        channels
    );

    return cudaGetLastError();
}

int lux_cuda_depthwise_conv2d_f32(
    void* output,
    const void* input,
    const void* weights,
    const void* params,
    int32_t kernel_h,
    int32_t kernel_w,
    int32_t stride_h,
    int32_t stride_w,
    bool flip,
    cudaStream_t stream
) {
    const ConvParams<2>* p = (const ConvParams<2>*)params;

    int tiles_h = (p->oS[0] + 3) / 4;
    int tiles_c = (p->C + 7) / 8;

    dim3 blocks(tiles_c, (p->oS[1] + 7) / 8, p->N_batch * tiles_h);
    dim3 threads(8, 8, 4);

    lux_depthwise_conv2d_kernel<float><<<blocks, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        (const float*)weights,
        p,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        flip
    );

    return cudaGetLastError();
}

int lux_cuda_winograd_weight_transform_f32(
    void* output,
    const void* input,
    int32_t C,
    int32_t O,
    cudaStream_t stream
) {
    dim3 blocks((O + 3) / 4, (C + 31) / 32);
    dim3 threads(32, 4);

    lux_winograd_weight_transform_kernel<float><<<blocks, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        C, O
    );

    return cudaGetLastError();
}

int lux_cuda_winograd_input_transform_f32(
    void* output,
    const void* input,
    const void* params,
    int32_t tiles_h,
    int32_t tiles_w,
    cudaStream_t stream
) {
    const ConvParams<2>* p = (const ConvParams<2>*)params;
    int total_tiles = p->N_batch * tiles_h * tiles_w;

    dim3 blocks(total_tiles, (p->C + 31) / 32);
    dim3 threads(64);

    lux_winograd_input_transform_kernel<float><<<blocks, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        p,
        tiles_h,
        tiles_w
    );

    return cudaGetLastError();
}

int lux_cuda_winograd_output_transform_f32(
    void* output,
    const void* input,
    const void* params,
    int32_t tiles_h,
    int32_t tiles_w,
    cudaStream_t stream
) {
    const ConvParams<2>* p = (const ConvParams<2>*)params;
    int total_tiles = p->N_batch * tiles_h * tiles_w;

    dim3 blocks(total_tiles, (p->O + 31) / 32);
    dim3 threads(64);

    lux_winograd_output_transform_kernel<float><<<blocks, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        p,
        tiles_h,
        tiles_w
    );

    return cudaGetLastError();
}

int lux_cuda_conv2d_direct_f32(
    void* output,
    const void* input,
    const void* weights,
    const void* params,
    cudaStream_t stream
) {
    const ConvParams<2>* p = (const ConvParams<2>*)params;

    dim3 blocks((p->oS[1] + 15) / 16, (p->oS[0] + 15) / 16, p->N_batch * p->O);
    dim3 threads(16, 16);

    lux_conv2d_direct_kernel<float><<<blocks, threads, 0, stream>>>(
        (float*)output,
        (const float*)input,
        (const float*)weights,
        p
    );

    return cudaGetLastError();
}

}  // extern "C"
