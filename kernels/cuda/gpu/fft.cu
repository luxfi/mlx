// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// FFT CUDA Kernels
// Implements: Cooley-Tukey radix-2/4, Rader's algorithm for prime sizes,
// Bluestein's algorithm for arbitrary sizes, Four-step FFT for large transforms
//
// All transforms operate on float2 (complex) data

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// ============================================================================
// Configuration
// ============================================================================

#define WARP_SIZE 32
#define MAX_SHARED_SIZE 4096  // Max complex elements in shared memory (32KB)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Complex Arithmetic
// ============================================================================

__device__ __forceinline__
float2 complex_mul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__
float2 complex_conj(float2 a) {
    return make_float2(a.x, -a.y);
}

__device__ __forceinline__
float2 complex_add(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__
float2 complex_sub(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ __forceinline__
float2 complex_scale(float2 a, float s) {
    return make_float2(a.x * s, a.y * s);
}

// Twiddle factor: exp(-2*pi*i*k/n) for forward, exp(+2*pi*i*k/n) for inverse
__device__ __forceinline__
float2 twiddle_factor(int k, int n, bool inverse) {
    float angle = (inverse ? 2.0f : -2.0f) * M_PI * k / n;
    return make_float2(cosf(angle), sinf(angle));
}

// ============================================================================
// Bit Reversal Permutation
// ============================================================================

__device__ __forceinline__
uint32_t bit_reverse(uint32_t x, int log2n) {
    uint32_t result = 0;
    for (int i = 0; i < log2n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

// ============================================================================
// Radix-2 FFT Butterfly
// ============================================================================

__device__ __forceinline__
void radix2_butterfly(float2& a, float2& b, float2 w) {
    float2 t = complex_mul(w, b);
    b = complex_sub(a, t);
    a = complex_add(a, t);
}

// ============================================================================
// Radix-4 FFT Butterfly
// ============================================================================

__device__ __forceinline__
void radix4_butterfly(float2& a, float2& b, float2& c, float2& d,
                      float2 w1, float2 w2, float2 w3, bool inverse) {
    // Apply twiddles
    b = complex_mul(w1, b);
    c = complex_mul(w2, c);
    d = complex_mul(w3, d);

    // First stage of radix-4
    float2 a0 = complex_add(a, c);
    float2 a1 = complex_sub(a, c);
    float2 a2 = complex_add(b, d);
    float2 a3 = complex_sub(b, d);

    // Multiply a3 by -i (forward) or +i (inverse)
    if (inverse) {
        a3 = make_float2(a3.y, -a3.x);  // * i
    } else {
        a3 = make_float2(-a3.y, a3.x);  // * -i
    }

    // Second stage
    a = complex_add(a0, a2);
    b = complex_add(a1, a3);
    c = complex_sub(a0, a2);
    d = complex_sub(a1, a3);
}

// ============================================================================
// Stockham Autosort FFT (no bit-reversal needed)
// ============================================================================

template <int TG_SIZE>
__global__ void lux_fft_stockham_kernel(
    float2* __restrict__ output,
    const float2* __restrict__ input,
    int n,
    int batch,
    bool inverse
) {
    extern __shared__ float2 smem[];
    float2* ping = smem;
    float2* pong = smem + TG_SIZE;

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch) return;

    input += batch_idx * n;
    output += batch_idx * n;

    // Load input to shared memory
    for (int i = tid; i < n; i += blockDim.x) {
        ping[i] = input[i];
    }
    __syncthreads();

    // Compute log2(n)
    int log2n = 0;
    for (int t = n; t > 1; t >>= 1) log2n++;

    // Stockham iterations
    float2* src = ping;
    float2* dst = pong;

    for (int s = 1; s < n; s <<= 1) {
        int m = n / (2 * s);

        for (int i = tid; i < n / 2; i += blockDim.x) {
            int j = i / s;
            int k = i % s;

            float2 w = twiddle_factor(k, 2 * s, inverse);

            float2 a = src[j * s + k];
            float2 b = src[j * s + k + n / 2];

            float2 t = complex_mul(w, b);
            dst[j * 2 * s + k] = complex_add(a, t);
            dst[j * 2 * s + k + s] = complex_sub(a, t);
        }
        __syncthreads();

        // Swap buffers
        float2* tmp = src;
        src = dst;
        dst = tmp;
    }

    // Write output (scale if inverse)
    float scale = inverse ? 1.0f / n : 1.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        output[i] = complex_scale(src[i], scale);
    }
}

// ============================================================================
// Cooley-Tukey Radix-2 FFT with Bit Reversal
// ============================================================================

template <int TG_SIZE>
__global__ void lux_fft_radix2_kernel(
    float2* __restrict__ output,
    const float2* __restrict__ input,
    int n,
    int batch,
    bool inverse
) {
    extern __shared__ float2 smem[];

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch) return;

    input += batch_idx * n;
    output += batch_idx * n;

    // Compute log2(n)
    int log2n = 0;
    for (int t = n; t > 1; t >>= 1) log2n++;

    // Load with bit-reversal
    for (int i = tid; i < n; i += blockDim.x) {
        int j = bit_reverse(i, log2n);
        smem[j] = input[i];
    }
    __syncthreads();

    // FFT butterflies
    for (int s = 1; s < n; s <<= 1) {
        int m = s << 1;

        for (int i = tid; i < n / 2; i += blockDim.x) {
            int j = (i / s) * m + (i % s);
            int k = j + s;

            float2 w = twiddle_factor(i % s, m, inverse);
            radix2_butterfly(smem[j], smem[k], w);
        }
        __syncthreads();
    }

    // Write output
    float scale = inverse ? 1.0f / n : 1.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        output[i] = complex_scale(smem[i], scale);
    }
}

// ============================================================================
// Rader's Algorithm for Prime-Size FFT
// Converts N-point DFT to (N-1)-point cyclic convolution
// ============================================================================

template <int TG_SIZE>
__global__ void lux_fft_rader_kernel(
    float2* __restrict__ output,
    const float2* __restrict__ input,
    const int* __restrict__ perm,       // Permutation table (generator powers)
    const int* __restrict__ inv_perm,   // Inverse permutation
    const float2* __restrict__ rader_twiddles,  // Precomputed DFT of twiddles
    int n,
    int batch,
    bool inverse
) {
    extern __shared__ float2 smem[];
    float2* x_perm = smem;
    float2* X = smem + n;

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch) return;

    input += batch_idx * n;
    output += batch_idx * n;

    // Handle DC component separately
    float2 x0 = make_float2(0.0f, 0.0f);
    if (tid == 0) {
        for (int i = 0; i < n; i++) {
            x0 = complex_add(x0, input[i]);
        }
    }

    // Permute input: x_perm[k] = x[g^k mod n] for k = 0..n-2
    for (int i = tid; i < n - 1; i += blockDim.x) {
        x_perm[i] = input[perm[i]];
    }
    __syncthreads();

    // FFT of permuted sequence (size n-1)
    int m = n - 1;

    // Direct DFT for the cyclic convolution
    for (int k = tid; k < m; k += blockDim.x) {
        float2 sum = make_float2(0.0f, 0.0f);
        for (int j = 0; j < m; j++) {
            float2 w = twiddle_factor(j * k, m, inverse);
            sum = complex_add(sum, complex_mul(x_perm[j], w));
        }
        X[k] = sum;
    }
    __syncthreads();

    // Multiply by DFT of twiddles (convolution in frequency domain)
    for (int k = tid; k < m; k += blockDim.x) {
        X[k] = complex_mul(X[k], rader_twiddles[k]);
    }
    __syncthreads();

    // Inverse FFT to get convolution result
    float2* Y = x_perm;  // Reuse buffer
    for (int k = tid; k < m; k += blockDim.x) {
        float2 sum = make_float2(0.0f, 0.0f);
        for (int j = 0; j < m; j++) {
            float2 w = twiddle_factor(-j * k, m, inverse);
            sum = complex_add(sum, complex_mul(X[j], w));
        }
        Y[k] = complex_scale(sum, 1.0f / m);
    }
    __syncthreads();

    // Inverse permute and add DC
    float2 input0 = input[0];
    for (int k = tid; k < n - 1; k += blockDim.x) {
        float2 val = complex_add(Y[k], input0);
        output[inv_perm[k]] = complex_scale(val, inverse ? 1.0f / n : 1.0f);
    }

    if (tid == 0) {
        output[0] = complex_scale(x0, inverse ? 1.0f / n : 1.0f);
    }
}

// ============================================================================
// Bluestein's Algorithm (Chirp-Z Transform)
// Converts arbitrary-size FFT to power-of-2 convolution
// ============================================================================

template <int TG_SIZE>
__global__ void lux_fft_bluestein_kernel(
    float2* __restrict__ output,
    const float2* __restrict__ input,
    const float2* __restrict__ chirp,       // Precomputed chirp: exp(-i*pi*k^2/n)
    const float2* __restrict__ chirp_fft,   // FFT of padded chirp
    int n,      // Original size
    int m,      // Padded size (power of 2)
    int batch,
    bool inverse
) {
    extern __shared__ float2 smem[];
    float2* a = smem;           // Chirp-modulated input
    float2* A = smem + m;       // FFT of a

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch) return;

    input += batch_idx * n;
    output += batch_idx * n;

    // Modulate input by chirp: a[k] = x[k] * chirp[k]
    for (int k = tid; k < m; k += blockDim.x) {
        if (k < n) {
            float2 c = inverse ? complex_conj(chirp[k]) : chirp[k];
            a[k] = complex_mul(input[k], c);
        } else {
            a[k] = make_float2(0.0f, 0.0f);
        }
    }
    __syncthreads();

    // FFT of modulated input (power-of-2 size m)
    int log2m = 0;
    for (int t = m; t > 1; t >>= 1) log2m++;

    // Bit-reversal permutation
    for (int i = tid; i < m; i += blockDim.x) {
        int j = bit_reverse(i, log2m);
        if (i < j) {
            float2 tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
        }
    }
    __syncthreads();

    // FFT butterflies
    for (int s = 1; s < m; s <<= 1) {
        for (int i = tid; i < m / 2; i += blockDim.x) {
            int j = (i / s) * (s << 1) + (i % s);
            int k = j + s;
            float2 w = twiddle_factor(i % s, s << 1, false);
            radix2_butterfly(a[j], a[k], w);
        }
        __syncthreads();
    }

    // Multiply by chirp FFT (convolution)
    for (int k = tid; k < m; k += blockDim.x) {
        A[k] = complex_mul(a[k], chirp_fft[k]);
    }
    __syncthreads();

    // Inverse FFT
    for (int i = tid; i < m; i += blockDim.x) {
        int j = bit_reverse(i, log2m);
        if (i < j) {
            float2 tmp = A[i];
            A[i] = A[j];
            A[j] = tmp;
        }
    }
    __syncthreads();

    for (int s = 1; s < m; s <<= 1) {
        for (int i = tid; i < m / 2; i += blockDim.x) {
            int j = (i / s) * (s << 1) + (i % s);
            int k = j + s;
            float2 w = twiddle_factor(i % s, s << 1, true);
            radix2_butterfly(A[j], A[k], w);
        }
        __syncthreads();
    }

    // Demodulate and extract result
    float scale = inverse ? 1.0f / n : 1.0f;
    for (int k = tid; k < n; k += blockDim.x) {
        float2 c = inverse ? complex_conj(chirp[k]) : chirp[k];
        output[k] = complex_scale(complex_mul(A[k], c), scale / m);
    }
}

// ============================================================================
// Four-Step FFT for Large Transforms
// Decomposes N = N1 * N2 into row and column FFTs with twiddle multiplication
// ============================================================================

template <int TG_SIZE>
__global__ void lux_fft_four_step_kernel(
    float2* __restrict__ output,
    const float2* __restrict__ input,
    int n1,     // Rows
    int n2,     // Columns
    int batch,
    bool inverse,
    int step    // 0: row FFT, 1: column FFT with twiddles
) {
    extern __shared__ float2 smem[];

    int batch_idx = blockIdx.y;
    int idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch) return;

    int n = n1 * n2;
    input += batch_idx * n;
    output += batch_idx * n;

    if (step == 0) {
        // Step 0: Row FFTs (each row of length n2)
        int row = idx;
        if (row >= n1) return;

        // Load row to shared memory
        for (int i = tid; i < n2; i += blockDim.x) {
            smem[i] = input[row * n2 + i];
        }
        __syncthreads();

        // DFT for the row
        int log2n2 = 0;
        for (int t = n2; t > 1; t >>= 1) log2n2++;

        // Bit-reversal
        for (int i = tid; i < n2; i += blockDim.x) {
            int j = bit_reverse(i, log2n2);
            if (i < j) {
                float2 tmp = smem[i];
                smem[i] = smem[j];
                smem[j] = tmp;
            }
        }
        __syncthreads();

        // Butterflies
        for (int s = 1; s < n2; s <<= 1) {
            for (int i = tid; i < n2 / 2; i += blockDim.x) {
                int j = (i / s) * (s << 1) + (i % s);
                int k = j + s;
                float2 w = twiddle_factor(i % s, s << 1, inverse);
                radix2_butterfly(smem[j], smem[k], w);
            }
            __syncthreads();
        }

        // Write back with twiddle multiplication
        for (int i = tid; i < n2; i += blockDim.x) {
            float2 tw = twiddle_factor(row * i, n, inverse);
            output[row * n2 + i] = complex_mul(smem[i], tw);
        }

    } else {
        // Step 1: Column FFTs (each column of length n1)
        int col = idx;
        if (col >= n2) return;

        // Load column to shared memory
        for (int i = tid; i < n1; i += blockDim.x) {
            smem[i] = input[i * n2 + col];
        }
        __syncthreads();

        // DFT for the column
        int log2n1 = 0;
        for (int t = n1; t > 1; t >>= 1) log2n1++;

        for (int i = tid; i < n1; i += blockDim.x) {
            int j = bit_reverse(i, log2n1);
            if (i < j) {
                float2 tmp = smem[i];
                smem[i] = smem[j];
                smem[j] = tmp;
            }
        }
        __syncthreads();

        for (int s = 1; s < n1; s <<= 1) {
            for (int i = tid; i < n1 / 2; i += blockDim.x) {
                int j = (i / s) * (s << 1) + (i % s);
                int k = j + s;
                float2 w = twiddle_factor(i % s, s << 1, inverse);
                radix2_butterfly(smem[j], smem[k], w);
            }
            __syncthreads();
        }

        // Write back (scale for inverse)
        float scale = inverse ? 1.0f / n : 1.0f;
        for (int i = tid; i < n1; i += blockDim.x) {
            output[i * n2 + col] = complex_scale(smem[i], scale);
        }
    }
}

// ============================================================================
// Real-to-Complex FFT (R2C)
// Exploits conjugate symmetry for real input
// ============================================================================

template <int TG_SIZE>
__global__ void lux_fft_r2c_kernel(
    float2* __restrict__ output,   // Complex output [n/2 + 1]
    const float* __restrict__ input,  // Real input [n]
    int n,
    int batch
) {
    extern __shared__ float2 smem[];

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch) return;

    input += batch_idx * n;
    output += batch_idx * (n / 2 + 1);

    // Pack real input as complex: z[k] = x[2k] + i*x[2k+1]
    int n2 = n / 2;
    for (int i = tid; i < n2; i += blockDim.x) {
        smem[i] = make_float2(input[2 * i], input[2 * i + 1]);
    }
    __syncthreads();

    // FFT of packed data (size n/2)
    int log2n2 = 0;
    for (int t = n2; t > 1; t >>= 1) log2n2++;

    for (int i = tid; i < n2; i += blockDim.x) {
        int j = bit_reverse(i, log2n2);
        if (i < j) {
            float2 tmp = smem[i];
            smem[i] = smem[j];
            smem[j] = tmp;
        }
    }
    __syncthreads();

    for (int s = 1; s < n2; s <<= 1) {
        for (int i = tid; i < n2 / 2; i += blockDim.x) {
            int j = (i / s) * (s << 1) + (i % s);
            int k = j + s;
            float2 w = twiddle_factor(i % s, s << 1, false);
            radix2_butterfly(smem[j], smem[k], w);
        }
        __syncthreads();
    }

    // Unpack to get full-size FFT of real input
    for (int k = tid; k <= n2; k += blockDim.x) {
        if (k == 0) {
            output[0] = make_float2(smem[0].x + smem[0].y, 0.0f);
        } else if (k == n2) {
            output[n2] = make_float2(smem[0].x - smem[0].y, 0.0f);
        } else {
            float2 zk = smem[k];
            float2 znk = complex_conj(smem[n2 - k]);
            float2 w = twiddle_factor(k, n, false);

            float2 even = complex_add(zk, znk);
            float2 odd = complex_mul(complex_sub(zk, znk), w);
            odd = make_float2(odd.y, -odd.x);  // Multiply by -i

            output[k] = complex_scale(complex_add(even, odd), 0.5f);
        }
    }
}

// ============================================================================
// Complex-to-Real IFFT (C2R)
// ============================================================================

template <int TG_SIZE>
__global__ void lux_fft_c2r_kernel(
    float* __restrict__ output,      // Real output [n]
    const float2* __restrict__ input,   // Complex input [n/2 + 1]
    int n,
    int batch
) {
    extern __shared__ float2 smem[];

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch) return;

    input += batch_idx * (n / 2 + 1);
    output += batch_idx * n;

    int n2 = n / 2;

    // Pack complex input for n/2 inverse FFT
    for (int k = tid; k < n2; k += blockDim.x) {
        float2 xk = input[k];
        float2 xnk = (k == 0) ? input[0] : complex_conj(input[n2 - k]);
        if (k == 0) xnk = input[n2];

        float2 w = twiddle_factor(-k, n, false);

        float2 even = complex_add(xk, xnk);
        float2 odd = complex_sub(xk, xnk);
        odd = make_float2(-odd.y, odd.x);  // Multiply by i
        odd = complex_mul(odd, w);

        smem[k] = complex_add(even, odd);
    }
    __syncthreads();

    // Inverse FFT of size n/2
    int log2n2 = 0;
    for (int t = n2; t > 1; t >>= 1) log2n2++;

    for (int i = tid; i < n2; i += blockDim.x) {
        int j = bit_reverse(i, log2n2);
        if (i < j) {
            float2 tmp = smem[i];
            smem[i] = smem[j];
            smem[j] = tmp;
        }
    }
    __syncthreads();

    for (int s = 1; s < n2; s <<= 1) {
        for (int i = tid; i < n2 / 2; i += blockDim.x) {
            int j = (i / s) * (s << 1) + (i % s);
            int k = j + s;
            float2 w = twiddle_factor(i % s, s << 1, true);
            radix2_butterfly(smem[j], smem[k], w);
        }
        __syncthreads();
    }

    // Unpack to real output
    float scale = 1.0f / n;
    for (int i = tid; i < n2; i += blockDim.x) {
        output[2 * i] = smem[i].x * scale;
        output[2 * i + 1] = smem[i].y * scale;
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

int lux_cuda_fft_f32(
    void* output,
    const void* input,
    int n,
    int batch,
    bool inverse,
    cudaStream_t stream
) {
    int threads = min(256, n);
    size_t smem_size = 2 * n * sizeof(float2);

    if (n <= 256) {
        lux_fft_stockham_kernel<256><<<batch, threads, smem_size, stream>>>(
            (float2*)output, (const float2*)input, n, batch, inverse
        );
    } else if (n <= 1024) {
        lux_fft_stockham_kernel<1024><<<batch, threads, smem_size, stream>>>(
            (float2*)output, (const float2*)input, n, batch, inverse
        );
    } else if (n <= 4096) {
        lux_fft_stockham_kernel<4096><<<batch, threads, smem_size, stream>>>(
            (float2*)output, (const float2*)input, n, batch, inverse
        );
    } else {
        return -1;  // Use four-step FFT for larger sizes
    }

    return cudaGetLastError();
}

int lux_cuda_fft_radix2_f32(
    void* output,
    const void* input,
    int n,
    int batch,
    bool inverse,
    cudaStream_t stream
) {
    int threads = min(256, n / 2);
    size_t smem_size = n * sizeof(float2);

    if (n <= 256) {
        lux_fft_radix2_kernel<256><<<batch, threads, smem_size, stream>>>(
            (float2*)output, (const float2*)input, n, batch, inverse
        );
    } else if (n <= 1024) {
        lux_fft_radix2_kernel<1024><<<batch, threads, smem_size, stream>>>(
            (float2*)output, (const float2*)input, n, batch, inverse
        );
    } else if (n <= 4096) {
        lux_fft_radix2_kernel<4096><<<batch, threads, smem_size, stream>>>(
            (float2*)output, (const float2*)input, n, batch, inverse
        );
    } else {
        return -1;
    }

    return cudaGetLastError();
}

int lux_cuda_fft_rader_f32(
    void* output,
    const void* input,
    const int* perm,
    const int* inv_perm,
    const void* rader_twiddles,
    int n,
    int batch,
    bool inverse,
    cudaStream_t stream
) {
    int threads = min(256, n);
    size_t smem_size = 2 * n * sizeof(float2);

    lux_fft_rader_kernel<256><<<batch, threads, smem_size, stream>>>(
        (float2*)output, (const float2*)input,
        perm, inv_perm, (const float2*)rader_twiddles,
        n, batch, inverse
    );

    return cudaGetLastError();
}

int lux_cuda_fft_bluestein_f32(
    void* output,
    const void* input,
    const void* chirp,
    const void* chirp_fft,
    int n,
    int m,
    int batch,
    bool inverse,
    cudaStream_t stream
) {
    int threads = min(256, m);
    size_t smem_size = 2 * m * sizeof(float2);

    lux_fft_bluestein_kernel<256><<<batch, threads, smem_size, stream>>>(
        (float2*)output, (const float2*)input,
        (const float2*)chirp, (const float2*)chirp_fft,
        n, m, batch, inverse
    );

    return cudaGetLastError();
}

int lux_cuda_fft_four_step_f32(
    void* output,
    const void* input,
    int n1,
    int n2,
    int batch,
    bool inverse,
    int step,
    cudaStream_t stream
) {
    int n = (step == 0) ? n2 : n1;
    int num_blocks = (step == 0) ? n1 : n2;
    int threads = min(256, n);
    size_t smem_size = n * sizeof(float2);

    dim3 blocks(num_blocks, batch);

    lux_fft_four_step_kernel<256><<<blocks, threads, smem_size, stream>>>(
        (float2*)output, (const float2*)input,
        n1, n2, batch, inverse, step
    );

    return cudaGetLastError();
}

int lux_cuda_fft_r2c_f32(
    void* output,
    const void* input,
    int n,
    int batch,
    cudaStream_t stream
) {
    int threads = min(256, n / 2);
    size_t smem_size = (n / 2) * sizeof(float2);

    lux_fft_r2c_kernel<256><<<batch, threads, smem_size, stream>>>(
        (float2*)output, (const float*)input, n, batch
    );

    return cudaGetLastError();
}

int lux_cuda_fft_c2r_f32(
    void* output,
    const void* input,
    int n,
    int batch,
    cudaStream_t stream
) {
    int threads = min(256, n / 2);
    size_t smem_size = (n / 2) * sizeof(float2);

    lux_fft_c2r_kernel<256><<<batch, threads, smem_size, stream>>>(
        (float*)output, (const float2*)input, n, batch
    );

    return cudaGetLastError();
}

}  // extern "C"
