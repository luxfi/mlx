// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// CPU Tensor Operations - SIMD-optimized implementations
// Uses intrinsics when available, falls back to scalar code

#include "tensor_ops.h"
#include <math.h>
#include <float.h>
#include <string.h>

// =============================================================================
// SIMD Detection
// =============================================================================

#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
    #define LUX_HAS_AVX2
#elif defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>
    #define LUX_HAS_NEON
#endif

// =============================================================================
// Elementwise Binary Operations
// =============================================================================

void lux_cpu_add_f32(float* out, const float* a, const float* b, size_t n) {
#ifdef LUX_HAS_AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
    }
    for (; i < n; i++) {
        out[i] = a[i] + b[i];
    }
#elif defined(LUX_HAS_NEON)
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vaddq_f32(va, vb));
    }
    for (; i < n; i++) {
        out[i] = a[i] + b[i];
    }
#else
    for (size_t i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
#endif
}

void lux_cpu_sub_f32(float* out, const float* a, const float* b, size_t n) {
#ifdef LUX_HAS_AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_sub_ps(va, vb));
    }
    for (; i < n; i++) {
        out[i] = a[i] - b[i];
    }
#elif defined(LUX_HAS_NEON)
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vsubq_f32(va, vb));
    }
    for (; i < n; i++) {
        out[i] = a[i] - b[i];
    }
#else
    for (size_t i = 0; i < n; i++) {
        out[i] = a[i] - b[i];
    }
#endif
}

void lux_cpu_mul_f32(float* out, const float* a, const float* b, size_t n) {
#ifdef LUX_HAS_AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(va, vb));
    }
    for (; i < n; i++) {
        out[i] = a[i] * b[i];
    }
#elif defined(LUX_HAS_NEON)
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vmulq_f32(va, vb));
    }
    for (; i < n; i++) {
        out[i] = a[i] * b[i];
    }
#else
    for (size_t i = 0; i < n; i++) {
        out[i] = a[i] * b[i];
    }
#endif
}

void lux_cpu_div_f32(float* out, const float* a, const float* b, size_t n) {
#ifdef LUX_HAS_AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_div_ps(va, vb));
    }
    for (; i < n; i++) {
        out[i] = a[i] / b[i];
    }
#elif defined(LUX_HAS_NEON)
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vdivq_f32(va, vb));
    }
    for (; i < n; i++) {
        out[i] = a[i] / b[i];
    }
#else
    for (size_t i = 0; i < n; i++) {
        out[i] = a[i] / b[i];
    }
#endif
}

// =============================================================================
// Unary Operations
// =============================================================================

void lux_cpu_exp_f32(float* out, const float* in, size_t n) {
    for (size_t i = 0; i < n; i++) {
        out[i] = expf(in[i]);
    }
}

void lux_cpu_log_f32(float* out, const float* in, size_t n) {
    for (size_t i = 0; i < n; i++) {
        out[i] = logf(in[i]);
    }
}

void lux_cpu_sqrt_f32(float* out, const float* in, size_t n) {
#ifdef LUX_HAS_AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(in + i);
        _mm256_storeu_ps(out + i, _mm256_sqrt_ps(va));
    }
    for (; i < n; i++) {
        out[i] = sqrtf(in[i]);
    }
#elif defined(LUX_HAS_NEON)
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(in + i);
        vst1q_f32(out + i, vsqrtq_f32(va));
    }
    for (; i < n; i++) {
        out[i] = sqrtf(in[i]);
    }
#else
    for (size_t i = 0; i < n; i++) {
        out[i] = sqrtf(in[i]);
    }
#endif
}

void lux_cpu_neg_f32(float* out, const float* in, size_t n) {
#ifdef LUX_HAS_AVX2
    size_t i = 0;
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(in + i);
        _mm256_storeu_ps(out + i, _mm256_xor_ps(va, sign_mask));
    }
    for (; i < n; i++) {
        out[i] = -in[i];
    }
#elif defined(LUX_HAS_NEON)
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(in + i);
        vst1q_f32(out + i, vnegq_f32(va));
    }
    for (; i < n; i++) {
        out[i] = -in[i];
    }
#else
    for (size_t i = 0; i < n; i++) {
        out[i] = -in[i];
    }
#endif
}

void lux_cpu_abs_f32(float* out, const float* in, size_t n) {
#ifdef LUX_HAS_AVX2
    size_t i = 0;
    __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(in + i);
        _mm256_storeu_ps(out + i, _mm256_and_ps(va, abs_mask));
    }
    for (; i < n; i++) {
        out[i] = fabsf(in[i]);
    }
#elif defined(LUX_HAS_NEON)
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(in + i);
        vst1q_f32(out + i, vabsq_f32(va));
    }
    for (; i < n; i++) {
        out[i] = fabsf(in[i]);
    }
#else
    for (size_t i = 0; i < n; i++) {
        out[i] = fabsf(in[i]);
    }
#endif
}

void lux_cpu_tanh_f32(float* out, const float* in, size_t n) {
    for (size_t i = 0; i < n; i++) {
        out[i] = tanhf(in[i]);
    }
}

void lux_cpu_sigmoid_f32(float* out, const float* in, size_t n) {
    for (size_t i = 0; i < n; i++) {
        out[i] = 1.0f / (1.0f + expf(-in[i]));
    }
}

void lux_cpu_relu_f32(float* out, const float* in, size_t n) {
#ifdef LUX_HAS_AVX2
    size_t i = 0;
    __m256 zero = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(in + i);
        _mm256_storeu_ps(out + i, _mm256_max_ps(va, zero));
    }
    for (; i < n; i++) {
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    }
#elif defined(LUX_HAS_NEON)
    size_t i = 0;
    float32x4_t zero = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(in + i);
        vst1q_f32(out + i, vmaxq_f32(va, zero));
    }
    for (; i < n; i++) {
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    }
#else
    for (size_t i = 0; i < n; i++) {
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    }
#endif
}

// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
void lux_cpu_gelu_f32(float* out, const float* in, size_t n) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;

    for (size_t i = 0; i < n; i++) {
        float x = in[i];
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x3);
        out[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

// =============================================================================
// Matrix Operations
// =============================================================================

void lux_cpu_matmul_f32(float* c, const float* a, const float* b, int M, int K, int N) {
    // Blocked matrix multiplication for better cache performance
    const int BLOCK_M = 64;
    const int BLOCK_N = 64;
    const int BLOCK_K = 64;

    // Initialize C to zero
    memset(c, 0, (size_t)M * N * sizeof(float));

    for (int ii = 0; ii < M; ii += BLOCK_M) {
        int iend = ii + BLOCK_M < M ? ii + BLOCK_M : M;
        for (int jj = 0; jj < N; jj += BLOCK_N) {
            int jend = jj + BLOCK_N < N ? jj + BLOCK_N : N;
            for (int kk = 0; kk < K; kk += BLOCK_K) {
                int kend = kk + BLOCK_K < K ? kk + BLOCK_K : K;

                // Micro-kernel
                for (int i = ii; i < iend; i++) {
                    for (int k = kk; k < kend; k++) {
                        float aik = a[i * K + k];
#ifdef LUX_HAS_AVX2
                        __m256 va = _mm256_set1_ps(aik);
                        int j = jj;
                        for (; j + 8 <= jend; j += 8) {
                            __m256 vb = _mm256_loadu_ps(b + k * N + j);
                            __m256 vc = _mm256_loadu_ps(c + i * N + j);
                            _mm256_storeu_ps(c + i * N + j, _mm256_fmadd_ps(va, vb, vc));
                        }
                        for (; j < jend; j++) {
                            c[i * N + j] += aik * b[k * N + j];
                        }
#else
                        for (int j = jj; j < jend; j++) {
                            c[i * N + j] += aik * b[k * N + j];
                        }
#endif
                    }
                }
            }
        }
    }
}

void lux_cpu_transpose_f32(float* out, const float* in, int rows, int cols) {
    // Simple transpose with blocking for cache efficiency
    const int BLOCK = 16;

    for (int ii = 0; ii < rows; ii += BLOCK) {
        int iend = ii + BLOCK < rows ? ii + BLOCK : rows;
        for (int jj = 0; jj < cols; jj += BLOCK) {
            int jend = jj + BLOCK < cols ? jj + BLOCK : cols;

            for (int i = ii; i < iend; i++) {
                for (int j = jj; j < jend; j++) {
                    out[j * rows + i] = in[i * cols + j];
                }
            }
        }
    }
}

// =============================================================================
// Reduction Operations
// =============================================================================

float lux_cpu_reduce_sum_f32(const float* in, size_t n) {
#ifdef LUX_HAS_AVX2
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        sum = _mm256_add_ps(sum, _mm256_loadu_ps(in + i));
    }
    // Horizontal sum
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float result = _mm_cvtss_f32(lo);
    for (; i < n; i++) {
        result += in[i];
    }
    return result;
#else
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += in[i];
    }
    return sum;
#endif
}

float lux_cpu_reduce_max_f32(const float* in, size_t n) {
    if (n == 0) return -FLT_MAX;

#ifdef LUX_HAS_AVX2
    __m256 max_val = _mm256_set1_ps(-FLT_MAX);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        max_val = _mm256_max_ps(max_val, _mm256_loadu_ps(in + i));
    }
    // Horizontal max
    __m128 lo = _mm256_castps256_ps128(max_val);
    __m128 hi = _mm256_extractf128_ps(max_val, 1);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, 0x4E));
    lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, 0xB1));
    float result = _mm_cvtss_f32(lo);
    for (; i < n; i++) {
        if (in[i] > result) result = in[i];
    }
    return result;
#else
    float max_val = in[0];
    for (size_t i = 1; i < n; i++) {
        if (in[i] > max_val) max_val = in[i];
    }
    return max_val;
#endif
}

float lux_cpu_reduce_min_f32(const float* in, size_t n) {
    if (n == 0) return FLT_MAX;

#ifdef LUX_HAS_AVX2
    __m256 min_val = _mm256_set1_ps(FLT_MAX);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        min_val = _mm256_min_ps(min_val, _mm256_loadu_ps(in + i));
    }
    // Horizontal min
    __m128 lo = _mm256_castps256_ps128(min_val);
    __m128 hi = _mm256_extractf128_ps(min_val, 1);
    lo = _mm_min_ps(lo, hi);
    lo = _mm_min_ps(lo, _mm_shuffle_ps(lo, lo, 0x4E));
    lo = _mm_min_ps(lo, _mm_shuffle_ps(lo, lo, 0xB1));
    float result = _mm_cvtss_f32(lo);
    for (; i < n; i++) {
        if (in[i] < result) result = in[i];
    }
    return result;
#else
    float min_val = in[0];
    for (size_t i = 1; i < n; i++) {
        if (in[i] < min_val) min_val = in[i];
    }
    return min_val;
#endif
}

float lux_cpu_reduce_mean_f32(const float* in, size_t n) {
    if (n == 0) return 0.0f;
    return lux_cpu_reduce_sum_f32(in, n) / (float)n;
}

void lux_cpu_reduce_sum_axis_f32(float* out, const float* in, size_t outer_size, size_t inner_size) {
    for (size_t i = 0; i < outer_size; i++) {
        out[i] = lux_cpu_reduce_sum_f32(in + i * inner_size, inner_size);
    }
}

void lux_cpu_reduce_max_axis_f32(float* out, const float* in, size_t outer_size, size_t inner_size) {
    for (size_t i = 0; i < outer_size; i++) {
        out[i] = lux_cpu_reduce_max_f32(in + i * inner_size, inner_size);
    }
}

void lux_cpu_reduce_mean_axis_f32(float* out, const float* in, size_t outer_size, size_t inner_size) {
    for (size_t i = 0; i < outer_size; i++) {
        out[i] = lux_cpu_reduce_mean_f32(in + i * inner_size, inner_size);
    }
}

// =============================================================================
// Softmax Operations
// =============================================================================

void lux_cpu_softmax_f32(float* out, const float* in, size_t batch_size, size_t dim) {
    for (size_t b = 0; b < batch_size; b++) {
        const float* x = in + b * dim;
        float* y = out + b * dim;

        // Find max for numerical stability
        float max_val = lux_cpu_reduce_max_f32(x, dim);

        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            float exp_val = expf(x[i] - max_val);
            y[i] = exp_val;
            sum += exp_val;
        }

        // Normalize
        float inv_sum = 1.0f / sum;
        for (size_t i = 0; i < dim; i++) {
            y[i] *= inv_sum;
        }
    }
}

void lux_cpu_log_softmax_f32(float* out, const float* in, size_t batch_size, size_t dim) {
    for (size_t b = 0; b < batch_size; b++) {
        const float* x = in + b * dim;
        float* y = out + b * dim;

        // Find max
        float max_val = lux_cpu_reduce_max_f32(x, dim);

        // Compute sum(exp(x - max))
        float sum = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            sum += expf(x[i] - max_val);
        }
        float log_sum = logf(sum);

        // Compute log softmax
        for (size_t i = 0; i < dim; i++) {
            y[i] = x[i] - max_val - log_sum;
        }
    }
}

// =============================================================================
// Normalization Operations
// =============================================================================

void lux_cpu_layer_norm_f32(float* out, const float* in, const float* gamma, const float* beta,
                            size_t batch_size, size_t dim, float eps) {
    for (size_t b = 0; b < batch_size; b++) {
        const float* x = in + b * dim;
        float* y = out + b * dim;

        // Compute mean
        float mean = lux_cpu_reduce_mean_f32(x, dim);

        // Compute variance
        float var = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            float diff = x[i] - mean;
            var += diff * diff;
        }
        var /= (float)dim;

        // Normalize and scale
        float inv_std = 1.0f / sqrtf(var + eps);
        for (size_t i = 0; i < dim; i++) {
            float normalized = (x[i] - mean) * inv_std;
            y[i] = normalized * gamma[i] + beta[i];
        }
    }
}

void lux_cpu_rms_norm_f32(float* out, const float* in, const float* weight,
                          size_t batch_size, size_t dim, float eps) {
    for (size_t b = 0; b < batch_size; b++) {
        const float* x = in + b * dim;
        float* y = out + b * dim;

        // Compute sum of squares
        float sum_sq = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            sum_sq += x[i] * x[i];
        }

        // RMS scale
        float rms_scale = 1.0f / sqrtf(sum_sq / (float)dim + eps);

        // Scale
        for (size_t i = 0; i < dim; i++) {
            y[i] = x[i] * rms_scale * weight[i];
        }
    }
}

// =============================================================================
// Copy
// =============================================================================

void lux_cpu_copy_f32(float* dst, const float* src, size_t n) {
    memcpy(dst, src, n * sizeof(float));
}
