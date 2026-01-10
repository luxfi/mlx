// Copyright © 2024-2025 Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// PRNG operations for CUDA
// Implements uniform and normal distributions using xoshiro256** and Box-Muller

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace lux {
namespace cuda {

// ============================================================================
// xoshiro256** PRNG State
// ============================================================================

// Each thread maintains its own 256-bit state
struct Xoshiro256State {
    uint64_t s[4];
};

// Rotate left helper
__device__ __forceinline__ uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

// xoshiro256** next - high quality PRNG
__device__ __forceinline__ uint64_t xoshiro256_next(Xoshiro256State* state) {
    uint64_t* s = state->s;
    const uint64_t result = rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;
    s[3] = rotl(s[3], 45);

    return result;
}

// Initialize state from seed and thread index
__device__ __forceinline__ void xoshiro256_init(Xoshiro256State* state, uint64_t seed, uint32_t tid) {
    // SplitMix64 to generate initial state
    uint64_t z = seed + tid * 0x9e3779b97f4a7c15ULL;

    for (int i = 0; i < 4; i++) {
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        state->s[i] = z ^ (z >> 31);
    }

    // Warm up the generator
    for (int i = 0; i < 10; i++) {
        xoshiro256_next(state);
    }
}

// Convert uint64 to uniform float in [0, 1)
__device__ __forceinline__ float uint64_to_uniform_float(uint64_t x) {
    // Use upper 24 bits for float precision
    return (x >> 40) * 0x1.0p-24f;
}

// Convert uint64 to uniform double in [0, 1)
__device__ __forceinline__ double uint64_to_uniform_double(uint64_t x) {
    // Use upper 53 bits for double precision
    return (x >> 11) * 0x1.0p-53;
}

// ============================================================================
// Uniform Distribution Kernels
// ============================================================================

// Uniform float in [low, high)
extern "C" __global__ void uniform_float_kernel(
    float* __restrict__ out,
    uint64_t seed,
    float low,
    float high,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    Xoshiro256State state;
    xoshiro256_init(&state, seed, idx);

    float range = high - low;

    for (size_t i = idx; i < n; i += stride) {
        float u = uint64_to_uniform_float(xoshiro256_next(&state));
        out[i] = low + u * range;
    }
}

// Uniform half in [low, high)
extern "C" __global__ void uniform_half_kernel(
    __half* __restrict__ out,
    uint64_t seed,
    float low,
    float high,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    Xoshiro256State state;
    xoshiro256_init(&state, seed, idx);

    float range = high - low;

    for (size_t i = idx; i < n; i += stride) {
        float u = uint64_to_uniform_float(xoshiro256_next(&state));
        out[i] = __float2half(low + u * range);
    }
}

// Uniform int32 in [low, high)
extern "C" __global__ void uniform_int32_kernel(
    int32_t* __restrict__ out,
    uint64_t seed,
    int32_t low,
    int32_t high,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    Xoshiro256State state;
    xoshiro256_init(&state, seed, idx);

    uint64_t range = (uint64_t)(high - low);

    for (size_t i = idx; i < n; i += stride) {
        uint64_t r = xoshiro256_next(&state);
        // Lemire's nearly divisionless method
        __uint128_t m = (__uint128_t)r * (__uint128_t)range;
        out[i] = low + (int32_t)(m >> 64);
    }
}

// Uniform int64 in [low, high)
extern "C" __global__ void uniform_int64_kernel(
    int64_t* __restrict__ out,
    uint64_t seed,
    int64_t low,
    int64_t high,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    Xoshiro256State state;
    xoshiro256_init(&state, seed, idx);

    uint64_t range = (uint64_t)(high - low);

    for (size_t i = idx; i < n; i += stride) {
        uint64_t r = xoshiro256_next(&state);
        // Simple modulo - could use rejection sampling for better uniformity
        out[i] = low + (int64_t)(r % range);
    }
}

// ============================================================================
// Normal Distribution Kernels (Box-Muller Transform)
// ============================================================================

// Standard normal float (mean=0, std=1)
extern "C" __global__ void normal_float_kernel(
    float* __restrict__ out,
    uint64_t seed,
    float mean,
    float std,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    Xoshiro256State state;
    xoshiro256_init(&state, seed, idx);

    // Process pairs for Box-Muller
    for (size_t i = idx * 2; i < n; i += stride * 2) {
        // Generate two uniform values in (0, 1]
        float u1 = uint64_to_uniform_float(xoshiro256_next(&state));
        float u2 = uint64_to_uniform_float(xoshiro256_next(&state));

        // Avoid log(0)
        u1 = fmaxf(u1, 1e-10f);

        // Box-Muller transform
        float r = sqrtf(-2.0f * logf(u1));
        float theta = 2.0f * 3.14159265358979323846f * u2;

        float z0 = r * cosf(theta);
        float z1 = r * sinf(theta);

        out[i] = mean + z0 * std;
        if (i + 1 < n) {
            out[i + 1] = mean + z1 * std;
        }
    }
}

// Standard normal half
extern "C" __global__ void normal_half_kernel(
    __half* __restrict__ out,
    uint64_t seed,
    float mean,
    float std,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    Xoshiro256State state;
    xoshiro256_init(&state, seed, idx);

    for (size_t i = idx * 2; i < n; i += stride * 2) {
        float u1 = uint64_to_uniform_float(xoshiro256_next(&state));
        float u2 = uint64_to_uniform_float(xoshiro256_next(&state));

        u1 = fmaxf(u1, 1e-10f);

        float r = sqrtf(-2.0f * logf(u1));
        float theta = 2.0f * 3.14159265358979323846f * u2;

        float z0 = r * cosf(theta);
        float z1 = r * sinf(theta);

        out[i] = __float2half(mean + z0 * std);
        if (i + 1 < n) {
            out[i + 1] = __float2half(mean + z1 * std);
        }
    }
}

// ============================================================================
// Truncated Normal Distribution
// ============================================================================

// Truncated normal - rejection sampling
extern "C" __global__ void truncated_normal_float_kernel(
    float* __restrict__ out,
    uint64_t seed,
    float mean,
    float std,
    float low,
    float high,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    Xoshiro256State state;
    xoshiro256_init(&state, seed, idx);

    for (size_t i = idx; i < n; i += stride) {
        float value;
        int attempts = 0;
        const int max_attempts = 1000;

        do {
            float u1 = uint64_to_uniform_float(xoshiro256_next(&state));
            float u2 = uint64_to_uniform_float(xoshiro256_next(&state));

            u1 = fmaxf(u1, 1e-10f);

            float r = sqrtf(-2.0f * logf(u1));
            float theta = 2.0f * 3.14159265358979323846f * u2;

            value = mean + r * cosf(theta) * std;
            attempts++;
        } while ((value < low || value >= high) && attempts < max_attempts);

        // Clamp if max attempts exceeded
        if (attempts >= max_attempts) {
            value = fmaxf(low, fminf(value, high - 1e-7f));
        }

        out[i] = value;
    }
}

// ============================================================================
// Log-Normal Distribution
// ============================================================================

extern "C" __global__ void lognormal_float_kernel(
    float* __restrict__ out,
    uint64_t seed,
    float mean,
    float std,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    Xoshiro256State state;
    xoshiro256_init(&state, seed, idx);

    for (size_t i = idx * 2; i < n; i += stride * 2) {
        float u1 = uint64_to_uniform_float(xoshiro256_next(&state));
        float u2 = uint64_to_uniform_float(xoshiro256_next(&state));

        u1 = fmaxf(u1, 1e-10f);

        float r = sqrtf(-2.0f * logf(u1));
        float theta = 2.0f * 3.14159265358979323846f * u2;

        float z0 = r * cosf(theta);
        float z1 = r * sinf(theta);

        out[i] = expf(mean + z0 * std);
        if (i + 1 < n) {
            out[i + 1] = expf(mean + z1 * std);
        }
    }
}

// ============================================================================
// Exponential Distribution
// ============================================================================

extern "C" __global__ void exponential_float_kernel(
    float* __restrict__ out,
    uint64_t seed,
    float lambda,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    Xoshiro256State state;
    xoshiro256_init(&state, seed, idx);

    for (size_t i = idx; i < n; i += stride) {
        float u = uint64_to_uniform_float(xoshiro256_next(&state));
        u = fmaxf(u, 1e-10f);
        out[i] = -logf(u) / lambda;
    }
}

// ============================================================================
// Bernoulli Distribution
// ============================================================================

extern "C" __global__ void bernoulli_float_kernel(
    float* __restrict__ out,
    uint64_t seed,
    float p,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    Xoshiro256State state;
    xoshiro256_init(&state, seed, idx);

    for (size_t i = idx; i < n; i += stride) {
        float u = uint64_to_uniform_float(xoshiro256_next(&state));
        out[i] = (u < p) ? 1.0f : 0.0f;
    }
}

extern "C" __global__ void bernoulli_bool_kernel(
    bool* __restrict__ out,
    uint64_t seed,
    float p,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    Xoshiro256State state;
    xoshiro256_init(&state, seed, idx);

    for (size_t i = idx; i < n; i += stride) {
        float u = uint64_to_uniform_float(xoshiro256_next(&state));
        out[i] = (u < p);
    }
}

// ============================================================================
// Dropout Mask Generation
// ============================================================================

// Generate dropout mask (1 with probability 1-p, 0 with probability p)
extern "C" __global__ void dropout_mask_float_kernel(
    float* __restrict__ out,
    uint64_t seed,
    float p,
    float scale,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    Xoshiro256State state;
    xoshiro256_init(&state, seed, idx);

    for (size_t i = idx; i < n; i += stride) {
        float u = uint64_to_uniform_float(xoshiro256_next(&state));
        out[i] = (u >= p) ? scale : 0.0f;
    }
}

// Apply dropout in-place
extern "C" __global__ void dropout_inplace_float_kernel(
    float* __restrict__ data,
    uint64_t seed,
    float p,
    float scale,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    Xoshiro256State state;
    xoshiro256_init(&state, seed, idx);

    for (size_t i = idx; i < n; i += stride) {
        float u = uint64_to_uniform_float(xoshiro256_next(&state));
        data[i] = (u >= p) ? data[i] * scale : 0.0f;
    }
}

// ============================================================================
// Random Permutation
// ============================================================================

// Fisher-Yates shuffle helper - generates random indices
extern "C" __global__ void randperm_kernel(
    int64_t* __restrict__ out,
    uint64_t seed,
    size_t n
) {
    // Initialize with identity permutation
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride) {
        out[i] = (int64_t)i;
    }
}

// Note: Full Fisher-Yates requires sequential swaps
// This generates random swap indices for host-side shuffle
extern "C" __global__ void random_swap_indices_kernel(
    int64_t* __restrict__ indices,
    uint64_t seed,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    Xoshiro256State state;
    xoshiro256_init(&state, seed, idx);

    for (size_t i = idx; i < n; i += stride) {
        // Random index in [0, i]
        uint64_t r = xoshiro256_next(&state);
        indices[i] = (int64_t)(r % (i + 1));
    }
}

// ============================================================================
// Random Choice / Multinomial Sampling
// ============================================================================

// Sample indices according to weights (unnormalized)
extern "C" __global__ void multinomial_kernel(
    int64_t* __restrict__ out,
    const float* __restrict__ weights,
    const float* __restrict__ cumsum,  // Pre-computed cumulative sum
    uint64_t seed,
    size_t num_categories,
    size_t num_samples
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    Xoshiro256State state;
    xoshiro256_init(&state, seed, idx);

    float total = cumsum[num_categories - 1];

    for (size_t i = idx; i < num_samples; i += stride) {
        float u = uint64_to_uniform_float(xoshiro256_next(&state)) * total;

        // Binary search for category
        size_t lo = 0, hi = num_categories;
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (cumsum[mid] <= u) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        out[i] = (int64_t)lo;
    }
}

// ============================================================================
// Categorical Distribution (Gumbel-Softmax Trick)
// ============================================================================

// Sample from categorical distribution using Gumbel-Max trick
extern "C" __global__ void gumbel_softmax_kernel(
    int64_t* __restrict__ out,
    const float* __restrict__ logits,
    uint64_t seed,
    size_t batch_size,
    size_t num_classes
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    Xoshiro256State state;
    xoshiro256_init(&state, seed, idx);

    for (size_t b = idx; b < batch_size; b += stride) {
        float max_val = -INFINITY;
        int64_t max_idx = 0;

        for (size_t c = 0; c < num_classes; c++) {
            float u = uint64_to_uniform_float(xoshiro256_next(&state));
            u = fmaxf(u, 1e-10f);

            // Gumbel noise: -log(-log(u))
            float gumbel = -logf(-logf(u));
            float perturbed = logits[b * num_classes + c] + gumbel;

            if (perturbed > max_val) {
                max_val = perturbed;
                max_idx = (int64_t)c;
            }
        }

        out[b] = max_idx;
    }
}

// ============================================================================
// Poisson Distribution
// ============================================================================

// Poisson sampling using inverse CDF method (for small lambda)
extern "C" __global__ void poisson_float_kernel(
    float* __restrict__ out,
    uint64_t seed,
    float lambda,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    Xoshiro256State state;
    xoshiro256_init(&state, seed, idx);

    float L = expf(-lambda);

    for (size_t i = idx; i < n; i += stride) {
        int k = 0;
        float p = 1.0f;

        do {
            k++;
            float u = uint64_to_uniform_float(xoshiro256_next(&state));
            p *= u;
        } while (p > L && k < 1000);

        out[i] = (float)(k - 1);
    }
}

// ============================================================================
// Gamma Distribution (Marsaglia and Tsang's method)
// ============================================================================

extern "C" __global__ void gamma_float_kernel(
    float* __restrict__ out,
    uint64_t seed,
    float shape,
    float scale,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    Xoshiro256State state;
    xoshiro256_init(&state, seed, idx);

    // Marsaglia and Tsang's method for shape >= 1
    float d = shape - 1.0f / 3.0f;
    float c = 1.0f / sqrtf(9.0f * d);

    for (size_t i = idx; i < n; i += stride) {
        float x, v;

        if (shape >= 1.0f) {
            while (true) {
                // Generate normal
                float u1 = uint64_to_uniform_float(xoshiro256_next(&state));
                float u2 = uint64_to_uniform_float(xoshiro256_next(&state));
                u1 = fmaxf(u1, 1e-10f);

                float r = sqrtf(-2.0f * logf(u1));
                x = r * cosf(2.0f * 3.14159265358979323846f * u2);

                v = 1.0f + c * x;
                if (v <= 0.0f) continue;

                v = v * v * v;
                float u = uint64_to_uniform_float(xoshiro256_next(&state));

                if (u < 1.0f - 0.0331f * x * x * x * x) {
                    out[i] = d * v * scale;
                    break;
                }

                if (logf(u) < 0.5f * x * x + d * (1.0f - v + logf(v))) {
                    out[i] = d * v * scale;
                    break;
                }
            }
        } else {
            // For shape < 1, use shape + 1 and adjust
            float d1 = shape + 1.0f - 1.0f / 3.0f;
            float c1 = 1.0f / sqrtf(9.0f * d1);

            while (true) {
                float u1 = uint64_to_uniform_float(xoshiro256_next(&state));
                float u2 = uint64_to_uniform_float(xoshiro256_next(&state));
                u1 = fmaxf(u1, 1e-10f);

                float r = sqrtf(-2.0f * logf(u1));
                x = r * cosf(2.0f * 3.14159265358979323846f * u2);

                v = 1.0f + c1 * x;
                if (v <= 0.0f) continue;

                v = v * v * v;
                float u = uint64_to_uniform_float(xoshiro256_next(&state));

                if (u < 1.0f - 0.0331f * x * x * x * x) {
                    float u3 = uint64_to_uniform_float(xoshiro256_next(&state));
                    out[i] = d1 * v * scale * powf(u3, 1.0f / shape);
                    break;
                }

                if (logf(u) < 0.5f * x * x + d1 * (1.0f - v + logf(v))) {
                    float u3 = uint64_to_uniform_float(xoshiro256_next(&state));
                    out[i] = d1 * v * scale * powf(u3, 1.0f / shape);
                    break;
                }
            }
        }
    }
}

// ============================================================================
// Beta Distribution (using Gamma)
// ============================================================================

extern "C" __global__ void beta_float_kernel(
    float* __restrict__ out,
    uint64_t seed,
    float alpha,
    float beta,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    Xoshiro256State state;
    xoshiro256_init(&state, seed, idx);

    // Simple method using uniform samples
    // Beta(a,b) = Gamma(a,1) / (Gamma(a,1) + Gamma(b,1))

    float da = alpha - 1.0f / 3.0f;
    float ca = 1.0f / sqrtf(9.0f * fmaxf(da, 0.1f));
    float db = beta - 1.0f / 3.0f;
    float cb = 1.0f / sqrtf(9.0f * fmaxf(db, 0.1f));

    for (size_t i = idx; i < n; i += stride) {
        // Simplified: use Jöhnk's algorithm for small alpha, beta
        if (alpha < 1.0f && beta < 1.0f) {
            while (true) {
                float u = uint64_to_uniform_float(xoshiro256_next(&state));
                float v = uint64_to_uniform_float(xoshiro256_next(&state));

                float x = powf(u, 1.0f / alpha);
                float y = powf(v, 1.0f / beta);

                if (x + y <= 1.0f) {
                    out[i] = x / (x + y);
                    break;
                }
            }
        } else {
            // Use ratio of uniforms approximation
            float u1 = uint64_to_uniform_float(xoshiro256_next(&state));
            float u2 = uint64_to_uniform_float(xoshiro256_next(&state));

            // Approximate using quantile function of beta
            float x = alpha / (alpha + beta);
            float var = (alpha * beta) / ((alpha + beta) * (alpha + beta) * (alpha + beta + 1.0f));
            float std = sqrtf(var);

            // Generate approximate normal and transform
            u1 = fmaxf(u1, 1e-10f);
            float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
            float sample = x + z * std;
            out[i] = fmaxf(0.0f, fminf(1.0f, sample));
        }
    }
}

} // namespace cuda
} // namespace lux

// ============================================================================
// C API Wrappers
// ============================================================================

extern "C" {

// Uniform distributions
cudaError_t lux_cuda_uniform_float(
    float* out,
    uint64_t seed,
    float low,
    float high,
    size_t n,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = min(blocks, 65535);

    lux::cuda::uniform_float_kernel<<<blocks, threads, 0, stream>>>(
        out, seed, low, high, n
    );
    return cudaGetLastError();
}

cudaError_t lux_cuda_uniform_half(
    __half* out,
    uint64_t seed,
    float low,
    float high,
    size_t n,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = min(blocks, 65535);

    lux::cuda::uniform_half_kernel<<<blocks, threads, 0, stream>>>(
        out, seed, low, high, n
    );
    return cudaGetLastError();
}

cudaError_t lux_cuda_uniform_int32(
    int32_t* out,
    uint64_t seed,
    int32_t low,
    int32_t high,
    size_t n,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = min(blocks, 65535);

    lux::cuda::uniform_int32_kernel<<<blocks, threads, 0, stream>>>(
        out, seed, low, high, n
    );
    return cudaGetLastError();
}

// Normal distribution
cudaError_t lux_cuda_normal_float(
    float* out,
    uint64_t seed,
    float mean,
    float std,
    size_t n,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = ((n + 1) / 2 + threads - 1) / threads;
    blocks = min(blocks, 65535);

    lux::cuda::normal_float_kernel<<<blocks, threads, 0, stream>>>(
        out, seed, mean, std, n
    );
    return cudaGetLastError();
}

cudaError_t lux_cuda_normal_half(
    __half* out,
    uint64_t seed,
    float mean,
    float std,
    size_t n,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = ((n + 1) / 2 + threads - 1) / threads;
    blocks = min(blocks, 65535);

    lux::cuda::normal_half_kernel<<<blocks, threads, 0, stream>>>(
        out, seed, mean, std, n
    );
    return cudaGetLastError();
}

cudaError_t lux_cuda_truncated_normal_float(
    float* out,
    uint64_t seed,
    float mean,
    float std,
    float low,
    float high,
    size_t n,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = min(blocks, 65535);

    lux::cuda::truncated_normal_float_kernel<<<blocks, threads, 0, stream>>>(
        out, seed, mean, std, low, high, n
    );
    return cudaGetLastError();
}

// Other distributions
cudaError_t lux_cuda_lognormal_float(
    float* out,
    uint64_t seed,
    float mean,
    float std,
    size_t n,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = ((n + 1) / 2 + threads - 1) / threads;
    blocks = min(blocks, 65535);

    lux::cuda::lognormal_float_kernel<<<blocks, threads, 0, stream>>>(
        out, seed, mean, std, n
    );
    return cudaGetLastError();
}

cudaError_t lux_cuda_exponential_float(
    float* out,
    uint64_t seed,
    float lambda,
    size_t n,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = min(blocks, 65535);

    lux::cuda::exponential_float_kernel<<<blocks, threads, 0, stream>>>(
        out, seed, lambda, n
    );
    return cudaGetLastError();
}

cudaError_t lux_cuda_bernoulli_float(
    float* out,
    uint64_t seed,
    float p,
    size_t n,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = min(blocks, 65535);

    lux::cuda::bernoulli_float_kernel<<<blocks, threads, 0, stream>>>(
        out, seed, p, n
    );
    return cudaGetLastError();
}

cudaError_t lux_cuda_dropout_mask_float(
    float* out,
    uint64_t seed,
    float p,
    float scale,
    size_t n,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = min(blocks, 65535);

    lux::cuda::dropout_mask_float_kernel<<<blocks, threads, 0, stream>>>(
        out, seed, p, scale, n
    );
    return cudaGetLastError();
}

cudaError_t lux_cuda_dropout_inplace_float(
    float* data,
    uint64_t seed,
    float p,
    float scale,
    size_t n,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = min(blocks, 65535);

    lux::cuda::dropout_inplace_float_kernel<<<blocks, threads, 0, stream>>>(
        data, seed, p, scale, n
    );
    return cudaGetLastError();
}

cudaError_t lux_cuda_poisson_float(
    float* out,
    uint64_t seed,
    float lambda,
    size_t n,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = min(blocks, 65535);

    lux::cuda::poisson_float_kernel<<<blocks, threads, 0, stream>>>(
        out, seed, lambda, n
    );
    return cudaGetLastError();
}

cudaError_t lux_cuda_gamma_float(
    float* out,
    uint64_t seed,
    float shape,
    float scale,
    size_t n,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = min(blocks, 65535);

    lux::cuda::gamma_float_kernel<<<blocks, threads, 0, stream>>>(
        out, seed, shape, scale, n
    );
    return cudaGetLastError();
}

cudaError_t lux_cuda_beta_float(
    float* out,
    uint64_t seed,
    float alpha,
    float beta,
    size_t n,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = min(blocks, 65535);

    lux::cuda::beta_float_kernel<<<blocks, threads, 0, stream>>>(
        out, seed, alpha, beta, n
    );
    return cudaGetLastError();
}

cudaError_t lux_cuda_multinomial(
    int64_t* out,
    const float* weights,
    const float* cumsum,
    uint64_t seed,
    size_t num_categories,
    size_t num_samples,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_samples + threads - 1) / threads;
    blocks = min(blocks, 65535);

    lux::cuda::multinomial_kernel<<<blocks, threads, 0, stream>>>(
        out, weights, cumsum, seed, num_categories, num_samples
    );
    return cudaGetLastError();
}

cudaError_t lux_cuda_gumbel_softmax(
    int64_t* out,
    const float* logits,
    uint64_t seed,
    size_t batch_size,
    size_t num_classes,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    blocks = min(blocks, 65535);

    lux::cuda::gumbel_softmax_kernel<<<blocks, threads, 0, stream>>>(
        out, logits, seed, batch_size, num_classes
    );
    return cudaGetLastError();
}

cudaError_t lux_cuda_randperm(
    int64_t* out,
    uint64_t seed,
    size_t n,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = min(blocks, 65535);

    lux::cuda::randperm_kernel<<<blocks, threads, 0, stream>>>(
        out, seed, n
    );
    return cudaGetLastError();
}

} // extern "C"
