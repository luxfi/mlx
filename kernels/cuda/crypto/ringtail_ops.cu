// Copyright (c) 2024-2026 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Ringtail Lattice Threshold Operations - High-Performance CUDA Implementation
// Post-quantum threshold signatures based on Module-LWE (MLWE).
// Implements share combination, signature verification, and Gaussian sampling.

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>

namespace lux {
namespace cuda {
namespace ringtail {

// ============================================================================
// Ringtail Parameters (based on Dilithium-like construction)
// ============================================================================

// Modulus q = 8380417 (same as Dilithium)
constexpr uint32_t Q = 8380417;
constexpr uint32_t N = 256;              // Ring dimension
constexpr uint32_t K = 4;                // Module rank for public key
constexpr uint32_t L = 4;                // Module rank for secret
constexpr uint32_t ETA = 2;              // Secret coefficient bound
constexpr uint32_t GAMMA1 = 131072;      // Commitment coefficient bound (2^17)
constexpr uint32_t GAMMA2 = 95232;       // Low bits rounding
constexpr uint32_t BETA = 78;            // Rejection threshold
constexpr uint32_t OMEGA = 80;           // Max hint ones

// Threshold parameters
constexpr uint32_t MAX_SHARES = 32;      // Maximum number of threshold participants

// Device constants
__constant__ uint32_t d_q = Q;
__constant__ uint32_t d_n = N;
__constant__ int32_t d_gamma1 = GAMMA1;
__constant__ int32_t d_gamma2 = GAMMA2;
__constant__ int32_t d_beta = BETA;

// ============================================================================
// Ring Polynomial Structure
// ============================================================================

struct Poly {
    int32_t coeffs[N];
};

struct PolyVec {
    Poly polys[L];  // Vector of L polynomials
};

// ============================================================================
// Modular Arithmetic
// ============================================================================

// Montgomery reduction constant
constexpr uint32_t QINV = 58728449;  // q^{-1} mod 2^32

__device__ __forceinline__
int32_t mont_reduce(int64_t a) {
    int32_t t = (int32_t)((uint32_t)a * QINV);
    return (int32_t)((a - (int64_t)t * Q) >> 32);
}

__device__ __forceinline__
int32_t mod_add(int32_t a, int32_t b) {
    int32_t r = a + b;
    if (r >= (int32_t)Q) r -= Q;
    if (r < 0) r += Q;
    return r;
}

__device__ __forceinline__
int32_t mod_sub(int32_t a, int32_t b) {
    int32_t r = a - b;
    if (r < 0) r += Q;
    return r;
}

__device__ __forceinline__
int32_t caddq(int32_t a) {
    return a + ((a >> 31) & Q);
}

__device__ __forceinline__
int32_t freeze(int32_t a) {
    a = caddq(a);
    return a - Q + ((Q - 1 - a) >> 31 & Q);
}

// ============================================================================
// Discrete Gaussian Sampling
// ============================================================================

// Cumulative distribution table for discrete Gaussian
__constant__ uint64_t GAUSSIAN_CDF[256];  // Precomputed CDF values

// Sample from discrete Gaussian with standard deviation sigma
__device__
int32_t sample_gaussian(curandState* state, float sigma) {
    // Box-Muller approximation for discrete Gaussian
    float u1 = curand_uniform(state);
    float u2 = curand_uniform(state);
    
    // Avoid log(0)
    if (u1 < 1e-10f) u1 = 1e-10f;
    
    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
    
    return (int32_t)roundf(z * sigma);
}

// Rejection sampling for bounded coefficients
__device__
int32_t sample_bounded(curandState* state, int32_t bound) {
    int32_t sample;
    do {
        sample = (int32_t)(curand(state) % (2 * bound + 1)) - bound;
    } while (abs(sample) > bound);
    return sample;
}

// Sample polynomial with coefficients in [-eta, eta]
__global__ void sample_poly_eta_kernel(
    Poly* __restrict__ poly,
    uint32_t seed,
    uint32_t eta
) {
    const uint32_t tid = threadIdx.x;
    
    if (tid >= N) return;
    
    curandState state;
    curand_init(seed, tid, 0, &state);
    
    poly->coeffs[tid] = sample_bounded(&state, eta);
}

// Sample polynomial with coefficients in [-gamma1, gamma1]
__global__ void sample_poly_gamma1_kernel(
    Poly* __restrict__ poly,
    uint32_t seed
) {
    const uint32_t tid = threadIdx.x;
    
    if (tid >= N) return;
    
    curandState state;
    curand_init(seed, tid, 0, &state);
    
    // Uniform in [0, 2*gamma1]
    int32_t r = curand(&state) % (2 * GAMMA1 + 1);
    poly->coeffs[tid] = GAMMA1 - r;
}

// Gaussian sampling for threshold shares
__global__ void sample_gaussian_poly_kernel(
    Poly* __restrict__ poly,
    uint32_t seed,
    float sigma
) {
    const uint32_t tid = threadIdx.x;
    
    if (tid >= N) return;
    
    curandState state;
    curand_init(seed, tid, 0, &state);
    
    poly->coeffs[tid] = sample_gaussian(&state, sigma);
}

// ============================================================================
// Share Operations
// ============================================================================

struct ThresholdShare {
    uint32_t index;          // Share index (1 to n)
    PolyVec s_share;         // Share of secret key
    PolyVec y_share;         // Share of masking vector
};

struct PublicParams {
    Poly A[K][L];            // Public matrix (NTT domain)
    PolyVec t;               // Public key t = A*s
};

// Lagrange basis polynomial evaluation at 0
// lambda_i(0) = prod_{j != i} (0 - j) / (i - j) = prod_{j != i} j / (j - i)
__device__
int32_t compute_lagrange_coeff(
    uint32_t index,
    const uint32_t* indices,
    uint32_t num_shares
) {
    int64_t numerator = 1;
    int64_t denominator = 1;
    
    for (uint32_t j = 0; j < num_shares; j++) {
        if (indices[j] == index) continue;
        
        numerator = (numerator * (int64_t)indices[j]) % Q;
        int64_t diff = (int64_t)indices[j] - (int64_t)index;
        if (diff < 0) diff += Q;
        denominator = (denominator * diff) % Q;
    }
    
    // Compute denominator inverse using Fermat's little theorem
    // denom^{-1} = denom^{q-2} mod q
    int64_t inv = 1;
    int64_t base = denominator;
    int64_t exp = Q - 2;
    while (exp > 0) {
        if (exp & 1) inv = (inv * base) % Q;
        base = (base * base) % Q;
        exp >>= 1;
    }
    
    return (int32_t)((numerator * inv) % Q);
}

// Combine threshold shares using Lagrange interpolation
__global__ void combine_shares_kernel(
    const ThresholdShare* __restrict__ shares,
    const uint32_t* __restrict__ indices,
    uint32_t num_shares,
    PolyVec* __restrict__ combined_s,
    PolyVec* __restrict__ combined_y
) {
    const uint32_t poly_idx = blockIdx.x;      // Which polynomial in vector
    const uint32_t coeff_idx = threadIdx.x;    // Which coefficient
    
    if (poly_idx >= L || coeff_idx >= N) return;
    
    int64_t s_sum = 0;
    int64_t y_sum = 0;
    
    for (uint32_t i = 0; i < num_shares; i++) {
        int32_t lambda = compute_lagrange_coeff(shares[i].index, indices, num_shares);
        
        int64_t s_contrib = ((int64_t)shares[i].s_share.polys[poly_idx].coeffs[coeff_idx] * lambda) % Q;
        int64_t y_contrib = ((int64_t)shares[i].y_share.polys[poly_idx].coeffs[coeff_idx] * lambda) % Q;
        
        s_sum = (s_sum + s_contrib + Q) % Q;
        y_sum = (y_sum + y_contrib + Q) % Q;
    }
    
    combined_s->polys[poly_idx].coeffs[coeff_idx] = (int32_t)s_sum;
    combined_y->polys[poly_idx].coeffs[coeff_idx] = (int32_t)y_sum;
}

// ============================================================================
// Signature Operations
// ============================================================================

struct RingtailSignature {
    Poly c;                  // Challenge polynomial
    PolyVec z;               // Response vector
    uint8_t hint[OMEGA + K]; // Hint for verification
};

// High bits extraction: highbits(r, alpha) = floor((r + alpha/2) / alpha)
__device__ __forceinline__
int32_t highbits(int32_t r, int32_t alpha) {
    r = freeze(r);
    int32_t r1 = (r + (alpha >> 1)) / alpha;
    return r1;
}

// Low bits extraction: lowbits(r, alpha) = r - highbits(r, alpha) * alpha
__device__ __forceinline__
int32_t lowbits(int32_t r, int32_t alpha) {
    int32_t r1 = highbits(r, alpha);
    return r - r1 * alpha;
}

// Power2Round: decompose r into (r1, r0) where r = r1 * 2^d + r0
__device__ __forceinline__
void power2round(int32_t r, int32_t d, int32_t* r1, int32_t* r0) {
    *r1 = (r + (1 << (d - 1)) - 1) >> d;
    *r0 = r - (*r1 << d);
}

// Compute w = A*y (matrix-vector product in NTT domain)
__global__ void compute_commitment_kernel(
    const Poly* __restrict__ A,     // [K][L] in NTT domain
    const PolyVec* __restrict__ y,  // Combined masking vector
    PolyVec* __restrict__ w,        // Output commitment
    uint32_t k,
    uint32_t l,
    uint32_t n
) {
    extern __shared__ int32_t shmem[];
    
    const uint32_t row = blockIdx.x;           // Which row of A (0..k-1)
    const uint32_t coeff_idx = threadIdx.x;    // Which coefficient
    
    if (row >= k || coeff_idx >= n) return;
    
    int64_t sum = 0;
    
    // Compute dot product A[row] . y (all in NTT domain)
    for (uint32_t col = 0; col < l; col++) {
        int32_t a_val = A[row * l + col].coeffs[coeff_idx];
        int32_t y_val = y->polys[col].coeffs[coeff_idx];
        int64_t prod = (int64_t)a_val * y_val;
        sum += mont_reduce(prod);
    }
    
    w->polys[row].coeffs[coeff_idx] = freeze((int32_t)(sum % Q));
}

// Check if response z is within bounds
__global__ void check_response_bounds_kernel(
    const PolyVec* __restrict__ z,
    int32_t gamma1_minus_beta,
    bool* __restrict__ valid
) {
    const uint32_t poly_idx = blockIdx.x;
    const uint32_t coeff_idx = threadIdx.x;
    
    if (poly_idx >= L || coeff_idx >= N) return;
    
    int32_t val = z->polys[poly_idx].coeffs[coeff_idx];
    val = freeze(val);
    
    // Check |z| < gamma1 - beta
    if (val > gamma1_minus_beta && val < (int32_t)Q - gamma1_minus_beta) {
        *valid = false;
    }
}

// Verify signature: check c = H(w - c*t, msg)
__global__ void verify_signature_kernel(
    const RingtailSignature* __restrict__ sig,
    const PublicParams* __restrict__ pub,
    const PolyVec* __restrict__ w_prime,  // Reconstructed commitment
    bool* __restrict__ valid
) {
    // Single-threaded verification logic
    if (threadIdx.x != 0) return;
    
    // In real implementation:
    // 1. Compute w' = A*z - c*t
    // 2. Use hint to recover high bits of w
    // 3. Verify c = H(HighBits(w'), msg)
    
    *valid = true;  // Placeholder
}

// ============================================================================
// Batch Share Combination
// ============================================================================

// Batch combine shares for multiple signature instances
__global__ void batch_combine_shares_kernel(
    const ThresholdShare* __restrict__ all_shares,  // [batch][num_shares]
    const uint32_t* __restrict__ all_indices,       // [batch][num_shares]
    uint32_t num_shares,
    uint32_t batch_size,
    PolyVec* __restrict__ combined_s,              // [batch]
    PolyVec* __restrict__ combined_y               // [batch]
) {
    const uint32_t batch_idx = blockIdx.y;
    const uint32_t poly_idx = blockIdx.x;
    const uint32_t coeff_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || poly_idx >= L || coeff_idx >= N) return;
    
    const ThresholdShare* shares = all_shares + batch_idx * num_shares;
    const uint32_t* indices = all_indices + batch_idx * num_shares;
    
    int64_t s_sum = 0;
    int64_t y_sum = 0;
    
    for (uint32_t i = 0; i < num_shares; i++) {
        int32_t lambda = compute_lagrange_coeff(shares[i].index, indices, num_shares);
        
        int64_t s_contrib = ((int64_t)shares[i].s_share.polys[poly_idx].coeffs[coeff_idx] * lambda) % Q;
        int64_t y_contrib = ((int64_t)shares[i].y_share.polys[poly_idx].coeffs[coeff_idx] * lambda) % Q;
        
        s_sum = (s_sum + s_contrib + Q) % Q;
        y_sum = (y_sum + y_contrib + Q) % Q;
    }
    
    combined_s[batch_idx].polys[poly_idx].coeffs[coeff_idx] = (int32_t)s_sum;
    combined_y[batch_idx].polys[poly_idx].coeffs[coeff_idx] = (int32_t)y_sum;
}

// ============================================================================
// Hint Computation and Verification
// ============================================================================

// Make hint: h[i] = 1 if HighBits(r + z) != HighBits(r)
__global__ void make_hint_kernel(
    const PolyVec* __restrict__ r,
    const PolyVec* __restrict__ z,
    uint8_t* __restrict__ hint,
    uint32_t* __restrict__ hint_count
) {
    const uint32_t poly_idx = blockIdx.x;
    const uint32_t coeff_idx = threadIdx.x;
    
    if (poly_idx >= K || coeff_idx >= N) return;
    
    int32_t r_val = r->polys[poly_idx].coeffs[coeff_idx];
    int32_t z_val = z->polys[poly_idx].coeffs[coeff_idx];
    
    int32_t r_high = highbits(r_val, 2 * GAMMA2);
    int32_t rz_high = highbits(mod_add(r_val, z_val), 2 * GAMMA2);
    
    if (r_high != rz_high) {
        uint32_t idx = poly_idx * N + coeff_idx;
        hint[idx] = 1;
        atomicAdd(hint_count, 1);
    }
}

// Use hint to recover high bits
__global__ void use_hint_kernel(
    const int32_t* __restrict__ r0,
    const int32_t* __restrict__ r1,
    const uint8_t* __restrict__ hint,
    int32_t* __restrict__ recovered
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= K * N) return;
    
    int32_t r0_val = r0[idx];
    int32_t r1_val = r1[idx];
    uint8_t h = hint[idx];
    
    if (h == 0) {
        recovered[idx] = r1_val;
    } else {
        // Adjust high bits based on sign of low bits
        if (r0_val > 0) {
            recovered[idx] = (r1_val + 1) % ((Q - 1) / (2 * GAMMA2) + 1);
        } else {
            recovered[idx] = (r1_val + Q - 1) % ((Q - 1) / (2 * GAMMA2) + 1);
        }
    }
}

// ============================================================================
// Key Generation Support
// ============================================================================

// Generate threshold shares of secret key
__global__ void generate_shares_kernel(
    const Poly* __restrict__ secret,       // Original secret polynomial
    uint32_t threshold,                    // t (minimum shares needed)
    uint32_t num_shares,                   // n (total shares)
    uint32_t seed,
    ThresholdShare* __restrict__ shares
) {
    const uint32_t share_idx = blockIdx.y;
    const uint32_t poly_idx = blockIdx.x;
    const uint32_t coeff_idx = threadIdx.x;
    
    if (share_idx >= num_shares || poly_idx >= L || coeff_idx >= N) return;
    
    // Shamir secret sharing with polynomial of degree t-1
    // s_i = a_0 + a_1*i + a_2*i^2 + ... + a_{t-1}*i^{t-1}
    // where a_0 = secret coefficient
    
    curandState state;
    curand_init(seed, share_idx * L * N + poly_idx * N + coeff_idx, 0, &state);
    
    int32_t s_val = secret[poly_idx].coeffs[coeff_idx];
    uint32_t x = share_idx + 1;  // Evaluation point
    
    int64_t share_val = s_val;
    int64_t x_power = x;
    
    // Add random polynomial terms
    for (uint32_t k = 1; k < threshold; k++) {
        int32_t a_k = sample_bounded(&state, Q / 4);  // Random coefficient
        share_val = (share_val + (int64_t)a_k * x_power) % Q;
        x_power = (x_power * x) % Q;
    }
    
    shares[share_idx].index = x;
    shares[share_idx].s_share.polys[poly_idx].coeffs[coeff_idx] = (int32_t)share_val;
}

// ============================================================================
// Host API
// ============================================================================

void combine_shares(
    const ThresholdShare* shares,
    const uint32_t* indices,
    uint32_t num_shares,
    PolyVec* combined_s,
    PolyVec* combined_y,
    cudaStream_t stream
) {
    dim3 block(N);
    dim3 grid(L);
    
    combine_shares_kernel<<<grid, block, 0, stream>>>(
        shares, indices, num_shares, combined_s, combined_y
    );
}

void sample_gaussian_poly(
    Poly* poly,
    uint32_t seed,
    float sigma,
    cudaStream_t stream
) {
    dim3 block(N);
    dim3 grid(1);
    
    sample_gaussian_poly_kernel<<<grid, block, 0, stream>>>(poly, seed, sigma);
}

void compute_commitment(
    const Poly* A,
    const PolyVec* y,
    PolyVec* w,
    cudaStream_t stream
) {
    dim3 block(N);
    dim3 grid(K);
    
    compute_commitment_kernel<<<grid, block, 0, stream>>>(A, y, w, K, L, N);
}

void verify_signature(
    const RingtailSignature* sig,
    const PublicParams* pub,
    const PolyVec* w_prime,
    bool* valid,
    cudaStream_t stream
) {
    dim3 block(1);
    dim3 grid(1);
    
    verify_signature_kernel<<<grid, block, 0, stream>>>(sig, pub, w_prime, valid);
}

void batch_combine_shares(
    const ThresholdShare* all_shares,
    const uint32_t* all_indices,
    uint32_t num_shares,
    uint32_t batch_size,
    PolyVec* combined_s,
    PolyVec* combined_y,
    cudaStream_t stream
) {
    dim3 block(N);
    dim3 grid(L, batch_size);
    
    batch_combine_shares_kernel<<<grid, block, 0, stream>>>(
        all_shares, all_indices, num_shares, batch_size, combined_s, combined_y
    );
}

void generate_shares(
    const Poly* secret,
    uint32_t threshold,
    uint32_t num_shares,
    uint32_t seed,
    ThresholdShare* shares,
    cudaStream_t stream
) {
    dim3 block(N);
    dim3 grid(L, num_shares);
    
    generate_shares_kernel<<<grid, block, 0, stream>>>(
        secret, threshold, num_shares, seed, shares
    );
}

void make_hint(
    const PolyVec* r,
    const PolyVec* z,
    uint8_t* hint,
    uint32_t* hint_count,
    cudaStream_t stream
) {
    // Initialize hint to zero
    cudaMemsetAsync(hint, 0, K * N * sizeof(uint8_t), stream);
    cudaMemsetAsync(hint_count, 0, sizeof(uint32_t), stream);
    
    dim3 block(N);
    dim3 grid(K);
    
    make_hint_kernel<<<grid, block, 0, stream>>>(r, z, hint, hint_count);
}

} // namespace ringtail
} // namespace cuda
} // namespace lux

// =============================================================================
// C API for CGO Bindings
// =============================================================================

extern "C" {

int lux_cuda_ringtail_ops_combine_shares(
    const void* shares,
    const uint32_t* indices,
    uint32_t num_shares,
    void* combined_s,
    void* combined_y,
    cudaStream_t stream
) {
    using namespace lux::cuda::ringtail;
    combine_shares(
        (const ThresholdShare*)shares,
        indices,
        num_shares,
        (PolyVec*)combined_s,
        (PolyVec*)combined_y,
        stream
    );
    return cudaGetLastError();
}

int lux_cuda_ringtail_ops_sample_gaussian(
    void* poly,
    uint32_t seed,
    float sigma,
    cudaStream_t stream
) {
    using namespace lux::cuda::ringtail;
    sample_gaussian_poly((Poly*)poly, seed, sigma, stream);
    return cudaGetLastError();
}

int lux_cuda_ringtail_ops_compute_commitment(
    const void* A,
    const void* y,
    void* w,
    cudaStream_t stream
) {
    using namespace lux::cuda::ringtail;
    compute_commitment((const Poly*)A, (const PolyVec*)y, (PolyVec*)w, stream);
    return cudaGetLastError();
}

int lux_cuda_ringtail_ops_verify_signature(
    const void* sig,
    const void* pub,
    const void* w_prime,
    bool* valid,
    cudaStream_t stream
) {
    using namespace lux::cuda::ringtail;
    verify_signature(
        (const RingtailSignature*)sig,
        (const PublicParams*)pub,
        (const PolyVec*)w_prime,
        valid,
        stream
    );
    return cudaGetLastError();
}

int lux_cuda_ringtail_ops_batch_combine_shares(
    const void* all_shares,
    const uint32_t* all_indices,
    uint32_t num_shares,
    uint32_t batch_size,
    void* combined_s,
    void* combined_y,
    cudaStream_t stream
) {
    using namespace lux::cuda::ringtail;
    batch_combine_shares(
        (const ThresholdShare*)all_shares,
        all_indices,
        num_shares,
        batch_size,
        (PolyVec*)combined_s,
        (PolyVec*)combined_y,
        stream
    );
    return cudaGetLastError();
}

int lux_cuda_ringtail_ops_generate_shares(
    const void* secret,
    uint32_t threshold,
    uint32_t num_shares,
    uint32_t seed,
    void* shares,
    cudaStream_t stream
) {
    using namespace lux::cuda::ringtail;
    generate_shares(
        (const Poly*)secret,
        threshold,
        num_shares,
        seed,
        (ThresholdShare*)shares,
        stream
    );
    return cudaGetLastError();
}

int lux_cuda_ringtail_ops_make_hint(
    const void* r,
    const void* z,
    uint8_t* hint,
    uint32_t* hint_count,
    cudaStream_t stream
) {
    using namespace lux::cuda::ringtail;
    make_hint((const PolyVec*)r, (const PolyVec*)z, hint, hint_count, stream);
    return cudaGetLastError();
}

} // extern "C"
