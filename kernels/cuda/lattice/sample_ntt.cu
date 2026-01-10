// Copyright (c) 2024-2026 Lux Industries Inc. All Rights Reserved.
// PROPRIETARY AND CONFIDENTIAL - NO LICENSE GRANTED
// Unauthorized use, copying, or distribution is strictly prohibited.
//
// Lattice Cryptography CUDA Kernels - Sampling and NTT Operations
// Implements discrete Gaussian sampling, NTT, and noise generation for:
// - Ringtail threshold signatures (MLWE-based)
// - FHE key generation (TFHE, CKKS, BGV)
// - Post-quantum cryptographic primitives

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>
#include <cmath>

// ============================================================================
// Security Level Parameters
// ============================================================================

// NIST security levels with corresponding parameters
// Level I:   n=512,  q=12289    (~128-bit classical)
// Level III: n=1024, q=12289    (~192-bit classical)
// Level V:   n=2048, q=134217601 (~256-bit classical)

// Common lattice moduli
#define Q_DILITHIUM     8380417ULL     // Dilithium/Ringtail modulus
#define Q_KYBER         3329ULL        // Kyber modulus
#define Q_TFHE          0xFFFFFFFF00000001ULL  // 2^64 - 2^32 + 1 (NTT-friendly)
#define Q_SEAL          0x7FFFFFFFE0001ULL     // SEAL default prime
#define Q_SMALL         12289ULL       // NewHope/Frodo prime

// Ring dimensions
#define N_512   512
#define N_1024  1024
#define N_2048  2048
#define N_4096  4096

// Shared memory limits
#define MAX_SHARED_COEFFS 4096
#define WARP_SIZE 32

// ============================================================================
// Parameter Structures
// ============================================================================

// Discrete Gaussian parameters
struct GaussianParams {
    double sigma;           // Standard deviation
    double tail_bound;      // Tail-cut bound (typically 6*sigma)
    uint32_t precision;     // Bits of precision for sampling
    bool use_cdt;           // Use CDT instead of Box-Muller
};

// NTT parameters (multi-modulus RNS support)
struct LatticeNTTParams {
    uint64_t q;             // Prime modulus
    uint64_t psi;           // 2N-th primitive root of unity
    uint64_t psi_inv;       // Inverse of psi
    uint64_t n_inv;         // N^{-1} mod q
    uint64_t mont_r;        // Montgomery R = 2^64 mod q
    uint64_t mont_r2;       // R^2 mod q
    uint64_t mont_q_inv;    // -q^{-1} mod 2^64
    uint32_t n;             // Ring dimension
    uint32_t log_n;         // log2(n)
};

// ============================================================================
// Device Constants
// ============================================================================

__constant__ LatticeNTTParams d_ntt_params;
__constant__ uint64_t d_psi_powers[8192];      // Powers of psi (forward NTT)
__constant__ uint64_t d_psi_inv_powers[8192];  // Powers of psi^{-1} (inverse NTT)

// CDT table for discrete Gaussian (sigma = 3.2, 128-bit precision)
// Each entry is P(X <= x) * 2^128 truncated to 64 bits of precision
__constant__ uint64_t d_cdt_table[128];
__constant__ uint32_t d_cdt_size;

// ============================================================================
// Montgomery Arithmetic
// ============================================================================

__device__ __forceinline__
uint64_t mont_reduce_64(uint64_t lo, uint64_t hi, uint64_t q, uint64_t q_inv) {
    // Montgomery reduction of 128-bit value [hi:lo]
    // Result = [hi:lo] * R^{-1} mod q where R = 2^64
    uint64_t m = lo * q_inv;
    uint64_t carry;

    // Compute (lo + m*q) >> 64
    uint64_t t_lo = lo + m * q;
    carry = (t_lo < lo) ? 1 : 0;

    uint64_t t_hi = hi + __umul64hi(m, q) + carry;

    return (t_hi >= q) ? (t_hi - q) : t_hi;
}

__device__ __forceinline__
uint64_t mont_mul_64(uint64_t a, uint64_t b, uint64_t q, uint64_t q_inv) {
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);
    return mont_reduce_64(lo, hi, q, q_inv);
}

__device__ __forceinline__
uint64_t mod_add_64(uint64_t a, uint64_t b, uint64_t q) {
    uint64_t sum = a + b;
    return (sum >= q) ? (sum - q) : sum;
}

__device__ __forceinline__
uint64_t mod_sub_64(uint64_t a, uint64_t b, uint64_t q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

// ============================================================================
// Box-Muller Discrete Gaussian Sampling
// ============================================================================

// Box-Muller transform for continuous Gaussian
__device__ __forceinline__
void box_muller(float u1, float u2, float* z0, float* z1) {
    // Avoid log(0) and division issues
    float safe_u1 = fmaxf(u1, 1e-10f);
    float r = sqrtf(-2.0f * logf(safe_u1));
    float theta = 2.0f * 3.14159265358979323846f * u2;

    *z0 = r * cosf(theta);
    *z1 = r * sinf(theta);
}

// Double-precision Box-Muller for higher precision
__device__ __forceinline__
void box_muller_double(double u1, double u2, double* z0, double* z1) {
    double safe_u1 = fmax(u1, 1e-16);
    double r = sqrt(-2.0 * log(safe_u1));
    double theta = 2.0 * 3.14159265358979323846 * u2;

    *z0 = r * cos(theta);
    *z1 = r * sin(theta);
}

// Discrete Gaussian sample via rounding Box-Muller
__device__ __forceinline__
int64_t sample_discrete_gaussian_bm(curandStatePhilox4_32_10_t* state, double sigma) {
    double u1 = curand_uniform_double(state);
    double u2 = curand_uniform_double(state);

    double z0, z1;
    box_muller_double(u1, u2, &z0, &z1);

    // Scale by sigma and round to nearest integer
    return (int64_t)llround(z0 * sigma);
}

// ============================================================================
// CDT-Based Discrete Gaussian Sampling (High Precision)
// ============================================================================

// Binary search in CDT table
__device__ __forceinline__
int32_t cdt_sample(uint64_t random_bits, const uint64_t* cdt, uint32_t cdt_size) {
    int32_t low = 0;
    int32_t high = cdt_size - 1;

    while (low < high) {
        int32_t mid = (low + high) >> 1;
        if (random_bits <= cdt[mid]) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }

    return low;
}

// Discrete Gaussian using CDT with rejection sampling for tails
__device__
int64_t sample_discrete_gaussian_cdt(curandStatePhilox4_32_10_t* state,
                                      double sigma, double tail_bound) {
    // Generate uniform random bits
    uint64_t random_bits = ((uint64_t)curand(state) << 32) | curand(state);

    // Sign bit
    int32_t sign = (curand(state) & 1) ? 1 : -1;

    // CDT lookup for |x|
    int32_t abs_val = cdt_sample(random_bits, d_cdt_table, d_cdt_size);

    // Check tail bound (reject if |x| > tail_bound)
    if ((double)abs_val > tail_bound) {
        // Rejection: fall back to Box-Muller with rejection
        double z;
        do {
            z = fabs(sample_discrete_gaussian_bm(state, sigma));
        } while (z > tail_bound);
        abs_val = (int32_t)llround(z);
    }

    return (int64_t)(sign * abs_val);
}

// ============================================================================
// Centered Binomial Distribution Sampling
// ============================================================================

// Sample from centered binomial distribution CBD(eta)
// Used in Kyber, SABER, and other schemes
__device__ __forceinline__
int32_t sample_cbd(uint32_t random_bits, uint32_t eta) {
    int32_t a = 0, b = 0;

    // Count bits in each half
    for (uint32_t i = 0; i < eta; i++) {
        a += (random_bits >> i) & 1;
        b += (random_bits >> (eta + i)) & 1;
    }

    return a - b;
}

// CBD kernel for polynomial sampling
__global__ void sample_cbd_poly_kernel(
    int32_t* __restrict__ poly,
    const uint32_t* __restrict__ random_bytes,
    uint32_t n,
    uint32_t eta
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Extract 2*eta bits for this coefficient
    uint32_t byte_idx = (idx * eta * 2) / 8;
    uint32_t bit_offset = (idx * eta * 2) % 8;

    uint32_t random_bits = (random_bytes[byte_idx] >> bit_offset);
    if (bit_offset + 2 * eta > 8 && byte_idx + 1 < (n * eta * 2 + 7) / 8) {
        random_bits |= (random_bytes[byte_idx + 1] << (8 - bit_offset));
    }

    random_bits &= (1U << (2 * eta)) - 1;
    poly[idx] = sample_cbd(random_bits, eta);
}

// ============================================================================
// Uniform Sampling mod q
// ============================================================================

// Rejection sampling for uniform mod q
__device__ __forceinline__
uint64_t sample_uniform_mod_q(curandStatePhilox4_32_10_t* state, uint64_t q) {
    // For 64-bit q, we need careful rejection sampling
    // to avoid modular bias

    uint64_t mask = 0xFFFFFFFFFFFFFFFFULL;
    uint64_t q_bits = 64 - __clzll(q);
    mask >>= (64 - q_bits);

    uint64_t sample;
    do {
        sample = ((uint64_t)curand(state) << 32) | curand(state);
        sample &= mask;
    } while (sample >= q);

    return sample;
}

// Uniform polynomial sampling kernel
__global__ void sample_uniform_poly_kernel(
    uint64_t* __restrict__ poly,
    uint64_t seed,
    uint32_t nonce,
    uint64_t q,
    uint32_t n
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Initialize PRNG state
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx + nonce * n, 0, &state);

    poly[idx] = sample_uniform_mod_q(&state, q);
}

// ============================================================================
// Noise Generation for Encryption
// ============================================================================

// Generate LWE error vector e with discrete Gaussian distribution
__global__ void sample_lwe_error_kernel(
    int64_t* __restrict__ error,
    uint64_t seed,
    uint32_t nonce,
    double sigma,
    double tail_bound,
    uint32_t n,
    bool use_cdt
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx + nonce * n, 0, &state);

    if (use_cdt) {
        error[idx] = sample_discrete_gaussian_cdt(&state, sigma, tail_bound);
    } else {
        error[idx] = sample_discrete_gaussian_bm(&state, sigma);
    }
}

// Generate RLWE error polynomial
__global__ void sample_rlwe_error_kernel(
    int64_t* __restrict__ error,     // Output error polynomial (n coefficients)
    uint64_t seed,
    uint32_t nonce,
    double sigma,
    double tail_bound,
    uint32_t n
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx + nonce * n, 0, &state);

    // Use high-precision CDT sampling
    error[idx] = sample_discrete_gaussian_cdt(&state, sigma, tail_bound);
}

// ============================================================================
// NTT for Lattice Polynomial Rings
// ============================================================================

// Cooley-Tukey butterfly for forward NTT
__device__ __forceinline__
void ntt_ct_butterfly(uint64_t* x0, uint64_t* x1, uint64_t psi_power,
                       uint64_t q, uint64_t q_inv) {
    uint64_t u = *x0;
    uint64_t v = mont_mul_64(*x1, psi_power, q, q_inv);
    *x0 = mod_add_64(u, v, q);
    *x1 = mod_sub_64(u, v, q);
}

// Gentleman-Sande butterfly for inverse NTT
__device__ __forceinline__
void ntt_gs_butterfly(uint64_t* x0, uint64_t* x1, uint64_t psi_inv_power,
                       uint64_t q, uint64_t q_inv) {
    uint64_t u = *x0;
    uint64_t v = *x1;
    *x0 = mod_add_64(u, v, q);
    *x1 = mont_mul_64(mod_sub_64(u, v, q), psi_inv_power, q, q_inv);
}

// Forward NTT for small polynomials (fused, shared memory)
__global__ void ntt_forward_fused_kernel(
    uint64_t* __restrict__ data,
    const uint64_t* __restrict__ psi_powers,
    uint64_t q,
    uint64_t q_inv,
    uint32_t n,
    uint32_t log_n,
    uint32_t batch_size
) {
    extern __shared__ uint64_t shmem[];

    const uint32_t batch_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t threads = blockDim.x;

    if (batch_idx >= batch_size) return;

    uint64_t* poly = data + batch_idx * n;

    // Load polynomial to shared memory
    for (uint32_t i = tid; i < n; i += threads) {
        shmem[i] = poly[i];
    }
    __syncthreads();

    // NTT stages
    for (uint32_t stage = 0; stage < log_n; stage++) {
        uint32_t m = 1U << stage;
        uint32_t half_m = n >> (stage + 1);

        for (uint32_t i = tid; i < n / 2; i += threads) {
            uint32_t group = i / half_m;
            uint32_t j = i % half_m;

            uint32_t idx0 = group * 2 * half_m + j;
            uint32_t idx1 = idx0 + half_m;

            uint64_t psi_power = psi_powers[m + group];

            uint64_t x0 = shmem[idx0];
            uint64_t x1 = shmem[idx1];

            ntt_ct_butterfly(&x0, &x1, psi_power, q, q_inv);

            shmem[idx0] = x0;
            shmem[idx1] = x1;
        }
        __syncthreads();
    }

    // Write back
    for (uint32_t i = tid; i < n; i += threads) {
        poly[i] = shmem[i];
    }
}

// Inverse NTT for small polynomials (fused, shared memory)
__global__ void ntt_inverse_fused_kernel(
    uint64_t* __restrict__ data,
    const uint64_t* __restrict__ psi_inv_powers,
    uint64_t q,
    uint64_t q_inv,
    uint64_t n_inv,        // N^{-1} mod q (in Montgomery form)
    uint32_t n,
    uint32_t log_n,
    uint32_t batch_size
) {
    extern __shared__ uint64_t shmem[];

    const uint32_t batch_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t threads = blockDim.x;

    if (batch_idx >= batch_size) return;

    uint64_t* poly = data + batch_idx * n;

    // Load polynomial to shared memory
    for (uint32_t i = tid; i < n; i += threads) {
        shmem[i] = poly[i];
    }
    __syncthreads();

    // Inverse NTT stages (reverse order)
    for (int32_t stage = log_n - 1; stage >= 0; stage--) {
        uint32_t m = n >> (stage + 1);
        uint32_t half_m = 1U << stage;

        for (uint32_t i = tid; i < n / 2; i += threads) {
            uint32_t group = i / half_m;
            uint32_t j = i % half_m;

            uint32_t idx0 = group * 2 * half_m + j;
            uint32_t idx1 = idx0 + half_m;

            uint64_t psi_inv_power = psi_inv_powers[m + group];

            uint64_t x0 = shmem[idx0];
            uint64_t x1 = shmem[idx1];

            ntt_gs_butterfly(&x0, &x1, psi_inv_power, q, q_inv);

            shmem[idx0] = x0;
            shmem[idx1] = x1;
        }
        __syncthreads();
    }

    // Scale by N^{-1} and write back
    for (uint32_t i = tid; i < n; i += threads) {
        poly[i] = mont_mul_64(shmem[i], n_inv, q, q_inv);
    }
}

// Single-stage NTT kernel (for large N > shared memory)
__global__ void ntt_forward_stage_kernel(
    uint64_t* __restrict__ data,
    const uint64_t* __restrict__ psi_powers,
    uint64_t q,
    uint64_t q_inv,
    uint32_t n,
    uint32_t stage,
    uint32_t batch_size
) {
    const uint32_t batch_idx = blockIdx.y;
    const uint32_t butterfly_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || butterfly_idx >= n / 2) return;

    uint64_t* poly = data + batch_idx * n;

    uint32_t m = 1U << stage;
    uint32_t half_m = n >> (stage + 1);

    uint32_t group = butterfly_idx / half_m;
    uint32_t j = butterfly_idx % half_m;

    uint32_t idx0 = group * 2 * half_m + j;
    uint32_t idx1 = idx0 + half_m;

    uint64_t psi_power = psi_powers[m + group];

    uint64_t x0 = poly[idx0];
    uint64_t x1 = poly[idx1];

    ntt_ct_butterfly(&x0, &x1, psi_power, q, q_inv);

    poly[idx0] = x0;
    poly[idx1] = x1;
}

// Single-stage inverse NTT kernel
__global__ void ntt_inverse_stage_kernel(
    uint64_t* __restrict__ data,
    const uint64_t* __restrict__ psi_inv_powers,
    uint64_t q,
    uint64_t q_inv,
    uint32_t n,
    uint32_t stage,
    uint32_t batch_size
) {
    const uint32_t batch_idx = blockIdx.y;
    const uint32_t butterfly_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || butterfly_idx >= n / 2) return;

    uint64_t* poly = data + batch_idx * n;

    uint32_t m = n >> (stage + 1);
    uint32_t half_m = 1U << stage;

    uint32_t group = butterfly_idx / half_m;
    uint32_t j = butterfly_idx % half_m;

    uint32_t idx0 = group * 2 * half_m + j;
    uint32_t idx1 = idx0 + half_m;

    uint64_t psi_inv_power = psi_inv_powers[m + group];

    uint64_t x0 = poly[idx0];
    uint64_t x1 = poly[idx1];

    ntt_gs_butterfly(&x0, &x1, psi_inv_power, q, q_inv);

    poly[idx0] = x0;
    poly[idx1] = x1;
}

// Final scaling for inverse NTT
__global__ void ntt_scale_kernel(
    uint64_t* __restrict__ data,
    uint64_t q,
    uint64_t q_inv,
    uint64_t n_inv,
    uint32_t n,
    uint32_t batch_size
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = n * batch_size;

    if (idx >= total) return;

    data[idx] = mont_mul_64(data[idx], n_inv, q, q_inv);
}

// ============================================================================
// Polynomial Arithmetic in NTT Domain
// ============================================================================

// Pointwise multiplication in NTT domain
__global__ void ntt_pointwise_mul_kernel(
    uint64_t* __restrict__ result,
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t q,
    uint64_t q_inv,
    uint32_t n,
    uint32_t batch_size
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = n * batch_size;

    if (idx >= total) return;

    result[idx] = mont_mul_64(a[idx], b[idx], q, q_inv);
}

// Pointwise multiply-accumulate: result += a * b (in NTT domain)
__global__ void ntt_pointwise_mac_kernel(
    uint64_t* __restrict__ result,
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t q,
    uint64_t q_inv,
    uint32_t n,
    uint32_t batch_size
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = n * batch_size;

    if (idx >= total) return;

    uint64_t prod = mont_mul_64(a[idx], b[idx], q, q_inv);
    result[idx] = mod_add_64(result[idx], prod, q);
}

// Polynomial addition
__global__ void poly_add_kernel(
    uint64_t* __restrict__ result,
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t q,
    uint32_t n,
    uint32_t batch_size
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = n * batch_size;

    if (idx >= total) return;

    result[idx] = mod_add_64(a[idx], b[idx], q);
}

// Polynomial subtraction
__global__ void poly_sub_kernel(
    uint64_t* __restrict__ result,
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t q,
    uint32_t n,
    uint32_t batch_size
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = n * batch_size;

    if (idx >= total) return;

    result[idx] = mod_sub_64(a[idx], b[idx], q);
}

// Scalar multiplication
__global__ void poly_scalar_mul_kernel(
    uint64_t* __restrict__ result,
    const uint64_t* __restrict__ a,
    uint64_t scalar,
    uint64_t q,
    uint64_t q_inv,
    uint32_t n,
    uint32_t batch_size
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = n * batch_size;

    if (idx >= total) return;

    result[idx] = mont_mul_64(a[idx], scalar, q, q_inv);
}

// ============================================================================
// NTT with Multiple Security Levels
// ============================================================================

// Precompute NTT parameters for given modulus and ring dimension
extern "C" __host__
void lux_cuda_lattice_compute_ntt_params(
    uint64_t q,           // Prime modulus
    uint64_t psi,         // Primitive 2N-th root of unity
    uint32_t n,           // Ring dimension (power of 2)
    uint64_t* psi_powers, // Output: n powers of psi
    uint64_t* psi_inv_powers  // Output: n powers of psi^{-1}
) {
    // Compute psi^{-1} mod q using Fermat's little theorem
    uint64_t psi_inv = 1;
    uint64_t base = psi;
    uint64_t exp = q - 2;
    while (exp > 0) {
        if (exp & 1) psi_inv = (__uint128_t)psi_inv * base % q;
        base = (__uint128_t)base * base % q;
        exp >>= 1;
    }

    // Bit-reversal permutation of powers
    uint32_t log_n = 0;
    for (uint32_t t = n; t > 1; t >>= 1) log_n++;

    psi_powers[0] = 1;
    psi_inv_powers[0] = 1;

    for (uint32_t i = 1; i < n; i++) {
        psi_powers[i] = (__uint128_t)psi_powers[i - 1] * psi % q;
        psi_inv_powers[i] = (__uint128_t)psi_inv_powers[i - 1] * psi_inv % q;
    }
}

// ============================================================================
// C API with lux_cuda_lattice_* prefix
// ============================================================================

extern "C" {

// Initialize PRNG states
cudaError_t lux_cuda_lattice_init_rng(
    curandStatePhilox4_32_10_t* states,
    uint64_t seed,
    uint32_t count,
    cudaStream_t stream
);

// Sample discrete Gaussian polynomial
cudaError_t lux_cuda_lattice_sample_gaussian(
    int64_t* output,
    uint64_t seed,
    uint32_t nonce,
    double sigma,
    double tail_bound,
    uint32_t n,
    uint32_t batch_size,
    bool use_cdt,
    cudaStream_t stream
) {
    const uint32_t threads = 256;
    const uint32_t blocks = (n * batch_size + threads - 1) / threads;

    sample_lwe_error_kernel<<<blocks, threads, 0, stream>>>(
        output, seed, nonce, sigma, tail_bound, n * batch_size, use_cdt
    );

    return cudaGetLastError();
}

// Sample centered binomial polynomial (CBD)
cudaError_t lux_cuda_lattice_sample_cbd(
    int32_t* output,
    const uint32_t* random_bytes,
    uint32_t n,
    uint32_t eta,
    uint32_t batch_size,
    cudaStream_t stream
) {
    const uint32_t threads = 256;
    const uint32_t total = n * batch_size;
    const uint32_t blocks = (total + threads - 1) / threads;

    sample_cbd_poly_kernel<<<blocks, threads, 0, stream>>>(
        output, random_bytes, total, eta
    );

    return cudaGetLastError();
}

// Sample uniform polynomial mod q
cudaError_t lux_cuda_lattice_sample_uniform(
    uint64_t* output,
    uint64_t seed,
    uint32_t nonce,
    uint64_t q,
    uint32_t n,
    uint32_t batch_size,
    cudaStream_t stream
) {
    const uint32_t threads = 256;
    const uint32_t total = n * batch_size;
    const uint32_t blocks = (total + threads - 1) / threads;

    sample_uniform_poly_kernel<<<blocks, threads, 0, stream>>>(
        output, seed, nonce, q, total
    );

    return cudaGetLastError();
}

// Forward NTT
cudaError_t lux_cuda_lattice_ntt_forward(
    uint64_t* data,
    const uint64_t* psi_powers,
    uint64_t q,
    uint64_t q_inv,
    uint32_t n,
    uint32_t batch_size,
    cudaStream_t stream
) {
    uint32_t log_n = 0;
    for (uint32_t t = n; t > 1; t >>= 1) log_n++;

    if (n <= MAX_SHARED_COEFFS) {
        // Fused kernel (single launch)
        const uint32_t threads = min(n / 2, 512U);
        const size_t shared_mem = n * sizeof(uint64_t);

        ntt_forward_fused_kernel<<<batch_size, threads, shared_mem, stream>>>(
            data, psi_powers, q, q_inv, n, log_n, batch_size
        );
    } else {
        // Staged kernel (multiple launches)
        const uint32_t threads = 256;
        dim3 blocks((n / 2 + threads - 1) / threads, batch_size);

        for (uint32_t stage = 0; stage < log_n; stage++) {
            ntt_forward_stage_kernel<<<blocks, threads, 0, stream>>>(
                data, psi_powers, q, q_inv, n, stage, batch_size
            );
        }
    }

    return cudaGetLastError();
}

// Inverse NTT
cudaError_t lux_cuda_lattice_ntt_inverse(
    uint64_t* data,
    const uint64_t* psi_inv_powers,
    uint64_t q,
    uint64_t q_inv,
    uint64_t n_inv,
    uint32_t n,
    uint32_t batch_size,
    cudaStream_t stream
) {
    uint32_t log_n = 0;
    for (uint32_t t = n; t > 1; t >>= 1) log_n++;

    if (n <= MAX_SHARED_COEFFS) {
        const uint32_t threads = min(n / 2, 512U);
        const size_t shared_mem = n * sizeof(uint64_t);

        ntt_inverse_fused_kernel<<<batch_size, threads, shared_mem, stream>>>(
            data, psi_inv_powers, q, q_inv, n_inv, n, log_n, batch_size
        );
    } else {
        const uint32_t threads = 256;
        dim3 blocks((n / 2 + threads - 1) / threads, batch_size);

        for (int32_t stage = log_n - 1; stage >= 0; stage--) {
            ntt_inverse_stage_kernel<<<blocks, threads, 0, stream>>>(
                data, psi_inv_powers, q, q_inv, n, stage, batch_size
            );
        }

        // Final scaling
        const uint32_t total = n * batch_size;
        const uint32_t scale_blocks = (total + threads - 1) / threads;
        ntt_scale_kernel<<<scale_blocks, threads, 0, stream>>>(
            data, q, q_inv, n_inv, n, batch_size
        );
    }

    return cudaGetLastError();
}

// Pointwise multiplication in NTT domain
cudaError_t lux_cuda_lattice_ntt_mul(
    uint64_t* result,
    const uint64_t* a,
    const uint64_t* b,
    uint64_t q,
    uint64_t q_inv,
    uint32_t n,
    uint32_t batch_size,
    cudaStream_t stream
) {
    const uint32_t threads = 256;
    const uint32_t total = n * batch_size;
    const uint32_t blocks = (total + threads - 1) / threads;

    ntt_pointwise_mul_kernel<<<blocks, threads, 0, stream>>>(
        result, a, b, q, q_inv, n, batch_size
    );

    return cudaGetLastError();
}

// Polynomial addition
cudaError_t lux_cuda_lattice_poly_add(
    uint64_t* result,
    const uint64_t* a,
    const uint64_t* b,
    uint64_t q,
    uint32_t n,
    uint32_t batch_size,
    cudaStream_t stream
) {
    const uint32_t threads = 256;
    const uint32_t total = n * batch_size;
    const uint32_t blocks = (total + threads - 1) / threads;

    poly_add_kernel<<<blocks, threads, 0, stream>>>(
        result, a, b, q, n, batch_size
    );

    return cudaGetLastError();
}

// Polynomial subtraction
cudaError_t lux_cuda_lattice_poly_sub(
    uint64_t* result,
    const uint64_t* a,
    const uint64_t* b,
    uint64_t q,
    uint32_t n,
    uint32_t batch_size,
    cudaStream_t stream
) {
    const uint32_t threads = 256;
    const uint32_t total = n * batch_size;
    const uint32_t blocks = (total + threads - 1) / threads;

    poly_sub_kernel<<<blocks, threads, 0, stream>>>(
        result, a, b, q, n, batch_size
    );

    return cudaGetLastError();
}

// Generate LWE secret key (small coefficients)
cudaError_t lux_cuda_lattice_gen_secret(
    int64_t* secret,
    uint64_t seed,
    uint32_t nonce,
    double sigma,
    uint32_t n,
    cudaStream_t stream
) {
    return lux_cuda_lattice_sample_gaussian(
        secret, seed, nonce, sigma, 6.0 * sigma, n, 1, true, stream
    );
}

// Generate LWE/RLWE error vector
cudaError_t lux_cuda_lattice_gen_error(
    int64_t* error,
    uint64_t seed,
    uint32_t nonce,
    double sigma,
    double tail_bound,
    uint32_t n,
    uint32_t batch_size,
    cudaStream_t stream
) {
    return lux_cuda_lattice_sample_gaussian(
        error, seed, nonce, sigma, tail_bound, n, batch_size, true, stream
    );
}

// Initialize CDT table for discrete Gaussian
cudaError_t lux_cuda_lattice_init_cdt(
    const uint64_t* cdt_table,
    uint32_t cdt_size
) {
    cudaError_t err = cudaMemcpyToSymbol(d_cdt_table, cdt_table,
                                          cdt_size * sizeof(uint64_t));
    if (err != cudaSuccess) return err;

    return cudaMemcpyToSymbol(d_cdt_size, &cdt_size, sizeof(uint32_t));
}

// Full polynomial multiplication: c = a * b in R_q
cudaError_t lux_cuda_lattice_poly_mul(
    uint64_t* result,
    const uint64_t* a,
    const uint64_t* b,
    const uint64_t* psi_powers,
    const uint64_t* psi_inv_powers,
    uint64_t q,
    uint64_t q_inv,
    uint64_t n_inv,
    uint32_t n,
    uint32_t batch_size,
    cudaStream_t stream
) {
    cudaError_t err;

    // Allocate temporary buffers for NTT
    uint64_t* a_ntt;
    uint64_t* b_ntt;
    size_t size = n * batch_size * sizeof(uint64_t);

    err = cudaMalloc(&a_ntt, size);
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&b_ntt, size);
    if (err != cudaSuccess) {
        cudaFree(a_ntt);
        return err;
    }

    // Copy inputs to NTT buffers
    cudaMemcpyAsync(a_ntt, a, size, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(b_ntt, b, size, cudaMemcpyDeviceToDevice, stream);

    // Forward NTT on both operands
    err = lux_cuda_lattice_ntt_forward(a_ntt, psi_powers, q, q_inv, n, batch_size, stream);
    if (err != cudaSuccess) goto cleanup;

    err = lux_cuda_lattice_ntt_forward(b_ntt, psi_powers, q, q_inv, n, batch_size, stream);
    if (err != cudaSuccess) goto cleanup;

    // Pointwise multiplication
    err = lux_cuda_lattice_ntt_mul(result, a_ntt, b_ntt, q, q_inv, n, batch_size, stream);
    if (err != cudaSuccess) goto cleanup;

    // Inverse NTT
    err = lux_cuda_lattice_ntt_inverse(result, psi_inv_powers, q, q_inv, n_inv, n, batch_size, stream);

cleanup:
    cudaFree(a_ntt);
    cudaFree(b_ntt);
    return err;
}

// Matrix-vector multiplication: result = A * s + e (for LWE encryption)
// A is k x n matrix, s is n-vector, e is k-vector
cudaError_t lux_cuda_lattice_lwe_encrypt(
    uint64_t* b_out,          // Output: k ciphertexts (in NTT domain)
    const uint64_t* A,        // k x n public matrix (in NTT domain)
    const uint64_t* s_ntt,    // Secret key (in NTT domain)
    const int64_t* e,         // Error vector (k elements, NOT in NTT domain)
    uint64_t q,
    uint64_t q_inv,
    uint32_t n,
    uint32_t k,               // Number of LWE samples
    cudaStream_t stream
) {
    const uint32_t threads = 256;

    // For each row i of A: b[i] = <A[i], s> + e[i]
    for (uint32_t i = 0; i < k; i++) {
        // Pointwise multiply A[i] with s
        lux_cuda_lattice_ntt_mul(
            b_out + i * n,
            A + i * n,
            s_ntt,
            q, q_inv, n, 1, stream
        );
    }

    // TODO: Add error terms (after INTT if needed)

    return cudaGetLastError();
}

} // extern "C"
