// Copyright (c) 2024-2026 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Ringtail Post-Quantum Signature Generation - CUDA Kernel
// Implements lattice-based threshold signature generation on GPU.
// Based on Module-LWE (MLWE) with Fiat-Shamir heuristic.
//
// Protocol Overview:
//   1. Key Generation: Generate secret s, compute public key t = A*s
//   2. Commitment: Sample random y, compute w = A*y
//   3. Challenge: c = H(w, message)
//   4. Response: z = y + c*s
//   5. Rejection Sampling: Ensure z is within bounds
//
// Parameters (Dilithium-style):
//   q = 8380417 (23-bit NTT-friendly prime)
//   n = 256 (ring dimension)
//   k = 4 (module rank for public key)
//   l = 4 (module rank for secret)
//   eta = 2 (secret coefficient bound)
//   gamma1 = 2^17 (commitment coefficient bound)
//   beta = 78 (rejection threshold)

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>

namespace lux {
namespace cuda {
namespace ringtail {

// ============================================================================
// Ringtail Parameters
// ============================================================================

// Ring modulus q = 8380417 = 2^23 - 2^13 + 1
constexpr uint32_t Q = 8380417U;
constexpr uint32_t Q_INV = 58728449U;    // q^{-1} mod 2^32

// Polynomial and matrix dimensions
constexpr uint32_t N = 256;              // Ring dimension
constexpr uint32_t K = 4;                // Module rank (public key dimension)
constexpr uint32_t L = 4;                // Module rank (secret dimension)
constexpr uint32_t LOG_N = 8;

// Signature bounds
constexpr int32_t ETA = 2;               // Secret coefficient bound
constexpr int32_t GAMMA1 = 131072;       // 2^17, commitment bound
constexpr int32_t GAMMA2 = 95232;        // Low bits rounding parameter
constexpr int32_t BETA = 78;             // tau * eta for rejection
constexpr int32_t TAU = 39;              // Challenge weight

// Threshold parameters
constexpr uint32_t MAX_THRESHOLD = 32;

// NTT parameters
constexpr uint32_t ROOT = 1753;          // Primitive 512th root of unity
constexpr uint32_t ROOT_INV = 8380416;

// Montgomery parameters
constexpr uint32_t MONT_R = 4193792U;    // 2^32 mod q
constexpr uint32_t MONT_R2 = 2365951U;   // 2^64 mod q

// ============================================================================
// Device Constants
// ============================================================================

__constant__ uint32_t d_twiddles[N];
__constant__ uint32_t d_twiddles_inv[N];
__constant__ int32_t d_gamma1 = GAMMA1;
__constant__ int32_t d_gamma2 = GAMMA2;
__constant__ int32_t d_beta = BETA;

// ============================================================================
// Data Structures
// ============================================================================

// Polynomial in R_q = Z_q[X]/(X^n + 1)
struct Poly {
    int32_t coeffs[N];
};

// Vector of L polynomials
struct PolyVecL {
    Poly polys[L];
};

// Vector of K polynomials
struct PolyVecK {
    Poly polys[K];
};

// Signing key (secret key share for threshold)
struct SigningKey {
    PolyVecL s;              // Secret key share
    PolyVecL s_ntt;          // Secret in NTT domain
    uint32_t share_index;    // Threshold share index
};

// Public parameters
struct PublicParams {
    Poly A[K][L];            // Public matrix (NTT domain)
    PolyVecK t;              // Public key t = A*s
    uint8_t rho[32];         // Seed for A
};

// Signature structure
struct Signature {
    Poly c;                  // Challenge polynomial
    PolyVecL z;              // Response vector
    uint8_t hint[K * N / 8]; // Hint bits for verification
    uint16_t hint_count;     // Number of hints set
};

// Commitment data
struct Commitment {
    PolyVecL y;              // Random masking vector
    PolyVecK w;              // Commitment w = A*y
    PolyVecK w1;             // High bits of w
};

// ============================================================================
// Modular Arithmetic
// ============================================================================

__device__ __forceinline__
int32_t mont_reduce(int64_t a) {
    int32_t t = (int32_t)((uint32_t)a * Q_INV);
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
int32_t mod_mul(int32_t a, int32_t b) {
    return mont_reduce((int64_t)a * b);
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
// Random Sampling
// ============================================================================

// Sample uniform random in [-bound, bound]
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
    uint32_t nonce
) {
    const uint32_t tid = threadIdx.x;
    if (tid >= N) return;

    curandState state;
    curand_init(seed, tid + nonce * N, 0, &state);

    poly->coeffs[tid] = sample_bounded(&state, ETA);
}

// Sample masking vector y with coefficients in [-gamma1, gamma1]
__global__ void sample_masking_vector_kernel(
    PolyVecL* __restrict__ y,
    uint32_t seed,
    uint32_t nonce
) {
    const uint32_t poly_idx = blockIdx.x;
    const uint32_t coeff_idx = threadIdx.x;

    if (poly_idx >= L || coeff_idx >= N) return;

    curandState state;
    curand_init(seed, poly_idx * N + coeff_idx + nonce * L * N, 0, &state);

    // Uniform in [-gamma1, gamma1]
    int32_t r = curand(&state) % (2 * GAMMA1 + 1);
    y->polys[poly_idx].coeffs[coeff_idx] = r - GAMMA1;
}

// ============================================================================
// NTT Operations
// ============================================================================

// Cooley-Tukey butterfly
__device__ __forceinline__
void ct_butterfly(int32_t* a, int32_t* b, int32_t zeta) {
    int32_t t = mod_mul(*b, zeta);
    *b = mod_sub(*a, t);
    *a = mod_add(*a, t);
}

// Gentleman-Sande butterfly
__device__ __forceinline__
void gs_butterfly(int32_t* a, int32_t* b, int32_t zeta) {
    int32_t t = *a;
    *a = mod_add(t, *b);
    *b = mod_mul(mod_sub(t, *b), zeta);
}

// Forward NTT in shared memory
__device__
void ntt_forward(int32_t* poly) {
    int k = 0;
    for (int len = 128; len >= 1; len >>= 1) {
        for (int start = 0; start < N; start += 2 * len) {
            int32_t zeta = d_twiddles[++k];
            for (int j = start; j < start + len; j++) {
                ct_butterfly(&poly[j], &poly[j + len], zeta);
            }
        }
    }
}

// Inverse NTT in shared memory
__device__
void ntt_inverse(int32_t* poly) {
    const int32_t N_INV = 8347681;  // 256^{-1} mod q

    int k = 256;
    for (int len = 1; len <= 128; len <<= 1) {
        for (int start = 0; start < N; start += 2 * len) {
            int32_t zeta = d_twiddles_inv[--k];
            for (int j = start; j < start + len; j++) {
                gs_butterfly(&poly[j], &poly[j + len], zeta);
            }
        }
    }

    // Scale by N^{-1}
    for (int i = 0; i < N; i++) {
        poly[i] = mod_mul(poly[i], N_INV);
    }
}

// Pointwise multiplication
__device__
void ntt_pointwise(int32_t* c, const int32_t* a, const int32_t* b) {
    for (int i = 0; i < N; i++) {
        c[i] = mod_mul(a[i], b[i]);
    }
}

// ============================================================================
// Commitment Generation
// ============================================================================

// Compute commitment w = A*y
__global__ void compute_commitment_kernel(
    const Poly* __restrict__ A,          // [K][L] in NTT domain
    const PolyVecL* __restrict__ y,      // Masking vector
    PolyVecK* __restrict__ w             // Output commitment
) {
    extern __shared__ int32_t shmem[];
    int32_t* acc = shmem;
    int32_t* y_poly = shmem + N;

    const uint32_t row = blockIdx.x;     // Which row of A (0..K-1)
    const uint32_t tid = threadIdx.x;

    if (row >= K || tid >= N) return;

    // Initialize accumulator
    acc[tid] = 0;
    __syncthreads();

    // Compute A[row] . y in NTT domain
    for (uint32_t col = 0; col < L; col++) {
        // Load y[col] and NTT
        y_poly[tid] = y->polys[col].coeffs[tid];
        __syncthreads();

        if (tid == 0) {
            ntt_forward(y_poly);
        }
        __syncthreads();

        // Pointwise multiply and accumulate
        int32_t a_val = A[row * L + col].coeffs[tid];
        int32_t prod = mod_mul(a_val, y_poly[tid]);
        acc[tid] = mod_add(acc[tid], prod);
        __syncthreads();
    }

    // INTT and store
    if (tid == 0) {
        ntt_inverse(acc);
    }
    __syncthreads();

    w->polys[row].coeffs[tid] = acc[tid];
}

// Extract high bits of commitment
__global__ void extract_high_bits_kernel(
    const PolyVecK* __restrict__ w,
    PolyVecK* __restrict__ w1
) {
    const uint32_t poly_idx = blockIdx.x;
    const uint32_t coeff_idx = threadIdx.x;

    if (poly_idx >= K || coeff_idx >= N) return;

    int32_t r = w->polys[poly_idx].coeffs[coeff_idx];
    r = freeze(r);

    // HighBits(r, 2*gamma2) = floor((r + gamma2) / (2*gamma2))
    const int32_t alpha = 2 * GAMMA2;
    int32_t r1 = (r + GAMMA2) / alpha;

    w1->polys[poly_idx].coeffs[coeff_idx] = r1;
}

// ============================================================================
// Challenge Generation
// ============================================================================

// SHAKE-256 state (simplified for challenge sampling)
struct Shake256State {
    uint64_t state[25];
    uint8_t buffer[136];
    size_t pos;
};

// Sample challenge polynomial with TAU non-zero coefficients in {-1, +1}
__global__ void sample_challenge_kernel(
    Poly* __restrict__ c,
    const uint8_t* __restrict__ seed    // 32 bytes = H(w1 || message)
) {
    const uint32_t tid = threadIdx.x;

    if (tid != 0) return;  // Single-threaded challenge sampling

    // Initialize c to zero
    for (int i = 0; i < N; i++) {
        c->coeffs[i] = 0;
    }

    // Simplified challenge generation using seed
    // In production, use full SHAKE-256 XOF
    uint32_t hash_state = 0;
    for (int i = 0; i < 32; i++) {
        hash_state = hash_state * 31 + seed[i];
    }

    // Sample TAU positions with +/-1 coefficients
    curandState rng;
    curand_init(hash_state, 0, 0, &rng);

    for (int i = N - TAU; i < N; i++) {
        // Sample j uniform in [0, i]
        int j;
        do {
            j = curand(&rng) % (i + 1);
        } while (j > i);

        // Swap c[i] and c[j]
        c->coeffs[i] = c->coeffs[j];

        // Set c[j] to +1 or -1
        c->coeffs[j] = (curand(&rng) & 1) ? 1 : (int32_t)Q - 1;
    }
}

// ============================================================================
// Response Computation
// ============================================================================

// Compute response z = y + c*s
__global__ void compute_response_kernel(
    const PolyVecL* __restrict__ y,      // Masking vector
    const Poly* __restrict__ c,          // Challenge
    const PolyVecL* __restrict__ s,      // Secret key
    PolyVecL* __restrict__ z             // Response
) {
    extern __shared__ int32_t shmem[];
    int32_t* c_ntt = shmem;
    int32_t* s_ntt = shmem + N;
    int32_t* cs = shmem + 2 * N;

    const uint32_t poly_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;

    if (poly_idx >= L || tid >= N) return;

    // Load challenge and NTT
    c_ntt[tid] = c->coeffs[tid];
    s_ntt[tid] = s->polys[poly_idx].coeffs[tid];
    __syncthreads();

    if (tid == 0) {
        ntt_forward(c_ntt);
        ntt_forward(s_ntt);
    }
    __syncthreads();

    // Pointwise multiply c * s
    cs[tid] = mod_mul(c_ntt[tid], s_ntt[tid]);
    __syncthreads();

    // INTT
    if (tid == 0) {
        ntt_inverse(cs);
    }
    __syncthreads();

    // z = y + c*s
    int32_t y_val = y->polys[poly_idx].coeffs[tid];
    z->polys[poly_idx].coeffs[tid] = mod_add(y_val, cs[tid]);
}

// ============================================================================
// Rejection Sampling
// ============================================================================

// Check if response z is within bounds
__global__ void check_rejection_kernel(
    const PolyVecL* __restrict__ z,
    const PolyVecK* __restrict__ w0,     // Low bits of w for checking
    bool* __restrict__ reject
) {
    const uint32_t poly_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;

    if (poly_idx >= L || tid >= N) return;

    int32_t z_val = z->polys[poly_idx].coeffs[tid];
    z_val = freeze(z_val);

    // Center around 0
    if (z_val > (int32_t)Q / 2) {
        z_val = z_val - (int32_t)Q;
    }

    // Check |z| < gamma1 - beta
    int32_t bound = GAMMA1 - BETA;
    if (abs(z_val) >= bound) {
        *reject = true;
    }
}

// ============================================================================
// Hint Computation
// ============================================================================

// Compute hint for verification
__global__ void compute_hint_kernel(
    const PolyVecK* __restrict__ w,
    const PolyVecK* __restrict__ cs2,    // c * s2 (second secret component)
    uint8_t* __restrict__ hint,
    uint32_t* __restrict__ hint_count
) {
    const uint32_t poly_idx = blockIdx.x;
    const uint32_t coeff_idx = threadIdx.x;

    if (poly_idx >= K || coeff_idx >= N) return;

    int32_t w_val = w->polys[poly_idx].coeffs[coeff_idx];
    int32_t cs2_val = cs2->polys[poly_idx].coeffs[coeff_idx];

    // Compute w - cs2
    int32_t r = mod_sub(w_val, cs2_val);
    r = freeze(r);

    // HighBits(r, 2*gamma2)
    const int32_t alpha = 2 * GAMMA2;
    int32_t r1 = (r + GAMMA2) / alpha;

    // HighBits(w, 2*gamma2)
    w_val = freeze(w_val);
    int32_t w1 = (w_val + GAMMA2) / alpha;

    // Hint = 1 if high bits differ
    uint32_t idx = poly_idx * N + coeff_idx;
    uint32_t byte_idx = idx / 8;
    uint32_t bit_idx = idx % 8;

    if (r1 != w1) {
        atomicOr((unsigned int*)&hint[byte_idx], 1U << bit_idx);
        atomicAdd(hint_count, 1);
    }
}

// ============================================================================
// Threshold Signing
// ============================================================================

// Lagrange coefficient computation for threshold
__device__
int32_t compute_lagrange_coeff(
    uint32_t index,
    const uint32_t* indices,
    uint32_t num_parties
) {
    int64_t numerator = 1;
    int64_t denominator = 1;

    for (uint32_t j = 0; j < num_parties; j++) {
        if (indices[j] == index) continue;

        numerator = (numerator * (int64_t)indices[j]) % Q;
        int64_t diff = (int64_t)indices[j] - (int64_t)index;
        if (diff < 0) diff += Q;
        denominator = (denominator * diff) % Q;
    }

    // Compute inverse using Fermat's little theorem
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

// Combine partial signatures into final signature
__global__ void combine_partial_signatures_kernel(
    const PolyVecL* __restrict__ z_shares,   // [num_parties] partial z values
    const uint32_t* __restrict__ indices,     // Party indices
    uint32_t num_parties,
    PolyVecL* __restrict__ z_combined
) {
    const uint32_t poly_idx = blockIdx.x;
    const uint32_t coeff_idx = threadIdx.x;

    if (poly_idx >= L || coeff_idx >= N) return;

    int64_t sum = 0;

    for (uint32_t i = 0; i < num_parties; i++) {
        int32_t lambda = compute_lagrange_coeff(indices[i], indices, num_parties);
        int32_t z_val = z_shares[i].polys[poly_idx].coeffs[coeff_idx];
        int64_t contrib = ((int64_t)z_val * lambda) % Q;
        sum = (sum + contrib + Q) % Q;
    }

    z_combined->polys[poly_idx].coeffs[coeff_idx] = (int32_t)sum;
}

// ============================================================================
// Full Signing Pipeline
// ============================================================================

// Complete signing kernel (single party, non-threshold)
extern "C" __global__
void ringtail_sign_kernel(
    Signature* __restrict__ sig,
    const PublicParams* __restrict__ pub,
    const SigningKey* __restrict__ sk,
    const uint8_t* __restrict__ msg_hash,  // 64 bytes
    uint32_t* __restrict__ attempts,       // Rejection sampling counter
    bool* __restrict__ success
) {
    extern __shared__ int32_t shmem[];

    const uint32_t tid = threadIdx.x;

    if (tid != 0) return;  // Single-threaded for now

    *success = false;
    *attempts = 0;

    // Rejection sampling loop
    for (uint32_t attempt = 0; attempt < 1000; attempt++) {
        (*attempts)++;

        // 1. Sample masking vector y
        curandState rng;
        curand_init((uint32_t)(size_t)msg_hash ^ attempt, tid, 0, &rng);

        PolyVecL y;
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < N; j++) {
                y.polys[i].coeffs[j] = (curand(&rng) % (2 * GAMMA1 + 1)) - GAMMA1;
            }
        }

        // 2. Compute commitment w = A*y
        PolyVecK w;
        int32_t* temp = shmem;

        for (int i = 0; i < K; i++) {
            for (int k = 0; k < N; k++) temp[k] = 0;

            for (int j = 0; j < L; j++) {
                // NTT(y[j])
                int32_t y_ntt[N];
                for (int k = 0; k < N; k++) y_ntt[k] = y.polys[j].coeffs[k];
                ntt_forward(y_ntt);

                // Accumulate A[i][j] * y[j]
                for (int k = 0; k < N; k++) {
                    temp[k] = mod_add(temp[k], mod_mul(pub->A[i][j].coeffs[k], y_ntt[k]));
                }
            }

            ntt_inverse(temp);
            for (int k = 0; k < N; k++) w.polys[i].coeffs[k] = temp[k];
        }

        // 3. Extract high bits w1
        PolyVecK w1;
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < N; j++) {
                int32_t r = freeze(w.polys[i].coeffs[j]);
                w1.polys[i].coeffs[j] = (r + GAMMA2) / (2 * GAMMA2);
            }
        }

        // 4. Compute challenge c = H(w1, msg)
        // Simplified hash
        uint32_t hash_state = 0;
        for (int i = 0; i < 64; i++) hash_state = hash_state * 31 + msg_hash[i];
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < N; j++) {
                hash_state = hash_state * 31 + w1.polys[i].coeffs[j];
            }
        }

        Poly c;
        for (int i = 0; i < N; i++) c.coeffs[i] = 0;

        curandState c_rng;
        curand_init(hash_state, 0, 0, &c_rng);

        for (int i = N - TAU; i < N; i++) {
            int j = curand(&c_rng) % (i + 1);
            c.coeffs[i] = c.coeffs[j];
            c.coeffs[j] = (curand(&c_rng) & 1) ? 1 : (int32_t)Q - 1;
        }

        // 5. Compute response z = y + c*s
        PolyVecL z;
        int32_t c_ntt[N];
        for (int k = 0; k < N; k++) c_ntt[k] = c.coeffs[k];
        ntt_forward(c_ntt);

        for (int i = 0; i < L; i++) {
            int32_t cs[N];
            ntt_pointwise(cs, c_ntt, sk->s_ntt.polys[i].coeffs);
            ntt_inverse(cs);

            for (int j = 0; j < N; j++) {
                z.polys[i].coeffs[j] = mod_add(y.polys[i].coeffs[j], cs[j]);
            }
        }

        // 6. Check rejection bounds
        bool reject = false;
        for (int i = 0; i < L && !reject; i++) {
            for (int j = 0; j < N && !reject; j++) {
                int32_t z_val = freeze(z.polys[i].coeffs[j]);
                if (z_val > (int32_t)Q / 2) z_val -= Q;
                if (abs(z_val) >= GAMMA1 - BETA) {
                    reject = true;
                }
            }
        }

        if (reject) continue;

        // 7. Success - copy to output
        sig->c = c;
        sig->z = z;
        sig->hint_count = 0;  // Simplified

        *success = true;
        return;
    }
}

// ============================================================================
// Host API
// ============================================================================

} // namespace ringtail
} // namespace cuda
} // namespace lux

#ifdef __cplusplus
extern "C" {
#endif

// Initialize twiddle factors
cudaError_t lux_cuda_ringtail_sign_init(void) {
    // Precompute twiddle factors for NTT
    uint32_t twiddles[256];
    uint32_t twiddles_inv[256];

    // Compute powers of root of unity
    uint64_t omega = lux::cuda::ringtail::ROOT;
    uint64_t omega_inv = lux::cuda::ringtail::ROOT_INV;
    uint64_t q = lux::cuda::ringtail::Q;

    uint64_t pow = 1;
    uint64_t pow_inv = 1;

    for (int i = 0; i < 256; i++) {
        twiddles[i] = (uint32_t)pow;
        twiddles_inv[i] = (uint32_t)pow_inv;

        pow = (pow * omega) % q;
        pow_inv = (pow_inv * omega_inv) % q;
    }

    cudaMemcpyToSymbol(lux::cuda::ringtail::d_twiddles, twiddles, sizeof(twiddles));
    cudaMemcpyToSymbol(lux::cuda::ringtail::d_twiddles_inv, twiddles_inv, sizeof(twiddles_inv));

    return cudaGetLastError();
}

// Sign a message
cudaError_t lux_cuda_ringtail_sign(
    void* sig,                    // Output signature
    const void* pub,              // Public parameters
    const void* sk,               // Signing key
    const uint8_t* msg_hash,      // 64-byte message hash
    uint32_t* attempts            // Number of rejection samples
) {
    using namespace lux::cuda::ringtail;

    // Allocate device memory
    Signature* d_sig;
    PublicParams* d_pub;
    SigningKey* d_sk;
    uint8_t* d_msg;
    uint32_t* d_attempts;
    bool* d_success;

    cudaMalloc(&d_sig, sizeof(Signature));
    cudaMalloc(&d_pub, sizeof(PublicParams));
    cudaMalloc(&d_sk, sizeof(SigningKey));
    cudaMalloc(&d_msg, 64);
    cudaMalloc(&d_attempts, sizeof(uint32_t));
    cudaMalloc(&d_success, sizeof(bool));

    // Copy inputs
    cudaMemcpy(d_pub, pub, sizeof(PublicParams), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sk, sk, sizeof(SigningKey), cudaMemcpyHostToDevice);
    cudaMemcpy(d_msg, msg_hash, 64, cudaMemcpyHostToDevice);

    // Launch kernel
    size_t shmem = 4 * N * sizeof(int32_t);
    ringtail_sign_kernel<<<1, 256, shmem>>>(
        d_sig, d_pub, d_sk, d_msg, d_attempts, d_success
    );

    // Copy results
    cudaMemcpy(sig, d_sig, sizeof(Signature), cudaMemcpyDeviceToHost);
    cudaMemcpy(attempts, d_attempts, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_sig);
    cudaFree(d_pub);
    cudaFree(d_sk);
    cudaFree(d_msg);
    cudaFree(d_attempts);
    cudaFree(d_success);

    return cudaGetLastError();
}

// Batch sign multiple messages
cudaError_t lux_cuda_ringtail_sign_batch(
    void* sigs,                   // [count] Output signatures
    const void* pub,              // Public parameters
    const void* sk,               // Signing key
    const uint8_t* msg_hashes,    // [count * 64] Message hashes
    uint32_t count,
    uint32_t* total_attempts
) {
    *total_attempts = 0;

    // Process each signature sequentially (parallel improvement possible)
    for (uint32_t i = 0; i < count; i++) {
        uint32_t attempts;
        cudaError_t err = lux_cuda_ringtail_sign(
            (uint8_t*)sigs + i * sizeof(lux::cuda::ringtail::Signature),
            pub,
            sk,
            msg_hashes + i * 64,
            &attempts
        );
        if (err != cudaSuccess) return err;
        *total_attempts += attempts;
    }

    return cudaSuccess;
}

// Get signature parameters
void lux_cuda_ringtail_get_params(
    uint32_t* q,
    uint32_t* n,
    uint32_t* k,
    uint32_t* l
) {
    *q = lux::cuda::ringtail::Q;
    *n = lux::cuda::ringtail::N;
    *k = lux::cuda::ringtail::K;
    *l = lux::cuda::ringtail::L;
}

// Cleanup
cudaError_t lux_cuda_ringtail_sign_cleanup(void) {
    return cudaSuccess;
}

#ifdef __cplusplus
}
#endif
