// Copyright (c) 2024-2026 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Ringtail Post-Quantum Signature Verification - CUDA Kernel
// Implements lattice-based threshold signature verification on GPU.
// Based on Module-LWE (MLWE) with Fiat-Shamir heuristic.
//
// Verification Equation:
//   1. Recompute w' = A*z - c*t
//   2. Use hint h to recover w1' from w'
//   3. Verify c == H(w1', message)
//
// Parameters (Dilithium-style):
//   q = 8380417 (23-bit NTT-friendly prime)
//   n = 256 (ring dimension)
//   k = 4 (module rank for public key)
//   l = 4 (module rank for signature)
//   gamma1 = 2^17 (commitment coefficient bound)
//   gamma2 = 95232 (low bits rounding parameter)
//   beta = 78 (rejection threshold)

#include <cuda_runtime.h>
#include <cstdint>

namespace lux {
namespace cuda {
namespace ringtail {

// ============================================================================
// Parameters (must match ringtail_sign.cu)
// ============================================================================

constexpr uint32_t Q = 8380417U;
constexpr uint32_t Q_INV = 58728449U;
constexpr uint32_t N = 256;
constexpr uint32_t K = 4;
constexpr uint32_t L = 4;
constexpr uint32_t LOG_N = 8;

constexpr int32_t GAMMA1 = 131072;
constexpr int32_t GAMMA2 = 95232;
constexpr int32_t BETA = 78;
constexpr int32_t TAU = 39;
constexpr int32_t D = 13;  // Power2Round parameter

constexpr uint32_t ROOT = 1753;
constexpr uint32_t ROOT_INV = 8380416;
constexpr uint32_t N_INV = 8347681;

// ============================================================================
// Device Constants
// ============================================================================

__constant__ uint32_t d_verify_twiddles[N];
__constant__ uint32_t d_verify_twiddles_inv[N];

// ============================================================================
// Data Structures
// ============================================================================

struct VerifyPoly {
    int32_t coeffs[N];
};

struct VerifyPolyVecL {
    VerifyPoly polys[L];
};

struct VerifyPolyVecK {
    VerifyPoly polys[K];
};

// Verification key (public key)
struct VerifyKey {
    uint8_t rho[32];           // Seed for A
    VerifyPolyVecK t1;         // High bits of t = A*s mod q
    VerifyPoly A_ntt[K][L];    // Expanded matrix (precomputed)
};

// Signature to verify
struct VerifySignature {
    VerifyPoly c;              // Challenge polynomial
    VerifyPolyVecL z;          // Response vector
    uint8_t hint[K * N / 8];   // Hint bits
    uint16_t hint_count;
};

// Verification context
struct VerifyContext {
    uint8_t msg_hash[64];      // H(tr || message)
    VerifyPolyVecK w_prime;    // Reconstructed commitment
    VerifyPolyVecK w1_prime;   // High bits after hint
};

// ============================================================================
// Modular Arithmetic
// ============================================================================

__device__ __forceinline__
int32_t verify_mont_reduce(int64_t a) {
    int32_t t = (int32_t)((uint32_t)a * Q_INV);
    return (int32_t)((a - (int64_t)t * Q) >> 32);
}

__device__ __forceinline__
int32_t verify_mod_add(int32_t a, int32_t b) {
    int32_t r = a + b;
    if (r >= (int32_t)Q) r -= Q;
    if (r < 0) r += Q;
    return r;
}

__device__ __forceinline__
int32_t verify_mod_sub(int32_t a, int32_t b) {
    int32_t r = a - b;
    if (r < 0) r += Q;
    return r;
}

__device__ __forceinline__
int32_t verify_mod_mul(int32_t a, int32_t b) {
    return verify_mont_reduce((int64_t)a * b);
}

__device__ __forceinline__
int32_t verify_freeze(int32_t a) {
    a += (a >> 31) & Q;
    a -= Q;
    a += (a >> 31) & Q;
    return a;
}

// ============================================================================
// NTT Operations
// ============================================================================

__device__ __forceinline__
void verify_ct_butterfly(int32_t* a, int32_t* b, int32_t zeta) {
    int32_t t = verify_mod_mul(*b, zeta);
    *b = verify_mod_sub(*a, t);
    *a = verify_mod_add(*a, t);
}

__device__ __forceinline__
void verify_gs_butterfly(int32_t* a, int32_t* b, int32_t zeta) {
    int32_t t = *a;
    *a = verify_mod_add(t, *b);
    *b = verify_mod_mul(verify_mod_sub(t, *b), zeta);
}

__device__
void verify_ntt_forward(int32_t* poly) {
    int k = 0;
    for (int len = 128; len >= 1; len >>= 1) {
        for (int start = 0; start < N; start += 2 * len) {
            int32_t zeta = d_verify_twiddles[++k];
            for (int j = start; j < start + len; j++) {
                verify_ct_butterfly(&poly[j], &poly[j + len], zeta);
            }
        }
    }
}

__device__
void verify_ntt_inverse(int32_t* poly) {
    int k = 256;
    for (int len = 1; len <= 128; len <<= 1) {
        for (int start = 0; start < N; start += 2 * len) {
            int32_t zeta = d_verify_twiddles_inv[--k];
            for (int j = start; j < start + len; j++) {
                verify_gs_butterfly(&poly[j], &poly[j + len], zeta);
            }
        }
    }

    // Scale by N^{-1}
    for (int i = 0; i < N; i++) {
        poly[i] = verify_mod_mul(poly[i], N_INV);
    }
}

__device__
void verify_ntt_pointwise(int32_t* c, const int32_t* a, const int32_t* b) {
    for (int i = 0; i < N; i++) {
        c[i] = verify_mod_mul(a[i], b[i]);
    }
}

// ============================================================================
// Hint Operations
// ============================================================================

// Decompose r into high and low bits
// r = r1 * alpha + r0, where alpha = 2 * gamma2
__device__
int32_t verify_decompose(int32_t r, int32_t* r0) {
    const int32_t alpha = 2 * GAMMA2;

    r = verify_freeze(r);
    int32_t r1 = (r + GAMMA2) / alpha;
    *r0 = r - r1 * alpha;

    // Handle corner cases
    if (*r0 > GAMMA2) {
        *r0 -= alpha;
        r1 += 1;
    }
    if (*r0 < -GAMMA2) {
        *r0 += alpha;
        r1 -= 1;
    }

    // Ensure r1 in valid range
    if (r1 < 0) r1 += (Q - 1) / alpha + 1;

    return r1;
}

// Use hint to recover high bits
__device__
int32_t verify_use_hint(int32_t hint, int32_t r) {
    int32_t r0;
    int32_t r1 = verify_decompose(r, &r0);

    if (hint == 0) return r1;

    // Adjust high bits based on sign of low bits
    const int32_t m = (Q - 1) / (2 * GAMMA2) + 1;
    if (r0 > 0) {
        return (r1 + 1) % m;
    } else {
        return (r1 - 1 + m) % m;
    }
}

// Power2Round: decompose r = r1 * 2^d + r0
__device__
int32_t verify_power2round(int32_t r, int32_t* r0) {
    r = verify_freeze(r);
    int32_t r1 = (r + (1 << (D - 1)) - 1) >> D;
    *r0 = r - (r1 << D);
    return r1;
}

// ============================================================================
// Matrix Operations
// ============================================================================

// Compute A*z in NTT domain
__global__ void compute_Az_kernel(
    const int32_t* __restrict__ A_ntt,   // [K][L][N] precomputed
    const VerifyPolyVecL* __restrict__ z,
    VerifyPolyVecK* __restrict__ Az
) {
    extern __shared__ int32_t shmem[];
    int32_t* z_ntt = shmem;
    int32_t* acc = shmem + N;

    const uint32_t row = blockIdx.x;     // 0..K-1
    const uint32_t tid = threadIdx.x;

    if (row >= K || tid >= N) return;

    // Initialize accumulator
    acc[tid] = 0;
    __syncthreads();

    // Compute sum_j A[row][j] * z[j]
    for (uint32_t col = 0; col < L; col++) {
        // Load and NTT z[col]
        z_ntt[tid] = z->polys[col].coeffs[tid];
        __syncthreads();

        if (tid == 0) {
            verify_ntt_forward(z_ntt);
        }
        __syncthreads();

        // Pointwise multiply and accumulate
        int32_t a_val = A_ntt[(row * L + col) * N + tid];
        int32_t prod = verify_mod_mul(a_val, z_ntt[tid]);
        acc[tid] = verify_mod_add(acc[tid], prod);
        __syncthreads();
    }

    // INTT and store
    if (tid == 0) {
        verify_ntt_inverse(acc);
    }
    __syncthreads();

    Az->polys[row].coeffs[tid] = acc[tid];
}

// Compute c * t1 * 2^d
__global__ void compute_ct1_kernel(
    const VerifyPoly* __restrict__ c,
    const VerifyPolyVecK* __restrict__ t1,
    VerifyPolyVecK* __restrict__ ct1
) {
    extern __shared__ int32_t shmem[];
    int32_t* c_ntt = shmem;
    int32_t* t1_scaled = shmem + N;

    const uint32_t row = blockIdx.x;     // 0..K-1
    const uint32_t tid = threadIdx.x;

    if (row >= K || tid >= N) return;

    // Load and NTT challenge
    c_ntt[tid] = c->coeffs[tid];
    __syncthreads();

    if (tid == 0) {
        verify_ntt_forward(c_ntt);
    }
    __syncthreads();

    // Load t1[row] and scale by 2^d
    int32_t t1_val = t1->polys[row].coeffs[tid];
    t1_scaled[tid] = verify_mod_mul(t1_val, 1 << D);  // t1 * 2^d
    __syncthreads();

    if (tid == 0) {
        verify_ntt_forward(t1_scaled);
    }
    __syncthreads();

    // Pointwise multiply c * t1
    int32_t result[N];
    verify_ntt_pointwise(result, c_ntt, t1_scaled);

    if (tid == 0) {
        verify_ntt_inverse(result);
    }
    __syncthreads();

    ct1->polys[row].coeffs[tid] = result[tid];
}

// Compute w' = Az - ct1 and apply hints
__global__ void recover_w1_kernel(
    const VerifyPolyVecK* __restrict__ Az,
    const VerifyPolyVecK* __restrict__ ct1,
    const uint8_t* __restrict__ hints,
    VerifyPolyVecK* __restrict__ w1_prime
) {
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;

    if (row >= K || tid >= N) return;

    // Compute w' = Az - ct1
    int32_t w_prime = verify_mod_sub(
        Az->polys[row].coeffs[tid],
        ct1->polys[row].coeffs[tid]
    );

    // Get hint bit
    uint32_t idx = row * N + tid;
    uint32_t byte_idx = idx / 8;
    uint32_t bit_idx = idx % 8;
    int32_t hint = (hints[byte_idx] >> bit_idx) & 1;

    // Apply hint to recover w1'
    w1_prime->polys[row].coeffs[tid] = verify_use_hint(hint, w_prime);
}

// ============================================================================
// Challenge Verification
// ============================================================================

// Recompute challenge hash and compare
__global__ void verify_challenge_kernel(
    const VerifyPolyVecK* __restrict__ w1_prime,
    const uint8_t* __restrict__ msg_hash,
    const VerifyPoly* __restrict__ expected_c,
    int* __restrict__ result
) {
    if (threadIdx.x != 0) return;

    // Simplified hash computation
    // In production, use full SHAKE-256
    uint32_t hash_state = 0;
    for (int i = 0; i < 64; i++) {
        hash_state = hash_state * 31 + msg_hash[i];
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            hash_state = hash_state * 31 + w1_prime->polys[i].coeffs[j];
        }
    }

    // Regenerate challenge from hash
    VerifyPoly c_recomputed;
    for (int i = 0; i < N; i++) c_recomputed.coeffs[i] = 0;

    // Simplified PRNG from hash
    uint32_t rng_state = hash_state;
    auto next_rand = [&]() -> uint32_t {
        rng_state = rng_state * 1103515245 + 12345;
        return rng_state;
    };

    for (int i = N - TAU; i < N; i++) {
        int j = next_rand() % (i + 1);
        c_recomputed.coeffs[i] = c_recomputed.coeffs[j];
        c_recomputed.coeffs[j] = (next_rand() & 1) ? 1 : (int32_t)Q - 1;
    }

    // Compare with expected challenge
    bool valid = true;
    for (int i = 0; i < N; i++) {
        if (c_recomputed.coeffs[i] != expected_c->coeffs[i]) {
            valid = false;
            break;
        }
    }

    *result = valid ? 1 : 0;
}

// ============================================================================
// Bounds Checking
// ============================================================================

// Check if z coefficients are within bounds
__global__ void check_z_bounds_kernel(
    const VerifyPolyVecL* __restrict__ z,
    int* __restrict__ valid
) {
    const uint32_t poly_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;

    if (poly_idx >= L || tid >= N) return;

    int32_t z_val = z->polys[poly_idx].coeffs[tid];
    z_val = verify_freeze(z_val);

    // Center around 0
    if (z_val > (int32_t)Q / 2) {
        z_val = z_val - (int32_t)Q;
    }

    // Check |z| < gamma1 - beta
    if (abs(z_val) >= GAMMA1 - BETA) {
        atomicExch(valid, 0);
    }
}

// Check hint count is within bounds
__global__ void check_hint_count_kernel(
    const uint8_t* __restrict__ hints,
    uint32_t expected_count,
    int* __restrict__ valid
) {
    if (threadIdx.x != 0) return;

    // Count actual hint bits
    uint32_t count = 0;
    for (uint32_t i = 0; i < K * N / 8; i++) {
        count += __popc(hints[i]);
    }

    // Verify count matches
    if (count != expected_count || count > (K + L)) {
        *valid = 0;
    }
}

// ============================================================================
// Full Verification Pipeline
// ============================================================================

// Complete verification kernel
extern "C" __global__
void ringtail_verify_kernel(
    int* __restrict__ result,
    const VerifyKey* __restrict__ vk,
    const VerifySignature* __restrict__ sig,
    const uint8_t* __restrict__ msg_hash
) {
    extern __shared__ int32_t shmem[];

    const uint32_t tid = threadIdx.x;

    if (tid != 0) return;  // Single-threaded verification

    *result = 0;  // Assume invalid until proven valid

    // 1. Check z bounds
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < N; j++) {
            int32_t z_val = verify_freeze(sig->z.polys[i].coeffs[j]);
            if (z_val > (int32_t)Q / 2) z_val -= Q;
            if (abs(z_val) >= GAMMA1 - BETA) {
                return;  // Invalid: z out of bounds
            }
        }
    }

    // 2. Check hint count
    uint32_t hint_count = 0;
    for (uint32_t i = 0; i < K * N / 8; i++) {
        hint_count += __popc(sig->hint[i]);
    }
    if (hint_count != sig->hint_count || hint_count > K + L) {
        return;  // Invalid: hint count mismatch
    }

    // 3. Compute A*z
    VerifyPolyVecK Az;
    int32_t* temp = shmem;

    for (int i = 0; i < K; i++) {
        for (int k = 0; k < N; k++) temp[k] = 0;

        for (int j = 0; j < L; j++) {
            // NTT(z[j])
            int32_t z_ntt[N];
            for (int k = 0; k < N; k++) z_ntt[k] = sig->z.polys[j].coeffs[k];
            verify_ntt_forward(z_ntt);

            // Accumulate A[i][j] * z[j]
            for (int k = 0; k < N; k++) {
                int32_t a_val = vk->A_ntt[i][j].coeffs[k];
                temp[k] = verify_mod_add(temp[k], verify_mod_mul(a_val, z_ntt[k]));
            }
        }

        verify_ntt_inverse(temp);
        for (int k = 0; k < N; k++) Az.polys[i].coeffs[k] = temp[k];
    }

    // 4. Compute c * t1 * 2^d
    int32_t c_ntt[N];
    for (int k = 0; k < N; k++) c_ntt[k] = sig->c.coeffs[k];
    verify_ntt_forward(c_ntt);

    VerifyPolyVecK ct1;
    for (int i = 0; i < K; i++) {
        int32_t t1_scaled[N];
        for (int k = 0; k < N; k++) {
            t1_scaled[k] = verify_mod_mul(vk->t1.polys[i].coeffs[k], 1 << D);
        }
        verify_ntt_forward(t1_scaled);

        int32_t product[N];
        verify_ntt_pointwise(product, c_ntt, t1_scaled);
        verify_ntt_inverse(product);

        for (int k = 0; k < N; k++) ct1.polys[i].coeffs[k] = product[k];
    }

    // 5. Compute w' = Az - ct1 and apply hints
    VerifyPolyVecK w1_prime;
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            int32_t w_prime = verify_mod_sub(Az.polys[i].coeffs[j], ct1.polys[i].coeffs[j]);

            uint32_t idx = i * N + j;
            uint32_t byte_idx = idx / 8;
            uint32_t bit_idx = idx % 8;
            int32_t hint = (sig->hint[byte_idx] >> bit_idx) & 1;

            w1_prime.polys[i].coeffs[j] = verify_use_hint(hint, w_prime);
        }
    }

    // 6. Recompute challenge and verify
    uint32_t hash_state = 0;
    for (int i = 0; i < 64; i++) {
        hash_state = hash_state * 31 + msg_hash[i];
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            hash_state = hash_state * 31 + w1_prime.polys[i].coeffs[j];
        }
    }

    // Regenerate challenge
    VerifyPoly c_recomputed;
    for (int i = 0; i < N; i++) c_recomputed.coeffs[i] = 0;

    uint32_t rng = hash_state;
    for (int i = N - TAU; i < N; i++) {
        rng = rng * 1103515245 + 12345;
        int j = rng % (i + 1);
        c_recomputed.coeffs[i] = c_recomputed.coeffs[j];
        rng = rng * 1103515245 + 12345;
        c_recomputed.coeffs[j] = (rng & 1) ? 1 : (int32_t)Q - 1;
    }

    // Compare
    bool valid = true;
    for (int i = 0; i < N; i++) {
        if (c_recomputed.coeffs[i] != sig->c.coeffs[i]) {
            valid = false;
            break;
        }
    }

    *result = valid ? 1 : 0;
}

// Batch verification kernel
extern "C" __global__
void ringtail_batch_verify_kernel(
    int* __restrict__ results,
    const VerifyKey* __restrict__ vks,
    const VerifySignature* __restrict__ sigs,
    const uint8_t* __restrict__ msg_hashes,
    uint32_t count
) {
    const uint32_t idx = blockIdx.x;
    if (idx >= count) return;

    // Each block verifies one signature
    extern __shared__ int32_t shmem[];

    // Dispatch to single verification
    ringtail_verify_kernel<<<1, 1, 4 * N * sizeof(int32_t)>>>(
        &results[idx],
        &vks[idx],
        &sigs[idx],
        &msg_hashes[idx * 64]
    );
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

// Initialize verification constants
cudaError_t lux_cuda_ringtail_verify_init(void) {
    using namespace lux::cuda::ringtail;

    // Precompute twiddle factors
    uint32_t twiddles[N];
    uint32_t twiddles_inv[N];

    uint64_t omega = ROOT;
    uint64_t omega_inv = ROOT_INV;

    uint64_t pow = 1;
    uint64_t pow_inv = 1;

    for (int i = 0; i < N; i++) {
        twiddles[i] = (uint32_t)pow;
        twiddles_inv[i] = (uint32_t)pow_inv;

        pow = (pow * omega) % Q;
        pow_inv = (pow_inv * omega_inv) % Q;
    }

    cudaMemcpyToSymbol(d_verify_twiddles, twiddles, sizeof(twiddles));
    cudaMemcpyToSymbol(d_verify_twiddles_inv, twiddles_inv, sizeof(twiddles_inv));

    return cudaGetLastError();
}

// Verify a single signature
cudaError_t lux_cuda_ringtail_verify(
    int* result,
    const void* vk,              // Verification key
    const void* sig,             // Signature
    const uint8_t* msg_hash      // 64-byte message hash
) {
    using namespace lux::cuda::ringtail;

    // Allocate device memory
    int* d_result;
    VerifyKey* d_vk;
    VerifySignature* d_sig;
    uint8_t* d_msg;

    cudaMalloc(&d_result, sizeof(int));
    cudaMalloc(&d_vk, sizeof(VerifyKey));
    cudaMalloc(&d_sig, sizeof(VerifySignature));
    cudaMalloc(&d_msg, 64);

    // Copy inputs
    cudaMemcpy(d_vk, vk, sizeof(VerifyKey), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sig, sig, sizeof(VerifySignature), cudaMemcpyHostToDevice);
    cudaMemcpy(d_msg, msg_hash, 64, cudaMemcpyHostToDevice);

    // Launch kernel
    size_t shmem = 4 * N * sizeof(int32_t);
    ringtail_verify_kernel<<<1, 256, shmem>>>(d_result, d_vk, d_sig, d_msg);

    // Copy result
    cudaMemcpy(result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_result);
    cudaFree(d_vk);
    cudaFree(d_sig);
    cudaFree(d_msg);

    return cudaGetLastError();
}

// Batch verify multiple signatures
cudaError_t lux_cuda_ringtail_verify_batch(
    int* results,                // [count] output results
    const void* vks,             // [count] verification keys
    const void* sigs,            // [count] signatures
    const uint8_t* msg_hashes,   // [count * 64] message hashes
    uint32_t count
) {
    using namespace lux::cuda::ringtail;

    // Allocate device memory
    int* d_results;
    VerifyKey* d_vks;
    VerifySignature* d_sigs;
    uint8_t* d_msgs;

    cudaMalloc(&d_results, count * sizeof(int));
    cudaMalloc(&d_vks, count * sizeof(VerifyKey));
    cudaMalloc(&d_sigs, count * sizeof(VerifySignature));
    cudaMalloc(&d_msgs, count * 64);

    // Copy inputs
    cudaMemcpy(d_vks, vks, count * sizeof(VerifyKey), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigs, sigs, count * sizeof(VerifySignature), cudaMemcpyHostToDevice);
    cudaMemcpy(d_msgs, msg_hashes, count * 64, cudaMemcpyHostToDevice);

    // Launch one block per signature
    size_t shmem = 4 * N * sizeof(int32_t);
    ringtail_batch_verify_kernel<<<count, 256, shmem>>>(
        d_results, d_vks, d_sigs, d_msgs, count
    );

    // Copy results
    cudaMemcpy(results, d_results, count * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_results);
    cudaFree(d_vks);
    cudaFree(d_sigs);
    cudaFree(d_msgs);

    return cudaGetLastError();
}

// Cleanup
cudaError_t lux_cuda_ringtail_verify_cleanup(void) {
    return cudaSuccess;
}

// Get verification parameters
void lux_cuda_ringtail_verify_get_params(
    uint32_t* q,
    uint32_t* n,
    uint32_t* k,
    uint32_t* l,
    int32_t* gamma1,
    int32_t* beta
) {
    *q = lux::cuda::ringtail::Q;
    *n = lux::cuda::ringtail::N;
    *k = lux::cuda::ringtail::K;
    *l = lux::cuda::ringtail::L;
    *gamma1 = lux::cuda::ringtail::GAMMA1;
    *beta = lux::cuda::ringtail::BETA;
}

#ifdef __cplusplus
}
#endif
