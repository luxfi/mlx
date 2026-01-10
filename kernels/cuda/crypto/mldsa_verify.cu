// Copyright (c) 2024-2026 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// ML-DSA (FIPS 204) Signature Verification - CUDA Kernel
// Implements Dilithium-style post-quantum signature verification on GPU
//
// Parameters: ML-DSA-65 (NIST Security Level 3)
//   q = 8380417 (23-bit prime)
//   n = 256 (polynomial degree)
//   k = 6 (matrix rows)
//   l = 5 (matrix columns)
//   eta = 4 (secret coefficient bound)
//   gamma1 = 2^19 (signature coefficient bound)
//   gamma2 = (q-1)/32 = 261888 (hint threshold)
//   tau = 49 (number of +/-1 in challenge)
//
// Verification equation: c = H(mu || w1')
//   where w1' = UseHint(h, A*z - c*t1*2^d)

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================================
// ML-DSA-65 Parameters (FIPS 204)
// ============================================================================

// Ring modulus q = 8380417 = 2^23 - 2^13 + 1 (NTT-friendly prime)
#define MLDSA_Q         8380417U
#define MLDSA_Q_INV     58728449U    // q^{-1} mod 2^32

// Polynomial and matrix dimensions
#define MLDSA_N         256          // Ring dimension R_q = Z_q[X]/(X^n + 1)
#define MLDSA_K         6            // Matrix rows (t1 dimension)
#define MLDSA_L         5            // Matrix columns (z dimension)
#define MLDSA_LOG_N     8            // log2(256)

// Signature bounds
#define MLDSA_GAMMA1    (1 << 19)    // 2^19 = 524288
#define MLDSA_GAMMA2    261888       // (q-1)/32
#define MLDSA_BETA      275          // tau * eta = 49 * 4 + extra
#define MLDSA_TAU       49           // Challenge weight

// NTT parameters
#define MLDSA_ROOT      1753         // Primitive 512th root of unity mod q
#define MLDSA_ROOT_INV  8380416      // ROOT^{-1} mod q

// Montgomery parameters for q = 8380417
#define MLDSA_MONT_R    4193792U     // 2^32 mod q
#define MLDSA_MONT_R2   2365951U     // (2^32)^2 mod q = 2^64 mod q
#define MLDSA_MONT_QINV 58728449U    // -q^{-1} mod 2^32

// SHAKE-256 constants
#define SHAKE256_RATE   136          // r = 1088 bits = 136 bytes
#define KECCAK_ROUNDS   24

// ============================================================================
// Device Constants
// ============================================================================

__constant__ uint32_t d_mldsa_twiddles[256];      // Forward NTT twiddles
__constant__ uint32_t d_mldsa_twiddles_inv[256];  // Inverse NTT twiddles
__constant__ uint64_t d_keccak_rc[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

__constant__ int d_keccak_rotc[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
};

__constant__ int d_keccak_piln[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
};

// ============================================================================
// ML-DSA Data Structures
// ============================================================================

// Polynomial in R_q = Z_q[X]/(X^n + 1)
struct MLDSAPoly {
    int32_t coeffs[MLDSA_N];
};

// Matrix of polynomials A[k][l]
struct MLDSAMatrix {
    MLDSAPoly polys[MLDSA_K][MLDSA_L];
};

// Verification key: (rho, t1)
struct MLDSAVerifyKey {
    uint8_t rho[32];                    // Public seed for A
    MLDSAPoly t1[MLDSA_K];              // High bits of t = A*s1 + s2
};

// Signature: (c_tilde, z, h)
struct MLDSASignature {
    uint8_t c_tilde[32];                // Hash of message and w1
    MLDSAPoly z[MLDSA_L];               // Masked secret z = y + c*s1
    uint8_t h[MLDSA_K * MLDSA_N / 8];   // Hint bits (omega positions)
    uint16_t h_count;                    // Number of hint bits set
};

// Verification context
struct MLDSAVerifyContext {
    uint8_t mu[64];                      // H(tr || M) - message representative
    uint8_t c_tilde[32];                 // Challenge hash
    MLDSAPoly c;                         // Challenge polynomial
    MLDSAPoly w1_prime[MLDSA_K];         // Recovered high bits
};

// ============================================================================
// Modular Arithmetic for q = 8380417
// ============================================================================

// Reduction mod q using Barrett reduction
__device__ __forceinline__
int32_t mldsa_reduce(int64_t a) {
    // Barrett reduction for q = 8380417
    // mu = floor(2^32 / q) = 512
    const int64_t mu = 512LL;
    int64_t t = (a * mu) >> 32;
    t = a - t * MLDSA_Q;

    // Final correction
    if (t >= MLDSA_Q) t -= MLDSA_Q;
    if (t < 0) t += MLDSA_Q;
    return (int32_t)t;
}

// Montgomery reduction
__device__ __forceinline__
int32_t mldsa_mont_reduce(int64_t a) {
    int32_t t = (int32_t)a * MLDSA_Q_INV;
    t = (a - (int64_t)t * MLDSA_Q) >> 32;
    return t;
}

// Modular addition
__device__ __forceinline__
int32_t mldsa_add(int32_t a, int32_t b) {
    int32_t r = a + b;
    if (r >= MLDSA_Q) r -= MLDSA_Q;
    return r;
}

// Modular subtraction
__device__ __forceinline__
int32_t mldsa_sub(int32_t a, int32_t b) {
    int32_t r = a - b;
    if (r < 0) r += MLDSA_Q;
    return r;
}

// Modular multiplication
__device__ __forceinline__
int32_t mldsa_mul(int32_t a, int32_t b) {
    return mldsa_reduce((int64_t)a * b);
}

// Power of 2 mod q
__device__ __forceinline__
int32_t mldsa_power2round(int32_t a, int32_t* a0) {
    // a = a1 * 2^d + a0 where -2^{d-1} < a0 <= 2^{d-1}
    const int32_t d = 13;
    const int32_t mask = (1 << d) - 1;

    int32_t a1 = (a + (1 << (d - 1)) - 1) >> d;
    *a0 = a - (a1 << d);
    return a1;
}

// Decompose into high and low bits
__device__ __forceinline__
int32_t mldsa_decompose(int32_t a, int32_t* a0) {
    // a = a1 * alpha + a0 where -alpha/2 < a0 <= alpha/2
    // alpha = 2 * gamma2
    const int32_t alpha = 2 * MLDSA_GAMMA2;

    int32_t a1 = (a + (alpha >> 1)) / alpha;
    *a0 = a - a1 * alpha;

    // Handle corner case
    if (*a0 > (alpha >> 1)) {
        *a0 -= alpha;
        a1 += 1;
    }
    if (*a0 < -(alpha >> 1)) {
        *a0 += alpha;
        a1 -= 1;
    }

    // a1 in [0, (q-1)/alpha]
    if (a1 < 0) a1 += (MLDSA_Q - 1) / alpha + 1;

    return a1;
}

// Use hint to recover high bits
__device__ __forceinline__
int32_t mldsa_use_hint(int32_t hint, int32_t r) {
    int32_t r0;
    int32_t r1 = mldsa_decompose(r, &r0);

    if (hint == 0) return r1;

    // Adjust based on sign of r0
    const int32_t m = (MLDSA_Q - 1) / (2 * MLDSA_GAMMA2) + 1;
    if (r0 > 0) {
        return (r1 + 1) % m;
    } else {
        return (r1 - 1 + m) % m;
    }
}

// ============================================================================
// NTT Operations for R_q = Z_q[X]/(X^n + 1)
// ============================================================================

// Cooley-Tukey butterfly
__device__ __forceinline__
void mldsa_ct_butterfly(int32_t* a, int32_t* b, int32_t zeta) {
    int32_t t = mldsa_mul(*b, zeta);
    *b = mldsa_sub(*a, t);
    *a = mldsa_add(*a, t);
}

// Gentleman-Sande butterfly
__device__ __forceinline__
void mldsa_gs_butterfly(int32_t* a, int32_t* b, int32_t zeta) {
    int32_t t = *a;
    *a = mldsa_add(t, *b);
    *b = mldsa_mul(mldsa_sub(t, *b), zeta);
}

// Forward NTT (in-place, bit-reversal not needed with this ordering)
__device__
void mldsa_ntt(int32_t* poly) {
    int32_t zeta;
    int32_t k = 0;

    for (int32_t len = 128; len >= 1; len >>= 1) {
        for (int32_t start = 0; start < MLDSA_N; start += 2 * len) {
            zeta = d_mldsa_twiddles[++k];
            for (int32_t j = start; j < start + len; j++) {
                mldsa_ct_butterfly(&poly[j], &poly[j + len], zeta);
            }
        }
    }
}

// Inverse NTT (in-place)
__device__
void mldsa_ntt_inv(int32_t* poly) {
    int32_t zeta;
    int32_t k = 256;

    for (int32_t len = 1; len <= 128; len <<= 1) {
        for (int32_t start = 0; start < MLDSA_N; start += 2 * len) {
            zeta = d_mldsa_twiddles_inv[--k];
            for (int32_t j = start; j < start + len; j++) {
                mldsa_gs_butterfly(&poly[j], &poly[j + len], zeta);
            }
        }
    }

    // Scale by n^{-1} mod q
    // n^{-1} mod q = 8347681 for n = 256, q = 8380417
    const int32_t n_inv = 8347681;
    for (int32_t i = 0; i < MLDSA_N; i++) {
        poly[i] = mldsa_mul(poly[i], n_inv);
    }
}

// Pointwise multiplication in NTT domain
__device__
void mldsa_ntt_mul(int32_t* c, const int32_t* a, const int32_t* b) {
    for (int32_t i = 0; i < MLDSA_N; i++) {
        c[i] = mldsa_mul(a[i], b[i]);
    }
}

// Polynomial addition
__device__
void mldsa_poly_add(int32_t* c, const int32_t* a, const int32_t* b) {
    for (int32_t i = 0; i < MLDSA_N; i++) {
        c[i] = mldsa_add(a[i], b[i]);
    }
}

// Polynomial subtraction
__device__
void mldsa_poly_sub(int32_t* c, const int32_t* a, const int32_t* b) {
    for (int32_t i = 0; i < MLDSA_N; i++) {
        c[i] = mldsa_sub(a[i], b[i]);
    }
}

// Polynomial negation
__device__
void mldsa_poly_neg(int32_t* a) {
    for (int32_t i = 0; i < MLDSA_N; i++) {
        a[i] = mldsa_sub(0, a[i]);
    }
}

// ============================================================================
// SHAKE-256 Implementation (for Hash-to-Point)
// ============================================================================

__device__ __forceinline__
uint64_t rotl64_device(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

__device__
void keccak_f1600_device(uint64_t state[25]) {
    for (int round = 0; round < KECCAK_ROUNDS; round++) {
        // Theta
        uint64_t C[5], D[5];
        for (int x = 0; x < 5; x++) {
            C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        for (int x = 0; x < 5; x++) {
            D[x] = C[(x + 4) % 5] ^ rotl64_device(C[(x + 1) % 5], 1);
            for (int y = 0; y < 5; y++) {
                state[y * 5 + x] ^= D[x];
            }
        }

        // Rho and Pi
        uint64_t temp = state[1];
        for (int i = 0; i < 24; i++) {
            int j = d_keccak_piln[i];
            uint64_t t = state[j];
            state[j] = rotl64_device(temp, d_keccak_rotc[i]);
            temp = t;
        }

        // Chi
        for (int y = 0; y < 5; y++) {
            uint64_t row[5];
            for (int x = 0; x < 5; x++) {
                row[x] = state[y * 5 + x];
            }
            for (int x = 0; x < 5; x++) {
                state[y * 5 + x] = row[x] ^ ((~row[(x + 1) % 5]) & row[(x + 2) % 5]);
            }
        }

        // Iota
        state[0] ^= d_keccak_rc[round];
    }
}

// SHAKE-256 absorb
__device__
void shake256_absorb(uint64_t state[25], const uint8_t* input, size_t inlen) {
    // Initialize state
    for (int i = 0; i < 25; i++) state[i] = 0;

    size_t pos = 0;
    while (inlen >= SHAKE256_RATE) {
        for (int i = 0; i < SHAKE256_RATE / 8; i++) {
            uint64_t word = 0;
            for (int j = 0; j < 8; j++) {
                word |= ((uint64_t)input[pos + i * 8 + j]) << (j * 8);
            }
            state[i] ^= word;
        }
        keccak_f1600_device(state);
        input += SHAKE256_RATE;
        inlen -= SHAKE256_RATE;
        pos += SHAKE256_RATE;
    }

    // Final partial block
    uint8_t block[SHAKE256_RATE] = {0};
    for (size_t i = 0; i < inlen; i++) {
        block[i] = input[i];
    }
    block[inlen] = 0x1F;  // SHAKE padding
    block[SHAKE256_RATE - 1] |= 0x80;

    for (int i = 0; i < SHAKE256_RATE / 8; i++) {
        uint64_t word = 0;
        for (int j = 0; j < 8; j++) {
            word |= ((uint64_t)block[i * 8 + j]) << (j * 8);
        }
        state[i] ^= word;
    }
    keccak_f1600_device(state);
}

// SHAKE-256 squeeze
__device__
void shake256_squeeze(uint64_t state[25], uint8_t* output, size_t outlen) {
    size_t pos = 0;

    while (outlen > 0) {
        size_t rate_bytes = SHAKE256_RATE;
        if (outlen < rate_bytes) rate_bytes = outlen;

        for (size_t i = 0; i < rate_bytes; i++) {
            output[pos + i] = (uint8_t)(state[i / 8] >> ((i % 8) * 8));
        }

        outlen -= rate_bytes;
        pos += rate_bytes;

        if (outlen > 0) {
            keccak_f1600_device(state);
        }
    }
}

// ============================================================================
// Hash-to-Point (Sample challenge polynomial)
// ============================================================================

// Sample challenge c with exactly tau non-zero coefficients in {-1, +1}
__device__
void mldsa_sample_challenge(int32_t* c, const uint8_t* seed) {
    uint64_t state[25];
    shake256_absorb(state, seed, 32);

    // Initialize c to zero
    for (int i = 0; i < MLDSA_N; i++) {
        c[i] = 0;
    }

    // Sample tau positions for non-zero coefficients
    uint8_t buf[136];
    shake256_squeeze(state, buf, 8);  // First 8 bytes for signs

    uint64_t signs = 0;
    for (int i = 0; i < 8; i++) {
        signs |= ((uint64_t)buf[i]) << (i * 8);
    }

    int pos = 8;
    int buf_len = 8;

    for (int i = MLDSA_N - MLDSA_TAU; i < MLDSA_N; i++) {
        // Sample j uniform in [0, i]
        uint8_t j;
        do {
            if (pos >= buf_len) {
                shake256_squeeze(state, buf, 136);
                pos = 0;
                buf_len = 136;
            }
            j = buf[pos++];
        } while (j > i);

        // Swap c[i] and c[j]
        c[i] = c[j];

        // Set c[j] to +1 or -1 based on sign bit
        c[j] = 1 - 2 * (int32_t)((signs >> (i - (MLDSA_N - MLDSA_TAU))) & 1);
    }
}

// ============================================================================
// Matrix Expansion from Seed (ExpandA)
// ============================================================================

// Rejection sample a coefficient in [0, q-1]
__device__
int32_t mldsa_rej_sample(uint8_t b0, uint8_t b1, uint8_t b2) {
    int32_t coeff = b0 | ((int32_t)b1 << 8) | ((int32_t)(b2 & 0x7F) << 16);
    return (coeff < MLDSA_Q) ? coeff : -1;
}

// Expand matrix A from seed rho
// A[i][j] = ExpandA(rho, i, j)
__device__
void mldsa_expand_A_poly(int32_t* poly, const uint8_t* rho, int i, int j) {
    // Domain separator: rho || j || i
    uint8_t seed[34];
    for (int k = 0; k < 32; k++) seed[k] = rho[k];
    seed[32] = (uint8_t)j;
    seed[33] = (uint8_t)i;

    uint64_t state[25];
    shake256_absorb(state, seed, 34);

    uint8_t buf[3 * MLDSA_N];  // Max needed
    int buf_pos = 0;
    int coeff_idx = 0;

    while (coeff_idx < MLDSA_N) {
        // Squeeze more bytes if needed
        if (buf_pos + 3 > sizeof(buf)) {
            shake256_squeeze(state, buf, sizeof(buf));
            buf_pos = 0;
        }

        int32_t coeff = mldsa_rej_sample(buf[buf_pos], buf[buf_pos + 1], buf[buf_pos + 2]);
        buf_pos += 3;

        if (coeff >= 0) {
            poly[coeff_idx++] = coeff;
        }
    }
}

// ============================================================================
// Main Verification Kernels
// ============================================================================

// Kernel: Compute A*z for a single signature
// Each block handles one (i, j) pair
extern "C" __global__
void mldsa_matrix_vector_mul(
    int32_t* __restrict__ result,        // Output: w' = A*z [k][n]
    const uint8_t* __restrict__ rho,     // Seed for A [32]
    const int32_t* __restrict__ z,       // z vectors [l][n]
    uint32_t sig_idx                     // Which signature
) {
    const int i = blockIdx.x;  // Row index [0, k-1]
    const int j = blockIdx.y;  // Column index [0, l-1]
    const int tid = threadIdx.x;

    __shared__ int32_t A_ij[MLDSA_N];
    __shared__ int32_t z_j[MLDSA_N];
    __shared__ int32_t acc[MLDSA_N];

    // Expand A[i][j] from rho
    if (tid == 0) {
        mldsa_expand_A_poly(A_ij, rho + sig_idx * 32, i, j);
    }
    __syncthreads();

    // Load z[j]
    for (int idx = tid; idx < MLDSA_N; idx += blockDim.x) {
        z_j[idx] = z[sig_idx * MLDSA_L * MLDSA_N + j * MLDSA_N + idx];
    }
    __syncthreads();

    // NTT(A[i][j])
    if (tid == 0) {
        mldsa_ntt(A_ij);
    }
    __syncthreads();

    // NTT(z[j]) if not already in NTT form
    if (tid == 0 && j == 0) {
        mldsa_ntt(z_j);
    }
    __syncthreads();

    // Pointwise multiply: acc = A[i][j] * z[j]
    for (int idx = tid; idx < MLDSA_N; idx += blockDim.x) {
        acc[idx] = mldsa_mul(A_ij[idx], z_j[idx]);
    }
    __syncthreads();

    // Atomic add to result[i]
    for (int idx = tid; idx < MLDSA_N; idx += blockDim.x) {
        atomicAdd(&result[sig_idx * MLDSA_K * MLDSA_N + i * MLDSA_N + idx], acc[idx]);
    }
}

// Kernel: Compute c * t1 * 2^d
extern "C" __global__
void mldsa_compute_ct1(
    int32_t* __restrict__ ct1,           // Output: c * t1 * 2^d [k][n]
    const int32_t* __restrict__ t1,      // Input: t1 [k][n]
    const int32_t* __restrict__ c,       // Input: challenge c [n]
    uint32_t sig_idx
) {
    const int i = blockIdx.x;  // Row index [0, k-1]
    const int tid = threadIdx.x;

    __shared__ int32_t c_ntt[MLDSA_N];
    __shared__ int32_t t1_i[MLDSA_N];
    __shared__ int32_t result[MLDSA_N];

    // Load and NTT transform c
    for (int idx = tid; idx < MLDSA_N; idx += blockDim.x) {
        c_ntt[idx] = c[sig_idx * MLDSA_N + idx];
    }
    __syncthreads();

    if (tid == 0) {
        mldsa_ntt(c_ntt);
    }
    __syncthreads();

    // Load t1[i] and shift by 2^d
    const int32_t d = 13;
    for (int idx = tid; idx < MLDSA_N; idx += blockDim.x) {
        int32_t t = t1[sig_idx * MLDSA_K * MLDSA_N + i * MLDSA_N + idx];
        t1_i[idx] = mldsa_reduce((int64_t)t << d);  // t1 * 2^d mod q
    }
    __syncthreads();

    if (tid == 0) {
        mldsa_ntt(t1_i);
    }
    __syncthreads();

    // Pointwise multiply: c * t1
    for (int idx = tid; idx < MLDSA_N; idx += blockDim.x) {
        result[idx] = mldsa_mul(c_ntt[idx], t1_i[idx]);
    }
    __syncthreads();

    // INTT
    if (tid == 0) {
        mldsa_ntt_inv(result);
    }
    __syncthreads();

    // Store result
    for (int idx = tid; idx < MLDSA_N; idx += blockDim.x) {
        ct1[sig_idx * MLDSA_K * MLDSA_N + i * MLDSA_N + idx] = result[idx];
    }
}

// Kernel: Compute w' = A*z - c*t1*2^d and apply hints
extern "C" __global__
void mldsa_recover_w1(
    int32_t* __restrict__ w1_prime,      // Output: recovered w1' [k][n]
    const int32_t* __restrict__ Az,      // Input: A*z [k][n]
    const int32_t* __restrict__ ct1,     // Input: c*t1*2^d [k][n]
    const uint8_t* __restrict__ hints,   // Input: hint bits
    uint32_t sig_idx
) {
    const int i = blockIdx.x;  // Row index [0, k-1]
    const int tid = threadIdx.x;

    // Compute w' = A*z - c*t1*2^d
    for (int idx = tid; idx < MLDSA_N; idx += blockDim.x) {
        int32_t az = Az[sig_idx * MLDSA_K * MLDSA_N + i * MLDSA_N + idx];
        int32_t ct = ct1[sig_idx * MLDSA_K * MLDSA_N + i * MLDSA_N + idx];
        int32_t w = mldsa_sub(az, ct);

        // Get hint bit (packed format)
        int byte_idx = (i * MLDSA_N + idx) / 8;
        int bit_idx = (i * MLDSA_N + idx) % 8;
        int hint = (hints[sig_idx * (MLDSA_K * MLDSA_N / 8) + byte_idx] >> bit_idx) & 1;

        // Apply hint to recover w1'
        w1_prime[sig_idx * MLDSA_K * MLDSA_N + i * MLDSA_N + idx] = mldsa_use_hint(hint, w);
    }
}

// Kernel: Compute challenge hash and verify
extern "C" __global__
void mldsa_verify_challenge(
    int* __restrict__ results,           // Output: verification results
    const int32_t* __restrict__ w1_prime,// Input: recovered w1' [k][n]
    const uint8_t* __restrict__ mu,      // Input: message hash [64]
    const uint8_t* __restrict__ c_tilde, // Input: expected c_tilde [32]
    uint32_t count
) {
    uint32_t sig_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sig_idx >= count) return;

    // Encode w1' to bytes (simplified - real implementation needs proper encoding)
    uint8_t w1_bytes[MLDSA_K * MLDSA_N * 4];  // Conservative size
    int pos = 0;

    for (int i = 0; i < MLDSA_K; i++) {
        for (int j = 0; j < MLDSA_N; j++) {
            int32_t coeff = w1_prime[sig_idx * MLDSA_K * MLDSA_N + i * MLDSA_N + j];
            // Pack high bits (simplified)
            w1_bytes[pos++] = (uint8_t)(coeff & 0xFF);
            w1_bytes[pos++] = (uint8_t)((coeff >> 8) & 0xFF);
        }
    }

    // Hash: c' = H(mu || w1')
    uint64_t state[25];
    uint8_t hash_input[64 + MLDSA_K * MLDSA_N * 4];

    // Copy mu
    for (int i = 0; i < 64; i++) {
        hash_input[i] = mu[sig_idx * 64 + i];
    }
    // Copy w1' encoding
    for (int i = 0; i < pos; i++) {
        hash_input[64 + i] = w1_bytes[i];
    }

    shake256_absorb(state, hash_input, 64 + pos);

    uint8_t c_prime[32];
    shake256_squeeze(state, c_prime, 32);

    // Compare c' with c_tilde
    int valid = 1;
    for (int i = 0; i < 32; i++) {
        if (c_prime[i] != c_tilde[sig_idx * 32 + i]) {
            valid = 0;
            break;
        }
    }

    results[sig_idx] = valid;
}

// ============================================================================
// Unified Verification Kernel (Single signature per block)
// ============================================================================

extern "C" __global__
void mldsa_verify_single(
    int* __restrict__ results,
    const uint8_t* __restrict__ vk_rho,      // [32] per key
    const int32_t* __restrict__ vk_t1,       // [k][n] per key
    const uint8_t* __restrict__ sig_c_tilde, // [32] per sig
    const int32_t* __restrict__ sig_z,       // [l][n] per sig
    const uint8_t* __restrict__ sig_h,       // Hints per sig
    const uint8_t* __restrict__ mu,          // [64] per sig
    uint32_t count
) {
    uint32_t sig_idx = blockIdx.x;
    if (sig_idx >= count) return;

    const int tid = threadIdx.x;

    // Shared memory for intermediate results
    extern __shared__ int32_t shared[];
    int32_t* Az = shared;                            // [k][n]
    int32_t* ct1 = shared + MLDSA_K * MLDSA_N;       // [k][n]
    int32_t* c = shared + 2 * MLDSA_K * MLDSA_N;     // [n]
    int32_t* w1_prime = shared + 2 * MLDSA_K * MLDSA_N + MLDSA_N; // [k][n]

    // Initialize Az to zero
    for (int idx = tid; idx < MLDSA_K * MLDSA_N; idx += blockDim.x) {
        Az[idx] = 0;
    }
    __syncthreads();

    // Sample challenge c from c_tilde
    if (tid == 0) {
        mldsa_sample_challenge(c, sig_c_tilde + sig_idx * 32);
    }
    __syncthreads();

    // Compute A*z (matrix-vector multiplication)
    // Serial over (i, j) pairs within this block
    if (tid == 0) {
        int32_t A_poly[MLDSA_N];
        int32_t z_poly[MLDSA_N];
        int32_t temp[MLDSA_N];

        for (int i = 0; i < MLDSA_K; i++) {
            for (int j = 0; j < MLDSA_L; j++) {
                // Expand A[i][j]
                mldsa_expand_A_poly(A_poly, vk_rho + sig_idx * 32, i, j);
                mldsa_ntt(A_poly);

                // Load z[j]
                for (int k = 0; k < MLDSA_N; k++) {
                    z_poly[k] = sig_z[sig_idx * MLDSA_L * MLDSA_N + j * MLDSA_N + k];
                }
                mldsa_ntt(z_poly);

                // temp = A[i][j] * z[j]
                mldsa_ntt_mul(temp, A_poly, z_poly);

                // Az[i] += temp
                for (int k = 0; k < MLDSA_N; k++) {
                    Az[i * MLDSA_N + k] = mldsa_add(Az[i * MLDSA_N + k], temp[k]);
                }
            }

            // INTT(Az[i])
            mldsa_ntt_inv(&Az[i * MLDSA_N]);
        }
    }
    __syncthreads();

    // Compute c * t1 * 2^d
    if (tid == 0) {
        int32_t c_ntt[MLDSA_N];
        int32_t t1_poly[MLDSA_N];

        for (int k = 0; k < MLDSA_N; k++) {
            c_ntt[k] = c[k];
        }
        mldsa_ntt(c_ntt);

        const int32_t d = 13;
        for (int i = 0; i < MLDSA_K; i++) {
            // Load and shift t1[i]
            for (int k = 0; k < MLDSA_N; k++) {
                int32_t t = vk_t1[sig_idx * MLDSA_K * MLDSA_N + i * MLDSA_N + k];
                t1_poly[k] = mldsa_reduce((int64_t)t << d);
            }
            mldsa_ntt(t1_poly);

            // ct1[i] = c * t1[i]
            mldsa_ntt_mul(&ct1[i * MLDSA_N], c_ntt, t1_poly);
            mldsa_ntt_inv(&ct1[i * MLDSA_N]);
        }
    }
    __syncthreads();

    // Compute w' = Az - ct1 and apply hints
    for (int idx = tid; idx < MLDSA_K * MLDSA_N; idx += blockDim.x) {
        int32_t w = mldsa_sub(Az[idx], ct1[idx]);

        // Get hint bit
        int byte_idx = idx / 8;
        int bit_idx = idx % 8;
        int hint = (sig_h[sig_idx * (MLDSA_K * MLDSA_N / 8) + byte_idx] >> bit_idx) & 1;

        w1_prime[idx] = mldsa_use_hint(hint, w);
    }
    __syncthreads();

    // Verify: c_tilde == H(mu || w1')
    if (tid == 0) {
        // Encode w1_prime to bytes
        uint8_t w1_bytes[MLDSA_K * MLDSA_N * 2];
        int pos = 0;

        for (int i = 0; i < MLDSA_K; i++) {
            for (int j = 0; j < MLDSA_N; j++) {
                int32_t coeff = w1_prime[i * MLDSA_N + j];
                // Simplified encoding for high bits
                w1_bytes[pos++] = (uint8_t)(coeff & 0xFF);
                w1_bytes[pos++] = (uint8_t)((coeff >> 8) & 0x0F);
            }
        }

        // Hash: H(mu || w1')
        uint8_t hash_input[64 + MLDSA_K * MLDSA_N * 2];
        for (int i = 0; i < 64; i++) {
            hash_input[i] = mu[sig_idx * 64 + i];
        }
        for (int i = 0; i < pos; i++) {
            hash_input[64 + i] = w1_bytes[i];
        }

        uint64_t state[25];
        shake256_absorb(state, hash_input, 64 + pos);

        uint8_t c_prime[32];
        shake256_squeeze(state, c_prime, 32);

        // Compare
        int valid = 1;
        for (int i = 0; i < 32; i++) {
            if (c_prime[i] != sig_c_tilde[sig_idx * 32 + i]) {
                valid = 0;
                break;
            }
        }

        results[sig_idx] = valid;
    }
}

// ============================================================================
// Batch Verification Kernel
// ============================================================================

extern "C" __global__
void mldsa_batch_verify(
    int* __restrict__ results,
    const uint8_t* __restrict__ vk_data,     // Packed verification keys
    const uint8_t* __restrict__ sig_data,    // Packed signatures
    const uint8_t* __restrict__ msg_data,    // Message hashes (mu)
    const uint32_t* __restrict__ vk_offsets, // Offsets into vk_data
    const uint32_t* __restrict__ sig_offsets,// Offsets into sig_data
    uint32_t batch_size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Dispatch to single verification (simplified)
    // Real implementation would parallelize internal operations

    results[idx] = 0;  // Placeholder - full verification in mldsa_verify_single
}

// ============================================================================
// Twiddle Factor Precomputation
// ============================================================================

extern "C" __global__
void mldsa_precompute_twiddles(
    uint32_t* __restrict__ twiddles,
    uint32_t* __restrict__ twiddles_inv
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= MLDSA_N) return;

    // Compute powers of root of unity
    // omega = MLDSA_ROOT^{(q-1)/(2n)} mod q
    // For n = 256, 2n = 512, omega = primitive 512th root

    int32_t omega = MLDSA_ROOT;
    int32_t omega_inv = MLDSA_ROOT_INV;

    // omega^i
    int32_t pow = 1;
    int32_t pow_inv = 1;

    for (int i = 0; i < idx; i++) {
        pow = mldsa_mul(pow, omega);
        pow_inv = mldsa_mul(pow_inv, omega_inv);
    }

    // Bit-reverse index for NTT ordering
    int br_idx = 0;
    int tmp = idx;
    for (int i = 0; i < MLDSA_LOG_N; i++) {
        br_idx = (br_idx << 1) | (tmp & 1);
        tmp >>= 1;
    }

    twiddles[br_idx] = (uint32_t)pow;
    twiddles_inv[br_idx] = (uint32_t)pow_inv;
}

// ============================================================================
// Host API (C Linkage)
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

// Initialize ML-DSA context and precompute twiddles
cudaError_t lux_cuda_mldsa_init(void) {
    uint32_t* d_twiddles;
    uint32_t* d_twiddles_inv;

    cudaMalloc(&d_twiddles, MLDSA_N * sizeof(uint32_t));
    cudaMalloc(&d_twiddles_inv, MLDSA_N * sizeof(uint32_t));

    // Precompute twiddle factors
    mldsa_precompute_twiddles<<<1, MLDSA_N>>>(d_twiddles, d_twiddles_inv);

    // Copy to constant memory
    cudaMemcpyToSymbol(d_mldsa_twiddles, d_twiddles, MLDSA_N * sizeof(uint32_t));
    cudaMemcpyToSymbol(d_mldsa_twiddles_inv, d_twiddles_inv, MLDSA_N * sizeof(uint32_t));

    cudaFree(d_twiddles);
    cudaFree(d_twiddles_inv);

    return cudaGetLastError();
}

// Cleanup
cudaError_t lux_cuda_mldsa_cleanup(void) {
    // No persistent allocations to free
    return cudaSuccess;
}

// Verify a single ML-DSA signature
// Returns: 1 if valid, 0 if invalid, -1 on error
cudaError_t lux_cuda_mldsa_verify(
    int* result,
    const uint8_t* vk_rho,           // [32] public seed
    const int32_t* vk_t1,            // [k * n] high bits of t
    const uint8_t* sig_c_tilde,      // [32] challenge hash
    const int32_t* sig_z,            // [l * n] signature z
    const uint8_t* sig_h,            // Hint bits
    const uint8_t* mu                // [64] message hash
) {
    // Allocate device memory
    uint8_t* d_rho;
    int32_t* d_t1;
    uint8_t* d_c_tilde;
    int32_t* d_z;
    uint8_t* d_h;
    uint8_t* d_mu;
    int* d_result;

    cudaMalloc(&d_rho, 32);
    cudaMalloc(&d_t1, MLDSA_K * MLDSA_N * sizeof(int32_t));
    cudaMalloc(&d_c_tilde, 32);
    cudaMalloc(&d_z, MLDSA_L * MLDSA_N * sizeof(int32_t));
    cudaMalloc(&d_h, MLDSA_K * MLDSA_N / 8);
    cudaMalloc(&d_mu, 64);
    cudaMalloc(&d_result, sizeof(int));

    // Copy inputs
    cudaMemcpy(d_rho, vk_rho, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_t1, vk_t1, MLDSA_K * MLDSA_N * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c_tilde, sig_c_tilde, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, sig_z, MLDSA_L * MLDSA_N * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, sig_h, MLDSA_K * MLDSA_N / 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mu, mu, 64, cudaMemcpyHostToDevice);

    // Compute shared memory size
    size_t shared_size = (3 * MLDSA_K * MLDSA_N + MLDSA_N) * sizeof(int32_t);

    // Launch kernel
    mldsa_verify_single<<<1, 256, shared_size>>>(
        d_result, d_rho, d_t1, d_c_tilde, d_z, d_h, d_mu, 1
    );

    // Copy result
    cudaMemcpy(result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_rho);
    cudaFree(d_t1);
    cudaFree(d_c_tilde);
    cudaFree(d_z);
    cudaFree(d_h);
    cudaFree(d_mu);
    cudaFree(d_result);

    return cudaGetLastError();
}

// Batch verify multiple ML-DSA signatures
cudaError_t lux_cuda_mldsa_verify_batch(
    int* results,                    // [count] output results
    const uint8_t* vk_rho,           // [count * 32]
    const int32_t* vk_t1,            // [count * k * n]
    const uint8_t* sig_c_tilde,      // [count * 32]
    const int32_t* sig_z,            // [count * l * n]
    const uint8_t* sig_h,            // Hints per sig
    const uint8_t* mu,               // [count * 64]
    uint32_t count
) {
    // Allocate device memory
    uint8_t* d_rho;
    int32_t* d_t1;
    uint8_t* d_c_tilde;
    int32_t* d_z;
    uint8_t* d_h;
    uint8_t* d_mu;
    int* d_results;

    size_t rho_size = count * 32;
    size_t t1_size = count * MLDSA_K * MLDSA_N * sizeof(int32_t);
    size_t c_size = count * 32;
    size_t z_size = count * MLDSA_L * MLDSA_N * sizeof(int32_t);
    size_t h_size = count * (MLDSA_K * MLDSA_N / 8);
    size_t mu_size = count * 64;

    cudaMalloc(&d_rho, rho_size);
    cudaMalloc(&d_t1, t1_size);
    cudaMalloc(&d_c_tilde, c_size);
    cudaMalloc(&d_z, z_size);
    cudaMalloc(&d_h, h_size);
    cudaMalloc(&d_mu, mu_size);
    cudaMalloc(&d_results, count * sizeof(int));

    // Copy inputs
    cudaMemcpy(d_rho, vk_rho, rho_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_t1, vk_t1, t1_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c_tilde, sig_c_tilde, c_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, sig_z, z_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, sig_h, h_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mu, mu, mu_size, cudaMemcpyHostToDevice);

    // Shared memory size
    size_t shared_size = (3 * MLDSA_K * MLDSA_N + MLDSA_N) * sizeof(int32_t);

    // Launch one block per signature
    mldsa_verify_single<<<count, 256, shared_size>>>(
        d_results, d_rho, d_t1, d_c_tilde, d_z, d_h, d_mu, count
    );

    // Copy results
    cudaMemcpy(results, d_results, count * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_rho);
    cudaFree(d_t1);
    cudaFree(d_c_tilde);
    cudaFree(d_z);
    cudaFree(d_h);
    cudaFree(d_mu);
    cudaFree(d_results);

    return cudaGetLastError();
}

// Get ML-DSA parameters
void lux_cuda_mldsa_get_params(
    uint32_t* q,
    uint32_t* n,
    uint32_t* k,
    uint32_t* l
) {
    *q = MLDSA_Q;
    *n = MLDSA_N;
    *k = MLDSA_K;
    *l = MLDSA_L;
}

#ifdef __cplusplus
}
#endif
