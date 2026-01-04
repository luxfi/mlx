// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Ringtail Lattice-Based Threshold Signatures
// GPU-accelerated MLWE-based threshold signing operations
// Optimized for Apple Silicon GPUs
//
// Parameters from Ringtail specification:
// - Ring dimension N = 256 (power of 2 for NTT)
// - Modulus Q = 8380417 (23-bit prime, NTT-friendly)
// - Vector dimensions: M = 8, N_vec = 7
// - Gaussian parameter sigma for rejection sampling

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Ringtail Parameters
// ============================================================================

// Ring parameters
constant uint RING_N = 256;                  // Polynomial degree
constant ulong RING_Q = 8380417UL;           // Modulus (23-bit prime)
constant ulong RING_Q_INV = 58728449UL;      // Montgomery inverse
constant uint LOG_N = 8;                     // log2(RING_N)

// Vector dimensions for signature scheme
constant uint VEC_M = 8;                     // Public key rows
constant uint VEC_N = 7;                     // Secret key / signature dimension

// Security parameters
constant int REJECTION_BOUND = 1 << 18;      // Rejection sampling bound
constant float SIGMA = 1.55f;                // Gaussian standard deviation

// NTT parameters (primitive root of unity for Q)
constant ulong OMEGA = 1753UL;               // Primitive 512th root of unity mod Q
constant ulong OMEGA_INV = 731434UL;         // Inverse of OMEGA mod Q
constant ulong N_INV = 8347649UL;            // Inverse of N=256 mod Q

// ============================================================================
// Data Types
// ============================================================================

// Single polynomial coefficient (fits in 32 bits for Q < 2^24)
typedef uint Coeff;

// Polynomial in ring Z_Q[X]/(X^N + 1)
struct Poly {
    Coeff coeffs[RING_N];
};

// Vector of M polynomials
struct PolyVecM {
    Poly polys[VEC_M];
};

// Vector of N polynomials
struct PolyVecN {
    Poly polys[VEC_N];
};

// Matrix M x N of polynomials
struct PolyMatrix {
    Poly polys[VEC_M * VEC_N];
};

// Threshold signature share
struct SignatureShare {
    uint participant_id;
    Poly c;                 // Challenge polynomial
    PolyVecN z;            // Response vector
    PolyVecM Delta;        // Rounding component
};

// Ringtail parameters for kernel dispatch
struct RingtailParams {
    uint num_participants;
    uint threshold;
    uint batch_size;
    uint ntt_stage;         // For staged NTT
};

// ============================================================================
// Modular Arithmetic
// ============================================================================

// Montgomery reduction: returns (a * R^-1) mod Q where R = 2^32
inline Coeff mont_reduce(ulong a) {
    ulong t = (a * RING_Q_INV) & 0xFFFFFFFFUL;
    ulong u = a + t * RING_Q;
    Coeff result = (Coeff)(u >> 32);
    return (result >= RING_Q) ? result - RING_Q : result;
}

// Modular addition
inline Coeff mod_add(Coeff a, Coeff b) {
    Coeff sum = a + b;
    return (sum >= RING_Q) ? sum - (Coeff)RING_Q : sum;
}

// Modular subtraction
inline Coeff mod_sub(Coeff a, Coeff b) {
    return (a >= b) ? a - b : a + (Coeff)RING_Q - b;
}

// Modular multiplication with Montgomery
inline Coeff mod_mul(Coeff a, Coeff b) {
    return mont_reduce((ulong)a * (ulong)b);
}

// Modular negation
inline Coeff mod_neg(Coeff a) {
    return (a == 0) ? 0 : (Coeff)RING_Q - a;
}

// Center reduction: map [0, Q) to [-(Q-1)/2, (Q-1)/2]
inline int center_reduce(Coeff a) {
    int t = (int)a;
    int half_q = (int)(RING_Q >> 1);
    return (t > half_q) ? t - (int)RING_Q : t;
}

// ============================================================================
// NTT Operations
// ============================================================================

// Precomputed twiddle factors would normally be in a buffer
// For simplicity, compute on-the-fly (production code uses lookup table)

inline Coeff power_of_omega(uint k) {
    // Compute OMEGA^k mod Q using binary exponentiation
    ulong result = 1;
    ulong base = OMEGA;
    while (k > 0) {
        if (k & 1) {
            result = (result * base) % RING_Q;
        }
        base = (base * base) % RING_Q;
        k >>= 1;
    }
    return (Coeff)result;
}

// Forward NTT (Cooley-Tukey, bit-reversal input)
inline void ntt_forward_inplace(thread Poly* p, device const Coeff* twiddles) {
    uint n = RING_N;
    
    for (uint len = 1; len < n; len <<= 1) {
        for (uint i = 0; i < n; i += 2 * len) {
            for (uint j = 0; j < len; j++) {
                Coeff w = twiddles[len + j];
                Coeff u = p->coeffs[i + j];
                Coeff v = mod_mul(p->coeffs[i + j + len], w);
                p->coeffs[i + j] = mod_add(u, v);
                p->coeffs[i + j + len] = mod_sub(u, v);
            }
        }
    }
}

// Inverse NTT (Gentleman-Sande, bit-reversal output)
inline void ntt_inverse_inplace(thread Poly* p, device const Coeff* inv_twiddles) {
    uint n = RING_N;
    
    for (uint len = n >> 1; len > 0; len >>= 1) {
        for (uint i = 0; i < n; i += 2 * len) {
            for (uint j = 0; j < len; j++) {
                Coeff w = inv_twiddles[len + j];
                Coeff u = p->coeffs[i + j];
                Coeff v = p->coeffs[i + j + len];
                p->coeffs[i + j] = mod_add(u, v);
                p->coeffs[i + j + len] = mod_mul(mod_sub(u, v), w);
            }
        }
    }
    
    // Scale by N^-1
    Coeff n_inv = (Coeff)N_INV;
    for (uint i = 0; i < n; i++) {
        p->coeffs[i] = mod_mul(p->coeffs[i], n_inv);
    }
}

// Pointwise multiplication in NTT domain
inline Poly poly_mul_ntt(Poly a, Poly b) {
    Poly c;
    for (uint i = 0; i < RING_N; i++) {
        c.coeffs[i] = mod_mul(a.coeffs[i], b.coeffs[i]);
    }
    return c;
}

// ============================================================================
// Polynomial Arithmetic
// ============================================================================

inline Poly poly_zero() {
    Poly p;
    for (uint i = 0; i < RING_N; i++) {
        p.coeffs[i] = 0;
    }
    return p;
}

inline Poly poly_add(Poly a, Poly b) {
    Poly c;
    for (uint i = 0; i < RING_N; i++) {
        c.coeffs[i] = mod_add(a.coeffs[i], b.coeffs[i]);
    }
    return c;
}

inline Poly poly_sub(Poly a, Poly b) {
    Poly c;
    for (uint i = 0; i < RING_N; i++) {
        c.coeffs[i] = mod_sub(a.coeffs[i], b.coeffs[i]);
    }
    return c;
}

inline Poly poly_neg(Poly a) {
    Poly c;
    for (uint i = 0; i < RING_N; i++) {
        c.coeffs[i] = mod_neg(a.coeffs[i]);
    }
    return c;
}

// Scalar multiplication
inline Poly poly_scalar_mul(Poly a, Coeff s) {
    Poly c;
    for (uint i = 0; i < RING_N; i++) {
        c.coeffs[i] = mod_mul(a.coeffs[i], s);
    }
    return c;
}

// ============================================================================
// Vector and Matrix Operations
// ============================================================================

inline PolyVecN vec_n_add(PolyVecN a, PolyVecN b) {
    PolyVecN c;
    for (uint i = 0; i < VEC_N; i++) {
        c.polys[i] = poly_add(a.polys[i], b.polys[i]);
    }
    return c;
}

inline PolyVecM vec_m_add(PolyVecM a, PolyVecM b) {
    PolyVecM c;
    for (uint i = 0; i < VEC_M; i++) {
        c.polys[i] = poly_add(a.polys[i], b.polys[i]);
    }
    return c;
}

// Matrix-vector multiplication: A (M x N) * v (N) = result (M)
// Assumes inputs are in NTT domain
inline PolyVecM matrix_vec_mul_ntt(device const PolyMatrix* A, PolyVecN v) {
    PolyVecM result;
    
    for (uint i = 0; i < VEC_M; i++) {
        result.polys[i] = poly_zero();
        for (uint j = 0; j < VEC_N; j++) {
            Poly product = poly_mul_ntt(A->polys[i * VEC_N + j], v.polys[j]);
            result.polys[i] = poly_add(result.polys[i], product);
        }
    }
    
    return result;
}

// ============================================================================
// Gaussian Sampling (Rejection Sampling with CDT)
// ============================================================================

// Simple box-muller approximation for discrete Gaussian
// Production code would use CDT or Knuth-Yao
inline int sample_gaussian(thread uint* rng_state, float sigma) {
    // LCG for random bits
    *rng_state = (*rng_state) * 1103515245u + 12345u;
    uint u1 = *rng_state;
    *rng_state = (*rng_state) * 1103515245u + 12345u;
    uint u2 = *rng_state;
    
    // Convert to uniform [0,1)
    float f1 = (float)(u1 >> 8) / 16777216.0f;
    float f2 = (float)(u2 >> 8) / 16777216.0f;
    
    // Box-Muller transform
    float r = sigma * sqrt(-2.0f * log(f1 + 0.000001f));
    float theta = 2.0f * 3.14159265f * f2;
    
    return (int)round(r * cos(theta));
}

// Sample a polynomial with coefficients from discrete Gaussian
inline Poly sample_poly_gaussian(thread uint* rng_state, float sigma) {
    Poly p;
    for (uint i = 0; i < RING_N; i++) {
        int sample = sample_gaussian(rng_state, sigma);
        // Reduce to [0, Q)
        p.coeffs[i] = (sample >= 0) ? (Coeff)sample : (Coeff)(sample + (int)RING_Q);
    }
    return p;
}

// ============================================================================
// Rejection Sampling for Signature
// ============================================================================

// Check if z vector is within rejection bound
inline bool check_rejection_bound(PolyVecN z, int bound) {
    for (uint i = 0; i < VEC_N; i++) {
        for (uint j = 0; j < RING_N; j++) {
            int coeff = center_reduce(z.polys[i].coeffs[j]);
            if (coeff > bound || coeff < -bound) {
                return false;
            }
        }
    }
    return true;
}

// Compute infinity norm of polynomial
inline int poly_norm_inf(Poly p) {
    int max_val = 0;
    for (uint i = 0; i < RING_N; i++) {
        int coeff = center_reduce(p.coeffs[i]);
        int abs_coeff = (coeff >= 0) ? coeff : -coeff;
        if (abs_coeff > max_val) max_val = abs_coeff;
    }
    return max_val;
}

// ============================================================================
// Ringtail Signing Kernels
// ============================================================================

// Kernel 1: Generate signature shares (per participant)
kernel void ringtail_generate_share(
    device const Poly* secret_shares [[buffer(0)]],       // s_i (participant's share)
    device const PolyMatrix* public_key_A [[buffer(1)]],  // Public matrix A
    device const PolyVecM* commitment_y [[buffer(2)]],    // Commitment y = A*r
    device const Poly* challenge [[buffer(3)]],           // Challenge c
    device const PolyVecN* randomness [[buffer(4)]],      // Randomness r_i
    device const Coeff* ntt_twiddles [[buffer(5)]],       // NTT twiddles
    device SignatureShare* shares [[buffer(6)]],          // Output shares
    constant RingtailParams& params [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.num_participants) return;
    
    // Load data
    Poly s_i = secret_shares[gid];
    Poly c = challenge[0];
    PolyVecN r_i = randomness[gid];
    
    // Convert to NTT domain
    thread Poly c_ntt = c;
    ntt_forward_inplace(&c_ntt, ntt_twiddles);
    
    thread Poly s_i_ntt = s_i;
    ntt_forward_inplace(&s_i_ntt, ntt_twiddles);
    
    // Compute z_i = r_i + c * s_i for each component
    PolyVecN z_i;
    for (uint j = 0; j < VEC_N; j++) {
        thread Poly r_ij_ntt = r_i.polys[j];
        ntt_forward_inplace(&r_ij_ntt, ntt_twiddles);
        
        // c * s_i in NTT domain
        Poly cs_ntt = poly_mul_ntt(c_ntt, s_i_ntt);
        
        // z_ij = r_ij + c * s_i (NTT domain)
        Poly z_ij_ntt = poly_add(r_ij_ntt, cs_ntt);
        
        // Inverse NTT
        thread Poly z_ij = z_ij_ntt;
        // ntt_inverse_inplace would be called here
        
        z_i.polys[j] = z_ij;
    }
    
    // Compute Delta for rounding (A * z_i - commitment_y)
    PolyVecN z_i_ntt;
    for (uint j = 0; j < VEC_N; j++) {
        z_i_ntt.polys[j] = z_i.polys[j];
        ntt_forward_inplace(&z_i_ntt.polys[j], ntt_twiddles);
    }
    
    PolyVecM Az = matrix_vec_mul_ntt(public_key_A, z_i_ntt);
    PolyVecM Delta;
    for (uint j = 0; j < VEC_M; j++) {
        Delta.polys[j] = poly_sub(Az.polys[j], commitment_y->polys[j]);
    }
    
    // Store signature share
    SignatureShare share;
    share.participant_id = gid + 1;  // 1-indexed
    share.c = c;
    share.z = z_i;
    share.Delta = Delta;
    
    shares[gid] = share;
}

// Kernel 2: Aggregate signature shares
kernel void ringtail_aggregate_shares(
    device const SignatureShare* shares [[buffer(0)]],
    device const uint* participant_ids [[buffer(1)]],
    device const Coeff* lagrange_coeffs [[buffer(2)]],     // Precomputed
    device Poly* aggregated_c [[buffer(3)]],
    device PolyVecN* aggregated_z [[buffer(4)]],
    device PolyVecM* aggregated_delta [[buffer(5)]],
    constant RingtailParams& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;  // Single-threaded aggregation
    
    // Initialize accumulators
    Poly c = shares[0].c;  // Challenge is the same for all
    PolyVecN z;
    PolyVecM Delta;
    
    for (uint i = 0; i < VEC_N; i++) z.polys[i] = poly_zero();
    for (uint i = 0; i < VEC_M; i++) Delta.polys[i] = poly_zero();
    
    // Aggregate: z = sum(lambda_i * z_i), Delta = sum(lambda_i * Delta_i)
    for (uint p = 0; p < params.num_participants; p++) {
        Coeff lambda = lagrange_coeffs[p];
        SignatureShare share = shares[p];
        
        // z += lambda * z_i
        for (uint i = 0; i < VEC_N; i++) {
            Poly scaled = poly_scalar_mul(share.z.polys[i], lambda);
            z.polys[i] = poly_add(z.polys[i], scaled);
        }
        
        // Delta += lambda * Delta_i
        for (uint i = 0; i < VEC_M; i++) {
            Poly scaled = poly_scalar_mul(share.Delta.polys[i], lambda);
            Delta.polys[i] = poly_add(Delta.polys[i], scaled);
        }
    }
    
    aggregated_c[0] = c;
    *aggregated_z = z;
    *aggregated_delta = Delta;
}

// Kernel 3: Batch NTT forward (one polynomial per thread)
kernel void ringtail_batch_ntt_forward(
    device Poly* polys [[buffer(0)]],
    device const Coeff* twiddles [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    thread Poly p = polys[gid];
    ntt_forward_inplace(&p, twiddles);
    polys[gid] = p;
}

// Kernel 4: Batch NTT inverse (one polynomial per thread)
kernel void ringtail_batch_ntt_inverse(
    device Poly* polys [[buffer(0)]],
    device const Coeff* inv_twiddles [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    thread Poly p = polys[gid];
    ntt_inverse_inplace(&p, inv_twiddles);
    polys[gid] = p;
}

// Kernel 5: Gaussian sampling for masking randomness
kernel void ringtail_sample_gaussian_vec(
    device PolyVecN* output [[buffer(0)]],
    device const uint* seeds [[buffer(1)]],
    constant RingtailParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.num_participants) return;
    
    thread uint rng = seeds[gid];
    PolyVecN result;
    
    for (uint i = 0; i < VEC_N; i++) {
        result.polys[i] = sample_poly_gaussian(&rng, SIGMA);
    }
    
    output[gid] = result;
}

// Kernel 6: Rejection sampling check (parallel per signature)
kernel void ringtail_check_rejection(
    device const PolyVecN* z_vectors [[buffer(0)]],
    device uint* valid [[buffer(1)]],
    constant RingtailParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.batch_size) return;
    
    PolyVecN z = z_vectors[gid];
    bool passes = check_rejection_bound(z, REJECTION_BOUND);
    valid[gid] = passes ? 1 : 0;
}

// Kernel 7: Polynomial norm computation (for security checks)
kernel void ringtail_compute_norms(
    device const Poly* polys [[buffer(0)]],
    device int* norms [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    Poly p = polys[gid];
    norms[gid] = poly_norm_inf(p);
}

// Kernel 8: Matrix-vector product (parallel per output row)
kernel void ringtail_matrix_vec_mul(
    device const PolyMatrix* A [[buffer(0)]],
    device const PolyVecN* v [[buffer(1)]],
    device PolyVecM* result [[buffer(2)]],
    constant uint& batch_idx [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= VEC_M) return;
    
    PolyVecN vec = v[batch_idx];
    Poly sum = poly_zero();
    
    for (uint j = 0; j < VEC_N; j++) {
        Poly product = poly_mul_ntt(A->polys[gid * VEC_N + j], vec.polys[j]);
        sum = poly_add(sum, product);
    }
    
    result[batch_idx].polys[gid] = sum;
}

// Kernel 9: Combine partial signatures with Lagrange interpolation
kernel void ringtail_lagrange_combine(
    device const SignatureShare* shares [[buffer(0)]],
    device const Coeff* lagrange_at_zero [[buffer(1)]],  // lambda_i(0)
    device Poly* combined_secret [[buffer(2)]],
    constant RingtailParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= RING_N) return;  // One thread per coefficient
    
    Coeff sum = 0;
    
    for (uint p = 0; p < params.num_participants; p++) {
        Coeff lambda = lagrange_at_zero[p];
        // Get p-th participant's first z polynomial coefficient at position gid
        Coeff z_coeff = shares[p].z.polys[0].coeffs[gid];
        sum = mod_add(sum, mod_mul(lambda, z_coeff));
    }
    
    combined_secret->coeffs[gid] = sum;
}
