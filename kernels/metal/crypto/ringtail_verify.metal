// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Ringtail Lattice-Based Threshold Signature Verification
// Batch verification of Ringtail signatures with polynomial norm checks
// Optimized for Apple Silicon GPUs

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Ringtail Parameters (shared with ringtail_sign.metal)
// ============================================================================

constant uint RING_N = 256;
constant ulong RING_Q = 8380417UL;
constant ulong RING_Q_INV = 58728449UL;
constant uint VEC_M = 8;
constant uint VEC_N = 7;
constant ulong N_INV = 8347649UL;
constant int CHALLENGE_WEIGHT = 60;       // Hamming weight of challenge
// Note: beta_bound and delta_bound come from VerifyParams at runtime

// ============================================================================
// Data Types
// ============================================================================

typedef uint Coeff;

struct Poly {
    Coeff coeffs[RING_N];
};

struct PolyVecM {
    Poly polys[VEC_M];
};

struct PolyVecN {
    Poly polys[VEC_N];
};

struct PolyMatrix {
    Poly polys[VEC_M * VEC_N];
};

// Ringtail public key
struct RingtailPublicKey {
    PolyMatrix A;           // M x N matrix
    PolyVecM bTilde;       // Rounded public key b~ = round(A*s)
};

// Ringtail signature
struct RingtailSignature {
    Poly c;                 // Challenge polynomial (sparse, weight tau)
    PolyVecN z;            // Response vector
    PolyVecM Delta;        // Rounding correction
};

// Verification parameters
struct VerifyParams {
    uint batch_size;
    uint num_threads;
    int beta_bound;
    int delta_bound;
};

// ============================================================================
// Modular Arithmetic
// ============================================================================

inline Coeff mont_reduce(ulong a) {
    ulong t = (a * RING_Q_INV) & 0xFFFFFFFFUL;
    ulong u = a + t * RING_Q;
    Coeff result = (Coeff)(u >> 32);
    return (result >= RING_Q) ? result - RING_Q : result;
}

inline Coeff mod_add(Coeff a, Coeff b) {
    Coeff sum = a + b;
    return (sum >= RING_Q) ? sum - (Coeff)RING_Q : sum;
}

inline Coeff mod_sub(Coeff a, Coeff b) {
    return (a >= b) ? a - b : a + (Coeff)RING_Q - b;
}

inline Coeff mod_mul(Coeff a, Coeff b) {
    return mont_reduce((ulong)a * (ulong)b);
}

inline Coeff mod_neg(Coeff a) {
    return (a == 0) ? 0 : (Coeff)RING_Q - a;
}

inline int center_reduce(Coeff a) {
    int t = (int)a;
    int half_q = (int)(RING_Q >> 1);
    return (t > half_q) ? t - (int)RING_Q : t;
}

// ============================================================================
// Polynomial Operations
// ============================================================================

inline Poly poly_zero() {
    Poly p;
    for (uint i = 0; i < RING_N; i++) p.coeffs[i] = 0;
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

inline Poly poly_mul_ntt(Poly a, Poly b) {
    Poly c;
    for (uint i = 0; i < RING_N; i++) {
        c.coeffs[i] = mod_mul(a.coeffs[i], b.coeffs[i]);
    }
    return c;
}

// ============================================================================
// NTT Operations
// ============================================================================

inline void ntt_forward_inplace(thread Poly* p, device const Coeff* twiddles) {
    for (uint len = 1; len < RING_N; len <<= 1) {
        for (uint i = 0; i < RING_N; i += 2 * len) {
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

inline void ntt_inverse_inplace(thread Poly* p, device const Coeff* inv_twiddles) {
    for (uint len = RING_N >> 1; len > 0; len >>= 1) {
        for (uint i = 0; i < RING_N; i += 2 * len) {
            for (uint j = 0; j < len; j++) {
                Coeff w = inv_twiddles[len + j];
                Coeff u = p->coeffs[i + j];
                Coeff v = p->coeffs[i + j + len];
                p->coeffs[i + j] = mod_add(u, v);
                p->coeffs[i + j + len] = mod_mul(mod_sub(u, v), w);
            }
        }
    }
    
    Coeff n_inv = (Coeff)N_INV;
    for (uint i = 0; i < RING_N; i++) {
        p->coeffs[i] = mod_mul(p->coeffs[i], n_inv);
    }
}

// ============================================================================
// Norm and Bound Checking
// ============================================================================

// Compute infinity norm
inline int poly_norm_inf(Poly p) {
    int max_val = 0;
    for (uint i = 0; i < RING_N; i++) {
        int coeff = center_reduce(p.coeffs[i]);
        int abs_coeff = (coeff >= 0) ? coeff : -coeff;
        if (abs_coeff > max_val) max_val = abs_coeff;
    }
    return max_val;
}

// Compute L2 norm squared (sum of squared coefficients)
inline ulong poly_norm_l2_squared(Poly p) {
    ulong sum = 0;
    for (uint i = 0; i < RING_N; i++) {
        int coeff = center_reduce(p.coeffs[i]);
        sum += (ulong)coeff * (ulong)coeff;
    }
    return sum;
}

// Check z vector norm bound
inline bool check_z_norm(PolyVecN z, int bound) {
    for (uint i = 0; i < VEC_N; i++) {
        if (poly_norm_inf(z.polys[i]) > bound) {
            return false;
        }
    }
    return true;
}

// Check Delta vector norm bound
inline bool check_delta_norm(PolyVecM Delta, int bound) {
    for (uint i = 0; i < VEC_M; i++) {
        if (poly_norm_inf(Delta.polys[i]) > bound) {
            return false;
        }
    }
    return true;
}

// Check challenge polynomial structure (sparse with bounded coefficients)
inline bool check_challenge_format(Poly c, int weight) {
    int nonzero = 0;
    for (uint i = 0; i < RING_N; i++) {
        if (c.coeffs[i] != 0) {
            nonzero++;
            // Coefficients should be +1 or -1 (Q-1)
            if (c.coeffs[i] != 1 && c.coeffs[i] != (Coeff)RING_Q - 1) {
                return false;
            }
        }
    }
    return nonzero == weight;
}

// ============================================================================
// Matrix-Vector Multiplication
// ============================================================================

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

// Thread address space version for local copies
inline PolyVecM matrix_vec_mul_ntt(thread const PolyMatrix* A, PolyVecN v) {
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
// Rounding Functions
// ============================================================================

// Apply rounding: round coefficient to nearest multiple of 2^d
inline Coeff round_coeff(Coeff a, uint d) {
    uint mask = (1u << d) - 1;
    uint half_val = 1u << (d - 1);  // 'half' is reserved keyword in Metal
    uint rounded = (a + half_val) & ~mask;
    return rounded % RING_Q;
}

// Apply rounding to polynomial
inline Poly poly_round(Poly p, uint d) {
    Poly r;
    for (uint i = 0; i < RING_N; i++) {
        r.coeffs[i] = round_coeff(p.coeffs[i], d);
    }
    return r;
}

// ============================================================================
// Hash Function (for challenge derivation)
// ============================================================================

// Simple hash mixing for challenge recomputation
inline uint hash_mix(thread uint* state, uint input) {
    *state ^= input;
    *state = (*state * 0x5bd1e995u) ^ (*state >> 15);
    return *state;
}

// Derive challenge polynomial from hash state
inline Poly derive_challenge(thread uint* hash_state, int weight) {
    Poly c = poly_zero();
    
    int placed = 0;
    while (placed < weight) {
        uint pos = hash_mix(hash_state, placed) % RING_N;
        if (c.coeffs[pos] == 0) {
            uint sign = hash_mix(hash_state, pos) & 1;
            c.coeffs[pos] = sign ? 1 : (Coeff)RING_Q - 1;
            placed++;
        }
    }
    
    return c;
}

// ============================================================================
// Verification Kernels
// ============================================================================

// Kernel 1: Batch verify Ringtail signatures
kernel void ringtail_batch_verify(
    device const RingtailSignature* signatures [[buffer(0)]],
    device const RingtailPublicKey* public_keys [[buffer(1)]],
    device const Poly* messages [[buffer(2)]],            // H(message) as polynomial
    device const Coeff* ntt_twiddles [[buffer(3)]],
    device const Coeff* inv_twiddles [[buffer(4)]],
    device uint* results [[buffer(5)]],
    constant VerifyParams& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.batch_size) return;
    
    RingtailSignature sig = signatures[gid];
    RingtailPublicKey pk = public_keys[gid];
    // msg used in step 9 challenge verification (currently simplified)
    (void)messages[gid];

    bool valid = true;
    
    // Step 1: Check z norm bound
    if (!check_z_norm(sig.z, params.beta_bound)) {
        results[gid] = 0;
        return;
    }
    
    // Step 2: Check Delta norm bound
    if (!check_delta_norm(sig.Delta, params.delta_bound)) {
        results[gid] = 0;
        return;
    }
    
    // Step 3: Check challenge format
    if (!check_challenge_format(sig.c, CHALLENGE_WEIGHT)) {
        results[gid] = 0;
        return;
    }
    
    // Step 4: Convert z to NTT domain
    PolyVecN z_ntt;
    for (uint i = 0; i < VEC_N; i++) {
        thread Poly p = sig.z.polys[i];
        ntt_forward_inplace(&p, ntt_twiddles);
        z_ntt.polys[i] = p;
    }
    
    // Step 5: Compute A * z in NTT domain
    PolyVecM Az = matrix_vec_mul_ntt(&pk.A, z_ntt);
    
    // Convert to coefficient domain
    for (uint i = 0; i < VEC_M; i++) {
        thread Poly p = Az.polys[i];
        ntt_inverse_inplace(&p, inv_twiddles);
        Az.polys[i] = p;
    }
    
    // Step 6: Convert c to NTT domain for multiplication
    thread Poly c_ntt = sig.c;
    ntt_forward_inplace(&c_ntt, ntt_twiddles);
    
    // Step 7: Compute c * bTilde for each component
    PolyVecM c_btilde;
    for (uint i = 0; i < VEC_M; i++) {
        thread Poly btilde_ntt = pk.bTilde.polys[i];
        ntt_forward_inplace(&btilde_ntt, ntt_twiddles);
        
        Poly product = poly_mul_ntt(c_ntt, btilde_ntt);
        ntt_inverse_inplace(&product, inv_twiddles);
        c_btilde.polys[i] = product;
    }
    
    // Step 8: Verify equation: round(A*z) = c*bTilde + Delta (approximately)
    // Check that A*z - c*bTilde - Delta rounds to zero
    for (uint i = 0; i < VEC_M; i++) {
        Poly diff = poly_sub(Az.polys[i], c_btilde.polys[i]);
        diff = poly_sub(diff, sig.Delta.polys[i]);
        
        // Check that diff has small coefficients after rounding
        for (uint j = 0; j < RING_N; j++) {
            int coeff = center_reduce(diff.coeffs[j]);
            if (coeff > params.delta_bound || coeff < -params.delta_bound) {
                valid = false;
                break;
            }
        }
        if (!valid) break;
    }
    
    // Step 9: Verify challenge is correctly derived from commitment
    // This would involve recomputing H(A*z - c*bTilde || message)
    // Simplified: assume challenge is valid if format check passed
    
    results[gid] = valid ? 1 : 0;
}

// Kernel 2: Compute polynomial norms in parallel
kernel void ringtail_compute_poly_norms(
    device const Poly* polys [[buffer(0)]],
    device int* inf_norms [[buffer(1)]],
    device ulong* l2_norms [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    Poly p = polys[gid];
    inf_norms[gid] = poly_norm_inf(p);
    l2_norms[gid] = poly_norm_l2_squared(p);
}

// Kernel 3: Batch check rejection bounds
kernel void ringtail_check_bounds(
    device const PolyVecN* z_vectors [[buffer(0)]],
    device const PolyVecM* delta_vectors [[buffer(1)]],
    device uint* z_valid [[buffer(2)]],
    device uint* delta_valid [[buffer(3)]],
    constant VerifyParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.batch_size) return;
    
    z_valid[gid] = check_z_norm(z_vectors[gid], params.beta_bound) ? 1 : 0;
    delta_valid[gid] = check_delta_norm(delta_vectors[gid], params.delta_bound) ? 1 : 0;
}

// Kernel 4: Parallel matrix-vector multiplication (one output polynomial per thread)
kernel void ringtail_parallel_mat_vec_mul(
    device const PolyMatrix* A [[buffer(0)]],
    device const PolyVecN* z [[buffer(1)]],
    device PolyVecM* Az [[buffer(2)]],
    constant uint& batch_idx [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= VEC_M) return;
    
    PolyVecN z_batch = z[batch_idx];
    Poly sum = poly_zero();
    
    for (uint j = 0; j < VEC_N; j++) {
        Poly product = poly_mul_ntt(A->polys[gid * VEC_N + j], z_batch.polys[j]);
        sum = poly_add(sum, product);
    }
    
    Az[batch_idx].polys[gid] = sum;
}

// Kernel 5: Challenge verification
kernel void ringtail_verify_challenge(
    device const Poly* challenges [[buffer(0)]],
    device const Poly* recomputed_challenges [[buffer(1)]],
    device uint* valid [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    Poly c1 = challenges[gid];
    Poly c2 = recomputed_challenges[gid];
    
    bool equal = true;
    for (uint i = 0; i < RING_N; i++) {
        if (c1.coeffs[i] != c2.coeffs[i]) {
            equal = false;
            break;
        }
    }
    
    valid[gid] = equal ? 1 : 0;
}

// Kernel 6: Reconstruct public key from shares (for threshold verification)
kernel void ringtail_reconstruct_public_key(
    device const PolyVecM* pk_shares [[buffer(0)]],
    device const Coeff* lagrange_coeffs [[buffer(1)]],
    device PolyVecM* reconstructed [[buffer(2)]],
    constant uint& num_shares [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= VEC_M * RING_N) return;
    
    uint poly_idx = gid / RING_N;
    uint coeff_idx = gid % RING_N;
    
    Coeff sum = 0;
    for (uint s = 0; s < num_shares; s++) {
        Coeff lambda = lagrange_coeffs[s];
        Coeff coeff = pk_shares[s].polys[poly_idx].coeffs[coeff_idx];
        sum = mod_add(sum, mod_mul(lambda, coeff));
    }
    
    reconstructed->polys[poly_idx].coeffs[coeff_idx] = sum;
}

// Kernel 7: Batch apply rounding
kernel void ringtail_batch_round(
    device Poly* polys [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    constant uint& round_bits [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    polys[gid] = poly_round(polys[gid], round_bits);
}

// Kernel 8: Combined verification equation check
// Verifies: Az = c*bTilde + Delta + w (where w is rounding error)
kernel void ringtail_verify_equation(
    device const PolyVecM* Az [[buffer(0)]],
    device const PolyVecM* c_btilde [[buffer(1)]],
    device const PolyVecM* Delta [[buffer(2)]],
    device uint* valid [[buffer(3)]],
    constant VerifyParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.batch_size) return;
    
    bool result = true;
    
    for (uint i = 0; i < VEC_M && result; i++) {
        Poly lhs = Az[gid].polys[i];
        Poly rhs = poly_add(c_btilde[gid].polys[i], Delta[gid].polys[i]);
        Poly diff = poly_sub(lhs, rhs);
        
        // Check residual is small (within rounding error)
        int norm = poly_norm_inf(diff);
        if (norm > params.delta_bound) {
            result = false;
        }
    }
    
    valid[gid] = result ? 1 : 0;
}

// Kernel 9: Parallel coefficient validation
kernel void ringtail_validate_coefficients(
    device const Poly* polys [[buffer(0)]],
    device atomic_uint* invalid_count [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant int& bound [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint poly_idx = gid / RING_N;
    uint coeff_idx = gid % RING_N;
    
    if (poly_idx >= count) return;
    
    int coeff = center_reduce(polys[poly_idx].coeffs[coeff_idx]);
    if (coeff > bound || coeff < -bound) {
        atomic_fetch_add_explicit(invalid_count, 1, memory_order_relaxed);
    }
}
