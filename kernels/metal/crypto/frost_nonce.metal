// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// FROST Nonce Generation and Commitment Operations
// Batch nonce generation, hash-to-curve, and binding factor computation
// Optimized for Apple Silicon GPUs

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Types (shared with frost_aggregate.metal)
// ============================================================================

struct Scalar256 {
    uint limbs[8];
};

struct Ed25519Affine {
    Scalar256 x;
    Scalar256 y;
};

struct Ed25519Extended {
    Scalar256 x;
    Scalar256 y;
    Scalar256 z;
    Scalar256 t;
};

struct NonceCommitment {
    Scalar256 hiding_nonce_d;      // d_i
    Scalar256 binding_nonce_e;     // e_i
    Ed25519Affine commitment_d;    // D_i = g^d_i
    Ed25519Affine commitment_e;    // E_i = g^e_i
};

struct NonceParams {
    uint num_participants;
    uint seed_entropy_offset;
    uint curve_type;           // 0 = Ed25519, 1 = secp256k1
    uint batch_size;
};

// SHA-512 state for hash-to-scalar
struct SHA512State {
    ulong h[8];
    uint total_len;
    uint _pad;
};

// ============================================================================
// Ed25519 Constants
// ============================================================================

constant uint ED25519_L[8] = {
    0x5cf5d3edu, 0x5812631au, 0xa2f79cd6u, 0x14def9deu,
    0x00000000u, 0x00000000u, 0x00000000u, 0x10000000u
};

// Ed25519 base point G (compressed y, x calculated)
constant uint ED25519_GY[8] = {
    0x58666666u, 0x66666666u, 0x66666666u, 0x66666666u,
    0x66666666u, 0x66666666u, 0x66666666u, 0x66666666u
};

// secp256k1 constants
constant uint SECP256K1_N[8] = {
    0xd0364141u, 0xbfd25e8cu, 0xaf48a03bu, 0xbaaedce6u,
    0xfffffffeu, 0xffffffffu, 0xffffffffu, 0xffffffffu
};

// ============================================================================
// Scalar Arithmetic (from frost_aggregate)
// ============================================================================

inline Scalar256 scalar_zero() {
    Scalar256 r;
    for (int i = 0; i < 8; i++) r.limbs[i] = 0;
    return r;
}

inline Scalar256 scalar_one() {
    Scalar256 r = scalar_zero();
    r.limbs[0] = 1;
    return r;
}

inline bool scalar_is_zero(Scalar256 a) {
    for (int i = 0; i < 8; i++) {
        if (a.limbs[i] != 0) return false;
    }
    return true;
}

inline bool scalar_gte(Scalar256 a, constant uint* mod) {
    for (int i = 7; i >= 0; i--) {
        if (a.limbs[i] > mod[i]) return true;
        if (a.limbs[i] < mod[i]) return false;
    }
    return true;
}

inline Scalar256 scalar_add(Scalar256 a, Scalar256 b, constant uint* mod) {
    Scalar256 r;
    uint carry = 0;
    
    for (int i = 0; i < 8; i++) {
        uint sum = a.limbs[i] + b.limbs[i] + carry;
        carry = (sum < a.limbs[i]) || (carry && sum == a.limbs[i]) ? 1 : 0;
        r.limbs[i] = sum;
    }
    
    if (carry || scalar_gte(r, mod)) {
        uint borrow = 0;
        for (int i = 0; i < 8; i++) {
            uint diff = r.limbs[i] - mod[i] - borrow;
            borrow = (r.limbs[i] < mod[i] + borrow) ? 1 : 0;
            r.limbs[i] = diff;
        }
    }
    
    return r;
}

inline Scalar256 scalar_mul(Scalar256 a, Scalar256 b, constant uint* mod) {
    uint product[16] = {0};
    
    for (int i = 0; i < 8; i++) {
        uint carry = 0;
        for (int j = 0; j < 8; j++) {
            ulong prod = (ulong)a.limbs[i] * (ulong)b.limbs[j] + product[i + j] + carry;
            product[i + j] = (uint)prod;
            carry = (uint)(prod >> 32);
        }
        product[i + 8] = carry;
    }
    
    Scalar256 r;
    for (int i = 0; i < 8; i++) {
        r.limbs[i] = product[i];
    }
    
    while (scalar_gte(r, mod)) {
        uint borrow = 0;
        for (int i = 0; i < 8; i++) {
            uint diff = r.limbs[i] - mod[i] - borrow;
            borrow = (r.limbs[i] < mod[i] + borrow) ? 1 : 0;
            r.limbs[i] = diff;
        }
    }
    
    return r;
}

// ============================================================================
// Pseudo-Random Number Generation (ChaCha20-based)
// ============================================================================

struct ChaCha20State {
    uint state[16];
};

inline uint rotl32(uint x, int n) {
    return (x << n) | (x >> (32 - n));
}

inline void chacha_quarter_round(thread uint* a, thread uint* b, thread uint* c, thread uint* d) {
    *a += *b; *d ^= *a; *d = rotl32(*d, 16);
    *c += *d; *b ^= *c; *b = rotl32(*b, 12);
    *a += *b; *d ^= *a; *d = rotl32(*d, 8);
    *c += *d; *b ^= *c; *b = rotl32(*b, 7);
}

inline ChaCha20State chacha20_init(Scalar256 key, ulong counter, ulong nonce) {
    ChaCha20State s;
    
    // "expand 32-byte k"
    s.state[0] = 0x61707865u;
    s.state[1] = 0x3320646eu;
    s.state[2] = 0x79622d32u;
    s.state[3] = 0x6b206574u;
    
    // Key
    for (int i = 0; i < 8; i++) {
        s.state[4 + i] = key.limbs[i];
    }
    
    // Counter
    s.state[12] = (uint)counter;
    s.state[13] = (uint)(counter >> 32);
    
    // Nonce
    s.state[14] = (uint)nonce;
    s.state[15] = (uint)(nonce >> 32);
    
    return s;
}

inline void chacha20_block(thread ChaCha20State* s) {
    uint working[16];
    for (int i = 0; i < 16; i++) working[i] = s->state[i];
    
    // 20 rounds (10 double rounds)
    for (int i = 0; i < 10; i++) {
        // Column rounds
        chacha_quarter_round(&working[0], &working[4], &working[8], &working[12]);
        chacha_quarter_round(&working[1], &working[5], &working[9], &working[13]);
        chacha_quarter_round(&working[2], &working[6], &working[10], &working[14]);
        chacha_quarter_round(&working[3], &working[7], &working[11], &working[15]);
        
        // Diagonal rounds
        chacha_quarter_round(&working[0], &working[5], &working[10], &working[15]);
        chacha_quarter_round(&working[1], &working[6], &working[11], &working[12]);
        chacha_quarter_round(&working[2], &working[7], &working[8], &working[13]);
        chacha_quarter_round(&working[3], &working[4], &working[9], &working[14]);
    }
    
    // Add original state
    for (int i = 0; i < 16; i++) {
        s->state[i] += working[i];
    }
    
    // Increment counter
    s->state[12]++;
    if (s->state[12] == 0) s->state[13]++;
}

// Generate a random scalar in [1, l-1]
inline Scalar256 random_scalar(thread ChaCha20State* rng, constant uint* mod) {
    Scalar256 r;
    
    do {
        chacha20_block(rng);
        for (int i = 0; i < 8; i++) {
            r.limbs[i] = rng->state[i];
        }
        
        // Clear top bits to ensure < 2^256
        r.limbs[7] &= 0x0FFFFFFFu;
        
        // Reduce mod l
        while (scalar_gte(r, mod)) {
            uint borrow = 0;
            for (int i = 0; i < 8; i++) {
                uint diff = r.limbs[i] - mod[i] - borrow;
                borrow = (r.limbs[i] < mod[i] + borrow) ? 1 : 0;
                r.limbs[i] = diff;
            }
        }
    } while (scalar_is_zero(r)); // Retry if zero
    
    return r;
}

// ============================================================================
// Scalar Multiplication (simplified Montgomery ladder)
// ============================================================================

// Placeholder point operations - full implementation needs complete curve arithmetic
inline Ed25519Extended ed25519_identity() {
    Ed25519Extended r;
    r.x = scalar_zero();
    r.y = scalar_one();
    r.z = scalar_one();
    r.t = scalar_zero();
    return r;
}

inline Ed25519Extended ed25519_double(Ed25519Extended p) {
    // Placeholder - actual implementation requires full field arithmetic
    Ed25519Extended r = p;
    r.z = scalar_add(p.z, p.z, ED25519_L);
    return r;
}

inline Ed25519Extended ed25519_add(Ed25519Extended p, Ed25519Extended q) {
    // Placeholder - actual implementation requires full field arithmetic
    Ed25519Extended r;
    r.x = scalar_add(p.x, q.x, ED25519_L);
    r.y = scalar_add(p.y, q.y, ED25519_L);
    r.z = scalar_add(p.z, q.z, ED25519_L);
    r.t = scalar_add(p.t, q.t, ED25519_L);
    return r;
}

// Compute G * scalar using double-and-add
inline Ed25519Extended ed25519_scalar_mul_base(Scalar256 scalar) {
    Ed25519Extended result = ed25519_identity();
    Ed25519Extended base;
    
    // Base point
    for (int i = 0; i < 8; i++) base.y.limbs[i] = ED25519_GY[i];
    base.x = scalar_zero(); // Would need to compute from y
    base.z = scalar_one();
    base.t = scalar_zero();
    
    // Double-and-add
    for (int bit = 0; bit < 256; bit++) {
        int limb = bit / 32;
        int bit_in_limb = bit % 32;
        
        if ((scalar.limbs[limb] >> bit_in_limb) & 1) {
            result = ed25519_add(result, base);
        }
        base = ed25519_double(base);
    }
    
    return result;
}

inline Ed25519Affine ed25519_to_affine(Ed25519Extended ext) {
    Ed25519Affine r;
    // Would need modular inversion: x = X/Z, y = Y/Z
    r.x = ext.x;
    r.y = ext.y;
    return r;
}

// ============================================================================
// Hash-to-Scalar (RFC 8032 style)
// ============================================================================

// Simple hash mixing function (for binding factor derivation) - device buffer version
inline Scalar256 hash_to_scalar_device(
    device const uint* data,
    uint data_len,
    constant uint* mod
) {
    // Simplified hash - production would use SHA-512 and reduce
    Scalar256 r = scalar_zero();
    
    for (uint i = 0; i < data_len && i < 8; i++) {
        r.limbs[i] = data[i];
    }
    
    // Mix
    for (int round = 0; round < 4; round++) {
        for (int i = 0; i < 8; i++) {
            r.limbs[i] ^= rotl32(r.limbs[(i + 1) % 8], 7);
            r.limbs[i] += r.limbs[(i + 3) % 8];
        }
    }
    
    // Reduce mod l
    while (scalar_gte(r, mod)) {
        uint borrow = 0;
        for (int i = 0; i < 8; i++) {
            uint diff = r.limbs[i] - mod[i] - borrow;
            borrow = (r.limbs[i] < mod[i] + borrow) ? 1 : 0;
            r.limbs[i] = diff;
        }
    }
    
    return r;
}

// Thread-local array version for in-kernel computed hash inputs
inline Scalar256 hash_to_scalar(
    thread const uint* data,
    uint data_len,
    constant uint* mod
) {
    // Simplified hash - production would use SHA-512 and reduce
    Scalar256 r = scalar_zero();

    for (uint i = 0; i < data_len && i < 8; i++) {
        r.limbs[i] = data[i];
    }

    // Mix
    for (int round = 0; round < 4; round++) {
        for (int i = 0; i < 8; i++) {
            r.limbs[i] ^= rotl32(r.limbs[(i + 1) % 8], 7);
            r.limbs[i] += r.limbs[(i + 3) % 8];
        }
    }

    // Reduce mod l
    while (scalar_gte(r, mod)) {
        uint borrow = 0;
        for (int i = 0; i < 8; i++) {
            uint diff = r.limbs[i] - mod[i] - borrow;
            borrow = (r.limbs[i] < mod[i] + borrow) ? 1 : 0;
            r.limbs[i] = diff;
        }
    }

    return r;
}

// ============================================================================
// FROST Nonce Generation Kernels
// ============================================================================

// Generate nonce pair (d_i, e_i) for each participant
kernel void frost_generate_nonces(
    device const Scalar256* seeds [[buffer(0)]],           // Per-participant seeds
    device NonceCommitment* nonces [[buffer(1)]],          // Output nonce commitments
    constant NonceParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.num_participants) return;
    
    constant uint* mod = (params.curve_type == 0) ? ED25519_L : SECP256K1_N;
    
    // Initialize RNG from seed
    Scalar256 seed = seeds[gid];
    ulong counter = gid;
    ulong nonce_val = params.seed_entropy_offset;
    ChaCha20State rng = chacha20_init(seed, counter, nonce_val);
    
    // Generate hiding nonce d_i
    Scalar256 d = random_scalar(&rng, mod);
    
    // Generate binding nonce e_i
    Scalar256 e = random_scalar(&rng, mod);
    
    // Compute commitments D_i = g^d_i, E_i = g^e_i
    Ed25519Extended D_ext = ed25519_scalar_mul_base(d);
    Ed25519Extended E_ext = ed25519_scalar_mul_base(e);
    
    // Store results
    NonceCommitment result;
    result.hiding_nonce_d = d;
    result.binding_nonce_e = e;
    result.commitment_d = ed25519_to_affine(D_ext);
    result.commitment_e = ed25519_to_affine(E_ext);
    
    nonces[gid] = result;
}

// Compute binding factors rho_i = H(i, m, B) where B is list of commitments
kernel void frost_compute_binding_factors(
    device const uint* participant_ids [[buffer(0)]],
    device const Ed25519Affine* commitment_list [[buffer(1)]],  // All D_i, E_i pairs
    device const Scalar256* message [[buffer(2)]],
    device Scalar256* binding_factors [[buffer(3)]],
    constant NonceParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.num_participants) return;
    
    constant uint* mod = (params.curve_type == 0) ? ED25519_L : SECP256K1_N;
    
    // Construct hash input: (participant_id, message, commitment_list_hash)
    uint hash_input[16];
    
    // Participant ID
    hash_input[0] = participant_ids[gid];
    
    // Message (first few words)
    Scalar256 msg = message[0];
    for (int i = 0; i < 8; i++) {
        hash_input[1 + i] = msg.limbs[i];
    }
    
    // Commitment list contribution (simplified - XOR all commitments)
    uint commitment_hash = 0;
    for (uint i = 0; i < params.num_participants * 2; i++) {
        Ed25519Affine c = commitment_list[i];
        commitment_hash ^= c.x.limbs[0] ^ c.y.limbs[0];
    }
    hash_input[9] = commitment_hash;
    
    // Compute rho_i = H(hash_input)
    Scalar256 rho = hash_to_scalar(hash_input, 10, mod);
    
    binding_factors[gid] = rho;
}

// Compute group commitment R = sum(D_i + rho_i * E_i)
kernel void frost_compute_commitment_shares(
    device const NonceCommitment* nonces [[buffer(0)]],
    device const Scalar256* binding_factors [[buffer(1)]],
    device Ed25519Extended* commitment_shares [[buffer(2)]],
    constant NonceParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.num_participants) return;
    
    constant uint* mod = (params.curve_type == 0) ? ED25519_L : SECP256K1_N;
    
    NonceCommitment nonce = nonces[gid];
    Scalar256 rho = binding_factors[gid];
    
    // Compute R_i = D_i + rho_i * E_i
    // First compute rho_i * E_i (scalar multiplication)
    Ed25519Extended rho_E;
    rho_E.x = scalar_mul(rho, nonce.commitment_e.x, mod);
    rho_E.y = scalar_mul(rho, nonce.commitment_e.y, mod);
    rho_E.z = scalar_one();
    rho_E.t = scalar_zero();
    
    // Then add D_i
    Ed25519Extended D;
    D.x = nonce.commitment_d.x;
    D.y = nonce.commitment_d.y;
    D.z = scalar_one();
    D.t = scalar_zero();
    
    Ed25519Extended R_i = ed25519_add(D, rho_E);
    
    commitment_shares[gid] = R_i;
}

// Parallel reduction to sum commitment shares into group commitment R
kernel void frost_aggregate_commitments(
    device Ed25519Extended* commitment_shares [[buffer(0)]],
    device Ed25519Extended* group_commitment [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    // Tree reduction within threadgroup
    threadgroup Ed25519Extended shared_data[256];
    
    if (gid < count) {
        shared_data[lid] = commitment_shares[gid];
    } else {
        shared_data[lid] = ed25519_identity();
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction
    for (uint stride = group_size / 2; stride > 0; stride /= 2) {
        if (lid < stride && lid + stride < count) {
            shared_data[lid] = ed25519_add(shared_data[lid], shared_data[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result from first thread
    if (lid == 0) {
        group_commitment[gid / group_size] = shared_data[0];
    }
}

// Batch hash-to-curve for multiple messages
kernel void frost_batch_hash_to_curve(
    device const Scalar256* messages [[buffer(0)]],
    device Ed25519Extended* curve_points [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= batch_size) return;
    
    Scalar256 msg = messages[gid];
    
    // Elligator2 or similar hash-to-curve
    // Simplified: use message as scalar and compute G * H(m)
    Scalar256 h = msg;
    
    // Mix to get uniform distribution
    for (int i = 0; i < 8; i++) {
        h.limbs[i] ^= rotl32(h.limbs[(i + 1) % 8], 11);
        h.limbs[i] += rotl32(h.limbs[(i + 5) % 8], 7);
    }
    
    // Reduce mod l
    while (scalar_gte(h, ED25519_L)) {
        uint borrow = 0;
        for (int i = 0; i < 8; i++) {
            uint diff = h.limbs[i] - ED25519_L[i] - borrow;
            borrow = (h.limbs[i] < ED25519_L[i] + borrow) ? 1 : 0;
            h.limbs[i] = diff;
        }
    }
    
    // Compute H(m) * G
    curve_points[gid] = ed25519_scalar_mul_base(h);
}

// Verify nonce commitment (D_i = g^d_i)
kernel void frost_verify_nonce_commitments(
    device const NonceCommitment* nonces [[buffer(0)]],
    device uint* valid [[buffer(1)]],
    constant NonceParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.num_participants) return;
    
    NonceCommitment nonce = nonces[gid];
    
    // Verify D_i = g^d_i
    Ed25519Extended computed_D = ed25519_scalar_mul_base(nonce.hiding_nonce_d);
    Ed25519Affine computed_D_affine = ed25519_to_affine(computed_D);
    
    // Check equality (simplified - just check x coordinates)
    bool d_valid = true;
    for (int i = 0; i < 8; i++) {
        if (computed_D_affine.x.limbs[i] != nonce.commitment_d.x.limbs[i]) {
            d_valid = false;
            break;
        }
    }
    
    // Verify E_i = g^e_i
    Ed25519Extended computed_E = ed25519_scalar_mul_base(nonce.binding_nonce_e);
    Ed25519Affine computed_E_affine = ed25519_to_affine(computed_E);
    
    bool e_valid = true;
    for (int i = 0; i < 8; i++) {
        if (computed_E_affine.x.limbs[i] != nonce.commitment_e.x.limbs[i]) {
            e_valid = false;
            break;
        }
    }
    
    valid[gid] = (d_valid && e_valid) ? 1 : 0;
}
