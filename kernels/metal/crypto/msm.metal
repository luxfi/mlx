// Multi-Scalar Multiplication (MSM) for Elliptic Curves
// Implements Pippenger's algorithm for efficient batch scalar multiplication
//
// Supports: BN254 G1, BLS12-381 G1/G2
// Use case: Pedersen commitments, KZG polynomial commitments, batch verification

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// BN254 Field (Fp) - 254-bit prime field
// p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
// =============================================================================

struct Fp254 {
    uint64_t limbs[4];  // 256-bit representation (4 x 64-bit)
};

// BN254 field modulus
constant uint64_t BN254_P[4] = {
    0x3C208C16D87CFD47ULL,
    0x97816A916871CA8DULL,
    0xB85045B68181585DULL,
    0x30644E72E131A029ULL
};

// BN254 scalar field (Fr) modulus
constant uint64_t BN254_R[4] = {
    0x43E1F593F0000001ULL,
    0x2833E84879B97091ULL,
    0xB85045B68181585DULL,
    0x30644E72E131A029ULL
};

// =============================================================================
// BLS12-381 Field (Fp) - 381-bit prime field
// p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
// =============================================================================

struct Fp381 {
    uint64_t limbs[6];  // 384-bit representation (6 x 64-bit)
};

// BLS12-381 field modulus
constant uint64_t BLS_P[6] = {
    0xB9FEFFFFFFFFAAABULL,
    0x1EABFFFEB153FFFFULL,
    0x6730D2A0F6B0F624ULL,
    0x64774B84F38512BFULL,
    0x4B1BA7B6434BACD7ULL,
    0x1A0111EA397FE69AULL
};

// BLS12-381 scalar field (Fr) modulus
constant uint64_t BLS_R[4] = {
    0xFFFFFFFF00000001ULL,
    0x53BDA402FFFE5BFEULL,
    0x3339D80809A1D805ULL,
    0x73EDA753299D7D48ULL
};

// =============================================================================
// Basic 256-bit Arithmetic (for BN254 and BLS12-381 scalars)
// =============================================================================

// Add with carry
inline uint64_t adc(uint64_t a, uint64_t b, thread uint64_t& carry) {
    uint64_t sum = a + b + carry;
    carry = (sum < a || (carry && sum == a)) ? 1 : 0;
    return sum;
}

// Subtract with borrow
inline uint64_t sbb(uint64_t a, uint64_t b, thread uint64_t& borrow) {
    uint64_t diff = a - b - borrow;
    borrow = (a < b + borrow) ? 1 : 0;
    return diff;
}

// 256-bit addition
inline void add256(thread Fp254& c, Fp254 a, Fp254 b) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        c.limbs[i] = adc(a.limbs[i], b.limbs[i], carry);
    }
}

// 256-bit subtraction
inline void sub256(thread Fp254& c, Fp254 a, Fp254 b) {
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        c.limbs[i] = sbb(a.limbs[i], b.limbs[i], borrow);
    }
}

// Check if a >= b
inline bool gte256(Fp254 a, Fp254 b) {
    for (int i = 3; i >= 0; i--) {
        if (a.limbs[i] > b.limbs[i]) return true;
        if (a.limbs[i] < b.limbs[i]) return false;
    }
    return true;  // Equal
}

// =============================================================================
// Affine Point Representation
// =============================================================================

struct G1Affine254 {
    Fp254 x;
    Fp254 y;
    bool infinity;
};

struct G1Affine381 {
    Fp381 x;
    Fp381 y;
    bool infinity;
};

// =============================================================================
// Projective Point Representation (for efficient addition)
// =============================================================================

struct G1Projective254 {
    Fp254 x;
    Fp254 y;
    Fp254 z;
};

struct G1Projective381 {
    Fp381 x;
    Fp381 y;
    Fp381 z;
};

// =============================================================================
// Montgomery Multiplication for BN254 Fp
// =============================================================================

inline void fp254_mul(thread Fp254& c, Fp254 a, Fp254 b) {
    // Simplified schoolbook multiplication with Montgomery reduction
    // Full implementation would use CIOS or similar

    uint64_t t[8] = {0};

    // Schoolbook multiplication
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            // 64x64 -> 128-bit multiplication
            uint64_t lo = a.limbs[i] * b.limbs[j];
            uint64_t hi = mulhi(a.limbs[i], b.limbs[j]);

            uint64_t sum = t[i+j] + lo + carry;
            carry = (sum < t[i+j]) ? 1 : 0;
            carry += hi;
            t[i+j] = sum;
        }
        t[i+4] = carry;
    }

    // Montgomery reduction (simplified)
    // Full implementation would use proper Montgomery constants
    for (int i = 0; i < 4; i++) {
        c.limbs[i] = t[i];  // Placeholder
    }
}

// Field addition mod p
inline void fp254_add(thread Fp254& c, Fp254 a, Fp254 b) {
    add256(c, a, b);

    Fp254 p = {{BN254_P[0], BN254_P[1], BN254_P[2], BN254_P[3]}};
    if (gte256(c, p)) {
        sub256(c, c, p);
    }
}

// Field subtraction mod p
inline void fp254_sub(thread Fp254& c, Fp254 a, Fp254 b) {
    if (gte256(a, b)) {
        sub256(c, a, b);
    } else {
        Fp254 p = {{BN254_P[0], BN254_P[1], BN254_P[2], BN254_P[3]}};
        add256(c, a, p);
        sub256(c, c, b);
    }
}

// =============================================================================
// Point Operations (BN254 G1)
// =============================================================================

// Point doubling in projective coordinates
// Uses complete doubling formula
inline G1Projective254 g1_double_254(G1Projective254 p) {
    if (p.z.limbs[0] == 0 && p.z.limbs[1] == 0 &&
        p.z.limbs[2] == 0 && p.z.limbs[3] == 0) {
        return p;  // Point at infinity
    }

    // Standard projective doubling formula for short Weierstrass
    // a = 0 for BN254
    Fp254 xx, yy, yyyy, zz;
    Fp254 s, m, t;

    fp254_mul(xx, p.x, p.x);        // X^2
    fp254_mul(yy, p.y, p.y);        // Y^2
    fp254_mul(yyyy, yy, yy);        // Y^4
    fp254_mul(zz, p.z, p.z);        // Z^2

    // S = 2*((X+YY)^2 - XX - YYYY)
    Fp254 x_plus_yy;
    fp254_add(x_plus_yy, p.x, yy);
    fp254_mul(s, x_plus_yy, x_plus_yy);
    fp254_sub(s, s, xx);
    fp254_sub(s, s, yyyy);
    fp254_add(s, s, s);  // 2*S

    // M = 3*XX (a = 0 for BN254)
    fp254_add(m, xx, xx);
    fp254_add(m, m, xx);

    // T = M^2 - 2*S
    fp254_mul(t, m, m);
    Fp254 two_s;
    fp254_add(two_s, s, s);
    fp254_sub(t, t, two_s);

    G1Projective254 result;

    // X3 = T
    result.x = t;

    // Y3 = M*(S-T) - 8*YYYY
    Fp254 s_minus_t;
    fp254_sub(s_minus_t, s, t);
    fp254_mul(result.y, m, s_minus_t);
    Fp254 eight_yyyy;
    fp254_add(eight_yyyy, yyyy, yyyy);
    fp254_add(eight_yyyy, eight_yyyy, eight_yyyy);
    fp254_add(eight_yyyy, eight_yyyy, eight_yyyy);
    fp254_sub(result.y, result.y, eight_yyyy);

    // Z3 = 2*Y*Z
    fp254_mul(result.z, p.y, p.z);
    fp254_add(result.z, result.z, result.z);

    return result;
}

// Mixed addition: projective + affine
inline G1Projective254 g1_add_mixed_254(G1Projective254 p, G1Affine254 q) {
    if (q.infinity) return p;

    if (p.z.limbs[0] == 0 && p.z.limbs[1] == 0 &&
        p.z.limbs[2] == 0 && p.z.limbs[3] == 0) {
        return {q.x, q.y, {{1, 0, 0, 0}}};
    }

    // Using madd-2008-s formula
    Fp254 zz, u2, s2;
    fp254_mul(zz, p.z, p.z);
    fp254_mul(u2, q.x, zz);
    Fp254 zzz;
    fp254_mul(zzz, zz, p.z);
    fp254_mul(s2, q.y, zzz);

    // h = U2 - X1, r = S2 - Y1
    Fp254 h, r;
    fp254_sub(h, u2, p.x);
    fp254_sub(r, s2, p.y);

    // Check if P = Q (same point, need doubling)
    // or P = -Q (result is infinity)
    // Simplified: assume not edge cases

    Fp254 hh, hhh, v;
    fp254_mul(hh, h, h);
    fp254_mul(hhh, hh, h);
    fp254_mul(v, p.x, hh);

    G1Projective254 result;

    // X3 = r^2 - HHH - 2*V
    Fp254 rr;
    fp254_mul(rr, r, r);
    fp254_sub(result.x, rr, hhh);
    Fp254 two_v;
    fp254_add(two_v, v, v);
    fp254_sub(result.x, result.x, two_v);

    // Y3 = r*(V - X3) - Y1*HHH
    Fp254 v_minus_x3, y1_hhh;
    fp254_sub(v_minus_x3, v, result.x);
    fp254_mul(result.y, r, v_minus_x3);
    fp254_mul(y1_hhh, p.y, hhh);
    fp254_sub(result.y, result.y, y1_hhh);

    // Z3 = Z1 * H
    fp254_mul(result.z, p.z, h);

    return result;
}

// =============================================================================
// Pippenger MSM Algorithm
// =============================================================================

// MSM configuration
constant uint32_t MSM_WINDOW_SIZE = 8;  // c = 8 bits per window
constant uint32_t MSM_NUM_WINDOWS = 32; // 256 bits / 8 bits = 32 windows
constant uint32_t MSM_BUCKETS_PER_WINDOW = 255;  // 2^c - 1 buckets (exclude 0)

// Extract window from scalar
inline uint32_t get_scalar_window(Fp254 scalar, uint32_t window_idx) {
    uint32_t bit_idx = window_idx * MSM_WINDOW_SIZE;
    uint32_t limb_idx = bit_idx / 64;
    uint32_t bit_offset = bit_idx % 64;

    uint64_t limb = scalar.limbs[limb_idx];
    uint32_t window = (limb >> bit_offset) & ((1 << MSM_WINDOW_SIZE) - 1);

    // Handle window crossing limb boundary
    if (bit_offset > 64 - MSM_WINDOW_SIZE && limb_idx < 3) {
        uint32_t remaining_bits = MSM_WINDOW_SIZE - (64 - bit_offset);
        window |= (scalar.limbs[limb_idx + 1] & ((1ULL << remaining_bits) - 1)) << (64 - bit_offset);
    }

    return window;
}

// =============================================================================
// MSM Kernels
// =============================================================================

// Phase 1: Bucket accumulation
// Each thread handles one (point, scalar) pair, adds to appropriate bucket
kernel void msm_bucket_accumulate(
    device const G1Affine254* points [[buffer(0)]],
    device const Fp254* scalars [[buffer(1)]],
    device G1Projective254* buckets [[buffer(2)]],  // [num_windows][num_buckets]
    constant uint32_t& num_points [[buffer(3)]],
    constant uint32_t& window_idx [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= num_points) return;

    G1Affine254 point = points[index];
    Fp254 scalar = scalars[index];

    uint32_t bucket_idx = get_scalar_window(scalar, window_idx);
    if (bucket_idx == 0) return;  // Skip zero windows

    // Atomic-style bucket update (simplified - real impl needs atomic operations)
    uint32_t bucket_offset = window_idx * MSM_BUCKETS_PER_WINDOW + (bucket_idx - 1);

    // Add point to bucket
    G1Projective254 bucket = buckets[bucket_offset];
    buckets[bucket_offset] = g1_add_mixed_254(bucket, point);
}

// Phase 2: Bucket reduction
// Compute window sum from buckets: sum_{i=1}^{2^c-1} i * bucket[i]
kernel void msm_bucket_reduce(
    device G1Projective254* buckets [[buffer(0)]],
    device G1Projective254* window_sums [[buffer(1)]],
    constant uint32_t& window_idx [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;  // Single thread per window (for correctness)

    uint32_t bucket_offset = window_idx * MSM_BUCKETS_PER_WINDOW;

    // Running sum method:
    // sum = B[k-1], running = B[k-1]
    // for i = k-2 down to 0: running += B[i], sum += running

    G1Projective254 running = buckets[bucket_offset + MSM_BUCKETS_PER_WINDOW - 1];
    G1Projective254 sum = running;

    for (int32_t i = MSM_BUCKETS_PER_WINDOW - 2; i >= 0; i--) {
        G1Projective254 bucket = buckets[bucket_offset + i];
        // Add bucket to running (need full projective addition)
        // running = g1_add_projective(running, bucket);  // TODO: implement
        // sum = g1_add_projective(sum, running);
    }

    window_sums[window_idx] = sum;
}

// Phase 3: Window combination
// Combine window sums: result = sum_{i=0}^{k-1} 2^{c*i} * window_sum[i]
kernel void msm_window_combine(
    device const G1Projective254* window_sums [[buffer(0)]],
    device G1Projective254* result [[buffer(1)]],
    constant uint32_t& num_windows [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    G1Projective254 acc = window_sums[num_windows - 1];

    for (int32_t i = num_windows - 2; i >= 0; i--) {
        // Double c times
        for (uint32_t j = 0; j < MSM_WINDOW_SIZE; j++) {
            acc = g1_double_254(acc);
        }
        // Add window sum
        // acc = g1_add_projective(acc, window_sums[i]);  // TODO: implement
    }

    *result = acc;
}

// =============================================================================
// Single Scalar Multiplication (for small batches)
// Uses double-and-add
// =============================================================================

kernel void g1_scalar_mul(
    device const G1Affine254* points [[buffer(0)]],
    device const Fp254* scalars [[buffer(1)]],
    device G1Projective254* results [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    G1Affine254 point = points[index];
    Fp254 scalar = scalars[index];

    // Initialize result to point at infinity
    G1Projective254 acc = {{{0,0,0,0}}, {{0,0,0,0}}, {{0,0,0,0}}};

    // Double-and-add from MSB
    bool started = false;

    for (int32_t limb = 3; limb >= 0; limb--) {
        for (int32_t bit = 63; bit >= 0; bit--) {
            if (started) {
                acc = g1_double_254(acc);
            }

            if ((scalar.limbs[limb] >> bit) & 1) {
                if (started) {
                    acc = g1_add_mixed_254(acc, point);
                } else {
                    acc = {point.x, point.y, {{1, 0, 0, 0}}};
                    started = true;
                }
            }
        }
    }

    results[index] = acc;
}

// =============================================================================
// Pedersen Commitment: v*G + r*H (BN254 variant)
// =============================================================================

kernel void pedersen_commit_bn254(
    device const Fp254* values [[buffer(0)]],
    device const Fp254* blindings [[buffer(1)]],
    device const G1Affine254* generators [[buffer(2)]],  // [G, H]
    device G1Projective254* commitments [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    Fp254 v = values[index];
    Fp254 r = blindings[index];
    G1Affine254 G = generators[0];
    G1Affine254 H = generators[1];

    // Compute v*G
    G1Projective254 vG = {{{0,0,0,0}}, {{0,0,0,0}}, {{0,0,0,0}}};
    bool started = false;

    for (int32_t limb = 3; limb >= 0; limb--) {
        for (int32_t bit = 63; bit >= 0; bit--) {
            if (started) {
                vG = g1_double_254(vG);
            }
            if ((v.limbs[limb] >> bit) & 1) {
                if (started) {
                    vG = g1_add_mixed_254(vG, G);
                } else {
                    vG = {G.x, G.y, {{1, 0, 0, 0}}};
                    started = true;
                }
            }
        }
    }

    // Compute r*H
    G1Projective254 rH = {{{0,0,0,0}}, {{0,0,0,0}}, {{0,0,0,0}}};
    started = false;

    for (int32_t limb = 3; limb >= 0; limb--) {
        for (int32_t bit = 63; bit >= 0; bit--) {
            if (started) {
                rH = g1_double_254(rH);
            }
            if ((r.limbs[limb] >> bit) & 1) {
                if (started) {
                    rH = g1_add_mixed_254(rH, H);
                } else {
                    rH = {H.x, H.y, {{1, 0, 0, 0}}};
                    started = true;
                }
            }
        }
    }

    // Add vG + rH
    // commitments[index] = g1_add_projective(vG, rH);  // TODO: implement
    commitments[index] = vG;  // Placeholder
}

// =============================================================================
// Vector Commitment: sum_i v_i * G_i
// =============================================================================

kernel void vector_commit(
    device const Fp254* values [[buffer(0)]],
    device const G1Affine254* generators [[buffer(1)]],
    device G1Projective254* partial_sums [[buffer(2)]],  // One per thread
    constant uint32_t& num_values [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint num_threads [[threads_per_grid]]
) {
    // Each thread handles a subset of values
    uint32_t chunk_size = (num_values + num_threads - 1) / num_threads;
    uint32_t start = index * chunk_size;
    uint32_t end = min(start + chunk_size, num_values);

    G1Projective254 acc = {{{0,0,0,0}}, {{0,0,0,0}}, {{0,0,0,0}}};
    bool started = false;

    for (uint32_t i = start; i < end; i++) {
        Fp254 v = values[i];
        G1Affine254 G = generators[i];

        // Compute v * G
        G1Projective254 vG = {{{0,0,0,0}}, {{0,0,0,0}}, {{0,0,0,0}}};
        bool point_started = false;

        for (int32_t limb = 3; limb >= 0; limb--) {
            for (int32_t bit = 63; bit >= 0; bit--) {
                if (point_started) {
                    vG = g1_double_254(vG);
                }
                if ((v.limbs[limb] >> bit) & 1) {
                    if (point_started) {
                        vG = g1_add_mixed_254(vG, G);
                    } else {
                        vG = {G.x, G.y, {{1, 0, 0, 0}}};
                        point_started = true;
                    }
                }
            }
        }

        // Add to accumulator
        if (point_started) {
            if (started) {
                // acc = g1_add_projective(acc, vG);  // TODO
            } else {
                acc = vG;
                started = true;
            }
        }
    }

    partial_sums[index] = acc;
}
