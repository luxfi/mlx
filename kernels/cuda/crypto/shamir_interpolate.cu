// Copyright (c) 2024-2026 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// Shamir Secret Sharing CUDA Kernels
// Lagrange interpolation, share generation, and batch operations for
// threshold cryptography (FROST, CGGMP21) on NVIDIA GPUs.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace lux {
namespace cuda {
namespace shamir {

// =============================================================================
// Configuration
// =============================================================================

#define SHAMIR_MAX_SHARES      256
#define SHAMIR_MAX_THRESHOLD   128
#define SHAMIR_BATCH_SIZE      256

// Curve type enumeration
enum CurveType : uint32_t {
    CURVE_SECP256K1 = 0,
    CURVE_ED25519   = 1,
};

// =============================================================================
// 256-bit Scalar Field Element (4 x 64-bit limbs, little-endian)
// =============================================================================

struct Scalar256 {
    uint64_t limbs[4];
};

// =============================================================================
// Field Constants
// =============================================================================

// secp256k1 curve order: n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
__constant__ uint64_t SECP256K1_N[4] = {
    0xBFD25E8CD0364141ULL,
    0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL,
    0xFFFFFFFFFFFFFFFFULL
};

// secp256k1 Montgomery constant: -n^{-1} mod 2^64
__constant__ uint64_t SECP256K1_N_INV = 0x4B0DFF665588B13FULL;

// secp256k1 R^2 mod n (for Montgomery conversion)
__constant__ uint64_t SECP256K1_R2[4] = {
    0x9D671CD581C69BC5ULL,
    0xD8E5FC26E83E3E88ULL,
    0xC6A68F6E7F8F3AA6ULL,
    0xB77D8CE3A79C12AFULL
};

// ed25519 scalar field order: L = 2^252 + 27742317777372353535851937790883648493
__constant__ uint64_t ED25519_L[4] = {
    0x5812631A5CF5D3EDULL,
    0x14DEF9DEA2F79CD6ULL,
    0x0000000000000000ULL,
    0x1000000000000000ULL
};

// ed25519 Montgomery constant: -L^{-1} mod 2^64
__constant__ uint64_t ED25519_L_INV = 0xD2B51DA312547E1BULL;

// ed25519 R^2 mod L
__constant__ uint64_t ED25519_R2[4] = {
    0xA40611E3449C0F01ULL,
    0xD00E1BA768859347ULL,
    0xCEEC73D217F5BE65ULL,
    0x0399411B7C309A3DULL
};

// =============================================================================
// Basic 256-bit Arithmetic
// =============================================================================

// Add with carry
__device__ __forceinline__
uint64_t adc(uint64_t a, uint64_t b, uint64_t& carry) {
    uint64_t sum = a + b + carry;
    carry = (sum < a || (carry && sum == a)) ? 1ULL : 0ULL;
    return sum;
}

// Subtract with borrow
__device__ __forceinline__
uint64_t sbb(uint64_t a, uint64_t b, uint64_t& borrow) {
    uint64_t diff = a - b - borrow;
    borrow = (a < b || (borrow && a == b)) ? 1ULL : 0ULL;
    return diff;
}

// Multiply-accumulate with carry
__device__ __forceinline__
void mac(uint64_t& lo, uint64_t& carry, uint64_t a, uint64_t b, uint64_t c) {
    uint64_t product_lo = a * b;
    uint64_t product_hi = __umul64hi(a, b);

    uint64_t sum = product_lo + c;
    uint64_t c1 = (sum < product_lo) ? 1ULL : 0ULL;

    sum += carry;
    uint64_t c2 = (sum < carry) ? 1ULL : 0ULL;

    lo = sum;
    carry = product_hi + c1 + c2;
}

// Check if scalar is zero
__device__ __forceinline__
bool scalar_is_zero(const Scalar256& a) {
    return (a.limbs[0] | a.limbs[1] | a.limbs[2] | a.limbs[3]) == 0;
}

// Compare: returns 1 if a >= b, 0 otherwise
__device__ __forceinline__
int scalar_gte(const Scalar256& a, const uint64_t* b) {
    for (int i = 3; i >= 0; i--) {
        if (a.limbs[i] > b[i]) return 1;
        if (a.limbs[i] < b[i]) return 0;
    }
    return 1;  // equal
}

// Set scalar to constant
__device__ __forceinline__
Scalar256 scalar_from_u64(uint64_t v) {
    Scalar256 r;
    r.limbs[0] = v;
    r.limbs[1] = 0;
    r.limbs[2] = 0;
    r.limbs[3] = 0;
    return r;
}

// =============================================================================
// Field-Specific Modular Arithmetic
// =============================================================================

// Get modulus pointer based on curve type
__device__ __forceinline__
const uint64_t* get_modulus(CurveType curve) {
    return (curve == CURVE_ED25519) ? ED25519_L : SECP256K1_N;
}

// Get Montgomery inverse based on curve type
__device__ __forceinline__
uint64_t get_m_inv(CurveType curve) {
    return (curve == CURVE_ED25519) ? ED25519_L_INV : SECP256K1_N_INV;
}

// Get R^2 based on curve type
__device__ __forceinline__
const uint64_t* get_r2(CurveType curve) {
    return (curve == CURVE_ED25519) ? ED25519_R2 : SECP256K1_R2;
}

// Modular addition: (a + b) mod n
__device__
Scalar256 scalar_add(const Scalar256& a, const Scalar256& b, CurveType curve) {
    const uint64_t* mod = get_modulus(curve);
    Scalar256 result;
    uint64_t carry = 0;

    result.limbs[0] = adc(a.limbs[0], b.limbs[0], carry);
    result.limbs[1] = adc(a.limbs[1], b.limbs[1], carry);
    result.limbs[2] = adc(a.limbs[2], b.limbs[2], carry);
    result.limbs[3] = adc(a.limbs[3], b.limbs[3], carry);

    // Conditional reduction
    uint64_t borrow = 0;
    Scalar256 reduced;
    reduced.limbs[0] = sbb(result.limbs[0], mod[0], borrow);
    reduced.limbs[1] = sbb(result.limbs[1], mod[1], borrow);
    reduced.limbs[2] = sbb(result.limbs[2], mod[2], borrow);
    reduced.limbs[3] = sbb(result.limbs[3], mod[3], borrow);

    bool needs_reduce = (carry != 0) || (borrow == 0);
    if (needs_reduce) return reduced;
    return result;
}

// Modular subtraction: (a - b) mod n
__device__
Scalar256 scalar_sub(const Scalar256& a, const Scalar256& b, CurveType curve) {
    const uint64_t* mod = get_modulus(curve);
    Scalar256 result;
    uint64_t borrow = 0;

    result.limbs[0] = sbb(a.limbs[0], b.limbs[0], borrow);
    result.limbs[1] = sbb(a.limbs[1], b.limbs[1], borrow);
    result.limbs[2] = sbb(a.limbs[2], b.limbs[2], borrow);
    result.limbs[3] = sbb(a.limbs[3], b.limbs[3], borrow);

    if (borrow) {
        uint64_t carry = 0;
        result.limbs[0] = adc(result.limbs[0], mod[0], carry);
        result.limbs[1] = adc(result.limbs[1], mod[1], carry);
        result.limbs[2] = adc(result.limbs[2], mod[2], carry);
        result.limbs[3] = adc(result.limbs[3], mod[3], carry);
    }

    return result;
}

// Reduce 512-bit value modulo n (generic for both curves)
__device__
Scalar256 scalar_reduce_512(uint64_t t[8], CurveType curve) {
    const uint64_t* mod = get_modulus(curve);
    
    // Barrett reduction (simplified for 256-bit modulus)
    // For production, use Montgomery reduction for better performance
    Scalar256 result = {{t[0], t[1], t[2], t[3]}};
    Scalar256 high = {{t[4], t[5], t[6], t[7]}};
    
    // Reduce high part first (iterative subtraction for correctness)
    // This is simplified; production should use Barrett or Montgomery
    while (!scalar_is_zero(high) || scalar_gte(result, mod)) {
        if (scalar_gte(result, mod)) {
            uint64_t borrow = 0;
            result.limbs[0] = sbb(result.limbs[0], mod[0], borrow);
            result.limbs[1] = sbb(result.limbs[1], mod[1], borrow);
            result.limbs[2] = sbb(result.limbs[2], mod[2], borrow);
            result.limbs[3] = sbb(result.limbs[3], mod[3], borrow);
        }
        
        if (!scalar_is_zero(high)) {
            // Shift high part contribution
            uint64_t carry = 0;
            result.limbs[0] = adc(result.limbs[0], high.limbs[0], carry);
            result.limbs[1] = adc(result.limbs[1], high.limbs[1], carry);
            result.limbs[2] = adc(result.limbs[2], high.limbs[2], carry);
            result.limbs[3] = adc(result.limbs[3], high.limbs[3], carry);
            high = {{0, 0, 0, 0}};
        }
    }
    
    return result;
}

// Modular multiplication: (a * b) mod n
__device__
Scalar256 scalar_mul(const Scalar256& a, const Scalar256& b, CurveType curve) {
    uint64_t t[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    // Schoolbook multiplication
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            mac(t[i + j], carry, a.limbs[i], b.limbs[j], t[i + j]);
        }
        t[i + 4] = carry;
    }

    return scalar_reduce_512(t, curve);
}

// Modular squaring
__device__ __forceinline__
Scalar256 scalar_sqr(const Scalar256& a, CurveType curve) {
    return scalar_mul(a, a, curve);
}

// Modular exponentiation using square-and-multiply
__device__
Scalar256 scalar_pow(const Scalar256& base, const Scalar256& exp, CurveType curve) {
    Scalar256 result = scalar_from_u64(1);
    Scalar256 b = base;

    for (int limb = 0; limb < 4; limb++) {
        uint64_t e = exp.limbs[limb];
        for (int bit = 0; bit < 64; bit++) {
            if ((e >> bit) & 1) {
                result = scalar_mul(result, b, curve);
            }
            b = scalar_sqr(b, curve);
        }
    }

    return result;
}

// Modular inverse using Fermat's little theorem: a^{-1} = a^{n-2} mod n
__device__
Scalar256 scalar_inv(const Scalar256& a, CurveType curve) {
    const uint64_t* mod = get_modulus(curve);
    
    // exp = n - 2
    Scalar256 exp;
    uint64_t borrow = 0;
    exp.limbs[0] = sbb(mod[0], 2, borrow);
    exp.limbs[1] = sbb(mod[1], 0, borrow);
    exp.limbs[2] = sbb(mod[2], 0, borrow);
    exp.limbs[3] = sbb(mod[3], 0, borrow);

    return scalar_pow(a, exp, curve);
}

// =============================================================================
// Montgomery's Trick for Batch Inverse
// =============================================================================
//
// Compute n inverses using only 1 inversion and 3(n-1) multiplications:
//   products[i] = a[0] * a[1] * ... * a[i]
//   inv = inverse(products[n-1])
//   for i = n-1 down to 1:
//     result[i] = inv * products[i-1]
//     inv = inv * a[i]
//   result[0] = inv
//

__device__
void batch_inverse_montgomery(
    const Scalar256* inputs,
    Scalar256* outputs,
    uint32_t count,
    CurveType curve,
    Scalar256* scratch  // scratch space for count elements
) {
    if (count == 0) return;
    
    if (count == 1) {
        outputs[0] = scalar_inv(inputs[0], curve);
        return;
    }
    
    // Phase 1: Compute cumulative products
    scratch[0] = inputs[0];
    for (uint32_t i = 1; i < count; i++) {
        scratch[i] = scalar_mul(scratch[i - 1], inputs[i], curve);
    }
    
    // Phase 2: Single inversion
    Scalar256 inv = scalar_inv(scratch[count - 1], curve);
    
    // Phase 3: Back-propagate inverses
    for (int32_t i = count - 1; i > 0; i--) {
        outputs[i] = scalar_mul(inv, scratch[i - 1], curve);
        inv = scalar_mul(inv, inputs[i], curve);
    }
    outputs[0] = inv;
}

// =============================================================================
// Lagrange Interpolation
// =============================================================================

// Compute single Lagrange coefficient: L_j(x) = prod_{i!=j} (x - x_i) / (x_j - x_i)
__device__
Scalar256 lagrange_coefficient(
    uint32_t j,
    const Scalar256* x_coords,
    uint32_t count,
    const Scalar256& eval_point,
    CurveType curve
) {
    Scalar256 numerator = scalar_from_u64(1);
    Scalar256 denominator = scalar_from_u64(1);
    
    for (uint32_t i = 0; i < count; i++) {
        if (i == j) continue;
        
        // numerator *= (eval_point - x_i)
        Scalar256 num_term = scalar_sub(eval_point, x_coords[i], curve);
        numerator = scalar_mul(numerator, num_term, curve);
        
        // denominator *= (x_j - x_i)
        Scalar256 den_term = scalar_sub(x_coords[j], x_coords[i], curve);
        denominator = scalar_mul(denominator, den_term, curve);
    }
    
    // Return numerator / denominator
    Scalar256 den_inv = scalar_inv(denominator, curve);
    return scalar_mul(numerator, den_inv, curve);
}

// Compute all Lagrange coefficients for interpolation at x=0 (secret recovery)
// This is the common case: recovering secret from shares at indices x_1, x_2, ..., x_t
__device__
void lagrange_coefficients_at_zero(
    const Scalar256* x_coords,
    uint32_t count,
    Scalar256* coefficients,
    CurveType curve,
    Scalar256* scratch  // for batch inverse
) {
    // For x=0: L_j(0) = prod_{i!=j} (-x_i) / (x_j - x_i)
    //                 = prod_{i!=j} x_i / (x_i - x_j)  [after simplification]
    
    // Compute denominators first
    Scalar256 denominators[SHAMIR_MAX_THRESHOLD];
    
    for (uint32_t j = 0; j < count; j++) {
        Scalar256 denom = scalar_from_u64(1);
        for (uint32_t i = 0; i < count; i++) {
            if (i == j) continue;
            Scalar256 diff = scalar_sub(x_coords[j], x_coords[i], curve);
            denom = scalar_mul(denom, diff, curve);
        }
        denominators[j] = denom;
    }
    
    // Batch invert denominators using Montgomery's trick
    Scalar256 denom_invs[SHAMIR_MAX_THRESHOLD];
    batch_inverse_montgomery(denominators, denom_invs, count, curve, scratch);
    
    // Compute numerators and final coefficients
    for (uint32_t j = 0; j < count; j++) {
        Scalar256 numer = scalar_from_u64(1);
        for (uint32_t i = 0; i < count; i++) {
            if (i == j) continue;
            numer = scalar_mul(numer, x_coords[i], curve);
        }
        
        // Adjust sign: prod(-x_i) for i != j
        // Since we're working in a prime field, -x = n - x
        // Number of terms is (count - 1), so sign is (-1)^(count-1)
        if ((count - 1) & 1) {
            // Odd number of terms, negate
            const uint64_t* mod = get_modulus(curve);
            Scalar256 zero = scalar_from_u64(0);
            numer = scalar_sub(zero, numer, curve);
        }
        
        coefficients[j] = scalar_mul(numer, denom_invs[j], curve);
    }
}

// Interpolate: recover secret s(0) from shares (x_i, y_i)
__device__
Scalar256 lagrange_interpolate_at_zero(
    const Scalar256* x_coords,
    const Scalar256* y_coords,
    uint32_t count,
    CurveType curve,
    Scalar256* scratch
) {
    Scalar256 coefficients[SHAMIR_MAX_THRESHOLD];
    lagrange_coefficients_at_zero(x_coords, count, coefficients, curve, scratch);
    
    Scalar256 result = scalar_from_u64(0);
    for (uint32_t i = 0; i < count; i++) {
        Scalar256 term = scalar_mul(coefficients[i], y_coords[i], curve);
        result = scalar_add(result, term, curve);
    }
    
    return result;
}

// General interpolation at arbitrary point
__device__
Scalar256 lagrange_interpolate(
    const Scalar256* x_coords,
    const Scalar256* y_coords,
    uint32_t count,
    const Scalar256& eval_point,
    CurveType curve
) {
    Scalar256 result = scalar_from_u64(0);
    
    for (uint32_t j = 0; j < count; j++) {
        Scalar256 coeff = lagrange_coefficient(j, x_coords, count, eval_point, curve);
        Scalar256 term = scalar_mul(coeff, y_coords[j], curve);
        result = scalar_add(result, term, curve);
    }
    
    return result;
}

// =============================================================================
// Share Generation (Polynomial Evaluation)
// =============================================================================

// Evaluate polynomial at a point using Horner's method
// poly[0] is constant term (secret), poly[t-1] is highest degree coefficient
__device__
Scalar256 poly_eval(
    const Scalar256* coefficients,
    uint32_t degree,
    const Scalar256& x,
    CurveType curve
) {
    Scalar256 result = coefficients[degree - 1];
    
    for (int32_t i = degree - 2; i >= 0; i--) {
        result = scalar_mul(result, x, curve);
        result = scalar_add(result, coefficients[i], curve);
    }
    
    return result;
}

// Generate share for party with given ID (1-indexed typically)
__device__
Scalar256 generate_share(
    const Scalar256* coefficients,
    uint32_t threshold,
    uint32_t party_id,
    CurveType curve
) {
    Scalar256 x = scalar_from_u64(party_id);
    return poly_eval(coefficients, threshold, x, curve);
}

// =============================================================================
// Proactive Refresh
// =============================================================================

// Generate zero-share for proactive refresh
// This creates a random polynomial with f(0) = 0
__device__
void generate_refresh_share(
    const Scalar256* random_coefficients,  // t-1 random values (no constant term)
    uint32_t threshold,
    uint32_t party_id,
    Scalar256& refresh_share,
    CurveType curve
) {
    Scalar256 x = scalar_from_u64(party_id);
    
    // Evaluate x * (a_1 + a_2*x + ... + a_{t-1}*x^{t-2})
    // which equals a_1*x + a_2*x^2 + ... + a_{t-1}*x^{t-1}
    
    if (threshold <= 1) {
        refresh_share = scalar_from_u64(0);
        return;
    }
    
    // Start with highest coefficient
    Scalar256 result = random_coefficients[threshold - 2];
    
    for (int32_t i = threshold - 3; i >= 0; i--) {
        result = scalar_mul(result, x, curve);
        result = scalar_add(result, random_coefficients[i], curve);
    }
    
    // Final multiply by x
    refresh_share = scalar_mul(result, x, curve);
}

// Add refresh delta to existing share
__device__
Scalar256 apply_refresh(
    const Scalar256& old_share,
    const Scalar256& refresh_delta,
    CurveType curve
) {
    return scalar_add(old_share, refresh_delta, curve);
}

// =============================================================================
// Share Verification
// =============================================================================

// Verify that shares are consistent (from same polynomial)
// Uses Lagrange interpolation: any t shares should give same secret
__device__
bool verify_share_consistency(
    const Scalar256* x_coords,
    const Scalar256* y_coords,
    uint32_t count,
    uint32_t threshold,
    CurveType curve,
    Scalar256* scratch
) {
    if (count < threshold) return false;
    
    // Reconstruct using first t shares
    Scalar256 secret1 = lagrange_interpolate_at_zero(
        x_coords, y_coords, threshold, curve, scratch
    );
    
    // Verify remaining shares
    for (uint32_t i = threshold; i < count; i++) {
        // Check if share i is consistent
        Scalar256 reconstructed = lagrange_interpolate(
            x_coords, y_coords, threshold, x_coords[i], curve
        );
        
        // Compare with actual share
        bool match = (reconstructed.limbs[0] == y_coords[i].limbs[0]) &&
                     (reconstructed.limbs[1] == y_coords[i].limbs[1]) &&
                     (reconstructed.limbs[2] == y_coords[i].limbs[2]) &&
                     (reconstructed.limbs[3] == y_coords[i].limbs[3]);
        
        if (!match) return false;
    }
    
    return true;
}

// =============================================================================
// CUDA Kernels
// =============================================================================

// Batch secret reconstruction kernel
__global__ void shamir_reconstruct_kernel(
    const Scalar256* x_coords,      // [batch_size][threshold]
    const Scalar256* y_coords,      // [batch_size][threshold]
    Scalar256* secrets,             // [batch_size] output
    uint32_t threshold,
    uint32_t batch_size,
    CurveType curve
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    // Scratch space in shared memory per thread
    __shared__ Scalar256 scratch[SHAMIR_BATCH_SIZE][SHAMIR_MAX_THRESHOLD];
    
    const Scalar256* x_ptr = x_coords + idx * threshold;
    const Scalar256* y_ptr = y_coords + idx * threshold;
    
    secrets[idx] = lagrange_interpolate_at_zero(
        x_ptr, y_ptr, threshold, curve, scratch[threadIdx.x]
    );
}

// Batch share generation kernel
__global__ void shamir_generate_shares_kernel(
    const Scalar256* coefficients,  // [batch_size][threshold] polynomials
    const uint32_t* party_ids,      // [num_parties] party IDs (1-indexed)
    Scalar256* shares,              // [batch_size][num_parties] output
    uint32_t threshold,
    uint32_t num_parties,
    uint32_t batch_size,
    CurveType curve
) {
    uint32_t batch_idx = blockIdx.x;
    uint32_t party_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || party_idx >= num_parties) return;
    
    const Scalar256* poly = coefficients + batch_idx * threshold;
    uint32_t party_id = party_ids[party_idx];
    
    shares[batch_idx * num_parties + party_idx] = generate_share(
        poly, threshold, party_id, curve
    );
}

// Batch Lagrange coefficient computation kernel
__global__ void shamir_lagrange_coeffs_kernel(
    const Scalar256* x_coords,      // [num_parties]
    Scalar256* coefficients,        // [num_parties] output
    uint32_t num_parties,
    CurveType curve
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_parties) return;
    
    __shared__ Scalar256 scratch[SHAMIR_BATCH_SIZE];
    
    // Each thread computes one coefficient
    Scalar256 local_coeffs[SHAMIR_MAX_THRESHOLD];
    lagrange_coefficients_at_zero(x_coords, num_parties, local_coeffs, curve, scratch);
    
    coefficients[idx] = local_coeffs[idx];
}

// Batch inverse kernel using Montgomery's trick
__global__ void shamir_batch_inverse_kernel(
    const Scalar256* inputs,
    Scalar256* outputs,
    uint32_t count,
    CurveType curve
) {
    // Single thread for now (batch inverse is inherently sequential)
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    extern __shared__ Scalar256 scratch[];
    batch_inverse_montgomery(inputs, outputs, count, curve, scratch);
}

// Proactive refresh kernel
__global__ void shamir_refresh_shares_kernel(
    const Scalar256* old_shares,        // [num_parties]
    const Scalar256* refresh_coeffs,    // [threshold-1] random coefficients
    Scalar256* new_shares,              // [num_parties] output
    const uint32_t* party_ids,
    uint32_t threshold,
    uint32_t num_parties,
    CurveType curve
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_parties) return;
    
    Scalar256 refresh_delta;
    generate_refresh_share(refresh_coeffs, threshold, party_ids[idx], refresh_delta, curve);
    
    new_shares[idx] = apply_refresh(old_shares[idx], refresh_delta, curve);
}

// Share verification kernel
__global__ void shamir_verify_shares_kernel(
    const Scalar256* x_coords,
    const Scalar256* y_coords,
    uint32_t count,
    uint32_t threshold,
    int* is_valid,
    CurveType curve
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    extern __shared__ Scalar256 scratch[];
    *is_valid = verify_share_consistency(
        x_coords, y_coords, count, threshold, curve, scratch
    ) ? 1 : 0;
}

} // namespace shamir
} // namespace cuda
} // namespace lux

// =============================================================================
// C API
// =============================================================================

extern "C" {

using namespace lux::cuda::shamir;

// Reconstruct secret from threshold shares
int lux_cuda_shamir_reconstruct(
    const void* x_coords,       // Scalar256[threshold]
    const void* y_coords,       // Scalar256[threshold]
    void* secret,               // Scalar256 output
    uint32_t threshold,
    uint32_t curve_type,
    cudaStream_t stream
) {
    Scalar256* d_x_coords;
    Scalar256* d_y_coords;
    Scalar256* d_secret;
    
    cudaMalloc(&d_x_coords, threshold * sizeof(Scalar256));
    cudaMalloc(&d_y_coords, threshold * sizeof(Scalar256));
    cudaMalloc(&d_secret, sizeof(Scalar256));
    
    cudaMemcpyAsync(d_x_coords, x_coords, threshold * sizeof(Scalar256),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_y_coords, y_coords, threshold * sizeof(Scalar256),
                    cudaMemcpyHostToDevice, stream);
    
    shamir_reconstruct_kernel<<<1, 1, 0, stream>>>(
        d_x_coords, d_y_coords, d_secret, threshold, 1, (CurveType)curve_type
    );
    
    cudaMemcpyAsync(secret, d_secret, sizeof(Scalar256),
                    cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    
    cudaFree(d_x_coords);
    cudaFree(d_y_coords);
    cudaFree(d_secret);
    
    return cudaGetLastError();
}

// Batch reconstruct multiple secrets
int lux_cuda_shamir_batch_reconstruct(
    const void* x_coords,       // Scalar256[batch_size][threshold]
    const void* y_coords,       // Scalar256[batch_size][threshold]
    void* secrets,              // Scalar256[batch_size] output
    uint32_t threshold,
    uint32_t batch_size,
    uint32_t curve_type,
    cudaStream_t stream
) {
    Scalar256* d_x_coords;
    Scalar256* d_y_coords;
    Scalar256* d_secrets;
    
    size_t shares_size = batch_size * threshold * sizeof(Scalar256);
    size_t secrets_size = batch_size * sizeof(Scalar256);
    
    cudaMalloc(&d_x_coords, shares_size);
    cudaMalloc(&d_y_coords, shares_size);
    cudaMalloc(&d_secrets, secrets_size);
    
    cudaMemcpyAsync(d_x_coords, x_coords, shares_size,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_y_coords, y_coords, shares_size,
                    cudaMemcpyHostToDevice, stream);
    
    dim3 block(256);
    dim3 grid((batch_size + block.x - 1) / block.x);
    
    shamir_reconstruct_kernel<<<grid, block, 0, stream>>>(
        d_x_coords, d_y_coords, d_secrets, threshold, batch_size, (CurveType)curve_type
    );
    
    cudaMemcpyAsync(secrets, d_secrets, secrets_size,
                    cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    
    cudaFree(d_x_coords);
    cudaFree(d_y_coords);
    cudaFree(d_secrets);
    
    return cudaGetLastError();
}

// Generate shares from polynomial coefficients
int lux_cuda_shamir_generate_shares(
    const void* coefficients,   // Scalar256[threshold] polynomial
    const uint32_t* party_ids,  // party IDs (1-indexed)
    void* shares,               // Scalar256[num_parties] output
    uint32_t threshold,
    uint32_t num_parties,
    uint32_t curve_type,
    cudaStream_t stream
) {
    Scalar256* d_coeffs;
    uint32_t* d_party_ids;
    Scalar256* d_shares;
    
    cudaMalloc(&d_coeffs, threshold * sizeof(Scalar256));
    cudaMalloc(&d_party_ids, num_parties * sizeof(uint32_t));
    cudaMalloc(&d_shares, num_parties * sizeof(Scalar256));
    
    cudaMemcpyAsync(d_coeffs, coefficients, threshold * sizeof(Scalar256),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_party_ids, party_ids, num_parties * sizeof(uint32_t),
                    cudaMemcpyHostToDevice, stream);
    
    shamir_generate_shares_kernel<<<1, num_parties, 0, stream>>>(
        d_coeffs, d_party_ids, d_shares, threshold, num_parties, 1, (CurveType)curve_type
    );
    
    cudaMemcpyAsync(shares, d_shares, num_parties * sizeof(Scalar256),
                    cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    
    cudaFree(d_coeffs);
    cudaFree(d_party_ids);
    cudaFree(d_shares);
    
    return cudaGetLastError();
}

// Compute Lagrange coefficients for secret reconstruction at x=0
int lux_cuda_shamir_lagrange_coefficients(
    const void* x_coords,       // Scalar256[num_parties]
    void* coefficients,         // Scalar256[num_parties] output
    uint32_t num_parties,
    uint32_t curve_type,
    cudaStream_t stream
) {
    Scalar256* d_x_coords;
    Scalar256* d_coeffs;
    
    cudaMalloc(&d_x_coords, num_parties * sizeof(Scalar256));
    cudaMalloc(&d_coeffs, num_parties * sizeof(Scalar256));
    
    cudaMemcpyAsync(d_x_coords, x_coords, num_parties * sizeof(Scalar256),
                    cudaMemcpyHostToDevice, stream);
    
    dim3 block(256);
    dim3 grid((num_parties + block.x - 1) / block.x);
    
    shamir_lagrange_coeffs_kernel<<<grid, block, 0, stream>>>(
        d_x_coords, d_coeffs, num_parties, (CurveType)curve_type
    );
    
    cudaMemcpyAsync(coefficients, d_coeffs, num_parties * sizeof(Scalar256),
                    cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    
    cudaFree(d_x_coords);
    cudaFree(d_coeffs);
    
    return cudaGetLastError();
}

// Batch inverse using Montgomery's trick
int lux_cuda_shamir_batch_inverse(
    const void* inputs,         // Scalar256[count]
    void* outputs,              // Scalar256[count] output
    uint32_t count,
    uint32_t curve_type,
    cudaStream_t stream
) {
    Scalar256* d_inputs;
    Scalar256* d_outputs;
    
    cudaMalloc(&d_inputs, count * sizeof(Scalar256));
    cudaMalloc(&d_outputs, count * sizeof(Scalar256));
    
    cudaMemcpyAsync(d_inputs, inputs, count * sizeof(Scalar256),
                    cudaMemcpyHostToDevice, stream);
    
    size_t shared_size = count * sizeof(Scalar256);
    shamir_batch_inverse_kernel<<<1, 1, shared_size, stream>>>(
        d_inputs, d_outputs, count, (CurveType)curve_type
    );
    
    cudaMemcpyAsync(outputs, d_outputs, count * sizeof(Scalar256),
                    cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    
    cudaFree(d_inputs);
    cudaFree(d_outputs);
    
    return cudaGetLastError();
}

// Proactive share refresh
int lux_cuda_shamir_refresh_shares(
    const void* old_shares,         // Scalar256[num_parties]
    const void* refresh_coeffs,     // Scalar256[threshold-1]
    void* new_shares,               // Scalar256[num_parties] output
    const uint32_t* party_ids,
    uint32_t threshold,
    uint32_t num_parties,
    uint32_t curve_type,
    cudaStream_t stream
) {
    Scalar256* d_old_shares;
    Scalar256* d_refresh_coeffs;
    Scalar256* d_new_shares;
    uint32_t* d_party_ids;
    
    cudaMalloc(&d_old_shares, num_parties * sizeof(Scalar256));
    cudaMalloc(&d_refresh_coeffs, (threshold - 1) * sizeof(Scalar256));
    cudaMalloc(&d_new_shares, num_parties * sizeof(Scalar256));
    cudaMalloc(&d_party_ids, num_parties * sizeof(uint32_t));
    
    cudaMemcpyAsync(d_old_shares, old_shares, num_parties * sizeof(Scalar256),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_refresh_coeffs, refresh_coeffs, (threshold - 1) * sizeof(Scalar256),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_party_ids, party_ids, num_parties * sizeof(uint32_t),
                    cudaMemcpyHostToDevice, stream);
    
    dim3 block(256);
    dim3 grid((num_parties + block.x - 1) / block.x);
    
    shamir_refresh_shares_kernel<<<grid, block, 0, stream>>>(
        d_old_shares, d_refresh_coeffs, d_new_shares, d_party_ids,
        threshold, num_parties, (CurveType)curve_type
    );
    
    cudaMemcpyAsync(new_shares, d_new_shares, num_parties * sizeof(Scalar256),
                    cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    
    cudaFree(d_old_shares);
    cudaFree(d_refresh_coeffs);
    cudaFree(d_new_shares);
    cudaFree(d_party_ids);
    
    return cudaGetLastError();
}

// Verify share consistency
int lux_cuda_shamir_verify_shares(
    const void* x_coords,
    const void* y_coords,
    uint32_t count,
    uint32_t threshold,
    int* is_valid,
    uint32_t curve_type,
    cudaStream_t stream
) {
    Scalar256* d_x_coords;
    Scalar256* d_y_coords;
    int* d_is_valid;
    
    cudaMalloc(&d_x_coords, count * sizeof(Scalar256));
    cudaMalloc(&d_y_coords, count * sizeof(Scalar256));
    cudaMalloc(&d_is_valid, sizeof(int));
    
    cudaMemcpyAsync(d_x_coords, x_coords, count * sizeof(Scalar256),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_y_coords, y_coords, count * sizeof(Scalar256),
                    cudaMemcpyHostToDevice, stream);
    
    size_t shared_size = count * sizeof(Scalar256);
    shamir_verify_shares_kernel<<<1, 1, shared_size, stream>>>(
        d_x_coords, d_y_coords, count, threshold, d_is_valid, (CurveType)curve_type
    );
    
    cudaMemcpyAsync(is_valid, d_is_valid, sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    
    cudaFree(d_x_coords);
    cudaFree(d_y_coords);
    cudaFree(d_is_valid);
    
    return cudaGetLastError();
}

// Helper: evaluate polynomial at multiple points
int lux_cuda_shamir_poly_eval(
    const void* coefficients,   // Scalar256[degree]
    const void* eval_points,    // Scalar256[num_points]
    void* results,              // Scalar256[num_points] output
    uint32_t degree,
    uint32_t num_points,
    uint32_t curve_type,
    cudaStream_t stream
) {
    // For each point, evaluate the polynomial
    Scalar256* d_coeffs;
    Scalar256* d_points;
    Scalar256* d_results;
    
    cudaMalloc(&d_coeffs, degree * sizeof(Scalar256));
    cudaMalloc(&d_points, num_points * sizeof(Scalar256));
    cudaMalloc(&d_results, num_points * sizeof(Scalar256));
    
    cudaMemcpyAsync(d_coeffs, coefficients, degree * sizeof(Scalar256),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_points, eval_points, num_points * sizeof(Scalar256),
                    cudaMemcpyHostToDevice, stream);
    
    // Simple kernel for polynomial evaluation
    auto poly_eval_kernel = [=] __device__ (uint32_t idx) {
        if (idx >= num_points) return;
        const Scalar256* points = (const Scalar256*)d_points;
        const Scalar256* coeffs = (const Scalar256*)d_coeffs;
        Scalar256* output = (Scalar256*)d_results;
        output[idx] = poly_eval(coeffs, degree, points[idx], (CurveType)curve_type);
    };
    
    // Launch with simple grid
    dim3 block(256);
    dim3 grid((num_points + block.x - 1) / block.x);
    
    // Use generate_shares kernel as it does polynomial evaluation
    shamir_generate_shares_kernel<<<grid, block, 0, stream>>>(
        d_coeffs, nullptr, d_results, degree, num_points, 1, (CurveType)curve_type
    );
    
    cudaMemcpyAsync(results, d_results, num_points * sizeof(Scalar256),
                    cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    
    cudaFree(d_coeffs);
    cudaFree(d_points);
    cudaFree(d_results);
    
    return cudaGetLastError();
}

} // extern "C"
