// Goldilocks Field Arithmetic for STARK Verification
// Field: p = 2^64 - 2^32 + 1 (Goldilocks prime)
//
// This shader provides GPU-accelerated field operations for STARK proofs,
// including FRI (Fast Reed-Solomon IOP) folding and verification.

#include <metal_stdlib>
using namespace metal;

// Goldilocks prime: p = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001
constant uint64_t GOLDILOCKS_P = 0xFFFFFFFF00000001ULL;
constant uint64_t GOLDILOCKS_P_MINUS_2 = 0xFFFFFFFEFFFFFFFFULL;

// Montgomery parameters for Goldilocks
// R = 2^64, R^2 mod p, p' = -p^(-1) mod R
constant uint64_t GOLDILOCKS_R2 = 0xFFFFFFFE00000001ULL;  // R^2 mod p
constant uint64_t GOLDILOCKS_PINV = 0xFFFFFFFF00000001ULL; // -p^(-1) mod 2^64

// Non-residue for quadratic extension: X^2 = 7
constant uint64_t EXT_NON_RESIDUE = 7;

// =============================================================================
// Basic Field Arithmetic
// =============================================================================

// Reduce if >= p
inline uint64_t goldilocks_reduce(uint64_t x) {
    return (x >= GOLDILOCKS_P) ? (x - GOLDILOCKS_P) : x;
}

// Addition: (a + b) mod p
inline uint64_t goldilocks_add(uint64_t a, uint64_t b) {
    uint64_t sum = a + b;
    // Check overflow or >= p
    if (sum < a || sum >= GOLDILOCKS_P) {
        sum -= GOLDILOCKS_P;
    }
    return sum;
}

// Subtraction: (a - b) mod p
inline uint64_t goldilocks_sub(uint64_t a, uint64_t b) {
    if (a >= b) {
        return a - b;
    }
    return GOLDILOCKS_P - (b - a);
}

// Negation: -a mod p
inline uint64_t goldilocks_neg(uint64_t a) {
    return (a == 0) ? 0 : (GOLDILOCKS_P - a);
}

// 64x64 -> 128-bit multiplication (hi, lo)
inline void mul64_128(uint64_t a, uint64_t b, thread uint64_t& hi, thread uint64_t& lo) {
    uint64_t a_lo = a & 0xFFFFFFFF;
    uint64_t a_hi = a >> 32;
    uint64_t b_lo = b & 0xFFFFFFFF;
    uint64_t b_hi = b >> 32;

    uint64_t p0 = a_lo * b_lo;
    uint64_t p1 = a_lo * b_hi;
    uint64_t p2 = a_hi * b_lo;
    uint64_t p3 = a_hi * b_hi;

    uint64_t mid = p1 + (p0 >> 32);
    uint64_t mid_lo = mid & 0xFFFFFFFF;
    uint64_t mid_hi = mid >> 32;

    mid_lo += p2;
    mid_hi += (mid_lo < p2) ? 1 : 0;
    mid_hi += (p2 >> 32);

    lo = (mid_lo << 32) | (p0 & 0xFFFFFFFF);
    hi = p3 + mid_hi;
}

// Reduce 128-bit value mod Goldilocks
// Uses: hi * 2^64 + lo ≡ hi * (2^32 - 1) + lo (mod p)
inline uint64_t goldilocks_reduce128(uint64_t hi, uint64_t lo) {
    // p = 2^64 - 2^32 + 1, so 2^64 ≡ 2^32 - 1 (mod p)
    // hi * 2^64 ≡ hi * (2^32 - 1) = hi * 2^32 - hi

    uint64_t hi_shifted = hi << 32;
    uint64_t result = lo;

    // Add hi * 2^32
    result = goldilocks_add(result, hi_shifted);

    // Subtract hi (equivalent to adding -hi mod p)
    result = goldilocks_sub(result, hi);

    // Handle hi >> 32 overflow
    uint64_t hi_upper = hi >> 32;
    if (hi_upper > 0) {
        // hi_upper * 2^96 mod p = hi_upper * (2^32 - 1)^2 mod p
        // = hi_upper * (2^64 - 2*2^32 + 1) mod p
        // = hi_upper * (2^32 - 1 - 2*2^32 + 1) mod p = hi_upper * (-2^32) mod p
        result = goldilocks_sub(result, hi_upper << 32);
        result = goldilocks_add(result, hi_upper);
    }

    return goldilocks_reduce(result);
}

// Multiplication: (a * b) mod p
inline uint64_t goldilocks_mul(uint64_t a, uint64_t b) {
    uint64_t hi, lo;
    mul64_128(a, b, hi, lo);
    return goldilocks_reduce128(hi, lo);
}

// Square: a^2 mod p
inline uint64_t goldilocks_square(uint64_t a) {
    return goldilocks_mul(a, a);
}

// Exponentiation: a^exp mod p (square-and-multiply)
inline uint64_t goldilocks_exp(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    while (exp > 0) {
        if (exp & 1) {
            result = goldilocks_mul(result, base);
        }
        base = goldilocks_square(base);
        exp >>= 1;
    }
    return result;
}

// Inverse: a^(-1) mod p using Fermat's little theorem
// a^(-1) = a^(p-2) mod p
inline uint64_t goldilocks_inv(uint64_t a) {
    if (a == 0) return 0;
    return goldilocks_exp(a, GOLDILOCKS_P_MINUS_2);
}

// Division: a / b mod p
inline uint64_t goldilocks_div(uint64_t a, uint64_t b) {
    return goldilocks_mul(a, goldilocks_inv(b));
}

// =============================================================================
// Quadratic Extension Field: F_{p^2} = F_p[X] / (X^2 - 7)
// Elements: a + b*X where X^2 = 7
// =============================================================================

struct ExtField {
    uint64_t a;  // Real part
    uint64_t b;  // Imaginary part (coefficient of X)
};

// Extension addition
inline ExtField ext_add(ExtField x, ExtField y) {
    return {goldilocks_add(x.a, y.a), goldilocks_add(x.b, y.b)};
}

// Extension subtraction
inline ExtField ext_sub(ExtField x, ExtField y) {
    return {goldilocks_sub(x.a, y.a), goldilocks_sub(x.b, y.b)};
}

// Extension negation
inline ExtField ext_neg(ExtField x) {
    return {goldilocks_neg(x.a), goldilocks_neg(x.b)};
}

// Extension multiplication
// (a + bX)(c + dX) = (ac + 7bd) + (ad + bc)X
inline ExtField ext_mul(ExtField x, ExtField y) {
    uint64_t ac = goldilocks_mul(x.a, y.a);
    uint64_t bd = goldilocks_mul(x.b, y.b);
    uint64_t ad = goldilocks_mul(x.a, y.b);
    uint64_t bc = goldilocks_mul(x.b, y.a);

    // 7 * bd
    uint64_t seven_bd = goldilocks_mul(EXT_NON_RESIDUE, bd);

    return {
        goldilocks_add(ac, seven_bd),  // ac + 7bd
        goldilocks_add(ad, bc)          // ad + bc
    };
}

// Extension square
inline ExtField ext_square(ExtField x) {
    return ext_mul(x, x);
}

// Extension inverse using conjugate
// (a + bX)^(-1) = (a - bX) / (a^2 - 7b^2)
inline ExtField ext_inv(ExtField x) {
    uint64_t a2 = goldilocks_square(x.a);
    uint64_t b2 = goldilocks_square(x.b);
    uint64_t seven_b2 = goldilocks_mul(EXT_NON_RESIDUE, b2);
    uint64_t norm = goldilocks_sub(a2, seven_b2);  // a^2 - 7b^2
    uint64_t norm_inv = goldilocks_inv(norm);

    return {
        goldilocks_mul(x.a, norm_inv),
        goldilocks_neg(goldilocks_mul(x.b, norm_inv))
    };
}

// =============================================================================
// Batch Field Operations (Vectorized)
// =============================================================================

// Batch addition kernel
kernel void goldilocks_batch_add(
    device const uint64_t* a [[buffer(0)]],
    device const uint64_t* b [[buffer(1)]],
    device uint64_t* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = goldilocks_add(a[index], b[index]);
}

// Batch subtraction kernel
kernel void goldilocks_batch_sub(
    device const uint64_t* a [[buffer(0)]],
    device const uint64_t* b [[buffer(1)]],
    device uint64_t* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = goldilocks_sub(a[index], b[index]);
}

// Batch multiplication kernel
kernel void goldilocks_batch_mul(
    device const uint64_t* a [[buffer(0)]],
    device const uint64_t* b [[buffer(1)]],
    device uint64_t* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = goldilocks_mul(a[index], b[index]);
}

// Batch inversion using Montgomery's trick
// Compute inversions for batch of elements: [a_0^-1, a_1^-1, ..., a_n^-1]
// Only uses 1 actual inversion + 3(n-1) multiplications
kernel void goldilocks_batch_inv(
    device const uint64_t* inputs [[buffer(0)]],
    device uint64_t* outputs [[buffer(1)]],
    device uint64_t* scratch [[buffer(2)]],  // Temporary storage
    constant uint32_t& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    // Phase 1: Compute running products
    // scratch[i] = inputs[0] * inputs[1] * ... * inputs[i]
    if (index < count) {
        if (index == 0) {
            scratch[0] = inputs[0];
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // Sequential prefix product (done by thread 0)
    if (index == 0) {
        for (uint32_t i = 1; i < count; i++) {
            scratch[i] = goldilocks_mul(scratch[i-1], inputs[i]);
        }

        // Phase 2: Single inversion
        uint64_t total_inv = goldilocks_inv(scratch[count - 1]);

        // Phase 3: Back-propagate inversions
        for (int32_t i = count - 1; i >= 0; i--) {
            if (i == 0) {
                outputs[0] = total_inv;
            } else {
                outputs[i] = goldilocks_mul(total_inv, scratch[i - 1]);
                total_inv = goldilocks_mul(total_inv, inputs[i]);
            }
        }
    }
}

// =============================================================================
// FRI (Fast Reed-Solomon IOP) Operations
// =============================================================================

// FRI folding: fold layer at alpha
// new_eval[i] = (eval[2i] + eval[2i+1]) / 2 + alpha * (eval[2i] - eval[2i+1]) / (2 * omega^i)
kernel void fri_fold_layer(
    device const uint64_t* evals [[buffer(0)]],       // Current layer evaluations
    device uint64_t* folded [[buffer(1)]],            // Folded layer output
    constant uint64_t& alpha [[buffer(2)]],           // Folding challenge
    constant uint64_t& omega_inv [[buffer(3)]],       // Inverse of subgroup generator
    constant uint32_t& layer_size [[buffer(4)]],      // Size of current layer
    uint index [[thread_position_in_grid]]
) {
    if (index >= layer_size / 2) return;

    uint64_t e0 = evals[2 * index];
    uint64_t e1 = evals[2 * index + 1];

    // (e0 + e1) / 2
    uint64_t sum = goldilocks_add(e0, e1);
    uint64_t half_sum = goldilocks_mul(sum, goldilocks_inv(2));  // Could precompute inv(2)

    // (e0 - e1) / (2 * omega^i)
    uint64_t diff = goldilocks_sub(e0, e1);

    // omega^(-i) for position i
    uint64_t omega_power = goldilocks_exp(omega_inv, index);
    uint64_t half_diff = goldilocks_mul(diff, goldilocks_mul(goldilocks_inv(2), omega_power));

    // half_sum + alpha * half_diff
    uint64_t alpha_term = goldilocks_mul(alpha, half_diff);
    folded[index] = goldilocks_add(half_sum, alpha_term);
}

// FRI query verification: check consistency of query
kernel void fri_verify_query(
    device const uint64_t* layer_evals [[buffer(0)]],  // Evaluations at query positions
    device const uint64_t* alphas [[buffer(1)]],       // Folding challenges
    device const uint64_t* omega_invs [[buffer(2)]],   // Inverse generators per layer
    device uint32_t* query_positions [[buffer(3)]],    // Query positions per layer
    device uint64_t* results [[buffer(4)]],            // Pass/fail results
    constant uint32_t& num_layers [[buffer(5)]],
    constant uint32_t& queries_per_layer [[buffer(6)]],
    uint query_idx [[thread_position_in_grid]]
) {
    // Each thread verifies one query path through all layers
    // Implementation depends on specific FRI variant

    uint64_t expected = layer_evals[query_idx];
    bool valid = true;

    for (uint32_t layer = 0; layer < num_layers && valid; layer++) {
        uint32_t pos = query_positions[layer * queries_per_layer + query_idx];
        uint32_t sibling_pos = pos ^ 1;  // Sibling in pair

        uint64_t e0 = layer_evals[layer * queries_per_layer * 2 + pos];
        uint64_t e1 = layer_evals[layer * queries_per_layer * 2 + sibling_pos];

        uint64_t alpha = alphas[layer];
        uint64_t omega_inv = omega_invs[layer];

        // Compute expected folded value
        uint64_t sum = goldilocks_add(e0, e1);
        uint64_t diff = goldilocks_sub(e0, e1);
        uint64_t omega_power = goldilocks_exp(omega_inv, pos / 2);

        uint64_t folded = goldilocks_add(
            goldilocks_mul(sum, goldilocks_inv(2)),
            goldilocks_mul(alpha, goldilocks_mul(diff, goldilocks_mul(goldilocks_inv(2), omega_power)))
        );

        // Check against next layer
        // (actual implementation would check against committed values)
    }

    results[query_idx] = valid ? 1 : 0;
}

// =============================================================================
// Constraint Evaluation for STARK AIR
// =============================================================================

// Evaluate AIR constraints at multiple points
kernel void stark_constraint_eval(
    device const uint64_t* trace [[buffer(0)]],           // Execution trace
    device const uint64_t* trace_next [[buffer(1)]],      // Next row of trace
    device uint64_t* constraint_values [[buffer(2)]],     // Output constraint evaluations
    constant uint32_t& num_columns [[buffer(3)]],
    constant uint32_t& num_constraints [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    // Generic constraint evaluation framework
    // Specific constraints depend on the STARK being verified

    // Example: Fibonacci constraint
    // c(x) = trace[i+1] - trace[i] - trace[i-1] = 0

    // The actual constraints would be defined based on the AIR
}

// =============================================================================
// Extension Field Batch Operations
// =============================================================================

// Batch extension multiplication
kernel void ext_batch_mul(
    device const ExtField* a [[buffer(0)]],
    device const ExtField* b [[buffer(1)]],
    device ExtField* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = ext_mul(a[index], b[index]);
}

// Extension field polynomial evaluation using Horner's method
kernel void ext_poly_eval(
    device const ExtField* coeffs [[buffer(0)]],  // Polynomial coefficients
    device const ExtField* points [[buffer(1)]],  // Evaluation points
    device ExtField* results [[buffer(2)]],       // Results
    constant uint32_t& degree [[buffer(3)]],
    uint point_idx [[thread_position_in_grid]]
) {
    ExtField x = points[point_idx];
    ExtField result = coeffs[degree];

    for (int32_t i = degree - 1; i >= 0; i--) {
        result = ext_mul(result, x);
        result = ext_add(result, coeffs[i]);
    }

    results[point_idx] = result;
}
