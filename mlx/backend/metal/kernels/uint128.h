// Copyright © 2024 Lux Partners Limited
// 128-bit and 256-bit unsigned integer support for Metal
// Used for lattice cryptography modular arithmetic

#pragma once
#include <metal_stdlib>

using namespace metal;

// ============================================================================
// uint128 - 128-bit unsigned integer using two 64-bit values
// ============================================================================

struct uint128 {
    ulong lo;  // Low 64 bits
    ulong hi;  // High 64 bits

    // Constructors
    constexpr uint128() : lo(0), hi(0) {}
    constexpr uint128(ulong v) : lo(v), hi(0) {}
    constexpr uint128(ulong h, ulong l) : lo(l), hi(h) {}

    // Check if zero
    bool is_zero() const { return lo == 0 && hi == 0; }

    // Comparison (pass by value for Metal compatibility)
    bool operator<(uint128 rhs) const {
        return hi < rhs.hi || (hi == rhs.hi && lo < rhs.lo);
    }

    bool operator>=(uint128 rhs) const {
        return !(*this < rhs);
    }

    bool operator==(uint128 rhs) const {
        return lo == rhs.lo && hi == rhs.hi;
    }

    // Addition with carry
    uint128 operator+(uint128 rhs) const {
        ulong sum_lo = lo + rhs.lo;
        ulong carry = (sum_lo < lo) ? 1 : 0;
        ulong sum_hi = hi + rhs.hi + carry;
        return uint128(sum_hi, sum_lo);
    }

    // Subtraction with borrow
    uint128 operator-(uint128 rhs) const {
        ulong diff_lo = lo - rhs.lo;
        ulong borrow = (lo < rhs.lo) ? 1 : 0;
        ulong diff_hi = hi - rhs.hi - borrow;
        return uint128(diff_hi, diff_lo);
    }

    // Left shift
    uint128 operator<<(int n) const {
        if (n >= 128) return uint128();
        if (n >= 64) return uint128(lo << (n - 64), 0);
        if (n == 0) return *this;
        return uint128((hi << n) | (lo >> (64 - n)), lo << n);
    }

    // Right shift
    uint128 operator>>(int n) const {
        if (n >= 128) return uint128();
        if (n >= 64) return uint128(0, hi >> (n - 64));
        if (n == 0) return *this;
        return uint128(hi >> n, (lo >> n) | (hi << (64 - n)));
    }

    // Bitwise OR
    uint128 operator|(uint128 rhs) const {
        return uint128(hi | rhs.hi, lo | rhs.lo);
    }

    // Bitwise AND
    uint128 operator&(uint128 rhs) const {
        return uint128(hi & rhs.hi, lo & rhs.lo);
    }
};

// ============================================================================
// 64x64 -> 128 bit multiplication
// ============================================================================

METAL_FUNC uint128 mul64x64(ulong a, ulong b) {
    // Split into 32-bit parts
    ulong a_lo = a & 0xFFFFFFFF;
    ulong a_hi = a >> 32;
    ulong b_lo = b & 0xFFFFFFFF;
    ulong b_hi = b >> 32;

    // Partial products
    ulong p00 = a_lo * b_lo;
    ulong p01 = a_lo * b_hi;
    ulong p10 = a_hi * b_lo;
    ulong p11 = a_hi * b_hi;

    // Combine with carries
    ulong mid = p01 + p10;
    ulong mid_carry = (mid < p01) ? 1UL << 32 : 0;

    ulong lo = p00 + (mid << 32);
    ulong lo_carry = (lo < p00) ? 1 : 0;

    ulong hi = p11 + (mid >> 32) + mid_carry + lo_carry;

    return uint128(hi, lo);
}

// Alternative using mulhi intrinsic (more efficient on Apple Silicon)
METAL_FUNC uint128 mul64x64_fast(ulong a, ulong b) {
    return uint128(mulhi(a, b), a * b);
}

// ============================================================================
// Modular reduction: compute a mod Q where a is uint128 and Q is uint64
// ============================================================================

// Simple mod for 128-bit / 64-bit
METAL_FUNC ulong mod_128_64(uint128 a, ulong Q) {
    if (a.hi == 0) {
        return a.lo % Q;
    }

    // 2^64 mod Q
    ulong pow64_mod_q;
    {
        ulong p32 = (1UL << 32) % Q;
        ulong prod = p32 * p32;  // May overflow if Q > 2^32
        if (Q <= (1UL << 32)) {
            pow64_mod_q = prod % Q;
        } else {
            // For large Q, compute more carefully
            uint128 p = mul64x64_fast(p32, p32);
            // Recursive call - will terminate as p.hi will be smaller
            pow64_mod_q = mod_128_64(p, Q);
        }
    }

    ulong lo_mod = a.lo % Q;
    ulong hi_mod = a.hi % Q;

    // hi_mod * pow64_mod_q
    ulong hi_contrib;
    {
        uint128 temp = mul64x64_fast(hi_mod, pow64_mod_q);
        hi_contrib = mod_128_64(temp, Q);
    }

    ulong result = lo_mod + hi_contrib;
    if (result >= Q || result < lo_mod) result -= Q;

    return result;
}

// ============================================================================
// Modular multiplication: (a * b) mod Q where a, b < Q
// ============================================================================

METAL_FUNC ulong mod_mul_128(ulong a, ulong b, ulong Q) {
    uint128 prod = mul64x64_fast(a, b);
    return mod_128_64(prod, Q);
}

// ============================================================================
// uint256 - 256-bit unsigned integer using four 64-bit values
// ============================================================================

struct uint256 {
    ulong w0;  // Bits 0-63 (least significant)
    ulong w1;  // Bits 64-127
    ulong w2;  // Bits 128-191
    ulong w3;  // Bits 192-255 (most significant)

    // Constructors
    constexpr uint256() : w0(0), w1(0), w2(0), w3(0) {}
    constexpr uint256(ulong v) : w0(v), w1(0), w2(0), w3(0) {}
    constexpr uint256(ulong a, ulong b, ulong c, ulong d) : w0(d), w1(c), w2(b), w3(a) {}

    // From uint128
    constexpr uint256(uint128 v) : w0(v.lo), w1(v.hi), w2(0), w3(0) {}

    // Check if zero
    bool is_zero() const { return w0 == 0 && w1 == 0 && w2 == 0 && w3 == 0; }

    // Comparison (pass by value for Metal compatibility)
    bool operator<(uint256 rhs) const {
        if (w3 != rhs.w3) return w3 < rhs.w3;
        if (w2 != rhs.w2) return w2 < rhs.w2;
        if (w1 != rhs.w1) return w1 < rhs.w1;
        return w0 < rhs.w0;
    }

    bool operator>=(uint256 rhs) const {
        return !(*this < rhs);
    }

    // Addition
    uint256 operator+(uint256 rhs) const {
        ulong carry = 0;

        ulong r0 = w0 + rhs.w0;
        carry = (r0 < w0) ? 1 : 0;

        ulong r1 = w1 + rhs.w1 + carry;
        carry = (r1 < w1 || (carry && r1 == w1)) ? 1 : 0;

        ulong r2 = w2 + rhs.w2 + carry;
        carry = (r2 < w2 || (carry && r2 == w2)) ? 1 : 0;

        ulong r3 = w3 + rhs.w3 + carry;

        uint256 result;
        result.w0 = r0;
        result.w1 = r1;
        result.w2 = r2;
        result.w3 = r3;
        return result;
    }

    // Subtraction
    uint256 operator-(uint256 rhs) const {
        ulong borrow = 0;

        ulong r0 = w0 - rhs.w0;
        borrow = (w0 < rhs.w0) ? 1 : 0;

        ulong r1 = w1 - rhs.w1 - borrow;
        borrow = (w1 < rhs.w1 + borrow) ? 1 : 0;

        ulong r2 = w2 - rhs.w2 - borrow;
        borrow = (w2 < rhs.w2 + borrow) ? 1 : 0;

        ulong r3 = w3 - rhs.w3 - borrow;

        uint256 result;
        result.w0 = r0;
        result.w1 = r1;
        result.w2 = r2;
        result.w3 = r3;
        return result;
    }

    // Left shift by n bits (0 <= n < 256)
    uint256 operator<<(int n) const {
        if (n >= 256) return uint256();
        if (n == 0) return *this;

        int words = n / 64;
        int bits = n % 64;

        ulong v[4] = {w0, w1, w2, w3};
        ulong r[4] = {0, 0, 0, 0};

        for (int i = 0; i < 4 - words; i++) {
            r[i + words] |= v[i] << bits;
            if (bits > 0 && i + words + 1 < 4) {
                r[i + words + 1] |= v[i] >> (64 - bits);
            }
        }

        uint256 result;
        result.w0 = r[0];
        result.w1 = r[1];
        result.w2 = r[2];
        result.w3 = r[3];
        return result;
    }

    // Right shift
    uint256 operator>>(int n) const {
        if (n >= 256) return uint256();
        if (n == 0) return *this;

        int words = n / 64;
        int bits = n % 64;

        ulong v[4] = {w0, w1, w2, w3};
        ulong r[4] = {0, 0, 0, 0};

        for (int i = words; i < 4; i++) {
            r[i - words] |= v[i] >> bits;
            if (bits > 0 && i - words > 0) {
                r[i - words - 1] |= v[i] << (64 - bits);
            }
        }

        uint256 result;
        result.w0 = r[0];
        result.w1 = r[1];
        result.w2 = r[2];
        result.w3 = r[3];
        return result;
    }
};

// ============================================================================
// 128x128 -> 256 bit multiplication
// ============================================================================

METAL_FUNC uint256 mul128x128(uint128 a, uint128 b) {
    // (a.hi * 2^64 + a.lo) * (b.hi * 2^64 + b.lo)
    // = a.hi * b.hi * 2^128 + (a.hi * b.lo + a.lo * b.hi) * 2^64 + a.lo * b.lo

    uint128 p00 = mul64x64_fast(a.lo, b.lo);   // a.lo * b.lo
    uint128 p01 = mul64x64_fast(a.lo, b.hi);   // a.lo * b.hi
    uint128 p10 = mul64x64_fast(a.hi, b.lo);   // a.hi * b.lo
    uint128 p11 = mul64x64_fast(a.hi, b.hi);   // a.hi * b.hi

    // Combine: result = p11 * 2^128 + (p01 + p10) * 2^64 + p00
    uint256 result;
    result.w0 = p00.lo;
    result.w1 = p00.hi;
    result.w2 = p11.lo;
    result.w3 = p11.hi;

    // Add (p01 + p10) << 64
    uint128 mid = p01 + p10;
    ulong carry = (mid < p01) ? 1 : 0;  // Carry from p01 + p10

    // Add mid.lo to w1
    ulong old_w1 = result.w1;
    result.w1 += mid.lo;
    if (result.w1 < old_w1) {
        result.w2++;
        if (result.w2 == 0) result.w3++;
    }

    // Add mid.hi + carry to w2
    ulong old_w2 = result.w2;
    result.w2 += mid.hi + carry;
    if (result.w2 < old_w2 || (carry && result.w2 == old_w2)) {
        result.w3++;
    }

    return result;
}
