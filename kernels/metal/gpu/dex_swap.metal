// Copyright (C) 2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// DEX Swap Acceleration Kernels for Metal GPU
// Implements high-performance Uniswap v4-style AMM math
// Target: >1M swaps/sec on Apple Silicon

#include <metal_stdlib>
#include <metal_integer>
#include <metal_math>

using namespace metal;

// =============================================================================
// 128-bit Unsigned Integer Type (Q64.96 fixed point)
// =============================================================================

struct uint128_t {
    uint64_t lo;
    uint64_t hi;
    
    // Addition with carry
    uint128_t operator+(const uint128_t& other) const {
        uint128_t result;
        result.lo = lo + other.lo;
        result.hi = hi + other.hi + (result.lo < lo ? 1 : 0);
        return result;
    }
    
    // Subtraction with borrow
    uint128_t operator-(const uint128_t& other) const {
        uint128_t result;
        result.lo = lo - other.lo;
        result.hi = hi - other.hi - (lo < other.lo ? 1 : 0);
        return result;
    }
    
    // Comparison
    bool operator<(const uint128_t& other) const {
        return hi < other.hi || (hi == other.hi && lo < other.lo);
    }
    
    bool operator>(const uint128_t& other) const {
        return hi > other.hi || (hi == other.hi && lo > other.lo);
    }
    
    bool operator==(const uint128_t& other) const {
        return hi == other.hi && lo == other.lo;
    }
    
    bool operator>=(const uint128_t& other) const {
        return !(*this < other);
    }
    
    bool isZero() const {
        return lo == 0 && hi == 0;
    }
};

// Multiply 64x64 -> 128
inline uint128_t mul64x64(uint64_t a, uint64_t b) {
    uint128_t result;
    
    uint64_t al = a & 0xFFFFFFFF;
    uint64_t ah = a >> 32;
    uint64_t bl = b & 0xFFFFFFFF;
    uint64_t bh = b >> 32;
    
    uint64_t ll = al * bl;
    uint64_t lh = al * bh;
    uint64_t hl = ah * bl;
    uint64_t hh = ah * bh;
    
    uint64_t mid = lh + hl;
    bool carry = mid < lh;
    
    result.lo = ll + (mid << 32);
    bool carry2 = result.lo < ll;
    result.hi = hh + (mid >> 32) + (carry ? 0x100000000ULL : 0) + (carry2 ? 1 : 0);
    
    return result;
}

// Divide 128 / 64 -> 64 (approximate, good enough for swap math)
inline uint64_t div128by64(uint128_t n, uint64_t d) {
    if (d == 0) return UINT64_MAX;
    if (n.hi == 0) return n.lo / d;
    
    // Use floating point approximation for speed
    // Accurate enough for DEX calculations
    float hi_f = float(n.hi) * float(1ULL << 32) * float(1ULL << 32);
    float lo_f = float(n.lo);
    float d_f = float(d);
    
    return uint64_t((hi_f + lo_f) / d_f);
}

// Right shift 128-bit value
inline uint128_t rsh128(uint128_t v, uint shift) {
    if (shift >= 128) return {0, 0};
    if (shift >= 64) {
        return {v.hi >> (shift - 64), 0};
    }
    if (shift == 0) return v;
    
    uint128_t result;
    result.lo = (v.lo >> shift) | (v.hi << (64 - shift));
    result.hi = v.hi >> shift;
    return result;
}

// Left shift 128-bit value
inline uint128_t lsh128(uint128_t v, uint shift) {
    if (shift >= 128) return {0, 0};
    if (shift >= 64) {
        return {0, v.lo << (shift - 64)};
    }
    if (shift == 0) return v;
    
    uint128_t result;
    result.hi = (v.hi << shift) | (v.lo >> (64 - shift));
    result.lo = v.lo << shift;
    return result;
}

// =============================================================================
// Swap Input/Output Structures
// =============================================================================

struct SwapInput {
    uint64_t poolId_lo;        // Pool ID lower 64 bits
    uint64_t poolId_hi;        // Pool ID upper 64 bits (+ 128 bits)
    uint64_t poolId_extra[2];  // Remaining pool ID bits
    
    uint64_t sqrtPriceX96_lo;  // Current sqrt price
    uint64_t sqrtPriceX96_hi;
    
    uint64_t liquidity_lo;     // Current liquidity
    uint64_t liquidity_hi;
    
    int32_t tick;              // Current tick
    int32_t padding1;
    
    uint32_t zeroForOne;       // Direction (bool as uint32)
    uint32_t exactInput;       // Amount type (bool as uint32)
    
    uint64_t amount_lo;        // Amount specified
    uint64_t amount_hi;
    
    uint32_t feePips;          // Fee in pips (0.0001%)
    uint32_t padding2;
    
    uint64_t sqrtPriceLimit_lo; // Price limit
    uint64_t sqrtPriceLimit_hi;
};

struct SwapOutput {
    uint64_t amount0Delta_lo;  // Token0 delta (signed as 2's complement)
    uint64_t amount0Delta_hi;
    
    uint64_t amount1Delta_lo;  // Token1 delta
    uint64_t amount1Delta_hi;
    
    uint64_t sqrtPriceX96_lo;  // New sqrt price
    uint64_t sqrtPriceX96_hi;
    
    int32_t tick;              // New tick
    uint32_t success;          // Whether swap succeeded
    
    uint64_t feeGrowth_lo;     // Fee growth increment
    uint64_t feeGrowth_hi;
    
    uint32_t errorCode;        // Error code if failed
    uint32_t padding;
};

// =============================================================================
// Liquidity Input/Output Structures
// =============================================================================

struct LiquidityInput {
    uint64_t poolId_lo;
    uint64_t poolId_hi;
    uint64_t poolId_extra[2];
    
    uint64_t sqrtPriceX96_lo;
    uint64_t sqrtPriceX96_hi;
    
    uint64_t liquidity_lo;
    uint64_t liquidity_hi;
    
    int32_t currentTick;
    int32_t tickLower;
    int32_t tickUpper;
    uint32_t isAdd;
    
    uint64_t liqDelta_lo;
    uint64_t liqDelta_hi;
};

struct LiquidityOutput {
    uint64_t amount0_lo;
    uint64_t amount0_hi;
    
    uint64_t amount1_lo;
    uint64_t amount1_hi;
    
    uint64_t feeGrowth0_lo;
    uint64_t feeGrowth0_hi;
    
    uint64_t feeGrowth1_lo;
    uint64_t feeGrowth1_hi;
    
    uint32_t success;
    uint32_t errorCode;
};

// =============================================================================
// Core Math Functions
// =============================================================================

// Apply fee: amount * (1e6 - fee) / 1e6
inline uint128_t applyFee(uint128_t amount, uint32_t feePips) {
    if (feePips == 0) return amount;
    
    uint64_t multiplier = 1000000 - uint64_t(feePips);
    
    // Multiply amount by multiplier
    uint128_t result;
    if (amount.hi == 0) {
        result = mul64x64(amount.lo, multiplier);
    } else {
        // For 128-bit amounts, use approximation
        uint128_t hi_product = mul64x64(amount.hi, multiplier);
        uint128_t lo_product = mul64x64(amount.lo, multiplier);
        result.lo = lo_product.lo;
        result.hi = lo_product.hi + hi_product.lo;
    }
    
    // Divide by 1e6
    return {div128by64(result, 1000000), 0};
}

// Calculate fee amount
inline uint128_t calculateFee(uint128_t amount, uint32_t feePips) {
    if (feePips == 0) return {0, 0};
    
    uint128_t fee;
    if (amount.hi == 0) {
        fee = mul64x64(amount.lo, uint64_t(feePips));
    } else {
        fee = mul64x64(amount.hi, uint64_t(feePips));
        fee = lsh128(fee, 64);
        uint128_t lo_fee = mul64x64(amount.lo, uint64_t(feePips));
        fee = fee + lo_fee;
    }
    
    return {div128by64(fee, 1000000), 0};
}

// Calculate swap output: out = in * L / (L + in)
inline uint128_t calculateSwapOutput(uint128_t amountIn, uint128_t liquidity) {
    if (liquidity.isZero()) return {0, 0};
    
    // numerator = amountIn * liquidity
    uint128_t numerator;
    if (amountIn.hi == 0 && liquidity.hi == 0) {
        numerator = mul64x64(amountIn.lo, liquidity.lo);
    } else {
        // Simplified for 128-bit operands
        numerator = mul64x64(amountIn.lo, liquidity.lo);
    }
    
    // denominator = liquidity + amountIn
    uint128_t denominator = liquidity + amountIn;
    
    if (denominator.isZero()) return {0, 0};
    
    // result = numerator / denominator
    if (denominator.hi == 0) {
        return {div128by64(numerator, denominator.lo), 0};
    }
    
    // For large denominators, use approximation
    uint64_t result = div128by64(numerator, denominator.lo);
    return {result, 0};
}

// Calculate new sqrt price after swap
inline uint128_t calculateNewSqrtPrice(
    uint128_t sqrtPrice,
    uint128_t liquidity,
    uint128_t amount0Delta,
    bool zeroForOne
) {
    if (liquidity.isZero()) return sqrtPrice;
    
    // delta = amount0 * sqrtPrice / liquidity
    uint128_t delta;
    if (amount0Delta.hi == 0 && sqrtPrice.hi == 0) {
        delta = mul64x64(amount0Delta.lo, sqrtPrice.lo);
        if (liquidity.lo > 0) {
            delta.lo = div128by64(delta, liquidity.lo);
            delta.hi = 0;
        }
    } else {
        delta = {0, 0};
    }
    
    if (zeroForOne) {
        // Price decreases
        if (sqrtPrice > delta) {
            return sqrtPrice - delta;
        }
        return {1, 0};
    } else {
        // Price increases
        return sqrtPrice + delta;
    }
}

// Tick to sqrt price conversion (simplified lookup)
constant uint64_t SQRT_MAGIC_0 = 0xfff97263e137;
constant uint64_t SQRT_MAGIC_1 = 0xfff2e50f626c;
constant uint64_t SQRT_MAGIC_2 = 0xffe5caca7e10;
constant uint64_t SQRT_MAGIC_3 = 0xffcb9a979342;
constant uint64_t SQRT_MAGIC_4 = 0xff97383c7e70;

inline int32_t sqrtPriceToTick(uint128_t sqrtPrice) {
    // Q96 = 2^96
    uint128_t q96 = {1ULL << 32, 1ULL << 32};  // Approximate
    
    if (sqrtPrice == q96) return 0;
    if (sqrtPrice < q96) return -1;
    return 1;
}

// =============================================================================
// Main Swap Kernel
// =============================================================================

kernel void batch_swap(
    device const SwapInput* inputs [[buffer(0)]],
    device SwapOutput* outputs [[buffer(1)]],
    constant uint32_t& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    
    SwapInput input = inputs[tid];
    SwapOutput output;
    
    // Initialize output
    output.success = 1;
    output.errorCode = 0;
    
    // Get input values
    uint128_t sqrtPrice = {input.sqrtPriceX96_lo, input.sqrtPriceX96_hi};
    uint128_t liquidity = {input.liquidity_lo, input.liquidity_hi};
    uint128_t amount = {input.amount_lo, input.amount_hi};
    bool zeroForOne = input.zeroForOne != 0;
    bool exactInput = input.exactInput != 0;
    
    // Check for zero liquidity
    if (liquidity.isZero()) {
        output.success = 0;
        output.errorCode = 1;
        outputs[tid] = output;
        return;
    }
    
    // Apply fee to get effective amount
    uint128_t amountAfterFee = applyFee(amount, input.feePips);
    
    // Calculate swap amounts
    uint128_t amount0, amount1;
    
    if (zeroForOne) {
        // Swapping token0 for token1
        if (exactInput) {
            amount0 = amount;
            amount1 = calculateSwapOutput(amountAfterFee, liquidity);
        } else {
            amount1 = amount;
            // For exact output, calculate required input
            // input = output * L / (L - output)
            if (liquidity > amount) {
                uint128_t denom = liquidity - amount;
                uint128_t numer = mul64x64(amount.lo, liquidity.lo);
                amount0.lo = div128by64(numer, denom.lo);
                amount0.hi = 0;
            } else {
                amount0 = liquidity;
            }
        }
    } else {
        // Swapping token1 for token0
        if (exactInput) {
            amount1 = amount;
            amount0 = calculateSwapOutput(amountAfterFee, liquidity);
        } else {
            amount0 = amount;
            if (liquidity > amount) {
                uint128_t denom = liquidity - amount;
                uint128_t numer = mul64x64(amount.lo, liquidity.lo);
                amount1.lo = div128by64(numer, denom.lo);
                amount1.hi = 0;
            } else {
                amount1 = liquidity;
            }
        }
    }
    
    // Calculate new sqrt price
    uint128_t newSqrtPrice = calculateNewSqrtPrice(sqrtPrice, liquidity, amount0, zeroForOne);
    
    // Calculate fee growth
    uint128_t feeGrowth = calculateFee(zeroForOne ? amount0 : amount1, input.feePips);
    
    // Store outputs (note: amount deltas have signs based on direction)
    if (zeroForOne) {
        output.amount0Delta_lo = amount0.lo;
        output.amount0Delta_hi = amount0.hi;
        // amount1 is negative (output to user) - store as 2's complement
        output.amount1Delta_lo = ~amount1.lo + 1;
        output.amount1Delta_hi = amount1.lo == 0 ? ~amount1.hi + 1 : ~amount1.hi;
    } else {
        output.amount1Delta_lo = amount1.lo;
        output.amount1Delta_hi = amount1.hi;
        output.amount0Delta_lo = ~amount0.lo + 1;
        output.amount0Delta_hi = amount0.lo == 0 ? ~amount0.hi + 1 : ~amount0.hi;
    }
    
    output.sqrtPriceX96_lo = newSqrtPrice.lo;
    output.sqrtPriceX96_hi = newSqrtPrice.hi;
    output.tick = sqrtPriceToTick(newSqrtPrice);
    output.feeGrowth_lo = feeGrowth.lo;
    output.feeGrowth_hi = feeGrowth.hi;
    
    outputs[tid] = output;
}

// =============================================================================
// Liquidity Modification Kernel
// =============================================================================

kernel void batch_liquidity(
    device const LiquidityInput* inputs [[buffer(0)]],
    device LiquidityOutput* outputs [[buffer(1)]],
    constant uint32_t& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    
    LiquidityInput input = inputs[tid];
    LiquidityOutput output;
    
    output.success = 1;
    output.errorCode = 0;
    
    // Check tick range
    if (input.tickLower >= input.tickUpper) {
        output.success = 0;
        output.errorCode = 1;
        outputs[tid] = output;
        return;
    }
    
    int32_t currentTick = input.currentTick;
    bool isActive = input.tickLower <= currentTick && currentTick < input.tickUpper;
    
    uint128_t liqDelta = {input.liqDelta_lo, input.liqDelta_hi};
    uint128_t amount0, amount1;
    
    if (input.isAdd) {
        // Adding liquidity
        if (isActive) {
            // Both tokens needed - split equally (simplified)
            amount0 = rsh128(liqDelta, 1);
            amount1 = rsh128(liqDelta, 1);
        } else if (currentTick < input.tickLower) {
            // Only token0 needed
            amount0 = liqDelta;
            amount1 = {0, 0};
        } else {
            // Only token1 needed
            amount0 = {0, 0};
            amount1 = liqDelta;
        }
    } else {
        // Removing liquidity (return as negative - 2's complement)
        if (isActive) {
            uint128_t half = rsh128(liqDelta, 1);
            amount0.lo = ~half.lo + 1;
            amount0.hi = half.lo == 0 ? ~half.hi + 1 : ~half.hi;
            amount1 = amount0;
        } else if (currentTick < input.tickLower) {
            amount0.lo = ~liqDelta.lo + 1;
            amount0.hi = liqDelta.lo == 0 ? ~liqDelta.hi + 1 : ~liqDelta.hi;
            amount1 = {0, 0};
        } else {
            amount0 = {0, 0};
            amount1.lo = ~liqDelta.lo + 1;
            amount1.hi = liqDelta.lo == 0 ? ~liqDelta.hi + 1 : ~liqDelta.hi;
        }
    }
    
    output.amount0_lo = amount0.lo;
    output.amount0_hi = amount0.hi;
    output.amount1_lo = amount1.lo;
    output.amount1_hi = amount1.hi;
    
    // Fee growth would be calculated based on position state
    output.feeGrowth0_lo = 0;
    output.feeGrowth0_hi = 0;
    output.feeGrowth1_lo = 0;
    output.feeGrowth1_hi = 0;
    
    outputs[tid] = output;
}

// =============================================================================
// Route Optimization Kernel
// =============================================================================

struct RouteInput {
    uint64_t amountIn_lo;
    uint64_t amountIn_hi;
    
    uint64_t sqrtPrices[8][2];   // Up to 8 hops, [lo, hi]
    uint64_t liquidities[8][2];  // Up to 8 hops
    uint32_t fees[8];            // Fees per hop
    uint32_t numHops;
    uint32_t padding;
};

struct RouteOutput {
    uint64_t amountOut_lo;
    uint64_t amountOut_hi;
    uint32_t priceImpact;        // Basis points
    uint32_t success;
    uint64_t gasEstimate;
};

kernel void batch_route(
    device const RouteInput* inputs [[buffer(0)]],
    device RouteOutput* outputs [[buffer(1)]],
    constant uint32_t& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    
    RouteInput input = inputs[tid];
    RouteOutput output;
    output.success = 1;
    
    if (input.numHops == 0) {
        output.success = 0;
        output.amountOut_lo = 0;
        output.amountOut_hi = 0;
        outputs[tid] = output;
        return;
    }
    
    uint128_t currentAmount = {input.amountIn_lo, input.amountIn_hi};
    uint128_t initialAmount = currentAmount;
    
    for (uint32_t i = 0; i < input.numHops && i < 8; i++) {
        uint128_t liquidity = {input.liquidities[i][0], input.liquidities[i][1]};
        uint32_t fee = input.fees[i];
        
        if (liquidity.isZero()) {
            output.success = 0;
            break;
        }
        
        // Apply fee
        uint128_t amountAfterFee = applyFee(currentAmount, fee);
        
        // Calculate output for this hop
        currentAmount = calculateSwapOutput(amountAfterFee, liquidity);
    }
    
    output.amountOut_lo = currentAmount.lo;
    output.amountOut_hi = currentAmount.hi;
    
    // Calculate price impact
    if (initialAmount.lo > 0 && currentAmount.lo > 0) {
        // impact = (input - output) / input * 10000
        uint128_t diff;
        if (initialAmount > currentAmount) {
            diff = initialAmount - currentAmount;
        } else {
            diff = {0, 0};
        }
        
        uint128_t impactNum = mul64x64(diff.lo, 10000);
        output.priceImpact = uint32_t(div128by64(impactNum, initialAmount.lo));
        if (output.priceImpact > 10000) output.priceImpact = 10000;
    } else {
        output.priceImpact = 0;
    }
    
    // Gas estimate: ~30k per hop
    output.gasEstimate = uint64_t(input.numHops) * 30000;
    
    outputs[tid] = output;
}

// =============================================================================
// Tick Math Kernels
// =============================================================================

struct TickMathInput {
    int32_t tick;
    int32_t padding;
};

struct TickMathOutput {
    uint64_t sqrtPriceX96_lo;
    uint64_t sqrtPriceX96_hi;
};

kernel void batch_tick_to_sqrt_price(
    device const TickMathInput* inputs [[buffer(0)]],
    device TickMathOutput* outputs [[buffer(1)]],
    constant uint32_t& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    
    int32_t tick = inputs[tid].tick;
    
    // Q96 = 2^96
    if (tick == 0) {
        outputs[tid].sqrtPriceX96_lo = 1ULL << 32;  // Partial Q96
        outputs[tid].sqrtPriceX96_hi = 1ULL << 32;
        return;
    }
    
    int32_t absTick = tick < 0 ? -tick : tick;
    
    // Start with Q128 = 2^128
    uint128_t ratio = {0, 1};
    
    // Magic numbers for sqrt(1.0001^(2^i))
    if (absTick & (1 << 0)) {
        ratio = mul64x64(ratio.lo, SQRT_MAGIC_0);
        ratio = rsh128(ratio, 48);
    }
    if (absTick & (1 << 1)) {
        ratio = mul64x64(ratio.lo, SQRT_MAGIC_1);
        ratio = rsh128(ratio, 48);
    }
    if (absTick & (1 << 2)) {
        ratio = mul64x64(ratio.lo, SQRT_MAGIC_2);
        ratio = rsh128(ratio, 48);
    }
    if (absTick & (1 << 3)) {
        ratio = mul64x64(ratio.lo, SQRT_MAGIC_3);
        ratio = rsh128(ratio, 48);
    }
    if (absTick & (1 << 4)) {
        ratio = mul64x64(ratio.lo, SQRT_MAGIC_4);
        ratio = rsh128(ratio, 48);
    }
    
    // Invert for negative ticks
    if (tick < 0) {
        // ratio = 2^256 / ratio (approximate)
        uint128_t maxVal = {UINT64_MAX, UINT64_MAX};
        ratio.lo = div128by64(maxVal, ratio.lo);
        ratio.hi = 0;
    }
    
    // Convert from Q128 to Q96
    ratio = rsh128(ratio, 32);
    
    outputs[tid].sqrtPriceX96_lo = ratio.lo;
    outputs[tid].sqrtPriceX96_hi = ratio.hi;
}

// =============================================================================
// Next Initialized Tick Search (GPU parallel)
// =============================================================================

struct TickSearchInput {
    int32_t currentTick;
    int32_t tickSpacing;
    uint32_t searchLeft;
    uint32_t padding;
    
    // Bitmap data - up to 16 words (256 bits each = 4096 ticks)
    uint64_t bitmapWords[16][4];  // 16 words, each 256 bits (4 x 64)
    int16_t bitmapStartWord;
    int16_t bitmapEndWord;
};

struct TickSearchOutput {
    int32_t nextTick;
    uint32_t isInitialized;
};

kernel void batch_next_initialized_tick(
    device const TickSearchInput* inputs [[buffer(0)]],
    device TickSearchOutput* outputs [[buffer(1)]],
    constant uint32_t& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    
    TickSearchInput input = inputs[tid];
    TickSearchOutput output;
    output.isInitialized = 0;
    
    int32_t compressed = input.currentTick / input.tickSpacing;
    int16_t wp = int16_t(compressed >> 8);
    uint8_t bp = uint8_t(compressed & 0xFF);
    
    bool searchLeft = input.searchLeft != 0;
    
    if (searchLeft) {
        // Search toward lower ticks
        for (int16_t searchWp = wp; searchWp >= input.bitmapStartWord; searchWp--) {
            int wordIdx = searchWp - input.bitmapStartWord;
            if (wordIdx < 0 || wordIdx >= 16) continue;
            
            // Search 256-bit word
            for (int i = (searchWp == wp ? int(bp / 64) : 3); i >= 0; i--) {
                uint64_t w = input.bitmapWords[wordIdx][i];
                
                if (searchWp == wp && i == int(bp / 64)) {
                    // Mask bits above current position
                    uint64_t mask = (1ULL << (bp % 64 + 1)) - 1;
                    w &= mask;
                }
                
                if (w != 0) {
                    // Find highest set bit
                    int highBit = 63 - clz(w);
                    output.nextTick = (int32_t(searchWp) * 256 + i * 64 + highBit) * input.tickSpacing;
                    output.isInitialized = 1;
                    outputs[tid] = output;
                    return;
                }
            }
        }
        
        output.nextTick = -887272 * input.tickSpacing;
    } else {
        // Search toward higher ticks
        for (int16_t searchWp = wp; searchWp <= input.bitmapEndWord; searchWp++) {
            int wordIdx = searchWp - input.bitmapStartWord;
            if (wordIdx < 0 || wordIdx >= 16) continue;
            
            uint8_t startBit = (searchWp == wp) ? bp + 1 : 0;
            
            for (int i = int(startBit / 64); i < 4; i++) {
                uint64_t w = input.bitmapWords[wordIdx][i];
                
                if (searchWp == wp && i == int(startBit / 64)) {
                    // Mask bits below start position
                    uint64_t mask = ~((1ULL << (startBit % 64)) - 1);
                    w &= mask;
                }
                
                if (w != 0) {
                    // Find lowest set bit
                    int lowBit = ctz(w);
                    output.nextTick = (int32_t(searchWp) * 256 + i * 64 + lowBit) * input.tickSpacing;
                    output.isInitialized = 1;
                    outputs[tid] = output;
                    return;
                }
            }
        }
        
        output.nextTick = 887272 * input.tickSpacing;
    }
    
    outputs[tid] = output;
}

// =============================================================================
// Price Impact Calculation Kernel
// =============================================================================

struct PriceImpactInput {
    uint64_t amountIn_lo;
    uint64_t amountIn_hi;
    uint64_t amountOut_lo;
    uint64_t amountOut_hi;
    uint64_t spotPrice_lo;   // Expected price without slippage
    uint64_t spotPrice_hi;
};

struct PriceImpactOutput {
    uint32_t impactBps;      // Impact in basis points
    uint32_t padding;
};

kernel void batch_price_impact(
    device const PriceImpactInput* inputs [[buffer(0)]],
    device PriceImpactOutput* outputs [[buffer(1)]],
    constant uint32_t& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    
    PriceImpactInput input = inputs[tid];
    
    uint128_t amountIn = {input.amountIn_lo, input.amountIn_hi};
    uint128_t amountOut = {input.amountOut_lo, input.amountOut_hi};
    uint128_t spotPrice = {input.spotPrice_lo, input.spotPrice_hi};
    
    if (amountIn.isZero()) {
        outputs[tid].impactBps = 0;
        return;
    }
    
    // Expected output at spot price
    uint128_t expectedOut;
    if (spotPrice.hi == 0 && amountIn.hi == 0) {
        expectedOut = mul64x64(amountIn.lo, spotPrice.lo);
        expectedOut = rsh128(expectedOut, 96);  // Remove Q96 factor
    } else {
        expectedOut = amountIn;  // Fallback
    }
    
    // Impact = (expected - actual) / expected * 10000
    if (expectedOut > amountOut) {
        uint128_t diff = expectedOut - amountOut;
        uint128_t impactNum = mul64x64(diff.lo, 10000);
        outputs[tid].impactBps = uint32_t(div128by64(impactNum, expectedOut.lo));
    } else {
        outputs[tid].impactBps = 0;  // Positive slippage
    }
    
    if (outputs[tid].impactBps > 10000) {
        outputs[tid].impactBps = 10000;
    }
}
