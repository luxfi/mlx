// Copyright (c) 2024-2026 Lux Industries Inc. All Rights Reserved.
// PROPRIETARY AND CONFIDENTIAL - NO LICENSE GRANTED
// Contact: licensing@luxindustries.xyz
//
// DEX Atomic Swap Operations - High-Performance CUDA Implementation
// Implements QuantumSwap/LX order matching with GPU-accelerated price matching,
// order book maintenance, and atomic swap execution.

#include <cuda_runtime.h>
#include <cstdint>

namespace lux {
namespace cuda {
namespace dex {

// ============================================================================
// Order Book Structures
// ============================================================================

// Price representation: fixed-point Q64.64
struct Price {
    uint64_t integer;
    uint64_t fraction;
};

// Order side
enum class Side : uint8_t {
    BID = 0,
    ASK = 1
};

// Order type
enum class OrderType : uint8_t {
    LIMIT = 0,
    MARKET = 1,
    IOC = 2,      // Immediate-or-Cancel
    FOK = 3       // Fill-or-Kill
};

// Order status
enum class OrderStatus : uint8_t {
    OPEN = 0,
    PARTIAL = 1,
    FILLED = 2,
    CANCELLED = 3
};

// Single order entry
struct Order {
    uint64_t order_id;
    uint64_t user_id;
    Price price;
    uint64_t quantity;          // Original quantity
    uint64_t remaining;         // Remaining quantity
    uint64_t timestamp;
    Side side;
    OrderType type;
    OrderStatus status;
    uint8_t _padding[5];
};

// Trade execution result
struct Trade {
    uint64_t trade_id;
    uint64_t maker_order_id;
    uint64_t taker_order_id;
    uint64_t maker_user_id;
    uint64_t taker_user_id;
    Price price;
    uint64_t quantity;
    uint64_t timestamp;
    Side taker_side;
    uint8_t _padding[7];
};

// Order book level (aggregated)
struct Level {
    Price price;
    uint64_t total_quantity;
    uint32_t order_count;
    uint32_t _padding;
};

// Match result for single order
struct MatchResult {
    uint32_t num_trades;
    uint64_t total_filled;
    uint64_t remaining;
    uint8_t fully_filled;
    uint8_t _padding[7];
};

// ============================================================================
// Price Comparison Utilities
// ============================================================================

__device__ __forceinline__
int price_compare(const Price& a, const Price& b) {
    if (a.integer != b.integer) {
        return (a.integer > b.integer) ? 1 : -1;
    }
    if (a.fraction != b.fraction) {
        return (a.fraction > b.fraction) ? 1 : -1;
    }
    return 0;
}

__device__ __forceinline__
bool price_eq(const Price& a, const Price& b) {
    return a.integer == b.integer && a.fraction == b.fraction;
}

__device__ __forceinline__
bool price_lt(const Price& a, const Price& b) {
    return price_compare(a, b) < 0;
}

__device__ __forceinline__
bool price_le(const Price& a, const Price& b) {
    return price_compare(a, b) <= 0;
}

__device__ __forceinline__
bool price_gt(const Price& a, const Price& b) {
    return price_compare(a, b) > 0;
}

__device__ __forceinline__
bool price_ge(const Price& a, const Price& b) {
    return price_compare(a, b) >= 0;
}

// Check if prices match for trade
// Bid matches ask if bid.price >= ask.price
__device__ __forceinline__
bool prices_match(const Price& bid, const Price& ask) {
    return price_ge(bid, ask);
}

// ============================================================================
// Atomic Order Book Operations
// ============================================================================

// Atomic quantity reduction
__device__ __forceinline__
uint64_t atomic_reduce_quantity(uint64_t* addr, uint64_t amount) {
    uint64_t old = *addr;
    uint64_t assumed;
    do {
        assumed = old;
        if (assumed < amount) {
            amount = assumed;  // Can only take what's available
        }
        old = atomicCAS((unsigned long long*)addr,
                        (unsigned long long)assumed,
                        (unsigned long long)(assumed - amount));
    } while (old != assumed);
    return amount;  // Return actual amount taken
}

// Atomic order status update
__device__ __forceinline__
OrderStatus atomic_update_status(OrderStatus* addr, OrderStatus new_status) {
    return (OrderStatus)atomicExch((unsigned int*)addr, (unsigned int)new_status);
}

// ============================================================================
// Order Matching Kernels
// ============================================================================

// Match single incoming order against order book
// Each thread handles matching against one resting order
__global__ void match_order_kernel(
    const Order* __restrict__ incoming,
    Order* __restrict__ book_orders,
    const uint32_t* __restrict__ book_indices,  // Indices of orders at matching prices
    uint32_t num_book_orders,
    Trade* __restrict__ trades,
    uint32_t* __restrict__ num_trades,
    uint64_t* __restrict__ remaining_qty,
    uint64_t trade_id_base,
    uint64_t timestamp
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_book_orders) return;

    __shared__ uint64_t s_remaining;
    __shared__ uint32_t s_trade_count;

    if (threadIdx.x == 0) {
        s_remaining = incoming->quantity;
        s_trade_count = 0;
    }
    __syncthreads();

    // Early exit if incoming order is filled
    if (s_remaining == 0) return;

    uint32_t book_idx = book_indices[tid];
    Order* book_order = &book_orders[book_idx];

    // Check if this order can match
    bool can_match = false;
    if (incoming->side == Side::BID) {
        // Incoming is bid, book order should be ask
        can_match = (book_order->side == Side::ASK) &&
                    prices_match(incoming->price, book_order->price);
    } else {
        // Incoming is ask, book order should be bid
        can_match = (book_order->side == Side::BID) &&
                    prices_match(book_order->price, incoming->price);
    }

    if (!can_match) return;
    if (book_order->status != OrderStatus::OPEN &&
        book_order->status != OrderStatus::PARTIAL) return;

    // Try to match quantities
    uint64_t book_available = book_order->remaining;
    if (book_available == 0) return;

    // Atomically claim quantity from remaining
    uint64_t to_fill = atomicMin((unsigned long long*)&s_remaining,
                                  (unsigned long long)book_available);
    if (to_fill == 0) return;

    uint64_t actual_fill = min(to_fill, book_available);

    // Atomically reduce book order quantity
    uint64_t taken = atomic_reduce_quantity(&book_order->remaining, actual_fill);
    if (taken == 0) return;

    // Update book order status
    if (book_order->remaining == 0) {
        atomic_update_status(&book_order->status, OrderStatus::FILLED);
    } else {
        atomic_update_status(&book_order->status, OrderStatus::PARTIAL);
    }

    // Record trade
    uint32_t trade_idx = atomicAdd(&s_trade_count, 1);

    Trade* trade = &trades[trade_idx];
    trade->trade_id = trade_id_base + trade_idx;
    trade->maker_order_id = book_order->order_id;
    trade->taker_order_id = incoming->order_id;
    trade->maker_user_id = book_order->user_id;
    trade->taker_user_id = incoming->user_id;
    trade->price = book_order->price;  // Maker price
    trade->quantity = taken;
    trade->timestamp = timestamp;
    trade->taker_side = incoming->side;

    // Update remaining
    atomicSub((unsigned long long*)&s_remaining, (unsigned long long)taken);

    __syncthreads();

    // Write back results
    if (threadIdx.x == 0) {
        *remaining_qty = s_remaining;
        *num_trades = s_trade_count;
    }
}

// Batch order matching - process multiple incoming orders in parallel
__global__ void batch_match_kernel(
    const Order* __restrict__ incoming_orders,
    uint32_t num_incoming,
    Order* __restrict__ book_orders,
    uint32_t num_book_orders,
    Level* __restrict__ bid_levels,
    Level* __restrict__ ask_levels,
    uint32_t max_levels,
    Trade* __restrict__ trades,
    uint32_t* __restrict__ trade_counts,     // Per incoming order
    MatchResult* __restrict__ results,
    uint64_t trade_id_base,
    uint64_t timestamp
) {
    const uint32_t order_idx = blockIdx.x;
    const uint32_t thread_in_block = threadIdx.x;

    if (order_idx >= num_incoming) return;

    const Order* incoming = &incoming_orders[order_idx];
    MatchResult* result = &results[order_idx];

    __shared__ uint64_t s_remaining;
    __shared__ uint32_t s_trade_count;

    if (thread_in_block == 0) {
        s_remaining = incoming->quantity;
        s_trade_count = 0;
    }
    __syncthreads();

    // Select which side of book to match against
    Level* match_levels = (incoming->side == Side::BID) ? ask_levels : bid_levels;

    // Each thread scans one level
    if (thread_in_block < max_levels) {
        Level* level = &match_levels[thread_in_block];

        if (level->total_quantity == 0) return;

        // Check price match
        bool can_match;
        if (incoming->side == Side::BID) {
            can_match = prices_match(incoming->price, level->price);
        } else {
            can_match = prices_match(level->price, incoming->price);
        }

        if (!can_match) return;

        // Match at this level
        uint64_t level_qty = level->total_quantity;
        uint64_t want = atomicMin((unsigned long long*)&s_remaining,
                                   (unsigned long long)level_qty);

        if (want > 0) {
            uint64_t got = min(want, level_qty);
            atomicSub((unsigned long long*)&level->total_quantity,
                      (unsigned long long)got);
            atomicSub((unsigned long long*)&s_remaining,
                      (unsigned long long)got);
            atomicAdd(&s_trade_count, 1);
        }
    }
    __syncthreads();

    if (thread_in_block == 0) {
        result->num_trades = s_trade_count;
        result->total_filled = incoming->quantity - s_remaining;
        result->remaining = s_remaining;
        result->fully_filled = (s_remaining == 0) ? 1 : 0;
        trade_counts[order_idx] = s_trade_count;
    }
}

// ============================================================================
// Order Book Aggregation Kernels
// ============================================================================

// Aggregate orders into price levels
// Each block handles one price level
__global__ void aggregate_levels_kernel(
    const Order* __restrict__ orders,
    uint32_t num_orders,
    Level* __restrict__ levels,
    const Price* __restrict__ level_prices,
    uint32_t num_levels,
    Side side
) {
    const uint32_t level_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;

    if (level_idx >= num_levels) return;

    __shared__ uint64_t s_total_qty;
    __shared__ uint32_t s_order_count;

    if (tid == 0) {
        s_total_qty = 0;
        s_order_count = 0;
    }
    __syncthreads();

    Price target_price = level_prices[level_idx];

    // Each thread scans a portion of orders
    for (uint32_t i = tid; i < num_orders; i += blockDim.x) {
        const Order* order = &orders[i];

        if (order->side != side) continue;
        if (order->status != OrderStatus::OPEN &&
            order->status != OrderStatus::PARTIAL) continue;
        if (!price_eq(order->price, target_price)) continue;

        atomicAdd((unsigned long long*)&s_total_qty,
                  (unsigned long long)order->remaining);
        atomicAdd(&s_order_count, 1);
    }
    __syncthreads();

    if (tid == 0) {
        Level* level = &levels[level_idx];
        level->price = target_price;
        level->total_quantity = s_total_qty;
        level->order_count = s_order_count;
    }
}

// Find best bid/ask prices
__global__ void find_best_prices_kernel(
    const Order* __restrict__ orders,
    uint32_t num_orders,
    Price* __restrict__ best_bid,
    Price* __restrict__ best_ask
) {
    __shared__ Price s_best_bid;
    __shared__ Price s_best_ask;

    const uint32_t tid = threadIdx.x;

    if (tid == 0) {
        s_best_bid.integer = 0;
        s_best_bid.fraction = 0;
        s_best_ask.integer = UINT64_MAX;
        s_best_ask.fraction = UINT64_MAX;
    }
    __syncthreads();

    // Thread-local best prices
    Price local_best_bid = {0, 0};
    Price local_best_ask = {UINT64_MAX, UINT64_MAX};

    for (uint32_t i = tid; i < num_orders; i += blockDim.x) {
        const Order* order = &orders[i];

        if (order->status != OrderStatus::OPEN &&
            order->status != OrderStatus::PARTIAL) continue;
        if (order->remaining == 0) continue;

        if (order->side == Side::BID) {
            if (price_gt(order->price, local_best_bid)) {
                local_best_bid = order->price;
            }
        } else {
            if (price_lt(order->price, local_best_ask)) {
                local_best_ask = order->price;
            }
        }
    }

    // Reduce across threads (simplified - use proper reduction for production)
    __syncthreads();

    // Atomic max for bid (compare integers first, then fractions)
    atomicMax((unsigned long long*)&s_best_bid.integer,
              (unsigned long long)local_best_bid.integer);

    // Atomic min for ask
    atomicMin((unsigned long long*)&s_best_ask.integer,
              (unsigned long long)local_best_ask.integer);

    __syncthreads();

    if (tid == 0) {
        *best_bid = s_best_bid;
        *best_ask = s_best_ask;
    }
}

// ============================================================================
// Price-Time Priority Sorting
// ============================================================================

// Comparison for bid orders (higher price, earlier time first)
__device__ __forceinline__
bool bid_order_less(const Order& a, const Order& b) {
    int pc = price_compare(a.price, b.price);
    if (pc != 0) return pc > 0;  // Higher price first
    return a.timestamp < b.timestamp;  // Earlier time first
}

// Comparison for ask orders (lower price, earlier time first)
__device__ __forceinline__
bool ask_order_less(const Order& a, const Order& b) {
    int pc = price_compare(a.price, b.price);
    if (pc != 0) return pc < 0;  // Lower price first
    return a.timestamp < b.timestamp;  // Earlier time first
}

// Bitonic sort step for order book
__global__ void bitonic_sort_step_kernel(
    Order* __restrict__ orders,
    uint32_t num_orders,
    uint32_t step,
    uint32_t stage,
    Side side
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t partner = tid ^ step;

    if (tid >= num_orders || partner >= num_orders || tid >= partner) return;

    bool ascending = ((tid & stage) == 0);

    Order a = orders[tid];
    Order b = orders[partner];

    bool should_swap;
    if (side == Side::BID) {
        should_swap = ascending ? bid_order_less(b, a) : bid_order_less(a, b);
    } else {
        should_swap = ascending ? ask_order_less(b, a) : ask_order_less(a, b);
    }

    if (should_swap) {
        orders[tid] = b;
        orders[partner] = a;
    }
}

// ============================================================================
// Atomic Swap Execution
// ============================================================================

// Token balance update structure
struct BalanceUpdate {
    uint64_t user_id;
    uint64_t token_id;
    int64_t delta;        // Positive = credit, negative = debit
    uint64_t _padding;
};

// Execute atomic swaps from matched trades
__global__ void execute_swaps_kernel(
    const Trade* __restrict__ trades,
    uint32_t num_trades,
    BalanceUpdate* __restrict__ updates,
    uint64_t base_token_id,
    uint64_t quote_token_id
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_trades) return;

    const Trade* trade = &trades[tid];

    // Each trade generates 4 balance updates:
    // 1. Maker sends base, receives quote (if maker is seller)
    // 2. Taker sends quote, receives base (if taker is buyer)
    // Or vice versa depending on taker_side

    uint32_t update_base = tid * 4;

    // Calculate amounts (price * quantity for quote amount)
    // Simplified: assume price is in quote units per base unit
    uint64_t base_amount = trade->quantity;
    // quote_amount = price * quantity (simplified integer math)
    uint64_t quote_amount = trade->price.integer * trade->quantity;

    if (trade->taker_side == Side::BID) {
        // Taker is buying base with quote
        // Taker: -quote, +base
        // Maker: -base, +quote

        updates[update_base + 0] = {trade->taker_user_id, quote_token_id,
                                     -(int64_t)quote_amount, 0};
        updates[update_base + 1] = {trade->taker_user_id, base_token_id,
                                     (int64_t)base_amount, 0};
        updates[update_base + 2] = {trade->maker_user_id, base_token_id,
                                     -(int64_t)base_amount, 0};
        updates[update_base + 3] = {trade->maker_user_id, quote_token_id,
                                     (int64_t)quote_amount, 0};
    } else {
        // Taker is selling base for quote
        // Taker: -base, +quote
        // Maker: -quote, +base

        updates[update_base + 0] = {trade->taker_user_id, base_token_id,
                                     -(int64_t)base_amount, 0};
        updates[update_base + 1] = {trade->taker_user_id, quote_token_id,
                                     (int64_t)quote_amount, 0};
        updates[update_base + 2] = {trade->maker_user_id, quote_token_id,
                                     -(int64_t)quote_amount, 0};
        updates[update_base + 3] = {trade->maker_user_id, base_token_id,
                                     (int64_t)base_amount, 0};
    }
}

// Aggregate balance updates by user and token
__global__ void aggregate_balance_updates_kernel(
    const BalanceUpdate* __restrict__ updates,
    uint32_t num_updates,
    int64_t* __restrict__ balances,  // [user_id * max_tokens + token_id]
    uint32_t max_tokens
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_updates) return;

    const BalanceUpdate* update = &updates[tid];
    uint64_t idx = update->user_id * max_tokens + update->token_id;

    atomicAdd((long long*)&balances[idx], (long long)update->delta);
}

// ============================================================================
// Host API (C Interface)
// ============================================================================

} // namespace dex
} // namespace cuda
} // namespace lux

extern "C" {

using namespace lux::cuda::dex;

void lux_cuda_dex_match_order(
    const Order* incoming,
    Order* book_orders,
    const uint32_t* book_indices,
    uint32_t num_book_orders,
    Trade* trades,
    uint32_t* num_trades,
    uint64_t* remaining_qty,
    uint64_t trade_id_base,
    uint64_t timestamp,
    cudaStream_t stream
) {
    if (num_book_orders == 0) {
        cudaMemsetAsync(num_trades, 0, sizeof(uint32_t), stream);
        cudaMemcpyAsync(remaining_qty, &incoming->quantity, sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice, stream);
        return;
    }

    dim3 block(256);
    dim3 grid((num_book_orders + 255) / 256);

    match_order_kernel<<<grid, block, 0, stream>>>(
        incoming, book_orders, book_indices, num_book_orders,
        trades, num_trades, remaining_qty, trade_id_base, timestamp
    );
}

void lux_cuda_dex_batch_match(
    const Order* incoming_orders,
    uint32_t num_incoming,
    Order* book_orders,
    uint32_t num_book_orders,
    Level* bid_levels,
    Level* ask_levels,
    uint32_t max_levels,
    Trade* trades,
    uint32_t* trade_counts,
    MatchResult* results,
    uint64_t trade_id_base,
    uint64_t timestamp,
    cudaStream_t stream
) {
    dim3 block(128);
    dim3 grid(num_incoming);

    batch_match_kernel<<<grid, block, 0, stream>>>(
        incoming_orders, num_incoming, book_orders, num_book_orders,
        bid_levels, ask_levels, max_levels, trades, trade_counts,
        results, trade_id_base, timestamp
    );
}

void lux_cuda_dex_aggregate_levels(
    const Order* orders,
    uint32_t num_orders,
    Level* levels,
    const Price* level_prices,
    uint32_t num_levels,
    uint8_t side,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(num_levels);

    aggregate_levels_kernel<<<grid, block, 0, stream>>>(
        orders, num_orders, levels, level_prices, num_levels, (Side)side
    );
}

void lux_cuda_dex_find_best_prices(
    const Order* orders,
    uint32_t num_orders,
    Price* best_bid,
    Price* best_ask,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(1);

    find_best_prices_kernel<<<grid, block, 0, stream>>>(
        orders, num_orders, best_bid, best_ask
    );
}

void lux_cuda_dex_sort_orders(
    Order* orders,
    uint32_t num_orders,
    uint8_t side,
    cudaStream_t stream
) {
    // Bitonic sort
    for (uint32_t stage = 2; stage <= num_orders; stage <<= 1) {
        for (uint32_t step = stage >> 1; step > 0; step >>= 1) {
            dim3 block(256);
            dim3 grid((num_orders + 255) / 256);

            bitonic_sort_step_kernel<<<grid, block, 0, stream>>>(
                orders, num_orders, step, stage, (Side)side
            );
        }
    }
}

void lux_cuda_dex_execute_swaps(
    const Trade* trades,
    uint32_t num_trades,
    BalanceUpdate* updates,
    uint64_t base_token_id,
    uint64_t quote_token_id,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((num_trades + 255) / 256);

    execute_swaps_kernel<<<grid, block, 0, stream>>>(
        trades, num_trades, updates, base_token_id, quote_token_id
    );
}

void lux_cuda_dex_aggregate_balances(
    const BalanceUpdate* updates,
    uint32_t num_updates,
    int64_t* balances,
    uint32_t max_tokens,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((num_updates + 255) / 256);

    aggregate_balance_updates_kernel<<<grid, block, 0, stream>>>(
        updates, num_updates, balances, max_tokens
    );
}

} // extern "C"
