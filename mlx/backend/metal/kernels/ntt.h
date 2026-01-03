// Copyright © 2024 Lux Partners Limited
// Metal NTT kernels for lattice cryptography

#pragma once
#include <metal_stdlib>
#include "mlx/backend/metal/kernels/steel/defines.h"
#include "mlx/backend/metal/kernels/uint128.h"

using namespace metal;

// ============================================================================
// Modular Arithmetic Primitives (using 128-bit support)
// ============================================================================

// Modular multiplication using proper 128-bit intermediate
METAL_FUNC ulong mod_mul(ulong a, ulong b, ulong Q) {
    uint128 prod = mul64x64_fast(a, b);
    return mod_128_64(prod, Q);
}

// Modular addition
METAL_FUNC ulong mod_add(ulong a, ulong b, ulong Q) {
    ulong sum = a + b;
    return (sum >= Q || sum < a) ? sum - Q : sum;
}

// Modular subtraction
METAL_FUNC ulong mod_sub(ulong a, ulong b, ulong Q) {
    return (a >= b) ? a - b : a + Q - b;
}

// ============================================================================
// Cooley-Tukey Butterfly (Forward NTT)
// ============================================================================

// In-place Cooley-Tukey butterfly:
// X = x0 + w * x1
// Y = x0 - w * x1
METAL_FUNC void ct_butterfly(
    thread ulong& x0,
    thread ulong& x1,
    ulong w,
    ulong Q) {
    ulong t = mod_mul(x1, w, Q);
    x1 = mod_sub(x0, t, Q);
    x0 = mod_add(x0, t, Q);
}

// ============================================================================
// Gentleman-Sande Butterfly (Inverse NTT)
// ============================================================================

// In-place Gentleman-Sande butterfly:
// X = x0 + x1
// Y = (x0 - x1) * w
METAL_FUNC void gs_butterfly(
    thread ulong& x0,
    thread ulong& x1,
    ulong w,
    ulong Q) {
    ulong t = mod_sub(x0, x1, Q);
    x0 = mod_add(x0, x1, Q);
    x1 = mod_mul(t, w, Q);
}

// ============================================================================
// NTT Kernels
// ============================================================================

// Fused forward NTT for small sizes (N <= 4096) - all in shared memory
template <int N>
[[kernel]] void ntt_forward_fused(
    device ulong* data [[buffer(0)]],
    constant const ulong* twiddles [[buffer(1)]],
    constant const ulong& Q [[buffer(2)]],
    constant const ulong& mu [[buffer(3)]],
    constant const uint& batch [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tgSize [[threads_per_threadgroup]]) {

    threadgroup ulong shared_data[N];

    uint batch_idx = gid.x;
    if (batch_idx >= batch) return;

    device ulong* batch_data = data + batch_idx * N;

    // Load to shared memory
    uint thread_idx = tid.x;
    uint threads = tgSize.x;
    for (uint i = thread_idx; i < N; i += threads) {
        shared_data[i] = batch_data[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cooley-Tukey NTT (decimation-in-time)
    uint log_n = 0;
    for (uint temp = N; temp > 1; temp >>= 1) log_n++;

    for (uint stage = 0; stage < log_n; stage++) {
        uint m = 1u << (stage + 1);
        uint half_m = m >> 1;

        for (uint k = thread_idx; k < N / 2; k += threads) {
            uint j = k / half_m;
            uint i = k % half_m;
            uint idx0 = j * m + i;
            uint idx1 = idx0 + half_m;

            ulong w = twiddles[half_m + i];  // Twiddle lookup

            ulong x0 = shared_data[idx0];
            ulong x1 = shared_data[idx1];
            ct_butterfly(x0, x1, w, Q);
            shared_data[idx0] = x0;
            shared_data[idx1] = x1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store result
    for (uint i = thread_idx; i < N; i += threads) {
        batch_data[i] = shared_data[i];
    }
}

// Fused inverse NTT for small sizes
template <int N>
[[kernel]] void ntt_inverse_fused(
    device ulong* data [[buffer(0)]],
    constant const ulong* inv_twiddles [[buffer(1)]],
    constant const ulong& Q [[buffer(2)]],
    constant const ulong& mu [[buffer(3)]],
    constant const ulong& inv_N [[buffer(4)]],
    constant const uint& batch [[buffer(5)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tgSize [[threads_per_threadgroup]]) {

    threadgroup ulong shared_data[N];

    uint batch_idx = gid.x;
    if (batch_idx >= batch) return;

    device ulong* batch_data = data + batch_idx * N;

    // Load to shared memory
    uint thread_idx = tid.x;
    uint threads = tgSize.x;
    for (uint i = thread_idx; i < N; i += threads) {
        shared_data[i] = batch_data[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Gentleman-Sande NTT (decimation-in-frequency)
    uint log_n = 0;
    for (uint temp = N; temp > 1; temp >>= 1) log_n++;

    for (uint stage = log_n; stage > 0; stage--) {
        uint m = 1u << stage;
        uint half_m = m >> 1;

        for (uint k = thread_idx; k < N / 2; k += threads) {
            uint j = k / half_m;
            uint i = k % half_m;
            uint idx0 = j * m + i;
            uint idx1 = idx0 + half_m;

            ulong w = inv_twiddles[half_m + i];

            ulong x0 = shared_data[idx0];
            ulong x1 = shared_data[idx1];
            gs_butterfly(x0, x1, w, Q);
            shared_data[idx0] = x0;
            shared_data[idx1] = x1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Apply inverse scaling
    for (uint i = thread_idx; i < N; i += threads) {
        shared_data[i] = mod_mul(shared_data[i], inv_N, Q);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Store result
    for (uint i = thread_idx; i < N; i += threads) {
        batch_data[i] = shared_data[i];
    }
}

// Pointwise modular multiplication
[[kernel]] void ntt_pointwise_mul(
    device ulong* result [[buffer(0)]],
    constant const ulong* a [[buffer(1)]],
    constant const ulong* b [[buffer(2)]],
    constant const ulong& Q [[buffer(3)]],
    constant const ulong& mu [[buffer(4)]],
    constant const uint& size [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {

    if (gid >= size) return;
    result[gid] = mod_mul(a[gid], b[gid], Q);
}

// Staged NTT for larger sizes - single stage
[[kernel]] void ntt_forward_stage(
    device ulong* data [[buffer(0)]],
    constant const ulong* twiddles [[buffer(1)]],
    constant const ulong& Q [[buffer(2)]],
    constant const ulong& mu [[buffer(3)]],
    constant const uint& N [[buffer(4)]],
    constant const uint& stage [[buffer(5)]],
    constant const uint& batch [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {

    uint total = batch * (N / 2);
    if (gid >= total) return;

    uint batch_idx = gid / (N / 2);
    uint k = gid % (N / 2);

    device ulong* batch_data = data + batch_idx * N;

    uint m = 1u << (stage + 1);
    uint half_m = m >> 1;
    uint j = k / half_m;
    uint i = k % half_m;
    uint idx0 = j * m + i;
    uint idx1 = idx0 + half_m;

    ulong w = twiddles[half_m + i];

    ulong x0 = batch_data[idx0];
    ulong x1 = batch_data[idx1];
    ct_butterfly(x0, x1, w, Q);
    batch_data[idx0] = x0;
    batch_data[idx1] = x1;
}

[[kernel]] void ntt_inverse_stage(
    device ulong* data [[buffer(0)]],
    constant const ulong* inv_twiddles [[buffer(1)]],
    constant const ulong& Q [[buffer(2)]],
    constant const ulong& mu [[buffer(3)]],
    constant const uint& N [[buffer(4)]],
    constant const uint& stage [[buffer(5)]],
    constant const uint& batch [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {

    uint total = batch * (N / 2);
    if (gid >= total) return;

    uint batch_idx = gid / (N / 2);
    uint k = gid % (N / 2);

    device ulong* batch_data = data + batch_idx * N;

    uint m = 1u << stage;
    uint half_m = m >> 1;
    uint j = k / half_m;
    uint i = k % half_m;
    uint idx0 = j * m + i;
    uint idx1 = idx0 + half_m;

    ulong w = inv_twiddles[half_m + i];

    ulong x0 = batch_data[idx0];
    ulong x1 = batch_data[idx1];
    gs_butterfly(x0, x1, w, Q);
    batch_data[idx0] = x0;
    batch_data[idx1] = x1;
}

// Apply inverse N scaling (after inverse NTT)
[[kernel]] void ntt_scale(
    device ulong* data [[buffer(0)]],
    constant const ulong& Q [[buffer(1)]],
    constant const ulong& inv_N [[buffer(2)]],
    constant const uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {

    if (gid >= size) return;
    data[gid] = mod_mul(data[gid], inv_N, Q);
}
