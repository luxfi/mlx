// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Negacyclic NTT for ML-DSA (Dilithium) and ML-KEM (Kyber)
// NTT over Z_q[X]/(X^n + 1) for n=256,512,1024
// Moduli: q=8380417 (Dilithium), q=3329 (Kyber)
//
// Part of the Lux Network GPU acceleration library

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Constants for ML-DSA and ML-KEM
// ============================================================================

// Dilithium parameters
constant uint32_t DILITHIUM_Q = 8380417;
constant uint32_t DILITHIUM_QINV = 58728449;  // q^(-1) mod 2^32
constant uint32_t DILITHIUM_R2 = 2365951;     // 2^64 mod q (for Montgomery)
constant uint32_t DILITHIUM_ROOT = 1753;      // 256th primitive root of unity

// Kyber parameters
constant uint32_t KYBER_Q = 3329;
constant uint32_t KYBER_QINV = 62209;         // q^(-1) mod 2^16
constant uint32_t KYBER_R2 = 1353;            // 2^32 mod q (for Montgomery)
constant uint32_t KYBER_ROOT = 17;            // 256th primitive root of unity

// ============================================================================
// Montgomery Multiplication
// ============================================================================

// Montgomery reduction for Dilithium (32-bit)
// Input: a < q * 2^32
// Output: a * 2^(-32) mod q
METAL_FUNC int32_t montgomery_reduce_dilithium(int64_t a) {
    int32_t t = (int32_t)a * (int32_t)DILITHIUM_QINV;
    return (int32_t)((a - (int64_t)t * (int64_t)DILITHIUM_Q) >> 32);
}

// Montgomery multiplication for Dilithium
// Returns a * b * 2^(-32) mod q in [-q, q)
METAL_FUNC int32_t mont_mul_dilithium(int32_t a, int32_t b) {
    return montgomery_reduce_dilithium((int64_t)a * (int64_t)b);
}

// Montgomery reduction for Kyber (16-bit)
// Input: a < q * 2^16
// Output: a * 2^(-16) mod q
METAL_FUNC int16_t montgomery_reduce_kyber(int32_t a) {
    int16_t t = (int16_t)a * (int16_t)KYBER_QINV;
    return (int16_t)((a - (int32_t)t * (int32_t)KYBER_Q) >> 16);
}

// Montgomery multiplication for Kyber
METAL_FUNC int16_t mont_mul_kyber(int16_t a, int16_t b) {
    return montgomery_reduce_kyber((int32_t)a * (int32_t)b);
}

// ============================================================================
// Barrett Reduction
// ============================================================================

// Barrett reduction for Dilithium
// Input: a in [0, 2^k) where k < 64
// Output: a mod q
METAL_FUNC int32_t barrett_reduce_dilithium(int32_t a) {
    // Barrett constant: floor(2^46 / q) = 8396807
    int64_t v = 8396807;
    int32_t t = (int32_t)((v * (int64_t)a) >> 46);
    t *= DILITHIUM_Q;
    return a - t;
}

// Barrett reduction for Kyber
// Input: a in [0, 2^k) where k < 32
// Output: a mod q in [0, q)
METAL_FUNC int16_t barrett_reduce_kyber(int16_t a) {
    // Barrett constant: floor(2^26 / q) + 1 = 20159
    int32_t v = 20159;
    int16_t t = (int16_t)((v * (int32_t)a + (1 << 25)) >> 26);
    t *= KYBER_Q;
    return a - t;
}

// ============================================================================
// Conditional Reduction
// ============================================================================

// Reduce to [0, q) for Dilithium
METAL_FUNC int32_t cond_sub_dilithium(int32_t a) {
    a += (a >> 31) & (int32_t)DILITHIUM_Q;  // If negative, add q
    a -= DILITHIUM_Q;
    a += (a >> 31) & (int32_t)DILITHIUM_Q;  // If negative, add q
    return a;
}

// Reduce to [0, q) for Kyber
METAL_FUNC int16_t cond_sub_kyber(int16_t a) {
    a += (a >> 15) & (int16_t)KYBER_Q;
    a -= KYBER_Q;
    a += (a >> 15) & (int16_t)KYBER_Q;
    return a;
}

// ============================================================================
// Bit-Reversal Permutation
// ============================================================================

// Compute bit-reversal of index for n-bit indices
METAL_FUNC uint32_t bit_reverse(uint32_t x, uint32_t log_n) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < log_n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

// In-place bit-reversal permutation kernel
kernel void ntt_bit_reverse_dilithium(
    device int32_t* data [[buffer(0)]],
    constant uint32_t& n [[buffer(1)]],
    constant uint32_t& log_n [[buffer(2)]],
    constant uint32_t& batch [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = gid.y;
    if (batch_idx >= batch) return;
    
    uint32_t i = gid.x;
    if (i >= n) return;
    
    uint32_t j = bit_reverse(i, log_n);
    
    // Only swap once (when i < j)
    if (i < j) {
        device int32_t* batch_data = data + batch_idx * n;
        int32_t tmp = batch_data[i];
        batch_data[i] = batch_data[j];
        batch_data[j] = tmp;
    }
}

kernel void ntt_bit_reverse_kyber(
    device int16_t* data [[buffer(0)]],
    constant uint32_t& n [[buffer(1)]],
    constant uint32_t& log_n [[buffer(2)]],
    constant uint32_t& batch [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = gid.y;
    if (batch_idx >= batch) return;
    
    uint32_t i = gid.x;
    if (i >= n) return;
    
    uint32_t j = bit_reverse(i, log_n);
    
    if (i < j) {
        device int16_t* batch_data = data + batch_idx * n;
        int16_t tmp = batch_data[i];
        batch_data[i] = batch_data[j];
        batch_data[j] = tmp;
    }
}

// ============================================================================
// Cooley-Tukey Butterfly (Forward NTT)
// ============================================================================

// CT butterfly for Dilithium: in-place
// (a, b) <- (a + w*b, a - w*b)
METAL_FUNC void ct_butterfly_dilithium(
    thread int32_t& a,
    thread int32_t& b,
    int32_t w
) {
    int32_t t = mont_mul_dilithium(b, w);
    b = a - t;
    a = a + t;
}

// CT butterfly for Kyber: in-place
METAL_FUNC void ct_butterfly_kyber(
    thread int16_t& a,
    thread int16_t& b,
    int16_t w
) {
    int16_t t = mont_mul_kyber(b, w);
    b = a - t;
    a = a + t;
}

// ============================================================================
// Gentleman-Sande Butterfly (Inverse NTT)
// ============================================================================

// GS butterfly for Dilithium: in-place
// (a, b) <- (a + b, (a - b) * w)
METAL_FUNC void gs_butterfly_dilithium(
    thread int32_t& a,
    thread int32_t& b,
    int32_t w
) {
    int32_t t = a;
    a = t + b;
    b = mont_mul_dilithium(t - b, w);
}

// GS butterfly for Kyber: in-place
METAL_FUNC void gs_butterfly_kyber(
    thread int16_t& a,
    thread int16_t& b,
    int16_t w
) {
    int16_t t = a;
    a = t + b;
    b = mont_mul_kyber(t - b, w);
}

// ============================================================================
// Forward NTT Kernels - Dilithium (q = 8380417)
// ============================================================================

// Single-stage forward NTT for Dilithium
kernel void ntt_forward_stage_dilithium(
    device int32_t* data [[buffer(0)]],
    constant const int32_t* twiddles [[buffer(1)]],
    constant uint32_t& n [[buffer(2)]],
    constant uint32_t& stage [[buffer(3)]],
    constant uint32_t& batch [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint32_t total = batch * (n / 2);
    if (gid >= total) return;
    
    uint32_t batch_idx = gid / (n / 2);
    uint32_t k = gid % (n / 2);
    
    device int32_t* batch_data = data + batch_idx * n;
    
    uint32_t m = 1u << (stage + 1);
    uint32_t half_m = m >> 1;
    uint32_t j = k / half_m;
    uint32_t i = k % half_m;
    uint32_t idx0 = j * m + i;
    uint32_t idx1 = idx0 + half_m;
    
    // Twiddle factor at position half_m + i
    int32_t w = twiddles[half_m + i];
    
    int32_t a = batch_data[idx0];
    int32_t b = batch_data[idx1];
    ct_butterfly_dilithium(a, b, w);
    batch_data[idx0] = a;
    batch_data[idx1] = b;
}

// Fused forward NTT for Dilithium (n=256, all stages in shared memory)
kernel void ntt_forward_fused_256_dilithium(
    device int32_t* data [[buffer(0)]],
    constant const int32_t* twiddles [[buffer(1)]],
    constant uint32_t& batch [[buffer(2)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tg_size [[threads_per_threadgroup]]
) {
    const uint32_t N = 256;
    const uint32_t LOG_N = 8;
    
    threadgroup int32_t shared_data[N];
    
    uint32_t batch_idx = gid.x;
    if (batch_idx >= batch) return;
    
    device int32_t* batch_data = data + batch_idx * N;
    uint32_t thread_idx = tid.x;
    uint32_t threads = tg_size.x;
    
    // Load with bit-reversal
    for (uint32_t i = thread_idx; i < N; i += threads) {
        uint32_t rev_i = bit_reverse(i, LOG_N);
        shared_data[rev_i] = batch_data[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Cooley-Tukey NTT (decimation-in-time)
    for (uint32_t stage = 0; stage < LOG_N; stage++) {
        uint32_t m = 1u << (stage + 1);
        uint32_t half_m = m >> 1;
        
        for (uint32_t k = thread_idx; k < N / 2; k += threads) {
            uint32_t j = k / half_m;
            uint32_t i = k % half_m;
            uint32_t idx0 = j * m + i;
            uint32_t idx1 = idx0 + half_m;
            
            int32_t w = twiddles[half_m + i];
            
            int32_t a = shared_data[idx0];
            int32_t b = shared_data[idx1];
            ct_butterfly_dilithium(a, b, w);
            shared_data[idx0] = a;
            shared_data[idx1] = b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Store result
    for (uint32_t i = thread_idx; i < N; i += threads) {
        batch_data[i] = shared_data[i];
    }
}

// ============================================================================
// Forward NTT Kernels - Kyber (q = 3329)
// ============================================================================

// Single-stage forward NTT for Kyber
kernel void ntt_forward_stage_kyber(
    device int16_t* data [[buffer(0)]],
    constant const int16_t* twiddles [[buffer(1)]],
    constant uint32_t& n [[buffer(2)]],
    constant uint32_t& stage [[buffer(3)]],
    constant uint32_t& batch [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint32_t total = batch * (n / 2);
    if (gid >= total) return;
    
    uint32_t batch_idx = gid / (n / 2);
    uint32_t k = gid % (n / 2);
    
    device int16_t* batch_data = data + batch_idx * n;
    
    uint32_t m = 1u << (stage + 1);
    uint32_t half_m = m >> 1;
    uint32_t j = k / half_m;
    uint32_t i = k % half_m;
    uint32_t idx0 = j * m + i;
    uint32_t idx1 = idx0 + half_m;
    
    int16_t w = twiddles[half_m + i];
    
    int16_t a = batch_data[idx0];
    int16_t b = batch_data[idx1];
    ct_butterfly_kyber(a, b, w);
    batch_data[idx0] = a;
    batch_data[idx1] = b;
}

// Fused forward NTT for Kyber (n=256, all stages in shared memory)
kernel void ntt_forward_fused_256_kyber(
    device int16_t* data [[buffer(0)]],
    constant const int16_t* twiddles [[buffer(1)]],
    constant uint32_t& batch [[buffer(2)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tg_size [[threads_per_threadgroup]]
) {
    const uint32_t N = 256;
    const uint32_t LOG_N = 8;
    
    threadgroup int16_t shared_data[N];
    
    uint32_t batch_idx = gid.x;
    if (batch_idx >= batch) return;
    
    device int16_t* batch_data = data + batch_idx * N;
    uint32_t thread_idx = tid.x;
    uint32_t threads = tg_size.x;
    
    // Load with bit-reversal
    for (uint32_t i = thread_idx; i < N; i += threads) {
        uint32_t rev_i = bit_reverse(i, LOG_N);
        shared_data[rev_i] = batch_data[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Cooley-Tukey NTT
    for (uint32_t stage = 0; stage < LOG_N; stage++) {
        uint32_t m = 1u << (stage + 1);
        uint32_t half_m = m >> 1;
        
        for (uint32_t k = thread_idx; k < N / 2; k += threads) {
            uint32_t j = k / half_m;
            uint32_t i = k % half_m;
            uint32_t idx0 = j * m + i;
            uint32_t idx1 = idx0 + half_m;
            
            int16_t w = twiddles[half_m + i];
            
            int16_t a = shared_data[idx0];
            int16_t b = shared_data[idx1];
            ct_butterfly_kyber(a, b, w);
            shared_data[idx0] = a;
            shared_data[idx1] = b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Store result
    for (uint32_t i = thread_idx; i < N; i += threads) {
        batch_data[i] = shared_data[i];
    }
}

// ============================================================================
// Inverse NTT Kernels - Dilithium
// ============================================================================

// Single-stage inverse NTT for Dilithium
kernel void ntt_inverse_stage_dilithium(
    device int32_t* data [[buffer(0)]],
    constant const int32_t* inv_twiddles [[buffer(1)]],
    constant uint32_t& n [[buffer(2)]],
    constant uint32_t& stage [[buffer(3)]],
    constant uint32_t& batch [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint32_t total = batch * (n / 2);
    if (gid >= total) return;
    
    uint32_t batch_idx = gid / (n / 2);
    uint32_t k = gid % (n / 2);
    
    device int32_t* batch_data = data + batch_idx * n;
    
    uint32_t m = 1u << stage;
    uint32_t half_m = m >> 1;
    uint32_t j = k / half_m;
    uint32_t i = k % half_m;
    uint32_t idx0 = j * m + i;
    uint32_t idx1 = idx0 + half_m;
    
    int32_t w = inv_twiddles[half_m + i];
    
    int32_t a = batch_data[idx0];
    int32_t b = batch_data[idx1];
    gs_butterfly_dilithium(a, b, w);
    batch_data[idx0] = a;
    batch_data[idx1] = b;
}

// Fused inverse NTT for Dilithium (n=256)
kernel void ntt_inverse_fused_256_dilithium(
    device int32_t* data [[buffer(0)]],
    constant const int32_t* inv_twiddles [[buffer(1)]],
    constant int32_t& inv_n [[buffer(2)]],
    constant uint32_t& batch [[buffer(3)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tg_size [[threads_per_threadgroup]]
) {
    const uint32_t N = 256;
    const uint32_t LOG_N = 8;
    
    threadgroup int32_t shared_data[N];
    
    uint32_t batch_idx = gid.x;
    if (batch_idx >= batch) return;
    
    device int32_t* batch_data = data + batch_idx * N;
    uint32_t thread_idx = tid.x;
    uint32_t threads = tg_size.x;
    
    // Load to shared memory
    for (uint32_t i = thread_idx; i < N; i += threads) {
        shared_data[i] = batch_data[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Gentleman-Sande NTT (decimation-in-frequency)
    for (uint32_t stage = LOG_N; stage > 0; stage--) {
        uint32_t m = 1u << stage;
        uint32_t half_m = m >> 1;
        
        for (uint32_t k = thread_idx; k < N / 2; k += threads) {
            uint32_t j = k / half_m;
            uint32_t i = k % half_m;
            uint32_t idx0 = j * m + i;
            uint32_t idx1 = idx0 + half_m;
            
            int32_t w = inv_twiddles[half_m + i];
            
            int32_t a = shared_data[idx0];
            int32_t b = shared_data[idx1];
            gs_butterfly_dilithium(a, b, w);
            shared_data[idx0] = a;
            shared_data[idx1] = b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Apply inverse N scaling and bit-reverse
    for (uint32_t i = thread_idx; i < N; i += threads) {
        uint32_t rev_i = bit_reverse(i, LOG_N);
        int32_t val = mont_mul_dilithium(shared_data[i], inv_n);
        batch_data[rev_i] = cond_sub_dilithium(val);
    }
}

// ============================================================================
// Inverse NTT Kernels - Kyber
// ============================================================================

// Single-stage inverse NTT for Kyber
kernel void ntt_inverse_stage_kyber(
    device int16_t* data [[buffer(0)]],
    constant const int16_t* inv_twiddles [[buffer(1)]],
    constant uint32_t& n [[buffer(2)]],
    constant uint32_t& stage [[buffer(3)]],
    constant uint32_t& batch [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint32_t total = batch * (n / 2);
    if (gid >= total) return;
    
    uint32_t batch_idx = gid / (n / 2);
    uint32_t k = gid % (n / 2);
    
    device int16_t* batch_data = data + batch_idx * n;
    
    uint32_t m = 1u << stage;
    uint32_t half_m = m >> 1;
    uint32_t j = k / half_m;
    uint32_t i = k % half_m;
    uint32_t idx0 = j * m + i;
    uint32_t idx1 = idx0 + half_m;
    
    int16_t w = inv_twiddles[half_m + i];
    
    int16_t a = batch_data[idx0];
    int16_t b = batch_data[idx1];
    gs_butterfly_kyber(a, b, w);
    batch_data[idx0] = a;
    batch_data[idx1] = b;
}

// Fused inverse NTT for Kyber (n=256)
kernel void ntt_inverse_fused_256_kyber(
    device int16_t* data [[buffer(0)]],
    constant const int16_t* inv_twiddles [[buffer(1)]],
    constant int16_t& inv_n [[buffer(2)]],
    constant uint32_t& batch [[buffer(3)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tg_size [[threads_per_threadgroup]]
) {
    const uint32_t N = 256;
    const uint32_t LOG_N = 8;
    
    threadgroup int16_t shared_data[N];
    
    uint32_t batch_idx = gid.x;
    if (batch_idx >= batch) return;
    
    device int16_t* batch_data = data + batch_idx * N;
    uint32_t thread_idx = tid.x;
    uint32_t threads = tg_size.x;
    
    // Load to shared memory
    for (uint32_t i = thread_idx; i < N; i += threads) {
        shared_data[i] = batch_data[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Gentleman-Sande NTT
    for (uint32_t stage = LOG_N; stage > 0; stage--) {
        uint32_t m = 1u << stage;
        uint32_t half_m = m >> 1;
        
        for (uint32_t k = thread_idx; k < N / 2; k += threads) {
            uint32_t j = k / half_m;
            uint32_t i = k % half_m;
            uint32_t idx0 = j * m + i;
            uint32_t idx1 = idx0 + half_m;
            
            int16_t w = inv_twiddles[half_m + i];
            
            int16_t a = shared_data[idx0];
            int16_t b = shared_data[idx1];
            gs_butterfly_kyber(a, b, w);
            shared_data[idx0] = a;
            shared_data[idx1] = b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Apply inverse N scaling and bit-reverse
    for (uint32_t i = thread_idx; i < N; i += threads) {
        uint32_t rev_i = bit_reverse(i, LOG_N);
        int16_t val = mont_mul_kyber(shared_data[i], inv_n);
        batch_data[rev_i] = cond_sub_kyber(val);
    }
}

// ============================================================================
// Note: For larger NTT sizes (512, 1024), use the staged kernels:
//   - ntt_forward_single_stage with appropriate stage parameters
//   - Multiple stages can be composed for larger transforms
// ============================================================================
