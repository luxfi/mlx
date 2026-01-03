// Copyright © 2024 Lux Partners Limited
// Metal NTT backend implementation

#include <cassert>
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ntt.h"
#include "mlx/utils.h"

namespace mlx::core::ntt {

namespace {

// Compute modular power: base^exp mod Q
uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t Q) {
    uint64_t result = 1;
    base %= Q;
    while (exp > 0) {
        if (exp & 1) {
            // result = (result * base) % Q
            __uint128_t prod = (__uint128_t)result * base;
            result = (uint64_t)(prod % Q);
        }
        exp >>= 1;
        __uint128_t prod = (__uint128_t)base * base;
        base = (uint64_t)(prod % Q);
    }
    return result;
}

// Find primitive N-th root of unity for NTT-friendly prime Q
// Q must be of form Q = k * N + 1 for some k
uint64_t find_primitive_root(uint32_t N, uint64_t Q) {
    // For Q = 998244353, primitive root is 3
    // Q - 1 = 998244352 = 2^23 * 7 * 17
    if (Q == 998244353ULL) {
        // Generator g = 3, primitive root = g^((Q-1)/N)
        return mod_pow(3, (Q - 1) / N, Q);
    }

    // Generic search for primitive root
    uint64_t g = 2;
    while (g < Q) {
        uint64_t root = mod_pow(g, (Q - 1) / N, Q);
        // Verify: root^N = 1 mod Q and root^(N/2) != 1 mod Q
        if (mod_pow(root, N, Q) == 1 && mod_pow(root, N / 2, Q) != 1) {
            return root;
        }
        g++;
    }
    return 0; // Should never reach here for valid NTT primes
}

// Compute Barrett reduction constant: floor(2^64 / Q)
uint64_t compute_barrett_mu(uint64_t Q) {
    return (uint64_t)(((__uint128_t)1 << 64) / Q);
}

// Compute modular inverse using extended Euclidean algorithm
uint64_t mod_inv(uint64_t a, uint64_t Q) {
    int64_t t = 0, newt = 1;
    int64_t r = Q, newr = a;

    while (newr != 0) {
        int64_t quotient = r / newr;
        int64_t temp = t;
        t = newt;
        newt = temp - quotient * newt;
        temp = r;
        r = newr;
        newr = temp - quotient * newr;
    }

    if (t < 0) t += Q;
    return (uint64_t)t;
}

// Precompute twiddle factors: powers of root
array compute_twiddles(uint32_t N, uint64_t root, uint64_t Q) {
    std::vector<uint64_t> twiddles(N);

    // Bit-reversal ordering for twiddles
    twiddles[0] = 1;
    for (uint32_t i = 1; i < N; i++) {
        __uint128_t prod = (__uint128_t)twiddles[i - 1] * root;
        twiddles[i] = (uint64_t)(prod % Q);
    }

    return array(twiddles.data(), {static_cast<int>(N)}, uint64);
}

} // anonymous namespace

bool gpu_available(StreamOrDevice s) {
#ifdef MLX_BUILD_METAL
    return metal::is_available();
#else
    return false;
#endif
}

const char* backend_name(StreamOrDevice s) {
#ifdef MLX_BUILD_METAL
    if (metal::is_available()) {
        return "Metal (MLX)";
    }
#endif
#ifdef MLX_BUILD_CUDA
    return "CUDA";
#endif
    return "CPU";
}

NTTContext create_context(uint32_t N, uint64_t Q, StreamOrDevice s) {
    NTTContext ctx;
    ctx.N = N;
    ctx.Q = Q;
    ctx.root = find_primitive_root(N, Q);
    ctx.inv_root = mod_inv(ctx.root, Q);
    ctx.inv_N = mod_inv(N, Q);
    ctx.barrett_mu = compute_barrett_mu(Q);

    // Precompute twiddle factors
    ctx.twiddles = compute_twiddles(N, ctx.root, Q);
    ctx.inv_twiddles = compute_twiddles(N, ctx.inv_root, Q);

    return ctx;
}

void destroy_context(NTTContext& ctx) {
    // Arrays are automatically cleaned up by MLX
    ctx.N = 0;
    ctx.Q = 0;
}

array forward(const NTTContext& ctx, const array& a, StreamOrDevice s) {
#ifdef MLX_BUILD_METAL
    if (!metal::is_available()) {
        throw std::runtime_error("Metal not available for NTT");
    }

    auto& d = metal::device(s.stream().device);
    auto& compute_encoder = d.get_command_encoder(s.stream().index);

    uint32_t N = ctx.N;
    uint32_t batch = (a.size() + N - 1) / N;

    // Create output array
    array out = array(a.shape(), a.dtype(), nullptr, {});
    out.set_data(allocator::malloc(out.nbytes()));

    // Get kernel based on N
    std::string kernel_name;
    if (N <= 4096) {
        kernel_name = "ntt_forward_fused_" + std::to_string(N);
    } else {
        kernel_name = "ntt_forward_stage";
    }

    auto kernel = d.get_kernel(kernel_name);
    compute_encoder.set_compute_pipeline_state(kernel);

    // Set buffers
    compute_encoder.set_input_array(a, 0);
    compute_encoder.set_input_array(ctx.twiddles, 1);
    compute_encoder.set_bytes(&ctx.Q, sizeof(ctx.Q), 2);
    compute_encoder.set_bytes(&ctx.barrett_mu, sizeof(ctx.barrett_mu), 3);
    compute_encoder.set_bytes(&batch, sizeof(batch), 4);
    compute_encoder.set_output_array(out, 0);

    if (N <= 4096) {
        // Fused kernel: one threadgroup per batch element
        uint32_t threads_per_group = std::min(N / 2, 256u);
        compute_encoder.dispatch_threadgroups(
            {batch, 1, 1},
            {threads_per_group, 1, 1});
    } else {
        // Staged kernel: log2(N) passes
        uint32_t log_n = 0;
        for (uint32_t temp = N; temp > 1; temp >>= 1) log_n++;

        for (uint32_t stage = 0; stage < log_n; stage++) {
            compute_encoder.set_bytes(&N, sizeof(N), 4);
            compute_encoder.set_bytes(&stage, sizeof(stage), 5);
            compute_encoder.set_bytes(&batch, sizeof(batch), 6);

            uint32_t total_threads = batch * (N / 2);
            compute_encoder.dispatch_threads(
                {total_threads, 1, 1},
                {256, 1, 1});
        }
    }

    d.commit_command_buffer(s.stream().index);
    return out;
#else
    throw std::runtime_error("Metal not available");
#endif
}

array inverse(const NTTContext& ctx, const array& a, StreamOrDevice s) {
#ifdef MLX_BUILD_METAL
    if (!metal::is_available()) {
        throw std::runtime_error("Metal not available for NTT");
    }

    auto& d = metal::device(s.stream().device);
    auto& compute_encoder = d.get_command_encoder(s.stream().index);

    uint32_t N = ctx.N;
    uint32_t batch = (a.size() + N - 1) / N;

    array out = array(a.shape(), a.dtype(), nullptr, {});
    out.set_data(allocator::malloc(out.nbytes()));

    std::string kernel_name;
    if (N <= 4096) {
        kernel_name = "ntt_inverse_fused_" + std::to_string(N);
    } else {
        kernel_name = "ntt_inverse_stage";
    }

    auto kernel = d.get_kernel(kernel_name);
    compute_encoder.set_compute_pipeline_state(kernel);

    compute_encoder.set_input_array(a, 0);
    compute_encoder.set_input_array(ctx.inv_twiddles, 1);
    compute_encoder.set_bytes(&ctx.Q, sizeof(ctx.Q), 2);
    compute_encoder.set_bytes(&ctx.barrett_mu, sizeof(ctx.barrett_mu), 3);
    compute_encoder.set_bytes(&ctx.inv_N, sizeof(ctx.inv_N), 4);
    compute_encoder.set_bytes(&batch, sizeof(batch), 5);
    compute_encoder.set_output_array(out, 0);

    if (N <= 4096) {
        uint32_t threads_per_group = std::min(N / 2, 256u);
        compute_encoder.dispatch_threadgroups(
            {batch, 1, 1},
            {threads_per_group, 1, 1});
    } else {
        uint32_t log_n = 0;
        for (uint32_t temp = N; temp > 1; temp >>= 1) log_n++;

        for (uint32_t stage = log_n; stage > 0; stage--) {
            compute_encoder.set_bytes(&N, sizeof(N), 4);
            compute_encoder.set_bytes(&stage, sizeof(stage), 5);
            compute_encoder.set_bytes(&batch, sizeof(batch), 6);

            uint32_t total_threads = batch * (N / 2);
            compute_encoder.dispatch_threads(
                {total_threads, 1, 1},
                {256, 1, 1});
        }

        // Apply inverse N scaling
        auto scale_kernel = d.get_kernel("ntt_scale");
        compute_encoder.set_compute_pipeline_state(scale_kernel);
        uint32_t size = batch * N;
        compute_encoder.set_bytes(&ctx.Q, sizeof(ctx.Q), 1);
        compute_encoder.set_bytes(&ctx.inv_N, sizeof(ctx.inv_N), 2);
        compute_encoder.set_bytes(&size, sizeof(size), 3);
        compute_encoder.dispatch_threads({size, 1, 1}, {256, 1, 1});
    }

    d.commit_command_buffer(s.stream().index);
    return out;
#else
    throw std::runtime_error("Metal not available");
#endif
}

array pointwise_mul(
    const NTTContext& ctx,
    const array& a,
    const array& b,
    StreamOrDevice s) {
#ifdef MLX_BUILD_METAL
    if (!metal::is_available()) {
        throw std::runtime_error("Metal not available for NTT");
    }

    auto& d = metal::device(s.stream().device);
    auto& compute_encoder = d.get_command_encoder(s.stream().index);

    array out = array(a.shape(), a.dtype(), nullptr, {});
    out.set_data(allocator::malloc(out.nbytes()));

    auto kernel = d.get_kernel("ntt_pointwise_mul");
    compute_encoder.set_compute_pipeline_state(kernel);

    uint32_t size = a.size();
    compute_encoder.set_output_array(out, 0);
    compute_encoder.set_input_array(a, 1);
    compute_encoder.set_input_array(b, 2);
    compute_encoder.set_bytes(&ctx.Q, sizeof(ctx.Q), 3);
    compute_encoder.set_bytes(&ctx.barrett_mu, sizeof(ctx.barrett_mu), 4);
    compute_encoder.set_bytes(&size, sizeof(size), 5);

    compute_encoder.dispatch_threads({size, 1, 1}, {256, 1, 1});

    d.commit_command_buffer(s.stream().index);
    return out;
#else
    throw std::runtime_error("Metal not available");
#endif
}

array polymul(
    const NTTContext& ctx,
    const array& a,
    const array& b,
    StreamOrDevice s) {
    // Polynomial multiplication via NTT convolution
    array ntt_a = forward(ctx, a, s);
    array ntt_b = forward(ctx, b, s);
    array ntt_result = pointwise_mul(ctx, ntt_a, ntt_b, s);
    return inverse(ctx, ntt_result, s);
}

} // namespace mlx::core::ntt
