// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// CPU Backend - SIMD-optimized fallback

#include "../../src/backend.hpp"
#include <cstring>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

class CPUBackend : public Backend {
public:
    CPUBackend() = default;
    ~CPUBackend() override = default;

    LuxBackend type() const override { return LUX_BACKEND_CPU; }
    const char* name() const override { return "cpu"; }

    LuxError sync() override { return LUX_OK; }

    LuxError device_info(LuxDeviceInfo* info) override {
        if (!info) return LUX_ERROR_INVALID_ARGUMENT;
        info->backend = LUX_BACKEND_CPU;
        info->index = 0;
        info->name = "CPU";
        info->vendor = "System";
        info->memory_total = 0;  // Unknown
        info->memory_available = 0;
        info->is_discrete = false;
        info->is_unified_memory = true;
#ifdef _OPENMP
        info->compute_units = omp_get_max_threads();
#else
        info->compute_units = 1;
#endif
        info->max_workgroup_size = 1;
        return LUX_OK;
    }

    LuxTensor* zeros(const int64_t* shape, int ndim, LuxDtype dtype) override {
        auto t = new LuxTensor();
        t->shape.assign(shape, shape + ndim);
        t->dtype = dtype;
        size_t bytes = t->size() * t->element_size();
        t->data.resize(bytes, 0);
        return t;
    }

    LuxTensor* ones(const int64_t* shape, int ndim, LuxDtype dtype) override {
        return full(shape, ndim, dtype, 1.0);
    }

    LuxTensor* full(const int64_t* shape, int ndim, LuxDtype dtype, double value) override {
        auto t = zeros(shape, ndim, dtype);
        if (dtype == LUX_FLOAT32) {
            float v = static_cast<float>(value);
            float* ptr = reinterpret_cast<float*>(t->data.data());
            for (int64_t i = 0; i < t->size(); i++) ptr[i] = v;
        }
        return t;
    }

    LuxTensor* from_data(const void* data, const int64_t* shape, int ndim, LuxDtype dtype) override {
        auto t = new LuxTensor();
        t->shape.assign(shape, shape + ndim);
        t->dtype = dtype;
        size_t bytes = t->size() * t->element_size();
        t->data.resize(bytes);
        std::memcpy(t->data.data(), data, bytes);
        return t;
    }

    void sync_to_device(LuxTensor*) override {
        // No-op for CPU - data is already in system memory
    }

    void destroy_tensor(LuxTensor* t) override {
        delete t;
    }

    LuxTensor* add(LuxTensor* a, LuxTensor* b) override {
        if (a->shape != b->shape || a->dtype != b->dtype) return nullptr;
        auto out = zeros(a->shape.data(), static_cast<int>(a->shape.size()), a->dtype);

        if (a->dtype == LUX_FLOAT32) {
            const float* pa = reinterpret_cast<const float*>(a->data.data());
            const float* pb = reinterpret_cast<const float*>(b->data.data());
            float* po = reinterpret_cast<float*>(out->data.data());
            int64_t n = a->size();

#ifdef _OPENMP
            #pragma omp parallel for simd
#endif
            for (int64_t i = 0; i < n; i++) {
                po[i] = pa[i] + pb[i];
            }
        }
        return out;
    }

    LuxTensor* sub(LuxTensor* a, LuxTensor* b) override {
        if (a->shape != b->shape || a->dtype != b->dtype) return nullptr;
        auto out = zeros(a->shape.data(), static_cast<int>(a->shape.size()), a->dtype);

        if (a->dtype == LUX_FLOAT32) {
            const float* pa = reinterpret_cast<const float*>(a->data.data());
            const float* pb = reinterpret_cast<const float*>(b->data.data());
            float* po = reinterpret_cast<float*>(out->data.data());
            int64_t n = a->size();

#ifdef _OPENMP
            #pragma omp parallel for simd
#endif
            for (int64_t i = 0; i < n; i++) {
                po[i] = pa[i] - pb[i];
            }
        }
        return out;
    }

    LuxTensor* mul(LuxTensor* a, LuxTensor* b) override {
        if (a->shape != b->shape || a->dtype != b->dtype) return nullptr;
        auto out = zeros(a->shape.data(), static_cast<int>(a->shape.size()), a->dtype);

        if (a->dtype == LUX_FLOAT32) {
            const float* pa = reinterpret_cast<const float*>(a->data.data());
            const float* pb = reinterpret_cast<const float*>(b->data.data());
            float* po = reinterpret_cast<float*>(out->data.data());
            int64_t n = a->size();

#ifdef _OPENMP
            #pragma omp parallel for simd
#endif
            for (int64_t i = 0; i < n; i++) {
                po[i] = pa[i] * pb[i];
            }
        }
        return out;
    }

    LuxTensor* matmul(LuxTensor* a, LuxTensor* b) override {
        if (a->shape.size() != 2 || b->shape.size() != 2) return nullptr;
        if (a->shape[1] != b->shape[0]) return nullptr;

        int64_t M = a->shape[0];
        int64_t K = a->shape[1];
        int64_t N = b->shape[1];

        int64_t out_shape[2] = {M, N};
        auto out = zeros(out_shape, 2, a->dtype);

        if (a->dtype == LUX_FLOAT32) {
            const float* pa = reinterpret_cast<const float*>(a->data.data());
            const float* pb = reinterpret_cast<const float*>(b->data.data());
            float* po = reinterpret_cast<float*>(out->data.data());

#ifdef _OPENMP
            #pragma omp parallel for collapse(2)
#endif
            for (int64_t i = 0; i < M; i++) {
                for (int64_t j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (int64_t k = 0; k < K; k++) {
                        sum += pa[i * K + k] * pb[k * N + j];
                    }
                    po[i * N + j] = sum;
                }
            }
        }
        return out;
    }

    LuxError bls_verify_batch(const uint8_t* const*, const size_t*,
                              const uint8_t* const*, const size_t*,
                              const uint8_t* const*, const size_t*,
                              int, bool*) override {
        return LUX_ERROR_NOT_SUPPORTED;  // Needs crypto library
    }

    LuxError ntt_forward(uint64_t* data, size_t n, uint64_t modulus) override {
        // Cooley-Tukey NTT (decimation-in-time)
        if (!data || n == 0 || (n & (n - 1)) != 0) return LUX_ERROR_INVALID_ARGUMENT;

        // Find primitive generator and compute n-th root of unity
        uint64_t g = find_primitive_root(n, modulus);
        if (g == 0) return LUX_ERROR_INVALID_ARGUMENT;

        // omega_n = g^((p-1)/n) is the principal n-th root of unity
        uint64_t omega_n = mod_pow(g, (modulus - 1) / n, modulus);

        // Bit-reversal permutation
        bit_reverse(data, n);

        // Butterfly operations
        for (size_t len = 2; len <= n; len *= 2) {
            // w = omega_n^(n/len) is the len-th root of unity
            uint64_t w = mod_pow(omega_n, n / len, modulus);
            for (size_t i = 0; i < n; i += len) {
                uint64_t wn = 1;
                for (size_t j = 0; j < len / 2; j++) {
                    uint64_t u = data[i + j];
                    uint64_t t = mod_mul(wn, data[i + j + len / 2], modulus);
                    data[i + j] = mod_add(u, t, modulus);
                    data[i + j + len / 2] = mod_sub(u, t, modulus);
                    wn = mod_mul(wn, w, modulus);
                }
            }
        }
        return LUX_OK;
    }

    LuxError ntt_inverse(uint64_t* data, size_t n, uint64_t modulus) override {
        if (!data || n == 0 || (n & (n - 1)) != 0) return LUX_ERROR_INVALID_ARGUMENT;

        // Compute inverse n-th root of unity
        uint64_t g = find_primitive_root(n, modulus);
        uint64_t omega_n = mod_pow(g, (modulus - 1) / n, modulus);
        uint64_t omega_n_inv = mod_pow(omega_n, modulus - 2, modulus);

        // Decimation-in-frequency butterfly (large to small)
        for (size_t len = n; len >= 2; len /= 2) {
            uint64_t w = mod_pow(omega_n_inv, n / len, modulus);
            for (size_t i = 0; i < n; i += len) {
                uint64_t wn = 1;
                for (size_t j = 0; j < len / 2; j++) {
                    uint64_t u = data[i + j];
                    uint64_t v = data[i + j + len / 2];
                    data[i + j] = mod_add(u, v, modulus);
                    data[i + j + len / 2] = mod_mul(mod_sub(u, v, modulus), wn, modulus);
                    wn = mod_mul(wn, w, modulus);
                }
            }
        }

        // Bit-reversal at end
        bit_reverse(data, n);

        // Scale by n^-1
        uint64_t n_inv = mod_pow(n, modulus - 2, modulus);
        for (size_t i = 0; i < n; i++) {
            data[i] = mod_mul(data[i], n_inv, modulus);
        }
        return LUX_OK;
    }

private:
    static uint64_t mod_add(uint64_t a, uint64_t b, uint64_t m) {
        // Use 128-bit addition to handle overflow
        return ((__uint128_t)a + b) % m;
    }

    static uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t m) {
        // Safe subtraction that handles underflow
        a %= m;
        b %= m;
        return (a >= b) ? (a - b) : (m - (b - a));
    }

    static uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t m) {
        return ((__uint128_t)a * b) % m;
    }

    static uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t m) {
        uint64_t result = 1;
        base %= m;
        while (exp > 0) {
            if (exp & 1) result = mod_mul(result, base, m);
            exp >>= 1;
            base = mod_mul(base, base, m);
        }
        return result;
    }

    static void bit_reverse(uint64_t* data, size_t n) {
        size_t j = 0;
        for (size_t i = 0; i < n; i++) {
            if (i < j) std::swap(data[i], data[j]);
            size_t m = n >> 1;
            while (m >= 1 && j >= m) {
                j -= m;
                m >>= 1;
            }
            j += m;
        }
    }

    static uint64_t find_primitive_root(size_t n, uint64_t m) {
        // For standard NTT primes, return known primitive root
        // This is simplified - real implementation needs proper generator finding
        if (m == 0xFFFFFFFF00000001ULL) return 7;  // Goldilocks prime
        if (m == 0x1000000000000001ULL) return 3;  // Another NTT-friendly prime
        return 3;  // Default guess
    }
};

// Factory function
std::unique_ptr<Backend> create_cpu_backend() {
    return std::make_unique<CPUBackend>();
}
