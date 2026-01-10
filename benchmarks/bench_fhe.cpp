// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// FHE Operation Benchmarks
// Tests NTT, TFHE bootstrap, and polynomial operations across all backends.

#include "bench_common.hpp"
#include <cstdio>
#include <map>
#include <memory>

using namespace bench;

// =============================================================================
// NTT Parameters
// =============================================================================

// Common NTT primes for FHE
constexpr uint64_t GOLDILOCKS_PRIME = 0xFFFFFFFF00000001ULL;  // 2^64 - 2^32 + 1
constexpr uint64_t NTT_PRIME_60 = 0x1000000000000001ULL;     // 2^60 + 1 approx

// =============================================================================
// NTT Benchmark
// =============================================================================

struct NTTBench {
    size_t n;
    uint64_t modulus;
    bool is_inverse;
    std::vector<BenchResult> results;

    NTTBench(size_t size, uint64_t mod, bool inverse = false)
        : n(size), modulus(mod), is_inverse(inverse) {}

    std::string size_str() const {
        for (size_t i = 8; i <= 24; i++) {
            if (n == (1ULL << i)) {
                char buf[16];
                snprintf(buf, sizeof(buf), "2^%zu", i);
                return buf;
            }
        }
        char buf[32];
        snprintf(buf, sizeof(buf), "%zu", n);
        return buf;
    }

    std::string op_name() const {
        return is_inverse ? "intt" : "ntt";
    }

    void run(LuxBackend backend) {
        BenchResult result;
        result.operation = op_name();
        result.size = size_str();
        result.backend = backend;
        result.throughput_unit = "us";

        if (!is_power_of_two(n)) {
            result.success = false;
            result.error_msg = "N must be power of 2";
            results.push_back(result);
            return;
        }

        LuxGPU* gpu = lux_gpu_create_with_backend(backend);
        if (!gpu) {
            result.success = false;
            result.error_msg = "Backend unavailable";
            results.push_back(result);
            return;
        }

        // Allocate polynomial data
        std::vector<uint64_t> data(n);
        fill_random_uint64(data.data(), n, modulus);

        // Make a copy for each iteration
        std::vector<uint64_t> work(n);

        auto setup = [&]() {
            std::memcpy(work.data(), data.data(), n * sizeof(uint64_t));
        };

        auto kernel = [&]() {
            LuxError err;
            if (is_inverse) {
                err = lux_ntt_inverse(gpu, work.data(), n, modulus);
            } else {
                err = lux_ntt_forward(gpu, work.data(), n, modulus);
            }
            (void)err;
        };

        auto sync = [&]() {
            lux_gpu_sync(gpu);
        };

        try {
            Stats stats = run_benchmark(setup, kernel, [](){}, sync, 3, 10);
            result.stats = stats;
            result.success = true;
            // Report time in microseconds for NTT
            result.throughput = stats.median_ms * 1000.0;
        } catch (...) {
            result.success = false;
            result.error_msg = "Exception during benchmark";
        }

        lux_gpu_destroy(gpu);
        results.push_back(result);
    }
};

// =============================================================================
// NTT Batch Benchmark
// =============================================================================

struct NTTBatchBench {
    size_t n;
    size_t batch_size;
    uint64_t modulus;
    std::vector<BenchResult> results;

    NTTBatchBench(size_t size, size_t batch, uint64_t mod)
        : n(size), batch_size(batch), modulus(mod) {}

    std::string size_str() const {
        char buf[64];
        snprintf(buf, sizeof(buf), "%zu x %zu", n, batch_size);
        return buf;
    }

    void run(LuxBackend backend) {
        BenchResult result;
        result.operation = "ntt_batch";
        result.size = size_str();
        result.backend = backend;
        result.throughput_unit = "GNTT/s";

        if (!is_power_of_two(n)) {
            result.success = false;
            result.error_msg = "N must be power of 2";
            results.push_back(result);
            return;
        }

        LuxGPU* gpu = lux_gpu_create_with_backend(backend);
        if (!gpu) {
            result.success = false;
            result.error_msg = "Backend unavailable";
            results.push_back(result);
            return;
        }

        // Allocate batch of polynomials
        std::vector<std::vector<uint64_t>> polys(batch_size);
        std::vector<uint64_t*> poly_ptrs(batch_size);

        for (size_t i = 0; i < batch_size; i++) {
            polys[i].resize(n);
            fill_random_uint64(polys[i].data(), n, modulus);
            poly_ptrs[i] = polys[i].data();
        }

        auto kernel = [&]() {
            LuxError err = lux_ntt_batch(gpu, poly_ptrs.data(), batch_size, n, modulus);
            (void)err;
        };

        auto sync = [&]() {
            lux_gpu_sync(gpu);
        };

        try {
            Stats stats = run_benchmark([](){}, kernel, [](){}, sync, 2, 5);
            result.stats = stats;
            result.success = true;
            // NTTs per second (in billions)
            result.throughput = (static_cast<double>(batch_size) / stats.median_ms) * 1000.0 / 1e9;
        } catch (...) {
            result.success = false;
            result.error_msg = "Exception during benchmark";
        }

        lux_gpu_destroy(gpu);
        results.push_back(result);
    }
};

// =============================================================================
// Polynomial Multiplication Benchmark (via NTT)
// =============================================================================

struct PolyMulBench {
    size_t n;
    uint64_t modulus;
    std::vector<BenchResult> results;

    PolyMulBench(size_t size, uint64_t mod) : n(size), modulus(mod) {}

    std::string size_str() const {
        for (size_t i = 8; i <= 24; i++) {
            if (n == (1ULL << i)) {
                char buf[16];
                snprintf(buf, sizeof(buf), "2^%zu", i);
                return buf;
            }
        }
        char buf[32];
        snprintf(buf, sizeof(buf), "%zu", n);
        return buf;
    }

    void run(LuxBackend backend) {
        BenchResult result;
        result.operation = "poly_mul";
        result.size = size_str();
        result.backend = backend;
        result.throughput_unit = "us";

        if (!is_power_of_two(n)) {
            result.success = false;
            result.error_msg = "N must be power of 2";
            results.push_back(result);
            return;
        }

        LuxGPU* gpu = lux_gpu_create_with_backend(backend);
        if (!gpu) {
            result.success = false;
            result.error_msg = "Backend unavailable";
            results.push_back(result);
            return;
        }

        // Two polynomials to multiply
        std::vector<uint64_t> poly_a(n);
        std::vector<uint64_t> poly_b(n);
        std::vector<uint64_t> poly_c(n);

        fill_random_uint64(poly_a.data(), n, modulus);
        fill_random_uint64(poly_b.data(), n, modulus);

        // NTT-based multiplication: NTT(a), NTT(b), pointwise mul, INTT
        auto kernel = [&]() {
            // Forward NTT on both
            lux_ntt_forward(gpu, poly_a.data(), n, modulus);
            lux_ntt_forward(gpu, poly_b.data(), n, modulus);

            // Pointwise multiplication (modular)
            for (size_t i = 0; i < n; i++) {
                __uint128_t prod = static_cast<__uint128_t>(poly_a[i]) * poly_b[i];
                poly_c[i] = prod % modulus;
            }

            // Inverse NTT
            lux_ntt_inverse(gpu, poly_c.data(), n, modulus);
        };

        auto sync = [&]() {
            lux_gpu_sync(gpu);
        };

        // Reset data each iteration
        auto setup = [&]() {
            fill_random_uint64(poly_a.data(), n, modulus);
            fill_random_uint64(poly_b.data(), n, modulus);
        };

        try {
            Stats stats = run_benchmark(setup, kernel, [](){}, sync, 2, 5);
            result.stats = stats;
            result.success = true;
            result.throughput = stats.median_ms * 1000.0;  // microseconds
        } catch (...) {
            result.success = false;
            result.error_msg = "Exception during benchmark";
        }

        lux_gpu_destroy(gpu);
        results.push_back(result);
    }
};

// =============================================================================
// TFHE Bootstrap Benchmark (Simulated)
// =============================================================================

struct TFHEBootstrapBench {
    size_t n;  // LWE dimension
    size_t N;  // GLWE dimension
    std::vector<BenchResult> results;

    TFHEBootstrapBench(size_t lwe_dim, size_t glwe_dim) : n(lwe_dim), N(glwe_dim) {}

    std::string size_str() const {
        char buf[32];
        snprintf(buf, sizeof(buf), "%zu/%zu", n, N);
        return buf;
    }

    void run(LuxBackend backend) {
        BenchResult result;
        result.operation = "tfhe_boot";
        result.size = size_str();
        result.backend = backend;
        result.throughput_unit = "ms";

        LuxGPU* gpu = lux_gpu_create_with_backend(backend);
        if (!gpu) {
            result.success = false;
            result.error_msg = "Backend unavailable";
            results.push_back(result);
            return;
        }

        // TFHE bootstrap consists of:
        // 1. Blind rotation (n NTT-based polynomial multiplications)
        // 2. Sample extraction
        // 3. Key switching

        // Simulate with NTT operations (simplified)
        std::vector<uint64_t> accumulator(N);
        std::vector<uint64_t> bsk_poly(N);  // One BSK polynomial
        fill_random_uint64(accumulator.data(), N, GOLDILOCKS_PRIME);
        fill_random_uint64(bsk_poly.data(), N, GOLDILOCKS_PRIME);

        auto kernel = [&]() {
            // Simulate blind rotation: n external products
            // Each external product = 2 NTTs + pointwise mul + INTT
            for (size_t i = 0; i < n; i++) {
                lux_ntt_forward(gpu, accumulator.data(), N, GOLDILOCKS_PRIME);
                lux_ntt_forward(gpu, bsk_poly.data(), N, GOLDILOCKS_PRIME);

                // Pointwise multiply
                for (size_t j = 0; j < N; j++) {
                    __uint128_t prod = static_cast<__uint128_t>(accumulator[j]) * bsk_poly[j];
                    accumulator[j] = prod % GOLDILOCKS_PRIME;
                }

                lux_ntt_inverse(gpu, accumulator.data(), N, GOLDILOCKS_PRIME);
            }
        };

        auto sync = [&]() {
            lux_gpu_sync(gpu);
        };

        try {
            // Fewer iterations due to long runtime
            Stats stats = run_benchmark([](){}, kernel, [](){}, sync, 1, 3);
            result.stats = stats;
            result.success = true;
            result.throughput = stats.median_ms;  // Report raw bootstrap time
        } catch (...) {
            result.success = false;
            result.error_msg = "Exception during benchmark";
        }

        lux_gpu_destroy(gpu);
        results.push_back(result);
    }
};

// =============================================================================
// Result Collection
// =============================================================================

struct FHEBenchResults {
    std::vector<BenchResult> all_results;

    void add(const std::vector<BenchResult>& results) {
        for (const auto& r : results) {
            all_results.push_back(r);
        }
    }

    void print_table() const {
        std::map<std::string, std::map<LuxBackend, BenchResult>> grouped;

        for (const auto& r : all_results) {
            std::string key = r.operation + "|" + r.size;
            grouped[key][r.backend] = r;
        }

        print_table_header();

        for (const auto& [key, backends] : grouped) {
            size_t sep = key.find('|');
            std::string op = key.substr(0, sep);
            std::string size = key.substr(sep + 1);

            TableRow row;
            row.operation = op;
            row.size = size;

            auto format_result = [](const BenchResult& r) -> std::string {
                if (!r.success) return "N/A";
                // For time-based metrics, format appropriately
                if (r.throughput_unit == "ms") {
                    return format_time(r.throughput);
                } else if (r.throughput_unit == "us") {
                    char buf[32];
                    snprintf(buf, sizeof(buf), "%.1f us", r.throughput);
                    return buf;
                }
                return format_throughput(r.throughput, r.throughput_unit.c_str());
            };

            auto it = backends.find(LUX_BACKEND_CPU);
            row.cpu = (it != backends.end()) ? format_result(it->second) : "N/A";

            it = backends.find(LUX_BACKEND_METAL);
            row.metal = (it != backends.end()) ? format_result(it->second) : "N/A";

            it = backends.find(LUX_BACKEND_CUDA);
            row.cuda = (it != backends.end()) ? format_result(it->second) : "N/A";

            it = backends.find(LUX_BACKEND_DAWN);
            row.webgpu = (it != backends.end()) ? format_result(it->second) : "N/A";

            print_table_row(row);
        }

        printf("\n");
    }
};

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    printf("=== Lux GPU FHE Benchmarks ===\n\n");

    auto backends = get_available_backends();
    printf("Available backends:\n");
    for (const auto& b : backends) {
        printf("  %s: %s\n", b.name, b.available ? "yes" : "no");
    }
    printf("\n");

    FHEBenchResults results;

    // NTT sizes
    std::vector<size_t> ntt_sizes = {
        1 << 10,   // 1024
        1 << 12,   // 4096
        1 << 14,   // 16384
        1 << 16,   // 65536
        1 << 18,   // 262144
    };

    printf("Running forward NTT benchmarks...\n");
    for (size_t n : ntt_sizes) {
        NTTBench bench(n, GOLDILOCKS_PRIME, false);
        printf("  N=%s: ", bench.size_str().c_str());
        for (const auto& b : backends) {
            if (b.available) {
                printf("%s ", b.name);
                fflush(stdout);
                bench.run(b.type);
            }
        }
        printf("\n");
        results.add(bench.results);
    }

    printf("\nRunning inverse NTT benchmarks...\n");
    for (size_t n : ntt_sizes) {
        NTTBench bench(n, GOLDILOCKS_PRIME, true);
        printf("  N=%s: ", bench.size_str().c_str());
        for (const auto& b : backends) {
            if (b.available) {
                printf("%s ", b.name);
                fflush(stdout);
                bench.run(b.type);
            }
        }
        printf("\n");
        results.add(bench.results);
    }

    // Polynomial multiplication
    printf("\nRunning polynomial multiplication benchmarks...\n");
    for (size_t n : ntt_sizes) {
        PolyMulBench bench(n, GOLDILOCKS_PRIME);
        printf("  N=%s: ", bench.size_str().c_str());
        for (const auto& b : backends) {
            if (b.available) {
                printf("%s ", b.name);
                fflush(stdout);
                bench.run(b.type);
            }
        }
        printf("\n");
        results.add(bench.results);
    }

    // Batch NTT
    std::vector<std::tuple<size_t, size_t>> batch_configs = {
        {4096, 64},
        {4096, 256},
        {16384, 16},
        {16384, 64},
    };

    printf("\nRunning batch NTT benchmarks...\n");
    for (auto& [n, batch] : batch_configs) {
        NTTBatchBench bench(n, batch, GOLDILOCKS_PRIME);
        printf("  %s: ", bench.size_str().c_str());
        for (const auto& b : backends) {
            if (b.available) {
                printf("%s ", b.name);
                fflush(stdout);
                bench.run(b.type);
            }
        }
        printf("\n");
        results.add(bench.results);
    }

    // TFHE bootstrap (simulated)
    std::vector<std::tuple<size_t, size_t>> tfhe_configs = {
        {630, 1024},    // TFHE-rs default parameters
        {722, 2048},    // Higher security
    };

    printf("\nRunning TFHE bootstrap benchmarks (simulated)...\n");
    for (auto& [n, N] : tfhe_configs) {
        TFHEBootstrapBench bench(n, N);
        printf("  n=%zu, N=%zu: ", n, N);
        for (const auto& b : backends) {
            if (b.available) {
                printf("%s ", b.name);
                fflush(stdout);
                bench.run(b.type);
            }
        }
        printf("\n");
        results.add(bench.results);
    }

    printf("\n\n");
    printf("=== Results ===\n");
    results.print_table();

    return 0;
}
