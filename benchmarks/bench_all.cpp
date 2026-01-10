// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Comprehensive Benchmark Suite
// Runs all benchmarks across all backends and outputs comparison tables.

#include "bench_common.hpp"
#include <cstdio>
#include <map>
#include <memory>
#include <ctime>

using namespace bench;

// =============================================================================
// Forward Declarations from Individual Benchmarks
// =============================================================================

// We inline essential benchmarks here to have a single-file comprehensive runner
// For full suite, compile individual bench_*.cpp files

// =============================================================================
// Quick Benchmark Classes (Subset of Full Benchmarks)
// =============================================================================

struct QuickMatmulBench {
    int M, N, K;
    BenchResult result;

    QuickMatmulBench(int m, int n, int k) : M(m), N(n), K(k) {}

    std::string size_str() const {
        char buf[32];
        snprintf(buf, sizeof(buf), "%d", M);
        return buf;
    }

    void run(LuxBackend backend) {
        result.operation = "matmul";
        result.size = size_str();
        result.backend = backend;
        result.throughput_unit = "GFLOPS";

        LuxGPU* gpu = lux_gpu_create_with_backend(backend);
        if (!gpu) {
            result.success = false;
            return;
        }

        const int64_t shape_a[] = {M, K};
        const int64_t shape_b[] = {K, N};

        std::vector<float> h_a(M * K);
        std::vector<float> h_b(K * N);
        fill_random_float(h_a.data(), M * K);
        fill_random_float(h_b.data(), K * N);

        LuxTensor* a = lux_tensor_from_data(gpu, h_a.data(), shape_a, 2, LUX_FLOAT32);
        LuxTensor* b = lux_tensor_from_data(gpu, h_b.data(), shape_b, 2, LUX_FLOAT32);

        if (!a || !b) {
            result.success = false;
            lux_gpu_destroy(gpu);
            return;
        }

        LuxTensor* c = nullptr;

        std::vector<double> times;
        for (int i = 0; i < 3; i++) {  // Warmup
            if (c) lux_tensor_destroy(c);
            c = lux_tensor_matmul(gpu, a, b);
            lux_gpu_sync(gpu);
        }

        for (int i = 0; i < 10; i++) {
            if (c) lux_tensor_destroy(c);
            Timer t;
            t.start();
            c = lux_tensor_matmul(gpu, a, b);
            lux_gpu_sync(gpu);
            times.push_back(t.elapsed_ms());
        }

        result.stats.compute(times);
        result.success = true;
        result.throughput = compute_gflops(gemm_flops(M, N, K), result.stats.median_ms);

        if (c) lux_tensor_destroy(c);
        lux_tensor_destroy(a);
        lux_tensor_destroy(b);
        lux_gpu_destroy(gpu);
    }
};

struct QuickNTTBench {
    size_t n;
    BenchResult result;

    explicit QuickNTTBench(size_t size) : n(size) {}

    std::string size_str() const {
        for (size_t i = 8; i <= 20; i++) {
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
        result.operation = "ntt";
        result.size = size_str();
        result.backend = backend;
        result.throughput_unit = "us";

        LuxGPU* gpu = lux_gpu_create_with_backend(backend);
        if (!gpu) {
            result.success = false;
            return;
        }

        std::vector<uint64_t> data(n);
        std::vector<uint64_t> work(n);
        uint64_t modulus = 0xFFFFFFFF00000001ULL;
        fill_random_uint64(data.data(), n, modulus);

        std::vector<double> times;
        for (int i = 0; i < 3; i++) {
            std::memcpy(work.data(), data.data(), n * sizeof(uint64_t));
            lux_ntt_forward(gpu, work.data(), n, modulus);
            lux_gpu_sync(gpu);
        }

        for (int i = 0; i < 10; i++) {
            std::memcpy(work.data(), data.data(), n * sizeof(uint64_t));
            Timer t;
            t.start();
            lux_ntt_forward(gpu, work.data(), n, modulus);
            lux_gpu_sync(gpu);
            times.push_back(t.elapsed_ms());
        }

        result.stats.compute(times);
        result.success = true;
        result.throughput = result.stats.median_ms * 1000.0;  // microseconds

        lux_gpu_destroy(gpu);
    }
};

struct QuickElementwiseBench {
    size_t n;
    BenchResult result;

    explicit QuickElementwiseBench(size_t size) : n(size) {}

    std::string size_str() const {
        if (n >= 1024 * 1024) {
            char buf[16];
            snprintf(buf, sizeof(buf), "%zuM", n / (1024 * 1024));
            return buf;
        }
        char buf[32];
        snprintf(buf, sizeof(buf), "%zu", n);
        return buf;
    }

    void run(LuxBackend backend) {
        result.operation = "add";
        result.size = size_str();
        result.backend = backend;
        result.throughput_unit = "GB/s";

        LuxGPU* gpu = lux_gpu_create_with_backend(backend);
        if (!gpu) {
            result.success = false;
            return;
        }

        const int64_t shape[] = {static_cast<int64_t>(n)};

        std::vector<float> h_a(n);
        std::vector<float> h_b(n);
        fill_random_float(h_a.data(), n);
        fill_random_float(h_b.data(), n);

        LuxTensor* a = lux_tensor_from_data(gpu, h_a.data(), shape, 1, LUX_FLOAT32);
        LuxTensor* b = lux_tensor_from_data(gpu, h_b.data(), shape, 1, LUX_FLOAT32);

        if (!a || !b) {
            result.success = false;
            lux_gpu_destroy(gpu);
            return;
        }

        LuxTensor* c = nullptr;

        std::vector<double> times;
        for (int i = 0; i < 3; i++) {
            if (c) lux_tensor_destroy(c);
            c = lux_tensor_add(gpu, a, b);
            lux_gpu_sync(gpu);
        }

        for (int i = 0; i < 10; i++) {
            if (c) lux_tensor_destroy(c);
            Timer t;
            t.start();
            c = lux_tensor_add(gpu, a, b);
            lux_gpu_sync(gpu);
            times.push_back(t.elapsed_ms());
        }

        result.stats.compute(times);
        result.success = true;
        result.throughput = compute_gbs(3 * n * sizeof(float), result.stats.median_ms);

        if (c) lux_tensor_destroy(c);
        lux_tensor_destroy(a);
        lux_tensor_destroy(b);
        lux_gpu_destroy(gpu);
    }
};

struct QuickReduceBench {
    size_t n;
    BenchResult result;

    explicit QuickReduceBench(size_t size) : n(size) {}

    std::string size_str() const {
        if (n >= 1024 * 1024) {
            char buf[16];
            snprintf(buf, sizeof(buf), "%zuM", n / (1024 * 1024));
            return buf;
        }
        char buf[32];
        snprintf(buf, sizeof(buf), "%zu", n);
        return buf;
    }

    void run(LuxBackend backend) {
        result.operation = "reduce_sum";
        result.size = size_str();
        result.backend = backend;
        result.throughput_unit = "GB/s";

        LuxGPU* gpu = lux_gpu_create_with_backend(backend);
        if (!gpu) {
            result.success = false;
            return;
        }

        const int64_t shape[] = {static_cast<int64_t>(n)};

        std::vector<float> h_data(n);
        fill_random_float(h_data.data(), n);

        LuxTensor* data = lux_tensor_from_data(gpu, h_data.data(), shape, 1, LUX_FLOAT32);

        if (!data) {
            result.success = false;
            lux_gpu_destroy(gpu);
            return;
        }

        LuxTensor* out = nullptr;
        int axes[] = {0};

        std::vector<double> times;
        for (int i = 0; i < 3; i++) {
            if (out) lux_tensor_destroy(out);
            out = lux_tensor_sum(gpu, data, axes, 1);
            lux_gpu_sync(gpu);
        }

        for (int i = 0; i < 10; i++) {
            if (out) lux_tensor_destroy(out);
            Timer t;
            t.start();
            out = lux_tensor_sum(gpu, data, axes, 1);
            lux_gpu_sync(gpu);
            times.push_back(t.elapsed_ms());
        }

        result.stats.compute(times);
        result.success = true;
        result.throughput = compute_gbs(n * sizeof(float), result.stats.median_ms);

        if (out) lux_tensor_destroy(out);
        lux_tensor_destroy(data);
        lux_gpu_destroy(gpu);
    }
};

struct QuickMSMBench {
    size_t num_points;
    BenchResult result;

    explicit QuickMSMBench(size_t log_n) : num_points(1ULL << log_n) {}

    std::string size_str() const {
        for (size_t i = 10; i <= 24; i++) {
            if (num_points == (1ULL << i)) {
                char buf[16];
                snprintf(buf, sizeof(buf), "2^%zu", i);
                return buf;
            }
        }
        char buf[32];
        snprintf(buf, sizeof(buf), "%zu", num_points);
        return buf;
    }

    void run(LuxBackend backend) {
        result.operation = "msm";
        result.size = size_str();
        result.backend = backend;
        result.throughput_unit = "ms";

        LuxGPU* gpu = lux_gpu_create_with_backend(backend);
        if (!gpu) {
            result.success = false;
            return;
        }

        std::vector<uint8_t> scalars(num_points * 32);
        std::vector<uint8_t> points(num_points * 64);
        std::vector<uint8_t> msm_result(64);
        fill_random_bytes(scalars.data(), scalars.size());
        fill_random_bytes(points.data(), points.size());

        std::vector<double> times;
        for (int i = 0; i < 2; i++) {
            lux_msm(gpu, scalars.data(), points.data(), msm_result.data(),
                   num_points, LUX_CURVE_BN254);
            lux_gpu_sync(gpu);
        }

        for (int i = 0; i < 5; i++) {
            Timer t;
            t.start();
            lux_msm(gpu, scalars.data(), points.data(), msm_result.data(),
                   num_points, LUX_CURVE_BN254);
            lux_gpu_sync(gpu);
            times.push_back(t.elapsed_ms());
        }

        result.stats.compute(times);
        result.success = true;
        result.throughput = result.stats.median_ms;

        lux_gpu_destroy(gpu);
    }
};

// =============================================================================
// Result Aggregation
// =============================================================================

struct AllBenchResults {
    std::vector<BenchResult> results;

    void add(const BenchResult& r) {
        results.push_back(r);
    }

    void print_full_table() const {
        std::map<std::string, std::map<LuxBackend, BenchResult>> grouped;

        for (const auto& r : results) {
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
                if (r.operation == "msm" || r.throughput_unit == "ms") {
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
    }

    void print_summary() const {
        printf("\n=== Summary Statistics ===\n\n");

        std::map<LuxBackend, int> success_count;
        std::map<LuxBackend, int> total_count;
        std::map<LuxBackend, double> total_speedup;
        int comparison_count = 0;

        for (const auto& r : results) {
            total_count[r.backend]++;
            if (r.success) {
                success_count[r.backend]++;
            }
        }

        // Calculate speedup vs CPU for each backend
        std::map<std::string, std::map<LuxBackend, double>> throughputs;
        for (const auto& r : results) {
            if (r.success && r.throughput > 0) {
                std::string key = r.operation + "|" + r.size;
                throughputs[key][r.backend] = r.throughput;
            }
        }

        for (const auto& [key, backends] : throughputs) {
            auto cpu_it = backends.find(LUX_BACKEND_CPU);
            if (cpu_it == backends.end() || cpu_it->second <= 0) continue;
            double cpu_perf = cpu_it->second;

            for (const auto& [backend, perf] : backends) {
                if (backend != LUX_BACKEND_CPU && perf > 0) {
                    double speedup = perf / cpu_perf;
                    total_speedup[backend] += speedup;
                    comparison_count++;
                }
            }
        }

        LuxBackend all_backends[] = {LUX_BACKEND_CPU, LUX_BACKEND_METAL, LUX_BACKEND_CUDA, LUX_BACKEND_DAWN};
        const char* backend_names[] = {"CPU", "Metal", "CUDA", "WebGPU"};

        printf("Backend     | Tests Passed | Avg Speedup vs CPU\n");
        printf("------------|--------------|-------------------\n");

        for (int i = 0; i < 4; i++) {
            LuxBackend b = all_backends[i];
            int passed = success_count[b];
            int total = total_count[b];

            if (total == 0) {
                printf("%-11s | N/A          | N/A\n", backend_names[i]);
            } else if (b == LUX_BACKEND_CPU) {
                printf("%-11s | %d/%d         | 1.00x (baseline)\n", backend_names[i], passed, total);
            } else {
                double avg_speedup = (comparison_count > 0) ? (total_speedup[b] / (comparison_count / 3)) : 0;
                printf("%-11s | %d/%d         | %.2fx\n", backend_names[i], passed, total, avg_speedup);
            }
        }

        printf("\n");
    }
};

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    // Print header with timestamp
    time_t now = time(nullptr);
    printf("================================================================================\n");
    printf("                     Lux GPU Comprehensive Benchmark Suite\n");
    printf("================================================================================\n");
    printf("Date: %s", ctime(&now));
    printf("\n");

    // Detect backends
    auto backends = get_available_backends();
    printf("Backend Detection:\n");
    for (const auto& b : backends) {
        printf("  %-10s: %s\n", b.name, b.available ? "AVAILABLE" : "not found");
    }
    printf("\n");

    // Get device info for available backends
    printf("Device Information:\n");
    for (const auto& b : backends) {
        if (b.available) {
            LuxGPU* gpu = lux_gpu_create_with_backend(b.type);
            if (gpu) {
                LuxDeviceInfo info = {};
                lux_gpu_device_info(gpu, &info);
                printf("  %s:\n", b.name);
                printf("    Name: %s\n", info.name ? info.name : "Unknown");
                printf("    Vendor: %s\n", info.vendor ? info.vendor : "Unknown");
                if (info.memory_total > 0) {
                    printf("    Memory: %.1f GB\n", info.memory_total / (1024.0 * 1024 * 1024));
                }
                printf("    Compute Units: %d\n", info.compute_units);
                lux_gpu_destroy(gpu);
            }
        }
    }
    printf("\n");

    AllBenchResults all_results;

    printf("================================================================================\n");
    printf("                              Running Benchmarks\n");
    printf("================================================================================\n\n");

    // Matrix multiplication benchmarks
    printf("--- Matrix Multiplication ---\n");
    std::vector<int> matmul_sizes = {256, 1024, 4096};
    for (int sz : matmul_sizes) {
        printf("  %dx%dx%d: ", sz, sz, sz);
        for (const auto& b : backends) {
            if (b.available) {
                printf("%s ", b.name);
                fflush(stdout);
                QuickMatmulBench bench(sz, sz, sz);
                bench.run(b.type);
                all_results.add(bench.result);
            }
        }
        printf("\n");
    }

    // Elementwise benchmarks
    printf("\n--- Elementwise Operations ---\n");
    std::vector<size_t> elem_sizes = {1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024};
    for (size_t sz : elem_sizes) {
        QuickElementwiseBench bench(sz);
        printf("  add %s: ", bench.size_str().c_str());
        for (const auto& b : backends) {
            if (b.available) {
                printf("%s ", b.name);
                fflush(stdout);
                bench.run(b.type);
                all_results.add(bench.result);
            }
        }
        printf("\n");
    }

    // Reduction benchmarks
    printf("\n--- Reduction Operations ---\n");
    std::vector<size_t> reduce_sizes = {16 * 1024 * 1024, 64 * 1024 * 1024, 256 * 1024 * 1024};
    for (size_t sz : reduce_sizes) {
        QuickReduceBench bench(sz);
        printf("  sum %s: ", bench.size_str().c_str());
        for (const auto& b : backends) {
            if (b.available) {
                printf("%s ", b.name);
                fflush(stdout);
                bench.run(b.type);
                all_results.add(bench.result);
            }
        }
        printf("\n");
    }

    // NTT benchmarks
    printf("\n--- NTT Operations ---\n");
    std::vector<size_t> ntt_sizes = {1 << 12, 1 << 14, 1 << 16};
    for (size_t sz : ntt_sizes) {
        QuickNTTBench bench(sz);
        printf("  N=%s: ", bench.size_str().c_str());
        for (const auto& b : backends) {
            if (b.available) {
                printf("%s ", b.name);
                fflush(stdout);
                bench.run(b.type);
                all_results.add(bench.result);
            }
        }
        printf("\n");
    }

    // MSM benchmarks
    printf("\n--- Multi-Scalar Multiplication ---\n");
    std::vector<size_t> msm_sizes = {10, 14, 16};  // 2^n points
    for (size_t log_n : msm_sizes) {
        QuickMSMBench bench(log_n);
        printf("  2^%zu points: ", log_n);
        for (const auto& b : backends) {
            if (b.available) {
                printf("%s ", b.name);
                fflush(stdout);
                bench.run(b.type);
                all_results.add(bench.result);
            }
        }
        printf("\n");
    }

    // Print results
    printf("\n");
    printf("================================================================================\n");
    printf("                              Benchmark Results\n");
    printf("================================================================================\n\n");

    all_results.print_full_table();
    all_results.print_summary();

    printf("================================================================================\n");
    printf("                                  Completed\n");
    printf("================================================================================\n");

    return 0;
}
