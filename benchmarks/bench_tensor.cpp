// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Tensor Operation Benchmarks
// Tests matmul, elementwise, and reduction operations across all backends.

#include "bench_common.hpp"
#include <cstdio>
#include <map>
#include <memory>

using namespace bench;

// =============================================================================
// Matrix Multiplication Benchmark
// =============================================================================

struct MatmulBench {
    int M, N, K;
    std::vector<BenchResult> results;

    MatmulBench(int m, int n, int k) : M(m), N(n), K(k) {}

    std::string size_str() const {
        char buf[64];
        if (M == N && N == K) {
            snprintf(buf, sizeof(buf), "%d", M);
        } else {
            snprintf(buf, sizeof(buf), "%dx%dx%d", M, N, K);
        }
        return buf;
    }

    void run(LuxBackend backend) {
        BenchResult result;
        result.operation = "matmul";
        result.size = size_str();
        result.backend = backend;
        result.throughput_unit = "GFLOPS";

        LuxGPU* gpu = lux_gpu_create_with_backend(backend);
        if (!gpu) {
            result.success = false;
            result.error_msg = "Backend unavailable";
            results.push_back(result);
            return;
        }

        // Allocate tensors
        const int64_t shape_a[] = {M, K};
        const int64_t shape_b[] = {K, N};
        const int64_t shape_c[] = {M, N};

        // Allocate host data
        std::vector<float> h_a(M * K);
        std::vector<float> h_b(K * N);
        fill_random_float(h_a.data(), M * K);
        fill_random_float(h_b.data(), K * N);

        LuxTensor* a = lux_tensor_from_data(gpu, h_a.data(), shape_a, 2, LUX_FLOAT32);
        LuxTensor* b = lux_tensor_from_data(gpu, h_b.data(), shape_b, 2, LUX_FLOAT32);

        if (!a || !b) {
            result.success = false;
            result.error_msg = "Tensor allocation failed";
            lux_gpu_destroy(gpu);
            results.push_back(result);
            return;
        }

        // Pre-allocate result tensor for reuse
        LuxTensor* c = nullptr;

        auto kernel = [&]() {
            if (c) lux_tensor_destroy(c);
            c = lux_tensor_matmul(gpu, a, b);
        };

        auto sync = [&]() {
            lux_gpu_sync(gpu);
        };

        auto teardown = [&]() {
            // Result tensor destroyed in kernel
        };

        try {
            Stats stats = run_benchmark([](){}, kernel, teardown, sync);
            result.stats = stats;
            result.success = true;

            double flops = gemm_flops(M, N, K);
            result.throughput = compute_gflops(flops, stats.median_ms);
        } catch (...) {
            result.success = false;
            result.error_msg = "Exception during benchmark";
        }

        // Cleanup
        if (c) lux_tensor_destroy(c);
        lux_tensor_destroy(a);
        lux_tensor_destroy(b);
        lux_gpu_destroy(gpu);

        results.push_back(result);
    }
};

// =============================================================================
// Elementwise Operation Benchmark
// =============================================================================

struct ElementwiseBench {
    size_t n;
    std::string op_name;
    std::vector<BenchResult> results;

    ElementwiseBench(size_t size, const char* op) : n(size), op_name(op) {}

    std::string size_str() const {
        char buf[32];
        if (n >= 1024 * 1024) {
            snprintf(buf, sizeof(buf), "%zuM", n / (1024 * 1024));
        } else if (n >= 1024) {
            snprintf(buf, sizeof(buf), "%zuK", n / 1024);
        } else {
            snprintf(buf, sizeof(buf), "%zu", n);
        }
        return buf;
    }

    void run(LuxBackend backend) {
        BenchResult result;
        result.operation = op_name;
        result.size = size_str();
        result.backend = backend;
        result.throughput_unit = "GB/s";

        LuxGPU* gpu = lux_gpu_create_with_backend(backend);
        if (!gpu) {
            result.success = false;
            result.error_msg = "Backend unavailable";
            results.push_back(result);
            return;
        }

        // 1D tensor
        const int64_t shape[] = {static_cast<int64_t>(n)};

        std::vector<float> h_a(n);
        std::vector<float> h_b(n);
        fill_random_float(h_a.data(), n, 0.1f, 2.0f);  // Avoid div by zero
        fill_random_float(h_b.data(), n, 0.1f, 2.0f);

        LuxTensor* a = lux_tensor_from_data(gpu, h_a.data(), shape, 1, LUX_FLOAT32);
        LuxTensor* b = lux_tensor_from_data(gpu, h_b.data(), shape, 1, LUX_FLOAT32);

        if (!a || !b) {
            result.success = false;
            result.error_msg = "Tensor allocation failed";
            lux_gpu_destroy(gpu);
            results.push_back(result);
            return;
        }

        LuxTensor* c = nullptr;

        auto kernel = [&]() {
            if (c) lux_tensor_destroy(c);
            if (op_name == "add") {
                c = lux_tensor_add(gpu, a, b);
            } else if (op_name == "mul") {
                c = lux_tensor_mul(gpu, a, b);
            } else if (op_name == "div") {
                c = lux_tensor_div(gpu, a, b);
            } else if (op_name == "sub") {
                c = lux_tensor_sub(gpu, a, b);
            }
        };

        auto sync = [&]() {
            lux_gpu_sync(gpu);
        };

        try {
            Stats stats = run_benchmark([](){}, kernel, [](){}, sync);
            result.stats = stats;
            result.success = true;

            // 2 reads + 1 write = 3 * n * sizeof(float) bytes
            size_t bytes = 3 * n * sizeof(float);
            result.throughput = compute_gbs(bytes, stats.median_ms);
        } catch (...) {
            result.success = false;
            result.error_msg = "Exception during benchmark";
        }

        if (c) lux_tensor_destroy(c);
        lux_tensor_destroy(a);
        lux_tensor_destroy(b);
        lux_gpu_destroy(gpu);

        results.push_back(result);
    }
};

// =============================================================================
// Reduction Benchmark
// =============================================================================

struct ReduceBench {
    size_t n;
    std::string op_name;
    std::vector<BenchResult> results;

    ReduceBench(size_t size, const char* op) : n(size), op_name(op) {}

    std::string size_str() const {
        char buf[32];
        if (n >= 1024 * 1024) {
            snprintf(buf, sizeof(buf), "%zuM", n / (1024 * 1024));
        } else if (n >= 1024) {
            snprintf(buf, sizeof(buf), "%zuK", n / 1024);
        } else {
            snprintf(buf, sizeof(buf), "%zu", n);
        }
        return buf;
    }

    void run(LuxBackend backend) {
        BenchResult result;
        result.operation = "reduce_" + op_name;
        result.size = size_str();
        result.backend = backend;
        result.throughput_unit = "GB/s";

        LuxGPU* gpu = lux_gpu_create_with_backend(backend);
        if (!gpu) {
            result.success = false;
            result.error_msg = "Backend unavailable";
            results.push_back(result);
            return;
        }

        const int64_t shape[] = {static_cast<int64_t>(n)};

        std::vector<float> h_data(n);
        fill_random_float(h_data.data(), n);

        LuxTensor* data = lux_tensor_from_data(gpu, h_data.data(), shape, 1, LUX_FLOAT32);

        if (!data) {
            result.success = false;
            result.error_msg = "Tensor allocation failed";
            lux_gpu_destroy(gpu);
            results.push_back(result);
            return;
        }

        LuxTensor* out = nullptr;
        int axes[] = {0};  // Reduce all

        auto kernel = [&]() {
            if (out) lux_tensor_destroy(out);
            if (op_name == "sum") {
                out = lux_tensor_sum(gpu, data, axes, 1);
            } else if (op_name == "max") {
                out = lux_tensor_max(gpu, data, axes, 1);
            } else if (op_name == "min") {
                out = lux_tensor_min(gpu, data, axes, 1);
            } else if (op_name == "mean") {
                out = lux_tensor_mean(gpu, data, axes, 1);
            }
        };

        auto sync = [&]() {
            lux_gpu_sync(gpu);
        };

        try {
            Stats stats = run_benchmark([](){}, kernel, [](){}, sync);
            result.stats = stats;
            result.success = true;

            // Read n elements
            size_t bytes = n * sizeof(float);
            result.throughput = compute_gbs(bytes, stats.median_ms);
        } catch (...) {
            result.success = false;
            result.error_msg = "Exception during benchmark";
        }

        if (out) lux_tensor_destroy(out);
        lux_tensor_destroy(data);
        lux_gpu_destroy(gpu);

        results.push_back(result);
    }
};

// =============================================================================
// Result Collection
// =============================================================================

struct TensorBenchResults {
    std::vector<BenchResult> all_results;

    void add(const std::vector<BenchResult>& results) {
        for (const auto& r : results) {
            all_results.push_back(r);
        }
    }

    void print_table() const {
        // Group by operation + size
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
    printf("=== Lux GPU Tensor Benchmarks ===\n\n");

    // Detect available backends
    auto backends = get_available_backends();
    printf("Available backends:\n");
    for (const auto& b : backends) {
        printf("  %s: %s\n", b.name, b.available ? "yes" : "no");
    }
    printf("\n");

    TensorBenchResults results;

    // Matrix sizes to benchmark
    std::vector<std::tuple<int, int, int>> matmul_sizes = {
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
    };

    // Elementwise sizes
    std::vector<size_t> elem_sizes = {
        1024,           // 1K
        1024 * 1024,    // 1M
        16 * 1024 * 1024,  // 16M
        64 * 1024 * 1024,  // 64M
    };

    // Reduction sizes
    std::vector<size_t> reduce_sizes = {
        1024 * 1024,       // 1M
        16 * 1024 * 1024,  // 16M
        64 * 1024 * 1024,  // 64M
        256 * 1024 * 1024, // 256M
    };

    printf("Running matmul benchmarks...\n");
    for (auto& [m, n, k] : matmul_sizes) {
        MatmulBench bench(m, n, k);
        printf("  %dx%dx%d: ", m, n, k);
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

    printf("\nRunning elementwise benchmarks...\n");
    const char* elem_ops[] = {"add", "mul", "div"};
    for (size_t sz : elem_sizes) {
        for (const char* op : elem_ops) {
            ElementwiseBench bench(sz, op);
            printf("  %s %s: ", op, bench.size_str().c_str());
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
    }

    printf("\nRunning reduction benchmarks...\n");
    const char* reduce_ops[] = {"sum", "max"};
    for (size_t sz : reduce_sizes) {
        for (const char* op : reduce_ops) {
            ReduceBench bench(sz, op);
            printf("  %s %s: ", op, bench.size_str().c_str());
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
    }

    printf("\n\n");
    printf("=== Results ===\n");
    results.print_table();

    return 0;
}
