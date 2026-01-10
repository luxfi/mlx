// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Benchmark Common Utilities
// Provides timing, statistics, and table formatting for cross-backend benchmarks.

#ifndef LUX_BENCH_COMMON_HPP
#define LUX_BENCH_COMMON_HPP

#include "lux/gpu.h"
#include "lux/gpu/backend_plugin.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <numeric>

namespace bench {

// =============================================================================
// Configuration
// =============================================================================

constexpr int DEFAULT_WARMUP_ITERS = 3;
constexpr int DEFAULT_BENCH_ITERS = 10;
constexpr double GFLOPS_FACTOR = 1e-9;
constexpr double GB_FACTOR = 1e-9;

// =============================================================================
// Timing Utilities
// =============================================================================

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

struct Timer {
    Clock::time_point start_;

    void start() { start_ = Clock::now(); }

    double elapsed_ms() const {
        auto end = Clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count() / 1000.0;
    }

    double elapsed_s() const { return elapsed_ms() / 1000.0; }
};

// =============================================================================
// Statistics
// =============================================================================

struct Stats {
    double min_ms = 0;
    double max_ms = 0;
    double mean_ms = 0;
    double median_ms = 0;
    double stddev_ms = 0;
    int iterations = 0;

    void compute(std::vector<double>& times) {
        if (times.empty()) return;

        iterations = static_cast<int>(times.size());
        std::sort(times.begin(), times.end());

        min_ms = times.front();
        max_ms = times.back();
        median_ms = times[times.size() / 2];

        double sum = std::accumulate(times.begin(), times.end(), 0.0);
        mean_ms = sum / times.size();

        double sq_sum = 0;
        for (double t : times) {
            sq_sum += (t - mean_ms) * (t - mean_ms);
        }
        stddev_ms = std::sqrt(sq_sum / times.size());
    }
};

// =============================================================================
// Backend Management
// =============================================================================

struct BackendInfo {
    LuxBackend type;
    const char* name;
    bool available;
};

inline std::vector<BackendInfo> get_available_backends() {
    std::vector<BackendInfo> backends;

    BackendInfo cpu = {LUX_BACKEND_CPU, "CPU", true};
    backends.push_back(cpu);

    BackendInfo metal = {LUX_BACKEND_METAL, "Metal", lux_backend_available(LUX_BACKEND_METAL)};
    backends.push_back(metal);

    BackendInfo cuda = {LUX_BACKEND_CUDA, "CUDA", lux_backend_available(LUX_BACKEND_CUDA)};
    backends.push_back(cuda);

    BackendInfo webgpu = {LUX_BACKEND_DAWN, "WebGPU", lux_backend_available(LUX_BACKEND_DAWN)};
    backends.push_back(webgpu);

    return backends;
}

inline const char* backend_name(LuxBackend backend) {
    switch (backend) {
        case LUX_BACKEND_CPU: return "CPU";
        case LUX_BACKEND_METAL: return "Metal";
        case LUX_BACKEND_CUDA: return "CUDA";
        case LUX_BACKEND_DAWN: return "WebGPU";
        default: return "Unknown";
    }
}

// =============================================================================
// Result Tracking
// =============================================================================

struct BenchResult {
    std::string operation;
    std::string size;
    LuxBackend backend;
    Stats stats;
    double throughput;        // Operation-specific (GFLOPS, GB/s, ops/sec)
    std::string throughput_unit;
    bool success;
    std::string error_msg;

    BenchResult() : backend(LUX_BACKEND_CPU), throughput(0), success(false) {}
};

// =============================================================================
// Benchmark Runner
// =============================================================================

using BenchFunc = std::function<void()>;

inline Stats run_benchmark(
    BenchFunc setup,
    BenchFunc kernel,
    BenchFunc teardown,
    BenchFunc sync,
    int warmup_iters = DEFAULT_WARMUP_ITERS,
    int bench_iters = DEFAULT_BENCH_ITERS
) {
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        setup();
        kernel();
        sync();
        teardown();
    }

    // Benchmark
    std::vector<double> times;
    times.reserve(bench_iters);

    for (int i = 0; i < bench_iters; i++) {
        setup();

        Timer timer;
        timer.start();
        kernel();
        sync();
        double elapsed = timer.elapsed_ms();

        teardown();
        times.push_back(elapsed);
    }

    Stats stats;
    stats.compute(times);
    return stats;
}

// Simplified version for in-place operations
inline Stats run_benchmark_simple(
    BenchFunc kernel,
    BenchFunc sync,
    int warmup_iters = DEFAULT_WARMUP_ITERS,
    int bench_iters = DEFAULT_BENCH_ITERS
) {
    return run_benchmark(
        [](){}, kernel, [](){}, sync,
        warmup_iters, bench_iters
    );
}

// =============================================================================
// Table Formatting
// =============================================================================

struct TableRow {
    std::string operation;
    std::string size;
    std::string cpu;
    std::string metal;
    std::string cuda;
    std::string webgpu;
};

inline void print_table_header() {
    printf("\n");
    printf("%-20s | %-10s | %-12s | %-12s | %-12s | %-12s\n",
           "Operation", "Size", "CPU", "Metal", "CUDA", "WebGPU");
    printf("%-20s-+-%-10s-+-%-12s-+-%-12s-+-%-12s-+-%-12s\n",
           "--------------------", "----------", "------------",
           "------------", "------------", "------------");
}

inline void print_table_row(const TableRow& row) {
    printf("%-20s | %-10s | %-12s | %-12s | %-12s | %-12s\n",
           row.operation.c_str(),
           row.size.c_str(),
           row.cpu.c_str(),
           row.metal.c_str(),
           row.cuda.c_str(),
           row.webgpu.c_str());
}

inline void print_table_separator() {
    printf("%-20s-+-%-10s-+-%-12s-+-%-12s-+-%-12s-+-%-12s\n",
           "--------------------", "----------", "------------",
           "------------", "------------", "------------");
}

inline std::string format_throughput(double value, const char* unit) {
    if (value <= 0) return "N/A";
    char buf[32];
    if (value >= 1000) {
        snprintf(buf, sizeof(buf), "%.1f %s", value, unit);
    } else if (value >= 1) {
        snprintf(buf, sizeof(buf), "%.2f %s", value, unit);
    } else {
        snprintf(buf, sizeof(buf), "%.3f %s", value, unit);
    }
    return buf;
}

inline std::string format_time(double ms) {
    if (ms <= 0) return "N/A";
    char buf[32];
    if (ms >= 1000) {
        snprintf(buf, sizeof(buf), "%.2f s", ms / 1000);
    } else if (ms >= 1) {
        snprintf(buf, sizeof(buf), "%.2f ms", ms);
    } else {
        snprintf(buf, sizeof(buf), "%.0f us", ms * 1000);
    }
    return buf;
}

// =============================================================================
// Random Data Generation
// =============================================================================

inline void fill_random_float(float* data, size_t n, float min_val = 0.0f, float max_val = 1.0f) {
    for (size_t i = 0; i < n; i++) {
        data[i] = min_val + (max_val - min_val) * (static_cast<float>(rand()) / RAND_MAX);
    }
}

inline void fill_random_uint64(uint64_t* data, size_t n, uint64_t modulus) {
    for (size_t i = 0; i < n; i++) {
        data[i] = (static_cast<uint64_t>(rand()) << 32 | rand()) % modulus;
    }
}

inline void fill_random_bytes(uint8_t* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = static_cast<uint8_t>(rand() & 0xFF);
    }
}

// =============================================================================
// FLOPS Calculation
// =============================================================================

// GEMM: 2*M*N*K FLOPs (multiply-add)
inline double gemm_flops(int M, int N, int K) {
    return 2.0 * M * N * K;
}

// Elementwise: N FLOPs
inline double elementwise_flops(size_t n) {
    return static_cast<double>(n);
}

// Reduction: N FLOPs
inline double reduction_flops(size_t n) {
    return static_cast<double>(n);
}

// NTT: N * log2(N) FLOPs (butterfly operations)
inline double ntt_flops(size_t n) {
    return static_cast<double>(n) * std::log2(static_cast<double>(n));
}

inline double compute_gflops(double flops, double time_ms) {
    return (flops / (time_ms * 1e-3)) * GFLOPS_FACTOR;
}

inline double compute_gbs(size_t bytes, double time_ms) {
    return (static_cast<double>(bytes) / (time_ms * 1e-3)) * GB_FACTOR;
}

// =============================================================================
// Test Utilities
// =============================================================================

inline bool is_power_of_two(size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

inline size_t next_power_of_two(size_t n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

} // namespace bench

#endif // LUX_BENCH_COMMON_HPP
