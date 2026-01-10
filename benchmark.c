// Benchmark comparing Metal, WebGPU, and CPU backends
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <lux/gpu.h>

#define WARMUP_ITERS 2
#define BENCH_ITERS 10
#define MATRIX_SIZE 1024

double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

void benchmark_backend(LuxBackend backend, const char* name) {
    printf("\n=== %s Backend ===\n", name);

    if (!lux_backend_available(backend)) {
        printf("  Not available\n");
        return;
    }

    LuxGPU* gpu = lux_gpu_create_with_backend(backend);
    if (!gpu) {
        printf("  Failed to create context\n");
        return;
    }

    LuxBackend actual = lux_gpu_backend(gpu);
    printf("  Active: %s\n", lux_gpu_backend_name(gpu));

    if (actual != backend && backend != LUX_BACKEND_AUTO) {
        printf("  (Fell back to different backend)\n");
    }

    LuxDeviceInfo info;
    if (lux_gpu_device_info(gpu, &info) == LUX_OK) {
        printf("  Device: %s\n", info.name);
    }

    // Test tensor creation
    int64_t shape[] = {MATRIX_SIZE, MATRIX_SIZE};

    printf("\n  Tensor Creation (%dx%d float32):\n", MATRIX_SIZE, MATRIX_SIZE);

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        LuxTensor* t = lux_tensor_zeros(gpu, shape, 2, LUX_FLOAT32);
        if (t) lux_tensor_destroy(t);
    }

    double start = get_time_ms();
    for (int i = 0; i < BENCH_ITERS; i++) {
        LuxTensor* t = lux_tensor_zeros(gpu, shape, 2, LUX_FLOAT32);
        if (t) lux_tensor_destroy(t);
    }
    double elapsed = get_time_ms() - start;
    printf("    zeros():  %.2f ms/op\n", elapsed / BENCH_ITERS);

    // ones
    start = get_time_ms();
    for (int i = 0; i < BENCH_ITERS; i++) {
        LuxTensor* t = lux_tensor_ones(gpu, shape, 2, LUX_FLOAT32);
        if (t) lux_tensor_destroy(t);
    }
    elapsed = get_time_ms() - start;
    printf("    ones():   %.2f ms/op\n", elapsed / BENCH_ITERS);

    // Test tensor operations
    printf("\n  Tensor Operations:\n");

    LuxTensor* a = lux_tensor_ones(gpu, shape, 2, LUX_FLOAT32);
    LuxTensor* b = lux_tensor_full(gpu, shape, 2, LUX_FLOAT32, 2.0f);

    if (a && b) {
        // Add
        LuxTensor* c = lux_tensor_add(gpu, a, b);
        if (c) {
            // Warmup
            for (int i = 0; i < WARMUP_ITERS; i++) {
                LuxTensor* tmp = lux_tensor_add(gpu, a, b);
                if (tmp) lux_tensor_destroy(tmp);
            }

            start = get_time_ms();
            for (int i = 0; i < BENCH_ITERS; i++) {
                LuxTensor* tmp = lux_tensor_add(gpu, a, b);
                if (tmp) lux_tensor_destroy(tmp);
            }
            elapsed = get_time_ms() - start;
            printf("    add():    %.2f ms/op (%.2f GFLOPS)\n",
                   elapsed / BENCH_ITERS,
                   (double)(MATRIX_SIZE * MATRIX_SIZE) / (elapsed / BENCH_ITERS) / 1e6);
            lux_tensor_destroy(c);
        } else {
            printf("    add():    NOT SUPPORTED\n");
        }

        // Matmul
        c = lux_tensor_matmul(gpu, a, b);
        if (c) {
            // Warmup
            for (int i = 0; i < WARMUP_ITERS; i++) {
                LuxTensor* tmp = lux_tensor_matmul(gpu, a, b);
                if (tmp) lux_tensor_destroy(tmp);
            }
            lux_gpu_sync(gpu);

            start = get_time_ms();
            for (int i = 0; i < BENCH_ITERS; i++) {
                LuxTensor* tmp = lux_tensor_matmul(gpu, a, b);
                if (tmp) lux_tensor_destroy(tmp);
            }
            lux_gpu_sync(gpu);
            elapsed = get_time_ms() - start;

            // 2*N^3 FLOPs for NxN matmul
            double flops = 2.0 * MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE;
            printf("    matmul(): %.2f ms/op (%.2f GFLOPS)\n",
                   elapsed / BENCH_ITERS,
                   flops / (elapsed / BENCH_ITERS) / 1e6);
            lux_tensor_destroy(c);
        } else {
            printf("    matmul(): NOT SUPPORTED\n");
        }

        lux_tensor_destroy(a);
        lux_tensor_destroy(b);
    }

    // Sync benchmark
    printf("\n  Sync latency:\n");
    start = get_time_ms();
    for (int i = 0; i < BENCH_ITERS; i++) {
        lux_gpu_sync(gpu);
    }
    elapsed = get_time_ms() - start;
    printf("    sync():   %.2f ms/op\n", elapsed / BENCH_ITERS);

    lux_gpu_destroy(gpu);
}

int main() {
    printf("=== Lux GPU Backend Benchmark ===\n");
    printf("Matrix size: %dx%d\n", MATRIX_SIZE, MATRIX_SIZE);
    printf("Iterations: %d (warmup: %d)\n", BENCH_ITERS, WARMUP_ITERS);

    // List available backends
    printf("\nAvailable backends:\n");
    printf("  CPU:    %s\n", lux_backend_available(LUX_BACKEND_CPU) ? "YES" : "NO");
    printf("  Metal:  %s\n", lux_backend_available(LUX_BACKEND_METAL) ? "YES" : "NO");
    printf("  CUDA:   %s\n", lux_backend_available(LUX_BACKEND_CUDA) ? "YES" : "NO");
    printf("  WebGPU: %s\n", lux_backend_available(LUX_BACKEND_DAWN) ? "YES" : "NO");

    // Benchmark each backend
    benchmark_backend(LUX_BACKEND_CPU, "CPU");
    benchmark_backend(LUX_BACKEND_METAL, "Metal");
    benchmark_backend(LUX_BACKEND_DAWN, "WebGPU");

    printf("\n=== Benchmark Complete ===\n");
    return 0;
}
