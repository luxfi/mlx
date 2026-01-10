// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Test for lux-gpu plugin-based architecture

#include "lux/gpu.h"
#include <cstdio>
#include <cmath>
#include <cstring>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  %-40s", name)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL() do { printf("FAIL\n"); tests_failed++; } while(0)
#define CHECK(cond) do { if (cond) PASS(); else FAIL(); } while(0)

int main() {
    printf("=== Lux GPU Plugin Architecture Test ===\n\n");

    // Test 1: Create GPU context (auto-detect)
    printf("Backend Detection:\n");
    TEST("Create GPU context (auto)");
    LuxGPU* gpu = lux_gpu_create();
    CHECK(gpu != nullptr);

    TEST("Get backend name");
    const char* name = lux_gpu_backend_name(gpu);
    printf("(%s) ", name);
    CHECK(name != nullptr && strlen(name) > 0);

    // Test 2: Device info
    printf("\nDevice Info:\n");
    LuxDeviceInfo info;
    TEST("Get device info");
    LuxError err = lux_gpu_device_info(gpu, &info);
    CHECK(err == LUX_OK);

    if (err == LUX_OK) {
        printf("  Device: %s (%s)\n", info.name, info.vendor);
        printf("  Compute units: %d\n", info.compute_units);
        printf("  Unified memory: %s\n", info.is_unified_memory ? "yes" : "no");
    }

    // Test 3: Tensor creation
    printf("\nTensor Creation:\n");
    int64_t shape[] = {4};

    TEST("Create zeros tensor");
    LuxTensor* zeros = lux_tensor_zeros(gpu, shape, 1, LUX_FLOAT32);
    CHECK(zeros != nullptr);

    TEST("Create ones tensor");
    LuxTensor* ones = lux_tensor_ones(gpu, shape, 1, LUX_FLOAT32);
    CHECK(ones != nullptr);

    TEST("Create full tensor (2.0)");
    LuxTensor* twos = lux_tensor_full(gpu, shape, 1, LUX_FLOAT32, 2.0);
    CHECK(twos != nullptr);

    // Test 4: Tensor operations
    printf("\nTensor Operations:\n");

    TEST("Add (1 + 2 = 3)");
    LuxTensor* sum = lux_tensor_add(gpu, ones, twos);
    float result[4];
    if (sum) {
        lux_tensor_to_host(sum, result, sizeof(result));
        CHECK(result[0] == 3.0f && result[1] == 3.0f && result[2] == 3.0f && result[3] == 3.0f);
    } else {
        FAIL();
    }

    TEST("Mul (1 * 2 = 2)");
    LuxTensor* prod = lux_tensor_mul(gpu, ones, twos);
    if (prod) {
        lux_tensor_to_host(prod, result, sizeof(result));
        CHECK(result[0] == 2.0f && result[1] == 2.0f && result[2] == 2.0f && result[3] == 2.0f);
    } else {
        FAIL();
    }

    TEST("Sub (2 - 1 = 1)");
    LuxTensor* diff = lux_tensor_sub(gpu, twos, ones);
    if (diff) {
        lux_tensor_to_host(diff, result, sizeof(result));
        CHECK(result[0] == 1.0f && result[1] == 1.0f && result[2] == 1.0f && result[3] == 1.0f);
    } else {
        FAIL();
    }

    // Test 5: Matrix multiplication
    printf("\nMatrix Multiplication:\n");
    int64_t mat_shape[] = {2, 2};
    float mat_a[] = {1, 2, 3, 4};
    float mat_b[] = {5, 6, 7, 8};

    LuxTensor* A = lux_tensor_from_data(gpu, mat_a, mat_shape, 2, LUX_FLOAT32);
    LuxTensor* B = lux_tensor_from_data(gpu, mat_b, mat_shape, 2, LUX_FLOAT32);

    TEST("Matmul [[1,2],[3,4]] * [[5,6],[7,8]]");
    LuxTensor* C = lux_tensor_matmul(gpu, A, B);
    float mat_result[4];
    if (C) {
        lux_tensor_to_host(C, mat_result, sizeof(mat_result));
        // Expected: [[19, 22], [43, 50]]
        bool pass = (mat_result[0] == 19.0f && mat_result[1] == 22.0f &&
                     mat_result[2] == 43.0f && mat_result[3] == 50.0f);
        if (!pass) {
            printf("\n    Got: [[%.0f, %.0f], [%.0f, %.0f]] ",
                   mat_result[0], mat_result[1], mat_result[2], mat_result[3]);
        }
        CHECK(pass);
    } else {
        FAIL();
    }

    // Test 6: NTT operations
    printf("\nNTT Operations:\n");
    uint64_t modulus = 0xFFFFFFFF00000001ULL;  // Goldilocks prime
    uint64_t ntt_data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint64_t original[8];
    memcpy(original, ntt_data, sizeof(original));

    TEST("NTT forward");
    err = lux_ntt_forward(gpu, ntt_data, 8, modulus);
    CHECK(err == LUX_OK);

    TEST("NTT inverse");
    err = lux_ntt_inverse(gpu, ntt_data, 8, modulus);
    CHECK(err == LUX_OK);

    TEST("NTT roundtrip preserves data");
    bool ntt_pass = true;
    for (int i = 0; i < 8; i++) {
        if (ntt_data[i] != original[i]) {
            ntt_pass = false;
            printf("\n    Mismatch at [%d]: got %llu, expected %llu ",
                   i, (unsigned long long)ntt_data[i], (unsigned long long)original[i]);
            break;
        }
    }
    CHECK(ntt_pass);

    // Test 7: Backend switching (if multiple available)
    printf("\nBackend Switching:\n");
    TEST("CPU backend available");
    CHECK(lux_backend_available(LUX_BACKEND_CPU));

    TEST("Switch to CPU");
    err = lux_gpu_set_backend(gpu, LUX_BACKEND_CPU);
    CHECK(err == LUX_OK && strcmp(lux_gpu_backend_name(gpu), "cpu") == 0);

    // Cleanup
    printf("\nCleanup:\n");
    TEST("Destroy tensors");
    lux_tensor_destroy(zeros);
    lux_tensor_destroy(ones);
    lux_tensor_destroy(twos);
    lux_tensor_destroy(sum);
    lux_tensor_destroy(prod);
    lux_tensor_destroy(diff);
    lux_tensor_destroy(A);
    lux_tensor_destroy(B);
    lux_tensor_destroy(C);
    PASS();

    TEST("Destroy GPU context");
    lux_gpu_destroy(gpu);
    PASS();

    // Summary
    printf("\n=== Test Summary ===\n");
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);
    printf("Total:  %d\n", tests_passed + tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
