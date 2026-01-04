// Copyright (c) 2024-2025 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Ecosystem
//
// Basic test for unified GPU API
//
// Build: c++ -std=c++20 -I../include test_gpu.cpp ../src/gpu.cpp -o test_gpu
// Run:   ./test_gpu

#include "lux/gpu/gpu.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <cstdlib>

// =============================================================================
// Test Utilities
// =============================================================================

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    do { \
        tests_run++; \
        printf("  [%s] ", #name); \
        fflush(stdout); \
    } while(0)

#define PASS() \
    do { \
        tests_passed++; \
        printf("PASS\n"); \
    } while(0)

#define FAIL(msg) \
    do { \
        printf("FAIL: %s\n", msg); \
    } while(0)

#define ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            FAIL(msg); \
            return; \
        } \
    } while(0)

// =============================================================================
// Tests
// =============================================================================

void test_context_create() {
    TEST(context_create);

    LuxGPU* gpu = lux_gpu_create();
    ASSERT(gpu != nullptr, "lux_gpu_create returned nullptr");

    LuxGPUBackend backend = lux_gpu_backend(gpu);
    ASSERT(backend != LUX_GPU_BACKEND_AUTO, "backend should not be AUTO after create");

    const char* name = lux_gpu_backend_name(gpu);
    ASSERT(name != nullptr, "backend name should not be null");
    printf("(%s) ", name);

    lux_gpu_destroy(gpu);
    PASS();
}

void test_context_specific_backend() {
    TEST(context_specific_backend);

    // Test CPU backend (always available)
    LuxGPU* gpu = lux_gpu_create_backend(LUX_GPU_BACKEND_CPU);
    ASSERT(gpu != nullptr, "CPU backend creation failed");
    ASSERT(lux_gpu_backend(gpu) == LUX_GPU_BACKEND_CPU, "wrong backend");

    const char* name = lux_gpu_device_name(gpu);
    ASSERT(name != nullptr && strlen(name) > 0, "device name should not be empty");

    lux_gpu_destroy(gpu);
    PASS();
}

void test_buffer_create() {
    TEST(buffer_create);

    LuxGPU* gpu = lux_gpu_create();
    ASSERT(gpu != nullptr, "context creation failed");

    const size_t size = 1024;
    LuxBuffer* buf = lux_gpu_buffer_create(gpu, size, LUX_MEM_DEVICE);
    ASSERT(buf != nullptr, "buffer creation failed");
    ASSERT(lux_gpu_buffer_size(buf) == size, "buffer size mismatch");

    lux_gpu_buffer_destroy(buf);
    lux_gpu_destroy(gpu);
    PASS();
}

void test_buffer_read_write() {
    TEST(buffer_read_write);

    LuxGPU* gpu = lux_gpu_create();
    ASSERT(gpu != nullptr, "context creation failed");

    const size_t size = 256;
    LuxBuffer* buf = lux_gpu_buffer_create(gpu, size, LUX_MEM_DEVICE);
    ASSERT(buf != nullptr, "buffer creation failed");

    // Write data
    uint8_t write_data[256];
    for (int i = 0; i < 256; i++) write_data[i] = (uint8_t)i;

    int rc = lux_gpu_buffer_write(buf, write_data, size, 0);
    ASSERT(rc == LUX_GPU_OK, "buffer write failed");

    // Read data back
    uint8_t read_data[256] = {0};
    rc = lux_gpu_buffer_read(buf, read_data, size, 0);
    ASSERT(rc == LUX_GPU_OK, "buffer read failed");

    // Verify
    ASSERT(memcmp(write_data, read_data, size) == 0, "data mismatch");

    lux_gpu_buffer_destroy(buf);
    lux_gpu_destroy(gpu);
    PASS();
}

void test_buffer_partial_write() {
    TEST(buffer_partial_write);

    LuxGPU* gpu = lux_gpu_create();
    ASSERT(gpu != nullptr, "context creation failed");

    const size_t size = 256;
    LuxBuffer* buf = lux_gpu_buffer_create(gpu, size, LUX_MEM_DEVICE);
    ASSERT(buf != nullptr, "buffer creation failed");

    // Write partial data at offset
    uint8_t pattern[32];
    for (int i = 0; i < 32; i++) pattern[i] = 0xAB;

    int rc = lux_gpu_buffer_write(buf, pattern, 32, 64);
    ASSERT(rc == LUX_GPU_OK, "partial write failed");

    // Read back
    uint8_t result[32] = {0};
    rc = lux_gpu_buffer_read(buf, result, 32, 64);
    ASSERT(rc == LUX_GPU_OK, "partial read failed");
    ASSERT(memcmp(pattern, result, 32) == 0, "partial data mismatch");

    lux_gpu_buffer_destroy(buf);
    lux_gpu_destroy(gpu);
    PASS();
}

void test_buffer_create_with_data() {
    TEST(buffer_create_with_data);

    LuxGPU* gpu = lux_gpu_create();
    ASSERT(gpu != nullptr, "context creation failed");

    float data[16] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                      9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};

    LuxBuffer* buf = lux_gpu_buffer_create_data(gpu, data, sizeof(data), LUX_MEM_DEVICE);
    ASSERT(buf != nullptr, "buffer creation failed");
    ASSERT(lux_gpu_buffer_size(buf) == sizeof(data), "size mismatch");

    float result[16] = {0};
    int rc = lux_gpu_buffer_read(buf, result, sizeof(result), 0);
    ASSERT(rc == LUX_GPU_OK, "read failed");
    ASSERT(memcmp(data, result, sizeof(data)) == 0, "data mismatch");

    lux_gpu_buffer_destroy(buf);
    lux_gpu_destroy(gpu);
    PASS();
}

void test_poseidon2_single() {
    TEST(poseidon2_single);

    LuxGPU* gpu = lux_gpu_create();
    ASSERT(gpu != nullptr, "context creation failed");

    LuxFr256 left = {{1, 0, 0, 0}};
    LuxFr256 right = {{2, 0, 0, 0}};
    LuxFr256 out = {{0, 0, 0, 0}};

    int rc = lux_gpu_poseidon2(gpu, &out, &left, &right, 1);
    ASSERT(rc == LUX_GPU_OK, "poseidon2 failed");

    // Hash should be non-zero and different from inputs
    bool non_zero = (out.limbs[0] != 0 || out.limbs[1] != 0 ||
                     out.limbs[2] != 0 || out.limbs[3] != 0);
    ASSERT(non_zero, "hash output is all zeros");

    bool different_from_left = (out.limbs[0] != left.limbs[0] ||
                                out.limbs[1] != left.limbs[1] ||
                                out.limbs[2] != left.limbs[2] ||
                                out.limbs[3] != left.limbs[3]);
    ASSERT(different_from_left, "hash equals left input");

    lux_gpu_destroy(gpu);
    PASS();
}

void test_poseidon2_batch() {
    TEST(poseidon2_batch);

    LuxGPU* gpu = lux_gpu_create();
    ASSERT(gpu != nullptr, "context creation failed");

    const size_t count = 16;
    LuxFr256 left[count], right[count], out[count];

    for (size_t i = 0; i < count; i++) {
        left[i] = {{i + 1, 0, 0, 0}};
        right[i] = {{i + 100, 0, 0, 0}};
        out[i] = {{0, 0, 0, 0}};
    }

    int rc = lux_gpu_poseidon2(gpu, out, left, right, count);
    ASSERT(rc == LUX_GPU_OK, "batch poseidon2 failed");

    // All outputs should be non-zero
    for (size_t i = 0; i < count; i++) {
        bool non_zero = (out[i].limbs[0] != 0 || out[i].limbs[1] != 0 ||
                         out[i].limbs[2] != 0 || out[i].limbs[3] != 0);
        ASSERT(non_zero, "batch hash output is all zeros");
    }

    // All outputs should be different (collision resistance)
    for (size_t i = 0; i < count; i++) {
        for (size_t j = i + 1; j < count; j++) {
            bool different = (out[i].limbs[0] != out[j].limbs[0] ||
                              out[i].limbs[1] != out[j].limbs[1] ||
                              out[i].limbs[2] != out[j].limbs[2] ||
                              out[i].limbs[3] != out[j].limbs[3]);
            ASSERT(different, "hash collision detected");
        }
    }

    lux_gpu_destroy(gpu);
    PASS();
}

void test_poseidon2_determinism() {
    TEST(poseidon2_determinism);

    LuxGPU* gpu = lux_gpu_create();
    ASSERT(gpu != nullptr, "context creation failed");

    LuxFr256 left = {{0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0x9ABCDEF0}};
    LuxFr256 right = {{0x11111111, 0x22222222, 0x33333333, 0x44444444}};
    LuxFr256 out1 = {{0}}, out2 = {{0}};

    int rc1 = lux_gpu_poseidon2(gpu, &out1, &left, &right, 1);
    int rc2 = lux_gpu_poseidon2(gpu, &out2, &left, &right, 1);

    ASSERT(rc1 == LUX_GPU_OK && rc2 == LUX_GPU_OK, "poseidon2 failed");
    ASSERT(memcmp(&out1, &out2, sizeof(LuxFr256)) == 0, "hash not deterministic");

    lux_gpu_destroy(gpu);
    PASS();
}

void test_merkle_root() {
    TEST(merkle_root);

    LuxGPU* gpu = lux_gpu_create();
    ASSERT(gpu != nullptr, "context creation failed");

    // 4 leaves
    LuxFr256 leaves[4] = {
        {{1, 0, 0, 0}},
        {{2, 0, 0, 0}},
        {{3, 0, 0, 0}},
        {{4, 0, 0, 0}}
    };
    LuxFr256 root = {{0}};

    int rc = lux_gpu_merkle_root(gpu, &root, leaves, 4);
    ASSERT(rc == LUX_GPU_OK, "merkle_root failed");

    bool non_zero = (root.limbs[0] != 0 || root.limbs[1] != 0 ||
                     root.limbs[2] != 0 || root.limbs[3] != 0);
    ASSERT(non_zero, "merkle root is all zeros");

    // Same leaves should produce same root
    LuxFr256 root2 = {{0}};
    rc = lux_gpu_merkle_root(gpu, &root2, leaves, 4);
    ASSERT(rc == LUX_GPU_OK, "second merkle_root failed");
    ASSERT(memcmp(&root, &root2, sizeof(LuxFr256)) == 0, "merkle root not deterministic");

    // Different leaves should produce different root
    leaves[0].limbs[0] = 999;
    LuxFr256 root3 = {{0}};
    rc = lux_gpu_merkle_root(gpu, &root3, leaves, 4);
    ASSERT(rc == LUX_GPU_OK, "third merkle_root failed");
    ASSERT(memcmp(&root, &root3, sizeof(LuxFr256)) != 0, "merkle root unchanged with different input");

    lux_gpu_destroy(gpu);
    PASS();
}

void test_commitment() {
    TEST(commitment);

    LuxGPU* gpu = lux_gpu_create();
    ASSERT(gpu != nullptr, "context creation failed");

    const size_t count = 4;
    LuxFr256 values[count], blindings[count], salts[count], out[count];

    for (size_t i = 0; i < count; i++) {
        values[i] = {{i + 1, 0, 0, 0}};
        blindings[i] = {{i + 100, 0, 0, 0}};
        salts[i] = {{i + 200, 0, 0, 0}};
        out[i] = {{0}};
    }

    int rc = lux_gpu_commitment(gpu, out, values, blindings, salts, count);
    ASSERT(rc == LUX_GPU_OK, "commitment failed");

    // All commitments should be non-zero
    for (size_t i = 0; i < count; i++) {
        bool non_zero = (out[i].limbs[0] != 0 || out[i].limbs[1] != 0 ||
                         out[i].limbs[2] != 0 || out[i].limbs[3] != 0);
        ASSERT(non_zero, "commitment is all zeros");
    }

    // All commitments should be different
    for (size_t i = 0; i < count; i++) {
        for (size_t j = i + 1; j < count; j++) {
            bool different = (out[i].limbs[0] != out[j].limbs[0] ||
                              out[i].limbs[1] != out[j].limbs[1] ||
                              out[i].limbs[2] != out[j].limbs[2] ||
                              out[i].limbs[3] != out[j].limbs[3]);
            ASSERT(different, "commitment collision");
        }
    }

    lux_gpu_destroy(gpu);
    PASS();
}

void test_sync() {
    TEST(sync);

    LuxGPU* gpu = lux_gpu_create();
    ASSERT(gpu != nullptr, "context creation failed");

    int rc = lux_gpu_sync(gpu);
    ASSERT(rc == LUX_GPU_OK, "sync failed");

    lux_gpu_destroy(gpu);
    PASS();
}

void test_stream() {
    TEST(stream);

    LuxGPU* gpu = lux_gpu_create();
    ASSERT(gpu != nullptr, "context creation failed");

    LuxStream* stream = lux_gpu_stream_create(gpu);
    ASSERT(stream != nullptr, "stream creation failed");

    int rc = lux_gpu_stream_sync(stream);
    ASSERT(rc == LUX_GPU_OK, "stream sync failed");

    lux_gpu_stream_destroy(stream);
    lux_gpu_destroy(gpu);
    PASS();
}

void test_global_instance() {
    TEST(global_instance);

    LuxGPU* gpu1 = lux_gpu_global();
    ASSERT(gpu1 != nullptr, "global instance is null");

    LuxGPU* gpu2 = lux_gpu_global();
    ASSERT(gpu2 == gpu1, "global instance not singleton");

    // Backend should be valid
    LuxGPUBackend backend = lux_gpu_backend(gpu1);
    ASSERT(backend != LUX_GPU_BACKEND_AUTO, "global backend is AUTO");

    PASS();
}

void test_error_handling() {
    TEST(error_handling);

    // Null pointer handling
    LuxGPU* gpu = lux_gpu_create();

    int rc = lux_gpu_poseidon2(nullptr, nullptr, nullptr, nullptr, 0);
    ASSERT(rc == LUX_GPU_ERROR_INVALID_ARGS, "null gpu should return error");

    rc = lux_gpu_buffer_write(nullptr, nullptr, 0, 0);
    ASSERT(rc == LUX_GPU_ERROR_INVALID_ARGS, "null buffer should return error");

    LuxBuffer* buf = lux_gpu_buffer_create(gpu, 64, LUX_MEM_DEVICE);
    rc = lux_gpu_buffer_write(buf, nullptr, 64, 0);
    ASSERT(rc == LUX_GPU_ERROR_INVALID_ARGS, "null data should return error");

    // Out of bounds
    uint8_t data[128];
    rc = lux_gpu_buffer_write(buf, data, 128, 0);  // 128 > 64
    ASSERT(rc == LUX_GPU_ERROR_INVALID_ARGS, "out of bounds should return error");

    lux_gpu_buffer_destroy(buf);
    lux_gpu_destroy(gpu);
    PASS();
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    printf("=== Lux GPU API Test Suite ===\n\n");

    printf("Context Tests:\n");
    test_context_create();
    test_context_specific_backend();

    printf("\nBuffer Tests:\n");
    test_buffer_create();
    test_buffer_read_write();
    test_buffer_partial_write();
    test_buffer_create_with_data();

    printf("\nZK Crypto Tests:\n");
    test_poseidon2_single();
    test_poseidon2_batch();
    test_poseidon2_determinism();
    test_merkle_root();
    test_commitment();

    printf("\nSync Tests:\n");
    test_sync();
    test_stream();
    test_global_instance();

    printf("\nError Handling Tests:\n");
    test_error_handling();

    printf("\n=== Results: %d/%d tests passed ===\n", tests_passed, tests_run);

    return (tests_passed == tests_run) ? 0 : 1;
}
