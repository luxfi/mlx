# Copyright (c) 2024-2025 Lux Industries Inc.
# SPDX-License-Identifier: BSD-3-Clause-Ecosystem

FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/onqtam/doctest.git
  GIT_TAG v2.4.12)
FetchContent_MakeAvailable(doctest)

add_executable(tests ${PROJECT_SOURCE_DIR}/tests/tests.cpp)

if(MLX_BUILD_METAL OR MLX_BUILD_CUDA)
  set(METAL_TEST_SOURCES gpu_tests.cpp)
endif()

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)

target_sources(
  tests
  PRIVATE allocator_tests.cpp
          array_tests.cpp
          arg_reduce_tests.cpp
          autograd_tests.cpp
          blas_tests.cpp
          compile_tests.cpp
          custom_vjp_tests.cpp
          creations_tests.cpp
          device_tests.cpp
          einsum_tests.cpp
          export_import_tests.cpp
          eval_tests.cpp
          fft_tests.cpp
          load_tests.cpp
          ops_tests.cpp
          random_tests.cpp
          scheduler_tests.cpp
          utils_tests.cpp
          vmap_tests.cpp
          linalg_tests.cpp
          ${METAL_TEST_SOURCES})

target_link_libraries(tests PRIVATE mlx doctest)
target_compile_options(tests PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(tests PRIVATE ${SANITIZER_LINK_FLAGS})
doctest_discover_tests(tests)
add_test(NAME tests COMMAND tests)

# =============================================================================
# ZK Kernel Tests: Validate WGSL kernels match CPU reference
# =============================================================================

add_executable(zk_tests
  ${PROJECT_SOURCE_DIR}/tests/zk_test.cpp
)

target_link_libraries(zk_tests PRIVATE mlx doctest)
target_compile_options(zk_tests PRIVATE ${SANITIZER_COMPILE_FLAGS})
target_link_options(zk_tests PRIVATE ${SANITIZER_LINK_FLAGS})

# Enable ZK module compilation flag if zk sources are available
if(EXISTS "${PROJECT_SOURCE_DIR}/mlx/zk/zk.h")
  target_compile_definitions(zk_tests PRIVATE MLX_BUILD_ZK)
endif()

# Enable WebGPU tests if WebGPU backend is built
if(MLX_BUILD_WEBGPU)
  target_compile_definitions(zk_tests PRIVATE LUX_HAVE_WEBGPU)
  target_include_directories(zk_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/mlx/backend/webgpu
  )
endif()

doctest_discover_tests(zk_tests)
add_test(NAME zk_tests COMMAND zk_tests)
