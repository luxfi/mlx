# Copyright (c) 2024-2026 Lux Industries Inc.
# SPDX-License-Identifier: BSD-3-Clause-Eco
#
# Lux GPU - Plugin-based GPU acceleration library
#
# Artifacts:
#   libluxgpu_core     - Core runtime + CPU backend + plugin loader
#   libluxgpu_backend_metal  - Metal plugin (macOS only)
#   libluxgpu_backend_cuda   - CUDA plugin (Linux/Windows)
#   libluxgpu_backend_webgpu - Dawn/wgpu plugin (optional)
#
# Build options:
#   cmake -B build                                    # Core only (CPU)
#   cmake -B build -DLUX_GPU_BUILD_METAL=ON          # + Metal plugin
#   cmake -B build -DLUX_GPU_BUILD_CUDA=ON           # + CUDA plugin
#   cmake -B build -DLUX_GPU_BUILD_WEBGPU=ON         # + WebGPU plugin
#   cmake -B build -DLUX_GPU_BUILD_ALL_BACKENDS=ON   # All available

cmake_minimum_required(VERSION 3.20)

# Set macOS SDK before project() for OBJCXX to work properly
if(APPLE)
    execute_process(
        COMMAND xcrun --show-sdk-path
        OUTPUT_VARIABLE MACOS_SDK_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(CMAKE_OSX_SYSROOT ${MACOS_SDK_PATH})
endif()

project(lux-gpu VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Shared library settings
set(CMAKE_MACOSX_RPATH ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# =============================================================================
# Options
# =============================================================================

option(LUX_GPU_BUILD_METAL "Build Metal backend plugin" OFF)
option(LUX_GPU_BUILD_CUDA "Build CUDA backend plugin" OFF)
option(LUX_GPU_BUILD_WEBGPU "Build WebGPU backend plugin" OFF)
option(LUX_GPU_BUILD_ALL_BACKENDS "Build all available backends" OFF)
option(LUX_GPU_BUILD_TESTS "Build tests" ON)
option(LUX_GPU_BUILD_EXAMPLES "Build examples" ON)
option(LUX_GPU_BUILD_BENCHMARKS "Build benchmark suite" OFF)
option(LUX_GPU_EMBED_KERNELS "Embed kernel sources at build time" ON)

# =============================================================================
# Auto-detection when BUILD_ALL_BACKENDS is ON
# =============================================================================

if(LUX_GPU_BUILD_ALL_BACKENDS)
    # Metal on Apple
    if(APPLE)
        find_library(METAL_FRAMEWORK Metal)
        if(METAL_FRAMEWORK)
            set(LUX_GPU_BUILD_METAL ON CACHE BOOL "Metal available" FORCE)
            message(STATUS "Auto-detected Metal backend")
        endif()
    endif()

    # CUDA
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        set(LUX_GPU_BUILD_CUDA ON CACHE BOOL "CUDA available" FORCE)
        message(STATUS "Auto-detected CUDA backend")
    endif()

    # Dawn/WebGPU
    find_package(Dawn QUIET)
    if(NOT Dawn_FOUND)
        find_package(wgpu QUIET)
    endif()
    if(Dawn_FOUND OR wgpu_FOUND)
        set(LUX_GPU_BUILD_WEBGPU ON CACHE BOOL "WebGPU available" FORCE)
        message(STATUS "Auto-detected WebGPU backend")
    endif()
endif()

# =============================================================================
# Third-party paths
# =============================================================================

set(MLX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/mlx)
set(GPUCPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/gpu.cpp)

# =============================================================================
# Private Kernel Source Paths
# =============================================================================
# CUDA and Metal kernel sources are in private repos (cuda/, metal/).
# WebGPU/WGSL shaders are public and stay in gpu/kernels/wgsl/.

# Private repo paths (used during internal builds)
set(LUX_CUDA_PRIVATE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../cuda)
set(LUX_METAL_PRIVATE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../metal)

# Check if private repos are available
if(EXISTS ${LUX_CUDA_PRIVATE_DIR}/src/kernels)
    set(LUX_KERNEL_DIR_CUDA ${LUX_CUDA_PRIVATE_DIR}/src/kernels)
    message(STATUS "Using private CUDA kernels from: ${LUX_KERNEL_DIR_CUDA}")
else()
    set(LUX_KERNEL_DIR_CUDA ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuda)
    message(STATUS "Using embedded CUDA kernels from: ${LUX_KERNEL_DIR_CUDA}")
endif()

if(EXISTS ${LUX_METAL_PRIVATE_DIR}/src/shaders)
    set(LUX_KERNEL_DIR_METAL ${LUX_METAL_PRIVATE_DIR}/src/shaders)
    message(STATUS "Using private Metal shaders from: ${LUX_KERNEL_DIR_METAL}")
else()
    set(LUX_KERNEL_DIR_METAL ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal)
    message(STATUS "Using embedded Metal shaders from: ${LUX_KERNEL_DIR_METAL}")
endif()

# WebGPU shaders are always public
set(LUX_KERNEL_DIR_WEBGPU ${CMAKE_CURRENT_SOURCE_DIR}/kernels/wgsl)

# =============================================================================
# Kernel Embedding Infrastructure
# =============================================================================

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedKernels.cmake)

# =============================================================================
# Core Library (always built - both static and shared)
# =============================================================================

# Static library (for embedding in Go/Rust)
add_library(luxgpu_core_static STATIC
    src/gpu_core.cpp
    src/cpu_backend.cpp
    src/kernel_loader.cpp
)

target_include_directories(luxgpu_core_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_definitions(luxgpu_core_static PRIVATE
    LUX_GPU_BUILDING_CORE
)

set_target_properties(luxgpu_core_static PROPERTIES
    OUTPUT_NAME "luxgpu_core"
    POSITION_INDEPENDENT_CODE ON
)

# Shared library (for dynamic linking)
add_library(luxgpu_core SHARED
    src/gpu_core.cpp
    src/cpu_backend.cpp
    src/kernel_loader.cpp
)

target_include_directories(luxgpu_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_definitions(luxgpu_core PRIVATE
    LUX_GPU_BUILDING_CORE
)

# Link dl for plugin loading (Unix)
if(NOT WIN32)
    target_link_libraries(luxgpu_core PRIVATE dl)
    target_link_libraries(luxgpu_core_static INTERFACE dl)
endif()

# OpenMP for CPU backend (optional)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(luxgpu_core PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(luxgpu_core_static PRIVATE OpenMP::OpenMP_CXX)
endif()

# Set output name
set_target_properties(luxgpu_core PROPERTIES
    OUTPUT_NAME "luxgpu_core"
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)

# =============================================================================
# Metal Backend Plugin (macOS only)
# =============================================================================

if(LUX_GPU_BUILD_METAL AND APPLE)
    # Enable OBJCXX for Metal
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 17)

    add_library(luxgpu_backend_metal MODULE
        backend/metal/metal_plugin.mm
        backend/metal/metal_kernel_loader.mm
    )

    target_include_directories(luxgpu_backend_metal PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    set_source_files_properties(
        backend/metal/metal_plugin.mm
        backend/metal/metal_kernel_loader.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )

    target_link_libraries(luxgpu_backend_metal PRIVATE
        "-framework Metal"
        "-framework Foundation"
        "-framework CoreGraphics"
    )

    set_target_properties(luxgpu_backend_metal PROPERTIES
        OUTPUT_NAME "luxgpu_backend_metal"
        PREFIX "lib"
        SUFFIX ".dylib"
        BUNDLE FALSE
    )

    # Embed Metal kernel sources as string constants
    if(LUX_GPU_EMBED_KERNELS)
        lux_embed_kernels(luxgpu_backend_metal metal "${LUX_KERNEL_DIR_METAL}")
    endif()

    # Compile Metal shaders to .metallib for binary loading (optional - can be skipped for faster builds)
    option(LUX_GPU_COMPILE_METALLIB "Compile Metal shaders to .metallib" OFF)
    if(LUX_GPU_COMPILE_METALLIB)
        lux_compile_metal_shaders(luxgpu_backend_metal "${LUX_KERNEL_DIR_METAL}")
    endif()

    message(STATUS "Building Metal backend plugin")
endif()

# =============================================================================
# CUDA Backend Plugin
# =============================================================================

if(LUX_GPU_BUILD_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)

    add_library(luxgpu_backend_cuda MODULE
        backend/cuda/cuda_plugin.cpp
        backend/cuda/cuda_kernel_loader.cpp
        backend/cuda/cuda_kernels.cu
    )

    target_include_directories(luxgpu_backend_cuda PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_compile_definitions(luxgpu_backend_cuda PRIVATE
        LUX_GPU_BUILD_CUDA
    )

    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(luxgpu_backend_cuda PRIVATE
        CUDA::cuda_driver  # Use driver API for PTX loading
    )

    set_target_properties(luxgpu_backend_cuda PROPERTIES
        OUTPUT_NAME "luxgpu_backend_cuda"
        PREFIX "lib"
        CUDA_SEPARABLE_COMPILATION ON
    )

    # Embed CUDA kernel sources (PTX compiled at build time)
    if(LUX_GPU_EMBED_KERNELS)
        lux_embed_kernels(luxgpu_backend_cuda cuda "${LUX_KERNEL_DIR_CUDA}")
        lux_compile_cuda_ptx(luxgpu_backend_cuda "${LUX_KERNEL_DIR_CUDA}")
    endif()

    message(STATUS "Building CUDA backend plugin")
endif()

# =============================================================================
# WebGPU Backend Plugin
# =============================================================================

if(LUX_GPU_BUILD_WEBGPU)
    add_library(luxgpu_backend_webgpu MODULE
        backend/webgpu/webgpu_plugin.cpp
        backend/webgpu/webgpu_kernel_loader.cpp
    )

    target_include_directories(luxgpu_backend_webgpu PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_compile_definitions(luxgpu_backend_webgpu PRIVATE
        LUX_GPU_BUILD_WEBGPU
    )

    # gpu.cpp headers
    if(EXISTS ${GPUCPP_DIR})
        target_include_directories(luxgpu_backend_webgpu PRIVATE ${GPUCPP_DIR})
    endif()

    # Link Dawn or wgpu-native
    find_package(Dawn QUIET)
    if(Dawn_FOUND)
        target_link_libraries(luxgpu_backend_webgpu PRIVATE dawn::webgpu_dawn)
        target_compile_definitions(luxgpu_backend_webgpu PRIVATE USE_DAWN_API)
    else()
        find_package(wgpu QUIET)
        if(wgpu_FOUND)
            target_link_libraries(luxgpu_backend_webgpu PRIVATE wgpu::wgpu)
            target_compile_definitions(luxgpu_backend_webgpu PRIVATE USE_WGPU_API)
        else()
            # Manual detection of wgpu-native (e.g., from Homebrew)
            find_library(WGPU_LIBRARY NAMES wgpu_native PATHS /opt/homebrew/lib /usr/local/lib)
            find_path(WGPU_INCLUDE_DIR NAMES wgpu.h PATHS /opt/homebrew/include /usr/local/include)
            if(WGPU_LIBRARY AND WGPU_INCLUDE_DIR)
                message(STATUS "Found wgpu-native: ${WGPU_LIBRARY}")
                target_include_directories(luxgpu_backend_webgpu PRIVATE ${WGPU_INCLUDE_DIR})
                target_link_libraries(luxgpu_backend_webgpu PRIVATE ${WGPU_LIBRARY})
                target_compile_definitions(luxgpu_backend_webgpu PRIVATE USE_WGPU_API)
            else()
                message(WARNING "WebGPU backend enabled but neither Dawn nor wgpu found")
            endif()
        endif()
    endif()

    set_target_properties(luxgpu_backend_webgpu PROPERTIES
        OUTPUT_NAME "luxgpu_backend_webgpu"
        PREFIX "lib"
        SUFFIX ".dylib"
    )

    # Embed WGSL kernel sources
    if(LUX_GPU_EMBED_KERNELS)
        lux_embed_kernels(luxgpu_backend_webgpu webgpu "${LUX_KERNEL_DIR_WEBGPU}")
    endif()

    message(STATUS "Building WebGPU backend plugin")
endif()

# =============================================================================
# Tests
# =============================================================================

if(LUX_GPU_BUILD_TESTS)
    enable_testing()

    add_executable(test_gpu_core test/test_core.cpp)
    target_link_libraries(test_gpu_core PRIVATE luxgpu_core)
    add_test(NAME test_gpu_core COMMAND test_gpu_core)

    # Kernel loader test
    add_executable(test_kernel_loader test/test_kernel_loader.cpp)
    target_link_libraries(test_kernel_loader PRIVATE luxgpu_core)
    add_test(NAME test_kernel_loader COMMAND test_kernel_loader)

    # Set library path for tests to find plugins
    set_tests_properties(test_gpu_core test_kernel_loader PROPERTIES
        ENVIRONMENT "LUX_GPU_BACKEND_PATH=${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()

# =============================================================================
# pkg-config
# =============================================================================

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/lux-gpu.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/luxgpu.pc
    @ONLY
)

# =============================================================================
# Install
# =============================================================================

install(TARGETS luxgpu_core luxgpu_core_static
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install backend plugins to lib/lux-gpu/
if(TARGET luxgpu_backend_metal)
    install(TARGETS luxgpu_backend_metal LIBRARY DESTINATION lib/lux-gpu)
    # Install metallib files
    get_target_property(METALLIB_FILES luxgpu_backend_metal LUX_METALLIB_FILES)
    if(METALLIB_FILES)
        install(FILES ${METALLIB_FILES} DESTINATION lib/lux-gpu/shaders)
    endif()
endif()
if(TARGET luxgpu_backend_cuda)
    install(TARGETS luxgpu_backend_cuda LIBRARY DESTINATION lib/lux-gpu)
    # Install PTX files
    get_target_property(PTX_FILES luxgpu_backend_cuda LUX_PTX_FILES)
    if(PTX_FILES)
        install(FILES ${PTX_FILES} DESTINATION lib/lux-gpu/kernels)
    endif()
endif()
if(TARGET luxgpu_backend_webgpu)
    install(TARGETS luxgpu_backend_webgpu LIBRARY DESTINATION lib/lux-gpu)
endif()

# Headers
install(DIRECTORY include/lux DESTINATION include)

# pkg-config
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/luxgpu.pc
    DESTINATION lib/pkgconfig
)

# =============================================================================
# Benchmarks
# =============================================================================

if(LUX_GPU_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
    message(STATUS "Building benchmark suite")
endif()

# =============================================================================
# Status
# =============================================================================

message(STATUS "")
message(STATUS "Lux GPU Configuration:")
message(STATUS "  Core (CPU):      ON (always)")
message(STATUS "  Metal:           ${LUX_GPU_BUILD_METAL}")
message(STATUS "  CUDA:            ${LUX_GPU_BUILD_CUDA}")
message(STATUS "  WebGPU:          ${LUX_GPU_BUILD_WEBGPU}")
message(STATUS "  Embed Kernels:   ${LUX_GPU_EMBED_KERNELS}")
message(STATUS "  Benchmarks:      ${LUX_GPU_BUILD_BENCHMARKS}")
message(STATUS "")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
