# Copyright (c) 2024-2026 Lux Industries Inc.
# SPDX-License-Identifier: BSD-3-Clause-Eco
#
# WebGPU Backend CMakeLists.txt
# Provides portable GPU compute via Dawn/wgpu-native
# Supports Metal, Vulkan, and D3D12 backends

cmake_minimum_required(VERSION 3.20)

option(LUX_BUILD_WEBGPU "Build WebGPU portable backend" OFF)

if(LUX_BUILD_WEBGPU)
    message(STATUS "WebGPU Backend: Enabled")

    # Find Dawn or wgpu-native
    find_package(Dawn QUIET)
    if(NOT Dawn_FOUND)
        find_package(wgpu QUIET)
    endif()

    if(Dawn_FOUND OR wgpu_FOUND)
        set(WEBGPU_FOUND TRUE)
    else()
        message(STATUS "WebGPU (Dawn/wgpu) not found. Attempting to fetch Dawn...")
        
        include(FetchContent)
        FetchContent_Declare(
            dawn
            GIT_REPOSITORY https://dawn.googlesource.com/dawn
            GIT_TAG main
            GIT_SHALLOW TRUE
        )
        
        # Configure Dawn options
        set(DAWN_ENABLE_D3D11 OFF CACHE BOOL "" FORCE)
        set(DAWN_ENABLE_D3D12 ON CACHE BOOL "" FORCE)
        set(DAWN_ENABLE_METAL ON CACHE BOOL "" FORCE)
        set(DAWN_ENABLE_VULKAN ON CACHE BOOL "" FORCE)
        set(DAWN_ENABLE_NULL OFF CACHE BOOL "" FORCE)
        set(DAWN_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
        set(TINT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(TINT_BUILD_CMD_TOOLS OFF CACHE BOOL "" FORCE)
        
        # Note: Dawn fetch is very slow, may want to use pre-built binaries
        # FetchContent_MakeAvailable(dawn)
        # set(WEBGPU_FOUND TRUE)
        
        message(WARNING "Dawn auto-fetch disabled. Install Dawn manually or use pre-built binaries.")
        set(WEBGPU_FOUND FALSE)
    endif()

    if(WEBGPU_FOUND)
        # Source files
        set(WEBGPU_SOURCES
            gpu.cpp
        )

        # Collect WGSL kernels
        file(GLOB WGSL_KERNELS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.wgsl")
        
        # Generate embedded kernel headers
        set(KERNEL_HEADER "${CMAKE_CURRENT_BINARY_DIR}/embedded_kernels.h")
        
        # Create the object library
        add_library(webgpu_backend OBJECT ${WEBGPU_SOURCES})
        
        target_include_directories(webgpu_backend PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/..
            ${CMAKE_CURRENT_BINARY_DIR}
        )
        
        target_compile_definitions(webgpu_backend PUBLIC
            MLX_BUILD_WEBGPU
        )
        
        if(Dawn_FOUND)
            target_link_libraries(webgpu_backend PUBLIC dawn::webgpu_dawn)
            target_compile_definitions(webgpu_backend PUBLIC USE_DAWN_API)
        elseif(wgpu_FOUND)
            target_link_libraries(webgpu_backend PUBLIC wgpu::wgpu)
        endif()
        
        target_compile_features(webgpu_backend PUBLIC cxx_std_17)
        
        # Export for parent scope
        set(WEBGPU_BACKEND_TARGET webgpu_backend PARENT_SCOPE)
        
        message(STATUS "WebGPU Backend: Configured successfully")
        message(STATUS "  WGSL Kernels: ${WGSL_KERNELS}")
    else()
        message(STATUS "WebGPU Backend: Disabled (Dawn/wgpu not available)")
    endif()
else()
    message(STATUS "WebGPU Backend: Disabled")
endif()
