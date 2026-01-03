# Copyright (c) 2024-2025 Lux Industries Inc.
# SPDX-License-Identifier: BSD-3-Clause-Ecosystem

# WebGPU ZK kernels using gpu.cpp

add_library(lux_zk_webgpu STATIC
    zk_webgpu.cpp
)

target_include_directories(lux_zk_webgpu
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Link to gpu.cpp (header-only, but needs Dawn)
target_link_libraries(lux_zk_webgpu
    PRIVATE
        gpu_cpp
)

# Install WGSL shaders
install(FILES
    poseidon2.wgsl
    merkle.wgsl
    DESTINATION share/lux/shaders/zk
)

# Install headers
install(FILES
    zk_webgpu.hpp
    DESTINATION include/lux/zk/webgpu
)
