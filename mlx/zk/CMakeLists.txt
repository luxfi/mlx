# Copyright © 2024-2025 Lux Industries Inc.
# SPDX-License-Identifier: Apache-2.0
#
# ZK (Zero-Knowledge) Module for MLX

target_sources(
  mlx
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/zk.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/zk_c_api.cpp
)

# Install headers
install(
  FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/zk.h
  ${CMAKE_CURRENT_SOURCE_DIR}/zk_c_api.h
  DESTINATION include/mlx/zk
)
