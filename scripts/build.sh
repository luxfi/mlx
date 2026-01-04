#!/bin/sh
# Copyright (c) 2024-2025 Lux Industries Inc.
# SPDX-License-Identifier: BSD-3-Clause-Ecosystem
#
# Build script for lux-gpu library
# Usage: ./scripts/build.sh [options]
#
# Options:
#   --prefix=PATH       Installation prefix (default: /usr/local)
#   --build-type=TYPE   Debug or Release (default: Release)
#   --shared            Build shared library (default: static)
#   --cuda              Enable CUDA backend
#   --webgpu            Enable WebGPU backend
#   --clean             Clean build directory first
#   --install           Install after build
#   --help              Show this help

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build"

# Defaults
PREFIX="/usr/local"
BUILD_TYPE="Release"
SHARED_LIBS="OFF"
CUDA="OFF"
WEBGPU="OFF"
CLEAN=""
INSTALL=""

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --prefix=*)
            PREFIX="${arg#*=}"
            ;;
        --build-type=*)
            BUILD_TYPE="${arg#*=}"
            ;;
        --shared)
            SHARED_LIBS="ON"
            ;;
        --cuda)
            CUDA="ON"
            ;;
        --webgpu)
            WEBGPU="ON"
            ;;
        --clean)
            CLEAN="1"
            ;;
        --install)
            INSTALL="1"
            ;;
        --help)
            head -n 16 "$0" | tail -n +2 | sed 's/^# //'
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 1
            ;;
    esac
done

# Clean if requested
if [ -n "$CLEAN" ] && [ -d "$BUILD_DIR" ]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
echo "Configuring lux-gpu..."
echo "  Prefix: $PREFIX"
echo "  Build type: $BUILD_TYPE"
echo "  Shared libs: $SHARED_LIBS"
echo "  CUDA: $CUDA"
echo "  WebGPU: $WEBGPU"

cmake "$PROJECT_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DBUILD_SHARED_LIBS="$SHARED_LIBS" \
    -DMLX_BUILD_CUDA="$CUDA" \
    -DMLX_BUILD_WEBGPU="$WEBGPU" \
    -DMLX_BUILD_TESTS=ON \
    -DMLX_BUILD_EXAMPLES=OFF

# Build
echo "Building lux-gpu..."
cmake --build . --parallel

# Install if requested
if [ -n "$INSTALL" ]; then
    echo "Installing to $PREFIX..."
    cmake --install .
fi

echo "Build complete."
echo ""
echo "Library: $BUILD_DIR/liblux-gpu.*"
echo ""
echo "To install: cmake --install $BUILD_DIR"
echo "Or run: $0 --install"
