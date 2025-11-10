#!/bin/bash

# Test script for available backends on the current machine

set -e

echo "========================================="
echo "MLX Go Bindings - Local Backend Testing"
echo "========================================="
echo ""
echo "This script tests all available backends on your current machine."
echo "It does NOT test all platforms - only what's available locally."
echo ""

# Detect platform
OS=$(uname -s)
ARCH=$(uname -m)

echo "Current Machine: $OS/$ARCH"
echo ""

# Function to run tests with a specific backend
test_backend() {
    local backend=$1
    echo "----------------------------------------"
    echo "Testing with backend: $backend"
    echo "----------------------------------------"
    
    export MLX_BACKEND=$backend
    export CGO_ENABLED=1
    
    # Run tests
    if go test -v -timeout 30s ./... > /tmp/mlx_test_$backend.log 2>&1; then
        echo "✅ Tests PASSED with $backend backend"
        grep -E "^(PASS|ok)" /tmp/mlx_test_$backend.log
    else
        echo "❌ Tests FAILED with $backend backend"
        tail -20 /tmp/mlx_test_$backend.log
        exit 1
    fi
    
    # Run benchmarks (short)
    echo "Running benchmarks..."
    if go test -bench=. -benchtime=100ms ./... > /tmp/mlx_bench_$backend.log 2>&1; then
        echo "✅ Benchmarks completed"
        grep -E "Benchmark" /tmp/mlx_bench_$backend.log | head -5
    else
        echo "⚠️  Benchmarks had issues"
    fi
    
    echo ""
}

# Detect available backends on this machine
echo "Detecting available backends on this machine..."
AVAILABLE_BACKENDS=()

# Always have CPU
AVAILABLE_BACKENDS+=("cpu")
echo "  ✓ CPU backend available"

# Check for Metal (macOS ARM64 only)
if [[ "$OS" == "Darwin" ]] && [[ "$ARCH" == "arm64" ]]; then
    AVAILABLE_BACKENDS+=("metal")
    echo "  ✓ Metal backend available (Apple Silicon)"
elif [[ "$OS" == "Darwin" ]]; then
    echo "  ✗ Metal not available (Intel Mac - CPU only)"
fi

# Check for CUDA (Linux/Windows with NVIDIA GPU)
if command -v nvidia-smi &> /dev/null; then
    AVAILABLE_BACKENDS+=("cuda")
    echo "  ✓ CUDA backend available (NVIDIA GPU detected)"
else
    echo "  ✗ CUDA not available (no NVIDIA GPU)"
fi

echo ""
echo "Will test the following backends: ${AVAILABLE_BACKENDS[@]}"
echo ""

# Test auto-detection first
echo "========================================="
echo "Testing auto-detection..."
test_backend "auto"

# Test each available backend
for backend in "${AVAILABLE_BACKENDS[@]}"; do
    echo "========================================="
    test_backend "$backend"
done

echo "========================================="
echo "Local backend tests completed successfully!"
echo "========================================="
echo ""
echo "Summary:"
echo "- Machine: $OS/$ARCH"
echo "- Available backends: ${AVAILABLE_BACKENDS[@]}"
echo "- Backends tested: $(ls /tmp/mlx_test_*.log 2>/dev/null | sed 's/.*mlx_test_//' | sed 's/.log//' | tr '\n' ', ' | sed 's/, $//')"
echo "- All tests: ✅ PASSED"
echo ""
echo "Note: To test on other platforms, run this script on those machines."
echo "GitHub Actions CI will test Linux and Windows automatically."