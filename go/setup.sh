#!/bin/bash

# Setup script for github.com/luxfi/mlx Go package

set -e

echo "🚀 Setting up github.com/luxfi/mlx package"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Detect platform
PLATFORM=$(uname -s)
ARCH=$(uname -m)

echo -e "${GREEN}Platform:${NC} $PLATFORM $ARCH"

# Check Go version
GO_VERSION=$(go version 2>/dev/null || echo "not installed")
echo -e "${GREEN}Go version:${NC} $GO_VERSION"

if [[ "$GO_VERSION" == "not installed" ]]; then
    echo -e "${RED}Error: Go is not installed${NC}"
    exit 1
fi

# Check for GPU support
echo ""
echo "Checking GPU support..."

if [[ "$PLATFORM" == "Darwin" ]]; then
    # macOS - check for Apple Silicon
    if [[ "$ARCH" == "arm64" ]]; then
        echo -e "${GREEN}✓ Apple Silicon detected - Metal support available${NC}"
        GPU_BACKEND="Metal"
    else
        echo -e "${YELLOW}Intel Mac detected - CPU only${NC}"
        GPU_BACKEND="CPU"
    fi
elif [[ "$PLATFORM" == "Linux" ]]; then
    # Linux - check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✓ NVIDIA GPU detected - CUDA support available${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
        GPU_BACKEND="CUDA"
    else
        echo -e "${YELLOW}No NVIDIA GPU detected - CPU only${NC}"
        GPU_BACKEND="CPU"
    fi
else
    echo -e "${YELLOW}Unknown platform - CPU only${NC}"
    GPU_BACKEND="CPU"
fi

# Build the package
echo ""
echo "Building MLX package..."

if [[ "$GPU_BACKEND" != "CPU" ]]; then
    echo "Building with CGO for GPU support..."
    CGO_ENABLED=1 go build -v .
else
    echo "Building without CGO (CPU only)..."
    CGO_ENABLED=0 go build -v .
fi

# Run tests
echo ""
echo "Running tests..."
go test -v ./...

# Run benchmarks
echo ""
echo "Running benchmarks..."
go test -bench=. -benchmem ./... || true

# Create example binary
echo ""
echo "Building example..."
cd example
go build -o mlx-demo .
cd ..

# Show usage
echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "Usage:"
echo "  import \"github.com/luxfi/mlx\""
echo ""
echo "Run example:"
echo "  cd example && ./mlx-demo"
echo ""
echo "Run tests:"
echo "  go test -v ./..."
echo ""
echo "Run benchmarks:"
echo "  go test -bench=. ./..."
echo ""
echo "Backend: $GPU_BACKEND"
echo ""

# Optional: Set up as local module for development
echo "To use this package locally in the DEX project:"
echo "  cd /Users/z/work/lx/dex"
echo "  go mod edit -replace github.com/luxfi/mlx=./luxfi-mlx-package"
echo ""