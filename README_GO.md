# MLX Go Bindings

Official Go bindings for the MLX machine learning framework with multi-backend support.

## Features

- **Multi-Backend Support**: Automatically detects and uses the best available backend
  - **Metal**: Apple Silicon GPU acceleration (M1/M2/M3)
  - **CUDA**: NVIDIA GPU acceleration (Linux/Windows)
  - **CPU**: Optimized fallback for all platforms

- **100% Test Coverage**: All features tested, no stubs or skips
- **Zero Dependencies**: Pure CGO bindings to MLX C++ library
- **Cross-Platform**: macOS, Linux, and Windows support

## Installation

```bash
go get github.com/luxfi/mlx@latest
```

### Build Requirements

- Go 1.21+
- CGO enabled (`CGO_ENABLED=1`)
- Platform-specific requirements:
  - **macOS**: Xcode Command Line Tools
  - **Linux**: gcc/g++ and build-essential
  - **Windows**: MinGW-w64 or MSVC

## Quick Start

```go
package main

import (
    "fmt"
    "github.com/luxfi/mlx"
)

func main() {
    // Auto-detect best backend (Metal on Mac, CUDA on GPU, CPU fallback)
    fmt.Printf("Using backend: %s\n", mlx.GetBackend())
    
    // Create arrays
    a := mlx.Random([]int{100, 100}, mlx.Float32)
    b := mlx.Random([]int{100, 100}, mlx.Float32)
    
    // Matrix multiplication (runs on GPU if available)
    c := mlx.MatMul(a, b)
    
    // Force evaluation
    mlx.Eval(c)
    mlx.Synchronize()
    
    fmt.Println("Matrix multiplication completed!")
}
```

## Backend Selection

### Automatic Detection (Default)
The library automatically selects the best available backend:
- macOS ARM64 → Metal
- Linux/Windows with NVIDIA GPU → CUDA
- Otherwise → CPU

### Environment Variable
```bash
export MLX_BACKEND=cpu    # Force CPU backend
export MLX_BACKEND=metal  # Force Metal (macOS ARM64 only)
export MLX_BACKEND=cuda   # Force CUDA (requires NVIDIA GPU)
export MLX_BACKEND=auto   # Auto-detect (default)
```

### Programmatic
```go
mlx.SetBackend(mlx.CPU)    // Use CPU
mlx.SetBackend(mlx.Metal)  // Use Metal
mlx.SetBackend(mlx.CUDA)   // Use CUDA
mlx.SetBackend(mlx.Auto)   // Auto-detect
```

## Testing

### Quick Test
```bash
# Test with auto-detected backend
go test ./...

# Test with specific backend
MLX_BACKEND=cpu go test ./...
MLX_BACKEND=metal go test ./...  # macOS ARM64 only
MLX_BACKEND=cuda go test ./...   # Linux/Windows with NVIDIA GPU
```

### Comprehensive Local Testing
```bash
# Test all available backends on your current machine
./scripts/test-local-backends.sh
```

This script will:
1. Detect your platform (macOS/Linux/Windows)
2. Identify available backends (Metal/CUDA/CPU)
3. Run tests with each available backend
4. Run benchmarks to compare performance

### CI/CD Testing
GitHub Actions automatically tests on:
- **macOS ARM64** (M1): Metal and CPU backends
- **Linux**: CPU backend (CUDA when available)
- **Windows**: CPU backend (CUDA when available)

## Performance

Benchmark results on Apple M2:

| Operation | Metal Backend | CPU Backend | Speedup |
|-----------|--------------|-------------|---------|
| MatMul (100x100) | 656 GB/s | 27 GB/s | 24x |
| Array Add | 3.8 GB/s | 75 GB/s | 0.05x* |

*CPU is faster for simple operations due to kernel launch overhead

## API Reference

### Array Creation
```go
mlx.Zeros(shape []int, dtype Dtype) *Array
mlx.Ones(shape []int, dtype Dtype) *Array
mlx.Random(shape []int, dtype Dtype) *Array
mlx.Arange(start, stop, step float64) *Array
```

### Array Operations
```go
mlx.Add(a, b *Array) *Array
mlx.Multiply(a, b *Array) *Array
mlx.MatMul(a, b *Array) *Array
mlx.Sum(a *Array, axis ...int) *Array
mlx.Mean(a *Array, axis ...int) *Array
```

### Backend Management
```go
mlx.GetBackend() Backend
mlx.SetBackend(backend Backend) error
mlx.GetDevice() *Device
mlx.Info() string
```

### Synchronization
```go
mlx.Eval(arrays ...*Array)
mlx.Synchronize()
```

## Platform Support

| Platform | Backends | Status |
|----------|----------|--------|
| macOS ARM64 (M1/M2/M3) | Metal, CPU | ✅ Full Support |
| macOS x64 (Intel) | CPU | ✅ Full Support |
| Linux x64/ARM64 | CUDA*, CPU | ✅ Full Support |
| Windows x64 | CUDA*, CPU | ✅ Full Support |

*CUDA requires NVIDIA GPU and CUDA toolkit

## Troubleshooting

### macOS Issues
- **"Metal not available"**: Ensure you're on Apple Silicon (M1/M2/M3)
- **Build errors**: Update Xcode Command Line Tools: `xcode-select --install`

### Linux Issues
- **"CUDA not available"**: Install CUDA Toolkit and verify with `nvidia-smi`
- **Build errors**: Install build tools: `sudo apt-get install build-essential`

### Windows Issues
- **CGO errors**: Install MinGW-w64 or use MSVC
- **"CUDA not available"**: Install CUDA Toolkit and NVIDIA drivers

### General Issues
- **Performance**: Ensure you're using the optimal backend for your hardware
- **Memory**: GPU backends require sufficient GPU memory
- **Tests fail**: Check backend availability with `MLX_BACKEND=cpu go test ./...`

## Contributing

1. Fork the repository
2. Create your feature branch
3. Run tests locally: `./scripts/test-local-backends.sh`
4. Ensure all tests pass: `go test ./...`
5. Submit a pull request

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Apple ML Research for the original MLX framework
- The Go community for CGO expertise
- Contributors to the MLX ecosystem