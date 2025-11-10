# MLX Go Bindings - Platform Support

## ✅ Supported Platforms

### macOS (Darwin)
- **Apple Silicon (M1/M2/M3)**: Full Metal GPU acceleration
- **Intel x64**: CPU backend only
- **Backends**: Metal (ARM64), CPU (all)
- **Build**: Requires Xcode Command Line Tools

### Linux
- **x64/ARM64**: Full support
- **CUDA**: Available with NVIDIA GPU and CUDA toolkit
- **Backends**: CUDA (with GPU), CPU (always)
- **Build**: Requires gcc/g++ and build-essential

### Windows
- **x64/ARM64**: Full support
- **CUDA**: Available with NVIDIA GPU and CUDA toolkit
- **Backends**: CUDA (with GPU), CPU (always)
- **Build**: Requires MinGW-w64 or MSVC

## 🎯 Backend Selection

The library automatically selects the best available backend:

1. **Auto-detection** (default):
   - macOS ARM64 → Metal
   - Linux/Windows with NVIDIA GPU → CUDA
   - Otherwise → CPU

2. **Environment variable** override:
   ```bash
   export MLX_BACKEND=cpu    # Force CPU backend
   export MLX_BACKEND=metal  # Force Metal (macOS only)
   export MLX_BACKEND=cuda   # Force CUDA (Linux/Windows)
   export MLX_BACKEND=auto   # Auto-detect (default)
   ```

3. **Programmatic** selection:
   ```go
   import "github.com/luxfi/mlx"
   
   // Set backend
   mlx.SetBackend(mlx.CPU)    // Use CPU
   mlx.SetBackend(mlx.Metal)  // Use Metal
   mlx.SetBackend(mlx.CUDA)   // Use CUDA
   mlx.SetBackend(mlx.Auto)   // Auto-detect
   ```

## 🧪 Testing

### Local Testing

Run tests with specific backend:
```bash
# Test with CPU backend
MLX_BACKEND=cpu go test ./...

# Test with Metal backend (macOS ARM64)
MLX_BACKEND=metal go test ./...

# Test with CUDA backend (Linux/Windows with GPU)
MLX_BACKEND=cuda go test ./...

# Test with auto-detection
MLX_BACKEND=auto go test ./...
```

### Cross-Platform Testing Script

```bash
# Run comprehensive tests for your platform
./scripts/test-all-platforms.sh
```

### CI/CD Testing

The GitHub Actions workflow tests:
- **macOS ARM64**: Metal and CPU backends
- **macOS x64**: CPU backend
- **Linux**: CPU backend (CUDA if available)
- **Windows**: CPU backend (CUDA if available)

## 📊 Performance by Platform

| Platform | Backend | Matrix Multiply | Array Add | Notes |
|----------|---------|----------------|-----------|-------|
| macOS M2 | Metal | 656 GB/s | 3.8 GB/s | Unified memory |
| macOS M2 | CPU | 27 GB/s | 75 GB/s | SIMD optimized |
| Linux x64 | CUDA | ~500 GB/s | ~100 GB/s | Depends on GPU |
| Linux x64 | CPU | ~20 GB/s | ~50 GB/s | AVX2 optimized |
| Windows | CUDA | ~500 GB/s | ~100 GB/s | Depends on GPU |
| Windows | CPU | ~15 GB/s | ~40 GB/s | Compiler dependent |

## 🔧 Build Requirements

### macOS
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Build with Metal support
CGO_ENABLED=1 go build
```

### Linux
```bash
# Install build tools
sudo apt-get update
sudo apt-get install -y build-essential gcc g++

# For CUDA support (optional)
# Install CUDA Toolkit from NVIDIA

# Build
CGO_ENABLED=1 go build
```

### Windows
```bash
# Install MinGW-w64
# Or use Visual Studio with MSVC

# For CUDA support (optional)
# Install CUDA Toolkit from NVIDIA

# Build
set CGO_ENABLED=1
go build
```

## 🚀 Quick Start

```go
package main

import (
    "fmt"
    "github.com/luxfi/mlx"
)

func main() {
    // Auto-detect best backend
    backend := mlx.GetBackend()
    device := mlx.GetDevice()
    
    fmt.Printf("Using backend: %s\n", backend)
    fmt.Printf("Device: %s (Memory: %.1f GB)\n", 
        device.Name, 
        float64(device.Memory)/(1024*1024*1024))
    
    // Create arrays
    a := mlx.Ones([]int{1000, 1000}, mlx.Float32)
    b := mlx.Ones([]int{1000, 1000}, mlx.Float32)
    
    // Perform operations (uses selected backend)
    c := mlx.MatMul(a, b)
    
    // Force evaluation
    mlx.Eval(c)
    mlx.Synchronize()
    
    fmt.Println("Matrix multiplication completed!")
}
```

## 🔍 Troubleshooting

### macOS Issues
- **"Metal not available"**: Ensure you're on Apple Silicon (M1/M2/M3)
- **Build errors**: Update Xcode Command Line Tools

### Linux Issues
- **"CUDA not available"**: Install CUDA Toolkit and verify with `nvidia-smi`
- **Build errors**: Install `build-essential` package

### Windows Issues
- **CGO errors**: Install MinGW-w64 or use MSVC
- **"CUDA not available"**: Install CUDA Toolkit and NVIDIA drivers

### General Issues
- **Performance**: Ensure you're using the optimal backend for your hardware
- **Memory**: GPU backends require sufficient GPU memory
- **Compatibility**: CPU backend works everywhere as fallback

## 📈 Benchmarking

Run benchmarks on your platform:
```bash
# Quick benchmark
go test -bench=. -benchtime=1s ./...

# Detailed benchmark with specific backend
MLX_BACKEND=metal go test -bench=. -benchtime=10s ./...

# Compare backends
for backend in auto cpu metal cuda; do
    echo "Testing $backend..."
    MLX_BACKEND=$backend go test -bench=. ./... 2>/dev/null | grep Benchmark
done
```

## 🤝 Contributing

When adding platform support:
1. Test on target platform
2. Update CI/CD matrix
3. Document performance characteristics
4. Add platform-specific optimizations

## 📝 License

MIT License - See LICENSE file for details