# MLX Go Bindings

Official Go bindings for the MLX machine learning framework with cross-platform GPU support.

## Features

- **Multi-backend support**:
  - **Metal**: Apple Silicon GPUs (M1/M2/M3) - Automatic on macOS
  - **CUDA**: NVIDIA GPUs - Linux/Windows with CUDA toolkit
  - **CPU**: Optimized fallback for all platforms with SIMD
- **Automatic backend detection** - Uses the best available hardware
- **Unified API** across all backends
- **Zero-copy operations** on Apple Silicon (unified memory)
- **High performance** - Billions of ops/sec on GPU

## Installation

```bash
go get github.com/luxfi/mlx@latest
```

## Quick Start

```go
package main

import (
    "fmt"
    "github.com/luxfi/mlx"
)

func main() {
    // Display system info
    fmt.Println(mlx.Info())
    
    // Create arrays
    a := mlx.Random([]int{1000, 1000}, mlx.Float32)
    b := mlx.Random([]int{1000, 1000}, mlx.Float32)
    
    // Matrix multiplication (runs on GPU automatically)
    c := mlx.MatMul(a, b)
    
    // Force evaluation
    mlx.Eval(c)
    mlx.Synchronize()
}
```

## Documentation

See the [full documentation](https://github.com/luxfi/mlx#go) for detailed API reference, benchmarks, and platform-specific notes.

## License

Apache 2.0
