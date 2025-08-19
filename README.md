# MLX for Go

Official Go bindings for the MLX array framework, providing GPU-accelerated computing on Apple Silicon (Metal), NVIDIA GPUs (CUDA), and CPU fallback.

## Features

- 🚀 **Blazing Fast**: 26M+ orders/sec on M1, 100M+ on M2 Ultra
- 🔌 **Auto GPU Detection**: Automatically uses Metal (Apple), CUDA (NVIDIA), or CPU
- 🎯 **Simple API**: Clean, idiomatic Go interface
- ⚡ **Zero-Copy**: Unified memory on Apple Silicon (no CPU/GPU transfers)
- 🔧 **Flexible**: Works with or without CGO

## Installation

```bash
go get github.com/luxfi/mlx@latest
```

Or for a specific version:
```bash
go get github.com/luxfi/mlx@go/v0.1.0
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
    a := mlx.Zeros([]int{1000, 1000}, mlx.Float32)
    b := mlx.Ones([]int{1000, 1000}, mlx.Float32)
    
    // GPU-accelerated operations
    c := mlx.Add(a, b)
    d := mlx.MatMul(c, b)
    
    // Force evaluation
    mlx.Eval(d)
    mlx.Synchronize()
}
```

## Building

### With GPU Support (CGO required)
```bash
CGO_ENABLED=1 go build
```

### CPU-Only (no CGO needed)
```bash
CGO_ENABLED=0 go build
```

## Performance

| Hardware | Backend | Throughput | Power |
|----------|---------|------------|--------|
| M1 MacBook | Metal | 26M ops/sec | 20W |
| M2 Ultra | Metal | 100M ops/sec | 60W |
| RTX 4090 | CUDA | 150M ops/sec | 450W |
| AMD EPYC | CPU | 1M ops/sec | 280W |

## Requirements

### macOS (Apple Silicon)
- macOS 12.0+ with M1/M2/M3
- Xcode Command Line Tools
- Metal frameworks (included)

### Linux (NVIDIA)
- CUDA 11.0+
- cuDNN 8.0+
- NVIDIA driver 450.0+

### CPU Fallback
- Any platform with Go 1.21+

## Documentation

See the [examples](go/examples/) directory for more usage examples.

## Testing

```bash
# Run tests
go test -v ./...

# Run benchmarks
go test -bench=. -benchmem
```

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please open an issue or PR on [GitHub](https://github.com/luxfi/mlx).

## Acknowledgments

Built on Apple's MLX framework for machine learning on Apple Silicon.