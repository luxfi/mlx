# Pre-compiled MLX Binaries

Pre-compiled `libmlx.a` binaries are available for major platforms, eliminating the need to build from source.

## Quick Download

### macOS ARM64 (M1/M2/M3) - Metal GPU

```bash
# Download latest release
curl -LO https://github.com/luxfi/mlx/releases/latest/download/libmlx-macos-arm64.tar.gz

# Extract to lib directory
mkdir -p lib
tar -xzf libmlx-macos-arm64.tar.gz -C lib/
```

### macOS x64 - CPU

```bash
curl -LO https://github.com/luxfi/mlx/releases/latest/download/libmlx-macos-x64.tar.gz
mkdir -p lib
tar -xzf libmlx-macos-x64.tar.gz -C lib/
```

### Linux x64 - CPU

```bash
curl -LO https://github.com/luxfi/mlx/releases/latest/download/libmlx-linux-x64.tar.gz
mkdir -p lib
tar -xzf libmlx-linux-x64.tar.gz -C lib/
```

### Windows x64 - CPU

```bash
curl -LO https://github.com/luxfi/mlx/releases/latest/download/libmlx-windows-x64.tar.gz
mkdir -p lib
tar -xzf libmlx-windows-x64.tar.gz -C lib/
```

## Using with Go

After downloading the appropriate binary:

```go
package main

import (
    "fmt"
    "github.com/luxfi/mlx"
)

func main() {
    // Initialize MLX context
    ctx := mlx.NewContext()
    defer ctx.Close()

    // Check detected backend
    fmt.Println("Backend:", mlx.GetBackend())

    // Create and manipulate arrays
    a := mlx.Zeros([]int{3, 3}, mlx.Float32)
    b := mlx.Ones([]int{3, 3}, mlx.Float32)

    c := mlx.Add(a, b)
    mlx.Eval(c)

    fmt.Println("Computation complete!")
}
```

Build with CGO enabled:

```bash
CGO_ENABLED=1 go build
```

## Platform Support

| Platform | Architecture | Acceleration | Binary Size | Status |
|----------|-------------|--------------|-------------|--------|
| macOS | ARM64 (Apple Silicon) | Metal GPU | ~84 MB | ✅ Available |
| macOS | x86_64 (Intel) | CPU | ~84 MB | ✅ Available |
| Linux | x86_64 | CPU | ~84 MB | ✅ Available |
| Windows | x86_64 | CPU | ~84 MB | ✅ Available |

## Building from Source

If you prefer to build from source or need a custom configuration:

```bash
# Clone repository
git clone https://github.com/luxfi/mlx.git
cd mlx

# Build MLX C++ library
mkdir -p build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DMLX_BUILD_METAL=ON  # OFF for CPU-only builds

# macOS
make -j$(sysctl -n hw.ncpu) mlx

# Linux
make -j$(nproc) mlx

# Windows (MinGW)
cmake --build . --target mlx -j

# Copy to lib directory
mkdir -p ../lib
cp libmlx.a ../lib/
```

## Release Process

Pre-compiled binaries are automatically built and published when a new version tag is pushed:

```bash
git tag -a v0.29.4 -m "Release v0.29.4"
git push origin v0.29.4
```

The `build-release.yml` workflow compiles binaries for all platforms and attaches them to the GitHub release.

## CI Integration

GitHub Actions workflows automatically download pre-built binaries when available, falling back to source builds if needed. This significantly speeds up CI runs.

## Checksums

Each release includes SHA256 checksums. Verify your download:

```bash
# Download checksum file
curl -LO https://github.com/luxfi/mlx/releases/latest/download/checksums.txt

# Verify (macOS/Linux)
shasum -a 256 -c checksums.txt

# Verify (Windows PowerShell)
Get-FileHash libmlx-windows-x64.tar.gz -Algorithm SHA256
```

## Troubleshooting

### "library 'mlx' not found" error

Ensure `libmlx.a` is in the `lib/` directory:

```bash
ls -lh lib/libmlx.a
```

### Wrong platform binary

Check your platform:

```bash
# macOS
uname -m  # Should show arm64 or x86_64

# Linux
uname -m  # Should show x86_64

# Windows
echo %PROCESSOR_ARCHITECTURE%  # Should show AMD64
```

### Build from source instead

If pre-built binaries don't work, follow the build instructions above.

## Support

- **Issues**: https://github.com/luxfi/mlx/issues
- **Discussions**: https://github.com/luxfi/mlx/discussions
- **Documentation**: https://github.com/luxfi/mlx/blob/main/README_GO.md
