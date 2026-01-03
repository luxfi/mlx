# lux-gpu

GPU acceleration foundation for the Lux crypto stack.

## Overview

This library provides high-performance array operations accelerated by Metal (Apple Silicon) and CUDA (NVIDIA). It serves as the foundation layer for all GPU-accelerated cryptographic operations in the Lux ecosystem.

Based on [MLX](https://github.com/ml-explore/mlx) from Apple machine learning research, with extensions for cryptographic workloads.

## Features

- **Unified Memory** - Arrays live in shared memory, accessible from CPU and GPU
- **Lazy Evaluation** - Computations deferred until results needed
- **Metal Backend** - Native Apple Silicon GPU acceleration
- **CUDA Backend** - NVIDIA GPU support (planned)
- **FFT/NTT** - Optimized transforms for polynomial arithmetic
- **Batch Operations** - Parallel processing of independent operations

## Dependencies

Built on top of:
- **lux-gpu** (this library) - Base array operations

Used by:
- **lux-lattice** - NTT acceleration for lattice cryptography
- **lux-crypto** - BLS pairing acceleration

## Installation

```bash
cmake -B build -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build -j
cmake --install build
```

## Usage

```cpp
#include <lux/gpu/array.h>
#include <lux/gpu/ops.h>

// Create arrays
auto a = lux::gpu::array({1.0f, 2.0f, 3.0f, 4.0f});
auto b = lux::gpu::array({5.0f, 6.0f, 7.0f, 8.0f});

// GPU-accelerated operations
auto c = lux::gpu::add(a, b);
auto d = lux::gpu::matmul(a.reshape({2, 2}), b.reshape({2, 2}));

// FFT for signal processing
auto spectrum = lux::gpu::fft::fft(a);
```

## CMake Integration

```cmake
find_package(lux-gpu REQUIRED)
target_link_libraries(myapp PRIVATE lux::gpu)
```

## Go Bindings

See [luxfi/crypto](https://github.com/luxfi/crypto) for Go bindings that wrap this library.

## Architecture

```
lux-gpu (this)     ← Foundation (Metal/CUDA)
    ▲
lux-lattice        ← NTT acceleration
    ▲
lux-fhe            ← TFHE/CKKS/BGV
```

## Documentation

- [GPU Acceleration Guide](https://luxfi.github.io/crypto/docs/gpu-acceleration)
- [C++ Libraries Overview](https://luxfi.github.io/crypto/docs/cpp-libraries)

## License

MIT License - see [LICENSE](LICENSE)

## Links

- [lux-lattice](https://github.com/luxcpp/lattice) - Lattice cryptography
- [lux-fhe](https://github.com/luxcpp/fhe) - Fully Homomorphic Encryption
- [lux-crypto](https://github.com/luxcpp/crypto) - Core cryptography
- [luxfi/crypto](https://github.com/luxfi/crypto) - Go bindings
