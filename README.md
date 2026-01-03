# Lux GPU - High-Performance Cryptographic GPU Library

> **Cross-platform GPU acceleration for cryptography, FHE, and zero-knowledge proofs**

[![License](https://img.shields.io/badge/License-BSD--3--Clause--Eco-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/std/the-standard)

## Overview

Lux GPU is a high-performance GPU compute library optimized for cryptographic operations. It provides portable, production-ready implementations of:

- **Number Theoretic Transform (NTT)** - Foundation for polynomial multiplication
- **Fast Fourier Transform (FFT)** - Complex signal processing  
- **Elliptic Curve Operations** - BLS12-381, BN254 curve arithmetic
- **Multi-Scalar Multiplication (MSM)** - Batched elliptic curve operations
- **Cryptographic Hashing** - Poseidon, Blake3
- **Fully Homomorphic Encryption (FHE)** - TFHE blind rotation, CKKS

## Backend Support

| Backend | Platform | Status |
|---------|----------|--------|
| **Metal** | macOS/iOS (Apple Silicon) | ✅ Full Support |
| **WebGPU** | Cross-platform (via Dawn/wgpu) | ✅ Full Support |
| **CPU** | All platforms (SIMD) | ✅ Fallback |
| **CUDA** | NVIDIA GPUs | 🔒 Private (contact us) |

## Quick Start

### Prerequisites

- CMake 3.20+
- C++17 compiler
- For Metal: Xcode 12+ on macOS
- For WebGPU: Dawn or wgpu-native

### Building

```bash
# Clone the repository
git clone https://github.com/luxfi/gpu.git
cd gpu

# Create build directory
mkdir build && cd build

# Configure with desired backends
cmake .. \
  -DLUX_BUILD_METAL=ON \
  -DLUX_BUILD_WEBGPU=OFF \
  -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

# Install
sudo make install
```

### CMake Integration

```cmake
find_package(lux-gpu REQUIRED)
target_link_libraries(your_target PRIVATE lux::gpu)
```

## Usage Examples

### NTT (Number Theoretic Transform)

```cpp
#include <lux/gpu/ntt.h>

// Initialize NTT context
auto ctx = lux::gpu::NttContext::create(1024);  // N = 1024

// Forward NTT
std::vector<uint64_t> poly(1024);
ctx->forward(poly.data(), poly.size());

// Inverse NTT
ctx->inverse(poly.data(), poly.size());
```

### BLS12-381 Operations

```cpp
#include <lux/gpu/bls12_381.h>

using namespace lux::gpu;

// Point multiplication
auto G1 = bls12::G1Affine::generator();
auto scalar = bls12::Scalar::from_bytes(data);
auto result = bls12::g1_mul(G1, scalar);

// Batch MSM (Multi-Scalar Multiplication)
std::vector<bls12::G1Affine> points = {...};
std::vector<bls12::Scalar> scalars = {...};
auto msm_result = bls12::msm(points, scalars);
```

### Poseidon Hash

```cpp
#include <lux/gpu/poseidon.h>

// Hash two field elements
auto a = lux::gpu::Fe::from_u64(42);
auto b = lux::gpu::Fe::from_u64(123);
auto hash = lux::gpu::poseidon_hash_2(a, b);
```

## Architecture

```
lux-gpu/
├── mlx/                    # Core library
│   ├── backend/
│   │   ├── metal/         # Apple Metal shaders (.metal)
│   │   └── webgpu/        # Portable WGSL shaders (.wgsl)
│   └── kernels/           # Kernel registry and dispatch
├── include/               # Public headers
├── benchmarks/            # Performance tests
└── tests/                 # Unit tests
```

## Performance

Benchmarked on Apple M1 Max:

| Operation | Lux GPU | Reference | Speedup |
|-----------|---------|-----------|---------|
| NTT (N=2^20) | 2.1ms | 12ms (CPU) | 5.7x |
| MSM (2^16 points) | 48ms | 320ms (CPU) | 6.7x |
| Poseidon (batch 10K) | 0.8ms | 8ms (CPU) | 10x |
| Blind Rotate (TFHE) | 1.2ms | 15ms (CPU) | 12.5x |

## CUDA Support

High-performance CUDA kernels are available for NVIDIA GPUs through a separate commercial license. These provide:

- 2-3x faster MSM than open-source alternatives
- Optimized memory access patterns
- Multi-GPU support
- Production-ready for blockchain validators

**Contact**: licensing@luxindustries.xyz

## License

BSD 3-Clause License - Ecosystem Edition (BSD-3-Clause-Eco)

```
Copyright (c) 2024-2026 Lux Industries Inc.

Commercial use of this software is permitted provided that the software 
operates as part of, or in connection with, the Lux Network of blockchains.

For external commercial licensing, contact: licensing@luxindustries.xyz
```

See [LICENSE](LICENSE) for full terms.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Related Projects

- [lux/node](https://github.com/luxfi/node) - Lux blockchain node
- [lux/coreth](https://github.com/luxfi/coreth) - EVM implementation
- [lux/fhe](https://github.com/luxfi/fhe) - FHE library using lux-gpu
- [lux/crypto](https://github.com/luxfi/crypto) - Cryptographic primitives

## Support

- **Documentation**: https://docs.lux.industries/gpu
- **Issues**: https://github.com/luxfi/gpu/issues
- **Discord**: https://discord.gg/lux

---

Built with ❤️ by [Lux Industries Inc.](https://lux.industries) | [Hanzo AI](https://hanzo.ai)
