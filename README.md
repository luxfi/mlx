# Lux GPU - GPU Acceleration Library

High-performance GPU acceleration for blockchain and ML workloads.

## Supported Backends

| Backend | Platform | GPU | Status |
|---------|----------|-----|--------|
| Metal | macOS | Apple Silicon, Intel | Stable |
| CUDA | Linux, Windows | NVIDIA | Stable |
| WebGPU | All | Any WebGPU-compatible | Beta |
| CPU | All | None (fallback) | Stable |

## Installation

### From Release Binaries

Download the appropriate package for your platform from [Releases](../../releases).

```bash
# Linux (CUDA)
tar -xzf libaccel-linux-x86_64-cuda.tar.gz
export LUX_GPU_BACKEND_PATH=$PWD/linux-x86_64-cuda

# macOS (Metal)
tar -xzf libaccel-macos-arm64.tar.gz
export LUX_GPU_BACKEND_PATH=$PWD/macos-arm64

# Windows (CUDA)
# Extract zip and add to PATH
```

### From Source

```bash
# Core + Metal (macOS)
cmake -B build -DLUX_GPU_BUILD_METAL=ON
cmake --build build

# Core + CUDA (Linux with NVIDIA GPU)
cmake -B build -DLUX_GPU_BUILD_CUDA=ON
cmake --build build

# All available backends
cmake -B build -DLUX_GPU_BUILD_ALL_BACKENDS=ON
cmake --build build
```

## Usage

### C API

```c
#include <lux/gpu.h>

int main() {
    // Initialize library
    if (lux_gpu_init() != LUX_GPU_SUCCESS) {
        return 1;
    }

    // Check available backends
    printf("Backends available: %d\n", lux_gpu_backend_count());

    // Create context (auto-selects best backend)
    LuxContext* ctx = lux_gpu_create_context(-1);

    // Allocate buffers
    LuxBuffer* a = lux_gpu_alloc(ctx, 1024 * sizeof(float));
    LuxBuffer* b = lux_gpu_alloc(ctx, 1024 * sizeof(float));
    LuxBuffer* c = lux_gpu_alloc(ctx, 1024 * sizeof(float));

    // Copy data to GPU
    float data[1024];
    lux_gpu_copy_to_device(ctx, a, data, sizeof(data));
    lux_gpu_copy_to_device(ctx, b, data, sizeof(data));

    // Perform operation
    lux_gpu_add_f32(ctx, a, b, c, 1024);

    // Sync and copy back
    lux_gpu_sync(ctx);
    lux_gpu_copy_to_host(ctx, c, data, sizeof(data));

    // Cleanup
    lux_gpu_free(ctx, a);
    lux_gpu_free(ctx, b);
    lux_gpu_free(ctx, c);
    lux_gpu_destroy_context(ctx);
    lux_gpu_shutdown();

    return 0;
}
```

### Go Bindings

```go
import "github.com/luxfi/node/accel"

func main() {
    if err := accel.Init(); err != nil {
        log.Fatal(err)
    }
    defer accel.Shutdown()

    // Check available backends
    for _, b := range accel.Backends() {
        fmt.Printf("Backend: %s\n", b)
    }

    // Create session with auto-detection
    session, _ := accel.NewSession()
    defer session.Close()

    // Or specify backend
    session, _ = accel.NewSessionWithBackend(accel.BackendMetal)
}
```

## Backend Selection

### Automatic Selection

By default, the library selects backends in this priority order:
1. **CUDA** - If NVIDIA GPU detected
2. **Metal** - If running on macOS with Apple GPU
3. **WebGPU** - Cross-platform fallback
4. **CPU** - Final fallback

### Manual Selection

```c
// Environment variable
export LUX_BACKEND=metal   # or cuda, webgpu, cpu

// Or via API
lux_gpu_set_backend(LUX_BACKEND_METAL);
```

### Backend Discovery

```c
// List available backends
int count = lux_gpu_backend_count();
for (int i = 0; i < count; i++) {
    LuxBackend backend = lux_gpu_get_backend(i);
    printf("Backend %d: %s\n", i, lux_gpu_backend_name(backend));
}

// Check specific capabilities
LuxCapabilities caps = lux_gpu_get_capabilities(LUX_BACKEND_METAL);
if (caps & LUX_CAP_MSM) {
    printf("MSM supported on Metal\n");
}
```

## Operations

### Tensor Operations
- Element-wise: add, sub, mul, div
- Unary: exp, log, sqrt, tanh, sigmoid, relu, gelu
- Matrix: matmul, transpose
- Reductions: sum, max, min, mean
- Normalization: layer_norm, rms_norm
- Activation: softmax, log_softmax

### Cryptographic Operations
- Curves: BLS12-381, BN254, secp256k1, Ed25519
- Hashing: Poseidon2, Blake3, SHA256, Keccak
- ZK: NTT/INTT, MSM, polynomial operations
- KZG: commit, open, verify

### FHE Operations
- TFHE: bootstrap, keyswitch
- Blind rotation
- Sample extraction
- Polynomial multiplication

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `LUX_BACKEND` | Force specific backend | `metal`, `cuda`, `webgpu`, `cpu` |
| `LUX_GPU_BACKEND_PATH` | Plugin search path | `/usr/local/lib/lux-gpu` |
| `LUX_GPU_DEVICE` | Device index | `0`, `1` |
| `LUX_GPU_DEBUG` | Enable debug logging | `1` |

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `LUX_GPU_BUILD_METAL` | OFF | Build Metal backend |
| `LUX_GPU_BUILD_CUDA` | OFF | Build CUDA backend |
| `LUX_GPU_BUILD_WEBGPU` | OFF | Build WebGPU backend |
| `LUX_GPU_BUILD_ALL_BACKENDS` | OFF | Auto-detect and build all |
| `LUX_GPU_BUILD_TESTS` | ON | Build test suite |
| `LUX_GPU_BUILD_BENCHMARKS` | OFF | Build benchmarks |
| `LUX_GPU_EMBED_KERNELS` | ON | Embed kernel source in plugins |

## License

BSD-3-Clause-Eco - See LICENSE file.
