# ONNX Runtime Integration

MLX Go bindings automatically fall back to ONNX Runtime on Windows when the MLX C++ library is not available.

## Why ONNX Runtime?

- **Windows Compatibility**: MLX uses GCC/Clang intrinsics that don't compile with MSVC
- **Production Ready**: ONNX Runtime is widely used and battle-tested
- **Performance**: Optimized for CPU and GPU inference on Windows
- **Ecosystem**: Works with models from PyTorch, TensorFlow, scikit-learn, and more

## Installation

### Option 1: Pre-built ONNX Runtime (Recommended)

1. Download ONNX Runtime for Windows:
```bash
# Download latest release
curl -LO https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-win-x64-1.17.0.zip

# Extract
unzip onnxruntime-win-x64-1.17.0.zip

# Copy to mlx directory
mkdir -p onnxruntime
cp -r onnxruntime-win-x64-1.17.0/include onnxruntime/
cp -r onnxruntime-win-x64-1.17.0/lib onnxruntime/
```

2. Build Go code with ONNX support:
```bash
set CGO_ENABLED=1
set MLX_BACKEND=onnx
go build
```

### Option 2: Build ONNX Runtime from Source

```bash
# Clone ONNX Runtime
git clone --recursive https://github.com/microsoft/onnxruntime.git
cd onnxruntime

# Build with CMake
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

## Usage

### Automatic Backend Detection

```go
package main

import (
    "fmt"
    "github.com/luxfi/mlx"
)

func main() {
    // Automatically detects ONNX Runtime on Windows
    fmt.Println(mlx.Info())
    // Output: MLX Fallback Mode - Backend: ONNX, Device: ONNX Runtime 1.17.0
    
    // Use MLX API normally
    ctx := mlx.DefaultContext
    backend := ctx.GetBackend()
    fmt.Printf("Using backend: %s\n", backend)
}
```

### Explicit ONNX Backend

```go
import "github.com/luxfi/mlx"

// Force ONNX Runtime backend
err := mlx.SetBackend(mlx.ONNX)
if err != nil {
    log.Fatal(err)
}
```

### Environment Variable

```bash
# Windows
set MLX_BACKEND=onnx
go run main.go

# PowerShell
$env:MLX_BACKEND = "onnx"
go run main.go
```

## API Compatibility

ONNX backend implements the same MLX API:

```go
// Create arrays
a := mlx.Zeros([]int{3, 3}, mlx.Float32)
b := mlx.Ones([]int{3, 3}, mlx.Float32)

// Operations
c := mlx.Add(a, b)
d := mlx.MatMul(a, b)

// Evaluation
mlx.Eval(c, d)
```

## Backend Priority

MLX automatically selects backends in this order:

1. **Metal** - macOS ARM64 (Apple Silicon)
2. **CUDA** - Linux/Windows with NVIDIA GPU
3. **ONNX** - Windows without MLX (recommended fallback)
4. **CPU** - All platforms (limited functionality without MLX)

## Performance Comparison

| Backend | Platform | Use Case | Performance |
|---------|----------|----------|-------------|
| Metal | macOS ARM64 | Training/Inference | Excellent |
| CUDA | Linux/Windows | Training/Inference | Excellent |
| ONNX | Windows | Inference Only | Very Good |
| MLX CPU | All | Training/Inference | Good |
| CPU Fallback | All | Testing Only | Limited |

## ONNX Runtime Features

✅ Supported:
- Inference with pre-trained models
- CPU execution
- GPU execution (with CUDA/DirectML)
- Model optimization
- Quantization

❌ Not supported via MLX API:
- Training (inference only)
- Custom operators
- Model export

## Converting Models to ONNX

### From PyTorch

```python
import torch
import torch.onnx

model = YourModel()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=17,
    input_names=['input'],
    output_names=['output']
)
```

### From TensorFlow

```python
import tf2onnx
import tensorflow as tf

model = tf.keras.models.load_model('model.h5')
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=17,
    output_path="model.onnx"
)
```

## Troubleshooting

### ONNX Runtime not found

```
Error: ONNX Runtime not available
```

**Solution**: Download pre-built binaries or build from source, then copy to `onnxruntime/` directory.

### Linking errors

```
error: ld returned 1 exit status
```

**Solution**: Ensure `onnxruntime/lib` contains `onnxruntime.dll` (Windows) or `libonnxruntime.so` (Linux).

### CPU-only mode

If ONNX Runtime doesn't detect GPU:

```bash
# Check CUDA availability
nvidia-smi

# Install CUDA version of ONNX Runtime
# Download onnxruntime-win-x64-gpu-1.17.0.zip instead
```

## CI/CD Integration

The test workflow automatically handles ONNX fallback:

```yaml
- name: Download pre-built MLX library
  continue-on-error: true  # Try MLX first
  run: |
    gh release download latest -p "libmlx-windows-x64.tar.gz"

- name: Install ONNX Runtime (fallback)
  if: hashFiles('lib/libmlx.a') == ''
  run: |
    curl -LO https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-win-x64-1.17.0.zip
    unzip onnxruntime-win-x64-1.17.0.zip -d onnxruntime

- name: Run tests
  env:
    MLX_BACKEND: auto  # Auto-detects ONNX on Windows
  run: go test -v ./...
```

## Resources

- [ONNX Runtime GitHub](https://github.com/microsoft/onnxruntime)
- [ONNX Runtime Docs](https://onnxruntime.ai/)
- [ONNX Model Zoo](https://github.com/onnx/models)
- [MLX Documentation](https://ml-explore.github.io/mlx/)

## Support

For ONNX-specific issues:
- ONNX Runtime: https://github.com/microsoft/onnxruntime/issues
- MLX Go Bindings: https://github.com/luxfi/mlx/issues
