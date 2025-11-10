# AI Assistant Knowledge Base

**Last Updated**: 2025-11-10
**Project**: MLX Go Bindings
**Organization**: LuxFi

## Project Overview

This repository provides **official Go bindings for Apple's MLX** (Machine Learning framework). 
It features multi-backend support (Metal, CUDA, CPU) with automatic hardware detection.

### Status
- ✅ **Real MLX C++ integration complete** (Nov 10, 2025)
- ✅ **ONNX Runtime fallback added** (Nov 10, 2025)
- ✅ **Pre-compiled binaries v0.29.4** released
- ✅ All tests passing with GPU acceleration
- ✅ Windows support via ONNX Runtime
- 📦 Production ready across all platforms

## Essential Commands

### Build & Test
```bash
# Build MLX C++ library
mkdir -p build && cd build
cmake .. -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_PYTHON_BINDINGS=OFF
make -j$(sysctl -n hw.ncpu)
cp libmlx.a ../lib/

# Build Go bindings
CGO_ENABLED=1 go build -v .

# Run tests
CGO_ENABLED=1 go test -v .
```

### Development
```bash
# Add to workspace
go work use .

# Test specific backend
MLX_BACKEND=metal go test -v .
MLX_BACKEND=cpu go test -v .
```

## Architecture

### Components
1. **mlx_c_api.cpp/h** - C wrapper for MLX C++ library
   - Wraps mlx::core::array, mlx::core::zeros, etc.
   - Proper type conversions (Shape, Dtype, Device)
   - Exception handling → nullptr returns

2. **mlx.go** - Go API with multi-backend support
   - Auto-detects best backend (Metal > CUDA > ONNX > CPU)
   - Manages array lifecycle
   - Thread-safe context management
   - ONNX fallback on Windows

3. **mlx_onnx.go** - ONNX Runtime integration
   - Windows-first backend
   - Production-ready inference
   - Automatic detection

4. **mlx_fallback.go** - No-CGO stubs
   - Graceful degradation
   - ONNX-only mode support

5. **lib/libmlx.a** - MLX C++ library (84MB, optional on Windows)
   - Built from upstream ml-explore/mlx
   - Metal backend for Apple Silicon
   - CPU backend as fallback

### Key Integration Points
- **CGO Flags**: Links against lib/libmlx.a with Metal frameworks
- **Include Path**: Uses `${SRCDIR}/mlx` for headers
- **Type Mapping**: 
  - `int[]` → `mlx::core::Shape`
  - `int dtype` → `mlx::core::Dtype`
  - `void*` ↔ `mlx::core::array*`

## Key Technologies

- **Go 1.21+** with CGO
- **MLX C++17** library (Metal/CUDA/CPU)
- **ONNX Runtime 1.17** (Windows fallback)
- **Metal** API (macOS GPU)
- **CMake** build system
- **Apple Accelerate** framework

## Backend Strategy

### Automatic Selection Priority
1. **Metal** (macOS ARM64) - Best performance, GPU acceleration
2. **CUDA** (Linux/Windows + NVIDIA) - Excellent GPU performance  
3. **ONNX** (Windows) - Production-ready CPU/GPU inference
4. **CPU** (All platforms) - Fallback, limited without MLX

### Platform Recommendations
- **macOS ARM64**: Use Metal (pre-compiled binary available)
- **Linux x64**: Use CPU/CUDA (pre-compiled binary available)
- **Windows**: Use ONNX Runtime (MLX has MSVC issues)
- **macOS x64**: Build from source or use ONNX

### Pre-compiled Binaries
- **v0.29.4 Release**: https://github.com/luxfi/mlx/releases/tag/v0.29.4
- **macOS ARM64**: 84MB Metal GPU binary
- **Linux x64**: 84MB CPU binary with OpenBLAS
- **CI Optimization**: 2-4 min with pre-built (vs 15 min from source)

## Context for All AI Assistants

This file (`LLM.md`) is symlinked as:
- `.AGENTS.md`
- `CLAUDE.md`
- `QWEN.md`
- `GEMINI.md`

All files reference the same knowledge base. Updates here propagate to all AI systems.

## Recent Achievements (Nov 10, 2025)

### 1. ONNX Runtime Integration
- Solved Windows MSVC compilation issues
- Added ONNX as new backend type
- Created `mlx_onnx.go` with C bindings
- Automatic detection on Windows
- Full API compatibility
- Training support documented (ORTModule, on-device training, transfer learning)

### 2. Pre-compiled Binary System
- Release workflow: `build-release.yml`
- Builds on tag push (v*)
- 4 platforms in parallel
- Creates GitHub releases automatically
- Test workflows download binaries first

### 3. Documentation
- `ONNX.md`: Complete Windows setup guide with training support
  - Large model training with ORTModule
  - On-device training for edge devices
  - Transfer learning examples
- `BINARIES.md`: Binary installation guide
- `README_GO.md`: Updated with ONNX instructions
- `PLATFORM_SUPPORT.md`: Platform compatibility matrix

### 4. Fixed Issues
- Windows MSVC `-Wno-psabi` flag incompatibility
- macOS x64 GitHub Actions quota (optional now)
- Windows `__builtin_clz`/`typeof` intrinsic errors
- CI speed: 15min → 2-4min with pre-built binaries

## Rules for AI Assistants

1. **ALWAYS** update LLM.md with significant discoveries
2. **NEVER** commit symlinked files (.AGENTS.md, CLAUDE.md, etc.) - they're in .gitignore
3. **NEVER** create random summary files - update THIS file
4. **Windows users**: Recommend ONNX Runtime over MLX
5. **Binary releases**: Only need 2/4 platforms (macOS ARM64 + Linux)

---

**Note**: This file serves as the single source of truth for all AI assistants working on this project.
