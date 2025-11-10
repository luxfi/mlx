# AI Assistant Knowledge Base

**Last Updated**: 2025-11-10
**Project**: MLX Go Bindings
**Organization**: LuxFi

## Project Overview

This repository provides **official Go bindings for Apple's MLX** (Machine Learning framework). 
It features multi-backend support (Metal, CUDA, CPU) with automatic hardware detection.

### Status
- ✅ **Real MLX C++ integration complete** (Nov 10, 2025)
- ✅ All tests passing with GPU acceleration
- ✅ 84MB libmlx.a library built and linked
- 📦 Ready for upstream merge

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

2. **mlx.go** - Go API with CGO bindings
   - Auto-detects best backend (Metal > CUDA > CPU)
   - Manages array lifecycle
   - Thread-safe context management

3. **lib/libmlx.a** - Real MLX C++ library (84MB)
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
- **MLX C++17** library
- **Metal** API (macOS GPU)
- **CMake** build system
- **Apple Accelerate** framework

## Context for All AI Assistants

This file (`LLM.md`) is symlinked as:
- `.AGENTS.md`
- `CLAUDE.md`
- `QWEN.md`
- `GEMINI.md`

All files reference the same knowledge base. Updates here propagate to all AI systems.

## Rules for AI Assistants

1. **ALWAYS** update LLM.md with significant discoveries
2. **NEVER** commit symlinked files (.AGENTS.md, CLAUDE.md, etc.) - they're in .gitignore
3. **NEVER** create random summary files - update THIS file

---

**Note**: This file serves as the single source of truth for all AI assistants working on this project.
