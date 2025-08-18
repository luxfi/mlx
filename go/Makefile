# MLX Go Package Makefile

# Variables
GO := go
CGO_ENABLED ?= 1
GOFLAGS := -v
LDFLAGS := -s -w

# Platform detection
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Platform-specific settings
ifeq ($(UNAME_S),Darwin)
	# macOS with Metal
	CGO_CXXFLAGS := -std=c++17 -O3 -DMLX_METAL_JIT
	CGO_LDFLAGS := -framework Foundation -framework Metal -framework MetalPerformanceShaders
	BACKEND := Metal
else ifeq ($(UNAME_S),Linux)
	# Linux with CUDA
	CGO_CXXFLAGS := -std=c++17 -O3 -DMLX_CUDA
	CGO_LDFLAGS := -lcudart -lcublas -lcudnn
	BACKEND := CUDA
else
	# Fallback to CPU
	CGO_ENABLED := 0
	BACKEND := CPU
endif

# Targets
.PHONY: all build test bench clean install info

all: build test

build:
	@echo "🔧 Building MLX Go package (Backend: $(BACKEND))..."
	CGO_ENABLED=$(CGO_ENABLED) \
	CGO_CXXFLAGS="$(CGO_CXXFLAGS)" \
	CGO_LDFLAGS="$(CGO_LDFLAGS)" \
	$(GO) build $(GOFLAGS) -ldflags="$(LDFLAGS)" .

test:
	@echo "🧪 Running tests..."
	CGO_ENABLED=$(CGO_ENABLED) $(GO) test $(GOFLAGS) ./...

bench:
	@echo "⚡ Running benchmarks..."
	CGO_ENABLED=$(CGO_ENABLED) $(GO) test -bench=. -benchmem ./...

install:
	@echo "📦 Installing MLX Go package..."
	CGO_ENABLED=$(CGO_ENABLED) \
	CGO_CXXFLAGS="$(CGO_CXXFLAGS)" \
	CGO_LDFLAGS="$(CGO_LDFLAGS)" \
	$(GO) install $(GOFLAGS) -ldflags="$(LDFLAGS)" .

clean:
	@echo "🧹 Cleaning..."
	$(GO) clean
	rm -f *.so *.dylib *.a

info:
	@echo "ℹ️  MLX Build Information"
	@echo "Platform: $(UNAME_S) $(UNAME_M)"
	@echo "Backend: $(BACKEND)"
	@echo "CGO_ENABLED: $(CGO_ENABLED)"
	@echo "Go version: $(shell go version)"
	@if [ "$(UNAME_S)" = "Darwin" ]; then \
		echo "Metal: Available"; \
		system_profiler SPDisplaysDataType | grep "Chipset Model" || true; \
	elif [ "$(UNAME_S)" = "Linux" ]; then \
		echo "CUDA: $(shell nvidia-smi --version 2>/dev/null | head -n1 || echo 'Not available')"; \
	fi

# Development helpers
fmt:
	@echo "🎨 Formatting code..."
	$(GO) fmt ./...

vet:
	@echo "🔍 Vetting code..."
	$(GO) vet ./...

lint:
	@echo "🔍 Linting code..."
	@which golangci-lint > /dev/null || echo "golangci-lint not installed"
	@golangci-lint run ./... 2>/dev/null || true

# Documentation
doc:
	@echo "📚 Generating documentation..."
	$(GO) doc -all

# Examples
example:
	@echo "🚀 Running example..."
	@cat > /tmp/mlx_example.go << 'EOF'
	package main
	import (
		"fmt"
		"github.com/luxfi/mlx"
	)
	func main() {
		fmt.Println(mlx.Info())
		a := mlx.Zeros([]int{10, 10}, mlx.Float32)
		b := mlx.Ones([]int{10, 10}, mlx.Float32)
		c := mlx.Add(a, b)
		mlx.Eval(c)
		fmt.Println("✅ MLX operations completed successfully!")
	}
	EOF
	@CGO_ENABLED=$(CGO_ENABLED) go run /tmp/mlx_example.go