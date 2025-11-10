// +build !cgo

// Package mlx provides stubs when CGO is disabled or MLX library is not available
package mlx

import (
	"errors"
	"fmt"
	"runtime"
)

var (
	// ErrMLXNotAvailable is returned when MLX library is not built
	ErrMLXNotAvailable = errors.New("MLX library not available - using ONNX fallback")
)

// Context provides a fallback implementation
type Context struct {
	backend Backend
	device  *Device
	version string
}

// Backend detection functions that work without CGO
func hasMetalSupport() bool {
	return false // No Metal without MLX
}

func hasCUDASupport() bool {
	return false // No CUDA without MLX
}

func getMetalDeviceName() string {
	return "N/A"
}

func getMetalMemory() int64 {
	return 0
}

func getCUDADeviceName() string {
	return "N/A"
}

func getCUDAMemory() int64 {
	return 0
}

func getSystemMemory() int64 {
	// Estimate system memory (8GB default)
	return 8 * 1024 * 1024 * 1024
}

// detectBackend falls back to ONNX on Windows when MLX not available
func (c *Context) detectBackend() {
	// On Windows, try ONNX Runtime
	if runtime.GOOS == "windows" {
		if hasONNXSupport() {
			c.backend = ONNX
			c.device = &Device{
				Type:   ONNX,
				ID:     0,
				Name:   "ONNX Runtime " + getONNXVersion(),
				Memory: getSystemMemory(),
			}
			return
		}
	}

	// Fallback to CPU (limited functionality)
	c.backend = CPU
	c.device = &Device{
		Type:   CPU,
		ID:     0,
		Name:   "CPU (no MLX)",
		Memory: getSystemMemory(),
	}
}

// SetBackend sets the backend (limited without MLX)
func (c *Context) SetBackend(backend Backend) error {
	if backend == Metal || backend == CUDA {
		return fmt.Errorf("%w: %s backend requires MLX library", ErrMLXNotAvailable, backend)
	}

	if backend == ONNX {
		if !hasONNXSupport() {
			return errors.New("ONNX Runtime not available")
		}
		c.backend = ONNX
		c.device = &Device{
			Type:   ONNX,
			ID:     0,
			Name:   "ONNX Runtime " + getONNXVersion(),
			Memory: getSystemMemory(),
		}
		return nil
	}

	c.backend = CPU
	c.device = &Device{
		Type:   CPU,
		ID:     0,
		Name:   "CPU (no MLX)",
		Memory: getSystemMemory(),
	}
	return nil
}

// GetBackend returns current backend
func (c *Context) GetBackend() Backend {
	return c.backend
}

// GetDevice returns current device
func (c *Context) GetDevice() *Device {
	return c.device
}

// Stub implementations for array operations
func (c *Context) Zeros(shape []int, dtype Dtype) *Array {
	return &Array{shape: shape, dtype: dtype}
}

func (c *Context) Ones(shape []int, dtype Dtype) *Array {
	return &Array{shape: shape, dtype: dtype}
}

func (c *Context) Random(shape []int, dtype Dtype) *Array {
	return &Array{shape: shape, dtype: dtype}
}

func (c *Context) Arange(start, stop, step float64) *Array {
	n := int((stop - start) / step)
	return &Array{shape: []int{n}, dtype: Float32}
}

func (c *Context) FromSlice(data []float32, shape []int, dtype Dtype) *Array {
	return &Array{shape: shape, dtype: dtype}
}

func (c *Context) Add(a, b *Array) *Array {
	return &Array{shape: a.shape, dtype: a.dtype}
}

func (c *Context) Maximum(a, b *Array) *Array {
	return &Array{shape: a.shape, dtype: a.dtype}
}

func (c *Context) Multiply(a, b *Array) *Array {
	return &Array{shape: a.shape, dtype: a.dtype}
}

func (c *Context) MatMul(a, b *Array) *Array {
	if len(a.shape) < 2 || len(b.shape) < 2 {
		return &Array{shape: a.shape, dtype: a.dtype}
	}
	m, n := a.shape[0], b.shape[1]
	return &Array{shape: []int{m, n}, dtype: a.dtype}
}

func (c *Context) Sum(a *Array, axis ...int) *Array {
	return &Array{shape: []int{1}, dtype: a.dtype}
}

func (c *Context) Mean(a *Array, axis ...int) *Array {
	return &Array{shape: []int{1}, dtype: a.dtype}
}

func (c *Context) Eval(arrays ...*Array) {
	// No-op without MLX
}

func (c *Context) Synchronize() {
	// No-op without MLX
}

func (c *Context) NewStream() *Stream {
	return &Stream{device: c.device}
}

// Info returns fallback mode information
func Info() string {
	backend := GetBackend()
	device := GetDevice()
	if backend == ONNX {
		return fmt.Sprintf("MLX Fallback Mode - Backend: %s, Device: %s (MLX library not available)",
			backend, device.Name)
	}
	return fmt.Sprintf("MLX Fallback Mode - Backend: %s, Device: %s (limited functionality)",
		backend, device.Name)
}
