// +build cgo

// Package mlx provides Go bindings for the MLX machine learning framework.
// MLX is an array framework for machine learning on Apple silicon and beyond.
//
// IMPORTANT: This package REQUIRES CGO to function. MLX is a C++ library
// and cannot work without CGO. Build with CGO_ENABLED=1.
//
// This package wraps the C++ MLX library to provide GPU acceleration for
// high-performance computing tasks like matrix operations and neural network inference.
package mlx

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// Backend represents the compute backend used by MLX
type Backend int

const (
	// CPU uses CPU-only computation
	CPU Backend = iota
	// Metal uses Apple Metal GPU acceleration
	Metal
	// CUDA uses NVIDIA CUDA GPU acceleration
	CUDA
	// Auto automatically selects the best available backend
	Auto
)

// String returns the string representation of the backend
func (b Backend) String() string {
	switch b {
	case CPU:
		return "CPU"
	case Metal:
		return "Metal"
	case CUDA:
		return "CUDA"
	case Auto:
		return "Auto"
	default:
		return "Unknown"
	}
}

// Device represents a compute device (CPU or GPU)
type Device struct {
	Type   Backend
	ID     int
	Name   string
	Memory int64 // Memory in bytes
}

// Array represents a multi-dimensional array
type Array struct {
	handle unsafe.Pointer
	shape  []int
	dtype  Dtype
}

// Dtype represents the data type of array elements
type Dtype int

const (
	Float32 Dtype = iota
	Float64
	Int32
	Int64
	Bool
)

// Stream represents a compute stream for async operations
type Stream struct {
	handle unsafe.Pointer
	device *Device
}

// Context manages MLX runtime and resources
type Context struct {
	mu      sync.RWMutex
	backend Backend
	device  *Device
	stream  *Stream
	
	// Version info
	version string
	
	// Resource management
	arrays  map[unsafe.Pointer]*Array
	streams map[unsafe.Pointer]*Stream
}

var (
	// DefaultContext is the global MLX context
	DefaultContext *Context
	
	// ErrNoGPU is returned when GPU is requested but not available
	ErrNoGPU = errors.New("no GPU available")
	
	// ErrInvalidBackend is returned for invalid backend selection
	ErrInvalidBackend = errors.New("invalid backend")
	
	// Version is the MLX library version
	Version = "0.1.0"
)

func init() {
	// Initialize default context on package load
	DefaultContext = &Context{
		backend: Auto,
		arrays:  make(map[unsafe.Pointer]*Array),
		streams: make(map[unsafe.Pointer]*Stream),
		version: Version,
	}
	
	// Auto-detect best backend
	DefaultContext.detectBackend()
}

// detectBackend automatically selects the best available backend
func (c *Context) detectBackend() {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	// Check for Metal on macOS
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		if hasMetalSupport() {
			c.backend = Metal
			c.device = &Device{
				Type:   Metal,
				ID:     0,
				Name:   getMetalDeviceName(),
				Memory: getMetalMemory(),
			}
			return
		}
	}
	
	// Check for CUDA on Linux/Windows
	if runtime.GOOS == "linux" || runtime.GOOS == "windows" {
		if hasCUDASupport() {
			c.backend = CUDA
			c.device = &Device{
				Type:   CUDA,
				ID:     0,
				Name:   getCUDADeviceName(),
				Memory: getCUDAMemory(),
			}
			return
		}
	}
	
	// Fall back to CPU
	c.backend = CPU
	c.device = &Device{
		Type:   CPU,
		ID:     0,
		Name:   "CPU",
		Memory: getSystemMemory(),
	}
}

// SetBackend sets the compute backend
func SetBackend(backend Backend) error {
	return DefaultContext.SetBackend(backend)
}

// SetBackend sets the compute backend for this context
func (c *Context) SetBackend(backend Backend) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	switch backend {
	case Metal:
		if !hasMetalSupport() {
			return ErrNoGPU
		}
	case CUDA:
		if !hasCUDASupport() {
			return ErrNoGPU
		}
	case CPU, Auto:
		// Always available
	default:
		return ErrInvalidBackend
	}
	
	c.backend = backend
	if backend == Auto {
		c.detectBackend()
	}
	
	return nil
}

// GetBackend returns the current compute backend
func GetBackend() Backend {
	return DefaultContext.GetBackend()
}

// GetBackend returns the current backend for this context
func (c *Context) GetBackend() Backend {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.backend
}

// GetDevice returns the current compute device
func GetDevice() *Device {
	return DefaultContext.GetDevice()
}

// GetDevice returns the current device for this context
func (c *Context) GetDevice() *Device {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.device
}

// Zeros creates a zero-filled array
func Zeros(shape []int, dtype Dtype) *Array {
	return DefaultContext.Zeros(shape, dtype)
}

// Ones creates an array filled with ones
func Ones(shape []int, dtype Dtype) *Array {
	return DefaultContext.Ones(shape, dtype)
}

// Random creates an array with random values
func Random(shape []int, dtype Dtype) *Array {
	return DefaultContext.Random(shape, dtype)
}

// Arange creates an array with sequential values
func Arange(start, stop, step float64) *Array {
	return DefaultContext.Arange(start, stop, step)
}

// Add performs element-wise addition
func Add(a, b *Array) *Array {
	return DefaultContext.Add(a, b)
}

// Multiply performs element-wise multiplication
func Multiply(a, b *Array) *Array {
	return DefaultContext.Multiply(a, b)
}

// MatMul performs matrix multiplication
func MatMul(a, b *Array) *Array {
	return DefaultContext.MatMul(a, b)
}

// Sum computes the sum of array elements
func Sum(a *Array, axis ...int) *Array {
	return DefaultContext.Sum(a, axis...)
}

// Mean computes the mean of array elements
func Mean(a *Array, axis ...int) *Array {
	return DefaultContext.Mean(a, axis...)
}

// Eval forces evaluation of lazy operations
func Eval(arrays ...*Array) {
	DefaultContext.Eval(arrays...)
}

// Synchronize waits for all operations to complete
func Synchronize() {
	DefaultContext.Synchronize()
}

// NewStream creates a new compute stream
func NewStream() *Stream {
	return DefaultContext.NewStream()
}

// Info returns information about the MLX installation
func Info() string {
	device := GetDevice()
	return fmt.Sprintf("MLX %s - Backend: %s, Device: %s, Memory: %.2f GB",
		Version,
		GetBackend(),
		device.Name,
		float64(device.Memory)/(1024*1024*1024))
}