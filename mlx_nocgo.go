// +build !cgo

package mlx

import (
	"unsafe"
)

// Stub implementations when CGO is disabled

func hasMetalSupport() bool {
	return false
}

func hasCUDASupport() bool {
	return false
}

func getMetalDeviceName() string {
	return "No Metal (CGO disabled)"
}

func getCUDADeviceName() string {
	return "No CUDA (CGO disabled)"
}

func getMetalMemory() int64 {
	return 0
}

func getCUDAMemory() int64 {
	return 0
}

func getSystemMemory() int64 {
	// Basic system memory estimate
	return 8 * 1024 * 1024 * 1024 // 8GB default
}

// CPU-only implementations

func (c *Context) Zeros(shape []int, dtype Dtype) *Array {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	
	return &Array{
		handle: unsafe.Pointer(&struct{}{}),
		shape:  shape,
		dtype:  dtype,
	}
}

func (c *Context) Ones(shape []int, dtype Dtype) *Array {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	
	return &Array{
		handle: unsafe.Pointer(&struct{}{}),
		shape:  shape,
		dtype:  dtype,
	}
}

func (c *Context) Random(shape []int, dtype Dtype) *Array {
	return &Array{
		handle: unsafe.Pointer(&struct{}{}),
		shape:  shape,
		dtype:  dtype,
	}
}

func (c *Context) Arange(start, stop, step float64) *Array {
	size := int((stop - start) / step)
	return &Array{
		handle: unsafe.Pointer(&struct{}{}),
		shape:  []int{size},
		dtype:  Float64,
	}
}

func (c *Context) Add(a, b *Array) *Array {
	return &Array{
		handle: unsafe.Pointer(&struct{}{}),
		shape:  a.shape,
		dtype:  a.dtype,
	}
}

func (c *Context) Multiply(a, b *Array) *Array {
	return &Array{
		handle: unsafe.Pointer(&struct{}{}),
		shape:  a.shape,
		dtype:  a.dtype,
	}
}

func (c *Context) MatMul(a, b *Array) *Array {
	shape := make([]int, len(a.shape))
	copy(shape, a.shape)
	if len(b.shape) > 0 {
		shape[len(shape)-1] = b.shape[len(b.shape)-1]
	}
	
	return &Array{
		handle: unsafe.Pointer(&struct{}{}),
		shape:  shape,
		dtype:  a.dtype,
	}
}

func (c *Context) Sum(a *Array, axis ...int) *Array {
	shape := calculateReducedShape(a.shape, axis)
	return &Array{
		handle: unsafe.Pointer(&struct{}{}),
		shape:  shape,
		dtype:  a.dtype,
	}
}

func (c *Context) Mean(a *Array, axis ...int) *Array {
	shape := calculateReducedShape(a.shape, axis)
	return &Array{
		handle: unsafe.Pointer(&struct{}{}),
		shape:  shape,
		dtype:  a.dtype,
	}
}

func (c *Context) Eval(arrays ...*Array) {
	// No-op in CPU mode
}

func (c *Context) Synchronize() {
	// No-op in CPU mode
}

func (c *Context) NewStream() *Stream {
	return &Stream{
		handle: unsafe.Pointer(&struct{}{}),
		device: c.device,
	}
}

func (a *Array) free() {
	// No-op for stub
}

func (s *Stream) free() {
	// No-op for stub
}

func calculateReducedShape(shape []int, axis []int) []int {
	if len(axis) == 0 {
		return []int{1}
	}
	
	result := make([]int, 0, len(shape))
	for i, dim := range shape {
		keep := true
		for _, ax := range axis {
			if i == ax {
				keep = false
				break
			}
		}
		if keep {
			result = append(result, dim)
		}
	}
	
	if len(result) == 0 {
		return []int{1}
	}
	return result
}