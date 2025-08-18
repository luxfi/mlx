// +build cgo

package mlx

// #cgo CXXFLAGS: -std=c++17 -O3 -march=native -fPIC
// #cgo darwin CXXFLAGS: -DMLX_METAL_JIT
// #cgo darwin LDFLAGS: -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph
// #cgo linux CXXFLAGS: -DMLX_CUDA
// #cgo linux LDFLAGS: -lcudart -lcublas -lcudnn
// #cgo LDFLAGS: -L${SRCDIR}/lib -lmlx -lc++ -lm
/*
#include <stdlib.h>
#include "mlx_c_api.h"
*/
import "C"
import (
	"runtime"
	"unsafe"
)

// hasMetalSupport checks if Metal is available (macOS Apple Silicon)
func hasMetalSupport() bool {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		return false
	}
	return bool(C.mlx_has_metal())
}

// hasCUDASupport checks if CUDA is available
func hasCUDASupport() bool {
	if runtime.GOOS == "darwin" {
		return false
	}
	return bool(C.mlx_has_cuda())
}

// getMetalDeviceName returns the Metal device name
func getMetalDeviceName() string {
	cstr := C.mlx_get_metal_device_name()
	defer C.free(unsafe.Pointer(cstr))
	return C.GoString(cstr)
}

// getCUDADeviceName returns the CUDA device name
func getCUDADeviceName() string {
	cstr := C.mlx_get_cuda_device_name()
	defer C.free(unsafe.Pointer(cstr))
	return C.GoString(cstr)
}

// getMetalMemory returns available Metal GPU memory
func getMetalMemory() int64 {
	return int64(C.mlx_get_metal_memory())
}

// getCUDAMemory returns available CUDA GPU memory
func getCUDAMemory() int64 {
	return int64(C.mlx_get_cuda_memory())
}

// getSystemMemory returns available system memory
func getSystemMemory() int64 {
	return int64(C.mlx_get_system_memory())
}

// Implementation of array operations using C++ MLX

func (c *Context) Zeros(shape []int, dtype Dtype) *Array {
	cShape := (*C.int)(unsafe.Pointer(&shape[0]))
	handle := C.mlx_zeros(cShape, C.int(len(shape)), C.int(dtype))
	
	arr := &Array{
		handle: handle,
		shape:  shape,
		dtype:  dtype,
	}
	
	c.mu.Lock()
	c.arrays[handle] = arr
	c.mu.Unlock()
	
	runtime.SetFinalizer(arr, (*Array).free)
	return arr
}

func (c *Context) Ones(shape []int, dtype Dtype) *Array {
	cShape := (*C.int)(unsafe.Pointer(&shape[0]))
	handle := C.mlx_ones(cShape, C.int(len(shape)), C.int(dtype))
	
	arr := &Array{
		handle: handle,
		shape:  shape,
		dtype:  dtype,
	}
	
	c.mu.Lock()
	c.arrays[handle] = arr
	c.mu.Unlock()
	
	runtime.SetFinalizer(arr, (*Array).free)
	return arr
}

func (c *Context) Random(shape []int, dtype Dtype) *Array {
	cShape := (*C.int)(unsafe.Pointer(&shape[0]))
	handle := C.mlx_random(cShape, C.int(len(shape)), C.int(dtype))
	
	arr := &Array{
		handle: handle,
		shape:  shape,
		dtype:  dtype,
	}
	
	c.mu.Lock()
	c.arrays[handle] = arr
	c.mu.Unlock()
	
	runtime.SetFinalizer(arr, (*Array).free)
	return arr
}

func (c *Context) Arange(start, stop, step float64) *Array {
	handle := C.mlx_arange(C.double(start), C.double(stop), C.double(step))
	
	// Calculate shape
	size := int((stop - start) / step)
	shape := []int{size}
	
	arr := &Array{
		handle: handle,
		shape:  shape,
		dtype:  Float64,
	}
	
	c.mu.Lock()
	c.arrays[handle] = arr
	c.mu.Unlock()
	
	runtime.SetFinalizer(arr, (*Array).free)
	return arr
}

func (c *Context) Add(a, b *Array) *Array {
	handle := C.mlx_add(a.handle, b.handle)
	
	arr := &Array{
		handle: handle,
		shape:  a.shape,
		dtype:  a.dtype,
	}
	
	c.mu.Lock()
	c.arrays[handle] = arr
	c.mu.Unlock()
	
	runtime.SetFinalizer(arr, (*Array).free)
	return arr
}

func (c *Context) Multiply(a, b *Array) *Array {
	handle := C.mlx_multiply(a.handle, b.handle)
	
	arr := &Array{
		handle: handle,
		shape:  a.shape,
		dtype:  a.dtype,
	}
	
	c.mu.Lock()
	c.arrays[handle] = arr
	c.mu.Unlock()
	
	runtime.SetFinalizer(arr, (*Array).free)
	return arr
}

func (c *Context) MatMul(a, b *Array) *Array {
	handle := C.mlx_matmul(a.handle, b.handle)
	
	// Calculate output shape
	shape := make([]int, len(a.shape))
	copy(shape, a.shape)
	shape[len(shape)-1] = b.shape[len(b.shape)-1]
	
	arr := &Array{
		handle: handle,
		shape:  shape,
		dtype:  a.dtype,
	}
	
	c.mu.Lock()
	c.arrays[handle] = arr
	c.mu.Unlock()
	
	runtime.SetFinalizer(arr, (*Array).free)
	return arr
}

func (c *Context) Sum(a *Array, axis ...int) *Array {
	var cAxis *C.int
	var nAxis C.int
	
	if len(axis) > 0 {
		cAxis = (*C.int)(unsafe.Pointer(&axis[0]))
		nAxis = C.int(len(axis))
	}
	
	handle := C.mlx_sum(a.handle, cAxis, nAxis)
	
	// Calculate output shape
	shape := calculateReducedShape(a.shape, axis)
	
	arr := &Array{
		handle: handle,
		shape:  shape,
		dtype:  a.dtype,
	}
	
	c.mu.Lock()
	c.arrays[handle] = arr
	c.mu.Unlock()
	
	runtime.SetFinalizer(arr, (*Array).free)
	return arr
}

func (c *Context) Mean(a *Array, axis ...int) *Array {
	var cAxis *C.int
	var nAxis C.int
	
	if len(axis) > 0 {
		cAxis = (*C.int)(unsafe.Pointer(&axis[0]))
		nAxis = C.int(len(axis))
	}
	
	handle := C.mlx_mean(a.handle, cAxis, nAxis)
	
	// Calculate output shape
	shape := calculateReducedShape(a.shape, axis)
	
	arr := &Array{
		handle: handle,
		shape:  shape,
		dtype:  a.dtype,
	}
	
	c.mu.Lock()
	c.arrays[handle] = arr
	c.mu.Unlock()
	
	runtime.SetFinalizer(arr, (*Array).free)
	return arr
}

func (c *Context) Eval(arrays ...*Array) {
	// For now, evaluate arrays one by one to avoid CGO complexity
	// In a real implementation, this would batch evaluate
	for _, arr := range arrays {
		if arr != nil && arr.handle != nil {
			// Force evaluation of this array
			// This is a stub - real MLX would evaluate the computation graph
		}
	}
}

func (c *Context) Synchronize() {
	C.mlx_synchronize()
}

func (c *Context) NewStream() *Stream {
	handle := C.mlx_new_stream()
	
	stream := &Stream{
		handle: handle,
		device: c.device,
	}
	
	c.mu.Lock()
	c.streams[handle] = stream
	c.mu.Unlock()
	
	runtime.SetFinalizer(stream, (*Stream).free)
	return stream
}

// Array methods

func (a *Array) free() {
	if a.handle != nil {
		C.mlx_free_array(a.handle)
		a.handle = nil
	}
}

// Stream methods

func (s *Stream) free() {
	if s.handle != nil {
		C.mlx_free_stream(s.handle)
		s.handle = nil
	}
}

// Helper functions

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