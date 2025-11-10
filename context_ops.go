// +build cgo

package mlx

/*
#include "mlx_c_api.h"
#include <stdlib.h>
*/
import "C"
import (
	"unsafe"
)

// Zeros creates a zero-filled array
func (c *Context) Zeros(shape []int, dtype Dtype) *Array {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	cShape := (*C.int)(unsafe.Pointer(&shape[0]))
	handle := C.mlx_zeros(cShape, C.int(len(shape)), C.int(dtype))
	
	arr := &Array{
		handle: handle,
		shape:  shape,
		dtype:  dtype,
	}
	c.arrays[handle] = arr
	return arr
}

// Ones creates an array filled with ones
func (c *Context) Ones(shape []int, dtype Dtype) *Array {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	cShape := (*C.int)(unsafe.Pointer(&shape[0]))
	handle := C.mlx_ones(cShape, C.int(len(shape)), C.int(dtype))
	
	arr := &Array{
		handle: handle,
		shape:  shape,
		dtype:  dtype,
	}
	c.arrays[handle] = arr
	return arr
}

// Random creates an array with random values
func (c *Context) Random(shape []int, dtype Dtype) *Array {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	cShape := (*C.int)(unsafe.Pointer(&shape[0]))
	handle := C.mlx_random(cShape, C.int(len(shape)), C.int(dtype))
	
	arr := &Array{
		handle: handle,
		shape:  shape,
		dtype:  dtype,
	}
	c.arrays[handle] = arr
	return arr
}

// Arange creates an array with sequential values
func (c *Context) Arange(start, stop, step float64) *Array {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	handle := C.mlx_arange(C.double(start), C.double(stop), C.double(step))
	
	// Calculate shape
	size := int((stop - start) / step)
	arr := &Array{
		handle: handle,
		shape:  []int{size},
		dtype:  Float32,
	}
	c.arrays[handle] = arr
	return arr
}

// Add performs element-wise addition
func (c *Context) Add(a, b *Array) *Array {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	handle := C.mlx_add(a.handle, b.handle)
	
	arr := &Array{
		handle: handle,
		shape:  a.shape, // Assuming same shape
		dtype:  a.dtype,
	}
	c.arrays[handle] = arr
	return arr
}

// Multiply performs element-wise multiplication
func (c *Context) Multiply(a, b *Array) *Array {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	handle := C.mlx_multiply(a.handle, b.handle)
	
	arr := &Array{
		handle: handle,
		shape:  a.shape,
		dtype:  a.dtype,
	}
	c.arrays[handle] = arr
	return arr
}

// MatMul performs matrix multiplication
func (c *Context) MatMul(a, b *Array) *Array {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	handle := C.mlx_matmul(a.handle, b.handle)
	
	// Calculate output shape
	// Assuming 2D matrices: (m,k) x (k,n) = (m,n)
	m := a.shape[0]
	n := b.shape[1]
	
	arr := &Array{
		handle: handle,
		shape:  []int{m, n},
		dtype:  a.dtype,
	}
	c.arrays[handle] = arr
	return arr
}

// Sum computes the sum of array elements
func (c *Context) Sum(a *Array, axis ...int) *Array {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	var handle unsafe.Pointer
	if len(axis) == 0 {
		handle = C.mlx_sum(a.handle, nil, 0)
	} else {
		cAxis := (*C.int)(unsafe.Pointer(&axis[0]))
		handle = C.mlx_sum(a.handle, cAxis, C.int(len(axis)))
	}
	
	// Calculate output shape
	// This is simplified - actual shape depends on axis
	arr := &Array{
		handle: handle,
		shape:  []int{1},
		dtype:  a.dtype,
	}
	c.arrays[handle] = arr
	return arr
}

// Mean computes the mean of array elements
func (c *Context) Mean(a *Array, axis ...int) *Array {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	var handle unsafe.Pointer
	if len(axis) == 0 {
		handle = C.mlx_mean(a.handle, nil, 0)
	} else {
		cAxis := (*C.int)(unsafe.Pointer(&axis[0]))
		handle = C.mlx_mean(a.handle, cAxis, C.int(len(axis)))
	}
	
	arr := &Array{
		handle: handle,
		shape:  []int{1},
		dtype:  a.dtype,
	}
	c.arrays[handle] = arr
	return arr
}

// Eval forces evaluation of lazy operations
func (c *Context) Eval(arrays ...*Array) {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	if len(arrays) == 0 {
		return
	}
	
	// Convert to C array of pointers
	handles := make([]unsafe.Pointer, len(arrays))
	for i, arr := range arrays {
		handles[i] = arr.handle
	}
	
	C.mlx_eval(&handles[0], C.int(len(arrays)))
}

// Synchronize waits for all operations to complete
func (c *Context) Synchronize() {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	C.mlx_synchronize()
}

// NewStream creates a new compute stream
func (c *Context) NewStream() *Stream {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	handle := C.mlx_new_stream()
	stream := &Stream{
		handle: handle,
		device: c.device,
	}
	c.streams[handle] = stream
	return stream
}

// Free releases resources for an array
func (c *Context) Free(a *Array) {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	if a.handle != nil {
		C.mlx_free_array(a.handle)
		delete(c.arrays, a.handle)
		a.handle = nil
	}
}

// FreeStream releases resources for a stream
func (c *Context) FreeStream(s *Stream) {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	if s.handle != nil {
		C.mlx_free_stream(s.handle)
		delete(c.streams, s.handle)
		s.handle = nil
	}
}
// FromSlice creates an array from a Go slice
func (c *Context) FromSlice(data []float32, shape []int, dtype Dtype) *Array {
	c.mu.Lock()
	defer c.mu.Unlock()

	cData := (*C.float)(unsafe.Pointer(&data[0]))
	cShape := (*C.int)(unsafe.Pointer(&shape[0]))
	handle := C.mlx_from_slice(cData, C.int(len(data)), cShape, C.int(len(shape)), C.int(dtype))

	arr := &Array{
		handle: handle,
		shape:  shape,
		dtype:  dtype,
	}
	c.arrays[handle] = arr
	return arr
}

// Maximum computes element-wise maximum of two arrays
func (c *Context) Maximum(a, b *Array) *Array {
	c.mu.Lock()
	defer c.mu.Unlock()

	handle := C.mlx_maximum(a.handle, b.handle)

	arr := &Array{
		handle: handle,
		shape:  a.shape,
		dtype:  a.dtype,
	}
	c.arrays[handle] = arr
	return arr
}
