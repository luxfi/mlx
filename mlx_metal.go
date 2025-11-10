// +build darwin,cgo

package mlx

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework CoreGraphics -framework Foundation
#include "metal/mtl_bridge.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// Metal kernel source for matrix multiplication
const matmulKernel = `
#include <metal_stdlib>
using namespace metal;

kernel void matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (uint i = 0; i < K; ++i) {
        sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
}
`

// Metal kernel for element-wise addition
const addKernel = `
#include <metal_stdlib>
using namespace metal;

kernel void add_arrays(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    C[index] = A[index] + B[index];
}
`

// MetalDevice wraps Metal device functionality
type MetalDevice struct {
	device unsafe.Pointer
	queue  unsafe.Pointer
	
	// Compiled kernels
	matmulPipeline unsafe.Pointer
	addPipeline    unsafe.Pointer
}

// InitMetal initializes Metal device and compiles kernels
func InitMetal() (*MetalDevice, error) {
	// Create system default device
	devicePtr := C.mtlCreateSystemDefaultDevice()
	if devicePtr == nil {
		return nil, fmt.Errorf("failed to create Metal device")
	}
	
	// Create command queue
	queuePtr := C.mtlNewCommandQueue(devicePtr)
	if queuePtr == nil {
		return nil, fmt.Errorf("failed to create command queue")
	}
	
	md := &MetalDevice{
		device: devicePtr,
		queue:  queuePtr,
	}
	
	// Compile kernels
	if err := md.compileKernels(); err != nil {
		return nil, err
	}
	
	return md, nil
}

func (md *MetalDevice) compileKernels() error {
	// Compile matmul kernel
	matmulSrc := C.CString(matmulKernel)
	defer C.free(unsafe.Pointer(matmulSrc))
	
	matmulLib := C.mtlNewLibraryWithSource(md.device, matmulSrc)
	if matmulLib == nil {
		return fmt.Errorf("failed to compile matmul kernel")
	}
	
	matmulFunc := C.mtlNewFunctionWithName(matmulLib, C.CString("matmul"))
	if matmulFunc == nil {
		return fmt.Errorf("failed to get matmul function")
	}
	
	md.matmulPipeline = C.mtlNewComputePipelineStateWithFunction(md.device, matmulFunc)
	if md.matmulPipeline == nil {
		return fmt.Errorf("failed to create matmul pipeline")
	}
	
	// Compile add kernel
	addSrc := C.CString(addKernel)
	defer C.free(unsafe.Pointer(addSrc))
	
	addLib := C.mtlNewLibraryWithSource(md.device, addSrc)
	if addLib == nil {
		return fmt.Errorf("failed to compile add kernel")
	}
	
	addFunc := C.mtlNewFunctionWithName(addLib, C.CString("add_arrays"))
	if addFunc == nil {
		return fmt.Errorf("failed to get add function")
	}
	
	md.addPipeline = C.mtlNewComputePipelineStateWithFunction(md.device, addFunc)
	if md.addPipeline == nil {
		return fmt.Errorf("failed to create add pipeline")
	}
	
	return nil
}

// MatMul performs matrix multiplication on Metal GPU
func (md *MetalDevice) MatMul(a, b []float32, m, n, k int) ([]float32, error) {
	if len(a) != m*k || len(b) != k*n {
		return nil, fmt.Errorf("invalid matrix dimensions")
	}
	
	// Create buffers
	aBuffer := C.mtlNewBufferWithBytes(md.device, 
		unsafe.Pointer(&a[0]), 
		C.ulong(len(a)*4), 
		C.int(0)) // MTLResourceStorageModeShared = 0
	
	bBuffer := C.mtlNewBufferWithBytes(md.device,
		unsafe.Pointer(&b[0]),
		C.ulong(len(b)*4),
		C.int(0)) // MTLResourceStorageModeShared = 0
	
	result := make([]float32, m*n)
	cBuffer := C.mtlNewBufferWithBytes(md.device,
		unsafe.Pointer(&result[0]),
		C.ulong(len(result)*4),
		C.int(0)) // MTLResourceStorageModeShared = 0
	
	// Create command buffer and encoder
	cmdBuffer := C.mtlCommandBuffer(md.queue)
	encoder := C.mtlComputeCommandEncoder(cmdBuffer)
	
	// Set pipeline and buffers
	C.mtlSetComputePipelineState(encoder, md.matmulPipeline)
	C.mtlSetBuffer(encoder, aBuffer, 0, 0)
	C.mtlSetBuffer(encoder, bBuffer, 0, 1)
	C.mtlSetBuffer(encoder, cBuffer, 0, 2)
	
	// Set dimensions
	mVal := uint32(m)
	nVal := uint32(n)
	kVal := uint32(k)
	C.mtlSetBytes(encoder, unsafe.Pointer(&mVal), 4, 3)
	C.mtlSetBytes(encoder, unsafe.Pointer(&nVal), 4, 4)
	C.mtlSetBytes(encoder, unsafe.Pointer(&kVal), 4, 5)
	
	// Dispatch threads
	C.mtlDispatchThreads(encoder, C.ulong(n), C.ulong(m), 1)
	
	// End encoding and commit
	C.mtlEndEncoding(encoder)
	C.mtlCommit(cmdBuffer)
	C.mtlWaitUntilCompleted(cmdBuffer)
	
	// Copy results back from GPU buffer
	C.mtlCopyBufferData(cBuffer, unsafe.Pointer(&result[0]), C.ulong(len(result)*4))
	
	return result, nil
}

// Add performs element-wise addition on Metal GPU
func (md *MetalDevice) Add(a, b []float32) ([]float32, error) {
	if len(a) != len(b) {
		return nil, fmt.Errorf("arrays must have same length")
	}
	
	// Create buffers
	aBuffer := C.mtlNewBufferWithBytes(md.device,
		unsafe.Pointer(&a[0]),
		C.ulong(len(a)*4),
		C.int(0)) // MTLResourceStorageModeShared = 0
	
	bBuffer := C.mtlNewBufferWithBytes(md.device,
		unsafe.Pointer(&b[0]),
		C.ulong(len(b)*4),
		C.int(0)) // MTLResourceStorageModeShared = 0
	
	result := make([]float32, len(a))
	cBuffer := C.mtlNewBufferWithBytes(md.device,
		unsafe.Pointer(&result[0]),
		C.ulong(len(result)*4),
		C.int(0)) // MTLResourceStorageModeShared = 0
	
	// Create command buffer and encoder
	cmdBuffer := C.mtlCommandBuffer(md.queue)
	encoder := C.mtlComputeCommandEncoder(cmdBuffer)
	
	// Set pipeline and buffers
	C.mtlSetComputePipelineState(encoder, md.addPipeline)
	C.mtlSetBuffer(encoder, aBuffer, 0, 0)
	C.mtlSetBuffer(encoder, bBuffer, 0, 1)
	C.mtlSetBuffer(encoder, cBuffer, 0, 2)
	
	// Dispatch threads
	C.mtlDispatchThreads(encoder, C.ulong(len(a)), 1, 1)
	
	// End encoding and commit
	C.mtlEndEncoding(encoder)
	C.mtlCommit(cmdBuffer)
	C.mtlWaitUntilCompleted(cmdBuffer)
	
	// Copy results back from GPU buffer
	C.mtlCopyBufferData(cBuffer, unsafe.Pointer(&result[0]), C.ulong(len(result)*4))
	
	return result, nil
}