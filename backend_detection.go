// +build cgo

package mlx

/*
#include "mlx_c_api.h"
#include <stdlib.h>
*/
import "C"
import (
	"runtime"
	"unsafe"
)

// hasMetalSupport checks if Metal is available
func hasMetalSupport() bool {
	if runtime.GOOS != "darwin" {
		return false
	}
	return bool(C.mlx_has_metal())
}

// hasCUDASupport checks if CUDA is available
func hasCUDASupport() bool {
	if runtime.GOOS == "darwin" {
		return false // No CUDA on macOS
	}
	return bool(C.mlx_has_cuda())
}

// getMetalDeviceName returns the Metal device name
func getMetalDeviceName() string {
	namePtr := C.mlx_get_metal_device_name()
	if namePtr == nil {
		return "Unknown Metal Device"
	}
	defer C.free(unsafe.Pointer(namePtr))
	return C.GoString(namePtr)
}

// getCUDADeviceName returns the CUDA device name
func getCUDADeviceName() string {
	namePtr := C.mlx_get_cuda_device_name()
	if namePtr == nil {
		return "Unknown CUDA Device"
	}
	defer C.free(unsafe.Pointer(namePtr))
	return C.GoString(namePtr)
}

// getMetalMemory returns available Metal memory
func getMetalMemory() int64 {
	return int64(C.mlx_get_metal_memory())
}

// getCUDAMemory returns available CUDA memory
func getCUDAMemory() int64 {
	return int64(C.mlx_get_cuda_memory())
}

// getSystemMemory returns system RAM
func getSystemMemory() int64 {
	return int64(C.mlx_get_system_memory())
}