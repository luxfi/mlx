// +build cgo

package mlx

import (
	"os"
	"runtime"
	"testing"
	"unsafe"
)

func TestONNXBackendDetection(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("ONNX backend detection is Windows-specific")
	}

	// Test ONNX support detection
	hasONNX := hasONNXSupport()
	t.Logf("ONNX Runtime available: %v", hasONNX)

	if hasONNX {
		version := getONNXVersion()
		t.Logf("ONNX Runtime version: %s", version)
		
		if version == "" {
			t.Error("ONNX version should not be empty when available")
		}
	}
}

func TestONNXBackendSelection(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("ONNX backend is Windows-specific")
	}

	// Save original backend
	originalBackend := os.Getenv("MLX_BACKEND")
	defer func() {
		if originalBackend != "" {
			os.Setenv("MLX_BACKEND", originalBackend)
		} else {
			os.Unsetenv("MLX_BACKEND")
		}
	}()

	// Test explicit ONNX backend selection
	os.Setenv("MLX_BACKEND", "onnx")
	
	ctx := &Context{
		arrays:  make(map[unsafe.Pointer]*Array),
		streams: make(map[unsafe.Pointer]*Stream),
		version: Version,
	}
	
	ctx.detectBackend()
	
	if hasONNXSupport() {
		if ctx.backend != ONNX {
			t.Errorf("Expected ONNX backend on Windows with ONNX Runtime, got %s", ctx.backend)
		}
		
		device := ctx.GetDevice()
		if device == nil {
			t.Fatal("Device should not be nil")
		}
		
		if device.Type != ONNX {
			t.Errorf("Expected device type ONNX, got %s", device.Type)
		}
		
		t.Logf("Device: %s", device.Name)
	} else {
		t.Log("ONNX Runtime not available, skipping backend test")
	}
}

func TestONNXFallbackMode(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("ONNX fallback is Windows-specific")
	}

	// Test that we can create context even without MLX library
	ctx := &Context{
		arrays:  make(map[unsafe.Pointer]*Array),
		streams: make(map[unsafe.Pointer]*Stream),
		version: Version,
	}
	
	ctx.detectBackend()
	
	backend := ctx.GetBackend()
	t.Logf("Selected backend: %s", backend)
	
	// Should be ONNX or CPU, never fail
	if backend != ONNX && backend != CPU {
		t.Errorf("Expected ONNX or CPU backend, got %s", backend)
	}
}

func TestONNXArrayOperations(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("ONNX backend is Windows-specific")
	}

	if !hasONNXSupport() {
		t.Skip("ONNX Runtime not available")
	}

	// Create context with ONNX backend
	ctx := &Context{
		backend: ONNX,
		device: &Device{
			Type:   ONNX,
			ID:     0,
			Name:   "ONNX Runtime " + getONNXVersion(),
			Memory: getSystemMemory(),
		},
		arrays:  make(map[unsafe.Pointer]*Array),
		streams: make(map[unsafe.Pointer]*Stream),
		version: Version,
	}

	// Test basic array operations
	a := ctx.Zeros([]int{3, 3}, Float32)
	if a == nil {
		t.Fatal("Zeros should return non-nil array")
	}
	
	if len(a.Shape()) != 2 {
		t.Errorf("Expected 2D array, got %v", a.Shape())
	}
	
	b := ctx.Ones([]int{3, 3}, Float32)
	if b == nil {
		t.Fatal("Ones should return non-nil array")
	}
	
	// Test operations
	c := ctx.Add(a, b)
	if c == nil {
		t.Fatal("Add should return non-nil array")
	}
	
	d := ctx.MatMul(a, b)
	if d == nil {
		t.Fatal("MatMul should return non-nil array")
	}
	
	t.Log("ONNX backend operations successful")
}

func TestONNXInfo(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("ONNX backend is Windows-specific")
	}

	// Test Info() function with ONNX backend
	info := Info()
	t.Logf("MLX Info: %s", info)
	
	if info == "" {
		t.Error("Info should not be empty")
	}
}
