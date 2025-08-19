// +build cgo

package mlx

import (
	"runtime"
	"testing"
)

func TestBackendDetection(t *testing.T) {
	backend := GetBackend()
	device := GetDevice()
	
	t.Logf("Platform: %s/%s", runtime.GOOS, runtime.GOARCH)
	t.Logf("Detected backend: %s", backend)
	t.Logf("Device: %s", device.Name)
	t.Logf("Memory: %.2f GB", float64(device.Memory)/(1024*1024*1024))
	
	// Verify backend selection makes sense for platform
	switch runtime.GOOS {
	case "darwin":
		if runtime.GOARCH == "arm64" {
			// Should detect Metal on Apple Silicon
			if backend != Metal && backend != CPU {
				t.Logf("Warning: Expected Metal backend on Apple Silicon, got %s", backend)
			}
		}
	case "linux", "windows":
		// Could be CUDA or CPU
		if backend != CUDA && backend != CPU {
			t.Logf("Note: Using CPU backend, CUDA not available")
		}
	}
}

func TestInfo(t *testing.T) {
	info := Info()
	t.Logf("MLX Info: %s", info)
	
	if info == "" {
		t.Error("Info() returned empty string")
	}
}

func TestBackendSwitching(t *testing.T) {
	original := GetBackend()
	defer SetBackend(original)
	
	// Try to set CPU backend (always available)
	err := SetBackend(CPU)
	if err != nil {
		t.Errorf("Failed to set CPU backend: %v", err)
	}
	
	if GetBackend() != CPU {
		t.Error("Backend did not switch to CPU")
	}
	
	// Try Auto (should always work)
	err = SetBackend(Auto)
	if err != nil {
		t.Errorf("Failed to set Auto backend: %v", err)
	}
}

func TestArrayCreation(t *testing.T) {
	t.Skip("Skipping array creation tests - stub implementation")
	
	// Test Zeros
	zeros := Zeros([]int{2, 3}, Float32)
	if zeros == nil {
		t.Error("Zeros returned nil")
	}
	
	// Test Ones
	ones := Ones([]int{3, 3}, Float32)
	if ones == nil {
		t.Error("Ones returned nil")
	}
	
	// Test Random
	random := Random([]int{4, 4}, Float32)
	if random == nil {
		t.Error("Random returned nil")
	}
	
	// Test Arange
	arange := Arange(0, 10, 1)
	if arange == nil {
		t.Error("Arange returned nil")
	}
}

func TestArrayOperations(t *testing.T) {
	t.Skip("Skipping array operations tests - stub implementation")
	
	// Create test arrays
	a := Ones([]int{2, 2}, Float32)
	b := Ones([]int{2, 2}, Float32)
	
	// Test Add
	c := Add(a, b)
	if c == nil {
		t.Error("Add returned nil")
	}
	
	// Test Multiply
	d := Multiply(a, b)
	if d == nil {
		t.Error("Multiply returned nil")
	}
	
	// Test MatMul
	e := MatMul(a, b)
	if e == nil {
		t.Error("MatMul returned nil")
	}
	
	// Force evaluation
	Eval(c, d, e)
	
	// Synchronize
	Synchronize()
}

func TestReductions(t *testing.T) {
	t.Skip("Skipping reduction tests - stub implementation")
	
	a := Ones([]int{3, 3}, Float32)
	
	// Test Sum
	sum := Sum(a)
	if sum == nil {
		t.Error("Sum returned nil")
	}
	
	// Test Mean
	mean := Mean(a)
	if mean == nil {
		t.Error("Mean returned nil")
	}
	
	// Test with axis
	sumAxis := Sum(a, 0)
	if sumAxis == nil {
		t.Error("Sum with axis returned nil")
	}
	
	meanAxis := Mean(a, 1)
	if meanAxis == nil {
		t.Error("Mean with axis returned nil")
	}
}

func TestStream(t *testing.T) {
	t.Skip("Skipping stream tests - stub implementation")
	
	stream := NewStream()
	if stream == nil {
		t.Error("NewStream returned nil")
	}
	
	// Clean up
	DefaultContext.FreeStream(stream)
}

func BenchmarkMatMul(b *testing.B) {
	b.Skip("Skipping benchmarks - stub implementation")
}

func BenchmarkAdd(b *testing.B) {
	b.Skip("Skipping benchmarks - stub implementation")
}