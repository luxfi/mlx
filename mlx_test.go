package mlx_test

import (
	"testing"
	
	"github.com/luxfi/mlx"
)

func TestBackendDetection(t *testing.T) {
	backend := mlx.GetBackend()
	t.Logf("Detected backend: %s", backend)
	
	device := mlx.GetDevice()
	if device != nil {
		t.Logf("Device: %s (Memory: %d GB)", 
			device.Name, 
			device.Memory/(1024*1024*1024))
	}
}

func TestArrayOperations(t *testing.T) {
	// Create arrays
	a := mlx.Zeros([]int{10, 10}, mlx.Float32)
	b := mlx.Ones([]int{10, 10}, mlx.Float32)
	
	// Add arrays
	c := mlx.Add(a, b)
	
	// Force evaluation
	mlx.Eval(c)
	
	// Wait for completion
	mlx.Synchronize()
	
	t.Log("Array operations completed")
}

func TestMatrixMultiplication(t *testing.T) {
	// Create matrices
	a := mlx.Random([]int{100, 50}, mlx.Float32)
	b := mlx.Random([]int{50, 75}, mlx.Float32)
	
	// Matrix multiplication
	c := mlx.MatMul(a, b)
	
	// Force evaluation
	mlx.Eval(c)
	mlx.Synchronize()
	
	t.Log("Matrix multiplication completed")
}

func TestReduction(t *testing.T) {
	// Create array
	a := mlx.Arange(0, 100, 1)
	
	// Sum all elements
	sum := mlx.Sum(a)
	
	// Mean of elements
	mean := mlx.Mean(a)
	
	// Evaluate
	mlx.Eval(sum, mean)
	mlx.Synchronize()
	
	t.Log("Reduction operations completed")
}

func TestStream(t *testing.T) {
	// Create compute stream
	_ = mlx.NewStream()
	
	// Create arrays
	a := mlx.Random([]int{1000, 1000}, mlx.Float32)
	b := mlx.Random([]int{1000, 1000}, mlx.Float32)
	
	// Perform operations
	c := mlx.MatMul(a, b)
	
	// Force evaluation
	mlx.Eval(c)
	
	// Synchronize stream
	mlx.Synchronize()
	
	t.Log("Stream operations completed")
}

func BenchmarkMatMul(b *testing.B) {
	// Create large matrices
	x := mlx.Random([]int{1000, 1000}, mlx.Float32)
	y := mlx.Random([]int{1000, 1000}, mlx.Float32)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		z := mlx.MatMul(x, y)
		mlx.Eval(z)
	}
	mlx.Synchronize()
}

func BenchmarkArrayCreation(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a := mlx.Zeros([]int{100, 100}, mlx.Float32)
		mlx.Eval(a)
	}
	mlx.Synchronize()
}

func BenchmarkReduction(b *testing.B) {
	a := mlx.Random([]int{10000}, mlx.Float32)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sum := mlx.Sum(a)
		mlx.Eval(sum)
	}
	mlx.Synchronize()
}