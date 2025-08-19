// +build darwin,cgo

package mlx

import (
	"math"
	"testing"
)

func TestMetalDevice(t *testing.T) {
	_, err := InitMetal()
	if err != nil {
		t.Skipf("Metal not available: %v", err)
	}
	
	t.Logf("Metal device initialized successfully")
}

func TestMetalAdd(t *testing.T) {
	md, err := InitMetal()
	if err != nil {
		t.Skipf("Metal not available: %v", err)
	}
	
	// Test data
	a := []float32{1.0, 2.0, 3.0, 4.0}
	b := []float32{5.0, 6.0, 7.0, 8.0}
	expected := []float32{6.0, 8.0, 10.0, 12.0}
	
	result, err := md.Add(a, b)
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}
	
	// Check results
	for i := range result {
		if math.Abs(float64(result[i]-expected[i])) > 0.0001 {
			t.Errorf("Add mismatch at index %d: got %f, expected %f", 
				i, result[i], expected[i])
		}
	}
	
	t.Logf("Metal Add test passed")
}

func TestMetalMatMul(t *testing.T) {
	md, err := InitMetal()
	if err != nil {
		t.Skipf("Metal not available: %v", err)
	}
	
	// Test 2x3 * 3x2 = 2x2
	a := []float32{
		1, 2, 3,
		4, 5, 6,
	}
	b := []float32{
		7, 8,
		9, 10,
		11, 12,
	}
	expected := []float32{
		58, 64,   // 1*7 + 2*9 + 3*11 = 58, 1*8 + 2*10 + 3*12 = 64
		139, 154, // 4*7 + 5*9 + 6*11 = 139, 4*8 + 5*10 + 6*12 = 154
	}
	
	result, err := md.MatMul(a, b, 2, 2, 3)
	if err != nil {
		t.Fatalf("MatMul failed: %v", err)
	}
	
	// Check results
	for i := range result {
		if math.Abs(float64(result[i]-expected[i])) > 0.0001 {
			t.Errorf("MatMul mismatch at index %d: got %f, expected %f",
				i, result[i], expected[i])
		}
	}
	
	t.Logf("Metal MatMul test passed")
}

func TestMetalLargeMatrix(t *testing.T) {
	md, err := InitMetal()
	if err != nil {
		t.Skipf("Metal not available: %v", err)
	}
	
	// Test larger matrices for performance
	size := 100
	a := make([]float32, size*size)
	b := make([]float32, size*size)
	
	// Initialize with simple values
	for i := range a {
		a[i] = float32(i % 10)
		b[i] = float32(i % 7)
	}
	
	result, err := md.MatMul(a, b, size, size, size)
	if err != nil {
		t.Fatalf("Large MatMul failed: %v", err)
	}
	
	t.Logf("Metal processed %dx%d matrix multiplication", size, size)
	t.Logf("Result has %d elements", len(result))
}

func BenchmarkMetalAdd(b *testing.B) {
	md, err := InitMetal()
	if err != nil {
		b.Skipf("Metal not available: %v", err)
	}
	
	// Create test arrays
	size := 1000000
	a := make([]float32, size)
	c := make([]float32, size)
	
	for i := range a {
		a[i] = float32(i)
		c[i] = float32(i * 2)
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = md.Add(a, c)
	}
	
	b.SetBytes(int64(size * 4 * 2)) // 2 arrays of float32
}

func BenchmarkMetalMatMul(b *testing.B) {
	md, err := InitMetal()
	if err != nil {
		b.Skipf("Metal not available: %v", err)
	}
	
	// Test 1000x1000 matrix multiplication
	size := 1000
	a := make([]float32, size*size)
	c := make([]float32, size*size)
	
	for i := range a {
		a[i] = float32(i % 100)
		c[i] = float32(i % 50)
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = md.MatMul(a, c, size, size, size)
	}
	
	ops := int64(size * size * size * 2) // multiply-add operations
	b.SetBytes(ops * 4) // float32 operations
}

func BenchmarkMetalMatMulSmall(b *testing.B) {
	md, err := InitMetal()
	if err != nil {
		b.Skipf("Metal not available: %v", err)
	}
	
	// Test 100x100 matrix multiplication
	size := 100
	a := make([]float32, size*size)
	c := make([]float32, size*size)
	
	for i := range a {
		a[i] = float32(i % 100)
		c[i] = float32(i % 50)
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = md.MatMul(a, c, size, size, size)
	}
	
	ops := int64(size * size * size * 2) // multiply-add operations
	b.SetBytes(ops * 4) // float32 operations
}