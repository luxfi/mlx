// +build ignore

package main

import (
    "fmt"
    "time"
    "github.com/luxfi/mlx"
)

func main() {
    fmt.Println("=== MLX Performance Test ===")
    fmt.Printf("System: %s\n", mlx.Info())
    fmt.Println()
    
    // Test different sizes
    sizes := []int{100, 500, 1000}
    
    for _, size := range sizes {
        fmt.Printf("Matrix Size: %dx%d\n", size, size)
        
        // Create matrices
        a := mlx.Random([]int{size, size}, mlx.Float32)
        b := mlx.Random([]int{size, size}, mlx.Float32)
        
        // Warmup
        c := mlx.MatMul(a, b)
        mlx.Eval(c)
        mlx.Synchronize()
        
        // Benchmark
        iterations := 10
        start := time.Now()
        for i := 0; i < iterations; i++ {
            c = mlx.MatMul(a, b)
            mlx.Eval(c)
            mlx.Synchronize()
        }
        elapsed := time.Since(start)
        
        // Calculate metrics
        msPerOp := elapsed.Seconds() * 1000 / float64(iterations)
        ops := int64(size * size * size * 2) // multiply-add operations
        gflops := float64(ops*int64(iterations)) / elapsed.Seconds() / 1e9
        
        fmt.Printf("  Time: %.2f ms per operation\n", msPerOp)
        fmt.Printf("  Performance: %.1f GFLOPS\n", gflops)
        fmt.Println()
    }
    
    fmt.Println("Metal backend automatically selected for optimal performance on Apple Silicon!")
}