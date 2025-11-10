// +build ignore

package main

import (
    "fmt"
    "github.com/luxfi/mlx"
)

func main() {
    fmt.Println("=== MLX Memory Detection Test ===")
    
    backend := mlx.GetBackend()
    device := mlx.GetDevice()
    
    fmt.Printf("Backend: %s\n", backend)
    fmt.Printf("Device: %s\n", device.Name)
    fmt.Printf("Memory: %.1f GB\n", float64(device.Memory)/(1024*1024*1024))
    
    // Show info
    fmt.Printf("\nFull Info: %s\n", mlx.Info())
}