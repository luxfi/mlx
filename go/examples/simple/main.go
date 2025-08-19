package main

import (
	"fmt"
	"log"
	
	"github.com/luxfi/mlx"
)

func main() {
	fmt.Println("🚀 MLX Go Bindings - Simple Example")
	fmt.Println("=====================================")
	
	// Display MLX info
	fmt.Printf("\nSystem Info:\n%s\n", mlx.Info())
	
	// Get backend and device info
	backend := mlx.GetBackend()
	device := mlx.GetDevice()
	
	fmt.Printf("\nBackend: %s\n", backend)
	fmt.Printf("Device: %s\n", device.Name)
	fmt.Printf("Memory: %.1f GB\n", float64(device.Memory)/(1024*1024*1024))
	fmt.Printf("GPU Available: %v\n", device.Type != mlx.CPU)
	
	// Create arrays
	fmt.Println("\nCreating arrays...")
	a := mlx.Zeros([]int{100, 100}, mlx.Float32)
	b := mlx.Ones([]int{100, 100}, mlx.Float32)
	
	// Perform operations
	fmt.Println("Performing operations...")
	c := mlx.Add(a, b)
	d := mlx.MatMul(c, b)
	
	// Force evaluation
	mlx.Eval(d)
	mlx.Synchronize()
	
	fmt.Println("✅ Operations completed successfully!")
	
	// Create matching engine for order book demo
	fmt.Println("\n--- Order Matching Demo ---")
	
	engine, err := mlx.NewEngine(mlx.Config{
		Backend: mlx.Auto,
	})
	if err != nil {
		log.Printf("Warning: Could not create engine: %v", err)
		return
	}
	defer engine.Close()
	
	// Create sample orders
	bids := []mlx.Order{
		{ID: 1, Price: 100.00, Size: 10.0, Side: 0},
		{ID: 2, Price: 99.99, Size: 20.0, Side: 0},
		{ID: 3, Price: 99.98, Size: 15.0, Side: 0},
	}
	
	asks := []mlx.Order{
		{ID: 4, Price: 100.00, Size: 5.0, Side: 1},
		{ID: 5, Price: 100.01, Size: 10.0, Side: 1},
		{ID: 6, Price: 100.02, Size: 25.0, Side: 1},
	}
	
	// Match orders
	trades := engine.BatchMatch(bids, asks)
	
	fmt.Printf("\nMatched %d trades:\n", len(trades))
	for i, trade := range trades {
		fmt.Printf("  Trade %d: Buy #%d ↔ Sell #%d @ %.2f (Size: %.2f)\n",
			i+1, trade.BuyOrderID, trade.SellOrderID, trade.Price, trade.Size)
	}
	
	// Run benchmark
	fmt.Println("\nRunning performance benchmark...")
	throughput := engine.Benchmark(100000)
	fmt.Printf("📊 Throughput: %.2f orders/sec\n", throughput)
	fmt.Printf("📊 Throughput: %.2f M orders/sec\n", throughput/1000000)
	
	fmt.Println("\n✨ Demo complete!")
}