package main

import (
	"fmt"
	"log"
	"time"
	
	"github.com/luxfi/mlx"
)

func main() {
	fmt.Println("🚀 MLX GPU Acceleration Demo")
	fmt.Println("=============================")
	
	// Display MLX info
	fmt.Printf("\n%s\n\n", mlx.Info())
	
	// Example 1: Array Operations
	fmt.Println("1. Array Operations Demo")
	fmt.Println("------------------------")
	
	// Create arrays
	a := mlx.Zeros([]int{1000, 1000}, mlx.Float32)
	b := mlx.Ones([]int{1000, 1000}, mlx.Float32)
	
	// Perform operations
	start := time.Now()
	c := mlx.Add(a, b)
	d := mlx.MatMul(c, b)
	mlx.Eval(d)
	mlx.Synchronize()
	elapsed := time.Since(start)
	
	fmt.Printf("✅ Matrix operations (1000x1000) completed in %v\n\n", elapsed)
	
	// Example 2: Order Matching Engine
	fmt.Println("2. Order Matching Engine Demo")
	fmt.Println("-----------------------------")
	
	// Create engine
	engine, err := mlx.NewEngine(mlx.Config{
		Backend: mlx.Auto,
	})
	if err != nil {
		log.Fatalf("Failed to create engine: %v", err)
	}
	defer engine.Close()
	
	fmt.Printf("Backend: %s\n", engine.Backend())
	fmt.Printf("Device: %s\n", engine.Device())
	fmt.Printf("GPU Available: %v\n\n", engine.IsGPUAvailable())
	
	// Create sample orders
	bids := []mlx.Order{
		{ID: 1, Price: 50000.0, Size: 1.0, Side: 0},
		{ID: 2, Price: 49999.0, Size: 2.0, Side: 0},
		{ID: 3, Price: 49998.0, Size: 1.5, Side: 0},
	}
	
	asks := []mlx.Order{
		{ID: 4, Price: 50000.0, Size: 0.5, Side: 1},
		{ID: 5, Price: 50001.0, Size: 1.0, Side: 1},
		{ID: 6, Price: 50002.0, Size: 2.0, Side: 1},
	}
	
	// Match orders
	start = time.Now()
	trades := engine.BatchMatch(bids, asks)
	elapsed = time.Since(start)
	
	fmt.Printf("Matched %d trades in %v\n", len(trades), elapsed)
	for i, trade := range trades {
		fmt.Printf("  Trade %d: Buy #%d ↔ Sell #%d @ %.2f (Size: %.2f)\n",
			i+1, trade.BuyOrderID, trade.SellOrderID, trade.Price, trade.Size)
	}
	
	// Example 3: Benchmark
	fmt.Println("\n3. Performance Benchmark")
	fmt.Println("------------------------")
	
	fmt.Println("Running benchmark with 1M orders...")
	throughput := engine.Benchmark(1000000)
	
	fmt.Printf("📊 Throughput: %.2f orders/sec\n", throughput)
	fmt.Printf("📊 Throughput: %.2f M orders/sec\n", throughput/1000000)
	
	if throughput > 10000000 {
		fmt.Println("🎯 GPU acceleration is working! (>10M orders/sec)")
	} else if throughput > 1000000 {
		fmt.Println("⚡ High performance mode (>1M orders/sec)")
	} else {
		fmt.Println("💻 CPU mode (standard performance)")
	}
	
	// Example 4: Stream Operations
	fmt.Println("\n4. Stream Operations Demo")
	fmt.Println("-------------------------")
	
	stream := mlx.NewStream()
	
	// Queue operations on stream
	x := mlx.Random([]int{5000, 5000}, mlx.Float32)
	y := mlx.Random([]int{5000, 5000}, mlx.Float32)
	
	start = time.Now()
	z := mlx.MatMul(x, y)
	mlx.Eval(z)
	mlx.Synchronize()
	elapsed = time.Since(start)
	
	fmt.Printf("✅ Large matrix multiplication (5000x5000) in %v\n", elapsed)
	
	fmt.Println("\n=============================")
	fmt.Println("✨ MLX Demo Complete!")
}