//go:build !onnx
// +build !onnx

package mlx

// Stub functions when ONNX is not available

func detectONNXBackend() bool {
	return false
}

func getONNXVersion() string {
	return "not available"
}

func hasONNXSupport() bool {
	return false
}
