// +build darwin,cgo

package mlx

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation
#include "metal/mtl_bridge.m"
*/
import "C"