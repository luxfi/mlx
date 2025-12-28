// +build linux,cgo windows,cgo
// +build cuda

package mlx

/*
#cgo LDFLAGS: -lcudart -lnccl

#include <cuda_runtime.h>
#include <nccl.h>
#include <stdlib.h>
#include <string.h>

// Multi-GPU management
typedef struct {
    int num_gpus;
    int* device_ids;
    cudaStream_t* streams;
    ncclComm_t* comms;
    int peer_access_enabled;
} MultiGPUContext;

static MultiGPUContext* g_mgpu_ctx = NULL;

int mgpu_init(int num_gpus) {
    if (g_mgpu_ctx != NULL) {
        return 0; // Already initialized
    }
    
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count < num_gpus) {
        num_gpus = device_count;
    }
    if (num_gpus == 0) {
        return -1;
    }
    
    g_mgpu_ctx = (MultiGPUContext*)malloc(sizeof(MultiGPUContext));
    g_mgpu_ctx->num_gpus = num_gpus;
    g_mgpu_ctx->device_ids = (int*)malloc(num_gpus * sizeof(int));
    g_mgpu_ctx->streams = (cudaStream_t*)malloc(num_gpus * sizeof(cudaStream_t));
    g_mgpu_ctx->comms = (ncclComm_t*)malloc(num_gpus * sizeof(ncclComm_t));
    g_mgpu_ctx->peer_access_enabled = 0;
    
    // Initialize each GPU
    for (int i = 0; i < num_gpus; i++) {
        g_mgpu_ctx->device_ids[i] = i;
        cudaSetDevice(i);
        cudaStreamCreate(&g_mgpu_ctx->streams[i]);
    }
    
    // Initialize NCCL communicators
    ncclUniqueId nccl_id;
    ncclGetUniqueId(&nccl_id);
    ncclGroupStart();
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        ncclCommInitRank(&g_mgpu_ctx->comms[i], num_gpus, nccl_id, i);
    }
    ncclGroupEnd();
    
    return num_gpus;
}

int mgpu_enable_peer_access() {
    if (g_mgpu_ctx == NULL) return -1;
    
    int num_gpus = g_mgpu_ctx->num_gpus;
    
    // Enable peer access between all GPU pairs
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        for (int j = 0; j < num_gpus; j++) {
            if (i != j) {
                int can_access = 0;
                cudaDeviceCanAccessPeer(&can_access, i, j);
                if (can_access) {
                    cudaDeviceEnablePeerAccess(j, 0);
                }
            }
        }
    }
    
    g_mgpu_ctx->peer_access_enabled = 1;
    return 0;
}

int mgpu_get_count() {
    if (g_mgpu_ctx == NULL) return 0;
    return g_mgpu_ctx->num_gpus;
}

void mgpu_set_device(int device_id) {
    if (g_mgpu_ctx != NULL && device_id < g_mgpu_ctx->num_gpus) {
        cudaSetDevice(g_mgpu_ctx->device_ids[device_id]);
    }
}

void* mgpu_get_stream(int device_id) {
    if (g_mgpu_ctx == NULL || device_id >= g_mgpu_ctx->num_gpus) {
        return NULL;
    }
    return (void*)g_mgpu_ctx->streams[device_id];
}

// Allocate memory on specific GPU
void* mgpu_malloc(int device_id, size_t size) {
    if (g_mgpu_ctx == NULL || device_id >= g_mgpu_ctx->num_gpus) {
        return NULL;
    }
    cudaSetDevice(g_mgpu_ctx->device_ids[device_id]);
    void* ptr = NULL;
    cudaMalloc(&ptr, size);
    return ptr;
}

void mgpu_free(int device_id, void* ptr) {
    if (g_mgpu_ctx == NULL || device_id >= g_mgpu_ctx->num_gpus || ptr == NULL) {
        return;
    }
    cudaSetDevice(g_mgpu_ctx->device_ids[device_id]);
    cudaFree(ptr);
}

// Copy between GPUs (uses NVLink if available)
int mgpu_memcpy_peer(int dst_device, void* dst, int src_device, void* src, size_t size) {
    if (g_mgpu_ctx == NULL) return -1;
    return cudaMemcpyPeer(dst, dst_device, src, src_device, size);
}

// Async copy with stream
int mgpu_memcpy_peer_async(int dst_device, void* dst, int src_device, void* src, size_t size) {
    if (g_mgpu_ctx == NULL) return -1;
    cudaStream_t stream = g_mgpu_ctx->streams[dst_device];
    return cudaMemcpyPeerAsync(dst, dst_device, src, src_device, size, stream);
}

// Synchronize specific GPU
void mgpu_sync(int device_id) {
    if (g_mgpu_ctx == NULL || device_id >= g_mgpu_ctx->num_gpus) return;
    cudaSetDevice(g_mgpu_ctx->device_ids[device_id]);
    cudaDeviceSynchronize();
}

// Synchronize all GPUs
void mgpu_sync_all() {
    if (g_mgpu_ctx == NULL) return;
    for (int i = 0; i < g_mgpu_ctx->num_gpus; i++) {
        cudaSetDevice(g_mgpu_ctx->device_ids[i]);
        cudaDeviceSynchronize();
    }
}

// AllReduce across GPUs using NCCL
int mgpu_allreduce_sum(void** buffers, size_t count, int dtype) {
    if (g_mgpu_ctx == NULL) return -1;
    
    ncclDataType_t nccl_dtype;
    switch (dtype) {
        case 0: nccl_dtype = ncclFloat32; break;
        case 1: nccl_dtype = ncclFloat64; break;
        case 2: nccl_dtype = ncclInt32; break;
        case 3: nccl_dtype = ncclInt64; break;
        default: return -1;
    }
    
    ncclGroupStart();
    for (int i = 0; i < g_mgpu_ctx->num_gpus; i++) {
        cudaSetDevice(g_mgpu_ctx->device_ids[i]);
        ncclAllReduce(buffers[i], buffers[i], count, nccl_dtype, ncclSum,
                      g_mgpu_ctx->comms[i], g_mgpu_ctx->streams[i]);
    }
    ncclGroupEnd();
    
    return 0;
}

// Get GPU memory info
void mgpu_get_memory_info(int device_id, size_t* free, size_t* total) {
    if (g_mgpu_ctx == NULL || device_id >= g_mgpu_ctx->num_gpus) {
        *free = 0;
        *total = 0;
        return;
    }
    cudaSetDevice(g_mgpu_ctx->device_ids[device_id]);
    cudaMemGetInfo(free, total);
}

// Check NVLink connectivity
int mgpu_has_nvlink(int gpu1, int gpu2) {
    int can_access = 0;
    cudaDeviceCanAccessPeer(&can_access, gpu1, gpu2);
    return can_access;
}

void mgpu_shutdown() {
    if (g_mgpu_ctx == NULL) return;
    
    // Destroy NCCL communicators
    for (int i = 0; i < g_mgpu_ctx->num_gpus; i++) {
        ncclCommDestroy(g_mgpu_ctx->comms[i]);
        cudaSetDevice(g_mgpu_ctx->device_ids[i]);
        cudaStreamDestroy(g_mgpu_ctx->streams[i]);
    }
    
    free(g_mgpu_ctx->device_ids);
    free(g_mgpu_ctx->streams);
    free(g_mgpu_ctx->comms);
    free(g_mgpu_ctx);
    g_mgpu_ctx = NULL;
}
*/
import "C"
import (
	"fmt"
	"sync"
	"unsafe"
)

// MultiGPU manages multiple CUDA GPUs with NVLink support
type MultiGPU struct {
	numGPUs  int
	mu       sync.RWMutex
	
	// Per-GPU info
	devices  []GPUDevice
}

// GPUDevice represents a single GPU in a multi-GPU setup
type GPUDevice struct {
	ID          int
	Name        string
	TotalMemory uint64
	FreeMemory  uint64
	NVLinkPeers []int  // IDs of GPUs connected via NVLink
}

// InitMultiGPU initializes multi-GPU support
// Returns the number of GPUs available
func InitMultiGPU(numGPUs int) (*MultiGPU, error) {
	n := int(C.mgpu_init(C.int(numGPUs)))
	if n <= 0 {
		return nil, fmt.Errorf("no CUDA GPUs available")
	}
	
	mg := &MultiGPU{
		numGPUs: n,
		devices: make([]GPUDevice, n),
	}
	
	// Get device info
	for i := 0; i < n; i++ {
		mg.devices[i] = GPUDevice{
			ID:          i,
			Name:        getGPUName(i),
			NVLinkPeers: make([]int, 0),
		}
		
		var free, total C.size_t
		C.mgpu_get_memory_info(C.int(i), &free, &total)
		mg.devices[i].TotalMemory = uint64(total)
		mg.devices[i].FreeMemory = uint64(free)
		
		// Check NVLink connectivity
		for j := 0; j < n; j++ {
			if i != j && C.mgpu_has_nvlink(C.int(i), C.int(j)) != 0 {
				mg.devices[i].NVLinkPeers = append(mg.devices[i].NVLinkPeers, j)
			}
		}
	}
	
	return mg, nil
}

// EnablePeerAccess enables direct GPU-to-GPU memory access via NVLink
func (mg *MultiGPU) EnablePeerAccess() error {
	if ret := C.mgpu_enable_peer_access(); ret != 0 {
		return fmt.Errorf("failed to enable peer access")
	}
	return nil
}

// NumGPUs returns the number of available GPUs
func (mg *MultiGPU) NumGPUs() int {
	return mg.numGPUs
}

// GetDevice returns info about a specific GPU
func (mg *MultiGPU) GetDevice(id int) (GPUDevice, error) {
	if id < 0 || id >= mg.numGPUs {
		return GPUDevice{}, fmt.Errorf("invalid GPU ID %d", id)
	}
	
	// Refresh memory info
	var free, total C.size_t
	C.mgpu_get_memory_info(C.int(id), &free, &total)
	mg.devices[id].FreeMemory = uint64(free)
	
	return mg.devices[id], nil
}

// TotalMemory returns total memory across all GPUs
func (mg *MultiGPU) TotalMemory() uint64 {
	var total uint64
	for _, d := range mg.devices {
		total += d.TotalMemory
	}
	return total
}

// TotalFreeMemory returns total free memory across all GPUs
func (mg *MultiGPU) TotalFreeMemory() uint64 {
	var total uint64
	for i := 0; i < mg.numGPUs; i++ {
		var free, _ C.size_t
		C.mgpu_get_memory_info(C.int(i), &free, &_)
		total += uint64(free)
	}
	return total
}

// SetDevice sets the current CUDA device for subsequent operations
func (mg *MultiGPU) SetDevice(id int) {
	C.mgpu_set_device(C.int(id))
}

// Malloc allocates memory on a specific GPU
func (mg *MultiGPU) Malloc(gpuID int, size uint64) unsafe.Pointer {
	return C.mgpu_malloc(C.int(gpuID), C.size_t(size))
}

// Free frees memory on a specific GPU
func (mg *MultiGPU) Free(gpuID int, ptr unsafe.Pointer) {
	C.mgpu_free(C.int(gpuID), ptr)
}

// MemcpyPeer copies data between GPUs (uses NVLink if available)
func (mg *MultiGPU) MemcpyPeer(dstGPU int, dst unsafe.Pointer, srcGPU int, src unsafe.Pointer, size uint64) error {
	ret := C.mgpu_memcpy_peer(C.int(dstGPU), dst, C.int(srcGPU), src, C.size_t(size))
	if ret != 0 {
		return fmt.Errorf("peer memcpy failed")
	}
	return nil
}

// MemcpyPeerAsync copies data between GPUs asynchronously
func (mg *MultiGPU) MemcpyPeerAsync(dstGPU int, dst unsafe.Pointer, srcGPU int, src unsafe.Pointer, size uint64) error {
	ret := C.mgpu_memcpy_peer_async(C.int(dstGPU), dst, C.int(srcGPU), src, C.size_t(size))
	if ret != 0 {
		return fmt.Errorf("peer memcpy async failed")
	}
	return nil
}

// Sync synchronizes a specific GPU
func (mg *MultiGPU) Sync(gpuID int) {
	C.mgpu_sync(C.int(gpuID))
}

// SyncAll synchronizes all GPUs
func (mg *MultiGPU) SyncAll() {
	C.mgpu_sync_all()
}

// HasNVLink checks if two GPUs are connected via NVLink
func (mg *MultiGPU) HasNVLink(gpu1, gpu2 int) bool {
	return C.mgpu_has_nvlink(C.int(gpu1), C.int(gpu2)) != 0
}

// Shutdown cleans up multi-GPU resources
func (mg *MultiGPU) Shutdown() {
	C.mgpu_shutdown()
}

// PrintTopology prints the GPU topology
func (mg *MultiGPU) PrintTopology() {
	fmt.Printf("Multi-GPU Topology (%d GPUs):\n", mg.numGPUs)
	for _, d := range mg.devices {
		fmt.Printf("  GPU %d: %s (%.1f GB total, %.1f GB free)\n",
			d.ID, d.Name,
			float64(d.TotalMemory)/(1024*1024*1024),
			float64(d.FreeMemory)/(1024*1024*1024))
		if len(d.NVLinkPeers) > 0 {
			fmt.Printf("    NVLink: %v\n", d.NVLinkPeers)
		}
	}
}

func getGPUName(id int) string {
	// Use existing cuda_device_name from mlx_cuda.go
	C.mgpu_set_device(C.int(id))
	namePtr := C.cuda_device_name(C.int(id))
	if namePtr == nil {
		return fmt.Sprintf("GPU %d", id)
	}
	defer C.free(unsafe.Pointer(namePtr))
	return C.GoString(namePtr)
}
