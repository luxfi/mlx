// C bridge for Metal API functions used by MLX

#ifndef MTL_BRIDGE_H
#define MTL_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

// Resource storage modes - use values that won't conflict
typedef enum {
    MLX_MTLResourceStorageModeShared = 0,
    MLX_MTLResourceStorageModeManaged = 1,
    MLX_MTLResourceStorageModePrivate = 2,
    MLX_MTLResourceStorageModeMemoryless = 3
} MLX_MTLResourceStorageMode;

// Device functions
void* mtlCreateSystemDefaultDevice();
void* mtlNewCommandQueue(void* device);

// Library and kernel compilation
void* mtlNewLibraryWithSource(void* device, const char* source);
void* mtlNewFunctionWithName(void* library, const char* name);
void* mtlNewComputePipelineStateWithFunction(void* device, void* function);

// Buffer management
void* mtlNewBufferWithBytes(void* device, void* pointer, unsigned long length, int options);
void* mtlNewBufferWithLength(void* device, unsigned long length, int options);

// Command buffer and encoding
void* mtlCommandBuffer(void* queue);
void* mtlComputeCommandEncoder(void* commandBuffer);
void mtlSetComputePipelineState(void* encoder, void* pipelineState);
void mtlSetBuffer(void* encoder, void* buffer, unsigned long offset, unsigned long index);
void mtlSetBytes(void* encoder, const void* bytes, unsigned long length, unsigned long index);
void mtlDispatchThreads(void* encoder, unsigned long width, unsigned long height, unsigned long depth);
void mtlEndEncoding(void* encoder);
void mtlCommit(void* commandBuffer);
void mtlWaitUntilCompleted(void* commandBuffer);

// Buffer data access
void* mtlGetBufferContents(void* buffer);
void mtlCopyBufferData(void* buffer, void* destination, unsigned long length);

#ifdef __cplusplus
}
#endif

#endif // MTL_BRIDGE_H