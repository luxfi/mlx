// Objective-C implementation of Metal bridge functions
// Provides C interface to Metal framework

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <string.h>

// Helper function to get Metal device
id<MTLDevice> getDevice(void* devicePtr) {
    return (__bridge id<MTLDevice>)devicePtr;
}

// Helper function to get command queue
id<MTLCommandQueue> getQueue(void* queuePtr) {
    return (__bridge id<MTLCommandQueue>)queuePtr;
}

// Device functions
void* mtlCreateSystemDefaultDevice() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        return NULL;
    }
    return (__bridge_retained void*)device;
}

void* mtlNewCommandQueue(void* devicePtr) {
    id<MTLDevice> device = getDevice(devicePtr);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    return (__bridge_retained void*)queue;
}

// Library and kernel compilation
void* mtlNewLibraryWithSource(void* devicePtr, const char* source) {
    id<MTLDevice> device = getDevice(devicePtr);
    NSError* error = nil;
    NSString* sourceStr = [NSString stringWithUTF8String:source];
    
    id<MTLLibrary> library = [device newLibraryWithSource:sourceStr
                                                   options:nil
                                                     error:&error];
    if (error) {
        NSLog(@"Failed to compile Metal library: %@", error);
        return NULL;
    }
    return (__bridge_retained void*)library;
}

void* mtlNewFunctionWithName(void* libraryPtr, const char* name) {
    id<MTLLibrary> library = (__bridge id<MTLLibrary>)libraryPtr;
    NSString* nameStr = [NSString stringWithUTF8String:name];
    id<MTLFunction> function = [library newFunctionWithName:nameStr];
    if (!function) {
        return NULL;
    }
    return (__bridge_retained void*)function;
}

void* mtlNewComputePipelineStateWithFunction(void* devicePtr, void* functionPtr) {
    id<MTLDevice> device = getDevice(devicePtr);
    id<MTLFunction> function = (__bridge id<MTLFunction>)functionPtr;
    NSError* error = nil;
    
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                  error:&error];
    if (error) {
        NSLog(@"Failed to create compute pipeline: %@", error);
        return NULL;
    }
    return (__bridge_retained void*)pipeline;
}

// Buffer management
void* mtlNewBufferWithBytes(void* devicePtr, void* pointer, unsigned long length, int options) {
    id<MTLDevice> device = getDevice(devicePtr);
    id<MTLBuffer> buffer = [device newBufferWithBytes:pointer
                                                length:length
                                               options:(MTLResourceOptions)options];
    return (__bridge_retained void*)buffer;
}

void* mtlNewBufferWithLength(void* devicePtr, unsigned long length, int options) {
    id<MTLDevice> device = getDevice(devicePtr);
    id<MTLBuffer> buffer = [device newBufferWithLength:length
                                                options:(MTLResourceOptions)options];
    return (__bridge_retained void*)buffer;
}

// Command buffer and encoding
void* mtlCommandBuffer(void* queuePtr) {
    id<MTLCommandQueue> queue = getQueue(queuePtr);
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    return (__bridge_retained void*)commandBuffer;
}

void* mtlComputeCommandEncoder(void* commandBufferPtr) {
    id<MTLCommandBuffer> commandBuffer = (__bridge id<MTLCommandBuffer>)commandBufferPtr;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    return (__bridge_retained void*)encoder;
}

void mtlSetComputePipelineState(void* encoderPtr, void* pipelineStatePtr) {
    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)encoderPtr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipelineStatePtr;
    [encoder setComputePipelineState:pipeline];
}

void mtlSetBuffer(void* encoderPtr, void* bufferPtr, unsigned long offset, unsigned long index) {
    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)encoderPtr;
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)bufferPtr;
    [encoder setBuffer:buffer offset:offset atIndex:index];
}

void mtlSetBytes(void* encoderPtr, const void* bytes, unsigned long length, unsigned long index) {
    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)encoderPtr;
    [encoder setBytes:bytes length:length atIndex:index];
}

void mtlDispatchThreads(void* encoderPtr, unsigned long width, unsigned long height, unsigned long depth) {
    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)encoderPtr;
    
    MTLSize gridSize = MTLSizeMake(width, height, depth);
    
    // Calculate appropriate threadgroup size
    NSUInteger w = MIN(width, 32);
    NSUInteger h = MIN(height, 32);
    NSUInteger d = 1;
    MTLSize threadgroupSize = MTLSizeMake(w, h, d);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
}

void mtlEndEncoding(void* encoderPtr) {
    id<MTLComputeCommandEncoder> encoder = (__bridge id<MTLComputeCommandEncoder>)encoderPtr;
    [encoder endEncoding];
}

void mtlCommit(void* commandBufferPtr) {
    id<MTLCommandBuffer> commandBuffer = (__bridge id<MTLCommandBuffer>)commandBufferPtr;
    [commandBuffer commit];
}

void mtlWaitUntilCompleted(void* commandBufferPtr) {
    id<MTLCommandBuffer> commandBuffer = (__bridge id<MTLCommandBuffer>)commandBufferPtr;
    [commandBuffer waitUntilCompleted];
}

// Buffer data access
void* mtlGetBufferContents(void* bufferPtr) {
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)bufferPtr;
    return [buffer contents];
}

void mtlCopyBufferData(void* bufferPtr, void* destination, unsigned long length) {
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)bufferPtr;
    memcpy(destination, [buffer contents], length);
}