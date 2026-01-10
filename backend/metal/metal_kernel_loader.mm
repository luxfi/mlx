// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Metal Kernel Loader - Compile and cache Metal compute pipelines

#if defined(__APPLE__)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "lux/gpu/kernel_loader.h"
#include <unordered_map>
#include <string>
#include <mutex>
#include <vector>

namespace lux::gpu::metal {

// =============================================================================
// Metal Kernel Implementation
// =============================================================================

struct MetalKernel {
    id<MTLComputePipelineState> pipeline;
    id<MTLLibrary> library;
    std::string entry_point;
    NSUInteger max_threads_per_group;
};

// =============================================================================
// Metal Kernel Cache
// =============================================================================

class MetalKernelCache {
public:
    static MetalKernelCache& instance() {
        static MetalKernelCache cache;
        return cache;
    }

    // Compile kernel from source
    MetalKernel* compile(
        id<MTLDevice> device,
        const char* source,
        size_t source_len,
        const char* entry_point,
        const char* compile_opts
    ) {
        @autoreleasepool {
            NSError* error = nil;

            // Create source string
            NSString* src = source_len > 0
                ? [[NSString alloc] initWithBytes:source length:source_len encoding:NSUTF8StringEncoding]
                : [NSString stringWithUTF8String:source];

            if (!src) {
                NSLog(@"MetalKernelCache: Invalid source encoding");
                return nullptr;
            }

            // Compile options
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            if (compile_opts && *compile_opts) {
                // Parse simple options like "-DFOO=1"
                NSString* opts = [NSString stringWithUTF8String:compile_opts];
                NSArray* parts = [opts componentsSeparatedByString:@" "];
                NSMutableDictionary* defines = [NSMutableDictionary dictionary];

                for (NSString* part in parts) {
                    if ([part hasPrefix:@"-D"]) {
                        NSString* def = [part substringFromIndex:2];
                        NSArray* kv = [def componentsSeparatedByString:@"="];
                        if (kv.count == 2) {
                            defines[kv[0]] = kv[1];
                        } else if (kv.count == 1) {
                            defines[kv[0]] = @"1";
                        }
                    }
                }

                if (defines.count > 0) {
                    options.preprocessorMacros = defines;
                }
            }

            // Compile library
            id<MTLLibrary> library = [device newLibraryWithSource:src options:options error:&error];
            if (!library) {
                NSLog(@"MetalKernelCache: Compilation failed: %@", error);
                return nullptr;
            }

            // Get function
            NSString* funcName = [NSString stringWithUTF8String:entry_point];
            id<MTLFunction> function = [library newFunctionWithName:funcName];
            if (!function) {
                NSLog(@"MetalKernelCache: Function '%s' not found", entry_point);
                return nullptr;
            }

            // Create pipeline
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            if (!pipeline) {
                NSLog(@"MetalKernelCache: Pipeline creation failed: %@", error);
                return nullptr;
            }

            // Create kernel object
            auto kernel = new MetalKernel();
            kernel->pipeline = pipeline;
            kernel->library = library;
            kernel->entry_point = entry_point;
            kernel->max_threads_per_group = [pipeline maxTotalThreadsPerThreadgroup];

            return kernel;
        }
    }

    // Load from metallib binary
    MetalKernel* load_binary(
        id<MTLDevice> device,
        const void* binary,
        size_t binary_len,
        const char* entry_point
    ) {
        @autoreleasepool {
            NSError* error = nil;

            // Create data from binary
            NSData* data = [NSData dataWithBytes:binary length:binary_len];
            if (!data) {
                NSLog(@"MetalKernelCache: Invalid binary data");
                return nullptr;
            }

            // Create library from binary
            dispatch_data_t dispatch_data = dispatch_data_create(
                data.bytes, data.length, nil, DISPATCH_DATA_DESTRUCTOR_DEFAULT
            );
            id<MTLLibrary> library = [device newLibraryWithData:dispatch_data error:&error];
            if (!library) {
                NSLog(@"MetalKernelCache: Binary library load failed: %@", error);
                return nullptr;
            }

            // Get function
            NSString* funcName = [NSString stringWithUTF8String:entry_point];
            id<MTLFunction> function = [library newFunctionWithName:funcName];
            if (!function) {
                NSLog(@"MetalKernelCache: Function '%s' not found in binary", entry_point);
                return nullptr;
            }

            // Create pipeline
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            if (!pipeline) {
                NSLog(@"MetalKernelCache: Pipeline creation failed: %@", error);
                return nullptr;
            }

            auto kernel = new MetalKernel();
            kernel->pipeline = pipeline;
            kernel->library = library;
            kernel->entry_point = entry_point;
            kernel->max_threads_per_group = [pipeline maxTotalThreadsPerThreadgroup];

            return kernel;
        }
    }

    // Load from file
    MetalKernel* load_file(
        id<MTLDevice> device,
        const char* path,
        const char* entry_point
    ) {
        @autoreleasepool {
            NSError* error = nil;
            NSString* nsPath = [NSString stringWithUTF8String:path];

            // Check extension
            if ([nsPath hasSuffix:@".metallib"]) {
                // Load precompiled library
                NSURL* url = [NSURL fileURLWithPath:nsPath];
                id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
                if (!library) {
                    NSLog(@"MetalKernelCache: Failed to load metallib: %@", error);
                    return nullptr;
                }

                NSString* funcName = [NSString stringWithUTF8String:entry_point];
                id<MTLFunction> function = [library newFunctionWithName:funcName];
                if (!function) {
                    return nullptr;
                }

                id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
                if (!pipeline) {
                    return nullptr;
                }

                auto kernel = new MetalKernel();
                kernel->pipeline = pipeline;
                kernel->library = library;
                kernel->entry_point = entry_point;
                kernel->max_threads_per_group = [pipeline maxTotalThreadsPerThreadgroup];
                return kernel;
            } else {
                // Load source file
                NSString* source = [NSString stringWithContentsOfFile:nsPath encoding:NSUTF8StringEncoding error:&error];
                if (!source) {
                    NSLog(@"MetalKernelCache: Failed to read source file: %@", error);
                    return nullptr;
                }

                return compile(device, [source UTF8String], 0, entry_point, nullptr);
            }
        }
    }

    void destroy(MetalKernel* kernel) {
        if (kernel) {
            @autoreleasepool {
                kernel->pipeline = nil;
                kernel->library = nil;
            }
            delete kernel;
        }
    }

private:
    MetalKernelCache() = default;
    std::mutex mutex_;
};

} // namespace lux::gpu::metal

// =============================================================================
// C API Implementation for Metal
// =============================================================================

extern "C" {

using namespace lux::gpu::metal;

LuxKernel* lux_metal_kernel_compile(
    void* device,
    const char* source,
    size_t source_len,
    const char* entry_point,
    const char* compile_opts
) {
    if (!device || !source || !entry_point) return nullptr;

    id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;
    auto kernel = MetalKernelCache::instance().compile(
        mtl_device, source, source_len, entry_point, compile_opts
    );
    return reinterpret_cast<LuxKernel*>(kernel);
}

LuxKernel* lux_metal_kernel_load_binary(
    void* device,
    const void* binary,
    size_t binary_len,
    const char* entry_point
) {
    if (!device || !binary || !entry_point) return nullptr;

    id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;
    auto kernel = MetalKernelCache::instance().load_binary(
        mtl_device, binary, binary_len, entry_point
    );
    return reinterpret_cast<LuxKernel*>(kernel);
}

LuxKernel* lux_metal_kernel_load_file(
    void* device,
    const char* path,
    const char* entry_point
) {
    if (!device || !path || !entry_point) return nullptr;

    id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;
    auto kernel = MetalKernelCache::instance().load_file(mtl_device, path, entry_point);
    return reinterpret_cast<LuxKernel*>(kernel);
}

void lux_metal_kernel_destroy(LuxKernel* kernel) {
    MetalKernelCache::instance().destroy(reinterpret_cast<MetalKernel*>(kernel));
}

const char* lux_metal_kernel_entry_point(LuxKernel* kernel) {
    auto k = reinterpret_cast<MetalKernel*>(kernel);
    return k ? k->entry_point.c_str() : nullptr;
}

// Get MTLComputePipelineState for dispatch
void* lux_metal_kernel_pipeline(LuxKernel* kernel) {
    auto k = reinterpret_cast<MetalKernel*>(kernel);
    return k ? (__bridge void*)k->pipeline : nullptr;
}

// Get max threads per threadgroup
uint32_t lux_metal_kernel_max_threads(LuxKernel* kernel) {
    auto k = reinterpret_cast<MetalKernel*>(kernel);
    return k ? static_cast<uint32_t>(k->max_threads_per_group) : 0;
}

} // extern "C"

// =============================================================================
// Metal Kernel Manager - High-level interface
// =============================================================================

namespace lux::gpu::metal {

class MetalKernelManager {
public:
    MetalKernelManager(id<MTLDevice> device) : device_(device) {
        cache_ = lux_kernel_cache_create();
        load_embedded_kernels();
    }

    ~MetalKernelManager() {
        lux_kernel_cache_clear(cache_);
        lux_kernel_cache_destroy(cache_);
    }

    // Get or compile kernel by name
    MetalKernel* get_kernel(const char* name, uint32_t dtype = 0, uint32_t size_hint = 0) {
        LuxKernelVariant variant = { name, dtype, size_hint, 0 };

        // Check cache
        LuxKernel* cached = lux_kernel_cache_get(cache_, &variant);
        if (cached) {
            return reinterpret_cast<MetalKernel*>(cached);
        }

        // Find in registry
        const LuxKernelRegistry* registry = lux_kernel_registry_get("metal");
        if (!registry) return nullptr;

        const LuxEmbeddedKernel* embedded = lux_kernel_registry_find(registry, name);
        if (!embedded) return nullptr;

        // Compile
        MetalKernel* kernel = nullptr;
        if (embedded->binary && embedded->binary_len > 0) {
            kernel = MetalKernelCache::instance().load_binary(
                device_, embedded->binary, embedded->binary_len, embedded->entry_point
            );
        } else if (embedded->source && embedded->source_len > 0) {
            kernel = MetalKernelCache::instance().compile(
                device_, embedded->source, embedded->source_len, embedded->entry_point, nullptr
            );
        }

        if (kernel) {
            lux_kernel_cache_put(cache_, &variant, reinterpret_cast<LuxKernel*>(kernel));
        }

        return kernel;
    }

    // Preload all embedded kernels
    void preload_all() {
        const LuxKernelRegistry* registry = lux_kernel_registry_get("metal");
        if (!registry) return;

        for (size_t i = 0; i < registry->count; ++i) {
            get_kernel(registry->kernels[i].name);
        }
    }

private:
    void load_embedded_kernels() {
        // Kernels are lazy-loaded on first use
    }

    id<MTLDevice> device_;
    LuxKernelCache* cache_;
};

} // namespace lux::gpu::metal

#endif // __APPLE__
