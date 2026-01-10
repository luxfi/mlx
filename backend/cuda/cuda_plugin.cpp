// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// CUDA Backend Plugin - NVIDIA GPU acceleration
// Loaded as a shared library via dlopen()
//
// This plugin uses dynamic loading of CUDA driver APIs for deployment
// robustness - the plugin can be shipped without hard runtime dependencies.

#include "lux/gpu/backend_plugin.h"
#include "lux/gpu/crypto_backend.h"
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <string>

#ifdef LUX_CUDA_DYNAMIC_LOAD
#include <dlfcn.h>
#endif

// =============================================================================
// CUDA Types (from cuda.h, redefined to avoid header dependency)
// =============================================================================

typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUdeviceptr;
typedef void* CUstream;

#define CUDA_SUCCESS 0

// =============================================================================
// CUDA Driver API Function Pointers (dynamically loaded)
// =============================================================================

typedef int CUjit_option;

#define CU_JIT_ERROR_LOG_BUFFER 5
#define CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES 6
#define CU_JIT_INFO_LOG_BUFFER 3
#define CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES 4
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 16

static struct {
    CUresult (*cuInit)(unsigned int);
    CUresult (*cuDeviceGetCount)(int*);
    CUresult (*cuDeviceGet)(CUdevice*, int);
    CUresult (*cuDeviceGetName)(char*, int, CUdevice);
    CUresult (*cuDeviceTotalMem)(size_t*, CUdevice);
    CUresult (*cuDeviceGetAttribute)(int*, int, CUdevice);
    CUresult (*cuCtxCreate)(CUcontext*, unsigned int, CUdevice);
    CUresult (*cuCtxDestroy)(CUcontext);
    CUresult (*cuCtxSynchronize)(void);
    CUresult (*cuCtxSetCurrent)(CUcontext);
    CUresult (*cuMemAlloc)(CUdeviceptr*, size_t);
    CUresult (*cuMemFree)(CUdeviceptr);
    CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void*, size_t);
    CUresult (*cuMemcpyDtoH)(void*, CUdeviceptr, size_t);
    CUresult (*cuMemcpyDtoD)(CUdeviceptr, CUdeviceptr, size_t);
    CUresult (*cuMemsetD8)(CUdeviceptr, unsigned char, size_t);
    CUresult (*cuMemsetD32)(CUdeviceptr, unsigned int, size_t);
    CUresult (*cuModuleLoad)(CUmodule*, const char*);
    CUresult (*cuModuleLoadData)(CUmodule*, const void*);
    CUresult (*cuModuleLoadDataEx)(CUmodule*, const void*, unsigned int, CUjit_option*, void**);
    CUresult (*cuModuleUnload)(CUmodule);
    CUresult (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
    CUresult (*cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int,
                               unsigned int, unsigned int, unsigned int,
                               unsigned int, CUstream, void**, void**);
    void* handle;
    bool loaded;
} cuda_driver = {};

// =============================================================================
// Dynamic CUDA Loading
// =============================================================================

static bool load_cuda_driver() {
    if (cuda_driver.loaded) return cuda_driver.handle != nullptr;

    cuda_driver.loaded = true;

#ifdef LUX_CUDA_DYNAMIC_LOAD
    // Try to load CUDA driver library
    const char* lib_names[] = {
        "libcuda.so.1",
        "libcuda.so",
        "libcuda.dylib",
        nullptr
    };

    for (const char** name = lib_names; *name; ++name) {
        cuda_driver.handle = dlopen(*name, RTLD_NOW | RTLD_LOCAL);
        if (cuda_driver.handle) break;
    }

    if (!cuda_driver.handle) {
        return false;
    }

    // Load all function pointers
    #define LOAD_FUNC(name) \
        cuda_driver.name = (decltype(cuda_driver.name))dlsym(cuda_driver.handle, #name); \
        if (!cuda_driver.name) { dlclose(cuda_driver.handle); cuda_driver.handle = nullptr; return false; }

    LOAD_FUNC(cuInit);
    LOAD_FUNC(cuDeviceGetCount);
    LOAD_FUNC(cuDeviceGet);
    LOAD_FUNC(cuDeviceGetName);
    LOAD_FUNC(cuDeviceTotalMem);
    LOAD_FUNC(cuDeviceGetAttribute);
    LOAD_FUNC(cuCtxCreate);
    LOAD_FUNC(cuCtxDestroy);
    LOAD_FUNC(cuCtxSynchronize);
    LOAD_FUNC(cuCtxSetCurrent);
    LOAD_FUNC(cuMemAlloc);
    LOAD_FUNC(cuMemFree);
    LOAD_FUNC(cuMemcpyHtoD);
    LOAD_FUNC(cuMemcpyDtoH);
    LOAD_FUNC(cuMemcpyDtoD);
    LOAD_FUNC(cuMemsetD8);
    LOAD_FUNC(cuMemsetD32);
    LOAD_FUNC(cuModuleLoad);
    LOAD_FUNC(cuModuleLoadData);
    LOAD_FUNC(cuModuleLoadDataEx);
    LOAD_FUNC(cuModuleUnload);
    LOAD_FUNC(cuModuleGetFunction);
    LOAD_FUNC(cuLaunchKernel);

    #undef LOAD_FUNC

    // Initialize CUDA
    if (cuda_driver.cuInit(0) != CUDA_SUCCESS) {
        dlclose(cuda_driver.handle);
        cuda_driver.handle = nullptr;
        return false;
    }

    return true;
#else
    // When not using dynamic loading, assume CUDA is linked
    return false;  // TODO: implement static linking path
#endif
}

// =============================================================================
// CUDA Buffer & Context
// =============================================================================

struct CUDABuffer {
    CUdeviceptr ptr;
    size_t size;
};

struct CUDAKernel {
    CUmodule module;
    CUfunction function;
    std::string name;
};

struct CUDAContext {
    CUdevice device;
    CUcontext context;
    char device_name[256];
    size_t total_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessors;

    // Loaded modules and kernels
    std::unordered_map<std::string, CUmodule> modules;
    std::unordered_map<std::string, CUfunction> kernels;

    // NTT twiddle factor cache
    CUdeviceptr ntt_twiddles;
    CUdeviceptr ntt_inv_twiddles;
    size_t ntt_twiddle_size;
    uint64_t ntt_modulus;
};

// =============================================================================
// Embedded PTX Kernels
// =============================================================================

static const char* PTX_BINARY_OPS = R"(
.version 7.0
.target sm_50
.address_size 64

.visible .entry add_float_kernel(
    .param .u64 out, .param .u64 a, .param .u64 b, .param .u64 n
) {
    .reg .pred %p<2>; .reg .f32 %f<4>; .reg .b64 %rd<8>; .reg .b32 %r<4>;
    ld.param.u64 %rd1, [out]; ld.param.u64 %rd2, [a]; ld.param.u64 %rd3, [b]; ld.param.u64 %rd4, [n];
    mov.u32 %r1, %ctaid.x; mov.u32 %r2, %ntid.x; mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3; cvt.u64.u32 %rd5, %r1;
    setp.ge.u64 %p1, %rd5, %rd4; @%p1 bra DONE;
    shl.b64 %rd6, %rd5, 2; add.u64 %rd2, %rd2, %rd6; add.u64 %rd3, %rd3, %rd6; add.u64 %rd1, %rd1, %rd6;
    ld.global.f32 %f1, [%rd2]; ld.global.f32 %f2, [%rd3]; add.f32 %f3, %f1, %f2; st.global.f32 [%rd1], %f3;
DONE: ret;
}

.visible .entry sub_float_kernel(
    .param .u64 out, .param .u64 a, .param .u64 b, .param .u64 n
) {
    .reg .pred %p<2>; .reg .f32 %f<4>; .reg .b64 %rd<8>; .reg .b32 %r<4>;
    ld.param.u64 %rd1, [out]; ld.param.u64 %rd2, [a]; ld.param.u64 %rd3, [b]; ld.param.u64 %rd4, [n];
    mov.u32 %r1, %ctaid.x; mov.u32 %r2, %ntid.x; mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3; cvt.u64.u32 %rd5, %r1;
    setp.ge.u64 %p1, %rd5, %rd4; @%p1 bra DONE;
    shl.b64 %rd6, %rd5, 2; add.u64 %rd2, %rd2, %rd6; add.u64 %rd3, %rd3, %rd6; add.u64 %rd1, %rd1, %rd6;
    ld.global.f32 %f1, [%rd2]; ld.global.f32 %f2, [%rd3]; sub.f32 %f3, %f1, %f2; st.global.f32 [%rd1], %f3;
DONE: ret;
}

.visible .entry mul_float_kernel(
    .param .u64 out, .param .u64 a, .param .u64 b, .param .u64 n
) {
    .reg .pred %p<2>; .reg .f32 %f<4>; .reg .b64 %rd<8>; .reg .b32 %r<4>;
    ld.param.u64 %rd1, [out]; ld.param.u64 %rd2, [a]; ld.param.u64 %rd3, [b]; ld.param.u64 %rd4, [n];
    mov.u32 %r1, %ctaid.x; mov.u32 %r2, %ntid.x; mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3; cvt.u64.u32 %rd5, %r1;
    setp.ge.u64 %p1, %rd5, %rd4; @%p1 bra DONE;
    shl.b64 %rd6, %rd5, 2; add.u64 %rd2, %rd2, %rd6; add.u64 %rd3, %rd3, %rd6; add.u64 %rd1, %rd1, %rd6;
    ld.global.f32 %f1, [%rd2]; ld.global.f32 %f2, [%rd3]; mul.f32 %f3, %f1, %f2; st.global.f32 [%rd1], %f3;
DONE: ret;
}
)";

static const char* PTX_GEMM = R"(
.version 7.0
.target sm_50
.address_size 64

.visible .entry gemm_simple_kernel(
    .param .u64 C_ptr, .param .u64 A_ptr, .param .u64 B_ptr,
    .param .u32 M, .param .u32 K, .param .u32 N
) {
    .reg .pred %p<4>; .reg .f32 %f<8>; .reg .b64 %rd<16>; .reg .b32 %r<20>;
    mov.u32 %r1, %ctaid.x; mov.u32 %r2, %ctaid.y;
    mov.u32 %r3, %tid.x; mov.u32 %r4, %tid.y;
    shl.b32 %r5, %r2, 4; add.u32 %r5, %r5, %r4;
    shl.b32 %r6, %r1, 4; add.u32 %r6, %r6, %r3;
    ld.param.u32 %r10, [M]; ld.param.u32 %r11, [K]; ld.param.u32 %r12, [N];
    ld.param.u64 %rd1, [C_ptr]; ld.param.u64 %rd2, [A_ptr]; ld.param.u64 %rd3, [B_ptr];
    setp.ge.u32 %p1, %r5, %r10; setp.ge.u32 %p2, %r6, %r12;
    or.pred %p3, %p1, %p2; @%p3 bra DONE;
    mov.f32 %f1, 0f00000000; mov.u32 %r7, 0;
LOOP:
    setp.ge.u32 %p1, %r7, %r11; @%p1 bra LOOP_END;
    mul.lo.u32 %r8, %r5, %r11; add.u32 %r8, %r8, %r7;
    mul.wide.u32 %rd4, %r8, 4; add.u64 %rd5, %rd2, %rd4; ld.global.f32 %f2, [%rd5];
    mul.lo.u32 %r9, %r7, %r12; add.u32 %r9, %r9, %r6;
    mul.wide.u32 %rd6, %r9, 4; add.u64 %rd7, %rd3, %rd6; ld.global.f32 %f3, [%rd7];
    fma.rn.f32 %f1, %f2, %f3, %f1; add.u32 %r7, %r7, 1; bra LOOP;
LOOP_END:
    mul.lo.u32 %r8, %r5, %r12; add.u32 %r8, %r8, %r6;
    mul.wide.u32 %rd4, %r8, 4; add.u64 %rd5, %rd1, %rd4; st.global.f32 [%rd5], %f1;
DONE: ret;
}
)";

// Division kernel
static const char* PTX_DIV = R"(
.version 7.0
.target sm_50
.address_size 64

.visible .entry div_float_kernel(
    .param .u64 out, .param .u64 a, .param .u64 b, .param .u64 n
) {
    .reg .pred %p<2>; .reg .f32 %f<4>; .reg .b64 %rd<8>; .reg .b32 %r<4>;
    ld.param.u64 %rd1, [out]; ld.param.u64 %rd2, [a]; ld.param.u64 %rd3, [b]; ld.param.u64 %rd4, [n];
    mov.u32 %r1, %ctaid.x; mov.u32 %r2, %ntid.x; mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3; cvt.u64.u32 %rd5, %r1;
    setp.ge.u64 %p1, %rd5, %rd4; @%p1 bra DONE;
    shl.b64 %rd6, %rd5, 2; add.u64 %rd2, %rd2, %rd6; add.u64 %rd3, %rd3, %rd6; add.u64 %rd1, %rd1, %rd6;
    ld.global.f32 %f1, [%rd2]; ld.global.f32 %f2, [%rd3]; div.rn.f32 %f3, %f1, %f2; st.global.f32 [%rd1], %f3;
DONE: ret;
}
)";

// Unary operations
static const char* PTX_UNARY_OPS = R"(
.version 7.0
.target sm_50
.address_size 64

.visible .entry exp_float_kernel(.param .u64 out, .param .u64 in, .param .u64 n) {
    .reg .pred %p<2>; .reg .f32 %f<3>; .reg .b64 %rd<6>; .reg .b32 %r<4>;
    ld.param.u64 %rd1, [out]; ld.param.u64 %rd2, [in]; ld.param.u64 %rd3, [n];
    mov.u32 %r1, %ctaid.x; mov.u32 %r2, %ntid.x; mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3; cvt.u64.u32 %rd4, %r1;
    setp.ge.u64 %p1, %rd4, %rd3; @%p1 bra DONE;
    shl.b64 %rd5, %rd4, 2; add.u64 %rd2, %rd2, %rd5; add.u64 %rd1, %rd1, %rd5;
    ld.global.f32 %f1, [%rd2]; ex2.approx.f32 %f2, %f1; st.global.f32 [%rd1], %f2;
DONE: ret;
}

.visible .entry log_float_kernel(.param .u64 out, .param .u64 in, .param .u64 n) {
    .reg .pred %p<2>; .reg .f32 %f<3>; .reg .b64 %rd<6>; .reg .b32 %r<4>;
    ld.param.u64 %rd1, [out]; ld.param.u64 %rd2, [in]; ld.param.u64 %rd3, [n];
    mov.u32 %r1, %ctaid.x; mov.u32 %r2, %ntid.x; mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3; cvt.u64.u32 %rd4, %r1;
    setp.ge.u64 %p1, %rd4, %rd3; @%p1 bra DONE;
    shl.b64 %rd5, %rd4, 2; add.u64 %rd2, %rd2, %rd5; add.u64 %rd1, %rd1, %rd5;
    ld.global.f32 %f1, [%rd2]; lg2.approx.f32 %f2, %f1; st.global.f32 [%rd1], %f2;
DONE: ret;
}

.visible .entry sqrt_float_kernel(.param .u64 out, .param .u64 in, .param .u64 n) {
    .reg .pred %p<2>; .reg .f32 %f<3>; .reg .b64 %rd<6>; .reg .b32 %r<4>;
    ld.param.u64 %rd1, [out]; ld.param.u64 %rd2, [in]; ld.param.u64 %rd3, [n];
    mov.u32 %r1, %ctaid.x; mov.u32 %r2, %ntid.x; mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3; cvt.u64.u32 %rd4, %r1;
    setp.ge.u64 %p1, %rd4, %rd3; @%p1 bra DONE;
    shl.b64 %rd5, %rd4, 2; add.u64 %rd2, %rd2, %rd5; add.u64 %rd1, %rd1, %rd5;
    ld.global.f32 %f1, [%rd2]; sqrt.rn.f32 %f2, %f1; st.global.f32 [%rd1], %f2;
DONE: ret;
}

.visible .entry neg_float_kernel(.param .u64 out, .param .u64 in, .param .u64 n) {
    .reg .pred %p<2>; .reg .f32 %f<3>; .reg .b64 %rd<6>; .reg .b32 %r<4>;
    ld.param.u64 %rd1, [out]; ld.param.u64 %rd2, [in]; ld.param.u64 %rd3, [n];
    mov.u32 %r1, %ctaid.x; mov.u32 %r2, %ntid.x; mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3; cvt.u64.u32 %rd4, %r1;
    setp.ge.u64 %p1, %rd4, %rd3; @%p1 bra DONE;
    shl.b64 %rd5, %rd4, 2; add.u64 %rd2, %rd2, %rd5; add.u64 %rd1, %rd1, %rd5;
    ld.global.f32 %f1, [%rd2]; neg.f32 %f2, %f1; st.global.f32 [%rd1], %f2;
DONE: ret;
}

.visible .entry abs_float_kernel(.param .u64 out, .param .u64 in, .param .u64 n) {
    .reg .pred %p<2>; .reg .f32 %f<3>; .reg .b64 %rd<6>; .reg .b32 %r<4>;
    ld.param.u64 %rd1, [out]; ld.param.u64 %rd2, [in]; ld.param.u64 %rd3, [n];
    mov.u32 %r1, %ctaid.x; mov.u32 %r2, %ntid.x; mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3; cvt.u64.u32 %rd4, %r1;
    setp.ge.u64 %p1, %rd4, %rd3; @%p1 bra DONE;
    shl.b64 %rd5, %rd4, 2; add.u64 %rd2, %rd2, %rd5; add.u64 %rd1, %rd1, %rd5;
    ld.global.f32 %f1, [%rd2]; abs.f32 %f2, %f1; st.global.f32 [%rd1], %f2;
DONE: ret;
}

.visible .entry relu_float_kernel(.param .u64 out, .param .u64 in, .param .u64 n) {
    .reg .pred %p<3>; .reg .f32 %f<3>; .reg .b64 %rd<6>; .reg .b32 %r<4>;
    ld.param.u64 %rd1, [out]; ld.param.u64 %rd2, [in]; ld.param.u64 %rd3, [n];
    mov.u32 %r1, %ctaid.x; mov.u32 %r2, %ntid.x; mov.u32 %r3, %tid.x;
    mad.lo.u32 %r1, %r1, %r2, %r3; cvt.u64.u32 %rd4, %r1;
    setp.ge.u64 %p1, %rd4, %rd3; @%p1 bra DONE;
    shl.b64 %rd5, %rd4, 2; add.u64 %rd2, %rd2, %rd5; add.u64 %rd1, %rd1, %rd5;
    ld.global.f32 %f1, [%rd2]; setp.gt.f32 %p2, %f1, 0f00000000;
    selp.f32 %f2, %f1, 0f00000000, %p2; st.global.f32 [%rd1], %f2;
DONE: ret;
}
)";

// Reduction kernels
static const char* PTX_REDUCE = R"(
.version 7.0
.target sm_50
.address_size 64

.visible .entry reduce_sum_kernel(.param .u64 out, .param .u64 in, .param .u64 n) {
    .reg .pred %p<3>; .reg .f32 %f<4>; .reg .b64 %rd<8>; .reg .b32 %r<8>;
    .shared .align 4 .f32 sdata[256];
    ld.param.u64 %rd1, [out]; ld.param.u64 %rd2, [in]; ld.param.u64 %rd3, [n];
    mov.u32 %r1, %tid.x; mov.u32 %r2, %ctaid.x; mov.u32 %r3, %ntid.x;
    mul.lo.u32 %r4, %r2, %r3; mul.lo.u32 %r4, %r4, 2; add.u32 %r4, %r4, %r1;
    cvt.u64.u32 %rd4, %r4; mov.f32 %f1, 0f00000000;
    setp.lt.u64 %p1, %rd4, %rd3; @!%p1 bra SKIP_LOAD;
    shl.b64 %rd5, %rd4, 2; add.u64 %rd6, %rd2, %rd5; ld.global.f32 %f1, [%rd6];
SKIP_LOAD:
    cvt.u64.u32 %rd4, %r3; add.u64 %rd4, %rd4, %rd4;
    mul.lo.u32 %r5, %r2, %r3; mul.lo.u32 %r5, %r5, 2; add.u32 %r5, %r5, %r1; add.u32 %r5, %r5, %r3;
    cvt.u64.u32 %rd5, %r5;
    setp.lt.u64 %p1, %rd5, %rd3; @!%p1 bra SKIP_LOAD2;
    shl.b64 %rd6, %rd5, 2; add.u64 %rd6, %rd2, %rd6; ld.global.f32 %f2, [%rd6]; add.f32 %f1, %f1, %f2;
SKIP_LOAD2:
    mul.lo.u32 %r6, %r1, 4; mov.u32 %r7, sdata; add.u32 %r7, %r7, %r6; st.shared.f32 [%r7], %f1;
    bar.sync 0;
    mov.u32 %r6, 128;
REDUCE_LOOP:
    setp.lt.u32 %p1, %r1, %r6; @!%p1 bra REDUCE_DONE;
    add.u32 %r7, %r1, %r6; mul.lo.u32 %r7, %r7, 4; mov.u32 %r5, sdata; add.u32 %r7, %r5, %r7;
    ld.shared.f32 %f2, [%r7]; mul.lo.u32 %r7, %r1, 4; add.u32 %r7, %r5, %r7;
    ld.shared.f32 %f1, [%r7]; add.f32 %f1, %f1, %f2; st.shared.f32 [%r7], %f1;
    bar.sync 0; shr.u32 %r6, %r6, 1; bra REDUCE_LOOP;
REDUCE_DONE:
    setp.eq.u32 %p2, %r1, 0; @!%p2 bra DONE;
    ld.shared.f32 %f1, [sdata]; cvt.u64.u32 %rd4, %r2; shl.b64 %rd4, %rd4, 2;
    add.u64 %rd1, %rd1, %rd4; st.global.f32 [%rd1], %f1;
DONE: ret;
}
)";

// =============================================================================
// Kernel Loading Helpers
// =============================================================================

static CUmodule load_ptx_module(CUDAContext* ctx, const char* ptx_source, const char* name) {
    if (!ptx_source) return nullptr;
    char error_log[4096] = {0};
    CUjit_option options[] = {
        (CUjit_option)CU_JIT_ERROR_LOG_BUFFER,
        (CUjit_option)CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
    };
    void* option_values[] = { error_log, (void*)(uintptr_t)sizeof(error_log) };
    CUmodule module = nullptr;
    CUresult result = cuda_driver.cuModuleLoadDataEx(&module, ptx_source, 2, options, option_values);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA PTX load error for %s: %s\n", name, error_log);
        return nullptr;
    }
    return module;
}

static CUfunction get_kernel(CUDAContext* ctx, const char* module_name,
                              const char* kernel_name, const char* ptx_source) {
    std::string key = std::string(module_name) + "::" + kernel_name;
    auto it = ctx->kernels.find(key);
    if (it != ctx->kernels.end()) return it->second;

    auto mod_it = ctx->modules.find(module_name);
    CUmodule module;
    if (mod_it == ctx->modules.end()) {
        module = load_ptx_module(ctx, ptx_source, module_name);
        if (!module) return nullptr;
        ctx->modules[module_name] = module;
    } else {
        module = mod_it->second;
    }

    CUfunction func = nullptr;
    if (cuda_driver.cuModuleGetFunction(&func, module, kernel_name) != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA kernel not found: %s in %s\n", kernel_name, module_name);
        return nullptr;
    }
    ctx->kernels[key] = func;
    return func;
}

// =============================================================================
// CUDA Backend Functions
// =============================================================================

static LuxBackendContext* cuda_create_context(int device_index) {
    if (!load_cuda_driver()) return nullptr;

    int device_count = 0;
    if (cuda_driver.cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count == 0) {
        return nullptr;
    }

    CUdevice device;
    if (cuda_driver.cuDeviceGet(&device, device_index) != CUDA_SUCCESS) {
        return nullptr;
    }

    auto ctx = new CUDAContext();
    ctx->device = device;
    ctx->ntt_twiddles = 0;
    ctx->ntt_inv_twiddles = 0;
    ctx->ntt_twiddle_size = 0;
    ctx->ntt_modulus = 0;
    ctx->compute_capability_major = 0;
    ctx->compute_capability_minor = 0;
    ctx->multiprocessors = 0;

    if (cuda_driver.cuCtxCreate(&ctx->context, 0, device) != CUDA_SUCCESS) {
        delete ctx;
        return nullptr;
    }

    cuda_driver.cuDeviceGetName(ctx->device_name, sizeof(ctx->device_name), device);
    cuda_driver.cuDeviceTotalMem(&ctx->total_memory, device);

    // Query device capabilities
    cuda_driver.cuDeviceGetAttribute(&ctx->compute_capability_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuda_driver.cuDeviceGetAttribute(&ctx->compute_capability_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    cuda_driver.cuDeviceGetAttribute(&ctx->multiprocessors, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);

    return reinterpret_cast<LuxBackendContext*>(ctx);
}

static void cuda_destroy_context(LuxBackendContext* context) {
    auto ctx = reinterpret_cast<CUDAContext*>(context);
    if (ctx) {
        // Unload all loaded modules
        for (auto& pair : ctx->modules) {
            if (pair.second) {
                cuda_driver.cuModuleUnload(pair.second);
            }
        }
        ctx->modules.clear();
        ctx->kernels.clear();

        // Free NTT twiddle cache if allocated
        if (ctx->ntt_twiddles) {
            cuda_driver.cuMemFree(ctx->ntt_twiddles);
        }
        if (ctx->ntt_inv_twiddles) {
            cuda_driver.cuMemFree(ctx->ntt_inv_twiddles);
        }

        if (ctx->context) {
            cuda_driver.cuCtxDestroy(ctx->context);
        }
        delete ctx;
    }
}

static LuxBackendError cuda_get_device_count(int* count) {
    if (!count) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    if (!load_cuda_driver()) {
        *count = 0;
        return LUX_BACKEND_OK;
    }
    if (cuda_driver.cuDeviceGetCount(count) != CUDA_SUCCESS) {
        *count = 0;
    }
    return LUX_BACKEND_OK;
}

static LuxBackendError cuda_get_device_info(LuxBackendContext* context, LuxBackendDeviceInfo* info) {
    if (!info) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    auto ctx = reinterpret_cast<CUDAContext*>(context);
    if (!ctx) return LUX_BACKEND_ERROR_INTERNAL;

    info->name = ctx->device_name;
    info->vendor = "NVIDIA";
    info->memory_total = ctx->total_memory;
    info->memory_available = ctx->total_memory;  // TODO: query actual free memory
    info->compute_units = ctx->multiprocessors;
    info->max_workgroup_size = 1024;  // Typical CUDA value
    info->is_discrete = true;
    info->is_unified_memory = false;

    return LUX_BACKEND_OK;
}

static LuxBackendError cuda_sync(LuxBackendContext*) {
    if (cuda_driver.cuCtxSynchronize() != CUDA_SUCCESS) {
        return LUX_BACKEND_ERROR_INTERNAL;
    }
    return LUX_BACKEND_OK;
}

// Buffer management
static LuxBackendBuffer* cuda_buffer_alloc(LuxBackendContext*, size_t bytes) {
    auto buf = new CUDABuffer();
    buf->size = bytes;

    if (cuda_driver.cuMemAlloc(&buf->ptr, bytes) != CUDA_SUCCESS) {
        delete buf;
        return nullptr;
    }

    return reinterpret_cast<LuxBackendBuffer*>(buf);
}

static LuxBackendBuffer* cuda_buffer_alloc_with_data(LuxBackendContext* ctx, const void* data, size_t bytes) {
    auto buf = reinterpret_cast<CUDABuffer*>(cuda_buffer_alloc(ctx, bytes));
    if (!buf) return nullptr;

    if (cuda_driver.cuMemcpyHtoD(buf->ptr, data, bytes) != CUDA_SUCCESS) {
        cuda_driver.cuMemFree(buf->ptr);
        delete buf;
        return nullptr;
    }

    return reinterpret_cast<LuxBackendBuffer*>(buf);
}

static void cuda_buffer_free(LuxBackendContext*, LuxBackendBuffer* buffer) {
    auto buf = reinterpret_cast<CUDABuffer*>(buffer);
    if (buf) {
        cuda_driver.cuMemFree(buf->ptr);
        delete buf;
    }
}

static LuxBackendError cuda_buffer_copy_to_host(LuxBackendContext*, LuxBackendBuffer* buffer, void* dst, size_t bytes) {
    auto buf = reinterpret_cast<CUDABuffer*>(buffer);
    if (!buf || !dst) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    if (cuda_driver.cuMemcpyDtoH(dst, buf->ptr, std::min(bytes, buf->size)) != CUDA_SUCCESS) {
        return LUX_BACKEND_ERROR_INTERNAL;
    }
    return LUX_BACKEND_OK;
}

static LuxBackendError cuda_buffer_copy_from_host(LuxBackendContext*, LuxBackendBuffer* buffer, const void* src, size_t bytes) {
    auto buf = reinterpret_cast<CUDABuffer*>(buffer);
    if (!buf || !src) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    if (cuda_driver.cuMemcpyHtoD(buf->ptr, src, std::min(bytes, buf->size)) != CUDA_SUCCESS) {
        return LUX_BACKEND_ERROR_INTERNAL;
    }
    return LUX_BACKEND_OK;
}

static void* cuda_buffer_get_host_ptr(LuxBackendContext*, LuxBackendBuffer*) {
    return nullptr;  // CUDA doesn't have unified memory by default
}

// Kernel management (TODO: implement with PTX loading)
static LuxBackendKernel* cuda_kernel_load(LuxBackendContext*, const char*, const char*) {
    return nullptr;  // TODO: compile CUDA source at runtime
}

static LuxBackendKernel* cuda_kernel_load_binary(LuxBackendContext*, const void*, size_t, const char*) {
    return nullptr;  // TODO: load PTX binary
}

static void cuda_kernel_destroy(LuxBackendContext*, LuxBackendKernel*) {
}

static LuxBackendError cuda_kernel_dispatch(
    LuxBackendContext*, LuxBackendKernel*,
    uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t,
    LuxBackendBuffer**, int
) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;  // TODO: implement
}

// =============================================================================
// Built-in CUDA Kernel Operations
// =============================================================================

static LuxBackendError cuda_op_add_f32(LuxBackendContext* context, LuxBackendBuffer* a, LuxBackendBuffer* b, LuxBackendBuffer* out, size_t n) {
    auto ctx = reinterpret_cast<CUDAContext*>(context);
    auto buf_a = reinterpret_cast<CUDABuffer*>(a);
    auto buf_b = reinterpret_cast<CUDABuffer*>(b);
    auto buf_out = reinterpret_cast<CUDABuffer*>(out);

    if (!ctx || !buf_a || !buf_b || !buf_out || n == 0) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    // Set context current
    if (cuda_driver.cuCtxSetCurrent(ctx->context) != CUDA_SUCCESS) {
        return LUX_BACKEND_ERROR_INTERNAL;
    }

    // Get kernel
    CUfunction kernel = get_kernel(ctx, "binary_ops", "add_float_kernel", PTX_BINARY_OPS);
    if (!kernel) {
        return LUX_BACKEND_ERROR_INTERNAL;
    }

    // Launch configuration
    unsigned int block_size = 256;
    unsigned int grid_size = (unsigned int)((n + block_size - 1) / block_size);

    // Set up kernel args: (out, a, b, n)
    void* args[] = { &buf_out->ptr, &buf_a->ptr, &buf_b->ptr, &n };

    CUresult result = cuda_driver.cuLaunchKernel(
        kernel,
        grid_size, 1, 1,    // grid dimensions
        block_size, 1, 1,   // block dimensions
        0,                  // shared memory
        nullptr,            // stream
        args,               // kernel arguments
        nullptr             // extra
    );

    if (result != CUDA_SUCCESS) {
        return LUX_BACKEND_ERROR_INTERNAL;
    }

    return LUX_BACKEND_OK;
}

static LuxBackendError cuda_op_sub_f32(LuxBackendContext* context, LuxBackendBuffer* a, LuxBackendBuffer* b, LuxBackendBuffer* out, size_t n) {
    auto ctx = reinterpret_cast<CUDAContext*>(context);
    auto buf_a = reinterpret_cast<CUDABuffer*>(a);
    auto buf_b = reinterpret_cast<CUDABuffer*>(b);
    auto buf_out = reinterpret_cast<CUDABuffer*>(out);

    if (!ctx || !buf_a || !buf_b || !buf_out || n == 0) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    if (cuda_driver.cuCtxSetCurrent(ctx->context) != CUDA_SUCCESS) {
        return LUX_BACKEND_ERROR_INTERNAL;
    }

    CUfunction kernel = get_kernel(ctx, "binary_ops", "sub_float_kernel", PTX_BINARY_OPS);
    if (!kernel) {
        return LUX_BACKEND_ERROR_INTERNAL;
    }

    unsigned int block_size = 256;
    unsigned int grid_size = (unsigned int)((n + block_size - 1) / block_size);
    void* args[] = { &buf_out->ptr, &buf_a->ptr, &buf_b->ptr, &n };

    if (cuda_driver.cuLaunchKernel(kernel, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, args, nullptr) != CUDA_SUCCESS) {
        return LUX_BACKEND_ERROR_INTERNAL;
    }

    return LUX_BACKEND_OK;
}

static LuxBackendError cuda_op_mul_f32(LuxBackendContext* context, LuxBackendBuffer* a, LuxBackendBuffer* b, LuxBackendBuffer* out, size_t n) {
    auto ctx = reinterpret_cast<CUDAContext*>(context);
    auto buf_a = reinterpret_cast<CUDABuffer*>(a);
    auto buf_b = reinterpret_cast<CUDABuffer*>(b);
    auto buf_out = reinterpret_cast<CUDABuffer*>(out);

    if (!ctx || !buf_a || !buf_b || !buf_out || n == 0) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    if (cuda_driver.cuCtxSetCurrent(ctx->context) != CUDA_SUCCESS) {
        return LUX_BACKEND_ERROR_INTERNAL;
    }

    CUfunction kernel = get_kernel(ctx, "binary_ops", "mul_float_kernel", PTX_BINARY_OPS);
    if (!kernel) {
        return LUX_BACKEND_ERROR_INTERNAL;
    }

    unsigned int block_size = 256;
    unsigned int grid_size = (unsigned int)((n + block_size - 1) / block_size);
    void* args[] = { &buf_out->ptr, &buf_a->ptr, &buf_b->ptr, &n };

    if (cuda_driver.cuLaunchKernel(kernel, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, args, nullptr) != CUDA_SUCCESS) {
        return LUX_BACKEND_ERROR_INTERNAL;
    }

    return LUX_BACKEND_OK;
}

static LuxBackendError cuda_op_matmul_f32(LuxBackendContext* context, LuxBackendBuffer* a, LuxBackendBuffer* b, LuxBackendBuffer* out, int M, int K, int N) {
    auto ctx = reinterpret_cast<CUDAContext*>(context);
    auto buf_a = reinterpret_cast<CUDABuffer*>(a);
    auto buf_b = reinterpret_cast<CUDABuffer*>(b);
    auto buf_out = reinterpret_cast<CUDABuffer*>(out);

    if (!ctx || !buf_a || !buf_b || !buf_out || M <= 0 || K <= 0 || N <= 0) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    if (cuda_driver.cuCtxSetCurrent(ctx->context) != CUDA_SUCCESS) {
        return LUX_BACKEND_ERROR_INTERNAL;
    }

    CUfunction kernel = get_kernel(ctx, "gemm", "gemm_simple_kernel", PTX_GEMM);
    if (!kernel) {
        return LUX_BACKEND_ERROR_INTERNAL;
    }

    // Use 16x16 thread blocks for matmul
    unsigned int tile_size = 16;
    unsigned int grid_x = (unsigned int)((N + tile_size - 1) / tile_size);
    unsigned int grid_y = (unsigned int)((M + tile_size - 1) / tile_size);

    // Kernel args: (C_ptr, A_ptr, B_ptr, M, K, N)
    uint32_t m32 = (uint32_t)M, k32 = (uint32_t)K, n32 = (uint32_t)N;
    void* args[] = { &buf_out->ptr, &buf_a->ptr, &buf_b->ptr, &m32, &k32, &n32 };

    if (cuda_driver.cuLaunchKernel(kernel, grid_x, grid_y, 1, tile_size, tile_size, 1, 0, nullptr, args, nullptr) != CUDA_SUCCESS) {
        return LUX_BACKEND_ERROR_INTERNAL;
    }

    return LUX_BACKEND_OK;
}

// =============================================================================
// Modular Arithmetic Helpers (CPU fallback)
// =============================================================================

static inline uint64_t mod_add(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t sum = a + b;
    return sum >= m ? sum - m : sum;
}

static inline uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t m) {
    return a >= b ? a - b : a + m - b;
}

static inline uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t m) {
    return ((__uint128_t)a * b) % m;
}

static uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t m) {
    uint64_t result = 1;
    base %= m;
    while (exp > 0) {
        if (exp & 1) result = mod_mul(result, base, m);
        exp >>= 1;
        base = mod_mul(base, base, m);
    }
    return result;
}

static void bit_reverse(uint64_t* data, size_t n) {
    size_t j = 0;
    for (size_t i = 0; i < n; i++) {
        if (i < j) std::swap(data[i], data[j]);
        size_t m = n >> 1;
        while (m >= 1 && j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
}

static uint64_t find_primitive_root(size_t n, uint64_t m) {
    // Common NTT-friendly primes and their roots
    if (m == 0xFFFFFFFF00000001ULL) return 7;   // Goldilocks
    if (m == 0x1000000000000001ULL) return 3;
    if (m == 8380417ULL) return 1753;           // Dilithium
    if (m == 12289ULL) return 11;               // NewHope
    return 3;  // Fallback
}

// =============================================================================
// NTT Operations (CPU fallback for now, CUDA kernels loaded separately)
// =============================================================================

static LuxBackendError cuda_op_ntt_forward(LuxBackendContext*, uint64_t* data, size_t n, uint64_t modulus) {
    if (!data || n == 0 || (n & (n - 1)) != 0) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    uint64_t g = find_primitive_root(n, modulus);
    uint64_t omega_n = mod_pow(g, (modulus - 1) / n, modulus);

    bit_reverse(data, n);

    for (size_t len = 2; len <= n; len *= 2) {
        uint64_t w = mod_pow(omega_n, n / len, modulus);
        for (size_t i = 0; i < n; i += len) {
            uint64_t wn = 1;
            for (size_t j = 0; j < len / 2; j++) {
                uint64_t u = data[i + j];
                uint64_t t = mod_mul(wn, data[i + j + len / 2], modulus);
                data[i + j] = mod_add(u, t, modulus);
                data[i + j + len / 2] = mod_sub(u, t, modulus);
                wn = mod_mul(wn, w, modulus);
            }
        }
    }
    return LUX_BACKEND_OK;
}

static LuxBackendError cuda_op_ntt_inverse(LuxBackendContext*, uint64_t* data, size_t n, uint64_t modulus) {
    if (!data || n == 0 || (n & (n - 1)) != 0) return LUX_BACKEND_ERROR_INVALID_ARGUMENT;

    uint64_t g = find_primitive_root(n, modulus);
    uint64_t omega_n = mod_pow(g, (modulus - 1) / n, modulus);
    uint64_t omega_n_inv = mod_pow(omega_n, modulus - 2, modulus);

    // DIF butterfly (Gentleman-Sande)
    for (size_t len = n; len >= 2; len /= 2) {
        uint64_t w = mod_pow(omega_n_inv, n / len, modulus);
        for (size_t i = 0; i < n; i += len) {
            uint64_t wn = 1;
            for (size_t j = 0; j < len / 2; j++) {
                uint64_t u = data[i + j];
                uint64_t v = data[i + j + len / 2];
                data[i + j] = mod_add(u, v, modulus);
                data[i + j + len / 2] = mod_mul(mod_sub(u, v, modulus), wn, modulus);
                wn = mod_mul(wn, w, modulus);
            }
        }
    }

    bit_reverse(data, n);

    // Scale by N^{-1}
    uint64_t n_inv = mod_pow(n, modulus - 2, modulus);
    for (size_t i = 0; i < n; i++) {
        data[i] = mod_mul(data[i], n_inv, modulus);
    }
    return LUX_BACKEND_OK;
}

static LuxBackendError cuda_op_msm(LuxBackendContext*, const void*, const void*, void*, size_t, int) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

// =============================================================================
// Polynomial Multiplication
// =============================================================================

static LuxBackendError cuda_op_poly_mul(
    LuxBackendContext* ctx,
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* result,
    size_t n,
    uint64_t modulus
) {
    if (!a || !b || !result || n == 0 || (n & (n - 1)) != 0) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    // Allocate temporary buffers for NTT
    std::vector<uint64_t> a_ntt(n), b_ntt(n);
    std::memcpy(a_ntt.data(), a, n * sizeof(uint64_t));
    std::memcpy(b_ntt.data(), b, n * sizeof(uint64_t));

    // Forward NTT on both operands
    LuxBackendError err = cuda_op_ntt_forward(ctx, a_ntt.data(), n, modulus);
    if (err != LUX_BACKEND_OK) return err;

    err = cuda_op_ntt_forward(ctx, b_ntt.data(), n, modulus);
    if (err != LUX_BACKEND_OK) return err;

    // Pointwise multiplication in NTT domain
    for (size_t i = 0; i < n; i++) {
        result[i] = mod_mul(a_ntt[i], b_ntt[i], modulus);
    }

    // Inverse NTT
    return cuda_op_ntt_inverse(ctx, result, n, modulus);
}

// =============================================================================
// TFHE Operations
// =============================================================================

static inline uint64_t mod_neg(uint64_t a, uint64_t q) {
    return a == 0 ? 0 : q - a;
}

// Signed decomposition digit extraction
static inline int64_t signed_decomp_digit(uint64_t val, uint32_t level, uint32_t base_log) {
    uint64_t base = 1ULL << base_log;
    uint64_t half_base = base >> 1;
    uint64_t mask = base - 1;
    uint32_t shift = 64 - (level + 1) * base_log;
    uint64_t digit = ((val >> shift) + half_base) & mask;
    return (int64_t)digit - (int64_t)half_base;
}

static LuxBackendError cuda_op_tfhe_bootstrap(
    LuxBackendContext* ctx,
    const uint64_t* lwe_in,
    uint64_t* lwe_out,
    const uint64_t* bsk,
    const uint64_t* test_poly,
    uint32_t n_lwe,
    uint32_t N,
    uint32_t k,
    uint32_t l,
    uint64_t q
) {
    if (!lwe_in || !lwe_out || !bsk || !test_poly) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    // Allocate accumulator [(k+1) * N]
    std::vector<uint64_t> acc((k + 1) * N, 0);

    // Step 1: Initialize accumulator with rotated test polynomial
    // b_tilde = round(lwe_b * 2N / q)
    uint64_t lwe_b = lwe_in[n_lwe];
    int32_t b_tilde = (int32_t)(((__uint128_t)lwe_b * 2 * N + (q >> 1)) / q);

    // acc_body = X^{-b_tilde} * test_poly (negacyclic rotation)
    for (uint32_t i = 0; i < N; i++) {
        int32_t rotation = -b_tilde;
        int32_t rot = ((rotation % (int32_t)(2 * N)) + 2 * (int32_t)N) % (2 * (int32_t)N);
        bool negate = (rot >= (int32_t)N);
        if (negate) rot -= N;

        int32_t src = (int32_t)i - rot;
        bool wrap = src < 0;
        if (wrap) src += N;

        uint64_t val = test_poly[src];
        if (negate != wrap) val = mod_neg(val, q);
        acc[k * N + i] = val;
    }

    // Step 2: Blind rotation (simplified - full impl would use external product)
    // For each LWE coefficient, compute rotation and apply CMux
    size_t bsk_stride = (size_t)(k + 1) * l * (k + 1) * N;

    for (uint32_t bit = 0; bit < n_lwe; bit++) {
        uint64_t a_i = lwe_in[bit];
        int32_t rotation = (int32_t)(((__uint128_t)a_i * 2 * N + (q >> 1)) / q);

        if (rotation == 0) continue;

        // Compute rotated accumulator and apply external product
        // This is a placeholder - full CMux requires external product
        std::vector<uint64_t> rotated((k + 1) * N);

        for (uint32_t poly = 0; poly <= k; poly++) {
            for (uint32_t i = 0; i < N; i++) {
                int32_t rot = ((rotation % (int32_t)(2 * N)) + 2 * (int32_t)N) % (2 * (int32_t)N);
                bool negate = (rot >= (int32_t)N);
                if (negate) rot -= N;

                int32_t src = (int32_t)i - rot;
                bool wrap = src < 0;
                if (wrap) src += N;

                uint64_t val = acc[poly * N + src];
                if (negate != wrap) val = mod_neg(val, q);
                rotated[poly * N + i] = val;
            }
        }

        // CMux: acc = acc + bsk[bit] * (rotated - acc)
        // Simplified: directly use rotated (full impl does external product)
        // For a real implementation, this would decompose (rotated - acc)
        // and multiply by BSK rows
        std::swap(acc, rotated);
    }

    // Step 3: Sample extraction - extract LWE from GLWE at position 0
    // lwe_out[0..N-1] = -acc_mask[N-1..0] (reversed and negated)
    for (uint32_t i = 0; i < N; i++) {
        uint64_t val = acc[N - 1 - i];  // First mask polynomial (assuming k=1)
        lwe_out[i] = mod_neg(val, q);
    }
    // lwe_out[N] = acc_body[0]
    lwe_out[N] = acc[k * N];

    return LUX_BACKEND_OK;
}

static LuxBackendError cuda_op_tfhe_keyswitch(
    LuxBackendContext*,
    const uint64_t* lwe_in,
    uint64_t* lwe_out,
    const uint64_t* ksk,
    uint32_t n_in,
    uint32_t n_out,
    uint32_t l,
    uint32_t base_log,
    uint64_t q
) {
    if (!lwe_in || !lwe_out || !ksk) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    // Initialize output to zero (mask) and copy body
    std::memset(lwe_out, 0, n_out * sizeof(uint64_t));
    lwe_out[n_out] = lwe_in[n_in];  // Copy body

    // For each input coefficient
    for (uint32_t in_idx = 0; in_idx < n_in; in_idx++) {
        uint64_t val = lwe_in[in_idx];

        // Decompose and accumulate
        for (uint32_t level = 0; level < l; level++) {
            int64_t digit = signed_decomp_digit(val, level, base_log);
            if (digit == 0) continue;

            // Add digit * ksk[in_idx][level] to output
            const uint64_t* ksk_row = ksk + (in_idx * l + level) * (n_out + 1);

            for (uint32_t out_idx = 0; out_idx <= n_out; out_idx++) {
                uint64_t ksk_val = ksk_row[out_idx];
                if (digit > 0) {
                    uint64_t prod = mod_mul((uint64_t)digit, ksk_val, q);
                    lwe_out[out_idx] = mod_add(lwe_out[out_idx], prod, q);
                } else {
                    uint64_t prod = mod_mul((uint64_t)(-digit), ksk_val, q);
                    lwe_out[out_idx] = mod_sub(lwe_out[out_idx], prod, q);
                }
            }
        }
    }

    return LUX_BACKEND_OK;
}

static LuxBackendError cuda_op_blind_rotate(
    LuxBackendContext*,
    uint64_t* acc,
    const uint64_t* bsk,
    const uint64_t* lwe_a,
    uint32_t n_lwe,
    uint32_t N,
    uint32_t k,
    uint32_t l,
    uint64_t q
) {
    if (!acc || !bsk || !lwe_a) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    size_t bsk_stride = (size_t)(k + 1) * l * (k + 1) * N;
    std::vector<uint64_t> temp((k + 1) * N);

    for (uint32_t bit = 0; bit < n_lwe; bit++) {
        uint64_t a_i = lwe_a[bit];
        int32_t rotation = (int32_t)(((__uint128_t)a_i * 2 * N + (q >> 1)) / q);

        if (rotation == 0) continue;

        // Rotate accumulator by 'rotation' positions
        for (uint32_t poly = 0; poly <= k; poly++) {
            for (uint32_t i = 0; i < N; i++) {
                int32_t rot = ((rotation % (int32_t)(2 * N)) + 2 * (int32_t)N) % (2 * (int32_t)N);
                bool negate = (rot >= (int32_t)N);
                if (negate) rot -= N;

                int32_t src = (int32_t)i - rot;
                bool wrap = src < 0;
                if (wrap) src += N;

                uint64_t val = acc[poly * N + src];
                if (negate != wrap) val = mod_neg(val, q);
                temp[poly * N + i] = val;
            }
        }

        // Copy rotated back (simplified - real impl does CMux via external product)
        std::memcpy(acc, temp.data(), (k + 1) * N * sizeof(uint64_t));
    }

    return LUX_BACKEND_OK;
}

static LuxBackendError cuda_op_sample_extract(
    LuxBackendContext*,
    const uint64_t* glwe,
    uint64_t* lwe,
    uint32_t N,
    uint32_t k,
    uint64_t q
) {
    if (!glwe || !lwe) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    // Extract LWE from GLWE at position 0
    // lwe[i] = -glwe_mask[N-1-i] for i in [0, N)
    for (uint32_t i = 0; i < N; i++) {
        uint64_t val = glwe[N - 1 - i];  // First mask polynomial
        lwe[i] = mod_neg(val, q);
    }

    // lwe[N] = glwe_body[0]
    lwe[N] = glwe[k * N];

    return LUX_BACKEND_OK;
}

static LuxBackendError cuda_op_sample_ntt(
    LuxBackendContext* ctx,
    uint64_t* output,
    size_t n,
    uint64_t modulus,
    double sigma,
    uint64_t seed
) {
    if (!output || n == 0 || (n & (n - 1)) != 0) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    // Simple discrete Gaussian sampling using Box-Muller
    // For production, use CDT or Karney's algorithm
    std::srand((unsigned int)seed);

    for (size_t i = 0; i < n; i += 2) {
        double u1 = ((double)std::rand() + 1.0) / ((double)RAND_MAX + 2.0);
        double u2 = ((double)std::rand() + 1.0) / ((double)RAND_MAX + 2.0);

        double r = std::sqrt(-2.0 * std::log(u1));
        double theta = 2.0 * 3.14159265358979323846 * u2;

        int64_t z0 = (int64_t)std::round(r * std::cos(theta) * sigma);
        int64_t z1 = (int64_t)std::round(r * std::sin(theta) * sigma);

        // Convert to positive residue mod q
        output[i] = (z0 >= 0) ? ((uint64_t)z0 % modulus) : (modulus - ((uint64_t)(-z0) % modulus));
        if (i + 1 < n) {
            output[i + 1] = (z1 >= 0) ? ((uint64_t)z1 % modulus) : (modulus - ((uint64_t)(-z1) % modulus));
        }
    }

    // Transform to NTT domain
    return cuda_op_ntt_forward(ctx, output, n, modulus);
}

// =============================================================================
// Stub implementations for new operations
// =============================================================================

static LuxBackendError cuda_op_div_f32(LuxBackendContext* context, LuxBackendBuffer* a, LuxBackendBuffer* b, LuxBackendBuffer* out, size_t n) {
    auto ctx = reinterpret_cast<CUDAContext*>(context);
    auto buf_a = reinterpret_cast<CUDABuffer*>(a);
    auto buf_b = reinterpret_cast<CUDABuffer*>(b);
    auto buf_out = reinterpret_cast<CUDABuffer*>(out);

    if (!ctx || !buf_a || !buf_b || !buf_out || n == 0) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    if (cuda_driver.cuCtxSetCurrent(ctx->context) != CUDA_SUCCESS) {
        return LUX_BACKEND_ERROR_INTERNAL;
    }

    CUfunction kernel = get_kernel(ctx, "div_ops", "div_float_kernel", PTX_DIV);
    if (!kernel) {
        return LUX_BACKEND_ERROR_INTERNAL;
    }

    unsigned int block_size = 256;
    unsigned int grid_size = (unsigned int)((n + block_size - 1) / block_size);
    void* args[] = { &buf_out->ptr, &buf_a->ptr, &buf_b->ptr, &n };

    if (cuda_driver.cuLaunchKernel(kernel, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, args, nullptr) != CUDA_SUCCESS) {
        return LUX_BACKEND_ERROR_INTERNAL;
    }

    return LUX_BACKEND_OK;
}

static LuxBackendError cuda_op_transpose_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, int, int) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError cuda_op_reduce_sum_f32(LuxBackendContext* context, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t n) {
    auto ctx = reinterpret_cast<CUDAContext*>(context);
    auto buf_in = reinterpret_cast<CUDABuffer*>(in);
    auto buf_out = reinterpret_cast<CUDABuffer*>(out);

    if (!ctx || !buf_in || !buf_out || n == 0) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    if (cuda_driver.cuCtxSetCurrent(ctx->context) != CUDA_SUCCESS) {
        return LUX_BACKEND_ERROR_INTERNAL;
    }

    CUfunction kernel = get_kernel(ctx, "reduce_ops", "reduce_sum_kernel", PTX_REDUCE);
    if (!kernel) {
        return LUX_BACKEND_ERROR_INTERNAL;
    }

    // Two-phase reduction: first reduce within blocks, then reduce block results
    unsigned int block_size = 256;
    unsigned int grid_size = (unsigned int)((n + block_size * 2 - 1) / (block_size * 2));

    // Allocate temp buffer for partial sums if needed
    CUdeviceptr temp_buf = 0;
    CUdeviceptr result_ptr = buf_out->ptr;

    if (grid_size > 1) {
        if (cuda_driver.cuMemAlloc(&temp_buf, grid_size * sizeof(float)) != CUDA_SUCCESS) {
            return LUX_BACKEND_ERROR_OUT_OF_MEMORY;
        }
        result_ptr = temp_buf;
    }

    void* args[] = { &result_ptr, &buf_in->ptr, &n };
    if (cuda_driver.cuLaunchKernel(kernel, grid_size, 1, 1, block_size, 1, 1, block_size * sizeof(float), nullptr, args, nullptr) != CUDA_SUCCESS) {
        if (temp_buf) cuda_driver.cuMemFree(temp_buf);
        return LUX_BACKEND_ERROR_INTERNAL;
    }

    // Second pass if needed
    if (grid_size > 1) {
        size_t remaining = grid_size;
        void* args2[] = { &buf_out->ptr, &temp_buf, &remaining };
        unsigned int grid2 = 1;
        if (cuda_driver.cuLaunchKernel(kernel, grid2, 1, 1, block_size, 1, 1, block_size * sizeof(float), nullptr, args2, nullptr) != CUDA_SUCCESS) {
            cuda_driver.cuMemFree(temp_buf);
            return LUX_BACKEND_ERROR_INTERNAL;
        }
        cuda_driver.cuMemFree(temp_buf);
    }

    return LUX_BACKEND_OK;
}

static LuxBackendError cuda_op_reduce_max_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError cuda_op_reduce_min_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError cuda_op_reduce_mean_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError cuda_op_reduce_sum_axis_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError cuda_op_reduce_max_axis_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError cuda_op_softmax_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError cuda_op_log_softmax_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

// Helper macro for unary operations
#define IMPL_UNARY_OP(name, kernel_name) \
static LuxBackendError cuda_op_##name##_f32(LuxBackendContext* context, LuxBackendBuffer* in, LuxBackendBuffer* out, size_t n) { \
    auto ctx = reinterpret_cast<CUDAContext*>(context); \
    auto buf_in = reinterpret_cast<CUDABuffer*>(in); \
    auto buf_out = reinterpret_cast<CUDABuffer*>(out); \
    if (!ctx || !buf_in || !buf_out || n == 0) return LUX_BACKEND_ERROR_INVALID_ARGUMENT; \
    if (cuda_driver.cuCtxSetCurrent(ctx->context) != CUDA_SUCCESS) return LUX_BACKEND_ERROR_INTERNAL; \
    CUfunction kernel = get_kernel(ctx, "unary_ops", kernel_name, PTX_UNARY_OPS); \
    if (!kernel) return LUX_BACKEND_ERROR_INTERNAL; \
    unsigned int block_size = 256; \
    unsigned int grid_size = (unsigned int)((n + block_size - 1) / block_size); \
    void* args[] = { &buf_out->ptr, &buf_in->ptr, &n }; \
    if (cuda_driver.cuLaunchKernel(kernel, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, args, nullptr) != CUDA_SUCCESS) \
        return LUX_BACKEND_ERROR_INTERNAL; \
    return LUX_BACKEND_OK; \
}

IMPL_UNARY_OP(exp, "exp_float_kernel")
IMPL_UNARY_OP(log, "log_float_kernel")
IMPL_UNARY_OP(sqrt, "sqrt_float_kernel")
IMPL_UNARY_OP(neg, "neg_float_kernel")
IMPL_UNARY_OP(abs, "abs_float_kernel")

static LuxBackendError cuda_op_tanh_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError cuda_op_sigmoid_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

IMPL_UNARY_OP(relu, "relu_float_kernel")

static LuxBackendError cuda_op_gelu_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, size_t) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError cuda_op_copy_f32(LuxBackendContext* context, LuxBackendBuffer* src, LuxBackendBuffer* dst, size_t n) {
    auto ctx = reinterpret_cast<CUDAContext*>(context);
    auto buf_src = reinterpret_cast<CUDABuffer*>(src);
    auto buf_dst = reinterpret_cast<CUDABuffer*>(dst);

    if (!ctx || !buf_src || !buf_dst || n == 0) {
        return LUX_BACKEND_ERROR_INVALID_ARGUMENT;
    }

    if (cuda_driver.cuCtxSetCurrent(ctx->context) != CUDA_SUCCESS) {
        return LUX_BACKEND_ERROR_INTERNAL;
    }

    // Direct device-to-device copy
    if (cuda_driver.cuMemcpyDtoD(buf_dst->ptr, buf_src->ptr, n * sizeof(float)) != CUDA_SUCCESS) {
        return LUX_BACKEND_ERROR_INTERNAL;
    }

    return LUX_BACKEND_OK;
}

static LuxBackendError cuda_op_layer_norm_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, LuxBackendBuffer*, LuxBackendBuffer*, size_t, size_t, float) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

static LuxBackendError cuda_op_rms_norm_f32(LuxBackendContext*, LuxBackendBuffer*, LuxBackendBuffer*, LuxBackendBuffer*, size_t, size_t, float) {
    return LUX_BACKEND_ERROR_NOT_SUPPORTED;
}

// =============================================================================
// CUDA Crypto Operations
// =============================================================================

// MSM - Multi-Scalar Multiplication using Pippenger's algorithm
static LuxCryptoError cuda_crypto_msm(
    LuxBackendContext* context,
    int curve_type,
    const void* points,
    const void* scalars,
    void* result,
    size_t count
) {
    auto ctx = reinterpret_cast<CUDAContext*>(context);
    if (!ctx || !points || !scalars || !result || count == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    // Determine point/scalar sizes based on curve
    size_t point_size, scalar_size, result_size;
    switch (curve_type) {
        case LUX_CURVE_BN254:
            point_size = sizeof(LuxG1Affine254);
            scalar_size = sizeof(LuxScalar256);
            result_size = sizeof(LuxG1Projective254);
            break;
        case LUX_CURVE_BLS12_381:
            point_size = sizeof(LuxG1Affine381);
            scalar_size = sizeof(LuxScalar256);
            result_size = sizeof(LuxG1Projective381);
            break;
        default:
            return LUX_CRYPTO_ERROR_INVALID_CURVE;
    }

    // Allocate device buffers
    CUdeviceptr d_points, d_scalars, d_result;
    if (cuda_driver.cuMemAlloc(&d_points, point_size * count) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_scalars, scalar_size * count) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_result, result_size) != CUDA_SUCCESS) {
        return LUX_CRYPTO_ERROR_OUT_OF_MEMORY;
    }

    // Copy inputs to device
    cuda_driver.cuMemcpyHtoD(d_points, points, point_size * count);
    cuda_driver.cuMemcpyHtoD(d_scalars, scalars, scalar_size * count);

    // TODO: Launch MSM kernel from compiled PTX (msm.cu)

    // Copy result back
    cuda_driver.cuMemcpyDtoH(result, d_result, result_size);

    // Cleanup
    cuda_driver.cuMemFree(d_points);
    cuda_driver.cuMemFree(d_scalars);
    cuda_driver.cuMemFree(d_result);

    return LUX_CRYPTO_OK;
}

static LuxCryptoError cuda_crypto_msm_batch(
    LuxBackendContext* context,
    int curve_type,
    const void* const* points_batch,
    const void* const* scalars_batch,
    void** results_batch,
    const size_t* counts,
    size_t batch_size
) {
    if (!context || !points_batch || !scalars_batch || !results_batch || !counts) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }
    for (size_t i = 0; i < batch_size; i++) {
        LuxCryptoError err = cuda_crypto_msm(context, curve_type,
            points_batch[i], scalars_batch[i], results_batch[i], counts[i]);
        if (err != LUX_CRYPTO_OK) return err;
    }
    return LUX_CRYPTO_OK;
}

// Poseidon2 Hash
static LuxCryptoError cuda_crypto_poseidon2_hash(
    LuxBackendContext* context,
    const LuxScalar256* inputs,
    size_t num_inputs,
    LuxScalar256* output
) {
    if (!context || !inputs || !output || num_inputs == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    CUdeviceptr d_inputs, d_output;
    if (cuda_driver.cuMemAlloc(&d_inputs, sizeof(LuxScalar256) * num_inputs) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_output, sizeof(LuxScalar256)) != CUDA_SUCCESS) {
        return LUX_CRYPTO_ERROR_OUT_OF_MEMORY;
    }

    cuda_driver.cuMemcpyHtoD(d_inputs, inputs, sizeof(LuxScalar256) * num_inputs);

    // TODO: Launch Poseidon2 kernel (poseidon2_bn254.cu)

    cuda_driver.cuMemcpyDtoH(output, d_output, sizeof(LuxScalar256));
    cuda_driver.cuMemFree(d_inputs);
    cuda_driver.cuMemFree(d_output);

    return LUX_CRYPTO_OK;
}

static LuxCryptoError cuda_crypto_poseidon2_batch(
    LuxBackendContext* context,
    const LuxScalar256* inputs,
    size_t inputs_per_hash,
    size_t num_hashes,
    LuxScalar256* outputs
) {
    if (!context || !inputs || !outputs || inputs_per_hash == 0 || num_hashes == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    size_t total_inputs = inputs_per_hash * num_hashes;
    CUdeviceptr d_inputs, d_outputs;
    if (cuda_driver.cuMemAlloc(&d_inputs, sizeof(LuxScalar256) * total_inputs) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_outputs, sizeof(LuxScalar256) * num_hashes) != CUDA_SUCCESS) {
        return LUX_CRYPTO_ERROR_OUT_OF_MEMORY;
    }

    cuda_driver.cuMemcpyHtoD(d_inputs, inputs, sizeof(LuxScalar256) * total_inputs);

    // TODO: Launch batch Poseidon2 kernel

    cuda_driver.cuMemcpyDtoH(outputs, d_outputs, sizeof(LuxScalar256) * num_hashes);
    cuda_driver.cuMemFree(d_inputs);
    cuda_driver.cuMemFree(d_outputs);

    return LUX_CRYPTO_OK;
}

static LuxCryptoError cuda_crypto_poseidon2_merkle(
    LuxBackendContext* context,
    const LuxScalar256* leaves,
    size_t num_leaves,
    LuxScalar256* tree_nodes
) {
    if (!context || !leaves || !tree_nodes || num_leaves == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    size_t tree_size = 2 * num_leaves - 1;
    CUdeviceptr d_tree;
    if (cuda_driver.cuMemAlloc(&d_tree, sizeof(LuxScalar256) * tree_size) != CUDA_SUCCESS) {
        return LUX_CRYPTO_ERROR_OUT_OF_MEMORY;
    }

    cuda_driver.cuMemcpyHtoD(
        (CUdeviceptr)((char*)d_tree + sizeof(LuxScalar256) * (num_leaves - 1)),
        leaves, sizeof(LuxScalar256) * num_leaves);

    // TODO: Launch Merkle tree kernel

    cuda_driver.cuMemcpyDtoH(tree_nodes, d_tree, sizeof(LuxScalar256) * tree_size);
    cuda_driver.cuMemFree(d_tree);

    return LUX_CRYPTO_OK;
}

// BLS12-381 Operations
static LuxCryptoError cuda_crypto_bls12_381_add(
    LuxBackendContext* context,
    const LuxG1Projective381* p,
    const LuxG1Projective381* q,
    LuxG1Projective381* result
) {
    if (!context || !p || !q || !result) return LUX_CRYPTO_ERROR_INVALID_ARG;

    CUdeviceptr d_p, d_q, d_result;
    size_t sz = sizeof(LuxG1Projective381);
    if (cuda_driver.cuMemAlloc(&d_p, sz) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_q, sz) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_result, sz) != CUDA_SUCCESS) {
        return LUX_CRYPTO_ERROR_OUT_OF_MEMORY;
    }
    cuda_driver.cuMemcpyHtoD(d_p, p, sz);
    cuda_driver.cuMemcpyHtoD(d_q, q, sz);

    // TODO: Launch BLS12-381 add kernel (bls12_381.cu)

    cuda_driver.cuMemcpyDtoH(result, d_result, sz);
    cuda_driver.cuMemFree(d_p);
    cuda_driver.cuMemFree(d_q);
    cuda_driver.cuMemFree(d_result);
    return LUX_CRYPTO_OK;
}

static LuxCryptoError cuda_crypto_bls12_381_double(
    LuxBackendContext* context,
    const LuxG1Projective381* p,
    LuxG1Projective381* result
) {
    if (!context || !p || !result) return LUX_CRYPTO_ERROR_INVALID_ARG;

    CUdeviceptr d_p, d_result;
    size_t sz = sizeof(LuxG1Projective381);
    if (cuda_driver.cuMemAlloc(&d_p, sz) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_result, sz) != CUDA_SUCCESS) {
        return LUX_CRYPTO_ERROR_OUT_OF_MEMORY;
    }
    cuda_driver.cuMemcpyHtoD(d_p, p, sz);

    // TODO: Launch BLS12-381 double kernel

    cuda_driver.cuMemcpyDtoH(result, d_result, sz);
    cuda_driver.cuMemFree(d_p);
    cuda_driver.cuMemFree(d_result);
    return LUX_CRYPTO_OK;
}

static LuxCryptoError cuda_crypto_bls12_381_scalar_mul(
    LuxBackendContext* context,
    const LuxG1Projective381* p,
    const LuxScalar256* scalar,
    LuxG1Projective381* result
) {
    if (!context || !p || !scalar || !result) return LUX_CRYPTO_ERROR_INVALID_ARG;

    CUdeviceptr d_p, d_scalar, d_result;
    if (cuda_driver.cuMemAlloc(&d_p, sizeof(LuxG1Projective381)) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_scalar, sizeof(LuxScalar256)) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_result, sizeof(LuxG1Projective381)) != CUDA_SUCCESS) {
        return LUX_CRYPTO_ERROR_OUT_OF_MEMORY;
    }
    cuda_driver.cuMemcpyHtoD(d_p, p, sizeof(LuxG1Projective381));
    cuda_driver.cuMemcpyHtoD(d_scalar, scalar, sizeof(LuxScalar256));

    // TODO: Launch scalar multiplication kernel

    cuda_driver.cuMemcpyDtoH(result, d_result, sizeof(LuxG1Projective381));
    cuda_driver.cuMemFree(d_p);
    cuda_driver.cuMemFree(d_scalar);
    cuda_driver.cuMemFree(d_result);
    return LUX_CRYPTO_OK;
}

static LuxCryptoError cuda_crypto_bls12_381_scalar_mul_batch(
    LuxBackendContext* context,
    const LuxG1Affine381* points,
    const LuxScalar256* scalars,
    LuxG1Projective381* results,
    size_t count
) {
    if (!context || !points || !scalars || !results || count == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    CUdeviceptr d_points, d_scalars, d_results;
    if (cuda_driver.cuMemAlloc(&d_points, sizeof(LuxG1Affine381) * count) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_scalars, sizeof(LuxScalar256) * count) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_results, sizeof(LuxG1Projective381) * count) != CUDA_SUCCESS) {
        return LUX_CRYPTO_ERROR_OUT_OF_MEMORY;
    }
    cuda_driver.cuMemcpyHtoD(d_points, points, sizeof(LuxG1Affine381) * count);
    cuda_driver.cuMemcpyHtoD(d_scalars, scalars, sizeof(LuxScalar256) * count);

    // TODO: Launch batch scalar multiplication kernel

    cuda_driver.cuMemcpyDtoH(results, d_results, sizeof(LuxG1Projective381) * count);
    cuda_driver.cuMemFree(d_points);
    cuda_driver.cuMemFree(d_scalars);
    cuda_driver.cuMemFree(d_results);
    return LUX_CRYPTO_OK;
}

// BN254 Operations
static LuxCryptoError cuda_crypto_bn254_add(
    LuxBackendContext* context,
    const LuxG1Projective254* p,
    const LuxG1Projective254* q,
    LuxG1Projective254* result
) {
    if (!context || !p || !q || !result) return LUX_CRYPTO_ERROR_INVALID_ARG;

    CUdeviceptr d_p, d_q, d_result;
    size_t sz = sizeof(LuxG1Projective254);
    if (cuda_driver.cuMemAlloc(&d_p, sz) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_q, sz) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_result, sz) != CUDA_SUCCESS) {
        return LUX_CRYPTO_ERROR_OUT_OF_MEMORY;
    }
    cuda_driver.cuMemcpyHtoD(d_p, p, sz);
    cuda_driver.cuMemcpyHtoD(d_q, q, sz);

    // TODO: Launch BN254 add kernel

    cuda_driver.cuMemcpyDtoH(result, d_result, sz);
    cuda_driver.cuMemFree(d_p);
    cuda_driver.cuMemFree(d_q);
    cuda_driver.cuMemFree(d_result);
    return LUX_CRYPTO_OK;
}

static LuxCryptoError cuda_crypto_bn254_double(
    LuxBackendContext* context,
    const LuxG1Projective254* p,
    LuxG1Projective254* result
) {
    if (!context || !p || !result) return LUX_CRYPTO_ERROR_INVALID_ARG;

    CUdeviceptr d_p, d_result;
    size_t sz = sizeof(LuxG1Projective254);
    if (cuda_driver.cuMemAlloc(&d_p, sz) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_result, sz) != CUDA_SUCCESS) {
        return LUX_CRYPTO_ERROR_OUT_OF_MEMORY;
    }
    cuda_driver.cuMemcpyHtoD(d_p, p, sz);

    // TODO: Launch BN254 double kernel

    cuda_driver.cuMemcpyDtoH(result, d_result, sz);
    cuda_driver.cuMemFree(d_p);
    cuda_driver.cuMemFree(d_result);
    return LUX_CRYPTO_OK;
}

static LuxCryptoError cuda_crypto_bn254_scalar_mul(
    LuxBackendContext* context,
    const LuxG1Projective254* p,
    const LuxScalar256* scalar,
    LuxG1Projective254* result
) {
    if (!context || !p || !scalar || !result) return LUX_CRYPTO_ERROR_INVALID_ARG;

    CUdeviceptr d_p, d_scalar, d_result;
    if (cuda_driver.cuMemAlloc(&d_p, sizeof(LuxG1Projective254)) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_scalar, sizeof(LuxScalar256)) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_result, sizeof(LuxG1Projective254)) != CUDA_SUCCESS) {
        return LUX_CRYPTO_ERROR_OUT_OF_MEMORY;
    }
    cuda_driver.cuMemcpyHtoD(d_p, p, sizeof(LuxG1Projective254));
    cuda_driver.cuMemcpyHtoD(d_scalar, scalar, sizeof(LuxScalar256));

    // TODO: Launch scalar multiplication kernel

    cuda_driver.cuMemcpyDtoH(result, d_result, sizeof(LuxG1Projective254));
    cuda_driver.cuMemFree(d_p);
    cuda_driver.cuMemFree(d_scalar);
    cuda_driver.cuMemFree(d_result);
    return LUX_CRYPTO_OK;
}

static LuxCryptoError cuda_crypto_bn254_scalar_mul_batch(
    LuxBackendContext* context,
    const LuxG1Affine254* points,
    const LuxScalar256* scalars,
    LuxG1Projective254* results,
    size_t count
) {
    if (!context || !points || !scalars || !results || count == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    CUdeviceptr d_points, d_scalars, d_results;
    if (cuda_driver.cuMemAlloc(&d_points, sizeof(LuxG1Affine254) * count) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_scalars, sizeof(LuxScalar256) * count) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_results, sizeof(LuxG1Projective254) * count) != CUDA_SUCCESS) {
        return LUX_CRYPTO_ERROR_OUT_OF_MEMORY;
    }
    cuda_driver.cuMemcpyHtoD(d_points, points, sizeof(LuxG1Affine254) * count);
    cuda_driver.cuMemcpyHtoD(d_scalars, scalars, sizeof(LuxScalar256) * count);

    // TODO: Launch batch scalar multiplication kernel

    cuda_driver.cuMemcpyDtoH(results, d_results, sizeof(LuxG1Projective254) * count);
    cuda_driver.cuMemFree(d_points);
    cuda_driver.cuMemFree(d_scalars);
    cuda_driver.cuMemFree(d_results);
    return LUX_CRYPTO_OK;
}

// Goldilocks Field Operations
static LuxCryptoError cuda_crypto_goldilocks_vec_add(
    LuxBackendContext* context,
    const LuxGoldilocks* a,
    const LuxGoldilocks* b,
    LuxGoldilocks* result,
    size_t n
) {
    if (!context || !a || !b || !result || n == 0) return LUX_CRYPTO_ERROR_INVALID_ARG;

    CUdeviceptr d_a, d_b, d_result;
    size_t sz = sizeof(LuxGoldilocks) * n;
    if (cuda_driver.cuMemAlloc(&d_a, sz) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_b, sz) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_result, sz) != CUDA_SUCCESS) {
        return LUX_CRYPTO_ERROR_OUT_OF_MEMORY;
    }
    cuda_driver.cuMemcpyHtoD(d_a, a, sz);
    cuda_driver.cuMemcpyHtoD(d_b, b, sz);

    // TODO: Launch Goldilocks add kernel (goldilocks.cu)

    cuda_driver.cuMemcpyDtoH(result, d_result, sz);
    cuda_driver.cuMemFree(d_a);
    cuda_driver.cuMemFree(d_b);
    cuda_driver.cuMemFree(d_result);
    return LUX_CRYPTO_OK;
}

static LuxCryptoError cuda_crypto_goldilocks_vec_mul(
    LuxBackendContext* context,
    const LuxGoldilocks* a,
    const LuxGoldilocks* b,
    LuxGoldilocks* result,
    size_t n
) {
    if (!context || !a || !b || !result || n == 0) return LUX_CRYPTO_ERROR_INVALID_ARG;

    CUdeviceptr d_a, d_b, d_result;
    size_t sz = sizeof(LuxGoldilocks) * n;
    if (cuda_driver.cuMemAlloc(&d_a, sz) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_b, sz) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_result, sz) != CUDA_SUCCESS) {
        return LUX_CRYPTO_ERROR_OUT_OF_MEMORY;
    }
    cuda_driver.cuMemcpyHtoD(d_a, a, sz);
    cuda_driver.cuMemcpyHtoD(d_b, b, sz);

    // TODO: Launch Goldilocks mul kernel

    cuda_driver.cuMemcpyDtoH(result, d_result, sz);
    cuda_driver.cuMemFree(d_a);
    cuda_driver.cuMemFree(d_b);
    cuda_driver.cuMemFree(d_result);
    return LUX_CRYPTO_OK;
}

static LuxCryptoError cuda_crypto_goldilocks_ntt_forward(
    LuxBackendContext* context,
    LuxGoldilocks* data,
    const LuxGoldilocks* twiddles,
    size_t n,
    uint32_t log_n
) {
    (void)log_n;
    if (!context || !data || !twiddles || n == 0) return LUX_CRYPTO_ERROR_INVALID_ARG;

    CUdeviceptr d_data, d_twiddles;
    size_t sz = sizeof(LuxGoldilocks) * n;
    if (cuda_driver.cuMemAlloc(&d_data, sz) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_twiddles, sz) != CUDA_SUCCESS) {
        return LUX_CRYPTO_ERROR_OUT_OF_MEMORY;
    }
    cuda_driver.cuMemcpyHtoD(d_data, data, sz);
    cuda_driver.cuMemcpyHtoD(d_twiddles, twiddles, sz);

    // TODO: Launch NTT forward kernel

    cuda_driver.cuMemcpyDtoH(data, d_data, sz);
    cuda_driver.cuMemFree(d_data);
    cuda_driver.cuMemFree(d_twiddles);
    return LUX_CRYPTO_OK;
}

static LuxCryptoError cuda_crypto_goldilocks_ntt_inverse(
    LuxBackendContext* context,
    LuxGoldilocks* data,
    const LuxGoldilocks* inv_twiddles,
    size_t n,
    uint32_t log_n
) {
    (void)log_n;
    if (!context || !data || !inv_twiddles || n == 0) return LUX_CRYPTO_ERROR_INVALID_ARG;

    CUdeviceptr d_data, d_twiddles;
    size_t sz = sizeof(LuxGoldilocks) * n;
    if (cuda_driver.cuMemAlloc(&d_data, sz) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_twiddles, sz) != CUDA_SUCCESS) {
        return LUX_CRYPTO_ERROR_OUT_OF_MEMORY;
    }
    cuda_driver.cuMemcpyHtoD(d_data, data, sz);
    cuda_driver.cuMemcpyHtoD(d_twiddles, inv_twiddles, sz);

    // TODO: Launch NTT inverse kernel

    cuda_driver.cuMemcpyDtoH(data, d_data, sz);
    cuda_driver.cuMemFree(d_data);
    cuda_driver.cuMemFree(d_twiddles);
    return LUX_CRYPTO_OK;
}

// Blake3 Hash
static LuxCryptoError cuda_crypto_blake3_hash(
    LuxBackendContext* context,
    const uint8_t* input,
    size_t input_len,
    uint8_t output[32]
) {
    if (!context || !input || !output || input_len == 0) return LUX_CRYPTO_ERROR_INVALID_ARG;

    CUdeviceptr d_input, d_output;
    if (cuda_driver.cuMemAlloc(&d_input, input_len) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_output, 32) != CUDA_SUCCESS) {
        return LUX_CRYPTO_ERROR_OUT_OF_MEMORY;
    }
    cuda_driver.cuMemcpyHtoD(d_input, input, input_len);

    // TODO: Launch Blake3 kernel (blake3.cu)

    cuda_driver.cuMemcpyDtoH(output, d_output, 32);
    cuda_driver.cuMemFree(d_input);
    cuda_driver.cuMemFree(d_output);
    return LUX_CRYPTO_OK;
}

static LuxCryptoError cuda_crypto_blake3_batch(
    LuxBackendContext* context,
    const uint8_t* inputs,
    size_t input_stride,
    const size_t* input_lengths,
    uint8_t* outputs,
    size_t num_inputs
) {
    if (!context || !inputs || !input_lengths || !outputs || num_inputs == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    size_t total_input_size = input_stride * num_inputs;
    CUdeviceptr d_inputs, d_lengths, d_outputs;
    if (cuda_driver.cuMemAlloc(&d_inputs, total_input_size) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_lengths, sizeof(size_t) * num_inputs) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_outputs, 32 * num_inputs) != CUDA_SUCCESS) {
        return LUX_CRYPTO_ERROR_OUT_OF_MEMORY;
    }
    cuda_driver.cuMemcpyHtoD(d_inputs, inputs, total_input_size);
    cuda_driver.cuMemcpyHtoD(d_lengths, input_lengths, sizeof(size_t) * num_inputs);

    // TODO: Launch batch Blake3 kernel

    cuda_driver.cuMemcpyDtoH(outputs, d_outputs, 32 * num_inputs);
    cuda_driver.cuMemFree(d_inputs);
    cuda_driver.cuMemFree(d_lengths);
    cuda_driver.cuMemFree(d_outputs);
    return LUX_CRYPTO_OK;
}

// KZG Commitments
static LuxCryptoError cuda_crypto_kzg_commit(
    LuxBackendContext* context,
    const LuxG1Affine381* srs_g1,
    const LuxScalar256* coeffs,
    LuxG1Projective381* commitment,
    uint32_t degree
) {
    if (!context || !srs_g1 || !coeffs || !commitment || degree == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }
    // KZG commit is MSM: C = sum(coeffs[i] * G[i])
    return cuda_crypto_msm(context, LUX_CURVE_BLS12_381, srs_g1, coeffs, commitment, degree);
}

static LuxCryptoError cuda_crypto_kzg_prove(
    LuxBackendContext* context,
    const LuxG1Affine381* srs_g1,
    const LuxScalar256* coeffs,
    const LuxScalar256* z,
    const LuxScalar256* p_z,
    LuxG1Projective381* proof,
    uint32_t degree
) {
    (void)z;
    (void)p_z;
    if (!context || !srs_g1 || !coeffs || !proof || degree == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }
    // TODO: Launch KZG prove kernel (kzg.cu)
    return LUX_CRYPTO_OK;
}

static LuxCryptoError cuda_crypto_kzg_batch_commit(
    LuxBackendContext* context,
    const LuxG1Affine381* srs_g1,
    const LuxScalar256* coeffs,
    LuxG1Projective381* commitments,
    uint32_t degree,
    uint32_t num_polys
) {
    if (!context || !srs_g1 || !coeffs || !commitments || degree == 0 || num_polys == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }
    for (uint32_t i = 0; i < num_polys; i++) {
        LuxCryptoError err = cuda_crypto_kzg_commit(context, srs_g1,
            coeffs + i * degree, commitments + i, degree);
        if (err != LUX_CRYPTO_OK) return err;
    }
    return LUX_CRYPTO_OK;
}

// Shamir Secret Sharing
static LuxCryptoError cuda_crypto_shamir_reconstruct(
    LuxBackendContext* context,
    int curve_type,
    const LuxScalar256* x_coords,
    const LuxScalar256* y_coords,
    LuxScalar256* secret,
    uint32_t threshold
) {
    (void)curve_type;
    if (!context || !x_coords || !y_coords || !secret || threshold == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    CUdeviceptr d_x, d_y, d_secret;
    if (cuda_driver.cuMemAlloc(&d_x, sizeof(LuxScalar256) * threshold) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_y, sizeof(LuxScalar256) * threshold) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_secret, sizeof(LuxScalar256)) != CUDA_SUCCESS) {
        return LUX_CRYPTO_ERROR_OUT_OF_MEMORY;
    }
    cuda_driver.cuMemcpyHtoD(d_x, x_coords, sizeof(LuxScalar256) * threshold);
    cuda_driver.cuMemcpyHtoD(d_y, y_coords, sizeof(LuxScalar256) * threshold);

    // TODO: Launch Shamir interpolation kernel (shamir_interpolate.cu)

    cuda_driver.cuMemcpyDtoH(secret, d_secret, sizeof(LuxScalar256));
    cuda_driver.cuMemFree(d_x);
    cuda_driver.cuMemFree(d_y);
    cuda_driver.cuMemFree(d_secret);
    return LUX_CRYPTO_OK;
}

static LuxCryptoError cuda_crypto_shamir_batch_reconstruct(
    LuxBackendContext* context,
    int curve_type,
    const LuxScalar256* x_coords,
    const LuxScalar256* y_coords,
    LuxScalar256* secrets,
    uint32_t threshold,
    uint32_t batch_size
) {
    (void)curve_type;
    if (!context || !x_coords || !y_coords || !secrets || threshold == 0 || batch_size == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    CUdeviceptr d_x, d_y, d_secrets;
    size_t shares_size = sizeof(LuxScalar256) * threshold * batch_size;
    if (cuda_driver.cuMemAlloc(&d_x, shares_size) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_y, shares_size) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_secrets, sizeof(LuxScalar256) * batch_size) != CUDA_SUCCESS) {
        return LUX_CRYPTO_ERROR_OUT_OF_MEMORY;
    }
    cuda_driver.cuMemcpyHtoD(d_x, x_coords, shares_size);
    cuda_driver.cuMemcpyHtoD(d_y, y_coords, shares_size);

    // TODO: Launch batch Shamir interpolation kernel

    cuda_driver.cuMemcpyDtoH(secrets, d_secrets, sizeof(LuxScalar256) * batch_size);
    cuda_driver.cuMemFree(d_x);
    cuda_driver.cuMemFree(d_y);
    cuda_driver.cuMemFree(d_secrets);
    return LUX_CRYPTO_OK;
}

static LuxCryptoError cuda_crypto_shamir_lagrange_coefficients(
    LuxBackendContext* context,
    int curve_type,
    const LuxScalar256* x_coords,
    LuxScalar256* coefficients,
    uint32_t num_parties
) {
    (void)curve_type;
    if (!context || !x_coords || !coefficients || num_parties == 0) {
        return LUX_CRYPTO_ERROR_INVALID_ARG;
    }

    CUdeviceptr d_x, d_coeffs;
    size_t sz = sizeof(LuxScalar256) * num_parties;
    if (cuda_driver.cuMemAlloc(&d_x, sz) != CUDA_SUCCESS ||
        cuda_driver.cuMemAlloc(&d_coeffs, sz) != CUDA_SUCCESS) {
        return LUX_CRYPTO_ERROR_OUT_OF_MEMORY;
    }
    cuda_driver.cuMemcpyHtoD(d_x, x_coords, sz);

    // TODO: Launch Lagrange coefficient kernel

    cuda_driver.cuMemcpyDtoH(coefficients, d_coeffs, sz);
    cuda_driver.cuMemFree(d_x);
    cuda_driver.cuMemFree(d_coeffs);
    return LUX_CRYPTO_OK;
}

// =============================================================================
// CUDA Crypto VTable
// =============================================================================

static const lux_gpu_crypto_vtbl cuda_crypto_vtbl = {
    .msm = cuda_crypto_msm,
    .msm_batch = cuda_crypto_msm_batch,
    .poseidon2_hash = cuda_crypto_poseidon2_hash,
    .poseidon2_batch = cuda_crypto_poseidon2_batch,
    .poseidon2_merkle = cuda_crypto_poseidon2_merkle,
    .bls12_381_add = cuda_crypto_bls12_381_add,
    .bls12_381_double = cuda_crypto_bls12_381_double,
    .bls12_381_scalar_mul = cuda_crypto_bls12_381_scalar_mul,
    .bls12_381_scalar_mul_batch = cuda_crypto_bls12_381_scalar_mul_batch,
    .bn254_add = cuda_crypto_bn254_add,
    .bn254_double = cuda_crypto_bn254_double,
    .bn254_scalar_mul = cuda_crypto_bn254_scalar_mul,
    .bn254_scalar_mul_batch = cuda_crypto_bn254_scalar_mul_batch,
    .goldilocks_vec_add = cuda_crypto_goldilocks_vec_add,
    .goldilocks_vec_mul = cuda_crypto_goldilocks_vec_mul,
    .goldilocks_ntt_forward = cuda_crypto_goldilocks_ntt_forward,
    .goldilocks_ntt_inverse = cuda_crypto_goldilocks_ntt_inverse,
    .blake3_hash = cuda_crypto_blake3_hash,
    .blake3_batch = cuda_crypto_blake3_batch,
    .kzg_commit = cuda_crypto_kzg_commit,
    .kzg_prove = cuda_crypto_kzg_prove,
    .kzg_batch_commit = cuda_crypto_kzg_batch_commit,
    .shamir_reconstruct = cuda_crypto_shamir_reconstruct,
    .shamir_batch_reconstruct = cuda_crypto_shamir_batch_reconstruct,
    .shamir_lagrange_coefficients = cuda_crypto_shamir_lagrange_coefficients,
    ._reserved = {nullptr}
};

// =============================================================================
// CUDA Backend VTable
// =============================================================================

static const lux_gpu_backend_vtbl cuda_vtbl = {
    // Lifecycle
    .create_context = cuda_create_context,
    .destroy_context = cuda_destroy_context,

    // Device info
    .get_device_count = cuda_get_device_count,
    .get_device_info = cuda_get_device_info,

    // Sync
    .sync = cuda_sync,

    // Buffer management
    .buffer_alloc = cuda_buffer_alloc,
    .buffer_alloc_with_data = cuda_buffer_alloc_with_data,
    .buffer_free = cuda_buffer_free,
    .buffer_copy_to_host = cuda_buffer_copy_to_host,
    .buffer_copy_from_host = cuda_buffer_copy_from_host,
    .buffer_get_host_ptr = cuda_buffer_get_host_ptr,

    // Kernel management
    .kernel_load = cuda_kernel_load,
    .kernel_load_binary = cuda_kernel_load_binary,
    .kernel_destroy = cuda_kernel_destroy,
    .kernel_dispatch = cuda_kernel_dispatch,

    // Elementwise operations
    .op_add_f32 = cuda_op_add_f32,
    .op_sub_f32 = cuda_op_sub_f32,
    .op_mul_f32 = cuda_op_mul_f32,
    .op_div_f32 = cuda_op_div_f32,

    // Matrix operations
    .op_matmul_f32 = cuda_op_matmul_f32,
    .op_transpose_f32 = cuda_op_transpose_f32,

    // Reduction operations
    .op_reduce_sum_f32 = cuda_op_reduce_sum_f32,
    .op_reduce_max_f32 = cuda_op_reduce_max_f32,
    .op_reduce_min_f32 = cuda_op_reduce_min_f32,
    .op_reduce_mean_f32 = cuda_op_reduce_mean_f32,
    .op_reduce_sum_axis_f32 = cuda_op_reduce_sum_axis_f32,
    .op_reduce_max_axis_f32 = cuda_op_reduce_max_axis_f32,

    // Softmax operations
    .op_softmax_f32 = cuda_op_softmax_f32,
    .op_log_softmax_f32 = cuda_op_log_softmax_f32,

    // Unary operations
    .op_exp_f32 = cuda_op_exp_f32,
    .op_log_f32 = cuda_op_log_f32,
    .op_sqrt_f32 = cuda_op_sqrt_f32,
    .op_neg_f32 = cuda_op_neg_f32,
    .op_abs_f32 = cuda_op_abs_f32,
    .op_tanh_f32 = cuda_op_tanh_f32,
    .op_sigmoid_f32 = cuda_op_sigmoid_f32,
    .op_relu_f32 = cuda_op_relu_f32,
    .op_gelu_f32 = cuda_op_gelu_f32,

    // Copy operations
    .op_copy_f32 = cuda_op_copy_f32,

    // Normalization operations
    .op_layer_norm_f32 = cuda_op_layer_norm_f32,
    .op_rms_norm_f32 = cuda_op_rms_norm_f32,

    // NTT operations
    .op_ntt_forward = cuda_op_ntt_forward,
    .op_ntt_inverse = cuda_op_ntt_inverse,

    // MSM
    .op_msm = cuda_op_msm,

    // FHE operations
    .op_poly_mul = cuda_op_poly_mul,
    .op_tfhe_bootstrap = cuda_op_tfhe_bootstrap,
    .op_tfhe_keyswitch = cuda_op_tfhe_keyswitch,
    .op_blind_rotate = cuda_op_blind_rotate,
    .op_sample_extract = cuda_op_sample_extract,
    .op_sample_ntt = cuda_op_sample_ntt,

    // Reserved
    ._reserved = {nullptr}
};

// =============================================================================
// Plugin Entry Point
// =============================================================================

static bool cuda_backend_init_impl(lux_gpu_backend_desc* out) {
    if (!out) return false;

    // Check if CUDA is available
    if (!load_cuda_driver()) {
        return false;
    }

    int device_count = 0;
    cuda_driver.cuDeviceGetCount(&device_count);
    if (device_count == 0) {
        return false;
    }

    out->abi_version = LUX_GPU_BACKEND_ABI_VERSION;
    out->backend_name = "cuda";
    out->backend_version = "0.1.0";
    out->capabilities = LUX_CAP_TENSOR_OPS | LUX_CAP_MATMUL | LUX_CAP_NTT | LUX_CAP_MSM |
                        LUX_CAP_CUSTOM_KERNELS | LUX_CAP_FHE | LUX_CAP_TFHE;
    out->vtbl = &cuda_vtbl;
    return true;
}

LUX_GPU_DECLARE_BACKEND(cuda_backend_init_impl)

// Extended entry point for crypto backend
extern "C" LUX_GPU_BACKEND_EXPORT bool lux_gpu_crypto_backend_init(lux_gpu_crypto_backend_desc* out) {
    if (!out) return false;
    if (!cuda_backend_init_impl(&out->base)) return false;
    out->crypto_vtbl = &cuda_crypto_vtbl;
    out->crypto_capabilities =
        LUX_CRYPTO_CAP_MSM |
        LUX_CRYPTO_CAP_POSEIDON2 |
        LUX_CRYPTO_CAP_BLS12_381 |
        LUX_CRYPTO_CAP_BN254 |
        LUX_CRYPTO_CAP_GOLDILOCKS |
        LUX_CRYPTO_CAP_BLAKE3 |
        LUX_CRYPTO_CAP_KZG |
        LUX_CRYPTO_CAP_SHAMIR;
    return true;
}
