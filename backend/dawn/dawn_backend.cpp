// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause-Eco
//
// Dawn/WebGPU Backend - Cross-platform GPU acceleration via gpu.cpp

#include "../../src/backend.hpp"

#ifdef LUX_HAS_DAWN

// gpu.cpp header-only library
#define USE_DAWN_API
#include "gpu.hpp"

#include <unordered_map>
#include <string>
#include <fstream>
#include <sstream>

// WGSL kernel sources (embedded)
namespace kernels {

static const char* kAddKernel = R"(
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&a)) {
        out[idx] = a[idx] + b[idx];
    }
}
)";

static const char* kSubKernel = R"(
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&a)) {
        out[idx] = a[idx] - b[idx];
    }
}
)";

static const char* kMulKernel = R"(
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&a)) {
        out[idx] = a[idx] * b[idx];
    }
}
)";

static const char* kMatmulKernel = R"(
struct Params {
    M: u32,
    K: u32,
    N: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;

    if (row >= params.M || col >= params.N) {
        return;
    }

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < params.K; k = k + 1u) {
        sum = sum + a[row * params.K + k] * b[k * params.N + col];
    }
    out[row * params.N + col] = sum;
}
)";

static const char* kNttForwardKernel = R"(
struct NTTParams {
    n: u32,
    modulus: u32,
    omega: u32,
    _pad: u32,
}

fn mod_mul(a: u32, b: u32, m: u32) -> u32 {
    let prod = u64(a) * u64(b);
    return u32(prod % u64(m));
}

fn mod_add(a: u32, b: u32, m: u32) -> u32 {
    let sum = a + b;
    return select(sum, sum - m, sum >= m);
}

fn mod_sub(a: u32, b: u32, m: u32) -> u32 {
    return select(a - b, m - b + a, a < b);
}

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<uniform> params: NTTParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n = params.n;
    let m = params.modulus;

    if (idx >= n / 2u) {
        return;
    }

    let i = idx * 2u;
    let u = data[i];
    let t = data[i + 1u];

    data[i] = mod_add(u, t, m);
    data[i + 1u] = mod_sub(u, t, m);
}
)";

static const char* kNttInverseKernel = R"(
struct NTTInvParams {
    n: u32,
    modulus: u32,
    omega_inv: u32,
    n_inv: u32,
}

fn mod_mul(a: u32, b: u32, m: u32) -> u32 {
    let prod = u64(a) * u64(b);
    return u32(prod % u64(m));
}

fn mod_add(a: u32, b: u32, m: u32) -> u32 {
    let sum = a + b;
    return select(sum, sum - m, sum >= m);
}

fn mod_sub(a: u32, b: u32, m: u32) -> u32 {
    return select(a - b, m - b + a, a < b);
}

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<uniform> params: NTTInvParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n = params.n;
    let m = params.modulus;

    if (idx >= n / 2u) {
        return;
    }

    let i = idx * 2u;
    let u = data[i];
    let t = data[i + 1u];

    data[i] = mod_mul(mod_add(u, t, m), params.n_inv, m);
    data[i + 1u] = mod_mul(mod_sub(u, t, m), params.n_inv, m);
}
)";

} // namespace kernels

class DawnBackend : public Backend {
public:
    explicit DawnBackend(int device_index = 0) : device_index_(device_index) {
        // Initialize gpu.cpp context
        ctx_ = gpu::createContext();
        if (!ctx_.device) {
            initialized_ = false;
            return;
        }
        initialized_ = true;
        compile_kernels();
    }

    ~DawnBackend() override = default;

    LuxBackend type() const override { return LUX_BACKEND_DAWN; }
    const char* name() const override { return "dawn"; }

    LuxError sync() override {
        if (!initialized_) return LUX_ERROR_BACKEND_NOT_AVAILABLE;
        // gpu.cpp handles synchronization internally
        return LUX_OK;
    }

    LuxError device_info(LuxDeviceInfo* info) override {
        if (!info || !initialized_) return LUX_ERROR_INVALID_ARGUMENT;

        info->backend = LUX_BACKEND_DAWN;
        info->index = device_index_;
        info->name = "WebGPU (Dawn)";
        info->vendor = "Google/Dawn";
        info->memory_total = 0;  // Not easily available via WebGPU
        info->memory_available = 0;
        info->is_discrete = false;
        info->is_unified_memory = true;
        info->compute_units = 0;
        info->max_workgroup_size = 256;

        return LUX_OK;
    }

    LuxTensor* zeros(const int64_t* shape, int ndim, LuxDtype dtype) override {
        auto t = new LuxTensor();
        t->shape.assign(shape, shape + ndim);
        t->dtype = dtype;
        size_t bytes = t->size() * t->element_size();
        t->data.resize(bytes, 0);

        // Create GPU buffer
        gpu::Shape gpu_shape;
        gpu_shape.rank = 1;
        gpu_shape.data[0] = t->size();

        auto tensor = gpu::createTensor(ctx_, gpu_shape, gpu::kf32, t->data.data());
        t->device_ptr = new gpu::Tensor(tensor);

        return t;
    }

    LuxTensor* ones(const int64_t* shape, int ndim, LuxDtype dtype) override {
        return full(shape, ndim, dtype, 1.0);
    }

    LuxTensor* full(const int64_t* shape, int ndim, LuxDtype dtype, double value) override {
        auto t = zeros(shape, ndim, dtype);
        if (dtype == LUX_FLOAT32) {
            float v = static_cast<float>(value);
            float* ptr = reinterpret_cast<float*>(t->data.data());
            for (int64_t i = 0; i < t->size(); i++) ptr[i] = v;
            sync_to_device(t);
        }
        return t;
    }

    LuxTensor* from_data(const void* data, const int64_t* shape, int ndim, LuxDtype dtype) override {
        auto t = new LuxTensor();
        t->shape.assign(shape, shape + ndim);
        t->dtype = dtype;
        size_t bytes = t->size() * t->element_size();
        t->data.resize(bytes);
        std::memcpy(t->data.data(), data, bytes);

        gpu::Shape gpu_shape;
        gpu_shape.rank = 1;
        gpu_shape.data[0] = t->size();

        auto tensor = gpu::createTensor(ctx_, gpu_shape, gpu::kf32, data);
        t->device_ptr = new gpu::Tensor(tensor);

        return t;
    }

    void sync_to_device(LuxTensor* t) override {
        if (t && t->device_ptr) {
            // Recreate GPU tensor with updated data
            gpu::Tensor* old = static_cast<gpu::Tensor*>(t->device_ptr);
            delete old;

            gpu::Shape gpu_shape;
            gpu_shape.rank = 1;
            gpu_shape.data[0] = t->size();

            auto tensor = gpu::createTensor(ctx_, gpu_shape, gpu::kf32, t->data.data());
            t->device_ptr = new gpu::Tensor(tensor);
        }
    }

    void destroy_tensor(LuxTensor* t) override {
        if (t && t->device_ptr) {
            delete static_cast<gpu::Tensor*>(t->device_ptr);
        }
        delete t;
    }

    LuxTensor* add(LuxTensor* a, LuxTensor* b) override {
        return binary_op(a, b, "add");
    }

    LuxTensor* sub(LuxTensor* a, LuxTensor* b) override {
        return binary_op(a, b, "sub");
    }

    LuxTensor* mul(LuxTensor* a, LuxTensor* b) override {
        return binary_op(a, b, "mul");
    }

    LuxTensor* matmul(LuxTensor* a, LuxTensor* b) override {
        if (a->shape.size() != 2 || b->shape.size() != 2) return nullptr;
        if (a->shape[1] != b->shape[0]) return nullptr;
        if (a->dtype != LUX_FLOAT32) return nullptr;

        int64_t M = a->shape[0];
        int64_t K = a->shape[1];
        int64_t N = b->shape[1];

        int64_t out_shape[2] = {M, N};
        auto out = zeros(out_shape, 2, a->dtype);

        auto it = kernels_.find("matmul");
        if (it == kernels_.end()) {
            // Fallback to CPU
            const float* pa = reinterpret_cast<const float*>(a->data.data());
            const float* pb = reinterpret_cast<const float*>(b->data.data());
            float* po = reinterpret_cast<float*>(out->data.data());

            for (int64_t i = 0; i < M; i++) {
                for (int64_t j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (int64_t k = 0; k < K; k++) {
                        sum += pa[i * K + k] * pb[k * N + j];
                    }
                    po[i * N + j] = sum;
                }
            }
            return out;
        }

        // Use GPU kernel
        gpu::Tensor* ta = static_cast<gpu::Tensor*>(a->device_ptr);
        gpu::Tensor* tb = static_cast<gpu::Tensor*>(b->device_ptr);
        gpu::Tensor* tout = static_cast<gpu::Tensor*>(out->device_ptr);

        // Create params buffer
        struct MatmulParams {
            uint32_t M, K, N, _pad;
        } params = {
            static_cast<uint32_t>(M),
            static_cast<uint32_t>(K),
            static_cast<uint32_t>(N),
            0
        };

        gpu::Shape params_shape = {1};
        auto params_tensor = gpu::createTensor(ctx_, params_shape, gpu::ki32, &params);

        // Dispatch
        gpu::Kernel kernel = it->second;
        // Reset and dispatch the kernel
        // (Simplified - actual implementation would need proper binding setup)

        // Copy result back
        gpu::toCPU(ctx_, *tout, out->data.data(), out->data.size());

        return out;
    }

    LuxError bls_verify_batch(const uint8_t* const*, const size_t*,
                              const uint8_t* const*, const size_t*,
                              const uint8_t* const*, const size_t*,
                              int, bool*) override {
        return LUX_ERROR_NOT_SUPPORTED;  // TODO: Use bls12_381.wgsl
    }

    LuxError ntt_forward(uint64_t* data, size_t n, uint64_t modulus) override {
        if (!data || n == 0 || (n & (n - 1)) != 0) return LUX_ERROR_INVALID_ARGUMENT;
        if (!initialized_) return LUX_ERROR_BACKEND_NOT_AVAILABLE;

        // For now, use CPU fallback (64-bit modular arithmetic in WGSL is limited)
        // Full implementation would use ntt.wgsl from kernels/wgsl/
        return cpu_ntt_forward(data, n, modulus);
    }

    LuxError ntt_inverse(uint64_t* data, size_t n, uint64_t modulus) override {
        if (!data || n == 0 || (n & (n - 1)) != 0) return LUX_ERROR_INVALID_ARGUMENT;
        if (!initialized_) return LUX_ERROR_BACKEND_NOT_AVAILABLE;

        return cpu_ntt_inverse(data, n, modulus);
    }

    bool is_available() const { return initialized_; }

private:
    gpu::Context ctx_;
    int device_index_;
    bool initialized_ = false;
    std::unordered_map<std::string, gpu::Kernel> kernels_;

    void compile_kernels() {
        // Compile WGSL kernels
        const struct {
            const char* name;
            const char* code;
            size_t workgroup_size;
        } kernel_defs[] = {
            {"add", kernels::kAddKernel, 256},
            {"sub", kernels::kSubKernel, 256},
            {"mul", kernels::kMulKernel, 256},
            {"matmul", kernels::kMatmulKernel, 256},
        };

        for (const auto& def : kernel_defs) {
            gpu::KernelCode code(def.code, def.workgroup_size, gpu::kf32);
            // Note: gpu.cpp createKernel requires bindings at creation time
            // We'll create kernels on-demand in actual operations
        }
    }

    LuxTensor* binary_op(LuxTensor* a, LuxTensor* b, const std::string& op_name) {
        if (a->shape != b->shape || a->dtype != b->dtype) return nullptr;
        if (a->dtype != LUX_FLOAT32) return nullptr;

        auto out = zeros(a->shape.data(), static_cast<int>(a->shape.size()), a->dtype);

        // CPU fallback (full GPU implementation would use WGSL kernels)
        const float* pa = reinterpret_cast<const float*>(a->data.data());
        const float* pb = reinterpret_cast<const float*>(b->data.data());
        float* po = reinterpret_cast<float*>(out->data.data());
        int64_t n = a->size();

        if (op_name == "add") {
            for (int64_t i = 0; i < n; i++) po[i] = pa[i] + pb[i];
        } else if (op_name == "sub") {
            for (int64_t i = 0; i < n; i++) po[i] = pa[i] - pb[i];
        } else if (op_name == "mul") {
            for (int64_t i = 0; i < n; i++) po[i] = pa[i] * pb[i];
        }

        return out;
    }

    // CPU fallback for NTT (64-bit arithmetic)
    LuxError cpu_ntt_forward(uint64_t* data, size_t n, uint64_t modulus) {
        uint64_t omega = find_primitive_root(n, modulus);
        if (omega == 0) return LUX_ERROR_INVALID_ARGUMENT;

        bit_reverse(data, n);

        for (size_t len = 2; len <= n; len *= 2) {
            uint64_t w = mod_pow(omega, (modulus - 1) / len, modulus);
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
        return LUX_OK;
    }

    LuxError cpu_ntt_inverse(uint64_t* data, size_t n, uint64_t modulus) {
        uint64_t omega = find_primitive_root(n, modulus);
        uint64_t omega_inv = mod_pow(omega, modulus - 2, modulus);

        bit_reverse(data, n);

        for (size_t len = 2; len <= n; len *= 2) {
            uint64_t w = mod_pow(omega_inv, (modulus - 1) / len, modulus);
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

        uint64_t n_inv = mod_pow(n, modulus - 2, modulus);
        for (size_t i = 0; i < n; i++) {
            data[i] = mod_mul(data[i], n_inv, modulus);
        }
        return LUX_OK;
    }

    static uint64_t mod_add(uint64_t a, uint64_t b, uint64_t m) {
        return (a + b) % m;
    }

    static uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t m) {
        return (a >= b) ? (a - b) : (m - b + a);
    }

    static uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t m) {
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
        if (m == 0xFFFFFFFF00000001ULL) return 7;  // Goldilocks
        if (m == 0x1000000000000001ULL) return 3;
        return 3;
    }
};

// Factory function
std::unique_ptr<Backend> create_dawn_backend(int device_index) {
    auto backend = std::make_unique<DawnBackend>(device_index);
    return backend->is_available() ? std::move(backend) : nullptr;
}

// Availability check
bool check_dawn_available_impl() {
    // Try to create a gpu.cpp context
    try {
        gpu::Context ctx = gpu::createContext();
        return ctx.device != nullptr;
    } catch (...) {
        return false;
    }
}

#endif // LUX_HAS_DAWN
