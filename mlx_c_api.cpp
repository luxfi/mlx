// MLX C API Implementation
// This provides the C interface to the C++ MLX library

#include "mlx_c_api.h"
#include "mlx/mlx.h"
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <stdexcept>

#ifdef __APPLE__
#include <TargetConditionals.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#if TARGET_OS_MAC && TARGET_CPU_ARM64
#define HAS_METAL 1
#endif
#endif

#ifdef __linux__
#include <unistd.h>
#ifdef __CUDACC__
#define HAS_CUDA 1
#endif
#endif

#ifdef _WIN32
#include <windows.h>
#endif

// Backend detection
extern "C" {

bool mlx_has_metal() {
#ifdef HAS_METAL
    return mlx::core::metal::is_available();
#else
    return false;
#endif
}

bool mlx_has_cuda() {
#ifdef HAS_CUDA
    // CUDA is not yet available in this version
    return false;
#else
    return false;
#endif
}

char* mlx_get_metal_device_name() {
#ifdef HAS_METAL
    if (mlx::core::metal::is_available()) {
        const char* name = "Apple M-series GPU";
        char* result = (char*)malloc(strlen(name) + 1);
        strcpy(result, name);
        return result;
    }
#endif
    const char* name = "No Metal Support";
    char* result = (char*)malloc(strlen(name) + 1);
    strcpy(result, name);
    return result;
}

char* mlx_get_cuda_device_name() {
#ifdef HAS_CUDA
    const char* name = "NVIDIA GPU";
    char* result = (char*)malloc(strlen(name) + 1);
    strcpy(result, name);
    return result;
#endif
    const char* name = "No CUDA Support";
    char* result = (char*)malloc(strlen(name) + 1);
    strcpy(result, name);
    return result;
}

size_t mlx_get_metal_memory() {
#ifdef HAS_METAL
    // On Apple Silicon, unified memory equals system memory
    return mlx_get_system_memory();
#else
    return 0;
#endif
}

size_t mlx_get_cuda_memory() {
#ifdef HAS_CUDA
    // Would query CUDA device for actual memory
    // For now, return a reasonable default
    return 8ULL * 1024 * 1024 * 1024; // 8GB default
#else
    return 0;
#endif
}

size_t mlx_get_system_memory() {
#ifdef __APPLE__
    // Get actual system memory on macOS
    int64_t memsize = 0;
    size_t len = sizeof(memsize);
    if (sysctlbyname("hw.memsize", &memsize, &len, NULL, 0) == 0) {
        return (size_t)memsize;
    }
    return 16ULL * 1024 * 1024 * 1024; // Fallback
#elif defined(__linux__)
    // Get system memory on Linux
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        return (size_t)pages * (size_t)page_size;
    }
    return 16ULL * 1024 * 1024 * 1024; // Fallback
#elif defined(_WIN32)
    // Get system memory on Windows
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return (size_t)status.ullTotalPhys;
    }
    return 16ULL * 1024 * 1024 * 1024; // Fallback
#else
    return 16ULL * 1024 * 1024 * 1024; // Default fallback
#endif
}

} // extern "C"

// Helper: Convert C dtype to MLX dtype (internal, C++ linkage)
static mlx::core::Dtype to_mlx_dtype(int dtype) {
    switch (dtype) {
        case 0: return mlx::core::float32;
        case 1: return mlx::core::float64;
        case 2: return mlx::core::int32;
        case 3: return mlx::core::int64;
        case 4: return mlx::core::bool_;
        default: return mlx::core::float32;
    }
}

extern "C" {

// Array operations using real MLX library
void* mlx_zeros(int* shape, int ndim, int dtype) {
    try {
        mlx::core::Shape shape_vec(shape, shape + ndim);
        mlx::core::Dtype mlx_dtype = to_mlx_dtype(dtype);
        mlx::core::array arr = mlx::core::zeros(shape_vec, mlx_dtype);
        return new mlx::core::array(std::move(arr));
    } catch (const std::exception& e) {
        // Log error if needed
        return nullptr;
    }
}

void* mlx_ones(int* shape, int ndim, int dtype) {
    try {
        mlx::core::Shape shape_vec(shape, shape + ndim);
        mlx::core::Dtype mlx_dtype = to_mlx_dtype(dtype);
        mlx::core::array arr = mlx::core::ones(shape_vec, mlx_dtype);
        return new mlx::core::array(std::move(arr));
    } catch (const std::exception& e) {
        return nullptr;
    }
}

void* mlx_random(int* shape, int ndim, int dtype) {
    try {
        mlx::core::Shape shape_vec(shape, shape + ndim);
        mlx::core::Dtype mlx_dtype = to_mlx_dtype(dtype);
        // Use random::uniform for random values
        mlx::core::array arr = mlx::core::random::uniform(
            0.0f, 1.0f, shape_vec, mlx_dtype);
        return new mlx::core::array(std::move(arr));
    } catch (const std::exception& e) {
        return nullptr;
    }
}

void* mlx_arange(double start, double stop, double step) {
    try {
        mlx::core::array arr = mlx::core::arange(start, stop, step);
        return new mlx::core::array(std::move(arr));
    } catch (const std::exception& e) {
        return nullptr;
    }
}

void* mlx_add(void* a, void* b) {
    try {
        auto* arr_a = static_cast<mlx::core::array*>(a);
        auto* arr_b = static_cast<mlx::core::array*>(b);
        mlx::core::array result = mlx::core::add(*arr_a, *arr_b);
        return new mlx::core::array(std::move(result));
    } catch (const std::exception& e) {
        return nullptr;
    }
}

void* mlx_multiply(void* a, void* b) {
    try {
        auto* arr_a = static_cast<mlx::core::array*>(a);
        auto* arr_b = static_cast<mlx::core::array*>(b);
        mlx::core::array result = mlx::core::multiply(*arr_a, *arr_b);
        return new mlx::core::array(std::move(result));
    } catch (const std::exception& e) {
        return nullptr;
    }
}

void* mlx_matmul(void* a, void* b) {
    try {
        auto* arr_a = static_cast<mlx::core::array*>(a);
        auto* arr_b = static_cast<mlx::core::array*>(b);
        mlx::core::array result = mlx::core::matmul(*arr_a, *arr_b);
        return new mlx::core::array(std::move(result));
    } catch (const std::exception& e) {
        return nullptr;
    }
}

void* mlx_sum(void* array, int* axis, int naxis) {
    try {
        auto* arr = static_cast<mlx::core::array*>(array);
        mlx::core::array result = (naxis == 0) 
            ? mlx::core::sum(*arr)
            : mlx::core::sum(*arr, std::vector<int>(axis, axis + naxis));
        return new mlx::core::array(std::move(result));
    } catch (const std::exception& e) {
        return nullptr;
    }
}

void* mlx_mean(void* array, int* axis, int naxis) {
    try {
        auto* arr = static_cast<mlx::core::array*>(array);
        mlx::core::array result = (naxis == 0)
            ? mlx::core::mean(*arr)
            : mlx::core::mean(*arr, std::vector<int>(axis, axis + naxis));
        return new mlx::core::array(std::move(result));
    } catch (const std::exception& e) {
        return nullptr;
    }
}

void mlx_eval(void* arrays[], int count) {
    try {
        std::vector<mlx::core::array> arr_vec;
        for (int i = 0; i < count; i++) {
            if (arrays[i]) {
                auto* arr = static_cast<mlx::core::array*>(arrays[i]);
                arr_vec.push_back(*arr);
            }
        }
        mlx::core::eval(arr_vec);
    } catch (const std::exception& e) {
        // Log error if needed
    }
}

void mlx_synchronize() {
    try {
        mlx::core::synchronize();
    } catch (const std::exception& e) {
        // Log error if needed
    }
}

void* mlx_new_stream() {
    try {
        mlx::core::Device default_device(mlx::core::Device::gpu, 0);
        mlx::core::Stream stream = mlx::core::new_stream(default_device);
        return new mlx::core::Stream(std::move(stream));
    } catch (const std::exception& e) {
        return nullptr;
    }
}

void mlx_free_stream(void* stream) {
    if (stream) {
        auto* s = static_cast<mlx::core::Stream*>(stream);
        delete s;
    }
}

void mlx_free_array(void* array) {
    if (array) {
        auto* arr = static_cast<mlx::core::array*>(array);
        delete arr;
    }
}

} // extern "C"

// Create array from data slice
void* mlx_from_slice(float* data, int data_len, int* shape, int ndim, int dtype) {
    try {
        mlx::core::Shape shape_vec(shape, shape + ndim);
        mlx::core::Dtype mlx_dtype = to_mlx_dtype(dtype);
        
        // Create array from data
        std::vector<float> data_vec(data, data + data_len);
        mlx::core::array arr(data_vec.data(), shape_vec, mlx_dtype);
        
        return new mlx::core::array(std::move(arr));
    } catch (const std::exception& e) {
        return nullptr;
    }
}

// Element-wise maximum
void* mlx_maximum(void* a, void* b) {
    try {
        auto* arr_a = static_cast<mlx::core::array*>(a);
        auto* arr_b = static_cast<mlx::core::array*>(b);
        mlx::core::array result = mlx::core::maximum(*arr_a, *arr_b);
        return new mlx::core::array(std::move(result));
    } catch (const std::exception& e) {
        return nullptr;
    }
}

// ============================================================================
// NTT (Number Theoretic Transform) Implementation
// ============================================================================

namespace {

// 128-bit unsigned integer for intermediate calculations
struct uint128_t {
    uint64_t lo;
    uint64_t hi;

    uint128_t() : lo(0), hi(0) {}
    uint128_t(uint64_t v) : lo(v), hi(0) {}
    uint128_t(uint64_t h, uint64_t l) : lo(l), hi(h) {}
};

// 64x64 -> 128 bit multiplication
inline uint128_t mul64x64(uint64_t a, uint64_t b) {
#if defined(__SIZEOF_INT128__)
    __uint128_t prod = (__uint128_t)a * b;
    return uint128_t((uint64_t)(prod >> 64), (uint64_t)prod);
#else
    // Software fallback for platforms without __int128
    uint64_t a_lo = a & 0xFFFFFFFF;
    uint64_t a_hi = a >> 32;
    uint64_t b_lo = b & 0xFFFFFFFF;
    uint64_t b_hi = b >> 32;

    uint64_t p00 = a_lo * b_lo;
    uint64_t p01 = a_lo * b_hi;
    uint64_t p10 = a_hi * b_lo;
    uint64_t p11 = a_hi * b_hi;

    uint64_t mid = p01 + p10;
    uint64_t mid_carry = (mid < p01) ? 1ULL << 32 : 0;

    uint64_t lo = p00 + (mid << 32);
    uint64_t lo_carry = (lo < p00) ? 1 : 0;

    uint64_t hi = p11 + (mid >> 32) + mid_carry + lo_carry;

    return uint128_t(hi, lo);
#endif
}

// Modular reduction: 128-bit / 64-bit
inline uint64_t mod_128_64(uint128_t a, uint64_t Q) {
#if defined(__SIZEOF_INT128__)
    // Use native 128-bit arithmetic when available
    __uint128_t val = ((__uint128_t)a.hi << 64) | a.lo;
    return (uint64_t)(val % Q);
#else
    // Software fallback for platforms without __int128
    if (a.hi == 0) {
        return a.lo % Q;
    }

    // Compute (hi * 2^64 + lo) mod Q without 128-bit arithmetic
    // We reduce hi and lo separately, then combine
    
    // Compute 2^64 mod Q iteratively (not recursively)
    uint64_t pow64_mod_q = 1;
    for (int i = 0; i < 64; i++) {
        pow64_mod_q <<= 1;
        if (pow64_mod_q >= Q) pow64_mod_q -= Q;
    }

    uint64_t lo_mod = a.lo % Q;
    uint64_t hi_mod = a.hi % Q;

    // Compute hi_mod * pow64_mod_q mod Q using shift-and-add
    uint64_t hi_contrib = 0;
    uint64_t factor = pow64_mod_q;
    uint64_t h = hi_mod;
    while (h > 0) {
        if (h & 1) {
            hi_contrib += factor;
            if (hi_contrib >= Q) hi_contrib -= Q;
        }
        factor <<= 1;
        if (factor >= Q) factor -= Q;
        h >>= 1;
    }

    uint64_t result = lo_mod + hi_contrib;
    if (result >= Q) result -= Q;

    return result;
#endif
}

// Modular multiplication: (a * b) mod Q
inline uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t Q) {
    uint128_t prod = mul64x64(a, b);
    return mod_128_64(prod, Q);
}

// Modular power: base^exp mod Q
inline uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t Q) {
    uint64_t result = 1;
    base %= Q;
    while (exp > 0) {
        if (exp & 1) {
            result = mod_mul(result, base, Q);
        }
        exp >>= 1;
        base = mod_mul(base, base, Q);
    }
    return result;
}

// Modular inverse using extended Euclidean algorithm
inline uint64_t mod_inv(uint64_t a, uint64_t Q) {
    int64_t t = 0, newt = 1;
    int64_t r = Q, newr = a;

    while (newr != 0) {
        int64_t quotient = r / newr;
        int64_t temp = t;
        t = newt;
        newt = temp - quotient * newt;
        temp = r;
        r = newr;
        newr = temp - quotient * newr;
    }

    if (t < 0) t += Q;
    return (uint64_t)t;
}

// Find primitive N-th root of unity
inline uint64_t find_primitive_root(uint32_t N, uint64_t Q) {
    // For common NTT primes
    if (Q == 998244353ULL) {
        // Generator g = 3
        return mod_pow(3, (Q - 1) / N, Q);
    }
    if (Q == 7340033ULL) {
        return mod_pow(3, (Q - 1) / N, Q);
    }
    if (Q == 104857601ULL) {
        return mod_pow(3, (Q - 1) / N, Q);
    }

    // Generic search
    for (uint64_t g = 2; g < Q; g++) {
        uint64_t root = mod_pow(g, (Q - 1) / N, Q);
        if (mod_pow(root, N, Q) == 1 && mod_pow(root, N / 2, Q) != 1) {
            return root;
        }
    }
    return 0;
}

} // anonymous namespace

// NTT Context structure
struct MlxNTTContext {
    uint32_t N;
    uint64_t Q;
    uint64_t root;
    uint64_t inv_root;
    uint64_t inv_N;
    std::vector<uint64_t> twiddles;
    std::vector<uint64_t> inv_twiddles;

#ifdef HAS_METAL
    // Metal resources (for GPU path)
    void* metal_ctx;
#endif
};

extern "C" {

bool mlx_ntt_gpu_available() {
#ifdef HAS_METAL
    return mlx::core::metal::is_available();
#else
    return false;
#endif
}

const char* mlx_ntt_backend_name() {
#ifdef HAS_METAL
    if (mlx::core::metal::is_available()) {
        return "Metal (MLX)";
    }
#endif
#ifdef HAS_CUDA
    return "CUDA";
#endif
    return "CPU";
}

MlxNTTContext* mlx_ntt_create(uint32_t N, uint64_t Q) {
    // Validate N is power of 2
    if (N == 0 || (N & (N - 1)) != 0) {
        return nullptr;
    }

    // Validate Q is NTT-friendly: (Q - 1) divisible by 2*N
    if ((Q - 1) % (2 * N) != 0) {
        return nullptr;
    }

    MlxNTTContext* ctx = new MlxNTTContext();
    ctx->N = N;
    ctx->Q = Q;
    ctx->root = find_primitive_root(N, Q);
    ctx->inv_root = mod_inv(ctx->root, Q);
    ctx->inv_N = mod_inv(N, Q);

    // Precompute twiddle factors
    ctx->twiddles.resize(N);
    ctx->inv_twiddles.resize(N);

    ctx->twiddles[0] = 1;
    ctx->inv_twiddles[0] = 1;

    for (uint32_t i = 1; i < N; i++) {
        ctx->twiddles[i] = mod_mul(ctx->twiddles[i - 1], ctx->root, Q);
        ctx->inv_twiddles[i] = mod_mul(ctx->inv_twiddles[i - 1], ctx->inv_root, Q);
    }

#ifdef HAS_METAL
    ctx->metal_ctx = nullptr;  // TODO: Initialize Metal context
#endif

    return ctx;
}

void mlx_ntt_destroy(MlxNTTContext* ctx) {
    if (ctx) {
#ifdef HAS_METAL
        // TODO: Cleanup Metal resources
#endif
        delete ctx;
    }
}

// CPU NTT implementation (Cooley-Tukey decimation-in-time)
static void ntt_forward_cpu(MlxNTTContext* ctx, uint64_t* data, uint32_t batch) {
    uint32_t N = ctx->N;
    uint64_t Q = ctx->Q;

    for (uint32_t b = 0; b < batch; b++) {
        uint64_t* poly = data + b * N;

        for (uint32_t stage = 0; stage < 32 && (1u << stage) < N; stage++) {
            uint32_t m = 1u << (stage + 1);
            uint32_t half_m = m >> 1;

            for (uint32_t k = 0; k < N / 2; k++) {
                uint32_t j = k / half_m;
                uint32_t i = k % half_m;
                uint32_t idx0 = j * m + i;
                uint32_t idx1 = idx0 + half_m;

                uint64_t w = ctx->twiddles[half_m + i];
                uint64_t t = mod_mul(poly[idx1], w, Q);

                poly[idx1] = (poly[idx0] >= t) ? poly[idx0] - t : poly[idx0] + Q - t;
                poly[idx0] = (poly[idx0] + t >= Q) ? poly[idx0] + t - Q : poly[idx0] + t;
            }
        }
    }
}

// CPU inverse NTT implementation (Gentleman-Sande decimation-in-frequency)
static void ntt_inverse_cpu(MlxNTTContext* ctx, uint64_t* data, uint32_t batch) {
    uint32_t N = ctx->N;
    uint64_t Q = ctx->Q;

    for (uint32_t b = 0; b < batch; b++) {
        uint64_t* poly = data + b * N;

        uint32_t log_n = 0;
        for (uint32_t temp = N; temp > 1; temp >>= 1) log_n++;

        for (uint32_t stage = log_n; stage > 0; stage--) {
            uint32_t m = 1u << stage;
            uint32_t half_m = m >> 1;

            for (uint32_t k = 0; k < N / 2; k++) {
                uint32_t j = k / half_m;
                uint32_t i = k % half_m;
                uint32_t idx0 = j * m + i;
                uint32_t idx1 = idx0 + half_m;

                uint64_t x0 = poly[idx0];
                uint64_t x1 = poly[idx1];

                uint64_t t = (x0 >= x1) ? x0 - x1 : x0 + Q - x1;

                poly[idx0] = (x0 + x1 >= Q) ? x0 + x1 - Q : x0 + x1;
                poly[idx1] = mod_mul(t, ctx->inv_twiddles[half_m + i], Q);
            }
        }

        // Apply inverse scaling
        for (uint32_t i = 0; i < N; i++) {
            poly[i] = mod_mul(poly[i], ctx->inv_N, Q);
        }
    }
}

int mlx_ntt_forward(MlxNTTContext* ctx, uint64_t* data, uint32_t batch) {
    if (!ctx || !data || batch == 0) {
        return -1;
    }

#ifdef HAS_METAL
    // TODO: Use Metal GPU path when available
    // For now, fall back to CPU
#endif

    ntt_forward_cpu(ctx, data, batch);
    return 0;
}

int mlx_ntt_inverse(MlxNTTContext* ctx, uint64_t* data, uint32_t batch) {
    if (!ctx || !data || batch == 0) {
        return -1;
    }

#ifdef HAS_METAL
    // TODO: Use Metal GPU path when available
#endif

    ntt_inverse_cpu(ctx, data, batch);
    return 0;
}

int mlx_ntt_pointwise_mul(
    MlxNTTContext* ctx,
    uint64_t* result,
    const uint64_t* a,
    const uint64_t* b,
    uint32_t batch) {
    if (!ctx || !result || !a || !b) {
        return -1;
    }

    uint32_t N = ctx->N;
    uint64_t Q = ctx->Q;
    uint32_t size = batch * N;

    for (uint32_t i = 0; i < size; i++) {
        result[i] = mod_mul(a[i], b[i], Q);
    }

    return 0;
}

int mlx_ntt_polymul(
    MlxNTTContext* ctx,
    uint64_t* result,
    const uint64_t* a,
    const uint64_t* b,
    uint32_t batch) {
    if (!ctx || !result || !a || !b) {
        return -1;
    }

    uint32_t N = ctx->N;
    uint32_t size = batch * N;

    // Allocate temporary arrays
    std::vector<uint64_t> ntt_a(a, a + size);
    std::vector<uint64_t> ntt_b(b, b + size);

    // Forward NTT
    ntt_forward_cpu(ctx, ntt_a.data(), batch);
    ntt_forward_cpu(ctx, ntt_b.data(), batch);

    // Pointwise multiplication
    mlx_ntt_pointwise_mul(ctx, result, ntt_a.data(), ntt_b.data(), batch);

    // Inverse NTT
    ntt_inverse_cpu(ctx, result, batch);

    return 0;
}

} // extern "C"

