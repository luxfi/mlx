// Copyright (c) 2024-2026 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// GPU/TEE Attestation Kernel for AI Mining (LP-2000)
// Implements NVIDIA NVTrust attestation data generation
// Extracts GPU identity, compute capability, memory status
// Generates cryptographic attestation reports for hardware verification

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

// ============================================================================
// Attestation Report Structure
// ============================================================================

// Maximum sizes
#define LUX_ATTEST_MAX_UUID_LEN      16
#define LUX_ATTEST_MAX_NAME_LEN      256
#define LUX_ATTEST_HASH_LEN          32
#define LUX_ATTEST_NONCE_LEN         32
#define LUX_ATTEST_SIGNATURE_LEN     64
#define LUX_ATTEST_REPORT_VERSION    1

// Privacy levels (from LP-2000)
#define LUX_ATTEST_LEVEL_PUBLIC       1   // Consumer GPU, stake-based
#define LUX_ATTEST_LEVEL_PRIVATE      2   // SGX/A100
#define LUX_ATTEST_LEVEL_CONFIDENTIAL 3   // H100/TDX/SEV
#define LUX_ATTEST_LEVEL_SOVEREIGN    4   // Blackwell, full TEE

// Error codes
#define LUX_ATTEST_OK                 0
#define LUX_ATTEST_ERR_NO_DEVICE      1
#define LUX_ATTEST_ERR_INVALID_PARAM  2
#define LUX_ATTEST_ERR_CUDA_FAILED    3
#define LUX_ATTEST_ERR_HASH_FAILED    4
#define LUX_ATTEST_ERR_NO_MEMORY      5

// GPU identity extracted from device
struct LuxGpuIdentity {
    uint8_t  uuid[LUX_ATTEST_MAX_UUID_LEN];  // Device UUID (16 bytes)
    char     name[LUX_ATTEST_MAX_NAME_LEN];  // Device name (e.g., "NVIDIA H100")
    uint32_t pci_bus_id;                      // PCI bus ID
    uint32_t pci_device_id;                   // PCI device ID
    uint32_t pci_domain_id;                   // PCI domain ID
};

// Compute capability and features
struct LuxComputeInfo {
    uint32_t major;                  // Compute capability major version
    uint32_t minor;                  // Compute capability minor version
    uint32_t multiprocessor_count;   // Number of SMs
    uint32_t max_threads_per_block;  // Max threads per block
    uint32_t max_threads_per_sm;     // Max threads per SM
    uint32_t warp_size;              // Warp size (typically 32)
    uint32_t clock_rate_khz;         // GPU clock rate in KHz
    uint32_t memory_clock_khz;       // Memory clock rate in KHz
    uint32_t driver_version;         // CUDA driver version
    uint32_t runtime_version;        // CUDA runtime version
    uint8_t  supports_tee;           // 1 if TEE/confidential compute supported
    uint8_t  supports_mig;           // 1 if MIG supported
    uint8_t  reserved[2];            // Padding
};

// Memory attestation
struct LuxMemoryInfo {
    uint64_t total_global_mem;       // Total global memory in bytes
    uint64_t free_global_mem;        // Free global memory at attestation time
    uint64_t total_constant_mem;     // Total constant memory
    uint32_t shared_mem_per_block;   // Shared memory per block
    uint32_t shared_mem_per_sm;      // Shared memory per SM
    uint32_t l2_cache_size;          // L2 cache size in bytes
    uint32_t memory_bus_width;       // Memory bus width in bits
    uint8_t  ecc_enabled;            // 1 if ECC enabled
    uint8_t  unified_addressing;     // 1 if unified addressing supported
    uint8_t  managed_memory;         // 1 if managed memory supported
    uint8_t  reserved;               // Padding
};

// Full attestation report
struct LuxAttestationReport {
    // Header
    uint32_t version;                // Report version (LUX_ATTEST_REPORT_VERSION)
    uint32_t report_size;            // Total size of this report
    uint64_t timestamp;              // Unix timestamp (seconds)
    uint64_t timestamp_ns;           // Nanosecond portion

    // Nonce for freshness
    uint8_t  nonce[LUX_ATTEST_NONCE_LEN];

    // Device info
    int32_t  device_id;              // CUDA device ID
    uint32_t privacy_level;          // Privacy/trust level (1-4)

    // Extracted data
    LuxGpuIdentity   identity;
    LuxComputeInfo   compute;
    LuxMemoryInfo    memory;

    // Cryptographic binding
    uint8_t  report_hash[LUX_ATTEST_HASH_LEN];    // SHA-256 of all above fields
    uint8_t  reserved[32];                         // Reserved for future use
};

// ============================================================================
// SHA-256 Implementation (Device)
// ============================================================================

// SHA-256 constants
__constant__ uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// Initial hash values
__constant__ uint32_t H256_INIT[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

// Rotate right
__device__ __forceinline__ uint32_t rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

// SHA-256 functions
__device__ __forceinline__ uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ uint32_t sigma0(uint32_t x) {
    return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22);
}

__device__ __forceinline__ uint32_t sigma1(uint32_t x) {
    return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25);
}

__device__ __forceinline__ uint32_t gamma0(uint32_t x) {
    return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ uint32_t gamma1(uint32_t x) {
    return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10);
}

// Process single SHA-256 block
__device__ void sha256_transform(uint32_t state[8], const uint8_t block[64]) {
    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h;

    // Parse block into 16 32-bit words (big-endian)
    for (int i = 0; i < 16; i++) {
        W[i] = ((uint32_t)block[i * 4 + 0] << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) |
               ((uint32_t)block[i * 4 + 3]);
    }

    // Extend 16 words to 64 words
    for (int i = 16; i < 64; i++) {
        W[i] = gamma1(W[i - 2]) + W[i - 7] + gamma0(W[i - 15]) + W[i - 16];
    }

    // Initialize working variables
    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];

    // 64 rounds
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + sigma1(e) + ch(e, f, g) + K256[i] + W[i];
        uint32_t t2 = sigma0(a) + maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    // Add to state
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

// Full SHA-256 hash (device-side, single-threaded)
__device__ void sha256_hash(const uint8_t* data, size_t len, uint8_t output[32]) {
    uint32_t state[8];
    for (int i = 0; i < 8; i++) {
        state[i] = H256_INIT[i];
    }

    // Process complete blocks
    size_t i = 0;
    while (i + 64 <= len) {
        sha256_transform(state, data + i);
        i += 64;
    }

    // Pad last block
    uint8_t block[64];
    size_t remaining = len - i;

    // Copy remaining bytes
    for (size_t j = 0; j < remaining; j++) {
        block[j] = data[i + j];
    }

    // Append 0x80
    block[remaining] = 0x80;
    remaining++;

    // Pad with zeros
    if (remaining <= 56) {
        for (size_t j = remaining; j < 56; j++) {
            block[j] = 0;
        }
    } else {
        // Need extra block
        for (size_t j = remaining; j < 64; j++) {
            block[j] = 0;
        }
        sha256_transform(state, block);
        for (int j = 0; j < 56; j++) {
            block[j] = 0;
        }
    }

    // Append length in bits (big-endian, 64-bit)
    uint64_t bit_len = len * 8;
    block[56] = (uint8_t)(bit_len >> 56);
    block[57] = (uint8_t)(bit_len >> 48);
    block[58] = (uint8_t)(bit_len >> 40);
    block[59] = (uint8_t)(bit_len >> 32);
    block[60] = (uint8_t)(bit_len >> 24);
    block[61] = (uint8_t)(bit_len >> 16);
    block[62] = (uint8_t)(bit_len >> 8);
    block[63] = (uint8_t)(bit_len);

    sha256_transform(state, block);

    // Output (big-endian)
    for (int j = 0; j < 8; j++) {
        output[j * 4 + 0] = (uint8_t)(state[j] >> 24);
        output[j * 4 + 1] = (uint8_t)(state[j] >> 16);
        output[j * 4 + 2] = (uint8_t)(state[j] >> 8);
        output[j * 4 + 3] = (uint8_t)(state[j]);
    }
}

// ============================================================================
// Attestation Report Hash Kernel
// ============================================================================

// Kernel to compute report hash on GPU
__global__ void compute_report_hash_kernel(
    const uint8_t* report_data,
    size_t data_len,
    uint8_t* hash_output
) {
    // Single thread computes hash
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        sha256_hash(report_data, data_len, hash_output);
    }
}

// ============================================================================
// Host-side Implementation
// ============================================================================

namespace lux {
namespace cuda {
namespace attestation {

// Host-side SHA-256 for systems without GPU hash
static void sha256_host(const uint8_t* data, size_t len, uint8_t output[32]) {
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

    auto rotr = [](uint32_t x, uint32_t n) -> uint32_t {
        return (x >> n) | (x << (32 - n));
    };

    auto ch = [](uint32_t x, uint32_t y, uint32_t z) -> uint32_t {
        return (x & y) ^ (~x & z);
    };

    auto maj = [](uint32_t x, uint32_t y, uint32_t z) -> uint32_t {
        return (x & y) ^ (x & z) ^ (y & z);
    };

    auto sigma0 = [&rotr](uint32_t x) -> uint32_t {
        return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
    };

    auto sigma1 = [&rotr](uint32_t x) -> uint32_t {
        return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
    };

    auto gamma0 = [&rotr](uint32_t x) -> uint32_t {
        return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
    };

    auto gamma1 = [&rotr](uint32_t x) -> uint32_t {
        return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
    };

    // Prepare padded message
    size_t padded_len = ((len + 9 + 63) / 64) * 64;
    uint8_t* padded = new uint8_t[padded_len];
    memcpy(padded, data, len);
    padded[len] = 0x80;
    memset(padded + len + 1, 0, padded_len - len - 9);

    uint64_t bit_len = len * 8;
    for (int i = 0; i < 8; i++) {
        padded[padded_len - 8 + i] = (uint8_t)(bit_len >> (56 - i * 8));
    }

    // Process blocks
    for (size_t blk = 0; blk < padded_len; blk += 64) {
        uint32_t W[64];
        for (int i = 0; i < 16; i++) {
            W[i] = ((uint32_t)padded[blk + i * 4 + 0] << 24) |
                   ((uint32_t)padded[blk + i * 4 + 1] << 16) |
                   ((uint32_t)padded[blk + i * 4 + 2] << 8) |
                   ((uint32_t)padded[blk + i * 4 + 3]);
        }
        for (int i = 16; i < 64; i++) {
            W[i] = gamma1(W[i - 2]) + W[i - 7] + gamma0(W[i - 15]) + W[i - 16];
        }

        uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
        uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

        for (int i = 0; i < 64; i++) {
            uint32_t t1 = h + sigma1(e) + ch(e, f, g) + k[i] + W[i];
            uint32_t t2 = sigma0(a) + maj(a, b, c);
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }

        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
    }

    delete[] padded;

    for (int i = 0; i < 8; i++) {
        output[i * 4 + 0] = (uint8_t)(state[i] >> 24);
        output[i * 4 + 1] = (uint8_t)(state[i] >> 16);
        output[i * 4 + 2] = (uint8_t)(state[i] >> 8);
        output[i * 4 + 3] = (uint8_t)(state[i]);
    }
}

// Determine privacy level from device properties
static uint32_t determine_privacy_level(const cudaDeviceProp& prop) {
    // Check compute capability and features
    int cc = prop.major * 10 + prop.minor;

    // Blackwell architecture (sm_100+) - Sovereign level
    if (cc >= 100) {
        return LUX_ATTEST_LEVEL_SOVEREIGN;
    }

    // Hopper architecture (sm_90) with CC mode - Confidential level
    // H100 with TDX/SEV support
    if (cc >= 90) {
        return LUX_ATTEST_LEVEL_CONFIDENTIAL;
    }

    // Ampere with SGX (A100) - Private level
    if (cc >= 80) {
        return LUX_ATTEST_LEVEL_PRIVATE;
    }

    // Everything else - Public level (stake-based only)
    return LUX_ATTEST_LEVEL_PUBLIC;
}

// Extract GPU identity from device
static void extract_gpu_identity(int device_id, LuxGpuIdentity* identity) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    // Copy UUID
    memcpy(identity->uuid, prop.uuid.bytes, LUX_ATTEST_MAX_UUID_LEN);

    // Copy name
    strncpy(identity->name, prop.name, LUX_ATTEST_MAX_NAME_LEN - 1);
    identity->name[LUX_ATTEST_MAX_NAME_LEN - 1] = '\0';

    // PCI info
    identity->pci_bus_id = prop.pciBusID;
    identity->pci_device_id = prop.pciDeviceID;
    identity->pci_domain_id = prop.pciDomainID;
}

// Extract compute capability info
static void extract_compute_info(int device_id, LuxComputeInfo* compute) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    compute->major = prop.major;
    compute->minor = prop.minor;
    compute->multiprocessor_count = prop.multiProcessorCount;
    compute->max_threads_per_block = prop.maxThreadsPerBlock;
    compute->max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    compute->warp_size = prop.warpSize;
    compute->clock_rate_khz = prop.clockRate;
    compute->memory_clock_khz = prop.memoryClockRate;

    // Get driver and runtime versions
    int driver_version = 0, runtime_version = 0;
    cudaDriverGetVersion(&driver_version);
    cudaRuntimeGetVersion(&runtime_version);
    compute->driver_version = driver_version;
    compute->runtime_version = runtime_version;

    // Check for TEE/CC support (Hopper+ with CC mode)
    compute->supports_tee = (prop.major >= 9) ? 1 : 0;

    // Check for MIG support (Ampere+)
    compute->supports_mig = (prop.major >= 8) ? 1 : 0;

    compute->reserved[0] = 0;
    compute->reserved[1] = 0;
}

// Extract memory info
static void extract_memory_info(int device_id, LuxMemoryInfo* memory) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    memory->total_global_mem = prop.totalGlobalMem;

    // Get current free memory
    size_t free_mem = 0, total_mem = 0;
    cudaSetDevice(device_id);
    cudaMemGetInfo(&free_mem, &total_mem);
    memory->free_global_mem = free_mem;

    memory->total_constant_mem = prop.totalConstMem;
    memory->shared_mem_per_block = prop.sharedMemPerBlock;
    memory->shared_mem_per_sm = prop.sharedMemPerMultiprocessor;
    memory->l2_cache_size = prop.l2CacheSize;
    memory->memory_bus_width = prop.memoryBusWidth;

    // ECC status
    memory->ecc_enabled = prop.ECCEnabled ? 1 : 0;

    // Unified addressing
    memory->unified_addressing = prop.unifiedAddressing ? 1 : 0;

    // Managed memory
    memory->managed_memory = prop.managedMemory ? 1 : 0;

    memory->reserved = 0;
}

// Get current timestamp
static void get_timestamp(uint64_t* seconds, uint64_t* nanoseconds) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    *seconds = ts.tv_sec;
    *nanoseconds = ts.tv_nsec;
}

} // namespace attestation
} // namespace cuda
} // namespace lux

// ============================================================================
// C API Implementation
// ============================================================================

extern "C" {

// Get the number of available CUDA devices
int lux_cuda_attestation_get_device_count(int* count) {
    if (!count) {
        return LUX_ATTEST_ERR_INVALID_PARAM;
    }

    cudaError_t err = cudaGetDeviceCount(count);
    if (err != cudaSuccess) {
        *count = 0;
        return LUX_ATTEST_ERR_NO_DEVICE;
    }

    return LUX_ATTEST_OK;
}

// Generate attestation report for a specific device
int lux_cuda_attestation_generate_report(
    int device_id,
    const uint8_t* nonce,
    size_t nonce_len,
    LuxAttestationReport* report
) {
    using namespace lux::cuda::attestation;

    if (!report) {
        return LUX_ATTEST_ERR_INVALID_PARAM;
    }

    // Check nonce
    if (nonce && nonce_len > LUX_ATTEST_NONCE_LEN) {
        return LUX_ATTEST_ERR_INVALID_PARAM;
    }

    // Check device exists
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_id >= device_count || device_id < 0) {
        return LUX_ATTEST_ERR_NO_DEVICE;
    }

    // Clear report
    memset(report, 0, sizeof(LuxAttestationReport));

    // Fill header
    report->version = LUX_ATTEST_REPORT_VERSION;
    report->report_size = sizeof(LuxAttestationReport);
    get_timestamp(&report->timestamp, &report->timestamp_ns);

    // Copy nonce
    if (nonce && nonce_len > 0) {
        memcpy(report->nonce, nonce, nonce_len);
    }

    // Device ID and privacy level
    report->device_id = device_id;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    report->privacy_level = determine_privacy_level(prop);

    // Extract device info
    extract_gpu_identity(device_id, &report->identity);
    extract_compute_info(device_id, &report->compute);
    extract_memory_info(device_id, &report->memory);

    // Compute hash of everything except the hash field itself
    size_t hash_input_len = offsetof(LuxAttestationReport, report_hash);
    sha256_host((const uint8_t*)report, hash_input_len, report->report_hash);

    return LUX_ATTEST_OK;
}

// Generate report with GPU-computed hash (requires device access)
int lux_cuda_attestation_generate_report_gpu(
    int device_id,
    const uint8_t* nonce,
    size_t nonce_len,
    LuxAttestationReport* report
) {
    using namespace lux::cuda::attestation;

    if (!report) {
        return LUX_ATTEST_ERR_INVALID_PARAM;
    }

    // First generate report with host-side hash
    int result = lux_cuda_attestation_generate_report(device_id, nonce, nonce_len, report);
    if (result != LUX_ATTEST_OK) {
        return result;
    }

    // Now recompute hash on GPU
    cudaSetDevice(device_id);

    uint8_t* d_report = nullptr;
    uint8_t* d_hash = nullptr;
    size_t hash_input_len = offsetof(LuxAttestationReport, report_hash);

    cudaError_t err = cudaMalloc(&d_report, hash_input_len);
    if (err != cudaSuccess) {
        return LUX_ATTEST_ERR_NO_MEMORY;
    }

    err = cudaMalloc(&d_hash, LUX_ATTEST_HASH_LEN);
    if (err != cudaSuccess) {
        cudaFree(d_report);
        return LUX_ATTEST_ERR_NO_MEMORY;
    }

    // Copy report data to device
    err = cudaMemcpy(d_report, report, hash_input_len, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_report);
        cudaFree(d_hash);
        return LUX_ATTEST_ERR_CUDA_FAILED;
    }

    // Compute hash on GPU
    compute_report_hash_kernel<<<1, 1>>>(d_report, hash_input_len, d_hash);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_report);
        cudaFree(d_hash);
        return LUX_ATTEST_ERR_HASH_FAILED;
    }

    cudaDeviceSynchronize();

    // Copy hash back
    err = cudaMemcpy(report->report_hash, d_hash, LUX_ATTEST_HASH_LEN, cudaMemcpyDeviceToHost);

    cudaFree(d_report);
    cudaFree(d_hash);

    if (err != cudaSuccess) {
        return LUX_ATTEST_ERR_CUDA_FAILED;
    }

    return LUX_ATTEST_OK;
}

// Verify attestation report hash
int lux_cuda_attestation_verify_report(const LuxAttestationReport* report) {
    using namespace lux::cuda::attestation;

    if (!report) {
        return LUX_ATTEST_ERR_INVALID_PARAM;
    }

    // Check version
    if (report->version != LUX_ATTEST_REPORT_VERSION) {
        return LUX_ATTEST_ERR_INVALID_PARAM;
    }

    // Recompute hash
    uint8_t computed_hash[LUX_ATTEST_HASH_LEN];
    size_t hash_input_len = offsetof(LuxAttestationReport, report_hash);
    sha256_host((const uint8_t*)report, hash_input_len, computed_hash);

    // Compare
    if (memcmp(computed_hash, report->report_hash, LUX_ATTEST_HASH_LEN) != 0) {
        return LUX_ATTEST_ERR_HASH_FAILED;
    }

    return LUX_ATTEST_OK;
}

// Get privacy level name
const char* lux_cuda_attestation_privacy_level_name(uint32_t level) {
    switch (level) {
        case LUX_ATTEST_LEVEL_PUBLIC:       return "Public";
        case LUX_ATTEST_LEVEL_PRIVATE:      return "Private";
        case LUX_ATTEST_LEVEL_CONFIDENTIAL: return "Confidential";
        case LUX_ATTEST_LEVEL_SOVEREIGN:    return "Sovereign";
        default:                             return "Unknown";
    }
}

// Get credits per minute for privacy level (from LP-2000)
float lux_cuda_attestation_credits_per_minute(uint32_t level) {
    switch (level) {
        case LUX_ATTEST_LEVEL_PUBLIC:       return 0.25f;
        case LUX_ATTEST_LEVEL_PRIVATE:      return 0.50f;
        case LUX_ATTEST_LEVEL_CONFIDENTIAL: return 1.00f;
        case LUX_ATTEST_LEVEL_SOVEREIGN:    return 1.50f;
        default:                             return 0.0f;
    }
}

// Serialize report to bytes
int lux_cuda_attestation_serialize_report(
    const LuxAttestationReport* report,
    uint8_t* buffer,
    size_t buffer_len,
    size_t* written
) {
    if (!report || !written) {
        return LUX_ATTEST_ERR_INVALID_PARAM;
    }

    *written = sizeof(LuxAttestationReport);

    if (buffer && buffer_len >= sizeof(LuxAttestationReport)) {
        memcpy(buffer, report, sizeof(LuxAttestationReport));
    } else if (buffer) {
        return LUX_ATTEST_ERR_INVALID_PARAM;
    }

    return LUX_ATTEST_OK;
}

// Deserialize report from bytes
int lux_cuda_attestation_deserialize_report(
    const uint8_t* buffer,
    size_t buffer_len,
    LuxAttestationReport* report
) {
    if (!buffer || !report || buffer_len < sizeof(LuxAttestationReport)) {
        return LUX_ATTEST_ERR_INVALID_PARAM;
    }

    memcpy(report, buffer, sizeof(LuxAttestationReport));

    // Verify after deserialize
    return lux_cuda_attestation_verify_report(report);
}

// Get report size
size_t lux_cuda_attestation_report_size(void) {
    return sizeof(LuxAttestationReport);
}

// Quick device check (can this device attest?)
int lux_cuda_attestation_can_attest(int device_id, int* can_attest) {
    if (!can_attest) {
        return LUX_ATTEST_ERR_INVALID_PARAM;
    }

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_id >= device_count || device_id < 0) {
        *can_attest = 0;
        return LUX_ATTEST_OK;
    }

    // All CUDA devices can attest (at minimum Public level)
    *can_attest = 1;
    return LUX_ATTEST_OK;
}

// Get device privacy level without full report
int lux_cuda_attestation_get_privacy_level(int device_id, uint32_t* level) {
    if (!level) {
        return LUX_ATTEST_ERR_INVALID_PARAM;
    }

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_id >= device_count || device_id < 0) {
        return LUX_ATTEST_ERR_NO_DEVICE;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    *level = lux::cuda::attestation::determine_privacy_level(prop);

    return LUX_ATTEST_OK;
}

} // extern "C"
