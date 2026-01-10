// Copyright (c) 2024-2025 Lux Partners Limited
// SPDX-License-Identifier: BSD-3-Clause
//
// TFHE Bootstrapping Key Prefetch CUDA Kernels
// Implements prefetching strategies for efficient bootstrapping key access.
//
// The bootstrapping key (BSK) is the dominant memory consumer in TFHE:
//   BSK size = n * (k+1) * L * (k+1) * N * 8 bytes
//   For n=630, k=1, L=3, N=1024: ~77 MB per BSK
//
// This module provides:
// 1. Asynchronous prefetch of BSK entries needed for upcoming blind rotation steps
// 2. Tile-based loading for L2/shared memory caching
// 3. Multi-stream orchestration for overlapping compute and memory

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================================
// Constants
// ============================================================================

#define MAX_N 2048
#define MAX_K 2
#define MAX_L 8
#define WARP_SIZE 32
#define CACHE_LINE_SIZE 128  // Bytes

// ============================================================================
// Prefetch Hint Instructions
// ============================================================================

// Prefetch to L2 cache (non-temporal)
__device__ __forceinline__
void prefetch_l2(const void* ptr) {
    asm volatile("prefetch.global.L2 [%0];" : : "l"(ptr));
}

// Prefetch to L1 cache
__device__ __forceinline__
void prefetch_l1(const void* ptr) {
    asm volatile("prefetch.global.L1 [%0];" : : "l"(ptr));
}

// ============================================================================
// BSK Layout Information
// ============================================================================

struct BSKLayout {
    uint32_t n;           // LWE dimension (number of BSK entries)
    uint32_t N;           // GLWE polynomial degree
    uint32_t k;           // GLWE dimension
    uint32_t L;           // Decomposition levels
    uint64_t entry_size;  // Size of one BSK entry in elements: (k+1) * L * (k+1) * N
    uint64_t total_size;  // Total BSK size in elements: n * entry_size
};

__device__ __forceinline__
uint64_t bsk_entry_offset(const BSKLayout& layout, uint32_t entry_idx) {
    return (uint64_t)entry_idx * layout.entry_size;
}

__device__ __forceinline__
uint64_t bsk_ggsw_offset(
    const BSKLayout& layout,
    uint32_t entry_idx,
    uint32_t in_poly,
    uint32_t level,
    uint32_t out_poly
) {
    // BSK[entry_idx][in_poly][level][out_poly][N]
    uint64_t base = bsk_entry_offset(layout, entry_idx);
    uint64_t offset = ((in_poly * layout.L + level) * (layout.k + 1) + out_poly) * layout.N;
    return base + offset;
}

// ============================================================================
// Prefetch Strategies
// ============================================================================

// Strategy 1: Prefetch entire next BSK entry
// Simple but requires large L2 cache

extern "C" __global__
void bsk_prefetch_entry(
    const uint64_t* __restrict__ bsk,
    uint32_t entry_idx,
    const BSKLayout layout
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;

    uint64_t offset = bsk_entry_offset(layout, entry_idx);
    const uint64_t* entry_ptr = bsk + offset;
    uint64_t entry_size = layout.entry_size;

    // Prefetch in cache-line sized chunks
    for (uint64_t i = tid; i < entry_size; i += stride) {
        prefetch_l2(&entry_ptr[i]);
    }
}

// Strategy 2: Prefetch specific GGSW rows needed for one blind rotation step
// More targeted, works with smaller cache

extern "C" __global__
void bsk_prefetch_ggsw_row(
    const uint64_t* __restrict__ bsk,
    uint32_t entry_idx,
    uint32_t out_poly_idx,     // Which output polynomial we're computing
    const BSKLayout layout
) {
    const uint32_t tid = threadIdx.x;
    const uint32_t N = layout.N;
    const uint32_t L = layout.L;
    const uint32_t k = layout.k;

    // Prefetch all levels for this output polynomial
    for (uint32_t in_poly = 0; in_poly <= k; ++in_poly) {
        for (uint32_t level = 0; level < L; ++level) {
            uint64_t offset = bsk_ggsw_offset(layout, entry_idx, in_poly, level, out_poly_idx);
            const uint64_t* row_ptr = bsk + offset;

            // Prefetch N elements in chunks
            for (uint32_t i = tid; i < N; i += blockDim.x) {
                prefetch_l2(&row_ptr[i]);
            }
        }
    }
}

// Strategy 3: Double-buffer prefetch with async copy
// Uses CUDA async copy for concurrent prefetch

#if __CUDA_ARCH__ >= 800  // Ampere and later

extern "C" __global__
void bsk_prefetch_async(
    uint64_t* __restrict__ shared_buffer,        // Shared memory buffer for BSK entry
    const uint64_t* __restrict__ bsk,
    uint32_t entry_idx,
    const BSKLayout layout
) {
    extern __shared__ uint64_t buffer[];

    const uint32_t tid = threadIdx.x;
    const uint32_t tpg = blockDim.x;

    uint64_t offset = bsk_entry_offset(layout, entry_idx);
    const uint64_t* entry_ptr = bsk + offset;
    uint64_t entry_size = layout.entry_size;

    // Use cp.async for non-blocking copy
    for (uint64_t i = tid; i < entry_size; i += tpg) {
        __pipeline_memcpy_async(&buffer[i], &entry_ptr[i], sizeof(uint64_t));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);  // Wait for all copies

    // Write to shared buffer (can be picked up by other kernels)
    __syncthreads();
}

#endif

// ============================================================================
// Multi-Entry Prefetch (Pipeline Multiple Steps)
// ============================================================================

// Prefetch multiple BSK entries in a pipeline fashion
// Call at the start of blind rotation to warm cache

extern "C" __global__
void bsk_prefetch_pipeline(
    const uint64_t* __restrict__ bsk,
    const uint32_t* __restrict__ entry_indices,  // Which entries to prefetch
    uint32_t num_entries,
    uint32_t prefetch_depth,                     // How many to prefetch ahead
    const BSKLayout layout
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;

    // Each block handles one entry
    for (uint32_t e = blockIdx.x; e < num_entries && e < prefetch_depth; e += gridDim.x) {
        uint32_t entry_idx = entry_indices[e];
        uint64_t offset = bsk_entry_offset(layout, entry_idx);
        const uint64_t* entry_ptr = bsk + offset;
        uint64_t entry_size = layout.entry_size;

        for (uint64_t i = threadIdx.x; i < entry_size; i += blockDim.x) {
            prefetch_l2(&entry_ptr[i]);
        }
    }
}

// ============================================================================
// BSK Tile Loading (For Shared Memory Based Bootstrap)
// ============================================================================

// Load BSK tile into shared memory for faster access during external product

struct BSKTileParams {
    uint32_t entry_idx;       // Which BSK entry
    uint32_t level_start;     // Starting decomposition level
    uint32_t level_count;     // Number of levels in this tile
    uint32_t poly_start;      // Starting polynomial index
    uint32_t poly_count;      // Number of polynomials in this tile
};

extern "C" __global__
void bsk_load_tile(
    uint64_t* __restrict__ tile_out,             // [level_count, poly_count, N]
    const uint64_t* __restrict__ bsk,
    BSKTileParams tile_params,
    uint32_t out_poly_idx,
    const BSKLayout layout
) {
    const uint32_t N = layout.N;
    const uint32_t tid = threadIdx.x;
    const uint32_t tpg = blockDim.x;

    uint32_t entry_idx = tile_params.entry_idx;
    uint32_t level_start = tile_params.level_start;
    uint32_t level_count = tile_params.level_count;
    uint32_t poly_start = tile_params.poly_start;
    uint32_t poly_count = tile_params.poly_count;

    for (uint32_t in_poly_off = 0; in_poly_off < poly_count; ++in_poly_off) {
        uint32_t in_poly = poly_start + in_poly_off;
        if (in_poly > layout.k) break;

        for (uint32_t level_off = 0; level_off < level_count; ++level_off) {
            uint32_t level = level_start + level_off;
            if (level >= layout.L) break;

            // Source offset in BSK
            uint64_t src_offset = bsk_ggsw_offset(layout, entry_idx, in_poly, level, out_poly_idx);
            const uint64_t* src = bsk + src_offset;

            // Destination offset in tile
            uint64_t dst_offset = (level_off * poly_count + in_poly_off) * N;
            uint64_t* dst = tile_out + dst_offset;

            // Coalesced copy
            for (uint32_t i = tid; i < N; i += tpg) {
                dst[i] = src[i];
            }
        }
    }
}

// ============================================================================
// Streaming BSK Access (For Very Large Keys)
// ============================================================================

// For bootstrapping keys that don't fit in L2, stream from HBM with prefetch

struct StreamingBSKContext {
    const uint64_t* bsk_ptr;
    BSKLayout layout;
    uint32_t current_entry;
    uint32_t prefetch_distance;  // How many entries ahead to prefetch
};

extern "C" __global__
void bsk_stream_step(
    uint64_t* __restrict__ entry_buffer,         // Buffer for current entry [entry_size]
    uint64_t* __restrict__ prefetch_buffer,      // Buffer for prefetch [entry_size]
    const StreamingBSKContext ctx,
    uint32_t step_idx
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;

    // Load current entry
    if (step_idx < ctx.layout.n) {
        uint64_t offset = bsk_entry_offset(ctx.layout, step_idx);
        const uint64_t* src = ctx.bsk_ptr + offset;

        for (uint64_t i = tid; i < ctx.layout.entry_size; i += stride) {
            entry_buffer[i] = __ldg(&src[i]);
        }
    }

    // Prefetch next entry
    uint32_t prefetch_idx = step_idx + ctx.prefetch_distance;
    if (prefetch_idx < ctx.layout.n) {
        uint64_t offset = bsk_entry_offset(ctx.layout, prefetch_idx);
        const uint64_t* prefetch_ptr = ctx.bsk_ptr + offset;

        for (uint64_t i = tid; i < ctx.layout.entry_size; i += stride) {
            prefetch_l2(&prefetch_ptr[i]);
        }
    }
}

// ============================================================================
// BSK Compression and Decompression
// ============================================================================

// Parameters for compressed BSK (quantized coefficients)
struct CompressedBSKParams {
    uint32_t bits_per_coeff;   // Bits per coefficient (e.g., 16 for half precision)
    uint64_t scale;            // Scale factor for dequantization
    int64_t offset;            // Offset for dequantization
};

// Decompress BSK entry on-the-fly
extern "C" __global__
void bsk_decompress_entry(
    uint64_t* __restrict__ bsk_out,              // Decompressed entry [entry_size]
    const uint8_t* __restrict__ bsk_compressed,  // Compressed BSK
    uint32_t entry_idx,
    const BSKLayout layout,
    const CompressedBSKParams compress_params
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;

    uint64_t entry_size = layout.entry_size;
    uint32_t bits = compress_params.bits_per_coeff;
    uint64_t scale = compress_params.scale;
    int64_t offset = compress_params.offset;

    // Compressed entry offset (bit-packed)
    uint64_t compressed_entry_bits = entry_size * bits;
    uint64_t compressed_entry_bytes = (compressed_entry_bits + 7) / 8;
    const uint8_t* compressed_entry = bsk_compressed + entry_idx * compressed_entry_bytes;

    for (uint64_t i = tid; i < entry_size; i += stride) {
        // Extract bits for coefficient i
        uint64_t bit_offset = i * bits;
        uint64_t byte_offset = bit_offset / 8;
        uint32_t bit_shift = bit_offset % 8;

        // Read bytes and extract value
        uint32_t value = 0;
        for (uint32_t b = 0; b < (bits + 7) / 8 + 1 && byte_offset + b < compressed_entry_bytes; ++b) {
            value |= ((uint32_t)compressed_entry[byte_offset + b]) << (8 * b);
        }
        value >>= bit_shift;
        value &= (1U << bits) - 1;

        // Dequantize
        bsk_out[i] = (uint64_t)((int64_t)value * (int64_t)scale + offset);
    }
}

// ============================================================================
// Multi-GPU BSK Distribution
// ============================================================================

// Distribute BSK across multiple GPUs for parallel bootstrapping
struct MultiGPUBSKParams {
    uint32_t num_gpus;
    uint32_t gpu_id;
    uint32_t entries_per_gpu;  // n / num_gpus (rounded up)
    uint32_t start_entry;      // First entry on this GPU
    uint32_t end_entry;        // Last entry + 1 on this GPU
};

// Initialize local portion of BSK on this GPU
extern "C" __global__
void bsk_distribute_init(
    uint64_t* __restrict__ local_bsk,            // Local BSK portion
    const uint64_t* __restrict__ full_bsk,       // Full BSK (on host or peer GPU)
    const MultiGPUBSKParams mgpu_params,
    const BSKLayout layout
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;

    uint32_t start = mgpu_params.start_entry;
    uint32_t end = mgpu_params.end_entry;
    uint64_t local_size = (end - start) * layout.entry_size;

    // Copy from full BSK to local portion
    uint64_t global_offset = bsk_entry_offset(layout, start);

    for (uint64_t i = tid; i < local_size; i += stride) {
        local_bsk[i] = full_bsk[global_offset + i];
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

// Prefetch single BSK entry to L2
cudaError_t lux_cuda_fhe_bsk_prefetch_entry(
    const uint64_t* bsk,
    uint32_t entry_idx,
    uint32_t n,
    uint32_t N,
    uint32_t k,
    uint32_t L,
    cudaStream_t stream
) {
    BSKLayout layout;
    layout.n = n;
    layout.N = N;
    layout.k = k;
    layout.L = L;
    layout.entry_size = (uint64_t)(k + 1) * L * (k + 1) * N;
    layout.total_size = layout.n * layout.entry_size;

    // Launch enough threads to cover the entry
    uint32_t threads = 256;
    uint32_t blocks = (layout.entry_size + threads - 1) / threads;
    blocks = min(blocks, 1024u);

    bsk_prefetch_entry<<<blocks, threads, 0, stream>>>(bsk, entry_idx, layout);

    return cudaGetLastError();
}

// Prefetch multiple entries in pipeline
cudaError_t lux_cuda_fhe_bsk_prefetch_pipeline(
    const uint64_t* bsk,
    const uint32_t* entry_indices,
    uint32_t num_entries,
    uint32_t prefetch_depth,
    uint32_t n,
    uint32_t N,
    uint32_t k,
    uint32_t L,
    cudaStream_t stream
) {
    BSKLayout layout;
    layout.n = n;
    layout.N = N;
    layout.k = k;
    layout.L = L;
    layout.entry_size = (uint64_t)(k + 1) * L * (k + 1) * N;
    layout.total_size = layout.n * layout.entry_size;

    uint32_t threads = 256;
    uint32_t blocks = min(prefetch_depth, 64u);

    bsk_prefetch_pipeline<<<blocks, threads, 0, stream>>>(
        bsk, entry_indices, num_entries, prefetch_depth, layout
    );

    return cudaGetLastError();
}

// Load BSK tile to buffer
cudaError_t lux_cuda_fhe_bsk_load_tile(
    uint64_t* tile_out,
    const uint64_t* bsk,
    uint32_t entry_idx,
    uint32_t out_poly_idx,
    uint32_t level_start,
    uint32_t level_count,
    uint32_t poly_start,
    uint32_t poly_count,
    uint32_t n,
    uint32_t N,
    uint32_t k,
    uint32_t L,
    cudaStream_t stream
) {
    BSKLayout layout;
    layout.n = n;
    layout.N = N;
    layout.k = k;
    layout.L = L;
    layout.entry_size = (uint64_t)(k + 1) * L * (k + 1) * N;
    layout.total_size = layout.n * layout.entry_size;

    BSKTileParams tile_params;
    tile_params.entry_idx = entry_idx;
    tile_params.level_start = level_start;
    tile_params.level_count = level_count;
    tile_params.poly_start = poly_start;
    tile_params.poly_count = poly_count;

    dim3 block(min(N, 256u));
    dim3 grid(1);

    bsk_load_tile<<<grid, block, 0, stream>>>(
        tile_out, bsk, tile_params, out_poly_idx, layout
    );

    return cudaGetLastError();
}

// Decompress BSK entry
cudaError_t lux_cuda_fhe_bsk_decompress(
    uint64_t* bsk_out,
    const uint8_t* bsk_compressed,
    uint32_t entry_idx,
    uint32_t bits_per_coeff,
    uint64_t scale,
    int64_t offset,
    uint32_t n,
    uint32_t N,
    uint32_t k,
    uint32_t L,
    cudaStream_t stream
) {
    BSKLayout layout;
    layout.n = n;
    layout.N = N;
    layout.k = k;
    layout.L = L;
    layout.entry_size = (uint64_t)(k + 1) * L * (k + 1) * N;
    layout.total_size = layout.n * layout.entry_size;

    CompressedBSKParams compress_params;
    compress_params.bits_per_coeff = bits_per_coeff;
    compress_params.scale = scale;
    compress_params.offset = offset;

    uint32_t threads = 256;
    uint32_t blocks = (layout.entry_size + threads - 1) / threads;
    blocks = min(blocks, 1024u);

    bsk_decompress_entry<<<blocks, threads, 0, stream>>>(
        bsk_out, bsk_compressed, entry_idx, layout, compress_params
    );

    return cudaGetLastError();
}

// Multi-GPU BSK distribution
cudaError_t lux_cuda_fhe_bsk_distribute(
    uint64_t* local_bsk,
    const uint64_t* full_bsk,
    uint32_t gpu_id,
    uint32_t num_gpus,
    uint32_t n,
    uint32_t N,
    uint32_t k,
    uint32_t L,
    cudaStream_t stream
) {
    BSKLayout layout;
    layout.n = n;
    layout.N = N;
    layout.k = k;
    layout.L = L;
    layout.entry_size = (uint64_t)(k + 1) * L * (k + 1) * N;
    layout.total_size = layout.n * layout.entry_size;

    uint32_t entries_per_gpu = (n + num_gpus - 1) / num_gpus;
    uint32_t start_entry = gpu_id * entries_per_gpu;
    uint32_t end_entry = min(start_entry + entries_per_gpu, n);

    MultiGPUBSKParams mgpu_params;
    mgpu_params.num_gpus = num_gpus;
    mgpu_params.gpu_id = gpu_id;
    mgpu_params.entries_per_gpu = entries_per_gpu;
    mgpu_params.start_entry = start_entry;
    mgpu_params.end_entry = end_entry;

    uint64_t local_size = (end_entry - start_entry) * layout.entry_size;
    uint32_t threads = 256;
    uint32_t blocks = (local_size + threads - 1) / threads;
    blocks = min(blocks, 65535u);

    bsk_distribute_init<<<blocks, threads, 0, stream>>>(
        local_bsk, full_bsk, mgpu_params, layout
    );

    return cudaGetLastError();
}

}  // extern "C"
