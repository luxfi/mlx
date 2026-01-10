// Copyright (c) 2024-2026 Lux Industries Inc. All Rights Reserved.
// PROPRIETARY AND CONFIDENTIAL - NO LICENSE GRANTED
// Contact: licensing@luxindustries.xyz
//
// Memory Fence and Synchronization Primitives - CUDA Implementation
// Provides memory barriers, atomic operations, and synchronization utilities
// for coordinating GPU compute across multiple kernels and streams.

#include <cuda_runtime.h>
#include <cstdint>

namespace lux {
namespace cuda {
namespace fence {

// ============================================================================
// Memory Fence Types
// ============================================================================

enum class FenceScope : uint32_t {
    BLOCK = 0,      // Synchronize within thread block
    DEVICE = 1,     // Synchronize across entire device
    SYSTEM = 2      // Synchronize across device and host
};

enum class FenceOrder : uint32_t {
    RELAXED = 0,    // No ordering constraints
    ACQUIRE = 1,    // Acquire semantics (loads after fence see prior stores)
    RELEASE = 2,    // Release semantics (stores before fence visible to subsequent loads)
    ACQ_REL = 3,    // Both acquire and release
    SEQ_CST = 4     // Sequentially consistent (total order)
};

// Fence descriptor for host-device coordination
struct FenceDescriptor {
    uint64_t fence_id;
    uint32_t scope;
    uint32_t order;
    volatile uint32_t* signal;
    uint32_t expected_value;
    uint32_t _padding;
};

// ============================================================================
// Thread Block Synchronization
// ============================================================================

// Standard block-level barrier
__device__ __forceinline__
void fence_block() {
    __syncthreads();
}

// Block-level barrier with predicate (returns true if all threads have true pred)
__device__ __forceinline__
bool fence_block_and(bool predicate) {
    return __syncthreads_and(predicate);
}

// Block-level barrier with predicate (returns true if any thread has true pred)
__device__ __forceinline__
bool fence_block_or(bool predicate) {
    return __syncthreads_or(predicate);
}

// Block-level barrier with count (returns count of threads with true pred)
__device__ __forceinline__
int fence_block_count(bool predicate) {
    return __syncthreads_count(predicate);
}

// ============================================================================
// Warp-Level Synchronization
// ============================================================================

// Warp-level barrier (all threads in warp must reach this point)
__device__ __forceinline__
void fence_warp() {
    __syncwarp();
}

// Warp-level barrier with mask (only specified threads participate)
__device__ __forceinline__
void fence_warp_masked(uint32_t mask) {
    __syncwarp(mask);
}

// ============================================================================
// Memory Fences
// ============================================================================

// Thread-fence at block scope
__device__ __forceinline__
void memory_fence_block() {
    __threadfence_block();
}

// Thread-fence at device scope
__device__ __forceinline__
void memory_fence_device() {
    __threadfence();
}

// Thread-fence at system scope (visible to host)
__device__ __forceinline__
void memory_fence_system() {
    __threadfence_system();
}

// Combined fence with scope selection
__device__ __forceinline__
void memory_fence(FenceScope scope) {
    switch (scope) {
        case FenceScope::BLOCK:
            __threadfence_block();
            break;
        case FenceScope::DEVICE:
            __threadfence();
            break;
        case FenceScope::SYSTEM:
            __threadfence_system();
            break;
    }
}

// ============================================================================
// Acquire-Release Semantics
// ============================================================================

// Load with acquire semantics
template<typename T>
__device__ __forceinline__
T load_acquire(const T* addr, FenceScope scope = FenceScope::DEVICE) {
    T value = *((volatile T*)addr);
    memory_fence(scope);
    return value;
}

// Store with release semantics
template<typename T>
__device__ __forceinline__
void store_release(T* addr, T value, FenceScope scope = FenceScope::DEVICE) {
    memory_fence(scope);
    *((volatile T*)addr) = value;
}

// Atomic load with ordering
template<typename T>
__device__ __forceinline__
T atomic_load(const T* addr, FenceOrder order = FenceOrder::SEQ_CST) {
    T value;
    if (order >= FenceOrder::ACQUIRE) {
        value = *((volatile T*)addr);
        __threadfence();
    } else {
        value = *((volatile T*)addr);
    }
    return value;
}

// Atomic store with ordering
template<typename T>
__device__ __forceinline__
void atomic_store(T* addr, T value, FenceOrder order = FenceOrder::SEQ_CST) {
    if (order >= FenceOrder::RELEASE) {
        __threadfence();
    }
    *((volatile T*)addr) = value;
    if (order == FenceOrder::SEQ_CST) {
        __threadfence();
    }
}

// ============================================================================
// Spinlock Implementation
// ============================================================================

// Simple spinlock
struct Spinlock {
    volatile uint32_t lock;
};

__device__ __forceinline__
void spinlock_init(Spinlock* lock) {
    lock->lock = 0;
}

__device__ __forceinline__
void spinlock_acquire(Spinlock* lock) {
    while (atomicCAS((uint32_t*)&lock->lock, 0, 1) != 0) {
        // Spin
    }
    __threadfence();
}

__device__ __forceinline__
bool spinlock_try_acquire(Spinlock* lock) {
    if (atomicCAS((uint32_t*)&lock->lock, 0, 1) == 0) {
        __threadfence();
        return true;
    }
    return false;
}

__device__ __forceinline__
void spinlock_release(Spinlock* lock) {
    __threadfence();
    lock->lock = 0;
}

// ============================================================================
// Semaphore Implementation
// ============================================================================

struct Semaphore {
    volatile int32_t count;
    int32_t max_count;
};

__device__ __forceinline__
void semaphore_init(Semaphore* sem, int32_t initial, int32_t max) {
    sem->count = initial;
    sem->max_count = max;
}

__device__ __forceinline__
void semaphore_wait(Semaphore* sem) {
    int32_t old;
    do {
        old = sem->count;
        if (old <= 0) continue;  // Spin if count is zero
    } while (atomicCAS((int*)&sem->count, old, old - 1) != old);
    __threadfence();
}

__device__ __forceinline__
bool semaphore_try_wait(Semaphore* sem) {
    int32_t old = sem->count;
    if (old <= 0) return false;
    if (atomicCAS((int*)&sem->count, old, old - 1) == old) {
        __threadfence();
        return true;
    }
    return false;
}

__device__ __forceinline__
void semaphore_signal(Semaphore* sem) {
    __threadfence();
    atomicAdd((int*)&sem->count, 1);
}

// ============================================================================
// Barrier Implementation (re-usable)
// ============================================================================

struct Barrier {
    volatile uint32_t count;
    volatile uint32_t generation;
    uint32_t threshold;
    uint32_t _padding;
};

__device__ __forceinline__
void barrier_init(Barrier* bar, uint32_t num_threads) {
    bar->count = 0;
    bar->generation = 0;
    bar->threshold = num_threads;
}

__device__ __forceinline__
void barrier_wait(Barrier* bar) {
    uint32_t gen = bar->generation;
    uint32_t arrived = atomicAdd((uint32_t*)&bar->count, 1) + 1;

    if (arrived == bar->threshold) {
        // Last thread to arrive
        bar->count = 0;
        __threadfence();
        atomicAdd((uint32_t*)&bar->generation, 1);
    } else {
        // Wait for generation to change
        while (bar->generation == gen) {
            // Spin
        }
    }
    __threadfence();
}

// ============================================================================
// Signal/Wait for Host-Device Coordination
// ============================================================================

// Signal kernel completion to host
__global__ void signal_fence_kernel(
    volatile uint32_t* __restrict__ signal,
    uint32_t value
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        __threadfence_system();
        *signal = value;
    }
}

// Wait for signal from host (polling kernel)
__global__ void wait_fence_kernel(
    volatile uint32_t* __restrict__ signal,
    uint32_t expected_value,
    uint32_t* __restrict__ result
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        while (*signal != expected_value) {
            // Spin
        }
        __threadfence_system();
        *result = 1;  // Indicate completion
    }
}

// ============================================================================
// Multi-Kernel Coordination
// ============================================================================

// Dependency descriptor
struct KernelDependency {
    volatile uint32_t* signal;
    uint32_t expected;
    uint32_t to_signal;
    uint32_t _padding;
};

// Wait for dependencies before executing kernel body
__device__ __forceinline__
void wait_for_dependencies(
    const KernelDependency* deps,
    uint32_t num_deps
) {
    if (threadIdx.x == 0) {
        for (uint32_t i = 0; i < num_deps; i++) {
            while (*deps[i].signal != deps[i].expected) {
                // Spin
            }
        }
        __threadfence();
    }
    __syncthreads();
}

// Signal completion of kernel
__device__ __forceinline__
void signal_completion(
    volatile uint32_t* signal,
    uint32_t value
) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        *signal = value;
    }
}

// ============================================================================
// Coalesced Memory Barrier
// ============================================================================

// Ensure all prior memory accesses from this warp are visible
__device__ __forceinline__
void coalesced_fence() {
    __syncwarp();
    __threadfence_block();
}

// ============================================================================
// Producer-Consumer Queue Utilities
// ============================================================================

struct RingBuffer {
    volatile uint64_t head;      // Write position (producer)
    volatile uint64_t tail;      // Read position (consumer)
    uint64_t capacity;
    uint64_t mask;               // capacity - 1 (for power-of-2 capacity)
};

__device__ __forceinline__
void ringbuffer_init(RingBuffer* rb, uint64_t capacity) {
    rb->head = 0;
    rb->tail = 0;
    rb->capacity = capacity;
    rb->mask = capacity - 1;
}

__device__ __forceinline__
bool ringbuffer_is_empty(const RingBuffer* rb) {
    return rb->head == rb->tail;
}

__device__ __forceinline__
bool ringbuffer_is_full(const RingBuffer* rb) {
    return (rb->head - rb->tail) >= rb->capacity;
}

__device__ __forceinline__
uint64_t ringbuffer_size(const RingBuffer* rb) {
    return rb->head - rb->tail;
}

// Reserve slot for producer (returns index or UINT64_MAX if full)
__device__ __forceinline__
uint64_t ringbuffer_reserve(RingBuffer* rb) {
    uint64_t head = rb->head;
    uint64_t tail = rb->tail;

    if ((head - tail) >= rb->capacity) {
        return UINT64_MAX;  // Full
    }

    uint64_t new_head = head + 1;
    if (atomicCAS((unsigned long long*)&rb->head,
                  (unsigned long long)head,
                  (unsigned long long)new_head) == head) {
        return head & rb->mask;
    }
    return UINT64_MAX;  // Contention, retry
}

// Commit producer slot (after writing data)
__device__ __forceinline__
void ringbuffer_commit(RingBuffer* rb) {
    __threadfence();
    // Head was already advanced in reserve
}

// Acquire consumer slot (returns index or UINT64_MAX if empty)
__device__ __forceinline__
uint64_t ringbuffer_acquire(RingBuffer* rb) {
    uint64_t tail = rb->tail;
    uint64_t head = rb->head;

    if (tail >= head) {
        return UINT64_MAX;  // Empty
    }

    uint64_t new_tail = tail + 1;
    if (atomicCAS((unsigned long long*)&rb->tail,
                  (unsigned long long)tail,
                  (unsigned long long)new_tail) == tail) {
        __threadfence();
        return tail & rb->mask;
    }
    return UINT64_MAX;  // Contention, retry
}

// ============================================================================
// Host API (C Interface)
// ============================================================================

} // namespace fence
} // namespace cuda
} // namespace lux

extern "C" {

using namespace lux::cuda::fence;

void lux_cuda_fence_signal(
    volatile uint32_t* signal,
    uint32_t value,
    cudaStream_t stream
) {
    signal_fence_kernel<<<1, 1, 0, stream>>>(signal, value);
}

void lux_cuda_fence_wait(
    volatile uint32_t* signal,
    uint32_t expected_value,
    uint32_t* result,
    cudaStream_t stream
) {
    wait_fence_kernel<<<1, 1, 0, stream>>>(signal, expected_value, result);
}

// Initialize spinlock on device
void lux_cuda_fence_spinlock_init(
    Spinlock* lock,
    cudaStream_t stream
) {
    uint32_t zero = 0;
    cudaMemcpyAsync(&lock->lock, &zero, sizeof(uint32_t),
                    cudaMemcpyHostToDevice, stream);
}

// Initialize semaphore on device
void lux_cuda_fence_semaphore_init(
    Semaphore* sem,
    int32_t initial,
    int32_t max,
    cudaStream_t stream
) {
    Semaphore host_sem = {initial, max};
    cudaMemcpyAsync(sem, &host_sem, sizeof(Semaphore),
                    cudaMemcpyHostToDevice, stream);
}

// Initialize barrier on device
void lux_cuda_fence_barrier_init(
    Barrier* bar,
    uint32_t num_threads,
    cudaStream_t stream
) {
    Barrier host_bar = {0, 0, num_threads, 0};
    cudaMemcpyAsync(bar, &host_bar, sizeof(Barrier),
                    cudaMemcpyHostToDevice, stream);
}

// Initialize ring buffer on device
void lux_cuda_fence_ringbuffer_init(
    RingBuffer* rb,
    uint64_t capacity,
    cudaStream_t stream
) {
    RingBuffer host_rb = {0, 0, capacity, capacity - 1};
    cudaMemcpyAsync(rb, &host_rb, sizeof(RingBuffer),
                    cudaMemcpyHostToDevice, stream);
}

// Host-side synchronization helper
void lux_cuda_fence_host_wait(
    volatile uint32_t* signal,
    uint32_t expected_value
) {
    while (*signal != expected_value) {
        // Spin with pause hint
        #if defined(__x86_64__) || defined(_M_X64)
        __asm__ __volatile__("pause" ::: "memory");
        #elif defined(__aarch64__)
        __asm__ __volatile__("yield" ::: "memory");
        #endif
    }
    __sync_synchronize();  // Full memory barrier
}

// Host-side signal
void lux_cuda_fence_host_signal(
    volatile uint32_t* signal,
    uint32_t value
) {
    __sync_synchronize();  // Full memory barrier
    *signal = value;
}

} // extern "C"
