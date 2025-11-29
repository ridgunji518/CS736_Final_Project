#ifndef LOCKERS_CUH
#define LOCKERS_CUH
#include <cuda_runtime.h>

__device__ __forceinline__ unsigned int get_random_delay(unsigned int attempt, int tid){
    unsigned long long time = clock64();
    unsigned int seed = (unsigned int)time ^ tid;
    unsigned int cap = 1024;
    unsigned int limit = 32 * (1 << (attempt < 10 ? attempt : 10));
    if (limit > cap) limit = cap;
    
    return (seed % limit) + 1;
}

__device__ __forceinline__ void acquire_lock_spin(int *lock) {
    while (atomicCAS(lock, 0, 1) != 0) {
        // busy wait
    }
}

__device__ __forceinline__ void acquire_lock_exponential(int *lock) {
    unsigned delay = 2;
    while (atomicCAS(lock, 0, 1) != 0) {
        unsigned long long start = clock64();
        while (clock64() - start < delay);   // Busy waiting delay
        delay = min(delay * 2, 1024u);       // Exponential Backoff
    }
}

__device__ __forceinline__ void acquire_lock_adaptive(int* mutex) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int attempt = 0;

    while (true) {
        // --- policy 1: Warp aggregation ---
        // check in the current warp which thread will attempt to acquire the lock
        unsigned int mask = __activemask();
        // select the thread with the smallest ID as the leader
        int leader = __ffs(mask) - 1; 
        int lane_id = threadIdx.x % 32;

        bool is_leader = (lane_id == leader);
        int lock_result = 0;

        if (is_leader) {
            // --- policy 2: TTAS (Test-and-Test-and-Set) ---
            // Only attempt atomic operation if it appears to be 0
            // The volatile here forces a read from memory each time
            if (*(volatile int*)mutex == 0) {
                lock_result = atomicCAS(mutex, 0, 1);
            } else {
                lock_result = 1; // Looks locked, treat as failure
            }
        }

        // Leader broadcasts the result to all threads in the Warp
        lock_result = __shfl_sync(mask, lock_result, leader);

        // If atomicCAS returns 0, the lock was acquired successfully
        if (lock_result == 0) {
            return; // Success, exit loop
        }

        // --- policy 3: Adaptive backoff ---
        // Only execute on failure, with added randomness
        unsigned int ns = get_random_delay(attempt++, tid);
        __nanosleep(ns);
    }
}

__device__ __forceinline__ void mutex_unlock(int* mutex) {
    __threadfence(); 
    atomicExch(mutex, 0);
}
__device__ __forceinline__ int get_local_pressure() {
    //__activemask() returns a 32-bit mask representing active threads in the warp
    unsigned int mask = __activemask();
    
    int active_threads = __popc(mask);
    
    return active_threads; 
}
__device__ __forceinline__ void acquire_lock(int* mutex) {

#if defined(USE_BACKOFF)
    if(get_local_pressure() <= 16) {
        acquire_lock_spin(mutex);
    } else {
        acquire_lock_adaptive(mutex);
    }
#else
    acquire_lock_spin(mutex);
#endif
}

__device__ __forceinline__ void release_lock(int* mutex) {
    mutex_unlock(mutex);
}
#endif // LOCKERS_CUH
