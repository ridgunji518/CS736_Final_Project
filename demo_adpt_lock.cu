#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<cuda_runtime.h>


__device__ unsigned int get_random_delay(unsigned int attempt, int tid){
    unsigned long long time = clock64();
    unsigned int seed = (unsigned int)time ^ tid;
    unsigned int cap = 1024;
    unsigned int limit = 32 * (1 << (attempt < 10 ? attempt : 10));
    if (limit > cap) limit = cap;
    
    return (seed % limit) + 1;
}

__device__ void adaptive_lock(int* mutex) {
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

__device__ void adaptive_unlock(int* mutex) {
    // Releasing the lock usually only requires a simple atomic exchange or direct write (depends on memory model)
    atomicExch(mutex, 0); 
    // Or on Volta+ use __threadfence() + direct assignment for better performance
    // __threadfence(); 
    // *(volatile int*)mutex = 0;
}