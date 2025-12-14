#ifndef LOCKERS_CUH
#define LOCKERS_CUH
#include <cuda_runtime.h>
__device__ __forceinline__ unsigned int fast_hash(unsigned int x) {
    x ^= x >> 13;
    x *= 0x5bd1e995;
    x ^= x >> 15;
    return x;
}
__device__ __forceinline__ unsigned int get_random_delay(unsigned int attempt, int tid){
    const unsigned int MAX_SHIFT = 5; 
    unsigned int shift = (attempt < MAX_SHIFT) ? attempt : MAX_SHIFT;
    unsigned int limit = 32u << shift;
    unsigned long long time = clock64();
    unsigned int seed = (unsigned int)time ^ tid ^ attempt;
    unsigned int random_val = fast_hash(seed);
    return (random_val & (limit - 1)) + 1;
}

__device__ __forceinline__ void acquire_lock_spin(int *lock) {
    while (atomicCAS(lock, 0, 1) != 0) {
        // busy wait
    }
}

__device__ __forceinline__ void acquire_lock_binary_exponential(int *lock) {
    unsigned delay = 2;
    while (atomicCAS(lock, 0, 1) != 0) {
        __nanosleep(min(delay*2, 1024u));       // Exponential Backoff
    }
}

__device__ __forceinline__ void acquire_lock_random_backoff(int *lock) {
    int attempt = 0;
    while (atomicCAS(lock, 0, 1) != 0) {
        attempt++;
        unsigned int ns = get_random_delay(attempt, threadIdx.x + blockIdx.x * blockDim.x);
        __nanosleep(ns);
    }
}
__device__ __forceinline__ void acquire_lock_linear_backoff(int *lock) {
    unsigned delay = 1;
    while (atomicCAS(lock, 0, 1) != 0) {
        __nanosleep(min(delay + 2, 1024u));       // Linear Backoff
    }
}
__device__ __forceinline__ void acquire_lock_uninform_backoff(int *lock) {
    while (atomicCAS(lock, 0, 1) != 0) {
        unsigned int ns = get_random_delay(0, threadIdx.x + blockIdx.x * blockDim.x);
        __nanosleep(ns);
    }
}
__device__ __forceinline__ void acquire_lock_adaptive(int* mutex) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int attempt = 0;

    // set a limit for leader spinning
    // the leader will spin for a certain number of iterations before giving up
    const int LEADER_SPIN_LIMIT = 100; 

    while (true) {
        // --- policy 1: Warp aggregation ---
        unsigned int mask = __activemask();
        int leader = __ffs(mask) - 1; 
        int lane_id = threadIdx.x % 32;

        bool is_leader = (lane_id == leader);
        int lock_result = 0;

        if (is_leader) {
            // --- policy 2: Enhanced TTAS (Spin-Then-Try) ---
            
            // stage A: self-spin
            // Leader will spin for a short time before attempting to acquire the lock
            int spin_count = 0;
            while (*(volatile int*)mutex != 0 && spin_count < LEADER_SPIN_LIMIT) {
                spin_count++;
                // can insert a small delay here if needed
                // __nanosleep(1); // 可选
            }

            // stage B: try to acquire
            // only initiate atomic operation if it looks like 0
            if (*(volatile int*)mutex == 0) {
                lock_result = atomicCAS(mutex, 0, 1);
            } else {
                lock_result = 1; // set to failure if still locked
            }
        }

        // Leader 广播结果
        // 注意：Leader 在上面自旋的时候，Warp 里其他线程都在这里等 Leader (SIMT 特性)
        lock_result = __shfl_sync(mask, lock_result, leader);

        // 成功获取
        if (lock_result == 0) {
            return; 
        }

        // --- policy 3: Adaptive/Exponential Backoff ---
        // 只有当 Leader 自旋 + 抢锁都失败了，整个 Warp 才去休眠
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
    if(get_local_pressure() <= 8) {
        acquire_lock_spin(mutex);
    } else {
        acquire_lock_adaptive(mutex);
    }
#elif defined(BIN_EXP_BACKOFF)
    acquire_lock_binary_exponential(mutex);
#elif defined(LINEAR_BACKOFF)
    acquire_lock_linear_backoff(mutex);
#elif defined(UNIFORM_BACKOFF)
    acquire_lock_uninform_backoff(mutex);
#else
    acquire_lock_spin(mutex);
#endif
}

__device__ __forceinline__ void release_lock(int* mutex) {
    mutex_unlock(mutex);
}
#endif // LOCKERS_CUH