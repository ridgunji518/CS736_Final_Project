#include <cstdio>
#include <cuda.h>
#include<cuda.h>
#include<cuda_runtime.h>

__device__ int lock_var = 0;
__device__ int counter  = 0;

__device__ unsigned int get_random_delay(unsigned int attempt, int tid){
    unsigned long long time = clock64();
    unsigned int seed = (unsigned int)time ^ tid;
    unsigned int cap = 1024;
    unsigned int limit = 32 * (1 << (attempt < 10 ? attempt : 10));
    if (limit > cap) limit = cap;
    
    return (seed % limit) + 1;
}

#ifdef USE_BACKOFF
__device__ void acquire_lock(int *lock) {
    unsigned delay = 2;
    while (atomicCAS(lock, 0, 1) != 0) {
        unsigned long long start = clock64();
        while (clock64() - start < delay);   // Busy waiting delay
        delay = min(delay * 2, 1024u);       // Exponential Backoff
    }
}
#elif defined(USE_SMART_BACKOFF)
__device__ void acquire_lock(int* mutex) {
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
#else
__device__ void acquire_lock(int *lock) {
    while (atomicCAS(lock, 0, 1) != 0) {
        // busy wait
    }
}
#endif

__device__ void release_lock(int *lock) {
    __threadfence();              // Make sure each thread had updated memory
    atomicExch(lock, 0);
}

struct BlockTiming {
    unsigned long long acquire_time;
    unsigned long long block_time;
};

__global__ void kernel_operations(float *sink_array, int sink_size, struct BlockTiming *block_times) {

    __shared__ unsigned long long start_clock;
    __shared__ unsigned long long acquire_clock;
    __shared__ unsigned long long end_clock;

    int threadsInBlock = blockDim.x;
    int length = sink_size / threadsInBlock;
    int startIndex = length * threadIdx.x;
    int finalIndex = length + startIndex;


    if(threadIdx.x == 0) {
        start_clock = clock64();
        acquire_lock(&lock_var);
        acquire_clock = clock64();
    }

    __syncthreads();
    // ---- Critical section ----
    sink_array[startIndex] += 1;
    for (int k = startIndex + 1; k < finalIndex; ++k) {
        sink_array[k] = sink_array[k-1];
    }
    // ---- End critical section ----
    __syncthreads();

    if(threadIdx.x == 0) {
        release_lock(&lock_var);
        end_clock = clock64();
    }

    if (threadIdx.x == 0) {
        block_times[blockIdx.x].acquire_time = acquire_clock - start_clock;
        block_times[blockIdx.x].block_time = end_clock - acquire_clock;
    }
}

void reset_globals() {
    int zero = 0;
    cudaMemcpyToSymbol(lock_var, &zero, sizeof(int));
    cudaMemcpyToSymbol(counter, &zero, sizeof(int));
}

int is_power_of_2(int x) {
    return x > 0 && (x & (x - 1)) == 0;
}

int main(int argc, char **argv) {
    int sink_size = (argc > 1) ? atoi(argv[1]) : 1024;  // Array size for critical section

    int num_runs = (argc > 2) ? atoi(argv[2]) : 1000;   // Number of iterations to average

    if(sink_size % 256 != 0) {
        printf("Sink Size must be divisible by 256\n");
        return -1;
    }
    
    // Allocate and initialize the sink array
    float *d_sink_array;
    cudaMalloc(&d_sink_array, sink_size * sizeof(float));

    // Allocate and initialize the block times array
    struct BlockTiming *d_block_times;
    constexpr int maxNumBlocks = 512;
    cudaMalloc(&d_block_times, maxNumBlocks * sizeof(struct BlockTiming));
    
    int block_sizes[] = {512};
    int thread_sizes[] = {256};
    
    int num_blocks = sizeof(block_sizes) / sizeof(block_sizes[0]);
    int num_threads = sizeof(thread_sizes) / sizeof(thread_sizes[0]);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

#ifdef USE_BACKOFF
    printf("=== Lock Contention Benchmark (WITH BACKOFF) ===\n");
#else
    printf("=== Lock Contention Benchmark (NO BACKOFF) ===\n");
#endif

    printf("Array size: %d elements (%.2f KB)\n", sink_size, sink_size * sizeof(float) / 1024.0f);
    printf("Runs per configuration: %d\n\n", num_runs);
    printf("Blocks,Threads,Total_Threads,Avg_Time_ms,Min_Time_ms,Max_Time_ms,StdDev_ms,Counter,Expected,Correct\n");

    for (int b = 0; b < num_blocks; b++) {
        for (int t = 0; t < num_threads; t++) {
            int blocks = block_sizes[b];
            int threads = thread_sizes[t];
            int total_threads = blocks * threads;
            
            float sum_time = 0.0f;
            float sum_sq_time = 0.0f;
            float min_time = 1e10f;
            float max_time = 0.0f;
            int final_counter = 0;
            
            // Run multiple times for this configuration
            for (int run = 0; run < num_runs; run++) {
                // Reset device globals before each run
                reset_globals();
                cudaMemset(d_sink_array, 0, sink_size * sizeof(float));
                
                cudaEventRecord(start);
                kernel_operations<<<blocks, threads>>>(d_sink_array, sink_size, d_block_times);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                // Check for kernel errors
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("Kernel error: %s\n", cudaGetErrorString(err));
                    break;
                }

                float ms = 0.0f;
                cudaEventElapsedTime(&ms, start, stop);
                
                sum_time += ms;
                sum_sq_time += ms * ms;
                min_time = (ms < min_time) ? ms : min_time;
                max_time = (ms > max_time) ? ms : max_time;
                
                // Only check counter on last run
                if (run == num_runs - 1) {
                    cudaMemcpyFromSymbol(&final_counter, counter, sizeof(int));
                }
            }
            
            float avg_time = sum_time / num_runs;
            float variance = (sum_sq_time / num_runs) - (avg_time * avg_time);
            float stddev = sqrtf(variance > 0 ? variance : 0);
            
            int expected = blocks * num_runs;  // Only 1 thread per block competes
            const char* correct = (final_counter == expected) ? "YES" : "NO";

            // print times it took for each block
            struct BlockTiming h_block_times[maxNumBlocks];
            cudaMemcpy(h_block_times, d_block_times,
                    blocks * sizeof(struct BlockTiming),
                    cudaMemcpyDeviceToHost);

            double averageAcquireTime = 0;
            double averageBlockTime = 0;
            for (int i = 0; i < blocks; i++) {
                averageAcquireTime += h_block_times[i].acquire_time;
                averageBlockTime += h_block_times[i].block_time;
            }

            averageAcquireTime /= blocks;
            averageBlockTime /= blocks;
            
            printf("%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%d,%d,%s,%0.6f,%0.6f\n",
                   blocks, threads, total_threads, avg_time, min_time, max_time, stddev,
                   final_counter, expected, correct, averageAcquireTime, averageBlockTime);


        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_sink_array);
    cudaFree(d_block_times);
    return 0;
}