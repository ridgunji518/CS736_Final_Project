#include <cstdio>
#include <cuda.h>

__device__ int lock_var = 0;
__device__ int counter  = 0;

#ifdef USE_BACKOFF
__device__ void acquire_lock(int *lock) {
    unsigned delay = 2;
    while (atomicCAS(lock, 0, 1) != 0) {
        unsigned long long start = clock64();
        while (clock64() - start < delay);   // Busy waiting delay
        delay = min(delay * 2, 1024u);       // Exponential Backoff
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

__global__ void kernel_operations(int iters, float *sink_array, int sink_size) {
    for (int i = 0; i < iters; i++) {
        acquire_lock(&lock_var);
        
        // ---- Critical section ----
        counter++;
        float x = threadIdx.x + blockIdx.x * 1.0f;
        
        sink_array[0] += 1;
        #pragma unroll 1
        for (int k = 1; k < sink_size; ++k) {
            sink_array[k] = sink_array[k-1];
        }
        // ---- End critical section ----
        
        release_lock(&lock_var);
    }
}

void reset_globals() {
    int zero = 0;
    cudaMemcpyToSymbol(lock_var, &zero, sizeof(int));
    cudaMemcpyToSymbol(counter, &zero, sizeof(int));
}

int main(int argc, char **argv) {
    int sink_size = (argc > 1) ? atoi(argv[1]) : 1024;  // Array size for critical section
    int iters = 1;  // Iterations per thread
    
    // Allocate and initialize the sink array
    float *d_sink_array;
    cudaMalloc(&d_sink_array, sink_size * sizeof(float));
    
    int block_sizes[] = {1, 2, 4, 8, 16, 32};
    int thread_sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    
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
    printf("Array size: %d elements (%.2f KB)\n\n", sink_size, sink_size * sizeof(float) / 1024.0f);
    printf("Blocks,Threads,Total_Threads,Time_ms,Counter,Expected,Correct\n");

    for (int b = 0; b < num_blocks; b++) {
        for (int t = 0; t < num_threads; t++) {
            int blocks = block_sizes[b];
            int threads = thread_sizes[t];
            int total_threads = blocks * threads;
            
            // Reset device globals before each run
            reset_globals();
            cudaMemset(d_sink_array, 0, sink_size * sizeof(float));
            
            cudaEventRecord(start);
            kernel_operations<<<blocks, threads>>>(iters, d_sink_array, sink_size);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            // Check for kernel errors
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel error: %s\n", cudaGetErrorString(err));
                continue;
            }

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);

            int result = 0;
            cudaMemcpyFromSymbol(&result, counter, sizeof(int));
            
            int expected = total_threads * iters;
            const char* correct = (result == expected) ? "YES" : "NO";
            
            printf("%d,%d,%d,%.3f,%d,%d,%s\n",
                   blocks, threads, total_threads, ms, result, expected, correct);
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_sink_array);
    return 0;
}