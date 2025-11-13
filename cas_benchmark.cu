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
`
    // Maybe: __threadfence();

}

#else
__device__ void acquire_lock(int *lock) {

    while (atomicCAS(lock, 0, 1) != 0) {

        // busy wait

    }

    // Maybe: fence（根据你的语义而定）

}

#endif



__device__ void release_lock(int *lock) {

    __threadfence();              // Make sure each thread had updated memory

    atomicExch(lock, 0);

}



__global__ void kernel_operations(int iters, float *sink_array, int sink_size) {

    for (int i = 0; i < iters; i++) {

        acquire_lock_backoff(&lock_var);

        counter++;

        float x = threadIdx.x + blockIdx.x * 1.0f;


        sink_array[0] += 1;
        #pragma unroll 1

        for (int k = 1; k < sink_size; ++k) {
            sink_array[k] = sink_array[k-1];
        }
        
        release_lock(&lock_var);

    }

}



int main(int argc, char **argv) {

    int blocks = 1024, threads = 1024;

    int iters  = 1;        // 每线程进入临界区次数

    const int sink_size = 1024 * 1024;

    float *d_sink;
    cudaMalloc(&d_sink, sink_size * sizeof(float));

    cudaEvent_t start, stop;

    cudaEventCreate(&start);

    cudaEventCreate(&stop);



    cudaEventRecord(start);

    kernel_operations<<<blocks, threads>>>(iters, d_sink, sink_size);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);



    float ms = 0.0f;

    cudaEventElapsedTime(&ms, start, stop);



    int result = 0;

    cudaMemcpyFromSymbol(&result, counter, sizeof(int));



    printf("Threads: %d  Blocks: %d  Iterations/thread: %d  Work: %d\n",

           threads, blocks, iters, work);

    printf("Final counter = %d (expected %d)\n",

           result, blocks * threads * iters);

    printf("Kernel execution time = %.3f ms\n", ms);



    cudaEventDestroy(start);

    cudaEventDestroy(stop);

    return 0;

}



