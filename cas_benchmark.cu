#include <cstdio>

#include <cuda.h>



__device__ int lock_var = 0;

__device__ int counter  = 0;

__device__ volatile float sink = 0.0f; // 防止临界区计算被优化掉



#ifdef USE_BACKOFF

__device__ void acquire_lock(int *lock) {

    unsigned delay = 2;

    while (atomicCAS(lock, 0, 1) != 0) {

        unsigned long long start = clock64();

        while (clock64() - start < delay);   // 忙等 delay 个周期

        delay = min(delay * 2, 1024u);       // 指数退避上限

    }

    // 根据需要可在此处加 __threadfence();

}

#else
__device__ void acquire_lock(int *lock) {

    while (atomicCAS(lock, 0, 1) != 0) {

        // busy wait

    }

    // 进入临界区前通常不需要 fence（根据你的语义而定）

}

#endif



__device__ void release_lock(int *lock) {

    __threadfence();              // 确保临界区写对其他线程可见

    atomicExch(lock, 0);

}



// 竞争锁：计数 + 可调“假工作”拉长临界区

__global__ void kernel_with_backoff(int iters, int work) {

    for (int i = 0; i < iters; i++) {

        acquire_lock_backoff(&lock_var);



        // ---- 临界区开始 ----

        counter++;  // 共享计数

        float x = threadIdx.x + blockIdx.x * 1.0f;

        #pragma unroll 1

        for (int k = 0; k < work; ++k) {

            x = x * 1.000001f + 0.999999f;   // 简单计算，制造占用

        }

        sink = x + counter;                  // 防优化

        // ---- 临界区结束 ----



        release_lock(&lock_var);

    }

}



int main(int argc, char **argv) {

    int blocks = 1024, threads = 1024;

    int iters  = 1;        // 每线程进入临界区次数

    int work = atoi(argv[1]);    // 临界区“假工作”量，越大越长



    cudaEvent_t start, stop;

    cudaEventCreate(&start);

    cudaEventCreate(&stop);



    cudaEventRecord(start);

    kernel_with_backoff<<<blocks, threads>>>(iters, work);

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



