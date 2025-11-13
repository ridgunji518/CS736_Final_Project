#include <cstdio>

#include <cuda.h>



__device__ int lock_var = 0;

__device__ int counter  = 0;

__device__ volatile float sink = 0.0f; // 防止编译器优化掉临界区里的计算



// 简单自旋锁（无退避）

__device__ void acquire_lock(int *lock) {

    while (atomicCAS(lock, 0, 1) != 0) {

        // busy wait

    }

    // 进入临界区前通常不需要 fence（根据你的语义而定）

}



__device__ void release_lock(int *lock) {

    __threadfence();            // 确保临界区写入对其他线程可见

    atomicExch(lock, 0);

}



// 所有线程竞争锁，对 counter 累加 + 做一段可控“假工作”

__global__ void kernel_no_backoff(int iters, int work) {

    for (int i = 0; i < iters; i++) {

        acquire_lock(&lock_var);



        // ---- 临界区开始 ----

        counter++;  // 原有的共享状态更新



        // 做一些计算工作，延长临界区时间（用 volatile 写回避免被优化掉）

        float x = threadIdx.x * 1.0f + blockIdx.x;

        #pragma unroll 1

        for (int k = 0; k < work; ++k) {

            // 简单但有数据相关性的计算，防止完全优化

            x = x * 1.000001f + 0.999999f;

        }

        sink = x + counter; // 防止上面的循环被优化掉

        // ---- 临界区结束 ----



        release_lock(&lock_var);

    }

}



int main(int argc, char **argv) {

    int blocks  = 1024;

    int threads = 1024;

    int iters   = 1;     // 每线程进入临界区次数

    int work = atoi(argv[1]); // 临界区“假工作”量，数值越大临界区越长



    cudaEvent_t start, stop;

    cudaEventCreate(&start);

    cudaEventCreate(&stop);



    cudaEventRecord(start);

    kernel_no_backoff<<<blocks, threads>>>(iters, work);

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


