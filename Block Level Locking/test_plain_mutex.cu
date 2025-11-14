// test_plain_mutex.cu
#include <cstdio>
#include <cuda_runtime.h>

#include "cudaLocks.h"
#include "cudaLocksMutexPlain.cuh"   // 刚刚那份无 backoff 的实现

// block-scope、无 backoff 锁的 kernel
__global__ void kernel_with_plain_lock(cudaMutex_t mutex,
                                       unsigned int *mutexBufferHeads,
                                       int NUM_SM,
                                       int *shared_counter,
                                       int iters_per_thread)
{
    const bool isMasterThread = (threadIdx.x == 0 &&
                                 threadIdx.y == 0 &&
                                 threadIdx.z == 0);

    for (int i = 0; i < iters_per_thread; ++i) {

        // 整个 block 一起调用锁（只有 master CAS）
        cudaMutexPlainLock(mutex, mutexBufferHeads, NUM_SM);
	float a=3.0;
	float b=0;
        // ===== 临界区：只让 master 改全局 counter =====
        if (isMasterThread) {
            int tmp = *shared_counter;
            tmp += 1;
            *shared_counter = tmp;
		for(int j=0;j<100000;j++){
		a=a+0.2;
		a=a+0.2;
		a=a+0.2;
		a=a+0.2;
		a=a+0.2;
		b=a+0.5;
		b=b+0.5;
		}
}
        // ===== 临界区结束 =====

        cudaMutexPlainUnlock(mutex, mutexBufferHeads, NUM_SM);
    }
}

int main() {
    cudaError_t err;

    // 1. 获取 NUM_SM
    cudaDeviceProp prop;
    int dev = 0;
    err = cudaGetDeviceProperties(&prop, dev);
    if (err != cudaSuccess) {
        printf("cudaGetDeviceProperties failed: %s\n",
               cudaGetErrorString(err));
        return 1;
    }
    int NUM_SM = prop.multiProcessorCount;
    printf("Device %d: %s, NUM_SM = %d\n",
           dev, prop.name, NUM_SM);

    // 2. 锁相关
    const int NUM_MUTEXES = 1;
    cudaMutex_t mutex;
    cudaMutexCreatePlain(&mutex, 0);   // 用 0 号锁

    unsigned int *d_mutexBufferHeads = nullptr;
    err = cudaMalloc(&d_mutexBufferHeads,
                     NUM_MUTEXES * NUM_SM * sizeof(unsigned int));
    if (err != cudaSuccess) {
        printf("cudaMalloc d_mutexBufferHeads failed: %s\n",
               cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemset(d_mutexBufferHeads, 0,
                     NUM_MUTEXES * NUM_SM * sizeof(unsigned int));
    if (err != cudaSuccess) {
        printf("cudaMemset d_mutexBufferHeads failed: %s\n",
               cudaGetErrorString(err));
        return 1;
    }

    // 3. counter
    int *d_counter = nullptr;
    int h_counter = 0;

    err = cudaMalloc(&d_counter, sizeof(int));
    if (err != cudaSuccess) {
        printf("cudaMalloc d_counter failed: %s\n",
               cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(d_counter, &h_counter,
                     sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("cudaMemcpy H2D counter failed: %s\n",
               cudaGetErrorString(err));
        return 1;
    }

    // 4. kernel 配置（建议和 EBO 版本一样，方便对比）
 dim3 block(512);

dim3 grid(4096);
    int iters_per_thread = 100000;

	cudaEvent_t start, stop;

cudaEventCreate(&start);

cudaEventCreate(&stop);

cudaEventRecord(start);


    printf("Launching kernel_with_plain_lock: grid=(%d), block=(%d), iters=%d\n",
           grid.x, block.x, iters_per_thread);

    kernel_with_plain_lock<<<grid, block>>>(mutex,
                                            d_mutexBufferHeads,
                                            NUM_SM,
                                            d_counter,
                                            iters_per_thread);

    err = cudaDeviceSynchronize();
	cudaEventRecord(stop);

cudaEventSynchronize(stop);



float ms = 0.0f;

cudaEventElapsedTime(&ms, start, stop);

printf("Kernel time (no-backoff) = %.3f ms\n", ms);



cudaEventDestroy(start);

cudaEventDestroy(stop);







    if (err != cudaSuccess) {
        printf("kernel_with_plain_lock failed: %s\n",
               cudaGetErrorString(err));
        return 1;
    }

    // 5. 拷回结果
    err = cudaMemcpy(&h_counter, d_counter,
                     sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy D2H counter failed: %s\n",
               cudaGetErrorString(err));
        return 1;
    }

    // 这里是 block-scope 锁：每个 block 每轮加一次
    int expected = grid.x * iters_per_thread;
    printf("Final counter = %d, expected = %d\n",
           h_counter, expected);

    cudaFree(d_counter);
    cudaFree(d_mutexBufferHeads);
    return 0;
}
