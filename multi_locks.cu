#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#define NEED_LOCKS 4       // 你要几把锁就写几把
#define ITERS 100


__device__ int d_locks[NEED_LOCKS];
__device__ int d_ticket_next = 0;
__device__ int d_ticket_now  = 0;
__device__ int d_counter = 0;


// ==========================
// 小工具
// ==========================
__device__ __inline__ void gpu_delay(unsigned cycles){
    unsigned long long s = clock64();
    while (clock64() - s < cycles) {}
}

__device__ __inline__ void backoff_wait(unsigned &v){
#ifdef USE_BACKOFF
    gpu_delay(v);
    v = (v < 2048 ? v * 2 : 2048);
#endif
}


// ==========================
// 内层：真实 CAS 多把锁
// 获取次序固定：0→1→2→…→K-1
// ==========================
__device__ void acquire_multiple_locks(int *locks, int K)
{
#ifdef USE_BACKOFF
    unsigned bo = 32;
#endif

    for (int i = 0; i < K; i++)
    {
        // 没拿到就一直等（你要求的语义）
        while (atomicCAS(&locks[i], 0, 1) != 0) {
#ifdef USE_BACKOFF
            backoff_wait(bo);   // backoff 版本
#endif
        }
    }
}

__device__ void release_multiple_locks(int *locks, int K)
{
    
    for (int i = K-1; i >= 0; i--){
        __threadfence();
        locks[i] = 0;
    }
}


// ==========================
// Kernel：ticket lock + 多锁 CAS
// ==========================
__global__ void kernel_simple_multi()
{
    int lane = threadIdx.x & 31;

    for (int it = 0; it < ITERS; it++)
    {
        if (lane == 0)
        {
            // 外层 ticket，保证公平
            
            int my = atomicAdd(&d_ticket_next, 1);
            while (atomicAdd(&d_ticket_now, 0) != my) {
                
                // ticket 等待也是没拿到就等
#ifdef USE_BACKOFF
                gpu_delay(64);
#endif
            }

            // 内层多锁获取
            acquire_multiple_locks(d_locks, NEED_LOCKS);
        }

        __syncwarp();

        // 临界区
        atomicAdd(&d_counter, 1);

        if (lane == 0)
        {
            release_multiple_locks(d_locks, NEED_LOCKS);
            atomicAdd(&d_ticket_now, 1);  // 让下一个 ticket 进
        }

        __syncwarp();
    }
}


// ==========================
// Host
// ==========================
int main()
{
    int zero = 0;
    int locks[NEED_LOCKS] = {0};

    cudaMemcpyToSymbol(d_locks,       locks,  sizeof(locks));
    cudaMemcpyToSymbol(d_counter,     &zero,  sizeof(int));
    cudaMemcpyToSymbol(d_ticket_next, &zero,  sizeof(int));
    cudaMemcpyToSymbol(d_ticket_now,  &zero,  sizeof(int));

    dim3 block(512), grid(512);  // 最容易死的配置，也跑得起来

    cudaEvent_t s,e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    cudaEventRecord(s);
    kernel_simple_multi<<<grid, block>>>();
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA error: %s\n", cudaGetErrorString(err));

    float ms;
    cudaEventElapsedTime(&ms, s, e);

    int result;
    cudaMemcpyFromSymbol(&result, d_counter, sizeof(int));

    printf("Time: %.3f ms\n", ms);
    printf("Counter = %d\n", result);
#ifdef USE_BACKOFF
    printf("Mode: backoff\n");
#else
    printf("Mode: no-backoff\n");
#endif

    return 0;
}