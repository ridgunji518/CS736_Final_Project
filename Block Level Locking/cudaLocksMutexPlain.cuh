#ifndef __CUDALOCKSMUTEXPLAIN_CU__
#define __CUDALOCKSMUTEXPLAIN_CU__

#include "cudaLocks.h"

// 创建：和 EBO 一样，就是把 handle 设成 mutex 编号
inline __host__ cudaError_t cudaMutexCreatePlain(cudaMutex_t * const handle,
                                                 const int mutexNumber)
{
  *handle = mutexNumber;
  return cudaSuccess;
}

// 无 backoff 的 block-scope 自旋锁
inline __device__ void cudaMutexPlainLock(const cudaMutex_t mutex,
                                          unsigned int * mutexBufferHeads,
                                          const int NUM_SM)
{
  __shared__ int done;
  const bool isMasterThread = (threadIdx.x == 0 &&
                               threadIdx.y == 0 &&
                               threadIdx.z == 0);
  unsigned int * mutexHeadPtr = NULL;

  if (isMasterThread)
  {
    done = 0;
    // 和 EBO 版本一样的寻址方式：每个 mutex 有 NUM_SM 个 slot，
    // 这里我们先用 base 指针（相当于只用第一个 slot）
    mutexHeadPtr = (mutexBufferHeads + (mutex * NUM_SM));
  }
  __syncthreads();

  while (!done)
  {
    __syncthreads();
    if (isMasterThread)
    {
      // 简单自旋：不停 atomicCAS，没有任何 backoff
      if (atomicCAS(mutexHeadPtr, 0, 1) == 0) {
        // 拿到锁，做一个全局的 TF
        __threadfence();
        done = 1;
      }
      // 否则什么都不做，直接下一轮 while
    }
    __syncthreads();
  }
}

// 解锁：基本和 EBO 完全一样，只是名字不同
inline __device__ void cudaMutexPlainUnlock(const cudaMutex_t mutex,
                                            unsigned int * mutexBufferHeads,
                                            const int NUM_SM)
{
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    __threadfence();
    atomicExch(mutexBufferHeads + (mutex * NUM_SM), 0);
  }
  __syncthreads();
}

#endif

