#include <cstdio>
#include <cstdlib>
#include <string>
#include <ctime>
#include <cuda_runtime.h>
#include "cudaMutexSimpleEBO.cuh"

#define LETTERS 26

__device__ void atomicAddCAS(int *addr, int val){
    int old = *addr;
    int assumed;
    do {
        assumed = old;
        old = atomicCAS(addr, assumed, assumed + val);
    } while (assumed != old);
}

__global__ void histogram_kernel(
    const char *text,
    int text_len,
    int chunk_size,
    int *global_hist,
    long long *cs_time,
    int mutex,
    unsigned int *mutexBuf)
{

    __shared__ int local_hist[LETTERS];
    for (int i = threadIdx.x; i < LETTERS; i += blockDim.x)
        local_hist[i] = 0;
    __syncthreads();

    int start = blockIdx.x * chunk_size;
    int end   = min(start + chunk_size, text_len);

    for (int i = start + threadIdx.x; i < end; i += blockDim.x){
        char c = text[i];
        if (c >= 'A' && c <= 'Z')
            atomicAdd(&local_hist[c - 'A'], 1);
        else if (c >= 'a' && c <= 'z')
            atomicAdd(&local_hist[c - 'a'], 1);
    }
    __syncthreads();

    if (threadIdx.x == 0){
        long long t0 = clock64();
        cudaMutexSimpleEBOLock(mutex, mutexBuf);
        for (int i = 0; i < LETTERS; i++)
            atomicAddCAS(&global_hist[i], local_hist[i]);
        cudaMutexSimpleEBOUnlock(mutex, mutexBuf);
        long long t1 = clock64();
        cs_time[blockIdx.x] = (t1 - t0);
    }
}

int main(){
    int N;
    printf("Enter the length of random string N: ");
    scanf("%d", &N);
    if (N <= 0) {
        printf("N must be > 0\n");
        return 0;
    }
    std::string text;
    text.resize(N);
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        int r = rand() % 52;
        if (r < 26) text[i] = 'A' + r;
        else        text[i] = 'a' + (r - 26);
    }
    char *d_text;
    cudaMalloc(&d_text, N);
    cudaMemcpy(d_text, text.data(), N, cudaMemcpyHostToDevice);
    int *d_hist;
    cudaMalloc(&d_hist, LETTERS * sizeof(int));
    cudaMemset(d_hist, 0, LETTERS * sizeof(int));
    unsigned int *d_mutexBuf;
    cudaMalloc(&d_mutexBuf, sizeof(unsigned int));
    cudaMemset(d_mutexBuf, 0, sizeof(unsigned int));
    int mutex;
    cudaMutexCreateSimpleEBO(&mutex, 0);
    int chunk_size = 1024;
    int grid = (N + chunk_size - 1) / chunk_size;
    int block = 256;
    printf("Launching kernel: grid=%d block=%d\n", grid, block);
    long long *d_cs_time;
    cudaMalloc(&d_cs_time, grid * sizeof(long long));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    histogram_kernel<<<grid, block>>>(
        d_text, N, chunk_size,
        d_hist,
        d_cs_time,
        mutex, d_mutexBuf);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernel_ms;
    cudaEventElapsedTime(&kernel_ms, start, stop);
    printf("Kernel time: %.3f ms\n", kernel_ms);
    int h_hist[LETTERS];
    cudaMemcpy(h_hist, d_hist, LETTERS * sizeof(int), cudaMemcpyDeviceToHost);
    long long *h_cs_time = new long long[grid];
    cudaMemcpy(h_cs_time, d_cs_time, grid * sizeof(long long), cudaMemcpyDeviceToHost);
    printf("\n===== Letter Histogram =====\n");
    for (int i = 0; i < LETTERS; i++)
        printf("%c: %d\n", 'A'+i, h_hist[i]);
    return 0;
}


