#include <cstdio>
#include <cstdlib>
#include <string>
#include <ctime>
#include <cuda_runtime.h>

#define LETTERS 26

__device__ int lock_var = 0;
__device__ int global_hist[LETTERS];
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

__global__ void kernel_operations(char *text, int text_len, int chunk_size) {
    if(threadIdx.x == 0) {  
        
        // 1. 定义局部直方图 (寄存器/Local Memory)
        int local_hist[LETTERS];
        for(int i=0; i<LETTERS; i++) local_hist[i] = 0;

        // 2. 【锁外计算】模拟真实负载：先在本地统计一小块数据
        // 每个 Block 处理 text 中的一小段 (chunk_size)
        // 为了避免内存越界，简单的取模或者循环读取
        int start_idx = (blockIdx.x * chunk_size) % text_len;
        for (int i = 0; i < chunk_size; i++) {
            char c = text[(start_idx + i) % text_len];
            if (c >= 'A' && c <= 'Z') {
                local_hist[c - 'A']++;
            } else if (c >= 'a' && c <= 'z') {
                local_hist[c - 'a']++; // 简单处理，不区分大小写
            }
        }

        // 3. 【抢锁】
        acquire_lock(&lock_var);
        
        // 4. 【临界区】将局部结果更新到全局
        // 这是一个"重"临界区，包含 26 次读写，非常适合展示 Backoff 的优势
        for (int i = 0; i < LETTERS; i++) {
            if (local_hist[i] > 0) {
                global_hist[i] += local_hist[i];
            }
        }
    }
        // 5. 【释放锁】
        release_lock(&lock_var);
    
}

void reset_globals() {
    int zero = 0;
    cudaMemcpyToSymbol(lock_var, &zero, sizeof(int));
    int zeros[LETTERS] = {0};
    cudaMemcpyToSymbol(global_hist, zeros, LETTERS * sizeof(int));
}


int main(int argc, char **argv) {
    int text_len = (argc > 1) ? atoi(argv[1]) : 1024 * 1024; // 默认 1MB 文本
    int num_runs = (argc > 2) ? atoi(argv[2]) : 1;   // Number of iterations to average
  // Iterations per thread in kernel
    
    int chunk_size = 256;
    //randomly generate text
    char *h_text = (char*)malloc(text_len);
    srand(time(NULL));
    for(int i=0; i<text_len; i++) {
        h_text[i] = 'a' + (rand() % 26); // 简单生成 a-z
    }
    
    char *d_text;
    cudaMalloc(&d_text, text_len);
    cudaMemcpy(d_text, h_text, text_len, cudaMemcpyHostToDevice);

    int block_sizes[] = {512}; // 4096 blocks = 4096 个竞争者
    int thread_sizes[] = {256};
    int num_blocks_configs = sizeof(block_sizes) / sizeof(block_sizes[0]);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

#ifdef USE_BACKOFF
    printf("=== Letter Histogram Benchmark (WITH BACKOFF) ===\n");
#elif defined(USE_SMART_BACKOFF)
    printf("=== Letter Histogram Benchmark (WITH SMART BACKOFF) ===\n");
#else
    printf("=== Letter Histogram Benchmark (NO BACKOFF) ===\n");
#endif
    printf("Text length: %d chars, Chunk per thread: %d\n", text_len, chunk_size);
    printf("Blocks,Threads,Total_Tasks,Avg_Time_ms,Throughput(GB/s),Correct\n");
    for (int b = 0; b < num_blocks_configs; b++) {
        int blocks = block_sizes[b];
        int threads = thread_sizes[0];
        
        float sum_time = 0.0f;
        
        // 运行多次取平均
        for (int run = 0; run < num_runs; run++) {
            reset_globals();
            
            cudaEventRecord(start);
            // 启动 Kernel
            kernel_operations<<<blocks, threads>>>(d_text, text_len, chunk_size);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            sum_time += ms;
        }
        
        float avg_time = sum_time / num_runs;
        
        // 验证结果
        int h_global_hist[LETTERS];
        cudaMemcpyFromSymbol(h_global_hist, global_hist, LETTERS * sizeof(int));
        
        long long total_count = 0;
        for(int i=0; i<LETTERS; i++) total_count += h_global_hist[i];
        
        // 预期处理的总字符数 = blocks * chunk_size
        long long expected = (long long)blocks * chunk_size;
        const char* correct = (total_count == expected) ? "YES" : "NO";
        
        printf("%d,%d,%lld,%.3f,%.2f,%s\n",
               blocks, threads, expected, avg_time, 
               (expected * sizeof(int) * 1e-6) / avg_time, // 简单吞吐量估算
               correct);
    }

    cudaFree(d_text);
    free(h_text);
    
    return 0;
}