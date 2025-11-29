#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include "include/lockers.cuh"
#define TABLE_SIZE   (1 << 15)   // Can be adjusted smaller to create more conflicts, e.g., 2^12
#define EMPTY_KEY    -1
#define MAX_PROBE    4096


__device__ __host__ inline
unsigned int simple_hash(int key) {
    unsigned int x = (unsigned int)key;
    // it is a simple mix function from MurmurHash3
    x ^= x >> 17;
    x *= 0xed5ad4bb;
    x ^= x >> 11;
    x *= 0xac4c1b51;
    x ^= x >> 15;
    x *= 0x31848bab;
    x ^= x >> 14;
    return x;
}

// ----------- calculate probe distance -----------
__device__ inline
int probe_distance(int key, int idx) {
    if (key == EMPTY_KEY) return 0;
    unsigned int h = simple_hash(key) % TABLE_SIZE;
    int dist = idx - (int)h;
    if (dist < 0) dist += TABLE_SIZE;
    return dist;
}

// ----------- initialize table (keys = EMPTY, vals = 0, locks = 0) -----------
__global__ void init_table(int *table_keys, int *table_vals, int *locks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < TABLE_SIZE) {
        table_keys[idx] = EMPTY_KEY;
        table_vals[idx] = 0;
        locks[idx]      = 0;
    }
}

// ----------- Robin Hood 插入 -----------
__global__ void robinhood_insert_kernel(
    const int *input_keys,
    const int *input_vals,
    int num_keys,
    int *table_keys,
    int *table_vals,
    int *locks)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_keys) return;

    int key   = input_keys[tid];
    int value = input_vals ? input_vals[tid] : tid;

    unsigned int h = simple_hash(key) % TABLE_SIZE;
    int idx  = (int)h;
    int dist = 0;

    // Classic Robin Hood insertion (with bucket locks)
    while (dist < MAX_PROBE) {

        // Every time we move to a new bucket, try to acquire its lock
        acquire_lock(&locks[idx]);

        int curr_key   = table_keys[idx];
        int curr_value = table_vals[idx];

        if (curr_key == EMPTY_KEY) {
            // empty bucket: put it here and done
            table_keys[idx] = key;
            table_vals[idx] = value;
            release_lock(&locks[idx]);
            return;
        }

        // Already have the same key: update or accumulate, depending on your design
        if (curr_key == key) {
            // simple replace
            table_vals[idx] = value;
            release_lock(&locks[idx]);
            return;
        }

        // Robin Hood: who is more "deserving"?
        int curr_dist = probe_distance(curr_key, idx);
        if (curr_dist < dist) {
            // I am more "deserving". Swap me in!
            table_keys[idx] = key;
            table_vals[idx] = value;

            // take the original key/value and continue
            key   = curr_key;
            value = curr_value;
            dist  = curr_dist;  // can also set to curr_dist

            // release current bucket lock, move to next bucket
            release_lock(&locks[idx]);
        } else {
            // current key is more "deserving", move to next bucket
            release_lock(&locks[idx]);
            dist++;
        }

        idx++;
        if (idx >= TABLE_SIZE) idx = 0;
    }

    //bigger than MAX_PROBE, give up
}
void reset_globals(int *d_table_keys, int *d_table_vals, int *d_locks) {
    // 把哈希表清空：keys = EMPTY_KEY, vals = 0, locks = 0
    // 用 kernel 或 cudaMemset 都行，例如：
    cudaMemset(d_table_keys, 0xFF, TABLE_SIZE * sizeof(int)); // 0xFF 对应 -1
    cudaMemset(d_table_vals, 0,    TABLE_SIZE * sizeof(int));
    cudaMemset(d_locks,      0,    TABLE_SIZE * sizeof(int));
}
// ===== host =====
int main(int argc, char** argv) {
    int num_runs =(argc > 1) ? atoi(argv[1]) : 1;
    int num_keys = TABLE_SIZE * 10;  // load factor > 1, more conflicts

    // ----- host generates random keys -----
    std::vector<int> h_keys(num_keys);
    std::vector<int> h_vals(num_keys);

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, 1 << 30);

    for (int i = 0; i < num_keys; ++i) {
        h_keys[i] = dist(rng);
        h_vals[i] = 1;  // or i anything else
    }

    // ----- allocate device memory -----
    int *d_keys = nullptr, *d_vals = nullptr;
    int *d_table_keys = nullptr, *d_table_vals = nullptr, *d_locks = nullptr;

    cudaMalloc(&d_keys,        num_keys * sizeof(int));
    cudaMalloc(&d_vals,        num_keys * sizeof(int));
    cudaMalloc(&d_table_keys,  TABLE_SIZE * sizeof(int));
    cudaMalloc(&d_table_vals,  TABLE_SIZE * sizeof(int));
    cudaMalloc(&d_locks,       TABLE_SIZE * sizeof(int));

    cudaMemcpy(d_keys, h_keys.data(), num_keys * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, h_vals.data(), num_keys * sizeof(int), cudaMemcpyHostToDevice);

    // ----- init list -----

    int block_sizes[] = {64,128,256,512,1024};
    int thread_sizes[] = {16,32,64,128,256,512};
    int num_blocks = sizeof(block_sizes) / sizeof(block_sizes[0]);
    int num_threads = sizeof(thread_sizes) / sizeof(thread_sizes[0]);

    // ：
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#ifdef USE_BACKOFF
    printf("=== Robin Hood Benchmark (WITH BACKOFF) ===\n");
#elif defined(USE_SMART_BACKOFF)
    printf("=== Robin Hood Benchmark (WITH SMART BACKOFF) ===\n");
#else
    printf("=== Robin Hood Benchmark (NO BACKOFF) ===\n");
#endif
    printf("Inserting %d keys into table of size %d\n", num_keys, TABLE_SIZE);
    printf("Blocks,Threads,Total_Threads,Avg_Time_ms,Min_Time_ms,Max_Time_ms,StdDev_ms\n");
    for(int b=0;b<num_blocks;b++){
        for(int t=0;t<num_threads;t++){
            int blocks = block_sizes[b];
            int threads = thread_sizes[t];
            dim3 block(threads);
            dim3 grid((num_keys + block.x - 1) / block.x);
            int total_threads = blocks * threads;
            float sum_time = 0.0f;
            float sum_sq_time = 0.0f;
            float min_time = 1e10f;
            float max_time = 0.0f;
            for(int run=0;run<num_runs;run++){
                reset_globals(d_table_keys, d_table_vals, d_locks);

                init_table<<<grid, block>>>(d_table_keys, d_table_vals, d_locks);
                cudaDeviceSynchronize();

                cudaEventRecord(start);
                robinhood_insert_kernel<<<grid, block>>>(
                    d_keys, d_vals, num_keys,
                    d_table_keys, d_table_vals, d_locks
                );
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
                if(ms < min_time) min_time = ms;
                if(ms > max_time) max_time = ms;
            }
            float avg_time = sum_time / num_runs;
            float variance = (sum_sq_time / num_runs) - (avg_time * avg_time);
            float stddev_time = sqrtf(variance);
            printf("%d,%d,%d,%.6f,%.6f,%.6f,%.6f\n",
                   blocks, threads, total_threads, avg_time, min_time, max_time, stddev_time);
        }
    }
    
    

    // ----- optional:copy back -----
    std::vector<int> h_table_keys(TABLE_SIZE);
    cudaMemcpy(h_table_keys.data(), d_table_keys, TABLE_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    int filled = 0;
    for (int i = 0; i < TABLE_SIZE; ++i)
        if (h_table_keys[i] != EMPTY_KEY) filled++;

    printf("Filled buckets = %d / %d\n", filled, TABLE_SIZE);

    // ----- cleanup -----
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_keys);
    cudaFree(d_vals);
    cudaFree(d_table_keys);
    cudaFree(d_table_vals);
    cudaFree(d_locks);

    return 0;
}