export PATH="/etc/alternatives/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/etc/alternatives/cuda/lib64:${LD_LIBRARY_PATH}"
##Counter-based benchmark
#nvcc -DUSE_BACKOFF -o benchmark_backoff cas_benchmark.cu
#nvcc -o benchmark_no_backoff cas_benchmark.cu

#nvcc -arch=sm_89 -DUSE_SMART_BACKOFF -o ada_benchmark_backoff cas_benchmark.cu
#./ada_benchmark_backoff 1024 1 \
#  > outputs/yes_adaptive_Backoff_1024ele_1iters.csv

#nvcc -arch=sm_89 -DUSE_BACKOFF -o no_ada_benchmark_backoff cas_benchmark.cu
#./no_ada_benchmark_backoff 1024 1 \
#  > outputs/no_adaptive_Backoff_1024ele_1iters.csv

#nvcc -o benchmark_no_backoff cas_benchmark.cu
#./benchmark_no_backoff 1024 1 \
#  > outputs/no_Backoff_1024ele_1iters.csv


##letter workload benchmark
#nvcc -arch=sm_89 -DUSE_SMART_BACKOFF -o ada_letter_backoff Letter_cas_bm.cu
#./ada_letter_backoff\
#  > outputs/adaptive_letter_Backoff_1024ele_1iters.csv

#nvcc -arch=sm_89 -DUSE_BACKOFF -o no_ada_letter_backoff Letter_cas_bm.cu
#./no_ada_letter_backoff \
#  > outputs/no_adaptive_letter_Backoff_1024ele_1iters.csv

#nvcc -o no_letter_backoff Letter_cas_bm.cu
#./no_letter_backoff \
#  > outputs/no_letter_Backoff_1024ele_1iters.csv

#ncu --set full -o ncu/ada_letter_rep ./ada_letter_backoff 100000000000 1
#ncu --set full -o ncu/no_ada_letter_rep ./no_ada_letter_backoff 100000000000 1
#ncu --set full -o ncu/no_bf_letter_rep ./no_letter_backoff 100000000000 1


##Multi locks benchmark
#nvcc -arch=sm_89 -DUSE_BACKOFF -o build/multi_backoff multi_locks.cu
#nvcc -arch=sm_89 -o build/multi_no_backoff multi_locks.cu

#ncu --set full -o ncu/multi_backoff_rep ./build/multi_backoff
#ncu --set full -o ncu/multi_no_backoff_rep ./build/multi_no_backoff

##Robin Hood Hashing multi locks benchmark
#nvcc -arch=sm_89 -DUSE_BACKOFF -o build/hash_noada_backoff hash_locks.cu
#nvcc -arch=sm_89 -o build/hash_no_backoff hash_locks.cu
#nvcc -arch=sm_89 -DUSE_SMART_BACKOFF -o build/hash_ada_backoff hash_locks.cu
#ncu --set full -o ncu/hash_ada_backoff_rep ./build/hash_ada_backoff
#ncu --set full -o ncu/hash_noada_backoff_rep ./build/hash_noada_backoff
#ncu --set full -o ncu/hash_no_backoff_rep ./build/hash_no_backoff
# ./build/hash_ada_backoff 10\
#    > outputs/ada_hash_backoff_10iters.csv
#./build/hash_noada_backoff 10\
#    > outputs/noada_hash_backoff_10iters.csv
# ./build/hash_no_backoff 10\
#    > outputs/no_hash_backoff_10iters.csv