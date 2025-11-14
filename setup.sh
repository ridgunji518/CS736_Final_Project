export PATH="/etc/alternatives/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/etc/alternatives/cuda/lib64:${LD_LIBRARY_PATH}"

nvcc -DUSE_BACKOFF -o benchmark_backoff cas_benchmark.cu
nvcc -o benchmark_no_backoff cas_benchmark.cu