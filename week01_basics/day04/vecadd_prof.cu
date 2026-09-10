// day04/vecadd_prof.cu
// nvcc -O2 -std=c++17 -I../common vecadd_prof.cu -o vecadd_prof
#include "benchmark.h"

__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int n = 1 << 26;                    // 256 MB/数组，工作集 768 MB
    size_t nb = (size_t)n * sizeof(float);
    float *da, *db, *dc;
    CUDA_CHECK(cudaMalloc(&da, nb));
    CUDA_CHECK(cudaMalloc(&db, nb));
    CUDA_CHECK(cudaMalloc(&dc, nb));
    CUDA_CHECK(cudaMemset(da, 1, nb));
    CUDA_CHECK(cudaMemset(db, 2, nb));

    const int block = 256, grid = (n + block - 1) / block;
    vecAdd<<<grid, block>>>(da, db, dc, n);   // 只跑一次
    CUDA_CHECK_LAUNCH();

    printf("n = %d, 理论 bytes = %zu\n", n, 3ull * n * sizeof(float));
    cudaFree(da); cudaFree(db); cudaFree(dc);
    return 0;
}