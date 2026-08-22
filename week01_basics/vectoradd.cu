#include <cstdio>
#include <cuda_runtime.h>

__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("GPU: %s  (compute capability sm_%d%d)\n", p.name, p.major, p.minor);

    const int n = 1 << 20;
    size_t bytes = n * sizeof(float);
    float *ha = (float*)malloc(bytes), *hb = (float*)malloc(bytes), *hc = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) { ha[i] = i * 1.0f; hb[i] = i * 2.0f; }

    float *da, *db, *dc;
    cudaMalloc(&da, bytes); cudaMalloc(&db, bytes); cudaMalloc(&dc, bytes);
    cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice);

    vecAdd<<<(n + 255) / 256, 256>>>(da, db, dc, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("launch FAILED: %s\n", cudaGetErrorString(err)); return 1; }

    err = cudaDeviceSynchronize(); 
    if (err != cudaSuccess) { printf("kernel FAILED: %s\n", cudaGetErrorString(err)); return 1; }
    
    cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        if (hc[i] != ha[i] + hb[i]) { printf("MISMATCH at %d\n", i); return 1; }
    }
    printf("Result: PASS (%d elements)\n", n);
    cudaFree(da); cudaFree(db); cudaFree(dc); free(ha); free(hb); free(hc);
    return 0;
}
