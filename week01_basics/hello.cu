#include <cstdio>

__global__ void hello() {
    printf("Hello from block %d, thread %d (global id %d)\n",
           blockIdx.x, threadIdx.x,
           blockIdx.x * blockDim.x + threadIdx.x);
}

int main() {
    hello<<<8, 64>>>();
    //cudaDeviceSynchronize();   // 不加这句 printf 出不来
    return 0;
}