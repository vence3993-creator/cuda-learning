#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)\
    do{\
        cudaError_t _e = (call);\
        if(_e != cudaSuccess){  \
            fprintf(stderr,"CUDA error: %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); \
            exit(EXIT_FAILURE);\
        }\
    }while(0)

#ifndef ITERS
#define ITERS 512
#endif 

__global__ void k_uniform(float* out,int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid>=n) return;
    float x = 1.0f + tid * 1e-7f;
    for(int i=0;i<ITERS;++i) x=fmaf(x,1.0001f,1.0f);
    out[tid] = x;
}
// warp 间分歧：条件只依赖 warp id，warp 内 32 个线程取值一致
__global__ void k_inter(float *out, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  float x = 1.0f + tid * 1e-7f;
  if (((tid >> 5) & 1) == 0) {
    for (int i = 0; i < ITERS; ++i) x = fmaf(x, 1.0001f, 1.0f);
  } else {
    for (int i = 0; i < ITERS; ++i) x = fmaf(x, 0.9999f, 2.0f);
  }
  out[tid] = x;
}

// warp 内分歧：奇偶线程分道，每个 warp 都得把两条路径都跑一遍
__global__ void k_intra(float *out, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  float x = 1.0f + tid * 1e-7f;
  if ((tid & 1) == 0) {
    for (int i = 0; i < ITERS; ++i) x = fmaf(x, 1.0001f, 1.0f);
  } else {
    for (int i = 0; i < ITERS; ++i) x = fmaf(x, 0.9999f, 2.0f);
  }
  out[tid] = x;
}

typedef void(*kernel_t)(float *,int);

static float bench(kernel_t k,float*out, int n, int block, int repeat){
  int grid = (n + block -1)/block;
  cudaEvent_t s,e;
  CUDA_CHECK(cudaEventCreate(&s));
  CUDA_CHECK(cudaEventCreate(&e));

  for(int i=0;i<5;++i) k<<<grid,block>>>(out,n);
  CUDA_CHECK(cudaDeviceSynchronize());
  float best = 1e30f;
  for(int i=0;i<repeat;++i){
    CUDA_CHECK(cudaEventRecord(s));
    k<<<grid,block>>>(out,n);
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms,s,e));
    if(ms<best) best = ms;
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventDestroy(s));
  CUDA_CHECK(cudaEventDestroy(e));
  return best;
}

int main(int argc,char **argv){
  const int n =1<<22;
  const int repeat = 20;
  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p,0));
  printf("GPU: %s (sm_%d%d, %d SMs)\n",p.name,p.major,p.minor,p.multiProcessorCount);

  float *out = nullptr;
  CUDA_CHECK(cudaMalloc(&out,(size_t)n * sizeof(float)));
  
  bool sweep = (argc>1&&strcmp(argv[1],"sweep")==0);

  if(!sweep){
    int block = (argc>1)?atoi(argv[1]):256;

    float tu = bench(k_uniform,out,n,block,repeat);
    float tb = bench(k_inter,out,n,block,repeat);
    float ta = bench(k_intra,out,n,block,repeat);
    printf("block = %d\n",block);
    printf("uniform( no branch ) %8.3f ms 1.00x\n",tu);
    printf("inter-warp divergence %8.3f ms %.2fx\n",tb , tb/tu);
    printf("intra-warp divergence %8.3f ms %.2fx <---attention\n",ta, ta/tu);
    printf("\nintra/inter = %.2f (理论值 2.00)\n",ta/tb);
  }
  else{
    const int blocks[] = {32,33,64,65,96,128,160,256,512,1024};
    printf("%6s %10s %10s %10s %8s\n","block","uniform","inter","intra","intra/uni");
    for (int b : blocks){
      float tu = bench(k_uniform,out,n,b,repeat);
      float tb = bench(k_inter,out,n,b,repeat);
      float ta = bench(k_intra,out,n,b,repeat);
      printf("%6d %10.3f%10.3f %10.3f %8.2f%s\n", b, tu, tb, ta, ta / tu,
             (b % 32) ? "   <- 非 32 倍数" : "");
    }
  }
  CUDA_CHECK(cudaFree(out));
  return 0;
}