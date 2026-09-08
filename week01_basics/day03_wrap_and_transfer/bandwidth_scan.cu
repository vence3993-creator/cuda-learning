#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<chrono>
#include<cuda_runtime.h>

#define CUDA_CHECK(call)\
    do{\
        cudaError_t err__ = (call);\
        if(err__ !=cudaSuccess){\
            fprintf(stderr,"CUDA error at %s:%d -> %s\n",__FILE__,__LINE__,cudaGetErrorString(err__));\
            exit(EXIT_FAILURE);\
        }\
    }while(0)

using clk = std::chrono::steady_clock;

static int pick_iters(size_t bytes){
    if(bytes <= (1u<<16)) return 2000;
    if(bytes <= (1u<<20)) return 500;
    if(bytes <= (1u<<24)) return 100;
    return 20;
}

static double measure(void*dst,const void*src,size_t bytes,cudaMemcpyKind kind){
    const int iters = pick_iters(bytes);
    
    for(int i=0;i<3;i++)
        CUDA_CHECK(cudaMemcpy(dst,src,bytes,kind));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t0 = clk::now();
    for(int i=0;i<iters;++i)
        CUDA_CHECK(cudaMemcpy(dst,src,bytes,kind));
    CUDA_CHECK(cudaDeviceSynchronize());
    double total=std::chrono::duration<double>(clk::now()-t0).count();
    double sec_per_copy = total/iters;
    return double(bytes)/sec_per_copy/1e9;
}//算速率

int main(int argc, char**argv){
    const size_t MIN_BYTES = 1024;
    size_t max_bytes = 1ull<<30;
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
    
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem,&total_mem));
    while(max_bytes > free_mem/4&&max_bytes>MIN_BYTES) max_bytes /=2;

    fprintf(stderr,"#device     :%s\n",prop.name);
    fprintf(stderr,"#显存       :%.1f GB(free %.1f GB)\n",total_mem/1e9,free_mem/1e9);
    fprintf(stderr,"#理论显存带宽:%.1f GB/S\n",2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    fprintf(stderr, "# 最大测试size: %.1f MB\n", max_bytes / 1e6);    

    // void *h_buf = malloc(max_bytes);
    // if(!h_buf){fprintf(stderr,"host malloc failed");return 1;}
    void *h_buf = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_buf,max_bytes));
    memset(h_buf,1,max_bytes);

    void* d_buf=nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf,max_bytes));

    printf("bytes,h2d_GBps,d2h_GBps\n");
    for(size_t bytes = MIN_BYTES;bytes <=max_bytes;bytes*=2){
        double h2d = measure(d_buf,h_buf,bytes,cudaMemcpyHostToDevice);
        double d2h = measure(h_buf,d_buf,bytes,cudaMemcpyDeviceToHost);
        printf("%zu,%.3f,%.3f\n",bytes,h2d,d2h);
        fflush(stdout);
    }

    CUDA_CHECK(cudaFree(d_buf));
    // free(h_buf);
    CUDA_CHECK(cudaFreeHost(h_buf));
    return 0;
}
