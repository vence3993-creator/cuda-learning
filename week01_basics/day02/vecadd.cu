#include "../../common/error.cuh"
#include <cmath>          // fabs
#include <cstdio>         // printf
#include <cstdlib>        // malloc/free/atoi

const int N = 1 << 24;
const float EPS = 1e-5f;

// ── kernel 定义（已有的四个版本）──────────────────
__global__ void add_v0(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
}
__global__ void add_v1(const float* a, const float* b, float* c, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) c[i] = a[i] + b[i];
}
__global__ void add_v2(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}
__global__ void add_v3(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main(int argc, char** argv) {
    // ① 选版本
    const int ver = (argc > 1) ? atoi(argv[1]) : 3;

    // ② 打印设备信息，测性能前先知道天花板在哪
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    const double peakBW = 2.0 * prop.memoryClockRate * 1e3
                        * (prop.memoryBusWidth / 8) / 1e9;
    printf("GPU: %s | SM=%d | peak BW ~%.0f GB/s\n\n",
           prop.name, prop.multiProcessorCount, peakBW);

    // ③ 主机内存
    const size_t bytes = (size_t)N * sizeof(float);
    float *ha = (float*)malloc(bytes);
    float *hb = (float*)malloc(bytes);
    float *hc = (float*)malloc(bytes);
    float *ref = (float*)malloc(bytes);
    if (!ha || !hb || !hc || !ref) { fprintf(stderr, "malloc failed\n"); return 1; }

    for (int i = 0; i < N; ++i) { ha[i] = 1.0f; hb[i] = 2.0f; }
    for (int i = 0; i < N; ++i) ref[i] = ha[i] + hb[i];

    // ④ 显存
    float *da, *db, *dc;
    CHECK(cudaMalloc(&da, bytes));
    CHECK(cudaMalloc(&db, bytes));
    CHECK(cudaMalloc(&dc, bytes));

    CHECK(cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dc, 0, bytes));          // 清零，避免残留掩盖错误

    // ⑤ 启动配置
    const int T = 256;
    const int B = (N + T - 1) / T;

    cudaEvent_t s, e;
    CHECK(cudaEventCreate(&s));
    CHECK(cudaEventCreate(&e));

    // ⑥ 预热一次：排除首次调用的上下文初始化开销
    add_v3<<<B, T>>>(da, db, dc, N);
    CHECK_KERNEL();
    CHECK(cudaMemset(dc, 0, bytes));

    // ⑦ 正式计时
    CHECK(cudaEventRecord(s));
    switch (ver) {
        case 0:  add_v0<<<1, 1>>>(da, db, dc, N); break;
        case 1:  add_v1<<<1, T>>>(da, db, dc, N); break;
        case 2:  add_v2<<<B, T>>>(da, db, dc, N); break;
        default: add_v3<<<B, T>>>(da, db, dc, N); break;
    }
    CHECK(cudaEventRecord(e));
    CHECK_KERNEL();

    float ms = 0.f;
    CHECK(cudaEventElapsedTime(&ms, s, e));

    // ⑧ 拷回验证
    CHECK(cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < N; ++i)
        if (fabs(hc[i] - ref[i]) > EPS) ++bad;

    printf("v%d: %8.3f ms | %6.1f GB/s (%.0f%% of peak) | mismatches = %d\n",
           ver, ms, 3.0 * bytes / ms / 1e6,
           100.0 * (3.0 * bytes / ms / 1e6) / peakBW, bad);

    // ⑨ 释放
    CHECK(cudaEventDestroy(s));
    CHECK(cudaEventDestroy(e));
    CHECK(cudaFree(da)); CHECK(cudaFree(db)); CHECK(cudaFree(dc));
    free(ha); free(hb); free(hc); free(ref);
    return bad == 0 ? 0 : 1;
}