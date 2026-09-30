// week03/day04/softmax_warp.cu
// W3 D4：safe softmax，[M, N] 按行做 softmax
//
// 编译：make
// 运行：./softmax_warp          正确性 + 计时
//      ./softmax_warp ncu      每个 kernel 只跑一次，给 ncu 用（make ncu）
//
// ② 阶段：kernel 全是空的，所有版本都会显示 FAIL —— 这是正常的，说明检查在起作用
// ③ 阶段：逐个填 TODO，每填一个跑一次

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include "benchmark.h"      // CUDA_CHECK / CUDA_CHECK_LAUNCH / benchmark<> / BenchResult
#include "reduce_utils.cuh" // warpReduceMax / warpReduceSum：xor 蝶形，32 个 lane 都拿到结果

#define WARP_SIZE 32
//输入是一个[M,N]矩阵，按行主序存储在显存里，也就是第r行从r*N开始，一行N个元素紧挨着。softmax沿每一行做，
//每行各自求自己的max和sum，航宇航之间互不相关
//放到大模型中，M通常是batch*seq数，N是词表大小（输出层）或者attention分数的长度，

constexpr int N = 1024;                                   // 每行长度
constexpr int M = (1 << 24) / N;                          // 行数 = 16384
constexpr int WARPS_PER_BLOCK = 4;                        // blockDim = 128，一个 block 处理 4 行
constexpr size_t BYTES = size_t(M) * N * sizeof(float);   // 64 MB
constexpr size_t TRAFFIC = 2 * BYTES;                     // 读 64 MB + 写 64 MB

// ============================ kernels（③ 里逐个填） ============================

// v0：不减 max。只用来展示溢出，不计时
__global__ void softmax_v0_unsafe(const float* x, float* y, int m, int n) {
    // TODO ③
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if(row >= m) return;
    const float* xr = x + (size_t)row * n;
    float* yr = y + (size_t)row * n;

    float d = 0.f;
    for(int j = 0; j < n; j++)
        d += expf(xr[j]);

    float inv_d = 1.f/d;
    for(int j = 0; j < n; j++)
        yr[j] = expf(xr[j]) * inv_d;
}

// v1：一个线程处理一整行，三遍
__global__ void softmax_v1_thread_row(const float* x, float* y, int m, int n) {
    // TODO ③
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= m) return;
    const float* xr = x + (size_t)row * n;
    float* yr = y + (size_t)row * n;

    float mx = -INFINITY;
    for(int j=0;j<n;j++)
        mx = fmaxf(mx,xr[j]);

    float d = 0.f;
    for(int j = 0; j < n; j++)
        d+= expf(xr[j]-mx);

    float inv_d = 1.f/d;
    for(int j = 0; j < n; j++)
        yr[j] = (expf(xr[j]-mx)) * inv_d;
}

// v2：一个 warp 处理一行，三遍都从 global 读
__global__ void softmax_v2_warp_3pass(const float* x, float* y, int m, int n) {
    // TODO ③
    int warp_id = (blockIdx.x*blockDim.x+threadIdx.x)/32;
    int lane = threadIdx.x % 32;
    int row = warp_id;
    if(row >= m) return;

    const float* xr = x + (size_t)row * n;
    float* yr = y + (size_t)row * n;

    float mx = -INFINITY;
    for(int j = lane; j < n; j+=32)
        mx = fmaxf(mx,xr[j]);
    mx = warpReduceMax(mx);

    float d = 0.f;
    for(int j = lane; j < n; j+=32)
        d += expf(xr[j]-mx);

    d = warpReduceSum(d);

    float inv_d = 1.f/d;
    for(int j = lane; j < n; j+=32)
        yr[j] = expf(xr[j] - mx) * inv_d;
}

// v3：一个 warp 处理一行，整行读进寄存器 float buf[NCOL / 32]
template <int NCOL>
__global__ void softmax_v3_warp_reg(const float* x, float* y, int m, int n) {
    // TODO ③
    static_assert(NCOL % 32 == 0, "NCOL must be a multiple of 32");
    constexpr int VPT = NCOL/32;

    int warp_id = (blockDim.x * blockIdx.x + threadIdx.x)/32;
    int lane = threadIdx.x % 32;
    int row = warp_id;
    if(row >= m) return;

    const float* xr = x + (size_t)row * n;
    float* yr = y + (size_t)row * n;

    float buf[VPT];
    float mx = -INFINITY;
    #pragma unroll 
    for(int i = 0; i < VPT; ++i){
        int j = i * 32 + lane;
        buf[i] = (j<n)? xr[j] : -INFINITY;
        mx = fmaxf(mx,buf[i]);
    }
    mx = warpReduceMax(mx);

    float d = 0.f;
    #pragma unroll
    for(int i = 0; i < VPT; ++i){
        buf[i] = expf(buf[i] - mx);
        d+=buf[i];
    }
    d = warpReduceSum(d);

    float inv_d = 1.f/d;
    #pragma unroll
    for(int i = 0; i < VPT; ++i){
        int j = i * 32 + lane;
        if(j < n) yr[j] = buf[i] * inv_d;
    }
}

// v4：v3 + float4 向量化读写
// 要求：每行长度恰好为 NCOL，且 NCOL 是 128 的倍数（32 个 lane × 每次 4 个 float）
template <int NCOL>
__global__ void softmax_v4_warp_f4(const float* x, float* y, int m) {
    static_assert(NCOL % (WARP_SIZE * 4) == 0, "NCOL must be a multiple of 128");
    constexpr int V4 = NCOL / (WARP_SIZE * 4);   // 每个 lane 负责的 float4 个数

    int row  = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (row >= m) return;

    // 把本行首地址按 float4 来看待：一个 float4 = 4 个连续 float = 16 字节
    const float4* xr = reinterpret_cast<const float4*>(x + (size_t)row * NCOL);
    float4*       yr = reinterpret_cast<float4*>(y + (size_t)row * NCOL);

    float4 buf[V4];

    // pass1：向量化读入寄存器，求 max
    float mx = -INFINITY;
    #pragma unroll
    for (int i = 0; i < V4; ++i) {
        buf[i] = xr[i * WARP_SIZE + lane];   // 一条指令读 16 字节，warp 一次读连续 512 字节
        mx = fmaxf(mx, fmaxf(fmaxf(buf[i].x, buf[i].y), fmaxf(buf[i].z, buf[i].w)));
    }
    mx = warpReduceMax(mx);

    // pass2：寄存器里算 exp，求 d
    float d = 0.f;
    #pragma unroll
    for (int i = 0; i < V4; ++i) {
        buf[i].x = expf(buf[i].x - mx);
        buf[i].y = expf(buf[i].y - mx);
        buf[i].z = expf(buf[i].z - mx);
        buf[i].w = expf(buf[i].w - mx);
        d += (buf[i].x + buf[i].y) + (buf[i].z + buf[i].w);
    }
    d = warpReduceSum(d);

    // pass3：归一化，向量化写回
    float inv_d = 1.f / d;
    #pragma unroll
    for (int i = 0; i < V4; ++i) {
        buf[i].x *= inv_d;
        buf[i].y *= inv_d;
        buf[i].z *= inv_d;
        buf[i].w *= inv_d;
        yr[i * WARP_SIZE + lane] = buf[i];
    }
}
// ================================ launchers ================================

using Launcher = void (*)(const float*, float*);

void launch_v0(const float* x, float* y) { softmax_v0_unsafe<<<(M + 255) / 256, 256>>>(x, y, M, N); }
void launch_v1(const float* x, float* y) { softmax_v1_thread_row<<<(M + 255) / 256, 256>>>(x, y, M, N); }
void launch_v2(const float* x, float* y) { softmax_v2_warp_3pass<<<M / WARPS_PER_BLOCK, WARPS_PER_BLOCK * 32>>>(x, y, M, N); }
void launch_v3(const float* x, float* y) { softmax_v3_warp_reg<N><<<M / WARPS_PER_BLOCK, WARPS_PER_BLOCK * 32>>>(x, y, M, N); }
void launch_v4(const float* x, float* y) { softmax_v4_warp_f4<N><<<M / WARPS_PER_BLOCK, WARPS_PER_BLOCK * 32>>>(x, y, M); }

// ============================== CPU 参考 + 检查 ==============================

void softmax_cpu(const std::vector<float>& x, std::vector<float>& y) {
    for (int r = 0; r < M; r++) {
        const float* xr = &x[size_t(r) * N];
        float* yr = &y[size_t(r) * N];
        double mx = -INFINITY;
        for (int j = 0; j < N; j++) mx = std::max(mx, (double)xr[j]);
        double d = 0.0;
        for (int j = 0; j < N; j++) d += std::exp((double)xr[j] - mx);
        for (int j = 0; j < N; j++) yr[j] = float(std::exp((double)xr[j] - mx) / d);
    }
}

struct CheckResult {
    double max_abs_err = 0.0;     // 与 CPU 参考的最大绝对误差
    double max_rowsum_err = 0.0;  // 每行 |Σy − 1| 的最大值
    size_t bad = 0;               // NaN / inf 的个数
};

CheckResult check(const std::vector<float>& got, const std::vector<float>& ref) {
    CheckResult c;
    for (int r = 0; r < M; r++) {
        double s = 0.0;
        for (int j = 0; j < N; j++) {
            size_t k = size_t(r) * N + j;
            float g = got[k];
            if (!std::isfinite(g)) { c.bad++; continue; }
            c.max_abs_err = std::max(c.max_abs_err, std::fabs((double)g - ref[k]));
            s += g;
        }
        c.max_rowsum_err = std::max(c.max_rowsum_err, std::fabs(s - 1.0));
    }
    return c;
}

bool passed(const CheckResult& c) {
    return c.bad == 0 && c.max_abs_err < 1e-6 && c.max_rowsum_err < 1e-5;
}

// ================================== 运行 ==================================

// 跑一个版本：先毒化输出 → 单次运行查正确性 → （可选）计时
void run(const char* name, Launcher launch, const float* d_x, float* d_y,
         std::vector<float>& h_y, const std::vector<float>& ref, bool timed) {
    // 0xFF 字节拼成的 float 是 NaN：kernel 漏写任何一个位置都会被查出来
    CUDA_CHECK(cudaMemset(d_y, 0xFF, BYTES));
    launch(d_x, d_y);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, BYTES, cudaMemcpyDeviceToHost));
    CheckResult c = check(h_y, ref);

    printf("%-18s %s  max_abs=%.2e  rowsum=%.2e  nan/inf=%-9zu",
           name, passed(c) ? "PASS" : "FAIL", c.max_abs_err, c.max_rowsum_err, c.bad);
    if (timed) {
        BenchResult r = benchmark([&] { launch(d_x, d_y); }, TRAFFIC);
        printf("| median %.4f ms  min %.4f ms  %.1f GB/s", r.median_ms, r.min_ms, r.gbps());
    }
    printf("\n");
}

// ================================== main ==================================

int main(int argc, char** argv) {
    const bool ncu_mode = (argc > 1 && strcmp(argv[1], "ncu") == 0);
    const bool timed = !ncu_mode;

    print_device_info();
    printf("softmax [M=%d, N=%d]，%.0f MB 读 + %.0f MB 写，理论下限 %.3f ms @ 448 GB/s\n\n",
           M, N, BYTES / 1048576.0, BYTES / 1048576.0, TRAFFIC / 448e9 * 1e3);

    // 两组输入：常规 [-1, 1)；溢出组 [80, 100)
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist_normal(-1.f, 1.f), dist_big(80.f, 100.f);
    std::vector<float> h_x(size_t(M) * N), h_x_big(size_t(M) * N);
    for (auto& v : h_x) v = dist_normal(gen);
    for (auto& v : h_x_big) v = dist_big(gen);

    std::vector<float> ref(h_x.size()), ref_big(h_x.size()), h_y(h_x.size());
    printf("CPU 参考计算中...\n");
    softmax_cpu(h_x, ref);
    softmax_cpu(h_x_big, ref_big);

    float *d_x, *d_x_big, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, BYTES));
    CUDA_CHECK(cudaMalloc(&d_x_big, BYTES));
    CUDA_CHECK(cudaMalloc(&d_y, BYTES));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_big, h_x_big.data(), BYTES, cudaMemcpyHostToDevice));

    if (timed) {
        BenchResult r = benchmark([&] {
            CUDA_CHECK(cudaMemcpyAsync(d_y, d_x, BYTES, cudaMemcpyDeviceToDevice));
        }, TRAFFIC);
        printf("\n== 上限：D2D copy ==\n%-18s median %.4f ms  min %.4f ms  %.1f GB/s\n",
               "D2D copy", r.median_ms, r.min_ms, r.gbps());
    }

    printf("\n== 常规输入 [-1, 1) ==\n");
    run("v0 unsafe", launch_v0, d_x, d_y, h_y, ref, false);
    run("v1 thread/row", launch_v1, d_x, d_y, h_y, ref, timed);
    run("v2 warp 3-pass", launch_v2, d_x, d_y, h_y, ref, timed);
    run("v3 warp reg", launch_v3, d_x, d_y, h_y, ref, timed);
    run("v4 warp float4", launch_v4, d_x, d_y, h_y, ref, timed);

    printf("\n== 溢出组 [80, 100)：v0 应该 FAIL（NaN），v3 应该 PASS ==\n");
    run("v0 unsafe", launch_v0, d_x_big, d_y, h_y, ref_big, false);
    run("v3 warp reg", launch_v3, d_x_big, d_y, h_y, ref_big, false);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_x_big));
    CUDA_CHECK(cudaFree(d_y));
    return 0;
}
