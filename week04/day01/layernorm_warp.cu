// week04/day01/layernorm_warp.cu
// W4 D1：LayerNorm，[M, N] 按行归一化，一个 warp 处理一行
//
// 编译：make（Makefile 从 W3 D4 复制，目标名改成 layernorm_warp，保留 -I../../common）
// 运行：./layernorm_warp          参考体检 + 正确性 + 计时
//      ./layernorm_warp ncu      只跑常规组、每个 kernel 一次，给 ncu 用
//
// ② 阶段：kernel 全是空的，所有版本都显示 FAIL（nan/inf = M*N）—— 正常，说明检查在起作用
// ③ 阶段：先把三个预测写进 notes，再逐个填 TODO，每填一个跑一次

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include "benchmark.h"      // CUDA_CHECK / CUDA_CHECK_LAUNCH / benchmark<> / BenchResult
#include "reduce_utils.cuh" // warpReduceSum：xor 蝶形 all-reduce，32 个 lane 都拿到结果

#define WARP_SIZE 32
#ifndef NCOLS
#define NCOLS 1024
#endif
constexpr int N = NCOLS;                                   // 每行长度。D2 预告：改成 4096 / 8192 配合 -Xptxas -v 看 spill
constexpr int M = (1 << 24) / N;                          // 行数 = 16384
constexpr int WARPS_PER_BLOCK = 4;                        // blockDim = 128，一个 block 处理 4 行
constexpr int BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
constexpr int GRID = M / WARPS_PER_BLOCK;
constexpr float EPS = 1e-5f;
constexpr size_t BYTES = size_t(M) * N * sizeof(float);   // 64 MB
constexpr size_t TRAFFIC = 2 * BYTES;                     // 读 x 64 MB + 写 y 64 MB；γ、β 只有 N 个，忽略

static_assert(M % WARPS_PER_BLOCK == 0, "M must be a multiple of WARPS_PER_BLOCK"); 

// ============================ kernels（③ 里逐个填） ============================
// 统一签名：x, gamma, beta → y。一个 warp 一行：row = 全局线程号 / 32，lane = threadIdx.x % 32

// v0：单遍公式（对照组）。寄存器缓存一行；一轮 all-reduce 同时求 Σx 和 Σx²，σ² = E[x²] − μ²
template <int NCOL>
__global__ void layernorm_v0_onepass(const float* x, const float* gamma, const float* beta,
                                     float* y, int m, float eps) {
    static_assert(NCOL % WARP_SIZE == 0, "NCOL must be a multiple of 32");
    constexpr int VPT = NCOL / WARP_SIZE;            // 每个 lane 负责的元素数

    int row  = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (row >= m) return;                            // 同一 warp 的 row 相同，整个 warp 一起退出，不影响 shuffle

    const float* xr = x + (size_t)row * NCOL;
    float*       yr = y + (size_t)row * NCOL;

    // 只读一次 DRAM，同时累加 Σx 和 Σx²
    float buf[VPT];
    float s1 = 0.f, s2 = 0.f;
    #pragma unroll
    for (int i = 0; i < VPT; ++i) {
        float v = xr[i * WARP_SIZE + lane];          // warp 一次读连续 128 B
        buf[i] = v;
        s1 += v;
        s2 += v * v;
    }
    s1 = warpReduceSum(s1);                          // 32 个 lane 都拿到 Σx
    s2 = warpReduceSum(s2);                          // 32 个 lane 都拿到 Σx²

    float mu   = s1 / NCOL;
    float var  = s2 / NCOL - mu * mu;                // 单遍公式 —— 偏移组就是在这一行崩掉的
    float rstd = rsqrtf(var + eps);

    #pragma unroll
    for (int i = 0; i < VPT; ++i) {
        int j = i * WARP_SIZE + lane;
        yr[j] = (buf[i] - mu) * rstd * gamma[j] + beta[j];
    }
}

// v1：两遍（寄存器上）。float buf[NCOL / 32]；
//     all-reduce 求 μ → 寄存器上算 Σ(x − μ)² → all-reduce 求 σ² → 归一化写出
template <int NCOL>
__global__ void layernorm_v1_twopass(const float* x, const float* gamma, const float* beta,
                                     float* y, int m, float eps) {
    // TODO ③
    static_assert(NCOL % WARP_SIZE == 0, "NCOL must be a multiple of 32");
    int row = (blockDim.x * blockIdx.x + threadIdx.x) / WARP_SIZE;
    if(row >= m) return;
    int lane = threadIdx.x % WARP_SIZE;
    constexpr int VPT = NCOL / WARP_SIZE;

    const float* xr = x + (size_t)row * NCOL;
    float* yr = y + (size_t)row * NCOL;

    float buf[VPT];
    float s1 = 0.f, s2 = 0.f;
    #pragma unroll
    for(int i = 0; i < VPT; ++i){
        float v = xr[i * WARP_SIZE + lane];
        buf[i] = v;
        s1 += v;
    }
    s1 = warpReduceSum(s1);
    float mu = s1/NCOL;
    #pragma unroll
    for(int i = 0; i < VPT; ++i){
        buf[i] -= mu;
        s2 += buf[i] * buf[i];
    }
    s2 = warpReduceSum(s2);
    float var = s2 / NCOL;
    float rstd = rsqrtf(var + eps);

    #pragma unroll
    for(int i = 0; i < VPT; ++i){
        int j = i * WARP_SIZE + lane;
        yr[j] = buf[i] * rstd * gamma[j] + beta[j];
    }
}

// v2：v1 + float4。x、y、γ、β 全部按 float4 读写；第 i 次访问下标 i * 32 + lane（以 float4 为单位）
//     要求 NCOL 是 128 的倍数
template <int NCOL>
__global__ void layernorm_v2_f4(const float* x, const float* gamma, const float* beta,
                                float* y, int m, float eps) {
    static_assert(NCOL % (WARP_SIZE * 4) == 0, "NCOL must be a multiple of 128");
    constexpr int V4 = NCOL / (WARP_SIZE * 4);       // 每个 lane 负责几个 float4（N=1024 → 8 个，仍然是 32 个 float）

    int row  = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (row >= m) return;

    // 把指针当成 float4 来看：下标 k 对应第 4k ~ 4k+3 个 float
    const float4* xr = reinterpret_cast<const float4*>(x + (size_t)row * NCOL);
    float4*       yr = reinterpret_cast<float4*>(y + (size_t)row * NCOL);
    const float4* g4 = reinterpret_cast<const float4*>(gamma);
    const float4* b4 = reinterpret_cast<const float4*>(beta);

    float4 buf[V4];

    // 第 1 遍：读入寄存器，求 Σx
    float s1 = 0.f;
    #pragma unroll
    for (int i = 0; i < V4; ++i) {
        buf[i] = xr[i * WARP_SIZE + lane];            // 一条指令读 16 B，warp 一次读连续 512 B
        s1 += (buf[i].x + buf[i].y) + (buf[i].z + buf[i].w);
    }
    s1 = warpReduceSum(s1);
    const float mu = s1 / NCOL;

    // 第 2 遍：寄存器上减均值，求 Σ(x − μ)²
    float s2 = 0.f;
    #pragma unroll
    for (int i = 0; i < V4; ++i) {
        buf[i].x -= mu;  buf[i].y -= mu;  buf[i].z -= mu;  buf[i].w -= mu;
        s2 += buf[i].x * buf[i].x + buf[i].y * buf[i].y
            + buf[i].z * buf[i].z + buf[i].w * buf[i].w;
    }
    s2 = warpReduceSum(s2);
    const float rstd = rsqrtf(s2 / NCOL + eps);

    // 第 3 遍：缩放、平移，向量化写回
    #pragma unroll
    for (int i = 0; i < V4; ++i) {
        int k = i * WARP_SIZE + lane;                 // x 和 γ/β 用同一个 float4 下标
        float4 g = g4[k], b = b4[k];
        float4 o;
        o.x = buf[i].x * rstd * g.x + b.x;
        o.y = buf[i].y * rstd * g.y + b.y;
        o.z = buf[i].z * rstd * g.z + b.z;
        o.w = buf[i].w * rstd * g.w + b.w;
        yr[k] = o;
    }
}
// ================================ launchers ================================

using Launcher = void (*)(const float*, const float*, const float*, float*);

void launch_v0(const float* x, const float* g, const float* b, float* y) { layernorm_v0_onepass<N><<<GRID, BLOCK>>>(x, g, b, y, M, EPS); }
void launch_v1(const float* x, const float* g, const float* b, float* y) { layernorm_v1_twopass<N><<<GRID, BLOCK>>>(x, g, b, y, M, EPS); }
void launch_v2(const float* x, const float* g, const float* b, float* y) { layernorm_v2_f4<N><<<GRID, BLOCK>>>(x, g, b, y, M, EPS); }

// ============================== CPU 参考 + 检查 ==============================

// double 两遍法。吃的是 GPU 用的同一份 float 输入
void layernorm_cpu(const float* x, const float* g, const float* b, float* y, int rows, double eps = 1e-5) {
    for (int r = 0; r < rows; r++) {
        const float* xr = x + size_t(r) * N;
        float* yr = y + size_t(r) * N;
        double mu = 0.0, var = 0.0;
        for (int j = 0; j < N; j++) mu += xr[j];
        mu /= N;
        for (int j = 0; j < N; j++) { double d = xr[j] - mu; var += d * d; }
        var /= N;
        double rstd = 1.0 / std::sqrt(var + eps);
        for (int j = 0; j < N; j++) yr[j] = float((xr[j] - mu) * rstd * g[j] + b[j]);
    }
}

// 参考体检：γ = 1、β = 0 时，每行输出应满足 均值 ≈ 0、方差 ≈ 1
void ref_selfcheck(const char* name, const std::vector<float>& x) {
    const int rows = 4;
    std::vector<float> ones(N, 1.f), zeros(N, 0.f), out(size_t(rows) * N);
    layernorm_cpu(x.data(), ones.data(), zeros.data(), out.data(), rows);
    double worst_mean = 0.0, worst_var = 0.0;
    for (int r = 0; r < rows; r++) {
        const float* o = &out[size_t(r) * N];
        double mu = 0.0, var = 0.0;
        for (int j = 0; j < N; j++) mu += o[j];
        mu /= N;
        for (int j = 0; j < N; j++) var += (o[j] - mu) * (o[j] - mu);
        var /= N;
        worst_mean = std::max(worst_mean, std::fabs(mu));
        worst_var = std::max(worst_var, std::fabs(var - 1.0));
    }
    bool ok = worst_mean < 1e-4 && worst_var < 1e-3;
    printf("参考体检 %-12s 前 %d 行：max|均值| = %.2e  max|方差-1| = %.2e  %s\n",
           name, rows, worst_mean, worst_var, ok ? "OK" : "<<< 参考实现有问题！");
}

struct CheckResult {
    double max_abs_err = 0.0;   // 与 CPU 参考的最大绝对误差（只统计有限值）
    size_t bad = 0;             // NaN / inf 的个数 —— 单独统计，fmax 会吞掉 NaN
};

CheckResult check(const std::vector<float>& got, const std::vector<float>& ref) {
    CheckResult c;
    for (size_t k = 0; k < got.size(); k++) {
        float g = got[k];
        if (!std::isfinite(g)) { c.bad++; continue; }
        c.max_abs_err = std::max(c.max_abs_err, std::fabs((double)g - ref[k]));
    }
    return c;
}

bool passed(const CheckResult& c) {
    return c.bad == 0 && c.max_abs_err < 1e-4;
}

// ================================== 运行 ==================================

double g_d2d_gbps = 0.0;   // D2D 上限，用来算百分比

// 跑一个版本：先毒化输出 → 单次运行查正确性（只报告，不中止）→ （可选）计时
void run(const char* name, Launcher launch, const float* d_x, const float* d_g, const float* d_b,
         float* d_y, std::vector<float>& h_y, const std::vector<float>& ref, bool timed) {
    // 0xFF 字节拼成的 float 是 NaN：kernel 漏写任何一个位置都会被查出来
    CUDA_CHECK(cudaMemset(d_y, 0xFF, BYTES));
    launch(d_x, d_g, d_b, d_y);
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, BYTES, cudaMemcpyDeviceToHost));
    CheckResult c = check(h_y, ref);

    printf("%-16s %s  max_abs=%.2e  nan/inf=%-9zu", name, passed(c) ? "PASS" : "FAIL", c.max_abs_err, c.bad);
    if (timed) {
        BenchResult r = benchmark([&] { launch(d_x, d_g, d_b, d_y); }, TRAFFIC);
        printf("| median %.4f ms  min %.4f ms  %.1f GB/s", r.median_ms, r.min_ms, r.gbps());
        if (g_d2d_gbps > 0) printf("  (%.1f%% D2D)", 100.0 * r.gbps() / g_d2d_gbps);
    }
    printf("\n");
}

// ================================== main ==================================

int main(int argc, char** argv) {
    const bool ncu_mode = (argc > 1 && strcmp(argv[1], "ncu") == 0);
    const bool timed = !ncu_mode;

    print_device_info();
    printf("layernorm [M=%d, N=%d]，%.0f MB 读 + %.0f MB 写，理论下限 %.3f ms @ 448 GB/s\n\n",
           M, N, BYTES / 1048576.0, BYTES / 1048576.0, TRAFFIC / 448e9 * 1e3);

    // 三组输入：常规 [-1, 1)；偏移组 1000 + U[-1, 1)；偏移组 1e4 + U[-1, 1)（预测 1 的追问）
    // γ ∈ [0.5, 1.5)、β ∈ [-0.5, 0.5)：随机取值，范围是自定的，只要求输出量级 ~1
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f), dist_g(0.5f, 1.5f), dist_b(-0.5f, 0.5f);
    std::vector<float> h_x(size_t(M) * N), h_x_1k(h_x.size()), h_x_10k(h_x.size());
    for (auto& v : h_x) v = dist(gen);
    for (auto& v : h_x_1k) v = 1000.f + dist(gen);     // 在 float 里生成，CPU 参考和 GPU 用的是同一份
    for (auto& v : h_x_10k) v = 10000.f + dist(gen);
    std::vector<float> h_g(N), h_b(N);
    for (auto& v : h_g) v = dist_g(gen);
    for (auto& v : h_b) v = dist_b(gen);

    ref_selfcheck("常规", h_x);
    ref_selfcheck("偏移 1000", h_x_1k);
    ref_selfcheck("偏移 1e4", h_x_10k);

    std::vector<float> ref(h_x.size()), ref_1k(h_x.size()), ref_10k(h_x.size()), h_y(h_x.size());
    printf("\nCPU 参考计算中...\n");
    layernorm_cpu(h_x.data(), h_g.data(), h_b.data(), ref.data(), M);
    layernorm_cpu(h_x_1k.data(), h_g.data(), h_b.data(), ref_1k.data(), M);
    layernorm_cpu(h_x_10k.data(), h_g.data(), h_b.data(), ref_10k.data(), M);

    float *d_x, *d_x_1k, *d_x_10k, *d_g, *d_b, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, BYTES));
    CUDA_CHECK(cudaMalloc(&d_x_1k, BYTES));
    CUDA_CHECK(cudaMalloc(&d_x_10k, BYTES));
    CUDA_CHECK(cudaMalloc(&d_g, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, BYTES));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_1k, h_x_1k.data(), BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_10k, h_x_10k.data(), BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g, h_g.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    if (timed) {
        BenchResult r = benchmark([&] {
            CUDA_CHECK(cudaMemcpyAsync(d_y, d_x, BYTES, cudaMemcpyDeviceToDevice));
        }, TRAFFIC);
        g_d2d_gbps = r.gbps();
        printf("\n== 上限：D2D copy ==\n%-16s median %.4f ms  min %.4f ms  %.1f GB/s\n",
               "D2D copy", r.median_ms, r.min_ms, r.gbps());
    }

    printf("\n== 常规输入 [-1, 1) ==\n");
    run("v0 onepass", launch_v0, d_x, d_g, d_b, d_y, h_y, ref, timed);
    run("v1 twopass reg", launch_v1, d_x, d_g, d_b, d_y, h_y, ref, timed);
    run("v2 float4", launch_v2, d_x, d_g, d_b, d_y, h_y, ref, timed);

    if (!ncu_mode) {
        printf("\n== 偏移组 1000 + U[-1, 1) ==\n");
        run("v0 onepass", launch_v0, d_x_1k, d_g, d_b, d_y, h_y, ref_1k, false);
        run("v1 twopass reg", launch_v1, d_x_1k, d_g, d_b, d_y, h_y, ref_1k, false);
        run("v2 float4", launch_v2, d_x_1k, d_g, d_b, d_y, h_y, ref_1k, false);

        printf("\n== 偏移组 1e4 + U[-1, 1) ==\n");
        run("v0 onepass", launch_v0, d_x_10k, d_g, d_b, d_y, h_y, ref_10k, false);
        run("v1 twopass reg", launch_v1, d_x_10k, d_g, d_b, d_y, h_y, ref_10k, false);
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_x_1k));
    CUDA_CHECK(cudaFree(d_x_10k));
    CUDA_CHECK(cudaFree(d_g));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_y));
    return 0;
}
