// ============================================================================
// benchmark.h —— 通用 CUDA kernel 计时 / 带宽 / 算力测量
//
// 用法：
//     #include "benchmark.h"
//     auto launch = [&]{ myKernel<<<grid, block>>>(...); };
//     BenchResult r = benchmark(launch, bytes, flops);
//     print_result("my_kernel", r);
//
// 编译：nvcc -O2 -I<本文件所在目录> your.cu -o your
// ============================================================================
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <numeric>

// ============================================================================
// 1. 错误检查
//    宏在预处理阶段展开，不存在 ODR 问题，可以直接放 header。
// ============================================================================
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d -> %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err_));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// kernel launch 本身不返回错误码，必须显式检查。
// 放在 launch 之后：抓 launch 配置错误（grid=0、block>1024、shmem 超限等）。
#define CUDA_CHECK_LAUNCH()                                                    \
    do {                                                                       \
        CUDA_CHECK(cudaGetLastError());                                        \
        CUDA_CHECK(cudaDeviceSynchronize());                                   \
    } while (0)

// ============================================================================
// 2. GpuTimer —— 手动打点，适合临时量一段代码
//    需要"跑 N 次取分布"时用下面的 benchmark<>，不要用这个。
// ============================================================================
class GpuTimer {
public:
    explicit GpuTimer(cudaStream_t s = 0) : stream_(s) {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        // 析构函数里不要用 CUDA_CHECK：它会 exit()，
        // 在栈展开（异常）过程中调用 exit 行为很难调试。
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    GpuTimer(const GpuTimer&)            = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void tic() { CUDA_CHECK(cudaEventRecord(start_, stream_)); }

    float toc() {  // 返回毫秒
        CUDA_CHECK(cudaEventRecord(stop_, stream_));
        CUDA_CHECK(cudaEventSynchronize(stop_));  // 少了这句 -> cudaErrorNotReady
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t  start_, stop_;
    cudaStream_t stream_;
};

// ============================================================================
// 3. 测量结果
//    bytes / flops 由调用方提供 —— 只有调用方知道自己的 kernel 搬了多少数据。
//    class 内部定义的成员函数自动是 inline 的，不需要手写 inline。
// ============================================================================
struct BenchResult {
    float  mean_ms   = 0.f;
    float  median_ms = 0.f;
    float  min_ms    = 0.f;
    float  max_ms    = 0.f;
    int    iters     = 0;
    size_t bytes     = 0;   // 一次调用必须搬运的总字节数
    size_t flops     = 0;   // 一次调用的浮点运算次数，0 表示不关心

    // 统一用 median：中位数抗噪声，偶尔一次被系统调度打断的慢样本不会污染结果。
    double gbps()   const { return bytes / (median_ms * 1e-3) / 1e9; }
    double gflops() const { return flops / (median_ms * 1e-3) / 1e9; }

    // min 是"这张卡在最理想情况下的能力"，噪声只会让时间变长不会变短。
    // median 和 min 差很多 => 测试环境有干扰（其他进程 / 降频 / 散热）。
    double gbps_best() const { return bytes / (min_ms * 1e-3) / 1e9; }
};

// ============================================================================
// 4. benchmark<> —— 核心
//    Body 是模板参数而不是 std::function：lambda 会被内联，零调用开销。
//    模板函数天然是 inline 语义，放 header 不会重复定义。
// ============================================================================
template <class Body>
BenchResult benchmark(Body&& body,
                      size_t  bytes,
                      size_t  flops  = 0,
                      int     warmup = 10,
                      int     iters  = 200,
                      cudaStream_t stream = 0)
{
    // ---- warmup ----
    // 消除三件事：CUDA context 创建、module 首次加载/JIT、GPU 从低频 boost 上来。
    for (int i = 0; i < warmup; ++i) body();
    CUDA_CHECK(cudaDeviceSynchronize());  // 把管子清空，warmup 不计入
    CUDA_CHECK(cudaGetLastError());       // launch 配置写错了在这里就暴露

    // ---- 准备 event ----
    // 每次迭代一对 event，事后统一读数，测量循环里不做任何同步。
    std::vector<cudaEvent_t> ev_a(iters), ev_b(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventCreate(&ev_a[i]));
        CUDA_CHECK(cudaEventCreate(&ev_b[i]));
    }

    // ---- 测量 ----
    // 循环体里只有 launch，没有 cudaMemcpy，没有 cudaDeviceSynchronize。
    // 一旦塞了拷贝，你测的就是 PCIe 而不是 kernel。
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(ev_a[i], stream));
        body();
        CUDA_CHECK(cudaEventRecord(ev_b[i], stream));
    }
    CUDA_CHECK(cudaEventSynchronize(ev_b[iters - 1]));  // 只等最后一个

    // ---- 读数 ----
    std::vector<float> t(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventElapsedTime(&t[i], ev_a[i], ev_b[i]));
        cudaEventDestroy(ev_a[i]);
        cudaEventDestroy(ev_b[i]);
    }

    std::vector<float> sorted = t;
    std::sort(sorted.begin(), sorted.end());

    BenchResult r;
    r.iters     = iters;
    r.bytes     = bytes;
    r.flops     = flops;
    r.min_ms    = sorted.front();
    r.max_ms    = sorted.back();
    r.median_ms = sorted[iters / 2];
    // 求和用 double：float 累加 200 个小数会丢精度。
    r.mean_ms   = static_cast<float>(
        std::accumulate(t.begin(), t.end(), 0.0) / iters);
    return r;
}

// ============================================================================
// 5. 输出
//    普通函数放 header 必须加 inline，否则多个 .cu include 会 multiple definition。
// ============================================================================
inline void print_result_header() {
    printf("%-22s %10s %10s %10s %10s %10s\n",
           "kernel", "median/ms", "min/ms", "max/ms", "GB/s", "GFLOP/s");
    printf("--------------------------------------------------------------"
           "----------------\n");
}

inline void print_result(const char* name, const BenchResult& r) {
    printf("%-22s %10.5f %10.5f %10.5f %10.2f",
           name, r.median_ms, r.min_ms, r.max_ms, r.gbps());
    if (r.flops) printf(" %10.2f", r.gflops());
    else         printf(" %10s", "-");
    printf("\n");
}

// ============================================================================
// 6. 设备信息 / 理论峰值带宽
// ============================================================================
inline double query_peak_bandwidth_gbs(int dev = 0) {
    int clk_khz = 0, bus_bits = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clk_khz,  cudaDevAttrMemoryClockRate,      dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev));
    if (clk_khz == 0 || bus_bits == 0) {
        fprintf(stderr,
                "[warn] 驱动未返回显存频率/位宽（HBM 卡或新驱动常见）。\n"
                "       请查官方 spec 页手动填一个常量，别在这里卡住。\n");
        return 0.0;
    }
    // ×2 是 DDR 双沿传输；clk_khz -> Hz，bus_bits -> byte
    return 2.0 * clk_khz * 1e3 * (bus_bits / 8.0) / 1e9;
}

inline void print_device_info(int dev = 0) {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    printf("GPU              : %s (sm_%d%d)\n", p.name, p.major, p.minor);
    printf("SM 数量          : %d\n", p.multiProcessorCount);
    printf("L2 cache         : %.1f MB   <- 工作集必须远大于它，否则测的是 L2\n",
           p.l2CacheSize / 1048576.0);
    printf("显存容量         : %.1f GB\n", p.totalGlobalMem / 1073741824.0);
    double peak = query_peak_bandwidth_gbs(dev);
    if (peak > 0) printf("理论显存带宽峰值 : %.1f GB/s\n", peak);
    printf("\n");
}