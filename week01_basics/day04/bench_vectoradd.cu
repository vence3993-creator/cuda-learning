#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d -> %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err_));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// ============================================================================
// 工具 1：GpuTimer —— 手动打点，适合临时量一段代码
// ============================================================================
class GpuTimer {
public:
    explicit GpuTimer(cudaStream_t s = 0) : stream_(s) {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);   // 析构里不要 CUDA_CHECK/exit
        cudaEventDestroy(stop_);
    }
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void tic() { CUDA_CHECK(cudaEventRecord(start_, stream_)); }

    float toc() {                                   // 返回毫秒
        CUDA_CHECK(cudaEventRecord(stop_, stream_));
        CUDA_CHECK(cudaEventSynchronize(stop_));    // 少了这句 -> cudaErrorNotReady
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t  start_, stop_;
    cudaStream_t stream_;
};

struct BenchResult {
    float mean_ms, median_ms, min_ms, max_ms;
    int   iters;
};

template <class Body>
BenchResult benchmark(Body&& body, int warmup = 10, int iters = 200,
                      cudaStream_t stream = 0) {
    for (int i = 0; i < warmup; ++i) body();
    CUDA_CHECK(cudaDeviceSynchronize());   // warmup 收尾：把管子清空，不计入

    std::vector<cudaEvent_t> ev_a(iters), ev_b(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventCreate(&ev_a[i]));
        CUDA_CHECK(cudaEventCreate(&ev_b[i]));
    }

    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(ev_a[i], stream));
        body();
        CUDA_CHECK(cudaEventRecord(ev_b[i], stream));
    }
    CUDA_CHECK(cudaEventSynchronize(ev_b[iters - 1]));   // 只等最后一个

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
    r.min_ms    = sorted.front();
    r.max_ms    = sorted.back();
    r.median_ms = sorted[iters / 2];
    r.mean_ms   = std::accumulate(t.begin(), t.end(), 0.0f) / iters;
    return r;
}

// 用 host 计时器的辅助
using Clock = std::chrono::high_resolution_clock;
static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ============================================================================
int main() {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    int mem_clk_khz = 0, bus_bits = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clk_khz, cudaDevAttrMemoryClockRate, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits,    cudaDevAttrGlobalMemoryBusWidth, 0));
    double peak_gbs = 2.0 * mem_clk_khz * 1e3 * (bus_bits / 8.0) / 1e9;  // DDR -> x2

    printf("GPU: %s  (sm_%d%d)\n", p.name, p.major, p.minor);
    printf("理论显存带宽峰值: %.1f GB/s\n\n", peak_gbs);

    const int n     = 1 << 26;
    size_t    bytes = (size_t)n * sizeof(float);

    float *ha = (float*)malloc(bytes);
    float *hb = (float*)malloc(bytes);
    float *hc = (float*)malloc(bytes);
    for (int i = 0; i < n; ++i) { ha[i] = i * 1.0f; hb[i] = i * 2.0f; }

    float *da, *db, *dc;
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMalloc(&dc, bytes));
    CUDA_CHECK(cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice));

    const int block = 256;
    const int grid  = (n + block - 1) / block;

        // 被测对象统一封装成一个 lambda：只有 launch，没有同步，没有拷贝
    auto launch = [&] { vecAdd<<<grid, block>>>(da, db, dc, n); };

    // ---- 全局 warmup：第一次 launch 包含 context 建立/模块加载，可能几十上百 ms ----
    for (int i = 0; i < 20; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    const int ITERS = 200;

    // ================= 方式 1：chrono，不同步 =================
    // 计时代码位置：t0 在 launch 前，t1 紧跟 launch 后。中间什么同步都没有。
    // 量到的是「CPU 把 kernel 塞进队列并返回」的时间，不是 kernel 的时间。
    CUDA_CHECK(cudaDeviceSynchronize());          // 起点干净：之前的活干完
    auto t0 = Clock::now();
    launch();
    double m1_single = ms_since(t0);
    CUDA_CHECK(cudaDeviceSynchronize());          // 收尾，不计入

    // 同样不同步，但循环很多次：队列（约 1024 个 pending launch）填满后
    // launch 本身会开始阻塞，平均值会被"反压"拉回真实值附近。
    const int BIG = 5000;
    CUDA_CHECK(cudaDeviceSynchronize());
    t0 = Clock::now();
    for (int i = 0; i < BIG; ++i) launch();
    double m1_loop = ms_since(t0) / BIG;
    CUDA_CHECK(cudaDeviceSynchronize());

    // ================= 方式 2：chrono + cudaDeviceSynchronize =================
    // 2a：同步放在循环内部 —— 每次都等 GPU 干完。
    //     量到的是「单次端到端延迟」= launch 开销 + kernel + 同步返回开销。
    CUDA_CHECK(cudaDeviceSynchronize());
    t0 = Clock::now();
    for (int i = 0; i < ITERS; ++i) {
        launch();
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    double m2a = ms_since(t0) / ITERS;

    // 2b：同步放在循环外面 —— 这才是学习计划里说的"正确位置"。
    //     launch 开销被流水线掩盖，结果应该很接近 cudaEvent。
    CUDA_CHECK(cudaDeviceSynchronize());
    t0 = Clock::now();
    for (int i = 0; i < ITERS; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    double m2b = ms_since(t0) / ITERS;

    // ================= 方式 3：cudaEvent =================
    // 3a：一对 event 包住整个循环，再除以 ITERS。
    //     event 分辨率 ~0.5us，除以 200 之后误差被摊薄。
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));              // 插进 stream 0 的时间线
    for (int i = 0; i < ITERS; ++i) launch();
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));         // 必须等 e1 真被 GPU 执行到
    float m3a_total = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&m3a_total, e0, e1));
    double m3a = m3a_total / ITERS;
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));

    // 3b：用上面写好的模板，拿到分布
    BenchResult r = benchmark(launch, /*warmup=*/10, /*iters=*/ITERS);

    // ================= 报告 =================
    auto gbs = [&](double ms) {
        // vector add：读 a、读 b、写 c，共 3 次 n*4 字节
        return 3.0 * bytes / (ms * 1e-3) / 1e9;
    };

    printf("n = %d, grid = %d, block = %d, iters = %d\n\n", n, grid, block, ITERS);
    printf("%-42s %10s %10s\n", "测法", "ms/次", "GB/s");
    printf("--------------------------------------------------------------\n");
    printf("%-42s %10.5f %10s\n", "1  chrono, 单次 launch, 不同步", m1_single, "— 假的");
    printf("%-42s %10.5f %10.1f\n", "1' chrono, 5000 次 launch, 不同步", m1_loop, gbs(m1_loop));
    printf("%-42s %10.5f %10.1f\n", "2a chrono + sync (同步在循环内)", m2a, gbs(m2a));
    printf("%-42s %10.5f %10.1f\n", "2b chrono + sync (同步在循环外)", m2b, gbs(m2b));
    printf("%-42s %10.5f %10.1f\n", "3a cudaEvent (一对包住整个循环)", m3a, gbs(m3a));
    printf("%-42s %10.5f %10.1f\n", "3b benchmark<> median", r.median_ms, gbs(r.median_ms));
    printf("--------------------------------------------------------------\n");
    printf("3b 分布: min %.5f / median %.5f / mean %.5f / max %.5f ms\n",
           r.min_ms, r.median_ms, r.mean_ms, r.max_ms);
    printf("带宽利用率 (按 median): %.1f%% of %.1f GB/s\n\n",
           100.0 * gbs(r.median_ms) / peak_gbs, peak_gbs);

    // ================= 校验（放在所有计时之后）=================
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        if (hc[i] != ha[i] + hb[i]) { printf("MISMATCH at %d\n", i); return 1; }
    }
    printf("Result: PASS (%d elements)\n", n);

    cudaFree(da); cudaFree(db); cudaFree(dc);
    free(ha); free(hb); free(hc);
    return 0;
}
