// bw_sweep.cu —— 扫描工作集大小，区分 L2 带宽和显存带宽
// nvcc -O3 -arch=native bw_sweep.cu -o bw_sweep

#include <cstdio>
#include <cstdlib>
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

// ---------------------------------------------------------------- kernels
__global__ void vecAdd(const float* __restrict__ a, const float* __restrict__ b,
                       float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// 在设备上初始化，避免为 512MB/数组 分配同样大的 host 内存
__global__ void initKernel(float* a, float* b, int n) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        a[i] = (float)(i & 1023);
        b[i] = (float)((i & 1023) * 2);
    }
}

// 校验也放设备上做，只把错误计数拷回来
__global__ void checkKernel(const float* a, const float* b, const float* c,
                            int n, int* nerr) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        if (c[i] != a[i] + b[i]) atomicAdd(nerr, 1);
    }
}

// ---------------------------------------------------------------- benchmark
struct BenchResult {
    float mean_ms, median_ms, min_ms, max_ms;
    int   iters;
};

template <class Body>
BenchResult benchmark(Body&& body, int warmup, int iters, cudaStream_t stream = 0) {
    for (int i = 0; i < warmup; ++i) body();
    CUDA_CHECK(cudaDeviceSynchronize());

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
    CUDA_CHECK(cudaEventSynchronize(ev_b[iters - 1]));

    std::vector<float> t(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventElapsedTime(&t[i], ev_a[i], ev_b[i]));
        cudaEventDestroy(ev_a[i]);
        cudaEventDestroy(ev_b[i]);
    }
    std::vector<float> s = t;
    std::sort(s.begin(), s.end());

    BenchResult r;
    r.iters     = iters;
    r.min_ms    = s.front();
    r.max_ms    = s.back();
    r.median_ms = s[iters / 2];
    r.mean_ms   = std::accumulate(t.begin(), t.end(), 0.0f) / iters;
    return r;
}

// ---------------------------------------------------------------- 单个尺寸
struct SizeResult {
    int    shift;
    double ws_mib;      // 工作集 = 3 * n * 4 字节
    double ms;
    double gbs;
    bool   ok;
};

static SizeResult runOne(int shift, bool verify) {
    SizeResult sr{shift, 0, 0, 0, false};

    const int    n     = 1 << shift;
    const size_t bytes = (size_t)n * sizeof(float);
    sr.ws_mib = 3.0 * bytes / (1024.0 * 1024.0);

    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    if (3 * bytes + (256u << 20) > freeB) return sr;   // 留 256MB 余量，装不下就跳过

    float *da, *db, *dc;
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMalloc(&dc, bytes));

    const int block = 256;
    const int grid  = (int)std::min<long long>((n + block - 1LL) / block, 1LL << 20);

    initKernel<<<1024, block>>>(da, db, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto launch = [&] { vecAdd<<<(n + block - 1) / block, block>>>(da, db, dc, n); };
    (void)grid;

    // 预跑一次估时长，据此决定 iters：目标总时长 ~100ms，且钳在 [10, 200]
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    for (int i = 0; i < 5; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < 5; ++i) launch();
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float pilot_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&pilot_ms, e0, e1));
    pilot_ms /= 5.f;
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));

    int iters = (pilot_ms > 0.f) ? (int)(100.0 / pilot_ms) : 200;
    iters = std::max(10, std::min(200, iters));

    BenchResult r = benchmark(launch, /*warmup=*/10, iters);
    CUDA_CHECK(cudaGetLastError());

    sr.ms  = r.median_ms;
    sr.gbs = 3.0 * bytes / (r.median_ms * 1e-3) / 1e9;
    sr.ok  = true;

    if (verify) {
        int *d_nerr, h_nerr = 0;
        CUDA_CHECK(cudaMalloc(&d_nerr, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_nerr, 0, sizeof(int)));
        checkKernel<<<1024, block>>>(da, db, dc, n, d_nerr);
        CUDA_CHECK(cudaMemcpy(&h_nerr, d_nerr, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_nerr));
        printf("校验 2^%d: %s (%d 个错误)\n\n", shift,
               h_nerr == 0 ? "PASS" : "FAIL", h_nerr);
    }

    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
    return sr;
}

// ---------------------------------------------------------------- main
int main(int argc, char** argv) {
    int lo = (argc > 1) ? atoi(argv[1]) : 18;
    int hi = (argc > 2) ? atoi(argv[2]) : 27;

    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));

    int mem_clk_khz = 0, bus_bits = 0;
    cudaDeviceGetAttribute(&mem_clk_khz, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0);
    double peak_gbs = 2.0 * mem_clk_khz * 1e3 * (bus_bits / 8.0) / 1e9;
    bool peak_known = (mem_clk_khz > 0 && bus_bits > 0);

    double l2_mib = p.l2CacheSize / (1024.0 * 1024.0);

    printf("GPU: %s (sm_%d%d), SM=%d\n", p.name, p.major, p.minor, p.multiProcessorCount);
    printf("L2 = %.1f MiB, 显存 = %.1f GiB\n", l2_mib,
           p.totalGlobalMem / 1073741824.0);
    if (peak_known) printf("理论显存带宽峰值: %.1f GB/s\n\n", peak_gbs);
    else            printf("理论峰值: 驱动未报告(该属性在新架构上可能返回0)，请查规格书\n\n");

    // 最小尺寸顺带做一次正确性校验
    std::vector<SizeResult> res;
    for (int s = lo; s <= hi; ++s) res.push_back(runOne(s, /*verify=*/s == lo));

    printf("%-6s %10s %10s %12s %12s %8s\n",
           "N", "工作集MiB", "工作集/L2", "median ms", "GB/s", "%峰值");
    printf("---------------------------------------------------------------------\n");
    for (auto& r : res) {
        if (!r.ok) { printf("2^%-4d %10.1f  显存不足，跳过\n", r.shift, r.ws_mib); continue; }
        printf("2^%-4d %10.1f %10.2f %12.5f %12.1f",
               r.shift, r.ws_mib, r.ws_mib / l2_mib, r.ms, r.gbs);
        if (peak_known) printf(" %7.1f%%", 100.0 * r.gbs / peak_gbs);
        printf("\n");
    }
    printf("---------------------------------------------------------------------\n");
    printf("读法: 工作集/L2 < 1 的行是 L2 带宽(会虚高甚至 >100%%峰值);\n");
    printf("      工作集/L2 远大于 1 之后曲线会压平，那个平台值才是真实显存带宽。\n");
    return 0;
}
