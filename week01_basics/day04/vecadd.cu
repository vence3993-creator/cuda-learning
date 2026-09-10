// ============================================================================
// day04/vecadd.cu —— 验证 benchmark.h，并扫出 N / block size 两条曲线
// 编译：nvcc -O2 -I../common vecadd.cu -o vecadd
// ============================================================================
#include "benchmark.h"

__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// vector add：读 a、读 b、写 c —— 每元素 3 次 4 字节访问
static inline size_t vecadd_bytes(int n) { return 3ull * n * sizeof(float); }
// 每元素 1 次浮点加法
static inline size_t vecadd_flops(int n) { return 1ull * n; }

int main() {
    print_device_info();

    double peak = query_peak_bandwidth_gbs();

    // 最大规模：2^27 个 float = 512 MB/数组，三个数组 1.5 GB。
    // 显存小于 4 GB 的卡把 MAX_SHIFT 调到 26。
    const int MAX_SHIFT = 27;
    const int max_n     = 1 << MAX_SHIFT;

    // 一次性按最大规模分配，扫描时只改 n，避免反复 malloc 干扰测量
    float *da, *db, *dc;
    CUDA_CHECK(cudaMalloc(&da, (size_t)max_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, (size_t)max_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dc, (size_t)max_n * sizeof(float)));

    float* h = (float*)malloc((size_t)max_n * sizeof(float));
    for (int i = 0; i < max_n; ++i) h[i] = (float)(i % 1000) * 0.5f;
    CUDA_CHECK(cudaMemcpy(da, h, (size_t)max_n * sizeof(float), cudaMemcpyHostToDevice));
    for (int i = 0; i < max_n; ++i) h[i] = (float)(i % 997) * 1.5f;
    CUDA_CHECK(cudaMemcpy(db, h, (size_t)max_n * sizeof(float), cudaMemcpyHostToDevice));

    // ------------------------------------------------------------------
    // 扫描 1：规模 N —— 找出 L2 失效、数字开始可信的拐点
    // ------------------------------------------------------------------
    printf("=== 扫描 N（block = 256）===\n");
    printf("%6s %12s %12s %10s %8s\n",
           "N", "工作集/MB", "median/ms", "GB/s", "%peak");
    printf("---------------------------------------------------------\n");

    const int block = 256;
    for (int shift = 18; shift <= MAX_SHIFT; ++shift) {
        int    n     = 1 << shift;
        size_t bytes = vecadd_bytes(n);
        int    grid  = (n + block - 1) / block;

        auto launch = [&] { vecAdd<<<grid, block>>>(da, db, dc, n); };

        // 大规模时迭代次数少一点，否则太慢
        int iters = (shift >= 25) ? 50 : 200;
        BenchResult r = benchmark(launch, bytes, vecadd_flops(n), 10, iters);

        printf("2^%-4d %12.1f %12.5f %10.2f %7.1f%%\n",
               shift, bytes / 1048576.0, r.median_ms, r.gbps(),
               peak > 0 ? 100.0 * r.gbps() / peak : 0.0);
    }

    // ------------------------------------------------------------------
    // 扫描 2：block size —— 在"数字可信"的规模上做
    // ------------------------------------------------------------------
    const int n_big = 1 << 26;   // 256 MB/数组，远超任何消费卡的 L2
    printf("\n=== 扫描 block size（N = 2^26, 工作集 %.0f MB）===\n",
           vecadd_bytes(n_big) / 1048576.0);
    print_result_header();

    for (int b : {64, 128, 256, 512, 1024}) {
        int  grid   = (n_big + b - 1) / b;
        auto launch = [&] { vecAdd<<<grid, b>>>(da, db, dc, n_big); };

        BenchResult r = benchmark(launch, vecadd_bytes(n_big),
                                  vecadd_flops(n_big), 10, 50);
        char name[32];
        snprintf(name, sizeof(name), "vecadd_block%d", b);
        print_result(name, r);
    }

    // ------------------------------------------------------------------
    // 正确性校验：放在所有计时之后，绝不放进计时循环
    // "跑得快但算错了" 是最容易发生的事
    // ------------------------------------------------------------------
    printf("\n=== 校验 ===\n");
    const int n_chk = 1 << 20;
    {
        int  grid   = (n_chk + block - 1) / block;
        vecAdd<<<grid, block>>>(da, db, dc, n_chk);
        CUDA_CHECK_LAUNCH();

        std::vector<float> ha(n_chk), hb(n_chk), hc(n_chk);
        size_t nb = (size_t)n_chk * sizeof(float);
        CUDA_CHECK(cudaMemcpy(ha.data(), da, nb, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hb.data(), db, nb, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hc.data(), dc, nb, cudaMemcpyDeviceToHost));

        for (int i = 0; i < n_chk; ++i) {
            if (hc[i] != ha[i] + hb[i]) {
                printf("MISMATCH at %d: %f != %f + %f\n", i, hc[i], ha[i], hb[i]);
                return 1;
            }
        }
        printf("PASS (%d elements)\n", n_chk);
    }

    cudaFree(da); cudaFree(db); cudaFree(dc);
    free(h);
    return 0;
}
