// roofline_sweep.cu —— 算术强度可调的 FMA kernel，扫描 AI 从 1 到 512
#include "benchmark.h"
#include <cmath>

constexpr float kMul = 0.999f;
constexpr float kAdd = 0.001f;

// 每线程：读 1 float -> K 次 FMA -> 写 1 float
// bytes = 8n, flops = 2Kn, AI = K/4
template <int K>
__global__ void fma_sweep(const float* __restrict__ x,
                          float* __restrict__ y, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;

    float v  = x[i];
    // 4 条互不依赖的链，填满 FFMA 的 ~4 cycle 延迟
    float a0 = v, a1 = v + 1.f, a2 = v + 2.f, a3 = v + 3.f;

    #pragma unroll 16
    for (int k = 0; k < K / 4; ++k) {
        a0 = fmaf(a0, kMul, kAdd);
        a1 = fmaf(a1, kMul, kAdd);
        a2 = fmaf(a2, kMul, kAdd);
        a3 = fmaf(a3, kMul, kAdd);
    }
    y[i] = a0 + a1 + a2 + a3;   // 少了这句 -> 整个循环被当死代码删除
}

// 对照组：单链版本，用来实测 ILP 的影响
template <int K>
__global__ void fma_sweep_1chain(const float* __restrict__ x,
                                 float* __restrict__ y, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = x[i];
    #pragma unroll 64
    for (int k = 0; k < K; ++k) a = fmaf(a, kMul, kAdd);
    y[i] = a;
}

// ---------------------------------------------------------------------------
static FILE* g_csv = nullptr;

template <int K>
void run_one(const float* d_x, float* d_y, size_t n, bool one_chain = false) {
    const int block = 256;
    const int grid  = (int)((n + block - 1) / block);

    auto launch = [&] {
        if (one_chain) fma_sweep_1chain<K><<<grid, block>>>(d_x, d_y, n);
        else           fma_sweep<K>       <<<grid, block>>>(d_x, d_y, n);
    };

    // 大 K 单次就要十几 ms，减少迭代次数，总时长控制在可接受范围
    int iters = (K <= 128) ? 200 : (K <= 512 ? 100 : 50);

    BenchResult r = benchmark(launch, 2 * n * sizeof(float),
                              2ull * K * n, 10, iters);

    double ai = K / 4.0;
    char name[64];
    snprintf(name, sizeof(name), "%sK=%-5d AI=%.0f",
             one_chain ? "[1ch] " : "", K, ai);
    print_result(name, r);

    // if (g_csv && !one_chain)
    //     fprintf(g_csv, "%d,%.4f,%zu,%zu,%.6f,%.6f,%.6f,%.3f,%.3f\n",
    //             K, ai, r.bytes, r.flops, r.median_ms, r.min_ms, r.max_ms,
    //             r.gbps(), r.gflops());
        if (g_csv)
        fprintf(g_csv, "%d,%.4f,%d,%zu,%zu,%.6f,%.6f,%.6f,%.3f,%.3f\n",
                K, ai, one_chain ? 1 : 0, r.bytes, r.flops,
                r.median_ms, r.min_ms, r.max_ms, r.gbps(), r.gflops());
}

// CPU 参考：复现同一条链，验证 kernel 没被优化掉
static float ref_chain(float v, int K) {
    float a0 = v, a1 = v + 1.f, a2 = v + 2.f, a3 = v + 3.f;
    for (int k = 0; k < K / 4; ++k) {
        a0 = fmaf(a0, kMul, kAdd); a1 = fmaf(a1, kMul, kAdd);
        a2 = fmaf(a2, kMul, kAdd); a3 = fmaf(a3, kMul, kAdd);
    }
    return a0 + a1 + a2 + a3;
}

int main() {
    print_device_info();

    const size_t n = 1ull << 26;          // 67,108,864 -> 每个数组 256 MiB
    const size_t nbytes = n * sizeof(float);
    printf("n = %zu, 工作集 = %.0f MiB (L2 的 %.0f 倍)\n\n",
           n, 2.0 * nbytes / 1048576.0, 2.0 * nbytes / (32.0 * 1048576.0));

    std::vector<float> h_x(n);
    for (size_t i = 0; i < n; ++i) h_x[i] = 0.5f + (i & 15) * 0.01f;

    float *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, nbytes));
    CUDA_CHECK(cudaMalloc(&d_y, nbytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), nbytes, cudaMemcpyHostToDevice));

    // ---- 正确性自检（必须在计时之前）----
    {
        fma_sweep<64><<<(int)(n / 256), 256>>>(d_x, d_y, n);
        CUDA_CHECK_LAUNCH();
        std::vector<float> h_y(16);
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, 16 * sizeof(float),
                              cudaMemcpyDeviceToHost));
        for (int i = 0; i < 16; ++i) {
            float want = ref_chain(h_x[i], 64);
            if (fabsf(h_y[i] - want) > 1e-3f * fabsf(want)) {
                fprintf(stderr, "[FAIL] i=%d got %.6f want %.6f\n",
                        i, h_y[i], want);
                return 1;
            }
        }
        printf("正确性自检通过（K=64）\n\n");
    }

    g_csv = fopen("roofline.csv", "w");
    // fprintf(g_csv, "K,AI,bytes,flops,median_ms,min_ms,max_ms,gbps,gflops\n");
    fprintf(g_csv, "K,AI,one_chain,bytes,flops,median_ms,min_ms,max_ms,gbps,gflops\n");
    print_result_header();
    run_one<4>   (d_x, d_y, n);
    run_one<8>   (d_x, d_y, n);
    run_one<16>  (d_x, d_y, n);
    run_one<32>  (d_x, d_y, n);
    run_one<64>  (d_x, d_y, n);
    run_one<128> (d_x, d_y, n);
    run_one<192> (d_x, d_y, n);   // 拐点附近加密采样
    run_one<256> (d_x, d_y, n);
    run_one<320> (d_x, d_y, n);
    run_one<512> (d_x, d_y, n);
    run_one<1024>(d_x, d_y, n);
    run_one<2048>(d_x, d_y, n);

    printf("\n--- ILP 对照：单依赖链 ---\n");
    run_one<512> (d_x, d_y, n, true);
    run_one<2048>(d_x, d_y, n, true);

    fclose(g_csv);
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    printf("\n-> roofline.csv 已写出\n");
    return 0;
}