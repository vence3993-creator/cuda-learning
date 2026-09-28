// ============================================================================
// week03/day01/reduce_basic.cu —— W3 D1：reduce 前三版 + v3（D2 提前）
//
//   v0 atomic       每个线程 atomicAdd 到同一地址（基线：感受争抢）
//   v1 interleaved  交错寻址 tid % (2s) —— warp 分化
//   v2 strided      idx = 2*s*tid     —— 去分化，但引入 bank conflict
//   v3 sequential   s 从 blockDim/2 减半 —— 线程连续 + 地址连续（D2 内容，提前完成）
//
// 两级归约：block 内在 shared 里归约出部分和，thread 0 atomicAdd 到结果（每 block 一次）
//
// 编译：make
// 运行：./reduce_basic          正确性 + 计时
//       ./reduce_basic --once   每个 kernel 只跑一次，只做正确性（给 ncu 用）
// ============================================================================
#include "benchmark.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

constexpr int BLOCK = 256;
static_assert((BLOCK & (BLOCK - 1)) == 0, "BLOCK 必须是 2 的幂，否则 v3 的对折会漏元素");

// // ---------------------------------------------------------------------------
// // v0：全体线程争抢同一个地址
// // ---------------------------------------------------------------------------
// __global__ void reduce_v0_atomic(const float* __restrict__ in, float* out, int n) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < n) atomicAdd(out, in[i]);
// }

// // ---------------------------------------------------------------------------
// // v1：交错寻址。活跃线程 0, 2s, 4s… 散落在每个 warp 里 → 分化
// // ---------------------------------------------------------------------------
// __global__ void reduce_v1_interleaved(const float* __restrict__ in, float* out, int n) {
//     extern __shared__ float sdata[];
//     unsigned int tid = threadIdx.x;
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     sdata[tid] = (i < n) ? in[i] : 0.f;
//     __syncthreads();

//     for (unsigned int s = 1; s < blockDim.x; s *= 2) {
//         if (tid % (2 * s) == 0)
//             sdata[tid] += sdata[tid + s];
//         __syncthreads();
//     }
//     if (tid == 0) atomicAdd(out, sdata[0]);
// }

// // ---------------------------------------------------------------------------
// // v2：线程连续，访问跨度 2s。warp 要么全干要么全歇，但 bank conflict 随 s 翻倍
// // ---------------------------------------------------------------------------
// __global__ void reduce_v2_strided(const float* __restrict__ in, float* out, int n) {
//     extern __shared__ float sdata[];
//     unsigned int tid = threadIdx.x;
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     sdata[tid] = (i < n) ? in[i] : 0.f;
//     __syncthreads();

//     for (unsigned int s = 1; s < blockDim.x; s *= 2) {
//         unsigned int idx = 2 * s * tid;
//         if (idx < blockDim.x)
//             sdata[idx] += sdata[idx + s];
//         __syncthreads();
//     }
//     if (tid == 0) atomicAdd(out, sdata[0]);
// }

// ---------------------------------------------------------------------------
// v3：对折。写前半、读后半，永不相撞；32 个连续线程访问 32 个连续地址 → 无冲突
// ---------------------------------------------------------------------------
__global__ void reduce_v3_sequential(const float* __restrict__ in, float* out, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? in[i] : 0.f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

__global__ void reduce_v4_a(const float* __restrict__ in, float* out, int n){
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    float a = (i < n) ? in[i] : 0.f;
    float b = (i + blockDim.x < n)? in[i+blockDim.x] : 0.f;
    sdata[tid] = a + b;
    __syncthreads();

    for(unsigned int s = blockDim.x / 2; s > 0; s>>=1){
        if(tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads(); 
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}
// v4b grid-stride loop：每线程先在寄存器里累加 for (i = gid; i < n; i += blockDim.x * gridDim.x) sum += in[i];
// 再写 shared 走 v3 的树形归约，最后 thread 0 atomicAdd
// ⚠️ 坑：保留 i < n 的边界判断；atomicAdd 版每次运行前照旧 cudaMemset 输出
__global__ void reduce_v4_b(const float* __restrict__ in, float* out, int n){
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;

    float sum = 0.f;
    for(int i = blockIdx.x * blockDim.x + tid; i<n; i+= blockDim.x * gridDim.x){
        sum += in[i];
    }
    sdata[tid] = sum;
    __syncthreads();
    for(unsigned int s = blockDim.x/2; s > 0; s>>=1){
        if(tid < s){
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }
    if(tid == 0) atomicAdd(out, sdata[0]);
}


static int v4b_full_grid(int* num_sm_out, int* nb_out) {
    int nb = 0, num_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &nb, reduce_v4_b, BLOCK, BLOCK * sizeof(float)));
    CUDA_CHECK(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));
    if (num_sm_out) *num_sm_out = num_sm;
    if (nb_out) *nb_out = nb;
    return num_sm * nb;   // 一波满载
}

static void scan_v4b(const float* d_in, float* d_out, int n,
                     double ref, double abs_sum, size_t bytes, double peak) {
    int num_sm = 0, nb = 0;
    const int full = v4b_full_grid(&num_sm, &nb);
    printf("\n[v4b] 每 SM 常驻 block = %d，满载 gridDim = %d x %d = %d\n",
           nb, num_sm, nb, full);

    // full 的 1/6 ... 16 倍，外加两个非整倍数点看尾波
    const int grids[] = {full / 6, full / 3, full / 2, full, full + 1,
                         2 * full, 2 * full + 1, 4 * full, 8 * full, 16 * full};

    FILE* csv = fopen("v4b_scan.csv", "w");
    if (csv) fprintf(csv, "gridDim,blocks_per_sm,median_ms,gbps,pct_peak\n");

    printf("%10s %10s %12s %10s %8s\n", "gridDim", "blk/SM", "median/ms", "GB/s", "%peak");
    for (int g : grids) {
        if (g <= 0) continue;

        // ① 正确性：每个 grid 单独验一次
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        reduce_v4_b<<<g, BLOCK, BLOCK * sizeof(float)>>>(d_in, d_out, n);
        CUDA_CHECK_LAUNCH();
        float got = 0.f;
        CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        double rel = std::fabs(double(got) - ref) / abs_sum;
        if (rel >= 1e-6) {
            printf("%10d  FAIL  rel_err=%.2e，跳过计时\n", g, rel);
            continue;
        }

        // ② 计时
        auto body = [&] { reduce_v4_b<<<g, BLOCK, BLOCK * sizeof(float)>>>(d_in, d_out, n); };
        BenchResult r = benchmark(body, bytes, 0, 10, 200);
        double pct = peak > 0 ? 100.0 * r.gbps() / peak : 0.0;
        printf("%10d %10.2f %12.5f %10.2f %7.1f%%\n",
               g, double(g) / num_sm, r.median_ms, r.gbps(), pct);
        if (csv) fprintf(csv, "%d,%.3f,%.5f,%.2f,%.2f\n",
                         g, double(g) / num_sm, r.median_ms, r.gbps(), pct);
    }
    if (csv) fclose(csv);
}
// ---------------------------------------------------------------------------
// host：四版共用一套启动 / 验证 / 计时
// ---------------------------------------------------------------------------
using ReduceFn = void (*)(const float*, float*, int);

struct KernelEntry {
    const char* name;
    ReduceFn    fn;
    int         warmup;
    int         iters;   // v0 慢得离谱，少跑几次
    int         elems_per_thread; //v3 = 1m v4a = 2;
};

static void launch(const KernelEntry& k, const float* d_in, float* d_out, int n) {
    if (k.elems_per_thread <= 0) {
        fprintf(stderr, "%s: elems_per_thread 未设置\n", k.name);
        std::exit(1);
    }
    int per_block = BLOCK * k.elems_per_thread;
    int grid = (n + per_block - 1) / per_block;
    k.fn<<<grid, BLOCK, BLOCK * sizeof(float)>>>(d_in, d_out, n);
}

int main(int argc, char** argv) {
    const bool once = (argc > 1 && std::strcmp(argv[1], "--once") == 0);
    constexpr int N = 1 << 26;   // 256 MB > L2

    if (!once) print_device_info();

    // ---- 输入：[-1, 1) 随机 float；CPU 用 double 累加作参考 ----
    std::vector<float> h_in(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    double ref = 0.0, abs_sum = 0.0;
    for (int i = 0; i < N; ++i) {
        h_in[i]  = dist(rng);
        ref     += h_in[i];
        abs_sum += std::fabs(h_in[i]);
    }

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  size_t(N) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), size_t(N) * sizeof(float),
                          cudaMemcpyHostToDevice));

    const KernelEntry kernels[] = {
        // {"v0_atomic",      reduce_v0_atomic,      2,  10},
        // {"v1_interleaved", reduce_v1_interleaved, 10, 200},
        // {"v2_strided",     reduce_v2_strided,     10, 200},
        {"v3_sequential",  reduce_v3_sequential,  10, 200, 1},
        {"v4a",  reduce_v4_a,  10, 200, 2},
    };
    constexpr int K = sizeof(kernels) / sizeof(kernels[0]);

    // ---- ① 正确性：每个 kernel 单独跑一次，跑前清零（atomicAdd 会累加）----
    // 用相对误差不用 ==：float 加法不满足结合律，归约顺序一变末几位就变；
    // atomicAdd 顺序不确定，每次运行结果都可能差一点 —— 正常现象
    bool all_ok = true;
    for (const auto& k : kernels) {
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        launch(k, d_in, d_out, N);
        CUDA_CHECK_LAUNCH();
        float got = 0.f;
        CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        double rel = std::fabs(double(got) - ref) / abs_sum;
        bool ok = rel < 1e-6;
        printf("[verify] %-15s %s  gpu=%12.4f  ref=%12.4f  rel_err=%.2e\n",
               k.name, ok ? "PASS" : "FAIL", got, ref, rel);
        all_ok = all_ok && ok;
    }

    if (once && all_ok) {
        int g = v4b_full_grid(nullptr, nullptr);
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        reduce_v4_b<<<g, BLOCK, BLOCK * sizeof(float)>>>(d_in, d_out, N);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (once || !all_ok) {
        if (!all_ok) fprintf(stderr, "\n有 kernel 算错，先修正确性，不计时。\n");
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        return all_ok ? 0 : 1;
    }

    // ---- ② 计时 ----
    // 多次迭代后 atomicAdd 一直在累加，d_out 早就不是正确值 —— 无所谓，正确性上面已验
    const size_t bytes = size_t(N) * sizeof(float);   // 有效带宽 = n × 4 B ÷ time
    const double peak  = query_peak_bandwidth_gbs();
    if (peak > 0)
        printf("\n理论下限: %.1f MB / %.0f GB/s = %.3f ms\n\n",
               bytes / 1e6, peak, bytes / (peak * 1e9) * 1e3);

    print_result_header();
    BenchResult res[K];
    for (int j = 0; j < K; ++j) {
        const KernelEntry& k = kernels[j];
        auto body = [&] { launch(k, d_in, d_out, N); };
        res[j] = benchmark(body, bytes, 0, k.warmup, k.iters);
        print_result(k.name, res[j]);
        if (peak > 0)
            printf("%-22s -> %.1f%% of peak bandwidth\n", "", 100.0 * res[j].gbps() / peak);
    }

    scan_v4b(d_in, d_out, N, ref, abs_sum, bytes, peak);

    // ---- ③ 对照计划里的三个预测题 ----
    // printf("\n[预测题]\n");
    // printf("  v0 比 v2 慢   : %.1fx\n",  res[0].median_ms / res[2].median_ms);
    // printf("  v1 -> v2 加速 : %.2fx   (slides 在 G80 上是 2.33x)\n",
    //        res[1].median_ms / res[2].median_ms);
    // if (peak > 0)
    //     printf("  v2 带宽利用率 : %.1f%%\n", 100.0 * res[2].gbps() / peak);
    // printf("  v2 -> v3 加速 : %.2fx   (D2 内容)\n", res[2].median_ms / res[3].median_ms);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
