// ============================================================================
// week01_basics/day06/transpose.cu
//
// 目的：不是优化转置，是用数据说清楚 naive 转置慢在哪。
//
// 三个 kernel 搬运的字节数完全一样（读 N² 个 float、写 N² 个 float），
// 唯一的差别是 global memory 的地址模式。所以它们之间的性能差距，
// 100% 来自 coalescing，没有任何其他变量。
//
// 用法：
//     ./transpose            bench 模式，跑完整计时表格
//     ./transpose profile    每个 kernel 只启动一次，专门喂给 ncu
//     ./transpose bench 2048 指定矩阵边长
// ============================================================================
#include "benchmark.h"

#include <vector>
#include <cstring>
#include <cstdio>

// ============================================================================
// kernels
//
// 故意不写 __restrict__：它会让编译器把 load 换成 LDG.NC（只读数据缓存路径），
// 引入一个和 coalescing 无关的变量。今天要的是干净的对照实验。
// ============================================================================

// ---- baseline：读、写都 coalesced ----
// 这是转置的性能天花板。同样搬 2×N²×4 字节，但零 stride 惩罚。
// 没有它，你手上的转置带宽数字没有参照物。
__global__ void copyKernel(const float* in, float* out, int n)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // n 能被 blockDim 整除时这个判断恒为真，但把 n 改成 4095 立刻就要它。
    // 不要因为"这次用不上"就删掉。
    if (x < n && y < n) {
        size_t idx = (size_t)y * n + x;
        out[idx] = in[idx];
    }
}

// ---- 变体 ①：读连续、写跨行 ----
// warp 内 x 变 y 不变：
//   读 in[y*n + x]  -> x 在低位 -> 地址步长 4B   -> 4 sectors
//   写 out[x*n + y] -> x 在高位 -> 地址步长 n*4B -> 32 sectors
__global__ void transposeReadCoalesced(const float* in, float* out, int n)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n)
        out[(size_t)x * n + y] = in[(size_t)y * n + x];
}

// ---- 变体 ②：读跨行、写连续 ----
// 和 ① 是同一个数学运算，只是把惩罚从写端挪到了读端。
// 两者的性能差值 = 硬件对 load miss 和 store miss 的救济能力之差。
__global__ void transposeWriteCoalesced(const float* in, float* out, int n)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n)
        out[(size_t)y * n + x] = in[(size_t)x * n + y];
}

// ============================================================================
// CPU 参考实现与校验
//
// 别跳过这步：索引写反的 transpose 照样能跑出漂亮的带宽数字，
// 因为它搬运的字节数一模一样。带宽正确 ≠ 结果正确。
// ============================================================================
static void cpuTranspose(const std::vector<float>& in,
                         std::vector<float>&       out, int n)
{
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x)
            out[(size_t)x * n + y] = in[(size_t)y * n + x];
}

static bool verify(const float* d_out, const std::vector<float>& ref,
                   std::vector<float>& h_tmp, const char* name)
{
    CUDA_CHECK(cudaMemcpy(h_tmp.data(), d_out,
                          ref.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < ref.size(); ++i) {
        if (h_tmp[i] != ref[i]) {          // 纯搬运，没有浮点运算，可以精确比较
            printf("  [FAIL] %s: idx=%zu got=%.1f want=%.1f\n",
                   name, i, h_tmp[i], ref[i]);
            return false;
        }
    }
    return true;
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv)
{
    const char* mode = (argc > 1) ? argv[1] : "bench";
    const bool  profileMode = (std::strcmp(mode, "profile") == 0);
    const int   n = (argc > 2) ? std::atoi(argv[2]) : 4096;

    // 4096×4096 float = 64 MB。必须远大于 L2，否则测的是 L2 不是 DRAM。
    // print_device_info 会把 L2 大小打出来，自己对一眼。
    const size_t elems = (size_t)n * n;
    const size_t matBytes = elems * sizeof(float);
    const size_t movedBytes = 2 * matBytes;   // 读一遍 + 写一遍

    print_device_info();
    printf("矩阵             : %d x %d float = %.1f MB / 单个矩阵\n",
           n, n, matBytes / 1048576.0);
    printf("有效字节数       : %.1f MB / 次调用 (读 %.0f MB + 写 %.0f MB)\n\n",
           movedBytes / 1048576.0,
           matBytes / 1048576.0, matBytes / 1048576.0);

    // ---- host 数据 ----
    // 填成递增序列：0..N²-1 在 float 里都能精确表示（24 位尾数，上限 16777216），
    // n=4096 时 N² 正好 16777216，刚好卡在边界内。每个元素唯一 -> 索引错位一定被抓到。
    std::vector<float> h_in(elems), h_ref(elems), h_tmp(elems);
    for (size_t i = 0; i < elems; ++i) h_in[i] = (float)i;
    cpuTranspose(h_in, h_ref, n);

    // ---- device 数据 ----
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  matBytes));
    CUDA_CHECK(cudaMalloc(&d_out, matBytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), matBytes, cudaMemcpyHostToDevice));

    // ---- profile 模式：每个 kernel 只跑一次 ----
    // ncu 会对每次 launch 做 replay 采集指标。如果直接 profile bench 模式
    // （3 kernel × 4 形状 × 210 次 launch），你会等到天荒地老。
    if (profileMode) {
        int bx = (argc > 3) ? std::atoi(argv[3]) : 32;
        int by = (argc > 4) ? std::atoi(argv[4]) : 8;  
        dim3 block(bx, by);
        dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

        copyKernel           <<<grid, block>>>(d_in, d_out, n);
        transposeReadCoalesced <<<grid, block>>>(d_in, d_out, n);
        transposeWriteCoalesced<<<grid, block>>>(d_in, d_out, n);

        CUDA_CHECK_LAUNCH();
        printf("profile 模式：3 个 kernel 各启动一次，交给 ncu 采集。\n");
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        return 0;
    }

    // ---- bench 模式 ----
    const double peakBW = query_peak_bandwidth_gbs();

    struct Cfg { int bx, by; const char* note; };
    const Cfg cfgs[] = {
        {32, 32, "warp = 一整行 32 线程，主力配置"},
        {32,  8, "warp 仍是一整行，但每 block 只有 8 行"},
        {16, 16, "warp = 两行各 16 -> 横跨两行，注意看变化"},
        { 8,  8, "warp = 四行各 8 -> 碎得更厉害"},
        {128, 2, "by=2 -> 跨行侧每行仅 8 B，理论 4 倍浪费"},
        { 64, 4, "by=4 -> 跨行侧每行仅 16 B，理论 2 倍浪费"},
    };

    for (const Cfg& c : cfgs) {
        dim3 block(c.bx, c.by);
        dim3 grid((n + c.bx - 1) / c.bx, (n + c.by - 1) / c.by);

        printf("block = (%d, %d)   grid = (%d, %d)   %d threads/block\n",
               c.bx, c.by, grid.x, grid.y, c.bx * c.by);
        printf("  %s\n", c.note);
        print_result_header();

        double copyGbps = 0.0;

        // --- copy baseline ---
        {
            auto launch = [&]{ copyKernel<<<grid, block>>>(d_in, d_out, n); };
            launch();
            CUDA_CHECK_LAUNCH();
            if (!verify(d_out, h_in, h_tmp, "copy")) return 1;

            BenchResult r = benchmark(launch, movedBytes);
            print_result("copy (baseline)", r);
            copyGbps = r.gbps();
        }

        // --- transpose ①：读连续、写跨行 ---
        double g1 = 0.0;
        {
            auto launch = [&]{ transposeReadCoalesced<<<grid, block>>>(d_in, d_out, n); };
            launch();
            CUDA_CHECK_LAUNCH();
            if (!verify(d_out, h_ref, h_tmp, "transpose_read_coalesced")) return 1;

            BenchResult r = benchmark(launch, movedBytes);
            print_result("t_write_strided", r);
            g1 = r.gbps();
        }

        // --- transpose ②：读跨行、写连续 ---
        double g2 = 0.0;
        {
            auto launch = [&]{ transposeWriteCoalesced<<<grid, block>>>(d_in, d_out, n); };
            launch();
            CUDA_CHECK_LAUNCH();
            if (!verify(d_out, h_ref, h_tmp, "transpose_write_coalesced")) return 1;

            BenchResult r = benchmark(launch, movedBytes);
            print_result("t_read_strided", r);
            g2 = r.gbps();
        }

        printf("  -> 相对 copy : rd_coal %.1f%%   wr_coal %.1f%%\n",
               100.0 * g1 / copyGbps, 100.0 * g2 / copyGbps);
        if (peakBW > 0)
            printf("  -> 相对峰值 : copy %.1f%%   rd_coal %.1f%%   wr_coal %.1f%%\n",
                   100.0 * copyGbps / peakBW,
                   100.0 * g1 / peakBW, 100.0 * g2 / peakBW);
        printf("\n");
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
