// ============================================================================
// week02/day02/transpose_smem.cu
//
// 目的：用 shared memory 把 naive 转置"不合并的那一端"搬到片上重排，
//       让 global 读、写两端的 sectors/request 都回到 4。
//
// 四个 kernel，搬运的字节数完全一样（读 R*C 个 float、写 R*C 个 float）：
//
//   copy               读写都合并，不经 shared                  -> 天花板
//   copy_smem          读写都合并，经 shared 中转，按行读 tile   -> 隔离"中转本身"的开销
//   transpose_naive    读合并、写跨行（= W1D6 的 transposeReadCoalesced）
//   transpose_smem_v2  读写都合并，经 shared 转置，按列读 tile
//
// 两组对照，各只差一个变量：
//   copy_smem  vs copy       -> shared 中转 + __syncthreads 的代价
//   smem_v2    vs copy_smem  -> global 访存模式完全相同，唯一差别是 tile 按列读
//                               这一组的差距就是今天 ncu 要解释的东西
//
// 相对 W1D6 的改动：kernel 支持 rows != cols。方阵测不出 rows/cols 写反的 bug。
//
// 用法：
//     ./transpose_smem              bench 模式，默认 4096
//     ./transpose_smem bench 2048   指定边长
//     ./transpose_smem profile      每个 kernel 各启动一次，交给 ncu
// ============================================================================
#include "benchmark.h"

#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>

// 今天固定 32×32 block：一个线程搬一个元素，tile 边长 = block 边长。
// 1024 threads/block —— 跑完去 ncu 看 Theoretical Occupancy，
// 对照 sm_120 每 SM 最大线程数想想一个 SM 能放几个 block。（Day 4 的伏笔）
constexpr int TILE = 32;

// 约定：in 是 rows × cols 行主序；转置后 out 是 cols × rows 行主序。
// grid 永远按"输入矩阵"切：grid.x 覆盖 cols，grid.y 覆盖 rows。
// 和 W1D6 一样故意不写 __restrict__，保持对照干净。

// ============================================================================
// kernels
// ============================================================================

// ---- copy：天花板 ----
__global__ void copyKernel(const float* in, float* out, int rows, int cols)
{
    int x = blockIdx.x * TILE + threadIdx.x;   // 列
    int y = blockIdx.y * TILE + threadIdx.y;   // 行
    if (x < cols && y < rows) {
        size_t idx = (size_t)y * cols + x;
        out[idx] = in[idx];
    }
}

// ---- copy_smem：经 shared 中转的拷贝 ----
// 写进 tile[ty][tx]，读出也是 tile[ty][tx]（按行），没有任何重排。
// 它和 copy 的差距只来自：多一次 shared 写 + 一次 shared 读 + 一次 __syncthreads。
__global__ void copySmem(const float* in, float* out, int rows, int cols)
{
    __shared__ float tile[TILE][TILE];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (x < cols && y < rows)
        tile[threadIdx.y][threadIdx.x] = in[(size_t)y * cols + x];

    // 放在 if 外面：__syncthreads 必须被 block 内所有线程执行到，
    // 放进分支里，边界 block 会有线程永远到不了 -> 死锁或未定义行为。
    __syncthreads();

    if (x < cols && y < rows)
        out[(size_t)y * cols + x] = tile[threadIdx.y][threadIdx.x];
}

// ---- naive：读合并、写跨行 ----
// out[x][y] = in[y][x]。warp 内 x 连续变化：
//   读 in[y*cols + x]  -> 步长 4 B         -> 4 sectors/request
//   写 out[x*rows + y] -> 步长 rows*4 B    -> 32 sectors/request
__global__ void transposeNaive(const float* in, float* out, int rows, int cols)
{
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < cols && y < rows)
        out[(size_t)x * rows + y] = in[(size_t)y * cols + x];
}

// ---- v2：shared memory tile 转置 ----
// 读阶段：和 copy 一模一样，合并读入 tile[ty][tx]。
// 写阶段：block (bx, by) 负责输入的 (行 by, 列 bx) 块，
//         转置后落在输出的 (行 bx, 列 by) 块 —— 所以 blockIdx.x / y 互换。
//         块内仍让 threadIdx.x 沿输出的"行"方向连续 -> 写也合并。
//         代价是 tile 要按列读：tile[tx][ty]。
__global__ void transposeSmemV2(const float* in, float* out, int rows, int cols)
{
    __shared__ float tile[TILE][TILE];

    // 输入坐标
    int x = blockIdx.x * TILE + threadIdx.x;   // 输入列
    int y = blockIdx.y * TILE + threadIdx.y;   // 输入行
    if (x < cols && y < rows)
        tile[threadIdx.y][threadIdx.x] = in[(size_t)y * cols + x];

    __syncthreads();

    // 输出坐标（输出矩阵 cols × rows）
    int ox = blockIdx.y * TILE + threadIdx.x;  // 输出列，范围 [0, rows)
    int oy = blockIdx.x * TILE + threadIdx.y;  // 输出行，范围 [0, cols)
    // out[oy][ox] 应等于 in[ox][oy]：
    //   in 的行 ox 在 tile 里是第 (ox - by*TILE) = tx 行，
    //   in 的列 oy 在 tile 里是第 (oy - bx*TILE) = ty 列  -> tile[tx][ty]
    if (ox < rows && oy < cols)
        out[(size_t)oy * rows + ox] = tile[threadIdx.x][threadIdx.y];
}

// ============================================================================
// kernel 表：统一签名，方便循环。__global__ 函数可以取地址后用 <<<>>> 启动。
// ============================================================================
using KernelFn = void (*)(const float*, float*, int, int);

struct KernelEntry {
    const char* name;
    KernelFn    fn;
    bool        isTranspose;   // 决定和哪个参考结果比
};

static const KernelEntry kKernels[] = {
    {"copy",              copyKernel,      false},
    {"copy_smem",         copySmem,        false},
    {"transpose_naive",   transposeNaive,  true },
    {"transpose_smem_v2", transposeSmemV2, true },
};
constexpr int kNumKernels = sizeof(kKernels) / sizeof(kKernels[0]);

// ============================================================================
// CPU 参考与校验
// ============================================================================
static void cpuTranspose(const std::vector<float>& in, std::vector<float>& out,
                         int rows, int cols)
{
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
            out[(size_t)x * rows + y] = in[(size_t)y * cols + x];
}

static bool verify(const float* d_out, const std::vector<float>& ref,
                   std::vector<float>& h_tmp, const char* name)
{
    CUDA_CHECK(cudaMemcpy(h_tmp.data(), d_out,
                          ref.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < ref.size(); ++i) {
        if (h_tmp[i] != ref[i]) {   // 纯搬运，可以精确比较；NaN != 任何数
            printf("  [FAIL] %s: idx=%zu got=%.1f want=%.1f\n",
                   name, i, h_tmp[i], ref[i]);
            return false;
        }
    }
    return true;
}

// 每次校验前把 out 刷成 0xFF（= NaN）。
// W1D6 没做这步：如果后一个 kernel 什么都没写，d_out 里还留着
// 前一个 kernel 的正确结果，校验会"假通过"。
static void poison(float* d_out, size_t bytes)
{
    CUDA_CHECK(cudaMemset(d_out, 0xFF, bytes));
}

// 非方阵 / 非 32 倍数的正确性检查，所有 kernel 都过一遍
static bool runCheck(int rows, int cols)
{
    const size_t elems = (size_t)rows * cols;
    const size_t bytes = elems * sizeof(float);

    std::vector<float> h_in(elems), h_ref(elems), h_tmp(elems);
    for (size_t i = 0; i < elems; ++i) h_in[i] = (float)i;
    cpuTranspose(h_in, h_ref, rows, cols);

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);

    bool ok = true;
    for (const KernelEntry& k : kKernels) {
        poison(d_out, bytes);
        k.fn<<<grid, block>>>(d_in, d_out, rows, cols);
        CUDA_CHECK_LAUNCH();
        if (!verify(d_out, k.isTranspose ? h_ref : h_in, h_tmp, k.name)) {
            printf("         shape = %d x %d\n", rows, cols);
            ok = false;
        }
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return ok;
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv)
{
    const char* mode        = (argc > 1) ? argv[1] : "bench";
    const bool  profileMode = (std::strcmp(mode, "profile") == 0);

    // 默认 4096：① 和 W1D6 同尺寸，naive 数字可以直接对上；
    // ② 递增序列在 float 里精确表示的上限是 2^24 = 4096²，再大校验会漏检。
    // 64 MB/矩阵，读+写 128 MB，已远大于 L2。
    const int n = (argc > 2) ? std::atoi(argv[2]) : 4096;

    const size_t elems      = (size_t)n * n;
    const size_t matBytes   = elems * sizeof(float);
    const size_t movedBytes = 2 * matBytes;

    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    // ---- profile 模式：每个 kernel 只启动一次 ----
    if (profileMode) {
        float *d_in = nullptr, *d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_in,  matBytes));
        CUDA_CHECK(cudaMalloc(&d_out, matBytes));
        CUDA_CHECK(cudaMemset(d_in, 0, matBytes));

        for (const KernelEntry& k : kKernels)
            k.fn<<<grid, block>>>(d_in, d_out, n, n);
        CUDA_CHECK_LAUNCH();

        printf("profile 模式：%d 个 kernel 各启动一次（n = %d），交给 ncu 采集。\n",
               kNumKernels, n);
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        return 0;
    }

    // ---- bench 模式 ----
    print_device_info();

    // 1. 正确性：先过非方阵，再谈性能
    printf("正确性检查（非方阵 / 非 32 倍数）...\n");
    const int shapes[][2] = { {1000, 1500}, {1500, 1000}, {33, 65} };
    for (const auto& s : shapes)
        if (!runCheck(s[0], s[1])) return 1;
    printf("  全部通过\n\n");

    // 2. 性能：n × n 方阵
    printf("矩阵             : %d x %d float = %.1f MB / 单个矩阵\n",
           n, n, matBytes / 1048576.0);
    printf("有效字节数       : %.1f MB / 次调用\n", movedBytes / 1048576.0);
    printf("block            : (%d, %d)   grid = (%d, %d)\n\n",
           block.x, block.y, grid.x, grid.y);

    std::vector<float> h_in(elems), h_ref(elems), h_tmp(elems);
    for (size_t i = 0; i < elems; ++i) h_in[i] = (float)i;
    cpuTranspose(h_in, h_ref, n, n);

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  matBytes));
    CUDA_CHECK(cudaMalloc(&d_out, matBytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), matBytes, cudaMemcpyHostToDevice));

    print_result_header();
    double gbps[kNumKernels] = {};

    for (int i = 0; i < kNumKernels; ++i) {
        const KernelEntry& k = kKernels[i];
        auto launch = [&] { k.fn<<<grid, block>>>(d_in, d_out, n, n); };

        poison(d_out, matBytes);
        launch();
        CUDA_CHECK_LAUNCH();
        if (!verify(d_out, k.isTranspose ? h_ref : h_in, h_tmp, k.name)) return 1;

        BenchResult r = benchmark(launch, movedBytes);
        print_result(k.name, r);
        gbps[i] = r.gbps();
    }

    // 3. 汇总
    const double peakBW = query_peak_bandwidth_gbs();
    const double copyG  = gbps[0];

    printf("\n%-22s %10s %10s\n", "kernel", "% copy", "% peak");
    for (int i = 0; i < kNumKernels; ++i) {
        printf("%-22s %9.1f%%", kKernels[i].name, 100.0 * gbps[i] / copyG);
        if (peakBW > 0) printf(" %9.1f%%", 100.0 * gbps[i] / peakBW);
        printf("\n");
    }

    // 下标对应 kKernels：0 copy / 1 copy_smem / 2 naive / 3 v2
    printf("\n");
    printf("v2 / naive        : %.2fx\n",   gbps[3] / gbps[2]);
    printf("copy_smem / copy  : %.1f%%   <- shared 中转本身的代价\n",
           100.0 * gbps[1] / gbps[0]);
    printf("v2 / copy_smem    : %.1f%%   <- global 访存相同，差在哪？看 ncu\n",
           100.0 * gbps[3] / gbps[1]);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
