# Day 1：环境搭建与硬件认知

> 2026-08-22 | RTX 5060 Ti (8GB) / WSL2 Ubuntu 22.04 / CUDA 12.9

---

## 一、环境层面

### WSL2 的 CUDA 架构

驱动在 Windows 侧，WSL 里**只装 toolkit**。`/usr/lib/wsl/lib/libcuda.so` 是驱动直通的存根。

由此推出两条硬规则：

- 仓库必须选 `wsl-ubuntu`，不是普通 Ubuntu 仓库
- 包名必须是 `cuda-toolkit-12-9`，**不能是 `cuda` 或 `cuda-12-9`** —— 后两个元包会拉 Linux 驱动覆盖存根，直接导致 `nvidia-smi` 和整个 CUDA 失效

### 版本约束

- Blackwell (sm_120) 需要 CUDA ≥ 12.8
- 驱动版本决定支持上限（591.86 → 最高 CUDA 13.1），toolkit 可以低于它但不能超过

### 关于下载加速

国内主流高校镜像站（清华 / 中科大 / 南大 / 浙大 / 北大 / 南科大 / 教育网联合）**都不镜像 NVIDIA CUDA 仓库**，实测全部 404，因为单版本就 4GB+。

真正的瓶颈不是"源在国外"，而是 **apt 单线程下载**。解法是用 aria2 / IDM 多线程拉 local deb，而不是换源。local deb 的额外好处：安装阶段零网络依赖，不会下到 90% 断了重来。

---

## 二、硬件参数（RTX 5060 Ti）

| 项 | 值 | 意义 |
|---|---|---|
| Compute Capability | 12.0 | `-arch=sm_120` |
| SM 数量 | 36 | 并行单元数 |
| CUDA Cores | 4608 (128/SM) | — |
| Warp size | 32 | — |
| Max threads / block | 1024 | 配置上限 |
| **Max threads / SM** | **1536** = 48 warp | occupancy 的分母 |
| Shared mem / block | 48 KB | tile 大小上限 |
| Shared mem / SM | 100 KB | 决定能驻留几个 block |
| Registers / block | 65536 | W6 sgemm 的主战场 |
| **L2 Cache** | **32 MB** | 异常地大，是测试时的陷阱 |
| 显存 | **8 GB**, 128-bit GDDR7 | 注意是 8G 版本 |
| **理论带宽** | **448 GB/s** | 14001 MHz × 2 = 28 Gbps × 128 bit ÷ 8 |
| **理论算力** | **24.3 TFLOPS** | 4608 × 2.632 GHz × 2 (FMA) |
| **平衡点** | **≈ 54 FLOP/Byte** | 低于此值即 memory bound |

> **带宽换算的坑**：`cudaDeviceProp` 报的是时钟频率，不同显存代际（GDDR6 / GDDR6X / GDDR7 / HBM）换算系数不同，不能套同一个公式。GDDR7 用 PAM3 编码，老的 `2 × clock × width / 8` 公式不通用。

---

## 三、概念收获

### 执行模型

```
thread → warp (32) → block → grid
```

全局索引：

```c
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

grid size 用 `(n + 255) / 256` 向上取整，必须配 `if (i < n)` 边界检查 —— **两者成对出现，缺一不可**。

### 异步性

kernel 启动后 CPU 立即返回，不等 GPU。由此推出三条：

1. 不加 `cudaDeviceSynchronize()`，kernel 里的 `printf` 看不到输出
2. 错误检查要**两道**：
   - `cudaGetLastError()` 抓**启动**错误（架构不匹配、block 配置非法）
   - `cudaDeviceSynchronize()` 的返回值抓**执行**错误（非法访存）
   - 只查一个会漏：kernel 若根本没启动，`cudaDeviceSynchronize()` 会安静地返回 `cudaSuccess`
3. 计时不能用 CPU 计时器，必须用 `cudaEvent`

### 输出无序性

`hello<<<2,4>>>()` 的打印顺序每次运行都不同 —— **没有同步就没有顺序保证**。

### 验证不可省

kernel 没跑、算错、只算了一部分，程序都会"正常退出"。**必须比对数值结果**才算跑通。

### 浮点比较

vectorAdd 用 `!=` 直接比较能过（GPU 和 CPU 做同样的单精度加法，IEEE 754 保证逐位相同）。

但从 **reduce（W3）开始必须换成容差比较** —— 多数累加时 GPU 与 CPU 求和顺序不同，浮点不满足结合律，结果必然有微小差异：

```c
if (fabs(hc[i] - ref) > 1e-5) { /* fail */ }
```

---

## 四、性能观察（第一手数据）

### context 初始化开销

nsys 显示三次 `cudaMalloc`：

```
Max: 198,754,088 ns  (198 ms)   ← 第一次
Med:     307,430 ns  (0.3 ms)   ← 后两次
```

**差 600 倍**。原因不是分配内存慢，而是首次 CUDA API 调用触发了驱动加载、上下文建立、模块载入显存。

> 这就是 **benchmark 必须做 warmup** 的原因。

### vecAdd 的 ncu 报告

```
Memory Throughput    92.24 %     ← 访存跑满
Compute Throughput   21.01 %     ← 计算闲置
Duration             22.18 µs
Achieved Occupancy   83.25 %
```

教科书级的 **memory bound**：访存单元 92%、计算单元 21%，GPU 大部分时间在等数据。

算术强度 = 1 FLOP / 12 Byte ≈ **0.083**，远低于平衡点 54，与观测完全吻合。

### L2 陷阱

反推带宽：`12.58 MB ÷ 22.18 µs ≈ 567 GB/s`，**超过了 448 GB/s 的理论值**。

原因：数据仅 12.5 MB，完全装得进 32 MB 的 L2，相当一部分访问没走 DRAM。

> **结论：测真实显存带宽，数据量必须远超 L2。**
> 待办：Day 4 把 `n` 从 `1<<20` 提到 `1<<26`（768 MB）重测，做对比实验。

### Occupancy 的正确读法

```
Block Limit Warps                6      ← 48 warp ÷ 8 warp/block = 6
Theoretical Active Warps per SM  48     ← 1536 threads ✓
Theoretical Occupancy           100 %
Achieved Occupancy            83.25 %
Waves Per SM                  18.96     ← 尾部效应来源
```

理论与实测的 17% 差值来自**尾部效应** —— 将近 19 波，最后 0.96 波没填满，部分 SM 提前空转。

**但这个不该优化。** memory bound 的 kernel 提高 occupancy 没有意义，瓶颈在 DRAM 不在并行度。

> 识别"**哪个指标该优化、哪个不该**"，比会优化更重要。

### 寄存器（留待 W6）

```
Registers Per Thread     16     ← 很低，完全不是瓶颈
Block Limit Registers    16
Block Limit Shared Mem   16
Block Limit SM           24
```

W6 写 sgemm 时每线程可能用到 100+ 寄存器，`Block Limit Registers` 会掉到个位数，届时成为限制 occupancy 的真正瓶颈。

---

## 五、工具链状态

| 工具 | 状态 | 用途 |
|---|---|---|
| nvcc | ✅ 12.9 | 编译，`-arch=sm_120` |
| **ncu** | ✅ 已解权限 | 单 kernel 详细指标，**命脉工具** |
| nsys | ⚠️ 采不到 GPU 数据 | WSL / CUPTI 限制，不影响后续 |
| cudaEvent | 待用 (Day 4) | WSL 下计时准确，benchmark 主力 |

### ERR_NVGPUCTRPERM 的解法

消费级显卡的默认限制，与 WSL 无关，**必须在 Windows 侧改**：

NVIDIA 控制面板 → 桌面 → 启用开发者设置 → 开发者 → 管理 GPU 性能计数器 → **允许所有用户访问** → **整机重启**

---

## 六、工程习惯

- `git add .` 之后**先 `git status` 再 commit`** —— 今天差点把 4.1 GB 的 deb 提交上去
- 编译产物统一输出到 `bin/`，`.gitignore` 一行挡住 —— 无扩展名的可执行文件没法用通配符匹配
- profiling 产物（`*.nsys-rep` `*.sqlite` `*.ncu-rep` `*.qdstrm`）一律忽略
- 每天结束前 commit，message 写清做了什么

---

## 七、Day 1 验收

- [x] `nvcc --version` = 12.9
- [x] `nvidia-smi` 正常（驱动直通未破坏）
- [x] `-arch=sm_120` 编译且**运行结果正确**
- [x] 硬件参数表 + 理论带宽 448 GB/s
- [x] ncu 可用，能读懂 Speed of Light 报告
- [x] git 仓库 + 远程推送

---

## 八、遗留与引子

**待办**

- Day 4：`n = 1<<26` 重测带宽，验证 L2 陷阱
- nsys GPU 侧 tracing（低优先级，不阻塞任何任务）

**Day 2 的三个问题**

1. 为什么 block size 选 256？换成 1024 会发生什么？
   （提示：1536 ÷ 1024 = 1.5，一个 SM 只能放 1 个 block，浪费 512 个线程槽位，occupancy 掉到 67%）
2. warp divergence 到底损失多少 —— 明天写代码实测
3. `if (i < n)` 这个分支本身，会不会造成 divergence？