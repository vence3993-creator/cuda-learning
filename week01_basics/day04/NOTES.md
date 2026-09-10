# Day 4 — 计时与 benchmark 模板

**日期**：2026-09-10
**硬件**：NVIDIA GeForce RTX 5060 Ti (sm_120, Blackwell)
**产出**：`common/benchmark.h`（可复用）、`day04/vecadd.cu`、`day04/vecadd_prof.cu`

**一句话结论**：vector add 在这张卡上跑到 **394 GB/s / 88% 理论峰值**（ncu 报 90.47% of sustained peak），
硬件计数器与手算字节数在 1.6% 内闭环。它是彻底的 memory bound，算力利用率只有 **0.14%**。

---

## 一、理论

### 1.1 为什么 CPU 计时器会骗人

CUDA kernel launch 是**异步**的：`kernel<<<>>>()` 只是把任务塞进 stream 队列，CPU 立刻返回。

| 测法 | 实际量到的东西 | 能不能用 |
|---|---|---|
| `chrono` 包住单次 launch，不同步 | CPU 把任务塞进队列的时间（几 µs） | ❌ 完全是假的 |
| `chrono` 包住 N 次 launch，不同步 | 队列填满（约 1024 个 pending）后产生反压，平均值被拉回真实值附近 | ⚠️ 碰巧接近，不可靠 |
| `chrono` + 循环**内**同步 | 单次端到端延迟 = launch + kernel + 同步返回 | ⚠️ 偏大，测的是延迟不是吞吐 |
| `chrono` + 循环**外**同步 | 稳态吞吐，launch 开销被流水线掩盖 | ✅ 可用 |
| `cudaEvent` | GPU 时间线上的真实耗时 | ✅ 标准做法 |

**`cudaDeviceSynchronize` 该放哪**：放在计时循环**外面**。
放里面测的是延迟，放里面还会破坏流水线。

### 1.2 cudaEvent 机制

```cpp
cudaEventRecord(start, stream);   // 把一个"标记"插进 stream 时间线，本身也是异步的
kernel<<<...>>>();
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);       // 必须：等 stop 真的被 GPU 执行到
cudaEventElapsedTime(&ms, start, stop);  // float，单位 ms
```

- 少了 `cudaEventSynchronize` → 返回 `cudaErrorNotReady`
- 分辨率 ≈ 0.5 µs → 小 kernel 单次测量误差占比大，必须多次迭代
- event 记录的是 **GPU 时间线**，与 CPU 无关

### 1.3 warmup 消除三件事

1. CUDA context 首次创建（几百 ms）
2. module 首次加载 / 可能的 JIT 编译
3. GPU 空闲降频，跑几轮才 boost 上来

跑 10 次丢掉，`cudaDeviceSynchronize()` 收尾，再开始正式测。

### 1.4 统计量的选择

| 量 | 含义 | 何时用 |
|---|---|---|
| **median** | 抗噪声，单次异常样本不污染 | ✅ 默认用这个报数 |
| **min** | 最理想情况下的能力（噪声只让时间变长，不会变短） | 看硬件上限 |
| **mean** | 会被慢样本拖累 | 一般不用 |
| **max** | 看噪声幅度 | median 与 max 差很大 → 环境有干扰 |

实测印证：block sweep 里 max 比 median 高 8%，而 min 极稳（±0.5%）。

### 1.5 计时循环里绝对不能放的东西

- `cudaMemcpy` → 你测的会变成 PCIe 而不是 kernel
- `cudaDeviceSynchronize` → 破坏流水线，变成测延迟
- 内存分配 / 释放
- 正确性校验（必须放在所有计时**之后**）

### 1.6 有效带宽与算力

```
GB/s    = bytes / (ms × 1e-3) / 1e9
GFLOP/s = flops / (ms × 1e-3) / 1e9
```

`bytes` 由**调用方**提供——只有调用方知道自己的 kernel 搬了多少数据。
vector add：读 a、读 b、写 c → `bytes = 3 × n × 4`，`flops = n`。

### 1.7 两个会让数字变成垃圾的陷阱

**陷阱 A：L2 缓存污染**
工作集小于 L2 时，测的是 L2 带宽而非显存带宽，结果会**超过理论峰值**。
本卡 L2 = 32 MB，工作集必须 ≥ 96 MB 才可信。

**陷阱 B：launch overhead 地板**
kernel 太小时，测出来的是 launch + event record 开销（本机地板约 6–8 µs），
与内存系统无关。
**规则：benchmark 的 kernel 至少要跑 100 µs 以上。**

### 1.8 read-for-ownership

写 cache line 时若不是整条写满，硬件会先把它读进来，导致实际 DRAM 写流量膨胀。
**检测方法：看 `dram__sectors_op_read : dram__sectors_op_write` 的比值。**
vector add 理论比值 2:1；若写向读靠拢（趋近 1:1）说明触发了。

---

## 二、工程：benchmark.h 的设计

### 2.1 header 里放东西的 ODR 规则

| 放什么 | 会重复定义吗 | 处理 |
|---|---|---|
| `#define` 宏 | 否（预处理阶段展开） | 直接写 |
| `struct` / `class` 定义 | 否（类型定义允许重复） | 直接写 |
| **模板函数** | 否（模板天然 inline 语义） | 直接写 |
| **class 内部定义的成员函数** | 否（自动 inline） | 直接写 |
| **普通函数（有函数体）** | **会** | **必须加 `inline`** |

外加 `#pragma once` 防止同一个 `.cu` 内重复 include（与上面是两回事）。

### 2.2 接口设计原则

```cpp
BenchResult benchmark(Body&& body, size_t bytes, size_t flops = 0,
                      int warmup = 10, int iters = 200, cudaStream_t s = 0);
```

- **`Body` 用模板而非 `std::function`** → lambda 被内联，零调用开销
- **`bytes` / `flops` 由调用方传入** → 接口对任意 kernel 稳定，不用为每个 kernel 改 header
- **lambda 里只放 launch**，不放拷贝、不放同步
- `flops = 0` 表示纯访存 kernel，输出时跳过 GFLOP/s 列

### 2.3 必须有的保险

kernel launch **不返回错误码**。`vecAdd<<<0, 256>>>()` 这类配置错误会静默失败，
kernel 压根没跑，然后 benchmark 报出 0.001 ms / 3000 GB/s 的"神迹"。

```cpp
CUDA_CHECK(cudaGetLastError());       // 抓 launch 配置错误
CUDA_CHECK(cudaDeviceSynchronize());  // 抓 kernel 执行期的异步错误
```

两个都要，warmup 之后立刻查。

### 2.4 其他细节

- 累加 200 个 `float` 样本时**用 `double` 求和**，否则丢精度
- 析构函数里不要用会 `exit()` 的 `CUDA_CHECK`
- `cudaMemset` 是**按字节**填的，`cudaMemset(p, 1, n)` 会让每个 float 都是 `0x01010101`；
  数据高度重复可能触发显存压缩，benchmark 输入应当有变化

---

## 三、实测结果

### 3.1 硬件参数

| 项 | 值 |
|---|---|
| GPU | RTX 5060 Ti (sm_120, Blackwell) |
| SM 数量 | 36 |
| CUDA core | 36 × 128 = 4608 |
| L2 cache | 32.0 MB |
| 显存 | 7.9 GB |
| 理论带宽（位宽×频率×2） | **448.0 GB/s** |
| ncu sustained peak（反推） | ≈ 441 GB/s |
| FP32 峰值算力（估） | ≈ 23.7 TFLOPS |

### 3.2 扫描 N（block = 256）

| N | 工作集 | median (ms) | GB/s | % peak | 所处 regime |
|---|---|---|---|---|---|
| 2^18 | 3 MB | 0.00858 | 366.81 | 81.9% | launch 开销主导 |
| 2^19 | 6 MB | 0.00970 | 648.87 | 144.8% | launch + L2 |
| 2^20 | 12 MB | 0.01011 | 1244.35 | 277.7% | L2 |
| 2^21 | 24 MB | 0.01562 | **1611.54** | 359.7% | **L2 峰值** |
| 2^22 | 48 MB | 0.09920 | 507.38 | 113.2% | L2 溢出中 |
| 2^23 | 96 MB | 0.25264 | 398.45 | 88.9% | **DRAM（真值）** |
| 2^24 | 192 MB | 0.50768 | 396.56 | 88.5% | DRAM |
| 2^25 | 384 MB | 1.01622 | 396.22 | 88.4% | DRAM |
| 2^26 | 768 MB | 2.03952 | 394.85 | 88.1% | DRAM |
| 2^27 | 1536 MB | 4.08861 | 393.93 | 87.9% | DRAM |

**三个结论**

1. **拐点精确落在 L2 边界**：24 MB 装得下 → 1611 GB/s；48 MB 装不下 → 崩到 507。
2. **L2 带宽 ≳ 1611 GB/s，约为 DRAM 的 4 倍。** 这个 4:1 就是后面所有 tiling / shared memory 优化的收益上限来源。
3. **左端不是内存慢，是 kernel 太小。** 数据量 ×4（3→12 MB）而时间只涨 18%，
   说明被 6–8 µs 的 launch 开销地板锁死了。

### 3.3 扫描 block size（N = 2^26, 768 MB）

| block | median (ms) | min (ms) | max (ms) | GB/s | GFLOP/s |
|---|---|---|---|---|---|
| 64 | 2.03955 | 2.02525 | 2.21530 | 394.84 | 32.90 |
| 128 | 2.03955 | 2.02317 | 2.22970 | 394.84 | 32.90 |
| 256 | 2.03859 | 2.02317 | 2.05888 | 395.03 | 32.92 |
| **512** | **2.02698** | **2.01264** | 2.19901 | **397.29** | 33.11 |
| 1024 | 2.04550 | 2.03325 | 2.20934 | 393.70 | 32.81 |

**结论：纯 streaming kernel 对 block size 完全不敏感**（全距 0.6%，落在噪声内）。

原因：36 个 SM，2^26/64 = 100 万个 block，怎么切都喂得饱；
且 vector add 每线程只用几个寄存器，不存在寄存器压力导致的占用率损失。

> 这个"不敏感"本身是特征。Day 6 的 transpose 会看到 block 形状直接改变性能，
> 那时候的对比才有信息量。

### 3.4 ncu 交叉验证（N = 2^26, block = 256）

```bash
ncu --metrics dram__bytes.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,\
dram__sectors_op_read.sum,dram__sectors_op_write.sum,gpu__time_duration.sum,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed ./vecadd_prof
```

| 指标 | 手算 | ncu 实测 | 偏差 |
|---|---|---|---|
| DRAM 读 | 536.87 MB | 543.11 MB | +1.2% |
| DRAM 写 | 268.44 MB | 249.49 MB | −7.1% |
| DRAM 合计 | 805.31 MB | **792.59 MB** | −1.6% |
| 读 sectors | 16,777,216 | 16,972,064 | +1.2% |
| 写 sectors | 8,388,608 | 7,796,504 | −7.1% |
| **读:写 比** | **2.00 : 1** | **2.18 : 1** | — |
| kernel 时间 | 2.04 ms (cudaEvent) | 1.99 ms | — |
| DRAM throughput | 394.85 GB/s | 399.29 GB/s | — |
| % of sustained peak | — | **90.47%** | — |

**三笔账全部闭环**：
`399.29 GB/s × 1.99 ms = 794.6 MB ≈ dram__bytes.sum 792.59 MB ≈ 手算 805.31 MB`

**读写比 2.18:1，未触发 read-for-ownership**（若触发，比值会向 1:1 靠拢）。

**为什么 ncu 时间比 cudaEvent 略小/略大**：ncu 插桩有开销（拉大），
但 ncu 默认每次 replay 前冲刷 L2（`--cache-control all`），测的是彻底冷缓存的单次执行；
而 cudaEvent 测的是 200 次连续执行的中位数。同量级即算对上。

**ncu 的 peak 与自算 peak 不是一回事**：自算 448 GB/s 是理论值，
ncu 用的是 sustained peak（≈441 GB/s），所以 ncu 的百分比会略高。两者都正常。

### 3.5 ncu 指标命名规则

`--query-metrics` 列出的是**基础计数器名**，不能直接用，必须补聚合后缀：

```
dram__bytes_op_read  .  sum
└─ 基础计数器名          └─ 必须补的后缀
```

| 类型 | 可用后缀 |
|---|---|
| Counter | `.sum` `.avg` `.min` `.max`，可再叠 `.per_second` |
| Throughput | `.avg.pct_of_peak_sustained_elapsed` |

sm_120 上 `dram__bytes_read.sum` 已不存在，正确写法是 **`dram__bytes_op_read.sum`**（注意 `op_` 中缀）。
1 sector = 32 bytes。

---

## 四、Roofline 预备（Day 5 直接用）

| 量 | 值 | 来源 |
|---|---|---|
| 斜坡斜率（访存屋顶） | **448 GB/s** | 今天测的 |
| 水平线（算力屋顶） | **≈ 23.7 TFLOPS** | 36 SM × 128 core × 2 flop × ~2.57 GHz |
| **拐点横坐标** | **≈ 53 flop/byte** | 23700 ÷ 448 |
| vector add 算术强度 | **0.083 flop/byte** | 1 flop ÷ 12 byte |
| vector add 实测 | 32.9 GFLOP/s | = 峰值的 **0.14%** |

vector add 落在拐点**左侧约 640 倍**的位置，且贴着斜线跑（88–90%）。
**这就是 memory bound 最赤裸的形态：99.86% 的算力在空转等数据。**

---

## 五、遗留问题

- [ ] **写流量少了 18.95 MB（−7.1%）**，两个候选：
  - **A：L2 脏数据残留** — L2 是 write-back，kernel 结束瞬间最后写的一批还在 L2 未刷回 DRAM。
    L2 = 32 MB，残留 19 MB 完全在射程内 → **嫌疑较大**
  - **B：显存压缩** — prof 版本用 `cudaMemset` 填充，数据高度重复
  - **判别实验**：把 `cudaMemset` 换成有变化的数据重跑 ncu。
    写涨到 268 MB → B；仍是 249 MB → A
- [ ] Makefile 里补上 `-arch=sm_120`（目前是默认架构，走 PTX JIT）
- [ ] 跑 `bandwidthTest --dtod` 拿到独立的"实测上限"参照
- [ ] `ncu --section SpeedOfLight` 看 Compute[%] vs Memory[%] 的并排对比

---

## 六、可复用规则清单

1. GPU kernel 计时**只用 cudaEvent**，CPU 计时器必须配合循环外同步
2. **warmup 10 次**并 `cudaDeviceSynchronize()` 收尾
3. 报数用 **median**，看上限用 **min**
4. **工作集必须 ≫ L2**（本卡 ≥ 96 MB）
5. **kernel 时间必须 ≥ 100 µs**，否则被 launch 开销污染
6. 计时循环里**只有 launch**
7. launch 之后**必须** `cudaGetLastError()`
8. 正确性校验放在**所有计时之后**，且必须做
9. benchmark 输入数据**要有变化**，不要用 `cudaMemset`
10. 自算带宽**必须与 ncu 的 `dram__bytes.sum` 对账**才可信

---

## 附：本卡基准（后续每天的参照）

```
RTX 5060 Ti (sm_120) · 36 SM · L2 32 MB · 8 GB

理论显存带宽峰值   448.0 GB/s
ncu sustained peak ≈441   GB/s
L2 带宽            ≳1611  GB/s   (≈ 4× DRAM)
vector add 实测    394.9  GB/s   (88.1% 理论 / 90.47% sustained)
FP32 算力峰值      ≈23.7  TFLOPS
Roofline 拐点      ≈53    flop/byte
launch 开销地板    ≈6–8   µs
```
