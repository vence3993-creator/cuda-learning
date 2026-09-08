# CUDA Day 3 学习笔记：PCIe 带宽扫描代码精读

> 源文件：`bandwidth_scan.cu`
> 主题：`cudaMemcpy` 吞吐测量、C++ 计时、`size_t` / `static` / chrono 等语言基础

---

## 0. 程序整体在干什么

CPU 内存（host）和 GPU 显存（device）是两块独立的物理内存，中间隔着 PCIe 总线。`cudaMemcpy` 就是在这条总线上搬数据。

程序流程：

```
从 1 KB 开始，每轮把传输大小翻倍，直到 1 GB
  ├─ 测 H2D (HostToDevice,   CPU → GPU)
  ├─ 测 D2H (DeviceToHost,   GPU → CPU)
  └─ 打印一行 CSV
```

输出格式：`bytes,h2d_GBps,d2h_GBps`

带宽定义就是最朴素的 `传输字节数 / 花的秒数`，单位用十进制 GB（1e9 字节）。

---

## 1. `measure()`：计时的三个坑

```cpp
static double measure(void *dst, const void *src, size_t bytes, cudaMemcpyKind kind) {
    const int iters = pick_iters(bytes);

    for (int i = 0; i < 3; ++i)                 // ← 坑一：warmup
        CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i)
        CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));
    CUDA_CHECK(cudaDeviceSynchronize());        // ← 坑二：同步在循环外
    double total = std::chrono::duration<double>(clk::now() - t0).count();

    double sec_per_copy = total / iters;
    return double(bytes) / sec_per_copy / 1e9;
}
```

### 坑一：第一次调用不算数（warmup）

CUDA 是**懒初始化**的。第一次调用真正干活的 API 时，驱动才会：

- 建立 CUDA context
- 加载模块
- 分配内部 staging 缓冲区

这一下可能几十甚至几百毫秒。另外 host 侧 pageable 内存第一次被 DMA 触摸时还可能触发缺页中断。这些一次性开销算进去，小 size 的结果会离谱得没法看。

### 坑二：`cudaDeviceSynchronize()` 放在循环外

GPU 很多操作是异步的——函数在 CPU 上返回了，不代表活干完了。所以必须同步一次，确保计时结束时活真的做完。

**放循环外 vs 放循环内，测的是两个不同的东西：**

| 位置 | 测到的是 | 数值特点 |
|---|---|---|
| 循环**外** | **吞吐量**（一连串拷贝的平均速率） | 正常 |
| 循环**内** | **单次往返延迟** | 明显偏低 |

这份代码要的是吞吐，所以放外面。

### 坑三：小 size 必须重复很多次

传 1 KB 可能只要 5 µs，而 `steady_clock` 的调用开销和系统调度抖动本身就在这个量级——单次测量基本是噪声。

---

## 2. `pick_iters()`：重复次数的查找表

```cpp
static int pick_iters(size_t bytes) {
    if (bytes <= (1u << 16)) return 2000;   // <= 64 KB
    if (bytes <= (1u << 20)) return 500;    // <= 1 MB
    if (bytes <= (1u << 24)) return 100;    // <= 16 MB
    return 20;                              // 兜底：> 16 MB
}
```

### 位移写法

`1u << n` 就是 2 的 n 次方。比写 `16777216` 更容易一眼看出量级。

| 写法 | 值 | 含义 |
|---|---|---|
| `1u << 16` | 65536 | 64 KB |
| `1u << 20` | 1048576 | 1 MB |
| `1u << 24` | 16777216 | 16 MB |

### `if` 顺序执行、命中即返回

所以每行隐含"并且比上一行阈值大"：

```
bytes ≤ 64 KB          → 2000 次
64 KB < bytes ≤ 1 MB   → 500 次
1 MB  < bytes ≤ 16 MB  → 100 次
bytes > 16 MB          → 20 次   ← 最后的 return 20 就是 else 分支
```

**最后那个 `return 20` 是必须的兜底**：C 函数必须在所有路径上都返回值，缺了它传 32 MB 进来就是未定义行为。

### 为什么大 size 只测 20 次

假设 pageable 带宽 ~6 GB/s、固定开销 ~10 µs：

| size | 单次耗时 | iters | 该点总耗时 |
|---|---|---|---|
| 1 KB | ~10 µs | 2000 | ~20 ms |
| 64 KB | ~21 µs | 2000 | ~42 ms |
| 1 MB | ~175 µs | 500 | ~88 ms |
| 16 MB | ~2.7 ms | 100 | ~270 ms |
| 1 GB | ~167 ms | 20 | **~3.3 s** |

设计意图：**小 size 单次太快、噪声占比大，靠重复平均掉；大 size 信噪比已经很好，测多了纯属浪费时间。**

> ⚠️ 注释里说"保证每点总时长 ~50ms"是理想化说法。实际从 20 ms 涨到 3 秒多，这个阶梯只是把增长压平了些，没真正拉成常数。

**更正规的做法**（不预设次数，跑到累计时间够为止）：

```cpp
auto t0 = clk::now();
cudaMemcpy(dst, src, bytes, kind);
cudaDeviceSynchronize();
double one = std::chrono::duration<double>(clk::now() - t0).count();
int iters = std::max(3, (int)(0.05 / one));   // 目标 50 ms
```

---

## 3. 语言基础：`size_t`

**它是无符号整数类型**，专门表示"大小"和"字节数"。64 位系统上是 64 位无符号（等价 `unsigned long`），范围 0 ~ 约 1.8×10¹⁹。

### 为什么不用 `int`

1. **位宽不够**：`int` 通常 32 位有符号，最大约 21 亿。测到 1 GB 勉强够，改成 4 GB 就溢出变负数。
2. **语义上不该有负数**：大小永远 ≥ 0。

### 约定俗成

`sizeof` 的结果、`malloc` 的参数、`strlen` 的返回值、STL 的 `.size()`，全是 `size_t`。CUDA 也一样：`cudaMalloc` 第二参数、`cudaMemcpy` 第三参数都是 `size_t`。

### ⚠️ 无符号回绕坑

```cpp
size_t bytes = 0;
bytes - 1024;   // 不是 -1024，而是一个巨大的正数！
```

所以循环条件写 `bytes <= max_bytes`，别写 `bytes - x >= 0`。

同理，主循环用 `size_t` 才安全：

```cpp
for (size_t bytes = MIN_BYTES; bytes <= max_bytes; bytes *= 2)
```

如果用 `int`，翻到 2³¹ 溢出成负数，循环条件直接失效。

---

## 4. 语言基础：`static`

**`static` 不是类型，是关键字**（存储类说明符）。

```cpp
static int pick_iters(size_t bytes)
│      │
│      └─ 这才是类型：返回 int
└─ 管的是"链接性/生命周期"
```

C++ 复用了这个关键字，**三种位置三种含义**：

| 位置 | 含义 | 记法 |
|---|---|---|
| 文件作用域的函数/变量前 | 内部链接，只在本编译单元可见 | "藏起来" |
| 函数内的局部变量前 | 静态生命周期，跨调用保留值 | "活得久" |
| 类成员前 | 属于类而非对象，所有实例共享 | "大家共用" |

### 本代码用的是第 ① 种

```cpp
static int pick_iters(...)     // 只有 bandwidth_scan.cu 能调用
static double measure(...)
```

好处：

- **避免符号冲突**：以后写第二个 .cu 也有 `measure` 时，不加 `static` 会链接报 "multiple definition"
- **帮助优化**：编译器确定没有外部调用者，更容易内联

单文件小程序里加不加无实质区别，但是好习惯。（C++ 更现代的做法是放进匿名 namespace。）

### CUDA 里额外注意

- **`static __device__` 变量**：跨文件访问设备变量需要 `nvcc -rdc=true`，不加 `static` 且没开 `-rdc` 容易踩坑
- **"静态共享内存" 是术语不是关键字**：
  ```cpp
  __shared__ float buf[256];      // 大小编译期确定 → 叫"静态共享内存"（没写 static！）
  extern __shared__ float buf[];  // 大小 kernel 启动时给 → "动态共享内存"
  ```
- **别在 `__device__`/`__global__` 里用 static 局部变量**：它**不是每线程一份**，所有线程共享同一份放在全局内存，会有数据竞争。每线程私有用普通局部变量，块内共享用 `__shared__`

> 本代码的两个 `static` 函数都是**纯 host 函数**（无 `__device__`/`__global__` 修饰），跟 CUDA 无关，nvcc 直接交给 gcc/msvc 处理。

---

## 5. 语言基础：chrono 计时

### `using clk = std::chrono::steady_clock;`

这是**类型别名**（C++11），等价于老式 `typedef`：

```cpp
using clk = std::chrono::steady_clock;      // C++11，新名在左
typedef std::chrono::steady_clock clk;      // C 遗留，方向相反
```

不创建新类型、无运行时开销，纯粹起个短外号。

### `<chrono>` 的三个核心概念

| 概念 | 是什么 | 例子 |
|---|---|---|
| **clock** | 时间的来源 | `steady_clock` |
| **time_point** | 某个瞬间 | `t0`、`clk::now()` 的返回值 |
| **duration** | 两点之间的间隔 | `clk::now() - t0` |

最大好处：**单位写进了类型里**，编译器帮你做换算，不会再"以为是毫秒其实是微秒"。

### 三个时钟怎么选

| 时钟 | 特点 | 用途 |
|---|---|---|
| `system_clock` | 墙上时钟，**可以往回跳**（NTP 校时、手动改时间、夏令时） | 显示日期 |
| `steady_clock` | **单调递增、步长恒定**，不知道"今天几号" | **测时间间隔** ✅ |
| `high_resolution_clock` | 标准没规定它必须单调，各实现不一（libstdc++ 上是前者，MSVC 上是后者） | 别用 |

> 规则：**测间隔用 `steady_clock`，显示日期用 `system_clock`。**

### 拆解那一行

```cpp
double total = std::chrono::duration<double>( clk::now() - t0 ).count();
                                             └──── ① ────┘
                                             └──── ② ─────┘
               └──────────── ③ ─────────────────────────┘
               └──────────────── ④ ───────────────────────────┘
```

假设实际跑了 52.3 毫秒：

| 步骤 | 干了什么 | 结果 |
|---|---|---|
| ① `clk::now()` | 读当前时刻 | `time_point`（终点） |
| ② `- t0` | 终点减起点 | `duration<long, nano>`，内含 `52300000` |
| ③ `duration<double>(...)` | 换算单位到秒 | `duration<double, ratio<1,1>>`，内含 `0.0523` |
| ④ `.count()` | 剥掉包装取裸数值 | `double 0.0523` |

**②和③都是 duration**，区别只在"用什么单位、什么类型存"。

`duration<double>` 是 `duration<double, std::ratio<1,1>>` 的简写：

- 第一个模板参数 = 用什么类型存数值（rep）
- 第二个 `ratio<1,1>` = 一个计数单位等于多少秒（period），这里就是 **1 秒**

### ⚠️ 为什么不用 `duration_cast`

```cpp
auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(clk::now() - t0);
double total = ms.count() / 1000.0;   // ❌ 会丢精度！
```

`duration_cast` 转到**整数单位时是截断的**，52.3 ms → 52，那 0.3 直接没了。

而 `duration<double>` 目标是浮点，**不会有精度损失**，所以标准允许隐式转换、不需要 cast。这就是这里写构造语法而非 cast 的原因。

### ⚠️ `.count()` 的单位取决于前面套了什么

```cpp
duration<double>(d).count()               // → 0.0523      秒
duration<double, std::milli>(d).count()   // → 52.3        毫秒
d.count()                                 // → 52300000    纳秒（原生单位）
```

同一个 `d`，三个完全不同的数字。看到 `.count()` 一定要回头看外面套的 duration 类型。

### 更好读的拆法

```cpp
using seconds_d = std::chrono::duration<double>;

auto elapsed = clk::now() - t0;              // duration，纳秒整数
double total = seconds_d(elapsed).count();   // 转成秒，取出数值
```

---

## 6. 带宽计算：`return double(bytes) / sec_per_copy / 1e9;`

两个 `/` 都是除号，**从左往右结合**：

```
第一步:  bytes / sec_per_copy    →  字节 ÷ 秒  =  字节/秒
第二步:  ↑结果 / 1e9             →  换算成 GB/秒
```

具体数字（16 MB，单次 2.7 ms）：

```
16777216 / 0.0027  =  6,213,783,703  字节/秒
6213783703 / 1e9   =  6.214          GB/s
```

### `double(bytes)` 是类型转换，不是除法的一部分

等价于 C 风格的 `(double)bytes`。

**为什么需要**：整数 ÷ 整数 = 整数除法，会截断。

```cpp
7 / 2      // → 3     整数除法，0.5 被丢掉
7.0 / 2    // → 3.5   有一边是浮点就是浮点除法
```

这里 `sec_per_copy` 本身已是 `double`，不写也会自动提升，结果一样。**写出来是为了明确表达意图**——让读者一眼看出没有截断。

### ⚠️ GB 有两种

| 除数 | 结果 | 单位 |
|---|---|---|
| `1e9` | 6.214 | GB（十进制） |
| `1 << 30` | 5.787 | GiB（二进制） |

两者差 7.4%。带宽场景**十进制是主流**（PCIe 官方标称、`nvidia-smi`、NVIDIA 自家 `bandwidthTest` 都是），所以这里用 `1e9` 是对的。但跟别人数据对比时要确认对方用哪种。

---

## 7. `main()` 里的设计决定

### 缓冲区只分配一次

```cpp
void *h_buf = malloc(max_bytes);
memset(h_buf, 1, max_bytes);      // 关键
cudaMalloc(&d_buf, max_bytes);
```

**如果每个 size 都重新分配**，测到的就不只是传输速度，还混进了分配开销、以及"物理页到底分配了没"这种状态差异。分配一次、小 size 只用开头那一段，所有数据点起跑线才一致。

**那个 `memset` 不是为了填数据**——Linux 的 `malloc` 返回的只是虚拟地址，物理页要等真正写入时才分配。先写一遍把物理页坐实，免得计时时撞上缺页中断。

### 输出参数模式：`cudaMemGetInfo`

```cpp
size_t free_mem = 0, total_mem = 0;
CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
```

C 函数**只能返回一个值**，但这里要返回两个数字。标准做法是调用方先准备好变量，把**变量的地址**交给函数，函数通过指针写回去。这种参数叫**输出参数**。

```cpp
cudaError_t cudaMemGetInfo(size_t *free, size_t *total);
//          ↑返回值留给错误码     ↑两个指针，用来写结果
```

**几乎所有 CUDA API 都是这个模式**：返回值永远是错误码（所以外面要套 `CUDA_CHECK`），真正的结果通过指针参数往外传：

```cpp
cudaGetDeviceProperties(&prop, dev);   // prop 是输出参数
cudaMalloc(&d_buf, max_bytes);         // d_buf 是输出参数（指针的指针）
```

**为什么初始化成 0**：在这段代码里其实没有实际作用（成功必被覆盖，失败会 `exit()`）。但这是值得保持的防御性习惯：

1. 未初始化的局部变量是**栈上残留的垃圾值**，可能是天文数字
2. 如果哪天把 `CUDA_CHECK` 换成不退出的版本，垃圾值会让后面的 `free_mem / 4` 判断彻底失效
3. 消除"debug 能跑 release 崩"这类不可复现的 bug

> 对比：同段的 `int dev = 0;` 里的 `0` 是**真正有意义的输入**（选择 0 号 GPU）。
> 判断方法：**看变量接下来是被读还是被写**。被 `&` 取地址传进函数的通常是输出参数，初值只是占位。

### 显存上限保护

```cpp
while (max_bytes > free_mem / 4 && max_bytes > MIN_BYTES) max_bytes /= 2;
```

`free_mem` 是**当前**可用显存，不是显卡总容量，会受影响于：

- 桌面环境、浏览器（Linux 上跑 X/Wayland 通常几百 MB）
- 别的进程正在用 GPU
- CUDA context 自身开销（几十到几百 MB）

留 3/4 余量是因为 `cudaMalloc` 需要**连续**的显存块，且 CUDA 内部还要留空间做 staging buffer。

### stdout / stderr 分开

设备信息走 `fprintf(stderr, ...)`，只有 CSV 数据走 `printf`（stdout）。

这样 `./bandwidth_scan > pageable.csv` 得到的是干净的、能直接喂给 pandas 的文件，而信息还打在终端上。**很实用的 Unix 习惯。**

---

## 8. 主循环逐句拆解

```cpp
for (size_t bytes = MIN_BYTES; bytes <= max_bytes; bytes *= 2) {
    double h2d = measure(d_buf, h_buf, bytes, cudaMemcpyHostToDevice);
    double d2h = measure(h_buf, d_buf, bytes, cudaMemcpyDeviceToHost);
    printf("%zu,%.3f,%.3f\n", bytes, h2d, d2h);
    fflush(stdout);
}
```

### 为什么是 `*= 2`（等比而非等差）

范围横跨六个数量级（1 KB → 1 GB）。等差的话每次加 1 KB 需要一百万个点，且 95% 挤在早已到平台期的大 size 区。

翻倍只有 **21 个点**，而且在**对数坐标上均匀分布**——正好对应要画的 log-x 曲线图。跨数量级测量几乎都用倍增扫描。

### ⚠️ 两次调用的前两个参数是对调的

```cpp
measure(d_buf, h_buf, ...)   // H2D: dst = d_buf, src = h_buf
measure(h_buf, d_buf, ...)   // D2H: dst = h_buf, src = d_buf
//      ↑目标   ↑源
```

参数顺序是 `(目标, 源)`，和 `cudaMemcpy`、`memcpy` 一致。

**经典陷阱**：中文语序说"把 A 拷到 B"，但代码里 B 在前。写反不会编译报错，只会得到莫名其妙的结果甚至段错误。

> 记法：**跟赋值语句 `dst = src` 同方向。**

第三个参数 `bytes` 只用到缓冲区的**前 `bytes` 个字节**（缓冲区是按 `max_bytes` 一次性分配的）。

### `%zu` 里的 `z` 是关键

| 占位符 | 含义 |
|---|---|
| `%zu` | `size_t` 类型的无符号整数 |
| `%.3f` | `double`，保留 3 位小数 |

`z` 是长度修饰符，表示"这个参数是 `size_t` 大小的"。

**不能直接写 `%u`**：`%u` 期待 32 位 `unsigned int`，而 64 位系统上 `size_t` 是 64 位。类型不匹配时 `printf` 会读错字节数，打印垃圾值——而且这是**未定义行为**。

`printf` 是变参函数，编译器本无法检查参数类型，但 gcc/clang 会特殊分析格式串，`-Wall` 能抓到。

### `fflush(stdout)` 为什么必须有

C 的 stdout 缓冲策略取决于输出目标：

| 输出到 | 缓冲方式 |
|---|---|
| 终端 | 行缓冲，遇 `\n` 就刷 |
| **重定向到文件** | **全缓冲，攒够 ~4 KB 才写** |

而这个程序的运行方式恰恰是重定向到文件。不加 `fflush`：

1. **看不到进度**：大 size 每点要跑好几秒，盯着文件半天啥也没有
2. **中断会丢数据**：`exit()` 会刷缓冲，但 `abort()` 和信号杀死不会

代价只是每行一次系统调用，相对几十毫秒的测量完全可忽略。**测量程序边算边刷是标准做法。**

### 整体耗时预估

| size 区间 | 点数 | 每点耗时 |
|---|---|---|
| 1KB ~ 64KB | 7 | 40~85 ms |
| 128KB ~ 1MB | 4 | 90~180 ms |
| 2MB ~ 16MB | 4 | 0.1~0.5 s |
| 32MB ~ 1GB | 6 | 0.2~7 s |

总计约 **十几到三十秒**，时间几乎全花在最后两三个点上。运行时会明显感觉"前面刷刷刷一堆，最后卡住不动"——正常现象。

---

## 9. 结果怎么看

把 CSV 画出来（**x 轴用对数**），典型形状：

```
带宽
 ↑                    ┌──────────  ← 平台期，逼近 PCIe 上限
 │                 ／
 │              ／      ← 快速爬升
 │        ／
 │  ／                  ← 小 size：带宽极低，几乎跟 size 成正比
 └──────────────────────→ size (log)
   1KB   64KB   1MB   16MB  1GB
```

**小 size 为什么低**：每次 `cudaMemcpy` 有固定启动开销（驱动调用、DMA 引擎配置），大概 5~10 µs。传 1 KB 时时间全花在这上面。

**拟合模型**：`T = α + bytes / β`

- `α` = 固定延迟
- `β` = 渐近带宽

拟合出的这两个数比肉眼看曲线更有说服力。

### 平台期数值参考

| 接口 | 理论 | 实测（pinned） |
|---|---|---|
| PCIe 3.0 x16 | ~16 GB/s | 11~12 GB/s |
| PCIe 4.0 x16 | ~32 GB/s | ~25 GB/s |

**⚠️ 这份代码测的是 pageable 内存**（输出文件名 `pageable.csv` 也在暗示），通常只有 pinned 的**一半左右**。

**原因**：pageable 内存可能被 OS 换出，DMA 引擎不敢直接读它。驱动必须先把数据拷到自己内部的一块 pinned staging buffer 再发起 DMA——**多了一次 CPU 拷贝**。

> 📌 下一步作业：学 `cudaHostAlloc` / `cudaMallocHost` 后用同一程序测 pinned 内存，两条曲线叠一起看，差异非常直观。

**D2H 通常比 H2D 略慢**，这是正常的，不是代码写错了。

---

## 10. 几个容易忽略的坑

### 理论带宽那行算的不是 PCIe

```cpp
2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6
```

这算的是**显存**（GDDR/HBM）带宽，几百到上千 GB/s，跟正在测的 PCIe 带宽差**一到两个数量级**。作为参考信息打出来没问题，但别拿测出的十几 GB/s 去跟它比而怀疑人生。

> `memoryClockRate` 在 CUDA 12 已被标记 deprecated，新代码建议用 `cudaDeviceGetAttribute` + `cudaDevAttrMemoryClockRate`。

### `CUDA_CHECK` 在计时循环里安全吗

它每次读返回值、做比较，开销纳秒级，相对微秒级的拷贝可忽略。

**但要警惕**：有些人的检查宏里会混进 `cudaDeviceSynchronize()`，那就是灾难——会把每次拷贝都变成同步的。**这份代码没这个问题。**

### Day 4 的 TODO：换成 cudaEvent

| | `chrono` | `cudaEvent` |
|---|---|---|
| 测的是 | **CPU 侧**时间，含 API 调用开销和 CPU 抖动 | **GPU 侧**实际执行时间，插在流里的时间戳 |
| 精度 | 几十 ns 分辨率，但 `now()` 本身有 20~30 ns 开销 | 约 0.5 µs，不受 CPU 抖动影响 |
| 优势场景 | 通用 | 异步操作、kernel 计时 |

对这个特定场景（同步拷贝 + 循环外同步 + 大量重复），`chrono` 其实够用，两者结果不会差太多。**那次重写更多是练手，不是修 bug。**

---

## 附：知识点速查

| 概念 | 一句话 |
|---|---|
| `size_t` | 无符号整数类型，专表大小/字节数，64 位系统上 64 位 |
| `static`（文件作用域） | 内部链接，只在本编译单元可见，避免符号冲突 |
| `using A = B` | 类型别名，等价 typedef，新名在左 |
| `steady_clock` | 单调时钟，测时间间隔专用 |
| `duration<double>` | 以秒为单位、double 存储的时长类型 |
| `.count()` | 从 duration 取出裸数值，单位取决于外层 duration 类型 |
| `double(x)` | C++ 风格类型转换，等价 `(double)x` |
| `%zu` | printf 中 `size_t` 的正确占位符 |
| `fflush(stdout)` | 强制刷缓冲，重定向到文件时必须 |
| 输出参数 | 传地址让函数写回结果，CUDA API 的通用模式 |
| pageable vs pinned | pageable 需多一次 staging 拷贝，带宽约为 pinned 一半 |

cd ~/CUDA
python3 -m venv .venv
source .venv/bin/activate