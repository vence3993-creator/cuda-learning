# Day 5 — 显存层级与 Roofline

## 产出
| 文件 | 说明 |
|---|---|
| `peaks.cu` | 打印设备峰值参数，验证 benchmark.h 的推算 |
| `roofline_sweep.cu` | 算术强度可调的 FMA kernel，AI 从 1 扫到 512 |
| `plot_roofline.py` | 画 roofline.png 与 time_vs_k.png |
| `notes.md` | 理论笔记 + 实测结论 + 踩过的坑 |
| `roofline.csv` | 原始测量数据 |
| `*.log` | 运行日志、频率日志、ncu 输出 |

## 关键结论（RTX 5060 Ti）
- 实测带宽 385 GB/s（1读1写），vector add 394.9 GB/s（2读1写）
- 实测算力平台 24373 GFLOP/s = 校准峰值 26450 的 92%
- 实测平衡点 63 FLOP/B；曲线在 AI≈48 即开始偏离屋顶
- K=4→128 计算量涨 32 倍而耗时不变 = memory bound 的判据
- 满 occupancy 下 TLP 足以掩盖 FFMA 延迟，ILP 无额外收益

## 重要的坑
- `cudaDevAttrClockRate` 返回空闲频率，峰值算力低估 9%，须用 nvidia-smi 校准
- ncu `-c N` 是「前 N 次 launch」，不是「每个 kernel N 次」
- ncu 默认锁 base clock，须加 `--clock-control none`
- profiling 会污染 benchmark 的 median，两者分开跑

## 复现
```bash
make && ./roofline_sweep | tee run.log
python3 plot_roofline.py
```