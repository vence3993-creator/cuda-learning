#!/usr/bin/env python3
"""
plot_bw.py — 画传输带宽曲线并拟合固定开销 t0

用法:
    python3 plot_bw.py pageable.csv
    python3 plot_bw.py pageable.csv pinned.csv     # 时段 4 用这个

输出: bandwidth.png
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # WSL2 无显示环境，用非交互后端
import matplotlib.pyplot as plt


def analyze(df, label):
    """打印峰值、饱和点，并用小 size 段拟合固定开销 t0。"""
    print(f"\n===== {label} =====")

    for direction in ("h2d", "d2h"):
        col = f"{direction}_GBps"
        gbps = df[col].values
        nbytes = df["bytes"].values.astype(float)

        peak = gbps.max()
        # 达到峰值 90% 的最小 size，即"饱和点"
        idx = np.argmax(gbps >= 0.9 * peak)
        knee = nbytes[idx]

        # t = t0 + N/B  ->  对 (N, t) 做线性回归
        # 只用 <=1MB 的点：大 size 点方差杠杆太高，会把截距压歪
        sec = nbytes / (gbps * 1e9)
        mask = nbytes <= 1 << 20
        slope, intercept = np.polyfit(nbytes[mask], sec[mask], 1)
        t0_us = intercept * 1e6

        # N/B == t0 时的 size：一半时间花在固定开销上的分界点
        breakeven = t0_us * 1e-6 * peak * 1e9 if peak > 0 else float("nan")

        print(f"{direction.upper():4s} 峰值      : {peak:6.2f} GB/s")
        print(f"     饱和点(90%): {knee/1024:8.0f} KB")
        print(f"     固定开销 t0: {t0_us:6.2f} us")
        print(f"     开销分界点 : {breakeven/1024:8.0f} KB  (小于此值,过半时间是纯开销)")


def main():
    files = sys.argv[1:] or ["pageable.csv"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["tab:blue", "tab:red", "tab:green"]

    for i, path in enumerate(files):
        if not os.path.exists(path):
            print(f"跳过不存在的文件: {path}")
            continue
        df = pd.read_csv(path)
        tag = os.path.splitext(os.path.basename(path))[0]
        c = colors[i % len(colors)]

        ax.semilogx(df.bytes, df.h2d_GBps, "o-", color=c,
                    label=f"{tag} H2D")
        ax.semilogx(df.bytes, df.d2h_GBps, "s--", color=c, alpha=0.6,
                    label=f"{tag} D2H")

        analyze(df, tag)

    # 标出理论上限：本机 PCIe Gen4 x8 = 8 lane * 1.97 GB/s
    ax.axhline(15.8, color="gray", ls=":", lw=1.2)
    ax.text(1500, 16.1, "PCIe Gen4 x8 理论上限 15.8 GB/s",
            fontsize=9, color="gray")

    ax.set_xlabel("transfer size (bytes)")
    ax.set_ylabel("bandwidth (GB/s)")
    ax.set_title("Host <-> Device transfer bandwidth")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("bandwidth.png", dpi=140)
    print("\n已保存 bandwidth.png")


if __name__ == "__main__":
    main()