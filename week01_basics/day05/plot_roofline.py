#!/usr/bin/env python3
# plot_roofline.py -- RTX 5060 Ti roofline, Day 5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- hardware params: all from Day5 measurement / calibration ----
BW_THEORY   = 448.0     # spec: 28 Gbps x 128 bit / 8
BW_MEASURED = 385.3     # fma_sweep low-K plateau, 1 read + 1 write
BW_VECADD   = 394.9     # Day4 vector add, 2 reads + 1 write
GF_NAIVE    = 24256.5   # from cudaDevAttrClockRate = 2632 MHz (idle clock, underestimates)
GF_CORRECT  = 26450.0   # nvidia-smi sustained 2.87 GHz under load
GF_MEASURED = 24373.0   # fma_sweep K=2048 measured plateau
NBYTES_GB   = 0.5369    # 2 * 2^26 * 4 B
N           = 1 << 26

df = pd.read_csv('roofline.csv')
d4 = df[df.one_chain == 0].sort_values('AI')
d1 = df[df.one_chain == 1].sort_values('AI')

# ===========================================================================
# Figure 1: roofline
# ===========================================================================
ai = np.logspace(-1.5, 3, 400)
fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(ai, np.minimum(GF_CORRECT, ai * BW_THEORY), color='#333', lw=2.0,
        label=f'Roof, calibrated  ({BW_THEORY:.0f} GB/s, {GF_CORRECT/1000:.1f} TFLOP/s)')
ax.plot(ai, np.minimum(GF_MEASURED, ai * BW_MEASURED), color='#D85A30', lw=2.0, ls='--',
        label=f'Roof, measured  ({BW_MEASURED:.0f} GB/s, {GF_MEASURED/1000:.1f} TFLOP/s)')
ax.plot(ai, np.minimum(GF_NAIVE, ai * BW_THEORY), color='#999', lw=1.2, ls=':',
        label=f'Roof, uncalibrated clock  ({GF_NAIVE/1000:.1f} TFLOP/s, 9% too low)')

ax.plot(d4.AI, d4.gflops, 'o-', color='#185FA5', ms=7, lw=1.5,
        label='fma_sweep, 4 chains')
ax.plot(d1.AI, d1.gflops, 's', color='#1D9E75', ms=9, mfc='none', mew=1.8,
        label='fma_sweep, 1 chain')

va_ai, va_gf = 1/12, BW_VECADD/12
ax.plot([va_ai], [va_gf], '*', color='#D4537E', ms=20, label='vector add (Day 4)')
ax.annotate(f'vector add\nAI = 0.083\n{va_gf:.0f} GFLOP/s = {BW_VECADD/BW_THEORY*100:.0f}% of its roof',
            xy=(va_ai, va_gf), xytext=(0.14, 3.5), fontsize=9, color='#993556',
            arrowprops=dict(arrowstyle='->', color='#D4537E', lw=1.2))

for x, c, lab in [(GF_CORRECT/BW_THEORY,    '#333',    'ridge 59'),
                  (GF_MEASURED/BW_MEASURED, '#D85A30', 'ridge 63')]:
    ax.axvline(x, color=c, ls=':', lw=1, alpha=0.55)
    ax.text(x, 12, lab, rotation=90, fontsize=8, color=c, ha='right', va='bottom')

ax.axvspan(48, 256, color='#EF9F27', alpha=0.10)
ax.text(110, 550, 'rounded knee\nAI 48-256', fontsize=9, color='#854F0B',
        ha='center', va='center')
ax.text(3, 25, 'memory bound', fontsize=11, color='#666', ha='center')
ax.text(400, 25, 'compute bound', fontsize=11, color='#666', ha='center')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(0.05, 1000); ax.set_ylim(10, 60000)
ax.set_xlabel('Arithmetic intensity (FLOP/Byte)', fontsize=11)
ax.set_ylabel('Performance (GFLOP/s)', fontsize=11)
ax.set_title('RTX 5060 Ti roofline  -  36 SM, 128-bit GDDR7, 32 MB L2', fontsize=12)
ax.grid(True, which='both', alpha=0.2)
ax.legend(loc='lower right', fontsize=9, framealpha=0.95)

fig.tight_layout()
fig.savefig('roofline.png', dpi=150)

# ===========================================================================
# Figure 2: runtime vs K (linear y, the more intuitive view)
# ===========================================================================
fig2, ax2 = plt.subplots(figsize=(9, 5.5))

t_mem  = NBYTES_GB / BW_MEASURED * 1000.0
k      = np.array(d4.K, dtype=float)
t_comp = 2.0 * k * N / GF_CORRECT / 1e6

ax2.plot(d4.K, d4.median_ms, 'o-', color='#185FA5', ms=7, lw=1.8, label='measured median')
ax2.axhline(t_mem, color='#D85A30', ls='--', lw=1.5,
            label=f'memory-only time = {t_mem:.2f} ms (constant)')
ax2.plot(k, t_comp, ls=':', color='#555', lw=1.5, label='compute-only time')
ax2.plot(k, np.maximum(t_mem, t_comp), ls='-', color='#999', lw=1.0, alpha=0.8,
         label='max() model')

ax2.annotate('32x more FLOPs, same runtime', xy=(20, t_mem), xytext=(6, 4.2),
             fontsize=9, color='#993556',
             arrowprops=dict(arrowstyle='->', color='#D4537E', lw=1.2))
ax2.annotate('knee', xy=(224, 1.75), xytext=(300, 3.2), fontsize=9, color='#854F0B',
             arrowprops=dict(arrowstyle='->', color='#BA7517', lw=1.2))

ax2.set_xscale('log', base=2)
ax2.set_xticks(list(d4.K)); ax2.set_xticklabels([str(int(v)) for v in d4.K], fontsize=8)
ax2.set_xlabel('K  (FMAs per element;  AI = K/4)', fontsize=11)
ax2.set_ylabel('Kernel time (ms)', fontsize=11)
ax2.set_title('Runtime vs arithmetic intensity  -  the flat segment is memory bound', fontsize=12)
ax2.grid(alpha=0.2); ax2.legend(fontsize=9)

fig2.tight_layout()
fig2.savefig('time_vs_k.png', dpi=150)
print('-> roofline.png, time_vs_k.png')