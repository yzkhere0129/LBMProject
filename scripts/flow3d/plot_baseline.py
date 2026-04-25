#!/usr/bin/env python3
"""Plot Sprint-1 LBM-vs-Flow3D baseline time series from compare.py CSV."""
import sys, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

csv_path = sys.argv[1] if len(sys.argv) > 1 else 'docs/reference/sprint1_baseline_rt_timeseries.csv'
out_png = sys.argv[2] if len(sys.argv) > 2 else 'docs/reference/sprint1_baseline.png'

t, hausm, chamm, T_rms, T_p95 = [], [], [], [], []
LF, WF, DF = [], [], []
LL, WL, DL = [], [], []
TmF, TmL, vL = [], [], []

with open(csv_path) as f:
    r = csv.DictReader(f)
    for row in r:
        # parse t from label like rt_50us
        try:
            label = row['label']
            t_us = float(label.split('_')[-1].rstrip('us'))
        except Exception:
            t_us = float('nan')
        t.append(t_us)
        hausm.append(float(row['haus_um']))
        chamm.append(float(row['chamfer_um']))
        T_rms.append(float(row['surface_T_rms_K']))
        T_p95.append(float(row['surface_T_p95_K']))
        LF.append(float(row['pool_L_F3D_um']))
        WF.append(float(row['pool_W_F3D_um']))
        DF.append(float(row['pool_D_F3D_um']))
        LL.append(float(row['pool_L_LBM_um']))
        WL.append(float(row['pool_W_LBM_um']))
        DL.append(float(row['pool_D_LBM_um']))
        TmF.append(float(row['Tmax_F3D_K']))
        TmL.append(float(row['Tmax_LBM_K']))
        vL.append(float(row['vmax_LBM_m_s']))

t = np.array(t)
order = np.argsort(t)
def srt(x): return np.array(x)[order]
t = srt(t); hausm = srt(hausm); chamm = srt(chamm); T_rms = srt(T_rms); T_p95 = srt(T_p95)
LF=srt(LF); WF=srt(WF); DF=srt(DF); LL=srt(LL); WL=srt(WL); DL=srt(DL)
TmF=srt(TmF); TmL=srt(TmL); vL=srt(vL)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Sprint-1 baseline: LBM vs Flow3D 316L 150W 0.8m/s (Δt cadence ≈ 50 μs)', fontsize=14)

ax = axes[0,0]
ax.plot(t, DF, 'k-o', label='Flow3D D', lw=2)
ax.plot(t, DL, 'r-s', label='LBM D', lw=2)
ax.set(title='Pool depth D [μm]', xlabel='t [μs]', ylabel='D [μm]')
ax.grid(); ax.legend()

ax = axes[0,1]
ax.plot(t, LF, 'k-o', label='Flow3D L', lw=2)
ax.plot(t, LL, 'r-s', label='LBM L', lw=2)
ax.set(title='Pool length L [μm]', xlabel='t [μs]', ylabel='L [μm]')
ax.grid(); ax.legend()

ax = axes[0,2]
ax.plot(t, WF, 'k-o', label='Flow3D W', lw=2)
ax.plot(t, WL, 'r-s', label='LBM W', lw=2)
ax.set(title='Pool width W [μm]  (LBM systematically too wide)', xlabel='t [μs]', ylabel='W [μm]')
ax.grid(); ax.legend()

ax = axes[1,0]
ax.plot(t, TmF, 'k-o', label='Flow3D Tmax', lw=2)
ax.plot(t, TmL, 'r-s', label='LBM Tmax', lw=2)
ax.axhline(1697.15, color='gray', ls='--', label='T_liq')
ax.axhline(3090, color='gray', ls=':', label='T_boil')
ax.set(title='T_max [K]', xlabel='t [μs]', ylabel='T_max [K]')
ax.grid(); ax.legend()

ax = axes[1,1]
ax.plot(t, hausm, 'b-o', label='Hausdorff (cropped)', lw=2)
ax.plot(t, chamm, 'g-s', label='Mean chamfer', lw=2)
ax.set(title='Surface distance [μm]', xlabel='t [μs]', ylabel='dist [μm]', yscale='log')
ax.grid(); ax.legend()

ax = axes[1,2]
ax.plot(t, T_rms, 'b-o', label='RMS', lw=2)
ax.plot(t, T_p95, 'g-s', label='p95', lw=2)
ax.set(title='Surface ΔT (LBM-F3D) [K]', xlabel='t [μs]', ylabel='|ΔT| [K]')
ax.grid(); ax.legend()

plt.tight_layout()
plt.savefig(out_png, dpi=110, bbox_inches='tight')
print(f'Saved {out_png}')
