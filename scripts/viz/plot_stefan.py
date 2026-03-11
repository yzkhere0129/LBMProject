#!/usr/bin/env python3
"""
Plot Stefan problem visualization: temperature profiles + liquid fraction
at multiple time snapshots, with Neumann analytical solution overlay.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Publication-quality settings
mpl.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.5,
})

# Load data
df = pd.read_csv('/home/yzk/LBMProject/scripts/viz/stefan_data.csv')

snapshots = sorted(df['snapshot'].unique())
n_snap = len(snapshots)

# Color palette
colors = plt.cm.viridis(np.linspace(0.15, 0.9, n_snap))

fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                          gridspec_kw={'hspace': 0.08})

ax_T = axes[0]
ax_fl = axes[1]

# --- Top panel: Temperature profiles ---
for i, snap in enumerate(snapshots):
    sub = df[df['snapshot'] == snap]
    t_ms = sub['time_ms'].iloc[0]
    x = sub['x_um'].values
    T = sub['T_K'].values
    T_ana = sub['T_analytical'].values
    front_um = sub['front_analytical_um'].iloc[0]

    # Numerical (solid line)
    ax_T.plot(x, T, color=colors[i], linestyle='-',
              label=f'LBM, t={t_ms:.1f} ms')
    # Analytical (dashed line, same color)
    ax_T.plot(x, T_ana, color=colors[i], linestyle='--', alpha=0.7, linewidth=1.2)
    # Mark analytical front position
    ax_T.axvline(front_um, color=colors[i], linestyle=':', alpha=0.4, linewidth=0.8)

# Reference lines
T_solidus = df['T_K'].min()
T_liquidus = df['T_K'].max()
# Get actual T_solidus and T_liquidus from the data pattern
# T at x=0 should be T_liquidus, T at far end should be T_solidus
T_solidus_val = df[df['x_um'] == df['x_um'].max()]['T_K'].iloc[0]
T_liquidus_val = df[df['x_um'] == df['x_um'].min()]['T_K'].iloc[0]

ax_T.axhline(T_liquidus_val, color='red', linestyle='-', alpha=0.3, linewidth=0.8)
ax_T.axhline(T_solidus_val, color='blue', linestyle='-', alpha=0.3, linewidth=0.8)
ax_T.text(650, T_liquidus_val + 2, r'$T_{\rm liquidus}$', color='red', fontsize=9, alpha=0.6)
ax_T.text(650, T_solidus_val + 2, r'$T_{\rm solidus}$', color='blue', fontsize=9, alpha=0.6)

# Dummy lines for analytical legend
ax_T.plot([], [], 'k-', linewidth=1.5, label='LBM (numerical)')
ax_T.plot([], [], 'k--', linewidth=1.2, alpha=0.7, label='Neumann (analytical)')

ax_T.set_ylabel('Temperature [K]')
ax_T.set_title('Stefan Problem: Melting Front Propagation (Ti6Al4V)')
ax_T.legend(loc='upper right', ncol=2, framealpha=0.9)
ax_T.set_xlim(0, 800)
ax_T.grid(True, alpha=0.2)

# --- Bottom panel: Liquid fraction ---
for i, snap in enumerate(snapshots):
    sub = df[df['snapshot'] == snap]
    t_ms = sub['time_ms'].iloc[0]
    x = sub['x_um'].values
    fl = sub['fl'].values
    front_um = sub['front_analytical_um'].iloc[0]

    ax_fl.plot(x, fl, color=colors[i], linestyle='-',
               label=f't={t_ms:.1f} ms')
    ax_fl.axvline(front_um, color=colors[i], linestyle=':', alpha=0.4, linewidth=0.8)

# Add arrow showing melting direction
ax_fl.annotate('', xy=(350, 0.5), xytext=(100, 0.5),
               arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
ax_fl.text(230, 0.55, 'melting front\npropagation', ha='center', fontsize=9,
           color='gray', style='italic')

ax_fl.set_xlabel(r'Position [$\mu$m]')
ax_fl.set_ylabel('Liquid Fraction $f_l$')
ax_fl.set_xlim(0, 800)
ax_fl.set_ylim(-0.05, 1.1)
ax_fl.legend(loc='upper right', framealpha=0.9)
ax_fl.grid(True, alpha=0.2)

# Add mushy zone annotation on first snapshot
sub0 = df[df['snapshot'] == 0]
fl0 = sub0['fl'].values
x0 = sub0['x_um'].values
# Find mushy zone boundaries (0 < fl < 1)
mushy_mask = (fl0 > 0.01) & (fl0 < 0.99)
if mushy_mask.any():
    mushy_x = x0[mushy_mask]
    ax_fl.axvspan(mushy_x.min(), mushy_x.max(), alpha=0.1, color='orange')
    ax_fl.text((mushy_x.min() + mushy_x.max()) / 2, 0.15,
               'mushy\nzone', ha='center', fontsize=8, color='orange')

out = '/home/yzk/LBMProject/scripts/viz/stefan_front.png'
plt.savefig(out)
print(f'Saved: {out}')
plt.close()
