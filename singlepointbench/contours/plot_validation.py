#!/usr/bin/env python3
"""
Publication-quality validation figure: LBM Enthalpy-Porosity vs OpenFOAM.
T = 1650 K (solidus) isotherm comparison at t = 25, 50, 60, 75 us.

Output: solidus_validation.png (300 dpi, journal-ready)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ============================================================================
# Style — journal-grade defaults
# ============================================================================
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# ============================================================================
# Data directory (script lives alongside the CSVs)
# ============================================================================
data_dir = Path(__file__).parent

# Snapshot times
times = ['25', '50', '60', '75']
labels_time = [r'$t = 25\;\mu\mathrm{s}$',
               r'$t = 50\;\mu\mathrm{s}$',
               r'$t = 60\;\mu\mathrm{s}$',
               r'$t = 75\;\mu\mathrm{s}$']

# ============================================================================
# Colour palette
# ============================================================================
COLOR_LBM = '#E6A125'       # warm gold
COLOR_OF  = '#4A4A4A'       # dark charcoal

# ============================================================================
# Figure: 2x2 subplots
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), sharex=True, sharey=True)
fig.suptitle('Validation results for melting process',
             fontsize=13, fontweight='bold', y=0.97)

for i, (t, label_t) in enumerate(zip(times, labels_time)):
    ax = axes[i // 2][i % 2]

    # --- Load data -----------------------------------------------------------
    of_file = data_dir / f'openfoam_contour_{t}us.csv'
    lbm_file = data_dir / f'lbm_contour_{t}us.csv'

    of_ok = of_file.exists()
    lbm_ok = lbm_file.exists()

    if of_ok:
        df_of = pd.read_csv(of_file)
        # Sort by angle from centroid for a clean closed contour
        cx_of = df_of['X_um'].mean()
        cz_of = df_of['Z_um'].mean()
        df_of['angle'] = np.arctan2(df_of['Z_um'] - cz_of,
                                     df_of['X_um'] - cx_of)
        df_of = df_of.sort_values('angle')

    if lbm_ok:
        df_lbm = pd.read_csv(lbm_file)
        cx_lbm = df_lbm['X_um'].mean()
        cz_lbm = df_lbm['Z_um'].mean()
        df_lbm['angle'] = np.arctan2(df_lbm['Z_um'] - cz_lbm,
                                      df_lbm['X_um'] - cx_lbm)
        df_lbm = df_lbm.sort_values('angle')

    # --- Plot ----------------------------------------------------------------
    if of_ok:
        ax.plot(df_of['X_um'], df_of['Z_um'],
                color=COLOR_OF, linewidth=2.0, linestyle='-',
                label='Results from OpenFOAM', zorder=2)
    if lbm_ok:
        ax.plot(df_lbm['X_um'], df_lbm['Z_um'],
                color=COLOR_LBM, linewidth=1.3, linestyle='-',
                marker='o', markersize=3, markeredgewidth=0.3,
                markeredgecolor='#B07A10', markerfacecolor=COLOR_LBM,
                label='Results from my algorithm', zorder=3)

    # --- Decorations ---------------------------------------------------------
    ax.set_xlim(60, 140)
    ax.set_ylim(110, 155)
    # Surface at top (Z increases upward — already correct since higher Z = closer to surface)

    ax.set_title(label_t, pad=4)
    ax.grid(True, linewidth=0.3, alpha=0.5, color='#CCCCCC')

    # Axis labels only on edges
    if i // 2 == 1:
        ax.set_xlabel(r'X ($\mu$m)')
    if i % 2 == 0:
        ax.set_ylabel(r'Depth Z ($\mu$m)')

    # Shade the melt pool interior (optional subtle fill)
    if lbm_ok:
        from matplotlib.patches import Polygon
        pts = np.column_stack([df_lbm['X_um'].values,
                               df_lbm['Z_um'].values])
        poly = Polygon(pts, closed=True, facecolor=COLOR_LBM,
                       alpha=0.08, edgecolor='none', zorder=1)
        ax.add_patch(poly)

# ============================================================================
# Legend — single instance in top-right subplot
# ============================================================================
handles, labels = axes[0][1].get_legend_handles_labels()
if handles:
    axes[0][1].legend(handles, labels, loc='lower right',
                      frameon=True, framealpha=0.9, edgecolor='#CCCCCC',
                      borderpad=0.4, handlelength=2.0)

fig.align_ylabels(axes[:, 0])
plt.tight_layout(rect=[0, 0, 1, 0.95])

out_path = data_dir / 'solidus_validation.png'
fig.savefig(out_path)
print(f'Saved: {out_path}')

# Also save PDF for journal submission
fig.savefig(data_dir / 'solidus_validation.pdf')
print(f'Saved: {data_dir / "solidus_validation.pdf"}')

plt.show()
