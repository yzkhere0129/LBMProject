#!/usr/bin/env python3
"""
Visualization for Stefan problem benchmark results.

Usage:
    cd build
    ./tests/validation/test_stefan_benchmark --gtest_filter="*SmokeTest*:*FrontPosition*:*MushyZone*"
    python3 ../scripts/plot_stefan_benchmark.py

Generates:
    output_stefan_benchmark/stefan_temperature_profile.png
    output_stefan_benchmark/stefan_mushy_convergence.png
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

OUTPUT_DIR = "output_stefan_benchmark"


def plot_temperature_profiles():
    """Plot temperature & liquid fraction profiles vs Neumann analytical solution."""
    csvs = sorted(glob.glob(f"{OUTPUT_DIR}/profile_*.csv"))
    if not csvs:
        print("No profile CSVs found. Run the test with verbose output first.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]})

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(csvs)))

    for csv_path, color in zip(csvs, colors):
        data = np.genfromtxt(csv_path, delimiter=',', names=True)
        label_parts = os.path.basename(csv_path).replace('.csv', '').split('_')
        nx_str = [p for p in label_parts if p.startswith('NX')][0]
        dt_str = [p for p in label_parts if p.startswith('dT')][0]
        label = f"{nx_str}, {dt_str}K"

        x = data['x_um']
        T_num = data['T_numerical']
        T_ana = data['T_analytical']
        fl = data['liquid_fraction']

        # Temperature
        axes[0].plot(x, T_num, '-', color=color, linewidth=1.5, label=f'LBM ({label})')
        axes[0].plot(x, T_ana, '--', color=color, linewidth=1.0, alpha=0.7)

        # Liquid fraction
        axes[1].plot(x, fl, '-', color=color, linewidth=1.5, label=label)

    # Analytical reference line (from first CSV)
    data0 = np.genfromtxt(csvs[0], delimiter=',', names=True)
    axes[0].plot([], [], 'k--', linewidth=1.0, label='Neumann analytical')

    # Temperature axis
    axes[0].set_ylabel('Temperature [K]', fontsize=12)
    axes[0].set_ylim(995, 1055)
    axes[0].legend(loc='upper right', fontsize=9, framealpha=0.9)
    axes[0].xaxis.set_minor_locator(AutoMinorLocator())
    axes[0].yaxis.set_minor_locator(AutoMinorLocator())
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Stefan Problem — Enthalpy LBM vs Neumann Analytical', fontsize=13)

    # Liquid fraction axis
    axes[1].set_xlabel('Position [μm]', fontsize=12)
    axes[1].set_ylabel('Liquid Fraction', fontsize=12)
    axes[1].set_ylim(-0.05, 1.15)
    axes[1].axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='fl = 0.5 (front)')
    axes[1].legend(loc='upper right', fontsize=9, framealpha=0.9)
    axes[1].xaxis.set_minor_locator(AutoMinorLocator())
    axes[1].grid(True, alpha=0.3)

    # Zoom to interesting region
    front_x = max(data0['x_um'][data0['liquid_fraction'] > 0.01])
    axes[0].set_xlim(0, min(front_x * 2, max(data0['x_um'])))

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/stefan_temperature_profile.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_mushy_convergence():
    """Plot mushy zone width convergence (error vs dT_melt)."""
    csv_path = f"{OUTPUT_DIR}/mushy_convergence.csv"
    if not os.path.exists(csv_path):
        print("No mushy_convergence.csv found. Run MushyZoneConvergence test first.")
        return

    data = np.genfromtxt(csv_path, delimiter=',', names=True)
    dT = data['dT_melt']
    err_std = data['standard_error_pct']
    err_cor = data['corrected_error_pct']

    fig, ax = plt.subplots(figsize=(8, 6))

    # Data
    ax.loglog(dT, err_std, 'o-', color='#2196F3', linewidth=2, markersize=8,
              label='vs Standard Neumann', zorder=5)
    ax.loglog(dT, err_cor, 's--', color='#FF9800', linewidth=2, markersize=7,
              label='vs Corrected (St_eff)', zorder=5)

    # Reference slope: linear in dT_melt
    dT_ref = np.array([dT.min(), dT.max()])
    slope = err_std[-2] / dT[-2]  # Use second-to-last point for fit
    ax.loglog(dT_ref, slope * dT_ref, ':', color='gray', linewidth=1,
              label=f'O(ΔT_melt) reference')

    # 0.1% threshold
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(dT.max() * 0.6, 0.12, '0.1% target', color='red', fontsize=10, alpha=0.7)

    ax.set_xlabel('Mushy Zone Width ΔT_melt [K]', fontsize=12)
    ax.set_ylabel('Front Position Error [%]', fontsize=12)
    ax.set_title('Stefan Benchmark — Convergence to Sharp-Interface Limit', fontsize=13)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(dT.min() * 0.5, dT.max() * 2)

    # Annotate key points
    for i in [0, len(dT) - 1]:
        ax.annotate(f'{err_std[i]:.2f}%',
                    xy=(dT[i], err_std[i]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, color='#2196F3')

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/stefan_mushy_convergence.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_front_position_detail():
    """Plot zoomed view near the melting front for the precise test."""
    csvs = sorted(glob.glob(f"{OUTPUT_DIR}/profile_NX400_*.csv"))
    if not csvs:
        csvs = sorted(glob.glob(f"{OUTPUT_DIR}/profile_*.csv"))
    if not csvs:
        return

    csv_path = csvs[-1]  # Use finest grid
    data = np.genfromtxt(csv_path, delimiter=',', names=True)

    x = data['x_um']
    T_num = data['T_numerical']
    T_ana = data['T_analytical']
    fl = data['liquid_fraction']

    # Find front region
    front_cells = np.where((fl > 0.01) & (fl < 0.99))[0]
    if len(front_cells) == 0:
        front_idx = np.argmin(np.abs(fl - 0.5))
        front_cells = [max(0, front_idx - 5), min(len(fl) - 1, front_idx + 5)]
    front_x = x[front_cells[len(front_cells) // 2]]

    # Zoom window: ±30% of front position around front
    x_min = max(0, front_x * 0.7)
    x_max = front_x * 1.3

    mask = (x >= x_min) & (x <= x_max)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})

    # Temperature comparison
    ax1.plot(x[mask], T_num[mask], 'o-', color='#2196F3', markersize=3,
             linewidth=1.5, label='LBM (enthalpy method)')
    ax1.plot(x[mask], T_ana[mask], '-', color='#E91E63', linewidth=2,
             label='Neumann analytical')

    # Mark the melting point
    ax1.axhline(y=1000, color='gray', linestyle=':', alpha=0.5)
    ax1.text(x_min + 5, 1000.5, 'T_melt = 1000 K', fontsize=9, color='gray')

    ax1.set_ylabel('Temperature [K]', fontsize=12)
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())

    label_parts = os.path.basename(csv_path).replace('.csv', '').split('_')
    nx_str = [p for p in label_parts if p.startswith('NX')][0]
    dt_str = [p for p in label_parts if p.startswith('dT')][0]
    ax1.set_title(f'Stefan Benchmark — Front Region Detail ({nx_str}, ΔT_melt={dt_str}K)',
                  fontsize=13)

    # Liquid fraction
    ax2.plot(x[mask], fl[mask], 's-', color='#4CAF50', markersize=3, linewidth=1.5)
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    # Mark analytical front
    # Front at fl=0.5, interpolate
    for i in range(1, len(fl)):
        if fl[i - 1] >= 0.5 and fl[i] < 0.5:
            frac = (0.5 - fl[i - 1]) / (fl[i] - fl[i - 1])
            front_pos = (x[i - 1] + frac * (x[i] - x[i - 1]))
            ax2.axvline(x=front_pos, color='#2196F3', linestyle='--', alpha=0.7,
                        label=f'Numerical front: {front_pos:.1f} μm')
            break

    # Analytical front
    ana_front = x[np.argmin(np.abs(T_ana - 1000))]
    ax2.axvline(x=ana_front, color='#E91E63', linestyle='--', alpha=0.7,
                label=f'Analytical front: {ana_front:.1f} μm')

    ax2.set_xlabel('Position [μm]', fontsize=12)
    ax2.set_ylabel('Liquid Fraction', fontsize=12)
    ax2.set_ylim(-0.05, 1.15)
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/stefan_front_detail.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    if not os.path.isdir(OUTPUT_DIR):
        print(f"Directory '{OUTPUT_DIR}' not found.")
        print("Run from the build directory after executing the Stefan benchmark test:")
        print("  ./tests/validation/test_stefan_benchmark "
              '--gtest_filter="*SmokeTest*:*FrontPosition*:*MushyZone*"')
        exit(1)

    plot_temperature_profiles()
    plot_mushy_convergence()
    plot_front_position_detail()
    print("\nDone. All plots saved to output_stefan_benchmark/")
