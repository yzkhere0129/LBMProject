#!/usr/bin/env python3
"""
Plot Stefan problem results: LBM (ESM) vs Neumann analytical solution.

Reads stefan_pure_metal.csv from the build directory.
Produces stefan_pure_metal.png with:
  - Top panel: Temperature T(x) profiles
  - Bottom panel: Liquid fraction f_l(x) profiles
  - Dashed lines: Neumann analytical solution
  - Solid lines: LBM numerical results
  - Inset text: front position error at each snapshot
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import sys
import os

def load_csv(path):
    """Load stefan_pure_metal.csv, parse header and data."""
    meta = {}
    data_lines = []

    with open(path, 'r') as f:
        for line in f:
            if line.startswith('# Ste='):
                parts = line.strip('# \n').split()
                for p in parts:
                    k, v = p.split('=')
                    meta[k] = float(v) if k != 'NX' else int(float(v))
            elif line.startswith('#'):
                continue
            else:
                data_lines.append(line.strip())

    # Parse CSV data
    rows = []
    for line in data_lines:
        if not line:
            continue
        vals = line.split(',')
        rows.append([int(vals[0])] + [float(v) for v in vals[1:]])

    arr = np.array(rows)
    return meta, arr


def main():
    # Find CSV file
    csv_path = None
    for candidate in ['stefan_pure_metal.csv',
                       'build/stefan_pure_metal.csv',
                       '../build/stefan_pure_metal.csv']:
        if os.path.exists(candidate):
            csv_path = candidate
            break

    if csv_path is None:
        print("ERROR: stefan_pure_metal.csv not found.")
        print("Run the stefan_benchmark executable first.")
        sys.exit(1)

    print(f"Loading {csv_path}...")
    meta, arr = load_csv(csv_path)

    Ste = meta['Ste']
    lam = meta['lambda']
    alpha = meta['alpha']
    dx = meta['dx']
    NX = meta['NX']

    # Snapshot times (must match the C++ code)
    t_snap = np.array([0.2e-3, 0.5e-3, 1.0e-3, 1.5e-3, 2.0e-3])
    snap_ids = np.unique(arr[:, 0].astype(int))
    n_snap = len(snap_ids)

    # Colors
    cmap = plt.cm.viridis
    colors = [cmap(i / max(n_snap - 1, 1)) for i in range(n_snap)]

    fig, (ax_T, ax_fl) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('Stefan Problem: Pure Metal (Ste = %.2f)' % Ste, fontsize=14, fontweight='bold')

    front_errors = []

    for i, sid in enumerate(snap_ids):
        mask = arr[:, 0] == sid
        x_m  = arr[mask, 1]
        T_lbm = arr[mask, 2]
        fl_lbm = arr[mask, 3]
        T_anal = arr[mask, 4]
        fl_anal = arr[mask, 5]

        t_ms = t_snap[i] * 1e3 if i < len(t_snap) else 0.0
        x_um = x_m * 1e6

        # Analytical front position
        s_anal = 2.0 * lam * np.sqrt(alpha * t_snap[i]) if i < len(t_snap) else 0.0

        # Numerical front position (fl = 0.5 crossing)
        s_num = 0.0
        for j in range(1, len(fl_lbm)):
            if fl_lbm[j - 1] >= 0.5 and fl_lbm[j] < 0.5:
                x0 = x_m[j - 1]
                s_num = x0 + (0.5 - fl_lbm[j - 1]) / (fl_lbm[j] - fl_lbm[j - 1]) * dx
                break

        err_pct = abs(s_num - s_anal) / s_anal * 100 if s_anal > 0 else -1
        front_errors.append((t_ms, s_anal * 1e6, s_num * 1e6, err_pct))

        label_lbm = f't={t_ms:.1f} ms'
        label_anal = f'Neumann t={t_ms:.1f} ms'

        # Temperature
        ax_T.plot(x_um, T_lbm, '-', color=colors[i], linewidth=1.5, label=label_lbm)
        ax_T.plot(x_um, T_anal, '--', color=colors[i], linewidth=1.0, alpha=0.7)

        # Liquid fraction
        ax_fl.plot(x_um, fl_lbm, '-', color=colors[i], linewidth=1.5, label=label_lbm)
        ax_fl.plot(x_um, fl_anal, '--', color=colors[i], linewidth=1.0, alpha=0.7)

        # Mark front positions
        ax_fl.axvline(s_anal * 1e6, color=colors[i], linestyle=':', alpha=0.4, linewidth=0.8)

    # Formatting
    ax_T.set_ylabel('Temperature [K]')
    ax_T.set_ylim(990, 1110)
    ax_T.axhline(1000.0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax_T.axhline(1000.1, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax_T.legend(loc='upper right', fontsize=8, ncol=2)
    ax_T.set_title('Temperature profile: solid lines = LBM (ESM), dashed = Neumann analytical')
    ax_T.grid(True, alpha=0.3)

    ax_fl.set_xlabel('Position [µm]')
    ax_fl.set_ylabel('Liquid fraction $f_l$')
    ax_fl.set_ylim(-0.05, 1.1)
    ax_fl.legend(loc='upper right', fontsize=8, ncol=2)
    ax_fl.set_title('Liquid fraction: solid lines = LBM (ESM), dashed = Neumann analytical')
    ax_fl.grid(True, alpha=0.3)

    # Limit x-axis to region of interest
    max_front = max(e[1] for e in front_errors) if front_errors else 200
    ax_fl.set_xlim(0, min(max_front * 3, NX * dx * 1e6))

    # Add error table as text
    text_lines = ['Front position errors:']
    text_lines.append(f'{"t [ms]":>8}  {"s_anal":>8}  {"s_num":>8}  {"err":>6}')
    for t_ms, sa, sn, err in front_errors:
        text_lines.append(f'{t_ms:8.1f}  {sa:7.1f}µm  {sn:7.1f}µm  {err:5.2f}%')

    # Compute average error
    valid_errors = [e[3] for e in front_errors if e[3] >= 0]
    if valid_errors:
        avg_err = np.mean(valid_errors)
        max_err = np.max(valid_errors)
        text_lines.append(f'\nAvg err: {avg_err:.2f}%  Max err: {max_err:.2f}%')
        pass_fail = 'PASS' if max_err < 5.0 else 'FAIL'
        text_lines.append(f'Criterion: max < 5%  →  {pass_fail}')

    ax_fl.text(0.02, 0.35, '\n'.join(text_lines), transform=ax_fl.transAxes,
               fontsize=7, fontfamily='monospace', verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    out_path = 'stefan_pure_metal.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved {out_path}')

    # Print summary
    print('\n=== Front Position Validation ===')
    print(f'{"t [ms]":>8}  {"s_anal [µm]":>12}  {"s_num [µm]":>12}  {"error [%]":>10}')
    for t_ms, sa, sn, err in front_errors:
        print(f'{t_ms:8.1f}  {sa:12.1f}  {sn:12.1f}  {err:10.2f}')

    if valid_errors:
        print(f'\nMax error: {max_err:.2f}%')
        if max_err < 5.0:
            print('PASS: Front position matches Neumann solution within 5%')
        else:
            print('FAIL: Front position error exceeds 5% threshold')

    plt.show()


if __name__ == '__main__':
    main()
