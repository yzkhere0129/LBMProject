#!/usr/bin/env python3
"""
Phase Change Module -- Comprehensive Validation Plots

Reads CSV outputs from comprehensive phase change tests and generates
publication-quality plots for each validation aspect.

Usage:
    cd build
    ./tests/validation/test_phase_change_comprehensive
    python3 ../scripts/plot_phase_change_validation.py

Generates (in output_phase_change_validation/):
    01_roundtrip_consistency.png
    02_liquid_fraction_all_materials.png
    03_energy_conservation.png
    04_apparent_heat_capacity.png
    05_newton_convergence.png
    06_enthalpy_change_verification.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# ---------------------------------------------------------------------------
# Paths & Style
# ---------------------------------------------------------------------------
OUTPUT_DIR = "output_phase_change_validation"

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

# Material database (mirrors material_database.cu) for annotations
MATERIALS = {
    'Ti6Al4V':   {'T_solidus': 1878.0, 'T_liquidus': 1923.0, 'color': '#2196F3'},
    'SS316L':    {'T_solidus': 1658.0, 'T_liquidus': 1700.0, 'color': '#FF9800'},
    'IN718':     {'T_solidus': 1533.0, 'T_liquidus': 1609.0, 'color': '#4CAF50'},
    'AlSi10Mg':  {'T_solidus':  833.0, 'T_liquidus':  873.0, 'color': '#9C27B0'},
    'Steel':     {'T_solidus': 1523.0, 'T_liquidus': 1723.0, 'color': '#F44336'},
}


def read_csv(path):
    """Read a CSV file, skipping comment lines starting with '#'.

    Returns (header_list, numpy_2d_array).  If the file has no data rows,
    returns (header_list, empty_array).
    """
    rows = []
    header = None
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if header is None:
                header = [h.strip() for h in line.split(',')]
                continue
            rows.append([float(x) for x in line.split(',')])
    if header is None:
        return None, np.empty((0, 0))
    return header, np.array(rows) if rows else (header, np.empty((0, len(header))))


# ============================================================
# Plot 1: H(T) and T(H) Roundtrip Consistency
# ============================================================
def plot_roundtrip():
    """Plot enthalpy-temperature roundtrip consistency.

    CSV columns: T_original, H_computed, T_recovered, error_K
    """
    csv_path = os.path.join(OUTPUT_DIR, "roundtrip_Ti6Al4V.csv")
    if not os.path.exists(csv_path):
        print(f"  SKIP: {csv_path} not found")
        return

    header, data = read_csv(csv_path)
    if data.size == 0:
        print(f"  SKIP: {csv_path} has no data rows")
        return

    T_orig = data[:, 0]
    H_comp = data[:, 1]
    T_recv = data[:, 2]
    err_K  = data[:, 3]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]})

    # --- Top subplot: H vs T curve ---
    ax = axes[0]
    ax.plot(T_orig, H_comp * 1e-9, '-', color='#2196F3', linewidth=2,
            label='H(T) computed')

    # Mark T_solidus and T_liquidus from the first material found in the data
    # Use Steel as default (widest mushy zone, most visible)
    for mat_name, props in MATERIALS.items():
        T_s, T_l = props['T_solidus'], props['T_liquidus']
        # Only annotate if within data range
        if T_orig.min() <= T_s <= T_orig.max():
            ax.axvline(T_s, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.text(T_s, ax.get_ylim()[0], f' $T_s$\n {T_s:.0f}K',
                    fontsize=8, color='gray', va='bottom')
        if T_orig.min() <= T_l <= T_orig.max():
            ax.axvline(T_l, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.text(T_l, ax.get_ylim()[0], f' $T_l$\n {T_l:.0f}K',
                    fontsize=8, color='gray', va='bottom')
        break  # Only annotate for the material actually tested

    ax.set_ylabel('Enthalpy $H$ (GJ/m$^3$)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)
    ax.set_title('Enthalpy-Temperature Roundtrip Consistency', fontsize=14)

    # Re-draw vlines after axis limits settle
    for mat_name, props in MATERIALS.items():
        T_s, T_l = props['T_solidus'], props['T_liquidus']
        if T_orig.min() <= T_s <= T_orig.max():
            ax.axvline(T_s, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        if T_orig.min() <= T_l <= T_orig.max():
            ax.axvline(T_l, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        break

    # --- Bottom subplot: roundtrip error ---
    ax = axes[1]
    ax.semilogy(T_orig, np.abs(err_K) + 1e-16, '-', color='#F44336', linewidth=1.5)
    ax.set_xlabel('Temperature $T$ (K)', fontsize=12)
    ax.set_ylabel('$|T_{recovered} - T_{original}|$ (K)', fontsize=12)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3, which='both')

    # Mark mushy zone on error plot too
    for mat_name, props in MATERIALS.items():
        T_s, T_l = props['T_solidus'], props['T_liquidus']
        if T_orig.min() <= T_s <= T_orig.max():
            ax.axvline(T_s, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        if T_orig.min() <= T_l <= T_orig.max():
            ax.axvline(T_l, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        break

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "01_roundtrip_consistency.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# ============================================================
# Plot 2: Liquid Fraction vs Temperature -- All Materials
# ============================================================
def plot_liquid_fraction():
    """Plot fl(T) curves for all 5 materials.

    CSV columns: material, T, fl
    Note: 'material' column is a string, so we parse specially.
    """
    csv_path = os.path.join(OUTPUT_DIR, "liquid_fraction_all_materials.csv")
    if not os.path.exists(csv_path):
        print(f"  SKIP: {csv_path} not found")
        return

    # Parse with material name as string
    materials_data = {}  # name -> (T_array, fl_array)
    header = None
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if header is None:
                header = [h.strip() for h in line.split(',')]
                continue
            parts = line.split(',')
            mat_name = parts[0].strip()
            T_val = float(parts[1])
            fl_val = float(parts[2])
            if mat_name not in materials_data:
                materials_data[mat_name] = ([], [])
            materials_data[mat_name][0].append(T_val)
            materials_data[mat_name][1].append(fl_val)

    if not materials_data:
        print(f"  SKIP: {csv_path} has no data rows")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for mat_name in materials_data:
        T_arr = np.array(materials_data[mat_name][0])
        fl_arr = np.array(materials_data[mat_name][1])

        # Determine color
        color = MATERIALS.get(mat_name, {}).get('color', '#888888')
        ax.plot(T_arr, fl_arr, '-', color=color, linewidth=2, label=mat_name)

        # Mark T_solidus and T_liquidus with small arrows
        props = MATERIALS.get(mat_name, {})
        T_s = props.get('T_solidus')
        T_l = props.get('T_liquidus')
        if T_s is not None:
            ax.annotate('', xy=(T_s, -0.05), xytext=(T_s, -0.12),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.2))
        if T_l is not None:
            ax.annotate('', xy=(T_l, 1.05), xytext=(T_l, 1.12),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

    ax.set_xlabel('Temperature $T$ (K)', fontsize=12)
    ax.set_ylabel('Liquid Fraction $f_l$', fontsize=12)
    ax.set_ylim(-0.15, 1.2)
    ax.set_title('Liquid Fraction vs Temperature -- All Materials', fontsize=14)
    ax.legend(loc='center right', fontsize=10, framealpha=0.9)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)
    ax.axhline(0.0, color='k', linewidth=0.4, alpha=0.3)
    ax.axhline(1.0, color='k', linewidth=0.4, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "02_liquid_fraction_all_materials.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# ============================================================
# Plot 3: Energy Conservation -- Adiabatic Phase Change
# ============================================================
def plot_energy_conservation():
    """Plot energy conservation during adiabatic phase change.

    CSV columns: step, total_energy, max_T, min_T, total_liquid_fraction
    """
    csv_path = os.path.join(OUTPUT_DIR, "adiabatic_energy.csv")
    if not os.path.exists(csv_path):
        print(f"  SKIP: {csv_path} not found")
        return

    header, data = read_csv(csv_path)
    if data.size == 0:
        print(f"  SKIP: {csv_path} has no data rows")
        return

    step   = data[:, 0]
    E_tot  = data[:, 1]
    T_max  = data[:, 2]
    T_min  = data[:, 3]
    fl_tot = data[:, 4]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1, 1]})

    # --- Top: total energy ---
    ax = axes[0]
    E_ref = E_tot[0] if E_tot[0] != 0 else 1.0
    E_rel = (E_tot - E_tot[0]) / np.abs(E_ref) * 100  # percent deviation
    ax.plot(step, E_rel, '-', color='#2196F3', linewidth=2)
    ax.set_ylabel('Energy Deviation (%)', fontsize=12)
    ax.axhline(0.0, color='k', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)
    ax.set_title('Energy Conservation -- Adiabatic Phase Change', fontsize=14)

    # Annotate max drift
    max_drift = np.max(np.abs(E_rel))
    ax.text(0.98, 0.92, f'Max drift: {max_drift:.2e}%',
            transform=ax.transAxes, fontsize=10, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # --- Middle: max/min temperature ---
    ax = axes[1]
    ax.plot(step, T_max, '-', color='#F44336', linewidth=1.5, label='$T_{max}$')
    ax.plot(step, T_min, '-', color='#2196F3', linewidth=1.5, label='$T_{min}$')
    ax.set_ylabel('Temperature (K)', fontsize=12)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    # --- Bottom: total liquid fraction ---
    ax = axes[2]
    ax.plot(step, fl_tot, '-', color='#4CAF50', linewidth=2)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Total Liquid Fraction', fontsize=12)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "03_energy_conservation.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# ============================================================
# Plot 4: Apparent Heat Capacity Verification
# ============================================================
def plot_apparent_cp():
    """Plot apparent heat capacity C_app(T) for each material.

    CSV columns: material, T, C_app, cp_base, C_app_expected
    """
    csv_path = os.path.join(OUTPUT_DIR, "apparent_heat_capacity.csv")
    if not os.path.exists(csv_path):
        print(f"  SKIP: {csv_path} not found")
        return

    # Parse with material name as string
    materials_data = {}  # name -> dict of arrays
    header = None
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if header is None:
                header = [h.strip() for h in line.split(',')]
                continue
            parts = line.split(',')
            mat_name = parts[0].strip()
            T_val = float(parts[1])
            C_app_val = float(parts[2])
            cp_base_val = float(parts[3])
            C_app_exp_val = float(parts[4])
            if mat_name not in materials_data:
                materials_data[mat_name] = {'T': [], 'C_app': [], 'cp_base': [],
                                            'C_app_expected': []}
            materials_data[mat_name]['T'].append(T_val)
            materials_data[mat_name]['C_app'].append(C_app_val)
            materials_data[mat_name]['cp_base'].append(cp_base_val)
            materials_data[mat_name]['C_app_expected'].append(C_app_exp_val)

    if not materials_data:
        print(f"  SKIP: {csv_path} has no data rows")
        return

    mat_names = list(materials_data.keys())
    n_mats = len(mat_names)
    ncols = 3
    nrows = (n_mats + ncols - 1) // ncols  # ceil division

    fig, all_axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    if nrows == 1:
        all_axes = np.array([all_axes])
    axes = all_axes.flatten()

    for i, mat_name in enumerate(mat_names):
        ax = axes[i]
        md = materials_data[mat_name]
        T = np.array(md['T'])
        C_app = np.array(md['C_app'])
        cp_base = np.array(md['cp_base'])
        C_app_exp = np.array(md['C_app_expected'])

        color = MATERIALS.get(mat_name, {}).get('color', '#888888')

        ax.plot(T, C_app, '-', color=color, linewidth=2, label='$C_{app}$ computed')
        ax.plot(T, cp_base, '--', color='gray', linewidth=1.2, label='$c_p$ (base)')
        ax.plot(T, C_app_exp, ':', color='red', linewidth=1.5, label='$C_{app}$ expected')

        ax.set_xlabel('Temperature (K)', fontsize=10)
        ax.set_ylabel('$C_{app}$ (J/(kg$\\cdot$K))', fontsize=10)
        ax.set_title(mat_name, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, framealpha=0.9, loc='upper right')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, alpha=0.3)

        # Mark mushy zone
        props = MATERIALS.get(mat_name, {})
        T_s = props.get('T_solidus')
        T_l = props.get('T_liquidus')
        if T_s is not None and T.min() <= T_s <= T.max():
            ax.axvline(T_s, color='gray', linestyle='--', linewidth=0.6, alpha=0.5)
        if T_l is not None and T.min() <= T_l <= T.max():
            ax.axvline(T_l, color='gray', linestyle='--', linewidth=0.6, alpha=0.5)

    # Hide unused subplots
    for j in range(n_mats, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Apparent Heat Capacity Verification', fontsize=15, fontweight='bold',
                 y=1.01)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "04_apparent_heat_capacity.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# ============================================================
# Plot 5: Newton-Bisection Solver Convergence
# ============================================================
def plot_newton_convergence():
    """Plot solver convergence rates.

    CSV columns: H_target, T_exact, max_iter, T_solved, error_K, converged
    """
    csv_path = os.path.join(OUTPUT_DIR, "newton_convergence.csv")
    if not os.path.exists(csv_path):
        print(f"  SKIP: {csv_path} not found")
        return

    header, data = read_csv(csv_path)
    if data.size == 0:
        print(f"  SKIP: {csv_path} has no data rows")
        return

    H_target  = data[:, 0]
    T_exact   = data[:, 1]
    max_iter  = data[:, 2].astype(int)
    T_solved  = data[:, 3]
    error_K   = data[:, 4]
    converged = data[:, 5].astype(int)

    # Group by H_target to get convergence curves
    unique_H = np.unique(H_target)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Pick representative H values (evenly spaced indices, max 8)
    n_curves = min(8, len(unique_H))
    indices = np.linspace(0, len(unique_H) - 1, n_curves, dtype=int)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_curves))

    for idx, color in zip(indices, colors):
        H_val = unique_H[idx]
        mask = H_target == H_val
        iters = max_iter[mask]
        errs = np.abs(error_K[mask])

        # Filter out zero errors for log scale
        valid = errs > 0
        if np.sum(valid) < 2:
            continue

        T_ex = T_exact[mask][0]
        label = f'H={H_val:.2e}, T={T_ex:.0f}K'
        ax.loglog(iters[valid], errs[valid], 'o-', color=color, linewidth=1.5,
                  markersize=5, label=label)

    # Reference convergence lines
    iter_ref = np.array([1, 2, 4, 8, 16, 32, 50])
    # Quadratic (Newton): error ~ C * r^(2^n)
    ax.loglog(iter_ref, 1e2 * (0.1 ** iter_ref), ':', color='gray', linewidth=1.2,
              label='Quadratic (Newton ref.)')
    # Linear (bisection): error ~ C * 0.5^n
    ax.loglog(iter_ref, 1e3 * (0.5 ** iter_ref), '--', color='lightcoral',
              linewidth=1.2, label='Linear (bisection ref.)')

    ax.set_xlabel('Maximum Iterations Allowed', fontsize=12)
    ax.set_ylabel('Temperature Error $|T_{solved} - T_{exact}|$ (K)', fontsize=12)
    ax.set_title('Newton-Bisection Solver Convergence', fontsize=14)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "05_newton_convergence.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# ============================================================
# Plot 6: addEnthalpyChange() Verification
# ============================================================
def plot_enthalpy_change():
    """Plot addEnthalpyChange verification.

    CSV columns: T_initial, H_initial, dH, H_final, T_final, fl_final
    """
    csv_path = os.path.join(OUTPUT_DIR, "add_enthalpy_change.csv")
    if not os.path.exists(csv_path):
        print(f"  SKIP: {csv_path} not found")
        return

    header, data = read_csv(csv_path)
    if data.size == 0:
        print(f"  SKIP: {csv_path} has no data rows")
        return

    T_initial = data[:, 0]
    H_initial = data[:, 1]
    dH        = data[:, 2]
    H_final   = data[:, 3]
    T_final   = data[:, 4]
    fl_final  = data[:, 5]

    H_expected = H_initial + dH

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: H_final vs H_expected scatter ---
    ax = axes[0]

    # Color by phase based on fl_final
    solid_mask = fl_final < 0.01
    mushy_mask = (fl_final >= 0.01) & (fl_final <= 0.99)
    liquid_mask = fl_final > 0.99

    if np.any(solid_mask):
        ax.scatter(H_expected[solid_mask] * 1e-9, H_final[solid_mask] * 1e-9,
                   c='#2196F3', s=25, alpha=0.7, label='Solid', zorder=5)
    if np.any(mushy_mask):
        ax.scatter(H_expected[mushy_mask] * 1e-9, H_final[mushy_mask] * 1e-9,
                   c='#FF9800', s=25, alpha=0.7, label='Mushy', zorder=5)
    if np.any(liquid_mask):
        ax.scatter(H_expected[liquid_mask] * 1e-9, H_final[liquid_mask] * 1e-9,
                   c='#F44336', s=25, alpha=0.7, label='Liquid', zorder=5)

    # y = x reference line
    h_min = min(H_expected.min(), H_final.min()) * 1e-9
    h_max = max(H_expected.max(), H_final.max()) * 1e-9
    margin = (h_max - h_min) * 0.05
    ax.plot([h_min - margin, h_max + margin], [h_min - margin, h_max + margin],
            'k--', linewidth=1, alpha=0.5, label='$y = x$')

    ax.set_xlabel('$H_{initial} + \\Delta H$ (GJ/m$^3$)', fontsize=12)
    ax.set_ylabel('$H_{final}$ (GJ/m$^3$)', fontsize=12)
    ax.set_title('addEnthalpyChange() Verification', fontsize=14)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)

    # --- Right: relative error ---
    ax = axes[1]
    H_err = np.abs(H_final - H_expected)
    H_ref = np.maximum(np.abs(H_expected), 1.0)  # avoid div by zero
    rel_err = H_err / H_ref * 100

    ax.semilogy(T_final, rel_err + 1e-16, 'o', color='#4CAF50', markersize=4,
                alpha=0.7)
    ax.set_xlabel('$T_{final}$ (K)', fontsize=12)
    ax.set_ylabel('$|H_{final} - H_{expected}| / |H_{expected}|$ (%)', fontsize=12)
    ax.set_title('Enthalpy Roundtrip Relative Error', fontsize=13)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3, which='both')

    # Annotate max error
    max_rel = np.max(rel_err)
    ax.text(0.98, 0.92, f'Max relative error: {max_rel:.2e}%',
            transform=ax.transAxes, fontsize=10, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "06_enthalpy_change_verification.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print('=' * 60)
    print('  Phase Change Module -- Validation Plots')
    print('=' * 60)
    print()

    if not os.path.isdir(OUTPUT_DIR):
        print(f"Directory '{OUTPUT_DIR}' not found.")
        print("Run from the build directory after executing the phase change test:")
        print("  ./tests/validation/test_phase_change_comprehensive")
        exit(1)

    expected_csvs = [
        "roundtrip_Ti6Al4V.csv",
        "liquid_fraction_all_materials.csv",
        "adiabatic_energy.csv",
        "apparent_heat_capacity.csv",
        "newton_convergence.csv",
        "add_enthalpy_change.csv",
    ]
    found = sum(1 for f in expected_csvs if os.path.exists(os.path.join(OUTPUT_DIR, f)))
    print(f"Found {found}/{len(expected_csvs)} CSV files in {OUTPUT_DIR}/")
    print()
    print("Generating plots...")
    print()

    plot_roundtrip()
    plot_liquid_fraction()
    plot_energy_conservation()
    plot_apparent_cp()
    plot_newton_convergence()
    plot_enthalpy_change()

    print()
    print('=' * 60)
    print(f'  All plots saved to: {OUTPUT_DIR}/')
    print('=' * 60)

    # List generated files
    if os.path.isdir(OUTPUT_DIR):
        pngs = sorted(f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png'))
        for f in pngs:
            fpath = os.path.join(OUTPUT_DIR, f)
            size_kb = os.path.getsize(fpath) / 1024
            print(f'  {f:50s} {size_kb:6.1f} KB')
