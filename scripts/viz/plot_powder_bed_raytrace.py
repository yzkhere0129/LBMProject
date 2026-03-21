#!/usr/bin/env python3
"""
Powder bed LPBF visualization with ray-tracing laser.

Extracts XZ midplane temperature field and VOF morphology from VTK snapshots.
Produces a 2-row × N-col panel figure:
  Row 1: Temperature field with melt pool contours
  Row 2: VOF fill_level showing powder morphology

Usage:
    python plot_powder_bed_raytrace.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# === PARAMETERS ===
VTK_DIR = "output_powder_bed_sim"
if not os.path.isdir(VTK_DIR):
    VTK_DIR = os.path.join(os.path.dirname(__file__), "../../build/output_powder_bed_sim")

OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "powder_bed_raytrace.png")

# Grid
NX, NY, NZ = 500, 150, 65
DX = 2e-6  # m
N = NX * NY * NZ

# Midplane
IY_MID = NY // 2

# Snapshots to plot (filename, physical time in μs)
SNAPSHOTS = [
    ("powder_sim_000000.vtk",    0),
    ("powder_sim_001250.vtk",  100),
    ("powder_sim_003750.vtk",  300),
    ("powder_sim_006250.vtk",  500),
    ("powder_sim_008750.vtk",  700),
    ("powder_sim_011250.vtk",  900),
    ("powder_sim_013750.vtk", 1100),
]

# Temperature colormap
T_MIN, T_MAX = 300.0, 4500.0
T_SOLIDUS = 1648.0   # 316L solidus [K]
T_LIQUIDUS = 1673.0  # 316L liquidus [K]

# === FAST VTK PARSER (numpy-based) ===

def parse_vtk_field(filepath, field_name):
    """Parse a single scalar field from ASCII VTK. Returns (NZ, NY, NX) array."""
    with open(filepath, 'r') as fh:
        lines = fh.readlines()

    # Find field header
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f'SCALARS {field_name}'):
            start = i + 2  # skip LOOKUP_TABLE line
            break

    if start is None:
        raise ValueError(f"Field '{field_name}' not found in {filepath}")

    # Read N values
    data = np.empty(N, dtype=np.float32)
    idx = 0
    for line in lines[start:]:
        if idx >= N:
            break
        vals = line.split()
        for v in vals:
            if idx >= N:
                break
            data[idx] = float(v)
            idx += 1

    return data.reshape(NZ, NY, NX)


def parse_vtk_vector(filepath, field_name):
    """Parse a vector field from ASCII VTK. Returns (NZ, NY, NX, 3) array."""
    with open(filepath, 'r') as fh:
        lines = fh.readlines()

    start = None
    for i, line in enumerate(lines):
        if line.startswith(f'VECTORS {field_name}'):
            start = i + 1
            break

    if start is None:
        raise ValueError(f"Vector field '{field_name}' not found in {filepath}")

    data = np.empty((N, 3), dtype=np.float32)
    idx = 0
    for line in lines[start:]:
        if idx >= N:
            break
        parts = line.split()
        if len(parts) >= 3:
            data[idx, 0] = float(parts[0])
            data[idx, 1] = float(parts[1])
            data[idx, 2] = float(parts[2])
            idx += 1

    return data.reshape(NZ, NY, NX, 3)


# === XZ SLICE ===

def xz_slice(data_3d, iy=IY_MID):
    """Extract XZ midplane: shape (NZ, NX)."""
    return data_3d[:, iy, :]


# === MAIN ===

def main():
    # Filter to existing files
    snapshots = []
    for fname, t_us in SNAPSHOTS:
        path = os.path.join(VTK_DIR, fname)
        if os.path.exists(path):
            snapshots.append((path, t_us))
        else:
            print(f"  Skip: {fname} (not found)")

    if not snapshots:
        print("ERROR: No VTK files found!")
        sys.exit(1)

    ncols = len(snapshots)
    print(f"Plotting {ncols} snapshots...")

    # Physical coordinates [μm]
    x_um = np.arange(NX) * DX * 1e6
    z_um = np.arange(NZ) * DX * 1e6
    X, Z = np.meshgrid(x_um, z_um)

    # --- Figure layout ---
    fig, axes = plt.subplots(2, ncols, figsize=(5*ncols, 6),
                              constrained_layout=True)
    if ncols == 1:
        axes = axes.reshape(2, 1)

    # Custom temperature colormap (black → red → yellow → white)
    cmap_T = plt.cm.hot

    for col, (vtk_path, t_us) in enumerate(snapshots):
        print(f"  Parsing {os.path.basename(vtk_path)} (t={t_us} μs)...")

        T = parse_vtk_field(vtk_path, 'temperature')
        f = parse_vtk_field(vtk_path, 'fill_level')

        T_xz = xz_slice(T)
        f_xz = xz_slice(f)

        # --- Row 0: Temperature ---
        ax = axes[0, col]
        # Mask gas cells (f < 0.01) to show as gray background
        T_masked = np.ma.masked_where(f_xz < 0.01, T_xz)
        cmap_bg = plt.cm.hot.copy()
        cmap_bg.set_bad(color='#222222')

        im_T = ax.pcolormesh(X, Z, T_masked, cmap=cmap_bg,
                              vmin=T_MIN, vmax=T_MAX, shading='auto')

        # Melt pool contours
        if T_xz.max() > T_SOLIDUS:
            ax.contour(X, Z, T_xz, levels=[T_SOLIDUS], colors='cyan',
                       linewidths=0.8, linestyles='--')
            ax.contour(X, Z, T_xz, levels=[T_LIQUIDUS], colors='white',
                       linewidths=0.8, linestyles='-')

        ax.set_title(f't = {t_us} μs', fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        if col == 0:
            ax.set_ylabel('z [μm]', fontsize=10)
        ax.set_xlabel('x [μm]', fontsize=9)
        ax.tick_params(labelsize=8)

        # --- Row 1: VOF fill_level ---
        ax2 = axes[1, col]
        cmap_vof = plt.cm.coolwarm
        im_f = ax2.pcolormesh(X, Z, f_xz, cmap=cmap_vof,
                               vmin=0, vmax=1, shading='auto')

        # Interface contour
        ax2.contour(X, Z, f_xz, levels=[0.5], colors='black',
                    linewidths=0.6)

        ax2.set_aspect('equal')
        if col == 0:
            ax2.set_ylabel('z [μm]', fontsize=10)
        ax2.set_xlabel('x [μm]', fontsize=9)
        ax2.tick_params(labelsize=8)

    # Colorbars
    cb_T = fig.colorbar(im_T, ax=axes[0, :], shrink=0.8, pad=0.02)
    cb_T.set_label('Temperature [K]', fontsize=10)

    cb_f = fig.colorbar(im_f, ax=axes[1, :], shrink=0.8, pad=0.02)
    cb_f.set_label('Fill Level', fontsize=10)

    fig.suptitle('LPBF 316L — P=150W, r₀=35μm, v=800mm/s\n'
                 'Ray Tracing + Gas Isolation + Contact Angle 10° (wetting)',
                 fontsize=12, fontweight='bold')

    plt.savefig(OUT_FILE, dpi=200, bbox_inches='tight')
    print(f"\nSaved: {OUT_FILE}")


if __name__ == '__main__':
    main()
