#!/usr/bin/env python3
"""
3D top-view rendering of melt track — f>0.5 isosurface colored by temperature.

Uses matplotlib 3D surface plot with z-projection (top-down view).
Renders the highest metal surface z(x,y) = max{k : f(i,j,k) > 0.5}.
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

NX, NY, NZ = 500, 150, 80
DX = 2e-6
N = NX * NY * NZ

VTK_DIR = "output_powder_bed_sim"

# Use the final snapshot
VTK_FILE = os.path.join(VTK_DIR, "powder_sim_016250.vtk")
OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "powder_bed_3d_topview.png")


def parse_vtk_field(filepath, field_name):
    with open(filepath, 'r') as fh:
        lines = fh.readlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f'SCALARS {field_name}'):
            start = i + 2; break
    if start is None:
        raise ValueError(f"Field '{field_name}' not found")
    data = np.empty(N, dtype=np.float32)
    idx = 0
    for line in lines[start:]:
        if idx >= N: break
        for v in line.split():
            if idx >= N: break
            data[idx] = float(v); idx += 1
    return data.reshape(NZ, NY, NX)


def main():
    if not os.path.exists(VTK_FILE):
        # Find the last VTK file
        import glob
        candidates = sorted(glob.glob(os.path.join(VTK_DIR, "powder_sim_*.vtk")))
        if not candidates:
            print("No VTK files found!"); sys.exit(1)
        vtk_path = candidates[-1]
    else:
        vtk_path = VTK_FILE

    print(f"Parsing {os.path.basename(vtk_path)}...", flush=True)
    f = parse_vtk_field(vtk_path, 'fill_level')
    T = parse_vtk_field(vtk_path, 'temperature')

    # Initial state for comparison
    vtk_t0 = os.path.join(VTK_DIR, "powder_sim_000000.vtk")
    f0 = parse_vtk_field(vtk_t0, 'fill_level') if os.path.exists(vtk_t0) else None

    x_um = np.arange(NX) * DX * 1e6
    y_um = np.arange(NY) * DX * 1e6

    # Compute surface height map: z_surf(x,y) = highest cell with f > 0.5
    z_surf = np.full((NY, NX), np.nan)
    T_surf = np.full((NY, NX), np.nan)
    for j in range(NY):
        for i in range(NX):
            for k in range(NZ-1, -1, -1):
                if f[k, j, i] > 0.5:
                    z_surf[j, i] = k * DX * 1e6
                    T_surf[j, i] = T[k, j, i]
                    break

    # Same for initial state
    if f0 is not None:
        z_surf0 = np.full((NY, NX), np.nan)
        for j in range(NY):
            for i in range(NX):
                for k in range(NZ-1, -1, -1):
                    if f0[k, j, i] > 0.5:
                        z_surf0[j, i] = k * DX * 1e6
                        break

    # Compute fill fraction per column (how much metal in z-column)
    fill_column = f.sum(axis=0)  # (NY, NX) — total fill in each column
    fill_column0 = f0.sum(axis=0) if f0 is not None else None

    # ===== FIGURE: 2×2 panels =====
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel (0,0): Surface height map — final state
    ax = axes[0, 0]
    im = ax.pcolormesh(x_um, y_um, z_surf, cmap='terrain',
                       vmin=30, vmax=100, shading='auto')
    ax.set_xlabel('x [μm]'); ax.set_ylabel('y [μm]')
    ax.set_title('Surface Height z_max (f>0.5) — Final', fontsize=11)
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='z [μm]', shrink=0.8)

    # Panel (0,1): Surface temperature
    ax2 = axes[0, 1]
    T_surf_masked = np.ma.masked_invalid(T_surf)
    im2 = ax2.pcolormesh(x_um, y_um, T_surf_masked, cmap='hot',
                          vmin=600, vmax=3500, shading='auto')
    ax2.set_xlabel('x [μm]'); ax2.set_ylabel('y [μm]')
    ax2.set_title('Surface Temperature — Final', fontsize=11)
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2, label='T [K]', shrink=0.8)

    # Panel (1,0): Height change (final - initial)
    ax3 = axes[1, 0]
    if f0 is not None:
        dz = z_surf - z_surf0
        dz_masked = np.ma.masked_invalid(dz)
        vlim = max(abs(np.nanmin(dz)), abs(np.nanmax(dz)), 5)
        im3 = ax3.pcolormesh(x_um, y_um, dz_masked, cmap='RdBu_r',
                              vmin=-vlim, vmax=vlim, shading='auto')
        ax3.set_xlabel('x [μm]'); ax3.set_ylabel('y [μm]')
        ax3.set_title('Surface Height Change Δz [μm] (red=up, blue=down)', fontsize=11)
        ax3.set_aspect('equal')
        plt.colorbar(im3, ax=ax3, label='Δz [μm]', shrink=0.8)
    else:
        ax3.text(0.5, 0.5, 'No initial state', ha='center', va='center',
                 transform=ax3.transAxes)

    # Panel (1,1): Column fill difference (densification map)
    ax4 = axes[1, 1]
    if fill_column0 is not None:
        dfill = fill_column - fill_column0
        vlim_f = max(abs(np.nanmin(dfill)), abs(np.nanmax(dfill)), 1)
        im4 = ax4.pcolormesh(x_um, y_um, dfill, cmap='RdBu_r',
                              vmin=-vlim_f, vmax=vlim_f, shading='auto')
        ax4.set_xlabel('x [μm]'); ax4.set_ylabel('y [μm]')
        ax4.set_title('Column Fill Change Δ(Σf) — Densification Map', fontsize=11)
        ax4.set_aspect('equal')
        plt.colorbar(im4, ax=ax4, label='Δ(Σf)', shrink=0.8)
    else:
        ax4.text(0.5, 0.5, 'No initial state', ha='center', va='center',
                 transform=ax4.transAxes)

    fig.suptitle('LPBF 316L — 3D Top View (P=150W, r₀=35μm, v=800mm/s, θ=10°)\n'
                 'Ray Tracing + Gas Isolation + Wetting, t=final',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=150, bbox_inches='tight')
    print(f"Saved: {OUT_FILE}")


if __name__ == '__main__':
    main()
