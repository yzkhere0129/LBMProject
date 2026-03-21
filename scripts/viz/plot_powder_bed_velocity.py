#!/usr/bin/env python3
"""
Powder bed velocity field visualization.

Row 1: Temperature + velocity quiver (XZ midplane)
Row 2: Velocity magnitude heatmap (XZ midplane)
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VTK_DIR = "output_powder_bed_sim"
if not os.path.isdir(VTK_DIR):
    VTK_DIR = os.path.join(os.path.dirname(__file__), "../../build/output_powder_bed_sim")

OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "powder_bed_velocity.png")

NX, NY, NZ = 500, 150, 90
DX = 2e-6
DT = 8e-8
N = NX * NY * NZ
IY_MID = NY // 2
LU_TO_MS = DX / DT  # 25 m/s per LU

SNAPSHOTS = [
    ("powder_sim_002500.vtk",  200),
    ("powder_sim_005000.vtk",  400),
    ("powder_sim_007500.vtk",  600),
    ("powder_sim_010000.vtk",  800),
]

T_SOLIDUS = 1648.0
T_LIQUIDUS = 1673.0
QUIVER_STRIDE = 5
FL_GATE = 0.1


def parse_vtk_fields(filepath):
    """Parse temperature, fill_level, and velocity from ASCII VTK."""
    with open(filepath, 'r') as fh:
        lines = fh.readlines()

    fields = {}
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith('VECTORS'):
            name = s.split()[1].lower()
            fields[name] = ('vector', i + 1)
        elif s.startswith('SCALARS'):
            name = s.split()[1].lower()
            fields[name] = ('scalar', i + 2)
        i += 1

    result = {}
    for name, (kind, start) in fields.items():
        if kind == 'scalar':
            data = np.empty(N, dtype=np.float32)
            idx = 0
            for line in lines[start:]:
                if idx >= N: break
                for v in line.split():
                    if idx >= N: break
                    data[idx] = float(v); idx += 1
            result[name] = data.reshape(NZ, NY, NX)
        else:
            data = np.empty((N, 3), dtype=np.float32)
            idx = 0
            for line in lines[start:]:
                if idx >= N: break
                parts = line.split()
                if len(parts) >= 3:
                    data[idx] = [float(parts[0]), float(parts[1]), float(parts[2])]
                    idx += 1
            result[name] = data.reshape(NZ, NY, NX, 3)
    return result


def main():
    snapshots = []
    for fname, t_us in SNAPSHOTS:
        path = os.path.join(VTK_DIR, fname)
        if os.path.exists(path):
            snapshots.append((path, t_us))
    if not snapshots:
        print("No VTK files found!"); sys.exit(1)

    ncols = len(snapshots)
    x_um = np.arange(NX) * DX * 1e6
    z_um = np.arange(NZ) * DX * 1e6
    X, Z = np.meshgrid(x_um, z_um)

    sx = slice(0, NX, QUIVER_STRIDE)
    sz = slice(0, NZ, QUIVER_STRIDE)
    Xq, Zq = X[sz, sx], Z[sz, sx]

    fig, axes = plt.subplots(2, ncols, figsize=(5.5*ncols, 7), constrained_layout=True)
    if ncols == 1:
        axes = axes.reshape(2, 1)

    im_v = None
    for col, (vtk_path, t_us) in enumerate(snapshots):
        print(f"  Parsing {os.path.basename(vtk_path)} (t={t_us} μs)...", flush=True)
        d = parse_vtk_fields(vtk_path)

        T_xz = d['temperature'][:, IY_MID, :]
        f_xz = d['fill_level'][:, IY_MID, :]
        vel = d['velocity'][:, IY_MID, :, :]  # (NZ, NX, 3) in lattice units
        vx_phys = vel[:, :, 0] * LU_TO_MS
        vz_phys = vel[:, :, 2] * LU_TO_MS
        vmag = np.sqrt(vx_phys**2 + vz_phys**2)

        # --- Row 0: Temperature + velocity quiver ---
        ax = axes[0, col]
        T_masked = np.ma.masked_where(f_xz < 0.01, T_xz)
        cmap_T = plt.cm.hot.copy(); cmap_T.set_bad('#222222')
        ax.pcolormesh(X, Z, T_masked, cmap=cmap_T, vmin=300, vmax=3000, shading='auto')

        if T_xz.max() > T_SOLIDUS:
            ax.contour(X, Z, T_xz, levels=[T_SOLIDUS], colors='cyan', linewidths=0.7, linestyles='--')

        # Quiver — only in metal (f>0.1) and liquid (T>T_solidus)
        mask_q = (f_xz[sz, sx] > FL_GATE) & (T_xz[sz, sx] > T_SOLIDUS)
        ux_q = np.where(mask_q, vx_phys[sz, sx], 0)
        uz_q = np.where(mask_q, vz_phys[sz, sx], 0)
        speed_q = np.sqrt(ux_q**2 + uz_q**2)

        if speed_q.max() > 0.01:
            ax.quiver(Xq, Zq, ux_q, uz_q, speed_q,
                      cmap='cool', scale=15, width=0.003,
                      headwidth=4, headlength=3, alpha=0.9,
                      clim=[0, 2.0])

        ax.set_title(f't = {t_us} μs', fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        if col == 0: ax.set_ylabel('z [μm]')
        ax.set_xlabel('x [μm]')
        ax.tick_params(labelsize=8)

        # --- Row 1: Velocity magnitude ---
        ax2 = axes[1, col]
        vmag_masked = np.ma.masked_where(f_xz < 0.01, vmag)
        cmap_v = plt.cm.inferno.copy(); cmap_v.set_bad('#111111')
        im_v = ax2.pcolormesh(X, Z, vmag_masked, cmap=cmap_v, vmin=0, vmax=2.0, shading='auto')

        if T_xz.max() > T_SOLIDUS:
            ax2.contour(X, Z, T_xz, levels=[T_SOLIDUS], colors='cyan', linewidths=0.5, linestyles='--')
        ax2.contour(X, Z, f_xz, levels=[0.5], colors='white', linewidths=0.4)

        ax2.set_aspect('equal')
        if col == 0: ax2.set_ylabel('z [μm]')
        ax2.set_xlabel('x [μm]')
        ax2.tick_params(labelsize=8)

    cb_v = fig.colorbar(im_v, ax=axes[1, :], shrink=0.8, pad=0.02)
    cb_v.set_label('|v| [m/s]', fontsize=10)

    fig.suptitle('LPBF 316L — Melt Pool Flow (P=75W, v=800mm/s, 80μm powder)\n'
                 'Top: T + velocity arrows in liquid | Bottom: velocity magnitude',
                 fontsize=12, fontweight='bold')

    plt.savefig(OUT_FILE, dpi=200, bbox_inches='tight')
    print(f"\nSaved: {OUT_FILE}")


if __name__ == '__main__':
    main()
