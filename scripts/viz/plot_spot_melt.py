#!/usr/bin/env python3
"""
Spot melting diagnostic: 2×4 panel figure showing temperature field and
velocity field at four timestamps in the XZ midplane.

Reads ASCII VTK STRUCTURED_POINTS files produced by the LBM solver.
"""

import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# === PARAMETERS ===
VTK_DIR   = "/home/yzk/LBMProject/output_spot_melt"
OUT_FILE  = "/home/yzk/LBMProject/scripts/viz/spot_melt_diagnostic.png"

# Grid dimensions (cells)
NX, NY, NZ = 75, 75, 50
DX         = 2e-6          # m per cell
DT         = 1e-7          # s per timestep  (dt = dx/20 => LU→m/s scale = DX/DT)
LU_TO_MS   = DX / DT      # 20.0 m/s per LU

# Midplane slice
IY_MID = NY // 2          # 37

# Files and physical times (µs)
SNAPSHOTS = [
    ("spot_melt_000249.vtk",  25),
    ("spot_melt_000498.vtk",  50),
    ("spot_melt_000996.vtk", 100),
    ("spot_melt_001494.vtk", 149),
]

# Quiver subsampling stride and melt pool gate
QUIVER_STRIDE  = 4         # subsample every N cells in x and z
LF_GATE        = 0.1       # show vectors only where fill_level > this value (metal, not gas)
# Quiver scale: arrow length in plot µm per m/s of velocity.
# At stride=4 cells = 8µm spacing, a scale of 0.6 makes v=5m/s arrows ~8µm long.
QUIVER_SCALE   = 0.6       # (m/s) per µm  →  arrow_length_µm = v_ms / QUIVER_SCALE

# Temperature colormap range (K) — fixed across all panels for comparison
T_MIN, T_MAX = 300.0, 3200.0

# Iron solidus (approximate), used for context (not plotted separately)
T_SOLIDUS = 1811.0        # K


# === VTK PARSER ===

def parse_vtk(filepath):
    """
    Parse an ASCII VTK STRUCTURED_POINTS file and return a dict of arrays.

    Fields returned (each shape NZ×NY×NX, i.e. z-major Fortran-like order
    matching VTK x-fastest indexing):
        velocity   : (NZ, NY, NX, 3) float32  — lattice units
        temperature: (NZ, NY, NX)    float32  — K
        liquid_fraction: (NZ, NY, NX) float32
        fill_level : (NZ, NY, NX)    float32
        curvature  : (NZ, NY, NX)    float32
        pressure   : (NZ, NY, NX)    float32
    """
    N = NX * NY * NZ

    with open(filepath, 'r') as fh:
        lines = fh.readlines()

    # Locate field start lines by scanning headers
    field_starts = {}
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith('VECTORS'):
            name = stripped.split()[1].lower()
            field_starts[name] = ('vector', i + 1)
        elif stripped.startswith('SCALARS'):
            name = stripped.split()[1].lower()
            # next line is LOOKUP_TABLE default — data starts after that
            field_starts[name] = ('scalar', i + 2)
        i += 1

    def read_scalar(start):
        data = np.empty(N, dtype=np.float32)
        for k, ln in enumerate(lines[start:start + N]):
            data[k] = float(ln)
        return data.reshape(NZ, NY, NX)

    def read_vector(start):
        data = np.empty((N, 3), dtype=np.float32)
        for k, ln in enumerate(lines[start:start + N]):
            parts = ln.split()
            data[k, 0] = float(parts[0])
            data[k, 1] = float(parts[1])
            data[k, 2] = float(parts[2])
        return data.reshape(NZ, NY, NX, 3)

    result = {}
    for name, (kind, start) in field_starts.items():
        if kind == 'scalar':
            result[name] = read_scalar(start)
        else:
            result[name] = read_vector(start)
    return result


# === SLICE EXTRACTION ===

def xz_slice(data_3d, iy=IY_MID):
    """
    Extract XZ midplane slice: returns array of shape (NZ, NX).
    data_3d has shape (NZ, NY, NX).
    """
    return data_3d[:, iy, :]


# === MAIN ===

def main():
    # Physical coordinate arrays for the XZ slice (in µm)
    x_um = np.arange(NX) * DX * 1e6   # 0 … 148 µm
    z_um = np.arange(NZ) * DX * 1e6   # 0 … 98 µm
    X, Z = np.meshgrid(x_um, z_um)    # shape (NZ, NX)

    # Subsampled grid for quiver
    sx = slice(0, NX, QUIVER_STRIDE)
    sz = slice(0, NZ, QUIVER_STRIDE)
    Xq, Zq = X[sz, sx], Z[sz, sx]

    # ---- figure layout ----
    fig, axes = plt.subplots(
        2, 4,
        figsize=(18, 9),
        constrained_layout=True,
    )
    fig.suptitle("Spot Melting Simulation — XZ Midplane Diagnostics", fontsize=13, y=1.01)

    # Shared colorbars: temperature and velocity magnitude
    t_norm   = mcolors.Normalize(vmin=T_MIN, vmax=T_MAX)
    t_cmap   = 'inferno'
    vel_cmap = 'plasma'

    vel_max_global = 0.0  # will be set after first pass for shared scale

    # --- First pass: collect velocity max across all snapshots ---
    snapshots_data = []
    for fname, t_us in SNAPSHOTS:
        fpath = f"{VTK_DIR}/{fname}"
        print(f"  Loading {fname} …")
        fields = parse_vtk(fpath)
        vel_slice = xz_slice(fields['velocity'])          # (NZ, NX, 3) LU
        vmag      = np.sqrt((vel_slice**2).sum(axis=-1)) * LU_TO_MS
        vel_max_global = max(vel_max_global, vmag.max())
        snapshots_data.append((t_us, fields))

    v_norm = mcolors.Normalize(vmin=0.0, vmax=vel_max_global)
    print(f"  Global velocity max: {vel_max_global:.2f} m/s")

    # --- Second pass: render panels ---
    im_t   = None
    im_vel = None

    for col, (t_us, fields) in enumerate(snapshots_data):
        temp  = xz_slice(fields['temperature'])           # (NZ, NX)  K
        lf    = xz_slice(fields['liquid_fraction'])       # (NZ, NX)
        fl    = xz_slice(fields['fill_level'])            # (NZ, NX)
        vel_s = xz_slice(fields['velocity'])              # (NZ, NX, 3) LU
        vmag  = np.sqrt((vel_s**2).sum(axis=-1)) * LU_TO_MS
        vx_ms = vel_s[:, :, 0] * LU_TO_MS
        vz_ms = vel_s[:, :, 2] * LU_TO_MS

        # ---- Row 0: temperature + contours ----
        ax0 = axes[0, col]
        im_t = ax0.pcolormesh(X, Z, temp, cmap=t_cmap, norm=t_norm, rasterized=True)
        ax0.set_aspect('equal')
        ax0.set_xlim(x_um[0], x_um[-1])
        ax0.set_ylim(z_um[0], z_um[-1])
        ax0.set_title(f"t = {t_us} µs", fontsize=11)
        ax0.set_xlabel("x (µm)")
        if col == 0:
            ax0.set_ylabel("z (µm)")
        else:
            ax0.set_yticklabels([])

        # Melt pool boundary: liquid_fraction = 0.5 contour
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ax0.contour(X, Z, lf, levels=[0.5], colors='white',
                        linewidths=1.2, linestyles='-')
            # Free surface: fill_level = 0.5 contour
            ax0.contour(X, Z, fl, levels=[0.5], colors='cyan',
                        linewidths=1.0, linestyles='--')

        # Legend proxies on first column only
        if col == 0:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='white', lw=1.2, label='Melt boundary (fl=0.5)'),
                Line2D([0], [0], color='cyan',  lw=1.0, ls='--', label='Free surface (fill=0.5)'),
            ]
            ax0.legend(handles=legend_elements, loc='lower left', fontsize=7,
                       framealpha=0.6, handlelength=1.5)

        # ---- Row 1: velocity magnitude + vectors ----
        ax1 = axes[1, col]
        im_vel = ax1.pcolormesh(X, Z, vmag, cmap=vel_cmap, norm=v_norm, rasterized=True)
        ax1.set_aspect('equal')
        ax1.set_xlim(x_um[0], x_um[-1])
        ax1.set_ylim(z_um[0], z_um[-1])
        ax1.set_xlabel("x (µm)")
        if col == 0:
            ax1.set_ylabel("z (µm)")
        else:
            ax1.set_yticklabels([])

        # Melt pool boundary contour
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ax1.contour(X, Z, lf, levels=[0.5], colors='white',
                        linewidths=1.2, linestyles='-')

        # Quiver: subsample, gate to melt pool
        fl_q  = fl[sz, sx]   # Use VOF fill_level (metal=1, gas=0), NOT liquid_fraction
        vx_q  = vx_ms[sz, sx]
        vz_q  = vz_ms[sz, sx]
        mask  = fl_q > LF_GATE

        if mask.any():
            vx_masked = np.where(mask, vx_q, np.nan)
            vz_masked = np.where(mask, vz_q, np.nan)
            ax1.quiver(
                Xq, Zq, vx_masked, vz_masked,
                scale=QUIVER_SCALE,   # m/s per µm: controls arrow length
                scale_units='xy',     # arrow length in data units (µm)
                angles='xy',          # arrow direction in data coords
                width=0.003,
                headwidth=4,
                headlength=4,
                color='white',
                alpha=0.75,
            )

    # ---- Colorbars ----
    cbar_t = fig.colorbar(
        plt.cm.ScalarMappable(norm=t_norm, cmap=t_cmap),
        ax=axes[0, :],
        orientation='vertical',
        fraction=0.015,
        pad=0.02,
        label="Temperature (K)",
    )

    cbar_v = fig.colorbar(
        plt.cm.ScalarMappable(norm=v_norm, cmap=vel_cmap),
        ax=axes[1, :],
        orientation='vertical',
        fraction=0.015,
        pad=0.02,
        label="|v| (m/s)",
    )

    # ---- Row labels ----
    axes[0, 0].annotate(
        "Temperature", xy=(-0.22, 0.5), xycoords='axes fraction',
        fontsize=11, ha='center', va='center', rotation=90, fontweight='bold',
    )
    axes[1, 0].annotate(
        "Velocity", xy=(-0.22, 0.5), xycoords='axes fraction',
        fontsize=11, ha='center', va='center', rotation=90, fontweight='bold',
    )

    fig.savefig(OUT_FILE, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {OUT_FILE}")


if __name__ == "__main__":
    import os
    import sys

    # Sanity checks
    for fname, _ in SNAPSHOTS:
        p = f"{VTK_DIR}/{fname}"
        if not os.path.exists(p):
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"Grid: {NX}x{NY}x{NZ}, dx={DX*1e6:.1f} µm, LU→m/s scale={LU_TO_MS:.1f}")
    print(f"Midplane slice: iy={IY_MID}")
    print(f"Quiver stride: {QUIVER_STRIDE} cells, gate: lf>{LF_GATE}")
    main()
