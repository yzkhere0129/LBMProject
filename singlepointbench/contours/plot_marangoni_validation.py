#!/usr/bin/env python3
"""
Marangoni Benchmark Validation — LBM vs OpenFOAM
=================================================
Uses matplotlib contour() on 2D temperature fields directly.
No manual point sorting — marching squares handles topology correctly.
"""
import numpy as np
import re, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

mpl.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.linewidth': 0.8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

ROOT = Path(__file__).parent.parent.parent   # LBMProject root
CONTOUR_DIR = Path(__file__).parent
T_ISO = 1650.0
TIMES = [25, 50, 60, 75]

# ═══════════════════════════════════════════════════════════════════════════
# OpenFOAM field reader
# ═══════════════════════════════════════════════════════════════════════════
OF_CASE = ROOT / "openfoam" / "spot_melting_marangoni"
OF_COND = ROOT / "openfoam" / "spot_melting_benchmark"  # pure conduction case (if exists)

NX_OF, NY_OF, NZ_OF = 100, 50, 100
X_MIN, X_MAX = -100e-6, 100e-6
Y_MIN, Y_MAX = -100e-6, 0.0
Z_MIN, Z_MAX = -100e-6, 100e-6
DX_OF = (X_MAX - X_MIN) / NX_OF
DY_OF = (Y_MAX - Y_MIN) / NY_OF
DZ_OF = (Z_MAX - Z_MIN) / NZ_OF

x_of = np.linspace(X_MIN + DX_OF/2, X_MAX - DX_OF/2, NX_OF)
y_of = np.linspace(Y_MIN + DY_OF/2, Y_MAX - DY_OF/2, NY_OF)
z_of = np.linspace(Z_MIN + DZ_OF/2, Z_MAX - DZ_OF/2, NZ_OF)
K_SLICE = int(np.argmin(np.abs(z_of)))

# Coordinate transform: OF [m] → LBM [μm]
X_OF_UM = x_of * 1e6 + 100.0   # OF x=0 → LBM X=100μm
Y_OF_UM = y_of * 1e6 + 150.0   # OF y=0 (surface) → LBM Z=150μm


def read_of_scalar(filepath):
    with open(filepath) as f:
        text = f.read()
    m = re.search(r"internalField\s+nonuniform\s+List<scalar>\s*\n\s*(\d+)\s*\n\s*\(", text)
    if m:
        n = int(m.group(1))
        start = m.end()
        end = text.index(")", start)
        return np.fromstring(text[start:end], sep="\n", count=n)
    m = re.search(r"internalField\s+uniform\s+([\d.eE+\-]+)", text)
    if m:
        return np.full(NX_OF * NY_OF * NZ_OF, float(m.group(1)))
    raise ValueError(f"Cannot parse {filepath}")


def find_of_time(case_dir, t_us):
    t_target = t_us * 1e-6
    best, best_err = None, 1e30
    for name in os.listdir(case_dir):
        try:
            err = abs(float(name) - t_target)
            if err < best_err:
                best, best_err = name, err
        except ValueError:
            pass
    return best if best_err < 2e-6 else None


def get_of_T2d(case_dir, t_us):
    """Read OpenFOAM T field, return 2D slice on z=0 plane as (Y_OF_UM, X_OF_UM, T2d)."""
    tdir = find_of_time(case_dir, t_us)
    if tdir is None:
        return None
    T_all = read_of_scalar(os.path.join(case_dir, tdir, "T"))
    T3d = T_all.reshape((NZ_OF, NY_OF, NX_OF))   # (z, y, x)
    T2d = T3d[K_SLICE]                             # (NY, NX) on (y, x) grid
    return T2d   # axes: row=y (depth), col=x (surface direction)


# ═══════════════════════════════════════════════════════════════════════════
# LBM field reader (from 2D CSV exported by benchmark)
# ═══════════════════════════════════════════════════════════════════════════
def get_lbm_T2d(tag, prefix="lbm_temperature"):
    """Read LBM 2D temperature CSV → (X_arr, Z_arr, T2d)."""
    fp = CONTOUR_DIR / f"{prefix}_{tag}us.csv"
    if not fp.exists():
        return None, None, None
    d = np.loadtxt(fp, delimiter=",", skiprows=1)
    xu = np.sort(np.unique(d[:, 0]))
    zu = np.sort(np.unique(d[:, 1]))
    nx, nz = len(xu), len(zu)
    T2d = np.full((nz, nx), np.nan)
    for row in d:
        ix = np.searchsorted(xu, row[0])
        iz = np.searchsorted(zu, row[1])
        if ix < nx and iz < nz:
            T2d[iz, ix] = row[2]
    return xu, zu, T2d


# ═══════════════════════════════════════════════════════════════════════════
# Extract contour paths from a 2D field using matplotlib (topology-safe)
# ═══════════════════════════════════════════════════════════════════════════
def extract_contour_paths(x1d, y1d, T2d, level):
    """Return list of (N,2) arrays — each is an ordered contour segment."""
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(x1d, y1d, T2d, levels=[level])
    plt.close(fig_tmp)
    paths = []
    for coll in cs.collections:
        for p in coll.get_paths():
            if len(p.vertices) > 2:
                paths.append(p.vertices.copy())
    return paths


# ═══════════════════════════════════════════════════════════════════════════
# MAIN: Build figure
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.2), sharex=True, sharey=True)
fig.suptitle('Solidus isotherm (T = 1650 K): LBM vs OpenFOAM with Marangoni',
             fontsize=12, fontweight='bold', y=0.97)

for i, t_us in enumerate(TIMES):
    ax = axes[i // 2][i % 2]

    # ── OpenFOAM Marangoni ──
    T2d_of = get_of_T2d(OF_CASE, t_us)
    if T2d_of is not None and T2d_of.max() >= T_ISO:
        paths_of = extract_contour_paths(X_OF_UM, Y_OF_UM, T2d_of, T_ISO)
        for j, seg in enumerate(paths_of):
            ax.plot(seg[:, 0], seg[:, 1], color='#4A4A4A', lw=2.2,
                    label='OpenFOAM + Marangoni' if (i == 0 and j == 0) else '_')

    # ── OpenFOAM pure conduction (if available) ──
    if OF_COND.exists():
        T2d_of_c = get_of_T2d(OF_COND, t_us)
        if T2d_of_c is not None and T2d_of_c.max() >= T_ISO:
            paths_of_c = extract_contour_paths(X_OF_UM, Y_OF_UM, T2d_of_c, T_ISO)
            for j, seg in enumerate(paths_of_c):
                ax.plot(seg[:, 0], seg[:, 1], color='#888888', lw=1.0, ls='--',
                        alpha=0.5,
                        label='OpenFOAM conduction' if (i == 0 and j == 0) else '_')

    # ── LBM Marangoni ──
    xu, zu, T2d_lbm = get_lbm_T2d(str(t_us))
    if T2d_lbm is not None and np.nanmax(T2d_lbm) >= T_ISO:
        paths_lbm = extract_contour_paths(xu, zu, T2d_lbm, T_ISO)
        for j, seg in enumerate(paths_lbm):
            ax.plot(seg[:, 0], seg[:, 1], color='#E6A125', lw=1.5,
                    marker='o', ms=1.8, markeredgewidth=0,
                    label='LBM + Marangoni' if (i == 0 and j == 0) else '_')

    # ── LBM conduction (from original benchmark contour CSV) ──
    f_cond = CONTOUR_DIR / f"lbm_contour_{t_us}us.csv"
    if f_cond.exists():
        dc = np.loadtxt(f_cond, delimiter=",", skiprows=1)
        if dc.size > 0:
            # These are already angle-sorted and convex — fine for conduction
            cx, cz = dc[:, 0].mean(), dc[:, 1].mean()
            ang = np.arctan2(dc[:, 1] - cz, dc[:, 0] - cx)
            idx = np.argsort(ang)
            ax.plot(dc[idx, 0], dc[idx, 1], color='#2196F3', lw=1.0, ls=':',
                    alpha=0.5, label='LBM conduction only' if i == 0 else '_')

    ax.set_xlim(50, 150)
    ax.set_ylim(85, 155)
    ax.set_title(f'$t = {t_us}$ $\\mu$s', fontsize=10)
    ax.grid(True, lw=0.3, alpha=0.4)
    if i // 2 == 1:
        ax.set_xlabel(r'X ($\mu$m)')
    if i % 2 == 0:
        ax.set_ylabel(r'Depth Z ($\mu$m)')

fig.legend(*axes[0][0].get_legend_handles_labels(),
           loc='upper right', bbox_to_anchor=(0.98, 0.93),
           fontsize=8, framealpha=0.9, edgecolor='#ccc')
plt.tight_layout(rect=[0, 0, 1, 0.95])

out = CONTOUR_DIR / 'marangoni_validation.png'
fig.savefig(out, dpi=200)
fig.savefig(CONTOUR_DIR / 'marangoni_validation.pdf')
print(f'Saved: {out}')
print(f'Saved: {CONTOUR_DIR / "marangoni_validation.pdf"}')
