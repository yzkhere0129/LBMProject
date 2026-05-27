"""Visualize a 3D F-18 aero run from its velocity VTK.

Reads the legacy ASCII STRUCTURED_POINTS snap_*.vtk (VECTORS velocity) and
plots two orthogonal slices through the aircraft:
  - top view  (xy plane at mid-height): speed + spanwise vortex shedding
  - side view (xz plane at mid-span):   speed + the body's low-speed wake

The aircraft planform is overlaid on the top view from mask_zmid.txt.
Flow is +x; nose is upstream (low x).

Usage:
  plot_f18_3d.py OUTPUT_DIR [--step N] [--out FIG.png]
  (default: latest snap_*.vtk in OUTPUT_DIR -> OUTPUT_DIR/wake.png)
"""
import argparse
import glob
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_vtk_header(path):
    """Return (nx, ny, nz, dx, n_skip) — lines to skip before numeric data."""
    dims = spacing = None
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if line.startswith("DIMENSIONS"):
                dims = [int(v) for v in line.split()[1:4]]
            elif line.startswith("SPACING"):
                spacing = float(line.split()[1])
            elif line.startswith("VECTORS"):
                return (*dims, spacing, i + 1)   # data starts on next line
    raise RuntimeError("no VECTORS block found")


def read_velocity(path):
    nx, ny, nz, dx, n_skip = parse_vtk_header(path)
    npts = nx * ny * nz
    try:
        import pandas as pd
        arr = pd.read_csv(path, skiprows=n_skip, nrows=npts, sep=r"\s+",
                          header=None, dtype="float32").to_numpy()
    except ImportError:
        arr = np.loadtxt(path, skiprows=n_skip, max_rows=npts, dtype="float32")
    # VTK structured points iterate x fastest -> (nz, ny, nx, 3).
    vel = arr.reshape(nz, ny, nx, 3)
    return vel, nx, ny, nz, dx


def load_mask(path, ny, nx):
    if not os.path.exists(path):
        return None
    m = np.loadtxt(path, comments="#", dtype=np.uint8)
    return m if m.shape == (ny, nx) else None


ap = argparse.ArgumentParser()
ap.add_argument("output_dir")
ap.add_argument("--step", type=int, default=-1)
ap.add_argument("--out", default=None)
a = ap.parse_args()

snaps = sorted(glob.glob(os.path.join(a.output_dir, "snap_*.vtk")))
if not snaps:
    sys.exit(f"no snap_*.vtk in {a.output_dir}")
if a.step >= 0:
    snaps = [s for s in snaps if f"{a.step:07d}" in s] or snaps[-1:]
vtk = snaps[-1]
print(f"reading {vtk} ...")
vel, nx, ny, nz, dx = read_velocity(vtk)
speed = np.linalg.norm(vel, axis=3)            # (nz, ny, nx)
U_inf = 0.05                                    # LU freestream (u_max_lu)

kz, jy = nz // 2, ny // 2
extent_xy = [0, nx * dx, 0, ny * dx]
extent_xz = [0, nx * dx, 0, nz * dx]

# Top view (xy, mid-height): speed + spanwise vorticity wz = dv/dx - du/dy.
u_xy, v_xy = vel[kz, :, :, 0], vel[kz, :, :, 1]
wz = np.gradient(v_xy, dx, axis=1) - np.gradient(u_xy, dx, axis=0)
# Side view (xz, mid-span): speed + wy = dux/dz - duz/dx.
u_xz, w_xz = vel[:, jy, :, 0], vel[:, jy, :, 2]
wy = np.gradient(u_xz, dx, axis=0) - np.gradient(w_xz, dx, axis=1)

mask = load_mask(os.path.join(a.output_dir, "mask_zmid.txt"), ny, nx)


def vlevels(w):
    """Symmetric contour levels from the 97th percentile of |vorticity|."""
    s = float(np.nanpercentile(np.abs(w), 97.0))
    return [-s, -0.4 * s, 0.4 * s, s] if s > 1e-9 else None

fig, ax = plt.subplots(2, 1, figsize=(13, 9))
step_tag = re.search(r"(\d+)\.vtk", vtk).group(1)

# --- top view ---
im0 = ax[0].imshow(speed[kz] / U_inf, origin="lower", extent=extent_xy,
                   cmap="turbo", vmin=0, vmax=1.6, aspect="equal")
lv = vlevels(wz)
if lv:
    ax[0].contour(wz, levels=lv, extent=extent_xy,
                  colors=["blue", "cyan", "orange", "red"], linewidths=0.6, alpha=0.7)
if mask is not None:
    ax[0].contour(mask, levels=[0.5], extent=extent_xy, colors="k", linewidths=1.2)
ax[0].set_title(f"TOP (xy, mid-height)  step {step_tag}  —  |u|/U∞ + ω_z")
ax[0].set_xlabel("x (chord)  →  FLOW +x"); ax[0].set_ylabel("y span")
fig.colorbar(im0, ax=ax[0], shrink=0.8, label="|u|/U∞")

# --- side view ---
im1 = ax[1].imshow(speed[:, jy, :] / U_inf, origin="lower", extent=extent_xz,
                   cmap="turbo", vmin=0, vmax=1.6, aspect="equal")
lv = vlevels(wy)
if lv:
    ax[1].contour(wy, levels=lv, extent=extent_xz,
                  colors=["blue", "cyan", "orange", "red"], linewidths=0.6, alpha=0.7)
ax[1].set_title("SIDE (xz, mid-span)  —  |u|/U∞ + ω_y")
ax[1].set_xlabel("x (chord)  →  FLOW +x"); ax[1].set_ylabel("z vertical")
fig.colorbar(im1, ax=ax[1], shrink=0.8, label="|u|/U∞")

out = a.out or os.path.join(a.output_dir, "wake.png")
plt.tight_layout(); plt.savefig(out, dpi=110)
print(f"wrote {out}")
print(f"  |u|/U∞ max={speed.max()/U_inf:.2f}  mean={speed.mean()/U_inf:.3f}  "
      f"solid-ish cells (|u|<0.02 U∞)={(speed < 0.02*U_inf).sum()}")
