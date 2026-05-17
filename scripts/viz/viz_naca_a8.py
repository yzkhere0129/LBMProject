#!/usr/bin/env python3
"""
NACA0012 α=+8° Re=1000 vorticity — von Karman vortex shedding behind airfoil.
Reads the high-resolution phaseE_E1 snapshot.
"""
import sys, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

VTK   = Path("/home/yzk/CompressibleCFD/output_phaseE_E1_final/snap_0050000.vtk")
OUT   = Path("/home/yzk/LBMProject_showcase/gallery/naca_a8_vorticity.png")

# Parse header to know dims, then stream-parse the vectors
nx = ny = nz = 0
dx = 0.0

header_lines = 0
with VTK.open() as fh:
    for ln in fh:
        header_lines += 1
        p = ln.split()
        if p[:1] == ['DIMENSIONS']:
            nx, ny, nz = int(p[1]), int(p[2]), int(p[3])
        elif p[:1] == ['SPACING']:
            dx = float(p[1])
        elif p[:1] == ['VECTORS']:
            break

print(f"Grid {nx}x{ny}x{nz}, dx={dx}")
n = nx * ny * nz
print(f"Total cells {n:,}, parsing velocity vectors ({header_lines} header lines)...")

# Skip header_lines lines, then load n rows × 3 cols
import io
with VTK.open() as fh:
    for _ in range(header_lines):
        fh.readline()
    arr = np.loadtxt(fh, dtype=np.float32, max_rows=n)

v = arr.reshape(nz, ny, nx, 3)
del arr
kz = nz // 2
ux = v[kz, ..., 0]
uy = v[kz, ..., 1]
del v
print("Vectors loaded.")

# Vorticity
duydx = np.zeros_like(uy)
duxdy = np.zeros_like(ux)
duydx[:, 1:-1] = (uy[:, 2:] - uy[:, :-2]) / (2 * dx)
duxdy[1:-1, :] = (ux[2:, :] - ux[:-2, :]) / (2 * dx)
omega_z = duydx - duxdy
print("Vorticity computed.")

# Tight crop around airfoil + wake (LE at chord=10m, TE at ~11m for α=8°)
# Domain is 20m × 8m. Airfoil at (10, 4) world coords.
xmin, xmax = 9.0, 17.0
ymin, ymax = 3.0, 5.0
i0, i1 = int(xmin / dx), int(xmax / dx)
j0, j1 = int(ymin / dx), int(ymax / dx)

# Mask
mask_path = VTK.parent / "mask_zmid.txt"
solid = None
if mask_path.exists():
    with mask_path.open() as fh:
        lines = fh.readlines()
    parts = lines[0].replace('#', '').split()
    pnx = int([p.split('=')[1] for p in parts if p.startswith('nx=')][0])
    pny = int([p.split('=')[1] for p in parts if p.startswith('ny=')][0])
    solid = np.zeros((pny, pnx), dtype=np.uint8)
    for j, ln in enumerate(lines[1:1+pny]):
        solid[j] = np.array(ln.split(), dtype=np.uint8)

fig, ax = plt.subplots(figsize=(15, 4.5), facecolor="#0d0d0d")
ax.set_facecolor("#0d0d0d")
extent = [i0 * dx, i1 * dx, j0 * dx, j1 * dx]

im = ax.imshow(omega_z[j0:j1, i0:i1], origin='lower', extent=extent,
               cmap='RdBu_r', vmin=-0.8, vmax=0.8,
               interpolation='lanczos', aspect='equal')

# Airfoil mask overlay
if solid is not None:
    crop_solid = solid[j0:j1, i0:i1].astype(float)
    crop_solid[crop_solid == 0] = np.nan
    ax.imshow(crop_solid, origin='lower', extent=extent,
              cmap='gray_r', vmin=0.5, vmax=1.0, aspect='equal', alpha=0.95,
              interpolation='nearest')

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_title(
    "NACA0012, Re=1000, α=+8° — D3Q27 Cumulant + Ladd inlet (D/dx=160)  "
    "—  vortex shedding wake",
    color="white", fontsize=12, fontweight="bold", pad=8
)
ax.set_xlabel("x [m]", color="#cccccc", fontsize=10)
ax.set_ylabel("y [m]", color="#cccccc", fontsize=10)
ax.tick_params(colors="#888888", labelsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor("#444444")

cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
cbar.set_label(r"$\omega_z$  [1/s]", color="white", fontsize=10)
cbar.ax.yaxis.set_tick_params(color="#888888", labelcolor="#aaaaaa", labelsize=8)
cbar.outline.set_edgecolor("#444444")

fig.savefig(OUT, dpi=180, bbox_inches="tight",
            facecolor="#0d0d0d", edgecolor="none")
print(f"Saved: {OUT}")
