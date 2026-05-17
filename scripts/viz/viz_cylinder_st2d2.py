#!/usr/bin/env python3
"""
Schäfer-Turek 2D-2 cylinder vorticity — striking visual of Karman vortex
street. Reads ASCII VTK velocity field, computes ω_z, renders a tight
crop around cylinder with the wake.
"""
import sys, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

VTK   = Path("/home/yzk/CompressibleCFD/output_BEST_cumulant_d20_st2d2/snap_0030000.vtk")
OUT   = Path("/home/yzk/LBMProject_showcase/gallery/cylinder_vorticity.png")

# Parse ASCII VTK ImageData with VECTORS velocity
with VTK.open() as fh:
    lines = fh.readlines()
nx = ny = nz = 0
dx = 0.0
data_start = -1
for i, ln in enumerate(lines):
    p = ln.split()
    if p[:1] == ['DIMENSIONS']:
        nx, ny, nz = int(p[1]), int(p[2]), int(p[3])
    elif p[:1] == ['SPACING']:
        dx = float(p[1])
    elif p[:1] == ['VECTORS']:
        data_start = i + 1
        break
n = nx * ny * nz
v = np.empty((n, 3), dtype=np.float32)
for k, ln in enumerate(lines[data_start:data_start + n]):
    pp = ln.split()
    v[k, 0] = float(pp[0]); v[k, 1] = float(pp[1]); v[k, 2] = float(pp[2])
v = v.reshape(nz, ny, nx, 3)
ux = v[nz // 2, ..., 0]
uy = v[nz // 2, ..., 1]

# Vorticity ω_z = ∂v/∂x − ∂u/∂y
duydx = np.zeros_like(uy)
duxdy = np.zeros_like(ux)
duydx[:, 1:-1] = (uy[:, 2:] - uy[:, :-2]) / (2 * dx)
duxdy[1:-1, :] = (ux[2:, :] - ux[:-2, :]) / (2 * dx)
omega_z = duydx - duxdy

# Convert to LU units consistent with simulation: dx_phys=0.005m, c=Dia 0.1m
# Show in 1/s. For DFG conventions U_mean = 1 m/s, D = 0.1, so charactaristic
# ω scale is U/D = 10 1/s. Use vmin/vmax = ±15 to show fine structure.

# Tight crop: cylinder at (0.2, 0.2) in 2.2×0.41 domain; show wake to x=1.5
xmin_crop, xmax_crop = 0.05, 1.5     # m
ymin_crop, ymax_crop = 0.0,  0.41
i0, i1 = int(xmin_crop / dx), int(xmax_crop / dx)
j0, j1 = int(ymin_crop / dx), min(ny, int(ymax_crop / dx))

fig, ax = plt.subplots(figsize=(14, 4), facecolor="#0d0d0d")
ax.set_facecolor("#0d0d0d")
extent = [i0 * dx, i1 * dx, j0 * dx, j1 * dx]
im = ax.imshow(omega_z[j0:j1, i0:i1], origin='lower', extent=extent,
               cmap='RdBu_r', vmin=-3, vmax=3, interpolation='lanczos',
               aspect='equal')
# Cylinder marker
from matplotlib.patches import Circle
ax.add_patch(Circle((0.2, 0.2), 0.05, fill=True, facecolor='#cccccc',
                    edgecolor='black', lw=1.0, zorder=5))

ax.set_xlim(xmin_crop, xmax_crop)
ax.set_ylim(ymin_crop, ymax_crop)
ax.set_title(
    "Schäfer-Turek 2D-2 cylinder, Re=100, D3Q27 Cumulant + Ladd inlet  "
    "—  $C_d$ = 3.156 (DFG band [3.22, 3.24], −2.3%)",
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
