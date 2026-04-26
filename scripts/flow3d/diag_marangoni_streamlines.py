#!/usr/bin/env python3
"""Dawn-1: Marangoni convection-roll visualisation.

Read a steady-state LBM VTK frame and produce a y-z plane streamline +
temperature contour PNG. Verifies that the Marangoni-driven convection
rolls (centre→outwards along the surface, then sinking on the sides
and returning along the pool floor) are present.

Usage:
    python diag_marangoni_streamlines.py <vtk> [x_offset_um=-100] [out=marangoni_yz.png]
"""
import sys, os
import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

if len(sys.argv) < 2:
    print(__doc__); sys.exit(1)
path = sys.argv[1]
x_off_um = float(sys.argv[2]) if len(sys.argv) > 2 else -100.0
out = sys.argv[3] if len(sys.argv) > 3 else 'marangoni_yz.png'

m = pv.read(path)
nx, ny, nz = m.dimensions
dx, dy, dz = m.spacing
ox, oy, oz = m.origin

dt_ns = 80.0
v_factor = dx / (dt_ns * 1e-9)
v_scan = 0.8
laser_start_um = 500.0

step = int(os.path.basename(path).split('_')[-1].split('.')[0])
t_us = step * dt_ns * 1e-3
laser_x_um = laser_start_um + v_scan * t_us
print(f"step={step}, t={t_us:.0f}us, laser_x={laser_x_um:.0f}um")

# Pick the y-z slab
x_target_um = laser_x_um + x_off_um
i_slab = int((x_target_um * 1e-6 - ox) / dx)
i_slab = max(0, min(i_slab, nx - 1))
print(f"slab at x={x_target_um:.0f}μm  (i={i_slab})")

T = np.asarray(m.point_data['temperature']).reshape((nx,ny,nz), order='F')
u = np.asarray(m.point_data['velocity']).reshape((nx,ny,nz,3), order='F')
f = np.asarray(m.point_data['fill_level']).reshape((nx,ny,nz), order='F')

T_yz = T[i_slab, :, :].T               # shape (nz, ny) — rows=z (flipped later)
v_y_yz = u[i_slab, :, :, 1].T * v_factor
v_z_yz = u[i_slab, :, :, 2].T * v_factor
fill_yz = f[i_slab, :, :].T

y_um = (oy + np.arange(ny) * dy) * 1e6
z_um = (oz + np.arange(nz) * dz) * 1e6
Y, Z = np.meshgrid(y_um, z_um)

# Mask gas region (fill < 0.5) to make the streamlines stop at free surface
mask_gas = fill_yz < 0.5
v_y_yz[mask_gas] = np.nan
v_z_yz[mask_gas] = np.nan

fig, ax = plt.subplots(figsize=(9, 5))
# Temperature contour
T_plot = T_yz.copy()
T_plot[mask_gas] = np.nan
levels = [300, 500, 800, 1200, 1697, 2200, 2800]
cf = ax.contourf(Y, Z, T_plot, levels=levels, cmap='hot', extend='max')
ax.contour(Y, Z, T_plot, levels=[1697], colors='cyan', linewidths=1.5,
           linestyles='--', label='liquidus')
ax.contour(Y, Z, fill_yz, levels=[0.5], colors='white', linewidths=2)

# Streamlines (only in metal cells)
try:
    ax.streamplot(Y, Z, v_y_yz, v_z_yz, color='black', density=1.5,
                  linewidth=0.8, arrowsize=1.2)
except Exception as e:
    print(f"streamplot failed: {e}; falling back to quiver")
    skip = (slice(None, None, 4), slice(None, None, 4))
    ax.quiver(Y[skip], Z[skip], v_y_yz[skip], v_z_yz[skip],
              color='black', scale=80, width=0.002)

ax.set_xlabel('y [μm]')
ax.set_ylabel('z [μm]')
ax.set_title(f'Marangoni convection roll (slab {x_off_um:+.0f}μm from laser)\n'
             f't={t_us:.0f}μs, v_factor={v_factor:.0f} m/s/LU')
ax.set_aspect('equal')
plt.colorbar(cf, ax=ax, label='Temperature [K]')
plt.tight_layout()
plt.savefig(out, dpi=120, bbox_inches='tight')
print(f"Saved {out}")

# Numerical verification: at the surface (top fill cell), v_y should
# point outward from centerline (Marangoni outward flow when dσ/dT<0).
mid_j = ny // 2
print(f"")
print(f"=== Marangoni signature check (surface only) ===")
print(f"{'y_um':>6} {'z_top':>6} {'T_K':>6} {'v_y':>8} {'v_z':>8}")
for j in range(0, ny, max(1, ny // 12)):
    col = fill_yz[:, j]
    if col.max() < 0.5:
        continue
    k_top = int(np.where(col > 0.5)[0].max())
    if k_top >= nz - 1:
        continue
    print(f"{y_um[j]:>6.0f} {z_um[k_top]:>6.1f} {T_yz[k_top,j]:>6.0f} "
          f"{v_y_yz[k_top,j]:>+8.3f} {v_z_yz[k_top,j]:>+8.3f}")
