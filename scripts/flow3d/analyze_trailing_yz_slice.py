#!/usr/bin/env python3
"""Extract a y-z slab 200 μm behind the laser at the final frame.

Reports:
  - z(y) profile (LBM ny_max along centerline + side ridges)
  - velocity_y(y), velocity_z(y) at the surface
  - reveals 'where' side ridges peak (physical y location vs y_mid)

Usage: python analyze_trailing_yz_slice.py <vtk> [z0_um=160] [v_scan=0.8]
"""
import sys
import numpy as np
import pyvista as pv

if len(sys.argv) < 2:
    print(__doc__); sys.exit(1)
path = sys.argv[1]
z0_um = float(sys.argv[2]) if len(sys.argv) > 2 else 160.0
v_scan = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
laser_start_um = 500.0

m = pv.read(path)
nx, ny, nz = m.dimensions
dx, dy, dz = m.spacing
ox, oy, oz = m.origin
dt_ns = 80.0
v_factor = dx / (dt_ns * 1e-9)

step = int(path.split('_')[-1].split('.')[0])
t_us = step * dt_ns * 1e-3
laser_x_um = laser_start_um + v_scan * t_us
print(f"step={step}, t={t_us:.0f}us, laser_x={laser_x_um:.0f}um")

f = np.asarray(m.point_data['fill_level']).reshape((nx,ny,nz), order='F')
T = np.asarray(m.point_data['temperature']).reshape((nx,ny,nz), order='F')
u = np.asarray(m.point_data['velocity']).reshape((nx,ny,nz,3), order='F')

# Find x_slab 200 μm behind laser
x_slab_um = laser_x_um - 200
i_slab = int((x_slab_um * 1e-6 - ox) / dx)
i_slab = max(0, min(i_slab, nx - 1))
print(f"y-z slab at x = {x_slab_um:.0f} μm (i={i_slab})")
print()

# z(y) profile of free surface in this slab
flipped = (f[i_slab,:,:] > 0.5)[:, ::-1]
has_metal = flipped.any(axis=1)
k_top = nz - 1 - np.argmax(flipped, axis=1)
z_surf_um = (oz + k_top * dz) * 1e6
z_surf_um = np.where(has_metal, z_surf_um, np.nan)

y_um = (oy + np.arange(ny) * dy) * 1e6

print(f"{'y_um':>6} {'z_um':>6} {'Δh':>5} {'T_K':>5} {'v_y_LU':>8} {'v_z_LU':>8} {'v_z_mps':>8}")
print('-' * 60)
mid_j = ny // 2
y_mid_um = y_um[mid_j]
for j in range(0, ny, max(1, ny // 30)):  # sample ~30 points
    if not has_metal[j]:
        continue
    k = k_top[j]
    z = z_surf_um[j]
    dh = z - z0_um
    T_K = T[i_slab, j, k]
    vy = u[i_slab, j, k, 1]
    vz = u[i_slab, j, k, 2]
    vz_mps = vz * v_factor
    yc = y_um[j] - y_mid_um
    print(f"{y_um[j]:>6.0f} {z:>6.1f} {dh:>+5.1f} {T_K:>5.0f} {vy:>+8.4f} {vz:>+8.4f} {vz_mps:>+8.3f}")

# Find where side-ridge peaks: max z (above z0) for j in side band
side_mask = np.abs(y_um - y_mid_um) > 15
valid = has_metal & side_mask
if valid.any():
    z_side = np.where(valid, z_surf_um, -np.inf)
    j_peak = int(np.argmax(z_side))
    z_peak = z_surf_um[j_peak]
    y_peak_offset = y_um[j_peak] - y_mid_um
    print()
    print(f"Side-ridge peak: y={y_um[j_peak]:.0f} μm "
          f"(offset {y_peak_offset:+.0f} from centerline), "
          f"z={z_peak:.1f} μm (Δh={z_peak-z0_um:+.1f})")
