#!/usr/bin/env python3
"""Side-by-side x-y top-down view: F3D vs LBM melt-pool, single time."""
import sys
import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if len(sys.argv) < 4:
    print("usage: viz_pool_compare.py <flow3d.vtk> <lbm.vtk> <out.png> [offset=1e-4,7.5e-5,7.75e-5]")
    sys.exit(1)
f3d_path, lbm_path, out_path = sys.argv[1:4]
offset = np.array([float(x) for x in (sys.argv[4] if len(sys.argv) > 4 else '1.0e-4,7.5e-5,7.75e-5').split(',')])
T_liq = 1697.15

# F3D PolyData
f3d = pv.read(f3d_path)
f3d_pts = np.asarray(f3d.points) + offset  # in LBM frame, meters
f3d_T = np.asarray(f3d.point_data['Temperature'])
f3d_above = f3d_T >= T_liq

# LBM ImageData
lbm = pv.read(lbm_path)
nx, ny, nz = lbm.dimensions
dx, dy, dz = lbm.spacing
ox, oy, oz = lbm.origin
T_lbm = np.asarray(lbm.point_data['temperature']).reshape((nx, ny, nz), order='F')
f_lbm = np.asarray(lbm.point_data['fill_level']).reshape((nx, ny, nz), order='F')
liquid_lbm = (T_lbm >= T_liq) & (f_lbm > 0.5)

# Top-down (max-T projection) view
T_max_xy_lbm = T_lbm.max(axis=2)  # collapse z
liq_xy_lbm = liquid_lbm.any(axis=2)

# F3D top-down: scatter pts, color by T (only z-near-top)
top_z = lbm.origin[2] + (nz-1)*dz
near_top_f3d = (f3d_pts[:, 2] > top_z - 30e-6) & f3d_above
sub = f3d_pts[near_top_f3d]
subT = f3d_T[near_top_f3d]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (0,0): LBM top-down T_max
ax = axes[0, 0]
extent = [ox*1e6, (ox+(nx-1)*dx)*1e6, oy*1e6, (oy+(ny-1)*dy)*1e6]
im = ax.imshow(T_max_xy_lbm.T, extent=extent, origin='lower', aspect='equal',
               cmap='hot', vmin=300, vmax=4500)
ax.contour(np.linspace(extent[0], extent[1], nx),
           np.linspace(extent[2], extent[3], ny),
           T_max_xy_lbm.T, levels=[T_liq], colors='cyan', linewidths=1.0)
ax.set(title=f'LBM top-down T_max [K]\n(cyan = T_liq)', xlabel='x [μm]', ylabel='y [μm]')
plt.colorbar(im, ax=ax)

# (0,1): F3D top-down (scatter)
ax = axes[0, 1]
if len(sub):
    sc = ax.scatter(sub[:, 0]*1e6, sub[:, 1]*1e6, c=subT, s=2, cmap='hot', vmin=300, vmax=4500)
    plt.colorbar(sc, ax=ax)
ax.set(title='Flow3D surface points T ≥ T_liq (top 30 μm of LBM domain)', xlabel='x [μm]', ylabel='y [μm]')
ax.set_xlim(extent[0], extent[1])
ax.set_ylim(extent[2], extent[3])
ax.set_aspect('equal')

# (1,0): LBM mid-y-slice T (x-z view)
ax = axes[1, 0]
mid_y = ny // 2
T_xz = T_lbm[:, mid_y, :]
extent_xz = [ox*1e6, (ox+(nx-1)*dx)*1e6, oz*1e6, (oz+(nz-1)*dz)*1e6]
im = ax.imshow(T_xz.T, extent=extent_xz, origin='lower', aspect='equal',
               cmap='hot', vmin=300, vmax=4500)
ax.contour(np.linspace(extent_xz[0], extent_xz[1], nx),
           np.linspace(extent_xz[2], extent_xz[3], nz),
           T_xz.T, levels=[T_liq], colors='cyan', linewidths=1.0)
ax.set(title=f'LBM x-z slice y=mid, T [K]', xlabel='x [μm]', ylabel='z [μm]')
plt.colorbar(im, ax=ax)

# (1,1): F3D x-z view (project sub pts onto x-z near y=mid)
ax = axes[1, 1]
y_mid_lbm = (oy + (ny//2)*dy) * 1e6
near_y = np.abs(f3d_pts[:, 1]*1e6 - y_mid_lbm) < 10  # within 10 μm of y-mid
sub_xz = f3d_pts[near_y]
subT_xz = f3d_T[near_y]
if len(sub_xz):
    sc = ax.scatter(sub_xz[:, 0]*1e6, sub_xz[:, 2]*1e6, c=subT_xz, s=3, cmap='hot', vmin=300, vmax=4500)
    plt.colorbar(sc, ax=ax)
ax.set(title=f'Flow3D x-z slice (y within 10μm of mid), T [K]', xlabel='x [μm]', ylabel='z [μm]')
ax.set_xlim(extent_xz[0], extent_xz[1])
ax.set_ylim(extent_xz[2], extent_xz[3])
ax.set_aspect('equal')

plt.suptitle(f'F3D: {f3d_path.split("/")[-1]}  vs  LBM: {lbm_path.split("/")[-1]}', fontsize=11)
plt.tight_layout()
plt.savefig(out_path, dpi=110, bbox_inches='tight')
print(f'Saved {out_path}')
