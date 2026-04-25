#!/usr/bin/env python3
"""Plot raised-track + keyhole-depth + v_max time evolution from a series of
LBM VTK frames, optionally overlaid with Flow3D reference values.

Usage: plot_track_evolution.py <output_dir> <out.png> [interface_z_um=160]
"""
import sys, glob, os
import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print(__doc__); sys.exit(1)
out_dir = sys.argv[1]
out_png = sys.argv[2]
z0 = float(sys.argv[3]) if len(sys.argv) > 3 else 160.0  # μm
v_scan_mps = 0.8
laser_start_um = 500.0  # sim_line_scan_316L M16d
dt_ns = 80.0  # sim_line_scan_316L M16d

vtks = sorted(glob.glob(os.path.join(out_dir, 'line_scan_*.vtk')))
print(f'Found {len(vtks)} frames')

t_us, kh_depth, raised_h, fwhm, vmag_max = [], [], [], [], []

for path in vtks:
    step = int(path.split('_')[-1].split('.')[0])
    t = step * dt_ns * 1e-3  # μs
    t_us.append(t)

    laser_x = laser_start_um + v_scan_mps * t
    m = pv.read(path)
    nx, ny, nz = m.dimensions
    dx, dy, dz = m.spacing
    ox, oy, oz = m.origin
    f = np.asarray(m.point_data['fill_level']).reshape((nx, ny, nz), order='F')

    flipped = (f > 0.5)[:, :, ::-1]
    has_metal = flipped.any(axis=2)
    k_top = nz - 1 - np.argmax(flipped, axis=2)
    z_surface_um = (oz + k_top * dz) * 1e6
    z_surface_um[~has_metal] = np.nan

    mid_j = ny // 2
    z_centerline = z_surface_um[:, mid_j]
    x_centerline = (ox + np.arange(nx) * dx) * 1e6

    behind = (x_centerline < laser_x) & ~np.isnan(z_centerline)
    z_min = np.nanmin(z_centerline) if has_metal.any() else np.nan
    z_max_behind = np.nanmax(z_centerline[behind]) if behind.any() else np.nan

    kh_depth.append(z0 - z_min if not np.isnan(z_min) else np.nan)
    raised_h.append(z_max_behind - z0 if not np.isnan(z_max_behind) else np.nan)
    if behind.any() and not np.isnan(z_max_behind) and (z_max_behind > z0):
        half = z0 + 0.5 * (z_max_behind - z0)
        raised_mask = (z_centerline > half) & behind
        if raised_mask.any():
            xr = x_centerline[raised_mask]
            fwhm.append(xr.max() - xr.min())
        else:
            fwhm.append(0.0)
    else:
        fwhm.append(0.0)

    if 'velocity' in m.point_data.keys():
        u = np.asarray(m.point_data['velocity'])
        vmag = np.sqrt(np.sum(u*u, axis=1))
        # FluidLBM stores velocity in *lattice units* in VTK output.
        # Convert to physical: v_phys = v_LU * dx/dt
        v_LU_max = float(vmag.max())
        v_phys_max = v_LU_max * dx / (dt_ns * 1e-9)
        vmag_max.append(v_phys_max)
    else:
        vmag_max.append(np.nan)

t_us = np.array(t_us); kh_depth = np.array(kh_depth)
raised_h = np.array(raised_h); fwhm = np.array(fwhm); vmag_max = np.array(vmag_max)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(f'M16d LBM (800×75×100 @ dx=2μm) free-surface evolution vs Flow3D', fontsize=13)

ax = axes[0, 0]
ax.plot(t_us, kh_depth, 'r-o', lw=2, markersize=5, label='LBM keyhole depth')
ax.axhline(y=78, color='k', linestyle='--', alpha=0.7, label='Flow3D ≈ 78 μm')
ax.set(xlabel='t [μs]', ylabel='Keyhole depth [μm]', title='Keyhole depth (z₀ − z_min)')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(t_us, raised_h, 'b-s', lw=2, markersize=5, label='LBM raised track Δh')
ax.axhline(y=15.3, color='k', linestyle='--', alpha=0.7, label='Flow3D Δh = +15.3 μm')
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax.set(xlabel='t [μs]', ylabel='Raised track Δh [μm]',
       title='Raised track height behind laser (Marangoni 翻滚+堆积)')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.plot(t_us, fwhm, 'g-^', lw=2, markersize=5)
ax.set(xlabel='t [μs]', ylabel='FWHM [μm]', title='FWHM of raised region')
ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.plot(t_us, vmag_max, 'm-D', lw=2, markersize=5, label='LBM v_max (3D)')
ax.axhspan(1.0, 4.0, color='orange', alpha=0.2, label='Flow3D / Khairallah 1-4 m/s')
ax.set(xlabel='t [μs]', ylabel='v_max [m/s]', title='v_max evolution')
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(out_png, dpi=110, bbox_inches='tight')
print(f'Saved {out_png}')
print(f'')
print(f'Final state @ t={t_us[-1]:.0f}μs:')
print(f'  keyhole depth  = {kh_depth[-1]:.1f} μm  (Flow3D 78 μm)')
print(f'  raised Δh      = {raised_h[-1]:+.1f} μm  (Flow3D +15.3 μm)')
print(f'  FWHM           = {fwhm[-1]:.0f} μm')
print(f'  v_max          = {vmag_max[-1]:.2f} m/s  (Flow3D range 1-4 m/s)')
