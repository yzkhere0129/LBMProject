#!/usr/bin/env python3
"""Extract free-surface z(x,y) from a F3D PolyData VTK file.

F3D writes triangle-mesh free surfaces. We:
  1. Load points + triangles
  2. Project to a regular (x,y) grid by snapping each point to nearest grid cell
  3. For each (x,y) cell, take max z (highest surface point) — this is the
     metal surface height. (For overhangs/keyhole drops, this captures the
     LIQUID surface seen from above, which matches LBM's `top fill_level=0.5`
     convention.)
  4. Apply offset: F3D z=0 is substrate top. LBM z₀=160 μm. Map F3D z → +160.

Output: same `phase1_summary` decision table for cross-comparison.

Usage:
    python extract_f3d_track.py <f3d.vtk> [t_us=2000]
"""
import sys
import numpy as np
import pyvista as pv

if len(sys.argv) < 2:
    print(__doc__); sys.exit(1)
path = sys.argv[1]
t_us = float(sys.argv[2]) if len(sys.argv) > 2 else 2000.0
v_scan = 0.8  # m/s

print(f"Reading {path} (F3D PolyData)...")
m = pv.read(path)
pts = np.asarray(m.points)  # (N, 3) in meters
print(f"Points: {pts.shape[0]}")
print(f"Domain (μm): x∈[{pts[:,0].min()*1e6:.0f}, {pts[:,0].max()*1e6:.0f}], "
      f"y∈[{pts[:,1].min()*1e6:.0f}, {pts[:,1].max()*1e6:.0f}], "
      f"z∈[{pts[:,2].min()*1e6:.0f}, {pts[:,2].max()*1e6:.0f}]")

# Compute laser position. F3D origin: laser_start = 0 (dum3 in prepin).
# At t_us, laser at: x_laser = v_scan * t_us
# In F3D coordinates, laser at +800 μm at t=1 ms.
laser_x_um = v_scan * t_us
print(f"Laser at x = {laser_x_um:.0f} μm (F3D coordinates), t={t_us:.0f} μs")

# Bin into a (x,y) grid at dx=2 μm to match LBM analysis cadence
dx_bin = 2.0  # μm
x_bins = np.arange(pts[:,0].min()*1e6, pts[:,0].max()*1e6 + dx_bin, dx_bin)
y_bins = np.arange(pts[:,1].min()*1e6, pts[:,1].max()*1e6 + dx_bin, dx_bin)
nx_b = len(x_bins) - 1
ny_b = len(y_bins) - 1

x_um = pts[:,0] * 1e6
y_um = pts[:,1] * 1e6
z_um = pts[:,2] * 1e6

ix = np.clip(np.searchsorted(x_bins, x_um) - 1, 0, nx_b - 1)
iy = np.clip(np.searchsorted(y_bins, y_um) - 1, 0, ny_b - 1)

# For each bin, take max z
z_grid = np.full((nx_b, ny_b), -1e9)
for i, j, z in zip(ix, iy, z_um):
    if z > z_grid[i, j]:
        z_grid[i, j] = z
z_grid[z_grid < -1e8] = np.nan
mid_j = ny_b // 2
# F3D points are sparse on exact centerline. Use ±5μm band around y=0.
band_lo = max(0, mid_j - 3)
band_hi = min(ny_b, mid_j + 4)
z_centerline = np.nanmax(z_grid[:, band_lo:band_hi], axis=1)
z_centerline[z_centerline < -1e8] = np.nan
x_centerline = 0.5 * (x_bins[:-1] + x_bins[1:])

# Apply offset: F3D z=0 → LBM z₀=160 μm
z_offset_um = 160.0
z_centerline_off = z_centerline + z_offset_um

print(f"")
print(f"=== F3D centerline trailing-zone (laser at {laser_x_um:.0f} μm) ===")
print(f"  x_off um  z (F3D zero)  z (LBM offset)  Δz vs z₀")
for off in (-50, -100, -150, -200, -250, -300):
    x_query = laser_x_um + off
    if x_query < x_centerline.min() or x_query > x_centerline.max():
        continue
    i = int(np.argmin(np.abs(x_centerline - x_query)))
    z_raw = z_centerline[i]
    z_off = z_centerline_off[i]
    dh = z_raw  # vs F3D z=0 (substrate top)
    print(f"  {off:>+8.0f}  {z_raw:>12.2f}  {z_off:>14.2f}  {dh:>+8.2f}")

# Side ridges: max z over y, in trailing band
behind = (x_centerline > 200) & (x_centerline < laser_x_um - 100)
side_band = z_grid[behind, :]
# Exclude near-centerline (|y - y_mid| > 15 μm)
y_centerline_um = 0.5 * (y_bins[mid_j] + y_bins[mid_j + 1])
y_um_mid = 0.5 * (y_bins[:-1] + y_bins[1:])
side_mask = np.abs(y_um_mid - y_centerline_um) > 15
ridge_max_per_x = np.nanmax(side_band[:, side_mask], axis=1)
print(f"")
print(f"  Behind-laser side-ridge MAX z (vs F3D z=0): {np.nanmax(ridge_max_per_x):+.2f} μm")
print(f"  Behind-laser side-ridge MEAN z           : {np.nanmean(ridge_max_per_x):+.2f} μm")

# Centerline 95%ile in trailing band (extract_track_height convention)
behind_ctr = z_centerline[behind]
behind_ctr_valid = behind_ctr[~np.isnan(behind_ctr)]
if len(behind_ctr_valid) > 0:
    p95 = float(np.percentile(behind_ctr_valid, 95))
    print(f"  Centerline 95%ile z (trailing)            : {p95:+.2f} μm")
    print(f"  Centerline median z (trailing)            : {float(np.nanmedian(behind_ctr_valid)):+.2f} μm")
