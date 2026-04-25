#!/usr/bin/env python3
"""Extract z(x,y) of free surface (fill_level=0.5 isoline) from LBM ImageData VTK
and report whether there is a raised track behind the laser (Marangoni signature).

Usage:
    python extract_track_height.py <lbm.vtk> [interface_z_um=200] [laser_x_um]

Reports:
    z_initial_top   : interface_z (where surface started)
    z_at_laser      : surface z below laser (should be deep, recoil)
    z_max_behind    : max surface z behind laser (raised track signature)
    track_height    : z_max_behind - z_initial_top (should be > 0 if backflow worked)
    fwhm_behind     : fwhm of raised region in x
"""
import sys
import numpy as np
import pyvista as pv

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)
path = sys.argv[1]
z0 = float(sys.argv[2]) if len(sys.argv) > 2 else 200.0  # μm
laser_x_um = float(sys.argv[3]) if len(sys.argv) > 3 else None

m = pv.read(path)
nx, ny, nz = m.dimensions
dx, dy, dz = m.spacing
ox, oy, oz = m.origin
f = np.asarray(m.point_data['fill_level']).reshape((nx, ny, nz), order='F')

# For each (i,j) column, find highest k where fill > 0.5
# z(x,y) = z position of free surface
mask = (f > 0.5)
# argmax of mask along z (returns first match), but we want last (highest)
# trick: reverse along z, then nz-1 - argmax
flipped = mask[:, :, ::-1]
has_metal = flipped.any(axis=2)
k_top = nz - 1 - np.argmax(flipped, axis=2)  # highest k with f>0.5
# Where there's no metal at all in column, set z to NaN
z_surface = oz + k_top * dz
z_surface_um = z_surface * 1e6
z_surface_um[~has_metal] = np.nan

# y-mid line: track centerline
mid_j = ny // 2
z_centerline = z_surface_um[:, mid_j]
x_centerline = (ox + np.arange(nx) * dx) * 1e6

# If laser_x not specified, find it as the deepest point
if laser_x_um is None:
    valid = ~np.isnan(z_centerline)
    if valid.any():
        i_min = np.nanargmin(z_centerline)
        laser_x_um = x_centerline[i_min]
    else:
        laser_x_um = float('nan')

# Behind laser: x < laser_x_um
behind_mask = (x_centerline < laser_x_um) & ~np.isnan(z_centerline)
front_mask  = (x_centerline >= laser_x_um) & ~np.isnan(z_centerline)

z_at_laser = z_centerline[np.argmin(np.abs(x_centerline - laser_x_um))] if not np.isnan(laser_x_um) else float('nan')
z_max_behind = np.nanmax(z_centerline[behind_mask]) if behind_mask.any() else float('nan')
z_p95_behind = np.percentile(z_centerline[behind_mask], 95) if behind_mask.any() else float('nan')
z_min_overall = np.nanmin(z_centerline)
z_max_overall = np.nanmax(z_centerline)

# Raised track height: difference from initial top.
# Use 95%-ile not max — max captures isolated boundary/numerical outliers.
# F3D 95%ile vs LBM 95%ile gives a fair bulk comparison.
track_height_max = z_max_behind - z0 if not np.isnan(z_max_behind) else float('nan')
track_height = z_p95_behind - z0 if not np.isnan(z_p95_behind) else float('nan')

# FWHM of raised region behind laser (z > z0 + 0.5 * (z_max_behind - z0))
if behind_mask.any() and not np.isnan(track_height) and track_height > 0:
    half = z0 + 0.5 * track_height
    raised = (z_centerline > half) & behind_mask
    if raised.any():
        x_raised = x_centerline[raised]
        fwhm = x_raised.max() - x_raised.min()
    else:
        fwhm = 0.0
else:
    fwhm = 0.0

# Summary
print(f"File: {path}")
print(f"Domain: {nx}x{ny}x{nz}, dx={dx*1e6:.2f}μm, origin=({ox*1e6:.0f},{oy*1e6:.0f},{oz*1e6:.0f})μm")
print(f"")
print(f"Initial substrate top:    z₀ = {z0:.1f} μm")
print(f"Laser position:           x_laser = {laser_x_um:.1f} μm")
print(f"")
print(f"Free surface (centerline y=mid):")
print(f"  z_min (deepest, keyhole):   {z_min_overall:.1f} μm   (depth = {z0-z_min_overall:.1f} μm below z₀)")
print(f"  z_max (overall):            {z_max_overall:.1f} μm")
print(f"  z_at_laser:                 {z_at_laser:.1f} μm")
print(f"  z_max_behind_laser:         {z_max_behind:.1f} μm")
print(f"")
print(f"== RAISED TRACK ANALYSIS ==")
print(f"  Track Δh (95%ile, bulk):  {track_height:+.1f} μm    ← preferred metric")
print(f"  Track Δh (max):           {track_height_max:+.1f} μm    (may include outliers)")
print(f"  FWHM of raised region:    {fwhm:.0f} μm")
if not np.isnan(track_height):
    if track_height > 5:
        verdict = "STRONG raised track"
    elif track_height > 1:
        verdict = "WEAK raised track"
    elif track_height > -5:
        verdict = "FLAT"
    else:
        verdict = "DEPRESSED"
    print(f"  Verdict (95%ile-based): {verdict}")
print(f"")
print(f"Flow3D reference @ t=1184μs: Δh_max=+10.1 μm, Δh_95%ile=+9.6 μm")
