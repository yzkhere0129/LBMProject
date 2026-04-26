#!/usr/bin/env python3
"""Phase-1 single-frame decision-ready summary for LBM LPBF VTK.

Computes four numerical metrics for one ImageData STRUCTURED_POINTS VTK frame:
  1) Trailing-zone z(y) profile at x = laser_x - 200 um
       reports: centerline z, side-ridge max z (max over y outside |y-y_mid|<=15um),
                groove depth (z0 - centerline_z)
  2) v_z along centerline behind laser at x_offset = -50, -100, ..., -300 um
       per-cell top-of-fill_level location, v_z in PHYSICAL m/s (LU * dx/dt)
  3) Side-ridge max z in band x in [laser_x-500, laser_x-100] (max over y per x)
  4) Center-line raised-track Δh using extract_track_height.py convention
       95%ile over centerline z in band [laser_start+200, laser_x-100] minus z0

Output is plain stdout text — non-interactive, no plots.

The script tolerates frames with no melt pool yet (early steps): each metric
falls back to NaN with a clear note rather than crashing.

Usage:
    python phase1_summary.py <vtk_frame> [z0_um=160] [laser_start_um=500] [v_scan=0.8]

Defaults match the phase1 line_scan_316L app:
  z0_um           = 160   (substrate top, interface_z=80 cells × dx=2um)
  laser_start_um  = 500   (scan starts at x=500um)
  v_scan          = 0.8   (m/s)
  dt              = 80 ns (used to convert LU velocity → m/s)
  dx              = read from VTK
  Conversion factor v_factor = dx / dt  (= 25 m/s per LU at dx=2um, dt=80ns)
"""
import sys
import numpy as np
import pyvista as pv

# ----- Args -----
if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)
path = sys.argv[1]
z0_um           = float(sys.argv[2]) if len(sys.argv) > 2 else 160.0
laser_start_um  = float(sys.argv[3]) if len(sys.argv) > 3 else 500.0
v_scan          = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8   # m/s
dt_ns           = 80.0                                                # ns

# ----- Read -----
m = pv.read(path)
nx, ny, nz = m.dimensions
dx, dy, dz = m.spacing
ox, oy, oz = m.origin
v_factor = dx / (dt_ns * 1e-9)   # m/s per LU

f    = np.asarray(m.point_data['fill_level']).reshape((nx, ny, nz), order='F')
T    = np.asarray(m.point_data['temperature']).reshape((nx, ny, nz), order='F')
fl   = np.asarray(m.point_data['liquid_fraction']).reshape((nx, ny, nz), order='F')
u    = np.asarray(m.point_data['velocity']).reshape((nx, ny, nz, 3), order='F')

# Step / time / laser_x from filename like '..._005000.vtk'
try:
    step = int(path.rsplit('_', 1)[-1].split('.')[0])
except ValueError:
    step = -1
t_us = step * dt_ns * 1e-3 if step >= 0 else float('nan')
laser_x_um = laser_start_um + v_scan * t_us if step >= 0 else float('nan')

# ----- Free-surface z(x,y) -----
mask = (f > 0.5)
flipped = mask[:, :, ::-1]
has_metal = flipped.any(axis=2)
k_top = nz - 1 - np.argmax(flipped, axis=2)
z_surf_um = (oz + k_top * dz) * 1e6
z_surf_um = np.where(has_metal, z_surf_um, np.nan)

x_um = (ox + np.arange(nx) * dx) * 1e6
y_um = (oy + np.arange(ny) * dy) * 1e6
y_mid = ny // 2
y_mid_um = y_um[y_mid]

# ----- 1) Trailing-zone z(y) at x = laser_x - 200 um -----
def slab_z_profile(x_target_um):
    """At slab x≈x_target_um, return z_surf along y (μm) and side-ridge max z (μm)."""
    if np.isnan(x_target_um):
        return None, None, None, None
    if x_target_um < x_um[0] or x_target_um > x_um[-1]:
        return None, None, None, None
    i = int(np.clip(round((x_target_um * 1e-6 - ox) / dx), 0, nx - 1))
    z_y = z_surf_um[i, :]
    if not np.any(np.isfinite(z_y)):
        return None, None, None, None
    z_center = z_y[y_mid]
    # Side ridge: |y - y_mid| > 15 μm  (outside ~7 cells on either side)
    side_mask = np.abs(y_um - y_mid_um) > 15.0
    valid_side = side_mask & np.isfinite(z_y)
    z_side_max = float(np.nanmax(z_y[valid_side])) if valid_side.any() else float('nan')
    groove_depth = (z0_um - z_center) if np.isfinite(z_center) else float('nan')
    return z_center, z_side_max, groove_depth, i

# ----- 2) v_z along centerline behind laser -----
def vz_centerline_offsets(offsets_um):
    """For each x_offset, take cell at (i, y_mid, k_top); return v_z in m/s."""
    rows = []
    for off in offsets_um:
        x_t = laser_x_um + off
        if np.isnan(x_t) or x_t < x_um[0] or x_t > x_um[-1]:
            rows.append((off, x_t, float('nan'), float('nan'), float('nan')))
            continue
        i = int(np.clip(round((x_t * 1e-6 - ox) / dx), 0, nx - 1))
        if not has_metal[i, y_mid]:
            rows.append((off, x_t, float('nan'), float('nan'), float('nan')))
            continue
        k = int(k_top[i, y_mid])
        z_top = (oz + k * dz) * 1e6
        vz_LU = float(u[i, y_mid, k, 2])
        vz_mps = vz_LU * v_factor
        rows.append((off, x_t, z_top, vz_LU, vz_mps))
    return rows

# ----- 3) Side-ridge max z behind laser (per-x max over y) -----
def side_ridge_profile(x_lo_um, x_hi_um):
    """In band [x_lo, x_hi], for each x return max z over y with |y-y_mid|>15um.
    Returns (x_arr, z_max_arr, peak_x, peak_z)."""
    if np.isnan(x_lo_um) or np.isnan(x_hi_um):
        return None, None, None, None
    i_lo = int(np.clip(round((x_lo_um * 1e-6 - ox) / dx), 0, nx - 1))
    i_hi = int(np.clip(round((x_hi_um * 1e-6 - ox) / dx), 0, nx - 1))
    if i_hi <= i_lo:
        return None, None, None, None
    side_mask = np.abs(y_um - y_mid_um) > 15.0
    z_band = z_surf_um[i_lo:i_hi + 1, :]
    z_band_side = np.where(side_mask[None, :], z_band, np.nan)
    with np.errstate(all='ignore'):
        z_max_per_x = np.nanmax(z_band_side, axis=1)
    x_band = x_um[i_lo:i_hi + 1]
    if np.all(np.isnan(z_max_per_x)):
        return x_band, z_max_per_x, float('nan'), float('nan')
    i_peak = int(np.nanargmax(z_max_per_x))
    return x_band, z_max_per_x, float(x_band[i_peak]), float(z_max_per_x[i_peak])

# ----- 4) Centerline raised-track Δh (95%ile, extract_track_height.py convention) -----
def centerline_track_dh():
    z_centerline = z_surf_um[:, y_mid]
    if np.isnan(laser_x_um):
        return float('nan'), float('nan'), float('nan')
    behind = ((x_um > laser_start_um + 200) &
              (x_um < laser_x_um - 100) &
              np.isfinite(z_centerline))
    if not behind.any():
        return float('nan'), float('nan'), float('nan')
    z_p95 = float(np.percentile(z_centerline[behind], 95))
    z_max = float(np.nanmax(z_centerline[behind]))
    dh_p95 = z_p95 - z0_um
    dh_max = z_max - z0_um
    return dh_p95, dh_max, z_p95

# ----- Run all four -----
z_center_x200, z_side_x200, groove_x200, i_x200 = slab_z_profile(
    laser_x_um - 200.0 if not np.isnan(laser_x_um) else float('nan'))

vz_offsets = [-50, -100, -150, -200, -250, -300]
vz_rows = vz_centerline_offsets(vz_offsets)

x_band, z_max_per_x, peak_x, peak_z = side_ridge_profile(
    (laser_x_um - 500.0) if not np.isnan(laser_x_um) else float('nan'),
    (laser_x_um - 100.0) if not np.isnan(laser_x_um) else float('nan'))

dh_p95, dh_max, z_p95 = centerline_track_dh()

# Find vz at -150 μm specifically (the headline metric)
vz_150 = float('nan')
for off, x_t, z_t, vz_LU, vz_mps in vz_rows:
    if off == -150:
        vz_150 = vz_mps
        break

# ----- Print -----
print("=" * 78)
print(f"PHASE-1 LBM FRAME SUMMARY")
print("=" * 78)
print(f"File           : {path}")
print(f"Grid           : {nx} x {ny} x {nz}, dx={dx*1e6:.2f} um")
print(f"Step / t       : {step}  /  t = {t_us:.1f} us")
print(f"Laser x        : {laser_x_um:.1f} um  (start={laser_start_um:.0f}, v={v_scan} m/s)")
print(f"z0 (substrate) : {z0_um:.1f} um")
print(f"v_factor       : {v_factor:.3f} m/s per LU  (dx={dx*1e6:.2f}um / dt={dt_ns:.0f}ns)")
print(f"Has any metal  : {bool(has_metal.any())}")
print(f"Melt pool cells (T>1697 & f>0.5): {int(((T>1697) & (f>0.5)).sum())}")
print()

print("-" * 78)
print(f"[1] TRAILING-ZONE PROFILE at x = laser_x - 200 um")
print("-" * 78)
if z_center_x200 is None:
    print("  Slab unavailable (out of domain or no metal column at this x).")
else:
    print(f"  x_slab          : {(ox + i_x200*dx)*1e6:.1f} um")
    print(f"  z_centerline    : {z_center_x200:.2f} um")
    print(f"  z_side_max      : {z_side_x200:.2f} um  (|y-y_mid|>15um)")
    print(f"  groove_depth    : {groove_x200:+.2f} um  (z0 - z_centerline)")
    print(f"  side-ridge dh   : {z_side_x200 - z0_um:+.2f} um  (vs z0)")
    print(f"  F3D ref         : center +4 um, side ridges +4 um")

print()
print("-" * 78)
print(f"[2] v_z ALONG CENTERLINE BEHIND LASER (top-of-fill, physical m/s)")
print("-" * 78)
print(f"  {'x_off um':>8} {'x_um':>7} {'z_top um':>8} {'v_z LU':>10} {'v_z m/s':>10}")
for off, x_t, z_t, vz_LU, vz_mps in vz_rows:
    if np.isnan(z_t):
        print(f"  {off:>8} {x_t:>7.1f}  {'--':>8} {'--':>10} {'--':>10}")
    else:
        print(f"  {off:>8} {x_t:>7.1f} {z_t:>8.1f} {vz_LU:>10.4f} {vz_mps:>10.3f}")

print()
print("-" * 78)
print(f"[3] SIDE-RIDGE MAX z BEHIND LASER (max over y, |y-y_mid|>15um)")
print(f"    Band x = [laser_x - 500, laser_x - 100] um")
print("-" * 78)
if x_band is None or np.all(np.isnan(z_max_per_x)):
    print("  No side ridges detected (no metal columns in band or NaN profile).")
else:
    print(f"  Peak side-ridge z   : {peak_z:.2f} um at x = {peak_x:.1f} um")
    print(f"  Peak vs z0 (Δh)     : {peak_z - z0_um:+.2f} um")
    # Robust sample at -100/-200/-300
    for off in (-100, -200, -300):
        xt = laser_x_um + off
        idx = np.argmin(np.abs(x_band - xt))
        if 0 <= idx < len(x_band):
            zv = z_max_per_x[idx]
            zd = (zv - z0_um) if np.isfinite(zv) else float('nan')
            zv_s = f"{zv:6.2f}" if np.isfinite(zv) else "  --  "
            zd_s = f"{zd:+6.2f}" if np.isfinite(zd) else "  --  "
            print(f"  x_off={off:>4} um  -> z_side_max={zv_s} um, Δh={zd_s} um")

print()
print("-" * 78)
print(f"[4] CENTERLINE RAISED-TRACK Δh (95%ile, extract_track_height convention)")
print("-" * 78)
if np.isnan(dh_p95):
    print("  Behind-laser band empty (no metal columns in [start+200, laser_x-100]).")
else:
    print(f"  z_p95 (centerline)  : {z_p95:.2f} um")
    print(f"  Δh (95%ile)         : {dh_p95:+.2f} um   ← preferred")
    print(f"  Δh (max)            : {dh_max:+.2f} um")
    if dh_p95 > 5:
        verdict = "STRONG raised track"
    elif dh_p95 > 1:
        verdict = "WEAK raised track"
    elif dh_p95 > -5:
        verdict = "FLAT"
    else:
        verdict = "DEPRESSED (groove)"
    print(f"  Verdict             : {verdict}")
    print(f"  F3D ref             : Δh ≈ +4 um (slight ridge / fill)")

# ----- Decision-ready table -----
print()
print("=" * 78)
print(f"DECISION TABLE")
print("=" * 78)
print(f"  {'metric':<32} {'value':>12}  {'F3D ref':>12}")
print(f"  {'-'*32} {'-'*12}  {'-'*12}")
def fmt(v, suf=" um"):
    return f"{v:+.2f}{suf}" if np.isfinite(v) else f"{'NaN':>10}"
print(f"  {'v_z @ -150 um (m/s)':<32} {vz_150:>+12.3f}  {'>0 expected':>12}")
print(f"  {'side-ridge peak Δh':<32} {fmt(peak_z - z0_um) if np.isfinite(peak_z) else 'NaN':>12}  {'+4.00 um':>12}")
print(f"  {'centerline Δh (95%ile)':<32} {fmt(dh_p95):>12}  {'+4.00 um':>12}")
groove = (z0_um - z_center_x200) if (z_center_x200 is not None and np.isfinite(z_center_x200)) else float('nan')
print(f"  {'groove depth @ -200 um':<32} {fmt(groove):>12}  {'~0 um':>12}")
print("=" * 78)
