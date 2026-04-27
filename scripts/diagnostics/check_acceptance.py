#!/usr/bin/env python3
"""Task-A acceptance check — reports all 7 pass conditions on a single VTK frame.

Usage:
    python check_acceptance.py <vtk_path> [vtk_initial_path]
        vtk_path        — frame to evaluate (e.g. line_scan_010000.vtk)
        vtk_initial_path — optional t=0 frame for mass-drift check

Reports (one per criterion):
  1. centerline Δh @ 95%ile, target ≥ -10 μm
  3. side-ridge Δh at x_offset = -100 μm and -200 μm, target both [+3, +10] μm
  4. mass drift |ΔM/M0| (only if t=0 frame given), target < 1.0%
  7. v_z @ centerline x_offset = -150 μm, target > +0.10 m/s
  + reject-condition guards: side ridges < +15 μm anywhere, no NaN.
"""
import sys
import numpy as np

try:
    import pyvista as pv
except ImportError:
    print("ERROR: pyvista not installed. Run: pip install pyvista", file=sys.stderr)
    sys.exit(2)

if len(sys.argv) < 2:
    print(__doc__); sys.exit(1)

vtk_path = sys.argv[1]
vtk0_path = sys.argv[2] if len(sys.argv) > 2 else None

# Constants matching sim_linescan_phase{2,4} configuration
LASER_START_UM = 500.0      # initial laser x position
SCAN_VEL_MS    = 0.80       # scan velocity m/s
SPLASH_GUARD_UM = 200.0      # exclude first 200 μm of scan path (start-up splash)
# Z0_UM is now auto-detected from the t=0 frame if provided, else from current
# frame's median centerline height in the un-laser-affected zone.

def free_surface_z(vtk_path):
    """Return (z_surf_um[nx,ny], has_metal[nx,ny], k_top[nx,ny], grid)."""
    m = pv.read(vtk_path)
    nx, ny, nz = m.dimensions
    dx, dy, dz = m.spacing
    ox, oy, oz = m.origin
    f = np.asarray(m.point_data['fill_level']).reshape((nx, ny, nz), order='F')
    mask = (f > 0.5)
    flipped = mask[:, :, ::-1]
    has_metal = flipped.any(axis=2)
    k_top = nz - 1 - np.argmax(flipped, axis=2)
    z_surface = oz + k_top * dz
    z_surf_um = z_surface * 1e6
    z_surf_um[~has_metal] = np.nan
    return z_surf_um, has_metal, k_top, (nx, ny, nz, dx, dy, dz, ox, oy, oz, m, f)

z_surf_um, has_metal, k_top, grid = free_surface_z(vtk_path)
nx, ny, nz, dx, dy, dz, ox, oy, oz, m, f = grid

# Auto-detect substrate top z0
if vtk0_path:
    z0_arr, _, _, _ = free_surface_z(vtk0_path)
    Z0_UM = float(np.nanmedian(z0_arr))
else:
    # fallback: use far-field median of CURRENT frame centerline
    Z0_UM = float(np.nanmedian(z_surf_um[:, ny//2]))
print(f"Substrate top z0 = {Z0_UM:.1f} μm  ({'from t=0 frame' if vtk0_path else 'from current frame median'})")

# Find velocity field — try common names
vel = None
for name in ('velocity', 'velocity_physical', 'u_phys', 'u'):
    if name in m.point_data:
        vel = np.asarray(m.point_data[name])
        break
if vel is None:
    print("WARN: no velocity field found in VTK; v_z criterion will be skipped",
          file=sys.stderr)
    vz = None
else:
    if vel.ndim == 2 and vel.shape[1] == 3:
        vz = vel[:, 2].reshape((nx, ny, nz), order='F')
    else:
        vz = None

# Free surface: highest k where f > 0.5
mask = (f > 0.5)
flipped = mask[:, :, ::-1]
has_metal = flipped.any(axis=2)
k_top = nz - 1 - np.argmax(flipped, axis=2)
z_surface = oz + k_top * dz
z_surf_um = z_surface * 1e6
z_surf_um[~has_metal] = np.nan

x_um = (ox + np.arange(nx) * dx) * 1e6
y_um = (oy + np.arange(ny) * dy) * 1e6
mid_j = ny // 2

# Recover the timestep from filename suffix → laser_x estimate
import re, os
mt = re.search(r'(\d{6})\.vtk', os.path.basename(vtk_path))
if mt:
    step = int(mt.group(1))
    # dt = 80 ns → t = step * 8e-8 s; laser_x = LASER_START + v_scan * t
    t_s = step * 8.0e-8
    laser_x_um = LASER_START_UM + SCAN_VEL_MS * t_s * 1e6
else:
    # Fallback: deepest centerline point
    z_centerline = z_surf_um[:, mid_j]
    valid = ~np.isnan(z_centerline)
    laser_x_um = x_um[np.nanargmin(z_centerline)] if valid.any() else float('nan')
    t_s = float('nan')

print(f"\n=== Task-A Acceptance Check ===")
print(f"File:     {vtk_path}")
print(f"Step:     {step if mt else '?'}, t = {t_s*1e6:.1f} μs (estimated)")
print(f"Laser x:  {laser_x_um:.1f} μm  (start={LASER_START_UM}, v={SCAN_VEL_MS} m/s)")
print(f"Domain:   {nx}×{ny}×{nz}, dx={dx*1e6:.1f} μm")
print(f"")

z_centerline = z_surf_um[:, mid_j]

# Trailing band: x in [start+splash_guard, laser − 100], same exclusion as
# extract_track_height.py to skip scan-start splash transient.
behind_mask = ((x_um > LASER_START_UM + SPLASH_GUARD_UM) &
               (x_um < laser_x_um - 100.0) &
               ~np.isnan(z_centerline))

results = {}

# -------- Criterion 1/2: centerline Δh (95%ile) --------
if behind_mask.any():
    z_p95 = np.percentile(z_centerline[behind_mask], 95)
    dh_centerline = z_p95 - Z0_UM
    n_behind = int(behind_mask.sum())
    print(f"[1/2] Centerline Δh (95%ile, n={n_behind} cells): "
          f"{dh_centerline:+.2f} μm  (target ≥ −10 μm)  "
          f"{'PASS' if dh_centerline >= -10.0 else 'FAIL'}")
    results['centerline_dh'] = dh_centerline
else:
    print(f"[1/2] Centerline Δh: NO TRAILING ZONE  FAIL")
    results['centerline_dh'] = None

# -------- Criterion 3: side ridges at -100 μm and -200 μm offsets --------
# offset is in x relative to laser; pick centerline columns at those x.
def ridge_at_offset(off_um):
    target_x = laser_x_um + off_um   # off_um is negative
    if np.isnan(target_x): return None
    if target_x < x_um[0] or target_x > x_um[-1]: return None
    # nearest x; then sweep y to find max raised height (excluding centerline)
    i_x = np.argmin(np.abs(x_um - target_x))
    z_col = z_surf_um[i_x, :]
    # Side cells: |y - y_mid| > 30 μm
    y_dist = np.abs(y_um - y_um[mid_j])
    side_mask = (y_dist > 30.0) & ~np.isnan(z_col)
    if not side_mask.any(): return None
    return float(np.nanmax(z_col[side_mask]) - Z0_UM)

ridge_100 = ridge_at_offset(-100.0)
ridge_200 = ridge_at_offset(-200.0)
def fmt_ridge(r):
    return f"{r:+.2f} μm" if r is not None else "n/a"
def ridge_pass(r):
    return r is not None and 3.0 <= r <= 10.0

p3 = ridge_pass(ridge_100) and ridge_pass(ridge_200)
print(f"[3]   Side ridge Δh @ −100 μm: {fmt_ridge(ridge_100)}, "
      f"@ −200 μm: {fmt_ridge(ridge_200)}  "
      f"(target both ∈ [+3, +10])  {'PASS' if p3 else 'FAIL'}")
results['ridge_100'] = ridge_100
results['ridge_200'] = ridge_200

# -------- Reject: any ridge above +15 μm anywhere in trailing band --------
trailing_2d_mask = (
    (x_um > LASER_START_UM + SPLASH_GUARD_UM) &
    (x_um < laser_x_um - 50.0))
if trailing_2d_mask.any():
    sub = z_surf_um[trailing_2d_mask, :]
    max_dh_trailing = np.nanmax(sub) - Z0_UM
    if max_dh_trailing > 15.0:
        print(f"      ⚠️  REJECT: max trailing-zone Δh = {max_dh_trailing:+.2f} μm > +15")
        results['reject_ridge15'] = True
    else:
        print(f"      Max Δh in trailing band: {max_dh_trailing:+.2f} μm (< +15 OK)")
        results['reject_ridge15'] = False

# -------- Criterion 4: mass drift (needs t=0 frame) --------
if vtk0_path:
    m0 = pv.read(vtk0_path)
    f0 = np.asarray(m0.point_data['fill_level']).reshape(
        m0.dimensions, order='F')
    M0 = float(np.sum(f0)) * dx * dy * dz   # in m³ (consistent units)
    M  = float(np.sum(f))  * dx * dy * dz
    drift = (M - M0) / M0 if M0 > 0 else float('nan')
    print(f"[4]   Mass drift |ΔM/M₀|: {drift*100:+.4f}%  "
          f"(target < ±1.0%)  {'PASS' if abs(drift) < 0.01 else 'FAIL'}")
    results['mass_drift'] = drift
else:
    print(f"[4]   Mass drift: SKIPPED (no t=0 frame given)")

# -------- Criterion 7: v_z at -150 μm offset, centerline --------
if vz is not None:
    target_x = laser_x_um - 150.0
    if not np.isnan(target_x) and x_um[0] <= target_x <= x_um[-1]:
        i_x = np.argmin(np.abs(x_um - target_x))
        # Find the surface k at this column
        if has_metal[i_x, mid_j]:
            k_top_here = int(k_top[i_x, mid_j])
            # Sample one cell below the surface (where flow is fully liquid)
            k_sample = max(0, k_top_here - 1)
            # IMPORTANT: VTK 'velocity' is in lattice units; convert to m/s.
            # multiplier = dx_meter / dt_seconds. For Phase-2/4: 2e-6 / 8e-8 = 25.
            DT_S = 8.0e-8   # all phase{1,2,4} apps use dt=80 ns
            lu_to_ms = dx / DT_S
            vz_value = float(vz[i_x, mid_j, k_sample]) * lu_to_ms
            p7 = vz_value > 0.10
            print(f"[7]   v_z @ centerline x={target_x:.1f} μm "
                  f"(col k={k_sample}): {vz_value:+.3f} m/s  "
                  f"(target > +0.10)  {'PASS' if p7 else 'FAIL'}")
            results['vz_back'] = vz_value
        else:
            print(f"[7]   v_z @ x_offset=−150 μm: column has no metal  FAIL")
    else:
        print(f"[7]   v_z @ x_offset=−150 μm: out of domain")
else:
    print(f"[7]   v_z criterion: SKIPPED (no velocity field in VTK)")

# -------- NaN check --------
if np.isnan(f).any():
    print(f"      ⚠️  REJECT: NaN detected in fill_level field")
    results['nan'] = True

print(f"\n=== Summary ===")
for k, v in results.items():
    print(f"  {k}: {v}")
