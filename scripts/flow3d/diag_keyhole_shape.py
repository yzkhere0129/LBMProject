#!/usr/bin/env python3
"""Sprint-3 diagnostic: measure keyhole geometry to verify the
'shallow bowl vs deep narrow knife' Sprint-2 architectural claim.

Reports for one frame at quasi-steady-state during scan:
  - keyhole depth (z0 - z_tip)
  - keyhole opening width at z=z0 plane
  - keyhole width at tip (last 10 um above z_tip)
  - aperture angle (deg) — half-angle from tip to opening
  - κ_max in keyhole-wall cells (from VTK curvature field)
  - implied P_cap = sigma * κ_max  (sigma=1.74 N/m for 316L)

Usage: python diag_keyhole_shape.py <vtk> [z0_um=160] [sigma=1.74]
"""
import sys
import numpy as np
import pyvista as pv

if len(sys.argv) < 2:
    print(__doc__); sys.exit(1)
path = sys.argv[1]
z0_um  = float(sys.argv[2]) if len(sys.argv) > 2 else 160.0
sigma  = float(sys.argv[3]) if len(sys.argv) > 3 else 1.74  # N/m

print(f"Reading {path} ...")
m = pv.read(path)
nx, ny, nz = m.dimensions
dx, dy, dz = m.spacing
ox, oy, oz = m.origin
print(f"Grid {nx}x{ny}x{nz}, dx={dx*1e6:.2f}um")

f    = np.asarray(m.point_data['fill_level']).reshape((nx, ny, nz), order='F')
T    = np.asarray(m.point_data['temperature']).reshape((nx, ny, nz), order='F')
kappa = np.asarray(m.point_data['curvature']).reshape((nx, ny, nz), order='F')

# z-axis in physical um
z_um = (oz + np.arange(nz) * dz) * 1e6
k0   = int(np.argmin(np.abs(z_um - z0_um)))

# Find laser position: deepest point along centerline (y=mid)
mid_j = ny // 2
flipped = (f > 0.5)[:, :, ::-1]
has_metal = flipped.any(axis=2)
k_top = nz - 1 - np.argmax(flipped, axis=2)
z_surf_um = (oz + k_top * dz) * 1e6
z_surf_um = np.where(has_metal, z_surf_um, np.nan)

z_centerline = z_surf_um[:, mid_j]
i_laser = int(np.nanargmin(z_centerline))
laser_x_um = (ox + i_laser * dx) * 1e6
z_tip_um = z_centerline[i_laser]
depth_um = z0_um - z_tip_um

print(f"")
print(f"Laser position: x={laser_x_um:.0f} um, depth={depth_um:.1f} um (z_tip={z_tip_um:.1f})")
print(f"")

# Cross-section at laser x (a y-z slab) — keyhole cavity = (f<0.5) AND below z0
yz_fill = f[i_laser, :, :]
yz_T    = T[i_laser, :, :]

# 1) opening width at z = z0:  number of cells with f<0.5 at k=k0 across y
open_mask = yz_fill[:, k0] < 0.5
open_w_um = open_mask.sum() * dy * 1e6

# 2) tip width: avg cells with f<0.5 across y in last 10um above tip
k_tip = int((z_tip_um * 1e-6 - oz) / dz)
k_tip = max(0, min(k_tip, nz - 1))
tip_band_um = 10.0
n_tip_cells = max(1, int(tip_band_um * 1e-6 / dz))
k_tip_top = min(nz - 1, k_tip + n_tip_cells)
tip_slab = yz_fill[:, k_tip:k_tip_top + 1] < 0.5  # (ny, n_tip)
# average width across the band
tip_widths = tip_slab.sum(axis=0)  # cells at each k
tip_w_um = tip_widths.mean() * dy * 1e6

# 3) aperture angle: half-angle from tip to opening
half_open_um = 0.5 * open_w_um
half_tip_um  = 0.5 * tip_w_um
height_um    = depth_um  # vertical span tip to opening
if height_um > 1.0:
    aperture_deg = np.degrees(np.arctan2(half_open_um - half_tip_um, height_um))
else:
    aperture_deg = float('nan')

# 4) max curvature in keyhole-wall cells = (interface cells inside keyhole region)
# interface mask at this slab: cells with 0.05 < f < 0.95 inside the keyhole bbox
keyhole_z_lo = k_tip
keyhole_z_hi = k0
interface = (yz_fill > 0.05) & (yz_fill < 0.95)
slab_kappa = np.abs(kappa[i_laser, :, keyhole_z_lo:keyhole_z_hi + 1])
slab_int   = interface[:, keyhole_z_lo:keyhole_z_hi + 1]
if slab_int.any():
    kappa_keyhole = slab_kappa[slab_int]
    k_max  = float(np.percentile(kappa_keyhole, 99))    # robust max
    k_p90  = float(np.percentile(kappa_keyhole, 90))
    k_med  = float(np.median(kappa_keyhole))
    P_max  = sigma * k_max  # Pa
else:
    k_max = k_p90 = k_med = float('nan')
    P_max = float('nan')

# 5) keyhole tip cells: pick lowest 5 layers in keyhole, their max curvature
n_tip_kappa = 5
kt_lo = max(0, k_tip)
kt_hi = min(nz - 1, k_tip + n_tip_kappa - 1)
slab_tip = (yz_fill[:, kt_lo:kt_hi + 1] > 0.05) & (yz_fill[:, kt_lo:kt_hi + 1] < 0.95)
slab_tip_k = np.abs(kappa[i_laser, :, kt_lo:kt_hi + 1])
if slab_tip.any():
    k_tip_max = float(np.percentile(slab_tip_k[slab_tip], 99))
    P_tip = sigma * k_tip_max
else:
    k_tip_max = float('nan')
    P_tip = float('nan')

# T_max in keyhole vapor region (high T but f<0.5)
vapor_mask = (yz_fill < 0.05) & (np.arange(nz)[None, :] >= k_tip) & (np.arange(nz)[None, :] <= k0)
T_max_vapor = float(yz_T[vapor_mask].max()) if vapor_mask.any() else float('nan')

# Implied F3D κ from L=78, W=73, D/W=1.08 (deep narrow knife)
# tip radius ~ W/2 / aspect ~ 73/2/2 ~ 18 um → κ_F3D ≈ 1/18e-6 = 5.5e4 m^-1

print(f"=== KEYHOLE GEOMETRY at x={laser_x_um:.0f} um ===")
print(f"  depth (D)         : {depth_um:.1f} um")
print(f"  opening width (W) : {open_w_um:.1f} um  (at z=z0)")
print(f"  tip width         : {tip_w_um:.1f} um  (band {tip_band_um:.0f}um above tip)")
print(f"  D/W ratio         : {depth_um/max(open_w_um,1e-6):.2f}  (F3D=1.08)")
print(f"  aperture half-deg : {aperture_deg:.1f} deg")
print(f"  T_max vapor       : {T_max_vapor:.0f} K")
print(f"")
print(f"=== CURVATURE (m^-1, |kappa|) ===")
print(f"  keyhole walls 99% : {k_max:.2e}")
print(f"  keyhole walls 90% : {k_p90:.2e}")
print(f"  keyhole walls med : {k_med:.2e}")
print(f"  TIP region 99%    : {k_tip_max:.2e}")
print(f"")
print(f"=== CAPILLARY PRESSURE (sigma={sigma:.2f} N/m) ===")
print(f"  P_cap walls 99%   : {P_max/1e3:.1f} kPa")
print(f"  P_cap TIP  99%    : {P_tip/1e3:.1f} kPa")
print(f"")
print(f"=== EXPECTATION (Sprint-2 architectural claim) ===")
print(f"  F3D 'deep narrow knife':  kappa_tip ~ 6.7e4 m^-1 (radius ~15 um)")
print(f"  LBM 'shallow bowl'    :  kappa_tip ~ 2e4 m^-1 (radius ~50 um)")
print(f"  F3D P_cap ~ 116 kPa, LBM ~ 35 kPa (3x weaker)")
print(f"")
print(f"=== VERDICT ===")
if not np.isnan(k_tip_max):
    if k_tip_max > 4e4:
        print(f"  Tip is SHARP (kappa_tip={k_tip_max:.1e}>=4e4 m^-1).")
        print(f"  Sprint-2 architectural claim REFUTED — keyhole shape is fine.")
        print(f"  Real bottleneck must be elsewhere (Darcy? solidification timing?).")
    elif k_tip_max > 2e4:
        print(f"  Tip is MODERATE (kappa_tip={k_tip_max:.1e} m^-1).")
        print(f"  Capillary force ~half F3D — ray-tracing fix would improve but")
        print(f"  may not fully close the gap.")
    else:
        print(f"  Tip is BLUNT (kappa_tip={k_tip_max:.1e}<2e4 m^-1).")
        print(f"  Sprint-2 architectural claim CONFIRMED — proceed with rewrite.")
