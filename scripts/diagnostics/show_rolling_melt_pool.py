#!/usr/bin/env python3
"""Visualize the rolling-melt-pool flow pattern: 3D streamlines + free surface
+ vortex structure to show the "翻滚的熔池" (rolling melt pool) the user
demanded.

The signature pattern in F3D LPBF results:
  1. Just-melted material at the trailing edge of the laser
  2. Marangoni-driven outward (away from hot center) along the surface
  3. Recoil-driven downward at the laser footprint (keyhole)
  4. Capillary back-flow from rear toward the cooled trailing groove
  5. This forms a TOROIDAL VORTEX — material rolls FROM the laser BACKWARDS
     ALONG THE SIDES, then DOWN at the trailing edge, then FORWARD ALONG
     THE BOTTOM back to the laser. The melt pool LOOKS LIKE A ROLLING WAVE.

Outputs:
  1. xz-slice at y=centerline: streamlines + free-surface contour
  2. yz-slice at trailing edge x=laser_x-100μm: rotation visible
  3. Top-down 3D: free-surface mesh colored by v_z, with arrows showing
     surface flow direction (shows whether the material is bunching up at
     ridges or flowing back to centerline).
  4. Rotational metric: vorticity field |∇×v| at xz-slice — should be
     concentrated around the rolling-vortex core.

Usage:
    python3 show_rolling_melt_pool.py <vtk_path>
"""
import sys, os, re
import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LASER_START_UM = 500.0
SCAN_VEL_MS    = 0.80
DT_S           = 8.0e-8
SUBSTRATE_Z_UM = 160.0

if len(sys.argv) < 2:
    print(__doc__); sys.exit(1)

vtk_path = sys.argv[1]
m = pv.read(vtk_path)
nx, ny, nz = m.dimensions
dx, dy, dz = m.spacing
ox, oy, oz = m.origin
f = np.asarray(m.point_data['fill_level']).reshape((nx, ny, nz), order='F')
v = np.asarray(m.point_data['velocity']).reshape((nx, ny, nz, 3), order='F')
T = np.asarray(m.point_data['temperature']).reshape((nx, ny, nz), order='F')

# Convert lattice → physical velocity
LU_TO_MS = dx / DT_S
v_phys = v * LU_TO_MS

# Detect time
mt = re.search(r'_(\d+)\.vtk', vtk_path)
t_us = (int(mt.group(1)) * DT_S * 1e6) if mt else 0
laser_x_um = LASER_START_UM + SCAN_VEL_MS * t_us

x_um = (ox + np.arange(nx) * dx) * 1e6
y_um = (oy + np.arange(ny) * dy) * 1e6
z_um = (oz + np.arange(nz) * dz) * 1e6
mid_j = ny // 2

print(f"=== Rolling-melt-pool analysis: {os.path.basename(vtk_path)} ===")
print(f"t = {t_us:.0f} μs, laser_x = {laser_x_um:.0f} μm")
print(f"Domain: {nx}×{ny}×{nz}, dx={dx*1e6:.1f} μm")

# ============================================================================
# Figure: 4-panel rolling-melt-pool view
# ============================================================================
fig = plt.figure(figsize=(15, 11))

# --- Panel 1: xz-slice at centerline (j=mid) ---
# Visualize as: f_contour + velocity arrows
ax1 = fig.add_subplot(2, 2, 1)
F_xz = f[:, mid_j, :]   # (nx, nz)
vx_xz = v_phys[:, mid_j, :, 0]
vz_xz = v_phys[:, mid_j, :, 2]
T_xz = T[:, mid_j, :]
extent_xz = [x_um[0], x_um[-1], z_um[0], z_um[-1]]
# Background: temperature
im = ax1.imshow(T_xz.T, origin='lower', extent=extent_xz, aspect='auto',
                cmap='hot', vmin=300, vmax=3000, alpha=0.6)
plt.colorbar(im, ax=ax1, label='T (K)')
# Free surface contour
ax1.contour(x_um, z_um, F_xz.T, levels=[0.5], colors='black', linewidths=2)
# Velocity arrows (subsample)
stride = max(nx // 30, 1), max(nz // 30, 1)
ax1.quiver(x_um[::stride[0]], z_um[::stride[1]],
           vx_xz[::stride[0], ::stride[1]].T,
           vz_xz[::stride[0], ::stride[1]].T,
           scale=15, scale_units='inches', color='blue', alpha=0.7)
ax1.axvline(laser_x_um, color='orange', lw=1, ls='--', alpha=0.7)
ax1.axhline(SUBSTRATE_Z_UM, color='gray', lw=0.5, ls=':')
ax1.set_xlabel("x (μm)"); ax1.set_ylabel("z (μm)")
ax1.set_title(f"xz-slice (centerline). T_max={T.max():.0f}K, |v|_max={np.sqrt(np.sum(v_phys**2, axis=-1)).max():.2f}m/s")
ax1.set_ylim(60, 200)

# --- Panel 2: yz-slice at trailing edge (x = laser_x - 80 μm) ---
target_x = laser_x_um - 80.0
i_x = int(np.argmin(np.abs(x_um - target_x)))
ax2 = fig.add_subplot(2, 2, 2)
F_yz = f[i_x, :, :]
vy_yz = v_phys[i_x, :, :, 1]
vz_yz = v_phys[i_x, :, :, 2]
T_yz = T[i_x, :, :]
extent_yz = [y_um[0], y_um[-1], z_um[0], z_um[-1]]
im = ax2.imshow(T_yz.T, origin='lower', extent=extent_yz, aspect='auto',
                cmap='hot', vmin=300, vmax=3000, alpha=0.6)
plt.colorbar(im, ax=ax2, label='T (K)')
ax2.contour(y_um, z_um, F_yz.T, levels=[0.5], colors='black', linewidths=2)
stride = max(ny // 25, 1), max(nz // 25, 1)
ax2.quiver(y_um[::stride[0]], z_um[::stride[1]],
           vy_yz[::stride[0], ::stride[1]].T,
           vz_yz[::stride[0], ::stride[1]].T,
           scale=15, scale_units='inches', color='blue', alpha=0.7)
ax2.axhline(SUBSTRATE_Z_UM, color='gray', lw=0.5, ls=':')
ax2.set_xlabel("y (μm)"); ax2.set_ylabel("z (μm)")
ax2.set_title(f"yz-slice at x={target_x:.0f} μm (trailing zone)")
ax2.set_ylim(60, 200)

# --- Panel 3: Top-down free-surface heightmap colored by v_z ---
mask = (f > 0.5)
flipped = mask[:, :, ::-1]
has = flipped.any(axis=2)
k_top = nz - 1 - np.argmax(flipped, axis=2)
z_surf_um = (oz + k_top * dz) * 1e6
z_surf_um[~has] = np.nan
# v_z at the top surface cell (free surface)
i_idx, j_idx = np.indices((nx, ny))
vz_top = v_phys[i_idx, j_idx, k_top, 2]
vz_top[~has] = np.nan

ax3 = fig.add_subplot(2, 2, 3)
extent_xy = [x_um[0], x_um[-1], y_um[0], y_um[-1]]
im = ax3.imshow(z_surf_um.T - SUBSTRATE_Z_UM, origin='lower', extent=extent_xy,
                aspect='auto', cmap='RdBu_r', vmin=-30, vmax=30)
plt.colorbar(im, ax=ax3, label='Δh (μm)')
# Overlay velocity arrows on the surface
i_idx, j_idx = np.indices((nx, ny))
vx_surf = v_phys[i_idx, j_idx, k_top, 0]
vy_surf = v_phys[i_idx, j_idx, k_top, 1]
vx_surf[~has] = 0
vy_surf[~has] = 0
stride2 = max(nx // 25, 1), max(ny // 15, 1)
ax3.quiver(x_um[::stride2[0]], y_um[::stride2[1]],
           vx_surf[::stride2[0], ::stride2[1]].T,
           vy_surf[::stride2[0], ::stride2[1]].T,
           scale=30, scale_units='inches', color='black', alpha=0.5)
ax3.axvline(laser_x_um, color='orange', lw=1, ls='--')
ax3.set_xlabel("x (μm)"); ax3.set_ylabel("y (μm)")
ax3.set_title("Top-down: free-surface Δh + velocity field at surface")

# --- Panel 4: Vorticity y-component (rolling axis) ---
# ω_y = ∂v_x/∂z - ∂v_z/∂x   (rotation in xz-plane)
v_x = v_phys[:, mid_j, :, 0]
v_z_ = v_phys[:, mid_j, :, 2]
dvxdz = np.gradient(v_x, dz, axis=1)
dvzdx = np.gradient(v_z_, dx, axis=0)
omega_y = dvxdz - dvzdx

ax4 = fig.add_subplot(2, 2, 4)
im = ax4.imshow(omega_y.T, origin='lower', extent=extent_xz, aspect='auto',
                cmap='RdBu_r', vmin=-1e6, vmax=1e6)
plt.colorbar(im, ax=ax4, label='ω_y (1/s)')
# Mask non-metal
F_xz_for_mask = f[:, mid_j, :]
ax4.contour(x_um, z_um, F_xz_for_mask.T, levels=[0.5], colors='black', linewidths=1)
ax4.axvline(laser_x_um, color='orange', lw=1, ls='--')
ax4.set_xlabel("x (μm)"); ax4.set_ylabel("z (μm)")
ax4.set_title("Vorticity ω_y (xz-plane): rolling melt-pool axis. Red=CCW, Blue=CW")
ax4.set_ylim(60, 200)

plt.tight_layout()
out_png = vtk_path.replace('.vtk', '_rolling.png')
plt.savefig(out_png, dpi=110)
print(f"Saved {out_png}")

# ============================================================================
# Quantitative rolling-melt-pool diagnostic
# ============================================================================
# Look for a rotation signature: at the trailing edge, in the xz-plane
# centerline, expect:
#   - Surface flow x-direction = -x (back toward melt-pool tail)
#   - Bottom flow x-direction = +x (forward, returning to laser)
#   - Net circulation: ∫v·dl around the closed pool boundary > 0

# Trailing-band pool centerline cells
band_mask = (x_um > LASER_START_UM + 100) & (x_um < laser_x_um - 50)
top_layer = (z_um > SUBSTRATE_Z_UM - 10) & (z_um < SUBSTRATE_Z_UM + 5)  # near surface
bottom_layer = (z_um > SUBSTRATE_Z_UM - 80) & (z_um < SUBSTRATE_Z_UM - 50)  # deep

vx_top = v_phys[band_mask, mid_j, :, 0][:, top_layer]
vx_bot = v_phys[band_mask, mid_j, :, 0][:, bottom_layer]

# Where in the pool? Mask to only liquid cells (lf > 0.5)
lf = np.asarray(m.point_data['liquid_fraction']).reshape((nx, ny, nz), order='F')
lf_top = lf[band_mask, mid_j, :][:, top_layer]
lf_bot = lf[band_mask, mid_j, :][:, bottom_layer]

vx_top_avg = float(np.nanmean(vx_top[lf_top > 0.5])) if (lf_top > 0.5).any() else 0
vx_bot_avg = float(np.nanmean(vx_bot[lf_bot > 0.5])) if (lf_bot > 0.5).any() else 0

print(f"\n=== Rolling-pool diagnostic ===")
print(f"  Trailing band centerline:")
print(f"    Top-layer  v_x avg (in liquid): {vx_top_avg:+.3f} m/s   (expect NEGATIVE = back-flow toward laser tail)")
print(f"    Bot-layer  v_x avg (in liquid): {vx_bot_avg:+.3f} m/s   (expect POSITIVE = forward to feed the laser zone)")
rolling_strength = vx_bot_avg - vx_top_avg
print(f"  Rolling strength (v_x_bot - v_x_top): {rolling_strength:+.3f} m/s   (should be POSITIVE if rolling)")
print(f"  ω_y peak in pool: {np.abs(omega_y[F_xz_for_mask > 0.5]).max():.2e} /s")
