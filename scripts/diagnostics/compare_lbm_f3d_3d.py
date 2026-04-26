#!/usr/bin/env python3
"""Full 3D comparison between LBM surface and F3D surface mesh.

LBM stores f on a structured grid; F3D stores the free-surface triangle mesh
(PolyData with Temperature). To compare:

1. Extract LBM free surface as z(x,y) heightmap from f=0.5 contour.
2. Resample F3D triangle mesh to z(x,y) heightmap on the same (x,y) grid.
3. Align coordinate systems (LBM substrate top = z=160μm; F3D substrate
   top = z=0; subtract per-system substrate level).
4. Diff: Δz_LBM(x,y) - Δz_F3D(x,y).

Outputs:
  - centerline z(x) overlay  (LBM red, F3D blue, F1 green for context)
  - side-ridge z(x) overlay at y_offset = ±50, ±80 μm
  - 2D heatmap of (LBM - F3D) deviation
  - 1D RMS error along x (entire domain, only where both have surface)
  - melt-pool dimension table (length, width, depth at laser_x for both)

Usage:
    python3 compare_lbm_f3d_3d.py <lbm_vtk> [f3d_frame_index]
        f3d_frame_index defaults to 99 (final, t=2 ms equivalent)
"""
import sys, os, re
import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

LBM_SUBSTRATE_Z_UM = 160.0   # LBM substrate top in absolute z [μm]
F3D_SUBSTRATE_Z_UM = 0.0     # F3D substrate top is at z=0 by convention
F3D_REF_DIR = "vtk-316L-150W-50um-V800mms"

def lbm_surface_height(vtk_path):
    """Return (x_um, y_um, z_surf_um[nx,ny] in absolute μm)."""
    m = pv.read(vtk_path)
    nx, ny, nz = m.dimensions
    dx, dy, dz = m.spacing
    ox, oy, oz = m.origin
    f = np.asarray(m.point_data['fill_level']).reshape((nx, ny, nz), order='F')
    mask = (f > 0.5)
    flipped = mask[:, :, ::-1]
    has = flipped.any(axis=2)
    k_top = nz - 1 - np.argmax(flipped, axis=2)
    z = (oz + k_top * dz) * 1e6
    z[~has] = np.nan
    x = (ox + np.arange(nx) * dx) * 1e6
    y = (oy + np.arange(ny) * dy) * 1e6
    return x, y, z

def f3d_resample_to_grid(f3d_path, x_grid, y_grid):
    """Resample F3D PolyData surface to z(x,y) on (x_grid, y_grid).
    F3D z is in absolute μm with substrate at z=0."""
    mesh = pv.read(f3d_path)
    pts = np.asarray(mesh.points) * 1e6   # m → μm
    # Each F3D point has (x,y,z) on the deformed surface.
    # Interpolate z = f(x,y) on our grid.
    Xg, Yg = np.meshgrid(x_grid, y_grid, indexing='ij')
    z_resampled = griddata((pts[:,0], pts[:,1]), pts[:,2],
                           (Xg, Yg), method='linear')
    # nearest fill outside convex hull
    z_nn = griddata((pts[:,0], pts[:,1]), pts[:,2],
                     (Xg, Yg), method='nearest')
    nan_mask = np.isnan(z_resampled)
    z_resampled[nan_mask] = z_nn[nan_mask]
    return z_resampled  # absolute μm

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    lbm_path = sys.argv[1]
    f3d_idx  = int(sys.argv[2]) if len(sys.argv) > 2 else 99
    f3d_path = os.path.join(F3D_REF_DIR, f"150WV800mms-50um_{f3d_idx}.vtk")
    if not os.path.exists(f3d_path):
        print(f"F3D ref not found: {f3d_path}"); sys.exit(2)

    print(f"LBM frame:  {lbm_path}")
    print(f"F3D frame:  {f3d_path}")

    x_um, y_um, z_lbm_abs = lbm_surface_height(lbm_path)
    z_lbm_rel = z_lbm_abs - LBM_SUBSTRATE_Z_UM   # Δh from substrate

    z_f3d_abs = f3d_resample_to_grid(f3d_path, x_um, y_um)
    z_f3d_rel = z_f3d_abs - F3D_SUBSTRATE_Z_UM   # Δh

    # F3D x-range is [-497, 2102]; LBM is [0, 1000]. Align: F3D scan-start at
    # x_F3D = -497 (initial laser position at edge), so for direct overlay we
    # shift F3D by +497.5 μm.
    F3D_X_OFFSET = 497.5  # μm
    # Re-resample F3D with shift
    Xg, Yg = np.meshgrid(x_um, y_um, indexing='ij')
    mesh = pv.read(f3d_path)
    pts = np.asarray(mesh.points) * 1e6
    pts_shifted = pts.copy()
    pts_shifted[:, 0] += F3D_X_OFFSET
    z_f3d_resampled = griddata(
        (pts_shifted[:,0], pts_shifted[:,1]), pts_shifted[:,2],
        (Xg, Yg), method='linear')
    # Don't fill outside hull — keep NaN there (those are regions F3D didn't compute)
    z_f3d_rel = z_f3d_resampled - F3D_SUBSTRATE_Z_UM

    # ============================================================================
    # Centerline x-profile
    # ============================================================================
    nx, ny = z_lbm_rel.shape
    mid_j = ny // 2
    # F3D has y=0 at centerline (substrate symmetric), shift to LBM j (centered)
    # already handled via griddata with y_um from LBM (which spans 0..ny*dx)
    f3d_mid_j = mid_j  # we resampled onto LBM grid
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(x_um, z_lbm_rel[:, mid_j], 'r-', label='LBM (mini)')
    ax.plot(x_um, z_f3d_rel[:, f3d_mid_j], 'b-', label='F3D ref', alpha=0.7)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("Δh (μm) — centerline")
    ax.set_title("Centerline surface profile")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(-100, 30)

    # ============================================================================
    # 2D Δh maps (LBM and F3D)
    # ============================================================================
    extent = [x_um[0], x_um[-1], y_um[0], y_um[-1]]
    vmax = 30
    vmin = -50

    ax = axes[0, 1]
    im = ax.imshow(z_lbm_rel.T, origin='lower', extent=extent, aspect='auto',
                   cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Δh (μm)')
    ax.set_title("LBM Δh(x,y)")
    ax.set_xlabel("x (μm)"); ax.set_ylabel("y (μm)")

    ax = axes[1, 0]
    im = ax.imshow(z_f3d_rel.T, origin='lower', extent=extent, aspect='auto',
                   cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Δh (μm)')
    ax.set_title("F3D Δh(x,y) (resampled, shifted)")
    ax.set_xlabel("x (μm)"); ax.set_ylabel("y (μm)")

    # ============================================================================
    # Diff map
    # ============================================================================
    diff = z_lbm_rel - z_f3d_rel
    ax = axes[1, 1]
    valid = ~np.isnan(diff)
    im = ax.imshow(diff.T, origin='lower', extent=extent, aspect='auto',
                   cmap='RdBu_r', vmin=-30, vmax=30)
    plt.colorbar(im, ax=ax, label='LBM − F3D (μm)')
    rms = float(np.sqrt(np.nanmean(diff**2)))
    ax.set_title(f"LBM − F3D, RMS = {rms:.2f} μm")
    ax.set_xlabel("x (μm)"); ax.set_ylabel("y (μm)")

    plt.tight_layout()
    out_png = lbm_path.replace('.vtk', '_vs_f3d.png')
    plt.savefig(out_png, dpi=110)
    print(f"Saved {out_png}")

    # ============================================================================
    # Numbers
    # ============================================================================
    # Take centerline trailing-band averages from each
    laser_x_um = 1140.0  # mini run final laser x (= 500 + 0.8*400 = 820)? actually for mini t=400μs: laser_x = 500 + 0.8*400 = 820
    # For mini t=400μs final: laser at x=820 μm
    # For phase2 t=800μs final: laser at x=1140 μm (used in full reference)
    # Detect from filename
    mt = re.search(r'_(\d+)\.vtk', lbm_path)
    if mt:
        step = int(mt.group(1))
        t_us = step * 8e-8 * 1e6
        laser_x_um = 500 + 0.8 * t_us
    print(f"\n=== Surface-height verdict ===")
    print(f"Inferred laser x: {laser_x_um:.1f} μm")
    # Trailing band: x in [start + 200, laser - 100]
    band_mask = ((x_um > 700) & (x_um < laser_x_um - 100))
    if band_mask.any():
        # Centerline 95%ile
        l_c = np.percentile(z_lbm_rel[band_mask, mid_j][~np.isnan(z_lbm_rel[band_mask, mid_j])], 95) if (~np.isnan(z_lbm_rel[band_mask, mid_j])).any() else float('nan')
        f_c = np.percentile(z_f3d_rel[band_mask, mid_j][~np.isnan(z_f3d_rel[band_mask, mid_j])], 95) if (~np.isnan(z_f3d_rel[band_mask, mid_j])).any() else float('nan')
        print(f"  Centerline Δh 95%ile  LBM={l_c:+5.1f} μm  F3D={f_c:+5.1f} μm  diff={l_c-f_c:+5.1f}")
        # Side ridges at ±40 μm
        side_mask_y = (np.abs(y_um - y_um[mid_j]) > 30)
        l_s = np.nanmax(z_lbm_rel[band_mask][:, side_mask_y])
        f_s = np.nanmax(z_f3d_rel[band_mask][:, side_mask_y])
        print(f"  Max ridge in trailing  LBM={l_s:+5.1f} μm  F3D={f_s:+5.1f} μm  diff={l_s-f_s:+5.1f}")
    print(f"  RMS surface error: {rms:.2f} μm  (over {valid.sum()}/{diff.size} valid cells)")

    # F3D temperature for reference
    T = np.asarray(mesh.point_data['Temperature'])
    print(f"  F3D T_max: {T.max():.0f} K   (used for Marangoni gradient and recoil scaling)")

if __name__ == '__main__':
    main()
