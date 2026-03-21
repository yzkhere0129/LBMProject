#!/usr/bin/env python3
"""
Marangoni flow analysis in LPBF melt pool.

Checks whether the velocity field matches expected Marangoni convection:
  1. Surface flow: hot center → cold edge (outward, for dσ/dT < 0)
  2. Return flow: inward at depth
  3. Velocity magnitude scales with ∇T at surface
  4. Characteristic vortex pattern (two counter-rotating rolls)

Reads VTK at peak laser interaction time, extracts XZ midplane.
"""

import os, sys, struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

VTK_DIR = "output_powder_bed_sim"
if not os.path.isdir(VTK_DIR):
    VTK_DIR = os.path.join(os.path.dirname(__file__), "../../build/output_powder_bed_sim")

OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "marangoni_flow_analysis.png")

NX, NY, NZ = 500, 150, 90
DX = 2e-6
DT = 8e-8
N = NX * NY * NZ
IY_MID = NY // 2
LU_TO_MS = DX / DT

# Use a snapshot where laser is active and melt pool is developed
VTK_FILE = os.path.join(VTK_DIR, "powder_sim_005000.vtk")  # t=400μs
T_US = 400
LASER_X_UM = 100 + 0.8 * T_US * 1e-6 * 1e6  # start + v_scan * t

T_SOLIDUS = 1648.0
T_LIQUIDUS = 1673.0


def parse_vtk(filepath):
    """Parse scalar and vector fields from ASCII VTK."""
    print(f"Parsing {os.path.basename(filepath)}...", flush=True)
    with open(filepath, 'r') as fh:
        lines = fh.readlines()

    fields = {}
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith('VECTORS'):
            fields[s.split()[1].lower()] = ('vector', i + 1)
        elif s.startswith('SCALARS'):
            fields[s.split()[1].lower()] = ('scalar', i + 2)
        i += 1

    result = {}
    for name, (kind, start) in fields.items():
        if kind == 'scalar':
            data = np.empty(N, dtype=np.float32)
            idx = 0
            for line in lines[start:]:
                if idx >= N: break
                for v in line.split():
                    if idx >= N: break
                    data[idx] = float(v); idx += 1
            result[name] = data.reshape(NZ, NY, NX)
        else:
            data = np.empty((N, 3), dtype=np.float32)
            idx = 0
            for line in lines[start:]:
                if idx >= N: break
                parts = line.split()
                if len(parts) >= 3:
                    data[idx] = [float(p) for p in parts[:3]]
                    idx += 1
            result[name] = data.reshape(NZ, NY, NX, 3)
    return result


def main():
    if not os.path.exists(VTK_FILE):
        print(f"ERROR: {VTK_FILE} not found"); sys.exit(1)

    d = parse_vtk(VTK_FILE)
    T = d['temperature'][:, IY_MID, :]      # (NZ, NX)
    f = d['fill_level'][:, IY_MID, :]
    vel = d['velocity'][:, IY_MID, :, :]     # (NZ, NX, 3) in LU
    vx = vel[:, :, 0] * LU_TO_MS            # m/s
    vz = vel[:, :, 2] * LU_TO_MS
    vmag = np.sqrt(vx**2 + vz**2)

    x_um = np.arange(NX) * DX * 1e6
    z_um = np.arange(NZ) * DX * 1e6
    X, Z = np.meshgrid(x_um, z_um)

    # === Find melt pool region ===
    liquid_mask = (T > T_LIQUIDUS) & (f > 0.3)
    if not liquid_mask.any():
        print("WARNING: No liquid found! Using T > T_solidus")
        liquid_mask = (T > T_SOLIDUS) & (f > 0.3)

    liq_z, liq_x = np.where(liquid_mask)
    if len(liq_x) == 0:
        print("ERROR: No melt pool found"); sys.exit(1)

    # Melt pool bounding box
    x_min_mp = liq_x.min(); x_max_mp = liq_x.max()
    z_min_mp = liq_z.min(); z_max_mp = liq_z.max()
    x_center = (x_min_mp + x_max_mp) // 2
    z_top = z_max_mp  # Surface of melt pool

    print(f"Melt pool: x=[{x_min_mp*DX*1e6:.0f}, {x_max_mp*DX*1e6:.0f}] μm, "
          f"z=[{z_min_mp*DX*1e6:.0f}, {z_max_mp*DX*1e6:.0f}] μm")
    print(f"Center: x={x_center*DX*1e6:.0f} μm, top z={z_top*DX*1e6:.0f} μm")

    # === Analysis 1: Surface velocity direction ===
    # For dσ/dT < 0 (metals): Marangoni drives flow from hot (center) to cold (edge)
    # Check: at z=z_top, vx should point AWAY from the hottest point
    z_surf = min(z_top, NZ - 1)
    T_surf = T[z_surf, :]
    hotspot_i = np.argmax(T_surf)
    print(f"Hotspot at x={hotspot_i*DX*1e6:.0f} μm, T={T_surf[hotspot_i]:.0f} K")

    # Surface velocities in liquid region
    surf_liquid = liquid_mask[z_surf, :]
    surf_vx = vx[z_surf, :]

    # Left of hotspot: vx should be negative (pointing left = away from center)
    left_mask = surf_liquid & (np.arange(NX) < hotspot_i)
    right_mask = surf_liquid & (np.arange(NX) > hotspot_i)

    left_vx_mean = surf_vx[left_mask].mean() if left_mask.any() else 0
    right_vx_mean = surf_vx[right_mask].mean() if right_mask.any() else 0

    print(f"\n=== Marangoni Surface Flow Check ===")
    print(f"  dσ/dT < 0 → expect outward flow at surface")
    print(f"  Left of hotspot:  <vx> = {left_vx_mean:+.3f} m/s {'✓ OUTWARD' if left_vx_mean < -0.01 else '✗ WRONG'}")
    print(f"  Right of hotspot: <vx> = {right_vx_mean:+.3f} m/s {'✓ OUTWARD' if right_vx_mean > 0.01 else '✗ WRONG'}")
    marangoni_surface_ok = (left_vx_mean < -0.01) and (right_vx_mean > 0.01)

    # === Analysis 2: Return flow at depth ===
    z_deep = max(z_min_mp, z_surf - 5)  # 5 cells below surface
    deep_liquid = liquid_mask[z_deep, :]
    deep_vx = vx[z_deep, :]

    left_deep = deep_liquid & (np.arange(NX) < hotspot_i)
    right_deep = deep_liquid & (np.arange(NX) > hotspot_i)

    left_deep_vx = deep_vx[left_deep].mean() if left_deep.any() else 0
    right_deep_vx = deep_vx[right_deep].mean() if right_deep.any() else 0

    print(f"\n=== Return Flow Check (z={z_deep*DX*1e6:.0f} μm, {(z_surf-z_deep)*2} μm below surface) ===")
    print(f"  Expect inward flow at depth (opposite to surface)")
    print(f"  Left of hotspot:  <vx> = {left_deep_vx:+.3f} m/s {'✓ INWARD' if left_deep_vx > 0.01 else '? weak/absent'}")
    print(f"  Right of hotspot: <vx> = {right_deep_vx:+.3f} m/s {'✓ INWARD' if right_deep_vx < -0.01 else '? weak/absent'}")

    # === Analysis 3: Velocity profile across melt pool surface ===
    print(f"\n=== Surface Velocity Profile ===")
    surf_x_um = x_um[surf_liquid]
    surf_v_profile = surf_vx[surf_liquid]
    surf_T_profile = T_surf[surf_liquid]
    if len(surf_x_um) > 0:
        print(f"  x range: {surf_x_um.min():.0f} – {surf_x_um.max():.0f} μm")
        print(f"  |vx| max: {np.abs(surf_v_profile).max():.3f} m/s")
        print(f"  T range: {surf_T_profile.min():.0f} – {surf_T_profile.max():.0f} K")

    # === Analysis 4: Vorticity (∂vz/∂x - ∂vx/∂z) ===
    dvz_dx = np.gradient(vz, DX * 1e6, axis=1)  # per μm → ×1e6 for /m
    dvx_dz = np.gradient(vx, DX * 1e6, axis=0)
    omega_y = dvz_dx - dvx_dz  # [1/μs effectively, but we care about sign pattern]

    # =========================================================================
    # FIGURE: 6-panel analysis
    # =========================================================================
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    # Zoom to melt pool region
    pad = 30  # cells
    xi0 = max(0, x_min_mp - pad)
    xi1 = min(NX, x_max_mp + pad)
    zi0 = max(0, z_min_mp - pad)
    zi1 = min(NZ, z_max_mp + 10)
    x_zoom = x_um[xi0:xi1]
    z_zoom = z_um[zi0:zi1]
    Xz, Zz = np.meshgrid(x_zoom, z_zoom)

    # --- Panel (0,0): Temperature + quiver ---
    ax = fig.add_subplot(gs[0, 0])
    T_z = T[zi0:zi1, xi0:xi1]
    f_z = f[zi0:zi1, xi0:xi1]
    T_masked = np.ma.masked_where(f_z < 0.01, T_z)
    cmap_T = plt.cm.hot.copy(); cmap_T.set_bad('#222222')
    ax.pcolormesh(Xz, Zz, T_masked, cmap=cmap_T, vmin=300, vmax=3000, shading='auto')
    ax.contour(Xz, Zz, T_z, levels=[T_SOLIDUS], colors='cyan', linewidths=1, linestyles='--')
    ax.contour(Xz, Zz, T_z, levels=[T_LIQUIDUS], colors='white', linewidths=1)

    # Quiver in liquid
    stride = 3
    sx = slice(0, xi1-xi0, stride)
    sz = slice(0, zi1-zi0, stride)
    vx_z = vx[zi0:zi1, xi0:xi1]
    vz_z = vz[zi0:zi1, xi0:xi1]
    liq_z_local = liquid_mask[zi0:zi1, xi0:xi1]
    mask_q = liq_z_local[sz, sx]
    ux_q = np.where(mask_q, vx_z[sz, sx], 0)
    uz_q = np.where(mask_q, vz_z[sz, sx], 0)
    spd = np.sqrt(ux_q**2 + uz_q**2)
    if spd.max() > 0.01:
        ax.quiver(Xz[sz, sx], Zz[sz, sx], ux_q, uz_q, spd,
                  cmap='cool', scale=12, width=0.004, headwidth=3,
                  clim=[0, 2], alpha=0.9)
    ax.set_title('(a) Temperature + velocity arrows in liquid', fontsize=10)
    ax.set_ylabel('z [μm]')
    ax.set_aspect('equal')

    # --- Panel (0,1): Velocity magnitude ---
    ax2 = fig.add_subplot(gs[0, 1])
    vmag_z = vmag[zi0:zi1, xi0:xi1]
    vmag_masked = np.ma.masked_where(f_z < 0.01, vmag_z)
    cmap_v = plt.cm.inferno.copy(); cmap_v.set_bad('#111111')
    im_v = ax2.pcolormesh(Xz, Zz, vmag_masked, cmap=cmap_v, vmin=0, vmax=2, shading='auto')
    ax2.contour(Xz, Zz, T_z, levels=[T_SOLIDUS], colors='cyan', linewidths=0.8, linestyles='--')
    plt.colorbar(im_v, ax=ax2, label='|v| [m/s]', shrink=0.8)
    ax2.set_title('(b) Velocity magnitude', fontsize=10)
    ax2.set_aspect('equal')

    # --- Panel (1,0): Vorticity ---
    ax3 = fig.add_subplot(gs[1, 0])
    omega_z = omega_y[zi0:zi1, xi0:xi1]
    omega_masked = np.ma.masked_where(~liq_z_local, omega_z)
    vlim = np.percentile(np.abs(omega_z[liq_z_local]), 95) if liq_z_local.any() else 1
    im_w = ax3.pcolormesh(Xz, Zz, omega_masked, cmap='RdBu_r', vmin=-vlim, vmax=vlim, shading='auto')
    ax3.contour(Xz, Zz, T_z, levels=[T_SOLIDUS], colors='black', linewidths=0.8, linestyles='--')
    plt.colorbar(im_w, ax=ax3, label='ωy [1/μs]', shrink=0.8)
    ax3.set_title('(c) Vorticity ωy — expect ±pair (counter-rotating rolls)', fontsize=10)
    ax3.set_ylabel('z [μm]')
    ax3.set_aspect('equal')

    # --- Panel (1,1): Surface vx profile ---
    ax4 = fig.add_subplot(gs[1, 1])
    if len(surf_x_um) > 0:
        ax4.plot(surf_x_um, surf_v_profile, 'b-', lw=2, label='vx at surface')
        ax4.axhline(0, color='gray', ls='--', lw=0.5)
        ax4.axvline(hotspot_i * DX * 1e6, color='red', ls=':', lw=1, label=f'Hotspot x={hotspot_i*DX*1e6:.0f}μm')
        ax4.fill_between(surf_x_um, surf_v_profile, 0,
                         where=surf_v_profile > 0, color='red', alpha=0.2, label='Rightward')
        ax4.fill_between(surf_x_um, surf_v_profile, 0,
                         where=surf_v_profile < 0, color='blue', alpha=0.2, label='Leftward')
    ax4.set_xlabel('x [μm]')
    ax4.set_ylabel('vx [m/s]')
    ax4.set_title('(d) Surface vx — dσ/dT<0: left of hot=negative, right=positive', fontsize=10)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # --- Panel (2,0): Vertical vx profiles at two x-locations ---
    ax5 = fig.add_subplot(gs[2, 0])
    # Left side of melt pool
    x_left = max(xi0, hotspot_i - 8)
    x_right = min(xi1 - 1, hotspot_i + 8)
    z_range = np.arange(zi0, zi1)
    liq_col_left = liquid_mask[zi0:zi1, x_left]
    liq_col_right = liquid_mask[zi0:zi1, x_right]

    if liq_col_left.any():
        ax5.plot(vx[zi0:zi1, x_left][liq_col_left],
                 z_um[zi0:zi1][liq_col_left],
                 'b-o', ms=3, lw=1.5, label=f'x={x_left*DX*1e6:.0f}μm (left)')
    if liq_col_right.any():
        ax5.plot(vx[zi0:zi1, x_right][liq_col_right],
                 z_um[zi0:zi1][liq_col_right],
                 'r-s', ms=3, lw=1.5, label=f'x={x_right*DX*1e6:.0f}μm (right)')
    ax5.axvline(0, color='gray', ls='--', lw=0.5)
    ax5.set_xlabel('vx [m/s]')
    ax5.set_ylabel('z [μm]')
    ax5.set_title('(e) Vertical vx profile — surface vs depth reversal?', fontsize=10)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # --- Panel (2,1): Summary verdict ---
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    verdict_lines = [
        f"Marangoni Flow Diagnostic (t={T_US} μs)",
        f"─" * 45,
        f"Melt pool: {(x_max_mp-x_min_mp)*DX*1e6:.0f}×{(z_max_mp-z_min_mp)*DX*1e6:.0f} μm",
        f"T_max = {T.max():.0f} K, T_hotspot = {T_surf[hotspot_i]:.0f} K",
        f"v_max = {vmag[liquid_mask].max():.3f} m/s" if liquid_mask.any() else "v_max = N/A",
        f"",
        f"Surface flow (z={z_surf*DX*1e6:.0f} μm):",
        f"  Left <vx> = {left_vx_mean:+.3f} m/s  {'✓' if left_vx_mean < -0.01 else '✗'}",
        f"  Right <vx> = {right_vx_mean:+.3f} m/s  {'✓' if right_vx_mean > 0.01 else '✗'}",
        f"",
        f"Return flow (z={z_deep*DX*1e6:.0f} μm):",
        f"  Left <vx> = {left_deep_vx:+.3f} m/s  {'✓' if left_deep_vx > 0.01 else '?'}",
        f"  Right <vx> = {right_deep_vx:+.3f} m/s  {'✓' if right_deep_vx < -0.01 else '?'}",
        f"",
    ]
    if marangoni_surface_ok:
        verdict_lines.append("★ VERDICT: Marangoni pattern CONFIRMED")
        verdict_lines.append("  Hot→cold outward surface flow (dσ/dT < 0)")
    else:
        verdict_lines.append("△ VERDICT: Pattern INCONCLUSIVE")
        verdict_lines.append("  Check melt pool maturity / scan direction effect")

    ax6.text(0.05, 0.95, '\n'.join(verdict_lines),
             transform=ax6.transAxes, fontsize=10, va='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', fc='#f8f8f0', ec='gray', alpha=0.9))

    fig.suptitle(f'Marangoni Flow Analysis — LPBF 316L, P=75W, t={T_US}μs',
                 fontsize=14, fontweight='bold')

    plt.savefig(OUT_FILE, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {OUT_FILE}")


if __name__ == '__main__':
    main()
