#!/usr/bin/env python3
"""Sprint-3 redirect: probe trailing-edge groove physics, not keyhole tip.

Sprint-2 architectural diagnosis was REFUTED — actual keyhole tip kappa
is 1.7e5 m^-1 (sharp). The real question is why the groove BEHIND the
laser doesn't fill.

For a frame at quasi-steady-state, scan multiple cross-sections behind
the laser at x_offset = -50, -100, -150, -200 um from laser. At each
slab, report:
  - centerline depth z(y=mid)
  - liquid-fraction profile (is it solid yet?)
  - wall curvature (kappa at f=0.5 isosurface)
  - velocity v_z at centerline (is liquid trying to rise?)
  - Darcy resistance (1-fl)^2 / fl^3 at centerline
  - capillary force vs Marangoni force vs Darcy at centerline interface

Usage: python diag_trailing_edge.py <vtk> [z0_um=160] [v_scan=0.8]
"""
import sys
import numpy as np
import pyvista as pv

if len(sys.argv) < 2:
    print(__doc__); sys.exit(1)
path = sys.argv[1]
z0_um = float(sys.argv[2]) if len(sys.argv) > 2 else 160.0
v_scan = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8  # m/s
laser_start_um = 500.0

print(f"Reading {path} ...")
m = pv.read(path)
nx, ny, nz = m.dimensions
dx, dy, dz = m.spacing
ox, oy, oz = m.origin
print(f"Grid {nx}x{ny}x{nz}, dx={dx*1e6:.2f}um")

f    = np.asarray(m.point_data['fill_level']).reshape((nx, ny, nz), order='F')
T    = np.asarray(m.point_data['temperature']).reshape((nx, ny, nz), order='F')
kappa= np.asarray(m.point_data['curvature']).reshape((nx, ny, nz), order='F')
fl   = np.asarray(m.point_data['liquid_fraction']).reshape((nx, ny, nz), order='F')
u    = np.asarray(m.point_data['velocity']).reshape((nx, ny, nz, 3), order='F')

# Convert lattice-unit velocity to physical (FluidLBM stores LU)
# dx_phys = dx, dt = 80ns
dt_ns = 80.0
v_factor = dx / (dt_ns * 1e-9)  # m/s per LU

step_str = path.split('_')[-1].split('.')[0]
step = int(step_str)
t_us = step * dt_ns * 1e-3
laser_x_um = laser_start_um + v_scan * t_us
print(f"step={step}, t={t_us:.0f}us, laser_x={laser_x_um:.0f}um")

# Free surface z(x,y)
flipped = (f > 0.5)[:, :, ::-1]
has_metal = flipped.any(axis=2)
k_top = nz - 1 - np.argmax(flipped, axis=2)
z_surf_um = (oz + k_top * dz) * 1e6
z_surf_um = np.where(has_metal, z_surf_um, np.nan)

mid_j = ny // 2
print(f"")
print(f"{'x_off':>6} {'x_um':>5} {'z_um':>6} {'fl':>5} {'kappa':>9} {'P_cap':>7} {'v_z_LU':>7} {'v_z_mps':>7} {'Darcy':>9} {'state':>10}")
print(f"{'um':>6} {'':>5} {'':>6} {'':>5} {'1/m':>9} {'kPa':>7} {'':>7} {'m/s':>7} {'1/m^2':>9}")
print('-' * 90)

# Probe at offsets -10, -30, -50, -100, -150, -200, -300 um from laser (groove zone)
offsets_um = [-10, -30, -50, -80, -100, -150, -200, -300, -400, -500]

# Carman-Kozeny constant — match defaults
K_CK = 1.6e9  # 1/m^2 typical for 50um grain

for off in offsets_um:
    x_um = laser_x_um + off
    if x_um < laser_start_um or x_um > (nx - 2) * dx * 1e6:
        continue
    i = int((x_um * 1e-6 - ox) / dx)
    i = max(0, min(i, nx - 1))

    # surface k at this slab centerline
    if not has_metal[i, mid_j]:
        continue
    k_top_ij = k_top[i, mid_j]

    # interface cell: just at f=0.5 layer, take cell at k_top
    z_surf_loc = z_surf_um[i, mid_j]
    fl_loc = float(fl[i, mid_j, k_top_ij])
    T_loc  = float(T[i, mid_j, k_top_ij])

    # Find interface cell properly: 0.05 < f < 0.95 nearest to surface
    fcol = f[i, mid_j, :]
    int_mask = (fcol > 0.05) & (fcol < 0.95)
    if int_mask.any():
        ks = np.where(int_mask)[0]
        # pick highest k that is interface
        k_int = ks.max()
    else:
        k_int = k_top_ij

    kappa_loc = abs(float(kappa[i, mid_j, k_int]))
    fl_int    = float(fl[i, mid_j, k_int])
    T_int     = float(T[i, mid_j, k_int])
    vz_LU     = float(u[i, mid_j, k_int, 2])
    vz_mps    = vz_LU * v_factor

    sigma = 1.74
    P_cap = sigma * kappa_loc / 1e3  # kPa

    # Darcy (Carman-Kozeny): K = K_CK * (1-fl)^2 / (fl^3 + eps)
    fl_eff = max(fl_int, 1e-3)
    K_LU_per_m2 = K_CK * (1.0 - fl_eff)**2 / (fl_eff**3 + 1e-3)

    # Frozen state classification
    if T_int > 1697:
        state = "LIQUID"
    elif T_int > 1500:
        state = "MUSHY"
    else:
        state = "SOLID"

    print(f"{off:>6} {x_um:>5.0f} {z_surf_loc:>6.1f} {fl_int:>5.2f} "
          f"{kappa_loc:>9.2e} {P_cap:>7.1f} {vz_LU:>7.4f} {vz_mps:>7.2f} "
          f"{K_LU_per_m2:>9.2e} {state:>10}")

print(f"")
print(f"=== INTERPRETATION ===")
print(f"If groove cells (z<<z0) are SOLID at t={t_us:.0f}us:")
print(f"  -> liquid froze before capillary could refill")
print(f"  -> root cause = solidification timing, NOT capillary force weakness")
print(f"")
print(f"If groove cells are LIQUID with strong capillary (P_cap>50kPa) but")
print(f"v_z is small or downward:")
print(f"  -> Marangoni outflow is winning over capillary inflow")
print(f"  -> root cause = force balance at interface, not capillary magnitude")
