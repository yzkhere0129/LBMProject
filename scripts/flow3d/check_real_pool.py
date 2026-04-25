#!/usr/bin/env python3
"""Probe the real liquid metal pool (T>=T_liq AND fill>0.5) at one LBM frame.
Diagnoses any bbox-metric ambiguity by separating "hot gas in keyhole cavity"
from "actual liquid metal pool"."""
import sys
import numpy as np
import pyvista as pv

if len(sys.argv) < 2:
    print(__doc__); sys.exit(1)
path = sys.argv[1]
T_liq = float(sys.argv[2]) if len(sys.argv) > 2 else 1697.15

m = pv.read(path)
nx, ny, nz = m.dimensions
dx, dy, dz = m.spacing
ox, oy, oz = m.origin
T = np.asarray(m.point_data['temperature']).reshape((nx,ny,nz), order='F')
fill = np.asarray(m.point_data['fill_level']).reshape((nx,ny,nz), order='F')

real_pool = (T >= T_liq) & (fill > 0.5)
T_only_pool = (T >= T_liq)
keyhole_vapor = T_only_pool & ~real_pool  # hot but no metal

def bbox(mask):
    if not mask.any(): return (0,0,0,0)
    ii, jj, kk = np.where(mask)
    return ((ii.max()-ii.min()+1)*dx*1e6,
            (jj.max()-jj.min()+1)*dy*1e6,
            (kk.max()-kk.min()+1)*dz*1e6,
            int(mask.sum()))

L1, W1, D1, n1 = bbox(real_pool)
L2, W2, D2, n2 = bbox(T_only_pool)
L3, W3, D3, n3 = bbox(keyhole_vapor)

print(f'File: {path}')
print(f'Domain: {nx}×{ny}×{nz}, dx={dx*1e6:.1f}μm')
print(f'')
print(f'{"Region":>30} {"L":>5} {"W":>5} {"D":>5}  cells')
print(f'{"-"*60}')
print(f'{"Real liquid metal":>30} {L1:5.0f} {W1:5.0f} {D1:5.0f}  {n1:>7}')
print(f'{"T-only (incl keyhole vapor)":>30} {L2:5.0f} {W2:5.0f} {D2:5.0f}  {n2:>7}')
print(f'{"Keyhole vapor only (gas)":>30} {L3:5.0f} {W3:5.0f} {D3:5.0f}  {n3:>7}')
print(f'')
print(f'T_max global:  {T.max():.0f} K')
print(f'T_max in liquid: {T[real_pool].max() if real_pool.any() else 0:.0f} K')

# Volume balance: Δh × L × W ≈ V_keyhole_displaced
# Compute crater (cells with fill<0.5 below substrate)
substrate_top_k = int(np.where(fill > 0.5)[2].max()) if (fill > 0.5).any() else nz - 1
# Roughly: initial top index = 80 (sim_line_scan_316L M16d/M17)
init_top_k = 80
crater = (fill < 0.5)[:, :, :init_top_k+1]
vol_crater = int(crater.sum()) * dx*dy*dz * 1e18  # μm³
print(f'Crater volume (fill<0.5 below init top): {vol_crater:.2e} μm³')
