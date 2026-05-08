#!/usr/bin/env python3
"""Quick diagnostic: replicate the host-side QBB q-fraction calculation
in pure Python and print a histogram + a single-row dump along the
cylinder mid-line. Lets us check if the C++ helper is producing values
in (0, 1] with a sensible distribution centred ~0.5.

Usage:  python3 dump_qfrac_diag.py
"""
import numpy as np

# Schäfer-Turek 2D-2 layout (D/dx=20)
nx, ny, nz = 440, 83, 4
dx = 0.005
cx, cy = 0.2, 0.2
R = 0.05

# D3Q19 host tables - copy from src/core/lattice/d3q19.cu
hex_ = [0, 1,-1, 0, 0, 0, 0,  1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0]
hey  = [0, 0, 0, 1,-1, 0, 0,  1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1]
hez  = [0, 0, 0, 0, 0, 1,-1,  0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1]

mask = np.zeros((nz, ny, nx), dtype=np.uint8)
for j in range(ny):
    yC = (j + 0.5) * dx
    dy = yC - cy
    for i in range(nx):
        xC = (i + 0.5) * dx
        dxc = xC - cx
        if dxc*dxc + dy*dy <= R*R:
            mask[:, j, i] = 1

n_solid = mask.sum()
print(f"solid cells = {n_solid}  (expected ≈ {np.pi*R*R/dx/dx*nz:.1f})")

# Compute qfrac like the C++ helper
all_qf = []
mid_row_q = []
mid_j = int(round(cy/dx))
for j in range(ny):
    yC = (j + 0.5) * dx
    for i in range(nx):
        if mask[0, j, i] != 0:
            continue
        xC = (i + 0.5) * dx
        for q in range(1, 19):
            ni, nj, nk = i + hex_[q], j + hey[q], 0 + hez[q]
            if not (0<=ni<nx and 0<=nj<ny and 0<=nk<nz):
                continue
            if mask[nk, nj, ni] == 0:
                continue
            dxL, dyL = hex_[q]*dx, hey[q]*dx
            A = dxL*dxL + dyL*dyL
            if A < 1e-20:
                continue
            dx0, dy0 = xC - cx, yC - cy
            B = 2.0 * (dxL*dx0 + dyL*dy0)
            C = dx0*dx0 + dy0*dy0 - R*R
            disc = B*B - 4*A*C
            if disc < 0:
                continue
            sd = np.sqrt(disc)
            s1 = (-B - sd) / (2*A)
            s2 = (-B + sd) / (2*A)
            s = 1.0
            if 1e-6 < s1 <= 1.0:
                s = s1
            elif 1e-6 < s2 <= 1.0:
                s = s2
            s = max(1e-6, min(1.0, s))
            all_qf.append(s)
            if j == mid_j and abs(xC - cx) < 0.075:
                mid_row_q.append((i, q, s))

all_qf = np.array(all_qf)
print(f"curved links: {len(all_qf)}  (per cylinder side, ≈ 2π R/dx · 4-z = "
      f"{int(2*np.pi*R/dx*nz*1.5)} avg)")
print(f"qfrac stats: min={all_qf.min():.3f}  median={np.median(all_qf):.3f}  "
      f"max={all_qf.max():.3f}  mean={all_qf.mean():.3f}")
hist, edges = np.histogram(all_qf, bins=10, range=(0, 1))
print("histogram (10 bins, 0->1):")
for h, e in zip(hist, edges):
    print(f"  [{e:.2f}, {e+0.1:.2f})  {h:5d}  {'#'*int(50*h/max(hist))}")
print()
print(f"Mid-row j={mid_j} (y={mid_j*dx+0.5*dx:.4f}) sample (i, q, qfrac):")
for entry in mid_row_q[:15]:
    print(f"  i={entry[0]:3d}  q={entry[1]:2d}  e=({hex_[entry[1]]:+d},{hey[entry[1]]:+d},{hez[entry[1]]:+d})  qfrac={entry[2]:.4f}")
