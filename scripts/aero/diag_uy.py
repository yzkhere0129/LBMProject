#!/usr/bin/env python3
"""Read ASCII VTK velocity, check u_y wake direction. For Cl>0 wake = downwash."""
import sys
import numpy as np
import matplotlib.pyplot as plt

vtk = sys.argv[1] if len(sys.argv) > 1 else 'output_naca_a8_kurtulus/snap_0032000.vtk'

with open(vtk) as f:
    lines = f.readlines()

# Parse header
nx = ny = nz = 0
dx = 0.0
data_start = -1
for i, ln in enumerate(lines):
    parts = ln.split()
    if len(parts) >= 4 and parts[0] == 'DIMENSIONS':
        nx, ny, nz = int(parts[1]), int(parts[2]), int(parts[3])
    elif len(parts) >= 4 and parts[0] == 'SPACING':
        dx = float(parts[1])
    elif parts[:1] == ['VECTORS']:
        data_start = i + 1
        break

n_cells = nx * ny * nz
print(f"VTK: {nx}x{ny}x{nz}, dx={dx}, file={vtk}")

vec = np.empty((n_cells, 3), dtype=np.float32)
for k, ln in enumerate(lines[data_start:data_start + n_cells]):
    p = ln.split()
    vec[k, 0] = float(p[0]); vec[k, 1] = float(p[1]); vec[k, 2] = float(p[2])

vec = vec.reshape(nz, ny, nx, 3)
kz = nz // 2
ux = vec[kz, ..., 0]; uy = vec[kz, ..., 1]

# Wake region just behind airfoil: x in [11.2, 11.8] (behind TE at x=11)
# For α=+8°, wake center is around y=10 (slightly above for downward-curving wake)
i_wake0, i_wake1 = int(11.2/dx), int(11.8/dx)
j_band_lo, j_band_hi = int(9.5/dx), int(10.5/dx)
uy_wake = uy[j_band_lo:j_band_hi, i_wake0:i_wake1]
print(f"\nWake region (x∈[11.2,11.8], y∈[9.5,10.5]):")
print(f"  Mean u_y = {uy_wake.mean():+.5f} LU")
print(f"  Min  u_y = {uy_wake.min():+.5f} LU")
print(f"  Max  u_y = {uy_wake.max():+.5f} LU")
print(f"  + Mean u_y < 0 → DOWNWASH → Cl > 0 (correct for α>0)")
print(f"  + Mean u_y > 0 → UPWASH   → Cl < 0 (WRONG for α>0)")

# Also check pressure proxy: stagnation should be on lower-front of airfoil for α>0
# Sample u magnitude near LE on lower vs upper surface
def speed_at(x, y):
    i, j = int(x/dx), int(y/dx)
    return np.hypot(ux[j, i], uy[j, i])

print(f"\nFlow speed near LE (stagnation indicator):")
print(f"  Just BELOW LE  (10.05, 9.95): |u| = {speed_at(10.05, 9.95):.5f}  ← should be SMALL for +AOA")
print(f"  Just ABOVE LE  (10.05, 10.05): |u| = {speed_at(10.05, 10.05):.5f} ← should be LARGER for +AOA")
print(f"  Behind airfoil (11.5, 10.0):   |u| = {speed_at(11.5, 10.0):.5f}")

# Check far-field at inlet to confirm true U_inf
i_inlet = 2
print(f"\nInlet region (i=2):")
print(f"  Mean u_x = {ux[:, i_inlet].mean():+.5f}")
print(f"  Mean u_y = {uy[:, i_inlet].mean():+.5f}  (should be ≈0 for clean inlet)")
