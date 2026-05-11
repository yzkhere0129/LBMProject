#!/usr/bin/env python3
"""Read VTK and check pressure (=ρ) above vs below airfoil at α=+8°."""
import sys, struct
import numpy as np
import matplotlib.pyplot as plt

vtk = sys.argv[1] if len(sys.argv) > 1 else 'output_naca_a8_kurtulus/snap_0032000.vtk'

# parse minimal VTK ImageData (legacy), big-endian floats
with open(vtk, 'rb') as f:
    raw = f.read()

def find(tag, raw, after=0):
    idx = raw.find(tag, after)
    return idx

hdr = raw[:raw.find(b'BINARY')+7]  # ASCII header

# parse header
def hdr_val(field):
    i = hdr.find(field.encode())
    if i < 0: return None
    line = hdr[i:hdr.find(b'\n', i)]
    return line.split()

dims = hdr_val('DIMENSIONS')
nx, ny, nz = int(dims[1]), int(dims[2]), int(dims[3])
spacing = hdr_val('SPACING')
dx = float(spacing[1])
n_cells = nx * ny * nz

# find SCALARS rho block
def read_scalar_field(name, raw):
    tag = f'SCALARS {name}'.encode()
    i = raw.find(tag)
    if i < 0: return None
    # skip past SCALARS line and LOOKUP_TABLE line
    eol = raw.find(b'\n', i); eol = raw.find(b'\n', eol+1)
    start = eol + 1
    arr = np.frombuffer(raw[start:start + n_cells*4], dtype='>f4').astype(np.float32)
    return arr.reshape(nz, ny, nx)

def read_vector_field(name, raw):
    tag = f'VECTORS {name}'.encode()
    i = raw.find(tag)
    if i < 0: return None
    eol = raw.find(b'\n', i)
    start = eol + 1
    arr = np.frombuffer(raw[start:start + n_cells*3*4], dtype='>f4').astype(np.float32)
    return arr.reshape(nz, ny, nx, 3)

rho = read_scalar_field('rho', raw)
u   = read_vector_field('velocity', raw)

if rho is None or u is None:
    print("Missing rho or velocity. Available:")
    for tag in [b'SCALARS', b'VECTORS']:
        i = 0
        while True:
            i = raw.find(tag, i)
            if i < 0: break
            line = raw[i:raw.find(b'\n', i)]
            print(' ', line.decode(errors='ignore'))
            i += 1
    sys.exit(1)

# midplane in z
kz = nz // 2
rho2 = rho[kz]    # ny x nx
u2 = u[kz]        # ny x nx x 3

# Airfoil at LE=(10,10), chord=1, α=+8°. Sample two cells:
# - one BELOW airfoil (world y < 9.95) at mid-chord x=10.5
# - one ABOVE airfoil (world y > 10.20) at mid-chord x=10.5
def cell_at(x, y):
    i, j = int(x/dx), int(y/dx)
    return rho2[j, i], u2[j, i]

print(f"Domain: {nx}x{ny}x{nz}, dx={dx}, file={vtk}")
print()
print("Pressure (=ρ/3) at slices around airfoil mid-chord (x=10.5):")
for y in [9.7, 9.8, 9.9, 10.0, 10.1, 10.2, 10.3]:
    r, uv = cell_at(10.5, y)
    print(f"  y={y:.2f}: ρ={r:.6f},  u=({uv[0]:+.4f}, {uv[1]:+.4f}, {uv[2]:+.4f})")

# Mean pressure above vs below airfoil chord line, in a small box
# airfoil is from x=10 to x=11. Take y range covering the airfoil region
i0, i1 = int(10.0/dx), int(11.0/dx)
# below the airfoil's lowest point (rotated TE at y=10.139, max thickness 0.06 at α=0,
# rotated could put lowest point near LE bottom y≈9.94). So below means y < 9.85.
# above the airfoil's highest point (rotated upper TE region y≈10.20).
j_below_lo, j_below_hi = int(9.50/dx), int(9.80/dx)
j_above_lo, j_above_hi = int(10.30/dx), int(10.60/dx)
p_below = rho2[j_below_lo:j_below_hi, i0:i1].mean()
p_above = rho2[j_above_lo:j_above_hi, i0:i1].mean()
print()
print(f"Mean ρ in box BELOW airfoil (y∈[9.50, 9.80]): {p_below:.6f}")
print(f"Mean ρ in box ABOVE airfoil (y∈[10.30, 10.60]): {p_above:.6f}")
print(f"Δρ = ρ_below - ρ_above = {p_below - p_above:+.6f}")
print(f"  +ve → higher pressure BELOW → POSITIVE Cl (correct for α>0)")
print(f"  -ve → higher pressure ABOVE → NEGATIVE Cl (WRONG for α>0)")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
extent = [0, nx*dx, 0, ny*dx]
# zoom around airfoil
zoom_extent = [9.5, 12.0, 9.0, 11.0]
i_z0, i_z1 = int(9.5/dx), int(12.0/dx)
j_z0, j_z1 = int(9.0/dx), int(11.0/dx)

# Pressure
p2 = (rho2 - 1.0) / 3.0  # gauge pressure relative to ρ=1
ax = axes[0]
im = ax.imshow(p2[j_z0:j_z1, i_z0:i_z1], origin='lower',
               extent=zoom_extent, cmap='RdBu_r',
               vmin=-0.001, vmax=0.001)
plt.colorbar(im, ax=ax, label='gauge pressure (ρ-1)/3')
ax.set_title(f'Pressure field (zoomed). α=+8°.\n'
             f'Red=high (should be UNDER airfoil), Blue=low (should be ABOVE)')
ax.plot([10.0, 10.99], [10.0, 10.139], 'k-', lw=2)
ax.plot([10.0], [10.0], 'go', markersize=8); ax.plot([10.99], [10.139], 'ms', markersize=8)
ax.set_xlabel('x'); ax.set_ylabel('y')

# u_y field
uy = u2[..., 1]
ax = axes[1]
im = ax.imshow(uy[j_z0:j_z1, i_z0:i_z1], origin='lower',
               extent=zoom_extent, cmap='RdBu_r',
               vmin=-0.05, vmax=0.05)
plt.colorbar(im, ax=ax, label='u_y [LU]')
ax.set_title('u_y field (zoomed). α=+8°.\n'
             'For Cl>0, downwash should be BEHIND airfoil (u_y < 0 in wake)')
ax.plot([10.0, 10.99], [10.0, 10.139], 'k-', lw=2)
ax.plot([10.0], [10.0], 'go', markersize=8); ax.plot([10.99], [10.139], 'ms', markersize=8)
ax.set_xlabel('x'); ax.set_ylabel('y')

fig.tight_layout()
fig.savefig(vtk.replace('.vtk', '_pressure_diag.png'), dpi=150)
print(f"\nSaved {vtk.replace('.vtk', '_pressure_diag.png')}")
