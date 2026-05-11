#!/usr/bin/env python3
"""Zoomed velocity component visualization near airfoil LE."""
import sys, numpy as np, matplotlib.pyplot as plt

vtk = sys.argv[1] if len(sys.argv) > 1 else 'output_naca_a8_kurtulus/snap_0032000.vtk'
with open(vtk) as f:
    lines = f.readlines()

nx = ny = nz = 0; dx = 0.0; data_start = -1
for i, ln in enumerate(lines):
    p = ln.split()
    if p[:1] == ['DIMENSIONS']: nx, ny, nz = int(p[1]), int(p[2]), int(p[3])
    elif p[:1] == ['SPACING']: dx = float(p[1])
    elif p[:1] == ['VECTORS']: data_start = i + 1; break

n = nx * ny * nz
v = np.empty((n, 3), dtype=np.float32)
for k, ln in enumerate(lines[data_start:data_start + n]):
    p = ln.split(); v[k, 0] = float(p[0]); v[k, 1] = float(p[1]); v[k, 2] = float(p[2])
v = v.reshape(nz, ny, nx, 3)
ux, uy = v[nz//2, ..., 0], v[nz//2, ..., 1]

i0, i1 = int(9.5/dx), int(11.5/dx)
j0, j1 = int(9.4/dx), int(10.6/dx)
extent = [i0*dx, i1*dx, j0*dx, j1*dx]

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
for ax, fld, name, vmin, vmax in [
    (axes[0], ux[j0:j1, i0:i1], 'u_x', -0.02, 0.07),
    (axes[1], uy[j0:j1, i0:i1], 'u_y', -0.04, 0.04),
    (axes[2], np.hypot(ux[j0:j1, i0:i1], uy[j0:j1, i0:i1]), '|u|', 0, 0.07),
]:
    cmap = 'RdBu_r' if 'u_x' in name or 'u_y' in name else 'viridis'
    im = ax.imshow(fld, origin='lower', extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label=name)
    ax.plot([10, 10.99], [10, 10.139], 'k-', lw=2)
    ax.plot(10, 10, 'go', markersize=10, label='LE')
    ax.plot(10.99, 10.139, 'ms', markersize=10, label='TE')
    ax.set_title(f'{name}, α=+8°')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    if 'u_x' in name: ax.legend()
fig.tight_layout()
out = vtk.replace('.vtk', '_naca_zoom.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved {out}")
