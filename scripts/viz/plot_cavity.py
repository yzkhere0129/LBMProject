#!/usr/bin/env python3
"""
Plot lid-driven cavity Re=100 velocity field.

Produces:
  - Velocity magnitude contourf
  - Streamline overlay
  - Shows primary vortex and corner vortices
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import os

# ---- Load CSV ----
csv_path = os.path.join(os.path.dirname(__file__), "cavity_velocity.csv")
data = np.loadtxt(csv_path, delimiter=',', skiprows=1)

i_col = data[:, 0].astype(int)
j_col = data[:, 1].astype(int)
x_col = data[:, 2]
y_col = data[:, 3]
ux_col = data[:, 4]
uy_col = data[:, 5]

nx = i_col.max() + 1
ny = j_col.max() + 1

# Reshape to 2D grids (row=j, col=i)
X = x_col.reshape(ny, nx)
Y = y_col.reshape(ny, nx)
Ux = ux_col.reshape(ny, nx)
Uy = uy_col.reshape(ny, nx)

# Lid velocity for normalization
U_lid = np.max(np.abs(Ux))
# Normalize
Ux_n = Ux / U_lid
Uy_n = Uy / U_lid
speed = np.sqrt(Ux_n**2 + Uy_n**2)

# ---- Figure ----
fig, ax = plt.subplots(1, 1, figsize=(8, 7.5))

# Velocity magnitude contour
levels = np.linspace(0, 1.0, 51)
cf = ax.contourf(X, Y, speed, levels=levels, cmap='RdYlBu_r', extend='max')
cb = fig.colorbar(cf, ax=ax, shrink=0.88, pad=0.02)
cb.set_label(r'$|\mathbf{u}|/U_{\mathrm{lid}}$', fontsize=13)

# Streamlines (subsample for clarity)
stride = 2
xs = X[::stride, ::stride]
ys = Y[::stride, ::stride]
us = Ux_n[::stride, ::stride]
vs = Uy_n[::stride, ::stride]

ax.streamplot(xs[0, :], ys[:, 0], us, vs,
              color='k', linewidth=0.5, density=2.5, arrowsize=0.6,
              arrowstyle='->')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.set_xlabel(r'$x / L$', fontsize=13)
ax.set_ylabel(r'$y / L$', fontsize=13)
ax.set_title(r'Lid-Driven Cavity Flow, $Re = 100$', fontsize=15, fontweight='bold')
ax.tick_params(labelsize=11)

# Annotate primary vortex location (find minimum speed in interior)
mask = (X > 0.2) & (X < 0.8) & (Y > 0.3) & (Y < 0.9)
speed_masked = np.where(mask, speed, 1e10)
ij_min = np.unravel_index(np.argmin(speed_masked), speed_masked.shape)
vx_center = X[ij_min]
vy_center = Y[ij_min]
ax.plot(vx_center, vy_center, 'k+', markersize=12, markeredgewidth=2)
ax.annotate(f'Primary vortex\n({vx_center:.2f}, {vy_center:.2f})',
            xy=(vx_center, vy_center),
            xytext=(vx_center + 0.18, vy_center + 0.08),
            fontsize=10, color='k',
            arrowprops=dict(arrowstyle='->', color='k', lw=1.2))

plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), "lid_driven_cavity.png")
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.close()
