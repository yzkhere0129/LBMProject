#!/usr/bin/env python3
"""
Taylor-Green Vortex Decay -- Analytical Solution Visualization.

Two-panel figure showing vorticity contours at t=0 (initial checkerboard)
and at a later time (viscous decay).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N = 256           # grid resolution
U0 = 1.0          # characteristic velocity
nu = 0.05         # kinematic viscosity
L = 2.0 * np.pi   # domain length

# Times to visualise
t0 = 0.0
t1 = 8.0   # later time -- shows significant decay

# Decay time scale: T_decay = 1/(2*nu) = 10.0 for nu=0.05
T_decay = 1.0 / (2.0 * nu)

# Grid
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')


def taylor_green_fields(X, Y, t, U0, nu):
    """Return (ux, uy, vorticity) from the analytical TG solution."""
    decay = np.exp(-2.0 * nu * t)
    ux = U0 * np.sin(X) * np.cos(Y) * decay
    uy = -U0 * np.cos(X) * np.sin(Y) * decay
    # omega_z = dv/dx - du/dy = 2*U0*sin(x)*sin(y)*exp(-2*nu*t)
    vort = 2.0 * U0 * np.sin(X) * np.sin(Y) * decay
    return ux, uy, vort


# Compute fields
ux0, uy0, vort0 = taylor_green_fields(X, Y, t0, U0, nu)
ux1, uy1, vort1 = taylor_green_fields(X, Y, t1, U0, nu)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

# Symmetric colormap normalisation anchored at zero
vmax0 = np.max(np.abs(vort0))
vmax1 = vmax0  # use same scale so decay is visually obvious

norm0 = TwoSlopeNorm(vmin=-vmax0, vcenter=0, vmax=vmax0)
norm1 = TwoSlopeNorm(vmin=-vmax0, vcenter=0, vmax=vmax0)

# Contour levels (shared)
n_levels = 30
levels = np.linspace(-vmax0, vmax0, n_levels + 1)

# --- Panel 1: t = 0 ---
ax = axes[0]
cf0 = ax.contourf(X, Y, vort0, levels=levels, cmap='RdBu_r', norm=norm0, extend='both')
# Overlay streamlines
skip = max(1, N // 28)
ax.streamplot(x, y, ux0.T, uy0.T, color='k', density=1.6,
              linewidth=0.5, arrowsize=0.8, arrowstyle='->')
ax.set_title(r'$t = 0$', fontsize=15, fontweight='bold')
ax.set_xlabel(r'$x$', fontsize=13)
ax.set_ylabel(r'$y$', fontsize=13)
ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_aspect('equal')
ax.set_xticks([0, np.pi, 2*np.pi])
ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
ax.set_yticks([0, np.pi, 2*np.pi])
ax.set_yticklabels(['0', r'$\pi$', r'$2\pi$'])
ax.tick_params(labelsize=11)

# --- Panel 2: t = t1 ---
ax = axes[1]
cf1 = ax.contourf(X, Y, vort1, levels=levels, cmap='RdBu_r', norm=norm1, extend='both')
ax.streamplot(x, y, ux1.T, uy1.T, color='k', density=1.6,
              linewidth=0.5, arrowsize=0.8, arrowstyle='->')
decay_pct = (1.0 - np.exp(-2.0 * nu * t1)) * 100
ax.set_title(rf'$t = {t1:.0f}$ $(\approx {t1/T_decay:.1f}\,T_{{\mathrm{{decay}}}})$',
             fontsize=15, fontweight='bold')
ax.set_xlabel(r'$x$', fontsize=13)
ax.set_ylabel(r'$y$', fontsize=13)
ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_aspect('equal')
ax.set_xticks([0, np.pi, 2*np.pi])
ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
ax.set_yticks([0, np.pi, 2*np.pi])
ax.set_yticklabels(['0', r'$\pi$', r'$2\pi$'])
ax.tick_params(labelsize=11)

# Shared colorbar
cbar = fig.colorbar(cf0, ax=axes, shrink=0.85, pad=0.03, aspect=30)
cbar.set_label(r'Vorticity $\omega_z$', fontsize=13)
cbar.ax.tick_params(labelsize=11)

# Annotations
fig.text(0.28, 0.01,
         rf'$U_0={U0}$,  $\nu={nu}$,  '
         rf'$T_{{\mathrm{{decay}}}} = 1/(2\nu) = {T_decay:.0f}$',
         ha='center', fontsize=11, color='#444444')

fig.text(0.68, 0.01,
         rf'Amplitude decayed to {np.exp(-2*nu*t1)*100:.1f}% of initial',
         ha='center', fontsize=11, color='#444444')

fig.suptitle('Taylor-Green Vortex Decay (Analytical Solution)',
             fontsize=16, fontweight='bold', y=1.01)

plt.tight_layout(rect=[0, 0.04, 1, 0.98])
out_path = '/home/yzk/LBMProject/scripts/viz/taylor_green_vortex.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_path}")
plt.close()
