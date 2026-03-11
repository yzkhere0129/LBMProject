#!/usr/bin/env python3
"""
Plot Taylor-Green 2D vortex vorticity field at t=0 and t=tau_visc.

Produces a two-panel figure showing:
  - Left:  Vorticity at t=0 (initial vortex array) with its own color scale
  - Right: Vorticity at t=tau_visc (decayed vortices) with its own color scale
Each panel uses its own symmetric color range so decayed structure is visible.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def load_field(csv_path):
    """Load CSV and reshape into 2D arrays."""
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    i_col = data[:, 0].astype(int)
    j_col = data[:, 1].astype(int)
    x_col = data[:, 2]
    y_col = data[:, 3]
    ux_col = data[:, 4]
    uy_col = data[:, 5]
    vort_col = data[:, 6]

    nx = i_col.max() + 1
    ny = j_col.max() + 1

    X = x_col.reshape(ny, nx)
    Y = y_col.reshape(ny, nx)
    Ux = ux_col.reshape(ny, nx)
    Uy = uy_col.reshape(ny, nx)
    Vort = vort_col.reshape(ny, nx)
    return X, Y, Ux, Uy, Vort

# ---- Load data ----
base = os.path.dirname(os.path.abspath(__file__))
X0, Y0, Ux0, Uy0, Vort0 = load_field(os.path.join(base, "tg_t0.csv"))
Xf, Yf, Uxf, Uyf, Vortf = load_field(os.path.join(base, "tg_tfinal.csv"))

# Convert to mm
X0_mm = X0 * 1e3
Y0_mm = Y0 * 1e3
Xf_mm = Xf * 1e3
Yf_mm = Yf * 1e3

vmax0 = np.max(np.abs(Vort0))
vmaxf = np.max(np.abs(Vortf))
decay_ratio = vmaxf / vmax0

# ---- Figure: 2-panel vorticity with independent scales ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# --- Panel 1: t = 0 ---
ax = axes[0]
levels0 = np.linspace(-vmax0, vmax0, 61)
cf0 = ax.contourf(X0_mm, Y0_mm, Vort0, levels=levels0, cmap='RdBu_r', extend='both')
cb0 = fig.colorbar(cf0, ax=ax, shrink=0.92, pad=0.02)
cb0.set_label(r'$\omega_z$ [1/s]', fontsize=11)

# Velocity arrows
stride = 6
ax.quiver(X0_mm[::stride, ::stride], Y0_mm[::stride, ::stride],
          Ux0[::stride, ::stride], Uy0[::stride, ::stride],
          color='k', alpha=0.45, scale=1.5, width=0.003)

ax.set_xlim(X0_mm.min(), X0_mm.max())
ax.set_ylim(Y0_mm.min(), Y0_mm.max())
ax.set_aspect('equal')
ax.set_xlabel('x [mm]', fontsize=12)
ax.set_ylabel('y [mm]', fontsize=12)
ax.set_title(r'$t = 0$ (initial)', fontsize=13, fontweight='bold')
ax.tick_params(labelsize=10)

# --- Panel 2: t = tau_visc (own color scale) ---
ax = axes[1]
levelsf = np.linspace(-vmaxf, vmaxf, 61)
cff = ax.contourf(Xf_mm, Yf_mm, Vortf, levels=levelsf, cmap='RdBu_r', extend='both')
cbf = fig.colorbar(cff, ax=ax, shrink=0.92, pad=0.02)
cbf.set_label(r'$\omega_z$ [1/s]', fontsize=11)

ax.quiver(Xf_mm[::stride, ::stride], Yf_mm[::stride, ::stride],
          Uxf[::stride, ::stride], Uyf[::stride, ::stride],
          color='k', alpha=0.45, scale=0.03, width=0.003)

ax.set_xlim(Xf_mm.min(), Xf_mm.max())
ax.set_ylim(Yf_mm.min(), Yf_mm.max())
ax.set_aspect('equal')
ax.set_xlabel('x [mm]', fontsize=12)
ax.set_ylabel('y [mm]', fontsize=12)
ax.set_title(r'$t = \tau_{\mathrm{visc}}$ (decayed)', fontsize=13, fontweight='bold')
ax.tick_params(labelsize=10)

fig.suptitle(
    r'Taylor-Green Vortex Decay, $Re = 100$' + '\n'
    + f'Peak vorticity: {vmax0:.0f} '
    + r'$\rightarrow$'
    + f' {vmaxf:.1f} [1/s]'
    + f'  (ratio = {decay_ratio:.4f}, '
    + r'analytical $e^{-4}$'
    + f' = {np.exp(-4):.4f})',
    fontsize=12, fontweight='bold', y=1.03)

plt.tight_layout()

out_path = os.path.join(base, "taylor_green_vortex.png")
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.close()
