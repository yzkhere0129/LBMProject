#!/usr/bin/env python3
"""
Plot natural convection visualization:
  - Panel 1: Temperature isotherms
  - Panel 2: Velocity streamlines + magnitude
Saved as a single figure natural_convection.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize

# Publication-quality settings
mpl.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.5,
})

# Load data
df = pd.read_csv('/home/yzk/LBMProject/scripts/viz/natural_convection_data.csv')

nx = df['ix'].max() + 1
ny = df['iy'].max() + 1

# Reshape into 2D arrays (note: CSV is ordered i-fast, j-slow)
x = df['x'].values.reshape(ny, nx)
y = df['y'].values.reshape(ny, nx)
theta = df['theta'].values.reshape(ny, nx)
T_K = df['T_K'].values.reshape(ny, nx)
ux = df['ux_star'].values.reshape(ny, nx)
uy = df['uy_star'].values.reshape(ny, nx)
speed = np.sqrt(ux**2 + uy**2)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                          gridspec_kw={'wspace': 0.35})

# =========================================================================
# Panel 1: Temperature isotherms
# =========================================================================
ax1 = axes[0]

# Filled contour for background
cf = ax1.contourf(x, y, theta, levels=30, cmap='RdYlBu_r')
cb1 = plt.colorbar(cf, ax=ax1, fraction=0.046, pad=0.04)
cb1.set_label(r'$\theta = (T - T_c)/(T_h - T_c)$')

# Isotherm contour lines
levels_iso = np.linspace(0.05, 0.95, 10)
cs = ax1.contour(x, y, theta, levels=levels_iso, colors='k', linewidths=0.6, alpha=0.6)
ax1.clabel(cs, inline=True, fontsize=7, fmt='%.2f')

ax1.set_xlabel('x / L')
ax1.set_ylabel('y / H')
ax1.set_title(r'Isotherms ($Ra = 10^3$)')
ax1.set_aspect('equal')

# Wall labels
ax1.text(-0.06, 0.5, r'$T_h$', transform=ax1.transAxes, fontsize=12,
         color='red', fontweight='bold', va='center', rotation=90)
ax1.text(1.02, 0.5, r'$T_c$', transform=ax1.transAxes, fontsize=12,
         color='blue', fontweight='bold', va='center', rotation=90)
ax1.text(0.5, -0.08, 'adiabatic', transform=ax1.transAxes, fontsize=9,
         color='gray', ha='center')
ax1.text(0.5, 1.04, 'adiabatic', transform=ax1.transAxes, fontsize=9,
         color='gray', ha='center')

# =========================================================================
# Panel 2: Streamlines + velocity magnitude
# =========================================================================
ax2 = axes[1]

# Velocity magnitude background
cf2 = ax2.contourf(x, y, speed, levels=30, cmap='inferno')
cb2 = plt.colorbar(cf2, ax=ax2, fraction=0.046, pad=0.04)
cb2.set_label(r'$|\mathbf{u}^*|$')

# Streamlines -- need perfectly evenly spaced coordinates for streamplot
x_lin = np.linspace(0, 1, nx)
y_lin = np.linspace(0, 1, ny)
strm = ax2.streamplot(x_lin, y_lin, ux, uy,
                       color='white', density=1.5,
                       linewidth=1.0,
                       arrowsize=0.8, arrowstyle='->')

ax2.set_xlabel('x / L')
ax2.set_ylabel('y / H')
ax2.set_title(r'Streamlines + $|\mathbf{u}^*|$ ($Ra = 10^3$)')
ax2.set_aspect('equal')

# Nusselt number annotation (computed from temperature gradient at hot wall)
# dT/dx at x=0: (-3T0 + 4T1 - T2)/2
L = float(nx - 1)
Nu_local = np.zeros(ny)
for j in range(1, ny - 1):
    T0 = T_K[j, 0]
    T1 = T_K[j, 1]
    T2 = T_K[j, 2]
    dTdx = (-3.0 * T0 + 4.0 * T1 - T2) / 2.0
    delta_T = T_K[0, 0] - T_K[0, -1]  # T_hot - T_cold
    Nu_local[j] = -dTdx * L / delta_T

Nu_avg = Nu_local[1:-1].mean()
ax2.text(0.03, 0.03,
         f'$Nu_{{avg}} = {Nu_avg:.3f}$\n(benchmark: 1.118)',
         transform=ax2.transAxes, fontsize=10,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
         verticalalignment='bottom')

# Max velocities
ux_max = np.abs(ux).max()
uy_max = np.abs(uy).max()
ax1.text(0.03, 0.03,
         f'$u^*_{{max}} = {ux_max:.2f}$\n$v^*_{{max}} = {uy_max:.2f}$',
         transform=ax1.transAxes, fontsize=9,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
         verticalalignment='bottom')

fig.suptitle('Natural Convection in Differentially Heated Cavity (de Vahl Davis 1983)',
             fontsize=14, y=1.02)

out = '/home/yzk/LBMProject/scripts/viz/natural_convection.png'
plt.savefig(out)
print(f'Saved: {out}')
plt.close()
