#!/usr/bin/env python3
"""
3D Gaussian Heat Diffusion Visualization
==========================================

Shows the analytical solution for 2D heat diffusion from a point source:

    T(r, t) = Q / (4 * pi * alpha * t) * exp(-r^2 / (4 * alpha * t))

Displayed as three side-by-side 2D color maps at successive times,
illustrating how the initial concentrated heat spreads and decays.

This corresponds to the Gaussian diffusion benchmark used in the
ThermalLBM validation tests (test_3d_gaussian_diffusion).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------
alpha = 1.0e-5       # Thermal diffusivity [m^2/s] (typical metal, ~order of Ti6Al4V)
Q = 1.0e-3           # Heat source strength [K*m^2] (arbitrary, for visualization)

# Domain
L = 0.01             # Domain half-width [m]
N = 400              # Grid points per direction

# Time snapshots
t1 = 0.005           # Early: tight Gaussian
t2 = 0.05            # Intermediate: spreading
t3 = 0.25            # Late: nearly uniform

times = [t1, t2, t3]
time_labels = [
    f't = {t1*1000:.0f} ms',
    f't = {t2*1000:.0f} ms',
    f't = {t3*1000:.0f} ms'
]

# ---------------------------------------------------------------------------
# Compute temperature fields
# ---------------------------------------------------------------------------
x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
X, Y = np.meshgrid(x, y)
R2 = X**2 + Y**2

fields = []
for t in times:
    T = (Q / (4.0 * np.pi * alpha * t)) * np.exp(-R2 / (4.0 * alpha * t))
    fields.append(T)

# Global max for consistent color scale
T_max = max(f.max() for f in fields)

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=200)

# Colormap
cmap = 'inferno'

for i, (ax, T, label) in enumerate(zip(axes, fields, time_labels)):
    # Use power normalization to reveal structure at low temperatures
    # while keeping the hot core visible
    im = ax.pcolormesh(
        X * 1000, Y * 1000, T,
        cmap=cmap,
        vmin=0,
        vmax=T_max,
        shading='gouraud',
        rasterized=True
    )

    # Contour lines for isotherms
    levels = np.array([0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]) * T_max
    valid_levels = levels[levels < T.max()]
    if len(valid_levels) > 1:
        cs = ax.contour(X * 1000, Y * 1000, T, levels=valid_levels,
                        colors='white', linewidths=0.4, alpha=0.4)

    ax.set_xlabel('x [mm]', fontsize=10)
    if i == 0:
        ax.set_ylabel('y [mm]', fontsize=10)
    ax.set_title(label, fontsize=12, fontweight='bold', color='#2d2d44')
    ax.set_aspect('equal')
    ax.tick_params(labelsize=9)

    # Characteristic diffusion radius circle: r = 2*sqrt(alpha*t)
    r_diff = 2.0 * np.sqrt(alpha * times[i]) * 1000  # to mm
    if r_diff < L * 1000 * 0.9:
        circle = plt.Circle((0, 0), r_diff, fill=False,
                             edgecolor='cyan', linewidth=1.0,
                             linestyle='--', alpha=0.7)
        ax.add_patch(circle)
        # Label the diffusion radius
        angle = np.pi / 4
        ax.annotate(
            f'$2\\sqrt{{\\alpha t}}$ = {r_diff:.1f} mm',
            xy=(r_diff * np.cos(angle), r_diff * np.sin(angle)),
            xytext=(r_diff * np.cos(angle) + 1.5,
                    r_diff * np.sin(angle) + 1.5),
            fontsize=7, color='cyan', alpha=0.9,
            arrowprops=dict(arrowstyle='->', color='cyan', alpha=0.6, lw=0.8)
        )

    # Peak temperature annotation
    T_peak = T.max()
    if T_peak > 0.01 * T_max:
        ax.annotate(
            f'$T_{{max}}$ = {T_peak:.1f} K/m$^2$',
            xy=(0, 0), xytext=(4, -7),
            fontsize=7, color='white', alpha=0.8,
            arrowprops=dict(arrowstyle='->', color='white', alpha=0.5, lw=0.7)
        )

# Shared colorbar
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.935, 0.15, 0.015, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Temperature [a.u.]', fontsize=10)
cbar.ax.tick_params(labelsize=8)

# Supertitle
fig.suptitle(
    'Gaussian Heat Diffusion:  $T(r,t) = \\frac{Q}{4\\pi\\alpha t}'
    '\\, \\exp\\!\\left(-\\frac{r^2}{4\\alpha t}\\right)$'
    f'     ($\\alpha$ = {alpha:.0e} m$^2$/s)',
    fontsize=12, fontweight='bold', y=1.02, color='#2d2d44'
)

plt.savefig('/home/yzk/LBMProject/scripts/viz/gaussian_diffusion.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Saved: /home/yzk/LBMProject/scripts/viz/gaussian_diffusion.png")
print(f"  Thermal diffusivity: {alpha:.1e} m^2/s")
print(f"  Time snapshots: {[f'{t*1000:.0f} ms' for t in times]}")
print(f"  Peak temperatures: {[f'{f.max():.2f}' for f in fields]}")
print(f"  Diffusion radii: {[f'{2*np.sqrt(alpha*t)*1000:.2f} mm' for t in times]}")
