"""
3D Gaussian Heat Diffusion Visualization - Publication-quality figure
Generates a 2x2 panel figure showing:
  (a) Initial Gaussian temperature (z-midplane contourf)
  (b) Final diffused temperature (z-midplane contourf)
  (c) Analytical solution for comparison
  (d) Centerline temperature profile (LBM vs analytical)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Style setup
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# ============================================================================
# Load data
# ============================================================================
temp_init = np.genfromtxt('temp_initial.csv', delimiter=',', skip_header=1)
temp_final = np.genfromtxt('temp_final.csv', delimiter=',', skip_header=1)
temp_anal = np.genfromtxt('temp_analytical.csv', delimiter=',', skip_header=1)

# Read x-axis coordinates from header
with open('temp_initial.csv', 'r') as f:
    x_coords = np.array([float(v) for v in f.readline().strip().split(',')])

dx_um = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 20.0
n = temp_init.shape[0]
y_coords = np.arange(n) * dx_um

print(f"Grid: {n}x{n}, dx = {dx_um:.1f} um")
print(f"T_initial range: [{temp_init.min():.1f}, {temp_init.max():.1f}] K")
print(f"T_final   range: [{temp_final.min():.1f}, {temp_final.max():.1f}] K")
print(f"T_anal    range: [{temp_anal.min():.1f}, {temp_anal.max():.1f}] K")

# ============================================================================
# Color scale: shared range for consistent comparison
# ============================================================================
T_min = 300.0
T_max_init = temp_init.max()
T_max_final = max(temp_final.max(), temp_anal.max())

# Custom colormap: cool blues to hot reds
cmap_thermal = plt.cm.inferno

# ============================================================================
# Create figure
# ============================================================================
fig = plt.figure(figsize=(13, 11))

# --------------------------------------------------------------------------
# (a) Initial Gaussian
# --------------------------------------------------------------------------
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_title('(a) Initial Temperature (t = 0)', fontweight='bold')

X, Y = np.meshgrid(x_coords, y_coords)
levels_init = np.linspace(T_min, T_max_init, 40)
cf1 = ax1.contourf(X, Y, temp_init, levels=levels_init, cmap=cmap_thermal)
cbar1 = plt.colorbar(cf1, ax=ax1, label='Temperature [K]', shrink=0.85)

# Add contour lines
cs1 = ax1.contour(X, Y, temp_init,
                   levels=[400, 600, 900, 1200, 1500, 1700],
                   colors='white', linewidths=0.5, alpha=0.6)
ax1.clabel(cs1, inline=True, fontsize=7, fmt='%.0f K')

ax1.set_xlabel('X [um]')
ax1.set_ylabel('Y [um]')
ax1.set_aspect('equal')

# Annotation
ax1.text(0.03, 0.97, f'T_peak = {temp_init.max():.0f} K',
         transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', color='white', fontweight='bold',
         bbox=dict(facecolor='black', alpha=0.4, pad=2))

# --------------------------------------------------------------------------
# (b) Final LBM solution
# --------------------------------------------------------------------------
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title('(b) LBM Solution (t = 2000 us)', fontweight='bold')

levels_final = np.linspace(T_min, T_max_final + 20, 40)
cf2 = ax2.contourf(X, Y, temp_final, levels=levels_final, cmap=cmap_thermal)
cbar2 = plt.colorbar(cf2, ax=ax2, label='Temperature [K]', shrink=0.85)

cs2 = ax2.contour(X, Y, temp_final,
                   levels=[320, 350, 400, 450, 500],
                   colors='white', linewidths=0.5, alpha=0.6)
ax2.clabel(cs2, inline=True, fontsize=7, fmt='%.0f K')

ax2.set_xlabel('X [um]')
ax2.set_ylabel('Y [um]')
ax2.set_aspect('equal')

ax2.text(0.03, 0.97, f'T_peak = {temp_final.max():.1f} K',
         transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', color='white', fontweight='bold',
         bbox=dict(facecolor='black', alpha=0.4, pad=2))

# --------------------------------------------------------------------------
# (c) Error field (LBM - Analytical)
# --------------------------------------------------------------------------
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_title('(c) Error: LBM - Analytical', fontweight='bold')

error_field = temp_final - temp_anal
err_max = max(abs(error_field.min()), abs(error_field.max()))
err_max = max(err_max, 0.1)  # prevent zero range

levels_err = np.linspace(-err_max, err_max, 40)
cf3 = ax3.contourf(X, Y, error_field, levels=levels_err,
                     cmap='RdBu_r')
cbar3 = plt.colorbar(cf3, ax=ax3, label='Temperature Error [K]', shrink=0.85)

ax3.set_xlabel('X [um]')
ax3.set_ylabel('Y [um]')
ax3.set_aspect('equal')

# Statistics
rms_err = np.sqrt(np.mean(error_field**2))
max_err = np.max(np.abs(error_field))
rel_err_peak = abs(temp_final.max() - temp_anal.max()) / (temp_anal.max() - 300.0) * 100

stats_text = (f'RMS = {rms_err:.3f} K\n'
              f'Max |err| = {max_err:.3f} K\n'
              f'Peak rel. err = {rel_err_peak:.2f}%')
ax3.text(0.03, 0.97, stats_text, transform=ax3.transAxes,
         fontsize=8, verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

# --------------------------------------------------------------------------
# (d) Centerline profiles
# --------------------------------------------------------------------------
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_title('(d) Centerline Temperature Profile', fontweight='bold')

mid = n // 2

# Initial (along x, at y=mid)
ax4.plot(x_coords, temp_init[mid, :], '-', color='crimson', linewidth=2.0,
         label=f't = 0 (initial)', alpha=0.8)

# Final LBM
ax4.plot(x_coords, temp_final[mid, :], '-', color='royalblue', linewidth=2.0,
         label='t = 2000 us (LBM)')

# Analytical
ax4.plot(x_coords, temp_anal[mid, :], '--', color='forestgreen', linewidth=1.5,
         label='t = 2000 us (analytical)', alpha=0.9)

ax4.set_xlabel('X [um]')
ax4.set_ylabel('Temperature [K]')
ax4.legend(fontsize=9, loc='upper right')
ax4.grid(True, alpha=0.3, linewidth=0.5)
ax4.set_xlim(x_coords[0], x_coords[-1])
ax4.set_ylim(T_min - 10, T_max_init + 50)

# Inset zoom near peak
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
ax_ins = inset_axes(ax4, width="40%", height="35%", loc='center right',
                    bbox_to_anchor=(-0.05, 0.02, 1, 1),
                    bbox_transform=ax4.transAxes)

# Zoom into the peak region of the final solution
peak_region = max(5, int(n * 0.15))
x_lo = max(0, mid - peak_region)
x_hi = min(n, mid + peak_region)

ax_ins.plot(x_coords[x_lo:x_hi], temp_final[mid, x_lo:x_hi], '-',
            color='royalblue', linewidth=1.5)
ax_ins.plot(x_coords[x_lo:x_hi], temp_anal[mid, x_lo:x_hi], '--',
            color='forestgreen', linewidth=1.2)
ax_ins.set_title('Peak region zoom', fontsize=7, pad=2)
ax_ins.tick_params(labelsize=6)
ax_ins.grid(True, alpha=0.2)

# ============================================================================
# Add physics annotation
# ============================================================================
physics_text = (
    r'$T(r,t) = T_0 + A\left(\frac{\sigma_0}{\sigma(t)}\right)^3 '
    r'\exp\!\left(-\frac{r^2}{2\sigma(t)^2}\right)$'
    '\n'
    r'$\sigma(t) = \sqrt{\sigma_0^2 + 2\alpha t}$'
    '\n\n'
    r'Material: Ti6Al4V-like ($\alpha$ = 2.88$\times$10$^{-6}$ m$^2$/s)'
    '\n'
    'D3Q7 thermal LBM, BGK collision'
)

fig.text(0.5, 0.005, physics_text, ha='center', fontsize=9,
         style='italic', color='dimgray',
         bbox=dict(facecolor='whitesmoke', alpha=0.8, boxstyle='round,pad=0.5'))

# ============================================================================
# Final layout and save
# ============================================================================
fig.suptitle('3D Gaussian Heat Diffusion: LBM Validation',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.08, 1, 0.95])
plt.savefig('gaussian_diffusion.png')
print("Saved: gaussian_diffusion.png")
plt.close()
