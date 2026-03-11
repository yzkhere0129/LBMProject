"""
Powder Bed Visualization - Publication-quality figure
Generates a 2x2 panel figure showing:
  (a) 3D particle scatter (top-down view with size encoding)
  (b) 3D particle scatter (perspective view)
  (c) VOF fill_level z-midplane slice
  (d) Particle size distribution histogram
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors

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
particles = np.genfromtxt('particles.csv', delimiter=',', names=True)
fill_mid = np.genfromtxt('fill_level_zmid.csv', delimiter=',', skip_header=1)
fill_low = np.genfromtxt('fill_level_zlow.csv', delimiter=',', skip_header=1)

# Read x-axis coordinates from header
with open('fill_level_zmid.csv', 'r') as f:
    x_coords = np.array([float(v) for v in f.readline().strip().split(',')])

dx_um = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 2.0
ny_fill = fill_mid.shape[0]
y_coords = np.arange(ny_fill) * dx_um

print(f"Loaded {len(particles)} particles")
print(f"Fill level grid: {fill_mid.shape}")
print(f"Diameter range: {particles['radius_um'].min()*2:.1f} - {particles['radius_um'].max()*2:.1f} um")

# ============================================================================
# Create figure
# ============================================================================
fig = plt.figure(figsize=(12, 10))

# Color map for particles by diameter
diameters = particles['radius_um'] * 2.0
norm = mcolors.Normalize(vmin=diameters.min() * 0.9, vmax=diameters.max() * 1.1)
cmap_part = plt.cm.RdYlBu_r

# --------------------------------------------------------------------------
# (a) Top-down view: circles showing actual particle sizes
# --------------------------------------------------------------------------
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_title('(a) Top-Down View (XY Projection)', fontweight='bold')

circles = []
colors_a = []
for i in range(len(particles)):
    c = Circle((particles['x_um'][i], particles['y_um'][i]),
               particles['radius_um'][i],
               linewidth=0.5)
    circles.append(c)
    colors_a.append(diameters[i])

pc = PatchCollection(circles, cmap=cmap_part, norm=norm, alpha=0.85,
                     edgecolors='k', linewidths=0.4)
pc.set_array(np.array(colors_a))
ax1.add_collection(pc)
cbar1 = plt.colorbar(pc, ax=ax1, label='Diameter [um]', shrink=0.85)

ax1.set_xlim(0, x_coords[-1])
ax1.set_ylim(0, y_coords[-1])
ax1.set_xlabel('X [um]')
ax1.set_ylabel('Y [um]')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.2, linewidth=0.5)

# --------------------------------------------------------------------------
# (b) Side view: XZ projection showing layer structure
# --------------------------------------------------------------------------
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title('(b) Side View (XZ Projection)', fontweight='bold')

circles2 = []
colors_b = []
for i in range(len(particles)):
    c = Circle((particles['x_um'][i], particles['z_um'][i]),
               particles['radius_um'][i],
               linewidth=0.5)
    circles2.append(c)
    colors_b.append(diameters[i])

pc2 = PatchCollection(circles2, cmap=cmap_part, norm=norm, alpha=0.85,
                      edgecolors='k', linewidths=0.4)
pc2.set_array(np.array(colors_b))
ax2.add_collection(pc2)
cbar2 = plt.colorbar(pc2, ax=ax2, label='Diameter [um]', shrink=0.85)

ax2.set_xlim(0, x_coords[-1])
ax2.set_ylim(0, max(particles['z_um']) + particles['radius_um'].max() + 5)
ax2.set_xlabel('X [um]')
ax2.set_ylabel('Z [um]')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.2, linewidth=0.5)

# Add substrate line
ax2.axhline(y=0, color='saddlebrown', linewidth=2, linestyle='-', label='Substrate')
ax2.legend(loc='upper right', fontsize=8)

# --------------------------------------------------------------------------
# (c) VOF fill_level z-midplane
# --------------------------------------------------------------------------
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_title('(c) VOF Fill Level (Z-Midplane)', fontweight='bold')

# Custom colormap: gas=white, interface=gradient, metal=dark blue
cmap_vof = plt.cm.YlOrRd
im3 = ax3.pcolormesh(x_coords, y_coords, fill_mid,
                      cmap=cmap_vof, vmin=0.0, vmax=1.0,
                      shading='auto')
cbar3 = plt.colorbar(im3, ax=ax3, label='Fill Level (0=gas, 1=metal)', shrink=0.85)
ax3.set_xlabel('X [um]')
ax3.set_ylabel('Y [um]')
ax3.set_aspect('equal')

# --------------------------------------------------------------------------
# (d) Particle size distribution
# --------------------------------------------------------------------------
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_title('(d) Particle Size Distribution', fontweight='bold')

n_bins = 12
counts, bin_edges, patches = ax4.hist(diameters, bins=n_bins,
                                       color='steelblue', edgecolor='white',
                                       alpha=0.85, linewidth=0.8)

# Color bars by diameter
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
for patch, bc in zip(patches, bin_centers):
    patch.set_facecolor(cmap_part(norm(bc)))

D50_target = 20.0  # matches viz_powder.cu config
ax4.axvline(x=D50_target, color='red', linestyle='--', linewidth=1.5,
            label=f'D50 = {D50_target:.0f} um')
ax4.axvline(x=np.median(diameters), color='navy', linestyle=':',
            linewidth=1.5,
            label=f'Median = {np.median(diameters):.1f} um')

ax4.set_xlabel('Particle Diameter [um]')
ax4.set_ylabel('Count')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.2, axis='y')

# Add statistics text
stats_text = (f'N = {len(diameters)}\n'
              f'D_mean = {diameters.mean():.1f} um\n'
              f'D_std = {diameters.std():.1f} um\n'
              f'Packing ~ 25%')
ax4.text(0.97, 0.95, stats_text, transform=ax4.transAxes,
         fontsize=8, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

# ============================================================================
# Final layout and save
# ============================================================================
fig.suptitle('LPBF Powder Bed Generation (Random Sequential Addition)',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('powder_bed.png')
print("Saved: powder_bed.png")
plt.close()
