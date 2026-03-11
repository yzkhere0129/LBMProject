#!/usr/bin/env python3
"""
Powder Bed Particle Distribution Visualization
================================================

Generates a 2D cross-section schematic of an LPBF powder bed, matching
the parameters from the project's PowderBed module:

  - Log-normal particle size distribution (D50=30um, sigma_g=1.4, D_min=15um, D_max=45um)
  - Random Sequential Addition (RSA) packing with collision detection
  - Periodic boundary conditions in x, non-periodic in z
  - Substrate below the powder layer

Reference: PowderBedConfig in include/physics/powder_bed.h
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Powder bed parameters (matching PowderBedConfig defaults)
# ---------------------------------------------------------------------------
D50 = 30.0e-6        # Median diameter [m]
sigma_g = 1.4         # Geometric standard deviation
D_min = 10.0e-6       # Minimum diameter [m]
D_max = 50.0e-6       # Maximum diameter [m]
layer_thickness = 100.0e-6  # Powder layer thickness [m] (show ~3 particle layers)
substrate_height = 20.0e-6  # Substrate thickness shown [m]
target_packing = 0.58        # Target area packing fraction (2D proxy)

# Domain (2D cross-section view)
Lx = 600.0e-6         # Width [m] (wide enough for ~80-100 particles)
Lz = layer_thickness  # Height of powder region [m]

seed = 42
max_attempts_per_particle = 8000
max_particles = 200
min_gap = 0.2e-6      # Small gap between particles [m]

# ---------------------------------------------------------------------------
# Sample from log-normal distribution
# ---------------------------------------------------------------------------
rng = np.random.RandomState(seed)


def sample_diameter():
    """Sample a single diameter from the clamped log-normal distribution."""
    ln_sigma = np.log(sigma_g)
    ln_D = np.log(D50) + ln_sigma * rng.randn()
    D = np.exp(ln_D)
    return np.clip(D, D_min, D_max)


# ---------------------------------------------------------------------------
# Random Sequential Addition (2D circles, periodic in x)
# ---------------------------------------------------------------------------
particles = []  # list of (x, z, r)


def check_collision(cx, cz, cr):
    """Check if a candidate circle collides with any existing particle.
    Uses minimum-image convention in x (periodic), no periodicity in z."""
    for (px, pz, pr) in particles:
        dx = cx - px
        # Periodic wrap in x
        dx -= Lx * round(dx / Lx)
        dz = cz - pz
        dist = np.sqrt(dx**2 + dz**2)
        if dist < cr + pr + min_gap:
            return True
    return False


total_area = Lx * Lz
target_area = total_area * target_packing
current_area = 0.0

n_placed = 0
n_failed = 0

# Two-pass strategy: first place normally, then do a gap-filling pass with
# preferentially smaller particles to increase count.
for pass_num in range(2):
    consecutive_failures = 0
    max_consecutive = 30 if pass_num == 0 else 50

    while current_area < target_area and n_placed < max_particles:
        if pass_num == 0:
            D = sample_diameter()
        else:
            # Second pass: bias toward smaller particles for gap-filling
            D = sample_diameter()
            D = min(D, D50 * 0.8)  # Cap at 80% of median
            D = max(D, D_min)

        R = D / 2.0

        z_range = Lz - 2.0 * R
        if z_range <= 0:
            continue

        placed = False
        attempts = max_attempts_per_particle if pass_num == 0 else max_attempts_per_particle * 2
        for _ in range(attempts):
            cx = rng.uniform(0, Lx)
            cz = substrate_height + R + rng.uniform(0, z_range)

            if not check_collision(cx, cz, R):
                particles.append((cx, cz, R))
                current_area += np.pi * R**2
                n_placed += 1
                placed = True
                consecutive_failures = 0
                break

        if not placed:
            consecutive_failures += 1
            if consecutive_failures > max_consecutive:
                break

actual_packing = current_area / total_area

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5), dpi=200)

# Convert to micrometers for display
um = 1e6

# Background: argon gas (very dark blue-gray)
ax.set_facecolor('#1a1a2e')

# Substrate: solid metal below the powder layer
substrate_rect = patches.FancyBboxPatch(
    (0, 0), Lx * um, substrate_height * um,
    boxstyle="square,pad=0", linewidth=0,
    facecolor='#3d3d5c', edgecolor='none'
)
ax.add_patch(substrate_rect)

# Subtle substrate hatch for texture
for hx in np.linspace(0, Lx * um, 60):
    ax.plot([hx, hx], [0, substrate_height * um],
            color='#4a4a6a', linewidth=0.3, alpha=0.5)

# Metallic colormap for particles (light to dark steel)
metal_cmap = LinearSegmentedColormap.from_list(
    'metal', ['#a8a8b8', '#c0c0cc', '#d8d8e0', '#b0b0c0', '#909098'], N=256
)

# Draw particles as circles with metallic shading
for (cx, cz, cr) in particles:
    # Base circle
    circle = plt.Circle(
        (cx * um, cz * um), cr * um,
        facecolor=metal_cmap(rng.uniform(0.2, 0.8)),
        edgecolor='#606068',
        linewidth=0.4,
        alpha=0.95
    )
    ax.add_patch(circle)

    # Specular highlight (small offset circle for 3D effect)
    highlight = plt.Circle(
        ((cx - cr * 0.25) * um, (cz + cr * 0.25) * um),
        cr * 0.35 * um,
        facecolor='white',
        edgecolor='none',
        alpha=0.15
    )
    ax.add_patch(highlight)

# Domain boundary
ax.plot([0, Lx * um], [substrate_height * um, substrate_height * um],
        color='#8080a0', linewidth=1.5, linestyle='-', label='Substrate surface')

# Powder layer top indicator
top_z = (substrate_height + layer_thickness) * um
ax.axhline(y=top_z, color='#ff6b6b', linewidth=1.0, linestyle='--',
           alpha=0.7, label=f'Layer thickness = {layer_thickness*um:.0f} $\\mu$m')

# Axis settings
ax.set_xlim(-5, Lx * um + 5)
ax.set_ylim(-2, (substrate_height + layer_thickness + 10e-6) * um)
ax.set_aspect('equal')
ax.set_xlabel('x [$\\mu$m]', fontsize=12)
ax.set_ylabel('z [$\\mu$m]', fontsize=12)
ax.tick_params(labelsize=10)

# Title with statistics
ax.set_title(
    f'LPBF Powder Bed Cross-Section  |  '
    f'{n_placed} particles, D$_{{50}}$={D50*um:.0f} $\\mu$m, '
    f'$\\sigma_g$={sigma_g:.1f}, '
    f'packing={actual_packing:.0%}',
    fontsize=12, fontweight='bold', color='#2d2d44'
)

# Legend
ax.legend(loc='upper right', fontsize=9, framealpha=0.85)

# Annotations
ax.annotate('Substrate (solid)', xy=(Lx * um * 0.5, substrate_height * um * 0.45),
            fontsize=9, color='#b0b0c8', ha='center', va='center',
            fontweight='bold')
ax.annotate('Ar gas', xy=(Lx * um * 0.88, (substrate_height + layer_thickness * 0.85) * um),
            fontsize=8, color='#6a6a8a', ha='center', style='italic')

# Size distribution inset
inset_ax = fig.add_axes([0.13, 0.62, 0.18, 0.25])
diameters_um = [2 * r * um for (_, _, r) in particles]
inset_ax.hist(diameters_um, bins=15, color='#909098', edgecolor='#606068',
              linewidth=0.5, alpha=0.9, density=True)

# Overlay theoretical log-normal PDF
d_range = np.linspace(D_min * um, D_max * um, 200)
ln_sigma = np.log(sigma_g)
pdf = (1.0 / (d_range * ln_sigma * np.sqrt(2 * np.pi))) * \
      np.exp(-(np.log(d_range) - np.log(D50 * um))**2 / (2 * ln_sigma**2))
inset_ax.plot(d_range, pdf, color='#ff6b6b', linewidth=1.5, label='Log-normal PDF')

inset_ax.set_xlabel('D [$\\mu$m]', fontsize=7)
inset_ax.set_ylabel('Density', fontsize=7)
inset_ax.set_title('Size distribution', fontsize=8, fontweight='bold')
inset_ax.tick_params(labelsize=6)
inset_ax.legend(fontsize=6)
inset_ax.set_facecolor('#f0f0f4')

plt.tight_layout()
plt.savefig('/home/yzk/LBMProject/scripts/viz/powder_bed.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Saved: /home/yzk/LBMProject/scripts/viz/powder_bed.png")
print(f"  Particles placed: {n_placed}")
print(f"  Actual 2D packing: {actual_packing:.1%}")
print(f"  Diameter range: {min(diameters_um):.1f} - {max(diameters_um):.1f} um")
print(f"  Mean diameter: {np.mean(diameters_um):.1f} um")
