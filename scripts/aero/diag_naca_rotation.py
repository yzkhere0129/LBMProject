#!/usr/bin/env python3
"""Replicate stampNacaAirfoil4Digit math in Python, visualize for α=+8."""
import numpy as np
import matplotlib.pyplot as plt
import sys, math

dx = 0.025
chord = 1.0
thick = 0.12
alpha_deg = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
alpha = math.radians(alpha_deg)

# Just the airfoil region: 30 cells before LE, 80 after, 50 above/below
xLE, yLE = 10.0, 10.0
i0, i1 = int((xLE - 0.4) / dx), int((xLE + 1.4) / dx)
j0, j1 = int((yLE - 0.5) / dx), int((yLE + 0.5) / dx)
nx, ny = i1 - i0, j1 - j0
mask = np.zeros((ny, nx), dtype=np.uint8)

cos_a, sin_a = math.cos(alpha), math.sin(alpha)

def naca_y_t(s):
    if s < 0 or s > 1: return -1.0
    return 5.0 * thick * (
        0.2969*math.sqrt(s) - 0.1260*s
        - 0.3516*s*s + 0.2843*s**3 - 0.1015*s**4
    )

for jj in range(ny):
    j = j0 + jj
    y = (j + 0.5) * dx
    for ii in range(nx):
        i = i0 + ii
        x = (i + 0.5) * dx
        # EXACTLY the formula in obstacle_geometry.h:88-126
        xr =  cos_a * (x - xLE) + sin_a * (y - yLE)
        yr = -sin_a * (x - xLE) + cos_a * (y - yLE)
        s = xr / chord
        y_t = naca_y_t(s) * chord
        if 0.0 <= s <= 1.0 and abs(yr) <= y_t:
            mask[jj, ii] = 1

# Draw expected orientation: TE position
TE_world_x = xLE + cos_a * chord
TE_world_y = yLE + sin_a * chord
print(f"α={alpha_deg}°")
print(f"LE world=({xLE:.3f}, {yLE:.3f})")
print(f"TE world=({TE_world_x:.3f}, {TE_world_y:.3f}) — Δy = {TE_world_y - yLE:+.3f}")
print(f"For α=+8°, TE should be UP relative to LE (Δy > 0) → standard +AOA convention")

extent = [(i0)*dx, (i1)*dx, (j0)*dx, (j1)*dx]
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(mask, origin='lower', extent=extent, cmap='gray_r', aspect='equal')
ax.plot([xLE, TE_world_x], [yLE, TE_world_y], 'r-', lw=2, label=f'expected chord (LE→TE)')
ax.plot([xLE], [yLE], 'go', markersize=10, label='LE')
ax.plot([TE_world_x], [TE_world_y], 'ms', markersize=10, label='TE')
ax.axhline(yLE, color='b', linestyle=':', alpha=0.5, label='y_LE horizontal')
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
ax.set_title(f'NACA0012 mask, α={alpha_deg}°. Black = solid cells from stamp function.\n'
             f'For α>0, the BLACK region should be tilted UP-RIGHT (TE above LE)')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)
fig.savefig(f'naca_mask_alpha{int(alpha_deg)}.png', dpi=150, bbox_inches='tight')
print(f"Saved naca_mask_alpha{int(alpha_deg)}.png")
