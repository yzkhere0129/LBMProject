#!/usr/bin/env python3
"""Compare C++ stamped mask vs Python expected mask for α=+8."""
import numpy as np, matplotlib.pyplot as plt

# Read C++ dump
with open('output_naca_a8_dbgmask/mask_zmid.txt') as f:
    lines = f.readlines()
hdr = lines[0]
parts = hdr.replace('#', '').split()
nx = int([p for p in parts if p.startswith('nx=')][0].split('=')[1])
ny = int([p for p in parts if p.startswith('ny=')][0].split('=')[1])
dx = float([p for p in parts if p.startswith('dx=')][0].split('=')[1])

mask = np.zeros((ny, nx), dtype=np.uint8)
for j, ln in enumerate(lines[1:1+ny]):
    mask[j] = np.array(ln.split(), dtype=np.uint8)

# Find solid cell extents
js, is_ = np.where(mask == 1)
print(f"C++ stamped mask: {len(js)} solid cells in z-mid slice")
print(f"  i range: {is_.min()}..{is_.max()}  (world x: {is_.min()*dx:.3f}..{is_.max()*dx:.3f})")
print(f"  j range: {js.min()}..{js.max()}  (world y: {js.min()*dx:.3f}..{js.max()*dx:.3f})")

# Find LE (smallest i) and TE (largest i)
i_LE = is_.min()
js_LE = js[is_ == i_LE]
print(f"  At i_LE={i_LE} (x={i_LE*dx:.3f}): j range {js_LE.min()}..{js_LE.max()} (y={js_LE.min()*dx:.3f}..{js_LE.max()*dx:.3f})")
i_TE = is_.max()
js_TE = js[is_ == i_TE]
print(f"  At i_TE={i_TE} (x={i_TE*dx:.3f}): j range {js_TE.min()}..{js_TE.max()} (y={js_TE.min()*dx:.3f}..{js_TE.max()*dx:.3f})")

# For α=+8°, expected: TE world y ≈ 10.139, LE world y ≈ 10.0
# So TE cells should be at higher y than LE cells
y_LE_avg = (js_LE.mean() + 0.5) * dx
y_TE_avg = (js_TE.mean() + 0.5) * dx
print(f"\n  y_LE_avg = {y_LE_avg:.3f} world y")
print(f"  y_TE_avg = {y_TE_avg:.3f} world y")
print(f"  Δy = y_TE - y_LE = {y_TE_avg - y_LE_avg:+.3f}")
print(f"  → For α=+8°, expected Δy ≈ +0.139 (TE above LE)")
print(f"  → If Δy < 0, geometry is FLIPPED (TE below LE)")

# Plot the actual C++ mask
i_z0, i_z1 = int(9.5/dx), int(11.5/dx)
j_z0, j_z1 = int(9.4/dx), int(10.6/dx)
fig, ax = plt.subplots(figsize=(12, 5))
ax.imshow(mask[j_z0:j_z1, i_z0:i_z1], origin='lower',
          extent=[i_z0*dx, i_z1*dx, j_z0*dx, j_z1*dx], cmap='gray_r')
ax.plot([10, 10.99], [10, 10.139], 'r-', lw=2, label='expected chord (α=+8°)')
ax.plot(10, 10, 'go', markersize=10, label='LE')
ax.plot(10.99, 10.139, 'ms', markersize=10, label='TE')
ax.set_title(f'C++ stamped mask (z-mid slice), α=+8°\n'
             f'Black = solid. Red line = expected chord (TE above LE for +AOA)')
ax.legend()
ax.grid(alpha=0.3)
fig.savefig('output_naca_a8_dbgmask/cppmask_zoom.png', dpi=150, bbox_inches='tight')
print("Saved output_naca_a8_dbgmask/cppmask_zoom.png")
