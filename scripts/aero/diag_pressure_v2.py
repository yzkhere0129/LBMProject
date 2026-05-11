#!/usr/bin/env python3
"""Read rho dump from driver, check pressure asymmetry around airfoil."""
import sys, numpy as np, matplotlib.pyplot as plt

rho_file = sys.argv[1] if len(sys.argv) > 1 else 'output_naca_a8_rho/rho_zmid_0016000.txt'

with open(rho_file) as f: lines = f.readlines()
hdr = lines[0]
parts = hdr.replace('#','').split()
nx = int([p.split('=')[1] for p in parts if p.startswith('nx=')][0])
ny = int([p.split('=')[1] for p in parts if p.startswith('ny=')][0])
dx = float([p.split('=')[1] for p in parts if p.startswith('dx=')][0])

rho = np.zeros((ny, nx), dtype=np.float32)
for j, ln in enumerate(lines[1:1+ny]):
    rho[j] = np.array(ln.split(), dtype=np.float32)

p = (rho - 1.0) / 3.0  # gauge pressure

# Sample pressure at points around airfoil (LE=10,10; TE=10.99,10.139 for α=+8°)
print(f"Pressure (gauge, ρ-1)/3 around airfoil chord at α=+8°")
print(f"  LE world (10, 10), TE world (10.99, 10.14)")
print()
print("  Mid-chord normal probes (perpendicular to chord, ±0.10 c each side):")
# Chord midpoint
cmid_x = 10.0 + 0.5 * np.cos(np.radians(8))
cmid_y = 10.0 + 0.5 * np.sin(np.radians(8))
# Perpendicular direction: chord-normal "upper" = (-sin α, cos α)
nx_dir, ny_dir = -np.sin(np.radians(8)), np.cos(np.radians(8))
for offset in [0.20, 0.15, 0.10, -0.10, -0.15, -0.20]:
    px = cmid_x + offset * nx_dir
    py = cmid_y + offset * ny_dir
    i, j = int(px/dx), int(py/dx)
    side = "UPPER" if offset > 0 else "LOWER"
    print(f"   offset {offset:+.2f}c (side={side}): world=({px:.3f},{py:.3f}), p={p[j,i]:+.6f}")

# Mean pressure on upper vs lower
upper_p = []
lower_p = []
for s in np.linspace(0.05, 0.95, 20):
    cx = 10 + s*np.cos(np.radians(8))
    cy = 10 + s*np.sin(np.radians(8))
    # Probe 0.15c off chord (well outside airfoil)
    for off in [0.10, 0.15, 0.20]:
        for sign, lst in [(+1, upper_p), (-1, lower_p)]:
            px = cx + sign*off*nx_dir
            py = cy + sign*off*ny_dir
            i, j = int(px/dx), int(py/dx)
            lst.append(p[j, i])

print()
print(f"Mean p on UPPER side (chord-relative +η, off=0.10..0.20c): {np.mean(upper_p):+.6f}")
print(f"Mean p on LOWER side (chord-relative -η, off=0.10..0.20c): {np.mean(lower_p):+.6f}")
print(f"Δp = p_lower - p_upper = {np.mean(lower_p) - np.mean(upper_p):+.6f}")
print(f"  + Δp > 0 → higher p below → LIFT UP → Cl > 0 (correct for α>0)")
print(f"  + Δp < 0 → higher p above → LIFT DOWN → Cl < 0 (WRONG for α>0)")

# Visualize pressure
fig, ax = plt.subplots(figsize=(12, 5))
i_z0, i_z1 = int(9.5/dx), int(11.5/dx)
j_z0, j_z1 = int(9.4/dx), int(10.6/dx)
extent = [i_z0*dx, i_z1*dx, j_z0*dx, j_z1*dx]
vmax = max(abs(p[j_z0:j_z1, i_z0:i_z1].max()), abs(p[j_z0:j_z1, i_z0:i_z1].min()))
im = ax.imshow(p[j_z0:j_z1, i_z0:i_z1], origin='lower', extent=extent,
               cmap='RdBu_r', vmin=-vmax, vmax=vmax)
plt.colorbar(im, ax=ax, label='gauge pressure (ρ-1)/3')
ax.plot([10, 10.99], [10, 10.139], 'k-', lw=2)
ax.plot(10, 10, 'go', markersize=10)
ax.plot(10.99, 10.139, 'ms', markersize=10)
ax.set_title(f'Pressure field (zoom). α=+8°. Red=high (= stagnation), Blue=low (= suction)\n'
             f'For Cl>0: high p BELOW chord, low p ABOVE chord')
ax.set_xlabel('x'); ax.set_ylabel('y')
out = rho_file.replace('.txt', '_pressure.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nSaved {out}")
