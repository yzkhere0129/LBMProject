#!/usr/bin/env python3
"""
Lid-Driven Cavity Flow (Re=100) via D2Q9 BGK Lattice Boltzmann Method.

Simulates the classic benchmark on a 65x65 grid and plots streamlines
showing the primary vortex and corner eddies.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ---------------------------------------------------------------------------
# D2Q9 lattice constants
# ---------------------------------------------------------------------------
NX, NY = 65, 65
Q = 9

# Lattice velocities (east, north, west, south, NE, NW, SW, SE, rest)
ex = np.array([1, 0, -1,  0, 1, -1, -1,  1, 0])
ey = np.array([0, 1,  0, -1, 1,  1, -1, -1, 0])
w  = np.array([1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36, 4/9])
opp = np.array([2, 3, 0, 1, 6, 7, 4, 5, 8])  # opposite directions

# Physics
Re = 100.0
U_lid = 0.1          # lattice velocity of lid
nu = U_lid * NX / Re # kinematic viscosity (lattice units)
tau = 3.0 * nu + 0.5
omega = 1.0 / tau

print(f"Re = {Re}, U_lid = {U_lid}, nu = {nu:.6f}, tau = {tau:.4f}, omega = {omega:.4f}")

# ---------------------------------------------------------------------------
# Initialise
# ---------------------------------------------------------------------------
rho = np.ones((NX, NY))
ux  = np.zeros((NX, NY))
uy  = np.zeros((NX, NY))

def feq(rho, ux, uy):
    """Compute equilibrium distribution."""
    f = np.zeros((Q, NX, NY))
    usq = ux**2 + uy**2
    for i in range(Q):
        cu = ex[i]*ux + ey[i]*uy
        f[i] = w[i] * rho * (1.0 + 3.0*cu + 4.5*cu**2 - 1.5*usq)
    return f

f = feq(rho, ux, uy)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
N_STEPS = 20000

for step in range(1, N_STEPS + 1):
    # --- Collision (BGK) ---
    feq_val = feq(rho, ux, uy)
    f_out = f - omega * (f - feq_val)

    # --- Streaming ---
    f_new = np.zeros_like(f_out)
    for i in range(Q):
        f_new[i] = np.roll(np.roll(f_out[i], ex[i], axis=0), ey[i], axis=1)

    # --- Bounce-back boundary conditions ---
    # Bottom wall (y = 0): stationary
    for i in range(Q):
        if ey[i] > 0:  # directions pointing into fluid
            f_new[i, :, 0] = f_out[opp[i], :, 0]

    # Top wall (y = NY-1): moving lid with velocity U_lid in +x
    for i in range(Q):
        if ey[i] < 0:  # directions pointing into fluid (downward)
            # Zou-He / momentum-corrected bounce-back for moving wall
            cu_wall = 2.0 * w[i] * rho[:, -1] * 3.0 * (ex[i]*U_lid)
            f_new[i, :, -1] = f_out[opp[i], :, -1] - cu_wall

    # Left wall (x = 0): stationary
    for i in range(Q):
        if ex[i] > 0:
            f_new[i, 0, :] = f_out[opp[i], 0, :]

    # Right wall (x = NX-1): stationary
    for i in range(Q):
        if ex[i] < 0:
            f_new[i, -1, :] = f_out[opp[i], -1, :]

    f = f_new

    # --- Macroscopic quantities ---
    rho = np.sum(f, axis=0)
    ux  = np.sum(f * ex[:, None, None], axis=0) / rho
    uy  = np.sum(f * ey[:, None, None], axis=0) / rho

    # Enforce wall velocities explicitly
    ux[:, 0]  = 0.0; uy[:, 0]  = 0.0  # bottom
    ux[:, -1] = U_lid; uy[:, -1] = 0.0  # top (lid)
    ux[0, :]  = 0.0; uy[0, :]  = 0.0  # left
    ux[-1, :] = 0.0; uy[-1, :] = 0.0  # right

    if step % 5000 == 0:
        residual = np.sqrt(np.mean(ux**2 + uy**2))
        print(f"  step {step:6d}/{N_STEPS}  |u|_rms = {residual:.6e}")

print("Simulation complete.")

# ---------------------------------------------------------------------------
# Compute stream function via integration of velocity
# ---------------------------------------------------------------------------
# psi satisfies: dpsi/dy = ux, dpsi/dx = -uy
# Integrate ux along y for each x column
psi = np.zeros((NX, NY))
dx = 1.0
for i in range(NX):
    for j in range(1, NY):
        psi[i, j] = psi[i, j-1] + ux[i, j] * dx

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(7.5, 7), dpi=150)

x = np.linspace(0, 1, NX)
y = np.linspace(0, 1, NY)
X, Y = np.meshgrid(x, y, indexing='ij')

# Velocity magnitude
speed = np.sqrt(ux**2 + uy**2) / U_lid

# Filled contour of velocity magnitude
levels_speed = np.linspace(0, 1.0, 32)
cf = ax.contourf(X, Y, speed, levels=levels_speed, cmap='RdYlBu_r', extend='max')
cbar = fig.colorbar(cf, ax=ax, shrink=0.82, pad=0.03)
cbar.set_label(r'$|\mathbf{u}| \;/\; U_{\mathrm{lid}}$', fontsize=13)

# Stream function contours (streamlines)
psi_norm = psi / (U_lid * NX)
# Choose contour levels that reveal both primary vortex and corner eddies
psi_min, psi_max = psi_norm.min(), psi_norm.max()
# Primary vortex levels (negative values for clockwise rotation)
n_primary = 18
primary_levels = np.linspace(psi_min * 0.98, 0, n_primary)
# Small positive levels for bottom corner eddies
n_corner = 6
if psi_max > 1e-6:
    corner_levels = np.linspace(1e-6, psi_max * 0.9, n_corner)
else:
    corner_levels = np.array([])

all_levels = np.sort(np.concatenate([primary_levels, corner_levels]))

cs = ax.contour(X, Y, psi_norm, levels=all_levels, colors='k',
                linewidths=0.7, linestyles='solid')

# Lid arrow indicating motion direction
arrow = FancyArrowPatch((0.15, 1.02), (0.85, 1.02),
                        arrowstyle='->', mutation_scale=18,
                        color='#CC0000', linewidth=2.5,
                        clip_on=False)
ax.add_patch(arrow)
ax.text(0.5, 1.055, r'$U_{\mathrm{lid}}$', transform=ax.transAxes,
        ha='center', va='bottom', fontsize=14, color='#CC0000', fontweight='bold')

# Formatting
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.set_xlabel('x / L', fontsize=13)
ax.set_ylabel('y / L', fontsize=13)
ax.set_title(f'Lid-Driven Cavity Flow  (Re = {int(Re)}, {NX}x{NY} LBM)',
             fontsize=14, fontweight='bold', pad=20)
ax.tick_params(labelsize=11)

# Wall labels
ax.text(0.5, -0.04, 'stationary wall', ha='center', va='top',
        fontsize=10, color='#555555', transform=ax.transAxes)
ax.text(-0.06, 0.5, 'wall', ha='right', va='center', fontsize=10,
        color='#555555', rotation=90, transform=ax.transAxes)
ax.text(1.06, 0.5, 'wall', ha='left', va='center', fontsize=10,
        color='#555555', rotation=-90, transform=ax.transAxes)

plt.tight_layout()
out_path = '/home/yzk/LBMProject/scripts/viz/lid_driven_cavity.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_path}")
plt.close()
