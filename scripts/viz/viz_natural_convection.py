#!/usr/bin/env python3
"""
Natural Convection in a Differentially Heated Square Cavity (Ra=1000).

Uses vorticity-stream function formulation with explicit time stepping.
The Poisson equation for psi is solved with Jacobi iteration (vectorized).

Governing equations (dimensionless):
  d(omega)/dt + u*dw/dx + v*dw/dy = Pr * nabla^2(omega) + Ra*Pr * dT/dx
  dT/dt + u*dT/dx + v*dT/dy = nabla^2(T)
  nabla^2(psi) = -omega
  u = dpsi/dy, v = -dpsi/dx

BCs: T=1 left, T=0 right, adiabatic top/bottom. No-slip walls.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm

# --- Parameters ---
Ra = 1000.0
Pr = 0.71
N = 41        # grid points (coarser for speed, fine enough for Ra=1000)
dx = 1.0 / (N - 1)
max_iter = 60000
tol = 1e-7
dt = 1e-4     # conservative pseudo time step

print(f"Solving natural convection: Ra={Ra}, Pr={Pr}, grid={N}x{N}, dt={dt}")
print(f"Diffusive CFL: Pr*dt/dx^2 = {Pr * dt / dx**2:.4f}, dt/dx^2 = {dt / dx**2:.4f}")

# --- Fields ---
T = np.zeros((N, N))
omega = np.zeros((N, N))
psi = np.zeros((N, N))

# Initial temperature: linear
for i in range(N):
    T[i, :] = 1.0 - i / (N - 1)

# --- Helper functions ---
def solve_poisson_jacobi(psi, omega, dx, n_sweeps=50):
    """Pure Jacobi iteration for nabla^2(psi) = -omega."""
    dx2 = dx * dx
    for _ in range(n_sweeps):
        psi_old = psi.copy()
        psi[1:-1, 1:-1] = 0.25 * (psi_old[2:, 1:-1] + psi_old[:-2, 1:-1] +
                                   psi_old[1:-1, 2:] + psi_old[1:-1, :-2] +
                                   dx2 * omega[1:-1, 1:-1])
    return psi

def laplacian_interior(phi, dx):
    """Returns laplacian values at interior points only (shape N-2 x N-2)."""
    return ((phi[2:, 1:-1] - 2*phi[1:-1, 1:-1] + phi[:-2, 1:-1]) / dx**2 +
            (phi[1:-1, 2:] - 2*phi[1:-1, 1:-1] + phi[1:-1, :-2]) / dx**2)

# --- Main loop ---
for iteration in range(max_iter):
    T_old_max = T.max()
    omega_old_max = omega.max()

    # 1. Solve Poisson for psi
    psi = solve_poisson_jacobi(psi, omega, dx, n_sweeps=40)

    # 2. Compute velocities (central differences)
    u = np.zeros((N, N))
    v = np.zeros((N, N))
    u[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2.0 * dx)
    v[1:-1, 1:-1] = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * dx)

    # 3. Advection of omega (upwind, interior only)
    u_int = u[1:-1, 1:-1]
    v_int = v[1:-1, 1:-1]

    # omega advection
    fwd_x_w = (omega[2:, 1:-1] - omega[1:-1, 1:-1]) / dx
    bwd_x_w = (omega[1:-1, 1:-1] - omega[:-2, 1:-1]) / dx
    fwd_y_w = (omega[1:-1, 2:] - omega[1:-1, 1:-1]) / dx
    bwd_y_w = (omega[1:-1, 1:-1] - omega[1:-1, :-2]) / dx
    adv_w = (u_int * np.where(u_int > 0, bwd_x_w, fwd_x_w) +
             v_int * np.where(v_int > 0, bwd_y_w, fwd_y_w))

    # T advection
    fwd_x_T = (T[2:, 1:-1] - T[1:-1, 1:-1]) / dx
    bwd_x_T = (T[1:-1, 1:-1] - T[:-2, 1:-1]) / dx
    fwd_y_T = (T[1:-1, 2:] - T[1:-1, 1:-1]) / dx
    bwd_y_T = (T[1:-1, 1:-1] - T[1:-1, :-2]) / dx
    adv_T = (u_int * np.where(u_int > 0, bwd_x_T, fwd_x_T) +
             v_int * np.where(v_int > 0, bwd_y_T, fwd_y_T))

    # 4. Diffusion
    lap_w = laplacian_interior(omega, dx)
    lap_T = laplacian_interior(T, dx)

    # 5. Buoyancy source: Ra*Pr*dT/dx (central)
    dTdx = (T[2:, 1:-1] - T[:-2, 1:-1]) / (2.0 * dx)

    # 6. Update (explicit Euler)
    omega[1:-1, 1:-1] += dt * (Pr * lap_w - adv_w + Ra * Pr * dTdx)
    T[1:-1, 1:-1] += dt * (lap_T - adv_T)

    # 7. BCs
    T[0, :] = 1.0       # hot left
    T[N-1, :] = 0.0     # cold right
    T[:, 0] = T[:, 1]   # adiabatic bottom
    T[:, N-1] = T[:, N-2]  # adiabatic top

    # Wall vorticity (Thom's formula)
    omega[0, :] = -2.0 * psi[1, :] / dx**2
    omega[N-1, :] = -2.0 * psi[N-2, :] / dx**2
    omega[:, 0] = -2.0 * psi[:, 1] / dx**2
    omega[:, N-1] = -2.0 * psi[:, N-2] / dx**2

    # Convergence check every 1000 steps
    if iteration % 1000 == 0 and iteration > 0:
        resid_psi = np.max(np.abs(
            (psi[2:, 1:-1] + psi[:-2, 1:-1] + psi[1:-1, 2:] + psi[1:-1, :-2]
             - 4*psi[1:-1, 1:-1]) / dx**2 + omega[1:-1, 1:-1]))
        dw_max = dt * np.max(np.abs(Pr * lap_w - adv_w + Ra * Pr * dTdx))
        dT_max = dt * np.max(np.abs(lap_T - adv_T))
        psi_max_val = np.max(np.abs(psi))

        if iteration % 5000 == 0:
            print(f"  iter {iteration:6d}: |psi|_max={psi_max_val:.6f}, "
                  f"dw={dw_max:.2e}, dT={dT_max:.2e}")

        if np.any(np.isnan(T)) or np.any(np.isnan(omega)):
            print("  DIVERGED! Aborting.")
            break

        if dw_max < tol and dT_max < tol:
            print(f"  Converged at iteration {iteration} (dw={dw_max:.2e}, dT={dT_max:.2e})")
            break

# Final velocities
u_final = np.zeros((N, N))
v_final = np.zeros((N, N))
u_final[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2.0 * dx)
v_final[1:-1, 1:-1] = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * dx)
speed = np.sqrt(u_final**2 + v_final**2)

# Nusselt number at hot wall
Nu_local = -(T[1, :] - T[0, :]) / dx
Nu_avg = np.trapz(Nu_local, dx=dx)
print(f"\nFinal: |psi|_max = {np.max(np.abs(psi)):.6f}")
print(f"Max speed = {np.max(speed):.4f}")
print(f"Average Nusselt number (hot wall): Nu = {Nu_avg:.3f}")
print(f"  (de Vahl Davis 1983 benchmark for Ra=1e3: Nu ~ 1.118)")

# --- Plotting ---
x_grid = np.linspace(0, 1, N)
y_grid = np.linspace(0, 1, N)
X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6))

cmap_T = cm.RdYlBu_r

# --- Panel (a): Isotherms + velocity vectors ---
ax1.set_aspect('equal')

levels_fill = np.linspace(0, 1, 41)
cf = ax1.contourf(X, Y, T, levels=levels_fill, cmap=cmap_T, alpha=0.85)

levels_lines = np.linspace(0.05, 0.95, 10)
cs = ax1.contour(X, Y, T, levels=levels_lines, colors='k', linewidths=0.8, alpha=0.6)
ax1.clabel(cs, inline=True, fontsize=7, fmt='%.2f')

# Velocity vectors (subsampled)
skip = 3
ax1.quiver(X[1::skip, 1::skip], Y[1::skip, 1::skip],
           u_final[1::skip, 1::skip], v_final[1::skip, 1::skip],
           color='#2C3E50', alpha=0.5, width=0.003,
           headwidth=3.5, headlength=4)

# Wall labels
ax1.text(-0.08, 0.5, "$T = 1$\n(Hot)", transform=ax1.transAxes,
         fontsize=10, ha='center', va='center', color='#C0392B', fontweight='bold',
         rotation=90)
ax1.text(1.08, 0.5, "$T = 0$\n(Cold)", transform=ax1.transAxes,
         fontsize=10, ha='center', va='center', color='#2980B9', fontweight='bold',
         rotation=90)
ax1.text(0.5, -0.06, "Adiabatic", transform=ax1.transAxes,
         fontsize=9, ha='center', color='gray')
ax1.text(0.5, 1.04, "Adiabatic", transform=ax1.transAxes,
         fontsize=9, ha='center', color='gray')

fig.colorbar(cf, ax=ax1, shrink=0.85, pad=0.12, label="Temperature $T$")
ax1.set_xlabel("$x$", fontsize=12)
ax1.set_ylabel("$y$", fontsize=12)
ax1.set_title("(a) Isotherms and Velocity Field", fontsize=13, pad=10)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)

# --- Panel (b): Streamlines ---
ax2.set_aspect('equal')

# Background: speed magnitude
cf2 = ax2.contourf(X, Y, speed, levels=30, cmap='YlOrRd', alpha=0.5)
fig.colorbar(cf2, ax=ax2, shrink=0.85, pad=0.05, label="Speed $|\\mathbf{u}|$")

# Streamlines with arrows (colored by speed)
strm = ax2.streamplot(x_grid, y_grid, u_final.T, v_final.T,
                       color='#2C3E50', density=1.8,
                       linewidth=1.0, arrowsize=1.2, arrowstyle='->')

ax2.set_xlabel("$x$", fontsize=12)
ax2.set_ylabel("$y$", fontsize=12)
ax2.set_title("(b) Streamlines and Flow Speed", fontsize=13, pad=10)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

# Parameter box
param_text = (f"$\\mathrm{{Ra}} = {Ra:.0f}$\n"
              f"$\\mathrm{{Pr}} = {Pr}$\n"
              f"Grid: ${N} \\times {N}$\n"
              f"$\\overline{{\\mathrm{{Nu}}}} = {Nu_avg:.3f}$")
ax2.text(0.97, 0.03, param_text, transform=ax2.transAxes,
         fontsize=9, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                   edgecolor='gray', alpha=0.9))

fig.suptitle("Natural Convection in a Differentially Heated Cavity ($\\mathrm{Ra} = 1000$)",
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig("/home/yzk/LBMProject/scripts/viz/natural_convection.png",
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("\nSaved: /home/yzk/LBMProject/scripts/viz/natural_convection.png")
