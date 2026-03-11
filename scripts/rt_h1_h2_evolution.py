#!/usr/bin/env python3
"""
Rayleigh-Taylor instability: bubble (h1) and spike (h2) amplitude evolution.

Reads the time-series CSV and plots h1, h2 vs t/t*, overlaying the theoretical
linear growth rate for early-time comparison.

Simulation parameters (from test_rt_benchmark_256x1024.cu):
  - At  = 0.5       (Atwood number)
  - g   = 1e-5      (lattice units, gravity)
  - L   = 256       (domain width, lattice units)
  - U*  = sqrt(At * g * L) = sqrt(0.5 * 1e-5 * 256) ≈ 0.03578 lu/step
  - t*  = L / U*    ≈ 7155 steps  (convective time)

Linear RT growth rate (classical, inviscid):
  gamma_RT = sqrt(At * g * k)    where k = 2*pi/L
  eta(t)   = eta_0 * exp(gamma_RT * t)

Note: t* in the CSV is the convective time L/U*, NOT 2*pi/gamma_RT.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# === PARAMETERS ===
CSV_PATH    = "/home/yzk/LBMProject/build/output_rt_benchmark/rt_benchmark_time_series.csv"
OUTPUT_PATH = "/home/yzk/LBMProject/build/output_rt_benchmark/rt_h1_h2_evolution.png"

# Physical / lattice parameters (must match simulation source)
AT  = 0.5         # Atwood number
G   = 1e-5        # gravitational acceleration (lattice units)
LX  = 256         # domain width (lattice units)

# Derived scales — matches source: t* = L / sqrt(At*g*L)
U_CHAR  = np.sqrt(AT * G * LX)        # characteristic velocity
T_STAR  = LX / U_CHAR                  # convective time scale (~7155 steps)

# Classical RT linear growth rate: gamma = sqrt(At * g * k)
K_WAVE  = 2.0 * np.pi / LX            # wavenumber of dominant mode
GAMMA   = np.sqrt(AT * G * K_WAVE)    # linear growth rate (steps^-1)

print(f"Simulation parameters:")
print(f"  At        = {AT}")
print(f"  g         = {G:.1e} lu/step^2")
print(f"  L         = {LX} lu")
print(f"  U*        = sqrt(At*g*L) = {U_CHAR:.5f} lu/step")
print(f"  t*        = L/U*         = {T_STAR:.2f} steps  (convective time)")
print(f"  k_wave    = 2*pi/L       = {K_WAVE:.6f} lu^-1")
print(f"  gamma_RT  = sqrt(At*g*k) = {GAMMA:.6f} step^-1")
print(f"  gamma*t*  = {GAMMA*T_STAR:.4f}   (growth per convective time)")

# === LOAD DATA ===
data = np.loadtxt(CSV_PATH, delimiter=',', skiprows=1)
step         = data[:, 0]
t_over_tstar = data[:, 1]
h1_cells     = data[:, 2]   # bubble amplitude (lattice cells from bottom)
h2_cells     = data[:, 3]   # spike  amplitude (lattice cells from bottom)
mass_error   = data[:, 4]
u_max        = data[:, 5]

# Convert t/t* back to lattice steps using T_STAR from formula
# (The CSV t* is computed the same way as our formula)
t_steps = t_over_tstar * T_STAR

print(f"\nData range:")
print(f"  t/t* : [{t_over_tstar.min():.3f}, {t_over_tstar.max():.3f}]")
print(f"  h1   : [{h1_cells.min():.2f}, {h1_cells.max():.2f}] cells")
print(f"  h2   : [{h2_cells.min():.2f}, {h2_cells.max():.2f}] cells")
print(f"  max mass error: {mass_error.max():.4f}%")

# Perturbation amplitude = displacement from initial equilibrium
h1_0  = h1_cells[0]
h2_0  = h2_cells[0]
eta1  = h1_cells - h1_0   # bubble rises upward
eta2  = h2_0 - h2_cells   # spike falls downward (positive convention)

# Theoretical exponential growth curve:
#   eta(t) = eta_0 * exp(gamma_RT * t)
# Fit eta_0 from the data at t/t* ~ 0.5 (early but detectable)
fit_idx = np.argmin(np.abs(t_over_tstar - 0.5))
eta1_ref = max(eta1[fit_idx], 1e-3)
t_ref    = t_steps[fit_idx]
# Theory: eta_0 = eta_ref * exp(-gamma * t_ref)
eta0_theory = eta1_ref * np.exp(-GAMMA * t_ref)

t_theory      = np.linspace(t_steps[1], t_steps.max(), 600)
tstar_theory  = t_theory / T_STAR
eta_theory    = eta0_theory * np.exp(GAMMA * t_theory)

# === EXPONENTIAL FIT in early linear regime (t/t* in [0.2, 2.0]) ===
early_mask = (eta1 > 1.0) & (t_over_tstar >= 0.2) & (t_over_tstar <= 2.0)
gamma_fit = np.nan
if early_mask.sum() >= 3:
    coeffs    = np.polyfit(t_steps[early_mask], np.log(eta1[early_mask]), 1)
    gamma_fit = coeffs[0]
    print(f"\nExponential fit (t/t* in [0.2, 2.0]):")
    print(f"  Fitted gamma  = {gamma_fit:.6f} step^-1")
    print(f"  Theory gamma  = {GAMMA:.6f} step^-1")
    print(f"  Ratio         = {gamma_fit/GAMMA:.3f}  (1.0 = perfect)")

# === PLOT ===
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ---- Left panel: h1 and h2 vs t/t* (absolute amplitudes) ----
ax = axes[0]
ax.plot(t_over_tstar, h1_cells, 'b-o', markersize=3, linewidth=1.5,
        label='h1 (bubble, light fluid rises)')
ax.plot(t_over_tstar, h2_cells, 'r-s', markersize=3, linewidth=1.5,
        label='h2 (spike, heavy fluid falls)')
ax.axhline(LX // 2, color='gray', linestyle=':', linewidth=1,
           label=f'Initial interface y={LX//2} (midpoint not domain centre)')
ax.set_xlabel('t / t*  (t* = L/U*, U* = sqrt(At·g·L))', fontsize=11)
ax.set_ylabel('Interface position (lattice cells from bottom)', fontsize=11)
ax.set_title('RT Instability: Bubble & Spike Positions', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

h1_final = h1_cells[-1]
h2_final = h2_cells[-1]
asymmetry = h1_final - h2_final
ax.annotate(f"h1={h1_final:.1f}", xy=(t_over_tstar[-1], h1_final),
            xytext=(-80, 10), textcoords='offset points', fontsize=9,
            color='blue', arrowprops=dict(arrowstyle='->', color='blue'))
ax.annotate(f"h2={h2_final:.1f}", xy=(t_over_tstar[-1], h2_final),
            xytext=(-80, -20), textcoords='offset points', fontsize=9,
            color='red', arrowprops=dict(arrowstyle='->', color='red'))

# ---- Right panel: perturbation amplitude on log scale vs t/t* ----
ax2 = axes[1]

valid_eta = eta1 > 0.05
ax2.semilogy(t_over_tstar[valid_eta], eta1[valid_eta], 'b-o', markersize=3,
             linewidth=1.5, label='eta1 = h1 - h1_0  (bubble displacement)')

valid_eta2 = eta2 > 0.05
ax2.semilogy(t_over_tstar[valid_eta2], eta2[valid_eta2], 'r-s', markersize=3,
             linewidth=1.5, label='eta2 = h2_0 - h2  (spike displacement)')

# Overlay theory
valid_theory = eta_theory > 0.01
ax2.semilogy(tstar_theory[valid_theory], eta_theory[valid_theory], 'k--',
             linewidth=1.8,
             label=f'Linear theory: eta_0·exp(gamma·t)\n'
                   f'gamma={GAMMA:.5f} step^-1')

# Mark nonlinear onset: k*eta ~ 1 → eta ~ L/(2*pi)
nl_onset = 1.0 / K_WAVE / (2 * np.pi)
ax2.axhline(nl_onset, color='orange', linestyle=':', linewidth=1.2,
            label=f'Nonlinear onset k*eta~0.1 → eta~{nl_onset:.1f} cells')

ax2.set_xlabel('t / t*', fontsize=11)
ax2.set_ylabel('Perturbation amplitude eta (lattice cells, log scale)', fontsize=11)
ax2.set_title('RT Instability: Exponential Growth Check (log scale)', fontsize=13)
ax2.legend(fontsize=8.5)
ax2.grid(True, which='both', alpha=0.3)
ax2.set_xlim(left=0)

# Bottom annotation
fit_str = f"{gamma_fit:.5f}" if not np.isnan(gamma_fit) else "N/A"
fig.text(0.5, 0.00,
         f"Final (t/t*={t_over_tstar[-1]:.2f}):  h1={h1_final:.1f}  h2={h2_final:.1f}  "
         f"h1-h2={asymmetry:.1f} cells  "
         f"(h1>h2: {h1_final > h2_final}, as expected for At=0.5)\n"
         f"Fitted gamma={fit_str}  Theory={GAMMA:.5f} step^-1  "
         f"t*={T_STAR:.0f} steps  At=0.5  g={G:.1e}  k=2pi/{LX}  "
         f"Max mass error={mass_error.max():.4f}%",
         ha='center', fontsize=9, style='italic',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

plt.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nPlot saved: {OUTPUT_PATH}")
