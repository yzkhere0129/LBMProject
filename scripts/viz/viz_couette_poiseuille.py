"""
viz_couette_poiseuille.py — Couette-Poiseuille Flow Visualization

Shows the analytical velocity profile decomposition for combined
Couette-Poiseuille flow, overlaid with LBM simulation data from:
  tests/validation/output_couette_poiseuille/velocity_profile.csv

Three cases on a single plot:
  1. Pure Couette   — linear shear driven by moving top wall
  2. Pure Poiseuille — parabolic profile driven by pressure gradient
  3. Combined       — superposition, matched to the simulation parameters

Simulation parameters (from CSV header):
  NY=128, H=127, U_top=0.05 lu, Re=10, nu=U_top*H/Re, fx=-1.1811e-05

Saved to:  scripts/viz/couette_poiseuille.png
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(
    SCRIPT_DIR,
    "../../tests/validation/output_couette_poiseuille/velocity_profile.csv",
)
OUT_PATH = os.path.join(SCRIPT_DIR, "couette_poiseuille.png")

# ---------------------------------------------------------------------------
# Flow parameters matching the simulation (from CSV header)
# ---------------------------------------------------------------------------
NY    = 128
H     = float(NY - 1)     # 127 lattice cells between walls
U_TOP = 0.05               # top wall speed [lattice units]
RE    = 10.0
NU    = U_TOP * H / RE     # kinematic viscosity [lu^2/ts]
# Pressure gradient chosen so that Poiseuille peak equals Couette peak
# (FX from CSV header: -1.1811e-05; recompute for analytical consistency)
FX    = -6.0 * U_TOP * NU / (H * H)

# ---------------------------------------------------------------------------
# Analytical profiles  u(eta),  eta = y/H in [0, 1]
# ---------------------------------------------------------------------------
eta_fine = np.linspace(0.0, 1.0, 600)
y_fine   = eta_fine * H

# Pure Couette: linear profile u_C = U_top * eta
u_couette = U_TOP * eta_fine

# Pure Poiseuille: parabolic profile u_P = (FX H^2)/(2 nu) * eta*(1-eta)
u_poiseuille = (FX * H * H) / (2.0 * NU) * eta_fine * (1.0 - eta_fine)

# Combined Couette-Poiseuille
u_combined = u_couette + u_poiseuille

# ---------------------------------------------------------------------------
# Read LBM simulation data from CSV
# ---------------------------------------------------------------------------
y_sim, u_sim = [], []
with open(CSV_PATH, "r") as fh:
    reader = csv.DictReader(row for row in fh if not row.startswith("#"))
    for row in reader:
        y_sim.append(float(row["y"]))
        u_sim.append(float(row["u_numerical"]))

y_sim  = np.array(y_sim)
u_sim  = np.array(u_sim)
eta_sim = y_sim / H

# ---------------------------------------------------------------------------
# Color and style
# ---------------------------------------------------------------------------
DARK_BG  = "#0f1117"
PANEL_BG = "#1a1d27"
GRID_CLR = "#2e3347"
TEXT_CLR = "#c8cfe8"
SUB_CLR  = "#8899bb"

C_COUETTE    = "#81c784"   # green
C_POISEUILLE = "#ce93d8"   # purple
C_COMBINED   = "#ff7043"   # orange-red (analytical)
C_LBM        = "#4fc3f7"   # cyan (numerical)
C_WALL       = "#ffd54f"   # yellow (wall nodes)

# ---------------------------------------------------------------------------
# Figure: single panel, horizontal velocity on x-axis, height (eta) on y-axis
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 7.0))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(PANEL_BG)
ax.tick_params(colors=TEXT_CLR, which="both", labelsize=10)
ax.xaxis.label.set_color(TEXT_CLR)
ax.yaxis.label.set_color(TEXT_CLR)
ax.title.set_color(TEXT_CLR)
for spine in ax.spines.values():
    spine.set_color(GRID_CLR)
ax.grid(True, color=GRID_CLR, linewidth=0.7, linestyle="--", alpha=0.6)

# Zero-velocity reference line
ax.axvline(0.0, color=GRID_CLR, linewidth=0.9, linestyle=":")

# ---- Analytical curves ----
ax.plot(u_couette, eta_fine,
        color=C_COUETTE, linewidth=2.2, linestyle="--",
        label="Pure Couette  (linear,  $u = U_{top}\\,\\eta$)")

ax.plot(u_poiseuille, eta_fine,
        color=C_POISEUILLE, linewidth=2.2, linestyle="-.",
        label=r"Pure Poiseuille  (parabolic,  $u = \frac{f_x H^2}{2\nu}\,\eta(1-\eta)$)")

ax.plot(u_combined, eta_fine,
        color=C_COMBINED, linewidth=2.8, linestyle="-",
        label="Combined Couette-Poiseuille  (analytical)", zorder=3)

# ---- LBM simulation data ----
# Interior nodes
interior = (eta_sim > 0.0) & (eta_sim < 1.0)
ax.scatter(u_sim[interior], eta_sim[interior],
           color=C_LBM, s=24, zorder=5, linewidths=0,
           label=f"LBM simulation  (BGK, $N={NY}$)")

# Wall nodes (boundary condition)
wall_mask = ~interior
ax.scatter(u_sim[wall_mask], eta_sim[wall_mask],
           color=C_WALL, s=55, zorder=6, marker="D", linewidths=0,
           label="Wall nodes (no-slip BC)")

# ---- Annotations: wall labels ----
ax.annotate("No-slip wall\n$u=0$",
            xy=(0.0, 0.0), xytext=(0.002, 0.05),
            color=TEXT_CLR, fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color=SUB_CLR, lw=0.8),
            ha="left")
ax.annotate(f"Moving wall\n$u = U_{{top}} = {U_TOP}$",
            xy=(U_TOP, 1.0), xytext=(U_TOP - 0.008, 0.92),
            color=TEXT_CLR, fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color=SUB_CLR, lw=0.8),
            ha="right")

# ---- Physics parameter box ----
info = (
    f"$Re = {RE:.0f}$\n"
    f"$U_{{top}} = {U_TOP}$ lu/ts\n"
    f"$H = {int(H)}$ cells\n"
    f"$\\nu = {NU:.4f}$ lu$^2$/ts\n"
    f"$f_x = {FX:.3e}$ lu/ts$^2$\n"
    f"Grid: $1 \\times {NY}$"
)
ax.text(0.015, 0.97, info,
        transform=ax.transAxes, ha="left", va="top",
        color=TEXT_CLR, fontsize=9.0,
        bbox=dict(boxstyle="round,pad=0.45", facecolor=PANEL_BG, edgecolor=GRID_CLR))

# ---- Axes labels ----
ax.set_xlabel("Velocity  $u$  [lattice units]", fontsize=12)
ax.set_ylabel(r"Normalised height  $\eta = y/H$", fontsize=12)
ax.set_ylim(-0.03, 1.06)

# Legend
leg = ax.legend(
    loc="upper right",
    facecolor="#1e2235", edgecolor=GRID_CLR,
    labelcolor=TEXT_CLR, fontsize=9.5,
    framealpha=0.9,
)

# ---------------------------------------------------------------------------
# Figure title
# ---------------------------------------------------------------------------
fig.suptitle(
    "Couette-Poiseuille Flow  |  BGK LBM vs Analytical",
    y=0.97, fontsize=15, fontweight="bold", color=TEXT_CLR,
)
fig.text(
    0.5, 0.925,
    r"$u(\eta) = U_{top}\,\eta \;+\; \frac{f_x H^2}{2\nu}\,\eta(1-\eta)$"
    r"    combined shear-driven and pressure-driven channel flow",
    ha="center", va="top", color=SUB_CLR, fontsize=10.5,
)

fig.subplots_adjust(top=0.87, bottom=0.10, left=0.10, right=0.97)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
fig.savefig(OUT_PATH, dpi=180, facecolor=DARK_BG, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
