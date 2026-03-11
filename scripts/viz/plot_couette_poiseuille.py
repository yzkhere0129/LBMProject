"""
Couette-Poiseuille flow visualization.

Reads the validated CSV from tests/validation/output_couette_poiseuille/
and plots numerical vs analytical velocity profiles for both pure Couette
and pure Poiseuille components, plus the combined profile.

Usage:
    python3 plot_couette_poiseuille.py
"""

import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "../../tests/validation/output_couette_poiseuille/velocity_profile.csv")
OUT_PATH = os.path.join(SCRIPT_DIR, "couette_poiseuille.png")

# ---------------------------------------------------------------------------
# Flow parameters (must match test_couette_poiseuille.cu)
# ---------------------------------------------------------------------------
NY   = 128
H    = float(NY - 1)        # 127
U_TOP = 0.05
RE   = 10.0
NU   = U_TOP * H / RE       # kinematic viscosity (lattice)
FX   = -6.0 * U_TOP * NU / (H * H)

def analytical(y):
    """Combined Couette-Poiseuille: u(eta) = U_top*(3*eta^2 - 2*eta)."""
    eta = y / H
    u_c = U_TOP * eta
    u_p = (FX * H * H) / (2.0 * NU) * eta * (1.0 - eta)
    return u_c + u_p, u_c, u_p

# ---------------------------------------------------------------------------
# Read CSV
# ---------------------------------------------------------------------------
y_num, u_num = [], []
with open(CSV_PATH, "r") as fh:
    reader = csv.DictReader(row for row in fh if not row.startswith("#"))
    for row in reader:
        y_num.append(float(row["y"]))
        u_num.append(float(row["u_numerical"]))

y_num = np.array(y_num)
u_num = np.array(u_num)
eta_num = y_num / H

# Analytical on fine grid
y_fine = np.linspace(0, H, 500)
eta_fine = y_fine / H
u_comb_fine, u_c_fine, u_p_fine = analytical(y_fine)

# Analytical at grid points
u_comb_grid = np.array([analytical(y)[0] for y in y_num])

# ---------------------------------------------------------------------------
# Figure layout: 1 row, 2 columns
# Left: combined profile (main result)
# Right: component breakdown
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(12, 7))
fig.patch.set_facecolor("#0f1117")

gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38,
                       left=0.09, right=0.95, top=0.88, bottom=0.12)

ax_main  = fig.add_subplot(gs[0])
ax_comps = fig.add_subplot(gs[1])

DARK_BG  = "#0f1117"
PANEL_BG = "#1a1d27"
GRID_CLR = "#2e3347"
TEXT_CLR = "#c8cfe8"
ACCENT1  = "#4fc3f7"   # cyan  - numerical
ACCENT2  = "#ff7043"   # orange-red - analytical
ACCENT3  = "#81c784"   # green - Couette component
ACCENT4  = "#ce93d8"   # purple - Poiseuille component

for ax in (ax_main, ax_comps):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_CLR, which="both")
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    for spine in ax.spines.values():
        spine.set_color(GRID_CLR)
    ax.grid(True, color=GRID_CLR, linewidth=0.6, linestyle="--", alpha=0.7)

# ---- Left panel: combined profile ----
ax_main.plot(u_comb_fine, eta_fine,
             color=ACCENT2, linewidth=2.5, label="Analytical", zorder=3)
ax_main.scatter(u_num[1:-1], eta_num[1:-1],
                color=ACCENT1, s=22, zorder=4,
                label=f"LBM (N={NY})", linewidths=0)
# Wall nodes
ax_main.scatter([u_num[0], u_num[-1]], [eta_num[0], eta_num[-1]],
                color="#ffd54f", s=40, zorder=5, marker="D",
                label="Wall nodes")

ax_main.axvline(0, color=GRID_CLR, linewidth=0.8, linestyle=":")
ax_main.set_xlabel("Velocity  u  (lattice units)", fontsize=12)
ax_main.set_ylabel(r"Normalised height  $\eta = y/H$", fontsize=12)
ax_main.set_title("Combined Couette-Poiseuille Profile", color=TEXT_CLR, fontsize=13, pad=8)
ax_main.legend(facecolor="#252836", edgecolor=GRID_CLR, labelcolor=TEXT_CLR,
               fontsize=10, loc="upper left")

# Error annotation
l2_err = math.sqrt(sum((u_num[j] - u_comb_grid[j])**2
                       for j in range(1, NY-1)) / (NY - 2)) / (max(u_comb_grid) - min(u_comb_grid))
ax_main.text(0.97, 0.04, f"L2 error = {l2_err*100:.2f}%",
             transform=ax_main.transAxes, ha="right", va="bottom",
             color=ACCENT1, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL_BG, edgecolor=GRID_CLR))

# ---- Right panel: component breakdown ----
ax_comps.plot(u_c_fine, eta_fine,
              color=ACCENT3, linewidth=2.5, linestyle="-",
              label="Couette (linear)")
ax_comps.plot(u_p_fine, eta_fine,
              color=ACCENT4, linewidth=2.5, linestyle="--",
              label="Poiseuille (parabolic)")
ax_comps.plot(u_comb_fine, eta_fine,
              color=ACCENT2, linewidth=2.5, linestyle="-.",
              label="Combined (analytical)")
ax_comps.scatter(u_num[1:-1], eta_num[1:-1],
                 color=ACCENT1, s=14, zorder=4,
                 label="LBM (combined)", linewidths=0)
ax_comps.axvline(0, color=GRID_CLR, linewidth=0.8, linestyle=":")

ax_comps.set_xlabel("Velocity  u  (lattice units)", fontsize=12)
ax_comps.set_ylabel(r"Normalised height  $\eta = y/H$", fontsize=12)
ax_comps.set_title("Flow Component Decomposition", color=TEXT_CLR, fontsize=13, pad=8)
ax_comps.legend(facecolor="#252836", edgecolor=GRID_CLR, labelcolor=TEXT_CLR,
                fontsize=9, loc="upper left")

# Physics annotation box
info = (
    f"Re = {RE:.0f}\n"
    f"U_top = {U_TOP:.3f} lu\n"
    f"$\\nu$ = {NU:.3f} lu²/ts\n"
    f"$f_x$ = {FX:.2e} lu/ts²\n"
    f"N = {NY}  (H = {int(H)})"
)
ax_comps.text(0.97, 0.04, info,
              transform=ax_comps.transAxes, ha="right", va="bottom",
              color=TEXT_CLR, fontsize=8.5,
              bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL_BG, edgecolor=GRID_CLR))

# ---------------------------------------------------------------------------
# Figure title
# ---------------------------------------------------------------------------
fig.text(0.5, 0.95,
         "Couette-Poiseuille Flow  |  BGK LBM vs Analytical Solution",
         ha="center", va="top", color=TEXT_CLR, fontsize=15, fontweight="bold")
fig.text(0.5, 0.905,
         r"$u(\eta) = U_{top}(3\eta^2 - 2\eta)$   where   $\eta = y/H$",
         ha="center", va="top", color="#8899bb", fontsize=11)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
fig.savefig(OUT_PATH, dpi=180, facecolor=fig.get_facecolor())
print(f"Saved: {OUT_PATH}")
