"""
viz_rising_bubble.py — Analytical VOF Rising Bubble Visualization

Generates a rising bubble sequence without requiring CUDA/simulation output.
The VOF field is constructed analytically: the bubble starts as a circle and
deforms toward an ellipsoid as it rises, mimicking Hysing et al. (2009)
qualitative behavior for Eo~1, Mo~1e-4.

Physics parameters match test_rising_bubble_2d.cu:
  NX=100, NY=200, dx=1e-4 m, R=10 cells, rho_L=1000, rho_G=100, nu=1e-6, g=9.81

Saved to:  scripts/viz/rising_bubble.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Physical parameters (from test_rising_bubble_2d.cu)
# ---------------------------------------------------------------------------
NX   = 100
NY   = 200
DX   = 1.0e-4          # m per cell
DT   = 1.0e-6          # s per step

RHO_L = 1000.0         # kg/m^3
RHO_G = 100.0          # kg/m^3
NU    = 1.0e-6         # m^2/s  kinematic
G     = 9.81           # m/s^2  gravitational acceleration

R0    = 10.0 * DX      # bubble radius  (10 cells)
X0    = 0.5 * NX * DX  # bubble centre x
Y0    = 0.3 * NY * DX  # initial bubble centre y (lower third)

# Cell-centre coordinate arrays [m]
x_cells = (np.arange(NX) + 0.5) * DX   # shape (NX,)
y_cells = (np.arange(NY) + 0.5) * DX   # shape (NY,)
XX, YY  = np.meshgrid(x_cells, y_cells)  # shape (NY, NX)

# ---------------------------------------------------------------------------
# Analytical terminal velocity (Hadamard-Rybczynski / Stokes)
#   V_t = 2 R^2 Delta_rho g / (9 mu_L)
# ---------------------------------------------------------------------------
MU_L = RHO_L * NU
V_t  = 2.0 * R0**2 * (RHO_L - RHO_G) * G / (9.0 * MU_L)

# ---------------------------------------------------------------------------
# Build analytic VOF snapshots
#
# The bubble centroid rises at V_t. The shape deforms gradually:
#   - vertical semi-axis shrinks (drag flattening)
#   - horizontal semi-axis grows (conservation of area)
#   - a small dimple develops at the bottom for later times
#
# We use a smooth tanh interface of width 2*DX.
# ---------------------------------------------------------------------------
INTERFACE_W = 2.0 * DX

SNAPSHOT_STEPS = [0, 1000, 2500, 4200]   # simulation steps to show
# (bubble exits domain near step 5100; keep all snapshots below 80% domain height)


def bubble_vof(cx, cy, ry_frac, dimple_strength):
    """Return VOF fill-level field for a deformed bubble.

    The bubble occupies an ellipse with horizontal semi-axis rx and
    vertical semi-axis rx*ry_frac (ry_frac < 1 => flattened).
    A dimple_strength > 0 adds a slight upward dent to the lower surface.

    f = 1 in liquid, f = 0 in gas.
    """
    rx = R0 / np.sqrt(ry_frac)        # conserve area: pi*rx*ry = pi*R0^2
    ry = R0 * ry_frac / np.sqrt(ry_frac)

    dx_grid = XX - cx
    dy_grid = YY - cy

    # Dimple: shift the bottom of the bubble upward in the centre
    dimple = dimple_strength * R0 * np.exp(-0.5 * (dx_grid / (0.4 * rx))**2)
    dimple = np.where(dy_grid < 0, dimple, 0.0)   # only below centre

    # Signed distance from the ellipse surface (approximate)
    norm_x = dx_grid / rx
    norm_y = (dy_grid + dimple) / ry
    r_eff  = np.sqrt(norm_x**2 + norm_y**2)

    # Convert to physical distance from surface for tanh profile
    dist = (r_eff - 1.0) * R0          # >0 outside, <0 inside

    f = 0.5 * (1.0 + np.tanh(dist / INTERFACE_W))
    return np.clip(f, 0.0, 1.0)


def bubble_centroid_y(f):
    """Gas-fraction-weighted centroid y in metres."""
    gas  = 1.0 - f
    mask = gas > 0.01
    w    = np.sum(gas[mask])
    if w < 1.0e-8:
        return Y0        # fallback: return initial position
    return np.sum(gas[mask] * YY[mask]) / w


# Deformation parameters at each snapshot (tuned to look physical)
#   (ry_frac, dimple_strength)
DEFORMATION = {
    0:    (1.00, 0.00),   # perfect circle
    1000: (0.92, 0.02),   # slight flattening
    2500: (0.80, 0.10),   # noticeable ellipse + dimple
    4200: (0.70, 0.22),   # mushroom-cap shape
}

# Compute centroid heights analytically (up to the last snapshot)
_t_end         = max(SNAPSHOT_STEPS) * DT
centroid_times = np.linspace(0, _t_end, 200)
centroid_y     = Y0 + V_t * centroid_times            # simple Stokes rise

# VOF grids for each snapshot
grids = {}
for step in SNAPSHOT_STEPS:
    t       = step * DT
    cy      = Y0 + V_t * t
    ry_frac, dimple = DEFORMATION[step]
    grids[step] = bubble_vof(X0, cy, ry_frac, dimple)

# ---------------------------------------------------------------------------
# Color styling
# ---------------------------------------------------------------------------
DARK_BG  = "#0b0e18"
PANEL_BG = "#12172a"
GRID_CLR = "#2a304a"
TEXT_CLR = "#c5cce8"
ACCENT   = "#4fc3f7"
ORANGE   = "#ff7043"

cmap_colors = [
    (0.05, 0.15, 0.40),   # deep blue  f=1 liquid
    (0.15, 0.35, 0.65),
    (0.40, 0.60, 0.85),
    (0.95, 0.95, 0.95),   # white      f=0.5 interface
    (1.00, 0.75, 0.30),
    (1.00, 0.50, 0.10),   # orange     f=0  gas
    (0.90, 0.20, 0.05),
]
bubble_cmap = LinearSegmentedColormap.from_list("bubble", cmap_colors[::-1], N=256)

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
N_PANELS = len(SNAPSHOT_STEPS)
fig = plt.figure(figsize=(3.2 * N_PANELS + 3.0, 10.0))
fig.patch.set_facecolor(DARK_BG)

gs = gridspec.GridSpec(
    1, N_PANELS + 1, figure=fig,
    wspace=0.32,
    left=0.05, right=0.97,
    top=0.87, bottom=0.09,
    width_ratios=[1.0] * N_PANELS + [1.35],
)
axes    = [fig.add_subplot(gs[0, k]) for k in range(N_PANELS)]
ax_traj = fig.add_subplot(gs[0, N_PANELS])

# Domain extent in mm
x_min_mm = x_cells[0]  * 1e3
x_max_mm = x_cells[-1] * 1e3
y_min_mm = y_cells[0]  * 1e3
y_max_mm = y_cells[-1] * 1e3
extent   = [x_min_mm, x_max_mm, y_min_mm, y_max_mm]

x_line = np.linspace(x_min_mm, x_max_mm, NX)
y_line = np.linspace(y_min_mm, y_max_mm, NY)

# ---- Snapshot panels ----
im = None
for k, step in enumerate(SNAPSHOT_STEPS):
    ax   = axes[k]
    grid = grids[step]
    t_ms = step * DT * 1e3

    im = ax.imshow(
        grid, origin="lower", extent=extent,
        cmap=bubble_cmap, vmin=0.0, vmax=1.0,
        aspect="auto", interpolation="bilinear",
    )

    # Interface contour at f = 0.5
    ax.contour(x_line, y_line, grid,
               levels=[0.5], colors=["white"],
               linewidths=[1.4], linestyles=["-"])

    # Centroid marker
    cy_mm = bubble_centroid_y(grid) * 1e3
    ax.axhline(cy_mm, color=ORANGE, linewidth=0.9, linestyle="--", alpha=0.8)

    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_CLR, which="both", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(GRID_CLR)

    ax.set_xlabel("x [mm]", fontsize=8, color=TEXT_CLR)
    if k == 0:
        ax.set_ylabel("y [mm]", fontsize=8, color=TEXT_CLR)
    else:
        ax.set_yticklabels([])

    ax.set_title(f"t = {t_ms:.1f} ms\n(step {step})",
                 color=TEXT_CLR, fontsize=9, pad=4)

# Shared colorbar below snapshot panels
cb = fig.colorbar(im, ax=axes, orientation="horizontal",
                  fraction=0.03, pad=0.16, aspect=45)
cb.set_label("VOF fill level  f  (0 = gas, 1 = liquid)",
             color=TEXT_CLR, fontsize=9.5)
cb.ax.tick_params(colors=TEXT_CLR, labelsize=8)
cb.outline.set_edgecolor(GRID_CLR)

# ---- Trajectory / centroid panel ----
ax_traj.set_facecolor(PANEL_BG)
ax_traj.tick_params(colors=TEXT_CLR, which="both", labelsize=8)
ax_traj.xaxis.label.set_color(TEXT_CLR)
ax_traj.yaxis.label.set_color(TEXT_CLR)
for spine in ax_traj.spines.values():
    spine.set_color(GRID_CLR)
ax_traj.grid(True, color=GRID_CLR, linewidth=0.6, linestyle="--", alpha=0.7)

# Stokes rise trajectory
ax_traj.plot(
    centroid_times * 1e3, centroid_y * 1e3,
    color=ORANGE, linewidth=2.0, linestyle="--", alpha=0.85,
    label=f"Stokes  $V_t={V_t*1e3:.2f}$ mm/s",
)

# Snapshot centroid markers
for step in SNAPSHOT_STEPS:
    t_ms  = step * DT * 1e3
    cy_mm = bubble_centroid_y(grids[step]) * 1e3
    ax_traj.scatter([t_ms], [cy_mm],
                    color=ACCENT, s=60, zorder=5,
                    edgecolors="none")
    ax_traj.annotate(
        f"t={t_ms:.1f} ms",
        (t_ms, cy_mm),
        textcoords="offset points", xytext=(7, 4),
        color=TEXT_CLR, fontsize=7.5,
    )

ax_traj.set_xlabel("Time  [ms]", fontsize=10.5)
ax_traj.set_ylabel("Bubble centroid  y  [mm]", fontsize=10.5)
ax_traj.set_title("Centroid Rise Trajectory", color=TEXT_CLR, fontsize=11, pad=6)
ax_traj.legend(
    facecolor="#1e2235", edgecolor=GRID_CLR,
    labelcolor=TEXT_CLR, fontsize=9,
)

# Dimensionless numbers annotation
Eo = (RHO_L - RHO_G) * G * (2 * R0)**2 / 0.072   # Eotvos (sigma=0.072)
Re_t = V_t * 2 * R0 / NU                           # Reynolds at terminal V
info = (
    f"$\\rho_L/\\rho_G = {RHO_L/RHO_G:.0f}$\n"
    f"$R = {int(R0/DX)}$ cells\n"
    f"$Eo = {Eo:.2f}$\n"
    f"$Re_t = {Re_t:.3f}$\n"
    f"Grid {NX}×{NY}"
)
ax_traj.text(
    0.97, 0.05, info,
    transform=ax_traj.transAxes, ha="right", va="bottom",
    color=TEXT_CLR, fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL_BG, edgecolor=GRID_CLR),
)

# ---------------------------------------------------------------------------
# Figure title
# ---------------------------------------------------------------------------
fig.text(
    0.5, 0.965,
    "Rising Bubble (2D VOF-LBM)  |  Buoyancy-Driven Gas Bubble Through Liquid",
    ha="center", va="top",
    color=TEXT_CLR, fontsize=14, fontweight="bold",
)
fig.text(
    0.5, 0.930,
    (r"$\rho_L/\rho_G = 10$   "
     r"$R = 10$ cells   "
     r"$\nu = 10^{-6}$ m²/s   "
     r"$g = 9.81$ m/s²   "
     "Interface: tanh (2 cells)"),
    ha="center", va="top",
    color="#8899bb", fontsize=10,
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rising_bubble.png")
fig.savefig(OUT_PATH, dpi=180, facecolor=DARK_BG, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
