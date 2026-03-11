#!/usr/bin/env python3
"""
Rising bubble: plot real LBM-VOF simulation results.
Reads bubble_trajectory.csv and bubble_snapshots.csv from viz_bubble.cu.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

DIR = os.path.dirname(os.path.abspath(__file__))
TRAJ_CSV = os.path.join(DIR, "bubble_trajectory.csv")
SNAP_CSV = os.path.join(DIR, "bubble_snapshots.csv")
OUT      = os.path.join(DIR, "rising_bubble.png")

# ---- Parse trajectory CSV ----
meta = {}
steps, times, cy_mm, vy_mm_s, masses = [], [], [], [], []
with open(TRAJ_CSV) as f:
    for line in f:
        if line.startswith("#"):
            for tok in line[1:].split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    meta[k.strip()] = v.strip()
            continue
        if line.startswith("step"):
            continue
        parts = line.strip().split(",")
        if len(parts) < 5:
            continue
        steps.append(int(parts[0]))
        times.append(float(parts[1]))
        cy_mm.append(float(parts[2]))
        vy_mm_s.append(float(parts[3]))
        masses.append(float(parts[4]))

times   = np.array(times)
cy_mm   = np.array(cy_mm)
vy_mm_s = np.array(vy_mm_s)
masses  = np.array(masses)
times_ms = times * 1e3

# ---- Extract physics params from metadata ----
DX       = float(meta.get("dx", 5e-4))
DT       = float(meta.get("dt", 3.91e-5))
NU_LB    = float(meta.get("nu_LB", 0.0667))
G_LB     = float(meta.get("g_LB", 3e-5))
R_cells  = float(meta.get("R", 12))
RHO_RAT  = float(meta.get("rho_ratio", 10))
VEL_CONV = float(meta.get("vel_conv", 12.787))

# Derived
NU_PHYS  = NU_LB * DX**2 / DT
D_cells  = 2 * R_cells
delta_rho_LU = 1.0 - 1.0/RHO_RAT

# Stokes velocity (LU, then physical)
V_stokes_LU = 2 * R_cells**2 * delta_rho_LU * G_LB / (9 * NU_LB)
V_stokes = V_stokes_LU * VEL_CONV  # m/s
Re_stokes = V_stokes_LU * D_cells / NU_LB

# Schiller-Naumann iterative (LU)
V_sn_LU = V_stokes_LU
for _ in range(50):
    Re = max(abs(V_sn_LU) * D_cells / NU_LB, 0.01)
    Cd = 24/Re * (1 + 0.15 * Re**0.687)
    V_new = np.sqrt(8 * R_cells * delta_rho_LU * G_LB / (3 * Cd))
    if abs(V_new - V_sn_LU) < 1e-10:
        break
    V_sn_LU = V_new
V_sn = V_sn_LU * VEL_CONV  # m/s
Re_sn = V_sn_LU * D_cells / NU_LB

# Measured terminal velocity (last 20%)
n_tail = max(1, len(vy_mm_s) // 5)
V_meas_mm_s = np.mean(vy_mm_s[-n_tail:])
Re_meas = abs(V_meas_mm_s * 1e-3) * D_cells / (NU_LB * VEL_CONV / (DX/DT))  # approximate

print(f"V_stokes = {V_stokes*1e3:.1f} mm/s  (Re={Re_stokes:.1f}) — INVALID")
print(f"V_SN     = {V_sn*1e3:.1f} mm/s  (Re={Re_sn:.1f})")
print(f"V_meas   = {V_meas_mm_s:.1f} mm/s")
print(f"Mass: initial={masses[0]:.1f}, final={masses[-1]:.1f}, "
      f"loss={(masses[0]-masses[-1])/masses[0]*100:.3f}%")

# ---- Parse snapshot CSV ----
snapshots = {}
with open(SNAP_CSV) as f:
    for line in f:
        if line.startswith("#") or line.startswith("step") or line.strip() == "":
            continue
        parts = line.strip().split(",")
        s = int(parts[0])
        x, y, fv = float(parts[1]), float(parts[2]), float(parts[3])
        snapshots.setdefault(s, ([], [], []))
        snapshots[s][0].append(x)
        snapshots[s][1].append(y)
        snapshots[s][2].append(fv)

snap_steps = sorted(snapshots.keys())
# Pick 4 evenly spaced snapshots
if len(snap_steps) >= 4:
    idx = np.linspace(0, len(snap_steps)-1, 4, dtype=int)
    show_steps = [snap_steps[i] for i in idx]
else:
    show_steps = snap_steps

# Reconstruct grids
xs = sorted(set(snapshots[snap_steps[0]][0]))
ys = sorted(set(snapshots[snap_steps[0]][1]))
NX_g, NY_g = len(xs), len(ys)

def to_grid(data):
    xv, yv, fv = data
    g = np.ones((NY_g, NX_g))
    dx_g = xs[1] - xs[0] if NX_g > 1 else 1
    dy_g = ys[1] - ys[0] if NY_g > 1 else 1
    for x, y, f in zip(xv, yv, fv):
        i = min(NX_g-1, max(0, round((x - xs[0]) / dx_g)))
        j = min(NY_g-1, max(0, round((y - ys[0]) / dy_g)))
        g[j, i] = f
    return g

grids = {s: to_grid(snapshots[s]) for s in show_steps}

# ---- Figure ----
cmap = LinearSegmentedColormap.from_list("bubble", [
    (0.05, 0.15, 0.40), (0.15, 0.35, 0.65), (0.40, 0.60, 0.85),
    (0.95, 0.95, 0.95), (1.00, 0.75, 0.30), (1.00, 0.50, 0.10),
    (0.90, 0.20, 0.05)][::-1], N=256)

fig = plt.figure(figsize=(18, 7), facecolor="white")
gs = gridspec.GridSpec(1, len(show_steps) + 2, figure=fig,
                       wspace=0.35, left=0.04, right=0.97,
                       top=0.85, bottom=0.08,
                       width_ratios=[1]*len(show_steps) + [1.5, 1.5])

x_mm = [v*1e3 for v in xs]
y_mm = [v*1e3 for v in ys]
extent = [x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]]

# Snapshot panels
for k, step in enumerate(show_steps):
    ax = fig.add_subplot(gs[0, k])
    ax.imshow(grids[step], origin="lower", extent=extent,
              cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="bilinear")
    ax.contour(np.linspace(x_mm[0], x_mm[-1], NX_g),
               np.linspace(y_mm[0], y_mm[-1], NY_g),
               grids[step], levels=[0.5], colors=["white"], linewidths=1.2)
    t_ms = step * DT * 1e3
    ax.set_title(f"t = {t_ms:.1f} ms\n(step {step})", fontsize=9, fontweight="bold")
    ax.set_xlabel("x [mm]", fontsize=8)
    if k == 0:
        ax.set_ylabel("y [mm]", fontsize=8)
    ax.tick_params(labelsize=7)

# Trajectory panel
ax_traj = fig.add_subplot(gs[0, len(show_steps)])
ax_traj.plot(times_ms, cy_mm, "b-", linewidth=2, label="LBM simulation")

# Stokes reference (dashed, labeled as invalid)
y0 = cy_mm[0]
t_ref = np.linspace(0, times_ms[-1], 200)
ax_traj.plot(t_ref, y0 + V_stokes*1e3 * t_ref*1e-3,
             "r--", linewidth=1.2, alpha=0.5,
             label=f"Stokes: {V_stokes*1e3:.0f} mm/s (Re={Re_stokes:.0f}, INVALID)")
# Schiller-Naumann reference
ax_traj.plot(t_ref, y0 + V_sn*1e3 * t_ref*1e-3,
             "g--", linewidth=1.2, alpha=0.7,
             label=f"S-N: {V_sn*1e3:.0f} mm/s (Re={Re_sn:.0f})")

ax_traj.set_xlabel("Time [ms]", fontsize=10)
ax_traj.set_ylabel("Bubble centroid y [mm]", fontsize=10)
ax_traj.set_title("Centroid Trajectory", fontsize=11, fontweight="bold")
ax_traj.legend(fontsize=7, loc="upper left")
ax_traj.grid(True, alpha=0.3)

# Velocity panel
ax_vel = fig.add_subplot(gs[0, len(show_steps)+1])
ax_vel.plot(times_ms, vy_mm_s, "b-", linewidth=1.0, alpha=0.4, label="LBM (raw)")

# Smooth velocity (moving average)
if len(vy_mm_s) > 20:
    kernel = np.ones(20) / 20
    vy_smooth = np.convolve(vy_mm_s, kernel, mode="same")
    ax_vel.plot(times_ms, vy_smooth, "b-", linewidth=2.5, label="LBM (smoothed)")

ax_vel.axhline(V_stokes*1e3, color="r", linestyle="--", alpha=0.5,
               label=f"Stokes: {V_stokes*1e3:.0f} mm/s (INVALID)")
ax_vel.axhline(V_sn*1e3, color="g", linestyle="--", alpha=0.7,
               label=f"S-N: {V_sn*1e3:.0f} mm/s")
ax_vel.axhline(V_meas_mm_s, color="b", linestyle=":", alpha=0.6,
               label=f"Measured: {V_meas_mm_s:.0f} mm/s")

ax_vel.set_xlabel("Time [ms]", fontsize=10)
ax_vel.set_ylabel("Rise velocity [mm/s]", fontsize=10)
ax_vel.set_title("Terminal Velocity", fontsize=11, fontweight="bold")
ax_vel.legend(fontsize=7, loc="lower right")
ax_vel.grid(True, alpha=0.3)
# Set reasonable y limits
v_max_plot = max(V_stokes*1e3, max(vy_mm_s)*1.1) if len(vy_mm_s) > 0 else 200
ax_vel.set_ylim(-20, v_max_plot * 1.1)

# Mass conservation annotation
mass_loss = (masses[0] - masses[-1]) / masses[0] * 100
ax_vel.text(0.97, 0.15, f"mass loss: {mass_loss:.2f}%",
            transform=ax_vel.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="#ccc"))

# Main title
fig.suptitle(
    f"Rising Bubble (2D VOF-LBM)  |  Real Simulation  |  "
    f"$\\rho_L/\\rho_G$={RHO_RAT:.0f}  "
    f"R={R_cells:.0f} cells  "
    f"$\\tau$={NU_LB*3+0.5:.2f}  "
    f"$g_{{LB}}$={G_LB:.0e}  "
    f"Grid {NX_g}x{NY_g}",
    fontsize=12, fontweight="bold", y=0.98)

fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {OUT}")
