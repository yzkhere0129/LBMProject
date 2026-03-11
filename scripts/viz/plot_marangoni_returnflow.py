#!/usr/bin/env python3
"""
Marangoni 1D Return Flow: plot LBM vs analytical velocity profile.
Reads marangoni_returnflow.csv produced by the validation test.

LOCKED ANALYTICAL SOLUTION:
  u(y) = (tau_s * H) / (4 * mu) * (3*(y/H)^2 - 2*(y/H))
  Peak = tau_s * H / (4 * mu) = 0.0025 LU for the standard parameters.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DIR = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(DIR, "marangoni_returnflow.csv")
OUT = os.path.join(DIR, "marangoni_returnflow.png")

# ---- Parse CSV ----
meta = {}
y_norm, u_lbm, u_analytical = [], [], []
with open(CSV) as f:
    for line in f:
        if line.startswith("#"):
            for tok in line[1:].split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    meta[k.strip()] = v.strip()
            continue
        if line.startswith("y_norm"):
            continue
        parts = line.strip().split(",")
        if len(parts) < 3:
            continue
        y_norm.append(float(parts[0]))
        u_lbm.append(float(parts[1]))
        u_analytical.append(float(parts[2]))

y_norm = np.array(y_norm)
u_lbm = np.array(u_lbm)
u_analytical = np.array(u_analytical)

# ---- Extract metadata ----
tau   = float(meta.get("tau", 0.8))
nu    = float(meta.get("nu", 0.1))
tau_s = float(meta.get("tau_s", "1e-5"))
H     = int(meta.get("H_eff", meta.get("H", 100)))
NY    = int(meta.get("NY", 100))
NX    = int(meta.get("NX", 10))
steps = int(meta.get("steps", 400000))

# ---- Verify analytical peak (LOCKED) ----
mu = nu  # rho=1
A_TRUE = tau_s * H / (4.0 * mu)
assert abs(A_TRUE - 0.0025) < 1e-8, \
    f"FATAL: A_TRUE = {A_TRUE} != 0.0025. Parameters corrupted!"

# Recompute the analytical solution to ensure it was not tampered with
u_ana_check = A_TRUE * (3.0 * y_norm**2 - 2.0 * y_norm)
ana_match = np.allclose(u_analytical, u_ana_check, rtol=1e-6)
if not ana_match:
    print("WARNING: CSV analytical column does not match locked formula. Using recomputed values.")
    u_analytical = u_ana_check

# ---- Error metrics ----
abs_err = u_lbm - u_analytical

# L2 relative error (interior points only, exclude wall/surface)
interior = slice(1, -1)
L2 = np.sqrt(np.sum(abs_err[interior]**2) / np.sum(u_analytical[interior]**2))

# Linf relative error
rel_mask = np.abs(u_analytical) > 1e-10
Linf = np.max(np.abs(abs_err[rel_mask]) / np.abs(u_analytical[rel_mask]))

# Peak velocities
u_peak_ana = A_TRUE  # at y/H = 1: A*(3-2) = A = 0.0025
u_peak_lbm = u_lbm[-1]  # last cell closest to surface

# Zero crossings
zc_ana = 2.0 / 3.0  # u = A*(3y^2 - 2y) = 0 at y = 2/3
zc_lbm = np.nan
for i in range(len(u_lbm) - 1):
    if u_lbm[i] * u_lbm[i + 1] < 0:
        frac = u_lbm[i] / (u_lbm[i] - u_lbm[i + 1])
        zc_lbm = y_norm[i] + frac * (y_norm[i + 1] - y_norm[i])
        break

# Relative error profile
rel_err = np.full_like(abs_err, np.nan)
rel_err[rel_mask] = abs_err[rel_mask] / u_analytical[rel_mask]

# ---- Print summary ----
print(f"Marangoni 1D Return Flow Validation")
print(f"  tau={tau}, nu={nu}, tau_s={tau_s}, H={H}, NY={NY}, steps={steps}")
print(f"  A_TRUE = {A_TRUE:.6f} (LOCKED, must be 0.002500)")
print(f"  L2  relative error: {L2:.6e}")
print(f"  Linf relative error: {Linf:.6e}")
print(f"  Peak velocity (analytical): {u_peak_ana:.8e} (MUST be 0.00250000)")
print(f"  Peak velocity (LBM):        {u_peak_lbm:.8e}")
print(f"  Peak error:                  {abs(u_peak_lbm - u_peak_ana)/abs(u_peak_ana)*100:.4f}%")
print(f"  Zero crossing (analytical):  {zc_ana:.6f}")
zc_lbm_str = f"{zc_lbm:.6f}" if not np.isnan(zc_lbm) else "N/A"
print(f"  Zero crossing (LBM):         {zc_lbm_str}")
if not np.isnan(zc_lbm):
    print(f"  Zero crossing error:         {abs(zc_lbm - zc_ana):.6e}")
print(f"  PASS: L2 < 1% = {L2 < 0.01}")

# ---- Figure ----
fig = plt.figure(figsize=(16, 6), facecolor="white")
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35,
                       left=0.06, right=0.96, top=0.84, bottom=0.12,
                       width_ratios=[1.2, 1.2, 1.0])

# ---- Panel 1: Velocity Profile ----
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(u_analytical, y_norm, "r-", linewidth=2.0, label="Analytical", zorder=2)
ax1.plot(u_lbm, y_norm, "o", color="#2166ac", markersize=3.5,
         markeredgewidth=0.3, markeredgecolor="#333", alpha=0.85,
         label="LBM (D3Q19 TRT)", zorder=3)
# Mark the peak velocity
ax1.axvline(x=A_TRUE, color="r", linestyle=":", alpha=0.5, linewidth=1)
ax1.annotate(f"  peak = {A_TRUE:.4f}", xy=(A_TRUE, 0.95), fontsize=8,
             color="red", alpha=0.7)
ax1.set_xlabel("u(y) [LU]", fontsize=10)
ax1.set_ylabel("y / H", fontsize=10)
ax1.set_title("Velocity Profile", fontsize=11, fontweight="bold")
ax1.legend(fontsize=9, loc="best")
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=8)

# ---- Panel 2: Error Profile ----
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(abs_err, y_norm, "k-", linewidth=1.5, label="Absolute error", zorder=2)

# Twin x-axis for relative error
ax2r = ax2.twiny()
ax2r.plot(rel_err, y_norm, "--", color="#d6604d", linewidth=1.2,
          alpha=0.8, label="Relative error", zorder=2)
ax2r.set_xlabel("Relative error", fontsize=9, color="#d6604d")
ax2r.tick_params(labelsize=7, colors="#d6604d")

ax2.set_xlabel("u_lbm - u_analytical [LU]", fontsize=10)
ax2.set_ylabel("y / H", fontsize=10)
ax2.set_title("Error Profile", fontsize=11, fontweight="bold")
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=8)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2r.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")

# ---- Panel 3: Validation Metrics ----
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis("off")
ax3.set_title("Validation Metrics", fontsize=11, fontweight="bold")

zc_err_str = f"{abs(zc_lbm - zc_ana):.6f}" if not np.isnan(zc_lbm) else "N/A"

# Color-code pass/fail
l2_color = "green" if L2 < 0.01 else "red"
peak_color = "green" if abs(u_peak_lbm - u_peak_ana)/abs(u_peak_ana) < 0.05 else "red"

text_lines = [
    r"$\bf{L_2\ Error\ vs\ TRUE\ Analytical}$",
    f"  L2  relative:  {L2:.4e}",
    f"  Linf relative: {Linf:.4e}",
    f"  {'PASS' if L2 < 0.01 else 'FAIL'} (threshold: 1%)",
    "",
    r"$\bf{Peak\ Velocity\ (LOCKED)}$",
    f"  Analytical:  {u_peak_ana:.6f}",
    f"  LBM:         {u_peak_lbm:.6f}",
    f"  Error:       {abs(u_peak_lbm-u_peak_ana)/abs(u_peak_ana)*100:.3f}%",
    "",
    r"$\bf{Zero\ Crossing\ (y/H)}$",
    f"  Analytical:  {zc_ana:.6f}",
    f"  LBM:         {zc_lbm_str}",
    f"  Error:       {zc_err_str}",
    "",
    r"$\bf{Parameters}$",
    f"  TRT (Lambda=3/16)",
    f"  tau={tau}, nu={nu}, tau_s={tau_s}",
    f"  H={H}, NY={NY}, steps={steps:,}",
]

text_block = "\n".join(text_lines)
ax3.text(0.05, 0.95, text_block, transform=ax3.transAxes,
         fontsize=9, fontfamily="monospace", verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.5", fc="#f7f7f7", ec="#aaa",
                   alpha=0.95))

# ---- Figure title ----
fig.suptitle(
    r"Marangoni 1D Return Flow — LBM vs Analytical  |  "
    rf"$\tau$={tau}, $\nu$={nu}, $\tau_s$={tau_s}, peak={A_TRUE:.4f}",
    fontsize=12, fontweight="bold", y=0.97)

fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {OUT}")
