#!/usr/bin/env python3
"""
Stefan Problem: 1D melting front visualization.

Analytical Neumann solution for a semi-infinite solid initially at T_melt,
with a hot wall suddenly applied at x=0.

Parameters:
  St = 0.5, T_hot = 1051 K, T_melt = 1000 K, alpha = 1e-6 m^2/s

The similarity variable lambda satisfies:
  lambda * exp(lambda^2) * erf(lambda) = St / sqrt(pi)

Temperature profiles:
  T_liquid(x,t) = T_hot - (T_hot - T_melt) * erf(x/(2*sqrt(alpha*t))) / erf(lambda)
  T_solid(x,t)  = T_melt * erfc(x/(2*sqrt(alpha_s*t))) / erfc(lambda * sqrt(alpha/alpha_s))

Front position:
  s(t) = 2 * lambda * sqrt(alpha * t)
"""

import numpy as np
from scipy.special import erf, erfc
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --- Physical parameters ---
T_hot = 1051.0    # K, hot wall temperature
T_melt = 1000.0   # K, melting temperature
T_cold = T_melt    # solid initially at T_melt (classical Stefan)
St = 0.5           # Stefan number = c_p*(T_hot - T_melt) / L
alpha_l = 1e-6     # m^2/s, liquid thermal diffusivity
alpha_s = alpha_l   # assume equal diffusivities for simplicity

# --- Solve transcendental equation for lambda ---
# St/sqrt(pi) = lambda * exp(lambda^2) * erf(lambda)
def transcendental(lam):
    return lam * np.exp(lam**2) * erf(lam) - St / np.sqrt(np.pi)

lam = brentq(transcendental, 0.01, 2.0)
print(f"Similarity parameter lambda = {lam:.6f}")

# --- Panel (a): Temperature and liquid fraction at a snapshot ---
t_snap = 50.0  # seconds
s_snap = 2.0 * lam * np.sqrt(alpha_l * t_snap)  # front position at t_snap

L_domain = 0.025  # 25 mm domain
x = np.linspace(0, L_domain, 2000)

# Temperature profile
T = np.zeros_like(x)
eta = x / (2.0 * np.sqrt(alpha_l * t_snap))

# Liquid region: x < s(t)
mask_liq = x < s_snap
T[mask_liq] = T_hot - (T_hot - T_melt) * erf(eta[mask_liq]) / erf(lam)

# Solid region: x >= s(t)
mask_sol = x >= s_snap
eta_s = x[mask_sol] / (2.0 * np.sqrt(alpha_s * t_snap))
lam_s = lam * np.sqrt(alpha_l / alpha_s)
T[mask_sol] = T_melt * erfc(eta_s) / erfc(lam_s)

# Liquid fraction: step function at front
f_l = np.where(x < s_snap, 1.0, 0.0)

# --- Panel (b): Front position vs time ---
t_arr = np.linspace(0.01, 100.0, 500)
s_arr = 2.0 * lam * np.sqrt(alpha_l * t_arr)

# --- Plotting ---
fig = plt.figure(figsize=(14, 5.5))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1], wspace=0.35)

# Color palette
col_T = "#C0392B"
col_fl = "#2980B9"
col_front = "#27AE60"
col_bg = "#FAFAFA"

# --- Panel (a) ---
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor(col_bg)

# Shade liquid and solid regions
ax1.axvspan(0, s_snap * 1000, alpha=0.08, color=col_T, label="_nolegend_")
ax1.axvspan(s_snap * 1000, L_domain * 1000, alpha=0.08, color=col_fl, label="_nolegend_")

# Temperature
ln1 = ax1.plot(x * 1000, T, color=col_T, linewidth=2.2, label="Temperature $T(x)$")
ax1.set_xlabel("Position $x$ [mm]", fontsize=12)
ax1.set_ylabel("Temperature $T$ [K]", fontsize=12, color=col_T)
ax1.tick_params(axis='y', labelcolor=col_T)
ax1.set_ylim(990, 1060)
ax1.set_xlim(0, L_domain * 1000)

# Melting front vertical line
ax1.axvline(s_snap * 1000, color=col_front, linewidth=1.8, linestyle='--',
            label=f"Melting front $s = {s_snap*1000:.2f}$ mm")

# Liquid fraction on twin axis
ax2 = ax1.twinx()
ln2 = ax2.plot(x * 1000, f_l, color=col_fl, linewidth=2.0, linestyle='-',
               label="Liquid fraction $f_l(x)$", alpha=0.85)
ax2.set_ylabel("Liquid fraction $f_l$", fontsize=12, color=col_fl)
ax2.tick_params(axis='y', labelcolor=col_fl)
ax2.set_ylim(-0.1, 1.5)

# T_melt annotation
ax1.axhline(T_melt, color='gray', linewidth=0.8, linestyle=':')
ax1.annotate("$T_{melt}$", xy=(L_domain * 1000 * 0.75, T_melt),
             xytext=(L_domain * 1000 * 0.78, T_melt + 12),
             fontsize=10, color='gray',
             arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

# Region labels
ax1.text(s_snap * 500, 1055, "Liquid", fontsize=11, ha='center',
         color=col_T, fontstyle='italic', fontweight='bold')
ax1.text((s_snap * 1000 + L_domain * 1000) / 2, 1055, "Solid", fontsize=11, ha='center',
         color=col_fl, fontstyle='italic', fontweight='bold')

# Combined legend
from matplotlib.lines import Line2D
front_handle = Line2D([0], [1], color=col_front, linewidth=1.8, linestyle='--')
lines_for_legend = ln1 + ln2 + [front_handle]
labels_for_legend = [l.get_label() for l in ln1 + ln2]
labels_for_legend.append(f"Melting front $s = {s_snap*1000:.2f}$ mm")
ax1.legend(lines_for_legend, labels_for_legend, loc='upper right', fontsize=9,
           framealpha=0.9, edgecolor='gray')

ax1.set_title(f"(a) Stefan Problem: Profiles at $t = {t_snap:.0f}$ s", fontsize=13, pad=10)

# --- Panel (b) ---
ax3 = fig.add_subplot(gs[1])
ax3.set_facecolor(col_bg)

ax3.plot(t_arr, s_arr * 1000, color=col_front, linewidth=2.5)
ax3.fill_between(t_arr, 0, s_arr * 1000, alpha=0.12, color=col_front)

# Mark the snapshot time
ax3.plot(t_snap, s_snap * 1000, 'o', color=col_T, markersize=8, zorder=5,
         markeredgecolor='white', markeredgewidth=1.5)
ax3.annotate(f"$t = {t_snap:.0f}$ s\n$s = {s_snap*1000:.2f}$ mm",
             xy=(t_snap, s_snap * 1000),
             xytext=(t_snap + 12, s_snap * 1000 + 1.5),
             fontsize=9, color=col_T,
             arrowprops=dict(arrowstyle='->', color=col_T, lw=1.2),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=col_T, alpha=0.8))

# sqrt(t) reference
ax3.text(75, 2.0 * lam * np.sqrt(alpha_l * 75) * 1000 + 1.0,
         r"$s(t) = 2\lambda\sqrt{\alpha t}$",
         fontsize=11, color=col_front, fontstyle='italic')

ax3.set_xlabel("Time $t$ [s]", fontsize=12)
ax3.set_ylabel("Front position $s(t)$ [mm]", fontsize=12)
ax3.set_xlim(0, 100)
ax3.set_ylim(0, None)
ax3.grid(True, alpha=0.3, linestyle='-')

# Parameter box
param_text = (f"$\\mathrm{{St}} = {St}$\n"
              f"$T_{{hot}} = {T_hot:.0f}$ K\n"
              f"$T_{{melt}} = {T_melt:.0f}$ K\n"
              f"$\\alpha = {alpha_l:.0e}$ m$^2$/s\n"
              f"$\\lambda = {lam:.4f}$")
ax3.text(0.05, 0.95, param_text, transform=ax3.transAxes,
         fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                   edgecolor='gray', alpha=0.9))

ax3.set_title("(b) Melting Front Position $s(t)$", fontsize=13, pad=10)

fig.suptitle("Stefan Problem: Analytical Neumann Solution for 1D Melting",
             fontsize=14, fontweight='bold', y=1.02)

fig.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.12)
plt.savefig("/home/yzk/LBMProject/scripts/viz/stefan_melting.png",
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: /home/yzk/LBMProject/scripts/viz/stefan_melting.png")
