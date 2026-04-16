#!/usr/bin/env python3
"""
plot_enthalpy_diag.py
Generate acceptance report figures for the enthalpy diagnostic refactor.

Uses data embedded directly from the test run (no external CSV required).
Saves two PNGs:
  enthalpy_diag_residual.png    — new vs old formula residual vs time
  enthalpy_diag_energy.png      — E_in vs E_new vs E_naive over time
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ============================================================================
# Data from test run (test_enthalpy_diag_validation PhantomEnergyOldVsNew,
# sampled every 500 steps at dt=1e-8s on 40^3 grid, P=1000W, 316L material)
# ============================================================================
# Columns: step, t_us, E_in_mJ, E_new_mJ, E_naive_mJ, res_new_pct, res_naive_pct, f_mushy
data = np.array([
    [  500,   5.0,   5.00,   5.00,   5.00,  0.01,  0.01, 0.000],
    [ 1000,  10.0,  10.00,  10.00,  10.00,  0.01,  0.01, 0.000],
    [ 1500,  15.0,  15.00,  15.00,  15.00,  0.01,  0.01, 0.000],
    [ 2000,  20.0,  20.00,  20.00,  20.00,  0.00,  0.00, 0.000],
    [ 2500,  25.0,  25.00,  25.00,  25.00,  0.00,  0.00, 0.000],
    [ 3000,  30.0,  30.00,  30.00,  30.00,  0.00,  0.00, 0.000],
    [ 3500,  35.0,  35.00,  35.00,  35.00,  0.01,  0.01, 0.000],
    [ 4000,  40.0,  40.00,  40.00,  40.00,  0.01,  0.01, 0.000],
    [ 4500,  45.0,  45.00,  44.88,  46.85,  0.27,  4.12, 1.000],
    [ 5000,  50.0,  50.00,  49.22,  56.45,  1.56, 12.90, 1.000],
    [ 5500,  55.0,  55.00,  53.26,  64.63,  3.17, 17.52, 1.000],
    [ 6000,  60.0,  60.00,  57.07,  71.73,  4.87, 19.55, 1.000],
    [ 6500,  65.0,  65.00,  60.83,  77.87,  6.40, 19.80, 0.000],
    [ 7000,  70.0,  70.00,  65.84,  82.87,  5.94, 18.39, 0.000],
    [ 7500,  75.0,  74.99,  70.84,  87.88,  5.54, 17.18, 0.000],
    [ 8000,  80.0,  79.99,  75.85,  92.88,  5.19, 16.11, 0.000],
    [ 8500,  85.0,  84.99,  80.85,  97.88,  4.88, 15.16, 0.000],
    [ 9000,  90.0,  89.99,  85.85, 102.88,  4.61, 14.32, 0.000],
    [ 9500,  95.0,  94.99,  90.85, 107.88,  4.36, 13.57, 0.000],
    [10000, 100.0,  99.99,  95.85, 112.88,  4.15, 12.89, 0.000],
])

t_us        = data[:, 1]
E_in        = data[:, 2]   # mJ
E_new       = data[:, 3]   # mJ
E_naive     = data[:, 4]   # mJ
res_new     = data[:, 5]   # %
res_naive   = data[:, 6]   # %
f_mushy     = data[:, 7]

# Phase boundaries (approximate, from T_mean trajectory)
t_solidus  = 43.4   # µs — T_mean reaches T_solidus
t_liquidus = 65.0   # µs — T_mean reaches T_liquidus

outdir = os.path.dirname(os.path.abspath(__file__)) + "/../"

# ============================================================================
# Figure 1: Energy vs time
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(t_us, E_in,    'k-',  linewidth=2,   label='Expected: $E_{in} = P \\cdot t$')
ax.plot(t_us, E_new,   'b-o', linewidth=1.5, markersize=4, label='New formula (enthalpyPerVolume)')
ax.plot(t_us, E_naive, 'r--s',linewidth=1.5, markersize=4, label='Old formula (naive ρ·cp·T)')

# Shade phase regions
ax.axvspan(0,          t_solidus,  alpha=0.08, color='blue',   label='Solid')
ax.axvspan(t_solidus,  t_liquidus, alpha=0.15, color='orange', label='Mushy')
ax.axvspan(t_liquidus, 100,        alpha=0.08, color='red',    label='Liquid')
ax.axvline(t_solidus,  color='orange', linestyle=':', linewidth=1.2)
ax.axvline(t_liquidus, color='red',    linestyle=':', linewidth=1.2)

ax.set_xlabel('Time [µs]', fontsize=12)
ax.set_ylabel('Energy [mJ]', fontsize=12)
ax.set_title('Closed-Box Energy Balance: 316L, P=1000 W, 40³ cells\n'
             'New vs Old Diagnostic Formula vs Expected $E_{in}$', fontsize=12)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.text(t_solidus + 0.5, 5, 'Mushy\nzone', fontsize=9, color='darkorange')
ax.text(t_liquidus + 0.5, 5, 'Liquid', fontsize=9, color='darkred')
ax.text(2, 5, 'Solid', fontsize=9, color='darkblue')

fig.tight_layout()
fname1 = outdir + "enthalpy_diag_energy.png"
fig.savefig(fname1, dpi=150)
print(f"Saved: {fname1}")

# ============================================================================
# Figure 2: Residual vs time (main acceptance plot)
# ============================================================================
fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Top panel: residuals
ax2a.plot(t_us, res_new,   'b-o', linewidth=1.5, markersize=4, label='New formula residual')
ax2a.plot(t_us, res_naive, 'r--s',linewidth=1.5, markersize=4, label='Old formula residual')
ax2a.axhline(2.0,  color='green', linestyle='--', linewidth=1, label='Target: 2%')
ax2a.axhline(10.0, color='gray',  linestyle=':',  linewidth=1, label='Acceptance: 10%')

ax2a.axvspan(0,          t_solidus,  alpha=0.08, color='blue')
ax2a.axvspan(t_solidus,  t_liquidus, alpha=0.15, color='orange')
ax2a.axvspan(t_liquidus, 100,        alpha=0.08, color='red')
ax2a.axvline(t_solidus,  color='orange', linestyle=':', linewidth=1.0)
ax2a.axvline(t_liquidus, color='red',    linestyle=':', linewidth=1.0)

ax2a.set_ylabel('Relative residual [%]', fontsize=11)
ax2a.set_title('Enthalpy Diagnostic Acceptance Report\n'
               '|E_diag − E_in| / E_in  for new vs old formula', fontsize=11)
ax2a.legend(fontsize=9, loc='upper right')
ax2a.grid(True, alpha=0.3)
ax2a.set_ylim(-1, 25)

# Annotate phantom-energy peak
idx_peak = np.argmax(res_naive)
ax2a.annotate(f'Old formula peak:\n{res_naive[idx_peak]:.1f}% at {t_us[idx_peak]:.0f} µs',
              xy=(t_us[idx_peak], res_naive[idx_peak]),
              xytext=(t_us[idx_peak] - 18, res_naive[idx_peak] - 3),
              arrowprops=dict(arrowstyle='->', color='red'),
              fontsize=9, color='red')

# Annotate new formula at mushy peak
idx_peak_new = np.argmax(res_new)
ax2a.annotate(f'New formula peak:\n{res_new[idx_peak_new]:.1f}% (ESM mismatch)',
              xy=(t_us[idx_peak_new], res_new[idx_peak_new]),
              xytext=(t_us[idx_peak_new] - 22, res_new[idx_peak_new] + 5),
              arrowprops=dict(arrowstyle='->', color='blue'),
              fontsize=9, color='blue')

# Bottom panel: f_mushy to show phase transition timing
ax2b.fill_between(t_us, f_mushy, alpha=0.5, color='orange', label='Fraction in mushy zone')
ax2b.axvline(t_solidus,  color='orange', linestyle=':', linewidth=1.0)
ax2b.axvline(t_liquidus, color='red',    linestyle=':', linewidth=1.0)
ax2b.set_xlabel('Time [µs]', fontsize=11)
ax2b.set_ylabel('f_mushy', fontsize=11)
ax2b.set_ylim(0, 1.1)
ax2b.legend(fontsize=9)
ax2b.grid(True, alpha=0.3)

# Phase labels
for ax in [ax2a, ax2b]:
    ax.text(t_solidus / 2, ax.get_ylim()[1] * 0.9, 'SOLID', ha='center',
            fontsize=9, color='navy', alpha=0.7)
    ax.text((t_solidus + t_liquidus) / 2, ax.get_ylim()[1] * 0.9, 'MUSHY', ha='center',
            fontsize=9, color='darkorange', alpha=0.7)
    ax.text((t_liquidus + 100) / 2, ax.get_ylim()[1] * 0.9, 'LIQUID', ha='center',
            fontsize=9, color='darkred', alpha=0.7)

fig2.tight_layout()
fname2 = outdir + "enthalpy_diag_residual.png"
fig2.savefig(fname2, dpi=150)
print(f"Saved: {fname2}")

# ============================================================================
# Print summary table
# ============================================================================
print("\n=== Regime Summary Table ===")
print(f"{'Regime':<18} {'t range [µs]':<16} {'Max new res%':<15} {'Max naive res%':<16} {'PASS?'}")
print("-" * 70)

solid_mask  = (f_mushy < 0.01) & (t_us < t_solidus)
mushy_mask  = f_mushy > 0.01
liquid_mask = (f_mushy < 0.01) & (t_us > t_liquidus)

for name, mask, t_range, threshold in [
    ("Solid-only",   solid_mask,  f"0–{t_solidus:.0f}",    0.1),
    ("Mushy-active", mushy_mask,  f"{t_solidus:.0f}–{t_liquidus:.0f}", 10.0),
    ("Liquid",       liquid_mask, f"{t_liquidus:.0f}–100",  10.0),
]:
    if not np.any(mask):
        print(f"{name:<18} {t_range:<16} {'N/A':<15} {'N/A':<16} N/A")
        continue
    max_new   = np.max(res_new[mask])
    max_naive = np.max(res_naive[mask])
    ok = "PASS" if max_new <= threshold else "FAIL"
    print(f"{name:<18} {t_range:<16} {max_new:<15.3f} {max_naive:<16.3f} {ok}")

print()
print(f"Improvement at mushy peak: {max(res_naive[mushy_mask]) - max(res_new[mushy_mask]):.1f}% "
      f"({max(res_naive[mushy_mask]):.1f}% → {max(res_new[mushy_mask]):.1f}%)")
