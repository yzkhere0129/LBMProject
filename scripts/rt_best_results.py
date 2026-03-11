#!/usr/bin/env python3
"""
Generate comprehensive RT benchmark visualization from the best run.
CSV: amplitude=5, variable viscosity (constant mu), 17054 steps.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, struct

# ── Parameters ──────────────────────────────────────────────────────
CSV_PATH = "/home/yzk/LBMProject/build/output_rt_benchmark/rt_benchmark_time_series.csv"
VTK_DIR  = "/home/yzk/LBMProject/build/output_rt_benchmark"
OUT_DIR  = VTK_DIR
NX, NY, NZ = 256, 1024, 4

At = 0.5
g  = 1e-5
k  = 2*np.pi/NX
nu = 0.050596
rho_H, rho_L = 3.0, 1.0
rho_avg = 0.5*(rho_H + rho_L)
mu = nu * rho_avg
nu_H = mu/rho_H
nu_L = mu/rho_L
nu_avg = 0.5*(nu_H + nu_L)

sigma = 5e-3  # surface tension (lattice units)
k3 = k**3
sigma_term = sigma * k3 / (rho_H + rho_L)

gamma_inv = np.sqrt(At*g*k)
gamma_vis = -nu_avg*k**2 + np.sqrt(nu_H*nu_L*k**4 + At*g*k - sigma_term)
gamma_simple = -nu*k**2 + np.sqrt(nu**2*k**4 + At*g*k - sigma_term)
gamma_vis_nosigma = -nu_avg*k**2 + np.sqrt(nu_H*nu_L*k**4 + At*g*k)

# ── Load CSV ────────────────────────────────────────────────────────
data = np.genfromtxt(CSV_PATH, delimiter=',', skip_header=1)
steps   = data[:,0].astype(int)
t_efold = data[:,1]
h_avg   = data[:,2]
h1      = data[:,3]
h2      = data[:,4]
ln_h    = data[:,5]
mass_e  = data[:,6]
u_max   = data[:,7]

# h0 from step 0 h_avg
h0 = h_avg[0]
# recompute ln(h/h0) consistently
ln_h_h0 = np.log(h_avg / h0)

# ── Read VTK (ASCII structured points) ──────────────────────────────
def read_vtk_fill(path):
    """Read fill_level from ASCII VTK structured points file."""
    with open(path, 'r') as f:
        lines = f.readlines()
    # find SCALARS fill_level
    idx = 0
    for i, line in enumerate(lines):
        if line.startswith("SCALARS fill_level"):
            idx = i + 2  # skip LOOKUP_TABLE line
            break
    vals = []
    for i in range(idx, idx + NX*NY*NZ):
        vals.append(float(lines[i].strip()))
    return np.array(vals).reshape((NZ, NY, NX))

# ── Figure 1: Interface Evolution (6 panels) ───────────────────────
fig1 = plt.figure(figsize=(15, 36))
fig1.suptitle(r'Rayleigh-Taylor Instability: Interface Evolution'
              f'\n$At={At}$, $\\nu={nu:.4f}$, variable $\\nu(f)=\\mu/\\rho(f)$, '
              f'$k\\eta_0={k*5:.3f}$',
              fontsize=14, fontweight='bold')

# Pick 6 representative steps from available VTKs
vtk_candidates = sorted([
    int(f.split('step')[1].split('.')[0])
    for f in os.listdir(VTK_DIR) if f.endswith('.vtk')
])
# Filter to steps <= 36000 (current run)
vtk_steps_all = [s for s in vtk_candidates if s <= 36000]
# Pick 6 evenly spaced
if len(vtk_steps_all) >= 6:
    indices = np.linspace(0, len(vtk_steps_all)-1, 6, dtype=int)
    vtk_steps = [vtk_steps_all[i] for i in indices]
else:
    vtk_steps = vtk_steps_all

kz_mid = NZ // 2
for idx, step in enumerate(vtk_steps):
    ax = fig1.add_subplot(2, 3, idx+1)
    vtk_path = os.path.join(VTK_DIR, f"rt_benchmark_step{step:06d}.vtk")
    if os.path.exists(vtk_path):
        fill3d = read_vtk_fill(vtk_path)
        fill2d = fill3d[kz_mid, :, :]  # y x x
        im = ax.imshow(fill2d, origin='lower', aspect='equal',
                       extent=[0, NX, 0, NY], cmap='RdBu_r',
                       vmin=0, vmax=1)
        # f=0.5 contour
        ax.contour(np.linspace(0, NX, NX), np.linspace(0, NY, NY),
                   fill2d, levels=[0.5], colors='lime', linewidths=1.5)
    t_star = step * gamma_vis
    ax.set_title(f'Step {step:,}  ($t/t_{{efold}}$={step/( 1/gamma_vis):.1f})',
                 fontsize=10)
    ax.set_xlabel('x [cells]')
    ax.set_ylabel('y [cells]')

fig1.tight_layout(rect=[0, 0, 0.95, 0.93])
cbar_ax = fig1.add_axes([0.96, 0.15, 0.015, 0.7])
fig1.colorbar(im, cax=cbar_ax, label='Fill Level $f$')
fig1.savefig(os.path.join(OUT_DIR, 'rt_best_evolution.png'), dpi=150, bbox_inches='tight')
print(f"Saved: rt_best_evolution.png")

# ── Figure 2: Growth Rate Analysis (3 panels) ──────────────────────
fig2, axes = plt.subplots(3, 1, figsize=(12, 14),
                          gridspec_kw={'height_ratios': [3, 2, 1.5]})
fig2.suptitle(r'RT Benchmark: Growth Rate Analysis'
              f'\n$At={At}$, variable $\\mu$, $k\\eta_0={k*5:.3f}$',
              fontsize=14, fontweight='bold')

# Panel A: ln(h/h0) vs time with theory
ax = axes[0]
ax.plot(steps, ln_h_h0, 'b-', lw=2, label=r'LBM simulation $\ln(\bar{h}/h_0)$')

# Theory lines
t_theory = np.linspace(0, steps[-1], 500)
ax.plot(t_theory, gamma_vis * t_theory, 'r--', lw=1.5, alpha=0.8,
        label=rf'$\gamma_{{visc}}={gamma_vis:.4e}$ (Chandrasekhar, equal-$\mu$)')
ax.plot(t_theory, gamma_inv * t_theory, 'g:', lw=1.2, alpha=0.6,
        label=rf'$\gamma_{{inv}}={gamma_inv:.4e}$ (inviscid)')

# Fit window
h_lo, h_hi = 20.0, 60.0  # early growth window where linear theory applies best
mask = (h_avg >= h_lo) & (h_avg <= h_hi)
fit_steps = steps[mask].astype(float)
fit_lnh = ln_h_h0[mask]

if len(fit_steps) >= 4:
    coeffs = np.polyfit(fit_steps, fit_lnh, 1)
    gamma_fit = coeffs[0]
    fit_line = np.polyval(coeffs, fit_steps)
    ax.fill_between(fit_steps, fit_lnh.min()-0.2, fit_lnh.max()+0.2,
                    alpha=0.15, color='yellow', label=f'Fit window: $h \\in [{h_lo}, {h_hi}]$')
    ax.plot(fit_steps, fit_line, 'm-', lw=2.5,
            label=rf'Fit: $\gamma_{{fit}}={gamma_fit:.4e}$ ($\gamma/\gamma_{{vis}}={gamma_fit/gamma_vis:.3f}$)')

    # R²
    ss_res = np.sum((fit_lnh - fit_line)**2)
    ss_tot = np.sum((fit_lnh - fit_lnh.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 1.0

    ratio = gamma_fit / gamma_vis
    ax.text(0.02, 0.97,
            f'$\\gamma_{{fit}}/\\gamma_{{vis}} = {ratio:.3f}$\n'
            f'$R^2 = {r2:.4f}$\n'
            f'Error: {(ratio-1)*100:+.1f}%',
            transform=ax.transAxes, fontsize=12, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_ylabel(r'$\ln(\bar{h}/h_0)$', fontsize=12)
ax.set_xlabel('Step', fontsize=12)
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, steps[-1])

# Panel B: h1 (bubble) and h2 (spike) evolution
ax = axes[1]
ax.plot(steps, h1, 'r-', lw=1.5, label=r'$h_1$ (spike, heavy $\rightarrow$ light)')
ax.plot(steps, h2, 'b-', lw=1.5, label=r'$h_2$ (bubble, light $\rightarrow$ heavy)')
ax.plot(steps, h_avg, 'k--', lw=2, label=r'$\bar{h}$ (column average)')

# Theory envelope
t_arr = np.arange(0, steps[-1]+1, 100)
h_theory = h0 * np.exp(gamma_vis * t_arr)
h_theory_clip = np.clip(h_theory, 0, NY*0.4)
ax.plot(t_arr, h_theory_clip, 'g:', lw=1.2, alpha=0.7,
        label=rf'$h_0 \exp(\gamma_{{vis}} t)$')

ax.set_ylabel('Amplitude [cells]', fontsize=12)
ax.set_xlabel('Step', fontsize=12)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, steps[-1])
ax.set_ylim(0, max(h1.max(), h2.max())*1.1)

# Panel C: Mass conservation
ax = axes[2]
ax.fill_between(steps, 0, mass_e*100 if mass_e.max() < 0.01 else mass_e,
                alpha=0.3, color='purple')
ax.plot(steps, mass_e, 'purple', lw=1.5)
ax.axhline(0.5, color='red', ls='--', lw=1, alpha=0.5, label='0.5% threshold')
ax.set_ylabel('Mass Error [%]', fontsize=12)
ax.set_xlabel('Step', fontsize=12)
ax.set_xlim(0, steps[-1])
ax.set_ylim(0, max(mass_e.max()*1.5, 0.001))
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.text(0.98, 0.95, f'Final: {mass_e[-1]:.4f}%',
        transform=ax.transAxes, fontsize=11, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='lavender'))

fig2.tight_layout(rect=[0, 0, 1, 0.95])
fig2.savefig(os.path.join(OUT_DIR, 'rt_best_growth_analysis.png'), dpi=150, bbox_inches='tight')
print(f"Saved: rt_best_growth_analysis.png")

# ── Figure 3: Instantaneous growth rate ─────────────────────────────
fig3, ax = plt.subplots(figsize=(12, 5))
fig3.suptitle('Instantaneous Growth Rate vs. Step', fontsize=13, fontweight='bold')

# Compute instantaneous gamma from finite differences
gamma_inst = np.zeros_like(ln_h_h0)
for i in range(1, len(gamma_inst)):
    dt_s = float(steps[i] - steps[i-1])
    if dt_s > 0 and ln_h_h0[i] > -50 and ln_h_h0[i-1] > -50:
        gamma_inst[i] = (ln_h_h0[i] - ln_h_h0[i-1]) / dt_s

# Smooth with moving average
window = 5
gamma_smooth = np.convolve(gamma_inst, np.ones(window)/window, mode='same')

ax.plot(steps[1:], gamma_inst[1:]*1e4, 'b-', alpha=0.3, lw=0.8, label='Raw')
ax.plot(steps[1:], gamma_smooth[1:]*1e4, 'b-', lw=2, label=f'Smoothed (window={window})')
ax.axhline(gamma_vis*1e4, color='r', ls='--', lw=1.5,
           label=rf'$\gamma_{{vis}}={gamma_vis:.3e}$ (Chandrasekhar)')
ax.axhline(gamma_inv*1e4, color='g', ls=':', lw=1.2,
           label=rf'$\gamma_{{inv}}={gamma_inv:.3e}$')

if len(fit_steps) >= 4:
    ax.axhline(gamma_fit*1e4, color='m', ls='-', lw=1.5, alpha=0.7,
               label=rf'$\gamma_{{fit}}={gamma_fit:.3e}$')

# Mark the fit window
ax.axvspan(fit_steps[0], fit_steps[-1], alpha=0.1, color='yellow')

ax.set_xlabel('Step', fontsize=12)
ax.set_ylabel(r'$\gamma_{inst} \times 10^4$ [step$^{-1}$]', fontsize=12)
ax.set_xlim(0, steps[-1])
ax.set_ylim(0, gamma_inv*1e4*2)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)

fig3.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, 'rt_best_gamma_inst.png'), dpi=150, bbox_inches='tight')
print(f"Saved: rt_best_gamma_inst.png")

# ── Print summary ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  RT BENCHMARK BEST RESULTS SUMMARY")
print(f"{'='*60}")
print(f"  Config: amplitude=25.6, W=4, MC limiter, C_compress=0.10, variable mu, TRT, sigma={sigma}")
print(f"  Steps: {steps[-1]:,}")
print(f"  h_avg final: {h_avg[-1]:.1f} cells")
print(f"  Mass error: {mass_e[-1]:.4f}%")
print(f"")
print(f"  Theory (with sigma={sigma}):")
print(f"    gamma_inviscid  = {gamma_inv:.6e}")
print(f"    gamma_viscous   = {gamma_vis:.6e} (Chandrasekhar + sigma)")
print(f"    gamma_vis_nosig = {gamma_vis_nosigma:.6e} (Chandrasekhar, sigma=0)")
print(f"    gamma_simple    = {gamma_simple:.6e} (equal-nu + sigma)")
print(f"")
if len(fit_steps) >= 4:
    print(f"  Fit (h_avg in [{h_lo}, {h_hi}]):")
    print(f"    gamma_fit      = {gamma_fit:.6e}")
    print(f"    R2             = {r2:.6f}")
    print(f"    gamma/gamma_vis = {gamma_fit/gamma_vis:.4f} ({(gamma_fit/gamma_vis-1)*100:+.1f}%)")
    print(f"    gamma/gamma_inv = {gamma_fit/gamma_inv:.4f} ({(gamma_fit/gamma_inv-1)*100:+.1f}%)")
    print(f"    gamma/gamma_sim = {gamma_fit/gamma_simple:.4f} ({(gamma_fit/gamma_simple-1)*100:+.1f}%)")
print(f"{'='*60}")
