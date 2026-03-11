#!/usr/bin/env python3
"""
Plot thermocapillary cavity results: isotherms, streamlines, velocity profiles.
Benchmark: Zebib, Homsy & Meiburg (1985), Ma=1000, Pr=1.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

DIR = Path(__file__).parent
BUILD_DIR = DIR.parent.parent / 'build'

def load_field(path):
    """Load 2D field CSV into structured arrays."""
    data = np.genfromtxt(path, delimiter=',', names=True)
    nx = int(data['ix'].max()) + 1
    ny = int(data['iy'].max()) + 1
    shape = (ny, nx)
    T  = data['T'].reshape(shape)
    ux = data['ux'].reshape(shape)
    uy = data['uy'].reshape(shape)
    x  = data['x'].reshape(shape)
    y  = data['y'].reshape(shape)
    return x, y, T, ux, uy, nx, ny

def compute_stream_function(ux, uy, nx, ny):
    """Compute stream function ψ by integrating ux along y.
       ψ(x,0) = 0,  ∂ψ/∂y = ux  →  ψ(x,j) = Σ_{k=0}^{j-1} ux(x,k)·Δy
    """
    dy = 1.0 / (ny - 1)
    psi = np.zeros_like(ux)
    for j in range(1, ny):
        psi[j, :] = psi[j-1, :] + 0.5 * (ux[j, :] + ux[j-1, :]) * dy
    return psi

def load_profiles(path):
    """Load midline profiles CSV."""
    vmid_y, vmid_u = [], []
    hmid_x, hmid_v = [], []
    with open(path) as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split(',')
            if parts[0] == 'vmid':
                vmid_y.append(float(parts[1]))
                vmid_u.append(float(parts[2]))
            elif parts[0] == 'hmid':
                hmid_x.append(float(parts[1]))
                hmid_v.append(float(parts[2]))
    return np.array(vmid_y), np.array(vmid_u), np.array(hmid_x), np.array(hmid_v)

def load_convergence(path):
    """Load convergence history."""
    data = np.genfromtxt(path, delimiter=',', names=True)
    return data

def main():
    # Look in build directory first, then script directory
    field_path = BUILD_DIR / 'marangoni_cavity_field.csv'
    if not field_path.exists():
        field_path = DIR / 'marangoni_cavity_field.csv'
    prof_path  = field_path.parent / 'marangoni_cavity_profiles.csv'
    conv_path  = field_path.parent / 'marangoni_cavity_convergence.csv'

    if not field_path.exists():
        print(f"Error: {field_path} not found. Run viz_marangoni_cavity first.")
        sys.exit(1)

    x, y, T, ux, uy, nx, ny = load_field(field_path)
    psi = compute_stream_function(ux, uy, nx, ny)
    vy, vu, hx, hv = load_profiles(prof_path)

    # Non-dimensionalise velocities by U_ref
    # Auto-detect: estimate U_ref from max surface velocity
    U_REF = max(abs(ux[ny-1, :]).max(), 1e-10)
    vu_nd = vu / U_REF
    hv_nd = hv / U_REF

    # ---- 4-panel figure ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Isotherms
    ax = axes[0, 0]
    levels_T = np.linspace(0, 1, 21)
    cs = ax.contour(x, y, T, levels=levels_T, cmap='RdYlBu_r', linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=7, fmt='%.2f')
    ax.set_xlabel('x/H')
    ax.set_ylabel('y/H')
    ax.set_title('(a) Isotherms')
    ax.set_aspect('equal')

    # (b) Streamlines
    ax = axes[0, 1]
    psi_abs = np.abs(psi).max()
    if psi_abs > 0:
        levels_psi = np.linspace(psi.min(), psi.max(), 25)
        ax.contour(x, y, psi, levels=levels_psi, colors='k', linewidths=0.6)
    ax.set_xlabel('x/H')
    ax.set_ylabel('y/H')
    ax.set_title(f'(b) Streamlines  |ψ|_max = {psi_abs:.4f}')
    ax.set_aspect('equal')

    # (c) u(y) on vertical midline (x = 0.5)
    ax = axes[1, 0]
    ax.plot(vu_nd, vy, 'b-', linewidth=1.5, label='LBM')
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('u / U_ref')
    ax.set_ylabel('y/H')
    ax.set_title('(c) u(y) at x = 0.5')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) v(x) on horizontal midline (y = 0.5)
    ax = axes[1, 1]
    ax.plot(hx, hv_nd, 'r-', linewidth=1.5, label='LBM')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('x/H')
    ax.set_ylabel('v / U_ref')
    ax.set_title('(d) v(x) at y = 0.5')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Thermocapillary Cavity  Ma={1000}, Pr={1}  ({nx}×{ny})',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()

    out = DIR / 'marangoni_cavity.png'
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")

    # ---- Convergence plot ----
    if conv_path.exists():
        conv = load_convergence(conv_path)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.semilogy(conv['step'], conv['du_rel'], label='Δu relative')
        ax2.semilogy(conv['step'], conv['dT_rel'], label='ΔT absolute')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Change per check interval')
        ax2.set_title('Convergence History')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        out2 = DIR / 'marangoni_cavity_convergence.png'
        fig2.savefig(out2, dpi=150)
        print(f"Saved {out2}")

    # ---- Print summary ----
    print(f"\n=== Summary ===")
    print(f"|ψ|_max = {psi_abs:.6f}")

    # Nusselt from temperature gradient at hot wall
    dTdx_hot = (T[:, 1] - T[:, 0]) * (nx - 1)
    Nu_hot = -np.mean(dTdx_hot)
    print(f"Nu (hot wall) = {Nu_hot:.3f}")

    # Max velocities on midlines
    iu_max = np.argmax(np.abs(vu_nd))
    iv_max = np.argmax(np.abs(hv_nd))
    print(f"u_max/U_ref on x=0.5: {vu_nd[iu_max]:.4f} at y={vy[iu_max]:.3f}")
    print(f"v_max/U_ref on y=0.5: {hv_nd[iv_max]:.4f} at x={hx[iv_max]:.3f}")

    # Stream function extremum location (vortex center)
    idx_psi = np.unravel_index(np.argmin(psi) if psi.min() < -psi.max()
                               else np.argmax(psi), psi.shape)
    print(f"Vortex center ≈ (x={x[idx_psi]:.3f}, y={y[idx_psi]:.3f})")

    plt.show()

if __name__ == '__main__':
    main()
