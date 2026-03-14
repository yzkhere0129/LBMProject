#!/usr/bin/env python3
"""
plot_marangoni_validation.py — Marangoni fix visual evidence

Plot A: 1D z-profile at laser spot edge: fill, T, and Marangoni force proxy
Plot B: 2D x-z midplane velocity vector field showing Marangoni vortices

Usage:
  python3 plot_marangoni_validation.py [step]
  Default step: 40
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import os
import sys

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_a_zprofile(step):
    """Plot A: 1D z-profile showing fill, T, and Marangoni force region."""
    path = os.path.join(DATA_DIR, f"marangoni_zprofile_step{step:02d}.csv")
    if not os.path.exists(path):
        print(f"  SKIP Plot A: {path} not found")
        return None

    data = np.genfromtxt(path, delimiter=",", names=True)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    z = data["z_um"]

    # Panel 1: Fill level + liquid fraction
    ax = axes[0]
    ax.plot(z, data["fill"], "b-", linewidth=2, label="Fill level $\\phi$")
    ax.plot(z, data["liquid_frac"], "r--", linewidth=2, label="Liquid frac $f_\\ell$")
    ax.set_xlabel("$z$ (μm)", fontsize=13)
    ax.set_ylabel("Field value", fontsize=13)
    ax.set_title("Interface Structure", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)

    # Shade the interface band (0.01 < fill < 0.99)
    fill = data["fill"]
    z_interface = z[(fill > 0.01) & (fill < 0.99)]
    if len(z_interface) > 0:
        ax.axvspan(z_interface.min(), z_interface.max(),
                   alpha=0.15, color="orange", label="Interface band")

    # Panel 2: Temperature
    ax = axes[1]
    ax.plot(z, data["temperature"], "k-", linewidth=2)
    ax.set_xlabel("$z$ (μm)", fontsize=13)
    ax.set_ylabel("Temperature (K)", fontsize=13)
    ax.set_title("Temperature Profile", fontsize=13)
    ax.grid(True, alpha=0.3)

    # Mark T_solidus and T_liquidus for Steel
    T_solidus = 1658.0
    T_liquidus = 1723.0
    for T_mark, label in [(T_solidus, "$T_s$"), (T_liquidus, "$T_\\ell$")]:
        ax.axhline(y=T_mark, color="gray", linestyle="--", alpha=0.5)
        ax.text(z.max() * 0.02, T_mark + 30, label, fontsize=10, color="gray")

    # Panel 3: Marangoni force proxy = f · |∇f| · |∇_s T|
    # Compute |∇f| from central differences
    dz = z[1] - z[0] if len(z) > 1 else 1.0
    grad_f = np.gradient(fill, dz * 1e-6)  # [1/m]
    grad_T = np.gradient(data["temperature"], dz * 1e-6)  # [K/m]

    # Tangential ∇T: for flat interface with normal in z, ∇_s T ≈ ∇T_xy
    # Since we only have z-profile, approximate |∇_s T| from horizontal components
    # Actually ux gives us the Marangoni-driven velocity which IS the proxy
    vmag = data["vmag"]

    # Marangoni force proxy: f · |∇f| (dimensionless weighting)
    marangoni_weight = np.abs(fill * grad_f)

    ax = axes[2]
    ax2 = ax.twinx()

    ax.plot(z, marangoni_weight / (marangoni_weight.max() + 1e-30),
            "g-", linewidth=2, label="$\\phi \\cdot |\\nabla\\phi|$ (CSF weight)")
    ax.set_ylabel("Normalized CSF weight", color="g", fontsize=13)
    ax.tick_params(axis="y", labelcolor="g")
    ax.set_ylim(-0.05, 1.1)

    vmag_safe = np.where(vmag > 0, vmag, 1e-25)
    ax2.semilogy(z, vmag_safe, "r-", linewidth=2, label="|u| (velocity)")
    ax2.set_ylabel("|u| (lattice units)", color="r", fontsize=13)
    ax2.tick_params(axis="y", labelcolor="r")
    ax2.set_ylim(1e-14, 1e-1)

    ax.set_xlabel("$z$ (μm)", fontsize=13)
    ax.set_title("Marangoni Force Distribution", fontsize=13)

    # Verify F_M = 0 in gas
    gas_mask = fill < 0.01
    if np.any(gas_mask):
        gas_vmag = vmag[gas_mask]
        gas_weight = marangoni_weight[gas_mask]
        ax.text(0.02, 0.95,
                f"Gas cells: max weight = {gas_weight.max():.2e}\n"
                f"Gas cells: max |u| = {gas_vmag.max():.2e}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=9)

    fig.suptitle(
        f"Plot A: 1D z-Profile at Laser Spot Edge (step {step})\n"
        f"CSF smearing: $F_M \\propto \\phi \\cdot |\\nabla\\phi|$ — zero in gas, smooth across interface",
        fontsize=13)
    fig.tight_layout()
    return fig


def plot_b_vortex(step):
    """Plot B: 2D x-z midplane velocity vector field."""
    path = os.path.join(DATA_DIR, f"marangoni_xz_midplane_step{step:02d}.csv")
    if not os.path.exists(path):
        print(f"  SKIP Plot B: {path} not found")
        return None

    data = np.genfromtxt(path, delimiter=",", names=True)

    NX = int(data["i"].max()) + 1
    NZ = int(data["k"].max()) + 1

    x = data["x_um"].reshape(NZ, NX)
    z = data["z_um"].reshape(NZ, NX)
    fill = data["fill"].reshape(NZ, NX)
    T = data["temperature"].reshape(NZ, NX)
    fl = data["liquid_frac"].reshape(NZ, NX)
    ux = data["ux"].reshape(NZ, NX)
    uz = data["uz"].reshape(NZ, NX)
    vmag = data["vmag"].reshape(NZ, NX)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Temperature field with velocity vectors
    ax = axes[0]
    im = ax.pcolormesh(x, z, T, cmap="hot", shading="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Temperature (K)")

    # Velocity vectors (subsample for readability)
    skip = 2
    mask = vmag > 1e-8  # Only show cells with meaningful velocity
    xs = x[::skip, ::skip]
    zs = z[::skip, ::skip]
    uxs = ux[::skip, ::skip]
    uzs = uz[::skip, ::skip]
    ms = mask[::skip, ::skip]

    if np.any(ms):
        scale = vmag[mask].max() if np.any(mask) else 1e-6
        ax.quiver(xs[ms], zs[ms], uxs[ms], uzs[ms],
                  color="cyan", alpha=0.8, scale=scale * 20,
                  width=0.003, headwidth=3)

    # Draw interface contour
    ax.contour(x, z, fill, levels=[0.5], colors=["lime"], linewidths=2)
    ax.set_xlabel("$x$ (μm)", fontsize=12)
    ax.set_ylabel("$z$ (μm)", fontsize=12)
    ax.set_title("Temperature + Velocity Vectors", fontsize=12)
    ax.set_aspect("equal")

    # Zoom to top 30% (melt pool region)
    z_min = z.max() * 0.6
    ax.set_ylim(z_min, z.max() * 1.02)

    # Panel 2: Velocity magnitude (log scale)
    ax = axes[1]
    vmag_safe = np.where(vmag > 1e-15, vmag, 1e-15)
    im = ax.pcolormesh(x, z, np.log10(vmag_safe), cmap="viridis",
                       shading="auto", vmin=-10, vmax=-2)
    plt.colorbar(im, ax=ax, shrink=0.8, label="log₁₀(|u|) [LU]")
    ax.contour(x, z, fill, levels=[0.5], colors=["lime"], linewidths=2)
    ax.set_xlabel("$x$ (μm)", fontsize=12)
    ax.set_ylabel("$z$ (μm)", fontsize=12)
    ax.set_title("Velocity Magnitude (log scale)", fontsize=12)
    ax.set_aspect("equal")
    ax.set_ylim(z_min, z.max() * 1.02)

    # Panel 3: Streamlines for vortex structure
    ax = axes[2]
    # Background: liquid fraction
    im = ax.pcolormesh(x, z, fl, cmap="Blues", shading="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Liquid fraction")

    # Streamlines
    x_1d = x[0, :]  # unique x values
    z_1d = z[:, 0]  # unique z values
    if np.any(vmag > 1e-10):
        try:
            speed = np.sqrt(ux**2 + uz**2)
            lw = 2 * speed / (speed.max() + 1e-15)
            ax.streamplot(x_1d, z_1d, ux, uz,
                          color="black", density=1.5, linewidth=lw,
                          arrowsize=1.2, broken_streamlines=False)
        except Exception as e:
            print(f"  Streamplot failed: {e}")
            ax.quiver(xs, zs, uxs, uzs, color="black", alpha=0.7,
                      scale=scale * 20 if 'scale' in dir() else 1)

    ax.contour(x, z, fill, levels=[0.5], colors=["lime"], linewidths=2)
    ax.set_xlabel("$x$ (μm)", fontsize=12)
    ax.set_ylabel("$z$ (μm)", fontsize=12)
    ax.set_title("Streamlines (Marangoni Vortices)", fontsize=12)
    ax.set_aspect("equal")
    ax.set_ylim(z_min, z.max() * 1.02)

    # Mark laser center
    laser_x = 20 * 3.75  # μm
    for a in axes:
        a.axvline(x=laser_x, color="yellow", linestyle="--", alpha=0.5, linewidth=1)

    fig.suptitle(
        f"Plot B: 2D X-Z Midplane at Step {step}\n"
        f"Marangoni convection: outward surface flow from hot center → "
        f"counter-rotating vortices",
        fontsize=13)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    step = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    print(f"=== Marangoni Fix Validation Plots (step {step}) ===")

    figs = {}
    print("Plot A: 1D z-profile...")
    figs["zprofile"] = plot_a_zprofile(step)
    print("Plot B: 2D vortex structure...")
    figs["vortex"] = plot_b_vortex(step)

    for name, fig in figs.items():
        if fig is not None:
            out = os.path.join(DATA_DIR, f"marangoni_{name}_step{step:02d}.png")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  Saved: {out}")

    # Also try step 100 if available
    if step == 200:
        print("\nAlso generating step 100 plots...")
        figs100 = {}
        figs100["zprofile"] = plot_a_zprofile(100)
        figs100["vortex"] = plot_b_vortex(100)
        for name, fig in figs100.items():
            if fig is not None:
                out = os.path.join(DATA_DIR, f"marangoni_{name}_step100.png")
                fig.savefig(out, dpi=150, bbox_inches="tight")
                print(f"  Saved: {out}")

    print("\n=== Done ===")
