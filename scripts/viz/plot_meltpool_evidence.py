#!/usr/bin/env python3
"""
plot_meltpool_evidence.py — Hard visual evidence for Marangoni fix

Plot B: 2D X-Z midplane at laser center (y=40)
  - Panel 1: Temperature + velocity vectors (liquid only)
  - Panel 2: Velocity magnitude (log scale, liquid only)
  - Panel 3: Streamlines masked to fl > 0.1 (no solid/mushy noise)

Masks out solid/mushy zone (fl < 0.1) so only liquid flow is visible.

Usage:
  python3 plot_meltpool_evidence.py [step]
  Default step: 1000
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import sys

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_midplane(step):
    """Load 2D x-z midplane CSV."""
    path = os.path.join(DATA_DIR, f"marangoni_xz_midplane_step{step:04d}.csv")
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return None
    return np.genfromtxt(path, delimiter=",", names=True)


def plot_meltpool_evidence(step):
    data = load_midplane(step)
    if data is None:
        return None

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

    # Masks
    liquid_mask = fl > 0.1       # Liquid region (for streamlines)
    interface_mask = (fill > 0.01) & (fill < 0.99)  # Gas-liquid interface
    solid_mask = fl < 0.1        # Solid/mushy (to grey out)

    # Melt pool boundary: fl = 0.5 contour
    # Interface: fill = 0.5 contour

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # =========================================================
    # Panel 1: Temperature field with velocity vectors (liquid only)
    # =========================================================
    ax = axes[0]

    # Temperature with solid greyed out
    T_display = np.ma.masked_where(fill < 0.01, T)  # mask gas
    im = ax.pcolormesh(x, z, T_display, cmap="hot", shading="auto",
                       vmin=300, vmax=3000)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Temperature (K)")

    # Grey overlay for solid
    solid_overlay = np.ma.masked_where(~solid_mask, np.ones_like(fl))
    ax.pcolormesh(x, z, solid_overlay, cmap="Greys", shading="auto",
                  alpha=0.3, vmin=0, vmax=2)

    # Velocity vectors in liquid only (subsample)
    skip = 2
    for arr in [x, z, ux, uz, vmag, liquid_mask]:
        pass
    xs = x[::skip, ::skip]
    zs = z[::skip, ::skip]
    uxs = ux[::skip, ::skip]
    uzs = uz[::skip, ::skip]
    ms = liquid_mask[::skip, ::skip] & (vmag[::skip, ::skip] > 1e-8)

    if np.any(ms):
        vmax = vmag[liquid_mask].max() if np.any(liquid_mask & (vmag > 0)) else 1e-6
        ax.quiver(xs[ms], zs[ms], uxs[ms], uzs[ms],
                  color="cyan", alpha=0.8, scale=vmax * 15,
                  width=0.003, headwidth=3)

    # Draw contours
    # Melt pool boundary (fl = 0.5)
    try:
        ax.contour(x, z, fl, levels=[0.5], colors=["lime"], linewidths=2)
    except Exception:
        pass
    # Free surface (fill = min + 0.1*(max-min)) for barely-recessed surfaces
    fill_thresh = fill.min() + 0.1 * (fill.max() - fill.min())
    if fill_thresh < fill.max():
        try:
            ax.contour(x, z, fill, levels=[fill_thresh], colors=["white"],
                       linewidths=1.5, linestyles="--")
        except Exception:
            pass

    ax.set_xlabel("$x$ (μm)", fontsize=12)
    ax.set_ylabel("$z$ (μm)", fontsize=12)
    ax.set_title("Temperature + Velocity (liquid only)", fontsize=12)
    ax.set_aspect("equal")

    # Zoom to melt pool region (top 40%)
    z_min = z.max() * 0.55
    ax.set_ylim(z_min, z.max() * 1.02)

    # =========================================================
    # Panel 2: Velocity magnitude (log scale, masked)
    # =========================================================
    ax = axes[1]

    vmag_masked = np.where(liquid_mask & (vmag > 1e-15), vmag, np.nan)
    im = ax.pcolormesh(x, z, np.log10(np.where(np.isnan(vmag_masked), 1e-15, vmag_masked)),
                       cmap="viridis", shading="auto", vmin=-8, vmax=-1)
    plt.colorbar(im, ax=ax, shrink=0.8, label="log₁₀(|u|) [LU]")

    # Grey out solid
    ax.pcolormesh(x, z, solid_overlay, cmap="Greys", shading="auto",
                  alpha=0.4, vmin=0, vmax=2)

    try:
        ax.contour(x, z, fill, levels=[0.5], colors=["lime"], linewidths=2)
    except Exception:
        pass
    try:
        ax.contour(x, z, fl, levels=[0.5], colors=["white"], linewidths=1.5,
                   linestyles="--")
    except Exception:
        pass

    ax.set_xlabel("$x$ (μm)", fontsize=12)
    ax.set_ylabel("$z$ (μm)", fontsize=12)
    ax.set_title("Velocity Magnitude (liquid only, log)", fontsize=12)
    ax.set_aspect("equal")
    ax.set_ylim(z_min, z.max() * 1.02)

    # =========================================================
    # Panel 3: Streamlines MASKED to liquid (fl > 0.1)
    # =========================================================
    ax = axes[2]

    # Background: liquid fraction
    im = ax.pcolormesh(x, z, fl, cmap="Blues", shading="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Liquid fraction $f_\\ell$")

    # Mask velocity in solid/mushy for streamplot
    ux_liquid = np.where(liquid_mask, ux, 0.0)
    uz_liquid = np.where(liquid_mask, uz, 0.0)

    x_1d = x[0, :]
    z_1d = z[:, 0]

    vmag_liquid = np.sqrt(ux_liquid**2 + uz_liquid**2)
    if np.any(vmag_liquid > 1e-10):
        try:
            speed = vmag_liquid
            speed_max = speed.max()
            lw = np.where(speed > 1e-12, 2.0 * speed / (speed_max + 1e-15), 0)
            ax.streamplot(x_1d, z_1d, ux_liquid, uz_liquid,
                          color="black", density=1.8, linewidth=lw,
                          arrowsize=1.2, broken_streamlines=True)
        except Exception as e:
            print(f"  Streamplot failed: {e}")
            # Fallback to quiver
            skip = 2
            ms2 = liquid_mask[::skip, ::skip] & (vmag[::skip, ::skip] > 1e-8)
            if np.any(ms2):
                ax.quiver(xs[ms2], zs[ms2], uxs[ms2], uzs[ms2],
                          color="black", alpha=0.7,
                          scale=vmag[liquid_mask].max() * 15 if np.any(liquid_mask) else 1)

    try:
        ax.contour(x, z, fl, levels=[0.5], colors=["lime"], linewidths=2)
    except Exception:
        pass

    ax.set_xlabel("$x$ (μm)", fontsize=12)
    ax.set_ylabel("$z$ (μm)", fontsize=12)
    ax.set_title("Streamlines (masked: $f_\\ell > 0.1$ only)", fontsize=12)
    ax.set_aspect("equal")
    ax.set_ylim(z_min, z.max() * 1.02)

    # Mark laser center
    laser_x = 20 * 3.75  # μm (i=20)
    for a in axes:
        a.axvline(x=laser_x, color="yellow", linestyle="--", alpha=0.4, linewidth=1)

    t_us = step * 75e-3  # step * dt(75ns) in μs
    fig.suptitle(
        f"Melt Pool Evidence — Step {step} (t = {t_us:.1f} μs)\n"
        f"Green = free surface (fill=0.5), White dashed = melt front ($f_\\ell$=0.5), "
        f"Grey = solid",
        fontsize=13)
    fig.tight_layout()

    # Print melt pool metrics — find deepest liquid cell at center column
    mid_i = NX // 2
    min_liquid_k = NZ
    for k in range(NZ):
        if fl[k, mid_i] > 0.5:
            min_liquid_k = k
            break
    melt_depth_cells = (NZ - 1) - min_liquid_k if min_liquid_k < NZ else 0
    melt_depth_um = melt_depth_cells * 3.75
    max_v_liquid = vmag[liquid_mask].max() if np.any(liquid_mask) else 0
    max_T_val = T.max()
    n_liquid = np.sum(fl > 0.5)

    print(f"\n  === Melt Pool Metrics (step {step}) ===")
    print(f"  Melt depth at center: {melt_depth_um:.1f} μm ({melt_depth_cells} cells)")
    print(f"  Max velocity (liquid): {max_v_liquid:.6f} LU")
    print(f"  Max temperature: {max_T_val:.0f} K")
    print(f"  Liquid cells (fl > 0.5): {n_liquid}")
    print(f"  =======================================\n")

    return fig


if __name__ == "__main__":
    step = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    print(f"=== Melt Pool Evidence Plot (step {step}) ===")

    fig = plot_meltpool_evidence(step)
    if fig is not None:
        out = os.path.join(DATA_DIR, f"meltpool_evidence_step{step:04d}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
    else:
        print("  No data available. Run viz_marangoni_validation first.")

    print("\n=== Done ===")
