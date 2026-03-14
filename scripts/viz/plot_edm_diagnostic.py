#!/usr/bin/env python3
"""
EDM Diagnostic Plot — Honest assessment of Marangoni convection state

4-panel figure for each timestep:
  Panel A: Temperature + velocity vectors (liquid only, fl>0.5)
  Panel B: Velocity magnitude (linear scale, masked to liquid)
  Panel C: Marangoni force activity zone (fl > 0.999 AND |∇f| > threshold)
  Panel D: Velocity time series from CSV (if available)

Key question: Is the Marangoni force actually active where it matters (at the free surface)?
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import os
import sys

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_midplane(step):
    path = os.path.join(DATA_DIR, f"marangoni_xz_midplane_step{step:04d}.csv")
    if not os.path.exists(path):
        print(f"  Not found: {path}")
        return None
    return np.genfromtxt(path, delimiter=",", names=True)


def plot_diagnostic(step):
    data = load_midplane(step)
    if data is None:
        return None

    NX = int(data["i"].max()) + 1
    NZ = int(data["k"].max()) + 1
    dx_um = 3.75  # μm

    x = data["x_um"].reshape(NZ, NX)
    z = data["z_um"].reshape(NZ, NX)
    fill = data["fill"].reshape(NZ, NX)
    T = data["temperature"].reshape(NZ, NX)
    fl = data["liquid_frac"].reshape(NZ, NX)
    ux = data["ux"].reshape(NZ, NX)
    uz = data["uz"].reshape(NZ, NX)
    vmag = data["vmag"].reshape(NZ, NX)

    # Masks
    liquid = fl > 0.5
    full_liquid = fl > 0.9
    mushy = (fl > 0.01) & (fl < 0.999)
    fl_gate_active = fl > 0.1  # Where Marangoni fl_gate allows force (matches code)
    gas = fill < 0.01

    # Compute |∇f| (fill gradient magnitude) — proxy for interface
    grad_f = np.zeros_like(fill)
    grad_f[1:-1, 1:-1] = np.sqrt(
        ((fill[1:-1, 2:] - fill[1:-1, :-2]) / (2 * dx_um))**2 +
        ((fill[2:, 1:-1] - fill[:-2, 1:-1]) / (2 * dx_um))**2
    )

    # Marangoni active zone: fl_gate AND near interface
    marangoni_active = fl_gate_active & (grad_f > 0.001)
    # Where Marangoni SHOULD be active: near interface AND liquid
    marangoni_should_be = (grad_f > 0.001) & (fl > 0.1) & (fill > 0.01) & (fill < 0.99)

    # Zoom limits (top portion of domain, adjusted for gas layer)
    z_min = z.max() * 0.40
    z_max = z.max() * 1.02

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # =========================================================
    # Panel A: Temperature + velocity vectors (liquid only)
    # =========================================================
    ax = axes[0, 0]

    # Temperature field (mask gas)
    T_display = np.ma.masked_where(fill < 0.01, T)
    im = ax.pcolormesh(x, z, T_display, cmap="hot", shading="auto",
                       vmin=300, vmax=3000)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Temperature (K)")

    # Grey overlay for solid/mushy
    solid_overlay = np.ma.masked_where(fl > 0.5, np.ones_like(fl))
    ax.pcolormesh(x, z, solid_overlay, cmap="Greys", shading="auto",
                  alpha=0.3, vmin=0, vmax=2)

    # Velocity vectors — only in fully liquid cells (fl > 0.5)
    skip = 1  # every cell for clarity
    xs = x[::skip, ::skip]
    zs = z[::skip, ::skip]
    uxs = ux[::skip, ::skip]
    uzs = uz[::skip, ::skip]
    ms = liquid[::skip, ::skip] & (vmag[::skip, ::skip] > 1e-6)

    if np.any(ms):
        v_max_liq = vmag[liquid].max() if np.any(liquid) else 1e-6
        q = ax.quiver(xs[ms], zs[ms], uxs[ms], uzs[ms],
                      vmag[::skip, ::skip][ms],
                      cmap="cool", alpha=0.9,
                      scale=v_max_liq * 10, width=0.004, headwidth=3,
                      clim=[0, v_max_liq])

    # Melt pool boundary (fl = 0.5)
    try:
        ax.contour(x, z, fl, levels=[0.5], colors=["lime"], linewidths=2, linestyles="-")
    except Exception:
        pass
    try:
        ax.contour(x, z, fl, levels=[0.1], colors=["cyan"], linewidths=1.5, linestyles="--")
    except Exception:
        pass

    ax.set_xlabel("$x$ (μm)", fontsize=11)
    ax.set_ylabel("$z$ (μm)", fontsize=11)
    ax.set_title("A: Temperature + Velocity Vectors\n"
                 "Green = melt front (fl=0.5), Cyan = fl_gate (fl=0.1)", fontsize=11)
    ax.set_aspect("equal")
    ax.set_ylim(z_min, z_max)

    # =========================================================
    # Panel B: Velocity magnitude (LINEAR, masked to fl > 0.1)
    # =========================================================
    ax = axes[0, 1]

    vmag_display = np.ma.masked_where(~(fl > 0.1), vmag)
    v_max_val = vmag[fl > 0.1].max() if np.any(fl > 0.1) else 0.01
    im = ax.pcolormesh(x, z, vmag_display * 50.0,  # Convert to m/s
                       cmap="turbo", shading="auto",
                       vmin=0, vmax=max(v_max_val * 50.0, 0.5))
    plt.colorbar(im, ax=ax, shrink=0.8, label="Velocity magnitude (m/s)")

    # Grey overlay for solid
    ax.pcolormesh(x, z, solid_overlay, cmap="Greys", shading="auto",
                  alpha=0.4, vmin=0, vmax=2)

    try:
        ax.contour(x, z, fl, levels=[0.5], colors=["lime"], linewidths=2)
    except Exception:
        pass
    try:
        ax.contour(x, z, fl, levels=[0.1], colors=["cyan"], linewidths=1.5, linestyles="--")
    except Exception:
        pass

    # Annotate velocity stats
    v_liq = vmag[fl > 0.1]
    v_full = vmag[fl > 0.9]
    stats_text = (
        f"Max (fl>0.1): {v_liq.max()*50:.2f} m/s\n"
        f"Max (fl>0.9): {v_full.max()*50:.2f} m/s\n"
        f"Mean (fl>0.9): {v_full.mean()*50:.3f} m/s\n"
        f"Cells v>0.5 m/s: {np.sum(v_liq > 0.01)}"
    )
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel("$x$ (μm)", fontsize=11)
    ax.set_ylabel("$z$ (μm)", fontsize=11)
    ax.set_title("B: Velocity Magnitude (liquid only, linear scale)\n"
                 "Physical units [m/s]", fontsize=11)
    ax.set_aspect("equal")
    ax.set_ylim(z_min, z_max)

    # =========================================================
    # Panel C: Marangoni force activity diagnostic
    # =========================================================
    ax = axes[1, 0]

    # Color code:
    # Red = where Marangoni SHOULD act (interface + liquid)
    # Green = where fl_gate ALLOWS Marangoni (fl > 0.999)
    # Blue = overlap (where Marangoni IS active)
    zone_map = np.zeros_like(fl)
    zone_map[marangoni_should_be] = 1.0  # Red: should be active
    zone_map[fl_gate_active] = 2.0       # Green: gate allows
    zone_map[marangoni_active] = 3.0     # Blue: actually active
    zone_map[gas] = -1.0
    zone_map[fl < 0.01] = -1.0

    zone_display = np.ma.masked_where(zone_map < 0, zone_map)

    from matplotlib.colors import ListedColormap
    cmap_zone = ListedColormap(['lightgray', 'red', 'green', 'blue'])
    im = ax.pcolormesh(x, z, zone_display, cmap=cmap_zone, shading="auto",
                       vmin=0, vmax=3)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, ticks=[0.375, 1.125, 1.875, 2.625])
    cbar.ax.set_yticklabels(['Bulk', 'Should act\n(interface)', 'fl>0.1\n(gate open)', 'Both\n(ACTIVE)'],
                            fontsize=8)

    try:
        ax.contour(x, z, fl, levels=[0.5], colors=["lime"], linewidths=2)
    except Exception:
        pass
    try:
        ax.contour(x, z, fl, levels=[0.1], colors=["cyan"], linewidths=1.5, linestyles="--")
    except Exception:
        pass

    n_should = np.sum(marangoni_should_be)
    n_gate = np.sum(fl_gate_active)
    n_active = np.sum(marangoni_active)
    ax.set_xlabel("$x$ (μm)", fontsize=11)
    ax.set_ylabel("$z$ (μm)", fontsize=11)
    ax.set_title(f"C: Marangoni Force Activity Zone\n"
                 f"Should act: {n_should} cells, Gate open: {n_gate}, ACTIVE: {n_active}", fontsize=11)
    ax.set_aspect("equal")
    ax.set_ylim(z_min, z_max)

    # =========================================================
    # Panel D: Streamlines + fl contours
    # =========================================================
    ax = axes[1, 1]

    # Background: liquid fraction
    im = ax.pcolormesh(x, z, fl, cmap="Blues", shading="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Liquid fraction $f_\\ell$")

    # Mask velocity in solid/mushy for streamplot
    ux_liquid = np.where(fl > 0.1, ux, 0.0)
    uz_liquid = np.where(fl > 0.1, uz, 0.0)

    x_1d = x[0, :]
    z_1d = z[:, 0]

    vmag_liquid = np.sqrt(ux_liquid**2 + uz_liquid**2)
    if np.any(vmag_liquid > 1e-10):
        try:
            speed = vmag_liquid
            speed_max = speed.max()
            lw = np.where(speed > 1e-12, 2.0 * speed / (speed_max + 1e-15), 0)
            ax.streamplot(x_1d, z_1d, ux_liquid, uz_liquid,
                          color="black", density=2.0, linewidth=lw,
                          arrowsize=1.2, broken_streamlines=True)
        except Exception as e:
            print(f"  Streamplot failed: {e}")
            # Fallback to quiver
            skip = 2
            ms2 = (fl > 0.1)[::skip, ::skip] & (vmag[::skip, ::skip] > 1e-6)
            if np.any(ms2):
                ax.quiver(x[::skip, ::skip][ms2], z[::skip, ::skip][ms2],
                          ux[::skip, ::skip][ms2], uz[::skip, ::skip][ms2],
                          color="black", alpha=0.7)

    try:
        ax.contour(x, z, fl, levels=[0.5], colors=["lime"], linewidths=2)
    except Exception:
        pass
    try:
        ax.contour(x, z, fl, levels=[0.1], colors=["cyan"], linewidths=1.5, linestyles="--")
    except Exception:
        pass

    # Mark mushy zone
    try:
        ax.contour(x, z, fl, levels=[0.01, 0.1], colors=["red", "cyan"],
                   linewidths=[1, 1.5], linestyles=[":", "--"])
    except Exception:
        pass

    ax.set_xlabel("$x$ (μm)", fontsize=11)
    ax.set_ylabel("$z$ (μm)", fontsize=11)
    ax.set_title("D: Streamlines + Liquid Fraction\n"
                 "Red dotted = fl=0.01, Cyan dashed = fl=0.1", fontsize=11)
    ax.set_aspect("equal")
    ax.set_ylim(z_min, z_max)

    # Compute Peclet number
    v_mean = vmag[fl > 0.9].mean() * 50.0 if np.any(fl > 0.9) else 0.0  # m/s
    L_pool = np.sum(fl > 0.5) ** 0.5 * dx_um * 1e-6  # approximate pool size in m
    alpha_thermal = 6.5e-6  # m²/s (steel thermal diffusivity)
    Pe = v_mean * L_pool / alpha_thermal if alpha_thermal > 0 else 0

    t_us = step * 75e-3
    fig.suptitle(
        f"EDM Forcing Diagnostic — Step {step} (t = {t_us:.1f} μs)\n"
        f"Pe = v·L/α = {v_mean:.2f}×{L_pool*1e6:.0f}μm / {alpha_thermal*1e6:.1f}μm²/s = {Pe:.2f}  "
        f"({'conduction-dominated' if Pe < 1 else 'convection-dominated'})\n"
        f"Liquid (fl>0.5): {np.sum(fl>0.5)} cells, T_max: {T.max():.0f} K",
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    return fig


if __name__ == "__main__":
    steps = [333, 667, 800, 1000]
    if len(sys.argv) > 1:
        steps = [int(s) for s in sys.argv[1:]]

    for step in steps:
        print(f"\n=== Step {step} ===")
        fig = plot_diagnostic(step)
        if fig is not None:
            out = os.path.join(DATA_DIR, f"edm_diagnostic_step{step:04d}.png")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  Saved: {out}")
            plt.close(fig)
        else:
            print("  No data.")

    print("\n=== Done ===")
