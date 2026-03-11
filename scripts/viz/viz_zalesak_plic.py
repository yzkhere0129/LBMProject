#!/usr/bin/env python3
"""
Zalesak slotted-disk: PLIC-VOF vs TVD-MC comparison.

Reads CSV snapshots from both PLIC and TVD runs and renders a 3-panel figure:
  Left   — 0° initial (sharp)
  Center — 360° TVD-MC (algebraic)
  Right  — 360° PLIC (geometric)

The f=0.5 contour overlays show the interface sharpness difference.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path

DATA_DIR = Path("/home/yzk/LBMProject/scripts/viz")
OUT_PATH = DATA_DIR / "zalesak_plic_comparison.png"

NX, NY = 128, 128

def load_csv(path):
    with path.open() as fh:
        rows = [list(map(float, line.split(","))) for line in fh if line.strip()]
    return np.asarray(rows, dtype=np.float32)

_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "disk",
    [(0.0,  "#f5f5f5"),
     (0.45, "#a8d8ea"),
     (0.55, "#3a7fc1"),
     (1.0,  "#1a3a6c")],
)

def main():
    # Load initial (PLIC sharp init)
    data_init = load_csv(DATA_DIR / "zalesak_plic_deg000.csv")
    # Load TVD result (from previous run with tanh init)
    data_tvd  = load_csv(DATA_DIR / "zalesak_deg360.csv")
    # Load PLIC result
    data_plic = load_csv(DATA_DIR / "zalesak_plic_deg360.csv")

    mass_init = data_init.sum()
    mass_tvd  = load_csv(DATA_DIR / "zalesak_deg000.csv").sum()
    mass_tvd_360 = data_tvd.sum()
    mass_plic = data_plic.sum()

    tvd_loss  = (mass_tvd - mass_tvd_360) / mass_tvd * 100
    plic_loss = (mass_init - mass_plic) / mass_init * 100

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), facecolor="white",
                              gridspec_kw={"wspace": 0.15})

    fig.suptitle(
        "Zalesak Slotted-Disk  —  TVD-MC vs PLIC-VOF  (128×128, 800 steps)",
        fontsize=13, fontweight="bold", y=0.97, color="#111111",
    )

    panels = [
        (axes[0], data_init, "0° (initial, sharp)"),
        (axes[1], data_tvd,  "360° TVD-MC (algebraic)"),
        (axes[2], data_plic, "360° PLIC (geometric)"),
    ]

    xs = np.linspace(0, NX, NX)
    ys = np.linspace(0, NY, NY)

    for ax, data, title in panels:
        ax.set_facecolor("#f5f5f5")
        ax.imshow(data, origin="lower", cmap=_CMAP, vmin=0, vmax=1,
                  extent=[0, NX, 0, NY], interpolation="lanczos")
        ax.contour(xs, ys, data, levels=[0.5], colors=["white"],
                   linewidths=1.2, alpha=0.95)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6, color="#111111")
        ax.set_xlabel("x [cells]", fontsize=9, color="#444444")
        ax.set_ylabel("y [cells]", fontsize=9, color="#444444")
        ax.tick_params(labelsize=8, colors="#555555")
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
        ax.plot(64, 64, marker="+", color="#ff4444", ms=8, mew=1.5, zorder=5)

    # Overlay initial contour on both result panels
    for ax in [axes[1], axes[2]]:
        ax.contour(xs, ys, data_init, levels=[0.5],
                   colors=["#ff8800"], linewidths=1.0,
                   linestyles="--", alpha=0.80)

    # Mass loss annotations
    axes[1].text(0.97, 0.03, f"mass loss: {tvd_loss:.1f}%",
                 transform=axes[1].transAxes, ha="right", va="bottom",
                 fontsize=9, color="#222222",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           alpha=0.85, edgecolor="#cccccc"))
    axes[2].text(0.97, 0.03, f"mass loss: {plic_loss:.4f}%",
                 transform=axes[2].transAxes, ha="right", va="bottom",
                 fontsize=9, color="#222222",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           alpha=0.85, edgecolor="#cccccc"))

    # Legend
    leg = [
        mpatches.Patch(facecolor="none", edgecolor="#ff8800",
                       linestyle="--", linewidth=1.2, label="initial (0°)"),
        mpatches.Patch(facecolor="none", edgecolor="white",
                       linewidth=1.2, label="final (360°)"),
    ]
    axes[2].legend(handles=leg, loc="upper right", fontsize=8,
                   framealpha=0.85, edgecolor="#cccccc")

    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved: {OUT_PATH}")
    print(f"  TVD mass loss:  {tvd_loss:.2f}%")
    print(f"  PLIC mass loss: {plic_loss:.4f}%")

if __name__ == "__main__":
    main()
