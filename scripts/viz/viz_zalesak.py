#!/usr/bin/env python3
"""
Zalesak slotted-disk rotation visualization.

Reads VOF fill-level CSV snapshots produced by viz_zalesak.cu and renders a
clean 2-panel figure showing the disk before and after one full 360° rotation.

Layout:
  Left  — 0°   initial slotted disk
  Right — 360° final shape after one complete revolution

The f = 0.5 contour is overlaid in white on both panels so the interface
sharpness and shape preservation are immediately apparent.

Output: /home/yzk/LBMProject/scripts/viz/zalesak_disk.png
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration  (must match viz_zalesak.cu geometry)
# ---------------------------------------------------------------------------
DATA_DIR = Path("/home/yzk/LBMProject/scripts/viz")
OUT_PATH = DATA_DIR / "zalesak_disk.png"

# Grid
NX, NY = 128, 128        # cells (dimensionless lattice units)
CX, CY = 64.0, 64.0     # disk center
R       = 30.0            # disk radius [cells]
SLOT_W  = 10.0            # slot width  [cells]
SLOT_D  = 50.0            # slot depth  [cells]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> np.ndarray:
    with path.open() as fh:
        rows = [list(map(float, line.split(","))) for line in fh if line.strip()]
    # CSV row j corresponds to lattice y=j (y=0 at bottom).
    # imshow with origin='lower' renders row 0 at the bottom, so no flip needed.
    return np.asarray(rows, dtype=np.float32)   # shape (NY, NX)


# ---------------------------------------------------------------------------
# Colormap: clean white background, disk in a saturated ocean-teal
# ---------------------------------------------------------------------------
_CMAP_DISK = mcolors.LinearSegmentedColormap.from_list(
    "disk",
    [(0.0,  "#f5f5f5"),    # near-white background
     (0.45, "#a8d8ea"),    # light teal for interface region
     (0.55, "#3a7fc1"),    # medium blue
     (1.0,  "#1a3a6c")],   # deep navy for bulk disk
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_0   = load_csv(DATA_DIR / "zalesak_deg000.csv")
    data_360 = load_csv(DATA_DIR / "zalesak_deg360.csv")

    mass_0   = data_0.sum()
    mass_360 = data_360.sum()
    mass_loss_pct = (mass_0 - mass_360) / mass_0 * 100.0

    # -----------------------------------------------------------------------
    # Figure layout: 2 equal panels + small gap
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(
        1, 2,
        figsize=(10, 5.5),
        facecolor="white",
        gridspec_kw={"wspace": 0.12},
    )

    fig.suptitle(
        "Zalesak Slotted-Disk  —  Solid-Body Rotation  (128×128, TVD-MC VOF)",
        fontsize=13, fontweight="bold", y=0.97, color="#111111",
    )

    panels = [
        (axes[0], data_0,   "0°  (initial)"),
        (axes[1], data_360, "360°  (after one revolution)"),
    ]

    for ax, data, title in panels:
        ax.set_facecolor("#f5f5f5")
        im = ax.imshow(
            data,
            origin="lower",
            cmap=_CMAP_DISK,
            vmin=0.0,
            vmax=1.0,
            extent=[0, NX, 0, NY],
            interpolation="lanczos",
        )

        # f = 0.5 interface contour
        xs = np.linspace(0, NX, NX)
        ys = np.linspace(0, NY, NY)
        ax.contour(xs, ys, data, levels=[0.5], colors=["white"],
                   linewidths=1.2, alpha=0.95)

        ax.set_title(title, fontsize=11, fontweight="bold", pad=6, color="#111111")
        ax.set_xlabel("x  [cells]", fontsize=9, color="#444444")
        ax.set_ylabel("y  [cells]", fontsize=9, color="#444444")
        ax.tick_params(labelsize=8, colors="#555555")
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")

        # Mark disk center
        ax.plot(CX, CX, marker="+", color="#ff4444", ms=8, mew=1.5, zorder=5)

    # -----------------------------------------------------------------------
    # Overlay the f=0.5 contours of BOTH snapshots on the right panel to
    # show how much the shape has drifted after 360°.
    # -----------------------------------------------------------------------
    # Initial contour in orange, final in navy — already drawn as white above.
    # Draw a second overlay with colour coding.
    xs = np.linspace(0, NX, NX)
    ys = np.linspace(0, NY, NY)

    # Orange dashed initial on right panel
    axes[1].contour(xs, ys, data_0,   levels=[0.5],
                    colors=["#ff8800"], linewidths=1.0,
                    linestyles="--", alpha=0.80)

    # -----------------------------------------------------------------------
    # Mass-loss annotation
    # -----------------------------------------------------------------------
    axes[1].text(
        0.97, 0.03,
        f"mass loss: {mass_loss_pct:.1f}%",
        transform=axes[1].transAxes,
        ha="right", va="bottom",
        fontsize=9, color="#222222",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  alpha=0.85, edgecolor="#cccccc"),
    )

    # Legend for contour overlay
    leg_handles = [
        mpatches.Patch(facecolor="none", edgecolor="#ff8800",
                       linestyle="--", linewidth=1.2, label="initial (0°)"),
        mpatches.Patch(facecolor="none", edgecolor="white",
                       linewidth=1.2, label="final (360°)"),
    ]
    axes[1].legend(
        handles=leg_handles,
        loc="upper right",
        fontsize=8,
        framealpha=0.85,
        edgecolor="#cccccc",
    )

    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved: {OUT_PATH}")
    print(f"  Mass at 0°:   {mass_0:.1f}")
    print(f"  Mass at 360°: {mass_360:.1f}")
    print(f"  Mass loss:    {mass_loss_pct:.2f}%")


if __name__ == "__main__":
    main()
